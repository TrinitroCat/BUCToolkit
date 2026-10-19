"""Correctness and efficiency comparison of the fused and eager MD integrators.

The tests mirror the MD validation of ``main_test.test_MD``:

* the Triton kernels are checked against plain Torch references on regular and
  irregular batches,
* complete NVE / NVT trajectories of the fused classes are compared with the
  eager classes from identical initial conditions, and
* the per-step cost of both implementations is measured for a cheap harmonic
  potential, so the integrator cost is not hidden behind a heavy model.

Run with ``python fused_md_test.py`` (CUDA and Triton are required).
"""

#  Copyright (c) 2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: fused_md_test.py
#  Environment: Python 3.12

import os
import sys
import tempfile
import time
import unittest
from importlib.util import find_spec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch as th

from BUCToolkit.BatchMD import NVE as FusedNVE, NVT as FusedNVT


class EagerNVE(FusedNVE):
    """The merged engine with its eager velocity-Verlet update forced."""

    def initialize(self, *args, **kwargs) -> None:
        """Initialize the fused engine, then select the eager update."""
        super().initialize(*args, **kwargs)
        self._updateXV = self._updateXV_eager


class EagerNVT(FusedNVT):
    """The merged engine with its eager thermostat paths forced."""

    def _build_fused_scheme(self, *args, **kwargs):
        """Skip the fused scheme so ``update_scheme`` stays eager."""
        return None
from BUCToolkit.BatchStructures import read_md_traj
from BUCToolkit.utils._Element_info import MASS
from BUCToolkit.api._io import _Model_Wrapper_pyg
try:
    from .._toy_harmonic_potential import SimpleSpringPotential, build_cubic_lattice_batch
except ImportError:  # direct ``python fused_md_test.py`` execution
    from _toy_harmonic_potential import SimpleSpringPotential, build_cubic_lattice_batch

HAS_TRITON = find_spec('triton') is not None
HAS_CUDA = th.cuda.is_available()
FUSED_AVAILABLE = HAS_TRITON and HAS_CUDA

if HAS_TRITON:
    from BUCToolkit.BatchMD.fused_ops import (
        csvr_alpha,
        langevin_middle,
        nose_half_step_1,
        nose_reduce_update,
        nose_throttle_reduce_update,
        scale_by_batch,
        vr_scale,
        vv1,
        vv2,
    )

    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['seed'])
    def _randn_probe(seed, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        """Write the ``tl.randn`` stream of ``(seed, flat offset)``."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        offsets = tl.max_contiguous(offsets, BLOCK)
        mask = offsets < n
        noise = tl.randn(seed, offsets)
        tl.store(out_ptr + offsets, noise, mask=mask)

    def _randn(seed: int, n: int) -> th.Tensor:
        """Return the same normal stream the fused kernels generate."""
        out = th.empty(n, dtype=th.float32, device=DEVICE)
        _randn_probe[(triton.cdiv(n, 256),)](seed, out, n, 256)
        return out

DEVICE = 'cuda'
ELEMENTS = ['H']
K_SPRING = 10.0
EK_CONST = 103.642696562621738
KB = 8.617333262145e-5
TIME_STEP = 1.5


class HarmonicPotential:
    """A cheap site-wise harmonic potential for integrator comparisons."""

    def __init__(self, pos0: th.Tensor, batch_indices=None) -> None:
        """Store the reference positions.

        Args:
            pos0: equilibrium positions with the shape of ``X``.
            batch_indices: split sizes of an irregular batch, or ``None``.
        """
        self.pos0 = pos0
        self.batch_indices = batch_indices

    def _site_energy(self, X: th.Tensor) -> th.Tensor:
        """Return the per-atom harmonic energy of ``X``."""
        return 0.5 * K_SPRING * ((X - self.pos0) ** 2).sum(dim=-1)

    def Energy(self, X: th.Tensor) -> th.Tensor:
        """Return the energy of every batch.

        Args:
            X: coordinates with the shape of ``pos0``.

        Returns:
            Tensor of shape ``(n_batch,)``.
        """
        site = self._site_energy(X)
        if self.batch_indices is None:
            return site.sum(dim=-1)
        return th.cat([_.sum(dim=1) for _ in th.split(site, self.batch_indices, dim=1)])

    def Grad(self, X: th.Tensor) -> th.Tensor:
        """Return the gradient of :meth:`Energy` with respect to ``X``."""
        return K_SPRING * (X - self.pos0)


class _NoDump:
    """Mixin disabling all per-step host transfers for timing runs."""

    def get_dump_vars(self) -> dict:
        """Return no dump quantities."""
        return {}

    def get_log_vars(self) -> dict:
        """Return no log quantities."""
        return {}


class FusedMDTest(unittest.TestCase):
    """Compare the fused Triton MD integrators with the eager implementations."""

    @classmethod
    def setUpClass(cls) -> None:
        """Prepare the trajectory output directory."""
        if sys.platform.startswith('linux'):
            cls.out_pt = '/dev/shm/BUCToolkit_fused_test/'
            os.makedirs(cls.out_pt, exist_ok=True)
        else:
            cls.out_pt = tempfile.mkdtemp(prefix='BUCToolkit_fused_test_') + os.sep
        th.manual_seed(0)

    def _build_system(
            self, layout: str, n_atom: int = 6, n_batch: int = 3, seed: int = 11,
    ):
        """Build a harmonic MD system of the requested layout.

        Args:
            layout: ``'regular'`` or ``'irregular'``.
            n_atom: atoms per batch (regular) or a size scale (irregular).
            n_batch: number of batches.
            seed: random seed of the initial state.

        Returns:
            Tuple ``(model, X0, V0, element_list, batch_indices)``.
        """
        gen = th.Generator(device='cpu').manual_seed(seed)
        if layout == 'regular':
            pos0 = th.randn(n_batch, n_atom, 3, generator=gen) * 2.0
            element_list = [ELEMENTS * n_atom for _ in range(n_batch)]
            batch_indices = None
        else:
            counts = tuple(n_atom + 317 * _i for _i in range(n_batch))
            pos0 = th.randn(1, sum(counts), 3, generator=gen) * 2.0
            element_list = [ELEMENTS * sum(counts)]
            batch_indices = list(counts)
        X0 = pos0 + 0.02 * th.randn(pos0.shape, generator=gen)
        V0 = 0.002 * th.randn(pos0.shape, generator=gen)
        return (
            HarmonicPotential(pos0.to(DEVICE), batch_indices),
            X0.to(DEVICE), V0.to(DEVICE), element_list, batch_indices,
        )

    def _run(
            self, cls, thermostat, steps: int, layout: str, n_atom: int = 6,
            seed: int = 3, output: bool = True, tag: str = 'md',
            time_step: float = TIME_STEP, n_batch: int = 3,
    ):
        """Run one MD trajectory and return the reader and elapsed time.

        Args:
            cls: MD class to instantiate.
            thermostat: NVT thermostat name, or ``None`` for NVE.
            steps: number of MD steps.
            layout: ``'regular'`` or ``'irregular'``.
            n_atom: atoms per batch / size scale.
            seed: seed of the RNG before the run.
            output: whether to dump the trajectory and return a reader.
            tag: file-name tag of the trajectory.
            time_step: MD time step in fs.
            n_batch: number of batches.

        Returns:
            Tuple ``(trajectory_or_None, seconds_per_step)``.
        """
        model, X0, V0, element_list, batch_indices = self._build_system(
            layout, n_atom=n_atom, n_batch=n_batch, seed=seed + 7,
        )
        if not output:
            cls = type(f'Timed{cls.__name__}', (_NoDump, cls), {})
        output_file = None
        if output:
            output_file = f'{self.out_pt}{tag}_{layout}_{thermostat or "NVE"}'
            if os.path.exists(output_file):
                os.remove(output_file)
        output_per_step = 1 if output else steps + 1
        if thermostat is None:
            runner = cls(
                time_step, steps, 300., output_file, output_per_step,
                device=DEVICE, verbose=0,
            )
        else:
            config = {
                'Langevin': {'damping_coeff': 0.01},
                'CSVR': {'time_const': 100.},
                'VR': {},
                'Nose-Hoover': {},
            }[thermostat]
            runner = cls(
                time_step, steps, thermostat, config, 300.,
                output_file, output_per_step, device=DEVICE, verbose=0,
            )
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        th.cuda.synchronize()
        t_start = time.perf_counter()
        runner.run(
            model.Energy, X0, element_list, None, V0, model.Grad,
            (), None, (), None, False, False, batch_indices,
        )
        th.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        trajectory = read_md_traj(output_file) if output else None
        if output:
            os.remove(output_file)
        return trajectory, elapsed / steps

    @staticmethod
    def _max_diff(a, b) -> float:
        """Return the largest element-wise difference of two sample lists."""
        if len(a) != len(b):
            raise AssertionError(f'sample count mismatch: {len(a)} != {len(b)}')
        return max(
            float(np.max(np.abs(np.asarray(_x) - np.asarray(_y))))
            for _x, _y in zip(a, b)
        )

    def assertTrajectoriesClose(self, fused, eager, atol_x=1e-3, atol_v=5e-4, atol_e=5e-4, msg=''):
        """Assert that two trajectories agree sample by sample.

        Args:
            fused: trajectory reader of the fused run.
            eager: trajectory reader of the eager run.
            atol_x: tolerance on coordinates.
            atol_v: tolerance on velocities.
            atol_e: tolerance on energies.
            msg: message prefix.

        Returns:
            None.
        """
        dx = self._max_diff(fused.Coords, eager.Coords)
        dv = self._max_diff(fused.Labels, eager.Labels)
        de = self._max_diff(fused.Energies, eager.Energies)
        print(f'{msg}: max|dX|={dx:.2e} max|dV|={dv:.2e} max|dE|={de:.2e}')
        self.assertLess(dx, atol_x, f'{msg}: coordinates deviate ({dx:.2e})')
        self.assertLess(dv, atol_v, f'{msg}: velocities deviate ({dv:.2e})')
        self.assertLess(de, atol_e, f'{msg}: energies deviate ({de:.2e})')

    # ------------------------------------------------------------------
    # Kernel-level checks against plain Torch references
    # ------------------------------------------------------------------

    def _layout_tensors(self, layout: str):
        """Build the flat tensors shared by the kernel checks."""
        if layout == 'regular':
            n_batch, n_atom = 3, 5
            X = th.randn(n_batch, n_atom, 3, device=DEVICE)
            masses = th.rand(n_batch, n_atom, 1, device=DEVICE) * 20 + 1
            masses = masses.expand_as(X).contiguous()
            atom_ranges = [(0, n_atom)] * n_batch
            batch_scatter = None
            n_chunk = n_atom * 3
            batch_starts = None
        else:
            counts = (4, 7, 5)
            n_batch = len(counts)
            X = th.randn(1, sum(counts), 3, device=DEVICE)
            masses = th.rand(1, sum(counts), 1, device=DEVICE) * 20 + 1
            masses = masses.expand_as(X).contiguous()
            batch_scatter = th.repeat_interleave(
                th.arange(n_batch, device=DEVICE),
                th.tensor(counts, device=DEVICE),
            )
            n_chunk = 0
            offsets = np.cumsum((0,) + counts)
            batch_starts = th.tensor(offsets * 3, dtype=th.int32, device=DEVICE)
            batch_starts = batch_starts.contiguous()
            atom_ranges = list(zip(offsets[:-1], offsets[1:]))
        return X, masses, batch_scatter, n_chunk, batch_starts, atom_ranges

    def _per_batch_sum(self, values: th.Tensor, layout: str, batch_indices) -> th.Tensor:
        """Sum ``values`` over the atoms of every batch."""
        if layout == 'regular':
            return values.sum(dim=(-2, -1))
        return th.stack([values[0, _a:_b].sum() for _a, _b in batch_indices])

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_kernels(self):
        """Check every fused kernel against a Torch reference implementation."""
        th.manual_seed(0)
        for layout in ('regular', 'irregular'):
            X, masses, batch_scatter, n_chunk, batch_starts, atom_ranges = self._layout_tensors(layout)
            inv_M = (1.0 / masses).contiguous()
            is_irregular = layout == 'irregular'
            V = th.randn_like(X) * 0.1
            F = th.randn_like(X)
            mask = (th.rand_like(X) > 0.3).float()
            n_elem = X.numel()
            n_batch = len(atom_ranges)
            half_step = 0.5 * TIME_STEP * 9.64853329045427e-3
            raw_half = 0.5 * TIME_STEP
            flat_offsets = th.arange(n_elem, device=DEVICE)
            batch_id = (
                batch_scatter[flat_offsets // 3]
                if is_irregular else flat_offsets // n_chunk
            )

            X1, V1 = X.clone(), V.clone()
            vv1(X1, V1, F, inv_M, TIME_STEP)
            V_ref = V + half_step * F * inv_M
            th.testing.assert_close(V1, V_ref, rtol=1e-6, atol=1e-6)
            th.testing.assert_close(X1, X + TIME_STEP * V_ref, rtol=1e-6, atol=1e-6)

            V2, F2 = V1.clone(), F.clone()
            vv2(V2, F2, mask, inv_M, TIME_STEP)
            F_ref = F * mask
            th.testing.assert_close(F2, F_ref, rtol=0, atol=0)
            th.testing.assert_close(V2, V1 + half_step * F_ref * inv_M, rtol=1e-6, atol=1e-6)

            scale = th.rand(n_batch, device=DEVICE) + 0.5
            V3 = V1.clone()
            scale_by_batch(V3, scale, batch_scatter, n_chunk, is_irregular)
            th.testing.assert_close(
                V3, (V1.reshape(-1) * scale[batch_id]).view_as(V1),
                rtol=1e-6, atol=1e-6,
            )

            Ek = th.rand(n_batch, device=DEVICE) + 0.1
            ek_target = th.rand(n_batch, device=DEVICE) + 0.1
            V4 = V1.clone()
            vr_scale(V4, Ek, ek_target, batch_scatter, n_chunk, is_irregular)
            vr_factor = th.sqrt(ek_target / Ek)
            th.testing.assert_close(
                V4, (V1.reshape(-1) * vr_factor[batch_id]).view_as(V1),
                rtol=1e-6, atol=1e-6,
            )

            free_degree = th.randint(3, 30, (n_batch,), device=DEVICE).float()
            seed = 12345
            # The normal draw is generated inside the kernel; reproduce the
            # same (seed, offset) stream with the probe for the reference.
            randn = _randn(seed, n_batch)
            chi2 = th.rand(n_batch, device=DEVICE) * 10
            sqrt_c, one_sub_c, epsK = 0.99, 0.01, 1e-12
            alpha = th.empty(n_batch, device=DEVICE)
            csvr_alpha(
                V2, masses, alpha, ek_target, free_degree, chi2,
                batch_starts, n_chunk, sqrt_c, one_sub_c, epsK, seed, is_irregular,
            )
            ek_ref = 0.5 * self._per_batch_sum(masses * V2 ** 2, layout, atom_ranges) * EK_CONST
            K = th.clamp(ek_ref, min=epsK)
            f = one_sub_c * ek_target / (free_degree * K)
            sqrt_f = th.sqrt(th.clamp(f, min=0.0))
            alpha2 = (sqrt_c + sqrt_f * randn) ** 2 + f * chi2
            th.testing.assert_close(
                alpha, th.sqrt(th.clamp(alpha2, min=epsK)), rtol=1e-5, atol=1e-6,
            )

            sigma = th.rand_like(X) * 0.01
            seed = 54321
            # Same in-kernel (seed, flat offset) stream as langevin_middle.
            noise = _randn(seed, n_elem).view_as(X)
            X5, V5 = X.clone(), V.clone()
            alpha_l = 0.98
            langevin_middle(X5, V5, F, inv_M, sigma, seed, alpha_l, half_step, raw_half)
            V0 = V + half_step * F * inv_M
            V_ref = alpha_l * V0 + sigma * noise
            X_ref = X + raw_half * V0 + raw_half * V_ref
            th.testing.assert_close(V5, V_ref, rtol=1e-6, atol=1e-6)
            th.testing.assert_close(X5, X_ref, rtol=1e-6, atol=1e-6)

            p_iota = th.zeros(n_batch, device=DEVICE)
            smass = th.rand(n_batch, device=DEVICE) * 10 + 1
            target_kT = 0.043
            X6, V6 = X.clone(), V.clone()
            nose_half_step_1(
                X6, V6, F, inv_M, p_iota, batch_scatter, n_chunk,
                TIME_STEP, half_step, raw_half, is_irregular,
            )
            iota = p_iota[batch_id]
            V_ref = ((V + half_step * F * inv_M).reshape(-1) * th.exp(-iota * raw_half))
            th.testing.assert_close(V6, V_ref.view_as(V1), rtol=1e-6, atol=1e-6)
            th.testing.assert_close(X6, X + TIME_STEP * V_ref.view_as(V1), rtol=1e-6, atol=1e-6)

            def reduced_ek(vel):
                """Return ``2 * Ek`` of every batch."""
                return self._per_batch_sum(masses * vel ** 2, layout, atom_ranges) * EK_CONST

            p_old = p_iota.clone()
            nose_reduce_update(
                V6, masses, p_iota, free_degree, smass,
                batch_starts, n_chunk, raw_half, target_kT, is_irregular,
            )
            expected = p_old + raw_half * (reduced_ek(V6) - free_degree * target_kT) / smass
            th.testing.assert_close(p_iota, expected, rtol=1e-5, atol=1e-6)

            p_old = p_iota.clone()
            V7 = V6.clone()
            nose_throttle_reduce_update(
                V7, masses, p_iota, free_degree, smass,
                batch_starts, n_chunk, raw_half, target_kT, is_irregular,
            )
            V_ref = V6.reshape(-1) * th.exp(-p_old[batch_id] * raw_half)
            th.testing.assert_close(V7, V_ref.view_as(V1), rtol=1e-6, atol=1e-6)
            expected = p_old + raw_half * (reduced_ek(V_ref.view_as(V1)) - free_degree * target_kT) / smass
            th.testing.assert_close(p_iota, expected, rtol=1e-5, atol=1e-6)

    # ------------------------------------------------------------------
    # Whole-trajectory parity with the eager integrators
    # ------------------------------------------------------------------

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_nve_parity(self):
        """The fused NVE trajectory must match the eager one."""
        for layout in ('regular', 'irregular'):
            fused = self._run(FusedNVE, None, 40, layout, tag='parity_nve')[0]
            eager = self._run(EagerNVE, None, 40, layout, tag='parity_nve_eager')[0]
            self.assertTrajectoriesClose(fused, eager, msg=f'NVE {layout}')

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_nvt_parity(self):
        """The fused NVT thermostats must match the eager thermostats."""
        # Langevin and CSVR draw random numbers inside the kernels, so their
        # trajectories are not bitwise comparable with the eager torch RNG;
        # they are validated statistically in test_nvt_statistics.
        for thermostat in ('VR', 'Nose-Hoover'):
            for layout in ('regular', 'irregular'):
                fused = self._run(
                    FusedNVT, thermostat, 40, layout, tag=f'parity_{thermostat}',
                )[0]
                eager = self._run(
                    EagerNVT, thermostat, 40, layout, tag=f'parity_{thermostat}_eager',
                )[0]
                # The Nose-Hoover reduction order differs between the eager
                # index_reduce path and the fused per-batch loop; the tiny
                # energy difference is amplified by the chaotic dynamics.
                atol_e = 1e-2 if thermostat == 'Nose-Hoover' else 5e-4
                self.assertTrajectoriesClose(
                    fused, eager, msg=f'{thermostat} {layout}', atol_e=atol_e,
                )

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_pyg_wrapper_parity(self):
        """Use the graph-model wrapper of ``main_test.test_MD`` end to end."""
        atoms = [4, 3, 5]
        data = build_cubic_lattice_batch(atoms, 1.3, 0.05)
        element_list = [[]]
        for _i, _element in enumerate(['Fe', 'Al', 'Pd']):
            element_list[0].extend([_element] * atoms[_i] ** 3)
        batch_indices = [len(_.pos) for _ in data.to_data_list()]

        def run(cls, thermostat):
            _data = data.to(DEVICE).clone()
            model = _Model_Wrapper_pyg(
                SimpleSpringPotential(_data.pos0, K_SPRING),
            ).to(DEVICE)
            output_file = f'{self.out_pt}pyg_{cls.__module__}_{thermostat or "NVE"}'
            if os.path.exists(output_file):
                os.remove(output_file)
            if thermostat is None:
                runner = cls(TIME_STEP, 30, 300., output_file, 1, device=DEVICE, verbose=0)
            else:
                config = {
                    'Langevin': {'damping_coeff': 0.01},
                    'CSVR': {'time_const': 100.},
                    'VR': {},
                    'Nose-Hoover': {},
                }[thermostat]
                runner = cls(
                    TIME_STEP, 30, thermostat, config, 300.,
                    output_file, 1, device=DEVICE, verbose=0,
                )
            th.manual_seed(0)
            th.cuda.manual_seed_all(0)
            runner.run(
                model.Energy, _data.pos, element_list, None, None, model.Grad,
                (_data,), None, (_data,), None, False, False, batch_indices,
            )
            th.cuda.synchronize()
            trajectory = read_md_traj(output_file)
            os.remove(output_file)
            return trajectory

        # Stochastic thermostats are validated statistically in
        # test_nvt_statistics; their in-kernel RNG is not torch's RNG.
        cases = [(None, EagerNVE, FusedNVE)]
        cases += [(_t, EagerNVT, FusedNVT) for _t in ('VR', 'Nose-Hoover')]
        for thermostat, eager_cls, fused_cls in cases:
            # The Nose-Hoover reduction order differs between the fused and
            # eager implementations; its chaotic dynamics amplifies the tiny
            # energy difference.
            atol_e = 1e-2 if thermostat == 'Nose-Hoover' else 5e-4
            self.assertTrajectoriesClose(
                run(fused_cls, thermostat), run(eager_cls, thermostat),
                msg=f'PyG wrapper {thermostat or "NVE"}', atol_e=atol_e,
            )

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_nvt_statistics(self):
        """The fused thermostats must reproduce the eager ensemble statistics."""
        # `_run` builds the runners with T_init = 300 K.
        temperature = 300.
        n_atom = 16
        n_batch = 64
        steps = 2000

        def ensemble_kinetic_energy(trajectory):
            """Return the equilibrated time- and batch-averaged kinetic energy."""
            values = np.asarray([
                0.5 * np.sum(np.asarray(MASS['H']) * np.asarray(_v) ** 2 * EK_CONST)
                for _v in trajectory.Labels
            ])
            return float(values[values.size // 2:].mean())

        for thermostat in ('VR', 'Langevin', 'CSVR', 'Nose-Hoover'):
            fused, _ = self._run(
                FusedNVT, thermostat, steps, 'regular', n_atom=n_atom,
                tag=f'stat_{thermostat}', time_step=1.0, n_batch=n_batch,
            )
            eager, _ = self._run(
                EagerNVT, thermostat, steps, 'regular', n_atom=n_atom,
                tag=f'stat_{thermostat}_eager', time_step=1.0, n_batch=n_batch,
            )
            ek_fused = ensemble_kinetic_energy(fused)
            ek_eager = ensemble_kinetic_energy(eager)
            expected = 0.5 * 3 * n_atom * KB * temperature
            print(
                f'{thermostat:12s} Ek fused={ek_fused:.4f} eager={ek_eager:.4f} '
                f'target={expected:.4f} eV'
            )
            self.assertLess(
                abs(ek_fused - ek_eager) / ek_eager, 0.1,
                f'{thermostat}: fused and eager kinetic energies differ',
            )
            self.assertLess(
                abs(ek_fused - expected) / expected, 0.15,
                f'{thermostat}: kinetic energy {ek_fused:.4f} eV is far from '
                f'{expected:.4f} eV',
            )

    # ------------------------------------------------------------------
    # Efficiency report
    # ------------------------------------------------------------------

    def _model_cost(self, layout: str, n_atom: int, steps: int = 200) -> float:
        """Return the per-step cost of the bare potential calls in seconds."""
        model, X0, _, _, _ = self._build_system(layout, n_atom=n_atom)
        with th.no_grad():
            for _ in range(20):
                model.Grad(X0)
                model.Energy(X0)
            th.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(steps):
                model.Energy(X0)
                model.Grad(X0)
            th.cuda.synchronize()
        return (time.perf_counter() - t_start) / steps

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_performance_report(self):
        """Measure the per-step cost of the fused and eager integrators."""
        steps = 800
        print(
            '\n'
            f'{"layout":>10s} {"n_atom":>7s} {"case":>13s} '
            f'{"eager(us)":>10s} {"fused(us)":>10s} {"total":>7s} '
            f'{"upd-eager":>10s} {"upd-fused":>10s} {"update":>7s}'
        )
        for layout in ('regular', 'irregular'):
            for n_atom in (128, 512):
                model_cost = self._model_cost(layout, n_atom)
                cases = [('NVE', None, EagerNVE, FusedNVE)]
                cases += [
                    (_t, _t, EagerNVT, FusedNVT)
                    for _t in ('VR', 'Langevin', 'CSVR', 'Nose-Hoover')
                ]
                for name, thermostat, eager_cls, fused_cls in cases:
                    # Warm up kernels and CUDA-graph captures before timing.
                    self._run(eager_cls, thermostat, 20, layout, n_atom=n_atom, output=False)
                    self._run(fused_cls, thermostat, 20, layout, n_atom=n_atom, output=False)
                    _, t_eager = self._run(
                        eager_cls, thermostat, steps, layout, n_atom=n_atom, output=False,
                    )
                    _, t_fused = self._run(
                        fused_cls, thermostat, steps, layout, n_atom=n_atom, output=False,
                    )
                    # The first timing of a variant can include garbage-collection
                    # pauses; keep the faster of two identical measurements.
                    t_eager = min(t_eager, self._run(
                        eager_cls, thermostat, steps, layout, n_atom=n_atom, output=False,
                    )[1])
                    t_fused = min(t_fused, self._run(
                        fused_cls, thermostat, steps, layout, n_atom=n_atom, output=False,
                    )[1])
                    upd_eager = max(t_eager - model_cost, 1e-9)
                    upd_fused = max(t_fused - model_cost, 1e-9)
                    print(
                        f'{layout:>10s} {n_atom:>7d} {name:>13s} '
                        f'{t_eager * 1e6:>10.1f} {t_fused * 1e6:>10.1f} '
                        f'{t_eager / t_fused:>6.2f}x '
                        f'{upd_eager * 1e6:>10.1f} {upd_fused * 1e6:>10.1f} '
                        f'{upd_eager / upd_fused:>6.2f}x'
                    )


if __name__ == '__main__':
    if not FUSED_AVAILABLE:
        print('CUDA and Triton are required for the fused MD comparison tests.')
        raise SystemExit(0)
    unittest.main(verbosity=2)
