"""Focused regressions for shared MD/MC/optimization trajectory handling."""

import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import torch as th

from BUCToolkit.BatchMC import MMC
from BUCToolkit.BatchMD.ConstrNVT import ConstrNVT
from BUCToolkit.BatchOptim.minimize.FIRE import FIRE
from BUCToolkit.BatchStructures import read_dump_arrays, read_mc_traj
from BUCToolkit.BatchStructures.StructuresIO import ArrayDumper


def _harmonic_energy(X: th.Tensor, X0: th.Tensor) -> th.Tensor:
    return 0.5 * th.sum((X - X0) ** 2, dim=(-2, -1))


def _harmonic_grad(X: th.Tensor, energy: th.Tensor, X0: th.Tensor) -> th.Tensor:
    return X - X0


class MotionIORegressionTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix='buc_motion_io_')

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _path(self, name: str) -> str:
        return os.path.join(self._tmpdir, name)

    def test_mc_configurable_named_quantities_cpu(self):
        """MC uses named state fields and preserves boolean dump dtypes."""
        output_file = self._path('mc.bin')
        runner = MMC(
            maxiter=4,
            temperature_init=300.,
            coordinate_update_param=0.01,
            output_file=output_file,
            output_structures_per_step=1,
            device='cpu',
            verbose=0,
            dump_quantities=(
                'Energy', 'X', 'delta_E', 'is_accept', 'temperature'
            ),
        )
        X = th.zeros((2, 3, 3), dtype=th.float32)
        runner.run(
            lambda coo: th.sum(coo ** 2, dim=(-2, -1)),
            X,
            [['H'] * 3, ['H'] * 3],
        )

        raw = read_dump_arrays(output_file)
        self.assertEqual(
            {'Energy', 'X', 'delta_E', 'is_accept', 'temperature'},
            set(raw) - {'cell_vec', 'atomic_numbers', 'fixed_mask'},
        )
        self.assertEqual(raw['is_accept'][0].dtype, np.dtype(np.bool_))
        self.assertEqual(len(raw['Energy']), 4)

        split = read_mc_traj(output_file, out_arrays=True)
        for name in ('Energy', 'X', 'delta_E', 'is_accept', 'temperature'):
            self.assertEqual(len(split[name]), 8)

    def test_irregular_reader_keeps_per_structure_matrices(self):
        """A 2-D per-structure field must not be split by atom pointers."""
        output_file = self._path('irregular.bin')
        dumper = ArrayDumper(output_file, mode='x')

        batch_indices = np.array([2, 3], dtype=np.int64)
        cells = np.zeros((2, 3, 3), dtype=np.float32)
        atomic_numbers = np.array([1, 1, 8, 1, 1], dtype=np.int64)
        fixed_mask = np.ones((1, 5, 3), dtype=np.float32)
        dumper.start_from_arrays(
            1, batch_indices, cells, atomic_numbers, fixed_mask
        )
        dumper.step(batch_indices, cells, atomic_numbers, fixed_mask)

        energies = np.zeros(2, dtype=np.float32)
        coordinates = np.zeros((1, 5, 3), dtype=np.float32)
        constraint_forces = np.arange(4, dtype=np.float32).reshape(2, 2)
        dumper.start_from_arrays(
            2,
            energies,
            coordinates,
            constraint_forces,
            names=('Energy', 'X', 'Fc'),
        )
        dumper.step(energies, coordinates, constraint_forces)
        dumper.step(energies + 1, coordinates + 1, constraint_forces + 10)
        dumper.close()

        split = read_mc_traj(output_file, out_arrays=True)
        self.assertEqual(
            [value.tolist() for value in split['Fc']],
            [[0., 1.], [2., 3.], [10., 11.], [12., 13.]],
        )
        self.assertEqual(
            [value.shape for value in split['X']],
            [(2, 3), (3, 3), (2, 3), (3, 3)],
        )

    def test_mc_irregular_batch_writes_ragged_atomic_numbers(self):
        """MC flattens irregular element rows and restores them on read."""
        output_file = self._path('mc_irregular.bin')
        runner = MMC(
            maxiter=2,
            temperature_init=300.,
            coordinate_update_param=0.01,
            output_file=output_file,
            output_structures_per_step=1,
            device='cpu',
            verbose=0,
        )
        X = th.zeros((1, 5, 3), dtype=th.float32)
        runner.run(
            lambda coo: th.stack((
                th.sum(coo[:, :2] ** 2),
                th.sum(coo[:, 2:] ** 2),
            )),
            X,
            [['H', 'H'], ['O', 'H', 'H']],
            batch_indices=(2, 3),
            move_to_center_freq=1,
        )

        raw = read_dump_arrays(output_file)
        self.assertEqual(raw['atomic_numbers'].tolist(), [1, 1, 8, 1, 1])
        split = read_mc_traj(output_file, out_arrays=True)
        self.assertEqual(
            [value.shape for value in split['X']],
            [(2, 3), (3, 3), (2, 3), (3, 3)],
        )

    def test_cpu_optimizer_final_dump_matches_returned_state(self):
        """The max-iteration frame contains the post-update CPU state."""
        output_file = self._path('opt.bin')
        X0 = th.zeros((1, 1, 3), dtype=th.float32)
        X = th.tensor([[[0.5, 0.3, 0.1]]], dtype=th.float32)
        runner = FIRE(
            E_threshold=0.,
            F_threshold=0.,
            maxiter=1,
            steplength=0.1,
            output_file=output_file,
            device='cpu',
            verbose=0,
        )
        runner.set_system_info(atomic_numbers=[[1]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            energies, coordinates, gradients = runner.run(
                _harmonic_energy,
                X,
                grad_func=_harmonic_grad,
                func_args=(X0,),
                grad_func_args=(X0,),
                is_grad_func_contain_y=True,
                output_grad=True,
            )

        raw = read_dump_arrays(output_file)
        self.assertTrue(np.allclose(raw['Energy'][-1], energies.numpy()))
        self.assertTrue(np.allclose(raw['X'][-1], coordinates.numpy()))
        self.assertTrue(np.allclose(raw['Force'][-1], (-gradients).numpy()))

    def test_constrained_nvt_updates_state_and_allows_repeated_runs(self):
        """Langevin/Nose-Hoover publish Fc and registration is idempotent."""
        constr_func = lambda coo: th.linalg.norm(
            coo[1] - coo[0]
        ).reshape(1)
        energy_func = lambda coo: 0.5 * th.sum(coo ** 2, dim=(-2, -1))
        grad_func = lambda coo, energy: coo
        X = th.tensor([[[0., 0., 0.], [1., 0., 0.]]], dtype=th.float32)
        V = th.tensor([[[0.1, 0., 0.], [-0.1, 0., 0.]]], dtype=th.float32)

        for scheme, config in (
                ('Langevin', {'damping_coeff': 0.01}),
                ('Nose-Hoover', {}),
        ):
            with self.subTest(scheme=scheme):
                output_file = self._path(f'{scheme}.bin')
                runner = ConstrNVT(
                    0.1,
                    3,
                    scheme,
                    config,
                    constr_func,
                    None,
                    1e-5,
                    False,
                    300.,
                    output_file,
                    1,
                    'cpu',
                    0,
                )
                runner.run(
                    energy_func,
                    X,
                    [['H', 'H']],
                    V_init=V,
                    grad_func=grad_func,
                    is_grad_func_contain_y=True,
                )
                raw = read_dump_arrays(output_file)
                self.assertIn('Fc', raw)
                self.assertTrue(any(np.any(value != 0.) for value in raw['Fc'][1:]))

        # A placeholder dumper permits a second run on the same instance and
        # isolates the persistent-registration behavior from file append mode.
        runner = ConstrNVT(
            0.01, 1, 'Langevin', {'damping_coeff': 0.01},
            constr_func, None, 1e-5, False, 300., None, 1, 'cpu', 0,
        )
        for _ in range(2):
            runner.run(
                energy_func,
                X,
                [['H', 'H']],
                V_init=th.zeros_like(V),
                grad_func=grad_func,
                is_grad_func_contain_y=True,
            )
        self.assertEqual(runner.get_dump_vars().count('Fc'), 1)
        self.assertEqual(runner.get_log_vars().count('Fc'), 1)


if __name__ == '__main__':
    unittest.main()
