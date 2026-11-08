"""Focused regressions for external-potential (biased) MD.

These tests use analytic toy potentials and verify the public contract:
external energy/force bookkeeping, time dependence, irregular batches,
selective dynamics, argument forwarding, and registered dump columns.
"""

from __future__ import annotations

import inspect
import os
import tempfile
import unittest

import numpy as np
import torch as th

from BUCToolkit.BatchMD import BiasedNVE, BiasedNVT, NVE, NVT
from BUCToolkit.BatchStructures import read_md_traj


K_MODEL = 0.7
K_EXT = 0.3
X0 = th.tensor([0.05, -0.10, 0.15], dtype=th.float32)


def _model_energy(X: th.Tensor) -> th.Tensor:
    return 0.5 * K_MODEL * th.sum(X ** 2, dim=(-2, -1))


def _model_grad(X: th.Tensor, energy: th.Tensor) -> th.Tensor:
    return K_MODEL * X


def _ext_energy(X: th.Tensor) -> th.Tensor:
    return 0.5 * K_EXT * th.sum((X - X0) ** 2, dim=(-2, -1))


def _ext_grad(X: th.Tensor, energy: th.Tensor) -> th.Tensor:
    return K_EXT * (X - X0)


def _ext_grad_no_y(X: th.Tensor) -> th.Tensor:
    return K_EXT * (X - X0)


def _free_energy(X: th.Tensor) -> th.Tensor:
    return th.zeros(X.shape[0], dtype=X.dtype, device=X.device)


def _free_grad(X: th.Tensor, energy: th.Tensor) -> th.Tensor:
    return th.zeros_like(X)


def _free_energy_n2(X: th.Tensor) -> th.Tensor:
    """Free-particle energy for an irregular two-structure batch."""
    return th.zeros(2, dtype=X.dtype, device=X.device)


def _ext_energy_irregular(X: th.Tensor) -> th.Tensor:
    """External energy split into the (1, 2) irregular batch layout."""
    return th.stack((
        0.5 * K_EXT * th.sum((X[0, :1] - X0) ** 2),
        0.5 * K_EXT * th.sum((X[0, 1:] - X0) ** 2),
    ))


def _make_nve(tmpdir: str, max_step: int = 1, output: str | None = None,
              is_output_ext_EF: bool = True):
    return BiasedNVE(
        0.5, max_step,
        T_init=300.,
        output_file=output,
        output_structures_per_step=1,
        device='cpu',
        verbose=0,
        is_compile=False,
        dump_quantities=('Energy', 'Ek', 'temperature', 'X', 'V', 'Force'),
        log_quantities=('Energy',),
        is_output_ext_EF=is_output_ext_EF,
    )


def _make_nvt(tmpdir: str, max_step: int = 1, output: str | None = None):
    return BiasedNVT(
        0.5, max_step,
        'Langevin', {'damping_coeff': 0.1},
        T_init=300.,
        output_file=output,
        output_structures_per_step=1,
        device='cpu',
        verbose=0,
        is_compile=False,
        dump_quantities=('Energy', 'Ek', 'temperature', 'X', 'V', 'Force'),
        log_quantities=('Energy',),
    )


class BiasedMDRegressionTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix='buc_biased_md_')

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _path(self, name: str) -> str:
        return os.path.join(self._tmpdir, name)

    def test_requires_external_potential(self):
        runner = _make_nve(self._tmpdir)
        X = th.zeros(1, 1, 3)
        with self.assertRaises(RuntimeError):
            runner.run(
                _free_energy, X, [['H']],
                V_init=th.zeros_like(X),
                grad_func=_free_grad,
                is_grad_func_contain_y=True,
                require_grad=False,
            )

    def test_set_external_potential_validation(self):
        runner = _make_nve(self._tmpdir)
        with self.assertRaises(TypeError):
            runner.set_external_potential(123)
        with self.assertRaises(TypeError):
            runner.set_external_potential(_ext_energy, potential_grad_func=123)
        # None gradient selects autograd; list args are accepted and normalized.
        runner.set_external_potential(_ext_energy, None, func_args=[K_EXT])
        self.assertIsInstance(runner.ext_pot_args, tuple)

    def test_static_external_energy_and_force(self):
        runner = _make_nve(self._tmpdir, max_step=2)
        runner.set_external_potential(_ext_energy, _ext_grad)
        X = th.tensor([[[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]]], dtype=th.float32)
        runner.run(
            _model_energy, X, [['H', 'H']],
            V_init=th.zeros_like(X),
            grad_func=_model_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        s = runner.s
        expected_e_ext = _ext_energy(s.X)
        expected_e_model = _model_energy(s.X)
        expected_f_ext = -_ext_grad(s.X, expected_e_ext)
        expected_f_model = -_model_grad(s.X, expected_e_model)
        th.testing.assert_close(s.ExternalEnergy, expected_e_ext, rtol=1e-5, atol=1e-6)
        th.testing.assert_close(s.ExternalForces, expected_f_ext, rtol=1e-5, atol=1e-6)
        th.testing.assert_close(s.Energy, expected_e_model + expected_e_ext, rtol=1e-5, atol=1e-6)
        th.testing.assert_close(s.Force, expected_f_model + expected_f_ext, rtol=1e-5, atol=1e-6)

    def test_autograd_external_force_with_analytic_model_grad(self):
        runner = _make_nve(self._tmpdir, max_step=1)
        runner.set_external_potential(_ext_energy, None)
        X = th.tensor([[[0.2, 0.1, -0.3]]], dtype=th.float32)
        runner.run(
            _model_energy, X, [['H']],
            V_init=th.zeros_like(X),
            grad_func=_model_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        s = runner.s
        expected = -_ext_grad(s.X, _ext_energy(s.X))
        th.testing.assert_close(s.ExternalForces, expected, rtol=1e-5, atol=1e-6)

    def test_external_grad_without_y(self):
        runner = _make_nve(self._tmpdir, max_step=1)
        runner.set_external_potential(
            _ext_energy, _ext_grad_no_y, is_extpot_grad_contain_y=False,
        )
        X = th.tensor([[[0.2, 0.1, -0.3]]], dtype=th.float32)
        runner.run(
            _model_energy, X, [['H']],
            V_init=th.zeros_like(X),
            grad_func=_model_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        s = runner.s
        expected = -_ext_grad_no_y(s.X)
        th.testing.assert_close(s.ExternalForces, expected, rtol=1e-5, atol=1e-6)

    def test_dynamic_time_and_run_reset(self):
        calls = []

        def ext_energy(X, t):
            calls.append(float(t))
            return th.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        def ext_grad(X, y, t):
            return th.zeros_like(X)

        runner = _make_nve(self._tmpdir, max_step=2)
        runner.set_external_potential(
            ext_energy, ext_grad, is_dynamic=True,
        )
        X = th.zeros(1, 1, 3)
        kwargs = dict(
            V_init=th.zeros_like(X),
            grad_func=_free_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        runner.run(_free_energy, X, [['H']], **kwargs)
        first_calls = list(calls)
        calls.clear()
        runner.run(_free_energy, X, [['H']], **kwargs)
        second_calls = list(calls)

        self.assertEqual(first_calls, [0.0, 0.0, 0.5])
        self.assertEqual(second_calls, [0.0, 0.0, 0.5])

    def test_irregular_batch_external_energy(self):
        runner = _make_nve(self._tmpdir, max_step=1)
        runner.set_external_potential(_ext_energy_irregular, _ext_grad)
        X = th.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
            dtype=th.float32,
        )
        runner.run(
            _free_energy_n2, X, [['H', 'H', 'H']],
            V_init=th.zeros_like(X),
            grad_func=_free_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
            batch_indices=(1, 2),
        )
        s = runner.s
        expected = th.stack((
            0.5 * K_EXT * th.sum((s.X[0, :1] - X0) ** 2),
            0.5 * K_EXT * th.sum((s.X[0, 1:] - X0) ** 2),
        ))
        th.testing.assert_close(s.ExternalEnergy, expected, rtol=1e-5, atol=1e-6)

    def test_fixed_atoms_keep_zero_total_force(self):
        runner = _make_nve(self._tmpdir, max_step=2)
        runner.set_external_potential(_ext_energy, _ext_grad)
        X = th.tensor([[[0.3, 0.1, -0.2], [0.2, -0.1, 0.4]]], dtype=th.float32)
        fixed = th.tensor([[[0., 0., 0.], [1., 1., 1.]]], dtype=th.float32)
        runner.run(
            _free_energy, X, [['H', 'H']],
            V_init=th.zeros_like(X),
            grad_func=_free_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
            fixed_atom_tensor=fixed,
        )
        s = runner.s
        th.testing.assert_close(s.X[0, 0], X[0, 0], rtol=0., atol=0.)
        th.testing.assert_close(
            s.Force[0, 0], th.zeros_like(s.Force[0, 0]), rtol=0., atol=1e-7,
        )
        # ExternalForces keeps the raw bias force, which is non-zero on fixed atoms.
        expected = -_ext_grad(s.X, _ext_energy(s.X))
        th.testing.assert_close(s.ExternalForces, expected, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(th.abs(s.ExternalForces[0, 0]).max()), 0.)

    def test_args_append_is_stable_across_runs(self):
        seen = []

        def ext_energy(X, *args, **kwargs):
            seen.append(args)
            return th.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        def ext_grad(X, y, *args, **kwargs):
            return th.zeros_like(X)

        def model_energy(X, main_arg):
            self.assertEqual(main_arg, 'main')
            return _free_energy(X)

        def model_grad(X, y, main_grad_arg):
            self.assertEqual(main_grad_arg, 'main_grad')
            return th.zeros_like(X)

        runner = _make_nve(self._tmpdir, max_step=1)
        runner.set_external_potential(
            ext_energy, ext_grad,
            func_args=('ext',),
            grad_func_args=('ext_grad',),
            is_args_append_main_func=True,
        )
        X = th.zeros(1, 1, 3)
        kwargs = dict(
            V_init=th.zeros_like(X),
            grad_func=model_grad,
            func_args=('main',),
            grad_func_args=('main_grad',),
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        runner.run(model_energy, X, [['H']], **kwargs)
        self.assertEqual(runner.ext_pot_args, ('main', 'ext'))
        self.assertEqual(runner.ext_pot_grad_args, ('main_grad', 'ext_grad'))
        self.assertEqual(seen[0], ('main', 'ext'))

        seen.clear()
        runner.run(model_energy, X, [['H']], **kwargs)
        self.assertEqual(runner.ext_pot_args, ('main', 'ext'))
        self.assertEqual(seen[0], ('main', 'ext'))

    def test_external_columns_first_frame_and_opt_out(self):
        out = self._path('traj_ext')
        runner = _make_nve(self._tmpdir, max_step=2, output=out, is_output_ext_EF=True)
        runner.set_external_potential(_ext_energy, _ext_grad)
        X = th.tensor([[[0.1, 0.2, 0.3]]], dtype=th.float32)
        X_init = X.clone()
        runner.run(
            _model_energy, X, [['H']],
            V_init=th.zeros_like(X),
            grad_func=_model_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        traj = read_md_traj(out, out_arrays=True)
        self.assertIn('ExternalEnergy', traj)
        self.assertIn('ExternalForces', traj)
        # The first dump happens before the first update: it must contain the
        # t=0 external values, not the registered zero prototype.
        th.testing.assert_close(
            th.as_tensor(traj['ExternalEnergy'][0], dtype=th.float32),
            _ext_energy(X_init)[0],
            rtol=1e-5, atol=1e-6,
        )
        th.testing.assert_close(
            th.as_tensor(traj['X'][0], dtype=th.float32), X_init[0],
            rtol=0., atol=0.,
        )

        out2 = self._path('traj_plain')
        runner2 = _make_nve(self._tmpdir, max_step=1, output=out2, is_output_ext_EF=False)
        runner2.set_external_potential(_ext_energy, _ext_grad)
        runner2.run(
            _model_energy, X, [['H']],
            V_init=th.zeros_like(X),
            grad_func=_model_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        traj2 = read_md_traj(out2, out_arrays=True)
        self.assertNotIn('ExternalEnergy', traj2)
        self.assertNotIn('ExternalForces', traj2)

    def test_leaves_and_signatures(self):
        self.assertIs(BiasedNVT._md_cls, NVT)
        self.assertIs(BiasedNVE._md_cls, NVE)
        nvt_sig = inspect.signature(BiasedNVT)
        self.assertIn('thermostat', nvt_sig.parameters)
        nve_sig = inspect.signature(BiasedNVE)
        self.assertIn('max_step', nve_sig.parameters)

        runner = _make_nvt(self._tmpdir, max_step=1)
        runner.set_external_potential(_ext_energy, _ext_grad)
        X = th.tensor([[[0.1, 0.2, 0.3]]], dtype=th.float32)
        runner.run(
            _model_energy, X, [['H']],
            V_init=th.zeros_like(X),
            grad_func=_model_grad,
            is_grad_func_contain_y=True,
            require_grad=False,
        )
        self.assertTrue(bool(th.isfinite(runner.s.Energy).all()))

    def test_bad_external_shapes_raise(self):
        runner = _make_nve(self._tmpdir, max_step=1)
        runner.set_external_potential(
            lambda X: th.zeros(1, dtype=X.dtype),
            lambda X, y: th.zeros_like(X),
        )
        X = th.zeros(1, 3, 3)
        with self.assertRaises(RuntimeError):
            runner.run(
                _free_energy_n2, X, [['H', 'H', 'H']],
                V_init=th.zeros_like(X),
                grad_func=_free_grad,
                is_grad_func_contain_y=True,
                require_grad=False,
                batch_indices=(1, 2),
            )

        runner2 = _make_nve(self._tmpdir, max_step=1)
        runner2.set_external_potential(
            lambda X: th.zeros(X.shape[0], dtype=X.dtype),
            lambda X, y: th.zeros(X.shape[-1], dtype=X.dtype),
        )
        X2 = th.zeros(1, 1, 3)
        with self.assertRaises(RuntimeError):
            runner2.run(
                _free_energy, X2, [['H']],
                V_init=th.zeros_like(X2),
                grad_func=_free_grad,
                is_grad_func_contain_y=True,
                require_grad=False,
            )


if __name__ == '__main__':
    unittest.main()
