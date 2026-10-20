"""Focused regressions for constraint Jacobian compilation."""

import unittest
from importlib.util import find_spec
from typing import Any

import torch as th

from BUCToolkit.Bases.BaseConstraints import BaseConstr
from BUCToolkit.BatchMD.ConstrNVE import ConstrNVE
from BUCToolkit.BatchMD.ConstrNVT import ConstrNVT
from BUCToolkit.BatchMD.NVE import NVE
from BUCToolkit.BatchMD.NVT import NVT

HAS_TRITON = find_spec('triton') is not None
FUSED_AVAILABLE = HAS_TRITON and th.cuda.is_available()


class _EagerConstrNVE(ConstrNVE):
    """Constrained NVE with the fused update forced back to eager."""

    def _initialize_no_constr(self, *args: Any, **kwargs: Any) -> None:
        """Resolve the eager update after the engine initialization."""
        super()._initialize_no_constr(*args, **kwargs)
        self._updateXV = self._updateXV_eager


class _EagerConstrNVT(ConstrNVT):
    """Constrained NVT with the fused scheme disabled."""

    def _build_fused_scheme(self, *args: Any, **kwargs: Any):
        """Skip the fused scheme so the eager thermostat is used."""
        return None


def _distance_constraints(X: th.Tensor) -> th.Tensor:
    """Return two local distances for one structure."""
    pairs = th.tensor(((0, 1), (2, 3)), device=X.device)
    return th.linalg.norm(X[pairs[:, 1]] - X[pairs[:, 0]], dim=-1)


class BaseConstraintsRegressionTest(unittest.TestCase):
    @staticmethod
    def _single_coordinate_constraint(X: th.Tensor) -> th.Tensor:
        """Return the first coordinate for each structure."""
        return X[0, 0].reshape(1)

    def _initialize_constraint(self, constr_val):
        """Initialize a small CPU constraint fixture without compilation."""
        X = th.tensor(
            (
                ((1., 0., 0.),),
                ((2., 0., 0.),),
            ),
            dtype=th.float32,
        )
        constr = BaseConstr(
            self._single_coordinate_constraint,
            constr_val,
            device='cpu',
            verbose=0,
        )
        constr.initialize(
            func=None,
            X=X,
            Element_list=None,
            masses=th.ones_like(X),
            compile_jacobian=False,
        )
        return constr, X

    def test_binary_time_target_receives_initial_constraint_and_derivative(self):
        """A binary target receives c0 and differentiates with respect to time."""
        calls = []

        def target(t: th.Tensor, c0: th.Tensor) -> th.Tensor:
            calls.append(c0)
            return c0 + 2. * t

        constr, X = self._initialize_constraint(target)
        c0 = th.tensor(((1.,), (2.,)), dtype=X.dtype)

        self.assertTrue(th.allclose(calls[0], c0))
        self.assertTrue(th.allclose(constr.constr_val_now, c0))
        self.assertTrue(th.allclose(constr.d_constr, th.full_like(c0, 2.)))

        constr._update_constr(th.tensor(0.5))
        self.assertTrue(th.allclose(constr.constr_val_now, c0 + 1.))
        self.assertTrue(th.allclose(constr.d_constr, th.full_like(c0, 2.)))

    def test_binary_time_target_reinitialization_refreshes_c0(self):
        """Each initialization binds the binary target to its new c0."""
        def target(t: th.Tensor, c0: th.Tensor) -> th.Tensor:
            return c0 + t

        constr, X = self._initialize_constraint(target)
        first_target = constr.constr_val_now.clone()

        X_second = X + 3.
        constr.initialize(
            func=None,
            X=X_second,
            Element_list=None,
            masses=th.ones_like(X_second),
            compile_jacobian=False,
        )

        self.assertTrue(th.allclose(first_target, th.tensor(((1.,), (2.,)))))
        self.assertTrue(th.allclose(
            constr.constr_val_now,
            th.tensor(((4.,), (5.,)), dtype=X.dtype),
        ))

    def test_unary_time_target_remains_compatible(self):
        """The original constr_val(t) callback form remains supported."""
        def target(t: th.Tensor) -> th.Tensor:
            return th.full((2,), 3., dtype=t.dtype) + t

        constr, _ = self._initialize_constraint(target)
        self.assertTrue(th.allclose(constr.constr_val_now, th.tensor(((3.,), (3.,)))))
        self.assertTrue(th.allclose(constr.d_constr, th.ones_like(constr.constr_val_now)))

    def test_time_target_type_error_from_body_is_not_signature_detection(self):
        """A TypeError raised by the callback body propagates unchanged."""
        def target(t: th.Tensor, c0: th.Tensor) -> th.Tensor:
            raise TypeError('target body failure')

        with self.assertRaisesRegex(TypeError, 'target body failure'):
            self._initialize_constraint(target)

    def test_binary_time_target_shape_mismatch_keeps_existing_error(self):
        """Invalid target shapes still fail during initialization."""
        def target(t: th.Tensor, c0: th.Tensor) -> tuple[th.Tensor, ...]:
            # A tuple target carries one entry per component; three components
            # are invalid for the two-batch, one-constraint fixture.
            return tuple(th.ones(1, dtype=c0.dtype) for _ in range(3))

        with self.assertRaisesRegex(RuntimeError, 'same shape'):
            self._initialize_constraint(target)

    def test_fixman_linear_constraint_has_zero_geometric_force(self):
        """A constant constraint Jacobian gives finite w and exactly zero G."""
        def linear_constraint(X: th.Tensor) -> th.Tensor:
            return X[0, 0].reshape(1)

        X = th.tensor(
            (((1., 0., 0.),), ((2., 0., 0.),)),
            dtype=th.float32,
        )
        constr = BaseConstr(
            linear_constraint,
            th.tensor(((1.,), (2.,)), dtype=X.dtype),
            constr_threshold=1e-6,
            require_fixman=True,
            device=X.device,
            verbose=0,
        )
        constr.initialize(
            func=None,
            X=X,
            Element_list=None,
            masses=th.ones_like(X),
            compile_jacobian=False,
        )
        _, G, w = constr._project2(X.clone(), X.clone())

        self.assertTrue(th.allclose(G, th.zeros_like(G)))
        self.assertTrue(th.allclose(w, th.ones_like(w)))

    def test_compiled_jacobian_pairs_each_batch_target(self):
        """Compilation preserves eager shapes and per-structure targets."""
        X = th.tensor(
            (
                ((0., 0., 0.), (1., 0., 0.), (0., 0., 0.), (0., 2., 0.)),
                ((0., 0., 0.), (3., 0., 0.), (0., 0., 0.), (0., 4., 0.)),
                ((0., 0., 0.), (5., 0., 0.), (0., 0., 0.), (0., 6., 0.)),
            ),
            dtype=th.float32,
        )
        constr_val = th.tensor(
            ((0.5, 1.5), (2.5, 3.5), (4.5, 5.5)),
            dtype=X.dtype,
        )
        constr = BaseConstr(
            _distance_constraints,
            constr_val,
            device=X.device,
            verbose=0,
        )

        expected_jac, expected_y = th.vmap(
            th.func.jacrev(constr._constr_func_wrapped, has_aux=True)
        )(X, constr_val)
        constr.initialize(
            func=None,
            X=X,
            Element_list=None,
            masses=th.ones_like(X),
        )

        self.assertIsNotNone(constr._compiled_jac)
        with th.no_grad():
            jac, y = constr._compiled_jac(X, constr_val)
        self.assertEqual(jac.shape, (3, 2, 4, 3))
        self.assertEqual(y.shape, (3, 2))
        self.assertTrue(th.allclose(jac, expected_jac, atol=1e-6))
        self.assertTrue(th.allclose(y, expected_y, atol=1e-6))

    def test_project2_refreshes_time_constraint_once(self):
        """Newton Jacobians reuse one time-dependent target per projection."""
        X_orig = th.tensor(
            (((0., 0., 0.), (1., 0., 0.), (0., 0., 0.), (0., 2., 0.)),),
            dtype=th.float32,
        )

        def constr_val(time_now: th.Tensor):
            return (1. + time_now.reshape(1), 2. + time_now.reshape(1))

        constr = BaseConstr(
            _distance_constraints,
            constr_val,
            constr_threshold=1e-5,
            device=X_orig.device,
            verbose=0,
        )
        constr.initialize(
            func=None,
            X=X_orig,
            Element_list=None,
            masses=th.ones_like(X_orig),
            compile_jacobian=False,
        )
        constr.time_step = 0.01

        update_count = 0
        update_times = []
        update_constr = constr._update_constr

        def counted_update(t) -> None:
            nonlocal update_count
            update_count += 1
            update_times.append(float(th.as_tensor(t)))
            update_constr(t)

        constr._update_constr = counted_update
        X = X_orig.clone()
        X[0, 1, 0] += 1e-3
        constr._project2(X, X_orig)
        self.assertEqual(update_count, 1)
        self.assertAlmostEqual(update_times[0], 0.01, places=6)
        self.assertAlmostEqual(float(constr.time_now), 0., places=6)
        self.assertTrue(th.allclose(
            constr.constr_val_now,
            th.tensor(((1.01, 2.01),), dtype=X.dtype),
        ))

        update_count = 0
        constr._jacobian(X)
        self.assertEqual(update_count, 0)
        constr._project1(th.zeros_like(X), X)
        self.assertEqual(update_count, 0)

    def test_constrained_md_stages_time_target_at_step_end(self):
        """MD evaluates one target at each committed end-of-step time."""
        time_step = 0.1

        def distance_constraint(X: th.Tensor) -> th.Tensor:
            return th.linalg.norm(X[1] - X[0]).reshape(1)

        def distance_target(t: th.Tensor) -> th.Tensor:
            return (1. + 0.01 * t).reshape(1)

        X = th.tensor(
            (((0., 0., 0.), (1., 0., 0.)),),
            dtype=th.float32,
        )
        runner = ConstrNVE(
            time_step=time_step,
            max_step=2,
            constr_func=distance_constraint,
            constr_val=distance_target,
            constr_threshold=1e-6,
            output_file=None,
            device='cpu',
            verbose=0,
        )
        update_times = []
        update_constr = runner._constr._update_constr

        def recorded_update(t) -> None:
            update_times.append(float(th.as_tensor(t)))
            update_constr(t)

        runner._constr._update_constr = recorded_update
        runner.run(
            lambda coo: th.zeros(coo.shape[0], device=coo.device),
            X,
            [['H', 'H']],
            V_init=th.zeros_like(X),
            grad_func=lambda coo, energy: th.zeros_like(coo),
            is_grad_func_contain_y=True,
        )

        self.assertEqual(len(update_times), 3)
        for actual, expected in zip(update_times, (0., 0.1, 0.2)):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(float(runner.time_now), 0.2, places=6)
        self.assertAlmostEqual(float(runner._constr.time_now), 0.2, places=6)
        self.assertAlmostEqual(float(runner._constr.time_step), time_step, places=6)
        self.assertAlmostEqual(
            float(distance_constraint(X[0])),
            1.002,
            places=5,
        )

    def test_plain_md_has_no_constraint_component(self):
        """Plain MD instances must not allocate a constraint solver."""
        nve = NVE(
            time_step=0.1, max_step=1, output_file=None,
            device='cpu', verbose=0,
        )
        nvt = NVT(
            time_step=0.1, max_step=1, thermostat='VR', output_file=None,
            device='cpu', verbose=0,
        )
        for runner in (nve, nvt):
            self.assertIsNone(runner._constr)
            self.assertFalse(runner._has_constr)

    def test_ignored_constraint_args_warn_only_above_verbose_zero(self):
        """Ignored constraint arguments are silent at verbose 0."""
        kwargs = dict(
            time_step=0.1,
            max_step=1,
            output_file=None,
            device='cpu',
            constr_val=th.tensor(1.0),
            require_fixman=True,
        )
        with self.assertNoLogs('Main.MD', level='WARNING'):
            runner = NVE(verbose=0, **kwargs)
        self.assertIsNone(runner._constr)
        self.assertFalse(runner._has_constr)

        with self.assertLogs('Main.MD', level='WARNING') as captured:
            NVE(verbose=1, **kwargs)
        self.assertTrue(any('will be ignored' in line for line in captured.output))

    def test_constrained_nvt_initializes_and_runs(self):
        """Constrained NVT resolves its scheme after the DOF correction."""
        def distance_constraint(X: th.Tensor) -> th.Tensor:
            return th.linalg.norm(X[1] - X[0]).reshape(1)

        X = th.tensor(
            (((0., 0., 0.), (1., 0., 0.)),),
            dtype=th.float32,
        )
        runner = ConstrNVT(
            time_step=0.1,
            max_step=2,
            thermostat='Nose-Hoover',
            constr_func=distance_constraint,
            constr_val=None,
            constr_threshold=1e-6,
            output_file=None,
            device='cpu',
            verbose=0,
        )
        runner.run(
            lambda coo: th.zeros(coo.shape[0], device=coo.device),
            X,
            [['H', 'H']],
            V_init=th.zeros_like(X),
            grad_func=lambda coo, energy: th.zeros_like(coo),
            is_grad_func_contain_y=True,
        )

        self.assertAlmostEqual(float(runner.free_degree[0]), 5., places=6)
        self.assertAlmostEqual(
            float(runner.EK_TARGET),
            2.5 * 8.617333262145e-5 * runner.T_init,
            places=6,
        )
        self.assertAlmostEqual(float(runner.time_now), 0.2, places=6)
        self.assertAlmostEqual(float(runner._constr.time_now), 0.2, places=6)
        self.assertAlmostEqual(float(distance_constraint(X[0])), 1.0, places=5)

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_constrained_nve_fused_matches_eager_on_cuda(self):
        """The constrained fused NVE path must mirror the eager path."""
        def distance_constraint(X: th.Tensor) -> th.Tensor:
            return th.linalg.norm(X[1] - X[0]).reshape(1)

        X = th.tensor(
            (((0., 0., 0.), (1., 0., 0.)),),
            dtype=th.float32,
            device='cuda',
        )
        X0 = X.clone()
        V = th.zeros_like(X)
        V[0, 0, 1] = 0.1
        V[0, 1, 1] = -0.1

        def potential(coo: th.Tensor) -> th.Tensor:
            return 0.5 * th.sum((coo - X0) ** 2, dim=(-2, -1))

        def grad_func(coo: th.Tensor, energy: th.Tensor) -> th.Tensor:
            return coo - X0

        def run(cls: type) -> th.Tensor:
            runner = cls(
                time_step=0.01,
                max_step=4,
                constr_func=distance_constraint,
                constr_val=None,
                constr_threshold=1e-6,
                output_file=None,
                device='cuda',
                verbose=0,
            )
            X_run = X.clone()
            runner.run(
                potential,
                X_run,
                [['H', 'H']],
                V_init=V.clone(),
                grad_func=grad_func,
                is_grad_func_contain_y=True,
            )
            return X_run

        th.testing.assert_close(run(ConstrNVE), run(_EagerConstrNVE), rtol=0., atol=1e-4)

    @unittest.skipUnless(FUSED_AVAILABLE, 'CUDA and Triton are required')
    def test_constrained_nvt_vr_fused_matches_eager_on_cuda(self):
        """The constrained fused NVT VR path must mirror the eager path."""
        def distance_constraint(X: th.Tensor) -> th.Tensor:
            return th.linalg.norm(X[1] - X[0]).reshape(1)

        X = th.tensor(
            (((0., 0., 0.), (1., 0., 0.)),),
            dtype=th.float32,
            device='cuda',
        )
        X0 = X.clone()
        V = th.zeros_like(X)
        V[0, 0, 1] = 0.1
        V[0, 1, 1] = -0.1

        def potential(coo: th.Tensor) -> th.Tensor:
            return 0.5 * th.sum((coo - X0) ** 2, dim=(-2, -1))

        def grad_func(coo: th.Tensor, energy: th.Tensor) -> th.Tensor:
            return coo - X0

        def run(cls: type) -> th.Tensor:
            runner = cls(
                time_step=0.01,
                max_step=4,
                thermostat='VR',
                constr_func=distance_constraint,
                constr_val=None,
                constr_threshold=1e-6,
                output_file=None,
                device='cuda',
                verbose=0,
            )
            X_run = X.clone()
            runner.run(
                potential,
                X_run,
                [['H', 'H']],
                V_init=V.clone(),
                grad_func=grad_func,
                is_grad_func_contain_y=True,
            )
            return X_run

        dynamic = run(ConstrNVT)
        th.testing.assert_close(dynamic, run(_EagerConstrNVT), rtol=0., atol=1e-4)
        self.assertAlmostEqual(float(distance_constraint(dynamic[0])), 1.0, places=4)

    @unittest.skipUnless(th.cuda.is_available(), 'CUDA is not available')
    def test_constrained_md_initializes_compiled_jacobian_on_cuda(self):
        """The constrained-MD initializer activates the compiled CUDA path."""
        device = th.device('cuda')
        X = th.tensor(
            (
                ((0., 0., 0.), (1., 0., 0.), (0., 0., 0.), (0., 2., 0.)),
                ((0., 0., 0.), (3., 0., 0.), (0., 0., 0.), (0., 4., 0.)),
            ),
            dtype=th.float32,
            device=device,
        )
        V = th.zeros_like(X)
        runner = ConstrNVE(
            time_step=0.01,
            max_step=1,
            constr_func=_distance_constraints,
            constr_val=None,
            constr_threshold=1e-5,
            output_file=None,
            device=device,
            verbose=2,
        )

        runner.run(
            lambda coo: th.sum(coo ** 2, dim=(-2, -1)),
            X,
            [['H'] * 4, ['H'] * 4],
            V_init=V,
            grad_func=lambda coo, energy: 2 * coo,
            is_grad_func_contain_y=True,
        )

        self.assertIsNotNone(runner._constr._compiled_jac)
        with th.no_grad():
            jac, y = runner._constr._jacobian(X)
        self.assertEqual(jac.shape, (2, 2, 4, 3))
        self.assertEqual(y.shape, (2, 2))

        # A repeated run may use a different static batch shape. The old graph
        # must be discarded before eager validation and rebuilt for this input.
        X_repeat = th.cat((X, X[:1] + 0.25), dim=0)
        runner.run(
            lambda coo: th.sum(coo ** 2, dim=(-2, -1)),
            X_repeat,
            [['H'] * 4] * 3,
            V_init=th.zeros_like(X_repeat),
            grad_func=lambda coo, energy: 2 * coo,
            is_grad_func_contain_y=True,
        )
        with th.no_grad():
            jac, y = runner._constr._jacobian(X_repeat)
        self.assertEqual(jac.shape, (3, 2, 4, 3))
        self.assertEqual(y.shape, (3, 2))


if __name__ == '__main__':
    unittest.main()
