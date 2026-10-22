"""Focused regressions for constraint Jacobian compilation."""

import math
import unittest
import warnings
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


def _device_list() -> list[str]:
    """Return the CPU device and CUDA when it is available."""
    return ['cpu', 'cuda:0'] if th.cuda.is_available() else ['cpu']


def _mixed_constraints(X: th.Tensor) -> th.Tensor:
    """Return three coordinate-dependent constraints for one structure."""
    return th.stack((
        th.sum(X[0] * X[1]),
        th.sum(X[0] ** 2) + th.sum(X[2] ** 2),
        th.sum(X[1] * X[2]),
    ))


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

    def test_constrained_md_uses_auto_threshold_by_default(self):
        """The runner default resolves to an estimated threshold."""
        def distance_constraint(X: th.Tensor) -> th.Tensor:
            return th.linalg.norm(X[1] - X[0]).reshape(1)

        X = th.tensor(
            (((0., 0., 0.), (1., 0., 0.)),),
            dtype=th.float32,
        )
        runner = ConstrNVT(
            time_step=0.1,
            max_step=1,
            thermostat='Langevin',
            thermostat_config={'damping_coeff': 0.01},
            constr_func=distance_constraint,
            constr_val=None,
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
        self.assertIsInstance(runner._constr.constr_thres, float)
        self.assertGreater(runner._constr.constr_thres, 0.)

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

    def test_fixman_distance_constraints_have_zero_geometric_force(self):
        """Distance constraints keep a constant Fixman determinant."""
        atom_masses = (1., 2., 3., 4.)
        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    ((0., 0., 0.), (1.2, 0., 0.), (0., 2.1, 0.), (0., 0., 1.9)),
                    dtype=th.float32,
                    device=device,
                ).unsqueeze(0)
                masses = (
                    th.ones_like(X)
                    * th.tensor(
                        atom_masses, dtype=X.dtype, device=device,
                    ).reshape(1, -1, 1)
                )
                targets = _distance_constraints(X[0]).reshape(1, -1)
                constr = BaseConstr(
                    _distance_constraints,
                    targets,
                    require_fixman=True,
                    device=device,
                    verbose=0,
                )
                constr.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=masses,
                    compile_jacobian=False,
                )
                constr.time_step = 0.5
                # The target equals the current distances, so the projection
                # converges immediately and only the Fixman terms are checked.
                _, G, w = constr._project2(X.clone(), X.clone())

                expected_w = 1. / math.sqrt(
                    (1. / atom_masses[0] + 1. / atom_masses[1])
                    * (1. / atom_masses[2] + 1. / atom_masses[3])
                )
                # The determinant is constant, so only round-off remains.
                self.assertTrue(th.allclose(G, th.zeros_like(G), atol=1e-6))
                self.assertTrue(th.allclose(
                    w, th.full_like(w, expected_w), rtol=1e-5,
                ))

    def test_fixman_sphere_constraint_matches_analytic_terms(self):
        """A sphere constraint reproduces the analytic Fixman terms."""
        rho = 4.
        for device in _device_list():
            with self.subTest(device=device):
                unit = th.tensor((1., 0.7, -0.3), dtype=th.float32, device=device)
                unit = unit / th.linalg.norm(unit)
                X = (unit * math.sqrt(rho)).reshape(1, 1, 3)
                constr = BaseConstr(
                    lambda sample: th.sum(sample[0] ** 2).reshape(1),
                    th.tensor(((rho,),), dtype=X.dtype, device=device),
                    require_fixman=True,
                    device=device,
                    verbose=0,
                )
                constr.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=th.ones_like(X),
                    compile_jacobian=False,
                )
                constr.time_step = 0.5
                _, G, w = constr._project2(X.clone(), X.clone())

                self.assertTrue(th.allclose(
                    G, th.full_like(G, 1. / (2. * rho)), rtol=1e-5,
                ))
                self.assertTrue(th.allclose(
                    w, th.full_like(w, 1. / (2. * math.sqrt(rho))), rtol=1e-5,
                ))

    def test_fixman_multi_constraint_matches_determinant_reference(self):
        """Several constraints match an independent determinant reference."""
        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    ((0.3, 0.1, 0.0), (1.1, 0.0, 0.2), (0., 2.2, 0.1)),
                    dtype=th.float32,
                    device=device,
                ).unsqueeze(0)
                masses = th.full_like(X, 2.)
                targets = _mixed_constraints(X[0]).reshape(1, -1)
                constr = BaseConstr(
                    _mixed_constraints,
                    targets,
                    require_fixman=True,
                    device=device,
                    verbose=0,
                )
                constr.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=masses,
                    compile_jacobian=False,
                )
                constr.time_step = 0.5
                _, G, w = constr._project2(X.clone(), X.clone())

                # Reference built from the mathematical definition
                # Z = J M^-1 J^T and G = Z^-1 J M^-1 grad(0.5 log det Z),
                # independently of the internal QR factors.
                X_ref = X.clone().requires_grad_(True)
                inverse_masses = 1. / masses
                jac = th.vmap(th.func.jacrev(_mixed_constraints))(X_ref)
                gram = th.einsum(
                    'bcnd,bnd,bend->bce', jac, inverse_masses, jac,
                )
                half_logdet = 0.5 * th.linalg.slogdet(gram)[1].sum()
                grad_half_logdet = th.autograd.grad(half_logdet, X_ref)[0]
                rhs = th.einsum(
                    'bcnd,bnd,bnd->bc', jac, inverse_masses, grad_half_logdet,
                )
                expected_G = th.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
                expected_w = th.rsqrt(th.linalg.det(gram))

                self.assertTrue(th.allclose(
                    G, expected_G, rtol=1e-5, atol=1e-7,
                ))
                self.assertTrue(th.allclose(w, expected_w, rtol=1e-5))

    def test_fixman_accepts_default_x_orig(self):
        """The default ``X_orig`` reference is detached from the copy it writes."""
        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    ((0.3, 0.1, 0.0), (1.1, 0.0, 0.2), (0., 2.2, 0.1)),
                    dtype=th.float32,
                    device=device,
                ).unsqueeze(0)
                targets = _mixed_constraints(X[0]).reshape(1, -1)
                constr = BaseConstr(
                    _mixed_constraints,
                    targets,
                    require_fixman=True,
                    device=device,
                    verbose=0,
                )
                constr.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=th.ones_like(X),
                    compile_jacobian=False,
                )
                constr.time_step = 0.5
                _, G_reference, w_reference = constr._project2(
                    X.clone(), X.clone(),
                )

                X_default = X.clone()
                _, G_default, w_default = constr._project2(X_default)
                # The projection rewrites X in place; the reference must not
                # leave a grad-requiring leaf behind.
                self.assertFalse(X_default.requires_grad)
                self.assertTrue(th.isfinite(G_default).all())
                self.assertTrue(th.allclose(
                    G_default, G_reference, rtol=1e-6, atol=1e-8,
                ))
                self.assertTrue(th.allclose(w_default, w_reference, rtol=1e-6))

    def test_fixman_large_determinant_remains_finite(self):
        """A determinant product that overflows keeps finite Fixman terms."""
        scale = 1e10

        def scaled_constraints(X: th.Tensor) -> th.Tensor:
            """Return four constraints with a large Jacobian scale."""
            return scale * th.stack((
                th.sum(X[0] * X[1]),
                th.sum(X[0] ** 2) + th.sum(X[2] ** 2),
                th.sum(X[1] * X[2]),
                th.sum(X[1] ** 2),
            ))

        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    ((0.3, 0.1, 0.0), (1.1, 0.0, 0.2), (0., 2.2, 0.1)),
                    dtype=th.float32,
                    device=device,
                ).unsqueeze(0)
                masses = th.ones_like(X)
                targets = scaled_constraints(X[0]).reshape(1, -1)
                constr = BaseConstr(
                    scaled_constraints,
                    targets,
                    constr_threshold=1e-5,
                    require_fixman=True,
                    device=device,
                    verbose=0,
                    # The intentionally unphysical scale is only used to exercise
                    # the Fixman determinant range; skip threshold estimation.
                    _detect_dynamic_thres='None',
                )
                constr.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=masses,
                    compile_jacobian=False,
                )
                constr.time_step = 0.5
                _, G, w = constr._project2(X.clone(), X.clone())

                # Keep the test inside the intended regime: a plain product of
                # the diagonal overflows single precision.
                diagonal = th.diagonal(constr.R, dim1=-2, dim2=-1).abs()
                self.assertFalse(
                    bool(th.isfinite(th.prod(diagonal, dim=-1)).all())
                )
                self.assertTrue(th.isfinite(G).all())
                self.assertTrue(th.isfinite(w).all())
                self.assertTrue(th.all(w >= 0.))

                # The geometric force stays accurate; w may saturate because
                # 1 / sqrt(det Z) is below the normal single-precision range.
                X_ref = X.clone().requires_grad_(True)
                inverse_masses = 1. / masses
                jac = th.vmap(th.func.jacrev(scaled_constraints))(X_ref)
                gram = th.einsum(
                    'bcnd,bnd,bend->bce', jac, inverse_masses, jac,
                )
                half_logdet = 0.5 * th.linalg.slogdet(gram)[1].sum()
                grad_half_logdet = th.autograd.grad(half_logdet, X_ref)[0]
                rhs = th.einsum(
                    'bcnd,bnd,bnd->bc', jac, inverse_masses, grad_half_logdet,
                )
                expected_G = th.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
                self.assertTrue(th.allclose(
                    G, expected_G.to(G.dtype), rtol=1e-4, atol=1e-16,
                ))

    def test_damped_retraction_matches_newton(self):
        """The damped solver reaches the same projection as plain Newton."""
        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    ((0.3, 0.1, 0.0), (1.1, 0.0, 0.2), (0., 2.2, 0.1)),
                    dtype=th.float32,
                    device=device,
                ).unsqueeze(0)
                masses = th.ones_like(X)
                # ``X_orig`` lies on the manifold, the integrated coordinates
                # are displaced, so both solvers run their iteration.
                targets = _mixed_constraints(X[0]).reshape(1, -1)
                displacement = th.zeros_like(X)
                displacement[..., 0] = 1e-2
                results = {}
                for solver in ('newton', 'damped'):
                    constr = BaseConstr(
                        _mixed_constraints,
                        targets,
                        require_fixman=True,
                        retraction_solver=solver,
                        device=device,
                        verbose=0,
                    )
                    self.assertEqual(constr.retraction_solver, solver)
                    constr.initialize(
                        func=None,
                        X=X,
                        Element_list=None,
                        masses=masses,
                        compile_jacobian=False,
                    )
                    constr.time_step = 0.5
                    X_projected = X + displacement
                    Fc, G, w = constr._project2(X_projected, X.clone())
                    results[solver] = (Fc, G, w, X_projected)

                newton, damped = results['newton'], results['damped']
                th.testing.assert_close(damped[0], newton[0], rtol=1e-5, atol=1e-6)
                th.testing.assert_close(damped[1], newton[1], rtol=1e-5, atol=1e-8)
                th.testing.assert_close(damped[2], newton[2], rtol=1e-5, atol=1e-8)
                th.testing.assert_close(damped[3], newton[3], rtol=1e-5, atol=1e-5)
                self.assertLess(
                    float(th.max(th.abs(
                        _mixed_constraints(damped[3][0]) - targets[0]
                    ))),
                    1e-4,
                )

                with self.assertRaises(ValueError):
                    BaseConstr(
                        _mixed_constraints,
                        targets,
                        retraction_solver='rk4',
                    )

    def test_solver_failure_reports_on_stderr_when_silent(self):
        """A stalled projection warns on stderr at verbose 0 and logs above it."""
        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    ((0.3, 0.1, 0.0), (1.1, 0.0, 0.2), (0., 2.2, 0.1)),
                    dtype=th.float32,
                    device=device,
                ).unsqueeze(0)
                masses = th.ones_like(X)
                # A far target stalls the retraction with an unconverged residual.
                targets = _mixed_constraints(X[0]).reshape(1, -1) + 5.

                silent = BaseConstr(
                    _mixed_constraints,
                    targets,
                    require_fixman=True,
                    device=device,
                    verbose=0,
                )
                silent.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=masses,
                    compile_jacobian=False,
                )
                silent.time_step = 0.5
                with self.assertWarns(RuntimeWarning):
                    silent._project2(X.clone(), X.clone())

                loud = BaseConstr(
                    _mixed_constraints,
                    targets,
                    require_fixman=True,
                    device=device,
                    verbose=1,
                )
                loud.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=masses,
                    compile_jacobian=False,
                )
                loud.time_step = 0.5
                with self.assertLogs('Main.Constraints', level='WARNING') as captured:
                    loud._project2(X.clone(), X.clone())
                self.assertTrue(any(
                    'not converged' in line for line in captured.output
                ))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    loud._project2(X.clone(), X.clone())
                self.assertEqual(caught, [])

    def test_dynamic_threshold_arguments_are_validated(self):
        """Invalid threshold-detection options are rejected early."""
        targets = th.zeros(1, 3)
        for value in ('Always', 0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BaseConstr(
                        _mixed_constraints,
                        targets,
                        _detect_dynamic_thres=value,
                    )
        with self.assertRaises(ValueError):
            BaseConstr(
                _mixed_constraints,
                targets,
                constr_threshold='auto',
                _detect_dynamic_thres='None',
            )
        with self.assertRaises(ValueError):
            BaseConstr(_mixed_constraints, targets, constr_threshold='fine')
        with self.assertRaises(ValueError):
            BaseConstr(_mixed_constraints, targets, constr_threshold=0.)

    def test_auto_threshold_resolves_at_initialization(self):
        """An ``'auto'`` threshold becomes three times the estimated floor."""
        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    ((0.3, 0.1, 0.0), (1.1, 0.0, 0.2), (0., 2.2, 0.1)),
                    dtype=th.float32,
                    device=device,
                ).unsqueeze(0)
                targets = _mixed_constraints(X[0]).reshape(1, -1)
                constr = BaseConstr(
                    _mixed_constraints,
                    targets,
                    constr_threshold='auto',
                    device=device,
                    verbose=0,
                )
                self.assertEqual(constr.constr_thres, 'auto')
                constr.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=th.ones_like(X),
                    compile_jacobian=False,
                )
                self.assertIsInstance(constr.constr_thres, float)
                self.assertGreater(constr.constr_thres, 0.)
                self.assertTrue(math.isfinite(constr.constr_thres))

                # The constructor default is the same automatic mode.
                default = BaseConstr(
                    _mixed_constraints,
                    targets,
                    device=device,
                    verbose=0,
                )
                default.initialize(
                    func=None,
                    X=X,
                    Element_list=None,
                    masses=th.ones_like(X),
                    compile_jacobian=False,
                )
                self.assertIsInstance(default.constr_thres, float)
                self.assertGreater(default.constr_thres, 0.)

    def test_threshold_warns_below_estimated_floor(self):
        """An explicit threshold below the probed evaluation noise warns."""
        big = 1e6
        value = 1.234

        def cancellation_constraint(sample: th.Tensor) -> th.Tensor:
            """Return a difference whose float32 evaluation is quantized."""
            return ((big + sample[0, 0]) - big).reshape(1)

        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    (((value, 0., 0.),),), dtype=th.float32, device=device,
                )
                target = ((big + X[0, 0, 0]) - big).reshape(1, 1)
                constr = BaseConstr(
                    cancellation_constraint,
                    target,
                    constr_threshold=1e-9,
                    device=device,
                    verbose=0,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    constr.initialize(
                        func=None,
                        X=X,
                        Element_list=None,
                        masses=th.ones_like(X),
                        compile_jacobian=False,
                    )
                self.assertTrue(any(
                    'below half of the estimated achievable floor'
                    in str(record.message)
                    for record in caught
                ))
                self.assertEqual(constr.constr_thres, 1e-9)

    def test_threshold_detection_failure_is_explicit(self):
        """A non-probeable constraint function never silently changes the mode."""
        def float32_only(sample: th.Tensor) -> th.Tensor:
            """Reject the higher-precision probe on purpose."""
            if sample.dtype != th.float32:
                raise RuntimeError('float32 only')
            return th.sum(sample[0] ** 2).reshape(1)

        X = th.tensor((((2., 0., 0.),),), dtype=th.float32)
        target = th.full((1, 1), 4.)
        masses = th.ones_like(X)

        auto = BaseConstr(
            float32_only,
            target,
            constr_threshold='auto',
            device='cpu',
            verbose=0,
        )
        with self.assertRaises(RuntimeError):
            auto.initialize(
                func=None,
                X=X,
                Element_list=None,
                masses=masses,
                compile_jacobian=False,
            )

        explicit = BaseConstr(
            float32_only,
            target,
            constr_threshold=1e-5,
            device='cpu',
            verbose=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            explicit.initialize(
                func=None,
                X=X,
                Element_list=None,
                masses=masses,
                compile_jacobian=False,
            )
        self.assertTrue(any(
            'Unable to estimate the constraint threshold floor'
            in str(record.message)
            for record in caught
        ))
        self.assertEqual(explicit.constr_thres, 1e-5)

    def test_dynamic_threshold_detects_on_cadence(self):
        """An integer cadence resolves the estimate during projection calls."""
        X = th.tensor(
            ((0.3, 0.1, 0.0), (1.1, 0.0, 0.2), (0., 2.2, 0.1)),
            dtype=th.float32,
        ).unsqueeze(0)
        targets = _mixed_constraints(X[0]).reshape(1, -1)
        constr = BaseConstr(
            _mixed_constraints,
            targets,
            constr_threshold='auto',
            _detect_dynamic_thres=2,
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
        self.assertEqual(constr.constr_thres, 'auto')
        constr._project2(X.clone(), X.clone())
        self.assertEqual(constr._proj2_call_number, 1)
        self.assertIsInstance(constr.constr_thres, float)
        constr._project2(X.clone(), X.clone())
        self.assertEqual(constr._proj2_call_number, 2)
        # An ``'auto'`` threshold keeps following the estimate on every
        # detection instead of freezing the first value.
        constr._apply_thres_estimate(1e-3, X)
        self.assertAlmostEqual(constr.constr_thres, 3e-3, places=12)

    def test_rank_deficiency_is_reported_at_initialization(self):
        """Nearly dependent constraints warn once during initialization."""
        def dependent_constraints(sample: th.Tensor) -> th.Tensor:
            """Return two proportional constraints."""
            radius2 = th.sum(sample[0] ** 2)
            return th.stack((radius2, radius2 * (1. + 1e-7)))

        for device in _device_list():
            with self.subTest(device=device):
                X = th.tensor(
                    (((2., 0., 0.),),), dtype=th.float32, device=device,
                )
                targets = th.vstack((
                    dependent_constraints(X[0]),
                )).reshape(1, -1)
                constr = BaseConstr(
                    dependent_constraints,
                    targets,
                    device=device,
                    verbose=0,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    constr.initialize(
                        func=None,
                        X=X,
                        Element_list=None,
                        masses=th.ones_like(X),
                        compile_jacobian=False,
                    )
                self.assertTrue(any(
                    'numerically rank-deficient' in str(record.message)
                    for record in caught
                ))


if __name__ == '__main__':
    unittest.main()
