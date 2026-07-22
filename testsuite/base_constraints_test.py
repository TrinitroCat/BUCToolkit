"""Focused regressions for constraint Jacobian compilation."""

import unittest

import torch as th

from BUCToolkit.Bases.BaseConstraints import BaseConstr
from BUCToolkit.BatchMD.ConstrNVE import ConstrNVE


def _distance_constraints(X: th.Tensor) -> th.Tensor:
    """Return two local distances for one structure."""
    pairs = th.tensor(((0, 1), (2, 3)), device=X.device)
    return th.linalg.norm(X[pairs[:, 1]] - X[pairs[:, 0]], dim=-1)


class BaseConstraintsRegressionTest(unittest.TestCase):
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
        self.assertAlmostEqual(
            float(distance_constraint(X[0])),
            1.002,
            places=5,
        )

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
            jac, y = runner._jacobian(X)
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
            jac, y = runner._jacobian(X_repeat)
        self.assertEqual(jac.shape, (3, 2, 4, 3))
        self.assertEqual(y.shape, (3, 2))


if __name__ == '__main__':
    unittest.main()
