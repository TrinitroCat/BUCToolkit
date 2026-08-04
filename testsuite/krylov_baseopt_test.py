import pytest
import torch as th

from BUCToolkit.BatchOptim.TS.Krylov import (
    KrylovDynamics as LegacyKrylovDynamics,
    KrylovNewton as LegacyKrylovNewton,
)
from BUCToolkit.BatchOptim.TS.Krylov import (
    KrylovDynamics,
    KrylovNewton,
)
from BUCToolkit.BatchOptim.TS._eigen_solver import FindEigen
from BUCToolkit.BatchOptim._BaseOpt import FLOAT_TYPE


def _energy(X, counts, scales):
    energies = []
    atom_start = 0
    for count, scale in zip(counts.tolist(), scales):
        coordinates = X[:, atom_start:atom_start + count].reshape(1, -1)
        coefficients = th.ones_like(coordinates)
        coefficients[:, 0] = -1.
        coordinates2 = coordinates.square()
        energies.append(scale * (
            coefficients * coordinates2 + 0.1 * coordinates2.square()
        ).sum())
        atom_start += count
    return th.stack(energies)


def _gradient(X, counts, scales):
    gradients = []
    atom_start = 0
    for count, scale in zip(counts.tolist(), scales):
        coordinates = X[:, atom_start:atom_start + count].reshape(1, -1)
        coefficients = th.ones_like(coordinates)
        coefficients[:, 0] = -1.
        gradients.append((
            scale * (
                2. * coefficients * coordinates
                + 0.4 * coordinates.pow(3)
            )
        ).reshape(1, count, 3))
        atom_start += count
    return th.cat(gradients, dim=1)


class _BatchUpdater:
    def __init__(self):
        self.masks = []

    def __call__(
            self,
            mask,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
    ):
        self.masks.append(mask.detach().cpu().clone())
        counts, scales = func_args
        selected = mask.to(counts.device)
        args = (counts[selected], scales[selected])
        return args, func_kwargs, args, grad_func_kwargs


class _CallCounter:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.func(*args, **kwargs)


def _make_optimizer(optimizer_type, device, maxiter=2, scheme='trust_region'):
    return optimizer_type(
        E_threshold=1.e-9,
        Torque_thres=1.e-4,
        Eigen_thres=1.e-4,
        F_threshold=1.e-5,
        maxiter_trans=maxiter,
        maxiter_eig=5,
        steplength=0.05,
        steplength_sheme=scheme,
        dx=0.02,
        device=device,
        verbose=0,
    )


def _run(
        optimizer,
        X,
        X_diff,
        counts,
        scales,
        fixed_atom_tensor=None,
        energy_func=_energy,
        gradient_func=_gradient,
):
    return optimizer.run(
        energy_func,
        X,
        X_diff=X_diff,
        grad_func=gradient_func,
        func_args=(counts, scales),
        grad_func_args=(counts, scales),
        is_grad_func_contain_y=False,
        output_grad=True,
        fixed_atom_tensor=fixed_atom_tensor,
        batch_indices=counts.tolist(),
    )


@pytest.mark.parametrize(('legacy_type', 'current_type', 'scheme'), [
    (LegacyKrylovNewton, KrylovNewton, 'trust_region'),
    (LegacyKrylovNewton, KrylovNewton, 'line_search'),
    (LegacyKrylovNewton, KrylovNewton, 'line_newton'),
])
def test_krylov_baseopt_matches_legacy(legacy_type, current_type, scheme):
    th.manual_seed(7)
    counts = th.tensor([3, 4], dtype=th.long)
    scales = th.tensor([0.7, 1.3])
    X = th.randn(1, int(counts.sum()), 3) * 0.15
    X_diff = th.randn_like(X)

    legacy = _make_optimizer(legacy_type, 'cpu', scheme=scheme)
    current = _make_optimizer(current_type, 'cpu', scheme=scheme)
    th.manual_seed(23)
    legacy_result = _run(
        legacy, X.clone(), X_diff.clone(), counts, scales,
    )
    th.manual_seed(23)
    current_result = _run(
        current, X.clone(), X_diff.clone(), counts, scales,
    )

    for legacy_value, current_value in zip(legacy_result, current_result):
        th.testing.assert_close(
            current_value, legacy_value, rtol=0., atol=0.,
        )


@pytest.mark.parametrize('optimizer_type', [
    KrylovNewton,
    KrylovDynamics,
])
def test_krylov_baseopt_updates_dynamic_batch(
        optimizer_type,
):
    th.manual_seed(13)
    counts = th.tensor([3, 4], dtype=th.long)
    scales = th.tensor([0.7, 1.3])
    X = th.randn(1, int(counts.sum()), 3) * 0.15
    X[:, :counts[0]].mul_(1.e-4)
    X_diff = th.randn_like(X)

    current_updater = _BatchUpdater()
    current = _make_optimizer(optimizer_type, 'cpu', maxiter=3)
    current.E_threshold = 1.e-6
    current.F_threshold = 1.e-3
    current.set_batch_updater(current_updater)

    th.manual_seed(31)
    current_result = _run(
        current, X.clone(), X_diff.clone(), counts, scales,
    )

    assert all(value.isfinite().all() for value in current_result)
    assert any(
        mask.tolist() == [False, True]
        for mask in current_updater.masks
    )


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param(
        'cuda:0',
        marks=pytest.mark.skipif(
            not th.cuda.is_available(),
            reason='CUDA unavailable',
        ),
    ),
])
@pytest.mark.parametrize('optimizer_type', [
    KrylovNewton,
    KrylovDynamics,
])
def test_krylov_baseopt_preserves_fixed_atoms(
        device,
        optimizer_type,
):
    th.manual_seed(29)
    counts = th.tensor([4], dtype=th.long, device=device)
    scales = th.tensor([1.], device=device)
    X = th.randn(1, 4, 3, device=device) * 0.2
    X_diff = th.randn_like(X)
    fixed_mask = th.ones_like(X)
    fixed_mask[:, -1].zero_()

    current = _make_optimizer(optimizer_type, device)
    th.manual_seed(37)
    current_result = _run(
        current,
        X.clone(),
        X_diff.clone(),
        counts,
        scales,
        fixed_mask,
    )

    assert all(value.isfinite().all() for value in current_result)
    th.testing.assert_close(
        current_result[1][:, -1],
        X[:, -1],
        rtol=0.,
        atol=0.,
    )


@pytest.mark.parametrize(('legacy_type', 'current_type'), [
    (LegacyKrylovNewton, KrylovNewton),
    (LegacyKrylovDynamics, KrylovDynamics),
])
def test_krylov_initial_solve_avoids_duplicate_evaluation(
        legacy_type,
        current_type,
):
    th.manual_seed(41)
    counts = th.tensor([4], dtype=th.long)
    scales = th.tensor([1.])
    X = th.randn(1, 4, 3) * 0.2
    X_diff = th.randn_like(X)

    legacy_energy = _CallCounter(_energy)
    legacy_gradient = _CallCounter(_gradient)
    current_energy = _CallCounter(_energy)
    current_gradient = _CallCounter(_gradient)
    legacy = _make_optimizer(legacy_type, 'cpu', maxiter=1)
    current = _make_optimizer(current_type, 'cpu', maxiter=1)

    _run(
        legacy,
        X.clone(),
        X_diff.clone(),
        counts,
        scales,
        energy_func=legacy_energy,
        gradient_func=legacy_gradient,
    )
    _run(
        current,
        X.clone(),
        X_diff.clone(),
        counts,
        scales,
        energy_func=current_energy,
        gradient_func=current_gradient,
    )

    assert current_energy.calls == legacy_energy.calls
    assert current_gradient.calls == legacy_gradient.calls


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param(
        'cuda:0',
        marks=pytest.mark.skipif(
            not th.cuda.is_available(),
            reason='CUDA unavailable',
        ),
    ),
])
@pytest.mark.parametrize('optimizer_type', [
    KrylovNewton,
    KrylovDynamics,
])
def test_krylov_baseopt_converges_on_test_ts_potential(
        device,
        optimizer_type,
):
    th.manual_seed(42)
    counts = th.tensor([27, 8], dtype=th.long, device=device)
    scales = th.ones(2, device=device)
    X = th.randn(1, int(counts.sum()), 3, device=device) * 0.5
    X_diff = th.randn_like(X)

    if optimizer_type is KrylovNewton:
        optimizer = optimizer_type(
            5.e-5, 0.01, 0.01, 0.05,
            500, 10, 0.05,
            device=device,
            verbose=0,
        )
    else:
        optimizer = optimizer_type(
            5.e-5, 0.01, 0.01, 0.05,
            500, 30, 0.1,
            device=device,
            verbose=0,
        )
    optimizer.set_batch_updater(_BatchUpdater())
    energies, coordinates = _run(
        optimizer,
        X,
        X_diff,
        counts,
        scales,
    )[:2]

    assert float(energies.abs().max()) < 0.05
    assert float(coordinates.abs().max()) < 0.05


def test_diag_trust_region_masks_only_final_inactive_result():
    optimizer = _make_optimizer(KrylovNewton, 'cpu')
    projected_gradient = th.tensor([[0.1, 2.0]])
    spectra = th.ones((2, 1))
    complement_square = th.zeros((2, 1))
    delta2 = th.ones((2, 1))

    batched_mu = optimizer._diag_trust_region(
        projected_gradient,
        spectra,
        complement_square,
        delta2,
    )
    active_mu = optimizer._diag_trust_region(
        projected_gradient[:, 1:],
        spectra[1:],
        complement_square[1:],
        delta2[1:],
    )

    assert batched_mu[0].item() == 0.
    assert batched_mu.dtype == delta2.dtype
    th.testing.assert_close(batched_mu[1:], active_mu)


def test_diag_trust_region_uses_double_internal_precision():
    optimizer = _make_optimizer(KrylovNewton, 'cpu')
    projected_gradient = th.tensor([
        [0.71234567, 1.2345679],
        [0.45678902, 0.9876543],
    ])
    spectra = th.tensor([
        [-0.12345678, 1.8765432],
        [-0.2345679, 0.7654321],
    ])
    complement_square = th.tensor([[0.31415927], [0.27182818]])
    delta2 = th.tensor([[0.12345679], [0.2345679]])

    mu = optimizer._diag_trust_region(
        projected_gradient,
        spectra,
        complement_square,
        delta2,
    )
    mu_double = optimizer._diag_trust_region(
        projected_gradient.double(),
        spectra.double(),
        complement_square.double(),
        delta2.double(),
    )

    assert mu.dtype == th.float32
    assert mu_double.dtype == th.float64
    th.testing.assert_close(mu, mu_double.float(), rtol=0., atol=0.)


def test_find_eigen_preserves_single_sample_updater_mask_shape():
    counts = th.tensor([2], dtype=th.long)
    scales = th.tensor([1.])
    X = th.randn(1, 2, 3) * 0.1
    direction = th.randn_like(X)
    updater = _BatchUpdater()
    finder = FindEigen(
        Torque_thres=1.e-12,
        Eigen_thres=1.e-12,
        maxiter_lanczos=4,
        dx=0.02,
        device='cpu',
        verbose=0,
    )
    finder.set_batch_updater(updater)

    finder.run(
        _energy,
        X,
        direction,
        grad_func=_gradient,
        func_args=(counts, scales),
        grad_func_args=(counts, scales),
        is_grad_func_contain_y=False,
        batch_indices=counts.tolist(),
        eigen_order=2,
    )

    assert updater.masks
    assert all(mask.shape == (1,) for mask in updater.masks)


def test_find_eigen_rejects_convergence_before_requested_dimension():
    hessian_diagonal = th.arange(1., 7.).reshape(1, 2, 3)

    def energy(X):
        return (0.5 * hessian_diagonal * X.square()).sum(dim=(1, 2))

    def gradient(X):
        return hessian_diagonal * X

    X = th.zeros(1, 2, 3)
    direction = th.zeros_like(X)
    direction[0, 0, 0] = (1. - 1.e-6) ** 0.5
    direction[0, 0, 1] = 1.e-3
    finder = FindEigen(
        Torque_thres=1.e-2,
        Eigen_thres=1.e-8,
        maxiter_lanczos=5,
        dx=0.02,
        device='cpu',
        verbose=0,
        _hold_samples=True,
    )

    with pytest.raises(
            RuntimeError,
            match='converged before the requested Krylov dimension',
    ):
        finder.run(
            energy,
            X,
            direction,
            grad_func=gradient,
            is_grad_func_contain_y=False,
            batch_indices=[2],
            eigen_order=2,
        )


def test_krylov_uses_configured_float_type():
    input_dtype = th.float64 if FLOAT_TYPE == th.float32 else th.float32
    counts = th.tensor([3], dtype=th.long)
    scales = th.tensor([1.], dtype=FLOAT_TYPE)
    X = th.randn(1, 3, 3, dtype=input_dtype) * 0.1
    direction = th.randn_like(X)
    optimizer = _make_optimizer(KrylovNewton, 'cpu', maxiter=1)

    result = _run(optimizer, X, direction, counts, scales)

    assert all(value.dtype == FLOAT_TYPE for value in result)


def test_krylov_dynamics_zero_force_remains_finite():
    counts = th.tensor([3], dtype=th.long)
    scales = th.tensor([1.])
    X = th.zeros(1, 3, 3)
    direction = th.randn_like(X)
    optimizer = _make_optimizer(KrylovDynamics, 'cpu', maxiter=1)

    result = _run(optimizer, X.clone(), direction, counts, scales)

    assert all(value.isfinite().all() for value in result)
    th.testing.assert_close(result[1], X)


def test_krylov_dynamics_force_displacement_scales_with_time_step_squared():
    counts = th.tensor([3], dtype=th.long)
    scales = th.tensor([1.])
    X = th.randn(1, 3, 3) * 0.1
    direction = th.randn_like(X)
    displacements = []

    for time_step in (0.05, 0.1):
        optimizer = KrylovDynamics(
            E_threshold=1.e-9,
            Torque_thres=1.e-4,
            Eigen_thres=1.e-4,
            F_threshold=1.e-5,
            maxiter_trans=1,
            maxiter_eig=5,
            steplength=time_step,
            dx=0.02,
            device='cpu',
            verbose=0,
        )
        th.manual_seed(73)
        result = _run(
            optimizer,
            X.clone(),
            direction.clone(),
            counts,
            scales,
        )
        displacements.append(result[1] - X)

    th.testing.assert_close(displacements[1], 4. * displacements[0])


def test_krylov_newton_stuck_iteration_returns_consistent_state():
    counts = th.tensor([3], dtype=th.long)
    scales = th.tensor([1.])
    X = th.zeros(1, 3, 3)
    direction = th.randn_like(X)
    optimizer = _make_optimizer(KrylovNewton, 'cpu', maxiter=2)

    with pytest.warns(RuntimeWarning, match='All active iterations are stuck'):
        energies, coordinates, gradients = _run(
            optimizer,
            X.clone(),
            direction,
            counts,
            scales,
        )

    th.testing.assert_close(coordinates, X)
    th.testing.assert_close(gradients, _gradient(X, counts, scales))
    assert energies is optimizer.s.Energy
    assert coordinates is optimizer.s.X
    assert gradients is optimizer.s.X_grad
