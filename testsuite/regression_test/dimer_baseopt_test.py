import pytest
import torch as th

from BUCToolkit.BatchOptim.TS.Dimer import Dimer as LegacyDimer
from BUCToolkit.BatchOptim.TS.Dimer import Dimer


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
            scale * (2. * coefficients * coordinates + 0.4 * coordinates.pow(3))
        ).reshape(1, count, 3))
        atom_start += count
    return th.cat(gradients, dim=1)


class _BatchUpdater:
    def __init__(self):
        self.masks = []

    def __call__(self, mask, func_args, func_kwargs, grad_func_args, grad_func_kwargs):
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


def _make_optimizer(optimizer_type, device, maxiter=5):
    return optimizer_type(
        E_threshold=1.e-7,
        Torque_thres=1.e-4,
        Curvature_thres=-0.1,
        F_threshold=2.e-3,
        maxiter_trans=maxiter,
        maxiter_rot=6,
        max_steplength=0.08,
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


def test_dimer_baseopt_matches_legacy_with_dynamic_batch():
    th.manual_seed(13)
    counts = th.tensor([2, 5], dtype=th.long)
    scales = th.tensor([0.7, 1.3])
    X = th.randn(1, int(counts.sum()), 3) * 0.15
    X[:, :counts[0]].zero_()
    X_diff = th.randn_like(X)

    legacy_translation_updater = _BatchUpdater()
    legacy_rotation_updater = _BatchUpdater()
    current_translation_updater = _BatchUpdater()
    current_rotation_updater = _BatchUpdater()
    legacy = _make_optimizer(LegacyDimer, 'cpu', maxiter=6)
    current = _make_optimizer(Dimer, 'cpu', maxiter=6)
    legacy.set_batch_updater(legacy_translation_updater, legacy_rotation_updater)
    current.set_batch_updater(current_translation_updater, current_rotation_updater)

    legacy_result = _run(legacy, X.clone(), X_diff.clone(), counts, scales)
    current_result = _run(current, X.clone(), X_diff.clone(), counts, scales)

    for legacy_value, current_value in zip(legacy_result, current_result):
        th.testing.assert_close(current_value, legacy_value, rtol=0., atol=0.)
    assert any(mask.tolist() == [False, True] for mask in current_translation_updater.masks)
    assert current._extra_converge_mask.tolist() == [True, True]
    assert current.s.Energy is current_result[0]
    assert current.s.X is current_result[1]


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param('cuda:0', marks=pytest.mark.skipif(not th.cuda.is_available(), reason='CUDA unavailable')),
])
def test_dimer_baseopt_matches_legacy_with_fixed_atoms(device):
    th.manual_seed(29)
    counts = th.tensor([4], dtype=th.long, device=device)
    scales = th.tensor([1.], device=device)
    X = th.randn(1, 4, 3, device=device) * 0.2
    X_diff = th.randn_like(X)
    fixed_mask = th.ones_like(X)
    fixed_mask[:, -1].zero_()

    legacy = _make_optimizer(LegacyDimer, device, maxiter=4)
    current = _make_optimizer(Dimer, device, maxiter=4)
    legacy_result = _run(legacy, X.clone(), X_diff.clone(), counts, scales, fixed_mask)
    current_result = _run(current, X.clone(), X_diff.clone(), counts, scales, fixed_mask)

    for legacy_value, current_value in zip(legacy_result, current_result):
        th.testing.assert_close(current_value, legacy_value, rtol=0., atol=0.)
    th.testing.assert_close(current_result[1][:, -1], X[:, -1], rtol=0., atol=0.)


def test_dimer_initial_rotation_is_the_first_energy_gradient_evaluation():
    th.manual_seed(41)
    counts = th.tensor([4], dtype=th.long)
    scales = th.tensor([1.])
    X = th.randn(1, 4, 3) * 0.2
    X_diff = th.randn_like(X)

    legacy_energy = _CallCounter(_energy)
    legacy_gradient = _CallCounter(_gradient)
    current_energy = _CallCounter(_energy)
    current_gradient = _CallCounter(_gradient)
    legacy = _make_optimizer(LegacyDimer, 'cpu', maxiter=1)
    current = _make_optimizer(Dimer, 'cpu', maxiter=1)

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
