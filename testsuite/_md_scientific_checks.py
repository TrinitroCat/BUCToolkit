"""Shared soft statistical checks for the central BUCToolkit test modes.

The central ``main`` suite reports statistical diagnostics, but it does not
turn a statistical deviation into a hard test failure.  ``check_statistical``
keeps that contract: it prints the measured and expected values and returns a
boolean, so one runner's deviation cannot hide the results of later runners.

Keep the mathematical standards (DOF, reference energies, temperatures) in the
individual tests: checking that the runner reports the correct internal
``free_degree`` is part of the test, so the expected values are intentionally
hard-coded there rather than derived from the runner.
"""

from __future__ import annotations

import math

import numpy as np


KB_EV_PER_K = 8.617333262145e-5
EV_PER_AMU_A2_PER_FS2 = 103.642696562621738


def assert_statistical_equal(a, b, rtol=1e-5, atol=1e-7, msg=None):
    """Raise ``AssertionError`` unless two statistical values agree.

    Args:
        a: Measured statistical value.
        b: Expected statistical value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        msg: Optional custom error message.

    Returns:
        None.

    Raises:
        AssertionError: If the values do not agree within the tolerances.
    """
    err_msg = str(msg) if msg is not None else f'Statistical validation Failed.\na: {a}\nb: {b}'
    if abs(a) * abs(b) < 1e-7:
        atol = max(1e-7, atol)
    if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(err_msg)


def check_statistical(name, value, expected, rtol=1e-5, atol=1e-7, printer=print, label=None):
    """Soft statistical check: print pass/fail and return ``True`` or ``False``.

    Args:
        name: Human-readable quantity name.
        value: Measured value.
        expected: Expected value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        printer: Callable used for the pass/fail line.
        label: Optional sub-test label; when given, the existing central-suite
            ``"<name>" Test <label>`` wording is used.

    Returns:
        ``True`` when the values agree, otherwise ``False``.  This function
        never raises for a statistical deviation.
    """
    try:
        assert_statistical_equal(value, expected, rtol=rtol, atol=atol)
    except AssertionError:
        if label is None:
            printer(f'"{name}" FAILED: value={value}, expected={expected}')
        else:
            printer(
                f'\n"{name}" Test {label} Failed:\n'
                f'    test value: {value}\n'
                f'    standard value: {expected}'
            )
        return False
    if label is None:
        printer(f'"{name}" passed: value={value}, expected={expected}')
    else:
        printer(f'\n"{name}" Test {label} passed. <<<<<')
    return True


def check_equipartition(values, standards, rtol, names=None, printer=print):
    """Soft-check nested per-term equipartition values against standards.

    Args:
        values: Nested sequence; ``values[i][j]`` is the measured value of the
            ``j``-th batch for term ``i``.
        standards: Nested sequence with the same layout as ``values``.
        rtol: Relative tolerance forwarded to :func:`check_statistical`.
        names: Optional term names.
        printer: Callable used for the pass/fail lines.

    Returns:
        A flat list of booleans, one per checked term/batch pair.
    """
    results = []
    for i, (term_values, term_standards) in enumerate(zip(values, standards)):
        term_name = names[i] if names is not None else f'term {i}'
        for j, (value, expected) in enumerate(zip(term_values, term_standards)):
            results.append(
                check_statistical(
                    term_name, value, expected, rtol=rtol,
                    printer=printer, label=j + 1,
                )
            )
    return results


def kinetic_energy(velocities, masses):
    """Kinetic energy in eV for velocities in Ang/fs and masses in amu.

    Args:
        velocities: Array shaped ``(..., n_atom, 3)`` or ``(n_atom, 3)``.
        masses: Per-atom masses in amu, shaped ``(n_atom,)``.

    Returns:
        Kinetic energy with the leading shape of ``velocities``, or a scalar
        when ``velocities`` is two-dimensional.
    """
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    return 0.5 * np.sum(m[..., None] * v * v, axis=(-2, -1)) * EV_PER_AMU_A2_PER_FS2


def energy_drift_metric(total_energy):
    """Normalized total-energy range used by the central NVE checks.

    Args:
        total_energy: Sequence of total-energy samples.

    Returns:
        ``len(E) * (max(E) - min(E)) / sum(E)``.
    """
    energy = np.asarray(total_energy, dtype=float)
    return len(energy) * (energy.max() - energy.min()) / energy.sum()


def production_start(n_frames, fraction=0.4, stride=1):
    """Return the first production frame, advanced to a multiple of ``stride``.

    Args:
        n_frames: Number of dumped frames.
        fraction: Fraction of the trajectory discarded as equilibration.
        stride: Frame stride used by the test's batch-major loop.

    Returns:
        First production frame index.
    """
    start = int(n_frames * fraction)
    while start % stride:
        start += 1
    return start


def virial_from_forces(positions, forces, center=None):
    """Return ``-sum((X - center) * F)``, the virial-theorem right-hand side.

    Args:
        positions: Array shaped ``(..., n_atom, 3)``.
        forces: Forces with the same shape as ``positions``.
        center: Optional reference position; the harmonic-trap case uses the
            trap center when it is not the origin.

    Returns:
        Virial contribution with the leading shape of ``positions``.
    """
    x = np.asarray(positions, dtype=float)
    f = np.asarray(forces, dtype=float)
    if center is not None:
        x = x - np.asarray(center, dtype=float)
    return -np.sum(x * f, axis=(-2, -1))


def check_virial_theorem(name, positions, forces, kinetic, rtol, center=None, printer=print):
    """Soft-check ``< -sum(r.F) > == 2 <K>``.

    Args:
        name: Human-readable check name.
        positions: Positions shaped ``(..., n_atom, 3)``.
        forces: Forces with the same shape as ``positions``.
        kinetic: Kinetic-energy samples matching the leading shape of
            ``positions``.
        rtol: Relative tolerance.
        center: Optional reference position for the virial sum.
        printer: Callable used for the pass/fail line.

    Returns:
        ``True`` when the virial identity agrees within ``rtol``.
    """
    virial = np.asarray(virial_from_forces(positions, forces, center=center))
    kinetic = np.asarray(kinetic, dtype=float)
    return check_statistical(
        name,
        float(np.mean(virial)),
        float(2.0 * np.mean(kinetic)),
        rtol=rtol,
        printer=printer,
    )


def work_energy_integral(times, dVdt):
    """Trapezoidal integral of ``dV/dt`` over ``times``.

    Args:
        times: Frame times.
        dVdt: Values of ``dV/dt`` at ``times``.

    Returns:
        The integrated work/energy change.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(dVdt, dtype=float)
    if t.shape != y.shape:
        raise ValueError(f'times and dVdt must have matching shapes, got {t.shape} and {y.shape}.')
    if t.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(t)))
