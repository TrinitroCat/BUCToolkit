#  Copyright (c) 2026.7.20, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: constraint_work.py
#  Environment: Python 3.12
"""Ideal-constraint work and Jarzynski exponential averages."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Callable, Literal

import numpy as np
import torch as th

from ._collective_variables import resolve_cv_trajectory
from .trajectory import MDTrajectory, as_md_trajectory


_KB_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class ConstraintWorkResult:
    """Work histories for an ensemble of driven constrained trajectories."""

    collective_variables: np.ndarray
    work_increment_by_constraint: np.ndarray
    work_increment: np.ndarray
    cumulative_work: np.ndarray
    work: np.ndarray
    beta: float | None
    exponential_work: np.ndarray | None
    exponential_average: float | None
    jarzynski_free_energy: float | None


class ConstraintWorkCalculator:
    """Calculate ideal-constraint work from ``Fc`` and the CV protocol.

    For the sign convention used by :class:`BaseConstr`, the Cartesian
    constraint reaction is ``J.T @ Fc``. The work performed on the system is

    ``dW = Fc . d(xi_target)``.

    The first dumped ``Fc`` value is an initialization placeholder. Therefore
    the default ``'right'`` rule pairs the transition
    ``xi[n-1] -> xi[n]`` with ``Fc[n]``. Alternative quadratures are provided
    for timestep-convergence studies.

    Args:
        trajectory: Binary constrained-MD path or loaded
            :class:`MDTrajectory`. The file must contain ``Fc`` and either
            coordinates ``X`` or a requested custom CV column.
    """

    def __init__(
            self,
            trajectory: MDTrajectory | str | PathLike[str],
    ) -> None:
        self.trajectory = as_md_trajectory(trajectory)

    @staticmethod
    def _stable_exponential_average(work: np.ndarray, beta: float) -> tuple:
        """Return individual exponentials, their mean, and ``Delta F`` stably."""
        exponent = -beta * np.asarray(work, dtype=np.float64)
        maximum = np.max(exponent)
        log_average = maximum + np.log(np.mean(np.exp(exponent - maximum)))
        with np.errstate(over='ignore', under='ignore'):
            exponential_work = np.exp(exponent)
            exponential_average = float(np.exp(log_average))
        free_energy = float(-log_average / beta)
        return exponential_work, exponential_average, free_energy

    def calculate(
            self,
            cv_func: Callable[[th.Tensor], th.Tensor] | None = None,
            cv_values=None,
            cv_column: str | None = None,
            temperature: float | None = None,
            start: int = 0,
            stop: int | None = None,
            stride: int = 1,
            integration: Literal['right', 'left', 'trapezoid'] = 'right',
            force_sign: float = 1.,
            require_common_protocol: bool = True,
            protocol_atol: float = 1e-4,
    ) -> ConstraintWorkResult:
        """Calculate work histories and an optional Jarzynski estimate.

        Args:
            cv_func: CV function evaluated independently on every dumped
                structure. It should use the same definition and ordering as
                the constrained MD run.
            cv_values: Explicit CV values. Accepted shapes are
                ``(frame, image, cv)``, ``(image, cv)`` for fixed image values,
                and their documented scalar-CV reductions.
            cv_column: Name of a custom dumped CV/target column.
            temperature: Initial canonical temperature in kelvin. If supplied,
                calculate ``<exp(-beta W)>`` and
                ``Delta F = -log(<exp(-beta W)>)/beta``.
            start: First dumped frame included.
            stop: Exclusive final dumped frame.
            stride: Positive dumped-frame stride. Accurate driven work normally
                requires every MD step to be dumped; a larger stride performs a
                correspondingly coarse quadrature.
            integration: Force sampling rule for each CV increment. ``'right'``
                matches the current constrained-MD dump staging.
            force_sign: Optional convention conversion applied to ``Fc``.
            require_common_protocol: Require all batch trajectories to follow
                the same CV target sequence before taking a Jarzynski ensemble
                average.
            protocol_atol: Absolute tolerance for the common-protocol check.

        Returns:
            :class:`ConstraintWorkResult`. Work quantities are in eV when each
            ``Fc[k]`` is in eV per unit of ``CV[k]``.

        Raises:
            ValueError: If fewer than two frames are selected, dimensions do
                not match, temperature is invalid, or batch trajectories do
                not share one protocol when required.
        """
        self.trajectory.require_columns('Fc')
        collective_variables = resolve_cv_trajectory(
            self.trajectory,
            cv_func=cv_func,
            cv_values=cv_values,
            cv_column=cv_column,
            start=start,
            stop=stop,
            stride=stride,
        )
        constraint_forces = np.asarray(
            self.trajectory.stack('Fc', start=start, stop=stop, stride=stride),
            dtype=np.float64,
        )
        if constraint_forces.ndim == 2:
            constraint_forces = constraint_forces[..., None]
        else:
            constraint_forces = constraint_forces.reshape(
                constraint_forces.shape[0], constraint_forces.shape[1], -1
            )
        if collective_variables.shape != constraint_forces.shape:
            raise ValueError(
                'CV and constraint-force arrays must have identical '
                f'(frame, image, constraint) shapes, but got '
                f'{collective_variables.shape} and {constraint_forces.shape}.'
            )
        if collective_variables.shape[0] < 2:
            raise ValueError('At least two dumped frames are required to calculate work.')
        if not np.isfinite(force_sign):
            raise ValueError(f'`force_sign` must be finite, but got {force_sign}.')
        if protocol_atol < 0. or not np.isfinite(protocol_atol):
            raise ValueError('`protocol_atol` must be finite and non-negative.')

        if require_common_protocol and collective_variables.shape[1] > 1:
            reference = collective_variables[:, :1, :]
            maximum_protocol_difference = np.max(np.abs(
                collective_variables - reference
            ))
            if maximum_protocol_difference > protocol_atol:
                raise ValueError(
                    'Jarzynski trajectories do not share one CV protocol: '
                    f'maximum difference {maximum_protocol_difference:.6e} '
                    f'exceeds protocol_atol={protocol_atol:.6e}.'
                )

        cv_increment = np.diff(collective_variables, axis=0)
        if integration == 'right':
            integration_force = constraint_forces[1:]
        elif integration == 'left':
            integration_force = constraint_forces[:-1]
        elif integration == 'trapezoid':
            integration_force = 0.5 * (
                constraint_forces[:-1] + constraint_forces[1:]
            )
        else:
            raise ValueError(
                f'Unknown integration rule {integration!r}; expected '
                "'right', 'left', or 'trapezoid'."
            )

        work_increment_by_constraint = (
            float(force_sign) * integration_force * cv_increment
        )
        work_increment = np.sum(work_increment_by_constraint, axis=-1)
        cumulative_work = np.concatenate((
            np.zeros((1, self.trajectory.n_images), dtype=np.float64),
            np.cumsum(work_increment, axis=0),
        ), axis=0)
        work = cumulative_work[-1]

        beta = None
        exponential_work = None
        exponential_average = None
        jarzynski_free_energy = None
        if temperature is not None:
            temperature = float(temperature)
            if not np.isfinite(temperature) or temperature <= 0.:
                raise ValueError(
                    f'`temperature` must be positive and finite, but got {temperature}.'
                )
            beta = 1. / (_KB_EV_PER_K * temperature)
            exponential_work, exponential_average, jarzynski_free_energy = (
                self._stable_exponential_average(work, beta)
            )

        return ConstraintWorkResult(
            collective_variables=collective_variables,
            work_increment_by_constraint=work_increment_by_constraint,
            work_increment=work_increment,
            cumulative_work=cumulative_work,
            work=work,
            beta=beta,
            exponential_work=exponential_work,
            exponential_average=exponential_average,
            jarzynski_free_energy=jarzynski_free_energy,
        )


__all__ = ['ConstraintWorkCalculator', 'ConstraintWorkResult']
