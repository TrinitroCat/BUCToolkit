#  Copyright (c) 2026.7.20, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: blue_moon.py
#  Environment: Python 3.12
"""Blue-Moon mean forces and free-energy path integration."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Callable, Literal, Sequence

import numpy as np
import torch as th

from ._collective_variables import resolve_cv_trajectory
from .trajectory import MDTrajectory, as_md_trajectory


_KB_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class BlueMoonResult:
    """Blue-Moon image statistics and integrated relative free energies."""

    image_order: np.ndarray
    collective_variable_path: np.ndarray
    mean_force: np.ndarray
    mean_force_standard_error: np.ndarray | None
    weight_sum: np.ndarray
    segment_free_energy: np.ndarray
    free_energy: np.ndarray


class BlueMoonCalculator:
    """Calculate a Blue-Moon free-energy path from a batched CMD trajectory.

    Each structure in the MD batch is interpreted as one constrained image of
    the same physical system at a different point along a collective-variable
    path. Frames provide equilibrium samples inside each image/window.

    With BUCToolkit's sign convention and ``require_fixman=True`` outputs, the
    conditional mean-force estimator is

    ``<w * (Fc + kB*T*G)>_constraint / <w>_constraint``.

    The resulting force vectors are integrated along the ordered CV path using
    line elements ``mean_force . dCV``. No biasing-potential unbiasing is
    included because this calculator is specifically for exact holonomic
    constraints.

    Args:
        trajectory: Binary constrained-MD path or loaded
            :class:`MDTrajectory`.
    """

    def __init__(
            self,
            trajectory: MDTrajectory | str | PathLike[str],
    ) -> None:
        self.trajectory = as_md_trajectory(trajectory)

    @staticmethod
    def _as_constraint_matrix(values: np.ndarray, name: str) -> np.ndarray:
        """Normalize a dumped constraint field to ``(frame, image, cv)``."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 2:
            return values[..., None]
        if values.ndim < 2:
            raise ValueError(
                f'Dumped field {name!r} must retain frame and image axes, '
                f'but got shape {values.shape}.'
            )
        return values.reshape(values.shape[0], values.shape[1], -1)

    @staticmethod
    def _validate_image_order(
            image_order: Sequence[int] | np.ndarray | None,
            n_images: int,
    ) -> np.ndarray:
        """Return a validated image permutation."""
        if image_order is None:
            return np.arange(n_images, dtype=np.int64)
        order = np.asarray(image_order, dtype=np.int64).reshape(-1)
        if order.size != n_images or not np.array_equal(
                np.sort(order), np.arange(n_images, dtype=np.int64)
        ):
            raise ValueError(
                '`image_order` must be a permutation of '
                f'0..{n_images - 1}, but got {order.tolist()}.'
            )
        return order

    @staticmethod
    def _block_standard_error(
            force_samples: np.ndarray,
            weights: np.ndarray,
            block_size: int | None,
    ) -> np.ndarray | None:
        """Estimate per-image mean-force errors from weighted block means."""
        if block_size is None:
            return None
        if not isinstance(block_size, int):
            raise TypeError(
                f'`block_size` must be int or None, but got {type(block_size)}.'
            )
        if block_size <= 0:
            raise ValueError(f'`block_size` must be positive, but got {block_size}.')
        n_frame = force_samples.shape[0]
        n_block = n_frame // block_size
        if n_block < 2:
            raise ValueError(
                f'`block_size={block_size}` leaves only {n_block} complete '
                'blocks; at least two are required for a standard error.'
            )
        force_samples = force_samples[:n_block * block_size]
        weights = weights[:n_block * block_size]
        force_blocks = force_samples.reshape(
            n_block, block_size, *force_samples.shape[1:]
        )
        weight_blocks = weights.reshape(n_block, block_size, weights.shape[1])
        denominator = np.sum(weight_blocks, axis=1)
        if np.any(denominator <= 0.):
            raise ValueError('A Blue-Moon block has non-positive total weight.')
        block_means = np.sum(
            force_blocks * weight_blocks[..., None], axis=1
        ) / denominator[..., None]
        return np.std(block_means, axis=0, ddof=1) / np.sqrt(n_block)

    def calculate(
            self,
            cv_func: Callable[[th.Tensor], th.Tensor] | None = None,
            cv_values=None,
            cv_column: str | None = None,
            temperature: float | None = None,
            use_fixman: bool = True,
            start: int = 0,
            stop: int | None = None,
            stride: int = 1,
            image_order: Sequence[int] | np.ndarray | None = None,
            integration: Literal['trapezoid', 'left', 'right'] = 'trapezoid',
            reference_image: int = 0,
            block_size: int | None = None,
    ) -> BlueMoonResult:
        """Estimate image mean forces and integrate the free-energy path.

        Args:
            cv_func: CV function evaluated on dumped coordinates.
            cv_values: Explicit CV values or fixed per-image CV targets.
            cv_column: Name of a custom dumped CV/target column.
            temperature: Ensemble temperature in kelvin. Required when
                ``use_fixman=True`` because ``kB*T*G`` enters the estimator.
            use_fixman: Use dumped ``G`` and ``w``. If ``False``, calculate
                ordinary unweighted means of ``Fc``.
            start: First equilibrium frame included.
            stop: Exclusive final frame.
            stride: Positive sampling stride.
            image_order: Optional permutation defining path order. By default
                the original batch order is the path order.
            integration: Numerical line-integration rule between neighboring
                images.
            reference_image: Position in the ordered path whose free energy is
                shifted to zero.
            block_size: Optional number of selected frames per error-analysis
                block. Incomplete trailing blocks are discarded.

        Returns:
            :class:`BlueMoonResult`. Mean forces have units eV per CV unit and
            integrated free energies have units eV.
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
        constraint_forces = self._as_constraint_matrix(
            self.trajectory.stack('Fc', start=start, stop=stop, stride=stride),
            'Fc',
        )
        if constraint_forces.shape != collective_variables.shape:
            raise ValueError(
                'CV and Fc arrays must have identical '
                f'(frame, image, constraint) shapes, but got '
                f'{collective_variables.shape} and {constraint_forces.shape}.'
            )

        if use_fixman:
            self.trajectory.require_columns('G', 'w')
            if temperature is None:
                raise ValueError('`temperature` is required when `use_fixman=True`.')
            temperature = float(temperature)
            if not np.isfinite(temperature) or temperature <= 0.:
                raise ValueError(
                    f'`temperature` must be positive and finite, but got {temperature}.'
                )
            geometric_force = self._as_constraint_matrix(
                self.trajectory.stack('G', start=start, stop=stop, stride=stride),
                'G',
            )
            if geometric_force.shape != constraint_forces.shape:
                raise ValueError(
                    f'G shape {geometric_force.shape} does not match '
                    f'Fc shape {constraint_forces.shape}.'
                )
            weights = np.asarray(
                self.trajectory.stack('w', start=start, stop=stop, stride=stride),
                dtype=np.float64,
            )
            weights = weights.reshape(weights.shape[0], weights.shape[1], -1)
            if weights.shape[-1] != 1:
                raise ValueError(
                    f'Blue-Moon weights must be scalar per frame/image, got {weights.shape}.'
                )
            weights = weights[..., 0]
            force_samples = (
                constraint_forces + _KB_EV_PER_K * temperature * geometric_force
            )
        else:
            weights = np.ones(constraint_forces.shape[:2], dtype=np.float64)
            force_samples = constraint_forces

        if not np.all(np.isfinite(force_samples)):
            raise ValueError('Blue-Moon force samples contain non-finite values.')
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.):
            raise ValueError('Blue-Moon weights must be positive and finite.')

        weight_sum = np.sum(weights, axis=0)
        mean_force = np.sum(
            force_samples * weights[..., None], axis=0
        ) / weight_sum[:, None]
        cv_path = np.sum(
            collective_variables * weights[..., None], axis=0
        ) / weight_sum[:, None]
        standard_error = self._block_standard_error(
            force_samples, weights, block_size
        )

        order = self._validate_image_order(image_order, self.trajectory.n_images)
        cv_path = cv_path[order]
        mean_force = mean_force[order]
        weight_sum = weight_sum[order]
        if standard_error is not None:
            standard_error = standard_error[order]

        if not isinstance(reference_image, int):
            raise TypeError(
                f'`reference_image` must be int, but got {type(reference_image)}.'
            )
        if reference_image < 0:
            reference_image += self.trajectory.n_images
        if not 0 <= reference_image < self.trajectory.n_images:
            raise ValueError(
                f'`reference_image` must index the ordered path of length '
                f'{self.trajectory.n_images}, but got {reference_image}.'
            )

        path_increment = np.diff(cv_path, axis=0)
        if integration == 'trapezoid':
            segment_force = 0.5 * (mean_force[:-1] + mean_force[1:])
        elif integration == 'left':
            segment_force = mean_force[:-1]
        elif integration == 'right':
            segment_force = mean_force[1:]
        else:
            raise ValueError(
                f'Unknown integration rule {integration!r}; expected '
                "'trapezoid', 'left', or 'right'."
            )
        segment_free_energy = np.sum(segment_force * path_increment, axis=-1)
        free_energy = np.concatenate((
            np.zeros(1, dtype=np.float64),
            np.cumsum(segment_free_energy),
        ))
        free_energy -= free_energy[reference_image]

        return BlueMoonResult(
            image_order=order,
            collective_variable_path=cv_path,
            mean_force=mean_force,
            mean_force_standard_error=standard_error,
            weight_sum=weight_sum,
            segment_free_energy=segment_free_energy,
            free_energy=free_energy,
        )


__all__ = ['BlueMoonCalculator', 'BlueMoonResult']
