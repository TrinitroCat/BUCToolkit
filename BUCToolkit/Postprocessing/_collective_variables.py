#  Copyright (c) 2026.7.20, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: _collective_variables.py
#  Environment: Python 3.12
"""Internal helpers shared by constrained-MD postprocessors."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch as th

from .trajectory import MDTrajectory


def _as_cv_vector(value) -> np.ndarray:
    """Convert one CV-function result to a finite one-dimensional vector."""
    if isinstance(value, (tuple, list)):
        arrays = [np.asarray(item.detach().cpu() if isinstance(item, th.Tensor) else item) for item in value]
        value = np.concatenate([array.reshape(-1) for array in arrays])
    elif isinstance(value, th.Tensor):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError('A collective-variable function returned an empty value.')
    if not np.all(np.isfinite(vector)):
        raise ValueError(f'A collective-variable function returned non-finite values {vector}.')
    return vector


def _normalize_supplied_cv_values(
        values,
        trajectory: MDTrajectory,
        frame_indices: np.ndarray,
) -> np.ndarray:
    """Normalize user-supplied CV values to ``(selected_frame, image, cv)``."""
    values = np.asarray(values, dtype=np.float64)
    n_frames = trajectory.n_frames
    n_images = trajectory.n_images

    if values.ndim == 1:
        if n_images == 1 and values.shape[0] == n_frames:
            values = values[:, None, None]
        elif values.shape[0] == n_images:
            values = values[None, :, None]
        else:
            raise ValueError(
                'One-dimensional `cv_values` must contain either one scalar '
                f'per image ({n_images}) or, for a single image, one scalar '
                f'per frame ({n_frames}); got shape {values.shape}.'
            )
    elif values.ndim == 2:
        if values.shape[0] == n_images:
            # One fixed CV vector for every image/window.
            values = values[None, :, :]
        elif n_images == 1 and values.shape[0] == n_frames:
            values = values[:, None, :]
        else:
            raise ValueError(
                'Two-dimensional `cv_values` must have shape '
                f'(n_images, n_cv)=({n_images}, n_cv), or for one image '
                f'(n_frames, n_cv)=({n_frames}, n_cv); got {values.shape}.'
            )
    elif values.ndim != 3:
        raise ValueError(
            '`cv_values` must be one-, two-, or three-dimensional, but got '
            f'shape {values.shape}.'
        )

    if values.shape[1] != n_images:
        raise ValueError(
            f'`cv_values` has {values.shape[1]} images, expected {n_images}.'
        )
    if values.shape[0] == 1:
        values = np.broadcast_to(values, (len(frame_indices), *values.shape[1:])).copy()
    elif values.shape[0] == n_frames:
        values = values[frame_indices]
    else:
        raise ValueError(
            f'`cv_values` has {values.shape[0]} frames; expected 1 or {n_frames}.'
        )
    return values


def resolve_cv_trajectory(
        trajectory: MDTrajectory,
        cv_func: Callable[[th.Tensor], th.Tensor] | None = None,
        cv_values=None,
        cv_column: str | None = None,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
) -> np.ndarray:
    """Resolve collective variables as ``(frame, image, n_cv)``.

    Exactly one source must be selected: explicit ``cv_values``, a named dump
    ``cv_column``, or a differentiable/PyTorch ``cv_func`` evaluated on dumped
    coordinates.
    """
    selected_sources = sum(
        source is not None for source in (cv_func, cv_values, cv_column)
    )
    if selected_sources != 1:
        raise ValueError(
            'Exactly one of `cv_func`, `cv_values`, or `cv_column` must be provided.'
        )
    frame_indices = trajectory.frame_indices(start, stop, stride)

    if cv_values is not None:
        return _normalize_supplied_cv_values(cv_values, trajectory, frame_indices)

    if cv_column is not None:
        values = trajectory.stack(cv_column, start=start, stop=stop, stride=stride)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 2:
            values = values[..., None]
        elif values.ndim < 2:
            raise ValueError(
                f'CV dump column {cv_column!r} must retain frame and image '
                f'axes, but got shape {values.shape}.'
            )
        else:
            values = values.reshape(values.shape[0], values.shape[1], -1)
        if not np.all(np.isfinite(values)):
            raise ValueError(f'CV dump column {cv_column!r} contains non-finite values.')
        return values

    trajectory.require_columns('X')
    coordinates = trajectory.by_image('X', start=start, stop=stop, stride=stride)
    by_image = []
    expected_n_cv = None
    with th.no_grad():
        for image_index, image_coordinates in enumerate(coordinates):
            image_values = []
            for frame_coordinates in image_coordinates:
                value = _as_cv_vector(cv_func(th.as_tensor(frame_coordinates)))
                if expected_n_cv is None:
                    expected_n_cv = value.size
                elif value.size != expected_n_cv:
                    raise ValueError(
                        'The collective-variable output size changed from '
                        f'{expected_n_cv} to {value.size} for image {image_index}.'
                    )
                image_values.append(value)
            by_image.append(np.stack(image_values, axis=0))
    return np.stack(by_image, axis=1)


__all__ = ['resolve_cv_trajectory']
