#  Copyright (c) 2026.7.20, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: trajectory.py
#  Environment: Python 3.12
"""Shared access to named molecular-dynamics trajectory columns."""

from __future__ import annotations

from os import PathLike
from typing import Dict, List, Tuple

import numpy as np

from BUCToolkit.BatchStructures import read_dump_arrays, read_md_traj
from BUCToolkit.utils._Element_info import N_MASS


class MDTrajectory:
    """Load and organize one named BUCToolkit MD binary trajectory.

    Dynamic arrays returned by :func:`read_md_traj` are ordered as
    ``frame0/image0, frame0/image1, ..., frame1/image0, ...``.  This class
    records that layout once and provides convenient frame-major stacks or
    per-image trajectories.  Per-image access also supports irregular batches
    whose structures contain different atom counts.

    Args:
        path: Binary trajectory written by a BUCToolkit MD runner.
        indices: Frame selection forwarded to :func:`read_md_traj`. A negative
            integer loads every dumped frame. Lists and slices are applied
            independently to every dynamic group in the file.
        is_copy: Whether arrays are copied out of the underlying mmap. The
            default is strongly recommended because calculators retain the
            arrays after the reader has closed the file.

    Attributes:
        path: Original trajectory path.
        columns: Mapping from every dumped dynamic quantity name to its flat,
            cycle-major list of per-image values.
        n_images: Number of structures/images in one dumped frame.
        n_frames: Number of dumped frames.

    Raises:
        ValueError: If no dynamic columns exist, column lengths are
            inconsistent, or a column length is not divisible by the batch
            size recorded in the static header.
    """

    def __init__(
            self,
            path: str | PathLike[str],
            indices: List[int] | slice | int = -1,
            is_copy: bool = True,
    ) -> None:
        self.path = str(path)
        raw = read_dump_arrays(self.path, indices=indices, is_copy=is_copy)
        self._raw_header = {
            name: raw[name]
            for name in ('batch_indices', 'cell_vec', 'atomic_numbers', 'fixed_mask')
            if name in raw
        }
        self.columns: Dict[str, List] = read_md_traj(
            self.path,
            indices=indices,
            is_copy=is_copy,
            out_arrays=True,
        )
        self.n_images = self._infer_n_images(self._raw_header)
        if not self.columns:
            raise ValueError(f'No dynamic trajectory columns were found in {self.path!r}.')

        lengths = {name: len(values) for name, values in self.columns.items()}
        first_length = next(iter(lengths.values()))
        if any(length != first_length for length in lengths.values()):
            raise ValueError(f'Inconsistent dynamic column lengths: {lengths}.')
        if first_length % self.n_images != 0:
            raise ValueError(
                f'Dynamic sample count {first_length} is not divisible by '
                f'the batch/image count {self.n_images}.'
            )
        self.n_frames = first_length // self.n_images
        self._atomic_numbers = self._split_atomic_numbers(self._raw_header)

    @staticmethod
    def _infer_n_images(header: Dict[str, np.ndarray]) -> int:
        """Infer the real batch size from the static trajectory header."""
        if 'batch_indices' in header:
            n_images = len(np.asarray(header['batch_indices']).reshape(-1))
        else:
            atomic_numbers = np.asarray(header['atomic_numbers'])
            n_images = 1 if atomic_numbers.ndim == 1 else atomic_numbers.shape[0]
        if n_images <= 0:
            raise ValueError(f'Invalid trajectory batch/image count {n_images}.')
        return int(n_images)

    @staticmethod
    def _split_atomic_numbers(header: Dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Return one one-dimensional atomic-number array per image."""
        atomic_numbers = np.asarray(header['atomic_numbers'])
        if 'batch_indices' in header:
            counts = np.asarray(header['batch_indices'], dtype=np.int64).reshape(-1)
            split_points = np.cumsum(counts)[:-1]
            return tuple(np.split(atomic_numbers.reshape(-1), split_points))
        if atomic_numbers.ndim == 1:
            return (atomic_numbers.reshape(-1),)
        return tuple(np.asarray(row).reshape(-1) for row in atomic_numbers)

    @property
    def available_columns(self) -> Tuple[str, ...]:
        """Names of all dynamic quantities available to calculators."""
        return tuple(self.columns)

    @property
    def atomic_numbers(self) -> Tuple[np.ndarray, ...]:
        """Atomic numbers for every image, returned as independent copies."""
        return tuple(values.copy() for values in self._atomic_numbers)

    @property
    def masses(self) -> Tuple[np.ndarray, ...]:
        """Atomic masses in amu for every image."""
        return tuple(
            np.asarray([N_MASS[int(number)] for number in numbers], dtype=np.float64)
            for numbers in self._atomic_numbers
        )

    def require_columns(self, *names: str) -> None:
        """Require named dynamic columns to exist.

        Args:
            *names: Column names required by a calculation.

        Raises:
            ValueError: If one or more requested columns were not dumped.
        """
        missing = [name for name in names if name not in self.columns]
        if missing:
            raise ValueError(
                f'Trajectory {self.path!r} does not contain columns {missing}. '
                f'Available columns are {list(self.columns)}.'
            )

    def frame_indices(
            self,
            start: int = 0,
            stop: int | None = None,
            stride: int = 1,
    ) -> np.ndarray:
        """Return validated dumped-frame indices for a calculation window."""
        if not isinstance(start, int):
            raise TypeError(f'`start` must be int, but got {type(start)}.')
        if stop is not None and not isinstance(stop, int):
            raise TypeError(f'`stop` must be int or None, but got {type(stop)}.')
        if not isinstance(stride, int):
            raise TypeError(f'`stride` must be int, but got {type(stride)}.')
        if stride <= 0:
            raise ValueError(f'`stride` must be positive, but got {stride}.')
        indices = np.arange(self.n_frames, dtype=np.int64)[slice(start, stop, stride)]
        if indices.size == 0:
            raise ValueError(
                f'The requested frame window start={start}, stop={stop}, '
                f'stride={stride} is empty for {self.n_frames} frames.'
            )
        return indices

    def by_image(
            self,
            name: str,
            start: int = 0,
            stop: int | None = None,
            stride: int = 1,
    ) -> Tuple[np.ndarray, ...]:
        """Return one stacked trajectory array per batch image.

        This is the preferred accessor for atom-wise columns because images in
        an irregular batch may contain different numbers of atoms.

        Args:
            name: Dynamic column name, such as ``'X'``, ``'V'``, or ``'Fc'``.
            start: First dumped frame included in the result.
            stop: Exclusive final dumped frame, or ``None``.
            stride: Positive dumped-frame stride.

        Returns:
            Tuple of length :attr:`n_images`. Element ``i`` is a NumPy array
            whose leading dimension is the selected frame count.

        Raises:
            ValueError: If the column is missing or values belonging to one
                image do not have a consistent shape across frames.
        """
        self.require_columns(name)
        frame_indices = self.frame_indices(start, stop, stride)
        flat_values = self.columns[name]
        out = []
        for image_index in range(self.n_images):
            values = [
                np.asarray(flat_values[frame * self.n_images + image_index])
                for frame in frame_indices
            ]
            try:
                out.append(np.stack(values, axis=0))
            except ValueError as error:
                shapes = [value.shape for value in values]
                raise ValueError(
                    f'Column {name!r}, image {image_index} changes shape '
                    f'across frames: {shapes}.'
                ) from error
        return tuple(out)

    def stack(
            self,
            name: str,
            start: int = 0,
            stop: int | None = None,
            stride: int = 1,
    ) -> np.ndarray:
        """Return a regular frame-major array ``(frame, image, ...)``.

        Raises:
            ValueError: If images have different trailing shapes. Use
                :meth:`by_image` for irregular atom-wise data.
        """
        by_image = self.by_image(name, start=start, stop=stop, stride=stride)
        try:
            return np.stack(by_image, axis=1)
        except ValueError as error:
            shapes = [value.shape for value in by_image]
            raise ValueError(
                f'Column {name!r} cannot form a regular batch because image '
                f'trajectories have shapes {shapes}. Use `by_image` instead.'
            ) from error


def as_md_trajectory(source: MDTrajectory | str | PathLike[str]) -> MDTrajectory:
    """Normalize a trajectory path or existing :class:`MDTrajectory`."""
    if isinstance(source, MDTrajectory):
        return source
    return MDTrajectory(source)


__all__ = ['MDTrajectory', 'as_md_trajectory']
