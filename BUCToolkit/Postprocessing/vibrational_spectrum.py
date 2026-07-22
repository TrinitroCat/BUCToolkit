#  Copyright (c) 2026.7.20, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: vibrational_spectrum.py
#  Environment: Python 3.12
"""Vibrational spectra from velocity autocorrelation functions."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Literal, Sequence

import numpy as np

from .trajectory import MDTrajectory, as_md_trajectory


_FS_INV_TO_WAVENUMBER = 1.0e15 / 2.99792458e10


@dataclass(frozen=True)
class VibrationalSpectrumResult:
    """Result of a velocity-autocorrelation spectrum calculation.

    Arrays retain a leading image/batch dimension. Frequency is provided in
    three common units; ``frequency_fs_inv`` is cycles per femtosecond, not
    angular frequency.
    """

    lag_time_fs: np.ndarray
    vacf: np.ndarray
    frequency_fs_inv: np.ndarray
    frequency_thz: np.ndarray
    wavenumber_cm1: np.ndarray
    spectrum: np.ndarray

    @property
    def mean_vacf(self) -> np.ndarray:
        """VACF averaged equally over all batch images."""
        return np.mean(self.vacf, axis=0)

    @property
    def mean_spectrum(self) -> np.ndarray:
        """Spectrum averaged equally over all batch images."""
        return np.mean(self.spectrum, axis=0)


class VibrationalSpectrumCalculator:
    """Calculate vibrational spectra from dumped MD velocities.

    The velocity autocorrelation function is evaluated using the
    Wiener--Khinchin FFT identity, including the unbiased ``N-lag`` time-origin
    normalization. A second real FFT produces the one-sided cosine spectrum.

    Args:
        trajectory: Binary MD path or an already loaded :class:`MDTrajectory`.
        sample_spacing_fs: Physical time in femtoseconds between two dumped
            frames. For an MD timestep ``dt`` and output interval ``n``, pass
            ``dt * n``. The binary trajectory currently does not store this
            metadata automatically.

    Notes:
        The returned spectrum is a discretized cosine transform of the VACF.
        Its absolute normalization depends on whether the VACF is normalized;
        peak locations and relative intensities are normally the quantities of
        interest. Quantum intensity corrections are intentionally not applied.
    """

    def __init__(
            self,
            trajectory: MDTrajectory | str | PathLike[str],
            sample_spacing_fs: float,
    ) -> None:
        self.trajectory = as_md_trajectory(trajectory)
        self.sample_spacing_fs = float(sample_spacing_fs)
        if not np.isfinite(self.sample_spacing_fs) or self.sample_spacing_fs <= 0.:
            raise ValueError(
                '`sample_spacing_fs` must be a positive finite number, but got '
                f'{sample_spacing_fs}.'
            )

    @staticmethod
    def _window_values(
            window: Literal['hann', 'hamming', 'blackman'] | Sequence[float] | None,
            n_lag: int,
    ) -> np.ndarray:
        """Resolve a named or explicit VACF window."""
        if window is None:
            return np.ones(n_lag, dtype=np.float64)
        if isinstance(window, str):
            functions = {
                'hann': np.hanning,
                'hamming': np.hamming,
                'blackman': np.blackman,
            }
            try:
                return functions[window.lower()](n_lag)
            except KeyError as error:
                raise ValueError(
                    f'Unknown window {window!r}; expected one of {tuple(functions)}.'
                ) from error
        values = np.asarray(window, dtype=np.float64).reshape(-1)
        if values.size != n_lag:
            raise ValueError(
                f'Explicit window has length {values.size}, expected {n_lag}.'
            )
        return values

    @staticmethod
    def _calculate_vacf(
            velocities: np.ndarray,
            masses: np.ndarray,
            mass_weighted: bool,
            remove_center_of_mass: bool,
            detrend: bool,
            normalize: bool,
    ) -> np.ndarray:
        """Calculate one image's VACF by FFT over all Cartesian components."""
        velocities = np.asarray(velocities, dtype=np.float64)
        if velocities.ndim != 3:
            raise ValueError(
                'Velocity trajectories must have shape (frame, atom, dimension), '
                f'but got {velocities.shape}.'
            )
        n_frame, n_atom, _ = velocities.shape
        masses = np.asarray(masses, dtype=np.float64).reshape(-1)
        if masses.size != n_atom:
            raise ValueError(
                f'Mass count {masses.size} does not match atom count {n_atom}.'
            )
        if np.any(masses <= 0.) or not np.all(np.isfinite(masses)):
            raise ValueError('All masses must be positive and finite.')

        values = velocities.copy()
        if remove_center_of_mass:
            center_velocity = np.sum(
                values * masses[None, :, None], axis=1, keepdims=True
            ) / np.sum(masses)
            values -= center_velocity
        if detrend:
            values -= np.mean(values, axis=0, keepdims=True)
        if mass_weighted:
            values *= np.sqrt(masses)[None, :, None]

        values = values.reshape(n_frame, -1)
        # At least 2*N points are required to prevent circular correlation.
        correlation_fft_size = 1 << (2 * n_frame - 1).bit_length()
        transformed = np.fft.rfft(values, n=correlation_fft_size, axis=0)
        autocorrelation = np.fft.irfft(
            transformed.conjugate() * transformed,
            n=correlation_fft_size,
            axis=0,
        )[:n_frame]
        autocorrelation = np.sum(autocorrelation, axis=1)
        autocorrelation /= np.arange(n_frame, 0, -1, dtype=np.float64)

        if normalize:
            if not np.isfinite(autocorrelation[0]) or autocorrelation[0] <= 0.:
                raise ValueError(
                    'The zero-lag velocity autocorrelation is not positive; '
                    'a normalized spectrum cannot be constructed.'
                )
            autocorrelation /= autocorrelation[0]
        return autocorrelation

    def calculate(
            self,
            start: int = 0,
            stop: int | None = None,
            stride: int = 1,
            max_lag: int | None = None,
            mass_weighted: bool = True,
            remove_center_of_mass: bool = True,
            detrend: bool = False,
            normalize_vacf: bool = True,
            window: Literal['hann', 'hamming', 'blackman'] | Sequence[float] | None = 'hann',
            n_fft: int | None = None,
            clip_negative: bool = False,
    ) -> VibrationalSpectrumResult:
        """Calculate per-image VACFs and their one-sided FFT spectra.

        Args:
            start: First dumped frame included, commonly used to discard
                equilibration.
            stop: Exclusive final dumped frame, or ``None``.
            stride: Positive stride between selected dumped frames. The
                effective spacing becomes ``sample_spacing_fs * stride``.
            max_lag: Number of VACF lag points retained, including zero. By
                default all selected frames are used.
            mass_weighted: Multiply atomic velocities by ``sqrt(mass)`` before
                forming the VACF.
            remove_center_of_mass: Remove each frame's mass-weighted center-of-
                mass velocity.
            detrend: Remove the time mean of every remaining velocity
                component.
            normalize_vacf: Divide each image's VACF by its zero-lag value.
            window: VACF apodization. Supported names are ``'hann'``,
                ``'hamming'``, and ``'blackman'``; ``None`` disables it.
            n_fft: FFT size used for the final spectrum. It must be at least
                ``max_lag``. The default is the next power of two.
            clip_negative: Replace negative noisy spectral values by zero.

        Returns:
            :class:`VibrationalSpectrumResult` containing per-image and mean
            VACFs/spectra.
        """
        velocities = self.trajectory.by_image('V', start=start, stop=stop, stride=stride)
        masses = self.trajectory.masses
        vacf = np.stack([
            self._calculate_vacf(
                image_velocity,
                image_mass,
                mass_weighted=mass_weighted,
                remove_center_of_mass=remove_center_of_mass,
                detrend=detrend,
                normalize=normalize_vacf,
            )
            for image_velocity, image_mass in zip(velocities, masses)
        ], axis=0)

        n_selected_frame = vacf.shape[1]
        if max_lag is None:
            max_lag = n_selected_frame
        if not isinstance(max_lag, int):
            raise TypeError(f'`max_lag` must be int or None, but got {type(max_lag)}.')
        if max_lag <= 1 or max_lag > n_selected_frame:
            raise ValueError(
                f'`max_lag` must satisfy 1 < max_lag <= {n_selected_frame}, '
                f'but got {max_lag}.'
            )
        vacf = vacf[:, :max_lag]
        window_values = self._window_values(window, max_lag)

        if n_fft is None:
            n_fft = 1 << (max_lag - 1).bit_length()
        if not isinstance(n_fft, int):
            raise TypeError(f'`n_fft` must be int or None, but got {type(n_fft)}.')
        if n_fft < max_lag:
            raise ValueError(f'`n_fft` ({n_fft}) must be at least max_lag ({max_lag}).')

        effective_spacing = self.sample_spacing_fs * stride
        spectrum = effective_spacing * np.real(np.fft.rfft(
            vacf * window_values[None, :],
            n=n_fft,
            axis=1,
        ))
        if clip_negative:
            spectrum = np.maximum(spectrum, 0.)
        frequency_fs_inv = np.fft.rfftfreq(n_fft, d=effective_spacing)

        return VibrationalSpectrumResult(
            lag_time_fs=np.arange(max_lag, dtype=np.float64) * effective_spacing,
            vacf=vacf,
            frequency_fs_inv=frequency_fs_inv,
            frequency_thz=frequency_fs_inv * 1000.,
            wavenumber_cm1=frequency_fs_inv * _FS_INV_TO_WAVENUMBER,
            spectrum=spectrum,
        )


__all__ = ['VibrationalSpectrumCalculator', 'VibrationalSpectrumResult']
