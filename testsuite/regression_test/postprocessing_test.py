"""Focused analytic tests for trajectory postprocessing calculators."""

import math
import tempfile
import unittest

import numpy as np
import torch as th

from BUCToolkit.BatchStructures import ArrayDumper
from BUCToolkit.Postprocessing import (
    BlueMoonCalculator,
    ConstraintWorkCalculator,
    MDTrajectory,
    VibrationalSpectrumCalculator,
)


KB_EV_PER_K = 8.617333262145e-5


def _write_named_md_dump(path, atomic_numbers, **columns):
    """Write one regular-batch named trajectory for calculator tests."""
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    if atomic_numbers.ndim == 1:
        atomic_numbers = atomic_numbers[None, :]
    n_batch, n_atom = atomic_numbers.shape
    n_frame = len(next(iter(columns.values())))
    if any(len(values) != n_frame for values in columns.values()):
        raise ValueError('All synthetic dump columns must have the same frame count.')

    cell = np.zeros((n_batch, 3, 3), dtype=np.float32)
    fixed_mask = np.ones((n_batch, n_atom, 3), dtype=np.float32)
    dumper = ArrayDumper(path, mode='x')
    dumper.start_from_arrays(
        1, cell, atomic_numbers, fixed_mask,
        names=('cell_vec', 'atomic_numbers', 'fixed_mask'),
    )
    dumper.step(cell, atomic_numbers, fixed_mask)

    names = tuple(columns)
    prototypes = tuple(np.asarray(columns[name][0]) for name in names)
    dumper.start_from_arrays(n_frame, *prototypes, names=names)
    for frame in range(n_frame):
        dumper.step(*(np.asarray(columns[name][frame]) for name in names))
    dumper.close()


class PostprocessingTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _path(self, name):
        return f'{self.tempdir.name}/{name}'

    def test_vibrational_spectrum_recovers_known_modes(self):
        """FFT(VACF) peaks at the two imposed harmonic frequencies."""
        n_frame = 128
        time = np.arange(n_frame, dtype=np.float64)
        frequencies = (8. / n_frame, 16. / n_frame)
        velocities = np.zeros((n_frame, 2, 2, 3), dtype=np.float32)
        for image, frequency in enumerate(frequencies):
            signal = np.cos(2. * np.pi * frequency * time)
            velocities[:, image, 0, image] = signal
            velocities[:, image, 1, image] = -signal

        path = self._path('spectrum.bin')
        _write_named_md_dump(
            path,
            atomic_numbers=np.ones((2, 2), dtype=np.int64),
            V=velocities,
        )
        calculator = VibrationalSpectrumCalculator(path, sample_spacing_fs=1.)
        result = calculator.calculate(
            window=None,
            n_fft=256,
            max_lag=n_frame,
            detrend=True,
        )

        for image, expected_frequency in enumerate(frequencies):
            peak_index = 1 + np.argmax(result.spectrum[image, 1:])
            self.assertAlmostEqual(
                result.frequency_fs_inv[peak_index],
                expected_frequency,
                delta=result.frequency_fs_inv[1] - result.frequency_fs_inv[0],
            )
        self.assertTrue(np.allclose(result.vacf[:, 0], 1., atol=1e-12))

    def test_trajectory_loader_preserves_irregular_images(self):
        """Atom-wise irregular columns remain separated per real structure."""
        path = self._path('irregular.bin')
        batch_indices = np.asarray((1, 2), dtype=np.int64)
        cell = np.zeros((2, 3, 3), dtype=np.float32)
        atomic_numbers = np.asarray((1, 8, 1), dtype=np.int64)
        fixed_mask = np.ones((1, 3, 3), dtype=np.float32)
        velocity = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        force = np.asarray(((1.,), (2.,)), dtype=np.float32)

        dumper = ArrayDumper(path, mode='x')
        dumper.start_from_arrays(
            1, batch_indices, cell, atomic_numbers, fixed_mask,
            names=('batch_indices', 'cell_vec', 'atomic_numbers', 'fixed_mask'),
        )
        dumper.step(batch_indices, cell, atomic_numbers, fixed_mask)
        dumper.start_from_arrays(2, velocity, force, names=('V', 'Fc'))
        dumper.step(velocity, force)
        dumper.step(velocity + 10., force + 10.)
        dumper.close()

        trajectory = MDTrajectory(path)
        velocities = trajectory.by_image('V')
        self.assertEqual(trajectory.n_images, 2)
        self.assertEqual(trajectory.n_frames, 2)
        self.assertEqual(velocities[0].shape, (2, 1, 3))
        self.assertEqual(velocities[1].shape, (2, 2, 3))
        self.assertTrue(np.array_equal(trajectory.atomic_numbers[1], (8, 1)))

    def test_constraint_work_and_jarzynski_average(self):
        """Right-staged multipliers reproduce analytic work trajectories."""
        n_frame = 5
        n_batch = 3
        target = np.linspace(0., 1., n_frame, dtype=np.float32)
        coordinates = np.zeros((n_frame, n_batch, 1, 3), dtype=np.float32)
        coordinates[:, :, 0, 0] = target[:, None]
        forces = np.zeros((n_frame, n_batch, 1), dtype=np.float32)
        forces[1:, :, 0] = np.asarray((1., 2., 3.), dtype=np.float32)

        path = self._path('work.bin')
        _write_named_md_dump(
            path,
            atomic_numbers=np.ones((n_batch, 1), dtype=np.int64),
            X=coordinates,
            Fc=forces,
        )
        temperature = 300.
        result = ConstraintWorkCalculator(path).calculate(
            cv_func=lambda X: X[0, 0].reshape(1),
            temperature=temperature,
        )

        expected_work = np.asarray((1., 2., 3.))
        beta = 1. / (KB_EV_PER_K * temperature)
        expected_delta_f = -math.log(np.mean(np.exp(-beta * expected_work))) / beta
        self.assertTrue(np.allclose(result.work, expected_work, atol=1e-7))
        self.assertTrue(np.allclose(result.cumulative_work[-1], expected_work))
        self.assertAlmostEqual(result.jarzynski_free_energy, expected_delta_f, places=12)

    def test_blue_moon_integrates_batched_image_path(self):
        """Fixman-corrected image forces integrate along batch order."""
        n_frame = 20
        n_batch = 3
        temperature = 300.
        cv_targets = np.asarray(((0.,), (1.,), (2.,)), dtype=np.float64)
        constraint_forces = np.broadcast_to(
            np.asarray((0., 2., 4.), dtype=np.float32)[None, :, None],
            (n_frame, n_batch, 1),
        ).copy()
        geometric_force = np.full((n_frame, n_batch, 1), 0.5, dtype=np.float32)
        weights = np.broadcast_to(
            np.asarray((1., 2., 4.), dtype=np.float32)[None, :],
            (n_frame, n_batch),
        ).copy()

        path = self._path('blue_moon.bin')
        _write_named_md_dump(
            path,
            atomic_numbers=np.ones((n_batch, 1), dtype=np.int64),
            Fc=constraint_forces,
            G=geometric_force,
            w=weights,
        )
        trajectory = MDTrajectory(path)
        result = BlueMoonCalculator(trajectory).calculate(
            cv_values=cv_targets,
            temperature=temperature,
            block_size=5,
        )

        correction = KB_EV_PER_K * temperature * 0.5
        expected_force = np.asarray((0., 2., 4.)) + correction
        expected_free_energy = np.asarray((0., 1. + correction, 4. + 2. * correction))
        self.assertTrue(np.allclose(result.mean_force[:, 0], expected_force))
        self.assertTrue(np.allclose(result.free_energy, expected_free_energy))
        self.assertTrue(np.allclose(result.weight_sum, n_frame * np.asarray((1., 2., 4.))))
        self.assertTrue(np.allclose(result.mean_force_standard_error, 0., atol=1e-14))


if __name__ == '__main__':
    unittest.main()
