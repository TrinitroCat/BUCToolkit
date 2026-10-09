"""Analytic free-energy validation for constrained MD postprocessing."""

import tempfile
import unittest

import numpy as np
import torch as th

from BUCToolkit.BatchMD.ConstrNVT import ConstrNVT
from BUCToolkit.Postprocessing import BlueMoonCalculator, ConstraintWorkCalculator
from BUCToolkit.utils._Element_info import MASS


class ConstrainedFreeEnergyTest(unittest.TestCase):
    r"""Validate Blue-Moon and Jarzynski calculations on one analytic PES.

    The three-dimensional potential is

        U(q, y, z) = a (q^2 - 1)^2
                     + 1/2 k(q) (y^2 + z^2),
        k(q) = k0 + k2 (q^2 - 1)^2,

    and the collective variable is ``xi(X) = q = X[0, 0]``. Integrating the
    two transverse harmonic coordinates gives the conditional free energy

        A(q) = a (q^2 - 1)^2 + kB T log(k(q)) + constant.

    Therefore the exact barrier from the minimum at ``q=-1`` to the saddle at
    ``q=0`` is ``a + kB*T*log((k0+k2)/k0)``. The same barrier is calculated by
    equilibrium Blue-Moon windows and by nonequilibrium Jarzynski pulling.
    """

    KB_EV_PER_K = 8.617333262145e-5
    VELOCITY_ENERGY_CONVERSION = 103.642696562621738
    TEMPERATURE = 300.
    TIME_STEP = 0.2
    A = 0.15
    K0 = 0.3
    K2 = 0.3

    @staticmethod
    def collective_variable(X: th.Tensor) -> th.Tensor:
        """Return the constrained reaction coordinate for one structure."""
        return X[0, 0].reshape(1)

    @classmethod
    def transverse_force_constant(cls, q: th.Tensor) -> th.Tensor:
        """Coordinate-dependent transverse harmonic force constant."""
        return cls.K0 + cls.K2 * (q ** 2 - 1.) ** 2

    @classmethod
    def energy(cls, X: th.Tensor) -> th.Tensor:
        """Return one PES energy per regular-batch structure."""
        q = X[:, 0, 0]
        transverse = X[:, 0, 1:]
        return (
            cls.A * (q ** 2 - 1.) ** 2
            + 0.5 * cls.transverse_force_constant(q)
            * th.sum(transverse ** 2, dim=-1)
        )

    @classmethod
    def gradient(cls, X: th.Tensor, energy: th.Tensor) -> th.Tensor:
        """Return the analytic Cartesian gradient of :meth:`energy`."""
        q = X[:, 0, 0]
        transverse = X[:, 0, 1:]
        q2_minus_1 = q ** 2 - 1.
        force_constant = cls.transverse_force_constant(q)

        gradient = th.zeros_like(X)
        gradient[:, 0, 0] = (
            4. * cls.A * q * q2_minus_1
            + 2. * cls.K2 * q * q2_minus_1
            * th.sum(transverse ** 2, dim=-1)
        )
        gradient[:, 0, 1:] = force_constant[:, None] * transverse
        return gradient

    @classmethod
    def exact_barrier(cls) -> float:
        """Return ``A(0)-A(-1)`` in eV."""
        return (
            cls.A
            + cls.KB_EV_PER_K * cls.TEMPERATURE
            * np.log((cls.K0 + cls.K2) / cls.K0)
        )

    @classmethod
    def _thermal_velocity_scale(cls) -> float:
        """Return one Cartesian H-atom velocity standard deviation."""
        return float(np.sqrt(
            cls.KB_EV_PER_K * cls.TEMPERATURE
            / (MASS['H'] * cls.VELOCITY_ENERGY_CONVERSION)
        ))

    def _run_blue_moon(self, output_path: str) -> float:
        """Run fixed-CV windows and return the integrated barrier."""
        th.manual_seed(1234)
        n_image = 7
        n_step = 3000
        targets = th.linspace(-1., 0., n_image).reshape(n_image, 1)
        force_constants = self.transverse_force_constant(targets[:, 0])

        coordinates = th.zeros(n_image, 1, 3)
        coordinates[:, 0, 0] = targets[:, 0]
        coordinates[:, 0, 1:] = (
            th.randn(n_image, 2)
            * th.sqrt(
                th.tensor(self.KB_EV_PER_K * self.TEMPERATURE)
                / force_constants
            )[:, None]
        )
        velocities = th.zeros_like(coordinates)
        velocities[:, :, 1:] = (
            th.randn(n_image, 1, 2) * self._thermal_velocity_scale()
        )

        runner = ConstrNVT(
            self.TIME_STEP,
            n_step,
            'CSVR',
            {'time_const': 50},
            self.collective_variable,
            targets,
            1e-6,
            True,
            self.TEMPERATURE,
            output_path,
            2,
            device='cuda:0',
            verbose=2,
        )
        runner.run(
            self.energy,
            coordinates,
            [['H']] * n_image,
            V_init=velocities,
            grad_func=self.gradient,
            is_grad_func_contain_y=True,
        )

        result = BlueMoonCalculator(output_path).calculate(
            cv_values=targets.numpy(),
            temperature=self.TEMPERATURE,
            start=int(0.3 * n_step / 2),
            block_size=50,
        )
        self.assertTrue(np.all(np.isfinite(result.mean_force)))
        self.assertTrue(np.all(np.isfinite(result.free_energy)))
        return float(result.free_energy[-1])

    def _run_jarzynski(self, output_path: str) -> float:
        """Run parallel moving-CV trajectories and return the Jarzynski barrier."""
        th.manual_seed(2468)
        n_trajectory = 256
        n_step = 2000
        duration = (n_step - 1) * self.TIME_STEP
        pulling_rate = 1. / duration

        def moving_target(time_now: th.Tensor) -> th.Tensor:
            return th.ones(
                n_trajectory,
                device=time_now.device,
                dtype=time_now.dtype,
            ) * (-1. + time_now / duration)

        coordinates = th.zeros(n_trajectory, 1, 3)
        coordinates[:, 0, 0] = -1.
        coordinates[:, 0, 1:] = (
            th.randn(n_trajectory, 2)
            * np.sqrt(self.KB_EV_PER_K * self.TEMPERATURE / self.K0)
        )
        velocities = th.zeros_like(coordinates)
        velocities[:, 0, 0] = pulling_rate
        velocities[:, 0, 1:] = (
            th.randn(n_trajectory, 2) * self._thermal_velocity_scale()
        )

        runner = ConstrNVT(
            self.TIME_STEP,
            n_step,
            'Langevin',
            {'damping_coeff': 0.1},
            self.collective_variable,
            moving_target,
            1e-6,
            False,
            self.TEMPERATURE,
            output_path,
            1,
            device='cpu',
            verbose=1,
        )
        runner.run(
            self.energy,
            coordinates,
            [['H']] * n_trajectory,
            V_init=velocities,
            grad_func=self.gradient,
            is_grad_func_contain_y=True,
        )

        # Use the exact imposed protocol rather than the projected coordinates
        # so the work quadrature is not contaminated by residual tolerances.
        frame_time = np.arange(n_step, dtype=np.float64) * self.TIME_STEP
        protocol = np.broadcast_to(
            (-1. + frame_time / duration)[:, None, None],
            (n_step, n_trajectory, 1),
        )
        result = ConstraintWorkCalculator(output_path).calculate(
            cv_values=protocol,
            temperature=self.TEMPERATURE,
        )
        self.assertTrue(np.all(np.isfinite(result.work)))
        self.assertTrue(np.isfinite(result.jarzynski_free_energy))
        return float(result.jarzynski_free_energy)

    def test_known_barrier_with_blue_moon_and_jarzynski(self):
        """Both constrained-MD routes reproduce the same analytic barrier."""
        with tempfile.TemporaryDirectory() as tempdir:
            blue_moon_barrier = self._run_blue_moon(
                f'{tempdir}/blue_moon.bin'
            )
            jarzynski_barrier = self._run_jarzynski(
                f'{tempdir}/jarzynski.bin'
            )

        exact_barrier = self.exact_barrier()
        self.assertAlmostEqual(
            blue_moon_barrier,
            exact_barrier,
            delta=0.012,
            msg=(
                f'Blue-Moon barrier {blue_moon_barrier} eV does not match '
                f'the exact barrier {exact_barrier} eV.'
            ),
        )
        self.assertAlmostEqual(
            jarzynski_barrier,
            exact_barrier,
            delta=0.018,
            msg=(
                f'Jarzynski barrier {jarzynski_barrier} eV does not match '
                f'the exact barrier {exact_barrier} eV.'
            ),
        )
        self.assertAlmostEqual(
            blue_moon_barrier,
            jarzynski_barrier,
            delta=0.015,
            msg=(
                f'Blue-Moon and Jarzynski barriers differ: '
                f'{blue_moon_barrier} versus {jarzynski_barrier} eV.'
            ),
        )


if __name__ == '__main__':
    unittest.main()
