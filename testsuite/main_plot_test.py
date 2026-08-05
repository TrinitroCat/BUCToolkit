#  Copyright (c) 2026.7.30, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  File: main_test_plot.py
#  Environment: Python 3.12
"""Paper-style trajectory plots for the MD, constrained MD, and MC tests.

The simulation and numerical assertions are inherited unchanged from
``main_test.MainTest``.  This module only observes each trajectory as the
parent test reads it, then writes publication-ready figures to ``plots``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th

import main_test as main_test_module
from _toy_harmonic_potential import build_cubic_lattice_data
from main_test import MainTest


READ_MD_TRAJ = main_test_module.read_md_traj
READ_MC_TRAJ = main_test_module.read_mc_traj
BUILD_CUBIC_LATTICE_DATA = build_cubic_lattice_data

KB_EV_PER_K = 8.617333262145e-5
VELOCITY_TO_EV = 103.642696562621738
POTENTIAL_COLOR = "#4C78A8"
KINETIC_COLOR = "#F28E2B"
TOTAL_COLOR = "#59A14F"
TEMPERATURE_COLOR = "#E15759"
CONSTRAINT_COLORS = (
    "#4C78A8", "#F28E2B", "#59A14F", "#E15759",
    "#B07AA1", "#76B7B2", "#EDC948", "#9C755F",
)


class MainTestPlot(MainTest):
    """Run the existing motion tests and save their trajectories as figures."""

    def setUp(self):
        """Set up the parent test data and a non-destructive figure directory."""
        super().setUp()
        self.plot_dir = os.path.join(self.out_pt, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
        self._cmd_initial_positions: list[np.ndarray] = []

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        """Convert a trajectory value to a detached NumPy array."""
        if isinstance(value, th.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _series(values, sample_index: int, n_samples: int) -> np.ndarray:
        """Extract one sample from cycle-major batch trajectory values."""
        return np.asarray(values[sample_index::n_samples], dtype=float)

    @staticmethod
    def _energy_scale(values: np.ndarray) -> float:
        """Return a robust non-zero energy scale for axis selection."""
        finite_values = np.abs(values[np.isfinite(values)])
        if finite_values.size == 0:
            return 0.0
        return float(np.percentile(finite_values, 95.0))

    @classmethod
    def _require_separate_energy_axes(
        cls,
        potential: np.ndarray,
        kinetic: np.ndarray,
        total: np.ndarray,
    ) -> bool:
        """Decide whether the three energy curves need independent axes."""
        scales = [cls._energy_scale(values) for values in (potential, kinetic, total)]
        positive_scales = [scale for scale in scales if scale > 1e-14]
        if len(positive_scales) < 2:
            return False
        return max(positive_scales) / min(positive_scales) > 30.0

    @staticmethod
    def _style_main_axis(axis) -> None:
        """Apply the shared four-sided, paper-style axes treatment."""
        for spine in axis.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color("#202020")
        axis.tick_params(
            axis="both",
            labelsize=16,
            width=1.6,
            length=6,
            color="#202020",
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, linestyle="--")
        axis.set_axisbelow(True)

    @staticmethod
    def _style_colored_axis(axis, side: str, color: str) -> None:
        """Style an auxiliary y-axis using its corresponding curve color."""
        axis.tick_params(axis="y", labelsize=16, width=1.6, length=6, colors=color)
        axis.yaxis.label.set_color(color)
        axis.spines[side].set_linewidth(2.0)
        axis.spines[side].set_color(color)

    @staticmethod
    def _add_figure_legend(figure, handles, labels) -> None:
        """Place a multi-column legend in the figure's reserved top margin."""
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.988),
            ncol=min(3, len(handles)),
            fontsize=16,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.2,
        )

    def _save_figure(self, figure, stem: str) -> None:
        """Save both vector and high-resolution raster versions of a figure."""
        base_path = os.path.join(self.plot_dir, stem)
        figure.savefig(f"{base_path}.pdf", bbox_inches="tight")
        figure.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _make_energy_references(
        runner_name: str,
        potential: np.ndarray,
        kinetic: np.ndarray,
        total: np.ndarray,
        potential_expectation: float,
        kinetic_expectation: float,
    ) -> list[tuple[str, float, str]]:
        """Build reference lines from the same standards used by the tests."""
        if "_NVE_" in runner_name:
            return [("NVE total-energy reference", float(total[0]), "total")]
        if "STATIC" in runner_name:
            return [("Static reference", 0.0, "potential")]
        return [
            ("NVT virial potential expectation", potential_expectation, "potential"),
            ("NVT kinetic expectation", kinetic_expectation, "kinetic"),
            (
                "NVT total-energy expectation",
                potential_expectation + kinetic_expectation,
                "total",
            ),
        ]

    def _plot_energy_panels(
        self,
        runner_name: str,
        times: np.ndarray,
        potential_series: list[np.ndarray],
        kinetic_series: list[np.ndarray],
        total_series: list[np.ndarray],
        temperature_series: list[np.ndarray],
        references: list[list[tuple[str, float, str]]],
        title_prefix: str,
    ) -> None:
        """Draw batch-resolved potential, kinetic, total-energy, and temperature curves."""
        n_samples = len(potential_series)
        use_separate_axes = any(
            self._require_separate_energy_axes(potential, kinetic, total)
            for potential, kinetic, total in zip(
                potential_series, kinetic_series, total_series
            )
        )
        figure, axes = plt.subplots(
            n_samples,
            1,
            figsize=(16.5, 4.8 * n_samples + 1.5),
            squeeze=False,
        )
        legend_handles = []
        legend_labels = []

        for sample_index, main_axis in enumerate(axes[:, 0]):
            potential = potential_series[sample_index]
            kinetic = kinetic_series[sample_index]
            total = total_series[sample_index]
            temperature = temperature_series[sample_index]
            self._style_main_axis(main_axis)
            main_axis.set_title(
                f"{title_prefix}: {runner_name}, sample {sample_index + 1}",
                fontsize=18,
                pad=12,
            )
            main_axis.set_xlabel("Simulation time (fs)", fontsize=18)

            if use_separate_axes:
                kinetic_axis = main_axis.twinx()
                kinetic_axis.spines["left"].set_position(("axes", -0.20))
                kinetic_axis.spines["left"].set_visible(True)
                kinetic_axis.spines["right"].set_visible(False)
                kinetic_axis.yaxis.set_label_position("left")
                kinetic_axis.yaxis.tick_left()
                total_axis = main_axis.twinx()
                temperature_axis = main_axis.twinx()
                temperature_axis.spines["right"].set_position(("axes", 1.20))
                self._style_colored_axis(main_axis, "left", POTENTIAL_COLOR)
                self._style_colored_axis(kinetic_axis, "left", KINETIC_COLOR)
                self._style_colored_axis(total_axis, "right", TOTAL_COLOR)
                self._style_colored_axis(temperature_axis, "right", TEMPERATURE_COLOR)
                main_axis.set_ylabel("Potential energy (eV)", fontsize=18, color=POTENTIAL_COLOR)
                kinetic_axis.set_ylabel("Kinetic energy (eV)", fontsize=18, color=KINETIC_COLOR)
                total_axis.set_ylabel("Total energy (eV)", fontsize=18, color=TOTAL_COLOR)
                temperature_axis.set_ylabel("Temperature (K)", fontsize=18, color=TEMPERATURE_COLOR)
                energy_axes = {
                    "potential": main_axis,
                    "kinetic": kinetic_axis,
                    "total": total_axis,
                }
                lines = [
                    main_axis.plot(times, potential, color=POTENTIAL_COLOR, linewidth=2.0,
                                   label="Potential energy")[0],
                    kinetic_axis.plot(times, kinetic, color=KINETIC_COLOR, linewidth=2.0,
                                      label="Kinetic energy")[0],
                    total_axis.plot(times, total, color=TOTAL_COLOR, linewidth=2.0,
                                    label="Total energy")[0],
                    temperature_axis.plot(times, temperature, color=TEMPERATURE_COLOR, linewidth=2.0,
                                          label="Temperature")[0],
                ]
            else:
                temperature_axis = main_axis.twinx()
                self._style_colored_axis(temperature_axis, "right", TEMPERATURE_COLOR)
                main_axis.set_ylabel("Energy (eV)", fontsize=18)
                temperature_axis.set_ylabel("Temperature (K)", fontsize=18, color=TEMPERATURE_COLOR)
                energy_axes = {
                    "potential": main_axis,
                    "kinetic": main_axis,
                    "total": main_axis,
                }
                lines = [
                    main_axis.plot(times, potential, color=POTENTIAL_COLOR, linewidth=2.0,
                                   label="Potential energy")[0],
                    main_axis.plot(times, kinetic, color=KINETIC_COLOR, linewidth=2.0,
                                   label="Kinetic energy")[0],
                    main_axis.plot(times, total, color=TOTAL_COLOR, linewidth=2.0,
                                   label="Total energy")[0],
                    temperature_axis.plot(times, temperature, color=TEMPERATURE_COLOR, linewidth=2.0,
                                          label="Temperature")[0],
                ]

            reference_lines = []
            for label, value, quantity in references[sample_index]:
                reference_lines.append(energy_axes[quantity].axhline(
                    value,
                    color="#000000",
                    linewidth=1.8,
                    linestyle=(0, (6, 3)),
                    label=label,
                ))
            if sample_index == 0:
                legend_handles.extend(lines + reference_lines)
                legend_labels.extend([line.get_label() for line in lines + reference_lines])

        self._add_figure_legend(figure, legend_handles, legend_labels)
        if use_separate_axes:
            figure.subplots_adjust(left=0.23, right=0.77, top=0.84, bottom=0.08, hspace=0.42)
        else:
            figure.subplots_adjust(left=0.13, right=0.87, top=0.84, bottom=0.08, hspace=0.42)
        self._save_figure(figure, f"{runner_name}_energy")

    def _plot_md_trajectory(self, trajectory, runner_name: str) -> None:
        """Plot one MD runner using its existing three-sample trajectory."""
        n_samples = len(self.N)
        output_interval = 1 if "STATIC" in runner_name or "_NVE_" in runner_name else 10
        times = np.arange(len(trajectory.Energies) // n_samples) * 1.5 * output_interval
        potential_series = []
        kinetic_series = []
        total_series = []
        temperature_series = []
        references = []
        for sample_index in range(n_samples):
            potential = self._series(trajectory.Energies, sample_index, n_samples)
            velocities = self._to_numpy(trajectory.Labels[sample_index::n_samples])
            mass = np.asarray(self.masses_list[sample_index], dtype=float)
            kinetic = np.sum(
                0.5 * mass[None, :, None] * velocities * velocities * VELOCITY_TO_EV,
                axis=(1, 2),
            )
            total = potential + kinetic
            temperature = 2.0 * kinetic / (self.DOF_vib[sample_index] * KB_EV_PER_K)
            potential_series.append(potential)
            kinetic_series.append(kinetic)
            total_series.append(total)
            temperature_series.append(temperature)
            potential_expectation = 0.5 * self.DOF_vib[sample_index] * KB_EV_PER_K * 500.0
            kinetic_expectation = 1.5 * (self.N[sample_index] - 3) * KB_EV_PER_K * 500.0
            references.append(self._make_energy_references(
                runner_name,
                potential,
                kinetic,
                total,
                potential_expectation,
                kinetic_expectation,
            ))
        self._plot_energy_panels(
            runner_name,
            times,
            potential_series,
            kinetic_series,
            total_series,
            temperature_series,
            references,
            "MD trajectory",
        )

    @staticmethod
    def _cmd_constraint_values(positions: np.ndarray) -> np.ndarray:
        """Evaluate the eight constraints from the parent CMD test."""
        coordinates = th.as_tensor(positions, dtype=th.float64)
        values = [
            th.linalg.norm(coordinates[[2, 3, 5]] - coordinates[[4, 7, 8]], dim=-1),
        ]
        x1 = coordinates[[5, 9]]
        x2 = coordinates[[7, 11]]
        x3 = coordinates[[8, 12]]
        values.append(th.sum((x2 - x1) * (x3 - x1), dim=-1) / (
            th.linalg.norm(x2 - x1, dim=-1) * th.linalg.norm(x3 - x1, dim=-1)
        ))
        radius = th.linalg.norm(
            coordinates[[14, 18]].unsqueeze(1) - coordinates.unsqueeze(0),
            dim=-1,
        )
        values.append(th.sum(0.5 * (1.0 + th.erf((radius - 1.5) / 1.0)), dim=-1))
        pair_distances = th.linalg.norm(
            coordinates.unsqueeze(0) - coordinates.unsqueeze(1),
            dim=-1,
        )
        values.append(th.std(pair_distances, unbiased=True).unsqueeze(0))
        return th.cat(values).cpu().numpy()

    def _plot_cmd_constraint_violations(self, trajectory, runner_name: str) -> None:
        """Plot each CMD sample's eight absolute constraint violations over time."""
        constraint_labels = (
            "d(2,4)", "d(3,7)", "d(5,8)", "cos(7-5-8)",
            "cos(11-9-12)", "CN(14)", "CN(18)", "R_std",
        )
        n_samples = len(self._cmd_initial_positions)
        if n_samples == 0:
            return
        n_cycles = len(trajectory.Coords) // n_samples
        times = np.arange(n_cycles) * 0.5 * 10
        figure, axes = plt.subplots(
            n_samples,
            1,
            figsize=(16.5, 4.8 * n_samples + 1.8),
            squeeze=False,
        )
        legend_handles = []
        legend_labels = []
        for sample_index, axis in enumerate(axes[:, 0]):
            self._style_main_axis(axis)
            target = self._cmd_constraint_values(self._cmd_initial_positions[sample_index])
            positions = trajectory.Coords[sample_index::n_samples]
            violations = np.asarray([
                np.abs(self._cmd_constraint_values(position) - target)
                for position in positions
            ])
            positive_violations = violations[
            np.isfinite(violations) & (violations > 0.0)
            ]
            violation_floor = 1e-10

            if positive_violations.size > 0:
                violation_floor = max(violation_floor, float(np.min(positive_violations)) * 0.1)
            axis.set_title(
                f"CMD constraint violations: {runner_name}, sample {sample_index + 1}",
                fontsize=18,
                pad=12,
            )
            axis.set_xlabel("Simulation time (fs)", fontsize=18)
            axis.set_ylabel("Absolute constraint violation", fontsize=18)
            axis.set_yscale("log")
            lines = []
            for constraint_index, label in enumerate(constraint_labels):
                lines.append(axis.plot(
                    times,
                    np.maximum(violations[:, constraint_index], violation_floor),
                    color=CONSTRAINT_COLORS[constraint_index],
                    linewidth=1.7,
                    label=label,
                )[0])
            tolerance_line = axis.axhline(
                1e-5,
                color="#000000",
                linewidth=1.8,
                linestyle=(0, (6, 3)),
                label="Constraint tolerance",
            )
            if sample_index == 0:
                legend_handles.extend(lines + [tolerance_line])
                legend_labels.extend([line.get_label() for line in lines + [tolerance_line]])
        self._add_figure_legend(figure, legend_handles, legend_labels)
        figure.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.08, hspace=0.42)
        self._save_figure(figure, f"{runner_name}_constraint_violations")

    def _plot_cmd_trajectory(self, trajectory, runner_name: str) -> None:
        """Plot CMD energies and the separate batch-resolved constraint figure."""
        n_samples = len(self._cmd_initial_positions)
        if n_samples == 0:
            return
        times = np.arange(len(trajectory.Energies) // n_samples) * 0.5 * 10
        potential_series = []
        kinetic_series = []
        total_series = []
        temperature_series = []
        references = []
        dof = 3 * 5 ** 3 - 8
        mass = np.full(5 ** 3, self.MASSES[1] if len(self.MASSES) > 1 else 26.9815385)
        for sample_index in range(n_samples):
            potential = self._series(trajectory.Energies, sample_index, n_samples)
            velocities = self._to_numpy(trajectory.Labels[sample_index::n_samples])
            kinetic = np.sum(
                0.5 * mass[None, :, None] * velocities * velocities * VELOCITY_TO_EV,
                axis=(1, 2),
            )
            total = potential + kinetic
            temperature = 2.0 * kinetic / (dof * KB_EV_PER_K)
            potential_series.append(potential)
            kinetic_series.append(kinetic)
            total_series.append(total)
            temperature_series.append(temperature)
            expectation = 0.5 * dof * KB_EV_PER_K * 500.0
            references.append(self._make_energy_references(
                runner_name,
                potential,
                kinetic,
                total,
                expectation,
                expectation,
            ))
        self._plot_energy_panels(
            runner_name,
            times,
            potential_series,
            kinetic_series,
            total_series,
            temperature_series,
            references,
            "CMD trajectory",
        )
        self._plot_cmd_constraint_violations(trajectory, runner_name)

    def _plot_mc_trajectory(self, trajectory, runner_name: str) -> None:
        """Plot the MC potential-energy trajectories as batch-resolved panels."""
        n_samples = len(self.N)
        n_cycles = len(trajectory.Energies) // n_samples
        steps = np.arange(n_cycles) * 10
        figure, axes = plt.subplots(
            n_samples,
            1,
            figsize=(15.0, 4.5 * n_samples + 1.5),
            squeeze=False,
        )
        legend_handles = []
        legend_labels = []
        for sample_index, axis in enumerate(axes[:, 0]):
            self._style_main_axis(axis)
            potential = self._series(trajectory.Energies, sample_index, n_samples)
            axis.set_title(
                f"MC trajectory: {runner_name}, sample {sample_index + 1}",
                fontsize=18,
                pad=12,
            )
            axis.set_xlabel("Monte Carlo step", fontsize=18)
            axis.set_ylabel("Potential energy (eV)", fontsize=18)
            potential_line = axis.plot(
                steps,
                potential,
                color=POTENTIAL_COLOR,
                linewidth=2.0,
                label="Potential energy",
            )[0]
            if "ANNEAL" in runner_name:
                reference_label = "Annealing minimum potential"
                reference_value = float(np.min(potential))
            else:
                reference_label = "NVT virial expectation"
                reference_value = 0.5 * self.DOF_vib[sample_index] * KB_EV_PER_K * 500.0
            reference_line = axis.axhline(
                reference_value,
                color="#000000",
                linewidth=1.8,
                linestyle=(0, (6, 3)),
                label=reference_label,
            )
            if sample_index == 0:
                legend_handles.extend([potential_line, reference_line])
                legend_labels.extend([potential_line.get_label(), reference_line.get_label()])
        self._add_figure_legend(figure, legend_handles, legend_labels)
        figure.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.08, hspace=0.42)
        self._save_figure(figure, f"{runner_name}_potential")

    def _read_md_traj_with_plots(self, path, *args, **kwargs):
        """Read an MD trajectory and draw the matching requested figure."""
        trajectory = READ_MD_TRAJ(path, *args, **kwargs)
        runner_name = Path(path).name
        if runner_name.startswith("MD_"):
            self._plot_md_trajectory(trajectory, runner_name)
        elif runner_name.startswith("CMD_") and runner_name != "CMD_MANIFOLD_CPU":
            self._plot_cmd_trajectory(trajectory, runner_name)
        return trajectory

    def _read_mc_traj_with_plots(self, path, *args, **kwargs):
        """Read an MC trajectory and draw the matching requested figure."""
        trajectory = READ_MC_TRAJ(path, *args, **kwargs)
        self._plot_mc_trajectory(trajectory, Path(path).name)
        return trajectory

    def _capture_cmd_initial_structure(self, *args, **kwargs):
        """Record CMD initial positions so violations use each sample's true target."""
        structure = BUILD_CUBIC_LATTICE_DATA(*args, **kwargs)
        self._cmd_initial_positions.append(self._to_numpy(structure.pos).copy())
        return structure

    def test_MD(self):
        """Run the parent MD checks while saving the MD trajectory figures."""
        with patch.object(
            main_test_module,
            "read_md_traj",
            side_effect=self._read_md_traj_with_plots,
        ):
            super().test_MD()

    def test_CMD(self):
        """Run the parent CMD checks while saving energy and violation figures."""
        with patch.object(
            main_test_module,
            "build_cubic_lattice_data",
            side_effect=self._capture_cmd_initial_structure,
        ), patch.object(
            main_test_module,
            "read_md_traj",
            side_effect=self._read_md_traj_with_plots,
        ):
            super().test_CMD()

    def test_MC(self):
        """Run the parent MC checks while saving the potential-energy figures."""
        with patch.object(
            main_test_module,
            "read_mc_traj",
            side_effect=self._read_mc_traj_with_plots,
        ):
            super().test_MC()


if __name__ == "__main__":
    import unittest

    unittest.main()