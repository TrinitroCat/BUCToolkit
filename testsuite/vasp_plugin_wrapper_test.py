"""Focused tests for the VASP Python-plugin wrapper lifecycle."""

from __future__ import annotations

import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch as th

from BUCToolkit.utils.model_wrappers import VASP_PluginModel
from BUCToolkit.utils.model_wrappers.VASP_plugin_wrapper import _PLUGIN_SOURCE


class VASPPluginWrapperTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="buc_vasp_plugin_")
        self._workdir = Path(self._temporary.name)
        for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
            (self._workdir / filename).write_text("", encoding="utf-8")
        repo_root = Path(__file__).resolve().parents[1]
        driver = textwrap.dedent(
            f"""
            import sys
            from dataclasses import dataclass
            import numpy as np
            sys.path.insert(0, {str(repo_root)!r})
            import vasp_plugin

            @dataclass(frozen=True)
            class Constants:
                positions: np.ndarray
                lattice_vectors: np.ndarray
                total_energy: float
                forces: np.ndarray
                stress: np.ndarray

            @dataclass
            class Additions:
                positions: np.ndarray
                lattice_vectors: np.ndarray

            positions = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float64)
            lattice = np.eye(3, dtype=np.float64)
            while True:
                additions = Additions(np.zeros_like(positions), np.zeros_like(lattice))
                position_pointer = additions.positions.__array_interface__["data"][0]
                constants = Constants(
                    positions=positions.copy(),
                    lattice_vectors=lattice.copy(),
                    total_energy=float(np.sum(positions ** 2)),
                    forces=-2. * positions,
                    stress=np.zeros((3, 3), dtype=np.float64),
                )
                try:
                    vasp_plugin.structure(constants, additions)
                except SystemExit:
                    break
                assert additions.positions.__array_interface__["data"][0] == position_pointer
                positions += additions.positions
                lattice += additions.lattice_vectors
            """
        )
        (self._workdir / "fake_vasp.py").write_text(driver, encoding="utf-8")
        submit = f"#!/bin/sh\nexec {sys.executable} fake_vasp.py\n"
        submit_path = self._workdir / "submit.sh"
        submit_path.write_text(submit, encoding="utf-8")
        submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self._temporary.cleanup()

    def test_generated_plugin_is_independent_of_buctoolkit_imports(self):
        self.assertNotIn("\nfrom BUCToolkit", _PLUGIN_SOURCE)
        self.assertNotIn("\nimport BUCToolkit", _PLUGIN_SOURCE)
        compile(_PLUGIN_SOURCE, "vasp_plugin.py", "exec")

    def test_long_lived_plugin_session_and_energy_force_cache(self):
        wrapper = VASP_PluginModel(
            str(self._workdir),
            "submit.sh",
            startup_timeout=5.0,
            evaluation_timeout=5.0,
        )
        try:
            coordinates = th.tensor(
                [[[0.25, 0.0, 0.0], [1.25, 0.0, 0.0]]],
                dtype=th.float64,
            )
            energy = wrapper.Energy(coordinates)
            gradient = wrapper.Grad(coordinates)
            self.assertTrue(wrapper._methods_replaced)
            self.assertEqual(tuple(energy.shape), (1,))
            self.assertTrue(th.allclose(energy, th.tensor([1.625], dtype=th.float64)))
            self.assertTrue(th.allclose(gradient, 2. * coordinates))

            second_coordinates = coordinates + 0.25
            second_energy = wrapper.Energy(second_coordinates)
            self.assertTrue(th.allclose(second_energy, th.tensor([2.75], dtype=th.float64)))
            self.assertEqual(wrapper._evaluation_id, 2)
        finally:
            wrapper.close()
            wrapper.close()
        self.assertIsNotNone(wrapper._job)
        self.assertIsNotNone(wrapper._job.returncode)

    def test_existing_user_plugin_is_not_overwritten(self):
        plugin_path = self._workdir / "vasp_plugin.py"
        plugin_path.write_text("def structure(constants, additions): pass\n", encoding="utf-8")
        wrapper = VASP_PluginModel(str(self._workdir), "submit.sh")
        with self.assertRaises(FileExistsError):
            wrapper.Energy(th.zeros((1, 2, 3), dtype=th.float64))
        wrapper.close()
        self.assertEqual(plugin_path.read_text(encoding="utf-8"), "def structure(constants, additions): pass\n")

    def test_nonzero_vasp_exit_before_handshake_is_reported(self):
        submit_path = self._workdir / "submit.sh"
        submit_path.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
        submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR)
        wrapper = VASP_PluginModel(
            str(self._workdir),
            "submit.sh",
            startup_timeout=5.0,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "status 7"):
                wrapper.Energy(th.zeros((1, 2, 3), dtype=th.float64))
        finally:
            wrapper.close()


if __name__ == "__main__":
    unittest.main()
