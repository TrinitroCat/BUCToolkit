"""Torch profiler wrapper for the fast central BUCToolkit test suite."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch as th

from fast_test import FastTest


PROFILE_ROOT = Path(__file__).resolve().parent


class ProfileTest(FastTest):
    """Run every fast central test and write one profiler report per method."""

    def _callTestMethod(self, method):
        """Profile one unittest method and persist CPU/CUDA timing tables.

        Args:
            method: Bound unittest method selected by the test loader.

        Return:
            None.

        Raises:
            Exception: The original test exception after its profile is saved.
        """
        activities = [th.profiler.ProfilerActivity.CPU]
        if th.cuda.is_available():
            activities.append(th.profiler.ProfilerActivity.CUDA)
        profile = th.profiler.profile(activities=activities, record_shapes=False)
        try:
            with profile:
                result = method()
        finally:
            report_path = PROFILE_ROOT / f"{method.__name__}.profile"
            with report_path.open("w", encoding="utf-8") as report:
                report.write(profile.key_averages().table(sort_by="self_cpu_time_total"))
                if th.cuda.is_available():
                    report.write("\n\nCUDA time\n")
                    try:
                        report.write(
                            profile.key_averages().table(sort_by="self_cuda_time_total")
                        )
                    except (KeyError, RuntimeError, ValueError):
                        report.write(
                            profile.key_averages().table(sort_by="self_device_time_total")
                        )
        return result


if __name__ == "__main__":
    unittest.main()
