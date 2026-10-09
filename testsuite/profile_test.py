"""Torch-profiler sections for the fast central BUCToolkit test suite."""

from __future__ import annotations

import gc
import re
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch as th

from fast_test import FastTest


PROFILE_ROOT = Path(__file__).resolve().parent


class ProfileTest(FastTest):
    """Run fast tests while profiling individual computational sections."""

    def setUp(self):
        """Prepare the fast-test fixture and reset profile-name counters.

        Return:
            None.
        """
        super().setUp()
        self._profile_paths_seen = set()

    @contextmanager
    def _profile_section(self, name: str | None):
        """Profile one calculation and immediately write its timing report.

        Args:
            name: Output stem for the section, or ``None`` to skip sampling.

        Return:
            A context manager that records CPU and CUDA operator totals.
        """
        if name is None:
            yield
            return

        activities = [th.profiler.ProfilerActivity.CPU]
        has_cuda = th.cuda.is_available()
        if has_cuda:
            activities.append(th.profiler.ProfilerActivity.CUDA)
        profile = th.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )
        try:
            with profile:
                yield
        finally:
            safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
            report_path = PROFILE_ROOT / f'{safe_name}.profile'
            try:
                try:
                    averages = profile.key_averages(
                        group_by_stack_n=5,
                        include_python_functions=True,
                    )
                except TypeError:
                    # Older Torch releases expose Python functions through
                    # ``with_stack`` but do not accept this key_averages flag.
                    averages = profile.key_averages(group_by_stack_n=5)
                cpu_table = averages.table(
                    sort_by='cpu_time_total',
                    row_limit=500,
                    max_src_column_width=200,
                    max_name_column_width=200,
                )
                try:
                    cuda_table = averages.table(
                        sort_by='cuda_time_total',
                        row_limit=500,
                        max_src_column_width=200,
                        max_name_column_width=200,
                    )
                except (KeyError, RuntimeError, ValueError):
                    cuda_table = 'CUDA profiling data is unavailable.\n'
                report_mode = 'a' if report_path in self._profile_paths_seen else 'w'
                self._profile_paths_seen.add(report_path)
                with report_path.open(report_mode, encoding='utf-8') as report:
                    if report_mode == 'a':
                        report.write('\n\n')
                    report.write(f'Section: {name}\n')
                    report.write(cpu_table)
                    report.write('\n\nCUDA time\n')
                    report.write(cuda_table)
            finally:
                del profile
                if 'averages' in locals():
                    del averages
                gc.collect()


if __name__ == '__main__':
    unittest.main()
