"""Fine-grained static Torch profiler tests for the central fast suite."""

from __future__ import annotations

import gc
import re
import unittest
from pathlib import Path

import torch as th

from fast_test import FastTest, PROFILE_KEYS


PROFILE_ROOT = Path.cwd() / 'profiles'
PROFILE_SKIP_KEYS = frozenset((
    # CMD contains JIT and CUDA-graph paths that are incompatible with
    # Torch profiler tree finalization on some Torch versions.
    'test_CMD.CMD_NVE_CPU', 'test_CMD.CMD_NVE_GPU',
    'test_CMD.CMD_CSVR_CPU', 'test_CMD.CMD_CSVR_GPU',
    'test_CMD.CMD_LANG_CPU', 'test_CMD.CMD_LANG_GPU',
    'test_CMD.CMD_NOSE_CPU', 'test_CMD.CMD_NOSE_GPU',
    'test_parallel.md_gpu_csvr_nvt.01',
    'test_parallel.md_gpu_csvr_nvt.17',
    'test_parallel.md_gpu_csvr_nvt.33',
    'test_parallel.md_gpu_csvr_nvt.49',
))


class StoredProfiler:
    """A profiler context that writes one report when its section ends."""

    def __init__(self, name: str, **profiler_kwargs):
        """Create one named profiler.

        Args:
            name: Stable section name used for the report filename.
            **profiler_kwargs: Arguments passed to ``torch.profiler.profile``.

        Return:
            None.
        """
        self.name = name
        self.has_cuda = any(
            activity == th.profiler.ProfilerActivity.CUDA
            for activity in profiler_kwargs.get('activities', ())
        )
        self.with_stack = bool(profiler_kwargs.get('with_stack', False))
        self.profile = th.profiler.profile(**profiler_kwargs)

    def __enter__(self):
        """Start this stored profiler section.

        Return:
            The underlying Torch profiler.
        """
        self.profile.__enter__()
        return self.profile

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the profiler and write CPU/CUDA reports.

        Args:
            exc_type: Exception class from the profiled section.
            exc_value: Exception value from the profiled section.
            traceback: Exception traceback from the profiled section.

        Return:
            ``False`` so the test exception propagates.
        """
        try:
            self.profile.__exit__(exc_type, exc_value, traceback)
            self._write_report()
        finally:
            # A stored section is entered at most once.  Drop Torch's event
            # buffer after the report so later sections do not retain it.
            self.profile = None
            gc.collect()
        return False

    def _write_report(self):
        """Write the bounded CPU and CUDA operator tables to disk.

        Return:
            None.
        """
        PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
        path = PROFILE_ROOT / f'{re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name)}.profile'
        try:
            with path.open('w', encoding='utf-8') as report:
                report.write(f'Section: {self.name}\n')
                report.write(f'with_stack: {self.with_stack}\n\n')
                try:
                    # Use Kineto's own aggregation so CPU and CUDA child
                    # events are attributed exactly once.
                    try:
                        averages = self.profile.key_averages(group_by_stack_n=5)
                    except TypeError:
                        averages = self.profile.key_averages()
                except TypeError:
                    averages = None
                    failure = 'TypeError: profiler aggregation is unavailable\n'
                    report.write(f'CPU time unavailable: {failure}')
                    if self.has_cuda:
                        report.write('\n\nCUDA time\n')
                        report.write(f'CUDA time unavailable: {failure}')
                except (RecursionError, UnicodeDecodeError):
                    # Stack grouping can recurse in some Torch releases or
                    # fail to decode a non-UTF-8 source file. Retry standard
                    # operator aggregation without stack grouping rather than
                    # inventing durations from the flat event list.
                    try:
                        averages = self.profile.key_averages(group_by_stack_n=0)
                    except Exception as exc:
                        averages = None
                        failure = f'{type(exc).__name__}: {exc}\n'
                        report.write(f'CPU time unavailable: {failure}')
                        if self.has_cuda:
                            report.write('\n\nCUDA time\n')
                            report.write(f'CUDA time unavailable: {failure}')
                except Exception as exc:
                    averages = None
                    failure = f'{type(exc).__name__}: {exc}\n'
                    report.write(f'CPU time unavailable: {failure}')
                    if self.has_cuda:
                        report.write('\n\nCUDA time\n')
                        report.write(f'CUDA time unavailable: {failure}')
                if averages is not None:
                    try:
                        report.write(averages.table(
                            sort_by='cpu_time_total',
                            row_limit=100,
                            max_src_column_width=200,
                            max_name_column_width=200,
                        ))
                    except Exception as exc:
                        report.write(
                            f'CPU profiling data is unavailable: '
                            f'{type(exc).__name__}: {exc}\n'
                        )
                    if self.has_cuda:
                        report.write('\n\nCUDA time\n')
                        try:
                            report.write(averages.table(
                                sort_by='cuda_time_total',
                                row_limit=100,
                                max_src_column_width=200,
                                max_name_column_width=200,
                            ))
                        except Exception as exc:
                            report.write(
                                f'CUDA profiling data is unavailable: '
                                f'{type(exc).__name__}: {exc}\n'
                            )
        finally:
            gc.collect()


class ProfileTest(FastTest):
    """Run the fast tests with one statically assigned profiler per section."""

    def setUp(self):
        """Create every profiler context before running the test body.

        Return:
            None.
        """
        super().setUp()
        experimental_config = th.profiler._ExperimentalConfig(
            profile_all_threads=True,
        )
        # FastTest already creates an entry for every static section, including
        # the unsampled parallel batch sizes.  Keep those EmptyProfiler entries
        # and replace only the sections listed in PROFILE_KEYS below.
        for name in PROFILE_KEYS:
            if name in PROFILE_SKIP_KEYS:
                continue
            activities = [th.profiler.ProfilerActivity.CPU]
            if (
                    th.cuda.is_available()
                    and any(token in name.lower() for token in ('gpu', 'cuda'))
            ):
                activities.append(th.profiler.ProfilerActivity.CUDA)
            self.profilers_dict[name] = StoredProfiler(
                name,
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                experimental_config=experimental_config,
            )
        for method in ('EnergyDiff', 'GradDiff', 'Autograd'):
            for device in ('cpu', 'cuda:0'):
                for dtype in (th.float32, th.float64):
                    name = f'test_VIB.{method}.{device}.{dtype}'
                    activities = [th.profiler.ProfilerActivity.CPU]
                    if th.cuda.is_available() and device.startswith('cuda'):
                        activities.append(th.profiler.ProfilerActivity.CUDA)
                    self.profilers_dict[name] = StoredProfiler(
                        name,
                        activities=activities,
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=True,
                        experimental_config=experimental_config,
                    )
        for n_atom in (7, 8):
            name = f'test_VIB.different_shapes.{n_atom}'
            self.profilers_dict[name] = StoredProfiler(
                name,
                activities=[th.profiler.ProfilerActivity.CPU],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                experimental_config=experimental_config,
            )


if __name__ == '__main__':
    unittest.main()
