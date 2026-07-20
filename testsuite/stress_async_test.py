#!/usr/bin/env python3
"""
Stress-test for async dump / logout handshake in _BaseOpt and _BaseMD.

Injects artificial ``time.sleep`` delays at critical handshake points so
that even narrow race windows become deterministic.  Each test runs many
times with randomised delays and compares dump-file data against the
in-memory log record — every mismatch is a handshake bug.
"""

import math
import os
import queue
import random
import tempfile
import time
import unittest
import warnings

import numpy as np
import torch as th

from BUCToolkit.BatchOptim.minimize.FIRE import FIRE
from BUCToolkit.BatchStructures.StructuresIO import read_opt_structures, read_md_traj
from BUCToolkit.utils._print_formatter import SCIENTIFIC_ARRAY_FORMAT, STRING_ARRAY_FORMAT

# ---------------------------------------------------------------------------
# Toy model – 3-D harmonic well (CPU-friendly, no autograd overhead)
# ---------------------------------------------------------------------------

def _harmonic_energy(X: th.Tensor, X0: th.Tensor) -> th.Tensor:
    """E = 0.5 Σ (X - X0)²  →  shape (n_batch,)."""
    return 0.5 * th.sum((X - X0) ** 2, dim=(-2, -1))


def _harmonic_grad(X: th.Tensor, y: th.Tensor, X0: th.Tensor) -> th.Tensor:
    """dE/dX = X - X0."""
    return X - X0


# ═══════════════════════════════════════════════════════════════════════════
# Debug / stress subclasses
# ═══════════════════════════════════════════════════════════════════════════

class StressOpt(FIRE):
    """FIRE whose async consumers sleep at handshake points.

    Parameters
    ----------
    dump_sleep : float
        Seconds to sleep inside ``_do_async_dump`` after dequeuing.
    print_sleep : float
        Seconds to sleep inside ``_do_async_print`` after dequeuing.
    """

    def __init__(self, dump_sleep: float = 0.02, print_sleep: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self._stress_dump_sleep = dump_sleep
        self._stress_print_sleep = print_sleep
        # Populated by _do_async_print – read after run() returns.
        self._stress_log_energies: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Override consumer threads — identical logic + injected sleep
    # ------------------------------------------------------------------

    def _do_async_dump(self, q: queue.Queue):
        while True:
            items = q.get()
            if items is None:
                break
            if self._stress_dump_sleep > 0:
                time.sleep(self._stress_dump_sleep)
            try:
                dumper, sync, *rest = items
                if sync is not None:
                    sync.synchronize()
                    rest = [t.numpy() for t in rest]
                dumper.step(*rest)
            except Exception as e:
                self.logger.error(f"Error: Failed to dump data due to \"{e}\"")
            finally:
                self._dump_done.set()

    def _do_async_print(self, q: queue.Queue):
        numit = 0
        log_energies: list[np.ndarray] = []
        while True:
            items = q.get()
            if items is None:
                break
            if self._stress_print_sleep > 0:
                time.sleep(self._stress_print_sleep)
            try:
                # Queue format: (event, numit, converge_str, t_st, *log_vals)
                sync, numit, converge_str, t_st = items[0], items[1], items[2], items[3]
                if sync is not None:
                    sync.synchronize()
                _data = dict(zip(self.get_log_vars(), items[4:]))
                _ep = _data.get('Energy')
                if _ep is not None:
                    log_energies.append(_ep.clone().numpy(force=True))
            except Exception as e:
                self.logger.error(
                    f"Error: Failed to logout at {numit}-th iteration due to \"{e}\"."
                )
            finally:
                self._print_done.set()
        self._stress_log_energies = log_energies


# ---------------------------------------------------------------------------
# MD stress – CUDA path exercised; consumers mirror _BaseMD logic + sleep
# ---------------------------------------------------------------------------

class StressNVE:
    """Factory — only importable when CUDA is present (avoids import errors).

    The real class is built inside :func:`_make_stress_nve` so that the
    module can be imported on CPU-only machines.
    """

    pass  # placeholder


def _make_stress_nve():
    """Return a StressNVE class bound to the current torch env."""
    from BUCToolkit.BatchMD.NVE import NVE
    from BUCToolkit.BatchMD._BaseMD import _BaseMD

    class _StressNVE(NVE):
        def __init__(self, dump_sleep=0.02, print_sleep=0.01, **kwargs):
            super().__init__(**kwargs)
            self._stress_dump_sleep = dump_sleep
            self._stress_print_sleep = print_sleep
            self._stress_log_ep: list[np.ndarray] = []

        def _do_async_dump(self, q: queue.Queue):
            while True:
                items = q.get()
                dumper, event = items[0], items[1]
                if dumper is None:
                    break
                if self._stress_dump_sleep > 0:
                    time.sleep(self._stress_dump_sleep)
                try:
                    event.synchronize()
                    dumper.step(*(t.numpy() for t in items[2:]))
                except Exception as e:
                    self.logger.error(f"Error: Failed to dump data due to \"{e}\"")
                finally:
                    self._dump_done.set()

        def _do_async_print(self, q: queue.Queue):
            formatter1 = {'float': '{:> .2f}'.format}
            i = 0
            log_ep: list[np.ndarray] = []
            while True:
                items = q.get()
                if items[0] is None:
                    break
                i, copy_event, batch_indices = items[0], items[1], items[2]
                if self._stress_print_sleep > 0:
                    time.sleep(self._stress_print_sleep)
                try:
                    copy_event.synchronize()
                    if self.verbose > 0:
                        _print_Ep, _print_Ek, _print_temperature, _print_X, _print_V = items[3:]
                        log_ep.append(_print_Ep.numpy(force=True).copy())
                        np.set_printoptions(
                            precision=8, linewidth=1024, floatmode='fixed',
                            suppress=True, formatter=formatter1, threshold=2000,
                        )
                        self.logger.info(
                            f'Step: {i:>12d}\n\t'
                            f'T     = {_print_temperature.numpy()}\n\t'
                            f'E_tol = {np.array2string((_print_Ek + _print_Ep).numpy(), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                            f'Ek    = {np.array2string(_print_Ek.numpy(), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                            f'Ep    = {np.array2string(_print_Ep.numpy(), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                        )
                        if self.verbose > 1:
                            self.handle_arrays_print(
                                self.logger, batch_indices, self.batch_slice_indx,
                                [[_print_X], [_print_V]],
                                [['Coordinates'], ['Forces']],
                                verbose=self.verbose, force=False,
                            )
                except Exception as e:
                    self.logger.error(
                        f"Error: Failed to logout at {i}-th iteration due to \"{e}\"."
                    )
                finally:
                    self._print_done.set()
            self._stress_log_ep = log_ep

    return _StressNVE



# Test cases

class TestStressAsyncOpt(unittest.TestCase):
    """Stress the _BaseOpt async handshake on CPU."""

    N_RUNS = 40          # independent runs per test
    MAX_STEPS = 30        # small problem → fast turnaround

    def setUp(self):
        self.X0 = th.tensor([[[0.5, 0.3, 0.1]]], dtype=th.float32)
        self.X_init = self.X0 + th.randn(1, 1, 3) * 0.5
        self.out_dir = tempfile.mkdtemp(prefix='stress_async_opt_')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.out_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_runner(self, out_file: str, **kwargs) -> StressOpt:
        defaults = dict(
            E_threshold=1e-5, F_threshold=0.01, maxiter=self.MAX_STEPS,
            steplength=0.1, device='cpu', verbose=1,
            output_file=out_file,
        )
        defaults.update(kwargs)
        runner = StressOpt(**defaults)
        runner._HOLD_DUMPER = False
        return runner

    def _run_and_compare(self, runner: StressOpt) -> tuple[int, int]:
        """Run one optimisation; return (n_steps, n_mismatch)."""
        X = self.X_init.clone()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            runner.run(
                _harmonic_energy, X,
                grad_func=_harmonic_grad,
                grad_func_args=(self.X0,),
                func_args=(self.X0,),
                is_grad_func_contain_y=True,
                batch_indices=None,
            )

        # Read binary dump
        bs = read_opt_structures(runner.output_file, only_opt=False)
        dump_E = np.asarray(bs.Energies).ravel()                # (n_steps * n_batch,)

        # log records one array per iteration, each shape (n_batch,)
        log_E = np.array(runner._stress_log_energies).ravel()   # (n_log_steps * n_batch,)
        # Final frame may have been added by the after-loop dump
        min_len = min(len(dump_E), len(log_E))
        dump_E = dump_E[:min_len]
        log_E = log_E[:min_len]

        n_mismatch = int((~np.isclose(dump_E, log_E, atol=1e-6)).sum())
        return min_len, n_mismatch

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_nonhold_random_delays(self):
        """Non-hold mode with random consumer delays – dump must match log."""
        failures: list[tuple[int, float, float, int, int]] = []
        for run_i in range(self.N_RUNS):
            out_file = os.path.join(self.out_dir, f'opt_{run_i}.bin')
            dump_sleep = random.uniform(0.001, 0.040)
            print_sleep = random.uniform(0.001, 0.030)

            runner = self._make_runner(out_file,
                                       dump_sleep=dump_sleep,
                                       print_sleep=print_sleep)
            n_steps, n_mm = self._run_and_compare(runner)
            if n_mm:
                failures.append((run_i, dump_sleep, print_sleep, n_steps, n_mm))

        self.assertEqual(
            len(failures), 0,
            f"{len(failures)}/{self.N_RUNS} runs had dump≠log "
            f"(first 5: {failures[:5]})"
        )

    def test_nonhold_max_delay(self):
        """Worst-case: consumer gets all the time in the world."""
        out_file = os.path.join(self.out_dir, 'opt_max.bin')
        runner = self._make_runner(out_file,
                                   dump_sleep=0.10,
                                   print_sleep=0.08)
        n_steps, n_mm = self._run_and_compare(runner)
        self.assertEqual(n_mm, 0,
                         f"Max-delay run: {n_mm} mismatches in {n_steps} steps")

    def test_hold_random_delays(self):
        """_hold_samples mode — the pre-mutation gate is tested."""
        failures: list[tuple[int, float, float, int, int]] = []
        for run_i in range(self.N_RUNS):
            out_file = os.path.join(self.out_dir, f'opt_hold_{run_i}.bin')
            dump_sleep = random.uniform(0.001, 0.040)
            print_sleep = random.uniform(0.001, 0.030)

            runner = self._make_runner(out_file,
                                       _hold_samples=True,
                                       dump_sleep=dump_sleep,
                                       print_sleep=print_sleep)
            n_steps, n_mm = self._run_and_compare(runner)
            if n_mm:
                failures.append((run_i, dump_sleep, print_sleep, n_steps, n_mm))

        self.assertEqual(
            len(failures), 0,
            f"_hold_samples: {len(failures)}/{self.N_RUNS} runs had dump≠log "
            f"(first 5: {failures[:5]})"
        )

    def test_hold_max_delay(self):
        """_hold_samples + max consumer delay."""
        out_file = os.path.join(self.out_dir, 'opt_hold_max.bin')
        runner = self._make_runner(out_file,
                                   _hold_samples=True,
                                   dump_sleep=0.10,
                                   print_sleep=0.08)
        n_steps, n_mm = self._run_and_compare(runner)
        self.assertEqual(n_mm, 0,
                         f"_hold_samples max-delay: {n_mm} mismatches in {n_steps} steps")


# ---------------------------------------------------------------------------
# MD CUDA stress (only when GPU present)
# ---------------------------------------------------------------------------

@unittest.skipUnless(th.cuda.is_available(), "CUDA not available")
class TestStressAsyncMD(unittest.TestCase):
    """Stress the _BaseMD async handshake (CUDA path only — CPU has no
    threading)."""

    N_RUNS = 15
    MAX_STEPS = 40

    def setUp(self):
        self.X0 = th.tensor([[[0.5, 0.3, 0.1]]], dtype=th.float32, device='cuda:0')
        self.X_init = self.X0 + th.randn(1, 1, 3, device='cuda:0') * 0.1
        self.V_init = th.randn(1, 1, 3, device='cuda:0') * 0.01
        self.out_dir = tempfile.mkdtemp(prefix='stress_async_md_')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_md_cuda_random_delays(self):
        """CUDA MD with random consumer delays."""
        StressNVE_cls = _make_stress_nve()
        failures: list[tuple[int, float, float, int, int]] = []

        for run_i in range(self.N_RUNS):
            out_file = os.path.join(self.out_dir, f'md_{run_i}.bin')
            dump_sleep = random.uniform(0.001, 0.30)
            print_sleep = random.uniform(0.001, 0.20)

            runner = StressNVE_cls(
                dump_sleep=dump_sleep,
                print_sleep=print_sleep,
                time_step=0.1,
                max_step=self.MAX_STEPS,
                output_file=out_file,
                device='cuda:0',
                verbose=1,
                output_structures_per_step=5,
            )
            runner._HOLD_DUMPER = False

            X = self.X_init.clone()
            V = self.V_init.clone()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                runner.run(
                    _harmonic_energy, X, Element_list=[['H']], V_init=V,
                    grad_func=_harmonic_grad,
                    grad_func_args=(self.X0,),
                    func_args=(self.X0,),
                    is_grad_func_contain_y=True,
                )

            # Read binary trajectory
            bs = read_md_traj(out_file)
            dump_E = np.asarray(bs.Energies).ravel()
            log_E = np.array(runner._stress_log_ep).ravel()

            # Element-wise compare (avoids broadcast surprises)
            min_len = min(dump_E.size, log_E.size)
            n_mm = 0
            for j in range(min_len):
                if not np.isclose(float(dump_E[j]), float(log_E[j]), atol=1e-6):
                    n_mm += 1
            if n_mm:
                failures.append((run_i, dump_sleep, print_sleep, min_len, n_mm, dump_E.size, log_E.size))

        self.assertEqual(
            len(failures), 0,
            f"MD CUDA: {len(failures)}/{self.N_RUNS} runs had dump≠log "
            f"(first 5: {failures[:5]})"
        )


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    unittest.main()
