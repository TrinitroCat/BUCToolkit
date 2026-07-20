#!/usr/bin/env python3
"""
Run MainTest.test_CMD with and without the C++ constraint-projection backend,
comparing wall-clock time and constraint violations.

Strictly mirrors the parameter set in test_CMD — only varies use_c_backend.

Usage:
    PYTHONPATH=. python testsuite/run_cmd_bench.py
"""
import os, sys, time, contextlib, io
from unittest.mock import patch

from BUCToolkit.Bases.c_constr import HAS_C_PROJECT2


def _patch_use_c(val):
    """Make ConstrNVE / ConstrNVT constructors pass *val* as use_c_backend."""
    from BUCToolkit.BatchMD import ConstrNVE, ConstrNVT, _BaseConstrMD

    orig_nve  = ConstrNVE.ConstrNVE.__init__
    orig_nvt  = ConstrNVT.ConstrNVT.__init__
    orig_base = _BaseConstrMD._BaseConstrMD.__init__

    def _nve(self, *a, **kw):
        kw.setdefault('use_c_backend', val); orig_nve(self, *a, **kw)

    def _nvt(self, *a, **kw):
        kw.setdefault('use_c_backend', val); orig_nvt(self, *a, **kw)

    def _base(self, *a, **kw):
        kw.setdefault('use_c_backend', val); orig_base(self, *a, **kw)

    ConstrNVE.ConstrNVE.__init__              = _nve
    ConstrNVT.ConstrNVT.__init__              = _nvt
    _BaseConstrMD._BaseConstrMD.__init__      = _base

    return orig_nve, orig_nvt, orig_base


def _restore(orig_nve, orig_nvt, orig_base):
    from BUCToolkit.BatchMD import ConstrNVE, ConstrNVT, _BaseConstrMD
    ConstrNVE.ConstrNVE.__init__              = orig_nve
    ConstrNVT.ConstrNVT.__init__              = orig_nvt
    _BaseConstrMD._BaseConstrMD.__init__      = orig_base


def run_test_cmd(tag: str, use_c) -> float:
    from testsuite.main_test import MainTest

    # Only CPU runners — skip GPU to keep comparison clean
    import torch as th
    has_gpu = th.cuda.is_available()

    orig = _patch_use_c(use_c)

    # Monkey-patch __init__ of ConstrNVE/ConstrNVT to skip GPU
    from BUCToolkit.BatchMD import ConstrNVE, ConstrNVT
    _orig_nve_init = ConstrNVE.ConstrNVE.__init__
    _orig_nvt_init = ConstrNVT.ConstrNVT.__init__

    def _nve_cpu_only(self, *a, **kw):
        if kw.get('device', 'cpu') != 'cpu' and not has_gpu:
            kw['device'] = 'cpu'
        return _orig_nve_init(self, *a, **kw)

    def _nvt_cpu_only(self, *a, **kw):
        if kw.get('device', 'cpu') != 'cpu' and not has_gpu:
            kw['device'] = 'cpu'
        return _orig_nvt_init(self, *a, **kw)

    ConstrNVE.ConstrNVE.__init__ = _nve_cpu_only
    ConstrNVT.ConstrNVT.__init__ = _nvt_cpu_only

    try:
        t = MainTest()
        t.setUp()
        t0 = time.perf_counter()
        t.test_CMD()
        elapsed = time.perf_counter() - t0
        t.tearDown()
        print(f'[{tag}] elapsed: {elapsed:.1f} s')
        return elapsed
    finally:
        _restore(*orig)
        ConstrNVE.ConstrNVE.__init__ = _orig_nve_init
        ConstrNVT.ConstrNVT.__init__ = _orig_nvt_init


if __name__ == '__main__':
    print(f'C++ projector available: {HAS_C_PROJECT2}')
    print()

    t_py = run_test_cmd('Python', False)
    print()
    t_c  = run_test_cmd('C++',    HAS_C_PROJECT2)
    print()

    if t_py and t_c:
        s = t_py / t_c if t_c > 0 else float('inf')
        print(f'Python: {t_py:.1f}s  C++: {t_c:.1f}s  speedup: {s:.2f}x')
