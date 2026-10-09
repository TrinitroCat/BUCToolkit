"""Programmatic dispatch for the central BUCToolkit test modes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


def build_test_suite(mode: str) -> unittest.TestSuite:
    """Build the requested central test suite.

    Args:
        mode: One of ``main``, ``fast`` or ``profile``.

    Return:
        A unittest suite containing the selected central test class.

    Raises:
        ValueError: If ``mode`` is not supported.
    """
    if mode == "main":
        from main_test import MainTest

        test_class = MainTest
    elif mode == "fast":
        from fast_test import FastTest

        test_class = FastTest
    elif mode == "profile":
        from profile_test import ProfileTest

        test_class = ProfileTest
    else:
        raise ValueError(f"Unsupported central test mode: {mode!r}")
    return unittest.defaultTestLoader.loadTestsFromTestCase(test_class)


def run_test_suite(mode: str, verbosity: int = 2) -> unittest.TestResult:
    """Run one central test suite and return its unittest result.

    Args:
        mode: One of ``main``, ``fast`` or ``profile``.
        verbosity: unittest runner verbosity.

    Return:
        The completed unittest result.
    """
    suite = build_test_suite(mode)
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a BUCToolkit central test suite.")
    parser.add_argument("mode", choices=("main", "fast", "profile"))
    args = parser.parse_args()
    result = run_test_suite(args.mode)
    raise SystemExit(0 if result.wasSuccessful() else 1)
