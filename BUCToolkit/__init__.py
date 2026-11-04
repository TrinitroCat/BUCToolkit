#  Copyright (c) 2024-2026.3.27, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: __init__.py
#  Environment: Python 3.12

import sys
import warnings

if sys.platform.startswith("win"):
    warnings.warn(
        "WARNING: Detected Windows os. "
        "BUCToolkit has not fully test on Windows yet, be careful!",
        Warning
    )
elif sys.platform.startswith("linux"):
    pass
else:
    warnings.warn(
        f"WARNING: Detected OS of {sys.platform}. "
        f"BUCToolkit has not been tested on this platform. Please be careful!",
        Warning
    )


from BUCToolkit.BatchStructures.BatchStructuresBase import BatchStructures as Structures
from BUCToolkit import utils
from BUCToolkit.utils._Element_info import TRANSITION_METALS, TRANSITION_P_METALS, NOBLE_METALS, NONRADIOACTIVE_METALS
from BUCToolkit.Preprocessing import load_files
from BUCToolkit.Preprocessing.load_files import load_from_structures as load
from BUCToolkit.Preprocessing import preprocessing
from BUCToolkit import io
from BUCToolkit.Bases.StdContainer import StdContainer
from BUCToolkit import api
from BUCToolkit import BatchMC, BatchMD, BatchOptim, BatchStructures, Postprocessing

__all__ = [
    'Structures',
    'StdContainer',
    'load',
    'load_files',
    'preprocessing',
    'TRANSITION_METALS',
    'TRANSITION_P_METALS',
    'NOBLE_METALS',
    'NONRADIOACTIVE_METALS',
    'utils',
    'io',
    'api',
    'BatchMC',
    'BatchMD',
    'BatchOptim',
    'BatchStructures',
    'Postprocessing'
]

# Handle the POSIX signal, take over the SIGTERM to exit program normally (through `finally` sections)
import signal
import threading


def _raise_on_sigterm(signum: int, _frame) -> None:
    """Convert SIGTERM into a Python termination exception.

    Args:
        signum: Received signal number.
        _frame: Interrupted stack frame, unused.

    Raises:
        SystemExit: Always, using the conventional 128 + signal number code.
    """
    # SystemExit is a BaseException, so `except Exception` blocks in the task
    # APIs cannot swallow it.  Propagating it unwinds the stack and runs every
    # active finally block, which is where dumpers are closed/flushed and
    # buffered checkpoints are joined.
    raise SystemExit(128 + signum)


def _install_sigterm_handler() -> None:
    """Install the SIGTERM handler when the current thread can do so.

    Returns:
        None.

    Notes:
        signal.signal is only valid in the interpreter main thread.  An
        existing application-defined handler or SIG_IGN is left untouched.
    """
    if not hasattr(signal, 'SIGTERM'):
        return
    if threading.current_thread() is not threading.main_thread():
        return

    try:
        current = signal.getsignal(signal.SIGTERM)
        if (current is not signal.SIG_DFL) and (current is not _raise_on_sigterm):
            # Do not affect the behaviour of SIG_IGN, None (external C) and u.d.f callables
            return
        signal.signal(signal.SIGTERM, _raise_on_sigterm)
    except (ValueError, OSError):
        # Embedded/restricted interpreter: keep the platform default.
        return


_install_sigterm_handler()
