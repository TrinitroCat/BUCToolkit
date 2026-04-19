#  Copyright (c) 2024-2026.3.27, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: __init__.py
#  Environment: Python 3.12

import warnings
import sys

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
from BUCToolkit import api
from BUCToolkit import BatchMC, BatchMD, BatchOptim, BatchStructures

__all__ = [
    'Structures',
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
    'BatchStructures'
]
