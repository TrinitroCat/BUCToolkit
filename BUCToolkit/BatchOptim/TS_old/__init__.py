#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: __init__.py
#  Environment: Python 3.12
from .Dimer import Dimer
from .Krylov import KrylovNewton
import warnings

warnings.warn('This module has been deprecated. Please use BUCToolkit.BatchOptim.TS instead', DeprecationWarning)
__all__ = [
    'Dimer',
    'KrylovNewton',
]
