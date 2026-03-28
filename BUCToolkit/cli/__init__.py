"""
An Advanced Interactive Command-Line Interface which can run end-to-end tasks of model training, structure optimization, molecular dynamics,
and Monte Carlo simulations.
"""

#  Copyright (c) 2026.3.26, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: __init__.py
#  Environment: Python 3.12

from .cli import BaseCLI

__all__ = [
    "BaseCLI",
]

def run_base_cli():
    f = BaseCLI()
    f.run()
