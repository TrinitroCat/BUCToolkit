""" Convert from atomic number to various properties, e.g., masses."""

#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: AtomicNumber2Properties.py
#  Environment: Python 3.12

from typing import Sequence
from _Element_info import MASS, N_MASS
import numpy as np


def _atomic_numbers_to_masses(Z: int):
    mass = N_MASS[int(Z)]
    return mass

atomic_numbers_to_masses = np.vectorize(_atomic_numbers_to_masses)


if __name__ == '__main__':
    z = [1, 2, 5, 3, 66, 32]
    m = atomic_numbers_to_masses(z)
    pass
