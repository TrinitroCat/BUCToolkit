""" Convert from atomic number to various properties, e.g., masses, elements."""

#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: AtomicNumber2Properties.py
#  Environment: Python 3.12

from typing import Sequence
from BM4Ckit.utils._Element_info import MASS, N_MASS, ATOMIC_SYMBOL, ATOMIC_NUMBER
import numpy as np


def _atomic_numbers_to_masses(Z: int):
    mass = N_MASS[int(Z)]
    return mass


def _atomic_numbers_to_elements(Z: int):
    elem = ATOMIC_NUMBER[int(Z)]
    return elem


def _elements_to_atomic_numbers(Z: str):
    atomic_numbers = ATOMIC_SYMBOL[str(Z)]
    return atomic_numbers


atomic_numbers_to_masses = np.vectorize(_atomic_numbers_to_masses)
atomic_numbers_to_elements = np.vectorize(_atomic_numbers_to_elements)
elements_to_atomic_numbers = np.vectorize(_elements_to_atomic_numbers)

__all__ = [
    "atomic_numbers_to_masses",
    "atomic_numbers_to_elements",
    "elements_to_atomic_numbers",
]

if __name__ == '__main__':
    z = [1, 2, 5, 3, 66, 32]
    m = atomic_numbers_to_masses(z)
    pass
