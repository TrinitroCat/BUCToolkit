""" Convert from atomic number to various properties, e.g., masses, elements."""

#  Copyright (c) 2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: AtomicNumber2Properties.py
#  Environment: Python 3.12

from typing import Sequence, List
from BUCToolkit.utils._Element_info import MASS, N_MASS, ATOMIC_SYMBOL, ATOMIC_NUMBER
import numpy as np


def atomic_numbers_to_masses(Z: Sequence[int]) -> List[float]:
    """Convert atomic numbers to masses."""
    mass = [N_MASS[int(_)] for _ in Z]
    return mass


def atomic_numbers_to_elements(Z: Sequence[int]) -> List[int]:
    elem = [ATOMIC_NUMBER[int(_)] for _ in Z]
    return elem


def elements_to_atomic_numbers(Z: Sequence[str]) -> List[int]:
    atomic_numbers = [ATOMIC_SYMBOL[str(_)] for _ in Z]
    return atomic_numbers


__all__ = [
    "atomic_numbers_to_masses",
    "atomic_numbers_to_elements",
    "elements_to_atomic_numbers",
]

if __name__ == '__main__':
    z = [1, 2, 5, 3, 66, 32]
    m = atomic_numbers_to_masses(z)
    pass
