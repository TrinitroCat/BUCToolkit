#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: ElemListReduce.py
#  Environment: Python 3.12
"""
Convert a dense element list (atomic number corresponds to coordinates one by one) into VASP-like compressed format.
"""
from typing import List, Tuple
from ._Element_info import ATOMIC_NUMBER, ATOMIC_SYMBOL
import numpy as np


def elem_list_reduce(
        atom_list: List[str|int]|np.ndarray
) -> Tuple[List[str], List[int], List[int]]:
    """
    Convert a dense element list (atomic number corresponds to coordinates one by one) into VASP-like compressed format.

    Args:
        atom_list:

    Returns: element symbol list, atomic numbers list, atom number (count) list

    """
    if isinstance(atom_list, np.ndarray): atom_list = atom_list.tolist()
    _elem_old = ATOMIC_NUMBER[atom_list[0]] if isinstance(atom_list[0], int) else atom_list[0]
    elements = [_elem_old, ]
    atomic_numbers = [ATOMIC_SYMBOL[_elem_old], ]
    elem_numbers = list()
    count_i = 0
    for elem in atom_list:
        if isinstance(elem, int):
            elem_symb = ATOMIC_NUMBER[elem]
            elem_numb = elem
        elif isinstance(elem, str):
            elem_symb = elem
            elem_numb = ATOMIC_SYMBOL[elem]
        else:
            raise TypeError(f'Expected str or int in `atom_list`, but got {type(elem)}.')

        if elem_symb == _elem_old:
            count_i += 1
        else:
            _elem_old = elem_symb
            elem_numbers.append(count_i)
            elements.append(elem_symb)
            atomic_numbers.append(elem_numb)
            count_i = 1
    elem_numbers.append(count_i)

    return elements, atomic_numbers, elem_numbers
