"""
Linear interpolate of 2 structures
"""

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: LinearInterpolate.py
#  Environment: Python 3.12

from typing import List
import numpy as np

from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures

def linear_interpolation(
        coords1: np.ndarray | List,
        coords2: np.ndarray | List,
        n_points: int,

):
    """

    Args:
        coords1: the coordinate of 1st structure.
        coords2: the coordinate of 2nd structure.
        n_points: the number of interpolation points.

    Returns:
        np.ndarray[(n_points, n_atoms, 3)], an array of interpolated structures' coordinates.
    """
    # check vars
    if isinstance(coords1, List):
        coords1 = np.asarray(coords1, dtype=np.float32)
    elif not isinstance(coords1, np.ndarray):
        raise TypeError(f'Invalid type of `coord1`: {type(coords1)}')
    if isinstance(coords2, List):
        coords2 = np.asarray(coords2, dtype=np.float32)
    elif not isinstance(coords2, np.ndarray):
        raise TypeError(f'Invalid type of `coord2`: {type(coords2)}')

    # main
    points_indx = np.linspace(0, 1, n_points)  # n_pt
    coord_diff = coords2 - coords1  # n_atom, 3
    interp_struc = coords1[None, :, :] + points_indx[:, None, None] * coord_diff[None, :, :]  # (1, n_atom, 3) + (n_pt, 1, 1) * (1, n_atom, 3)

    return interp_struc

