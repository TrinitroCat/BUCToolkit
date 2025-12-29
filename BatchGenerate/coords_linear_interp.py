"""
Linear interpolate of 2 structures
"""

#  Copyright (c) 2024-2025.10.10, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: coords_linear_interp.py
#  Environment: Python 3.12

from typing import List
import numpy as np
import torch as th


def linear_interpolation(
        coords1: np.ndarray | List,
        coords2: np.ndarray | List,
        n_points: int,

):
    """
    Linear interpolation between `coords1` and `coords2` to generate a series of coordinates, usually used for NEB-like methods.
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

def linear_interpolation_tens(
        coords1: th.Tensor,
        coords2: th.Tensor,
        n_points: int,
        atom_mask: th.Tensor|None = None,
):
    """
    The linear interpolation between `coords1` and `coords2` for torch.Tensor inputs. See `linear_interpolation`.
    Args:
        coords1: the coordinate of 1st structure.
        coords2: the coordinate of 2nd structure.
        n_points: the number of interpolation points.
        atom_mask: the mask of `coords1` and `coords2` that fixes atoms not to move. wherein, 1 is for free and 0 is for fixation.

    Returns:
        th.Tensor[(n_points, n_atoms, 3)], an array of interpolated structures' coordinates.
    """
    # check
    if atom_mask is None:
        atom_mask = th.ones_like(coords1)[None, :, :]
    elif (isinstance(atom_mask, th.Tensor)) or isinstance(atom_mask, np.ndarray):
        if atom_mask.shape != coords1.shape:
            raise TypeError(f'Invalid shape of `atom_mask`: {atom_mask.shape}. It should be the same as `coord1`: {coords1.shape}.')
        atom_mask = atom_mask[None, :, :]
    else:
        raise TypeError(f'Invalid type of `atom_mask`: {type(atom_mask)}')
    # main
    points_indx = th.linspace(0, 1, n_points, device=coords1.device, dtype=coords1.dtype)  # n_pt
    coord_diff = coords2 - coords1  # n_atom, 3
    interp_struc = coords1[None, :, :] + points_indx[:, None, None] * coord_diff[None, :, :] * atom_mask  # (1, n_atom, 3) + (n_pt, 1, 1) * (1, n_atom, 3)

    return interp_struc

