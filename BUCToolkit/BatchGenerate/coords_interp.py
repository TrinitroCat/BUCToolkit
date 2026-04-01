"""
Interpolation from 2 structures
"""

#  Copyright (c) 2024-2025.10.10, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: coords_interp.py
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
        th.Tensor[(n_points, n_atoms, n_dim)], an array of interpolated structures' coordinates.
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

def direction_for_finite_diff(
        coords1: th.Tensor | np.ndarray,
        coords2: th.Tensor | np.ndarray,
        atom_mask: th.Tensor | np.ndarray | None = None,
        interp_position: float = 0.5,
):
    """
    Generate a direction for finite difference. Usually used for Dimer/Lanczos algo.
    Args:
        coords1: the initial coordinate.
        coords2: the final coordinate.
        atom_mask: the mask of `coords1` and `coords2` that fixes atoms not to move. wherein, 1 is for free and 0 is for fixation.
        interp_position: the interpolation position of returned interp. coordinate. It must be between 0 and 1,
            and gives the coordinate of `coords1 + interp_position * (coords2 - coords1)`.

    Returns:
        interp_coo: th.Tensor[(n_points, n_atoms, n_dim)], an array of interpolated structures' coordinates.
        diff_coo  : the normalized difference between `coords1` and `coords2`, i.e., a unit vector.
    """
    # check
    if not isinstance(coords1, th.Tensor):
        dev = 'cpu'
    else:
        dev = coords1.device
    if atom_mask is None:
        atom_mask = th.ones_like(coords1)[None, :, :]
    elif isinstance(atom_mask, th.Tensor) or isinstance(atom_mask, np.ndarray):
        if atom_mask.shape != coords1.shape:
            raise TypeError(f'Invalid shape of `atom_mask`: {atom_mask.shape}. It should be the same as `coord1`: {coords1.shape}.')
        atom_mask = atom_mask[None, :, :]
        atom_mask = th.as_tensor(atom_mask, dtype=atom_mask.dtype, device=dev)
    else:
        raise TypeError(f'Invalid type of `atom_mask`: {type(atom_mask)}')

    coords1 = th.as_tensor(coords1, dtype=coords1.dtype, device=dev)
    coords2 = th.as_tensor(coords2, dtype=coords2.dtype, device=dev)

    interp_position = float(interp_position)
    if interp_position > 1. or interp_position < 0.:
        raise ValueError(f"`interp_position` should be between 0 and 1.")
    diff_coo = (coords2 - coords1) * atom_mask
    interp_coo = th.add(coords1, diff_coo, alpha=interp_position)
    th.div(diff_coo, th.linalg.norm(diff_coo, dim=(-2, -1), keepdim=True), out=diff_coo)

    return interp_coo, diff_coo

