""" Selecting atoms by given conditions """
#  Copyright (c) 2025.11.26, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: AtomSelector.py
#  Environment: Python 3.12

import torch as th
from typing import Sequence, List, Literal
from ._Element_info import ATOMIC_SYMBOL

def atom_fix_selector(
        mask: th.Tensor,
        elements: th.Tensor,
        coords: th.Tensor,
        select_element: Sequence[str | int] | None = None,
        select_height: float | None = None,
        atom_index: List[List[float]] | None = None,
        select_mode: Literal['fix', 'inv_fix', 'free'] = 'fix'
) -> th.Tensor:
    """
    Selecting which atoms are fixed and updating the input mask accordingly.
    Args:
        mask:
        elements:
        coords:
        select_element:
        select_height:
        atom_index:
        select_mode:
            fix: the atoms satisfied given conditions will be set to 0 in given mask, and others do not change.
            inv_fix: only the atoms satisfied given conditions will be set to 1 in given mask, and others set to 0.
            free: the atoms satisfied given conditions will be set to 1 in given mask, and others do not change.

    Returns:

    """
    # check
    if mask.shape != coords.shape:
        raise ValueError(f'Expected `mask` has the same shape as `coords` ({coords.shape}), but got {mask.shape}.')
    if elements.shape != (mask.shape[0], ):
        raise ValueError(f'Expected `elements` has the same shape as the 1st dim of `mask` ({mask.shape[0]}), but got {elements.shape}.')

    # initialize
    FIX_DICT = {'fix': 0, 'inv_fix': 1, 'free': 1}
    fill_elem = FIX_DICT[select_mode]
    if select_mode == 'inv_fix':
        _mask = th.zeros_like(mask).to(th.int8)
    else:
        _mask = th.where(mask.to(th.int8) == 0, 0, 1).to(th.int8)
    elements = elements.unsqueeze(-1).broadcast_to(mask.shape)

    # main
    if select_element is not None:
        sel_elem = [_ if isinstance(_, int) else ATOMIC_SYMBOL[_] for _ in select_element]
        for elem in sel_elem:
            _mask = th.where(elements == elem, fill_elem, _mask)
    if select_height is not None:
        _mask = th.where(coords[:, -1].unsqueeze(-1) <= select_height, fill_elem, _mask)
    if atom_index is not None:
        atom_index = th.tensor(atom_index, dtype=th.int64, device=_mask.device)
        _mask[atom_index] = fill_elem

    return _mask
