#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: supercells.py
#  Environment: Python 3.12

# Calculate the lattice indices of all cells within given cut-off radius.

from typing import List, Tuple

import numpy as np
import torch as th


def supercells_indices_within_cutoff(cell_vectors: th.Tensor, r_cut_off: float, device: str | th.device = 'cpu') -> th.Tensor:
    r"""
    Generate the indices of all lattices within r_cut_off for each input cell.
    return the supercell indices with shape (n_cell, 3) where "3" represents expansion directions.

    Parameters:
        cell_vectors: Tensor(n_batch, 3, 3), a batch of lattice vectors.
        r_cut_off: float, the cut-off radius.
        device: str, the device that calculation performed.

    Returns: Tensor(n_cell, 3), the tensor of all supercell indices for the given batch.
    """
    # Volume, Areas, and Height of cells
    V = th.linalg.det(cell_vectors)  # (n_batch)
    s_ab = th.linalg.cross(cell_vectors[:, 0, :], cell_vectors[:, 1, :])  # (n_batch, 3)
    s_bc = th.linalg.cross(cell_vectors[:, 1, :], cell_vectors[:, 2, :])  # (n_batch, 3)
    s_ac = th.linalg.cross(cell_vectors[:, 0, :], cell_vectors[:, 2, :])  # (n_batch, 3)

    #h_vec_ab = V/((s_ab.unsqueeze(1)@s_ab.unsqueeze(-1)).squeeze(-1)) * s_ab
    #h_vec_bc = V/((s_bc.unsqueeze(1)@s_bc.unsqueeze(-1)).squeeze(-1)) * s_bc
    #h_vec_ac = V/((s_ac.unsqueeze(1)@s_ac.unsqueeze(-1)).squeeze(-1)) * s_ac

    h_ab = V / (th.linalg.norm(s_ab, ord=2, dim=1))  # (n_batch, )
    h_bc = V / (th.linalg.norm(s_bc, ord=2, dim=1))  # (n_batch, )
    h_ac = V / (th.linalg.norm(s_ac, ord=2, dim=1))  # (n_batch, )

    # repitition times of cell in direction a, b, c 
    repitition_a = th.ceil(r_cut_off / h_bc)  # (n_batch, )
    repitition_b = th.ceil(r_cut_off / h_ac)  # (n_batch, )
    repitition_c = th.ceil(r_cut_off / h_ab)  # (n_batch, )

    # max of repititions
    rep_max_a = int(th.max(repitition_a).item())
    rep_max_b = int(th.max(repitition_b).item())
    rep_max_c = int(th.max(repitition_c).item())

    # construct supercell
    a_ = th.empty(2 * rep_max_a + 1, 1, dtype=th.float, device=device)
    b_ = th.empty(2 * rep_max_b + 1, 1, dtype=th.float, device=device)
    c_ = th.empty(2 * rep_max_c + 1, 1, dtype=th.float, device=device)
    a_[:rep_max_a + 1, 0] = th.arange(0, rep_max_a + 1, dtype=th.float, device=device)
    a_[rep_max_a + 1:, 0] = th.arange(- rep_max_a, 0, dtype=th.float, device=device)
    b_[:rep_max_b + 1, 0] = th.arange(0, rep_max_b + 1, dtype=th.float, device=device)
    b_[rep_max_b + 1:, 0] = th.arange(- rep_max_b, 0, dtype=th.float, device=device)
    c_[:rep_max_c + 1, 0] = th.arange(0, rep_max_c + 1, dtype=th.float, device=device)
    c_[rep_max_c + 1:, 0] = th.arange(- rep_max_c, 0, dtype=th.float, device=device)

    supercell_index = th.cat(
        (
            a_.repeat_interleave((2 * rep_max_b + 1) * (2 * rep_max_c + 1), 0),
            th.tile(b_.repeat_interleave((2 * rep_max_c + 1), 0), ((2 * rep_max_a + 1), 1)),
            th.tile(c_, ((2 * rep_max_a + 1) * (2 * rep_max_b + 1), 1))
        )
        , dim=1)

    return supercell_index


def supercell(supercell_index: Tuple[int, int, int] | List[int],
              Cells: th.Tensor,
              Coords: th.Tensor,
              Elements: List[List[str]],
              Numbers: List[List[int]],
              device: str | th.device = 'cpu'
              ) -> (th.Tensor, th.Tensor, List, List):
    """
    expand given cells to supercell.
    Args:
        supercell_index: (3, )
        Cells: (n_batch, 3, 3)
        Coords: (n_batch, n_atom, 3)
        Elements: (n_batch, n_elem)
        Numbers: (n_batch, n_elem)
        device: the device that program run on.

    Returns:
        supercell_index
        Cells
        Coords
        Elements
        Numbers
    """
    # check vars
    if not isinstance(Cells, th.Tensor): raise TypeError(f'Cells must be torch.Tensor, but occurred {type(Cells)}')
    if not isinstance(Coords, th.Tensor): raise TypeError(f'Coords must be torch.Tensor, but occurred {type(Coords)}')
    n_batch, n_atom, _ = Coords.shape
    if Cells.shape != (n_batch, 3, 3): raise ValueError(f'Uncorrected Cell shape: {Cells.shape}')
    if (len(Elements) != len(Numbers)) or (len(Elements) != n_batch):
        raise ValueError(f'Uncorrected Elements or Numbers length: {len(Elements), len(Numbers)}')

    Cells = Cells.to(device)
    Coords = Coords.to(device)

    # supercells' cell vectors
    sc_id = th.from_numpy(np.asarray(supercell_index, dtype=np.float32)).to(device)  # (3, )
    sc = th.diag(sc_id, 0)  # (3, 3)
    Cells_ = sc @ Cells  # (n_batch, 3, 3)

    # construct supercells' coordinate
    rep_max_a, rep_max_b, rep_max_c = supercell_index
    a_ = th.empty(rep_max_a, 1, dtype=th.float, device=device)
    b_ = th.empty(rep_max_b, 1, dtype=th.float, device=device)
    c_ = th.empty(rep_max_c, 1, dtype=th.float, device=device)
    a_[:, 0] = th.arange(0, rep_max_a, 1, dtype=th.float, device=device)
    b_[:, 0] = th.arange(0, rep_max_b, 1, dtype=th.float, device=device)
    c_[:, 0] = th.arange(0, rep_max_c, 1, dtype=th.float, device=device)

    indx = th.cat(
        (
            a_.repeat_interleave(rep_max_b * rep_max_c, 0),
            th.tile(b_.repeat_interleave(rep_max_c, 0), (rep_max_a, 1)),
            th.tile(c_, (rep_max_a * rep_max_b, 1))
        )
        , dim=1
    )  # (n_cells, 3), n_cells = multiple(supercell_index)
    n_cell = len(indx)

    # (n_batch, 1, n_atom, 3) + (1, n_cells, 1, 3) @ (n_batch, 1, 3, 3) -> (n_batch, n_cells, n_atom, 3)
    Coords_ = Coords.unsqueeze(1) + indx.unsqueeze(0).unsqueeze(-2) @ Cells.unsqueeze(1)
    Coords_ = Coords_.flatten(1, 2)  # (n_batch, n_cells * n_atom, 3)

    # Elements & Numbers
    Elements_ = [Elements[i] * n_cell for i in range(n_batch)]
    Numbers_ = [Numbers[i] * n_cell for i in range(n_batch)]

    return Cells_, Coords_, Elements_, Numbers_
