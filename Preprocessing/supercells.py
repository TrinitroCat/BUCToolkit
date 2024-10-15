# Calculate the lattice indices of all cells within given cut-off radius.

import torch as th

def supercells(cell_vectors:th.Tensor, r_cut_off:float, device:str|th.device='cpu') -> th.Tensor:
    r'''
    Generate the indices of all lattices within r_cut_off for each input cell.
    return the supercell indices with shape (n_cell, 3) where "3" represents expansion directions.

    Parameters:
        cell_vectors: Tensor(n_batch, 3, 3), a batch of lattice vectors.
        r_cut_off: float, the cut-off radius.
        device: str, the device that calculation performed.
    
    Returns: Tensor(n_cell, 3), the tensor of all supercell indices for the given batch.
    '''
    # Volume, Areas, and Height of cells
    V = th.linalg.det(cell_vectors) # (n_batch)
    s_ab = th.linalg.cross(cell_vectors[:,0,:], cell_vectors[:,1,:]) # (n_batch, 3)
    s_bc = th.linalg.cross(cell_vectors[:,1,:], cell_vectors[:,2,:]) # (n_batch, 3)
    s_ac = th.linalg.cross(cell_vectors[:,0,:], cell_vectors[:,2,:]) # (n_batch, 3)

    #h_vec_ab = V/((s_ab.unsqueeze(1)@s_ab.unsqueeze(-1)).squeeze(-1)) * s_ab
    #h_vec_bc = V/((s_bc.unsqueeze(1)@s_bc.unsqueeze(-1)).squeeze(-1)) * s_bc
    #h_vec_ac = V/((s_ac.unsqueeze(1)@s_ac.unsqueeze(-1)).squeeze(-1)) * s_ac

    h_ab = V/(th.linalg.norm(s_ab, ord=2, dim=1)) # (n_batch, )
    h_bc = V/(th.linalg.norm(s_bc, ord=2, dim=1)) # (n_batch, )
    h_ac = V/(th.linalg.norm(s_ac, ord=2, dim=1)) # (n_batch, )

    # repitition times of cell in direction a, b, c 
    repitition_a = th.ceil(r_cut_off/h_bc) # (n_batch, )
    repitition_b = th.ceil(r_cut_off/h_ac) # (n_batch, )
    repitition_c = th.ceil(r_cut_off/h_ab) # (n_batch, )

    # max of repititions
    rep_max_a = int(th.max(repitition_a).item())
    rep_max_b = int(th.max(repitition_b).item())
    rep_max_c = int(th.max(repitition_c).item())


    # construct supercell
    a_ = th.empty(2*rep_max_a+1, 1, dtype=th.float, device=device)
    b_ = th.empty(2*rep_max_b+1, 1, dtype=th.float, device=device)
    c_ = th.empty(2*rep_max_c+1, 1, dtype=th.float, device=device)
    a_[:rep_max_a+1, 0] = th.arange(0, rep_max_a+1, dtype=th.float, device=device)
    a_[rep_max_a+1:, 0] = th.arange(- rep_max_a, 0, dtype=th.float, device=device)
    b_[:rep_max_b+1, 0] = th.arange(0, rep_max_b+1, dtype=th.float, device=device)
    b_[rep_max_b+1:, 0] = th.arange(- rep_max_b, 0, dtype=th.float, device=device)
    c_[:rep_max_c+1, 0] = th.arange(0, rep_max_c+1, dtype=th.float, device=device)
    c_[rep_max_c+1:, 0] = th.arange(- rep_max_c, 0, dtype=th.float, device=device)

    supercell_index = th.cat(
        (
            a_.repeat_interleave((2*rep_max_b+1)*(2*rep_max_c+1), 0),
            th.tile(b_.repeat_interleave((2*rep_max_c+1), 0), ((2*rep_max_a+1), 1)),
            th.tile(c_, ((2*rep_max_a+1)*(2*rep_max_b+1), 1))
        )
        ,dim=1)

    return supercell_index