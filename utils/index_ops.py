""" Indexed operations with given batch_indices """
#  Copyright (c) 2026.1.22, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: index_ops.py
#  Environment: Python 3.12

import warnings

import numpy as np
import torch as th
from typing import Optional, Literal, Callable, Tuple, Dict


def scatter_reduce(
        src: th.Tensor,
        batch_indices: th.Tensor,
        dim: int = -1,
        ops: Literal['sum', 'prod', 'mean', 'amin', 'amax'] = 'sum',
        init_value: float|int|bool = 0.
) -> th.Tensor:
    """
    scatter_reduce method that reduces element in `src` with the same indices in `batch_indices` by `ops`.
    Args:
        src: source tensor
        batch_indices: indices tensor
        dim: reduce dimension
        ops: reduce operation
        init_value: default value of the output tensor that reduced values filled in.

    Returns:

    """
    # manage batch_indices
    _batch_indices = batch_indices.clone().to(th.int64)
    if dim < 0:
        dim = src.dim() + dim
    if _batch_indices.dim() == 1:
        for _ in range(0, dim): _batch_indices.unsqueeze_(0)
    for _ in range(_batch_indices.dim(), src.dim()): _batch_indices.unsqueeze_(-1)
    _batch_indices = _batch_indices.expand(src.shape)
    # main
    src_shape = list(src.shape)
    src_shape[dim] = th.max(_batch_indices).item() + 1
    out = th.full(src_shape, init_value, device=src.device)
    out.scatter_reduce_(
        dim,
        _batch_indices,
        src,
        ops
    )

    return out

def index_reduce(
        src: th.Tensor,
        batch_indices: th.Tensor,
        dim: int = 0,
        ops: Literal['prod', 'mean', 'amin', 'amax', 'sum'] = 'sum',
        init_value: float = 0.,
        out_size: int|None = None
):
    """
    scatter_reduce method that reduces element in `src` with the same indices in `batch_indices` by `ops`.
    Args:
        src: source tensor
        batch_indices: indices tensor of 1D
        dim: reduce dimension
        ops: reduce operation
        init_value: default value of the output tensor that reduced values filled in.
        out_size: size of the output tensor at the given dim. if None, it is equal to `batch_indices.max().item() + 1`

    Returns:

    """
    assert batch_indices.dim() == 1, f'Invalid dimension of batch indices: {batch_indices.shape}. It must be 1D.'
    dim = dim if dim >= 0 else src.dim() + dim
    output_size = list(src.shape)
    output_size[dim] = (batch_indices.max().item() + 1) if out_size is None else out_size

    out = th.full(output_size, init_value, device=src.device, dtype=src.dtype)
    if ops == 'sum':
        out.index_add_(dim, batch_indices, src)
    else:
        out.index_reduce_(dim, batch_indices, src, ops, include_self=True)
    return out

def index_inner_product(
        u: th.Tensor,
        v: th.Tensor,
        dim: int,
        batch_indices: th.Tensor,
        out_size = None,
):
    """
    Calculate the inner product of two tensors with given batch_indices .
    Args:
        u: tensor 1, in the dual space.
        v: tensor 2, in the linear space.
        dim: reduce dimension
        batch_indices: indices tensor of 1D
        out_size: size of the output tensor. If None, it is equal to `batch_indices.max().item() + 1`

    Returns: u @ v with shape (u.shape[: `dim`], batch_number, u.shape[`dim` + 1: ])

    """
    w = u * v
    w = index_reduce(w, batch_indices, dim, ops='sum', init_value=0., out_size=out_size)
    return w

def indices_pairwise_dist(
        X: th.Tensor,            # [N, M]
        Y: th.Tensor,
        Qx: th.Tensor,            # [N]，continuous sample indices in the batch X, 0,0,...,1,1,...,B-1
        Qy: th.Tensor | None = None,
        thres: float | None = None,
        metric: Literal['euclidean', 'dot', 'cosine']|Callable[[th.Tensor, th.Tensor, ...], th.Tensor] = "euclidean",  # 'euclidean' | 'dot' | 'cosine'
        metric_kwargs: Dict|None = None,
        relation: Literal['lt', 'gt'] = "gt",
        exclude_diag: bool = True,
        is_symmetric: bool = False,
        return_values: bool = True,
        is_coalesce: bool = True,
        eps: float = 1e-12,
):
    """
    Compute pairwise distance between two tensors on only the same indices in Qx and Qy.
    Args:
        X: input tensor 1.
        Y: input tensor 2.
        Qx: Tensor[int64], indices tensor of X.
        Qy: Tensor[int64], indices tensor of Y.
        thres: if not None, only distance greater/less (depends on `relation`) than `thres` will be output. Otherwise, all dist will be output.
        metric: distance calculation method.
        metric_kwargs: the keyword arguments passed to `metric` if `metric` is Callable.
        relation: check `thres` greater (gt) or less (lt) than `thres`.
        exclude_diag: whether to exclude diagonals.
        is_symmetric: whether the distance between `X` and `Y` is symmetric. If True, only lower triangular of dist. mat. will be computed.
        return_values: whether to return both indices and distance or only indices of distance < `thres`.
        is_coalesce: whether to coalesce/resort indices and distance before output.
        eps: epsilon for numerical stability.

    Returns:
        Tuple[indices: Tensor[int64], batch_tensor of dist: Tensor[int64]] |
        Tuple[indices: Tensor[int64], distance: Tensor, batch_tensor of dist: Tensor[int64]]

    """
    # check
    assert relation in ("lt", "gt")
    assert X.device == Y.device, f'X and Y must be on the same device, but got {X.device} and {Y.device}.'
    assert X.dtype == Y.dtype, f'X and Y must be on the same dtype, but got {X.dtype} and {Y.dtype}.'
    # initialize
    if metric_kwargs is None:
        metric_kwargs = dict()
    if is_symmetric:
        assert X.shape == Y.shape, f'For symmetric metric, X and Y must have the same shape.'
    if Qy is None:
        Qy = Qx.clone()
    device = X.device
    if thres is None:
        thres = th.inf if relation == "lt" else -th.inf
    # sort & rearrange X, Y, Qx, Qy
    Qx_sorted, Qx_sort_indx = th.sort(Qx)
    Qy_sorted, Qy_sort_indx = th.sort(Qy)
    Xs, Ys = X[Qx_sort_indx], Y[Qy_sort_indx]

    # calc. indices
    def _starts_sizes(Qs):
        N = Qs.numel()
        ch = th.ones(N, dtype=th.bool, device=Qs.device)
        ch[1:] = (Qs[1:] != Qs[:-1])
        st = th.nonzero(ch, as_tuple=True)[0]
        en = th.cat([st[1:], th.tensor([N], device=Qs.device)])
        sz = (en - st).to(th.long)
        labs = Qs[st]
        return st, sz, labs

    ptr_x, nx, labs_x = _starts_sizes(Qx_sorted)
    ptr_y, ny, labs_y = _starts_sizes(Qy_sorted)

    # align two indices
    ix = th.searchsorted(labs_x, labs_y)
    # screen into the intersection (labs_y[pos] == labs_x)
    valid = (ix < labs_x.numel()) & (labs_x[ix] == labs_y)
    labs_y = labs_y[valid]
    ptr_y, ny = ptr_y[valid], ny[valid]
    ix = ix[valid]
    ptr_x, nx = ptr_x[ix], nx[ix]  # ptr_x that aligned with y
    if labs_y.numel() == 0:
        empty = th.empty(2, 0, dtype=th.long, device=device)
        Qe = th.empty(0, dtype=Qx.dtype, device=device)
        return (empty, th.empty(0, device=device), Qe) if return_values else (empty, Qe)

    # vectorized enumerate pairwise nx_i * ny_i, and mapping to Xs and Ys
    pairs_per_img = nx * ny  # [B]
    total_pairs = int(pairs_per_img.sum().item())

    pair_ptr = th.cumsum(pairs_per_img, 0) - pairs_per_img  # [B]
    pair_off_ex = th.repeat_interleave(pair_ptr, pairs_per_img)  # [∑ nx_i*ny_i]
    ny_ex = th.repeat_interleave(ny, pairs_per_img)  # [∑ nx_i*ny_i]
    k = th.arange(total_pairs, device=device) - pair_off_ex

    i_loc = th.div(k, ny_ex, rounding_mode="floor")  # [∑ nx_i*ny_i]
    j_loc = (k % ny_ex)

    xs_ex = th.repeat_interleave(ptr_x, pairs_per_img)  # [∑ nx_i*ny_i]
    ys_ex = th.repeat_interleave(ptr_y, pairs_per_img)

    row_s = xs_ex + i_loc  # indices of Xs
    col_s = ys_ex + j_loc  # indices of Ys
    Qe_s = th.repeat_interleave(labs_y, pairs_per_img)  # indices of pairwise-dist, i.e., the edge.

    # Symmetric reduce
    if is_symmetric:
        if not th.all(nx == ny):
            raise ValueError(f"Symmetric requires Qx, Qy have the same shape in each sample, but got {nx} and {ny}")
        mask = th.ones(total_pairs, dtype=th.bool, device=device)
        mask &= (i_loc <= j_loc)  # select only lower triangular parts.
        if exclude_diag:
            mask &= (i_loc != j_loc)  # remove diag.
        row_s = row_s[mask]
        col_s = col_s[mask]
        Qe_s = Qe_s[mask]

    if row_s.numel() == 0:
        empty = th.empty(2, 0, dtype=th.long, device=device)
        Qe = th.empty(0, dtype=Qx.dtype, device=device)
        return (empty, th.empty(0, device=device), Qe) if return_values else (empty, Qe)

    # Main Calc. Dist.
    xr, yc = Xs[row_s], Ys[col_s]  # [E, M], x_row, y_column

    if isinstance(metric, Callable):
        val = metric(xr, yc, **metric_kwargs)
    elif metric == "euclidean":
        n2x = th.linalg.vecdot(Xs, Xs)
        n2y = th.linalg.vecdot(Ys, Ys)
        dot = th.linalg.vecdot(xr, yc)
        val = n2x[row_s] + n2y[col_s] - 2.0 * dot  # [E]
        thres **= 2 if th.isfinite(th.tensor(thres)) else 1
    elif metric == "dot":
        val = th.linalg.vecdot(xr, yc)
    elif metric == "cosine":
        val = th.linalg.vecdot(xr, yc) / (th.linalg.vector_norm(xr, dim=-1) * th.linalg.vector_norm(yc, dim=-1) + eps)
    else:
        raise ValueError(f"Unknown metric {metric}.")

    is_selected = (th.abs(val) < thres) if relation == "lt" else (th.abs(val) > thres)
    row_s, col_s, Qe = row_s[is_selected], col_s[is_selected], Qe_s[is_selected]

    # remapping to unsorted original X, Y
    row = Qx_sort_indx[row_s]
    col = Qy_sort_indx[col_s]
    indices = th.stack([row, col], dim=0)
    if return_values:
        out_vals: th.Tensor = val[is_selected].clone()
        if metric == "euclidean": out_vals.add_(eps**2).sqrt_()
    else:
        out_vals = val  # only a placeholder
    if is_symmetric:  # appending upper diagonal part
        nd_mask = (row != col)  # non-diagonal
        if th.any(nd_mask):
            upper_diag = th.stack([col[nd_mask], row[nd_mask]], dim=0)
            indices = th.cat([indices, upper_diag], dim=1)
            if return_values:
                out_vals = th.cat([out_vals, out_vals[nd_mask]], dim=0)
            Qe = th.cat([Qe, Qe[nd_mask]], dim=0)

    # coalesce & rearrange indices
    if is_coalesce:
        coalesce_indx = th.argsort(indices[0], dim=0)
        indices = indices[:, coalesce_indx]
        Qe = Qe[coalesce_indx]
        out_vals = out_vals[coalesce_indx] if return_values else out_vals

    if return_values:
        return indices, out_vals, Qe
    return indices, Qe


def inner_product_for_halfspace(u: th.Tensor, v: th.Tensor, volumes: th.Tensor|float = 1.) -> th.Tensor:
    """

    Args:
        u:
        v:
        volumes:

    Returns:

    """
    u0 = u[:, 0]  # (B,)
    v0 = v[:, 0]
    u_tail = u[:, 1:]  # (B, nG-1)
    v_tail = v[:, 1:]  # (B, nG-1)

    p = th.baddbmm(
        th.empty_like(u[:,0]).unsqueeze(-1).unsqueeze(-1),
        u_tail.conj().unsqueeze(1).contiguous(),  # (B, 1, n)
        v_tail.unsqueeze(-1).contiguous(),  # (B, n, 1)
        beta=0.0,
        alpha=2.0,
    ).squeeze(-2, -1)  # (B,)

    # adding gamma point
    p.add_(u0.conj() * v0)  # (B,)
    p = p.real
    if isinstance(volumes, th.Tensor) or volumes != 1.0:
        p.mul_(volumes)

    return p


def standardize_cell(cell: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    Rotation a cell vector matrix into a standardized form where ay = az = 0, bz = 0, cz > 0.
    Args:
        cell: Tensor[B, 3, 3], batch of cell vectors.

    Returns:
        L: Tensor[B, 3, 3], standardized cell vector matrix with lower triangular forms.
        Q: Tensor[B, 3, 3], the corresponding rotation transform

    """
    # determined the cell is a right-hand system
    _cell_std = cell.clone()
    vol = th.linalg.det(_cell_std).unsqueeze(-1)  # (B, 1)
    _cell_std[:, 2, :] = th.where(vol < 0., - cell[:, 2, :], cell[:, 2, :])
    ill_mask = th.abs(vol) < 1e-5
    if ill_mask.any():
        warnings.warn(f'A cell very close to low-dimensional system was detected in [{th.argwhere(ill_mask)}]-th samples.')
    # QR factorization
    Q, R = th.linalg.qr(_cell_std.mT.contiguous())
    _cell_std = R.mT.contiguous()
    _rot_op = Q.mT.contiguous()
    # put diag of R.mT positive
    cell_diag = th.diagonal(_cell_std, dim1=-2, dim2=-1)  # (B, 3)
    neg_diag_mask = (cell_diag < 0.)  # (B, 3)
    _rot_op = th.where(neg_diag_mask.unsqueeze(-1), -_rot_op, _rot_op)  # D Q^T
    _cell_std = th.where(neg_diag_mask.unsqueeze(-2), -_cell_std, _cell_std)  # R^T D

    return _cell_std, _rot_op

def generate_half_space_grids(n_grids: Tuple[int, int, int], device: str = 'cpu') -> th.Tensor:
    """
    Generate a half space grid of given `n_grids`. where： h > 0 or (h = 0 and k > 0) or ((h=0 and k=0) or l>=0)
      * Conventionally, position 0 is set to the gamma point [0, 0, 0] *
    Args:
        n_grids: the number of grids at directions x, y, z.
        device: the device on which the grid is to be generated.

    Returns: (N_grids, 3), grids

    """
    N = [abs(int(_)) + (_ + 1) % 2 for _ in n_grids]  # to ensure an odd positive integer grid number.

    # check the longest axis
    cut_axis = max(range(3), key=lambda i: N[i])
    perm = [cut_axis] + [i for i in range(3) if i != cut_axis]
    N_h, N_k, N_l = (N[perm[0]], N[perm[1]], N[perm[2]])

    # centered int indices [-Nh//2, ..., 0, ..., -Nh//2+Nh-1]
    h_vals = th.arange(0, -(N_h // 2) + N_h, device=device, dtype=th.int64)
    k_vals = th.arange(-(N_k // 2), -(N_k // 2) + N_k, device=device, dtype=th.int64)
    l_vals = th.arange(-(N_l // 2), -(N_l // 2) + N_l, device=device, dtype=th.int64)

    idx_parts = []

    # ---------- 1) h > 0, k,l \in Z ----------
    h_pos = h_vals[1:]
    if h_pos.numel() > 0:
        N_h_pos = h_pos.numel()
        N_k_all, N_l_all = N_k, N_l

        hh1 = h_pos.view(-1, 1, 1).expand(-1, N_k_all, N_l_all)
        kk1 = k_vals.view(1, -1, 1).expand(N_h_pos, -1, N_l_all)
        ll1 = l_vals.view(1, 1, -1).expand(N_h_pos, N_k_all, -1)

        idx1 = th.empty((N_h_pos * N_k_all * N_l_all, 3), device=device, dtype=th.int64)
        idx1[:, 0] = hh1.reshape(-1)
        idx1[:, 1] = kk1.reshape(-1)
        idx1[:, 2] = ll1.reshape(-1)
        idx_parts.append(idx1)

    # ---------- 2) h = 0, k > 0, l \in Z ----------
    k_pos = k_vals[k_vals > 0]
    if k_pos.numel() > 0:
        N_k_pos = k_pos.numel()
        N_l_all = N_l

        hh2 = th.zeros((N_k_pos, N_l_all), device=device, dtype=th.int64)
        kk2 = k_pos.view(-1, 1).expand(-1, N_l_all)
        ll2 = l_vals.view(1, -1).expand(N_k_pos, -1)

        idx2 = th.empty((N_k_pos * N_l_all, 3), device=device, dtype=th.int64)
        idx2[:, 0] = hh2.reshape(-1)
        idx2[:, 1] = kk2.reshape(-1)
        idx2[:, 2] = ll2.reshape(-1)
        idx_parts.append(idx2)

    # ---------- 3) h = 0, k = 0, l > 0 ----------
    l_pos = l_vals[l_vals > 0]
    if l_pos.numel() > 0:
        idx3 = th.empty((l_pos.numel(), 3), device=device, dtype=th.int64)
        idx3[:, 0] = 0
        idx3[:, 1] = 0
        idx3[:, 2] = l_pos
        idx_parts.append(idx3)

    # all non-zero (h,k,l)
    if len(idx_parts) > 0:
        indices_pos = th.cat(idx_parts, dim=0)  # (nG_half, 3)
    else:
        # only G=0
        indices_pos = th.empty((0, 3), device=device, dtype=th.int64)

    # re-permute to original order
    n_pos = indices_pos.shape[0]
    indices_orig = th.empty_like(indices_pos)
    indices_orig[:, perm[0]] = indices_pos[:, 0]
    indices_orig[:, perm[1]] = indices_pos[:, 1]
    indices_orig[:, perm[2]] = indices_pos[:, 2]
    # manually set the 1st element to [0,0,0]
    G_indices = th.empty((n_pos + 1, 3), device=device, dtype=th.int64)
    G_indices[0, :] = 0.0
    G_indices[1:, :] = indices_orig

    return G_indices
