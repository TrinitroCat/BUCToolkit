

import torch as th
import triton
import triton.language as tl


@triton.jit
def velocity_verlet_1_kernel(
        X_ptr,
        V_ptr,
        F_ptr: tl.const,
        inv_M_ptr: tl.const,
        time_step: tl.constexpr,
        size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    dtype = X_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)
    # update velocity
    F = tl.mul(F, tl.full([], 0.5 * time_step * 9.64853329045427e-3, dtype))
    V = tl.fma(F, inv_M, V)
    tl.store(V_ptr + offsets, V, mask=mask)
    # update X
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    X = tl.fma(V, tl.full([], time_step, dtype), X)
    tl.store(X_ptr + offsets, X, mask=mask)


@triton.jit
def velocity_verlet_2_kernel(
        V_ptr,
        F_ptr,
        mask_ptr: tl.const,
        inv_M_ptr: tl.const,
        time_step: tl.constexpr,
        size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    dtype = V_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)
    atom_mask = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    # mask the forces and write the masked forces back into the state
    F = tl.mul(F, atom_mask)
    tl.store(F_ptr + offsets, F, mask=mask)
    # main update
    F = tl.mul(F, tl.full([], 0.5 * time_step * 9.64853329045427e-3, dtype))
    V = tl.fma(F, inv_M, V)
    tl.store(V_ptr + offsets, V, mask=mask)


def vv1(
        X: th.Tensor,
        V: th.Tensor,
        F: th.Tensor,
        inv_M: th.Tensor,
        time_step: float,
):
    n_elem = X.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    velocity_verlet_1_kernel[grids](
        X, V, F, inv_M,
        time_step,
        n_elem,
        BLOCK_SIZE=256
    )

def vv2(
        V: th.Tensor,
        F: th.Tensor,
        atom_masks: th.Tensor,
        inv_M: th.Tensor,
        time_step: float,
):
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    velocity_verlet_2_kernel[grids](
        V, F, atom_masks, inv_M,
        time_step,
        n_elem,
        BLOCK_SIZE=256
    )
