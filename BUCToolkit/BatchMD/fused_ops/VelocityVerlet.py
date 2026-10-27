#  Copyright (c) 2024-2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: VelocityVerlet.py
#  Environment: Python 3.12

"""Fused velocity-Verlet kernels.

The two half-steps of one velocity-Verlet update are

    v <- v + 0.5 * dt * C * F / m
    x <- x + dt * v

with ``C = 9.64853329045427e-3`` converting the force term to the BUCToolkit
velocity unit.  Both kernels operate in place on flat contiguous buffers and
support float16, float32, and float64 states: the element type is read from
the pointer itself and every constant is materialized in that type.

Layout
------
``X``, ``V``, ``F``, ``inv_M`` and the selective-dynamics mask all have the
same flat shape, either ``(n_batch * n_atom * 3,)`` for a regular batch or
``(sum_n_atom * 3,)`` for an irregular batch.  The kernels are elementwise, so
they never need to know which layout is active.
"""

import torch as th
import triton
import triton.language as tl

from .utils import DEFAULT_ELEMENTWISE_BLOCK


# Section: first half-step (velocity update followed by drift)
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
    """Advance one velocity-Verlet first half-step.

    The force is read from the previous step, the velocity is updated by half
    a kick, and the position is then drifted with the new velocity.

    Args:
        X_ptr: flat coordinates, updated in place.
        V_ptr: flat velocities, updated in place.
        F_ptr: flat forces of the previous step (read-only).
        inv_M_ptr: flat inverse masses broadcast to the coordinates.
        time_step: MD time step in fs.
        size: total number of flat elements.
        BLOCK_SIZE: Triton block size.

    Returns:
        None. ``X`` and ``V`` are updated in place.

    Note:
        ``F_ptr`` and ``inv_M_ptr`` are never written, so they are marked
        ``tl.const``; ``size`` and ``time_step`` are fixed for a run and are
        ``tl.constexpr`` so they are inlined into the generated code.
    """
    # Read the element type from the pointer so the kernel specializes once per
    # state dtype (float16 / float32 / float64).
    dtype = X_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    # The offsets of one program are contiguous; tell the compiler so it can
    # coalesce and vectorize the loads and stores.
    offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < size

    # Section: load the current state
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)

    # Section: half kick, v += 0.5 * dt * C * F * inv_M
    # The constant is created in the state dtype instead of relying on the
    # default float32 promotion of a Python literal.
    half_time_step_const = tl.full([], 0.5 * time_step * 9.64853329045427e-3, dtype)
    F = tl.mul(F, half_time_step_const)
    V = tl.fma(F, inv_M, V)
    tl.store(V_ptr + offsets, V, mask=mask)

    # Section: drift, x += dt * v
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    X = tl.fma(V, tl.full([], time_step, dtype), X)
    tl.store(X_ptr + offsets, X, mask=mask)


# Section: second half-step (force masking and velocity update)
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
    """Advance one velocity-Verlet second half-step.

    The selective-dynamics mask is applied to the new forces, the masked
    forces are written back to the state (the next first half-step consumes
    them), and the velocity receives the second half kick.  The position is
    not touched here; it was already drifted by the first kernel.

    Args:
        V_ptr: flat velocities, updated in place.
        F_ptr: flat forces from the current step; overwritten by the masked
            forces.
        mask_ptr: selective-dynamics mask in ``{0, 1}`` (read-only).
        inv_M_ptr: flat inverse masses broadcast to the velocities.
        time_step: MD time step in fs.
        size: total number of flat elements.
        BLOCK_SIZE: Triton block size.

    Returns:
        None. ``V`` is updated and ``F`` is overwritten in place.

    Note:
        ``F_ptr`` must be writable: the eager engine stores the masked force in
        ``s.Force`` and the next step's first half-step reads it back.
    """
    dtype = V_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < size

    # Section: load the new forces and the selective-dynamics mask
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)
    atom_mask = tl.load(mask_ptr + offsets, mask=mask, other=0.0)

    # Section: mask the forces and store them back into the state
    # ``atom_mask`` is 0 for fixed atoms, so their accumulated force is cleared
    # before the velocity update and the cleared value is what the next step
    # reads.  Writing it here avoids a separate elementwise multiply in Python.
    F = tl.mul(F, atom_mask)
    tl.store(F_ptr + offsets, F, mask=mask)

    # Section: second half kick, v += 0.5 * dt * C * F * inv_M
    half_time_step_const = tl.full([], 0.5 * time_step * 9.64853329045427e-3, dtype)
    F = tl.mul(F, half_time_step_const)
    V = tl.fma(F, inv_M, V)
    tl.store(V_ptr + offsets, V, mask=mask)


# Section: Python entry points
def vv1(
        X: th.Tensor,
        V: th.Tensor,
        F: th.Tensor,
        inv_M: th.Tensor,
        time_step: float,
        block_size: int = DEFAULT_ELEMENTWISE_BLOCK,
) -> None:
    """Launch the first velocity-Verlet half-step.

    Args:
        X: flat coordinates, updated in place.
        V: flat velocities, updated in place.
        F: flat forces of the previous step (read-only).
        inv_M: flat inverse masses.
        time_step: MD time step in fs.
        block_size: Triton block size.

    Returns:
        None.
    """
    n_elem = X.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    velocity_verlet_1_kernel[grids](
        X, V, F, inv_M,
        time_step,
        n_elem,
        block_size,
    )


def vv2(
        V: th.Tensor,
        F: th.Tensor,
        atom_masks: th.Tensor,
        inv_M: th.Tensor,
        time_step: float,
        block_size: int = DEFAULT_ELEMENTWISE_BLOCK,
) -> None:
    """Launch the second velocity-Verlet half-step.

    Args:
        V: flat velocities, updated in place.
        F: flat forces, masked and overwritten in place.
        atom_masks: selective-dynamics mask.
        inv_M: flat inverse masses.
        time_step: MD time step in fs.
        block_size: Triton block size.

    Returns:
        None.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    velocity_verlet_2_kernel[grids](
        V, F, atom_masks, inv_M,
        time_step,
        n_elem,
        block_size,
    )
