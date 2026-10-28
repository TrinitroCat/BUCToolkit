#  Copyright (c) 2024-2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: thermostats.py
#  Environment: Python 3.12

"""Fused Triton kernels of the NVT thermostats.

The kernels operate on flat contiguous buffers and support float16, float32,
and float64 states.  The element type is read from the pointers; per-batch
scalars use a "compute dtype" (float16 states accumulate in float32, float32
and float64 states use their own precision).  Fixed scalars of a run are
``tl.constexpr``; read-only tensors are marked ``tl.const``.

Energy conventions of :mod:`BUCToolkit.BatchMD._BaseMD`
------------------------------------------------------
* ``Ek = 0.5 * sum(m * v**2) * 103.642696562621738`` (eV);
* ``reduced_Ek = sum(m * v**2) * 103.642696562621738 = 2 * Ek``.

The Nose-Hoover thermostat advances ``p_iota`` from ``reduced_Ek``.

Batch layouts
-------------
Regular batch (``n_batch, n_atom, 3``)
    The flat buffer has ``n_chunk = n_atom * 3`` elements per batch, so the
    batch of a flat element is simply ``offset // n_chunk``.

Irregular batch (``1, sum_n_atom, 3``)
    The flat buffer concatenates the batches.  Two auxiliary tensors are
    needed:

    * ``batch_scatter`` maps every **atom** index (``flat_offset // 3``) to its
      batch id.  It is used by elementwise kernels that need a per-batch
      scalar.
    * ``batch_starts`` holds the cumulative flat start of every batch,
      ``[0, n_0*3, (n_0+n_1)*3, ...]``.  A reduction kernel for batch ``pid``
      scans ``[starts[pid], starts[pid+1])``.

    The division by 3 is the number of Cartesian components per atom; the
    reduction loops include the tail with ``mask = idx < end`` and load with
    ``other=0`` so that masked lanes contribute nothing to the energy.

Random numbers
--------------
Langevin white noise and the normal draw of CSVR are generated inside the
kernels with ``tl.randn(seed, offsets)``.  ``seed`` is a runtime scalar that
changes every MD step and is passed with ``do_not_specialize`` so the kernel is
compiled once.  The chi-square draw of CSVR keeps using the external torch
sampler because its non-integer degrees of freedom have no direct Triton
counterpart.
"""

import torch as th
import triton
import triton.language as tl

from .utils import DEFAULT_ELEMENTWISE_BLOCK, DEFAULT_REDUCTION_BLOCK


# Section: CSVR rescaling factor
@triton.jit(do_not_specialize=['seed'])
def csvr_alpha_kernel(
        V_ptr: tl.const,
        mass_ptr: tl.const,
        alpha_ptr,
        ek_target_ptr: tl.const,
        free_degree_ptr: tl.const,
        chi2_ptr: tl.const,
        starts_ptr: tl.const,
        n_chunk: tl.constexpr,
        sqrt_c: tl.constexpr,
        one_sub_c: tl.constexpr,
        epsK: tl.constexpr,
        seed,
        IS_IRREGULAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Compute one CSVR rescaling factor per batch.

    The kernel reduces ``2 * Ek`` of every batch, evaluates

        f = (1 - c) * EK_TARGET / (N_f * K)
        alpha^2 = (sqrt(c) + sqrt(f) * R)^2 + f * Chi2
        alpha = sqrt(max(alpha^2, epsK))

    and stores ``alpha`` for the following velocity scaling.

    Args:
        V_ptr: flat velocities (read-only here).
        mass_ptr: atomic masses broadcast to ``V``.
        alpha_ptr: per-batch output factors.
        ek_target_ptr: per-batch target kinetic energy.
        free_degree_ptr: per-batch number of degrees of freedom.
        chi2_ptr: per-batch chi-square samples from the external torch sampler.
        starts_ptr: cumulative flat starts for an irregular batch; unused for
            a regular batch.
        n_chunk: flat elements per batch for a regular batch; unused for an
            irregular batch.
        sqrt_c: ``sqrt(exp(-dt / tau))``.
        one_sub_c: ``1 - exp(-dt / tau)``.
        epsK: kinetic-energy floor.
        seed: runtime RNG seed, changed every MD step.  ``do_not_specialize``
            keeps the kernel compiled across seed values.
        IS_IRREGULAR: whether the batch is the concatenated irregular layout.
        BLOCK_SIZE: Triton block size of the reduction loop.

    Returns:
        None. ``alpha_ptr`` is written in place.

    Note:
        ``starts_ptr`` is a dummy pointer for a regular batch; the
        ``IS_IRREGULAR`` branch is resolved at compile time and the dummy is
        never loaded.
    """
    # Section: dtype selection
    dtype = V_ptr.dtype.element_ty
    alpha_dtype = alpha_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype

    # Section: resolve the flat element range of this batch
    pid = tl.program_id(axis=0)
    if IS_IRREGULAR:
        # Indirect addressing: the batch owns [start, end) in the flat buffer.
        start = tl.load(starts_ptr + pid)
        end = tl.load(starts_ptr + pid + 1)
    else:
        # Regular layout: every batch owns exactly n_chunk contiguous elements.
        start = pid * n_chunk
        end = start + n_chunk

    # Section: reduce sum(m * v^2) over the batch
    # A vector accumulator keeps the loop-carried type stable and lets Triton
    # reduce once at the end.
    acc = tl.zeros([BLOCK_SIZE], dtype=calc)
    for offset in range(start, end, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        idx = tl.max_contiguous(idx, BLOCK_SIZE)
        mask = idx < end
        v = tl.load(V_ptr + idx, mask=mask, other=0.0).to(calc)
        m = tl.load(mass_ptr + idx, mask=mask, other=0.0).to(calc)
        acc += m * v * v
    Ek = tl.mul(tl.sum(acc), tl.full([], 0.5 * 103.642696562621738, calc))

    # Section: CSVR scalar chain
    epsK_calc = tl.full([], epsK, calc)
    K = tl.maximum(Ek, epsK_calc)
    ek_target = tl.load(ek_target_ptr + pid).to(calc)
    free_degree = tl.load(free_degree_ptr + pid).to(calc)
    f = tl.full([], one_sub_c, calc) * ek_target / (free_degree * K)
    sqrt_f = tl.sqrt(tl.maximum(f, tl.full([], 0.0, calc)))
    # In-kernel normal draw: one scalar per batch, deterministic in
    # (seed, pid) and independent of the external torch generator.
    randn = tl.randn(seed, pid).to(calc)
    chi2 = tl.load(chi2_ptr + pid).to(calc)
    alpha2 = tl.full([], sqrt_c, calc) + sqrt_f * randn
    alpha2 = alpha2 * alpha2
    alpha2 = alpha2 + f * chi2
    alpha2 = tl.maximum(alpha2, epsK_calc)
    tl.store(alpha_ptr + pid, tl.sqrt(alpha2).to(alpha_dtype))


# Section: velocity rescaling by a per-batch target
@triton.jit
def vr_scale_kernel(
        V_ptr,
        ek_ptr: tl.const,
        ek_target_ptr: tl.const,
        scatter_ptr: tl.const,
        n_chunk: tl.constexpr,
        size: tl.constexpr,
        IS_IRREGULAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Scale velocities by ``sqrt(EK_TARGET / Ek)`` of their batch.

    Args:
        V_ptr: flat velocities, updated in place.
        ek_ptr: per-batch current kinetic energy.
        ek_target_ptr: per-batch target kinetic energy.
        scatter_ptr: atom -> batch map for an irregular batch; a dummy pointer
            for a regular batch.
        n_chunk: flat elements per batch for a regular batch.
        size: total number of flat elements.
        IS_IRREGULAR: whether the batch is irregular.
        BLOCK_SIZE: Triton block size.

    Returns:
        None. ``V`` is updated in place.

    Note:
        For an irregular batch, ``offsets // 3`` converts a flat element index
        to its atom index before the scatter load.  The mask keeps the tail of
        the last block from touching out-of-range elements.
    """
    dtype = V_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < size

    # Section: resolve the batch id of every flat element
    if IS_IRREGULAR:
        batch_id = tl.load(scatter_ptr + offsets // 3, mask=mask, other=0)
    else:
        batch_id = offsets // n_chunk

    # Section: load, scale, and store
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    ek = tl.load(ek_ptr + batch_id, mask=mask, other=1.0).to(calc)
    ek_target = tl.load(ek_target_ptr + batch_id, mask=mask, other=0.0).to(calc)
    scale = tl.sqrt(ek_target / ek).to(dtype)
    V = tl.mul(V, scale)
    tl.store(V_ptr + offsets, V, mask=mask)


# Section: velocity scaling by one factor per batch
@triton.jit
def scale_by_batch_kernel(
        V_ptr,
        scale_ptr: tl.const,
        scatter_ptr: tl.const,
        n_chunk: tl.constexpr,
        size: tl.constexpr,
        IS_IRREGULAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Scale velocities by one factor per batch.

    Args:
        V_ptr: flat velocities, updated in place.
        scale_ptr: per-batch scaling factors.
        scatter_ptr: atom -> batch map for an irregular batch; a dummy pointer
            for a regular batch.
        n_chunk: flat elements per batch for a regular batch.
        size: total number of flat elements.
        IS_IRREGULAR: whether the batch is irregular.
        BLOCK_SIZE: Triton block size.

    Returns:
        None. ``V`` is updated in place.
    """
    dtype = V_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < size

    # Section: resolve the batch id of every flat element
    if IS_IRREGULAR:
        batch_id = tl.load(scatter_ptr + offsets // 3, mask=mask, other=0)
    else:
        batch_id = offsets // n_chunk

    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + batch_id, mask=mask, other=1.0).to(dtype)
    V = tl.mul(V, scale)
    tl.store(V_ptr + offsets, V, mask=mask)


# Section: Langevin BAOAB middle step
@triton.jit(do_not_specialize=['seed'])
def langevin_middle_kernel(
        X_ptr,
        V_ptr,
        F_ptr: tl.const,
        inv_M_ptr: tl.const,
        sigma_ptr: tl.const,
        seed,
        alpha: tl.constexpr,
        half_time_step_const: tl.constexpr,
        raw_half_time_step_const: tl.constexpr,
        size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Fused BAOAB drift, Ornstein-Uhlenbeck, and drift sub-step.

    The kernel performs

        V0 = V + h * F * inv_M
        V1 = alpha * V0 + sigma * N(0, 1)
        X  = X + h * V0 + h * V1

    where ``h`` is the half time step in BUCToolkit velocity units.

    Args:
        X_ptr: flat coordinates, updated in place.
        V_ptr: flat velocities, updated in place.
        F_ptr: flat masked forces from the previous step (read-only).
        inv_M_ptr: flat inverse masses.
        sigma_ptr: per-element noise amplitude.
        seed: runtime RNG seed, changed every MD step.  The white noise is
            generated in the kernel, avoiding an external ``randn_like``.
        alpha: Langevin damping factor ``exp(-damping_coeff * dt)``.
        half_time_step_const: ``0.5 * dt * 9.64853329045427e-3``.
        raw_half_time_step_const: ``0.5 * dt``.
        size: total number of flat elements.
        BLOCK_SIZE: Triton block size.

    Returns:
        None. ``X`` and ``V`` are updated in place.

    Note:
        ``tl.randn`` returns float32; the noise is cast to the state dtype
        before the fused multiply-add.
    """
    dtype = V_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < size

    # Section: load the current state and the noise amplitude
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)
    sigma = tl.load(sigma_ptr + offsets, mask=mask, other=0.0)

    # Section: first drift with the previous force
    V0 = tl.fma(tl.mul(F, tl.full([], half_time_step_const, dtype)), inv_M, V)

    # Section: Ornstein-Uhlenbeck update with in-kernel white noise
    noise = tl.randn(seed, offsets).to(dtype)
    V1 = tl.fma(V0, tl.full([], alpha, dtype), tl.mul(sigma, noise))

    # Section: two half drifts, taken with V0 and V1 respectively
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    X = tl.fma(V0, tl.full([], raw_half_time_step_const, dtype), X)
    X = tl.fma(V1, tl.full([], raw_half_time_step_const, dtype), X)

    tl.store(V_ptr + offsets, V1, mask=mask)
    tl.store(X_ptr + offsets, X, mask=mask)


# Section: Nose-Hoover first half-step
@triton.jit
def nose_half_step_1_kernel(
        X_ptr,
        V_ptr,
        F_ptr: tl.const,
        inv_M_ptr: tl.const,
        p_iota_ptr: tl.const,
        scatter_ptr: tl.const,
        time_step: tl.constexpr,
        half_time_step_const: tl.constexpr,
        raw_half_time_step_const: tl.constexpr,
        n_chunk: tl.constexpr,
        size: tl.constexpr,
        IS_IRREGULAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Nose-Hoover first half-step: velocity update, throttle, and drift.

    Args:
        X_ptr: flat coordinates, updated in place.
        V_ptr: flat velocities, updated in place.
        F_ptr: flat masked forces from the previous step (read-only).
        inv_M_ptr: flat inverse masses.
        p_iota_ptr: per-batch thermostat variable (read-only here).
        scatter_ptr: atom -> batch map for an irregular batch; a dummy pointer
            for a regular batch.
        time_step: MD time step.
        half_time_step_const: ``0.5 * dt * 9.64853329045427e-3``.
        raw_half_time_step_const: ``0.5 * dt``.
        n_chunk: flat elements per batch for a regular batch.
        size: total number of flat elements.
        IS_IRREGULAR: whether the batch is irregular.
        BLOCK_SIZE: Triton block size.

    Returns:
        None. ``X`` and ``V`` are updated in place.

    Note:
        ``exp`` is evaluated in ``calc`` because Triton only supports float32
        and float64 exponentials; the throttle is cast back to the state dtype.
    """
    dtype = V_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(offsets, BLOCK_SIZE)
    mask = offsets < size

    # Section: first half kick with the previous force
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)
    V = tl.fma(tl.mul(F, tl.full([], half_time_step_const, dtype)), inv_M, V)

    # Section: throttle with the old thermostat variable
    if IS_IRREGULAR:
        batch_id = tl.load(scatter_ptr + offsets // 3, mask=mask, other=0)
    else:
        batch_id = offsets // n_chunk
    iota = tl.load(p_iota_ptr + batch_id, mask=mask, other=0.0).to(calc)
    throttle = tl.exp(tl.mul(iota, tl.full([], -raw_half_time_step_const, calc))).to(dtype)
    V = tl.mul(V, throttle)
    tl.store(V_ptr + offsets, V, mask=mask)

    # Section: drift with the throttled velocity
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    X = tl.fma(V, tl.full([], time_step, dtype), X)
    tl.store(X_ptr + offsets, X, mask=mask)


# Section: Nose-Hoover reductions and thermostat updates
@triton.jit
def nose_reduce_update_kernel(
        V_ptr: tl.const,
        mass_ptr: tl.const,
        p_iota_ptr,
        free_degree_ptr: tl.const,
        smass_ptr: tl.const,
        starts_ptr: tl.const,
        n_chunk: tl.constexpr,
        raw_half_time_step_const: tl.constexpr,
        target_kT: tl.constexpr,
        IS_IRREGULAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Reduce ``2 * Ek`` per batch and advance ``p_iota`` by a half step.

    Args:
        V_ptr: flat velocities (read-only here).
        mass_ptr: atomic masses broadcast to ``V``.
        p_iota_ptr: per-batch thermostat variable, updated in place.
        free_degree_ptr: per-batch number of degrees of freedom.
        smass_ptr: per-batch thermostat mass.
        starts_ptr: cumulative flat starts for an irregular batch; unused for
            a regular batch.
        n_chunk: flat elements per batch for a regular batch.
        raw_half_time_step_const: ``0.5 * dt``.
        target_kT: target ``kB * T``.
        IS_IRREGULAR: whether the batch is irregular.
        BLOCK_SIZE: Triton block size of the reduction loop.

    Returns:
        None. ``p_iota_ptr`` is updated in place.
    """
    dtype = V_ptr.dtype.element_ty
    iota_dtype = p_iota_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype

    # Section: resolve the flat element range of this batch
    pid = tl.program_id(axis=0)
    if IS_IRREGULAR:
        start = tl.load(starts_ptr + pid)
        end = tl.load(starts_ptr + pid + 1)
    else:
        start = pid * n_chunk
        end = start + n_chunk

    # Section: reduce sum(m * v^2), then convert to reduced_Ek = 2 * Ek
    acc = tl.zeros([BLOCK_SIZE], dtype=calc)
    for offset in range(start, end, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        idx = tl.max_contiguous(idx, BLOCK_SIZE)
        mask = idx < end
        v = tl.load(V_ptr + idx, mask=mask, other=0.0).to(calc)
        m = tl.load(mass_ptr + idx, mask=mask, other=0.0).to(calc)
        acc += m * v * v
    reduced_Ek = tl.mul(tl.sum(acc), tl.full([], 103.642696562621738, calc))

    # Section: p_iota += 0.5 * dt * (reduced_Ek - N_f * kT) / smass
    free_degree = tl.load(free_degree_ptr + pid).to(calc)
    smass = tl.load(smass_ptr + pid).to(calc)
    p_iota = tl.load(p_iota_ptr + pid).to(calc)
    p_iota = p_iota + tl.full([], raw_half_time_step_const, calc) * (
        reduced_Ek - free_degree * tl.full([], target_kT, calc)
    ) / smass
    tl.store(p_iota_ptr + pid, p_iota.to(iota_dtype))


@triton.jit
def nose_throttle_reduce_update_kernel(
        V_ptr,
        mass_ptr: tl.const,
        p_iota_ptr,
        free_degree_ptr: tl.const,
        smass_ptr: tl.const,
        starts_ptr: tl.const,
        n_chunk: tl.constexpr,
        raw_half_time_step_const: tl.constexpr,
        target_kT: tl.constexpr,
        IS_IRREGULAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Throttle velocities, reduce ``2 * Ek``, and advance ``p_iota``.

    Args:
        V_ptr: flat velocities, throttled in place.
        mass_ptr: atomic masses broadcast to ``V``.
        p_iota_ptr: per-batch thermostat variable, updated in place.
        free_degree_ptr: per-batch number of degrees of freedom.
        smass_ptr: per-batch thermostat mass.
        starts_ptr: cumulative flat starts for an irregular batch; unused for
            a regular batch.
        n_chunk: flat elements per batch for a regular batch.
        raw_half_time_step_const: ``0.5 * dt``.
        target_kT: target ``kB * T``.
        IS_IRREGULAR: whether the batch is irregular.
        BLOCK_SIZE: Triton block size of the reduction loop.

    Returns:
        None. ``V`` and ``p_iota_ptr`` are updated in place.

    Note:
        The throttle factor is computed once per batch from the freshly
        updated ``p_iota`` and applied while the loop reduces the throttled
        velocities, so the two operations share one pass over ``V``.
    """
    dtype = V_ptr.dtype.element_ty
    iota_dtype = p_iota_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype

    # Section: resolve the flat element range of this batch
    pid = tl.program_id(axis=0)
    if IS_IRREGULAR:
        start = tl.load(starts_ptr + pid)
        end = tl.load(starts_ptr + pid + 1)
    else:
        start = pid * n_chunk
        end = start + n_chunk

    # Section: throttle factor exp(-p_iota * 0.5 * dt), computed in calc
    iota = tl.load(p_iota_ptr + pid).to(calc)
    throttle = tl.exp(
        tl.mul(iota, tl.full([], -raw_half_time_step_const, calc))
    ).to(dtype)

    # Section: apply the throttle and reduce the throttled velocities
    acc = tl.zeros([BLOCK_SIZE], dtype=calc)
    for offset in range(start, end, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        idx = tl.max_contiguous(idx, BLOCK_SIZE)
        mask = idx < end
        v = tl.load(V_ptr + idx, mask=mask, other=0.0)
        v = tl.mul(v, throttle)
        tl.store(V_ptr + idx, v, mask=mask)
        m = tl.load(mass_ptr + idx, mask=mask, other=0.0).to(calc)
        v = v.to(calc)
        acc += m * v * v
    reduced_Ek = tl.mul(tl.sum(acc), tl.full([], 103.642696562621738, calc))

    # Section: p_iota += 0.5 * dt * (reduced_Ek - N_f * kT) / smass
    free_degree = tl.load(free_degree_ptr + pid).to(calc)
    smass = tl.load(smass_ptr + pid).to(calc)
    p_iota = tl.load(p_iota_ptr + pid).to(calc)
    p_iota = p_iota + tl.full([], raw_half_time_step_const, calc) * (
        reduced_Ek - free_degree * tl.full([], target_kT, calc)
    ) / smass
    tl.store(p_iota_ptr + pid, p_iota.to(iota_dtype))


# Section: Python entry points
def csvr_alpha(
        V: th.Tensor,
        masses: th.Tensor,
        alpha: th.Tensor,
        ek_target: th.Tensor,
        free_degree: th.Tensor,
        chi2: th.Tensor,
        batch_starts: th.Tensor | None,
        n_chunk: int,
        sqrt_c: float,
        one_sub_c: float,
        epsK: float,
        seed: int,
        is_irregular: bool,
        block_size: int = DEFAULT_REDUCTION_BLOCK,
) -> None:
    """Compute the CSVR rescaling factor of every batch.

    Args:
        V: flat velocities.
        masses: atomic masses broadcast to ``V``.
        alpha: destination buffer of the per-batch factors.
        ek_target: per-batch target kinetic energy.
        free_degree: per-batch number of degrees of freedom.
        chi2: per-batch chi-square samples from the torch sampler.
        batch_starts: flat start offsets for an irregular batch, or ``None``.
        n_chunk: flat elements per batch for a regular batch.
        sqrt_c: ``sqrt(exp(-dt / tau))``.
        one_sub_c: ``1 - exp(-dt / tau)``.
        epsK: kinetic-energy floor.
        seed: runtime RNG seed of this step.
        is_irregular: whether ``batch_starts`` describes an irregular batch.
        block_size: Triton block size of the reduction loop.

    Returns:
        None. ``alpha`` is written in place.
    """
    n_batch = alpha.numel()
    csvr_alpha_kernel[(n_batch,)](
        V, masses, alpha,
        ek_target, free_degree, chi2,
        batch_starts if is_irregular else V,
        n_chunk, sqrt_c, one_sub_c, epsK,
        seed, is_irregular, block_size,
        num_warps=8,
    )


def vr_scale(
        V: th.Tensor,
        Ek: th.Tensor,
        ek_target: th.Tensor,
        batch_scatter: th.Tensor | None,
        n_chunk: int,
        is_irregular: bool,
        block_size: int = DEFAULT_ELEMENTWISE_BLOCK,
) -> None:
    """Scale velocities by ``sqrt(EK_TARGET / Ek)`` of their batch.

    Args:
        V: flat velocities, updated in place.
        Ek: per-batch current kinetic energy.
        ek_target: per-batch target kinetic energy.
        batch_scatter: atom -> batch map for an irregular batch, or ``None``.
        n_chunk: flat elements per batch for a regular batch.
        is_irregular: whether ``batch_scatter`` describes an irregular batch.
        block_size: Triton block size.

    Returns:
        None.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    vr_scale_kernel[grids](
        V, Ek, ek_target,
        batch_scatter if is_irregular else V,
        n_chunk, n_elem, is_irregular,
        block_size,
    )


def scale_by_batch(
        V: th.Tensor,
        scale: th.Tensor,
        batch_scatter: th.Tensor | None,
        n_chunk: int,
        is_irregular: bool,
        block_size: int = DEFAULT_ELEMENTWISE_BLOCK,
) -> None:
    """Scale velocities by one factor per batch.

    Args:
        V: flat velocities, updated in place.
        scale: per-batch scaling factors.
        batch_scatter: atom -> batch map for an irregular batch, or ``None``.
        n_chunk: flat elements per batch for a regular batch.
        is_irregular: whether ``batch_scatter`` describes an irregular batch.
        block_size: Triton block size.

    Returns:
        None.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    scale_by_batch_kernel[grids](
        V, scale,
        batch_scatter if is_irregular else V,
        n_chunk, n_elem, is_irregular,
        block_size,
    )


def langevin_middle(
        X: th.Tensor,
        V: th.Tensor,
        F: th.Tensor,
        inv_M: th.Tensor,
        sigma: th.Tensor,
        seed: int,
        alpha: float,
        half_time_step_const: float,
        raw_half_time_step_const: float,
        block_size: int = DEFAULT_ELEMENTWISE_BLOCK,
) -> None:
    """Fused Langevin BAOAB drift / noise / drift sub-step.

    Args:
        X: flat coordinates, updated in place.
        V: flat velocities, updated in place.
        F: flat masked forces.
        inv_M: flat inverse masses.
        sigma: per-element noise amplitude.
        seed: runtime RNG seed of this step.
        alpha: Langevin damping factor.
        half_time_step_const: ``0.5 * dt * 9.64853329045427e-3``.
        raw_half_time_step_const: ``0.5 * dt``.
        block_size: Triton block size.

    Returns:
        None.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    langevin_middle_kernel[grids](
        X, V, F, inv_M, sigma, seed,
        alpha, half_time_step_const, raw_half_time_step_const,
        n_elem, block_size,
    )


def nose_half_step_1(
        X: th.Tensor,
        V: th.Tensor,
        F: th.Tensor,
        inv_M: th.Tensor,
        p_iota: th.Tensor,
        batch_scatter: th.Tensor | None,
        n_chunk: int,
        time_step: float,
        half_time_step_const: float,
        raw_half_time_step_const: float,
        is_irregular: bool,
        block_size: int = DEFAULT_ELEMENTWISE_BLOCK,
) -> None:
    """Nose-Hoover first half-step: velocity update, throttle, and drift.

    Args:
        X: flat coordinates, updated in place.
        V: flat velocities, updated in place.
        F: flat masked forces.
        inv_M: flat inverse masses.
        p_iota: per-batch thermostat variable, flat shape ``(n_batch,)``.
        batch_scatter: atom -> batch map for an irregular batch, or ``None``.
        n_chunk: flat elements per batch for a regular batch.
        time_step: MD time step.
        half_time_step_const: ``0.5 * dt * 9.64853329045427e-3``.
        raw_half_time_step_const: ``0.5 * dt``.
        is_irregular: whether ``batch_scatter`` describes an irregular batch.
        block_size: Triton block size.

    Returns:
        None.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    nose_half_step_1_kernel[grids](
        X, V, F, inv_M, p_iota,
        batch_scatter if is_irregular else V,
        time_step, half_time_step_const, raw_half_time_step_const,
        n_chunk, n_elem, is_irregular,
        block_size,
    )


def nose_reduce_update(
        V: th.Tensor,
        masses: th.Tensor,
        p_iota: th.Tensor,
        free_degree: th.Tensor,
        smass: th.Tensor,
        batch_starts: th.Tensor | None,
        n_chunk: int,
        raw_half_time_step_const: float,
        target_kT: float,
        is_irregular: bool,
        block_size: int = DEFAULT_REDUCTION_BLOCK,
) -> None:
    """Reduce ``2 * Ek`` per batch and advance ``p_iota`` by a half step.

    Args:
        V: flat velocities.
        masses: atomic masses broadcast to ``V``.
        p_iota: per-batch thermostat variable, flat shape ``(n_batch,)``.
        free_degree: per-batch number of degrees of freedom.
        smass: per-batch thermostat mass.
        batch_starts: flat start offsets for an irregular batch, or ``None``.
        n_chunk: flat elements per batch for a regular batch.
        raw_half_time_step_const: ``0.5 * dt``.
        target_kT: target ``kB * T``.
        is_irregular: whether ``batch_starts`` describes an irregular batch.
        block_size: Triton block size of the reduction loop.

    Returns:
        None.
    """
    n_batch = p_iota.numel()
    nose_reduce_update_kernel[(n_batch,)](
        V, masses, p_iota, free_degree, smass,
        batch_starts if is_irregular else V,
        n_chunk, raw_half_time_step_const, target_kT,
        is_irregular, block_size,
        num_warps=8,
    )


def nose_throttle_reduce_update(
        V: th.Tensor,
        masses: th.Tensor,
        p_iota: th.Tensor,
        free_degree: th.Tensor,
        smass: th.Tensor,
        batch_starts: th.Tensor | None,
        n_chunk: int,
        raw_half_time_step_const: float,
        target_kT: float,
        is_irregular: bool,
        block_size: int = DEFAULT_REDUCTION_BLOCK,
) -> None:
    """Throttle velocities, reduce ``2 * Ek``, and advance ``p_iota``.

    Args:
        V: flat velocities, updated in place.
        masses: atomic masses broadcast to ``V``.
        p_iota: per-batch thermostat variable, flat shape ``(n_batch,)``.
        free_degree: per-batch number of degrees of freedom.
        smass: per-batch thermostat mass.
        batch_starts: flat start offsets for an irregular batch, or ``None``.
        n_chunk: flat elements per batch for a regular batch.
        raw_half_time_step_const: ``0.5 * dt``.
        target_kT: target ``kB * T``.
        is_irregular: whether ``batch_starts`` describes an irregular batch.
        block_size: Triton block size of the reduction loop.

    Returns:
        None.
    """
    n_batch = p_iota.numel()
    nose_throttle_reduce_update_kernel[(n_batch,)](
        V, masses, p_iota, free_degree, smass,
        batch_starts if is_irregular else V,
        n_chunk, raw_half_time_step_const, target_kT,
        is_irregular, block_size,
        num_warps=8,
    )
