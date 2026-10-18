"""Fused Triton kernels of the NVT thermostats.

The kernels operate on flat contiguous buffers and support float16, float32 and
float64 states.  Every fixed scalar of a run is a ``tl.constexpr`` and every
read-only tensor pointer is marked ``tl.const``.  Per-batch reductions use a
float32 accumulator for float16 states and the native precision otherwise.

Energy unit conventions follow :mod:`BUCToolkit.BatchMD._BaseMD`:

* ``Ek = 0.5 * sum(m * v**2) * 103.642696562621738`` (eV),
* ``reduced_Ek = sum(m * v**2) * 103.642696562621738 = 2 * Ek``.
"""

import torch as th
import triton
import triton.language as tl


@triton.jit
def csvr_alpha_kernel(
        V_ptr: tl.const,
        mass_ptr: tl.const,
        alpha_ptr,
        ek_target_ptr: tl.const,
        free_degree_ptr: tl.const,
        randn_ptr: tl.const,
        chi2_ptr: tl.const,
        starts_ptr: tl.const,
        n_chunk: tl.constexpr,
        sqrt_c: tl.constexpr,
        one_sub_c: tl.constexpr,
        epsK: tl.constexpr,
        IS_IRREGULAR: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Compute one CSVR rescaling factor per batch."""
    dtype = V_ptr.dtype.element_ty
    alpha_dtype = alpha_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype
    pid = tl.program_id(axis=0)
    if IS_IRREGULAR:
        start = tl.load(starts_ptr + pid)
        end = tl.load(starts_ptr + pid + 1)
    else:
        start = pid * n_chunk
        end = start + n_chunk
    acc = tl.zeros([BLOCK_SIZE], dtype=calc)
    for offset in range(start, end, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < end
        v = tl.load(V_ptr + idx, mask=mask, other=0.0).to(calc)
        m = tl.load(mass_ptr + idx, mask=mask, other=0.0).to(calc)
        acc += m * v * v
    Ek = tl.mul(tl.sum(acc), tl.full([], 0.5 * 103.642696562621738, calc))
    epsK_calc = tl.full([], epsK, calc)
    K = tl.maximum(Ek, epsK_calc)
    ek_target = tl.load(ek_target_ptr + pid).to(calc)
    free_degree = tl.load(free_degree_ptr + pid).to(calc)
    f = tl.full([], one_sub_c, calc) * ek_target / (free_degree * K)
    sqrt_f = tl.sqrt(tl.maximum(f, tl.full([], 0.0, calc)))
    alpha2 = tl.full([], sqrt_c, calc) + sqrt_f * tl.load(randn_ptr + pid).to(calc)
    alpha2 = alpha2 * alpha2
    alpha2 = alpha2 + f * tl.load(chi2_ptr + pid).to(calc)
    alpha2 = tl.maximum(alpha2, epsK_calc)
    tl.store(alpha_ptr + pid, tl.sqrt(alpha2).to(alpha_dtype))


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
    """Scale velocities by ``sqrt(EK_TARGET / Ek)`` of their batch."""
    dtype = V_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    if IS_IRREGULAR:
        batch_id = tl.load(scatter_ptr + offsets // 3, mask=mask, other=0)
    else:
        batch_id = offsets // n_chunk
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    ek = tl.load(ek_ptr + batch_id, mask=mask, other=1.0).to(calc)
    ek_target = tl.load(ek_target_ptr + batch_id, mask=mask, other=0.0).to(calc)
    scale = tl.sqrt(ek_target / ek).to(dtype)
    V = tl.mul(V, scale)
    tl.store(V_ptr + offsets, V, mask=mask)


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
    """Scale velocities by one factor per batch."""
    dtype = V_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    if IS_IRREGULAR:
        batch_id = tl.load(scatter_ptr + offsets // 3, mask=mask, other=0)
    else:
        batch_id = offsets // n_chunk
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + batch_id, mask=mask, other=1.0).to(dtype)
    V = tl.mul(V, scale)
    tl.store(V_ptr + offsets, V, mask=mask)


@triton.jit
def langevin_middle_kernel(
        X_ptr,
        V_ptr,
        F_ptr: tl.const,
        inv_M_ptr: tl.const,
        sigma_ptr: tl.const,
        noise_ptr: tl.const,
        alpha: tl.constexpr,
        half_time_step_const: tl.constexpr,
        raw_half_time_step_const: tl.constexpr,
        size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    """Fused BAOAB drift + Ornstein-Uhlenbeck + drift sub-step."""
    dtype = V_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)
    sigma = tl.load(sigma_ptr + offsets, mask=mask, other=0.0)
    noise = tl.load(noise_ptr + offsets, mask=mask, other=0.0)
    V0 = tl.fma(tl.mul(F, tl.full([], half_time_step_const, dtype)), inv_M, V)
    V1 = tl.fma(V0, tl.full([], alpha, dtype), tl.mul(sigma, noise))
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    X = tl.fma(V0, tl.full([], raw_half_time_step_const, dtype), X)
    X = tl.fma(V1, tl.full([], raw_half_time_step_const, dtype), X)
    tl.store(V_ptr + offsets, V1, mask=mask)
    tl.store(X_ptr + offsets, X, mask=mask)


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
    """Nose-Hoover first half-step: velocity update, throttle and drift."""
    dtype = V_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    V = tl.load(V_ptr + offsets, mask=mask, other=0.0)
    F = tl.load(F_ptr + offsets, mask=mask, other=0.0)
    inv_M = tl.load(inv_M_ptr + offsets, mask=mask, other=0.0)
    V = tl.fma(tl.mul(F, tl.full([], half_time_step_const, dtype)), inv_M, V)
    if IS_IRREGULAR:
        batch_id = tl.load(scatter_ptr + offsets // 3, mask=mask, other=0)
    else:
        batch_id = offsets // n_chunk
    iota = tl.load(p_iota_ptr + batch_id, mask=mask, other=0.0).to(calc)
    throttle = tl.exp(tl.mul(iota, tl.full([], -raw_half_time_step_const, calc))).to(dtype)
    V = tl.mul(V, throttle)
    tl.store(V_ptr + offsets, V, mask=mask)
    X = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    X = tl.fma(V, tl.full([], time_step, dtype), X)
    tl.store(X_ptr + offsets, X, mask=mask)


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
    """Nose-Hoover: reduce ``2 * Ek`` of each batch and update ``p_iota``."""
    dtype = V_ptr.dtype.element_ty
    iota_dtype = p_iota_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype
    pid = tl.program_id(axis=0)
    if IS_IRREGULAR:
        start = tl.load(starts_ptr + pid)
        end = tl.load(starts_ptr + pid + 1)
    else:
        start = pid * n_chunk
        end = start + n_chunk
    acc = tl.zeros([BLOCK_SIZE], dtype=calc)
    for offset in range(start, end, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < end
        v = tl.load(V_ptr + idx, mask=mask, other=0.0).to(calc)
        m = tl.load(mass_ptr + idx, mask=mask, other=0.0).to(calc)
        acc += m * v * v
    reduced_Ek = tl.mul(tl.sum(acc), tl.full([], 103.642696562621738, calc))
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
    """Nose-Hoover: throttle velocities, reduce ``2 * Ek`` and update iota."""
    dtype = V_ptr.dtype.element_ty
    iota_dtype = p_iota_ptr.dtype.element_ty
    if dtype == tl.float16:
        calc = tl.float32
    else:
        calc = dtype
    pid = tl.program_id(axis=0)
    if IS_IRREGULAR:
        start = tl.load(starts_ptr + pid)
        end = tl.load(starts_ptr + pid + 1)
    else:
        start = pid * n_chunk
        end = start + n_chunk
    iota = tl.load(p_iota_ptr + pid).to(calc)
    throttle = tl.exp(
        tl.mul(iota, tl.full([], -raw_half_time_step_const, calc))
    ).to(dtype)
    acc = tl.zeros([BLOCK_SIZE], dtype=calc)
    for offset in range(start, end, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < end
        v = tl.load(V_ptr + idx, mask=mask, other=0.0)
        v = tl.mul(v, throttle)
        tl.store(V_ptr + idx, v, mask=mask)
        m = tl.load(mass_ptr + idx, mask=mask, other=0.0).to(calc)
        v = v.to(calc)
        acc += m * v * v
    reduced_Ek = tl.mul(tl.sum(acc), tl.full([], 103.642696562621738, calc))
    free_degree = tl.load(free_degree_ptr + pid).to(calc)
    smass = tl.load(smass_ptr + pid).to(calc)
    p_iota = tl.load(p_iota_ptr + pid).to(calc)
    p_iota = p_iota + tl.full([], raw_half_time_step_const, calc) * (
        reduced_Ek - free_degree * tl.full([], target_kT, calc)
    ) / smass
    tl.store(p_iota_ptr + pid, p_iota.to(iota_dtype))


def csvr_alpha(
        V: th.Tensor,
        masses: th.Tensor,
        alpha: th.Tensor,
        ek_target: th.Tensor,
        free_degree: th.Tensor,
        randn: th.Tensor,
        chi2: th.Tensor,
        batch_starts: th.Tensor | None,
        n_chunk: int,
        sqrt_c: float,
        one_sub_c: float,
        epsK: float,
        is_irregular: bool,
        block_size: int = 1024,
) -> None:
    """Compute the CSVR rescaling factor of every batch.

    Args:
        V: contiguous velocities.
        masses: contiguous atomic masses broadcast to ``V``.
        alpha: destination buffer of the per-batch factors.
        ek_target: per-batch target kinetic energy.
        free_degree: per-batch number of degrees of freedom.
        randn: per-batch standard normal samples.
        chi2: per-batch chi-square samples.
        batch_starts: flat start offsets of each batch for irregular batches.
        n_chunk: number of flat elements per batch for regular batches.
        sqrt_c: ``sqrt(exp(-dt / tau))``.
        one_sub_c: ``1 - exp(-dt / tau)``.
        epsK: kinetic-energy floor.
        is_irregular: whether ``batch_starts`` describes an irregular batch.
        block_size: Triton block size of the reduction loop.

    Returns:
        None. ``alpha`` is written in place.
    """
    n_batch = alpha.numel()
    csvr_alpha_kernel[(n_batch,)](
        V, masses, alpha,
        ek_target, free_degree, randn, chi2,
        batch_starts if is_irregular else V,
        n_chunk, sqrt_c, one_sub_c, epsK,
        is_irregular, block_size,
        num_warps=8,
    )


def vr_scale(
        V: th.Tensor,
        Ek: th.Tensor,
        ek_target: th.Tensor,
        batch_scatter: th.Tensor | None,
        n_chunk: int,
        is_irregular: bool,
        block_size: int = 256,
) -> None:
    """Scale velocities by ``sqrt(EK_TARGET / Ek)`` of their batch.

    Args:
        V: contiguous velocities.
        Ek: per-batch kinetic energy.
        ek_target: per-batch target kinetic energy.
        batch_scatter: atom -> batch map for irregular batches.
        n_chunk: number of flat elements per batch for regular batches.
        is_irregular: whether ``batch_scatter`` describes an irregular batch.
        block_size: Triton block size.

    Returns:
        None. ``V`` is updated in place.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    vr_scale_kernel[grids](
        V, Ek, ek_target,
        batch_scatter if is_irregular else V,
        n_chunk, n_elem, is_irregular,
        BLOCK_SIZE=block_size,
    )


def scale_by_batch(
        V: th.Tensor,
        scale: th.Tensor,
        batch_scatter: th.Tensor | None,
        n_chunk: int,
        is_irregular: bool,
        block_size: int = 256,
) -> None:
    """Scale velocities by one factor per batch.

    Args:
        V: contiguous velocities.
        scale: per-batch scaling factors.
        batch_scatter: atom -> batch map for irregular batches.
        n_chunk: number of flat elements per batch for regular batches.
        is_irregular: whether ``batch_scatter`` describes an irregular batch.
        block_size: Triton block size.

    Returns:
        None. ``V`` is updated in place.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    scale_by_batch_kernel[grids](
        V, scale,
        batch_scatter if is_irregular else V,
        n_chunk, n_elem, is_irregular,
        BLOCK_SIZE=block_size,
    )


def langevin_middle(
        X: th.Tensor,
        V: th.Tensor,
        F: th.Tensor,
        inv_M: th.Tensor,
        sigma: th.Tensor,
        noise: th.Tensor,
        alpha: float,
        half_time_step_const: float,
        raw_half_time_step_const: float,
        block_size: int = 256,
) -> None:
    """Fused BAOAB velocity-Verlet / Langevin / velocity-Verlet sub-step.

    Args:
        X: contiguous coordinates.
        V: contiguous velocities.
        F: contiguous masked forces.
        inv_M: inverse atomic masses broadcast to ``V``.
        sigma: per-element noise amplitude.
        noise: standard normal samples with the shape of ``V``.
        alpha: Langevin damping factor.
        half_time_step_const: ``0.5 * dt * 9.64853329045427e-3``.
        raw_half_time_step_const: ``0.5 * dt``.
        block_size: Triton block size.

    Returns:
        None. ``X`` and ``V`` are updated in place.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    langevin_middle_kernel[grids](
        X, V, F, inv_M, sigma, noise,
        alpha, half_time_step_const, raw_half_time_step_const,
        n_elem,
        BLOCK_SIZE=block_size,
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
        block_size: int = 256,
) -> None:
    """Nose-Hoover first half-step: velocity update, throttle and drift.

    Args:
        X: contiguous coordinates.
        V: contiguous velocities.
        F: contiguous masked forces.
        inv_M: inverse atomic masses broadcast to ``V``.
        p_iota: per-batch thermostat variable, flat shape ``(n_batch,)``.
        batch_scatter: atom -> batch map for irregular batches.
        n_chunk: number of flat elements per batch for regular batches.
        time_step: MD time step.
        half_time_step_const: ``0.5 * dt * 9.64853329045427e-3``.
        raw_half_time_step_const: ``0.5 * dt``.
        is_irregular: whether ``batch_scatter`` describes an irregular batch.
        block_size: Triton block size.

    Returns:
        None. ``X`` and ``V`` are updated in place.
    """
    n_elem = V.numel()
    grids = lambda meta: (triton.cdiv(n_elem, meta["BLOCK_SIZE"]),)
    nose_half_step_1_kernel[grids](
        X, V, F, inv_M, p_iota,
        batch_scatter if is_irregular else V,
        time_step, half_time_step_const, raw_half_time_step_const,
        n_chunk, n_elem, is_irregular,
        BLOCK_SIZE=block_size,
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
        block_size: int = 1024,
) -> None:
    """Reduce ``2 * Ek`` per batch and advance ``p_iota`` by a half step.

    Args:
        V: contiguous velocities.
        masses: contiguous atomic masses broadcast to ``V``.
        p_iota: per-batch thermostat variable, flat shape ``(n_batch,)``.
        free_degree: per-batch number of degrees of freedom.
        smass: per-batch thermostat mass.
        batch_starts: flat start offsets of each batch for irregular batches.
        n_chunk: number of flat elements per batch for regular batches.
        raw_half_time_step_const: ``0.5 * dt``.
        target_kT: target ``kB * T``.
        is_irregular: whether ``batch_starts`` describes an irregular batch.
        block_size: Triton block size of the reduction loop.

    Returns:
        None. ``p_iota`` is updated in place.
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
        block_size: int = 1024,
) -> None:
    """Throttle velocities, reduce ``2 * Ek`` and advance ``p_iota``.

    Args:
        V: contiguous velocities.
        masses: contiguous atomic masses broadcast to ``V``.
        p_iota: per-batch thermostat variable, flat shape ``(n_batch,)``.
        free_degree: per-batch number of degrees of freedom.
        smass: per-batch thermostat mass.
        batch_starts: flat start offsets of each batch for irregular batches.
        n_chunk: number of flat elements per batch for regular batches.
        raw_half_time_step_const: ``0.5 * dt``.
        target_kT: target ``kB * T``.
        is_irregular: whether ``batch_starts`` describes an irregular batch.
        block_size: Triton block size of the reduction loop.

    Returns:
        None. ``V`` and ``p_iota`` are updated in place.
    """
    n_batch = p_iota.numel()
    nose_throttle_reduce_update_kernel[(n_batch,)](
        V, masses, p_iota, free_degree, smass,
        batch_starts if is_irregular else V,
        n_chunk, raw_half_time_step_const, target_kT,
        is_irregular, block_size,
        num_warps=8,
    )
