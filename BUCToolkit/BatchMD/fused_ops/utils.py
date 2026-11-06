#  Copyright (c) 2024-2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: utils.py
#  Environment: Python 3.12

"""Utilities shared by the fused Triton kernels.

The module owns three pieces of infrastructure:

* :class:`KernelLauncher` -- direct launch of a compiled Triton kernel;
* :func:`resolve_block_sizes` -- the ``BUCTOOLKIT_KERNEL_BLOCKSIZE`` policy;
* :func:`autotune_block_size` -- one-time block-size tuning for long runs.

It deliberately contains no MD state and no kernel definitions.
"""

import os
from typing import Any

import torch as th
from triton.runtime import driver as _triton_driver

# Fallbacks used when the run is too short to amortize autotuning.
DEFAULT_ELEMENTWISE_BLOCK = 256
DEFAULT_REDUCTION_BLOCK = 1024

# Candidate sets of the one-time autotuner.
ELEMENTWISE_CANDIDATES = (128, 256, 512, 1024)
REDUCTION_CANDIDATES = (256, 512, 1024, 2048)

# A long run amortizes the one-time compile/timing cost of the autotuner.
# Measured on the development GPU: the cold-cache autotune costs 0.1-0.4 s
# and block-size tuning saves 0-1 us/step for 512-2048 atoms; the break-even
# is about 1.2e5 steps at 2048 atoms, so 2e5 is a conservative cutoff.
AUTOTUNE_STEP_THRESHOLD = 200_000


class KernelLauncher:
    """Launch one Triton kernel through its compiled C launcher.

    Triton's ``JITFunction.run`` re-binds and re-specializes every argument on
    every call.  For the small and medium MD systems targeted here that Python
    bookkeeping costs more than the kernel itself.  The first call compiles (or
    retrieves) the kernel and launches it; later calls reuse the compiled
    kernel and skip the per-call binding.  A launcher is bound to one kernel
    and one grid, and is created once per run.

    If the installed Triton version exposes a different compiled-kernel layout,
    the launcher keeps using the public ``kernel[grid](...)`` path.

    Attributes:
        _kernel: Triton JIT function.
        _grid: fixed launch grid.
        _options: compilation options such as ``num_warps``.
        _compiled: cached compiled kernel, or ``None`` before the first call.
        _direct: whether the compiled kernel can be launched directly.
        _device: CUDA device index used for the stream lookup.
    """

    __slots__ = ('_kernel', '_grid', '_options', '_compiled', '_direct', '_device')

    def __init__(self, kernel, grid, **options) -> None:
        """Create the launcher.

        Args:
            kernel: Triton JIT function.
            grid: launch grid tuple.
            **options: compilation options such as ``num_warps``.

        Returns:
            None.
        """
        self._kernel = kernel
        self._grid = tuple(grid)
        self._options = options
        self._compiled = None
        self._direct = True
        self._device = th.cuda.current_device()

    def __call__(self, *args) -> None:
        """Launch the kernel with its full positional argument list.

        Args:
            *args: kernel arguments in parameter order, including the
                ``tl.constexpr`` parameters and the runtime RNG seed.

        Returns:
            None.
        """
        compiled = self._compiled
        if compiled is None:
            compiled = self._kernel.run(*args, grid=self._grid, warmup=False, **self._options)
            if all(hasattr(compiled, _name) for _name in ('run', 'function', 'packed_metadata')):
                self._compiled = compiled
            else:
                self._direct = False
            return
        if not self._direct:
            self._kernel[self._grid](*args, **self._options)
            return
        grid = self._grid
        stream = _triton_driver.active.get_current_stream(self._device)
        compiled.run(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1,
            stream,
            compiled.function,
            compiled.packed_metadata,
            None, None, None,
            *args,
        )


def resolve_block_sizes(max_step: int) -> tuple[int | None, int | None]:
    """Resolve the elementwise and reduction block sizes of one run.

    ``BUCTOOLKIT_KERNEL_BLOCKSIZE`` controls the result:

    * an integer: use it for every kernel;
    * ``'auto'`` (default): short runs use 256 for elementwise kernels and 1024
      for per-batch reduction kernels; long runs return ``(None, None)`` so the
      caller can autotune representative kernels once.

    Args:
        max_step: total number of MD steps of the run.

    Returns:
        ``(elementwise_block, reduction_block)``.  ``None`` entries mean that
        the caller must autotune that block size.

    Raises:
        ValueError: if the environment value is neither ``'auto'`` nor a
            positive integer.
    """
    raw = os.environ.get('BUCTOOLKIT_KERNEL_BLOCKSIZE', 'auto').strip()
    if raw != 'auto':
        try:
            block = int(raw)
        except ValueError as error:
            raise ValueError(
                "BUCTOOLKIT_KERNEL_BLOCKSIZE must be 'auto' or a positive integer, "
                f"but got {raw!r}."
            ) from error
        if block <= 0:
            raise ValueError(
                "BUCTOOLKIT_KERNEL_BLOCKSIZE must be a positive integer, "
                f"but got {block}."
            )
        return block, block
    if max_step < AUTOTUNE_STEP_THRESHOLD:
        return DEFAULT_ELEMENTWISE_BLOCK, DEFAULT_REDUCTION_BLOCK
    return None, None


def autotune_block_size(
        kernel,
        candidates: tuple[int, ...],
        args_fn,
        grid_fn,
        reps: int = 50,
        **options,
) -> int:
    """Time candidate block sizes of one kernel and return the fastest.

    Every candidate is compiled and warmed up once before timing, so the
    measured interval contains only launches.  The caller passes dummy tensors
    of the run's shapes and dtypes; no MD state is touched.

    Args:
        kernel: Triton JIT function whose last positional argument is the
            ``BLOCK_SIZE`` constexpr.
        candidates: block sizes to compare.
        args_fn: callable ``block_size -> tuple`` returning the full positional
            argument list for that block size.
        grid_fn: callable ``block_size -> tuple`` returning the launch grid.
        reps: number of timed launches per candidate.
        **options: compilation options such as ``num_warps``.

    Returns:
        The candidate with the smallest average launch time.

    Note:
        The wall time of this function also includes one compilation per
        candidate on a cold Triton cache.  Long runs amortize that cost.
    """
    best_block = candidates[0]
    best_time = float('inf')
    for block in candidates:
        grid = grid_fn(block)
        args = args_fn(block)
        # Compile and warm up this specialization before timing it.
        kernel[grid](*args, **options)
        th.cuda.synchronize()
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            kernel[grid](*args, **options)
        end.record()
        th.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        if elapsed < best_time:
            best_time = elapsed
            best_block = block
    return best_block

def validate_fused_run_state(
        owner: Any,
        s: Any,
        masses: th.Tensor,
        atom_masks: th.Tensor,
        batch_indices: Any,
) -> None:
    """Validate the fused-kernel buffers and batch layout once per run.

    The base :meth:`_BaseMD._validate_run_state` checks the common state
    tensors.  This function adds the fused-only contract: the flat layout and
    the per-batch buffers.  It runs once, before the main loop, so the hot
    path never needs per-call checks.  NVE only uses elementwise kernels and
    therefore checks just its inverse masses; NVT additionally checks the
    batch layout and thermostat buffers.

    Args:
        owner: NVE or NVT instance whose fused scheme is bound.
        s: state container holding ``X`` and ``Energy``.
        masses: per-element masses with the shape of ``s.X``.
        atom_masks: selective-dynamics mask with the shape of ``s.X``.
        batch_indices: irregular-batch split sizes, or ``None``.

    Returns:
        None.

    Raises:
        RuntimeError: if a fused buffer or the batch layout violates the kernel
            contract.  The message lists every violation.
    """
    issues: list[str] = []
    X = s.X
    device = X.device
    dtype = X.dtype

    def check(
            name: str,
            tensor: th.Tensor,
            numel: int | None = None,
            expected_dtype: th.dtype | None = None,
            shape: tuple[int, ...] | None = None,
            contiguous: bool = True,
    ) -> None:
        """Append a friendly violation for one fused tensor."""
        if not th.is_tensor(tensor):
            issues.append(f'`{name}` is not a torch.Tensor (got {type(tensor).__name__}).')
            return
        if tensor.device != device:
            issues.append(f'`{name}.device` is {tensor.device}; expected {device}.')
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            issues.append(f'`{name}.dtype` is {tensor.dtype}; expected {expected_dtype}.')
        if shape is not None and tuple(tensor.shape) != tuple(shape):
            issues.append(f'`{name}.shape` is {tuple(tensor.shape)}; expected {tuple(shape)}.')
        if numel is not None and tensor.numel() != numel:
            issues.append(f'`{name}.numel()` is {tensor.numel()}; expected {numel}.')
        if contiguous and not tensor.is_contiguous():
            issues.append(f'`{name}` is not contiguous.')

    def fail() -> None:
        """Raise the aggregated validation error."""
        raise RuntimeError(
            'Invalid fused-kernel state before the main loop:\n  - '
            + '\n  - '.join(issues)
            + '\nThe fused kernels expect a contiguous state of one dtype, a '
              'consistent batch layout, and per-batch buffers with the exact '
              'documented dtype/shape. Fix the listed inputs or fall back to '
              'the eager engine.'
        )

    # Section: flat layout and run scalars common to NVE and NVT
    if owner._n_elem != X.numel():
        issues.append(f'`_n_elem` is {owner._n_elem}; expected {X.numel()} from `s.X`.')
    if not isinstance(owner._block_size, int) or owner._block_size <= 0:
        issues.append(f'`_block_size` is {owner._block_size}; expected a positive int.')
    check('inv_m', owner.inv_m, shape=tuple(X.shape), expected_dtype=dtype)

    # Section: NVE uses elementwise kernels only, so the layout checks end here
    if not hasattr(owner, '_n_batch'):
        if issues:
            fail()
        return

    # Section: NVT batch layout
    scalar_dtype = owner._scalar_dtype
    n_batch = owner._n_batch
    if n_batch != s.Energy.numel():
        issues.append(f'`_n_batch` is {n_batch}; expected {s.Energy.numel()} from `s.Energy`.')
    if not isinstance(owner._reduction_block, int) or owner._reduction_block <= 0:
        issues.append(f'`_reduction_block` is {owner._reduction_block}; expected a positive int.')

    if owner._is_irregular:
        if X.shape[0] != 1:
            issues.append(f'An irregular batch must have leading size 1, but `s.X.shape[0]` is {X.shape[0]}.')
        if owner._n_elem % 3 != 0:
            issues.append(f'An irregular flat buffer must contain whole atoms, but `_n_elem` is {owner._n_elem}.')
        if owner._n_chunk != 0:
            issues.append(f'`_n_chunk` must be 0 for an irregular batch, but it is {owner._n_chunk}.')
        if batch_indices is None or len(batch_indices) != n_batch:
            issues.append('`batch_indices` must contain one entry per irregular batch.')
        starts = owner._batch_starts
        check('_batch_starts', starts, numel=n_batch + 1, expected_dtype=th.int32)
        if th.is_tensor(starts) and starts.numel() == n_batch + 1:
            starts_list = [int(_v) for _v in starts.tolist()]
            if starts_list[0] != 0:
                issues.append(f'`_batch_starts[0]` is {starts_list[0]}; expected 0.')
            if starts_list[-1] != owner._n_elem:
                issues.append(f'`_batch_starts[-1]` is {starts_list[-1]}; expected {owner._n_elem}.')
            if any(_b < _a for _a, _b in zip(starts_list, starts_list[1:])):
                issues.append('`_batch_starts` must be non-decreasing.')
        scatter = owner.batch_scatter
        check('batch_scatter', scatter, numel=owner._n_elem // 3)
        if th.is_tensor(scatter) and scatter.numel() > 0:
            low, high = int(scatter.min()), int(scatter.max())
            if low < 0 or high >= n_batch:
                issues.append(
                    f'`batch_scatter` spans [{low}, {high}]; expected values in [0, {n_batch}).'
                )
    else:
        if owner._n_chunk != X.shape[1] * X.shape[2]:
            issues.append(
                f'`_n_chunk` is {owner._n_chunk}; expected {X.shape[1] * X.shape[2]} from `s.X`.'
            )
        if owner._n_chunk * n_batch != owner._n_elem:
            issues.append('For a regular batch, `_n_chunk * _n_batch` must equal `_n_elem`.')
        if owner._batch_starts is not None:
            issues.append('`_batch_starts` must be None for a regular batch.')

    # Section: NVT thermostat buffers
    if getattr(owner, '_alpha', None) is not None:
        check('_alpha', owner._alpha, numel=n_batch, expected_dtype=scalar_dtype)
    if getattr(owner, '_sigma', None) is not None:
        check('_sigma', owner._sigma, shape=tuple(X.shape), expected_dtype=dtype)
    if getattr(owner, '_p_iota', None) is not None:
        check('_p_iota', owner._p_iota, numel=n_batch, expected_dtype=dtype)
    if getattr(owner, '_smass', None) is not None:
        check('_smass', owner._smass, numel=n_batch, expected_dtype=scalar_dtype)
    check('_free_degree', owner._free_degree, numel=n_batch, expected_dtype=scalar_dtype)
    check('_ek_target', owner._ek_target, numel=n_batch, expected_dtype=scalar_dtype)
    if owner.thermostat == "VR":
        check('Ek', owner.Ek, numel=n_batch, expected_dtype=dtype)

    if issues:
        fail()
