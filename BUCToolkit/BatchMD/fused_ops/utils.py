#  Copyright (c) 2024-2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
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
