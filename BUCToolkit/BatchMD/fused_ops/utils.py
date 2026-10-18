#  Copyright (c) 2024-2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: utils.py
#  Environment: Python 3.12

"""Miscellaneous helpers of the fused Triton kernels."""

import torch as th
from triton.runtime import driver as _triton_driver


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
                ``tl.constexpr`` parameters.

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
