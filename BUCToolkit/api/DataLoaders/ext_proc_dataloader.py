#  Copyright (c) 2024-2026.11.2, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: ext_proc_dataloader.py
#  Environment: Python 3.12
import random
from typing import Any, Dict, Callable, List, Self, Tuple, Literal, Sequence

import torch as th

from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.BatchStructures import Batch
from .pyg_dataloader import PyGDataLoader


# check modules
_pyg = check_module('torch_geometric.data')
if _pyg is not None:
    pygBatch = _pyg.Batch
else:
    pygBatch = Batch


class ExtProcDataLoader(PyGDataLoader):
    """PyG loader that signals completion of each external-process batch.

    The loader keeps the ordinary :class:`PyGDataLoader` data contract while
    adding a lifecycle boundary for calculators backed by a persistent
    external process.  The callback is sent before every batch after the first
    one and once when iteration reaches ``StopIteration``.

    Args:
        signal_function: Optional callback used to release the previous
            external-process session.
        *args, **kwargs: Arguments accepted by :class:`PyGDataLoader`.

    Notes:
        ``is_first`` is reset when a batch boundary is signalled.  This keeps
        repeated calls after exhaustion harmless and prevents duplicate
        cleanup callbacks.
    """

    def __init__(
            self,
            *args: Any,
            signal_function: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.batchsize != 1:
            raise ValueError(
                "`ExtProcDataLoader` requires batch_size = 1 because one "
                "external-process session serves exactly one structure."
            )
        self.is_first = True
        self._signal_function: Callable[[], Any] = lambda: None
        if signal_function is not None:
            self.set_signal(signal_function)

    def set_signal(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Register the callback invoked at external-process boundaries.

        Args:
            func: Callable that releases the active external-process session.
            *args, **kwargs: Arguments bound to ``func`` for every callback.

        Raises:
            TypeError: If ``func`` is not callable.
        """
        if not callable(func):
            raise TypeError(f"`func` must be callable, but got {type(func)}.")

        def send_signal() -> Any:
            return func(*args, **kwargs)

        self._signal_function = send_signal

    def _sent_signal(self) -> Any:
        """Invoke the registered external-process lifecycle callback."""
        return self._signal_function()

    def __next__(self) -> Tuple[pygBatch, Dict[Literal['energy', 'forces'], th.Tensor] | None]:
        """Return the next batch after releasing the preceding one."""
        if not self.is_first:
            self._sent_signal()
            self.is_first = True
        try:
            batch = super().__next__()
        except StopIteration:
            raise
        self.is_first = False
        return batch
