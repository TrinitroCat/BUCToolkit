#  Copyright (c) 2024-2026.11.2, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: mace_dataloader.py
#  Environment: Python 3.12
"""Data loaders for the built-in MACE model path."""

import queue
import random
import threading
from typing import Any, Dict, List, Mapping, Self, Tuple, Literal, Sequence

import numpy as np
import torch as th

from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.BatchStructures import Batch
from BUCToolkit.utils.model_wrappers.MACE_model_wrapper import (
    MACEDataAdapter,
    _require_mace,
)

# check modules
_pyg = check_module('torch_geometric.data')
if _pyg is not None:
    pygBatch = _pyg.Batch
else:
    pygBatch = Batch


# Sentinel put into the prefetch queue after the last structure.  ``None`` is a
# valid queue item in principle, so use a private object to mark completion.
_MACE_PREFETCH_END = object()


class MACEDataLoader:
    """Form MACE batches from in-memory structures with a bounded prefetch buffer.

    Args:
        data: Mapping containing a ``data`` sequence and, during training, a
            ``labels`` mapping with per-structure ``energy`` and optional
            per-atom ``forces`` values.
        batch_size: Number of structures in each batch.
        device: Device receiving the MACE batch and labels.
        shuffle: Whether to shuffle structures and labels together.
        is_train: Whether the loader should return training labels.
        data_names: Optional structure names returned when ``is_train`` is false.
        model_config: MACE configuration mapping. It must contain
            ``atomic_numbers`` and ``r_max`` and may contain ``heads`` and
            ``default_head``. Direct loader construction requires this mapping;
            task APIs inject the resolved ``MODEL_CONFIG`` automatically.
        buffer_size: Maximum number of converted ``AtomicData`` objects kept in
            the prefetch queue. Defaults to ``4 * batch_size`` and is never
            smaller than ``batch_size``.

    Yields:
        A MACE Batch paired with training labels, structure names, or ``None``.

    Notes:
        A single background producer thread converts structures with
        :meth:`MACEDataAdapter.to_atomic_data` and places them into a
        ``queue.Queue``.  The main thread builds one MACE Batch at a time, so the
        number of resident COO neighbor lists is bounded by ``buffer_size``
        instead of by the size of the whole data set.

        Items in ``data['data']`` may be raw Data-like structures or already
        converted MACE ``AtomicData`` objects.  Both cases use the same
        :meth:`MACEDataAdapter.to_atomic_data` call; the adapter returns an
        ``AtomicData`` input unchanged, which keeps pre-converted callers
        working while the raw-data path performs the conversion lazily.

        A conversion failure stops the producer.  A complete batch that was
        already assembled is returned first; the error is raised by the next
        ``__next__`` call, or immediately when the current batch is incomplete.

        The background thread is a static worker that never retains ``self``.
        This keeps the loader collectible when a task abandons it, for example
        when ``Trainer._warm_up`` breaks after a few batches.
    """

    def __init__(
            self,
            data: Dict[Literal['data', 'labels'] | Literal['data', 'names'], Any],
            batch_size: int,
            device: str | th.device = 'cpu',
            shuffle: bool = True,
            is_train: bool = True,
            data_names: Sequence[str] | None = None,
            model_config: Mapping[str, Any] | None = None,
            buffer_size: int | None = None,
    ) -> None:

        self._batch_class = _require_mace()['Batch']
        self.data: List = data['data']
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f'`batch_size` must be a positive integer, but got {batch_size!r}.')
        if model_config is None:
            raise ValueError(
                '`MACEDataLoader` requires `model_config` containing '
                '`atomic_numbers` and `r_max`.'
            )
        if not isinstance(model_config, Mapping):
            raise TypeError(
                f'`model_config` must be a mapping, but got {type(model_config).__name__}.'
            )
        try:
            atomic_numbers = model_config['atomic_numbers']
            r_max = model_config['r_max']
        except KeyError as exc:
            raise ValueError(
                f'`model_config` is missing required key {exc.args[0]!r}.'
            ) from exc
        self._data_adapter = MACEDataAdapter(
            atomic_numbers=atomic_numbers,
            r_max=r_max,
            heads=model_config.get('heads'),
            default_head=model_config.get('default_head'),
        )

        if buffer_size is None:
            buffer_size = 4 * batch_size
        if not isinstance(buffer_size, int) or isinstance(buffer_size, bool) or buffer_size <= 0:
            raise ValueError(
                f'`buffer_size` must be a positive integer, but got {buffer_size!r}.'
            )
        self.buffer_size = max(buffer_size, batch_size)

        self.is_train = is_train
        if is_train:
            self.energy: Any = data['labels']['energy']
            self.forces: Any = data['labels'].get('forces', None)
            if shuffle: self.shuffle()
            if not isinstance(self.energy, th.Tensor):
                self.energy_ = th.from_numpy(np.array(self.energy, dtype=np.float32))
            else:
                self.energy_ = self.energy
        self.data_names = data_names
        self.batchsize = batch_size
        self._n_samp = len(self.data)
        self._index = 0
        self.device = device
        # set the flag
        self._LOADER_TYPE = 'Train'

        # Prefetch state.  The queue and the producer thread are created on the
        # first ``__next__`` so loaders that are constructed but never iterated
        # (for example a validation loader that is skipped) do no conversion.
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop: threading.Event | None = None
        self._prefetch_errors: List[BaseException] = []
        self._closed = False

    def shuffle(self, ) -> None:
        """Shuffle structures and labels with one shared local seed."""
        _seed = random.randint(0, 2147483647)
        random.Random(_seed).shuffle(self.data)
        random.Random(_seed).shuffle(self.energy)
        if self.forces is not None:
            random.Random(_seed).shuffle(self.forces)

    @staticmethod
    def _do_async_mace_prepare(
            data: Sequence[Any],
            adapter: MACEDataAdapter,
            buffer: queue.Queue,
            stop_event: threading.Event,
            error_holder: List[BaseException],
    ) -> None:
        """Convert structures in a background producer thread.

        The worker deliberately receives every dependency explicitly instead of
        being a bound method.  A bound method would create a reference cycle
        through the loader and prevent ``__del__`` from stopping an abandoned
        loader.  Exceptions are stored in *error_holder* and re-raised by the
        consuming thread after the sentinel is observed.

        Args:
            data: Structures in the order expected by the consumer.
            adapter: MACE adapter used for one-structure conversion.
            buffer: Bounded queue shared with the consuming thread.
            stop_event: Set when the loader is closed; polled while the queue is
                full so a blocked producer can exit promptly.
            error_holder: Shared list receiving the first conversion exception.
        """
        try:
            for structure in data:
                if stop_event.is_set():
                    return
                atomic_data = adapter.to_atomic_data(structure)
                while not stop_event.is_set():
                    try:
                        buffer.put(atomic_data, timeout=0.1)
                        break
                    except queue.Full:
                        # A timeout-based put lets ``close`` interrupt a producer
                        # that is blocked because the consumer stopped early.
                        continue
            while not stop_event.is_set():
                try:
                    buffer.put(_MACE_PREFETCH_END, timeout=0.1)
                    break
                except queue.Full:
                    continue
        except BaseException as exc:
            error_holder.append(exc)
            while not stop_event.is_set():
                try:
                    buffer.put(_MACE_PREFETCH_END, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _start_prefetch(self) -> None:
        """Create the bounded queue and start the producer thread once."""
        if self._prefetch_queue is not None:
            return
        self._prefetch_queue = queue.Queue(maxsize=self.buffer_size)
        self._prefetch_stop = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._do_async_mace_prepare,
            args=(
                self.data,
                self._data_adapter,
                self._prefetch_queue,
                self._prefetch_stop,
                self._prefetch_errors,
            ),
            daemon=True,
            name='MACEDataLoaderPrefetch',
        )
        self._prefetch_thread.start()

    def _take_prefetch_batch(self) -> List[Any]:
        """Collect up to ``batch_size`` converted structures from the queue."""
        self._start_prefetch()
        data_list: List[Any] = []
        while len(data_list) < self.batchsize:
            try:
                item = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                thread = self._prefetch_thread
                if thread is not None and not thread.is_alive() and self._prefetch_queue.empty():
                    break
                continue
            if item is _MACE_PREFETCH_END:
                break
            data_list.append(item)
        return data_list

    def __iter__(self, ) -> Self:
        return self

    def __next__(self, ) -> Tuple[Any, Dict[Literal['energy', 'forces'], th.Tensor] | None]:
        """Return the next MACE batch and its labels.

        Returns:
            A ``(batch, label)`` tuple.  ``label`` is a dictionary during
            training, a sequence of structure names for prediction, or ``None``.

        Raises:
            StopIteration: When all structures have been consumed.
            Exception: The first conversion error raised by the producer.  A
                complete batch already collected before the failure is yielded
                first; the error is raised by the following call.
        """
        if self._closed:
            raise StopIteration

        data_list = self._take_prefetch_batch()
        if len(data_list) == 0:
            self.close()
            if self._prefetch_errors:
                raise self._prefetch_errors[0]
            raise StopIteration
        if len(data_list) < self.batchsize and self._prefetch_errors:
            # Do not yield a partial batch after a conversion failure.  A full
            # batch is returned first so that the data which was successfully
            # converted is not discarded.
            self.close()
            raise self._prefetch_errors[0]

        data = self._batch_class.from_data_list(data_list)
        data = data.to(self.device)
        # MACE's vendored Batch exposes ``num_graphs`` but omits the
        # ``batch_size`` and ``idx`` attributes required by BUCToolkit's
        # Trainer contract. Keep these compatibility attributes at the loader
        # boundary instead of modifying either implementation.
        data.batch_size = data.num_graphs
        data.idx = [
            structure.get('idx') if isinstance(structure, Mapping)
            else getattr(structure, 'idx', None)
            for structure in data_list
        ]

        start = self._index * self.batchsize
        stop = start + self.batchsize
        if self.is_train:
            energy = self.energy_[start:stop]
            energy = energy.to(self.device)

            if self.forces is not None:
                force = self.forces[start:stop]
                force = th.from_numpy(np.concatenate(force)).to(self.device)
                label: Dict[Literal['energy', 'forces'], th.Tensor] | None = {'energy': energy, 'forces': force}
            else:
                label: Dict[Literal['energy'], th.Tensor] | None = {'energy': energy}

        elif self.data_names is not None:
            label: Sequence[str] = self.data_names[start:stop]

        else:
            label = None

        self._index += 1
        return data, label

    def close(self) -> None:
        """Stop the producer, join it, and release buffered structures.

        The method is idempotent.  It is called on normal exhaustion and from
        ``__del__``; an explicit call is also safe when a consumer stops early.
        """
        if self._closed:
            return
        self._closed = True
        if self._prefetch_stop is not None:
            self._prefetch_stop.set()
        thread = self._prefetch_thread
        if thread is not None and thread.is_alive():
            # The producer polls the stop flag with a 0.1 s timeout.  A longer
            # join only waits for an in-flight structure conversion; a timeout
            # is harmless because the daemon thread exits after that conversion.
            thread.join(timeout=0.5)
        if self._prefetch_queue is not None:
            while True:
                try:
                    self._prefetch_queue.get_nowait()
                except queue.Empty:
                    break

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Module teardown during interpreter shutdown can make queue or
            # threading internals unavailable; cleanup is best-effort there.
            pass


class ISFSMACEDataLoader:
    """
    A Data loader to form pygData IN MEMORY.
    This Data loader yields Tuple(pygData1, pygData2) instead of (pygData, label).
    Wherein, pygData1 and pygData2 are corresponding to the initial state and finale state of structures.

    Args:
        data: Dict: {'dataIS': D1, 'dataFS': D2}, where
            D1 and D2 are 2 Lists of pyg.Data that contain attributes `pos`, `cell`, `atomic_numbers`, `natoms`, `tags`, `fixed`, `pbc`, `idx`.
              `pos`: Tensor, atom coordinates.
              `cell`: Tensor, cell vectors.
              `atomic_numbers`: Tensor, atomic numbers, corresponding to `pos` one by one.
              `natoms`: int, number of atoms.
              `tags`: Tensor, to be compatible with 'FAIR-CHEM' (https://fair-chem.github.io/),
                      which fixed slab part is set to 0, free slab part is 1, adsorbate is 2.
              `fixed`: Tensor, fixed tag, which fixed atoms are 0, free atoms are 1.
              `pbc`: List[bool, bool, bool], where to be periodic at x, y, z directions.
              `idx`: str, the name of this structure.
        batch_size: batch size.
        device: the device that data put on.
        data_names: if `data_names` is not None, it should be a Sequence(data names) with the same order as data,
                and the returned `labels` [i.e., next(iter(dataloader))] would be data_names instead of "energy" or "forces",
                else `labels` would be None.

    Yields:
            (pyg.Data, pyg.Data)

    Notes:
        This loader materializes MACE ``AtomicData`` before iteration and is
        intended for pre-converted static paired data.  For NEB/CMD runs whose
        coordinates change between evaluations, the task APIs use
        ``ISFSPyGDataLoader`` so that ``MACEWrapper.forward`` rebuilds the
        neighbor graph from the current PyG batch.
    """
    def __init__(
            self,
            data: Dict[Literal['dataIS', 'dataFS'], Any],
            batch_size: int,
            device: str | th.device = 'cpu',
            data_names: Sequence[str] | None = None
    ) -> None:

        # check module
        self._batch_class = _require_mace()['Batch']
        self.dataIS: List = data['dataIS']
        self.dataFS: List = data['dataFS']
        self.data_names = data_names
        self.batchsize = batch_size
        self._n_samp = len(self.dataIS)
        _n_samp_check = len(self.dataFS)
        if _n_samp_check != self._n_samp:
            raise RuntimeError(
                f"The number of samples in `dataIS` and `dataFS` do not match. Number of IS: {self._n_samp} & number of FS: {_n_samp_check}."
            )
        self._index = 0
        self.device = device
        # set the flag
        self._LOADER_TYPE = 'ISFS'  # initial state & finale state

    def __iter__(self, ) -> Self:
        return self

    def __next__(self, ) -> Tuple[pygBatch, pygBatch]:
        if self._index * self.batchsize < self._n_samp:
            dataIS_list = self.dataIS[self._index * self.batchsize: (self._index + 1) * self.batchsize]
            dataIS = self._batch_class.from_data_list(dataIS_list)
            dataIS = dataIS.to(self.device)  # type: ignore
            dataFS_list = self.dataFS[self._index * self.batchsize: (self._index + 1) * self.batchsize]
            dataFS = self._batch_class.from_data_list(dataFS_list)
            dataFS = dataFS.to(self.device)  # type: ignore
            dataIS.batch_size = dataIS.num_graphs
            dataIS.idx = [
                structure.get('idx') if isinstance(structure, Mapping)
                else getattr(structure, 'idx', None)
                for structure in dataIS_list
            ]
            dataFS.batch_size = dataFS.num_graphs
            dataFS.idx = [
                structure.get('idx') if isinstance(structure, Mapping)
                else getattr(structure, 'idx', None)
                for structure in dataFS_list
            ]

            self._index += 1
            return dataIS, dataFS
        else:
            raise StopIteration
