#  Copyright (c) 2024-2026.7.20, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0a
#  File: _BaseMC.py
#  Environment: Python 3.12

import math
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.Bases.BaseMotion import BaseMotion
from BUCToolkit.Bases.StdContainer import StdContainer
from BUCToolkit.utils._Element_info import ATOMIC_SYMBOL, DTYPE
from BUCToolkit.utils._print_formatter import SCIENTIFIC_ARRAY_FORMAT
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.utils.index_ops import index_reduce


FLOAT_TYPE = DTYPE.get(os.environ.get('BT_FLOAT_TYPE', 'float32'), th.float32)


class _BaseMC(BaseMotion):
    """Base framework for batched Monte-Carlo algorithms.

    The class handles device placement, regular and irregular batches,
    selective dynamics, geometric-center correction, model evaluation,
    trajectory dumping, and logging. Subclasses only need to implement the
    proposal rule and any algorithm-specific parameter schedule.

    Coordinates use one of two layouts:

    * regular batch: ``X[n_batch, n_atom, n_dim]``;
    * irregular batch: ``X[1, sum(n_atoms), n_dim]`` together with
      ``batch_indices=(n_0, n_1, ..., n_N)``.

    The trajectory and logging implementation follows :class:`_BaseMD` and
    :class:`_BaseOpt`. Live values are stored in :class:`StdContainer`, the
    registered quantity names define the binary columns and log fields, and
    CPU/CUDA execution shares one asynchronous consumer protocol. The state
    available for registration is ``Energy``, ``X``, ``delta_E``,
    ``is_accept``, and ``temperature``.
    """

    #: Quantities maintained by the MC framework and therefore valid in
    #: ``dump_quantities`` / ``log_quantities``.
    ALLOWED_QUANTITIES: Set[str] = {
        'Energy', 'X', 'delta_E', 'is_accept', 'temperature'
    }

    def __init__(
            self,
            iter_scheme: str,
            maxiter: int = 100,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 0,
            is_compile: bool = False,
            compile_kwargs: Dict | None = None,
            dump_quantities: Tuple[str, ...] | List[str] = ('Energy', 'X'),
            log_quantities: Tuple[str, ...] | List[str] = (
                'Energy', 'delta_E', 'is_accept', 'temperature', 'X'
            ),
    ) -> None:
        r"""Initialize the common Monte-Carlo runtime.

        Args:
            iter_scheme: Name of the coordinate proposal scheme implemented by
                the subclass. It is retained as ``self.iterform`` and printed
                in the run header when ``verbose > 0``.
            maxiter: Maximum number of Monte-Carlo proposal/update iterations.
                It must be a positive integer.
            output_file: Path of the binary trajectory file. If ``None``, a
                no-operation dumper is used and no file is created.
            output_structures_per_step: Dump/log interval measured in MC
                iterations. A snapshot is produced at iterations
                ``0, interval, 2 * interval, ...`` and the value must be a
                positive integer.
            device: Torch device on which model evaluation and MC updates run,
                for example ``'cpu'``, ``'cuda'``, or ``torch.device('cuda')``.
            verbose: Amount of logging information. ``0`` disables iteration
                logs, ``1`` prints registered scalar/vector quantities, and
                ``2`` or greater additionally prints registered arrays such as
                coordinates.
            is_compile: Whether to pass the shared main loop through
                :func:`torch.compile`. Python I/O and synchronization sections
                remain graph breaks; tensor update regions may still compile.
            compile_kwargs: Optional keyword arguments forwarded unchanged to
                :func:`torch.compile` when ``is_compile`` is ``True``.
            dump_quantities: Ordered names of state tensors written as columns
                in each trajectory frame. Valid names are listed in
                :attr:`ALLOWED_QUANTITIES`. The default ``('Energy', 'X')``
                preserves the legacy MC file layout.
            log_quantities: Ordered names copied to the asynchronous logging
                consumer. Valid names are listed in
                :attr:`ALLOWED_QUANTITIES`. The default logs energies, energy
                changes, acceptance masks, temperatures, and coordinates.

        Raises:
            ValueError: If ``maxiter`` or ``output_structures_per_step`` is not
                positive, or if a requested dump/log quantity is unknown.
        """
        if not isinstance(maxiter, int) or maxiter <= 0:
            raise ValueError(f'maxiter must be a positive integer, but got {maxiter}.')
        if not isinstance(output_structures_per_step, int) or output_structures_per_step <= 0:
            raise ValueError(
                'output_structures_per_step must be a positive integer, '
                f'but got {output_structures_per_step}.'
            )

        self.iterform = str(iter_scheme)
        self.maxiter = maxiter
        self.output_structures_per_step = output_structures_per_step
        self.device = device if isinstance(device, th.device) else th.device(device)
        self.verbose = int(verbose)
        self.is_compile = bool(is_compile)
        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        self.batch_scatter = None
        self.batch_tensor = None
        self.batch_slice_indx = None
        self.scatter_dim_out_size = None
        self.atom_masks = None
        self.fixed_indices = None
        self.free_degree = None
        self.n_batch, self.n_atom, self.n_dim = None, None, None
        self.is_concat_X = False
        self.is_accept = None
        self.T_now = None

        # Advanced APIs hold one dumper open across several ``run()`` calls.
        self._HOLD_DUMPER = False

        super().__init__(output_file)
        self.init_logger('Main.MC')
        self._setup_register_vars(dump_quantities, log_quantities)

    def _do_async_dump(self, q: queue.Queue):
        """Consume named MC snapshots and write them to the active dumper.

        Queue items use ``(dumper, sync_event, *cpu_tensors)``. ``sync_event``
        is a CUDA event for asynchronous D2H copies and ``None`` on CPU.
        ``_dump_done`` is always set in ``finally`` so a failed write cannot
        deadlock the producer loop.

        Args:
            q: Single-slot producer/consumer queue. A ``None`` item terminates
                the thread. Every other item contains the dumper, an optional
                CUDA synchronization event, and tensors ordered according to
                :meth:`get_dump_vars`.

        Returns:
            None. Errors are reported through ``self.logger`` because they
            occur in a background thread and cannot be raised to ``run()``.
        """
        while True:
            items = q.get()
            if items is None:
                break
            try:
                dumper, sync_event, *values = items
                if sync_event is not None:
                    sync_event.synchronize()
                dumper.step(*(value.numpy() for value in values))
            except Exception as exc:
                self.logger.error(f'Error: Failed to dump data due to "{exc}"')
            finally:
                self._dump_done.set()

    def _do_async_print(self, q: queue.Queue):
        """Format registered log quantities without hardcoded tuple layouts.

        Values are packed in ``get_log_vars()`` order. Scalars/vectors are
        printed in one iteration block, while matrices and higher-dimensional
        arrays are delegated to ``handle_arrays_print``.

        Args:
            q: Single-slot producer/consumer queue. A ``None`` item terminates
                the thread. Normal items have the form
                ``(sync_event, iteration, batch_indices, *cpu_tensors)``;
                ``cpu_tensors`` follows :meth:`get_log_vars` order.

        Returns:
            None. The method signals ``self._print_done`` after every item,
            including items whose formatting raises an exception.
        """
        _display_names = {
            'Energy': 'Energies',
            'delta_E': 'delta E',
            'is_accept': 'Accepted',
            'temperature': 'Temperature',
            'X': 'Coordinates',
        }
        _numit = 0
        while True:
            items = q.get()
            if items is None:
                break
            try:
                sync_event, _numit, batch_indices = items[:3]
                if sync_event is not None:
                    sync_event.synchronize()

                _data = dict(zip(self.get_log_vars(), items[3:]))
                _valid = [(_name, _value) for _name, _value in _data.items()
                          if _name and _value is not None]
                _scalars = [(_name, _value) for _name, _value in _valid
                            if _value.ndim <= 1]
                _arrays = [(_name, _value) for _name, _value in _valid
                           if _value.ndim >= 2]

                if _scalars and self.verbose > 0:
                    _lines = [f'ITERATION    {_numit:>5d}']
                    for _name, _value in _scalars:
                        _label = _display_names.get(_name, _name)
                        if _value.dtype == th.bool:
                            _value_str = np.array2string(_value.numpy())
                        else:
                            _value_str = np.array2string(
                                _value.numpy(), **SCIENTIFIC_ARRAY_FORMAT
                            )
                        _lines.append(f' {_label:<12s}: {_value_str}')
                    self.logger.info('\n'.join(_lines))

                if _arrays and self.verbose > 1:
                    self.handle_arrays_print(
                        self.logger,
                        batch_indices,
                        self.batch_slice_indx,
                        [[_value for _, _value in _arrays]],
                        [[_display_names.get(_name, _name)
                          for _name, _ in _arrays]],
                        verbose=self.verbose,
                        force=False,
                    )
            except Exception as exc:
                self.logger.error(
                    f'Error: Failed to logout at {_numit}-th iteration '
                    f'due to "{exc}".'
                )
            finally:
                self._print_done.set()

    def calc_shape_center(self, Xr: th.Tensor) -> th.Tensor:
        """Calculate each structure's geometric center.

        Args:
            Xr: Coordinate tensor. For a regular batch its shape is
                ``[n_batch, n_atom, n_dim]``. For an irregular concatenated
                batch its shape is ``[1, sum(n_atoms), n_dim]`` and
                ``self.batch_scatter`` must already be initialized.

        Returns:
            Tensor containing the geometric center expanded over the atom axis
            of ``Xr``. The returned shape is ``[n_batch, 1, n_dim]`` for a
            regular batch and ``[1, sum(n_atoms), n_dim]`` for an irregular
            batch, so it can be added to or subtracted from ``Xr`` directly.

        Notes:
            This is a purely geometric center and does not use atomic masses.
        """
        if self.batch_scatter is None:
            return th.mean(Xr, dim=1, keepdim=True)

        shape_center = index_reduce(
            Xr,
            self.batch_scatter,
            dim=1,
            out_size=self.scatter_dim_out_size,
        ) / self.batch_tensor.reshape(1, -1, 1)
        return shape_center.index_select(1, self.batch_scatter)

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]] | None = None,
            Cell_vector: th.Tensor | None = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            move_to_center_freq: int = -1,
    ) -> None:
        """Run the configured Monte-Carlo algorithm.

        Args:
            func: Energy model or callable. It receives ``X`` followed by
                ``func_args`` and ``func_kwargs`` and must return one energy per
                real structure with shape ``[n_true_batch]``.
            X: Initial Cartesian coordinates. Accepted shapes are
                ``[n_atom, n_dim]``, ``[n_batch, n_atom, n_dim]``, or the
                irregular layout ``[1, sum(n_atoms), n_dim]`` when
                ``batch_indices`` is supplied. A two-dimensional input is
                promoted to a one-structure batch.
            Element_list: Element symbols or atomic numbers for each real
                structure. In an irregular batch the nested rows may have
                different lengths and must follow ``batch_indices`` order. If
                ``None``, all atomic numbers in the trajectory header are set
                to zero. Elements are metadata only and are not passed to
                ``func`` automatically.
            Cell_vector: Optional cell vectors with shape
                ``[n_true_batch, 3, 3]``. They are written as trajectory
                metadata and do not participate in MC calculations. If
                omitted, zero cell vectors are stored.
            func_args: Positional arguments appended after ``X`` when calling
                ``func``.
            func_kwargs: Keyword arguments passed to ``func``. ``None`` is
                normalized to an empty dictionary.
            batch_indices: Atom counts ``(n_0, n_1, ..., n_N)`` for an
                irregular concatenated batch. The counts define the split
                points for coordinates, element rows, fixed masks, geometric
                centers, and model outputs. If supplied, ``X.shape[0]`` must be
                one.
            fixed_atom_tensor: Optional selective-dynamics mask broadcastable
                to ``X.shape``. Components equal to zero are fixed; nonzero
                components may move. If ``None``, all components are movable.
            move_to_center_freq: Frequency for translating every structure
                back to its initial geometric center. A value less than or
                equal to zero disables this correction.

        Returns:
            None. The method does not return the final MC state; selected
            snapshots are written to ``output_file`` when configured and
            registered values are emitted through the logger.

        Raises:
            TypeError: If ``X`` is not a tensor, ``move_to_center_freq`` is not
                an integer, or the element/batch metadata has an invalid type.
            ValueError: If ``X`` has an invalid rank or the energy model output
                does not contain one value per real structure.
            RuntimeError: If ``batch_indices`` is used with a non-concatenated
                leading coordinate dimension.
        """
        if not isinstance(X, th.Tensor):
            raise TypeError(f'`X` must be torch.Tensor, but got {type(X)}.')
        if not isinstance(move_to_center_freq, int):
            raise TypeError(
                '`move_to_center_freq` must be an integer, '
                f'but got {type(move_to_center_freq)}.'
            )

        X, Cell_vector = self.handle_dtype_device(
            FLOAT_TYPE, self.device, X, Cell_vector
        )
        self.reset_register_vars()
        try:
            self._run(
                func,
                X,
                Element_list,
                Cell_vector,
                func_args,
                {} if func_kwargs is None else func_kwargs,
                batch_indices,
                fixed_atom_tensor,
                move_to_center_freq,
            )
        finally:
            if not self._HOLD_DUMPER:
                self.dumper.close()

    def _run(
            self,
            func,
            X,
            Element_list,
            Cell_vector,
            func_args,
            func_kwargs,
            batch_indices,
            fixed_atom_tensor,
            move_to_center_freq,
    ) -> None:
        """Prepare one MC run and execute the shared CPU/CUDA main loop.

        Args:
            func: Preloadable energy callable returning one value per real
                structure.
            X: Coordinate tensor already converted to ``FLOAT_TYPE`` and
                ``self.device`` by :meth:`run`.
            Element_list: Per-structure element symbols or atomic numbers used
                to construct the static trajectory header.
            Cell_vector: Optional per-structure cell vectors used only as
                trajectory metadata.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Dictionary of keyword arguments forwarded to ``func``.
            batch_indices: Optional per-structure atom counts for an irregular
                concatenated coordinate tensor.
            fixed_atom_tensor: Optional selective-dynamics mask broadcastable
                to ``X``.
            move_to_center_freq: Geometric-center restoration interval; values
                less than or equal to zero disable restoration.

        Returns:
            None. The method initializes batch maps, algorithm state, named I/O
            buffers, consumer threads, and then runs :meth:`_main_for_loop`.
        """
        _t_main = time.perf_counter()
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f'`X` must be 2D or 3D, but got shape {X.shape}.')

        _n_batch, _n_atom, _n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = _n_batch, _n_atom, _n_dim
        (
            _n_true_batch,
            batch_indices,
            self.batch_tensor,
            self.batch_scatter,
            self.batch_slice_indx,
        ) = self.handle_batch_indices(batch_indices, _n_batch, self.device)
        self.scatter_dim_out_size = (
            self.batch_scatter.max().item() + 1
            if self.batch_scatter is not None else None
        )
        self.is_concat_X = batch_indices is not None

        if Element_list is None:
            atomic_numbers = [[0] * _n_atom] * _n_batch
        else:
            atomic_numbers = []
            for _elements in Element_list:
                if not isinstance(_elements, list):
                    raise TypeError(
                        'Expected `Element_list` of List[List[int | str]], '
                        f'but got an item of type {type(_elements)}.'
                    )
                atomic_numbers.append([
                    ATOMIC_SYMBOL[_element]
                    if isinstance(_element, str) else int(_element)
                    for _element in _elements
                ])

        self.atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)
        self.fixed_indices = th.where(th.any(self.atom_masks, dim=-1))[1]
        _fix_shape_center = move_to_center_freq > 0

        if batch_indices is None:
            _free_degree = _n_atom * _n_dim - (3 if _fix_shape_center else 0)
            self.free_degree = th.full(
                (_n_batch,), _free_degree, dtype=th.int64, device=self.device
            )
            self.free_degree -= th.where(
                th.abs(self.atom_masks) < 1e-6, 1, 0
            ).sum(dim=(-2, -1))
        else:
            self.free_degree = self.batch_tensor * _n_dim
            if _fix_shape_center:
                self.free_degree -= 3
            _n_reduced = index_reduce(
                th.where(th.abs(self.atom_masks) < 1e-6, 1, 0).sum(dim=-1),
                self.batch_scatter,
                dim=1,
                out_size=self.scatter_dim_out_size,
            ).squeeze(0)
            self.free_degree -= _n_reduced

        func = preload_func(func, self.device)
        _atom_masks_array = self.atom_masks.numpy(force=True).astype(
            X.numpy(force=True).dtype.str
        )
        if Cell_vector is None:
            Cell_vector = np.zeros((_n_true_batch, 3, 3), dtype=np.float32)
        elif isinstance(Cell_vector, th.Tensor):
            Cell_vector = Cell_vector.numpy(force=True)
        elif not isinstance(Cell_vector, np.ndarray):
            Cell_vector = np.asarray(Cell_vector)

        # A regular batch has one rectangular ``(n_atom,)`` atomic-number
        # row per structure, so NumPy can represent the nested list directly.
        # An irregular batch instead contains rows of different lengths;
        # converting that ragged list with ``np.asarray`` raises an error on
        # recent NumPy versions. Its coordinates and masks already use one
        # concatenated atom axis, therefore store atomic numbers in the same
        # flat order. The header's ``batch_indices`` reconstructs the
        # per-structure rows when the trajectory is read.
        if self.batch_tensor is None:
            _atomic_numbers_array = np.asarray(atomic_numbers, dtype=np.int64)
        else:
            _atomic_numbers_array = np.concatenate([
                np.asarray(_numbers, dtype=np.int64)
                for _numbers in atomic_numbers
            ])

        # Header group: static system metadata. The data group is started only
        # after model evaluation reveals the exact registered tensor shapes.
        if self.batch_tensor is not None:
            _batch_array = self.batch_tensor.numpy(force=True)
            self.dumper.start_from_arrays(
                1, _batch_array, Cell_vector,
                _atomic_numbers_array, _atom_masks_array,
                names=('batch_indices', 'cell_vec', 'atomic_numbers', 'fixed_mask'),
            )
            self.dumper.step(
                _batch_array, Cell_vector,
                _atomic_numbers_array, _atom_masks_array,
            )
        else:
            self.dumper.start_from_arrays(
                1, Cell_vector, _atomic_numbers_array, _atom_masks_array,
                names=('cell_vec', 'atomic_numbers', 'fixed_mask'),
            )
            self.dumper.step(
                Cell_vector, _atomic_numbers_array, _atom_masks_array,
            )

        self.initialize_algo_param()
        if self.verbose > 0:
            self.logger.info('-' * 100)
            self.logger.info(f'Iteration Scheme: {self.iterform}')

        with th.no_grad():
            energies: th.Tensor = func(X, *func_args, **func_kwargs)
            if batch_indices is None and energies.shape[0] != _n_batch:
                raise ValueError(
                    f'Model output shape {energies.shape} does not match '
                    f'batch size {_n_batch}.'
                )
            if batch_indices is not None and energies.shape[0] != _n_true_batch:
                raise ValueError(
                    f'Model output shape {energies.shape} does not match '
                    f'{_n_true_batch} entries in batch_indices.'
                )

            self.is_accept = th.zeros_like(energies, dtype=th.bool)
            _temperature = 0. if self.T_now is None else float(self.T_now)
            s = StdContainer(
                Energy=energies,
                X=X,
                delta_E=th.zeros_like(energies),
                is_accept=self.is_accept,
                temperature=th.full_like(energies, _temperature),
            )
            self.set_registered_var_values(s)

            dump_names = self.get_dump_vars()
            log_names = self.get_log_vars()
            total_names = self.get_transfer_vars()
            s_cpu, s_buf = self._allocate_cpu_buffers(
                s,
                total_names,
                self.device,
                require_buffer=(self.device.type == 'cuda'),
            )
            if dump_names:
                _n_dump = math.ceil(
                    self.maxiter / self.output_structures_per_step
                )
                self.dumper.start_from_arrays(
                    _n_dump,
                    *(getattr(s_cpu, _name).numpy() for _name in dump_names),
                    names=dump_names,
                )

            if batch_indices is None:
                _shape_center = th.mean(X, dim=1, keepdim=True)
            else:
                _shape_center = index_reduce(
                    X,
                    self.batch_scatter,
                    dim=1,
                    out_size=self.scatter_dim_out_size,
                ) / self.batch_tensor.reshape(1, -1, 1)
                _shape_center = _shape_center.index_select(
                    1, self.batch_scatter
                )

            if self.device.type == 'cuda':
                copy_stream = th.cuda.Stream()
                copy_event = th.cuda.Event()
            else:
                copy_stream = None
                copy_event = None

            dump_queue: queue.Queue = queue.Queue(maxsize=1)
            log_queue: queue.Queue = queue.Queue(maxsize=1)
            dump_thread = threading.Thread(
                target=self._do_async_dump, args=(dump_queue,), daemon=True
            )
            log_thread = threading.Thread(
                target=self._do_async_print, args=(log_queue,), daemon=True
            )
            self._dump_done = threading.Event()
            self._print_done = threading.Event()
            self._dump_done.set()
            self._print_done.set()

            try:
                dump_thread.start()
                log_thread.start()
                if self.is_compile:
                    _main_loop = th.compile(
                        self._main_for_loop, **self.compile_kwargs
                    )
                else:
                    _main_loop = self._main_for_loop
                _main_loop(
                    s, s_cpu, s_buf,
                    dump_names, log_names, total_names,
                    copy_stream, copy_event,
                    dump_queue, log_queue,
                    func, func_args, func_kwargs,
                    batch_indices,
                    _fix_shape_center, move_to_center_freq, _shape_center,
                )
                self._dump_done.wait()
                self._print_done.wait()
                if self.device.type == 'cuda':
                    th.cuda.synchronize(self.device)
            finally:
                dump_queue.put(None)
                dump_thread.join()
                log_queue.put(None)
                log_thread.join()

        if self.verbose > 0:
            self.logger.info(
                '-' * 100 + '\nMAIN LOOP Done. Total Time: '
                f'{time.perf_counter() - _t_main:<.4f} s\n'
            )

    def _main_for_loop(
            self,
            s,
            s_cpu,
            s_buf,
            dump_names,
            log_names,
            total_names,
            copy_stream,
            copy_event,
            dump_queue,
            log_queue,
            func,
            func_args,
            func_kwargs,
            batch_indices,
            fix_shape_center,
            move_to_center_freq,
            shape_center,
    ):
        """Execute the proposal/update loop with a shared snapshot protocol.

        Args:
            s: Live :class:`StdContainer` holding every allowed MC quantity.
            s_cpu: Pinned CPU mirror used by the dump and print consumers.
            s_buf: CUDA staging mirror used to protect asynchronous D2H copies;
                ``None`` on CPU.
            dump_names: Ordered state names written to the trajectory.
            log_names: Ordered state names passed to the log consumer.
            total_names: Union of dump and log names copied at each output
                iteration.
            copy_stream: CUDA stream used for D2H copies, or ``None`` on CPU.
            copy_event: CUDA event recorded after a D2H snapshot, or ``None``
                on CPU.
            dump_queue: Queue consumed by :meth:`_do_async_dump`.
            log_queue: Queue consumed by :meth:`_do_async_print`.
            func: Energy model used to evaluate proposed coordinates.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            batch_indices: Normalized irregular atom counts, or ``None`` for a
                regular batch. Passed to the array logger for splitting.
            fix_shape_center: Whether geometric-center restoration is enabled.
            move_to_center_freq: Number of iterations between center
                restorations.
            shape_center: Initial per-atom expanded geometric center used as
                the restoration target.

        Returns:
            None. ``s`` is updated in place. At output iterations, snapshots
            represent the state at the beginning of that iteration, matching
            the historical MC trajectory timing.

        Notes:
            Before reusing ``s_cpu``, the producer waits for both consumers to
            signal completion. On CUDA, the consumers additionally synchronize
            with ``copy_event`` before accessing NumPy views of pinned memory.
        """
        _is_cuda = self.device.type == 'cuda'
        for numit in range(self.maxiter):
            if numit % self.output_structures_per_step == 0 and total_names:
                # The two CPU buffers are shared with consumer threads. Wait
                # until both consumers release the previous snapshot before
                # overwriting any registered field.
                self._dump_done.wait()
                self._print_done.wait()

                if _is_cuda:
                    self._transfer_buffers_D2H(
                        s, s_buf, s_cpu, total_names,
                        copy_stream, self.device,
                    )
                    copy_event.record(copy_stream)
                    sync_event = copy_event
                else:
                    for _name in total_names:
                        getattr(s_cpu, _name).copy_(getattr(s, _name))
                    sync_event = None

                if dump_names:
                    self._dump_done.clear()
                    dump_queue.put((
                        self.dumper,
                        sync_event,
                        *(getattr(s_cpu, _name) for _name in dump_names),
                    ))
                if self.verbose > 0 and log_names:
                    self._print_done.clear()
                    log_queue.put((
                        sync_event,
                        numit,
                        batch_indices,
                        *(getattr(s_cpu, _name) for _name in log_names),
                    ))

            # Proposal/update work overlaps with disk I/O and log formatting.
            _X_old = s.X.clone()
            _energies_old = s.Energy.clone()
            _energies, _delta_E, _X = self._update_X(
                func, func_args, func_kwargs, _energies_old, s.X
            )
            _X_diff = _X - _X_old
            s.Energy = _energies
            s.delta_E = _delta_E
            s.X = _X
            s.is_accept = self.is_accept

            self._update_algo_param(numit, _X_diff)
            _temperature = 0. if self.T_now is None else float(self.T_now)
            s.temperature.fill_(_temperature)

            if fix_shape_center and numit % move_to_center_freq == 0:
                s.X.add_(shape_center - self.calc_shape_center(s.X))

    def initialize_algo_param(self):
        """Initialize algorithm-specific state before model evaluation.

        Subclasses may allocate proposal distributions, schedules, or cached
        tensors here. The method is called once per :meth:`run`, after batch
        metadata and selective-dynamics masks are available but before the
        initial energy evaluation.

        Returns:
            None.
        """
        raise NotImplementedError

    def _update_X(
            self,
            func,
            func_args,
            func_kwargs,
            energies_old: th.Tensor,
            X: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Propose coordinates and apply the algorithm's acceptance rule.

        Args:
            func: Energy model used to evaluate the proposed coordinates.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            energies_old: Accepted energies before the proposal, with shape
                ``[n_true_batch]``.
            X: Accepted coordinates before the proposal. Shape is
                ``[n_batch, n_atom, n_dim]`` for a regular batch or
                ``[1, sum(n_atoms), n_dim]`` for an irregular batch.

        Returns:
            Tuple ``(energies, delta_E, X_new)`` containing the accepted energy
            for each structure, the raw proposed energy change, and the
            accepted coordinates. Implementations must also refresh
            ``self.is_accept`` with a boolean tensor of shape
            ``[n_true_batch]``.
        """
        raise NotImplementedError

    def _update_algo_param(self, i: int, displace: th.Tensor) -> None:
        """Update algorithm-specific parameters after one proposal.

        Args:
            i: Zero-based MC iteration index of the proposal just completed.
            displace: Accepted coordinate displacement ``X_new - X_old``. Its
                shape matches the active coordinate tensor.

        Returns:
            None. Implementations update scheduling attributes such as the
            current temperature in place.
        """
        pass
