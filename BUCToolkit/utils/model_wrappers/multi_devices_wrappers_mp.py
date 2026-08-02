"""Run synchronous multi-GPU inference with persistent worker processes.

The main process owns the MD state and one shared coordinate/energy/force
buffer. Each GPU process gathers its assigned atoms with ``index_select``, runs
one model replica, and restores disjoint results with ``index_copy_``. The main
process resumes only after every active GPU has completed the current step.
"""
import warnings
from typing import List
import time
import traceback

import torch as th
import torch.multiprocessing as mp
from BUCToolkit.BatchStructures.batch import Batch
from BUCToolkit.utils.exceptions import FatalError
from BUCToolkit.utils.function_utils import _BaseWrapper, compare_tensors


_WORKER_START_TIMEOUT = 300.
_WORKER_COMMAND_TIMEOUT = 60.
_WORKER_TERMINATE_TIMEOUT = 5.


def _async_call_model_process(device, model, pos_attr_name, connection):
    """Run one persistent model replica and serve commands from the parent.

    Args:
        device: Canonical CUDA device name assigned exclusively to this worker,
            such as ``"cuda:1"``.
        model: CPU-resident ``torch.nn.Module`` to copy to ``device``. It must
            accept a PyG Batch and return a mapping containing per-structure
            ``energy`` and per-atom ``forces`` tensors.
        pos_attr_name: Name of the Cartesian-coordinate attribute in the local
            graph, normally ``"pos"``.
        connection: Child endpoint of a duplex ``multiprocessing.Pipe`` used
            for commands, completion acknowledgements, and error reports.

    Returns:
        None. The function serves commands until ``close`` is received or an
        error terminates the worker.

    Notes:
        Every parent request is a tuple beginning with ``(command,
        generation)``. The remaining positional items depend on ``command``:

        - ``('configure', generation, payload)`` binds the shared allocation,
          the worker's disjoint structure/atom indices, and its local graph.
          ``payload`` contains ``shared_buffer``, ``atom_capacity``,
          ``structure_capacity``, ``structure_indices``, ``atom_indices``, and
          ``graph``. The index order must match the structure and atom order in
          that local graph.
        - ``('run', generation, is_grad_enabled)`` gathers the assigned
          coordinates, evaluates the model, and restores energy and force in
          original batch order.
        - ``('eval', generation)`` switches the local model to evaluation
          mode.
        - ``('to', generation, args, kwargs)`` applies a model conversion but
          replaces any requested device with this worker's assigned device.
        - ``('close', 0)`` terminates the worker and has no acknowledgement.

        The one-dimensional shared CPU allocation is laid out as
        ``[coordinates, energies, forces]``, with region lengths
        ``atom_capacity * 3``, ``structure_capacity``, and
        ``atom_capacity * 3``. All regions therefore share one dtype. A worker
        must be configured before ``run``; workers write disjoint output
        indices, so restoring output requires ``index_copy_`` rather than a
        numerical reduction.

        Startup reports ``(0, None)``. Every successful command except
        ``close`` reports ``(generation, None)`` only after its GPU work and
        device-to-host copies are complete. A command failure reports
        ``(generation, traceback_text)`` and permanently exits the worker.
    """
    try:
        th.set_num_threads(1)
        th.set_num_interop_threads(1)
        th.cuda.set_device(device)
        model = model.to(device)
        local_graph = None
        shared_coordinates = None
        shared_energies = None
        shared_forces = None
        structure_indices = None
        atom_indices = None
        coordinates_cpu = None
        energies_cpu = None
        forces_cpu = None
        connection.send((0, None))

        while True:
            # The positional command protocol and reply barrier are documented
            # above; Pipe traffic never carries per-step coordinates or output.
            command_items = connection.recv()
            command, generation = command_items[0], command_items[1]
            if command == 'close':
                break
            try:
                if command == 'configure':
                    payload = command_items[2]
                    shared_buffer = payload['shared_buffer']
                    atom_capacity = payload['atom_capacity']
                    structure_capacity = payload['structure_capacity']
                    # One shared allocation has stable coordinate, energy, and
                    # force regions. Rebinding occurs only when its capacity or
                    # this worker's LPT assignment changes.
                    coordinate_size = atom_capacity * 3
                    energy_offset = coordinate_size
                    force_offset = energy_offset + structure_capacity
                    shared_coordinates = shared_buffer[:coordinate_size].view(atom_capacity, 3)
                    shared_energies = shared_buffer[energy_offset:force_offset]
                    shared_forces = shared_buffer[force_offset:].view(atom_capacity, 3)
                    structure_indices = payload['structure_indices']
                    atom_indices = payload['atom_indices']
                    local_graph = payload['graph'].to(device)
                    # Pinned worker-local staging makes the H2D/D2H copies
                    # asynchronous; shared CPU storage itself is not pinned.
                    coordinates_cpu = th.empty(
                        (atom_indices.numel(), 3), dtype=shared_buffer.dtype,
                        pin_memory=True,
                    )
                    energies_cpu = th.empty(
                        structure_indices.numel(), dtype=shared_buffer.dtype,
                        pin_memory=True,
                    )
                    forces_cpu = th.empty(
                        coordinates_cpu.shape, dtype=shared_buffer.dtype,
                        pin_memory=True,
                    )
                    if getattr(local_graph, pos_attr_name).shape != coordinates_cpu.shape:
                        raise ValueError(
                            f'Assigned graph position shape {tuple(getattr(local_graph, pos_attr_name).shape)} '
                            f'does not match {tuple(coordinates_cpu.shape)}.'
                        )
                elif command == 'run':
                    if local_graph is None:
                        raise RuntimeError('Inference was requested before worker configuration.')
                    th.set_grad_enabled(command_items[2])
                    # Cached atom indices gather the worker's structures in its
                    # local Batch order without sending coordinates by Pipe.
                    th.index_select(shared_coordinates, 0, atom_indices, out=coordinates_cpu)
                    getattr(local_graph, pos_attr_name).copy_(coordinates_cpu, non_blocking=True)
                    model_output = model(local_graph)
                    energies = model_output['energy'].reshape(-1)
                    forces = model_output['forces'].reshape(-1, 3)
                    if energies.shape != energies_cpu.shape:
                        raise ValueError(
                            f'Expected energy shape {tuple(energies_cpu.shape)}, but got {tuple(energies.shape)}.'
                        )
                    if forces.shape != forces_cpu.shape:
                        raise ValueError(
                            f'Expected force shape {tuple(forces_cpu.shape)}, but got {tuple(forces.shape)}.'
                        )
                    if energies.dtype != shared_buffer.dtype or forces.dtype != shared_buffer.dtype:
                        raise TypeError(
                            f'Model output dtype must be {shared_buffer.dtype}, but got '
                            f'{energies.dtype} and {forces.dtype}.'
                        )
                    energies_cpu.copy_(energies.detach(), non_blocking=True)
                    forces_cpu.copy_(forces.detach(), non_blocking=True)
                    # A completion reply is not published until all GPU work
                    # and D2H copies on this device are globally complete.
                    th.cuda.synchronize(device)
                    # Assignments form disjoint complete partitions, so these
                    # writes restore original batch/atom order without a sum.
                    shared_energies.index_copy_(0, structure_indices, energies_cpu)
                    shared_forces.index_copy_(0, atom_indices, forces_cpu)
                elif command == 'eval':
                    model.eval()
                elif command == 'to':
                    args, kwargs = command_items[2], dict(command_items[3])
                    args = list(args)
                    if len(args) > 0 and isinstance(args[0], (str, int, th.device)):
                        args[0] = device
                    if 'device' in kwargs:
                        kwargs['device'] = device
                    model = model.to(*args, **kwargs)
                else:
                    raise ValueError(f'Unknown worker command {command!r}.')
            except Exception:
                connection.send((generation, traceback.format_exc()))
                break
            connection.send((generation, None))
    except Exception:
        try:
            connection.send((0, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class Model_Wrapper_pyg_MultiDevice(_BaseWrapper):
    """Expose synchronous PyG energy and gradient calls across CUDA devices.

    Persistent spawned processes own the model replicas. Coordinates and
    restored outputs cross process boundaries through one shared CPU tensor;
    control pipes carry only commands and occasional assignment updates.
    """
    __slots__ = ('_model', 'forces', 'X', )
    def __init__(self, model, pos_attr_name='pos', devices_list: List[str] = None,
                 is_dynamic_workload: bool = False) -> None:
        """
        Create persistent workers and place one model replica on each device.

        Args:
            model: Instantiated PyTorch module returning ``energy`` and
                ``forces`` tensors from a PyG batch.
            pos_attr_name: Graph attribute containing Cartesian coordinates.
            devices_list: Canonical CUDA device names. All visible CUDA
                devices are used when this argument is ``None``.
            is_dynamic_workload: Whether to reduce the number of active devices
                as the batch workload decreases. The first non-empty batch
                calibrates the target workload per device. The default False
                preserves fixed-device LPT scheduling.

        Raises:
            TypeError: If ``model`` or ``is_dynamic_workload`` has an invalid
                type.
            RuntimeError: If fewer than two unique requested CUDA devices are
                available.
            FatalError: If a worker cannot initialize its model replica.

        Notes:
            Workers use the ``spawn`` context. The calling program must follow
            Python's normal guarded-entry-point requirement for spawned
            processes. Device placement remains fixed by ``devices_list``.
        """
        if type(is_dynamic_workload) is not bool:
            raise TypeError(f'is_dynamic_workload must be a bool, but got {type(is_dynamic_workload)}.')
        if not th.cuda.is_available():
            raise RuntimeError('CUDA not available, please check your devices or torch version.')
        if not isinstance(model, th.nn.Module):
            raise TypeError(f'model must be a Pytorch model, but got {type(model)}.')
        # Device order is authoritative throughout the wrapper: worker,
        # connection, process, and per-worker cache index all use this order.
        if devices_list is None:
            self.devices_list: List[str] = [f"cuda:{_}" for _ in range(th.cuda.device_count())]
        else:
            self.devices_list = [str(_) for _ in devices_list]
            # validate
            try:
                for _ in self.devices_list:
                    if (not _.startswith('cuda:')) or (not _[5:].isdigit()) or (_ != f'cuda:{int(_[5:])}'):
                        raise ValueError(f'Device name must be in the canonical "cuda:$i" format, but got "{_}".')
                    th.cuda.get_device_name(_)
            except Exception as e:
                raise RuntimeError(f'There is device that not available due to "{e}".')
        if len(self.devices_list) <= 1:
            raise RuntimeError('There is only ONE device on this computer, multi-devices are impossible.')
        if len(set(self.devices_list)) != len(self.devices_list):
            raise RuntimeError('There are duplicated devices in devices_list.')
        self.master_device = self.devices_list[0]

        # Keep the initial model on CPU so spawn never serializes CUDA state.
        model = model.to('cpu')
        super().__init__(model)
        self.pos_attr_name = pos_attr_name

        # ``Energy`` retains the exact input object in X and the corresponding
        # force in the inherited ``forces`` field. The next matching ``Grad``
        # consumes that force once instead of repeating model inference.
        self.X = None

        # Closing is terminal. A fatal worker error is retained so every later
        # public call can fail consistently with the original cause chained.
        self._closed = False
        self._close_cause = None

        # Fixed mode reuses one LPT assignment while the graph is unchanged.
        # Dynamic mode reruns LPT each step, but calibrates this workload target
        # only from the first non-empty batch and keeps it for the wrapper life.
        self.is_dynamic_workload = is_dynamic_workload
        self._target_workload_per_device = None

        # Every synchronous dispatch increments generation; replies must carry
        # the same value. Buffer version changes only when shared storage is
        # reallocated, forcing affected workers to rebind their tensor views.
        self._generation = 0
        self._buffer_version = 0

        # One flat, shared CPU tensor owns all storage. The three named tensors
        # are non-owning views into its coordinate, energy, and force regions.
        # Capacities describe allocated rows, not the current batch sizes; they
        # grow geometrically and do not shrink between calls.
        self._shared_buffer = None
        self._shared_coordinates = None
        self._shared_energies = None
        self._shared_forces = None
        self._atom_capacity = 0
        self._structure_capacity = 0

        # These caches avoid graph reconstruction and worker reconfiguration.
        # A worker configuration is exactly ``(graph_signature,
        # structure_indices, buffer_version)``. LPT assignments always use a
        # leading prefix of workers; inactive workers retain their model and
        # wait on their Pipe.
        self._graph_signature_cache = None
        self._atom_counts_cache = None
        self._worker_structure_indices_cache = None
        self._worker_configurations = [None] * len(self.devices_list)

        # Use an isolated spawn context instead of changing the application's
        # global multiprocessing start method or forking initialized CUDA.
        self._process_context = mp.get_context('spawn')
        # Connections and processes remain one-to-one with ``devices_list`` and
        # preserve its order for the complete wrapper lifetime.
        self._connections = list()
        self._all_processes = list()
        try:
            for device in self.devices_list:
                parent_connection, child_connection = self._process_context.Pipe()
                process = self._process_context.Process(
                    target=_async_call_model_process,
                    args=(
                        device, self._model, self.pos_attr_name, child_connection,
                    ),
                    daemon=True,
                )
                process.start()
                child_connection.close()
                self._connections.append(parent_connection)
                self._all_processes.append(process)
            deadline = time.monotonic() + _WORKER_START_TIMEOUT
            for worker_index, connection in enumerate(self._connections):
                remaining = deadline - time.monotonic()
                if remaining <= 0.:
                    raise TimeoutError('Timed out while initializing multi-device workers.')
                if not connection.poll(remaining):
                    process = self._all_processes[worker_index]
                    if process.exitcode is not None:
                        raise RuntimeError(
                            f'Worker {worker_index} exited with code {process.exitcode} '
                            'during initialization.'
                        )
                    raise TimeoutError(
                        f'Worker {worker_index} timed out during initialization.'
                    )
                completed_generation, error_text = connection.recv()
                if completed_generation != 0:
                    raise RuntimeError(
                        f'Worker {worker_index} reported invalid initialization '
                        f'generation {completed_generation}.'
                    )
                if error_text is not None:
                    raise RuntimeError(f'Worker {worker_index} failed:\n{error_text}')
        except Exception as e:
            self._close_cause = e
            self._closed = True
            self._release_resources()
            raise FatalError(f'Failed to initialise multi-device workers: {e}') from e
        # Each child now owns its replica; the parent does no model inference.
        self._model = None

    @staticmethod
    def _lpt_assign(tasks, n_device):
        """Run deterministic LPT for ``n_device`` candidate devices."""
        loads = [0] * n_device
        indices = [[] for _ in range(n_device)]
        for size, structure_index in tasks:
            # ``min`` returns the lowest device index for equal loads. Together
            # with the task ordering below, this makes every candidate and its
            # restored batch order deterministic.
            device_index = min(range(n_device), key=lambda index: loads[index])
            loads[device_index] += size
            indices[device_index].append(structure_index)
        return loads, indices

    def _balance_work_load_indices(self, workloads: List[int]) -> List[List[int]]:
        """Return the existing deterministic LPT assignment without rebuilding graphs."""
        n_structure = len(workloads)
        if n_structure == 0:
            return list()
        # Only leading workers participate. Capping the count by the number of
        # structures prevents empty local batches.
        max_n_device = min(len(self.devices_list), n_structure)

        tasks = [(workloads[index], index) for index in range(n_structure)]

        # LPT sorts larger structures first. Original index is the stable
        # secondary key for equal-size structures.
        tasks.sort(key=lambda x: (-x[0], x[1]))

        if not self.is_dynamic_workload:
            # Fixed mode preserves the original behavior: every available
            # device participates, subject only to the structure-count cap.
            n_active_device = max_n_device
            _, worker_structure_indices = self._lpt_assign(tasks, n_active_device)
        elif self._target_workload_per_device is None:
            # The first complete, non-empty batch establishes one persistent
            # per-device target and must use all currently available devices.
            # Workload remains pos.numel(), matching the original LPT metric.
            n_active_device = max_n_device
            self._target_workload_per_device = sum(workloads) / n_active_device
            _, worker_structure_indices = self._lpt_assign(tasks, n_active_device)
        else:
            target = self._target_workload_per_device
            candidates = list()
            for n_candidate_device in range(1, max_n_device + 1):
                loads, candidate_indices = self._lpt_assign(tasks, n_candidate_device)
                # The lexicographic score first keeps every active worker close
                # to the calibrated target, then minimizes the longest worker,
                # and finally prefers fewer devices for an exact tie.
                score = (
                    max(abs(load - target) for load in loads),
                    max(loads),
                    n_candidate_device,
                )
                candidates.append((score, candidate_indices))
            (_, worker_structure_indices) = min(candidates, key=lambda _: _[0])

        return worker_structure_indices

    def _dispatch_and_wait(self, worker_indices, command,
                           shared_args=tuple(), worker_args=None):
        """Send one generation and wait until every selected worker completes."""
        self._generation += 1
        generation = self._generation
        error = None
        for worker_index in worker_indices:
            process = self._all_processes[worker_index]
            if process.exitcode is not None:
                error = RuntimeError(
                    f'Worker {worker_index} exited with code {process.exitcode}.'
                )
                break
        if error is None:
            for worker_index in worker_indices:
                if worker_args is None:
                    command_args = shared_args
                else:
                    command_args = worker_args[worker_index]
                self._connections[worker_index].send(
                    (command, generation, *command_args)
                )
            deadline = time.monotonic() + _WORKER_COMMAND_TIMEOUT
            for worker_index in worker_indices:
                remaining = deadline - time.monotonic()
                if remaining <= 0. or not self._connections[worker_index].poll(remaining):
                    error = TimeoutError(
                        f'Worker {worker_index} timed out during generation {generation}.'
                    )
                    break
                try:
                    completed_generation, error_text = self._connections[worker_index].recv()
                except EOFError:
                    error = RuntimeError(
                        f'Worker {worker_index} closed its connection during generation {generation}.'
                    )
                    break
                if completed_generation != generation:
                    error = RuntimeError(
                        f'Worker {worker_index} reported generation {completed_generation}, '
                        f'expected {generation}.'
                    )
                    break
                if error_text is not None:
                    error = RuntimeError(
                        f'Worker {worker_index} failed during {command}:\n{error_text}'
                    )
                    break
        if error is not None:
            self._close_cause = error
            self.close()
            raise FatalError(f'An Error Occurred during model calculation: {error}') from error

    def _release_resources(self):
        """
        Stop all persistent workers before model references are released.
        """
        try:
            for connection in self._connections:
                try:
                    connection.send(('close', 0))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for process in self._all_processes:
                process.join(timeout=_WORKER_COMMAND_TIMEOUT)
            release_state = 0
            for process in self._all_processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=_WORKER_TERMINATE_TIMEOUT)
                    release_state = 1
            for connection in self._connections:
                connection.close()
            return release_state
        except Exception as e:
            warnings.warn(f"An Error Occurred when trying to release resources: {e}")
            return 1

    def _raise_closed_error(self, *args, **kwargs):
        if self._close_cause is not None:
            raise FatalError('The multi-device model wrapper was closed after a fatal error.') from self._close_cause
        raise FatalError('The multi-device model wrapper has been closed.')

    def close(self):
        """Irreversibly close the wrapper and release worker resources.

        Returns:
            None.

        Raises:
            FatalError: If a worker process cannot be released normally.
        """
        if self._closed: return
        self._closed = True
        self.X = None
        self.forces = None
        # Redirect first so every later public operation fails consistently
        # without adding a closed-state branch to each hot-path method.
        for _ in ('Energy', 'Grad', 'to', 'eval'):
            setattr(self, _, self._raise_closed_error)
        _state = self._release_resources()
        if _state != 0 and self._close_cause is None:
            raise FatalError('Some errors occurred when releasing multi-device resources.') from self._close_cause
        self._model = None
        self._shared_buffer = None
        self._shared_coordinates = None
        self._shared_energies = None
        self._shared_forces = None

    def eval(self):
        """Set every worker-owned model replica to evaluation mode.

        Returns:
            None.
        """
        worker_indices = list(range(len(self.devices_list)))
        self._dispatch_and_wait(worker_indices, 'eval')

    def to(self, *args, **kwargs):
        """Apply a model conversion while retaining each worker's CUDA device.

        Args:
            *args: Positional arguments accepted by ``torch.nn.Module.to``.
                An explicit device is replaced by the worker's assigned device.
            **kwargs: Keyword arguments accepted by ``torch.nn.Module.to``.

        Returns:
            The wrapper itself.
        """
        worker_indices = list(range(len(self.devices_list)))
        self._dispatch_and_wait(
            worker_indices, 'to', shared_args=(args, kwargs)
        )
        return self

    def Energy(self, X, graph):
        """Evaluate and restore one complete batch of structure energies.

        Args:
            X: Coordinate tensor ending in shape ``[..., n_atom, 3]`` on the
                master device.
            graph: PyG Batch whose non-position attributes describe the same
                structures and whose ``batch`` tensor maps atoms to structures.

        Returns:
            A detached energy tensor on the master device in original
            structure order.

        Raises:
            ValueError: If the coordinate and graph atom counts differ or the
                input batch is empty.
            FatalError: If a worker fails, exits, or times out.

        Notes:
            The returned energy and cached force are available only after all
            active devices complete this step. Model outputs must share the
            coordinate dtype because all three arrays occupy one typed shared
            allocation.
        """
        if not th.is_tensor(X):
            raise TypeError(f'X must be a torch.Tensor, but got {type(X)}.')
        if X.ndim < 2 or X.shape[-1] != 3:
            raise ValueError(
                f'X must end in Cartesian dimension 3, but got shape {tuple(X.shape)}.'
            )
        has_graph_interface = (
            hasattr(graph, 'keys')
            and hasattr(graph, 'to_data_list')
            and hasattr(graph, 'batch')
        )
        if not has_graph_interface:
            raise TypeError(
                f'graph must provide keys, to_data_list(), and batch, but got {type(graph)}.'
            )
        if not th.is_tensor(graph.batch):
            raise TypeError(
                f'graph.batch must be a torch.Tensor, but got {type(graph.batch)}.'
            )
        # Cache validity covers every retained non-position graph attribute.
        # Tensor version counters detect in-place metadata changes without
        # reading tensor values or synchronizing CUDA.
        keys = graph.keys() if callable(graph.keys) else graph.keys
        graph_attributes = list()
        for key in sorted(keys):
            if key == self.pos_attr_name:
                continue
            value = getattr(graph, key)
            if th.is_tensor(value):
                value_signature = (
                    value.untyped_storage().data_ptr(), value.storage_offset(),
                    tuple(value.shape), tuple(value.stride()), value.device,
                    value.dtype, value._version,
                )
            elif isinstance(value, (str, int, float, bool, type(None))):
                value_signature = value
            else:
                value_signature = (id(value), type(value))
            graph_attributes.append((key, value_signature))
        graph_signature = id(graph), type(graph), tuple(graph_attributes)
        has_graph_changed = graph_signature != self._graph_signature_cache
        if has_graph_changed:
            atom_counts = th.bincount(graph.batch).to('cpu').tolist()
        else:
            atom_counts = self._atom_counts_cache
        n_atom = sum(atom_counts)
        if X.numel() != n_atom * 3:
            raise ValueError(
                f'X and graph contain different atom counts: {X.numel() // 3} and {n_atom}.'
            )

        # Dynamic mode intentionally recomputes LPT every step. Fixed mode
        # reuses its assignment until the retained graph changes.
        require_balance = (
            self.is_dynamic_workload
            or self._worker_structure_indices_cache is None
            or has_graph_changed
        )
        if require_balance:
            worker_structure_indices = self._balance_work_load_indices(
                [n_structure_atom * 3 for n_structure_atom in atom_counts]
            )
            if len(worker_structure_indices) == 0:
                raise ValueError('ERROR in model wrapper: Got an empty input.')
            worker_structure_indices = tuple(
                tuple(indices) for indices in worker_structure_indices
            )
        else:
            worker_structure_indices = self._worker_structure_indices_cache

        # Coordinates, energies, and forces are views into one flat shared
        # allocation. Capacity grows geometrically and never shrinks, so a
        # decreasing dynamic batch does not remap shared memory.
        require_buffer_allocation = (
            self._shared_buffer is None
            or n_atom > self._atom_capacity
            or len(atom_counts) > self._structure_capacity
            or self._shared_buffer.dtype != X.dtype
        )
        if require_buffer_allocation:
            self._atom_capacity = max(n_atom, max(1, self._atom_capacity * 2))
            self._structure_capacity = max(
                len(atom_counts), max(1, self._structure_capacity * 2)
            )
            coordinate_size = self._atom_capacity * 3
            energy_offset = coordinate_size
            force_offset = energy_offset + self._structure_capacity
            self._shared_buffer = th.empty(
                force_offset + coordinate_size, dtype=X.dtype
            ).share_memory_()
            self._shared_coordinates = self._shared_buffer[:coordinate_size].view(
                self._atom_capacity, 3
            )
            self._shared_energies = self._shared_buffer[energy_offset:force_offset]
            self._shared_forces = self._shared_buffer[force_offset:].view(
                self._atom_capacity, 3
            )
            self._buffer_version += 1

        # Rebuild only workers whose LPT assignment, graph metadata, or shared
        # allocation changed. Dynamic LPT may therefore run every step without
        # causing Batch reconstruction when it returns the same assignment.
        workers_to_configure = list()
        for worker_index, structure_indices in enumerate(worker_structure_indices):
            configuration = (graph_signature, structure_indices, self._buffer_version)
            if (
                require_buffer_allocation
                or self._worker_configurations[worker_index] != configuration
            ):
                workers_to_configure.append(worker_index)
        if len(workers_to_configure) > 0:
            setattr(graph, self.pos_attr_name, X.reshape(-1, 3).contiguous())
            graph_list = graph.to_data_list()
            atom_offsets = [0]
            for n_structure_atom in atom_counts:
                atom_offsets.append(atom_offsets[-1] + n_structure_atom)
            worker_args = dict()
            for worker_index in workers_to_configure:
                structure_indices = worker_structure_indices[worker_index]
                atom_indices = list()
                for structure_index in structure_indices:
                    atom_indices.extend(range(
                        atom_offsets[structure_index], atom_offsets[structure_index + 1]
                    ))
                worker_args[worker_index] = ({
                    'shared_buffer': self._shared_buffer,
                    'atom_capacity': self._atom_capacity,
                    'structure_capacity': self._structure_capacity,
                    'structure_indices': th.as_tensor(structure_indices, dtype=th.int64),
                    'atom_indices': th.as_tensor(atom_indices, dtype=th.int64),
                    'graph': Batch.from_data_list(
                        [graph_list[index] for index in structure_indices]
                    ).to('cpu'),
                },)
            self._dispatch_and_wait(
                workers_to_configure, 'configure', worker_args=worker_args,
            )
            for worker_index in workers_to_configure:
                self._worker_configurations[worker_index] = (
                    graph_signature, worker_structure_indices[worker_index],
                    self._buffer_version,
                )

        self._graph_signature_cache = graph_signature
        self._atom_counts_cache = atom_counts
        self._worker_structure_indices_cache = worker_structure_indices

        # The main process publishes one complete coordinate frame before
        # waking workers. Worker completion is a per-step barrier before these
        # output views are copied back to the master GPU.
        setattr(graph, self.pos_attr_name, X.reshape(-1, 3).contiguous())
        self._shared_coordinates[:n_atom].copy_(
            X.detach().reshape(-1, 3), non_blocking=False
        )
        n_active_device = len(worker_structure_indices)

        # Assignments always target leading devices. Inactive workers retain
        # their model replicas and wait on their pipes, so changing the active
        # count never recreates a process or CUDA context.
        self._dispatch_and_wait(
            list(range(n_active_device)), 'run',
            shared_args=(th.is_grad_enabled(),),
        )

        energy_tensor = self._shared_energies[:len(atom_counts)].to(self.master_device)
        forces_tensor = self._shared_forces[:n_atom].to(self.master_device)

        # Cache the exact input object and its forces so the immediately
        # following Grad call can reuse this inference result once.
        self.X = X
        self.forces = forces_tensor
        return energy_tensor

    def Grad(self, X, graph):
        """Return the negative model force for one coordinate batch.

        Args:
            X: Coordinate tensor with the same layout accepted by ``Energy``.
            graph: PyG Batch matching ``X``.

        Returns:
            A contiguous gradient tensor with the same shape as ``X``. The
            immediately preceding ``Energy`` result is consumed once; otherwise
            this method performs one synchronous energy/force evaluation.
        """
        origin_shape = X.shape
        if (self.X is None) or (not compare_tensors(X, self.X)):
            self.forces = None
        if self.forces is None:
            self.Energy(X, graph)
        force = self.forces
        self.forces = None
        return -force.reshape(origin_shape).contiguous()
