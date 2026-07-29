""" To support model running on multi-device in parallel and reduce to the master device. """
import warnings
from typing import List, Dict, Any, Tuple
import threading, queue, copy

import torch as th
from BUCToolkit.BatchStructures.batch import Batch
from BUCToolkit.utils.exceptions import FatalError
from BUCToolkit.utils.function_utils import _BaseWrapper, compare_tensors


class Model_Wrapper_pyg_MultiDevice(_BaseWrapper):
    """
    The pyg wrapper of multi-device version.
    """
    __slots__ = ('_model', 'forces', 'X', )
    grad_enabled = True  # caller grad mode shared with all persistent workers

    def __init__(self, model, pos_attr_name='pos', devices_list: List[str] = None,
                 is_dynamic_workload: bool = False) -> None:
        """
        A format transformer for converting Tensor X into PygData.pos
        Wrap the model(graph, ...) into f(X)

        Args:
            model: An instantiate nn.Module
            is_dynamic_workload: Whether to reduce the number of active devices
                as the batch workload decreases. The first non-empty batch
                calibrates the target workload per device. The default False
                preserves fixed-device LPT scheduling.

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Grad: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].

        """
        if type(is_dynamic_workload) is not bool:
            raise TypeError(f'is_dynamic_workload must be a bool, but got {type(is_dynamic_workload)}.')
        if not th.cuda.is_available():
            raise RuntimeError('CUDA not available, please check your devices or torch version.')
        if not isinstance(model, th.nn.Module):
            raise TypeError(f'model must be a Pytorch model, but got {type(model)}.')
        # get devices
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

        # initialise model device
        model = model.to(self.master_device)

        # get streams
        self.streams = [th.cuda.Stream(device=_) for _ in self.devices_list]

        super().__init__(model)
        self.pos_attr_name = pos_attr_name
        self.X = None
        # Build K replicas with only K - 1 deep copies. The original model is
        # moved through the first K - 1 devices while each copy is made, then
        # retained on the last device instead of making one more copy.
        self._scattered_model: Dict[str, th.nn.Module] = {
            _:copy.deepcopy(self._model.to(_)) for _ in self.devices_list[:-1]
        }
        self._scattered_model[self.devices_list[-1]] = self._model.to(self.devices_list[-1])
        # _BaseWrapper users expect _model to be the model on the master device.
        # The original object remains alive through the last dictionary entry.
        self._model = self._scattered_model[self.master_device]
        self._energy_piece: Dict[str, Any] = {_: None for _ in self.devices_list}
        self._forces_piece: Dict[str, Any] = {_: None for _ in self.devices_list}
        self._has_thread_ERROR: int = 0  # whether any worker has failed terminally
        self._thread_ERROR = None  # original exception raised inside a worker
        self._closed = False  # terminal lifecycle state; a closed wrapper cannot reopen
        self._close_cause = None  # fatal cause retained for later exception chaining
        self.is_dynamic_workload = is_dynamic_workload
        # Dynamic scheduling calibrates this once from the first non-empty
        # batch. It intentionally remains unchanged for the wrapper lifetime,
        # including when a later batch is larger than the calibration batch.
        self._target_workload_per_device = None

        # multi-threads
        self._all_queue = [queue.Queue(maxsize=1) for _, __ in enumerate(self.devices_list)]
        self._all_threads = [
            threading.Thread(target=self._async_call_model, args=(self._all_queue[_],), daemon=True)
            for _, __ in enumerate(self.devices_list)
        ]
        self._threads_events = [
            threading.Event() for _ in range(len(self.devices_list))
        ]
        for _ in self._all_threads: _.start()

    def _async_call_model(self, q: queue.Queue):
        """
        Consume ``(device, event, graph)`` jobs for one persistent worker.

        Worker exceptions are stored for ``Energy`` to raise on the caller
        thread. ``event`` is always set so that the caller cannot remain
        blocked after either a successful inference or a failed one.
        """
        while True:
            items = q.get()
            _device, event, graph = items[0], items[1], items[2]
            event: threading.Event
            if _device is None:
                break
            try:
                # PyTorch grad mode is thread-local. Energy publishes the
                # caller's mode through the class attribute before dispatch.
                th.set_grad_enabled(self.grad_enabled)
                origin_shape = getattr(graph, self.pos_attr_name).shape
                y = self._scattered_model[_device](graph)
                th.cuda.synchronize(_device)
                self._energy_piece[_device] = y['energy'].reshape(-1)  # (local_batch, )
                self._forces_piece[_device] = y['forces'].reshape(origin_shape)  # (-1, 3)
            except Exception as e:
                # A partial multi-device result is unusable, so a worker error
                # permanently terminates this wrapper after caller-side cleanup.
                self._thread_ERROR = e
                self._has_thread_ERROR = 1
                break
            finally:
                event.set()

    @staticmethod
    def _lpt_assign(tasks, device_num):
        """Run deterministic LPT for ``device_num`` candidate devices."""
        loads = [0] * device_num
        assignments = [[] for _ in range(device_num)]
        indices = [[] for _ in range(device_num)]
        for size, idx, grp in tasks:
            # ``min`` returns the lowest device index for equal loads. Together
            # with the task ordering below, this makes every candidate and its
            # restored batch order deterministic.
            dev = min(range(device_num), key=lambda _: loads[_])
            loads[dev] += size
            assignments[dev].append(grp)
            indices[dev].append(idx)
        return loads, assignments, indices

    def _balance_work_load_assign(self, X, graph) -> Tuple[Tuple[Batch, ...], th.Tensor, List[int], List[List[int]]] | Tuple[None, None, None, None]:
        """
        Balance structures by LPT and record how to restore batch order.

        ``indices[d]`` stores original graph indices in device-local order.
        ``pos_map[i]`` stores the position of original graph ``i`` after all
        device-local batches are concatenated in device order.
        """
        graph: Batch
        setattr(graph, self.pos_attr_name, X.reshape(-1, 3).contiguous())
        graph_list = graph.to_data_list()
        batch_sizes = [getattr(_, self.pos_attr_name).numel() for _ in graph_list]
        N = len(graph_list)
        if N == 0: return None, None, None, None
        # K is the number of leading devices/workers that will participate in
        # this call. Capping it by N prevents empty local batches.
        max_device_num = min(len(self.devices_list), N)

        # (sizes, index, graph_data)
        tasks = [(batch_sizes[i], i, grp) for i, grp in enumerate(graph_list)]

        # LPT sorts larger structures first. Original index is the stable
        # secondary key for equal-size structures.
        tasks.sort(key=lambda x: (-x[0], x[1]))

        if not self.is_dynamic_workload:
            # Fixed mode preserves the original behavior: every available
            # device participates, subject only to the structure-count cap.
            K = max_device_num
            _, assignments, indices = self._lpt_assign(tasks, K)
        elif self._target_workload_per_device is None:
            # The first complete, non-empty batch establishes one persistent
            # per-device target and must use all currently available devices.
            # Workload remains pos.numel(), matching the original LPT metric.
            K = max_device_num
            self._target_workload_per_device = sum(batch_sizes) / K
            _, assignments, indices = self._lpt_assign(tasks, K)
        else:
            target = self._target_workload_per_device
            candidates = list()
            for candidate_K in range(1, max_device_num + 1):
                loads, candidate_assignments, candidate_indices = self._lpt_assign(tasks, candidate_K)
                # The lexicographic score first keeps every active worker close
                # to the calibrated target, then minimizes the longest worker,
                # and finally prefers fewer devices for an exact tie.
                score = (
                    max(abs(load - target) for load in loads),
                    max(loads),
                    candidate_K,
                )
                candidates.append((score, candidate_assignments, candidate_indices))
            (score, assignments, indices) = min(candidates, key=lambda _: _[0])
            K = score[2]

        # Reformat assignments and build original-index -> scattered-position.
        assignments_tuple = tuple(Batch.from_data_list(dev_assign) for dev_assign in assignments)
        pos_map = [0] * N
        current_pos = 0
        for d in range(K):
            for idx in indices[d]:
                pos_map[idx] = current_pos
                current_pos += 1
        index_tensor = th.as_tensor(pos_map, device=self.master_device, dtype=th.int64)

        return assignments_tuple, index_tensor, pos_map, indices

    def _release_resources(self):
        """
        Stop all persistent workers before model references are released.
        """
        try:
            # One sentinel is sent to every queue, including a worker that may
            # already have exited after reporting an inference exception.
            for que in self._all_queue: que.put((None, None, None))
            for thr in self._all_threads: thr.join(timeout=60.)
            for thr in self._all_threads:
                if thr.is_alive():
                    raise RuntimeError(f"Thread {thr.name} stuck after 60s timeout")
            for _ in self.devices_list: th.cuda.synchronize(_)
            return 0
        except Exception as e:
            warnings.warn(f"An Error Occurred when trying to release resources: {e}")
            return 1

    def _raise_closed_error(self, *args, **kwargs):
        if self._close_cause is not None:
            raise FatalError('The multi-device model wrapper was closed after a fatal error.') from self._close_cause
        raise FatalError('The multi-device model wrapper has been closed.')

    def close(self):
        """
        Irreversibly close the wrapper and release worker-owned resources.
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
        if _state != 0:
            raise FatalError('Some errors occurred when releasing multi-device resources.') from self._close_cause
        self._scattered_model.clear()
        self._energy_piece.clear()
        self._forces_piece.clear()
        self._model = None

    def eval(self):
        for _ in self._scattered_model.values(): _.eval()

    def Energy(self, X, graph):

        batch_tensor = th.bincount(graph.batch).to(self.master_device)

        # TODO: pre-alloc to optimise in future
        # batch_ptr = th.cumsum(batch_tensor, dim=0)
        # force_cache = th.empty_like(X, device=self.master_device)
        # energy_cache = th.empty_like(batch_tensor, device=self.master_device)

        _data_split, resume_indices_tens, resume_indices_list, nested_resume_ind_list = self._balance_work_load_assign(X, graph)
        if _data_split is None:
            raise ValueError('ERROR in model wrapper: Got an empty input.')
        # Both resume indices map original graph index -> scattered position;
        # nested_resume_ind_list separately records original indices per device.
        chunk_sizes = [len(_) for _ in nested_resume_ind_list]  # structures assigned to each active device
        active_device_num = len(_data_split)  # workers participating in this batch
        # This class-wide value is read independently by every worker thread.
        Model_Wrapper_pyg_MultiDevice.grad_enabled = th.is_grad_enabled()

        # Assignments always target the leading K devices. Workers after K keep
        # their resident model replicas and remain blocked on their queues; no
        # worker or CUDA resource is recreated when dynamic mode reduces K.
        for evt in self._threads_events[:active_device_num]: evt.clear()
        for i, _inp in enumerate(_data_split):
            _inp: th.Tensor
            self._all_queue[i].put((self.devices_list[i], self._threads_events[i], _inp.to(self.devices_list[i], non_blocking=True)))
        for evt in self._threads_events[:active_device_num]: evt.wait()  # ensure synchronisation

        # safeguard
        if self._has_thread_ERROR == 1:
            _error = self._thread_ERROR
            self._close_cause = _error
            # Discard all partial results and stop every worker before exposing
            # the fatal error to the caller.
            self.close()
            raise FatalError(f'An Error Occurred during model calculation: {_error}') from _error

        # Worker outputs are concatenated in device order and detached before
        # the master-device algorithm consumes them independently of inference.
        energy_tensor = th.cat([
            self._energy_piece[_].to(self.master_device).detach() for _ in self.devices_list[:active_device_num]
        ], dim=0)  # (batch, )
        forces_tens_list = list()
        # resume_indices_tens maps original -> scattered. Its inverse gives the
        # scattered graph order needed to split concatenated forces by atom count.
        _rearranged_batch_tensor = batch_tensor.index_select(0, th.argsort(resume_indices_tens)).tolist()
        _ii = 0; _ptr = 0  # device index and offset in scattered graph order
        for _ in self.devices_list[:active_device_num]:
            _v = self._forces_piece[_]
            _local_index = _rearranged_batch_tensor[_ptr: _ptr + chunk_sizes[_ii]]
            forces_tens_list.extend(th.split(_v.to(self.master_device).detach(), _local_index))
            _ptr += chunk_sizes[_ii]
            _ii += 1
        # Restore energies and per-structure force blocks to original batch order.
        energy_tensor = energy_tensor.index_select(0, resume_indices_tens)
        forces_tensor = th.cat([forces_tens_list[_] for _ in resume_indices_list], dim=0)

        # Cache the exact input object and its forces so the immediately
        # following Grad call can reuse this inference result once.
        self.X = X
        self.forces = forces_tensor
        return energy_tensor

    def Grad(self, X, graph):
        origin_shape = X.shape
        if (self.X is None) or (not compare_tensors(X, self.X)):
            self.forces = None
        if self.forces is None:
            self.Energy(X, graph)
            force = self.forces
            self.forces = None
            return - force.reshape(origin_shape).contiguous()
        else:
            force = self.forces
            self.forces = None
            return - force.reshape(origin_shape).contiguous()
