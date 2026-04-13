#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseMC.py
#  Environment: Python 3.12

import logging
import math
import sys
from itertools import accumulate
from typing import Dict, Any, Optional, Sequence, Tuple, List
import time
import threading, queue
import warnings
import os

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.BatchOptim._utils._warnings import NotConvergeWarning
from BUCToolkit.utils._Element_info import ATOMIC_SYMBOL
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BUCToolkit.utils.setup_loggers import has_any_handler, clear_all_handlers, BaseIO
from BUCToolkit.utils.index_ops import index_reduce
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.BatchStructures.StructuresIO import structures_io_dumper


class _BaseMC(BaseIO):
    def __init__(
            self,
            iter_scheme: str,
            maxiter: int = 100,
            output_file: str|None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 0,
            is_compile: bool = False,
            compile_kwargs: Dict|None = None,
    ) -> None:
        r"""
        A Base Framework of Algorithm for optimization.
        
        Args:
            maxiter: int, max iterations.
            output_file: the path to the binary file that stores trajectories.
            output_structures_per_step: int, output structures per output_structures_per_step steps.
            device: The device that program runs on.
            verbose: amount of print information.
            is_compile: whether to use jit to compile the update function or not.
            compile_kwargs: keyword arguments passed to compile. Only work when is_compile is True.

        Method:
            run: running the main optimization program.

        """
        self.batch_scatter = None
        self.batch_tensor = None
        self.atom_masks = None
        self.T_now = None
        self.is_compile = is_compile
        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else dict()

        self.iterform = str(iter_scheme)
        self.verbose = int(verbose)
        self.device = device
        self.maxiter = int(maxiter)
        self.output_structures_per_step = int(output_structures_per_step)

        self.n_batch, self.n_atom, self.n_dim = None, None, None
        self.is_concat_X = False   # whether the output of `func` was concatenated.
        self.scatter_dim_out_size = None

        # An inner attr that lets the dumper do not close after `self.run`.
        # It is used to contiguously run within a loop.
        # Adv. API `MonteCarlo` turns on it.
        self._HOLD_DUMPER = False

        # logger
        super().__init__(output_file)
        self.init_logger('Main.MC')

    def _do_async_dump(self, q: queue.Queue):
        """
        A backend thread to async. dump
        Args:
            q: queue to receive data. contains: tuple of (dumper, event, *data)

        Returns: None

        """
        while True:
            try:
                dumper, event, _print_E, _print_X = q.get()
                if dumper is None:
                    break
                # event: th.cuda.Event, ensure copy done
                event.synchronize()
                dumper.step(
                    _print_E.numpy(),
                    _print_X.numpy(),
                )
            except Exception as e:
                self.logger.error(f"Error: Failed to dump data due to \"{e}\"")

    def _do_async_print(self, q: queue.Queue):
        """

        Returns:

        """
        # Print information / Verbose
        while True:
            numit, _print_E, _print_dE, batch_indices, X, batch_slice_indx = q.get()
            if numit is None:
                break
            if self.verbose > 0:
                self.logger.info(
                    f"ITERATION    {numit:>5d}\n "
                    f"delta E:     {np.array2string(_print_dE.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"Accepted:    {np.array2string(self.is_accept.numpy(force=True))}\n "
                    f"Energies:    {np.array2string(_print_E.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"Temperature: {self.T_now:.3e}\n "
                    # f"TIME:        {time.perf_counter() - t_step:>6.4f} s"
                )
                # t_step = time.perf_counter()
            if self.verbose > 1:
                # split batches if specified batch
                if batch_indices is not None:
                    X_np = X.numpy(force=True)
                    X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                else:
                    X_tup = (X.numpy(force=True),)
                self.logger.info(f" Coordinates:\n")
                X_str = [
                    np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    for xi in X_tup
                ]
                [self.logger.info(f'{x_str}\n') for x_str in X_str]

    def calc_shape_center(self, Xr, ) -> th.Tensor:
        """
        calculate shape center based on Xr.
        Args:
            Xr:

        Returns: shape_center

        """
        if self.batch_scatter is None:
            # initialize topologie
            shape_center = th.mean(Xr, dim=1, keepdim=True)  # (n_batch, 1, n_dim)
        else:
            # initialize topologie
            shape_center = index_reduce(Xr, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size) / self.batch_tensor  # (1, n_batch, n_dim)
            shape_center = shape_center.index_select(1, self.batch_scatter)  # (1, sumN*A, n_dim)

        return shape_center

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
            move_to_center_freq: int = -1
    ):
        try:
            if th.device(self.device).type == "cuda":
                self.__run_on_cuda(
                    func,
                    X,
                    Element_list,
                    Cell_vector,
                    func_args,
                    func_kwargs,
                    batch_indices,
                    fixed_atom_tensor,
                    move_to_center_freq
                )
            elif th.device(self.device).type == "cpu":
                self.__run_on_cpu(
                    func,
                    X,
                    Element_list,
                    Cell_vector,
                    func_args,
                    func_kwargs,
                    batch_indices,
                    fixed_atom_tensor,
                    move_to_center_freq
                )
            else:
                raise NotImplementedError(F"device {self.device} not supported.")
        finally:
            pass

    def __run_on_cuda(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]] | None = None,
            Cell_vector: th.Tensor | None = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            move_to_center_freq: int = -1
    ) -> None:
        """
        Run MC Algorithm.

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            Cell_vector: Tensor[n_batch, 3, 3], the cell vectors. Only for logging out information, not involves in calculation.
            Element_list: The list of elements/atom types. Only for logging out information, not involves in calculation.
                if None, all atomic number of points will be labeled to 0.
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]
            move_to_center_freq: the period of translating coordinates and velocities of atoms into the shape center & 0.
                Note that is not MASS CENTER but shape center, different from MD.
                if `move_to_center_freq` <= 0, the translation would not apply.

        Return: None
        """

        t_main = time.perf_counter()
        if func_kwargs is None:
            func_kwargs = dict()
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            if len(X.shape) == 2:
                X = X.unsqueeze(0)
            elif len(X.shape) != 3:
                raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (Tuple, List)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be Sequence[int] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            n_true_batch = len(batch_indices)
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )
            self.scatter_dim_out_size = self.batch_scatter.max().item() + 1
        else:
            n_true_batch = n_batch
            self.batch_scatter = None
            self.batch_tensor = None
            batch_slice_indx = None
        # initialize vars
        maxiter = self.maxiter
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        if Element_list is not None:
            atomic_numbers = list()
            for _Elem in Element_list:
                atomic_numbers.append([ATOMIC_SYMBOL[__elem] if isinstance(__elem, str) else int(__elem) for __elem in _Elem])
        else:
            atomic_numbers = [[0]*self.n_atom]*self.n_batch
        if not isinstance(move_to_center_freq, int):
            raise TypeError(f'`move_to_center_freq` must be an integer, but got {type(move_to_center_freq)}.')
        is_fix_mass_center = (move_to_center_freq > 0)

        # Selective dyamics
        if fixed_atom_tensor is None:
            self.atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            self.atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'The shape of fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        # other check
        if (not isinstance(maxiter, int)) or (maxiter <= 0):
            raise ValueError(f'Invalid value of maxiter: {maxiter}. It would be an integer greater than 0.')

        # set variables device
        func = preload_func(func, self.device)
        X = X.to(self.device)

        # calc. freedom degree
        if batch_indices is None:
            _free_degree = X.shape[1] * n_dim
            if is_fix_mass_center:
                _free_degree -= 3
            self.free_degree = th.full((n_batch,), _free_degree, dtype=th.int64, device=self.device)
            n_reduce = th.where(th.abs(self.atom_masks) < 1e-6, 1, 0).sum(dim=(-2, -1))  # (n_batch, )
            self.free_degree -= n_reduce
        else:
            self.free_degree = self.batch_tensor * n_dim  # (n_batch, )
            if is_fix_mass_center:
                self.free_degree -= 3
            n_reduce_tensor = th.where(th.abs(self.atom_masks) < 1e-6, 1, 0).sum(dim=-1)
            n_reduce = index_reduce(n_reduce_tensor, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size).squeeze(0)  # (n_batch, )
            self.free_degree -= n_reduce

        # initialize the dumper
        X_arr = X.numpy(force=True)
        _x_dtype = X_arr.dtype.str
        atom_masks_arr = self.atom_masks.numpy(force=True).astype(_x_dtype)
        _num_dump = math.ceil(self.maxiter / self.output_structures_per_step)
        # pre-allocate the tensors
        #   _buf_* is the vars on GPU that apply copy.
        #   _print_* is the vars on CPU that async. do D2H for _buf_*.
        if batch_indices is not None:
            _print_E = th.empty(len(batch_indices), device='cpu', dtype=th.float32, pin_memory=True)
            _buf_E = th.empty(len(batch_indices), device=self.device, dtype=th.float32)
            _print_dE = th.zeros(len(batch_indices), device='cpu', dtype=th.float32, pin_memory=True)
            _buf_dE = th.empty(len(batch_indices), device=self.device, dtype=th.float32)
        else:
            _print_E = th.empty(n_batch, device='cpu', dtype=th.float32, pin_memory=True)
            _buf_E = th.empty(n_batch, device=self.device, dtype=th.float32)
            _print_dE = th.zeros(n_batch, device='cpu', dtype=th.float32, pin_memory=True)
            _buf_dE = th.empty(n_batch, device=self.device, dtype=th.float32)
        _print_X = th.empty_like(X, device='cpu', dtype=th.float32, pin_memory=True)
        _buf_X = th.empty_like(X, device=self.device, dtype=th.float32)
        # write head information.
        if Cell_vector is None:
            Cell_vector = np.zeros((n_true_batch, 3, 3), dtype=np.float32)
        elif isinstance(Cell_vector, th.Tensor):
            Cell_vector = Cell_vector.numpy(force=True)
        elif not isinstance(Cell_vector, np.ndarray):
            Cell_vector = np.asarray(Cell_vector)
        if self.batch_tensor is not None:
            self.dumper.start_from_arrays(
                1,
                self.batch_tensor.numpy(force=True),  # batch indices
                Cell_vector,  # cell
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            self.dumper.step(
                self.batch_tensor.numpy(force=True),
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )
        else:
            self.dumper.start_from_arrays(
                1,
                Cell_vector,
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            self.dumper.step(
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )
        # continue to write main data
        self.dumper.start_from_arrays(
            _num_dump,
            _print_E.numpy(),
            _print_X.numpy(),
        )

        # initialize
        t_step = time.perf_counter()
        self.initialize_algo_param()
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
            self.logger.info(f'Iteration Scheme: {self.iterform}')
        self.logger.info('-' * 100)
        # MAIN LOOP
        with th.no_grad():
            energies: th.Tensor = func(X, *func_args, **func_kwargs)
            delta_E: th.Tensor = th.zeros_like(energies)
            # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
            if energies.shape[0] != self.n_batch:
                if batch_indices is None:
                    raise ValueError(
                        f"batch indices is None "
                        f"while shape of model output ({energies.shape}) does not match batch size ({self.n_batch})."
                    )
                if energies.shape[0] != n_true_batch:
                    raise ValueError(f"shape of output ({energies.shape}) does not match given batch indices")
            self.is_concat_X = (batch_indices is not None)
            self.is_accept = th.full_like(energies, False, dtype=th.bool)

            # preload a graph of shape center. A very small overhead that no longer `is_fix_center`, computation does.
            if batch_indices is None:
                SHAPE_CENTER = th.mean(X, dim=1, keepdim=True)  # (n_batch, 1, n_dim)
            else:
                SHAPE_CENTER = index_reduce(X, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size) / self.batch_tensor
                SHAPE_CENTER = SHAPE_CENTER.index_select(1, self.batch_scatter)  # (1, sumN*A, n_dim)

            copy_stream = th.cuda.Stream()
            copy_event = th.cuda.Event()
            compute_event = th.cuda.Event()
            compute_event.record(th.cuda.default_stream(self.device))  # the default stream is the compute (main) stream.
            dump_queue = queue.Queue()
            dump_thread = threading.Thread(target=self._do_async_dump, args=(dump_queue,))
            logout_queue = queue.Queue()
            logout_thread = threading.Thread(target=self._do_async_print, args=(logout_queue,))
            try:
                dump_thread.start()
                logout_thread.start()
                fl = th.compile(self._main_for_loop_cuda, **self.compile_kwargs, disable=(not self.is_compile))
                fl(
                    compute_event,
                    _buf_E,
                    _buf_dE,
                    _buf_X,
                    _print_E,
                    _print_dE,
                    _print_X,
                    energies,
                    delta_E,
                    X,
                    copy_stream,
                    copy_event,
                    dump_queue,
                    logout_queue,
                    func, func_args, func_kwargs,
                    batch_indices, batch_slice_indx,
                    is_fix_mass_center,
                    move_to_center_freq,
                    SHAPE_CENTER
                )

                if not self._HOLD_DUMPER:
                    self.dumper.close()
                th.cuda.synchronize()
            finally:
                dump_queue.put([None]*4)
                dump_thread.join()
                logout_queue.put([None]*6)
                logout_thread.join()

        if self.verbose > 0:
            self.logger.info(
                '-' * 100 + f'\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s\n'
            )
            if self.verbose < 2:  # verbose = 1, brief mode only output last step coords.
                # split batches if specified batch
                if batch_indices is not None:
                    X_np = X.numpy(force=True)
                    X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                else:
                    X_tup = (X.numpy(force=True),)
                self.logger.info(f" Final Coordinates:\n")
                X_str = [
                    np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    for xi in X_tup
                ]
                [self.logger.info(f'{x_str}\n') for x_str in X_str]
        # output
        #return energies, X  #, ptlist  # test <<<

    def _main_for_loop_cuda(
            self,
            compute_event,
            _buf_E,
            _buf_dE,
            _buf_X,
            _print_E,
            _print_dE,
            _print_X,
            energies,
            delta_E,
            X,
            copy_stream,
            copy_event,
            dump_queue,
            logout_queue,
            func, func_args, func_kwargs,
            batch_indices, batch_slice_indx,
            is_fix_mass_center,
            move_to_center_freq,
            SHAPE_CENTER
    ):
        #t_step = time.perf_counter()
        _do_print = False
        for numit in range(self.maxiter):
            # dump
            if numit % self.output_structures_per_step == 0:
                _do_print = True
                compute_event.wait(th.cuda.default_stream(self.device))
                # D2D, fast copy purely on GPU
                _buf_E.copy_(energies.squeeze().contiguous())
                _buf_dE.copy_(delta_E)
                _buf_X.copy_(X)
                # D2H, async.
                with th.cuda.stream(copy_stream):
                    copy_stream.wait_stream(th.cuda.default_stream(self.device))
                    _print_E.copy_(_buf_E, non_blocking=True)
                    _print_dE.copy_(_buf_dE, non_blocking=True)
                    _print_X.copy_(_buf_X, non_blocking=True)  # D2H
                copy_event.record(copy_stream)
                # use backend thread to dump
                dump_queue.put((self.dumper, copy_event, _print_E, _print_X))

            # update X & energy. update before printing to cover the dumping cost by update.
            X_old = X  # .clone() # note that X is not in-place updated in `self._update_X` due to `where` ops. Hence `clone` is not necessary.
            energies_old = energies.clone()
            energies, delta_E, X = self._update_X(func, func_args, func_kwargs, energies_old, X)  # (n_batch, n_atom, n_dim)
            X_diff = X - X_old
            compute_event.record(th.cuda.default_stream(self.device))  # the default stream is the compute (main) stream.

            if _do_print:
                logout_queue.put((numit, _print_E, _print_dE, batch_indices, X, batch_slice_indx))
                _do_print = False

            # update algo. parameters.
            self._update_algo_param(numit, X_diff)

            # move to center
            if is_fix_mass_center and (numit % move_to_center_freq == 0):
                X.add_(SHAPE_CENTER - self.calc_shape_center(X))  # (n_batch, n_atom, n_dim) - (n_batch, 1, n_dim)

            # Check NaN
            # if th.any(energies != energies): raise RuntimeError(f'NaN Occurred in output: {energies}')

            # ptlist.append(X[:, None, :, 0].numpy(force=True))  # test <<<

    def __run_on_cpu(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]] | None = None,
            Cell_vector: th.Tensor | None = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            move_to_center_freq: int = -1
    ) -> None:
        """
        Run MC Algorithm.

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            Cell_vector: Tensor[n_batch, 3, 3], the cell vectors. Only for logging out information, not involves in calculation.
            Element_list: The list of elements/atom types. Only for logging out information, not involves in calculation.
                if None, all atomic number of points will be labeled to 0.
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]

        Return:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
            grad of argmin func: Tensor(X.shape), only output when `output_grad` == True. The gradient of X corresponding to minimum.
        """
        t_main = time.perf_counter()
        if func_kwargs is None:
            func_kwargs = dict()
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            if len(X.shape) == 2:
                X = X.unsqueeze(0)
            elif len(X.shape) != 3:
                raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
            if not isinstance(move_to_center_freq, int):
                raise TypeError(f'`move_to_center_freq` must be an integer, but got {type(move_to_center_freq)}.')
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (Tuple, List)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be Sequence[int] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            n_true_batch = len(batch_indices)
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )
        else:
            n_true_batch = n_batch
            self.batch_scatter = None
            self.batch_tensor = None
            batch_slice_indx = None
        # initialize vars
        maxiter = self.maxiter
        n_batch, n_atom, n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        if Element_list is not None:
            atomic_numbers = list()
            for _Elem in Element_list:
                atomic_numbers.append([ATOMIC_SYMBOL[__elem] if isinstance(__elem, str) else int(__elem) for __elem in _Elem])
        else:
            atomic_numbers = [[0]*self.n_atom]*self.n_batch
        if not isinstance(move_to_center_freq, int):
            raise TypeError(f'`move_to_center_freq` must be an integer, but got {type(move_to_center_freq)}.')
        is_fix_mass_center = (move_to_center_freq > 0)

        # Selective dynamics
        if fixed_atom_tensor is None:
            self.atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            self.atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'The shape of fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        self.fixed_indices = th.where(th.any(self.atom_masks, dim=-1))[1]  # COO sumN or n_atom dim
        # other check
        if (not isinstance(maxiter, int)) or (maxiter <= 0):
            raise ValueError(f'Invalid value of maxiter: {maxiter}. It would be an integer greater than 0.')

        # set variables device
        func = preload_func(func, self.device)
        X = X.to(self.device)

        # calc. freedom degree
        if batch_indices is None:
            _free_degree = X.shape[1] * n_dim
            if is_fix_mass_center:
                _free_degree -= 3
            self.free_degree = th.full((n_batch,), _free_degree, dtype=th.int64, device=self.device)
            n_reduce = th.where(th.abs(self.atom_masks) < 1e-6, 1, 0).sum(dim=(-2, -1))  # (n_batch, )
            self.free_degree -= n_reduce
        else:
            self.free_degree = self.batch_tensor * n_dim  # (n_batch, )
            if is_fix_mass_center:
                self.free_degree -= 3
            n_reduce_tensor = th.where(th.abs(self.atom_masks) < 1e-6, 1, 0).sum(dim=-1)
            n_reduce = index_reduce(n_reduce_tensor, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size).squeeze(0)  # (n_batch, )
            self.free_degree -= n_reduce

        # initialize the dumper
        X_arr = X.numpy(force=True)
        _x_dtype = X_arr.dtype.str
        atom_masks_arr = self.atom_masks.numpy(force=True).astype(_x_dtype)
        _num_dump = math.ceil(self.maxiter / self.output_structures_per_step)
        # write head information.
        if Cell_vector is None:
            Cell_vector = np.zeros((n_true_batch, 3, 3), dtype=np.float32)
        elif isinstance(Cell_vector, th.Tensor):
            Cell_vector = Cell_vector.numpy(force=True)
        elif not isinstance(Cell_vector, np.ndarray):
            Cell_vector = np.asarray(Cell_vector)
        if self.batch_tensor is not None:
            self.dumper.start_from_arrays(
                1,
                self.batch_tensor.numpy(force=True),  # batch indices
                Cell_vector,  # cell
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            self.dumper.step(
                self.batch_tensor.numpy(force=True),
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )
        else:
            self.dumper.start_from_arrays(
                1,
                Cell_vector,
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            self.dumper.step(
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )

        # initialize
        t_step = time.perf_counter()
        self.initialize_algo_param()
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
            self.logger.info(f'Iteration Scheme: {self.iterform}')
        self.logger.info('-' * 100)
        # MAIN LOOP
        with th.no_grad():
            energies: th.Tensor = func(X, *func_args, **func_kwargs)
            delta_E = th.zeros_like(energies)
            # continue to write main data
            self.dumper.start_from_arrays(
                _num_dump,
                energies.numpy(),
                X.numpy(),
            )
            # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
            if energies.shape[0] != self.n_batch:
                if batch_indices is None:
                    raise ValueError(
                        f"batch indices is None "
                        f"while shape of model output ({energies.shape}) does not match batch size ({self.n_batch})."
                    )
                if energies.shape[0] != n_true_batch:
                    raise ValueError(f"shape of output ({energies.shape}) does not match given batch indices")
            self.is_concat_X = (batch_indices is not None)
            self.is_accept = th.full_like(energies, False, dtype=th.bool)

            # preload a graph of shape center. A very small overhead that no longer `is_fix_center`, computation does.
            if batch_indices is None:
                SHAPE_CENTER = th.mean(X, dim=1, keepdim=True)  # (n_batch, 1, n_dim)
            else:
                SHAPE_CENTER = index_reduce(X, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size) / self.batch_tensor
                SHAPE_CENTER = SHAPE_CENTER.index_select(1, self.batch_scatter)  # (1, sumN*A, n_dim)

            fl = th.compile(self._main_for_loop_cpu, **self.compile_kwargs, disable=(not self.is_compile))
            fl(
                energies,
                delta_E,
                X,
                func, func_args, func_kwargs,
                batch_indices, batch_slice_indx,
                is_fix_mass_center,
                move_to_center_freq,
                SHAPE_CENTER
            )

            if not self._HOLD_DUMPER:
                self.dumper.close()

        if self.verbose > 0:
            self.logger.info(
                '-' * 100 + f'\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s\n'
            )
            if self.verbose < 2:  # verbose = 1, brief mode only output last step coords.
                # split batches if specified batch
                if batch_indices is not None:
                    X_np = X.numpy(force=True)
                    X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                else:
                    X_tup = (X.numpy(force=True),)
                self.logger.info(f" Final Coordinates:\n")
                X_str = [
                    np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    for xi in X_tup
                ]
                [self.logger.info(f'{x_str}\n') for x_str in X_str]

    def _main_for_loop_cpu(
            self,
            energies,
            delta_E,
            X,
            func, func_args, func_kwargs,
            batch_indices, batch_slice_indx,
            is_fix_mass_center,
            move_to_center_freq,
            SHAPE_CENTER
    ):
        _do_print = False
        for numit in range(self.maxiter):
            # dump
            if numit % self.output_structures_per_step == 0:
                _do_print = True
                self.dumper.step(
                    energies.numpy(),
                    X.numpy(),
                )
            # Print information / Verbose
            if _do_print:
                if self.verbose > 0:
                    self.logger.info(
                        f"ITERATION    {numit:>5d}\n "
                        f"delta E:     {np.array2string(delta_E.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Accepted:    {np.array2string(self.is_accept.numpy(force=True))}\n "
                        f"Energies:    {np.array2string(energies.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Temperature: {self.T_now:.3e}\n "
                        #f"TIME:        {time.perf_counter() - t_step:>6.4f} s"
                    )
                    #t_step = time.perf_counter()
                if self.verbose > 1:
                    # split batches if specified batch
                    if batch_indices is not None:
                        X_np = X.numpy(force=True)
                        X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                    else:
                        X_tup = (X.numpy(force=True),)
                    self.logger.info(f" Coordinates:\n")
                    X_str = [
                        np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        for xi in X_tup
                    ]
                    [self.logger.info(f'{x_str}\n') for x_str in X_str]

            # update X & energy
            X_old = X.clone()
            energies_old = energies.clone()
            energies, delta_E, X = self._update_X(func, func_args, func_kwargs, energies_old, X)  # (n_batch, n_atom, n_dim)
            X_diff = X - X_old

            # update algo. parameters.
            self._update_algo_param(numit, X_diff)

            # Check NaN
            # if not th.all(energies.isfinite()): raise RuntimeError(f'NaN Occurred in output: {energies}')

            # ptlist.append(X[:, None, :, 0].numpy(force=True))  # test <<<

            # move to center
            if is_fix_mass_center and (numit % move_to_center_freq == 0):
                X.add_(SHAPE_CENTER - self.calc_shape_center(X))  # (n_batch, n_atom, n_dim) - (n_batch, 1, n_dim)

    def initialize_algo_param(self):
        """
        Override this method to initialize attribute variables for self._update_direction.
        Examples:

        Returns: None
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
        """
        Override this method to implement X update algorithm.
        Args:
            func: function
            func_args:
            func_kwargs:
            X: (n_batch, n_atom, 3), the independent vars X.

        Returns:
           energies, delta_E, X
        """
        raise NotImplementedError

    def _update_algo_param(self, i: int, displace: th.Tensor) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.
        Args:
            i: iteration step now.
            displace: (n_batch, n_atom*3, 1), the displacement of X at this step. displace = step-length * p

        Returns: None
        """
        pass
