#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseOpt.py
#  Environment: Python 3.12
import logging
import sys
import queue
import threading
from itertools import accumulate
from typing import Dict, Any, Literal, Optional, Sequence, Tuple, List, Callable
import time
import warnings
from abc import ABC, abstractmethod
import os

import numpy as np
import torch as th
from torch import nn
from BUCToolkit.BatchOptim._utils._line_search import LineSearch
from BUCToolkit.BatchOptim._utils._warnings import NotConvergeWarning
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.Bases.BaseMotion import BaseMotion
from BUCToolkit.Bases.StdContainer import StdContainer
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT, STRING_ARRAY_FORMAT
from BUCToolkit.utils._Element_info import ATOMIC_NUMBER, ATOMIC_SYMBOL
from BUCToolkit.utils.index_ops import index_reduce, index_inner_product


class _BaseOpt(BaseMotion, ABC):

    #: Quantities known to the optimisation framework that may appear in
    #: ``dump_quantities`` / ``log_quantities``.
    ALLOWED_QUANTITIES: set = {'Energy', 'X', 'Force', 'E_diff', 'F_eps', 'X_grad'}

    def __init__(
            self,
            iter_scheme: str,
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            linesearch: Literal['Backtrack', 'B', 'Wolfe', 'W', 'MT', 'EXACT', 'None', 'N'] = 'Backtrack',
            linesearch_maxiter: int = 10,
            linesearch_thres: float = 0.02,
            linesearch_factor: float = 0.6,
            steplength: float = 0.5,
            use_bb: bool = True,
            output_file: str | None = None,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            _hold_samples: bool = False,
            dump_quantities: Tuple[str, ...] | List[str] = ('Energy', 'X', 'Force'),
            log_quantities: Tuple[str, ...] | List[str] = ('Energy', 'E_diff', 'F_eps', 'X', 'X_grad'),
    ) -> None:
        r"""
        A Base Framework of Algorithm for optimization.

        Args:
            E_threshold: float, threshold of difference of func between 2 iterations.
            F_threshold: float, threshold of gradient of func.
            maxiter: int, max iterations.
            linesearch: Scheme of linesearch.
                'Backtrack': Backtrack line search to satisfy Armijo's condition.
                'B': Alias for 'Backtrack'.
                'Wolfe': More-Thuente algorithm line search to satisfy Wolfe-Powell (weak Wolfe) condition.
                'W': Alias for 'Wolfe'.
                'MT': Alias for 'Wolfe'.
                'EXACT': Exact line search by Brent algorithm.
                'BRENT': Alias for 'BRENT'.
                'None': No line search. Directly return input steplength.
            linesearch_maxiter: Max iterations for linesearch.
            linesearch_thres: Threshold for linesearch. Only for "Golden" and "Newton".
            linesearch_factor: A factor in linesearch. Shrinkage factor for "Backtrack", scaling factor in interval search for "Golden" and line steplength for "Newton".
            steplength: The initial step length.
            use_bb: whether to use Barzilai-Borwein steplength (BB1 or long BB) as initial steplength instead of fixed one.
            output_file: the file to dump trajectory. if None, nothing will be dumped.
            device: The device that program runs on.
            verbose: Print level. ``0`` is silent, ``1`` prints selected
                scalars only, and ``2`` or greater also prints selected arrays.
            _hold_samples: ONLY FOR SPECIAL USES (e.g., CI-NEB or DEBUG).
                If True, optimizer will not remove any sample in a batch even if the sample has converged.
        
        Method:
            run: running the main optimization program.

        """
        warnings.filterwarnings('always', category=NotConvergeWarning)
        warnings.filterwarnings('always', )

        self.iterform = iter_scheme
        self.n_true_batch = None

        self.linesearch: str = linesearch
        self.steplength: float = steplength
        self.linesearch_maxiter = linesearch_maxiter
        self.linesearch_thres = linesearch_thres
        self.linesearch_factor = linesearch_factor
        self.use_bb = use_bb
        self._line_search = LineSearch(
            linesearch,
            maxiter=linesearch_maxiter,
            thres=linesearch_thres,
            factor=linesearch_factor,
            verbose=verbose,
        )

        self.E_threshold = E_threshold
        self.F_threshold = F_threshold
        self.maxiter = maxiter
        self.n_batch, self.n_atom, self.n_dim = None, None, None
        self.converge_mask = None  # To record the batch which has converged and not update.
        self.is_concat_X = False   # whether the output of `func` was concatenated.

        self._hold_samples = _hold_samples
        self.device = th.device(device)
        self.verbose = verbose

        # If True, `dumper.close()` is skipped in `run()`, allowing continuous
        # dumping across multiple `run()` calls (used by `StructureOptimization`).
        self._HOLD_DUMPER = False

        # Validated dump-header metadata retained across runs.
        self.cell_vec: np.ndarray | None = None
        self.atomic_numbers: List[List[int]] | None = None

        # logger
        super().__init__(output_file)
        self.init_logger('Main.OPT')
        self._setup_register_vars(dump_quantities, log_quantities)

    def _update_batch(self, mask: th.Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict):
        """
        Default update method for the input of func if the func has non-opt variables, i.e., the identical transform.
        Args:
            mask:

        Returns:

        """
        return func_args, func_kwargs, grad_func_args, grad_func_kwargs

    def set_system_info(
            self,
            cell_vec: th.Tensor | np.ndarray | Sequence | None = None,
            atomic_numbers: List[List[str | int]] | None = None,
    ) -> None:
        """Validate and register static metadata for optimizer trajectories.

        Element rows follow the same per-structure nested layout used by the
        motion APIs. Symbols are converted to atomic numbers immediately, so
        the retained metadata always has one canonical integer row per real
        structure. Compatibility with the current coordinates and optional
        irregular ``batch_indices`` is checked when :meth:`run` begins.

        Args:
            cell_vec: Optional numeric array-like object with shape
                ``[n_structure, 3, 3]``. Values must be finite.
            atomic_numbers: Optional nested element symbols or integer atomic
                numbers. The outer list indexes structures and each inner list
                contains the atoms of that structure.

        Returns:
            None. Non-``None`` inputs replace the corresponding values retained
            on the optimizer for subsequent :meth:`run` calls.

        Raises:
            TypeError: If a cell or element container has an unsupported type.
            ValueError: If shapes, symbols, atomic numbers, or structure counts
                are invalid.
        """
        normalized_cell = self.cell_vec
        if cell_vec is not None:
            if isinstance(cell_vec, th.Tensor):
                cell_vec = cell_vec.detach().cpu().to(th.float32).numpy()
            try:
                cell_array = np.array(cell_vec, dtype=np.float32, copy=True)
            except (TypeError, ValueError) as error:
                raise TypeError('`cell_vec` must contain numeric values.') from error
            if cell_array.ndim != 3 or cell_array.shape[1:] != (3, 3):
                raise ValueError(
                    '`cell_vec` must have shape [n_structure, 3, 3], but got '
                    f'{cell_array.shape}.'
                )
            if not np.all(np.isfinite(cell_array)):
                raise ValueError('`cell_vec` must contain only finite values.')
            normalized_cell = cell_array

        normalized_numbers = self.atomic_numbers
        if atomic_numbers is not None:
            if not isinstance(atomic_numbers, list) or any(
                    not isinstance(row, list) for row in atomic_numbers
            ):
                raise TypeError(
                    '`atomic_numbers` must be List[List[str]] or List[List[int]].'
                )
            if (len(atomic_numbers) < 1) or any(not row for row in atomic_numbers):
                raise ValueError('`atomic_numbers` must contain non-empty structure rows.')
            try:
                normalized_numbers = [
                    [
                        int(ATOMIC_SYMBOL.get(str(element), element))
                        for element in structure_elements
                    ]
                    for structure_elements in atomic_numbers
                ]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    'Atomic entries must be symbols or integers.'
                ) from error

            if any(
                    number not in ATOMIC_NUMBER
                    for structure_numbers in normalized_numbers
                    for number in structure_numbers
            ):
                raise ValueError(
                    '`atomic_numbers` contains an unsupported atomic number.'
                )

        if (
                normalized_cell is not None and
                normalized_numbers is not None and
                len(normalized_cell) != len(normalized_numbers)
        ):
            raise ValueError(
                '`cell_vec` and `atomic_numbers` describe different numbers of '
                f'structures: {len(normalized_cell)} and '
                f'{len(normalized_numbers)}.'
            )

        self.cell_vec = normalized_cell
        self.atomic_numbers = normalized_numbers

    def _do_async_dump(self, q: queue.Queue):
        """Consumer thread: drain queue and call ``dumper.step()``.

        ``self._dump_done`` is set in ``finally`` to guarantee the
        handshake even on exception.

        Args:
            q: Single-slot queue. Normal items contain
                ``(dumper, sync_event, *cpu_tensors)`` and ``None`` terminates
                the thread. ``sync_event`` is a CUDA event or ``None`` on CPU;
                tensors follow :meth:`get_dump_vars` order.

        Returns:
            None. Write failures are reported through ``self.logger`` and the
            completion event is still set so the optimizer cannot deadlock.
        """
        while True:
            items = q.get()
            if items is None:
                break
            try:
                dumper, sync, *rest = items
                if sync is not None:
                    sync.synchronize()
                    rest = [t.numpy() for t in rest]
                dumper.step(*rest)
            except Exception as e:
                self.logger.error(f"Error: Failed to dump data due to \"{e}\"")
            finally:
                self._dump_done.set()

    def _do_async_print(self, q: queue.Queue):
        """Consumer thread: format and log.

        Queue items: ``(event, numit, converge_str, t_st, *log_vals)``
        where *log_vals* are packed in :meth:`get_log_vars` order.
        Scalars (ndim ≤ 1) are printed under an ``ITERATION`` header;
        arrays (ndim ≥ 2) go through ``handle_arrays_print``.
        ``self._print_done`` is set in ``finally``.

        Args:
            q: Single-slot queue. Normal items have the form
                ``(sync_event, iteration, converge_flags, step_start_time,
                *cpu_tensors)``; tensors follow :meth:`get_log_vars` order and
                ``None`` terminates the thread.

        Returns:
            None. Formatting failures are logged and always release the shared
            snapshot buffer through ``self._print_done``.
        """
        numit = 0
        while True:
            items = q.get()
            if items is None:
                break
            try:
                sync, numit, converge_str, t_st = items[0], items[1], items[2], items[3]
                if sync is not None:
                    sync.synchronize()
                # Map log-var name → value (items[4:] in log_names order)
                _data = dict(zip(self.get_log_vars(), items[4:]))
                _valid = [(_n, _v) for _n, _v in _data.items() if _n and _v is not None]
                _scalars = [(_n, _v) for _n, _v in _valid if _v.ndim <= 1]
                _arrays  = [(_n, _v) for _n, _v in _valid if _v.ndim >= 2]

                if _scalars and self.verbose > 0:
                    _DISP = {'Energy': 'Energies', 'E_diff': 'MAD_energies', 'F_eps': 'MAX_F'}
                    _lines = [f'ITERATION {numit:>5d}']
                    for _n, _v in _scalars:
                        _label = _DISP.get(_n, _n)
                        _lines.append(f' {_label:<14s}: {np.array2string(_v.numpy(), **SCIENTIFIC_ARRAY_FORMAT)}')
                    _lines.append(f' Converged:    {np.array2string(converge_str, **STRING_ARRAY_FORMAT)}')
                    _lines.append(f' TIME:         {time.perf_counter() - t_st:>6.4f} s')
                    self.logger.info('\n'.join(_lines))

                if _arrays and self.verbose > 1:
                    # Map internal names to display labels (X -> Coordinates,
                    # X_grad -> Forces with sign flip)
                    _ARR_DISP = {'X': 'Coordinates', 'X_grad': 'Forces'}
                    _arr_names = [_ARR_DISP.get(_n, _n) for _n, _ in _arrays]
                    _arr_vals  = [(-_v if _n == 'X_grad' else _v) for _n, _v in _arrays]
                    self.handle_arrays_print(
                        self.logger, self.batch_indices, self.batch_slice_indx,
                        [_arr_vals], [_arr_names],
                        verbose=self.verbose,
                    )
            except Exception as e:
                self.logger.error(f"Error: Failed to logout at {numit}-th iteration due to \"{e}\".")
            finally:
                self._print_done.set()

    def set_batch_updater(
            self,
            method: Callable[[th.Tensor, Tuple|None, Dict|None, Tuple|None, Dict|None], Tuple[Tuple, Dict, Tuple, Dict]] | None,
            line_search_method: Callable[[th.Tensor, Tuple|None, Dict|None, Tuple|None, Dict|None], Tuple[Tuple, Dict, Tuple, Dict]] | None = None,
    ) -> None:
        """
        Set a method to update the taget function when variables change.
        If input Callables, these Callables receive a mask tensor of shape (n_batch, )
        that only selects the `True` part to input to the function, the old
        `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`,
        returns the corresponding masked new `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`.
        If input None, self._hold_samples will be set to True that toggles off the dynamic removal.

        This method is used to dynamically 'remove' the samples which have been converged in a batch to avoid
        redundant calculation of converged samples.

        Default transform is identical transform (i.e., do nothing)

        `method` is for main loop update; and `line_search_method` is for line search subroutine update;
        Args:
            method: Callable(
                    mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict
                ) -> Tuple[Tuple, Dict, Tuple, Dict], the method of updating function arguments for a mask.
            line_search_method: as the same use of `method`, but for line search subroutines.

        Returns: None
        """
        if method is None:
            self._hold_samples = True
        elif callable(method):
            self._update_batch = method
        else:
            raise TypeError(f'`method` must be a callable, but {type(method)} is not.')
        self._line_search.set_batch_updater(line_search_method)

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            grad_func: Any | nn.Module = None,
            func_args: Tuple[Any] = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Tuple[Any] = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor] | Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Run the Optimization Algorithm.

        Call :meth:`set_system_info` before this method to include validated
        cell vectors and element metadata in the dump header group.

        Args:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X
                i.e., grad = grad_func(X, y, *grad_func_args, **grad_func_kwargs), else grad = grad_func(X, *grad_func_args, **grad_func_kwargs)
            require_grad: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs) calculation.
            output_grad: bool, whether output gradient of last step.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is the same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]

        Returns:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
            grad of argmin func: Tensor(X.shape), only output when `output_grad` == True. The gradient of X corresponding to minimum.
        """
        t_main = time.perf_counter()
        if func_kwargs is None:
            func_kwargs = dict()
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
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
                self.batch_tensor = batch_indices
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (List, Tuple)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            self.batch_indices = batch_indices
            self.batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)  # the tensor version of batch_indices which is a List.
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )  # scatter mask of the int tensor with the same shape as X.shape[1], which the data in one batch have one index.
            n_true_batch = len(batch_indices)   # the true batch size for irregular batches
            # steplength
            steplength_tensor = th.full(
                (1, n_true_batch, 1), fill_value=self.steplength, device=self.device, dtype=th.float32
            )  # (n_batch, sumN, 1), initial step length
            batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        else:
            n_true_batch = n_batch
            # steplength
            steplength_tensor = th.full(
                (n_batch, 1, 1), fill_value=self.steplength, device=self.device, dtype=th.float32
            )  # (n_batch, sumN, 1), initial step length
            batch_tensor_indx_cache = None
            self.batch_indices = None
            self.batch_slice_indx = None
            self.batch_tensor = None
            self.batch_scatter = None

        # initialize vars
        self.n_true_batch = n_true_batch
        maxiter = self.maxiter
        n_batch, n_atom, n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim

        if self.cell_vec is not None and len(self.cell_vec) != n_true_batch:
            raise ValueError(
                f'`cell_vec` contains metadata for {len(self.cell_vec)} structures, but this run contains {n_true_batch}.'
            )
        if self.atomic_numbers is not None:
            expected_atom_counts = (
                list(batch_indices)
                if batch_indices is not None
                else [n_atom] * n_true_batch
            )
            actual_atom_counts = [len(row) for row in self.atomic_numbers]
            if (
                    len(self.atomic_numbers) != n_true_batch or
                    actual_atom_counts != expected_atom_counts
            ):
                raise ValueError(
                    '`atomic_numbers` must match the run layout: expected '
                    f'{expected_atom_counts}, got {actual_atom_counts}.'
                )

        p = th.zeros_like(X)  # like X, the previous direction
        self.converge_mask = None  # (n_true_batch, )
        X_grad_old = th.full_like(X, 1e-20, dtype=th.float32, device=self.device)  # like X, initial old grad.
        displace = th.full_like(X_grad_old, 0.)  # like X, the X displacement
        # grad_func
        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(
            grad_func,
            is_grad_func_contain_y,
            require_grad,
        )
        self._line_search.require_grad = require_grad  # set linear search
        # Selective dyamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)  # has the same shape as X
        # other check
        if (not isinstance(maxiter, int)) or (maxiter <= 0):
            raise ValueError(f'Invalid value of maxiter: {maxiter}. It would be an integer greater than 0.')

        # set variables device
        func = preload_func(func, self.device)

        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.detach()
        X = X.to(self.device)

        # initialize
        ############################## BATCHED ALGORITHM ###################################
        # variables with '_' refer to the dynamically changed variables during iteration,
        # and they will in-placed copy into origin variables (i.e., without '_') at the end
        # of each iteration to update data.
        #
        ####################################################################################
        is_main_loop_converge = False
        t_st = time.perf_counter()
        # Section: initialize
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
            self.logger.info(f'Iteration Scheme: {self.iterform}')
            self.logger.info('-' * 100)

        with th.no_grad():
            with th.set_grad_enabled(require_grad):
                X.requires_grad_(require_grad)
                energies: th.Tensor = func(X, *func_args, **func_kwargs)
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
                # calc. grad
                if is_grad_func_contain_y:
                    X_grad = grad_func_(X, energies, *grad_func_args, **grad_func_kwargs)
                else:
                    X_grad = grad_func_(X, *grad_func_args, **grad_func_kwargs)
                if X_grad.shape != X.shape:
                    raise RuntimeError(f'X_grad ({X_grad.shape}) and X ({X.shape}) have different shapes.')
            energies = energies.detach()
            energies_old = th.full_like(energies, th.inf)
            X_grad = X_grad.detach()
            X_grad.mul_(atom_masks)
            X = X.detach()
            # Section: initialize custom algorithm state.
            self.initialize_algo_param()

            # ================================================================
            # Section: async dump infrastructure
            # ================================================================
            dumper = self.dumper
            dump_names = self.get_dump_vars()
            log_names  = self.get_log_vars()
            total_names = self.get_transfer_vars()
            _is_cuda = (self.device.type == 'cuda')

            # --- state container (all transfer vars, lazily filled) ---
            s = StdContainer(Energy=energies, X=X, Force=-X_grad)
            # Add log-only vars that aren't yet on s
            for _n in total_names:
                if not hasattr(s, _n):
                    setattr(s, _n, th.empty_like(energies) if _n in ('E_diff', 'F_eps') else th.empty_like(X))

            # --- header group (static metadata, 1 cycle) ---
            if self.cell_vec is None:
                _cell_np = np.zeros((n_true_batch, 3, 3), dtype=np.float32)
            else:
                _cell_np = self.cell_vec

            # --- resolve atomic_numbers to numpy ---
            #   Regular batch  -> 2-D (n_batch, n_atom)
            #   Irregular batch -> 1-D (\sum n_i,)
            _atm = self.atomic_numbers
            if _atm is None:
                # No atomic numbers provided → zeros placeholder
                if self.is_concat_X:
                    _atm_np = np.zeros(X.shape[1], dtype=np.int64)
                else:
                    _atm_np = np.zeros((n_batch, n_atom), dtype=np.int64)
            else:
                _parts = [np.asarray(row, dtype=np.int64) for row in _atm]
                _atm_np = np.concatenate(_parts) if self.is_concat_X else np.stack(_parts)

            _fix_np = atom_masks.numpy(force=True)

            if self.batch_tensor is not None:
                _batch_np = self.batch_tensor.numpy(force=True)
                dumper.start_from_arrays(
                    1, _batch_np, _cell_np, _atm_np, _fix_np,
                    names=('batch_indices', 'cell_vec', 'atomic_numbers', 'fixed_mask'),
                )
                dumper.step(_batch_np, _cell_np, _atm_np, _fix_np)
            else:
                dumper.start_from_arrays(
                    1, _cell_np, _atm_np, _fix_np,
                    names=('cell_vec', 'atomic_numbers', 'fixed_mask'),
                )
                dumper.step(_cell_np, _atm_np, _fix_np)

            # --- data group (trajectory, dynamic steps) ---
            # Allocate CPU (pinned) + optional GPU staging buffers for every
            # name in total_names (union of dump + log).  The same s_cpu is
            # shared by both consumers — each picks the attrs it needs.
            _dump_queue: queue.Queue = queue.Queue(maxsize=1)
            _log_queue: queue.Queue = queue.Queue(maxsize=1)
            _s_cpu, _s_buf = self._allocate_cpu_buffers(s, total_names, self.device, require_buffer=self._hold_samples)
            if _is_cuda:
                _copy_stream = th.cuda.Stream()
                _copy_event = th.cuda.Event()
            else:
                _copy_stream = None  # CPU path: no async D2H
                _copy_event = None
            # Prototype arrays for the dumper (shapes / dtypes from _s_cpu)
            _protos = tuple(getattr(_s_cpu, name).numpy() for name in dump_names)
            dumper.start_from_arrays(-1, *_protos, names=dump_names)

            # Launch async consumer threads (shared by CPU and CUDA paths)
            _dump_thread = threading.Thread(target=self._do_async_dump, args=(_dump_queue,), daemon=True)
            _log_thread = threading.Thread(target=self._do_async_print, args=(_log_queue,), daemon=True)
            # Handshake events — initially set so iteration 0 passes without blocking
            self._dump_done = threading.Event()
            self._print_done = threading.Event()
            self._dump_done.set()
            self._print_done.set()

            try:
                _dump_thread.start()
                _log_thread.start()

                # cache for dynamically changed batch indices due to convergence, avoiding reallocate mem.
                # MAIN LOOP
                for numit in range(maxiter):
                    # Calc. Criteria
                    E_diff = energies - energies_old
                    E_eps = th.abs(E_diff)  # (n_batch, )
                    energies_old.copy_(energies)
                    # manage the irregular tensors
                    if self.is_concat_X:
                        # (1, n_batch*n_atom, 3)
                        F_eps = index_reduce(
                            th.max(th.abs(X_grad[0]), dim=-1).values, self.batch_scatter, 0, 'amax', -1.
                        )
                        f_converge = F_eps < self.F_threshold
                        converge_mask = (E_eps < self.E_threshold) & f_converge  # (n_true_batch, ), to stop the update of converged samples.
                        converge_check = converge_mask
                        self.converge_mask = converge_check
                        converge_str = converge_check.numpy(force=True)
                        converge_mask = converge_mask.reshape(1, -1, 1)[:, self.batch_scatter, ...]  # (1, n_batch*n_atom, 3)
                    else:
                        F_eps = th.amax(th.abs(X_grad), dim=(-2, -1))  # (n_batch, n_atom, 3) -> (n_batch)
                        f_converge = (F_eps < self.F_threshold).reshape(-1, 1, 1)
                        converge_mask = (E_eps < self.E_threshold).reshape(-1, 1, 1) & f_converge  # To stop the update of converged samples.
                        converge_check = converge_mask[:, 0, 0]
                        self.converge_mask = converge_check
                        converge_str = (converge_mask[:, 0, 0]).numpy(force=True)

                    # --- dump + print snapshot ---
                    # Update live state container with current iteration values
                    s.Energy = energies; s.X = X; s.Force = -X_grad
                    s.E_diff = E_diff; s.F_eps = F_eps; s.X_grad = X_grad
                    self._dump_done.clear()
                    if _is_cuda:
                        # GPU: D2D staging → async D2H on copy stream
                        self._transfer_buffers_D2H(s, _s_buf, _s_cpu, total_names, _copy_stream, self.device)
                        _copy_event.record(_copy_stream)
                        _dump_queue.put((dumper, _copy_event, *(getattr(_s_cpu, n) for n in dump_names)))
                        if self.verbose > 0:
                            self._print_done.clear()
                            _log_queue.put((_copy_event, numit, converge_str, t_st,
                                            *(getattr(_s_cpu, n) for n in log_names)))
                    else:
                        # CPU: synchronous snapshot s → _s_cpu, then dispatch
                        for _n in total_names:
                            getattr(_s_cpu, _n).copy_(getattr(s, _n))
                        _dump_queue.put((dumper, None, *(getattr(_s_cpu, n) for n in dump_names)))
                        if self.verbose > 0:
                            self._print_done.clear()
                            _log_queue.put((None, numit, converge_str, t_st,
                                            *(getattr(_s_cpu, n) for n in log_names)))
                    # judge thres
                    if th.all(self.converge_mask):
                        is_main_loop_converge = True
                        break

                    #g: th.Tensor = th.flatten(X_grad, 1, 2).unsqueeze(-1).contiguous()  # (n_batch, n_atom*3, 1)
                    # Section: update batch
                    if not self._hold_samples:
                        func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                            ~converge_check,
                            func_args,
                            func_kwargs,
                            grad_func_args,
                            grad_func_kwargs
                        )
                        if self.is_concat_X:
                            select_mask = ~(converge_mask[0, :, 0])
                            select_mask_short = ~converge_check
                            energies_ = energies[select_mask_short]
                            X_grad_ = X_grad[:, select_mask, :]
                            X_grad_old_ = X_grad_old[:, select_mask, :]
                            p_ = p[:, select_mask, :]
                            X_ = X[:, select_mask, :]
                            displace_ = displace[:, select_mask, :]
                            atom_masks_ = atom_masks[:, select_mask, :]
                            steplength_ = steplength_tensor[:, select_mask_short, :]
                            batch_tensor_ = self.batch_tensor[select_mask_short]
                            batch_scatter_ = th.repeat_interleave(
                                batch_tensor_indx_cache[:len(batch_tensor_)],
                                batch_tensor_,
                                dim=0
                            )
                        else:
                            select_mask = ~converge_check
                            select_mask_short = select_mask
                            energies_ = energies[select_mask]
                            X_grad_ = X_grad[select_mask, ...]
                            X_grad_old_ = X_grad_old[select_mask, ...]
                            p_ = p[select_mask, ...]
                            X_ = X[select_mask, ...]
                            atom_masks_ = atom_masks[select_mask, ...]
                            displace_ = displace[select_mask, ...]
                            steplength_ = steplength_tensor[select_mask, ...]
                            batch_tensor_ = None
                            batch_scatter_ = None
                    else:
                        func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = (
                            func_args,
                            func_kwargs,
                            grad_func_args,
                            grad_func_kwargs
                        )
                        select_mask = ~(converge_mask[0, :, 0])
                        select_mask_short = ~converge_check
                        energies_ = energies
                        X_grad_ = X_grad
                        X_grad_old_ = X_grad_old
                        p_ = p
                        X_ = X
                        displace_ = displace
                        atom_masks_ = atom_masks
                        steplength_ = steplength_tensor
                        batch_tensor_ = self.batch_tensor
                        batch_scatter_ = self.batch_scatter

                    # Section: update algo. parameters.
                    self._update_algo_param(
                        select_mask,
                        select_mask_short,
                        batch_scatter_,
                        X_grad_,
                        X_grad_old_,
                        p_,
                        displace_
                    )
                    self.select_mask = select_mask

                    t_st = time.perf_counter()
                    # Section: search directions
                    p_ = self._update_direction(
                        X_grad_,
                        X_grad_old_,
                        p_,
                        X_,
                        batch_scatter_,
                    )  # (n_batch, n_atom, n_dim)
                    # use BB steplength_tensor
                    if self.use_bb:
                        g_go = X_grad_ - X_grad_old_  # (n_batch, n_atom, n_dim)
                        if self.is_concat_X:
                            _steplength_ = th.sum(index_inner_product(
                                displace_,
                                displace_,
                                dim=1,
                                batch_indices=batch_scatter_
                            ), dim=-1, keepdim=True) / th.sum(index_inner_product(
                                displace_,
                                g_go,
                                dim=1,
                                batch_indices=batch_scatter_
                            ), dim=-1, keepdim=True)  # BB1, (1, B, 1)
                            _steplength_ = th.where(
                                (_steplength_ < 2. * self.steplength) & (_steplength_ > 1e-4),
                                _steplength_,
                                steplength_
                            )
                        else:
                            # (n_batch, 1, n_atom*n_dim) @ (n_batch, n_atom*n_dim, 1) =
                            _steplength_ = th.sum(
                                displace_ * displace_, dim=(-2, -1), keepdim=True
                            ) / th.sum(
                                displace_ * g_go, dim=(-2, -1), keepdim=True
                            )  # BB1
                            _steplength_ = th.where(
                                (_steplength_ < 2. * self.steplength) * (_steplength_ > 1e-4),
                                _steplength_,
                                steplength_
                            )
                    else:
                        _steplength_ = steplength_
                    # Section: search step length -> steplength_tensor: (n_batch, 1, 1)
                    steplength_: th.Tensor = self._line_search.run(
                        func,
                        grad_func_,
                        X_,
                        energies_,
                        X_grad_,
                        p_,
                        _steplength_,
                        is_grad_func_contain_y,
                        require_grad,
                        func_args=func_args_,
                        func_kwargs=func_kwargs_,
                        grad_func_args=grad_func_args_,
                        grad_func_kwargs=grad_func_kwargs_,
                        batch_indices=batch_tensor_
                    )
                    # update X
                    if self.is_concat_X:
                        alpha = steplength_.index_select(1, batch_scatter_)
                    else:
                        alpha = steplength_
                    displace_ = alpha * p_  # (n_batch, 1, 1) * (n_batch, n_atom, n_dim) or (1, sumN, 1) * (1, sumN, n_dim)

                    if self._hold_samples and not _is_cuda:
                        # gate: in this mode `X_` aliases `X`, whose zero-copy views
                        # were queued to the consumers above; `_update_direction` /
                        # `X_.add_` below mutate it in place, so consumers must
                        # finish reading first. (CUDA path is protected by s_buf
                        # D2D staging — stream ordering after wait_event suffices.)
                        self._dump_done.wait()
                        self._print_done.wait()

                    # MAIN UPDATE
                    X_.add_(displace_)  # (n_batch, n_atom, 3) + (n_batch, n_atom, 3)
                    # update old grad
                    X_grad_old_ = X_grad_  # (n_batch, n_atom, n_dim)
                    # calc. new energy & grad.
                    if not self._line_search.HAS_GRAD:
                        energies_, X_grad_ = self._calc_y_grad(
                            X_,
                            func,
                            func_args_,
                            func_kwargs_,
                            grad_func_,
                            grad_func_args_,
                            grad_func_kwargs_,
                            require_grad,
                            is_grad_func_contain_y
                        )
                    else:
                        energies_ = self._line_search.STORE_Y
                        X_grad_ = self._line_search.STORE_GRAD
                    energies_ = energies_.detach()
                    X_grad_ = X_grad_.detach()
                    X_grad_.mul_(atom_masks_)
                    X_.detach_()

                    # Section: rewrite. update origin variables
                    # --- gate: wait for consumers from previous iteration ---
                    if _is_cuda: th.cuda.default_stream(self.device).wait_event(_copy_event)
                    self._dump_done.wait()
                    self._print_done.wait()
                    if not self._hold_samples:
                        if self.is_concat_X:
                            select_indices = th.where(select_mask)[0]
                            select_indices_short = th.where(select_mask_short)[0]
                            energies.index_copy_(0, select_indices_short, energies_)
                            X_grad.index_copy_(1, select_indices, X_grad_)
                            X_grad_old.index_copy_(1, select_indices, X_grad_old_)
                            p.index_copy_(1, select_indices, p_)
                            X.index_copy_(1, select_indices, X_)
                            displace.index_copy_(1, select_indices, displace_)
                            #atom_masks.index_copy_(1, select_indices, atom_masks_)
                            #steplength_tensor.index_copy_(1, select_indices_short, steplength_)

                        else:
                            select_indices = th.where(select_mask)[0]
                            select_indices_short = th.where(select_mask_short)[0]
                            energies.index_copy_(0, select_indices, energies_)
                            X_grad.index_copy_(0, select_indices, X_grad_)
                            X_grad_old.index_copy_(0, select_indices, X_grad_old_)
                            p.index_copy_(0, select_indices, p_)
                            X.index_copy_(0, select_indices, X_)
                            displace.index_copy_(0, select_indices, displace_)
                            #atom_masks.index_copy_(0, select_indices, atom_masks_)
                            #steplength_tensor.index_copy_(0, select_indices, steplength_)
                    else:
                        select_indices = th.where(select_mask)[0]
                        select_indices_short = th.where(select_mask_short)[0]
                        energies = energies_
                        X_grad = X_grad_
                        X_grad_old = X_grad_old_
                        p = p_
                        X = X_
                        displace = displace_
                        #steplength_tensor = steplength_
                    # Section: update batch information of algos if necessary
                    self._update_algo_batches(select_indices, select_indices_short)
                    # Check NaN
                    #if not th.all(energies.isfinite()): raise RuntimeError(f'NaN Occurred in output: {energies}')

                    #ptlist.append(X[:, None, :, 0].numpy(force=True))  # test <<<

                # --- wait for last dump consumer before scatter-back touched tensors ---
                self._dump_done.wait()
                self._print_done.wait()

                # Final human-readable output is independent of the configured
                # dump/log columns. Copy directly from the live result so
                # omitting ``X`` or ``Force`` from ``dump_quantities`` does not
                # make final reporting access a missing ``s_cpu`` attribute.
                _X_print = X.detach().cpu()
                _F_print = (-X_grad).detach().cpu()

                if is_main_loop_converge:
                    if self.verbose > 0:
                        self.logger.info(
                            '-' * 100 + f'\nAll Structures were Converged.\n'
                                        f'MAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s\n'
                        )
                else:
                    if self.verbose > 0:
                        self.logger.info(
                            '-' * 100 + f'\nSome Structures were NOT Converged yet!\n'
                                        f'MAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s\n'
                        )
                    # final dump: capture last state when maxiter exhausted
                    self._dump_done.clear()
                    # Refresh every built-in state quantity before taking the
                    # final snapshot. During the main loop ``s`` describes the
                    # state at the beginning of an iteration; after the last
                    # update, local tensors contain the true max-iteration
                    # result. This assignment is required on both CPU and CUDA
                    # paths, especially for ``Energy``/``Force`` which may have
                    # been replaced rather than mutated in-place.
                    if self.is_concat_X:
                        _final_F_eps = index_reduce(
                            th.max(th.abs(X_grad[0]), dim=-1).values,
                            self.batch_scatter, 0, 'amax', -1.
                        )
                    else:
                        _final_F_eps = th.amax(th.abs(X_grad), dim=(-2, -1))
                    s.Energy = energies
                    s.X = X
                    s.Force = -X_grad
                    s.E_diff = energies - energies_old
                    s.F_eps = _final_F_eps
                    s.X_grad = X_grad
                    if _is_cuda:
                        self._transfer_buffers_D2H(s, _s_buf, _s_cpu, dump_names, _copy_stream, self.device)
                        _copy_event.record(_copy_stream)
                        _dump_queue.put((dumper, _copy_event, *(getattr(_s_cpu, n) for n in dump_names)))
                    else:
                        for _n in dump_names:
                            getattr(_s_cpu, _n).copy_(getattr(s, _n))
                        _dump_queue.put((dumper, None, *(getattr(_s_cpu, n) for n in dump_names)))
                    self._dump_done.wait()

                if not is_main_loop_converge and self.verbose > 0:
                    # The unconverged final scalar state has not been printed by
                    # the main-loop consumer yet.
                    E_diff = energies - energies_old
                    self.logger.info(
                        f"FINAL STEP\n "
                        f"MAD_energies: {np.array2string(E_diff.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Energies:     {np.array2string(energies.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Converged:    {np.array2string(converge_str, **STRING_ARRAY_FORMAT)}\n "
                    )
                if not is_main_loop_converge and self.verbose > 1:
                    self.handle_arrays_print(
                        self.logger,
                        batch_indices,
                        self.batch_slice_indx,
                        [[_X_print, _F_print]],
                        [['Final Coordinates', 'Final Forces']],
                        verbose=self.verbose,
                    )
            # release resources
            finally:
                # --- dump cleanup ---
                _dump_queue.put(None)  # sentinel
                _dump_thread.join()
                _log_queue.put(None)
                _log_thread.join()
                if not self._HOLD_DUMPER:
                    dumper.close()

        # output
        if output_grad:
            return energies, X, X_grad
        else:
            return energies, X  #, ptlist  # test <<<

    @abstractmethod
    def initialize_algo_param(self):
        """
        Override this method to initialize attribute variables for self._update_direction.
        Examples:
            (BFGS algo.)
            # descent direction
            self.p = 0.
            # Initial quasi-inverse Hessian Matrix  (n_batch, n_atom*n_dim, n_atom*n_dim)
            self.H_inv = (th.eye(n_atom * n_dim, device=self.device).unsqueeze(0)).expand(n_batch, -1, -1)
            # prepared identity matrix
            self.Ident = (th.eye(n_atom * n_dim, device=self.device).unsqueeze(0)).expand(n_batch, -1, -1)

        Returns: None
        """
        raise NotImplementedError

    @abstractmethod
    def _update_direction(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices: th.Tensor | None,
    ) -> th.Tensor:
        """
        Override this method to implement X update algorithm.
        Args:
            g: (n_batch, n_atom, n_dim), the gradient of X at this step
            g_old: (n_batch, n_atom, n_dim), the gradient of X at last step
            p: (n_batch, n_atom, n_dim), the update direction of X at last step
            X: (n_batch, n_atom, n_dim), the independent vars X.
            batch_scatter_indices: the batch indices. See `_update_algo_param`.

        Returns:
            p: th.Tensor, the new update direction of X.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_algo_param(
            self,
            select_mask: th.Tensor,
            select_mask_short: th.Tensor | None,
            batch_scatter_indices: th.Tensor | None,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            displace: th.Tensor
    ) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.
        Args:
            select_mask: (n_batch, ), the mask of batch that converged. Only the position of `True` would be selected to calculate.
            batch_scatter_indices: (sumN, ), the batch scatter indices of sample that not yet converged.
                format: [0, 0, ..., 0, 1, 1, ..., 1, ..., N-1], where the same number means the line in the same sample.
                If samples are in a regular batch, it would be set to None.
            g: (n_batch, n_atom, n_dim), the gradient of X at this step
            g_old: (n_batch, n_atom, n_dim), the gradient of X at last step
            p: (n_batch, n_atom, n_dim), the update direction of X at last step
            displace: (n_batch, n_atom, n_dim), the displacement of X at this step. displace = step-length * p

        Returns: None
        """
        raise NotImplementedError

    def _update_algo_batches(
            self,
            select_indices: th.Tensor,
            select_indices_short: th.Tensor | None,
    ):
        """
        Optional.
        Override this method to update the batch information of algorithm parameters, i.e., self.iterform.
        Returns:

        """
        pass
