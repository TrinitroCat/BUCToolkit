#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseOpt.py
#  Environment: Python 3.12
import logging
import sys
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
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT, STRING_ARRAY_FORMAT
from BUCToolkit.utils.index_ops import index_reduce, index_inner_product


class _BaseOpt(BaseMotion, ABC):
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
            verbose: amount of print information.
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
        self.device = device
        self.verbose = verbose

        # logger
        super().__init__(output_file)
        self.init_logger('Main.OPT')

    def _update_batch(self, mask: th.Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict):
        """
        Default update method for the input of func if the func has non-opt variables, i.e., the identical transform.
        Args:
            mask:

        Returns:

        """
        return func_args, func_kwargs, grad_func_args, grad_func_kwargs

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
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
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
            batch_slice_indx = None

        # initialize vars
        self.n_true_batch = n_true_batch
        maxiter = self.maxiter
        n_batch, n_atom, n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
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
        # MAIN LOOP
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
            # cache for dynamically changed batch indices due to convergence, avoiding reallocate mem.
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

                # Print information / Verbose
                if self.verbose > 0:
                    self.logger.info(f"ITERATION {numit:>5d}\n "
                                     f"MAD_energies: {np.array2string(E_diff.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"MAX_F:        {np.array2string(F_eps.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Energies:     {np.array2string(energies.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Converged:    {np.array2string(converge_str, **STRING_ARRAY_FORMAT)}\n "
                                     f"TIME:         {time.perf_counter() - t_st:>6.4f} s")
                if self.verbose > 1:
                    # split batches if specified batch
                    if batch_indices is not None:
                        X_np = X.numpy(force=True)
                        X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                        if self.verbose > 2:
                            F_np = (- X_grad).numpy(force=True)
                            F_tup = np.split(F_np, batch_slice_indx[1:-1], axis=1)
                    else:
                        X_tup = (X.numpy(force=True),)
                        if self.verbose > 2:
                            F_tup = (- X_grad.numpy(force=True),)
                    self.logger.info(f" Coordinates:\n")
                    X_str = [
                        np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        for xi in X_tup
                    ]
                    [self.logger.info(f'{x_str}\n') for x_str in X_str]
                if self.verbose > 2:
                    self.logger.info(f" Forces:\n")
                    X_str = [
                        np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        for xi in F_tup
                    ]
                    [self.logger.info(f'{x_str}\n') for x_str in X_str]

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

        if self.verbose > 0:
            if is_main_loop_converge:
                self.logger.info(
                    '-' * 100 + f'\nAll Structures were Converged.\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s\n'
                )
            else:
                self.logger.info(
                    '-' * 100 + f'\nSome Structures were NOT Converged yet!\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s\n'
                )

            if self.verbose < 2:  # verbose = 1, brief mode only output last step coords.
                # split batches if specified batch
                if batch_indices is not None:
                    X_np = X.numpy(force=True)
                    X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                    F_np = (- X_grad).numpy(force=True)
                    F_tup = np.split(F_np, batch_slice_indx[1:-1], axis=1)
                else:
                    X_tup = (X.numpy(force=True),)
                    F_tup = (- X_grad.numpy(force=True),)
                self.logger.info(f" Final Coordinates:\n")
                X_str = [
                    np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    for xi in X_tup
                ]
                [self.logger.info(f'{x_str}\n') for x_str in X_str]
                self.logger.info(f" Final Forces:\n")
                X_str = [
                    np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    for xi in F_tup
                ]
                [self.logger.info(f'{x_str}\n') for x_str in X_str]
        else:
            if not is_main_loop_converge: warnings.warn('Some Structures were NOT Converged yet!',
                                                        NotConvergeWarning)
        # release

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
