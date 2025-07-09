#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseOpt.py
#  Environment: Python 3.12

import logging
import sys
from itertools import accumulate
from typing import Dict, Any, Literal, Optional, Sequence, Tuple, List
import time
import warnings

import numpy as np
import torch as th
from torch import nn
from BM4Ckit.BatchOptim._utils._line_search import _LineSearch
from BM4Ckit.BatchOptim._utils._warnings import FaildToConvergeWarning
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BM4Ckit.utils.scatter_reduce import scatter_reduce


class _BaseOpt:
    def __init__(
            self,
            iter_scheme: str,
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            linesearch: Literal['Backtrack', 'Wolfe', 'NWolfe', '2PT', '3PT', 'Golden', 'Newton', 'None'] = 'Backtrack',
            linesearch_maxiter: int = 10,
            linesearch_thres: float = 0.02,
            linesearch_factor: float = 0.6,
            steplength: float = 0.5,
            use_bb: bool = True,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ) -> None:
        r"""
        A Base Framework of Algorithm for optimization.
        
        Args:
            E_threshold: float, threshold of difference of func between 2 iterations.
            F_threshold: float, threshold of gradient of func.
            maxiter: int, max iterations.
            linesearch: Scheme of linesearch.
                "None" for fixed steplength.
                "Backtrack" for backtracking until Armijo's condition [5] was satisfied.
                "Golden" for the golden section algo.
                "Newton" for 1D Newton algo., which was modified to avoid divergence.
                "Wolfe" for quadratic interpolation search until weak Wolfe condition was satisfied.
                "NWolfe" is the same as "Wolfe" but directional derivative was calculated by finite difference (might use less memory).
                "2PT" is a simple 1-step 2-point quadratic interpolation.
                "3PT" is a simple 1-step 3-point cubic interpolation.
            linesearch_maxiter: Max iterations for linesearch.
            linesearch_thres: Threshold for linesearch. Only for "Golden" and "Newton".
            linesearch_factor: A factor in linesearch. Shrinkage factor for "Backtrack", scaling factor in interval search for "Golden" and line steplength for "Newton".
            steplength: The initial step length.
            use_bb: whether to use Barzilai-Borwein steplength (BB1 or long BB) as initial steplength instead of fixed one.
            device: The device that program runs on.
            verbose: amount of print information.
        
        Method:
            run: running the main optimization program.

        """
        warnings.filterwarnings('always', category=FaildToConvergeWarning)
        warnings.filterwarnings('always', )

        self.iterform = iter_scheme
        self.linesearch: str = linesearch
        self.steplength: float = steplength
        self.linesearch_maxiter = linesearch_maxiter
        self.linesearch_thres = linesearch_thres
        self.linesearch_factor = linesearch_factor
        self.verbose = verbose
        self.device = device
        self.use_bb = use_bb
        self._line_search = _LineSearch(
            linesearch,
            maxiter=linesearch_maxiter,
            thres=linesearch_thres,
            factor=linesearch_factor
        )
        self.E_threshold = E_threshold
        self.F_threshold = F_threshold
        self.maxiter = maxiter

        self.n_batch, self.n_atom, self.n_dim = None, None, None
        self.converge_mask = None  # To record the batch which has converged and not update.
        self.is_concat_X = False   # whether the output of `func` was concatenated.

        # logger
        self.logger = logging.getLogger('Main.OPT')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if not self.logger.hasHandlers():
            log_handler = logging.StreamHandler(sys.stdout, )
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor] | Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Run the Optimization Algorithm.

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X
                i.e., grad = grad_func(X, y, *grad_func_args, **grad_func_kwargs), else grad = grad_func(X, *grad_func_args, **grad_func_kwargs)
            output_grad: bool, whether output gradient of last step.
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
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
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
            n_inner_batch = len(batch_indices)
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            batch_indx_dict = {
                i: slice(_, batch_slice_indx[i + 1]) for i, _ in enumerate(batch_slice_indx[:-1])
            }  # dict of {batch indx: split point slice}
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)
            batch_tensor = self.batch_tensor
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )
        else:
            n_inner_batch = 1
            batch_indx_dict = dict()
            batch_tensor = None
        # initialize vars
        E_threshold = self.E_threshold
        F_threshold = self.F_threshold
        maxiter = self.maxiter
        n_batch, n_atom, n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        if grad_func is None:
            is_grad_func_contain_y = True
            def grad_func_(y, x, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                _g = th.autograd.grad(y, x, grad_shape)
                return _g[0]
        else:
            grad_func_ = grad_func
        # Selective dyamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'The shape of fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        #atom_masks = atom_masks.flatten(-2, -1).unsqueeze(-1)  # (n_batch, n_atom*n_dim, 1)
        # other check
        if (not isinstance(maxiter, int)) or (maxiter <= 0):
            raise ValueError(f'Invalid value of maxiter: {maxiter}. It would be an integer greater than 0.')

        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.eval()
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)

        # initialize
        energies_old = th.inf
        is_main_loop_converge = False
        t_st = time.perf_counter()
        self.initialize_algo_param()
        alpha = th.full_like(X, self.steplength)  # initial step length
        p = 0.
        self.converge_mask = th.full((n_batch, 1, 1), fill_value=False, device=self.device, dtype=th.bool)
        g_old = th.full((n_batch, n_atom * n_dim, 1), 1e-6, dtype=th.float32, device=self.device)  # initial old grad
        displace = th.full_like(g_old, th.inf)
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
            self.logger.info(f'Iteration Scheme: {self.iterform}')
        self.logger.info('-' * 100)
        # MAIN LOOP
        with th.no_grad():
            with th.enable_grad():
                X.requires_grad_()
                energies: th.Tensor = th.where(self.converge_mask[:, 0, 0], energies_old, func(X, *func_args, **func_kwargs))
                # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
                if energies.shape[0] != self.n_batch:
                    assert batch_indices is not None, (f"batch indices is None "
                                                       f"while shape of model output ({energies.shape}) does not match batch size ({self.n_batch}).")
                    assert energies.shape[0] == n_inner_batch, f"shape of output ({energies.shape}) does not match given batch indices"
                    self.is_concat_X = True
                else:
                    self.is_concat_X = False
                # calc. grad
                if is_grad_func_contain_y:
                    X_grad = th.where(self.converge_mask, 0., grad_func_(energies, X, *grad_func_args, **grad_func_kwargs))
                else:
                    X_grad = th.where(self.converge_mask, 0., grad_func_(X, *grad_func_args, **grad_func_kwargs))
                if X_grad.shape != X.shape:
                    raise RuntimeError(f'X_grad ({X_grad.shape}) and X ({X.shape}) have different shapes.')
            energies.detach_()
            X_grad = X_grad.detach() * atom_masks
            X = X.detach()
            g: th.Tensor = th.flatten(X_grad, 1, 2)  # (n_batch, n_atom*3)
            g.unsqueeze_(-1)  # grad: (n_batch, n_atom*3, 1)
            for numit in range(maxiter):
                # Calc. Criteria
                E_eps = th.abs(energies - energies_old)  # (n_batch, )
                energies_old = energies.detach().clone()
                # manage the irregular tensors
                if self.is_concat_X:
                    F_eps = th.abs(X_grad)  # (1, n_batch*n_atom, 3)
                    f_converge = scatter_reduce(
                        th.all(F_eps[0] < F_threshold, dim=-1).to(th.int64), self.batch_scatter, 0, 'amin', 1
                    ).bool()
                    self.converge_mask = (E_eps < E_threshold) * f_converge  # (n_inner_batch, ), to stop the update of converged samples.
                    _stepsize_converge_mask = self.converge_mask.unsqueeze(0).unsqueeze(-1)
                    converge_str = self.converge_mask.numpy(force=True)
                    self.converge_mask = self.converge_mask.unsqueeze(0).unsqueeze(-1)[:, self.batch_scatter, ...]
                else:
                    F_eps = th.abs(X_grad)  # (n_batch, n_atom, 3)
                    self.converge_mask = ((E_eps < E_threshold).unsqueeze(-1).unsqueeze(-1) *
                                     th.all(F_eps < F_threshold, dim=(1, 2),keepdim=True))  # (n_batch, 1, 1), to stop the update of converged samples.
                    _stepsize_converge_mask = self.converge_mask.transpose(-2, -1)
                    converge_str = (self.converge_mask[:, 0, 0]).numpy(force=True)

                # Print information / Verbose
                if self.verbose > 0:
                    self.logger.info(f"ITERATION {numit:>5d}\n "
                                     f"MAD_energies: {np.array2string(E_eps.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"MAX_F:        {th.max(F_eps):>5.7e}\n "
                                     f"Energies:     {np.array2string(energies.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Converged:    {converge_str}\n "
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
                t_st = time.perf_counter()

                # search directions
                p = self._update_direction(g, g_old, p, X)  # (n_batch, n_atom*3, 1)
                # use BB steplength
                if self.use_bb:
                    g_go = g - g_old
                    _steplength = displace.mT@displace/(displace.mT@g_go)  # BB1
                    _steplength = th.where(
                        (_steplength < 1.5 * self.steplength) * (_steplength > 1e-4),
                        _steplength,
                        self.steplength
                    )
                else:
                    _steplength = self.steplength
                # search step length -> steplength: (n_batch, 1, 1)
                alpha = th.where(
                    th.all(_stepsize_converge_mask, dim=1, keepdim=True),
                    0.,
                    self._line_search(
                        func, grad_func_, X, energies, g, p, _steplength, is_grad_func_contain_y,
                        func_args=func_args, func_kwargs=func_kwargs, grad_func_args=grad_func_args, grad_func_kwargs=grad_func_kwargs,
                        converge_mask=_stepsize_converge_mask.squeeze((0, -1))
                    )
                )
                # update X
                displace = alpha * p  # (n_batch, 1, 1) * (n_batch, n_atom*3, 1)
                X = X + displace.view(n_batch, n_atom, n_dim)  # (n_batch, n_atom, 3) + (n_batch, n_atom, 3)
                # update old grad
                g_old = g.detach().clone()  # (n_batch, n_atom*3, 1)
                # calc. new energy & grad.
                with th.enable_grad():
                    X.requires_grad_()
                    energies: th.Tensor = th.where(self.converge_mask[:, 0, 0], energies_old, func(X, *func_args, **func_kwargs))
                    if is_grad_func_contain_y:
                        X_grad = th.where(
                            self.converge_mask,
                            F_eps,
                            grad_func_(energies, X, *grad_func_args, **grad_func_kwargs)
                        )
                    else:
                        X_grad = th.where(
                            self.converge_mask,
                            F_eps,
                            grad_func_(X, *grad_func_args, **grad_func_kwargs)
                        )
                energies.detach_()
                X_grad = X_grad.detach() * atom_masks * (~self.converge_mask)
                X.detach_()
                g: th.Tensor = th.flatten(X_grad, 1, 2)  # (n_batch, n_atom*3)
                g.unsqueeze_(-1)  # grad: (n_batch, n_atom*3, 1)
                # update algo. parameters.
                self._update_algo_param(g, g_old, p, displace)

                # print steplength
                if self.verbose > 0:
                    self.logger.info(f" step length: {alpha[:, 0, 0].squeeze().numpy(force=True)}\n")
                # Check NaN
                if th.any(energies != energies): raise RuntimeError(f'NaN Occurred in output: {energies}')

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
                                                        FaildToConvergeWarning)
        # release

        # output
        if output_grad:
            return energies, X, X_grad
        else:
            return energies, X  #, ptlist  # test <<<

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

    def _update_direction(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor) -> th.Tensor:
        """
        Override this method to implement X update algorithm.
        Args:
            g: (n_batch, n_atom*3, 1), the gradient of X at this step
            g_old: (n_batch, n_atom*3, 1), the gradient of X at last step
            p: (n_batch, n_atom*3, 1), the update direction of X at last step
            X: (n_batch, n_atom, 3), the independent vars X.

        Returns:
            p: th.Tensor, the new update direction of X.
        """
        raise NotImplementedError

    def _update_algo_param(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, displace: th.Tensor) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.
        Args:
            g: (n_batch, n_atom*3, 1), the gradient of X at this step
            g_old: (n_batch, n_atom*3, 1), the gradient of X at last step
            p: (n_batch, n_atom*3, 1), the update direction of X at last step
            displace: (n_batch, n_atom*3, 1), the displacement of X at this step. displace = step-length * p

        Returns: None
        """
        raise NotImplementedError
