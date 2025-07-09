"""
FIRE Optimization Algorithm. Phys. Rev. Lett., 2006, 97:170201.
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: FIRE.py
#  Environment: Python 3.12

import logging
import sys
import time
import warnings
from itertools import accumulate
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import numpy as np
import torch as th
from torch import nn

from BM4Ckit.utils._Element_info import MASS, N_MASS
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from .._utils._warnings import FaildToConvergeWarning
from BM4Ckit.utils.scatter_reduce import scatter_reduce


class FIRE:
    def __init__(
            self,
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            steplength: float = 1.,
            alpha: float = 0.1,
            alpha_fac: float = 0.99,
            fac_inc: float = 1.1,
            fac_dec: float = 0.5,
            N_min: int = 5,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            **kwargs
    ) -> None:
        r"""
        FIRE Algorithm for optimization.

        Args:
            E_threshold: float, threshold of difference of func between 2 iteration.
            F_threshold: float, threshold of gradient of func.
            maxiter: int, the maximum iteration steps.
            steplength: The initial step length i.e. the BatchMD time step.
            alpha:
            alpha_fac:
            fac_inc:
            fac_dec:
            N_min:
            device: The device that program runs on.
            verbose: amount of print information.

        Method:
            run: running the main optimization program.
        """

        if not (0. < alpha_fac < 1.):
            raise ValueError('alpha_fac must between 0 and 1.')
        if not (0. < fac_dec < 1.):
            raise ValueError('fac_dec must between 0 and 1.')
        if fac_inc <= 1.:
            raise ValueError('fac_inc must be greater than 1.')

        self.E_threshold = E_threshold
        self.F_threshold = F_threshold
        self.maxiter = maxiter
        self.steplength = steplength

        self.alpha = alpha
        self.alpha_fac = alpha_fac
        self.fac_inc = fac_inc
        self.fac_dec = fac_dec
        self.N_min = N_min

        self.device = device
        self.verbose = verbose

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
            batch_indices: List | Tuple | th.Tensor = None,
            elements: Sequence[Sequence[str | int]] | None = None
    ) -> Tuple[th.Tensor, th.Tensor] | Tuple[th.Tensor, th.Tensor, th.Tensor]:
        r"""
        Run the Conjugate gradient

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            grad_func: user-defined function that grad_func(X, ...) return the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X i.e., grad = grad_func(X, y, ...), else grad = grad_func(X, ...)
            output_grad: bool, whether output gradient of last step.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]
            elements: Optional[Sequence[Sequence[str | int]]], the Element of each given atom in X.

        Return:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
            grad of argmin func: Tensor(X.shape), only output when `output_grad` == True. The gradient of X corresponding to minimum.
        """

        t_main = time.perf_counter()
        if func_kwargs is None: func_kwargs = dict()
        if grad_func_kwargs is None: grad_func_kwargs = dict()
        # grad func
        if grad_func is None:
            is_grad_func_contain_y = True

            def grad_func_(y, x, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                g = th.autograd.grad(y, x, grad_shape)
                return g[0]
        else:
            grad_func_ = grad_func
        # check batch indices
        if isinstance(X, th.Tensor):
            n_batch, n_atom, n_dim = X.shape
            self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        else:
            raise TypeError(f'`X` must be torch.Tensor or List[torch.Tensor], but occurred {type(X)}.')
        # Check batch indices; irregular batch
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
            batch_tensor = th.tensor(batch_indices, device=self.device)
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                batch_tensor,
                dim=0
            )
            batch_indx_dict = {i: slice(_, batch_slice_indx[i + 1]) for i, _ in enumerate(batch_slice_indx[:-1])}  # dict of {batch indx: split point slice}
        else:
            n_inner_batch = 1
            batch_slice_indx = []
            batch_indx_dict = dict()
        # Selective dynamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not have the same shape of X (shape: {X.shape}).')
        # other check
        if (not isinstance(self.maxiter, int)) or (self.maxiter <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.maxiter}. It would be an integer greater than 0.')
        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.eval()
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)

        X = X.to(self.device)
        # Variables initialize
        v = th.zeros_like(X, device=self.device)
        # manage Atomic Type & Masses
        if elements is None:
            masses = 1.
        elif isinstance(elements, Sequence):
            masses = list()
            for _Elem in elements:
                masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
            masses = th.tensor(masses, dtype=th.float32, device=self.device)
            masses = masses.unsqueeze(-1).expand_as(X)  # (n_batch, n_atom, n_dim)
        else:
            raise TypeError(f'Expected masses is a Sequence[Sequence[...]], but occurred {type(elements)}.')

        y0 = th.inf
        with th.enable_grad():
            X.requires_grad_()
            y = func(X, *func_args, **func_kwargs)
            # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
            if y.shape[0] != self.n_batch:
                assert batch_indices is not None, f"batch indices is None while shape of model output ({y.shape}) does not match batch size."
                assert y.shape[0] == n_inner_batch, f"shape of output ({y.shape}) does not match given batch indices"
                self.is_concat_X = True
            else:
                self.is_concat_X = False
            is_main_loop_converge = False
            converge_mask = th.full((n_batch, 1, 1), False, device=self.device)
            t = th.full((n_batch, 1, 1), self.steplength, device=self.device)
            a = th.full((n_batch, 1, 1), self.alpha, device=self.device)
            n_count = th.zeros((n_batch, 1, 1), dtype=th.int, device=self.device)
            if is_grad_func_contain_y:
                F = - grad_func_(y, X, *grad_func_args, **grad_func_kwargs) * atom_masks
            else:
                F = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

        y = y.detach()
        F = F.detach()
        X = X.detach()

        if self.verbose:
            self.logger.info('-' * 100)
            self.logger.info('Iteration Scheme: FIRE')
            self.logger.info('-' * 100)
        # MAIN LOOP
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        with th.no_grad():
            for numit in range(self.maxiter):  # Simple Euler
                t_st = time.perf_counter()
                # split batches if specified batch
                if batch_indices is not None:
                    X_tup = np.split(X.numpy(force=True), batch_slice_indx[1:-1], axis=1)
                    if self.verbose > 2:
                        F_np = F.numpy(force=True)
                        F_tup = np.split(F_np, batch_slice_indx[1:-1], axis=1)
                    # F_tup = th.split(- X_grad, batch_indices)
                else:
                    X_tup = (X.numpy(force=True),)
                    F_tup = (F.numpy(force=True),)
                # Verbose
                E_eps = th.abs(y - y0)
                if self.is_concat_X:
                    F_eps = th.abs(F)  # (1, n_batch*n_atom, 3)
                    f_converge = scatter_reduce(
                        th.all(F_eps[0] < self.F_threshold, dim=-1).to(th.int64), self.batch_scatter, 0, 'amin', 1
                    ).bool()
                    '''f_converge = th.cat([th.atleast_1d(th.all(F_eps[0, batch_indx_dict[i]] < self.F_threshold)) for i in
                                         range(n_inner_batch)])  # Tensor[bool], length == n_inner_batch'''
                    converge_mask = (E_eps < self.E_threshold) * f_converge  # (n_inner_batch, ), to stop the update of converged samples.
                    converge_check = converge_mask
                    converge_str = converge_check.numpy(force=True)
                    converge_mask = converge_mask.unsqueeze(0).unsqueeze(-1)[:, self.batch_scatter, ...]  # (1, n_batch*n_atom, 3)
                else:
                    F_eps = th.abs(F)  # (n_batch, n_atom, 3)
                    converge_mask = (E_eps < self.E_threshold).unsqueeze(-1).unsqueeze(-1) * th.all(F_eps < self.F_threshold, dim=(1, 2),
                                                                                               keepdim=True)  # To stop the update of converged samples.
                    converge_check = converge_mask[:, 0, 0]
                    converge_str = (converge_mask[:, 0, 0]).numpy(force=True)

                if self.verbose > 0:
                    self.logger.info(f"ITERATION {numit:>5d}\n "
                                     f"MAD_energies: {np.array2string(E_eps.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"MAX_F: {th.max(F_eps):>5.7e}\n "
                                     f"Energies: {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Converged: {converge_str}\n "
                                     f"TIME: {time.perf_counter() - t_st:>6.4f} s\n")
                if self.verbose > 1:
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

                # Criteria
                if th.all(converge_check):  # `converge_check` is shorter than `converge_mask` thus reducing complexity.
                    is_main_loop_converge = True
                    break

                # Forward Euler Algo. to update X & v
                v = v + F * t / masses  # (n_batch, n_atom, n_dim)
                X = X + v * t * (~converge_mask)
                y0 = y.detach().clone()
                with th.enable_grad():
                    X.requires_grad_()
                    y = func(X, *func_args, **func_kwargs)
                    if is_grad_func_contain_y:
                        F = - grad_func_(y, X, *grad_func_args, **grad_func_kwargs) * atom_masks
                    else:
                        F = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

                y = y.detach()
                F = F.detach()
                X = X.detach()
                F_hat = F / th.linalg.norm(F, dim=(-2, -1), keepdim=True)
                # (n_batch, n_dim, n_atom) @ (n_batch, n_atom, n_dim) -> (n_batch, 1, 1)
                p = (F.flatten(-2, -1).unsqueeze(1)) @ (v.flatten(-2, -1).unsqueeze(-1))
                # update velocity
                v = (1 - a) * v + a * th.linalg.norm(v, dim=(-2, -1), keepdim=True) * F_hat
                # if P > 0.
                n_count = n_count + th.where(p > 0., 1, -n_count)
                new_t = th.where(t * self.fac_inc < 10 * self.steplength, t * self.fac_inc, 10 * self.steplength)
                t = th.where(n_count >= self.N_min, new_t, t)
                a = th.where(n_count >= self.N_min, a * self.alpha_fac, a)
                # if P <= 0.
                t = th.where(p <= 0., t * self.fac_dec, t)
                v = th.where(p <= 0., 0., v)
                a = th.where(p <= 0., self.alpha, a)

                #ptlist.append(X[:, None, :, 0].numpy(force=True))  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if self.verbose > 0:
            if is_main_loop_converge:
                self.logger.info('-' * 100 + f'\nAll Structures were Converged.\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s')
            else:
                self.logger.info('-' * 100 + '\nSome Structures were NOT Converged yet!\nMAIN LOOP Done.')

            if self.verbose < 2:  # verbose = 1, brief mode only output last step coords.
                # split batches if specified batch
                if batch_indices is not None:
                    X_np = X.numpy(force=True)
                    X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                    F_np = F.numpy(force=True)
                    F_tup = np.split(F_np, batch_slice_indx[1:-1], axis=1)
                else:
                    X_tup = (X.numpy(force=True),)
                    F_tup = (F.numpy(force=True),)
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
            if not is_main_loop_converge: warnings.warn('Some Structures were NOT Converged yet!', FaildToConvergeWarning)
        # output
        if output_grad:
            return y, X, F
        else:
            return y, X  #, ptlist  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
