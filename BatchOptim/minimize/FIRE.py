"""
FIRE Optimization Algorithm. Phys. Rev. Lett., 2006, 97:170201.
"""
import logging
import sys
import time
import warnings
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import numpy as np
import torch as th
from torch import nn

from BM4Ckit.utils._Masses import MASS, N_MASS
from BM4Ckit._print_formatter import FLOAT_ARRAY_FORMAT
from .._utils._warnings import FaildToConvergeWarning


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
            elements: Optional[Sequence[Sequence[str | int]]], the Element of each given atom in X.

        Return:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
            grad of argmin func: Tensor(X.shape), only output when `output_grad` == True. The gradient of X corresponding to minimum.
        """

        t_main = time.perf_counter()
        if func_kwargs is None: func_kwargs = dict()
        if grad_func_kwargs is None: grad_func_kwargs = dict()
        n_batch, n_atom, n_dim = X.shape
        # grad func
        if grad_func is None:
            is_grad_func_contain_y = True

            def grad_func_(x, y, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                g = th.autograd.grad(y, x, grad_shape)
                return g[0]
        else:
            grad_func_ = grad_func
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
        for _args in (func_args, func_kwargs.values(), grad_func_args, grad_func_kwargs.values()):
            for _arg in _args:
                if isinstance(_arg, th.Tensor):
                    _arg.to(self.device)
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
        y = func(X, *func_args, **func_kwargs)
        is_main_loop_converge = False
        converge_mask = th.full((n_batch, 1, 1), False, device=self.device)
        t = th.full((n_batch, 1, 1), self.steplength, device=self.device)
        a = th.full((n_batch, 1, 1), self.alpha, device=self.device)
        n_count = th.zeros((n_batch, 1, 1), dtype=th.int, device=self.device)
        if is_grad_func_contain_y:
            F = - grad_func_(X, y, *grad_func_args, **grad_func_kwargs) * atom_masks
        else:
            F = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

        if self.verbose:
            self.logger.info('-' * 100)
            self.logger.info('Iteration Scheme: FIRE')
            self.logger.info('-' * 100)
        # MAIN LOOP
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        for i in range(self.maxiter):  # Simple Euler
            with th.no_grad():
                t_st = time.perf_counter()

                # Verbose
                E_eps = th.abs(y - y0)
                F_eps = th.abs(F)
                converge_mask = (E_eps < self.E_threshold).unsqueeze(-1).unsqueeze(-1) * (F_eps < self.F_threshold)
                if self.verbose > 0: self.logger.info(f'ITERATION {i:>5d}: MAD_energies: {th.mean(E_eps):>5.7e}, '
                                                      f'MAX_F: {th.max(F_eps):>5.7e}, TIME: {time.perf_counter() - t_st:>6.4f} s')
                if self.verbose > 1:
                    X_str = np.array2string(X.numpy(force=True), **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    self.logger.info(f'\n{X_str}\n')
                    self.logger.info(f'Energies: {y.detach().cpu().numpy()}')
                    self.logger.info(f'step length: {t[:, 0, 0].squeeze().detach().cpu().numpy()}')
                    self.logger.info(f'Converged: {th.all(converge_mask, dim=(1, 2)).cpu().numpy()}\n')
                # Criteria
                if th.all(converge_mask):
                    is_main_loop_converge = True
                    break

                # Forward Euler Algo. to update X & v
                v = v + F * t / masses  # (n_batch, n_atom, n_dim)
                X = X + v * t  #* atom_masks
            y0 = y.detach().clone()
            X.requires_grad_()
            y = func(X, *func_args, **func_kwargs)
            if is_grad_func_contain_y:
                F = - grad_func_(X, y, *grad_func_args, **grad_func_kwargs) * atom_masks
            else:
                F = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

            with th.no_grad():
                F_hat = F / th.linalg.norm(F, dim=(-2, -1), keepdim=True)
                p = (F.flatten(-2, -1).unsqueeze(1)) @ (
                    v.flatten(-2, -1).unsqueeze(-1))  # (n_batch, n_dim, n_atom) @ (n_batch, n_atom, n_dim) -> (n_batch, 1, 1)
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
        else:
            if not is_main_loop_converge: warnings.warn('Some Structures were NOT Converged yet!', FaildToConvergeWarning)
        # output
        if output_grad:
            return y, X, F
        else:
            return y, X  #, ptlist  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
