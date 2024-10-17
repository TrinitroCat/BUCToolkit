# ruff: noqa: E701, F401, E703
import logging
import sys
from typing import Dict, Any, Literal, Optional, Sequence
import time
import warnings

import numpy as np
import torch as th
from torch import nn
from BM4Ckit.BatchOptim._utils._line_search import _LineSearch
from BM4Ckit.BatchOptim._utils._warnings import FaildToConvergeWarning
from BM4Ckit._print_formatter import FLOAT_ARRAY_FORMAT


class QN:
    """  """

    def __init__(self,
                 iter_scheme: Literal['BFGS', 'Newton'] = 'BFGS',
                 E_threshold: float = 1e-3, F_threshold: float = 0.05, maxiter: int = 100,
                 linesearch: Literal['Backtrack', 'Wolfe', 'Golden', 'Newton', 'None'] = 'Backtrack',
                 linesearch_maxiter: int = 10,
                 linesearch_thres: float = 0.02,
                 linesearch_factor: float = 0.6,
                 steplength: float = 0.5,
                 device: str | th.device = 'cpu',
                 verbose: int = 2) -> None:
        r"""
        Quasi-Newton Algorithm for optimization.
        
        Parameters:
            iter_scheme: Iterative scheme of conjugate gradient algorithm. 
                "BFGS" for Broyden, Fletcher, Goldfarb, Shanno scheme [1-4].
                "XWW" for Xiao-Wei-Wang scheme [5].
            E_threshold: float, threshold of difference of func between 2 iteration.
            F_threshold: float, threshold of gradient of func.
            maxiter: int, max iterations.
            linesearch: Scheme of linesearch.
                "None" for fixed steplength.
                "Backtrack" for backtracking while Armijo's condition [5] was satisfied.
                "Golden" for golden section algo.
                "Newton" for 1D Newton algo., which was modified to avoid divergence.
            linesearch_maxiter: Max iterations for linesearch.
            linesearch_thres: Threshold for linesearch. Only for "Golden" and "Newton".
            linesearch_factor: A factor in linesearch. Shrinkage factor for "Backtrack", scaling factor in interval search for "Golden" and line steplength for "Newton".
            steplength: The initial step length.
            device: The device that program runs on.
            verbose: amount of print information.
        
        Method:
            run: running the main optimization program.

        References:
            [1] 
            [2] 
            [3] 
            [4] 
            [5] Comput. Math. Appl., 2008, 56: 1001.

        """
        warnings.filterwarnings('always', category=FaildToConvergeWarning)
        warnings.filterwarnings('always', )

        self.iterform: str = iter_scheme
        self.linesearch: str = linesearch
        self.steplength: float = steplength
        self.linesearch_maxiter = linesearch_maxiter
        self.linesearch_thres = linesearch_thres
        self.linesearch_factor = linesearch_factor
        self.verbose = verbose
        self.device = device
        self._line_search = _LineSearch(linesearch, maxiter=linesearch_maxiter, thres=linesearch_thres,
                                        steplength=steplength, factor=linesearch_factor)
        self.E_threshold = E_threshold
        self.F_threshold = F_threshold
        self.maxiter = maxiter

        # logger
        self.logger = logging.getLogger('Main.OPT')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if not self.logger.hasHandlers():
            log_handler = logging.StreamHandler(sys.stdout, )
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)

    def run(self, func: Any | nn.Module, X: th.Tensor, grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(), func_kwargs: Dict | None = None, grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None, ):
        """
        Run the Quasi-Newton Algorithm.

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

        Return:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
        """
        t_main = time.perf_counter()
        if func_kwargs is None:
            func_kwargs = dict()
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        E_threshold = self.E_threshold
        F_threshold = self.F_threshold
        maxiter = self.maxiter
        n_batch, n_atom, n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        if grad_func is None:
            is_grad_func_contain_y = True
            if self.iterform != 'Newton':
                def grad_func_(x, y, grad_shape=None):
                    if grad_shape is None:
                        grad_shape = th.ones_like(y)
                    g = th.autograd.grad(y, x, grad_shape)
                    return g[0]
            else:  # for 'Newton', required 2nd derivative
                def grad_func_(x, y, grad_shape=None):
                    if grad_shape is None:
                        grad_shape = th.ones_like(y)
                    g = th.autograd.grad(y, x, grad_shape, create_graph=True)
                    return g[0]
        else:
            grad_func_ = grad_func
        # Selective dyamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not have the same shape of X (shape: {X.shape}).')
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
        for _args in (func_args, func_kwargs.values(), grad_func_args, grad_func_kwargs.values()):
            for _arg in _args:
                if isinstance(_arg, th.Tensor):
                    _arg.to(self.device)

        # initialize
        energies_old = th.inf
        is_main_loop_converge = False
        t_st = time.perf_counter()
        p = 0.  # descent direction
        H_inv = (th.eye(n_atom * n_dim, device=self.device).unsqueeze(0)).expand(n_batch, -1, -1)  # Initial quasi-inverse Hessian Matrix  (n_batch, n_atom*n_dim, n_atom*n_dim)
        Ident = (th.eye(n_atom * n_dim, device=self.device).unsqueeze(0)).expand(n_batch, -1, -1)  # prepared identity matrix
        alpha = th.full_like(X, self.steplength)  # steplength
        converge_mask = th.full((n_batch, 1, 1), fill_value=False, device=self.device, dtype=th.bool)
        ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc.
        if self.verbose:
            print('-' * 100)
            print(f'Iteration Scheme: {self.iterform}')
        print('-' * 100)
        # MAIN LOOP
        X.requires_grad_()
        energies: th.Tensor = th.where(converge_mask[:, 0, 0], energies_old, func(X, *func_args, **func_kwargs))
        if is_grad_func_contain_y:
            X_grad = th.where(converge_mask, 0., grad_func_(X, energies, *grad_func_args, **grad_func_kwargs))
        else:
            X_grad = th.where(converge_mask, 0., grad_func_(X, *grad_func_args, **grad_func_kwargs))
        if X_grad.shape != X.shape:
            raise RuntimeError(f'X_grad ({X_grad.shape}) and X ({X.shape}) have different shapes.')
        X_grad = X_grad * atom_masks
        g: th.Tensor = th.flatten(X_grad, 1, 2)  # (n_batch, n_atom*3)
        g.unsqueeze_(-1)  # grad: (n_batch, n_atom*3, 1)
        for numit in range(maxiter):
            with th.no_grad():
                # Calc. Criteria
                E_eps = th.abs(energies - energies_old)  # (n_batch, )
                energies_old = energies.detach().clone()
                F_eps = th.abs(X_grad) * atom_masks  # (n_batch, n_atom, 3)
                converge_mask = (E_eps < E_threshold).unsqueeze(-1).unsqueeze(-1) * th.all(F_eps < F_threshold, dim=(1, 2),
                                                                                           keepdim=True)  # To stop the update of converged samples.
                # Verbose
                if self.verbose > 0: print(
                    f'ITERATION {numit:>5d}: MAD_energies: {th.mean(E_eps):>5.7e}, MAX_F: {th.max(F_eps):>5.7e}, TIME: {time.perf_counter() - t_st:>6.4f} s')
                if self.verbose > 1:
                    print(f'Energies: {energies.detach().cpu().numpy()}')
                    X_str = np.array2string(X.numpy(force=True), **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    self.logger.info(f'\n{X_str}\n')
                    print(f'step length: {alpha[:, 0, 0].squeeze().detach().cpu().numpy()}')
                    print(f'Converged: {th.all(converge_mask, dim=(1, 2)).cpu().numpy()}\n')

                # judge thres
                if th.all(converge_mask):
                    is_main_loop_converge = True
                    break
                t_st = time.perf_counter()

            # QN scheme
            if self.iterform == 'BFGS':
                p = -H_inv @ g  # (n_batch, n_atom*3, n_atom*3) @ (n_batch, n_atom*3, 1)
                # Line Search -> steplength: (n_batch, 1, 1)
                alpha = th.where(converge_mask[:, :1, :1],
                                 0.,
                                 self._line_search(func, X, energies, g, p, func_args=func_args,
                                                   func_kwargs=func_kwargs))
                with th.no_grad():
                    # update X
                    displace = alpha * p  # (n_batch, 1, 1) * (n_batch, n_atom*3, 1)
                    X = X + displace.view(n_batch, n_atom, n_dim)  # (n_batch, n_atom, 3) + (n_batch, n_atom, 3)
                # update old grad
                g_old = g.detach().clone()  # (n_batch, n_atom*3, 1)
                X.requires_grad_()
                energies: th.Tensor = th.where(converge_mask[:, 0, 0], energies_old, func(X, *func_args, **func_kwargs))
                if is_grad_func_contain_y:
                    X_grad = th.where(converge_mask, F_eps,
                                      grad_func_(X, energies, *grad_func_args, **grad_func_kwargs))
                else:
                    X_grad = th.where(converge_mask, F_eps, grad_func_(X, *grad_func_args, **grad_func_kwargs))
                X_grad = X_grad * atom_masks
                g: th.Tensor = th.flatten(X_grad, 1, 2)  # (n_batch, n_atom*3)
                g.unsqueeze_(-1)  # grad: (n_batch, n_atom*3, 1)
                with th.no_grad():
                    # update H_inv
                    g_go = g - g_old
                    gamma = 1 / (
                            displace.mT @ g_go + 1e-20)  # (n_batch, 1, n_atom*3) @ (n_batch, n_atom*3, 1) -> (n_batch, 1, 1), 1e-20 to avoid 1/0
                    # BFGS Scheme:
                    # (n_batch, n_atom*n_dim, n_atom*n_dim) - (n_batch, 1, 1) * (n_batch, n_atom*n_dim, 1)@(n_batch, 1, n_atom*n_dim)
                    H_inv = th.where(converge_mask[:, :1, :1],
                                     0.,
                                     (Ident - gamma * displace @ g_go.mT) @ H_inv @ (Ident - gamma * g_go @ displace.mT) + gamma * displace @ displace.mT)

            elif self.iterform == 'XWW':  # Xiao-Wei-Wang # TODO
                p = -H_inv @ g  # (n_batch, n_atom*3, n_atom*3) @ (n_batch, n_atom*3, 1)
                # Line Search -> steplength: (n_batch, 1, 1)
                alpha = th.where(converge_mask[:, :1, :1],
                                 0.,
                                 self._line_search(func, X, energies, g, p, func_args=func_args,
                                                   func_kwargs=func_kwargs))
                pass

            elif self.iterform == 'Newton':
                # Hessian
                H = th.zeros((n_batch, n_atom * n_dim, n_atom * n_dim), device=self.device)
                hess_mask = th.zeros(n_batch, n_atom * n_dim, 1, device=self.device)
                for i in range(n_atom * n_dim):
                    hess_mask[:, i] = 1.
                    H_line = th.where(converge_mask[:, :1, :1], hess_mask,
                                      th.autograd.grad(g, X, hess_mask, retain_graph=True)[
                                          0])  # (n_batch, n_atom*n_dim, 1)
                    H[:, :, i] = H_line.squeeze(-1)
                    hess_mask[:, i] = 0.
                p = - th.where(converge_mask[:, :1, :1], 0., th.linalg.solve(H, g))
                # Line Search -> steplength: (n_batch, 1, 1)
                alpha = th.where(converge_mask[:, :1, :1],
                                 0.,
                                 self._line_search(func, X, energies, g, p, func_args=func_args,
                                                   func_kwargs=func_kwargs))
                # update X
                with th.no_grad():
                    displace = alpha * p * atom_masks  # (n_batch, 1, 1) * (n_batch, n_atom*3, 1) * (n_batch, n_atom*3, 1)
                    X = X + displace.view(n_batch, n_atom, n_dim)  # (n_batch, n_atom, 3) + (n_batch, n_atom, 3)
                # update old grad
                g_old = g.detach().clone()  # (n_batch, n_atom*3, 1)
                X.detach_()
                X.requires_grad_()
                energies: th.Tensor = th.where(converge_mask[:, 0, 0], energies_old, func(X, *func_args, **func_kwargs))
                if is_grad_func_contain_y:
                    X_grad = th.where(converge_mask, 0., grad_func_(X, energies, *grad_func_args, **grad_func_kwargs))
                else:
                    X_grad = th.where(converge_mask, 0., grad_func_(X, *grad_func_args, **grad_func_kwargs))
                g: th.Tensor = th.flatten(X_grad, 1, 2)  # (n_batch, n_atom*3)
                g.unsqueeze_(-1)  # grad: (n_batch, n_atom*3, 1)

            else:
                raise NotImplementedError
            # Check NaN
            if th.any(energies != energies): raise RuntimeError(f'NaN Occurred in output: {energies}')

            ptlist.append(X[:, None, :, 0].numpy(force=True))  # test <<<

        if self.verbose > 0:
            if is_main_loop_converge:
                print(
                    '-' * 100 + f'\nAll Structures were Converged.\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s')
            else:
                print('-' * 100 + '\nSome Structures were NOT Converged yet!\nMAIN LOOP Done.')
        else:
            if not is_main_loop_converge: warnings.warn('Some Structures were NOT Converged yet!',
                                                        FaildToConvergeWarning)
        # output
        if output_grad:
            return energies, X, X_grad
        else:
            return energies, X, ptlist  # test <<<
