""" Conjugate gradient algorithm for structure optimization """
import logging
import sys
import time
import warnings
from itertools import accumulate
from typing import Dict, Any, Literal, Optional, Sequence, Tuple, List

import numpy as np
import torch as th
from torch import nn

from BM4Ckit._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from utils import IrregularTensorReformat
from .._utils._line_search import _LineSearch
from .._utils._warnings import FaildToConvergeWarning


class CG:
    def __init__(
            self,
            iter_scheme: Literal["PR", "PR+", "FR", "SD", "WYL"] = "PR+",
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            linesearch: Literal["Backtrack", "Wolfe", "Golden", "Newton", "None"] = "Backtrack",
            linesearch_maxiter: int = 10,
            linesearch_thres: float = 0.02,
            linesearch_factor: float = 0.6,
            steplength: float = 0.1,
            device: str | th.device = "cpu",
            verbose: int = 2,
    ) -> None:
        r"""
        Conjugate Gradient for optimization.

        Parameters:
            iter_scheme: Iterative scheme of conjugate gradient algorithm.
                "PR" for Polak-Ribière-Polyak algo. [1, 2],
                "PR+" for modified PR by Powell [3],
                "FR" for Fletcher-Reeves algo. [4],
                "SD" for steepest descent algo.
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
            [1] USSR Computational Mathematics and Mathematical Physics, 1969, 9: 94–112.
            [2] Rev. Francaise Informat Recherche Operationelle 3e Annee, 1969, 16(3): 35–43.
            [3] SIAM Review, 1986, 28: 487–500.
            [4] Computer Journal, 1964, 7: 149–154.
            [5] Pacific J. Math., 1966, 16: 1-3.

        """
        warnings.filterwarnings("always", category=FaildToConvergeWarning)
        warnings.filterwarnings("always")

        self.iterform: str = iter_scheme
        self.linesearch: str = linesearch
        self.steplength: float = steplength
        self.linesearch_maxiter = linesearch_maxiter
        self.linesearch_thres = linesearch_thres
        self.linesearch_factor = linesearch_factor
        self.verbose = verbose
        self.device = device
        self._line_search = _LineSearch(
            linesearch,
            maxiter=linesearch_maxiter,
            thres=linesearch_thres,
            steplength=steplength,
            factor=linesearch_factor,
        )
        assert (maxiter > 0) and isinstance(maxiter, int), f'Invalid `maxiter` value: {maxiter}. It must be a positive integer.'
        self.E_threshold = E_threshold
        self.F_threshold = F_threshold
        self.maxiter = maxiter

        # shape of X
        self.n_dim = None
        self.n_atom = None
        self.n_batch = None

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
            X: th.Tensor | List[th.Tensor],
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs:Dict | None=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None=None,
            is_grad_func_contain_y: bool = True,
            output_grad: bool = False,
            batch_indices: List | Tuple | th.Tensor = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor] | Tuple[th.Tensor, th.Tensor, th.Tensor]:
        r"""
        Run the Conjugate gradient

        Args:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            grad_func: user-defined function that grad_func(X, ...) return the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X i.e., grad = grad_func(X, y, ...), else grad = grad_func(X, ...)
            output_grad: bool, whether output gradient of last step.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.

        Return:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
            grad of argmin func: Tensor(X.shape), only output when `output_grad` == True. The gradient of X corresponding to minimum.
        """
        # check vars
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        if func_kwargs is None:
            func_kwargs = dict()
        if isinstance(X, th.Tensor):
            n_batch, n_atom, n_dim = X.shape
            irr_tensor_converter = None
        elif isinstance(X, List):
            irr_tensor_converter = IrregularTensorReformat()
            X, pad_mask = irr_tensor_converter.regularize(X)
            n_batch, n_atom, n_dim = X.shape
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
            batch_indx_dict = {i: slice(_, batch_slice_indx[i + 1]) for i, _ in enumerate(batch_slice_indx[:-1])}  # dict of {batch indx: split point slice}
        else:
            n_inner_batch = 1
            batch_slice_indx = []
            batch_indx_dict = dict()

        t_main = time.perf_counter()
        E_threshold = self.E_threshold
        F_threshold = self.F_threshold
        maxiter = self.maxiter
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        if grad_func is None:
            is_grad_func_contain_y = True
            # default grad function
            def grad_func_(x, y, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                _g = th.autograd.grad(y, x, grad_shape)
                return _g[0]
        else:
            grad_func_ = grad_func
        # Selective dynamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f"fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not have the same shape of X (shape: {X.shape}).")
        if irr_tensor_converter is not None: atom_masks = atom_masks * pad_mask

        # set variables to given device
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
        p = 0.0  # descent direction
        F_eps = 0.0
        g_old = th.full((n_batch, n_atom * n_dim, 1), 1e-6, dtype=th.float32, device=self.device)  # initial old grad
        beta = 0.0  # deprecated, the coefficient of conjugate direction: p(k+1) = - g(k) + beta * p(k)
        converge_mask = th.full(
            (n_batch, 1, 1), fill_value=False, device=self.device, dtype=th.bool
        )
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # test plot <<<
        if self.verbose:
            self.logger.info("-" * 100)
            self.logger.info(f"Iteration Scheme: {self.iterform}")
            self.logger.info("-" * 100)
        # MAIN LOOP
        for numit in range(maxiter):  # type: ignore
            # backward & obtain grad
            #th.cuda.empty_cache()
            X.requires_grad_()
            energies: th.Tensor = th.where(converge_mask[:, 0, 0], energies_old, func(X, *func_args, **func_kwargs))
            # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
            if energies.shape[0] != self.n_batch:
                assert batch_indices is not None, f"batch indices is None while shape of model output ({energies.shape}) does not match batch size."
                assert energies.shape[0] == n_inner_batch, f"shape of output ({energies.shape}) does not match given batch indices"
                self.is_concat_X = True
            else:
                self.is_concat_X = False
            if is_grad_func_contain_y:
                X_grad = th.where(
                    converge_mask,
                    F_eps,
                    grad_func_(X, energies, *grad_func_args, **grad_func_kwargs),
                )
            else:
                X_grad = th.where(
                    converge_mask,
                    F_eps,
                    grad_func_(X, *grad_func_args, **grad_func_kwargs),
                )
            X_grad = X_grad.detach() * atom_masks
            X = X.detach()
            energies = energies.detach()
            with th.no_grad():
                if X_grad.shape != X.shape:
                    raise RuntimeError(f"X_grad ({X_grad.shape}) and X ({X.shape}) have different shapes.")
                # calc. criteria
                E_eps = th.abs(energies - energies_old)  # (n_inner_batch, )
                energies_old = energies.detach().clone()
                # manage the irregular tensors
                if self.is_concat_X:
                    F_eps = th.abs(X_grad)  # (1, n_batch*n_atom, 3)
                    f_converge = th.cat([th.atleast_1d(th.all(F_eps[0, batch_indx_dict[i]] < F_threshold)) for i in range(n_inner_batch)])  # Tensor[bool], length == n_inner_batch
                    converge_mask = (E_eps < E_threshold) * f_converge  # (n_inner_batch, ), to stop the update of converged samples.
                    converge_str = converge_mask.numpy(force=True)
                    converge_mask = th.repeat_interleave(
                        converge_mask.unsqueeze(0).unsqueeze(-1),
                        th.tensor(batch_indices, device=self.device),
                        dim=1
                    )  # (1, n_inner_batch*n_atom, 1)
                else:
                    F_eps = th.abs(X_grad)  # (n_batch, n_atom, 3)
                    converge_mask = (E_eps < E_threshold).unsqueeze(-1).unsqueeze(-1) * th.all(F_eps < F_threshold, dim=(1, 2), keepdim=True)  # To stop the update of converged samples.
                    converge_str = (converge_mask[:, 0, 0]).numpy(force=True)
                # split batches if specified batch
                if batch_indices is not None:
                    X_tup = th.split(X, batch_indices, dim=1)
                    #F_tup = th.split(- X_grad, batch_indices)
                else:
                    X_tup = (X, )
                    #F_tup = (- X_grad, )
                # judge & print
                # Verbose
                if self.verbose > 0:
                    self.logger.info(f"ITERATION {numit:>5d}\n "
                                     f"MAD_energies: {np.array2string(E_eps.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"MAX_F: {th.max(F_eps):>5.7e}\n "
                                     f"Energies: {np.array2string(energies.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Converged: {converge_str}\n "
                                     f"TIME: {time.perf_counter() - t_st:>6.4f} s\n")
                if self.verbose > 1:
                    self.logger.info(f" Coordinates:\n")
                    X_str = [
                        np.array2string(xi.numpy(force=True), **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        for xi in X_tup
                    ]
                    [self.logger.info(f'{x_str}\n') for x_str in X_str]

                if th.all(converge_mask):
                    is_main_loop_converge = True
                    break

                t_st = time.perf_counter()

                # Conj. form
                g: th.Tensor = th.flatten(X_grad, 1, 2)  # (n_batch, n_atom*3)
                g.unsqueeze_(-1)  # grad: (n_batch, n_atom*3, 1)
                if self.iterform != "SD":
                    gogo = g_old.mT @ g_old
                    ggo = g.mT @ g_old  # (n_batch, 1, 1)
                    gg = g.mT @ g  # (n_batch, 1, 1)
                    if self.iterform == "PR+":
                        beta = (gg - ggo) / gogo  # (n_batch, 1, 1)
                        beta = th.where(beta < 0.0, 0.0, beta)
                    elif self.iterform == "PR":
                        beta = (gg - ggo) / gogo  # (n_batch, 1, 1)
                    elif self.iterform == "FR":
                        beta = gg / gogo
                    elif self.iterform == "WYL":
                        beta = (gg - th.sqrt(gg / gogo) * ggo) / gogo
                    else:
                        raise NotImplementedError("Unknown Iterative Scheme.")
                    # Restart
                    is_restart = (ggo >= 0) * (gg > ggo) * (gogo >= gg)
                    beta = th.where(is_restart, 0.0, beta)
                    if self.verbose > 0:
                        self.logger.info(f"Restart: {is_restart.flatten().cpu().numpy()}")
                    # update old grad
                    g_old = g.detach().clone()  # (n_batch, n_atom*3, 1)
                    # update directions
                    p = -g + beta * p  # (n_batch, n_atom*3, 1)
                else:
                    p = -g

            # Line Search -> steplength: (n_batch, 1, 1)
            alpha = th.where(
                converge_mask,
                0.0,
                self._line_search(
                    func,
                    X,
                    energies,
                    g,
                    p,
                    func_args=func_args,
                    func_kwargs=func_kwargs,
                ),
            )
            with th.no_grad():
                if self.verbose > 0:
                    self.logger.info(
                        f"step length: {alpha[:, 0, 0].squeeze().numpy(force=True)}"
                    )
                # update X
                X = X + alpha * p.view(n_batch, n_atom, n_dim) # (n_batch, n_atom, 3) + (n_batch, 1, 1) * (n_batch, n_atom, 3) * (n_batch, n_atom, 3)
                #ptlist.append(X[:, None, :, 0].numpy(force=True))  # test plot <<<
                # Check NaN
                if th.any(energies != energies):
                    raise RuntimeError(f"NaN Occurred in output: {energies}")

        if self.verbose > 0:
            if is_main_loop_converge:
                self.logger.info(
                    "-" * 100
                    + f"\nAll Structures were Converged.\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s"
                )
            else:
                self.logger.info(
                    "-" * 100
                    + "\nSome Structures were NOT Converged yet!\nMAIN LOOP Done."
                )
        else:
            if not is_main_loop_converge:
                warnings.warn(
                    "Some Structures were NOT Converged yet!", FaildToConvergeWarning
                )

        # output
        if irr_tensor_converter is not None:
            X = irr_tensor_converter.recover(X)
            X_grad = irr_tensor_converter.recover(X_grad)
        if output_grad:
            return energies, X, X_grad
        else:
            return energies, X  #, ptlist  # test plot <<<
