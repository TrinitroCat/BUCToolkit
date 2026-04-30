""" Dimer Algo. for transition state searching. J. Chem. Phys. 1999, 111: 7010. """
from itertools import accumulate
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Dimer.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import time
import warnings
import logging
import sys
import os

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.utils._print_formatter import GLOBAL_SCIENTIFIC_ARRAY_FORMAT, FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT, STRING_ARRAY_FORMAT
from BUCToolkit.utils import index_ops
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.utils.setup_loggers import has_any_handler, clear_all_handlers
from BUCToolkit.Bases.BaseMotion import BaseMotion

np.set_printoptions(**GLOBAL_SCIENTIFIC_ARRAY_FORMAT)

def fin_diff_hvp(
        f: Callable,
        f_args: Tuple,
        f_kwargs: None | Dict,
        g: Callable,
        g_args: Tuple,
        g_kwargs: None | Dict,
        X: th.Tensor,
        v: th.Tensor,
        batch_indices: th.Tensor|None = None,
        is_g_contain_y: bool = False,
        require_grad: bool = False,
        normalize_v: bool = True,
        delta: float = 1e-2,
        eps: float = 1e-20,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Solve the Hessian vector product (for f = \nabla func) by finite difference.
    Args:
        f: the function
        f_args: positional arguments of f
        f_kwargs: keyword arguments of f
        g: gradient function
        g_args: positional arguments of g
        g_kwargs: keyword arguments of g
        X: the input tensor of f & g
        v: the output tensor of f & g
        batch_indices: the batched indices of X & v with format [0, 0, 0, ..., 1, 1, ..., n_batch - 1]
        is_g_contain_y: whether the grad function requires function value.
        require_grad: whether to require gradient.
        normalize_v: whether to normalize v to ||v|| = 1.
        delta: the step size for finite difference approximation
        eps: a small number to avoid division by zero

    Returns: y_mean, g_mean, hvp
        y_mean: th.Tensor, the function value of (y+ + y-) / 2
        g_mean: th.Tensor, the function value of (g+ + g-) / 2
        hvp: th.Tensor, the finite difference solution.

    """
    if f_kwargs is None:
        f_kwargs = dict()
    #n_batch, n_atom, n_dim = X.shape
    if batch_indices is None:
        v_norm = (th.sqrt(th.sum(v * v, dim=-1, keepdim=True)) + eps)
        v = v / v_norm  # (B, A, D)
        dX_f = X + delta * v
        dX_b = X - delta * v

    else:
        # X, v: (1, B0*A, D)
        v_norm = th.sqrt(th.sum(index_ops.index_inner_product(v, v, 1, batch_indices, None), dim=-1, keepdim=True)) + eps  # (1, B0, 1)
        v = v / v_norm.index_select(1, batch_indices)
        dX_f = X + delta * v
        dX_b = X - delta * v

    with th.set_grad_enabled(require_grad):
        dX_f.requires_grad_(require_grad)
        dX_b.requires_grad_(require_grad)
        if is_g_contain_y:
            y_f = f(dX_f, *f_args, **f_kwargs)
            g_f = g(dX_f, y_f, *g_args, **g_kwargs)
            y_b = f(dX_b, *f_args, **f_kwargs)
            g_b = g(dX_b, y_b, *g_args, **g_kwargs)
            hvp = (g_f - g_b) / (2 * delta)
        else:
            y_f = f(dX_f, *f_args, **f_kwargs)
            g_f = g(dX_f, *g_args, **g_kwargs)
            y_b = f(dX_b, *f_args, **f_kwargs)
            g_b = g(dX_b, *g_args, **g_kwargs)
            hvp = (g_f - g_b) / (2 * delta)
    y_mean = (y_f + y_b).detach() / 2.
    g_mean = (g_f + g_b).detach() / 2.
    if not normalize_v:
        hvp.mul_(v_norm.index_select(1, batch_indices))

    return y_mean, g_mean, hvp


class FindMinEigen(BaseMotion):
    """
    Find the eigenvector with minimum eigenvalue by Riemann gradient descent on S^2 manifold v^T v = 1.
    In fact, dimer only requires the direction within negative cone, i.e., v^T H v < 0.
    """
    def __init__(
            self,
            Torque_thres: float = 1e-2,
            Curve_thres: float = -0.1,
            maxiter_rot: int = 10,
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            _hold_samples: bool = False,
    ):
        """

        Args:
            Torque_thres: convergence threshold of torque.
            Curve_thres: convergence threshold of curvature, that v^T H v < `Curve_thres` is viewed as converged.
            maxiter_rot: maximum number of rotation iterations.
            dx: step size for finite difference approximation.
            device: the device on which the computation runs.
            verbose: the verbosity level.
            _hold_samples: whether to hold samples during optimization. if True, samples will not be removed even they have converged.
        """

        warnings.filterwarnings('always')
        self.Torque_thres = float(Torque_thres)
        self.Curve_thres = float(Curve_thres)
        if self.Curve_thres >= 0.:
            warnings.warn(f'`Curve_thres` should be less than 0 to determind the negative cone direction, but occurred {Curve_thres}.\n'
                          f'Now it has been set to its opposite number: {- Curve_thres}.')
            self.Curve_thres =  - Curve_thres
        assert (maxiter_rot > 0) and isinstance(maxiter_rot, int), '`maxiter_rot` must be an integer greater than 0.'
        self.maxiter_rot = int(maxiter_rot)
        self.dx = float(dx)
        self.subspace_hessian = None
        self.device = device
        self.verbose = verbose
        self._hold_samples = _hold_samples

        # logger
        super().__init__()
        self.init_logger('Main.TS.Eigen')

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
            method: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]]
    ) -> None:
        """
        Set a method to update the taget function when variables change.
        It receives a mask tensor of shape (n_batch, ) that only selects the `True` part to input to the function, and receives the old
        `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`,
        returns the corresponding masked new `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`.

        This method is used to dynamically 'remove' the samples which have been converged in a batch to avoid
        redundant calculation of converged samples.

        Default transform is identical transform (i.e., do nothing)
        Args:
            method: Callable(mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict) -> Tuple[Tuple, Dict, Tuple, Dict],
        the method of updating function arguments for a mask.

        Returns: None
        """
        self._update_batch = method
        self._hold_samples = False

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            v: th.Tensor,
            grad_func: Any | nn.Module = None,
            func_args: Tuple = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Find the eigenvector of Hessian at X with the minimum eigenvalue, by Riemannian gradient descent on S^n manifold v^T v = I.

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            v: Tensor[n_batch, n_atom, 3], the atom direction used to finite difference.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X
                i.e., grad = grad_func(X, y, *grad_func_args, **grad_func_kwargs), else grad = grad_func(X, *grad_func_args, **grad_func_kwargs)
            require_grad: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs) calculation.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is the same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]

        Return:
            v: the eigenvector with min eigenvalue at X (or at least the negative cone direction)
            y: the mean value of function at X, i.e., (f(X + delta * v) + f(X - delta * v))/2
            g: the mean grad of function at X, i.e., (grad(X + delta * v) + grad(X - delta * v))/2
            vHv: the curvature at X given by finite difference.
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
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            n_inner_batch = len(batch_indices)
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )
        else:
            raise NotImplementedError
            n_inner_batch = 1
            batch_indx_dict = dict()
            batch_tensor = None
        # initialize vars
        n_batch, n_atom, n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(grad_func, is_grad_func_contain_y, require_grad)

        if hasattr(self._update_batch, 'initialize'):
            self._update_batch.initialize()
        elif hasattr(self._update_batch, '__init__'):
            self._update_batch.__init__()
        # Selective dynamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'The shape of fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        # atom_masks = atom_masks.flatten(-2, -1).unsqueeze(-1)  # (n_batch, n_atom*n_dim, 1)
        # other check
        if (not isinstance(self.maxiter_rot, int)) or (self.maxiter_rot <= 0):
            raise ValueError(f'Invalid value of maxiter_rot: {self.maxiter_rot}. It would be an integer greater than 0.')

        # set variables device
        func = preload_func(func, self.device)

        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.detach()
        X = X.to(self.device)
        # normalize v
        v.mul_(atom_masks)
        v_norm = th.sqrt(th.sum(index_ops.index_inner_product(v, v, 1, self.batch_scatter), dim=-1, keepdim=True))
        v = v / v_norm.index_select(1, self.batch_scatter)

        # initialize
        ############################## BATCHED ALGORITHM ###################################
        # variables with '_' refer to the dynamically changed variables during iteration,
        # and they will in-placed copy into origin variables (i.e., without '_') at the end
        # of each iteration to update data.
        #
        ####################################################################################
        is_main_loop_converge = False
        t_st = time.perf_counter()
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
        # MAIN LOOP
        # X (1, n_batch * n_atom, n_dim)
        theta = th.zeros(n_inner_batch, device=self.device)
        func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = func_args, func_kwargs, grad_func_args, grad_func_kwargs
        y, g, Hv = fin_diff_hvp(
            func,
            func_args_,
            func_kwargs_,
            grad_func_,
            grad_func_args_,
            grad_func_kwargs_,
            X,
            v,
            self.batch_scatter,
            is_g_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            delta=self.dx
        )
        g.mul_(atom_masks)
        Hv.mul_(atom_masks)
        # curvature, vHv (1, B0, 1)
        vHv = th.sum(
            index_ops.index_inner_product(v, Hv, dim=1, batch_indices=self.batch_scatter),
            dim=-1,
            keepdim=True
        )
        # grad in the tangent space
        gT = Hv - vHv.index_select(1, self.batch_scatter) * v
        gT_norm = th.sqrt(th.sum(index_ops.index_inner_product(gT, gT, 1, self.batch_scatter), dim=-1, keepdim=True))
        w = v.clone()  #gT / (gT_norm.index_select(1, self.batch_scatter) + 1e-20)  # (1, sumB*A, N)
        # cache for dynamically changed batch indices due to convergence, avoiding reallocate mem.
        batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        for i in range(self.maxiter_rot):
            # threshold. Only need v in the negative cone, i.e., vHv < 0.
            converge_mask_curve = (vHv < self.Curve_thres)
            converge_mask_torque = (gT_norm < self.Torque_thres)
            #   reinsurance: When w and v are very collinear, stop meaningless update
            abort_mask = index_ops.index_inner_product(v, w, 1, self.batch_scatter) < 1.e-7
            converge_mask = (converge_mask_curve | converge_mask_torque)  # (1, B, 1)
            # print
            if self.verbose > 0:
                self.logger.info(f"Rot {i:>5d}\n "
                                 f"Torque:       {np.array2string(gT_norm.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Curvature:    {np.array2string(vHv.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Energies:     {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Theta (rad):  {np.array2string(theta.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Rot. Conv.:   {np.array2string(converge_mask.squeeze().numpy(force=True), **STRING_ARRAY_FORMAT)}\n "
                                 f"TIME:         {time.perf_counter() - t_st:>6.4f} s")
                t_st = time.perf_counter()
            # judge thres
            if th.all(converge_mask):
                is_main_loop_converge = True
                break
            elif th.all(converge_mask | abort_mask):
                break
            converge_mask_short = converge_mask | abort_mask
            converge_mask = converge_mask_short[:, self.batch_scatter, ...]  # (1, sumB*A, 1)
            # update batch, remove the already converged ones.
            if not self._hold_samples:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_mask_short.squeeze(),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_mask[0, :, 0])  # (sumB*A, )
                select_mask_short = ~converge_mask_short[0, :, 0]  # (B, )
                #y_ = y[select_mask_short]
                vHv_ = vHv[:, select_mask_short, :]
                Hv_ = Hv[:, select_mask, :]
                X_ = X[:, select_mask, :]
                v_ = v[:, select_mask, :]
                gT_ = gT[:, select_mask, :]
                gT_norm_ = gT_norm[:, select_mask_short, :]
                atom_masks_ = atom_masks[:, select_mask, :]
                batch_tensor_ = self.batch_tensor[select_mask_short]
                batch_scatter_ = th.repeat_interleave(
                    batch_tensor_indx_cache[:len(batch_tensor_)],
                    batch_tensor_,
                    dim=0
                )
            else:
                select_mask = None
                select_mask_short = None
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = func_args, func_kwargs, grad_func_args, grad_func_kwargs
                vHv_ = vHv
                Hv_ = Hv
                X_ = X
                v_ = v
                gT_ = gT
                gT_norm_ = gT_norm
                atom_masks_ = atom_masks
                batch_tensor_ = self.batch_tensor
                batch_scatter_ = self.batch_scatter

            # construction subspace Hessian [[vHv vHw] [wHv wHw]] with shape (B0, 2, 2) for 2nd order precise linear search
            w_ = gT_ / (gT_norm_.index_select(1, batch_scatter_) + 1e-20)  # (1, sumB*A, N)
            self.logger.debug(f"w:\n{w_}")
            y2_t, g2_, Hw_ = fin_diff_hvp(
                func,
                func_args_,
                func_kwargs_,
                grad_func_,
                grad_func_args_,
                grad_func_kwargs_,
                X_,
                w_,
                batch_scatter_,
                is_g_contain_y=is_grad_func_contain_y,
                require_grad=require_grad,
                delta=self.dx
            )
            g2_.mul_(atom_masks_)
            Hw_.mul_(atom_masks_)  # mask
            # subspace Hessian
            vHw_ = th.sum(
                index_ops.index_inner_product(v_, Hw_, dim=1, batch_indices=batch_scatter_),
                dim=-1,
                keepdim=True
            )
            wHv_ = th.sum(
                index_ops.index_inner_product(w_, Hv_, dim=1, batch_indices=batch_scatter_),
                dim=-1,
                keepdim=True
            )
            wHw_ = th.sum(
                index_ops.index_inner_product(w_, Hw_, dim=1, batch_indices=batch_scatter_),
                dim=-1,
                keepdim=True
            )
            nondiag_ = 0.5 * (wHv_ + vHw_)  # (1, B, 1)
            H22_ = th.cat((vHv_, nondiag_, nondiag_, wHw_), dim=-1).reshape(-1, 2, 2)
            self.logger.debug(f"H22:\n{H22_.numpy(force=True)}")

            sub_eigval_, sub_eigvec_ = th.linalg.eigh(H22_)  # (B, 2), (B, 2, 2)
            cos_t_ = sub_eigvec_[None, :, 0:1, 0].index_select(1, batch_scatter_)
            sin_t_ = sub_eigvec_[None, :, 1:2, 0].index_select(1, batch_scatter_)  # (1, sumB*A, 1)
            theta_ = th.atan2(sub_eigvec_[:, 0, 0], sub_eigvec_[:, 0, 1])  # (B, )
            # update
            v_.mul_(cos_t_)
            v_.add_(w_ * sin_t_)
            v_.mul_(atom_masks_)
            Hv_.mul_(cos_t_)
            Hv_.add_(Hw_ * sin_t_)
            Hv_.mul_(atom_masks_)
            # curvature, vHv (1, B0, 1)
            vHv_ = th.sum(
                index_ops.index_inner_product(v_, Hv_, dim=1, batch_indices=batch_scatter_),
                dim=-1,
                keepdim=True
            )
            # grad in the tangent space
            gT_ = Hv_ - vHv_.index_select(1, batch_scatter_) * v_
            gT_norm_ = th.sqrt(th.sum(index_ops.index_inner_product(gT_, gT_, 1, batch_scatter_), dim=-1, keepdim=True))

            # update origin variables
            if not self._hold_samples:
                select_indices = th.where(select_mask)[0]
                select_indices_short = th.where(select_mask_short)[0]
                y.index_copy_(0, select_indices_short, y2_t)
                v.index_copy_(1, select_indices, v_)
                w.index_copy_(1, select_indices, w_)
                #X.index_copy_(1, select_indices, X_)
                Hv.index_copy_(1, select_indices, Hv_)
                vHv.index_copy_(1, select_indices_short, vHv_)
                gT_norm.index_copy_(1, select_indices_short, gT_norm_)
                g.index_copy_(1, select_indices, g2_)
                theta.index_copy_(0, select_indices_short, theta_)  # (B, )
                #atom_masks.index_copy_(1, select_indices, atom_masks_)
            else:
                y = y2_t
                v = v_
                w = w_
                #X = X_
                Hv = Hv_
                vHv = vHv_
                gT_norm = gT_norm_
                g = g2_
                theta = theta_
            pass

        if self.verbose and is_main_loop_converge:
            self.logger.info(
                '-' * 100 + f'\nrotation done. time: {time.perf_counter() - t_main:<.4f} s\n'
            )
        else:
            self.logger.warning(
                '-' * 100 + f'\nWARNING: Some Structures\' Rotation were NOT Converged yet!\n'
                            f'rotation done. time: {time.perf_counter() - t_main:<.4f} s\n'
            )

        # recalc y, g, Hv (Optional)
        #y, g, Hv = fin_diff_hvp(
        #    func,
        #    func_args,
        #    func_kwargs,
        #    grad_func_,
        #    grad_func_args,
        #    grad_func_kwargs,
        #    X,
        #    v,
        #    self.batch_scatter,
        #    is_g_contain_y=is_grad_func_contain_y,
        #    require_grad=require_grad,
        #)
        Hv.mul_(atom_masks)
        g.mul_(atom_masks)
        # curvature, vHv (1, B0, 1)
        vHv = th.sum(
            index_ops.index_inner_product(v, Hv, dim=1, batch_indices=self.batch_scatter),
            dim=-1,
            keepdim=True
        )

        return v, y, g, Hv, vHv


class Dimer(BaseMotion):
    """
    Modified Dimer
    Ref. J Chem Phys 2005, 132, 224101.

    """

    def __init__(
            self,
            E_threshold: float = 1e-3,
            Torque_thres: float = 1e-2,
            Curvature_thres: float = - 0.1,
            F_threshold: float = 0.05,
            maxiter_trans: int = 100,
            maxiter_rot: int = 10,
            max_steplength: float = 0.5,
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        warnings.filterwarnings('always')
        self.E_threshold = float(E_threshold)
        self.Torque_thres = float(Torque_thres)
        self.Curvature_thres = float(Curvature_thres)
        self.F_threshold = float(F_threshold)
        self.maxiter_trans = int(maxiter_trans)
        assert (maxiter_rot > 0) and isinstance(maxiter_rot, int), '`maxiter_rot` must be an integer greater than 0.'
        self.maxiter_rot = int(maxiter_rot)
        self.max_steplength = float(max_steplength)
        self.dx = float(dx)
        self.device = device
        self.verbose = verbose

        self.Rotator = FindMinEigen(
            self.Torque_thres,
            self.Curvature_thres,
            self.maxiter_rot,
            self.dx,
            self.device,
            self.verbose,
            _hold_samples=True
        )

        # logger
        super().__init__()
        self.init_logger('Main.TS')

    def _update_batch(self, mask: th.Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict):
        """
        Default update method for the input of func if the func has non-opt variables, i.e., the identical transform.
        Args:
            mask:

        Returns:

        """
        return func_args, func_kwargs, grad_func_args, grad_func_kwargs

    def _linear_search(self, dX_, ):
        """
        Linear search for dimer translation
        Returns:

        """
        # TODO, adding linear search algo. to determine steplength.
        pass

    def set_batch_updater(
            self,
            method_trans: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]],
            method_rot: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]] | None = None,
    ) -> None:
        """
        Set a method to update the taget function when variables change.
        It receives a mask tensor of shape (n_batch, ) that only selects the `True` part to input to the function, and receives the old
        `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`,
        returns the corresponding masked new `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`.

        This method is used to dynamically 'remove' the samples which have been converged in a batch to avoid
        redundant calculation of converged samples.

        Default transform is identical transform (i.e., do nothing)
        Args:
            method_trans: Callable(mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict) -> Tuple[Tuple, Dict, Tuple, Dict],
            method_rot: batch updater for rotations. the method of updating function arguments for a mask.

        Returns: None
        """
        if method_rot is not None:
            self.Rotator.set_batch_updater(method_rot)
        else:
            self.Rotator._hold_samples = True
        self._update_batch = method_trans

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            X_diff: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: Optional[th.Tensor | List] = None,
    ):
        """
        run Dimer algo.
        Args:
            func:
            X:
            X_diff:
            grad_func:
            func_args:
            func_kwargs:
            grad_func_args:
            grad_func_kwargs:
            is_grad_func_contain_y:
            require_grad:
            output_grad:
            fixed_atom_tensor:
            batch_indices:

        Returns:

        """
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        if func_kwargs is None:
            func_kwargs = dict()
        func_args = tuple(func_args)
        grad_func_args = tuple(grad_func_args)
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            if len(X.shape) == 2:
                X = X.unsqueeze(0)
            elif len(X.shape) != 3:
                raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        if X_diff is None:
            X_diff = th.randn_like(X)

        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(grad_func, is_grad_func_contain_y, require_grad)
        if isinstance(X, th.Tensor):
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        # batch_check
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (Tuple, List)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be Sequence[int] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            n_inner_batch = len(batch_indices)
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )
        else:  # temporarily not implemented
            raise NotImplementedError(f'Regular batch algo. is not implemented yet. You may specify a uniform `batch_indices` instead.')
            n_inner_batch = 1
            batch_indx_dict = dict()
            batch_tensor = None

        # Selective dynamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)
        # other check
        if (not isinstance(self.maxiter_trans, int)) or (not isinstance(self.maxiter_rot, int)) \
                or (self.maxiter_trans <= 0) or (self.maxiter_rot <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.maxiter_trans}. It would be an integer greater than 0.')

        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)
        X_diff = X_diff.to(self.device)
        v = X_diff.mul(atom_masks)
        plist = list()  # TEST <<<<
        is_main_loop_converge = False
        # Main Loop
        # initialize
        #y, g, Hv = fin_diff_hvp(
        #    func,
        #    func_args,
        #    func_kwargs,
        #    grad_func_,
        #    grad_func_args,
        #    grad_func_kwargs,
        #    X,
        #    v,
        #    self.batch_scatter,
        #    is_g_contain_y=is_grad_func_contain_y,
        #    require_grad=require_grad,
        #)
        #g.mul_(atom_masks)
        #Hv.mul_(atom_masks)
        #vHv = th.sum(
        #    index_ops.index_inner_product(v, Hv, dim=1, batch_indices=self.batch_scatter),
        #    dim=-1,
        #    keepdim=True
        #)
        v, y, g, Hv, vHv = self.Rotator.run(
            func=func,
            X=X,
            v=v,
            grad_func=grad_func_,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            fixed_atom_tensor=atom_masks,
            batch_indices=self.batch_tensor
        )
        y_old = th.full_like(y, th.inf, device=self.device)
        # NOW hard coded
        eig_thres_neg = - 0.1
        eig_thres_pos = 0.1
        batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        t_st = time.perf_counter()
        with th.no_grad():
            for i in range(self.maxiter_trans):
                #plist.append(X[:, None, :, 0].numpy(force=True))  # TEST <<<<<<<<<<<<<
                #plist.append(X2[:, None, :, 0].numpy(force=True))
                # Section: check threshold  <<<
                # threshold.
                converge_mask_curve = (vHv < 0.)
                F_eps = index_ops.index_reduce(
                    th.max(th.abs(g), dim=-1, keepdim=True).values,
                    self.batch_scatter, 1, 'amax', -th.inf
                )
                E_eps = th.abs(y - y_old)
                converge_mask_g = (F_eps < self.F_threshold)
                converge_mask_e = th.lt(E_eps, self.E_threshold).reshape(1, -1, 1)
                converge_mask = converge_mask_curve & converge_mask_g & converge_mask_e  # (1, B, 1)
                y_old = y.clone()
                # print
                if self.verbose > 0:
                    self.logger.info(f"Translation {i:>5d}\n "
                                     f"MAD_Energies: {np.array2string(E_eps.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"MAX_F:        {np.array2string(F_eps.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Curvature:    {np.array2string(vHv.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Energies:     {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Converged:    {np.array2string(converge_mask.squeeze().numpy(force=True), **STRING_ARRAY_FORMAT)}\n "
                                     f"TIME:         {time.perf_counter() - t_st:>6.4f} s")
                    t_st = time.perf_counter()
                # OUTPUT COORD
                if self.verbose > 1:
                    # split batches if specified batch
                    if batch_indices is not None:
                        X_np = X.numpy(force=True)
                        X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                        if self.verbose > 2:
                            F_np = (- g).numpy(force=True)
                            F_tup = np.split(F_np, batch_slice_indx[1:-1], axis=1)
                    else:
                        X_tup = (X.numpy(force=True),)
                        if self.verbose > 2:
                            F_tup = (- g.numpy(force=True),)
                    self.logger.info(f" Coordinates:\n")
                    X_str = [
                        np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        for xi in X_tup
                    ]
                    for x_str in X_str: self.logger.info(f'{x_str}\n')
                if self.verbose > 2:
                    self.logger.info(f" Forces:\n")
                    X_str = [
                        np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        for xi in F_tup
                    ]
                    for x_str in X_str: self.logger.info(f'{x_str}\n')
                # judge thres
                if th.all(converge_mask):
                    is_main_loop_converge = True
                    break
                converge_mask_short = converge_mask
                converge_mask = converge_mask[:, self.batch_scatter, ...]  # (1, sumB*A, 1)
                # update batch, remove the already converged ones.
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_mask_short.squeeze(),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_mask[0, :, 0])  # (sumB*A, )
                select_mask_short = ~converge_mask_short[0, :, 0]  # (B, )
                X_ = X[:, select_mask, :]
                g_ = g[:, select_mask, :]
                v_ = v[:, select_mask, :]
                Hv_ = Hv[:, select_mask, :]
                atom_masks_ = atom_masks[:, select_mask, :]
                batch_tensor_ = self.batch_tensor[select_mask_short]
                batch_scatter_ = th.repeat_interleave(
                    batch_tensor_indx_cache[:len(batch_tensor_)],
                    batch_tensor_,
                    dim=0
                )

                # Section: Transition to maximum  <<<
                # 2D subspace Newton Search
                vg_ = th.sum(index_ops.index_inner_product(v_, g_, dim=1, batch_indices=batch_scatter_), dim=-1, keepdim=True)  # (1, B0, 1)
                normal_grad_ = vg_.index_select(1, batch_scatter_) * v_  # (1, sumB*A, N)
                tangent_grad_ = g_ - normal_grad_
                tangent_norm_ = th.sqrt(
                    th.sum(index_ops.index_inner_product(tangent_grad_, tangent_grad_, 1, batch_scatter_), dim=-1, keepdim=True)
                )
                u_ = tangent_grad_ / (tangent_norm_.index_select(1, batch_scatter_) + 1e-20)
                ug_ = th.sum(index_ops.index_inner_product(u_, g_, dim=1, batch_indices=batch_scatter_), dim=-1, keepdim=True)  # (1, B0, 1)
                _, _, Hu_ = fin_diff_hvp(
                    func,
                    func_args_,
                    func_kwargs_,
                    grad_func_,
                    grad_func_args_,
                    grad_func_kwargs_,
                    X_,
                    u_,
                    batch_scatter_,
                    is_g_contain_y=is_grad_func_contain_y,
                    require_grad=require_grad,
                )
                Hv_.mul_(atom_masks_)
                Hu_.mul_(atom_masks_)
                # curvature, vHv (1, B0, 1)
                vHv_ = th.sum(
                    index_ops.index_inner_product(v_, Hv_, dim=1, batch_indices=batch_scatter_),
                    dim=-1,
                    keepdim=True
                )
                vHu_ = th.sum(
                    index_ops.index_inner_product(v_, Hu_, dim=1, batch_indices=batch_scatter_),
                    dim=-1,
                    keepdim=True
                )
                uHv_ = th.sum(
                    index_ops.index_inner_product(u_, Hv_, dim=1, batch_indices=batch_scatter_),
                    dim=-1,
                    keepdim=True
                )
                uHu_ = th.sum(
                    index_ops.index_inner_product(u_, Hu_, dim=1, batch_indices=batch_scatter_),
                    dim=-1,
                    keepdim=True
                )
                nondiag_ = 0.5 * (uHv_ + vHu_)  # (1, B, 1)
                H22_ = th.cat((vHv_, nondiag_, nondiag_, uHu_), dim=-1).reshape(-1, 2, 2)
                Lambda, U = th.linalg.eigh(H22_)  # (B, 2), (B, 2, 2)
                # modify spectra
                Lambda[:, 0] = th.where(Lambda[:, 0] >= 0., - Lambda[:, 0], Lambda[:, 0])
                Lambda[:, 0] = th.where(th.gt(Lambda[:, 0], eig_thres_neg), eig_thres_neg, Lambda[:, 0])
                Lambda[:, 1] = th.where(th.le(Lambda[:, 1], eig_thres_pos), eig_thres_pos, Lambda[:, 1])
                # subspace Newton step
                sub_g_ = th.cat((vg_, ug_), dim=-1).squeeze(0)
                # (B, 2, 2) @ (B, 2, 2) @ (B, 2, 1) -> (B, 2)
                p_ = th.einsum('bij, bi -> bj', - U @ (th.reciprocal(Lambda.unsqueeze(-1)) * U.mT), sub_g_)
                dX_ = p_[batch_scatter_, 0:1] * v_ + p_[batch_scatter_, 1:] * u_  # (sumB * A, D)
                dX_norm_ = th.sqrt(th.sum(index_ops.index_inner_product(dX_, dX_, 1, batch_scatter_), dim=-1, keepdim=True))
                dX_norm_scatter_ = dX_norm_.index_select(1, batch_scatter_)
                dX_ = th.where(
                    dX_norm_scatter_ > self.max_steplength,
                    self.max_steplength / dX_norm_scatter_ * dX_,
                    dX_
                )
                # update X
                dX_.mul_(atom_masks_)
                X_.add_(dX_)
                # traditional scheme
                # dx = th.where(
                #    vHv_ > 0.,
                #    - normal_grad,
                #    tangent_grad - normal_grad,
                # )  # (1, sumB*A, N)
                _small_dX_ = (dX_norm_ < 1.e-6)  # (1, B0, 1)
                if th.any(_small_dX_):
                    _too_small_step_indx = th.where(select_mask_short)[0][th.where(_small_dX_)[0]]
                    warnings.warn(
                        RuntimeWarning(
                        f'Convergence is not met while the steplengths of {_too_small_step_indx.tolist()}-th structure(s) are 0. '
                        )
                    )
                    if th.all(_small_dX_):
                        self.logger.error('ERROR: All unconverged structures reached 0 steplength. LOOP BREAK.')
                        break


                # Section: Rotate to minimum  <<<
                v_, y_, g_, Hv_, vHv_ = self.Rotator.run(
                    func=func,
                    X=X_,
                    v=v_,
                    grad_func=grad_func_,
                    func_args=func_args_,
                    func_kwargs=func_kwargs_,
                    grad_func_args=grad_func_args_,
                    grad_func_kwargs=grad_func_kwargs_,
                    is_grad_func_contain_y=is_grad_func_contain_y,
                    require_grad=require_grad,
                    fixed_atom_tensor=atom_masks_,
                    batch_indices=batch_tensor_
                )

                # update origin variables
                select_indices = th.where(select_mask)[0]
                select_indices_short = th.where(select_mask_short)[0]
                y.index_copy_(0, select_indices_short, y_)
                v.index_copy_(1, select_indices, v_)
                X.index_copy_(1, select_indices, X_)
                Hv.index_copy_(1, select_indices, Hv_)
                vHv.index_copy_(1, select_indices_short, vHv_)
                g.index_copy_(1, select_indices, g_)

        if self.verbose:
            if is_main_loop_converge:
                self.logger.info('-' * 100 + '\nAll Structures Were Converged.\nMAIN LOOP Done.')
            else:
                self.logger.info('-' * 100 + '\nSome Structures were NOT Converged yet!\nMAIN LOOP Done.')

        if output_grad:
            return y, X, g
        else:
            return y, X#, plist  # TEST <<<<<<

