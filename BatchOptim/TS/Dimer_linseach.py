"""
Dimer Algo. for transition state searching. J. Chem. Phys. 1999, 111: 7010.
While the steplength determination was revised into line search strategy.
"""

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Dimer_linseach.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import time
import warnings
import logging
import sys

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as Fn

from BM4Ckit.utils._print_formatter import GLOBAL_SCIENTIFIC_ARRAY_FORMAT

np.set_printoptions(**GLOBAL_SCIENTIFIC_ARRAY_FORMAT)


class Dimer:
    """
    Ref. J Chem Phys 2005, 132, 224101.

    """

    def __init__(
            self,
            E_threshold: float = 1e-3,
            Torque_thres: float = 1e-2,
            F_threshold: float = 0.05,
            maxiter_trans: int = 100,
            maxiter_rot: int = 10,
            maxiter_linsearch: int = 10,
            steplength: float|th.Tensor = 0.5,
            max_steplength: float = 0.5,
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):

        warnings.filterwarnings('always')
        self.E_threshold = E_threshold
        self.Torque_thres = Torque_thres
        self.F_threshold = F_threshold
        self.maxiter_trans = maxiter_trans
        assert (maxiter_rot > 1) and isinstance(maxiter_rot, int), '`maxiter_rot` must be an integer greater than 1.'
        self.maxiter_rot = maxiter_rot
        self.steplength = steplength
        self.maxiter_linsearch = maxiter_linsearch
        self.max_steplength = max_steplength
        self.dx = dx
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
        pass

    def _rotate(self, ):
        pass

    def _transition(self, ):
        pass

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            X_diff: th.Tensor,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            is_grad_func_contain_y: bool = True,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
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
            output_grad:
            fixed_atom_tensor:

        Returns:

        """

        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        if func_kwargs is None:
            func_kwargs = dict()

        t_main = time.perf_counter()
        n_batch, n_atom, n_dim = X.shape
        if grad_func is None:
            is_grad_func_contain_y = True

            def grad_func_(y, x, grad_shape=None):
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
        atom_masks_ = atom_masks.flatten(-2, -1)  # (n_batch, n_atom*n_dim)
        # other check
        if (not isinstance(self.maxiter_trans, int)) or (not isinstance(self.maxiter_rot, int)) \
                or (self.maxiter_trans <= 0) or (self.maxiter_rot <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.maxiter_trans}. It would be an integer greater than 0.')
        if isinstance(self.steplength, float):
            self.steplength = th.full((n_batch, 1, 1), self.steplength, device=self.device, dtype=X.dtype)
        elif isinstance(self.steplength, th.Tensor):
            if self.steplength.shape != (n_batch, 1, 1):
                if self.steplength.shape == (n_batch, ):
                    self.steplength = self.steplength.unsqueeze(-1).unsqueeze(-1)
                else:
                    raise ValueError(f'Expected self.steplength has shape of {(n_batch, 1, 1)} or {(n_batch, )}, but got {self.steplength.shape}.')

        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)
        X_diff = X_diff.to(self.device)
        for _args in (func_args, func_kwargs.values(), grad_func_args, grad_func_kwargs.values()):
            for _arg in _args:
                if isinstance(_arg, th.Tensor):
                    _arg.to(self.device)

        #plist = list()  # TEST <<<<
        is_main_loop_converge = False
        # Main Loop
        X1 = X + X_diff  # (n_batch, n_atom, n_dim)
        X2 = X
        X1_ = X1.flatten(-2, -1)  # (n_batch, n_atom*n_dim). NOTE: '_' means the flatten variables.
        X2_ = X2.flatten(-2, -1)
        with th.no_grad():
            for i in range(self.maxiter_trans):
                t_st = time.perf_counter()
                #plist.append(X1[:, None, :, 0].numpy(force=True))  # TEST <<<<<<<<<<<<<
                #plist.append(X2[:, None, :, 0].numpy(force=True))
                # Section: Rotate to minimum
                dth = th.tensor([1e-3, ], device=self.device)
                is_rotate_loop_converge = False
                if self.verbose > 1: print('-' * 100)
                # section ROTATION
                for j in range(self.maxiter_rot):
                    # original X
                    X_diff_ = X_diff.flatten(-2, -1)
                    Xc_ = (X1_ + X2_) / 2.
                    NI_ = Fn.normalize(X_diff_, dim=-1)  # (n_batch, n_atom*n_dim), the element vectors.
                    dX_ = 0.5 * (X_diff_.unsqueeze(1) @ NI_.unsqueeze(-1))  # (n_batch, 1, n_atom*n_dim)@(n_batch, n_atom*n_dim, 1)
                    dX_.squeeze_(-1)  # (n_batch, 1)

                    # Calc. Force
                    with th.enable_grad():
                        X1.requires_grad_()
                        X2.requires_grad_()
                        y1 = func(X1, *func_args, **func_kwargs)
                        if is_grad_func_contain_y:
                            F1_ = - grad_func_(y1, X1, *grad_func_args, **grad_func_kwargs) * atom_masks
                        else:
                            F1_ = - grad_func_(X1, *grad_func_args, **grad_func_kwargs) * atom_masks
                        y2 = func(X2, *func_args, **func_kwargs)
                        if is_grad_func_contain_y:
                            F2_ = - grad_func_(y2, X2, *grad_func_args, **grad_func_kwargs) * atom_masks
                        else:
                            F2_ = - grad_func_(X2, *grad_func_args, **grad_func_kwargs) * atom_masks
                    ener = y1 + y2 / 2.
                    del y1, y2
                    F1_ = F1_.flatten(-2, -1).detach()
                    F2_ = F2_.flatten(-2, -1).detach()
                    F_tol_ = (F1_ + F2_) / 2.
                    Fc_ = (F1_ - F2_)
                    F1_vert_ = F1_ - (F1_.unsqueeze(1) @ NI_.unsqueeze(-1)).squeeze(-1) * NI_
                    F2_vert_ = F2_ - (F2_.unsqueeze(1) @ NI_.unsqueeze(-1)).squeeze(-1) * NI_
                    Fc_vert_ = F1_vert_ - F2_vert_
                    TI_ = Fn.normalize(Fc_vert_, dim=-1)

                    # Judge
                    F_max: th.Tensor = th.max(th.abs(F_tol_), dim=-1)[0]  # (n_batch, )
                    Curv = ((F2_ - F1_).unsqueeze(1) @ NI_.unsqueeze(-1)).squeeze(-1) / (2 * dX_)  # curvature
                    Torque = th.linalg.norm(Fc_vert_, dim=-1)
                    if self.verbose > 1:
                        print(f'Iteration:   {i + 1}\n'
                              f'Rotation:    {j + 1}\n'
                              f'Energies:    {ener.numpy(force=True)}\n'
                              f'Curvature:   {Curv.squeeze().numpy(force=True)}\n'
                              f'Torque:      {Torque.numpy(force=True)}\n'
                              f'Max Force:   {F_max.numpy(force=True)}')
                    converge_mask = (F_max < self.F_threshold) * (Curv.flatten(-2, -1) <= 0.)
                    if self.verbose > 1: print(f'Converged:   {converge_mask.numpy(force=True)}')
                    if th.all(converge_mask):  # judge transition state
                        is_main_loop_converge = True
                        break
                    if th.all(Torque < self.Torque_thres):  # judge rot
                        is_rotate_loop_converge = True
                        break

                    # finite difference. NOTE: _d means the vars after finite displacement.
                    X1_d_ = Xc_ + (NI_ * th.cos(dth) + TI_ * th.sin(dth)) * dX_
                    X2_d_ = Xc_ - (NI_ * th.cos(dth) + TI_ * th.sin(dth)) * dX_
                    X1_d = X1_d_.reshape(n_batch, n_atom, n_dim)
                    X2_d = X2_d_.reshape(n_batch, n_atom, n_dim)
                    NI_d_ = Fn.normalize(X1_d_ - X2_d_, dim=-1)
                    with th.enable_grad():
                        X1_d.requires_grad_()
                        X2_d.requires_grad_()
                        y1_d = func(X1_d, *func_args, **func_kwargs)
                        y2_d = func(X2_d, *func_args, **func_kwargs)
                        if is_grad_func_contain_y:
                            F1_d = - grad_func_(y1_d, X1_d, *grad_func_args, **grad_func_kwargs) * atom_masks
                            F2_d = - grad_func_(y2_d, X2_d, *grad_func_args, **grad_func_kwargs) * atom_masks
                        else:
                            F1_d = - grad_func_(X1_d, *grad_func_args, **grad_func_kwargs) * atom_masks
                            F2_d = - grad_func_(X2_d, *grad_func_args, **grad_func_kwargs) * atom_masks
                    del y1_d, y2_d
                    F1_d_ = F1_d.flatten(-2, -1).detach()
                    F2_d_ = F2_d.flatten(-2, -1).detach()
                    Fc_d_ = F1_d_ - F2_d_
                    F1_d_vert_ = F1_d_ - (F1_d_.unsqueeze(1) @ NI_d_.unsqueeze(-1)).squeeze(-1) * NI_d_
                    F2_d_vert_ = F2_d_ - (F2_d_.unsqueeze(1) @ NI_d_.unsqueeze(-1)).squeeze(-1) * NI_d_
                    Fc_d_vert_ = F1_d_vert_ - F2_d_vert_
                    TI_d_ = Fn.normalize(Fc_d_vert_, dim=-1)
                    # rotate
                    F_half_d_ = ((Fc_d_.unsqueeze(1) @ TI_d_.unsqueeze(-1) + Fc_.unsqueeze(1) @ TI_.unsqueeze(-1)) / 2.).squeeze(-1)
                    dF_half_d_ = ((Fc_d_.unsqueeze(1) @ TI_d_.unsqueeze(-1) - Fc_.unsqueeze(1) @ TI_.unsqueeze(-1)) / dth).squeeze(-1)
                    rot_theta = -0.5 * th.arctan((2. * F_half_d_) / dF_half_d_) - dth / 2.

                    # update X
                    X1_ = Xc_ + (NI_ * th.cos(rot_theta) + TI_ * th.sin(rot_theta)) * dX_
                    X2_ = Xc_ - (NI_ * th.cos(rot_theta) + TI_ * th.sin(rot_theta)) * dX_
                    X1 = X1_.reshape(n_batch, n_atom, n_dim)
                    X2 = X2_.reshape(n_batch, n_atom, n_dim)
                    X_diff = X1 - X2
                    #plist.append(X1[:, None, :, 0].numpy(force=True))  # TEST <<<<<<<<<<<<<<<<<<<<<<<<
                    #plist.append(X2[:, None, :, 0].numpy(force=True))

                if is_main_loop_converge: break
                if not is_rotate_loop_converge: warnings.warn('Rotation did not converge!', RuntimeWarning)
                # Section: Transition to maximum  <<<
                # 1D Newton Search
                Fc_hori_ = th.where(Curv > 0.,
                                    - ((F_tol_.unsqueeze(1) @ NI_.unsqueeze(-1)).squeeze(-1) * NI_),
                                    (F_tol_ - 2 * (F_tol_.unsqueeze(1) @ NI_.unsqueeze(-1)).squeeze(-1)) * NI_)  # (n_batch, n_atom * n_dim)
                direct_ = Fn.normalize(Fc_hori_, dim=-1)
                # line search
                rho = 0.05
                beta = 0.6
                is_step_converge = False
                steplength = self.steplength
                with th.enable_grad():
                    Xc0 = Xc_.reshape(n_batch, n_atom, n_dim)
                    Xc0.requires_grad_()
                    yc = func(Xc0, *func_args, **func_kwargs)
                    if is_grad_func_contain_y:
                        Fc_tmp = - grad_func_(yc, Xc0, *grad_func_args, **grad_func_kwargs) * atom_masks
                    else:
                        Fc_tmp = - grad_func_(Xc0, *grad_func_args, **grad_func_kwargs) * atom_masks
                    # (n_batch, 1, n_atom * n_dim) @ (n_batch, n_atom * n_dim, 1) -> (n_batch, 1, 1)
                    direct_grad0 = Fc_tmp.flatten(-2, -1).unsqueeze(-1).mT @ direct_.unsqueeze(-1)
                for i_ls in range(self.maxiter_linsearch):
                    with th.enable_grad():
                        Xc_tmp = Xc0 + steplength * direct_.reshape(n_batch, n_atom, n_dim)
                        Xc_tmp.requires_grad_()
                        yc_tmp = func(Xc_tmp, *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)
                        if is_grad_func_contain_y:
                            Fc_tmp = - grad_func_(yc_tmp, Xc_tmp, *grad_func_args, **grad_func_kwargs) * atom_masks
                        else:
                            Fc_tmp = - grad_func_(Xc_tmp, *grad_func_args, **grad_func_kwargs) * atom_masks
                        # (n_batch, 1, n_atom * n_dim) @ (n_batch, n_atom * n_dim, 1) -> (n_batch, 1, 1)
                        direct_grad1 = Fc_tmp.flatten(-2, -1).unsqueeze(-1).mT @ direct_.unsqueeze(-1)

                    # note: Linear Search Strategy:
                    #   for positive curvature, i.e. the convex region, using inverse Armijo cond. to get a sufficient increase;
                    #   for negative curvature, using curvature cond. in weak Powell-Wolfe cond. to reduce gradient.
                    energy_rise_mask = yc_tmp >= (yc.unsqueeze(-1).unsqueeze(-1) + rho * steplength * direct_grad0)  # (n_batch, 1, 1)
                    grad_descent_mask = direct_grad1 <= beta * direct_grad0
                    stepl_converge_mask = th.where(
                        Curv > 0.,
                        energy_rise_mask,
                        grad_descent_mask
                    )
                    if th.all(stepl_converge_mask):
                        is_step_converge = True
                        break
                    steplength = (steplength - (steplength - 1e-4) * ((i_ls + 1) / self.maxiter_linsearch) ** 1.5)
                if not is_step_converge:
                    steplength = th.where(steplength > self.max_steplength, self.max_steplength, steplength)

                if self.verbose > 1: print(f'step length: {steplength.flatten().numpy(force=True)}\n')
                if th.any(th.abs(steplength)) < 1.e-5:
                    warnings.warn(RuntimeWarning('Saddle point does not met while steplength is 0 in some structures. '))
                    break
                # update X
                X1_ = X1_ + steplength * direct_
                X2_ = X2_ + steplength * direct_
                X1 = X1_.reshape(n_batch, n_atom, n_dim)
                X2 = X2_.reshape(n_batch, n_atom, n_dim)
                X_diff = X1 - X2

        Xc_ = (X1_ + X2_) / 2.
        Xc = Xc_.reshape(n_batch, n_atom, n_dim)
        energies = func(Xc, *func_args, **func_kwargs)
        if self.verbose:
            if is_main_loop_converge:
                print('-' * 100 + '\nAll Structures Were Converged.\nMAIN LOOP Done.')
            else:
                print('-' * 100 + '\nSome Structures were NOT Converged yet!\nMAIN LOOP Done.')

        return energies, Xc, #plist  # TEST <<<<<<

        pass
