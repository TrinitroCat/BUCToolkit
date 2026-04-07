"""
FIRE Optimization Algorithm. Phys. Rev. Lett., 2006, 97:170201.
"""
import copy
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: FIRE.py
#  Environment: Python 3.12

import os
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import torch as th
from torch import nn

from BUCToolkit.utils._Element_info import MASS, N_MASS
from .._BaseOpt import _BaseOpt
from BUCToolkit.utils.index_ops import index_inner_product

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class FIRE(_BaseOpt):
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
            _hold_samples: bool = False,
            **kwargs
    ):
        r"""
        FIRE Algorithm for optimization.

        Args:
            E_threshold: float, threshold of difference of func between 2 iterations.
            F_threshold: float, threshold of gradient of func.
            maxiter: int, the maximum iteration steps.
            steplength: The initial step length, i.e. the BatchMD time step.
            alpha: FIRE arg. alpha.
            alpha_fac: FIRE arg. the factor of alpha change.
            fac_inc: FIRE arg. the increment factor.
            fac_dec: FIRE arg. the decrement factor.
            N_min: FIRE arg. the minimum number that keeps steplength static.
            device: The device that program runs on.
            verbose: amount of print information.
            _hold_samples: ONLY FOR SPECIAL USE (e.g., CI-NEB or DEBUG).
                If True, FIRE optimizer will not remove any sample in a batch even if the sample has converged.

        Method:
            run: running the main optimization program.
        """
        super().__init__(
            'FIRE',
            E_threshold,
            F_threshold,
            maxiter,
            'None',
            1,
            1.,
            1.,
            1.,
            use_bb = False,
            device = device,
            verbose = verbose,
            _hold_samples=_hold_samples
        )
        self.MAX_STEPLENGTH = th.scalar_tensor(10 * steplength, device=self.device)
        self.FIRE_t_init = steplength
        if not (0. < alpha_fac < 1.):
            raise ValueError('alpha_fac must between 0 and 1.')
        if not (0. < fac_dec < 1.):
            raise ValueError('fac_dec must between 0 and 1.')
        if fac_inc <= 1.:
            raise ValueError('fac_inc must be greater than 1.')

        self.alpha = alpha
        self.alpha_fac = alpha_fac
        self.fac_inc = fac_inc
        self.fac_dec = fac_dec
        self.N_min = N_min

        self.v: th.Tensor | None = None        # shape as X
        self.v_: th.Tensor | None = None       #
        self.masses: th.Tensor | None = None   # shape as X
        self.masses_: th.Tensor | None = None  #
        self.t: th.Tensor | None = None        # (1, sumN, 1) or (B, 1, 1)
        self.t_: th.Tensor | None = None
        self.a: th.Tensor | None = None        # (1, sumN, 1) or (B, 1, 1)
        self.n_count: th.Tensor | None = None  # (1, sumN, 1) or (B, 1, 1)

    def initialize_algo_param(self):
        """ """
        if self.is_concat_X:
            self.t = th.full((1, self.n_atom, 1), self.FIRE_t_init, device=self.device)
            self.a = th.full((1, self.n_atom, 1), self.alpha, device=self.device)
            self.n_count = th.zeros((1, self.n_atom, 1), dtype=th.int, device=self.device)
        else:
            self.t = th.full((self.n_true_batch, 1, 1), self.FIRE_t_init, device=self.device)
            self.a = th.full((self.n_true_batch, 1, 1), self.alpha, device=self.device)
            self.n_count = th.zeros((self.n_true_batch, 1, 1), dtype=th.int, device=self.device)

    def _update_algo_param(
            self,
            select_mask,
            select_mask_short,
            batch_scatter_indices,
            X_grad_,
            X_grad_old_,
            p,
            displace_
    ):
        F = - X_grad_
        if self.is_concat_X:
            self.masses_ = self.masses[:, select_mask, :]
            self.t_: th.Tensor = self.t[:, select_mask, :]  # (1, sumN, 1)
            a_ = self.a[:, select_mask, :]  # (1, sumN, 1)
            n_count_ = self.n_count[:, select_mask, :]
            self.v_ = self.v[:, select_mask, :]
            # (1, sumN, n_dim)
            F_hat = F / th.sum(index_inner_product(
                F,
                F,
                dim=1,
                batch_indices=batch_scatter_indices
            ), dim=-1, keepdim=True).sqrt_().index_select(1, batch_scatter_indices)
            # (1, sumN, 1)
            momenta = th.sum(index_inner_product(
                F,
                self.v_,
                dim=1,
                batch_indices=batch_scatter_indices
            ), dim=-1, keepdim=True).index_select(1, batch_scatter_indices)
            # (1, sumN, 1)
            v_norm = th.sum(index_inner_product(
                self.v_,
                self.v_,
                dim=1,
                batch_indices=batch_scatter_indices
            ), dim=-1, keepdim=True).sqrt_().index_select(1, batch_scatter_indices)
            # update velocity: v = v * (1 - a) + a * |v| * \hat{F}
            F_hat.mul_(v_norm)
            self.v_.mul_((1. - a_))
            self.v_.addcmul_(a_, F_hat)
            # if P > 0
            n_count_ += th.where(momenta > 0., 1, -n_count_)  # (1, sumN, 1)
            is_ncount_gt_Nmin = n_count_ >= self.N_min
            #
            new_t_ = (self.t_ * self.fac_inc).clamp_max_(self.MAX_STEPLENGTH)
            self.t_ = th.where(is_ncount_gt_Nmin, new_t_, self.t_)
            a_ = th.where(is_ncount_gt_Nmin, (a_ * self.alpha_fac), a_)
            # if P <= 0.
            is_p_lt_0 = momenta <= 0.
            self.t_ = th.where(is_p_lt_0, (self.t_ * self.fac_dec), self.t_)
            self.v_.masked_fill_(
                is_p_lt_0,
                0.
            )
            a_.masked_fill_(is_p_lt_0, self.alpha)
            # re-write
            select_indices = th.where(select_mask)[0]
            self.t.index_copy_(1, select_indices, self.t_)
            self.a.index_copy_(1, select_indices, a_)
            self.n_count.index_copy_(1, select_indices, n_count_)

        else:
            self.v_ = self.v[select_mask, ...]
            self.masses_ = self.masses[select_mask, ...]
            self.t_ = self.t[select_mask, ...]
            a_ = self.a[select_mask, ...]
            n_count_ = self.n_count[select_mask, ...]

            F_hat = F / (th.linalg.norm(F, dim=(-2, -1), keepdim=True) + 1e-20)
            # (n_batch, n_dim, n_atom) @ (n_batch, n_atom, n_dim) -> (n_batch, 1, 1)
            momenta = th.sum(F * self.v_, dim=(-1, -2), keepdim=True)
            # update velocity
            v_norm = th.linalg.norm(self.v_, dim=(-2, -1), keepdim=True)
            F_hat.mul_(v_norm)
            self.v_.mul_((1. - a_))
            self.v_.addcmul_(a_, F_hat)
            # if P > 0.
            n_count_ += th.where(momenta > 0., 1, -n_count_)
            is_ncount_gt_Nmin = n_count_ >= self.N_min
            new_t_ = (self.t_ * self.fac_inc).clamp_max_(self.MAX_STEPLENGTH)
            self.t_ = th.where(is_ncount_gt_Nmin, new_t_, self.t_)
            a_ = th.where(is_ncount_gt_Nmin, (a_ * self.alpha_fac), a_)
            # if P <= 0.
            is_p_lt_0 = momenta <= 0.
            self.t_ = th.where(is_p_lt_0, (self.t_ * self.fac_dec), self.t_)
            self.v_.masked_fill_(
                is_p_lt_0,
                0.
            )
            a_.masked_fill_(is_p_lt_0, self.alpha)
            # re-write
            select_indices = th.where(select_mask)[0]
            self.t.index_copy_(0, select_indices, self.t_)
            self.a.index_copy_(0, select_indices, a_)
            self.n_count.index_copy_(0, select_indices, n_count_)
        pass

    def _update_direction(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices: th.Tensor | None,
    ) -> th.Tensor:
        self.v_: th.Tensor
        self.v_.addcdiv_(g * self.t_, self.masses_, value=-9.64853329045427e-3)
        disp = self.v_ * self.t_
        return disp

    def _update_algo_batches(
            self,
            select_indices: th.Tensor,
            select_indices_short: th.Tensor | None,
    ):
        if self.is_concat_X:
            self.v.index_copy_(1, select_indices, self.v_)
        else:
            self.v.index_copy_(0, select_indices, self.v_)

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
            require_grad: bool = False,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
            elements: List[List[str | int]] | None = None,
    ) -> Tuple[th.Tensor, th.Tensor] | Tuple[th.Tensor, th.Tensor, th.Tensor]:
        r"""
        Run the Conjugate gradient

        Args:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the optimization variables (e.g., atom coordinates) that input to func.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X i.e., grad = grad_func(X, y, ...), else grad = grad_func(X, ...)
            require_grad: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs) calculation.
            output_grad: bool, whether output gradient of last step.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed. An integer tensor with the same shape as X where 1 is for free and 0 is for fixation.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is the same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]
            elements: Optional[List[List[str | int]]], the Element of each given atom in X.

        Returns:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
            grad of argmin func: Tensor(X.shape), only output when `output_grad` == True. The gradient of X corresponding to minimum.
        """
        # manage Atomic Type & Masses
        if elements is None:
            self.masses = th.ones_like(X)
        elif isinstance(elements, Sequence):
            masses = list()
            for _Elem in elements:
                masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
            masses = th.tensor(masses, dtype=th.float32, device=self.device)
            self.masses = masses.unsqueeze(-1).expand_as(X).clone()  # (n_batch, n_atom, n_dim)
        else:
            raise TypeError(f'Expected masses is a Sequence[Sequence[...]], but occurred {type(elements)}.')
        self.v = th.zeros_like(X, device=self.device)

        _results = super().run(
            func,
            X,
            grad_func,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            output_grad=output_grad,
            fixed_atom_tensor=fixed_atom_tensor,
            batch_indices=batch_indices,
        )

        return _results

