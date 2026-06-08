"""
Line Search for optimizations.
"""

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _line_search.py
#  Environment: Python 3.12

import warnings
from typing import Any, Literal, Sequence, Tuple, Callable, Dict
import logging
import sys
import os

import torch as th
from torch import nn

from BUCToolkit.utils.index_ops import index_reduce, index_inner_product
from BUCToolkit.utils.setup_loggers import has_any_handler,clear_all_handlers
from BUCToolkit.Bases.BaseMotion import BaseMotion


class LineSearch(BaseMotion):
    def __init__(
            self,
            method: Literal['Backtrack', 'B', 'Wolfe', 'W', 'MT', 'EXACT', 'None', 'N'] = 'Backtrack',
            maxiter: int = 10,
            thres: float = 0.02,
            factor: float = 1.5,
            verbose: int = 1,
            hold_samples: bool = False,
    ) -> None:
        """
        Line Search for Optimizations.
        Args:
            method: the method to use for line search.
                'Backtrack': Backtrack line search to satisfy Armijo's condition.
                'B': Alias for 'Backtrack'.
                'Wolfe': More-Thuente algorithm line search to satisfy Wolfe-Powell (weak Wolfe) condition.
                'W': Alias for 'Wolfe'.
                'MT': Alias for 'Wolfe'.
                'EXACT': Exact line search by modified More-Thuente algorithm. (or some kind of Brent)
                'None': No line search. Directly return input steplength.
            maxiter: maximum number of line search iterations.
            thres: directional gradient norm threshold for exact line search. DO NOT WORK FOR INEXACT SEARCH ('B', 'W', 'N').
            factor: the factor used for backtrack line search.
            verbose: verbosity level.
            hold_samples: whether to hold samples during line search iterations.
        """
        self.method = method
        self.maxiter = maxiter
        self.linesearch_thres = thres
        self.factor = factor
        self.factor2 = 0.9/maxiter  # relaxation factor that might be used in backtracking algo.
        self.require_grad = False
        self.verbose = verbose
        self._stp_list = list()
        self._min_steplength_cache = th.scalar_tensor(1e-4)
        self._epsilon_cache = th.scalar_tensor(1e-20)
        self.steplength = None
        self._hold_samples = hold_samples

        ALGORITHMS = {
            'B': self._backtrack,
            'W': self._more_thuente,
            'E': self._exact,
            'N': self._fix_step,
        }
        MAX_STEPLENGTH_COEFFICIENT_DICT = {
            'B': 3.,
            'W': 10.,
            'E': 10.,
            'N': 1.,
        }
        # Literal['Backtrack', 'B', 'Wolfe', 'W', 'MT', 'EXACT', 'BRENT', 'None', 'N']
        if method == 'MT': method = 'W'
        try:
            self._updater = ALGORITHMS[method[0].capitalize()]
        except Exception as e:
            logging.fatal(f"Unknown line search method '{method}'. Error: {e}.")
            raise RuntimeError(f"Unknown line search method '{method}'. Error: {e}.")

        # default of various algorithms
        self.MAX_STEPLENGTH_COEFFICIENT = MAX_STEPLENGTH_COEFFICIENT_DICT[method[0].capitalize()]
        self.WOLFE_RHO = 0.001 if method[0] == 'W' else 0.4
        self.WOLFE_BETA = 0.4
        self.EXPAND_COEFF = 1.5

        # If linear search contains gradient (e.g., Wolfe condition), it can be stored to reuse
        #   thus avoiding a repeated function evalution.
        self.HAS_GRAD = False
        self.STORE_Y: th.Tensor | None = None
        self.STORE_GRAD: th.Tensor | None = None

        # logger
        super().__init__()
        self.init_logger('Main.OPT.LineSearch')

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
            method: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]] | None = None
    ) -> None:
        """
        Set a method to update the taget function when variables change.
        If input Callables, this Callable receives a mask tensor of shape (n_batch, )
        that only selects the `True` part to input to the function, the old
        `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`,
        returns the corresponding masked new `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`.

        If input None, self._hold_samples will be set to True that toggles off the dynamic removal.

        This method is used to dynamically 'remove' the samples which have been converged in a batch to avoid
        redundant calculation of converged samples.

        Default transform is identical transform (i.e., do nothing)
        Args:
            method: Callable(mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict) -> Tuple[Tuple, Dict, Tuple, Dict],
the method of updating function arguments for a mask.

        Returns: None
        """
        if method is None:
            self._hold_samples = True
        elif callable(method):
            self._update_batch = method
        else:
            raise TypeError(f'`method` must be a callable, but {type(method)} is not.')

    def _Armijo_cond_irreg(
            self,
            func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            direct_grad0: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,  # (1, n_true_batch, 1)
            batch_scatter_indices: th.Tensor,
            rho: float = 0.05,
            func_args: Sequence = tuple(),
            func_kwargs=None
    ) -> th.Tensor:
        """
        Armijo conditioned for irregular batches.

        Returns:  (1, n_batch, 1)

        """
        if func_kwargs is None:
            func_kwargs = dict()
        steplength_long = steplength.index_select(1, batch_scatter_indices)
        _X_cache = th.addcmul(X0, steplength_long, p, value=1.)  # (1, B*A, D)
        y1 = func(_X_cache, *func_args, **func_kwargs)  # (B, )
        #direct_grad0 = th.sum(index_inner_product(grad, p, 1, batch_scatter_indices), dim=-1)  # (1, B, 1)
        # (1, B, 1) <= (1, B, 1) + 1. * (1, B, 1) * (1, B, 1)
        amj_mask: th.Tensor = (y1.unsqueeze(-1) <= th.addcmul(y0.unsqueeze(-1), steplength, direct_grad0, value=rho))
        return amj_mask  # (1, B, 1)

    def _Armijo_cond_reg(
            self,
            func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            direct_grad0: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,  # (1, n_true_batch, 1)
            batch_scatter_indices: th.Tensor,
            rho: float = 0.05,
            func_args: Sequence = tuple(),
            func_kwargs=None
    ) -> th.Tensor:
        """
        Armijo conditioned for normal batches.

        Returns:  (1, n_batch, 1)

        """
        if func_kwargs is None:
            func_kwargs = dict()
        _X_cache = th.addcmul(X0, steplength, p, value=1.)  # (B, A, D)
        y1 = func(_X_cache, *func_args, **func_kwargs)  # (B, )
        amj_mask: th.Tensor = (y1.reshape(-1, 1, 1) <= (y0.reshape(-1, 1, 1) + rho * steplength * direct_grad0))
        return amj_mask  # (n_batch, 1, 1)

    def _inv_Armijo_cond_irreg(
            self,
            func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            direct_grad0: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,
            batch_scatter_indices: th.Tensor,
            rho: float = 0.05,
            func_args: Sequence = tuple(),
            func_kwargs=None
    ) -> th.Tensor:
        """
        Inverse Armijo conditioned for irregular batches, i.e., the sufficient raising condition.

        """
        if func_kwargs is None:
            func_kwargs = dict()
        steplength_long = steplength.index_select(1, batch_scatter_indices)
        _X_cache = th.addcmul(X0, steplength_long, p, value=1.)  # (1, B*A, D)
        y1 = func(_X_cache, *func_args, **func_kwargs)  # (B, )
        #direct_grad0 = th.sum(index_inner_product(grad, p, 1, batch_scatter_indices), dim=-1)  # (1, B, 1)
        amj_mask: th.Tensor = (y1.unsqueeze(-1) > th.addcmul(y0.unsqueeze(-1), steplength, direct_grad0, value=rho))
        return amj_mask  # (1, B, 1)

    def _Wolfe_cond_irreg(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            direct_grad0: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,
            is_grad_func_contain_y: bool,
            atom_masks: th.Tensor,
            batch_scatter_indices: th.Tensor,
            rho: float = 0.05,
            beta: float = 0.9,
            func_args: Tuple = tuple(),
            func_kwargs=None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs=None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Weak Wolfe-Powell condition.
        Returns:
            a: (1, B, 1), Armijo condition mask
            b: (1, B, 1), Wolfe gradient condition mask
            y1: (B, ), function value at current steplength
            direct_grad1: (1, B, 1), direction gradient at current point

        """
        if func_kwargs is None:
            func_kwargs = dict()
        steplength_long = steplength.index_select(1, batch_scatter_indices)
        _X_cache = th.addcmul(X0, steplength_long, p, value=1.)  # (1, B*A, D)
        y1, dy1 = self._calc_y_grad(
            _X_cache,
            func,
            func_args,
            func_kwargs,
            grad_func,
            grad_func_args,
            grad_func_kwargs,
            self.require_grad,
            is_grad_func_contain_y
        )
        # detach grad graph
        dy1 = dy1.detach().mul_(atom_masks)  # (1, B*A, D)
        y1 = y1.detach()    # (B, )
        # store to reuse
        self.tmp_HAS_GRAD = True
        self.tmp_STORE_Y = y1
        self.tmp_STORE_GRAD = dy1

        direct_grad1 = th.sum(index_inner_product(dy1, p, 1, batch_scatter_indices), dim=-1, keepdim=True)  # (1, B, 1)
        amj: th.Tensor = (y1.unsqueeze(-1) <= th.addcmul(y0.unsqueeze(-1), steplength, direct_grad0, value=rho))  # descent cond, (1, B, 1)
        curv = (direct_grad1 > beta * direct_grad0)  # curve cond, (1, B, 1)

        return amj, curv, y1, direct_grad1

    def _Wolfe_cond_reg(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            direct_grad0: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,       # (n_batch, 1, 1)
            is_grad_func_contain_y: bool,
            atom_masks: th.Tensor,
            batch_scatter_indices: None = None,  # a placeholder
            rho: float = 0.05,
            beta: float = 0.9,
            func_args: Tuple = tuple(),
            func_kwargs=None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs=None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Weak Wolfe-Powell condition.
        Returns:
            a: (B, 1, 1), Armijo condition mask
            b: (B, 1, 1), Wolfe gradient condition mask
            y1: (B, ), function value at current steplength
            direct_grad1: (B, 1, 1), direction gradient at current point

        """
        if func_kwargs is None:
            func_kwargs = dict()
        _X_cache = th.addcmul(X0, steplength, p, value=1.)  # (B, A, D)
        y1, dy1 = self._calc_y_grad(
            _X_cache,
            func,
            func_args,
            func_kwargs,
            grad_func,
            grad_func_args,
            grad_func_kwargs,
            self.require_grad,
            is_grad_func_contain_y
        )
        # detach grad graph
        dy1 = dy1.detach().mul_(atom_masks)  # (B, A, D)
        y1 = y1.detach()    # (B, )
        # store to reuse
        self.tmp_HAS_GRAD = True
        self.tmp_STORE_Y = y1
        self.tmp_STORE_GRAD = dy1

        direct_grad1 = th.sum(dy1 * p, dim=(-2, -1), keepdim=True)  # (B, 1, 1)
        a: th.Tensor = (y1.reshape(-1, 1, 1) <= th.addcmul(y0.reshape(-1, 1, 1), steplength, direct_grad0, value=rho))  # descent cond, (B, 1, 1)
        b = (direct_grad1 > beta * direct_grad0)  # curve cond, (B, 1, 1)

        return a, b, y1, direct_grad1

    def _backtrack_update(self, steplength_in, i_: int):
        # applied shrinking
        steplength_out = (steplength_in - (steplength_in - 1e-4)*((i_ + 1)/self.maxiter)**self.factor)
        # update factor. If not converge, decrease the factor.
        return steplength_out

    def _3ord_hermite_interp(
            self,
            x_h,
            y_h,
            g_h,
            x_l,
            y_l,
            g_l
    ):
        """
        Third order hermite interpolation to find the 2 stationary points.
        Args:
            x_h: x of the high endpoint
            y_h: function value of the high endpoint
            g_h: function directional gradient of the high endpoint
            x_l: x of the low endpoint
            y_l: function value of the low endpoint
            g_l: function directional gradient of the low endpoint
        Returns:
            neg_Delta_mask: the mask shows where Delta (b^2 - 4 * a * c) < 0, leading to imag. root.
            s1: the shorter stationary points corresponding to the smaller root of 2nd order equation of derivation.
            s2: the longer stationary points corresponding to the larger root of 2nd order equation of derivation.

        """
        _h = x_h - x_l
        _gh_plus_gl = g_h + g_l
        # _a = 6. * (y_h - y_l) - 3. * _h * (g_h + g_l)
        _a: th.Tensor = 6. * y_h
        _a.add_(y_l, alpha = -6.)
        _a.addcmul_(_h, _gh_plus_gl, value=-3.)
        # _b = - 6. * (y_h - y_l) + 2. * _h * (g_h + 2 * g_l)
        _b: th.Tensor = -6. * y_h
        _b.add_(y_l, alpha = 6.)
        _gh_plus_gl.add_(g_l)                      # WARNING: IN-PLACE change of _gh_plus_gl.
        _b.addcmul_(_h, _gh_plus_gl, value=2.)
        _c = _h * g_l
        # D = _b ** 2 - 4. * _a * _c
        D = th.addcmul(_b**2, _a, _c, value=-4.)
        _neg_D_mask = (D < 0.)
        # q = - _b +- th.sqrt(D)
        (D.clamp_min_(self._epsilon_cache)).sqrt_()  # WARNING: IN-PLACE change of D.
        _a.add_(self._epsilon_cache)
        q1 = - _b - D
        q2 = - _b + D
        #interp3_1 = x_l + _h * q1 / (2 * _a.clamp_min_(self._epsilon_cache))
        #interp3_2 = x_l + _h * q2 / (2 * _a.clamp_min_(self._epsilon_cache))
        interp3_1 = th.addcdiv(x_l, _h * q1, _a, value=0.5)
        interp3_2 = th.addcdiv(x_l, _h * q2, _a, value=0.5)

        return _neg_D_mask, interp3_1, interp3_2

    def _2ord_interp(
            self,
            x_h,
            y_h,
            x_l,
            y_l,
            g_l
    ):
        """
        Second order interpolation to find the stationary points.
        Args:
            x_h:
            y_h:
            x_l:
            y_l:
            g_l:

        Returns:
        """
        # y = x_l - (g_l * (x_h - x_l) ** 2) / (2 * (y_h - y_l - g_l * (x_h - x_l)))
        #   = x_l - (g_l * d_xhl ** 2) / (2 * (y_h - y_l - g_l * d_xhl))
        #   = {x_l - 0.5 * (g_l * d_xhl ** 2) / (y_h - {y_l + g_l * d_xhl})}
        d_xhl: th.Tensor = x_h - x_l
        y_add_gd = th.addcmul(y_l, g_l, d_xhl)
        d_xhl.mul_(d_xhl)
        y = th.addcdiv(x_l, g_l * d_xhl, (y_h - y_add_gd).add_(self._epsilon_cache), value=-0.5)
        return y

    def _extrapolation(
            self,
            steplength_in: th.Tensor,
            direct_grad_now: th.Tensor,
            _direct_grad0: th.Tensor,
            max_steplen: th.Tensor | float,
    ):
        """
        Try 2nd-order extrapolation, i.e., the secant method, otherwise simple linear expansion.
        Args:
            steplength_in:
            direct_grad_now:
            _direct_grad0:
            max_steplen

        Returns:

        """
        # secant
        secant_try = - steplength_in * _direct_grad0 / (direct_grad_now - _direct_grad0 + self._epsilon_cache)
        # linear
        line_try = 0.5 * (max_steplen + steplength_in)

        ext_step = th.where(
            (secant_try <= steplength_in).bitwise_or_(secant_try > max_steplen),
            line_try,
            secant_try
        )

        return ext_step

    def _backtrack(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor | float,
            is_grad_func_contain_y: bool,
            func_args: Tuple = tuple(),
            func_kwargs=None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs=None,
            fixed_atom_tensor: th.Tensor = None,
            batch_indices: th.Tensor = None
    ):
        """
        Simple backtracking scheme with Armijo condition.

        Returns: (B, 1, 1) for regular batches, and (1, B, 1) for irregular batches.

        """
        self.factor = max(self.factor, 0.2)
        self._min_steplength_cache = self._min_steplength_cache.to(X0.device)
        _steplength = steplength
        is_converge = False
        converge_check = th.atleast_1d(th.full_like(steplength, False, dtype=th.bool).squeeze_())
        if batch_indices is None:
            direct_grad0 = th.sum(grad * p, dim=(-1, -2), keepdim=True)  # (B, 1, 1)
            Armijo_updater = self._Armijo_cond_reg
            batch_tensor_indx_cache = None
            batch_scatter = None
        else:
            batch_tensor_indx_cache = th.arange(len(batch_indices), dtype=th.int64, device=X0.device)
            batch_scatter = th.repeat_interleave(
                batch_tensor_indx_cache,
                batch_indices,
                dim=0
            )
            direct_grad0 = th.sum(index_inner_product(grad, p, 1, batch_scatter), dim=-1, keepdim=True)  # (1, B, 1)
            Armijo_updater = self._Armijo_cond_irreg

        # main loop
        for i in range(self.maxiter):
            if not self._hold_samples:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_check,
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                if batch_indices is not None:
                    select_mask = ~converge_check
                    select_mask_long = select_mask[batch_scatter]
                    y0_ = y0[select_mask]
                    direct_grad0_ = direct_grad0[:, select_mask, :]
                    p_ = p[:, select_mask_long, :]
                    X0_ = X0[:, select_mask_long, :]
                    _steplength_ = _steplength[:, select_mask, :]
                    ori_steplength_ = steplength[:, select_mask, :]
                    batch_tensor_ = batch_indices[select_mask]
                    batch_scatter_ = th.repeat_interleave(
                        batch_tensor_indx_cache[:len(batch_tensor_)],
                        batch_tensor_,
                        dim=0
                    )
                else:
                    select_mask = ~converge_check
                    y0_ = y0[select_mask]
                    direct_grad0_ = direct_grad0[select_mask, ...]
                    p_ = p[select_mask, ...]
                    X0_ = X0[select_mask, ...]
                    _steplength_ = _steplength[select_mask, ...]
                    ori_steplength_ = steplength[:, select_mask, :]
                    batch_tensor_ = None
                    batch_scatter_ = None
            else:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = (
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~converge_check
                y0_ = y0
                direct_grad0_ = direct_grad0
                p_ = p
                X0_ = X0
                _steplength_ = _steplength
                ori_steplength_ = steplength
                batch_tensor_ = batch_indices
                batch_scatter_ = batch_indices
            # Section: main update
            mask = Armijo_updater(
                func,
                X0_,
                y0_,
                direct_grad0_,
                p_,
                _steplength_,
                batch_scatter_,
                rho=0.05,
                func_args=func_args_,
                func_kwargs=func_kwargs_
            )
            converge_check_ = mask.reshape(-1)
            if th.all(mask):
                is_converge = True
                break
            # A kind of 'slow' backtrack.
            _steplength_ = th.where(mask, _steplength_, self._backtrack_update(ori_steplength_, i))
            if self.verbose > 1:
                self.logger.info(
                    f"Steplength Line Search {i + 1}: "
                    f"{' '.join([f'{_: 6.4f}' for _ in _steplength.reshape(-1).tolist()])}"
                )
            # Section: re-write
            if not self._hold_samples:
                if batch_indices is not None:
                    select_indices = th.where(select_mask)[0]
                    _steplength.index_copy_(1, select_indices, _steplength_)
                    converge_check.index_copy_(0, select_indices, converge_check_)
                else:
                    select_indices = th.where(select_mask)[0]
                    _steplength.index_copy_(0, select_indices, _steplength_)
                    converge_check.index_copy_(0, select_indices, converge_check_)
            else:
                _steplength = _steplength_
                converge_check = converge_check_

        _steplength.clamp_min_(self._min_steplength_cache)
        if not is_converge:
            self.logger.warning(f'WARNING: linear search did not converge in {self.maxiter} steps.')
        return _steplength  # (1, B, 1)/(B, 1, 1)

    def _exact(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor | float,
            is_grad_func_contain_y: bool,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            fixed_atom_tensor: th.Tensor = None,
            batch_indices: th.Tensor = None
    ):

        raise NotImplementedError

    def _more_thuente(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor | float,
            is_grad_func_contain_y: bool,
            func_args: Tuple = tuple(),
            func_kwargs=None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs=None,
            fixed_atom_tensor: th.Tensor = None,
            batch_indices: th.Tensor = None
    ):
        """
        More-Thuente algorithm that search step length satisfied Wolfe condition by interpolation.

        Returns: (B, 1, 1) for regular batches, and (1, B, 1) for irregular batches.

        """
        _steplength = steplength
        _max_steplength = self.MAX_STEPLENGTH_COEFFICIENT * steplength.max()  # step length supremum, A SCALAR.
        _steplength_old = th.full_like(_steplength ,th.inf)
        self._min_steplength_cache = self._min_steplength_cache.to(X0.device)
        self._epsilon_cache = self._epsilon_cache.to(X0.device)
        self.factor = max(self.factor, 0.2)
        is_converge = False
        if batch_indices is None:
            batch_tensor_indx_cache = None
            batch_scatter_indices = None
            direct_grad_orig = th.sum(grad * p, dim=(-2, -1), keepdim=True)  # (n_batch, 1, 1)
            direct_grad0 = direct_grad_orig
            WolfCond = self._Wolfe_cond_reg
            _long_shape = (-1, 1, 1)
        else:
            batch_tensor_indx_cache = th.arange(0, len(batch_indices), dtype=th.int64, device=X0.device)
            batch_scatter_indices = th.repeat_interleave(
                        batch_tensor_indx_cache,
                        batch_indices,
                        dim=0
                    )
            direct_grad_orig = th.sum(index_inner_product(grad, p, 1, batch_scatter_indices), dim=-1, keepdim=True)  # (1, B, 1)
            direct_grad0 = direct_grad_orig
            WolfCond = self._Wolfe_cond_irreg
            _long_shape = (1, -1, 1)
        # Selective dyamics
        atom_masks = self.handle_motion_mask(X0, fixed_atom_tensor)

        # main loop
        # initial intervals
        _left = th.zeros_like(steplength)  # step length infimum
        _y_left = y0.reshape(_long_shape).clone()
        _g_left = direct_grad0.clone()
        _right = _steplength.clone()
        armijo_mask, curve_mask, _y_right, _g_right = WolfCond(
            func,
            grad_func,
            X0,
            y0,
            direct_grad0,
            p,
            _right,  # (B, 1, 1)
            is_grad_func_contain_y,
            atom_masks,
            batch_scatter_indices,
            rho=self.WOLFE_RHO,
            beta=self.WOLFE_BETA,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs
        )
        # store cache values
        self.HAS_GRAD = True
        self.STORE_Y = _y_right
        self.STORE_GRAD = self.tmp_STORE_GRAD

        all_armijo = th.all(armijo_mask)
        if all_armijo and th.all(curve_mask):
            is_converge = True
            return _right
        converge_mask = (armijo_mask & curve_mask)  # (1, B, 1)
        converge_check = converge_mask.reshape(-1)  # a short version of conv. mask
        cumulative_mask = th.full_like(_steplength, False, dtype=th.bool)
        backup_steplength = th.full_like(_steplength, 1.e-4)
        # (B, ), mask of where is stopped yet not converged (fixed or reached max)
        stop_mask = th.full_like(converge_check, False, dtype=th.bool)

        y1 = _y_right.clone()
        direct_grad1 = _g_right.clone()
        _y_right = _y_right.reshape(_long_shape)
        for i in range(self.maxiter):
            # back up the steplength that at least met Armijo cond.
            first_time_mask = armijo_mask & (~cumulative_mask)  # the mask of first time meet Armijo
            backup_steplength = th.where(first_time_mask, _steplength, backup_steplength)
            cumulative_mask.bitwise_or_(armijo_mask)
            # check thres
            converge_check = (armijo_mask & curve_mask).reshape(-1)
            if self.verbose > 1:
                self.logger.info(
                    f"Steplength Line Search {i}: "
                    f"{' '.join([f'{_: 6.4f}' for _ in _steplength.reshape(-1).tolist()])}"
                )
            if th.all(converge_check):
                is_converge = True
                break
            elif th.all(converge_check | stop_mask):
                break

            # Section: select unconverged samples
            if not self._hold_samples:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~(converge_check & stop_mask),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                if batch_scatter_indices is not None:
                    select_mask = ~(converge_check & stop_mask)
                    select_mask_long = select_mask[batch_scatter_indices]
                    y0_ = y0[select_mask]
                    direct_grad0_ = direct_grad0[:, select_mask, :]
                    p_ = p[:, select_mask_long, :]
                    X0_ = X0[:, select_mask_long, :]
                    atom_masks_ = atom_masks[:, select_mask_long, :]

                    _steplength_ = _steplength[:, select_mask, :]
                    y1_ = y1[select_mask]
                    direct_grad1_ = direct_grad1[:, select_mask, :]
                    _left_ = _left[:, select_mask, :]
                    _y_left_ = _y_left[:, select_mask, :]
                    _g_left_ = _g_left[:, select_mask, :]
                    _right_ = _right[:, select_mask, :]
                    _y_right_ = _y_right[:, select_mask, :]
                    _g_right_ = _g_right[:, select_mask, :]

                    armijo_mask_ = armijo_mask[:, select_mask, :]
                    # curve_mask_ = curve_mask[:, select_mask, :]
                    batch_tensor_ = batch_indices[select_mask]
                    batch_scatter_ = th.repeat_interleave(
                        batch_tensor_indx_cache[:len(batch_tensor_)],
                        batch_tensor_,
                        dim=0
                    )
                else:
                    select_mask = ~(converge_check & stop_mask)
                    select_mask_long = select_mask
                    y0_ = y0[select_mask]
                    direct_grad0_ = direct_grad0[select_mask, ...]
                    p_ = p[select_mask, ...]
                    X0_ = X0[select_mask, ...]
                    atom_masks_ = atom_masks[select_mask, ...]

                    _steplength_ = _steplength[select_mask, ...]
                    y1_ = y1[select_mask]
                    direct_grad1_ = direct_grad1[select_mask, :, :]
                    _left_ = _left[select_mask, ...]
                    _y_left_ = _y_left[select_mask, ...]
                    _g_left_ = _g_left[select_mask, ...]
                    _right_ = _right[select_mask, ...]
                    _y_right_ = _y_right[select_mask, ...]
                    _g_right_ = _g_right[select_mask, ...]

                    armijo_mask_ = armijo_mask[select_mask, :, :]
                    # curve_mask_ = curve_mask[select_mask, :, :]
                    batch_tensor_ = None
                    batch_scatter_ = None
            else:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = (
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_check & stop_mask)
                select_mask_long = select_mask
                y0_ = y0
                direct_grad0_ = direct_grad0
                p_ = p
                X0_ = X0
                atom_masks_ = atom_masks

                _steplength_ = _steplength
                y1_ = y1
                direct_grad1_ = direct_grad1
                armijo_mask_ = armijo_mask
                curve_mask_ = curve_mask
                batch_tensor_ = batch_indices
                batch_scatter_ = batch_scatter_indices
                _left_ = _left
                _y_left_ = _y_left
                _g_left_ = _g_left
                _right_ = _right
                _y_right_ = _y_right
                _g_right_ = _g_right

            # Section: main update
            y1_long = y1_.reshape(_long_shape)
            # determine new interval
            _is_grad_positiv = (direct_grad1_ >= 0.)
            _is_descent = (y1_ <= y0_).reshape(_long_shape)
            _is_expansion = (~_is_grad_positiv) & _is_descent
            # *_tmp is used with assuming interval satisfied (where _is_expansion == False),
            #   otherwise, do expansion instead.
            _right_tmp = th.where(
                armijo_mask_,
                th.where(
                    _is_grad_positiv,
                    _steplength_,
                    _right_,
                ),
                _steplength_
            )
            _y_right_tmp = th.where(
                armijo_mask_,
                th.where(
                    _is_grad_positiv,
                    y1_long,
                    _y_right_,
                ),
                y1_long
            )
            _g_right_tmp = th.where(
                armijo_mask_,
                th.where(
                    _is_grad_positiv,
                    direct_grad1_,
                    _g_right_,
                ),
                direct_grad1_
            )
            _left_ = th.where(
                armijo_mask_,
                th.where(
                    _is_grad_positiv,
                    _left_,
                    _steplength_
                ),
                _left_
            )
            _y_left_ = th.where(
                armijo_mask_,
                th.where(
                    _is_grad_positiv,
                    _y_left_,
                    y1_long
                ),
                _y_left_
            )
            _g_left_ = th.where(
                armijo_mask_,
                th.where(
                    _is_grad_positiv,
                    _g_left_,
                    direct_grad1_
                ),
                _g_left_
            )
            # interpolation step 3nd interp
            _neg_D_mask, interp3_1 ,interp3_2 = self._3ord_hermite_interp(
                _right_tmp,
                _y_right_tmp,
                _g_right_tmp,
                _left_,
                _y_left_,
                _g_left_,
            )
            # interpolation step 2nd interp
            interp2 = self._2ord_interp(
                _right_tmp,
                _y_right_tmp,
                _left_,
                _y_left_,
                _g_left_,
            )
            # simple cut
            golcut = th.add(_left_, (_right_tmp - _left_), alpha = 0.6180339887)
            # choice steplength. A more strict interval that [left * 1.1, right * 0.9], leaving 10% redundancy.
            _low_order_choice = th.where(
                (interp2 > _left_ * 1.1) & (interp2 < _right_tmp * 0.9),
                interp2,
                golcut
            )
            _steplength_tmp = th.where(
                _neg_D_mask,
                _low_order_choice,
                th.where(
                    (interp3_2 > _left_ * 1.1) & (interp3_2 < _right_tmp * 0.9),   # firstly use longer steplength to avoid too short interval
                    interp3_2,
                    th.where(
                        (interp3_1 > _left_ * 1.1) & (interp3_1 < _right_tmp * 0.9),
                        interp3_1,
                        _low_order_choice
                    )
                )
            )
            # handling the expansion
            #   when expansion, _steplength_ is always equal to _right_
            expanse_stp = self._extrapolation(
                _steplength_,
                direct_grad1_,
                direct_grad0_,
                _max_steplength
            )
            _steplength_ = th.where(
                _is_expansion,
                expanse_stp,
                _steplength_tmp
            )
            # check Wolfe
            armijo_mask_, curve_mask_, y1_, direct_grad1_ = WolfCond(
                func,
                grad_func,
                X0_,
                y0_,
                direct_grad0_,
                p_,
                _steplength_,  # (B, 1, 1)
                is_grad_func_contain_y,
                atom_masks_,
                batch_scatter_,
                rho=self.WOLFE_RHO,
                beta=self.WOLFE_BETA,
                func_args=func_args_,
                func_kwargs=func_kwargs_,
                grad_func_args=grad_func_args_,
                grad_func_kwargs=grad_func_kwargs_
            )
            # if still expansion, set the right end to the current steplength
            _right_ = th.where(
                _is_expansion,
                _steplength_,
                _right_tmp
            )
            _y_right_ = th.where(
                _is_expansion,
                y1_.reshape(_long_shape),
                _y_right_tmp
            )
            _g_right_ = th.where(
                _is_expansion,
                direct_grad1_,
                _g_right_tmp
            )

            # Section: re-write
            if not self._hold_samples:
                if batch_scatter_indices is not None:
                    select_indices = th.where(select_mask)[0]
                    _steplength.index_copy_(1, select_indices, _steplength_)
                    y1.index_copy_(0, select_indices, y1_)
                    direct_grad1.index_copy_(1, select_indices, direct_grad1_)
                    _left.index_copy_(1, select_indices, _left_)
                    _y_left.index_copy_(1, select_indices, _y_left_)
                    _g_left.index_copy_(1, select_indices, _g_left_)
                    _right.index_copy_(1, select_indices, _right_)
                    _y_right.index_copy_(1, select_indices, _y_right_)
                    _g_right.index_copy_(1, select_indices, _g_right_)
                    armijo_mask.index_copy_(1, select_indices, armijo_mask_)
                    curve_mask.index_copy_(1, select_indices, curve_mask_)
                    # update storages
                    select_indices_long = th.where(select_mask_long)[0]
                    self.STORE_GRAD.index_copy_(1, select_indices_long, self.tmp_STORE_GRAD)
                    self.STORE_Y = y1
                else:
                    select_indices = th.where(select_mask)[0]
                    _steplength.index_copy_(0, select_indices, _steplength_)
                    y1.index_copy_(0, select_indices, y1_)
                    direct_grad1.index_copy_(0, select_indices, direct_grad1_)
                    _left.index_copy_(0, select_indices, _left_)
                    _y_left.index_copy_(0, select_indices, _y_left_)
                    _g_left.index_copy_(0, select_indices, _g_left_)
                    _right.index_copy_(0, select_indices, _right_)
                    _y_right.index_copy_(0, select_indices, _y_right_)
                    _g_right.index_copy_(0, select_indices, _g_right_)
                    armijo_mask.index_copy_(0, select_indices, armijo_mask_)
                    curve_mask.index_copy_(0, select_indices, curve_mask_)

                    self.STORE_GRAD.index_copy_(0, select_indices, self.tmp_STORE_GRAD)
                    self.STORE_Y = y1
            else:
                _steplength = _steplength_
                _left = _left_
                _y_left = _y_left_
                _g_left = _g_left_
                _right = _right_
                _y_right = _y_right_
                _g_right = _g_right_
                armijo_mask = armijo_mask_
                curve_mask = curve_mask_

                self.STORE_GRAD = self.tmp_STORE_GRAD
                self.STORE_Y = y1

            # Section reinsurance
            #   stop when steplength fixed
            #   or reached max steplength
            #   or interval too small
            stop_mask = (th.abs(_steplength - _steplength_old) < 5.e-5).reshape(-1)
            stop_mask.bitwise_or_(th.abs(_steplength.reshape(-1) - _max_steplength) < 5.e-5)
            stop_mask.bitwise_or_(th.abs(_right - _left).reshape(-1) < 5.e-5)

            _steplength_old.copy_(_steplength)

            self.logger.debug(f">>>>>>>>>>>>>>> Interval    {i+1}:\n{_left} ;\n{_right}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            self.logger.debug(f">>>>>>>>>>>>>>> Steplength  {i+1}:\n{_steplength}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            self.logger.debug(f">>>>>>>>>>>>>>> Armijo mask {i+1}:\n{armijo_mask}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            self.logger.debug(f">>>>>>>>>>>>>>> Curve mask  {i+1}:\n{curve_mask}\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        if not is_converge:
            self.logger.warning(f'WARNING: linear search did not converge in {self.maxiter} steps.')
            # clear stored information, they are misleading
            self.HAS_GRAD = False
            self.STORE_Y = None
            self.STORE_GRAD = None
            # try to use backup
            _steplength = th.where(converge_check.reshape(_long_shape), _steplength, backup_steplength)
        _steplength.clamp_min_(self._min_steplength_cache)
        return _steplength  # (n_batch, 1, 1)

    def _fix_step(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor | float,
            is_grad_func_contain_y: bool,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            atom_masks_: th.Tensor | None = None,
            batch_scatter_indices: th.Tensor = None
    ):
        """
        simple fixed step length.
        """
        return steplength

    def run(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor|float,
            is_grad_func_contain_y: bool,
            require_grad: bool = False,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            atom_masks: th.Tensor = None,
            batch_indices: th.Tensor=None
    ) -> th.Tensor:
        """

        Args:
            func:
            grad_func:
            X0: (1, sumN, n_dim)
            y0: (n_batch, )
            grad: (1, sumN, n_dim)
            p: (1, sumN, n_dim)
            steplength: (1, sumN, 1)
            is_grad_func_contain_y:
            require_grad:
            func_args:
            func_kwargs:
            grad_func_args:
            grad_func_kwargs:
            atom_masks:
            batch_indices:

        Returns:

        """
        if func_kwargs is None:
            func_kwargs = dict()
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        steplength = steplength.reshape(-1, 1, 1) if batch_indices is None else steplength.reshape(1, -1, 1)
        self.steplength = steplength
        self.require_grad = require_grad
        with th.no_grad():
            _result = self._updater(
                func,
                grad_func,
                X0,
                y0,
                grad,
                p,
                steplength,
                is_grad_func_contain_y,
                func_args,
                func_kwargs,
                grad_func_args,
                grad_func_kwargs,
                atom_masks,
                batch_indices
            )
        if self.verbose > 0:
            self.logger.info(f"Steplength: {' '.join([f'{_: 6.4f}' for _ in th.atleast_1d(_result.squeeze()).tolist()])}")

        return _result
