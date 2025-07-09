"""
Line Search for optimizations.
"""

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _line_search.py
#  Environment: Python 3.12

import warnings
# ruff: noqa: E701, E702, E703, F401
from typing import Any, Literal, Sequence, Tuple

import torch as th
from torch import nn


class _LineSearch:
    def __init__(
            self,
            method: Literal['Backtrack', 'Wolfe', 'NWolfe', '2PT', '3PT', 'Golden', 'Newton', 'None'],
            maxiter: int = 10,
            thres: float = 0.02,
            factor: float = 1.5,
    ) -> None:
        self.method = method
        self.maxiter = maxiter
        self.linesearch_thres = thres
        self.factor = factor
        self.factor2 = 0.9/maxiter  # relaxation factor that might be used in backtracking algo.
        pass

    def _Armijo_cond(
            self,
            func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,
            rho: float = 0.05,
            func_args: Sequence = tuple(),
            func_kwargs=None
    ) -> th.Tensor:
        if func_kwargs is None:
            func_kwargs = dict()
        with th.no_grad():
            y1 = func(X0 + steplength * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)
            if self.is_concat_X: y1 = th.sum(y1 * self.had_converge_mask, dim=(0, -1), keepdim=True)
        a: th.Tensor = (y1 <= (
                y0 + rho * steplength * (grad.mT @ p)))  # Tensor[bool]: (n_batch, ) = (n_batch, ) + (n_batch, 1, 1) * (n_batch, ) * (n_batch, )
        return a  # (n_batch, 1, 1)

    def _inv_Armijo_cond(
            self,
            func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,
            rho: float = 0.05,
            func_args: Sequence = tuple(),
            func_kwargs=None
    ) -> th.Tensor:
        if func_kwargs is None:
            func_kwargs = dict()
        with th.no_grad():
            y1 = func(X0 + steplength * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)
            if self.is_concat_X: y1 = th.sum(y1 * self.had_converge_mask, dim=(0, -1), keepdim=True)
        # Tensor[bool]: (n_batch, 1, 1) = (n_batch, 1, 1) + (n_batch, 1, 1) * (n_batch, ) * (n_batch, )
        a: th.Tensor = (y1 >= (y0 + rho * steplength * (grad.mT @ p)))
        return a  # (n_batch, 1, 1)

    def _Wolfe_cond(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            direct_grad0: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,
            is_grad_func_contain_y: bool,
            rho: float = 0.05,
            beta: float = 0.9,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Weak Wolfe-Powell condition.
        Args:
            func:
            X0:
            y0:
            direct_grad0:
            p:
            steplength:
            rho:
            func_args:
            func_kwargs:

        Returns:
            a: Armijo condition mask
            b: Wolfe gradient condition mask
            y1: function value at current steplength
            direct_grad1: direction gradient at current point

        """
        if func_kwargs is None:
            func_kwargs = dict()
        Xn = X0 + steplength * p.view(self.n_batch, self.n_atom, self.n_dim)  # (n_batch, n_atom, n_dim)
        with th.enable_grad():
            Xn.requires_grad_()
            y1 = (func(Xn, *func_args, **func_kwargs)).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
            if self.is_concat_X: y1 = th.sum(y1 * self.had_converge_mask, dim=(0, -1), keepdim=True)
            if is_grad_func_contain_y:
                dy1 = grad_func(y1, Xn, *grad_func_args, **grad_func_kwargs)
            else:
                dy1 = grad_func(Xn, *grad_func_args, **grad_func_kwargs)
        # detach grad graph
        dy1 = dy1.detach()
        y1 = y1.detach()
        Xn = Xn.detach()
        dy1 = dy1.flatten(-2, -1).unsqueeze(-1)  # (n_batch, n_atom*n_dim, 1)
        with th.no_grad():
            direct_grad1 = dy1.mT @ p
            a: th.Tensor = (y1 <= (y0 + rho * steplength * direct_grad0))  # descent cond
            b = (direct_grad1 > beta * direct_grad0)  # curve cond

        return a, b, y1, direct_grad1

    def _NWolfe_cond(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor,
            is_grad_func_contain_y: bool,
            rho: float = 0.05,
            beta: float = 0.9,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Weak Wolfe-Powell condition with numeric gradient.
        Args:
            func:
            X0:
            y0:
            grad:
            p:
            steplength:
            rho:
            func_args:
            func_kwargs:

        Returns:
            a: Armijo condition mask
            b: Wolfe gradient condition mask
            y1:

        """
        ds = 1e-2  # finite difference steplength
        if func_kwargs is None:
            func_kwargs = dict()
        Xn = X0 + steplength * p.view(self.n_batch, self.n_atom, self.n_dim)  # (n_batch, n_atom, n_dim)
        y1 = (func(Xn, *func_args, **func_kwargs)).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
        if self.is_concat_X: y1 = th.sum(y1 * self.had_converge_mask, dim=(0, -1), keepdim=True)
        # detach grad graph
        direct_grad0 = grad.mT @ p  # (n_batch, 1, n_atom*n_dim) @ (n_batch, n_atom*n_dim, 1) -> (n_batch, 1, 1)
        Xn_a = Xn + ds * p.view(self.n_batch, self.n_atom, self.n_dim)
        Xn_b = Xn - ds * p.view(self.n_batch, self.n_atom, self.n_dim)
        direct_grad1 = ((func(Xn_a, *func_args, **func_kwargs) - func(Xn_b, *func_args, **func_kwargs)) /
                        (2 * ds)).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
        if self.is_concat_X: direct_grad1 = th.sum(direct_grad1 * self.had_converge_mask, keepdim=True)
        a: th.Tensor = (y1 <= (y0 + rho * steplength * (grad.mT @ p)))  # descent cond
        b = (direct_grad1 > beta * direct_grad0)  # curve cond

        return a, b, y1, direct_grad0, direct_grad1

    def _backtrack_update(self, steplength_in, i_: int):
        # applied shrinking
        steplength_out = (steplength_in - (steplength_in - 1e-4)*((i_ + 1)/self.maxiter)**self.factor)
        # update factor. If not converge, decrease the factor.
        return steplength_out

    def __call__(
            self,
            func: Any | nn.Module,
            grad_func: Any | nn.Module,
            X0: th.Tensor,
            y0: th.Tensor,
            grad: th.Tensor,
            p: th.Tensor,
            steplength: th.Tensor|float,
            is_grad_func_contain_y: bool,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            converge_mask: th.Tensor|None = None  # the mask of the batch that has been converged has the same shape as y0
    ) -> th.Tensor:
        self.had_converge_mask = ~converge_mask if converge_mask is not None else th.ones_like(y0)
        self.steplength = steplength
        if func_kwargs is None:
            func_kwargs = dict()
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        self.n_batch, self.n_atom, self.n_dim = X0.shape
        y0 = y0.unsqueeze(-1).unsqueeze(-1)
        # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
        if self.n_batch != y0.shape[0]:
            if self.n_batch == 1:
                y0 = th.sum(y0 * self.had_converge_mask, dim=(0, -1), keepdim=True)
                self.is_concat_X = True
            else:
                raise RuntimeError(f'Batch size of X ({self.n_batch}) and y ({y0.shape[0]}) do not match.')
        else:
            self.is_concat_X = False
        # reformat input
        grad = th.atleast_3d(grad)
        p = th.atleast_3d(p)

        self.device = X0.device
        if isinstance(self.steplength, float):
            steplength = th.full((self.n_batch, 1, 1), self.steplength, device=self.device)
        elif self.steplength.shape != (self.n_batch, 1, 1):
            raise RuntimeError(f'Expected a steplength with shape {(self.n_batch, 1, 1)}, but got {self.steplength.shape}')
        is_converge = False
        _steplength = steplength
        if self.method == 'Backtrack':  # Adaptive backtrack
            self.factor = max(self.factor, 0.2)
            with th.no_grad():
                for i in range(self.maxiter):
                    mask = self._Armijo_cond(func, X0, y0, grad, p, _steplength, rho=0.05, func_args=func_args, func_kwargs=func_kwargs)
                    if th.all(mask):
                        is_converge = True
                        break
                    # A kind of 'slow' backtrack.
                    _steplength = th.where(mask, _steplength, self._backtrack_update(steplength, i))

                if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
                return th.where(_steplength > 1.e-4, _steplength, 1.e-4)  # (n_batch, 1, 1)

        elif self.method == '_raise_backtrack':  # advance & retreat algo.
            self.factor = max(self.factor, 0.2)
            with th.no_grad():
                for i in range(self.maxiter):
                    mask = self._inv_Armijo_cond(func, X0, y0, grad, p, _steplength, rho=0.05, func_args=func_args, func_kwargs=func_kwargs)
                    if th.all(mask):
                        is_converge = True
                        break
                    # A kind of 'slow' backtrack.
                    _steplength = th.where(mask, _steplength, self._backtrack_update(steplength, i))

                if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
                return th.where(_steplength > 1.e-4, _steplength, 1.e-4)  # (n_batch, 1, 1)

        elif (self.method == 'Wolfe') or (self.method == 'NWolfe'):  # 2-points interpolation algo.
            a1 = th.zeros_like(_steplength)  # init min step length
            direct_grad0 = grad.mT @ p
            self.factor = max(self.factor, 0.2)
            if self.method == 'Wolfe':
                wolfCond = self._Wolfe_cond
            else:
                wolfCond = self._NWolfe_cond

            cumulative_mask = th.full_like(_steplength, False, dtype=th.bool)
            stored_values = th.zeros_like(_steplength)
            for i in range(self.maxiter):
                armijo_mask, curve_mask, y1, direct_grad1 = (
                    wolfCond(
                        func, grad_func, X0, y0, direct_grad0, p, _steplength, is_grad_func_contain_y, rho=0.05,
                        func_args=func_args, func_kwargs=func_kwargs, grad_func_args=grad_func_args, grad_func_kwargs=grad_func_kwargs
                    )
                )
                if th.all(armijo_mask * curve_mask):
                    is_converge = True
                    break

                with th.no_grad():
                    # Armijo cond. judge; interpolation step
                    interp2 = a1 - (direct_grad0 * (_steplength - a1) ** 2) / (2 * (y1 - y0 - direct_grad0 * (_steplength - a1)))  # 2nd interp
                    bt = (_steplength - (steplength - 1e-4) * (1 / self.maxiter))  # backtrack
                    interval_mask = (interp2 <= 0.) + (interp2 > 2 * self.steplength)
                    shrink_step = th.where(
                        interval_mask,
                        bt,
                        interp2
                    )
                    _steplength_temp = th.where(
                        armijo_mask,
                        _steplength,
                        shrink_step
                    )
                    # save steplength that 1st time satisfied Armijo condition.
                    first_time_mask = armijo_mask * (~cumulative_mask)
                    stored_values = th.where(first_time_mask, _steplength_temp, stored_values)
                    cumulative_mask = cumulative_mask + armijo_mask
                    # Wolfe curve. cond. judge; extrapolation
                    # 3rd extrapolation
                    exterp3 = _steplength + (direct_grad1 * (_steplength - a1)) / (direct_grad0 - direct_grad1)
                    linexterp = _steplength + (2 * self.steplength - _steplength)/self.maxiter
                    interval_mask2 = (exterp3 <= 0.) + (exterp3 > 2 * self.steplength)
                    extend_step = th.where(
                        interval_mask2,
                        linexterp,
                        exterp3
                    )
                    _steplength_temp = th.where(
                        armijo_mask * (~curve_mask),
                        extend_step,
                        _steplength_temp
                    )

                    # minimum steplength
                    _steplength = th.where(
                        _steplength_temp < 1e-4,
                        1e-4,
                        _steplength_temp
                    )

            if not is_converge:
                warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
                _steplength = th.where(
                    cumulative_mask,
                    stored_values,
                    1e-4
                )

            return _steplength  # (n_batch, 1, 1)

        elif self.method == '2PT_':
            a1 = th.zeros_like(_steplength)  # init min step length
            a2 = 2. * _steplength  # init max step length
            dy1 = grad.mT @ p  # (n_batch, 1, n_atom*n_dim) @ (n_batch, n_atom*n_dim, 1) -> (n_batch, 1, 1)
            Xn = X0 + _steplength * p.view(self.n_batch, self.n_atom, self.n_dim)  # (n_batch, n_atom, n_dim)
            y1 = func(Xn, *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)
            a = -dy1 * _steplength - y0 + y1
            a_mask = a > 0.
            if self.is_concat_X:
                y1 = th.sum(y1, keepdim=True)
            if th.all(a_mask):
                # if all coeff. of quadratic term > 0, interpolations
                _steplength = (- dy1 * _steplength ** 2) / (2 * a)
            else:
                # else extrapolations for points with quadratic term < 0.
                # forward finite difference to calc. dev of x2, 3 points in fact.
                ds = 1e-4
                Xn_a = Xn + ds * p.view(self.n_batch, self.n_atom, self.n_dim)
                dy2 = ((func(Xn_a, *func_args, **func_kwargs) - y1) / ds).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
                _steplength = th.where(
                    a > 0.,
                    (- dy1 * _steplength ** 2) / (2 * a),
                    _steplength + (dy2 * (_steplength - a1)) / (dy1 - dy2)
                )

            return th.where(_steplength > 2 * steplength, 0.05, _steplength)  # (n_batch, 1, 1)

        elif self.method == '2PT':
            a1 = th.zeros_like(_steplength)  # init min step length
            a2 = 2. * _steplength  # init max step length
            dy1 = grad.mT @ p  # (n_batch, 1, n_atom*n_dim) @ (n_batch, n_atom*n_dim, 1) -> (n_batch, 1, 1)
            Xn = X0 + _steplength * p.view(self.n_batch, self.n_atom, self.n_dim)  # (n_batch, n_atom, n_dim)
            y1 = func(Xn, *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)
            a = -dy1 * _steplength - y0 + y1
            a_mask = a > 0.
            if self.is_concat_X:
                y1 = th.sum(y1, keepdim=True)
            _steplength = (- dy1 * _steplength ** 2) / (2 * a)
            _steplength = th.where(
                (_steplength < 1e-6) * (_steplength > 2 * steplength),
                0.05 * steplength,
                _steplength
            )

            return _steplength  # (n_batch, 1, 1)

        elif self.method == '3PT':
            # cubic interpolation search. points: 0, dy0, mid_step, step
            step_mid = 0.5 * _steplength
            dy1 = grad.mT @ p  # (n_batch, 1, n_atom*n_dim) @ (n_batch, n_atom*n_dim, 1) -> (n_batch, 1, 1)
            Xn_mid = X0 + step_mid * p.view(self.n_batch, self.n_atom, self.n_dim)
            y_mid = func(Xn_mid, *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)  # (n_batch, )
            Xn = X0 + _steplength * p.view(self.n_batch, self.n_atom, self.n_dim)  # (n_batch, n_atom, n_dim)
            y1 = func(Xn, *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)
            if self.is_concat_X:
                y_mid = th.sum(y_mid, keepdim=True)
                y1 = th.sum(y1, keepdim=True)
            # Coefficients
            a = (dy1 * step_mid ** 2 * _steplength - dy1 * step_mid * _steplength ** 2 + step_mid ** 2 * y0
                 - _steplength ** 2 * y0 + _steplength ** 2 * y_mid - step_mid ** 2 * y1) / (
                        step_mid ** 2 * _steplength ** 2 * (step_mid - _steplength))
            b = (- dy1 * step_mid ** 3 * _steplength + dy1 * step_mid * _steplength ** 3 - step_mid ** 3 * y0
                 + _steplength ** 3 * y0 - _steplength ** 3 * y_mid + step_mid ** 3 * y1) / (
                        step_mid ** 2 * _steplength ** 2 * (step_mid - _steplength))
            c = (dy1 * step_mid ** 3 * _steplength ** 2 - dy1 * step_mid ** 2 * _steplength ** 3) / (
                    step_mid ** 2 * _steplength ** 2 * (step_mid - _steplength))
            a = 3 * a
            b = 2 * b
            delta_ = b ** 2 - 4 * a * c
            step1 = (-b + th.sqrt(delta_)) / (2 * a + 1e-10)
            step2 = (-b - th.sqrt(delta_)) / (2 * a + 1e-10)
            _steplength = th.where(
                delta_ >= 0,
                th.where(
                    2 * a * step1 + b > 0.,
                    step1,
                    step2
                ),
                _steplength
            )
            # if steplength < 0 or > 2 * steplength, use a fixed steplength.
            _steplength = th.where(
                (_steplength < 1e-6) * (_steplength > 2 * steplength),
                0.1 * steplength,
                _steplength
            )

            return _steplength  # (n_batch, 1, 1)

        elif self.method == 'Golden':  # golden section method
            with th.no_grad():
                _steplength1 = th.zeros_like(steplength)
                _steplength2 = steplength
                GOLDEN_SEC = 0.6180339887498948482
                is_converge = False
                is_get_inteval = False
                Xn = X0
                f1 = y0
                # Search Interval
                for _ in range(self.maxiter):
                    f2 = func(Xn + _steplength2 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(
                        -1)  # (n_batch, )
                    mask_interval = (f2 > f1)
                    if th.all(mask_interval):
                        is_get_inteval = True
                        break
                    _steplength2 = th.where(mask_interval, _steplength2, (1. + self.factor) * _steplength2)
                # Search Point
                if is_get_inteval:  # if Not get a suitable interval, steplength would keep the input value.
                    _steplength_m1 = _steplength1 + (_steplength2 - _steplength1) * (1. - GOLDEN_SEC)  # *0.382
                    _steplength_m2 = _steplength1 + (_steplength2 - _steplength1) * GOLDEN_SEC  # *0.618
                    f1 = func(Xn + _steplength_m1 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    f2 = func(Xn + _steplength_m2 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    for _ in range(self.maxiter):
                        if th.max(th.abs(_steplength1 - _steplength2)) < self.linesearch_thres:
                            is_converge = True
                            break
                        cond = (f1 <= f2)
                        cond_x = cond.unsqueeze(-1).unsqueeze(-1)
                        (_steplength1, _steplength2,
                         _steplength_m1, _steplength_m2) = (th.where(cond_x, _steplength1, _steplength_m1),
                                                            th.where(cond_x, _steplength_m2, _steplength2),
                                                            th.where(cond_x, _steplength_m2 - (_steplength_m2 - _steplength1) * GOLDEN_SEC,
                                                                     _steplength_m2),
                                                            th.where(cond_x, _steplength_m1,
                                                                     _steplength_m1 + (_steplength2 - _steplength_m1) * GOLDEN_SEC))
                        (f1, f2) = (
                            th.where(cond, func(Xn + _steplength_m1 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs), f2),
                            th.where(cond, f1, func(Xn + _steplength_m2 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)))
                    steplength = (_steplength1 + _steplength2) / 2
                else:
                    warnings.warn('linesearch did not find a suitable interval. The last steplength would be used.', RuntimeWarning)
                    is_converge = True
                    steplength = _steplength2

            if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
            return steplength  # (n_batch, 1, 1)

        elif self.method == 'Newton':
            with th.no_grad():
                dx = 1e-3
                Xn = X0
                _steplength = steplength
                for _ in range(self.maxiter):
                    f0 = func(Xn + _steplength * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    fpd = func(Xn + (_steplength + 2 * dx) * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    fsd = func(Xn + (_steplength - 2 * dx) * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    dev1 = (fpd - fsd) / (4 * dx)  # f'
                    dev2 = (fpd + fsd - 2 * f0)  # f"
                    newtn = - ((fpd - fsd) * dx) / dev2
                    # if f" <= 0, use grad. desc., else d = - f'/f"  # (n_batch, )
                    newton_direct = th.where((dev2 <= 1e-10) + (newtn >= 2 * steplength.squeeze(-1, -2)), - 0.01 * dev1, newtn)
                    mask = th.abs(dev1) < self.linesearch_thres
                    if th.all(mask):
                        is_converge = True
                        break
                    _steplength = th.where(mask.unsqueeze(-1).unsqueeze(-1), _steplength,
                                           _steplength + self.factor * newton_direct.unsqueeze(-1).unsqueeze(-1))

                if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
                return th.where(th.isnan(_steplength), 0.05, _steplength)  # (n_batch, 1, 1)

        elif self.method == 'None':
            return steplength
        else:
            raise NotImplementedError(f'The input linear search method {self.method} did not implemented.')
