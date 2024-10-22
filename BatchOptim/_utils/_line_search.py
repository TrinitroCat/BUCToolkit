

import warnings
# ruff: noqa: E701, E702, E703, F401
from typing import Any, Literal, Sequence, Tuple

import torch as th
from torch import nn


class _LineSearch:
    def __init__(
            self,
            method:Literal['Backtrack', 'Wolfe', 'Golden', 'Newton', 'None'],
            maxiter:int=10,
            thres:float=0.02,
            steplength=0.5,
            factor:float=0.05,
    ) -> None:
        self.method = method
        self.maxiter = maxiter
        self.linesearch_thres = thres
        self.steplength = steplength
        self.factor = factor
        pass

    def _Armijo_cond(self, func:Any|nn.Module, X0:th.Tensor, y0:th.Tensor, grad:th.Tensor, p:th.Tensor, steplength:th.Tensor, rho:float=0.05,
                     func_args:Sequence=tuple(), func_kwargs=None) -> th.Tensor:
        if func_kwargs is None:
            func_kwargs = dict()
        with th.no_grad():
            y1 = func(X0 + steplength * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs).unsqueeze(-1).unsqueeze(-1)
            if self.is_concat_X: y1 = th.sum(y1, keepdim=True)
        a:th.Tensor = (y1 <= (y0 + rho * steplength * (grad.mT@p)))  # Tensor[bool]: (n_batch, ) = (n_batch, ) + (n_batch, 1, 1) * (n_batch, ) * (n_batch, )
        return a  # (n_batch, 1, 1)
    
    def _Wolfe_cond(self, func:Any|nn.Module, X0:th.Tensor, y0:th.Tensor, grad:th.Tensor, p:th.Tensor, steplength:th.Tensor, rho:float=0.5,
                    func_args:Sequence=tuple(), func_kwargs=None) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Weak Wolfe-Powell condition.
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

        """
        if func_kwargs is None:
            func_kwargs = dict()
        Xn = X0 + steplength * p.view(self.n_batch, self.n_atom, self.n_dim) # (n_batch, n_atom, n_dim)
        Xn.requires_grad_()
        y1 = (func(Xn, *func_args, **func_kwargs)).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
        if self.is_concat_X: y1 = th.sum(y1, keepdim=True)
        dy1 = th.autograd.grad(y1, Xn, th.ones_like(y1))[0] # (n_batch, n_atom, n_dim)
        # detach grad graph
        dy1 = dy1.detach()
        y1 = y1.detach()
        Xn = Xn.detach()
        dy1 = dy1.flatten(-2, -1).unsqueeze(-1) #(n_batch, n_atom*n_dim, 1)
        with th.no_grad():
            direct_grad0 = grad.mT @ p
            direct_grad1 = dy1.mT @ p
            a:th.Tensor = (y1 <= (y0 + rho * steplength * (grad.mT@p))) # descent cond
            b = (direct_grad1 > rho * direct_grad0) # curve cond

        return a, b, y1, direct_grad0, direct_grad1

    def __call__(
            self,
            func:Any|nn.Module,
            X0:th.Tensor,
            y0:th.Tensor,
            grad:th.Tensor,
            p:th.Tensor,
            func_args:Sequence=tuple(),
            func_kwargs=None,
            batch_indices=None
    ) -> th.Tensor:
        if func_kwargs is None:
            func_kwargs = dict()
        self.n_batch, self.n_atom, self.n_dim = X0.shape
        y0 = y0.unsqueeze(-1).unsqueeze(-1)
        # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
        if self.n_batch != y0.shape[0]:
            if self.n_batch == 1:
                y0 = th.sum(y0, keepdim=True)
                self.is_concat_X = True
            else:
                raise RuntimeError(f'Batch size of X ({self.n_batch}) and y ({y0.shape[0]}) do not match.')
        else:
            self.is_concat_X = False

        self.device = X0.device
        steplength = th.full((self.n_batch, 1, 1), self.steplength, device=self.device)
        is_converge = False; _steplength = steplength
        if self.method == 'Backtrack':
            with th.no_grad():
                for i in range(self.maxiter):
                    mask = self._Armijo_cond(func, X0, y0, grad, p, _steplength, rho=0.05, func_args=func_args, func_kwargs=func_kwargs)
                    if th.all(mask):
                        is_converge = True
                        break
                    # A kind of 'slow' backtrack.
                    _steplength = th.where(mask, _steplength, steplength*((self.maxiter-2 - i)**2/(self.maxiter-1)**2)**(1./self.factor))

                if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
                return th.where(_steplength > 1.e-4, _steplength, 1.e-4)  # (n_batch, 1, 1)
        
        elif self.method == 'Wolfe_old':  # advance & retreat algo.
            a = 0.5
            for i in range(self.maxiter):
                armijo_mask, curve_mask, y1, direct_grad0, direct_grad1 = (
                    self._Wolfe_cond(func, X0, y0, grad, p, _steplength, rho=0.05, func_args=func_args, func_kwargs=func_kwargs)
                )
                if th.all(armijo_mask * curve_mask):
                    is_converge = True
                    break
                _steplength = th.where(armijo_mask, _steplength, _steplength*self.factor)
                _steplength = th.where(armijo_mask*(~curve_mask), _steplength*(((1 - a)*self.factor + a)/self.factor), _steplength)

            if not is_converge: warnings.warn(f'line search did not converge in {self.maxiter} steps.', RuntimeWarning)
            return _steplength  # (n_batch, 1, 1) 

        elif self.method == 'Wolfe':  # 2-points interpolation algo.
            a1 = th.zeros_like(_steplength)  # init min step length
            a2 = 2. * _steplength  # init max step length
            for i in range(self.maxiter):
                armijo_mask, curve_mask, y1, direct_grad0, direct_grad1 = (
                    self._Wolfe_cond(func, X0, y0, grad, p, _steplength, rho=0.05, func_args=func_args, func_kwargs=func_kwargs)
                )
                if th.all(armijo_mask * curve_mask):
                    is_converge = True
                    break
                # Armijo cond. judge; interpolation
                with th.no_grad():
                    _steplength = th.where(armijo_mask,
                                           _steplength,
                                           a1 - (direct_grad0 * (_steplength - a1)**2)/(2 * (y1 - y0 - direct_grad0 * (_steplength - a1))))
                    # Wolfe curve. cond. judge; extrapolation
                    _steplength = th.where(armijo_mask * (~curve_mask),
                                           _steplength + (direct_grad1 * (_steplength - a1))/(direct_grad0 - direct_grad1),
                                           _steplength)

            if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
            return _steplength  # (n_batch, 1, 1)

        elif self.method == 'Golden': # golden section method
            with th.no_grad():
                _steplength1 = th.zeros_like(steplength); _steplength2 = steplength; GOLDEN_SEC = 0.6180339887498948482
                is_converge = False; is_get_inteval = False; Xn = X0
                f1 = y0
                # Search Interval
                for _ in range(self.maxiter):
                    f2 = func(Xn + _steplength2 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)  # (n_batch, )
                    mask_interval = (f2 > f1)
                    if th.all(mask_interval):
                        is_get_inteval = True
                        break
                    _steplength2 = th.where(mask_interval.unsqueeze(-1).unsqueeze(-1), _steplength2, (1. + self.factor)*_steplength2)
                # Search Point
                if is_get_inteval:  # if Not get a suitable interval, steplength would keep the input value.
                    _steplength_m1 = _steplength1 + (_steplength2 - _steplength1)*(1. - GOLDEN_SEC)  # *0.382
                    _steplength_m2 = _steplength1 + (_steplength2 - _steplength1)*GOLDEN_SEC         # *0.618
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
                                                        th.where(cond_x, _steplength_m2 - (_steplength_m2 - _steplength1)*GOLDEN_SEC, _steplength_m2), 
                                                        th.where(cond_x, _steplength_m1, _steplength_m1 + (_steplength2 - _steplength_m1)*GOLDEN_SEC))
                        (f1, f2) = (th.where(cond, func(Xn + _steplength_m1 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs), f2), 
                                    th.where(cond, f1, func(Xn + _steplength_m2 * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)))
                    steplength = (_steplength1 + _steplength2)/2
                else:
                    warnings.warn('linesearch did not find a suitable interval. The last steplength would be used.', RuntimeWarning)
                    is_converge = True
                    steplength = _steplength2
            
            if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
            return steplength  # (n_batch, 1, 1)

        elif self.method == 'Newton':
            with th.no_grad():
                dx = 5e-4; Xn = X0; _steplength = steplength
                for _ in range(self.maxiter):
                    f0  = func(Xn + _steplength * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    fpd = func(Xn + (_steplength + 2*dx) * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    fsd = func(Xn + (_steplength - 2*dx) * p.view(self.n_batch, self.n_atom, self.n_dim), *func_args, **func_kwargs)
                    dev1 = (fpd - fsd)/(4*dx)
                    dev2 = (fpd + fsd - 2*f0)
                    newtn = - ((fpd - fsd) * dx)/dev2
                    newton_direct = th.where((dev2 <= 1e-10) + (newtn >= 2*steplength.squeeze(-1,-2)), - 0.01*dev1, newtn)  # if f" <= 0, use grad. desc., else d = - f'/f"  # (n_batch, )
                    mask = th.abs(dev1) < self.linesearch_thres
                    if th.all(mask):
                        is_converge = True
                        break
                    _steplength = th.where(mask.unsqueeze(-1).unsqueeze(-1), _steplength, _steplength + self.factor * newton_direct.unsqueeze(-1).unsqueeze(-1))

                if not is_converge: warnings.warn(f'linesearch did not converge in {self.maxiter} steps.', RuntimeWarning)
                return th.where(th.isnan(_steplength), 0.05, _steplength)  # (n_batch, 1, 1)
            
        elif self.method == 'None':
            return steplength
        else:
            raise NotImplementedError('The input linear search method did not implemented.')