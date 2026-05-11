""" functions of gradient-related calculations. """
#  Copyright (c) 2025.10.17, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: jac_and_hvp.py
#  Environment: Python 3.12

import torch as th
from typing import Callable, Tuple, Dict
import BUCToolkit.utils.index_ops as index_ops


def jvp_wrapper(func: Callable, X:Tuple[th.Tensor, ...], v:Tuple[th.Tensor, ...], has_aux=False):
    """
    A function to map func, X, v to J_func_X @ v, i.e., the jvp.
    Args:
        func:
        X:
        v:
        has_aux:

    Returns:

    """
    if has_aux:
        _, y, aux = th.func.jvp(func, X, v, has_aux=has_aux)
        return y, aux
    else:
        _, y = th.func.jvp(func, X, v, has_aux=has_aux)
        return y

def bjvp(func: Callable, X:Tuple[th.Tensor, ...], v:Tuple[th.Tensor, ...], has_aux=False):
    """
    Batched Jacobian-vector product.
    Args:
        func: The function to calculate the batched JVP.
        X: the input of the function.
        v: the vector that Jacobian product to.
        has_aux: whether to output auxiliary variables.

    Returns: Tuple[th.Tensor, th.Tensor] | Tuple[th.Tensor, th.Tensor, th.Tensor]
        if `has_aux` is True, return function values y, jvp, aux
        otherwise, return function values y and jvp.

    """
    if has_aux:
        val, jvp, aux = th.vmap(th.func.jvp, (None, (0, ), (0, )))(func, X, v, has_aux=has_aux)
        return val, jvp, aux
    else:
        val, jvp = th.vmap(th.func.jvp, (None, (0,), (0,)))(func, X, v, has_aux=has_aux)
        return val, jvp

def bhvp(func: Callable, X:Tuple[th.Tensor, ...], v:Tuple[th.Tensor, ...], has_aux:bool=False):
    """
    Batched Hessian-vector product.
    Args:
        func: The function to calculate the batched HVP.
        X: the input of the function.
        v: the vector that Hessian product to.
        has_aux: whether to output auxiliary variables.

    Returns: th.Tensor | Tuple[th.Tensor, th.Tensor], if `has_aux` is True: hvp, jvp; otherwise, hvp.

    """
    if has_aux:
        y_, jvp_ = th.vmap(th.func.jacrev(jvp_wrapper, argnums=1, has_aux=has_aux), (None, (0,), (0,), None))(func, X, v, has_aux)
        return y_[0], jvp_
    else:
        y_ = th.vmap(th.func.jacrev(jvp_wrapper, argnums=1, has_aux=has_aux), (None, (0,), (0,), None))(func, X, v, has_aux)
        return y_[0]

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
