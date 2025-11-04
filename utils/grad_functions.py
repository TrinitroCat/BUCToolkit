""" functions of gradient-related calculations. """
#  Copyright (c) 2025.10.17, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: jac_and_hvp.py
#  Environment: Python 3.12

import torch as th
from typing import Callable, Tuple


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
