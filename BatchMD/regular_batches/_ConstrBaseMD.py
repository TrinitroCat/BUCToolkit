""" Molecular Dynamics base framework with constrains """

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVE.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import torch as th
from torch import nn

from ._BaseMD import _rBaseMD


class _rConstrBase(_rBaseMD):
    """
    Constrained Base BatchMD with regular batches
    micro canonical ensemble (NVE) molecular dynamics implemented via velocity Verlet algo. on the tangent space.

    Args:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        constr_func: Callable, a tuple of Python functions as the constraint functions s_k that map R^n -> R^k and s_k(X) == 0. It takes one or more arguments, one of which must be a Tensor, and returns one Tensor with shape (k, ). `None` for identity function. see example below.
        constr_val: th.Tensor,
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: str|torch.device, device that program rum on.
        verbose: int, control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.

    Examples for constrains:
        def constr_func(X):
            y = list()
            # X: shape(N, D), which means the batch dimension would NOT be considered in constraints calculation of X.
            # fix the distance between atoms (2, 4), (3, 7), (5, 8) into 1.5, 1.6, and 2.1, respectively.
            y.append(th.linalg.norm(X[[2, 3, 5]] - X[[4, 7, 8]], dim=-1) - th.tensor([1.5, 1.6, 2.1]))

            # fix the angle of atom7-atom5-atom8 and atom11-atom9-atom12 into 109.5 and 120 degrees, respectively.
            x1 = X[[5, 9]]
            x2 = X[[7, 11]]
            x3 = X[[8, 12]]
            y.append(
                (
                    th.sum((x2 - x1) * (x3 - x1))
                ) / (th.linalg.norm((x2 - x1)) * th.linalg.norm((x3 - x1))) - th.cos(th.deg2rad(th.tensor([109.5, 120])))
            )
            z = th.cat(y)
            return z

    Methods:
        run: run BatchMD.

    """
    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            constr_func: Callable | None = None,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        super().__init__(
            time_step,
            max_step,
            T_init,
            output_file,
            output_structures_per_step,
            device,
            verbose
        )
        if constr_func is None:
            constr_func = lambda X: X
        if not isinstance(constr_func, Callable):
            raise TypeError(f"`constr_func` must consist of callable, but got {type(constr_func)}")
        self.constr_func = constr_func
        self.Q = None  # Q(n_batch, n_atoms * n_dim, n_constr)
        self.sqrtM = None  # M^1/2, (n_batch, n_atoms, n_dim)
        self.negsqrtM = None  # M^-1/2

    def _constr_func_wrapped(self, X: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Just simply repeat the output of self.constr_func to use `has_aux` in th.func.jacrev to return the function values.
        Returns: y, y

        """
        y = self.constr_func(X)
        return y, y

    def _jacobian(self, X: th.Tensor) -> th.Tensor:
        """
        Compute the Jacobian of all constrains.
        Args:
            X:

        Returns:

        """
        with th.enable_grad():
            jac, y = th.vmap(th.func.jacrev(self._constr_func_wrapped, has_aux=True))(X)
        if y.ndim != 2:
            raise ValueError(f'`constr_func` must return a 2D tensor of shape (n_batch, n_constr), but got {y.shape}.')
        constr_err = th.max(y).item()
        if constr_err > 1e-4:
            self.logger.warning(f'WARNING: Too large constraint error: {constr_err:.4e}')
        self.logger.info(f'Max Constraint Error: {constr_err:.4e}')

        return jac

    def _do_qr(self, masses: th.Tensor, jac: th.Tensor):
        """
        Do QR decomposition and save/update self.Q.
        Args:
            masses: the masses tensor with the same shape as `X`.
            jac: Jacobian matrix of s_k at X.

        Returns: None

        """
        # weighted jac: J_w = J @ M^-1/2,
        # (n_batch, n_constr, n_atoms, n_dim) * (n_batch, 1, n_atoms, n_dim) -flatten-> (n_batch, n_constr, n_atoms * n_dim)
        _jac = th.flatten(jac * self.negsqrtM.unsqueeze(1), -2, -1)
        _jacT = _jac.mT.contiguous()
        Q = th.linalg.qr(_jacT).Q  # Q(n_batch, n_atoms * n_dim, n_constr) , R(n_batch, n_constr, n_constr)

        self.Q = Q

    def _projected(self, X:th.Tensor) -> th.Tensor:
        """
        Compute the projector of all constrains by QR decomposition.
        for Jacobian matrix J of constraints s_k at X, let J^T M^-1/2 = QR, thus s_k(X + v * dt) = s_k(X) + R^T Q^T M^1/2 v * dt + O(dt^2)
        define the projector: P = M^1/2 Q_2 Q_2^T M^-1/2 = M^1/2 (I - Q_1 Q_1^T) M^-1/2, where Q = [Q_1, Q_2], Q_1 is from the financial QRD.
        Args:
            X: the input tensor of shape (n_batch, n_atom, n_dim). It might be coordinates, velocity, and higher deviations.

        Returns: th.Tensor, the projected X.

        """
        n_batch, n_atoms, n_dim = X.shape
        sqrtM = self.sqrtM  # M^1/2, (n_batch, n_atoms, n_dim)
        negsqrtM = self.negsqrtM  # M^-1/2
        # P = M^1/2 @ (I - Q Q^T) @ M^-1/2
        Q = self.Q  # Q(n_batch, n_atoms * n_dim, n_constr)
        Px = (negsqrtM * X).reshape(n_batch, n_atoms * n_dim, 1).contiguous()  # (n_batch, n_atoms * n_dim, 1)
        Px.sub_(Q @ (Q.mT.contiguous() @ Px))  # (n_batch, n_atoms * n_dim, 1)
        Px = Px.reshape(n_batch, n_atoms, n_dim)
        Px.mul_(sqrtM)

        return Px


