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
        constr_func: Callable, a tuple of Python functions as the constraint functions s_k(X) that map R^n -> R^k. It takes one or more arguments, one of which must be a Tensor, and returns one Tensor with shape (k, ). `None` for identity function. see example below.
        constr_val: Callable[th.Tensor[1], th.Tensor] | th.Tensor, the constraint value of `constr_func`, i.e., constraints are `constr_func(X) = constr_val`.
        By defining it as a callable constr_val = constr_val(t) where `t` is a scalar Tensor, it can be set to the time-dependent constraints.
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
            constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor = None,
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
        self.time_now_tens = th.tensor(self.time_now, dtype=th.float32, device=self.device)
        if constr_func is None:
            constr_func = lambda X: 0.
        if isinstance(constr_val, th.Tensor):
            self.is_const_constr = True
            self.constr_val_raw = None
            self.constr_val_now = constr_val.to(self.device)
        elif not isinstance(constr_val, Callable):
            raise TypeError(f'Expected `constr_val` is Callable or torch.Tensor, but got {type(constr_val)}')
        else:
            self.is_const_constr = False
            self.constr_val_raw = constr_val
            self.constr_val_now = constr_val(self.time_now_tens)

        if not isinstance(constr_func, Callable):
            raise TypeError(f"`constr_func` must consist of callable, but got {type(constr_func)}")
        self.constr_func = constr_func
        self.Q = th.tensor([], device=self.device, dtype=th.float32)  # Q(n_batch, n_atoms * n_dim, n_constr)
        self.jac = None # Jacobian Matrix
        self.R = th.tensor([], device=self.device, dtype=th.float32)  # R(n_batch, n_constr, n_constr)
        self.q = th.tensor([], device=self.device, dtype=th.float32)  # (n_constr, ), solution of `R^T q = d/dt constr_val(t)`
        self.sqrtM = None  # M^1/2, (n_batch, n_atoms, n_dim)
        self.negsqrtM = None  # M^-1/2
        self.max_proj_iter = 10
        self.proj_thres = 1e-6

    def initialize(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            masses: th.Tensor,
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            is_fix_mass_center: bool = False
    ):
        self.sqrtM = th.sqrt(masses)  # M^1/2, (n_batch, n_atoms, n_dim)
        self.negsqrtM = 1 / th.sqrt(masses)  # M^-1/2
        jac, y = self._jacobian(X)
        if y.ndim != 2:
            raise ValueError(f'`constr_func` must return a 2D tensor of shape (n_batch, n_constr), but got {y.shape}.')
        self._do_qr(jac)
        self.Q_tmp = self.Q.clone()
        ProjV = self._project1(V_init)
        Ek = th.sum(
            masses * V_init ** 2,
            dim=(-2, -1),
            keepdim=True
        )
        Ek_p = th.sum(
            masses * ProjV ** 2,
            dim=(-2, -1),
            keepdim=True
        )
        V_init.copy_(Ek/Ek_p * ProjV)
        self.n_reduce += jac.shape[1]
        # recalculate target Ek under constraints
        _, n_atom, n_dim = X.shape
        self.EK_TARGET = 0.5 * (n_dim * (n_atom - 1) - self.n_reduce) * 8.617333262145e-5 * self.T_init
        # debug
        self._debug_X_check = list()
        self._debug_V_check = list()

    def _constr_func_wrapped(self, X, constr_val_now):
        """
        Manage the constraint functions.
        Converting constraint functions into s_k(X) = constr_func(X) - constr_val, thereby constraints are s_k(X) = 0.
        Returns:

        """
        y = self.constr_func(X) - constr_val_now
        y = th.atleast_1d(y)
        return y, y  # repeat the output to use `has_aux` in th.func.jacrev to return the function values

    def _constr_val_wrapped(self, t):
        """
        Manage the time-dependent constraint values.
        Args:
            t:

        Returns:

        """
        if self.is_const_constr:
            return None
        else:
            y = self.constr_val_raw(t)
            if isinstance(y, (Tuple, List)):
                y = th.stack(y)

            return y, y

    def _update_constr(self):
        """
        Update constraint function value.
        Returns:

        """
        if not self.is_const_constr:
            self.time_now_tens.copy_(self.time_now)
            self.d_constr, self.constr_val_now = th.func.jacrev(self._constr_val_wrapped, has_aux=True)(self.time_now_tens)  #  d s_k(r)/dt

    def _jacobian(self, X: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the Jacobian of all constrains.
        Args:
            X:

        Returns: Jacobian (n_batch, n_constr, n_atoms, n_dim), and the constraint values.

        """
        self._update_constr()
        with th.enable_grad():
            jac, y = th.vmap(th.func.jacrev(self._constr_func_wrapped, has_aux=True))(X, self.constr_val_now)
        self.jac = jac
        return jac, y

    def _do_qr(self, jac: th.Tensor):
        """
        Do QR decomposition and save/update self.Q.
        Args:
            jac: Jacobian matrix of s_k at X.

        Returns: None

        """
        # weighted jac: J_w = J @ M^-1/2,
        # (n_batch, n_constr, n_atoms, n_dim) * (n_batch, 1, n_atoms, n_dim) -flatten-> (n_batch, n_constr, n_atoms * n_dim)
        _jac = th.flatten(jac * self.negsqrtM.unsqueeze(1), -2, -1)
        _jacT = _jac.mT.contiguous()
        th.linalg.qr(_jacT, out=(self.Q, self.R))  # Q(n_batch, n_atoms * n_dim, n_constr) , R(n_batch, n_constr, n_constr)

    def _project1(self, X:th.Tensor) -> th.Tensor:
        """
        Compute the projector for 1st-order quantity (e.g. velocity) of all constrains by QR decomposition.
        for Jacobian matrix J of constraints s_k at X, let J^T M^-1/2 = QR, thus s_k(X + v * dt) = s_k(X) + R^T Q^T M^1/2 v * dt + O(dt^2)
        define the projector: P = M^-1/2 Q_2 Q_2^T M^1/2 = M^-1/2 (I - Q_1 Q_1^T) M^1/2, where Q = [Q_1, Q_2], Q_1 is from the financial QRD.
        Args:
            X: the input tensor of shape (n_batch, n_atom, n_dim). It might be coordinates, velocity, and higher deviations.

        Returns: th.Tensor, the projected X.

        """
        n_batch, n_atoms, n_dim = X.shape
        sqrtM = self.sqrtM  # M^1/2, (n_batch, n_atoms, n_dim)
        negsqrtM = self.negsqrtM  # M^-1/2
        # P = M^-1/2 @ (I - Q Q^T) @ M^1/2 = I - M^-1/2 @ Q @ Q^T @ M^1/2
        Q = self.Q  # Q(n_batch, n_atoms * n_dim, n_constr)
        Px = (sqrtM * X).reshape(n_batch, n_atoms * n_dim, 1).contiguous()  # (n_batch, n_atoms * n_dim, 1)
        Px = (Q @ (Q.mT.contiguous() @ Px)).reshape(n_batch, n_atoms, n_dim)  # (n_batch, n_atoms, n_dim)
        Px.mul_(negsqrtM)
        Px = X - Px
        # manifold veloc. correction
        if not self.is_const_constr:  # time-dependent constraints
            th.linalg.solve_triangular(self.R.mT.contiguous(), self.d_constr, upper=False, out=self.q)
            Px.add_(self.negsqrtM * (self.Q @ self.q.unsqueeze(-1)).reshape(n_batch, n_atoms, n_dim))

        return Px

    def _project2(self, X:th.Tensor, V:th.Tensor) -> th.Tensor:
        """
        Continuously project the Jacobian of all constrains to the exact manifold (by Newton-like iteration).
        Update X in-place.
        Args:
            X: the input tensor of shape (n_batch, n_atom, n_dim). It might be coordinates, velocity, and higher deviations.

        Returns: th.Tensor, the constraint forces \lambda.

        """
        n_batch, n_atoms, n_dim = X.shape
        ### first QRF for velocity ###
        # update Jacobian
        jac, y = self._jacobian(X)
        self._do_qr(jac)
        # define the constr. force
        Fc = th.zeros(n_batch, y.shape[1], 1, device=self.device)
        #self.Q_tmp.copy_(self.Q)  # to use to update velocity constraints
        # print
        constr_err = th.max(th.abs(y)).item()
        if self.verbose > 0:
            self.logger.info(f' 0 Constraint errors are now: {constr_err:.4e}')
        # threshold
        if constr_err < self.proj_thres:
            return Fc
        _lamb = th.linalg.solve_triangular(self.R.mT.contiguous(), y.unsqueeze(-1), upper=False)  # (n_batch, n_constr, 1)
        # the constr forces
        Fc += - th.linalg.solve_triangular(self.R, _lamb, upper=True)
        # (n_batch, n_atoms, n_dim) * (n_batch, n_atoms * n_dim, n_constr) @ (n_batch, n_constr, 1)
        corr = self.negsqrtM * (self.Q @ _lamb).reshape(n_batch, n_atoms, n_dim)
        X.add_(corr, alpha=-1.)
        #V.add_(corr/self.time_step, alpha=-1.)
        ### manually unrolling for first 3 steps ###
        jac, y = self._jacobian(X)
        constr_err = th.max(th.abs(y)).item()
        self._do_qr(jac)
        if self.verbose > 0:
            self.logger.info(f' 1 Constraint errors are now: {constr_err:.4e}')
        if constr_err < self.proj_thres:
            return Fc
        _lamb = th.linalg.solve_triangular(self.R.mT.contiguous(), y.unsqueeze(-1), upper=False)
        # the constr forces
        Fc += - th.linalg.solve_triangular(self.R, _lamb, upper=True)
        corr = self.negsqrtM * (self.Q @ _lamb).reshape(n_batch, n_atoms, n_dim)
        X.add_(corr, alpha=-1.)
        #V.add_(corr/self.time_step, alpha=-1.)

        jac, y = self._jacobian(X)
        constr_err = th.max(th.abs(y)).item()
        self._do_qr(jac)
        if self.verbose > 0:
            self.logger.info(f' 2 Constraint errors are now: {constr_err:.4e}')
        if constr_err < self.proj_thres:
            return Fc
        _lamb = th.linalg.solve_triangular(self.R.mT.contiguous(), y.unsqueeze(-1), upper=False)
        Fc += - th.linalg.solve_triangular(self.R, _lamb, upper=True)
        corr = self.negsqrtM * (self.Q @ _lamb).reshape(n_batch, n_atoms, n_dim)
        X.add_(corr, alpha=-1.)
        #V.add_(corr/self.time_step, alpha=-1.)

        # if still not converge, entering loop
        for i in range(3, self.max_proj_iter):
            # update Jacobian
            jac, y = self._jacobian(X)
            # print
            constr_err = th.max(th.abs(y)).item()
            self._do_qr(jac)
            if self.verbose > 0:
                self.logger.info(f'{i: < 3d} Constraint errors are now: {constr_err:.4e}')
            # threshold
            if constr_err < self.proj_thres:
                return Fc
            # P = M^1/2 @ (I - Q Q^T) @ M^-1/2
            # Q(n_batch, n_atoms * n_dim, n_constr)
            _lamb = th.linalg.solve_triangular(self.R.mT.contiguous(), y.unsqueeze(-1) , upper=False) # (n_batch, n_constr, 1)
            # constr. forces lambda
            Fc += - th.linalg.solve_triangular(self.R, _lamb, upper=True)
            # (n_batch, n_atoms, n_dim) * (n_batch, n_atoms * n_dim, n_constr) @ (n_batch, n_constr, 1)
            corr = self.negsqrtM * (self.Q @ _lamb).reshape(n_batch, n_atoms, n_dim)
            X.add_(corr, alpha=-1.)
            #V.add_(corr/self.time_step, alpha=-1.)

        # if not converged
        self.logger.warning("Projection of X to the manifold is not converged.")
        return Fc

    def _project1_std(self, X:th.Tensor) -> th.Tensor:
        """
        Compute the projector for 1st-order quantity (e.g. velocity) of all constrains by QR decomposition.
        for Jacobian matrix J of constraints s_k at X, let J^T M^-1/2 = QR, thus s_k(X + v * dt) = s_k(X) + R^T Q^T M^1/2 v * dt + O(dt^2)
        define the projector: P = M^1/2 Q_2 Q_2^T M^-1/2 = M^1/2 (I - Q_1 Q_1^T) M^-1/2, where Q = [Q_1, Q_2], Q_1 is from the financial QRD.
        Args:
            X: the input tensor of shape (n_batch, n_atom, n_dim). It might be coordinates, velocity, and higher deviations.

        Returns: th.Tensor, the projected X.

        """
        n_batch, n_atoms, n_dim = X.shape
        sqrtM = self.sqrtM  # M^1/2, (n_batch, n_atoms, n_dim)
        negsqrtM = self.negsqrtM  # M^-1/2
        # P = M^-1/2 @ (I - Q Q^T) @ M^1/2 = I - M^-1/2 @ Q @ Q^T @ M^1/2
        Q = self.Q  # Q(n_batch, n_atoms * n_dim, n_constr)
        Px = th.diag_embed(self.sqrtM.flatten(-2, -1), dim1=1, dim2=2) @ X.reshape(n_batch, n_atoms * n_dim, 1).contiguous()  # (n_batch, n_atoms * n_dim, 1)
        Px = (Q @ (Q.mT.contiguous() @ Px))  # (n_batch, n_atoms, n_dim)
        Px = th.diag_embed(self.negsqrtM.flatten(-2, -1), dim1=1, dim2=2) @ Px
        Px = X - Px.reshape(n_batch, n_atoms, n_dim)
        # manifold veloc. correction
        if not self.is_const_constr:  # time-dependent constraints
            th.linalg.solve_triangular(self.R.mT.contiguous(), self.d_constr, upper=False, out=self.q)
            Px.add_(self.negsqrtM * (self.Q @ self.q.unsqueeze(-1)).reshape(n_batch, n_atoms, n_dim))

        return Px

    def _project2_std(self, X:th.Tensor) -> None:
        """
        Compute position corrections by directly solving JM^-1J^T q = s(X)
        Args:
            X:

        Returns:

        """
        n_batch, n_atoms, n_dim = X.shape
        for i in range(0, self.max_proj_iter):
            # update Jacobian
            jac, y = self._jacobian(X)  # jac: (n_batch, n_constr, n_atom, n_dim)
            # print
            constr_err = th.max(th.abs(y)).item()
            lamb = 0.
            if self.verbose > 0:
                self.logger.info(f'{i: < 3d} Constraint errors are now: {constr_err:.4e}')
            # threshold
            if constr_err < self.proj_thres:
                return
            # solve J M^-1 J^T q = s
            Z = jac*self.negsqrtM  # TODO
            # Q(n_batch, n_atoms * n_dim, n_constr)
            corr = th.linalg.solve_triangular(self.R.mT.contiguous(), y.unsqueeze(-1) , upper=False) # (n_batch, n_constr, 1)
            # (n_batch, n_atoms, n_dim) * (n_batch, n_atoms * n_dim, n_constr) @ (n_batch, n_constr, 1)
            corr = self.negsqrtM * (self.Q @ corr).reshape(n_batch, n_atoms, n_dim)
            X.add_(corr, alpha=-1.)
            #V.add_(corr/self.time_step, alpha=-1.)

        # if not converged
        self.logger.warning("Projection of X to the manifold is not converged.")