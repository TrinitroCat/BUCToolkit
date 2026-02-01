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
import numpy as np

from ._BaseMD import _rBaseMD
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BM4Ckit.utils.grad_functions import bjvp, bhvp


class _rConstrBase(_rBaseMD):
    """
    Constrained Base Dynamics

    Args:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        constr_func: Callable, a tuple of Python functions as the constraint functions s_k(X) that map R^n -> R^k. It takes one or more arguments, one of which must be a Tensor, and returns one Tensor with shape (k, ). `None` for identity function. see example below.
        constr_val: Callable[th.Tensor[1], th.Tensor] | th.Tensor, the constraint value of `constr_func`, i.e., constraints are `constr_func(X) = constr_val`.
        By defining it as a callable constr_val = constr_val(t) where `t` is a scalar Tensor, it can be set to the time-dependent constraints.
        constr_threshold: float, the threshold of constraint convergence (error of manifold violation)
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
            constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor | None = None,
            constr_threshold: float = 1e-5,
            output_file: str | None = None,
            dump_path: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        super().__init__(
            time_step,
            max_step,
            T_init,
            output_file,
            dump_path,
            output_structures_per_step,
            device,
            verbose
        )
        self.time_now_tens = th.tensor(self.time_now, dtype=th.float32, device=self.device)
        if constr_func is None:
            constr_func = lambda X: th.tensor(0.)
        self._lazy_calc_constr_val = False  # use the value of constr_func(X_init) to determine constr_val (and fixed).
        if isinstance(constr_val, th.Tensor):
            self.is_const_constr = True
            self.constr_val_func_raw = None
            self.constr_val_now = constr_val.to(self.device)
        elif isinstance(constr_val, Callable):
            self.is_const_constr = False
            self.constr_val_func_raw = constr_val
            self.constr_val_now = constr_val(self.time_now_tens)
        elif constr_val is None:
            self.is_const_constr = True
            self.constr_val_func_raw = None
            self._lazy_calc_constr_val = True
        else:
            raise TypeError(f'Expected `constr_val` is Callable or torch.Tensor, but got {type(constr_val)}')

        if not isinstance(constr_func, Callable):
            raise TypeError(f"`constr_func` must consist of callable, but got {type(constr_func)}")
        self.constr_func = constr_func
        self.Q = th.tensor([], device=self.device, dtype=th.float32)  # Q(n_batch, n_atoms * n_dim, n_constr)
        self.jac = None # Jacobian Matrix
        self.R = th.tensor([], device=self.device, dtype=th.float32)  # R(n_batch, n_constr, n_constr)
        self.q = th.tensor([], device=self.device, dtype=th.float32)  # (n_batch, n_constr, 1), solution of `R^T q = d/dt constr_val(t)`
        self.sqrtM = None  # M^1/2, (n_batch, n_atoms, n_dim)
        self.negsqrtM = None  # M^-1/2
        self.max_proj_iter = 10
        self.constr_thres = constr_threshold

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
        _y_check = th.vmap(self.constr_func)(X)
        if self._lazy_calc_constr_val:
            self.constr_val_now = _y_check
        # check constr_val shape
        if _y_check.shape != self.constr_val_now.shape:
            raise RuntimeError(
                f'`constr_val` must have the same shape as what constr_func returned {self.constr_val_now.shape}, but got {_y_check.shape}.'
            )
        jac, y = self._jacobian(X)
        if self.verbose:
            self.logger.info(
                f'Constraint values are now {np.array2string(self.constr_val_now.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}'
            )
        if y.ndim != 2:
            raise ValueError(f'`constr_func` must return a 2D tensor of shape (n_batch, n_constr), but got {y.shape}.')
        self._do_qr(jac)
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
        V_init.copy_(th.where(Ek_p < 1e-5, 0., th.sqrt(Ek/Ek_p) * ProjV))
        self.n_reduce += jac.shape[1]
        # recalculate target Ek under constraints
        _, n_atom, n_dim = X.shape
        self.EK_TARGET = 0.5 * (n_dim * (n_atom - 1) - self.n_reduce) * 8.617333262145e-5 * self.T_init
        # calc. constr. intensity
        n_batch, n_constr, _ = self.R.shape
        self.lamb = th.zeros((n_batch, n_constr), device=self.device)
        # debug
        #self._debug_X_check = list()
        #self._debug_V_check = list()
        # ...

    def _constr_func_wrapped(self, X, constr_val_now):
        """
        Manage the constraint functions.
        Converting constraint functions into s_k(X) = constr_func(X) - constr_val, thereby constraints are s_k(X) = 0.
        Returns:

        """
        y = self.constr_func(X) - constr_val_now  # (n_batch, n_constr, )
        y = th.atleast_1d(y)
        return y, y  # repeat the output to use `has_aux` in th.func.jacrev to return the function values

    def _constr_func_for_hvp(self, X):
        """
        Manage the constraint functions used for HVP.
        Converting constraint functions into s_k(X) = constr_func(X) - constr_val, thereby constraints are s_k(X) = 0.
        Returns:

        """
        y = self.constr_func(X) - self.constr_val_now  # (n_batch, n_constr, )
        y = th.atleast_1d(y)
        return y

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
            y = self.constr_val_func_raw(t)
            if isinstance(y, (Tuple, List)):
                y = th.vstack(y).mT
            else:
                y = y.reshape(-1, 1)

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
        Project X into the manifold defined by all constrains by QR decomposition.
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
            th.linalg.solve_triangular(self.R.mT.contiguous(), self.d_constr.unsqueeze(-1), upper=False, out=self.q)
            Px.add_(self.negsqrtM * (self.Q @ self.q).reshape(n_batch, n_atoms, n_dim))

        return Px

    def _project_norm(self, X:th.Tensor) -> th.Tensor:
        """
        Project X to normal space.
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

        return Px

    def _project2(self, X:th.Tensor) -> th.Tensor:
        """
        Continuously project the Jacobian of all constrains to the exact manifold (by Newton-like iteration).
        Update X in-place.
        Args:
            X: the input tensor of shape (n_batch, n_atom, n_dim). It might be coordinates, velocity, and higher deviations.

        Returns: th.Tensor, the constraint forces \lambda.

        """
        n_batch, n_atoms, n_dim = X.shape
        _lamb = th.zeros(n_batch, self.R.shape[-1], 1, device=self.device)

        # MAIN loop
        for i in range(0, self.max_proj_iter):
            # update Jacobian
            jac, y = self._jacobian(X)
            # print
            constr_err = th.max(th.abs(y)).item()
            if self.verbose > 0:
                self.logger.info(f'{i: < 3d} Constraint errors are now: {constr_err:.4e}')
            # threshold
            if constr_err < self.constr_thres:
                # constr. forces lambda
                Fc = th.linalg.solve_triangular(self.R, _lamb, upper=True)
                # unit conversion
                Fc *= 207.28617 / (self.time_step ** 2)  # convert g/mol Angstrom^2 fs^-2 to eV/Atom
                return Fc
            # QR factor.
            self._do_qr(jac)
            # P = M^1/2 @ (I - Q Q^T) @ M^-1/2
            # Q(n_batch, n_atoms * n_dim, n_constr)
            _lamb = th.linalg.solve_triangular(self.R.mT.contiguous(), y.unsqueeze(-1) , upper=False) # (n_batch, n_constr, 1)
            # (n_batch, n_atoms, n_dim) * (n_batch, n_atoms * n_dim, n_constr) @ (n_batch, n_constr, 1)
            corr = self.negsqrtM * (self.Q @ _lamb).reshape(n_batch, n_atoms, n_dim)
            X.add_(corr, alpha=-1.)
            #V.add_(corr/self.time_step, alpha=-1.)

        # if not converged
        self.logger.warning("Projection of X to the manifold is not converged.")
        # constr. forces lambda
        Fc = th.linalg.solve_triangular(self.R, _lamb, upper=True)
        # unit conversion
        Fc *= 207.28617 / (self.time_step ** 2)  # convert g/mol Angstrom^2 fs^-2 to eV/Atom
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
            if constr_err < self.constr_thres:
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

    def _jacobian_derivative(self, X, v: th.Tensor) -> th.Tensor:
        """
        Compute `dJ/dt = v^T H` by auto-differentiation.
        Args:
            X:
            v:

        Returns: v^T H with shape (n_batch, n_constr, n_atom, n_dim)

        """
        vH: th.Tensor = bhvp(self._constr_func_for_hvp, (X,), (v,), False)  # (n_batch, n_constr, n_atom, n_dim)
        return vH.contiguous()

    def christoffel_solver(self, X: th.Tensor, V:th.Tensor, max_iter=20, tol=1e-6, omega=0.7):
        """
        iteratively solve the Christoffel symbol term by implicit midpoint method.
        function implicit_midpoint_step(r_n, v_n, h, tol=1e-8, max_iter=10)
        """
        n_batch, n_atom, n_dim = X.shape
        V_old = V.clone()
        V_new = V
        X_old = X.clone()
        X_new = X
        X_new.add_(V_old, alpha=self.time_step)

        for _ in range(max_iter):
            X_mid = (X_new + X_old)/2.  # r_{1/2} = (r_n + r_{n+1})/2
            V_mid = (V_new + V_old)/2.  # v_{1/2} = (v_n + v_{n+1})/2
            # calc. Christoffel. Christoffel symbol is `M^-1/2 Q R^-T J' v`
            dJ = self._jacobian_derivative(X_mid, V_mid)  # the J', which J' = v^T H, shape: (n_batch, n_constr, n_atom, n_dim)
            djr = th.einsum('ijkl, ikl -> ij', dJ, V_mid)  # the `J' v`, (n_batch, n_constr)
            # The `R^-T J' v`, solving (n_batch, n_constr, n_constr) x = (n_batch, n_constr, 1) -> x(n_batch, n_constr, 1)
            _RTjr = th.linalg.solve_triangular(self.R.mT.contiguous(), djr.unsqueeze(-1), upper=False)
            # Christoffel symbol: (B, N, N) @ (B, N, K) @ (B, K, 1) -> (n_batch, n_free, 1) -> (n_batch, n_atom, n_dim)
            Christoffel = self.negsqrtM * (self.Q @ _RTjr).reshape(n_batch, n_atom, n_dim)
            # residuals
            resi_x = X_new - th.add(X_old, V_mid, alpha=self.time_step)
            resi_v = V_new - th.add(V_old, Christoffel, alpha=-self.time_step)
            converge_mask_x = th.linalg.norm(resi_x, dim=(-2, -1)) < self.constr_thres
            converge_mask_v = th.linalg.norm(resi_v, dim=(-2, -1)) < self.constr_thres
            if th.all(converge_mask_x & converge_mask_v):
                return X_new, V_new

            # denote B = M^-1/2 Q R^-T, B(n_batch, n_free, n_constr), n_free = n_atom * n_dim.
            # B H v = B j': (n_batch, n_constr, n_free) @ (n_batch, n_constr, n_free)
            # Here Newton algo. is used to solve the implicit midpoint eq.:
            #   X_new = X_old + dt * V_mid
            #   V_new = V_old + dt * B V_mid^T H V_mid
            # and corresp. residuals:
            #   R_X = X_new - X_old - dt * V_mid
            #   R_V = V_new - V_old - dt * B V_mid^T H V_mid
            #
            # The Newton iter. formulae:
            #   [X_new, V_new]^T = [X_new, V_new]^T - J([R_X, R_V])^-1 [R_X, R_V]^T
            # where J([R_X, R_V]) =
            #   [[I,       -0.5 * dt * I],
            #    [DH, I + dt * B H V_mid]]
            # DH = \partial R_V / \partial X_new, that requires the complicated derivative of H. Here we ignore this term.
            # Hence J([R_X, R_V]) =
            #   [[I,      -0.5 * dt * I],
            #    [O, I + dt * B H V_mid]]
            # and J^-1 =
            #   [[I, 0.5 * dt * P],
            #    [O,            P]]
            # where P is the inverse of `I + dt * B H V_mid`.
            #
            # Hence, the iteration formulae are:
            #   X_new = X_new - (R_X + 0.5 * dt * P @ R_V)
            #   V_new = V_new - P @ R_V
            # For solving P, Woodbury identity: (I + U V^T)^-1 = I - U (I + V^T U)^-1 V^T can be applied:
            #   denote `H V_mid` with dJ^T, (I + dt * B H V_mid)^-1 = (I + dt * B dJ^T)^-1
            #   (I + dt * B dJ^T)^-1 = I - dt * B (I + dJ^T B*dt)^-1 dJ^T
            # Hence P @ R_V can be calculated as :
            # P @ R_V = dt * B @ linalg.solve(I + dJ^T @ B*dt, dJ^T @ R_V)
            #
            # Fully expand:
            #   P @ R_V = dt * M^-1/2 @ Q @ R^-T @ linalg.solve(I + dJ^T @ M^-1/2 @ Q @ R^-T * dt, dJ^T @ R_V)

            # #######################################################################3
            J_mid, y = self._jacobian(r_mid)  # (n_constr, n_atom * n_dim)
            self._do_qr(J_mid)  # (n_f, n_constr), (n_constr, n_constr)
            J_dot_mid = self._jacobian_derivative(r_mid, v_mid).reshape(1, -1)  # (n_constr, n_atom * n_dim)
            _v = v_mid.reshape(-1, 1)
            _y = J_dot_mid @ _v  # (n_constr, n_f) @ (n_atom*n_dim, 1) = (n_constr, 1)
            _y = np.linalg.solve(R_mid.T, _y)  # (n_constr, n_constr) @ (n_constr, 1) = (n_constr, 1)
            # (n_f, n_f) @ (n_f, n_constr) @ (n_constr, 1) = (n_f, 1)
            Gamma = M_sqrt_inv @ Q_mid @ _y

            # 带松弛的更新
            v_candidate = v_old - dt * Gamma.reshape(n_atom, n_dim)
            v_next = omega * v_candidate + (1 - omega) * v_new

            eps = np.linalg.norm(v_next - v_new)
            print(f'{_ + 1} Residuals: {eps}')
            if eps < tol:
                return v_next

            v_new = v_next

        print('Not Converged !')
        return v_new

    def implicit_midpoint_step(
            self,
            X0,
            X,
            V0,
            V,
            F,
            lamb,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y,
    ):
        """
        Solve manifold dynamics by implicit midpoint method.
        Here the input X0, V0 are at the endpoint in last step,
        X, V are the predicted endpoint in this step,
        input F (the potential forces) and lamb (the constraint forces) are at the midpoint of last step,
        and returned X_mid, V_mid, Energy, Force and lamb are all at the **midpoint**.
        The X, V in the end point are updated **in-place** to the input variables.
        Args:
            X:
            X0:
            V:
            V0:
            F:
            lamb:
            func:
            grad_func_:
            func_args:
            func_kwargs:
            grad_func_args:
            grad_func_kwargs:
            masses:
            atom_masks:
            is_grad_func_contain_y:

        Returns:

        """
        n_batch, n_atoms, n_dim = X.shape
        # MAIN LOOP
        is_converged = False
        for i in range(self.max_proj_iter):
            X_mid = 0.5 * (X0 + X)
            V_mid = 0.5 * (V0 + V)
            # update Jacobian & QR factor.
            jac, y = self._jacobian(X_mid)  # (n_batch, n_constr, n_atoms, n_dim)
            self._do_qr(jac)
            # 1st forces eval
            with th.set_grad_enabled(self.require_grad):
                X_mid.requires_grad_(self.require_grad)
                E_mid = func(X_mid, *func_args, **func_kwargs)
                if is_grad_func_contain_y:
                    F_mid = - grad_func_(X_mid, E_mid, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    F_mid = - grad_func_(X_mid, *grad_func_args, **grad_func_kwargs) * atom_masks
            # residuals
            f_constr = th.einsum('bijk, bi -> bjk', jac, lamb)
            RX = X - X0 - V_mid * self.time_step
            RV = V - V0 - (F_mid + f_constr) / masses * self.time_step * 9.64853329045427e-3
            # check thres.
            norm_X, norm_V, norm_s = th.linalg.norm(RX), th.linalg.norm(RV), th.linalg.norm(y)
            if self.verbose > 0:
                self.logger.info(
                    f'{i} residuals X, V, constraints: {norm_X.item():.4e}, {norm_V.item():.4e}, {norm_s.item():.4e}'
                )
            if (norm_X < self.constr_thres) and (norm_V < self.constr_thres) and (norm_s < self.constr_thres):
                is_converged = True
                break

            # update V & X by quasi-Newton
            tinvR_y = th.linalg.solve_triangular(self.R.mT.contiguous(), y.unsqueeze(-1) , upper=False)  # R^-T y : (n_batch, n_constr, 1)
            d_lambda = (
                    - 4./self.time_step * tinvR_y.squeeze(-1)
                    + 2./self.time_step * th.einsum('bij, bi -> bj', self.Q, (self.sqrtM * RX).flatten(-2, -1))
                    + th.einsum('bij, bi -> bj', self.Q, (self.sqrtM * (RV - 2*V_mid)).flatten(-2, -1))
            )
            d_lambda = 1./(self.time_step * 9.64853329045427e-3) * th.linalg.solve_triangular(self.R, d_lambda.unsqueeze(-1), upper=True).squeeze(-1)
            ds = 4./self.time_step * self.negsqrtM * (self.Q @ tinvR_y).reshape(n_batch, n_atoms, n_dim)
            dV = - ds - self._project1(RV) + 2./self.time_step * self._project_norm(RX)
            dX = 0.5 * self.time_step * dV - RX
            lamb.add_(d_lambda, alpha=1.)
            X.add_(dX, alpha=1.)
            V.add_(dV, alpha=1.)
        if not is_converged:
            self.logger.warning('Implicit midpoint method did not converge!')

        return X_mid, V_mid, E_mid, F_mid, lamb