"""Dimer transition-state search implemented on the optimizer framework."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch as th
from torch import nn

from BUCToolkit.Bases.StdContainer import StdContainer
from BUCToolkit.BatchOptim._BaseOpt import _BaseOpt
from BUCToolkit.BatchOptim.TS.Dimer import FindMinEigen
from BUCToolkit.utils import index_ops
from BUCToolkit.utils.grad_functions import fin_diff_hvp

class Dimer(_BaseOpt):
    """Modified Dimer transition-state search on the optimizer framework."""

    def __init__(
            self,
            E_threshold: float = 1e-3,
            Torque_thres: float = 1e-2,
            Curvature_thres: float = -0.1,
            F_threshold: float = 0.05,
            maxiter_trans: int = 100,
            maxiter_rot: int = 10,
            max_steplength: float = 0.5,
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2,
    ) -> None:
        self.Torque_thres = float(Torque_thres)
        self.Curvature_thres = float(Curvature_thres)
        if not isinstance(maxiter_rot, int) or maxiter_rot <= 0:
            raise ValueError('`maxiter_rot` must be an integer greater than 0.')
        self.maxiter_trans = int(maxiter_trans)
        self.maxiter_rot = maxiter_rot
        self.max_steplength = float(max_steplength)
        self.dx = float(dx)

        super().__init__(
            iter_scheme='Dimer',
            E_threshold=float(E_threshold),
            F_threshold=float(F_threshold),
            maxiter=self.maxiter_trans,
            linesearch='None',
            steplength=1.,
            use_bb=False,
            device=device,
            verbose=verbose,
        )
        self._is_inplace_update = True
        self.Rotator = FindMinEigen(
            self.Torque_thres,
            self.Curvature_thres,
            self.maxiter_rot,
            self.dx,
            self.device,
            self.verbose,
            _hold_samples=True,
        )

        self._X_diff_init: th.Tensor | None = None
        self._func: Callable | None = None
        self._grad_func: Callable | None = None
        self._is_grad_func_contain_y: bool | None = None
        self._require_grad: bool | None = None
        self.v: th.Tensor | None = None
        self.Hv: th.Tensor | None = None
        self.vHv: th.Tensor | None = None
        self._v: th.Tensor | None = None
        self._Hv: th.Tensor | None = None
        self._vHv: th.Tensor | None = None
        self._local_extra_converge_mask: th.Tensor | None = None

    def set_batch_updater(
            self,
            method_trans: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]],
            method_rot: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]] | None = None,
    ) -> None:
        """Register translation and rotation batch updaters."""
        super().set_batch_updater(method_trans, method_trans)
        if method_rot is None:
            self.Rotator._hold_samples = True
        else:
            self.Rotator.set_batch_updater(method_rot)

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            X_diff: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs=None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs=None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            output_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: Optional[th.Tensor | List] = None,
    ):
        """Run the Dimer transition-state search."""
        if not isinstance(X, th.Tensor):
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
        if batch_indices is None:
            raise NotImplementedError(
                'Regular batch version is not implemented yet. You may specify '
                'a `batch_indices` with identity values instead.'
            )

        if X_diff is None:
            X_diff = th.randn_like(X)
        elif not isinstance(X_diff, th.Tensor):
            raise TypeError(f'`X_diff` must be torch.Tensor, but occurred {type(X_diff)}.')
        elif X_diff.ndim == 2:
            X_diff = X_diff.unsqueeze(0)
        elif X_diff.ndim != 3:
            raise ValueError(f'`X_diff` must be 2D or 3D, but got shape [{X_diff.shape}]')
        if X_diff.shape != X.shape:
            raise ValueError(f'`X_diff` and `X` must have the same shape, but got {X_diff.shape} and {X.shape}.')

        self._X_diff_init = X_diff
        try:
            return super().run(
                func=func,
                X=X,
                grad_func=grad_func,
                func_args=tuple(func_args),
                func_kwargs={} if func_kwargs is None else func_kwargs,
                grad_func_args=tuple(grad_func_args),
                grad_func_kwargs={} if grad_func_kwargs is None else grad_func_kwargs,
                is_grad_func_contain_y=is_grad_func_contain_y,
                require_grad=require_grad,
                output_grad=output_grad,
                fixed_atom_tensor=fixed_atom_tensor,
                batch_indices=batch_indices,
            )
        finally:
            self._X_diff_init = None

    def _init_check_y_grad(
            self,
            func: Callable,
            X: th.Tensor,
            grad_func: Callable,
            func_args: Tuple,
            func_kwargs: Dict,
            grad_func_args: Tuple,
            grad_func_kwargs: Dict,
            is_grad_func_contain_y: bool,
            require_grad: bool,
            atom_masks: th.Tensor,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | None,
    ) -> StdContainer:
        """Initialize the Dimer mode and validate its first energy and gradient."""
        if batch_indices is None:
            raise NotImplementedError('Dimer requires irregular `batch_indices`.')
        if self._X_diff_init is None:
            raise RuntimeError('Dimer initial direction was not prepared before initialization.')

        v = self._X_diff_init.to(self.device).mul(atom_masks)
        self.v, energies, X_grad, self.Hv, self.vHv = self.Rotator.run(
            func=func,
            X=X,
            v=v,
            grad_func=grad_func,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            fixed_atom_tensor=atom_masks,
            batch_indices=batch_indices,
        )
        if energies.shape[0] != self.n_true_batch:
            raise ValueError(f"shape of output ({energies.shape}) does not match given batch indices")
        if X_grad.shape != X.shape:
            raise RuntimeError(f'X_grad ({X_grad.shape}) and X ({X.shape}) have different shapes.')

        self.is_concat_X = True
        energies = energies.detach()
        X_grad = X_grad.detach()
        X_grad.mul_(atom_masks)
        X = X.detach()
        return StdContainer(
            Energy=energies,
            X=X,
            Force=-X_grad,
            X_grad=X_grad,
        )

    def initialize(
            self,
            func: Callable,
            X: th.Tensor,
            grad_func: Callable,
            func_args: Tuple,
            func_kwargs: Dict,
            grad_func_args: Tuple,
            grad_func_kwargs: Dict,
            is_grad_func_contain_y: bool,
            require_grad: bool,
            atom_masks: th.Tensor,
            batch_indices: th.Tensor | None,
    ) -> None:
        """Initialize the current mode and Dimer runtime context."""
        if batch_indices is None:
            raise NotImplementedError('Dimer requires irregular `batch_indices`.')

        self._func = func
        self._grad_func = grad_func
        self.s.func_args = func_args
        self.s.func_kwargs = func_kwargs
        self.s.grad_func_args = grad_func_args
        self.s.grad_func_kwargs = grad_func_kwargs
        self.s.atom_masks = atom_masks
        self.s.batch_tensor = batch_indices
        self._extra_converge_mask = self.vHv.reshape(-1) < 0.
        self._is_grad_func_contain_y = is_grad_func_contain_y
        self._require_grad = require_grad
        self._line_search.HAS_GRAD = False

    def initialize_algo_param(self) -> None:
        pass

    def _update_algo_param(
            self,
            select_mask: th.Tensor,
            select_mask_short: th.Tensor | None,
            batch_scatter_indices: th.Tensor | None,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            displace: th.Tensor,
    ) -> None:
        if self._hold_samples:
            self._v = self.v
            self._Hv = self.Hv
            self._vHv = self.vHv
        else:
            self._v = self.v[:, select_mask, :]
            self._Hv = self.Hv[:, select_mask, :]
            self._vHv = self.vHv[:, select_mask_short, :]

    def _update_direction(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices: th.Tensor | None,
    ) -> None:
        if batch_scatter_indices is None:
            raise NotImplementedError('Dimer requires irregular `batch_indices`.')
        atom_masks = self.s.atom_masks

        vg = th.sum(
            index_ops.index_inner_product(self._v, g, dim=1, batch_indices=batch_scatter_indices),
            dim=-1,
            keepdim=True,
        )
        tangent_grad = g - vg.index_select(1, batch_scatter_indices) * self._v
        tangent_norm = th.sqrt(th.sum(
            index_ops.index_inner_product(tangent_grad, tangent_grad, 1, batch_scatter_indices),
            dim=-1,
            keepdim=True,
        ))
        u = tangent_grad / (tangent_norm.index_select(1, batch_scatter_indices) + 1e-20)
        ug = th.sum(
            index_ops.index_inner_product(u, g, dim=1, batch_indices=batch_scatter_indices),
            dim=-1,
            keepdim=True,
        )
        _, _, Hu = fin_diff_hvp(
            self._func,
            self.s.func_args,
            self.s.func_kwargs,
            self._grad_func,
            self.s.grad_func_args,
            self.s.grad_func_kwargs,
            X,
            u,
            batch_scatter_indices,
            is_g_contain_y=self._is_grad_func_contain_y,
            require_grad=self._require_grad,
        )
        self._Hv.mul_(atom_masks)
        Hu.mul_(atom_masks)

        vHv = th.sum(index_ops.index_inner_product(
            self._v, self._Hv, dim=1, batch_indices=batch_scatter_indices,
        ), dim=-1, keepdim=True)
        vHu = th.sum(index_ops.index_inner_product(
            self._v, Hu, dim=1, batch_indices=batch_scatter_indices,
        ), dim=-1, keepdim=True)
        uHv = th.sum(index_ops.index_inner_product(
            u, self._Hv, dim=1, batch_indices=batch_scatter_indices,
        ), dim=-1, keepdim=True)
        uHu = th.sum(index_ops.index_inner_product(
            u, Hu, dim=1, batch_indices=batch_scatter_indices,
        ), dim=-1, keepdim=True)
        nondiag = 0.5 * (uHv + vHu)
        H22 = th.cat((vHv, nondiag, nondiag, uHu), dim=-1).reshape(-1, 2, 2)
        eigenvalues, eigenvectors = th.linalg.eigh(H22)
        eigenvalues[:, 0] = th.where(eigenvalues[:, 0] >= 0., -eigenvalues[:, 0], eigenvalues[:, 0])
        eigenvalues[:, 0].clamp_(max=-0.1)
        eigenvalues[:, 1].clamp_(min=0.1)
        subspace_grad = th.cat((vg, ug), dim=-1).squeeze(0)
        coefficients = th.einsum(
            'bij,bi->bj',
            -eigenvectors @ (eigenvalues.unsqueeze(-1).reciprocal() * eigenvectors.mT),
            subspace_grad,
        )
        dX = (
            coefficients[batch_scatter_indices, 0:1] * self._v +
            coefficients[batch_scatter_indices, 1:] * u
        )
        dX_norm = th.sqrt(th.sum(
            index_ops.index_inner_product(dX, dX, 1, batch_scatter_indices),
            dim=-1,
            keepdim=True,
        ))
        dX_norm_atoms = dX_norm.index_select(1, batch_scatter_indices)
        dX = th.where(
            dX_norm_atoms > self.max_steplength,
            self.max_steplength / dX_norm_atoms * dX,
            dX,
        )
        dX.mul_(atom_masks)

        X.add_(dX)

        self._v, energy, X_grad, self._Hv, self._vHv = self.Rotator.run(
            func=self._func,
            X=X,
            v=self._v,
            grad_func=self._grad_func,
            func_args=self.s.func_args,
            func_kwargs=self.s.func_kwargs,
            grad_func_args=self.s.grad_func_args,
            grad_func_kwargs=self.s.grad_func_kwargs,
            is_grad_func_contain_y=self._is_grad_func_contain_y,
            require_grad=self._require_grad,
            fixed_atom_tensor=atom_masks,
            batch_indices=self.s.batch_tensor,
        )
        self._local_extra_converge_mask = self._vHv.reshape(-1) < 0.
        self._line_search.HAS_GRAD = True
        self._line_search.STORE_Y = energy
        self._line_search.STORE_GRAD = X_grad
        return None

    def _update_algo_batches(
            self,
            select_indices: th.Tensor,
            select_indices_short: th.Tensor | None,
    ) -> None:
        if self._hold_samples:
            self.v = self._v
            self.Hv = self._Hv
            self.vHv = self._vHv
            self._extra_converge_mask = self._local_extra_converge_mask
        else:
            self.v.index_copy_(1, select_indices, self._v)
            self.Hv.index_copy_(1, select_indices, self._Hv)
            self.vHv.index_copy_(1, select_indices_short, self._vHv)
            self._extra_converge_mask.index_copy_(
                0,
                select_indices_short,
                self._local_extra_converge_mask,
            )
