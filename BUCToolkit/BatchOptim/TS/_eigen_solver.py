"""Independent eigen solvers for the BaseOpt transition-state implementations.

The solver classes are intentionally copied from the legacy TS implementations.
They do not import or depend on the legacy optimizer classes.
"""

from itertools import accumulate
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple
import time
import warnings
import logging
import sys
import os

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.utils._print_formatter import (
    GLOBAL_SCIENTIFIC_ARRAY_FORMAT,
    FLOAT_ARRAY_FORMAT,
    SCIENTIFIC_ARRAY_FORMAT,
    STRING_ARRAY_FORMAT,
)
from BUCToolkit.utils import index_ops
from BUCToolkit.utils.grad_functions import fin_diff_hvp
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.utils.setup_loggers import has_any_handler, clear_all_handlers
from BUCToolkit.Bases.BaseMotion import BaseMotion, FLOAT_TYPE

np.set_printoptions(**GLOBAL_SCIENTIFIC_ARRAY_FORMAT)

__all__ = ['FindEigen', 'FindMinEigen']

class FindEigen(BaseMotion):
    """
    Find the eigenvector with minimum eigenvalue by Riemann gradient descent on S^2 manifold v^T v = 1.
    In fact, dimer only requires the direction within negative cone, i.e., v^T H v < 0.
    """
    def __init__(
            self,
            Torque_thres: float = 1e-2,
            Eigen_thres: float = -0.1,
            maxiter_lanczos: int = 10,
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            _hold_samples: bool = False,
    ):
        """

        Args:
            Torque_thres: convergence threshold of torque.
            Eigen_thres: convergence threshold of the minimal eigenvalue differences.
            maxiter_lanczos: maximum number of lanczos iterations.
            dx: step size for finite difference approximation.
            device: the device on which the computation runs.
            verbose: the verbosity level.
            _hold_samples: whether to hold samples during optimization. if True, samples will not be removed even they have converged.
        """

        warnings.filterwarnings('always')
        self.Torque_thres = abs(float(Torque_thres))
        self.Eigen_thres = abs(float(Eigen_thres))
        assert (maxiter_lanczos > 0) and isinstance(maxiter_lanczos, int), '`maxiter_rot` must be an integer greater than 0.'
        self.maxiter_lanczos = int(maxiter_lanczos)
        if self.maxiter_lanczos <= 1:
            raise ValueError(f'`maxiter_rot` must be greater than 1, but got {self.maxiter_lanczos}.')
        self.dx = float(dx)
        self.subspace_hessian = None
        self.device = device
        self.verbose = verbose
        self.subspace_maxdim = self.maxiter_lanczos

        self._hold_samples = _hold_samples

        # logger
        super().__init__()
        self.init_logger('Main.TS.Eigen')

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
            method: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]]
    ) -> None:
        """
        Set a method to update the taget function when variables change.
        It receives a mask tensor of shape (n_batch, ) that only selects the `True` part to input to the function, and receives the old
        `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`,
        returns the corresponding masked new `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`.

        This method is used to dynamically 'remove' the samples which have been converged in a batch to avoid
        redundant calculation of converged samples.

        Default transform is identical transform (i.e., do nothing)
        Args:
            method: Callable(mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict) -> Tuple[Tuple, Dict, Tuple, Dict],
        the method of updating function arguments for a mask.

        Returns: None
        """
        self._update_batch = method
        self._hold_samples = False

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            v: th.Tensor,
            grad_func: Any | nn.Module = None,
            func_args: Tuple = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
            eigen_order: int = 1,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Find the eigenvector of Hessian at X with the minimum eigenvalue, by Riemannian gradient descent on S^n manifold v^T v = I.

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            v: Tensor[n_batch, n_atom, 3], the atom direction used to finite difference.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X
                i.e., grad = grad_func(X, y, *grad_func_args, **grad_func_kwargs), else grad = grad_func(X, *grad_func_args, **grad_func_kwargs)
            require_grad: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs) calculation.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is the same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]
            eigen_order: int, the number of required minimum eigenvalues.

        Return: y, g, KRYLOV_BASES, KRYLOV_EIGENVAL, KRYLOV_EIGENVEC;
            y: the mean value of function at X, i.e., (f(X + delta * v) + f(X - delta * v))/2
            g: the mean grad of function at X, i.e., (grad(X + delta * v) + grad(X - delta * v))/2
            KRYLOV_EIGENVAL: the eigen values of krylov subspace Hessian
            KRYLOV_EIGENVEC: the corresponding eigen vectors of Hessian
        """
        t_main = time.perf_counter()
        if func_kwargs is None:
            func_kwargs = dict()
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            if X.ndim == 2:
                X.unsqueeze_(0)
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but got {type(X)}.')
        if isinstance(v, th.Tensor):
            if v.ndim == 2:
                v.unsqueeze_(0)
            if v.shape != X.shape:
                raise ValueError(f"`v` must have same shape as `X`, but got shape {v.shape}.")
        else:
            raise TypeError(f'`v` must be torch.Tensor, but got {type(v)}.')
        X, v = self.handle_dtype_device(FLOAT_TYPE, self.device, X, v)
        if batch_indices is None:
            raise NotImplementedError(
                f'Regular batch version is not implemented yet. You may specify a `batch_indices` with identity values instead.'
                f'It is fully compatible with regular batches, but merely a little performance loss.'
            )
        if eigen_order >= self.maxiter_lanczos:
            raise ValueError(
                f"Solving `eigen_order` ({eigen_order}) eigenvalues in {self.maxiter_lanczos} steps is impossible. "
                f"`maxiter_lanczos` greater than `eigen_order + 5` is recommended."
            )

        n_true_batch, batch_indices, self.batch_tensor, self.batch_scatter, batch_slice_indx = self.handle_batch_indices(
            batch_indices, n_batch, device=self.device
        )
        # initialize vars
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim
        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(
            grad_func, is_grad_func_contain_y, require_grad
        )

        if hasattr(self._update_batch, 'initialize'):
            self._update_batch.initialize()
        elif hasattr(self._update_batch, '__init__'):
            self._update_batch.__init__()
        # Selective dynamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)
        # atom_masks = atom_masks.flatten(-2, -1).unsqueeze(-1)  # (n_batch, n_atom*n_dim, 1)
        # other check
        if (not isinstance(self.maxiter_lanczos, int)) or (self.maxiter_lanczos <= 0):
            raise ValueError(f'Invalid value of maxiter_rot: {self.maxiter_lanczos}. It would be an integer greater than 0.')

        # set variables device
        func = preload_func(func, self.device)

        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.detach()
        # normalize v
        v.mul_(atom_masks)
        v_norm = th.sqrt(th.sum(index_ops.index_inner_product(
            v, v, 1, self.batch_scatter, out_size=n_true_batch,
        ), dim=-1, keepdim=True))
        v = v / v_norm.index_select(1, self.batch_scatter)

        # Full-size tensors own results in original structure order for the
        # whole solve. Underscored tensors inside the loop are the currently
        # active structures after optional dynamic removal. Structure-layout
        # data use `select_mask_short`; concatenated atom-layout data use the
        # expanded `select_mask`. Each iteration scatters the local state back
        # to its full owner before the next convergence decision.
        is_main_loop_converge = False
        t_st = time.perf_counter()
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
        # MAIN LOOP
        # X (1, n_batch * n_atom, n_dim)
        func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = func_args, func_kwargs, grad_func_args, grad_func_kwargs
        y, g, Hv = fin_diff_hvp(
            func,
            func_args_,
            func_kwargs_,
            grad_func_,
            grad_func_args_,
            grad_func_kwargs_,
            X,
            v,
            self.batch_scatter,
            is_g_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            delta=self.dx
        )
        g.mul_(atom_masks)
        Hv.mul_(atom_masks)
        vHv = th.sum(
            index_ops.index_inner_product(
                v,
                Hv,
                dim=1,
                batch_indices=self.batch_scatter,
                out_size=n_true_batch,
            ),
            dim=-1,
            keepdim=True
        ) # curvature, vHv (1, B0, 1), essentially the Lanczos alpha
        # grad in the tangent space 1st
        u = Hv - vHv.index_select(1, self.batch_scatter) * v
        beta = th.sqrt(th.sum(index_ops.index_inner_product(
            u, u, 1, self.batch_scatter, out_size=n_true_batch,
        ), dim=-1, keepdim=True)
        )  # i.e., the lanczos beta. (1, B0, 1)
        # Full-size owners keep every original structure in stable order. Local
        # underscored tensors below contain only the structures still active in
        # a dynamically shrinking Lanczos solve and are scattered back here.
        KRYLOV_BASES = th.zeros((self.subspace_maxdim, v.shape[1], v.shape[2]), device=self.device, dtype=X.dtype)
        KRYLOV_BASES[0] = v[0].clone()
        KRYLOV_HESSIAN = th.zeros(
            (n_true_batch, self.subspace_maxdim, self.subspace_maxdim),
            device=self.device, dtype=X.dtype
        )
        KRYLOV_HESSIAN[:, 0, 0] = vHv.reshape(n_true_batch).clone()
        KRYLOV_EIGENVAL = th.zeros(
            (n_true_batch, self.subspace_maxdim),
            device=self.device,
            dtype=X.dtype,
        )
        KRYLOV_EIGENVAL[:, 0] = vHv.reshape(n_true_batch).clone()
        KRYLOV_EIGENVEC = th.zeros_like(KRYLOV_HESSIAN)
        KRYLOV_EIGENVEC[:, 0, 0] = 1.
        krylov_eigenval_old = th.full_like(KRYLOV_EIGENVAL, th.inf)
        # cache for dynamically changed batch indices due to convergence, avoiding reallocate mem.
        batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        iter_min = eigen_order
        for i in range(1, self.maxiter_lanczos):
            # For a Ritz pair (theta, q), beta times the last component of q is
            # its Lanczos residual norm. All requested low modes must satisfy
            # the threshold before this structure can leave the active batch.
            last_comp = KRYLOV_EIGENVEC[:, i - 1, :eigen_order]  # (B, k)
            residual_norms = (beta * th.abs(last_comp)).amax(dim=-1, keepdim=True)  # (1, B, 1)
            converge_mask_eig = (residual_norms < self.Eigen_thres)  # (1, B, 1)
            #monitor_indx = min(eigen_order, i) - 1
            converge_mask_torque = (beta < self.Torque_thres)
            converge_mask = (converge_mask_eig | converge_mask_torque)  # (1, B, 1)
            # print
            #self.logger.debug(f"LANCZOS: H_sub:\n{KRYLOV_HESSIAN}")
            #self.logger.debug(f"LANCZOS: K_EIG_VEC:\n{KRYLOV_EIGENVEC}")
            #self.logger.debug(f"LANCZOS: K_EIG_VAL:\n{KRYLOV_EIGENVAL}")
            if self.verbose > 0:
                self.logger.info(
                    f"Eigen {i:>5d}\n "
                    f"Krylov Resi.: {np.array2string(residual_norms.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"target Eig.:  {np.array2string(KRYLOV_EIGENVAL[:, eigen_order].squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"Energies:     {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"Eig. Conv.:   {np.array2string(converge_mask.squeeze().numpy(force=True), **STRING_ARRAY_FORMAT)}\n "
                    f"TIME:         {time.perf_counter() - t_st:>6.4f} s"
                )
                t_st = time.perf_counter()
            # At loop entry the available Ritz subspace has dimension i. If a
            # sample terminates before it can contain all requested modes, no
            # meaningful continuation exists: dividing its vanishing residual
            # by a clamped beta would only manufacture a basis direction.
            if (i < iter_min) and th.any(converge_mask):
                raise RuntimeError(
                    'Lanczos converged before the requested Krylov dimension '
                    f'was available: got dimension {i}, required {iter_min}.'
                )
            if th.all(converge_mask):
                is_main_loop_converge = True
                break
            # Reaching this point means at least one structure still needs a
            # larger subspace. Keep a structure-level mask for model-argument
            # updates and expand
            # it through the original scatter only for atom-layout tensors.
            converge_mask_short = converge_mask
            converge_mask = converge_mask_short[:, self.batch_scatter, ...]  # (1, sumB*A, 1)
            # update batch, remove the already converged ones.
            # Dynamic mode removes converged samples before normalizing the
            # next residual; hold mode deliberately retains the full batch.
            should_remove_converged = not self._hold_samples
            if should_remove_converged:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_mask_short.reshape(-1),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_mask[0, :, 0])  # (sumB*A, )
                select_mask_short = ~converge_mask_short[0, :, 0]  # (B, )
                #y_ = y[select_mask_short]
                Hv_ = Hv[:, select_mask, :]
                X_ = X[:, select_mask, :]
                v_ = v[:, select_mask, :]
                u_ = u[:, select_mask, :]
                beta_ = beta[:, select_mask_short, :]
                atom_masks_ = atom_masks[:, select_mask, :]
                batch_tensor_ = self.batch_tensor[select_mask_short]
                n_local_batch = len(batch_tensor_)
                batch_scatter_ = th.repeat_interleave(
                    batch_tensor_indx_cache[:len(batch_tensor_)],
                    batch_tensor_,
                    dim=0
                )

                krylov_bases_ = KRYLOV_BASES[:, select_mask, :]
                krylov_hessian_ = KRYLOV_HESSIAN[select_mask_short, ...]
                sub_eigval_ = KRYLOV_EIGENVAL[select_mask_short, ...]
                sub_eigvec_ = KRYLOV_EIGENVEC[select_mask_short, ...]
            else:
                select_mask = None
                select_mask_short = None
                n_local_batch = n_true_batch
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = func_args, func_kwargs, grad_func_args, grad_func_kwargs
                Hv_ = Hv
                X_ = X
                v_ = v
                u_ = u
                beta_ = beta
                atom_masks_ = atom_masks
                batch_tensor_ = self.batch_tensor
                batch_scatter_ = self.batch_scatter
                krylov_bases_ = KRYLOV_BASES
                krylov_hessian_ =  KRYLOV_HESSIAN
                sub_eigval_ = KRYLOV_EIGENVAL
                sub_eigvec_ = KRYLOV_EIGENVEC

            # In dynamic mode converged samples have left the local batch. Hold
            # mode may retain a sample whose residual is merely below tolerance
            # while another sample continues. The clamp is only an arithmetic
            # safeguard for the normalization; it does not authorize an exact
            # Lanczos breakdown before the requested subspace dimension, which
            # is rejected by the check above.
            w_ = u_ / (beta_.index_select(1, batch_scatter_).clamp_min_(1e-20))  # (1, sumB*A, N)
            krylov_eigenval_old_ = sub_eigval_.clone()

            y2_t, g2_, Hw_ = fin_diff_hvp(
                func,
                func_args_,
                func_kwargs_,
                grad_func_,
                grad_func_args_,
                grad_func_kwargs_,
                X_,
                w_,
                batch_scatter_,
                is_g_contain_y=is_grad_func_contain_y,
                require_grad=require_grad,
                delta=self.dx
            )
            g2_.mul_(atom_masks_)
            Hw_.mul_(atom_masks_)  # mask
            # The three-term recurrence supplies the new diagonal and
            # off-diagonal entries of the symmetric tridiagonal Ritz matrix.
            wHw_ = th.sum(
                index_ops.index_inner_product(
                    w_, Hw_, dim=1, batch_indices=batch_scatter_,
                    out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True
            )
            u_ = Hw_ - wHw_.index_select(1, batch_scatter_) * w_ - beta_.index_select(1, batch_scatter_) * v_
            # Two-pass classical Gram-Schmidt controls loss of orthogonality in
            # finite precision; the reduction produces one coefficient per
            # saved basis vector and per active structure.
            Vu = index_ops.index_reduce(
                th.sum(krylov_bases_[:i] * u_, dim=-1),
                batch_scatter_,
                dim=1,
                out_size=n_local_batch,
            ).index_select(1, batch_scatter_)
            u_.sub_(th.einsum("kbd, kb -> bd", krylov_bases_[:i], Vu).unsqueeze(0))
            Vu = index_ops.index_reduce(
                th.sum(krylov_bases_[:i] * u_, dim=-1),
                batch_scatter_,
                dim=1,
                out_size=n_local_batch,
            ).index_select(1, batch_scatter_)
            u_.sub_(th.einsum("kbd, kb -> bd", krylov_bases_[:i], Vu).unsqueeze(0))


            # update bases and T, Only save the LOWER TRIANGULAR PART
            krylov_hessian_[:, i, i] = wHw_.reshape(n_local_batch).clone()
            krylov_hessian_[:, i, i - 1] = beta_.reshape(n_local_batch).clone()
            krylov_bases_[i, :, :] = w_[0].clone()

            # update beta
            beta_ = th.sqrt_(th.sum(index_ops.index_inner_product(
                u_, u_, 1, batch_scatter_, out_size=n_local_batch,
            ), dim=-1, keepdim=True)
            )  # i.e., the lanczos beta.

            #sub_eigval_, sub_eigvec_ = th.linalg.eigh(krylov_hessian_)  # (B, i), (B, i, i), default is using the lower triangular part
            sub_eigval_[:, :i + 1], sub_eigvec_[:, :i + 1, :i + 1] = th.linalg.eigh(
                krylov_hessian_[:, :i + 1, :i + 1]
            )

            # update next loop vars
            v_ = w_
            Hv_ = Hw_

            # Commit local active results to the stable full-batch owners. Hold
            # mode has no removed samples, so direct replacement is sufficient.
            # Dynamic mode leaves converged rows untouched and scatters only
            # the still-active structure and atom rows through their respective
            # indices.
            if should_remove_converged:
                select_indices = th.where(select_mask)[0]
                select_indices_short = th.where(select_mask_short)[0]
                y.index_copy_(0, select_indices_short, y2_t)
                v.index_copy_(1, select_indices, v_)
                u.index_copy_(1, select_indices, u_)
                #w.index_copy_(1, select_indices, w_)
                #X.index_copy_(1, select_indices, X_)
                Hv.index_copy_(1, select_indices, Hv_)
                beta.index_copy_(1, select_indices_short, beta_)
                g.index_copy_(1, select_indices, g2_)
                KRYLOV_BASES.index_copy_(1, select_indices, krylov_bases_)
                KRYLOV_HESSIAN.index_copy_(0, select_indices_short, krylov_hessian_)
                KRYLOV_EIGENVEC.index_copy_(0, select_indices_short, sub_eigvec_)
                KRYLOV_EIGENVAL.index_copy_(0, select_indices_short, sub_eigval_)
                krylov_eigenval_old.index_copy_(0, select_indices_short, krylov_eigenval_old_)
            else:
                y = y2_t
                v = v_
                u = u_
                #w = w_
                #X = X_
                Hv = Hv_
                beta = beta_
                g = g2_
                KRYLOV_HESSIAN = krylov_hessian_
                KRYLOV_BASES = krylov_bases_
                KRYLOV_EIGENVAL = sub_eigval_
                KRYLOV_EIGENVEC = sub_eigvec_
                krylov_eigenval_old = krylov_eigenval_old_
            pass

        if self.verbose:
            if is_main_loop_converge:
                self.logger.info(
                    '-' * 100 + f'\neig done. time: {time.perf_counter() - t_main:<.4f} s\n'
                )
            else:
                self.logger.warning(
                    '-' * 100 + f'\nWARNING: Some Structures\' Hessian eigenvectors were NOT Converged yet!\n'
                                f'eig done. time: {time.perf_counter() - t_main:<.4f} s\n'
                )

        # DEBUG
        #H = th.autograd.functional.hessian(func, X)[0].squeeze()
        #eigval, eigvec = th.linalg.eigh(H)
        #print(f"KRYLOV_EIGENVAL: {KRYLOV_EIGENVAL[0, :eigen_order+1]}\nTRUE_EIGENVAL: {eigval[:eigen_order+1]}")

        # Lift Ritz vectors from the tridiagonal subspace back into Cartesian
        # coordinates. batch_scatter selects each atom's structure-level Ritz
        # coefficients without materializing a padded regular batch.
        EIGVEC = th.einsum('mad, amk -> kad', KRYLOV_BASES, KRYLOV_EIGENVEC.index_select(0, self.batch_scatter))

        return y, g, KRYLOV_EIGENVAL, EIGVEC


class FindMinEigen(BaseMotion):
    """
    Find the eigenvector with minimum eigenvalue by Riemann gradient descent on S^2 manifold v^T v = 1.
    In fact, dimer only requires the direction within negative cone, i.e., v^T H v < 0.
    """
    def __init__(
            self,
            Torque_thres: float = 1e-2,
            Curve_thres: float = -0.1,
            maxiter_rot: int = 10,
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            _hold_samples: bool = False,
    ):
        """

        Args:
            Torque_thres: convergence threshold of torque.
            Curve_thres: convergence threshold of curvature, that v^T H v < `Curve_thres` is viewed as converged.
            maxiter_rot: maximum number of rotation iterations.
            dx: step size for finite difference approximation.
            device: the device on which the computation runs.
            verbose: the verbosity level.
            _hold_samples: whether to hold samples during optimization. if True, samples will not be removed even they have converged.
        """

        warnings.filterwarnings('always')
        self.Torque_thres = float(Torque_thres)
        self.Curve_thres = float(Curve_thres)
        if self.Curve_thres >= 0.:
            warnings.warn(f'`Curve_thres` should be less than 0 to determind the negative cone direction, but occurred {Curve_thres}.\n'
                          f'Now it has been set to its opposite number: {- Curve_thres}.')
            self.Curve_thres =  - Curve_thres
        assert (maxiter_rot > 0) and isinstance(maxiter_rot, int), '`maxiter_rot` must be an integer greater than 0.'
        self.maxiter_rot = int(maxiter_rot)
        self.dx = float(dx)
        self.subspace_hessian = None
        self.device = device
        self.verbose = verbose
        self._hold_samples = _hold_samples

        # logger
        super().__init__()
        self.init_logger('Main.TS.Eigen')

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
            method: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]]
    ) -> None:
        """
        Set a method to update the taget function when variables change.
        It receives a mask tensor of shape (n_batch, ) that only selects the `True` part to input to the function, and receives the old
        `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`,
        returns the corresponding masked new `func_args`, `func_kwargs`, `grad_func_args`, and `grad_func_kwargs`.

        This method is used to dynamically 'remove' the samples which have been converged in a batch to avoid
        redundant calculation of converged samples.

        Default transform is identical transform (i.e., do nothing)
        Args:
            method: Callable(mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict) -> Tuple[Tuple, Dict, Tuple, Dict],
        the method of updating function arguments for a mask.

        Returns: None
        """
        self._update_batch = method
        self._hold_samples = False

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            v: th.Tensor,
            grad_func: Any | nn.Module = None,
            func_args: Tuple = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Find the eigenvector of Hessian at X with the minimum eigenvalue, by Riemannian gradient descent on S^n manifold v^T v = I.

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            v: Tensor[n_batch, n_atom, 3], the atom direction used to finite difference.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X
                i.e., grad = grad_func(X, y, *grad_func_args, **grad_func_kwargs), else grad = grad_func(X, *grad_func_args, **grad_func_kwargs)
            require_grad: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs) calculation.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is the same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]

        Return:
            v: the eigenvector with min eigenvalue at X (or at least the negative cone direction)
            y: the mean value of function at X, i.e., (f(X + delta * v) + f(X - delta * v))/2
            g: the mean grad of function at X, i.e., (grad(X + delta * v) + grad(X - delta * v))/2
            vHv: the curvature at X given by finite difference.
        """
        t_main = time.perf_counter()
        # Sanitize kwargs
        func_kwargs = func_kwargs or dict()
        grad_func_kwargs = grad_func_kwargs or dict()
        func_args = tuple(func_args)
        grad_func_args = tuple(grad_func_args)

        # X shape
        if not isinstance(X, th.Tensor):
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        n_batch, n_atom, n_dim = X.shape
        X, v = self.handle_dtype_device(FLOAT_TYPE, self.device, X, v)

        # Grad func
        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(grad_func, is_grad_func_contain_y, require_grad)

        # Batch indices (irregular batch)
        if batch_indices is None:
            raise NotImplementedError(
                f'Regular batch version is not implemented yet. You may specify a `batch_indices` with identity values instead.'
                f'It is fully compatible with regular batches, but merely a little performance loss.'
            )
        n_inner_batch, batch_indices, self.batch_tensor, self.batch_scatter, batch_slice_indx = self.handle_batch_indices(
            batch_indices, n_batch, device=self.device
        )

        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim

        # Batch updater init
        if hasattr(self._update_batch, 'initialize'):
            self._update_batch.initialize()
        elif hasattr(self._update_batch, '__init__'):
            self._update_batch.__init__()

        # Selective dynamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)

        # Maxiter check
        if not isinstance(self.maxiter_rot, int) or self.maxiter_rot <= 0:
            raise ValueError(f'Invalid value of maxiter_rot: {self.maxiter_rot}. It would be an integer greater than 0.')

        # Device placement
        func = preload_func(func, self.device)
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.detach()
        # normalize v
        v.mul_(atom_masks)
        v_norm = th.sqrt(th.sum(index_ops.index_inner_product(
            v, v, 1, self.batch_scatter, out_size=n_inner_batch,
        ), dim=-1, keepdim=True))
        v = v / v_norm.index_select(1, self.batch_scatter)

        # Full-size tensors retain the original structure order. Underscored
        # tensors below are the active rotation batch: structure quantities use
        # `select_mask_short`, atom quantities use its scatter-expanded mask,
        # and every completed rotation step writes those local results back.
        is_main_loop_converge = False
        t_st = time.perf_counter()
        #ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
        # MAIN LOOP
        # X (1, n_batch * n_atom, n_dim)
        theta = th.zeros(n_inner_batch, device=self.device, dtype=X.dtype)
        func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = func_args, func_kwargs, grad_func_args, grad_func_kwargs
        y, g, Hv = fin_diff_hvp(
            func,
            func_args_,
            func_kwargs_,
            grad_func_,
            grad_func_args_,
            grad_func_kwargs_,
            X,
            v,
            self.batch_scatter,
            is_g_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            delta=self.dx
        )
        g.mul_(atom_masks)
        Hv.mul_(atom_masks)
        # curvature, vHv (1, B0, 1)
        vHv = th.sum(
            index_ops.index_inner_product(
                v,
                Hv,
                dim=1,
                batch_indices=self.batch_scatter,
                out_size=n_inner_batch,
            ),
            dim=-1,
            keepdim=True
        )
        # grad in the tangent space
        gT = Hv - vHv.index_select(1, self.batch_scatter) * v
        gT_norm = th.sqrt(th.sum(index_ops.index_inner_product(
            gT, gT, 1, self.batch_scatter, out_size=n_inner_batch,
        ), dim=-1, keepdim=True))
        w = v.clone()  #gT / (gT_norm.index_select(1, self.batch_scatter) + 1e-20)  # (1, sumB*A, N)
        # cache for dynamically changed batch indices due to convergence, avoiding reallocate mem.
        batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        for i in range(self.maxiter_rot):
            # threshold. Only need v in the negative cone, i.e., vHv < 0.
            converge_mask_curve = (vHv < self.Curve_thres)
            converge_mask_torque = (gT_norm < self.Torque_thres)
            #   reinsurance: When w and v are very collinear, stop meaningless update
            abort_mask = th.sum(index_ops.index_inner_product(
                v, w, 1, self.batch_scatter, out_size=n_inner_batch,
            ), dim=-1, keepdim=True) < 1.e-7
            converge_mask = (converge_mask_curve | converge_mask_torque)  # (1, B, 1)
            # print
            if self.verbose > 0:
                self.logger.info(f"Rot {i:>5d}\n "
                                 f"Torque:       {np.array2string(gT_norm.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Curvature:    {np.array2string(vHv.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Energies:     {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Theta (rad):  {np.array2string(theta.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                 f"Rot. Conv.:   {np.array2string(converge_mask.squeeze().numpy(force=True), **STRING_ARRAY_FORMAT)}\n "
                                 f"TIME:         {time.perf_counter() - t_st:>6.4f} s")
                t_st = time.perf_counter()
            # A fully converged batch exits normally. The collinearity abort is
            # a safety stop for a degenerate two-vector rotation subspace; it
            # removes only the affected structures when other structures can
            # still make progress.
            if th.all(converge_mask):
                is_main_loop_converge = True
                break
            elif th.all(converge_mask | abort_mask):
                break
            converge_mask_short = converge_mask | abort_mask
            converge_mask = converge_mask_short[:, self.batch_scatter, ...]  # (1, sumB*A, 1)
            # update batch, remove the already converged ones.
            if not self._hold_samples:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_mask_short.reshape(-1),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_mask[0, :, 0])  # (sumB*A, )
                select_mask_short = ~converge_mask_short[0, :, 0]  # (B, )
                #y_ = y[select_mask_short]
                vHv_ = vHv[:, select_mask_short, :]
                Hv_ = Hv[:, select_mask, :]
                X_ = X[:, select_mask, :]
                v_ = v[:, select_mask, :]
                gT_ = gT[:, select_mask, :]
                gT_norm_ = gT_norm[:, select_mask_short, :]
                atom_masks_ = atom_masks[:, select_mask, :]
                batch_tensor_ = self.batch_tensor[select_mask_short]
                n_local_batch = len(batch_tensor_)
                batch_scatter_ = th.repeat_interleave(
                    batch_tensor_indx_cache[:len(batch_tensor_)],
                    batch_tensor_,
                    dim=0
                )
            else:
                select_mask = None
                select_mask_short = None
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = func_args, func_kwargs, grad_func_args, grad_func_kwargs
                vHv_ = vHv
                Hv_ = Hv
                X_ = X
                v_ = v
                gT_ = gT
                gT_norm_ = gT_norm
                atom_masks_ = atom_masks
                batch_tensor_ = self.batch_tensor
                batch_scatter_ = self.batch_scatter
                n_local_batch = n_inner_batch

            # construction subspace Hessian [[vHv vHw] [wHv wHw]] with shape (B0, 2, 2) for 2nd order precise linear search
            w_ = gT_ / (gT_norm_.index_select(1, batch_scatter_) + 1e-20)  # (1, sumB*A, N)
            self.logger.debug(f"w:\n{w_}")
            y2_t, g2_, Hw_ = fin_diff_hvp(
                func,
                func_args_,
                func_kwargs_,
                grad_func_,
                grad_func_args_,
                grad_func_kwargs_,
                X_,
                w_,
                batch_scatter_,
                is_g_contain_y=is_grad_func_contain_y,
                require_grad=require_grad,
                delta=self.dx
            )
            g2_.mul_(atom_masks_)
            Hw_.mul_(atom_masks_)  # mask
            # subspace Hessian
            vHw_ = th.sum(
                index_ops.index_inner_product(
                    v_, Hw_, dim=1, batch_indices=batch_scatter_,
                    out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True
            )
            wHv_ = th.sum(
                index_ops.index_inner_product(
                    w_, Hv_, dim=1, batch_indices=batch_scatter_,
                    out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True
            )
            wHw_ = th.sum(
                index_ops.index_inner_product(
                    w_, Hw_, dim=1, batch_indices=batch_scatter_,
                    out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True
            )
            nondiag_ = 0.5 * (wHv_ + vHw_)  # (1, B, 1)
            H22_ = th.cat((vHv_, nondiag_, nondiag_, wHw_), dim=-1).reshape(-1, 2, 2)
            self.logger.debug(f"H22:\n{H22_.numpy(force=True)}")

            sub_eigval_, sub_eigvec_ = th.linalg.eigh(H22_)  # (B, 2), (B, 2, 2)
            cos_t_ = sub_eigvec_[None, :, 0:1, 0].index_select(1, batch_scatter_)
            sin_t_ = sub_eigvec_[None, :, 1:2, 0].index_select(1, batch_scatter_)  # (1, sumB*A, 1)
            theta_ = th.atan2(sub_eigvec_[:, 0, 0], sub_eigvec_[:, 0, 1])  # (B, )
            # update
            v_.mul_(cos_t_)
            v_.add_(w_ * sin_t_)
            v_.mul_(atom_masks_)
            Hv_.mul_(cos_t_)
            Hv_.add_(Hw_ * sin_t_)
            Hv_.mul_(atom_masks_)
            # curvature, vHv (1, B0, 1)
            vHv_ = th.sum(
                index_ops.index_inner_product(
                    v_, Hv_, dim=1, batch_indices=batch_scatter_,
                    out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True
            )
            # grad in the tangent space
            gT_ = Hv_ - vHv_.index_select(1, batch_scatter_) * v_
            gT_norm_ = th.sqrt(th.sum(index_ops.index_inner_product(
                gT_, gT_, 1, batch_scatter_, out_size=n_local_batch,
            ), dim=-1, keepdim=True))

            # update origin variables
            if not self._hold_samples:
                select_indices = th.where(select_mask)[0]
                select_indices_short = th.where(select_mask_short)[0]
                y.index_copy_(0, select_indices_short, y2_t)
                v.index_copy_(1, select_indices, v_)
                w.index_copy_(1, select_indices, w_)
                #X.index_copy_(1, select_indices, X_)
                Hv.index_copy_(1, select_indices, Hv_)
                vHv.index_copy_(1, select_indices_short, vHv_)
                gT_norm.index_copy_(1, select_indices_short, gT_norm_)
                g.index_copy_(1, select_indices, g2_)
                theta.index_copy_(0, select_indices_short, theta_)  # (B, )
                #atom_masks.index_copy_(1, select_indices, atom_masks_)
            else:
                y = y2_t
                v = v_
                w = w_
                #X = X_
                Hv = Hv_
                vHv = vHv_
                gT_norm = gT_norm_
                g = g2_
                theta = theta_
            pass

        if self.verbose:
            if is_main_loop_converge:
                self.logger.info(
                    '-' * 100 + f'\nrotation done. time: {time.perf_counter() - t_main:<.4f} s\n'
                )
            else:
                self.logger.warning(
                    '-' * 100 + f'\nWARNING: Some Structures\' Rotation were NOT Converged yet!\n'
                                f'rotation done. time: {time.perf_counter() - t_main:<.4f} s\n'
                )

        # recalc y, g, Hv (Optional)
        #y, g, Hv = fin_diff_hvp(
        #    func,
        #    func_args,
        #    func_kwargs,
        #    grad_func_,
        #    grad_func_args,
        #    grad_func_kwargs,
        #    X,
        #    v,
        #    self.batch_scatter,
        #    is_g_contain_y=is_grad_func_contain_y,
        #    require_grad=require_grad,
        #)
        Hv.mul_(atom_masks)
        g.mul_(atom_masks)
        # curvature, vHv (1, B0, 1)
        vHv = th.sum(
            index_ops.index_inner_product(
                v,
                Hv,
                dim=1,
                batch_indices=self.batch_scatter,
                out_size=n_inner_batch,
            ),
            dim=-1,
            keepdim=True
        )

        return v, y, g, Hv, vHv
