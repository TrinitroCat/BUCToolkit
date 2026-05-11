#  Copyright (c) 2026.4.29, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: Krylov.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import time
import warnings

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.utils._print_formatter import SCIENTIFIC_ARRAY_FORMAT, STRING_ARRAY_FORMAT, FLOAT_ARRAY_FORMAT
from BUCToolkit.utils import index_ops
from BUCToolkit.utils.grad_functions import fin_diff_hvp
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.Bases.BaseMotion import BaseMotion
from BUCToolkit.BatchOptim._BaseOpt import _BaseOpt


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
            eigen_order: int, the number of required minimum eigenvalues.

        Return: y, g, KRYLOV_BASES, KRYLOV_EIGENVAL, KRYLOV_EIGENVEC;
            y: the mean value of function at X, i.e., (f(X + delta * v) + f(X - delta * v))/2
            g: the mean grad of function at X, i.e., (grad(X + delta * v) + grad(X - delta * v))/2
            KRYLOV_BASES: the subspace bases of krylov subspace
            KRYLOV_EIGENVAL: the eigen values of krylov subspace Hessian
            KRYLOV_EIGENVEC: the eigen vectors of krylov subspace Hessian
        """
        t_main = time.perf_counter()
        if func_kwargs is None:
            func_kwargs = dict()
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
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
        X = X.to(self.device)
        # normalize v
        v.mul_(atom_masks)
        v_norm = th.sqrt(th.sum(index_ops.index_inner_product(v, v, 1, self.batch_scatter), dim=-1, keepdim=True))
        v = v / v_norm.index_select(1, self.batch_scatter)

        # initialize
        ############################## BATCHED ALGORITHM ###################################
        # variables with '_' refer to the dynamically changed variables during iteration,
        # and they will in-placed copy into origin variables (i.e., without '_') at the end
        # of each iteration to update data.
        #
        ####################################################################################
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
            index_ops.index_inner_product(v, Hv, dim=1, batch_indices=self.batch_scatter),
            dim=-1,
            keepdim=True
        ) # curvature, vHv (1, B0, 1), essentially the Lanczos alpha
        # grad in the tangent space 1st
        u = Hv - vHv.index_select(1, self.batch_scatter) * v
        beta = th.sqrt(th.sum(index_ops.index_inner_product(
            u, u, 1, self.batch_scatter), dim=-1, keepdim=True)
        )  # i.e., the lanczos beta.
        KRYLOV_BASES = th.zeros((self.subspace_maxdim, v.shape[1], v.shape[2]), device=self.device, dtype=X.dtype)
        KRYLOV_BASES[0] = v[0].clone()
        KRYLOV_HESSIAN = th.zeros(
            (n_true_batch, self.subspace_maxdim, self.subspace_maxdim),
            device=self.device, dtype=X.dtype
        )
        KRYLOV_HESSIAN[:, 0, 0] = vHv.reshape(n_true_batch).clone()
        KRYLOV_EIGENVAL = th.zeros((n_true_batch, self.subspace_maxdim), device=self.device)
        KRYLOV_EIGENVAL[:, 0] = vHv.reshape(n_true_batch).clone()
        KRYLOV_EIGENVEC = th.zeros_like(KRYLOV_HESSIAN)
        KRYLOV_EIGENVEC[:, 0, 0] = 1.
        krylov_eigenval_old = th.full_like(KRYLOV_EIGENVAL, th.inf)
        # cache for dynamically changed batch indices due to convergence, avoiding reallocate mem.
        batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        iter_min = eigen_order
        for i in range(1, self.maxiter_lanczos):
            # threshold. Only need v in the negative cone, i.e., vHv < 0.
            monitor_indx = min(eigen_order, i) - 1
            converge_mask_eig = (
                    th.abs(krylov_eigenval_old[:, monitor_indx] - KRYLOV_EIGENVAL[:, monitor_indx]) < self.Eigen_thres
            ).reshape(1, -1, 1)
            converge_mask_torque = (beta < self.Torque_thres)
            converge_mask = (converge_mask_eig | converge_mask_torque)  # (1, B, 1)
            # print
            #self.logger.debug(f"LANCZOS: H_sub:\n{KRYLOV_HESSIAN}")
            #self.logger.debug(f"LANCZOS: K_EIG_VEC:\n{KRYLOV_EIGENVEC}")
            #self.logger.debug(f"LANCZOS: K_EIG_VAL:\n{KRYLOV_EIGENVAL}")
            if self.verbose > 0:
                self.logger.info(
                    f"Eigen {i:>5d}\n "
                    f"Torque:       {np.array2string(beta.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"target Eig.:  {np.array2string(KRYLOV_EIGENVAL[:, monitor_indx].squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"Energies:     {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                    f"Eig. Conv.:   {np.array2string(converge_mask.squeeze().numpy(force=True), **STRING_ARRAY_FORMAT)}\n "
                    f"TIME:         {time.perf_counter() - t_st:>6.4f} s"
                )
                t_st = time.perf_counter()
            # judge thres
            if th.all(converge_mask):
                is_main_loop_converge = True
                break
            converge_mask_short = converge_mask
            converge_mask = converge_mask_short[:, self.batch_scatter, ...]  # (1, sumB*A, 1)
            # update batch, remove the already converged ones.
            if not self._hold_samples:
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_mask_short.squeeze(),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_mask[0, :, 0])  # (sumB*A, )
                select_mask_short = ~converge_mask_short[0, :, 0]  # (B, )
                n_local_batch = th.sum(select_mask_short)
                #y_ = y[select_mask_short]
                Hv_ = Hv[:, select_mask, :]
                X_ = X[:, select_mask, :]
                v_ = v[:, select_mask, :]
                u_ = u[:, select_mask, :]
                beta_ = beta[:, select_mask_short, :]
                atom_masks_ = atom_masks[:, select_mask, :]
                batch_tensor_ = self.batch_tensor[select_mask_short]
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

            # construction subspace Hessian [[vHv vHw] [wHv wHw]] with shape (B0, 2, 2) for 2nd order precise linear search
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
            # subspace Hessian, It should be tridiagonal.
            wHw_ = th.sum(
                index_ops.index_inner_product(w_, Hw_, dim=1, batch_indices=batch_scatter_),
                dim=-1,
                keepdim=True
            )
            # Lanczos recursion formula
            u_ = Hw_ - wHw_.index_select(1, batch_scatter_) * w_ - beta_.index_select(1, batch_scatter_) * v_
            # re-orthogonalize twice: (K, ba, d) * (1, ba, d) -> (K, b, 1)
            Vu = index_ops.index_reduce(
                th.sum(krylov_bases_[:i] * u_, dim=-1),
                batch_scatter_,
                dim=1,
            ).index_select(1, batch_scatter_)
            u_.sub_(th.einsum("kbd, kb -> bd", krylov_bases_[:i], Vu).unsqueeze(0))
            Vu = index_ops.index_reduce(
                th.sum(krylov_bases_[:i] * u_, dim=-1),
                batch_scatter_,
                dim=1,
            ).index_select(1, batch_scatter_)
            u_.sub_(th.einsum("kbd, kb -> bd", krylov_bases_[:i], Vu).unsqueeze(0))


            # update bases and T, Only save the LOWER TRIANGULAR PART
            krylov_hessian_[:, i, i] = wHw_.reshape(n_local_batch).clone()
            krylov_hessian_[:, i, i - 1] = beta_.reshape(n_local_batch).clone()
            krylov_bases_[i, :, :] = w_[0].clone()

            # update beta
            beta_ = th.sqrt_(th.sum(index_ops.index_inner_product(
                u_, u_, 1, batch_scatter_), dim=-1, keepdim=True)
            )  # i.e., the lanczos beta.

            #sub_eigval_, sub_eigvec_ = th.linalg.eigh(krylov_hessian_)  # (B, i), (B, i, i), default is using the lower triangular part
            sub_eigval_[:, :i + 1], sub_eigvec_[:, :i + 1, :i + 1] = th.linalg.eigh(
                krylov_hessian_[:, :i + 1, :i + 1]
            )

            # update next loop vars
            v_ = w_
            Hv_ = Hw_

            # update origin variables
            if not self._hold_samples:
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

        if self.verbose and is_main_loop_converge:
            self.logger.info(
                '-' * 100 + f'\nrotation done. time: {time.perf_counter() - t_main:<.4f} s\n'
            )
        elif not is_main_loop_converge:
            self.logger.warning(
                '-' * 100 + f'\nWARNING: Some Structures\' Rotation were NOT Converged yet!\n'
                            f'rotation done. time: {time.perf_counter() - t_main:<.4f} s\n'
            )

        # DEBUG
        #H = th.autograd.functional.hessian(func, X)[0].squeeze()
        #eigval, eigvec = th.linalg.eigh(H)
        #print(f"KRYLOV_EIGENVAL: {KRYLOV_EIGENVAL[0, :eigen_order+1]}\nTRUE_EIGENVAL: {eigval[:eigen_order+1]}")

        return y, g, KRYLOV_BASES, KRYLOV_EIGENVAL, KRYLOV_EIGENVEC


class KrylovNewton(BaseMotion):
    """
    Using the krylov subspace Hessian with spectra modification to search 1-order saddle points
    """

    def __init__(
            self,
            E_threshold: float = 1e-3,
            Torque_thres: float = 1.e-2,
            Eigen_thres: float = 1.e-2,
            F_threshold: float = 0.05,
            maxiter_trans: int = 300,
            maxiter_eig: int = 10,
            steplength: float = 0.5,
            steplength_sheme: Literal['trust_region', 'line_newton', 'line_search'] = 'trust_region',
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            morse_index: int = 1,
            neg_spectra_cutoff: float = 0.01,
            pos_spectra_cutoff: float = 0.01,
    ):
        warnings.filterwarnings('always')
        self.E_threshold = float(E_threshold)
        self.Torque_thres = abs(float(Torque_thres))
        self.Eigen_thres = abs(float(Eigen_thres))
        self.F_threshold = float(F_threshold)
        self.maxiter_trans = int(maxiter_trans)
        assert (maxiter_eig > 0) and isinstance(maxiter_eig, int), '`maxiter_rot` must be an integer greater than 0.'
        self.maxiter_eig = int(maxiter_eig)

        self.steplength = float(steplength)
        self._trust_reg_rad_max = 5. * self.steplength
        self._trust_reg_rad_min = 0.1 * self.steplength

        self.dx = float(dx)
        self.device = device
        self.verbose = verbose
        self._morse_index = int(morse_index)
        self.steplength_sheme = steplength_sheme

        if self._morse_index >= self.maxiter_eig:
            raise ValueError(
                f"To solve k-order saddle, one must ensure the krylov subspace dimension sufficiently higher than k, "
                f"but got morse index {self._morse_index} and subspace dim {self.maxiter_eig}."
            )
        elif self._morse_index < 0:
            raise ValueError(f"Morse index should be positive, but got {self._morse_index}.")

        # INNER parameter
        self.neg_spectra_cutoff = - abs(float(neg_spectra_cutoff))  # the negative clamp of k-th spectra
        self.pos_spectra_cutoff = abs(float(pos_spectra_cutoff))    # the positive clamp of (n - k)-th spectra

        self.EigenFinder = FindEigen(
            self.Torque_thres,
            self.Eigen_thres,
            self.maxiter_eig,
            self.dx,
            self.device,
            self.verbose,
            _hold_samples=True
        )

        # logger
        super().__init__()
        self.init_logger('Main.TS')

    def _update_batch(self, mask: th.Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict):
        """
        Default update method for the input of func if the func has non-opt variables, i.e., the identical transform.
        Args:
            mask:

        Returns:

        """
        return func_args, func_kwargs, grad_func_args, grad_func_kwargs

    def _linear_search(self, dX_, ):
        """
        Linear search for dimer translation
        Returns:

        """
        # TODO, adding linear search algo. to determine steplength.
        pass

    def set_batch_updater(
            self,
            method_trans: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]],
            method_rot: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]] | None = None,
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
            method_trans: Callable(mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict) -> Tuple[Tuple, Dict, Tuple, Dict],
            method_rot: batch updater for rotations. the method of updating function arguments for a mask.

        Returns: None
        """
        if method_rot is not None:
            self.EigenFinder.set_batch_updater(method_rot)
        else:
            self.EigenFinder._hold_samples = True
        self._update_batch = method_trans

    def _diag_trust_region(self, vg, dii, g_comp_square, Delta2, tol=1e-4, max_iter=50):
        """
        纯黄金分割法求解对角信赖域子问题，保证收敛。

        Args:
            vg: (k, b) projected gradient on subspace bases.
            dii: (b, k) eigenvalues of subspace metric (diag of D).
            Delta2: (b, 1) squared trust-region radius.
            tol: 收敛容差（基于区间宽度）。
            max_iter: 最大迭代次数（黄金分割收敛较慢，可适当设大）。

        Returns:
            mu: (b, 1) optimal Lagrange multiplier.
        """
        g_comp_square = g_comp_square.to(th.float64)  # (b, 1)
        vg = vg.T.to(th.float64)  # (b, k)
        dii = dii.to(th.float64)  # (b, k)
        #Delta2 = th.as_tensor(Delta2, dtype=th.float64).view(-1, 1)

        # 检查是否需要非零的 mu
        f0 = ((vg / dii)**2).sum(dim=-1, keepdim=True) + g_comp_square - Delta2
        active = (f0 > 0).squeeze(-1)
        if not active.any():
            return th.zeros_like(f0, dtype=th.float32)

        # Bounds
        v_norm_sq = ((dii * vg) ** 2).sum(dim=-1, keepdim=True) + g_comp_square
        mu_min = th.zeros_like(v_norm_sq) + 1e-4
        L = mu_min.clone()
        R = mu_min + th.sqrt(v_norm_sq / Delta2) + 1e-8
        R = th.maximum(R, L + 1e-4)  # 强制 R > L

        # 黄金分割常数
        inv_phi = (2.236067977 - 1.0) / 2.0  # 0.618

        # 初始试探点（为了对称，选两个点之一即可，这里用右分区点）
        # 标准黄金分割搜索用两点，但求根时一个点够了。
        # 我们用 R - inv_phi*(R-L) 保证 f(x) 可能为负时能收缩右边界。
        x = R - inv_phi * (R - L)
        f_x = ((dii * vg / (dii ** 2 + x))**2).sum(dim=-1, keepdim=True) + g_comp_square/(1. + x)**2 - Delta2
        is_converged = False

        for _ in range(max_iter):
            # 收敛条件：区间宽度或函数值足够小
            width = R - L
            converged = (width < tol * Delta2) | (th.abs(f_x) < tol * Delta2)
            #print(th.abs(f_x).squeeze())
            if converged.all():
                is_converged = True
                break

            # 根据 f(x) 的符号收缩区间
            # f(x) > 0 → 根在右侧，左边界右移
            # f(x) < 0 → 根在左侧，右边界左移
            update_L = f_x > 0
            update_R = f_x < 0

            L = th.where(update_L, x, L)
            R = th.where(update_R, x, R)

            # 重新计算试探点，保持区间长度黄金比例
            # 保守起见，每次都基于新的 [L, R] 重新选点
            x = R - inv_phi * (R - L)

            # 计算新函数值
            f_x = ((dii * vg / (dii ** 2 + x))**2).sum(dim=-1, keepdim=True) + g_comp_square/(1. + x)**2 - Delta2

        # 最终 mu 取区间中点（或 x），这里取中点更安全
        mu = (L + R) * 0.5
        #print(f"mu: {mu}")

        if not is_converged:
            self.logger.warning(
                f"Golden-section trust-region not fully converged. "
                f"Max residual: {th.abs(f_x).max():.2e}"
            )

        return mu.to(th.float32)

    def _curve_cond_linesearch(
            self,
            func,
            func_args_,
            func_kwargs_,
            grad_func_,
            grad_func_args_,
            grad_func_kwargs_,
            X_,
            dX_,
            g_,
            steplength,
            max_steplength,
            n_local_batch,
            batch_scatter_,
            is_grad_func_contain_y,
            require_grad,
            atom_masks
    ):
        COEFF = 0.618
        BETA = 0.9
        is_converge = False
        g_norm_ = th.sum(
            index_ops.index_inner_product(g_, g_, 1, batch_scatter_, out_size=n_local_batch),
            dim=-1, keepdim=True
        ).sqrt_()  # (1, b, 1)
        for _ in range(10):
            _X = th.addcmul(X_, atom_masks * dX_, steplength)
            y, g_now_ = self._calc_y_grad(
                _X,
                func,
                func_args_,
                func_kwargs_,
                grad_func_,
                grad_func_args_,
                grad_func_kwargs_,
                require_grad,
                is_grad_func_contain_y,
            )
            g_now_.mul_(atom_masks)
            g_norm_now_ = th.sum(
                index_ops.index_inner_product(g_now_, g_now_, 1, batch_scatter_, out_size=n_local_batch),
                dim=-1, keepdim=True
            ).sqrt_()  # (1, b, 1)
            #print(f"{_}: g_norm_: {g_norm_}, g_norm_now_: {g_norm_now_}")
            converge_mask = (g_norm_now_ < (BETA * g_norm_))
            if th.all(converge_mask):
                is_converge = True
                break
            steplength *= COEFF

        if not is_converge:
            self.logger.warning(f"Line search did not converge.")

        return steplength

    def _newton_steplength(
            self,
            func,
            func_args_,
            func_kwargs_,
            grad_func_,
            grad_func_args_,
            grad_func_kwargs_,
            X_,
            dX_,
            g_,
            steplength,
            max_steplength,
            n_local_batch,
            batch_scatter_,
            is_grad_func_contain_y,
            require_grad,
            atom_masks
    ):
        _y, _g, Hp = fin_diff_hvp(
            func,
            func_args_,
            func_kwargs_,
            grad_func_,
            grad_func_args_,
            grad_func_kwargs_,
            X_,
            dX_,
            batch_scatter_,
            is_g_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            delta=self.dx
        )
        Hp.mul_(atom_masks)
        gp = index_ops.index_inner_product(g_, dX_, 1, batch_scatter_, out_size=n_local_batch)
        pHp = index_ops.index_inner_product(dX_, Hp, 1, batch_scatter_, out_size=n_local_batch)
        steplength_ = (- gp / pHp).clamp_(0.01, max_steplength).index_select(1, batch_scatter_)
        #print(f"gp: {gp}, pHp: {pHp}, gp/pHp: {- gp / pHp}, steplength_: {steplength_.flatten()}")

        return steplength_

    def _find_steplength(
            self
    ):
        pass

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
            extra_krylov_dim: int = 1
    ):
        """
        run krylov search algo.
        Args:
            func:
            X:
            X_diff:
            grad_func:
            func_args:
            func_kwargs:
            grad_func_args:
            grad_func_kwargs:
            is_grad_func_contain_y:
            require_grad:
            output_grad:
            fixed_atom_tensor:
            batch_indices:
            extra_krylov_dim: the number of extra converged eigenvalues higher than `the morse index`
                requiring EigenFiner to solve. It can provide more 2nd order information to accelerate the convergence.

        Returns:

        """
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        if func_kwargs is None:
            func_kwargs = dict()
        func_args = tuple(func_args)
        grad_func_args = tuple(grad_func_args)
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            if len(X.shape) == 2:
                X = X.unsqueeze(0)
            elif len(X.shape) != 3:
                raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        if X_diff is None:
            X_diff = th.randn_like(X)
        elif isinstance(X_diff, th.Tensor):
            if len(X_diff.shape) == 2:
                X_diff = X_diff.unsqueeze(0)
            elif len(X_diff.shape) != 3:
                raise ValueError(f'`X_diff` must be 2D or 3D, but got shape [{X_diff.shape}]')
        else:
            raise TypeError(f'`X_diff` must be torch.Tensor, but occurred {type(X_diff)}.')

        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(grad_func, is_grad_func_contain_y, require_grad)
        # batch_check
        if batch_indices is None:
            raise NotImplementedError(
                f'Regular batch version is not implemented yet. You may specify a `batch_indices` with identity values instead.'
                f'It is fully compatible with regular batches, but merely a little performance loss.'
            )
        n_true_batch, batch_indices, self.batch_tensor, self.batch_scatter, batch_slice_indx = self.handle_batch_indices(
            batch_indices, n_batch, device=self.device
        )

        # Selective dynamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)
        # other check
        if (not isinstance(self.maxiter_trans, int)) or (not isinstance(self.maxiter_eig, int)) \
                or (self.maxiter_trans <= 0) or (self.maxiter_eig <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.maxiter_trans}. It would be an integer greater than 0.')
        extra_krylov_dim = int(extra_krylov_dim)
        if extra_krylov_dim < 1:
            raise ValueError(f"At least one time extra krylov eigenvalues is required to ensure the stability.")

        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)
        X_diff = X_diff.to(self.device)
        v = X_diff.mul(atom_masks)
        plist = list()  # TEST <<<<
        is_main_loop_converge = False
        # initialize
        #   Constants
        steplength_scheme = self.steplength_sheme
        max_steplength = self.steplength
        _stp_cache = th.scalar_tensor(1., device=self.device, dtype=X.dtype)
        EIG_THRES_NEG = th.scalar_tensor(self.neg_spectra_cutoff, dtype=X.dtype, device=self.device)
        EIG_THRES_POS = th.scalar_tensor(self.pos_spectra_cutoff, dtype=X.dtype, device=self.device)
        MORSE_INDEX = self._morse_index
        if n_atom <= (MORSE_INDEX + extra_krylov_dim):
            raise ValueError(f"The sum of the Morse index and extra Krylov dimensions is larger than total free degree.")
        DELTA2 = th.full((n_true_batch, 1), max_steplength**2, device=self.device, dtype=X.dtype)
        IOTA1 = 0.25
        IOTA2 = 0.75
        #   init Krylov
        y, g, KRYLOV_BASES, KRYLOV_EIGENVAL, KRYLOV_EIGENVEC = self.EigenFinder.run(
            func=func,
            X=X,
            v=v,
            grad_func=grad_func_,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            fixed_atom_tensor=atom_masks,
            batch_indices=self.batch_tensor,
            eigen_order=MORSE_INDEX + extra_krylov_dim
        )
        # y: (B, )
        # g: (1, B*A, D)
        # KRYLOV_BASES: (K, B*A, D), K is the Krylov subspaces dimension.
        # KRYLOV_EIGENVAL: (B, K)
        # KRYLOV_EIGENVEC: (B, K, K)
        y_old = th.full_like(y, th.inf, device=self.device)
        # Main loop
        batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        t_st = time.perf_counter()
        with th.no_grad():
            for i in range(self.maxiter_trans):
                plist.append(X[:, None, :, 0].clone().numpy(force=True))  # TEST <<<<<<<<<<<<<
                # Section: check threshold  <<<
                # threshold.
                min_eig = KRYLOV_EIGENVAL[:, 0]  # (B, )
                converge_mask_curve = (min_eig < 0.).reshape(1, -1, 1)
                F_eps = index_ops.index_reduce(
                    th.max(th.abs(g), dim=-1, keepdim=True).values,
                    self.batch_scatter, 1, 'amax', -th.inf
                )  # (1, B, 1)
                E_eps = th.abs(y - y_old)
                converge_mask_g = (F_eps < self.F_threshold)
                converge_mask_e = th.lt(E_eps, self.E_threshold).reshape(1, -1, 1)
                converge_mask = converge_mask_curve & converge_mask_g & converge_mask_e  # (1, B, 1)
                y_old = y.clone()
                # print
                if self.verbose > 0:
                    self.logger.info(
                        f"Translation {i:>5d}\n "
                        f"MAD_Energies: {np.array2string(E_eps.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"MAX_F:        {np.array2string(F_eps.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Curvature:    {np.array2string(min_eig.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Energies:     {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Converged:    {np.array2string(converge_mask.squeeze().numpy(force=True), **STRING_ARRAY_FORMAT)}\n "
                        f"TIME:         {time.perf_counter() - t_st:>6.4f} s"
                    )
                    t_st = time.perf_counter()
                # OUTPUT COORD
                self.handle_arrays_print(
                    self.logger,
                    batch_indices,
                    batch_slice_indx,
                    ((X, g.neg()), ),
                    (('Coordinates', 'Forces'), ),
                    verbose=self.verbose,
                )
                # judge thres
                if th.all(converge_mask):
                    is_main_loop_converge = True
                    break
                converge_mask_short = converge_mask
                converge_mask = converge_mask[:, self.batch_scatter, ...]  # (1, sumB*A, 1)

                # Section: dynamically update batch, remove the already converged ones.
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_mask_short.squeeze(),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_mask[0, :, 0])  # (sumB*A, )
                select_mask_short = ~converge_mask_short[0, :, 0]  # (B, )
                n_local_batch = th.sum(select_mask_short)
                X_ = X[:, select_mask, :]
                g_ = g[:, select_mask, :]  # (1, ba, D)
                krylov_bases_ = KRYLOV_BASES[:, select_mask, :]        # (K, ba, D)
                sub_eigval_ = KRYLOV_EIGENVAL[select_mask_short, ...]  # (b, K)
                sub_eigvec_ = KRYLOV_EIGENVEC[select_mask_short, ...]  # (b, K, K)

                atom_masks_ = atom_masks[:, select_mask, :]
                batch_tensor_ = self.batch_tensor[select_mask_short]
                batch_scatter_ = th.repeat_interleave(
                    batch_tensor_indx_cache[:len(batch_tensor_)],
                    batch_tensor_,
                    dim=0
                )

                # Section: Transition to the saddle  <<<
                # Krylov subspace Newton Search
                #   spectra modification. The unconverged Krylov eigenvec are not reliable, thus dropping them.
                spectra_cut_off = (
                        MORSE_INDEX * sub_eigval_[:, 0].abs() +
                        extra_krylov_dim * sub_eigval_[:, -1] +
                        batch_tensor_ - MORSE_INDEX - extra_krylov_dim
                ) / batch_tensor_
                spectra_cut_off.unsqueeze_(-1)
                print(f"neg_cut_val: {spectra_cut_off}")
                T = th.zeros_like(sub_eigval_[:, :MORSE_INDEX + extra_krylov_dim])
                T[:, :MORSE_INDEX] = sub_eigval_[:, :MORSE_INDEX].clamp(-spectra_cut_off, EIG_THRES_NEG)
                T[:, MORSE_INDEX:] = sub_eigval_[:, MORSE_INDEX:MORSE_INDEX + extra_krylov_dim].clamp(EIG_THRES_POS, spectra_cut_off)
                krylov_bases_cut_ = krylov_bases_[:MORSE_INDEX + extra_krylov_dim]
                Vg = index_ops.index_reduce(
                    th.sum(krylov_bases_cut_ * g_, dim=-1),
                    batch_scatter_,
                    dim=1,
                    out_size=n_local_batch
                )  # (K, b)
                # the complement positive-definite space
                #   x' += a * (I - V V^T) g, a > 0.
                dX_complement = g_ - th.einsum("kbd, kb -> bd", krylov_bases_cut_, Vg.index_select(1, batch_scatter_))  # (1, ba, D)
                # self.logger.debug(f"TRANSLATION: dX_complement:\n{dX_complement}")
                # complement trust-reg:
                dX_comp_square_ = th.sum(
                    index_ops.index_inner_product(dX_complement, dX_complement, 1, batch_scatter_, out_size=n_local_batch),
                    dim=-1, keepdim=True
                )  # (1, b, 1)
                #complement_steplength = (DELTA_cmp / dX_comp_norm_).clamp_max_(1.).index_select(1, batch_scatter_)
                if steplength_scheme == 'trust_region':
                    #   Section: trust region search (K, ba, D) (1, ba, D)
                    DELTA2_ = DELTA2[select_mask_short, ...]  # (b, 1)
                    steplength_ = _stp_cache
                    mu = self._diag_trust_region(Vg, T, dX_comp_square_[0], DELTA2_, )*0.1
                    print(f"{i}: mu = {mu}")
                    #(D + mu D ^ -1)
                    T_inv = (T + mu * T.reciprocal()).reciprocal_()
                    complement_steplength = (mu + 1.).reciprocal_().index_select(0, batch_scatter_).unsqueeze(0)
                    print(f"{i}: T_inv = {T_inv}")

                else:
                    T_inv = T.reciprocal()
                    complement_steplength = _stp_cache
                #   x' = - V_nk Q_kk Ainv_kk Q^T V^T g_n
                sub_eigvec_cut_ = sub_eigvec_[:, :MORSE_INDEX + extra_krylov_dim, :MORSE_INDEX + extra_krylov_dim]
                A_ = sub_eigvec_cut_ @ (T_inv.unsqueeze(-1) * sub_eigvec_cut_.mT)  # (b, K, K)
                AVg = th.einsum("bij, jb -> bi", A_, Vg).index_select(0, batch_scatter_)  # (ba, K)
                dX_tangent_ = th.einsum("kbd, bk -> bd", krylov_bases_cut_, AVg).unsqueeze(0)
                dX_ = th.addcmul(dX_tangent_, dX_complement, complement_steplength).neg_()

                dX_norm_ = th.sum(
                    index_ops.index_inner_product(dX_, dX_, 1, batch_scatter_, out_size=n_local_batch),
                    dim=-1, keepdim=True
                ).sqrt_()  # (1, B, 1)

                #   where is positive definition zone
                is_neg_curv = (sub_eigval_[:, 0] < 0.).index_select(0, batch_scatter_).reshape(1, -1, 1)
                # steplength search
                if steplength_scheme == 'line_search':
                    steplength_ = th.full((1, X_.shape[1], 1), max_steplength, dtype=X.dtype, device=X.device)
                    steplength_ = self._curve_cond_linesearch(
                        func,
                        func_args_,
                        func_kwargs_,
                        grad_func_,
                        grad_func_args_,
                        grad_func_kwargs_,
                        X_,
                        dX_,
                        g_,
                        steplength_,
                        max_steplength,
                        n_local_batch,
                        batch_scatter_,
                        is_grad_func_contain_y,
                        require_grad,
                        atom_masks_
                    )
                elif steplength_scheme == 'line_newton':
                    steplength_ = self._newton_steplength(
                        func,
                        func_args_,
                        func_kwargs_,
                        grad_func_,
                        grad_func_args_,
                        grad_func_kwargs_,
                        X_,
                        dX_,
                        g_,
                        max_steplength,
                        max_steplength,
                        n_local_batch,
                        batch_scatter_,
                        is_grad_func_contain_y,
                        require_grad,
                        atom_masks_
                    )
                elif steplength_scheme == 'trust_region':
                    # prepare updating trust-region radii Delta
                    g_norm_ = th.sum(
                        index_ops.index_inner_product(g_, g_, 1, batch_scatter_, out_size=n_local_batch),
                        dim=-1, keepdim=True
                    ).sqrt_()  # (1, B, 1)
                    # (b, 1)/((b, k)**2 + (b, 1))
                    g_Gg_norm_ = (th.einsum('bk,bk,bk->b', mu / (T ** 2 + mu), Vg.T, Vg.T)
                                  + (mu / (1. + mu)) ** 2 * dX_comp_square_[0]).sqrt_().reshape(1, -1, 1)

                    predicted_grad_desc_ = g_norm_ - g_Gg_norm_
                else:
                    raise NotImplementedError

                ##   main update
                dX_ = th.where(
                    is_neg_curv,
                    dX_,
                    dX_ + 0.1 * th.randn_like(dX_) * dX_norm_.index_select(1, batch_scatter_)
                )

                X_.addcmul_(steplength_ * dX_, atom_masks_)
                # reinsurance
                print(f"{i}: displacement length = {dX_norm_}")
                _small_dX_ = (dX_norm_ < 1.e-6)  # (1, B0, 1)
                if th.any(_small_dX_):
                    _too_small_step_indx = th.where(select_mask_short)[0][th.where(_small_dX_)[0]]
                    warnings.warn(
                        RuntimeWarning(
                        f'Convergence is not met while the steplengths of {_too_small_step_indx.tolist()}-th structure(s) are 0. '
                        )
                    )
                    if th.all(_small_dX_):
                        self.logger.error('ERROR: All unconverged structures reached 0 steplength. LOOP BREAK.')
                        break

                # Section: Find Eigen at new points  <<<
                # update initial guess v_
                #   v_ = Q V[:, 0], the eigenvec with min eigenval given by last Lanczos iteration
                v_ = th.einsum(
                    "kbd, bk -> bd",
                    krylov_bases_cut_, sub_eigvec_cut_[:, :, 0].index_select(0, batch_scatter_)
                ).unsqueeze(0) + g_  # (1, ba, D)
                #   Lanczos finder
                y_, g_, krylov_bases_, sub_eigval_, sub_eigvec_ = self.EigenFinder.run(
                    func=func,
                    X=X_,
                    v=v_,
                    grad_func=grad_func_,
                    func_args=func_args_,
                    func_kwargs=func_kwargs_,
                    grad_func_args=grad_func_args_,
                    grad_func_kwargs=grad_func_kwargs_,
                    is_grad_func_contain_y=is_grad_func_contain_y,
                    require_grad=require_grad,
                    fixed_atom_tensor=atom_masks_,
                    batch_indices=batch_tensor_,
                    eigen_order=MORSE_INDEX + extra_krylov_dim
                )
                print(f"{i}: Gradient norm = {th.linalg.norm(g_)}")

                # update origin variables
                select_indices = th.where(select_mask)[0]
                select_indices_short = th.where(select_mask_short)[0]
                y.index_copy_(0, select_indices_short, y_)
                v.index_copy_(1, select_indices, v_)
                X.index_copy_(1, select_indices, X_)
                g.index_copy_(1, select_indices, g_)
                KRYLOV_BASES.index_copy_(1, select_indices, krylov_bases_)
                KRYLOV_EIGENVEC.index_copy_(0, select_indices_short, sub_eigvec_)
                KRYLOV_EIGENVAL.index_copy_(0, select_indices_short, sub_eigval_)

                if self.steplength_sheme == 'trust_region':
                    # Section Now update trust-region radii Delta
                    g_norm_new_ = th.sum(
                        index_ops.index_inner_product(g_, g_, 1, batch_scatter_, out_size=n_local_batch),
                        dim=-1, keepdim=True
                    ).sqrt_()  # (1, B, 1)
                    rho = th.where(
                        predicted_grad_desc_ < 0.,
                        0.,
                        (g_norm_ - g_norm_new_) / predicted_grad_desc_
                    )
                    _DELTA = DELTA2_.sqrt()
                    _DELTA = th.where(
                        rho >= IOTA2,
                        _DELTA + 0.5 * (self._trust_reg_rad_max - _DELTA),
                        th.where(
                            rho >= IOTA1,
                            _DELTA,
                            _DELTA - 0.5 * (_DELTA - self._trust_reg_rad_min)
                        )
                    )
                    DELTA2_ = _DELTA.square_().reshape(-1, 1)
                    DELTA2.index_copy_(0, select_indices_short, DELTA2_)
                    print(f"Delta2 now: {DELTA2_}")

                # Section DEBUG
                #var_dict = {}
                #for var_name, var in locals().items():
                #    if isinstance(var, th.Tensor):
                #        mem = var.untyped_storage()
                #        if mem not in var_dict:
                #            var_dict[mem] = var_name
                #        else:
                #            print('*'*80 + f"\n{var_name} and {var_dict[mem]} share the same memory.\n" + '*'*80)
                # Section END <<<<<<<<

        if self.verbose:
            if is_main_loop_converge:
                self.logger.info('-' * 100 + '\nAll Structures Were Converged.\nMAIN LOOP Done.')
            else:
                self.logger.info('-' * 100 + '\nSome Structures were NOT Converged yet!\nMAIN LOOP Done.')

        if output_grad:
            return y, X, g
        else:
            return y, X, plist  # TEST <<<<<<


class KrylovDynamics(BaseMotion):
    """
    Using the krylov subspace Hessian with spectra modification to search 1-order saddle points
    """

    def __init__(
            self,
            E_threshold: float = 1e-3,
            Torque_thres: float = 1.e-2,
            Eigen_thres: float = 1.e-2,
            F_threshold: float = 0.05,
            maxiter_trans: int = 300,
            maxiter_eig: int = 10,
            steplength: float = 0.5,
            steplength_sheme: Literal['trust_region', 'line_newton', 'line_search'] = 'trust_region',
            dx: float = 1.e-2,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            morse_index: int = 1,
            neg_spectra_cutoff: float = 0.1,
            pos_spectra_cutoff: float = 0.1,
            alpha: float = 0.1,
            alpha_fac: float = 0.99,
            fac_inc: float = 1.1,
            fac_dec: float = 0.5,
            N_min: int = 5,
    ):
        warnings.filterwarnings('always')
        self.E_threshold = float(E_threshold)
        self.Torque_thres = abs(float(Torque_thres))
        self.Eigen_thres = abs(float(Eigen_thres))
        self.F_threshold = float(F_threshold)
        self.maxiter_trans = int(maxiter_trans)
        assert (maxiter_eig > 0) and isinstance(maxiter_eig, int), '`maxiter_rot` must be an integer greater than 0.'
        self.maxiter_eig = int(maxiter_eig)
        self.max_steplength = float(steplength) * 5.
        self.dx = float(dx)
        self.device = device
        self.verbose = verbose
        self._morse_index = int(morse_index)
        self.steplength_sheme = steplength_sheme

        self.alpha = alpha
        self.alpha_fac = alpha_fac
        self.fac_inc = fac_inc
        self.fac_dec = fac_dec
        self.N_min = N_min
        self.t_init = float(steplength)

        if self._morse_index >= self.maxiter_eig:
            raise ValueError(
                f"To solve k-order saddle, one must ensure the krylov subspace dimension sufficiently higher than k, "
                f"but got morse index {self._morse_index} and subspace dim {self.maxiter_eig}."
            )
        elif self._morse_index < 0:
            raise ValueError(f"Morse index should be positive, but got {self._morse_index}.")

        # INNER parameter
        self.neg_spectra_cutoff = - abs(float(neg_spectra_cutoff))  # the negative clamp of k-th spectra
        self.pos_spectra_cutoff = abs(float(pos_spectra_cutoff))    # the positive clamp of (n - k)-th spectra

        self.EigenFinder = FindEigen(
            self.Torque_thres,
            self.Eigen_thres,
            self.maxiter_eig,
            self.dx,
            self.device,
            self.verbose,
            _hold_samples=True
        )

        # logger
        super().__init__()
        self.init_logger('Main.TS')

    def _update_batch(self, mask: th.Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict):
        """
        Default update method for the input of func if the func has non-opt variables, i.e., the identical transform.
        Args:
            mask:

        Returns:

        """
        return func_args, func_kwargs, grad_func_args, grad_func_kwargs

    def _linear_search(self, dX_, ):
        """
        Linear search for dimer translation
        Returns:

        """
        # TODO, adding linear search algo. to determine steplength.
        pass

    def set_batch_updater(
            self,
            method_trans: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]],
            method_rot: Callable[[th.Tensor, Tuple | None, Dict | None, Tuple | None, Dict | None], Tuple[Tuple, Dict, Tuple, Dict]] | None = None,
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
            method_trans: Callable(mask: Tensor, func_args: Tuple, func_kwargs: Dict, grad_func_args: Tuple, grad_func_kwargs: Dict) -> Tuple[Tuple, Dict, Tuple, Dict],
            method_rot: batch updater for rotations. the method of updating function arguments for a mask.

        Returns: None
        """
        if method_rot is not None:
            self.EigenFinder.set_batch_updater(method_rot)
        else:
            self.EigenFinder._hold_samples = True
        self._update_batch = method_trans

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
            extra_krylov_dim: int = 1
    ):
        """
        run krylov search algo.
        Args:
            func:
            X:
            X_diff:
            grad_func:
            func_args:
            func_kwargs:
            grad_func_args:
            grad_func_kwargs:
            is_grad_func_contain_y:
            require_grad:
            output_grad:
            fixed_atom_tensor:
            batch_indices:
            extra_krylov_dim: the number of extra converged eigenvalues higher than `the morse index`
                requiring EigenFiner to solve. It can provide more 2nd order information to accelerate the convergence.

        Returns:

        """
        if grad_func_kwargs is None:
            grad_func_kwargs = dict()
        if func_kwargs is None:
            func_kwargs = dict()
        func_args = tuple(func_args)
        grad_func_args = tuple(grad_func_args)
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            if len(X.shape) == 2:
                X = X.unsqueeze(0)
            elif len(X.shape) != 3:
                raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        if X_diff is None:
            X_diff = th.randn_like(X)
        elif isinstance(X_diff, th.Tensor):
            if len(X_diff.shape) == 2:
                X_diff = X_diff.unsqueeze(0)
            elif len(X_diff.shape) != 3:
                raise ValueError(f'`X_diff` must be 2D or 3D, but got shape [{X_diff.shape}]')
        else:
            raise TypeError(f'`X_diff` must be torch.Tensor, but occurred {type(X_diff)}.')

        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(grad_func, is_grad_func_contain_y, require_grad)
        # batch_check
        if batch_indices is None:
            raise NotImplementedError(
                f'Regular batch version is not implemented yet. You may specify a `batch_indices` with identity values instead.'
                f'It is fully compatible with regular batches, but merely a little performance loss.'
            )
        n_true_batch, batch_indices, self.batch_tensor, self.batch_scatter, batch_slice_indx = self.handle_batch_indices(
            batch_indices, n_batch, device=self.device
        )

        # Selective dynamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)
        # other check
        if (not isinstance(self.maxiter_trans, int)) or (not isinstance(self.maxiter_eig, int)) \
                or (self.maxiter_trans <= 0) or (self.maxiter_eig <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.maxiter_trans}. It would be an integer greater than 0.')
        extra_krylov_dim = int(extra_krylov_dim)
        if extra_krylov_dim < 1:
            raise ValueError(f"At least one time extra krylov eigenvalues is required to ensure the stability.")

        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.zero_grad()
        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)
        X_diff = X_diff.to(self.device)
        v = X_diff.mul(atom_masks)
        plist = list()  # TEST <<<<
        is_main_loop_converge = False
        # initialize
        #   Constants
        _stp_cache = th.scalar_tensor(1., device=self.device, dtype=X.dtype)
        EIG_THRES_NEG = self.neg_spectra_cutoff
        EIG_THRES_POS = self.pos_spectra_cutoff
        MORSE_INDEX = self._morse_index
        if n_atom <= (MORSE_INDEX + extra_krylov_dim):
            raise ValueError(f"The sum of the Morse index and extra Krylov dimensions is larger than total free degree.")
        t = th.full((1, n_atom, 1), self.t_init, device=self.device)
        a = th.full((1, n_atom, 1), self.alpha, device=self.device)
        n_count = th.zeros((1, n_atom, 1), dtype=th.int, device=self.device)
        veloc = th.zeros_like(X, device=self.device)
        alpha = self.alpha
        alpha_fac = self.alpha_fac
        fac_inc = self.fac_inc
        fac_dec = self.fac_dec
        N_min = self.N_min

        #   init Krylov
        y, g, KRYLOV_BASES, KRYLOV_EIGENVAL, KRYLOV_EIGENVEC = self.EigenFinder.run(
            func=func,
            X=X,
            v=v,
            grad_func=grad_func_,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            fixed_atom_tensor=atom_masks,
            batch_indices=self.batch_tensor,
            eigen_order=MORSE_INDEX + extra_krylov_dim
        )
        # y: (B, )
        # g: (1, B*A, D)
        # KRYLOV_BASES: (K, B*A, D), K is the Krylov subspaces dimension.
        # KRYLOV_EIGENVAL: (B, K)
        # KRYLOV_EIGENVEC: (B, K, K)
        y_old = th.full_like(y, th.inf, device=self.device)
        # Main loop
        batch_tensor_indx_cache = th.arange(0, len(self.batch_tensor), dtype=th.int64, device=self.device)
        t_st = time.perf_counter()
        with th.no_grad():
            for i in range(self.maxiter_trans):
                plist.append(X[:, None, :, 0].clone().numpy(force=True))  # TEST <<<<<<<<<<<<<
                # Section: check threshold  <<<
                # threshold.
                min_eig = KRYLOV_EIGENVAL[:, 0]  # (B, )
                converge_mask_curve = (min_eig < 0.).reshape(1, -1, 1)
                F_eps = index_ops.index_reduce(
                    th.max(th.abs(g), dim=-1, keepdim=True).values,
                    self.batch_scatter, 1, 'amax', -th.inf
                )  # (1, B, 1)
                E_eps = th.abs(y - y_old)
                converge_mask_g = (F_eps < self.F_threshold)
                converge_mask_e = th.lt(E_eps, self.E_threshold).reshape(1, -1, 1)
                converge_mask = converge_mask_curve & converge_mask_g & converge_mask_e  # (1, B, 1)
                y_old = y.clone()
                # print
                if self.verbose > 0:
                    self.logger.info(
                        f"Translation {i:>5d}\n "
                        f"MAD_Energies: {np.array2string(E_eps.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"MAX_F:        {np.array2string(F_eps.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Curvature:    {np.array2string(min_eig.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Energies:     {np.array2string(y.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                        f"Converged:    {np.array2string(converge_mask.squeeze().numpy(force=True), **STRING_ARRAY_FORMAT)}\n "
                        f"TIME:         {time.perf_counter() - t_st:>6.4f} s"
                    )
                    t_st = time.perf_counter()
                # OUTPUT COORD
                self.handle_arrays_print(
                    self.logger,
                    batch_indices,
                    batch_slice_indx,
                    ((X, g.neg()), ),
                    (('Coordinates', 'Forces'), ),
                    verbose=self.verbose,
                )
                # judge thres
                if th.all(converge_mask):
                    is_main_loop_converge = True
                    break
                converge_mask_short = converge_mask
                converge_mask = converge_mask[:, self.batch_scatter, ...]  # (1, sumB*A, 1)

                # Section: dynamically update batch, remove the already converged ones.
                func_args_, func_kwargs_, grad_func_args_, grad_func_kwargs_ = self._update_batch(
                    ~converge_mask_short.squeeze(),
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs
                )
                select_mask = ~(converge_mask[0, :, 0])  # (sumB*A, )
                select_mask_short = ~converge_mask_short[0, :, 0]  # (B, )
                n_local_batch = th.sum(select_mask_short)
                X_ = X[:, select_mask, :]
                g_ = g[:, select_mask, :]  # (1, ba, D)
                krylov_bases_ = KRYLOV_BASES[:, select_mask, :]        # (K, ba, D)
                sub_eigval_ = KRYLOV_EIGENVAL[select_mask_short, ...]  # (b, K)
                sub_eigvec_ = KRYLOV_EIGENVEC[select_mask_short, ...]  # (b, K, K)

                atom_masks_ = atom_masks[:, select_mask, :]
                batch_tensor_ = self.batch_tensor[select_mask_short]
                batch_scatter_ = th.repeat_interleave(
                    batch_tensor_indx_cache[:len(batch_tensor_)],
                    batch_tensor_,
                    dim=0
                )

                t_: th.Tensor = t[:, select_mask, :]  # (1, sumN, 1)
                a_ = a[:, select_mask, :]  # (1, sumN, 1)
                n_count_ = n_count[:, select_mask, :]
                veloc_ = veloc[:, select_mask, :]

                # Section: Transition to the saddle  <<<
                # Krylov subspace Newton Search
                #   spectra modification. The unconverged Krylov eigenvec are not reliable, thus dropping them.
                T = th.zeros_like(sub_eigval_[:, :MORSE_INDEX + extra_krylov_dim])
                T[:, :MORSE_INDEX] = sub_eigval_[:, :MORSE_INDEX].clamp_(-1., EIG_THRES_NEG)
                T[:, MORSE_INDEX:] = th.where(
                    sub_eigval_[:, MORSE_INDEX:MORSE_INDEX + extra_krylov_dim] < EIG_THRES_POS,
                    EIG_THRES_POS,
                    sub_eigval_[:, MORSE_INDEX:MORSE_INDEX + extra_krylov_dim],
                )
                T_inv = T.reciprocal()

                krylov_bases_cut_ = krylov_bases_[:MORSE_INDEX + extra_krylov_dim]
                Vg = index_ops.index_reduce(
                    th.sum(krylov_bases_cut_ * g_, dim=-1),
                    batch_scatter_,
                    dim=1,
                    out_size=n_local_batch
                )  # (K, b)
                # the complement positive-definite space
                #   x' += a * (I - V V^T) g, a > 0.
                dX_complement = g_ - th.einsum("kbd, kb -> bd", krylov_bases_cut_, Vg.index_select(1, batch_scatter_))  # (1, ba, D)
                complement_steplength = _stp_cache
                #   x' = - V_nk Q_kk Ainv_kk Q^T V^T g_n
                sub_eigvec_cut_ = sub_eigvec_[:, :MORSE_INDEX + extra_krylov_dim, :MORSE_INDEX + extra_krylov_dim]
                A_ = sub_eigvec_cut_ @ (T_inv.unsqueeze(-1) * sub_eigvec_cut_.mT)  # (b, K, K)
                #self.logger.debug(f"TRANSLATION: T_inv:\n{T_inv}")
                #self.logger.debug(f"TRANSLATION: A:\n{A_}")
                #self.logger.debug(f"TRANSLATION: K_EIG_VEC:\n{sub_eigvec_}")
                #self.logger.debug(f"TRANSLATION: K_EIG_VAL:\n{sub_eigval_}")
                #self.logger.debug(f"TRANSLATION: K_BASES:{krylov_bases_.shape}\n{krylov_bases_}")

                AVg = th.einsum("bij, jb -> bi", A_, Vg).index_select(0, batch_scatter_)  # (ba, K)
                dX_tangent_ = th.einsum("kbd, bk -> bd", krylov_bases_cut_, AVg).unsqueeze(0)
                #self.logger.debug(f"TRANSLATION: dX_tengent:\n{dX_tangent_}")
                Ginv_g_ = th.add(dX_tangent_, dX_complement)  # (1, ba, d)
                F_ = Ginv_g_.neg() * atom_masks_  # reuse the memory of dX_tangent_

                F_norm_ = th.sum(
                    index_ops.index_inner_product(F_, F_, 1, batch_scatter_, out_size=n_local_batch),
                    dim=-1, keepdim=True
                ).sqrt_()  # (1, B, 1)

                #   where is positive definition zone
                is_neg_curv = (sub_eigval_[:, 0] < 0.).index_select(0, batch_scatter_).reshape(1, -1, 1)
                # steplength search
                # Section <<<<<<<<<<<<<<<<<<<<<
                # (1, sumN, n_dim)
                F_hat_ = F_ / F_norm_.index_select(1, batch_scatter_)
                # (1, b, 1)
                power_ = index_ops.index_reduce(
                    th.sum(F_hat_ * veloc_, dim=-1, keepdim=True),
                    batch_scatter_,
                    dim=1,
                    out_size=n_local_batch
                ).index_select(1, batch_scatter_)
                # (1, sumN, 1)
                v_norm_ = th.sum(
                    index_ops.index_inner_product(
                        veloc_,
                        veloc_,
                        dim=1,
                        batch_indices=batch_scatter_
                    ),
                    dim=-1,
                    keepdim=True
                ).sqrt_().index_select(1, batch_scatter_)
                # update velocity: v = v * (1 - a) + a * |v| * \hat{F}
                F_hat_.mul_(v_norm_)
                veloc_.mul_((1. - a_))
                veloc_.addcmul_(a_, F_hat_)
                # if P > 0
                n_count_ += th.where(power_ > 0., 1, -n_count_)  # (1, sumN, 1)
                is_ncount_gt_Nmin = n_count_ >= N_min
                #
                new_t_ = (t_ * fac_inc).clamp_max_(self.max_steplength)
                t_ = th.where(is_ncount_gt_Nmin, new_t_, t_)
                a_ = th.where(is_ncount_gt_Nmin, (a_ * alpha_fac), a_)
                # if P <= 0.
                is_p_lt_0 = power_ <= 0.
                t_ = th.where(is_p_lt_0, (t_ * fac_dec), t_)
                veloc_.masked_fill_(
                    is_p_lt_0,
                    0.
                )
                a_.masked_fill_(is_p_lt_0, alpha)

                veloc_.addcdiv_(F_, t_, value=9.64853329045427e-3)

                # Section END
                X_.addcmul_(veloc_, t_)

                # Section: Find Eigen at new points  <<<
                # update initial guess v_
                #   v_ = Q V[:, 0], the eigenvec with min eigenval given by last Lanczos iteration
                v_ = th.einsum(
                    "kbd, bk -> bd",
                    krylov_bases_cut_, sub_eigvec_cut_[:, :, 0].index_select(0, batch_scatter_)
                ).unsqueeze(0) + g_  # (1, ba, D)
                #   Lanczos finder
                y_, g_, krylov_bases_, sub_eigval_, sub_eigvec_ = self.EigenFinder.run(
                    func=func,
                    X=X_,
                    v=v_,
                    grad_func=grad_func_,
                    func_args=func_args_,
                    func_kwargs=func_kwargs_,
                    grad_func_args=grad_func_args_,
                    grad_func_kwargs=grad_func_kwargs_,
                    is_grad_func_contain_y=is_grad_func_contain_y,
                    require_grad=require_grad,
                    fixed_atom_tensor=atom_masks_,
                    batch_indices=batch_tensor_,
                    eigen_order=MORSE_INDEX + extra_krylov_dim
                )
                print(f"{i}: Gradient norm = {th.linalg.norm(g_)}")

                # update origin variables
                select_indices = th.where(select_mask)[0]
                select_indices_short = th.where(select_mask_short)[0]
                y.index_copy_(0, select_indices_short, y_)
                v.index_copy_(1, select_indices, v_)
                X.index_copy_(1, select_indices, X_)
                g.index_copy_(1, select_indices, g_)
                KRYLOV_BASES.index_copy_(1, select_indices, krylov_bases_)
                KRYLOV_EIGENVEC.index_copy_(0, select_indices_short, sub_eigvec_)
                KRYLOV_EIGENVAL.index_copy_(0, select_indices_short, sub_eigval_)

                t.index_copy_(1, select_indices, t_)
                a.index_copy_(1, select_indices, a_)
                n_count.index_copy_(1, select_indices, n_count_)
                veloc.index_copy_(1, select_indices, veloc_)

        if self.verbose:
            if is_main_loop_converge:
                self.logger.info('-' * 100 + '\nAll Structures Were Converged.\nMAIN LOOP Done.')
            else:
                self.logger.info('-' * 100 + '\nSome Structures were NOT Converged yet!\nMAIN LOOP Done.')

        if output_grad:
            return y, X, g
        else:
            return y, X, plist  # TEST <<<<<<

