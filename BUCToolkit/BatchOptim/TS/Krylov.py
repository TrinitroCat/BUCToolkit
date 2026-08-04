"""Krylov transition-state searches implemented on the optimizer framework.

Both methods use ``_BaseOpt`` only for the outer optimization lifecycle.  A
direction update is one complete transition-state iteration: it constructs a
spectrally modified direction, changes the active coordinates in place, solves
the low Hessian modes at the new coordinates, and publishes that model result
through the existing line-search cache.  This ownership convention is why the
classes declare ``_is_inplace_update = True``.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple
import warnings

import torch as th
from torch import nn

from BUCToolkit.Bases.StdContainer import StdContainer
from BUCToolkit.BatchOptim._BaseOpt import _BaseOpt
from BUCToolkit.BatchOptim.TS._eigen_solver import FindEigen
from BUCToolkit.utils import index_ops
from BUCToolkit.utils.exceptions import IterationStuckError
from BUCToolkit.utils.grad_functions import fin_diff_hvp


class _KrylovBase(_BaseOpt):
    """Shared BaseOpt lifecycle and eigen-state ownership for Krylov methods.

    Full-batch attributes have names without a leading local marker, while
    ``_eigenval`` and related attributes are views/copies for the structures
    still active in the current BaseOpt iteration. The active results are
    scattered back only after the in-place translation and eigen solve finish.
    """

    def __init__(
            self,
            iter_scheme: str,
            E_threshold: float,
            Torque_thres: float,
            Eigen_thres: float,
            F_threshold: float,
            maxiter_trans: int,
            maxiter_eig: int,
            steplength: float,
            dx: float = 0.01,
            output_file: str | None = None,
            device: str | th.device = 'cpu',
            verbose: int = 1,
            morse_index: int = 1,
            neg_spectra_cutoff: float = -0.1,
            pos_spectra_cutoff: float = 0.1,
    ) -> None:
        self.Torque_thres = abs(float(Torque_thres))
        self.Eigen_thres = abs(float(Eigen_thres))
        self.maxiter_trans = int(maxiter_trans)
        if not isinstance(maxiter_eig, int) or maxiter_eig <= 1:
            raise ValueError(
                f'maxiter_eig must be an integer greater than 1, but got {maxiter_eig}.'
            )
        self.maxiter_eig = maxiter_eig
        self.dx = float(dx)
        self._morse_index = int(morse_index)
        if self._morse_index < 0:
            raise ValueError(f'Morse index should be positive, but got {self._morse_index}.')
        if self._morse_index >= self.maxiter_eig:
            raise ValueError(
                'The Krylov subspace dimension must exceed the Morse index, '
                f'but got {self.maxiter_eig} and {self._morse_index}.'
            )
        self.neg_spectra_cutoff = -abs(float(neg_spectra_cutoff))
        self.pos_spectra_cutoff = abs(float(pos_spectra_cutoff))

        super().__init__(
            iter_scheme=iter_scheme,
            E_threshold=float(E_threshold),
            F_threshold=float(F_threshold),
            maxiter=self.maxiter_trans,
            linesearch='None',
            steplength=float(steplength),
            output_file=output_file,
            use_bb=False,
            device=device,
            verbose=verbose,
        )
        # A Krylov direction update also translates X and evaluates the model
        # at the accepted point. BaseOpt must therefore bypass its ordinary
        # line-search/addition path and consume the stored energy and gradient.
        self._is_inplace_update = True
        self.EigenFinder = FindEigen(
            self.Torque_thres,
            self.Eigen_thres,
            self.maxiter_eig,
            self.dx,
            self.device,
            self.verbose,
            _hold_samples=True,
        )

        self._X_diff_init: th.Tensor | None = None
        self._extra_krylov_dim = 1
        self._func: Callable | None = None
        self._grad_func: Callable | None = None
        self._is_grad_func_contain_y: bool | None = None
        self._require_grad: bool | None = None
        self.eigenval: th.Tensor | None = None
        self.eigenvec: th.Tensor | None = None
        self._eigenval: th.Tensor | None = None
        self._eigenvec: th.Tensor | None = None
        self._local_extra_converge_mask: th.Tensor | None = None

    def set_batch_updater(
            self,
            method_trans: Callable,
            method_rot: Callable | None = None,
    ) -> None:
        """Register translation and eigen-solver batch updaters."""
        super().set_batch_updater(method_trans, method_trans)
        # Translation and finite-difference eigen evaluations may require
        # different graph/data rebuilders. Without a rotation updater the
        # eigen solver preserves the complete active translation batch.
        if method_rot is None:
            self.EigenFinder._hold_samples = True
        else:
            self.EigenFinder.set_batch_updater(method_rot)

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
            extra_krylov_dim: int = 1,
    ):
        """Run a Krylov transition-state search.

        ``X_diff`` seeds the first Lanczos solve. ``extra_krylov_dim`` is the
        number of positive-side Ritz modes retained in addition to the
        requested unstable Morse subspace. Irregular batches are required:
        structure-layout state is indexed by ``batch_indices``, while atom-
        layout coordinates and eigenvectors use BaseOpt's scatter indices.

        The initial eigensolve supplies the checked energy and gradient, so it
        replaces rather than supplements BaseOpt's ordinary first evaluation.
        Subsequent eigensolves likewise supply the accepted state after each
        in-place translation.
        """
        if not isinstance(X, th.Tensor):
            raise TypeError(f'X must be torch.Tensor, but occurred {type(X)}.')
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim != 3:
            raise ValueError(f'X must be 2D or 3D, but got shape {X.shape}.')
        if X_diff is None:
            X_diff = th.randn_like(X)
        elif not isinstance(X_diff, th.Tensor):
            raise TypeError(f'X_diff must be torch.Tensor, but occurred {type(X_diff)}.')
        elif X_diff.ndim == 2:
            X_diff = X_diff.unsqueeze(0)
        elif X_diff.ndim != 3:
            raise ValueError(f'X_diff must be 2D or 3D, but got shape {X_diff.shape}.')
        if X_diff.shape != X.shape:
            raise ValueError(
                f'X_diff and X must have the same shape, but got '
                f'{X_diff.shape} and {X.shape}.'
            )
        if batch_indices is None:
            raise NotImplementedError('Krylov requires irregular batch_indices.')
        extra_krylov_dim = int(extra_krylov_dim)
        if extra_krylov_dim < 1:
            raise ValueError('At least one extra Krylov eigenvalue is required.')

        self._X_diff_init = X_diff
        self._extra_krylov_dim = extra_krylov_dim
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
        """Run the initial Krylov solve and validate its energy and gradient."""
        if batch_indices is None:
            raise NotImplementedError('Krylov requires irregular batch_indices.')
        if self._X_diff_init is None:
            raise RuntimeError('Krylov initial direction was not prepared.')
        eigen_order = self._morse_index + self._extra_krylov_dim
        if X.shape[1] <= eigen_order:
            raise ValueError(
                'The sum of the Morse index and extra Krylov dimensions is '
                f'not smaller than the total atom count ({X.shape[1]}).'
            )
        # BaseOpt has already normalized X to BT_FLOAT_TYPE. The initial
        # Lanczos direction follows that dtype so every finite difference uses
        # one numerical precision even when the caller supplied another dtype.
        v = self._X_diff_init.to(
            device=self.device,
            dtype=X.dtype,
        ).mul(atom_masks)
        energies, X_grad, self.eigenval, self.eigenvec = self.EigenFinder.run(
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
            eigen_order=eigen_order,
        )
        if energies.shape[0] != self.n_true_batch:
            raise ValueError(
                f'shape of output ({energies.shape}) does not match given batch indices'
            )
        if X_grad.shape != X.shape:
            raise RuntimeError(
                f'X_grad ({X_grad.shape}) and X ({X.shape}) have different shapes.'
            )
        self.is_concat_X = True
        energies = energies.detach()
        X_grad = X_grad.detach()
        X_grad.mul_(atom_masks)
        X = X.detach()
        return StdContainer(Energy=energies, X=X, Force=-X_grad, X_grad=X_grad)

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
        """Store model context for subsequent Krylov direction updates."""
        if batch_indices is None:
            raise NotImplementedError('Krylov requires irregular batch_indices.')
        # These references are immutable run context. Per-iteration active
        # arguments live on self.s and are replaced by BaseOpt's updater before
        # each direction hook; keeping both kinds of state separate avoids
        # silently using full-batch arguments for a reduced active batch.
        self._func = func
        self._grad_func = grad_func
        self.s.func_args = func_args
        self.s.func_kwargs = func_kwargs
        self.s.grad_func_args = grad_func_args
        self.s.grad_func_kwargs = grad_func_kwargs
        self.s.atom_masks = atom_masks
        self.s.batch_tensor = batch_indices
        self._is_grad_func_contain_y = is_grad_func_contain_y
        self._require_grad = require_grad
        # BaseOpt may report force/energy convergence only after the lowest
        # Ritz value confirms that each structure remains in a negative mode.
        self._extra_converge_mask = self.eigenval[:, 0] < 0.
        # The initial eigen solve already supplied the checked energy/gradient;
        # no line-search cache exists until the first in-place update.
        self._line_search.HAS_GRAD = False

    def _update_algo_param(
            self,
            select_mask: th.Tensor,
            select_mask_short: th.Tensor | None,
            batch_scatter_indices: th.Tensor | None,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            displace: th.Tensor | None,
    ) -> None:
        # BaseOpt passes atom- and structure-level masks separately. Select the
        # complete eigen state with the corresponding layout for this update:
        # eigenvalues are [structure, mode], whereas eigenvectors are
        # [mode, concatenated atom, Cartesian component]. Underscored members
        # are the active working set and non-underscored members remain the
        # stable full-batch owners until `_update_algo_batches()` commits them.
        if self._hold_samples:
            self._eigenval = self.eigenval
            self._eigenvec = self.eigenvec
        else:
            self._eigenval = self.eigenval[select_mask_short]
            self._eigenvec = self.eigenvec[:, select_mask, :]

    def _store_eigen_result(
            self,
            energies: th.Tensor,
            gradients: th.Tensor,
            eigenval: th.Tensor,
            eigenvec: th.Tensor,
    ) -> None:
        # Publish the new local eigen solve through the existing line-search
        # cache protocol, avoiding a duplicate model evaluation in BaseOpt.
        self._eigenval = eigenval
        self._eigenvec = eigenvec
        self._local_extra_converge_mask = eigenval[:, 0] < 0.
        self._line_search.HAS_GRAD = True
        self._line_search.STORE_Y = energies
        self._line_search.STORE_GRAD = gradients

    def _update_eigen_batches(
            self,
            select_indices: th.Tensor,
            select_indices_short: th.Tensor | None,
    ) -> None:
        # BaseOpt calls this only after the active in-place update has finished.
        # Atom-layout eigenvectors and structure-layout eigenvalues must use
        # their matching scatter indices to keep irregular batches aligned.
        if self._hold_samples:
            self.eigenval = self._eigenval
            self.eigenvec = self._eigenvec
            self._extra_converge_mask = self._local_extra_converge_mask
        else:
            self.eigenval.index_copy_(0, select_indices_short, self._eigenval)
            self.eigenvec.index_copy_(1, select_indices, self._eigenvec)
            self._extra_converge_mask.index_copy_(
                0, select_indices_short, self._local_extra_converge_mask,
            )


class KrylovNewton(_KrylovBase):
    """Krylov subspace Newton search with spectral modification."""

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
            output_file: str | None = None,
            verbose: int = 2,
            morse_index: int = 1,
            neg_spectra_cutoff: float = 0.01,
            pos_spectra_cutoff: float = 0.01,
    ) -> None:
        super().__init__(
            'KrylovNewton',
            E_threshold, Torque_thres, Eigen_thres, F_threshold,
            maxiter_trans, maxiter_eig, steplength, dx,
            output_file,
            device, verbose,
            morse_index, neg_spectra_cutoff, pos_spectra_cutoff,
        )
        self.steplength_sheme = steplength_sheme
        self._trust_reg_rad_max = 5. * self.steplength
        self._trust_reg_rad_min = 0.001 * self.steplength
        self.delta2: th.Tensor | None = None
        self._delta2: th.Tensor | None = None

    def initialize_algo_param(self) -> None:
        """Initialize the Newton trust-region radii."""
        self.delta2 = th.full(
            (self.n_true_batch, 1),
            self.steplength ** 2,
            device=self.device,
            dtype=self.s.X.dtype,
        )

    def _update_algo_param(
            self,
            select_mask: th.Tensor,
            select_mask_short: th.Tensor | None,
            batch_scatter_indices: th.Tensor | None,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            displace: th.Tensor | None,
    ) -> None:
        super()._update_algo_param(
            select_mask, select_mask_short, batch_scatter_indices,
            g, g_old, p, displace,
        )
        self._delta2 = (
            self.delta2 if self._hold_samples else self.delta2[select_mask_short]
        )

    def _diag_trust_region(
            self,
            vg: th.Tensor,
            dii: th.Tensor,
            g_comp_square: th.Tensor,
            delta2: th.Tensor,
            tol: float = 1e-4,
            max_iter: int = 50,
    ) -> th.Tensor:
        """Solve the diagonal trust-region equation in double precision."""
        # In the modified Ritz basis the step norm is monotone in the
        # non-negative multiplier mu. Samples whose unshifted Newton step is
        # already inside the radius are inactive and receive mu = 0 at return;
        # all samples remain in the batched scalar solve until then.
        # The secular equation is sensitive to cancellation near its root.
        # Keep this small scalar subproblem in float64 even when BT_FLOAT_TYPE
        # is float32, then return mu in the caller's dtype so the surrounding
        # optimizer retains its configured precision.
        output_dtype = delta2.dtype
        vg = vg.T.to(dtype=th.float64)
        dii = dii.to(dtype=th.float64)
        g_comp_square = g_comp_square.to(dtype=th.float64)
        delta2 = delta2.to(dtype=th.float64)
        f0 = ((vg / dii) ** 2).sum(dim=-1, keepdim=True) + g_comp_square - delta2
        active = f0 > 0.
        if not active.any():
            return th.zeros_like(f0, dtype=output_dtype)

        # The upper bound follows from the norm of the diagonally scaled
        # gradient. Golden-section updates are cheap here because the problem
        # has only one scalar unknown per structure.
        v_norm_sq = ((dii * vg) ** 2).sum(dim=-1, keepdim=True) + g_comp_square
        left = th.zeros_like(v_norm_sq) + 1e-4
        right = left + th.sqrt(v_norm_sq / delta2) + 1e-8
        right = th.maximum(right, left + 1e-4)
        inv_phi = (2.236067977 - 1.0) / 2.0
        x = right - inv_phi * (right - left)
        f_x = (
            ((dii * vg / (dii ** 2 + x)) ** 2).sum(dim=-1, keepdim=True)
            + g_comp_square / (1. + x) ** 2
            - delta2
        )
        is_converged = False
        for _ in range(max_iter):
            converged = (
                ((right - left) < tol * delta2)
                | (th.abs(f_x) < tol * delta2)
                | ~active
            )
            if converged.all():
                is_converged = True
                break
            left = th.where(f_x > 0, x, left)
            right = th.where(f_x < 0, x, right)
            x = right - inv_phi * (right - left)
            f_x = (
                ((dii * vg / (dii ** 2 + x)) ** 2).sum(dim=-1, keepdim=True)
                + g_comp_square / (1. + x) ** 2
                - delta2
            )
        if not is_converged:
            max_residual = th.abs(f_x[active]).max()
            self.logger.warning(
                'Golden-section trust-region not fully converged. '
                f'Max residual: {max_residual:.2e}'
            )
        mu = th.where(active, (left + right) * 0.5, 0.)
        return mu.to(dtype=output_dtype)

    def _curve_cond_linesearch(
            self,
            X: th.Tensor,
            direction: th.Tensor,
            gradient: th.Tensor,
            steplength: th.Tensor,
            n_local_batch: int,
            batch_scatter: th.Tensor,
            atom_masks: th.Tensor,
    ) -> th.Tensor:
        """Run the original curve-condition line search."""
        gradient_norm = th.sum(
            index_ops.index_inner_product(
                gradient, gradient, 1, batch_scatter, out_size=n_local_batch,
            ),
            dim=-1,
            keepdim=True,
        ).sqrt_()
        is_converged = False
        for _ in range(10):
            trial_X = th.addcmul(X, atom_masks * direction, steplength)
            _, trial_gradient = self._calc_y_grad(
                trial_X,
                self._func,
                self.s.func_args,
                self.s.func_kwargs,
                self._grad_func,
                self.s.grad_func_args,
                self.s.grad_func_kwargs,
                self._require_grad,
                self._is_grad_func_contain_y,
            )
            trial_gradient.mul_(atom_masks)
            trial_norm = th.sum(
                index_ops.index_inner_product(
                    trial_gradient,
                    trial_gradient,
                    1,
                    batch_scatter,
                    out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True,
            ).sqrt_()
            if th.all(trial_norm < 0.9 * gradient_norm):
                is_converged = True
                break
            steplength *= 0.618
        if not is_converged:
            self.logger.warning('Line search did not converge.')
        return steplength

    def _newton_steplength(
            self,
            X: th.Tensor,
            direction: th.Tensor,
            gradient: th.Tensor,
            n_local_batch: int,
            batch_scatter: th.Tensor,
            atom_masks: th.Tensor,
    ) -> th.Tensor:
        """Run the original directional Newton step estimate."""
        _, _, Hp = fin_diff_hvp(
            self._func,
            self.s.func_args,
            self.s.func_kwargs,
            self._grad_func,
            self.s.grad_func_args,
            self.s.grad_func_kwargs,
            X,
            direction,
            batch_scatter,
            is_g_contain_y=self._is_grad_func_contain_y,
            require_grad=self._require_grad,
            delta=self.dx,
        )
        Hp.mul_(atom_masks)
        gp = index_ops.index_inner_product(
            gradient, direction, 1, batch_scatter, out_size=n_local_batch,
        )
        pHp = index_ops.index_inner_product(
            direction, Hp, 1, batch_scatter, out_size=n_local_batch,
        )
        return (
            (-gp / pHp)
            .clamp_(0.01, self.steplength)
            .index_select(1, batch_scatter)
        )

    def _update_direction(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices: th.Tensor | None,
    ) -> None:
        if batch_scatter_indices is None:
            raise NotImplementedError('Krylov requires irregular batch_indices.')

        # All tensors below describe only the structures that BaseOpt selected
        # as active. The structure-level Ritz data and atom-level coordinates
        # are linked by batch_scatter throughout this complete in-place step.
        batch_scatter = batch_scatter_indices
        batch_tensor = self.s.batch_tensor
        atom_masks = self.s.atom_masks
        n_local_batch = self._eigenval.shape[0]
        morse_index = self._morse_index
        extra = self._extra_krylov_dim
        eigenval = self._eigenval
        eigenvec = self._eigenvec

        # `spectra_cut_off` is deliberately a cheap curvature scale, not a
        # rigorous bound on the omitted Hessian spectrum.  For a structure
        # with n atoms it evaluates
        #
        #   (m * |lambda_min| + q * lambda_max + (n - m - q) * 1) / n,
        #
        # where m is the Morse index and q is the number of explicitly kept
        # positive modes. Lanczos converges from both spectral edges, so the
        # lowest Ritz value is a useful scale for the m unstable modes and the
        # largest Ritz value above the Morse subspace is a cheap positive-edge
        # estimate for the q retained modes. The unresolved modes keep the
        # algorithm's unit-curvature model. Atom-count averaging, rather than a
        # Cartesian-degree average, intentionally preserves the legacy scale.
        # `amax` over all columns above the Morse subspace finds the resolved
        # positive edge even when unequal Krylov dimensions leave zero padding
        # at the right side of a batch row.
        max_pos_eigenval = eigenval[:, morse_index:].amax(dim=1)
        spectra_cut_off = (
            morse_index * eigenval[:, 0].abs()
            + extra * max_pos_eigenval
            + batch_tensor - morse_index - extra
        ) / batch_tensor
        spectra_cut_off.unsqueeze_(-1)
        eig_thres_neg = th.as_tensor(
            self.neg_spectra_cutoff, dtype=X.dtype, device=X.device,
        )
        eig_thres_pos = th.as_tensor(
            self.pos_spectra_cutoff, dtype=X.dtype, device=X.device,
        )
        # Enforce the requested saddle signature: unstable modes stay negative
        # and all retained complement modes stay positive. The upper negative
        # bound prevents one extreme mode from dominating the Newton solve.
        spectra = th.zeros_like(eigenval[:, :morse_index + extra])
        spectra[:, :morse_index] = eigenval[:, :morse_index].clamp(
            -spectra_cut_off, eig_thres_neg,
        )
        spectra[:, morse_index:] = eigenval[
            :, morse_index:morse_index + extra
        ].clamp(eig_thres_pos, None)

        # Split the gradient into the resolved Ritz subspace and its orthogonal
        # complement. The latter uses unit curvature because no Hessian
        # information is available there.
        eigenvec_cut = eigenvec[:morse_index + extra]
        projected_gradient = index_ops.index_reduce(
            th.sum(eigenvec_cut * g, dim=-1),
            batch_scatter,
            dim=1,
            out_size=n_local_batch,
        )
        complement = g - th.einsum(
            'kbd,kb->bd',
            eigenvec_cut,
            projected_gradient.index_select(1, batch_scatter),
        )
        complement_square = th.sum(
            index_ops.index_inner_product(
                complement,
                complement,
                1,
                batch_scatter,
                out_size=n_local_batch,
            ),
            dim=-1,
            keepdim=True,
        )

        if self.steplength_sheme == 'trust_region':
            # mu shifts both the resolved spectrum and the unit-curvature
            # complement so the combined step satisfies the current radius.
            mu = self._diag_trust_region(
                projected_gradient,
                spectra,
                complement_square[0],
                self._delta2,
            )
            spectra_inv = (
                spectra + mu * spectra.reciprocal()
            ).reciprocal_()
            complement_step = (
                (mu + 1.)
                .reciprocal_()
                .index_select(0, batch_scatter)
                .unsqueeze(0)
            )
        else:
            mu = None
            spectra_inv = spectra.reciprocal()
            complement_step = th.ones(
                (), dtype=X.dtype, device=X.device,
            )

        # Reconstruct the spectral Newton component in Cartesian coordinates,
        # then add the independently scaled unresolved complement.
        tangent = th.einsum(
            'kad,ak,ka->ad',
            eigenvec_cut,
            spectra_inv.index_select(0, batch_scatter),
            projected_gradient.index_select(1, batch_scatter),
        ).unsqueeze(0)
        direction = th.addcmul(
            tangent, complement, complement_step,
        ).neg_()
        direction_norm = th.sum(
            index_ops.index_inner_product(
                direction,
                direction,
                1,
                batch_scatter,
                out_size=n_local_batch,
            ),
            dim=-1,
            keepdim=True,
        ).sqrt_()
        has_negative_curvature = (
            eigenval[:, 0]
            .lt(0.)
            .index_select(0, batch_scatter)
            .reshape(1, -1, 1)
        )

        # The three experimental step policies share the same modified Newton
        # direction. Trust-region already scales the direction through mu and
        # therefore applies a scalar unit steplength below.
        if self.steplength_sheme == 'line_search':
            steplength = th.full(
                (1, X.shape[1], 1),
                self.steplength,
                dtype=X.dtype,
                device=X.device,
            )
            steplength = self._curve_cond_linesearch(
                X,
                direction,
                g,
                steplength,
                n_local_batch,
                batch_scatter,
                atom_masks,
            )
        elif self.steplength_sheme == 'line_newton':
            steplength = self._newton_steplength(
                X,
                direction,
                g,
                n_local_batch,
                batch_scatter,
                atom_masks,
            )
        elif self.steplength_sheme == 'trust_region':
            steplength = th.ones((), dtype=X.dtype, device=X.device)
            gradient_norm = th.sum(
                index_ops.index_inner_product(
                    g, g, 1, batch_scatter, out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True,
            ).sqrt_()
            residual_norm = (
                th.einsum(
                    'bk,bk,bk->b',
                    mu / (spectra ** 2 + mu),
                    projected_gradient.T,
                    projected_gradient.T,
                ).unsqueeze(-1)
                + (mu / (1. + mu)) ** 2 * complement_square[0]
            ).sqrt_().reshape(1, -1, 1)
            predicted_gradient_descent = gradient_norm - residual_norm
        else:
            raise NotImplementedError(
                f'Unknown steplength scheme: {self.steplength_sheme}.'
            )

        # Before a negative mode is located, perturb the direction to avoid
        # repeatedly following a purely positive-curvature Newton trajectory.
        # This decision is structure-local but is expanded through the atom
        # scatter so every coordinate of that structure follows one branch.
        direction = th.where(
            has_negative_curvature,
            direction,
            direction + 0.1 * th.randn_like(direction) * direction_norm.index_select(1, batch_scatter),
        )
        displacement = steplength * direction * atom_masks
        displacement_norm = th.sum(
            index_ops.index_inner_product(
                displacement,
                displacement,
                1,
                batch_scatter,
                out_size=n_local_batch,
            ),
            dim=-1,
            keepdim=True,
        ).sqrt_()
        # `step_tolerance` is an absolute floating-point floor for the applied
        # Cartesian displacement, not an energy/force convergence criterion.
        # For n atoms in d dimensions the displacement norm combines n*d
        # components; assuming independent roundoff at the coordinate scale,
        # its accumulated noise is approximately sqrt(n*d) * machine epsilon.
        # This system-size scaling avoids one hard-coded threshold for every
        # structure while remaining far below a physical optimizer tolerance.
        # It follows X.dtype and therefore also follows BT_FLOAT_TYPE. The test
        # is made after step selection and fixed-atom masking, so it measures
        # the coordinates that would actually be added to X.
        step_tolerance = (
            batch_tensor.to(dtype=X.dtype)
            .mul(X.shape[-1])
            .sqrt_()
            .mul_(th.finfo(X.dtype).eps)
            .reshape(1, -1, 1)
        )
        small_step = displacement_norm <= step_tolerance
        if th.all(small_step):
            raise IterationStuckError(
                'All active iterations are stuck because their coordinate '
                'displacements are below the numerical threshold.'
            )
        elif th.any(small_step):
            warnings.warn(
                RuntimeWarning(
                    'Convergence is not met while some coordinate '
                    'displacements are below the numerical threshold.'
                )
            )
        # This is the coordinate mutation declared by `_is_inplace_update`.
        # It occurs only after the stuck check, leaving X and its cached model
        # state mutually consistent when IterationStuckError exits the loop.
        # BaseOpt has also protected any aliased output buffers before entry.
        X.add_(displacement)

        # Reuse the retained low-mode subspace as the next Lanczos seed. The
        # returned midpoint energy/gradient belong to the translated X and are
        # handed back to BaseOpt through `_store_eigen_result`.
        next_v = th.mean(eigenvec_cut, dim=0, keepdim=True)
        energies, gradients, eigenval, eigenvec = self.EigenFinder.run(
            func=self._func,
            X=X,
            v=next_v,
            grad_func=self._grad_func,
            func_args=self.s.func_args,
            func_kwargs=self.s.func_kwargs,
            grad_func_args=self.s.grad_func_args,
            grad_func_kwargs=self.s.grad_func_kwargs,
            is_grad_func_contain_y=self._is_grad_func_contain_y,
            require_grad=self._require_grad,
            fixed_atom_tensor=atom_masks,
            batch_indices=batch_tensor,
            eigen_order=morse_index + extra,
        )

        if self.steplength_sheme == 'trust_region':
            # Compare predicted and observed gradient-norm reduction, then
            # update each structure's radius for the next outer iteration.
            # The radius is local working state here; it is not published to
            # the full batch until the energy, gradient, and Ritz state are all
            # ready to be committed together below.
            new_gradient_norm = th.sum(
                index_ops.index_inner_product(
                    gradients,
                    gradients,
                    1,
                    batch_scatter,
                    out_size=n_local_batch,
                ),
                dim=-1,
                keepdim=True,
            ).sqrt_()
            rho = th.where(
                predicted_gradient_descent < 0.,
                0.,
                (
                    gradient_norm - new_gradient_norm
                ) / predicted_gradient_descent,
            )
            radius = self._delta2.sqrt()
            radius = th.where(
                rho >= 0.75,
                (radius * 1.1).clamp_max_(self._trust_reg_rad_max),
                th.where(
                    rho >= 0.25,
                    radius,
                    (radius * 0.5).clamp_min_(self._trust_reg_rad_min),
                ),
            )
            self._delta2 = radius.square_().reshape(-1, 1)

        self._store_eigen_result(
            energies, gradients, eigenval, eigenvec,
        )
        return None

    def _update_algo_batches(
            self,
            select_indices: th.Tensor,
            select_indices_short: th.Tensor | None,
    ) -> None:
        # Commit the active eigen solve and its next trust radius together so
        # the following BaseOpt iteration observes one consistent state. The
        # dynamic-batch path scatters only active structures; removed samples
        # retain the last state at which BaseOpt declared them converged.
        self._update_eigen_batches(select_indices, select_indices_short)
        if self._hold_samples:
            self.delta2 = self._delta2
        else:
            self.delta2.index_copy_(
                0, select_indices_short, self._delta2,
            )


class KrylovDynamics(_KrylovBase):
    """Krylov dynamics transition-state search."""

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
    ) -> None:
        super().__init__(
            'KrylovDynamics',
            E_threshold, Torque_thres, Eigen_thres, F_threshold,
            maxiter_trans, maxiter_eig, steplength, dx, device, verbose,
            morse_index, neg_spectra_cutoff, pos_spectra_cutoff,
        )
        self.steplength_sheme = steplength_sheme
        self.max_steplength = float(steplength) * 5.
        self.alpha = float(alpha)
        self.alpha_fac = float(alpha_fac)
        self.fac_inc = float(fac_inc)
        self.fac_dec = float(fac_dec)
        self.N_min = int(N_min)
        self.t_init = float(steplength)

        self.t: th.Tensor | None = None
        self.a: th.Tensor | None = None
        self.n_count: th.Tensor | None = None
        self.veloc: th.Tensor | None = None
        self._t: th.Tensor | None = None
        self._a: th.Tensor | None = None
        self._n_count: th.Tensor | None = None
        self._veloc: th.Tensor | None = None

    def initialize_algo_param(self) -> None:
        """Initialize FIRE-like translation state."""
        X = self.s.X
        # FIRE's t, a, and positive-power count are logically structure-level,
        # but are stored in atom layout so they can share BaseOpt's existing
        # active-atom selection/scatter protocol with velocity. Every atom of a
        # structure receives the same value and follows the same FIRE branch.
        self.t = th.full(
            (1, X.shape[1], 1),
            self.t_init,
            device=self.device,
            dtype=X.dtype,
        )
        self.a = th.full_like(self.t, self.alpha)
        self.n_count = th.zeros_like(self.t, dtype=th.int)
        self.veloc = th.zeros_like(X)

    def _update_algo_param(
            self,
            select_mask: th.Tensor,
            select_mask_short: th.Tensor | None,
            batch_scatter_indices: th.Tensor | None,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            displace: th.Tensor | None,
    ) -> None:
        super()._update_algo_param(
            select_mask, select_mask_short, batch_scatter_indices,
            g, g_old, p, displace,
        )
        # Underscored tensors are the active-batch working state. Hold mode may
        # alias the full tensors because BaseOpt will not scatter them later.
        if self._hold_samples:
            self._t = self.t
            self._a = self.a
            self._n_count = self.n_count
            self._veloc = self.veloc
        else:
            self._t = self.t[:, select_mask, :]
            self._a = self.a[:, select_mask, :]
            self._n_count = self.n_count[:, select_mask, :]
            self._veloc = self.veloc[:, select_mask, :]

    def _update_direction(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices: th.Tensor | None,
    ) -> None:
        if batch_scatter_indices is None:
            raise NotImplementedError('Krylov requires irregular batch_indices.')

        # As in KrylovNewton, this hook owns the complete coordinate update and
        # publishes the following eigen solve through BaseOpt's cache protocol.
        batch_scatter = batch_scatter_indices
        batch_tensor = self.s.batch_tensor
        atom_masks = self.s.atom_masks
        n_local_batch = self._eigenval.shape[0]
        morse_index = self._morse_index
        extra = self._extra_krylov_dim
        eigenval = self._eigenval
        eigenvec = self._eigenvec

        # Use the same deliberately coarse cutoff as KrylovNewton:
        #
        #   (m * |lambda_min| + q * lambda_max + (n - m - q) * 1) / n.
        #
        # The two Ritz edge values represent the retained unstable and positive
        # modes, while omitted modes retain unit curvature. This is only a
        # low-cost per-structure scale for clipping the modified spectrum, not
        # a claim that the interior spectrum has been reconstructed. Lanczos'
        # two-sided edge convergence makes `amax` above the Morse subspace a
        # reasonable positive-edge estimate, and it cannot be displaced by the
        # trailing zero padding used for unequal effective Krylov dimensions.
        max_pos_eigenval = eigenval[:, morse_index:].amax(dim=1)
        spectra_cut_off = (
            morse_index * eigenval[:, 0].abs()
            + extra * max_pos_eigenval
            + batch_tensor - morse_index - extra
        ) / batch_tensor
        spectra_cut_off.unsqueeze_(-1)
        eig_thres_neg = th.as_tensor(
            self.neg_spectra_cutoff, dtype=X.dtype, device=X.device,
        )
        eig_thres_pos = th.as_tensor(
            self.pos_spectra_cutoff, dtype=X.dtype, device=X.device,
        )
        spectra = th.zeros_like(eigenval[:, :morse_index + extra])
        spectra[:, :morse_index] = eigenval[:, :morse_index].clamp(
            -spectra_cut_off, eig_thres_neg,
        )
        spectra[:, morse_index:] = eigenval[
            :, morse_index:morse_index + extra
        ].clamp(eig_thres_pos, spectra_cut_off)
        spectra_inv = spectra.reciprocal()

        # Apply the inverse modified spectrum only in the resolved Ritz
        # subspace. The unresolved orthogonal component keeps unit response.
        eigenvec_cut = eigenvec[:morse_index + extra]
        projected_gradient = index_ops.index_reduce(
            th.sum(eigenvec_cut * g, dim=-1),
            batch_scatter,
            dim=1,
            out_size=n_local_batch,
        )
        complement = g - th.einsum(
            'kbd,kb->bd',
            eigenvec_cut,
            projected_gradient.index_select(1, batch_scatter),
        )
        tangent = th.einsum(
            'kad,ak,ka->ad',
            eigenvec_cut,
            spectra_inv.index_select(0, batch_scatter),
            projected_gradient.index_select(1, batch_scatter),
        ).unsqueeze(0)
        effective_force = th.add(
            tangent, complement,
        ).neg_().mul(atom_masks)
        force_norm = th.sum(
            index_ops.index_inner_product(
                effective_force,
                effective_force,
                1,
                batch_scatter,
                out_size=n_local_batch,
            ),
            dim=-1,
            keepdim=True,
        ).sqrt_()
        # Zero effective force is a valid stationary state; clamping only the
        # divisor leaves its normalized direction exactly zero.
        force_norm.clamp_min_(th.finfo(force_norm.dtype).eps)

        force_hat = (
            effective_force
            / force_norm.index_select(1, batch_scatter)
        )
        # FIRE power is evaluated per structure and then expanded to atoms so
        # all atoms in one structure share the same accept/reset decision.
        power = index_ops.index_reduce(
            th.sum(force_hat * self._veloc, dim=-1, keepdim=True),
            batch_scatter,
            dim=1,
            out_size=n_local_batch,
        ).index_select(1, batch_scatter)
        velocity_norm = th.sum(
            index_ops.index_inner_product(
                self._veloc,
                self._veloc,
                dim=1,
                batch_indices=batch_scatter,
                out_size=n_local_batch,
            ),
            dim=-1,
            keepdim=True,
        ).sqrt_().index_select(1, batch_scatter)

        # FIRE control flow is intentionally structure-local:
        #   1. mix velocity toward the normalized effective force;
        #   2. update the consecutive-positive-power counter;
        #   3. after N_min positive steps, increase dt and reduce mixing a;
        #   4. on nonpositive power, reduce dt and reset velocity/a/counter;
        #   5. integrate velocity and then coordinates with the final state.
        # Structure decisions are already expanded to atom layout, so no atom
        # within one structure can enter a different FIRE branch.
        force_hat.mul_(velocity_norm)
        self._veloc.mul_(1. - self._a)
        self._veloc.addcmul_(self._a, force_hat)
        self._n_count += th.where(power > 0., 1, -self._n_count)
        enough_positive_steps = self._n_count >= self.N_min
        new_t = (
            self._t * self.fac_inc
        ).clamp_max_(self.max_steplength)
        self._t = th.where(
            enough_positive_steps, new_t, self._t,
        )
        self._a = th.where(
            enough_positive_steps,
            self._a * self.alpha_fac,
            self._a,
        )

        nonpositive_power = power <= 0.
        self._t = th.where(
            nonpositive_power,
            self._t * self.fac_dec,
            self._t,
        )
        self._veloc.masked_fill_(nonpositive_power, 0.)
        self._a.masked_fill_(nonpositive_power, self.alpha)
        # FIRE integration order is v <- v + c F dt, then X <- X + v dt;
        # consequently a force-only first displacement scales as dt squared.
        self._veloc.addcmul_(
            effective_force,
            self._t,
            value=9.64853329045427e-3,
        )
        X.addcmul_(self._veloc, self._t)

        # Refresh the low modes and cache the translated energy/gradient for
        # BaseOpt instead of evaluating the model a second time.
        next_v = th.mean(eigenvec_cut, dim=0, keepdim=True)
        energies, gradients, eigenval, eigenvec = self.EigenFinder.run(
            func=self._func,
            X=X,
            v=next_v,
            grad_func=self._grad_func,
            func_args=self.s.func_args,
            func_kwargs=self.s.func_kwargs,
            grad_func_args=self.s.grad_func_args,
            grad_func_kwargs=self.s.grad_func_kwargs,
            is_grad_func_contain_y=self._is_grad_func_contain_y,
            require_grad=self._require_grad,
            fixed_atom_tensor=atom_masks,
            batch_indices=batch_tensor,
            eigen_order=morse_index + extra,
        )
        self._store_eigen_result(
            energies, gradients, eigenval, eigenvec,
        )
        return None

    def _update_algo_batches(
            self,
            select_indices: th.Tensor,
            select_indices_short: th.Tensor | None,
    ) -> None:
        # Scatter active eigen and FIRE state back to their full-batch owners;
        # hold mode can directly replace those owners with the working tensors.
        # This method is the sole publication point for underscored working
        # state, keeping the next outer iteration independent of whether the
        # current batch was held intact or dynamically reduced.
        self._update_eigen_batches(
            select_indices, select_indices_short,
        )
        if self._hold_samples:
            self.t = self._t
            self.a = self._a
            self.n_count = self._n_count
            self.veloc = self._veloc
        else:
            self.t.index_copy_(1, select_indices, self._t)
            self.a.index_copy_(1, select_indices, self._a)
            self.n_count.index_copy_(1, select_indices, self._n_count)
            self.veloc.index_copy_(1, select_indices, self._veloc)
