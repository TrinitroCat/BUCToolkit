"""
Quasi-Newton Algorithm for optimization.
"""
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: QN.py
#  Environment: Python 3.12

from typing import Literal
import os

import torch as th

from BUCToolkit.BatchOptim._BaseOpt import _BaseOpt
from BUCToolkit.utils.index_ops import indices_pairwise_dist, index_inner_product

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class QN(_BaseOpt):
    """
    Quasi-Newton Algo.

    Args:
        iter_scheme: Literal['BFGS', 'Newton'],
        E_threshold: float = 1e-3,
        F_threshold: float = 0.05,
        maxiter: int = 100,
        linesearch: Literal['Backtrack', 'Wolfe', 'NWolfe', '2PT', '3PT', 'Golden', 'Newton', 'None'] = 'Backtrack',
        linesearch_maxiter: int = 10,
        linesearch_thres: float = 0.02,
        linesearch_factor: float = 0.6,
        steplength: float = 0.5,
        use_bb: whether to use Barzilai-Borwein steplength (BB1 or long BB) as initial steplength instead of fixed one.
        external_Hessian: manually input Hessian matrix as initial guess.
        device: str | th.device = 'cpu',
        verbose: int = 2

    Methods:
        run: launch the optimization.
        set_update_batch: setting the method to update the taget function when variables change.
    """
    def __init__(
            self,
            iter_scheme: Literal['BFGS', ],
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            linesearch: Literal['Backtrack', 'B', 'Wolfe', 'W', 'MT', 'EXACT', 'None', 'N'] = 'Backtrack',
            linesearch_maxiter: int = 10,
            linesearch_thres: float = 0.02,
            linesearch_factor: float = 0.6,
            steplength: float = 0.5,
            use_bb: bool = True,
            external_Hessian: th.Tensor|None = None,
            output_file: str | None = None,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        super().__init__(
            iter_scheme,
            E_threshold,
            F_threshold,
            maxiter,
            linesearch,
            linesearch_maxiter,
            linesearch_thres,
            linesearch_factor,
            steplength,
            use_bb,
            output_file,
            device,
            verbose
        )
        self.external_Hessian = external_Hessian

    def initialize_algo_param(self):
        """
        Override this method to initialize attribute variables for self._update_direction.

        Returns: None
        """
        if self.external_Hessian is None:
            # Initial quasi-inverse Hessian Matrix  (n_batch, n_atom*n_dim, n_atom*n_dim)
            self.H_inv = (th.eye(self.n_atom * self.n_dim, device=self.device).unsqueeze(0)).expand(self.n_batch, -1, -1)
            self.H_inv = self.H_inv.reshape(self.n_batch, self.n_atom, self.n_dim, self.n_atom, self.n_dim).contiguous()
        elif self.external_Hessian.shape == (self.n_batch, self.n_atom * self.n_dim, self.n_atom * self.n_dim):
            # check positive definite
            eigval, eigvec = th.linalg.eigh(self.external_Hessian)
            if th.any(eigval < 0.):
                self.logger.warning('Some external_Hessian are not positive definite. Their negative eigenvalues would be replaced by 1.')
                eigval = th.where(eigval < 0., 1., eigval)
                _external_Hessian = eigvec@th.diag_embed(eigval, 0, -2, -1)@eigvec.mH
            else:
                _external_Hessian = self.external_Hessian

            self.H_inv = th.linalg.inv(_external_Hessian).reshape(self.n_batch, self.n_atom, self.n_dim, self.n_atom, self.n_dim).contiguous()
        else:
            self.logger.warning(f'Expected external Hessian has a shape of {(self.n_batch, self.n_atom * self.n_dim, self.n_atom * self.n_dim)}, '
                                f'but got {self.external_Hessian.shape}. Thus identity matrix would use instead.')
            self.H_inv = (th.eye(self.n_atom * self.n_dim, device=self.device).unsqueeze(0)).expand(self.n_batch, -1, -1)
            self.H_inv = self.H_inv.reshape(self.n_batch, self.n_atom, self.n_dim, self.n_atom, self.n_dim).contiguous()
        # prepared identity matrix
        self.Ident = (th.eye(self.n_atom * self.n_dim, device=self.device).unsqueeze(0)).expand(self.n_batch, -1, -1)
        self.Ident = self.Ident.reshape(self.n_batch, self.n_atom, self.n_dim, self.n_atom, self.n_dim).contiguous()

    def _update_direction(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices: th.Tensor | None,
    ) -> th.Tensor:
        """
        Override this method to implement X update algorithm.
        Args:
            g: (n_batch, n_atom, 3), the gradient of X at this step
            g_old: (n_batch, n_atom, 3), the gradient of X at last step
            p: (n_batch, n_atom, 3), the update direction of X at last step

        Returns:
            p: th.Tensor, the new update direction of X.
        """
        _shape = g.shape
        if self.is_concat_X:
            H_inv_now = self.H_inv[:, self.select_mask][..., self.select_mask, :]
        else:
            H_inv_now = self.H_inv[self.select_mask]
        # QN scheme
        if self.iterform == 'BFGS':
            p = - th.einsum('ijklm, ilm -> ijk', H_inv_now, g)  # (n_batch, n_atom*3, n_atom*3) @ (n_batch, n_atom*3, 1)
            return p

        else:
            raise NotImplementedError

    @staticmethod
    def _index_outer_prod(u: th.Tensor, v: th.Tensor, big_batch_scatter: th.Tensor):
        """

        Args:
            u: (1, sumN, 3)
            v: (1, sumN, 3)
            big_batch_scatter:  (1, sumN * 3, 1)

        Returns:

        """
        # skip check shape for this inner methods
        _, sumN, n_dim = u.shape
        u = u.reshape(sumN * n_dim, 1)
        v = v.reshape(sumN * n_dim, 1)
        coo, nzv, eid = indices_pairwise_dist(
            u,
            v,
            big_batch_scatter,
            big_batch_scatter,
            None,
            "dot",
            None,
            "gt",
            False,
            False,
            True,
            True,
        )
        # I know that coo are square matrix and the last one is the N-1
        size = (coo[0, -1] + 1, coo[0, -1] + 1)
        outerv = th.sparse_coo_tensor(
            coo,
            nzv,
            size,
            device=u.device,
            is_coalesced=True
        )
        return outerv

    def _update_algo_param(
            self,
            select_mask: th.Tensor,
            select_mask_short,
            batch_scatter_indices,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            displace: th.Tensor
    ) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.

        Returns: None
        """
        g_go = g - g_old
        if self.is_concat_X:
            # whence H has shape of (1, sumN, n_dim, sumN, n_dim),
            # selected to (1, N_unconverged, n_dim, N_unconverged, n_dim)
            H_inv_now = self.H_inv[:, select_mask][..., select_mask, :]
            Ident_now = self.Ident[:, select_mask][..., select_mask, :]
            Ident_now_ = Ident_now.flatten(-2, -1).flatten(-3, -2)  # (1, sumN*D, sumN*D)
            H_inv_now_ = H_inv_now.flatten(-2, -1).flatten(-3, -2)  # (1, sumN*D, sumN*D)
            big_batch_scatter = th.repeat_interleave(batch_scatter_indices, self.n_dim)  # for flatten tensors
            # A diagonal approximation of Powell damping.
            diag_invH = th.diagonal(H_inv_now_, dim1=-2, dim2=-1).reciprocal_().reshape(1, -1, self.n_dim)  # (1, sumN, D)
            ss = th.sum(index_inner_product(displace, diag_invH * displace, 1, batch_scatter_indices), dim=-1, keepdim=True)  # (1, B, 1)
            sy = th.sum(index_inner_product(displace, g_go, 1, batch_scatter_indices), dim=-1, keepdim=True)
            non_pos_deter_mask = (sy < 0.2 * ss)  # (1, B, 1)
            if th.any(non_pos_deter_mask):
                phi = (0.2 * ss - sy)/ss
                phi.clamp_min_(0.)
                g_go.addcmul_(phi.index_select(1, batch_scatter_indices), displace)
                sy = th.sum(index_inner_product(displace, g_go, 1, batch_scatter_indices), dim=-1, keepdim=True)  # (1, B, 1)
            # main update
            gamma = (1 / (sy + 1e-20)).index_select(1, big_batch_scatter)  # (1, B*D, 1)
            _dgo = self._index_outer_prod(displace, g_go, big_batch_scatter).to_dense()  # (B*D, B*D) in sparse COO
            # TODO: Now applied dense metrix H_inv. Could be fully converted to sparse coo in future and use spmm. handling shape is complex.
            #   now the dense scheme cost 1 GB memory for 64 samples with 85 atoms (float32) in a batch.
            #   Tolerable, compared with common deep learning potential models.
            I_gamma_dgo = th.addcmul(Ident_now_, gamma, _dgo, value=-1.)[0]
            I_gamma_god = th.addcmul(Ident_now_, gamma, _dgo.mT, value=-1.)[0]
            gdd = gamma * self._index_outer_prod(displace, displace, big_batch_scatter).to_dense()
            th.addmm(gdd[0], I_gamma_dgo, H_inv_now_[0] @ I_gamma_god, out=H_inv_now_[0])
            H_inv_now = H_inv_now_.reshape(H_inv_now.shape)
        else:
            H_inv_now = self.H_inv[select_mask]  # simply: (N_unconverged, n_atom, n_dim, n_atom, n_dim)
            Ident_now = self.Ident[select_mask]
            H_inv_now_ = H_inv_now.flatten(-2, -1).flatten(-3, -2)  # (B, A*D, A*D)
            diag_invH = th.diagonal(H_inv_now_, dim1=-2, dim2=-1).reciprocal_().reshape(-1, self.n_atom, self.n_dim)  # (B, A*D)
            ss = th.sum(displace * diag_invH * displace, dim=(-2, -1), keepdim=True)  # (B, 1, 1)
            sy = th.sum(displace * g_go, dim=(-2, -1), keepdim=True)
            non_pos_deter_mask = (sy < 0.2 * ss)  # (B, 1, 1)
            if th.any(non_pos_deter_mask):
                phi = (0.2 * ss - sy) / ss
                phi.clamp_min_(0.)
                g_go.addcmul_(phi, displace)
                sy = th.sum(displace * g_go, dim=(-2, -1), keepdim=True)  # (B, 1, 1)
            gamma = 1 / (sy + 1e-20)  # (n_batch, n_atom, 3) * (n_batch, n_atom, 3) -sum-> (n_batch, 1, 1), 1e-20 to avoid 1/0
            gamma = gamma.reshape(-1, 1, 1, 1, 1)
            # BFGS Scheme:
            # ((n_batch, n_atom*n_dim, n_atom*n_dim) - (n_batch, 1, 1) * (n_batch, n_atom*n_dim, 1)@(n_batch, 1, n_atom*n_dim)) -> (B, AD, AD)
            # (B, AD, AD) @ (B, AD, AD)
            # ((B, A, D, A, D) - (B, 1, 1, 1, 1) * (B, A, D, A, D)) @ (B, A, D, A, D) -(-2, -1)-> (B, A, D)
            H_inv_now = th.einsum(
                'mijkl, mklqr -> mijqr',
                th.einsum(
                    'mijkl, mklqr -> mijqr',
                    (Ident_now - gamma * th.einsum('ijk, ilm-> ijklm', displace, g_go)),
                    H_inv_now
                ),
                (Ident_now - gamma * th.einsum('ijk, ilm-> ijklm', g_go, displace))
            ) + gamma * th.einsum('ijk, ilm-> ijklm', displace, displace)

        if self.is_concat_X:
            tmp = self.H_inv[:, select_mask]
            tmp[..., select_mask, :] = H_inv_now
            self.H_inv[:, select_mask] = tmp
        else:
            self.H_inv[select_mask] = H_inv_now

        pass


