"""
Quasi-Newton Algorithm for optimization.
"""
#  Copyright (c) 2024.12.10, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.7b
#  File: QN.py
#  Environment: Python 3.12

from typing import Literal

from BM4Ckit.BatchOptim._utils._BaseOpt import _BaseOpt
import torch as th


class QN(_BaseOpt):
    """
    Quasi-Newton Algo.
    """
    def __init__(
            self,
            iter_scheme: Literal['BFGS', 'Newton'],
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            linesearch: Literal['Backtrack', 'Wolfe', 'NWolfe', '2PT', '3PT', 'Golden', 'Newton', 'None'] = 'Backtrack',
            linesearch_maxiter: int = 10,
            linesearch_thres: float = 0.02,
            linesearch_factor: float = 0.6,
            steplength: float = 0.5,
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
            device,
            verbose
        )

    def initialize_algo_param(self):
        """
        Override this method to initialize attribute variables for self._update_direction.

        Returns: None
        """
        # Initial quasi-inverse Hessian Matrix  (n_batch, n_atom*n_dim, n_atom*n_dim)
        self.H_inv = (th.eye(self.n_atom * self.n_dim, device=self.device).unsqueeze(0)).expand(self.n_batch, -1, -1)
        # prepared identity matrix
        self.Ident = (th.eye(self.n_atom * self.n_dim, device=self.device).unsqueeze(0)).expand(self.n_batch, -1, -1)

    def _update_direction(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor) -> th.Tensor:
        """
        Override this method to implement X update algorithm.
        Args:
            g: (n_batch, n_atom*3, 1), the gradient of X at this step
            g_old: (n_batch, n_atom*3, 1), the gradient of X at last step
            p: (n_batch, n_atom*3, 1), the update direction of X at last step

        Returns:
            p: th.Tensor, the new update direction of X.
        """
        # QN scheme
        if self.iterform == 'BFGS':
            p = - self.H_inv @ g  # (n_batch, n_atom*3, n_atom*3) @ (n_batch, n_atom*3, 1)
            return p

        elif self.iterform == 'Newton':
            # Hessian
            H = th.zeros((self.n_batch, self.n_atom * self.n_dim, self.n_atom * self.n_dim), device=self.device)
            hess_mask = th.zeros(self.n_batch, self.n_atom * self.n_dim, 1, device=self.device)
            for i in range(self.n_atom * self.n_dim):
                hess_mask[:, i] = 1.
                H_line = th.where(
                    self.converge_mask[:, :1, :1],
                    hess_mask,
                    th.autograd.grad(g, X, hess_mask, retain_graph=True)[0]
                )  # (n_batch, n_atom*n_dim, 1)
                H[:, :, i] = H_line.squeeze(-1)
                hess_mask[:, i] = 0.
            del H_line
            H = H.detach()
            g = g.detach()
            p = - th.where(self.converge_mask[:, :1, :1], 0., th.linalg.solve(H, g))
            return p

        else:
            raise NotImplementedError

    def _update_algo_param(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, displace: th.Tensor) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.

        Returns: None
        """
        # update H_inv
        g_go = g - g_old
        gamma = 1 / (displace.mT @ g_go + 1e-20)  # (n_batch, 1, n_atom*3) @ (n_batch, n_atom*3, 1) -> (n_batch, 1, 1), 1e-20 to avoid 1/0
        # BFGS Scheme:
        # (n_batch, n_atom*n_dim, n_atom*n_dim) - (n_batch, 1, 1) * (n_batch, n_atom*n_dim, 1)@(n_batch, 1, n_atom*n_dim)
        self.H_inv = th.where(
            self.converge_mask[:, :1, :1],
            0.,
            (self.Ident - gamma * displace @ g_go.mT) @ self.H_inv @ (self.Ident - gamma * g_go @ displace.mT) + gamma * displace @ displace.mT
        )

