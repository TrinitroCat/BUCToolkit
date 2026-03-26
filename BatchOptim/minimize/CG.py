"""
Conjugate Gradient Algorithm for optimization.
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: CG.py
#  Environment: Python 3.12

from typing import Literal
import os

import torch as th

from BM4Ckit.BatchOptim._BaseOpt import _BaseOpt
from BM4Ckit.utils.index_ops import index_inner_product

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class CG(_BaseOpt):
    """
    Conjugate Gradient Algo.
    Args:
        iter_scheme: Literal['PR+', 'FR', 'SD'],
        E_threshold: float = 1e-3,
        F_threshold: float = 0.05,
        maxiter: int = 100,
        linesearch: Literal['Backtrack', 'Wolfe', 'NWolfe', '2PT', '3PT', 'Golden', 'Newton', 'None'] = 'Backtrack',
        linesearch_maxiter: int = 10,
        linesearch_thres: float = 0.02,
        linesearch_factor: float = 0.6,
        steplength: float = 0.5,
        use_bb: whether to use Barzilai-Borwein steplength (BB1 or long BB) as initial steplength instead of fixed one.
        device: str | th.device = 'cpu',
        verbose: int = 2

    Methods:
        run: launch the optimization.
        set_update_batch: setting the method to update the taget function when variables change.
    """
    def __init__(
            self,
            iter_scheme: Literal['PR+', 'FR', 'SD'],
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            linesearch: Literal['Backtrack', 'B', 'Wolfe', 'W', 'MT', 'EXACT', 'None', 'N'] = 'Backtrack',
            linesearch_maxiter: int = 10,
            linesearch_thres: float = 0.02,
            linesearch_factor: float = 0.6,
            steplength: float = 0.5,
            use_bb: bool = True,
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
            device,
            verbose
        )
        self.__update_scheme = None
        self.MIN_CACHE = th.scalar_tensor(1.e-20, device=self.device)

    @staticmethod
    def __check_restart(gg, ggo, g, p, beta):
        """
        check if restarting is needed.

        """
        ortho_check = th.abs(ggo) > 0.1 * gg
        return ortho_check

    def __PRP_irreg(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices: th.Tensor | None,
    ) -> th.Tensor:
        """
        POLAK-RIBIERE CG ALGORITHM for irregular batch.
        """
        gogo = th.sum(index_inner_product(g_old, g_old, 1, batch_scatter_indices), dim=-1, keepdim=True).clamp_min_(self.MIN_CACHE)  # (1, B, 1)
        ggo  = th.sum(index_inner_product(g, g_old, 1, batch_scatter_indices), dim=-1, keepdim=True)
        gg   = th.sum(index_inner_product(g, g, 1, batch_scatter_indices), dim=-1, keepdim=True)
        beta = (gg - ggo) / gogo  # (1, B, 1)
        beta.clamp_min_(self.MIN_CACHE)
        # Restart
        is_restart = self.__check_restart(gg, ggo, g, p, beta)
        beta.masked_fill_(is_restart, 0.)
        beta = beta.index_select(1, batch_scatter_indices)
        if self.verbose > 1:
            self.logger.info(f" Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        #p = -g + beta * p  # (n_batch, n_atom, n_dim)
        th.addcmul(-g, beta, p, out=p)

        return p

    def __PRP_reg(
            self ,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_
    ) -> th.Tensor:
        """
        POLAK-RIBIERE CG ALGORITHM for regular batch.
        """
        gogo = th.sum(g_old * g_old, dim=(-2, -1), keepdim=True).clamp_min_(self.MIN_CACHE) # to avoid divide 0
        ggo = th.sum(g * g_old, dim=(-2, -1), keepdim=True)  # (n_batch, 1, 1)
        gg = th.sum(g * g, dim=(-2, -1), keepdim=True)  # (n_batch, 1, 1)
        beta = (gg - ggo) / gogo  # (1, B, 1)
        beta.clamp_min_(self.MIN_CACHE)
        # Restart
        is_restart = self.__check_restart(gg, ggo, g, p, beta)
        beta.masked_fill_(is_restart, 0.)
        if self.verbose > 1:
            self.logger.info(f" Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        #p = -g + beta * p  # (n_batch, n_atom, n_dim)
        th.addcmul(-g, beta, p, out=p)

        return p

    def __FR_irreg(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_indices
    ) -> th.Tensor:
        """ FLETCHER-REEVES ALGO. for irregular batch. """
        gogo = th.sum(index_inner_product(g_old, g_old, 1, batch_scatter_indices), dim=-1, keepdim=True).clamp_min_(self.MIN_CACHE)  # (1, B, 1)
        ggo = th.sum(index_inner_product(g, g_old, 1, batch_scatter_indices), dim=-1, keepdim=True)
        gg = th.sum(index_inner_product(g, g, 1, batch_scatter_indices), dim=-1, keepdim=True)
        beta = gg / gogo
        # Restart
        is_restart = self.__check_restart(gg, ggo, g, p, beta)
        beta.masked_fill_(is_restart, 0.)  # (1, B, 1)
        beta = beta.index_select(1, batch_scatter_indices)  # (1, sumN, 1)
        if self.verbose > 1:
            self.logger.info(f" Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        th.addcmul(-g, beta, p, out=p)

        return p

    def __FR_reg(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_
    ) -> th.Tensor:
        """ FLETCHER-REEVES ALGO. for regular batch. """
        gogo = th.sum(g_old * g_old, dim=(-2, -1), keepdim=True).clamp_min_(self.MIN_CACHE)  # to avoid divide 0
        ggo = th.sum(g * g_old, dim=(-2, -1), keepdim=True)  # (n_batch, 1, 1)
        gg = th.sum(g * g, dim=(-2, -1), keepdim=True)  # (n_batch, 1, 1)
        beta = gg / gogo
        # Restart
        is_restart = self.__check_restart(gg, ggo, g, p, beta)
        beta.masked_fill_(is_restart, 0.)
        if self.verbose > 1:
            self.logger.info(f" Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        th.addcmul(-g, beta, p, out=p)  # (n_batch, n_atom, n_dim)

        return p

    @staticmethod
    def __SD(g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor, batch_scatter_) -> th.Tensor:
        """ Steepest descent """
        return -g


    def initialize_algo_param(self):
        """
        Override this method to initialize attribute variables for self._update_direction.

        Returns: None
        """
        if self.iterform == "SD":
            self.__update_scheme = self.__SD
            return
        if self.is_concat_X:
            if self.iterform == "PR+":
                self.__update_scheme = self.__PRP_irreg
            elif self.iterform == "FR":
                self.__update_scheme = self.__FR_irreg
            else:
                raise NotImplementedError("Unknown Iterative Scheme.")
        else:
            if self.iterform == "PR+":
                self.__update_scheme = self.__PRP_reg
            elif self.iterform == "FR":
                self.__update_scheme = self.__FR_reg
            else:
                raise NotImplementedError("Unknown Iterative Scheme.")

    def _update_direction(
            self,
            g: th.Tensor,
            g_old: th.Tensor,
            p: th.Tensor,
            X: th.Tensor,
            batch_scatter_: th.Tensor | None,
    ) -> th.Tensor:
        """
        Override this method to implement X update algorithm.
        Args:
            g: (n_batch, n_atom, n_dim), the gradient of X at this step
            g_old: (n_batch, n_atom, n_dim), the gradient of X at last step
            p: (n_batch, n_atom, n_dim), the update direction of X at last step
            X: (n_batch, n_atom, n_dim), the independent vars X.

        Returns:
            p: th.Tensor, the new update direction of X.
        """
        p = self.__update_scheme(g, g_old, p, X, batch_scatter_)

        return p

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
        pass

