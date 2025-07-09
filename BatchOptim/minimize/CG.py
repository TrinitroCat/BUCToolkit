"""
Conjugate Gradient Algorithm for optimization.
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: CG.py
#  Environment: Python 3.12

from typing import Literal

from BM4Ckit.BatchOptim._BaseOpt import _BaseOpt
import torch as th


class CG(_BaseOpt):
    """
    Conjugate Gradient Algo.
    Args:
        iter_scheme: Literal['PR+', 'PR', 'FR', 'SD', 'WYL'],
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
    """
    def __init__(
            self,
            iter_scheme: Literal['PR+', 'PR', 'FR', 'SD', 'WYL'],
            E_threshold: float = 1e-3,
            F_threshold: float = 0.05,
            maxiter: int = 100,
            linesearch: Literal['Backtrack', 'Wolfe', 'NWolfe', '2PT', '3PT', 'Golden', 'Newton', 'None'] = 'Backtrack',
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
        self.__update_scheme = self.__resolve_update_scheme()

    def __resolve_update_scheme(self):
        """
        resolve different iteration scheme
        Returns:

        """
        if self.iterform != "SD":
            if self.iterform == "PR+":
                return self.__PRP
            elif self.iterform == "PR":
                return self.__PR
            elif self.iterform == "FR":
                return self.__FR
            elif self.iterform == "WYL":
                return self.__WYL
            else:
                raise NotImplementedError("Unknown Iterative Scheme.")
        else:
            return self.__SD

    def __PRP(self , g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor) -> th.Tensor:
        """

        """
        gogo = g_old.mT @ g_old + 1e-16  # to avoid divide 0
        ggo = g.mT @ g_old  # (n_batch, 1, 1)
        gg = g.mT @ g  # (n_batch, 1, 1)
        beta = (gg - ggo) / gogo  # (n_batch, 1, 1)
        beta = th.where(beta < 0.0, 0.0, beta)
        # Restart
        is_restart = (ggo >= 0) * (gg > ggo) * (gogo >= gg)
        beta = th.where(is_restart, 0.0, beta)
        if self.verbose > 0:
            self.logger.info(f"Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        p = -g + beta * p  # (n_batch, n_atom*3, 1)

        return p

    def __PR(self , g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor) -> th.Tensor:
        """ """
        gogo = g_old.mT @ g_old + 1e-16  # to avoid divide 0
        ggo = g.mT @ g_old  # (n_batch, 1, 1)
        gg = g.mT @ g  # (n_batch, 1, 1)
        beta = (gg - ggo) / gogo  # (n_batch, 1, 1)
        # Restart
        is_restart = (ggo >= 0) * (gg > ggo) * (gogo >= gg)
        beta = th.where(is_restart, 0.0, beta)
        if self.verbose > 0:
            self.logger.info(f"Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        p = -g + beta * p  # (n_batch, n_atom*3, 1)

        return p

    def __FR(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor) -> th.Tensor:
        """ """
        gogo = g_old.mT @ g_old + 1e-16  # to avoid divide 0
        ggo = g.mT @ g_old  # (n_batch, 1, 1)
        gg = g.mT @ g  # (n_batch, 1, 1)
        beta = gg / gogo
        # Restart
        is_restart = (ggo >= 0) * (gg > ggo) * (gogo >= gg)
        beta = th.where(is_restart, 0.0, beta)
        if self.verbose > 0:
            self.logger.info(f"Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        p = -g + beta * p  # (n_batch, n_atom*3, 1)

        return p

    def __WYL(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor) -> th.Tensor:
        """ """
        gogo = g_old.mT @ g_old + 1e-16  # to avoid divide 0
        ggo = g.mT @ g_old  # (n_batch, 1, 1)
        gg = g.mT @ g  # (n_batch, 1, 1)
        beta = (gg - th.sqrt(gg / gogo) * ggo) / gogo
        # Restart
        is_restart = (ggo >= 0) * (gg > ggo) * (gogo >= gg)
        beta = th.where(is_restart, 0.0, beta)
        if self.verbose > 0:
            self.logger.info(f"Restart: {is_restart.flatten().cpu().numpy()}")
        # update directions
        p = -g + beta * p  # (n_batch, n_atom*3, 1)

        return p

    def __SD(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, X: th.Tensor) -> th.Tensor:
        """ Steepest descent """
        return -g


    def initialize_algo_param(self):
        """
        Override this method to initialize attribute variables for self._update_direction.

        Returns: None
        """
        pass

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
        p = self.__update_scheme(g, g_old, p, X)

        return p

    def _update_algo_param(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, displace: th.Tensor) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.

        Returns: None
        """
        pass

