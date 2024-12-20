"""
Conjugate Gradient Algorithm for optimization.
"""
#  Copyright (c) 2024.12.10, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.7b
#  File: CG.py
#  Environment: Python 3.12

from typing import Literal

from BM4Ckit.BatchOptim._utils._BaseOpt import _BaseOpt
import torch as th


class CG(_BaseOpt):
    """
    Conjugate Gradient Algo.
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
        if self.iterform != "SD":
            gogo = g_old.mT @ g_old
            ggo = g.mT @ g_old  # (n_batch, 1, 1)
            gg = g.mT @ g  # (n_batch, 1, 1)
            if self.iterform == "PR+":
                beta = (gg - ggo) / gogo  # (n_batch, 1, 1)
                beta = th.where(beta < 0.0, 0.0, beta)
            elif self.iterform == "PR":
                beta = (gg - ggo) / gogo  # (n_batch, 1, 1)
            elif self.iterform == "FR":
                beta = gg / gogo
            elif self.iterform == "WYL":
                beta = (gg - th.sqrt(gg / gogo) * ggo) / gogo
            else:
                raise NotImplementedError("Unknown Iterative Scheme.")
            # Restart
            is_restart = (ggo >= 0) * (gg > ggo) * (gogo >= gg)
            beta = th.where(is_restart, 0.0, beta)
            if self.verbose > 0:
                self.logger.info(f"Restart: {is_restart.flatten().cpu().numpy()}")
            # update directions
            p = -g + beta * p  # (n_batch, n_atom*3, 1)
        else:
            p = -g

        return p

    def _update_algo_param(self, g: th.Tensor, g_old: th.Tensor, p: th.Tensor, displace: th.Tensor) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.

        Returns: None
        """
        pass

