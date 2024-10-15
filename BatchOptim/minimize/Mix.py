r"""
Batched structures optimization by pytorch models or functions.

2024/6/24 PPX
"""

# ruff: noqa: E701, E702, E703
from typing import (
    Dict,
    Any,
    Literal,
    Optional,
    Sequence,
)

import torch as th
from torch import nn

from .CG import CG
from .QN import QN
from BM4Ckit.BatchOptim._utils._line_search import _LineSearch


class Mix:
    def __init__(
        self,
        iterschem_CG: Literal["PR", "PR+", "FR", "SD", "WYL"] = "PR+",
        iterschem_QN: Literal["BFGS", "Newton"] = "BFGS",
        switch_cond: Optional[float] = None,
        E_threshold: float = 1e-3,
        F_threshold: float = 0.05,
        maxiter: int = 100,
        linesearch: Literal[
            "Backtrack", "Golden", "Wolfe", "Newton", "None"
        ] = "Backtrack",
        linesearch_maxiter: int = 10,
        linesearch_thres: float = 0.02,
        linesearch_factor: float = 0.6,
        steplength: float = 0.1,
        device: str | th.device = "cpu",
        verbose: int = 2,
    ) -> None:
        self.iterschem_CG = iterschem_CG
        self.iterschem_QN = iterschem_QN
        self.linesearch: str = linesearch
        self.steplength: float = steplength
        self.linesearch_maxiter = linesearch_maxiter
        self.linesearch_thres = linesearch_thres
        self.linesearch_factor = linesearch_factor
        self.verbose = verbose
        self.device = device
        self._line_search = _LineSearch(
            linesearch,
            maxiter=linesearch_maxiter,
            thres=linesearch_thres,
            steplength=steplength,
            factor=linesearch_factor,
        )
        self.E_threshold = E_threshold
        self.F_threshold = F_threshold
        self.maxiter = maxiter
        if switch_cond is None:
            self.switch_cond = 100.0 * F_threshold
        else:
            self.switch_cond = switch_cond

    def run(
        self,
        func: Any | nn.Module,
        X: th.Tensor,
        grad_func: Any | nn.Module = None,
        func_args: Sequence = tuple(),
        func_kwargs: Dict = dict(),
        grad_func_args: Sequence = tuple(),
        grad_func_kwargs: Dict = dict(),
        is_grad_func_contain_y: bool = True,
        output_grad: bool = False,
        fixed_atom_tensor: Optional[th.Tensor] = None,
    ):
        cg_E_thres = max(1000 * self.E_threshold, 0.5)

        CG_optim = CG(
            self.iterschem_CG,
            cg_E_thres,
            self.switch_cond,
            self.maxiter // 2,  # type: ignore
            self.linesearch,
            self.linesearch_maxiter,
            self.linesearch_thres,
            self.linesearch_factor,  # type: ignore
            self.steplength,
            self.device,
            self.verbose,
        )
        QN_optim = QN(
            self.iterschem_QN,
            self.E_threshold,
            self.F_threshold,
            self.maxiter // 2,  # type: ignore
            self.linesearch,
            self.linesearch_maxiter,
            self.linesearch_thres,
            self.linesearch_factor,  # type: ignore
            self.steplength,
            self.device,
            self.verbose,
        )

        if output_grad:
            y_mid, x_mid = CG_optim.run(
                func,
                X,
                grad_func,
                func_args,
                func_kwargs,
                grad_func_args,
                grad_func_kwargs,  # type: ignore
                is_grad_func_contain_y,
                False,
                fixed_atom_tensor,
            )
            y_fin, x_fin, X_grad = QN_optim.run(
                func,
                x_mid,
                grad_func,
                func_args,
                func_kwargs,
                grad_func_args,
                grad_func_kwargs,  # type: ignore
                is_grad_func_contain_y,
                output_grad,
                fixed_atom_tensor,
            )
        else:
            y_mid, x_mid = CG_optim.run(
                func,
                X,
                grad_func,
                func_args,
                func_kwargs,
                grad_func_args,
                grad_func_kwargs,  # type: ignore
                is_grad_func_contain_y,
                output_grad,
                fixed_atom_tensor,
            )
            y_fin, x_fin = QN_optim.run(
                func,
                x_mid,
                grad_func,
                func_args,
                func_kwargs,
                grad_func_args,
                grad_func_kwargs,  # type: ignore
                is_grad_func_contain_y,
                output_grad,
                fixed_atom_tensor,
            )
        # plist1.extend(plist2)

        if output_grad:
            return y_fin, x_fin, X_grad
        else:
            return y_fin, x_fin
