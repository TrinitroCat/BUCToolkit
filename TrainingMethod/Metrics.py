""" Metrics """
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Metrics.py
#  Environment: Python 3.12

from typing import Literal, Dict, List, Tuple
import warnings
import torch as th
import torch.nn.functional as F


def _r2_score(y_pred: th.Tensor, y_true: th.Tensor) -> th.Tensor:
        """
        Calculate R^2
        """
        # initialize
        y1 = y_pred.float()
        y2 = y_true.float()
        if y1.dim() == 1:
            y1 = y1.view(-1, 1)
            y2 = y2.view(-1, 1)

        # SS_res
        ss_res = th.sum((y1 - y2) ** 2, dim=0)

        # SS_tol
        y1_mean = th.mean(y1, dim=0, keepdim=True)
        ss_tot = th.sum((y1 - y1_mean) ** 2, dim=0)

        # Main
        r2 = th.where(
            ss_tot != 0,
            1 - (ss_res / ss_tot),
            th.where(
                ss_res != 0,
                -th.inf,
                th.ones_like(ss_res)
            )
        )

        return r2.squeeze()

def _rmse(x, y, reduction='mean') -> th.Tensor:
    return th.sqrt(F.mse_loss(x, y, reduction=reduction))

def E_MAE(pred: Dict[Literal['energy', 'forces'], th.Tensor],
          label: Dict[Literal['energy', 'forces'], th.Tensor],
          reduction: Literal['mean', 'sum', 'none'] = 'mean'):
    with th.no_grad():
        if isinstance(pred['energy'], th.Tensor):
            mae = F.l1_loss(pred['energy'], label['energy'], reduction=reduction)
        elif isinstance(pred['energy'], List|Tuple):
            mae = sum(F.l1_loss(pred_, label['energy'], reduction=reduction) for pred_ in pred['energy'])

    return mae


def E_R2(pred: Dict[Literal['energy', 'forces'], th.Tensor],
         label: Dict[Literal['energy', 'forces'], th.Tensor],):
    if len(pred['energy']) <= 2:
        r2 = th.tensor([1.], dtype=th.float32)
        warnings.warn('Input samples less than 2, r2 was set to 1.', RuntimeWarning)
    else:
        with th.no_grad():
            r2 = _r2_score(pred['energy'], label['energy'])
    return r2


def F_MAE(pred: Dict[Literal['energy', 'forces'], th.Tensor],
          label: Dict[Literal['energy', 'forces'], th.Tensor],
          reduction: Literal['mean', 'sum', 'none'] = 'mean'):
    with th.no_grad():
        mae = F.l1_loss(pred['forces'], label['forces'], reduction=reduction)
    return mae


def F_MaxE(pred: Dict[Literal['energy', 'forces'], th.Tensor],
           label: Dict[Literal['energy', 'forces'], th.Tensor]):
    with th.no_grad():
        max_ = th.max(pred['forces'] - label['forces'])
    return max_
