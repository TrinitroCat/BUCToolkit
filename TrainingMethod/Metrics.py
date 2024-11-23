""" Metrics """
from typing import Literal, Dict, List, Tuple
import warnings
import torch as th
import torch.nn.functional as F
from torcheval.metrics.functional import r2_score


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
         label: Dict[Literal['energy', 'forces'], th.Tensor],
         multioutput: Literal['uniform_average', 'raw_values', 'variance_weighted'] = 'uniform_average'):
    if len(pred['energy']) <= 2:
        r2 = th.tensor([1.], dtype=th.float32)
        warnings.warn('Input samples less than 2, r2 was set to 1.', RuntimeWarning)
    else:
        with th.no_grad():
            r2 = r2_score(pred['energy'], label['energy'], multioutput=multioutput)
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
