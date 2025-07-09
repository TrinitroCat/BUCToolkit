"""
Build-in Losses function
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Losses.py
#  Environment: Python 3.12

from typing import Literal, List, Dict, Sequence, Tuple
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Energy_Force_Loss(nn.Module):
    """
    A loss function that evaluates both predicted energy and forces.
    Both energy & force losses would be averaged over each atom.

    Parameters:
        loss_E: the loss function of energy.
        loss_F: the loss function of forces.
        coeff_E: the coefficient of energy.
        coeff_F: the coefficient of forces.

    forward:
        pred: Dict[Literal['energy'], th.Tensor], output of models;
        label: Dict[Literal['energy'], th.Tensor], labels.

    """
    def __init__(self, 
                 loss_E: Literal['MAE', 'MSE']|nn.Module='MAE', 
                 loss_F: Literal['MAE', 'MSE']|nn.Module='MAE', 
                 coeff_E: float=1., 
                 coeff_F: float=1.) -> None:
        super().__init__()
        if loss_E == 'MAE':
            self.loss_E = nn.SmoothL1Loss(reduction='sum')
        elif loss_E == 'MSE':
            self.loss_E = nn.MSELoss(reduction='sum')
        elif loss_E == 'SmoothMAE':
            self.loss_E = nn.SmoothL1Loss(reduction='sum')
        else:
            self.loss_E = loss_E

        if loss_F == 'MAE':
            self.loss_F = nn.SmoothL1Loss(reduction='sum')
        elif loss_F == 'MSE':
            self.loss_F = nn.MSELoss(reduction='sum')
        else:
            self.loss_F = loss_F
        
        self.coeff_E = coeff_E
        self.coeff_F = coeff_F
    
    def forward(self, pred:Dict[Literal['energy', 'forces'], th.Tensor], label:Dict[Literal['energy', 'forces'], th.Tensor]):
        n_atom = math.prod(pred['forces'].shape[:-1])
        loss = (self.coeff_E * self.loss_E(pred['energy'], label['energy']) +
                self.coeff_F * self.loss_F(pred['forces'], label['forces']))/n_atom
        return loss

class Energy_Loss(nn.Module):
    """
    A loss function that evaluates both predicted energy and forces.

    Parameters:
        loss_E: the loss function of energy.

    forward:
        pred: Dict[Literal['energy'], th.Tensor], output of models;
        label: Dict[Literal['energy'], th.Tensor], labels.

    """
    def __init__(self, loss_E: Literal['MAE', 'MSE', 'SmoothMAE', 'Huber'] | nn.Module = 'MAE',) -> None:
        super().__init__()
        if loss_E == 'MAE':
            self.loss_E = nn.L1Loss()
        elif loss_E == 'SmoothMAE':
            self.loss_E = nn.SmoothL1Loss()
        elif loss_E == 'MSE':
            self.loss_E = nn.MSELoss()
        elif loss_E == 'Huber':
            self.loss_E = nn.HuberLoss()
        else:
            self.loss_E = loss_E

    def forward(self, pred: Dict[Literal['energy'], th.Tensor], label: Dict[Literal['energy'], th.Tensor]):
        loss = self.loss_E(pred['energy'], label['energy'])
        return loss


class WrapperBoostLoss(nn.Module):
    def __init__(self, loss_E: Literal['MAE', 'MSE', 'SmoothMAE', 'Huber'] | nn.Module = 'MAE', loss_F = None) -> None:
        super().__init__()
        if loss_E == 'MAE':
            self.loss_E = nn.L1Loss()
        elif loss_E == 'SmoothMAE':
            self.loss_E = nn.SmoothL1Loss()
        elif loss_E == 'MSE':
            self.loss_E = nn.MSELoss()
        elif loss_E == 'Huber':
            self.loss_E = nn.HuberLoss()
        else:
            self.loss_E = loss_E

        if loss_F is None:
            self.has_loss_f = False
        else:
            if loss_F == 'MAE':
                self.loss_F = nn.SmoothL1Loss(reduction='sum')
            elif loss_F == 'MSE':
                self.loss_F = nn.MSELoss(reduction='sum')
            else:
                self.loss_F = loss_F

    def forward(self, pred: Dict[Literal['energy', 'forces'], List[th.Tensor]], label: Dict[Literal['energy', 'forces'], th.Tensor]):
        loss = 0.
        res = label['energy']
        for pred_ in pred['energy']:
            loss = loss + self.loss_E(res, pred_)
            res = res - pred_
        if self.has_loss_f:
            res_f = label['forces']
            for pred_ in pred['forces']:
                loss = loss + self.loss_F(res_f, pred_)
                res_f = res - pred_

        return loss


class WrapperMeanLoss(nn.Module):
    def __init__(
            self,
            loss_E: Literal['MAE', 'MSE', 'SmoothMAE', 'Huber'] | nn.Module = 'MAE',
            loss_F: Literal['MAE', 'MSE', 'SmoothMAE', 'Huber'] | nn.Module | None = None
    ) -> None:
        super().__init__()
        if loss_E == 'MAE':
            self.loss_E = nn.L1Loss()
        elif loss_E == 'SmoothMAE':
            self.loss_E = nn.SmoothL1Loss()
        elif loss_E == 'MSE':
            self.loss_E = nn.MSELoss()
        elif loss_E == 'Huber':
            self.loss_E = nn.HuberLoss()
        else:
            self.loss_E = loss_E

        if loss_F is None:
            self.has_loss_f = False
        else:
            self.has_loss_f = True
            if loss_F == 'MAE':
                self.loss_F = nn.SmoothL1Loss(reduction='sum')
            elif loss_F == 'MSE':
                self.loss_F = nn.MSELoss(reduction='sum')
            else:
                self.loss_F = loss_F

    def forward(self, pred: Dict[Literal['energy', 'forces'], List[th.Tensor]], label: Dict[Literal['energy', 'forces'], th.Tensor]):
        loss = 0.
        for pred_ in pred['energy']:
            loss = loss + self.loss_E(label['energy'], pred_)
        if self.has_loss_f:
            for pred_ in pred['forces']:
                loss = loss + self.loss_F(label['forces'], pred_)

        return loss
