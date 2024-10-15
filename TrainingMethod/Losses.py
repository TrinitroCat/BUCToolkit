'''

'''
from typing import Literal, List, Dict, Sequence, Tuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Energy_Force_Loss(nn.Module):
    """
    A loss function that evaluate both predicted energy and forces.

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
            self.loss_E = nn.SmoothL1Loss()
        elif loss_E == 'MSE':
            self.loss_E = nn.MSELoss()
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
        loss = self.coeff_E * self.loss_E(label['energy'], pred['energy']) + self.coeff_F * self.loss_F(label['forces'], pred['forces'])
        return loss

class Energy_Loss(nn.Module):
    """
    A loss function that evaluate both predicted energy and forces.

    Parameters:
        loss_E: the loss function of energy.

    forward:
        pred: Dict[Literal['energy'], th.Tensor], output of models;
        label: Dict[Literal['energy'], th.Tensor], labels.

    """
    def __init__(self,
                 loss_E: Literal['MAE', 'MSE'] | nn.Module = 'MAE',) -> None:
        super().__init__()
        if loss_E == 'MAE':
            self.loss_E = nn.SmoothL1Loss()
        elif loss_E == 'MSE':
            self.loss_E = nn.MSELoss()
        else:
            self.loss_E = loss_E

    def forward(self, pred: Dict[Literal['energy'], th.Tensor], label: Dict[Literal['energy'], th.Tensor]):
        loss = self.loss_E(label['energy'], pred['energy'])
        return loss
