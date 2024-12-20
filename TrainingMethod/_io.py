""" Input / Output Module """

#  Copyright (c) 2024.12.10, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.7b
#  File: _io.py
#  Environment: Python 3.12

import logging
import os
import sys
import time
from typing import Optional, Dict, Callable, Any, Literal, Sequence

import torch as th
import yaml
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ChainedScheduler, ConstantLR, LambdaLR, LinearLR
from torcheval.metrics.functional import r2_score

from .Losses import Energy_Force_Loss, Energy_Loss
from .Metrics import E_MAE, E_R2, F_MAE, F_MaxE


class _LoggingEnd:
    """
    A Context Manager of Setting Terminator of Logger.
    Temporarily change the terminator of log handler into `end`, and reset to '\n' when it closed.

    Parameters:
        logger: the handler of logging.
        end: str, the set terminator of logger.
    """

    def __init__(self, logger: logging.StreamHandler, end: str = ''):
        self.handler = logger
        self.end = end

    def __enter__(self):
        self.handler.terminator = self.end

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler.terminator = '\n'


class _CONFIGS(object):
    r"""
    A base class of loading configs.
    """

    def __init__(self, config_file: str, device: str | th.device = 'cpu') -> None:
        self.config_file = config_file
        self.DEVICE = device
        self._OPTIM_DICT = {'Adam': th.optim.Adam, 'SGD': th.optim.SGD, 'AdamW': th.optim.AdamW, 'Adadelta': th.optim.Adadelta,
                            'Adagrad': th.optim.Adagrad, 'ASGD': th.optim.ASGD, 'Adamax': th.optim.Adamax, 'custom': None}
        self._LR_SCHEDULER_DICT = {'StepLR': StepLR, 'ExponentialLR': ExponentialLR, 'ChainedScheduler': ChainedScheduler,
                                   'ConstantLR': ConstantLR, 'LambdaLR': LambdaLR, 'LinearLR': LinearLR, 'custom': None}
        self._LOSS_DICT = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss, 'Hubber': nn.HuberLoss, 'CrossEntropy': nn.CrossEntropyLoss,
                           'Energy_Force_Loss': Energy_Force_Loss, 'Energy_Loss': Energy_Loss, 'custom': None}
        self._METRICS_DICT = {'MSE': F.mse_loss, 'MAE': F.l1_loss, 'R2': r2_score, 'RMSE': _rmse,
                              'E_MAE': E_MAE, 'E_R2': E_R2, 'F_MAE': F_MAE, 'F_MaxE': F_MaxE, 'custom': None}
        self.param = None
        self._has_load_data = False
        self._data_loader = None

    def set_device(self, device: str | th.device) -> None:
        """ reset the device that model would train on """
        self.DEVICE = device

    def set_loss_fn(self, loss_fn, loss_config: Optional[Dict] = None) -> None:
        """
        Reset loss function, and reset configs of loss function optionally.
        parameters:
            loss_fn: uninstantiated class torch.nn.Module, a user-defind loss function.
            loss_config: Dict[str, Any]|None, the new configs of given loss function. if None, loss_config would not change.
        """
        if loss_config is None:
            pass
        elif not isinstance(loss_config, Dict):
            raise TypeError('loss_config must be a dictionary.')
        else:
            self.LOSS_CONFIG = loss_config
        self.LOSS = loss_fn

    def set_metrics(self, metrics_fn: Dict[str, Callable], metrics_fn_config: Dict[str, Dict] | None = None):
        """
        Set user-defined metrics function.
        Parameters:
            metrics_fn: Dict[str, Callable], str is the name of metrics function.
            metrics_fn_config: Dict[str, Dict]|None, the configs of metrics function corresponding to the function name str.
        """
        if metrics_fn_config is None: metrics_fn_config = dict()
        for _key in metrics_fn.keys():
            if _key not in metrics_fn_config:
                metrics_fn_config[_key] = dict()
        self.METRICS.update(metrics_fn)
        self.METRICS_CONFIG.update(metrics_fn_config)

    def set_model_config(self, model_config: Dict[str, Any] | None = None) -> None:
        """
        Set the new configs (hyperparameters) of model.
        """
        if model_config is None: model_config = dict()
        self.MODEL_CONFIG = model_config

    def set_lr_scheduler(self, lr_scheduler, lr_scheduler_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the lr_scheduler that inherit from torch.optim.lr_scheduler.LRScheduler
        """
        self.LR_SCHEDULER = lr_scheduler
        if lr_scheduler_config is not None:
            self.LR_SCHEDULER_CONFIG = lr_scheduler_config

    def set_model_param(self, model_state_dict: Dict, is_strict: bool = True, is_assign: bool = False) -> None:
        """
        Set the trained model parameters from direct input.
        Parameters:
            model_state_dict: Dict, a dict containing parameters and persistent buffers.
            is_strict: bool,whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function.
            is_assign: bool, When False, the properties of the tensors in the current module are preserved while when True,
            the properties of the Tensors in the state dict are preserved.
        """
        if isinstance(model_state_dict, Dict):
            self.param = model_state_dict
        else:
            raise TypeError(f'model_state_dict must be a Dict, but occurred {type(model_state_dict)}')
        self.is_strict = is_strict
        self.is_assign = is_assign

    def set_optimizer(self, optimizer, optim_config: Optional[Dict] = None) -> None:
        r"""
        Set the optimizer that inherit from torch.optim.Optimizer, and reset optimizer configs optionally.
        parameters:
            optimizer: torch.optim.Optimizer, a user-defind optimizer.
            optim_config: Dict[str, Any]|None, the new configs of given optimizer. if None, optim_config would not change.
        """
        if optim_config is None:
            pass
        elif not isinstance(optim_config, Dict):
            raise TypeError('optim_config must be a dictionary.')
        else:
            self.OPTIM_CONFIG = optim_config
        self.OPTIMIZER = optimizer

    def set_dataloader(self, DataLoader, DataLoader_configs: Dict | None = None) -> None:
        r"""
        Set the data loader which is :
            * DataLoader(data, batchsize, device, **kwargs) -> Iterable
            * next(iter( DataLoader(data) )) -> (data, label)
        The argument 'batchsize' and 'device' of DataLoader would be read from self.BATCH_SIZE and self.DEVICE respectively.
        """
        if DataLoader_configs is None: DataLoader_configs = dict()
        self._data_loader = DataLoader
        self._data_loader_configs = DataLoader_configs

    def set_dataset(self, train_data: Dict[Literal['data', 'labels'], Any], valid_data: Optional[Dict[Literal['data', 'labels'], Any]] = None):
        r"""
        Load the data that put into DataLoader.
        Parameters:
            train_data: {'data': Any, 'labels':Any}, the Dict of training set.
            valid_data: {'data': Any, 'labels':Any}, the Dict of validation set.
        Both training and validation set data must implement __len__() method, and they are correspond to the input of dataloader.
        """
        self._has_load_data = True
        self.TRAIN_DATA = train_data
        if valid_data is not None:
            self.VALID_DATA = valid_data
        else:
            self.VALID_DATA = {'data': list(), 'label': list()}

    def reload_config(self, config_file_path: str) -> None:
        """
        Reload the yaml configs file.
        """
        # load config file
        with open(config_file_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        self.config = config

        # global information
        self.START = self.config.get('START', 0)
        if self.START != 'from_scratch' and self.START != 0:
            self.LOAD_CHK_FILE_PATH = self.config['LOAD_CHK_FILE_PATH']
            if not isinstance(self.LOAD_CHK_FILE_PATH, str): raise TypeError('LOAD_CHK_FILE_PATH must be a str.')
        self.EPOCH: int = self.config.get('EPOCH', 0)
        self.BATCH_SIZE: int = self.config.get('BATCH_SIZE', 1)
        self.VAL_PER_STEP: int = self.config.get('VAL_PER_STEP', 10)
        self.VAL_BATCH_SIZE: int = self.config.get('VAL_BATCH_SIZE', self.BATCH_SIZE)

        # model info
        self.MODEL_NAME: str = self.config.get('MODEL_NAME', 'Untitled')
        if not isinstance(self.MODEL_NAME, str): raise TypeError('MODEL_NAME must be a str.')
        self.MODEL_CONFIG = self.config.get('MODEL_CONFIG', dict())
        if not isinstance(self.MODEL_CONFIG, Dict): raise ValueError('MODEL_CONFIG must be a dictionary.')

        # optim info
        optim_name = self.config.get('OPTIM', None)
        if optim_name is not None:
            self.OPTIMIZER = self._OPTIM_DICT.get(optim_name, None)
        else:
            self.OPTIMIZER = None
        self.OPTIM_CONFIG = self.config.get('OPTIM_CONFIG', dict())
        if not isinstance(self.OPTIM_CONFIG, Dict): raise ValueError('OPTIM_CONFIG must be a dictionary.')

        self.GRAD_CLIP: bool = self.config.get('GRAD_CLIP', False)
        self.GRAD_CLIP_MAX_NORM: float = self.config.get('GRAD_CLIP_MAX_NORM', 100)
        self.GRAD_CLIP_CONFIG = self.config.get('GRAD_CLIP_CONFIG', dict())
        if not isinstance(self.GRAD_CLIP, bool): raise TypeError('GRAD_CLIP must be a boolean.')
        if not isinstance(self.GRAD_CLIP_CONFIG, Dict): raise ValueError('GRAD_CLIP_CONFIG must be a dictionary.')

        self.ACCUMULATE_STEP = self.config.get('ACCUMULATE_STEP', 1)
        if not (isinstance(self.ACCUMULATE_STEP, int) and self.ACCUMULATE_STEP > 0): raise TypeError('ACCUMULATE_STEP must be a positive integer.')

        self.LR_SCHEDULER = self.config.get('LR_SCHEDULER', None)
        self.LR_SCHEDULER_CONFIG = self.config.get('LR_SCHEDULER_CONFIG', dict())
        if not isinstance(self.LR_SCHEDULER_CONFIG, Dict): raise TypeError('LR_SCHEDULER_CONFIG must be a dict.')

        # loss & criterion info
        self.loss_name = self.config.get('LOSS', None)
        if self.loss_name is not None:
            self.LOSS = self._LOSS_DICT.get(self.loss_name, None)
        else:
            self.LOSS = None
        self.LOSS_CONFIG = self.config.get('LOSS_CONFIG', dict())
        if not isinstance(self.LOSS_CONFIG, Dict): raise TypeError('LOSS_CONFIG must be a dictionary.')
        __metric_name = self.config.get('METRICS', tuple())
        self.METRICS_CONFIG = self.config.get('METRICS_CONFIG', dict())
        if not isinstance(__metric_name, Sequence): raise TypeError(f'METRICS must be a sequence, but occurred {type(__metric_name)}')
        if not isinstance(self.METRICS_CONFIG, Dict): raise TypeError(f'METRICS_CONFIG must be a dict, but occurred {type(self.METRICS_CONFIG)}')
        self.METRICS = dict()
        for __metric in __metric_name:
            if __metric not in self._METRICS_DICT:
                self.METRICS[__metric] = None
                self.METRICS_CONFIG[__metric] = dict()
            elif __metric not in self.METRICS_CONFIG:
                self.METRICS[__metric] = self._METRICS_DICT[__metric]
                self.METRICS_CONFIG[__metric] = dict()
            else:
                self.METRICS[__metric] = self._METRICS_DICT[__metric]

        # output info
        self.REDIRECT = self.config.get('REDIRECT', True)
        self.SAVE_CHK = self.config.get('SAVE_CHK', False)
        self.SAVE_PREDICTIONS = self.config.get('SAVE_PREDICTIONS', False)
        if not isinstance(self.SAVE_PREDICTIONS, bool):
            raise TypeError(f'SAVE_PREDICTIONS must be a boolean, but occurred {type(self.SAVE_PREDICTIONS)}.')
        self.PREDICTIONS_SAVE_FILE = self.config.get('PREDICTIONS_SAVE_FILE', './Predictions_Ef_Origin')
        while os.path.exists(self.PREDICTIONS_SAVE_FILE + '.npz'):  # avoid overwrite existent data. Automatically rename.
            self.PREDICTIONS_SAVE_FILE += '_1'
        if (self.SAVE_PREDICTIONS) and (not isinstance(self.PREDICTIONS_SAVE_FILE, str)):
            raise TypeError(f'PREDICTIONS_SAVE_PATH must be a str, but occurred {type(self.PREDICTIONS_SAVE_FILE)}.')
        if not isinstance(self.REDIRECT, bool): raise TypeError('REDIRECT must be a boolean.')
        if self.REDIRECT:
            self.OUTPUT_PATH = self.config.get('OUTPUT_PATH', './')
            self.OUTPUT_POSTFIX = self.config.get('OUTPUT_POSTFIX', 'Untitled')
        self.CHK_SAVE_PATH = self.config.get('CHK_SAVE_PATH', './')
        self.CHK_SAVE_POSTFIX = self.config.get('CHK_SAVE_POSTFIX', '')
        if self.CHK_SAVE_POSTFIX != '': self.CHK_SAVE_POSTFIX = '_' + self.CHK_SAVE_POSTFIX  # use '_' to delimit chk name.
        if not isinstance(self.CHK_SAVE_PATH, str): raise TypeError('CHK_SAVE_PATH must be a str.')

        # debug mode
        self.DEBUG_MODE = self.config.get('DEBUG_MODE', False)
        if not isinstance(self.DEBUG_MODE, bool): raise TypeError('DEBUG_MODE must be a boolean.')

        # logging
        # logging.getLogger().disabled = True
        self.logger = logging.getLogger('Main')
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if self.REDIRECT:
            output_file = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
            # check whether path exist
            if not os.path.isdir(self.OUTPUT_PATH): os.makedirs(self.OUTPUT_PATH)
            # set log handler
            self.log_handler = logging.FileHandler(output_file, 'w', delay=True)
            self.log_handler.setLevel(logging.INFO)
            self.log_handler.setFormatter(formatter)
        else:
            self.log_handler = logging.StreamHandler(stream=sys.stdout)
            self.log_handler.setLevel(logging.INFO)
            self.log_handler.setFormatter(formatter)
        if not self.logger.hasHandlers(): self.logger.addHandler(self.log_handler)

        # If Structure opt.
        self.RELAXATION = self.config.get('RELAXATION', None)

        # If Molecular Dynamics
        self.MD = self.config.get('MD', None)


class _Model_Wrapper_pyg:
    def __init__(self, model) -> None:
        """
        A format transformer for converting Tensor X into PygData.pos
        Wrap the model(graph, ...) into f(X)

        Args:
            model: An instantiate nn.Module

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Forces: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].

        """
        self._model = model
        self.forces = None
        pass

    def Energy(self, X, graph, return_format: Literal['sum', 'origin'] = 'origin'):
        graph.pos = X.squeeze(0)
        y = self._model(graph)
        energy = y['energy']
        self.forces = y['forces']
        if return_format == 'sum':
            energy = th.sum(energy).unsqueeze(0)
        return energy

    def Grad(self, X, graph):
        if self.forces is None:
            graph.pos = X.squeeze(0)
            return - ((self._model(graph))['forces']).unsqueeze(0)
        else:
            force = self.forces
            del self.forces
            return - force.unsqueeze(0)


class _Model_Wrapper_dgl:
    pass


def _rmse(x, y, reduction='mean') -> th.Tensor:
    return th.sqrt(F.mse_loss(x, y, reduction=reduction))
