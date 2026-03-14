""" Input / Output Module """
import gc
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _io.py
#  Environment: Python 3.12

import logging
import os
import queue
import sys
import threading
import time
import traceback
import warnings
from typing import Optional, Dict, Callable, Any, Literal, Sequence, List
import joblib as jb
import numpy as np

import torch as th
import yaml
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ChainedScheduler, ConstantLR, LambdaLR, LinearLR

from .Losses import Energy_Force_Loss, Energy_Loss
from .Metrics import E_MAE, E_R2, F_MAE, F_MaxE, _r2_score, _rmse

from BM4Ckit.utils._CheckModules import check_module
from BM4Ckit import Structures
from BM4Ckit.utils.ElemListReduce import elem_list_reduce
from BM4Ckit.utils._Element_info import ATOMIC_NUMBER, ATOMIC_SYMBOL


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

    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.logger = None
        self.config = dict()
        self.DEVICE = 'cpu'
        self._OPTIM_DICT = {'Adam': th.optim.Adam, 'SGD': th.optim.SGD, 'AdamW': th.optim.AdamW, 'Adadelta': th.optim.Adadelta,
                            'Adagrad': th.optim.Adagrad, 'ASGD': th.optim.ASGD, 'Adamax': th.optim.Adamax, 'custom': None}
        self._LR_SCHEDULER_DICT = {'StepLR': StepLR, 'ExponentialLR': ExponentialLR, 'ChainedScheduler': ChainedScheduler,
                                   'ConstantLR': ConstantLR, 'LambdaLR': LambdaLR, 'LinearLR': LinearLR, 'custom': None}
        self._LOSS_DICT = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss, 'Hubber': nn.HuberLoss, 'CrossEntropy': nn.CrossEntropyLoss,
                           'Energy_Force_Loss': Energy_Force_Loss, 'Energy_Loss': Energy_Loss, 'custom': None}
        self._METRICS_DICT = {'MSE': F.mse_loss, 'MAE': F.l1_loss, 'R2': _r2_score, 'RMSE': _rmse,
                              'E_MAE': E_MAE, 'E_R2': E_R2, 'F_MAE': F_MAE, 'F_MaxE': F_MaxE, 'custom': None}
        self.param = None
        self._has_load_data = False
        self._data_loader = None
        self.reload_config(config_file)
        self.reset_logger()

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

    def set_dataset(
            self,
            train_data: Dict[Literal['data', 'labels', 'dataIS', 'dataFS'], Any],
            valid_data: Optional[Dict[Literal['data', 'labels'], Any]] = None
    ):
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

    def logout_element_information(self, element_tensor, batch_indx):
        """
        Logout element information.
        Args:
            element_tensor: the atom-wise element tensor.
            batch_indx: the number of atoms in a sample within a batch.

        Returns:

        """
        element_list = element_tensor.tolist()
        elem_list = list()
        _element_list = list()
        if batch_indx is not None:
            indx_old = 0
            for indx in batch_indx:
                _element_list.append(element_list[0][indx_old: indx_old + indx])
                indx_old += indx
        else:
            _element_list = element_list
        for elements in _element_list:
            __element_now = ''
            __elem = ''
            elem_info = ''
            __elem_count = ''
            for i, elem in enumerate(elements, 1):
                # get element symbol
                if isinstance(elem, int):
                    __elem = ATOMIC_NUMBER[elem]
                else:
                    __elem = elem
                # count element number
                if __elem == __element_now:
                    __elem_count += 1
                else:
                    elem_info = elem_info + str(__elem_count) + '  '
                    elem_info = elem_info + __elem + ': '
                    __elem_count = 1
                    __element_now = __elem
            elem_info = elem_info + str(__elem_count)
            elem_list.append(elem_info)
        # log out
        for i, ee in enumerate(elem_list):
            self.logger.info(f'Structure {i:>5d}: {ee}')
        self.logger.info('*' * 100)

    def fixation_resolve(self):
        """
        resolving fixation information.
        Returns:

        """
        if self.FIXATIONS is None:
            raise ValueError(f'No fixation information is available for fixation resolving.')

        MODE_DICT = {'FIX':'fix', 'INV_FIX':'inv_fix', 'FREE':'free'}
        SELECT_DICT = {'ELEMENT': 'select_element', 'HEIGHT': 'select_height', 'INDEX': 'atom_index'}

        fix_mode_list = list()
        for k, v in self.FIXATIONS.items():
            if k not in MODE_DICT:
                raise ValueError(f'Unknown fixation mode: {k}')
            key = MODE_DICT[k]
            _tmp_dict = dict()
            for fix_info, fix_val in v.items():
                if fix_info not in SELECT_DICT:
                    raise ValueError(f'Unknown fixation config: {fix_info}')
                _tmp_dict[SELECT_DICT[fix_info]] = fix_val
            _tmp_dict['select_mode'] = key
            # put the inv_fix to the 1st to avoiding overriding
            if key == 'inv_fix':
                fix_mode_list = [_tmp_dict, ] + fix_mode_list
            else:
                fix_mode_list.append(_tmp_dict)

        return fix_mode_list

    def reset_logger(self):
        """
        Reset logger & its handlers.
        Returns: None

        """
        # logging
        # logging.getLogger().disabled = True
        self.logger = logging.getLogger('Main')
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        # remove existing handlers
        for _hdl in list(self.logger.handlers):
            self.logger.removeHandler(_hdl)
            try:
                _hdl.close()
            except Exception as ehdl:
                warnings.warn(f'Failed to close handler {_hdl}: {ehdl}', RuntimeWarning)
        formatter = logging.Formatter('%(message)s')
        if self.REDIRECT:
            output_file = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
            # check whether path exists
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

    def reload_config(self, config_file_path: str| None = None) -> None:
        """
        Reload the yaml configs file.
        """
        # load config file
        config_file_path = self.config_file if config_file_path is None else config_file_path
        with open(config_file_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        self.config = config

        # global information
        self.START = self.config.get('START', 0)
        if self.START != 'from_scratch' and self.START != 0:
            self.LOAD_CHK_FILE_PATH: str = self.config['LOAD_CHK_FILE_PATH']
            if not isinstance(self.LOAD_CHK_FILE_PATH, str): raise TypeError('LOAD_CHK_FILE_PATH must be a str.')
            self.STRICT_LOAD: bool = self.config.get('STRICT_LOAD', True)
            if not isinstance(self.STRICT_LOAD, bool): raise TypeError(f'STRICT_LOAD must be a boolean, but got {type(self.STRICT_LOAD)}')
        self.COMMENTS: str = self.config.get('COMMENTS', 'None.')
        self.VERBOSE: int = int(self.config.get('VERBOSE', 1))
        self.DEVICE: str|th.device = self.config.get('DEVICE', 'cpu')
        self.EPOCH: int = self.config.get('EPOCH', 0)
        self.BATCH_SIZE: int = self.config.get('BATCH_SIZE', 1)
        self.VAL_PER_STEP: int = self.config.get('VAL_PER_STEP', 10)
        self.VAL_BATCH_SIZE: int = self.config.get('VAL_BATCH_SIZE', self.BATCH_SIZE)
        self.VAL_IF_TRN_LOSS_BELOW: float = self.config.get('VAL_IF_TRN_LOSS_BELOW', th.inf)

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

        self.EMA = self.config.get('EMA', False)  # exponential moving average strategy. best_checkpoint saves using ema param, and others do not.
        self.EMA_DECAY = float(self.config.get('EMA_DECAY', 0.999))

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
        self._PREDICTIONS_SAVE_FILE = self.config.get('PREDICTIONS_SAVE_FILE', './_Predictions')
        while os.path.exists(self._PREDICTIONS_SAVE_FILE):  # avoid overwrite existent data. Automatically rename.
            warnings.warn(
                f'`PREDICTIONS_SAVE_FILE`: "{self._PREDICTIONS_SAVE_FILE}" already exists. '
                f'It will be renamed as "{self._PREDICTIONS_SAVE_FILE}_1".',
                RuntimeWarning
            )
            self._PREDICTIONS_SAVE_FILE += '_1'
        if self.SAVE_PREDICTIONS and (not isinstance(self.PREDICTIONS_SAVE_FILE, str)):
            raise TypeError(f'PREDICTIONS_SAVE_PATH must be a str, but occurred {type(self.PREDICTIONS_SAVE_FILE)}.')
        if self.SAVE_PREDICTIONS:
            self.DUMP_FREQUENCY = self.config.get('DUMP_FREQUENCY', 10)
            self.dumper = DumpStructures(self.PREDICTIONS_SAVE_FILE, self.DUMP_FREQUENCY)
        else:
            self.dumper = DumpStructures(None, 1)
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
        self.CHECK_NAN = self.config.get('CHECK_NAN', True)
        if not isinstance(self.CHECK_NAN, bool): raise TypeError('CHECK_NAN must be a boolean.')

        # loading atoms fixation info.
        self.FIXATIONS = self.config.get('FIXATIONS', None)

        # If Structure opt.
        self.RELAXATION = self.config.get('RELAXATION', None)
        self.TRANSITION_STATE = self.config.get('TRANSITION_STATE', None)

        # If Molecular Dynamics
        self.MD = self.config.get('MD', None)

        # NEB Transition State Search
        self.NEB = self.config.get('NEB', None)

        # If Vibration Calc.
        self.VIBRATION = self.config.get('VIBRATION', None)

    @property
    def PREDICTIONS_SAVE_FILE(self):
        return self._PREDICTIONS_SAVE_FILE

    @PREDICTIONS_SAVE_FILE.setter
    def PREDICTIONS_SAVE_FILE(self, value):
        self._PREDICTIONS_SAVE_FILE = value
        if setattr(self, 'DUMP_FREQUENCY', None) is None:
            self.DUMP_FREQUENCY = self.config.get('DUMP_FREQUENCY', 10)
        self.dumper = DumpStructures(self.PREDICTIONS_SAVE_FILE, 10)


def compare_tensors(X1: th.Tensor, X2: th.Tensor):
    """Compare two tensors."""
    char1 = (X1.untyped_storage().data_ptr(),
             X1.storage_offset(),
             tuple(X1.shape),
             tuple(X1.stride()),
             X1.device,
             X1.dtype)
    char2 = (X2.untyped_storage().data_ptr(),
             X2.storage_offset(),
             tuple(X2.shape),
             tuple(X2.stride()),
             X2.device,
             X2.dtype)

    return char1 == char2


class _Model_Wrapper_pyg:
    def __init__(self, model) -> None:
        """
        A format transformer for converting Tensor X into PygData.pos
        Wrap the model(graph, ...) into f(X)

        Args:
            model: An instantiate nn.Module

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Grad: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].

        """
        self._model = model
        self.forces = None
        self.X = None
        if check_module('torch_geometric') is None:
            ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')
        pass

    def Energy(self, X, graph):
        self.X = X
        graph.pos = self.X.reshape(-1,3).contiguous()
        y = self._model(graph)
        energy = y['energy']
        self.forces = y['forces']
        return energy

    def Grad(self, X, graph):
        origin_shape = X.shape
        if (self.X is None) or (not compare_tensors(X, self.X)):
            self.forces = None
        if self.forces is None:
            self.X = X
            graph.pos = self.X.reshape(-1,3)
            return - ((self._model(graph))['forces']).reshape(origin_shape)
        else:
            force = self.forces
            self.forces = None
            return - force.reshape(origin_shape).contiguous()

class _Model_Wrapper_pyg_only_X:
    def __init__(self, model , graph) -> None:
        """
        A format transformer for converting Tensor X into PygData.pos
        Wrap the model(graph, ...) into f(X)

        Args:
            model: An instantiate nn.Module

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Grad: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].
        """
        self._model = model
        self.forces = None
        self.graph = graph
        if check_module('torch_geometric') is None:
            ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')
        pass

    def Energy(self, X,):
        self.X = X
        self.graph.pos = self.X.squeeze(0).reshape(-1,3)
        y = self._model(self.graph )
        energy = y['energy']
        energy = th.sum(energy).unsqueeze(0)
        return energy

    def Grad(self, X):
        origin_shape = X.shape
        if not th.allclose(X, self.X, atol=1e-6):
            self.forces = None
        if self.forces is None:
            self.X = X
            self.graph.pos = self.X.squeeze(0)
            return - ((self._model(self.graph ))['forces']).unsqueeze(0)
        else:
            force = self.forces
            self.forces = None
            return - force.unsqueeze(0)


class _Model_Wrapper_dgl:
    def __init__(self, model) -> None:
        """
        A format transformer for converting Tensor X into DGLGraph.ndata['pos'] i.e., wrapping the model(graph, ...) into f(X)
        The output DGLGraph has format as follows:
        dgl.heterograph(
            {
                ('atom', 'bond', 'atom'): ([], []),
                ('cell', 'disp', 'cell'): ([], [])
            },
            num_nodes_dict={
                'atom': n_atom,
                'cell': 1
            }
        )
        data.nodes['atom'].data['pos']: (n_atom, 3), Atom positions in Cartesian coordinates.
        data.nodes['atom'].data['Z']: (n_atom, ), Atomic numbers.
        data.nodes['cell'].data['cell']: (1, 3, 3), Cell vectors.

        Args:
            model: An instantiate nn.Module subclass

        Methods:
            Energy: input Tensor `X` and DGLGraph `graph`, it will update data.nodes['atom'].data['pos'] into X and return model(graph)['energy'].
            Grad: input Tensor `X` and DGLGraph `graph`, it will update data.nodes['atom'].data['pos'] into X and return model(graph)['forces'].

        """
        self._model = model
        self.forces = None
        if check_module('dgl') is None:
            ImportError('The method is unavailable because the `dgl` cannot be imported.')
        pass

    def Energy(self, X, graph, return_format: Literal['sum', 'origin'] = 'origin'):
        self.X = X
        graph.nodes['atom'].data['pos'] = self.X.squeeze(0)
        y = self._model(graph)
        energy = y['energy']
        self.forces = y['forces']
        if return_format == 'sum':
            energy = th.sum(energy).unsqueeze(0)
        return energy

    def Grad(self, X, graph):
        if not th.allclose(X, self.X, atol=1e-6):
            self.forces = None
        if self.forces is None:
            self.X = X
            graph.nodes['atom'].data['pos'] = self.X.squeeze(0)
            return - ((self._model(graph))['forces']).unsqueeze(0)
        else:
            force = self.forces
            self.forces = None
            return - force.unsqueeze(0)

class _Model_Wrapper_regularBatch_pyg:
    def __init__(self, model) -> None:
        """
        A format transformer for converting Tensor X into PygData.pos
        Wrap the model(graph, ...) into f(X), but here the batch size of graph is only 1, and the batch size (1st dimension) of X is many.
        This wrapper would expand batch of graph into the same as X.

        Args:
            model: An instantiate nn.Module

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Grad: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].

        """
        self._model = model
        self.forces = None
        _pyg = check_module('torch_geometric.data')
        if _pyg is not None:
            import torch_geometric.data as _pyg
            self.pygBatch = _pyg.Batch
        else:
            ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')
        pass

    def Energy(self, X: th.Tensor, graph, return_format: Literal['sum', 'origin'] = 'origin'):
        self.X = X.flatten(0, 1)  # convert X: (n_batch, n_atom, n_dim) into X': (n_batch * n_atom, 3)
        batch_size = X.size(0)
        if graph.batch_size == 1:
            graph = self.pygBatch.from_data_list([graph] * batch_size)
        graph.pos = self.X
        y = self._model(graph)  # (n_batch, )
        energy = y['energy']
        self.forces = y['forces']
        if return_format == 'sum':
            energy = th.sum(energy).unsqueeze(0)
        return energy

    def Grad(self, X, graph):
        origin_shape = X.shape
        if not th.allclose(X, self.X, atol=1e-6):
            self.forces = None
        if self.forces is None:
            self.Energy(X, graph)

        force: th.Tensor = self.forces
        self.forces = None
        return - force.unsqueeze(0)

class ExpMovingAverage:
    """
    Applying exponential moving average for training models.
    """
    def __init__(self, model:nn.Module, decay:float=0.999):
        """

        Args:
            model: The torch model of nn.Module
            decay: The decay coefficient.
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, th.Tensor] = dict()
        self.backup = dict()
        # initialize
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def step(self):
        """
        Applying ema 1 time.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name].mul_(self.decay).add_(param.data.detach(), alpha=1-self.decay)

    def apply(self):
        """
        Applying ema shadow parameters to the trained model, and the original parameters of trained model will be stored in `self.backup`.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.detach().clone()
                param.copy_(self.shadow[name])

    def restore(self):
        """
        Restore backup of original parameters of model from `self.backup` to continue training.
        """
        try:
            if len(self.backup) != 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        assert name in self.backup, f'Parameter {name} is not in backup!'
                        param.copy_(self.backup[name])
            else:
                pass
        finally:
            self.backup = dict()


class DumpStructures:
    """
    Dump structures calculated by `Predictor`, `StructureOptimization`, etc.
    """
    def __init__(self, path:str|None=None, dump_freq: int=1):
        """

        Args:
            path: the path to dump structures. If None, structures will always store in memory.
            dump_freq: the frequency of dump structures.
        """
        self._structures = Structures()
        self._structures.Mode = 'A'
        #self._structures.change_mode('A')
        self.save_path = path
        self.parallelize = None
        self.dump_freq = dump_freq
        self.collect_count = 0
        # initialize Structure
        self.initialize()

    def initialize(self):
        self.idx: List = list()
        # Batch info part
        self.batch_indices_arr: List = list()  # storing the split point (ptr) in each array with n_atom length. It both concludes 0 and len(arr).
        self.elem_batch: List = list()  # storing the split point (ptr) in each array with n_elem length. It both concludes 0 and len(arr).
        # Structure part
        self.cells: List = list()  # cell vectors
        self.pos_type: List = list()  # coordinate type
        self.pos: List = list()  # atomic coordinates
        self.fixations: List = list()  # fixed atoms masks, 0 for fixed and 1 for free.
        self.elem_arr: List = list()  # elements
        self.num_arr: List = list()  # atom number of each element
        # Properties part
        self.energies: List = list()  # energies of structures. None | List[float]
        self.forces: List = list()  # forces of structures. None | List[np.NdArray[N, 3]]

    @staticmethod
    def _exit(errt = None, errv = None, tr = None):
        if errt is not None:
            raise errt(f'{errv}, {tr}')

    def _check_data(self) -> int:
        """
        Check structure numbers and return it.
        Returns: the structure number.

        """
        length_list = [
            len(self.batch_indices_arr),
            len(self.elem_batch),
            len(self.idx),
            len(self.cells),
            len(self.elem_arr),
            len(self.num_arr),
            len(self.pos_type),
            len(self.pos),
            len(self.fixations),
            len(self.energies),
            len(self.forces),
        ]
        if len(set(length_list)) != 1:
            raise RuntimeError(f'structure numbers of some attributes are inconsistent.')

        return length_list[0]

    def collect(
            self,
            batch_indices: List,
            idx: List[str],
            elements: th.Tensor,
            cells: th.Tensor | np.ndarray,
            pos_type: List[str],
            pos: th.Tensor,
            fixations: th.Tensor,
            energies: th.Tensor,
            forces: th.Tensor,
    ):
        """
        Store structures into a `Structures` instance in the ARRAY FORMAT (Mode = 'A') as `self._structres`.
        Args:
            batch_indices: List, the list of atom numbers in each sample.
            idx: List[str],
            elements: th.Tensor[int] | List[List[int]], the atom-wise elements sequence
            cells: th.Tensor | np.ndarray,
            pos_type: List[str],
            pos: th.Tensor,
            fixations: th.Tensor,
            energies: th.Tensor,
            forces: th.Tensor,

        Returns: None

        """
        try:
            # Postprocessing & save TODO: reformat it in future to apply to all functions.
            batch_indices_arr = np.array(batch_indices)
            cells = cells.cpu().numpy(force = True) if isinstance(cells, th.Tensor) else cells
            pos_type = np.array(pos_type)
            pos = pos.cpu().squeeze(0).numpy(force=True)
            forces = forces.cpu().squeeze(0).numpy(force=True)
            energies = energies.cpu().numpy(force=True)
            fixations = fixations.cpu().squeeze(0).numpy(force=True)
            # check elements type
            if isinstance(elements, th.Tensor):
                elem_list = th.split(elements.squeeze(0), batch_indices)
                elem_list = [_.tolist() for _ in elem_list]
            elif isinstance(elements, list):
                elem_list = list(elements)
            else:
                raise TypeError(f'`elements` should be a tensor or a list, but got {type(elements)}.')
            elem_comp_list = list()
            numb_comp_list = list()
            elem_batch_indices = list()
            #if self.parallelize is not None:
            #    _res = self.parallelize(jb.delayed(elem_list_reduce)(_element_list) for _element_list in elem_list)
            #    for _element_list in _res:
            #        _elem_comp_list, _, _number_list = _element_list
            #        elem_comp_list.extend(_elem_comp_list)
            #        numb_comp_list.extend(_number_list)
            #        elem_batch_indices.append(len(_element_list))
            #else:
            for _element_list in elem_list:
                _elem_comp_list, _, _number_list = elem_list_reduce(_element_list)
                elem_comp_list.extend(_elem_comp_list)
                numb_comp_list.extend(_number_list)
                elem_batch_indices.append(len(_number_list))
            elem_batch = np.asarray(elem_batch_indices)
            elem_arr = np.asarray(elem_comp_list)
            num_arr = np.asarray(numb_comp_list)

            self.batch_indices_arr.append(batch_indices_arr)
            self.elem_batch.append(elem_batch)
            self.idx.append(idx)
            self.cells.append(cells)
            self.elem_arr.append(elem_arr)
            self.num_arr.append(num_arr)
            self.pos_type.append(pos_type)
            self.pos.append(pos)
            self.fixations.append(fixations)
            self.energies.append(energies)
            self.forces.append(forces)

            # dump saved structures
            self.collect_count += 1
            if self.save_path is not None:
                if self.collect_count % self.dump_freq == 0:
                    self.flush()

        except Exception as e:
            self._exit(Exception, e, traceback.format_exc())

    def flush(self):
        """
        dump the collected structures in `self._structures` into disk as memory-mapped files.
        Returns:

        """
        try:
            if self._check_data() <= 0:
                return None

            batch_indices_arr = np.concatenate(self.batch_indices_arr)
            elem_batch = np.concatenate(self.elem_batch)
            idx = np.concatenate(self.idx)
            cells = np.concatenate(self.cells)
            elem_arr = np.concatenate(self.elem_arr)
            num_arr = np.concatenate(self.num_arr)
            pos_type = np.concatenate(self.pos_type)
            pos = np.concatenate(self.pos)
            fixations = np.concatenate(self.fixations)
            energies = np.concatenate(self.energies)
            forces = np.concatenate(self.forces)
            self.initialize()

            self._structures.append_from_array(
                batch_indices_arr,
                elem_batch,
                idx,
                cells,
                elem_arr,
                num_arr,
                pos_type,
                pos,
                fixations,
                energies,
                forces
            )

            if (self.save_path is not None) and (len(self._structures) > 0):
                self._structures.save(self.save_path, 'a')
            del self._structures
            self._structures = Structures()
            self._structures.Mode = 'A'
        except Exception as e:
            self._exit(Exception, e, traceback.format_exc())

    def __del__(self):
        if self.save_path is not None: self.flush()
        self._exit()
        gc.collect()


class PygBatchUpdater:
    """ batch updater for torch-geometric objects """

    def __init__(self):
        self.__check_old = None
        self.pygData = check_module('torch_geometric.data.batch')

    def initialize(self):
        self.__check_old = None

    def _reallocate(
            self,
            converge_check: th.Tensor,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs
    ):
        # main
        g = func_args[0]
        g_list = g.index_select(converge_check)
        g_new = self.pygData.Batch.from_data_list(g_list)
        self.__check_old = converge_check
        self.__g_old = g_new
        return (g_new,), func_kwargs, (g_new,), grad_func_kwargs

    def __call__(
            self,
            converge_check: th.Tensor,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs
    ):
        # adding a buffer
        is_new = (self.__check_old is None) or (converge_check.shape != self.__check_old.shape)
        if is_new:
            self.__check_old = converge_check
            self.__g_old = func_args[0]
            if th.all(converge_check):  # if all are unconverged. usually occurred for new ones.
                return func_args, func_kwargs, grad_func_args, grad_func_kwargs
            else:
                return self._reallocate(converge_check, func_args, func_kwargs, grad_func_args, grad_func_kwargs)

        elif th.all(th.eq(self.__check_old, converge_check)):
            return (self.__g_old,), func_kwargs, (self.__g_old,), grad_func_kwargs
        else:
            return self._reallocate(converge_check, func_args, func_kwargs, grad_func_args, grad_func_kwargs)

