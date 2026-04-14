""" Input / Output Module """
import gc
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _io.py
#  Environment: Python 3.12

import logging
import os
import re
import sys
import time
import traceback
import warnings
from typing import Optional, Dict, Callable, Any, Literal, Sequence, List

import numpy as np
import torch as th
import yaml
from torch import nn

from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.cli.print_logo import generate_display_art
from BUCToolkit.utils.setup_loggers import has_any_handler
from BUCToolkit.utils._Element_info import ATOMIC_NUMBER, ATOMIC_SYMBOL
from BUCToolkit.utils.function_utils import _BaseWrapper
from BUCToolkit.BatchStructures.StructuresIO import structures_io_dumper
from BUCToolkit.BatchStructures import Batch


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
        self.param = None
        self._has_load_data = False
        self._data_loader = None
        self.reload_config(config_file)
        self.reset_logger()

    def set_device(self, device: str | th.device) -> None:
        """ reset the device that model would train on """
        self.DEVICE = device

    def set_model_config(self, model_config: Dict[str, Any] | None = None) -> None:
        """
        Set the new configs (hyperparameters) of model.
        """
        if model_config is None: model_config = dict()
        self.MODEL_CONFIG = model_config

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
        element_list = element_tensor.tolist() if isinstance(element_tensor, th.Tensor) else list(element_tensor)
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
        self.logger.info('*' * 89)

    def logout_task_information(
            self,
            _model: nn.Module,
            mode: str,
            algo_config: Dict[str, Any] | None,
            n_samp: int | None
    ) -> None:
        """
        Logout task head information.
        Args:
            _model: nn.Module, a user-defind model to count & print parameter number.
            mode: task mode keyword.
            algo_config: Dict[str, Any], kwargs/configs of this task.
            n_samp: int, number of samples loaded in.

        Returns: None

        """
        if algo_config is None: algo_config = dict()
        __time = time.strftime("%Y%m%d_%H:%M:%S")
        para_count = sum(p.numel() for p in _model.parameters())
        self.logger.info('\n' + generate_display_art())
        self.logger.info('\n' + '*' * 89 + f'\n TIME: {__time}')
        # parse mode
        algo_info = None
        if mode == 'OPT':
            self.logger.info(' TASK: Structure Optimization (minimize) <<')
            algo_info = self.RELAXATION["ALGO"]
        elif mode == 'TS':
            self.logger.info(' TASK: Structure Optimization (TS) <<')
            algo_info = self.TRANSITION_STATE["ALGO"]
        elif mode == 'MD':
            self.logger.info(' TASK: Molecular Dynamics <<')
            algo_info = f"{self.MD["ENSEMBLE"]} Ensemble"
        elif mode == 'NEB':
            self.logger.info(' TASK: Nudged Elastic Band Transition State Search <<')
            algo_info = self.NEB.get('ALGO', 'CI-NEB')
        elif mode == 'CMD':
            self.logger.info(' TASK: Constrained Molecular Dynamics <<')
            algo_info = f"{self.MD["ENSEMBLE"]} Ensemble"
        elif mode == 'VIB':
            self.logger.info(' TASK: Vibrational Analysis (Harmonic) <<')
            algo_info = "Finite Difference" if self.VIBRATION.get('METHOD', 'Coord') == "Coord" else "Automatic Differentiation"
        elif mode == 'MC':
            self.logger.info(' TASK: Monte Carlo <<')
            algo_info = f"{self.MC["ENSEMBLE"]} Ensemble"
        elif mode == 'TRAIN':
            self.logger.info(' TASK: TRAINING & VALIDATION <<')
        elif mode == 'PREDICT':
            self.logger.info(' TASK: Predict <<')
        else:
            self.logger.info(' TASK: Unknown Mode <<')
        # mode end
        if (self.START == 0) or (self.START == 'from_scratch'):
            self.logger.info(' FROM_SCRATCH <<')
        elif (self.START == 1) or (self.START == 'resume'):
            self.logger.info(' RESUME <<')
        elif (self.START == 2) or (self.START == 'load_param'):
            self.logger.info(' LOAD_PARAMETER <<')
        self.logger.info(f' COMMENTS: {self.COMMENTS}')
        self.logger.info(f' I/O INFORMATION:')
        self.logger.info(f'\tVERBOSITY LEVEL: {self.VERBOSE}')
        if not self.REDIRECT:
            self.logger.info('\tLOG WILL OUTPUT TO STDOUT')
        else:
            output_file = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
            self.logger.info(f'\tLOG WILL OUTPUT TO {output_file}')  # type: ignore
        if (self.START == 2) or (self.START == 'load_param'):
            self.logger.info(f'\tMODEL PARAMETERS LOAD FROM: {self.LOAD_CHK_FILE_PATH}')
        elif (self.START == 1) or (self.START == 'resume'):
            self.logger.info(f'\tMODEL CHECKPOINT FILE LOAD FROM: {self.LOAD_CHK_FILE_PATH}')
        if mode == 'TRAIN': self.logger.info(f'\tCHECKPOINT FILE SAVE TO: {self.CHK_SAVE_PATH}_{self.CHK_SAVE_POSTFIX}')
        self.logger.info(f' MODEL NAME: {self.MODEL_NAME}')
        self.logger.info(f' MODEL INFORMATION:')
        self.logger.info(f'\tTOTAL PARAMETERS: {para_count}')
        if mode == 'TRAIN':
            para_count_train = sum(p.numel() for p in _model.parameters() if p.requires_grad)
            self.logger.info(f'\tTOTAL TRAINABLE PARAMETERS: {para_count_train}')
        if self.VERBOSE > 1:
            self.logger.info(f'\tHYPER-PARAMETERS:')
            for hp, hpv in self.MODEL_CONFIG.items():
                self.logger.info(f'\t\t{hp}: {hpv}')

        self.logger.info(f' TASK WILL RUN ON {self.DEVICE}')
        if mode == 'TRAIN':
            self.logger.info(f' LOSS FUNCTION: {self.loss_name}')
            if len(self.METRICS) > 0:
                self.logger.info(f" # note that `METRICS` only counts the data of last true batch instead of accumulated ones.")
                with _LoggingEnd(self.log_handler):
                    self.logger.info(f' METRICS: ')
                    for _name in self.METRICS.keys():
                        self.logger.info(f'{_name}  ')
                self.logger.info('')
            else:
                self.logger.info(' METRICS: None')
            self.logger.info(f' OPTIMIZER INFORMATION:')
            __opt_repr = re.split(
                r'\(\n|\)$|\s{2,}|\n',
                repr(self.OPTIMIZER(list(th.zeros(1)), **self.OPTIM_CONFIG))
            )
            self.logger.info(f'\tOPTIMIZER: {__opt_repr[0]}')
            if (self.VERBOSE > 1) and hasattr(self, '_layerwise_opt_configs'):
                if self._layerwise_opt_configs is not None:
                    _layer_grp_name = list(self._layerwise_opt_configs.keys())
                else:
                    _layer_grp_name = None
                kk = 0
                for __partt in __opt_repr[1:-2]:
                    if __partt.startswith('Parameter Group'):
                        self.logger.info(f'\t{__partt}')
                        if self._layerwise_opt_configs is not None:
                            self.logger.info(
                                f'\t\tcontaining layers: {
                                _layer_grp_name[kk].pattern if kk < len(_layer_grp_name) else "all remaining layers"
                                }'
                            )
                            kk += 1
                    else:
                        self.logger.info(f'\t\t{__partt}')
                del kk, _layer_grp_name
            if self.LR_SCHEDULER is not None:
                self.logger.info(f'\tLR_SCHEDULER: {str(self.LR_SCHEDULER)}')
                self.logger.info(f'\tLR_SCHEDULER CONFIG:')
                for hp, hpv in self.LR_SCHEDULER_CONFIG.items(): self.logger.info(f'\t\t{hp}: {hpv}')
            else:
                self.logger.info('\tLR_SCHEDULER: None')
            if self.GRAD_CLIP:
                self.logger.info(f'\tGRAD_CLIP: True')
                self.logger.info(f'\tGRAD_CLIP_MAX_NORM: {self.GRAD_CLIP_MAX_NORM:<.2f}')
                if len(self.GRAD_CLIP_CONFIG) > 0: self.logger.info(f'\tGRAD_CLIP_CONFIG: {self.GRAD_CLIP_CONFIG}')
            else:
                self.logger.info(f'\tGRAD_CLIP: False')
            if self.EMA:
                self.logger.info(f'\tEXPONENTIAL MOVING AVERAGE (EMA): True')
                self.logger.info(f'\tEMA_DECAY: {self.EMA_DECAY:<.5f}')
                self.logger.info(
                    f'\tNOTE: `best_checkpoint` will save with EMA parameters, while `checkpoint` and `stop_checkpoint` will not.'
                )
            else:
                self.logger.info(f'\tEXPONENTIAL MOVING AVERAGE (EMA): False')
            self.logger.info(f' ITERATION INFORMATION:')
            self.logger.info(f'\tEPOCH: {self.EPOCH}\n\tBATCH SIZE: {self.BATCH_SIZE}\n\tVALID BATCH SIZE: {self.VAL_BATCH_SIZE}' +
                             f'\n\tGRADIENT ACCUMULATION STEPS: {self.ACCUMULATE_STEP}\n\tEVAL PER {self.VAL_PER_STEP} STEPS\n' +
                             '*' * 89 + '\n' + 'ENTERING MAIN LOOP...')
        else:
            if self.SAVE_PREDICTIONS:
                self.logger.info(f' PREDICTIONS WILL SAVE TO {self.PREDICTIONS_SAVE_FILE}')
            else:
                self.logger.info(f' PREDICTIONS WILL SAVE IN MEMORY AND RETURN AS A VARIABLE.')
            self.logger.info(f' ITERATION INFORMATION:')
            # ALGORITHM PRINT
            if algo_info is not None:
                self.logger.info(f'\tALGORITHM: {algo_info}')
            for _algo_conf_name, _algo_conf in algo_config.items():
                self.logger.info(f'\t{_algo_conf_name}: {_algo_conf}')
            n_samp = 'Unknown' if n_samp is None else n_samp
            self.logger.info(f'\tBATCH SIZE: {self.BATCH_SIZE}' +
                             f'\n\tTOTAL SAMPLE NUMBER: {n_samp}\n' +
                             '*' * 89 + '\n' + 'ENTERING MAIN LOOP...')

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
        if not has_any_handler(self.logger): self.logger.addHandler(self.log_handler)

    def reload_config(self, config_file_path: str| None = None) -> None:
        """
        Reload the yaml configs file.
        """
        # load config file
        config_file_path = self.config_file if config_file_path is None else config_file_path
        with open(config_file_path, 'r', encoding='utf-8') as f:
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
        self.BATCH_SIZE: int = self.config.get('BATCH_SIZE', 1)

        # model info
        self.MODEL_NAME: str = self.config.get('MODEL_NAME', 'Untitled')
        if not isinstance(self.MODEL_NAME, str): raise TypeError('MODEL_NAME must be a str.')
        self.MODEL_CONFIG = self.config.get('MODEL_CONFIG', dict())
        if not isinstance(self.MODEL_CONFIG, Dict): raise ValueError('MODEL_CONFIG must be a dictionary.')

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
            self.dumper = DumpStructures(self.PREDICTIONS_SAVE_FILE)
        else:
            self.dumper = DumpStructures(None, )
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

        # If Train
        self.TRAIN = self.config.get('TRAIN', None)

        # If Structure opt.
        self.RELAXATION = self.config.get('RELAXATION', None)
        self.TRANSITION_STATE = self.config.get('TRANSITION_STATE', None)

        # If Molecular Dynamics
        self.MD = self.config.get('MD', None)

        # NEB Transition State Search
        self.NEB = self.config.get('NEB', None)

        # If Vibration Calc.
        self.VIBRATION = self.config.get('VIBRATION', None)

        # If Monte Carlo
        self.MC = self.config.get('MC', None)

    @property
    def PREDICTIONS_SAVE_FILE(self):
        return self._PREDICTIONS_SAVE_FILE

    @PREDICTIONS_SAVE_FILE.setter
    def PREDICTIONS_SAVE_FILE(self, value):
        if not self.SAVE_PREDICTIONS:
            self.logger.warning(
                f'WARNING: You are setting a new predictions save file, while `SAVE_PREDICTIONS` is still False.\n'
                f'Hence, NOTHING WILL HAPPEN. BYE!'
            )
            return
        self._PREDICTIONS_SAVE_FILE = value
        self.dumper = DumpStructures(self.PREDICTIONS_SAVE_FILE)

def compare_tensors(X1: th.Tensor, X2: th.Tensor):
    """Compare two tensors. Return True if they are the same, False otherwise."""
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


class _Model_Wrapper_pyg(_BaseWrapper):

    __slots__ = ('_model', 'forces', 'X', )

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
        super().__init__(model)
        #if check_module('torch_geometric') is None:
        #    ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')
        pass

    def Energy(self, X, graph: Batch):
        self.X = X
        graph.pos = self.X.reshape(-1,3).contiguous()
        y = self._model(graph)
        energy = y['energy']
        self.forces = y['forces']
        return energy

    def Grad(self, X, graph: Batch):
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


class _Model_Wrapper_pyg_only_X(_BaseWrapper):
    def __init__(self, model , graph: Batch) -> None:
        """
        A format transformer for converting Tensor X into PygData.pos
        Wrap the model(graph, ...) into f(X)

        Args:
            model: An instantiate nn.Module

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Grad: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].
        """
        super().__init__(model)
        self.graph = graph
        #if check_module('torch_geometric') is None:
        #    ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')
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
        if (self.X is None) or (not compare_tensors(X, self.X)):
            self.forces = None
        if self.forces is None:
            self.X = X
            self.graph.pos = self.X.reshape(-1, 3)
            return - ((self._model(self.graph))['forces']).reshape(origin_shape)
        else:
            force = self.forces
            self.forces = None
            return - force.reshape(origin_shape).contiguous()


class _Model_Wrapper_dgl(_BaseWrapper):
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
        super().__init__(model)
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
        if (self.X is None) or (not compare_tensors(X, self.X)):
            self.forces = None
        if self.forces is None:
            self.X = X
            graph.nodes['atom'].data['pos'] = self.X.squeeze(0)
            return - ((self._model(graph))['forces']).unsqueeze(0)
        else:
            force = self.forces
            self.forces = None
            return - force.unsqueeze(0)


class _Model_Wrapper_regularBatch_pyg(_BaseWrapper):
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
        super().__init__(model)
        _pyg = check_module('torch_geometric.data')
        if _pyg is not None:
            import torch_geometric.data as _pyg
            self.pygBatch = _pyg.Batch
        else:
            #ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')
            self.pygBatch = Batch

    def Energy(self, X: th.Tensor, graph: Batch):
        self.X = X.flatten(0, 1)  # convert X: (n_batch, n_atom, n_dim) into X': (n_batch * n_atom, 3)
        batch_size = X.size(0)
        if graph.batch_size == 1:
            graph = self.pygBatch.from_data_list([graph] * batch_size)
        graph.pos = self.X
        y = self._model(graph)  # (n_batch, )
        energy = y['energy']
        self.forces = y['forces']
        return energy

    def Grad(self, X, graph: Batch):
        if (self.X is None) or (not compare_tensors(X, self.X)):
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
    Use the `ArrayDumper` as the backend, which dump structure information into binary file.
    """
    def __init__(self, path:str|None=None):
        """
        Args:
            path: the path to dump structures. If None, structures will always store in memory.
        """
        self.dumper = structures_io_dumper(path, 'x', )
        self._step = self._first_step
        self._has_initialized = False

    def initialize(self):
        self.dumper.initialize()

    @staticmethod
    def _exit(errt = None, errv = None, tr = None):
        if errt is not None:
            raise errt(f'{errv}, {tr}')

    @staticmethod
    def _reformat(
            batch_indices: List,
            idx: List[str],
            elements: th.Tensor,
            cells: th.Tensor | np.ndarray,
            pos: th.Tensor,
            fixations: th.Tensor,
            energies: th.Tensor,
            forces: th.Tensor,
    ):
        """
        Reformat input into dumping format.
        Args:
            batch_indices:
            idx:
            elements:
            cells:
            pos:
            fixations:
            energies:
            forces:

        Returns:
            batch_indices, idx, cells, elements, pos, fixations, energies, forces
        """
        # Postprocessing & save TODO: reformat it in future to apply to all functions.
        n_batch = len(batch_indices)
        batch_indices = np.array(batch_indices, dtype=np.int64)
        if cells.shape != (n_batch, 3, 3):
            raise ValueError(f'`cells` is expected to have 3 dimensions (n_batch, 3, 3), but got {cells.shape}.')
        cells = cells.numpy(force=True).astype(np.float32) if isinstance(cells, th.Tensor) else np.asarray(cells, dtype=np.float32)
        if len(idx) != n_batch:
            raise ValueError(f'The number of `idx` is expected to be batch size {n_batch}, but got {len(idx)}.')
        idx = np.array(idx, dtype='<U128')
        if elements.ndim == 2:
            elements = elements.squeeze(0)
        elif elements.ndim != 1:
            raise ValueError(f'`elements` should be a 1D array, but got {elements.shape}')
        elements = elements.numpy(force=True).astype(np.int64) if isinstance(elements, th.Tensor) else np.asarray(elements, dtype=np.int64)
        # pos_type = np.array(pos_type)
        if pos.ndim == 3:
            pos = pos.squeeze(0)
        elif pos.ndim != 2:
            raise ValueError(f'`pos` should be a 2D array, but got {pos.shape}')
        pos = pos.numpy(force=True).astype(np.float32) if isinstance(pos, th.Tensor) else np.asarray(pos, dtype=np.float32)
        if forces.ndim == 3:
            forces = forces.squeeze(0)
        if forces.shape != pos.shape:
            raise ValueError(f'`forces` should have the same shape as `pos`, but got {forces.shape} rather than {pos.shape}.')
        forces = forces.numpy(force=True).astype(np.float32) if isinstance(forces, th.Tensor) else np.asarray(forces, dtype=np.float32)
        if energies.ndim == 2:
            energies = energies.squeeze(0)
        elif energies.ndim != 1:
            raise ValueError(f'`energies` should be a 1D array, but got {energies.shape}')
        energies = energies.numpy(force=True)
        if fixations.ndim == 3:
            fixations = fixations.squeeze(0)
        if fixations.shape != pos.shape:
            raise ValueError(f'`fixations` should have the same shape as `pos`, but got {fixations.shape} rather than {pos.shape}.')
        fixations = fixations.numpy(force=True).astype(np.float32) if isinstance(fixations, th.Tensor) else np.asarray(fixations, dtype=np.float32)

        return batch_indices, idx, cells, elements, pos, fixations, energies, forces

    def _first_step(
            self,
            batch_indices: List,
            idx: List[str],
            elements: th.Tensor,
            cells: th.Tensor | np.ndarray,
            pos: th.Tensor,
            fixations: th.Tensor,
            energies: th.Tensor,
            forces: th.Tensor,
    ):
        """
        First step to initialize
        Args:
            batch_indices:
            idx:
            elements:
            cells:
            pos:
            fixations:
            energies:
            forces:

        Returns:

        """
        if self._has_initialized:
            self.initialize()
        batch_indices, idx, cells, elements, pos, fixations, energies, forces = self._reformat(
            batch_indices,
            idx,
            elements,
            cells,
            pos,
            fixations,
            energies,
            forces,
        )
        self.dumper.start_from_arrays(1, batch_indices, idx, cells, elements, pos, fixations, energies, forces)
        self.dumper.step(
            batch_indices, idx, cells, elements, pos, fixations, energies, forces
        )
        self._step = self._continue_step

    def _continue_step(
            self,
            batch_indices: List,
            idx: List[str],
            elements: th.Tensor,
            cells: th.Tensor | np.ndarray,
            pos: th.Tensor,
            fixations: th.Tensor,
            energies: th.Tensor,
            forces: th.Tensor,
    ):
        batch_indices, idx, cells, elements, pos, fixations, energies, forces = self._reformat(
            batch_indices,
            idx,
            elements,
            cells,
            pos,
            fixations,
            energies,
            forces,
        )
        self.dumper.start_from_arrays(1, batch_indices, idx, cells, elements, pos, fixations, energies, forces)
        self.dumper.step(
            batch_indices, idx, cells, elements, pos, fixations, energies, forces
        )

    def collect(
            self,
            batch_indices: List,
            idx: List[str],
            elements: th.Tensor,
            cells: th.Tensor | np.ndarray,
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
            pos: th.Tensor,
            fixations: th.Tensor,
            energies: th.Tensor,
            forces: th.Tensor,

        Returns: None

        """
        try:
            # Postprocessing & save TODO: reformat it in future to apply to all functions.
            self._step(
                batch_indices,
                idx,
                elements,
                cells,
                pos,
                fixations,
                energies,
                forces,
            )

        except Exception as e:
            warnings.warn(f'Failed to collect structures due to ERROR: {e}')

    def flush(self):
        """
        dump the collected structures in `self._structures` into disk as memory-mapped files.
        Returns:

        """
        try:
            self.dumper.flush()
        except Exception as e:
            warnings.warn(f'Failed to flush structures due to ERROR: {e}')

    def close(self):
        self.dumper.close()


class PygBatchUpdater:
    """
    batch updater for torch-geometric objects.
    It can be directly called after initialization.
    Examples:
        ```
        updater = PygBatchUpdater()
        updater.initialize()
        optimizer = CG(...)
        optimizer.set_batch_updater(updater)
        optimizer.run(...)
        ```
    One can use `self.initialize()` to reset this updater.
    """

    def __init__(self):
        self.__check_old = None
        _pyg = check_module('torch_geometric.data.batch')
        if _pyg is None:
            self.pygData = Batch
        else:
            self.pygData = _pyg.Batch

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
        g_new = self.pygData.from_data_list(g_list)
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
            if th.all(converge_check):  # if all are unconverged. usually occurred for new ones.
                self.__check_old = None
                return func_args, func_kwargs, grad_func_args, grad_func_kwargs
            else:
                self.__check_old = converge_check
                self.__g_old = func_args[0]
                return self._reallocate(converge_check, func_args, func_kwargs, grad_func_args, grad_func_kwargs)

        elif th.all(th.eq(self.__check_old, converge_check)):
            return (self.__g_old,), func_kwargs, (self.__g_old,), grad_func_kwargs
        else:
            return self._reallocate(converge_check, func_args, func_kwargs, grad_func_args, grad_func_kwargs)
