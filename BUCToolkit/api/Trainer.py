""" Training Methods """

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Trainer.py
#  Environment: Python 3.12

import copy
import logging
import os
import re
import time
from math import ceil
from typing import Dict, Tuple, Literal, List, Any, Callable, Optional
import traceback
from inspect import isclass

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import (StepLR, ExponentialLR, ChainedScheduler, ConstantLR, CyclicLR, MultiStepLR,
                                      LambdaLR, LinearLR, CosineAnnealingWarmRestarts, CosineAnnealingLR)

from .Losses import Energy_Force_Loss, Energy_Loss
from .Metrics import E_MAE, E_R2, F_MAE, F_MaxE, _r2_score, _rmse
from .ModelOptims import FIRELikeOptimizer, LangevinOptimizer
from ._io import _CONFIGS, _LoggingEnd, ExpMovingAverage


class Trainer(_CONFIGS):
    r"""
    The model trainer class.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: the path of input file

    Methods:
        train(model: torch.nn.Module), run model training.
        set_device(device: str | torch.device), manually set model and data running device.
        set_loss_fn(loss_fn: Any, loss_config: Optional[Dict] = None), manually set loss function which can be customized.
        set_metrics(metrics_fn: Dict[str, Callable], metrics_fn_config: Dict[str, Dict] | None = None), manually set metrics functions which can be customized.
        set_model_config(model_config: Dict[str, Any] | None = None), manually (re)set the configs (hyperparameters) of model.
        set_lr_scheduler(lr_scheduler: th.optim.lr_scheduler.LRScheduler, lr_scheduler_config: Optional[Dict[str, Any]] = None), set the learning rate scheduler.
        set_model_param(model_state_dict: Dict, is_strict: bool = True, is_assign: bool = False), manually set model parameters by giving model state dict.

    """

    def __init__(self, config_file: str, *args) -> None:
        super().__init__(config_file)

        self.config_file = config_file
        self.reload_config(config_file)
        if self.VERBOSE: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None
        self._layerwise_opt_configs = None

        # CONSTANT of training information
        self._OPTIM_DICT = {
            'Adam': th.optim.Adam, 'SGD': th.optim.SGD, 'AdamW': th.optim.AdamW, 'Adadelta': th.optim.Adadelta,
            'Adagrad': th.optim.Adagrad, 'ASGD': th.optim.ASGD, 'Adamax': th.optim.Adamax,
            'FIRE': FIRELikeOptimizer, 'Langevin': LangevinOptimizer,
            'custom': None
        }
        self._LR_SCHEDULER_DICT = {
            'StepLR': StepLR, 'ExponentialLR': ExponentialLR, 'ChainedScheduler': ChainedScheduler,
            'ConstantLR': ConstantLR, 'LambdaLR': LambdaLR, 'LinearLR': LinearLR, 'CosineAnnealingLR': CosineAnnealingLR,
            'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts, 'CyclicLR': CyclicLR, 'MultiStepLR': MultiStepLR,
            'None': None, 'custom': None
        }
        self._LOSS_DICT = {
            'MSE': nn.MSELoss, 'MAE': nn.L1Loss, 'Hubber': nn.HuberLoss, 'CrossEntropy': nn.CrossEntropyLoss,
            'Energy_Force_Loss': Energy_Force_Loss, 'Energy_Loss': Energy_Loss, 'custom': None
        }
        self._METRICS_DICT = {
            'MSE': F.mse_loss, 'MAE': F.l1_loss, 'R2': _r2_score, 'RMSE': _rmse,
            'E_MAE': E_MAE, 'E_R2': E_R2, 'F_MAE': F_MAE, 'F_MaxE': F_MaxE, 'custom': None
        }

        # Section: parse train parameters
        trn_config = self.config.get('TRAIN', None)
        if trn_config is None:
            self.logger.critical('** ERROR: Training task is required without setting args `TRAIN`. Task aborted. BYE!!! **')
            raise RuntimeError('** ERROR: Training task is required without setting args `TRAIN`. Task aborted. BYE!!! **')

        # io
        self.SAVE_CHK = bool(trn_config.get('SAVE_CHK', False))
        if not self.SAVE_CHK:
            self.logger.warning(
                f"WARNING: The model is training but NO CHECKPOINT FILE WILL BE SAVED. "
                f"I HOPE YOU KNOW WHAT YOU ARE DOING!!!"
            )
        self.CHK_SAVE_PATH = self.config.get('CHK_SAVE_PATH', './')
        self.CHK_SAVE_POSTFIX = self.config.get('CHK_SAVE_POSTFIX', '')
        if self.CHK_SAVE_POSTFIX != '': self.CHK_SAVE_POSTFIX = '_' + self.CHK_SAVE_POSTFIX  # use '_' to delimit chk name.
        if not isinstance(self.CHK_SAVE_PATH, str): raise TypeError('CHK_SAVE_PATH must be a str.')

        # epoches & validation set
        self.EPOCH: int = int(trn_config.get('EPOCH', 0))
        self.ACCUMULATE_STEP = trn_config.get('ACCUMULATE_STEP', 1)
        if not (isinstance(self.ACCUMULATE_STEP, int) and self.ACCUMULATE_STEP > 0):
            raise TypeError(f'ACCUMULATE_STEP must be a positive integer, but got {self.ACCUMULATE_STEP}')
        self.VAL_PER_STEP: int = int(trn_config.get('VAL_PER_STEP', 10))
        self.VAL_BATCH_SIZE: int = int(trn_config.get('VAL_BATCH_SIZE', self.BATCH_SIZE))
        self.VAL_IF_TRN_LOSS_BELOW: float = float(trn_config.get('VAL_IF_TRN_LOSS_BELOW', th.inf))

        # optim info
        optim_name = trn_config.get('OPTIM', None)
        if optim_name is not None:  # lazy to check. One may set/reset it by `self.set_optimizer`
           self.OPTIMIZER = self._OPTIM_DICT.get(optim_name, None)
        else:
           self.OPTIMIZER = None
        self.OPTIM_CONFIG = trn_config.get('OPTIM_CONFIG', dict())
        if not isinstance(self.OPTIM_CONFIG, Dict): raise ValueError('OPTIM_CONFIG must be a dictionary.')
        layerwise_optim_conf = trn_config.get('LAYERWISE_OPTIM_CONFIG', None)
        if isinstance(layerwise_optim_conf, dict):
            self.set_layerwise_optim_config(layerwise_optim_conf)
        elif layerwise_optim_conf is not None:
            self.logger.warning(f"Invalid format of `LAYERWISE_OPTIM_CONFIG`: {layerwise_optim_conf}, thus ignoring it.")

        self.GRAD_CLIP: bool = trn_config.get('GRAD_CLIP', False)
        self.GRAD_CLIP_MAX_NORM: float = float(trn_config.get('GRAD_CLIP_MAX_NORM', 100))
        self.GRAD_CLIP_CONFIG = trn_config.get('GRAD_CLIP_CONFIG', dict())
        if not isinstance(self.GRAD_CLIP, bool): raise TypeError('GRAD_CLIP must be a boolean.')
        if not isinstance(self.GRAD_CLIP_CONFIG, Dict): raise ValueError('GRAD_CLIP_CONFIG must be a dictionary.')

        lr_scheduler_name = trn_config.get('LR_SCHEDULER', None)
        if lr_scheduler_name is not None:
            self.LR_SCHEDULER = self._LR_SCHEDULER_DICT.get(lr_scheduler_name, None)
            if lr_scheduler_name not in self._LR_SCHEDULER_DICT:
                self.logger.warning(f"Unknown lr scheduler name: {lr_scheduler_name}. Learning rate scheduler will be ignored.")
        else:
            self.LR_SCHEDULER = None
        self.LR_SCHEDULER_CONFIG = trn_config.get('LR_SCHEDULER_CONFIG', dict())
        if not isinstance(self.LR_SCHEDULER_CONFIG, Dict): raise TypeError('LR_SCHEDULER_CONFIG must be a dict.')

        self.EMA = trn_config.get('EMA', False)  # exponential moving average strategy. best_checkpoint saves using ema param, and others do not.
        self.EMA_DECAY = float(trn_config.get('EMA_DECAY', 0.999))

        # loss & criterion info
        self.loss_name = trn_config.get('LOSS', None)
        if self.loss_name is not None:
           self.LOSS = self._LOSS_DICT.get(self.loss_name, None)
        else:
           self.LOSS = None
        self.LOSS_CONFIG = trn_config.get('LOSS_CONFIG', dict())
        if not isinstance(self.LOSS_CONFIG, Dict): raise TypeError('LOSS_CONFIG must be a dictionary.')
        _metric_name = trn_config.get('METRICS', tuple())
        self.METRICS_CONFIG = trn_config.get('METRICS_CONFIG', dict())
        if not isinstance(_metric_name, (List, Tuple)): raise TypeError(f'METRICS must be a sequence, but occurred {type(_metric_name)}')
        if not isinstance(self.METRICS_CONFIG, Dict): raise TypeError(f'METRICS_CONFIG must be a dict, but occurred {type(self.METRICS_CONFIG)}')
        self.METRICS = dict()
        for __metric in _metric_name:
           if __metric not in self._METRICS_DICT:
               self.METRICS[__metric] = None
               self.METRICS_CONFIG[__metric] = dict()
           elif __metric not in self.METRICS_CONFIG:
               self.METRICS[__metric] = self._METRICS_DICT[__metric]
               self.METRICS_CONFIG[__metric] = dict()
           else:
               self.METRICS[__metric] = self._METRICS_DICT[__metric]

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

    def set_lr_scheduler(self, lr_scheduler, lr_scheduler_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the lr_scheduler that inherit from torch.optim.lr_scheduler.LRScheduler
        """
        self.LR_SCHEDULER = lr_scheduler
        if lr_scheduler_config is not None:
            self.LR_SCHEDULER_CONFIG = lr_scheduler_config

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

    def set_layerwise_optim_config(self, layer_config_dict: Dict[str, Dict[str, Any]] | None = None):
        """
        The optimizer configs of layers in `layer_config_dict.keys()` would set to the corresponding values,
        and other unspecified layers would use the config of `OPTIM_CONFIG` in the input file.
        The parameters of the layer which lr is set to `None` would be fixed during training without calculating gradients.
        The name of layers can be specified by regular expressions e.g.,
         {"fc1.*": {"lr": 1e-4, "weight_decay": 1e-4}, "fc2.[a-zA-Z]+Norm.*": {"lr": None}}

        Args:
            layer_config_dict: dict of named layers' learning config: {layer name: {'lr': ...}}.

        Returns: None

        """
        if layer_config_dict is not None:
            if not isinstance(layer_config_dict, Dict):
                self.logger.exception(f'ERROR: Expected `layer_lr_dict` is a Dict, but got {type(layer_config_dict)}')
                raise TypeError(f'Expected `layer_lr_dict` is a Dict, but got {type(layer_config_dict)}')
            # fixed_parameter_dict: {name: name length before glob '*'}
            self._layerwise_opt_configs = {re.compile(nam): val for nam, val in layer_config_dict.items()}
        else:
            self._layerwise_opt_configs = None

    def train(self, model):
        r"""
        Start Training.

        Herein the input model must be an `uninstantiated` nn.Module class.
        """
        # check logger
        if not self.logger.hasHandlers(): self.logger.addHandler(self.log_handler)
        # check vars
        if not isclass(model):
            raise TypeError('`model` must be a class. You may not instantiate it.')
        _model: nn.Module = model(**self.MODEL_CONFIG)
        if self.START != 'from_scratch' and self.START != 0:
            chk_data = th.load(self.LOAD_CHK_FILE_PATH, weights_only=True)
            if self.param is None:
                _model.load_state_dict(chk_data['model_state_dict'], strict=self.STRICT_LOAD)
            else:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
            if self.START == 'resume' or self.START == 1:
                epoch_now = chk_data['epoch']
                val_loss_old = chk_data['val_loss'] if isinstance(chk_data['val_loss'], float) else chk_data['val_loss'][0]
            elif self.START == 'param_only' or self.START == 2:
                epoch_now = 0
                val_loss_old = th.inf
            else:
                self.logger.exception(f'ERROR: Invalid `START` value {self.START}. It should be "from_scratch" / 0 , "resume" / 1 , "param_only" / 2')
                raise ValueError(f'Invalid `START` value {self.START}. It should be "from_scratch" / 0 , "resume" / 1 , "param_only" / 2')
        elif self.START == 'from_scratch' or self.START == 0:
            epoch_now = 0
            chk_data = None
            if self.param is not None:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
            val_loss_old = th.inf
        else:
            self.logger.exception(f'ERROR: Invalid parameter of `START`: {self.START}, it must be 0, 1, or 2.')
            raise ValueError(f'Invalid parameter of `START`: {self.START}, it must be 0, 1, or 2.')

        # model vars
        _model = _model.to(self.DEVICE)
        _model.requires_grad_()
        # loss & eval vars
        if self.LOSS is None:
            self.logger.exception('ERROR: the loss function is None. Please set the loss function.')
            raise RuntimeError('the loss function is None. Please set the loss function.')
        else:
            LOSS = self.LOSS(**self.LOSS_CONFIG)
            LOSS = LOSS.to(self.DEVICE)
        if None in self.METRICS.values():
            self.logger.exception('ERROR: Occurred an unknown metric function. Please use self.set_metrics to specify it.')
            raise RuntimeError('Occurred an unknown metric function. Please use self.set_metrics to specify it.')

        # check layer-wise optimizer config
        if self._layerwise_opt_configs is not None:
            # classifying
            group_conf_list = list() # List[Dict[opt config]]
            group_id_set = list()  # List[Set[layer params id]], has the same order as `group_conf_list`.
            rest_group_ids = set(id(_) for _ in _model.parameters())
            fixed_param_ids = set()
            for parttern, conf in self._layerwise_opt_configs.items():
                _group_id_set = set()
                for para_name, para in _model.named_parameters():
                    q_name_len = re.match(parttern, para_name)
                    if q_name_len is not None:
                        if conf['lr'] is None:
                            para.requires_grad_(False)  # default self.is_invert_fix_param = False, that to turn off the grad.
                            fixed_param_ids.add(id(para))
                        else:
                            _group_id_set.add(id(para))
                group_id_set.append(_group_id_set)
                group_conf_list.append(conf)
                rest_group_ids.difference_update(_group_id_set)
            rest_group_ids.difference_update(fixed_param_ids)
            del para_name, para, conf, parttern, q_name_len, _group_id_set, fixed_param_ids
            # arrange groups
            parameter_iterator = list()
            for ii, id_set in enumerate(group_id_set):
                par_conf = {'params': [_ for _ in _model.parameters() if id(_) in id_set]}
                par_conf.update(group_conf_list[ii])
                parameter_iterator.append(par_conf)
            par_conf = {'params': filter(lambda p: id(p) in rest_group_ids, _model.parameters())}
            parameter_iterator.append(par_conf)
            del ii, id_set, par_conf
        else:
            parameter_iterator = _model.parameters()

        # optim vars
        if self.OPTIMIZER is None:
            self.logger.exception('ERROR: the optimizer is None. Please set the optimizer.')
            raise RuntimeError('the optimizer is None. Please set the optimizer.')
        else:
            OPTIMIZER = self.OPTIMIZER(parameter_iterator, **self.OPTIM_CONFIG)
            if self.START == 'resume' or self.START == 1:
                OPTIMIZER.load_state_dict(chk_data['optimizer_state_dict'])

        if self.LR_SCHEDULER is not None:
            scheduler = self.LR_SCHEDULER(OPTIMIZER, last_epoch=(epoch_now - 1), **self.LR_SCHEDULER_CONFIG)
            if self.START == 'resume' or self.START == 1:
                scheduler.load_state_dict(chk_data['lr_scheduler_state_dict'])
        else:
            scheduler = None

        if self.EMA:
            ema = ExpMovingAverage(_model, self.EMA_DECAY)
        else:
            ema = None

        # preprocessing data # TODO
        if self._data_loader is None:
            self.logger.exception('ERROR: Please Set the DataLoader.')
            raise RuntimeError('Please Set the DataLoader.')
        if not self._has_load_data:
            self.logger.exception('ERROR: Please Set the Training Data and Validation Data.')
            raise RuntimeError('Please Set the Training Data and Validation Data')

        # initialize
        n_trn_samp = len(self.TRAIN_DATA['data'])  # sample number
        n_val_samp = len(self.VALID_DATA['data'])
        if n_val_samp < 1:
            self.logger.warning(
                'WARNING: Validation sample is empty, so all validations would be skipped. I HOPE YOU KNOW WHAT YOU ARE DOING!'
            )
            self.VAL_IF_TRN_LOSS_BELOW = th.nan  # set it to nan to avoid validation.
        n_batch = ceil(n_trn_samp / self.BATCH_SIZE)  # total batch number per epoch
        history: Dict[Literal['train_loss', 'val_loss'], List[float]] = {'train_loss': list(), 'val_loss': list()}
        OPTIMIZER.zero_grad()
        i = epoch_now
        if not os.path.isdir(self.CHK_SAVE_PATH):
            os.makedirs(self.CHK_SAVE_PATH, )
        if not os.path.isdir(self.CHK_SAVE_PATH):
            os.makedirs(self.CHK_SAVE_PATH, )

        try:
            # I/O
            if self.VERBOSE > 0:
                self.logout_task_information(_model, 'TRAIN', None, n_trn_samp + n_val_samp)

            # MAIN LOOP
            if self.DEBUG_MODE:
                th.autograd.set_detect_anomaly(True)
                para_arr_old_list = [0. for _ in _model.named_parameters()]
            time_tol = time.perf_counter()
            __loss = th.inf
            num_update = 0  # number of parameter updating
            _can_valid = False  # To avoid stacking, ensuring that validation is always occurred after param. update.
            nan_count = 0  # number of nan occurred.
            n_err = 0  # error number during training.
            for i in range(epoch_now, self.EPOCH):
                time_ep = time.perf_counter()
                real_n_samp = 0
                # load the training set
                trn_set = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, **self._data_loader_configs)  # TODO

                self.logger.info('-' * 60 + f'EPOCH {i + 1}' + '-' * 60)
                # Training
                num_step = 1
                accum_loss = 0.  # the accumulation of loss within gradient accumulation steps.
                accum_step = 0   # the actual steps that accumulated. normally `accum_step` == `self.ACCUMULATE_STEP`, except the last
                time_gp = time.perf_counter()
                for batch_data, batch_label in trn_set:
                    try:
                        _model.train()
                        # to avoid get an empty batch
                        if not isinstance(batch_data, (th.Tensor, )):
                            len_data = batch_data.batch_size
                        else:
                            len_data = len(batch_data)
                        if len_data <= 0:
                            if self.VERBOSE: self.logger.info(f'An empty batch occurred in step {num_step}. Skipped.')
                            continue
                        # batch device
                        batch_data = batch_data.to(self.DEVICE)
                        #batch_label = batch_label.to(self.DEVICE)

                        # pred & loss, pred must be a Dict.
                        pred_y = _model(batch_data)
                        # check nan
                        if self.CHECK_NAN:
                            for key in pred_y.keys():
                                if isinstance(pred_y[key], th.Tensor):
                                    is_nan = th.isnan(pred_y[key]) + th.isinf(pred_y[key])
                                elif isinstance(pred_y[key], list):  # for ensemble wrapper model
                                    is_nan = sum([th.isnan(_p) + th.isinf(_p) for _p in pred_y[key]])
                                else:
                                    break
                                if th.any(is_nan):
                                    nan_count += 1
                                    pred_y[key] = th.where(th.isnan(pred_y[key]), 0., pred_y[key])
                                    self.logger.warning(f'NaN occurred in model output, and has been set to 0. Total NaN number: {nan_count}')
                                    if nan_count > 100:
                                        raise RuntimeError(f'Too many NaNs occurred in the model output (>= 100).')
                                    if self.DEBUG_MODE: self.logger.warning(f'batch_data:\n {batch_data}\n\nlabels:\n {batch_label}')
                        # backward
                        raw_loss:th.Tensor = LOSS(pred_y, batch_label)  # loss before grad. accum.
                        loss = raw_loss / self.ACCUMULATE_STEP
                        loss.backward()
                        accum_loss += raw_loss.item()
                        accum_step += 1

                        _is_update_params = ((num_step % self.ACCUMULATE_STEP == 0) or (num_step == n_batch)) # reached the accum. step or last step
                        # Debug Mode -----------------------------------------------------------------------------------------------------------
                        if self.DEBUG_MODE and _is_update_params:
                            i__ = 0
                            para_arr_old_temp = list()
                            self.logger.info(f'\n{"Debug"*24}\nDEBUG IN STEP NUMBER: {num_step}\n{"Debug"*24}')
                            for para_name, para in _model.named_parameters():
                                para_arr = para.numpy(force=True)
                                para_arr_old_temp.append(para_arr)
                                self.logger.info('PARAMETER  %s : %.3e ~ %.3e' % (para_name, np.min(para_arr), np.max(para_arr)))
                                if para.grad is not None:
                                    para_grad_arr = para.grad.numpy(force=True)
                                    self.logger.info('PARA_GRAD  %s : %.3e ~ %.3e' % (para_name, np.min(para_grad_arr), np.max(para_grad_arr)))
                                else:
                                    self.logger.warning(f'PARAMETER `{para_name}` does not have grad. <<<')
                                self.logger.info(f'PARAMETER RMSD: {np.sqrt(np.mean((para_arr - para_arr_old_list[i__]) ** 2)):<4.3e}')
                                i__ += 1
                            para_arr_old_list = para_arr_old_temp
                            if loss != loss:
                                self.logger.exception(f'ERROR: NaN occurred in loss.\n\nModel outputs:\n{pred_y}\n\nTraining labels:\n{batch_label}')
                                raise Exception('ERROR: NaN occurred in loss.\n\nModel outputs:\n%s\n\nTraining labels:\n%s' % (pred_y, batch_label))
                        # Debug Mode END-----------------------------------------------------------------------------------------------------------

                        # update & metrics & output
                        real_n_samp += len_data  # sample number count
                        if _is_update_params:
                            # grad clip
                            if self.GRAD_CLIP:
                                nn.utils.clip_grad_norm_(
                                    parameters=parameter_iterator,
                                    max_norm=self.GRAD_CLIP_MAX_NORM,
                                    **self.GRAD_CLIP_CONFIG
                                )
                            # update
                            OPTIMIZER.step()
                            OPTIMIZER.zero_grad()
                            if scheduler is not None: scheduler.step()
                            if self.EMA: ema.step()

                            # metrics
                            with th.no_grad():
                                if len(self.METRICS) > 0:
                                    _metr_list = dict()
                                    _model.eval()
                                    for _name, _metr_func in self.METRICS.items():
                                        _metr = _metr_func(pred_y, batch_label, **self.METRICS_CONFIG[_name])
                                        _metr_list[_name] = _metr.item()
                                __loss = accum_loss / accum_step
                            history['train_loss'].append(__loss)
                            accum_loss = 0.
                            accum_step = 0
                            num_update += 1
                            _can_valid = True
                            # print per step
                            if self.VERBOSE:
                                with _LoggingEnd(self.log_handler):
                                    self.logger.info(f'epoch: {i + 1:>6}, ({real_n_samp:>8d}/{n_trn_samp:>8d}), train_loss: {__loss:> 4.4e}')
                                    if len(self.METRICS) > 0:
                                        for _name, _metr in _metr_list.items():  # type: ignore
                                            self.logger.info(f', {_name}: {_metr:> 4.4e}')
                                    if scheduler is not None: self.logger.info(f', lr: {' '.join([f'{_:< 4.2e}' for _ in scheduler.get_last_lr()])}')
                                self.logger.info(f', time: {time.perf_counter() - time_gp:>10.4f}, [UPDATE GRAD]')

                        # validation
                        _is_start_val = (__loss < self.VAL_IF_TRN_LOSS_BELOW)
                        _is_step_to_val = (num_update % self.VAL_PER_STEP == 0)
                        #_is_at_least_val = (num_step < self.VAL_PER_STEP and num_step == n_batch)  # if not val. in the whole epoch, val.
                        if _is_start_val and _is_step_to_val and _can_valid:
                            time_val = time.perf_counter()
                            with _LoggingEnd(self.log_handler):
                                if self.VERBOSE: self.logger.info('VALIDATION...')
                            with th.no_grad():
                                if self.EMA: ema.apply()
                                _val_loss, _metr_list = self._val(_model, LOSS)
                                # print val results
                                if _val_loss is not None:
                                    history['val_loss'].append(_val_loss)
                                    if self.VERBOSE:
                                        self.logger.info('Done.')
                                        with _LoggingEnd(self.log_handler):
                                            self.logger.info(f'Validation loss: {_val_loss:> 4.4e}')
                                            for _name, _metr in _metr_list.items():
                                                self.logger.info(f', {_name}: {_metr:> 4.4e}')
                                        self.logger.info(f', time: {time.perf_counter() - time_val:<.4f}')

                                    if self.SAVE_CHK:
                                        if _val_loss < val_loss_old:
                                            with _LoggingEnd(self.log_handler):
                                                if self.VERBOSE: self.logger.info('Validation loss descent. Saving checkpoint file...')
                                            val_loss_old = copy.deepcopy(_val_loss)
                                            states = {
                                                'epoch': i,
                                                'model_state_dict': _model.state_dict(),
                                                'optimizer_state_dict': OPTIMIZER.state_dict(),
                                                'val_loss': _val_loss,
                                            }
                                            if scheduler is not None: states['lr_scheduler_state_dict'] = scheduler.state_dict()
                                            th.save(states, os.path.join(self.CHK_SAVE_PATH, f'best_checkpoint{self.CHK_SAVE_POSTFIX}.pt'))
                                            if self.VERBOSE: self.logger.info('Done.')
                                        else:
                                            if self.VERBOSE: self.logger.info(f'Validation loss NOT descent. Minimum loss: {val_loss_old:< 4.4e}.')
                                if self.EMA: ema.restore()
                            _can_valid = False

                        time_gp = time.perf_counter()
                        num_step += 1

                    except Exception as e:
                        _can_valid = False
                        time_gp = time.perf_counter()
                        num_step += 1
                        n_err += 1
                        if n_err <= 200:
                            self.logger.error(
                                f'An error occurred in the step {num_step} of epoch {i+1}: {e}; idx: {batch_data.idx}.\n'
                                f'Total ERROR NUMBER: {n_err}.\n{traceback.format_exc()}'
                                f'Training will continue, but metrics and losses in this step may not be accurate.'
                            )
                        else:
                            self.logger.fatal(
                                f'An error occurred in the step {num_step} of epoch {i+1}: {e}. Total ERROR NUMBER: {n_err}.\n'
                                f'TOO MANY ERRORS OCCURRED DURING TRAINING. I REFUSE TO CONTINUE SUCH A SICK JOB. BYE!'
                            )
                            exit(1)
                # print per epoch
                self.logger.info(f'\n*** EPOCH {i + 1:>6} Done.  LOOP TIME: {time.perf_counter() - time_ep}')
            self.logger.info('-' * 60 + f'MAIN LOOP DONE' + '-' * 60 + f'\nTOTAL TIME: {time.perf_counter() - time_tol}')

            # save the last chkpt.
            if len(history['val_loss']) == 0:  # avoid empty validation loss.
                if self.VERBOSE: self.logger.info('\nThe model has not yet been validated, skipped saving.')
            elif self.SAVE_CHK:
                with _LoggingEnd(self.log_handler):
                    if self.VERBOSE: self.logger.info('\nSaving the last check point file...')
                states = {
                    'epoch': self.EPOCH,
                    'model_state_dict': _model.state_dict(),
                    'optimizer_state_dict': OPTIMIZER.state_dict(),
                    'val_loss': history['val_loss'][-1],
                }
                if scheduler is not None: states['lr_scheduler_state_dict'] = scheduler.state_dict()
                th.save(states, os.path.join(self.CHK_SAVE_PATH, f'checkpoint{self.CHK_SAVE_POSTFIX}.pt'))
                if self.VERBOSE: self.logger.info('Done.')

        except Exception as e:
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:{traceback.format_exc()}\n')

        finally:
            # If the program stopped, recording checkpoints.
            if self.EMA: ema.restore()
            if i != self.EPOCH - 1:
                self.logger.info('*' * 89)
                self.logger.info(f'*** STOPPED AT {time.strftime("%Y%m%d_%H:%M:%S")} ***')
                if self.SAVE_CHK:
                    states = {
                        'epoch': i,
                        'model_state_dict': _model.state_dict(),
                        'optimizer_state_dict': OPTIMIZER.state_dict(),
                        'val_loss': th.inf
                    }
                    if len(history['val_loss']) != 0:
                        states['val_loss'] = history['val_loss'][-1],
                    if scheduler is not None: states['lr_scheduler_state_dict'] = scheduler.state_dict()
                    th.save(states, os.path.join(self.CHK_SAVE_PATH, f'stop_checkpoint{self.CHK_SAVE_POSTFIX}.pt'))
                    self.logger.info(
                        f'*** Checkpoint file was saved in {os.path.join(self.CHK_SAVE_PATH, f"stop_checkpoint{self.CHK_SAVE_POSTFIX}.pt")}'
                    )
            # close handler
            self.logger.removeHandler(self.log_handler)
            if isinstance(self.log_handler, logging.FileHandler):
                self.log_handler.close()

    def _val(self, model, LOSS) -> Tuple[float, Dict]:
        """
        the model and LOSS here were instantiated.
        """
        model.eval()
        _val_loss = 0.
        _num_step = 0
        val_set = self._data_loader(self.VALID_DATA, self.VAL_BATCH_SIZE, self.DEVICE, **self._data_loader_configs)  # type: ignore # TODO
        _metr_list = {_name: 0. for _name in self.METRICS.keys()}
        with th.no_grad():
            for val_data, val_label in val_set:
                # to avoid get an empty batch
                if not isinstance(val_data, (th.Tensor,)):
                    len_data = val_data.batch_size
                else:
                    len_data = len(val_data)
                if len_data <= 0:
                    if self.VERBOSE: self.logger.info(f'An empty batch occurred in validation. Skipped.')
                    continue
                # pred & loss
                pred_y = model(val_data)
                val_loss = LOSS(pred_y, val_label)
                _val_loss += val_loss.item()
                # metrics
                if len(self.METRICS) > 0:
                    for _name, _metr_func in self.METRICS.items():
                        _metr = _metr_func(pred_y, val_label, **self.METRICS_CONFIG[_name])
                        _metr_list[_name] += _metr.item()
                _num_step += 1
            _val_loss = _val_loss / _num_step
        if len(self.METRICS) > 0:
            _metr_list = {_name: _val / _num_step for _name, _val in _metr_list.items()}  # note: mean value per batch.
        else:
            _metr_list = dict()
        return _val_loss, _metr_list

