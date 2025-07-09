""" Training Methods """

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Trainer.py
#  Environment: Python 3.12

import copy
import os
import re
import time
from math import ceil
from typing import Dict, Tuple, Literal, List, Any

import numpy as np
import torch as th
from torch import nn

from ._io import _CONFIGS, _LoggingEnd


class Trainer(_CONFIGS):
    r"""
    The model trainer class.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: the path of input file
        verbose: control the verboseness of output
        device: the device that models run on

    Methods:
        train(model: torch.nn.Module), run model training.
        set_device(device: str | torch.device), manually set model and data running device.
        set_loss_fn(loss_fn: Any, loss_config: Optional[Dict] = None), manually set loss function which can be customized.
        set_metrics(metrics_fn: Dict[str, Callable], metrics_fn_config: Dict[str, Dict] | None = None), manually set metrics functions which can be customized.
        set_model_config(model_config: Dict[str, Any] | None = None), manually (re)set the configs (hyperparameters) of model.
        set_lr_scheduler(lr_scheduler: th.optim.lr_scheduler.LRScheduler, lr_scheduler_config: Optional[Dict[str, Any]] = None), set the learning rate scheduler.
        set_model_param(model_state_dict: Dict, is_strict: bool = True, is_assign: bool = False), manually set model parameters by giving model state dict.

    """

    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)

        self.config_file = config_file
        self.reload_config(config_file)
        if self.VERBOSE: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None
        self._layerwise_opt_configs = None

    def train(self, model):
        r"""
        Start Training.

        Herein the input model must be an `uninstantiated` nn.Module class.
        """
        # check vars
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
                __time = time.strftime("%Y%m%d_%H:%M:%S")
                para_count_all = sum(p.numel() for p in _model.parameters())
                para_count_train = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                self.logger.info('*' * 60 + f'\n BM4CKit 0.7a')
                self.logger.info(f' TIME: {__time}')
                self.logger.info(' TASK: TRAINING & VALIDATION <<')
                if (self.START == 0) or (self.START == 'from_scratch'):
                    self.logger.info(' FROM_SCRATCH <<')
                elif (self.START == 1) or (self.START == 'resume'):
                    self.logger.info(' RESUME <<')
                else:
                    self.logger.info(' RESUME (ONLY MODEL PARAMETERS) <<')
                self.logger.info(f' COMMENTS: {self.COMMENTS}')
                self.logger.info(f' I/O INFORMATION:')
                self.logger.info(f'\tVERBOSITY LEVEL: {self.VERBOSE}')
                if not self.REDIRECT:
                    self.logger.info('\tTRAINING LOG OUTPUT TO SCREEN')
                else:
                    output_file = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
                    self.logger.info(f'\tTRAINING LOG OUTPUT TO: {output_file}')  # type: ignore
                if (self.START != 0) and (self.START != 'from_scratch'):
                    self.logger.info(f'\tCHECKPOINT FILE LOAD FROM: {self.LOAD_CHK_FILE_PATH}')
                self.logger.info(f'\tCHECKPOINT FILE SAVE TO: {self.CHK_SAVE_PATH}_{self.CHK_SAVE_POSTFIX}')
                #self.logger.info(f' TRAIN DATA SET: {self.TRAIN_DATA}')
                #self.logger.info(f' VALID DATA SET: {self.TRAIN_DATA}')
                self.logger.info(f' MODEL NAME: {self.MODEL_NAME}')
                self.logger.info(f' MODEL INFORMATION:')
                self.logger.info(f'\tTOTAL PARAMETERS: {para_count_all}')
                self.logger.info(f'\tTOTAL TRAINABLE PARAMETERS: {para_count_train}')
                self.logger.info(f'\tHYPER-PARAMETERS:')
                if self.VERBOSE > 1:
                    for hp, hpv in self.MODEL_CONFIG.items():
                        self.logger.info(f'\t\t{hp}: {hpv}')
                self.logger.info(f' MODEL WILL TRAIN ON {self.DEVICE}')
                self.logger.info(f' LOSS FUNCTION: {self.loss_name}')
                if len(self.METRICS) > 0:
                    with _LoggingEnd(self.log_handler):
                        self.logger.info(f' METRICS: ')
                        for _name in self.METRICS.keys():
                            self.logger.info(f'{_name}  ')
                    self.logger.info('')
                else:
                    self.logger.info(' METRICS: None')
                self.logger.info(f' OPTIMIZER INFORMATION:')
                __opt_repr = re.split(r'\(\n|\)$|\s{2,}|\n', repr(OPTIMIZER))  # type: ignore
                self.logger.info(f'\tOPTIMIZER: {__opt_repr[0]}')
                if self.VERBOSE > 1:
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
                                    f'\t\tcontaining layers: {_layer_grp_name[kk].pattern if kk < len(_layer_grp_name) else "all remaining layers"}'
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
                self.logger.info(f' ITERATION INFORMATION:')
                self.logger.info(f'\tEPOCH: {self.EPOCH}\n\tBATCH SIZE: {self.BATCH_SIZE}\n\tVALID BATCH SIZE: {self.VAL_BATCH_SIZE}' +
                                 f'\n\tGRADIENT ACCUMULATION STEPS: {self.ACCUMULATE_STEP}\n\tEVAL PER {self.VAL_PER_STEP} STEPS\n' +
                                 '*' * 60 + '\n' + 'ENTERING MAIN LOOP...')

            # MAIN LOOP
            if self.DEBUG_MODE:
                th.autograd.set_detect_anomaly(True)
                para_arr_old_list = [0. for _ in _model.named_parameters()]
            time_tol = time.perf_counter()
            __loss = th.inf
            num_update = 0  # number of parameter updating
            _can_valid = False  # To avoid stacking
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
                            elif isinstance(pred_y[key], List):  # for ensemble wrapper model
                                is_nan = sum([th.isnan(_p) + th.isinf(_p) for _p in pred_y[key]])
                            else:
                                break
                            if th.any(is_nan):
                                pred_y[key] = th.where(th.isnan(pred_y[key]), 0., pred_y[key])
                                self.logger.warning('NaN occurred in model output, and has been set to 0.')
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
                        _can_valid = False

                    time_gp = time.perf_counter()
                    num_step += 1
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
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n')

        finally:
            # If the program stopped, recording checkpoints.
            if i != self.EPOCH - 1:
                self.logger.info('*' * 100)
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
                    self.logger.info(f'*** Checkpoint file was saved in {os.path.join(self.CHK_SAVE_PATH, f"stop_checkpoint{self.CHK_SAVE_POSTFIX}.pt")}')

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
