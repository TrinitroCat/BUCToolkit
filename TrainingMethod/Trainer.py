""" Training Methods """

import copy
import sys
import time
import os
import re
import traceback
from typing import Dict, Tuple, Literal, List

import numpy as np

import torch as th
from torch import nn

from ._io import _CONFIGS, _LoggingEnd


class Trainer(_CONFIGS):
    r"""
    The Trainer class.
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

    def __init__(self, config_file: str, verbose: int = 0, device: str | th.device = 'cpu') -> None:
        super().__init__(config_file, device)

        self.config_file = config_file
        self.verbose = verbose
        self.DEVICE = device
        self.reload_config(config_file)
        if self.verbose: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None
        pass

    def train(self, model):
        r"""
        Start Training.

        Herein the input model must be an `uninstantiated` nn.Module class.
        """
        # check vars
        _model: nn.Module = model(**self.MODEL_CONFIG)
        if self.START != 'from_scratch' and self.START != 0:
            chk_data = th.load(self.LOAD_CHK_FILE_PATH)
            if self.param is None:
                _model.load_state_dict(chk_data['model_state_dict'])
            else:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
            if self.START == 'resume' or self.START == 1:
                epoch_now = chk_data['epoch']
                val_loss_old = chk_data['val_loss'] if isinstance(chk_data['val_loss'], float) else chk_data['val_loss'][0]
            elif self.START == 'param_only' or self.START == 2:
                epoch_now = 0
                val_loss_old = th.inf
            else:
                raise ValueError('Invalid START value. It should be "from_scratch" / 0 , "resume" / 1 , "param_only" / 2')
        elif self.START == 'from_scratch' or self.START == 0:
            epoch_now = 0
            if self.param is not None:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
            val_loss_old = th.inf

        # model vars
        _model = _model.to(self.DEVICE)
        # loss & eval vars
        if self.LOSS is None:
            raise RuntimeError('the loss function is None. Please set the loss function.')
        else:
            LOSS = self.LOSS(**self.LOSS_CONFIG)
            LOSS = LOSS.to(self.DEVICE)
        if None in self.METRICS.values(): raise RuntimeError('Occurred an unknown metric function. Please use self.set_metrics to specify it.')
        # optim vars
        if self.OPTIMIZER is None:
            raise RuntimeError('the optimizer is None. Please set the optimizer.')
        else:
            OPTIMIZER = self.OPTIMIZER(_model.parameters(), **self.OPTIM_CONFIG)
            if self.START == 'resume' or self.START == 1:
                OPTIMIZER.load_state_dict(chk_data['optimizer_state_dict'])

        if self.LR_SCHEDULER is not None:
            scheduler = self.LR_SCHEDULER(OPTIMIZER, last_epoch=(epoch_now - 1), **self.LR_SCHEDULER_CONFIG)
            if self.START == 'resume' or self.START == 1:
                scheduler.load_state_dict(chk_data['lr_scheduler_state_dict'])
        else:
            scheduler = None

        # preprocessing data # TODO
        if self._data_loader is None: raise RuntimeError('Please Set the DataLoader.')
        if not self._has_load_data: raise RuntimeError('Please Set the Training Data and Validation Data')

        # initialize
        n_trn_samp = len(self.TRAIN_DATA['data'])  # sample number
        n_val_samp = len(self.VALID_DATA['data'])
        n_batch = n_trn_samp // self.BATCH_SIZE + 1  # total batch number per epoch
        history: Dict[Literal['train_loss', 'val_loss'], List[float]] = {'train_loss': list(), 'val_loss': list()}
        OPTIMIZER.zero_grad()
        i = epoch_now
        if not os.path.isdir(self.CHK_SAVE_PATH):
            os.makedirs(self.CHK_SAVE_PATH, )
        if not os.path.isdir(self.CHK_SAVE_PATH):
            os.makedirs(self.CHK_SAVE_PATH, )

        try:
            # I/O
            if self.verbose > 0:
                __time = time.strftime("%Y%m%d_%H:%M:%S")
                para_count = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                self.logger.info('*' * 60 + f'\n TIME: {__time}')
                self.logger.info(' TASK: TRAINING & VALIDATION <<')
                if (self.START == 0) or (self.START == 'from_scratch'):
                    self.logger.info(' FROM_SCRATCH <<')
                elif (self.START == 1) or (self.START == 'resume'):
                    self.logger.info(' RESUME <<')
                else:
                    self.logger.info(' RESUME (ONLY MODEL PARAMETERS) <<')
                self.logger.info(f' I/O INFORMATION:')
                if not self.REDIRECT:
                    self.logger.info('\tTRAINING LOG OUTPUT TO SCREEN')
                else:
                    output_file = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
                    self.logger.info(f'\tTRAINING LOG OUTPUT TO {output_file}')  # type: ignore
                if (self.START != 0) and (self.START != 'from_scratch'):
                    self.logger.info(f'\tCHECKPOINT FILE LOAD FROM: {self.LOAD_CHK_FILE_PATH}')
                self.logger.info(f'\tCHECKPOINT FILE SAVE TO: {self.CHK_SAVE_PATH}')
                #self.logger.info(f' TRAIN DATA SET: {self.TRAIN_DATA}')
                #self.logger.info(f' VALID DATA SET: {self.TRAIN_DATA}')
                self.logger.info(f' MODEL NAME: {self.MODEL_NAME}')
                self.logger.info(f' MODEL INFORMATION:')
                self.logger.info(f'\tTOTAL PARAMETERS: {para_count}')
                if self.verbose > 1:
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
                if self.verbose > 1:
                    self.logger.info(f'\t{__opt_repr[1]}')
                    for __partt in __opt_repr[2:-2]:
                        self.logger.info(f'\t\t{__partt}')
                if self.LR_SCHEDULER is not None:
                    self.logger.info(f'\tLR_SCHEDULER: {self.LR_SCHEDULER}')
                    self.logger.info(f'\tLR_SCHEDULER CONFIG:')
                    for hp, hpv in self.LR_SCHEDULER_CONFIG.items(): self.logger.info(f'\t\t{hp}: {hpv}')
                else:
                    self.logger.info('\tLR_SCHEDULER: None')
                self.logger.info(f' ITERATION INFORMATION:')
                self.logger.info(f'\tEPOCH: {self.EPOCH}\n\tBATCH SIZE: {self.BATCH_SIZE}\n\tVALID BATCH SIZE: {self.VAL_BATCH_SIZE}' +
                                 f'\n\tGRADIENT ACCUMULATION STEPS: {self.ACCUMULATE_STEP}\n\tEVAL PER {self.VAL_PER_STEP} STEPS\n' +
                                 '*' * 60 + '\n' + 'ENTERING MAIN LOOP...')

            # MAIN LOOP
            time_tol = time.perf_counter()
            for i in range(epoch_now, self.EPOCH):
                time_ep = time.perf_counter()
                real_n_samp = 0
                # load training set
                trn_set = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, **self._data_loader_configs)  # TODO

                self.logger.info('-' * 60 + f'EPOCH {i + 1}' + '-' * 60)
                # Training
                num_step = 1
                time_gp = time.perf_counter()
                for batch_data, batch_label in trn_set:
                    _model.train()
                    # to avoid get an empty batch
                    if not isinstance(batch_data, (th.Tensor, )):
                        len_data = batch_data.batch_size
                    else:
                        len_data = len(batch_data)
                    if len_data <= 0:
                        if self.verbose: self.logger.info(f'An empty batch occurred in step {num_step}. Skipped.')
                        continue
                    # batch device
                    batch_data.to(self.DEVICE)
                    #batch_label.to(self.DEVICE)

                    # pred & loss, pred must be a Dict.
                    pred_y = _model(batch_data)
                    # check nan
                    for key in pred_y.keys():
                        is_nan = th.isnan(pred_y[key]) + th.isinf(pred_y[key])
                        if th.any(is_nan):
                            pred_y[key] = th.where(th.isnan(pred_y[key]), 0., pred_y[key])
                            self.logger.warning('NaN occurred in model output, and has been set to 0.')
                            if self.DEBUG_MODE: self.logger.warning(f'batch_data:\n {batch_data}\n\nlabels:\n {batch_label}')
                    # backward
                    loss = LOSS(pred_y, batch_label) / self.ACCUMULATE_STEP
                    loss.backward()

                    # Debug Mode -----------------------------------------------------------------------------------------------------------
                    if self.DEBUG_MODE:
                        val_name = list()
                        val_para = list()
                        val_grad = list()
                        if num_step == 1:
                            old_val_para = [0. for _ in _model.named_parameters()]
                        for para_name, para in _model.named_parameters():
                            val_name.append(para_name)
                            val_para.append(para.cpu().detach().numpy() if para is not None else [0])
                            val_grad.append(para.grad.cpu().detach().numpy() if para.grad is not None else [0])
                        for i__ in range(len(val_name)):
                            self.logger.info('PARAMETER  %s : %.3e ~ %.3e' % (val_name[i__], np.min(val_para[i__]), np.max(val_para[i__])))
                            self.logger.info('PARA_GRAD  %s : %.3e ~ %.3e' % (val_name[i__], np.min(val_grad[i__]), np.max(val_grad[i__])))
                            self.logger.info(f'PARAMETER MSD: {np.mean((val_para[i__] - old_val_para[i__]) ** 2):<4.3e}')  # type: ignore
                        old_val_para = val_para
                        if loss != loss:
                            raise Exception('ERROR: NaN occurred in loss.\n\nModel outputs: \n%s\n\nTraining labels: \n%s' % (pred_y, batch_label))
                    # Debug Mode END-----------------------------------------------------------------------------------------------------------

                    # update & metrics & output
                    real_n_samp += len_data  # sample number count
                    if (num_step % self.ACCUMULATE_STEP == 0) or (num_step == n_batch):
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
                            __loss = loss.item() * self.ACCUMULATE_STEP
                        history['train_loss'].append(__loss)
                        # print per step
                        if self.verbose:
                            with _LoggingEnd(self.log_handler):
                                self.logger.info(f'epoch: {i + 1:>6}, ({real_n_samp:>8d}/{n_trn_samp:>8d}), train_loss: {__loss:> 4.4e}')
                                if len(self.METRICS) > 0:
                                    for _name, _metr in _metr_list.items():  # type: ignore
                                        self.logger.info(f', {_name}: {_metr:> 4.4e}')
                                if scheduler is not None: self.logger.info(f', lr: {scheduler.get_last_lr()[0]:< 4.4e}')
                            self.logger.info(f', time: {time.perf_counter() - time_gp:>10.4f}, [UPDATE GRAD]')

                    # validation
                    if ((n_val_samp >= 1) and (num_step % self.VAL_PER_STEP) == 0) or (num_step < self.VAL_PER_STEP and num_step == n_batch):
                        time_val = time.perf_counter()
                        with _LoggingEnd(self.log_handler):
                            if self.verbose: self.logger.info('VALIDATION...')
                        _val_loss, _metr_list = self._val(_model, LOSS)
                        # print val results
                        if _val_loss is not None:
                            history['val_loss'].append(_val_loss)
                            if self.verbose:
                                self.logger.info('Done.')
                                with _LoggingEnd(self.log_handler):
                                    self.logger.info(f'Validation loss: {_val_loss:> 4.4e}')
                                    for _name, _metr in _metr_list.items():
                                        self.logger.info(f', {_name}: {_metr:> 4.4e}')
                                self.logger.info(f', time: {time.perf_counter() - time_val:<.4f}')

                            if self.SAVE_CHK:
                                if _val_loss < val_loss_old:
                                    with _LoggingEnd(self.log_handler):
                                        if self.verbose: self.logger.info('Validation loss descent. Saving checkpoint file...')
                                    val_loss_old = copy.deepcopy(_val_loss)
                                    states = {
                                        'epoch': i,
                                        'model_state_dict': _model.state_dict(),
                                        'optimizer_state_dict': OPTIMIZER.state_dict(),
                                        'val_loss': _val_loss,
                                    }
                                    if scheduler is not None: states['lr_scheduler_state_dict'] = scheduler.state_dict()
                                    th.save(states, os.path.join(self.CHK_SAVE_PATH, f'best_checkpoint{self.CHK_SAVE_POSTFIX}.pt'))
                                    if self.verbose: self.logger.info('Done.')
                                else:
                                    if self.verbose: self.logger.info(f'Validation loss NOT descent. Minimum loss: {val_loss_old:< 4.4e}.')

                    time_gp = time.perf_counter()
                    num_step += 1
                # print per epoch
                self.logger.info(f'\n*** EPOCH {i + 1:>6} Done.  LOOP TIME: {time.perf_counter() - time_ep}')
            self.logger.info('-' * 60 + f'MAIN LOOP DONE' + '-' * 60 + f'\nTOTAL TIME: {time.perf_counter() - time_tol}')
            with _LoggingEnd(self.log_handler):
                if self.verbose: self.logger.info('Saving the last check point file...')
            # save the last chkpt.
            if len(history['val_loss']) == 0:  # avoid empty validation loss.
                if self.verbose: self.logger.info('\nThe model has not yet been validated, skipped saving.')
            elif self.SAVE_CHK:
                states = {
                    'epoch': self.EPOCH,
                    'model_state_dict': _model.state_dict(),
                    'optimizer_state_dict': OPTIMIZER.state_dict(),
                    'val_loss': history['val_loss'][-1],
                }
                if scheduler is not None: states['lr_scheduler_state_dict'] = scheduler.state_dict()
                th.save(states, os.path.join(self.CHK_SAVE_PATH, f'checkpoint{self.CHK_SAVE_POSTFIX}.pt'))
                if self.verbose: self.logger.info('Done.')

        except Exception as e:
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n')

        finally:
            # If program stopped, recording checkpoints.
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
        for val_data, val_label in val_set:
            # to avoid get an empty batch
            if not isinstance(val_data, (th.Tensor,)):
                len_data = val_data.batch_size
            else:
                len_data = len(val_data)
            if len_data <= 0:
                if self.verbose: self.logger.info(f'An empty batch occurred in validation. Skipped.')
                continue
            # pred & loss
            with th.no_grad():
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
