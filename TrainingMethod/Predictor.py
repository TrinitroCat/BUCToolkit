#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Predictor.py
#  Environment: Python 3.12

import os
import time
import warnings
from typing import Any

import numpy as np
import torch as th
from torch import nn

from BM4Ckit.TrainingMethod._io import _CONFIGS, _LoggingEnd


class Predictor(_CONFIGS):
    """
    A Base Predictor class.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: the path of input file
        verbose: control the verboseness of output
        device: the device that models run on
    """

    def __init__(self, config_file: str) -> None:
        super().__init__(config_file)

        self.config_file = config_file
        self.reload_config(config_file)
        if self.VERBOSE: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None

    def predict(self, model, test_model: bool = False, warm_up: bool = False):
        """
        Parameters:
            model: the input model which is uninstantiated nn.Module class.
            test_model: bool, if True, consumed time per batch and max memory per batch would be returned.
            warm_up: bool, if True, model will idle on some pseudo-samples for warming up.

        Returns:
            None, if `SAVE_PREDICTIONS` in config_file is true.
            np.NdArray, if `SAVE_PREDICTIONS` in config_file is false.
        """
        # check vars
        _model: nn.Module = model(**self.MODEL_CONFIG)
        if self.START == 'resume' or self.START == 1:
            chk_data = th.load(self.LOAD_CHK_FILE_PATH)
            if self.param is None:
                _model.load_state_dict(chk_data['model_state_dict'])
            else:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
            epoch_now = chk_data['epoch']
        elif self.START == 'from_scratch' or self.START == 0:
            warnings.warn('The model was not read trained parameters from checkpoint file. \nI HOPE YOU KNOW WHAT YOU ARE DOING!', RuntimeWarning)
            epoch_now = 0
            if self.param is not None:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
        else:
            raise ValueError('Invalid START value. It should be "from_scratch" / 0 or "resume" / 1 ')
        # model vars
        _model = _model.to(self.DEVICE)

        # preprocessing data # TODO
        if self._data_loader is None: raise RuntimeError('Please Set the DataLoader.')
        if not self._has_load_data: raise RuntimeError('Please Set the Data to Predict.')

        # initialize
        self.n_samp = len(self.TRAIN_DATA['data'])  # sample number
        self.n_batch = self.n_samp // self.BATCH_SIZE + 1  # total batch number per epoch

        try:
            # I/O
            if self.VERBOSE > 0:
                __time = time.strftime("%Y%m%d_%H:%M:%S")
                para_count = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                self.logger.info('*' * 60 + f'\n TIME: {__time}')
                self.logger.info(' TASK: PREDICTION <<')
                if (self.START == 0) or (self.START == 'from_scratch'):
                    self.logger.info(' FROM_SCRATCH <<')
                else:
                    self.logger.info(' RESUME <<')
                self.logger.info(f' COMMENTS: {self.COMMENTS}')
                self.logger.info(f' I/O INFORMATION:')
                if not self.REDIRECT:
                    self.logger.info('\tPREDICTION LOG OUTPUT TO SCREEN')
                else:
                    output_file = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
                    self.logger.info(f'\tPREDICTION LOG OUTPUT TO {output_file}')  # type: ignore
                if (self.START != 0) and (self.START != 'from_scratch'):
                    self.logger.info(f'\tMODEL PARAMETERS LOAD FROM: {self.LOAD_CHK_FILE_PATH}')
                self.logger.info(f' MODEL NAME: {self.MODEL_NAME}')
                self.logger.info(f' MODEL INFORMATION:')
                self.logger.info(f'\tTOTAL PARAMETERS: {para_count}')
                if self.VERBOSE > 1:
                    for hp, hpv in self.MODEL_CONFIG.items():
                        self.logger.info(f'\t\t{hp}: {hpv}')
                self.logger.info(f' MODEL WILL RUN ON {self.DEVICE}')
                if self.SAVE_PREDICTIONS:
                    self.logger.info(f' PREDICTIONS WILL SAVE TO {self.PREDICTIONS_SAVE_FILE}')
                else:
                    self.logger.info(f' PREDICTIONS WILL SAVE IN MEMORY AND RETURN AS A VARIBLES.')
                self.logger.info(f' ITERATION INFORMATION:')
                self.logger.info(f'\tBATCH SIZE: {self.BATCH_SIZE}' +
                                 f'\n\tTOTAL SAMPLE NUMBER: {self.n_samp}\n' +
                                 '*' * 60 + '\n' + 'ENTERING MAIN LOOP...')

            time_tol = time.perf_counter()
            _model.eval()
            # warm_up
            if warm_up:
                with _LoggingEnd(self.log_handler):
                    if self.VERBOSE: self.logger.info('Warm up...')
                t_wu = time.perf_counter()
                val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
                __warm_up = 0
                for val_data, val_label in val_set:
                    with th.no_grad():
                        pred_y = _model(val_data)
                    __warm_up += 1
                    if __warm_up >= 5: break
                th.cuda.synchronize()
                th.cuda.empty_cache()
                th.cuda.reset_peak_memory_stats()
                if self.VERBOSE: self.logger.info(f'Done. Warm up time: {time.perf_counter() - t_wu:<.4f}')
            # MAIN LOOP
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            _results = list()
            _results_names = list()
            n_c = 1
            t_1 = 0.
            t_per_batch = list()
            max_alloc_mem = list()
            val_label = None
            for val_data, val_label in val_set:
                t_bt = time.perf_counter()
                # to avoid get an empty batch
                if len(val_data) <= 0:
                    if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                    continue
                # pred & loss
                with th.no_grad():
                    pred_y = _model(val_data)
                    _results.append(pred_y['energy'].detach().cpu().numpy())  # TODO, add Force & other properties.
                    if val_label is not None:
                        _results_names.extend(val_label)
                # Print info
                if 0 < self.VERBOSE <= 2:
                    if time.perf_counter() - t_1 > 1.:  # To avoid output too frequent
                        t_1 = time.perf_counter()
                        with _LoggingEnd(self.log_handler, '\r'):
                            self.logger.info('PROGRESS: ' + '>' * round(20 * n_c / self.n_batch) + f'{100 * n_c / self.n_batch:<.2f}%')
                elif self.VERBOSE > 2 and self.DEVICE != 'cpu':
                    th.cuda.synchronize()
                    self.logger.info(f'batch {n_c}, time: {time.perf_counter() - t_bt:<5.4f}, GPU_memory: {th.cuda.max_memory_reserved() / (1024 ** 2):>6.3f}')
                if test_model:
                    t_per_batch.append(time.perf_counter() - t_bt)
                    max_alloc_mem.append(th.cuda.max_memory_reserved() / (1024 ** 2))
                    th.cuda.empty_cache()
                    th.cuda.reset_peak_memory_stats()
                n_c += 1

            if self.VERBOSE: self.logger.info('PROGRESS: ' + '>' * 20 + f'100.0%')
            if self.VERBOSE: self.logger.info(f'MAIN LOOP DONE. Total Time: {time.perf_counter() - time_tol:<.4f}')
            _results = np.concatenate(_results)
            # save predictions
            if self.SAVE_PREDICTIONS:
                t_save = time.perf_counter()
                with _LoggingEnd(self.log_handler):
                    if self.VERBOSE: self.logger.info(f'SAVING RESULTS...')
                if val_label is not None:
                    if len(_results_names) != len(_results):
                        warnings.warn('The number of sample names does Not match the number of predicted results. PLEASE DOUBLE CHECK.',
                                      RuntimeWarning)
                    np.savez_compressed(self.PREDICTIONS_SAVE_FILE + '.npz', predictions=_results, names=_results_names)
                else:
                    np.savez_compressed(self.PREDICTIONS_SAVE_FILE + '.npz', predictions=_results)
                if self.VERBOSE: self.logger.info(f'Done. Saving Time: {time.perf_counter() - t_save:<.4f}')
            elif test_model:
                return _results, t_per_batch, max_alloc_mem
            else:
                return _results

        except Exception as e:
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n')

        finally:
            pass
