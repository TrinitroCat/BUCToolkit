#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: Predictor.py
#  Environment: Python 3.12
import logging
import math
import os
import time
import traceback
import warnings
from typing import Any, Literal
from queue import Queue
import threading

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.api._io import _CONFIGS, _LoggingEnd
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BUCToolkit.BatchStructures.StructuresIO import structures_io_dumper, ArrayDumper


class Predictor(_CONFIGS):
    """
    A Base Predictor class.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: the path of input file
    """

    def __init__(
            self,
            config_file: str,
            data_type: Literal['pyg', 'dgl'] = 'pyg',
    ) -> None:
        super().__init__(config_file)

        self.config_file = config_file
        self.reload_config(config_file)
        self.data_type = data_type
        if self.VERBOSE: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None
        self.dumper = None

    def predict(self, model, test_model: bool = False, warm_up: bool = False):
        """ Alias for `run(...)` """
        self.run(model, test_model, warm_up)

    @staticmethod
    def _async_dump(q: Queue):
        """

        Args:
            q: queue of `dumper, data`, where data is a dict of {str: Tensor}

        Returns:

        """
        while True:
            _ = q.get()
            if _[0] is None:
                break
            dumper: ArrayDumper = _[0]
            copy_stream = _[1]
            copy_stream.synchronize()
            data = _[2]
            data = [data[_k].numpy() for _k in sorted(data) if isinstance(data[_k], th.Tensor)]
            dumper.start_from_arrays(
                1,
                *data,
            )
            dumper.step(*data)

    def _async_print(self, q: Queue):
        """

        Args:
            q: queue of idx, batch_indx, n_s, data, where data is a dict of {str: Tensor}

        Returns:

        """
        while True:
            idx, batch_indx, n_s, copy_stream, data = q.get()
            data: dict
            if idx is None:
                break
            copy_stream.synchronize()
            idx = idx if idx is not None else [f'Untitled{_}' for _ in range(n_s, n_s + len(batch_indx))]
            self.logger.info(f"\nSample: {idx}")
            for key, dat in data.items():
                dat_print = np.array2string(dat.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT) if dat is not None else None
                self.logger.info(f"\t{key}:\n\t\t{dat_print}")

    def run(self, model, test_model: bool = False, warm_up: bool = False):
        """
        Parameters:
            model: the input model which is uninstantiated nn.Module class.
            test_model: bool, if True, consumed time per batch and max memory per batch would be returned.
            warm_up: bool, if True, model will idle on some pseudo-samples for warming up.

        Returns:
            None, if `SAVE_PREDICTIONS` in config_file is true.
            np.NdArray, if `SAVE_PREDICTIONS` in config_file is false.
        """
        # check logger
        if not self.logger.hasHandlers(): self.logger.addHandler(self.log_handler)
        dumper = structures_io_dumper(
            path=self.PREDICTIONS_SAVE_FILE,
            mode='x',
        )
        # check vars
        _model: nn.Module = model(**self.MODEL_CONFIG)
        if self.START == 'resume' or self.START == 1 or self.START == 2:
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
        self.n_batch = math.ceil(self.n_samp / self.BATCH_SIZE)  # total batch number per epoch

        try:
            # I/O
            if self.VERBOSE > 0:
                self.logout_task_information(_model, 'PREDICT', None, self.n_samp)

            time_tol = time.perf_counter()
            _model.eval()
            # data type parse
            if self.data_type == 'pyg':
                def get_indx(data):
                    _indx: dict = getattr(data, 'idx', None)
                    return _indx

                def get_batch_indx(data):
                    return [len(dat.pos) for dat in data.to_data_list()]
            else:
                def get_indx(data):
                    _indx: dict = data.nodes['atom'].data
                    return _indx.get('idx', None)

                def get_batch_indx(data):
                    return val_data.batch_num_nodes('atom')

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
            # initialize dumper
            dumper.initialize()
            # MAIN LOOP
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            _results = list()
            _results_names = list()
            n_c = 1
            t_1 = 0.
            t_per_batch = list()
            max_alloc_mem = list()
            val_label = None
            n_s = 0  # number of calculated samples. each sample in batches in each for-loop += 1.
            n_err = 0  # error count
            dump_queue = Queue(maxsize=100)
            dump_thread = threading.Thread(target=self._async_dump, args=(dump_queue, ), daemon=True)
            print_queue = Queue(maxsize=100)
            print_thread = threading.Thread(target=self._async_print, args=(print_queue, ), daemon=True)
            if not self.DEVICE.startswith('cpu'):
                copy_stream = th.cuda.Stream()
                cp_ctx = th.cuda.stream(copy_stream)
                default_stream = th.cuda.default_stream(self.DEVICE)
            else:
                copy_stream = EmptyContextManager()
                cp_ctx = EmptyContextManager()  # A placeholder
                default_stream = None
            try:
                dump_thread.start()
                print_thread.start()
                for val_data, val_label in val_set:
                    try:
                        t_bt = time.perf_counter()
                        # to avoid get an empty batch
                        if len(val_data) <= 0:
                            if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                            continue
                        # pred & loss
                        with th.no_grad():
                            pred_y = _model(val_data)
                            if not isinstance(pred_y, dict):
                                raise ValueError(f'The predictions of models should be a dict as the standard format, but got {type(pred_y)}.')

                            with cp_ctx:
                                copy_stream.wait_stream(default_stream)
                                _pred_cpu = dict()
                                for k, v in pred_y.items():
                                    if isinstance(v, th.Tensor):
                                        _pred_cpu[k] = v.to('cpu', non_blocking=True)

                            if self.VERBOSE > 1:
                                idx = get_indx(val_data)
                                batch_indx = get_batch_indx(val_data)
                                idx = idx if idx is not None else [f'Untitled{_}' for _ in range(n_s, n_s + len(batch_indx))]
                                n_s += len(batch_indx)
                                print_queue.put((idx, batch_indx, n_s, copy_stream, _pred_cpu))

                            dump_queue.put((dumper, copy_stream, _pred_cpu))

                        if self.VERBOSE > 2 and (not self.DEVICE.startswith('cpu')):
                            th.cuda.synchronize()
                            self.logger.info(f'batch {n_c}, time: {time.perf_counter() - t_bt:<5.4f}, '
                                             f'GPU_memory: {th.cuda.max_memory_reserved() / (1024 ** 2):>6.3f}')
                        if test_model:
                            t_per_batch.append(time.perf_counter() - t_bt)
                            max_alloc_mem.append(th.cuda.max_memory_reserved() / (1024 ** 2))
                            th.cuda.empty_cache()
                            th.cuda.reset_peak_memory_stats()
                        n_c += 1
                    except Exception as e:
                        n_err += 1
                        self.logger.error(f"ERROR: An error occurred in the {n_c}-th batch: {e}. Total errors: {n_err}.")
                        if n_err >= 100:
                            self.logger.fatal(f"** Too many errors (over 100) occurred while running model predictions. Program terminated.")
                            raise RuntimeError('** Too many errors (over 100) occurred while running model predictions.')
            finally:
                dump_queue.put([None]*3)
                dump_thread.join()
                print_queue.put([None]*5)
                print_thread.join()

            if self.VERBOSE: self.logger.info(f'MAIN LOOP DONE. Total Time: {time.perf_counter() - time_tol:<.4f}')
            dumper.close()
            if test_model:
                return _results, t_per_batch, max_alloc_mem
            else:
                return _results

        except Exception as e:
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n{traceback.format_exc()}')

        finally:
            self.logger.removeHandler(self.log_handler)
            if isinstance(self.log_handler, logging.FileHandler):
                self.log_handler.close()
            pass


class EmptyContextManager:
    def __enter__(self) -> 'EmptyContextManager':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def synchronize(self):
        pass

    def wait_stream(self, *args, **kwargs):
        pass


