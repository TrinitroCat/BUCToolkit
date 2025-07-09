""" Calculating Normal Mode Vibration Frequencies """
#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: VibrationAnalysis.py
#  Environment: Python 3.12

import os
import time
import traceback
import warnings
from typing import Any, Literal, Dict

import numpy as np
import torch as th
from torch import nn

from BM4Ckit.BatchOptim.frequency import Frequency
from BM4Ckit.TrainingMethod._io import _CONFIGS, _LoggingEnd, _Model_Wrapper_regularBatch_pyg, _Model_Wrapper_dgl
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT
from BM4Ckit.utils._Element_info import ATOMIC_NUMBER, MASS, N_MASS

REVISED_FLOAT_ARRAY_FORMAT = FLOAT_ARRAY_FORMAT  # reduce the line_width to exhibit eigen-frequencies.
REVISED_FLOAT_ARRAY_FORMAT['max_line_width'] = 64


class VibrationAnalysis(_CONFIGS):
    """
    The class of normal mode frequencies calculation by finite difference algo.
    Due to the huge computation cost, it would run sequentially instead of batched.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: the path of input file.
        data_type: graph data type. 'pyg' for torch-geometric BatchData, 'dgl' for dgl DGLGraph.
        verbose: control the verboseness of output.
        device: the device that models run on.

    Input file parameters:
        BLOCK_SIZE: int, the batch size of points (i.e., one structure image of finite difference) for parallel computing at one time. Default: 1.
        DELTA: float, the step length of finite difference. Default: 1e-2.
        SAVE_HESSIAN: bool, whether to save calculated Hessian matrix. Default: False.

    """

    def __init__(
            self,
            config_file: str,
            data_type: Literal['pyg', 'dgl'] = 'pyg',
    ) -> None:
        super().__init__(config_file)

        self.config_file = config_file
        assert data_type in {'pyg', 'dgl'}, f'Invalid data type {data_type}. It must be "pyg" or "dgl".'
        self.data_type = data_type
        self.reload_config(config_file)
        if self.VERBOSE: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None

        self.Vib_config = {
            'method': self.VIBRATION.get('METHOD', 'Coord'),
            'block_size': int(self.VIBRATION.get('BLOCK_SIZE', 1)),
            'delta': float(self.VIBRATION.get('DELTA', 1e-2)),
        }
        self.is_save_Hessian = bool(self.VIBRATION.get('SAVE_HESSIAN', False))

    def run(self, model):
        """
        Parameters:
            model: the input model which is `uninstantiated` nn.Module class.
        """
        # check vars
        _model: nn.Module = model(**self.MODEL_CONFIG)
        if self.START == 'resume' or self.START == 1:
            chk_data = th.load(self.LOAD_CHK_FILE_PATH, weights_only=True)
            if self.param is None:
                _model.load_state_dict(chk_data['model_state_dict'], strict=False)
            else:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
            epoch_now = chk_data['epoch']
        elif self.START == 'from_scratch' or self.START == 0:
            self.logger.warning(
                'WARNING: The model was not read the trained parameters from checkpoint file. I HOPE YOU KNOW WHAT YOU ARE DOING!'
            )
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

        # I/O
        try:
            if self.VERBOSE > 0:
                __time = time.strftime("%Y%m%d_%H:%M:%S")
                para_count = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                self.logger.info('*' * 60 + f'\n TIME: {__time}')
                self.logger.info(' TASK: Vibration Calculation <<')
                if (self.START == 0) or (self.START == 'from_scratch'):
                    self.logger.info(' FROM_SCRATCH <<')
                else:
                    self.logger.info(' RESUME <<')
                self.logger.info(f' COMMENTS: {self.COMMENTS}')
                self.logger.info(f' I/O INFORMATION:')
                self.logger.info(f'\tVERBOSITY LEVEL: {self.VERBOSE}')
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
                    self.logger.info(f' PREDICTIONS WILL SAVE IN MEMORY AND RETURN AS A VARIABLE.')
                self.logger.info(f' ITERATION INFORMATION:')
                self.logger.info(f'\tALGO: {"Finite Difference" if self.Vib_config["method"] == "Coord" else "Automatic Differentiation"}')
                for _algo_conf_name, _algo_conf in self.Vib_config.items():
                    self.logger.info(f'\t{_algo_conf_name}: {_algo_conf}')
                self.logger.info(f'\tBATCH SIZE: {self.BATCH_SIZE}' +
                                 f'\n\tTOTAL SAMPLE NUMBER: {self.n_samp}\n' +
                                 '*' * 60 + '\n' + 'ENTERING MAIN LOOP...')

            time_tol = time.perf_counter()
            _model.eval()
            # MAIN LOOP
            # define the model wrapper & batch size getter & cell vector getter for different data type
            if self.data_type == 'pyg':
                model_wrap = _Model_Wrapper_regularBatch_pyg(_model)

                def get_batch_size(data):
                    return len(data)

                def get_cell_vec(data):
                    return data.cell.numpy(force=True)

                def get_atomic_number(data):
                    return data.atomic_numbers.unsqueeze(0).tolist()

                def get_indx(data):
                    _indx: Dict = getattr(data, 'idx', None)
                    return _indx

                def get_fixed_mask(data):
                    return data.fixed.unsqueeze(0)

                def rebatched_graph(single_graph, batch_size: int):
                    return model_wrap.pygBatch.from_data_list([single_graph, ]*batch_size)

            else:
                model_wrap = _Model_Wrapper_dgl(_model)
                raise NotImplementedError  # TODO <<<< complete dgl format

                def get_batch_size(data):
                    return data.num_nodes('atom')

                def get_cell_vec(data):
                    return data.nodes['cell'].data['cell'].numpy(force=True)

                def get_atomic_number(data):
                    return data.nodes['atom'].data['Z'].unsqueeze(0).tolist()

                def get_indx(data):
                    _indx: Dict = data.nodes['atom'].data
                    return _indx.get('idx', None)

                def get_fixed_mask(data):
                    return data.nodes['atom'].data.get('fix', None)

            vib_calculator = Frequency(**self.Vib_config)
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            n_c = 1  # number of cycles. each for-loop += 1.
            n_s = 0  # number of calculated samples. each sample in batches in each for-loop += 1.
            # To record the minimized X, Force, and Energies.
            freq_dict = dict()
            normal_mode_dict = dict()
            hessian_dict = dict()
            if (self.VERBOSE < 1) and (not self.SAVE_PREDICTIONS):
                warnings.warn('WARNING: Neither`verbose` nor `self.SAVE_PREDICTIONS` was turn on.'
                              ' Hence NOTHING WILL BE OUTPUT. I HOPE YOU KNOW WHAT YOU ARE DOING!')
                if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
            for val_data, val_label in val_set:
                try:  # Catch error in each loop & continue, instead of directly exit.
                    t_bt = time.perf_counter()
                    # to avoid get an empty batch
                    n_batch = get_batch_size(val_data)
                    if n_batch <= 0:
                        if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
                    elif n_batch > 1:
                        if self.VERBOSE: self.logger.error(f'Vibration do not support batched calculation yet. You should set BATCH_SIZE to 1.')
                        raise RuntimeError(f'Vibration do not support batched calculation yet. You should set BATCH_SIZE to 1.')
                    # batch indices
                    batch_indx1 = th.sum(
                        th.eq(val_data.batch, th.arange(0, val_data.batch_size, dtype=th.int64, device=self.DEVICE).unsqueeze(-1)), dim=-1
                    )
                    if self.data_type == 'pyg':
                        batch_indx = [len(dat.pos) for dat in val_data.to_data_list()]
                    else:
                        batch_indx = val_data.batch_num_nodes('atom')
                        # initial atom coordinates
                    if self.data_type == 'pyg':
                        X_init = val_data.pos.unsqueeze(0)
                    else:
                        X_init = val_data.nodes['atom'].data['pos'].unsqueeze(0)
                    CELL = get_cell_vec(val_data)
                    fixed_mask = get_fixed_mask(val_data)
                    element_list = get_atomic_number(val_data)
                    # get masses
                    masses = list()
                    for _Elem in element_list:
                        masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
                    masses = th.tensor(masses, dtype=th.float32, device=self.DEVICE)
                    masses = masses.unsqueeze(-1).expand_as(X_init)  # (n_batch, n_atom, n_dim)
                    # get id
                    idx = get_indx(val_data)
                    idx = idx if idx is not None else [f'Untitled{_}' for _ in range(n_s, len(batch_indx))]
                    n_s += len(batch_indx)
                    if self.VERBOSE > 0:
                        self.logger.info('*' * 100)
                        self.logger.info(f'Vibration Calculation Batch {n_c}.')
                        cell_str = np.array2string(
                            CELL, **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")
                        self.logger.info(f'Structure names: {idx}\n')
                        self.logger.info(f'Cell Vectors:\n{cell_str}\n')
                        # print Atoms Information
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
                    # run
                    #graph = rebatched_graph(val_data, )  # TODO, determining batch size in priori is difficult because batch size may change in calculation.
                    with th.no_grad():
                        eig_freq, normal_mode = vib_calculator.normal_mode(
                            model_wrap.Energy,
                            X_init.squeeze(0),  # input format: (n_atom, 3) instead of (1, n_atom, 3) like structure opt. or MD.
                            masses.squeeze(0),
                            func_args=(val_data,),
                            grad_func=model_wrap.Grad,
                            grad_func_args=(val_data,),
                            fixed_atom_tensor=fixed_mask.squeeze(0),
                            save_hessian=self.is_save_Hessian
                        )
                        if self.is_save_Hessian:
                            vib_hessian = vib_calculator.hessian
                    # Print info
                    eig_freq_THz = eig_freq * 15.63330456  # THz
                    eig_freq_cm1 = eig_freq * 521.47091 # cm^-1
                    if self.VERBOSE > 0:
                        self.logger.info(f'Eigen Vibration Frequency (cm^-1):\n{
                        np.array2string(eig_freq_cm1.numpy(force=True), **REVISED_FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        }')
                        self.logger.info(f'\nNormal Mode:\n{
                        np.array2string(normal_mode.numpy(force=True), **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        }\n')
                        if self.is_save_Hessian:
                            self.logger.info(f'Hessian Matrix:\n{
                            np.array2string(vib_hessian.numpy(force=True), **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                            }\n')
                        self.logger.info('-' * 100)
                    if self.SAVE_PREDICTIONS:
                        freq_dict[idx] = eig_freq
                        normal_mode_dict[idx] = normal_mode
                        if self.is_save_Hessian:
                            hessian_dict[idx] = vib_hessian
                    n_c += 1

                except Exception as e:
                    self.logger.warning(f'WARNING: An error occurred in {n_c}th batch. Error: {e}.')
                    if self.VERBOSE > 1:
                        excp = traceback.format_exc()
                        self.logger.warning(f"Traceback:\n{excp}")
                    n_c += 1

            if self.VERBOSE: self.logger.info(f'VIBRATION CALCULATION DONE. Total Time: {time.perf_counter() - time_tol:<.4f}')
            if self.SAVE_PREDICTIONS:
                t_save = time.perf_counter()
                with _LoggingEnd(self.log_handler):
                    if self.VERBOSE: self.logger.info(f'SAVING RESULTS...')
                info_dict = {'Frequencies': freq_dict, 'Forces': normal_mode_dict,}
                if self.is_save_Hessian:
                    info_dict['Hessian'] = vib_hessian
                th.save(info_dict, self.PREDICTIONS_SAVE_FILE)
                if self.VERBOSE: self.logger.info(f'Done. Saving Time: {time.perf_counter() - t_save:<.4f}')

        except Exception as e:
            th.cuda.synchronize()
            excp = traceback.format_exc()
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n{excp}')

        finally:
            th.cuda.synchronize()
            pass
