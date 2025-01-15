#  Copyright (c) 2024.12.10, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.7b
#  File: StuctureOptimization.py
#  Environment: Python 3.12

import os
import time
import traceback
import warnings
from typing import Any, Literal, Dict

import numpy as np
import torch as th
from torch import nn

from BM4Ckit.BatchOptim.minimize import CG, QN, FIRE
from BM4Ckit.TrainingMethod._io import _CONFIGS, _LoggingEnd, _Model_Wrapper_pyg, _Model_Wrapper_dgl
from BM4Ckit._print_formatter import FLOAT_ARRAY_FORMAT
from BM4Ckit.utils._Masses import ATOMIC_NUMBER
from BM4Ckit.utils import IrregularTensorReformat


class StructureOptimization(_CONFIGS):
    """
    Structure optimization for relaxation and transition state.

    Input file parameters:
          ALGO: Literal[CG, BFGS, MIX, FIRE], the optimization algo.
          ITER_SCHEME: Literal['PR+', 'FR', 'PR', 'WYL'], only use for ALGO=CG, the iteration scheme of CG. Default: PR+.
          E_THRES: float, threshold of Energy difference (eV). Default: 1e-4.
          F_THRES: float, threshold of max Force (eV/Ang). Default: 5e-2.
          MAXITER: int, the maximum iteration numbers. Default: 300.
          STEPLENGTH: float, the initial step length for line search. Default: 0.5.

          LINESEARCH: Literal[Backtrack, Golden, Wolfe], only used for ALGO=CG, BFGS or MIX.
            'Backtrack' with Armijo's cond., 'Golden' for exact line search by golden sec. Algo., 'Wolfe' for advance & retreat algo. With weak Wolfe cond.
          LINESEARCH_MAXITER: the maximum iteration numbers of line search, only used for CG, BFGS and MIX. Default: 10.
          LINESEARCH_THRES: float, threshold of exact line search. Only used for LINESEARCH=Golden.
          LINESEARCH_FACTOR: A factor in linesearch. Shrinkage factor for "Backtrack" & "Wolfe", scaling factor in interval search for "Golden". Default: 0.6.

          # Following parameters are only for ALGO=FIRE
          ALPHA:
          ALPHA_FAC:
          FAC_INC:
          FAC_DEC: float = 0.5
          N_MIN: int = 5
          MASS: float = 20.0
    """

    def __init__(self, config_file: str, data_type: Literal['pyg', 'dgl'] = 'pyg', verbose: int = 0, device: str | th.device = 'cpu') -> None:
        super().__init__(config_file, device)

        self.config_file = config_file
        assert data_type in {'pyg', 'dgl'}, f'Invalid data type {data_type}. It must be "pyg" or "dgl".'
        self.data_type = data_type
        self.verbose = verbose
        self.DEVICE = device
        self.reload_config(config_file)
        if self.verbose: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None

        __relax_dict = {'CG': CG, 'BFGS': QN, 'FIRE': FIRE}
        if self.RELAXATION is None:
            raise RuntimeError('Structure Optimization Algorithm was NOT Set.')
        if self.RELAXATION['ALGO'] == 'BFGS':
            _iterschem = 'BFGS'
            iterschem = 'iter_scheme'
        elif self.RELAXATION['ALGO'] == 'MIX':
            _iterschem = self.RELAXATION.get('ITER_SCHEME', 'PR+')
            iterschem = 'iterschem_CG'  # iter_scheme QN was automatically set to 'BFGS'
        elif self.RELAXATION['ALGO'] == 'CG':
            _iterschem = self.RELAXATION.get('ITER_SCHEME', 'PR+')
            iterschem = 'iter_scheme'
        else:
            _iterschem = 'FIRE'
            iterschem = 'iter_scheme'
        self.Stru_Opt = __relax_dict[self.RELAXATION['ALGO']]
        if self.RELAXATION['ALGO'] != 'FIRE':
            self.Stru_Opt_config = {iterschem: _iterschem,
                                    'E_threshold': self.RELAXATION.get('E_THRES', 1.e-4),
                                    'F_threshold': self.RELAXATION.get('F_THRES', 0.05),
                                    'maxiter': self.RELAXATION.get('MAXITER', 300),
                                    'linesearch': self.RELAXATION.get('LINESEARCH', 'Backtrack'),
                                    'linesearch_maxiter': self.RELAXATION.get('LINESEARCH_MAXITER', 10),
                                    'linesearch_thres': self.RELAXATION.get('LINESEARCH_THRES', 0.05),
                                    'linesearch_factor': self.RELAXATION.get('LINESEARCH_FACTOR', 0.6),
                                    'steplength': self.RELAXATION.get('STEPLENGTH', 0.5),
                                    'device': self.DEVICE,
                                    'verbose': self.verbose}
        else:
            self.Stru_Opt_config = {'E_threshold': self.RELAXATION.get('E_THRES', 1.e-4),
                                    'F_threshold': self.RELAXATION.get('F_THRES', 0.05),
                                    'maxiter': self.RELAXATION.get('MAXITER', 300),
                                    'steplength': self.RELAXATION.get('STEPLENGTH', 0.5),
                                    'alpha': self.RELAXATION.get('ALPHA', 0.1),
                                    'alpha_fac': self.RELAXATION.get('ALPHA_FAC', 0.99),
                                    'fac_inc': self.RELAXATION.get('FAC_INC', 1.1),
                                    'fac_dec': self.RELAXATION.get('FAC_DEC', 0.5),
                                    'N_min': self.RELAXATION.get('ALPHA_FAC', 5),
                                    'mass': self.RELAXATION.get('MASS', 20),
                                    'device': self.DEVICE,
                                    'verbose': self.verbose}

    def relax(self, model):
        """
        Parameters:
            model: the input model which is `uninstantiated` nn.Module class.
        """
        # check vars
        _model: nn.Module = model(**self.MODEL_CONFIG)
        if self.START == 'resume' or self.START == 1:
            chk_data = th.load(self.LOAD_CHK_FILE_PATH)
            if self.param is None:
                _model.load_state_dict(chk_data['model_state_dict'], strict=False)
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

        # I/O
        try:
            if self.verbose > 0:
                __time = time.strftime("%Y%m%d_%H:%M:%S")
                para_count = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                self.logger.info('*' * 60 + f'\n TIME: {__time}')
                self.logger.info(' TASK: Structure Optimization <<')
                if (self.START == 0) or (self.START == 'from_scratch'):
                    self.logger.info(' FROM_SCRATCH <<')
                else:
                    self.logger.info(' RESUME <<')
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
                if self.verbose > 1:
                    for hp, hpv in self.MODEL_CONFIG.items():
                        self.logger.info(f'\t\t{hp}: {hpv}')
                self.logger.info(f' MODEL WILL RUN ON {self.DEVICE}')
                if self.SAVE_PREDICTIONS:
                    self.logger.info(f' PREDICTIONS WILL SAVE TO {self.PREDICTIONS_SAVE_FILE}')
                else:
                    self.logger.info(f' PREDICTIONS WILL SAVE IN MEMORY AND RETURN AS A VARIABLE.')
                self.logger.info(f' ITERATION INFORMATION:')
                self.logger.info(f'\tALGO: {self.RELAXATION["ALGO"]}')
                for _algo_conf_name, _algo_conf in self.Stru_Opt_config.items():
                    self.logger.info(f'\t{_algo_conf_name}: {_algo_conf}')
                self.logger.info(f'\tBATCH SIZE: {self.BATCH_SIZE}' +
                                 f'\n\tTOTAL SAMPLE NUMBER: {self.n_samp}\n' +
                                 '*' * 60 + '\n' + 'ENTERING MAIN LOOP...')

            time_tol = time.perf_counter()
            _model.eval()
            # MAIN LOOP
            # define the model wrapper & batch size getter & cell vector getter for different data type
            if self.data_type == 'pyg':
                model_wrap = _Model_Wrapper_pyg(_model)

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

            else:
                model_wrap = _Model_Wrapper_dgl(_model)

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

            optimizer = self.Stru_Opt(**self.Stru_Opt_config)
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            n_c = 1  # number of cycles. each for-loop += 1.
            n_s = 0  # number of calculated samples. each sample in batches in each for-loop += 1.
            # To record the minimized X, Force, and Energies.
            X_dict = dict()
            F_dict = dict()
            E_dict = dict()
            for val_data, val_label in val_set:
                try:  # Catch error in each loop & continue, instead of directly exit.
                    t_bt = time.perf_counter()
                    # to avoid get an empty batch
                    if get_batch_size(val_data) <= 0:
                        if self.verbose: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
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
                    # get id
                    idx = get_indx(val_data)
                    idx = idx if idx is not None else [f'Untitled{_}' for _ in range(n_s, len(batch_indx))]
                    n_s += len(batch_indx)
                    if self.verbose > 0:
                        self.logger.info('*' * 100)
                        self.logger.info(f'Relaxation Batch {n_c}.')
                        cell_str = np.array2string(
                            CELL, **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")
                        self.logger.info(f'Structure names: {idx}\n')
                        self.logger.info(f'Cell Vectors:\n{cell_str}\n')
                        # print Atoms Information
                        element_list = get_atomic_number(val_data)
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
                    # relax
                    min_ener, min_x, min_force = optimizer.run(
                        model_wrap.Energy,
                        X_init,
                        model_wrap.Grad,
                        func_args=(val_data,),
                        grad_func_args=(val_data,),
                        is_grad_func_contain_y=False,
                        output_grad=True,
                        batch_indices=batch_indx,
                        fixed_atom_tensor=fixed_mask,
                    )
                    with th.no_grad():
                        min_ener.detach_()
                        min_x.detach_()
                        min_force.detach_()

                        min_x = min_x.cpu().squeeze(0)
                        min_force = min_force.cpu().squeeze(0)
                        min_ener = min_ener.cpu()
                        x_list = th.split(min_x, batch_indx)
                        f_list = th.split(min_force, batch_indx)
                        X_dict.update({_id: x_list[i] for i, _id in enumerate(idx)})
                        F_dict.update({_id: f_list[i] for i, _id in enumerate(idx)})
                        E_dict.update({_id: min_ener[i] for i, _id in enumerate(idx)})
                    # Print info
                    if self.verbose > 0:
                        self.logger.info('-' * 100)
                    n_c += 1
                except Exception as e:
                    self.logger.warning(f'An error occurred in {n_c}th batch. Error: {e}.')
                    if self.verbose > 1:
                        excp = traceback.format_exc()
                        self.logger.warning(f"traceback:\n{excp}")
                    n_c += 1

            if self.verbose: self.logger.info(f'RELAXATION DONE. Total Time: {time.perf_counter() - time_tol:<.4f}')
            if self.SAVE_PREDICTIONS:
                t_save = time.perf_counter()
                with _LoggingEnd(self.log_handler):
                    if self.verbose: self.logger.info(f'SAVING RESULTS...')
                th.save({'Coordinates': X_dict, 'Forces': F_dict, 'Energies': E_dict}, self.PREDICTIONS_SAVE_FILE)
                if self.verbose: self.logger.info(f'Done. Saving Time: {time.perf_counter() - t_save:<.4f}')
            else:
                return X_dict

        except Exception as e:
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n')

        finally:
            pass

    def TS(self, model):
        """

        """
        raise NotImplementedError
