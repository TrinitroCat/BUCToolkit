#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: MolecularDynamics.py
#  Environment: Python 3.12

import os
import time
import traceback
from typing import Dict, Any, Literal

import numpy as np
import torch as th
from torch import nn

from BM4Ckit.BatchMD import NVE, NVT
from ._io import _CONFIGS, _LoggingEnd, _Model_Wrapper_pyg, _Model_Wrapper_dgl
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT


class MolecularDynamics(_CONFIGS):
    """
    Class of molecular dynamics simulation.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: path to input file.
        data_type: graph data type. 'pyg' for torch-geometric BatchData, 'dgl' for dgl DGLGraph.
        verbose: control the verboseness of output.
        device: the device that models run on.

    Input file parameters:
        ENSEMBLE: Literal[NVE, NVT], the ensemble for MD.
        THERMOSTAT: Literal[Langevin, VR, CSVR, Nose-Hoover], the thermostat type. only used for ENSEMBLE=NVT.
                    'VR' is Velocity Rescaling and 'CSVR' is Canonical Sampling Velocity Rescaling by Bussi et al. [1].
        THERMOSTAT_CONFIG: Dict, the configs of thermostat.
                           * For 'Langevin', option key 'damping_coeff' (fs^-1) is to control the damping coefficient. Large damping_coeff lead to a strong coupling. Default: 0.01
                           * For 'CSVR', option key 'time_const' (fs) is to control the characteristic timescale. Large time_const leads to a weak coupling. Default: 10*TIME_STEP
        TIME_STEP: float, the time step (fs) for MD. Default: 1
        MAX_STEP: int, total time (fs) = TIME_STEP * MAX_STEP
        T_INIT: float, initial Temperature (K). Default: 298.15
                * For ENSEMBLE=NVE, T_INIT is only used to generate ramdom initial velocities by Boltzmann dist if V_init was not given.
                * For ENSEMBLE=NVT, T_INIT is the target temperature of thermostat.
        OUTPUT_COORDS_PER_STEP: int, to control the frequency of outputting atom coordinates. If verbose = 3, atom velocities would also be outputted. Default: 1

    References:
        [1] J. Chem. Phys., 2007, 126, 014101.
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

        __ensembles = {'NVE': NVE, 'NVT': NVT}
        if self.MD is None:
            raise RuntimeError('Molecular Dynamics Configs was NOT Set.')
        self.MD: Dict

        try:
            self.MDType = __ensembles[self.MD['ENSEMBLE']]  # The Main Function of MD. MD_config is its parameters.
        except KeyError:
            raise NotImplementedError(f'Unknown Ensemble {self.MD["ENSEMBLE"]}.')
        self.MD_config = {'time_step': self.MD.get('TIME_STEP', 1),
                          'max_step': self.MD.get('MAX_STEP'),
                          'T_init': self.MD.get('T_INIT', 298.15),
                          'output_structures_per_step': self.MD.get('OUTPUT_COORDS_PER_STEP', 1),
                          'device': self.DEVICE,
                          'verbose': self.VERBOSE}
        if self.REDIRECT:
            self.MD_config['output_file'] = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
        if self.MD['ENSEMBLE'] == 'NVT':
            self.MD_config['thermostat'] = self.MD.get('THERMOSTAT', 'CSVR')
            self.MD_config['thermostat_config'] = self.MD.get('THERMOSTAT_CONFIG', dict())

    def run(self, model):
        """
        Parameters:
            model: the input model which is non-instantiated nn.Module class.
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

        try:
            # I/O
            if self.VERBOSE > 0:
                __time = time.strftime("%Y%m%d_%H:%M:%S")
                para_count = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                self.logger.info('*' * 60 + f'\n TIME: {__time}')
                self.logger.info(' TASK: Molecular Dynamics <<')
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
                self.logger.info(f'\tENSEMBLE: {self.MD["ENSEMBLE"]}')
                for _algo_conf_name, _algo_conf in self.MD_config.items():
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

            else:
                model_wrap = _Model_Wrapper_dgl(_model)
                def get_batch_size(data):
                    return data.num_nodes('atom')

                def get_cell_vec(data):
                    return data.nodes['cell'].data['cell'].numpy(force=True)

                def get_atomic_number(data):
                    return data.nodes['atom'].data['Z'].unsqueeze(0).tolist()

            mole_dynam = self.MDType(**self.MD_config)
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            n_c = 1  # running batch now
            X_dict = dict()
            for val_data, val_label in val_set:
                try:
                    # to avoid get an empty batch
                    if get_batch_size(val_data) <= 0:
                        if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
                    # MD
                    if self.VERBOSE > 0:
                        self.logger.info('*' * 100)
                        self.logger.info(f'Running Batch {n_c}.')
                        self.logger.info('*' * 100)
                        cell_str = np.array2string(
                            get_cell_vec(val_data), **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")  # TODO, Now it supports pygData and DGLGraph.
                        self.logger.info(f'Cell Vectors:\n{cell_str}')

                    if self.data_type == 'pyg':
                        batch_indx = [len(dat.pos) for dat in val_data.to_data_list()]
                    else:
                        batch_indx = val_data.batch_num_nodes('atom')
                    # initial atom coordinates
                    if self.data_type == 'pyg':
                        X_init = val_data.pos.unsqueeze(0)
                    else:
                        X_init = val_data.nodes['atom'].data['pos'].unsqueeze(0)

                    mole_dynam.run(
                        model_wrap.Energy,
                        X_init,  # TODO, Now it support pygData and DGLGraph.
                        get_atomic_number(val_data),
                        V_init=None,  # TODO, Support user-defined initial velocities.
                        grad_func=model_wrap.Grad,
                        func_args=(val_data,), grad_func_args=(val_data,),
                        is_grad_func_contain_y=False,
                        fixed_atom_tensor=None,  # TODO, The Selective Dynamics.
                        batch_indices=batch_indx,
                    )

                    # Print info
                    if self.VERBOSE > 0:
                        self.logger.info(f'Batch {n_c} done.')
                    n_c += 1

                except Exception as e:
                    self.logger.warning(f'WARNING: An error occurred in {n_c}th batch. Error: {e}.')
                    if self.VERBOSE > 1:
                        excp = traceback.format_exc()
                        self.logger.warning(f"Traceback:\n{excp}")
                    n_c += 1

            if self.VERBOSE: self.logger.info(f'Molecular Dynamics Done. Total Time: {time.perf_counter() - time_tol:<.4f}')
            if self.SAVE_PREDICTIONS:
                t_save = time.perf_counter()
                with _LoggingEnd(self.log_handler):
                    if self.VERBOSE: self.logger.info(f'SAVING RESULTS...')
                th.save(X_dict, self.PREDICTIONS_SAVE_FILE)
                if self.VERBOSE: self.logger.info(f'Done. Saving Time: {time.perf_counter() - t_save:<.4f}')
            else:
                return X_dict

        except Exception as e:
            th.cuda.synchronize()
            excp = traceback.format_exc()
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n{excp}')

        finally:
            th.cuda.synchronize()
            pass
