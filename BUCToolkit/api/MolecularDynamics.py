#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: MolecularDynamics.py
#  Environment: Python 3.12
import logging
import math
import os
import time
import traceback
from typing import Dict, Any, Literal

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.BatchMD import NVE, NVT
from ._io import _CONFIGS, _LoggingEnd, _Model_Wrapper_pyg, _Model_Wrapper_dgl
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT
from BUCToolkit.BatchMD._BaseMD import _BaseMD


class MolecularDynamics(_CONFIGS):
    """
    Class of molecular dynamics simulation.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: path to input file.
        data_type: graph data type. 'pyg' for torch-geometric BatchData, 'dgl' for dgl DGLGraph.

    Input file parameters: (Under the section `MD` in input files)
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
        self.require_grad = self.MD.get('REQUIRE_GRAD', False)
        self.MD_config = {
            'time_step': self.MD.get('TIME_STEP', 1),
            'max_step': self.MD.get('MAX_STEP'),
            'T_init': self.MD.get('T_INIT', 298.15),
            'output_structures_per_step': self.MD.get('OUTPUT_COORDS_PER_STEP', 1),
            'device': self.DEVICE,
            'verbose': self.VERBOSE
        }
        if self.SAVE_PREDICTIONS:
            self.MD_config['output_file'] = self.PREDICTIONS_SAVE_FILE
        if self.MD['ENSEMBLE'] == 'NVT':
            self.MD_config['thermostat'] = self.MD.get('THERMOSTAT', 'CSVR')
            self.MD_config['thermostat_config'] = self.MD.get('THERMOSTAT_CONFIG', dict())

    def run(self, model):
        """
        Parameters:
            model: the input model which is non-instantiated nn.Module class.
        """
        # check logger
        if not self.logger.hasHandlers(): self.logger.addHandler(self.log_handler)
        # check vars
        _model: nn.Module = model(**self.MODEL_CONFIG)
        if (self.START == 'resume') or (self.START == 1) or (self.START == 2):
            chk_data = th.load(self.LOAD_CHK_FILE_PATH, weights_only=True)
            if self.param is None:
                _model.load_state_dict(chk_data['model_state_dict'], strict=self.STRICT_LOAD)
            else:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
        elif self.START == 'from_scratch' or self.START == 0:
            self.logger.warning(
                'WARNING: The model does not read the trained parameters from checkpoint file. I HOPE YOU KNOW WHAT YOU ARE DOING!'
            )
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

        mole_dynam = None
        try:
            # I/O
            if self.VERBOSE > 0:
                self.logout_task_information(_model, 'MD', self.MD_config, self.n_samp)

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

                def get_fixed_mask(data):
                    mask = getattr(data, 'fixed', None)
                    if mask is not None:
                        mask = mask.unsqueeze(0)
                    return mask

                def get_batch_indx(data):
                    return [len(dat.pos) for dat in data.to_data_list()]

                def get_init_veloc(data):
                    veloc = getattr(data, 'velocity', None)
                    if veloc is not None:
                        veloc = veloc.unsqueeze(0)
                    return veloc

                def get_indx(data):
                    _indx: Dict = getattr(data, 'idx', None)
                    return _indx

            else:
                model_wrap = _Model_Wrapper_dgl(_model)
                def get_batch_size(data):
                    return data.num_nodes('atom')

                def get_cell_vec(data):
                    return data.nodes['cell'].data['cell'].numpy(force=True)

                def get_atomic_number(data):
                    return data.nodes['atom'].data['Z'].unsqueeze(0).tolist()

                def get_fixed_mask(data):
                    mask = data.nodes['atom'].data.get('fix', None)
                    if mask is not None:
                        mask = mask.unsqueeze(0)
                    return mask

                def get_batch_indx(data):
                    return data.batch_num_nodes('atom')

                def get_init_veloc(data):
                    veloc = data.nodes['atom'].data.get('velocity', None)
                    if veloc is not None:
                        veloc = veloc.unsqueeze(0)
                    return veloc

                def get_indx(data):
                    _indx: Dict = data.nodes['atom'].data
                    return _indx.get('idx', None)

            mole_dynam = self.MDType(**self.MD_config)
            #if self.REDIRECT:
            #    _file_handler = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
            #    mole_dynam.reset_logger_handler(_file_handler)
            if self.SAVE_PREDICTIONS:
                mole_dynam._HOLD_DUMPER = True
                # change dumping mode to 'a' to contiguously dumper within the whole for-loop
                mole_dynam.dumper.reset_args(self.PREDICTIONS_SAVE_FILE, mode='a', cache_size=4096, )
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            n_c = 1  # running batch now
            n_s = 0  # number of calculated samples. each sample in batches in each for-loop += 1.
            for val_data, val_label in val_set:
                try:
                    # to avoid get an empty batch
                    if get_batch_size(val_data) <= 0:
                        if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
                    # MD
                    # cells & fixations
                    _cell = get_cell_vec(val_data)
                    fixed_mask = get_fixed_mask(val_data)
                    # get batch
                    batch_indx = get_batch_indx(val_data)
                    # get id
                    idx = get_indx(val_data)
                    idx = idx if idx is not None else [f'Untitled{_}' for _ in range(n_s, n_s + len(batch_indx))]
                    n_s += len(batch_indx)
                    element_tensor = get_atomic_number(val_data)
                    if self.VERBOSE > 0:
                        self.logger.info('*' * 89)
                        self.logger.info(f'Running Batch {n_c}.')
                        cell_str = np.array2string(
                            _cell, **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")
                        self.logger.info(f'Structure names: {idx}\n')
                        self.logger.info(f'Cell Vectors:\n{cell_str}\n')
                        # print structures titles with elements
                        self.logout_element_information(element_tensor, batch_indx)

                    # initial atom coordinates
                    if self.data_type == 'pyg':
                        X_init = val_data.pos.unsqueeze(0)
                    else:
                        X_init = val_data.nodes['atom'].data['pos'].unsqueeze(0)

                    mv2cent_freq = int(self.MD.get('MOVE_TO_CENTER_FREQ', -1))
                    mole_dynam.run(
                        model_wrap.Energy,
                        X_init,
                        Element_list=get_atomic_number(val_data),
                        Cell_vector=_cell,
                        V_init=get_init_veloc(val_data),  # user-defined initial velocities.
                        grad_func=model_wrap.Grad,
                        func_args=(val_data,),
                        grad_func_args=(val_data,),
                        is_grad_func_contain_y=False,
                        require_grad=self.require_grad,
                        batch_indices=batch_indx,
                        fixed_atom_tensor=fixed_mask,  # The Selective Dynamics.
                        move_to_center_freq=mv2cent_freq
                    )

                    # Print info
                    if self.VERBOSE > 0:
                        self.logger.info(f'Batch {n_c} done.')
                    n_c += 1

                except Exception as e:
                    self.logger.warning(f'WARNING: An error occurred in {n_c}th batch. Error: {e}.')
                    if self.VERBOSE > 0:
                        excp = traceback.format_exc()
                        self.logger.warning(f"Traceback:\n{excp}")
                    n_c += 1

            if self.VERBOSE: self.logger.info(f'Molecular Dynamics Done. Total Time: {time.perf_counter() - time_tol:<.4f}')

        except Exception as e:
            th.cuda.synchronize()
            excp = traceback.format_exc()
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n{excp}')

        finally:
            th.cuda.synchronize()
            if mole_dynam is not None:
                mole_dynam.dumper.close()
            self.logger.removeHandler(self.log_handler)
            if isinstance(self.log_handler, logging.FileHandler):
                self.log_handler.close()
            pass
