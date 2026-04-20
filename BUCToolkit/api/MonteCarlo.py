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

from BUCToolkit.BatchMC import MMC
from ._io import _CONFIGS, _LoggingEnd, _Model_Wrapper_pyg, _Model_Wrapper_dgl
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT


class MonteCarlo(_CONFIGS):
    """
    Class of Monte Carlo algorithms, including NVT g Monte Carlo & simulated annealing (temperature varied MC).
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: path to input file.
        data_type: graph data type. 'pyg' for torch-geometric BatchData, 'dgl' for dgl DGLGraph.

    Input file parameters: (Under the section `MC` in input files.)
        ITER_SCHEME: Literal['Gaussian', 'Cauchy', 'Uniform'], the scheme of atoms moving. Default: Gaussian.
        COORDINATE_UPDATE_PARAM: float, the scale parameter for coordinates update. variation for Gaussian/range for Uniform/scale for Cauchy.
        MAXITER: int, the maximum iteration numbers. Default: 300.
        TEMPERATURE_INIT: initial temperature.
        TEMPERATURE_SCHEME: Literal['constant', 'linear', 'exponential', 'log', 'fast'], temperature update scheme.
            * `constant` for a fixed temperature during the whole simulation.
            * `linear` for a linearly changed temperature from `TEMPERATURE_INIT` to `TEMPERATURE_SCHEME_PARAM` during MAXITER steps.
            * `exponential` for temperature (T) changing by T^(i + 1) = `TEMPERATURE_SCHEME_PARAM` * T^(i).
            * `log` for temperature (T) changing by T^(i) = `TEMPERATURE_INIT`/(1. + log(1. + `TEMPERATURE_SCHEME_PARAM` * i)), `i` is the step number.
            * `fast` for temperature (T) changing by T^(i) = `TEMPERATURE_INIT`/(1. + `TEMPERATURE_SCHEME_PARAM` * i), `i` is the step number.
        TEMPERATURE_SCHEME_PARAM: float, to control the temperature scheme. See args `TEMPERATURE_SCHEME`.
        TEMPERATURE_UPDATE_FREQ: update temperature per `TEMPERATURE_UPDATE_FREQ` step.
        OUTPUT_COORDS_PER_STEP: int, to control the frequency of outputting atom coordinates. If verbose = 3, atom velocities would also be outputted. Default: 1

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

        __ensembles = {'METROPOLIS': MMC, }
        if self.MC is None:
            raise RuntimeError('Monte Carlo Configs was NOT Set.')
        self.MC: Dict

        try:
            self.MCType = __ensembles[self.MC['TYPE'].upper()]
        except KeyError:
            raise NotImplementedError(f'Type {self.MC['TYPE']} is not supported yet.')
        self.MC_config = {
            'maxiter': int(self.MC.get('MAXITER', 10000)),
            'output_structures_per_step': int(self.MC.get('OUTPUT_COORDS_PER_STEP', 1)),
            'device': self.DEVICE,
            'verbose': self.VERBOSE
        }
        if self.REDIRECT:
            self.MC_config['output_file'] = self.PREDICTIONS_SAVE_FILE
        if self.MC['TYPE'].upper() == 'METROPOLIS':
            self.MC_config.update(
                {
                    'iter_scheme': str(self.MC.get('ITER_SCHEME', 'Gaussian')),
                    'temperature_init': float(self.MC.get('T_INIT', 298.15)),
                    'temperature_scheme': str(self.MC.get('T_SCHEME', 'constant')),
                    'temperature_update_freq': int(self.MC.get('T_UPDATE_FREQ', 1)),
                    'temperature_scheme_param': float(self.MC.get('T_SCHEME_PARAM', None)),
                    'coordinate_update_param': float(self.MC.get('COORDINATE_UPDATE_PARAM', 0.2)),
                }
            )

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

        mont_carlo = None
        try:
            # I/O
            if self.VERBOSE > 0:
                self.logout_task_information(_model, 'MC', self.MC_config, self.n_samp)

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
                    batch = getattr(data, 'batch', None)
                    if batch is not None:
                        n_batch_ = th.bincount(batch).tolist()
                    else:
                        n_batch_ = [len(dat.pos) for dat in data.to_data_list()]
                    return n_batch_

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

            mont_carlo = self.MCType(**self.MC_config)
            #if self.REDIRECT:
            #    _file_handler = os.path.join(self.OUTPUT_PATH, f'{time.strftime("%Y%m%d_%H_%M_%S")}_{self.OUTPUT_POSTFIX}.out')
            #    mont_carlo.reset_logger_handler(_file_handler)
            if self.SAVE_PREDICTIONS:
                mont_carlo._HOLD_DUMPER = True
                # change dumping mode to 'a' to contiguously dumper within the whole for-loop
                mont_carlo.dumper.reset_args(self.PREDICTIONS_SAVE_FILE, mode='a', cache_size=4096, )
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            n_c = 1  # running batch now
            n_s = 0  # number of calculated samples. each sample in batches in each for-loop += 1.
            for val_data, val_label in val_set:
                try:
                    # to avoid get an empty batch
                    if get_batch_size(val_data) <= 0:
                        if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
                    # MCc
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

                    mv2cent_freq = int(self.MC.get('MOVE_TO_CENTER_FREQ', -1))
                    mont_carlo.run(
                        model_wrap.Energy,
                        X_init,
                        Element_list=get_atomic_number(val_data),
                        Cell_vector=_cell,
                        func_args=(val_data,),
                        fixed_atom_tensor=fixed_mask,
                        batch_indices=batch_indx,
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

            if self.VERBOSE: self.logger.info(f'Monte Carlo Simulation Done. Total Time: {time.perf_counter() - time_tol:<.4f}')

        except Exception as e:
            th.cuda.synchronize()
            excp = traceback.format_exc()
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n{excp}')

        finally:
            th.cuda.synchronize()
            if mont_carlo is not None:
                mont_carlo.dumper.close()
            self.logger.removeHandler(self.log_handler)
            if isinstance(self.log_handler, logging.FileHandler):
                self.log_handler.close()
            pass
