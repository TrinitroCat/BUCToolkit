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
from typing import Dict, Any, Literal, Callable, Tuple

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.BatchMD.constrained_md import ConstrNVT, ConstrNVE
from BUCToolkit.utils._CheckModules import check_module
from ._io import _CONFIGS, _LoggingEnd, _Model_Wrapper_pyg, _Model_Wrapper_dgl
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT
from BUCToolkit.BatchGenerate.coords_interp import linear_interpolation_tens


class ConstrainedMolecularDynamics(_CONFIGS):
    """
    Class of constrained molecular dynamics simulation.
    Users need to set the dataset, dataloader, and constraint functions manually.

    Args:
        config_file: path to input file.
        data_type: graph data type. 'pyg' for torch-geometric BatchData, 'dgl' for dgl DGLGraph.

    Input file parameters: (Under the section `MD` in input files)
        ENSEMBLE: Literal[NVE, NVT], the ensemble for MD.
        CONSTR_MD_SCHEME: Literal[BLUE_MOON, SLOW_GROWTH], the scheme of constrained MD.
        NIMAGE: int, the number of images in BLUE_MOON MD. SLOW_GROWTH will ignore it.
        THERMOSTAT: Literal[Langevin, VR, CSVR, Nose-Hoover], the thermostat type. only used for ENSEMBLE=NVT.
                    'VR' is Velocity Rescaling and 'CSVR' is Canonical Sampling Velocity Rescaling by Bussi et al. [1].
        THERMOSTAT_CONFIG: Dict, the configs of thermostat.
                           * For 'Langevin', option key 'damping_coeff' (fs^-1) is to control the damping coefficient. Large damping_coeff lead to a strong coupling. Default: 0.01
                           * For 'CSVR', option key 'time_const' (fs) is to control the characteristic timescale. Large time_const leads to a weak coupling. Default: 10*TIME_STEP
        TIME_STEP: float, the time step (fs) for MD. Default: 1
        MAX_STEP: int, total time (fs) = TIME_STEP * MAX_STEP
        REQUIRE_GRAD: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs). Default: False.
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
        self.constr_func = None
        self.constr_val = None

        __ensembles = {'NVE': ConstrNVE, 'NVT': ConstrNVT}
        if self.MD is None:
            raise RuntimeError('Molecular Dynamics Configs was NOT Set.')
        self.MD: Dict

        try:
            self.MDType = __ensembles[self.MD['ENSEMBLE']]  # The Main Function of MD. MD_config is its parameters.
        except KeyError:
            raise NotImplementedError(f'Unknown Ensemble {self.MD["ENSEMBLE"]}.')
        self.require_grad = self.MD.get('REQUIRE_GRAD', False)
        # check modes
        self.CMD_MODE = self.MD['CONSTR_MD_SCHEME']
        if self.CMD_MODE == 'BLUE_MOON':
            # interp images read & check
            self.NIMAGE = self.MD['NIMAGE']
            if self.NIMAGE <= 0:
                raise RuntimeError(f'Images of configurations must be greater than 0, but got {self.NIMAGE}.')
        elif self.CMD_MODE == 'SLOW_GROWTH':
            pass
        else:
            raise NotImplementedError(f'Unknown Constrained MD Scheme {self.CMD_MODE}.')

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

    def set_constr_func(self, constr_func: Callable) -> None:
        self.constr_func = constr_func
        pass

    def set_constr_val(self, constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor) -> None:
        self.constr_val = constr_val
        pass

    def run(self, model):
        """
        Parameters:
            model: the input model which is non-instantiated nn.Module class.
        """
        # check logger
        if not self.logger.hasHandlers(): self.logger.addHandler(self.log_handler)
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
        #if self.constr_val is None: raise RuntimeError('Please Set the Constraint Values `constr_val`.')
        if self.constr_func is None: raise RuntimeError('Please Set the Constraint Function `constr_func`.')
        self.MD_config['constr_func'] = self.constr_func
        self.MD_config['constr_val'] = self.constr_val

        # initialize
        self.n_samp = len(self.TRAIN_DATA['dataIS'])  # sample number
        self.n_batch = math.ceil(self.n_samp / self.BATCH_SIZE)  # total batch number per epoch

        try:
            # PRINT TASK INFO
            if self.VERBOSE > 0:
                self.logout_task_information(_model, 'CMD', self.MD_config, self.n_samp)

            time_tol = time.perf_counter()
            _model.eval()
            # MAIN LOOP
            # define the model wrapper & batch size getter & cell vector getter for different data type
            if self.data_type == 'pyg':
                _pyg = check_module('torch_geometric.data')
                if _pyg is not None:
                    import torch_geometric.data as _pyg
                    self.pygBatch = _pyg.Batch
                else:
                    ImportError('The method is unavailable because the `torch-geometric` cannot be imported.')

                model_wrap = _Model_Wrapper_pyg(_model)
                def get_batch_size(data):
                    return len(data)

                def get_cell_vec(data):
                    return data.cell.numpy(force=True)

                def get_atomic_number(data):
                    return data.atomic_numbers.tolist()  # note: here is different from MD/Relaxations that NOT unsqueeze(0)

                def rebatched_graph(single_graph, X):
                    """ expand batches """
                    X_in = X.flatten(0, 1)  # convert X: (n_batch, n_atom, n_dim) into X': (n_batch * n_atom, 3)
                    batch_size = X.size(0)
                    graph = self.pygBatch.from_data_list([single_graph] * batch_size)
                    graph.pos = X_in
                    return graph

            else:
                model_wrap = _Model_Wrapper_dgl(_model)
                raise NotImplementedError  # TODO <<<<
                def get_batch_size(data):
                    return data.num_nodes('atom')

                def get_cell_vec(data):
                    return data.nodes['cell'].data['cell'].numpy(force=True)

                def get_atomic_number(data):
                    return data.nodes['atom'].data['Z'].unsqueeze(0).tolist()

            mole_dynam = self.MDType(**self.MD_config)
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, **self._data_loader_configs)
            if getattr(val_set, '_LOADER_TYPE', None) != 'ISFS':
                __err_msg = f'Data loader of ConstrainedMolecularDynamics requires ISFS, but got {getattr(val_set, '_LOADER_TYPE', None)}'
                if self.VERBOSE: self.logger.error(__err_msg)
                raise ValueError(__err_msg)
            n_c = 1  # running batch now
            for dataIS, dataFS in val_set:
                try:
                    # Check Batch
                    if get_batch_size(dataIS) != get_batch_size(dataFS):
                        __err_msg = f'The batch size of {dataIS} and {dataFS} do not match: {get_batch_size(dataIS)} != {get_batch_size(dataFS)}'
                        if self.VERBOSE: self.logger.error(__err_msg)
                        raise RuntimeError(__err_msg)
                    if get_batch_size(dataIS) <= 0:
                        if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
                    elif get_batch_size(dataIS) > 1:
                        if self.VERBOSE: self.logger.error(f'Constrained MD do not support batched calculation yet. You should set BATCH_SIZE to 1.')
                        raise RuntimeError(f'Constrained MD do not support batched calculation yet. You should set BATCH_SIZE to 1.')
                    # MD
                    if self.VERBOSE > 0:
                        self.logger.info('*' * 89)
                        self.logger.info(f'Running Batch {n_c}.')
                        self.logger.info('*' * 89)
                        cell_str = np.array2string(
                            get_cell_vec(dataIS), **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")  # TODO, Now it supports pygData and DGLGraph.
                        self.logger.info(f'Cell Vectors:\n{cell_str}')

                    if self.data_type == 'pyg':
                        batch_indx = [len(dat.pos) for dat in dataIS.to_data_list()]
                        _check_batch_indx = [len(dat.pos) for dat in dataFS.to_data_list()]
                    elif self.data_type == 'dgl':
                        batch_indx = dataIS.batch_num_nodes('atom')
                        _check_batch_indx = dataFS.batch_num_nodes('atom')
                    else:
                        raise RuntimeError(f'Data type {self.data_type} is not supported.')
                    # check atom numbers
                    for _, __ in enumerate(batch_indx):
                        if __ != _check_batch_indx[_]:
                            __err_msg = f'The atom numbers of {_}-th sample do not match: {__} != {_check_batch_indx[_]}'
                            if self.VERBOSE > 0: self.logger.error(__err_msg)
                            raise RuntimeError(__err_msg)

                    # initial atom coordinates
                    if self.data_type == 'pyg':
                        X_is = dataIS.pos
                        X_fs = dataFS.pos
                    else:
                        X_is = dataIS.nodes['atom'].data['pos']
                        X_fs = dataFS.nodes['atom'].data['pos']
                    X_init_ = linear_interpolation_tens(X_is, X_fs, self.NIMAGE)
                    # rebatch data
                    origin_elem_list = get_atomic_number(dataIS)
                    dataIS = rebatched_graph(dataIS, X_init_)

                    # run
                    mole_dynam.run(
                        model_wrap.Energy,
                        X_init_,
                        [origin_elem_list]*self.NIMAGE,
                        V_init=None,  # TODO, Support user-defined initial velocities.
                        grad_func=model_wrap.Grad,
                        func_args=(dataIS,), grad_func_args=(dataIS,),
                        is_grad_func_contain_y=False,
                        require_grad=self.require_grad,
                        fixed_atom_tensor=None,  # TODO, The Selective Dynamics.
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

            if self.VERBOSE:
                self.logger.info(
                    f'Constrained Molecular Dynamics Done. Total Time: {time.perf_counter() - time_tol:<.4f}'
                )

        except Exception as e:
            th.cuda.synchronize()
            excp = traceback.format_exc()
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n{excp}')

        finally:
            th.cuda.synchronize()
            self.logger.removeHandler(self.log_handler)
            if isinstance(self.log_handler, logging.FileHandler):
                self.log_handler.close()
            pass
