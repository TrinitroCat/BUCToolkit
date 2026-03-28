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

from BUCToolkit.BatchOptim.TS.CI_NEB import CI_NEB
from BUCToolkit.utils._CheckModules import check_module
from ._io import _CONFIGS, _LoggingEnd, _Model_Wrapper_pyg, _Model_Wrapper_dgl
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT
from BUCToolkit.utils.AtomSelector import atom_fix_selector


class ClimbingImageNudgedElasticBand(_CONFIGS):
    """
    Class of Climbing Image Nudged Elastic Band algorithm to search transition state.
    Users need to set the dataset and dataloader manually.

    Args:
        config_file: path to input file.
        data_type: graph data type. 'pyg' for torch-geometric BatchData, 'dgl' for dgl DGLGraph.

    Input file parameters: (Under the section `MD` in input files)
        ALGO: Literal['CI-NEB'] = 'CI-NEB', the algorithm to use (now only supports CI-NEB).
        N_IMAGES: int, the image number of NEB.
        SPRING_CONST: float = 1, the elastic (spring) constant.
        OPTIMIZER: Literal['FIRE'] = 'FIRE', the optimizer to use (now only support FIRE).
        OPTIMIZER_CONFIGS: Optional[Dict[str, Any]] = None, other kwargs of optimizer.
        STEPLENGTH: float = 0.05, the step length of the optimization.
        E_THRESHOLD: float = 1e-3, the energy threshold of convergence.
        F_THRESHOLD: float = 0.05, the force threshold of convergence.
        MAXITER: int = 100, the maximum number of optimization iterations.
        REQUIRE_GRAD: bool = False, bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs).

    References:
        [1] G. Henkelman and H. Jónsson, J. Chem. Phys. 2000, 113, 9901-9904.
    """

    def __init__(
            self,
            config_file: str,
            data_type: Literal['pyg', 'dgl'] = 'pyg',
    ) -> None:
        super().__init__(config_file)

        __algos = {'CI-NEB': CI_NEB}
        self.config_file = config_file
        assert data_type in {'pyg', 'dgl'}, f'Invalid data type {data_type}. It must be "pyg" or "dgl".'
        self.data_type = data_type
        self.reload_config(config_file)
        if self.VERBOSE: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None

        if self.NEB is None:
            raise RuntimeError('Nudged Elastic Band Configs was NOT Set.')
        self.NEB: Dict

        try:
            self.NEB_ALGO = __algos.get(self.NEB['ALGO'], CI_NEB)  # The Main Function of MD. MD_config is its parameters.
        except KeyError:
            raise NotImplementedError(f'Unknown Algorithm {self.NEB['ALGO']}.')
        self.require_grad = self.NEB.get('REQUIRE_GRAD', False)
        # check modes
        self.N_IMAGES = self.NEB['N_IMAGES']
        self.NEB_config = {
            'N_images': self.NEB['N_IMAGES'],
            'spring_const': float(self.NEB.get('SPRING_CONST', 1.)),
            'optimizer': self.NEB.get('OPTIMIZER', 'FIRE'),
            'optimizer_configs': self.NEB.get('OPTIMIZER_CONFIGS', None),
            'steplength': self.NEB.get('STEPLENGTH', 0.2),
            'E_threshold': float(self.NEB.get('E_THRESHOLD', 1e-3)),
            'F_threshold': float(self.NEB.get('F_THRESHOLD', 0.05)),
            'maxiter': int(self.NEB.get('MAXITER', 100)),
            'device': self.DEVICE,
            'verbose': self.VERBOSE
        }

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
        self.n_samp = len(self.TRAIN_DATA['dataIS'])  # sample number
        self.n_batch = math.ceil(self.n_samp / self.BATCH_SIZE)  # total batch number per epoch

        try:
            # I/O
            if self.VERBOSE > 0:
                self.logout_task_information(_model, 'NEB', self.NEB_config, self.n_samp)

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
                def get_indx(data):
                    _indx = getattr(data, 'idx', None)
                    # To correctly manage the names after NEB opt. There names will be [[`idx`], [`idx`], ...] (repeat N_images times).
                    if (_indx is not None) and isinstance(_indx[0], list):
                        _indx = [_[0] + f'_{iii}' for iii, _ in enumerate(_indx)]

                    return _indx

                def get_batch_size(data):
                    return len(data)

                def get_batch_indx(data):
                    return [len(dat.pos) for dat in data.to_data_list()]

                def get_cell_vec(data):
                    return data.cell.numpy(force=True)

                def get_atomic_number(data):
                    return data.atomic_numbers.unsqueeze(0)

                def get_fixed_mask(data):
                    mask = getattr(data, 'fixed', None)
                    if mask is not None:
                        mask = mask.unsqueeze(0)
                    return mask

                def rebatched_graph(single_graph, N_images):
                    """ expand batches """
                    graph = self.pygBatch.from_data_list([single_graph] * (N_images + 2))
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

                def get_fixed_mask(data):
                    return data.nodes['atom'].data.get('fix', None)

            # Instantiate NEB ALGO class
            neb_ops = self.NEB_ALGO(**self.NEB_config)
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, **self._data_loader_configs)
            if getattr(val_set, '_LOADER_TYPE', None) != 'ISFS':
                __err_msg = f'Data loader of ConstrainedMolecularDynamics requires ISFS, but got {getattr(val_set, '_LOADER_TYPE', None)}'
                if self.VERBOSE: self.logger.error(__err_msg)
                raise ValueError(__err_msg)
            n_c = 1  # running batch now
            for dataIS, dataFS in val_set:
                try:
                    # get basic information of IS.
                    batch_size = get_batch_size(dataIS)
                    cells = get_cell_vec(dataIS)
                    # Check Batch
                    if batch_size != get_batch_size(dataFS):
                        __err_msg = f'The batch size of {dataIS} and {dataFS} do not match: {get_batch_size(dataIS)} != {get_batch_size(dataFS)}'
                        if self.VERBOSE: self.logger.error(__err_msg)
                        raise RuntimeError(__err_msg)
                    if batch_size <= 0:
                        if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
                    elif batch_size > 1:
                        if self.VERBOSE: self.logger.error(f'Constrained MD do not support batched calculation yet. You should set BATCH_SIZE to 1.')
                        raise RuntimeError(f'Constrained MD do not support batched calculation yet. You should set BATCH_SIZE to 1.')
                    if self.data_type == 'pyg':
                        batch_indx = [len(dat.pos) for dat in dataIS.to_data_list()]
                        _check_batch_indx = [len(dat.pos) for dat in dataFS.to_data_list()]
                    elif self.data_type == 'dgl':
                        batch_indx = dataIS.batch_num_nodes('atom')
                        _check_batch_indx = dataFS.batch_num_nodes('atom')
                    else:
                        raise RuntimeError(f'Data type {self.data_type} is not supported.')
                    # NEB
                    if self.VERBOSE > 0:
                        self.logger.info('*' * 89)
                        self.logger.info(f'Running Batch {n_c}.')
                        self.logger.info(f'Structure names: {get_indx(dataIS)}\n')
                        cell_str = np.array2string(
                            cells, **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")  # TODO, Now it supports pygData and DGLGraph.
                        self.logger.info(f'Cell Vectors:\n{cell_str}')
                        self.logout_element_information(get_atomic_number(dataIS), batch_indx)

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
                    # rebatch data
                    origin_elem_tensor = get_atomic_number(dataIS).squeeze(0)
                    # creat fixation
                    if self.FIXATIONS is not None:
                        fixed_atom_tensor = th.ones_like(X_is, dtype=th.int8)
                        fix_list = self.fixation_resolve()
                        for fix_info in fix_list:
                            fixed_atom_tensor = atom_fix_selector(
                                fixed_atom_tensor,
                                origin_elem_tensor,
                                X_is,
                                **fix_info
                            )
                        fixed_atom_tensor = fixed_atom_tensor.repeat((self.N_IMAGES, 1))
                    else:
                        fixed_atom_tensor = get_fixed_mask(dataIS)
                    dataIS = rebatched_graph(dataIS, self.N_IMAGES)

                    # run
                    _energy, _X, out_grad = neb_ops.run(
                        model_wrap.Energy,
                        X_is,
                        X_fs,
                        grad_func=model_wrap.Grad,
                        func_args=(dataIS,), grad_func_args=(dataIS,),
                        is_grad_func_contain_y=False,
                        require_grad=self.require_grad,
                        output_grad = True,
                        fixed_atom_tensor=fixed_atom_tensor,  # TODO, The Selective Dynamics.
                    )
                    _energy.detach_()
                    _X.detach_()
                    out_grad.detach_()
                    idx = get_indx(dataIS)

                    self.dumper.collect(
                        batch_indx,
                        idx,
                        get_atomic_number(dataIS).squeeze(0),
                        get_cell_vec(dataIS),
                        _X.flatten(0, 1),
                        fixed_atom_tensor,
                        _energy,
                        - out_grad.flatten(0, 1),
                    )

                    # Print info
                    if self.VERBOSE > 0:
                        self.logger.info(f'Batch {n_c} done.')
                    n_c += 1
                    if self.SAVE_PREDICTIONS:
                        t_save = time.perf_counter()
                        with _LoggingEnd(self.log_handler):
                            if self.VERBOSE: self.logger.info(f'SAVING RESULTS TO {self.PREDICTIONS_SAVE_FILE} ...')
                        self.dumper.flush()
                        if self.VERBOSE: self.logger.info(f'Done. Saving Time: {time.perf_counter() - t_save:<.4f}')

                except Exception as e:
                    self.logger.warning(f'WARNING: An error occurred in {n_c}th batch. Error: {e}.')
                    if self.VERBOSE > 0:
                        excp = traceback.format_exc()
                        self.logger.warning(f"Traceback:\n{excp}")
                    n_c += 1

            if self.VERBOSE: self.logger.info(f'NEB Search Done. Total Time: {time.perf_counter() - time_tol:<.4f}')

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
