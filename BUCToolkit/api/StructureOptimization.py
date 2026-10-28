#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: StructureOptimization.py
#  Environment: Python 3.12
import logging
import math
import os
import time
import traceback
from typing import Any, Literal, Dict, List

import numpy as np
import torch as th
from torch import nn

from BUCToolkit import BatchOptim
from BUCToolkit.BatchOptim.minimize import CG, QN, FIRE
from BUCToolkit.BatchOptim.TS import Dimer, KrylovNewton, KrylovDynamics
from BUCToolkit.api._io import _BaseAPI, _Model_Wrapper_dgl, PygBatchUpdater
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT
from BUCToolkit.utils._Element_info import ATOMIC_NUMBER
from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.utils.ElemListReduce import elem_list_reduce
from BUCToolkit.utils.setup_loggers import has_any_handler


class StructureOptimization(_BaseAPI):
    """
    The class of structure optimization for relaxation and transition-state
    searches (DIMER, KRYLOV, and KRYDYN). Users need to set the dataset and
    dataloader manually.

    Args:
        config_file: the path of input file.
        data_type: graph data type. 'pyg' for torch-geometric BatchData, 'dgl' for dgl DGLGraph.

    Input file parameters:
        # For relaxation tasks:
        RELAXATION:
          ALGO: Literal[CG, BFGS, FIRE], the optimization algo.
          REQUIRE_GRAD: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs). Default: False.
          ITER_SCHEME: Literal['PR+', 'FR', 'PR', ], only use for ALGO=CG, the iteration scheme of CG. Default: PR+.
          E_THRES: float, threshold of Energy difference (eV). Default: 1e-4.
          F_THRES: float, threshold of max Force (eV/Ang). Default: 5e-2.
          MAXITER: int, the maximum iteration numbers. Default: 300.
          STEPLENGTH: float, the initial step length for line search. Default: 0.5.
          USE_BB: bool, if True, use Barzilai-Borwein steplength (i.e., BB1 or long BB) as initial steplength instead of fixed `STEPLENGTH`.

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

        # For transition state tasks:
        TRANSITION_STATE:
            ALGO: Literal[DIMER, KRYLOV, KRYDYN], the transition-state algorithm.
                DIMER uses the modified Dimer method, KRYLOV uses KrylovNewton, and
                KRYDYN uses KrylovDynamics.

            # Parameters shared by DIMER, KRYLOV, and KRYDYN:
            X_DIFF_ATTR: str, the name of the data attribute that seeds the initial
                direction guess, i.e., `X_diff = data.X_DIFF_ATTR`. If the attribute is
                absent or has an incompatible shape, a random direction is generated.
                An explicitly registered direction from `set_dimer_init_direction`
                takes precedence over this attribute.
            E_THRES: float, threshold of Energy difference (eV). Default: 1e-4 for
                DIMER, and 1e-3 for KRYLOV and KRYDYN.
            TORQ_THRES: float, the threshold of max torque (i.e., the eigenvector residual). Default: 1e-2.
            F_THRES: float, threshold of max Force (eV/Ang). Default: 5e-2.
            MAXITER_TRANS: int, the maximum iteration numbers of translation steps. Default: 300.
            DX: float, the step length of finite difference for Hessian-vector products. Default: 1.e-2.
            DUMP_QUANTITIES: tuple/list of str, quantities written to the optimizer dump.
                Default: ('Energy', 'X', 'Force').
            LOG_QUANTITIES: tuple/list of str, quantities printed by the logger.
                Default: ('Energy', 'E_diff', 'F_eps', 'X', 'X_grad').
            REQUIRE_GRAD: bool, if True, autograd will be turned on for the energy
                function. Default: False.

            # Parameters only for ALGO=DIMER:
            CURVATURE_THRES: float, the threshold of curvature for rotation convergence.
                Once the curvatures are less than `CURVATURE_THRES`, convergence will be
                viewed as reached. Default: -0.1.
            MAXITER_ROT: int, the maximum iteration numbers of rotation steps. Default: 10.
            MAX_STEPLENGTH: float, the maximum step length for dimer translations. Default: 0.5.

            # Parameters only for ALGO=KRYLOV or ALGO=KRYDYN:
            EIGEN_THRES: float, the residual threshold of the eigen solver. Default: 1e-2.
            MAXITER_EIG: int, the maximum iteration numbers of the eigen solver. It must be
                greater than 1 and greater than MORSE_INDEX. Default: 10.
            STEPLENGTH: float, the initial translation step length. KRYDYN also uses it as
                the initial FIRE-like time step. Default: 0.5.
            MORSE_INDEX: int, the number of unstable modes to follow. Default: 1.
            NEG_SPECTRA_CUTOFF: float, the cutoff used to clamp negative spectral modes.
                Default: 1e-2.
            POS_SPECTRA_CUTOFF: float, the cutoff used to clamp positive spectral modes.
                Default: 1e-2.

            # KRYLOV fixes `steplength_sheme` to 'trust_region' internally. The other
            # schemes are not exposed because they remain experimental.

    """

    def __init__(
            self,
            config_file: str,
            data_type: Literal['pyg', 'dgl'] = 'pyg',
            *args,
            **kwargs
    ) -> None:
        super().__init__(config_file)

        self.dX = None
        self._dimer_init_direction: th.Tensor | None = None
        self.config_file = config_file
        self._set_data_type(data_type)
        #if data_type == 'pyg':
        #    self.pygData = check_module('torch_geometric.data.batch')
        #else:
        #    self.pygData = None
        if self.VERBOSE: self.logger.info('Config File Was Successfully Read.')
        self.param = None
        self._has_load_data = False
        self._data_loader = None

        __optimiser_dict = {
            'CG': CG, 'BFGS': QN, 'FIRE': FIRE,
            'DIMER': Dimer, 'KRYLOV': KrylovNewton, 'KRYDYN': KrylovDynamics
        }
        if self.RELAXATION is not None:
            if self.TRANSITION_STATE is not None:
                self.logger.warning(
                    '** WARNING: Both `RELAXATION` and `TRANSITION_STATE` were set. Hence, `TRANSITION_STATE` will be ignored. **'
                )
                self.TRANSITION_STATE = None
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
            self.Stru_Opt = __optimiser_dict[self.RELAXATION['ALGO']]
            # judge whether ALGO is minimization algo.
            self.require_grad = self.RELAXATION.get('REQUIRE_GRAD', False)
            if self.RELAXATION['ALGO'] not in {'CG', 'BFGS', 'FIRE'}:
                self.logger.error(f'{self.RELAXATION['ALGO']} is NOT RELAXATION ALGORITHM!!!')
                raise ValueError(f'{self.RELAXATION['ALGO']} is NOT RELAXATION ALGORITHM!!!')
            if self.RELAXATION['ALGO'] != 'FIRE':
                self.Stru_Opt_config = {iterschem: _iterschem,
                                        'E_threshold': float(self.RELAXATION.get('E_THRES', 1.e-4)),
                                        'F_threshold': float(self.RELAXATION.get('F_THRES', 0.05)),
                                        'maxiter': int(self.RELAXATION.get('MAXITER', 300)),
                                        'linesearch': self.RELAXATION.get('LINESEARCH', 'Backtrack'),
                                        'linesearch_maxiter': int(self.RELAXATION.get('LINESEARCH_MAXITER', 10)),
                                        'linesearch_thres': float(self.RELAXATION.get('LINESEARCH_THRES', 0.05)),
                                        'linesearch_factor': float(self.RELAXATION.get('LINESEARCH_FACTOR', 0.6)),
                                        'use_bb': bool(self.RELAXATION.get('USE_BB', True)),
                                        'steplength': float(self.RELAXATION.get('STEPLENGTH', 0.5)),
                                        'device': self.DEVICE,
                                        'verbose': self.VERBOSE}
            else:
                self.Stru_Opt_config = {'E_threshold': float(self.RELAXATION.get('E_THRES', 1.e-4)),
                                        'F_threshold': float(self.RELAXATION.get('F_THRES', 0.05)),
                                        'maxiter': int(self.RELAXATION.get('MAXITER', 300)),
                                        'steplength': float(self.RELAXATION.get('STEPLENGTH', 0.5)),
                                        'alpha': float(self.RELAXATION.get('ALPHA', 0.1)),
                                        'alpha_fac': float(self.RELAXATION.get('ALPHA_FAC', 0.99)),
                                        'fac_inc': float(self.RELAXATION.get('FAC_INC', 1.1)),
                                        'fac_dec': float(self.RELAXATION.get('FAC_DEC', 0.5)),
                                        'N_min': int(self.RELAXATION.get('N_MIN', 5)),
                                        'mass': float(self.RELAXATION.get('MASS', 20)),
                                        'device': self.DEVICE,
                                        'verbose': self.VERBOSE}
        elif self.TRANSITION_STATE is not None:
            self.Stru_Opt = __optimiser_dict[self.TRANSITION_STATE['ALGO']]
            self.require_grad = self.TRANSITION_STATE.get('REQUIRE_GRAD', False)
            self.x_diff_attr = self.TRANSITION_STATE.get('X_DIFF_ATTR', 'x_dimer')
            if self.TRANSITION_STATE['ALGO'] == 'DIMER':
                self.Stru_Opt_config = {
                    'E_threshold': self.TRANSITION_STATE.get('E_THRES', 1.e-4),
                    'Torque_thres': self.TRANSITION_STATE.get('TORQ_THRES', 1e-2),
                    'Curvature_thres': self.TRANSITION_STATE.get('CURVATURE_THRES', -0.1),
                    'F_threshold': self.TRANSITION_STATE.get('F_THRES', 0.05),
                    'maxiter_trans': int(self.TRANSITION_STATE.get('MAXITER_TRANS', 300)),
                    'maxiter_rot': int(self.TRANSITION_STATE.get('MAXITER_ROT', 10)),
                    'max_steplength': self.TRANSITION_STATE.get('MAX_STEPLENGTH', 0.5),
                    'dx': self.TRANSITION_STATE.get('DX', 1.e-2),
                    'device': self.DEVICE,
                    'verbose': self.VERBOSE,
                    'dump_quantities': self.TRANSITION_STATE.get('DUMP_QUANTITIES', ('Energy', 'X', 'Force')),
                    'log_quantities': self.TRANSITION_STATE.get('LOG_QUANTITIES', ('Energy', 'E_diff', 'F_eps', 'X', 'X_grad')),
                }

            elif self.TRANSITION_STATE['ALGO'] == 'KRYLOV':
                # KrylovNewton hardcodes its iteration scheme in `__init__`.
                self.Stru_Opt_config = {
                    'E_threshold': float(self.TRANSITION_STATE.get('E_THRES', 1.e-3)),
                    'Torque_thres': float(self.TRANSITION_STATE.get('TORQ_THRES', 1.e-2)),
                    'Eigen_thres': float(self.TRANSITION_STATE.get('EIGEN_THRES', 1.e-2)),
                    'F_threshold': float(self.TRANSITION_STATE.get('F_THRES', 0.05)),
                    'maxiter_trans': int(self.TRANSITION_STATE.get('MAXITER_TRANS', 300)),
                    'maxiter_eig': int(self.TRANSITION_STATE.get('MAXITER_EIG', 10)),
                    'steplength': float(self.TRANSITION_STATE.get('STEPLENGTH', 0.5)),
                    'steplength_sheme': 'trust_region',  # This method is the fully tested, others are experimental
                    'dx': float(self.TRANSITION_STATE.get('DX', 1.e-2)),
                    'device': self.DEVICE,
                    'verbose': self.VERBOSE,
                    'morse_index': int(self.TRANSITION_STATE.get('MORSE_INDEX', 1)),
                    'neg_spectra_cutoff': float(self.TRANSITION_STATE.get('NEG_SPECTRA_CUTOFF', 1.e-2)),
                    'pos_spectra_cutoff': float(self.TRANSITION_STATE.get('POS_SPECTRA_CUTOFF', 1.e-2)),
                    'dump_quantities': self.TRANSITION_STATE.get('DUMP_QUANTITIES', ('Energy', 'X', 'Force')),
                    'log_quantities': self.TRANSITION_STATE.get('LOG_QUANTITIES', ('Energy', 'E_diff', 'F_eps', 'X', 'X_grad')),
                }

            elif self.TRANSITION_STATE['ALGO'] == 'KRYDYN':
                # KrylovDynamics hardcodes its iteration scheme in `__init__`.
                self.Stru_Opt_config = {
                    'E_threshold': float(self.TRANSITION_STATE.get('E_THRES', 1.e-3)),
                    'Torque_thres': float(self.TRANSITION_STATE.get('TORQ_THRES', 1.e-2)),
                    'Eigen_thres': float(self.TRANSITION_STATE.get('EIGEN_THRES', 1.e-2)),
                    'F_threshold': float(self.TRANSITION_STATE.get('F_THRES', 0.05)),
                    'maxiter_trans': int(self.TRANSITION_STATE.get('MAXITER_TRANS', 300)),
                    'maxiter_eig': int(self.TRANSITION_STATE.get('MAXITER_EIG', 10)),
                    'steplength': float(self.TRANSITION_STATE.get('STEPLENGTH', 0.5)),
                    'dx': float(self.TRANSITION_STATE.get('DX', 1.e-2)),
                    'device': self.DEVICE,
                    'verbose': self.VERBOSE,
                    'morse_index': int(self.TRANSITION_STATE.get('MORSE_INDEX', 1)),
                    'neg_spectra_cutoff': float(self.TRANSITION_STATE.get('NEG_SPECTRA_CUTOFF', 1.e-2)),
                    'pos_spectra_cutoff': float(self.TRANSITION_STATE.get('POS_SPECTRA_CUTOFF', 1.e-2)),
                    'dump_quantities': self.TRANSITION_STATE.get('DUMP_QUANTITIES', ('Energy', 'X', 'Force')),
                    'log_quantities': self.TRANSITION_STATE.get('LOG_QUANTITIES', ('Energy', 'E_diff', 'F_eps', 'X', 'X_grad')),
                }

        if (self.RELAXATION is None) and (self.TRANSITION_STATE is None):
            self.logger.error('** ERROR: Both `RELAXATION` and `TRANSITION_STATE` were NOT set. NO TASK HERE, BYE!!! **')
            raise RuntimeError('** ERROR: Both `RELAXATION` and `TRANSITION_STATE` were NOT set. NO TASK HERE, BYE!!! **')
        if self.SAVE_PREDICTIONS:
            self.Stru_Opt_config['output_file'] = self.PREDICTIONS_SAVE_FILE

    def set_dimer_init_direction(self, directions: th.Tensor):
        """
        Set the initial direction guess for transition-state searches.

        The registered direction is used by DIMER, KRYLOV, and KRYDYN, and takes
        precedence over the data attribute named by `X_DIFF_ATTR`.

        Args:
            directions: initial direction with shape (n_atom, 3) or
                (1, n_atom, 3). A batch whose coordinate shape differs from
                `directions` falls back to a random direction with a warning.

        Returns: None

        Raise:
            TypeError: if `directions` is not a torch.Tensor.
            ValueError: if `directions` is not 2D or 3D.

        """
        if not isinstance(directions, th.Tensor):
            raise TypeError(
                f'`directions` must be a torch.Tensor, but occurred {type(directions)}.'
            )
        if directions.ndim == 2:
            directions = directions.unsqueeze(0)
        elif directions.ndim != 3:
            raise ValueError(
                f'`directions` must be 2D or 3D, but got shape {directions.shape}.'
            )
        self._dimer_init_direction = directions

    def run(self, model, mode: Literal['minimize', 'ts']='minimize'):
        """
        Run structure optimization algorithms to search local minima or saddle point (TS).
        Parameters:
            model: the input model which is `uninstantiated` nn.Module class.
            mode: choose whether `minimize` or `ts`.
        """
        # check logger
        if not has_any_handler(self.logger): self.logger.addHandler(self.log_handler)
        # check vars
        if (self.START == 'resume') or (self.START == 1) or (self.START == 2):
            _chk_hyperparam, _chk_model_param, _chk_train_state = self._load_chk()
            hyperparam = self._checkpoint_hyperparameters(_chk_hyperparam)
        else:
            _chk_hyperparam, _chk_model_param, _chk_train_state = None, None, {}
            hyperparam = self._current_model_hyperparameters()
        _model = self._instantiate_model(model, hyperparam)
        if (self.START == 'resume') or (self.START == 1) or (self.START == 2):
            if self.param is None:
                _model.load_state_dict(_chk_model_param, strict=self.STRICT_LOAD)
            else:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
        elif self.START == 'from_scratch' or self.START == 0:
            self.logger.warning(
                '** WARNING: The model was not read the trained parameters from checkpoint file. I HOPE YOU KNOW WHAT YOU ARE DOING! **'
            )
            if self.param is not None:
                _model.load_state_dict(self.param, self.is_strict, self.is_assign)
        else:
            raise ValueError('Invalid START value. It should be "from_scratch" / 0 or "resume" / 1 ')
        if mode not in {'minimize', 'ts'}:
            raise ValueError(f'Invalid mode value. It should be "minimize" or "ts", but got {mode}')
        if (mode == 'minimize') and self.RELAXATION is None:
            self.logger.error('** ERROR: `mode` is "minimize" but the configs `RELAXATION` is None! NO TASK HERE, BYE!!! **')
            raise RuntimeError('** ERROR: `mode` is "minimize" but the configs `RELAXATION` is None! NO TASK HERE, BYE!!! **')
        elif (mode == 'ts') and self.TRANSITION_STATE is None:
            self.logger.error('** ERROR: `mode` is "ts" but the configs `TRANSITION_STATE` is None! NO TASK HERE, BYE!!! **')
            raise RuntimeError('** ERROR: `mode` is "ts" but the configs `TRANSITION_STATE` is None! NO TASK HERE, BYE!!! **')
        # model vars
        _model = _model.to(self.DEVICE)

        # preprocessing data # TODO
        if self._data_loader is None: raise RuntimeError('Please Set the DataLoader.')
        if not self._has_load_data: raise RuntimeError('Please Set the Data to Predict.')

        # initialize
        self.n_samp = len(self.TRAIN_DATA['data'])  # sample number
        self.n_batch = math.ceil(self.n_samp / self.BATCH_SIZE)  # total batch number per epoch

        optimizer = None
        # I/O
        try:
            if self.VERBOSE > 0:
                self.logout_task_information(_model, mode, self.Stru_Opt_config, self.n_samp, checkpoint_hyperparam=_chk_hyperparam, model_hyperparam=hyperparam)

            time_tol = time.perf_counter()
            _model.eval()
            # MAIN LOOP
            # define the model wrapper & batch size getter & cell vector getter for different data type
            if self.data_type == 'pyg':
                model_wrap = self._build_model_wrapper(_model, hyperparam=hyperparam)

                def get_batch_size(data):
                    return len(data)

                def get_cell_vec(data):
                    return data.cell.numpy(force=True)

                def get_atomic_number(data):
                    return data.atomic_numbers.unsqueeze(0)

                def get_indx(data):
                    _indx: Dict = getattr(data, 'idx', None)
                    return _indx

                def get_fixed_mask(data):
                    return data.fixed.unsqueeze(0)

                def get_batch_indx(data):
                    batch = getattr(data, 'batch', None)
                    if batch is not None:
                        n_batch_ = th.bincount(batch).tolist()
                    else:
                        n_batch_ = [len(dat.pos) for dat in data.to_data_list()]
                    return n_batch_

                def get_init_dX(data):
                    """ get dimer initial guess """
                    return getattr(data, self.x_diff_attr, None)

                self.__check_old: th.Tensor|None = None
                update_batch = PygBatchUpdater()
                update_batch_rot = PygBatchUpdater()

            else:
                model_wrap = self._build_model_wrapper(_model, hyperparam=hyperparam)

                def get_batch_size(data):
                    return data.num_nodes('atom')

                def get_cell_vec(data):
                    return data.nodes['cell'].data['cell'].numpy(force=True)

                def get_atomic_number(data):
                    return data.nodes['atom'].data['Z'].unsqueeze(0)

                def get_indx(data):
                    _indx: Dict = data.nodes['atom'].data
                    return _indx.get('idx', None)

                def get_fixed_mask(data):
                    return data.nodes['atom'].data.get('fix', None)

                def get_batch_indx(data):
                    return data.batch_num_nodes('atom').tolist()

                def get_init_dX(data):
                    """ get dimer initial guess """
                    return data.nodes['atom'].data['dx'].unsqueeze(0)

                def update_batch(
                        converge_check,
                        func_args,
                        func_kwargs,
                        grad_func_args,
                        grad_func_kwargs
                ):
                    # TODO <<<<<<<<<
                    pass

                update_batch_rot = None
                raise NotImplementedError

            optimizer = self.Stru_Opt(**self.Stru_Opt_config)
            if self.SAVE_PREDICTIONS:
                optimizer._HOLD_DUMPER = True
                optimizer.dumper.reset_args(
                    self.PREDICTIONS_SAVE_FILE,
                    mode='a',
                    cache_size=4096,
                )
            val_set: Any = self._data_loader(self.TRAIN_DATA, self.BATCH_SIZE, self.DEVICE, is_train=False, **self._data_loader_configs)
            n_c = 1  # number of cycles. each for-loop += 1.
            n_s = 0  # number of calculated samples. each sample in batches in each for-loop += 1.
            for val_data, val_label in val_set:
                try:  # Catch error in each loop & continue, instead of directly exit.
                    # to avoid get an empty batch
                    if get_batch_size(val_data) <= 0:
                        if self.VERBOSE: self.logger.info(f'An empty batch occurred. Skipped.')
                        continue
                    # batch indices
                    batch_indx:List = get_batch_indx(val_data)
                    # initial atom coordinates
                    if self.data_type == 'pyg':
                        X_init = val_data.pos.unsqueeze(0) if val_data.pos.dim() == 2 else val_data.pos
                    else:
                        _ = val_data.nodes['atom'].data['pos']
                        X_init = _.unsqueeze(0) if _.dim() == 2 else _
                        del _
                    # initialize X_diff
                    if mode == 'ts':
                        # An explicitly registered direction takes precedence over
                        # the data attribute named by `X_DIFF_ATTR`.
                        X_diff = self._dimer_init_direction
                        if X_diff is None:
                            X_diff = get_init_dX(val_data)
                        if X_diff is None:
                            self.logger.warning(
                                f'WARNING: No initial transition-state direction given. A random direction will be used.'
                            )
                        elif not isinstance(X_diff, th.Tensor):
                            X_diff = None
                            self.logger.warning(
                                f'WARNING: The transition-state direction `X_diff` is expected to be a torch.Tensor, but got {type(X_diff)}, '
                                f'so it will be reset randomly.'
                            )
                        else:
                            if X_diff.dim() == 2:
                                X_diff = X_diff.unsqueeze(0)
                            X_diff = X_diff.to(device=X_init.device, dtype=X_init.dtype)
                            if X_diff.shape != X_init.shape:
                                X_diff = None  # TS methods automatically create a random X_diff when input `X_diff = None`.
                                self.logger.warning(
                                    f'WARNING: The transition-state direction X_diff has different shape from `X_init`, so it will be reset randomly.'
                                )
                    else:
                        X_diff = None  # placeholder
                    # cells & fixations
                    CELL = get_cell_vec(val_data)
                    fixed_mask = get_fixed_mask(val_data)
                    # get id
                    idx = get_indx(val_data)
                    idx = idx if idx is not None else [f'Untitled{_}' for _ in range(n_s, n_s + len(batch_indx))]
                    n_s += len(batch_indx)
                    element_tensor = get_atomic_number(val_data)
                    atomic_number_rows = [
                        numbers.tolist()
                        for numbers in th.split(element_tensor.reshape(-1), batch_indx)
                    ]
                    optimizer.set_system_info(CELL, atomic_number_rows)
                    if self.VERBOSE > 0:
                        self.logger.info('*' * 89)
                        self.logger.info(f'Running Batch {n_c}.')
                        cell_str = np.array2string(
                            CELL, **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")
                        self.logger.info(f'Structure names: {idx}\n')
                        self.logger.info(f'Cell Vectors:\n{cell_str}\n')
                        # print structures titles with elements
                        self.logout_element_information(element_tensor, batch_indx)
                    # relax
                    with th.no_grad():
                        update_batch.initialize()
                        update_batch_rot.initialize()
                        if mode == 'minimize':
                            optimizer: BatchOptim.FIRE
                            optimizer.set_batch_updater(update_batch, update_batch_rot)
                            fin_ener, fin_x, fin_grad = optimizer.run(
                                func=model_wrap.Energy,
                                X=X_init,
                                grad_func=model_wrap.Grad,
                                func_args=(val_data,),
                                grad_func_args=(val_data,),
                                is_grad_func_contain_y=False,
                                output_grad=True,
                                require_grad = self.require_grad,
                                batch_indices=batch_indx,
                                fixed_atom_tensor=fixed_mask,
                            )
                        else:  # i.e., mode == 'ts'
                            optimizer: (
                                BatchOptim.Dimer
                                | BatchOptim.KrylovNewton
                                | BatchOptim.KrylovDynamics
                            )
                            optimizer.set_batch_updater(update_batch, update_batch_rot)
                            fin_ener, fin_x, fin_grad = optimizer.run(
                                func=model_wrap.Energy,
                                X=X_init,
                                X_diff=X_diff,
                                grad_func=model_wrap.Grad,
                                func_args=(val_data,),
                                grad_func_args=(val_data,),
                                is_grad_func_contain_y=False,
                                output_grad=True,
                                require_grad=self.require_grad,
                                batch_indices=batch_indx,
                                fixed_atom_tensor=fixed_mask,
                                # There is an arg of `extra_krylov_dim` for Krylov based method,
                                # but here set the default (i.e., 1) for simple.
                            )
                        #fin_ener = fin_ener.detach()#.squeeze(0)
                        #fin_x = fin_x.detach().squeeze(0)
                        #fin_grad = fin_grad.detach().squeeze(0)

                    # Print info
                    if self.VERBOSE > 0:
                        self.logger.info('-' * 100)
                    n_c += 1

                except Exception as e:
                    self.logger.warning(f'** WARNING: An error occurred in the {n_c}-th batch. Error: {e}. **')
                    if self.VERBOSE > 0:
                        excp = traceback.format_exc()
                        self.logger.warning(f"Traceback:\n{excp}")
                    n_c += 1

            if self.VERBOSE: self.logger.info(f'RELAXATION DONE. Total Time: {time.perf_counter() - time_tol:<.4f}')

        except Exception as e:
            if th.cuda.is_available(): th.cuda.synchronize()
            excp = traceback.format_exc()
            self.logger.exception(f'An ERROR occurred:\n\t{e}\nTraceback:\n{excp}')

        finally:
            if th.cuda.is_available(): th.cuda.synchronize()
            self.logger.removeHandler(self.log_handler)
            if isinstance(self.log_handler, logging.FileHandler):
                self.log_handler.close()
            if optimizer is not None:
                optimizer.dumper.close()

    def relax(self, model):
        """
        Run minimization algorithms to relax the structures.
        Alias of `self.run(model, mode='minimize')`.
        Args:
            model: the input model which is `uninstantiated` nn.Module class.

        """
        return self.run(model, mode='minimize')

    def transition_state(self, model):
        """
        Use single-point methods (DIMER, KRYLOV, or KRYDYN, selected by
        `TRANSITION_STATE.ALGO`) to search the transition states.
        Alias of `self.run(model, mode='ts')`.
        Args:
            model: the input model which is `uninstantiated` nn.Module class.
        """
        return self.run(model, mode='ts')

    def ts(self, model):
        """
        Alias of `self.transition_state`.
        """
        return self.transition_state(model)
