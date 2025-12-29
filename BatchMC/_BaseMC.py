#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseMC.py
#  Environment: Python 3.12

import logging
import sys
from itertools import accumulate
from typing import Dict, Any, Optional, Sequence, Tuple, List
import time
import warnings

import numpy as np
import torch as th
from torch import nn
from BM4Ckit.BatchOptim._utils._warnings import FaildToConvergeWarning
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BM4Ckit.utils.scatter_reduce import scatter_reduce


class _BaseMC:
    def __init__(
            self,
            iter_scheme: str,
            maxiter: int = 100,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ) -> None:
        r"""
        A Base Framework of Algorithm for optimization.
        
        Args:
            maxiter: int, max iterations.
            device: The device that program runs on.
            verbose: amount of print information.
        
        Method:
            run: running the main optimization program.

        """
        self.batch_scatter = None
        self.batch_tensor = None
        self.atom_masks = None
        self.T_now = None
        warnings.filterwarnings('always', category=FaildToConvergeWarning)
        warnings.filterwarnings('always', )

        self.iterform = iter_scheme
        self.verbose = verbose
        self.device = device
        self.maxiter = int(maxiter)

        self.n_batch, self.n_atom, self.n_dim = None, None, None
        self.is_concat_X = False   # whether the output of `func` was concatenated.

        # logger
        self.logger = logging.getLogger('Main.OPT')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if not self.logger.hasHandlers():
            log_handler = logging.StreamHandler(sys.stdout, )
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            batch_indices: None | List[int] | Tuple[int, ...] | th.Tensor = None,
    ) -> Tuple[th.Tensor, th.Tensor] | Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Run MC Algorithm.

        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func.
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            fixed_atom_tensor: Optional[th.Tensor], the indices of X that fixed.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]

        Return:
            min func: Tensor(n_batch, ), the minimum of func.
            argmin func: Tensor(X.shape), the X corresponds to min func.
            grad of argmin func: Tensor(X.shape), only output when `output_grad` == True. The gradient of X corresponding to minimum.
        """
        t_main = time.perf_counter()
        if func_kwargs is None:
            func_kwargs = dict()
        # Check batch indices; irregular batch
        if isinstance(X, th.Tensor):
            n_batch, n_atom, n_dim = X.shape
        else:
            raise TypeError(f'`X` must be torch.Tensor, but occurred {type(X)}.')
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (Tuple, List)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be Sequence[int] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            n_inner_batch = len(batch_indices)
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            batch_indx_dict = {
                i: slice(_, batch_slice_indx[i + 1]) for i, _ in enumerate(batch_slice_indx[:-1])
            }  # dict of {batch indx: split point slice}
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)
            batch_tensor = self.batch_tensor
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )
        else:
            n_inner_batch = 1
            batch_indx_dict = dict()
            batch_tensor = None
        # initialize vars
        maxiter = self.maxiter
        n_batch, n_atom, n_dim = X.shape
        self.n_batch, self.n_atom, self.n_dim = n_batch, n_atom, n_dim

        # Selective dyamics
        if fixed_atom_tensor is None:
            self.atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            self.atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'The shape of fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        #atom_masks = atom_masks.flatten(-2, -1).unsqueeze(-1)  # (n_batch, n_atom*n_dim, 1)
        # other check
        if (not isinstance(maxiter, int)) or (maxiter <= 0):
            raise ValueError(f'Invalid value of maxiter: {maxiter}. It would be an integer greater than 0.')

        # set variables device
        if isinstance(func, nn.Module):
            func = func.to(self.device)
            func.eval()
            func.zero_grad()
        X = X.to(self.device)

        # initialize
        t_st = time.perf_counter()
        self.initialize_algo_param()
        ptlist = [X[:, None, :, 0].numpy(force=True)]  # for converged samp, stop calc., test <<<
        if self.verbose:
            self.logger.info('-' * 100)
            self.logger.info(f'Iteration Scheme: {self.iterform}')
        self.logger.info('-' * 100)
        # MAIN LOOP
        with th.no_grad():
            energies: th.Tensor = func(X, *func_args, **func_kwargs)
            delta_E = th.zeros_like(energies)
            # note: irregular tensor regularized by concat. thus n_batch of X shown as 1, but y has shape of the true batch size.
            if energies.shape[0] != self.n_batch:
                assert batch_indices is not None, (f"batch indices is None "
                                                   f"while shape of model output ({energies.shape}) does not match batch size ({self.n_batch}).")
                assert energies.shape[0] == n_inner_batch, f"shape of output ({energies.shape}) does not match given batch indices"
                self.is_concat_X = True
            else:
                self.is_concat_X = False

            for numit in range(maxiter):
                # Print information / Verbose
                if self.verbose > 0:
                    self.logger.info(f"ITERATION    {numit:>5d}\n "
                                     f"delta E:     {np.array2string(delta_E.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Energies:    {np.array2string(energies.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n "
                                     f"Temperature: {self.T_now:.3e}\n "
                                     f"TIME:        {time.perf_counter() - t_st:>6.4f} s")
                if self.verbose > 1:
                    # split batches if specified batch
                    if batch_indices is not None:
                        X_np = X.numpy(force=True)
                        X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                    else:
                        X_tup = (X.numpy(force=True),)
                    self.logger.info(f" Coordinates:\n")
                    X_str = [
                        np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                        for xi in X_tup
                    ]
                    [self.logger.info(f'{x_str}\n') for x_str in X_str]

                t_st = time.perf_counter()

                # update X & energy
                X_old = X.clone()
                energies_old = energies.clone()
                energies, delta_E, X = self._update_X(func, func_args, func_kwargs, energies_old, X) # (n_batch, n_atom*3, 1)
                X_diff = X - X_old

                # update algo. parameters.
                self._update_algo_param(numit, X_diff)

                # Check NaN
                if th.any(energies != energies): raise RuntimeError(f'NaN Occurred in output: {energies}')

                ptlist.append(X[:, None, :, 0].numpy(force=True))  # test <<<

        if self.verbose > 0:
            self.logger.info(
                '-' * 100 + f'\nMAIN LOOP Done. Total Time: {time.perf_counter() - t_main:<.4f} s\n'
            )
            if self.verbose < 2:  # verbose = 1, brief mode only output last step coords.
                # split batches if specified batch
                if batch_indices is not None:
                    X_np = X.numpy(force=True)
                    X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                else:
                    X_tup = (X.numpy(force=True),)
                self.logger.info(f" Final Coordinates:\n")
                X_str = [
                    np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                    for xi in X_tup
                ]
                [self.logger.info(f'{x_str}\n') for x_str in X_str]
        # output
        return energies, X  , ptlist  # test <<<

    def initialize_algo_param(self):
        """
        Override this method to initialize attribute variables for self._update_direction.
        Examples:

        Returns: None
        """
        raise NotImplementedError

    def _update_X(
            self,
            func,
            func_args,
            func_kwargs,
            energies_old: th.Tensor,
            X: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Override this method to implement X update algorithm.
        Args:
            func: function
            func_args:
            func_kwargs:
            X: (n_batch, n_atom, 3), the independent vars X.

        Returns:
           energies, delta_E, X
        """
        raise NotImplementedError

    def _update_algo_param(self, i: int, displace: th.Tensor) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.
        Args:
            i: iteration step now.
            displace: (n_batch, n_atom*3, 1), the displacement of X at this step. displace = step-length * p

        Returns: None
        """
        pass
