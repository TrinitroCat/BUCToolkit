#  Copyright (c) 2026.4.24, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: BaseMotion.py
#  Environment: Python 3.12
import logging
import os
import sys
import warnings
from typing import Any, Tuple, Dict, Callable, List
from itertools import accumulate

import torch as th
import numpy as np

from BUCToolkit.BatchStructures.StructuresIO import structures_io_dumper
from BUCToolkit.utils.setup_loggers import has_any_handler, clear_all_handlers
from BUCToolkit.utils._print_formatter import SCIENTIFIC_ARRAY_FORMAT, STRING_ARRAY_FORMAT, FLOAT_ARRAY_FORMAT


class BaseIO:
    """
    Base class for I/O operations.
    """
    def __init__(self, output_file: str|None = None) -> None:
        self.logger = None
        self.log_handler = None
        self.output_file = str(output_file) if output_file is not None else None
        self.dumper = structures_io_dumper(
            path=self.output_file,
            mode='x',
        )  # as the default. One can use `reset

    def init_logger(self, logger_name: str):
        # logging
        if self.logger is None:
            # cut off propagation
            supreme_name = logger_name.split('.')[0]
            top_logger = logging.getLogger(supreme_name)
            top_logger.propagate = False
            # set true logger
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            if not has_any_handler(top_logger):
                self.log_handler = logging.StreamHandler(sys.stdout)
                self.log_handler.setLevel(logging.INFO)
                self.log_handler.setFormatter(formatter)
                top_logger.addHandler(self.log_handler)
        else:
            warnings.warn('Logger has already initialized. Nothing will be done.', RuntimeWarning)

    def reset_logger_handler(self, handler: str|logging.StreamHandler|logging.FileHandler, level: int = logging.INFO):
        """
        Clear all logging handlers including current logger and its ancestors, and reset one.
        Args:
            handler: the new handler.
            level: the logger handler output level.

        Returns:

        """
        clear_all_handlers(self.logger)
        # redirect to supreme logger
        self.logger.setLevel(level)
        top_logger = self.logger
        while top_logger.parent and top_logger.propagate:
            top_logger = top_logger.parent
            top_logger.setLevel(level)

        top_logger.setLevel(level)

        formatter = logging.Formatter('%(message)s')
        if isinstance(handler, logging.Handler):
            self.log_handler = handler
        elif isinstance(handler, str):
            output_path = os.path.dirname(handler)
            # check whether path exists
            if not os.path.isdir(output_path): os.makedirs(output_path)
            # set log handler
            self.log_handler = logging.FileHandler(handler, 'w', delay=True)
        else:
            raise TypeError("handler must be a string path or a logging.Handler instance")

        self.log_handler.setLevel(level)
        self.log_handler.setFormatter(formatter)
        top_logger.addHandler(self.log_handler)

    def reset_dumper(self, dumper: Any) -> None:
        """
        Reset a new dumper and delete the old one.
        Args:
            dumper: A dumper object that satisfied the protocol as `_ArrayDumperPlaceHolder` class.

        Returns:

        """
        if self.output_file is not None:
            self.dumper.close()
            del self.dumper
            self.dumper = dumper
        else:
            self.logger.error(
                "ERROR: No output file specified. Hence, resetting dumper is meaningless.\n"
                "'reset_dumper': Operation REFUSED."
            )

class BaseMotion(BaseIO):
    """
    Base class for all processes that atoms move & evolution.
    """

    def __init__(self, output_file: str|None = None):
        super().__init__(output_file)

    @staticmethod
    def handle_motion_mask(
            X,
            fixed_atom_tensor,
    ):
        """
        normalize format of atom_masks
        Returns: the standardized atom_masks

        """
        if fixed_atom_tensor is None:
            fixed_atom_tensor = th.ones_like(X, device=X.device)
        else:
            fixed_atom_tensor = fixed_atom_tensor.broadcast_to(X.shape)
        if fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(X.device)
        else:
            raise RuntimeError(f'The shape of fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')

        return atom_masks

    @staticmethod
    def handle_grad_func(
            grad_func: Callable[[th.Tensor, Any, ...], th.Tensor] | None,
            is_grad_func_contain_y: bool,
            require_grad: bool,
            **kwargs
    ):
        """

        Returns: grad_func_, require_grad, is_grad_func_contain_y

        """
        if grad_func is None:
            is_grad_func_contain_y = True
            require_grad = True
            def grad_func_(x, y, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                _g = th.autograd.grad(y, x, grad_shape)
                return _g[0]
        else:
            grad_func_ = grad_func

        return grad_func_, require_grad, is_grad_func_contain_y

    @staticmethod
    def handle_batch_indices(
            batch_indices,
            n_batch,
            device
    ):
        r"""
        Calculating `n_true_batch`, `batch_tensor`, `batch_scatter`, and `batch_slice_indx` from input batch_indices.

        Args:
            batch_indices: input batch_indices that each element is the atom number of each sample: [n_0, n_1, ..., n_N].
            n_batch: the length of 1st dimension of X.
            device: torch device

        Returns: n_true_batch, batch_indices, batch_tensor, batch_scatter, batch_slice_indx;
            n_true_batch: the true batch size
            batch_indices: the batch_indices in List format
            batch_tensor: the batch_indices in torch.Tensor format
            batch_scatter: the batch indices in the form of Tensor[0, 0, 0, ..., 1, 1, ..., N - 1]
            batch_slice_indx: the batch indices in the form of ptr List[0, n_0, n_0 + n_1, ..., \sum n_i]
        """
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (List, Tuple)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            batch_slice_indx = [0] + list(accumulate(batch_indices))  # convert n_atom of each batch into split point of each batch
            batch_tensor = th.as_tensor(batch_indices, device=device)  # the tensor version of batch_indices which is a List.
            batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=device),
                batch_tensor,
                dim=0
            )  # scatter mask of the int tensor with the same shape as X.shape[1], which the data in one batch have one index.
            n_true_batch = len(batch_indices)  # the true batch size for irregular batches
        else:
            n_true_batch = n_batch
            batch_tensor = None
            batch_scatter = None
            batch_slice_indx = None

        return n_true_batch, batch_indices, batch_tensor, batch_scatter, batch_slice_indx

    @staticmethod
    def handle_arrays_print(
            logger: logging.Logger | Any,
            batch_indices: List[int],
            batch_slice_indx: List[int],
            arrays: List[List[th.Tensor]] | Tuple[Tuple[th.Tensor, ...]],
            array_names: List[List[str]] | Tuple[Tuple[str, ...]],
            verbose: int
    ):
        """
        Logging function for printing arrays with corresponding names controlled by the verbosity level.
        Args:
            logger: logger object
            batch_indices: input batch_indices that each element is the atom number of each sample
            batch_slice_indx: the batch_indices in ptr slice format
            arrays: input arrays. Format: [[tensors11, tensors12, ...], [tensors21, ...], ...],
                the i-th List in the outer list corresponds to the i-th verbosity level to log,
                and the tensors in the inner list will be all logged.
            array_names: input arrays names. Format: [[name11, name12, ...], [name21, ...], ...],
            verbose: verbosity level

        Returns:

        """
        if len(arrays) != len(array_names):
            raise ValueError(f'arrays and array_names must have the same length, but got {len(arrays)} and {len(array_names)}.')
        if batch_indices is not None:
            for v_lev in range(len(arrays)):
                if verbose > v_lev + 1:  # "+ 1" is a fixed offset to make that `verbose < 2` does not log large arrays.
                    for na, arr in enumerate(arrays[v_lev]):
                        X_np = arr.numpy(force=True)
                        X_tup = np.split(X_np, batch_slice_indx[1:-1], axis=1)
                        logger.info(f" {array_names[v_lev][na]}:\n")
                        X_str = [
                            np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                            for xi in X_tup
                        ]
                        for x_str in X_str: logger.info(f'{x_str}\n')
                else:
                    break  # logging verbosity level higher than input verbose, thus directly break to avoid useless loop
        else:
            for v_lev in range(len(arrays)):
                if verbose > v_lev + 1:  # "+ 1" is a fixed offset
                    for na, arr in enumerate(arrays[v_lev]):
                        X_tup = (arr.numpy(force=True),)
                        logger.info(f" {array_names[v_lev][na]}:\n")
                        X_str = [
                            np.array2string(xi, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]", " ")
                            for xi in X_tup
                        ]
                        for x_str in X_str: logger.info(f'{x_str}\n')
                else:
                    break

    def _calc_EF(
            self,
            X: th.Tensor,
            func: Callable[[th.Tensor, Any, ...], th.Tensor],
            func_args: Tuple,
            func_kwargs: Dict,
            grad_func_: Callable[[th.Tensor, Any, ...], th.Tensor],
            grad_func_args: Tuple,
            grad_func_kwargs: Dict,
            require_grad: bool,
            is_grad_func_contain_y: bool,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Calculate the energy and forces. In fact, it is function value y and its NEGATIVE gradient -g.

        Returns: Tuple[th.Tensor, th.Tensor], energy and forces.

        """
        y, g = self._calc_y_grad(
            X,
            func,
            func_args,
            func_kwargs,
            grad_func_,
            grad_func_args,
            grad_func_kwargs,
            require_grad,
            is_grad_func_contain_y,
        )

        return y, g.neg_()

    @th.compiler.disable
    def _calc_y_grad(
            self,
            X: th.Tensor,
            func: Callable[[th.Tensor, Any, ...], th.Tensor],
            func_args: Tuple,
            func_kwargs: Dict,
            grad_func_: Callable[[th.Tensor, Any, ...], th.Tensor],
            grad_func_args: Tuple,
            grad_func_kwargs: Dict,
            require_grad: bool,
            is_grad_func_contain_y: bool,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Calculate the function value y and the corresponding gradient of the y.

        Returns: Tuple[th.Tensor, th.Tensor], y and the gradient of the y

        """
        with th.set_grad_enabled(require_grad):
            X.requires_grad_(require_grad)
            y = func(X, *func_args, **func_kwargs)
            if is_grad_func_contain_y:
                g = grad_func_(X, y, *grad_func_args, **grad_func_kwargs)
            else:
                g = grad_func_(X, *grad_func_args, **grad_func_kwargs)
        return y, g
