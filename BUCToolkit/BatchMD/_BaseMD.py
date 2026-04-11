""" Molecular Dynamics via Verlet algo. """
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseMD.py
#  Environment: Python 3.12

import sys
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import time
import math
import logging
import warnings
import os
import threading, queue

import torch as th
from torch import nn

import numpy as np

from BUCToolkit.utils._Element_info import MASS, N_MASS, ATOMIC_NUMBER, ATOMIC_SYMBOL
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BUCToolkit.utils.index_ops import index_reduce
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.BatchStructures.StructuresIO import structures_io_dumper
from BUCToolkit.utils.setup_loggers import has_any_handler, clear_all_handlers, BaseLogger


class _BaseMD(BaseLogger):
    """ Base BatchMD """

    __slots__ = [
        'time_step', 'time_now',
        'verbose', 'logger',
        'batch_tensor', 'batch_scatter',
        'free_degree',
        'require_grad',
        'EK_TARGET',
        'Ekt_vir',
        'Ek',
        'p_iota',
        #'__dict__'
    ]

    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 0,
            is_compile: bool = False,
            compile_kwargs: dict | None = None,
    ):
        """
        Parameters:
            time_step: float, time per step (fs).
            max_step: maximum steps.
            T_init: initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution.
                If V_init is given, T_init will be ignored.
            output_file: the path to the binary file that stores trajectories.
            output_structures_per_step: int, output structures per output_structures_per_step steps.
            device: device that program run on.
            verbose: control the detailed degree of output text information.
                0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.
                Note: verbose > 0 will be very slow, especially for computation on GPU.
            is_compile: whether to use jit to compile integrator or not.
            compile_kwargs: keyword arguments passed to compile. Only work when is_compile is True.
        """
        self.time_step = time_step
        self.time_now = th.scalar_tensor(0., device=device)  # the accumulated time
        assert (max_step > 0) and isinstance(max_step, int), f'max_step must be a positive integer, but occurred {max_step}.'
        self.max_step = int(max_step)
        self.T_init = float(T_init)
        self.output_file = str(output_file) if output_file is not None else None
        self.output_structures_per_step = int(output_structures_per_step)
        self.device = device
        self.verbose = int(verbose)
        self.is_compile = bool(is_compile)
        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else dict()

        self.EK_TARGET = None  # target kinetic energy under set temperature.
        self.Ekt_vir = None    # virtual kinetic energy for _CSVR_ thermostat.
        self.Ek = None         # kinetic energy at each timestep.
        self.p_iota = None       # thermostat var. for Nose-Hoover.

        self.batch_tensor = None  # tensor form of `batch_indices` if it was given.
        self.batch_scatter = None # tensor indices form of `batch_indices` if it was given
                                  # e.g., batch_indices = (3, 2, 1), thus self.scatter = tensor([0, 0, 0, 1, 1, 2])
        self.free_degree = None  # (n_batch, ), freedom degree tensor
        self.require_grad = None

        # An inner attr that lets the dumper do not close after `self.run`.
        # It is used to contiguously run within a loop.
        # Adv. API `MolecularDynamics` turns on it.
        self._HOLD_DUMPER = False

        # set dumper
        # Note: cache_size: NOW it be hard coded as 4 MB / 4096 bytes
        self.dumper = structures_io_dumper(
            path=self.output_file,
            mode='x',
        )

        # logging
        super().__init__()
        self.init_logger('Main.MD')

    def reset_dumper(self, dumper: Any) -> None:
        if self.output_file is not None:
            self.dumper.close()
            del self.dumper
            self.dumper = dumper
        else:
            self.logger.error(
                "ERROR: No output file specified. Hence, resetting dumper is meaningless.\n"
                "'reset_dumper': Operation REFUSED."
            )

    def _reduce_Ek_T(self, batch_indices, masses, V):
        if batch_indices is not None:
            Ek = th.sum(
                0.5 * index_reduce(
                    masses * V * V,
                    self.batch_scatter,
                    1,
                    out_size=self.scatter_dim_out_size
                ) * 103.642696562621738,
                dim=-1
            ).squeeze_(0)  # (n_batch, ), eV/atom. Faraday constant F = 96485.3321233100184.
        else:
            Ek = 0.5 * th.sum(
                masses * V * V,
                dim=(-2, -1)
            ) * 103.642696562621738  # (n_batch, ), eV/atom. Faraday constant F = 96485.3321233100184.
        temperature = (2 * Ek) / ((self.free_degree + 1e-20) * 8.617333262145e-5) # Boltzmann constant kB = 8.617333262145e-5 eV/K

        return Ek, temperature

    def calc_mass_center(self, mass, mass_short, Xr, ) -> th.Tensor:
        """
        calculate mass center based on mass_short and Xr.
        Args:
            mass:
            mass_short:
            Xr:

        Returns:

        """
        if self.batch_scatter is None:
            # initialize topologie
            mass_sum = th.sum(mass_short, dim=1, keepdim=True).unsqueeze(-2)  # (n_batch, 1, 1)
            mass_center = th.sum(Xr * mass, dim=1, keepdim=True) / mass_sum  # (n_batch, 1, n_dim)
        else:
            # initialize topologie
            mass_sum = index_reduce(mass_short, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size).unsqueeze(-1)  # (1, n_batch, 1)
            mass_center = index_reduce(Xr * mass, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size) / mass_sum  # (1, n_batch, n_dim)
            mass_center = mass_center.index_select(1, self.batch_scatter)  # (1, sumN*A, n_dim)

        return mass_center

    def _do_async_dump(self, q: queue.Queue):
        """
        A backend thread to async. dump
        Args:
            q: queue to receive data. contains: tuple of (dumper, event, *data)

        Returns: None

        """
        while True:
            try:
                dumper, event, _print_Ep, _print_X, _print_V, _print_F = q.get()
                if dumper is None:
                    break
                # event: th.cuda.Event, ensure copy done
                event.synchronize()
                dumper.step(
                    _print_Ep.numpy(),
                    _print_X.numpy(),
                    _print_V.numpy(),
                    _print_F.numpy(),
                )
            except Exception as e:
                self.logger.error(f"Error: Failed to dump data due to \"{e}\"")

    def _do_async_print(self, q: queue.Queue):
        """

        Args:
            q:

        Returns:

        """
        formatter1 = {'float': '{:> .2f}'.format}
        formatter2 = {'float': '{:> 5.10f}'.format}
        i = 0  # reinsurance for Exception part
        # Note: PRINTING IS VERY EXPENSIVE !!!
        while True:
            try:
                i, batch_indices, _print_Ep, _print_Ek, _print_temperature, _print_X, _print_V, _print_F = q.get()
                if i is None:
                    break
                if self.verbose > 0:
                    # print format
                    np.set_printoptions(
                        precision=8,
                        linewidth=1024,
                        floatmode='fixed',
                        suppress=True,
                        formatter=formatter1,
                        threshold=2000
                    )
                    self.logger.info(
                        f'Step: {i:>12d}\n\t'
                        f'T     = {_print_temperature.numpy(force=True)}\n\t'
                        f'E_tol = {np.array2string((_print_Ek + _print_Ep).numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                        f'Ek    = {np.array2string(_print_Ek.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                        f'Ep    = {np.array2string(_print_Ep.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                        # f'Time: {time.perf_counter() - t_step:>5.4f}'
                    )
                    # t_step = time.perf_counter()
                if self.verbose > 1:
                    # split to print
                    if batch_indices is not None:
                        X_tup = th.split(_print_X, batch_indices, dim=1)
                        V_tup = th.split(_print_V, batch_indices, dim=1)
                    else:
                        X_tup = (_print_X,)
                        V_tup = (_print_V,)
                    np.set_printoptions(
                        precision=8,
                        floatmode='fixed',
                        suppress=True,
                        formatter=formatter2,
                        threshold=3000000
                    )
                    self.logger.info('_' * 100)
                    self.logger.info(f'Configuration {i}:')
                    for __x in X_tup:
                        X_str = np.array2string(
                            __x.numpy(force=True), **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")
                        self.logger.info(f'{X_str}\n')
                    del X_str, X_tup
                    if self.verbose > 2:
                        self.logger.info(f'Velocities {i}:')
                        for __x in V_tup:
                            V_str = np.array2string(
                                __x.numpy(force=True), **FLOAT_ARRAY_FORMAT
                            ).replace("[", " ").replace("]", " ")
                            self.logger.info(f'{V_str}\n')
                        del V_str
                    self.logger.info('_' * 100)
            except Exception as e:
                self.logger.error(f"Error: Failed to logout at {i}-th iteration due to \"{e}\".")

    def _print_elem_info(self, Element_list, batch_indices):
        # elem info
        elem_list = list()
        _element_list = list()
        if batch_indices is not None:
            indx_old = 0
            for indx in batch_indices:
                _element_list.append(Element_list[0][indx_old: indx_old + indx])
                indx_old += indx
        else:
            _element_list = Element_list
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

    @th.compiler.disable
    def _calc_EF(
            self,
            X,
            func,
            func_args,
            func_kwargs,
            grad_func_,
            grad_func_args,
            grad_func_kwargs,
            is_grad_func_contain_y,
    ) -> Tuple[th.Tensor, th.Tensor]:
        with th.set_grad_enabled(self.require_grad):
            X.requires_grad_(self.require_grad)
            Energy = func(X, *func_args, **func_kwargs)
            if is_grad_func_contain_y:
                Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs)
            else:
                Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs)
        return Energy, Force

    def initialize(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            masses: th.Tensor,
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            is_fix_mass_center: bool = False
    ) -> None:
        """
        Do some possible initialization before entering main loop.
        Default is doing nothing.
        """
        pass

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            Cell_vector: th.Tensor | None = None,
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            move_to_center_freq: int = -1
    ) -> None:
        """
        Parameters:
            func: the main function of instantiated torch.nn.Module class.
            X: Tensor[n_batch, n_atom, 3], the atom coordinates that input to func. If a 2D X was given, the first dimension would be set to 1.
            Element_list: List[List[str | int]], the atomic type (element) corresponding to each row of each batch in X.
            Cell_vector: Tensor[n_batch, 3, 3], the cell vectors. Only for logging out information, no really calculate. If not given, set all zeros.
            V_init: the initial velocities of each atom. If None, a random velocity generated by Boltzmann distribution would be set.
            grad_func: user-defined function that grad_func(X, ...) returns the func's gradient at X. if None, grad_func(X, ...) = th.autograd.grad(func(X, ...), X).
            func_args: optional, other input of func.
            func_kwargs: optional, other input of func.
            grad_func_args: optional, other input of grad_func.
            grad_func_kwargs: optional, other input of grad_func.
            is_grad_func_contain_y: bool, if True, grad_func contains output of func followed by X i.e., grad = grad_func(X, y, ...), else grad = grad_func(X, ...)
            require_grad: bool, if True, autograd will be turned on for func(X, *func_args, **func_kwargs) calculation.
            batch_indices: the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is the same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]
            fixed_atom_tensor: the indices of X that fixed.
            move_to_center_freq: the period of translating coordinates and velocities of atoms into the mass center & 0.
                if `move_to_center_freq` <= 0, the translation would not apply.

        Returns: None

        """
        try:
            if th.device(self.device).type == "cuda":
                self.__run_on_cuda(
                    func,
                    X,
                    Element_list,
                    Cell_vector,
                    V_init,
                    grad_func,
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs,
                    is_grad_func_contain_y,
                    require_grad,
                    batch_indices,
                    fixed_atom_tensor,
                    move_to_center_freq
                )
            elif th.device(self.device).type == "cpu":
                self.__run_on_cpu(
                    func,
                    X,
                    Element_list,
                    Cell_vector,
                    V_init,
                    grad_func,
                    func_args,
                    func_kwargs,
                    grad_func_args,
                    grad_func_kwargs,
                    is_grad_func_contain_y,
                    require_grad,
                    batch_indices,
                    fixed_atom_tensor,
                    move_to_center_freq
                )
            else:
                raise NotImplementedError(F"device {self.device} not supported.")
        finally:
            pass

    def __run_on_cuda(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            Cell_vector: th.Tensor | None = None,
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            move_to_center_freq: int = -1
    ) -> None:
        """
        GPU version of BatchMD.run(...), including buffer, sync. D2H copy, etc.
        """
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        elif len(X.shape) != 3:
            raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
        if not isinstance(move_to_center_freq, int):
            raise TypeError(f'`move_to_center_freq` must be an integer, but got {type(move_to_center_freq)}.')
        elif move_to_center_freq <= 0:
            is_fix_mass_center = False
        else:
            is_fix_mass_center = True
        n_batch, n_atom, n_dim = X.shape
        if func_kwargs is None: func_kwargs = dict()
        if grad_func_kwargs is None: grad_func_kwargs = dict()
        # Check batch indices
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                self.batch_tensor = batch_indices
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (List, Tuple)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)  # the tensor version of batch_indices which is a List.
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )  # scatter mask of the int tensor with the same shape as X.shape[1], which the data in one batch have one index.
            self.scatter_dim_out_size = self.batch_scatter.max().item() + 1
            n_true_batch = len(batch_indices)
        else:
            n_true_batch = n_batch

        # Manage Atomic Type & Masses
        masses = list()
        atomic_numbers = list()
        for _Elem in Element_list:
            atomic_numbers.append([ATOMIC_SYMBOL[__elem] if isinstance(__elem, str) else int(__elem) for __elem in _Elem])
            masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
        masses_short = th.tensor(masses, dtype=th.float32, device=self.device)  # (n_batch, n_atom)
        masses = masses_short.unsqueeze(-1).expand_as(X).contiguous()  # (n_batch, n_atom, n_dim)
        # grad_func
        if grad_func is None:
            is_grad_func_contain_y = True
            def grad_func_(x, y, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                g = th.autograd.grad(y, x, grad_shape)
                return g[0]
        else:
            grad_func_ = grad_func
        # Selective dynamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        # other check
        if (not isinstance(self.max_step, int)) or (self.max_step <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.max_step}. It would be an integer greater than 0.')

        # set variables device
        func = preload_func(func, self.device)

        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)
        # calc. freedom degree
        if batch_indices is None:
            _free_degree = X.shape[1] * n_dim
            if is_fix_mass_center:
                _free_degree -= 3
                # initialize topologie
                MASS_SUM = th.sum(masses_short, dim=1, keepdim=True).unsqueeze(-2)  # (n_batch, 1, 1)
                MASS_CENTER = th.sum(X * masses, dim=1, keepdim=True)/MASS_SUM  # (n_batch, 1, n_dim)
            self.free_degree = th.full((n_batch, ), _free_degree, dtype=th.int64, device=self.device)
            n_reduce = th.where(th.abs(atom_masks) < 1e-6, 1, 0).sum(dim=(-2, -1))  # (n_batch, )
            self.free_degree -= n_reduce
        else:
            self.free_degree = self.batch_tensor * n_dim  # (n_batch, )
            if is_fix_mass_center:
                self.free_degree -= 3
                # initialize topologie
                MASS_SUM = index_reduce(masses_short, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size).unsqueeze(-1)  # (1, n_batch, 1)
                MASS_CENTER = index_reduce(X * masses, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size)/MASS_SUM  # (1, n_batch, n_dim)
                MASS_CENTER = MASS_CENTER.index_select(1, self.batch_scatter)  # (1, sumN*A, n_dim)
            n_reduce_tensor = th.where(th.abs(atom_masks) < 1e-6, 1, 0).sum(dim=-1)
            n_reduce = index_reduce(n_reduce_tensor, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size).squeeze(0)  # (n_batch, )
            self.free_degree -= n_reduce
        # target kinetic energy for NVT|NPT ensembles
        self.EK_TARGET = (self.free_degree / 2.) * 8.617333262145e-5 * self.T_init
        # Generate initial Velocities
        if V_init is not None:
            if V_init.shape != X.shape:
                raise ValueError(f'V_init and X must have the same shape, but got {V_init.shape} and {X.shape}')
            if self.verbose > 0: self.logger.info('Initial velocities was given, T_init will be ignored.')
            V_init = V_init.to(self.device)
            V = V_init.detach() * atom_masks
            V = V.to(self.device)
        else:
            V = th.normal(
                0.,
                th.sqrt(self.T_init * 8314.46261815324 / masses) * 1.e-5,  # Unit: Ang/fs, R = kB * L = 8.31446261815324 J/(mol * K)
            ) * atom_masks
        # remove translation veloc.
        if is_fix_mass_center:
            V.sub_(self.calc_mass_center(masses, masses_short, V))
        # split by batch_indices
        if batch_indices is not None:
            masses_tup = th.split(masses, batch_indices, dim=1)
            V_tup = th.split(V, batch_indices, dim=1)
        else:
            masses_tup = (masses,)
            V_tup = (V,)

        # initialize thermostat parameters
        # (n_batch, ), eV/atom. The initial virtual Ek_t for CSVR.
        self.Ekt_vir = th.cat(
            [0.5 * th.sum(_m * V_tup[_] ** 2, dim=(-2, -1)) * 103.642696562621738 for _, _m in enumerate(masses_tup)]
        )
        # The initial iota for Nose-Hoover
        if batch_indices is not None:
            self.p_iota = th.zeros(1, len(batch_indices), 1, device=self.device, dtype=th.float32)
        else:
            self.p_iota = th.zeros(n_batch, 1, 1, device=self.device, dtype=th.float32)
        # whether grad needs autograd
        self.require_grad = require_grad

        # pre-allocate the tensors
        #   _buf_* is the vars on GPU that apply copy.
        #   _print_* is the vars on CPU that async. do D2H for _buf_*.
        if batch_indices is not None:
            _print_temperature = th.empty(len(batch_indices), device='cpu', dtype=th.float32, pin_memory=True)
            _print_Ek = th.empty(len(batch_indices), device='cpu', dtype=th.float32, pin_memory=True)
            _print_Ep = th.empty(len(batch_indices), device='cpu', dtype=th.float32, pin_memory=True)
            _buf_Tp = th.empty(len(batch_indices), device=self.device, dtype=th.float32)
            _buf_Ek = th.empty(len(batch_indices), device=self.device, dtype=th.float32)
            _buf_Ep = th.empty(len(batch_indices), device=self.device, dtype=th.float32)
        else:
            _print_temperature = th.empty(n_batch, device='cpu', dtype=th.float32, pin_memory=True)
            _print_Ek = th.empty(n_batch, device='cpu', dtype=th.float32, pin_memory=True)
            _print_Ep = th.empty(n_batch, device='cpu', dtype=th.float32, pin_memory=True)
            _buf_Tp = th.empty(n_batch, device=self.device, dtype=th.float32)
            _buf_Ek = th.empty(n_batch, device=self.device, dtype=th.float32)
            _buf_Ep = th.empty(n_batch, device=self.device, dtype=th.float32)
        _print_X = th.empty_like(X, device='cpu', dtype=th.float32, pin_memory=True)
        _print_V = th.empty_like(V, device='cpu', dtype=th.float32, pin_memory=True)
        _print_F = th.empty_like(X, device='cpu', dtype=th.float32, pin_memory=True)
        _buf_X = th.empty_like(X, device=self.device, dtype=th.float32)
        _buf_V = th.empty_like(V, device=self.device, dtype=th.float32)
        _buf_F = th.empty_like(X, device=self.device, dtype=th.float32)
        # initialize the dumper
        X_arr = X.numpy(force=True)
        _x_dtype = X_arr.dtype.str
        atom_masks_arr = atom_masks.numpy(force=True).astype(_x_dtype)
        _num_dump =  math.ceil(self.max_step/self.output_structures_per_step)
        dumper = self.dumper
        # write head information.
        if Cell_vector is None:
            Cell_vector = np.zeros((n_true_batch, 3, 3), dtype=np.float32)
        elif isinstance(Cell_vector, th.Tensor):
            Cell_vector = Cell_vector.numpy(force=True)
        elif not isinstance(Cell_vector, np.ndarray):
            Cell_vector = np.asarray(Cell_vector)
        if self.batch_tensor is not None:
            dumper.start_from_arrays(
                1,
                self.batch_tensor.numpy(force=True),  # batch indices
                Cell_vector, # cell
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            dumper.step(
                self.batch_tensor.numpy(force=True),
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )
        else:
            dumper.start_from_arrays(
                1,
                Cell_vector,
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            dumper.step(
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )
        # continue to write main data
        dumper.start_from_arrays(
            _num_dump,
            _print_Ep.numpy(),
            _print_X.numpy(),
            _print_V.numpy(),
            _print_F.numpy(),
        )
        # custom initialization
        self.initialize(
            func,
            X,
            Element_list,
            masses,
            V,
            grad_func,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            is_grad_func_contain_y,
            require_grad,
            batch_indices,
            fixed_atom_tensor,
            is_fix_mass_center
        )

        # print Atoms Information
        #   if has no handler, means the handler is upper level 'Main', thus not print repeatedly
        if (self.verbose > 0) and len(self.logger.handlers) > 0:
            self._print_elem_info(Element_list, batch_indices)

        # MAIN Loop
        with th.no_grad():
            X = X.contiguous()
            V = V.contiguous()
            masses = masses.contiguous()
            atom_masks = atom_masks.contiguous()
            t_step = time.perf_counter()
            t_main_loop = time.perf_counter()
            with th.set_grad_enabled(require_grad):
                X.requires_grad_(require_grad)
                Energy = func(X, *func_args, **func_kwargs)  # Note: func must return th.Tensor(n_batch, )
                if batch_indices is not None:
                    if len(Energy) != len(batch_indices):  # check batch number of output
                        raise RuntimeError(
                            f'batch number of `func` output ({len(Energy)}) does not match the input `batch_indices` ({len(batch_indices)}'
                        )

                if is_grad_func_contain_y:
                    Forces = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Forces = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks
                Forces = Forces.contiguous()

            Ek = th.zeros_like(Energy)
            temperature = th.zeros_like(Energy)
            # preload a graph of Ek, T
            Ek_T_graph = th.cuda.CUDAGraph()
            with th.cuda.graph(Ek_T_graph):
                _e_tmp, _t_tmp = self._reduce_Ek_T(batch_indices, masses, V)
                Ek.copy_(_e_tmp)
                temperature.copy_(_t_tmp)
            # preload a graph of mass center
            if is_fix_mass_center:
                mass_center_graph = th.cuda.CUDAGraph()
                with th.cuda.graph(mass_center_graph):
                    _dX = MASS_CENTER - self.calc_mass_center(masses, masses_short, X)
                    _dV = - self.calc_mass_center(masses, masses_short, V)
                    X.add_(_dX)  # (n_batch, n_atom, n_dim) - (n_batch, 1, n_dim)
                    V.add_(_dV)
            else:
                mass_center_graph = None
            # preload Verlet update
            #self.Verlet1_graph = th.cuda.CUDAGraph()
            #with th.cuda.graph(self.Verlet1_graph):
            #    V.addcdiv_(Forces, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            #    X.add_(V, alpha=self.time_step)
            #self.Verlet2_graph = th.cuda.CUDAGraph()
            #with th.cuda.graph(self.Verlet2_graph):
            #    Forces.mul_(atom_masks)
            #    # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
            #    V.addcdiv_(Forces, masses, value=0.5 * self.time_step * 9.64853329045427e-3)

            copy_stream = th.cuda.Stream()
            copy_event = th.cuda.Event()
            compute_event = th.cuda.Event()
            compute_event.record(th.cuda.default_stream(self.device))  # the default stream is the compute (main) stream.
            # launch the dumping thread
            dump_queue = queue.Queue()
            dump_thread = threading.Thread(target=self._do_async_dump, args=(dump_queue, ), daemon=True)
            logout_queue = queue.Queue()
            logout_thread = threading.Thread(target=self._do_async_print, args=(logout_queue, ), daemon=True)
            try:
                dump_thread.start()
                logout_thread.start()
                #ptlist = list()  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                fl = th.compile(self._main_for_loop_cuda, **self.compile_kwargs, disable=(not self.is_compile))
                fl(
                    Ek_T_graph,
                    Ek,
                    Energy,
                    X,
                    V,
                    Forces,
                    compute_event,
                    temperature,
                    copy_stream,
                    _buf_Tp,
                    _buf_Ek,
                    _buf_Ep,
                    _buf_X,
                    _buf_V,
                    _buf_F,
                    _print_temperature,
                    _print_Ek,
                    _print_Ep,
                    _print_X,
                    _print_V,
                    _print_F,
                    copy_event,
                    dump_queue,
                    dumper,
                    logout_queue,
                    func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                    masses, atom_masks, is_grad_func_contain_y, batch_indices,
                    is_fix_mass_center,
                    move_to_center_freq,
                    mass_center_graph
                )
                if not self._HOLD_DUMPER:
                    dumper.close()
                th.cuda.synchronize()
                if self.verbose > 0:
                    self.logger.info(f'MAIN LOOP DONE. Elapsed time: {time.perf_counter() - t_main_loop:>5.4f} s')
            finally:
                dump_queue.put([None]*6)
                dump_thread.join()
                logout_queue.put([None]*8)
                logout_thread.join()

        del self.Ekt_vir
        #return ptlist  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def _main_for_loop_cuda(
            self,
            Ek_T_graph,
            Ek,
            Energy,
            X,
            V,
            Forces,
            compute_event,
            temperature,
            copy_stream,
            _buf_Tp,
            _buf_Ek,
            _buf_Ep,
            _buf_X,
            _buf_V,
            _buf_F,
            _print_temperature,
            _print_Ek,
            _print_Ep,
            _print_X,
            _print_V,
            _print_F,
            copy_event,
            dump_queue,
            dumper,
            logout_queue,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices,
            is_fix_mass_center,
            move_to_center_freq,
            mass_center_graph
    ):
        #t_step = time.perf_counter()
        _do_print = False
        for i in range(self.max_step):
            #with th.profiler.record_function('Calc. E, T <<<<<'):
            Ek_T_graph.replay()
            # Ek, temperature = self._reduce_Ek_T(batch_indices, masses, V)
            self.Ek = Ek  # th.sum(Ek, dim=0)  # saving the real kinetic energy for VR & CSVR to avoid double counting.
            # if self.verbose > 0:
            #with th.profiler.record_function('D2H COPY <<<<<'):
            if i % self.output_structures_per_step == 0:
                _do_print = True
                compute_event.wait(th.cuda.default_stream(self.device))
                # D2D, fast copy purely on GPU
                _buf_Tp.copy_(temperature.squeeze().contiguous())
                _buf_Ek.copy_(Ek.squeeze().contiguous())
                _buf_Ep.copy_(Energy.squeeze().contiguous())
                _buf_X.copy_(X)
                _buf_V.copy_(V)
                _buf_F.copy_(Forces)
                # D2H, async.
                with th.cuda.stream(copy_stream):
                    copy_stream.wait_stream(th.cuda.default_stream(self.device))
                    _print_temperature.copy_(_buf_Tp, non_blocking=True)
                    _print_Ek.copy_(_buf_Ek, non_blocking=True)
                    _print_Ep.copy_(_buf_Ep, non_blocking=True)
                    _print_X.copy_(_buf_X, non_blocking=True)  # D2H
                    _print_V.copy_(_buf_V, non_blocking=True)
                    _print_F.copy_(_buf_F, non_blocking=True)
                # ptlist.append(X.numpy(force=True))  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                copy_event.record(copy_stream)
                # use backend thread to dump
                dump_queue.put((dumper, copy_event, _print_Ep, _print_X, _print_V, _print_F))

            # Update X, V
            #with th.profiler.record_function('MAIN UPDATE <<<<<'):
            X, V, Energy, Forces = self._updateXV(
                X, V, Forces,
                func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                masses, atom_masks, is_grad_func_contain_y, batch_indices
            )
            compute_event.record(th.cuda.default_stream(self.device))  # the default stream is the compute (main) stream.
            # print
            #with th.profiler.record_function('PRINT <<<<<'):
            if _do_print:
                logout_queue.put((i, batch_indices, _print_Ep, _print_Ek, _print_temperature, _print_X, _print_V, _print_F))
                _do_print = False

            # Correct barycentric transition
            #with th.profiler.record_function('MOVE TO CENTER <<<<<'):
            if is_fix_mass_center and (i % move_to_center_freq == 0):
                mass_center_graph.replay()
            self.time_now += self.time_step

    def __run_on_cpu(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            Cell_vector: th.Tensor | None = None,
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Sequence = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Sequence = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            move_to_center_freq: int = -1
    ) -> None:
        """
        Pure CPU version of BaseMD.run(...).
        """
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        elif len(X.shape) != 3:
            raise ValueError(f'`X` must be 2D or 3D, but got shape [{X.shape}]')
        if not isinstance(move_to_center_freq, int):
            raise TypeError(f'`move_to_center_freq` must be an integer, but got {type(move_to_center_freq)}.')
        elif move_to_center_freq <= 0:
            is_fix_mass_center = False
        else:
            is_fix_mass_center = True
        n_batch, n_atom, n_dim = X.shape
        if func_kwargs is None: func_kwargs = dict()
        if grad_func_kwargs is None: grad_func_kwargs = dict()
        # Check batch indices
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                self.batch_tensor = batch_indices
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, (List, Tuple)):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'
            self.batch_tensor = th.as_tensor(batch_indices, device=self.device)  # the tensor version of batch_indices which is a List.
            self.batch_scatter = th.repeat_interleave(
                th.arange(0, len(batch_indices), dtype=th.int64, device=self.device),
                self.batch_tensor,
                dim=0
            )  # scatter mask of the int tensor with the same shape as X.shape[1], which the data in one batch have one index.
            self.scatter_dim_out_size = self.batch_scatter.max().item() + 1
            n_true_batch = len(batch_indices)
        else:
            n_true_batch = n_batch

        # Manage Atomic Type & Masses
        masses = list()
        atomic_numbers = list()
        for _Elem in Element_list:
            atomic_numbers.append([ATOMIC_SYMBOL[__elem] if isinstance(__elem, str) else ATOMIC_NUMBER[__elem] for __elem in _Elem])
            masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
        masses_short = th.tensor(masses, dtype=th.float32, device=self.device)  # (n_batch, n_atom)
        masses = masses_short.unsqueeze(-1).expand_as(X).contiguous()  # (n_batch, n_atom, n_dim)
        # grad_func
        if grad_func is None:
            is_grad_func_contain_y = True
            def grad_func_(x, y, grad_shape=None):
                if grad_shape is None:
                    grad_shape = th.ones_like(y)
                g = th.autograd.grad(y, x, grad_shape)
                return g[0]
        else:
            grad_func_ = grad_func
        # Selective dynamics
        if fixed_atom_tensor is None:
            atom_masks = th.ones_like(X, device=self.device)
        elif fixed_atom_tensor.shape == X.shape:
            atom_masks = fixed_atom_tensor.to(self.device)
        else:
            raise RuntimeError(f'fixed_atom_tensor (shape: {fixed_atom_tensor.shape}) does not match X (shape: {X.shape}).')
        # other check
        if (not isinstance(self.max_step, int)) or (self.max_step <= 0):
            raise ValueError(f'Invalid value of maxiter: {self.max_step}. It would be an integer greater than 0.')

        # set variables device
        func = preload_func(func, self.device)

        if isinstance(grad_func_, nn.Module):
            grad_func_ = grad_func_.to(self.device)
        X = X.to(self.device)
        # calc. freedom degree
        if batch_indices is None:
            _free_degree = X.shape[1] * n_dim
            if is_fix_mass_center:
                _free_degree -= 3
                # initialize topologie
                MASS_SUM = th.sum(masses_short, dim=1, keepdim=True).unsqueeze(-2)  # (n_batch, 1, 1)
                MASS_CENTER = th.sum(X * masses, dim=1, keepdim=True)/MASS_SUM  # (n_batch, 1, n_dim)
            else:
                MASS_CENTER = None
            self.free_degree = th.full((n_batch, ), _free_degree, dtype=th.int64, device=self.device)
            n_reduce = th.where(th.abs(atom_masks) < 1e-6, 1, 0).sum(dim=(-2, -1))  # (n_batch, )
            self.free_degree -= n_reduce
        else:
            self.free_degree = self.batch_tensor * n_dim  # (n_batch, )
            if is_fix_mass_center:
                self.free_degree -= 3
                # initialize topologie
                MASS_SUM = index_reduce(masses_short, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size).unsqueeze(-1)  # (1, n_batch, 1)
                MASS_CENTER = index_reduce(X * masses, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size)/MASS_SUM  # (1, n_batch, n_dim)
                MASS_CENTER = MASS_CENTER.index_select(1, self.batch_scatter)  # (1, sumN*A, n_dim)
            else:
                MASS_CENTER = None
            n_reduce_tensor = th.where(th.abs(atom_masks) < 1e-6, 1, 0).sum(dim=-1)
            n_reduce = index_reduce(n_reduce_tensor, self.batch_scatter, dim=1, out_size=self.scatter_dim_out_size).squeeze(0)  # (n_batch, )
            self.free_degree -= n_reduce
        # target kinetic energy for NVT|NPT ensembles
        self.EK_TARGET = (self.free_degree / 2.) * 8.617333262145e-5 * self.T_init
        # Generate initial Velocities
        if V_init is not None:
            if V_init.shape != X.shape:
                raise ValueError(f'V_init and X must have the same shape, but got {V_init.shape} and {X.shape}')
            if self.verbose > 0: self.logger.info('Initial velocities was given, T_init will be ignored.')
            V_init = V_init.to(self.device)
            V = V_init.detach() * atom_masks
            V = V.to(self.device)
        else:
            V = th.normal(
                0.,
                th.sqrt(self.T_init * 8314.46261815324 / masses) * 1.e-5,  # Unit: Ang/fs, R = kB * L = 8.31446261815324 J/(mol * K)
            ) * atom_masks
        # remove translation veloc.
        if is_fix_mass_center:
            V.sub_(self.calc_mass_center(masses, masses_short, V))
        # split by batch_indices
        if batch_indices is not None:
            masses_tup = th.split(masses, batch_indices, dim=1)
            V_tup = th.split(V, batch_indices, dim=1)
        else:
            masses_tup = (masses,)
            V_tup = (V,)

        # initialize thermostat parameters
        # (n_batch, ), eV/atom. The initial virtual Ek_t for CSVR.
        self.Ekt_vir = th.cat(
            [0.5 * th.sum(_m * V_tup[_] ** 2, dim=(-2, -1)) * 103.642696562621738 for _, _m in enumerate(masses_tup)]
        )
        # The initial iota for Nose-Hoover
        if batch_indices is not None:
            self.p_iota = th.zeros(1, len(batch_indices), 1, device=self.device, dtype=th.float32)
        else:
            self.p_iota = th.zeros(n_batch, 1, 1, device=self.device, dtype=th.float32)
        # whether grad needs autograd
        self.require_grad = require_grad

        # initialize the dumper
        X_arr = X.numpy(force=True)
        _x_dtype = X_arr.dtype.str
        atom_masks_arr = atom_masks.numpy(force=True).astype(_x_dtype)
        # Note: cache_size: NOW it be hard coded as 4 MB
        _num_dump =  math.ceil(self.max_step/self.output_structures_per_step)
        dumper = self.dumper
        if Cell_vector is None:
            Cell_vector = np.zeros((n_true_batch, 3, 3), dtype=np.float32)
        elif isinstance(Cell_vector, th.Tensor):
            Cell_vector = Cell_vector.numpy(force=True)
        elif not isinstance(Cell_vector, np.ndarray):
            Cell_vector = np.asarray(Cell_vector)
        # write head information.
        if self.batch_tensor is not None:
            dumper.start_from_arrays(
                1,
                self.batch_tensor.numpy(force=True),  # batch indices
                Cell_vector,
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            dumper.step(
                self.batch_tensor.numpy(force=True),
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )
        else:
            dumper.start_from_arrays(
                1,
                Cell_vector,
                np.asarray(atomic_numbers),  # element type / atomic number
                atom_masks_arr,  # fixation mask
            )
            dumper.step(
                Cell_vector,
                np.asarray(atomic_numbers),
                atom_masks_arr
            )
        # continue to write main data
        dumper.start_from_arrays(
            _num_dump,
            self.Ekt_vir.numpy(),
            X.numpy(),
            V.numpy(),
            V.numpy(),  # Here F is not calculated yet, but V has the same shape, hence can do this replacement
        )
        # custom initialization
        self.initialize(
            func,
            X,
            Element_list,
            masses,
            V,
            grad_func,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            is_grad_func_contain_y,
            require_grad,
            batch_indices,
            fixed_atom_tensor,
            is_fix_mass_center
        )

        # print Atoms Information
        #   if has no handler, means the handler is upper level 'Main', thus not print repeatedly
        if (self.verbose > 0) and len(self.logger.handlers) > 0:
            self._print_elem_info(Element_list, batch_indices)
        formatter1 = {'float': '{:> .2f}'.format}
        formatter2 = {'float': '{:> 5.10f}'.format}

        # MAIN Loop
        with th.no_grad():
            X = X.contiguous()
            V = V.contiguous()
            masses = masses.contiguous()
            atom_masks = atom_masks.contiguous()

            t_in = time.perf_counter()
            with th.set_grad_enabled(require_grad):
                X.requires_grad_(require_grad)
                Energy = func(X, *func_args, **func_kwargs)  # Note: func must return th.Tensor(n_batch, )
                if batch_indices is not None:
                    if len(Energy) != len(batch_indices):  # check batch number of output
                        raise RuntimeError(
                            f'batch number of `func` output ({len(Energy)}) does not match the input `batch_indices` ({len(batch_indices)}'
                        )

                if is_grad_func_contain_y:
                    Forces = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Forces = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks
                Forces = Forces.contiguous()

            #ptlist = list()  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            fl = th.compile(self._main_for_loop_cpu, **self.compile_kwargs, disable=(not self.is_compile))
            fl(
                Energy,
                X,
                V,
                Forces,
                dumper,
                func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                masses, atom_masks, is_grad_func_contain_y, batch_indices,
                MASS_CENTER,
                masses_short,
                formatter1,
                formatter2,
                is_fix_mass_center,
                move_to_center_freq,
            )
            if not self._HOLD_DUMPER:
                dumper.close()

        del self.Ekt_vir
        #return ptlist  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def _main_for_loop_cpu(
            self,
            Energy,
            X,
            V,
            Forces,
            dumper,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices,
            MASS_CENTER,
            masses_short,
            formatter1,
            formatter2,
            is_fix_mass_center,
            move_to_center_freq,
    ):
        _do_print = False
        for i in range(self.max_step):
            #with th.profiler.record_function('Calc. E, T <<<<<'):
            Ek, temperature = self._reduce_Ek_T(batch_indices, masses, V)
            self.Ek = Ek  # th.sum(Ek, dim=0)  # saving the real kinetic energy for VR & CSVR to avoid double counting.
            # if self.verbose > 0:
            #with th.profiler.record_function('DUMPING <<<<<'):
            if i % self.output_structures_per_step == 0:
                _do_print = True
                dumper.step(
                    Energy.numpy(),
                    X.numpy(),
                    V.numpy(),
                    Forces.numpy(),
                )
            # print, ensure print data correspond to dumping data, and then update. differ from cuda orders.
            #with th.profiler.record_function('PRINT <<<<<'):
            if _do_print:
                # Note: PRINTING IS VERY EXPENSIVE !!!
                if self.verbose > 0:
                    # print format
                    np.set_printoptions(
                        precision=8,
                        linewidth=1024,
                        floatmode='fixed',
                        suppress=True,
                        formatter=formatter1,
                        threshold=2000
                    )
                    self.logger.info(
                        f'Step: {i:>12d}\n\t'
                        f'T     = {temperature.numpy(force=True)}\n\t'
                        f'E_tol = {np.array2string((Ek + Energy).numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                        f'Ek    = {np.array2string(Ek.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                        f'Ep    = {np.array2string(Energy.numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}\n\t'
                        #f'Time: {time.perf_counter() - t_in:>5.4f}'
                    )
                    #t_in = time.perf_counter()
                if self.verbose > 1:
                    # split to print
                    if batch_indices is not None:
                        X_tup = th.split(X, batch_indices, dim=1)
                        V_tup = th.split(V, batch_indices, dim=1)
                    else:
                        X_tup = (X,)
                        V_tup = (V,)
                    np.set_printoptions(
                        precision=8, floatmode='fixed', suppress=True, formatter=formatter2, threshold=3000000
                    )
                    self.logger.info('_' * 100)
                    self.logger.info(f'Configuration {i}:')
                    for __x in X_tup:
                        X_str = np.array2string(
                            __x.numpy(force=True), **FLOAT_ARRAY_FORMAT
                        ).replace("[", " ").replace("]", " ")
                        self.logger.info(f'{X_str}\n')
                    del X_str, X_tup
                    if self.verbose > 2:
                        self.logger.info(f'Velocities {i}:')
                        for __x in V_tup:
                            V_str = np.array2string(
                                __x.numpy(force=True), **FLOAT_ARRAY_FORMAT
                            ).replace("[", " ").replace("]", " ")
                            self.logger.info(f'{V_str}\n')
                        del V_str
                    self.logger.info('_' * 100)
                _do_print = False

            # Update X, V
            #with th.profiler.record_function('MAIN UPDATE <<<<<'):
            X, V, Energy, Forces = self._updateXV(
                X, V, Forces,
                func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                masses, atom_masks, is_grad_func_contain_y, batch_indices
            )

            # Correct barycentric transition
            #with th.profiler.record_function('MOVE TO CENTER <<<<<'):
            if is_fix_mass_center and (i % move_to_center_freq == 0):
                dX = MASS_CENTER - self.calc_mass_center(masses, masses_short, X)
                dV = - self.calc_mass_center(masses, masses_short, V)
                X.add_(dX)  # (n_batch, n_atom, n_dim) - (n_batch, 1, n_dim)
                V.add_(dV)
            self.time_now += self.time_step

    # OVERRIDE THIS METHOD TO IMPLEMENT BatchMD UNDER VARIOUS ENSEMBLES.
    def _updateXV(
            self,
            X,
            V,
            Force,
            func,
            grad_func_,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            masses,
            atom_masks,
            is_grad_func_contain_y,
            batch_indices
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """ Update X, V, Force, and return X, V, Energy, Force. """
        raise NotImplementedError
        # return X, V, Energy, Force
