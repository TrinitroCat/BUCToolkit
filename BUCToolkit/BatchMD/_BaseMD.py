""" Molecular Dynamics via Verlet algo. """
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _BaseMD.py
#  Environment: Python 3.12

import sys
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Set, Tuple  # noqa: F401
import time
import math
import logging
import warnings
import os
import threading, queue

import torch as th
from torch import nn

import numpy as np

from BUCToolkit.utils._Element_info import MASS, N_MASS, ATOMIC_NUMBER, ATOMIC_SYMBOL, DTYPE
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BUCToolkit.utils.index_ops import index_reduce
from BUCToolkit.utils.function_utils import preload_func
from BUCToolkit.Bases.BaseMotion import BaseMotion
from BUCToolkit.Bases.StdContainer import StdContainer

FLOAT_TYPE = os.environ.get('BT_FLOAT_TYPE', 'float32')
FLOAT_TYPE = DTYPE.get(FLOAT_TYPE, th.float32)


class _BaseMD(BaseMotion):
    """ Base BatchMD """

    #: Quantities known to the MD framework that may appear in
    #: ``dump_quantities`` / ``log_quantities``.  Subclasses extend
    #: this set (e.g. ``_BaseConstrMD`` adds ``'Fc'``, ``'G'``, ``'w'``).
    ALLOWED_QUANTITIES: Set[str] = {'Energy', 'Ek', 'temperature', 'X', 'V', 'Force'}

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
            dump_quantities: Tuple[str, ...] | List[str] = ('Energy', 'X', 'V', 'Force'),
            log_quantities: Tuple[str, ...] | List[str] = ('Energy', 'Ek', 'temperature', 'X', 'V'),
    ):
        """
        Parameters:
            time_step: float, time per step (fs).
            max_step: maximum steps.
            T_init: initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution.
                If V_init is given, T_init will be ignored.
            output_file: the path to the binary file that stores trajectories. If None, tractories will not output.
            output_structures_per_step: int, output structures per output_structures_per_step steps.
            device: device that program run on.
            verbose: control the detailed degree of output text information.
                0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.
                Note: verbose > 0 will be very slow, especially for computation on GPU.
            is_compile: whether to use jit to compile integrator or not.
            compile_kwargs: keyword arguments passed to compile. Only work when is_compile is True.
            dump_quantities: names of StdContainer attributes to write to the
                binary trajectory file.  Must be a subset of :attr:`ALLOWED_QUANTITIES`.
            log_quantities: names of StdContainer attributes to include in the
                per-step log queue / inline print.  Must be a subset of
                :attr:`ALLOWED_QUANTITIES`.
        """
        self.time_step = time_step
        self.time_now = th.scalar_tensor(0., device=device)  # the accumulated time
        assert (max_step > 0) and isinstance(max_step, int), f'max_step must be a positive integer, but occurred {max_step}.'
        self.max_step = int(max_step)
        self.T_init = float(T_init)
        self.output_structures_per_step = int(output_structures_per_step)
        self.device = device if isinstance(device, th.device) else th.device(device)
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

        # logging & dumper
        super().__init__(output_file)
        self.init_logger('Main.MD')
        self._setup_register_vars(dump_quantities, log_quantities)

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
        """Consumer thread: drain queue, call ``dumper.step()``, signal done.

        Queue item: ``(dumper, event, *cpu_tensors)``.
        *event* may be ``None`` on the CPU path (no async D2H).

        ``self._dump_done`` is set in ``finally`` after ``step()`` to
        guarantee the handshake even on exception.

        Args:
            q: Producer/consumer queue. Normal items have the form
                ``(dumper, event, *cpu_tensors)`` and a sentinel whose dumper
                is ``None`` terminates the thread. ``event`` is a CUDA event or
                ``None`` on CPU; tensors follow :meth:`get_dump_vars` order.

        Returns:
            None. Background-thread errors are written to ``self.logger``.
        """
        while True:
            items = q.get()
            dumper, event = items[0], items[1]
            if dumper is None:
                break
            try:
                if event is not None:
                    event.synchronize()
                dumper.step(*(t.numpy() for t in items[2:]))
            except Exception as e:
                self.logger.error(f"Error: Failed to dump data due to \"{e}\"")
            finally:
                self._dump_done.set()

    def _do_async_print(self, q: queue.Queue):
        """Consumer thread: format and log.

        Drains *q* (items packed in :meth:`get_log_vars` order after
        the ``(i, copy_event, batch_indices)`` header) and dispatches
        by tensor dimensionality:

        * ndim ≤ 1 — printed inline as ``name = value`` under a
          ``Step:`` header (verbosity ≥ 1).
        * ndim ≥ 2 — printed via :meth:`handle_arrays_print`
          (verbosity ≥ 2).

        ``self._print_done`` is set in ``finally`` to guarantee the
        handshake even on exception.

        Args:
            q: Producer/consumer queue. Normal items contain
                ``(step, copy_event, batch_indices, *cpu_tensors)`` with
                tensors ordered by :meth:`get_log_vars`; a sentinel whose
                first item is ``None`` terminates the thread.

        Returns:
            None. The method formats registered scalar and array values and
            signals ``self._print_done`` after every consumed snapshot.
        """
        formatter1 = {'float': '{:> .2f}'.format}
        i = 0
        while True:
            items = q.get()
            if items[0] is None:
                break
            i, copy_event, batch_indices = items[0], items[1], items[2]
            try:
                if copy_event is not None:
                    copy_event.synchronize()

                # Build name → value map from the log-var order
                _data = dict(zip(self.get_log_vars(), items[3:]))
                _valid = [(_n, _v) for _n, _v in _data.items() if _n and _v is not None]
                _scalars = [(_n, _v) for _n, _v in _valid if _v.ndim <= 1]
                _arrays  = [(_n, _v) for _n, _v in _valid if _v.ndim >= 2]

                # Scalars — verbose ≥ 1
                if _scalars and self.verbose > 0:
                    np.set_printoptions(
                        precision=8, linewidth=1024, floatmode='fixed',
                        suppress=True, formatter=formatter1, threshold=2000,
                    )
                    self.logger.info(f'Step: {i:>12d}')
                    for _n, _v in _scalars:
                        self.logger.info(
                            f'\t{_n:<12s} = '
                            f'{np.array2string(_v.numpy(), **SCIENTIFIC_ARRAY_FORMAT)}'
                        )

                # Arrays — verbose ≥ 2
                if _arrays and self.verbose > 1:
                    self.handle_arrays_print(
                        self.logger, batch_indices, self.batch_slice_indx,
                        [[_v for _, _v in _arrays]],
                        [[_n for _, _n in _arrays]],
                        verbose=self.verbose, force=False,
                    )
            except Exception as e:
                self.logger.error(f"Error: Failed to logout at {i}-th iteration due to \"{e}\".")
            finally:
                self._print_done.set()

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
            X, Cell_vector, V_init = self.handle_dtype_device(FLOAT_TYPE, self.device, X, Cell_vector, V_init)

            # Reset to the base set each run(); subclasses may extend in
            # __init__ or initialize() via register_extra_dump_vars /
            # register_extra_print_vars.
            self.reset_register_vars()
            if self.device.type == "cuda":
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
            elif self.device.type == "cpu":
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
            if not self._HOLD_DUMPER:
                self.dumper.close()

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

        n_true_batch, batch_indices, self.batch_tensor, self.batch_scatter, self.batch_slice_indx = self.handle_batch_indices(
            batch_indices, n_batch, device=self.device
        )
        self.scatter_dim_out_size = self.batch_scatter.max().item() + 1 if self.batch_scatter is not None else None

        # Manage Atomic Type & Masses
        masses = list()
        atomic_numbers = list()
        for _Elem in Element_list:
            if not isinstance(_Elem, list): raise TypeError(f'Expected `Element_list` of List[List[int|str]], but got List[{type(_Elem)}].')
            atomic_numbers.append([ATOMIC_SYMBOL[__elem] if isinstance(__elem, str) else int(__elem) for __elem in _Elem])
            masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
        masses_short = th.tensor(masses, dtype=FLOAT_TYPE, device=self.device)  # (n_batch, n_atom)
        masses = masses_short.unsqueeze(-1).expand_as(X).contiguous()  # (n_batch, n_atom, n_dim)
        # grad_func
        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(
            grad_func,
            is_grad_func_contain_y,
            require_grad,
        )

        # Selective dynamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)

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
            self.p_iota = th.zeros(1, len(batch_indices), 1, device=self.device, dtype=FLOAT_TYPE)
        else:
            self.p_iota = th.zeros(n_batch, 1, 1, device=self.device, dtype=FLOAT_TYPE)
        # whether grad needs autograd
        self.require_grad = require_grad

        # initialize the dumper
        X_arr = X.numpy(force=True)
        _x_dtype = X_arr.dtype.str
        atom_masks_arr = atom_masks.numpy(force=True).astype(_x_dtype)
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
        # (main data arrays registered dynamically after first model eval)
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

            # wrap into StdContainer
            s = StdContainer(
                X=X,
                V=V,
                Force=Forces,
                Energy=Energy,
                Ek=th.zeros_like(Energy),
                temperature=th.zeros_like(Energy),
                **self._extra_vars
            )

            # dynamic dump buffers
            dump_names = self.get_dump_vars()  # self._dump_vars
            log_names = self.get_log_vars()
            total_names = self.get_transfer_vars()
            s_cpu, s_buf = self._allocate_cpu_buffers(s, total_names, self.device)
            _num_dump = math.ceil(self.max_step / self.output_structures_per_step)
            dumper.start_from_arrays(
                _num_dump,
                *(getattr(s_cpu, name).numpy() for name in dump_names),
                names=dump_names,
            )

            # preload a graph of Ek, T
            Ek_T_graph = th.cuda.CUDAGraph()
            with th.cuda.graph(Ek_T_graph):
                _e_tmp, _t_tmp = self._reduce_Ek_T(batch_indices, masses, s.V)
                s.Ek.copy_(_e_tmp)
                s.temperature.copy_(_t_tmp)
            self.Ek_T_graph = Ek_T_graph
            self.Ek = s.Ek
            # preload a graph of mass center
            if is_fix_mass_center:
                mass_center_graph = th.cuda.CUDAGraph()
                with th.cuda.graph(mass_center_graph):
                    _dX = MASS_CENTER - self.calc_mass_center(masses, masses_short, s.X)
                    _dV = - self.calc_mass_center(masses, masses_short, s.V)
                    s.X.add_(_dX)  # (n_batch, n_atom, n_dim) - (n_batch, 1, n_dim)
                    s.V.add_(_dV)
            else:
                mass_center_graph = None

            copy_stream = th.cuda.Stream()
            copy_event = th.cuda.Event()
            compute_event = th.cuda.Event()
            compute_event.record(th.cuda.default_stream(self.device))  # the default stream is the compute (main) stream.
            # launch the dumping thread
            dump_queue = queue.Queue(maxsize=1)
            dump_thread = threading.Thread(target=self._do_async_dump, args=(dump_queue, ), daemon=True)
            logout_queue = queue.Queue(maxsize=1)
            logout_thread = threading.Thread(target=self._do_async_print, args=(logout_queue, ), daemon=True)
            # consumer→main handshake events
            self._dump_done = threading.Event()
            self._print_done = threading.Event()
            self._dump_done.set()
            self._print_done.set()
            try:
                dump_thread.start()
                logout_thread.start()
                #ptlist = list()  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                if self.is_compile:
                    _main_loop = th.compile(self._main_for_loop_cuda, **self.compile_kwargs, disable=(not self.is_compile))
                else:
                    _main_loop = self._main_for_loop_cuda
                _main_loop(
                    s,
                    s_cpu,
                    s_buf,
                    dump_names,
                    log_names,
                    total_names,
                    Ek_T_graph,
                    compute_event,
                    copy_stream,
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
                th.cuda.synchronize()
                if self.verbose > 0:
                    self.logger.info(f'MAIN LOOP DONE. Elapsed time: {time.perf_counter() - t_main_loop:>5.4f} s')
            finally:
                dump_queue.put((None, None))
                dump_thread.join()
                logout_queue.put((None, None))
                logout_thread.join()

        del self.Ekt_vir
        #return ptlist  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def _main_for_loop_cuda(
            self,
            s,
            s_cpu,
            s_buf,
            dump_names,
            log_names,
            total_names,
            Ek_T_graph,
            compute_event,
            copy_stream,
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
        for i in range(self.max_step):
            Ek_T_graph.replay()
            self.Ek = s.Ek
            if i % self.output_structures_per_step == 0:
                # gate: wait for consumers from previous dump cycle
                self._dump_done.wait()
                self._print_done.wait()
                compute_event.wait(th.cuda.default_stream(self.device))
                th.cuda.default_stream(self.device).wait_event(copy_event)
                # dynamic dump transition
                self._transfer_buffers_D2H(s, s_buf, s_cpu, total_names, copy_stream, self.device)
                copy_event.record(copy_stream)
                # dump
                self._dump_done.clear()
                dump_queue.put((dumper, copy_event, *(getattr(s_cpu, _n) for _n in dump_names)))
                # print logs
                self._print_done.clear()
                logout_queue.put((i, copy_event, batch_indices, *(getattr(s_cpu, _n) for _n in log_names)))

            self._updateXV(
                s,
                func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                masses, atom_masks, is_grad_func_contain_y, batch_indices
            )
            compute_event.record(th.cuda.default_stream(self.device))

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
        n_true_batch, batch_indices, self.batch_tensor, self.batch_scatter, batch_slice_indx = self.handle_batch_indices(
            batch_indices, n_batch, device=self.device
        )
        self.scatter_dim_out_size = self.batch_scatter.max().item() + 1 if self.batch_scatter is not None else None

        # Manage Atomic Type & Masses
        masses = list()
        atomic_numbers = list()
        for _Elem in Element_list:
            if not isinstance(_Elem, list): raise TypeError(f'Expected `Element_list` of List[List[int|str]], but got List[{type(_Elem)}].')
            atomic_numbers.append([ATOMIC_SYMBOL[__elem] if isinstance(__elem, str) else ATOMIC_NUMBER[__elem] for __elem in _Elem])
            masses.append([MASS[__elem] if isinstance(__elem, str) else N_MASS[__elem] for __elem in _Elem])
        masses_short = th.tensor(masses, dtype=FLOAT_TYPE, device=self.device)  # (n_batch, n_atom)
        masses = masses_short.unsqueeze(-1).expand_as(X).contiguous()  # (n_batch, n_atom, n_dim)
        # grad_func
        grad_func_, require_grad, is_grad_func_contain_y = self.handle_grad_func(
            grad_func,
            is_grad_func_contain_y,
            require_grad,
        )

        # Selective dynamics
        atom_masks = self.handle_motion_mask(X, fixed_atom_tensor)
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
            self.p_iota = th.zeros(1, len(batch_indices), 1, device=self.device, dtype=FLOAT_TYPE)
        else:
            self.p_iota = th.zeros(n_batch, 1, 1, device=self.device, dtype=FLOAT_TYPE)
        # whether grad needs autograd
        self.require_grad = require_grad
        self.Ek_T_graph = None
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
        # (main data arrays registered dynamically after first model eval)
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

            # wrap into StdContainer
            s = StdContainer(
                X=X,
                V=V,
                Force=Forces,
                Energy=Energy,
                Ek=th.zeros_like(Energy),
                temperature=th.zeros_like(Energy),
                **self._extra_vars,
            )

            # dynamic dump buffers
            dump_names = self.get_dump_vars()
            log_names = self.get_log_vars()
            total_names = self.get_transfer_vars()
            _num_dump = math.ceil(self.max_step / self.output_structures_per_step)
            if dump_names:
                s_cpu, s_buf = self._allocate_cpu_buffers(s, total_names, self.device)
                dumper.start_from_arrays(
                    _num_dump,
                    *(getattr(s_cpu, name).numpy() for name in dump_names),
                    names=dump_names,
                )
            else:
                s_cpu, s_buf = None, None

            #ptlist = list()  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # --- async threading (same pattern as CUDA, GIL released during I/O) ---
            dump_queue = queue.Queue(maxsize=1)
            dump_thread = threading.Thread(target=self._do_async_dump, args=(dump_queue,), daemon=True)
            logout_queue = queue.Queue(maxsize=1)
            logout_thread = threading.Thread(target=self._do_async_print, args=(logout_queue,), daemon=True)
            self._dump_done = threading.Event()
            self._print_done = threading.Event()
            self._dump_done.set()
            self._print_done.set()
            try:
                dump_thread.start()
                logout_thread.start()
                t_main_loop = time.perf_counter()
                for i in range(self.max_step):
                    Ek, temperature = self._reduce_Ek_T(batch_indices, masses, s.V)
                    self.Ek = Ek
                    s.Ek.copy_(Ek)
                    s.temperature.copy_(temperature)

                    if i % self.output_structures_per_step == 0:
                        # Gate: wait for consumers from previous dump cycle
                        self._dump_done.wait()
                        self._print_done.wait()

                        # Snapshot s → s_cpu (CPU memcpy — synchronous, fast)
                        for _n in total_names:
                            getattr(s_cpu, _n).copy_(getattr(s, _n))

                        # Dispatch consumers (event=None: no GPU D2H to sync)
                        self._dump_done.clear()
                        dump_queue.put((dumper, None, *(getattr(s_cpu, _n) for _n in dump_names)))
                        self._print_done.clear()
                        logout_queue.put((i, None, batch_indices, *(getattr(s_cpu, _n) for _n in log_names)))

                    self._updateXV(
                        s, func, grad_func_, func_args, func_kwargs,
                        grad_func_args, grad_func_kwargs,
                        masses, atom_masks, is_grad_func_contain_y, batch_indices,
                    )

                    if is_fix_mass_center and (i % move_to_center_freq == 0):
                        dX = MASS_CENTER - self.calc_mass_center(masses, masses_short, s.X)
                        dV = - self.calc_mass_center(masses, masses_short, s.V)
                        s.X.add_(dX)
                        s.V.add_(dV)
                    self.time_now += self.time_step
                if self.verbose > 0:
                    self.logger.info(f'MAIN LOOP DONE. Elapsed time: {time.perf_counter() - t_main_loop:>5.4f} s')
            finally:
                dump_queue.put((None, None))
                dump_thread.join()
                logout_queue.put((None, None))
                logout_thread.join()

        del self.Ekt_vir
        #return ptlist  # test <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # OVERRIDE THIS METHOD TO IMPLEMENT BatchMD UNDER VARIOUS ENSEMBLES.
    def _updateXV(
            self,
            s: StdContainer,
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
    ) -> None:
        """Advance one ensemble-specific MD step in place.

        Args:
            s: Live :class:`StdContainer` containing at least ``X``, ``V``,
                ``Force``, and ``Energy``.
            func: Potential-energy callable.
            grad_func_: Normalized force/gradient callable.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            grad_func_args: Positional arguments forwarded to ``grad_func_``.
            grad_func_kwargs: Keyword arguments forwarded to ``grad_func_``.
            masses: Atomic masses broadcastable to the coordinate shape.
            atom_masks: Selective-dynamics mask multiplied into new forces.
            is_grad_func_contain_y: Whether ``grad_func_`` receives the energy
                output as its second positional argument.
            batch_indices: Irregular-batch atom counts, or ``None`` for a
                regular batch.

        Returns:
            None. Implementations update ``s.X``, ``s.V``, ``s.Force``, and
            ``s.Energy`` in place and may additionally update ensemble fields.
        """
        raise NotImplementedError
