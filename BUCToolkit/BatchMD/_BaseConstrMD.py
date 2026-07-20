""" Molecular Dynamics base framework with constrains """

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVE.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Set, Tuple  # noqa: F401
import os

import torch as th
from torch import nn
import numpy as np

from ._BaseMD import _BaseMD
from BUCToolkit.Bases.BaseConstraints import BaseConstr
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BUCToolkit.utils.grad_functions import bjvp, bhvp
from BUCToolkit.utils._Element_info import DTYPE

FLOAT_TYPE = os.environ.get('BT_FLOAT_TYPE', 'float32')
FLOAT_TYPE = DTYPE.get(FLOAT_TYPE, th.float32)


class _BaseConstrMD(_BaseMD):
    """
    Constrained Base Dynamics

    Args:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        constr_func: Callable, a tuple of Python functions as the constraint functions s_k(X) that map R^n -> R^k. It takes one or more arguments, one of which must be a Tensor, and returns one Tensor with shape (k, ). `None` for identity function. see example below.
        constr_val: Callable[th.Tensor[1], th.Tensor] | th.Tensor, the constraint value of `constr_func`, i.e., constraints are `constr_func(X) = constr_val`.
        By defining it as a callable constr_val = constr_val(t) where `t` is a scalar Tensor, it can be set to the time-dependent constraints.
        constr_threshold: float, the threshold of constraint convergence (error of manifold violation)
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: str|torch.device, device that program rum on.
        verbose: int, control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.

    Examples for constrains:
        def constr_func(X):
            y = list()
            # X: shape(N, D), Note the batch dimension would NOT be considered in constraints calculation of X.
            # fix the distance between atoms (2, 4), (3, 7), (5, 8) into corresponding `constr_val[:3]`
            y.append(th.linalg.norm(X[[2, 3, 5]] - X[[4, 7, 8]], dim=-1))

            # fix the angle of atom7-atom5-atom8 and atom11-atom9-atom12 into corresponding `constr_val[3:6]`
            x1 = X[[5, 9]]
            x2 = X[[7, 11]]
            x3 = X[[8, 12]]
            y.append(
                (
                    th.sum((x2 - x1) * (x3 - x1))
                ) / (th.linalg.norm(x2 - x1) * th.linalg.norm(x3 - x1))
            )
            z = th.cat(y)
            return z

    Methods:
        run: run BatchMD.

    """

    #: Additional allowed dump / log names beyond the base MD set.
    ALLOWED_QUANTITIES: Set[str] = _BaseMD.ALLOWED_QUANTITIES | {'Fc', 'G', 'w'}

    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            constr_func: Callable | None = None,
            constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor | None = None,
            constr_threshold: float = 1e-5,
            require_fixman: bool = False,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        self._constr = BaseConstr(
            constr_func,
            constr_val,
            constr_threshold,
            require_fixman,
            device,
            verbose,
        )
        _BaseMD.__init__(
            self,
            time_step,
            max_step,
            T_init,
            output_file,
            output_structures_per_step,
            device,
            verbose,
            is_compile=False,  # compiler does not support the proxy scheme `__getattr__`, so that must turn it off.
        )

    def __getattr__(self, name):
        """
        Do a proxy that transmits methods in BaseConstr.

        """
        if '_constr' in self.__dict__:
            constr = self._constr
            if hasattr(constr, name):
                return getattr(constr, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # I should synchronize the homonymous attr manually.
    @property
    def time_step(self):
        return self.__dict__['time_step']

    @time_step.setter
    def time_step(self, value):
        self.__dict__['time_step'] = value
        self._constr.time_step = value

    @property
    def time_now(self):
        return self.__dict__['time_now']

    @time_now.setter
    def time_now(self, value):
        self.__dict__['time_now'] = value
        self._constr.time_now = value

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
    ):
        self._constr.sqrtM = th.sqrt(masses)  # M^1/2, (n_batch, n_atoms, n_dim)
        self._constr.negsqrtM = 1 / th.sqrt(masses)  # M^-1/2
        _y_check = th.vmap(self.constr_func)(X)
        if self._lazy_calc_constr_val:
            self._constr.constr_val_now = _y_check
        # check constr_val shape
        if _y_check.shape != self.constr_val_now.shape:
            raise RuntimeError(
                f'`constr_val` must have the same shape as what constr_func returned {self.constr_val_now.shape}, but got {_y_check.shape}.'
            )
        jac, y = self._jacobian(X)
        if self.verbose:
            self.logger.info(
                f'Constraint values are now {np.array2string(self.constr_val_now.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}'
            )
        if y.ndim != 2:
            raise ValueError(f'`constr_func` must return a 2D tensor of shape (n_batch, n_constr), but got {y.shape}.')
        self._do_qr(jac)
        # Register constraint diagnostics so they appear in StdContainer
        # s / s_cpu and are automatically dumped and logged.
        _n_batch = X.shape[0]
        _n_constr = self.R.shape[-1]
        _Fc = th.zeros(_n_batch, _n_constr, device=self.device, dtype=FLOAT_TYPE)
        if self.require_fixman:
            _G = th.zeros(_n_batch, _n_constr, device=self.device, dtype=FLOAT_TYPE)
            _w = th.zeros(_n_batch, device=self.device, dtype=FLOAT_TYPE)
            _constraint_vars = {'Fc': _Fc, 'G': _G, 'w': _w}
        else:
            _constraint_vars = {'Fc': _Fc}

        # ``initialize()`` runs for every invocation of ``run()``. Keep the
        # registration persistent, but refresh the placeholder tensors because
        # a later run may use a different batch size or number of constraints.
        # Registering the same names twice would otherwise raise on the second
        # run even though ``reset_register_vars()`` intentionally preserves
        # late-bound extension quantities.
        self._extra_vars.update(_constraint_vars)
        _new_print_vars = {
            _name: _value for _name, _value in _constraint_vars.items()
            if _name not in self._extra_print_names
        }
        _new_dump_vars = {
            _name: _value for _name, _value in _constraint_vars.items()
            if _name not in self._extra_dump_names
        }
        if _new_print_vars:
            self.register_extra_print_vars(**_new_print_vars)
        if _new_dump_vars:
            self.register_extra_dump_vars(**_new_dump_vars)

        ProjV = self._project1(V_init)
        Ek = th.sum(
            masses * V_init ** 2,
            dim=(-2, -1),
            keepdim=True
        )
        Ek_p = th.sum(
            masses * ProjV ** 2,
            dim=(-2, -1),
            keepdim=True
        )
        V_init.copy_(th.where(Ek_p < 1e-5, 0., th.sqrt(Ek/Ek_p) * ProjV))
        self.free_degree -= jac.shape[1]  # reduce the constr. free deg.
        # recalculate target Ek under constraints
        _, n_atom, n_dim = X.shape
        # target kinetic energy for NVT|NPT ensembles
        if batch_indices:  # Unit: eV/atom. Boltzmann constant kB = 8.6173332621e-5 eV/K
            self.EK_TARGET = th.tensor(
                [((self.free_degree / 2.) * 8.617333262145e-5 * self.T_init) for _n_atom in batch_indices],
                dtype=X.dtype,
                device=self.device
            )
        else:
            self.EK_TARGET = (self.free_degree / 2.) * 8.617333262145e-5 * self.T_init
        # calc. constr. intensity
        n_batch, n_constr, _ = self.R.shape
        self._constr.X_cache = th.empty_like(X)
