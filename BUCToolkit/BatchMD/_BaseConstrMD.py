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
        verbose: print level. 0 is silent, 1 prints selected scalars only, and
            2 or greater also prints selected arrays.
        dump_quantities: names written to the binary trajectory. By default,
            constrained trajectories include ``Fc`` and, when Fixman terms
            are enabled, ``G`` and the Blue-Moon reweighting factor ``w``.
        log_quantities: names printed in the text log. Constraint diagnostics
            follow the dump defaults and are included unless an explicit
            selection is provided. Array selection does not override the
            ``verbose >= 2`` requirement.

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
            verbose: int = 2,
            dump_quantities: Tuple[str, ...] | List[str] | None = None,
            log_quantities: Tuple[str, ...] | List[str] | None = None,
    ):
        if dump_quantities is None:
            dump_quantities = ('Energy', 'X', 'V', 'Force', 'Fc')
            if require_fixman:
                dump_quantities += ('G', 'w')
        if log_quantities is None:
            log_quantities = ('Energy', 'Ek', 'temperature', 'X', 'V', 'Force', 'Fc')
            if require_fixman:
                log_quantities += ('G', 'w')

        if not require_fixman:
            unavailable = {'G', 'w'} & (set(dump_quantities) | set(log_quantities))
            if unavailable:
                raise ValueError(
                    f'Fixman quantities {sorted(unavailable)} require '
                    f'require_fixman=True.'
                )

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
            dump_quantities=dump_quantities,
            log_quantities=log_quantities,
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
        # BaseConstr owns mass factors, lazy targets, eager validation,
        # Jacobian compilation, QR initialization, and X_cache.
        # Keeping that lifecycle in one public method prevents proxy users from
        # silently missing new constraint initialization steps.
        self._constr.initialize(
            func=func,
            X=X,
            Element_list=Element_list,
            masses=masses,
            V_init=V_init,
            grad_func=grad_func,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            batch_indices=batch_indices,
            fixed_atom_tensor=fixed_atom_tensor,
            is_fix_mass_center=is_fix_mass_center,
        )
        # re-initialise the `time_new` to ensure the constr value correct when calling `run` more than one time.
        self.time_now = th.scalar_tensor(0., device=self.device)
        # Register selected constraint fields with correctly shaped prototypes.
        # Ordered dictionaries update repeated names without duplicating them.
        _n_batch = X.shape[0]
        _n_constr = self.R.shape[-1]
        _Fc = th.zeros(_n_batch, _n_constr, device=self.device, dtype=FLOAT_TYPE)
        if self.require_fixman:
            _G = th.zeros(_n_batch, _n_constr, device=self.device, dtype=FLOAT_TYPE)
            _w = th.zeros(_n_batch, device=self.device, dtype=FLOAT_TYPE)
            _constraint_vars = {'Fc': _Fc, 'G': _G, 'w': _w}
        else:
            _constraint_vars = {'Fc': _Fc}

        _dump_constraint_vars = {
            _name: _value for _name, _value in _constraint_vars.items()
            if _name in self.get_dump_vars()
        }
        _log_constraint_vars = {
            _name: _value for _name, _value in _constraint_vars.items()
            if _name in self.get_log_vars()
        }
        if _dump_constraint_vars:
            self.register_extra_dump_vars(**_dump_constraint_vars)
        if _log_constraint_vars:
            self.register_extra_print_vars(**_log_constraint_vars)

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
        self.free_degree -= _n_constr  # reduce the constr. free deg.
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
