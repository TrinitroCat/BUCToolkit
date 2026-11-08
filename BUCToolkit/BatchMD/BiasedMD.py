#  Copyright (c) 2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: BiasedMD.py
#  Environment: Python 3.12


from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import inspect
import warnings

import numpy as np
import torch as th
from torch import nn

from ._BaseMD import _BaseMD
from .NVE import NVE
from .NVT import NVT


class BiasedMD:
    """
    External-biased potentials mainly for MD.
    """
    _md_cls = _BaseMD

    def __init__(
            self,
            *args,
            is_output_ext_EF: bool = True,
            **kwargs,
    ):
        """
        A wrapper of _BaseMD that forwards its all methods with modifying the _calc_EF
            to add the external potentials
        Args:
            *args: All args of required MD,
            is_args_appended: whether to append the args and kwargs of the main function
                (i.e., func and grad_func passed into _BaseMD.run(...)) to them of external potentials.
            is_output_ext_EF: whether to register `ExternalEnergy` and `ExternalForces` as the dump and log
                quantities.
            **kwargs:
        """
        object.__setattr__(self, "_md", self._md_cls(*args, **kwargs))

        self.ext_pot_grad_kwargs = None
        self.ext_pot_grad_args = None
        self.ext_pot_grad: None | Callable = None
        self.ext_pot_kwargs = None
        self.ext_pot_args = None
        self.ext_pot: None | Callable = None
        self.ext_is_grad_contain_y = None
        self.ext_is_dynamic = None
        self.is_output_ext_EF = bool(is_output_ext_EF)
        self.is_args_append_main_func = False
        self._is_args_appended = False  # a flag to avoid repeatedly append func_args when is_args_append_main_func is True
        self.__is_set_ext_func = False
        self._ext_energy_buf = None  # the energy of external potential
        self._ext_forces_buf = None  # the forces of external potential

        # monkey patch to replace original obj methods
        object.__setattr__(self._md, 'initialize', self.initialize)  # to avoid once MD impl its own `__setattr__`
        object.__setattr__(self._md, '_init_check_y_grad', self._init_check_y_grad)
        object.__setattr__(self._md, '_calc_EF', self._calc_EF)

    @classmethod
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls._md_cls is not None:
            cls.__signature__ = inspect.signature(cls._md_cls)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._md, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._md, name, value)

    @staticmethod
    def _auto_grad_func(X: th.Tensor, y: th.Tensor | None = None, *args):
        """ Here args is for compatible to g(X, y, t), since t is automatically contained by autograd of f """
        grad_shape = th.ones_like(y)
        g = th.autograd.grad(y, X, grad_shape)
        return g[0]

    def set_external_potential(
            self,
            potential_func: Callable,
            potential_grad_func: Callable | None = None,
            func_args:Tuple = tuple(),
            func_kwargs: None | Dict = None,
            grad_func_args: Tuple = tuple(),
            grad_func_kwargs: None | Dict = None,
            is_extpot_grad_contain_y: bool = True,
            is_dynamic: bool = False,
            is_args_append_main_func: bool = False,
    ):
        """
        Set the external potential function.
        Args:
            potential_func: the function of external potential
            potential_grad_func: the gradient of external potential. if None, torch.autograd is applied.
            func_args: arguments of `potential_func`
            func_kwargs: key words arguments of `potential_func`
            grad_func_args: arguments of `potential_grad_func`
            grad_func_kwargs: key words arguments of `potential_grad_func`
            is_extpot_grad_contain_y: whether `potential_grad_func` needs dependent variable y as input.
                if True, potential_grad_func = g(X, y), else g(X);
                if is_dynamic is also True, potential_grad_func is g(X, y, t) / g(X, t)
            is_dynamic: whether potential function containing time as independent variables.
                if True, the signature of potential_func is f(X, t), otherwise f(X), and
                the signature of potential_grad_func is g(X, y, t) / g(X, t), otherwise g(X, y) / g(X)
            is_args_append_main_func: if True, the (grad_)func_(kw)args of the external potential
                will be appended to the args and kwargs of main function that is passed into the self.run(...).

        Returns:

        """
        if func_kwargs is None: func_kwargs = {}
        if grad_func_kwargs is None: grad_func_kwargs = {}
        if not callable(potential_func):
            raise TypeError(f"`potential_func` must be callable, but got {type(potential_func)}.")
        if not (callable(potential_grad_func) or (potential_grad_func is None)):
            raise TypeError(f"`potential_grad_func` must be callable, but got {type(potential_grad_func)}.")
        if not isinstance(func_args, (tuple, list)):
            raise TypeError(f"`func_args` must be tuple or list, but got {type(func_args)}.")
        if not isinstance(func_kwargs, dict):
            raise TypeError(f"`func_kwargs` must be dict, but got {type(func_kwargs)}.")
        if not isinstance(grad_func_args, (tuple, list)):
            raise TypeError(f"`grad_func_args` must be tuple or list, but got {type(grad_func_args)}.")
        if not isinstance(grad_func_kwargs, dict):
            raise TypeError(f"`grad_func_kwargs` must be dict, but got {type(grad_func_kwargs)}.")
        self.ext_pot = potential_func
        self.ext_pot_args = tuple(func_args)
        self.ext_pot_kwargs = func_kwargs
        # ext_grad_func
        if potential_grad_func is None:
            self.ext_pot_grad = self._auto_grad_func
            is_extpot_grad_contain_y = True
        else:
            self.ext_pot_grad = potential_grad_func
        # args inherit
        self.is_args_append_main_func = bool(is_args_append_main_func)

        self.ext_pot_grad_args = tuple(grad_func_args)
        self.ext_pot_grad_kwargs = grad_func_kwargs
        self.ext_is_grad_contain_y = is_extpot_grad_contain_y
        self.ext_is_dynamic = is_dynamic
        self.__is_set_ext_func = True
        # reset the args flags
        self._is_args_appended = False

    def initialize(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            masses: th.Tensor,
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: tuple = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: tuple = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            require_grad: bool = False,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            is_fix_mass_center: bool = False
    ) -> None:
        if not self.__is_set_ext_func:
            err_msg = 'Please set the external potential function by `self.set_external_potential(...)` first.'
            if self._md.verbose > 0:
                self._md.logger.fatal(err_msg)
            raise RuntimeError(err_msg)
        # handle the args/kwargs pass
        if self.is_args_append_main_func:
            if self._is_args_appended:
                # if already appended, replace the corresponding position
                _len = len(func_args)
                ext_pot_args = list(self.ext_pot_args)
                ext_pot_args[:_len] = func_args
                self.ext_pot_args = tuple(ext_pot_args)
                _len = len(grad_func_args)
                ext_pot_grad_args = list(self.ext_pot_grad_args)
                ext_pot_grad_args[:_len] = grad_func_args
                self.ext_pot_grad_args = tuple(ext_pot_grad_args)
            else:
                self.ext_pot_args = func_args + self.ext_pot_args
                self.ext_pot_grad_args = grad_func_args + self.ext_pot_grad_args
                self._is_args_appended = True
            self.ext_pot_kwargs = self.ext_pot_kwargs | func_kwargs if func_kwargs is not None else self.ext_pot_kwargs
            self.ext_pot_grad_kwargs = self.ext_pot_grad_kwargs | grad_func_kwargs \
                if grad_func_kwargs is not None else self.ext_pot_grad_kwargs

        type(self._md).initialize(
            self._md,
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
            is_fix_mass_center=is_fix_mass_center
        )

        # handle the energy and forces buffer of pure external potentials
        if self.is_output_ext_EF:
            n_batch = len(batch_indices) if batch_indices is not None else X.shape[0]
            self._ext_energy_buf = th.zeros(n_batch, device=X.device, dtype=X.dtype)
            self._ext_forces_buf = th.zeros_like(X)
            self._md.register_extra_dump_vars(
                ExternalEnergy=self._ext_energy_buf,
                ExternalForces=self._ext_forces_buf,
            )
            self._md.register_extra_print_vars(
                ExternalEnergy=self._ext_energy_buf,
                ExternalForces=self._ext_forces_buf,
            )

    def _init_check_y_grad(
            self,
            func: Callable,
            X: th.Tensor,
            grad_func: Callable,
            func_args: Tuple,
            func_kwargs: Dict,
            grad_func_args: Tuple,
            grad_func_kwargs: Dict,
            is_grad_func_contain_y: bool,
            require_grad: bool,
            atom_masks: th.Tensor,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | None,
    ):
        s = type(self._md)._init_check_y_grad(
            self._md,
            func=func,
            X=X,
            grad_func=grad_func,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            atom_masks=atom_masks,
            batch_indices=batch_indices,
        )
        X.requires_grad_(self.ext_is_grad_contain_y)
        with th.set_grad_enabled(self.ext_is_grad_contain_y):
            if self.ext_is_dynamic:
                ext_energy = self.ext_pot(
                    X,
                    self._md.time_now,
                    *self.ext_pot_args,
                    **self.ext_pot_kwargs
                )
                if self.ext_is_grad_contain_y:
                    ext_grad = self.ext_pot_grad(
                        X,
                        ext_energy,
                        self._md.time_now,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )
                else:
                    ext_grad = self.ext_pot_grad(
                        X,
                        self._md.time_now,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )
            else:
                ext_energy = self.ext_pot(
                    X,
                    *self.ext_pot_args,
                    **self.ext_pot_kwargs
                )
                if self.ext_is_grad_contain_y:
                    ext_grad = self.ext_pot_grad(
                        X,
                        ext_energy,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )
                else:
                    ext_grad = self.ext_pot_grad(
                        X,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )
        if batch_indices is not None:
            if len(ext_energy) != len(batch_indices):  # check batch number of output
                raise RuntimeError(
                    f'batch number of `func` output ({len(ext_energy)}) '
                    f'does not match the input `batch_indices` ({len(batch_indices)})'
                )
        if ext_energy.shape != s.Energy.shape:
            raise RuntimeError(
                f"The shape of `ext_energy` {ext_energy} does not match the shape of main energy {s.Energy.shape}"
            )
        if ext_grad.shape != X.shape:
            raise RuntimeError(f"Forces shape ({ext_grad.shape}) does not match input ({X.shape}).")

        ext_forces = ext_grad.neg_()  # in-place update does not matter
        s.Energy += ext_energy
        s.Force += ext_forces * atom_masks

        # store the 1st step ext EF
        if self.is_output_ext_EF:
            self._ext_energy_buf.copy_(ext_energy)
            self._ext_forces_buf.copy_(ext_forces)
            s.ExternalEnergy = self._ext_energy_buf
            s.ExternalForces = self._ext_forces_buf

        return s

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

        Energy, Forces = type(self._md)._calc_EF(
            self._md,
            func=func,
            X=X,
            grad_func_=grad_func_,
            func_args=func_args,
            func_kwargs=func_kwargs,
            grad_func_args=grad_func_args,
            grad_func_kwargs=grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
        )
        X.requires_grad_(self.ext_is_grad_contain_y)
        with th.set_grad_enabled(self.ext_is_grad_contain_y):
            if self.ext_is_dynamic:
                ext_energy = self.ext_pot(
                    X,
                    self._md.time_now,
                    *self.ext_pot_args,
                    **self.ext_pot_kwargs
                )
                if self.ext_is_grad_contain_y:
                    ext_grad = self.ext_pot_grad(
                        X,
                        ext_energy,
                        self._md.time_now,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )
                else:
                    ext_grad = self.ext_pot_grad(
                        X,
                        self._md.time_now,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )
            else:
                ext_energy = self.ext_pot(
                    X,
                    *self.ext_pot_args,
                    **self.ext_pot_kwargs
                )
                if self.ext_is_grad_contain_y:
                    ext_grad = self.ext_pot_grad(
                        X,
                        ext_energy,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )
                else:
                    ext_grad = self.ext_pot_grad(
                        X,
                        *self.ext_pot_grad_args,
                        **self.ext_pot_grad_kwargs
                    )

        ext_force = ext_grad.neg_()  # in-place update does not matter
        Energy += ext_energy
        Forces += ext_force

        if self.is_output_ext_EF:
            self._md.s.ExternalEnergy = ext_energy.detach()
            self._md.s.ExternalForces = ext_force.detach()

        return Energy, Forces


class BiasedNVE(BiasedMD):
    _md_cls = NVE

class BiasedNVT(BiasedMD):
    _md_cls = NVT
