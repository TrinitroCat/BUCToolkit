#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: BiasedMD.py
#  Environment: Python 3.12


from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import warnings

import numpy as np
import torch as th
from torch import nn

from BM4Ckit.utils.scatter_reduce import scatter_reduce
from ._BaseMD import _BaseMD
from .NVT import NVT
from .NVE import NVE


class BiasedMD(NVT):
    """
    Performing MD with given external-biased potentials.
    """
    def __init__(
            self,
            time_step: float,
            max_step: int,
            thermostat: Literal['Langevin', 'VR', 'Nose-Hoover', 'CSVR'],
            thermostat_config: Dict | None = None,
            T_init: float = 298.15,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        super().__init__(
            time_step,
            max_step,
            thermostat,
            thermostat_config,
            T_init,
            output_file,
            output_structures_per_step,
            device,
            verbose
        )
        self.ext_pot_grad_kwargs = None
        self.ext_pot_grad_args = None
        self.ext_pot_grad = None
        self.ext_pot_kwargs = None
        self.ext_pot_args = None
        self.ext_pot = None
        self.__is_set_ext_func = False

    @staticmethod
    def _auto_grad_func(X: th.Tensor, y: th.Tensor | None = None):
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
            is_grad_contain_y: bool = True,
            #is_dynamic: bool = False
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
            is_grad_contain_y: whether `potential_grad_func` needs dependent variable y as input. if True, potential_grad_func = g(X, y), else g(X)
            #is_dynamic: whether potential function containing time as independent variables. if True, potential_func = f(X, t), else f(X).

        Returns:

        """
        if func_kwargs is None: func_kwargs = {}
        if grad_func_kwargs is None: grad_func_kwargs = {}
        self.ext_pot = potential_func
        self.ext_pot_args = func_args
        self.ext_pot_kwargs = func_kwargs
        # ext_grad_func
        if potential_grad_func is None:
            self.ext_pot_grad = self._auto_grad_func
        else:
            self.ext_pot_grad = potential_grad_func

        self.ext_pot_grad_args = grad_func_args
        self.ext_pot_grad_kwargs = grad_func_kwargs
        self.is_grad_contain_y = is_grad_contain_y
        #self.is_dynamic = is_dynamic
        self.__is_set_ext_func = True

    def run(
            self,
            func: Any | nn.Module,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            V_init: th.Tensor | None = None,
            grad_func: Any | nn.Module = None,
            func_args: Tuple|List = tuple(),
            func_kwargs: Dict | None = None,
            grad_func_args: Tuple|List = tuple(),
            grad_func_kwargs: Dict | None = None,
            is_grad_func_contain_y: bool = True,
            batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
            fixed_atom_tensor: Optional[th.Tensor] = None,
            is_fix_mass_center: bool = False
    ) -> None:
        """

        Args:
            func:
            X:
            Element_list:
            V_init:
            grad_func:
            func_args:
            func_kwargs:
            grad_func_args:
            grad_func_kwargs:
            is_grad_func_contain_y:
            batch_indices:
            fixed_atom_tensor:
            is_fix_mass_center:

        Returns:

        """
        if func_kwargs is None: func_kwargs = dict()
        if grad_func_kwargs is None: grad_func_kwargs = dict()
        if grad_func is not None:
            _grad_func = grad_func
        else:
            _grad_func = self._auto_grad_func
            is_grad_func_contain_y = True

        if self.__is_set_ext_func:
            wrapper = _FuncWrapperWithExtPot(
                func,
                self.ext_pot,
                _grad_func,
                self.ext_pot_grad,
                len(func_args),
                func_kwargs.keys(),
                len(grad_func_args),
                grad_func_kwargs.keys()
            )
            wrapped_args = func_args + self.ext_pot_args
            wrapped_kwargs = func_kwargs.update(self.ext_pot_kwargs)
            wrapped_grad_args = grad_func_args + self.ext_pot_grad_args
            wrapped_grad_kwargs = grad_func_kwargs.update(self.ext_pot_grad_kwargs)

            wrapped_func = wrapper.Energy
            if self.is_grad_contain_y:
                wrapped_grad_func = wrapper.Grad_y
            else:
                wrapped_grad_func = wrapper.Grad

            res = super().run(
                wrapped_func,
                X,
                Element_list,
                V_init,
                wrapped_grad_func,
                wrapped_args,
                wrapped_kwargs,
                wrapped_grad_args,
                wrapped_grad_kwargs,
                is_grad_func_contain_y,
                batch_indices,
                fixed_atom_tensor,
                is_fix_mass_center
            )

            return res


class _FuncWrapperWithExtPot:
    def __init__(
            self,
            func,
            ext_func,
            grad_func,
            ext_grad_func,
            n_func_args = 0,
            k_func_kwargs = None,
            n_grad_func_args = 0,
            k_grad_func_kwargs = None,

    ) -> None:
        """
        A wrapper to add external function
        Wrap the f(X, ...) into f(X, ...) + ext_func(X, ...)

        And the wrapped func_args would be the concatenation of `func_args` and `ext_func_args`.
        Hence, the number of func_args, i.e., the `n_func_args` is required to determine that
         the first `n_func_args`-th args belong to `func` and the rest belong to `ext_func`

        Keyword Args is similar, that require the keys of `func_kwargs`, and the rest belongs to `ext_func`.

        `grad_func` is so, too.

        Args:
            func: the original function
            ext_func: the external adding function
            n_func_args: number of args that belongs to `func`
            k_func_kwargs: keywords list that belongs to `func`
            n_grad_func_args,
            k_grad_func_kwargs

        Methods:
            Energy: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['energy'].
            Grad: input Tensor `X` and PygData `graph`, it will update graph.pos into X and return model(graph)['forces'].

        """
        self._model = func
        self._ext_func = ext_func
        self._grad_func = grad_func
        self._ext_grad_func = ext_grad_func
        self.n_func_args = n_func_args
        self.k_func_kwargs = set(k_func_kwargs) if k_func_kwargs is not None else set()
        self.n_grad_func_args = n_grad_func_args
        self.k_grad_func_kwargs = set(k_grad_func_kwargs) if k_grad_func_kwargs is not None else set()

        self.forces = None

    @staticmethod
    def _auto_grad_func(X: th.Tensor, y: th.Tensor | None = None):
        grad_shape = th.ones_like(y)
        g = th.autograd.grad(y, X, grad_shape)
        return g[0]

    def Energy(self, X, *func_args, **func_kwargs, ):
        self.X = X
        true_func_args = func_args[:self.n_func_args]
        ext_func_args = func_args[self.n_func_args:]
        true_func_kwargs = {k:func_kwargs[k] for k in (func_kwargs.keys() & self.k_func_kwargs)}
        ext_func_kwargs = {k:func_kwargs[k] for k in (func_kwargs.keys() - self.k_func_kwargs)}

        self.y = self._model(self.X, *true_func_args, **true_func_kwargs) + self._ext_func(X, *ext_func_args, **ext_func_kwargs)

        return self.y

    def Grad(self, X, *grad_func_args, **grad_func_kwargs,):
        true_func_args = grad_func_args[:self.n_func_args]
        ext_func_args = grad_func_args[self.n_func_args:]
        true_func_kwargs = {k: grad_func_kwargs[k] for k in (grad_func_kwargs.keys() & self.k_func_kwargs)}
        ext_func_kwargs = {k: grad_func_kwargs[k] for k in (grad_func_kwargs.keys() - self.k_func_kwargs)}
        g = self._grad_func(X, *true_func_args, **true_func_kwargs) + self._ext_grad_func(X, *ext_func_args, **ext_func_kwargs)

        return g

    def Grad_y(self, X, y, *grad_func_args, **grad_func_kwargs, ):
        true_func_args = grad_func_args[:self.n_func_args]
        ext_func_args = grad_func_args[self.n_func_args:]
        true_func_kwargs = {k: grad_func_kwargs[k] for k in (grad_func_kwargs.keys() & self.k_func_kwargs)}
        ext_func_kwargs = {k: grad_func_kwargs[k] for k in (grad_func_kwargs.keys() - self.k_func_kwargs)}
        g = self._grad_func(X, y, *true_func_args, **true_func_kwargs) + self._ext_grad_func(X, y, *ext_func_args, **ext_func_kwargs)

        return g
