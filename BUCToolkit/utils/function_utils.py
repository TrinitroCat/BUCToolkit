"""
utils used for functions and models, including wrapper, preloader, etc.
"""
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: function_utils.py
#  Environment: Python 3.12

from typing import Tuple, Dict, List, Any, Mapping, Sequence, Callable
from abc import ABC, abstractmethod

import torch as th
import torch.nn as nn

__all__ = ['MeanModel', 'preload_func']

class MeanModel(nn.Module):
    """
    calculate the mean value of multiple models as new output.

    Args:
        models: sequence of models
        hyperparams: hyperparameters of models. if Dict, all models would share the same hyperparameters.
    """
    def __init__(self, models: Tuple[Any, ...] | List[Any], hyperparams: Tuple[Dict, ...] | Dict, model_output_key: Tuple[str, ...] | None = None):
        super().__init__()
        if isinstance(hyperparams, Sequence):
            if len(models) != len(hyperparams):
                raise RuntimeError(f'Number of hyperparameters {len(hyperparams)} and models {len(models)} does not match.')
            self.models = nn.ModuleList([(models[_])(**hparams) for _, hparams in enumerate(hyperparams)])
        elif isinstance(hyperparams, Dict):
            self.models = nn.ModuleList([_model(**hyperparams) for _model in models])
        self.output_key = model_output_key

    def load_state_tuple(self, state_dict: Tuple[Mapping[str, Any]] | Mapping[str, Any], strict: bool = True, assign: bool = False):
        """

        Args:
            state_dict:
            strict:
            assign:

        Returns:

        """
        if not isinstance(state_dict, Mapping):
            _state_dict = dict()
            for i, stdict in enumerate(state_dict):
                _state_dict.update({f'models.{i}.{key}': val for key, val in stdict.items()})
        else:
            _state_dict = state_dict
        self.load_state_dict(_state_dict, strict, assign)

    def forward(self, X: Any):
        if self.output_key is None:
            y = 0.
            if self.training:
                pred_list = list()
                for mod in self.models:
                    res = mod(X)
                    pred_list.append(res)
                return pred_list
            else:
                for mod in self.models:
                    y = y + mod(X)
        else:
            y = {key: 0. for key in self.output_key}
            if self.training:
                pred_list = list()
                for mod in self.models:
                    res = mod(X)
                    pred_list.append(res)
                return pred_list
            else:
                l = len(self.models)
                for mod in self.models:
                    y = {key: y[key] + mod(X)[key]/l for key in self.output_key}

        return y


class _BaseWrapper(ABC):

    def __init__(self, model):
        self._model = model
        self.forces = None
        self.X = None

    @abstractmethod
    def Energy(self, *args, **kwargs) -> th.Tensor:
        pass

    @abstractmethod
    def Grad(self, *args, **kwargs) -> th.Tensor:
        pass

    def to(self, *args, **kwargs):
        """
        wrap the `to` method of wrapped model. If the wrapped model has not `to` method, simply passed.
        Args:
            *args:
            **kwargs:

        Returns: the obj that the wrapped model would return

        """
        if hasattr(self._model, 'to') and callable(self._model.to):
            self._model = self._model.to(*args, **kwargs)

        return self

    def eval(self):
        """
        wrap the `eval` method of wrapped model. If the wrapped model has not `eval` method, simply passed.
        Returns: None

        """
        if hasattr(self._model, 'eval') and callable(self._model.eval):
            self._model.eval()


def preload_func(func: Callable, device: Any) -> Callable:
    """
    Apply the `func.to(device)` and `func.eval()` methods to the input function if it has such method.
    if `func` is just a subclass of torch.nn.Module, `func.zero_grad()` will be called, too.
    Args:
        func:
        device:

    Returns: callable, the function self.

    """
    if isinstance(func, nn.Module):
        func = func.to(device)
        func.eval()
        func.zero_grad()
    else:
        if hasattr(func, 'to'):
            _ = func.to(device)
            func = _ if _ is not None else func
        if hasattr(func, 'eval'):
            func.eval()
        if hasattr(func, 'zero_grad'):
            func.zero_grad()

    return func

def compare_tensors(X1: th.Tensor, X2: th.Tensor):
    """Compare two tensors. Return True if they are the same, False otherwise."""
    char1 = (X1.untyped_storage().data_ptr(),
             X1.storage_offset(),
             tuple(X1.shape),
             tuple(X1.stride()),
             X1.device,
             X1.dtype)
    char2 = (X2.untyped_storage().data_ptr(),
             X2.storage_offset(),
             tuple(X2.shape),
             tuple(X2.stride()),
             X2.device,
             X2.dtype)

    return char1 == char2
