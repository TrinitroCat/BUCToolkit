"""
Model wrapper of ensemble method
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: ModelWrapper.py
#  Environment: Python 3.12

from typing import Tuple, Dict, List, Any, Mapping, Sequence

import torch as th
import torch.nn as nn

__all__ = ['MeanModel', 'ResidualsBoostModel']

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


class ResidualsBoostModel(nn.Module):
    """
    sequentially fit the prediction residuals.
    Args:
        models: sequence of models.
        hyperparams: hyperparameters of models. if Dict, all models would share the same hyperparameters.
        model_output_key: keys of model output. if None, model output would be a tensor, else a Dict with key in `model_output_key`.
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

    def load_state_dict(self, state_dict: Tuple[Mapping[str, Any]], strict: bool = True, assign: bool = False):
        r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict` unless
            :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When ``False``, the properties of the tensors
                in the current module are preserved while when ``True``, the
                properties of the Tensors in the state dict are preserved. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
                for which the value from the module is preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            _state_dict = dict()
            for i, stdict in enumerate(state_dict):
                _state_dict.update({f'models.{i}.{key}': val for key, val in stdict.items()})
        else:
            _state_dict = state_dict
        super().load_state_dict(_state_dict, strict, assign)

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
                for mod in self.models:
                    y = {key: y[key] + mod(X)[key] for key in self.output_key}

        return y
