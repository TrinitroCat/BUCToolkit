#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: MetropolisMC.py
#  Environment: Python 3.12
import copy
import math

import torch as th
from ._BaseMC import _BaseMC
from typing import Callable, Literal


class MMC(_BaseMC):
    """

    """

    def __init__(
            self,
            iter_scheme: Literal['Gaussian', 'Cauchy', 'Uniform'] = 'Gaussian',
            maxiter: int = 100,
            temperature_init: float = 1000.,
            temperature_scheme: Literal['constant', 'linear', 'exponential', 'log', 'fast'] = 'constant',
            temperature_update_freq: int = 1,
            temperature_scheme_param: float | None = None,
            coordinate_update_param: float = 0.2,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        """

        Args:
            iter_scheme: Algorithm for stochastic X update.
                "Gaussian": perturbating X by Gaussian distribution.
                "Cauchy": perturbating X by Cauchy distribution.
                "Uniform": perturbating X by uniform distribution.
            maxiter: max iteration steps.
            temperature_init: initial temperature.
            temperature_scheme: scheme of temperature update from
            temperature_update_freq: update temperature per `temperature_update_freq` step.
            temperature_scheme_param
            coordinate_update_param
            device:
            verbose:
        """
        super().__init__(
            iter_scheme,
            maxiter,
            device,
            verbose
        )
        self.T_begin = float(temperature_init)
        self.T_now = copy.deepcopy(self.T_begin)
        self.temperature_update_freq = temperature_update_freq
        __T_alpha_default = {
            'constant': 0.,
            'linear': 1e-3,
            'exponential': 0.99,
            'log': 1.,
            'fast': 0.2
        }
        self.T_alpha = float(temperature_scheme_param) if temperature_scheme_param is not None else __T_alpha_default[temperature_scheme]
        __T_scheme_dict = {
            'constant': self.__t_constant,
            'linear': self.__t_linear,
            'exponential': self.__t_exponential,
            'log': self.__t_log,
            'fast': self.__t_fast
        }
        self.T_update_func = __T_scheme_dict[temperature_scheme]

        self.coordinate_update_param = coordinate_update_param
        __X_scheme_dict = {
            'Gaussian': self.__x_Gaussian,
            'Cauchy': self.__x_Cauchy,
            'Uniform': self.__x_Uniform,
        }
        self.X_update_func = __X_scheme_dict[iter_scheme]


    @staticmethod
    def __t_constant(i):
        pass

    def __t_linear(self, i):
        self.T_now += (self.T_alpha - self.T_begin)/self.maxiter

    def __t_exponential(self, i):
        self.T_now *= self.T_alpha

    def __t_log(self, i):
        self.T_now =  self.T_begin/(1. + th.log(1. + self.T_alpha * i))

    def __t_fast(self, i):
        self.T_now = self.T_begin/(1. + self.T_alpha * i)

    def __x_Gaussian(self, X: th.Tensor):
        T_tmp = max(10., self.T_now)
        _x = th.full_like(X, self.coordinate_update_param * math.sqrt(T_tmp/298.15))
        g = th.distributions.Normal(X, _x)
        X_new = g.sample()
        return X_new

    def __x_Cauchy(self, X: th.Tensor):
        T_tmp = max(10., self.T_now)
        _x = th.full_like(X, self.coordinate_update_param * math.sqrt(T_tmp/298.15))
        g = th.distributions.Cauchy(X, _x)
        X_new = g.sample()
        return X_new

    def __x_Uniform(self, X: th.Tensor):
        T_tmp = max(10., self.T_now)
        _x = th.full_like(X, self.coordinate_update_param * math.sqrt(T_tmp/298.15))
        g = th.distributions.Uniform(X - _x, X + _x)
        X_new = g.sample()
        return X_new

    def _update_X(self, func, func_args, func_kwargs, energies_old, X: th.Tensor):
        """
        Override this method to implement X update algorithm.
        Args:
            X: (n_batch, n_atom, 3), the independent vars X.
            func,
            func_args,
            func_kwargs,
            energies_old,

        Returns:
            p: th.Tensor, the new update direction of X.
        """
        _X = self.X_update_func(X) * self.atom_masks
        _energy = func(_X, *func_args, **func_kwargs)
        delta_E = _energy - energies_old  # (n_batch, )
        metropolis_mask = th.exp(- delta_E / (8.617333262145e-5 * self.T_now))  # (n_batch, ); Boltzmann constant kB = 8.617333262145e-5 eV/K
        metropolis_mask = th.where(
            metropolis_mask > 1.,
            1.,
            metropolis_mask
        )
        _x_mask = th.rand_like(_energy) < metropolis_mask
        energy_new = th.where(
            _x_mask,
            _energy,
            energies_old
        )
        delta_E = th.where(
            _x_mask,
            delta_E,
            0.
        )
        if self.batch_scatter is not None:
            X_new = th.where(
                _x_mask[self.batch_scatter],
                _X,
                X
            )
        else:
            X_new = th.where(
                _x_mask.unsqueeze(-1).unsqueeze(-1),
                _X,
                X
            )

        return energy_new, delta_E, X_new

    def _update_algo_param(self, i:int, displace: th.Tensor) -> None:
        """
        Override this method to update the parameters of X update algorithm i.e., self.iterform.
        Args:
            i: iteration step now.
            displace: (n_batch, n_atom*3, 1), the displacement of X at this step. displace = step-length * p

        Returns: None
        """
        # update T
        if i%self.temperature_update_freq == 0:
            self.T_update_func(i/self.temperature_update_freq)

        pass

    def initialize_algo_param(self):
        pass
