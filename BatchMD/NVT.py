""" Canonical ensemble (NVT) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024.12.10, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.7b
#  File: NVT.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import warnings

import torch as th

from BM4Ckit.utils.scatter_reduce import scatter_reduce
from ._BaseMD import _BaseMD


class NVT(_BaseMD):
    """
    Canonical ensemble (NVT) molecular dynamics.

    Parameters:
        time_step: float, time per step (ps).
        max_step: maxmimum steps.
        thermostat: str, the thermostat of NVT ensemble.
        thermostat_config: Dict|None, configs of thermostat. {'damping_coeff': float} for Langevin, {'time_const': float} for CSVR, {'virt_mass': float} for Nose-Hoover.
        T_init: initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: device that the program rum on.
        verbose: control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.

    Methods:
        run: run the NVT ensemble BatchMD.
    """

    def __init__(self,
                 time_step: float,
                 max_step: int,
                 thermostat: Literal['Langevin', 'VR', 'Nose-Hoover', 'CSVR'],
                 thermostat_config: Dict | None = None,
                 T_init: float = 298.15,
                 output_file: str | None = None,
                 output_structures_per_step: int = 1,
                 device: str | th.device = 'cpu',
                 verbose: int = 2) -> None:
        """
        Parameters:
            time_step: float, time per step (ps).
            max_step: maxmimum steps.
            thermostat: str, the thermostat of NVT ensemble.
            thermostat_config: Dict|None, configs of thermostat. {'damping_coeff': float} for Langevin, {'time_const': float} for CSVR, {'virt_mass': float} for Nose-Hoover.
            T_init: initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
            output_structures_per_step: int, output structures per output_structures_per_step steps.
            device: device that program rum on.
            verbose: control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.
        """
        super().__init__(
            time_step,
            max_step,
            T_init,
            output_file,
            output_structures_per_step,
            device,
            verbose
        )
        __ENSEMBLES_DICT = {'Langevin': None, 'VR': None, 'Nose-Hoover': None, 'CSVR': None}
        if thermostat not in {'Langevin', 'Langevin_old', 'test', 'VR', 'CSVR', 'Nose-Hoover'}: raise ValueError(f'Unknown Thermostat {thermostat}')
        self.thermostat = thermostat
        if thermostat_config is None:
            thermostat_config = dict()
        self.thermostat_config = thermostat_config

    def _updateXV(
            self, X, V, Force,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices
    ) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor):

        n_batch, n_atom, n_dim = X.shape
        if self.thermostat == 'Langevin_old':
            warnings.warn(
                'WARNING: Langevin_old was deprecated because it use direct Euler method'
                ' thus leading to overestimation of temperature and inefficiency.',
                DeprecationWarning
            )
            # read thermostat configs
            X = X.detach()
            with th.no_grad():
                damp_coeff = self.thermostat_config.get('damping_coeff', 0.01)  # Unit: fs^-1
                sigma = th.sqrt((2. * masses * damp_coeff * self.T_init / self.time_step) * (0.138064853 / (6.022140857 * 1.602176634**2)))  # eV/Ang
                # Update X
                Force = (Force
                         - damp_coeff * masses * 1.e3 / (1.602176634 * 6.022140857) * V  # damping term (eV/Ang)
                         + th.normal(0., sigma))                          # stochastic term (ev/Ang)
                X = X + V * self.time_step #+ (Force / (2. * masses)) * self.time_step**2 * 9.64853329045427e-3
                V = V + (Force / (1. * masses)) * self.time_step * 9.64853329045427e-3
                with th.enable_grad():
                    # Update V
                    X.requires_grad_()
                    _Energy = func(X, *func_args, **func_kwargs)
                    if batch_indices is not None:
                        Energy = th.sum(_Energy, ).unsqueeze(0)
                    else:
                        Energy = _Energy
                    if is_grad_func_contain_y:
                        Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                    else:
                        Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks
                # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.

            return X, V, Energy, Force

        elif self.thermostat == 'Langevin':  # BAOAB algo.
            # read thermostat configs
            X = X.detach()
            with th.no_grad():
                damp_coeff = self.thermostat_config.get('damping_coeff', 0.01)  # Unit: fs^-1
                alpha = th.e ** (- damp_coeff * self.time_step)
                # half-step
                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
                X = X + 0.5 * V * self.time_step
                # stochastic update velocity
                V = alpha * V + th.sqrt((8314.462618 * self.T_init * (1 - alpha**2))/masses) * 1e-5 * th.randn_like(V)
                # the rest half-step
                X = X + 0.5 * V * self.time_step
                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
                # update energy & forces
                with th.enable_grad():
                    # Update V
                    X.requires_grad_()
                    _Energy = func(X, *func_args, **func_kwargs)
                    if batch_indices is not None:
                        Energy = th.sum(_Energy, ).unsqueeze(0)
                    else:
                        Energy = _Energy
                    if is_grad_func_contain_y:
                        Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                    else:
                        Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

            return X, V, Energy, Force

        elif self.thermostat == 'test':
            pass

        elif self.thermostat == 'VR':
            # NVE Step
            X = X.detach()
            with th.no_grad():
                X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
                with th.enable_grad():
                    # Update V
                    X.requires_grad_()
                    _Energy = func(X, *func_args, **func_kwargs)
                    if batch_indices is not None:
                        Energy = th.sum(_Energy, ).unsqueeze(0)
                    else:
                        Energy = _Energy
                    if is_grad_func_contain_y:
                        Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                    else:
                        Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
                if batch_indices is not None:
                    # Rescaling factor
                    alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1) | (irregular n_batch, 1, 1)
                    V *= alpha.transpose(0, 1)[:, self.batch_scatter, :]
                else:
                    # Rescaling factor
                    alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1) | (irregular n_batch, 1, 1)
                    V *= alpha  # (n_batch, n_atom, n_dim) * (n_batch, 1, 1)

            return X, V, Energy, Force

        elif self.thermostat == 'CSVR':
            # read thermostat configs
            time_const = self.thermostat_config.get('time_const', 10*self.time_step)  # Unit: fs^-1
            # NVE Step
            X = X.detach()
            with th.no_grad():
                X += V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
                V += (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
                with th.enable_grad():
                    # Update V
                    X.requires_grad_()
                    _Energy = func(X, *func_args, **func_kwargs)
                    if batch_indices is not None:
                        Energy = th.sum(_Energy, ).unsqueeze(0)
                    else:
                        Energy = _Energy
                    if is_grad_func_contain_y:
                        Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                    else:
                        Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

                V += (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3

                # Kinetic Energy Redistribution, Rescale Velocities
                # sigma = th.sqrt((2. * masses * self.T_init / (self.time_step * time_const)) * (0.138064853 / (6.022140857 * 1.602176634 ** 2)))
                sigma = self.time_step ** 0.5
                if batch_indices is not None:
                    self.Ekt_vir += (
                            (self.EK_TARGET - self.Ek) * self.time_step / time_const
                            + 2. * th.sqrt(self.EK_TARGET * self.Ek / ((3 * self.batch_tensor - 3) * time_const))
                            * th.normal(0., sigma, size=(len(batch_indices),), device=self.device)
                    )  # Unit: eV/Ang
                    # Rescaling factor
                    alpha = th.sqrt(self.Ekt_vir / self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1) | (irregular n_batch, 1, 1)
                    V *= alpha.transpose(0, 1)[:, self.batch_scatter, :]
                else:
                    self.Ekt_vir += (
                            (self.EK_TARGET - self.Ek) * self.time_step / time_const
                            + 2. * th.sqrt(self.EK_TARGET * self.Ek / ((3 * n_atom - 3) * time_const))
                            * th.normal(0., sigma, size=(n_batch,), device=self.device)
                    )  # Unit: eV/Ang
                    # Rescaling factor
                    alpha = th.sqrt(self.Ekt_vir / self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1) | (irregular n_batch, 1, 1)
                    V *= alpha  # (n_batch, n_atom, n_dim) * (n_batch, 1, 1)

            return X, V, _Energy, Force

        elif self.thermostat == 'Nose-Hoover':
            # freedom degree
            n_free = n_dim * (self.batch_tensor - 1) if batch_indices is not None else n_dim * (n_atom - 1)
            # read thermostat config
            smass = self.thermostat_config.get('virt_mass', n_free * 8.617333262145e-5 * self.T_init * 40**2)
            if isinstance(smass, float):
                smass = th.tensor(smass, device=self.device).view(1, 1, 1)
                if batch_indices is not None:
                    smass = smass.expand((1, len(batch_indices), 1))
                else:
                    smass = smass.expand(n_batch, 1, n_dim)
            elif batch_indices is not None:
                smass = smass.unsqueeze(0).expand((1, len(batch_indices)))
            else:
                smass = smass.unsqueeze(-1).expand(n_batch, 1)
            # Main update
            with th.no_grad():
                if batch_indices is not None:
                    _iota = self.p_iota[:, self.batch_scatter, None]
                else:
                    _iota = self.p_iota
                V += (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3 - 0.5 * _iota * V * self.time_step
                X += V * self.time_step
                with th.enable_grad():
                    # Update V
                    X.requires_grad_()
                    _Energy = func(X, *func_args, **func_kwargs)
                    if batch_indices is not None:
                        Energy = th.sum(_Energy, ).unsqueeze(0)
                    else:
                        Energy = _Energy
                    if is_grad_func_contain_y:
                        Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                    else:
                        Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

                if batch_indices is not None:
                    self.p_iota += 0.5/smass * (
                            th.sum(scatter_reduce(masses * V**2 * 103.642696562621738, self.batch_scatter, 1), -1)
                            - n_free * 8.617333262145e-5 * self.T_init
                    ) * self.time_step  # (1, n_batch)
                    _iota = self.p_iota[:, self.batch_scatter, None]  # (1, n_batch*n_atom, 1)
                    V += (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3 - 0.5 * _iota * V * self.time_step
                    self.p_iota += 0.5 / smass * (
                            th.sum(scatter_reduce(masses * V**2 * 103.642696562621738, self.batch_scatter, 1), -1)
                            - n_free * 8.617333262145e-5 * self.T_init
                    ) * self.time_step  # (1, n_batch)
                else:
                    self.p_iota += 0.5/smass * (
                            th.sum(masses * V**2 * 103.642696562621738, dim=(-1, -2), keepdim=True) - n_free * 8.617333262145e-5 * self.T_init
                    ) * self.time_step  # (n_batch, 1, n_dim)
                    V += (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3 - 0.5 * self.p_iota * V * self.time_step
                    self.p_iota += 0.5 / smass * (
                            th.sum(masses * V**2 * 103.642696562621738, dim=(-1, -2), keepdim=True) - n_free * 8.617333262145e-5 * self.T_init
                    ) * self.time_step  # (n_batch, 1, n_dim)

            return X, V, _Energy, Force

        else:
            raise RuntimeError('Unknown Thermostat.')
        # TODO
