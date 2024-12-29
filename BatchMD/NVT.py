""" Canonical ensemble (NVT) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024.12.10, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.7b
#  File: NVT.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import torch as th

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
        device: device that program rum on.
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
        if thermostat not in {'Langevin', 'test', 'VR', 'CSVR', 'Nose-Hoover'}: raise ValueError(f'Unknown Thermostat {thermostat}')
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
        if self.thermostat == 'Langevin':
            # read thermostat configs
            damp_coeff = self.thermostat_config.get('damping_coeff', 0.01)  # Unit: fs^-1
            sigma = th.sqrt((2. * masses * damp_coeff * self.T_init / self.time_step) * (0.138064853 / (6.022140857 * 1.602176634**2)))  # Unit: eV/Ang
            X = X.detach()
            # Update X
            with th.no_grad():
                Force = (Force
                         - damp_coeff * masses * 1.e3 / (1.602176634 * 6.022140857) * V  # damping term (eV/Ang)
                         + th.normal(0., sigma))                          # stochastic term (ev/Ang)
                X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step**2 * 9.64853329045427e-3
                V = V + (Force / (1. * masses)) * self.time_step * 9.64853329045427e-3
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

        elif self.thermostat == 'test':
            pass

        elif self.thermostat == 'VR':
            # NVE Step
            X = X.detach()
            with th.no_grad():
                X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
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
            with th.no_grad():
                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
            # Rescaling factor
            alpha = th.sqrt(self.EK_TARGET/self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
            # Rescale Velocities
            V = V * alpha  # (n_batch, n_atom, n_dim) * (n_batch, 1, 1)

            return X, V, Energy, Force

        elif self.thermostat == 'CSVR':
            # read thermostat configs
            time_const = self.thermostat_config.get('time_const', 10*self.time_step)  # Unit: fs^-1
            # NVE Step
            X = X.detach()
            with th.no_grad():
                X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
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
            with th.no_grad():
                V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3

            # Kinetic Energy Redistribution
            # sigma = th.sqrt((2. * masses * self.T_init / (self.time_step * time_const)) * (0.138064853 / (6.022140857 * 1.602176634 ** 2)))
            sigma = self.time_step**0.5
            self.Ekt_vir += ((self.EK_TARGET - self.Ek) * self.time_step / time_const
                             + 2. * th.sqrt(self.EK_TARGET * self.Ek / ((3 * n_atom - 3) * time_const)) * th.normal(0., sigma, size=(n_batch,), device=self.device))  # Unit: eV/Ang

            # Rescaling factor
            alpha = th.sqrt(self.Ekt_vir / self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
            # Rescale Velocities
            V = V * alpha  # (n_batch, n_atom, n_dim) * (n_batch, 1, 1)

            return X, V, Energy, Force

        elif self.thermostat == 'Nose-Hoover':
            # TODO, Nose-Hoover Thermostat.
            raise NotImplementedError

        else:
            raise RuntimeError('Unknown Thermostat.')
        # TODO
