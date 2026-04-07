""" Canonical ensemble (NVT) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVT.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import warnings

import torch as th
from torch import nn
import numpy as np

from BUCToolkit.utils.index_ops import index_reduce
from ._ConstrBaseMD import _rConstrBase
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT


class ConstrNVT(_rConstrBase):
    """
    Canonical ensemble (NVT) molecular dynamics with constraints.

    Parameters:
        time_step: float, time per step (ps).
        max_step: maxmimum steps.
        thermostat: str, the thermostat of NVT ensemble.
        thermostat_config: Dict|None, configs of thermostat. {'damping_coeff': float} for Langevin, {'time_const': float} for CSVR, {'virt_mass': float} for Nose-Hoover.
        constr_func:
        constr_val:
        constr_threshold:
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
                 constr_func: Callable[[th.Tensor], th.Tensor] = None,
                 constr_val: Callable[[th.Tensor], th.Tensor | Tuple[th.Tensor]] | th.Tensor = None,
                 constr_threshold: float = 1e-5,
                 T_init: float = 298.15,
                 output_file: str | None = None,
                 dump_path: str | None = None,
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
            device: device that program run on.
            verbose: control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.
        """
        super().__init__(
            time_step,
            max_step,
            T_init,
            constr_func,
            constr_val,
            constr_threshold,
            output_file,
            dump_path,
            output_structures_per_step,
            device,
            verbose
        )
        __ENSEMBLES_DICT = {'Langevin': None, 'VR': None, 'Nose-Hoover': None, 'CSVR': None}
        if thermostat not in {'Langevin', 'Langevin_old', 'test', 'VR', 'CSVR', 'Nose-Hoover'}: raise ValueError(f'Unknown Thermostat {thermostat}')
        self.thermostat = thermostat
        if thermostat_config is None:
            thermostat_config = dict()
        self.update_scheme = self.__resolve_update_scheme()
        self.thermostat_config = thermostat_config

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
        super().initialize(
            func,
            X,
            Element_list,
            masses,
            V_init,
            grad_func,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            is_grad_func_contain_y=is_grad_func_contain_y,
            require_grad=require_grad,
            batch_indices=batch_indices,
            fixed_atom_tensor=fixed_atom_tensor,
            is_fix_mass_center=is_fix_mass_center
        )
        # recalculate E_vir
        self.Ekt_vir = 0.5 * th.sum(masses * V_init ** 2, dim=(-2, -1)) * 103.642696562621738

    def __resolve_update_scheme(self):
        """
        resolve different iteration scheme
        Returns:

        """
        if self.thermostat == "Langevin":
            damp_coeff = self.thermostat_config.get('damping_coeff', 0.01)  # Unit: fs^-1
            self.alpha = th.exp(- damp_coeff * self.time_step)
            return self.__Langevin
        elif self.thermostat == "VR":
            return self.__VR
        elif self.thermostat == "Nose-Hoover":
            # read thermostat config
            smass = self.thermostat_config.get('virt_mass', self.free_degree * 8.617333262145e-5 * self.T_init * 40 ** 2)
            if isinstance(smass, float):
                smass = th.as_tensor(smass, device=self.device).view(1, 1, 1)
            self.smass = smass
            return self.__NoseHoover
        elif self.thermostat == "CSVR":
            # read thermostat configs
            self.time_const = self.thermostat_config.get('time_const', 10 * self.time_step)  # Unit: fs^-1
            return self.__CSVR
        else:
            raise NotImplementedError("Unknown Thermostat Type.")

    def __Langevin(
            self,
            X,
            V,
            Force,
            func,
            grad_func_,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            masses,
            atom_masks,
            is_grad_func_contain_y,
    )-> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # read thermostat configs
        X = X.detach()
        with th.no_grad():
            alpha = self.alpha
            # half-step
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            self._project2(X)
            # stochastic update velocity
            V.mul_(alpha)
            V.add_(th.sqrt((8314.462618 * self.T_init * (1 - alpha ** 2)) / masses) * 1e-5 * th.randn_like(V))
            # V = alpha * V + th.sqrt((8314.462618 * self.T_init * (1 - alpha ** 2)) / masses) * 1e-5 * th.randn_like(V)
            # the rest half-step
            X.add_(V, alpha=0.5 * self.time_step)
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            # retrack & proj
            X.add_(V, alpha=0.5 * self.time_step)
            V.copy_(self._project1(V))
            # update energy & forces
            with th.set_grad_enabled(self.require_grad):
                X.requires_grad_(self.require_grad)
                Energy = func(X, *func_args, **func_kwargs)
                if is_grad_func_contain_y:
                    Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

        return X, V, Energy, Force

    def __VR(
            self,
            X,
            V,
            Force,
            func,
            grad_func_,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            masses,
            atom_masks,
            is_grad_func_contain_y,
            batch_indices,
    )-> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # NVE Step
        X = X.detach()
        with th.no_grad():
            # X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha=self.time_step)
            Fc = self._project2(X)
            if self.verbose > 0:
                self.logger.info(f'Constraint forces \\lambda: {np.array2string(Fc.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}')
            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
            # Update V
            with th.set_grad_enabled(self.require_grad):
                X.requires_grad_(self.require_grad)
                Energy = func(X, *func_args, **func_kwargs)
                if is_grad_func_contain_y:
                    Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            V.copy_(self._project1(V))
            if batch_indices is not None:
                # Rescaling factor
                alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1) | (irregular n_batch, 1, 1)
                V *= alpha.transpose(0, 1)[:, self.batch_scatter, :]
            else:
                # Rescaling factor
                alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1) | (irregular n_batch, 1, 1)
                V *= alpha  # (n_batch, n_atom, n_dim) * (n_batch, 1, 1)

        return X, V, Energy, Force

    def __CSVR(
            self,
            X: th.Tensor,
            V: th.Tensor,
            Force,
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
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        The Analytic solution of CSVR that uses Chi^2 distribution and exp.
        Returns:

        """
        n_batch, n_atom, n_dim = X.shape
        # read thermostat configs
        time_const = self.time_const
        # NVE Step
        dtT = self.time_step
        tauT = time_const

        # c = exp(-dt/tau)  (scalar)
        c = th.exp(th.as_tensor(-dtT / tauT, device=self.device, dtype=V.dtype))
        sqrt_c = th.sqrt(c)

        # avoid divide-by-zero in K
        epsK = th.as_tensor(1e-12, device=self.device, dtype=V.dtype)

        with th.no_grad():
            # X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha=self.time_step)
            Fc = self._project2(X)
            if self.verbose > 0:
                self.logger.info(f'Constraint forces \\lambda: {np.array2string(Fc.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}')
            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
            # Update V
            with th.set_grad_enabled(self.require_grad):
                X.requires_grad_(self.require_grad)
                Energy = func(X, *func_args, **func_kwargs)
                if is_grad_func_contain_y:
                    Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            V.copy_(self._project1(V))

            if batch_indices is not None:
                # Nf per (irregular) configuration; keep your convention (3*N-3)
                Nf = self.free_degree  # shape (n_batch,)
                K = th.clamp(self.Ek, min=epsK)  # shape (n_batch,)
                K0 = self.EK_TARGET  # scalar or shape-compatible

                # f = (1-c) * K0 / (Nf*K)
                f = (1.0 - c) * K0 / (Nf * K)

                # R ~ N(0,1)
                R = th.randn_like(K)

                # S ~ Chi2(df=Nf-1)
                df = th.clamp(Nf - 1.0, min=1.0)
                S = th.distributions.chi2.Chi2(df=df).sample()  # shape (n_batch,)

                # alpha^2 = (sqrt(c)+sqrt(f)R)^2 + f S
                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2 = (sqrt_c + sqrt_f * R) ** 2 + f * S
                alpha2 = th.clamp(alpha2, min=epsK)

                # (optional bookkeeping: post-thermostat kinetic energy)
                self.Ekt_vir = alpha2 * K

                alpha = th.sqrt(alpha2).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
                V *= alpha.transpose(0, 1)[:, self.batch_scatter, :]

            else:
                # Nf is same for all batches; keep your convention (3*N-3)
                Nf = th.as_tensor(3 * n_atom - 3, device=self.device, dtype=V.dtype)  # scalar tensor
                K = th.clamp(self.Ek, min=epsK)  # (n_batch,)
                K0 = self.EK_TARGET  # scalar or (n_batch,)

                f = (1.0 - c) * K0 / (Nf * K)

                R = th.randn_like(K)
                df = th.clamp(Nf - 1.0, min=1.0)
                # sample per-batch
                S = th.distributions.chi2.Chi2(df=df).sample(sample_shape=K.shape)

                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2 = (sqrt_c + sqrt_f * R) ** 2 + f * S
                alpha2 = th.clamp(alpha2, min=epsK)

                self.Ekt_vir = alpha2 * K

                alpha = th.sqrt(alpha2).unsqueeze(-1).unsqueeze(-1)  # (n_batch, 1, 1)
                V *= alpha

        return X, V, Energy, Force

    def __NoseHoover(
            self,
            X,
            V,
            Force,
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
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        n_batch, n_atom, n_dim = X.shape
        smass = self.smass
        if batch_indices is not None:
            smass = smass.unsqueeze(0).expand((1, len(batch_indices)))
        else:
            smass = smass.unsqueeze(-1).expand(n_batch, 1)
        # Main update
        with th.no_grad():
            if batch_indices is not None:
                _iota = self.p_iota[:, self.batch_scatter, :]
            else:
                _iota = self.p_iota
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            V.mul_(th.exp(- _iota * 0.5 * self.time_step))
            X.add_(V, alpha=self.time_step)
            Fc = self._project2(X)
            if self.verbose > 0:
                self.logger.info(f'Constraint forces \\lambda: {np.array2string(Fc.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}')

            with th.set_grad_enabled(self.require_grad):
                X.requires_grad_(self.require_grad)
                Energy = func(X, *func_args, **func_kwargs)
                if is_grad_func_contain_y:
                    Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks

            if batch_indices is not None:  # for cuda, it would be further optimized by Graph.replay
                reduced_Ek = th.sum(index_reduce(masses * V ** 2 * 103.642696562621738, self.batch_scatter, 1), dim=-1, keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )
                _iota = self.p_iota[:, self.batch_scatter, :]  # (1, n_batch*n_atom, 1)
                V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
                V.mul_(th.exp(- _iota * 0.5 * self.time_step))
                reduced_Ek = th.sum(index_reduce(masses * V ** 2 * 103.642696562621738, self.batch_scatter, 1), dim=-1, keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )
            else:
                reduced_Ek = th.sum(masses * V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )
                V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
                V.mul_(th.exp(- _iota * 0.5 * self.time_step))
                reduced_Ek = th.sum(masses * V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )
            V.copy_(self._project1(V))

        return X, V, Energy, Force


    def _updateXV(
            self,
            X,
            V,
            Force,
            func,
            grad_func_,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            masses,
            atom_masks,
            is_grad_func_contain_y,
            batch_indices,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        x, v, energy, forces = self.update_scheme(
            X,
            V,
            Force,
            func,
            grad_func_,
            func_args,
            func_kwargs,
            grad_func_args,
            grad_func_kwargs,
            masses,
            atom_masks,
            is_grad_func_contain_y,
        )

        return x, v, energy, forces
