""" Canonical ensemble (NVT) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVT.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import warnings
import math

import numpy as np
import torch as th
from torch import nn

from BUCToolkit.utils.index_ops import index_reduce
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
        is_compile: whether to use jit to compile integrator or not.
        compile_kwargs: keyword arguments passed to compile. Only work when is_compile is True.
    Methods:
        run: run the NVT ensemble BatchMD.
    """

    def __init__(
            self,
            time_step: float,
            max_step: int,
            thermostat: Literal['Langevin', 'VR', 'Nose-Hoover', 'CSVR'],
            thermostat_config: Dict[Literal['damping_coeff', 'time_const', 'virt_mass'], float] | None = None,
            T_init: float = 298.15,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            is_compile: bool = False,
            compile_kwargs: dict | None = None,
    ) -> None:
        """
        Parameters:
            time_step: float, time per step (ps).
            max_step: maxmimum steps.
            thermostat: str, the thermostat of NVT ensemble.
            thermostat_config: Dict|None, configs of thermostat. {'damping_coeff': float} for Langevin, {'time_const': float} for CSVR, {'virt_mass': float} for Nose-Hoover.
            T_init: initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
            output_file: the path to the binary file that stores trajectories. If None, tractories will not output.
            output_structures_per_step: int, output structures per output_structures_per_step steps.
            device: device that program run on.
            verbose: control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.
        """
        super().__init__(
            time_step,
            max_step,
            T_init,
            output_file,
            output_structures_per_step,
            device,
            verbose,
            is_compile,
            compile_kwargs
        )
        __ENSEMBLES_DICT = {'Langevin': None, 'VR': None, 'Nose-Hoover': None, 'CSVR': None}
        if thermostat not in {'Langevin', 'Langevin_old', 'test', 'VR', 'CSVR', 'Nose-Hoover'}:
            raise ValueError(f'Unknown Thermostat {thermostat}')
        self.thermostat = thermostat
        if thermostat_config is None:
            thermostat_config = dict()
        self.thermostat_config = thermostat_config
        self.update_scheme = None  # lazy loaded in self.initialize
        self.half_time_step_const = 0.5 * self.time_step * 9.64853329045427e-3

    def __resolve_update_scheme(self, batch_indices):
        """
        resolve different iteration scheme & initialize corresponding parameters.
        Returns:

        """
        if self.thermostat == "Langevin":
            damp_coeff = self.thermostat_config.get('damping_coeff', 0.01)  # Unit: fs^-1
            self.alpha = math.exp(- damp_coeff * self.time_step)
            return self.__Langevin
        elif self.thermostat == "VR":
            return self.__VR
        elif self.thermostat == "Nose-Hoover":
            # read thermostat config
            smass = self.thermostat_config.get('virt_mass', self.free_degree * 8.617333262145e-5 * self.T_init * (40. * self.time_step) ** 2)
            if isinstance(smass, float):
                smass = th.as_tensor(smass, device=self.device).view(1, 1, 1)
            elif isinstance(smass, th.Tensor):
                smass = smass.to(self.device)
                smass = smass.view(1, -1, 1) if self.batch_tensor is not None else smass.view(-1, 1, 1)
            self.smass = smass
            if batch_indices is not None:
                self.long_free_degree = self.free_degree.reshape(1, -1, 1)
            else:
                self.long_free_degree = self.free_degree.reshape(-1, 1, 1)
            return self.__NoseHoover
        elif self.thermostat == "CSVR":
            # read thermostat configs
            self.time_const = self.thermostat_config.get('time_const', 10 * self.time_step)  # Unit: fs^-1
            # NVE Step
            dtT = self.time_step
            tauT = self.time_const
            # c = exp(-dt/tau)  (scalar)
            c = th.exp(th.scalar_tensor(-dtT / tauT, device=self.device, dtype=th.float32))
            self.sqrt_c = th.sqrt(c)
            self.one_sub_c = 1 - c
            # avoid divide-by-zero in K
            self.epsK = th.scalar_tensor(1e-12, device=self.device, dtype=th.float32)
            return self.__CSVR
        else:
            raise NotImplementedError("Unknown Thermostat Type.")

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
    ) -> None:
        self.update_scheme = self.__resolve_update_scheme(batch_indices)

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
            batch_indices
    )-> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # read thermostat configs
        X = X.detach()
        with th.no_grad():
            alpha = self.alpha
            # half-step
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha = 0.5 * self.time_step)
            # stochastic update velocity
            V.mul_(alpha)
            V.add_(th.sqrt((8314.462618 * self.T_init * (1 - alpha ** 2)) / masses) * 1e-5 * th.randn_like(V))
            #V = alpha * V + th.sqrt((8314.462618 * self.T_init * (1 - alpha ** 2)) / masses) * 1e-5 * th.randn_like(V)
            X.add_(V, alpha = 0.5 * self.time_step)
            # update energy & forces
            Energy, Force = self._calc_EF(
                X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y
            )
            Force.mul_(atom_masks)
            # the rest half-step
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)


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
            batch_indices
    )-> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # NVE Step
        X = X.detach()
        with th.no_grad():
            # X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha=self.time_step)
            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
            # Update V
            Energy, Force = self._calc_EF(
                X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y
            )
            Force.mul_(atom_masks)

            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
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
        """
        The Analytic solution of CSVR that uses Chi^2 distribution and exp.
        Returns:

        """

        with th.no_grad():
            # X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha=self.time_step)
            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
            # Update V
            Energy, Force = self._calc_EF(
                X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y
            )
            Force.mul_(atom_masks)

            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            Nf = self.free_degree  # shape (n_batch,)
            if batch_indices is not None:
                K = th.clamp(self.Ek, min=self.epsK)  # shape (n_batch,)
                K0 = self.EK_TARGET  # (n_batch, )

                # f = (1-c) * K0 / (Nf*K)
                f = self.one_sub_c * K0 / (Nf * K)

                # R ~ N(0,1)
                R = th.randn_like(K)

                # S ~ Chi2(df=Nf-1)
                df = th.clamp(Nf - 1.0, min=1.0)
                S = th.distributions.chi2.Chi2(df=df).sample()  # shape (n_batch,)

                # alpha^2 = (sqrt(c)+sqrt(f)R)^2 + f S
                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2: th.Tensor = th.addcmul(self.sqrt_c, sqrt_f, R)**2
                alpha2.addcmul_(f, S).clamp_min_(self.epsK)

                # (optional bookkeeping: post-thermostat kinetic energy)
                #self.Ekt_vir = alpha2 * K

                alpha = th.sqrt(alpha2).reshape(1, -1, 1)  # (n_batch, 1, 1)
                V *= alpha.index_select(1, self.batch_scatter)

            else:
                K = th.clamp(self.Ek, min=self.epsK)  # (n_batch,)
                K0 = self.EK_TARGET  # scalar or (n_batch,)

                f = self.one_sub_c * K0 / (Nf * K)

                R = th.randn_like(K)
                df = th.clamp(Nf - 1.0, min=1.0)
                # sample per-batch
                S = th.distributions.chi2.Chi2(df=df).sample()

                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2: th.Tensor = th.addcmul(self.sqrt_c, sqrt_f, R) ** 2
                alpha2.addcmul_(f, S).clamp_min_(self.epsK)

                #self.Ekt_vir = alpha2 * K

                alpha = th.sqrt(alpha2).reshape(-1, 1, 1)  # (n_batch, 1, 1)
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
        # Main update
        with th.no_grad():
            if batch_indices is not None:
                _iota = self.p_iota[:, self.batch_scatter, :]
            else:
                _iota = self.p_iota
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            V.mul_(th.exp(- _iota * 0.5 * self.time_step))
            X.add_(V, alpha=self.time_step)

            Energy, Force = self._calc_EF(
                X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y
            )
            Force.mul_(atom_masks)

            if batch_indices is not None:  # for cuda, it would be further optimized by Graph.replay
                reduced_Ek = th.sum(
                    index_reduce(masses * V ** 2 * 103.642696562621738, self.batch_scatter, 1, out_size=self.scatter_dim_out_size),
                    dim=-1,
                    keepdim=True
                )
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )
                _iota = self.p_iota[:, self.batch_scatter, :]  # (1, n_batch*n_atom, 1)
                V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
                V.mul_(th.exp(- _iota * 0.5 * self.time_step))
                reduced_Ek = th.sum(
                    index_reduce(masses * V ** 2 * 103.642696562621738, self.batch_scatter, 1, out_size=self.scatter_dim_out_size),
                    dim=-1,
                    keepdim=True
                )
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )
            else:
                reduced_Ek = th.sum(masses * V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )
                V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
                V.mul_(th.exp(- _iota * 0.5 * self.time_step))
                reduced_Ek = th.sum(masses * V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=self.time_step * 0.5
                )

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
            batch_indices
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
            batch_indices
        )

        return x, v, energy, forces
