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
            max_step: maximum steps.
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
        self.raw_half_time_step_const = 0.5 * self.time_step
        # CUDAGraph handles for full-loop capture (lazy-init on first step)
        self._graph_A = None
        self._graph_B = None

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
            self._chi2_dist = th.distributions.chi2.Chi2(
                df=th.clamp(self.free_degree.float() - 1.0, min=1.0)
            )
            return self.__CSVR_cpu if self.device.type != 'cuda' else self.__CSVR_cuda
        else:
            raise NotImplementedError("Unknown Thermostat Type.")

    def _register_dump_vars(self):
        return ['Energy', 'X', 'V', 'Force']

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
        if masses is None:
            masses = th.ones_like(X)
        self.update_scheme = self.__resolve_update_scheme(batch_indices)
        # Pre-allocate random buffers for CSVR thermostat
        if self.thermostat == 'CSVR':
            _sample = X[0:1]
            _K_shape = (len(batch_indices),) if batch_indices is not None else (1,)
            self._buf_R = th.empty(_K_shape, device=self.device, dtype=th.float32)
            self._buf_S = th.empty(_K_shape, device=self.device, dtype=th.float32)
        else:
            self._buf_R = None
            self._buf_S = None
        # Graph/buffer will be built on first __CSVR call with real tensors
        self._buf_F = None
        self._graph_A = None
        self._graph_B = None

    def _build_csver_graphs(self, s, masses, atom_masks, batch_indices):
        """Build CUDAGraphs A and B for CSVR thermostat using real simulation tensors."""
        if self.device.type != 'cuda':
            return
        self._buf_F = th.empty_like(s.X)
        half_dt = 0.5 * self.time_step * 9.64853329045427e-3
        # --- Graph A: V += 0.5*dt*F/m, X += dt*V ---
        self._graph_A = th.cuda.CUDAGraph()
        with th.cuda.graph(self._graph_A):
            s.V.addcdiv_(self._buf_F, masses, value=half_dt)
            s.X.add_(s.V, alpha=self.time_step)
        # --- Graph B: F*mask, V += 0.5*dt*F/m, Ek_T, CSVR thermostat ---
        Nf = self.free_degree
        self._graph_B = th.cuda.CUDAGraph()
        with th.cuda.graph(self._graph_B):
            self._buf_F.mul_(atom_masks)
            s.V.addcdiv_(self._buf_F, masses, value=half_dt)
            self.Ek, _ = self._reduce_Ek_T(batch_indices, masses, s.V)
            if batch_indices is not None:
                K = th.clamp(self.Ek, min=self.epsK)
                f = self.one_sub_c * self.EK_TARGET / (Nf * K)
                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2 = th.addcmul(self.sqrt_c, sqrt_f, self._buf_R) ** 2
                alpha2.addcmul_(f, self._buf_S).clamp_min_(self.epsK)
                alpha = th.sqrt(alpha2).reshape(1, -1, 1)
                s.V *= alpha.index_select(1, self.batch_scatter)
            else:
                K = th.clamp(self.Ek, min=self.epsK)
                f = self.one_sub_c * self.EK_TARGET / (Nf * K)
                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2 = th.addcmul(self.sqrt_c, sqrt_f, self._buf_R) ** 2
                alpha2.addcmul_(f, self._buf_S).clamp_min_(self.epsK)
                alpha = th.sqrt(alpha2).reshape(-1, 1, 1)
                s.V *= alpha

    def __Langevin(
            self,
            s,
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
    ) -> None:
        """
        BAOAB style Langevin thermostat.
        References: J. Chem. Phys. 138, 174102 (2013)

        """
        s.X = s.X.detach()
        half_time_step_const = self.half_time_step_const
        raw_half_time_step_const = self.raw_half_time_step_const
        with th.no_grad():
            alpha = self.alpha
            # half-step
            s.V.addcdiv_(s.Force, masses, value=half_time_step_const)
            s.X.add_(s.V, alpha=raw_half_time_step_const)
            # stochastic update velocity
            s.V.mul_(alpha)
            s.V.add_(th.sqrt((8314.462618 * self.T_init * (1 - alpha ** 2)) / masses) * 1e-5 * th.randn_like(s.V))
            s.X.add_(s.V, alpha=raw_half_time_step_const)
            # update energy & forces
            Energy, Forces = self._calc_EF(
                s.X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y
            )
            Forces.mul_(atom_masks)
            # the rest half-step
            s.V.addcdiv_(Forces, masses, value=half_time_step_const)
            s.Energy = Energy
            s.Force = Forces

    def __VR(
            self,
            s,
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
    ) -> None:
        # NVE Step
        s.X = s.X.detach()
        with th.no_grad():
            s.V.addcdiv_(s.Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            s.X.add_(s.V, alpha=self.time_step)
            Energy, Forces = self._calc_EF(
                s.X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y
            )
            Forces.mul_(atom_masks)

            s.V.addcdiv_(Forces, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            if batch_indices is not None:
                # Rescaling factor
                alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)
                s.V *= alpha.transpose(0, 1)[:, self.batch_scatter, :]
            else:
                # Rescaling factor
                alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)
                s.V *= alpha
            s.Energy = Energy
            s.Force = Forces

    def __CSVR_cpu(
            self,
            s,
            func, grad_func_,
            func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices
    ) -> None:
        """ Eager CSVR update (CPU path). """
        half_dt = self.half_time_step_const
        time_step = self.time_step
        with th.no_grad():
            s.V.addcdiv_(s.Force, masses, value=half_dt)
            s.X.add_(s.V, alpha=time_step)
            Energy, Forces = self._calc_EF(
                s.X, func, func_args, func_kwargs,
                grad_func_, grad_func_args, grad_func_kwargs,
                self.require_grad, is_grad_func_contain_y
            )
            Forces.mul_(atom_masks)
            s.V.addcdiv_(Forces, masses, value=half_dt)
            if self.Ek_T_graph is not None:
                self.Ek_T_graph.replay()
            else:
                self.Ek, _ = self._reduce_Ek_T(batch_indices, masses, s.V)
            Nf = self.free_degree
            if batch_indices is not None:
                K = th.clamp(self.Ek, min=self.epsK)
                f = self.one_sub_c * self.EK_TARGET / (Nf * K)
                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                R = th.randn_like(K)
                S = self._chi2_dist.sample()
                alpha2 = th.addcmul(self.sqrt_c, sqrt_f, R) ** 2
                alpha2.addcmul_(f, S).clamp_min_(self.epsK)
                alpha = th.sqrt(alpha2).reshape(1, -1, 1)
                s.V *= alpha.index_select(1, self.batch_scatter)
            else:
                K = th.clamp(self.Ek, min=self.epsK)
                f = self.one_sub_c * self.EK_TARGET / (Nf * K)
                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                R = th.randn_like(K)
                S = self._chi2_dist.sample()
                alpha2 = th.addcmul(self.sqrt_c, sqrt_f, R) ** 2
                alpha2.addcmul_(f, S).clamp_min_(self.epsK)
                alpha = th.sqrt(alpha2).reshape(-1, 1, 1)
                s.V *= alpha
            s.Energy = Energy
            s.Force = Forces

    def __CSVR_cuda(
            self,
            s,
            func, grad_func_,
            func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices
    ) -> None:
        """ First CSVR call on CUDA: build CUDAGraphs, then delegate to graph version. """
        self._buf_F = th.empty_like(s.X)
        self._build_csver_graphs(s, masses, atom_masks, batch_indices)
        self.update_scheme = self.__CSVR_cuda_replay  # self replacement, the rest steps use replay.
        self.__CSVR_cuda_replay(
            s, func, grad_func_,
            func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices
        )

    def __CSVR_cuda_replay(
            self,
            s,
            func, grad_func_,
            func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices
    ) -> None:
        """ CUDAGraph replay for CSVR. """
        with th.no_grad():
            self._buf_F.copy_(s.Force)
            self._graph_A.replay()
        Energy, Forces = self._calc_EF(
            s.X, func, func_args, func_kwargs,
            grad_func_, grad_func_args, grad_func_kwargs,
            self.require_grad, is_grad_func_contain_y
        )
        Nf = self.free_degree
        if batch_indices is not None:
            self._buf_R.copy_(th.randn_like(self._buf_R))
            self._buf_S.copy_(self._chi2_dist.sample())
        else:
            self._buf_R.copy_(th.randn_like(self._buf_R))
            self._buf_S.copy_(self._chi2_dist.sample())
        with th.no_grad():
            self._buf_F.copy_(Forces)
            self._graph_B.replay()
        s.Energy = Energy
        s.Force = Forces

    def __NoseHoover(
            self,
            s,
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
    ) -> None:
        half_time_step_const = self.half_time_step_const
        raw_half_time_step_const = self.raw_half_time_step_const
        time_step = self.time_step
        smass = self.smass
        # Main update
        with th.no_grad():
            if batch_indices is not None:
                _iota = self.p_iota[:, self.batch_scatter, :]
            else:
                _iota = self.p_iota
            s.V.addcdiv_(s.Force, masses, value=half_time_step_const)
            s.V.mul_(th.exp(- _iota * raw_half_time_step_const))
            s.X.add_(s.V, alpha=time_step)

            Energy, Forces = self._calc_EF(
                s.X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y
            )
            Forces.mul_(atom_masks)

            if batch_indices is not None:  # for cuda, it would be further optimized by Graph.replay
                reduced_Ek = th.sum(
                    index_reduce(masses * s.V ** 2 * 103.642696562621738, self.batch_scatter, 1, out_size=self.scatter_dim_out_size),
                    dim=-1,
                    keepdim=True
                )
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=raw_half_time_step_const
                )
                _iota = self.p_iota[:, self.batch_scatter, :]  # (1, n_batch*n_atom, 1)
                s.V.addcdiv_(Forces, masses, value=half_time_step_const)
                s.V.mul_(th.exp(- _iota * raw_half_time_step_const))
                reduced_Ek = th.sum(
                    index_reduce(masses * s.V ** 2 * 103.642696562621738, self.batch_scatter, 1, out_size=self.scatter_dim_out_size),
                    dim=-1,
                    keepdim=True
                )
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=raw_half_time_step_const
                )
            else:
                reduced_Ek = th.sum(masses * s.V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=raw_half_time_step_const
                )
                s.V.addcdiv_(Forces, masses, value=half_time_step_const)
                s.V.mul_(th.exp(- _iota * raw_half_time_step_const))
                reduced_Ek = th.sum(masses * s.V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                # self.p_iota = p_iota + 0.5 * dt * (reducedEk - Nf * T)/smass
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=raw_half_time_step_const
                )
            s.Energy = Energy
            s.Force = Forces

    def _updateXV(
            self,
            s,
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
    ) -> None:

        self.update_scheme(
            s,
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
