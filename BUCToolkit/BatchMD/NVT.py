""" Canonical ensemble (NVT) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVT.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import warnings
import math
from itertools import accumulate

import numpy as np
import triton
import torch as th
from torch import nn

from BUCToolkit.utils.index_ops import index_reduce
from ._BaseMD import _BaseMD
from .fused_ops.utils import KernelLauncher
from .fused_ops.VelocityVerlet import velocity_verlet_1_kernel, velocity_verlet_2_kernel
from .fused_ops.thermostats import (
    csvr_alpha_kernel,
    langevin_middle_kernel,
    nose_half_step_1_kernel,
    nose_reduce_update_kernel,
    nose_throttle_reduce_update_kernel,
    scale_by_batch_kernel,
    vr_scale_kernel,
)


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
        verbose: print level. 0 is silent, 1 prints selected scalars only, and
            2 or greater also prints selected arrays.
        is_compile: whether to use jit to compile integrator or not.
        compile_kwargs: keyword arguments passed to compile. Only work when is_compile is True.
    Methods:
        run: run the NVT ensemble BatchMD.
    """

    __slots__ = (
        'inv_m', '_alpha', '_randn', '_sigma', '_p_iota', '_smass',
        '_free_degree', '_ek_target', '_n_batch', '_n_chunk', '_n_elem',
        '_is_irregular', '_batch_starts', '_sqrt_c', '_one_sub_c',
        '_epsK', '_target_kT', '_scalar_dtype',
        '_vv1', '_vv2', '_csvr_alpha', '_vr_scale', '_scale',
        '_langevin', '_nose_half', '_nose_reduce', '_nose_throttle',
    )

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
            is_compile: Literal['full', 'partial', 'no'] | bool = False,
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
            verbose: print level. 0 is silent, 1 prints selected scalars only,
                and 2 or greater also prints selected arrays.
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

    def __resolve_update_scheme(self, batch_indices, X, masses) -> Callable:
        """
        Resolve the update scheme and initialize its parameters.

        Args:
            batch_indices: the split points for given X.
            X: initial coordinates of the run.
            masses: atomic masses broadcast to X.

        Returns:
            The bound eager or fused update method.

        """
        if self.thermostat == "Langevin":
            damp_coeff = self.thermostat_config.get('damping_coeff', 0.01)  # Unit: fs^-1
            self.alpha = math.exp(- damp_coeff * self.time_step)
            eager_scheme = self.__Langevin
        elif self.thermostat == "VR":
            # VR rescales the new velocity from the kinetic energy evaluated
            # at the beginning of the same integration step.
            self.require_Ek_update = True
            eager_scheme = self.__VR
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
            eager_scheme = self.__NoseHoover
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
            eager_scheme = self.__CSVR_cpu if self.device.type != 'cuda' else self.__CSVR_cuda
        else:
            raise NotImplementedError("Unknown Thermostat Type.")

        # Here build the Triton kernel
        fused_scheme = self._build_fused_scheme(batch_indices, X, masses)
        if fused_scheme is not None:
            return fused_scheme
        return eager_scheme

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
        self.update_scheme = self.__resolve_update_scheme(batch_indices, X, masses)
        # Pre-allocate random buffers for CSVR thermostat
        if self.thermostat == 'CSVR':
            _sample = X[0:1]
            # One random pair per batch, matching the eager CPU path.
            _K_shape = (self.free_degree.numel(),)
            self._buf_R = th.empty(_K_shape, device=self.device, dtype=th.float32)
            self._buf_S = th.empty(_K_shape, device=self.device, dtype=th.float32)
        else:
            self._buf_R = None
            self._buf_S = None
        # Graph/buffer will be built on first __CSVR call with real tensors
        self._buf_F = None
        self._graph_A = None
        self._graph_B = None

    # =========================================================================
    #                   Langevin THERMOSTAT
    # =========================================================================
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

    # =========================================================================
    #                   Naive VR THERMOSTAT
    # =========================================================================

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

    # =========================================================================
    #                   CSVR THERMOSTAT
    # =========================================================================

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
            self.Ek, _ = self._reduce_Ek_T(self.batch_tensor, masses, s.V)
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
                self.Ek, _ = self._reduce_Ek_T(self.batch_tensor, masses, s.V)
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

    # =========================================================================
    #                   Nose-Hoover THERMOSTAT
    # =========================================================================
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

    # =========================================================================
    #                   END THERMOSTATS
    # =========================================================================

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


    def _warn_fused_fallback(self, error: Exception) -> None:
        """Log the fused-kernel compilation failure and its fallback.

        Args:
            error: exception raised while compiling or warming up the kernels.

        Returns:
            None.
        """
        self.logger.warning(
            "WARNING: An error occurred when trying to compile the MD kernel, the eager version will be used.  "
            "You can set verbose >= 2 to show the detailed error information."
        )
        if self.verbose >= 2:
            self.logger.warning(f"Error:{error}")

    def _build_fused_scheme(self, batch_indices, X, masses):
        """Compile, warm up and return the fused thermostat scheme.

        Args:
            batch_indices: the split points for given X.
            X: initial coordinates of the run.
            masses: atomic masses broadcast to X.

        Returns:
            The bound fused update method, or None when the fused path is
            unavailable or its warm-up fails.
        """
        if X.device.type != 'cuda' or X.dtype not in (th.float16, th.float32, th.float64):
            return None
        scalar_dtype = th.float32 if X.dtype == th.float16 else X.dtype
        try:
            self.inv_m = (1.0 / masses).contiguous()
            self._scalar_dtype = scalar_dtype
            self._n_batch = self.free_degree.numel()
            self._n_elem = X.numel()
            self._is_irregular = batch_indices is not None
            if self._is_irregular:
                offsets = [0] + list(accumulate(batch_indices))
                self._batch_starts = th.as_tensor(
                    [_ * X.shape[2] for _ in offsets],
                    dtype=th.int32,
                    device=X.device,
                ).contiguous()
                self._n_chunk = 0
            else:
                self._batch_starts = None
                self._n_chunk = X.shape[1] * X.shape[2]
            self._free_degree = self.free_degree.to(scalar_dtype).contiguous()
            self._ek_target = self.EK_TARGET.to(scalar_dtype).contiguous()
            grid = (triton.cdiv(self._n_elem, 256),)
            X_dummy = th.zeros_like(X)
            V_dummy = th.zeros_like(X)
            F_dummy = th.zeros_like(X)
            mask_dummy = th.ones_like(X)
            batch_scatter = self.batch_scatter if self._is_irregular else V_dummy
            batch_starts = self._batch_starts if self._is_irregular else V_dummy
            if self.thermostat == "Langevin":
                self._sigma = (
                    th.sqrt((8314.462618 * self.T_init * (1 - self.alpha ** 2)) / masses) * 1e-5
                ).to(X.dtype).contiguous()
                self._langevin = KernelLauncher(langevin_middle_kernel, grid)
                self._vv2 = KernelLauncher(velocity_verlet_2_kernel, grid)
                self._langevin(
                    X_dummy, V_dummy, F_dummy, self.inv_m, self._sigma,
                    th.zeros_like(X), self.alpha,
                    self.half_time_step_const, self.raw_half_time_step_const,
                    self._n_elem, 256,
                )
                self._vv2(
                    V_dummy, F_dummy, mask_dummy, self.inv_m,
                    self.time_step, self._n_elem, 256,
                )
                return self.__Langevin_fused
            if self.thermostat == "VR":
                self._alpha = th.empty(self._n_batch, dtype=scalar_dtype, device=X.device)
                self._vv1 = KernelLauncher(velocity_verlet_1_kernel, grid)
                self._vv2 = KernelLauncher(velocity_verlet_2_kernel, grid)
                self._vr_scale = KernelLauncher(vr_scale_kernel, grid)
                Ek_dummy = th.ones(self._n_batch, dtype=X.dtype, device=X.device)
                self._vv1(X_dummy, V_dummy, F_dummy, self.inv_m, self.time_step, self._n_elem, 256)
                self._vv2(V_dummy, F_dummy, mask_dummy, self.inv_m, self.time_step, self._n_elem, 256)
                self._vr_scale(
                    V_dummy, Ek_dummy, self._ek_target, batch_scatter,
                    self._n_chunk, self._n_elem, self._is_irregular, 256,
                )
                return self.__VR_fused
            if self.thermostat == "CSVR":
                self._alpha = th.empty(self._n_batch, dtype=scalar_dtype, device=X.device)
                self._randn = th.empty(self._n_batch, dtype=scalar_dtype, device=X.device)
                self._sqrt_c = float(self.sqrt_c)
                self._one_sub_c = float(self.one_sub_c)
                self._epsK = float(self.epsK)
                self._vv1 = KernelLauncher(velocity_verlet_1_kernel, grid)
                self._vv2 = KernelLauncher(velocity_verlet_2_kernel, grid)
                self._csvr_alpha = KernelLauncher(csvr_alpha_kernel, (self._n_batch,), num_warps=8)
                self._scale = KernelLauncher(scale_by_batch_kernel, grid)
                randn = th.zeros(self._n_batch, dtype=scalar_dtype, device=X.device)
                chi2 = th.zeros(self._n_batch, dtype=scalar_dtype, device=X.device)
                self._vv1(X_dummy, V_dummy, F_dummy, self.inv_m, self.time_step, self._n_elem, 256)
                self._vv2(V_dummy, F_dummy, mask_dummy, self.inv_m, self.time_step, self._n_elem, 256)
                self._csvr_alpha(
                    V_dummy, masses, self._alpha, self._ek_target,
                    self._free_degree, randn, chi2, batch_starts,
                    self._n_chunk, self._sqrt_c, self._one_sub_c, self._epsK,
                    self._is_irregular, 1024,
                )
                self._scale(
                    V_dummy, self._alpha, batch_scatter,
                    self._n_chunk, self._n_elem, self._is_irregular, 256,
                )
                return self.__CSVR_fused
            if self.thermostat == "Nose-Hoover":
                self._p_iota = self.p_iota.reshape(-1)
                self._smass = self.smass.reshape(-1).expand(self._n_batch).to(scalar_dtype).contiguous()
                self._target_kT = float(self.T_init * 8.617333262145e-5)
                self._nose_half = KernelLauncher(nose_half_step_1_kernel, grid)
                self._vv2 = KernelLauncher(velocity_verlet_2_kernel, grid)
                self._nose_reduce = KernelLauncher(nose_reduce_update_kernel, (self._n_batch,), num_warps=8)
                self._nose_throttle = KernelLauncher(nose_throttle_reduce_update_kernel, (self._n_batch,), num_warps=8)
                p_iota = self._p_iota.clone()
                self._nose_half(
                    X_dummy, V_dummy, F_dummy, self.inv_m, p_iota,
                    batch_scatter, self.time_step,
                    self.half_time_step_const, self.raw_half_time_step_const,
                    self._n_chunk, self._n_elem, self._is_irregular, 256,
                )
                self._nose_reduce(
                    V_dummy, masses, p_iota, self._free_degree, self._smass,
                    batch_starts, self._n_chunk,
                    self.raw_half_time_step_const, self._target_kT,
                    self._is_irregular, 1024,
                )
                self._vv2(V_dummy, F_dummy, mask_dummy, self.inv_m, self.time_step, self._n_elem, 256)
                self._nose_throttle(
                    V_dummy, masses, p_iota, self._free_degree, self._smass,
                    batch_starts, self._n_chunk,
                    self.raw_half_time_step_const, self._target_kT,
                    self._is_irregular, 1024,
                )
                return self.__NoseHoover_fused
        except Exception as e:
            self._warn_fused_fallback(e)
        return None

    def __Langevin_fused(
            self, s, func, grad_func_, func_args, func_kwargs,
            grad_func_args, grad_func_kwargs, masses, atom_masks,
            is_grad_func_contain_y, batch_indices,
    ) -> None:
        """Advance one BAOAB Langevin step with the fused kernels."""
        with th.no_grad():
            noise = th.randn_like(s.V)
            self._langevin(
                s.X, s.V, s.Force, self.inv_m, self._sigma, noise,
                self.alpha, self.half_time_step_const,
                self.raw_half_time_step_const, self._n_elem, 256,
            )
            Energy, Forces = self._calc_EF(
                s.X, func, func_args, func_kwargs,
                grad_func_, grad_func_args, grad_func_kwargs,
                self.require_grad, is_grad_func_contain_y,
            )
            Forces = Forces.contiguous()
            self._vv2(s.V, Forces, atom_masks, self.inv_m, self.time_step, self._n_elem, 256)
            s.Energy = Energy
            s.Force = Forces

    def __VR_fused(
            self, s, func, grad_func_, func_args, func_kwargs,
            grad_func_args, grad_func_kwargs, masses, atom_masks,
            is_grad_func_contain_y, batch_indices,
    ) -> None:
        """Advance one VR step with the fused kernels."""
        with th.no_grad():
            self._vv1(s.X, s.V, s.Force, self.inv_m, self.time_step, self._n_elem, 256)
            Energy, Forces = self._calc_EF(
                s.X, func, func_args, func_kwargs,
                grad_func_, grad_func_args, grad_func_kwargs,
                self.require_grad, is_grad_func_contain_y,
            )
            Forces = Forces.contiguous()
            self._vv2(s.V, Forces, atom_masks, self.inv_m, self.time_step, self._n_elem, 256)
            self._vr_scale(
                s.V, self.Ek, self._ek_target,
                self.batch_scatter if self._is_irregular else s.V,
                self._n_chunk, self._n_elem, self._is_irregular, 256,
            )
            s.Energy = Energy
            s.Force = Forces

    def __CSVR_fused(
            self, s, func, grad_func_, func_args, func_kwargs,
            grad_func_args, grad_func_kwargs, masses, atom_masks,
            is_grad_func_contain_y, batch_indices,
    ) -> None:
        """Advance one CSVR step with the fused kernels."""
        with th.no_grad():
            self._vv1(s.X, s.V, s.Force, self.inv_m, self.time_step, self._n_elem, 256)
            Energy, Forces = self._calc_EF(
                s.X, func, func_args, func_kwargs,
                grad_func_, grad_func_args, grad_func_kwargs,
                self.require_grad, is_grad_func_contain_y,
            )
            Forces = Forces.contiguous()
            self._vv2(s.V, Forces, atom_masks, self.inv_m, self.time_step, self._n_elem, 256)
            self._randn.normal_()
            chi2 = self._chi2_dist.sample().to(self._scalar_dtype)
            self._csvr_alpha(
                s.V, masses, self._alpha, self._ek_target,
                self._free_degree, self._randn, chi2,
                self._batch_starts if self._is_irregular else s.V,
                self._n_chunk, self._sqrt_c, self._one_sub_c, self._epsK,
                self._is_irregular, 1024,
            )
            self._scale(
                s.V, self._alpha,
                self.batch_scatter if self._is_irregular else s.V,
                self._n_chunk, self._n_elem, self._is_irregular, 256,
            )
            s.Energy = Energy
            s.Force = Forces

    def __NoseHoover_fused(
            self, s, func, grad_func_, func_args, func_kwargs,
            grad_func_args, grad_func_kwargs, masses, atom_masks,
            is_grad_func_contain_y, batch_indices,
    ) -> None:
        """Advance one Nose-Hoover step with the fused kernels."""
        with th.no_grad():
            self._nose_half(
                s.X, s.V, s.Force, self.inv_m, self._p_iota,
                self.batch_scatter if self._is_irregular else s.V,
                self.time_step,
                self.half_time_step_const, self.raw_half_time_step_const,
                self._n_chunk, self._n_elem, self._is_irregular, 256,
            )
            Energy, Forces = self._calc_EF(
                s.X, func, func_args, func_kwargs,
                grad_func_, grad_func_args, grad_func_kwargs,
                self.require_grad, is_grad_func_contain_y,
            )
            Forces = Forces.contiguous()
            self._nose_reduce(
                s.V, masses, self._p_iota, self._free_degree, self._smass,
                self._batch_starts if self._is_irregular else s.V,
                self._n_chunk, self.raw_half_time_step_const, self._target_kT,
                self._is_irregular, 1024,
            )
            self._vv2(s.V, Forces, atom_masks, self.inv_m, self.time_step, self._n_elem, 256)
            self._nose_throttle(
                s.V, masses, self._p_iota, self._free_degree, self._smass,
                self._batch_starts if self._is_irregular else s.V,
                self._n_chunk, self.raw_half_time_step_const, self._target_kT,
                self._is_irregular, 1024,
            )
            s.Energy = Energy
            s.Force = Forces
