""" Canonical ensemble (NVT) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2026.4.25, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: ConstrNVT.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import warnings
import math

import torch as th
from torch import nn
import numpy as np

from BUCToolkit.utils.index_ops import index_reduce
from ._BaseConstrMD import _BaseConstrMD


class ConstrNVT(_BaseConstrMD):
    """
    Constrained canonical ensemble (NVT) molecular dynamics implemented via velocity Verlet algo.

    Parameters:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        constr_func: Callable[[th.Tensor], th.Tensor] = None, the constraint function.
        constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor = None, the constraint value that can depend on the accumulate time.
        constr_threshold: float = 1e-5, the constraint error tolerance.
        require_fixman: bool = False, whether to calculate the Fixman term for constraint MD.
        thermostat: str, the thermostat of NVT ensemble.
        thermostat_config: Dict|None, configs of thermostat. {'damping_coeff': float} for Langevin, {'time_const': float} for CSVR, {'virt_mass': float} for Nose-Hoover.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: str|torch.device, device that program rum on.
        verbose: print level. 0 is silent, 1 prints selected scalars only, and
            2 or greater also prints selected arrays.
        dump_quantities: names written to the binary trajectory. ``Fc`` and
            enabled Fixman fields are included by default.
        log_quantities: names printed in the text log. Constraint diagnostics
            are included by default and may be changed explicitly.

    Methods:
        run: run BatchMD.
    """

    def __init__(
            self,
            time_step: float,
            max_step: int,
            thermostat: Literal['Langevin', 'VR', 'Nose-Hoover', 'CSVR'],
            thermostat_config: Dict | None = None,
            constr_func: Callable[[th.Tensor], th.Tensor] = None,
            constr_val: Callable[[th.Tensor], th.Tensor | Tuple[th.Tensor]] | th.Tensor = None,
            constr_threshold: float = 1e-5,
            require_fixman: bool = False,
            T_init: float = 298.15,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2,
            dump_quantities: Tuple[str, ...] | List[str] | None = None,
            log_quantities: Tuple[str, ...] | List[str] | None = None,
    ) -> None:
        """
        Constrained canonical ensemble (NVT) molecular dynamics implemented via velocity Verlet algo.

        Parameters:
            time_step: float, time per step (ps).
            max_step: int, maximum steps.
            constr_func: Callable[[th.Tensor], th.Tensor] = None, the constraint function.
            constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor = None, the constraint value that can depend on the accumulate time.
            constr_threshold: float = 1e-5, the constraint error tolerance.
            require_fixman: bool = False, whether to calculate the Fixman term for constraint MD.
            thermostat: str, the thermostat of NVT ensemble.
            thermostat_config: Dict|None, configs of thermostat. {'damping_coeff': float} for Langevin, {'time_const': float} for CSVR, {'virt_mass': float} for Nose-Hoover.
            T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
            output_structures_per_step: int, output structures per output_structures_per_step steps.
            device: str|torch.device, device that program rum on.
            verbose: print level. 0 is silent, 1 prints selected scalars only,
                and 2 or greater also prints selected arrays.
            dump_quantities: names written to the binary trajectory. ``Fc``
                and enabled Fixman fields are included by default.
            log_quantities: names printed in the text log. Constraint
                diagnostics are included by default.

        Methods:
            run: run BatchMD.

        """
        super().__init__(
            time_step,
            max_step,
            T_init,
            constr_func,
            constr_val,
            constr_threshold,
            require_fixman,
            output_file,
            output_structures_per_step,
            device,
            verbose,
            dump_quantities,
            log_quantities,
        )
        __ENSEMBLES_DICT = {'Langevin': None, 'VR': None, 'Nose-Hoover': None, 'CSVR': None}
        if thermostat not in {'Langevin', 'Langevin_old', 'test', 'VR', 'CSVR', 'Nose-Hoover'}: raise ValueError(f'Unknown Thermostat {thermostat}')
        self.thermostat = thermostat
        if thermostat_config is None:
            thermostat_config = dict()
        self.thermostat_config = thermostat_config
        self.update_scheme = None
        self.half_time_step_const = 0.5 * self.time_step * 9.64853329045427e-3
        self.raw_half_time_step_const = 0.5 * self.time_step

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
            # VR rescales the new velocity from the kinetic energy evaluated
            # at the beginning of the same integration step.
            self.require_Ek_update = True
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
        self.update_scheme = self.__resolve_update_scheme(batch_indices)

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
            batch_indices,
    ) -> None:
        # read thermostat configs
        s.X = s.X.detach()
        X_init = s.X.clone()
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
            # the rest half-step
            s.X.add_(s.V, alpha=raw_half_time_step_const)
            Fc, G, w = self._project2(s.X, X_init, s.V)  # in-place update
            s.Fc = Fc
            if G is not None: s.G = G
            if w is not None: s.w = w
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
            # last update
            s.V.addcdiv_(Forces, masses, value=self.half_time_step_const)
            self._project1(s.V, s.X, out=s.V)
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
            batch_indices,
    ) -> None:
        # NVE Step
        s.X = s.X.detach()
        with th.no_grad():
            X_init = s.X.clone()
            s.V.addcdiv_(s.Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            s.X.add_(s.V, alpha=self.time_step)
            Fc, G, w = self._project2(s.X, X_init, s.V)  # in-place update
            s.Fc = Fc
            if G is not None: s.G = G
            if w is not None: s.w = w
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
            self._project1(s.V, s.X, out=s.V)
            if batch_indices is not None:
                alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)
                s.V *= alpha.transpose(0, 1)[:, self.batch_scatter, :]
            else:
                alpha = th.sqrt(self.EK_TARGET / self.Ek).unsqueeze(-1).unsqueeze(-1)
                s.V *= alpha
            s.Energy = Energy
            s.Force = Forces

    def __CSVR(
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
        """Advance one constrained CSVR step using analytic rescaling.

        The thermostat factor combines exponential relaxation with a
        chi-squared random variate. Position and velocity constraints are
        projected before stochastic velocity rescaling.

        Args:
            s: Live constrained-MD state container.
            func: Potential-energy callable.
            grad_func_: Normalized gradient callable.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            grad_func_args: Positional arguments forwarded to ``grad_func_``.
            grad_func_kwargs: Keyword arguments forwarded to ``grad_func_``.
            masses: Atomic masses broadcastable to the coordinate shape.
            atom_masks: Selective-dynamics mask applied to new forces.
            is_grad_func_contain_y: Whether the gradient callable accepts the
                energy output.
            batch_indices: Irregular-batch atom counts, or ``None`` for a
                regular batch.

        Returns:
            None. The method updates the live state and constraint quantities
            in place.
        """
        half_time_step_const = self.half_time_step_const
        time_step = self.time_step
        X_init = s.X.clone()
        with th.no_grad():
            s.V.addcdiv_(s.Force, masses, value=half_time_step_const)
            s.X.add_(s.V, alpha=time_step)
            # apply constraints
            Fc, G, w = self._project2(s.X, X_init, s.V)  # in-place update
            s.Fc = Fc
            if G is not None: s.G = G
            if w is not None: s.w = w
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
            s.V.addcdiv_(Forces, masses, value=half_time_step_const)
            self._project1(s.V, s.X, out=s.V)
            if self.Ek_T_graph is not None:
                self.Ek_T_graph.replay()
            else:
                self.Ek, _ = self._reduce_Ek_T(batch_indices, masses, s.V)

            Nf = self.free_degree  # shape (n_batch,)
            if batch_indices is not None:
                K = th.clamp(self.Ek, min=self.epsK)  # shape (n_batch,)
                K0 = self.EK_TARGET  # (n_batch, )

                f = self.one_sub_c * K0 / (Nf * K)
                R = th.randn_like(K)
                S = self._chi2_dist.sample()  # shape (n_batch,)

                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2: th.Tensor = th.addcmul(self.sqrt_c, sqrt_f, R) ** 2
                alpha2.addcmul_(f, S).clamp_min_(self.epsK)

                alpha = th.sqrt(alpha2).reshape(1, -1, 1)  # (n_batch, 1, 1)
                s.V *= alpha.index_select(1, self.batch_scatter)

            else:
                K = th.clamp(self.Ek, min=self.epsK)  # (n_batch,)
                K0 = self.EK_TARGET  # scalar or (n_batch,)

                f = self.one_sub_c * K0 / (Nf * K)
                R = th.randn_like(K)
                S = self._chi2_dist.sample()

                sqrt_f = th.sqrt(th.clamp(f, min=0.0))
                alpha2: th.Tensor = th.addcmul(self.sqrt_c, sqrt_f, R) ** 2
                alpha2.addcmul_(f, S).clamp_min_(self.epsK)

                alpha = th.sqrt(alpha2).reshape(-1, 1, 1)  # (n_batch, 1, 1)
                s.V *= alpha
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
        X_init = s.X.clone()
        # Main update
        with th.no_grad():
            if batch_indices is not None:
                _iota = self.p_iota[:, self.batch_scatter, :]
            else:
                _iota = self.p_iota
            s.V.addcdiv_(s.Force, masses, value=half_time_step_const)
            s.V.mul_(th.exp(- _iota * raw_half_time_step_const))
            s.X.add_(s.V, alpha=time_step)
            # apply constraints
            Fc, G, w = self._project2(s.X, X_init, s.V)  # in-place update
            s.Fc = Fc
            if G is not None: s.G = G
            if w is not None: s.w = w

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
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=raw_half_time_step_const
                )
            else:
                reduced_Ek = th.sum(masses * s.V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=raw_half_time_step_const
                )
                s.V.addcdiv_(Forces, masses, value=half_time_step_const)
                s.V.mul_(th.exp(- _iota * raw_half_time_step_const))
                reduced_Ek = th.sum(masses * s.V ** 2 * 103.642696562621738, dim=(-2, -1), keepdim=True)
                self.p_iota.addcdiv_(
                    th.sub(reduced_Ek, self.long_free_degree, alpha=self.T_init * 8.617333262145e-5),
                    smass, value=raw_half_time_step_const
                )
            self._project1(s.V, s.X, out=s.V)
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
            batch_indices,
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
