""" Micro canonical ensemble (NVE) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVE.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import numpy as np
import torch as th
import torch.nn as nn

from ._BaseMD import _BaseMD
from .utils import _initialize_constr, _resolve_constrained_quantities
from BUCToolkit.Bases.BaseConstraints import BaseConstr
from BUCToolkit.utils._CheckModules import check_module

# Triton is optional. CPU-only installations use the eager update, and CUDA
# runs without Triton also fall back to the eager update.
triton = check_module('triton')
_HAS_TRITON = triton is not None

if _HAS_TRITON:
    from .fused_ops.utils import (
        ELEMENTWISE_CANDIDATES,
        KernelLauncher,
        autotune_block_size,
        resolve_block_sizes,
    )
    from .fused_ops.VelocityVerlet import velocity_verlet_1_kernel, velocity_verlet_2_kernel


class NVE(_BaseMD):
    """
    Micro canonical ensemble (NVE) molecular dynamics implemented via velocity Verlet algo.

    Parameters:
        time_step: float, time per step (fs).
        max_step: int, maximum steps.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: str|torch.device, device that program rum on.
        verbose: print level. 0 is silent, 1 prints selected scalars only, and
            2 or greater also prints selected arrays.
        is_compile: whether to use jit to compile integrator or not.
        compile_kwargs: keyword arguments passed to compile. Only work when is_compile is True.
        dump_quantities: names written to the binary trajectory.
        log_quantities: names printed in the text log.
        constr_func: constraint function; None runs plain NVE, otherwise the
            constraint component is enabled.
        constr_val: constant constraint target or a time-dependent callback.
        constr_threshold: absolute constraint error tolerance, or `'auto'` to estimate it at initialization.
        require_fixman: whether the Fixman geometric force and Blue-Moon
            weight are evaluated.

    Methods:
        run: run BatchMD.

    """

    __slots__ = ('inv_m', '_vv1', '_vv2', '_n_elem', '_block_size')

    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 0,
            is_compile: Literal['full', 'partial', 'no'] | bool = False,
            compile_kwargs: dict | None = None,
            dump_quantities: Tuple[str, ...] | List[str] | None = None,
            log_quantities: Tuple[str, ...] | List[str] | None = None,
            constr_func: Callable[[th.Tensor], th.Tensor] | None = None,
            constr_val: Callable[..., th.Tensor | Tuple[th.Tensor]] | th.Tensor | None = None,
            constr_threshold: float | Literal['auto'] = 'auto',
            require_fixman: bool = False,
    ) -> None:
        """Create an NVE runner, optionally with a constraint component.

        Args:
            time_step: Time per step in fs.
            max_step: Maximum number of steps.
            T_init: Initial temperature.
            output_file: Binary trajectory path, or None.
            output_structures_per_step: Output period in steps.
            device: Device of the run.
            verbose: Print level.
            is_compile: Whether to compile the integrator using `torch.compile`.
                Note that this jit compilation is probably NOT improve the efficiency.
            compile_kwargs: Keyword arguments for compilation.
            dump_quantities: Binary columns, or None for the defaults.
            log_quantities: Text-log fields, or None for the defaults.
            constr_func: Constraint function; None runs plain NVE.
            constr_val: Constant target or a time-dependent target callback.
            constr_threshold: Absolute constraint error tolerance, or `'auto'` to estimate it at initialization.
            require_fixman: Whether the Fixman terms are evaluated.

        Returns:
            None.

        Note:
            A non-None ``constr_func`` enables the constraint path and forces
            ``is_compile`` off. If ``constr_func`` is None, ``constr_val`` and
            ``require_fixman`` are ignored with a warning.
        """
        is_constrained = constr_func is not None
        if is_constrained:
            self.ALLOWED_QUANTITIES = type(self).ALLOWED_QUANTITIES | {'Fc', 'G', 'w'}
            dump_quantities, log_quantities = _resolve_constrained_quantities(
                require_fixman, dump_quantities, log_quantities,
            )
            is_compile = False
        else:
            if dump_quantities is None:
                dump_quantities = ('Energy', 'X', 'V', 'Force')
            if log_quantities is None:
                log_quantities = ('Energy', 'Ek', 'temperature', 'X', 'V')
        super().__init__(
            time_step,
            max_step,
            T_init,
            output_file,
            output_structures_per_step,
            device,
            verbose,
            is_compile,
            compile_kwargs,
            dump_quantities,
            log_quantities,
        )
        if is_constrained:
            self._constr = BaseConstr(
                constr_func=constr_func,
                constr_val=constr_val,
                constr_threshold=constr_threshold,
                require_fixman=require_fixman,
                device=self.device,
                verbose=self.verbose,
            )
            self._has_constr = True
        elif (constr_val is not None or require_fixman) and self.verbose > 0:
            self.logger.warning(
                "The `constr_func` is None while `constr_val` is not None or `require_fixman`. "
                "`constr_val` and `require_fixman` will be ignored. "
            )

    def _register_dump_vars(self):
        """Return the legacy default NVE trajectory-column names.

        Returns:
            List ``['Energy', 'X', 'V', 'Force']`` in binary-column order.
        """
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
        """Initialize one run with or without constraints.

        Args:
            func: Potential-energy callable.
            X: Coordinates of the run.
            Element_list: Element symbols or atomic numbers per atom.
            masses: Atomic masses broadcast to ``X``.
            V_init: Initial velocities, or ``None``.
            grad_func: Force/gradient callable, or ``None``.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            grad_func_args: Positional arguments forwarded to ``grad_func``.
            grad_func_kwargs: Keyword arguments forwarded to ``grad_func``.
            is_grad_func_contain_y: Whether the gradient callable accepts the
                energy output.
            require_grad: Whether the potential is evaluated with autograd.
            batch_indices: Split sizes of an irregular batch, or ``None``.
            fixed_atom_tensor: Indices of fixed atoms, or ``None``.
            is_fix_mass_center: Whether the mass center is fixed.

        Returns:
            None.
        """
        if self._has_constr:
            _initialize_constr(
                self, func, X, Element_list, masses, V_init,
                grad_func, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                is_grad_func_contain_y, require_grad, batch_indices,
                fixed_atom_tensor, is_fix_mass_center,
            )
        else:
            self._initialize_no_constr(
                func, X, Element_list, masses, V_init,
                grad_func, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
                is_grad_func_contain_y, require_grad, batch_indices,
                fixed_atom_tensor, is_fix_mass_center,
            )

    def _initialize_no_constr(
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
        """Precompute the inverse masses and bind the update implementation.

        The fused kernels are compiled and warmed up once on dummy tensors.  On
        CUDA with a supported dtype they replace the eager update; any failure
        keeps the eager implementation.

        Args:
            func: Potential-energy callable.
            X: Coordinates of the run.
            Element_list: Element symbols or atomic numbers per atom.
            masses: Atomic masses broadcast to ``X``.
            V_init: Initial velocities, or ``None``.
            grad_func: Force/gradient callable, or ``None``.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            grad_func_args: Positional arguments forwarded to ``grad_func``.
            grad_func_kwargs: Keyword arguments forwarded to ``grad_func``.
            is_grad_func_contain_y: Whether the gradient callable accepts the
                energy output.
            require_grad: Whether the potential is evaluated with autograd.
            batch_indices: Split sizes of an irregular batch, or ``None``.
            fixed_atom_tensor: Indices of fixed atoms, or ``None``.
            is_fix_mass_center: Whether the mass center is fixed.

        Returns:
            None.
        """
        self.inv_m = (1.0 / masses).contiguous()
        self._updateXV = self._updateXV_eager
        if _HAS_TRITON and X.device.type == 'cuda' and X.dtype in (th.float16, th.float32, th.float64):
            # Section: resolve the run's block size
            # This happens before the fused fallback so an invalid environment
            # value raises instead of silently selecting the eager path.
            block_size, _ = resolve_block_sizes(self.max_step)
            try:
                n_elem = X.numel()
                X_dummy = th.zeros_like(X)
                V_dummy = th.zeros_like(X)
                F_dummy = th.zeros_like(X)
                mask_dummy = th.ones_like(X)
                if block_size is None:
                    block_size = autotune_block_size(
                        velocity_verlet_1_kernel,
                        ELEMENTWISE_CANDIDATES,
                        lambda bs: (
                            X_dummy, V_dummy, F_dummy, self.inv_m,
                            self.time_step, n_elem, bs,
                        ),
                        lambda bs: (triton.cdiv(n_elem, bs),),
                    )
                self._block_size = block_size
                self._n_elem = n_elem
                grid = (triton.cdiv(n_elem, block_size),)

                # Section: create launchers and warm up their specializations
                self._vv1 = KernelLauncher(velocity_verlet_1_kernel, grid)
                self._vv2 = KernelLauncher(velocity_verlet_2_kernel, grid)
                self._vv1(
                    X_dummy, V_dummy, F_dummy, self.inv_m,
                    self.time_step, n_elem, block_size,
                )
                self._vv2(
                    V_dummy, F_dummy, mask_dummy, self.inv_m,
                    self.time_step, n_elem, block_size,
                )
                self._updateXV = self._updateXV_fused
                if self.verbose > 0: self.logger.info(f"JIT compiled fused kernels are used...")
            except Exception as e:
                self._warn_fused_fallback(e)

    def _warn_fused_fallback(self, error: Exception) -> None:
        """Log the fused-kernel compilation failure and its fallback.

        Args:
            error: Exception raised while compiling or warming up the kernels.

        Returns:
            None.
        """
        self.logger.warning(
            "WARNING: An error occurred when trying to compile the MD kernel, the eager version will be used.  "
            "You can set verbose >= 2 to show the detailed error information."
        )
        if self.verbose >= 2:
            self.logger.warning(f"Error:{error}")

    def _updateXV_fused(
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
        """Advance one NVE step with the fused Triton half-steps.

        Args:
            s: Live state containing coordinates, velocities, forces, and
                energies.
            func: Potential-energy callable.
            grad_func_: Normalized gradient callable.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            grad_func_args: Positional arguments forwarded to ``grad_func_``.
            grad_func_kwargs: Keyword arguments forwarded to ``grad_func_``.
            masses: Atomic masses broadcastable to ``s.X``.
            atom_masks: Selective-dynamics mask.
            is_grad_func_contain_y: Whether the gradient callable accepts the
                energy output.
            batch_indices: Irregular-batch atom counts, or ``None``.

        Returns:
            None. The state container is updated in place. When constraints are
            enabled, the position projection runs before the force evaluation
            and the velocity projection after the second half-step.
        """
        with th.no_grad():
            # The position projection needs the coordinates before the drift;
            # the kick does not change X, so cloning before the first kernel is
            # equivalent to cloning between the kick and the drift.
            if self._has_constr:
                X_init = s.X.clone()
                self._vv1(s.X, s.V, s.Force, self.inv_m, self.time_step, self._n_elem, self._block_size)
                Fc, G, w = self._constr._project2(s.X, X_init, s.V)
                s.Fc = Fc
                if G is not None: s.G = G
                if w is not None: s.w = w
            else:
                self._vv1(s.X, s.V, s.Force, self.inv_m, self.time_step, self._n_elem, self._block_size)

            Energy, Forces = self._calc_EF(
                s.X,
                func,
                func_args,
                func_kwargs,
                grad_func_,
                grad_func_args,
                grad_func_kwargs,
                self.require_grad,
                is_grad_func_contain_y,
            )
            Forces = Forces.contiguous()
            self._vv2(s.V, Forces, atom_masks, self.inv_m, self.time_step, self._n_elem, self._block_size)
            if self._has_constr: self._constr._project1(s.V, s.X, out=s.V)
            s.Energy = Energy
            s.Force = Forces

    def _updateXV_eager(
            self, s,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices,
    ) -> None:
        """Advance one velocity-Verlet NVE step in place.

        Args:
            s: Live state containing coordinates, velocities, forces, and
                energies.
            func: Potential-energy callable evaluated at the new coordinates.
            grad_func_: Normalized gradient callable used by ``_calc_EF``.
            func_args: Positional arguments forwarded to ``func``.
            func_kwargs: Keyword arguments forwarded to ``func``.
            grad_func_args: Positional arguments forwarded to ``grad_func_``.
            grad_func_kwargs: Keyword arguments forwarded to ``grad_func_``.
            masses: Atomic masses broadcastable to ``s.X``.
            atom_masks: Selective-dynamics mask applied to the new forces.
            is_grad_func_contain_y: Whether the gradient callable accepts the
                energy output.
            batch_indices: Irregular-batch atom counts, or ``None`` for a
                regular batch. The NVE update itself does not split tensors.

        Returns:
            None. ``s.X``, ``s.V``, ``s.Force``, and ``s.Energy`` are updated
            in place. When constraints are enabled, the position projection
            runs before the force evaluation and the velocity projection after
            the second half-step.
        """
        with th.no_grad():
            if self._has_constr:
                X_init = s.X.clone()
                s.V.addcdiv_(s.Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
                s.X.add_(s.V, alpha=self.time_step)
                Fc, G, w = self._constr._project2(s.X, X_init, s.V)
                s.Fc = Fc
                if G is not None: s.G = G
                if w is not None: s.w = w
            else:
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
            if self._has_constr: self._constr._project1(s.V, s.X, out=s.V)
            s.Energy = Energy
            s.Force = Forces
