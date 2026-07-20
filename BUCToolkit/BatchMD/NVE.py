""" Micro canonical ensemble (NVE) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVE.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import torch as th

from ._BaseMD import _BaseMD


class NVE(_BaseMD):
    """
    Micro canonical ensemble (NVE) molecular dynamics implemented via velocity Verlet algo.

    Parameters:
        time_step: float, time per step (fs).
        max_step: int, maximum steps.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: str|torch.device, device that program rum on.
        verbose: int, control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.
        is_compile: whether to use jit to compile integrator or not.
        compile_kwargs: keyword arguments passed to compile. Only work when is_compile is True.

    Methods:
        run: run BatchMD.

    """

    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 0,
            is_compile: bool = False,
            compile_kwargs: dict | None = None,
    ):
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

    def _register_dump_vars(self):
        """Return the legacy default NVE trajectory-column names.

        Returns:
            List ``['Energy', 'X', 'V', 'Force']`` in binary-column order.
        """
        return ['Energy', 'X', 'V', 'Force']

    def _updateXV(
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
            in place.
        """
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
            s.Energy = Energy
            s.Force = Forces
