""" Micro canonical ensemble (NVE) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVE.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import torch as th
from torch import nn
import numpy as np

from ._ConstrBaseMD import _rConstrBase
from BM4Ckit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT


class ConstrNVE(_rConstrBase):
    """
    Constrained micro canonical ensemble (NVE) molecular dynamics implemented via velocity Verlet algo.

    Parameters:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: str|torch.device, device that program rum on.
        verbose: int, control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.

    Methods:
        run: run BatchMD.

    """

    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            constr_func: Callable[[th.Tensor], th.Tensor] = None,
            constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor = None,
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        super().__init__(
            time_step,
            max_step,
            T_init,
            constr_func,
            constr_val,
            output_file,
            output_structures_per_step,
            device,
            verbose
        )

    def _updateXV(
            self, X, V, Force,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y,
    ) -> (th.Tensor, th.Tensor, th.Tensor):
        """ Update X, V, Force, and return X, V, Energy, Force. """
        X: th.Tensor = X.detach()
        with th.no_grad():
            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
            V.add_(Force / (2. * masses), alpha=self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha=self.time_step)
            # iteratively modify positions into the manifold, i.e., the approx. exp. mapping
            Fc = self._project2(X, V)
            self.logger.info(f'Constraint forces \\lambda: {np.array2string(Fc.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}')
            # Update F
            with th.set_grad_enabled(self.require_grad):
                X.requires_grad_(self.require_grad)
                Energy = func(X, *func_args, **func_kwargs)
                if is_grad_func_contain_y:
                    Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks
            # update another half step
            V.add_(Force / (2. * masses), alpha=self.time_step * 9.64853329045427e-3)
            # project V, i.e., approx. parallel trans.
            V.copy_(self._project1(V))
            # debug
            #jac, y = self._jacobian(X)
            #self._debug_X_check.append(abs(y.item()))
            #self._debug_V_check.append(abs(th.sum(jac * V.unsqueeze(1), dim=(-2, -1)).item()))

        return X, V, Energy, Force
