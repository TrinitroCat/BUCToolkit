""" Micro canonical ensemble (NVE) Molecular Dynamics via Verlet algo. """

#  Copyright (c) 2024-2026.4.25, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: ConstrNVE.py
#  Environment: Python 3.12


import os
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import torch as th
from torch import nn
import numpy as np

from BUCToolkit.BatchMD._BaseConstrMD import _BaseConstrMD
from BUCToolkit.utils._print_formatter import FLOAT_ARRAY_FORMAT, SCIENTIFIC_ARRAY_FORMAT
from BUCToolkit.utils._Element_info import DTYPE

FLOAT_TYPE = os.environ.get('BT_FLOAT_TYPE', 'float32')
FLOAT_TYPE = DTYPE.get(FLOAT_TYPE, th.float32)


class ConstrNVE(_BaseConstrMD):
    """
    Constrained micro canonical ensemble (NVE) molecular dynamics implemented via velocity Verlet algo.

    Parameters:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        constr_func: Callable[[th.Tensor], th.Tensor] = None, the constraint function.
        constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor = None, the constraint value that can depend on the accumulate time.
        constr_threshold: float = 1e-5, the constraint error tolerance.
        require_fixman: bool = False, whether to calculate the Fixman term for constraint MD.
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
            constr_func: Callable[[th.Tensor], th.Tensor] = None,
            constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor = None,
            constr_threshold: float = 1e-5,
            require_fixman: bool = False,
            T_init: float = 298.15,
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
            constr_threshold,
            require_fixman,
            output_file,
            output_structures_per_step,
            device,
            verbose
        )

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
            is_grad_func_contain_y,
            require_grad,
            batch_indices,
            fixed_atom_tensor,
            is_fix_mass_center
        )

    def _updateXV(
            self, X, V, Force,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y, batch_indices,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """ Update X, V, Force, and return X, V, Energy, Force. """
        # X: th.Tensor = X.contiguous()
        # V: th.Tensor = V.contiguous()
        # masses: th.Tensor = masses.contiguous()
        with th.no_grad():
            X_init = X.clone()
            # X = X + V * self.time_step + (Force / (2. * masses)) * self.time_step ** 2 * 9.64853329045427e-3
            V.addcdiv_(Force, masses, value=0.5 * self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha=self.time_step)
            Fc, G, w = self._project2(X, X_init, V)  # in-place update
            if self.verbose > 0:
                self.logger.info(f'Constraint forces lambda: {np.array2string(Fc.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}')
                if self.require_fixman:
                    self.logger.info(f'1/2 dln|Z|/dX: {np.array2string(G.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}')
                    self.logger.info(f'|Z|^(-1/2): {np.array2string(w.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}')
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
            self._project1(V, X, out=V)

        return X, V, Energy, Force

