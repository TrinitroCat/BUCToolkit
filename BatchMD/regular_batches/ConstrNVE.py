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
            constr_func: Callable[[th.Tensor], th.Tensor] = None,
            constr_val: Callable[[th.Tensor], th.Tensor|Tuple[th.Tensor]] | th.Tensor = None,
            constr_threshold: float = 1e-5,
            T_init: float = 298.15,
            output_file: str | None = None,
            dump_path: str | None = None,
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
            output_file,
            dump_path,
            output_structures_per_step,
            device,
            verbose
        )
        self._X, self._V = None, None

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
            fixed_atom_tensor,
            is_fix_mass_center
        )
        self._X, self._V = X, V_init

    def _updateXV(
            self, X, V, Force,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y,
    ) -> (th.Tensor, th.Tensor, th.Tensor):
        """ Update X, V, Force, and return X, V, Energy, Force. """
        X: th.Tensor = X.detach()
        with th.no_grad():
            X0 = X.clone()
            V0 = V.clone()
            # V = V + (Force / (2. * masses)) * self.time_step * 9.64853329045427e-3  # half-step veloc. update, to avoid saving 2 Forces Tensors.
            V.add_(Force / masses, alpha=0.5 * self.time_step * 9.64853329045427e-3)
            X.add_(V, alpha=self.time_step)
            # iteratively modify positions into the manifold, i.e., the approx. exp. mapping
            lamb = self._project2(X)  # update X in-place
            # Update F
            with th.set_grad_enabled(self.require_grad):
                X.requires_grad_(self.require_grad)
                Energy = func(X, *func_args, **func_kwargs)
                if is_grad_func_contain_y:
                    Force = - grad_func_(X, Energy, *grad_func_args, **grad_func_kwargs) * atom_masks
                else:
                    Force = - grad_func_(X, *grad_func_args, **grad_func_kwargs) * atom_masks
            # update another half step
            V.add_(Force / masses, alpha=0.5 * self.time_step * 9.64853329045427e-3)
            # project V, i.e., approx. parallel trans.
            V.copy_(self._project1(V))
            # debug
            #self._debug_X_check.append(abs(y[0].item()))
            #self._debug_V_check.append(abs(th.sum(jac * V.unsqueeze(1), dim=(-2, -1))[0].item()))

        return X, V, Energy, Force

    def __updateXV(
            self, X, V, Force,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y,
    ) -> (th.Tensor, th.Tensor, th.Tensor):
        """
        Here the returned X, V, Force and Energy are all the quantities at the midpoint.
        And the quantities in the end point are additionally stored in external attr. self._X, self._V,
        and these attributes would be updated in-place within `self.implicit_midpoint_step`
        """
        # predict the new endpoint by last midpoint and endpoint
        X0 = self._X
        V0 = self._V
        f_constr = th.einsum('bijk, bi -> bjk', self.jac, self.lamb)
        V_mid_pred = V.add((Force + f_constr) / (2. * masses), alpha=self.time_step * 9.64853329045427e-3)
        X_mid_pred = X.add(V_mid_pred, alpha=self.time_step)
        self._V = 2 * V_mid_pred - self._V
        self._X = 2 * X_mid_pred - self._X

        X, V, E, F, lamb = self.implicit_midpoint_step(
            X0, self._X, V0, self._V, Force, self.lamb,
            func, grad_func_, func_args, func_kwargs, grad_func_args, grad_func_kwargs,
            masses, atom_masks, is_grad_func_contain_y,
        )
        self.logger.info(
            f'Constraint forces \\lambda: {np.array2string(lamb.squeeze().numpy(force=True), **SCIENTIFIC_ARRAY_FORMAT)}'
        )
        jac, y = self._jacobian(X)
        self._debug_X_check.append(abs(y[0].item()))
        self._debug_V_check.append(abs(th.sum(jac * V.unsqueeze(1), dim=(-2, -1))[0].item()))
        print(abs(th.sum(jac * V.unsqueeze(1), dim=(-2, -1))[0].item()))

        return X, V, E, F

