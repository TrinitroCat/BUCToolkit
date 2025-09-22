""" Molecular Dynamics base framework with constrains """

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: NVE.py
#  Environment: Python 3.12

# ruff: noqa: E701, E702, E703
from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401

import torch as th

from ._BaseMD import _rBaseMD


class _rConstrBase(_rBaseMD):
    """
    Constrained Base BatchMD with regular batches
    micro canonical ensemble (NVE) molecular dynamics implemented via velocity Verlet algo. on the tangent space.

    Args:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        T_init: float, initial temperature, only to generate initial velocities of atoms by Maxwell-Boltzmann distribution. If V_init is given, T_init will be ignored.
        constr_func: Tuple[Callable, ...], a tuple of Python functions as the constraint functions s_k which s_k(R) == 0. Each function takes one or more arguments, one of which must be a Tensor, and returns one or more Tensors. `None` for identity function. see example below.
        constr_indx: Tuple[th.Tensor, ...], indices of atoms to be constrained. The sequence is the same as `constr_func`.
        output_structures_per_step: int, output structures per output_structures_per_step steps.
        device: str|torch.device, device that program rum on.
        verbose: int, control the detailed degree of output information. 0 for silence, 1 for output Energy and Forces per step, 2 for output all structures.

    Examples for constrains:
        constr_func = (
            lambda x,y: th.linalg.norm(x, y) - 1.5,  # fix the distance between 2 atoms into 1.5

            lambda x1, x2, x3: th.einsum(
            'bij, bij -> b', (x2 - x1), (x3 - x1)
            )/(th.linalg.norm((x2 - x1))*th.linalg.norm((x3 - x1))) - cos(deg2rad(109.5))  # fix the angle of atom5-atom7-atom9 into 109.5 degree
        )

        constr_indx = (
            th.tensor([[0, 2], [3, 1], [4, 5]]),  # according to `constr_func[0]`, here the distance of atoms (0, 2), (3, 1), (4, 5) will be fixed into 1.5.
            th.tensor([[1, 0, 6], [2, 3, 5]])  # according to `constr_func[1]`, here the angle of atoms (1, 0, 6) and (2, 3, 5) will be fixed into 109.5 degree.
        )

    Methods:
        run: run BatchMD.

    """
    def __init__(
            self,
            time_step: float,
            max_step: int,
            T_init: float = 298.15,
            constr_func: Tuple[Callable, ...] | None = None,
            constr_indx: Tuple[th.Tensor, ...] = tuple(),
            output_file: str | None = None,
            output_structures_per_step: int = 1,
            device: str | th.device = 'cpu',
            verbose: int = 2
    ):
        super().__init__(
            time_step,
            max_step,
            T_init,
            output_file,
            output_structures_per_step,
            device,
            verbose
        )
        if constr_func is None:
            constr_func = lambda X: X
        elif len(constr_indx) != len(constr_func):
            raise ValueError(f"`constr_indx` and `constr_func` must have the same length, but got {len(constr_indx)} and {len(constr_func)}")
        else:
            for _f in constr_func:
                if not isinstance(_f, Callable):
                    raise TypeError(f"`constr_func` must consist of callable, but got {type(_f)}")
        self.constr_func = constr_func

        self.constr_indx = constr_indx
        # constraint number
        self.n_tot_constr: int = 0
        for i in constr_indx:
            self.n_tot_constr += len(i)

    def _jacobian(self, X: th.Tensor) -> th.Tensor:
        """
        Compute the Jacobian of all constrains.
        Args:
            X:

        Returns:

        """
        z = th.empty(self.n_tot_constr, device=X.device)
        ptr = 0
        for i, idx in enumerate(self.constr_indx):
            z[ptr, ptr + len(idx)] = self.constr_func(X)

