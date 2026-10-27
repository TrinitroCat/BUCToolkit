""" Helper functions shared by the molecular-dynamics engines.

This module contains run-level plumbing that does not belong to an integration
scheme: the default dump/log selections of constrained MD and the one-shot
initialization of an optional :class:`BaseConstr` component.
"""

#  Copyright (c) 2024-2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: utils.py
#  Environment: Python 3.12

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np
import torch as th
from torch import nn

from ._BaseMD import FLOAT_TYPE

if TYPE_CHECKING:
    from ._BaseMD import _BaseMD


def _resolve_constrained_quantities(
        require_fixman: bool,
        dump_quantities: Tuple[str, ...] | List[str] | None,
        log_quantities: Tuple[str, ...] | List[str] | None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Resolve the default dump/log selections of a constrained run.

    Args:
        require_fixman: whether the Blue-Moon Fixman fields are enabled.
        dump_quantities: user selection, or None for the constrained defaults.
        log_quantities: user selection, or None for the constrained defaults.

    Returns:
        The resolved dump and log selections as tuples.

    Raises:
        ValueError: if G/w are selected while ``require_fixman`` is False.
    """
    if dump_quantities is None:
        dump_quantities = ('Energy', 'X', 'V', 'Force', 'Fc')
        if require_fixman:
            dump_quantities += ('G', 'w')
    if log_quantities is None:
        log_quantities = ('Energy', 'Ek', 'temperature', 'X', 'V', 'Force', 'Fc')
        if require_fixman:
            log_quantities += ('G', 'w')
    if not require_fixman:
        _has_GW = {'G', 'w'} & (set(dump_quantities) | set(log_quantities))
        if len(_has_GW) > 0:
            raise ValueError(
                f'Fixman quantities {sorted(_has_GW)} require '
                f'require_fixman=True.'
            )
    return tuple(dump_quantities), tuple(log_quantities)


def _initialize_constr(
        md: _BaseMD,
        func: Any | nn.Module,
        X: th.Tensor,
        Element_list: List[List[str]] | List[List[int]],
        masses: th.Tensor,
        V_init: th.Tensor,
        grad_func: Any | nn.Module = None,
        func_args: Sequence = tuple(),
        func_kwargs: Dict | None = None,
        grad_func_args: Sequence = tuple(),
        grad_func_kwargs: Dict | None = None,
        is_grad_func_contain_y: bool = True,
        require_grad: bool = False,
        batch_indices: List[int] | Tuple[int, ...] | th.Tensor | np.ndarray | None = None,
        fixed_atom_tensor: th.Tensor | None = None,
        is_fix_mass_center: bool = False,
) -> None:
    """Initialize a constrained MD run and delegate the engine setup.

    The constraint clock, the mass-weighted target, and the initial velocities
    are corrected before the engine resolves its integration scheme.  The
    thermostat parameters and fused kernels built afterwards therefore use the
    constrained degrees of freedom and the constrained target kinetic energy.

    Args:
        md: MD engine that owns ``_constr`` and implements
            ``_initialize_no_constr``.
        func: Potential-energy callable.
        X: Initial coordinates with shape ``(n_batch, n_atom, n_dim)``.
        Element_list: Element symbols or atomic numbers per atom.
        masses: Atomic masses broadcast to ``X``.
        V_init: Initial velocities with the shape of ``X``.
        grad_func: Force/gradient callable, or None.
        func_args: Positional arguments forwarded to ``func``.
        func_kwargs: Keyword arguments forwarded to ``func``.
        grad_func_args: Positional arguments forwarded to ``grad_func``.
        grad_func_kwargs: Keyword arguments forwarded to ``grad_func``.
        is_grad_func_contain_y: Whether ``grad_func`` receives the energy
            output.
        require_grad: Whether the potential is evaluated with autograd.
        batch_indices: Split sizes of an irregular batch, or None.
        fixed_atom_tensor: Indices of fixed atoms, or None.
        is_fix_mass_center: Whether the mass center is fixed.

    Returns:
        None.  The engine and the constraint component are initialized, and
        the integration scheme is resolved by ``md._initialize_no_constr``.

    Note:
        ``V_init`` is projected onto the constraint-consistent velocity space
        and rescaled to preserve its kinetic energy before the main loop.
    """
    # Start a fresh trajectory clock. The main loop advances ``md.time_now``
    # in place and mirrors it into the constraint component after every step,
    # so the end-of-step constraint target is always ``time_now + time_step``.
    md.time_now = th.scalar_tensor(0., device=md.device, dtype=FLOAT_TYPE)
    md._constr.time_now = th.scalar_tensor(0., device=md.device, dtype=FLOAT_TYPE)
    md._constr.time_step = md.time_step
    md._constr.initialize(
        func=None,
        X=X,
        Element_list=None,
        masses=masses,
    )

    # Register selected constraint fields with correctly shaped prototypes.
    # Ordered dictionaries update repeated names without duplicating them.
    _n_batch = X.shape[0]
    _n_constr = md._constr.R.shape[-1]
    _Fc = th.zeros(_n_batch, _n_constr, device=md.device, dtype=FLOAT_TYPE)
    if md._constr.require_fixman:
        _G = th.zeros(_n_batch, _n_constr, device=md.device, dtype=FLOAT_TYPE)
        _w = th.zeros(_n_batch, device=md.device, dtype=FLOAT_TYPE)
        _constraint_vars = {'Fc': _Fc, 'G': _G, 'w': _w}
    else:
        _constraint_vars = {'Fc': _Fc}

    _dump_constraint_vars = {
        _name: _value for _name, _value in _constraint_vars.items()
        if _name in md.get_dump_vars()
    }
    _log_constraint_vars = {
        _name: _value for _name, _value in _constraint_vars.items()
        if _name in md.get_log_vars()
    }
    if _dump_constraint_vars:
        md.register_extra_dump_vars(**_dump_constraint_vars)
    if _log_constraint_vars:
        md.register_extra_print_vars(**_log_constraint_vars)

    # Project the initial velocity and rescale it to its original kinetic
    # energy, so the constrained run starts tangent to the manifold.
    ProjV = md._constr._project1(V_init)
    Ek = th.sum(
        masses * V_init ** 2,
        dim=(-2, -1),
        keepdim=True,
    )
    Ek_p = th.sum(
        masses * ProjV ** 2,
        dim=(-2, -1),
        keepdim=True,
    )
    V_init.copy_(th.where(Ek_p < 1e-5, 0., th.sqrt(Ek / Ek_p) * ProjV))

    # Each independent constraint removes one degree of freedom, so the target
    # kinetic energy must be recalculated before the thermostat is resolved.
    md.free_degree -= _n_constr
    if batch_indices is not None:  # Unit: eV/atom. kB = 8.6173332621e-5 eV/K
        md.EK_TARGET = th.tensor(
            [((md.free_degree / 2.) * 8.617333262145e-5 * md.T_init) for _n_atom in batch_indices],
            dtype=X.dtype,
            device=md.device,
        )
    else:
        md.EK_TARGET = (md.free_degree / 2.) * 8.617333262145e-5 * md.T_init

    # Resolve the integration scheme only after the constrained degrees of
    # freedom, target kinetic energy, and initial velocities are final.
    md._initialize_no_constr(
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
        is_fix_mass_center,
    )
