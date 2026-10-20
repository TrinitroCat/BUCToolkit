""" Canonical ensemble (NVT) Molecular Dynamics with constraints. """

#  Copyright (c) 2024-2026.4.25, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: ConstrNVT.py
#  Environment: Python 3.12

from typing import Callable, Dict, List, Literal, Tuple

import torch as th

from .NVT import NVT


class ConstrNVT(NVT):
    """
    Compatibility shell of the constrained canonical ensemble (NVT).

    Constraint support is now part of :class:`NVT` itself and is enabled by a
    non-None ``constr_func``.  This class only preserves the historical
    ``ConstrNVT`` argument order and forwards every argument to ``NVT``.

    Parameters:
        time_step: float, time per step (ps).
        max_step: int, maximum steps.
        thermostat: str, the thermostat of NVT ensemble.
        thermostat_config: Dict|None, configs of thermostat. {'damping_coeff': float} for Langevin, {'time_const': float} for CSVR, {'virt_mass': float} for Nose-Hoover.
        constr_func: Callable[[th.Tensor], th.Tensor] = None, the constraint function.
        constr_val: Callable[..., th.Tensor|Tuple[th.Tensor]] | th.Tensor = None, the constraint value that can depend on time.
            It may accept `(t)` or `(t, c0)` which `c0` is the initial constraint values.
        constr_threshold: float = 1e-5, the constraint error tolerance.
        require_fixman: bool = False, whether to calculate the Fixman term for constraint MD.
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

    Note:
        This class is exactly the same as :class:`NVT`; new code should use
        ``NVT(..., constr_func=...)`` directly.
    """

    def __init__(
            self,
            time_step: float,
            max_step: int,
            thermostat: Literal['Langevin', 'VR', 'Nose-Hoover', 'CSVR'],
            thermostat_config: Dict | None = None,
            constr_func: Callable[[th.Tensor], th.Tensor] | None = None,
            constr_val: Callable[..., th.Tensor | Tuple[th.Tensor]] | th.Tensor | None = None,
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
        """Create a constrained NVT runner through the NVT constructor.

        Args:
            time_step: Time per step in fs.
            max_step: Maximum number of steps.
            thermostat: Thermostat name.
            thermostat_config: Thermostat parameters, or None for defaults.
            constr_func: Constraint function, or None for plain MD.
            constr_val: Constant target or a time-dependent target callback.
            constr_threshold: Absolute constraint error tolerance.
            require_fixman: Whether the Fixman terms are evaluated.
            T_init: Initial temperature.
            output_file: Binary trajectory path, or None.
            output_structures_per_step: Output period in steps.
            device: Device of the run.
            verbose: Print level.
            dump_quantities: Binary columns, or None for the defaults.
            log_quantities: Text-log fields, or None for the defaults.

        Returns:
            None.
        """
        super().__init__(
            time_step=time_step,
            max_step=max_step,
            thermostat=thermostat,
            thermostat_config=thermostat_config,
            T_init=T_init,
            output_file=output_file,
            output_structures_per_step=output_structures_per_step,
            device=device,
            verbose=verbose,
            is_compile=False,
            compile_kwargs=None,
            dump_quantities=dump_quantities,
            log_quantities=log_quantities,
            constr_func=constr_func,
            constr_val=constr_val,
            constr_threshold=constr_threshold,
            require_fixman=require_fixman,
        )
        if self.verbose > 0:
            self.logger.info(
                "Note: the ConstrNVT has been integrated into NVT in current version. "
                "They are exactly the same now."
            )
