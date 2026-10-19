"""
Universal simulation-state container for MD, MC, Structure-Optimization, etc.
"""
#  Copyright (c) 2026.7.10, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: StdContainer.py
#  Environment: Python 3.12

import torch as th


class StdContainer:
    """Universal simulation-state container for MD, MC, Structure-Optimization, etc.

    Attributes are mutated in-place by integrators/optimizers.
    ``Force`` and ``Energy`` are replaced each step by model evaluation.
    ``Cell``, ``Virial``, ``Stress``, ``Pressure`` are reserved for NpT.
    ``__dict__`` in ``__slots__`` keeps the class open for customisation.
    """
    __slots__ = ('X', 'V', 'Force', 'Energy',
                 'Cell', 'Virial', 'Stress', 'Pressure',
                 '__dict__')

    def __init__(
            self,
            X: th.Tensor | None = None,
            V: th.Tensor | None = None,
            Force: th.Tensor | None = None,
            Energy: th.Tensor | None = None,
            Cell: th.Tensor | None = None,
            Virial: th.Tensor | None = None,
            Stress: th.Tensor | None = None,
            Pressure: th.Tensor | None = None,
            **kwargs
    ):
        self.X = X
        self.V = V
        self.Force = Force
        self.Energy = Energy
        self.Cell = Cell
        self.Virial = Virial
        self.Stress = Stress
        self.Pressure = Pressure
        self.__dict__.update(kwargs)
