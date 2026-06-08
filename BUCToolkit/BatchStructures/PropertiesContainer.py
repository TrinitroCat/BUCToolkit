"""
A standard container class to store properties of the chemical systems
"""
#  Copyright (c) 2026.4.27, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: PropertiesContainer.py
#  Environment: Python 3.12

from dataclasses import dataclass, field, fields
import torch as th

@dataclass(slots=True)
class PropertiesContainer:
    batch_indices: th.Tensor | None = None
    elements: th.Tensor | None = None
    masses: th.Tensor | None = None
    cell: th.Tensor | None = None
    stresses: th.Tensor | None = None
    pressure: th.Tensor | None = None
    virial: th.Tensor | None = None
    coordinates: th.Tensor | None = None
    velocities: th.Tensor | None = None
    energy: th.Tensor | None = None
    forces: th.Tensor | None = None
    #__dict__: dict = field(default_factory=dict, repr=False, compare=False)

    @staticmethod
    def _std_check(x, y):
        if (x is None) or (y is None):
            return True

        return x.shape == y.shape

    @staticmethod
    def _const_check(shape, y):
        if y is None:
            return True

        return y.shape == shape

    @staticmethod
    def _chain_check(*args, std_shape: tuple|None=None):
        if len(args) < 2:
            return True

        _cache_shape = None if std_shape is None else std_shape
        for arg in args:
            if arg is None:
                continue
            if _cache_shape is None:
                _cache_shape = arg.shape
            elif arg.shape != _cache_shape:
                return False

        return True


    def __post_init__(self):
        """ auto check """
        for fd in fields(self):
            val = getattr(self, fd.name)
            if (val is not None) and (not isinstance(val, th.Tensor)):
                raise TypeError(f"All properties must be torch.Tensor or None, but got {fd.name} of {type(val)}.")

        if self.cell is not None:
            cbatch, _, __ = self.cell.shape
            if _ != __ or _ != 3:
                raise ValueError(f"The cell must have the shape of (n_batch, 3, 3), but got {self.cell.shape}.")
        else:
            cbatch = None

        if not self._std_check(self.cell, self.stresses):
            raise ValueError(
                f'Expected stress and cell vector have the same shape, but got {self.cell.shape} and {self.stresses.shape}.'
            )

        # large tensor check
        if not self._chain_check(self.coordinates, self.velocities, self.forces):
            raise ValueError(f"Shapes of coordinates, velocities, and forces do not match.")

        # short tensor check
        sbatch = None
        for pp in (self.coordinates, self.velocities, self.forces):
            if pp is not None:
                sbatch = pp.shape[:2]
        if not self._chain_check(self.elements, self.masses, std_shape=sbatch):
            raise ValueError(f"Shapes of elements and masses do not match.")
        _batch_indices = self.batch_indices.unsqueeze(0) if self.batch_indices is not None else None
        if not self._std_check(_batch_indices, self.masses):
            raise ValueError(f"Shapes of batch_indices and masses do not match.")

        # scalars checks
        if not self._chain_check(self.energy, self.virial, self.pressure, std_shape=(cbatch, )):
            raise ValueError(f"Shapes of energy, virial, and pressure do not match.")

    @property
    def positions(self):
        return self.coordinates

    @property
    def pos(self):
        return self.coordinates

    @property
    def vel(self):
        return self.velocities

    @property
    def atomic_numbers(self):
        return self.elements

    @property
    def batch(self):
        return self.batch_indices

    @property
    def lattice(self):
        return self.cell

    @property
    def batch_size(self):
        if self.batch_indices is None:
            for pp in (self.coordinates, self.velocities, self.forces):
                if pp is not None:
                    return pp.shape[0]
            return 0
        else:
            return th.bincount(self.batch_indices)

    def __len__(self):
        return self.batch_size

if __name__ == '__main__':
    properties = PropertiesContainer(coordinates=th.rand((1, 100, 3)))
    pass