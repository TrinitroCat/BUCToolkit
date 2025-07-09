"""

"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: DataLoaders.py
#  Environment: Python 3.12

import random
from typing import Any, Dict, List, Self, Tuple, Literal, Sequence

import numpy as np
import torch as th

from BM4Ckit.utils._CheckModules import check_module

# check modules
_pyg = check_module('torch_geometric.data')
if _pyg is not None:
    pygBatch = _pyg.Batch
else:
    pygBatch = None

dgl = check_module('dgl')


class DglGraphLoader:
    """
    A Data loader to form dgl graph IN MEMORY

    Args:
        data:
            when is_train == True:
                Dict: {
                    'data': List[dgl.Graph],
                    'label': Dict{'energy': Sequence[float], 'forces': Sequence[np.NDArray[n_atom, 3]]}
                }. Wherein 'force' is optional.
            else see `data_names`.
        batch_size: batch size.
        device: the device that data put on.
        shuffle: whether shuffle data.
        is_train: if `is_train` = True, data need to contain labels; else data need not contain labels and the return of labels [i.e., next(iter(dataloader))] was depended on `contain_names`.
        data_names: only works when `is_train` = False.
                if `data_names` is not None, it should be a Sequence(data names) with the same order as data,
                and the returned `labels` [i.e., next(iter(dataloader))] would be data_names instead of "energy" or "forces",
                else `labels` would be None.

    Yields:
            (dgl.DGLGraph, {'energy': energy, 'forces': force})
         or (dgl.DGLGraph, {'energy': energy, })
         or (dgl.DGLGraph, data_names | None) [when is_train == False]

    """

    def __init__(
            self,
            data: Dict[Literal['data', 'labels'] | Literal['data', 'names'], Any],
            batch_size: int,
            device: str | th.device = 'cpu',
            shuffle: bool = True,
            is_train: bool = True,
            data_names: Sequence[str] | None = None
    ) -> None:
        # check import
        if dgl is None:
            raise ImportError('`DglGraphLoader` requires package `dgl` which could not be imported.')
        self.data: List = data['data']
        self.is_train = is_train
        if is_train:
            self.energy: Any = data['labels']['energy']
            self.forces: Any = data['labels'].get('forces', None)
            if shuffle: self.shuffle()
            if not isinstance(self.energy, th.Tensor):
                self.energy_ = th.from_numpy(np.array(self.energy, dtype=np.float32))
        self.data_names = data_names
        self.batchsize = batch_size
        self._n_samp = len(self.data)
        self._index = 0
        self.device = device

    def shuffle(self, ):
        _seed = random.randint(0, 2147483647)
        random.seed(_seed)
        random.shuffle(self.data)
        random.seed(_seed)
        random.shuffle(self.energy)
        if self.forces is not None:
            random.seed(_seed)
            random.shuffle(self.forces)

    def __iter__(self, ) -> Self:
        return self

    def __next__(self, ) -> "Tuple[dgl.DGLGraph, Dict[Literal['energy', 'forces'], th.Tensor] | None]":
        if self._index * self.batchsize < self._n_samp:
            data = self.data[self._index * self.batchsize: (self._index + 1) * self.batchsize]
            data = dgl.batch(data)
            data = data.to(self.device)  # type: ignore

            if self.is_train:  # when training
                energy = self.energy_[self._index * self.batchsize: (self._index + 1) * self.batchsize]
                energy = energy.to(self.device)

                if self.forces is not None:
                    force = self.forces[self._index * self.batchsize: (self._index + 1) * self.batchsize]
                    force = th.from_numpy(np.concatenate(force)).to(self.device)
                    label: Dict[Literal['energy', 'forces'], th.Tensor] | None = {'energy': energy, 'forces': force}
                else:
                    label: Dict[Literal['energy'], th.Tensor] | None = {'energy': energy}

            elif self.data_names is not None:  # when validating && testing
                label: Sequence[str] = self.data_names[self._index * self.batchsize: (self._index + 1) * self.batchsize]

            else:
                label = None

            self._index += 1
            return data, label
        else:
            raise StopIteration


class PyGDataLoader:
    """
    A Data loader to form pygData IN MEMORY

    Args:
        data: pyg.Data that contain attributes `pos`, `cell`, `atomic_numbers`, `natoms`, `tags`, `fixed`, `pbc`, `idx`.
              `pos`: Tensor, atom coordinates.
              `cell`: Tensor, cell vectors.
              `atomic_numbers`: Tensor, atomic numbers, corresponding to `pos` one by one.
              `natoms`: int, number of atoms.
              `tags`: Tensor, to be compatible with 'FAIR-CHEM' (https://fair-chem.github.io/),
                      which fixed slab part is set to 0, free slab part is 1, adsorbate is 2.
              `fixed`: Tensor, fixed tag, which fixed atoms are 0, free atoms are 1.
              `pbc`: List[bool, bool, bool], where to be periodic at x, y, z directions.
        batch_size: batch size.
        device: the device that data put on.
        shuffle: whether shuffle data.
        is_train: if `is_train` = True, data need to contain labels; else data need not contain labels and the return of labels [i.e., next(iter(dataloader))] was depended on `contain_names`.
        data_names: only works when `is_train` = False.
                if `data_names` is not None, it should be a Sequence(data names) with the same order as data,
                and the returned `labels` [i.e., next(iter(dataloader))] would be data_names instead of "energy" or "forces",
                else `labels` would be None.

    Yields:
            (pyg.Data, {'energy': energy, 'forces': force})
         or (pyg.Data, {'energy': energy, })
         or (pyg.Data, data_names | None) [when is_train == False]

    """
    def __init__(
            self,
            data: Dict[Literal['data', 'labels'] | Literal['data', 'names'], Any],
            batch_size: int,
            device: str | th.device = 'cpu',
            shuffle: bool = True,
            is_train: bool = True,
            data_names: Sequence[str] | None = None
    ) -> None:

        # check module
        if pygBatch is None:
            raise ImportError('`PyGDataLoader` requires package `torch-geometric` which could not be imported.')
        self.data: List = data['data']
        self.is_train = is_train
        if is_train:
            self.energy: Any = data['labels']['energy']
            self.forces: Any = data['labels'].get('forces', None)
            if shuffle: self.shuffle()
            if not isinstance(self.energy, th.Tensor):
                self.energy_ = th.from_numpy(np.array(self.energy, dtype=np.float32))
        self.data_names = data_names
        self.batchsize = batch_size
        self._n_samp = len(self.data)
        self._index = 0
        self.device = device

    def shuffle(self, ):
        _seed = random.randint(0, 2147483647)
        random.seed(_seed)
        random.shuffle(self.data)
        random.seed(_seed)
        random.shuffle(self.energy)
        if self.forces is not None:
            random.seed(_seed)
            random.shuffle(self.forces)

    def __iter__(self, ) -> Self:
        return self

    def __next__(self, ) -> Tuple[pygBatch, Dict[Literal['energy', 'forces'], th.Tensor] | None]:
        if self._index * self.batchsize < self._n_samp:
            data = self.data[self._index * self.batchsize: (self._index + 1) * self.batchsize]
            data = pygBatch.from_data_list(data)
            data = data.to(self.device)  # type: ignore

            if self.is_train:
                energy = self.energy_[self._index * self.batchsize: (self._index + 1) * self.batchsize]
                energy = energy.to(self.device)

                if self.forces is not None:
                    force = self.forces[self._index * self.batchsize: (self._index + 1) * self.batchsize]
                    force = th.from_numpy(np.concatenate(force)).to(self.device)
                    label: Dict[Literal['energy', 'forces'], th.Tensor] | None = {'energy': energy, 'forces': force}
                else:
                    label: Dict[Literal['energy'], th.Tensor] | None = {'energy': energy}

            elif self.data_names is not None:
                label: Sequence[str] = self.data_names[self._index * self.batchsize: (self._index + 1) * self.batchsize]

            else:
                label = None

            self._index += 1
            return data, label
        else:
            raise StopIteration


class BatchStructuresDataLoader:
    """
    A Data loader to form BatchStructure
    """

    def __init__(
            self,
            data_path: str,
            batch_size: int,
            device: str | th.device = 'cpu',
            shuffle: bool = True,
            is_train: bool = True,
            data_names: Sequence[str] | None = None
    ) -> None:
        self.data_path = data_path
        self.data_names = data_names
        self.batchsize = batch_size
        self._index = 0
        self.device = device
