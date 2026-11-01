#  Copyright (c) 2024-2026.11.2, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: pyg_dataloader.py
#  Environment: Python 3.12
import random
from typing import Any, Dict, List, Self, Tuple, Literal, Sequence

import numpy as np
import torch as th

from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.BatchStructures import Batch

# check modules
_pyg = check_module('torch_geometric.data')
if _pyg is not None:
    pygBatch = _pyg.Batch
else:
    pygBatch = Batch


class PyGDataLoader:
    """
    A Data loader to form pygData IN MEMORY

    Args:
        data: Dict: {'data': D, label: L}, where
            D is a List of pyg.Data that contains attributes `pos`, `cell`, `atomic_numbers`, `natoms`, `tags`, `fixed`, `pbc`, `idx`.
              `pos`: Tensor, atom coordinates.
              `cell`: Tensor, cell vectors.
              `atomic_numbers`: Tensor, atomic numbers, corresponding to `pos` one by one.
              `natoms`: int, number of atoms.
              `tags`: Tensor, to be compatible with 'FAIR-CHEM' (https://fair-chem.github.io/),
                      which fixed slab part is set to 0, free slab part is 1, adsorbate is 2.
              `fixed`: Tensor, fixed tag, which fixed atoms are 0, free atoms are 1.
              `pbc`: List[bool, bool, bool], where to be periodic at x, y, z directions.
              `idx`: str, the name of this structure.
            L is Dict{'energy': Sequence[float], 'forces': Sequence[np.NDArray[n_atom, 3]]}, 'forces' is optional.
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
            else:
                self.energy_ = self.energy

        self.data_names = data_names
        self.batchsize = batch_size
        self._n_samp = len(self.data)
        self._index = 0
        self.device = device
        # set the flag
        self._LOADER_TYPE = 'Train'

    def shuffle(self, ):
        _seed = random.randint(0, 2147483647)
        # use local seed instead
        random.Random(_seed).shuffle(self.data)
        random.Random(_seed).shuffle(self.energy)
        if self.forces is not None:
            random.Random(_seed).shuffle(self.forces)

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


class ISFSPyGDataLoader:
    """
    A Data loader to form pygData IN MEMORY.
    This Data loader yields Tuple(pygData1, pygData2) instead of (pygData, label).
    Wherein, pygData1 and pygData2 are corresponding to the initial state and finale state of structures.

    Args:
        data: Dict: {'dataIS': D1, 'dataFS': D2}, where
            D1 and D2 are 2 Lists of pyg.Data that contain attributes `pos`, `cell`, `atomic_numbers`, `natoms`, `tags`, `fixed`, `pbc`, `idx`.
              `pos`: Tensor, atom coordinates.
              `cell`: Tensor, cell vectors.
              `atomic_numbers`: Tensor, atomic numbers, corresponding to `pos` one by one.
              `natoms`: int, number of atoms.
              `tags`: Tensor, to be compatible with 'FAIR-CHEM' (https://fair-chem.github.io/),
                      which fixed slab part is set to 0, free slab part is 1, adsorbate is 2.
              `fixed`: Tensor, fixed tag, which fixed atoms are 0, free atoms are 1.
              `pbc`: List[bool, bool, bool], where to be periodic at x, y, z directions.
              `idx`: str, the name of this structure.
        batch_size: batch size.
        device: the device that data put on.
        data_names: if `data_names` is not None, it should be a Sequence(data names) with the same order as data,
                and the returned `labels` [i.e., next(iter(dataloader))] would be data_names instead of "energy" or "forces",
                else `labels` would be None.

    Yields:
            (pyg.Data, pyg.Data)

    """
    def __init__(
            self,
            data: Dict[Literal['dataIS', 'dataFS'], Any],
            batch_size: int,
            device: str | th.device = 'cpu',
            data_names: Sequence[str] | None = None
    ) -> None:

        # check module
        if pygBatch is None:
            raise ImportError('`PyGDataLoader` requires package `torch-geometric` which could not be imported.')
        self.dataIS: List = data['dataIS']
        self.dataFS: List = data['dataFS']
        self.data_names = data_names
        self.batchsize = batch_size
        self._n_samp = len(self.dataIS)
        _n_samp_check = len(self.dataFS)
        if _n_samp_check != self._n_samp:
            raise RuntimeError(
                f"The number of samples in `dataIS` and `dataFS` do not match. Number of IS: {self._n_samp} & number of FS: {_n_samp_check}."
            )
        self._index = 0
        self.device = device
        # set the flag
        self._LOADER_TYPE = 'ISFS'  # initial state & finale state

    def __iter__(self, ) -> Self:
        return self

    def __next__(self, ) -> Tuple[pygBatch, pygBatch]:
        if self._index * self.batchsize < self._n_samp:
            dataIS = self.dataIS[self._index * self.batchsize: (self._index + 1) * self.batchsize]
            dataIS = pygBatch.from_data_list(dataIS)
            dataIS = dataIS.to(self.device)  # type: ignore
            dataFS = self.dataFS[self._index * self.batchsize: (self._index + 1) * self.batchsize]
            dataFS = pygBatch.from_data_list(dataFS)
            dataFS = dataFS.to(self.device)  # type: ignore

            self._index += 1
            return dataIS, dataFS
        else:
            raise StopIteration
