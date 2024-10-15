"""

"""
from typing import Any, Dict, List, Self, Tuple, Literal, Sequence

import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader

import random

from torch_geometric.data import Data as pygData
from torch_geometric.data import Batch as pygBatch

import yaml
from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures


class PyGData:
    """

    """

    def __init__(self) -> None:
        super().__init__()


class PyGDataLoader:
    """
    A Data loader to form a pygData.
    """
    def __init__(self,
                 data: Dict[Literal['data', 'labels'] | Literal['data', 'names'], Any],
                 batchsize: int, device: str | th.device = 'cpu',
                 shuffle: bool = True,
                 is_train: bool = True,
                 data_names: Sequence[str] | None = None
                 ) -> None:
        """

        Args:
            data: Dict: {'data': List[pyg.Data], 'label': Dict['energy': Sequence[float], 'forces': Sequence[np.NDArray[n_atom, 3]]]}. Wherein 'force' is optional.
            batchsize: batch size.
            device: the device that data put on.
            shuffle: whether shuffle data.
            is_train: if `is_train` = True, data need to contain labels; else data need not contain labels and the return of labels [i.e., next(iter(dataloader))] was depended on `contain_names`.
            data_names: only work when `is_train` = False.
                    if `data_names` is not None, it should be a Sequence(data names) with the same order as data,
                    and the returned `labels` [i.e., next(iter(dataloader))] would be data_names, else `labels` is None.
        """
        self.data: List = data['data']
        self.is_train = is_train
        if is_train:
            self.energy: Any = data['labels']['energy']
            self.forces: Any = data['labels'].get('forces', None)
            if shuffle: self.shuffle()
            if not isinstance(self.energy, th.Tensor):
                self.energy_ = th.from_numpy(np.array(self.energy, dtype=np.float32))
        self.data_names = data_names
        self.batchsize = batchsize
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

    """
    def __init__(self):  # TODO
        pass
