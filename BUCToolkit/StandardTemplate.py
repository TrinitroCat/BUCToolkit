"""
The standard template of input object and models.
"""
#  Copyright (c) 2026.4.3, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: StandardTemplate.py
#  Environment: Python 3.12

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List, Any

import torch as th


class StandardModel(ABC):
    """
    An abstract base class that convert arbitrary function into BUCToolkit supported format to structure opt, MD, and MC, etc.
    It can receive any user-defined Callable object `model` which returns torch.Tensors, and convert it to the standard dict format
        {'energy': energy, 'forces': forces} by user-override `initialize_model` (optional), `calc_energy`, and `calc_forces` with
        input `args` and `kwargs`.
    DO NOT SUPPORT FOR TRAINING.

    """
    def __init__(
            self,
            model,
            args_for_init: Tuple = tuple(),
            kwargs_for_init: Dict | None = None,
            device: str = 'cpu',
            *args,
            **kwargs
    ):
        """
        The original PyTorch model with (hyper-)parameters args and kwargs.
        Args:
            model: the main function
            args_for_init: arguments passed to model
            kwargs_for_init: keyword arguments passed to model
            args: additional positional arguments for calc_energy or calc_forces, if necessary
            kwargs: additional keyword arguments for calc_energy or calc_forces, if necessary
        """
        if kwargs_for_init is None: kwargs_for_init = {}
        super().__init__()
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.device = th.device(device)

        self.initialize_model(*args_for_init, **kwargs_for_init)

    def initialize_model(self, *args, **kwargs):
        """
        Initialize & config the model, if necessary.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def calc_energy(self, x) -> th.Tensor:
        """
        The method that calculates the energy of given x.
        One may use input args/kwargs in __init__
        Args:
            x: input variable

        Returns: energy

        """
        pass

    @abstractmethod
    def calc_forces(self, x) -> th.Tensor:
        """
        The method that calculates the forces of given x.
        One may use input args/kwargs in __init__
        Args:
            x: input variable

        Returns: forces

        """
        pass

    def __call__(self, x):
        with th.no_grad():
            ener = th.as_tensor(self.calc_energy(x), device=self.device)
            forc = th.as_tensor(self.calc_forces(x), device=self.device)
            return {'energy': ener, 'forces': forc}


class StandardInput(ABC):
    """
    An abstract container as the standard input of advanced API `api`.
    It is fully compatible with `torch-geometric.data.Batch` object.
    Following properties with corresponding type are standard and may require by BUCToolkit:
    _ATTR_TYPE = {
        'idx': '<U128',               # optional, the list of sample names to log out

        'batch': '<i8',               # shape (N, ), the sample sizes in a batch

        'cell': '<f4',                # shape (N, 3, 3), cell vectors. N is the batch size

        'pos': '<f4',                 # shape (\sum_i n_atom_i, n_dim), batched structure coordinates

        'mask': '|i1',                # mask of fixed atoms with the same shape as `pos`. 0 is for fixed and 1 is for free

        'atomic_numbers': '<i4',      # shape (\sum_i n_atom_i, ), tensor of batched atomic numbers.

        'x_diff': '<f4',              # optional, initial displacement direction for Dimer, etc.

        'velocity': '<f4',            # optional, initial velocity for MD.
    }

    where '<' is for Little-endian order, 'i' is for integer, 'f' is for float, and the last number is the bytes of single element.
    For example, `'pos': '<f4'` means each element of `pos` is 4-bytes float, i.e., float32 or single precision float.

    It has properties that can be inquired by follow methods as the input arg `data`:
        def get_batch_size(data):
            # total batch size
            return len(data)

        def get_cell_vec(data):
            # `batch_size` cell vectors stacked at the 1st dim.
            return data.cell.numpy(force=True)

        def get_atomic_number(data):
            # `batch_size` atomic numbers tensor(int64) concatenated at the 1st dim.
            return data.atomic_numbers.unsqueeze(0)

        def get_indx(data):
            # Python list object with `batch_size` elements of each sample's name.
            # each name must have less than 128 bytes.
            _indx: Dict = getattr(data, 'idx', None)
            return _indx

        def get_pos(data):
            # Atomic coordinates tensor in dtype float32 concatenated at the 1st dim.
            return data.pos.unsqueeze(0)

        def get_fixed_mask(data):
            # the same shape as `pos` with dtype int8. Exactly, only 0, 1.
            # Optional. default is all ones.
            mask = getattr(data, 'fixed', None)
            if mask is not None:
                mask = mask.unsqueeze(0)
            return mask

        def get_batch_indx(data):
            # batch_indices in the form of 1d int64 th.Tensor [0, 0, 0, ..., 1, 1, ..., n]
            # where the same number means the indexed data of the same sample.
            return [len(dat.pos) for dat in data.to_data_list()] # or can directly store this attr and return `data.batch`

        def get_init_dX(data):
            # used for Dimer & other algo. requiring finite difference. Have the same shape & dtype as `pos`
            # Optional.
            return getattr(data, self.x_diff_attr, None)

        def get_init_veloc(data):
            # used for MD. The initial velocities that have the same shape & dtype as `pos`
            # Optional.
            veloc = getattr(data, 'velocity', None)
            if veloc is not None:
                veloc = veloc.unsqueeze(0)
            return veloc

    They are fully compatible with `torch_geometric.data.batch` object.
    """
    _ATTR_TYPE = {
        'idx': '<U128',
        'batch': '<i8',
        'cell': '<f4',
        'pos': '<f4',
        'fixed': '|i1',
        'atomic_numbers': '<i4',
        'x_diff': '<f4',
        'velocity': '<f4',
    }

    def __init__(
            self,
            batch: th.Tensor,
            cell: th.Tensor,
            atomic_numbers: th.Tensor,
            pos: th.Tensor,
            idx: Optional[List[str]] = None,
            fixed: Optional[th.Tensor] = None,
            velocity: Optional[th.Tensor] = None,
            x_diff: Optional[th.Tensor] = None,
            **kwargs
    ):
        self._batch = batch
        self._cell = cell
        self._atomic_numbers = atomic_numbers
        self._pos = pos
        self.idx = idx
        self.fixed = fixed
        self.velocity = velocity
        self.x_diff = x_diff
        self.__dict__.update(kwargs)

    # the following properties must be implemented.
    @abstractmethod
    def __len__(self) -> int:
        """ return the sample number in one batch """
        pass

    @property
    @abstractmethod
    def batch(self) -> Optional[th.Tensor]:
        """
        batch indices tensor in the form of 1-D int64 th.Tensor [0, 0, 0, ..., 1, 1, ..., n]
            where the same number means the indexed data of the same sample.
        """
        pass

    @batch.setter
    @abstractmethod
    def batch(self, value: Optional[th.Tensor]) -> None:
        pass

    @property
    @abstractmethod
    def pos(self) -> th.Tensor:
        """
        Atomic coordinates tensor in dtype float32 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, 3).
        """
        pass

    @pos.setter
    @abstractmethod
    def pos(self, value: th.Tensor) -> None:
        pass

    @property
    @abstractmethod
    def cell(self) -> th.Tensor:
        """
        cell tensor in dtype float32 concatenated at the 1st dim. shape: (n_batch, 3, 3)
        """
        pass

    @cell.setter
    @abstractmethod
    def cell(self, value: th.Tensor) -> None:
        pass

    @property
    @abstractmethod
    def atomic_numbers(self) -> th.Tensor:
        """
        Atomic number tensor in dtype int64 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, ).
        """
        pass

    @atomic_numbers.setter
    @abstractmethod
    def atomic_numbers(self, value: th.Tensor) -> None:
        pass

    @abstractmethod
    def to_data_list(self) -> List[Any]:
        """
        Split batched data into the List of each sample.
        Similar to torch_geometric.data.Batch.
        Optional to override.
        """
        raise NotImplementedError(f'Please implement `self.to_data_list()` by overriding in the subclass.')

    def to(self, target: th.dtype | th.device | str) -> None:
        """
        Move data of self to the given device or dtype.
        Args:
            target: the device or dtype to which to move the data.

        """
        # normal attr
        for k, dat in self.__dict__.items():
            if k.startswith('__') and k.endswith('__'):
                continue
            elif isinstance(dat, th.Tensor):
                self.__setattr__(k, dat.to(target))

        # properties
        for k in self._ATTR_TYPE.keys():
            _x = getattr(self, k)
            if isinstance(_x, th.Tensor):
                _x = _x.to(target)
            try:
                setattr(self, k, _x)
            except AttributeError:
                pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert self to the dict format.
        Note that private and special attr (start with '_'/'__') would not put in.
        Returns: Dict of attributes.
        """
        # normal attr
        attr_dict = {k:dat for k, dat in self.__dict__.items() if not k.startswith('_')}
        # properties
        attr_dict.update({k: getattr(self, k) for k in self._ATTR_TYPE.keys() if not k.startswith('_')})

        return attr_dict

class BatchData(StandardInput):
    """
    An example of `StandardInput` for Batched data.
    """

    def __init__(
            self,
            batch: th.Tensor,
            cell: th.Tensor,
            atomic_numbers: th.Tensor,
            pos: th.Tensor,
            idx: Optional[List[str]] = None,
            fixed: Optional[th.Tensor] = None,
            velocity: Optional[th.Tensor] = None,
            x_diff: Optional[th.Tensor] = None,
            **kwargs
    ):
        super(BatchData, self).__init__(
            batch=batch,
            cell=cell,
            atomic_numbers=atomic_numbers,
            pos=pos,
            idx=idx,
            fixed=fixed,
            velocity=velocity,
            x_diff=x_diff,
            **kwargs
        )

    def __len__(self) -> int:
        """ return the sample number in one batch """
        return len(self._batch)

    @property
    def batch(self) -> th.Tensor:
        """
        batch indices tensor in the form of 1-D int64 th.Tensor [0, 0, 0, ..., 1, 1, ..., n]
            where the same number means the indexed data of the same sample.
        """
        return self._batch

    @batch.setter
    def batch(self, value: th.Tensor) -> None:
        if not isinstance(value, th.Tensor):
            raise TypeError(f'Batch must be of type {th.Tensor}, not {type(value)}.')
        elif value.dim() != 1:
            raise ValueError(f'Batch must be a 1-D Tensor, but got {value.dim()}-D.')
        elif th.sum(value) != self.pos.shape[0]:
            raise ValueError(f'Atoms number given by `value` and `self.pos` do not match.')

        self._batch = value.to(th.int64)

    @property
    def pos(self) -> th.Tensor:
        """
        Atomic coordinates tensor in dtype float32 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, 3).
        """
        return self._pos

    @pos.setter
    def pos(self, value: th.Tensor) -> None:
        if not isinstance(value, th.Tensor):
            raise TypeError(f'`pos` must be of type {th.Tensor}, not {type(value)}.')
        self._pos = value

    @property
    def cell(self) -> th.Tensor:
        """
        cell tensor in dtype float32 concatenated at the 1st dim. shape: (n_batch, 3, 3)
        """
        return self._cell

    @cell.setter
    def cell(self, value: th.Tensor) -> None:
        if not isinstance(value, th.Tensor):
            raise TypeError(f'`cell` must be of type {th.Tensor}, not {type(value)}.')
        elif value.dim() != 3:
            raise ValueError(f'`cell` must be a 3-D tensor, but got {value.dim()}-D.')
        elif value.shape != (len(self._batch), 3, 3):
            raise ValueError(f'batch size given by `value` and `self.batch` do not match.')
        self._cell = value

    @property
    def atomic_numbers(self) -> th.Tensor:
        """
        Atomic number tensor in dtype int64 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, ).
        """
        return self._atomic_numbers

    @atomic_numbers.setter
    def atomic_numbers(self, value: th.Tensor) -> None:
        if not isinstance(value, th.Tensor):
            raise TypeError(f"`atomic_numbers` must be of type {th.Tensor}, not {type(value)}.")
        elif value.dim() != 1:
            raise ValueError(f"`atomic_numbers` must be a 1-D tensor, but got {value.dim()}-D.")
        elif len(value) != self.pos.shape[0]:
            raise ValueError(f"Atomic number given by `value` and `self.pos` do not match.")
        self._atomic_numbers = value.to(th.int32)

    def to_data_list(self) -> List[Any]:
        """
        Split batched data into the List of each sample.
        Similar to torch_geometric.data.Batch.
        Optional to override.
        """
        raise NotImplementedError(f'Please implement `self.to_data_list()` by overriding in the subclass.')