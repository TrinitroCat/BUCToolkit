"""
BatchStructure base class.
"""
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: BatchStructuresBase.py
#  Environment: Python 3.12

import gc
import re
import logging
import pickle
import os
import random
import sys
import copy
import time
import traceback
import warnings
from itertools import accumulate
import hashlib
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, Any, Iterable, Self

import joblib as jb
import numpy as np

from BUCToolkit.utils._para_flatt_list import flatten
from BUCToolkit.Preprocessing.write_files import WritePOSCARs, write_xyz, write_cif
from BUCToolkit.utils.setup_loggers import has_any_handler


class BatchStructures(object):
    r"""
    The base class of batch structures.
    """

    LIST_ATTR_NAME = (
        '_Sample_ids',
        'Cells',
        'Coords_type',
        'Coords',
        'Fixed',
        'Elements',
        'Numbers',
        'Energies',
        'Forces',
        'Labels'
    )

    def __init__(self) -> None:
        self._ALL_ELEMENTS = ('H', 'He',
                              'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                              'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                              'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                              'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                              'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                              'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                              'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                              'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',)  # element List
        self._ATTR_NAMES = (
            'Sample_ids',
            'Batch_indices',
            'Elements_batch_indices',
            'Cells',
            'Coords_type',
            'Coords',
            'Fixed',
            'Elements',
            'Numbers',
            'Energies',
            'Forces',
            'Labels'
        )
        self._ATTR_TYPE = {
            'Sample_ids': '<U128',
            'Batch_indices': '<i8',
            'Elements_batch_indices': '<i8',
            'Cells': '<f4',
            'Coords_type': '<U2',
            'Coords': '<f4',
            'Fixed': '|i1',
            'Elements': '<U3',
            'Numbers': '<i4',
            'Energies': '<f4',
            'Forces': '<f4',
            'Labels': '|O'
        }

        self._Sample_ids = list()
        self.Mode: Literal['A', 'L'] = 'L'
        # Python List Version
        # Structure part
        self.Atom_list = None
        self.Atomic_number_list = None
        self.Cells: List[np.ndarray] = list()  # cell vectors, float32
        self.Coords_type: List[Literal['C', 'D']] = list()  # coordinate type
        self.Coords: List[np.ndarray] = list()  # atomic coordinates, float32
        self.Fixed: List[np.ndarray] = list()  # fixed atoms masks, 0 for fixed and 1 for free. int8
        self.Elements: List[List[str]] = list()  # elements
        self.Numbers: List[List[int]] = list()  # atom number of each element. int4
        # Properties part
        self.Energies = None  # energies of structures. None | List[float]
        self.Forces = None  # forces of structures. None | List[np.NdArray[N, 3]]
        self.Dist_mat = None  # distance matrices of structures. None | List[np.NdArray[N, N]]
        # Others
        self._indices = None  # a dict of {sample_id: index}, str -> int.
        Z_dict = {key: i for i, key in enumerate(self._ALL_ELEMENTS, 1)}  # a dict which maps element symbols into their atomic numbers.
        self._Z_dict = Z_dict
        self.Labels = None  # structure labels

        # NumPy.NdArray Version
        self._Sample_ids_: np.ndarray | None = None
        # Batch info part
        self.Batch_indices_: np.ndarray | None = None  # storing the split point (ptr) in each array with n_atom length. It both concludes 0 and len(arr).
        self.Elements_batch_indices_: np.ndarray | None = None  # storing the split point (ptr) in each array with n_elem length. It both concludes 0 and len(arr).
        # Structure part
        self.Atom_list_ = None
        self.Atomic_number_list_ = None
        self.Cells_: np.ndarray | None = None  # cell vectors
        self.Coords_type_: np.ndarray[Literal['C', 'D']] | None = None  # coordinate type
        self.Coords_: np.ndarray | None = None  # atomic coordinates
        self.Fixed_: np.ndarray | None = None  # fixed atoms masks, 0 for fixed and 1 for free.
        self.Elements_: np.ndarray[str] | None = None  # elements
        self.Numbers_: np.ndarray[int] | None = None  # atom number of each element
        # Properties part
        self.Energies_ = None  # energies of structures. None | List[float]
        self.Forces_ = None  # forces of structures. None | List[np.NdArray[N, 3]]
        self.Dist_mat_ = None  # distance matrices of structures. None | List[np.NdArray[N, N]]
        # Others
        self.Labels_ = None  # structure labels
        # Security
        self._checksum = None

        # logging
        self.logger = logging.getLogger('Main.BS')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if not has_any_handler(self.logger):
            log_handler = logging.StreamHandler(sys.stdout, )
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)

    @property
    def Sample_ids(self, ):
        return self._Sample_ids

    @Sample_ids.setter
    def Sample_ids(self, val):
        raise ValueError('Sample_ids cannot be directly modified.')

    @classmethod
    def load_from_file(cls, path: str, data_slice: Tuple[int, int] | None = None) -> Self:
        """
        The class method version for loading data from saved files.
        WARNING: It would overwrite the existing BatchStructure data. Use `self.load(..., mode='a')` to retain existing data.
        Args:
            path: the file path.
            data_slice: The piece of data in files to be read. If None, all data in files would be read into memory.

        Returns: Self with loaded data.

        """
        c = cls()
        c.load(path, data_slice, 'w')
        return c

    def _generate_checksum_in_mem(self, ):
        checksum = hashlib.sha256()
        checksum.update(pickle.dumps(self))
        return checksum.hexdigest()

    def _hash_check_in_mem(self, ):
        if self._checksum is None:
            warnings.warn('Could not check data integrity due to missing checksum. It would be generate right now.', RuntimeWarning)
            self._checksum = self._generate_checksum_in_mem()
        else:
            if self._checksum != self._generate_checksum_in_mem():
                raise RuntimeError('Hash check failed. Data might be modified.')

    def _clear_np(self, ):
        """
        Set all np.NdArray information to None.
        """
        self._Sample_ids_: np.ndarray | None = None
        # Batch info part
        self.Batch_indices_: np.ndarray | None = None
        self.Elements_batch_indices_: np.ndarray | None = None
        # Structure part
        self.Atom_list_ = None
        self.Atomic_number_list_ = None
        self.Cells_: np.ndarray | None = None  # cell vectors
        self.Coords_type_: np.ndarray[Literal['C', 'D']] | None = None  # coordinate type
        self.Coords_: np.ndarray | None = list()  # atomic coordinates
        self.Fixed_: np.ndarray | None = list()  # fixed atoms masks, 0 for fixed and 1 for free.
        self.Elements_: np.ndarray[str] | None = None  # elements
        self.Numbers_: np.ndarray[int] | None = None  # atom number of each element
        # Properties part
        self.Energies_ = None  # energies of structures. None | List[float]
        self.Forces_ = None  # forces of structures. None | List[np.NdArray[N, 3]]
        self.Dist_mat_ = None  # distance matrices of structures. None | List[np.NdArray[N, N]]
        # Others
        self.Labels_ = None  # structure labels

    def _np2list(self, release_mem: bool = False):
        """
        Convert data from np.NdArray-format to python-List-format.
        It hardly Consumes Memory.
        """
        self.Coords = np.split(self.Coords_, self.Batch_indices_[1:-1], )
        self.Fixed = np.split(self.Fixed_, self.Batch_indices_[1:-1], )
        if self.Forces_ is not None: self.Forces = np.split(self.Forces_, self.Batch_indices_[1:-1], )

        #self.Numbers = np.split(self.Numbers_, self.Elements_batch_indices_[1:-1], )
        #self.Elements = np.split(self.Elements_, self.Elements_batch_indices_[1:-1], )

        self.Numbers = [
            self.Numbers_[_indx: self.Elements_batch_indices_[i + 1]].tolist()
            for i, _indx in enumerate(self.Elements_batch_indices_[:-1])
        ]
        self.Elements = [
            self.Elements_[_indx: self.Elements_batch_indices_[i + 1]].tolist()
            for i, _indx in enumerate(self.Elements_batch_indices_[:-1])
        ]

        self.Cells = [_ for _ in self.Cells_]

        self._Sample_ids = self._Sample_ids_.tolist()
        self.Coords_type = self.Coords_type_.tolist()
        if self.Energies_ is not None: self.Energies = self.Energies_.tolist()
        if self.Labels_ is not None: self.Labels = self.Labels_.tolist()

        if release_mem:
            self.Numbers_ = None
            self.Elements_ = None
            self._Sample_ids_ = None
            self.Coords_type_ = None
            self.Energies_ = None
            self.Labels_ = None
            gc.collect()

        self.Mode = 'L'

    def _list2np(self, release_mem: bool = False):
        """
        Convert data from python-List-format to np.NdArray-format.
        It Consumes Memory.
        """
        self.Batch_indices_ = np.array([0] + list(accumulate([len(_) for _ in self.Coords])), dtype=np.int64)
        #self.Batch_indices__ = np.apply_along_axis(lambda x: len(x), 0, self.Coords)
        #self.Batch_indices__ = np.cumsum(self.Batch_indices__) - self.Batch_indices__
        self.Elements_batch_indices_ = np.array([0] + list(accumulate([len(_) for _ in self.Numbers])), dtype=np.int64)
        if release_mem:
            # Batch_indices control the coords, atom_list, atomic_number_list, fixed, forces
            self.Coords_ = np.concatenate(self.Coords, dtype=np.float32)
            self.Coords = list()
            self.Fixed_ = np.concatenate(self.Fixed, dtype=np.int8)
            self.Fixed = list()
            if self.Forces is not None:
                self.Forces_ = np.concatenate(self.Forces, dtype=np.float32)
                self.Forces = None
            # Element_batch_indices control the Elements, Numbers
            self.Numbers_ = np.asarray(flatten(self.Numbers, 1), dtype=np.int32)
            self.Numbers = list()
            self.Elements_ = np.asarray(flatten(self.Elements, 1), dtype='<U3')
            self.Elements = list()
            # Others are sequentially filled
            self.Cells_ = np.asarray(self.Cells, dtype=np.float32)
            self.Cells = None
            self.Coords_type_ = np.asarray(self.Coords_type, dtype='<U2')
            self.Coords_type = list()
            self._Sample_ids_ = np.asarray(self._Sample_ids, dtype='<U128')
            self._Sample_ids = list()
            if self.Energies is not None:
                self.Energies_ = np.asarray(self.Energies, dtype=np.float32)
                self.Energies = None
            if self.Labels is not None:
                self.Labels_ = np.asarray(self.Labels)
                self.Labels = None
            gc.collect()
        else:
            # Batch_indices control the coords, atom_list, atomic_number_list, fixed, forces
            self.Coords_ = np.concatenate(self.Coords, dtype=np.float32)
            self.Fixed_ = np.concatenate(self.Fixed, dtype=np.int8)
            if self.Forces is not None: self.Forces_ = np.concatenate(self.Forces, dtype=np.float32)
            # Element_batch_indices control the Elements, Numbers
            self.Numbers_ = np.asarray(flatten(self.Numbers), dtype=np.int32)
            self.Elements_ = np.asarray(flatten(self.Elements), dtype='<U3')
            # Others are sequentially filled
            self.Cells_ = np.asarray(self.Cells, dtype=np.float32)
            self.Coords_type_ = np.asarray(self.Coords_type, dtype='<U2')
            self._Sample_ids_ = np.asarray(self._Sample_ids, dtype='<U128')
            if self.Energies is not None: self.Energies_ = np.asarray(self.Energies, dtype=np.float32)
            if self.Labels is not None: self.Labels_ = np.asarray(self.Labels)

        self.Mode = 'A'

    def change_mode(self, mode: Literal['A', 'L'], release_mem: bool = True):
        """
        Change BatchStructure mode.
        Args:
            mode: 'A' is 'array mode' that all data are stored as concatenated np.ndarray,
             and two ptr arrays of split each structure's atom number are accordingly stored. Some methods in this mode are unavailable.
             'L' is 'list mode' that all data are stored as Lists of np.ndarray/List with (different) shapes of atom numbers.
            release_mem: if True, the data of other mode will be released after change.

        Returns: None

        """
        if mode == 'A':
            if self.Mode == 'L':
                self._list2np(release_mem=release_mem)
                self.Mode = 'A'
        elif mode == 'L':
            if self.Mode == 'A':
                self._np2list(release_mem=release_mem)
                self.Mode = 'L'
        else:
            raise ValueError(f"'mode' must be 'A' or 'L', but got '{mode}'.")

    def _update_indices(self, ):
        if self.Mode == 'L':
            self._indices = {_: ii for ii, _ in enumerate(self._Sample_ids)}
        else:
            self._indices = {_: ii for ii, _ in enumerate(self._Sample_ids_.tolist())}

    def _check_id(self, ):
        """ check whether duplication in Sample_ids """
        self._update_indices()
        if len(self._indices) != len(self):
            raise RuntimeError(
                f'There are duplicate names in the Sample_ids: {set([x for x in self._Sample_ids if self._Sample_ids.count(x) > 1])}'
            )

    def _check_len(self, ):
        """ check whether the length of all prop. consistent. """
        if self.Mode == 'A':
            self._np2list()
            is_convert = True
        else:
            is_convert = False

        qe = True
        qf = True
        ql = True
        q = (len(self.Coords) == len(self.Coords_type) == len(self.Numbers)
             == len(self.Elements) == len(self._Sample_ids) == len(self.Coords_type) == len(self.Cells) == len(self.Fixed))
        if self.Energies is not None: qe = (len(self.Energies) == len(self))
        if self.Labels is not None: ql = (len(self.Labels) == len(self))
        if self.Forces is not None: qf = (len(self.Forces) == len(self))

        if not (q & qe & qf & ql): raise RuntimeError(
            'Sample size (numbers) of some properties does not match. PLEASE DOUBLE CHECK!\n'
            'If you did not manually modify any attributes in this class, '
            'please report this bug to us.\nSample size of all properties:'
            f' Sample ids: {len(self._Sample_ids)}\n Coordinates: {len(self.Coords)}\n Coordinates type: {len(self.Coords_type)}\n'
            f' Elements: {len(self.Elements)}\n Element Numbers: {len(self.Numbers)}\n Cells: {len(self.Cells)}\n Fixations: {len(self.Fixed)}\n'
            f' Energies: {len(self.Energies) if self.Energies is not None else None}\n'
            f' Forces: {len(self.Forces) if self.Forces is not None else None}\n'
            f' Labels: {len(self.Labels) if self.Labels is not None else None}'
        )
        if is_convert:
            self._list2np()

    def check_full(self, ):
        """
        Fully check all data's information
        Returns:

        """
        for ii, _data in enumerate(self):
            try:
                lcoo = _data.Coords[0].shape
                lfix = _data.Fixed[0].shape
                lelem = len(_data.Elements[0])
                lnumb = len(_data.Numbers[0])
                natm = sum(_data.Numbers[0])
                # check coo
                if lcoo[1] != 3 or len(lcoo) != 2:
                    self.logger.warning(f"Wrong coordinates shape of {ii}-th data ({_data.Sample_ids[0]}): {lcoo}. It should be (N, 3).")
                    continue
                if lcoo != lfix:
                    self.logger.warning(
                        f"The {ii}-th data ({_data.Sample_ids[0]}) has inconsistent shape of coordinates and fixations: {lcoo} != {lfix}. "
                    )
                    continue
                if lelem != lnumb:
                    self.logger.warning(
                        f"The {ii}-th data ({_data.Sample_ids[0]}) has inconsistent shape of elements and element numbers: {lelem} != {lnumb}. "
                    )
                    continue
                if lcoo[0] != natm:
                    self.logger.warning(
                        f"The {ii}-th data ({_data.Sample_ids[0]}) has inconsistent atom number of coordinates and elements: {lcoo[0]} != {natm}. "
                    )
                    continue
                if (_data.Forces is not None) and (_data.Forces[0].shape != lcoo):
                    self.logger.warning(
                        f"The {ii}-th data ({_data.Sample_ids[0]}) has inconsistent atom number of coordinates and forces: "
                        f"{lcoo} != {_data.Forces.shape}. "
                    )
                    continue

            except Exception as e:
                self.logger.warning(f"Unknown error occurred while checking {ii}-th data ({_data.Sample_ids[0]}): {e}.")

    def save(self, path: str, mode: Literal['w', 'a'] = 'w'):
        """
        Saving this BatchStructures to a numpy memory-mapping file.
        Each attribute in BatchStructures will be saved in the numpy memmap file with the same name.
        Additionally, file 'head' will record the information which is data type & shape of all attributes, and record which attributes are `None`.

        Args:
            path: the saving directory.
            mode: if 'w', create or overwrite the existing file for reading and writing. If 'a', data will append to the existing file.

        Returns: None

        """
        # check files
        _STANDARD_FILE_LIST = set(list(self._ATTR_NAMES) + ['head', ])
        if mode not in {'w', 'a'}:
            raise ValueError(f'Unknown `mode` value {mode}. It must be "w" or "a".')
        if not os.path.isdir(path):  # path do not exist
            mode = 'w'
            os.makedirs(path, )
        elif len(os.listdir(path)) > 0:
            if mode == 'a':
                file_set = set(os.listdir(path))
                assert file_set == _STANDARD_FILE_LIST, 'Existing files do not match the saving data.'
            else:
                for ff in os.listdir(path):
                    if ff not in _STANDARD_FILE_LIST:
                        raise RuntimeError(f'The file `{ff}` have already exist in given path.')
        elif mode == 'a':  # exist the path, but it is empty
            mode = 'w'
        # check self
        if len(self) == 0:
            self.logger.error('ERROR: BatchStructures are empty now!')
            return None
        self._update_indices()
        self._check_id()
        #self._check_len()

        if self.Mode != 'A':
            self._list2np()
            has_conv = True
        else:
            has_conv = False
        _temp_attr_list = [
            self._Sample_ids_,
            self.Batch_indices_,
            self.Elements_batch_indices_,
            self.Cells_,
            self.Coords_type_,
            self.Coords_,
            self.Fixed_,
            self.Elements_,
            self.Numbers_,
            self.Energies_,
            self.Forces_,
            self.Labels_
        ]
        if mode == 'w':
            #############################################################################################################################
            # head file: A pickle file containing the tag which attr. is None, the batch number, the shape of each attr., and the type of attr.
            # head_info: Dict, {
            #   "which_None": Set[str], {_attr_names1, _attr_names2, ...},
            #   "n_batch": int, the number of samples,
            #   `_attr1`(str): Tuple[str, Tuple[int, ...]], (_dtype, _shape),
            #   `_attr2`(str): Tuple[str, Tuple[int, ...]], (_dtype, _shape),
            #   ...
            # }
            #############################################################################################################################
            head_info = {'which_None': set(), 'n_batch': len(self)}
            for i, filename in enumerate(self._ATTR_NAMES):
                if _temp_attr_list[i] is not None:
                    _dtype = _temp_attr_list[i].dtype.str
                    _shape = _temp_attr_list[i].shape
                else:
                    _dtype = '|O'
                    _shape = (1,)
                    head_info['which_None'].add(filename)
                _data = np.memmap(os.path.join(path, filename), dtype=_dtype, mode='w+', offset=0, shape=_shape)
                _data[:] = _temp_attr_list[i]
                _data.flush()
                del _data
                gc.collect()
                # head information: dtype shape
                head_info[filename] = (_dtype, _shape)
            with open(os.path.join(path, 'head'), 'wb') as head:
                head_info = pickle.dumps(head_info)
                head.write(head_info)
            del _temp_attr_list

        else:  # mode == 'a', append
            # Read head file
            with open(os.path.join(path, 'head'), 'rb') as head:
                head_info = pickle.load(head)
            # rearrange batch indices & element batch indices (change it which start at 0 into starting at the last indices in old indices)
            # batch indx.
            bat_indx = np.memmap(
                os.path.join(path, 'Batch_indices'),
                dtype=head_info['Batch_indices'][0],
                mode='r',
                shape=head_info['Batch_indices'][1]
            )
            last_indx = bat_indx[-1].copy()
            del bat_indx
            _temp_attr_list[1] = _temp_attr_list[1][1:] + last_indx
            # elem. indx.
            elem_indx = np.memmap(
                os.path.join(path, 'Elements_batch_indices'),
                dtype=head_info['Elements_batch_indices'][0],
                mode='r',
                shape=head_info['Elements_batch_indices'][1]
            )
            last_indx = elem_indx[-1].copy()
            _temp_attr_list[2] = _temp_attr_list[2][1:] + last_indx
            del elem_indx, last_indx
            # Append to file
            new_head_info = {'which_None': set()}
            for i, filename in enumerate(self._ATTR_NAMES):
                old_dtype = head_info[filename][0]
                old_shape = head_info[filename][1]
                if _temp_attr_list[i] is not None:
                    if filename in head_info['which_None']:
                        raise ValueError(f'The existing data {filename} is None, while appending data is not. The type of both should match.')
                    _dtype = _temp_attr_list[i].dtype.str
                    _shape = _temp_attr_list[i].shape
                else:
                    if filename not in head_info['which_None']:
                        raise ValueError(f'The existing data {filename} is not None, while appending data is None. The type of both should match.')
                    _dtype = '|O'
                    _shape = (1,)
                    new_head_info['which_None'].add(filename)
                # check the consistency of old & new data
                if old_shape[1:] != _shape[1:]:
                    raise RuntimeError(f'The existing file has a shape on non-1st dim {old_shape[1:]} '
                                       f'does not match the new data {_temp_attr_list[i].shape[1:]}')
                if old_dtype != _dtype:
                    raise RuntimeError(f'The data type of existing file is {old_dtype} '
                                       f'which is not match the new data type {_temp_attr_list[i].dtype}')

                # write data
                if _temp_attr_list[i] is not None:
                    new_shape = tuple([old_shape[0] + _temp_attr_list[i].shape[0]] + list(_temp_attr_list[i].shape[1:]))
                    _data = np.memmap(os.path.join(path, filename), dtype=old_dtype, mode='r+', shape=new_shape)
                    _data[old_shape[0]:] = _temp_attr_list[i]
                    _data.flush()
                    del _data
                    gc.collect()
                    new_head_info[filename] = (old_dtype, new_shape)
                else:
                    new_head_info[filename] = head_info[filename]

            # Update the head file
            new_head_info['n_batch'] = head_info['n_batch'] + len(self)
            with open(os.path.join(path, 'head'), 'wb') as head:
                head_info = pickle.dumps(new_head_info)
                head.write(head_info)
            del _temp_attr_list
        if has_conv:
            self._np2list(True)
            gc.collect()
        return None

    def load(self, path: str, data_slice: Tuple[int, int] | None = None, mode: Literal['w', 'a'] = 'w'):
        """
        Load data from saved files.
        Args:
            path: the file path.
            data_slice: The piece of data in files to be read. If None, all data in files would be read into memory.
            mode: the data load mode. 'W' for write or overwrite existing data from data in file; 'a' for appending data in file to existing data.

        Returns: None

        """
        # check vars
        if data_slice is not None:
            assert isinstance(data_slice[0], int) and isinstance(data_slice[1], int), (f"`data_slice` must be an integer tuple, "
                                                                                       f"but got ({type(data_slice[0])}, {type(data_slice[1])})")
            assert data_slice[0] <= data_slice[1], f"`data_slice[0]` cannot be greater than `data_slice[1]`, but got {data_slice}"
        # Read the head file
        with open(os.path.join(path, 'head'), 'rb') as head:
            head_info = pickle.load(head)
        # Read mmap file
        _data_dict = dict()
        if data_slice is None:
            for i, filename in enumerate(self._ATTR_NAMES):
                _dtype, _shape = head_info[filename]
                if filename not in head_info['which_None']:
                    _mmdata = np.memmap(os.path.join(path, filename), dtype=_dtype, mode='r', shape=_shape)
                    _data_dict[filename] = np.array(_mmdata)
                    del _mmdata
                    gc.collect()
                else:
                    _data_dict[filename] = None
        else:
            #if data_slice[1] > head_info['n_batch']: raise ValueError(f'`data_slice` out of range. Sample number: {head_info['n_batch']}')
            _dtype, _shape = head_info['Batch_indices']
            natom_slice = np.memmap(os.path.join(path, 'Batch_indices'), dtype=_dtype, mode='r', shape=_shape)
            natom_slice = np.array(natom_slice[data_slice[0]: data_slice[1] + 1])
            _dtype, _shape = head_info['Elements_batch_indices']
            elem_slice = np.memmap(os.path.join(path, 'Elements_batch_indices'), dtype=_dtype, mode='r', shape=_shape)
            elem_slice = np.array(elem_slice[data_slice[0]: data_slice[1] + 1])
            for filename in ('Coords', 'Fixed', 'Forces'):
                if filename not in head_info['which_None']:
                    _dtype, _shape = head_info[filename]
                    _mmdata = np.memmap(os.path.join(path, filename), dtype=_dtype, mode='r', shape=_shape)
                    _data_dict[filename] = np.array(_mmdata[natom_slice[0]:natom_slice[-1]])
                    del _mmdata
                    gc.collect()
                else:
                    _data_dict[filename] = None
            for filename in ('Elements', 'Numbers'):
                if filename not in head_info['which_None']:
                    _dtype, _shape = head_info[filename]
                    _mmdata = np.memmap(os.path.join(path, filename), dtype=_dtype, mode='r', shape=_shape)
                    _data_dict[filename] = np.array(_mmdata[elem_slice[0]:elem_slice[-1]])
                    del _mmdata
                    gc.collect()
                else:
                    _data_dict[filename] = None
            for filename in ('Sample_ids', 'Cells', 'Coords_type', 'Energies', 'Labels'):
                if filename not in head_info['which_None']:
                    _dtype, _shape = head_info[filename]
                    _mmdata = np.memmap(os.path.join(path, filename), dtype=_dtype, mode='r', shape=_shape)
                    _data_dict[filename] = np.array(_mmdata[data_slice[0]:data_slice[1]])
                    del _mmdata
                    gc.collect()
                else:
                    _data_dict[filename] = None
            _data_dict['Batch_indices'] = natom_slice - natom_slice[0]
            _data_dict['Elements_batch_indices'] = elem_slice - elem_slice[0]

        # load to memory
        if mode == 'w':
            self._Sample_ids_ = _data_dict['Sample_ids']
            self.Batch_indices_ = _data_dict['Batch_indices']
            self.Elements_batch_indices_ = _data_dict['Elements_batch_indices']
            self.Cells_ = _data_dict['Cells']
            self.Coords_type_ = _data_dict['Coords_type']
            self.Coords_ = _data_dict['Coords']
            self.Fixed_ = _data_dict['Fixed']
            self.Elements_ = _data_dict['Elements']
            self.Numbers_ = _data_dict['Numbers']
            self.Energies_ = _data_dict['Energies']
            self.Forces_ = _data_dict['Forces']
            self.Labels_ = _data_dict['Labels']
            if self.Mode == 'L':
                self._np2list(release_mem=True)
                self._clear_np()
        else:
            subcls = BatchStructures()
            subcls._Sample_ids_ = _data_dict['Sample_ids']
            subcls.Batch_indices_ = _data_dict['Batch_indices']
            subcls.Elements_batch_indices_ = _data_dict['Elements_batch_indices']
            subcls.Cells_ = _data_dict['Cells']
            subcls.Coords_type_ = _data_dict['Coords_type']
            subcls.Coords_ = _data_dict['Coords']
            subcls.Fixed_ = _data_dict['Fixed']
            subcls.Elements_ = _data_dict['Elements']
            subcls.Numbers_ = _data_dict['Numbers']
            subcls.Energies_ = _data_dict['Energies']
            subcls.Forces_ = _data_dict['Forces']
            subcls.Labels_ = _data_dict['Labels']
            subcls._np2list()
            if self.Mode == 'L':
                self.append(subcls)
                del subcls
            else:
                self._np2list(release_mem=True)
                self._clear_np()
                self.append(subcls)
                self._list2np(release_mem=True)
                del subcls
        gc.collect()
        self._update_indices()
        self._check_id()
        self._check_len()

    def write2text(
            self,
            output_path: str = './',
            indices: int | str | List[int] | Tuple[int, int] | None = None,
            file_format: Literal['POSCAR', 'cif', 'xyz', 'xyz_forces'] = 'POSCAR',
            file_name_list: str | Sequence[str] | None = None,
            n_core: int = -1
    ) -> None:
        """
        Write selected structures to text files in given format.
        Args:
            indices: the selection indices of `self`. If Tuple, structures between `indices[0]` and `indices[1]` will be selected.
            file_format: the format of written files.
                'POSCAR': vasp POSCAR format
                'cif': crystallographic information file
                'xyz': ext-xyz file that only contains atomic positions
                'xyz_forces': ext-xyz file that contains atomic positions and forces
            output_path: the directory of output file.
            file_name_list: the list of file names. If None, it would be set to `Sample_ids`.
            n_core: number of CPU cores to write in parallel. `-1` for all available CPU cores.

        Returns: None

        """
        # check vars
        if file_format not in {'POSCAR', 'cif', 'xyz', 'xyz_forces'}:
            raise ValueError(f'Invalid value of `file_format`: {file_format}.')
        if indices is None:
            sub_self = self
        elif isinstance(indices, Tuple):
            sub_self = self[indices[0]: indices[1]]
        else:
            sub_self = self[indices]
        if file_name_list is None:
            file_name_list = sub_self.Sample_ids
        self._check_id()
        self._check_len()
        if self.Mode == 'A':
            self._np2list()
            is_convert = True
        else:
            is_convert = False
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        try:
            file_name_list = [f'{_}' for _ in file_name_list]
            if file_format == 'POSCAR':
                WritePOSCARs(
                    sub_self.Cells,
                    sub_self.Coords,
                    sub_self.Elements,
                    sub_self.Numbers,
                    sub_self.Fixed,
                    output_path,
                    file_name_list,
                    sub_self.Sample_ids,
                    sub_self.Coords_type,
                    n_core
                )
            elif file_format == 'xyz':
                write_xyz(
                    sub_self.Elements,
                    sub_self.Coords,
                    sub_self.Cells,
                    sub_self.Energies,
                    sub_self.Numbers,
                    sub_self.Forces,
                    output_path,
                    file_name_list,
                    output_xyz_type='only_position_xyz',
                    n_core=n_core
                )
            elif file_format == 'xyz_forces':
                write_xyz(
                    sub_self.Elements,
                    sub_self.Coords,
                    sub_self.Cells,
                    sub_self.Energies,
                    sub_self.Numbers,
                    sub_self.Forces,
                    output_path,
                    file_name_list,
                    output_xyz_type='write_position_and_force',
                    n_core=n_core
                )
            elif file_format == 'cif':
                write_cif(
                    sub_self.Cells,
                    sub_self.Coords,
                    sub_self.Elements,
                    sub_self.Numbers,
                    output_path,
                    file_name_list,
                    sub_self.Sample_ids,
                    sub_self.Coords_type,
                    n_core
                )

            else:
                raise NotImplementedError
        except Exception as e:
            self.logger.error(f'An error occurred:\n\t{e}\ntraceback:\n\t{traceback.print_exc()}')
        finally:
            if is_convert:
                self._list2np(release_mem=True)

    def set_Labels(self, val: Sequence | Dict | np.ndarray) -> None:
        """
        Set or reset self.Labels from input `val`.
        Args:
            val: the input values of self.Labels to set.
            If Dict, val: {sample_id: energy, ...}; if Sequence, it must have the same order of self.Sample_ids.

        Returns: None

        """
        if len(val) != len(self):
            raise ValueError(f'labels (length{len(val)}) must have the same number of structures (length: {len(self)}).')
        if isinstance(val, Dict):
            try:
                self.Labels = [val[k_] for k_ in self._Sample_ids]
            except KeyError as e:
                raise KeyError(f'Some keys: {e} in Sample_ids do not in labels.')
        elif isinstance(val, Sequence):
            self.Labels = list(val)
        elif isinstance(val, np.ndarray):
            self.Labels = val.tolist()
        else:
            raise ValueError('Unknown type of input labels.')

    def set_Energies(self, val: Sequence | Dict):
        """
        Set or reset self.Energies from input `val`.
        Args:
            val: the input values to set self.Energies
                if Dict, val: {sample_id: energy, ...}; if Sequence, it must have the same order of self.Sample_ids.

        Returns: None

        """
        if len(val) != len(self):
            raise ValueError(f'labels (length: {len(val)}) must have the same number of structures (length: {len(self)}).')
        if isinstance(val, Dict):
            try:
                self.Energies = [val[k_] for k_ in self._Sample_ids]
            except KeyError as e:
                raise KeyError(f'Some keys: {e} in Sample_ids do not in val.')
        elif isinstance(val, Sequence):
            self.Energies = list(val)
        elif isinstance(val, np.ndarray):
            self.Energies = val.tolist()
        else:
            raise ValueError('Unknown type of input val.')

    def set_Forces(self, val: Sequence[np.ndarray] | Dict[str, np.ndarray]):
        """
        Set or reset self.Forces from `val`.
        Args:
            val: the input values to set self.Forces
                if Dict, val: {sample_id: forces, ...}; if Sequence, it must have the same order of self.Sample_ids.

        Returns:

        """
        if len(val) != len(self):
            raise ValueError(f'labels (length{len(val)}) must have the same number of structures (length: {len(self)}).')
        if isinstance(val, Dict):
            try:
                self.Forces = [val[k_] for k_ in self._Sample_ids]
            except KeyError as e:
                raise KeyError(f'Some keys: {e} in Sample_ids do not in val.')
        elif isinstance(val, Sequence):
            self.Forces = list(val)
        else:
            raise ValueError('Unknown type of input val.')
        # check
        for i, forces in enumerate(self.Forces):
            if forces.shape != self.Coords[i].shape:
                self.Forces = None
                raise ValueError('Some forces data do not have the same shape of corresponding coordinates.')

    def generate_dist_mat(self, supercell_indices=np.zeros((1, 3))):
        r"""
        Parameters:
            supercell_indices: NDArray[n_supercell, 3], the indices of supercells that calculate distance.

        self.Dist_mat Format: List[array[(n_prim_cells, n_atom, n_atom)]]
        """
        self.Dist_mat = list()
        for i, atomic_coordinates in enumerate(self.Coords):
            cell_vectors: np.ndarray = self.Cells[i]
            # calculate cross-cell dist.; cell_diff = supercell_indices @ cell_vec; <<<
            # shape: (n_prim_cells, 1, 3)@(1, 3, 3) -> (n_prim_cells, 1, 3)
            cell_diff = (supercell_indices[:, None, :]) @ (cell_vectors[None, :, :])
            # convert direct to cartesian
            if self.Coords_type[i] == 'D':
                atomic_coordinates = atomic_coordinates @ cell_vectors
            # calculate the atom coords across cells (x_j + R_k) <<<
            # shape: (1, n_atom, 3) + (n_prim_cells, 1, 3)
            #     -> (n_prim_cells, n_atom, 3)
            coord_cross = atomic_coordinates[None, :, :] + cell_diff
            n_prim_cells, n_atom, _ = coord_cross.shape
            # calculate actual dist. vec.; dist_vec = r_ijk = ||x_i - (x_j + R_k)|| where R_k is the cell vector; <<<
            # shape: (1, n_atom, 1, 3) - (n_prim_cells, 1, n_atom, 3)
            #        -> (n_prim_cells, n_atom, n_atom, 3)  # coord_diff
            #   -norm-> (n_prim_cells, n_atom, n_atom)    # euclid distance
            distance = atomic_coordinates[None, :, None, :] - coord_cross[:, None, :, :]
            distance = np.linalg.norm(distance, ord=2, axis=-1)
            self.Dist_mat.append(distance)

    def generate_atom_list(self, force_update: bool = False) -> None:
        r"""
        Generate an attribute self.Atom_list (List[str]) which is the atom lists in order of the atomic coordinates sequence.

        Parameters:
            force_update: bool. if False, Atom_list would not be updated when it's not None.
        """
        if (self.Atom_list is None) or force_update:
            self.Atom_list = list()
            for i, list_ in enumerate(self.Elements):
                self.Atom_list.append([])
                for j, elem in enumerate(list_):
                    self.Atom_list[i] += [elem, ] * self.Numbers[i][j]

        elif isinstance(self.Atom_list, Sequence):
            warnings.warn('Atom_list already existed, so it would not be updated.')
        else:
            raise RuntimeError(f'An Unknown Atom_list occurred. Atom_list == {self.Atom_list}')

    def generate_atomic_number_list(self, force_update: bool = False) -> None:
        r"""
        Generate attribute self.Atomic_number_list (List[int]) which is the atomic number lists in order of the sequence of atom coordinates.

        Parameters:
            force_update: bool. if False, Atomic_number_list would not be updated when it's not None.
        """
        if (self.Atomic_number_list is None) or force_update:
            if self.Atom_list is None:
                self.Atom_list = list()
                for i, list_ in enumerate(self.Elements):
                    self.Atom_list.append([])
                    for j, elem in enumerate(list_):
                        self.Atom_list[i] += [elem, ] * self.Numbers[i][j]

            self.Atomic_number_list = list()
            for i, list_ in enumerate(self.Atom_list):
                self.Atomic_number_list.append([self._Z_dict[symb] for symb in list_])
        elif isinstance(self.Atomic_number_list, Sequence):
            warnings.warn('Atomic_number_list already existed, so it would not be updated.')
        else:
            raise RuntimeError(f'An Unknown Atomic_number_list occurred. Atomic_number_list == {self.Atomic_number_list}')

    def cartesian2direct(self, ):
        """ Convert Cartesian coordinates to Direct coordinates. Only work in 'L' Mode. """
        assert self.Mode == 'L'
        for i, cootype in enumerate(self.Coords_type):
            if cootype == 'C':
                cell = self.Cells[i]
                self.Coords[i] @= np.linalg.inv(cell)
                self.Coords_type[i] = 'D'

    def direct2cartesian(self, ):
        """ Convert Direct coordinates to Cartesian coordinates. Only work in 'L' Mode. """
        assert self.Mode == 'L'
        for i, cootype in enumerate(self.Coords_type):
            if cootype == 'D':
                cell = self.Cells[i]
                self.Coords[i] @= cell
                self.Coords_type[i] = 'C'

    def sort_ids(self, ref_attr: str = 'Sample_ids', reverse: bool = False):
        """
        Sort structures by their Sample_ids order.
        Args:
            ref_attr: str, the attribute to be the reference for structure sort.
            reverse: bool. if True, the sort order is reversed, i.e., the descending order.
        Returns: None

        """
        if self.Mode == 'A':
            raise NotImplementedError(f"Sort for Array mode is not implemented yet.")
        lst = getattr(self, ref_attr, None)
        if lst is None:
            raise ValueError(f"Attribute '{ref_attr}' not found.")
        _sort_idx = sorted(range(len(lst)), key=lambda i: lst[i], reverse=reverse)

        for attr_name in self.LIST_ATTR_NAME:
            orig_attr = getattr(self, attr_name)
            if orig_attr is None:
                continue
            setattr(self, attr_name, [orig_attr[_] for _ in _sort_idx])


    def standardize(self, ):
        """
        Standardize the atomic coordinates, i.e., translation the centre of all coordinates to their lattice centre.
        Only work in 'L' Mode.

        Returns: None
        """
        # TODO, standardize coo
        raise NotImplementedError('Standardize not implemented yet.')
        _CELL_CENTRE = np.asarray([0.5, 0.5, 0.5])
        for i, coo in enumerate(self.Coords):
            if self.Coords_type[i] == 'C':  # cartesian
                # shape center
                c1 = np.mean(coo, axis=0, keepdims=True)
                #   check out of cell
                fc1 = c1 @ np.linalg.inv(self.Cells[i])
                if np.any((fc1 < 0.) | (fc1 > 1.)):
                    self.logger.warning(f'Center out of cell in {self.Sample_ids[i]}')
                    fc1 -= np.floor(fc1)
                    self.Coords[i] = fc1 @ self.Cells[i]
                lc1 = _CELL_CENTRE @ self.Cells[i]
                self.Coords[i] -= c1 - lc1  # (n_atom, 3) - (1, 3)
            elif self.Coords_type[i] == 'D':  # frac
                # shape center
                fc1 = np.mean(coo, axis=0, keepdims=True)
                if np.any((fc1 < 0.) | (fc1 > 1.)):
                    self.logger.warning(f'Center out of cell in {self.Sample_ids[i]}')
                    fc1 -= np.floor(fc1)
                self.Coords[i] -= fc1 - _CELL_CENTRE  # (n_atom, 3) - (1, 3)

    def shuffle(self, seed:int = None):
        """
        Shuffle the samples with given random seed.
        Args:
            seed: random seed.

        Returns: None

        """
        # generate indices
        indx = list(range(len(self)))
        random.seed(seed)
        random.shuffle(indx)
        self.rearrange(indx)

    def rearrange(self, indices: List[int]):
        """
        Rearrange the order of samples by given indices.
        Args:
            indices: indices of sample orders.

        Returns: None

        """
        indx = list(indices)
        if self.Energies is not None:
            self.Energies = [self.Energies[_] for _ in indx]
        if self.Forces is not None:
            self.Forces = [self.Forces[_] for _ in indx]
        if self.Labels is not None:
            self.Labels = [self.Labels[_] for _ in indx]

        self._Sample_ids = [self._Sample_ids[_] for _ in indx]  # mp_ids
        self.Cells = [self.Cells[_] for _ in indx]  # Lattice parameters
        self.Elements = [self.Elements[_] for _ in indx]  # Element symbols
        self.Numbers = [self.Numbers[_] for _ in indx]  # Element numbers
        self.Coords_type = [self.Coords_type[_] for _ in indx]  # coordinates type
        self.Coords = [self.Coords[_] for _ in indx]  # Atom coordinates
        self.Fixed = [self.Fixed[_] for _ in indx]  # fixed masks
        self._check_id()
        self._check_len()

    def fix_atoms_by_height(self, height:float, coord_type: Literal['C', 'D']='C', direction: Literal['x', 'y', 'z'] = 'z'):
        """
        Fix atoms by height at the given direction.
        All atoms with coordinates lower than the given height will be fixed.
        Returns: None

        """
        _DIRECT_DICT = {'x': slice(None, 1), 'y': slice(1, 2), 'z': slice(-1, None)}
        if coord_type == 'C':
            self.direct2cartesian()
        elif coord_type == 'D':
            self.cartesian2direct()
        else:
            self.logger.error(f'ERROR: Unknown coord_type `{coord_type}`.')
            return

        if direction not in _DIRECT_DICT:
            self.logger.error(f'ERROR: Unknown direction `{direction}`.')
            return

        i = 0
        try:
            for i, m in enumerate(self.Coords):
                mask = np.where(self.Coords[i][:, _DIRECT_DICT[direction]] <= height, 0, 1)
                mask = np.repeat(mask, 3, axis=-1)
                self.Fixed[i] = mask.astype(np.int8)
        except Exception as e:
            self.logger.error(f'ERROR: Fixation failed in the {i}-th structure:')
            self.logger.error(e)
            self.logger.error(traceback.format_exc())

    def revise(
            self,
            index: int|str,
            rev_Sample_ids: None | str = None,
            rev_Cells: None | np.ndarray = None,
            rev_Elements: None | List[str] = None,
            rev_Numbers: None | List[int] = None,
            rev_Coords_type: None | Literal['C', 'D'] = None,
            rev_Coords: None | np.ndarray = None,
            rev_Fixed: None | np.ndarray = None,
            rev_Energies: None | float = None,
            rev_Forces: None | np.ndarray = None,
            rev_Labels: Any | None = None,
    ) -> None:
        """
        Revising data of given `index`. None is for not changing.
        Args:
            index: the index of structure to revise.
            rev_Sample_ids: Sample ids of structure to revise.
            rev_Cells: Cell ids of structure to revise.
            rev_Elements: Elements ids of structure to revise.
            rev_Numbers: Numbers of structure to revise.
            rev_Coords_type: Coords of structure to revise.
            rev_Coords: Coords of structure to revise.
            rev_Fixed: Fixation mask of atoms to revise.
            rev_Energies: Energies of structure to revise.
            rev_Forces: Forces of structure to revise.
            rev_Labels: Labels of structure to revise.

        """
        assert self.Mode == 'L', f'Mode "A" does not support yet.'
        if isinstance(index, str):
            if self._indices is None:
                self._update_indices()
            _indx:int = self._indices[index]
        elif isinstance(index, int):
            _indx:int = index
        else:
            raise TypeError(f'Expected `index` to be int or str, but got {type(index)}.')
        # check types
        if rev_Sample_ids is not None:
            assert isinstance(rev_Sample_ids, type(self._Sample_ids[0]))
        else:
            rev_Sample_ids = self._Sample_ids[_indx]
        if rev_Cells is not None:
            assert isinstance(rev_Cells, np.ndarray)
            assert rev_Cells.shape == (3, 3)
        else:
            rev_Cells = self.Cells[_indx]
        if rev_Elements is not None:
            assert isinstance(rev_Elements, List)
        else:
            rev_Elements = self.Elements[_indx]
        if rev_Numbers is not None:
            assert isinstance(rev_Numbers, List)
        else:
            rev_Numbers = self.Numbers[_indx]
        if rev_Coords_type is not None:
            assert rev_Coords_type in {'C', 'D'}, f'rev_Coords_type must be either "C" or "D", but got {rev_Coords_type}.'
        else:
            rev_Coords_type = self.Coords_type[_indx]
        if rev_Coords is not None:
            assert isinstance(rev_Coords, np.ndarray)
        else:
            rev_Coords = self.Coords[_indx]
        if rev_Fixed is not None:
            assert isinstance(rev_Fixed, np.ndarray)
            assert np.all(rev_Fixed - 2 < 0) and np.all(rev_Fixed >= 0), '`Fixed` must be either 0 or 1'
        else:
            rev_Fixed = self.Fixed[_indx]
        if rev_Energies is None:
            if self.Energies is not None: rev_Energies = self.Energies[_indx]
        elif self.Energies is None:
            warnings.warn('`rev_Energies` is given while `self.Energies` is None. Hence `rev_Energies` will be skipped.', RuntimeWarning)
        if rev_Forces is None:
            if self.Forces is not None:
                rev_Forces = self.Forces[_indx]
                force_shape = rev_Forces.shape
            else:
                force_shape = rev_Coords.shape
        else:
            assert isinstance(rev_Forces, np.ndarray)
            if self.Forces is None:
                warnings.warn('`rev_Forces` is given while `self.Forces` is None. Hence `rev_Force` will be skipped.', RuntimeWarning)
            force_shape = rev_Forces.shape
        if rev_Labels is None:
            if not self.Labels is None: rev_Labels = self.Labels[_indx]
        elif self.Labels is None:
            warnings.warn('`rev_Labels` is given while `self.Labels` is None. Hence `rev_Labels` will be skipped.', RuntimeWarning)
        # check numbers
        assert (rev_Coords.shape == rev_Fixed.shape == force_shape), f'Coords, Fixed, and Forces (optional) have different shapes.'
        assert len(rev_Elements) == len(rev_Numbers), f'Elements & Numbers have different shapes.'
        assert sum(rev_Numbers) == len(rev_Coords), f'Number of atoms is mismatch in Numbers and Coord.'
        # load
        self._Sample_ids[_indx] = rev_Sample_ids
        self.Cells[_indx] = rev_Cells.astype(np.float32)
        self.Elements[_indx] = rev_Elements
        self.Numbers[_indx] = rev_Numbers
        self.Coords_type[_indx] = rev_Coords_type
        self.Coords[_indx] = rev_Coords.astype(np.float32)
        self.Fixed[_indx] = rev_Fixed.astype(np.int8)
        if self.Energies is not None:
            self.Energies[_indx] = float(rev_Energies)
        if self.Forces is not None:
            self.Forces[_indx] = rev_Forces.astype(np.float32)
        if self.Labels is not None:
            self.Labels[_indx] = rev_Labels
        # postprocess
        if self.Atom_list is not None:
            self.generate_atom_list(True)
        if self.Atomic_number_list is not None:
            self.generate_atomic_number_list(True)
        if self.Dist_mat is not None:
            warnings.warn(f'Due to the data change, old Dist_mat might be incorrect. Hence Dist_mat is reset to None.', RuntimeWarning)
            self.Dist_mat = None

    def append_from_lists(
            self,
            add_Sample_ids: List,
            add_Cells: List[np.ndarray],
            add_Elements: List[List[str]],
            add_Numbers: List[List[int]],
            add_Coords_type: List[Literal['C', 'D']],
            add_Coords: List[np.ndarray],
            add_Fixed: List[np.ndarray],
            add_Energies: None | List[float] = None,
            add_Forces: None | List[np.ndarray] = None,
            add_Labels: List[Any] | None = None,
    ) -> None:
        """
        Appending data from given Lists of ids, cells, elements, numbers, and coords.
        Args:
            add_Sample_ids: the data in List to be appended.
            add_Cells: the data in List to be appended.
            add_Elements: the data in List to be appended.
            add_Numbers: the data in List to be appended.
            add_Coords_type: the data in List to be appended.
            add_Coords: the data in List to be appended.
            add_Fixed: the data in List to be appended.
            add_Energies: the data in List to be appended.
            add_Forces: the data in List to be appended.
            add_Labels: the data in List to be appended.

        """
        assert self.Mode == 'L', f'Mode "A" does not support yet.'
        # check types
        if not isinstance(add_Sample_ids, list):
            raise TypeError(f'`add_Sample_ids` must be a list, but got {type(add_Sample_ids)}.')
        n_samp = len(add_Sample_ids)

        if not isinstance(add_Cells, list):
            raise TypeError(f'`add_Cells` must be a list, but got {type(add_Cells)}.')
        else:
            _chk = [type(_) for _ in add_Cells if not isinstance(_, np.ndarray)]
            if len(_chk) > 0:
                raise TypeError(f'`add_Cells` must contain only np.ndarray elements, but got {_chk}.')
        n_cells = len(add_Cells)

        if not isinstance(add_Elements, list):
            raise TypeError(f'`add_Elements` must be a list, but got {type(add_Elements)}.')
        else:
            _chk = [_ for _ in add_Elements if (not isinstance(_, List)) or ({type(__) for __ in _} != {str})]
            if len(_chk) > 0:
                raise TypeError(f'`add_Elements` must contain only List[str] elements, but got {_chk}.')
        n_elems = len(add_Elements)

        if not isinstance(add_Numbers, list):
            raise TypeError(f'`add_Numbers` must be a list, but got {type(add_Numbers)}.')
        else:
            _chk = [_ for _ in add_Numbers if (not isinstance(_, List)) or ({type(__) for __ in _} != {int})]
            if len(_chk) > 0:
                raise TypeError(f'`add_Numbers` must contain only List[int] elements, but got {_chk}.')
        n_nums = len(add_Numbers)

        if not isinstance(add_Coords_type, list):
            raise TypeError(f'`add_Coords_type` must be a list, but got {type(add_Coords_type)}.')
        else:
            _chk = [_ for _ in add_Coords_type if _ not in {'C', 'D'}]
            if len(_chk) > 0:
                raise ValueError(f'`add_Coords_type` must contain only "C" or "D" as elements, but got {_chk}.')
        n_coords_type = len(add_Coords_type)

        if not isinstance(add_Coords, list):
            raise TypeError(f'`add_Coords` must be a list, but got {type(add_Coords)}.')
        else:
            _chk = [_ for _ in add_Coords if not isinstance(_, np.ndarray)]
            if len(_chk) > 0:
                raise TypeError(f'`add_Coords` must contain only np.ndarray elements, but got {_chk}.')
        n_coords = len(add_Coords)

        if not isinstance(add_Fixed, list):
            raise TypeError(f'`add_Fixed` must be a list, but got {type(add_Fixed)}.')
        else:
            _chk = [_ for _ in add_Fixed if not isinstance(_, np.ndarray)]
            if len(_chk) > 0:
                raise TypeError(f'`add_Fixed` must contain only np.ndarray elements, but got {_chk}.')
        n_fixed = len(add_Fixed)

        if add_Energies is None:
            if self.Energies is not None:
                raise ValueError(f"Other structures have energy but here given `add_Energies` is None, which is not allowed.")
            n_energies = n_samp
        elif not isinstance(add_Energies, List):
            raise TypeError(f'Expected `add_Energies` of list, but got {type(add_Energies)}.')
        else:
            _chk = [_ for _ in add_Energies if not isinstance(_, float)]
            if len(_chk) > 0:
                raise TypeError(f'`add_Energies` must contain only float elements, but got {type(_chk)}.')
            n_energies = len(add_Energies)
            if self.Energies is None:  # adding data into None
                if len(self) == 0:
                    self.Energies = list()
                else:
                    raise ValueError(f'Other structures do not have energy, but here some energies is adding.')

        if add_Forces is None:
            if self.Forces is not None:
                raise ValueError(f"Other structures have forces but here given `add_Forces` is None, which is not allowed.")
            n_forces = n_samp
        elif not isinstance(add_Forces, List):
            raise TypeError(f'Expected `add_Forces` of list, but got {type(add_Forces)}.')
        else:
            _chk = [_ for _ in add_Forces if not isinstance(_, np.ndarray)]
            if len(_chk) > 0:
                raise TypeError(f'`add_Forces` must contain only np.ndarray elements, but got {type(_chk)}.')
            n_forces = len(add_Forces)
            if self.Forces is None:  # adding data into None
                if len(self) == 0:
                    self.Forces = list()
                else:
                    raise ValueError(f'Other structures do not have forces, but here some forces is adding.')

        if add_Labels is None:
            if self.Labels is not None:
                raise ValueError(f"Other structures have labels but here given `add_Labels` is None, which is not allowed.")
            n_labels = n_samp
        elif not isinstance(add_Labels, List):
            raise TypeError(f'Expected `add_Labels` of list, but got {type(add_Labels)}.')
        else:
            n_labels = len(add_Labels)
            if self.Labels is None:  # adding data into None
                if len(self) == 0:
                    self.Labels = list()
                else:
                    raise ValueError(f'Other structures do not have forces, but here some forces is adding.')

        samp_number_set = {n_samp, n_cells, n_elems, n_nums, n_coords_type, n_coords, n_fixed, n_energies, n_forces, n_labels}
        if len(samp_number_set) > 1:
            # i.e., some sample numbers are not equal
            raise ValueError(
                f'Sample numbers must be the same, '
                f'but [n_samp, n_cells, n_elems, n_nums, n_coords_type, n_coords, n_fixed, n_energies, n_forces, n_labels] '
                f'have sample numbers of {n_samp, n_cells, n_elems, n_nums, n_coords_type, n_coords, n_fixed, n_energies, n_forces, n_labels}.'
            )

        # load
        _new_attr = [
            add_Sample_ids,    # self.Sample_ids,
            add_Cells,    # self.Cells,
            add_Elements,    # self.Elements,
            add_Numbers,    # self.Numbers,
            add_Coords_type,    # self.Coords_type,
            add_Coords,    # self.Coords,
            add_Fixed,    # self.Fixed,
            add_Energies,    # self.Energies,
            add_Forces,    # self.Forces,
            add_Labels,    # self.Labels
        ]
        _new_attr_type = [
            None,
            np.float32,
            None,
            None,
            None,
            np.float32,
            np.int8,
            None,
            np.float32,
            None,
        ]
        for _i, _attr in enumerate([
            self.Sample_ids,
            self.Cells,
            self.Elements,
            self.Numbers,
            self.Coords_type,
            self.Coords,
            self.Fixed,
            self.Energies,
            self.Forces,
            self.Labels
        ]):
            if (_new_attr_type[_i] is not None) and (_new_attr[_i] is not None):
                _at = [_.astype(_new_attr_type[_i]) for _ in _new_attr[_i]]
            else:
                _at = _new_attr[_i]

            if _attr is not None:
                _attr.extend(_at)
        # postprocess
        if self.Atom_list is not None:
            self.generate_atom_list(True)
        if self.Atomic_number_list is not None:
            self.generate_atomic_number_list(True)
        if self.Dist_mat is not None:
            warnings.warn(f'Due to the data change, old Dist_mat might be incorrect. Hence Dist_mat is reset to None.', RuntimeWarning)
            self.Dist_mat = None
        # check
        self._check_id()
        self._check_len()

    def append_from_array(
            self,
            add_atom_batch_sizes: np.ndarray,
            add_element_batch_sizes: np.ndarray,
            add_Sample_ids: np.ndarray,
            add_Cells: np.ndarray,
            add_Elements: np.ndarray,
            add_Numbers: np.ndarray,
            add_Coords_type: np.ndarray,
            add_Coords: np.ndarray,
            add_Fixed: np.ndarray,
            add_Energies: None | np.ndarray = None,
            add_Forces: None | np.ndarray = None,
            add_Labels: np.ndarray | None = None,
            is_check_length: bool = False,
    ):
        """
        Appending datta from np.ndarray, only work in self.Mode = 'A'.
        Args:
            add_atom_batch_sizes: array of atom numbers in each structure.
            add_element_batch_sizes: array of element numbers in each structure.
            add_Sample_ids: array of element numbers in each structure.
            add_Cells: array of element numbers in each structure.
            add_Elements: array of element numbers in each structure.
            add_Numbers: array of element numbers in each structure.
            add_Coords_type: array of element numbers in each structure.
            add_Coords: array of element numbers in each structure.
            add_Fixed: array of element numbers in each structure.
            add_Energies: array of element numbers in each structure.
            add_Forces: array of element numbers in each structure.
            add_Labels: array of element numbers in each structure.
            is_check_length: bool, whether to check the consistency of sample numbers of all `add_*` vars. Very expensive!

        Returns:

        """
        # check
        if self.Mode != 'A':
            raise ValueError(f'`append_from_array` only work in `A` mode.')
        if add_atom_batch_sizes.ndim != 1:
            raise ValueError(f'Expected `add_atom_batch_sizes` of 1 dimension, but got {add_atom_batch_sizes.ndim}.')
        if add_element_batch_sizes.ndim != 1:
            raise ValueError(f'Expected `add_element_batch_sizes` of 1 dimension, but got {add_element_batch_sizes.shape}.')

        _new_atom_ptr = np.empty(len(add_atom_batch_sizes) + 1, dtype=int)
        _new_atom_ptr[0] = 0
        np.cumsum(add_atom_batch_sizes, out=_new_atom_ptr[1:])
        _new_element_ptr = np.empty(len(add_element_batch_sizes) + 1, dtype=int)
        _new_element_ptr[0] = 0
        np.cumsum(add_element_batch_sizes, out=_new_element_ptr[1:])
        # standarize shape
        add_Sample_ids = add_Sample_ids.astype('<U128')
        add_Cells = add_Cells.astype('<f4')
        add_Elements = add_Elements.astype('<U3')
        add_Numbers = add_Numbers.astype('<i4')
        add_Coords = add_Coords.astype('<f4')
        add_Coords_type = add_Coords_type.astype('<U2')
        add_Fixed = add_Fixed.astype('|i1')
        if add_Energies is not None: add_Energies = add_Energies.astype('<f4')
        if add_Forces  is not None: add_Forces = add_Forces.astype('<f4')
        if add_Labels  is not None: add_Labels = add_Labels.astype('|O')
        # write
        if self._Sample_ids_ is None:  # a new container
            self.Batch_indices_ = _new_atom_ptr
            self.Elements_batch_indices_ = _new_element_ptr
            self._Sample_ids_ = add_Sample_ids
            self.Cells_ = add_Cells
            self.Elements_ = add_Elements
            self.Numbers_ = add_Numbers
            self.Coords_ = add_Coords
            self.Coords_type_ = add_Coords_type
            self.Fixed_ = add_Fixed
            self.Energies_ = add_Energies
            self.Forces_ = add_Forces
            self.Labels_ = add_Labels
        else:
            _new_atom_ptr += self.Batch_indices_[-1]
            _new_element_ptr += self.Elements_batch_indices_[-1]
            self.Batch_indices_ = np.append(self.Batch_indices_, _new_atom_ptr[1:])  # drop the 1st element '0' in `_new_atom_ptr`
            self.Elements_batch_indices_ = np.append(self.Elements_batch_indices_, _new_element_ptr[1:])
            self._Sample_ids_ = np.append(self._Sample_ids_, add_Sample_ids)
            self.Cells_ = np.append(self.Cells_, add_Cells, axis=0)
            self.Elements_ = np.append(self.Elements_, add_Elements)
            self.Numbers_ = np.append(self.Numbers_, add_Numbers)
            self.Coords_type_ = np.append(self.Coords_type_, add_Coords_type)
            self.Coords_ = np.append(self.Coords_, add_Coords, axis=0)
            self.Fixed_ = np.append(self.Fixed_, add_Fixed, axis=0)
            if self.Energies_ is not None:
                self.Energies_ = np.append(self.Energies_, add_Energies)
            elif add_Energies is not None:
                warnings.warn(f'`add_Energies` will be dropped because `self.Energies` is None.')
            if self.Forces_ is not None:
                self.Forces_ = np.append(self.Forces_, add_Forces)
            elif add_Forces is not None:
                warnings.warn(f'`add_Forces` will be dropped because `self.Forces` is None.')
            if self.Labels_ is not None:
                self.Labels_ = np.append(self.Labels_, add_Labels)
            elif add_Labels is not None:
                warnings.warn(f'`add_Labels` will be dropped because `self.Labels` is None.')

        self._check_id()
        if is_check_length: self._check_len()

    def sort_split_by_natoms(
            self,
            labels: Optional[Dict | List] = None,
            split_ratio: float = 0.2,
            n_core: int = 1,
            verbose: int = 0
    ) -> (Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]):
        """
        Rearrange samples with different atom number into List[np.array] that list of NDArrays with the same atom number.
        Only work in 'L' Mode.

        Parameters:
            labels: Dict[id, label]|List[label]|None, the labels of samples. If labels is a List, it must have the same order as self.Sample_ids. If None, labels would be set to self.Labels.
            split_ratio: float between 0 and 1, the ratio of validation set.
            n_core:
            verbose: control the verboseness of output

        Returns:
            training_batches, val_batches: List_1[List_2[NDArray]]. List_1 is of diff. n_atom cells, List_2 is of (Cell, Atomic_number_list, Atom_coords)
            training_labels, val_labels: List[NDArray]. List is of diff. n_atom cells, NDArray is label.
            training_args, val_args: List[NDArray]. List is of diff. n_atom cells, NDArray is the indices of samples in self.Samp_id.
        """
        t_st = time.perf_counter()
        if self.Atomic_number_list is None:
            self.generate_atomic_number_list()
        if labels is None:
            if self.Labels is None:
                raise ValueError('Input labels and self.Labels are both None. Please assign values to at least one variable.')
            else:
                labels = {_id: self.Labels[i] for i, _id in enumerate(self._Sample_ids)}
        elif isinstance(labels, Sequence):
            labels = {_id: labels[i] for i, _id in enumerate(self._Sample_ids)}
        elif not isinstance(labels, Dict):
            raise TypeError('Invalid type of labels')

        n_samp = len(self.Numbers)
        if (split_ratio < 0.) or (split_ratio > 1.):
            raise ValueError(f'Invalid value of split_ratio: {split_ratio}')
        VAL_SPLIT_RATIO = split_ratio
        count = np.empty(n_samp, dtype=np.int32)
        for i, list_ in enumerate(self.Numbers):  # total atom number list
            count[i] = np.sum(np.array(list_, dtype=np.int32))
        batch_args = np.argsort(count)
        temp_ = 0
        batch_dict = dict()  # the dict with format: {atom number: indices of sample}
        for arg in batch_args:
            if count[arg] > temp_:
                temp_ = count[arg]
                batch_dict[count[arg]] = [arg, ]
            elif count[arg] == temp_ and (count[arg] in batch_dict.keys()):
                batch_dict[count[arg]].append(arg)
            elif count[arg] == temp_ and (count[arg] not in batch_dict.keys()):
                warnings.warn(f'ERROR, unexpected situation has occurred in {arg}th sample.')
            else:
                warnings.warn(f'ERROR2, unexpected situation has occurred in {arg}th sample.')

        ''' Split Batches '''
        tot_batches = list()  # total feature batches <<<
        # list of NDArrays. The list shape is (batches_with_diff_n_atom, 3), 
        # and '3' contains 3 NDArrays that are [cell(n_samp, 3, 3), atomic_number(n_samp, n_atom), coord(n_samp, n_atom, 3)]
        label_batches = list()  # total labels <<<
        # list of NDArrays. The list of energies NDArray with shape (n_samp, 1), wherein '1' contains the energy values.
        samp_args = list()  # sample indices

        # closure to parallel
        def _rearrange_single(n_atom, sample_args_, Cells, Atomic_number_list, Coords, labels_, Sample_ids) -> Tuple[np.ndarray, ...]:
            Cell_batch = np.empty((len(sample_args_), 3, 3), dtype=np.float32)
            Atomic_number_batch = np.empty((len(sample_args_), n_atom), dtype=np.int32, )
            Coord_batch = np.empty((len(sample_args_), n_atom, 3), dtype=np.float32)
            Energies_batch = np.empty(len(sample_args_), dtype=np.float32)
            Samp_args = np.empty((len(sample_args_)), dtype=np.int32, )
            for j, arg_ in enumerate(sample_args_):
                Cell_batch[j] = np.array(Cells[arg_], dtype=np.float32)
                Atomic_number_batch[j] = np.array(Atomic_number_list[arg_], dtype=np.int32)  # type: ignore
                Coord_batch[j] = np.array(Coords[arg_], dtype=np.float32)
                Energies_batch[j] = np.array(labels_[Sample_ids[arg_]])  # grep DFT energies by mp-ids in energy dict; files_list[arg_][:-5] is mp-id.
                Samp_args[j] = np.array(arg_)
            return Energies_batch, Cell_batch, Atomic_number_batch, Coord_batch, Samp_args

        if n_core != 1:
            _parallel = jb.Parallel(n_jobs=n_core, verbose=verbose, backend='loky')
            _dat = _parallel(
                jb.delayed(_rearrange_single)(n_atom, sample_args_, self.Cells, self.Atomic_number_list, self.Coords, labels, self._Sample_ids) for
                n_atom, sample_args_ in
                batch_dict.items())
            if _dat is None: raise RuntimeError('Occurred None data.')
            for temp in _dat:
                label_batches.append(temp[0], )
                tot_batches.append([temp[1], temp[2], temp[3]])
                samp_args.append(temp[4])
        else:
            if verbose > 0: self.logger.info('Rearrange sequentially...')
            for n_atom, sample_args_ in batch_dict.items():
                temp = _rearrange_single(n_atom, sample_args_, self.Cells, self.Atomic_number_list, self.Coords, labels, self._Sample_ids)
                label_batches.append(temp[0], )
                tot_batches.append([temp[1], temp[2], temp[3]])
                samp_args.append(temp[4])

        if verbose > 0: self.logger.info('Done.\nSplitting...')
        # Split training and validation set
        k__ = 0
        training_batches: List[
            List[np.ndarray]] = list()  # List_1[List_2[NDArray]]. List_1 is of diff. n_atom cells, List_2 is of (Cell, Atom_list, Atom_coord)
        val_batches: List[List[np.ndarray]] = list()
        training_labels: List[np.ndarray] = list()  # List[NDArray]. List is of diff. n_atom cells, NDArray is Energy
        val_labels: List[np.ndarray] = list()
        training_args: List[np.ndarray] = list()  # List[NDArray]. List is of diff. n_atom cells, NDArray is the indices of samples in f.Samp_id.
        val_args: List[np.ndarray] = list()
        if VAL_SPLIT_RATIO > 1e-8:
            for i, batches_with_diff_n_atom in enumerate(tot_batches):
                n_samp_in_batch_ = len(label_batches[i])
                val_samples_temp = (np.ceil(n_samp_in_batch_ * VAL_SPLIT_RATIO)).astype(int)
                # To avoid empty samples
                if val_samples_temp == n_samp_in_batch_:
                    _ = batches_with_diff_n_atom[1].shape
                    k__ += 1
                    self.logger.info(
                        f'WARNING: Too few samples occurred. The number of samples with {_[1]} atoms is only {n_samp_in_batch_}, '
                        f'so it would be put in neither training set nor validation set. Total {k__} warnings.')
                    continue
                # split
                training_batches.append([batches_with_diff_n_atom[j][:-val_samples_temp] for j in range(3)])
                val_batches.append([batches_with_diff_n_atom[j][-val_samples_temp:] for j in range(3)])
                training_labels.append(label_batches[i][:-val_samples_temp])
                val_labels.append(label_batches[i][-val_samples_temp:])
                training_args.append(samp_args[i][:-val_samples_temp])
                val_args.append(samp_args[i][-val_samples_temp:])

        elif abs(VAL_SPLIT_RATIO) <= 1e-8:
            training_batches = tot_batches
            training_labels = label_batches
            training_args = samp_args

        if verbose > 0: self.logger.info(f'Done. Total Time: {time.perf_counter() - t_st}')
        return training_batches, val_batches, training_labels, val_labels, training_args, val_args

    def _raw_append(self, batch_structures):
        """
        Directly append without a check.
        """
        if batch_structures.Energies is not None:
            if self.Energies is None:
                self.Energies = [None, ] * len(self)
            self.Energies.extend(batch_structures.Energies)
        elif self.Energies is not None:
            self.Energies.extend([None] * len(batch_structures))
        if batch_structures.Forces is not None:
            if self.Forces is None:
                self.Forces = [None, ] * len(self)
            self.Forces.extend(batch_structures.Forces)
        elif self.Forces is not None:
            self.Forces.extend([None, ] * len(batch_structures))
        if batch_structures.Labels is not None:
            if self.Labels is None:
                self.Labels = [None, ] * len(self)
            self.Labels.extend(batch_structures.Labels)
        elif self.Labels is not None:
            self.Labels.extend([None] * len(batch_structures))

        self._Sample_ids.extend(batch_structures.Sample_ids)  # mp_ids
        self.Cells.extend(batch_structures.Cells)  # Lattice parameters
        self.Elements.extend(batch_structures.Elements)  # Element symbols
        self.Numbers.extend(batch_structures.Numbers)  # Element numbers
        self.Coords_type.extend(batch_structures.Coords_type)  # coordinates type
        self.Coords.extend(batch_structures.Coords)  # Atom coordinates
        self.Fixed.extend(batch_structures.Fixed)  # fixed masks

    def append(self, batch_structures, strict: bool = True):
        """
        Append information of another BatchStructures to self.
        If original data do not have some properties but appending data do,
        these properties of original data would be padded with `None`, and vice versa.
        Args:
            batch_structures: the BatchStructures to append
            strict: If True, `batch_structures` contains any id which is already in `self` would raise an Error;
                    otherwise, the data will be overwritten if encountered the same id.

        Returns: None

        """
        if not isinstance(batch_structures, BatchStructures):
            raise TypeError('`batch_structures` must be BatchStructures')
        if self._indices is None:
            self._update_indices()
        for i, _ids in enumerate(batch_structures.Sample_ids):
            if _ids in self._indices:
                if strict:
                    raise RuntimeError(f'The {i}th sample in given `batch_structures` has the same sample_id ({_ids}) as existing samples.')
                else:
                    self.remove(_ids)
        self._raw_append(batch_structures)
        self._update_indices()
        self._check_id()
        self._check_len()

    def extend(self, batch_structures_list: Sequence, strict: bool = True):
        """
        Extend a list of BatchStructures to self.
        If original data do not have some properties but appending data do,
         these properties of original data would be padded with `None`, and vice versa.
        Args:
            batch_structures_list: the BatchStructures to append
            strict: If True, `batch_structures` contains any id which is already in `self` would raise an Error;
                    otherwise, the data will be overwritten if encountered the same id.

        Returns: None

        """
        if not isinstance(batch_structures_list, Sequence):
            raise TypeError('`batch_structures_list` must be BatchStructures')
        _temp_bs = BatchStructures()
        _bs: BatchStructures
        for _bs in batch_structures_list:
            assert isinstance(_bs, BatchStructures), f'Encountered invalid type {type(_bs)} in `batch_structures_list`'
            _temp_bs._raw_append(_bs)
        _temp_bs._check_id()
        # check repetitions of id
        if self._indices is None:
            self._update_indices()
        for i, _ids in enumerate(_temp_bs.Sample_ids):
            if _ids in self._indices:
                if strict:
                    raise RuntimeError(f'The {i}th sample in given `batch_structures` has the same sample_id ({_ids}) as existing samples.')
                else:
                    self.remove(_ids)
        self._raw_append(_temp_bs)
        self._update_indices()
        self._check_id()
        self._check_len()

    def remove(self, key: int | str | slice | List):
        """
        Remove the data of given `key` in-place.
        Args:
            key: int for index, str for Sample_ids, slice for a slice of indices.

        Returns: None

        """
        # delete ops
        def _remove_single(indx):
            if self.Atom_list is not None:
                del self.Atom_list[indx]
            if self.Atomic_number_list is not None:
                del self.Atomic_number_list[indx]
            del self.Sample_ids[indx]
            del self.Cells[indx]
            del self.Elements[indx]
            del self.Numbers[indx]
            del self.Coords[indx]
            del self.Coords_type[indx]
            del self.Fixed[indx]
            if self.Energies is not None: del self.Energies[indx]
            if self.Forces is not None:   del self.Forces[indx]
            if self.Labels is not None:   del self.Labels[indx]
        # check key
        if isinstance(key, int):
            if key >= len(self): raise ValueError(f'key {key} is out of range {len(self)}')
            _remove_single(key)
        elif isinstance(key, slice):
            _remove_single(key)
        elif isinstance(key, str):
            if self._indices is None:
                self._update_indices()
                self._check_id()
            if key not in self._indices: raise KeyError(f'the input key `"{key}"` does not exist.')
            indices = self._indices[key]
            _remove_single(indices)
        elif isinstance(key, list):
            _sort_key = sorted(key, reverse=True)
            if _sort_key[-1] < 0: raise IndexError(f'Negative index {_sort_key[-1]} in the list is not supported.')
            if _sort_key[0] > len(self): raise ValueError(f'key {_sort_key[0]} is out of range {len(self)}')
            _check_repeat = _sort_key[0]
            _remove_single(_check_repeat)
            for del_id in _sort_key[1:]:
                if not isinstance(del_id, int):
                    raise TypeError(f'Expected all elements in key list are int, but got {type(del_id)}.')
                if del_id != _check_repeat:  # avoid duplicated indices
                    _remove_single(del_id)
                    _check_repeat = del_id
        else:
            raise TypeError(f'Invalid type of key: {type(key)}')

        # update id dict
        self._update_indices()

    def remove_copy(self, key: int | str | slice):
        """
        Remove the data of given `key` in a copy of self and return. The original data is not affected.
        Args:
            key: int for index, str for Sample_ids, slice for a slice of indices.

        Returns: copy of

        """
        sub_self = copy.deepcopy(self)
        sub_self.remove(key)

        return sub_self

    ########################################################################################
    #                                    QUERIES
    #
    ########################################################################################
    def where(self, key: str | List[str]):
        """
        Return the index/indices of given sample id(s) `key` in self
        Returns: int | List[int]

        """
        if isinstance(key, list):
            indx_list = list()
            for kk in key:
                indx_list.append(self._indices[kk])
        else:
            indx_list = self._indices[key]

        return indx_list

    def contain_any(self, elements: List[str] | Set[str]):
        """
        Return a view of BatchStructures that contain any of given elements.
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(elements, (Sequence, Set)): raise TypeError(f'elements must be a Sequence but occurred {type(elements)}')
        if not isinstance(elements, Set): elements = set(elements)
        for i, elems in enumerate(self.Elements):
            for elem_ in elems:
                if elem_ in elements:
                    sub_self._Sample_ids.append(self.Sample_ids[i])
                    sub_self.Cells.append(self.Cells[i])
                    sub_self.Elements.append(self.Elements[i])
                    sub_self.Numbers.append(self.Numbers[i])
                    sub_self.Coords.append(self.Coords[i])
                    sub_self.Coords_type.append(self.Coords_type[i])
                    sub_self.Fixed.append(self.Fixed[i])
                    if sub_self.Energies is not None: sub_self.Energies.append(self.Energies[i])  # type: ignore
                    if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                    if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore
                    break
        return sub_self

    def contain_all(self, elements: List[str] | Set[str]):
        """
        Return a view of BatchStructures that contain all given elements.
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(elements, (Sequence, Set)): raise TypeError(f'elements must be a Sequence but occurred {type(elements)}')
        if not isinstance(elements, Set): elements = set(elements)
        for i, elems in enumerate(self.Elements):
            _elem = set(elems)
            if _elem >= elements:
                sub_self._Sample_ids.append(self.Sample_ids[i])
                sub_self.Cells.append(self.Cells[i])
                sub_self.Elements.append(self.Elements[i])
                sub_self.Numbers.append(self.Numbers[i])
                sub_self.Coords.append(self.Coords[i])
                sub_self.Coords_type.append(self.Coords_type[i])
                sub_self.Fixed.append(self.Fixed[i])
                if sub_self.Energies is not None: sub_self.Energies.append(self.Energies[i])  # type: ignore
                if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore

        return sub_self

    def contain_only_in(self, elements: List[str] | Set[str]):
        """
        Return a view of BatchStructures that only contain given elements.
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(elements, (List, Tuple, Set)): raise TypeError(f'elements must be a List or Set but occurred {type(elements)}')
        if not isinstance(elements, Set): elements = set(elements)
        for i, elems in enumerate(self.Elements):
            Not_in = False
            for elem_ in elems:
                if elem_ not in elements:
                    Not_in = True
                    break
            if not Not_in:
                sub_self._Sample_ids.append(self.Sample_ids[i])
                sub_self.Cells.append(self.Cells[i])
                sub_self.Elements.append(self.Elements[i])
                sub_self.Numbers.append(self.Numbers[i])
                sub_self.Coords.append(self.Coords[i])
                sub_self.Coords_type.append(self.Coords_type[i])
                sub_self.Fixed.append(self.Fixed[i])
                if sub_self.Energies is not None: sub_self.Energies.append(self.Energies[i])  # type: ignore
                if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore

        return sub_self

    def not_contain_any(self, elements: List[str] | Set[str], verbose: int = 0) -> "BatchStructures":
        """
        Return a view of BatchStructures that Not contain any of given elements.
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(elements, (Sequence, Set)): raise TypeError(f'elements must be a Sequence but occurred {type(elements)}')
        if not isinstance(elements, Set): elements = set(elements)
        for i, elems in enumerate(self.Elements):
            Not_in = False
            for elem_ in elems:
                if elem_ in elements:
                    Not_in = True
                    break
            if not Not_in:
                sub_self._Sample_ids.append(self.Sample_ids[i])
                sub_self.Cells.append(self.Cells[i])
                sub_self.Elements.append(self.Elements[i])
                sub_self.Numbers.append(self.Numbers[i])
                sub_self.Coords.append(self.Coords[i])
                sub_self.Coords_type.append(self.Coords_type[i])
                sub_self.Fixed.append(self.Fixed[i])
                if sub_self.Energies is not None: sub_self.Energies.append(self.Energies[i])  # type: ignore
                if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore

            if verbose > 0:
                if i % round(0.1 * len(self.Elements)) == 0:
                    self.logger.info(f'Progress: {100 * i / len(self.Elements):>5.2f}%')

        return sub_self

    def not_contain_all(self, elements: List[str] | Set[str]):
        """
        Return a view of BatchStructures that Not contain all given elements at the same time.
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(elements, (Sequence, Set)): raise TypeError(f'elements must be a Sequence but occurred {type(elements)}')
        if not isinstance(elements, Set): elements = set(elements)
        for i, elems in enumerate(self.Elements):
            Put_in = False
            for elem_ in elements:
                if elem_ not in elems:
                    Put_in = True
                    break
            if Put_in:
                sub_self._Sample_ids.append(self.Sample_ids[i])
                sub_self.Cells.append(self.Cells[i])
                sub_self.Elements.append(self.Elements[i])
                sub_self.Numbers.append(self.Numbers[i])
                sub_self.Coords.append(self.Coords[i])
                sub_self.Coords_type.append(self.Coords_type[i])
                sub_self.Fixed.append(self.Fixed[i])
                if sub_self.Energies is not None: sub_self.Energies.append(self.Energies[i])  # type: ignore
                if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore

        return sub_self

    def select_by_energies(self, energy_range: Tuple[float, float]):
        """
        Return a view of BatchStructures that Energies between energy_range.
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(energy_range, (Tuple, List)): raise TypeError(f'energy_range must be a Tuple but occurred {type(energy_range)}')
        if energy_range[0] >= energy_range[1]: raise ValueError('Interval error.')
        if self.Energies is None: raise RuntimeError('No Energy Information yet.')

        for i, ener in enumerate(self.Energies):
            # judge
            if (ener < energy_range[1]) and (ener >= energy_range[0]):
                sub_self._Sample_ids.append(self.Sample_ids[i])
                sub_self.Cells.append(self.Cells[i])
                sub_self.Elements.append(self.Elements[i])
                sub_self.Numbers.append(self.Numbers[i])
                sub_self.Coords.append(self.Coords[i])
                sub_self.Coords_type.append(self.Coords_type[i])
                sub_self.Fixed.append(self.Fixed[i])
                sub_self.Energies.append(self.Energies[i])  # type: ignore
                if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore

        return sub_self

    def select_by_element_number(self, element_number_range: Tuple[float, float]):
        """
        Return a view of BatchStructures that element number between element_number_range.
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(element_number_range, (Tuple, List)): raise TypeError(
            f'energy_range must be a Tuple but occurred {type(element_number_range)}')
        if element_number_range[0] >= element_number_range[1]: raise ValueError('Interval error.')
        if self.Energies is None: raise RuntimeError('No Energy Information yet.')

        for i, nums in enumerate(self.Numbers):
            num = len(nums)
            # judge
            if (num < element_number_range[1]) and (num >= element_number_range[0]):
                sub_self._Sample_ids.append(self.Sample_ids[i])
                sub_self.Cells.append(self.Cells[i])
                sub_self.Elements.append(self.Elements[i])
                sub_self.Numbers.append(self.Numbers[i])
                sub_self.Coords.append(self.Coords[i])
                sub_self.Coords_type.append(self.Coords_type[i])
                sub_self.Fixed.append(self.Fixed[i])
                if sub_self.Energies is not None: sub_self.Energies.append(self.Energies[i])  # type: ignore
                if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore

        return sub_self

    def select_by_prop(self, prop: str, prop_range: Tuple[float, float]):
        """
        Return a view of BatchStructures that given properties of BatchStructures between prop_range.
        The given properties must be comparable.

        Args:
            prop: an attribute name of self e.g., "Number", "Energies"
            prop_range: Tuple[float, float], range of prop to select. It is a left closed right open interval.

        Returns:
            BatchStructures which is a view of self
        """
        sub_self = BatchStructures()
        if self.Energies is not None: sub_self.Energies = list()
        if self.Forces is not None: sub_self.Forces = list()
        if self.Labels is not None: sub_self.Labels = list()

        if not isinstance(prop_range, (Tuple, List)): raise TypeError(f'energy_range must be a Tuple but occurred {type(prop_range)}')
        if prop_range[0] >= prop_range[1]: raise ValueError('Interval error.')
        if self.Energies is None: raise RuntimeError('No Energy Information yet.')

        properties = getattr(self, prop)
        for i, ener in enumerate(properties):
            # judge
            if (ener < prop_range[1]) and (ener >= prop_range[0]):
                sub_self._Sample_ids.append(self.Sample_ids[i])
                sub_self.Cells.append(self.Cells[i])
                sub_self.Elements.append(self.Elements[i])
                sub_self.Numbers.append(self.Numbers[i])
                sub_self.Coords.append(self.Coords[i])
                sub_self.Coords_type.append(self.Coords_type[i])
                sub_self.Fixed.append(self.Fixed[i])
                if sub_self.Energies is not None: sub_self.Energies.append(self.Energies[i])  # type: ignore
                if sub_self.Forces is not None: sub_self.Forces.append(self.Forces[i])  # type: ignore
                if sub_self.Labels is not None: sub_self.Labels.append(self.Labels[i])  # type: ignore

        return sub_self

    def select_by_sample_id(self, pattern: str):
        """
        Return a view of BatchStructures in which sample id could match the given pattern.
        Regular expressions are supported.
        For non-string sample id, str(*) would be applied first to convert id into string.
        Args:
            pattern: the (regular expressions) pattern to match sample ids in self.

        Returns: BatchStructures

        """
        _pattern = re.compile(pattern)
        sub_self = BatchStructures()
        for i, name in enumerate(self.Sample_ids):
            _name = str(name)
            has_match = re.match(_pattern, _name)
            if has_match is not None:
                sub_self._raw_append(self[i])
        sub_self._check_id()
        sub_self._check_len()
        return sub_self

    def elem_distribution(self, count_for: Literal['Atom', 'Structure'] = 'Structure'):
        """
        return a Dict[str, int] of {element: frequency} that shows the distribution of elements in BatchStructures.

        Parameters:
            count_for: 'Atom' or 'Structure', count element frequency by each atom in structures or each structure.
        """
        elem_distribution_dict = {elem_: 0 for elem_ in self._ALL_ELEMENTS}
        if count_for == 'Atom':
            for i, elems in enumerate(self.Elements):
                for elem_ in elems:
                    elem_distribution_dict[elem_] += 1
        if count_for == 'Structure':
            for elem_ in elem_distribution_dict.keys():
                for i, elems in enumerate(self.Elements):
                    if elem_ in elems:
                        elem_distribution_dict[elem_] += 1
        return elem_distribution_dict

    def eq(self, other, rtol: float = 1e-6, atol: float = 1e-6, verbose: bool = False):
        """
        Check if two BatchStructures are equivalent.
        Args:
            other: the BatchStructures to compare.
            rtol: relative tolerance for comparing float number data.
            atol: absolute tolerance for comparing float number data.
            verbose: To control the verbosity. if True, the equality of each attribute will be printed.

        Returns: bool

        """
        other._check_len()
        self._check_len()
        try:
            if len(self) != len(other): return False
            Qid = (self.Sample_ids == other.Sample_ids)
            Qcoo = all([np.allclose(self.Coords[_], other.Coords[_], rtol=rtol, atol=atol) for _ in range(len(self))])
            Qct = (self.Coords_type == other.Coords_type)
            Qcel = all([np.allclose(self.Cells[_], other.Cells[_], rtol=rtol, atol=atol) for _ in range(len(self))])
            Qelm = all([np.all(self.Elements[_] == other.Elements[_]) for _ in range(len(self))])
            Qnum = all([np.all(self.Numbers[_] == other.Numbers[_]) for _ in range(len(self))])
            Qfix = all([np.allclose(self.Fixed[_], other.Fixed[_], rtol=rtol, atol=atol) for _ in range(len(self))])
            Qe = True
            Qf = True
            Ql = True
            if self.Energies is not None:
                Qe = (self.Energies == other.Energies)
            if self.Forces is not None:
                Qf = all([np.allclose(self.Forces[_], other.Forces[_], rtol=rtol, atol=atol) for _ in range(len(self))])
            if self.Labels is not None:
                Ql = (self.Labels == other.Label)
            qq = (Qid & Qcoo & Qct & Qcel & Qelm & Qnum & Qfix & Qe & Qf & Ql)
            if verbose:
                self.logger.info(
                    f'Sample_ids: {Qid}\n'
                    f'Cells: {Qcel}\n'
                    f'Elements: {Qelm}\n'
                    f'Numbers: {Qnum}\n'
                    f'Coords: {Qcoo}\n'
                    f'Fixed: {Qfix}\n'
                    f'Energies: {Qe}\n'
                    f'Forces: {Qf}\n'
                    f'Labels: {Ql}'
                )
            return qq
        except Exception as e:
            if verbose:
                self.logger.warning(
                    f"An Error occurred while comparing self and other: {e}\n"
                    f"Hence, `False` is returned as the result."
                )
            return False

    def __len__(self, ) -> int:
        if self.Mode == 'L':
            return len(self._Sample_ids)
        elif self.Mode == 'A':
            return len(self._Sample_ids_) if self._Sample_ids_ is not None else 0
        else:
            raise ValueError(f'Mode {self.Mode} not supported.')

    def __repr__(self) -> str:
        info = (
            f'BatchStructures[\n\ttotal {len(self)} structures\n\thas Atom_list: {self.Atom_list is not None}\n\t'
            f'has Atomic_number_list: {self.Atomic_number_list is not None}'
            f'\n\thas Energies: {self.Energies is not None}\n\thas Forces: {self.Forces is not None}\n\t'
            f'has Labels: {self.Labels is not None}\n{" " * 15}]'
        )
        return info

    def __contains__(self, val) -> bool:
        if val in self._Sample_ids:
            return True
        else:
            return False

    def __eq__(self, other):
        return self.eq(other)

    def __getitem__(self, key: int | str | slice | Iterable):  # tips: this method was not efficient and used to occasionally inquire information
        """
        Args:
            key: if `int`, inquiring the `key`th data;
                 if other `Hashable` types (e.g., str), inquire the data which Samp_id is `key`;
                 if `slice`, inquiring the range which slice indicates.
                 if `Iterable`, each element in the `key` will be recursively inquired and appended to return.

        Returns: a subclass of BatchStructure.
        """
        # check key
        if isinstance(key, int):
            if key >= len(self): raise IndexError(f'key {key} is out of range {len(self)}')
            indx = slice(key, key + 1) if key != -1 else slice(key, None)

        elif isinstance(key, slice):
            indx = key

        elif isinstance(key, str):
            if self._indices is None:
                self._update_indices()
                self._check_id()
            if key not in self._indices: raise KeyError(f'the input key `"{key}"` does not exist.')
            indx = self._indices[key]
            indx = slice(indx, indx + 1)

        elif isinstance(key, Iterable):  # use a recursive strategy
            sub_self = BatchStructures()
            for _indx_ in key:
                sub_self._raw_append(self[_indx_])
            return sub_self

        else:
            raise TypeError(f'Invalid type of key: {type(key)}')
        # load data
        if self.Atom_list is None:
            atomlist = None
        else:
            atomlist = self.Atom_list[indx]
        if self.Atomic_number_list is None:
            atomicnumberlist = None
        else:
            atomicnumberlist = self.Atomic_number_list[indx]

        sub_self = BatchStructures()
        sub_self._Sample_ids = self.Sample_ids[indx]
        sub_self.Cells = self.Cells[indx]
        sub_self.Elements = self.Elements[indx]
        sub_self.Numbers = self.Numbers[indx]
        sub_self.Coords = self.Coords[indx]
        sub_self.Coords_type = self.Coords_type[indx]
        sub_self.Fixed = self.Fixed[indx]
        sub_self.Atom_list = atomlist
        sub_self.Atomic_number_list = atomicnumberlist
        if self.Energies is not None: sub_self.Energies = self.Energies[indx]
        if self.Forces is not None: sub_self.Forces = self.Forces[indx]
        if self.Labels is not None: sub_self.Labels = self.Labels[indx]

        return sub_self

    def __setitem__(self, key: int | str | slice, value, *args, **kwargs):
        if not isinstance(value, BatchStructures):
            raise TypeError('The set value must be a BatchStructure type.')
        if not isinstance(key, (int, slice, str)):
            raise TypeError(f'`key` does not support type {type(key)}')
        if key not in value:
            raise ValueError(f'The sample id `{key}` does not in given value.')
        elif key not in self:
            self.append(value)
        else:
            raise NotImplementedError(
                'Directly modifying the data of `key` is not implemented yet. You may use self.append with `strict = False` instead.'
            )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
