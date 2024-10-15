""" Save BatchStructures as NumPy NdArray format with continuous memory """
## TODO
from itertools import accumulate
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple, Any, Hashable
import warnings
import numpy as np
import time
import logging
import sys

from .BatchStructuresBase import BatchStructures
from ._para_flatt_list import flatten


class BatchStructuresNP(object):
    """
    BatchStructures in NumPy.NdArray format.

    Denote Structures nuber, atom number, and element type number in ith Structure as N, ni, and ei, respectively. N*ni means n1+n2+...+nN, N*ei is same.
    Data type format: Array[shape, dtype].
    Attributes:
        self.Batch_indices: Array[(N, ), int64], batch information of Coords, Atom_list, Atomic_number_list, Fixed, and Forces.
        self.Coords: Array[(N*ni, 3), float32], atom coordinates of structures.
        self.Fixed: Array[(N*ni, 3), int8], whether the atom fixed. 0 for fixed and 1 for free.
        self.Forces: Array[(N*ni, 3), float32], atomic forces of structures.
        self.Atom_list: Array[(N*ni, ), '<U3'], elements symbol of each atom in structures.
        self.Atomic_number_list: Array[(N*ni, ), int16], atomic number of each atom in structures.

        self.Elements_batch_indices: Array[(N, ), int64], batch information of Elements, Numbers.
        self.Numbers: Array[(N*ei, ), int32], number of atoms for each element.
        self.Elements: Array[(N*ei, ), '<U3'], element list of structures.

        self.Cells: Array[(N, 3, 3), float32], lattice vectors of structures.
        self.Coords_type: Array[(N, ), '<U2'], coordinate type of structures. 'C' for Cartesian, and 'D' for Lattice.
        self.Sample_ids: Array[(N, ), str], ID of each structure.
        self.Energies: Array[(N, ), float32], energies (or other numeric label) of each structure.
        self.Labels: Array[(N, ), Any], other label of structures.

    """
    def __init__(self):
        super().__init__()
        self._ALL_ELEMENTS = ['H', 'He',
                              'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                              'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                              'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                              'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                              'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                              'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                              'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                              'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', ]  # element List

        self._Sample_ids_: np.ndarray | None = None
        # Batch info part
        self.Batch_indices_: List | np.ndarray | None = None
        self.Elements_batch_indices_: List | None = None
        # Structure part
        self.Atom_list_ = None
        self.Atomic_number_list_ = None
        self.Cells_: np.ndarray|None = None  # cell vectors
        self.Coords_type_: np.ndarray[Literal['C', 'D']]|None = None  # coordinate type
        self.Coords_: np.ndarray|None = list()  # atomic coordinates
        self.Fixed_: np.ndarray|None = list()  # fixed atoms masks, 0 for fixed and 1 for free.
        self.Elements_: np.ndarray[str]|None = None  # elements
        self.Numbers_: np.ndarray[int]|None = None  # atom number of each element
        # Properties part
        self.Energies_ = None  # energies of structures. None | List[float]
        self.Forces_ = None  # forces of structures. None | List[np.NdArray[N, 3]]
        self.Dist_mat_ = None  # distance matrices of structures. None | List[np.NdArray[N, N]]
        # Others
        self._indices = None
        Z_dict = {key: i for i, key in enumerate(self._ALL_ELEMENTS, 1)}  # a dict which map element symbols into their atomic numbers.
        self._Z_dict = Z_dict
        self.Labels_ = None  # structure labels

        # logging
        self.logger = logging.getLogger('Main.BS')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        if not self.logger.hasHandlers():
            log_handler = logging.StreamHandler(sys.stdout, )
            log_handler.setLevel(logging.INFO)
            log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)

    def load_data(self, data: BatchStructures):
        """
        load data from BatchStructures to convert to BatchStructuresNP

        Args:
            data: BatchStructures data

        Returns: None
        """
        # Batch_indices control the coords, atom_list, atomic_number_list, fixed, forces
        tt = time.perf_counter()
        self.Batch_indices_ = np.array([0, ] + list(accumulate([len(_) for _ in data.Coords])), dtype=np.int64)
        self.Coords_ = np.concatenate(data.Coords, dtype=np.float32)
        self.Fixed_ = np.concatenate(data.Fixed, dtype=np.int8)
        if data.Forces is not None: self.Forces_ = np.concatenate(data.Forces, dtype=np.float32)
        # Element_batch_indices control the Elements, Numbers
        self.Elements_batch_indices_ = np.array([0, ] + list(accumulate([len(_) for _ in data.Numbers])), dtype=np.int64)
        self.Numbers_ = np.asarray(flatten(data.Numbers, 1, ncore=4), dtype=np.int32)
        self.Elements_ = np.asarray(flatten(data.Elements, 1, ncore=4), dtype='<U3')
        # Others are sequentially filled
        self.Cells_ = np.asarray(data.Cells, dtype=np.float32)
        self.Coords_type_ = np.asarray(data.Coords_type, dtype='<U2')
        self._Sample_ids_ = np.asarray(data.Sample_ids, dtype=str)
        if data.Energies is not None: self.Energies_ = np.asarray(data.Energies, dtype=np.float32)
        if data.Labels is not None: self.Labels_ =  np.asarray(data.Labels)
        print(f'TIME: <<< {time.perf_counter() - tt} >>>')

    def generate_atom_list(self, force_update: bool = False) -> None:
        r"""
        Generate attribute self.Atom_list (List[str]) which is the atom lists in order of the sequence of atom coordinates.

        Parameters:
            force_update: bool. if False, Atom_list would not be updated when it's not None.
        """
        if (self.Atom_list_ is None) or force_update:
            self.Atom_list_ = list()
            for i, elem in enumerate(self.Elements_):
                self.Atom_list_.extend([elem, ] * self.Numbers_[i])
            self.Atom_list_ = np.asarray(self.Atom_list_, dtype='<U3')
        elif isinstance(self.Atom_list_, Sequence):
            warnings.warn('Atom_list already existed, so it would not be updated.')
        else:
            raise RuntimeError(f'An Unknown Atom_list occurred. Atom_list: {self.Atom_list_}')

    def generate_atomic_number_list(self, force_update: bool = False) -> None:
        r"""
        Generate attribute self.Atomic_number_list (List[int]) which is the atomic number lists in order of the sequence of atom coordinates.

        Parameters:
            force_update: bool. if False, Atomic_number_list would not be updated when it's not None.
        """
        if (self.Atomic_number_list_ is None) or force_update:
            if self.Atom_list_ is None:
                self.generate_atom_list()
            self.Atomic_number_list_ = np.asarray([self._Z_dict[symb] for symb in self.Atom_list_], dtype=np.int32)
        elif isinstance(self.Atomic_number_list_, np.ndarray):
            warnings.warn('Atomic_number_list already existed, so it would not be updated.')
        else:
            raise RuntimeError(f'An Unknown Atomic_number_list occurred. Atomic_number_list: {self.Atomic_number_list_}')

    def set_Energies(self, energies: np.ndarray | Sequence | Dict):
        """
        Set or reset self.Energies from input `val`.
        Args:
            energies: the input values to set self.Energies
                if Dict, val: {sample_id: energy, ...};
                if Sequence, it must be 1D and have the same order of self.Sample_ids.

        Returns: None

        """
        if len(energies) != len(self):
            raise ValueError(f'labels (length: {len(energies)}) must have the same number of structures (length: {len(self)}).')
        if isinstance(energies, Dict):
            try:
                self.Energies_ = np.asarray([energies[k_] for k_ in self._Sample_ids_], dtype=np.float32)
            except KeyError as e:
                raise KeyError(f'Some keys: {e} in Sample_ids do not in val.')
        elif isinstance(energies, Sequence):
            self.Energies_ = list(energies)
        else:
            raise ValueError('Unknown type of input val.')

    def __len__(self):
        return len(self._Sample_ids_)




