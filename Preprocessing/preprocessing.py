""" Preprocessing, including data transformations. """

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: preprocessing.py
#  Environment: Python 3.12

import copy
import gc
import random
# basic modules
import re
import time
import warnings
from math import floor
from typing import Dict, Set, Tuple, List, Sequence, Any, Union, Literal

import joblib as jb
import numpy as np
import torch as th

from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures
from BM4Ckit.utils._CheckModules import check_module
from .load_files import OUTCAR2Feat, ExtXyz2Feat, POSCARs2Feat

ase = check_module('ase')
_pyg = check_module('torch_geometric.data')
dgl = check_module('dgl')
if _pyg is not None:
    pygData = _pyg.Data
else:
    pygData = None

""" Constance """
ALL_ELEMENT = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
               'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
               'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
               'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
               'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
               'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', }  # element set

__all__ = [
    "CreateASE",
    "CreatePygData",
    "CreateDglData",
    "BlockedRW",
    "split_dataset"
]


class ScreenSampleByElement:
    """
    Screening samples by given conditions.

    Methods:
        screen_from_elements: select samples that only contain elements in input element_contains from input formula_dict, and return selected dict.
        screen_from_formula: select samples that only contain elements in input element_contains from input formula_dict, and return selected dict.
    """

    def __init__(self, ) -> None:
        """
        Screening samples by given conditions.

        Methods:
            screen_from_elements: select samples that only contain elements in input element_contains from input formula_dict, and return selected dict.
            screen_from_formula: select samples that only contain elements in input element_contains from input formula_dict, and return selected dict.
        """
        warnings.warn(
            'This Method Has Been Deprecated. Use BatchStructure.contain_... or BatchStructures.not_contain_... instead', DeprecationWarning
        )
        self.non_radio = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                          'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                          'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                          'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                          'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'}

    def screen_from_elements(self, elem_list_dict: Dict[str, List[str]],
                             element_contains: str | Set[str] = 'non-radioactive',
                             output_removed_formula: bool = False,
                             *args, **kwargs) -> Union[Tuple[Dict, List], Dict, Any]:
        r"""
        select samples that only contain elements in input element_contains from input formula_dict, and return selected dict.

        Parameter:
            elem_list_dict: Dict[str, List[str]], the dict of {id:List[elements]}
            element_contains: Set[str], the set of elements.
        """
        __elem_list_dict = copy.deepcopy(elem_list_dict)
        if element_contains == 'non-radioactive':
            Sequence_elements = self.non_radio
            remove_elements = ALL_ELEMENT.difference(Sequence_elements)
        else:
            Sequence_elements = element_contains
            remove_elements = ALL_ELEMENT.difference(Sequence_elements)

        removed_formula = list()
        for elem in remove_elements:
            for __id, __elem in elem_list_dict.items():
                if elem in __elem:
                    removed_formula_ = __elem_list_dict.pop(__id, elem_list_dict[__id])
                    if output_removed_formula:
                        removed_formula.append(removed_formula_)
        if output_removed_formula:
            return __elem_list_dict, removed_formula
        else:
            return __elem_list_dict

    def screen_from_elements_para(self, elem_list_dict: Dict[str, List[str]],
                                  element_contains: str | Set[str] = 'non-radioactive',
                                  output_removed_formula: bool = False,
                                  n_core=-1) -> Union[Tuple[Dict, List], Dict, Any]:
        r"""
        select samples that only contains elements in input element_contains from input formula_dict, and return selected dict.

        """
        __elem_list_dict = copy.deepcopy(elem_list_dict)
        if element_contains == 'non-radioactive':
            Sequence_elements = self.non_radio
            remove_elements = ALL_ELEMENT.difference(Sequence_elements)
        else:
            Sequence_elements = element_contains
            remove_elements = ALL_ELEMENT.difference(Sequence_elements)

        removed_formula = list()

        def _rm_singl(elem, ):
            for __id, __elem in elem_list_dict.items():
                if elem in __elem:
                    removed_formula_ = __elem_list_dict.pop(__id, elem_list_dict[__id])
                    if output_removed_formula:
                        removed_formula.append(removed_formula_)

        _para = jb.Parallel(n_jobs=n_core, verbose=1, require='sharedmem')
        _para(jb.delayed(_rm_singl)(elem) for elem in remove_elements)
        if output_removed_formula:
            return __elem_list_dict, removed_formula
        else:
            return __elem_list_dict

    def screen_from_formula(self, formula_dict: 'dict of {id : formula}',  # type: ignore
                            element_contains='non-radioactive',  # type: ignore
                            output_removed_formula: 'bool, whether output a list of removed formula' = False,  # type: ignore
                            *args, **kwargs) -> Any:
        r"""
        select samples that only contain elements in input element_contains from input formula_dict, and return selected dict.

        Returns:
            'dict of selected samples | dict, list of removed formula'
        """

        # copy a formula_dict
        __formula_dict = copy.deepcopy(formula_dict)
        if element_contains == 'non-radioactive':
            Sequence_elements = self.non_radio
            remove_elements = ALL_ELEMENT.difference(Sequence_elements)
        else:
            Sequence_elements = element_contains
            remove_elements = ALL_ELEMENT.difference(Sequence_elements)

        removed_formula = list()
        for elem in remove_elements:
            for __id, __formula in formula_dict.items():
                pattern = elem + '[A-Z0-9()]|(%s$)' % elem  # match an element either before [A-Z0-9()] or at the end.
                n_temp = re.search(pattern, __formula)
                if n_temp is not None:
                    removed_formula_ = __formula_dict.pop(__id, formula_dict[__id])
                    if output_removed_formula:
                        removed_formula.append(removed_formula_)
        if output_removed_formula:
            return __formula_dict, removed_formula
        else:
            return __formula_dict


def element_distribution(formulas_list: List):
    """
    Counting the element distribution of formulas.
    para:
        formulas: list (or other Sequence type) of formula strings e.g., [NiPt3, Al2OCo4S, Li2O,...]
    """
    # initialize
    elem_freq_dict = dict()  # frequencies of every element in all input formulas, {element symbol : freq}
    n_element_component = dict()  # frequencies of number of elements in every compound, {element number : freq}
    Sequence_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                         'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                         'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                         'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                         'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']

    for elem in Sequence_elements:
        elem_freq_dict[elem] = 0
    for n in range(20):
        n_element_component['%i' % (n + 1)] = 0

    # counting
    for i, formula in enumerate(formulas_list):
        n_ = 0
        for elem in Sequence_elements:
            pattern = elem + '[A-Z0-9()]|(%s$)' % elem  # match an element either before [A-Z0-9()] or at the end.
            n_temp = re.search(pattern, formula)
            if n_temp is not None:
                elem_freq_dict[elem] += 1
                n_ += 1
        try:
            n_element_component['%i' % n_] += 1
        except Exception as e:
            print(f'WARNING : Failed to count up formula {formula}, because an error "{e}" occurred')

    return elem_freq_dict, n_element_component


class BandKit:
    """
    A Toolkit to load VASP band file EIGENVAL and KPOINT.

    Attributes:
        self.eig_info: eigenvalues for each (n, k)
        self.kpt_path: k-points information. Format: [[x1, y1, z1, weight1], ...]
        self.is_spin: whether spin polarized. 1 for False, 2 for True.
        self.k_labels: label of k-points.
    """

    def __init__(self):
        self.zero_weight_indices = None
        self.kpt_path = None
        self.eig_info = None
        self.is_spin = 1
        self.k_labels = None
        self.k_interval = None
        self.band_interp_func = None

    def load_EIGENVAL(
            self,
            file: str = './EIGENVAL',
            *args,
            **kwargs
    ) -> None:
        """
        Read VASP output file EIGENVAL to np.array
        """
        with open(file, 'r') as f:
            data = f.readlines()
        self.is_spin = int(data[0].split()[-1])
        nn, k_points, n_bands = data[5].split()
        k_points, n_bands = int(k_points), int(n_bands)

        k_coords = [data[7 + i * (n_bands + 2)].split() for i in range(k_points)]
        k_coords = np.array(k_coords, dtype=np.float64)

        eigs = [data[8 + i * (n_bands + 2): 6 + (i + 1) * (n_bands + 2)] for i in range(k_points)]
        eigs = [a.split() for eigs_ in eigs for a in eigs_]
        eigs = np.array(eigs, dtype=np.float64).reshape((k_points, n_bands, 5))

        self.kpt_path = k_coords  # ([[x1, y1, z1, weight1], ...])
        self.zero_weight_indices = np.where(np.abs(self.kpt_path[:, -1]) <= 1e-5)[0]
        self.eig_info = eigs

    def load_k_path(
            self,
            file: str = './KPATH.in'
    ):
        """
        Read k-path file to determine k-path range.
        """
        with open(file, 'r') as f:
            data = f.readlines()
        if (data[2].split()[0][0] != 'L') and (data[2].split()[0][0] != 'l'):
            raise ValueError('k-path file is not a linear mode VASP KPOINTS file.')

        k_num_in_each_line = int(data[1].split()[0])
        self.k_labels = list()
        self.k_interval = list()
        has_start = False
        main_data = data[4:]
        for i, kl in enumerate(main_data):
            if kl == ' \n': continue
            if not has_start:
                st1 = kl.split()
                st2 = main_data[i + 1].split()
                labels = (st1[-1], st2[-1])
                self.k_labels.append(labels)
                interval = (np.asarray(st1[:-1], dtype=np.float64), np.asarray(st2[:-1], dtype=np.float64))
                self.k_interval.append(interval)
                has_start = True
            else:
                has_start = False
                continue

    def calc_effective_mass(self, n: int, k: int | Tuple | np.ndarray = (0, 0, 0)):
        """
        Calculate the effective mass at `k` k-point.
        Args:
            n: the index of band to calc. effective mass.
            k: for `k` is int, the `k`th k-point with 0 weight would be selected. The order is as same as given EIGENVAL.
                for `k` is ndarray or List, it must have the shape of (3, ), and the k-point with this given coordinate would be selected.
                If there are more than 1 k-points had same coordinate, the 1st one would be selected.

        Returns: float, effective mass.

        """
        k_0w_points = self.kpt_path[self.zero_weight_indices, :-1]
        eigval_0w = self.eig_info[self.zero_weight_indices]
        if isinstance(k, int):
            kpt = k_0w_points[k]
            k_indx = k
            eig = eigval_0w[k_indx, n, 1:1 + self.is_spin]
        elif isinstance(k, (List, Tuple, np.ndarray)):
            k = np.asarray(k)
            k_indx = np.where(np.linalg.norm(k - k_0w_points, axis=-1) < 1e-6)[0]
            if len(k_indx) != 0:
                kpt = k
                k_indx = k_indx[0]
                eig = eigval_0w[k_indx, n, 1:1 + self.is_spin]
            else:
                raise RuntimeError('Given k-point `k` does not in EIGENVAL.')
        else:
            raise TypeError(f'Invalid `k` type: {type(k)}')

        find_k_intval = False
        for intval in self.k_interval:
            if np.linalg.norm(kpt - intval[0], axis=-1) < 1e-6:  # given k-points at interval start
                kpt_front = k_0w_points[k_indx + 1]
                eig_front = eigval_0w[k_indx + 1, n, 1:1 + self.is_spin]
                kpt_back = 2 * kpt - kpt_front  # center symmetry
                eig_back = eig_front
                # collinear judge
                cos_t = (((kpt_front - intval[0]) @ (kpt_front - intval[1])) /
                         (np.linalg.norm(kpt_front - intval[0]) * np.linalg.norm(kpt_front - intval[1])))
                if abs(cos_t + 1.) > 1e-5:
                    raise RuntimeError(f'No collinear k-point between interval {intval}, so directional derivative cannot be calculate.')
                find_k_intval = True
                break
            elif np.linalg.norm(kpt - intval[1], axis=-1) < 1e-6:  # given k-points at interval end
                kpt_back = k_0w_points[k_indx - 1]
                eig_back = eigval_0w[k_indx - 1, n, 1:1 + self.is_spin]
                kpt_front = 2 * kpt - kpt_back  # center symmetry
                eig_front = eig_back
                # collinear judge
                cos_t = (((kpt_back - intval[0]) @ (kpt_back - intval[1])) /
                         (np.linalg.norm(kpt_back - intval[0]) * np.linalg.norm(kpt_back - intval[1])))
                if abs(cos_t + 1.) > 1e-5:
                    raise RuntimeError(f'No collinear k-point between interval {intval}, so directional derivative cannot be calculate.')
                find_k_intval = True
                break
            else:  # given k-points within interval
                cos_t = (((kpt - intval[0]) @ (kpt - intval[1])) /
                         (np.linalg.norm(kpt - intval[0]) * np.linalg.norm(kpt - intval[1])))
                if abs(cos_t + 1.) < 1e-5:
                    kpt_front = k_0w_points[k_indx + 1]
                    eig_front = eigval_0w[k_indx + 1, n, 1:1 + self.is_spin]
                    kpt_back = k_0w_points[k_indx - 1]
                    eig_back = eigval_0w[k_indx - 1, n, 1:1 + self.is_spin]
                    find_k_intval = True
                    break
                else:
                    find_k_intval = False

        if not find_k_intval:
            raise RuntimeError('Given k-points do not belong any k-path.')
        # interpolation
        # x_ = np.concatenate((kpt_back[None, :], kpt[None, :], kpt_front[None, :]), axis=0)
        # y_ = np.concatenate((eig_back[None, :], eig[None, :], eig_front[None, :]), axis=0)
        # f_interp = interp.interp1d(x_, y_, kind='quadratic', axis=0)

        # 2nd derivative
        df = np.linalg.norm(kpt - kpt_back)
        db = np.linalg.norm(kpt_front - kpt)
        if (df > 0.02) or (db > 0.02):
            warnings.warn(f'distance among 3 k-points for 2-order central difference quotient are too large ({df, db}), '
                          f'thus leading to an inaccurate result.')
        HBAR = 1.  # a.u., equal to 1.054571726e-34 J*s
        dev2 = (1 / 27.211) * (eig_front + eig_back - 2 * eig) / (df * db * (1 / 1.889727 ** 2))
        # m_eff = HBAR**2 / dev2 * 2.740322006e-28   # kg, hbar**2/dev2: (6.62606957e-34 J*s)**2/((1.6021766e-19 J)(1e-10 m)**2)
        m_eff = HBAR ** 2 / dev2  # a.u., effective mass relative to static electron mass 9.1093826e-31 kg.

        return m_eff


class CreateASE:
    r"""
    Create a List[ASE.Atoms] by input crystal information.

    Parameters:
        verbose: int, control the verboseness of output.

    Methods:
        create:
            parameters:
                cell_vectors: tensor with shape (batch_size, 3, 3), batch of cell vectors.
                atomic_numbers: tensor with shape (batch_size, n_atom), batch of atomic numbers in each cell.
                atomic_coordinates: tensor with shape (batch_size, n_atom, 3), batch of Cartesian coordinates x,y,z in each cell.
                supercell_index: tensor with shape (3,), the index of cell with respect to the original cell.
            return:
                List[ase.Atoms] with length batch_size
    """

    def __init__(self, verbose: int = 0) -> None:
        # check model
        if ase is None:
            raise ImportError('`CreateASE` requires package `ase` which could not be imported.')
        self.verbose = verbose
        pass

    def feat2ase(self, feat: BatchStructures, set_tags: bool = True, n_core=-1) -> List[ase.Atoms]:
        r"""
        Convert to ase.Atoms from given BatchStructures.

        Parameters:
            feat: BatchStructures.
            set_tags: bool, whether set Atoms. tags = np.ones(n_atom) automatically.
            n_core: number of CPU cores in parallel.
        
        Returns:
            List[ase.Atoms], the list of Atoms instances.
        """
        t_st = time.perf_counter()
        feat.generate_atom_list()
        if self.verbose: print('Converting to ASE.Atoms ...')

        def _base_convert(symb: Sequence, pos: Sequence, cell: Sequence, pbc: Tuple[bool, bool, bool] | bool = (True, True, True),
                          set_tags: bool = True) -> ase.Atoms:
            samp = ase.Atoms(symbols=symb, positions=pos, cell=cell, pbc=pbc)
            if set_tags:
                samp.set_tags(np.ones(len(samp)))
            return samp

        _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose)
        ase_list = _para(jb.delayed(_base_convert)(symb, feat.Coords[i], feat.Cells[i], True, set_tags) for i, symb in enumerate(feat.Atom_list))

        if ase_list is None: raise RuntimeError
        if self.verbose: print(f'Done. Total Time: {time.perf_counter() - t_st:<5.4f}')

        return ase_list

    def array2ase(self, symb: Sequence, pos: Sequence, cell: Sequence, pbc: Tuple[bool, bool, bool] | bool = (True, True, True),
                  set_tags: bool = True, n_core: int = -1) -> List[
        ase.Atoms]:
        r"""
        Convert to ase.Atoms from the given Sequence of symbols, positions, cell vectors and pbc information.

        Parameters:
            n_core:
            set_tags:
            symb: Sequence[Sequence], the sequence of element lists.
            pos: Sequence[Sequence], the sequence of atom coordinates lists.
            cell: Sequence[Sequence], the sequence of cell vectors lists.
            pbc: bool|Tuple[bool, bool, bool], the direction of periodic boundary condition (x, y, z).
        
        Returns:
            Dict[ase.Atoms], the list of Atoms instances.
        """
        t_st = time.perf_counter()
        if self.verbose: print('Converting to ASE.Atoms ...')

        def _base_convert(symb: Sequence, pos: Sequence, cell: Sequence, pbc: Tuple[bool, bool, bool] | bool = (True, True, True),
                          set_tags: bool = True) -> ase.Atoms:
            samp = ase.Atoms(symbols=symb, positions=pos, cell=cell, pbc=pbc)
            if set_tags:
                samp.set_tags(np.ones(len(samp)))
            return samp

        _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose)
        ase_list = _para(jb.delayed(_base_convert)(_symb, pos[i], cell[i], pbc, set_tags) for i, _symb in enumerate(symb))

        if ase_list is None: raise RuntimeError
        if self.verbose: print(f'Done. Total Time: {time.perf_counter() - t_st:<5.4f}')

        return ase_list

    def feat2ase_dict(self, feat: BatchStructures, set_tags: bool = True, n_core: int = -1) -> Dict[str, ase.Atoms]:
        r"""
        Convert to ase.Atoms from given BatchStructures.

        Parameters:
            n_core:
            feat: BatchStructures.
            set_tags: bool, whether set Atoms.tags = np.ones(n_atom) automatically.
        
        Returns:
            Dict[samp_id:ase.Atoms], the dict of Atoms instances with keys sample id.
        """
        t_st = time.perf_counter()
        ase_dict = dict()
        feat.generate_atom_list()
        if self.verbose: print('Converting to ASE.Atoms ...')

        def _sig_conv(i, symb):
            samp = ase.Atoms(symbols=symb, positions=feat.Coords[i], cell=feat.Cells[i], pbc=True)
            if set_tags:
                samp.set_tags(np.ones(len(samp)))
            ase_dict[feat.Sample_ids[i]] = samp

        _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose, require='sharedmem')
        _para(jb.delayed(_sig_conv)(i, s) for i, s in enumerate(feat.Atom_list))  # type: ignore

        if self.verbose: print(f'Done. Total Time: {time.perf_counter() - t_st:<5.4f}')

        return ase_dict


class CreatePygData:
    r"""
    create torch-geometric.data.Data or Batch from various types.
    """

    def __init__(self, verbose: int = 0) -> None:
        # check module
        if _pyg is None:
            raise ImportError('`CreatePygData` requires package `torch-geometric` which could not be imported.')
        self.verbose = verbose
        pass

    @staticmethod
    def single_ase2data(atoms: ase.Atoms):
        """Convert a single atomic structure to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

        Returns:
            data (torch_geometric.data.Data): A geometric data object with positions, atomic_numbers, tags,
            and optionally, energy, forces, distances, edges, and periodic boundary conditions.
            Optional properties can include by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = th.Tensor(atoms.get_atomic_numbers())
        positions = th.Tensor(atoms.get_positions())
        cell = th.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
        natoms = positions.shape[0]
        # initialized to th.zeros(natoms) if tags missing.
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_tags
        tags = th.Tensor(atoms.get_tags())
        fixed = th.zeros_like(atomic_numbers)
        pbc = th.from_numpy(atoms.pbc)
        # put the minimum data in th geometric data object
        data = pygData(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
            fixed=fixed,
            pbc=pbc,
        )
        return data

    def ase2data_list(self, atom_list: List[ase.Atoms], n_core: int = 1) -> List[pygData]:
        r"""
        Convert a list of ase.Atoms into a pyg.Batch
        """
        if n_core == 1:
            if self.verbose: print('Converting ase Atoms to pyg Batch sequentially...')
            data_list = [self.single_ase2data(_atom) for _atom in atom_list]
        else:
            if self.verbose: print(f'Converting ase Atoms to pyg Batch in parallel with {n_core} cores...')
            _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose)
            data_list: List | None = _para(jb.delayed(self.single_ase2data)(_atom) for _atom in atom_list)
        if data_list is None: raise RuntimeError('Occurred None data.')

        if self.verbose: print('Done.')
        return data_list

    def feat2data_list(self, feat: BatchStructures, n_core: int = 1) -> List[pygData]:
        r"""
        Convert BatchStructures into a list of pyg.Data for fair-chem model
        """
        if feat.Atomic_number_list is None: feat.generate_atomic_number_list()

        def _convert_single(_id, _cell, _coords, _atomic_numbers, _fix):
            cell = th.from_numpy(_cell).view(1, 3, 3).to(th.float32)
            positions = th.from_numpy(_coords).to(th.float32)
            atomic_numbers = th.tensor(_atomic_numbers)  # type: ignore
            natoms = len(atomic_numbers)
            tags = th.ones_like(atomic_numbers, dtype=th.float32)
            fixed = th.from_numpy(_fix)  #.unsqueeze(0)  # fixme
            pbc = th.tensor([True, True, True])
            # put the minimum data in th geometric data object
            _data = pygData(
                cell=cell,
                pos=positions,
                atomic_numbers=atomic_numbers,
                natoms=natoms,
                tags=tags,
                fixed=fixed,
                pbc=pbc,
                idx=_id,
                sid=_id
            )
            return _data

        if n_core == 1:
            if self.verbose: print('Converting BatchStructures to pyg Batch sequentially...')
            data_list: List = [_convert_single(_id,
                                               feat.Cells[i],
                                               feat.Coords[i],
                                               feat.Atomic_number_list[i],
                                               feat.Fixed[i]) for i, _id in enumerate(feat.Sample_ids)]  # type: ignore
        else:
            if self.verbose: print(f'Converting BatchStructures to pyg Batch in parallel with {n_core} cores...')
            _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose)
            data_list = _para(jb.delayed(_convert_single)(_id,
                                                          feat.Cells[i],
                                                          feat.Coords[i],
                                                          feat.Atomic_number_list[i],
                                                          feat.Fixed[i]) for i, _id in enumerate(feat.Sample_ids))  # type: ignore

        if self.verbose: print('Done.')
        return data_list


class CreateDglData:
    r"""
    create dgl.graph from various types.
    The output DGLGraph has format as follows:
        dgl.heterograph(
            {
                ('atom', 'bond', 'atom'): ([], []),
                ('cell', 'disp', 'cell'): ([], [])
            },
            num_nodes_dict={
                'atom': n_atom,
                'cell': 1
            }
        )
        data.nodes['atom'].data['pos']: (n_atom, 3), Atom positions in Cartesian coordinates.
        data.nodes['atom'].data['Z']: (n_atom, ), Atomic numbers.
        data.nodes['cell'].data['cell']: (1, 3, 3), Cell vectors.
    """

    def __init__(self, verbose: int = 0) -> None:
        # check module
        if dgl is None:
            raise ImportError('`CreateDglData` requires package `dgl` which could not be imported.')
        self.verbose = verbose
        pass

    @staticmethod
    def single_ase2graph(atoms: ase.Atoms):
        """
        Convert a single atomic structure to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

        Returns:
            data (torch_geometric.data.Data): A geometric data object with positions, atomic_numbers, tags,
            and optionally, energy, forces, distances, edges, and periodic boundary conditions.
            Optional properties can include by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = th.Tensor(atoms.get_atomic_numbers())
        positions = th.Tensor(atoms.get_positions())
        cell = th.from_numpy(np.array(atoms.get_cell())).view(1, 3, 3)
        n_atoms = positions.shape[0]
        # initialized to th.zeros(natoms) if tags missing.
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_tags
        # TODO: these other properties might be used in future
        tags = th.Tensor(atoms.get_tags())
        fixed = th.zeros_like(atomic_numbers)
        pbc = th.from_numpy(atoms.pbc)
        # TODO: END
        # put the minimum data in th geometric data object
        data = dgl.heterograph(
            {
                ('atom', 'bond', 'atom'): ([], []),
                ('cell', 'disp', 'cell'): ([], [])
            },
            num_nodes_dict={
                'atom': n_atoms,
                'cell': 1
            }
        )
        data.nodes['atom'].data['pos'] = positions
        data.nodes['atom'].data['Z'] = atomic_numbers
        data.nodes['cell'].data['cell'] = cell
        return data

    def ase2graph_list(self, atom_list: List[ase.Atoms], n_core: int = 1) -> List[pygData]:
        r"""
        Convert a list of ase.Atoms into a list of dgl.DGLGraph
        """
        if n_core == 1:
            if self.verbose: print('Converting ase.Atoms to dgl.DGLGraph sequentially...')
            data_list = [self.single_ase2graph(_atom) for _atom in atom_list]
        else:
            if self.verbose: print(f'Converting ase Atoms to pyg Batch in parallel with {n_core} cores...')
            _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose)
            data_list: List | None = _para(jb.delayed(self.single_ase2graph)(_atom) for _atom in atom_list)
        if data_list is None: raise RuntimeError('Occurred None data.')

        if self.verbose: print('Done.')
        return data_list

    def feat2graph_list(self, feat: BatchStructures, n_core: int = 1) -> List[pygData]:
        r"""
        Convert BatchStructures into a list of pyg.Data for fair-chem model
        """
        if feat.Atomic_number_list is None: feat.generate_atomic_number_list()

        def _convert_single(_id, _cell, _coords, _atomic_numbers, _fix):
            cell = th.from_numpy(_cell).view(1, 3, 3).to(th.float32)
            positions = th.from_numpy(_coords).to(th.float32)
            atomic_numbers = th.tensor(_atomic_numbers)  # type: ignore
            n_atoms = len(atomic_numbers)
            fixed = th.from_numpy(_fix).to(th.float32)  # .unsqueeze(0)  # fixme
            # put the minimum data in th geometric data object
            _data = dgl.heterograph(
                {
                    ('atom', 'bond', 'atom'): ([], []),
                    ('cell', 'disp', 'cell'): ([], [])
                },
                num_nodes_dict={
                    'atom': n_atoms,
                    'cell': 1
                }
            )
            _data.nodes['atom'].data['pos'] = positions
            _data.nodes['atom'].data['Z'] = atomic_numbers
            _data.nodes['cell'].data['cell'] = cell
            _data.nodes['cell'].data['idx'] = th.tensor([_id], dtype=th.int64)
            return _data

        if n_core == 1:
            if self.verbose: print('Converting BatchStructures to dgl.Graph sequentially...')
            data_list: List = [
                _convert_single(
                    i,
                    feat.Cells[i],
                    feat.Coords[i],
                    feat.Atomic_number_list[i],
                    feat.Fixed[i]
                ) for i, _id in enumerate(feat.Sample_ids)
            ]  # type: ignore
        else:
            if self.verbose: print(f'Converting BatchStructures to dgl.Graph in parallel with {n_core} cores...')
            with jb.Parallel(n_jobs=n_core, verbose=self.verbose, backend='threading') as _para:
                data_list = _para(
                    jb.delayed(_convert_single)(
                        i,
                        feat.Cells[i],
                        feat.Coords[i],
                        feat.Atomic_number_list[i],
                        feat.Fixed[i]
                    ) for i, _id in enumerate(feat.Sample_ids)
                )

        if self.verbose: print('Done.')
        return data_list


class BlockedRW:
    """
    Blocked read files and save to a memory-mapping file, or load the memory-mapping file and write to specific files inversely.
    Used to manage big files that memory cannot be loaded at once.
    """

    def __init__(self, file_format: Literal['OUTCAR', 'EXTXYZ', 'POSCAR']):
        """
        Args:
            file_format: format of files to read or write.
        """
        if file_format == 'OUTCAR':
            ff = OUTCAR2Feat
        elif file_format == 'EXTXYZ':
            ff = ExtXyz2Feat
        elif file_format == 'POSCAR':
            ff = POSCARs2Feat
        else:
            raise ValueError(f'`file_format {file_format} is invalid.')

        self.reader = ff
        self.file_format = file_format

    def save(
            self,
            files_path: str,
            file_name_list: List[str] | None = None,
            read_configs: Dict | None = None,
            save_path: str = './data',
            chunk_size: int = 32,
            verbose: int = 1
    ):
        """
        Blocked reading files in `files_path`/`file_name_list` by files_reader, and save to `save_path`.
        Args:
            files_path: path of files.
            file_name_list: list of file names to read. None for all files in given `files_path`.
            read_configs: kwargs of file reader.
            save_path: the path to save memory-mapping files.
            chunk_size: Number of files (NOT Structures) read at a time.
            verbose: verboseness of printing.

        Returns: None
        """
        if read_configs is None: read_configs = dict()
        # split chunks
        n_file = len(file_name_list)
        n_loop = n_file // chunk_size
        # write the first chunk file
        f = self.reader(files_path, verbose)
        f.read(file_name_list[:chunk_size], **read_configs)
        f.save(save_path, 'w')
        del f
        gc.collect()
        for i in range(1, n_loop + 1):
            if verbose > 0:
                print(f'Chunk {i}/{n_loop + 1}, total {chunk_size * i} files have been saved.')
            files = file_name_list[chunk_size * i: chunk_size * (i + 1)]
            f = self.reader(files_path, verbose)
            f.read(files, **read_configs)
            f.save(save_path, 'a')
            del f
            gc.collect()

    def load(
            self,
            load_path: str,
            load_range: Tuple | None = None,
            chunck_size: int = 5000,
            verbose: int = 1
    ):
        """
        Blocked load data from given BatchStructure memory-mapping path.
        Args:
            load_path: the path to load memory-mapping.
            load_range:
            chunck_size:
            verbose:

        Returns: None
        """
        raise NotImplementedError

        pass


def split_dataset(
        data: BatchStructures,
        ratio: float|List[float] = 0.9,
        save_path: str|List[str]|None = None,
        shuffle: bool = False,
        seed: int = None
):
    """
    Splitting the data set into parts with given ratios. It would overwrite if `save_path` already exists.
    Args:
        data: data set with BatchStructure format
        ratio: splitting ratio. If only a float number was given, data would be split into 2 parts with ratio `ratio` and `1 - ratio`.
               The summation of all ratios must be 1.
        save_path: if None, List of split data would be directly returned; otherwise they will be saved as memory-mapping files in given paths.
                   The length of `save_path` must be the same as `ratio`.
        shuffle: whether to shuffle data. If not, data will be divided sequentially.
        seed: random seed for shuffle.

    Returns: List | None

    """
    # check vars
    if not isinstance(data, BatchStructures):
        raise TypeError('data must be BatchStructures.')
    if isinstance(ratio, float): ratio = [ratio, 1 - ratio]
    if sum(ratio) != 1.:
        raise ValueError(f'Summation of `ratio` must be 1., but got {sum(ratio)}.')
    for _ in ratio:
        if (_ >= 1.) or (_ <= 0.): raise ValueError(f'All values in `ratio` must between (0, 1), but got {_}.')
    if save_path is not None:
        assert len(save_path) == len(ratio), f'`save_path` must have a same length with `ratio`, but got {len(save_path), len(ratio)}.'
    result_list: list[BatchStructures] = list()
    # shuffle
    if shuffle:
        # create a shuffle indices
        indx = list(range(len(data)))
        random.seed(seed)
        random.shuffle(indx)
        # create an inverse indices
        inv_indx = [0] * len(data)
        for j, k in enumerate(indx):
            inv_indx[k] = j
        data.rearrange(indx)

    n_sample = len(data)
    data_indx_now = 0
    for i, _ in enumerate(ratio[:-1]):
        sub_BS = data[data_indx_now : data_indx_now + floor(n_sample * _)]
        if save_path is not None:
            sub_BS.save(save_path[i])
        else:
            result_list.append(sub_BS)
        data_indx_now += floor(n_sample * _)
    sub_BS = data[data_indx_now:]
    if save_path is not None:
        sub_BS.save(save_path[-1])
    else:
        result_list.append(sub_BS)

    return result_list if save_path is None else None




