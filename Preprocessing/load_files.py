r"""
Methods of reading and transform various files
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: load_files.py
#  Environment: Python 3.12

import re
import sys
from typing import Any, Dict, List, Sequence, Set, Tuple, Optional, Literal
import time
import os
import copy
import warnings

import joblib as jb
import numpy as np

from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures
from BM4Ckit.utils._CheckModules import check_module
from BM4Ckit.utils.ElemListReduce import elem_list_reduce

__all__ = [
    'POSCARs2Feat',
    'OUTCAR2Feat',
    'ASETraj2Feat',
    'ExtXyz2Feat',
    'load_from_csv',
    'Cif2Feat',
    'load_from_structures',
    'Output2Feat'

]

''' CONSTANCE '''
_ALL_ELEMENT = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', ]  # element List

''' Load labels from local file '''


def load_from_csv(
        file_name: str,
        label_type: type = float,
        ignore_None_label_samp: bool = True,
        read_column: Tuple[int, int] = (0, 1),
        has_title_line: bool = False
) -> Dict[str, Any]:
    r"""
    load information of a csv file into a dict of {samp_1:label_1, ...}.

    Parameters:
        file_name: str, csv file of 2 columns (sample name, sample label)
        label_type: type, type of sample label
        ignore_None_label_samp: bool, whether ignore None label samples i.e., not put them into output dict
        read_column: Tuple[int,int], the 2 columns to read, where the 1st column as keys and the 2nd as values.
        has_title_line: bool, whether ignore the 1st line which is the title or descriptions.

    Return:
        dict of {samp_1:label_1, ...} | dict, id_list of [samp_names]
    """
    f = np.loadtxt(file_name, dtype=str, delimiter=',')
    if has_title_line:
        f = f[1:]
    col1 = read_column[0]
    col2 = read_column[1]

    if ignore_None_label_samp:
        #print('load_from_file : The sample which has None label was ignored...')
        energy_dict = {cont[col1]: label_type(cont[col2])
                       for cont in f if cont[col2] not in {'None', '', 'nan'}}  # a dict of {sample_id: energy}
    else:
        energy_dict = {
            cont[col1]: label_type(cont[col2]) if not (cont[col2] in {'None', 'nan', ''})
            else None
            for cont in f
        }  # a dict of {sample_id: energy}

    return energy_dict

def load_from_structures(path: str):
    """
    Load BatchStructure memory-mapping files from `path` and return this BatchStructures.
    Args:
        path: the path of BatchStructure memory-mapping files.

    Returns: BatchStructures with read data.

    """
    f = BatchStructures()
    f.load(path)
    return f


class POSCARs2Feat(BatchStructures):
    """
    Read and convert a folder of POSCAR files from given path into arrays of atoms, coordinates, cell vectors, etc.
    """

    def __init__(self, path: str = './', verbose: int = 0, *args, **kwargs) -> None:
        """
        Read and convert a folder of POSCAR files from given path into arrays of atoms, coordinates, cell vectors, etc.

        Parameters:
            path: str, the path of POSCAR files
            verbose: int, the verbose of print information

        Method:
            read: read files
            para_read: read files in parallel

        Attributes:
            file_list: list, name list of structure files
            Coord: list, coordinates of atoms. Shape: List[NDArray[(n_atoms, 3), dtype=float32]]
            Cells: list, cell vector of crystals. Shape: List[NDArray[(3,3), dtype=float32]]
            Elements: list, elements of crystals. Shape: List[List[str(element symbols)]]
            Numbers: list, element numbers of crystals. Shape: List[List[int(atom number of each element)]]. The same order of Elements.

        Returns: None
        """
        super().__init__()
        self.path = path
        self.verbose = verbose
        self.files_list = list()

    def para_read(
            self,
            file_list: Optional[List] = None,
            output_coord_type: str = 'cartesian',
            n_core: int = -1,
            backend='loky'
    ):
        r"""
        Reading file in parallel.
        Parameters:
            file_list: list (or other Sequences), the list of selected files to be read. 'None' means read all files in the input path.
            output_coord_type: str, 'cartesian' or 'direct'. The coordination type of output atom coordination.
            n_core: int, the number of CPU cores used in reading files.
            backend: backend for parallelization in joblib. Options: 'loky', 'threading', 'multiprocessing'.
        """
        time_st = time.perf_counter()

        # loading files
        if self.verbose > 0: print('*' * 60 + '\nReading files...')
        if file_list is None:
            self.files_list = os.listdir(self.path)
        else:
            self.files_list = file_list
        if self.files_list is None: raise RuntimeError('occurred a None file list.')
        self.n_samp = len(self.files_list)

        # check vars
        _type_converter = {'cartesian': 'C', 'direct': 'D'}
        if output_coord_type != 'cartesian' and output_coord_type != 'direct':
            raise ValueError('Unknown output coordination type. Please input "cartesian" or "direct".')
        self.Coords_type = [_type_converter[output_coord_type], ] * self.n_samp

        # check parallel and initialize
        if n_core == -1:
            n_core = jb.cpu_count()
        elif n_core <= 0:
            raise ValueError('Invalid CPU numbers.')
        elif n_core > jb.cpu_count():
            warnings.warn('Input n_core is greater than total available CPU number, so it would be set to -1.')
            n_core = jb.cpu_count()

        # Parallel read
        Z_dict = {key: i for i, key in enumerate(_ALL_ELEMENT, 1)}  # a dictionary which map element symbols into their atomic numbers.
        self._Z_dict = Z_dict
        _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose, backend=backend)
        _temp = _para(jb.delayed(self._load)(filename) for filename in self.files_list)
        if _temp is None: raise RuntimeError('Occurred None data.')
        for temp in _temp:
            self._Sample_ids.append(temp[0])  # mp_ids
            self.Cells.append(temp[1])      # Lattice parameters
            self.Elements.append(temp[2])  # Element symbols
            self.Numbers.append(temp[3])  # Element numbers
            self.Coords.append(temp[4])  # Atom coordination (cartesian)
            self.Fixed.append(temp[5])
        self._update_indices()
        self._check_id()
        self._check_len()
        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - time_st:<5.4f}')

    def read(self, file_list: Optional[List] = None, output_coord_type: str = 'cartesian', ) -> None:
        """

        Args:
            file_list: list (or other Sequences), the list of selected files to be read. 'None' means read all files in the input path.
            output_coord_type: str, 'cartesian' or 'direct'. The coordination type of output atom coordination.

        Returns:

        """
        time_st = time.perf_counter()

        Z_dict = {key: i for i, key in enumerate(_ALL_ELEMENT, 1)}  # a dictionary which map element symbols into their atomic numbers.
        self._Z_dict = Z_dict

        # loading files
        if self.verbose > 0: print('*' * 60 + '\nReading files...\n')
        if file_list is None:
            self.files_list = [__f.name for __f in os.scandir(self.path) if __f.is_file()]
        elif isinstance(file_list, (List, Tuple)):
            for __f in file_list:
                if os.path.isfile(os.path.join(self.path, __f)):
                    self.files_list.append(__f)
                else:
                    warnings.warn(f'No such file: {os.path.join(self.path, __f)}, skipped.', RuntimeWarning)
        else:
            raise TypeError(f'Invalid type of `file_list`: {type(file_list)}')
        self.n_samp = len(self.files_list)
        if self.n_samp == 0: raise RuntimeError('Occurred empty file_list. No file could be read.')

        # check vars
        _type_converter = {'cartesian': 'C', 'direct': 'D'}
        if output_coord_type != 'cartesian' and output_coord_type != 'direct':
            raise ValueError('Unknown output coordination type. Please input "cartesian" or "direct".')

        # loading atoms coordinates
        self.Coords_type = [_type_converter[output_coord_type], ] * self.n_samp
        self._Sample_ids = list()
        self.Atom_list = None
        self.Atomic_number_list = None
        self.Cells = list()
        self.Coords = list()
        self.Elements = list()
        self.Numbers = list()
        time_old = copy.deepcopy(time_st)
        err_num = 0
        if self.verbose > 0: print('Progress: ', end='\r')
        for i, fname in enumerate(self.files_list):
            time_mid = time.perf_counter()
            try:
                temp = self._load(fname)
                # update attr
                self._Sample_ids.append(temp[0])  # mp_ids
                self.Cells.append(temp[1])  # Lattice parameters
                self.Elements.append(temp[2])  # Element symbols
                self.Numbers.append(temp[3])  # Element numbers
                self.Coords.append(temp[4])  # Atom coordinates (cartesian)
                self.Fixed.append(temp[5])  # fixed masks
            except Exception as e:
                err_num += 1
                print(f'WARNING : Failed to read file {fname}, because an error "{e}" occurred.'
                      f' ERROR NUMBER: {err_num}')

            # progress bar
            if (self.verbose > 0) and (time_mid - time_old > 1):
                time_old = copy.deepcopy(time_mid)
                prog = ((i + 1) / self.n_samp)
                print('Progress: ' + '>' * int(20 * prog) + f'|{(100 * prog):.1f}%', end='\r')
                i_old = copy.deepcopy(i)

        if self.verbose > 0: print('Progress: ' + '>' * 20 + '|100%  ', end='\r')
        self._indecies = {_id: ii for ii, _id in enumerate(self._Sample_ids)}
        self._update_indices()
        self._check_id()
        self._check_len()
        if self.verbose > 0: print('\nAll files were read successfully!\n' + '*' * 60)
        if self.verbose > 0: print(f'Total time: {(time.perf_counter() - time_st):<5.4f}')
        #print(self.coord_data)

    def _load(self, fileName: str) -> Tuple[str, np.ndarray, List[str], List[int], np.ndarray, np.ndarray]:
        """
        Loading data from files(POSCAR-like files) in given path.
        """
        with open(os.path.join(self.path, fileName), 'r') as f:
            data = f.readlines()  # read raw data

        # atom type
        atom_type = data[5].split()

        # atom number
        _atom_num = data[6].split()
        atom_num = [int(num) for num in _atom_num]
        n_atom = sum(atom_num)
        # whether Selective Dynamics
        if data[7].split()[0][0] == 'S' or data[7].split()[0][0] == 's':
            is_selective_dynamics = 1
        else:
            is_selective_dynamics = 0
        # coordination system(Direct or Cartesian)
        coord_sys = data[7 + is_selective_dynamics].split()
        if coord_sys[0][0] == 'D' or coord_sys[0][0] == 'd':
            in_coord_type = 'D'
        elif coord_sys[0][0] == 'C' or coord_sys[0][0] == 'c':
            in_coord_type = 'C'
        else:
            raise RuntimeError(f'Unknown coordination type "{coord_sys[0][0]}" in file {fileName}')
        # cell vectors
        cell = np.empty((3, 3), dtype=np.float32)
        for i in [2, 3, 4]:
            cell[i - 2] = np.array(data[i].split(), dtype=np.float32)
        # atoms cartesian coordinates # TODO: fill list first, and convert to array then.
        atom_coord = [['-'*10, '-'*10, '-'*10]] * n_atom
        atom_fixed = [[1, 1, 1]] * n_atom
        if is_selective_dynamics == 0:
            for i in range(n_atom):
                atom_coord[i] = (data[8 + i].split())[:3]
        else:
            _slec_dynam_convertor = {'T': 1, 'F': 0}
            for i in range(n_atom):  # if selective dynamics, coordinates shift down 1 line
                line_info = data[8 + 1 + i].split()
                atom_coord[i] = line_info[:3]
                atom_fixed[i] = [_slec_dynam_convertor[_] for _ in line_info[3:6]]
        atom_coord = np.array(atom_coord, dtype=np.float32)
        atom_fixed = np.array(atom_fixed, dtype=np.int8)

        if self.Coords_type[0] == in_coord_type:
            pass
        elif in_coord_type == 'D':
            atom_coord = atom_coord @ cell
        elif in_coord_type == 'C':
            atom_coord = atom_coord @ np.linalg.inv(cell)

        return fileName, cell, atom_type, atom_num, atom_coord, atom_fixed


class ConcatPOSCAR2Feat(BatchStructures):
    r"""
    Read and convert the single file of a batch POSCARs contents from given path into arrays of atoms, coordinates, cell vectors etc.
    All numerical data were stored as float32 in memory.
    The format of the read file:

        # id1
        TITLE
        SCALE FACTOR
        LATTICE VECTOR AXIS X (x1, x2, x3)
        LATTICE VECTOR AXIS Y (y1, y2, y3)
        LATTICE VECTOR AXIS Z (z1, z2, z3)
        CHEMICAL ELEMENTS LIST (Element1 Element2 Element3 ...)
        ATOM NUMBERS OF EACH ELEMENT (N1 N2 N3 ...)
        COORDINATE TYPE ("Direct" or "Cartesian")
        ATOM COORDINATES IN ORDER OF CHEMICAL ELEMENTS LIST (x1 y1 z1)
        (x2 y2 z2)
        (x3 y3 z3)
        ...
        # id2
        ...
        # id3
        ...
        ...

    EOF

    Parameters:
        path: str, the path of POSCAR files
        select_ids: list (or other Sequences), the list of selected id to be read. 'None' means read all files in the input path.
        output_coord_type: str, 'cartesian' or 'direct'. The coordination type of output atom coordination.

    Attributes:
        Sample_ids: list, name list of structrues
        Coords: list, coordinations of atoms. Shape: List[NDArray[(n_atoms, 3), dtype=float32]]
        Cells: list, cell vector of crystals. Shape: List[NDArray[(3,3), dtype=float32]]
        Elements: list, elements of crystals. Shape: List[List[str(element symbols)]]
        Numbers: list, element numbers of crystals. Shape: List[List[int(atom number of each element)]]. the same order of Elements.

    Returns: None
    """

    def __init__(self, path: str = './', select_ids: List[str] | Set[str] | None = None, output_coord_type: str = 'cartesian', verbose: int = 0) -> None:
        warnings.warn('Deprecated.', DeprecationWarning)
        raise Exception('Deprecated.')
        super().__init__()
        time_st = time.perf_counter()
        Z_dict = {key: i for i, key in enumerate(_ALL_ELEMENT, 1)}  # a dictionary which map element symbols into their atomic numbers.
        self._Z_dict = Z_dict

        # loading files
        if verbose: print('*' * 60 + '\nReading files...')
        self.output_coord_type = output_coord_type
        self._Sample_ids = list()
        self.Atom_list = None
        self.Atomic_number_list = None
        self.Cells = list()
        self.Coords = list()
        self.Elements = list()
        self.Numbers = list()
        time_old = copy.deepcopy(time_st)

        with open(path, 'r') as f:
            if select_ids is None:  # read all files
                while True:
                    _text = f.readline()
                    if _text == '':
                        break
                    if _text[0] == '#':  # whether the id line. Find the headline.
                        _id = _text.replace(' ', '')  # str
                        _id = _id[1:-1]
                        self._Sample_ids.append(_id)

                        next(f)  # skip title line
                        scale = float(f.readline())  # scale factor line
                        # Cell Vector
                        _cell_vec = list()
                        _cell_vec.append((f.readline()).split())  # cell axis a
                        _cell_vec.append((f.readline()).split())  # cell axis b
                        _cell_vec.append((f.readline()).split())  # cell axis c
                        _cell_vec = np.asarray(_cell_vec, dtype=np.float32) * scale  # np.NDArray
                        self.Cells.append(_cell_vec)
                        # Atoms
                        _atoms = (f.readline()).split()
                        self.Elements.append(_atoms)  # List[str]
                        # Atom Numbers
                        _atom_num = [int(_) for _ in (f.readline()).split()]
                        self.Numbers.append(_atom_num)  # List[int]
                        _atom_tol_num = sum(_atom_num)
                        # Coord Type
                        _type = (f.readline()).strip()
                        # Coords
                        if _type[0] == 's' or _type[0] == 'S':  # whether 'Selective dynamics'
                            _type = (f.readline()).strip()
                        _coords = np.empty((_atom_tol_num, 3), dtype=np.float32)
                        for ii in range(_atom_tol_num):
                            _coords[ii] = np.asarray((f.readline().split())[:3], dtype=np.float32)
                        if _type[0] == 'd' or _type[0] == 'D':
                            _coords = _coords @ _cell_vec
                        elif _type[0] == 'c' or _type[0] == 'C':
                            pass
                        else:
                            warnings.warn(f'Unknown Coordinate type. Coordinate type of sample *** {_id} *** would be considered as "Direct"')
                            _coords = _coords @ _cell_vec
                        self.Coords.append(_coords)
            else:
                select_ids = set(select_ids)
                _have_read_ids = set()
                while True:
                    _text = f.readline()
                    if _text == '':  # judge EOF
                        break
                    if _text[0] == '#':  # whether the id line. Find the head line.
                        _id = _text.replace(' ', '')  # str
                        _id = _id[1:-1]
                        if _id not in select_ids:
                            continue
                        _have_read_ids.add(_id)
                        self._Sample_ids.append(_id)

                        next(f)  # skip title line
                        scale = float(f.readline())  # scale factor line
                        # Cell Vector
                        _cell_vec = list()
                        _cell_vec.append((f.readline()).split())  # cell axis a
                        _cell_vec.append((f.readline()).split())  # cell axis b
                        _cell_vec.append((f.readline()).split())  # cell axis c
                        _cell_vec = np.asarray(_cell_vec, dtype=np.float32) * scale  # np.NDArray
                        self.Cells.append(_cell_vec)
                        # Atoms
                        _atoms = (f.readline()).split()
                        self.Elements.append(_atoms)  # List[str]
                        # Atom Numbers
                        _atom_num = [int(_) for _ in (f.readline()).split()]
                        self.Numbers.append(_atom_num)  # List[int]
                        _atom_tol_num = sum(_atom_num)
                        # Coord Type
                        _type = (f.readline()).strip()
                        # Coords
                        if _type[0] == 's' or _type[0] == 'S':  # whether 'Selective dynamics'
                            _type = (f.readline()).strip()
                        _coords = np.empty((_atom_tol_num, 3), dtype=np.float32)
                        for ii in range(_atom_tol_num):
                            _coords[ii] = np.asarray((f.readline().split())[:3], dtype=np.float32)
                        if _type[0] == 'd' or _type[0] == 'D':
                            _coords = _coords @ _cell_vec
                        elif _type[0] == 'c' or _type[0] == 'C':
                            pass
                        else:
                            warnings.warn(f'Unknown Coordinate type. Coordinate type of sample *** {_id} *** would be considered as "Direct"')
                            _coords = _coords @ _cell_vec
                        self.Coords.append(_coords)

        if verbose: print('All files were read successfully!\n' + '*' * 60)

        time_ed = time.perf_counter()
        if verbose: print('Total time: %s' % (time_ed - time_st))
        pass


class OUTCAR2Feat(BatchStructures):
    r"""
    Read atoms, coordinates, atom numbers, energies and force in OUTCARs from given path.
    Notes: Because OUTCAR does not store any information of fixed atom (selective dynamics), all atoms would be set to FREE!!

    Parameters:
        path: str, the path of batched OUTCAR-like files. The Sample ids would be labeled as its file names.
    """

    def __init__(self, path: str, verbose: int = 1):
        super().__init__()
        self.verbose = verbose
        self.path = path
        self.Energies = list()
        self.Forces = list()

        self.__n_atom_partt = re.compile(r"(?<=NIONS =)\s+[0-9]+\n")
        self.__pos_force_partt = re.compile(
            r"(?<=TOTAL-FORCE.\(eV/Angst\)\n.-----------------------------------------------------------------------------------\n)\s+[-|0-9\s\n.]+")
        self.__energy_partt = re.compile(r'FREE ENERGIE OF THE ION-ELECTRON SYSTEM.*\n.*\n.*\n.*\n.*energy\(sigma->0\) =\s*([0-9\-.]+)\n')
        self.__cell_partt = re.compile(r'VOLUME and BASIS-vectors are now.*\n.*\n.*\n.*\n.*direct lattice vectors.*\n(.*\n.*\n.*)')
        self.__is_converge_partt = re.compile(r'aborting loop .')

    @staticmethod
    def _get_atom_info(data: str) -> Tuple[List, List]:
        r""" get atom list and atom number from OUTCAR """
        numbers = re.search(r"(?<=ions per type =)\s+[0-9|\s]+\n", data)
        if numbers is not None:
            numbers = numbers.group().split()
        else:
            raise RuntimeError('atom numbers could not find.')
        n_type = len(numbers)

        atoms = re.findall(r"POTCAR:\s*PAW_PBE\s*([A-Za-z]{1,2})", data)
        atoms = atoms[:n_type]

        return atoms, numbers

    def _read_single_file(self, file_name: str, parallel: bool = False) -> Any:
        r"""
        Read single file.

        Parameters:
            file_name: str, the OUTCAR-like file name.
            parallel: bool,
        """
        full_path = os.path.join(self.path, file_name)
        if not os.path.isfile(full_path):
            warnings.warn(f'No OUTCAR file in given directory {os.path.join(self.path, file_name)}')
            return None
        try:
            with open(full_path, "r") as file:  #打开文件
                data = file.read()  #读取文件
            n_atom = re.search(self.__n_atom_partt, data)  #从OUTCAR文件搜索原子数
            if n_atom is None:
                warnings.warn(f'No atoms matched in {file_name}, skipped.', RuntimeWarning)
                if parallel:
                    return [], [], [], [], [], [], [], [], []
                else:
                    return None
            n_atom = int(n_atom.group())  #将输出的原子个数保存为整数形式
            position_iter = re.finditer(self.__pos_force_partt, data)  #通过re匹配的迭代器函数，找到原子坐标
            _energies = re.findall(self.__energy_partt, data)
            is_converged = re.findall(self.__is_converge_partt, data)

            # ATOM & Number
            atoms, numbers = self._get_atom_info(data)
            atoms = np.array(atoms, dtype='<U4')
            numbers = np.array(numbers, dtype=np.int32)
            # Cell Vectors
            cells_str = re.findall(self.__cell_partt, data, )

            cells = list()
            _data = list()
            energies = list()
            for i, match_for in enumerate(position_iter):  #循环的取每次迭代找到的一个结构的原子坐标与受力信息
                # if SCF converged, content in OUTCAR would be 'aborting loop because ...' else 'aborting loop EDIFF was not reached ...'
                if is_converged[i][-1] == 'b':
                    _dat = re.split(r"\n+", match_for.group())  #通过换行符进行划分
                    _dat = [re.split(r'[0-9][\n\s\t]+', dat_) for dat_ in _dat[:-2]]  #去除空字符或都是横线的行，然后循环的对每一行进行划分 <<< # TODO
                    _data.append(_dat)  #将原子坐标的列表添加进列表中
                    energies.append(float(_energies[i]))
                    # sometimes the cell vector would not split by \s e.g., "0.000000000-10.955671660 20.223087260"
                    _cell_line: List[str] = cells_str[i].split()
                    if len(_cell_line) != 3:
                        __cell_line = list()
                        [__cell_line.extend(_.replace('-', ' -').split()) for _ in _cell_line]
                        _cell_line = __cell_line
                        del __cell_line
                    cells.append(_cell_line)

            cells = np.array(cells, dtype=np.float32).reshape(-1, 3, 6)
            cells = cells[:, :, :3]
            _data = np.array(_data, dtype=np.float32)  # (n_step, n_atom, 3)
            if len(_data) == 0:
                warnings.warn(f'Occurred empty data in file {file_name}, skipped.', RuntimeWarning)
                if parallel:
                    return [], [], [], [], [], [], [], [], []
                else:
                    return None

            # formatted
            n_step, n_atom, _ = _data.shape
            coords = _data[:, :, :3]
            coords = [coo for coo in coords]
            forces = _data[:, :, 3:]
            forces = [forc for forc in forces]
            atoms = atoms[None, :].repeat(n_step, axis=0)
            numbers = numbers[None, :].repeat(n_step, axis=0)
            fixed = np.ones_like(coords, dtype=np.int8)

            _id = [file_name + f'_{i}' for i in range(n_step)]
            # output
            if parallel:
                return _id, atoms, numbers, cells, coords, energies, forces, fixed, ['C',] * n_step
            else:
                self._Sample_ids.extend(_id)
                self.Elements.extend(atoms.tolist())
                self.Numbers.extend(numbers.tolist())
                self.Cells.extend(cells)
                self.Coords.extend(coords)
                self.Energies.extend(energies)
                self.Forces.extend(forces)
                self.Fixed.extend(fixed)
                self.Coords_type.extend(["C"] * n_step)
        except Exception as err:
            warnings.warn(f'An Error occurred when reading file {file_name}, skipped.\nError: {err}.')
            if parallel:
                return [], [], [], [], [], [], [], [], []
            else:
                return None

    def read(self, file_list: Optional[List[str]] = None, n_core: int = -1, backend: str = 'loky'):
        r"""
        Parameters:
            file_list: List[str], the list of files to read. Default for all files under given path.
            n_core: int, the number of CPU cores used in reading files.
            backend: backend for parallelization in joblib. Options: 'loky', 'threading', 'multiprocessing'.

        Return: None
        Update the attribute self.data.
        """
        t_st = time.perf_counter()
        if file_list is None:
            file_list = os.listdir(self.path)
            file_list = [f_ for f_ in file_list if os.path.isfile(os.path.join(self.path, f_))]
        elif not isinstance(file_list, Sequence):
            raise TypeError(f'Invalid type of files_list: {type(file_list)}')

        if n_core > len(file_list):
            warnings.warn(f'`ncore` is greater than file numbers, so `ncore` was reset to {len(file_list)}', RuntimeWarning)
            n_core = len(file_list)
        elif n_core == -1:
            n_core = jb.cpu_count()
        elif not(isinstance(n_core, int)) or n_core < -1:
            raise ValueError(f'Invalid `n_core` number: {n_core}.')

        if n_core == 1:
            if self.verbose: print('Sequential Reading...');print('Progress: 0%', end='\r')
            for i, fi_name in enumerate(file_list):
                self._read_single_file(fi_name, parallel=False)
                if self.verbose > 0:
                    if (i + 1) % 50 == 0:
                        prog_ = (i + 1) / len(file_list)
                        prog = round(prog_ / 0.05)
                        print('Progress: ' + '>' * prog + f'{prog_ * 100:>3.2f}%', end='\r')
            if self.verbose: print('Progress: ' + '>' * 20 + f'{100:>3d}%')
        else:
            _para = jb.Parallel(n_jobs=n_core, backend=backend, verbose=self.verbose)
            _dat = _para(jb.delayed(self._read_single_file)(fi_name, parallel=True) for fi_name in file_list)
            for temp in _dat:
                self._Sample_ids.extend(temp[0])
                self.Elements.extend(temp[1])
                self.Numbers.extend(temp[2])
                self.Cells.extend(temp[3])
                self.Coords.extend(temp[4])
                self.Energies.extend(temp[5])
                self.Forces.extend(temp[6])
                self.Fixed.extend(temp[7])
                self.Coords_type.extend(temp[8])
        if not self._Sample_ids: raise RuntimeError('All data are empty.')
        self._update_indices()
        self._check_id()
        self._check_len()
        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - t_st:>5.4f}')

class Xyz2Feat(BatchStructures):
    """
    read xyz files to BatchStructures.
    """
    def __init__(self, path: str, verbose: int = 1, has_cell: bool = False, has_forces: bool = False):
        super().__init__()
        #warnings.warn()
        self.verbose = verbose
        self.path = path
        self.has_cell = has_cell
        self.has_forces = has_forces
        self.Energies = list()
        if has_forces: self.Forces = list()

    def _read_single(self, file_name: str, ) -> Tuple[List[str], List, List, List,
    List[List], List[List], List[np.ndarray], List]:
        with open(os.path.join(self.path, file_name), 'r') as _f:
            data = _f.readlines()
        # match atoms number, energy of every structure and generate atom_number_all_list, Energy_list
        line_of_energy = [d for d in data if len(re.findall(r'i\s=', d)) != 0]
        if self.has_cell: raise NotImplementedError #
        atom_list = [
            int(re.findall(r'\s+[0-9]+\n', d)[0]) for d in data if len(re.findall(r'\s+[0-9]+\n', d)) != 0
        ]
        energy_list = [data[1 + (atom_list[i] + 2) * i].split()[-1] for i in range(len(line_of_energy))]
        element_position_list = [
            data[2 + (atom_list[i] + 2) * i: (atom_list[i] + 2) * (i + 1)] for i in range(len(line_of_energy))
        ]

        # Match coordinate, Element_type ,Element_number and generate Coordinate_list, Element_type_Without_repetition_list, Element_Number_Without_repetition_list
        coordinate_list = list()
        forces_list = list()
        element_type_without_repetition_list = list()
        element_number_without_repetition_list = list()
        idx = list()
        for i, P in enumerate(element_position_list):
            position_list_ = [l.split()[1:] for l in P]
            element_list_ = [l.split()[0] for l in P]
            E = element_list_[:1]
            N = list()
            count_ = 0
            for j, el in enumerate(element_list_):
                if el != E[-1]:
                    E.append(el)
                    N.append(count_)
                    count_ = 1
                else:
                    count_ += 1
            N.append(count_)
            element_type_without_repetition_list.append(E)
            element_number_without_repetition_list.append(N)
            pos_forc_array = np.array(position_list_, dtype=np.float32)
            coordinate_list.append(pos_forc_array[:, :3])
            if self.has_forces:
                forces_list.append(pos_forc_array[:, 3:])
            # idx
            idx.append(file_name + '_' + str(i))
        cells = [None] * len(element_position_list)
        coo_type = ['C'] * len(element_position_list)

        return (idx, cells, energy_list, forces_list, element_type_without_repetition_list,
                element_number_without_repetition_list, coordinate_list, coo_type)

    def read(self, file_list):
        t_st = time.perf_counter()
        if file_list is None:
            file_list = os.listdir(self.path)
            file_list = [f_ for f_ in file_list if os.path.isfile(os.path.join(self.path, f_))]
        elif not isinstance(file_list, Sequence):
            raise TypeError(f'Invalid type of files_list: {type(file_list)}')

        err = 0
        for i, fil in enumerate(file_list):
            try:
                temp = self._read_single(fil)
                self._Sample_ids.extend(temp[0])
                self.Cells.extend(temp[1])
                self.Energies.extend(temp[2])
                if self.has_forces:
                    self.Forces.extend(temp[3])
                self.Elements.extend(temp[4])
                self.Numbers.extend(temp[5])
                self.Coords.extend(temp[6])
                self.Coords_type.extend(temp[7])
                self.Fixed.extend([np.ones_like(_) for _ in temp[6]])

            except Exception as e:
                err += 1
                warnings.warn(
                    f"An error occurred in {i}th file: {fil}, skipped. Total errors: {err}", RuntimeWarning
                )
                if self.verbose > 0:
                    warnings.warn(f"Error: {e}")
        self._update_indices()
        self._check_id()
        self._check_len()


class ExtXyz2Feat(BatchStructures):
    """
    Read extxyz file with multiple structures.
    """
    def __init__(self, path: str, verbose: int = 1):
        super().__init__()
        self.verbose = verbose
        self.path = path
        self.Energies = list()
        self.Forces = list()
        self.fix_mask_trans_func = np.vectorize(self._fixed_info_transformer)
        self.FIXED_DICT = {'T': 1, 'F': 0}

    @staticmethod
    def _fixed_info_transformer(a: np.ndarray, fixed_dict: dict):
        y = fixed_dict[a]
        return y

    def _read_single(
            self,
            file_name: str,
            lattice_tag: str = 'lattice',
            energy_tag: str = 'energy',
            column_info_tag: str = 'properties',
            element_tag: str = 'species',
            coordinates_tag: str = 'pos',
            forces_tag: str|None = 'forces',
            fixed_atom_tag: str|None = 'move_mask'
    ) -> Tuple[List[str], List, List, List, List[List], List[List], List[np.ndarray], List, List]:
        """
        read single extxyz file.
        Args:
            file_name:
            lattice_tag:
            energy_tag:
            column_info_tag:
            element_tag:
            coordinates_tag:
            forces_tag:
            fixed_atom_tag:

        Returns:

        """
        # save infos
        idx = list()
        Cell_list = list()
        Energy_list = list()
        Elem_list = list()
        Number_list = list()
        Coords_list = list()
        Forces_list = list()
        Fixed_list = list()
        if not os.path.isfile(os.path.join(self.path, file_name)):
            warnings.warn(f'No OUTCAR file in given directory {os.path.join(self.path, file_name)}')
            return [], [], [], [], [], [], [], [], []

        try:
            with open(os.path.join(self.path, file_name), 'r') as _f:
                data = _f.readlines()

            line_now = 0  # flag of line now, start from 0
            i_structure = 0  # a counter
            while line_now + 1 < len(data):
                n_atom = int(data[line_now])
                info = data[line_now + 1]
                # PREFERENCES
                # cell
                cell = re.search(self.CELL_PARTTEN, info)
                assert cell is not None, 'Lattice was not found.'
                cell = cell.groups()[0].split()
                cell = np.asarray(cell, dtype=np.float32).reshape(3, 3)
                # properties & column split
                col_content = dict()  # dict recorded column content that {name: (type, column_number_start, end)}
                col_split = re.search(self.PROP_PARTTEN, info)
                if col_split is not None:
                    col_split = col_split.groups()[0].split(':')
                    n_col = len(col_split)
                    assert len(col_split)%3 == 0, 'Invalid properties information format.'
                    _i_now = 0
                    for i_col in range(0, n_col, 3):
                        col_content[col_split[i_col]] = (col_split[i_col + 1], _i_now, _i_now + int(col_split[i_col + 2]))
                        _i_now += int(col_split[i_col + 2])
                else:
                    col_content = {
                        f'{element_tag}': ('S', 0, 1),
                        f'{coordinates_tag}': ('R', 1, 3)
                    }
                # energy
                ener = re.search(self.ENER_PARTTEN, info)
                if ener is None:
                    has_energy = False  # a flag shown whether file contain energy.
                else:
                    has_energy = True
                    ener = float(ener.groups()[0])

                # MAIN INFO
                main_info = np.asarray([_.split() for _ in data[line_now + 2 : line_now + 2 + n_atom]])
                # element & its number
                element_list_ = main_info[:, col_content[f'{element_tag}'][1]:col_content[f'{element_tag}'][2]].flatten()
                elements = [element_list_[0], ]
                numbers = list()
                count_ = 0
                for j, el in enumerate(element_list_):
                    if el != elements[-1]:
                        elements.append(el)
                        numbers.append(count_)
                        count_ = 1
                    else:
                        count_ += 1
                numbers.append(count_)
                # coords
                coords = main_info[:, col_content[f'{coordinates_tag}'][1]:col_content[f'{coordinates_tag}'][2]].astype(np.float32)
                # forces
                forces_info = col_content.get(f'{forces_tag}', None) if forces_tag is not None else None
                if forces_info is not None:
                    forces = main_info[:, forces_info[1]:forces_info[2]].astype(np.float32)
                else:
                    forces = None
                    warnings.warn('`forces_tag` was set, while it cannot be found in the column information.')
                # fixed masks
                if fixed_atom_tag is not None:
                    fixed = main_info[:, col_content[f'{fixed_atom_tag}'][1]:col_content[f'{fixed_atom_tag}'][2]]
                    fixed: np.ndarray = self.fix_mask_trans_func(fixed, self.FIXED_DICT)
                    if fixed.ndim != 2: raise RuntimeError('Unsupported format of atom fixing.')
                    if fixed.shape[1:2] != (3, ):
                        fixed = fixed.repeat(3, 1)
                    fixed.astype(np.int8)
                else:
                    fixed = np.ones_like(coords, dtype=np.int8)

                # write data
                idx.append(f'{os.path.splitext(file_name)[0]}_{i_structure}')
                Energy_list.append(ener)
                Cell_list.append(cell)
                Coords_list.append(coords)
                Elem_list.append(elements)
                Number_list.append(numbers)
                Fixed_list.append(fixed)
                Forces_list.append(forces)

                line_now += 2 + n_atom
                i_structure += 1

            return idx, Cell_list, Energy_list, Forces_list, Elem_list, Number_list, Coords_list, Fixed_list, ['C'] * i_structure

        except Exception as err:
            warnings.warn(f'An Error occurred when reading file {file_name}, skipped.\nError: {err}.')
            return [], [], [], [], [], [], [], [], []

    def read(
            self,
            file_list = None,
            n_core: int = -1,
            backend: Literal['loky', 'threading', 'multiprocessing']|str = 'loky',
            lattice_tag: str = 'lattice',
            energy_tag: str = 'energy',
            column_info_tag: str = 'properties',
            element_tag: str = 'species',
            coordinates_tag: str = 'pos',
            forces_tag: str | None = 'forces',
            fixed_atom_tag: str | None = 'move_mask'
    ):
        """
        Read *.extxyz files to BatchStructures.
        Args:
            file_list: List[str], the list of files to read. Default for all files under given path.
            n_core: int, the number of CPU cores used in reading files. -1 for all cores.
            backend: backend of parallel reading in `joblib`.
            lattice_tag: key-words of lattice information in *.extxyz file, at each structure's 2nd line.
            energy_tag: key-words of structure's energy in *.extxyz file.
            column_info_tag: key-words of column content information in *.extxyz file. The column content has such format:
                             `content 1`:`data type 1`:`column occupied number 1`:`content 2`:`data type 2`:`column occupied number 2`: ...
                             detailed see https://wiki.fysik.dtu.dk/ase/dev/_modules/ase/io/extxyz.html.
            element_tag: As a content in `column_info_tag`, name of element column.
            coordinates_tag: As a content in `column_info_tag`, name of atom coordinates column.
            forces_tag: As a content in `column_info_tag`, name of forces column. If the file does not contain forces, set it to None.
            fixed_atom_tag: As a content in `column_info_tag`, name of the mask column specified which atom was fixed.

        Returns: None
        All data would update as class attributes.
        """
        t_st = time.perf_counter()
        # file check
        if file_list is None:
            file_list = os.listdir(self.path)
            file_list = [f_ for f_ in file_list if os.path.isfile(os.path.join(self.path, f_))]
        elif not isinstance(file_list, Sequence):
            raise TypeError(f'Invalid type of files_list: {type(file_list)}')
        # para. set
        if n_core > len(file_list):
            warnings.warn(f'`n_core` is greater than file numbers, so `n_core` was reset to {len(file_list)}', RuntimeWarning)
            n_core = len(file_list)
        elif n_core == -1:
            n_core = jb.cpu_count(only_physical_cores=True)
        elif (not isinstance(n_core, int)) or n_core < -1:
            raise ValueError(f'Invalid `n_core` number: {n_core}.')
        # search pattern
        self.CELL_PARTTEN = re.compile(rf'{lattice_tag}\s?=\s?\"([-+E0-9.\s]+)\"', re.IGNORECASE)
        self.PROP_PARTTEN = re.compile(rf'{column_info_tag}\s?=\s?([a-zA-Z:0-9_]+)', re.IGNORECASE)
        self.ENER_PARTTEN = re.compile(rf'{energy_tag}\s?=\s?([-+.0-9e]+)')

        _para = jb.Parallel(n_jobs=n_core, backend=backend, verbose=self.verbose)
        _dat = _para(jb.delayed(self._read_single)(
            fi_name,
            lattice_tag,
            energy_tag,
            column_info_tag,
            element_tag,
            coordinates_tag,
            forces_tag,
            fixed_atom_tag
        ) for fi_name in file_list)
        # idx, Cell_list, Energy_list, Forces_list, Elem_list, Number_list, Coords_list, Coord_type
        for temp in _dat:
            self._Sample_ids.extend(temp[0])
            self.Cells.extend(temp[1])
            self.Energies.extend(temp[2])
            self.Forces.extend(temp[3])
            self.Elements.extend(temp[4])
            self.Numbers.extend(temp[5])
            self.Coords.extend(temp[6])
            self.Fixed.extend(temp[7])
            self.Coords_type.extend(temp[8])
        if not self._Sample_ids: raise RuntimeError('All data are empty.')
        self._update_indices()
        self._check_id()
        self._check_len()
        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - t_st:>5.4f}')


class Cif2Feat(BatchStructures):
    """
    Read CIF file with multiple structures.
    Notes: CIF file could not contain the fixation information, so all atoms would be free.

    """

    def __init__(self, path: str, verbose: int = 1):
        super().__init__()
        self.verbose = verbose
        self.path = path
        self.Energies = None
        self.Forces = None

    def read(
            self,
            file_list = None,
            n_core: int = -1,
            backend: Literal['loky', 'threading', 'multiprocessing'] | str = 'loky',
    ):
        t_st = time.perf_counter()
        # file check
        if file_list is None:
            file_list = os.listdir(self.path)
            file_list = [f_ for f_ in file_list if os.path.isfile(os.path.join(self.path, f_))]
        elif not isinstance(file_list, Sequence):
            raise TypeError(f'Invalid type of files_list: {type(file_list)}')
        # para. set
        if n_core > len(file_list):
            warnings.warn(f'`n_core` is greater than file numbers, so `n_core` was reset to {len(file_list)}', RuntimeWarning)
            n_core = len(file_list)
        elif n_core == -1:
            n_core = jb.cpu_count(only_physical_cores=True)
        elif (not isinstance(n_core, int)) or n_core < -1:
            raise ValueError(f'Invalid `n_core` number: {n_core}.')

        _para = jb.Parallel(n_jobs=n_core, backend=backend, verbose=self.verbose)
        _dat = _para(jb.delayed(self._read_single)(fi_name) for fi_name in file_list)
        # file_name, cell, elements, elem_numbers, coo,
        for temp in _dat:
            if temp is not None:
                self._Sample_ids.append(temp[0])
                self.Cells.append(temp[1])
                self.Elements.append(temp[2])
                self.Numbers.append(temp[3])
                self.Coords.append(temp[4])
                self.Fixed.append(temp[5])
        if not self._Sample_ids: raise RuntimeError('All data are empty.')
        self.Coords_type = ['D'] * len(self._Sample_ids)
        self._update_indices()
        self._check_id()
        self._check_len()

        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - t_st:>5.4f}')

    def _read_single(
            self,
            file_name: str,
    ):
        if not os.path.isfile(os.path.join(self.path, file_name)):
            warnings.warn(f'No OUTCAR file in given directory {os.path.join(self.path, file_name)}')
            return None

        try:
            with open(os.path.join(self.path, file_name), 'r') as _f:
                data = _f.readlines()

            # manage information
            cell_info_dict = dict()
            atom_info_seq_dict = dict()
            coords_info_list = list()
            i_seq = 0  # serial numbers of atom information
            is_read_atom = False  # detect if reading atom coo. info.
            for dat in data[1:]:
                _data_list = dat.split()
                if _data_list[0] == 'loop_': is_read_atom = False
                if is_read_atom and _data_list[0][0] != '_':
                    coords_info_list.append(_data_list)
                elif _data_list[0][:5] == '_cell':
                    cell_info_dict[_data_list[0]] = _data_list[1]
                elif _data_list[0][:5] == '_atom':
                    atom_info_seq_dict[_data_list[0]] = i_seq
                    i_seq = i_seq + 1
                    is_read_atom = True

            # manage cell vectors
            a = float(cell_info_dict['_cell_length_a'])
            b = float(cell_info_dict['_cell_length_b'])
            c = float(cell_info_dict['_cell_length_c'])
            alpha = np.deg2rad(float(cell_info_dict['_cell_angle_alpha']))
            beta  = np.deg2rad(float(cell_info_dict['_cell_angle_beta']))
            gamma = np.deg2rad(float(cell_info_dict['_cell_angle_gamma']))
            b1 = b * np.cos(gamma)
            b2 = np.sqrt(b**2 - b1**2)
            c1 = c * np.cos(beta)
            c2 = (b * c * np.cos(alpha) - b1 * c1)/b2
            c3 = np.sqrt(c**2 - c1**2 - c2**2)
            cell = np.array(
                [[a, 0., 0.],
                 [b1, b2, 0.],
                 [c1, c2, c3]],
                dtype=np.float32
            )

            # manage coordinates
            coords_info_list = np.asarray(coords_info_list)
            ix, iy, iz = atom_info_seq_dict['_atom_site_fract_x'], atom_info_seq_dict['_atom_site_fract_y'], atom_info_seq_dict['_atom_site_fract_z']
            coo = coords_info_list[:, [ix, iy, iz]].astype(np.float32)
            fix = np.ones_like(coo, dtype=np.int8)
            atom_list = coords_info_list[:, atom_info_seq_dict['_atom_site_type_symbol']].tolist()
            # elements
            _elem_old = atom_list[0]
            elements = [atom_list[0], ]
            elem_numbers = list()
            count_i = 0
            for elem in atom_list:
                if elem == _elem_old:
                    count_i += 1
                else:
                    _elem_old = elem
                    elem_numbers.append(count_i)
                    elements.append(elem)
                    count_i = 1
            elem_numbers.append(count_i)

            return file_name, cell, elements, elem_numbers, coo, fix

        except Exception as err:
            warnings.warn(f'An Error occurred when reading file {file_name}, skipped.\nError: {err}.')
            return None


class ASETraj2Feat(BatchStructures):
    """
    read ASE files to BatchStructures.
    """

    def __init__(self, path: str, verbose: int = 1) -> None:
        super().__init__()
        ase_io = check_module('ase.io')
        if ase_io is None:
            raise ImportError(f'`ASETraj2Feat` requires the ase package which is not found.')
        self.ase_io = ase_io
        self.path = path
        self.verbose = verbose
        self.Energies = list()
        self.Forces = list()

    def _read_single(self, ase_file):
        try:
            idx = list()
            Cell_list = list()
            Energy_list = list()
            Elem_list = list()
            Number_list = list()
            Coords_list = list()
            Coords_type = list()
            Forces_list = list()
            Fixed_list = list()
            atom_list = self.ase_io.read(ase_file, ':')
            base_name = os.path.basename(ase_file)
            base_name, _ = os.path.splitext(base_name)  # remove postfix
            for i, atm in enumerate(atom_list):
                if getattr(atm, 'calc', None) is None:  # skip the energy-empty sample
                    continue
                Cell_list.append(np.asarray(atm.cell, dtype=np.float32))
                Coords_list.append(atm.positions.astype(np.float32))
                coord_type = 'D' if np.all(np.abs(atm.positions) <= 1.) else 'C'
                Coords_type.append(coord_type)
                Energy_list.append(atm.get_potential_energy())
                Forces_list.append(np.asarray(atm.get_forces(), dtype=np.float32))
                fix = np.ones_like(atm.positions, dtype=np.int8)
                fix[atm.constraints[0].index] = 0
                Fixed_list.append(fix)
                elem_sym, atomic_num, atom_count = elem_list_reduce(atm.numbers)
                Elem_list.append(elem_sym)
                Number_list.append(atom_count)
                idx.append(f'{base_name}_{i}')

            return idx, Cell_list, Energy_list, Forces_list, Elem_list, Number_list, Coords_list, Coords_type, Fixed_list

        except Exception as err:
            warnings.warn(f'An Error occurred when reading file {ase_file}, skipped.\nError: {err}.')
            return [], [], [], [], [], [], [], [], []

    def read(
            self,
            file_list = None,
            n_core: int = -1,
            backend: Literal['loky', 'threading', 'multiprocessing'] | str = 'loky',
    ):
        t_st = time.perf_counter()
        # file check
        if file_list is None:
            file_list = os.listdir(self.path)
            file_list = [f_ for f_ in file_list if os.path.isfile(os.path.join(self.path, f_))]
        elif not isinstance(file_list, Sequence):
            raise TypeError(f'Invalid type of files_list: {type(file_list)}')
        # para. set
        if n_core > len(file_list):
            warnings.warn(f'`n_core` is greater than file numbers, so `n_core` was reset to {len(file_list)}', RuntimeWarning)
            n_core = len(file_list)
        elif n_core == -1:
            n_core = jb.cpu_count(only_physical_cores=True)
        elif (not isinstance(n_core, int)) or n_core < -1:
            raise ValueError(f'Invalid `n_core` number: {n_core}.')

        _para = jb.Parallel(n_jobs=n_core, backend=backend, verbose=self.verbose)
        _dat = _para(jb.delayed(self._read_single)(os.path.join(self.path, fi_name)) for fi_name in file_list)
        # file_name, cell, elements, elem_numbers, coo,
        # idx, Cell_list, Energy_list, Forces_list, Elem_list, Number_list, Coords_list, Coord_type, fixed
        for temp in _dat:
            self._Sample_ids.extend(temp[0])
            self.Cells.extend(temp[1])
            self.Energies.extend(temp[2])
            self.Forces.extend(temp[3])
            self.Elements.extend(temp[4])
            self.Numbers.extend(temp[5])
            self.Coords.extend(temp[6])
            self.Coords_type.extend(temp[7])
            self.Fixed.extend(temp[8])
        if not self._Sample_ids: raise RuntimeError('All data are empty.')
        self._update_indices()
        self._check_id()
        self._check_len()

        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - t_st:>5.4f}')

class Output2Feat(BatchStructures):
    """
    A class of post-processing to reading output files of BatchOptimization and BatchMD
    """
    def __init__(self, path: str, verbose: int = 1) -> None:
        super().__init__()
        self.path = path
        self.verbose = verbose
        self.Energies = list()
        self.Forces = list()

    def read(self, file_list: List[str], n_core: int = -1, backend='loky'):
        t_st = time.perf_counter()
        # file check
        if file_list is None:
            file_list = os.listdir(self.path)
            file_list = [f_ for f_ in file_list if os.path.isfile(os.path.join(self.path, f_))]
        elif not isinstance(file_list, Sequence):
            raise TypeError(f'Invalid type of files_list: {type(file_list)}')
        # para. set
        if n_core > len(file_list):
            warnings.warn(f'`n_core` is greater than file numbers, so `n_core` was reset to {len(file_list)}', RuntimeWarning)
            n_core = len(file_list)
        elif n_core == -1:
            n_core = min(jb.cpu_count(only_physical_cores=True), len(file_list))
        elif (not isinstance(n_core, int)) or n_core < -1:
            raise ValueError(f'Invalid `n_core` number: {n_core}.')

        _para = jb.Parallel(n_jobs=n_core, backend=backend, verbose=self.verbose)
        _dat = _para(jb.delayed(self._read_single)(fi_name) for fi_name in file_list)
        # file_name, cell, elements, elem_numbers, coo,
        # samp_id, cell, elements, elem_numbers, energies, forces, coo, fix
        is_no_force = False
        for temp in _dat:
            self._Sample_ids.extend(temp[0])
            self.Cells.extend(temp[1])
            self.Elements.extend(temp[2])
            self.Numbers.extend(temp[3])
            self.Energies.extend(temp[4])
            if len(temp[5]) != 0:
                self.Forces.extend(temp[5])
            else:
                is_no_force = True
                warnings.warn(f'Forces are not output in some files, so that all forces are set to None.')
            self.Coords.extend(temp[6])
            self.Fixed.extend(temp[7])
        if is_no_force:
            self.Forces = None
        self.Coords_type = ['C'] * len(self)
        if not self._Sample_ids: raise RuntimeError('All data are empty.')
        self._update_indices()
        self._check_id()
        self._check_len()

        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - t_st:>5.4f}')
        pass

    def _read_single(self, file_name):
        if not os.path.isfile(os.path.join(self.path, file_name)):
            warnings.warn(f'No output file in given directory {os.path.join(self.path, file_name)}')
            return None
        samp_id = list()
        cell = list()
        elements = list()
        elem_numbers = list()
        energies = list()
        forces = list()
        coo = list()
        fix = list()
        number_pattern = re.compile(r'[\s\-.0-9]+')
        try:
            with open(os.path.join(self.path, file_name), 'r') as _f:
                data = _f.readlines()
            # identify tasks
            assert len(data) > 0, f'Empty data occurred.'
            task_tag = None
            is_task_opt = None
            is_task_md = None
            for line in data:
                tmp = re.search(r'VERBOSITY LEVEL: (.*)$', line)
                if tmp is not None:
                    data_verbosity = int(tmp.group(1))
                    continue
                tmp = re.search(r'TASK: Structure Optimization <<', line)
                if tmp is not None:
                    is_task_opt = tmp
                    continue
                tmp = re.search(r'TASK: Molecular Dynamics <<', line)
                if tmp is not None:
                    is_task_md = tmp
                    continue
                EOF = re.match(r'^ENTERING MAIN', line)  # match the end of head information
                if EOF is not None:
                    break
            if is_task_opt is not None:
                task_tag = 'opt'
                Id_pattern = re.compile(r'Structure names:\s\[(.+)]')
                Elem_tag = re.compile(r'Structure\s*[0-9]+:\s*(.*)$')
                Iter_pattern = re.compile(r'ITERATION\s*([0-9]+)')
                E_pattern = re.compile(r'\s*Energies: +\[([+\s0-9e.-]+)]')
                F_tag = re.compile(r'\s*Forces:')
                Coo_tag = re.compile(r'\s*Coordinates:')
                Cell_tag = re.compile(r'Cell Vectors:')
                Final_Coo_tag = re.compile(r'^ Final Coordinates:')
                Final_F_tag = re.compile(r'^ Final Forces:')
            elif is_task_md is not None:
                task_tag = 'md'
                raise NotImplementedError
            if task_tag is None: raise RuntimeError(f'Unknown task type.')

            i_batch = -1
            # MAIN LOOP
            for i, line in enumerate(data):
                is_name = re.match(Id_pattern, line)
                if is_name is not None:  # it is the name line
                    name = re.split(', *', is_name.group(1).replace('\'', ''))
                    i_batch += 1
                    n_batch = len(name)
                    cell_temp = list()
                    elements_temp = list()
                    elem_numbers_temp = list()
                    continue
                # cell
                is_cell = re.match(Cell_tag, line)
                if is_cell is not None:
                    _cell = list()
                    for _sub_line in data[i+1:]:
                        is_empty = re.match(r'^\s*$', _sub_line)
                        if is_empty is not None:
                            cell_temp.append(np.asarray(_cell, dtype=np.float32))
                            _cell = list()
                            continue
                        is_numbers = re.match(number_pattern, _sub_line)
                        if is_numbers is not None:
                            _numbers = is_numbers.group().split()
                            _cell.append(_numbers)
                        else:
                            break  # neither empty line nor numbers, thus skipping
                    continue
                # element & numbers
                is_elem = re.match(Elem_tag, line)
                if is_elem is not None:
                    _elem = list()
                    _number = list()
                    for ii, _ in enumerate(is_elem.group(1).replace(':', '').split()):
                        if ii % 2 == 0:
                            _elem.append(_)
                        else:
                            _number.append(int(_))
                    elements_temp.append(_elem)
                    elem_numbers_temp.append(_number)
                    continue
                # Main iterations. Here each iteration would repeat the global information, e.g., cell, elements, numbers, etc.
                if data_verbosity > 1:  # for verbosity == 1, only the last coords. will be output
                    is_in_iter = re.match(Iter_pattern, line)
                    if is_in_iter is not None:
                        _samp_id = is_in_iter.group(1)
                        for ii in range(n_batch):
                            samp_id.append(f'{file_name}_{name[ii]}_iter{_samp_id}')
                            cell.append(cell_temp[-n_batch + ii])
                            elements.append(elements_temp[-n_batch + ii])
                            elem_numbers.append(elem_numbers_temp[-n_batch + ii])
                        continue
                    # energy
                    is_energy = re.match(E_pattern, line)
                    if is_energy is not None:
                        _ener = is_energy.group(1).split()
                        for _ in _ener:
                            energies.append(float(_))
                        continue
                    # coords
                    is_coo = re.match(Coo_tag, line)
                    if is_coo is not None:
                        _coo = list()
                        for _sub_line in data[i+2:]:
                            is_empty = re.match(r'^\s*$', _sub_line)
                            if is_empty is not None:
                                if len(_coo[0]) == 0: continue  # avoid multiple empty lines are read as coordinates
                                _coo = np.asarray(_coo, dtype=np.float32)
                                coo.append(_coo)
                                fix.append(np.ones_like(_coo, dtype=np.int8))
                                _coo = list()
                                continue
                            is_numbers = re.match(number_pattern, _sub_line)
                            if is_numbers is not None:
                                _numbers = is_numbers.group().split()
                                _coo.append(_numbers)
                            else:
                                break  # neither empty line nor numbers, thus skipping
                        continue
                    # forces
                    is_forc = re.match(F_tag, line)
                    if is_forc is not None:
                        _forc = list()
                        for _sub_line in data[i + 2:]:
                            is_empty = re.match(r'^\s*$', _sub_line)
                            if is_empty is not None:
                                forces.append(np.asarray(_forc, dtype=np.float32))
                                _forc = list()
                                continue
                            is_numbers = re.match(number_pattern, _sub_line)
                            if is_numbers is not None:
                                _numbers = is_numbers.group().split()
                                _forc.append(_numbers)
                            else:
                                break  # neither empty line nor numbers, thus skipping
                        continue
                else:  # only got the final coords & forces
                    # energy
                    is_energy = re.match(E_pattern, line)
                    if is_energy is not None:
                        _ener = is_energy.group(1).split()
                    is_final = re.match(Final_Coo_tag, line)
                    if is_final is not None:
                        for ii in range(n_batch):
                            samp_id.append(f'{file_name}_{name[ii]}')
                            cell.append(cell_temp[-n_batch + ii])
                            elements.append(elements_temp[-n_batch + ii])
                            elem_numbers.append(elem_numbers_temp[-n_batch + ii])
                            energies.append(float(_ener[ii]))
                        _coo = list()
                        for _sub_line in data[i + 2:]:
                            is_empty = re.match(r'^\s*$', _sub_line)
                            if is_empty is not None:
                                if len(_coo[0]) == 0: continue  # avoid multiple empty lines are read as coordinates
                                _coo = np.asarray(_coo, dtype=np.float32)
                                coo.append(_coo)
                                fix.append(np.ones_like(_coo, dtype=np.int8))
                                _coo = list()
                                continue
                            is_numbers = re.match(number_pattern, _sub_line)
                            if is_numbers is not None:
                                _numbers = is_numbers.group().split()
                                _coo.append(_numbers)
                            else:
                                break  # neither empty line nor numbers, thus skipping
                        continue

                    is_forc = re.match(Final_F_tag, line)
                    if is_forc is not None:
                        _forc = list()
                        for _sub_line in data[i + 2:]:
                            is_empty = re.match(r'^\s*$', _sub_line)
                            if is_empty is not None:
                                forces.append(np.asarray(_forc, dtype=np.float32))
                                _forc = list()
                                continue
                            is_numbers = re.match(number_pattern, _sub_line)
                            if is_numbers is not None:
                                _numbers = is_numbers.group().split()
                                _forc.append(_numbers)
                            else:
                                break  # neither empty line nor numbers, thus skipping
                        continue

            return samp_id, cell, elements, elem_numbers, energies, forces, coo, fix

        except Exception as err:
            warnings.warn(f'An Error occurred when reading file {file_name}, skipped.\nError: {err}.')
            return None
