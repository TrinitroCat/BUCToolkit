r"""
Methods of reading and transform various files
"""
import re
from typing import Any, Dict, List, Sequence, Set, Tuple, Optional
import time
import os
import copy
import warnings
import joblib as jb
import numpy as np

from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures

''' CONSTANCE '''
_ALL_ELEMENT = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
                'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', ]  # element List

''' Load labels from local file '''


def load_from_csv(file_name: str,
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
        energy_dict = {cont[col1]: label_type(cont[col2]) if not (cont[col2] in {'None', 'nan', ''})
        else None
                       for cont in f}  # a dict of {sample_id: energy}

    return energy_dict


class POSCARs2Feat(BatchStructures):
    """
    Read and convert a folder of POSCAR files from given path into arrays of atoms, coordinates, cell vectors etc.
    """

    def __init__(self, path: str = './', verbose: int = 0, *args, **kwargs) -> None:
        """
        Read and convert a folder of POSCAR files from given path into arrays of atoms, coordinates, cell vectors etc.

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
            Numbers: list, element numbers of crystals. Shape: List[List[int(atom number of each element)]]. the same order of Elements.

        Returns: None
        """
        super().__init__()
        self.path = path
        self.verbose = verbose

    def para_read(self,
                  file_list: Optional[List] = None,
                  output_coord_type: str = 'cartesian',
                  n_core: int = -1,
                  backend='loky'):
        r"""
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

        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - time_st:<5.4f}')

    def read(self, file_list: Optional[List] = None, output_coord_type: str = 'cartesian', ) -> None:
        time_st = time.perf_counter()

        Z_dict = {key: i for i, key in enumerate(_ALL_ELEMENT, 1)}  # a dictionary which map element symbols into their atomic numbers.
        self._Z_dict = Z_dict

        # loading files
        if self.verbose > 0: print('*' * 60 + '\nReading files...\n')
        if file_list is None:
            self.files_list = os.listdir(self.path)
        else:
            self.files_list = file_list
        self.n_samp = len(self.files_list)

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
        if self.verbose > 0: print('\nAll files were read successfully!\n' + '*' * 60)

        time_ed = time.perf_counter()
        if self.verbose > 0: print(f'Total time: {(time_ed - time_st):<5.4f}')
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

        if self.Coords_type == in_coord_type:
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

        with open(full_path, "r") as file:  #打开文件
            data = file.read()  #读取文件
        n_atom = re.search(self.__n_atom_partt, data)  #从OUTCAR文件搜索原子数
        if n_atom is None: raise RuntimeError('No atoms matched.')
        n_atom = int(n_atom.group())  #将输出的原子个数保存为整数形式
        position_iter = re.finditer(self.__pos_force_partt, data)  #通过re匹配的迭代器函数，找到原子坐标
        energies = re.findall(self.__energy_partt, data)

        # ATOM & Number
        atoms, numbers = self._get_atom_info(data)
        atoms = np.array(atoms, dtype='<U4')
        numbers = np.array(numbers, dtype=np.int32)
        # Cell Vectors
        cells = re.findall(self.__cell_partt, data, )
        cells = [_cell.split() for _cell in cells]
        cells = np.array(cells, dtype=np.float32).reshape(-1, 3, 6)
        cells = cells[:, :, :3]

        _data = list()
        energies = [float(_en) for _en in energies]
        for match_for in position_iter:  #循环的取每次迭代找到的一个结构的原子坐标与受力信息
            _dat = re.split(r"\n+", match_for.group())  #通过换行符进行划分
            _dat = [re.split(r'[0-9][\n\s\t]+', dat_) for dat_ in _dat[:-2]]  #去除空字符或都是横线的行，然后循环的对每一行进行划分 <<< # TODO
            _data.append(_dat)  #将原子坐标的列表添加进列表中

        _data = np.array(_data, dtype=np.float32)  # (n_step, n_atom, 3)
        if len(_data) == 0:
            warnings.warn(f'Occurred empty data in file {file_name}, skipped.', RuntimeWarning)
            if parallel:
                return [], [], [], [], [], []
            else:
                return

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
            return _id, atoms, numbers, cells, coords, energies, forces, fixed
        else:
            self._Sample_ids.extend(_id)
            self.Elements.extend(atoms.tolist())
            self.Numbers.extend(numbers.tolist())
            self.Cells.extend(cells)
            self.Coords.extend(coords)
            self.Energies.extend(energies)
            self.Forces.extend(forces)
            self.Fixed.extend(fixed)

    def read_files(self, file_list: Optional[List[str]] = None, n_core: int = -1, backend: str = 'loky'):
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

        if n_core == 1:
            if self.verbose: print('Sequential Reading...');print('Progress: 0%', end='\r')
            for i, fi_name in enumerate(file_list):
                path_ = os.path.join(self.path, fi_name)
                self._read_single_file(path_, parallel=False)
                if self.verbose > 0:
                    if (i + 1) % 50 == 0:
                        prog_ = (i + 1) / len(file_list)
                        prog = round(prog_ / 0.05)
                        print('Progress: ' + '>' * prog + f'{prog_ * 100:>3.2f}%', end='\r')
            if self.verbose: print('Progress: ' + '>' * 20 + f'{100:>3d}%')
        else:
            _para = jb.Parallel(n_jobs=n_core, backend=backend, verbose=self.verbose)
            _dat = _para(jb.delayed(self._read_single_file)(fi_name, parallel=True) for fi_name in file_list)
            if (_dat is None) or (_dat[0] == []): raise RuntimeError('Occurred empty data.')
            for temp in _dat:
                self._Sample_ids.extend(temp[0])
                self.Elements.extend(temp[1])
                self.Numbers.extend(temp[2])
                self.Cells.extend(temp[3])
                self.Coords.extend(temp[4])
                self.Energies.extend(temp[5])
                self.Forces.extend(temp[6])
                self.Fixed.extend(temp[7])
        if self.verbose > 0: print(f'Done. Total Time: {time.perf_counter() - t_st:>5.4f}')


class ASEDB2Feat(BatchStructures):
    """
    read ASE_db files to BatchStructures.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        from ase import io
        self.ase_io = io
        self.Energies = list()
        self.Forces = list()

        atom_list = io.read(path)
        # TODO


if __name__ == '__main__':
    tst = time.perf_counter()
    f = OUTCAR2Feat('/home/ppx/PythonProjects/DataBases/PropDehydro/OUTCARs', verbose=1)
    f.read_files(n_core=6, )
    print(f'TIME: {-tst + time.perf_counter()}')
    del f
    tst = time.perf_counter()
    f = OUTCAR2Feat('/home/ppx/PythonProjects/DataBases/PropDehydro/OUTCARs', verbose=1)
    f.read_files(n_core=1, )
    print(f'TIME: {-tst + time.perf_counter()}')
    pass
