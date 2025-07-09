#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: write_files.py
#  Environment: Python 3.12

import os
import warnings
from typing import Sequence, List, Literal, Tuple
import joblib as jb
import numpy as np

# from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures
from BM4Ckit.utils._print_formatter import STRING_ARRAY_FORMAT, AS_PRINT_COORDS, FLOAT_ARRAY_FORMAT


class WritePOSCARs:
    """

    """

    def __init__(
            self,
            cells: Sequence[Sequence],
            coords: Sequence[np.ndarray],
            atom_labels: Sequence[Sequence[str]],
            atom_numbers: Sequence[Sequence[int]],
            fixed: Sequence[np.ndarray] | None = None,
            output_path: str = './',
            file_name_list: str | Sequence[str] = 'POSCAR',
            system_list: str | Sequence[str] = 'untitled',
            coord_type: List[Literal['C', 'D']] | Literal['C', 'D'] = 'C',
            ncore: int = -1
    ) -> None:
        """
        Convert coordinates matrices to POSCAR format, and write files to the "output_path".

        Args:
            cells: List|ndarray, a batch of lattice vectors. shape: (n_batch, 3, 3)
            atom_labels: 2D list|ndarray of str, a batch of list of element symbols. shape: (n_batch, n_atom)
            atom_numbers:2D list|ndarray of int, list of atom number of each element, in the order of atom_label. shape: (n_batch, n_atom)
            coords: list|ndarray, the batch of atoms coordinates, in the order of atom_label. shape: (n_batch, n_atom, 3)
            fixed: list|ndarray|None, the batch of atoms fixed directions, in the order of atom_label. shape: (n_batch, n_atom, 3), dtype: int.
                `0` for fixed, `1` for free. if None, `fixed` would fill with `1`.
            output_path: str, the output path.
            file_name_list: list(str), the list of file names.
            system_list: str|list(str), the 1st line of output file i.e., the annotation or title of the file.
            coord_type: List[Literal['C', 'D']]|Literal['C', 'D'], 'D' or 'C', which means whether the input coordinates are "Direct" or "Cartesian".
                If only a string, `coord_type` would set to be '[`coord_type`] * n_batch'.
            ncore: int, the number of CPU cores to write files in parallel.

        Returns: None
        """
        # check vars
        n_batch = len(cells)
        if isinstance(file_name_list, str):
            file_name_list = [file_name_list + str(i) for i in range(n_batch)]
        if isinstance(system_list, str):
            system_list = [system_list] * n_batch
        if not (n_batch == len(coords) == len(atom_labels) == len(atom_numbers)):
            raise ValueError(f'number of cells in cell vector, atom coordinates, atom labels, and atom_numbers should be the same,\
                              but occurred {n_batch, len(coords), len(atom_labels), len(atom_numbers)}')
        if isinstance(coord_type, str):
            coord_type = [coord_type] * n_batch
        elif (not isinstance(coord_type, (List, Tuple))) or (len(coord_type) != n_batch):
            raise ValueError(f'Invalid value of `coord_type`: type: {type(coord_type)}, length: {len(coord_type)}.'
                             f' It should be type: List | Tuple, length: {n_batch}')
        if fixed is None:
            fixed = [np.full_like(_, 1, dtype=np.int8) for _ in coords]
        # check len
        if isinstance(file_name_list, Sequence):
            if not (len(file_name_list) == len(cells) == len(atom_labels) == len(atom_numbers) == len(coords) == len(system_list) == len(fixed)):
                raise ValueError(
                    f'Inconsistent length of inputs.'
                    f'length of file_name_list, cells, atom_labels, atom_numbers, coords, system_list, fixed: '
                    f'{
                    (len(file_name_list), len(cells), len(atom_labels), len(atom_numbers), len(coords), len(system_list), len(fixed))
                    }'
                )
        # check parallel
        tot_cores = jb.cpu_count()
        if not isinstance(ncore, int):
            raise TypeError(f'`ncore` must be an integer, but occurred {type(ncore)}.')
        elif ncore == -1:
            ncore = tot_cores
        elif ncore > tot_cores:
            warnings.warn('Input `ncore` is greater than total CPU cores and was set to total CPU cores automatically.', RuntimeWarning)
            ncore = tot_cores

        if ncore != 1:
            _para = jb.Parallel(ncore, backend="threading")
            _para(
                jb.delayed(self.__write)(
                    cell,
                    coords[i],
                    fixed[i],
                    atom_labels[i],
                    atom_numbers[i],
                    output_path=output_path,
                    file_name=file_name_list[i],
                    system=system_list[i],
                    coord_type=coord_type[i]
                )
                for i, cell in enumerate(cells)
            )

        else:
            for i in range(n_batch):
                self.__write(
                    cells[i],
                    coords[i],
                    fixed[i],
                    atom_labels[i],
                    atom_numbers[i],
                    output_path=output_path,
                    file_name=file_name_list[i],
                    system=system_list[i],
                    coord_type=coord_type[i]
                )

        pass

    @staticmethod
    def __write(cell: Sequence,
                coord: np.ndarray,
                fixed: np.ndarray,
                atom_label: Sequence,
                atom_number: Sequence,
                output_path: str,
                file_name: str,
                system: str,
                coord_type: str) -> None:
        """
        Convert coordinates matrix to POSCAR format, and write a file to the output_path.

        Args:
            cell: list|ndarray, the lattice vector of cell.
            atom_label:1D list|ndarray of str, list of element symbols.
            atom_number:1D list|ndarray of int, list of atom number of each element, in the order of atom_label.
            coord: list|ndarray, the coordinates of atoms, in the order of atom_label.
            fixed: list|ndarray, the fixed direction of atoms.
            output_path: str, the output path.
            file_name: str, the file name.
            system: str, the 1st line of an output file i.e., the annotation or title of the file.
            coord_type: str, 'D' or 'C', which means whether the coordinates are "Direct" or "Cartesian".

        Return: None
        """
        # check vars
        if not isinstance(cell, (list, np.ndarray)):
            raise TypeError(f'Unknown type of cell, type : {type(cell)}')
        if isinstance(coord, List):
            coord = np.asarray(coord, dtype=np.float32)
        elif not isinstance(coord, np.ndarray):
            raise TypeError(f'Unknown type of coord, type : {type(coord)}')
        if not (isinstance(output_path, str) and isinstance(file_name, str) and isinstance(system, str)):
            raise TypeError(f'output_path||file_name||system must be strings, '
                            f'but got type {type(output_path)}||{type(file_name)}||{type(system)}.')
        elif coord_type != 'C' and coord_type != 'D':
            raise ValueError(f'Unknown coord_type : "{coord_type}"')

        # main
        with open(os.path.join(output_path, file_name), 'w') as POSCAR:
            POSCAR.write(system)
            POSCAR.write('\n    1\n')
            # cell
            for vx in cell:
                for xx in vx:
                    POSCAR.write(f'    {xx:0< 14.8f}')
                POSCAR.write('\n')
            # atom element
            for label in atom_label:
                POSCAR.write(f' {label: <6s}')
            POSCAR.write('\n')
            for label in atom_number:
                POSCAR.write(f' {label: <6d}')
            POSCAR.write('\n')
            # selective dynamics
            POSCAR.write('Selective Dynamics\n')
            if coord_type == 'C':
                POSCAR.write('Cartesian\n')
            elif coord_type == 'D':
                POSCAR.write('Direct\n')
            else: raise ValueError(f'Invalid `coord_type`: {coord_type}')
            # atom coordinates
            fixed = np.where(fixed == 1, 'T', 'F')
            coord_str = AS_PRINT_COORDS(coord[:, :3])
            print_arr = np.concatenate((coord_str, fixed[:, :3]), axis=1)
            print_str = np.array2string(print_arr, **STRING_ARRAY_FORMAT).replace('[', ' ').replace(']', ' ')
            POSCAR.write(print_str)
            POSCAR.write('\n')


class Write2JDFTX:

    def __init__(self, ):
        pass

    def write(self,
              batch_structures,
              output_path: str = './',
              file_name_list: str | Sequence[str] = 'POSCAR',
              system_list: str | Sequence[str] = 'untitled',
              coord_type: str = 'C') -> None:
        """
        Convert coordinates matrices to POSCAR format, and write files to the "output_path".

        Parameters:
            batch_structures: batch of structures.
            output_path: str, the output path.
            file_name_list: list(str), the list of file names.
            system_list: str|list(str), the 1st line of output file i.e., the annotation or title of the file.
            coord_type: str, 'D' or 'C', which means whether the coordinates is "Direct" or "Cartesian".

        Returns: None
        """
        batch_structures.generate_atom_list()
        n_batch = len(batch_structures.Cells)
        if isinstance(file_name_list, str):
            file_name_list = [file_name_list + str(i) for i in range(n_batch)]
        if isinstance(system_list, str):
            system_list = [system_list] * n_batch
        if not (n_batch == len(batch_structures.Coords) == len(batch_structures.Elements) == len(batch_structures.Numbers)):
            raise ValueError(f'number of cells in cell vector, atom coordinates, atom labels, and atom_numbers should be the same, '
                             f'but occurred {n_batch, len(batch_structures.Coords), len(batch_structures.Elements), len(batch_structures.Numbers)}')

        for i in range(n_batch):
            self.__write(batch_structures.Cells[i],
                         batch_structures.Coords_type[i],
                         batch_structures.Coords[i],
                         batch_structures.Fixed[i],
                         batch_structures.Atom_list[i],
                         output_path=output_path,
                         file_name=file_name_list[i],
                         system=system_list[i],
                         out_coord_type=coord_type)

        pass

    @staticmethod
    def __write(cell: Sequence,
                in_coord_type: str,
                coord: np.ndarray,
                fixed: np.ndarray,
                atom_list: List,
                output_path: str,
                file_name: str,
                system: str,
                out_coord_type: str,
                ) -> None:
        """
        Convert coordinates matrix to POSCAR format, and write a file to the output_path.

        Parameters:
            cell: list|ndarray|torchTensor, the lattice vector of cell.
            coord: list|ndarray|torchTensor, the coordinates of atoms, in the order of atom_label.
            atom_list: list|
            output_path: str, the output path.
            file_name: str, the file name.
            system: str, the 1st line of output file i.e., the annotation or title of the file.
            out_coord_type: str, 'D' or 'C', which means whether the coordinate type is "Direct" or "Cartesian".

        Return: None
        """
        # check vars
        if not isinstance(cell, (list, np.ndarray)):
            raise TypeError(f'Unknown type of cell, type : {type(cell)}')
        elif not isinstance(coord, (list, np.ndarray)):
            raise TypeError(f'Unknown type of coord, type : {type(coord)}')
        elif not (isinstance(output_path, str) and isinstance(file_name, str) and isinstance(system, str)):
            raise TypeError(
                f'output_path||file_name||system must be strings, but type {type(output_path)}||{type(file_name)}||{type(system)} occurred.')
        elif out_coord_type != 'C' and out_coord_type != 'D':
            raise ValueError(f'Unknown coord_type : "{out_coord_type}"')

        # main
        with open(os.path.join(output_path, file_name), 'w') as POSCAR:
            POSCAR.write('#' + system)
            POSCAR.write('\n')
            # cell
            POSCAR.write('lattice\\\n')
            cell = np.asarray(cell, dtype=np.float32) * 1.89036  # 1 Angstrom = 1.89036 Bohr. Convert from A to Bohr.
            for vx in cell[:-1]:
                for xx in vx:
                    POSCAR.write(f'{xx:>14.8f}')
                POSCAR.write('\\\n')
            for xx in cell[-1]:
                POSCAR.write(f'{xx:>14.8f}')
            POSCAR.write('\n')
            # others
            POSCAR.write('coulomb-interaction Slab 001\n')
            POSCAR.write('coulomb-truncation-embed 0 0 0\n')
            # atom coordinates
            coord = np.asarray(coord, dtype=np.float32)
            if out_coord_type == 'C':
                POSCAR.write('coords-type Cartesian\n')
                if in_coord_type == 'D':
                    coord = coord @ cell
                elif in_coord_type == 'C':
                    coord = coord * 1.89036
                else:
                    raise ValueError(f'Unknown coordinate type {in_coord_type}. It must be "C" or "D".')
            elif out_coord_type == 'D':
                POSCAR.write('coords-type Lattice\n')
                if in_coord_type == 'D':
                    pass
                elif in_coord_type == 'C':
                    coord = coord @ np.linalg.inv(cell / 1.89036)
            for ind, vx in enumerate(coord[:, :3]):
                POSCAR.write(f'ion {atom_list[ind]:>3s}')
                for xx in vx:
                    POSCAR.write(f'{xx: > 14.8f}')
                POSCAR.write(f'    {fixed[ind][0]}')  # Selective Dynamics
                POSCAR.write('\n')

def write_xyz(
        elements,
        coords,
        cells,
        energies,
        numbers,
        forces,
        output_path: str,
        filename_list: List[str]|None=None,
        output_xyz_type: Literal['only_position_xyz','write_position_and_force']='only_position_xyz',
        n_core:int=-1
) -> None:
    """
        'only_position_xyz':output .xyz file with only position,without force
        'write_position_and_force':output .xyz file with position and force
    """

    if output_xyz_type not in {'only_position_xyz','write_position_and_force'}:
        raise ValueError(f"`output_xyz_type` must be 'only_position_xyz' or 'write_position_and_force',"
                         f" but occurred {output_xyz_type}.")
    # loading data
    n_batch = len(cells)
    if filename_list is None:
        filename_list = [f'{_}.xyz' for _ in range(n_batch)]
    elif not isinstance(filename_list, (List, Tuple)):
        raise TypeError(f'Invalid type of `filename_list`: {type(filename_list)}')
    elif len(filename_list) != n_batch:
        raise ValueError(f'Number of file names ({len(filename_list)}) and structures ({n_batch}) does not match')
    # check n_core
    if n_core == -1:
        n_core = jb.cpu_count(True)
    if n_core > n_batch:
        n_core = n_batch

    def _write_single(
            file_name: str,
            elements,
            coords,
            cells,
            energies,
            numbers,
            forces,
    ):
        with open(os.path.join(output_path, file_name), "w") as xyz:  # 遍历生成不同所有文件，并进行编写
            xyz.write(f"{len(coords)}\n")#total number of atoms

            cells_str = np.array2string(cells, **FLOAT_ARRAY_FORMAT).replace("[", " ").replace("]"," ").replace("\n","").replace("   "," ")
            xyz.write(f"Lattice ='{cells_str}' Properties=species:S:1:pos:R:3:forces:R:3 energy={energies:<.7e} pbc=T T T\n")
            # generate elem_list which contain the element type of every structure. (data type:list(array()),)
            elem_ = [[f"{elements[i]: <2s}"] * int(numbers[i]) for i in range(len(elements))]
            elem = sum(elem_, [])
            elem_list = np.array(elem).reshape((len(coords), 1))

            if output_xyz_type == "write_position_and_force":
                coo_force_str = AS_PRINT_COORDS(np.concatenate((coords, forces), axis=1))
                elem_pos_force_array = np.concatenate((elem_list, coo_force_str), axis=1)
                elem_pos_force = np.array2string(elem_pos_force_array, **STRING_ARRAY_FORMAT).replace("[", " ").replace("]"," ")#.replace("'", "")
                xyz.write(f"{elem_pos_force}\n")
            else:
                elem_pos_array = np.concatenate((elem_list, AS_PRINT_COORDS(coords)), axis=1)
                elem_pos_array = np.array2string(elem_pos_array, **STRING_ARRAY_FORMAT).replace("[", " ").replace(
                    "]", " ")  # .replace("'", "")
                xyz.write(f"{elem_pos_array}\n")

    if n_core == 1:
        for ii, fn in enumerate(filename_list):
            _write_single(
                fn,
                elements[ii],
                coords[ii],
                cells[ii],
                energies[ii],
                numbers[ii],
                forces[ii]
            )
    else:
        _para = jb.Parallel(n_core, )
        _para(
            jb.delayed(_write_single)(
                fn,
                elements[ii],
                coords[ii],
                cells[ii],
                energies[ii],
                numbers[ii],
                forces[ii]
            )
            for ii, fn in enumerate(filename_list)
        )


def write_cif(
        cells: Sequence[Sequence],
        coords: Sequence[np.ndarray],
        atom_labels: Sequence[Sequence[str]],
        atom_numbers: Sequence[Sequence[int]],
        output_path: str = './',
        file_name_list: str | Sequence[str] = 'POSCAR',
        system_list: str | Sequence[str] = 'untitled',
        coord_type: List[Literal['C', 'D']] | Literal['C', 'D'] = 'C',
        n_core: int = -1
    ) -> None:
    """
    Convert coordinates matrices to POSCAR format, and write files to the "output_path".

    Args:
        cells: List|ndarray, a batch of lattice vectors. shape: (n_batch, 3, 3)
        atom_labels: 2D list|ndarray of str, a batch of list of element symbols. shape: (n_batch, n_atom)
        atom_numbers:2D list|ndarray of int, list of atom number of each element, in the order of atom_label. shape: (n_batch, n_atom)
        coords: list|ndarray, the batch of atoms coordinates, in the order of atom_label. shape: (n_batch, n_atom, 3)
        output_path: str, the output path.
        file_name_list: list(str), the list of file names.
        system_list: str|list(str), the 1st line of output file i.e., the annotation or title of the file.
        coord_type: List[Literal['C', 'D']]|Literal['C', 'D'], 'D' or 'C', which means whether the input coordinates are "Direct" or "Cartesian".
            If only a string, `coord_type` would set to be '[`coord_type`] * n_batch'.
        n_core: int, the number of CPU cores to write files in parallel.

    Returns: None

    """
    def _write_single(
            file_name,
            cell,
            coord,
            atom_label,
            atom_number,
            _coord_type,
            system_name: str | Sequence[str] = 'untitled',
    ):
        atom_label_str = ' '.join(atom_label)
        formula = ''.join([f'{atom_label[_]}{atom_number[_]}' for _ in range(len(atom_number))])

        atom_label_list = list()
        atom_sequence_list = list()
        for i, num in enumerate(atom_number):
            atom_label_list.extend([atom_label[i]] * num)
            atom_sequence_list.extend(range(1, num + 1))

        a, b, c = np.linalg.norm(cell, axis=1)
        alpha = np.rad2deg(np.arccos(cell[1] @ cell[2] / (b * c)))
        beta = np.rad2deg(np.arccos(cell[0] @ cell[2] / (a * c)))
        gamma = np.rad2deg(np.arccos(cell[0] @ cell[1] / (a * b)))
        volume = np.linalg.det(cell)
        if volume < 0.: warnings.warn(f'The cell of {file_name} is a left-hand system.', RuntimeWarning)
        volume = np.abs(volume)
        if _coord_type == 'C':
            coord = coord @ np.linalg.inv(cell)

        with open(os.path.join(output_path, file_name), "w") as cif:
            cif.write(f'# {system_name}\ndata_{formula}\n')
            cif.write("_symmetry_space_group_name_H-M   'P 1'\n")
            cif.write(
                f'_cell_length_a   {a}\n'
                f'_cell_length_b   {b}\n'
                f'_cell_length_c   {c}\n'
                f'_cell_angle_alpha   {alpha}\n'
                f'_cell_angle_beta   {beta}\n'
                f'_cell_angle_gamma   {gamma}\n'
                '_symmetry_Int_Tables_number   1\n'
                f'_chemical_formula_structural   \'{atom_label_str}\'\n'
                f'_chemical_formula_sum   {formula}\n'
                f'_cell_volume   {volume}\n'
                'loop_\n'
                ' _symmetry_equiv_pos_site_id\n'
                ' _symmetry_equiv_pos_as_xyz\n' 
                '  1  \'x, y, z\'\n'
                'loop_\n'
                ' _atom_site_label\n'
                ' _atom_site_type_symbol\n'
                ' _atom_site_symmetry_multiplicity\n'
                ' _atom_site_fract_x\n'
                ' _atom_site_fract_y\n'
                ' _atom_site_fract_z\n'
                ' _atom_site_occupancy\n'
            )
            for i, coo in enumerate(coord):
                cif.write(
                    f'{atom_label_list[i]}{atom_sequence_list[i]:<4d}  {atom_label_list[i]}  1  '
                    f'{np.array2string(coo, **FLOAT_ARRAY_FORMAT)[1:-1]}  1\n'
                )

    # loading data
    n_batch = len(cells)
    if file_name_list is None:
        filename_list = [f'{_}.cif' for _ in range(n_batch)]
    elif not isinstance(file_name_list, (List, Tuple)):
        raise TypeError(f'Invalid type of `filename_list`: {type(file_name_list)}')
    elif len(file_name_list) != n_batch:
        raise ValueError(f'Number of file names ({len(file_name_list)}) and structures ({n_batch}) does not match')
    # check n_core
    if n_core == -1:
        n_core = jb.cpu_count(True)
    if n_core > n_batch:
        n_core = n_batch

    _para = jb.Parallel(n_core, )
    _para(
        jb.delayed(_write_single)(
            fn,
            cells[ii],
            coords[ii],
            atom_labels[ii],
            atom_numbers[ii],
            coord_type[ii],
            system_list[ii]
        )
        for ii, fn in enumerate(file_name_list)
    )

