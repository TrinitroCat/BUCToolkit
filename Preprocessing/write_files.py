import os
import warnings
from typing import Sequence, List, Literal, Tuple

import joblib as jb
import numpy as np

# from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures
from BM4Ckit._print_formatter import STRING_ARRAY_FORMAT, AS_PRINT_COORDS


class WritePOSCARs:
    """

    """

    def __init__(self,
                 cells: Sequence[Sequence],
                 coords: Sequence[np.ndarray],
                 atom_labels: Sequence[Sequence[str]],
                 atom_numbers: Sequence[Sequence[int]],
                 fixed: Sequence[np.ndarray] | None = None,
                 output_path: str = './',
                 file_name_list: str | Sequence[str] = 'POSCAR',
                 system_list: str | Sequence[str] = 'untitled',
                 coord_type: List[Literal['C', 'D']] | Literal['C', 'D'] = 'C',
                 ncore: int = -1) -> None:
        """
        Convert coordination matrices to POSCAR format, and write files to the "output_path".

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
            coord_type: List[Literal['C', 'D']]|Literal['C', 'D'], 'D' or 'C', which means whether the coordination is "Direct" or "Cartesian".
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
            raise ValueError(f'Invalid value of `coord_type`: type: {type(coord_type)}, length: {len(coord_type)}')
        if fixed is None:
            fixed = [np.full_like(_, 1, dtype=np.int8) for _ in coords]
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
            _para = jb.Parallel(ncore, )
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
        Convert coordination matrix to POSCAR format, and write a file to the output_path.

        Args:
            cell: list|ndarray, the lattice vector of cell.
            atom_label:1D list|ndarray of str, list of element symbols.
            atom_number:1D list|ndarray of int, list of atom number of each element, in the order of atom_label.
            coord: list|ndarray, the coordination of atoms, in the order of atom_label.
            fixed: list|ndarray, the fixed direction of atoms.
            output_path: str, the output path.
            file_name: str, the file name.
            system: str, the 1st line of output file i.e., the annotation or title of the file.
            coord_type: str, 'D' or 'C', which means whether the coordination is "Direct" or "Cartesian".

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
                            f'but type {type(output_path)}||{type(file_name)}||{type(system)} occurred.')
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
        Convert coordination matrices to POSCAR format, and write files to the "output_path".

        Parameters:
            batch_structures: batch of structures.
            output_path: str, the output path.
            file_name_list: list(str), the list of file names.
            system_list: str|list(str), the 1st line of output file i.e., the annotation or title of the file.
            coord_type: str, 'D' or 'C', which means whether the coordination is "Direct" or "Cartesian".

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
        Convert coordination matrix to POSCAR format, and write a file to the output_path.

        Parameters:
            cell: list|ndarray|torchTensor, the lattice vector of cell.
            coord: list|ndarray|torchTensor, the coordination of atoms, in the order of atom_label.
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
