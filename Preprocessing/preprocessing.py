""" Preprocessing """

import copy
# basic modules
import re
import time
import warnings
from typing import Dict, Set, Tuple, List, Sequence, Any, Union

import joblib as jb
import numpy as np
import torch as th

from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures
from BM4Ckit.utils._CheckModules import check_module

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

""" screening samples """


class ScreenSampleByElement:
    """
    Screening samples by given conditions.

    Methods:
        screen_from_elements: select samples that only contains elements in input element_contains from input formula_dict, and return selected dict.
        screen_from_formula: select samples that only contains elements in input element_contains from input formula_dict, and return selected dict.
    """

    def __init__(self, ) -> None:
        """
        Screening samples by given conditions.

        Methods:
            screen_from_elements: select samples that only contains elements in input element_contains from input formula_dict, and return selected dict.
            screen_from_formula: select samples that only contains elements in input element_contains from input formula_dict, and return selected dict.
        """
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
        select samples that only contains elements in input element_contains from input formula_dict, and return selected dict.

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
        select samples that only contains elements in input element_contains from input formula_dict, and return selected dict.

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


def EIGENVAL2array(file='./EIGENVAL', *args, **kwargs) -> Tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    """
    Read VASP output file EIGENVAL to np.array
    """
    f = open(file, )
    data = f.readlines()
    nn, k_points, n_bands = data[5].split()
    k_points, n_bands = int(k_points), int(n_bands)

    k_coords = [data[7 + i * (n_bands + 2)].split() for i in range(k_points)]
    k_coords = np.array(k_coords, dtype=np.float32)

    eigs = [data[8 + i * (n_bands + 2): 6 + (i + 1) * (n_bands + 2)] for i in range(k_points)]
    eigs = [a.split() for eigs_ in eigs for a in eigs_]
    eigs = np.array(eigs, dtype=np.float32).reshape((k_points, n_bands, 5))

    f.close()

    return k_coords[:, :-1], eigs


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

    def feat2ase(self, feat, set_tags: bool = True, n_core=-1) -> List[ase.Atoms]:
        r"""
        Convert to ase.Atoms from given instance feat that generated by POSCARs2Feat or ConcatPOSCAR2Feat.

        Parameters:
            feat: feat instance that generated by POSCARs2Feat or ConcatPOSCAR2Feat.
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
        ase_list = _para(jb.delayed(_base_convert)(symb, feat.Coords[i], feat.Cells[i], True, set_tags) for i, symb in enumerate(feat.Atom_list_))

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
        Convert to ase.Atoms from given instance feat that generated by POSCARs2Feat or ConcatPOSCAR2Feat.

        Parameters:
            feat: feat instance that generated by POSCARs2Feat or ConcatPOSCAR2Feat.
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
                idx=_id)
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
    create torch-geometric.data.Data or Batch from various types.
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
            data_list: List = [_convert_single(i,
                                               feat.Cells[i],
                                               feat.Coords[i],
                                               feat.Atomic_number_list[i],
                                               feat.Fixed[i]) for i, _id in enumerate(feat.Sample_ids)]  # type: ignore
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
