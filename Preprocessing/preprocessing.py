# Preprocessings

# basic modules
import re
import copy
import time
import traceback
import importlib
from typing import Dict, Set, Tuple, List, Optional, Sequence, Any, Union
import warnings
import dgl

import lmdb

import pickle as pkl

import torch as th

import numpy as np

import joblib as jb

import ase

import torch_geometric as pyg
from torch_geometric.data import Data as pygData
from torch_geometric.data import Batch as pygBatch

from .load_files import POSCARs2Feat, load_from_csv
from .supercells import supercells
from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures


# Check modules func # TODO
def check_module(module_name):
    try:
        pkg = importlib.import_module(module_name)
        return pkg
    except ImportError:
        warnings.warn(f'Package {module_name} was not found, therefore some related methods would be unavailable.')


""" Constance """
ALL_ELEMENT = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
               'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
               'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
               'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
               'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
               'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', }  # element set

""" screening samples """


class Screen_samp_by_element():
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


class Feat2LMDB():  # TODO: Imperfect.
    r""" Convert tensor type features into LMDB """

    def __init__(self, lmdb_path: str, max_alloc_memory: int = 10737418240) -> None:
        self.lmdb_path = lmdb_path
        self.max_alloc_memory = max_alloc_memory

    def write(self, ids: List[str], cells: List[th.Tensor], atoms: List[th.Tensor], coords: List[th.Tensor], labels: List, progress_bar: bool = False) -> None:
        """
        
        """
        if len(ids) != len(cells) != len(atoms) != len(coords) != len(labels):
            raise ValueError(f'ids, cells, atoms, coords, and labels must have the same length, but occurred {len(ids), len(cells), len(atoms), len(coords), len(labels)}')
        try:
            env = lmdb.open(self.lmdb_path, map_size=self.max_alloc_memory)
            k = 1

            for i, id_ in enumerate(ids):
                txn = env.begin(write=True)
                # load data
                n_atom = len(atoms[i])
                _key = f'{n_atom}'.encode()
                # check whether n_atom in dict
                _batch = txn.get(_key)
                if _batch is None:
                    _batch = [[], [], [], [], []]  # [id, cell, atoms, coords, label]
                else:
                    _batch = pkl.loads(_batch)
                # insert data
                _batch[0].append(id_)
                _batch[1].append(cells[i])
                _batch[2].append(atoms[i])
                _batch[3].append(coords[i])
                _batch[4].append(labels[i])
                # write in lmdb
                txn.put(key=_key, value=pkl.dumps(_batch))
                txn.commit()
                # progress bar
                if progress_bar:
                    _prog_piece = len(ids) // 20
                    if i > _prog_piece * k:
                        print(f'PROGRESS: ' + '>' * k + f'{100 * (i / len(ids)):.2f}%', end='\r')
                        k += 1

        except Exception as e:
            print(f'An ERROR occurred.\nERROR: {e}')
            traceback.print_exc()

        finally:
            env.close()  # type: ignore

    def read(self, keys: Sequence):
        """
        """
        try:
            env = lmdb.open(self.lmdb_path, map_size=self.max_alloc_memory)

            batch_dict = dict()
            for k in keys:
                _key = k.encode()
                txn = env.begin()

                if k not in batch_dict:
                    batch_dict[k] = list()

                batch_dict[k].append(pkl.loads(txn.get(key=_key)))

        except Exception as e:
            print(f'An ERROR occurred.\nERROR: {e}')

        finally:
            env.close()  # type: ignore
            return batch_dict  # type: ignore


class Create_Graphs():  # TODO: Deprecated.
    r"""
    Create DGL-type graphs and I/O created graphs list.

    Saved Graph information:
        edges: the atom pair with distance less than r_cut_off
        node_features['Z']: the atomic number
        node_features['X']: the atomic coordinates
        edge_features['R']: the coordinates difference between 2 nodes

    Init Parameters:
        r_cut_off: float, spatial distance cut-off. Unit: Angstrom.

    Methods:
        create_primitive:
            parameters:
                cell_vectors: tensor with shape (batch_size, 3, 3), batch of cell vectors.
                atomic_numbers: tensor with shape (batch_size, n_atom), batch of atomic numbers in each cell.
                atomic_coordinates: tensor with shape (batch_size, n_atom, 3), batch of Cartesian coordinates x,y,z in each cell.
                supercell_index: tensor with shape (3,), the index of cell with respect to the original cell.
            return:
                List[dgl.Graph] with length batch_size

        create_supercells:
            parameters:
                cell_vectors: tensor with shape (batch_size, 3, 3), batch of cell vectors.
                atomic_numbers: tensor with shape (batch_size, n_atom), batch of atomic numbers in each cell.
                atomic_coordinates: tensor with shape (batch_size, n_atom, 3), batch of Cartesian coordinates x,y,z in each cell.
                supercell_index: tensor with shape (3,), the index of cell with respect to the original cell.
            return:
                List[dgl.Graph] with length batch_size
        
        create_overlap:
            parameters:
                cell_vectors: tensor with shape (batch_size, 3, 3), batch of cell vectors.
                atomic_numbers: tensor with shape (batch_size, n_atom), batch of atomic numbers in each cell.
                atomic_coordinates: tensor with shape (batch_size, n_atom, 3), batch of Cartesian coordinates x,y,z in each cell.
                supercell_index: tensor with shape (3,), the index of cell with respect to the original cell.
            return:
                List[dgl.Graph] with length batch_size

        create_save:
            parameters:
                path: str, the path to save graphs.
                cell_vectors: tensor with shape (batch_size, 3, 3), batch of cell vectors.
                atomic_numbers: tensor with shape (batch_size, n_atom), batch of atomic numbers in each cell.
                atomic_coordinates: tensor with shape (batch_size, n_atom, 3), batch of Cartesian coordinates x,y,z in each cell.
                supercell_index: tensor with shape (3,), the index of cell with respect to the original cell.
            return:
                None
    """

    def __init__(self, r_cut_off: float = 6., verbose: int = 0, *args, **kwargs) -> None:
        super().__init__()
        self.r_cut_off = r_cut_off
        self.verbose = verbose

    def _overlap_int(self, a: th.Tensor, R1: th.Tensor, R2: th.Tensor):
        r"""
        a: (n_batch, n_atom, n_zeta)
        R1: (n_batch, n_atom, 3)
        R2: (n_batch, n_supercell, n_atom, 3)
        """
        if th.any(a <= 0): raise ValueError('exponential "a" must be greater than 0.')
        a1, a2 = th.broadcast_tensors(a.unsqueeze(2), a.unsqueeze(1))  # (n_batch, n_atom, n_atom, n_zeta)
        zeta: th.Tensor = -a1 * a2 / (a1 + a2)  # (n_batch, n_atom, n_atom, n_zeta)
        r_AB = R1.unsqueeze(1).unsqueeze(-2) - R2.unsqueeze(2)  # (n_batch, n_supercell, n_atom1, n_atom2, 3)
        y = ((th.pi / ((a1 + a2).unsqueeze(1))) ** 1.5) * th.exp(zeta.unsqueeze(1) * (th.norm(r_AB, p=2, dim=-1,
                                                                                              keepdim=True)) ** 2)  # (n_batch, 1, n_atom1, n_atom2, n_zeta) * ((n_batch, n_supercell, n_atom1, n_atom2, 3) @ (..., 3, 1)) -> (n_batch, n_supercell, n_atom1, n_atom2, n_zeta)
        y = th.sum(y, dim=(1,))  # (n_batch, n_atom1, n_atom2, n_zeta)
        return y

    def create_primitive(self,
                         cell_vectors: th.Tensor,
                         atomic_numbers: th.Tensor,
                         atomic_coordinates: th.Tensor) -> List[dgl.DGLGraph]:
        r"""
        Creat a list of Hetrogeneous graphs that G({Atom-Bond-Atom}, {Cell-Dispersion-Cell})
        """
        # TODO
        raise NotImplementedError
        device = atomic_coordinates.device
        n_batch, n_atom = atomic_numbers.shape
        supercell_indices = supercells(cell_vectors, self.r_cut_off, device=device)  # (n_cell, 3)
        n_cell = len(supercell_indices)
        # calculate cross-cell dist.; cell_diff = supercell_indices @ cell_vec; <<<
        # shape: (1, n_prim_cells, 1, 3)@(n_batch, 1, 3, 3) -> (n_batch, n_prim_cells, 1, 3)
        cell_diff = (supercell_indices.unsqueeze(-2).unsqueeze(0)) @ (cell_vectors.unsqueeze(1))
        cell_diff.squeeze_(-2)  # (n_batch, n_prim_cells, 3)
        # calculate in-cell coordinate dist.
        # shape: (n_batch, n_atom, 1, 3) - (n_batch, 1, n_atom, 3) -> (n_batch, n_atom, n_atom, 3)
        coord_diff = atomic_coordinates.unsqueeze(2) - atomic_coordinates.unsqueeze(1)
        # dist. mat.: shape: (n_batch, n_atom, n_atom)
        sparse_dist = th.linalg.norm(coord_diff, ord=2, dim=-1)
        # gaussian dist: dist = e^(-a*(R)^2), where a = 9.2103/r_cutoff^2 to ensure dist < 1e-4 when R > r_cutoff
        sparse_dist = (th.where(sparse_dist < self.r_cut_off, th.exp(-(9.2103 / (self.r_cut_off) ** 2) * (sparse_dist) ** 2), 0.)).to_sparse()

        Graph_list = list()
        for batch_indx, samp in enumerate(sparse_dist):
            index = (samp.coalesce()).indices()
            sing_gra_dict = {('Atom', 'Bond', 'Atom'): (index[0], index[1]),
                             ('Cell', 'Dispersion', 'Cell'): (th.zeros(n_cell, dtype=th.int32, device=device), th.arange(0, n_cell, dtype=th.int32, device=device))}
            n_node_dict = {'Atom': n_atom, 'Cell': n_cell}
            single_graph: dgl.DGLHeteroGraph = dgl.heterograph(sing_gra_dict, n_node_dict)  # type: ignore
            single_graph.nodes['Atom'].data['Z'] = atomic_numbers[batch_indx]
            single_graph.nodes['Atom'].data['X'] = atomic_coordinates[batch_indx]
            single_graph.edges['Bond'].data['R'] = coord_diff[batch_indx, index[0], index[1]]
            single_graph.edges['Dispersion'].data['R'] = cell_diff[batch_indx]
            Graph_list.append(single_graph)

        return Graph_list

    def create_supercells(self,
                          cell_vectors: th.Tensor,
                          atomic_numbers: th.Tensor,
                          atomic_coordinates: th.Tensor,
                          device: th.device | str | None = None) -> List[dgl.DGLGraph]:
        r"""
        Creat a list of DGLGraph that Nodes are all atoms in r_cut_off.
        """
        # TODO
        raise NotImplementedError
        if device is None:
            device = atomic_coordinates.device
        else:
            cell_vectors = cell_vectors.to(device)
            atomic_numbers = atomic_numbers.to(device)
            atomic_coordinates = atomic_coordinates.to(device)
        supercell_indices = supercells(cell_vectors, self.r_cut_off, device=device)  # (n_cell, 3)
        # calculate cross-cell dist.; cell_diff = supercell_indices @ cell_vec; <<<
        # shape: (1, n_prim_cells, 1, 3)@(n_batch, 1, 3, 3) -> (n_batch, n_prim_cells, 1, 3)
        cell_diff = (supercell_indices.unsqueeze(-2).unsqueeze(0)) @ (cell_vectors.unsqueeze(1))
        # calculate the atom coords across cells (x_j + R_k) <<<
        # shape: (n_batch, 1, n_atom, 3) + (n_batch, n_prim_cells, 1, 3)
        #     -> (n_batch, n_prim_cells, n_atom, 3) -flat-> (n_batch, n_prim_cells*n_atom, 3)
        coord_cross = atomic_coordinates.unsqueeze(1) + cell_diff
        n_batch, n_prim_cells, n_atom, _ = coord_cross.shape
        coord_cross = coord_cross.flatten(1, 2)
        # calculate actual dist. vec.; dist_vec = r_ijk = ||x_i - (x_j + R_k)|| where R_k is the cell vector; <<<
        # shape: (n_batch, n_atom, 1, 3) - (n_batch, 1, n_prim_cells*n_atom, 3)
        #        -> (n_batch, n_atom, n_prim_cells*n_atom, 3)  # coord_diff
        #   -norm-> (n_batch, n_atom, n_prim_cells*n_atom)    # euclid distance
        distance = atomic_coordinates.unsqueeze(2) - coord_cross.unsqueeze(1)
        distance = th.linalg.norm(distance, ord=2, dim=-1)
        # cut-off to create COO sparse tensor of pairwise distance. Indices: (ind_batch, ind_atom1, ind_atom2); Values: distance.
        sparse_distance = (th.where(distance < self.r_cut_off, th.exp(-(9.2103 / (self.r_cut_off) ** 2) * (distance) ** 2), 0.))
        sparse_distance = sparse_distance.to_sparse()

        # update atomic number tensor
        atomic_numbers = th.broadcast_to(atomic_numbers.unsqueeze(1), (n_batch, n_prim_cells, n_atom))  # (n_batch, n_atom) -brod-> (n_batch, n_prim_cells, n_atom)
        atomic_numbers = th.flatten(atomic_numbers, -2, -1)
        #atomic_numbers = th.broadcast_to(atomic_numbers, (n_batch, n_atom, n_atom*n_prim_cells))

        # generate graph
        Graph_list = list()
        for batch_indx, samp in enumerate(sparse_distance):
            index = (samp.coalesce()).indices()
            single_graph = dgl.graph((index[0], index[1]), num_nodes=n_prim_cells * n_atom)
            single_graph.ndata['Z'] = atomic_numbers[batch_indx]
            single_graph.ndata['X'] = coord_cross[batch_indx]
            single_graph.edata['R'] = distance[batch_indx, index[0], index[1]]
            Graph_list.append(single_graph)

        return Graph_list

    def create_overlap(self,
                       cell_vectors: th.Tensor | np.ndarray,
                       atomic_numbers: th.Tensor | np.ndarray,
                       atomic_coordinates: th.Tensor | np.ndarray, *,
                       zeta_range: Tuple[float, float, int] = (0.2, 10, 30), overlap_thres: float = 1e-5,
                       device: str | th.device | None = None
                       ) -> List[dgl.DGLGraph]:
        r"""
        
        """
        if isinstance(cell_vectors, np.ndarray):
            cell_vectors = th.from_numpy(cell_vectors)
        if isinstance(atomic_numbers, np.ndarray):
            atomic_numbers = th.from_numpy(atomic_numbers)
        if isinstance(atomic_coordinates, np.ndarray):
            atomic_coordinates = th.from_numpy(atomic_coordinates)

        n_batch, n_atom, _ = atomic_coordinates.shape
        a = (th.linspace(*zeta_range, device=device)).unsqueeze(0).unsqueeze(0)  # alpha, coeff of exp.
        a = th.broadcast_to(a, (n_batch, n_atom, -1))
        if device is None:
            device = atomic_coordinates.device
        else:
            cell_vectors = cell_vectors.to(device)
            atomic_coordinates = atomic_coordinates.to(device)
            atomic_numbers = atomic_numbers.to(device)

        supercell_indices = supercells(cell_vectors, self.r_cut_off, device=device)  # (n_cell, 3)
        # calculate cross-cell dist.; cell_diff = supercell_indices @ cell_vec; <<<
        # shape: (1, n_prim_cells, 1, 3)@(n_batch, 1, 3, 3) -> (n_batch, n_prim_cells, 1, 3)
        cell_diff = (supercell_indices.unsqueeze(-2).unsqueeze(0)) @ (cell_vectors.unsqueeze(1))
        # calculate the atom coords across cells (x_j + R_k) <<<
        # shape: (n_batch, 1, n_atom, 3) + (n_batch, n_prim_cells, 1, 3)
        #     -> (n_batch, n_prim_cells, n_atom, 3)
        coord_cross = atomic_coordinates.unsqueeze(1) + cell_diff
        n_batch, n_prim_cells, n_atom, _ = coord_cross.shape
        # calculate overlap matrix;
        # shape: (n_batch, n_atom1, n_atom2)
        overlap = self._overlap_int(a, atomic_coordinates, coord_cross)
        indices = th.where(overlap[..., 0] > overlap_thres)

        # generate graph
        Graph_list = list();
        ori_b_indx = 0;
        k = 0;
        k_old = 0;
        batch_indx = 0;
        atom1_indx, atom2_indx = indices[1], indices[2]
        for i, batch_indx in enumerate(indices[0]):
            if batch_indx == ori_b_indx:
                k += 1
            else:
                single_graph = dgl.graph((atom1_indx[k_old:k], atom2_indx[k_old:k]), num_nodes=n_atom)
                single_graph.ndata['Z'] = atomic_numbers[batch_indx - 1]
                single_graph.ndata['X'] = atomic_coordinates[batch_indx - 1]
                single_graph.edata['R'] = overlap[batch_indx - 1, atom1_indx[k_old:k], atom2_indx[k_old:k]]
                Graph_list.append(single_graph)

                ori_b_indx += 1
                k_old = copy.deepcopy(k)
                k += 1
        # The last sample
        single_graph = dgl.graph((atom1_indx[k_old:k], atom2_indx[k_old:k]), num_nodes=n_atom)
        single_graph.ndata['Z'] = atomic_numbers[batch_indx]
        single_graph.ndata['X'] = atomic_coordinates[batch_indx]
        single_graph.edata['R'] = overlap[batch_indx, atom1_indx[k_old:k], atom2_indx[k_old:k]]
        Graph_list.append(single_graph)

        #Graph_batch = dgl.batch(Graph_list)

        return Graph_list

    def feat2graphs(self, feat: BatchStructures, n_core: int = 1, device: str | th.device | None = None):
        r"""
        Conver BatchStructures to list of dgl.graph
        """
        graph_list = list()
        training_batches, val_batches, training_labels, val_labels, training_args, val_args = feat.rearrange(list(range(len(feat._Sample_ids))), 0., n_core, self.verbose)
        for tb_ in training_batches:
            _dat = self.create_overlap(*tb_, device=device)
            graph_list.extend(_dat)
        return graph_list


class Create_ASE():
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
        self.verbose = verbose
        pass

    def feat2ase(self, feat, set_tags: bool = True, n_core=-1) -> List[ase.Atoms]:
        r"""
        Convert to ase.Atoms from given instance feat that generated by POSCARs2Feat or ConcatPOSCAR2Feat.

        Parameters:
            feat: feat instance that generated by POSCARs2Feat or ConcatPOSCAR2Feat.
            set_tags: bool, whether set Atoms.tags = np.ones(n_atom) automatically.
        
        Returns:
            List[ase.Atoms], the list of Atoms instances.
        """
        t_st = time.perf_counter()
        feat.generate_atom_list()
        if self.verbose: print('Converting to ASE.Atoms ...')

        def _base_convert(symb: Sequence, pos: Sequence, cell: Sequence, pbc: Tuple[bool, bool, bool] | bool = (True, True, True), set_tags: bool = True) -> ase.Atoms:
            samp = ase.Atoms(symbols=symb, positions=pos, cell=cell, pbc=pbc)
            if set_tags:
                samp.set_tags(np.ones(len(samp)))
            return samp

        _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose)
        ase_list = _para(jb.delayed(_base_convert)(symb, feat.Coords[i], feat.Cells[i], True, set_tags) for i, symb in enumerate(feat.Atom_list_))

        if ase_list is None: raise RuntimeError
        if self.verbose: print(f'Done. Total Time: {time.perf_counter() - t_st:<5.4f}')

        return ase_list

    def array2ase(self, symb: Sequence, pos: Sequence, cell: Sequence, pbc: Tuple[bool, bool, bool] | bool = (True, True, True), set_tags: bool = True, n_core: int = -1) -> List[
        ase.Atoms]:
        r"""
        Convert to ase.Atoms from the given Sequence of symbols, positions, cell vectors and pbc information.

        Parameters:
            symb: Sequence[Sequence], the sequence of element lists.
            pos: Sequence[Sequence], the sequence of atom coordinates lists.
            cell: Sequence[Sequence], the sequence of cell vectors lists.
            pbc: bool|Tuple[bool, bool, bool], the direction of periodic boundary condition (x, y, z).
        
        Returns:
            Dict[ase.Atoms], the list of Atoms instances.
        """
        t_st = time.perf_counter()
        if self.verbose: print('Converting to ASE.Atoms ...')

        def _base_convert(symb: Sequence, pos: Sequence, cell: Sequence, pbc: Tuple[bool, bool, bool] | bool = (True, True, True), set_tags: bool = True) -> ase.Atoms:
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
            ase_dict[feat._Sample_ids[i]] = samp

        _para = jb.Parallel(n_jobs=n_core, verbose=self.verbose, require='sharedmem')
        _para(jb.delayed(_sig_conv)(i, s) for i, s in enumerate(feat.Atom_list))  # type: ignore

        if self.verbose: print(f'Done. Total Time: {time.perf_counter() - t_st:<5.4f}')

        return ase_dict


class Create_PyGdata():
    r"""
    create torch-geometric.data.Data or Batch from various types.
    """

    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        pass

    @staticmethod
    def single_ase2data(atoms: ase.Atoms):
        """Convert a single atomic structure to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

            sid (uniquely identifying object): An identifier that can be used to track the structure in downstream
            tasks. Common sids used in OCP datasets include unique strings or integers.

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
            fixed = th.from_numpy(_fix)#.unsqueeze(0)  # fixme
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
