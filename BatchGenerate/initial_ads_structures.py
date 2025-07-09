#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: initial_ads_structures.py
#  Environment: Python 3.12

import copy
from typing import List, Literal, Dict
import warnings

import torch as th
from torch import nn
import numpy as np

from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures
from BM4Ckit.utils.IrregularTensorReformat import IrregularTensorReformat
from BM4Ckit.BatchOptim.minimize import CG, QN, FIRE


class InitAdsStructure:
    def __init__(
            self,
            slabs: BatchStructures,
            ads: th.Tensor,
            ads_elements: Dict,
            vaccuum_direction: Literal['x', 'y', 'z'] = 'z',
            device: str | th.device='cpu',
            verbose:int = 0):
        """
        Generate initial guesses of surface adsorption structures.
        Args:
            slabs: batch structures of slabs.
            ads: Tensor(n_atom, 3), coordinates of the adsorbate.
            ads_elements: Dict(element symbol: atom number).
            vaccuum_direction: direction of vacuum layers.
            device: the device that program run on.
            verbose: to control the verboseness of output information.
        """
        self.slabs = slabs
        self.ads = ads.to(device)
        self.ads_elem = ads_elements
        self.verbose = verbose
        self.device = device
        n_batch = len(slabs)
        if not isinstance(self.slabs, BatchStructures):
            raise TypeError(f'`slabs` must be BatchStructures, but occurred {type(slabs)}')
        if not isinstance(self.ads, th.Tensor):
            raise TypeError(f'`ads` must be torch.Tensor, but occurred {type(ads)}')
        if sum(ads_elements.values()) != len(self.ads):
            raise ValueError(f'Number of atoms in `ads_elements` does not match it in `ads`.')

        # vacuum dirct
        vac_d: dict = {'x': 0, 'y': 1, 'z': 2}
        vac_d: int = vac_d[vaccuum_direction]

        # dist. mat. of ads. mol.
        self.dist_mol = th.linalg.norm(self.ads.unsqueeze(0) - self.ads.unsqueeze(1), dim=-1)  # (n_atom, n_atom)
        max_atom_len = th.max(self.dist_mol)
        # coord of slabs
        # TODO: Convert to Cartesian coordinates.
        self.slab_coords = [th.from_numpy(_).to(self.device) for _ in self.slabs.Coords]
        self.slab_cells = [th.from_numpy(_).to(self.device) for _ in self.slabs.Cells]
        # check vacuum length
        vacuum_length_pak = [(self.slab_cells[i][vac_d, vac_d] - th.max(self.slab_coords[i][:, vac_d]), th.max(self.slab_coords[i][:, vac_d]))  for i in range(len(slabs))]
        vacuum_length, highest_slab = zip(*vacuum_length_pak)
        highest_slab= th.tensor(highest_slab)  # (n_batch, )
        for i, _vac_len in enumerate(vacuum_length):
            if _vac_len < 3.:
                raise RuntimeError(f'Occurred unacceptable vacuum length {_vac_len} in {i}th structure. I HOPE YOU KNOWN WHAT YOU ARE DOING.')
            elif _vac_len < 9.:
                warnings.warn(f'The vacuum length of {i}th structure was too short: {vacuum_length}.')
            elif (_vac_len < (max_atom_len + 5.)) or th.any(th.linalg.norm(self.slab_cells[i], dim=1) - 5. < max_atom_len):
                warnings.warn(f'The {i}th cell are too small that distance between 2 adsorbates in neighbor lattice might be less than 5 Angstrom.')
        # set the lowest atom in ads to 3 Angstrom higher than the highest atom in slab
        lowest_mol = th.min(self.ads[:, vac_d], dim=-1).values  # (n_batch, )
        self.ads = (self.ads.unsqueeze(0)).expand(n_batch, -1, -1)  # (n_batch, n_mol_atom, n_dim)
        self.ads = self.ads + (highest_slab + 3. - lowest_mol).unsqueeze(-1).unsqueeze(-1)
        pass

    def run(
            self,
            slab_site: List[List[int]],
            ads_site: List[int],
            bond_length: float,
            bond_coeff: float = 5.,
            non_bond_coeff: float = 3.,
            keep_ads_conf_coeff: float = 2.
    ):
        """

        Args:
            slab_site: The indices of adsorption site in slabs.
            ads_site: The indices of adsorption site in adsorbate.
            bond_length: The target bond length of adsorbate and slabs.
            bond_coeff: Strength coefficient of ads-slab bond.
            non_bond_coeff: Strength coefficient of repulsiveness of non-bond atom between ads. and slab.
            keep_ads_conf_coeff: Strength coefficient of keeping original adsorbate configuration.

        Returns:
            BatchStructures: the batch structures of slabs with ads.

        """
        regularizor = IrregularTensorReformat()
        slab_pad_coords, pad_mask = regularizor.regularize(self.slab_coords, 0.)
        n_batch, n_atom, n_dim = slab_pad_coords.shape
        # slabs
        if isinstance(slab_site, List):
            assert len(slab_site) == n_batch, f'number of specified adsorption sites {len(slab_site)} does not match the number of structures {n_batch}.'
            #assert len(bond_length) == n_batch, f'number of specified adsorption sites {len(slab_site)} does not match the number of structures {n_batch}.'
            slab_mask = th.full_like(slab_pad_coords, False, device=self.device, dtype=th.bool)  # (n_batch, n_atom, n_dim)
            #target_bond_length = th.full((n_batch, n_atom), 0., device=self.device)  # (n_batch, n_atom) # TODO auto-adaptive bond length
            target_bond_length = bond_length
            for i, _sites in enumerate(slab_site):
                slab_mask[i, _sites] = True
                #target_bond_length[i, _sites] = th.tensor(bond_length[i], device=self.device)  # TODO auto-adaptive bond length
            slab_mask = slab_mask * pad_mask
        else:
            raise TypeError('`slab_site` must be a List[List[int]].')
        # ads molecules
        if isinstance(ads_site, List):
            ads_mask = th.full_like(self.ads[0], False, device=self.device, dtype=th.bool)  # (n_mol_atom, n_dim)
            ads_mask[ads_site] = True
        else:
            raise TypeError('`ads_site` must be a List[int].')
        # calc. force
        loss_fn = _Loss()
        #self._random_perturbate()

        # optimize
        optimizer = FIRE(
            1e-3,
            0.1,
            200,
            0.01,
            device=self.device,
            verbose=self.verbose
        )
        min_loss, X_ads = optimizer.run(
            loss_fn,
            self.ads,
            None,
            (slab_pad_coords, slab_mask, ads_mask, self.dist_mol, target_bond_length, bond_coeff, non_bond_coeff, keep_ads_conf_coeff),
            None,
        )

        ads_structures = self._integrate_coords(X_ads)
        return ads_structures

    def _integrate_coords(self, X_ads: th.Tensor) -> BatchStructures:
        """
        Integrate coordinates of adsorption and slabs.
        Args:
            X_ads: coordinates of ads.

        Returns: BatchStructure of adsorbed slabs.

        """
        ads_elem = list(self.ads_elem.keys())
        ads_number = [self.ads_elem[_] for _ in ads_elem]
        f = copy.deepcopy(self.slabs)
        X_ads = X_ads.numpy(force=True)
        for i in range(len(f)):
            f.Coords[i] = np.concatenate([f.Coords[i], X_ads[i]])
            f.Fixed[i] = np.concatenate([f.Fixed[i], np.ones_like(X_ads[i])])
            f.Elements[i].extend(ads_elem)
            f.Numbers[i].extend(ads_number)
        if f.Atom_list is not None:
            f.generate_atom_list(True)
        if f.Atomic_number_list is not None:
            f.generate_atomic_number_list(True)
        return f

    def _random_perturbate(self, ):
        """
        Ramdom perturbate adsorptions by orthogonal and translational transform to search more configuration.
        Returns:

        """
        ln = len(self.ads)
        rx = th.rand(ln) * 2 * th.pi
        ry = th.rand(ln) * 2 * th.pi
        rz = th.rand(ln) * 2 * th.pi
        tx = th.randn(ln) * 0.05
        ty = th.randn(ln) * 0.05
        tz = th.randn(ln) * 0.05

        self.ads = self._distance_preserv_transform(rx, ry ,rz, tx, ty, tz, self.ads)

    def _distance_preserv_transform(
            self,
            rx: th.Tensor,
            ry: th.Tensor,
            rz: th.Tensor,
            tx: th.Tensor,
            ty: th.Tensor,
            tz: th.Tensor,
            X: th.Tensor
    ) -> th.Tensor:
        """
        Orthogonal and transitional transform on input X.
        Args:
            rx: rotation degree around X-axis.
            ry: rotation degree around Y-axis.
            rz: rotation degree around Z-axis.
            tx: transition along X-axis.
            ty: transition along Y-axis.
            tz: transition along Z-axis.
            X: the coordinated to transform.

        Returns:

        """
        ln = len(X)
        if not (len(rx) == len(ry) == len(rz) == len(tx) == len(ty) == len(tz) == ln):
            raise RuntimeError(f'Incompatible shape of rx~rz & tx~tz & X. They must have same batch number.')
        # mass center
        center = th.sum(X, dim=1, keepdim=True)  # (n_batch, 1, 3)
        # rotation matrix
        rot_x = th.tensor(
            [
                [[1.]*ln,             [0.]*ln,              [0.]*ln],
                [[0.]*ln, th.cos(rx).tolist(), th.sin(-rx).tolist()],
                [[0.]*ln, th.sin(rx).tolist(),  th.cos(rx).tolist()]
            ],
            device=self.device
        )
        rot_y = th.tensor(
            [
                [th.cos(ry).tolist(), [0.]*ln,  th.sin(ry).tolist()],
                [[0.]*ln,             [1.]*ln,              [0.]*ln],
                [th.sin(-ry).tolist(), [0.]*ln, th.cos(ry).tolist()]
            ],
            device=self.device
        )
        rot_z = th.tensor(
            [
                [th.cos(rz).tolist(), th.sin(-rz).tolist(), [0.]*ln],
                [th.sin(rz).tolist(),  th.cos(rz).tolist(), [0.]*ln],
                [            [0.]*ln,              [0.]*ln, [1.]*ln]
            ],
            device=self.device
        )
        rot_x = rot_x.transpose(0, -1)
        rot_y = rot_y.transpose(0, -1)
        rot_z = rot_z.transpose(0, -1)
        trans = th.cat((tx.unsqueeze(-1).unsqueeze(-1), ty.unsqueeze(-1).unsqueeze(-1), tz.unsqueeze(-1).unsqueeze(-1)), dim=-1)  # (n_batch, 1, 3)
        X_trans = rot_z @ rot_y @ rot_x @ (X - center) + center + trans # (n_batch, n_atom, 3)

        return X_trans


def get_highest_atom_indices(structure: BatchStructures, vaccuum_direction: Literal['x', 'y', 'z'] = 'z',):
    """
    Get the highest atom index of given batch structures.
    Returns:
        List[List[int]], the indices of highest atom in each structure
    """
    # vacuum dirct
    vac_d: dict = {'x': 0, 'y': 1, 'z': 2}
    vac_d: int = vac_d[vaccuum_direction]
    # coords
    converter = IrregularTensorReformat()
    padded_coords = [th.from_numpy(_) for _ in structure.Coords]
    padded_coords, pad_mask = converter.regularize(padded_coords, -th.inf)  # (n_batch, n_atom, 3)
    indices = th.argmax(padded_coords[:, :, vac_d], dim=1)

    return indices.unsqueeze(-1).tolist()


class _Loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X_ads, X_slab, slab_mask, ads_mask, mol_origin_dist_mat, target_bond_len, c_b, c_nb, c_k):
        """

        Args:
            X_ads: (n_batch, n_mol_atom, n_dim)
            X_slab: (n_batch, n_atom, n_dim)
            slab_mask: (n_batch, n_atom, n_dim)
            ads_mask: (n_mol_atom, n_dim)
            mol_origin_dist_mat: (n_mol_atom, n_mol_atom)
            target_bond_len: float
            c_b:
            c_nb:
            c_k:

        Returns:

        """
        loss = (c_b * self._bond_force_field(X_ads, X_slab, slab_mask, ads_mask, target_bond_len) +
                c_nb * self._non_bond_loss(X_ads, X_slab, slab_mask, ads_mask) +
                c_k * self._keep_origin_loss(X_ads, mol_origin_dist_mat))
        return loss

    @staticmethod
    def _bond_force_field(X_ads, X_slab, slab_mask: th.Tensor, ads_mask: th.Tensor, target_bond_len: float):
        """

        Args:
            X_ads:
            X_slab:
            slab_mask:
            ads_mask:

        Returns:

        """
        # (n_batch, n_atom, 1, n_dim) - (n_batch, 1, n_mol_atom, n_dim) -> (n_batch, n_atom, n_mol_atom, n_dim)
        dist_nb = th.linalg.norm(X_slab.unsqueeze(2) - X_ads.unsqueeze(1), dim=-1)
        # quadratic potential
        b_loss = (dist_nb - target_bond_len)**2 # (n_batch, n_atom, n_mol_atom)
        # tol mask
        tol_mask = (slab_mask.unsqueeze(-2) * ads_mask.unsqueeze(0).unsqueeze(0))[..., 0]  # (n_batch, n_atom, 1, n_dim) * (1, 1, n_mol_atom, n_dim) -> (n_batch, n_atom, n_mol_atom, n_dim)
        b_loss = th.sum(b_loss * tol_mask, dim=(-1, -2))
        return b_loss

    @staticmethod
    def _non_bond_loss(X_ads, X_slab, slab_mask: th.Tensor, ads_mask: th.Tensor):
        """

        Args:
            X_ads:
            X_slab:
            slab_mask: (n_batch, n_atom, n_dim)
            ads_mask:  (n_mol_atom, n_dim)

        Returns:

        """
        # (n_batch, n_atom, 1, n_dim) - (n_batch, 1, n_mol_atom, n_dim) -> (n_batch, n_atom, n_mol_atom, n_dim) -> (n_batch, n_atom, n_mol_atom)
        dist_nb = th.linalg.norm(X_slab.unsqueeze(2) - X_ads.unsqueeze(1), dim=-1)
        # exponential repulsive
        n_b_loss = th.where(dist_nb < 3., th.exp(6./dist_nb) - 1, 0.)  # (n_batch, n_atom, n_mol_atom)
        # polynomial repulsive
        #n_b_loss = (th.linalg.norm(dist_nb, dim=-1) - 3.5)**3
        #n_b_loss = th.where(n_b_loss < 0., 0., n_b_loss) # (n_batch, n_atom, n_mol_atom)
        # half L-J repulsive
        #n_b_loss = 5./(th.linalg.norm(dist_nb, dim=-1))**12  # (n_batch, n_atom, n_mol_atom)
        # tol mask
        tol_mask = (~slab_mask.unsqueeze(-2) * ~ads_mask.unsqueeze(0).unsqueeze(0))[..., 0]  # (n_batch, n_atom, 1, n_dim) * (1, 1, n_mol_atom, n_dim) -> (n_batch, n_atom, n_mol_atom, n_dim)
        n_b_loss = th.sum(n_b_loss * tol_mask, dim=(-1, -2))
        return n_b_loss

    @staticmethod
    def _keep_origin_loss(X_ads, origin_dist_mat):
        dist_mat = th.linalg.norm(X_ads.unsqueeze(1) - X_ads.unsqueeze(2), dim=-1)  # (n_batch, n_mol_atom, n_mol_atom)
        k_o_loss = th.sum((dist_mat - origin_dist_mat)**2)
        return k_o_loss

