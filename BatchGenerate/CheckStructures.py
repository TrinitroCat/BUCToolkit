"""
Check Structures by atomic distance, surface smoothness, and coordination.
"""
#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: CheckStructures.py
#  Environment: Python 3.12

import time
import warnings
from typing import Literal

from BM4Ckit.BatchStructures.BatchStructuresBase import BatchStructures
import numpy as np

class CheckStructures:
    """

    """
    def __init__(self, batch_structures: BatchStructures):
        assert isinstance(batch_structures, BatchStructures), (f'`batch_structures` must be a BatchStructures type, '
                                                               f'but occurred {type(batch_structures)}.')
        self.BS = batch_structures

    def check_distance(self, radius_threshold: float=1):
        """
        check atomic pairwise distances that are too close (less than given `radius_threshold`)
        Args:
            radius_threshold: float, the threshold radius.
        Returns:
            mask_array: ndarray[(n_batch, ), bool], where False for failed to the check.
        """
        self.BS.direct2cartesian()
        coords = self.BS.Coords
        mask = np.full(len(coords), True, )

        for i, coo in enumerate(coords):
            if self.BS.Dist_mat is None:
                dist_mat = np.linalg.norm(coo[:, None, :] - coo[None, :, :], axis=-1)
            else:
                dist_mat = self.BS.Dist_mat[i]
            # exclude the diag. part
            diag_indx = np.arange(len(coo))
            dist_mat[diag_indx, diag_indx] = 20.

            if np.min(dist_mat) < radius_threshold:
                mask[i] = False

        return mask

    @staticmethod
    #@jit(nopython=True)
    def _k_means(x:np.ndarray, c:np.ndarray, maxiter:int=100, thres:float=1e-4):
        """
        k-mean clustering algorithm
        Args:
            x: (n_atom, ), sample points
            c: (nc, ), cluster center points
            maxiter: max iteration numbers

        Returns:
            variance
        """
        n_c = len(c)
        nc_indx = np.arange(n_c)
        c_old = c.copy()
        assert maxiter >= 1, f'Invalid `maxiter` value {maxiter}.'
        for i in range(maxiter):
            # calc. dist.
            dist = np.abs(x[:, None] - c[None, :])  # (n_atom, nc)
            min_dist_indx = np.argmin(dist, -1)  # (n_atom, )
            # update center
            _mask = (min_dist_indx[:, None] == nc_indx)  # (n_atom, nc)
            n_samp_in_cluster = np.sum(_mask, axis=0)  # (n_c, )
            drop_mask = n_samp_in_cluster == 0  # drop the non-sample cluster
            c = np.where(drop_mask, 0., np.sum(x[:, None] * _mask, axis=0)/(n_samp_in_cluster + 1e-20))  # (n_atom, 1) * (n_atom, nc) -mean-> (nc, )
            # delete the empty cluster
            if np.any(drop_mask):
                select_indx = np.where(~drop_mask)[0]
                c = c[select_indx]
                c_old = c_old[select_indx]
                _mask = _mask[:, select_indx]
                n_c = len(c)
                nc_indx = np.arange(n_c)
            # judge threshold
            if np.linalg.norm(c - c_old) < thres:
                return c, np.max(np.sum((x[:, None] - c[None, :])**2 * _mask, axis=0)/np.sum(_mask, axis=0))  # return the cluster points and metrics for smoothness
            else:
                c_old = c.copy()

        #warnings.warn(f'K-means does not converge in {maxiter} steps.', RuntimeWarning)
        return c, np.max(np.sum((x[:, None] - c[None, :])**2 * _mask, axis=0)/np.sum(_mask, axis=0))

    @staticmethod
    def _spectrum_clustering(x: np.ndarray, r_cutoff: float=3.):
        """
        Diagonalize laplace matrix of surface and bottom of slab to determined subgraph number.
        Args:
            x: (n_atom, 3), coordinates of surf. or bott.
            r_cutoff: threshold of radius.

        Returns:
            n: number of connected subgraphs.
        """
        dist = np.linalg.norm(x[:, None] - x[None, :], axis=-1) + (r_cutoff + 10.) * np.eye(len(x))  # exclude the diag.
        adj = np.where(dist < r_cutoff, 1., 0.)
        degree = np.diag(np.sum(adj, axis=-1))
        laplace = degree - adj
        eigs = np.linalg.eigvalsh(laplace)
        n_sub = np.sum(np.abs(eigs - 0.) < 1e-7)
        return n_sub

    def _periodic_reformat(self, cell, coords):
        """
        Transition the atom moved out of a cell into the periodic position in the cell.
        Args:
            cell:
            coords:

        Returns:

        """
        # TODO
        raise NotImplementedError

    def check_surf_smoothness(
            self,
            smoothness_threshold: float = 0.5,
            r_cutoff: float = 3.,
            surf_check_method: Literal['ConnectedGraph', 'C', 'MinDistance', 'M']|None = 'MinDistance'
    ):
        """
        check whether the surface smooth enough (whether the max difference among surface atoms along z axis < `smoothness_threshold`)
        Args:
            smoothness_threshold: smoothness threshold.
            r_cutoff: the cut-off distance to form the graph or inter-atomic distance checking.
            surf_check_method: method of surface checking.
                None: no check will applied;
                'ConnectedGraph' or 'C': it will check whether the surface atoms form only one connected graph. If not, mask returns False.
                'MinDistance' or 'M': it will check for atoms whose shortest distance from other atoms exceeds the cut-off distance.

        Returns:
            mask_array: ndarray[(n_batch, ), bool], where False for failed to the check.
        """
        self.BS.direct2cartesian()
        coords = self.BS.Coords
        mask = np.full(len(coords), True, )
        std_val = 1e8
        for i, coo in enumerate(coords):
            # check empty coordinates
            if len(coo) == 0:
                warnings.warn(f'Occurred empty coordinates in {i}th structure, skipped.', RuntimeWarning)
                continue
            # convert to float64
            coo = coo.astype(np.float64)
            sorted_coo = np.sort(coo[:, -1])
            std_val_old = np.inf
            tol_cont = 0  # tolerance step of std_val not decrease.

            for j in range(len(coo)):
                _centers = np.array([np.mean(_) for _ in np.array_split(sorted_coo, j+1)])
                _centers, std_val = self._k_means(coo[:, -1], _centers)
                # raise for 3 previous steps to break the loop
                if (std_val > std_val_old) or (len(_centers) < j+1):
                    tol_cont += 1
                else:
                    tol_cont = 0
                if (tol_cont > 2) or (np.min(np.abs(_centers[:, None] - _centers[None, :]) + 20.*np.eye(len(_centers))) < smoothness_threshold):
                    break
                elif std_val < 1e-5:  # completely converged
                    std_val_old = std_val
                    centers = _centers
                    break
                elif tol_cont == 0:
                    std_val_old = std_val
                    centers = _centers
            if std_val > smoothness_threshold:
                mask[i] = False
            else:
                # check surface sparseness
                surf_mask = np.abs(coo[:, -1] - np.max(centers)) < smoothness_threshold
                bott_mask = np.abs(coo[:, -1] - np.min(centers)) < smoothness_threshold
                surf_coo = coo[surf_mask]
                bott_coo = coo[bott_mask]
                if surf_check_method == ('ConnectedGraph' or 'C'):
                    n_sub_surf = self._spectrum_clustering(surf_coo, r_cutoff=r_cutoff)
                    n_sub_bott = self._spectrum_clustering(bott_coo, r_cutoff=r_cutoff)
                    if (n_sub_surf > 1) or (n_sub_bott > 1):
                        mask[i] = False
                elif surf_check_method == ('MinDistance'or 'M'):
                    surf_dist = np.linalg.norm(surf_coo[:, None] - surf_coo[None, :], axis=-1) + (r_cutoff + 10.) * np.eye(len(surf_coo))  # exclude the diag.
                    bott_dist = np.linalg.norm(bott_coo[:, None] - bott_coo[None, :], axis=-1) + (r_cutoff + 10.) * np.eye(len(bott_coo))
                    if (not np.all(np.sum(surf_dist < r_cutoff, axis=-1))) or (not np.all(np.sum(bott_dist < r_cutoff, axis=-1))):
                        mask[i] = False
        return mask


