"""
Check whether the structures in `BatchStructures` have abnormalities which contain:
    * too large Forces (if forces are not None)
    * too large Coordinates
    * too close distance
    * too large or close distance between adsorbate and slab

"""
#  Copyright (c) 2025.7.30, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: check_if_converge.py
#  Environment: Python 3.12
import warnings
from typing import List, Callable, Tuple, Any, Dict
import os

import numpy as np
import joblib as jb
from BM4Ckit.utils.AtomicNumber2Properties import elements_to_atomic_numbers


def check_if_converge(
        data,
        forces_threshold: float = 0.05,
):
    """

    Args:
        data: BatchStructures data.
        forces_threshold: force threshold for convergence.

    Returns: list of unconverged structure names.

    """
    # check
    data.change_mode('L')
    if data.Forces is None:
        warnings.warn('Forces are not available, skipping.', RuntimeWarning)
        return list()
    unconverged_list = list()
    for i, coo in enumerate(data.Forces):
        if np.any(coo > forces_threshold):
            unconverged_list.append(data.Sample_ids[i])

    return unconverged_list

def check_if_abnormal(
        data,
        coo_upper_bound: float = 25.,
        distance_threshold: float = 0.7,
        adsorbate_indices: List[int]|None = None,
        adsorbate_elements: List[str]|None = None,
        exclude_indices: List[int]|None = None,
        exclude_elements: List[str]|None = None,
        ads_slab_dist_upper_bound: float = 5.,
        ads_slab_dist_lower_bound: float = 1.3,
):
    """
    Check whether the structures in `BatchStructures` have abnormalities.
    Args:
        data: the data of BatchStructures.
        coo_upper_bound: the upper bound of coordinates.
        distance_threshold: the distance threshold. distance less than it would be set to abnormal.
        adsorbate_indices: atomic indices of absorbate. the selected atoms as adsorbate are the combination of `adsorbate_indices` and `adsorbate_elements`
        adsorbate_elements: elements of selected adsorbates. the selected atoms as adsorbate are the combination of `adsorbate_indices` and `adsorbate_elements`
        exclude_indices: the indices of atoms that do not involve in abnormal check. the REST atom will be considered as SLAB.
        exclude_elements: the indices of atoms that do not involve in abnormal check. the REST atom will be considered as SLAB. it will be combination with `exclude_indices`.
        ads_slab_dist_upper_bound: distance upper bound between adsorbate and slab.
        ads_slab_dist_lower_bound: distance lower bound between adsorbate and slab.

    Returns:

    """
    # TODO: setting specific cut-off distance for different atom pairs. for example, use sum of covalent radii in a pre-stored dict.
    def parse_indx2mask(_indices, _elements, all_elements):
        _elements = elements_to_atomic_numbers(_elements) if _elements is not None else list()
        all_elements_set = set(all_elements)
        _mask = np.full_like(all_elements, False, dtype=bool)
        if _indices is not None:
            _mask[_indices] = True
        for ii, _elem in enumerate(_elements):
            if _elem in all_elements_set:
                _tmp_mask = (all_elements == _elem)
                _mask = _mask + _tmp_mask
            else:
                warnings.warn(f"The given element {_elem} not in {all_elements_set}.", RuntimeWarning)

        return _mask  # (n_atoms, )

    data.change_mode('L')
    data.direct2cartesian()
    data.generate_atomic_number_list()
    data.generate_dist_mat()
    err_config_list = list()
    for i, coo in enumerate(data.Coords):
        is_overflow = np.any(coo > coo_upper_bound)
        is_too_close = np.any(data.Dist_mat[i] + 10. * np.eye(data.Dist_mat[i].shape[-1]) < distance_threshold)

        atm_num_arr = np.asarray(data.Atomic_number_list[i])
        # parse ads.
        ads_mask = parse_indx2mask(adsorbate_indices, adsorbate_elements, atm_num_arr)
        exclude_mask = parse_indx2mask(exclude_indices, exclude_elements, atm_num_arr)
        slab_mask = np.full_like(atm_num_arr, True, dtype=bool)
        slab_mask[exclude_mask] = False
        slab_mask[ads_mask] = False
        if np.any(ads_mask) and np.any(slab_mask):
            ads_coo = coo[ads_mask]
            rest_coo = coo[slab_mask]
            ads_slab_dist_mat = np.linalg.norm(ads_coo[:, None, ...] - rest_coo[None, ...], axis=-1)
            is_ads_too_far = np.all(ads_slab_dist_mat > ads_slab_dist_upper_bound)
            is_ads_too_close = np.any(ads_slab_dist_mat < ads_slab_dist_lower_bound)
        else:
            is_ads_too_far = False
            is_ads_too_close = False

        if is_overflow or is_too_close or is_ads_too_far or is_ads_too_close:
            err_config_list.append(data.Sample_ids[i])

    return err_config_list

def batched_check_files(
        path: str,
        file_loader: Callable,
        file_loader_args: Tuple[Any] = tuple(),
        file_loader_kwargs: Dict[str, Any] | None =None,
        file_list: List[str] | None = None,
        n_core: int = 1,
        is_check_convergence: bool = True,
        force_threshold: float = 0.05,
        is_check_configuration: bool = True,
        coo_upper_bound: float = 25.,
        distance_threshold: float = 0.7,
        adsorbate_indices: List[int]|None = None,
        adsorbate_elements: List[str]|None = None,
        exclude_indices: List[int]|None = None,
        exclude_elements: List[str]|None = None,
        ads_slab_dist_upper_bound: float = 5.,
        ads_slab_dist_lower_bound: float = 1.3,
) -> Tuple[Dict, Dict]:
    """
    Check multiple files in given path in parallel.
    Args:
        path: the path of files to check.
        file_loader: the function that takes a file path and returns an `BatchStructure` type data. All data will be loaded in memory in priori, and then checking.
        file_loader_args: arguments passed to `file_loader`.
        file_loader_kwargs: keyword arguments passed to `file_loader`.
        file_list: the list of files to check. if None, all files in the given path are checked.
        n_core: the number of CPU cores to use.
        is_check_convergence: whether to check forces convergence.
        force_threshold: if is_check_convergence is True, the force threshold to check.
        is_check_configuration: whether to check configurations abnormalities.
        coo_upper_bound: the upper bound of coordinates.
        distance_threshold: the distance threshold. distance less than it would be set to abnormal.
        adsorbate_indices: atomic indices of absorbate. the selected atoms as adsorbate are the combination of `adsorbate_indices` and `adsorbate_elements`
        adsorbate_elements: elements of selected adsorbates. the selected atoms as adsorbate are the combination of `adsorbate_indices` and `adsorbate_elements`
        exclude_indices: the indices of atoms that do not involve in abnormal check. the REST atom will be considered as SLAB.
        exclude_elements: the indices of atoms that do not involve in abnormal check. the REST atom will be considered as SLAB. it will be combination with `exclude_indices`.
        ads_slab_dist_upper_bound: distance upper bound between adsorbate and slab.
        ads_slab_dist_lower_bound: distance lower bound between adsorbate and slab.

    Returns: Two Dict of {file_name: name_of_structures_not_converged} and {file_name: name_of_abnormal_structures}

    """
    if file_loader_kwargs is None:
        file_loader_kwargs = dict()
    if file_list is None:
        file_list = os.listdir(path)
    file_list = [os.path.join(path, _f) for _f in file_list]

    def _check_single(file_name):
        data = file_loader(file_name, *file_loader_args, **file_loader_kwargs)
        if is_check_convergence:
            converge_info = check_if_converge(data, force_threshold)
        else:
            converge_info = list()
        if is_check_configuration:
            abnormal_info = check_if_abnormal(
                data,
                coo_upper_bound = coo_upper_bound,
                distance_threshold = distance_threshold,
                adsorbate_indices = adsorbate_indices,
                adsorbate_elements = adsorbate_elements,
                exclude_indices = exclude_indices,
                exclude_elements = exclude_elements,
                ads_slab_dist_upper_bound = ads_slab_dist_upper_bound,
                ads_slab_dist_lower_bound = ads_slab_dist_lower_bound,
            )
        else:
            abnormal_info = list()
        basename = os.path.basename(file_name)
        return {basename: converge_info}, {basename: abnormal_info}

    _para = jb.parallel.Parallel(n_core, )
    check_dict_list = _para(jb.delayed(_check_single)(filename) for filename in file_list)
    if check_dict_list is None:
        raise RuntimeError('Occurred None data.')
    check_dict_conv = dict()
    check_dict_abnormal = dict()
    for _check_dict in check_dict_list:
        check_dict_conv.update(_check_dict[0])
        check_dict_abnormal.update(_check_dict[1])

    return check_dict_conv, check_dict_abnormal
