""" Flatten Python list into 1D in parallel """

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: _para_flatt_list.py
#  Environment: Python 3.12

import multiprocessing as mp
import warnings
from typing import List, Iterable, Tuple
from itertools import chain
import sys
import time

import numpy as np
import torch as th

def _flatten_until_1d(x: List):
    """
    Recursively flatten x until

    Args:
        x: input list

    Returns: flatten list

    """
    y = list()
    for sub_x in x:
        if isinstance(sub_x, (List, Tuple, np.ndarray)):
            y.extend(_flatten_until_1d(sub_x))
        else:
            y.append(sub_x)
    return y

def _flatten_1time(x:List):
    """
    Flatten one layer of x

    Args:
        x:

    Returns: List
    """
    y = list()
    for sub_x in x:
        if isinstance(sub_x, (List, Tuple, np.ndarray)):
            y.extend(sub_x)
        else:
            y.append(sub_x)
    return y

def flatten(x, flat_num: int=-1, ncore: int=-1):
    """
    flatten x in parallel

    Args:
        x: input list.
        flat_num: number of flattens. -1 for flatten x until 1d-list.
        ncore: number of parallel cores.

    Returns:

    """
    #mp.set_start_method('spawn', force=True)
    tot_core = mp.cpu_count()
    if ncore == -1:
        ncore = tot_core
    elif (ncore > tot_core) or (ncore <= 0) or (not isinstance(ncore, int)):
        raise ValueError('`ncore` must be an integer between 0 and total cpu cores.')
    if sys.platform == 'win32' and ncore != 1:
        warnings.warn('Windows OS does not support multi-process yet. `ncore` was automatically set to 1.', RuntimeWarning)
        ncore = 1

    n_chunk = len(x)//(ncore - 1) if ncore > 1 else len(x)
    if n_chunk == 0: n_chunk = len(x)
    # Process Pools
    pools = mp.Pool(ncore)
    # run
    if flat_num == -1:  # flatten until 1D
        z = list()
        result = [pools.apply_async(_flatten_until_1d, args=(x[i * n_chunk: (i + 1) * n_chunk],)) for i in range(ncore)]
        [z.extend(_.get()) for _ in result]

    elif isinstance(flat_num, int) and (flat_num > 0):  # flatten specified times
        z = x
        for i in range(flat_num):
            n_chunk = len(z) // (ncore - 1) if ncore > 1 else len(z)
            if n_chunk == 0: n_chunk = len(z)
            result = [pools.apply_async(_flatten_1time, args=(z[i * n_chunk: (i + 1) * n_chunk],)) for i in range(ncore)]
            z = list()
            [z.extend(_.get()) for _ in result]
    else:
        raise ValueError('`flat_num` must be an integer greater than 0, or == -1')

    pools.close()
    pools.join()

    return z
