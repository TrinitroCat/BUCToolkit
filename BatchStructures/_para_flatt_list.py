""" Flatten Python list into 1D in parallel """

import multiprocessing as mp
import warnings
from typing import List, Iterable, Tuple
import sys

import numpy as np

def _flatten_1layer_with_check(x: List):
    """
    flatten 1 layer of x

    Args:
        x: input list

    Returns: flatten list

    """
    y = list()
    is_1d = True
    for sub_x in x:
        if isinstance(sub_x, (List, Tuple, np.ndarray)):
            y.extend(sub_x)
            is_1d = False
        else:
            y.append(sub_x)
    return y, is_1d


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

    n_chunk = len(x)//(ncore - 1)
    if n_chunk == 0: n_chunk = len(x)
    # Process Pools
    pools = mp.Pool(ncore)
    # run
    if flat_num == -1:  # flatten until 1D
        while True:
            result = [pools.apply_async(_flatten_1layer_with_check, args=(x[i * n_chunk: (i + 1) * n_chunk],)) for i in range(ncore)]
            x_mid = list()
            [x_mid.extend(_x.get()[0]) for _x in result]
            is_1d = True
            for _x in result:
                if not _x.get()[1]:
                    is_1d = False
                    break
            if is_1d:
                break
            x = x_mid
    elif isinstance(flat_num, int) and (flat_num > 0):  # flatten specified times
        for i in range(flat_num):
            result = [pools.apply_async(_flatten_1layer_with_check, args=(x[i * n_chunk: (i + 1) * n_chunk],)) for i in range(ncore)]
            x_mid = list()
            [x_mid.extend(_x.get()[0]) for _x in result]
            x = x_mid
    else:
        raise ValueError('`flat_num` must be an integer greater than 0, or == -1')

    pools.close()
    pools.join()
    return x


if __name__ == '__main__':
    a = [2, [3411, 422], [53, 44, [231, 44], 23], 351]
    b = _flatten_1layer_with_check(a)
    c = flatten(a)
    pass
