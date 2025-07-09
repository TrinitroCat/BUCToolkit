""" Generate new structure by stochastically replace atoms in given structures """

#  Copyright (c) 2024-2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: random_replace.py
#  Environment: Python 3.12

from typing import Iterable, Dict, Any, List, Literal, Optional, Callable, Sequence, Tuple  # noqa: F401
import time

import torch as th

import numpy as np

from BM4Ckit.utils._Element_info import ATOMIC_SYMBOL


class RandReplace:
    """


    """

    def __init__(self,
                 size: int = 1,
                 seed: int = None,
                 device: str | th.device = 'cpu',
                 verbose: int = 2
                 ):
        """

        Args:
            size: the number of structures to generate.
            seed: random seed.
            device: device that program run on.
            verbose: verboseness of output information.
        """
        self.size = size
        self.seed = seed
        self.device = device
        self.verbose = verbose
        pass

    def run(self,
            X: th.Tensor,
            Element_list: List[List[str]] | List[List[int]],
            replace_element_range: Dict[str | int, Tuple[str | int]],
            replace_numbers: Sequence[int],
            batch_indices: Sequence[int] | th.Tensor | np.ndarray | None = None,

            ):
        """

        Args:
            X: Tensor[n_batch, n_atom, 3], the atom coordinates.
            Element_list: List[List[str | int]], the atomic type (element) corresponding to each row of each batch in X.
            replace_element_range: dict((symbol of element to be replaced | index of atom in X) to be replaced : tuple(symbol | atomic number of elements to replace))
            replace_numbers: the number of times that each atom is replaced. It must have the same length of `replace_element_range`.
            batch_indices: Sequence | th.Tensor | np.ndarray | None, the split points for given X, Element_list & V_init, must be 1D integer array_like.
                the format of batch_indices is same as `split_size_or_sections` in torch.split:
                batch_indices = (n1, n2, ..., nN) will split X, Element_list & V_init into N parts, and ith parts has ni atoms. sum(n1, ..., nN) = X.shape[1]

        Returns:

        """
        t_main = time.perf_counter()
        n_batch, n_atom, n_dim = X.shape
        # Check batch indices
        if batch_indices is not None:
            if n_batch != 1:
                raise RuntimeError(f'If batch_indices was specified, the 1st dimension of X must be 1 instead of {n_batch}.')
            if isinstance(batch_indices, (th.Tensor, np.ndarray)):
                batch_indices = batch_indices.tolist()
            elif not isinstance(batch_indices, Sequence):
                raise TypeError(f'Invalid type of batch_indices {type(batch_indices)}. '
                                f'It must be Sequence[int] | th.Tensor | np.ndarray | None')
            for i in batch_indices: assert isinstance(i, int), f'All elements in batch_indices must be int, but occurred {type(i)}'

        # Manage Atomic Type & Masses
        repl_atoms = list()
        for _Elem in Element_list:
            repl_atoms.append([__elem if isinstance(__elem, int) else ATOMIC_SYMBOL[__elem] for __elem in _Elem])
        repl_atoms = th.tensor(repl_atoms, dtype=th.int16, device=self.device)
        repl_atoms = repl_atoms.unsqueeze(-1).expand_as(X)  # (n_batch, n_atom, n_dim)


        pass
