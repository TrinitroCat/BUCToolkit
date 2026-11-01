
#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0.1
#  File: DataLoaders.py
#  Environment: Python 3.12

from typing import Any, Callable, Dict, List, Self, Tuple, Literal, Sequence

import numpy as np
import torch as th

from BUCToolkit.utils._CheckModules import check_module
from BUCToolkit.BatchStructures import Batch

# check modules
_pyg = check_module('torch_geometric.data')
if _pyg is not None:
    pygBatch = _pyg.Batch
else:
    pygBatch = Batch


class BatchStructuresDataLoader:
    """
    A Data loader to form BatchStructure
    """

    def __init__(
            self,
            data_path: str,
            batch_size: int,
            device: str | th.device = 'cpu',
            shuffle: bool = True,
            is_train: bool = True,
            data_names: Sequence[str] | None = None
    ) -> None:
        self.data_path = data_path
        self.data_names = data_names
        self.batchsize = batch_size
        self._index = 0
        self.device = device
        # TODO <<<
        raise NotImplementedError
