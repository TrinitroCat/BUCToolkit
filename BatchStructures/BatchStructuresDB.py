import lmdb
from .BatchStructuresBase import BatchStructures
import numpy as np
import tarfile as tar
import logging
import joblib as jb
from typing import Dict, List, Tuple, Literal, Any


class BatchStructuresDB(BatchStructures):
    """

    """
    def __init__(self, ):
        """
        BatchStructures DataBase that map BatchStructures to Disk.
        """
        super().__init__()
        self.env: lmdb.Environment | None = None  # lmdb envir
        self.db_file: str | None = None  # file of database
        self.num_chunk: int | None = None

    def _initialize(self,):
        self.env = lmdb.open(self.db_file)

    def _end(self, ):
        self.env.close()

    def _load_from_file(self, n_core:int=-1):
        """ load data from lmdb file """
        # def a func of single thread
        def _single_read(
                _dat: Dict[Literal['Cell', 'Coords_type', 'Coords', 'Atoms', 'AtomicNumbers', 'Energy', 'Forces', 'Label'], Any]
        ):
            return _dat

        self._initialize()
        txn = self.env.begin()
        try:
            _data_dict = jb.Parallel(n_core)(jb.delayed(_single_read)())
            txn.commit()

            self._Sample_ids = list()
            # Structure part
            self.Atom_list = None
            self.Atomic_number_list = None
            self.Cells: List[np.ndarray] = list()  # cell vectors
            self.Coords_type: List[Literal['C', 'D']] = list()  # coordinate type
            self.Coords: List[np.ndarray] = list()  # atomic coordinates
            self.Fixed: List[np.ndarray] = list()  # fixed atoms masks, 0 for fixed and 1 for free.
            self.Elements: List[List[str]] = list()  # elements
            self.Numbers: List[List[int]] = list()  # atom number of each element
            # Properties part
            self.Energies = None  # energies of structures. None | List[float]
            self.Forces = None  # forces of structures. None | List[np.NdArray[N, 3]]
            self.Dist_mat = None  # distance matrices of structures. None | List[np.NdArray[N, N]]
            # Others
            self._indices = None
            Z_dict = {key: i for i, key in enumerate(self._ALL_ELEMENTS, 1)}  # a dict which map element symbols into their atomic numbers.
            self._Z_dict = Z_dict
            self.Labels = None  # structure labels

        except Exception as e:
            self.logger.info(f'An Error Occurred at :\n{e}')
        finally:
            self._end()

    def _load_from_mem(self, data):
        """ load data from BatchStructures in memory """
        if not isinstance(data, BatchStructures):
            raise TypeError('`batch_structures` must be BatchStructures')
        if len(self) > 0:
            raise RuntimeError('Ths BatchStructuresDB has already contained data. To append data, use `self.append` instead.')

        self.Coords = data.Coords
        self.Fixed = data.Fixed
        self.Forces = data.Forces

        self.Numbers = data.Numbers
        self.Elements = data.Elements
        # Others are sequentially filled
        self.Cells = data.Cells
        self.Coords_type = data.Coords_type
        self._Sample_ids = data.Sample_ids
        self.Energies = data.Energies
        self.Labels = data.Labels

    def load(self, path:str | None=None, BS_in_mem: BatchStructures | None=None):
        """

        Args:
            path: the file path of database. Exclusive with BS_in_mem.
            BS_in_mem: BatchStructures data in memory. Exclusive with path.

        Returns: None
        """
        if (path is not None) and (BS_in_mem is None):  # load from file
            self.db_file = path
            self._load_from_file()
        elif (path is None) and (BS_in_mem is not None):  # load from mem
            self._load_from_mem()
        elif (path is not None) and (BS_in_mem is not None):
            raise RuntimeError('Argument `path` and `BS_in_mem` are exclusive. Please specify only one of them.')
        else:
            raise RuntimeError('Both `path` and `BS_in_mem` are not specified.')

    def insert(sid, name):
        txn = self.env.begin(write=True)
        txn.put(str(sid).encode(), name.encode())
        txn.commit()

    def delete(env, sid):
        txn = env.begin(write=True)
        txn.delete(str(sid).encode())
        txn.commit()

    def update(env, sid, name):
        txn = env.begin(write=True)
        txn.put(str(sid).encode(), name.encode())
        txn.commit()

    def search(env, sid):
        txn = env.begin()
        name = txn.get(str(sid).encode())
        return name

    def display(env):
        txn = env.begin()
        cur = txn.cursor()
        for key, value in cur:
            print(key, value)