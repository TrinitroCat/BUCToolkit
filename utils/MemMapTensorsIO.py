#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: MemMapTensorsIO.py
#  Environment: Python 3.12
from typing import Dict, List, Literal
import warnings
import numpy as np
import os, mmap



class MemMapTensorIO:  # TODO
    """

    """
    def __init__(
            self,
            path: str,
            mode: str = 'r'
    ):
        self.path = path
        self.mode = mode
        self.__mmp_access_dict = {'r': mmap.ACCESS_READ, 'w': mmap.ACCESS_WRITE, 'a': mmap.ACCESS_WRITE}
        self.__arr_dtype_size_dict = {  # unit: byte
            'float16': 2,
            'float32': 4,
            'float64': 8,
            'float128': 16,
            'int8': 1,
            'int16': 2,
            'int32': 4,
            'int64': 8,
        }
        self.__arr_dtype_num_dict = {
            'float16': 0,
            'float32': 1,
            'float64': 2,
            'float128': 3,
            'int8': 4,
            'int16': 5,
            'int32': 6,
            'int64': 7,
        }

    def __enter__(self):
        """
        FORMAT:
            Int64[length of data name 1]
            Str[data name 1, utf-8]
            Int64[dtype of data1]
            Int64[dim of main data 1]
            Int64*[shape of data 1]
            Byte[main data 1 of np.ndarray]
            (data 2)
            (data 3)
            ...

        """
        self.f_raw = open(self.path, self.mode)
        self.f_mmap = mmap.mmap(self.f_raw.fileno(), 1024*1024*1024, access=self.__mmp_access_dict.get(self.mode, mmap.ACCESS_WRITE))

        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f_raw.close()
        self.f_mmap.close()
        pass

    def _raw_save(
            self,
            data: Dict[str, np.ndarray]
    ):
        """
        Directly save without check
        Args:
            data:

        Returns:

        """
        for _k, _v in data.items():
            _nam = _k.encode()
            _len_dat_name = len(_nam).to_bytes(2)

            _type = self.__arr_dtype_num_dict[_v.dtype.name]
            _dim = _v.ndim.to_bytes(2)
            _shape = bytes(_v.shape)
            _dat = _v.tobytes()

            self.f_mmap.write(_nam)
            self.f_mmap.write(_len_dat_name)
            self.f_mmap.write(_type)
            self.f_mmap.write(_dim)
            self.f_mmap.write(_shape)
            self.f_mmap.write(_dat)
        self.f_mmap.flush()  # TODO


    def save(
            self,
            data: np.ndarray | List[np.ndarray] | Dict[str, np.ndarray]
    ):
        """
        Args:
            data:

        Returns:

        """
        # check
        if isinstance(data, np.ndarray):
            _data = {'1': data}
        elif isinstance(data, List):
            _data = dict()
            for i, _dat in enumerate(data):
                if isinstance(_dat, np.ndarray):
                    _data[str(i)] = _dat
                else:
                    warnings.warn(f'Invalid type of {i}th data: {type(_dat)}. Expected a numpy.ndarray.')
                    continue
        elif isinstance(data, Dict):
            for i, _dat in data.items():
                if not isinstance(_dat, np.ndarray):
                    warnings.warn(f'Invalid type of data named {i}: {type(_dat)}. Expected a numpy.ndarray.')
                    data.pop(i)
                    continue
            _data = data
        else:
            raise ValueError(f'Invalid type of data: {type(data)}')

        # main
        self._raw_save(_data)
