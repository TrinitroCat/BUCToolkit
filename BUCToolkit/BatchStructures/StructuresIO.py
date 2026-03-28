"""
Contiguous dumping and reading arrays data by memory mapping.

"""
#  Copyright (c) 2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: MemMapTensorsIO.py
#  Environment: Python 3.12
from typing import Dict, List, Literal, Tuple, ByteString
from io import BufferedRandom
import warnings, os, mmap, copy, gc, math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Any, Literal, Sequence, List

import numpy as np
import torch as th
from torch import nn

from BUCToolkit._version import __version__
from BUCToolkit.BatchStructures.BatchStructuresBase import BatchStructures
from BUCToolkit.utils.ElemListReduce import elem_list_reduce


class ArrayDumper:
    """
    Arrays Dumper for continuing writing arrays to the disk during a contiguous iteration process.
    """
    def __init__(
            self,
            path: str,
            mode: Literal['w', 'x', 'a'] = 'x',
            cache_size: int = 4096,
            head_order: Literal['<', '>'] = '<',
            use_mmap: bool = False,
    ):
        """

        Args:
            path: the file path to save arrays. The File will not open until `self.initialize()` is called.
            mode: the mode of dumper. 'w' for writing & overwriting, 'x' for writing only if file does not exist, 'a' for appending.
            cache_size: the max cache size in kilobytes (kb) to flush to the disk.
            head_order: the head order of the array head information. '<' is the little order, and '>' is the big order.
            use_mmap: whether to use memory map to save arrays.
        """
        # init vars
        self._dump_file = None  # the TextIOWrapper of the file
        self._mmp_f: BufferedRandom|mmap.mmap|None = None  # the mmap obj of the file
        self._has_initialized = False
        self._has_started = False
        self._n_arrays = None  # number of arrays in each collect step
        self._n_groups = None  # number of the array groups.
        self._nbytes_list_to_check = None  # check bytes when collect in each step.
        self._count: int|None = None  # count the number of steps to collect
        self._cache_size_now: int|None = None
        self._ptr: int|None = None  # the current pointer/indices
        self._endptr: int|None = None  # the endpoint of the file
        self._current_group_head_position: int|None = None  # the start position of the current group head
        self._use_dynamic_steps = False  # whether to use dynamic steps. False is to use fixed steps.
        #self.initialize()
        # check input
        self.reset_args(
            path=path,
            mode=mode,
            cache_size=cache_size,
            head_order=head_order,
            use_mmap=use_mmap,
        )
        # test BOM
        if len('a'.encode(self._str_fmt)) != 2:
            raise NotImplementedError(
                f'You may check here whether the encode format {self._str_fmt} is correct. '
                f'Normally it would not be added BOM at the start of bytes, thus resulting 2 bytes, but here string "a" '
                f'is encoded into {len("a".encode(self._str_fmt))} bytes instead.'
            )

    def reset_args(
            self,
            path: str,
            mode: Literal['w', 'x', 'a'] = 'x',
            cache_size: int = 4096,
            head_order: Literal['<', '>'] = '<',
            use_mmap: bool = False,
    ):
        """
        Reset args values BEFORE initializing.
        Returns: None

        """
        if self._has_initialized:
            warnings.warn(f'The dumper has been initialized already. Resetting args is unavailable.')
            return None
        self.path = str(path)
        self.cache_size = int(cache_size) * 1024  # convert to bytes
        if not self.cache_size > 0: raise ValueError(f"cache_size must be greater than 0, but got {cache_size}")
        self.use_mmap = use_mmap
        if mode in ('w', 'x', 'a'):
            self.mode = mode
        else:
            raise ValueError(f"mode must be 'w', 'x', or 'a', but got {mode}")
        if os.path.isdir(self.path):
            raise IOError(f"The path '{self.path}' has already exist as a directory.")
        if (not os.path.isfile(self.path)) and (self.mode == 'a'):
            # warnings.warn(f'The mode is "a" but file {self.path} does not exist. Hence, mode has been reset to "w".')
            self.mode = 'w'

        __ORDER1 = {'<': 'utf-16-le', '>': 'utf-16-be'}
        if not head_order in __ORDER1: raise ValueError(f'head_order must be "<" or ">", but got {head_order}.')
        self.head_order = head_order
        self._str_fmt = __ORDER1[head_order]
        __ORDER2: Dict[Literal['<', '>'], Literal['little', 'big']] = {'<': 'little', '>': 'big'}
        self._num_fmt: Literal['little', 'big'] = __ORDER2[head_order]

    def initialize(self, ):
        r"""
        do initialization before saving arrays, writing the head information.
        Note:
            Use below format as the full file header (in total 16 byte):
                `head_order``magic``version``n_groups`
                 char2       char4  2*int1   uint8
                  where char2 is one unicode-16 that use 2 bytes, magic is hard coded as "BM" in unicode-16, and version is 2 8-bit-ints.
                 `head_order` is Literal['<', '>'], hence one can try both little and big order until one of "<" and ">" is read.
                 `n_groups` is the total number of groups in the mmap file. It will be dynamically updated (+1) once `self.start`
                 or `start_from_arrays` is called.
            Use below format as the array data header, where int8 (here "8" is 8 bytes) "0" is as the delimiter among shapes:
                `char``n_cycle``n_array``dtype1``shape1[]`0`dtype2``shape2[]`0...`dtype_n``shape_n[]`0`byte_data`...
                 HEAD  uint8    uint8    char8   [int8, ] 0 char8   [int8, ] 0 ...
                  where `char` is the character number indicate the array head information, which is hard coded as "HEAD",
                  8 bytes `dtype` are in form of `order``type``length(byte)`, e.g., "<i4" is 32 bit integer in the little order.
                  `order` can be "<", "|", or ">"; `type` can be "i" (signed int), "u" (unsigned int), "f" (float), or "c" (complex), they are
                  both in unicode-16 (total 4 bytes), and the last number `length` is a 4-byte-int.
        """
        try:
            if self._has_initialized:
                warnings.warn('Cannot create a new dumping file because it is already initialized.')
                return
            _open_mode = {'w': 'wb+', 'x': 'xb+', 'a': 'rb+'}.get(self.mode)
            self._n_groups = 0
            if self._dump_file is None:
                self._dump_file = open(self.path, _open_mode)
            else:
                warnings.warn('??? BUG: self is not initialized but there is already an opened dumping file ???')
                return

            if self.mode == 'a':
                if self.use_mmap:
                    self._mmp_f = mmap.mmap(self._dump_file.fileno(), 0, access=mmap.ACCESS_WRITE)  # allocate 16 bytes to write the head
                else:  # simply reference the normal IOWrapper
                    self._mmp_f = self._dump_file
                self._mmp_f.seek(0)
                file_head = self._mmp_f.read(16)
                self._parse_head(file_head)
                # jump to the end to append
                self._mmp_f.seek(0, 2)
                self._ptr = self._mmp_f.tell()
                self._endptr = self._mmp_f.tell()
                self._has_initialized = True
            else:
                self._dump_file.truncate(16)
                if self.use_mmap:
                    self._mmp_f = mmap.mmap(self._dump_file.fileno(), 0, access=mmap.ACCESS_WRITE)  # allocate 16 bytes to write the head
                else:  # simply reference the normal IOWrapper
                    self._mmp_f = self._dump_file
                self._mmp_f.write(f'{self.head_order}BM'.encode(self._str_fmt))
                _v = __version__.split('.', 2)
                v1 = int(_v[0])
                v2 = int(_v[1])
                if v1 >= 255 or v2 >= 255:
                    raise NotImplementedError(f'The version numbers have been reached {_v}. How frightful!')
                __version_bytes = bytearray(2)  # It is frightful that version number > 255
                __version_bytes[0:1] = v1.to_bytes(1, byteorder=self._num_fmt)
                __version_bytes[1:2] = v2.to_bytes(1, byteorder=self._num_fmt)
                self._mmp_f.write(__version_bytes)  # version: `v1`.`v2`
                self._mmp_f.write(self._n_groups.to_bytes(8, byteorder=self._num_fmt))
                self._has_initialized = True
                self._ptr = 16
                self._endptr = 16

        except Exception as e:
            raise RuntimeError(f'Failed to initialize dumping file `{self.path}`: {e}')

        finally:
            self._tmp_close()

    def _parse_head(self, file_head: bytes):
        """
        Parse the head of the mmap file for append mode.
        """
        try:
            if len(file_head) != 16:
                raise ValueError(f'The file head is not complete.')
            # find the head order
            try_order_le = file_head[:2].decode('utf-16-le')
            try_order_be = file_head[:2].decode('utf-16-be')
            if try_order_le == '<':
                _str_fmt = 'utf-16-le'
                self._num_fmt: Literal['little', 'big'] = 'little'
            elif try_order_be == '>':
                _str_fmt = 'utf-16-be'
                self._num_fmt: Literal['little', 'big'] = 'big'
            else:
                raise ValueError(f'Unexpected endianness: {try_order_le}/{try_order_be}.')
            if _str_fmt != self._str_fmt:
                warnings.warn(f'Inconsistent endianness between input and read file: {self._str_fmt} and {_str_fmt}. ')
            self._str_fmt = _str_fmt
            # check magik
            magik = file_head[2:6].decode(self._str_fmt)
            if magik != "BM":
                raise ValueError(f'Unknown file format: {magik}.')
            # check version
            v1 = int.from_bytes(file_head[6:7], self._num_fmt, signed=False)
            v2 = int.from_bytes(file_head[7:8], self._num_fmt, signed=False)
            _version_now = __version__.split('.', 2)
            v1_now = int(_version_now[0])
            v2_now = int(_version_now[1])
            if v1_now >= 255 or v2_now >= 255:
                raise NotImplementedError(f'The version numbers have been reached {_version_now}. How frightful!')
            if (v1, v2) != (v1_now, v2_now):
                warnings.warn(
                    f"The file {self.path} (ver \"{'.'.join((str(v1), str(v2)))}\") is incompatible with "
                    f"the current version \"{__version__}\".",
                    RuntimeWarning,
                )
            # read the group number
            self._n_groups = int.from_bytes(file_head[8:16], self._num_fmt, signed=False)

        except Exception as e:
            raise RuntimeError(f'Failed to parse file header: {e}')

    def allocate(self, size: int):
        """
        Append a new space at the end of the mmap file. It will keep the file and mmap opening.
        Args:
            size (int): new space size in bytes.
        Returns: None

        """
        if not self._has_initialized:
            self.initialize()
        if size <= 0:
            raise ValueError(f'size must be greater than 0, but got {size}.')

        try:
            if self._dump_file.closed:
                self._dump_file = open(self.path, 'rb+')

            self._endptr += size
            if self.use_mmap:
                if self._mmp_f.closed:
                    self._mmp_f = mmap.mmap(self._dump_file.fileno(), 0, access=mmap.ACCESS_WRITE)
                self._mmp_f.resize(self._endptr)
            else:
                self._mmp_f = self._dump_file
                self._mmp_f.truncate(self._endptr)
            self._mmp_f.seek(self._ptr)
        except Exception as e:
            self._tmp_close()
            raise RuntimeError(f'Failed to allocate new {size} bytes to `{self.path}`: {e}')

    def start_from_arrays(self, steps: int, *arrays: np.ndarray, force: bool = False):
        """
        Write the head information of array from a prototype arrays, allocate the blank space, thus starting a new dumping series.
        Note that arrays input here are only a prototype that will NOT be written to the disk.
        One must call `self.step` method to really write arrays.
        Args:
            steps: the iteration number of arrays, which will be dumped. If `steps` = -1, a dynamic steps will be used.
            *arrays: sequence of arrays as the prototype.
            force: whether to force starting a new dumping series even if there are still blanks at the end of the mmap file.

        Returns: None
        """
        try:
            # check file status
            if not self._has_initialized:
                self.initialize()
            if self._ptr < self._endptr:
                if force:
                    warnings.warn(
                        f'There are still {self._endptr - self._ptr} bytes are blank at the end of the mmap file. '
                        f'Now these blanks will be dropped.',
                        RuntimeWarning
                    )
                    self.truncate()
                else:
                    warnings.warn(
                        f'There are still {self._endptr - self._ptr} bytes are blank at the end of the mmap file. '
                        f'starting process aborted.',
                        RuntimeWarning
                    )
                    return None
            # check inputs:
            if steps == -1:
                self._use_dynamic_steps = True
                steps = 500
            elif steps <= 0:
                raise ValueError(f'A non-positive steps value {steps} is absurd.')
            else:
                self._use_dynamic_steps = False
            # count head length
            self._nbytes_list_to_check = list()
            self._n_arrays = len(arrays)
            _head_nbytes = 24  # `char``n_cycle``n_array`, 3 * 8
            head_len_list = list()
            arr_type_list: List[Tuple[str, str, int]] = list()
            arr_tol_size = 0
            for i, arr in enumerate(arrays):
                if 0 in arr.shape:
                    raise RuntimeError(
                        f'Some dimension of the {i}-th array are zero, which means this array is actually empty. '
                        f'Dumping such array is MEANINGLESS. Writing is REFUSED.'
                    )
                _l = arr.ndim * 8
                _head_nbytes += 8 + _l + 8  # bytes of 'dtype + shape + 0'
                head_len_list.append(_l)
                _dtype = arr.dtype.str  # `order``type``len`
                arr_type_list.append((str(_dtype[0]), str(_dtype[1]), int(_dtype[2:])))
                # calc. the arrays size
                self._nbytes_list_to_check.append(arr.nbytes)
                arr_tol_size += arr.nbytes

            dump_content = bytearray(_head_nbytes)
            dump_content[:8] = 'HEAD'.encode(self._str_fmt)
            dump_content[8:16] = steps.to_bytes(8, self._num_fmt, signed=False)  # n_cycle
            dump_content[16:24] = self._n_arrays.to_bytes(8, self._num_fmt, signed=False)  # n_arrays
            _ptr = 24
            for i, _shape_len in enumerate(head_len_list):
                # dtype, 8 bytes
                dump_content[_ptr:_ptr + 8] = (
                        ''.join(arr_type_list[i][:2]).encode(self._str_fmt) + arr_type_list[i][2].to_bytes(4, self._num_fmt, signed=False)
                )
                _ptr += 8
                # shape, n * 8 bytes
                dump_content[_ptr:_ptr + _shape_len] = b''.join(_.to_bytes(8, self._num_fmt, signed=False) for _ in arrays[i].shape)
                _ptr += _shape_len
                # delimiter "0"
                dump_content[_ptr: _ptr + 8] = (0).to_bytes(8, self._num_fmt, signed=False)
                _ptr += 8

            # write
            self._n_groups += 1
            self.allocate(_head_nbytes + arr_tol_size * steps)
            self._current_group_head_position = self._ptr
            self._mmp_f.write(dump_content)
            # reset the global n_group information:
            #   it should be `self._mmp_f[8:16] = self._n_groups.to_bytes(8, self._num_fmt, signed=False)`
            #   while _io.BufferedRandom class does not support directly indexed modification,
            #   so that uses below ptr operation.
            __tmp_ptr = self._mmp_f.tell()
            self._mmp_f.seek(8)
            self._mmp_f.write(self._n_groups.to_bytes(8, self._num_fmt, signed=False))
            self._mmp_f.seek(__tmp_ptr)

            self._ptr += _head_nbytes
            self._mmp_f.flush()
            self._has_started = True
            self._count = 0
            self._cache_size_now = 0

        except Exception as e:
            self.close()
            raise RuntimeError(f'Failed to start in the file `{self.path}`. ERROR: {e}')

    def start(
            self,
            steps: int,
            dtype_list: List[str],
            shape_list: List[Tuple[int, ...]],
            force: bool = False
    ):
        """
        Write the head information of array, allocate the blank space, thus starting a new dumping series.
        Args:
            steps: the iteration number of arrays, which will be dumped. If `steps` = -1, a dynamic steps will be used.
            dtype_list: list of the arrays' dtypes.
            shape_list: list of the arrays' shapes.
            force: whether to force starting a new dumping series even if there are still blanks at the end of the mmap file.

        Returns: None
        """
        try:
            # check file status
            if not self._has_initialized:
                self.initialize()

            if self._ptr < self._endptr:
                if force:
                    warnings.warn(
                        f'There are still {self._endptr - self._ptr} bytes are blank at the end of the mmap file. '
                        f'Now these blanks will be dropped.',
                        RuntimeWarning
                    )
                    self.truncate()
                else:
                    warnings.warn(
                        f'There are still {self._endptr - self._ptr} bytes are blank at the end of the mmap file. '
                        f'starting process aborted.',
                        RuntimeWarning
                    )
                    return None
            # check inputs:
            if steps == -1:
                self._use_dynamic_steps = True
                steps = 500
            elif steps <= 0:
                raise ValueError(f'A non-positive steps value {steps} is absurd.')
            else:
                self._use_dynamic_steps = False
            # count head length
            self._nbytes_list_to_check = list()
            self._n_arrays = len(dtype_list)
            if self._n_arrays != len(shape_list):
                raise ValueError(f'The length of `dtype_list` and `shape_list` must match, but got {self._n_arrays} and {len(shape_list)}.')
            _head_nbytes = 24  # `char``n_cycle``n_array`, 3 * 8
            head_len_list = list()
            arr_type_list: List[Tuple[str, str, int]] = list()
            arr_tol_size = 0
            for i in range(self._n_arrays):
                if 0 in shape_list[i]:
                    raise RuntimeError(
                        f'Some dimension of the {i}-th array are zero, which means this array is actually empty. '
                        f'Dumping such array is MEANINGLESS. Writing is REFUSED.'
                    )
                _l = len(shape_list[i]) * 8
                _head_nbytes += 8 + _l + 8  # bytes of 'dtype + shape + 0'
                head_len_list.append(_l)
                _dtype = dtype_list[i]  # `order``type``len`
                nbytes = int(_dtype[2:])
                # for Unicode in numpy, each char has 4 bytes rather than 1 byte.
                if str(_dtype[1]) == 'U':
                    nbytes *= 4
                arr_type_list.append((str(_dtype[0]), str(_dtype[1]), nbytes))
                # calc. the arrays size
                tol_nbytes = nbytes * math.prod(shape_list[i])
                self._nbytes_list_to_check.append(tol_nbytes)
                arr_tol_size += tol_nbytes

            dump_content = bytearray(_head_nbytes)
            dump_content[:8] = 'HEAD'.encode(self._str_fmt)
            dump_content[8:16] = steps.to_bytes(8, self._num_fmt, signed=False)  # n_cycle
            dump_content[16:24] = self._n_arrays.to_bytes(8, self._num_fmt, signed=False)  # n_arrays
            _ptr = 24
            for i, _shape_len in enumerate(head_len_list):
                # dtype, 8 bytes
                dump_content[_ptr:_ptr + 8] = (
                        ''.join(arr_type_list[i][:2]).encode(self._str_fmt)
                        + arr_type_list[i][2].to_bytes(4, self._num_fmt, signed=False)
                )
                _ptr += 8
                # shape, n * 8 bytes
                dump_content[_ptr:_ptr + _shape_len] = b''.join(_.to_bytes(8, self._num_fmt, signed=False) for _ in shape_list[i])
                _ptr += _shape_len
                # delimiter, "0"
                dump_content[_ptr: _ptr + 8] = (0).to_bytes(8, self._num_fmt, signed=False)
                _ptr += 8

            # write
            self._n_groups += 1
            self.allocate(_head_nbytes + arr_tol_size * steps)
            self._current_group_head_position = self._ptr
            self._mmp_f.write(dump_content)
            # reset the global n_group information:
            #   it should be `self._mmp_f[8:16] = self._n_groups.to_bytes(8, self._num_fmt, signed=False)`
            #   while _io.BufferedRandom class does not support directly indexed modification,
            #   so that uses below ptr operation.
            __tmp_ptr = self._mmp_f.tell()
            self._mmp_f.seek(8)
            self._mmp_f.write(self._n_groups.to_bytes(8, self._num_fmt, signed=False))
            self._mmp_f.seek(__tmp_ptr)

            self._ptr += _head_nbytes
            self._mmp_f.flush()
            self._has_started = True
            self._count = 0
            self._cache_size_now = 0

        except Exception as e:
            self.close()
            raise RuntimeError(f'Failed to start in the file `{self.path}`. ERROR: {e}')

    def step(
            self,
            *arrays: np.ndarray,
    ):
        """
        Do a collect step that store a list of arrays.
        Args:
            *arrays (np.ndarray): arrays to collect.
        Returns: None

        """
        try:
            if not self._has_started:
                raise RuntimeError(f'The dumping series was not started. Please call `self.start(...)/start_from_arrays(...)` first.')
            if self._n_arrays != len(arrays):
                raise RuntimeError(
                    f'Inconsistent array number between the record in `self.start` and here input. '
                    f'Expected {self._n_arrays}, but got {len(arrays)}.'
                )
            if self._ptr >= self._endptr:
                if not self._use_dynamic_steps:
                    raise RuntimeError(f'Data are out of range.')
                else:  # extend capacities
                    _tol_nbytes = sum(self._nbytes_list_to_check)
                    # overwrite n_step in current group
                    self._mmp_f.seek(self._current_group_head_position + 16)  # the position of n_cycle
                    _current_steps = int.from_bytes(self._mmp_f.read(8), self._num_fmt)
                    _add_steps = max(1, _current_steps >> 1)  # _current_steps * 1.5
                    _new_steps = _current_steps + _add_steps  # 1.5 times extension
                    self._mmp_f.seek(self._current_group_head_position + 16)
                    self._mmp_f.write(_new_steps.to_bytes(8, self._num_fmt, signed=False))
                    self._mmp_f.seek(self._ptr)
                    # allocate
                    self.allocate(_add_steps * _tol_nbytes)

            # main dump
            #   TODO: adding multiprocess in future
            for i, arr in enumerate(arrays):
                _nb = arr.nbytes
                if _nb == self._nbytes_list_to_check[i]:
                    arr = np.ascontiguousarray(arr)
                    self._mmp_f.write(memoryview(arr))
                    #self._mmp_f[self._ptr: self._ptr + _nb] = memoryview(arr)
                    self._ptr += _nb
                    self._cache_size_now += _nb
                    #self._mmp_f.seek(self._ptr)
                else:
                    raise RuntimeError(
                        f'Inconsistent array bytes between the record in `self.start` and here input. '
                        f'Expected {self._nbytes_list_to_check[i]}, but got {arr.nbytes}.'
                    )

            # dump
            self._count += 1
            if self._cache_size_now >= self.cache_size:
                self._mmp_f.flush()
                self._cache_size_now = 0

        except Exception as e:
            self.close()
            raise RuntimeError(f'Failed to collect `{self.path}`. ERROR: {e}')

    def truncate(self):
        """
        Truncate the blank steps at the end of the file, and exit the current group.
        One must re-start a dumping series to write new data.
        Returns:

        """
        if not self._has_started:
            raise RuntimeError(f'The dumping series was not started. Please call `self.start(...)/start_from_arrays(...)` first.')

        if self._dump_file.closed:
            self._dump_file = open(self.path, 'rb+')
        # truncate
        if self._ptr < self._endptr:
            if self.use_mmap:
                if self._mmp_f.closed:
                    self._mmp_f = mmap.mmap(self._dump_file.fileno(), 0, access=mmap.ACCESS_WRITE)
                self._mmp_f.resize(self._ptr)
            else:
                self._mmp_f = self._dump_file
                self._mmp_f.truncate(self._ptr)
            self._endptr = self._ptr
            # overwrite n_step in current group
            self._mmp_f.seek(self._current_group_head_position + 8)  # the position of n_cycle
            self._mmp_f.write(self._count.to_bytes(8, self._num_fmt, signed=False))
            self._mmp_f.seek(self._ptr)
        self._has_started = False
        self._current_group_head_position = None

    def dump(self):
        """
        Manually do a dumping.
        Returns:

        """
        if (self._mmp_f is None) or self._mmp_f.closed:
            warnings.warn(f"The dumper does not open yet. Please call `self.start(...)/start_from_arrays(...)` first.")
        else:
            self._mmp_f.flush()

    def flush(self):
        """
        Alias for `self.dump`.
        Returns:

        """
        self.dump()

    def _tmp_close(self):
        """
        temporarily close the mmap file
        Returns:

        """
        if (self._dump_file is not None) and (not self._dump_file.closed):
            self._dump_file.close()
        if (self._mmp_f is not None) and (not self._mmp_f.closed):
            self._mmp_f.close()
        self._has_started = False
        self._current_group_head_position = None

    def close(self):
        """
        close the file
        Returns:

        """
        try:
            if (self._mmp_f is not None) and (not self._mmp_f.closed):
                self._mmp_f.flush()
                if self._has_started:
                    self.truncate()
            self._tmp_close()
            #gc.collect()
            self._has_initialized = False
            self._dump_file = None
            self._mmp_f = None
        except Exception as e:
            warnings.warn(f'Failed to close `{self.path}`. ERROR: {e}')

    @property
    def closed(self):
        _q = (self._mmp_f is None) and (self._dump_file is None) and (self._has_initialized is None)
        return _q

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _ArrayDumperPlaceHolder:
    """
    A pure placeholder to compatible with path is None,
    It copies all methods of ArrayDumper but did nothing.
    """
    def __init__(self, path: None, *args, **kwargs) -> None:
        if path is not None:
            raise ValueError(f"This is a placeholder, which only receives path = None, but got {path}.")

    def reset_args(
            self,
            path: str,
            mode: Literal['w', 'x', 'a'] = 'x',
            cache_size: int = 4096,
            head_order: Literal['<', '>'] = '<',
            use_mmap: bool = False,
            *args, **kwargs
    ):
        pass

    def initialize(self):
        pass

    def allocate(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def dump(self):
        pass

    def start(self, *args, **kwargs):
        pass

    def start_from_arrays(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def truncate(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ArrayDumpReader:
    """
    Reading the mmap file dumped by class `ArrayDumper`.
    """
    def __init__(self, path: str):
        """

        Args:
            path: the path to the mmap file.
        """
        self._path = path
        # init vars
        self._ptr = 0
        self._dump_file = None
        self._mmp_f: mmap.mmap|None = None
        # check path
        try:
            with open(self._path, 'rb') as f:
                file_head = f.read(16)
        except FileNotFoundError:
            raise RuntimeError(f'File `{self._path}` was not found.')
        # find the head order
        try_order_le = file_head[:2].decode('utf-16-le')
        try_order_be = file_head[:2].decode('utf-16-be')
        if try_order_le == '<':
            self._str_fmt = 'utf-16-le'
            self._num_fmt: Literal['little', 'big'] = 'little'
        elif try_order_be == '>':
            self._str_fmt = 'utf-16-be'
            self._num_fmt: Literal['little', 'big'] = 'big'
        else:
            raise ValueError(f'Unexpected endianness: {try_order_le}/{try_order_be}.')
        # test BOM
        if len('a'.encode(self._str_fmt)) != 2:
            raise NotImplementedError(
                f'You may check here whether the encode format {self._str_fmt} is correct. '
                f'Normally it would not be added BOM at the start of bytes, thus resulting 2 bytes, but here string "a" '
                f'is encoded into {len('a'.encode(self._str_fmt))} bytes instead.'
            )
        # check magik
        magik = file_head[2:6].decode(self._str_fmt)
        if magik != "BM":
            raise ValueError(f'Unknown file format: {magik}.')
        # check version
        v1 = int.from_bytes(file_head[6:7], self._num_fmt, signed=False)
        v2 = int.from_bytes(file_head[7:8], self._num_fmt, signed=False)
        _version_now = __version__.split('.', 2)
        v1_now = int(_version_now[0])
        v2_now = int(_version_now[1])
        if v1_now >= 255 or v2_now >= 255:
            raise NotImplementedError(f'The version numbers have been reached {_version_now}. How frightful!')
        if (v1, v2) != (v1_now, v2_now):
            warnings.warn(
                f"The file {self._path} (ver \"{'.'.join((str(v1), str(v2)))}\") is incompatible with "
                f"the current version \"{__version__}\".",
                RuntimeWarning,
            )
        # read the group number
        self.n_groups = int.from_bytes(file_head[8:16], self._num_fmt, signed=False)
        # mv ptr
        self._ptr += 16

    def read(
            self,
            groups: List[int]|slice|int = -1,
            indices: List[int]|slice|int = -1,
            is_copy: bool = True
    ) -> Dict[str, List[List[np.ndarray]]]:
        """
        Read the specific arrays from the mmap file by given groups and indices.
        Args:
            groups: the group number to read. A negative number means read all.
                a single int means read this one group of arrays, and a list of ints means read the ones indexed by the list.
                Note: if `group` in the list form, all element must be unique, and it will be SORTED.
            indices: the indices in each group of the arrays to read. A negative number means read all,
            is_copy: whether to copy the arrays from the mmap file.
                Note: if `is_copy` is False, the mmap file cannot be closed due to the exported pointers used by read arrays.
                 One must release all references to the mmap file first to close the memory map file.

        Returns: Dict[str, List[List[np.ndarray]]], {f'group{i}': List[List[np.ndarray]]}, the outer List is steps, inner List is groups.

        """
        if isinstance(groups, int):
            if not groups < self.n_groups: raise ValueError(f'There is only {self.n_groups} groups available, but requested {groups}.')
            groups = range(self.n_groups) if groups < 0 else [groups, ]
        elif isinstance(groups, slice):
            _start, _stop, _step = groups.indices(self.n_groups)
            if not _start >= 0:
                raise ValueError(f'`groups` slice must start from the number greater than or equal to 0, but got {groups.start}.')
            if not _stop <= self.n_groups:
                raise ValueError(f'There is only {self.n_groups} groups available, but requested {groups.stop}.')
            groups = range(_start, _stop, _step)
        elif isinstance(groups, list):
            if not min(groups) >= 0:
                raise ValueError(f'elements in the `groups` must be greater than or equal to 0, but got {min(groups)}.')
            if not max(groups) < self.n_groups:
                raise ValueError(f'There is only {self.n_groups} groups available, but requested {max(groups)}.')
            if len(groups) != len(set(groups)):
                raise ValueError(f'`groups` must have unique elements, but some duplicate elements are found.')
            groups = sorted(groups)
        else:
            raise TypeError(f'`groups` must be a list or int, but got {type(groups)}.')

        _grp_ptr = 0
        _n_select_groups = len(groups)
        output_arrs = dict()
        for i_grp in range(self.n_groups):
            if _grp_ptr >= _n_select_groups:
                break
            if i_grp == groups[_grp_ptr]:
                output_arrs[f'group{i_grp}'] = self._read_once(indices, is_copy)
                _grp_ptr += 1
            else:
                self._skip_once()

        if is_copy:  # Note: if not copy, exported pointers will exist that the read arrays use them. Hence it cannot be closed.
            self.close()

        return output_arrs

    def _skip_once(self):
        """
        Skip this group data once, instead of reading it.

        Returns:

        """
        try:
            n_cycles, n_arrays, dtype_list, shape_list, stride_list = self._parse_arr_head()
            _block_total_stride = sum(stride_list)  # total stride (bytes) of this group.
            self._ptr += n_cycles * _block_total_stride  # move the ptr to the end of the group
            self._mmp_f.seek(self._ptr)

        except Exception as e:
            self._tmp_close()
            raise RuntimeError(f'An error occurred while reading file {self._path}. ERROR: {e}.')

    def _read_once(self, indices: List[int]|slice|int = -1, is_copy: bool = True) -> List[List[np.ndarray]]:
        """
        Read one time from the beginning of the current group.
        self._ptr must be at the start of the group, and then will be moved to the end of the group.
        Args:
            indices: the indices in each group of the arrays to read. A negative number means read all.
            is_copy: whether to copy the array from mmap file or not.

        Returns: List[np.ndarray], list of arrays.

        """
        try:
            n_cycles, n_arrays, dtype_list, shape_list, stride_list = self._parse_arr_head()
            output_list_arrays = list()
            if isinstance(indices, list):
                if not min(indices) >= 0:
                    raise ValueError(f'elements in the `indices` must be greater than or equal to 0, but got {min(indices)}.')
                if not max(indices) < n_cycles:
                    raise ValueError(f"indices {max(indices)} is out of range.")
                cycle_list = indices
            elif isinstance(indices, slice):
                _start, _stop, _step = indices.indices(n_cycles)
                if not ((_start >= 0) and (_stop <= n_cycles)):
                    raise ValueError(f"indices {indices} is out of range: [{indices.start}, {indices.stop}].")
                cycle_list = range(_start, _stop, _step)
            elif isinstance(indices, int):
                if not indices < n_cycles: raise ValueError(f"indices {indices} is out of range.")
                cycle_list = range(n_cycles) if indices < 0 else [indices, ]
            else:
                raise TypeError(f'Expected indices of List[int], slice, or int, but got {type(indices)}.')

            _block_total_stride = sum(stride_list)  # total stride (bytes) of this group.
            for i_cyc in cycle_list:
                # calculate the ptr position
                _outer_ptr = self._ptr + i_cyc*_block_total_stride
                self._mmp_f.seek(_outer_ptr)
                if is_copy:
                    _raw_data = self._mmp_f.read(_block_total_stride)
                    # Note: Here can be implemented as multiprocessing.
                    _arr_now = list()
                    inner_ptr = 0
                    for _ist, _stride in enumerate(stride_list):
                        _arr = np.frombuffer(_raw_data[inner_ptr:inner_ptr + _stride], dtype=dtype_list[_ist]).reshape(shape_list[_ist])
                        _arr_now.append(_arr)
                        inner_ptr += _stride
                    output_list_arrays.append(_arr_now)
                else:
                    _raw_data = memoryview(self._mmp_f)[_outer_ptr:_outer_ptr + _block_total_stride]
                    # Note: Here can be implemented as multiprocessing, too.
                    _arr_now = list()
                    inner_ptr = 0
                    for _i, _stride in enumerate(stride_list):
                        _arr = np.frombuffer(_raw_data[inner_ptr:inner_ptr + _stride], dtype=dtype_list[_i]).reshape(shape_list[_i])
                        _arr_now.append(_arr)
                        inner_ptr += _stride
                    output_list_arrays.append(_arr_now)
                    self._mmp_f.seek(_outer_ptr + _block_total_stride)

            # move the ptr to the end of the group
            self._ptr += n_cycles * _block_total_stride
            self._mmp_f.seek(self._ptr)
            #if self._ptr != self._mmp_f.tell():
            #    raise RuntimeError(f'???? BUG: WHY DOES THE self._ptr MISMATCH ???? Expected {self._mmp_f.tell()} but got {self._ptr}.')

            return output_list_arrays

        except Exception as e:
            self._tmp_close()
            raise RuntimeError(f'An error occurred while reading file {self._path}. ERROR: {e}.')

    def _parse_arr_head(self, ) -> Tuple[int, int, List[str], List[Tuple], List[int]]:
        """
        Parse the array head information once.
        self._ptr must be at the end of a group.
        Then after calling this method, self._ptr will be moved to the start of the group data.

        Returns: n_cycles, n_arrays, dtype_list, shape_list, stride_list
        """
        r"""
        Head information:
            `char``n_cycle``n_array``dtype1``shape1[]`0`dtype2``shape2[]`0...`dtype_n``shape_n[]`0`byte_data`...
            wherein each term occupies 8 bytes.
        """
        try:
            if (self._dump_file is None) or self._dump_file.closed:
                self._dump_file = open(self._path, 'rb')
            if (self._mmp_f is None) or self._mmp_f.closed:
                self._mmp_f = mmap.mmap(self._dump_file.fileno(), 0, access=mmap.ACCESS_READ)

            self._mmp_f.seek(self._ptr)
            is_head = self._mmp_f.read(8).decode(self._str_fmt)
            if is_head != 'HEAD':
                raise RuntimeError(f'Could not find head byte in mmap file: {self._path}. This file may be corrupted.')
            n_cycles = int.from_bytes(self._mmp_f.read(8), self._num_fmt, signed=False)
            n_arrays = int.from_bytes(self._mmp_f.read(8), self._num_fmt, signed=False)

            dtype_list = list()
            shape_list = list()
            stride_list = list()
            for _ in range(n_arrays):
                _dtp_ot = self._mmp_f.read(4).decode(self._str_fmt)  # order and type without length
                _dtp_len = int.from_bytes(self._mmp_f.read(4), self._num_fmt, signed=False)  # the byte length of each elem in arr.
                _dtype = f'{_dtp_ot}{_dtp_len}'
                # Special for Unicode char in numpy which applied utf-32 of 4 bytes.
                if _dtp_ot[1] == 'U':
                    elem_size = _dtp_len * 4
                else:
                    elem_size = _dtp_len
                dtype_list.append(_dtype)
                _shape = list()
                while True:
                    _sp_num = int.from_bytes(self._mmp_f.read(8), self._num_fmt, signed=False)
                    if _sp_num == 0:  # reached the delimiter
                        break
                    _shape.append(_sp_num)
                shape_list.append(tuple(_shape))
                stride_list.append(elem_size * math.prod(_shape))
            self._ptr = self._mmp_f.tell()

            return n_cycles, n_arrays, dtype_list, shape_list, stride_list

        except Exception as e:
            self._tmp_close()
            raise RuntimeError(f'An error occurred while reading mmap file: {self._path}: {e}')

    def _tmp_close(self):
        """
        temporarily close the mmap file
        Returns:

        """
        if (self._dump_file is not None) and (not self._dump_file.closed):
            self._dump_file.close()
        if (self._mmp_f is not None) and (not self._mmp_f.closed):
            self._mmp_f.close()

    def _raw_close(self):
        """
        directly close the memmap and gc
        Returns:

        """
        self._tmp_close()
        gc.collect()
        self._dump_file = None
        self._mmp_f = None

    def close(self):
        """
        Close the memmap and gc with checks.
        Returns:

        """
        try:
            self._raw_close()
        except BufferError as bufe:
            warnings.warn(
                f'Failed to close mmap file {self._path}: {bufe}.\n'
                f'YOU MAY RELEASE EVERY REFERENCE FROM THE MEMORY MAPPING FILE, '
                f'AND THEN CLOSE IT MANUALLY AGAIN. '
                f'OTHERWISE, THIS FILE WILL KEEP OPENING.\n'
                f'I HOPE YOU KNOW WHAT YOU ARE DOING!!!'
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._raw_close()
        except BufferError as bufe:
            warnings.warn(
                f'Failed to close mmap file {self._path}: {bufe}.\n'
                f'YOU MAY RELEASE EVERY REFERENCE FROM THE MEMORY MAPPING FILE, '
                f'AND THEN CLOSE IT MANUALLY AGAIN. '
                f'OTHERWISE, THIS FILE WILL KEEP OPENING.\n'
                f'I HOPE YOU KNOW WHAT YOU ARE DOING!!!'
            )


class StandardModel(ABC):
    """
    An abstract base class that convert arbitrary function into BUCToolkit supported format to structure opt, MD, and MC, etc.
    It can receive any user-defined Callable object `model` which returns torch.Tensors, and convert it to the standard dict format
        {'energy': energy, 'forces': forces} by user-override `initialize_model` (optional), `calc_energy`, and `calc_forces` with
        input `args` and `kwargs`.
    DO NOT SUPPORT FOR TRAINING.

    """
    def __init__(
            self,
            model,
            args_for_init: Tuple = tuple(),
            kwargs_for_init: Dict | None = None,
            device: str = 'cpu',
            *args,
            **kwargs
    ):
        """
        The original PyTorch model with (hyper-)parameters args and kwargs.
        Args:
            model: the main function
            args_for_init: arguments passed to model
            kwargs_for_init: keyword arguments passed to model
            args: additional positional arguments for calc_energy or calc_forces, if necessary
            kwargs: additional keyword arguments for calc_energy or calc_forces, if necessary
        """
        if kwargs_for_init is None: kwargs_for_init = {}
        super().__init__()
        self.model = model
        self.args = args
        self.kwargs = kwargs
        self.device = th.device(device)

        self.initialize_model(*args_for_init, **kwargs_for_init)

    def initialize_model(self, *args, **kwargs):
        """
        Initialize & config the model, if necessary.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def calc_energy(self, x) -> th.Tensor:
        """
        The method that calculates the energy of given x.
        One may use input args/kwargs in __init__
        Args:
            x: input variable

        Returns: energy

        """
        pass

    @abstractmethod
    def calc_forces(self, x) -> th.Tensor:
        """
        The method that calculates the forces of given x.
        One may use input args/kwargs in __init__
        Args:
            x: input variable

        Returns: forces

        """
        pass

    def __call__(self, x):
        with th.no_grad():
            ener = th.as_tensor(self.calc_energy(x), device=self.device)
            forc = th.as_tensor(self.calc_forces(x), device=self.device)
            return {'energy': ener, 'forces': forc}


class StandardInput(ABC):
    """
    An abstract container as the standard input of advanced API `api`,
     which have properties that can be inquired by follow methods as the input arg `data`:

        def get_batch_size(data):
            # total batch size
            return len(data)

        def get_cell_vec(data):
            # `batch_size` cell vectors stacked at the 1st dim.
            return data.cell.numpy(force=True)

        def get_atomic_number(data):
            # `batch_size` atomic numbers tensor(int64) concatenated at the 1st dim.
            return data.atomic_numbers.unsqueeze(0)

        def get_indx(data):
            # Python list object with `batch_size` elements of each sample's name.
            # each name must have less than 128 bytes.
            _indx: Dict = getattr(data, 'idx', None)
            return _indx

        def get_pos(data):
            # Atomic coordinates tensor in dtype float32 concatenated at the 1st dim.
            return data.pos.unsqueeze(0)

        def get_fixed_mask(data):
            # the same shape as `pos` with dtype int8. Exactly, only 0, 1.
            # Optional. default is all ones.
            mask = getattr(data, 'fixed', None)
            if mask is not None:
                mask = mask.unsqueeze(0)
            return mask

        def get_batch_indx(data):
            # batch_indices in the form of 1d int64 th.Tensor [0, 0, 0, ..., 1, 1, ..., n]
            # where the same number means the indexed data of the same sample.
            return [len(dat.pos) for dat in data.to_data_list()] # or can directly store this attr and return `data.batch`

        def get_init_dX(data):
            # used for Dimer & other algo. requiring finite difference. Have the same shape & dtype as `pos`
            # Optional.
            return getattr(data, self.x_diff_attr, None)

        def get_init_veloc(data):
            # used for MD. The initial velocities that have the same shape & dtype as `pos`
            # Optional.
            veloc = getattr(data, 'velocity', None)
            if veloc is not None:
                veloc = veloc.unsqueeze(0)
            return veloc

    They are fully compatible with `torch_geometric.data.batch` object.
    """
    _ATTR_TYPE = {
        'idx': '<U128',
        'batch': '<i8',
        'cell': '<f4',
        'pos': '<f4',
        'mask': '|i1',
        'atomic_numbers': '<i4',
        'x_diff': '<f4',
        'velocity': '<f4',
    }

    # the following properties are must have implemented.

    @abstractmethod
    def __len__(self) -> int:
        """ return the sample number in one batch """
        pass

    @property
    @abstractmethod
    def batch(self) -> Optional[th.Tensor]:
        """
        batch indices tensor in the form of 1-D int64 th.Tensor [0, 0, 0, ..., 1, 1, ..., n]
            where the same number means the indexed data of the same sample.
        """
        pass

    @batch.setter
    @abstractmethod
    def batch(self, value: Optional[th.Tensor]) -> None:
        pass

    @property
    @abstractmethod
    def pos(self) -> th.Tensor:
        """
        Atomic coordinates tensor in dtype float32 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, 3).
        """
        pass

    @pos.setter
    @abstractmethod
    def pos(self, value: th.Tensor) -> None:
        pass

    @property
    @abstractmethod
    def cell(self) -> th.Tensor:
        """
        cell tensor in dtype float32 concatenated at the 1st dim. shape: (n_batch, 3, 3)
        """
        pass

    @cell.setter
    @abstractmethod
    def cell(self, value: th.Tensor) -> None:
        pass

    @property
    @abstractmethod
    def atomic_numbers(self) -> th.Tensor:
        """
        Atomic number tensor in dtype int64 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, ).
        """
        pass

    @atomic_numbers.setter
    @abstractmethod
    def atomic_numbers(self, value: th.Tensor) -> None:
        pass

    @property
    def idx(self) -> Optional[List[str]]:
        """
        Optional. The sample name list. Each name must have less than 128 bytes.
        """
        return None

    @idx.setter
    def idx(self, value: Optional[List[str]]) -> None:
        pass

    @property
    def fixed(self) -> Optional[th.Tensor]:
        """
        Optional. The atomic fixation tensor in dtype int8 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, 3).
        element can only be 0 (fixed) or 1 (free).
        """
        return None

    @fixed.setter
    def fixed(self, value: Optional[th.Tensor]) -> None:
        pass

    @property
    def velocity(self) -> Optional[th.Tensor]:
        """
        Optional. The atom velocity tensor in dtype float32 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, 3).
        Only used for MD.
        """
        return None

    @velocity.setter
    def velocity(self, value: Optional[th.Tensor]) -> None:
        pass

    @property
    def x_diff(self) -> Optional[th.Tensor]:
        """
        Optional. The atom difference tensor in dtype float32 concatenated at the 1st dim. shape: (sum_i^{n_batch} n_i, 3).
        Only used for Dimer or other algorithms which require finite differences.
        """
        return None

    @x_diff.setter
    def x_diff(self, value: Optional[th.Tensor]) -> None:
        pass

    @abstractmethod
    def to_data_list(self) -> List[Any]:
        """
        Split batched data into the List of each sample.
        Similar to torch_geometric.data.Batch.
        Optional to override.
        """
        raise NotImplementedError(f'Please implement `self.to_data_list()` by overriding in the subclass.')

def structures_io_dumper(path: str|None, mode: Literal['w', 'x', 'a'] = 'x', disable: bool = False):
    """
    Auxiliary function for structure IO. It will be added into Batch* methods as a general dumper.
    if `path` is None, or `disable` is True,
     a placeholder which contains all methods of ArrayDumper but does nothing when called will be assigned.
    """
    if (not disable) and (path is not None):
        dumper = ArrayDumper(path, mode=mode, cache_size=4096, use_mmap=False)
    else:
        dumper = _ArrayDumperPlaceHolder(path)

    return dumper

def read_md_traj(
        path,
        indices: List[int]|slice|int = -1,
        is_copy: bool = True
):
    """
    A specialized reader for dump files generated by MD.
    For BaseMD class, the information is as follows
    with denoting shape [n_batch, n_atom, n_dim] (regular batch) or [1, sumNi, n_atom] (irregular batch) as "sX":
        group 1: 1-step
            batch_indices_tensor[n_batch, ] (optional, exists for irregular batches)
            cell_vec[n_batch, 3, 3] / [1], if No cell_vec input, it will be set to [0] (shape: [1, ]).
            atomic_numbers[n]
            fixed_mask[sX]
        group 2: n_time_steps step
            E[n_batch, ] (energy)
            X[sX]        (coordinates)
            V[sX]        (velocities)
            F[sX]        (forces)
    Args:
        path: the path to the dump file.
        indices: the indices in each group of the arrays to read. A negative number means read all.
        is_copy: whether to copy the arrays from the mmap file.
            Note: if `is_copy` is False, the mmap file cannot be closed due to the exported pointers used by read arrays.
             One must release all references to the mmap file first to close the memory map file.
    Returns:
        BatchStructures

    """
    reader = ArrayDumpReader(path)
    raw_results = reader.read(groups=-1, indices=indices, is_copy=is_copy)
    n_grp = len(raw_results)
    if n_grp % 2 != 0:
        raise EOFError(f"Molecular dynamics file must contain even number of groups, but got {n_grp}.")

    smp_ids = list()
    cell_list = list()
    element_list = list()
    numbers_list = list()
    coo_t_list = list()
    coo_list = list()
    fixed_list = list()
    energy_list = list()
    force_list = list()

    for i in range(0, n_grp, 2):
        i_head = i
        i_content = i + 1
        i_cyc = i_head // 2
        if len(raw_results[f'group{i_head}'][0]) == 3:  # no irregular batch_indices
            batch_indices_tensor = None
            (
                cell_vec,
                atomic_numbers,
                fixed_mask
            ) = raw_results[f'group{i_head}'][0]
            n_batch = len(cell_vec)
            _elements = list()
            _numbers = list()
            _id_per_frame = list()
            for ii, _atml in enumerate(atomic_numbers):
                elements, _, numbers = elem_list_reduce(_atml)
                _elements.append(elements)
                _numbers.append(numbers)
                _id_per_frame.append(ii)
            _fixed = [_ for _ in fixed_mask]
            _cells = [_ for _ in cell_vec]
            kk = 0
            n_cyc = len(raw_results[f'group{i_content}'])
            cell_list.extend(_cells * n_cyc)
            element_list.extend(_elements * n_cyc)
            numbers_list.extend(_numbers * n_cyc)
            coo_t_list.extend(['C'] * (n_batch * n_cyc))
            fixed_list.extend(_fixed * n_cyc)
            for en, x, v, f in raw_results[f'group{i_content}']:
                _x = [_ for _ in x]
                _f = [_ for _ in f]
                smp_ids.extend([f'samp{i_cyc}_struc{_}_step{kk}' for _ in _id_per_frame])
                coo_list.extend(_x)
                energy_list.extend(en.tolist())
                force_list.extend(_f)
                kk += 1

        elif len(raw_results[f'group{i_head}'][0]) == 4:  # irregular situation
            (
                batch_indices_tensor,
                cell_vec,
                atomic_numbers,
                fixed_mask
            ) = raw_results[f'group{i_head}'][0]
            n_batch = len(batch_indices_tensor)
            _split_indices = np.cumsum(batch_indices_tensor)[:-1]
            _cells = [_ for _ in cell_vec]
            _tol_atm_list = np.split(atomic_numbers[0], _split_indices, axis=0)
            _elements = list()
            _numbers = list()
            _id_per_frame = list()
            for ii, _atml in enumerate(_tol_atm_list):
                elements, _, numbers = elem_list_reduce(_atml)
                _elements.append(elements)
                _numbers.append(numbers)
                _id_per_frame.append(ii)
            _fixed = np.split(fixed_mask[0], _split_indices, axis=0)
            # main data
            kk = 0
            n_cyc = len(raw_results[f'group{i_content}'])
            cell_list.extend(_cells * n_cyc)
            element_list.extend(_elements * n_cyc)
            numbers_list.extend(_numbers  * n_cyc)
            coo_t_list.extend(['C'] * (n_batch * n_cyc))
            fixed_list.extend(_fixed * n_cyc)
            for en, x, v, f in raw_results[f'group{i_content}']:
                _x = np.split(x[0], _split_indices, axis=0)
                _f = np.split(f[0], _split_indices, axis=0)
                smp_ids.extend([f'samp{i_cyc}_struc{_}_step{kk}' for _ in _id_per_frame])
                coo_list.extend(_x)
                energy_list.extend(en.tolist())
                force_list.extend(_f)
                kk += 1
        else:
            raise ValueError(f"Invalid file format: {path}. It may be not a MD dump file.")

    bs = BatchStructures()
    bs.append_from_lists(
        smp_ids,
        cell_list,
        element_list,
        numbers_list,
        coo_t_list,
        coo_list,
        fixed_list,
        energy_list,
        force_list,
    )
    bs._check_id()
    bs._check_len()

    return bs

def read_opt_structures(
        path,
        indices: List[int]|slice|int = -1,
        is_copy: bool = True
):
    """
    A specialized reader for dump files generated by Structure Optimization.
    For `StructureOptimization` class, the information is as follows
    with denoting shape [1, sumNi, n_atom] (irregular batch) as "sX":
        group 1: 1-step
            batch_indices[n_batch, ]
            idx[n_batch, ], dtype='<U128', the name of structures.
            cells[n_batch, 3, 3]
            elements[sumNi]
            pos[sX]
            fixations[sX]
            energies[sumNi]
            forces[sX]
    The coordinates type is forever 'Cartesian'.

    Args:
        path: the path to the dump file.
        indices: the indices in each group of the arrays to read. A negative number means read all.
        is_copy: whether to copy the arrays from the mmap file.
            Note: if `is_copy` is False, the mmap file cannot be closed due to the exported pointers used by read arrays.
             One must release all references to the mmap file first to close the memory map file.
    Returns:
        BatchStructures

    """
    reader = ArrayDumpReader(path)
    raw_results = reader.read(groups=-1, indices=indices, is_copy=is_copy)
    n_grp = len(raw_results)

    smp_ids = list()
    cell_list = list()
    element_list = list()
    numbers_list = list()
    coo_t_list = list()
    coo_list = list()
    fixed_list = list()
    energy_list = list()
    force_list = list()

    for i in range(n_grp):
        if len(raw_results[f'group{i}'][0]) != 8:  # irregular situation
            raise ValueError(f"Invalid file format: {path}. It may be not a Structure Optimization dumped file.")
        (
            batch_indices,
            idx,
            cells,
            elements,
            pos,
            fixations,
            energies,
            forces
        ) = raw_results[f'group{i}'][0]

        n_batch = len(batch_indices)
        _split_indices = np.cumsum(batch_indices)[:-1]
        _cells = [_ for _ in cells]
        _tol_atm_list = np.split(elements, _split_indices, axis=0)
        _elements = list()
        _numbers = list()
        _id_per_frame = list()
        for ii, _atml in enumerate(_tol_atm_list):
            elements, _, numbers = elem_list_reduce(_atml)
            _elements.append(elements)
            _numbers.append(numbers)
            _id_per_frame.append(ii)
        _fixed = np.split(fixations, _split_indices, axis=0)
        # main data
        kk = 0
        n_cyc = len(raw_results[f'group{i}'])
        if n_cyc != 1:
            raise RuntimeError(f'??? BUG: why is not the cycle number of structure optimization 1, but {n_cyc} cycles? Report us please! ???')
        cell_list.extend(_cells)
        element_list.extend(_elements)
        numbers_list.extend(_numbers)
        coo_t_list.extend(['C'] * n_batch)
        fixed_list.extend(_fixed)
        _x = np.split(pos, _split_indices, axis=0)
        _f = np.split(forces, _split_indices, axis=0)
        smp_ids.extend(idx.tolist())
        coo_list.extend(_x)
        energy_list.extend(energies.tolist())
        force_list.extend(_f)

    bs = BatchStructures()
    bs.append_from_lists(
        smp_ids,
        cell_list,
        element_list,
        numbers_list,
        coo_t_list,
        coo_list,
        fixed_list,
        energy_list,
        force_list,
    )
    bs._check_id()
    bs._check_len()

    return bs



if __name__ == '__main__' and False:
    import time
    import cProfile

    STEPS = 100
    #arrdmp1 = ArrayDumper('/tmp/arrdmp1', cache_size=4096, use_mmap=True)
    #arrdmp2 = ArrayDumper('/tmp/arrdmp2', cache_size=4096, use_mmap=False)
    #arrdmp.start(
    #    10000,
    #    ['<f4', '<i1', '<f4', '<f4', '<f8'],
    #    [(1, 500, 3), (1, 500, 3), (30, ), (30, ), (1, 8)]
    #)
    a = np.random.random_sample((1, 500, 3)).astype(np.float32)
    b = np.random.random((1, 500, 3)).astype(np.int8)
    c = np.random.random((30,)).astype(np.float32)
    d = np.random.random((30,)).astype(np.float32)
    e = np.random.random((1, 8,))
    #arrdmp1.start_from_arrays(STEPS, a, b, c, d, e)
    #arrdmp2.start_from_arrays(STEPS, a, b, c, d, e)

    def mtest1(writer):
        dumptime = 0.
        res_list = list()
        for _ in range(STEPS):
            a = np.random.random_sample((1, 500, 3)).astype(np.float32)
            b = np.random.random((1, 500, 3)).astype(np.int8)
            c = np.random.random((30, )).astype(np.float32)
            d = np.random.random((30, )).astype(np.float32)
            e = np.random.random((1, 8, ))
            tt1 = time.perf_counter()
            writer.step(a, b, c, d, e)
            dumptime += time.perf_counter() - tt1
            res_list.append([a, b, c, d, e])
        writer.close()
        print(f'Dumping took {dumptime:.4f} seconds')
        return res_list


    def mtest2():
        dumptime = 0.
        res_list = dict()
        for _ in range(STEPS):
            writer = ArrayDumper('/tmp/arrdmp_append', mode='a', cache_size=4096, use_mmap=False)
            tt1 = time.perf_counter()
            for __ in range(500):
                a = np.random.random_sample((1, 500, 3)).astype(np.float32)
                b = np.random.random((1, 500, 3)).astype(np.int8)
                c = np.random.random((30,)).astype(np.float32)
                d = np.random.random((30,)).astype(np.float32)
                e = np.random.random((1, 8,))
                if __ == 0:
                    writer.start_from_arrays(-1, a, b, c, d, e)
                    res_list[f'group{_}'] = list()
                writer.step(a, b, c, d, e)
                res_list[f'group{_}'].append([a, b, c, d, e])
            writer.close()
            dumptime += time.perf_counter() - tt1

        print(f'Dumping took {dumptime:.4f} seconds')
        return res_list

    #cProfile.run('mtest1(arrdmp1)')
    #print('\n'*2 + '*'*80 + '\n'*2)
    #cProfile.run('mtest1(arrdmp2)')

    res_list1 = mtest2()
    #res_list2 = mtest1(arrdmp2)

    def validate(path, res_list):
        with ArrayDumpReader(path, ) as arr_reader:
            tt1 = time.perf_counter()
            result_dict = arr_reader.read(is_copy=True)
            print(f'Reading took {time.perf_counter() - tt1:.4f} seconds')

        for _, tmparr in enumerate(result_dict["group0"][0:100]):
            for __, _tmparr in enumerate(tmparr):
                if not np.all(_tmparr == res_list["group0"][_][__]):
                    print(f'ERROR: the ({_}, {__})-th data mismatched.')

        for _, tmparr in enumerate(result_dict["group0"][-100:], STEPS - 100):
            for __, _tmparr in enumerate(tmparr):
                if not np.all(_tmparr == res_list["group0"][_][__]):
                    print(f'ERROR: the ({_}, {__})-th data mismatched.')

    validate('/tmp/arrdmp_append', res_list1)
    #validate('/tmp/arrdmp2', res_list2)

if __name__ == '__main__' and True:
    test_bs = list()
    for ijk in [1, 5, 10, 15, 20, 25]:
        test_bs.append(read_md_traj(f'/home/ppx/PythonProjects/test_files/ASE_vs_BM/MD/MD_dump_bt{ijk}', slice(0, 500), ))
    pass
