"""
Contiguous dumping and reading arrays data by memory mapping.

"""
#  Copyright (c) 2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: MemMapTensorsIO.py
#  Environment: Python 3.12
from typing import Dict, List, Literal, Tuple, ByteString
from _io import BufferedRandom
import warnings, os, mmap, copy, gc, math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Any, Literal, Sequence, List

import numpy as np
import torch as th
from torch import nn

from BUCToolkit._version import __db_version__
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

    _MAX_NAME_LENGTH: int = 256  # max UTF-16 code units per array name

    def _encode_names_section(self, names: List[str]) -> bytes:
        """
        Encode array names into the names section bytes.

        Format:
            n_names   (8 bytes, uint64) — number of names
            For each name:
                name_len    (8 bytes, uint64) — number of UTF-16 code units
                name_data   (name_len × 2 bytes) — UTF-16 encoded string
                padding     (0–6 bytes of 0x00) — align to 64-bit boundary

        Args:
            names: list of name strings, one per array.

        Returns: bytes of the complete names section.

        Raises:
            ValueError: if any name exceeds ``_MAX_NAME_LENGTH`` UTF-16 code units.
        """
        parts = [len(names).to_bytes(8, self._num_fmt, signed=False)]
        for name in names:
            name = str(name)  # force type conversion
            encoded = name.encode(self._str_fmt)
            n_units = len(encoded) // 2  # number of UTF-16 code units
            if n_units > self._MAX_NAME_LENGTH:
                raise ValueError(
                    f'Array name "{name[:40]}..." ({n_units} UTF-16 code units) '
                    f'exceeds the maximum length of {self._MAX_NAME_LENGTH}.'
                )
            parts.append(n_units.to_bytes(8, self._num_fmt, signed=False))
            parts.append(encoded)
            # 64-bit alignment padding
            remainder = len(encoded) % 8
            if remainder:
                parts.append(b'\x00' * (8 - remainder))
        return b''.join(parts)

    def _prepare_names_section(
            self,
            names: Sequence[str] | None,
            n_arrays: int,
    ) -> bytes:
        """Validate and encode the array names of one DB 2.0 group.

        Canonical DB 2.0 groups are self-describing: every stored array has
        one non-empty, unique name.  This method performs the group-level
        validation shared by :meth:`start` and :meth:`start_from_arrays`, then
        delegates the binary UTF-16 encoding to :meth:`_encode_names_section`.

        Args:
            names: Array names in exactly the same order as the arrays passed
                to the group-start method. ``None`` is not valid in DB 2.0.
            n_arrays: Number of arrays contained in each cycle of the group.

        Returns:
            Encoded names-section bytes, including the number of names, each
            UTF-16 name and its 64-bit alignment padding.
        """
        if names is None:
            raise ValueError(
                'DB 2.0 requires one unique name for every dumped array.'
            )
        normalized_names = [str(name) for name in names]
        if len(normalized_names) != n_arrays:
            raise ValueError(
                f'Number of names ({len(normalized_names)}) must match number '
                f'of arrays ({n_arrays}).'
            )
        if any(not name for name in normalized_names):
            raise ValueError('Array names must be non-empty strings.')
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError(f'Array names must be unique, got {normalized_names}.')
        return self._encode_names_section(normalized_names)

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
                `char``n_cycle``n_array``n_names``name1` ... `name_n``dtype1``shape1[]`0`dtype2``shape2[]`0...`dtype_n``shape_n[]`0`byte_data`...
                 HEAD  uint8    uint8    uint8    [uint8, utf16, padding] ...
                  where `char` is the character number indicate the array head information, which is hard coded as "HEAD",
                  `n_names` equals `n_array` in canonical DB 2.0. Each name is stored as:
                    name_len   (8 bytes, uint64): number of UTF-16 code units,
                    name_data  (name_len × 2 bytes): UTF-16 encoded string,
                    padding    (0–6 bytes): zero-padding to 64-bit boundary.
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
                _v = __db_version__.split('.', 2)
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
            _version_now = __db_version__.split('.', 2)
            v1_now = int(_version_now[0])
            v2_now = int(_version_now[1])
            if v1_now >= 255 or v2_now >= 255:
                raise NotImplementedError(f'The version numbers have been reached {_version_now}. How frightful!')
            if (v1, v2) != (v1_now, v2_now):
                raise ValueError(
                    f'Database version {v1}.{v2} cannot be appended by the '
                    f'canonical {__db_version__} dumper. Convert the file first.'
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

    def start_from_arrays(self, steps: int, *arrays: np.ndarray,
                          force: bool = False, names: Optional[List[str]] = None):
        """
        Write the head information of array from a prototype arrays, allocate the blank space, thus starting a new dumping series.
        Note that arrays input here are only a prototype that will NOT be written to the disk.
        One must call `self.step` method to really write arrays.
        Args:
            steps: the iteration number of arrays, which will be dumped. If `steps` = -1, a dynamic steps will be used.
            *arrays: sequence of arrays as the prototype.
            force: whether to force starting a new dumping series even if there are still blanks at the end of the mmap file.
            names: unique human-readable names for each array. DB 2.0 requires
                exactly one non-empty name per array.

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
            names_section_bytes = self._prepare_names_section(names, self._n_arrays)
            _head_nbytes = 24 + len(names_section_bytes)  # `char``n_cycle``n_array``names_section`, 3*8 + names
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
            # names section: n_names field + name entries
            dump_content[24:24 + len(names_section_bytes)] = names_section_bytes
            _ptr = 24 + len(names_section_bytes)
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
            force: bool = False,
            names: Optional[List[str]] = None,
    ):
        """
        Write the head information of array, allocate the blank space, thus starting a new dumping series.
        Args:
            steps: the iteration number of arrays, which will be dumped. If `steps` = -1, a dynamic steps will be used.
            dtype_list: list of the arrays' dtypes.
            shape_list: list of the arrays' shapes.
            force: whether to force starting a new dumping series even if there are still blanks at the end of the mmap file.
            names: unique human-readable names for each array. DB 2.0 requires
                exactly one non-empty name per array.

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
            names_section_bytes = self._prepare_names_section(names, self._n_arrays)
            _head_nbytes = 24 + len(names_section_bytes)  # `char``n_cycle``n_array``names_section`, 3*8 + names
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
            # names section: n_names field + name entries
            dump_content[24:24 + len(names_section_bytes)] = names_section_bytes
            _ptr = 24 + len(names_section_bytes)
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

    def start(self, *args, names=None, **kwargs):
        pass

    def start_from_arrays(self, *args, force=False, names=None, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def truncate(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ArrayDumpReaderOld:
    """Read legacy DB 1.0 array dumps.

    DB 1.0 originally stored only array dtype and shape information.  Some
    beta files also contain the later names section while retaining version
    1.0.  This reader accepts both layouts so retained files can be inspected
    through ``read_*_old`` or converted to canonical DB 2.0.

    The reader exposes raw groups and does not assign semantic names to an
    unnamed legacy group.  Specialized legacy trajectory readers provide the
    historical positional mapping for MD, MC, and optimizer files.

    Args:
        path: Path to an existing DB 1.0 binary dump.

    Attributes:
        db_version: Two-integer database version parsed from the file header.
        n_groups: Number of array groups recorded in the file header.
    """

    _MAX_NAMES: int = 256       # sanity cap for n_names discriminator
    _MAX_NAME_LENGTH: int = 256  # max UTF-16 code units per array name
    _SUPPORTED_DB_VERSION: Tuple[int, int] = (1, 0)
    _REQUIRE_NAMES: bool = False

    def __init__(self, path: str):
        """Open and validate the fixed DB 1.0 file header.

        Args:
            path: Path to the legacy binary dump. The data region is mapped
                lazily when :meth:`read` is called.
        """
        self._path = path
        # init vars
        self._ptr = 0
        self._dump_file = None
        self._mmp_f: mmap.mmap|None = None
        self._group_names: Dict[int, List[str]] = {}  # group_idx -> list of array names
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
        # DB 1.0 is intentionally isolated behind the explicit legacy reader.
        v1 = int.from_bytes(file_head[6:7], self._num_fmt, signed=False)
        v2 = int.from_bytes(file_head[7:8], self._num_fmt, signed=False)
        self.db_version = (v1, v2)
        if self.db_version != self._SUPPORTED_DB_VERSION:
            expected = '.'.join(str(value) for value in self._SUPPORTED_DB_VERSION)
            raise ValueError(
                f'Database version {v1}.{v2} is not supported by '
                f'{type(self).__name__}; expected {expected}.'
            )
        # read the group number
        self.n_groups = int.from_bytes(file_head[8:16], self._num_fmt, signed=False)
        # mv ptr
        self._ptr += 16

    @property
    def names(self) -> Optional[Dict[int, List[str]]]:
        """
        Return the array names read from the last ``read()`` call, keyed by
        group index.  Returns ``None`` if ``read()`` has not been called yet.

        Each value is a list of strings, one per array in that group, in the
        same order as the arrays returned by ``read()``.
        """
        if not self._group_names:
            return None
        return dict(self._group_names)

    @staticmethod
    def _resolve_names(names: List[str], names_list: List[str]) -> List[int]:
        """Convert requested array names to column indices.

        This is the bridge between the human-facing ``names`` parameter
        and the internal column-index machinery.  Because the user may
        request names in any order (e.g. ``["forces", "energy"]``), the
        result preserves that order rather than file order.

        Args:
            names: requested array names (may be reordered).
            names_list: names stored in the data header, in file order.

        Returns:
            List of column indices in the requested order.

        Raises:
            ValueError: if any requested name is not found in *names_list*.
        """
        _name_to_idx = {n: i for i, n in enumerate(names_list)}
        _cols = []
        for nm in names:
            if nm not in _name_to_idx:
                raise ValueError(
                    f'Name "{nm}" not found. Available names: {names_list}'
                )
            _cols.append(_name_to_idx[nm])
        return _cols

    def read(
            self,
            groups: List[int]|slice|int = -1,
            indices: List[int]|slice|int = -1,      # alias: indices_iter
            is_copy: bool = True,
            indices_array: Optional[List[int]] = None,
            names: Optional[List[str]] = None,
            **kwargs,
    ) -> Dict[str, List[List[np.ndarray]]]:
        """
        Read arrays from the mmap file with independent cycle-level and
        array-level selection.

        Cycle selection (rows):
            *indices* (alias ``indices_iter``) — which cycles to read.
            ``-1`` means all cycles.

        Array selection (columns — mutually exclusive):
            *indices_array* — select arrays by column index.
            *names* — select arrays by stored name (requires new-format file).

        Args:
            groups: group number(s) to read.  ``-1`` = all.
            indices: cycle indices to read (alias: ``indices_iter``).
            is_copy: if True, copy data out of the mmap buffer.
            indices_array: column indices of the arrays to extract.
                Mutually exclusive with ``names``.
            names: array names to extract (resolved to column indices).
                Mutually exclusive with ``indices_array``.
                Raises ``RuntimeError`` for old-format files (no metadata).
            indices_iter: alias for ``indices``.

        Returns:
            Dict[str, List[List[np.ndarray]]],
            ``{f'group{i}': List[List[np.ndarray]]}``.

        Raises:
            ValueError: if ``indices_array`` and ``names`` are both specified.
            RuntimeError: if ``names`` is used on an old-format file.
        """
        # -- alias: indices_iter --
        if 'indices_iter' in kwargs:
            if indices != -1 and kwargs['indices_iter'] != indices:
                raise ValueError('`indices` and `indices_iter` are aliases; pass only one.')
            indices = kwargs['indices_iter']

        # -- mutually exclusive column selection --
        if indices_array is not None and names is not None:
            raise ValueError(
                '`indices_array` and `names` are mutually exclusive. '
                'Use one to select which arrays (columns) to extract.'
            )

        if isinstance(groups, int):
            if not groups < self.n_groups: raise ValueError(f'There is only {self.n_groups} groups available, but requested {groups}.')
            groups = range(self.n_groups) if groups < 0 else [groups, ]
        elif isinstance(groups, slice):
            _start, _stop, _step = groups.indices(self.n_groups)
            if not _start >= 0:
                raise ValueError(f'`groups` slice must start from the number >= 0, got {groups.start}.')
            if not _stop <= self.n_groups:
                raise ValueError(f'There is only {self.n_groups} groups available, but requested {groups.stop}.')
            groups = range(_start, _stop, _step)
        elif isinstance(groups, list):
            if not min(groups) >= 0:
                raise ValueError(f'elements in the `groups` must be >= 0, got {min(groups)}.')
            if not max(groups) < self.n_groups:
                raise ValueError(f'There is only {self.n_groups} groups available, but requested {max(groups)}.')
            if len(groups) != len(set(groups)):
                raise ValueError(f'`groups` must have unique elements, but some duplicate elements are found.')
            groups = sorted(groups)
        else:
            raise TypeError(f'`groups` must be a list or int, got {type(groups)}.')

        self._ptr = 16
        _grp_ptr = 0
        _n_select_groups = len(groups)
        output_arrs = dict()
        self._group_names = {}
        for i_grp in range(self.n_groups):
            if _grp_ptr >= _n_select_groups:
                break
            if i_grp == groups[_grp_ptr]:
                result = self._read_once(indices, is_copy,
                                         indices_array=indices_array, names=names)
                output_arrs[f'group{i_grp}'] = result['arrays']
                if result['names']:
                    self._group_names[i_grp] = result['names']
                _grp_ptr += 1
            else:
                self._skip_once()

        if is_copy:
            self.close()

        return output_arrs

    def _skip_once(self):
        """
        Skip this group data once, instead of reading it.

        Returns:

        """
        try:
            n_cycles, n_arrays, dtype_list, shape_list, stride_list, _names = self._parse_arr_head()
            _block_total_stride = sum(stride_list)  # total stride (bytes) of this group.
            self._ptr += n_cycles * _block_total_stride  # move the ptr to the end of the group
            self._mmp_f.seek(self._ptr)

        except Exception as e:
            self._tmp_close()
            raise RuntimeError(f'An error occurred while reading file {self._path}. ERROR: {e}.')

    @staticmethod
    def _extract_arrays(
        raw: bytes,
        cols: Optional[List[int]],
        dtypes: List[str],
        shapes: List[Tuple[int, ...]],
        strides: List[int],
        offsets: Optional[List[int]],
    ) -> List[np.ndarray]:
        """Slice one cycle's raw byte block into numpy arrays.

        Two modes, controlled by *cols*:

        *cols* is **None**
            Extract **all** arrays in file order.  Walks *strides*
            sequentially, advancing a byte pointer through *raw*.

        *cols* is a list of column indices
            Extract only the requested columns in the **requested
            order**.  Uses pre-computed cumulative byte *offsets*
            for O(1) slice boundaries per column.

        Args:
            raw: Byte string or mmap-backed memory view containing exactly one
                complete cycle of the current group.
            cols: Requested array-column indices in return order. ``None``
                extracts every array in file order.
            dtypes: NumPy dtype descriptors parsed from the group header.
            shapes: Array shapes parsed from the group header.
            strides: Number of bytes occupied by each array in one cycle.
            offsets: Cumulative byte offsets with length ``n_arrays + 1``.
                Required when ``cols`` is not ``None`` and ignored otherwise.

        Returns:
            List of NumPy arrays for the requested columns. If ``raw`` is a
            memory view, the returned arrays remain backed by the mmap file;
            otherwise they are backed by the copied cycle byte string.
        """
        result = []
        if cols is None:
            pos = 0
            for dt, sh, st in zip(dtypes, shapes, strides):
                result.append(np.frombuffer(raw[pos:pos + st], dtype=dt).reshape(sh))
                pos += st
        else:
            for col in cols:
                result.append(
                    np.frombuffer(raw[offsets[col]:offsets[col + 1]], dtype=dtypes[col]).reshape(shapes[col])
                )
        return result

    def _read_once(self, indices: List[int]|slice|int = -1, is_copy: bool = True,
                   indices_array: Optional[List[int]] = None,
                   names: Optional[List[str]] = None) -> Dict[str, object]:
        """Read selected cycles and columns from the current group.

        ``self._ptr`` must point to the beginning of the group's ``HEAD``
        record before this method is called. The pointer is advanced to the
        beginning of the next group after the read, regardless of which cycles
        were selected.

        Args:
            indices: Cycle selection inside this group. A negative integer
                reads all cycles; a nonnegative integer reads one cycle; a
                list preserves the requested order; and a slice follows normal
                Python slicing rules within the available cycle count.
            is_copy: If ``True``, copy each selected cycle out of the mmap file
                before constructing NumPy arrays. If ``False``, return arrays
                that directly reference the mmap region; all such references
                must be released before the reader can close the mapping.
            indices_array: Optional array-column indices to extract. Their
                order determines the order of arrays in each returned cycle.
                Mutually exclusive with ``names`` at the public ``read`` API.
            names: Optional stored array names to extract. Names are resolved
                to column indices while preserving the requested order and
                require name metadata in the group header.

        Returns:
            Dictionary with two entries:

            * ``'arrays'`` -- ``List[List[np.ndarray]]`` where the outer list
              follows the selected cycle order and the inner list follows the
              selected column order;
            * ``'names'`` -- names corresponding to the returned columns, or
              an empty list for an old unnamed group.

        Raises:
            RuntimeError: If the group header/data is corrupt, a requested
                named column is unavailable, or mmap-backed arrays prevent
                cleanup after another read error.
            ValueError: If a cycle or column selection is outside the group.
            TypeError: If ``indices`` has an unsupported type.
        """
        try:
            # --- parse header ---
            n_cycles, n_arrays, dtypes, shapes, strides, file_names = self._parse_arr_head()

            # --- resolve column selection ---
            if names is not None:
                if not file_names:
                    raise RuntimeError(
                        f'`names` requested but file `{self._path}` has no array name metadata.'
                    )
                selected_cols = self._resolve_names(names, file_names)
            elif indices_array is not None:
                selected_cols = list(indices_array)
            else:
                selected_cols = None

            col_offsets = None
            if selected_cols is not None:
                col_offsets = [0]
                for s in strides:
                    col_offsets.append(col_offsets[-1] + s)

            # --- resolve cycle selection ---
            if isinstance(indices, list):
                if not min(indices) >= 0:
                    raise ValueError(f'elements in the `indices` must be >= 0, got {min(indices)}.')
                if not max(indices) < n_cycles:
                    raise ValueError(f'indices {max(indices)} out of range ({n_cycles} cycles).')
                cycle_indices = indices
            elif isinstance(indices, slice):
                _start, _stop, _step = indices.indices(n_cycles)
                if not (_start >= 0 and _stop <= n_cycles):
                    raise ValueError(f'indices {indices} out of range.')
                cycle_indices = range(_start, _stop, _step)
            elif isinstance(indices, int):
                if not indices < n_cycles:
                    raise ValueError(f'indices {indices} out of range ({n_cycles} cycles).')
                cycle_indices = range(n_cycles) if indices < 0 else [indices]
            else:
                raise TypeError(f'Expected List[int], slice, or int for indices, got {type(indices)}.')

            # --- read cycles ---
            cycle_bytes = sum(strides)
            all_cycles = []
            for cyc in cycle_indices:
                cyc_start = self._ptr + cyc * cycle_bytes
                self._mmp_f.seek(cyc_start)
                raw = (self._mmp_f.read(cycle_bytes) if is_copy
                       else memoryview(self._mmp_f)[cyc_start:cyc_start + cycle_bytes])
                all_cycles.append(self._extract_arrays(raw, selected_cols, dtypes, shapes, strides, col_offsets))
                if not is_copy:
                    self._mmp_f.seek(cyc_start + cycle_bytes)

            self._ptr += n_cycles * cycle_bytes
            self._mmp_f.seek(self._ptr)

            # --- return only the requested names ---
            ret_names = [file_names[c] for c in selected_cols] if (selected_cols is not None and file_names) else file_names
            return {'arrays': all_cycles, 'names': ret_names}

        except Exception as e:
            self._tmp_close()
            raise RuntimeError(f'An error occurred while reading file {self._path}. ERROR: {e}.')

    def _parse_arr_head(self, ) -> Tuple[int, int, List[str], List[Tuple], List[int], List[str]]:
        """
        Parse the array head information once.
        self._ptr must be at the end of a group.
        Then after calling this method, self._ptr will be moved to the start of the group data.

        Returns: n_cycles, n_arrays, dtype_list, shape_list, stride_list, names_list
        """
        r"""
        Head information (new format):
            `char``n_cycle``n_array``n_names``name1`...`name_n``dtype1``shape1[]`0`dtype2``shape2[]`0...`dtype_n``shape_n[]`0`byte_data`...

        Old format (no n_names field):
            `char``n_cycle``n_array``dtype1``shape1[]`0`dtype2``shape2[]`0...`dtype_n``shape_n[]`0`byte_data`...

        The two formats are distinguished by reading the 8 bytes after n_array as n_names:
        old-format dtype bytes (UTF-16 order+type + int32 length) decode as huge
        uint64 values (> 4 billion), while legitimate n_names is 0 <= n_names <= n_array.

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
            if n_arrays == 0:
                raise RuntimeError(f'Corrupt file: n_arrays is zero in `{self._path}`.')

            # DB 2.0 always stores exactly one name per array. The legacy
            # reader retains the discriminator only for DB 1.0 files created
            # during the transition to named groups.
            names_pos = self._mmp_f.tell()
            n_names_raw = self._mmp_f.read(8)
            n_names = int.from_bytes(n_names_raw, self._num_fmt, signed=False)
            names_list: List[str] = []

            if self._REQUIRE_NAMES and (
                    n_names != n_arrays or n_names > self._MAX_NAMES
            ):
                raise RuntimeError(
                    f'Canonical DB {__db_version__} group contains {n_arrays} '
                    f'arrays but {n_names} names.'
                )
            if n_names == 0:
                # No names (either old format or new format with n_names=0).
                # Fall through to read dtype descriptors.
                pass
            elif n_names == n_arrays and n_names <= self._MAX_NAMES:
                # New format: n_names matches array count — read names section.
                for _ in range(n_names):
                    name_len = int.from_bytes(self._mmp_f.read(8), self._num_fmt, signed=False)
                    if name_len > self._MAX_NAME_LENGTH:
                        raise RuntimeError(
                            f'Corrupt file: array name length {name_len} exceeds '
                            f'maximum ({self._MAX_NAME_LENGTH}) in file `{self._path}`.'
                        )
                    name_data = self._mmp_f.read(name_len * 2)
                    name = name_data.decode(self._str_fmt)
                    names_list.append(name)
                    # Skip padding to 64-bit (8-byte) boundary
                    remainder = (name_len * 2) % 8
                    if remainder:
                        self._mmp_f.read(8 - remainder)
                if self._REQUIRE_NAMES:
                    if any(not name for name in names_list):
                        raise RuntimeError('Canonical array names must be non-empty.')
                    if len(set(names_list)) != len(names_list):
                        raise RuntimeError(
                            f'Canonical array names must be unique, got {names_list}.'
                        )
            else:
                # Old format: the bytes read as n_names are actually the start of
                # the first dtype field (UTF-16 order+type + int32 itemsize), which
                # decodes to a huge uint64 (> 4 billion). Seek back.
                self._mmp_f.seek(names_pos)

            # --- Read array descriptors (same for old and new format) ---
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

            return n_cycles, n_arrays, dtype_list, shape_list, stride_list, names_list

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


class ArrayDumpReader(ArrayDumpReaderOld):
    """Read strict, fully named canonical array dumps from DB 2.0 files.

    The binary traversal and array-selection operations are shared with
    :class:`ArrayDumpReaderOld`, but this class accepts only the independent
    database version declared by ``__db_version__`` and requires one stored
    name for every array in every group.

    Args:
        path: Path to an existing canonical DB 2.0 binary dump.

    Legacy DB 1.0 files must be opened explicitly with
    :class:`ArrayDumpReaderOld` or a public ``read_*_old`` helper. They can
    then be rewritten with :func:`BUCToolkit.BatchStructures.convert_dump`.
    """

    _SUPPORTED_DB_VERSION = tuple(
        int(value) for value in __db_version__.split('.', 2)[:2]
    )
    _REQUIRE_NAMES = True


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

def _read_dump_segments(
        path: str,
        reader_type,
        indices: List[int] | slice | int,
        is_copy: bool,
        legacy_data_names: Tuple[str, ...] | None = None,
) -> List[dict]:
    """Read alternating header/data groups as independent dump segments.

    Motion trajectories store one static, single-cycle header group followed
    by one dynamic, multi-cycle data group. Appended runs repeat that pair and
    may contain different metadata. Canonical names are read directly from the
    file. Only unnamed DB 1.0 groups use the legacy positional mapping supplied
    by ``legacy_data_names``.

    Args:
        path: Path to the binary dump file.
        reader_type: Reader class used to enforce either canonical DB 2.0 or
            legacy DB 1.0 parsing.
        indices: Cycle selection applied to every dynamic data group. Header
            groups are always read at their single stored cycle.
        is_copy: Whether returned arrays are copied from the memory map.
        legacy_data_names: Positional names assigned to unnamed DB 1.0 data
            columns. If omitted, positional string keys such as ``'0'`` and
            ``'1'`` are used.

    Returns:
        List of ``{'header': header, 'data': data}`` dictionaries in file
        order. ``header`` maps stored metadata names to arrays. ``data`` maps
        each dynamic name to its list of selected cycle arrays.

    Raises:
        EOFError: If groups cannot be paired or a header has multiple cycles.
        ValueError: If an unnamed legacy group has an unsupported number of
            columns.
    """
    reader = reader_type(path)
    if reader.n_groups % 2 != 0:
        raise EOFError(
            f'Expected paired header/data groups, got {reader.n_groups} groups.'
        )

    # ``reader.names`` describes only the most recent read, so preserve the
    # header and data name maps immediately after their respective reads.
    raw_headers = reader.read(groups=slice(0, None, 2), is_copy=is_copy)
    header_names = reader.names or {}
    raw_data = reader.read(
        groups=slice(1, None, 2), indices=indices, is_copy=is_copy
    )
    data_names = reader.names or {}

    segments = []
    for segment_index in range(reader.n_groups // 2):
        header_group_index = 2 * segment_index
        data_group_index = header_group_index + 1
        header_cycles = raw_headers[f'group{header_group_index}']
        data_cycles = raw_data[f'group{data_group_index}']
        if len(header_cycles) != 1:
            raise EOFError(
                f'Header group {header_group_index} must contain one cycle, '
                f'got {len(header_cycles)}.'
            )

        header_arrays = header_cycles[0]
        current_header_names = header_names.get(header_group_index)
        if current_header_names is None:
            if reader_type is not ArrayDumpReaderOld:
                raise ValueError(
                    f'Canonical header group {header_group_index} has no names.'
                )
            # DB 1.0 headers were positional. Keep that fixed schema confined
            # to the legacy path; canonical headers are entirely name-driven.
            current_header_names = ['cell_vec', 'atomic_numbers', 'fixed_mask']
            if len(header_arrays) == 3:
                pass
            elif len(header_arrays) == 4:
                current_header_names.insert(0, 'batch_indices')
            else:
                raise ValueError(
                    f'Legacy header group {header_group_index} has '
                    f'{len(header_arrays)} arrays; expected 3 or 4.'
                )

        current_data_names = data_names.get(data_group_index)
        if current_data_names is None:
            if legacy_data_names is None:
                current_data_names = [
                    str(index) for index in range(len(data_cycles[0]))
                ]
            else:
                if data_cycles and len(data_cycles[0]) != len(legacy_data_names):
                    raise ValueError(
                        f'Legacy data group {data_group_index} has '
                        f'{len(data_cycles[0])} arrays; expected '
                        f'{len(legacy_data_names)} for {legacy_data_names}.'
                    )
                current_data_names = list(legacy_data_names)

        # Canonical metadata stays separate from dynamic columns. Therefore
        # adding an optional header field requires no reader-side name list.
        header = dict(zip(current_header_names, header_arrays))
        data = {
            name: [cycle[column_index] for cycle in data_cycles]
            for column_index, name in enumerate(current_data_names)
        }
        segments.append({'header': header, 'data': data})
    return segments


def read_dump_segments(
        path: str,
        indices: List[int] | slice | int = -1,
        is_copy: bool = True,
) -> List[dict]:
    """Return lossless DB 2.0 trajectory segments with their own metadata.

    This is the lowest-level public trajectory reader. Unlike
    :func:`read_dump_arrays`, it never merges appended runs, so a changed batch
    size, atom count, or optional header field remains associated with the
    dynamic group that was written under that metadata.

    Args:
        path: Path to a canonical DB 2.0 MD, MC, or optimizer dump.
        indices: Cycle selection applied independently to each dynamic group.
            A negative integer reads all cycles.
        is_copy: Whether arrays are copied out of the underlying memory map.

    Returns:
        List of segment dictionaries in file order. Each segment contains a
        ``header`` dictionary of named static arrays and a ``data`` dictionary
        whose values are lists of selected cycle arrays. No arrays are stacked
        or split by structure.
    """
    return _read_dump_segments(
        path, ArrayDumpReader, indices=indices, is_copy=is_copy
    )


def _merge_uniform_dump_segments(segments: List[dict]) -> dict:
    """Merge segments that share identical metadata and dynamic schemas.

    Static metadata is copied once from the first segment. Dynamic cycle lists
    are then concatenated in segment order. The merge is intentionally refused
    if a later segment changes either its header or its ordered column names,
    because flattening such segments would discard their association.

    Args:
        segments: Segment dictionaries returned by
            :func:`read_dump_segments` or :func:`_read_dump_segments`.

    Returns:
        One dictionary containing the shared static arrays and concatenated
        dynamic cycle lists. An empty input returns an empty dictionary.

    Raises:
        ValueError: If static metadata or ordered dynamic names differ between
            segments.
    """
    if not segments:
        return {}
    reference_header = segments[0]['header']
    reference_names = tuple(segments[0]['data'])
    output = dict(reference_header)
    output.update({name: [] for name in reference_names})

    for segment_index, segment in enumerate(segments):
        header = segment['header']
        if tuple(header) != tuple(reference_header) or any(
                not np.array_equal(header[name], reference_header[name])
                for name in reference_header
        ):
            raise ValueError(
                f'Segment {segment_index} has different static metadata. Use '
                f'read_dump_segments() to retain per-segment headers.'
            )
        if tuple(segment['data']) != reference_names:
            raise ValueError(
                f'Segment {segment_index} has different dynamic columns. Use '
                f'read_dump_segments() to retain the individual schemas.'
            )
        for name in reference_names:
            output[name].extend(segment['data'][name])
    return output


def read_dump_arrays(
        path: str,
        indices: List[int]|slice|int = -1,
        is_copy: bool = True,
) -> dict:
    """Read named trajectory columns without converting them to structures.

    BUCToolkit motion dumps contain alternating groups. A one-cycle static
    header group stores system metadata, followed by a multi-cycle data group
    containing the registered state quantities. This convenience function
    merges segments only when their metadata and dynamic schemas are equal.
    Use :func:`read_dump_segments` for lossless access to heterogeneous
    appended runs. Dynamic arrays remain in their on-disk batch layout.

    Args:
        path: Path to a binary dump written by :class:`ArrayDumper` through an
            MD, MC, or optimization runner.
        indices: Cycle selection applied independently to each dynamic data
            group. A negative integer reads every cycle; a nonnegative integer
            reads one cycle; lists and slices follow :meth:`ArrayDumpReader.read`.
        is_copy: Whether to copy arrays out of the mmap file. If ``False``, the
            returned arrays may retain exported mmap pointers. Every reference
            must be released before the underlying mapping can be closed.

    Returns:
        Dictionary containing every named static header entry as stored,
        followed by every dynamic registered quantity as
        ``{name: [cycle_array, ...]}``. Dynamic arrays are neither stacked nor
        split: for example, irregular atom-wise values remain shaped
        ``[1, sum(n_atoms), ...]``.

    Raises:
        EOFError: If the number of static header groups and dynamic data groups
            differs, indicating a truncated or corrupt dump.
        ValueError: If appended segments have different metadata or columns.
        RuntimeError: If the file cannot be opened or an array group cannot be
            parsed by :class:`ArrayDumpReader`.
    """
    return _merge_uniform_dump_segments(
        read_dump_segments(path, indices=indices, is_copy=is_copy)
    )


def read_dump_arrays_old(
        path: str,
        indices: List[int] | slice | int = -1,
        is_copy: bool = True,
) -> dict:
    """Read a uniform legacy DB 1.0 header/data dump.

    The function preserves the stored array layout and merges appended
    segments only when their static metadata and ordered columns match.
    Unnamed dynamic arrays are exposed under positional string keys. Use
    :func:`read_md_traj_old` or :func:`read_mc_traj_old` when the producing
    framework is known and semantic column names are required.

    Args:
        path: Path to a legacy DB 1.0 motion dump containing alternating
            header/data groups.
        indices: Cycle selection applied independently to every data group. A
            negative integer reads all cycles.
        is_copy: Whether arrays are copied out of the underlying memory map.

    Returns:
        Dictionary containing the shared named or positionally reconstructed
        header arrays and every dynamic column as a list of cycle arrays.
    """
    segments = _read_dump_segments(
        path, ArrayDumpReaderOld, indices=indices, is_copy=is_copy
    )
    return _merge_uniform_dump_segments(segments)


def _split_dump_columns(header: dict, data: dict) -> Tuple[dict, int]:
    """Split raw trajectory columns into cycle-major per-sample values.

    The irregular-batch file layout uses ``(1, sum(n_atoms), ...)`` for
    atom-wise tensors, while per-structure tensors use
    ``(n_structures, ...)``.  Tensor dimensionality alone cannot distinguish
    these cases: a constraint matrix such as ``Fc[n_structure, n_constraint]``
    is two-dimensional but must *not* be split at atom boundaries.

    This helper therefore classifies a column from its leading dimensions:

    * ``(1, total_atoms, ...)`` in an irregular batch is atom-wise and is split
      by ``batch_indices``;
    * ``(n_structures, ...)`` is already per-structure and is expanded along
      its first axis;
    * a scalar or system-wide tensor is repeated for every structure so every
      returned column remains aligned with the cycle-major sample order.

    Args:
        header: Arbitrary named metadata for this segment. ``batch_indices``
            alone selects the irregular layout; other entries are optional.
        data: Dynamic named columns for this segment. All columns must have the
            same number of cycles.

    Returns:
        ``(_columns, n_cycles)`` where every value in ``_columns`` is a flat
        list ordered as ``cycle0/sample0, cycle0/sample1, ...``. Per-structure
        NumPy scalar values are converted to Python scalars to preserve the
        historical ``BatchStructures`` input contract.

    Raises:
        EOFError: If dynamic columns contain inconsistent cycle counts.
    """
    _batch = header.get('batch_indices')
    _is_irr = _batch is not None
    if _is_irr:
        _batch = np.asarray(_batch).reshape(-1)
        _batch_ptr = np.cumsum(_batch)[:-1]
        _n_batch = len(_batch)
        _n_atoms = int(np.sum(_batch))
    else:
        if data.get('X'):
            _n_batch = np.asarray(data['X'][0]).shape[0]
        elif 'atomic_numbers' in header:
            _atomic_numbers = np.asarray(header['atomic_numbers'])
            _n_batch = 1 if _atomic_numbers.ndim == 1 else len(_atomic_numbers)
        else:
            _first_column = next((values for values in data.values() if values), None)
            if _first_column is None or np.asarray(_first_column[0]).ndim == 0:
                raise ValueError('Cannot infer the regular batch size.')
            _n_batch = np.asarray(_first_column[0]).shape[0]
        _batch_ptr = None
        _n_atoms = None

    _data_names = list(data)
    if not _data_names:
        return {}, 0

    _n_cycles = len(data[_data_names[0]])
    _columns: dict = {}
    for _name in _data_names:
        _col = data[_name]
        if len(_col) != _n_cycles:
            raise EOFError(
                f'Inconsistent cycle count for column {_name!r}: '
                f'expected {_n_cycles}, got {len(_col)}.'
            )

        _flat = []
        for _arr in _col:
            _arr = np.asarray(_arr)
            if (
                    _is_irr
                    and _arr.ndim >= 2
                    and _arr.shape[0] == 1
                    and _arr.shape[1] == _n_atoms
            ):
                # Concatenated atom-wise tensor: remove the synthetic leading
                # batch axis and split the atom axis into real structures.
                _flat.extend(np.split(_arr[0], _batch_ptr, axis=0))
            elif _arr.ndim >= 1 and _arr.shape[0] == _n_batch:
                # Scalar/vector/matrix values already indexed by structure.
                for _i in range(_n_batch):
                    _value = _arr[_i]
                    # Keep the historical reader contract: per-structure
                    # scalar columns (Energy, temperature, acceptance, ...)
                    # are ordinary Python scalars, while vectors/matrices stay
                    # as numpy arrays.
                    if np.ndim(_value) == 0:
                        _value = _value.item()
                    _flat.append(_value)
            elif _arr.ndim == 0:
                # A global scalar (for example a shared temperature) applies
                # equally to every structure in this cycle.
                _flat.extend(_arr.item() for _ in range(_n_batch))
            else:
                # Preserve uncommon system-wide tensors without guessing an
                # atom axis.  Each sample receives its own copy so callers may
                # mutate the returned values independently.
                _flat.extend(_arr.copy() for _ in range(_n_batch))
        _columns[_name] = _flat

    return _columns, _n_cycles


def _select_final_dump_cycle(segments: List[dict]) -> List[dict]:
    """Select the last available cycle from the final dump segment.

    This implements ``read_opt_structures(..., only_opt=True)``. Appended
    optimization runs are ordered chronologically, so the optimized structure
    is represented by the final selected cycle of the final segment. Header
    arrays are retained unchanged and the input list is not modified.

    Args:
        segments: Ordered segment dictionaries with named ``header`` and
            ``data`` mappings.

    Returns:
        A one-segment list whose dynamic columns each contain at most their
        final cycle. An empty input returns an empty list.
    """
    if not segments:
        return []
    final_segment = segments[-1]
    final_data = {
        name: values[-1:] if values else []
        for name, values in final_segment['data'].items()
    }
    return [{'header': final_segment['header'], 'data': final_data}]


def _combine_split_segment_columns(segments: List[dict]) -> dict:
    """Split dynamic columns per structure and combine compatible segments.

    Each segment is split using its own ``batch_indices`` and other metadata,
    which preserves appended runs whose batch shapes differ. Dynamic schemas
    must still have the same ordered names so the resulting per-sample lists
    remain aligned across the returned dictionary.

    Args:
        segments: Ordered segment dictionaries returned by a canonical or
            legacy segment reader.

    Returns:
        Dictionary mapping each dynamic name to a flat list ordered by
        ``segment/cycle/structure``. An empty input returns an empty dictionary.

    Raises:
        ValueError: If the ordered dynamic column names differ between
            segments.
    """
    if not segments:
        return {}
    reference_names = tuple(segments[0]['data'])
    combined_columns = {name: [] for name in reference_names}
    for segment_index, segment in enumerate(segments):
        if tuple(segment['data']) != reference_names:
            raise ValueError(
                f'Segment {segment_index} has different dynamic columns. Use '
                f'read_dump_segments() for heterogeneous schemas.'
            )
        split_columns, _ = _split_dump_columns(
            segment['header'], segment['data']
        )
        for name in reference_names:
            combined_columns[name].extend(split_columns[name])
    return combined_columns


def _append_segment_structures(
        segments: List[dict],
        required_names: Tuple[str, ...],
        include_force: bool,
        include_velocity: bool,
) -> BatchStructures:
    """Build ``BatchStructures`` from independently interpreted segments.

    The method reconstructs element rows and fixed masks from each segment's
    static metadata, splits its dynamic columns into cycle-major samples, and
    appends all samples to one structure collection. ``batch_indices`` alone
    selects irregular reconstruction; other header fields do not participate
    in that branch decision.

    Args:
        segments: Ordered segment dictionaries containing static ``cell_vec``,
            ``atomic_numbers``, and ``fixed_mask`` entries plus dynamic data.
        required_names: Dynamic columns required by the calling MD, MC, or
            optimizer reader.
        include_force: Whether the ``Force`` column is appended to the output.
        include_velocity: Whether the ``V`` column is appended to the output.

    Returns:
        A validated :class:`BatchStructures` containing one sample for every
        selected ``segment/cycle/structure`` combination.

    Raises:
        ValueError: If a segment lacks one of ``required_names``.
    """
    sample_ids = []
    cell_list = []
    element_list = []
    number_list = []
    coordinate_type_list = []
    coordinate_list = []
    fixed_mask_list = []
    energy_list = []
    force_list = []
    velocity_list = []

    for segment_index, segment in enumerate(segments):
        missing_names = [
            name for name in required_names if name not in segment['data']
        ]
        if missing_names:
            raise ValueError(
                f'Columns {missing_names} are absent from segment '
                f'{segment_index}. Add them to dump_quantities.'
            )

        header = segment['header']
        batch_counts = header.get('batch_indices')
        atomic_numbers = np.asarray(header['atomic_numbers'])
        fixed_mask = np.asarray(header['fixed_mask'])
        # Irregular headers store atom-wise metadata on one concatenated atom
        # axis. Regular headers already store one rectangular row per image.
        if batch_counts is not None:
            batch_counts = np.asarray(batch_counts).reshape(-1)
            split_points = np.cumsum(batch_counts)[:-1]
            atoms_per_structure = np.split(
                atomic_numbers.reshape(-1), split_points, axis=0
            )
            fixed_per_structure = np.split(
                fixed_mask[0], split_points, axis=0
            )
        else:
            atoms_per_structure = (
                [atomic_numbers] if atomic_numbers.ndim == 1
                else [row for row in atomic_numbers]
            )
            fixed_per_structure = (
                [fixed_mask] if fixed_mask.ndim == 2
                else [row for row in fixed_mask]
            )

        elements_per_structure = []
        numbers_per_structure = []
        for numbers in atoms_per_structure:
            elements, _, reduced_numbers = elem_list_reduce(numbers)
            elements_per_structure.append(elements)
            numbers_per_structure.append(reduced_numbers)

        # Split dynamic values with the same segment-local metadata before
        # extending the common cycle-major output lists.
        columns, n_cycles = _split_dump_columns(header, segment['data'])
        n_structures = len(atoms_per_structure)
        segment_prefix = (
            f'segment{segment_index}_' if len(segments) > 1 else ''
        )
        sample_ids.extend(
            f'{segment_prefix}samp{structure_index}_step{cycle_index}'
            for cycle_index in range(n_cycles)
            for structure_index in range(n_structures)
        )
        cell_list.extend([cell for cell in header['cell_vec']] * n_cycles)
        element_list.extend(elements_per_structure * n_cycles)
        number_list.extend(numbers_per_structure * n_cycles)
        coordinate_type_list.extend(['C'] * (n_structures * n_cycles))
        coordinate_list.extend(columns['X'])
        fixed_mask_list.extend(fixed_per_structure * n_cycles)
        energy_list.extend(columns['Energy'])
        if include_force:
            force_list.extend(columns['Force'])
        if include_velocity:
            velocity_list.extend(columns['V'])

    structures = BatchStructures()
    append_args = [
        sample_ids,
        cell_list,
        element_list,
        number_list,
        coordinate_type_list,
        coordinate_list,
        fixed_mask_list,
        energy_list,
    ]
    if include_force:
        append_args.append(force_list)
    if include_velocity:
        append_args.append(velocity_list)
    structures.append_from_lists(*append_args)
    structures._check_id()
    structures._check_len()
    return structures


def read_md_traj(
        path,
        indices: List[int]|slice|int = -1,
        is_copy: bool = True,
        out_arrays: bool = False,
):
    """Read a molecular-dynamics trajectory dump.

    The expected file layout is an alternating static-header/data pair. With
    ``sX`` denoting ``[n_batch, n_atom, n_dim]`` for a regular batch or
    ``[1, sum(n_atoms), n_dim]`` for an irregular batch, the standard columns
    are:

    * static header, one cycle: optional ``batch_indices[n_batch]``,
      ``cell_vec[n_batch, 3, 3]``, ``atomic_numbers``, and ``fixed_mask[sX]``;
    * dynamic data: ``Energy[n_batch]``, ``X[sX]``, ``V[sX]``, and
      ``Force[sX]`` plus any extra registered quantities.

    All header/data pairs are loaded. ``indices`` controls the selected frames
    inside every dynamic group. Irregular atom-wise arrays are split according
    to ``batch_indices``; per-structure vectors and matrices retain their
    trailing dimensions.

    Args:
        path: Path to the binary MD dump file.
        indices: Frame indices selected independently in each dynamic group. A
            negative integer reads every frame.
        is_copy: Whether to copy arrays out of the mmap file. If ``False``, all
            mmap-backed references must be released before the mapping closes.
        out_arrays: If ``False``, require the standard ``Energy``, ``X``, ``V``,
            and ``Force`` columns and construct :class:`BatchStructures`. If
            ``True``, return every available dynamic column after per-sample
            splitting, allowing custom dumps that omit standard columns.

    Returns:
        If ``out_arrays`` is ``False``, a :class:`BatchStructures` containing
        one sample for every selected ``cycle/structure`` pair in cycle-major
        order. If ``True``, a dictionary
        ``{name: [per_sample_value, ...]}`` with the same ordering.

    Raises:
        ValueError: If a standard column required for ``BatchStructures`` is
            missing. Pass ``out_arrays=True`` to read a partial/custom dump.
        EOFError: If header/data groups or dynamic cycle counts are inconsistent.
    """
    segments = read_dump_segments(path, indices=indices, is_copy=is_copy)
    if out_arrays:
        return _combine_split_segment_columns(segments)
    return _append_segment_structures(
        segments,
        required_names=('Energy', 'X', 'V', 'Force'),
        include_force=True,
        include_velocity=True,
    )

def read_mc_traj(
        path,
        indices: List[int]|slice|int = -1,
        is_copy: bool = True,
        out_arrays: bool = False,
):
    """Read a Monte-Carlo trajectory dump.

    With ``sX`` denoting ``[n_batch, n_atom, n_dim]`` for a regular batch or
    ``[1, sum(n_atoms), n_dim]`` for an irregular batch, the standard layout is:

    * static header, one cycle: optional ``batch_indices[n_batch]``,
      ``cell_vec[n_batch, 3, 3]``, ``atomic_numbers``, and ``fixed_mask[sX]``;
    * dynamic data: legacy columns ``Energy[n_batch]`` and ``X[sX]``, plus any
      configured quantities such as ``delta_E``, ``is_accept``, and
      ``temperature``.

    All header/data pairs are loaded. Irregular atom-wise arrays are split at
    the atom counts in ``batch_indices``. Per-structure scalar, vector, and
    matrix columns are expanded along their existing structure axis rather
    than being mistaken for atom-wise data.

    Args:
        path: Path to the binary MC dump file.
        indices: Frame indices selected independently in each dynamic group. A
            negative integer reads every frame.
        is_copy: Whether to copy arrays out of the mmap file. If ``False``, all
            mmap-backed references must be released before the mapping closes.
        out_arrays: If ``False``, require ``Energy`` and ``X`` and construct
            :class:`BatchStructures`. If ``True``, return all available dynamic
            columns after per-sample splitting, including custom quantities.

    Returns:
        If ``out_arrays`` is ``False``, a :class:`BatchStructures` containing
        one structure for every selected ``cycle/structure`` pair. If ``True``,
        a dictionary ``{name: [per_sample_value, ...]}`` in cycle-major order.

    Raises:
        ValueError: If ``Energy`` or ``X`` is absent while ``out_arrays`` is
            ``False``.
        EOFError: If header/data groups or dynamic cycle counts are inconsistent.
    """
    segments = read_dump_segments(path, indices=indices, is_copy=is_copy)
    if out_arrays:
        return _combine_split_segment_columns(segments)
    return _append_segment_structures(
        segments,
        required_names=('Energy', 'X'),
        include_force=False,
        include_velocity=False,
    )

def read_opt_structures(
        path,
        indices: List[int]|slice|int = -1,
        is_copy: bool = True,
        only_opt: bool = False,
        out_arrays: bool = False,
):
    """Read structures from a named-column optimization trajectory.

    The current optimizer dump format uses the same alternating group layout
    as MD and MC. The one-cycle static header contains optional
    ``batch_indices``, cell vectors, atomic numbers, and fixed masks. Dynamic
    groups normally contain ``Energy[n_batch]``, coordinates ``X[sX]``, and
    forces ``Force[sX]``; additional registered convergence quantities such as
    ``E_diff``, ``F_eps``, and ``X_grad`` may also be present. Coordinates in
    the returned structures are Cartesian.

    Args:
        path: Path to the binary optimization dump file.
        indices: Cycle indices selected independently in each dynamic group. A
            negative integer reads every cycle.
        is_copy: Whether to copy arrays out of the mmap file. If ``False``, all
            mmap-backed references must be released before the mapping closes.
        only_opt: If ``True``, retain only the final selected cycle of every
            dynamic column after all data groups have been merged. This is the
            optimized/final snapshot when the full trajectory is selected.
        out_arrays: If ``False``, require ``Energy``, ``X``, and ``Force`` and
            construct :class:`BatchStructures`. If ``True``, return all
            available dynamic columns after per-sample splitting.

    Returns:
        If ``out_arrays`` is ``False``, a :class:`BatchStructures` containing
        the selected optimization snapshots. If ``True``, a dictionary
        ``{name: [per_sample_value, ...]}`` in cycle-major order.

    Raises:
        ValueError: If a standard column required for ``BatchStructures`` is
            missing while ``out_arrays`` is ``False``.
        EOFError: If header/data groups or dynamic cycle counts are inconsistent.
    """
    segments = read_dump_segments(path, indices=indices, is_copy=is_copy)
    if only_opt:
        segments = _select_final_dump_cycle(segments)
    if out_arrays:
        return _combine_split_segment_columns(segments)
    return _append_segment_structures(
        segments,
        required_names=('Energy', 'X', 'Force'),
        include_force=True,
        include_velocity=False,
    )


def read_md_traj_old(
        path,
        indices: List[int] | slice | int = -1,
        is_copy: bool = True,
        out_arrays: bool = False,
):
    """Read a legacy DB 1.0 MD trajectory.

    DB 1.0 MD files use alternating static header and dynamic data groups. An
    unnamed data group is interpreted in the historical order ``Energy``,
    ``X``, ``V``, and ``Force``; transitional named DB 1.0 groups retain their
    stored names. Every appended segment is split with its own header metadata.

    Args:
        path: Path to a legacy DB 1.0 MD trajectory.
        indices: Cycle selection applied independently to every dynamic group.
            A negative integer reads all cycles.
        is_copy: Whether arrays are copied out of the underlying memory map.
        out_arrays: If ``False``, construct :class:`BatchStructures` and require
            the four historical MD columns. If ``True``, return all available
            dynamic columns after per-structure splitting.

    Returns:
        A :class:`BatchStructures` object when ``out_arrays`` is ``False``;
        otherwise a dictionary of cycle-major per-structure column lists.

    Retained files should be converted to DB 2.0 before they are passed to the
    canonical :func:`read_md_traj` reader.
    """
    segments = _read_dump_segments(
        path,
        ArrayDumpReaderOld,
        indices=indices,
        is_copy=is_copy,
        legacy_data_names=('Energy', 'X', 'V', 'Force'),
    )
    if out_arrays:
        return _combine_split_segment_columns(segments)
    return _append_segment_structures(
        segments,
        required_names=('Energy', 'X', 'V', 'Force'),
        include_force=True,
        include_velocity=True,
    )


def read_mc_traj_old(
        path,
        indices: List[int] | slice | int = -1,
        is_copy: bool = True,
        out_arrays: bool = False,
):
    """Read a legacy DB 1.0 MC trajectory.

    DB 1.0 MC files use alternating static header and dynamic data groups. An
    unnamed data group is interpreted in the historical order ``Energy`` and
    ``X``; transitional named DB 1.0 groups retain their stored names. Every
    appended segment is split with its own header metadata.

    Args:
        path: Path to a legacy DB 1.0 MC trajectory.
        indices: Cycle selection applied independently to every dynamic group.
            A negative integer reads all cycles.
        is_copy: Whether arrays are copied out of the underlying memory map.
        out_arrays: If ``False``, construct :class:`BatchStructures` and require
            ``Energy`` and ``X``. If ``True``, return all available dynamic
            columns after per-structure splitting.

    Returns:
        A :class:`BatchStructures` object when ``out_arrays`` is ``False``;
        otherwise a dictionary of cycle-major per-structure column lists.

    Retained files should be converted to DB 2.0 before they are passed to the
    canonical :func:`read_mc_traj` reader.
    """
    segments = _read_dump_segments(
        path,
        ArrayDumpReaderOld,
        indices=indices,
        is_copy=is_copy,
        legacy_data_names=('Energy', 'X'),
    )
    if out_arrays:
        return _combine_split_segment_columns(segments)
    return _append_segment_structures(
        segments,
        required_names=('Energy', 'X'),
        include_force=False,
        include_velocity=False,
    )


def read_opt_structures_old(
        path,
        indices: List[int]|slice|int = -1,
        is_copy: bool = True
):
    """
    Legacy reader for the old single-group 8-array opt dump format.

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
    reader = ArrayDumpReaderOld(path)
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
