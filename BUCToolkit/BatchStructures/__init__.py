#  Copyright (c) 2024-2025.7.4, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: __init__.py
#  Environment: Python 3.12

from BUCToolkit.BatchStructures.BatchStructuresBase import BatchStructures
from BUCToolkit.BatchStructures.StructuresIO import (
    ArrayDumper,
    ArrayDumpReader,
    ArrayDumpReaderOld,
    read_dump_arrays,
    read_dump_arrays_old,
    read_dump_segments,
    read_freq,
    read_mc_traj,
    read_mc_traj_old,
    read_md_traj,
    read_md_traj_old,
    read_opt_structures,
    read_opt_structures_old,
)
from BUCToolkit.BatchStructures.batch import Batch
from BUCToolkit.BatchStructures.data import Data


def convert_dump(input_path, output_path, kind, overwrite=False):
    """Convert a legacy DB 1.0 dump to the canonical DB 2.0 format.

    The implementation is imported lazily so executing
    ``python -m BUCToolkit.BatchStructures.convert_dump`` does not preload the
    conversion module through this package initializer.

    Args:
        input_path: Path to the existing DB 1.0 file.
        output_path: Destination path for the DB 2.0 file.
        kind: Producing framework: ``'md'``, ``'mc'``, or ``'opt'``.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        None. The converted file is written to ``output_path``.
    """
    from BUCToolkit.BatchStructures.convert_dump import convert_dump as _convert_dump
    return _convert_dump(input_path, output_path, kind, overwrite)


__all__ = [
    "read_dump_arrays",
    "read_dump_arrays_old",
    "read_dump_segments",
    "read_freq",
    "read_opt_structures",
    "read_opt_structures_old",
    "read_md_traj",
    "read_md_traj_old",
    "read_mc_traj",
    "read_mc_traj_old",
    "convert_dump",
    "ArrayDumper",
    "ArrayDumpReader",
    "ArrayDumpReaderOld",
    'Batch',
    'Data',
    'BatchStructures'
]
