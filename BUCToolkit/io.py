"""
A fast API for all i/o operations
"""
#  Copyright (c) 2026.4.13, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: io.py
#  Environment: Python 3.12

from BUCToolkit.Preprocessing import load_files, write_files
from BUCToolkit.Preprocessing.load_files import OUTCAR2Feat, Xyz2Feat, ExtXyz2Feat, POSCARs2Feat, ASETraj2Feat, Cif2Feat, load_from_structures
from BUCToolkit.BatchStructures.StructuresIO import read_opt_structures, read_mc_traj, read_md_traj, ArrayDumpReader, ArrayDumper

__all__ = [
    'OUTCAR2Feat',
    'Xyz2Feat',
    'ExtXyz2Feat',
    'POSCARs2Feat',
    'ASETraj2Feat',
    'Cif2Feat',
    'read_opt_structures',
    'read_md_traj',
    'read_mc_traj',
    'ArrayDumpReader',
    'ArrayDumper',
    'load_from_structures',
    'load_files',
    'write_files',
]

