#  Copyright (c) 2026.4.15, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: convert_data.py
#  Environment: Python 3.12
import inspect
import os
from BUCToolkit.io import read_md_traj, read_mc_traj, read_opt_structures, OUTCAR2Feat, POSCARs2Feat, Cif2Feat, ASETraj2Feat
import BUCToolkit as bt

def main_convert(inp: str, ipath: str, out: str, opath: str):
    INP_DICT = {
        'md': read_md_traj,
        'mc': read_mc_traj,
        'opt': read_opt_structures,
        'outcar': OUTCAR2Feat,
        'poscar': POSCARs2Feat,
        'cif': Cif2Feat,
        'ase_traj': ASETraj2Feat,
        'bs': None
    }
    OUT_DICT = {
        'poscar': 'POSCAR',
        'cif': 'cif',
        'xyz': 'xyz',
        'bs': None
    }

    inp = inp.lower()
    out = out.lower()
    if inp not in INP_DICT:
        print(f'ERROR: The input format {inp} is not supported.')
        return
    if out not in OUT_DICT:
        print(f'ERROR: The output format {out} is not supported.')
        return

    converter = INP_DICT[inp]
    if converter is None:
        f = bt.load(ipath)
    elif inspect.isclass(converter):
        f = converter(ipath)
        f.read()
    else:
        f = converter(ipath)

    # output
    out_format = OUT_DICT[out]
    if out_format is not None:
        f.write2text(opath, None, file_format=out_format)
    else:
        if os.path.isdir(opath):
            print(f'WARNING: The output directory {opath} has already existed. Trying to write with appending mode...')
        f.save(opath, 'a')


