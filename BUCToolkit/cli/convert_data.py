#  Copyright (c) 2026.4.15, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: convert_data.py
#  Environment: Python 3.12
import inspect
import os
from BUCToolkit.cli._config import raise_output_backup_warning, backup_output
from BUCToolkit.io import read_md_traj, read_mc_traj, read_opt_structures, OUTCAR2Feat, POSCARs2Feat, Cif2Feat, ASETraj2Feat
import BUCToolkit as bt


def _conversion_paths(input_path: str, output_path: str) -> tuple[str, str]:
    """Resolve conversion paths while protecting input data from backup moves.

    Args:
        input_path: Source path supplied to the converter.
        output_path: Destination directory supplied to the converter.

    Returns:
        Absolute source and destination paths.

    Raises:
        ValueError: If the destination equals or contains the source path.
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    real_input = os.path.realpath(input_path)
    real_output = os.path.realpath(output_path)
    try:
        output_contains_input = os.path.commonpath((real_input, real_output)) == real_output
    except ValueError:
        output_contains_input = False
    if real_input == real_output or output_contains_input:
        raise ValueError(
            f"Conversion output `{output_path}` must not equal or contain input `{input_path}`."
        )
    return input_path, output_path


def main_convert(inp: str, ipath: str, out: str, opath: str):
    """Convert structures between formats supported by the CLI.

    Args:
        inp: Input format name.
        ipath: Input file or directory path.
        out: Output format name.
        opath: Output directory path.

    Returns:
        None. Converted structures are written below ``opath``.

    Raises:
        ValueError: If a format or input/output path relationship is invalid.
        OSError: If input data cannot be read or the output cannot be backed
            up and created.
    """
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
        raise ValueError(f'The input format {inp} is not supported.')
    if out not in OUT_DICT:
        raise ValueError(f'The output format {out} is not supported.')

    ipath, opath = _conversion_paths(ipath, opath)
    converter = INP_DICT[inp]
    if converter is None:
        f = bt.load(ipath)
    elif inspect.isclass(converter):
        f = converter(ipath)
        f.read()
    else:
        f = converter(ipath)

    old_output = backup_output(opath, 'directory')
    if old_output is not None:
        raise_output_backup_warning(opath, old_output)
    os.makedirs(opath)
    out_format = OUT_DICT[out]
    if out_format is not None:
        f.write2text(opath, None, file_format=out_format)
    else:
        f.save(opath, 'w')
