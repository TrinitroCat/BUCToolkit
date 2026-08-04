#  Copyright (c) 2026.4.15, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: convert_data.py
#  Environment: Python 3.12
import inspect
import os
import time
from typing import Literal
from BUCToolkit.io import read_md_traj, read_mc_traj, read_opt_structures, OUTCAR2Feat, POSCARs2Feat, Cif2Feat, ASETraj2Feat
import BUCToolkit as bt


def backup_output(path: str, expected_kind: Literal['file', 'directory']) -> str | None:
    """Move an existing output to a timestamped backup path.

    Args:
        path: Output path that may already exist.
        expected_kind: Whether ``path`` is required to be a regular file or a
            directory when it exists.

    Returns:
        The absolute backup path, or ``None`` when ``path`` did not exist.

    Raises:
        ValueError: If the output is a symbolic link or ``expected_kind`` is
            unsupported.
        IsADirectoryError: If a file output names an existing directory.
        NotADirectoryError: If a directory output names an existing file.
        OSError: If the existing output cannot be renamed.
    """
    path = os.path.abspath(path)
    if not os.path.lexists(path):
        return None
    if os.path.islink(path):
        raise ValueError(f"Output `{path}` must not be a symbolic link.")
    if expected_kind == 'file' and not os.path.isfile(path):
        raise IsADirectoryError(f"Output file `{path}` is an existing directory.")
    if expected_kind == 'directory' and not os.path.isdir(path):
        raise NotADirectoryError(f"Output directory `{path}` is an existing file.")
    if expected_kind not in {'file', 'directory'}:
        raise ValueError(f"Unknown output kind `{expected_kind}`.")

    backup_base = f"{path}.bak{time.strftime('%Y%m%d_%H%M%S')}"
    backup_path = backup_base
    suffix = 1
    while os.path.lexists(backup_path):
        backup_path = f'{backup_base}_{suffix}'
        suffix += 1
    os.rename(path, backup_path)
    return backup_path


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

    backup_output(opath, 'directory')
    os.makedirs(opath)
    out_format = OUT_DICT[out]
    if out_format is not None:
        f.write2text(opath, None, file_format=out_format)
    else:
        f.save(opath, 'w')
