"""
The Traditional Style Main Function/Program that runs tasks by a single command line with
input files and args.
"""
#  Copyright (c) 2026.3.27, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: main.py
#  Environment: Python 3.12

import sys
import argparse
import time
import os
import re
import warnings
from typing import Dict, Any, Callable
import importlib.util
import hashlib

import torch as th
import numpy as np

import BUCToolkit as bt
from BUCToolkit.BatchGenerate.coords_interp import direction_for_finite_diff
from BUCToolkit.cli.print_logo import generate_display_art
import BUCToolkit.api as api
from BUCToolkit.api.DataLoaders import PyGDataLoader, ISFSPyGDataLoader
import BUCToolkit.Preprocessing.load_files as load_files
from BUCToolkit.cli.convert_data import backup_output, main_convert
from BUCToolkit.cli._config import load_input_config, prepare_output_root


def _selected_sample_indices(data, pattern: str | None) -> list[int]:
    """Return original positions selected by the CLI sample-ID pattern.

    Args:
        data: Structures object exposing ``Sample_ids``.
        pattern: Regular expression passed through ``DATA_NAME_SELECTOR``.

    Returns:
        Original integer positions in ``data``. All positions are returned when
        ``pattern`` is ``None``.

    Raises:
        RuntimeError: If a configured pattern matches no sample IDs.
        re.error: If ``pattern`` is not a valid regular expression.
    """
    if pattern is None:
        return list(range(len(data)))
    compiled_pattern = re.compile(pattern)
    indices = [
        index
        for index, sample_id in enumerate(data.Sample_ids)
        if re.match(compiled_pattern, str(sample_id)) is not None
    ]
    if len(indices) == 0:
        raise RuntimeError("No data matches the pattern given by `DATA_NAME_SELECTOR`.")
    return indices


def _validate_paired_structures(initial_data, paired_data, paired_name: str) -> None:
    """Validate position-wise compatibility of two structure collections.

    Sample IDs are intentionally excluded: paired files commonly use different
    initial/final suffixes, and position is the authoritative relationship.

    Args:
        initial_data: Initial-state Structures collection.
        paired_data: Final-state or displacement Structures collection.
        paired_name: Configuration field naming ``paired_data`` for errors.

    Returns:
        None.

    Raises:
        ValueError: If sample counts, coordinate layout, atom composition,
            cell layout, or fixed-mask layout are incompatible.
    """
    if len(initial_data) != len(paired_data):
        raise ValueError(
            f"`DATA_PATH` and `{paired_name}` sample counts differ: "
            f"{len(initial_data)} != {len(paired_data)}."
        )

    for index in range(len(initial_data)):
        initial_coords = np.asarray(initial_data.Coords[index])
        paired_coords = np.asarray(paired_data.Coords[index])
        if initial_coords.shape != paired_coords.shape:
            raise ValueError(
                f"Paired sample {index} coordinate shape differs between `DATA_PATH` "
                f"and `{paired_name}`: {initial_coords.shape} != {paired_coords.shape}."
            )
        if initial_coords.ndim != 2 or initial_coords.shape[1] != 3:
            raise ValueError(
                f"Paired sample {index} coordinates must have shape (n_atom, 3), "
                f"got {initial_coords.shape}."
            )

        initial_elements = list(initial_data.Elements[index])
        paired_elements = list(paired_data.Elements[index])
        initial_numbers = list(initial_data.Numbers[index])
        paired_numbers = list(paired_data.Numbers[index])
        if initial_elements != paired_elements or initial_numbers != paired_numbers:
            raise ValueError(
                f"Paired sample {index} element identities or ordering differ between "
                f"`DATA_PATH` and `{paired_name}`."
            )
        if sum(initial_numbers) != initial_coords.shape[0]:
            raise ValueError(
                f"Paired sample {index} atom count from elements does not match its coordinates."
            )

        initial_cell_shape = np.asarray(initial_data.Cells[index]).shape
        paired_cell_shape = np.asarray(paired_data.Cells[index]).shape
        if initial_cell_shape != paired_cell_shape:
            raise ValueError(
                f"Paired sample {index} cell shape differs between `DATA_PATH` and "
                f"`{paired_name}`: {initial_cell_shape} != {paired_cell_shape}."
            )

        initial_fixed_shape = np.asarray(initial_data.Fixed[index]).shape
        paired_fixed_shape = np.asarray(paired_data.Fixed[index]).shape
        if initial_fixed_shape != initial_coords.shape or paired_fixed_shape != paired_coords.shape:
            raise ValueError(
                f"Paired sample {index} fixed-mask shape must match its coordinate shape."
            )


def parse_center_input_file(path: str):
    """
    Build a CLI task runner, its paired dataset, and user model callable.

    Args:
        path: YAML input file path.

    Returns:
        A tuple of canonical task name, configured runner, and model callable.

    Raises:
        ValueError: If task configuration, output ownership, or paired
            structures are invalid.
        FileNotFoundError: If a configured input path does not exist.
    """
    config: Dict[str, Any] = load_input_config(path, require_output_root=True)
    prepare_output_root(config['OUTPUT_ROOT'])

    TASKS_TYPE = {
        'TRAIN': api.Trainer,
        'PREDICT': api.Predictor,
        'OPT': api.StructureOptimization,
        'TS': api.StructureOptimization,
        'VIB': api.VibrationAnalysis,
        'NEB': api.ClimbingImageNudgedElasticBand,
        'MD': api.MolecularDynamics,
        'CMD': api.ConstrainedMolecularDynamics,
        'MC': api.MonteCarlo,
    }
    TASKS_TYPE_ALIAS = {
        'TRAIN': 'TRAIN',
        'PREDICT': 'PREDICT',
        'PREDICTION': 'PREDICT',
        'OPT': 'OPT',
        'STRUCTURE_OPTIMIZATION': 'OPT',
        'STRUC_OPT': 'OPT',
        'DIMER': 'TS',
        'TS': 'TS',
        'VIB': 'VIB',
        'VIBRATIONAL_ANALYSIS': 'VIB',
        'NEB': 'NEB',
        'CINEB': 'NEB',
        'CI_NEB': 'NEB',
        'MD': 'MD',
        'MOLECULAR_DYNAMICS': 'MD',
        'CMD': 'CMD',
        'CONSTRAINED_MOLECULAR_DYNAMICS': 'CMD',
        'CONSTR_MD': 'CMD',
        'MC': 'MC',
        'MONTE_CARLO': 'MC',
    }

    # Section: check task
    task_type = config.get('TASK', None)
    if task_type is None:
        raise ValueError(
            'Task type is missing. '
            f'You must specify a task_type by argument `TASK` in the {path} file.\n'
            f'Available `TASK` values are:\n{", ".join(TASKS_TYPE_ALIAS.keys())} '
        )
    else:
        task_type: str
        task_type = task_type.upper()
    if task_type not in TASKS_TYPE_ALIAS:
        raise ValueError(
            'Task type is invalid. '
            f'Available `TASK` values are:\n{", ".join(TASKS_TYPE_ALIAS.keys())}'
        )
    else:
        task_type = TASKS_TYPE_ALIAS[task_type]

    # Section: load model
    model_file = config.get('MODEL_FILE', None)
    model_name = config.get('MODEL_NAME', None)
    if model_file is None:
        raise ValueError(f"`MODEL_FILE` must be specified.")
    if model_name is None:
        raise ValueError(f"`MODEL_NAME` must be specified.")
    udf_model = load_model(model_file, model_name)

    # Section: load data
    data_type = config.get('DATA_TYPE', 'POSCAR').upper()
    data_path = config.get('DATA_PATH', '')
    data_selector = config.get('DATA_NAME_SELECTOR', None)
    if data_path == '': raise ValueError(f'`DATA_PATH` is not defined.')
    data_loader_kwargs = config.get('DATA_LOADER_KWARGS', {})
    source_data: bt.Structures = load_data(data_type, data_path, data_loader_kwargs)
    selected_indices = _selected_sample_indices(source_data, data_selector)
    data = source_data[selected_indices]
    is_shuffle = config.get('IS_SHUFFLE', False)

    if task_type == 'TRAIN':
        # handle the validation set data
        val_set_path = config.get('VAL_SET_PATH', None)
        validation_ratio = config.get('VAL_SPLIT_RATIO', None)
        if val_set_path is not None:
            val_data = load_data(data_type, val_set_path, data_loader_kwargs)
            if data_selector is not None:
                val_data = val_data.select_by_sample_id(rf"{data_selector}")
        elif validation_ratio is not None:
            # check validation if ratio correct
            validation_ratio = float(validation_ratio)
            if validation_ratio >= 1. or validation_ratio <= 0.:
                raise ValueError(f"`VAL_SPLIT_RATIO` must be in interval (0, 1), but got {validation_ratio}.")
            n_val = int(validation_ratio * len(data))
            if n_val >= len(data) or n_val <= 0:
                raise ValueError(
                    f"Unreasonable `VAL_SPLIT_RATIO` value `{validation_ratio}`, "
                    f"which will cause {n_val} validation samples and {len(data) - n_val} training samples."
                )
            val_data = data[:n_val]
            data = data[n_val:]
        else:
            raise ValueError(f"THERE IS NO VALIDATION DATA SPECIFIED. TRAINING MAY BE MEANINGLESS.")

        data_list = bt.preprocessing.CreatePygData(1).feat2data_list(data, n_core=1)
        val_data_list = bt.preprocessing.CreatePygData(1).feat2data_list(val_data, n_core=1)

        trn_ener = [data[atm.idx].Energies[0] for atm in data_list]
        if data.Forces is None:
            trn_forc = None
        else:
            trn_forc = [data[atm.idx].Forces[0] for atm in data_list]
        train_data = {'data': data_list, 'labels': {'energy': trn_ener, 'forces': trn_forc}}
        val_ener = [val_data[atm.idx].Energies[0] for atm in val_data_list]
        if val_data.Forces is None:
            val_forc = None
        else:
            val_forc = [val_data[atm.idx].Forces[0] for atm in val_data_list]
        valid_data = {'data': val_data_list, 'labels': {'energy': val_ener, 'forces': val_forc}}
        dataset_args = (train_data, valid_data)

    elif task_type == 'NEB' or task_type == 'CMD':  # They use ISFSDataLoader
        if is_shuffle:
            raise ValueError(f"`IS_SHUFFLE` must be false for paired task `{task_type}`.")
        # handle the final state configuration data
        fs_data_path = config.get('FSDATA_PATH', None)
        if fs_data_path is not None:
            fs_data = load_data(data_type, fs_data_path, data_loader_kwargs)
        else:
            raise ValueError(f"`FSDATA_PATH` is not defined. For TASK `{task_type}`, "
                             f"you must specify the final-state-configuration data path by `FSDATA_PATH`.")
        _validate_paired_structures(source_data, fs_data, 'FSDATA_PATH')
        fs_data = fs_data[selected_indices]
        is_data_list = bt.preprocessing.CreatePygData(1).feat2data_list(data, n_core=1)
        fs_data_list = bt.preprocessing.CreatePygData(1).feat2data_list(fs_data, n_core=1)
        run_data = {'dataIS': is_data_list, 'dataFS': fs_data_list}
        dataset_args = (run_data,)

    elif task_type == 'TS':  # Need a dimer initial guess
        # handle the final state configuration data
        disp_data_path = config.get('DISPDATA_PATH', None)
        displace_flag = config.get('TRANSITION_STATE', None)
        if displace_flag is None:
            raise ValueError(
                f"Transition state search is required but input arguments not provided. "
                f"Please set `TRANSITION_STATE` section in the config file. "
            )
        else:
            displace_flag = displace_flag.get('X_DIFF_ATTR', None)

        is_data_list = bt.preprocessing.CreatePygData(1).feat2data_list(data, n_core=1)

        if displace_flag is not None:  # IF DISPDATA_PATH is given, canonically use data read from DISPDATA_PATH
            if disp_data_path is not None:
                disp_data = load_data(data_type, disp_data_path, data_loader_kwargs)
                _validate_paired_structures(source_data, disp_data, 'DISPDATA_PATH')
                disp_data = disp_data[selected_indices]
                for i, dat in enumerate(is_data_list):
                    disp_tensor = th.as_tensor(disp_data.Coords[i])
                    setattr(dat, str(displace_flag), disp_tensor)
            else:
                fs_data_path = config.get('FSDATA_PATH', None)  # ELIF FSDATA_PATH, use interpolated middle point of is/fs conf.
                if fs_data_path is not None:
                    fs_data = load_data(data_type, fs_data_path, data_loader_kwargs)
                    _validate_paired_structures(source_data, fs_data, 'FSDATA_PATH')
                    fs_data = fs_data[selected_indices]
                    # convert to displacement by interpolation
                    for i, dat in enumerate(is_data_list):
                        disp_coo, diff = direction_for_finite_diff(dat.pos, fs_data.Coords[i], dat.fixed)
                        setattr(dat, 'pos', disp_coo)
                        setattr(dat, str(displace_flag), diff)
                else:
                    warnings.warn(f"`X_DIFF_ATTR` is set but no displacement is provided. `X_DIFF_ATTR` will be ignored.")
                    for i, dat in enumerate(is_data_list):
                        setattr(dat, str(displace_flag), None)
        run_data = {'data': is_data_list, 'labels': None}
        dataset_args = (run_data,)

    else:
        data_list = bt.preprocessing.CreatePygData(1).feat2data_list(data, n_core=1)
        run_data = {'data': data_list, 'labels': None}
        dataset_args = (run_data,)

    # dataloader
    if task_type == 'NEB' or task_type == 'CMD':
        dataloader = ISFSPyGDataLoader
    else:
        dataloader = PyGDataLoader

    # set runner
    runner = TASKS_TYPE[task_type](path, 'pyg')
    runner.set_dataset(*dataset_args, )  # type: ignore
    dataloader_config = {} if task_type in {'NEB', 'CMD'} else {'shuffle': is_shuffle}
    runner.set_dataloader(dataloader, dataloader_config)
    # set constraints function
    if task_type == 'CMD':
        md_config = config.get('MD', {})
        constr_file = md_config.get('CONSTRAINTS_FILE', config.get('CONSTRAINTS_FILE', None))
        constr_name = md_config.get('CONSTRAINTS_FUNC', config.get('CONSTRAINTS_FUNC', None))
        if 'CONSTRAINTS_FILE' in config or 'CONSTRAINTS_FUNC' in config:
            warnings.warn(
                "Top-level `CONSTRAINTS_FILE` and `CONSTRAINTS_FUNC` are deprecated; "
                "place them under `MD`.",
                FutureWarning,
                stacklevel=2,
            )
        if constr_file is None:
            raise ValueError(f"`CONSTRAINTS_FILE` must be specified.")
        if constr_name is None:
            raise ValueError(f"`CONSTRAINTS_FUNC` must be specified.")
        constr_func = load_model(constr_file, constr_name)
        runner.set_constr_func(constr_func)

    return task_type, runner, udf_model


def load_data(data_type, data_path, data_loader_kwargs):
    """
    Load structures using a CLI data-type name.

    Args:
        data_type: External structure format or built-in trajectory type.
        data_path: File or directory passed to the selected reader.
        data_loader_kwargs: Reader-specific keyword arguments.

    Returns:
        A populated ``BatchStructures``-compatible collection.

    Raises:
        ValueError: If ``data_type`` is unsupported.

    """
    DATA_TYPE = {
        'external': {
            'POSCAR': load_files.POSCARs2Feat,
            'OUTCAR': load_files.OUTCAR2Feat,
            'CIF': load_files.Cif2Feat,
            'ASE_TRAJ': load_files.ASETraj2Feat,
        },
        'buildin': {
            'BS': bt.load,
            'OPT': bt.io.read_opt_structures,
            'MD': bt.io.read_md_traj,
            'MC': bt.io.read_mc_traj,
        }
    }

    if data_type in DATA_TYPE['external']:
        data = DATA_TYPE['external'][data_type](data_path)
        data.read(**data_loader_kwargs)
    elif data_type in DATA_TYPE['buildin']:
        data_reader = DATA_TYPE['buildin'][data_type]
        data = data_reader(data_path, **data_loader_kwargs)
    else:
        raise ValueError(
            f'Data type "{data_type}" is invalid. '
            f'Available `DATA_TYPE` values are:\n{', '.join(DATA_TYPE['external'].keys())}, {', '.join(DATA_TYPE['buildin'].keys())}'
        )

    return data


def load_model(func_file_path, func_name) -> Callable:
    """
    Load a user-defined callable from a Python source file.

    Args:
        func_file_path: Python file containing the target object.
        func_name: Name of the callable to load from that module.

    Returns:
        The requested callable object.

    Raises:
        FileNotFoundError: If ``func_file_path`` does not exist.
        ImportError: If Python cannot create an executable loader for the file.
        AttributeError: If ``func_name`` is absent from the loaded module.
        TypeError: If the named object is not callable.
    """
    if not os.path.isfile(func_file_path):
        raise FileNotFoundError(f"File {func_file_path} not found.")

    realpath = os.path.realpath(func_file_path, strict=True)
    enc_path = hashlib.md5(realpath.encode()).hexdigest()  # hash the file name
    module_name = f"udf_{enc_path}"
    spec = importlib.util.spec_from_file_location(module_name, func_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {func_file_path}: no module loader is available.")

    module = importlib.util.module_from_spec(spec)

    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise

    # get target function
    udf = getattr(module, func_name, None)
    if udf is None:
        raise AttributeError(f"Function/Model {func_name} not found in {func_file_path}.")
    if not callable(udf):
        raise TypeError(f"Function/Model {func_name} in {func_file_path} must be callable.")

    return udf

def launch_task(inp):
    """Launch the task configured by one YAML input file.

    Args:
        inp: YAML input file path.

    Returns:
        None.

    Raises:
        ValueError: If task or data configuration is invalid.
        FileNotFoundError: If an input, model, or data path is missing.
    """
    task_type, runner, udf_model = parse_center_input_file(inp)
    if task_type == 'TRAIN':
        runner.train(udf_model)
    elif task_type == 'TS':
        runner.ts(udf_model)
    elif task_type == 'OPT':
        runner.relax(udf_model)
    else:
        runner.run(udf_model)

def main():
    """Run the command-line interface selected by process arguments.

    With no arguments this enters the interactive CLI. Otherwise ``-i`` runs
    one task and ``-c`` converts structures between supported formats.

    Returns:
        None.

    Raises:
        OSError: If a requested output cannot be backed up or opened.
        ValueError: If task configuration or conversion paths are invalid.
    """
    parser = argparse.ArgumentParser(
        description=f'BUCToolkit MAIN PROGRAM INTERFACES\n{generate_display_art()}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('-i', '--input', help='The path to input file.', default=None)
    parser.add_argument(
        '-o', '--output',
        help='The path to output file. It will change the stdout. One can also cleanly redirect output by setting '
             '`REDIRECT: true`, `OUTPUT_PATH: your/path/to/output/log/`, and `OUTPUT_POSTFIX: your_log_postfix` '
             'in the input file. If so, this argument will be ignored.',
        default=None,
        type=str,
    )
    group.add_argument(
        '-c', '--convert', nargs=4,
        help='Convert the format of structure files to the output file. Need four values. '
             'Usage: buctoolkit -c `$input_type` `$input_path` `$output_type` `$output_path`; '
             '`$input_path` can be one of "bs", "md", "mc", "opt", "outcar", "poscar", "cif", and "ase_traj"; '
             '`$output_type` can be one of "poscar", "cif", "xyz", "bs".',
        default=None,
        type=str,
        metavar=('input_type'.upper(), 'input_path'.upper(), 'output_type'.upper(), 'output_path'.upper())
    )

    opened_file = None
    original_stdout = sys.stdout
    try:
        if len(sys.argv) == 1:  # Enter the interactive CLI
            bt.cli.run_base_cli()
        else:  # otherwise directly run in one-line command
            args = parser.parse_args()
            if args.convert is not None and args.output is not None:
                parser.error('`-o/--output` cannot be used with `-c/--convert`.')
            if args.input is not None:
                if args.output is not None:
                    backup_path = backup_output(args.output, 'file')
                    if backup_path is not None:
                        warnings.warn(
                            f'Output file `{args.output}` already exists. Moved it to `{backup_path}`.',
                            stacklevel=2,
                        )
                    opened_file = open(args.output, 'w')
                    sys.stdout = opened_file
                launch_task(args.input)
            elif args.convert is not None:
                inp = args.convert
                main_convert(*inp)
    finally:
        if opened_file is not None:
            if not opened_file.closed:
                opened_file.close()
        sys.stdout = original_stdout

if __name__ == '__main__':
    main()
