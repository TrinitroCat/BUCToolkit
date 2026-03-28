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
import warnings
from typing import Dict, Any, Callable
import importlib.util
import hashlib

import yaml

import BUCToolkit as bt
from BUCToolkit.cli.print_logo import generate_display_art
import BUCToolkit.api as api
from BUCToolkit.api.DataLoaders import PyGDataLoader, ISFSPyGDataLoader
import BUCToolkit.Preprocessing.load_files as load_files


def parse_center_input_file(path: str):
    """
    Parse the input file with the highest level that control the overall task.
    Args:
        path: path of the input file.

    Returns:
        runner: the api of the task. Use `runner.run(model)` to run the task.

    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, 'r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)

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
    model_path = config.get('MODEL_PATH', None)
    model_name = config.get('MODEL_NAME', None)
    if model_path is None:
        raise ValueError(f"`MODEL_PATH` must be specified.")
    if model_name is None:
        raise ValueError(f"`MODEL_NAME` must be specified.")
    udf_model = load_model(model_path, model_name)

    # Section: load data
    data_type = config.get('DATA_TYPE', 'POSCAR').upper()
    data_path = config.get('DATA_PATH', '')
    data_selector = config.get('DATA_NAME_SELECTOR', None)
    if data_path == '': raise ValueError(f'`DATA_PATH` is not defined.')
    data_loader_kwargs = config.get('DATA_LOADER_KWARGS', {})
    data: bt.Structures = load_data(data_type, data_path, data_loader_kwargs)
    if data_selector is not None:
        data = data.select_by_sample_id(rf"{data_selector}")
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
                    f"which causes {n_val} validation samples and {len(data) - n_val} training samples."
                )
            val_data = data[:n_val]
            data = data[n_val:]
        else:
            raise ValueError(f"THERE IS NO VALIDATION DATA SPECIFIED. TRAINING MAY BE MEANINGLESS.")

        data_list = bt.preprocessing.CreatePygData(1).feat2data_list(data, n_core=1)
        val_data_list = bt.preprocessing.CreatePygData(1).feat2data_list(val_data, n_core=1)

        trn_ener = [data[atm.idx].Energies[0] for atm in data_list]
        trn_forc = [data[atm.idx].Forces[0] for atm in data_list]
        train_data = {'data': data_list, 'labels': {'energy': trn_ener, 'forces': trn_forc}}
        val_ener = [val_data[atm.idx].Energies[0] for atm in val_data_list]
        val_forc = [val_data[atm.idx].Forces[0] for atm in val_data_list]
        valid_data = {'data': val_data_list, 'labels': {'energy': val_ener, 'forces': val_forc}}
        dataset_args = (train_data, valid_data)
        # TODO: ADD layerwise train scheme
        # TODO: ADD lr_scheduler

    elif task_type == 'NEB' or task_type == 'CMD':  # They use ISFSDataLoader
        # handle the final state configuration data
        fs_data_path = config.get('FSDATA_PATH', None)
        if fs_data_path is not None:
            fs_data = load_data(data_type, fs_data_path, data_loader_kwargs)
        else:
            raise ValueError(f"`FSDATA_PATH` is not defined. For TASK `{task_type}`, "
                             f"you must specify the final-state-configuration data path by `FSDATA_PATH`.")
        is_data_list = bt.preprocessing.CreatePygData(1).feat2data_list(data, n_core=1)
        fs_data_list = bt.preprocessing.CreatePygData(1).feat2data_list(fs_data, n_core=1)
        run_data = {'dataIS': is_data_list, 'dataFS': fs_data_list}
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
    runner.set_dataloader(dataloader, {'shuffle': is_shuffle})
    # set constraints function
    if task_type == 'CMD':
        constr_file = config.get('CONSTRAINTS_FILE', None)
        constr_name = config.get('CONSTRAINTS_FUNC', None)
        if constr_file is None:
            raise ValueError(f"`CONSTRAINTS_FILE` must be specified.")
        if constr_name is None:
            raise ValueError(f"`CONSTRAINTS_FUNC` must be specified.")
        constr_func = load_model(constr_file, constr_name)
        runner.set_constr_func(constr_func)

    return task_type, runner, udf_model


def load_data(data_type, data_path, data_loader_kwargs):
    """
    A data loader
    Returns: BatchStructure formated data

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
            'OPT': bt.read_opt_structures,
            'MD': bt.read_md_traj,
        }
    }

    if data_type in DATA_TYPE['external']:
        data_reader = DATA_TYPE['external'][data_type](data_path)
        data = data_reader.read(**data_loader_kwargs)
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
    Load user defined model.
    Args:
        func_file_path: user-defined function file path
        func_name: user-defined function name

    Returns:

    """
    if not os.path.isfile(func_file_path):
        raise FileNotFoundError(f"File {func_file_path} not found.")

    realpath = os.path.realpath(func_file_path, strict=True)
    enc_path = hashlib.md5(realpath.encode()).hexdigest()  # hash the file name
    module_name = f"udf_{enc_path}"
    spec = importlib.util.spec_from_file_location(module_name, func_file_path)
    if spec is None:
        raise ImportError(f"Cannot load module from {func_file_path}.")

    module = importlib.util.module_from_spec(spec)

    # run
    spec.loader.exec_module(module)

    # get target function
    udf = getattr(module, func_name, None)
    if udf is None:
        raise AttributeError(f"Function/Model {func_name} not found in {func_file_path}.")

    return udf


def main():
    """
    Main program interface
    Usage Convention:
        buctoolkit -i xxx.inp -o xxx.oup
        bctk
    """
    parser = argparse.ArgumentParser(
        description=f'BUCToolkit MAIN PROGRAM INTERFACES\n{generate_display_art()}',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--input', help='The path to input file.', required=True)
    parser.add_argument(
        '-o', '--output',
        help='The path to output file. It will change the stdout. One can also cleanly redirect output by setting '
             '`REDIRECT: true`, `OUTPUT_PATH: your/path/to/output/log/`, and `OUTPUT_POSTFIX: your_log_postfix` '
             'in the input file. If so, this argument will be ignored.',
        default=None,
        type=str,
    )

    opened_file = None
    try:
        if len(sys.argv) == 1:  # Enter the interactive CLI
            bt.cli.run_base_cli()
        else:  # otherwise directly run in one-line command
            args = parser.parse_args()
            if args.output is not None:
                _rename = args.output
                while os.path.exists(_rename):
                    _rename += '_1'
                warnings.warn(f'Output file `{args.output}` already exists. Rename to {_rename}.')
                opened_file = open(args.output, 'w')
                sys.stdout = opened_file
            task_type, runner, udf_model = parse_center_input_file(args.input)
            if task_type == 'TS':
                runner.ts(udf_model)
            elif task_type == 'OPT':
                runner.relax(udf_model)
            else:
                runner.run(udf_model)

    finally:
        if opened_file is not None:
            if not opened_file.closed:
                opened_file.close()
        sys.stdout = sys.__stdout__

if __name__ == '__main__':
    main()
