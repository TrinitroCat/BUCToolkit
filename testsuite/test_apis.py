""" API unit tests: Trainer and Predictor with GNN-LJ-EAM model on OUTCAR data. """

#  Copyright (c) 2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: test_apis.py
#  Environment: Python 3.12

import os
import tarfile
import shutil
import random
from typing import List

import numpy as np
import torch as th

from BUCToolkit.io import OUTCAR2Feat
from BUCToolkit.Preprocessing.preprocessing import CreatePygData
from BUCToolkit.api.DataLoaders import PyGDataLoader, ISFSPyGDataLoader
from BUCToolkit.api.Trainer import Trainer
from BUCToolkit.api.Predictor import Predictor
from BUCToolkit.api.StructureOptimization import StructureOptimization
from BUCToolkit.api.MolecularDynamics import MolecularDynamics
from BUCToolkit.api.MonteCarlo import MonteCarlo
from BUCToolkit.api.VibrationAnalysis import VibrationAnalysis
from BUCToolkit.api.ConstrainedMolecularDynamics import ConstrainedMolecularDynamics
from BUCToolkit.api.NEB import ClimbingImageNudgedElasticBand

_HERE = os.path.dirname(os.path.abspath(__file__))
_TGZ_PATH = os.path.join(_HERE, 'test_structures', 'OUTCARs', 'outcars.tgz')


def _untar(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(_TGZ_PATH, 'r:gz') as tf:
        tf.extractall(path=target_dir, filter='data')


def _write_train_inp(path, batch_size, device, epochs, output_dir, data_path):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str TRAIN
START: !!int 0
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str api_test
REDIRECT: !!bool true

TRAIN:
  EPOCH: !!int {epochs}
  SAVE_CHK: !!bool true
  CHK_SAVE_PATH: !!str {output_dir}
  CHK_SAVE_POSTFIX: !!str test_model_chk
  OPTIM: !!str AdamW
  OPTIM_CONFIG:
    lr: 1.e-3
  LOSS: Energy_Force_Loss
  LOSS_CONFIG:
    coeff_F: 1.0
  METRICS: ['E_MAE', 'F_MAE']
  DATA_LOADER_KWARGS:
    shuffle: True

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def _write_predict_inp(path, batch_size, device, output_dir, data_path):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str PREDICT
START: !!int 2
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str pred_test
REDIRECT: !!bool true
SAVE_PREDICTIONS: !!bool true
PREDICTIONS_SAVE_FILE: {output_dir}/test_pred.db
LOAD_CHK_FILE_PATH: '{output_dir}/best_checkpoint_test_model_chk.pt'

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def _write_opt_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str OPT
START: !!int 2
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str opt_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

RELAXATION:
  ALGO: !!str FIRE
  OPTIMIZER: !!str FIRE
  STEPLENGTH: !!float 0.05
  F_THRESHOLD: !!float 0.05
  MAXITER: !!int 100

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def _write_md_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str MD
START: !!int 2
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str md_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

MD:
  ENSEMBLE: !!str NVT
  THERMOSTAT: !!str CSVR
  THERMOSTAT_CONFIG:
    TIME_CONST: !!float 120
  TIME_STEP: !!float 0.5
  MAX_STEP: !!int 200
  T_INIT: !!float 500.0
  OUTPUT_COORDS_PER_STEP: !!int 5
  MOVE_TO_CENTER_FREQ: !!int 5

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")



def _write_mc_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str MC
START: !!int 1
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str mc_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

MC:
  ENSEMBLE: !!str NVT
  TYPE: !!str METROPOLIS
  MAXITER: !!int 200
  T_INIT: !!float 500.0
  T_SCHEME: !!str constant
  T_SCHEME_PARAM: !!float 0.0
  COORDINATE_UPDATE_PARAM: !!float 0.2

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def _write_vib_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str VIB
START: !!int 1
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str vib_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

VIBRATION:
  METHOD: !!str Coord
  BLOCK_SIZE: !!int 6
  DELTA: !!float 0.01
  SAVE_HESSIAN: !!bool true

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def _write_cmd_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str CMD
START: !!int 1
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str cmd_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

MD:
  ENSEMBLE: !!str NVT
  THERMOSTAT: !!str CSVR
  THERMOSTAT_CONFIG:
    TIME_CONST: !!float 120
  CONSTR_MD_SCHEME: !!str SLOW_GROWTH
  NIMAGE: !!int 5
  TIME_STEP: !!float 0.5
  MAX_STEP: !!int 100
  T_INIT: !!float 500.0
  OUTPUT_COORDS_PER_STEP: !!int 5
  MOVE_TO_CENTER_FREQ: !!int -1

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")

def _write_blum_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str CMD
START: !!int 1
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str cmd_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

MD:
  ENSEMBLE: !!str NVT
  THERMOSTAT: !!str CSVR
  THERMOSTAT_CONFIG:
    TIME_CONST: !!float 120
  CONSTR_MD_SCHEME: !!str BLUE_MOON
  NIMAGE: !!int 5
  TIME_STEP: !!float 0.5
  MAX_STEP: !!int 100
  T_INIT: !!float 500.0
  OUTPUT_COORDS_PER_STEP: !!int 5
  MOVE_TO_CENTER_FREQ: !!int 10

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def _write_neb_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str NEB
START: !!int 1
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str neb_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

NEB:
  ALGO: !!str CI-NEB
  N_IMAGES: !!int 3
  SPRING_CONST: !!float 1.0
  OPTIMIZER: !!str FIRE
  STEPLENGTH: !!float 0.2
  MAXITER: !!int 10
  E_THRESHOLD: !!float 0.01
  F_THRESHOLD: !!float 0.05

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def _write_ts_inp(path, batch_size, device, output_dir, chk_dir):
    with open(path, 'w') as f:
        f.write(f"""TASK: !!str OPT
START: !!int 2
VERBOSE: !!int 1
DEVICE: !!str '{device}'
BATCH_SIZE: !!int {batch_size}
OUTPUT_PATH: !!str {output_dir}
OUTPUT_POSTFIX: !!str ts_test
REDIRECT: !!bool true
LOAD_CHK_FILE_PATH: '{chk_dir}/best_checkpoint_test_model_chk.pt'

TRANSITION_STATE:
  ALGO: !!str DIMER
  OPTIMIZER: !!str FIRE
  STEPLENGTH: !!float 0.05
  MAXITER: !!int 10
  F_THRESHOLD: !!float 0.05

MODEL_NAME: !!str GNNLJDirectionalEAM
MODEL_CONFIG:
  max_atomic_number: 100
  embedding_dim: 8
  hidden_dim: 16
  cutoff: 6.0
  eam_hidden_dim: 16
""")


def run_api_tests(tmp_base: str = '/dev/shm') -> List[str]:
    r"""Run Trainer + Predictor tests. Returns error messages."""
    errors = []
    outcar_dir = os.path.join(tmp_base, 'api_test_outcars')
    train_inp = os.path.join(tmp_base, 'api_test_train.inp')
    predict_inp = os.path.join(tmp_base, 'api_test_predict.inp')
    opt_inp = os.path.join(tmp_base, 'api_test_opt.inp')
    md_inp = os.path.join(tmp_base, 'api_test_md.inp')
    opt_log_dir = os.path.join(tmp_base, 'api_test_opt_logs')
    md_log_dir = os.path.join(tmp_base, 'api_test_md_logs')
    log_dir = os.path.join(tmp_base, 'api_test_logs')
    mc_inp = os.path.join(tmp_base, 'api_test_mc.inp')
    vib_inp = os.path.join(tmp_base, 'api_test_vib.inp')
    cmd_inp = os.path.join(tmp_base, 'api_test_cmd.inp')
    blum_inp = os.path.join(tmp_base, 'api_test_blum.inp')
    neb_inp = os.path.join(tmp_base, 'api_test_neb.inp')
    ts_inp = os.path.join(tmp_base, 'api_test_ts.inp')
    mc_log_dir = os.path.join(tmp_base, 'api_test_mc_logs')
    vib_log_dir = os.path.join(tmp_base, 'api_test_vib_logs')
    cmd_log_dir = os.path.join(tmp_base, 'api_test_cmd_logs')
    blum_log_dir = os.path.join(tmp_base, 'api_test_blum_logs')
    neb_log_dir = os.path.join(tmp_base, 'api_test_neb_logs')
    ts_log_dir = os.path.join(tmp_base, 'api_test_ts_logs')

    extra_dirs = [mc_log_dir, vib_log_dir, cmd_log_dir, blum_log_dir, neb_log_dir, ts_log_dir]
    for d in [outcar_dir, log_dir, opt_log_dir, md_log_dir] + extra_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    for f in [train_inp, predict_inp, opt_inp, md_inp, mc_inp, vib_inp, cmd_inp, blum_inp, neb_inp, ts_inp]:
        if os.path.isfile(f):
            os.remove(f)

    try:
        # ----------------------------------------------------------------
        # Step 1: Load OUTCAR data
        # ----------------------------------------------------------------
        _untar(outcar_dir)
        feat = OUTCAR2Feat(outcar_dir, verbose=0)
        feat.read()

        # Use at most 50 structures for speed
        n_total = len(feat)
        n_use = min(n_total, 50)
        indices = sorted(random.sample(range(n_total), n_use))
        # Use all as both train and valid for API smoke test
        data_list = CreatePygData(1).feat2data_list(feat, n_core=1)
        data_list = [data_list[i] for i in indices]

        energies = [float(feat[i].Energies[0]) for i in indices]
        forces = [np.array(feat[i].Forces[0], dtype=np.float32) for i in indices]

        train_data = {
            'data': data_list,
            'labels': {'energy': energies, 'forces': forces},
        }
        # use same data as validation (smoke test)
        valid_data = {
            'data': data_list,
            'labels': {'energy': energies, 'forces': forces},
        }

        # ----------------------------------------------------------------
        # Step 2: Trainer test
        # ----------------------------------------------------------------
        device = 'cuda:0' if th.cuda.is_available() else 'cpu'
        _write_train_inp(train_inp, batch_size=4, device=device, epochs=20,
                         output_dir=log_dir, data_path='')
        trainer = Trainer(train_inp)
        trainer.set_dataset(train_data, valid_data)
        trainer.set_dataloader(PyGDataLoader, {'shuffle': True})

        from _toy_models import GNNLJDirectionalEAM
        trainer.train(GNNLJDirectionalEAM)

        print(f'  Trainer: 2 epoch on {n_use} structures OK')

        # ----------------------------------------------------------------
        # Step 3: Predictor test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM

        # Write predict config
        _write_predict_inp(predict_inp, batch_size=4, device=device,
                           output_dir=log_dir, data_path='')
        predictor = Predictor(predict_inp)
        predictor.set_dataset(
            {'data': data_list[:12], 'labels': {'energy': energies[:12], 'forces': forces[:12]}},
        )
        predictor.set_dataloader(PyGDataLoader, {'shuffle': False})
        predictor.predict(GNNLJDirectionalEAM)

        print(f'  Predictor: {min(12, n_use)} structures OK')

        # ----------------------------------------------------------------
        # Step 4: Structure Optimization test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        opt_data = {'data': data_list[:12]}  # opt only needs 'data', no labels
        _write_opt_inp(opt_inp, batch_size=4, device=device, output_dir=opt_log_dir, chk_dir=log_dir)
        optimizer = StructureOptimization(opt_inp)
        optimizer.set_dataset(opt_data)
        optimizer.set_dataloader(PyGDataLoader)
        optimizer.relax(GNNLJDirectionalEAM)
        print(f'  StructureOptimization: {min(12, n_use)} structures OK')

        # ----------------------------------------------------------------
        # Step 5: Molecular Dynamics test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        md_data = {'data': data_list[:12]}
        _write_md_inp(md_inp, batch_size=4, device=device, output_dir=md_log_dir, chk_dir=log_dir)
        runner = MolecularDynamics(md_inp)
        runner.set_dataset(md_data)
        runner.set_dataloader(PyGDataLoader)
        runner.run(GNNLJDirectionalEAM)
        print(f'  MolecularDynamics: {min(12, n_use)} structures OK')

        # ----------------------------------------------------------------
        # Step 6: Monte Carlo test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        mc_data = {'data': data_list[:12]}
        _write_mc_inp(mc_inp, batch_size=4, device=device, output_dir=mc_log_dir, chk_dir=log_dir)
        runner_mc = MonteCarlo(mc_inp)
        runner_mc.set_dataset(mc_data)
        runner_mc.set_dataloader(PyGDataLoader)
        runner_mc.run(GNNLJDirectionalEAM)
        print(f'  MonteCarlo: {min(12, n_use)} structures OK')

        # ----------------------------------------------------------------
        # Step 7: Vibration Analysis test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        vib_data = {'data': data_list[:12]}
        _write_vib_inp(vib_inp, batch_size=1, device=device, output_dir=vib_log_dir, chk_dir=log_dir)
        runner_vib = VibrationAnalysis(vib_inp)
        runner_vib.set_dataset(vib_data)
        runner_vib.set_dataloader(PyGDataLoader, {'shuffle': False})
        runner_vib.run(GNNLJDirectionalEAM)
        print(f'  VibrationAnalysis: {min(12, n_use)} structures OK')

        # ----------------------------------------------------------------
        # Step 8: Constrained Molecular Dynamics test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        cmd_data = {'data': data_list[:4]}
        _write_cmd_inp(cmd_inp, batch_size=1, device=device, output_dir=cmd_log_dir, chk_dir=log_dir)
        runner_cmd = ConstrainedMolecularDynamics(cmd_inp)

        # SLOW_GROWTH needs a constraint function
        # Use a simple distance constraint
        def constr_func(X):
            return (X[[0]] - X[[1]]).norm()
        runner_cmd.set_constr_func(constr_func)
        #runner_cmd.set_constr_val(constr_func(data_list[0].pos))

        runner_cmd.set_dataset(cmd_data)
        runner_cmd.set_dataloader(PyGDataLoader)
        runner_cmd.run(GNNLJDirectionalEAM)
        print(f'  ConstrainedMD (Slow-growth): {min(12, n_use)} structures OK')

        # ----------------------------------------------------------------
        # Step 9: Climbing-image NEB test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        # NEB needs pairs of start/end images
        # Use first 2 samples as start/end pair
        # find the same sample pair:
        feat_neb = OUTCAR2Feat(outcar_dir, verbose=0)
        feat_neb.read(os.listdir(outcar_dir)[0:1])
        data_neb_list = CreatePygData(0).feat2data_list(feat_neb, n_core=1)

        neb_data = {'dataIS': data_neb_list[0:3], 'dataFS': data_neb_list[-3:]}
        _write_neb_inp(neb_inp, batch_size=1, device=device, output_dir=neb_log_dir, chk_dir=log_dir)
        runner_neb = ClimbingImageNudgedElasticBand(neb_inp)
        runner_neb.set_dataset(neb_data)
        runner_neb.set_dataloader(ISFSPyGDataLoader, )
        runner_neb.run(GNNLJDirectionalEAM)
        print(f'  NEB: 2 bands OK')

        # ----------------------------------------------------------------
        # Step 10: Blue-Moon ensemble test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        blum_data = {'dataIS': data_neb_list[0:3], 'dataFS': data_neb_list[-3:]}
        _write_blum_inp(blum_inp, batch_size=1, device=device, output_dir=blum_log_dir, chk_dir=log_dir)
        runner_cmd = ConstrainedMolecularDynamics(blum_inp)

        # SLOW_GROWTH needs a constraint function
        # Use a simple distance constraint
        def constr_func(X):
            return (X[[0]] - X[[1]]).norm()

        runner_cmd.set_constr_func(constr_func)
        # runner_cmd.set_constr_val(constr_func(data_list[0].pos))

        runner_cmd.set_dataset(blum_data)
        runner_cmd.set_dataloader(ISFSPyGDataLoader)
        runner_cmd.run(GNNLJDirectionalEAM)
        print(f'  ConstrainedMD (Blue-Moon): {min(12, n_use)} structures OK')

        # ----------------------------------------------------------------
        # Step 11: Transition State search test
        # ----------------------------------------------------------------
        from _toy_models import GNNLJDirectionalEAM
        ts_data = {'data': data_list[:12]}
        _write_ts_inp(ts_inp, batch_size=4, device=device, output_dir=ts_log_dir, chk_dir=log_dir)
        runner_ts = StructureOptimization(ts_inp)
        runner_ts.set_dataset(ts_data)
        runner_ts.set_dataloader(PyGDataLoader)
        runner_ts.run(GNNLJDirectionalEAM, mode='ts')
        print(f'  TS(DIMER): {min(12, n_use)} structures OK')

    except Exception as e:
        import traceback
        errors.append(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
    finally:
        #for d in [outcar_dir, log_dir]:
        #    if os.path.isdir(d):
        #        shutil.rmtree(d, ignore_errors=True)
        #for f in [train_inp, predict_inp]:
        #    if os.path.isfile(f):
        #        os.remove(f)
        pass

    return errors
