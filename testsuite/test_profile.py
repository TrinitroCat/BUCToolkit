#  Copyright (c) 2026.1.27, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: fast_test.py
#  Environment: Python 3.12
import time
import unittest
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # add BUCToolkit root to path
sys.path.insert(0, os.path.dirname(__file__))

import torch as th
import numpy as np

from BUCToolkit.cli.main import launch_task
from BUCToolkit.BatchStructures import read_freq, read_md_traj, read_opt_structures, read_mc_traj
from BUCToolkit.api._io import _Model_Wrapper_pyg
from BUCToolkit.BatchMD import NVE, NVT
from BUCToolkit.BatchMD import ConstrNVE, ConstrNVT
from BUCToolkit.BatchOptim import QN, CG, FIRE, Frequency
from BUCToolkit.BatchMC import MMC
from BUCToolkit.utils.AtomicNumber2Properties import MASS

from _toy_harmonic_potential import (HarmonicLatticePotential, SimpleSpringPotential, LennardJonesCluster, DoubleWellPotential,
                                               MullerBrownPotential, FreeParticles, build_cubic_lattice_batch, build_cubic_lattice_data)
from BUCToolkit.api._io import PygBatchUpdater
from _test_config import get_test_section, load_test_config

INPUT_PATH = './inputs4test/'
HERE = os.path.dirname(os.path.abspath(__file__))

class FastTest(unittest.TestCase):

    TEST_MODE = 'fast'

    def setUp(self):
        """Prepare the shared model fixture and load the selected YAML mode.

        Returns:
            None.
        """
        self.test_config = load_test_config(self.TEST_MODE)
        # data
        ATOMS = [8, 5, 10]
        data = build_cubic_lattice_batch(ATOMS, 1.3, 0.05)
        ELEM = ['Fe', 'Al', 'Pd']
        DOF_reduce = 0
        self.MASSES = [MASS[_] for _ in ELEM]
        self.elem_list = [[]]
        for i, el in enumerate(ELEM):
            self.elem_list[0].extend([el] * ATOMS[i]**3)
        self.masses_list = [[MASS['Fe']] * ATOMS[0] ** 3, [MASS['Al']] * ATOMS[1] ** 3, [MASS['Pd']] * ATOMS[2] ** 3]
        self.DOF_vib = [3 * ATOMS[0] ** 3 - DOF_reduce, 3 * ATOMS[1] ** 3 - DOF_reduce, 3 * ATOMS[2] ** 3 - DOF_reduce]
        self.N = [_**3 for _ in ATOMS]
        #raw_model = HarmonicLatticePotential(100., 1.)
        raw_model = SimpleSpringPotential(data.pos0, 10., )
        #raw_model = LennardJonesCluster()
        #raw_model = DoubleWellPotential()
        self.model_test = _Model_Wrapper_pyg(raw_model)
        self.data = data
        self.REQUIRE_GRAD = False

        # io
        if sys.platform.startswith("win"):
            print("WARNING: Detected Windows os. BUCToolkit has not fully test on Windows yet, be careful!")
            self.out_pt = './'
        elif sys.platform.startswith("linux"):
            self.out_pt = '/dev/shm/BUCToolkit/'
            os.makedirs(self.out_pt, exist_ok=True)
        else:
            print(f"WARNING: Detected OS of {sys.platform}. BUCToolkit has not been tested on this platform. Please be careful!")
            self.out_pt = './'

        for ptt in ['results', 'logs']:
            if not os.path.exists(f'{self.out_pt}{ptt}'):
                os.makedirs(f'{self.out_pt}{ptt}')
            elif os.path.isfile(f'{self.out_pt}{ptt}'):
                raise FileExistsError(
                    f'Test requires creating directory of {self.out_pt}{ptt}, but now there is a file. '
                    'Please clear such files to continue tests.'
                )
        print(f"NOTE: Some test logs and chk file will be saved to path '{self.out_pt}logs' and '{self.out_pt}results'")

    def _config(self, test_name: str) -> dict:
        """Return one test method's YAML configuration section.

        Args:
            test_name: Test method name used as the YAML section name.

        Return:
            Configuration mapping for the requested test.
        """
        return get_test_section(self.test_config, test_name)

    def test_Train(self):
        """
        Test training loop: loss should decrease over optimization steps.
        """
        print("Training and prediction tests are moved into the test_APIs, please run `test_APIs` instead.")
        pass

    def test_Pred(self):
        """
        Test model prediction: verify output shapes and values.
        """
        print("Training and prediction tests are moved into the test_APIs, please run `test_APIs` instead.")
        pass

    def test_MD(self):
        """
        Test Molecular Dynamics.
        """
        # purge remaining testfiles
        logfiles = glob.glob(os.path.join(self.out_pt, 'logs/MD*.log'))
        resultfiles = glob.glob(os.path.join(self.out_pt, 'results/MD*'))
        for logfile in logfiles:
            os.remove(logfile)
        for resultfile in resultfiles:
            os.remove(resultfile)

        # static test
        data = self.data
        elem_list = self.elem_list
        TEMPERATURE = 500.
        TIME_STEP = 1.5
        md_config = self._config('test_MD')
        md_steps = md_config.get('steps', {})
        static_steps = md_steps['static']
        nve_steps = md_steps['nve']
        nvt_steps = md_steps['nvt']

        # runner sets
        runner_cpu_static_nve = NVE(
            TIME_STEP, static_steps, 0., f'{self.out_pt}results/MD_STATIC_CPU', 1, device='cpu', verbose=1
        )
        runner_gpu_static_nve = NVE(
            TIME_STEP, static_steps, 0., f'{self.out_pt}results/MD_STATIC_GPU', 1, device='cuda:0', verbose=1
        )
        runner_gpu_move_nve = NVE(
            TIME_STEP, nve_steps, TEMPERATURE, f'{self.out_pt}results/MD_NVE_GPU', 1, device='cuda:0', verbose=0,
            is_compile=False
        )
        runner_cpu_csvr_nvt = NVT(
            TIME_STEP, nvt_steps, 'CSVR', {'time_const': 100},
            TEMPERATURE, f'{self.out_pt}results/MD_CSVR_CPU', 10, device='cpu', verbose=1,
            is_compile=False,
            compile_kwargs={'dynamic': False, 'options': {'epilogue_fusion': True, 'max_autotune': True}}
        )
        runner_gpu_csvr_nvt = NVT(
            TIME_STEP, nvt_steps, 'CSVR', {'time_const': 100},
            TEMPERATURE, f'{self.out_pt}results/MD_CSVR_GPU', 10, device='cuda:0', verbose=1,
            is_compile=False,
            compile_kwargs={'dynamic': False, 'options': {'epilogue_fusion': True, 'max_autotune': True}}
        )
        runner_cpu_lang_nvt = NVT(
            TIME_STEP, nvt_steps, 'Langevin', {'damping_coeff': 0.01},
            TEMPERATURE, f'{self.out_pt}results/MD_LANG_CPU', 10, device='cpu', verbose=1,
            is_compile=False
        )
        runner_gpu_lang_nvt = NVT(
            TIME_STEP, nvt_steps, 'Langevin', {'damping_coeff': 0.01},
            TEMPERATURE, f'{self.out_pt}results/MD_LANG_GPU', 10, device='cuda:0', verbose=1,
            is_compile=False
        )
        runner_cpu_nose_nvt = NVT(
            TIME_STEP, nvt_steps, 'Nose-Hoover', {},
            TEMPERATURE, f'{self.out_pt}results/MD_NOSE_CPU', 10, device='cpu', verbose=1
        )
        runner_gpu_nose_nvt = NVT(
            TIME_STEP, nvt_steps, 'Nose-Hoover', {},
            TEMPERATURE, f'{self.out_pt}results/MD_NOSE_GPU', 10, device='cuda:0', verbose=1
        )

        RUNNER_NAME = [
            'MD_STATIC_CPU', 'MD_STATIC_GPU', 'MD_NVE_GPU',
            'MD_CSVR_CPU', 'MD_CSVR_GPU',
            'MD_LANG_CPU', 'MD_LANG_GPU',
            'MD_NOSE_CPU', 'MD_NOSE_GPU',
        ]
        for i, runner in enumerate([
            runner_cpu_static_nve,
            runner_gpu_static_nve,
            runner_gpu_move_nve,
            runner_cpu_csvr_nvt,
            runner_gpu_csvr_nvt,
            runner_cpu_lang_nvt,
            runner_gpu_lang_nvt,
            runner_cpu_nose_nvt,
            runner_gpu_nose_nvt,
        ]):
            _data = data.to(runner.device).clone()
            model_test = self.model_test.to(runner.device)
            if 'STATIC' in RUNNER_NAME[i]:
                _data.pos = _data.pos0  # avoid uneq perturbation

            print("*"*89 + f"\nNow running {RUNNER_NAME[i]} ...\n" + "*"*89 + '\n')
            t_st = time.perf_counter()
            runner.reset_logger_handler(f"{self.out_pt}logs/{RUNNER_NAME[i]}.log")
            runner.run(
                model_test.Energy,
                _data.pos,
                elem_list,
                None,
                None,
                model_test.Grad,
                (_data, ),
                None,
                (_data, ),
                None,
                False,
                self.REQUIRE_GRAD,
                [len(_.pos) for _ in _data.to_data_list()],
                move_to_center_freq=-1
            )
            th.cuda.synchronize()
            print(f"{RUNNER_NAME[i]} finished. Elapsed time: {(time.perf_counter() - t_st):.2f} s")
            fbs = read_md_traj(f"{self.out_pt}results/{RUNNER_NAME[i]}")
            self._assert_finite_trajectory(fbs, RUNNER_NAME[i])
            os.remove(f"{self.out_pt}results/{RUNNER_NAME[i]}")
        pass

    def _assert_finite_trajectory(self, trajectory, test_name: str) -> None:
        """Check numeric trajectory fields for NaN or infinite values.

        Args:
            trajectory: Parsed MD or MC trajectory.
            test_name: Runner name used in the assertion message.

        Return:
            None.
        """
        for field_name in ('Energies', 'Coords', 'Labels'):
            values = getattr(trajectory, field_name, None)
            if values is None:
                continue
            for index, value in enumerate(values):
                self.assertTrue(
                    np.isfinite(np.asarray(value, dtype=float)).all(),
                    f'{test_name} produced non-finite {field_name} at frame {index}.',
                )

    def test_CMD(self):
        """
        Test Constrained Molecular Dynamics with regular batches (B, N, 3).
        All batches have equal size — uniform batch input.
        Covers NVE, CSVR, Langevin, and Nose-Hoover integrators with the same
        constraint function as the strict test, checking only finite output.
        """
        os.makedirs(f'{self.out_pt}logs', exist_ok=True)
        os.makedirs(f'{self.out_pt}results', exist_ok=True)

        # purge remaining testfiles
        logfiles = glob.glob(os.path.join(self.out_pt, 'logs/CMD*.log'))
        resultfiles = glob.glob(os.path.join(self.out_pt, 'results/CMD*'))
        for logfile in logfiles:
            os.remove(logfile)
        for resultfile in resultfiles:
            os.remove(resultfile)

        TEMPERATURE = 500.
        TIME_STEP = 0.5
        cmd_config = self._config('test_CMD')
        cmd_steps = cmd_config.get('steps', {})
        MAX_STEP = cmd_steps['integrator']
        CONSTR_THRESHOLD = cmd_config['convergence']['constraint_threshold']
        N_BATCH = 3  # 3 batches, all same size → regular batch (B, N, 3)
        N_ATOM = 5**3  # 125 Al atoms per batch → uniform
        ELEM = 'Al'
        MASS_val = MASS[ELEM]

        # ============================================================
        from BUCToolkit.BatchStructures.batch import Batch
        data_list = [build_cubic_lattice_data(5, 1.3, 0.05) for _ in range(N_BATCH)]
        # Batch for model (graph with pos0, batch vector)
        graph = Batch.from_data_list(data_list)
        # Regular input X: (B, N, 3)
        pos = th.stack([d.pos for d in data_list])
        elem_list = [[ELEM] * N_ATOM] * N_BATCH

        # Model
        raw_model = SimpleSpringPotential(graph.pos0, 10.)
        model_base = _Model_Wrapper_pyg(raw_model)

        # ============================================================
        # Multi-type constraint function (8 constraints)
        # ============================================================
        def constr_func(X):
            # X: (n_atom, n_dim) → returns the configured constraint values.
            y = list()
            # CONSTRAINT 1: fixed distances d(2,4), d(3,7), d(5,8)
            y.append(th.linalg.norm(X[[2, 3, 5]] - X[[4, 7, 8]], dim=-1))
            # CONSTRAINT 2: fixed angles cos(7-5-8), cos(11-9-12)
            x1 = X[[5, 9]]; x2 = X[[7, 11]]; x3 = X[[8, 12]]
            y.append(th.sum((x2 - x1) * (x3 - x1), dim=-1)
                     / (th.linalg.norm(x2 - x1, dim=-1) * th.linalg.norm(x3 - x1, dim=-1)))
            # CONSTRAINT 3: soft coordination numbers for atoms 14, 18
            r0 = 1.5; sigma = 1.
            r_ij = th.linalg.norm(X[[14, 18]].unsqueeze(1) - X.unsqueeze(0), dim=-1)
            s_i = th.sum(0.5 * (1.0 + th.erf((r_ij - r0) / sigma)), dim=-1)
            y.append(s_i)
            # CONSTRAINT 4: R_std
            R_ij = th.linalg.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1)
            R_std = th.std(R_ij, unbiased=True).unsqueeze(0)
            y.append(R_std)
            return th.cat(y)

        # ============================================================
        # Build runners: 4 integrators × (CPU + GPU) = 8 runners
        # ============================================================
        runner_cpu_nve = ConstrNVE(
            TIME_STEP, MAX_STEP,
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_NVE_CPU', 10,
            device='cpu', verbose=1
        )
        runner_gpu_nve = ConstrNVE(
            TIME_STEP, MAX_STEP,
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_NVE_GPU', 10,
            device='cuda:0', verbose=1
        )
        runner_cpu_csvr = ConstrNVT(
            TIME_STEP, MAX_STEP, 'CSVR', {'time_const': 100},
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_CSVR_CPU', 10,
            device='cpu', verbose=1
        )
        runner_gpu_csvr = ConstrNVT(
            TIME_STEP, MAX_STEP, 'CSVR', {'time_const': 100},
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_CSVR_GPU', 10,
            device='cuda:0', verbose=1
        )
        runner_cpu_lang = ConstrNVT(
            TIME_STEP, MAX_STEP, 'Langevin', {'damping_coeff': 0.01},
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_LANG_CPU', 10,
            device='cpu', verbose=1
        )
        runner_gpu_lang = ConstrNVT(
            TIME_STEP, MAX_STEP, 'Langevin', {'damping_coeff': 0.01},
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_LANG_GPU', 10,
            device='cuda:0', verbose=1
        )
        runner_cpu_nose = ConstrNVT(
            TIME_STEP, MAX_STEP, 'Nose-Hoover', {},
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_NOSE_CPU', 10,
            device='cpu', verbose=1
        )
        runner_gpu_nose = ConstrNVT(
            TIME_STEP, MAX_STEP, 'Nose-Hoover', {},
            constr_func, None, CONSTR_THRESHOLD,
            False, TEMPERATURE,
            f'{self.out_pt}results/CMD_NOSE_GPU', 10,
            device='cuda:0', verbose=1
        )

        RUNNER_NAME = [
            'CMD_NVE_CPU', 'CMD_NVE_GPU',
            'CMD_CSVR_CPU', 'CMD_CSVR_GPU',
            'CMD_LANG_CPU', 'CMD_LANG_GPU',
            'CMD_NOSE_CPU', 'CMD_NOSE_GPU',
        ]
        for i, runner in enumerate([
            runner_cpu_nve, runner_gpu_nve,
            runner_cpu_csvr, runner_gpu_csvr,
            runner_cpu_lang, runner_gpu_lang,
            runner_cpu_nose, runner_gpu_nose,
        ]):
            _pos = pos.to(runner.device)
            _graph = graph.to(runner.device)
            model_test = model_base.to(runner.device)

            print("*" * 89 + f"\nNow running {RUNNER_NAME[i]} ...\n" + "*" * 89 + '\n')
            runner.reset_logger_handler(f"{self.out_pt}logs/{RUNNER_NAME[i]}.log")
            t_st = time.perf_counter()
            runner.run(
                model_test.Energy,
                _pos,           # X: (B, N, 3) — regular batch
                elem_list,
                None, None,
                model_test.Grad,
                (_graph,),      # graph with pos0, batch
                None,
                (_graph,), None,
                False, False,   # is_grad_contain_y, require_grad
                None,           # batch_indices=None → regular batch
                move_to_center_freq=-1
            )
            if runner.device.type == 'cuda':
                th.cuda.synchronize()
            print(f"{RUNNER_NAME[i]} finished. Elapsed time: {(time.perf_counter() - t_st):.2f} s")

            # --- Read trajectory ---
            fbs = read_md_traj(f"{self.out_pt}results/{RUNNER_NAME[i]}")
            self._assert_finite_trajectory(fbs, RUNNER_NAME[i])
            os.remove(f"{self.out_pt}results/{RUNNER_NAME[i]}")
        pass

    def test_MC(self):
        """
        Test the Monte Carlo algorithms.
        """
        # purge remaining testfiles
        logfiles = glob.glob(os.path.join(self.out_pt, 'logs/MC*.log'))
        resultfiles = glob.glob(os.path.join(self.out_pt, 'results/MC*'))
        for logfile in logfiles:
            os.remove(logfile)
        for resultfile in resultfiles:
            os.remove(resultfile)

        # static test
        data = self.data
        elem_list = self.elem_list
        TEMPERATURE = 500.
        mc_config = self._config('test_MC')
        mc_steps = mc_config['steps']

        # runner sets
        runner_cpu_nvt = MMC(
            'Gaussian',
            mc_steps,
            TEMPERATURE,
            'constant',
            1,
            None,
            0.07,
            f'{self.out_pt}results/MC_GAUSS_NVT_CPU',
            10,
            device='cpu',
            verbose=1,
            is_compile=False
        )
        runner_cpu_cauchy_nvt = MMC(
            'Cauchy',
            mc_steps,
            TEMPERATURE,
            'constant',
            1,
            None,
            0.07,
            f'{self.out_pt}results/MC_CAUCHY_NVT_CPU',
            10,
            device='cpu',
            verbose=1,
            is_compile=False
        )
        runner_cpu_uniform_nvt = MMC(
            'Uniform',
            mc_steps,
            TEMPERATURE,
            'constant',
            1,
            None,
            0.07,
            f'{self.out_pt}results/MC_UNIFORM_NVT_CPU',
            10,
            device='cpu',
            verbose=1,
            is_compile=False
        )
        runner_gpu_nvt = MMC(
            'Gaussian',
            mc_steps,
            TEMPERATURE,
            'constant',
            1,
            None,
            0.07,
            f'{self.out_pt}results/MC_GAUSS_NVT_GPU',
            10,
            device='cuda:0',
            verbose=1,
            is_compile = False
        )
        runner_gpu_anneal = MMC(
            'Gaussian',
            mc_steps,
            TEMPERATURE,
            'fast',
            1,
            None,
            0.07,
            f'{self.out_pt}results/MC_GAUSS_ANNEAL_GPU',
            10,
            device='cuda:0',
            verbose=0
        )

        RUNNER_NAME = [
            'MC_GAUSS_NVT_CPU',
            'MC_CAUCHY_NVT_CPU',
            'MC_UNIFORM_NVT_CPU',
            'MC_GAUSS_NVT_GPU',
            'MC_GAUSS_ANNEAL_GPU',
        ]
        for i, runner in enumerate([
            runner_cpu_nvt,
            runner_cpu_cauchy_nvt,
            runner_cpu_uniform_nvt,
            runner_gpu_nvt,
            runner_gpu_anneal,
        ]):
            _data = data.to(runner.device).clone()
            model_test = self.model_test.to(runner.device)
            print("*" * 89 + f"\nNow running {RUNNER_NAME[i]} ...\n" + "*" * 89 + '\n')
            t_st = time.perf_counter()
            runner.reset_logger_handler(f"{self.out_pt}logs/{RUNNER_NAME[i]}.log")
            runner.run(
                model_test.Energy,
                _data.pos,
                elem_list,
                None,
                (_data,),
                None,
                [len(_.pos) for _ in _data.to_data_list()],
                fixed_atom_tensor=None,
                move_to_center_freq=-1
            )
            print(f"{RUNNER_NAME[i]} finished. Elapsed time: {(time.perf_counter() - t_st):.2f} s")
            # validation
            fbs = read_mc_traj(f"{self.out_pt}results/{RUNNER_NAME[i]}")
            self._assert_finite_trajectory(fbs, RUNNER_NAME[i])
            os.remove(f"{self.out_pt}results/{RUNNER_NAME[i]}")


    def test_OPT(self):
        """
        Test the structure optimizations by various algorithms.
        """
        # purge remaining testfiles
        for f in glob.glob(os.path.join(self.out_pt, 'logs/OPT*')):
            try: os.remove(f)
            except OSError: pass
        for f in glob.glob(os.path.join(self.out_pt, 'results/OPT*')):
            try: os.remove(f)
            except OSError: pass

        # static test
        data = self.data
        elem_list = self.elem_list
        kB = 8.617333262145e-5  # eV/K
        TIME_STEP = 0.5
        opt_config = self._config('test_OPT')
        opt_convergence = opt_config.get('convergence', {})
        opt_optimizer = opt_config.get('optimizer', {})
        MAXITER = opt_config['maxiter']
        CPU_QN_MAXITER = opt_config['cpu_qn_maxiter']
        OPT_E_THRESHOLD = opt_optimizer['energy_threshold']
        OPT_F_THRESHOLD = opt_optimizer['force_threshold']

        # runner sets
        runner_cpu_cg_mt = CG(
            'PR+',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            'MT',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cpu',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_CG_MT_CPU_DUMP.bin',
        )
        runner_gpu_cg_mt = CG(
            'PR+',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            'MT',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cuda:0',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_CG_MT_GPU_DUMP.bin',
        )
        runner_cpu_cg_bk = CG(
            'PR+',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            'Backtrack',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cpu',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_CG_BK_CPU_DUMP.bin',
        )
        runner_gpu_cg_bk = CG(
            'PR+',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            'Backtrack',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cuda:0',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_CG_BK_GPU_DUMP.bin',
        )
        runner_cpu_bfgs_mt = QN(
            'BFGS',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            CPU_QN_MAXITER,
            'MT',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cpu',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_QN_MT_CPU_DUMP.bin',
        )
        runner_gpu_bfgs_mt = QN(
            'BFGS',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            'MT',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cuda:0',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_QN_MT_GPU_DUMP.bin',
        )
        runner_cpu_bfgs_bk = QN(
            'BFGS',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            CPU_QN_MAXITER,
            'Backtrack',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cpu',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_QN_BK_CPU_DUMP.bin',
        )
        runner_gpu_bfgs_bk = QN(
            'BFGS',
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            'Backtrack',
            10,
            0.2,
            0.6,
            TIME_STEP,
            use_bb=True,
            device='cuda:0',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_QN_BK_GPU_DUMP.bin',
        )
        runner_cpu_fire = FIRE(
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            TIME_STEP,
            device='cpu',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_FIRE_CPU_DUMP.bin',
        )
        runner_gpu_fire = FIRE(
            OPT_E_THRESHOLD,
            OPT_F_THRESHOLD,
            MAXITER,
            TIME_STEP,
            device='cuda:0',
            verbose=1,
            output_file=f'{self.out_pt}results/OPT_FIRE_GPU_DUMP.bin',
        )


        RUNNER_NAME = [
            'OPT_CG_MT_CPU',
            'OPT_CG_MT_GPU',
            'OPT_CG_BK_CPU',
            'OPT_CG_BK_GPU',
            'OPT_BFGS_MT_CPU',
            'OPT_BFGS_MT_GPU',
            'OPT_BFGS_BK_CPU',
            'OPT_BFGS_BK_GPU',
            'OPT_FIRE_CPU',
            'OPT_FIRE_GPU',
        ]
        for i, runner in enumerate([
            runner_cpu_cg_mt,
            runner_gpu_cg_mt,
            runner_cpu_cg_bk,
            runner_gpu_cg_bk,
            runner_cpu_bfgs_mt,
            runner_gpu_bfgs_mt,
            runner_cpu_bfgs_bk,
            runner_gpu_bfgs_bk,
            runner_cpu_fire,
            runner_gpu_fire,
        ]):
            _data = data.to(runner.device).clone()
            model_test = self.model_test.to(runner.device)
            print("*" * 89 + f"\nNow running {RUNNER_NAME[i]} ...\n" + "*" * 89 + '\n')
            t_st = time.perf_counter()
            runner.reset_logger_handler(f"{self.out_pt}logs/{RUNNER_NAME[i]}.log")
            updater = PygBatchUpdater()
            updater.initialize()
            runner.set_batch_updater(updater, updater)
            runner: FIRE
            # set dump metadata
            _atm_per_struct = [[26] * (8**3), [13] * (5**3), [46] * (10**3)]
            runner.set_system_info(atomic_numbers=_atm_per_struct)
            y, x_min, g = runner.run(
                model_test.Energy,
                _data.pos,
                model_test.Grad,
                (_data,),
                None,
                (_data, ),
                None,
                False,
                self.REQUIRE_GRAD,
                True,
                None,
                [len(_.pos) for _ in _data.to_data_list()],
            )
            print(f"{RUNNER_NAME[i]} finished. Elapsed time: {(time.perf_counter() - t_st):.2f} s\n")
            # validation
            ene1, ene2, ene3 = y[0], y[1], y[2]
            #std_pos = [_.pos for _ in _data.to_data_list()]
            for _i, _en in enumerate((ene1, ene2, ene3)):
                # plt.plot(_en)
                # plt.show()
                # plt.clf()
                try:
                    self.assertAlmostEqual(_en, 0., delta=opt_convergence['energy_delta'])
                    self.assertAlmostEqual(
                        th.max(th.abs(g)).item(), 0.,
                        delta=opt_convergence['force_delta'],
                    )
                    max_diff = th.max(th.abs(_data.pos - data.pos0)).item()
                    self.assertAlmostEqual(
                        max_diff, 0.,
                        delta=opt_convergence['position_delta'],
                    )
                    print(f'"OPT" Test {_i + 1} passed. <<<<<')
                    print(f"Energy: {_en}, STD Energy: 0.")
                    print(f"Max Coordinates difference of standard value: {max_diff}\n")
                except AssertionError:
                    print(f'\n"OPT" Test {_i + 1} Failed:\n'
                          f'test value:\n\tenergy: {_en}\n\tmax forces: {th.max(g.abs()).item()}'
                          f'\nstandard value:\n\tenergy: 0.\n\tmax forces: 0.\n'
                          f'position displacement max error: {th.max(th.abs(_data.pos - data.pos0)).item()}')

            # --- dump verification: read full trajectory, cross-check with log ---
            bs_full = read_opt_structures(runner.output_file)
            _logfile = f"{self.out_pt}logs/{RUNNER_NAME[i]}.log"
            with open(_logfile) as lf:
                _log_E = [float(v) for ln in lf if 'Energies' in ln
                          for v in ln.split('[')[1].split(']')[0].split()]
            self.assertEqual(len(bs_full.Energies), len(_log_E),
                             f'{RUNNER_NAME[i]}: dump ({len(bs_full.Energies)}) '
                             f'!= log ({len(_log_E)})')
            for _j in range(len(_log_E)):
                self.assertAlmostEqual(bs_full.Energies[_j], _log_E[_j],
                    delta=opt_convergence['dump_energy_delta'],
                    msg=f'{RUNNER_NAME[i]} step {_j}: '
                        f'dump={bs_full.Energies[_j]:.8f} '
                        f'log={_log_E[_j]:.8f}')

            # only_opt returns just the last frame
            bs_last = read_opt_structures(runner.output_file, only_opt=True)
            _n_structs = len(_atm_per_struct)
            self.assertEqual(len(bs_last), _n_structs,
                             f'{RUNNER_NAME[i]}: only_opt should have {_n_structs} structures')
            for _s in range(_n_structs):
                self.assertAlmostEqual(
                    bs_last.Energies[_s], y[_s].item(),
                    delta=opt_convergence['last_energy_delta'],
                )

    def test_TS(self):
        """
        TS search on 3D Cerjan-Miller potential with irregular batch.
        Saddle at origin, E=0. Tol: |E|<5e-2, |X|_oo<0.01.
        """
        from BUCToolkit.BatchOptim.TS.Dimer import Dimer
        from BUCToolkit.BatchOptim.TS.Krylov import KrylovNewton, KrylovDynamics
        from BUCToolkit.BatchStructures.batch import Data, Batch
        import torch as th

        # Build irregular 3D batch: 2 structures with 27 + 8 atoms,
        # initialized randomly near the saddle at origin
        th.manual_seed(42)
        d1 = Data(pos=th.randn(27, 3) * 0.5)
        d2 = Data(pos=th.randn(8, 3) * 0.5)
        data = Batch.from_data_list([d1, d2])
        X0 = data.pos.unsqueeze(0)          # (1, 35, 3)
        bi = [27, 8]

        # Energy: flatten each structure to (N_atoms*3,) vector,
        # one saddle per structure, deterministic coefficients.
        # E(x) = sum_i c2[i]*x_i^2 + 0.1x_i^4,  with c2[0]=-1, c2[i>0]=1.
        def Energy(X, data):
            X_ = X.squeeze(0)
            out = th.zeros(data.num_graphs, device=X.device, dtype=X.dtype)
            for s in range(data.num_graphs):
                x_s = X_[data.batch == s].reshape(1, -1)
                n = x_s.shape[-1]
                c2 = th.ones(n, device=x_s.device, dtype=x_s.dtype)
                c2[:1] = -1.0
                x2 = x_s ** 2
                out[s] = (c2 * x2 + 0.1 * x2 ** 2).sum()
            return out

        def Grad(X, data):
            from torch.func import grad
            return grad(lambda x: Energy(x, data).sum())(X)

        # Batch updater: filter data when structures converge
        class BatchUpdater:
            def initialize(self): pass
            def __call__(self, mask, f_args, f_kw, g_args, g_kw):
                # mask: (n_struct,) bool, True=keep(unconverged)
                d = f_args[0]
                if mask.all():
                    return f_args, f_kw, g_args, g_kw
                kept = [d for d, m in zip(d.to_data_list(), mask) if m]
                new_d = type(d).from_data_list(kept)
                return (new_d,), f_kw, (new_d,), g_kw

        updater = BatchUpdater()
        updater.initialize()

        ts_config = self._config('test_TS')
        ts_convergence = ts_config.get('convergence', {})
        ts_optimizer = ts_config.get('optimizer', {})
        ts_maxiter = ts_config['maxiter']
        TS_E_THRESHOLD = ts_optimizer['energy_threshold']
        TS_F_THRESHOLD = ts_optimizer['force_threshold']
        TS_X_THRESHOLD = ts_optimizer['position_threshold']
        for dtp in ('cpu', 'cuda:0'):
            X0 = X0.to(dtp)
            data = data.to(dtp)
            # Dimer
            dimer = Dimer(
                TS_E_THRESHOLD, TS_F_THRESHOLD, -0.1, TS_X_THRESHOLD,
                ts_maxiter, 10, 0.5, 0.02,
                device=dtp, verbose=0
            )
            updater.initialize(); dimer.set_batch_updater(updater)
            t_st = time.perf_counter()
            y_d, X_d = dimer.run(Energy, X0.clone(), grad_func=Grad, func_args=(data,),
                               grad_func_args=(data,), batch_indices=bi,
                               is_grad_func_contain_y=False, require_grad=False)
            th.cuda.synchronize()
            print(
                f'  [{dtp}] Dimer:          E={float(y_d.abs().max()):.6e}, |X|={float(X_d.abs().max()):.6e}, '
                f'Elapsed time: {(time.perf_counter() - t_st):.6e}'
            )
            try:
                self.assertLess(float(y_d.abs().max()), ts_convergence['energy_abs'])
                self.assertLess(float(X_d.abs().max()), ts_convergence['position_abs'])
            except AssertionError:
                print('Dimer failed.')

            # KrylovNewton
            kn = KrylovNewton(
                TS_E_THRESHOLD, TS_F_THRESHOLD, 0.01, TS_X_THRESHOLD,
                ts_maxiter, 10, 0.05, steplength_sheme='trust_region',
                device=dtp, verbose=0
            )
            updater.initialize(); kn.set_batch_updater(updater)
            t_st = time.perf_counter()
            y_kn, X_kn = kn.run(Energy, X0.clone(), grad_func=Grad, func_args=(data,),
                                grad_func_args=(data,), batch_indices=bi,
                                is_grad_func_contain_y=False, require_grad=False)
            th.cuda.synchronize()
            print(
                f'  [{dtp}] KrylovNewton:   E={float(y_kn.abs().max()):.6e}, |X|={float(X_kn.abs().max()):.6e}, '
                f'Elapsed time: {(time.perf_counter() - t_st):.6e}'
            )
            try:
                self.assertLess(float(y_kn.abs().max()), ts_convergence['energy_abs'])
                self.assertLess(float(X_kn.abs().max()), ts_convergence['position_abs'])
            except AssertionError:
                print('KrylovNewton failed.')

            # KrylovDynamics
            kd = KrylovDynamics(
                TS_E_THRESHOLD, TS_F_THRESHOLD, 0.01, TS_X_THRESHOLD,
                ts_maxiter, 30, 0.1,
                device=dtp, verbose=0
            )
            updater.initialize(); kd.set_batch_updater(updater)
            t_st = time.perf_counter()
            y_kd, X_kd = kd.run(Energy, X0.clone(), grad_func=Grad, func_args=(data,),
                                grad_func_args=(data,), batch_indices=bi,
                                is_grad_func_contain_y=False, require_grad=False, extra_krylov_dim=1)
            th.cuda.synchronize()
            print(
                f'  [{dtp}] KrylovDynamics: E={float(y_kd.abs().max()):.6e}, |X|={float(X_kd.abs().max()):.6e}, '
                f'Elapsed time: {(time.perf_counter() - t_st):.6e}'
            )
            try:
                self.assertLess(float(y_kd.abs().max()), ts_convergence['energy_abs'])
                self.assertLess(float(X_kd.abs().max()), ts_convergence['position_abs'])
            except AssertionError:
                print('KrylovDynamics failed.')
            th.cuda.synchronize()

    def test_VIB(self):
        """Validate all Hessian paths and the frequency dump round trip."""
        vib_config = self._config('test_VIB')
        vib_block_size = vib_config['block_size']
        vib_delta = vib_config['finite_difference_delta']
        vib_float32_tolerance = vib_config['float32_tolerance']
        vib_float64_tolerance = vib_config['float64_tolerance']
        devices = ['cpu']
        if th.cuda.is_available():
            devices.append('cuda:0')

        output_root = os.path.join(self.out_pt, 'results')
        for output_file in glob.glob(os.path.join(output_root, 'VIB_*.bin')):
            if os.path.isfile(output_file):
                os.remove(output_file)

        n_atom = 8
        n_free_atom = 7
        n_free_dof = 3 * n_free_atom
        for device in devices:
            for dtype in (th.float32, th.float64):
                hessian_diagonal = th.linspace(
                    2., 4., n_free_dof, device=device, dtype=dtype
                )
                expected_hessian = th.diag(hessian_diagonal)
                hessian_coupling = th.full(
                    (n_free_dof - 1,), 0.05, device=device, dtype=dtype
                )
                expected_hessian += (
                    th.diag(hessian_coupling, diagonal=1)
                    + th.diag(hessian_coupling, diagonal=-1)
                )
                full_hessian = th.zeros(
                    3 * n_atom, 3 * n_atom, device=device, dtype=dtype
                )
                full_hessian[:n_free_dof, :n_free_dof] = expected_hessian
                full_hessian[n_free_dof:, n_free_dof:] = th.eye(
                    3, device=device, dtype=dtype
                )
                coordinates = th.linspace(
                    -0.02, 0.02, 3 * n_atom, device=device, dtype=dtype
                ).reshape(n_atom, 3)
                free_atom_mask = th.tensor(
                    [1] * n_free_atom + [0], device=device
                )

                def energy(X):
                    X_flat = X.reshape(X.size(0), -1)
                    return 0.5 * th.einsum(
                        'bi,ij,bj->b', X_flat, full_hessian, X_flat
                    )

                def gradient(X):
                    return (X.reshape(X.size(0), -1) @ full_hessian).reshape_as(X)

                expected_eigenvalues, expected_eigenvectors = th.linalg.eigh(
                    expected_hessian
                )
                expected_frequencies = th.sqrt(expected_eigenvalues)
                expected_modes = expected_eigenvectors.reshape(
                    n_free_dof, n_free_atom, 3
                )
                tolerance = (
                    vib_float32_tolerance
                    if dtype == th.float32 else vib_float64_tolerance
                )

                for method in ('EnergyDiff', 'GradDiff', 'Autograd'):
                    with self.subTest(
                            method=method, device=device, dtype=dtype,
                    ):
                        dtype_name = str(dtype).split('.')[-1]
                        output_file = os.path.join(
                            output_root,
                            f'VIB_{method}_{device.replace(":", "_")}_{dtype_name}.bin',
                        )
                        calculator = Frequency(
                            method=method,
                            block_size=vib_block_size,
                            delta=vib_delta,
                            output_file=output_file,
                            dump_hessian=True,
                        )
                        try:
                            frequencies, normal_mode = calculator.normal_mode(
                                energy,
                                coordinates.clone(),
                                grad_func=gradient,
                                fixed_atom_tensor=free_atom_mask,
                                save_hessian=True,
                            )
                        finally:
                            calculator.dumper.close()

                        th.testing.assert_close(
                            calculator.hessian,
                            expected_hessian,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                        th.testing.assert_close(
                            frequencies,
                            expected_frequencies,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                        th.testing.assert_close(
                            normal_mode.abs(),
                            expected_modes.abs(),
                            rtol=tolerance,
                            atol=tolerance,
                        )

                        dumped = read_freq(output_file)
                        self.assertEqual(
                            tuple(dumped),
                            ('frequencies', 'normal_mode', 'hessian'),
                        )
                        self.assertEqual(
                            {name: len(values) for name, values in dumped.items()},
                            {'frequencies': 1, 'normal_mode': 1, 'hessian': 1},
                        )
                        th.testing.assert_close(
                            dumped['frequencies'][0], frequencies.cpu()
                        )
                        th.testing.assert_close(
                            dumped['normal_mode'][0], normal_mode.cpu()
                        )
                        th.testing.assert_close(
                            dumped['hessian'][0], calculator.hessian.cpu()
                        )

        output_file = os.path.join(output_root, 'VIB_different_shapes.bin')
        calculator = Frequency(
            method='GradDiff',
            block_size=vib_block_size,
            output_file=output_file,
        )
        try:
            for n_atom in (7, 8):
                coordinates = th.zeros(n_atom, 3)
                calculator.normal_mode(
                    lambda X: X.square().sum(dim=(-2, -1)),
                    coordinates,
                    grad_func=lambda X: 2. * X,
                )
        finally:
            calculator.dumper.close()
        dumped = read_freq(output_file)
        self.assertEqual(tuple(dumped), ('frequencies', 'normal_mode'))
        self.assertEqual(
            [value.shape for value in dumped['frequencies']],
            [(21,), (24,)],
        )
        self.assertEqual(
            [value.shape for value in dumped['normal_mode']],
            [(21, 7, 3), (24, 8, 3)],
        )

        self.assertEqual(Frequency('Coord').method, 'EnergyDiff')
        self.assertEqual(Frequency('Grad').method, 'Autograd')
        with self.assertRaisesRegex(ValueError, 'method'):
            Frequency('invalid')

    def test_parallel(self):
        """
        Test of parallel efficiency
        Returns:

        """
        # purge old files
        filelist = glob.glob(f'{self.out_pt}logs/*_paratest.log')
        for ff in filelist: os.remove(ff)
        # have 64 samples in total
        SMALL_BATCHES = [
            8,  9,  4,  8,  7,  9,  4,  5,  4,  5, 10,  9,  6, 10,  5,  3,  5,  8,
            6,  6,  4,  4,  8,  8,  5,  6,  9,  6,  8,  6,  7,  5,  9,  5,  3,  5,
            9,  3,  4,  4,  9,  8,  9,  6,  5,  7,  3,  8,  6, 10,  8, 10,  5,  5,
            8,  6,  9,  3,  9,  6,  3,  4,  9,  3
        ]
        LARGE_BATCHES = [
            12, 12, 18, 20, 16, 17, 13, 11, 20, 19, 19, 20, 20, 13, 19, 15, 16, 19,
            18, 14, 13, 20, 20, 18, 12, 14, 17, 10, 13, 11, 10, 15, 18, 15, 19, 12,
            10, 16, 11, 15, 16, 12, 10, 17, 10, 17, 19, 13, 15, 19, 20, 17, 12, 10,
            18, 20, 15, 10, 10, 15, 11, 11, 16, 12
        ]
        TOTAL_ELEM = ['Fe', 'Al', 'Pd', 'C'] * 16

        # input const
        parallel_config = self._config('test_parallel')
        parallel_steps = parallel_config.get('steps', {})
        parallel_convergence = parallel_config.get('convergence', {})
        MAXITER = parallel_config['maxiter']
        md_steps = parallel_steps['md']
        mc_steps = parallel_steps['mc']
        warmup_steps = parallel_steps['warmup']
        PARALLEL_E_THRESHOLD = parallel_convergence['energy_threshold']
        PARALLEL_F_THRESHOLD = parallel_convergence['force_threshold']
        TIME_STEP = 0.5
        TEMPERATURE = 873

        # runners
        runners = {
            #   opt
            'opt_cpu_cg_mt' : CG(
                'PR+',
                PARALLEL_E_THRESHOLD,
                PARALLEL_F_THRESHOLD,
                MAXITER,
                'MT',
                10,
                0.2,
                0.6,
                TIME_STEP,
                use_bb=True,
                device='cpu',
                verbose=1,
            ),
            'opt_gpu_cg_mt' : CG(
                'PR+',
                PARALLEL_E_THRESHOLD,
                PARALLEL_F_THRESHOLD,
                MAXITER,
                'MT',
                10,
                0.2,
                0.6,
                TIME_STEP,
                use_bb=True,
                device='cuda:0',
                verbose=1,
            ),
            'opt_cpu_fire' : FIRE(
                PARALLEL_E_THRESHOLD,
                PARALLEL_F_THRESHOLD,
                MAXITER,
                TIME_STEP,
                device='cpu',
                verbose=1,
            ),
            'opt_gpu_fire' : FIRE(
                PARALLEL_E_THRESHOLD,
                PARALLEL_F_THRESHOLD,
                MAXITER,
                TIME_STEP,
                device='cuda:0',
                verbose=1,
            ),
            #   mc
            'mc_cpu_nvt' : MMC(
                'Gaussian',
                mc_steps,
                TEMPERATURE,
                'constant',
                1,
                None,
                0.07,
                f'{self.out_pt}results/MC_GAUSS_NVT_CPU_PARA',
                10,
                device='cpu',
                verbose=1,
                is_compile=False
            ),
            'mc_gpu_nvt' : MMC(
                'Gaussian',
                mc_steps,
                TEMPERATURE,
                'constant',
                1,
                None,
                0.07,
                f'{self.out_pt}results/MC_GAUSS_NVT_GPU_PARA',
                10,
                device='cuda:0',
                verbose=1,
                is_compile=False
            ),
            #   md
            'md_cpu_nve': NVE(
                TIME_STEP, md_steps, TEMPERATURE, f'{self.out_pt}results/MD_NVE_CPU_PARA',
                10, device='cpu', verbose=0,
                is_compile=False
            ),
            'md_gpu_nve' : NVE(
            TIME_STEP, md_steps, TEMPERATURE, f'{self.out_pt}results/MD_NVE_GPU_PARA',
                10, device='cuda:0', verbose=0,
            is_compile=False
            ),
            'md_cpu_csvr_nvt' : NVT(
                TIME_STEP, md_steps, 'CSVR', {'time_const': 100},
                TEMPERATURE, f'{self.out_pt}results/MD_CSVR_CPU_PARA', 10, device='cpu', verbose=1,
                is_compile=False,
                compile_kwargs={'dynamic': False, 'options': {'epilogue_fusion': True, 'max_autotune': True}}
            ),
            'md_gpu_csvr_nvt' : NVT(
                TIME_STEP, md_steps, 'CSVR', {'time_const': 100},
                TEMPERATURE, f'{self.out_pt}results/MD_CSVR_GPU_PARA', 10, device='cuda:0', verbose=1,
                is_compile=False,
                compile_kwargs={'dynamic': False, 'options': {'epilogue_fusion': True, 'max_autotune': True}}
            ),
            'md_cpu_lang_nvt' : NVT(
                TIME_STEP, md_steps, 'Langevin', {'damping_coeff': 0.01},
                TEMPERATURE, f'{self.out_pt}results/MD_LANG_CPU_PARA', 10, device='cpu', verbose=0,
                is_compile=False
            ),
            'md_gpu_lang_nvt' : NVT(
                TIME_STEP, md_steps, 'Langevin', {'damping_coeff': 0.01},
                TEMPERATURE, f'{self.out_pt}results/MD_LANG_GPU_PARA', 10, device='cuda:0', verbose=0,
                is_compile=False
            ),
            'md_cpu_nose_nvt' : NVT(
                TIME_STEP, md_steps, 'Nose-Hoover', {},
                TEMPERATURE, f'{self.out_pt}results/MD_NOSE_CPU_PARA', 10, device='cpu', verbose=0
            ),
            'md_gpu_nose_nvt' : NVT(
                TIME_STEP, md_steps, 'Nose-Hoover', {},
                TEMPERATURE, f'{self.out_pt}results/MD_NOSE_GPU_PARA', 10, device='cuda:0', verbose=0
            )
        }

        # warm start
        data = build_cubic_lattice_batch([5, 3], 3., 1.).to('cuda:0')
        pre_runner = MMC(
                'Gaussian',
                warmup_steps,
                TEMPERATURE,
                'constant',
                1,
                None,
                0.07,
                None,
                10,
                device='cuda:0',
                verbose=1,
                is_compile=False
            )
        pre_runner.run(
            self.model_test.to('cuda:0').Energy,
            data.pos,
            None,
            None,
            func_args=(data,),
            batch_indices=[len(_.pos) for _ in data.to_data_list()],
            move_to_center_freq=-1
        )

        # small batches test:
        for name, runner in runners.items():
            print(f"TASK: {name} started... ")
            # main loop
            for i in range(1, len(SMALL_BATCHES)+1, 4):
                # handle inp data
                ATOMS = SMALL_BATCHES[:i]
                data = build_cubic_lattice_batch(ATOMS, 3., 1.)
                elem_list = [[]]
                for _, __ in enumerate(TOTAL_ELEM[:i]):
                    elem_list[0].extend([__] * (ATOMS[_] ** 3))
                model_test = self.model_test.to(runner.device)
                # purge old file
                fileslist = glob.glob(f'{self.out_pt}results/*_PARA')
                for ff in fileslist: os.remove(ff)
                # running
                _data = data.to(runner.device).clone()
                t_st = time.perf_counter()
                if name.startswith('opt_'):
                    runner.reset_logger_handler(f"{self.out_pt}logs/{name}_paratest.log")
                    updater = PygBatchUpdater()
                    updater.initialize()
                    runner.set_batch_updater(updater, updater)
                    runner: FIRE
                    y, x_min, g = runner.run(
                        model_test.Energy,
                        _data.pos,
                        model_test.Grad,
                        (_data,),
                        None,
                        (_data,),
                        None,
                        False,
                        self.REQUIRE_GRAD,
                        True,
                        None,
                        [len(_.pos) for _ in _data.to_data_list()],
                    )
                    th.cuda.synchronize()

                elif name.startswith('md_'):
                    runner: NVT
                    runner.reset_logger_handler(f"{self.out_pt}logs/{name}_paratest.log")
                    runner.run(
                        model_test.Energy,
                        _data.pos,
                        elem_list,
                        None,
                        None,
                        model_test.Grad,
                        (_data,),
                        None,
                        (_data,),
                        None,
                        False,
                        self.REQUIRE_GRAD,
                        [len(_.pos) for _ in _data.to_data_list()],
                        move_to_center_freq=-1
                    )
                    th.cuda.synchronize()

                elif name.startswith('mc_'):
                    runner: MMC
                    runner.reset_logger_handler(f"{self.out_pt}logs/{name}_paratest.log")
                    runner.run(
                        model_test.Energy,
                        _data.pos,
                        elem_list,
                        None,
                        func_args=(_data,),
                        batch_indices=[len(_.pos) for _ in _data.to_data_list()],
                        move_to_center_freq=-1
                    )

                print(
                    f"BATCH SIZE: {i}, ATOMS: {sum(_ ** 3 for _ in ATOMS)}. "
                    f"Elapsed time: {(time.perf_counter() - t_st):.2f} s <<<\n"
                )

            print(f"TASK: {name} TEST DONE.\n" + "*"*89)

    def test_IO(self):
        """Test I/O: OUTCAR → POSCAR/cif → binary round-trip."""
        tmp = self.out_pt
        try:
            from io_test import run_io_tests
            errors = run_io_tests(tmp)
            if errors:
                self.fail('\n'.join(errors))
        finally:
            #shutil.rmtree(tmp, ignore_errors=True)
            pass


    def test_APIS(self):
        """Test Trainer + Predictor APIs with GNN-LJ-EAM model."""
        tmp = self.out_pt
        try:
            from test_apis import run_api_tests
            errors = run_api_tests(tmp, config=self._config('test_APIS'))
            if errors:
                self.fail('\n'.join(errors))
        finally:
            #shutil.rmtree(tmp, ignore_errors=True)
            pass


if __name__ == '__main__':
    import sys
    import datetime
    try:
        with open(f'./testsuite.log', 'w') as f:
            f.write(f'START TIME: {datetime.datetime.now()}\n\n')
            sys.stdout = f
            unittest.main()
            #test.test_parallel()
    finally:
        sys.stdout = sys.__stdout__
