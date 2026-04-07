#  Copyright (c) 2026.1.27, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: main_test.py
#  Environment: Python 3.12
import time
import unittest
import os
import glob
import math

import torch as th
import numpy as np

from BUCToolkit.cli.main import launch_task
from BUCToolkit.BatchStructures import read_md_traj, read_opt_structures, read_mc_traj
from BUCToolkit.api._io import _Model_Wrapper_pyg
from BUCToolkit.BatchMD import NVE, NVT
from BUCToolkit.BatchMD.constrained_md import ConstrNVE, ConstrNVT
from BUCToolkit.BatchOptim import QN, CG, FIRE, Frequency
from BUCToolkit.BatchMC import MMC
from BUCToolkit.utils.AtomicNumber2Properties import MASS
from testsuite._toy_harmonic_potential import (HarmonicLatticePotential, SimpleSpringPotential,
                                               build_cubic_lattice_batch, build_cubic_lattice_data)

INPUT_PATH = './inputs4test/'
HERE = os.path.dirname(os.path.abspath(__file__))

class MainTest(unittest.TestCase):

    @staticmethod
    def assertStatisticalEqual(a, b, rtol=1e-05, atol=1e-08, msg=None):
        """
        Used for MD/MC ensemble validation
        Args:
            a: statistical magnitude a
            b: statistical magnitude b
            rtol: relative tolerance
            atol: absolute tolerance
            msg: custom error message

        Returns: None

        """
        err_msg = str(msg) if msg is not None else f'Statistical validation Failed.\na: {a}\nb: {b}'
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(err_msg)

    def setUp(self):
        # data
        ATOMS = [8, 5, 10]
        data = build_cubic_lattice_batch(ATOMS, 3., 0.1)
        ELEM = ['Fe', 'Al', 'Pd']
        DOF_reduce = 0
        self.MASSES = [MASS[_] for _ in ELEM]
        self.elem_list = [['Fe'] * ATOMS[0] ** 3 + ['Al'] * ATOMS[1] ** 3 + ['Pd'] * ATOMS[2] ** 3]
        self.masses_list = [[MASS['Fe']] * ATOMS[0] ** 3, [MASS['Al']] * ATOMS[1] ** 3, [MASS['Pd']] * ATOMS[2] ** 3]
        self.DOF_vib = [3 * ATOMS[0] ** 3 - DOF_reduce, 3 * ATOMS[1] ** 3 - DOF_reduce, 3 * ATOMS[2] ** 3 - DOF_reduce]
        self.N = [_**3 for _ in ATOMS]
        #raw_model = HarmonicLatticePotential(100., 1.)
        raw_model = SimpleSpringPotential(data.pos0, 10., )
        self.model_test = _Model_Wrapper_pyg(raw_model)
        self.data = data

    def test_Train(self):
        pass

    def test_Pred(self):
        pass

    def test_MD(self):
        """
        Test Molecular Dynamics.
        """
        # purge remaining testfiles
        logfiles = glob.glob(os.path.join('/dev/shm', 'logs/MD*.log'))
        resultfiles = glob.glob(os.path.join('/dev/shm', 'results/MD*'))
        for logfile in logfiles:
            os.remove(logfile)
        for resultfile in resultfiles:
            os.remove(resultfile)

        # static test
        data = self.data
        MASSES = self.MASSES
        elem_list = self.elem_list
        masses_list = self.masses_list
        DOF_vib = self.DOF_vib
        N = self.N
        kB = 8.617333262145e-5 # eV/K
        TEMPERATURE = 500.
        TIME_STEP = 1.5

        # runner sets
        runner_cpu_static_nve = NVE(
            TIME_STEP, 100, 0., f'/dev/shm/results/MD_STATIC_CPU', 1, device='cpu', verbose=1
        )
        runner_gpu_static_nve = NVE(
            TIME_STEP, 100, 0., f'/dev/shm/results/MD_STATIC_GPU', 1, device='cuda:0', verbose=1
        )
        runner_gpu_move_nve = NVE(
            TIME_STEP, 10000, 500., f'/dev/shm/results/MD_NVE_GPU', 1, device='cuda:0', verbose=0,
            is_compile=False
        )
        runner_cpu_csvr_nvt = NVT(
            TIME_STEP, 10000, 'CSVR', {'time_const': 100},
            TEMPERATURE, f'/dev/shm/results/MD_CSVR_CPU', 1, device='cpu', verbose=0,
            is_compile=False
        )
        runner_gpu_csvr_nvt = NVT(
            TIME_STEP, 10000, 'CSVR', {'time_const': 100},
            TEMPERATURE, f'/dev/shm/results/MD_CSVR_GPU', 1, device='cuda:0', verbose=0,
            is_compile=False
        )
        runner_cpu_lang_nvt = NVT(
            TIME_STEP, 10000, 'Langevin', {'damping_coeff': 0.01},
            TEMPERATURE, f'/dev/shm/results/MD_LANG_CPU', 1, device='cpu', verbose=0,
            is_compile=False
        )
        runner_gpu_lang_nvt = NVT(
            TIME_STEP, 10000, 'Langevin', {'damping_coeff': 0.01},
            TEMPERATURE, f'/dev/shm/results/MD_LANG_GPU', 1, device='cuda:0', verbose=0,
            is_compile=False
        )
        runner_cpu_nose_nvt = NVT(
            TIME_STEP, 10000, 'Nose-Hoover', {},
            TEMPERATURE, f'/dev/shm/results/MD_NOSE_CPU', 1, device='cpu', verbose=0
        )
        runner_gpu_nose_nvt = NVT(
            TIME_STEP, 10000, 'Nose-Hoover', {},
            TEMPERATURE, f'/dev/shm/results/MD_NOSE_GPU', 1, device='cuda:0', verbose=0
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
            #if ('CPU' in RUNNER_NAME[i]) or ('STATICE' in RUNNER_NAME[i]) or ('NVE' in RUNNER_NAME[i]): continue
            #if 'CPU' in RUNNER_NAME[i] or ('STATIC' in RUNNER_NAME[i]): continue
            _data = data.to(runner.device).clone()
            model_test = self.model_test.to(runner.device)
            print("*"*89 + f"\nNow running {RUNNER_NAME[i]} ...\n" + "*"*89 + '\n')
            with th.profiler.profile(
                    activities=[th.profiler.ProfilerActivity.CPU, th.profiler.ProfilerActivity.CUDA],
                    with_stack=False,
                    profile_memory=False,
            ) as prof:
                pass
            t_st = time.perf_counter()
            runner.reset_logger_handler(f"/dev/shm/logs/{RUNNER_NAME[i]}.log")
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
                False,
                [len(_.pos) for _ in _data.to_data_list()],
                move_to_center_freq=-1
            )
            th.cuda.synchronize()
            print(f"{RUNNER_NAME[i]} finished. Elapsed time: {(time.perf_counter() - t_st):.2f} s")
            #with open(f"/dev/shm/logs/{RUNNER_NAME[i]}.prof", "w") as f:
            #    print(
            #        prof.key_averages(group_by_stack_n=5).table(
            #            sort_by='cpu_time_total', row_limit=500, max_src_column_width=200, max_name_column_width=200
            #        ),
            #        file=f
            #    )
            #    if 'GPU' in RUNNER_NAME[i]:
            #        print('\n\n' + '*'*89 + '\n\n', file=f)
            #        print(
            #            prof.key_averages(group_by_stack_n=5).table(
            #                sort_by='cuda_time_total', row_limit=500, max_src_column_width=200, max_name_column_width=200
            #            ),
            #            file=f
            #        )
            #continue
            # validation
            fbs = read_md_traj(f"/dev/shm/results/{RUNNER_NAME[i]}")
            etol1, etol2, etol3 = [], [], []
            ene1, ene2, ene3 = [], [], []
            vel1, vel2, vel3 = [], [], []
            vacf1, vacf2, vacf3 = [], [], []
            v0_1, v0_2, v0_3 = fbs.Labels[0], fbs.Labels[1], fbs.Labels[2]
            vv1, vv2, vv3 = np.linalg.norm(v0_1)**2, np.linalg.norm(v0_2)**2, np.linalg.norm(v0_3)**2  # Label attr is actually the velocity
            ek1, ek2, ek3 = [], [], []
            mass1, mass2, mass3 = np.asarray(masses_list[0]), np.asarray(masses_list[1]), np.asarray(masses_list[2])
            max_coord1, max_coord2, max_coord3 = 0., 0., 0.

            # cut the short simulations
            if len(fbs) < 15000:
                continue

            for ibs in range(15000, len(fbs), 3):
                # potential energy
                ene1.append(fbs.Energies[ibs])
                ene2.append(fbs.Energies[ibs + 1])
                ene3.append(fbs.Energies[ibs + 2])
                vn1, vn2, vn3 = fbs.Labels[ibs], fbs.Labels[ibs + 1], fbs.Labels[ibs + 2]
                vel1.append(vn1)
                vel2.append(vn2)
                vel3.append(vn3)
                # veloc. auto-correlation func
                vacf1.append(np.sum(v0_1 * vn1)/(vv1 + 1e-20))
                vacf2.append(np.sum(v0_2 * vn2)/(vv2 + 1e-20))
                vacf3.append(np.sum(v0_3 * vn3)/(vv3 + 1e-20))
                # kinetic energy
                ek1.append(np.sum(0.5 * mass1[:, None] * vn1 * vn1 * 103.642696562621738))
                ek2.append(np.sum(0.5 * mass2[:, None] * vn2 * vn2 * 103.642696562621738))
                ek3.append(np.sum(0.5 * mass3[:, None] * vn3 * vn3 * 103.642696562621738))
                # total energy
                etol1.append(ene1[-1] + ek1[-1])
                etol2.append(ene2[-1] + ek2[-1])
                etol3.append(ene3[-1] + ek3[-1])
                # check coords converge
                max_coord1 = max(np.abs(fbs.Coords[ibs]).max(), max_coord1)
                max_coord2 = max(np.abs(fbs.Coords[ibs + 1]).max(), max_coord2)
                max_coord3 = max(np.abs(fbs.Coords[ibs + 2]).max(), max_coord3)

            # Scalar check
            print(f"Max Coordinates Range: {max_coord1, max_coord2, max_coord3}")
            TEST_TERM_NAME = [
                'Ep mean',
                'Ep var',
                'Ek mean',
                'Ek var',
                'single veloc. mean',
                'single veloc. var',
            ]
            STANDARD_VALUES = [
                [0.5 * dof * kB * TEMPERATURE for dof in DOF_vib],         # Ep mean
                [0.5 * dof * (kB * TEMPERATURE)**2 for dof in DOF_vib],    # Ep var
                [1.5 * (na - 3) * kB * TEMPERATURE for na in N],           # Ek mean
                [1.5 * (na - 3) * (kB * TEMPERATURE)**2 for na in N],      # Ek var
                [0., 0., 0.],                                              # single veloc. mean
                [kB * TEMPERATURE / _m for _m in MASSES]                   # single veloc. var
            ]
            TEST_VALUES = [
                [np.mean(np.asarray(_ep)) for _ep in (ene1, ene2, ene3)],
                [np.var(np.asarray(_ep)) for _ep in (ene1, ene2, ene3)],
                [np.mean(np.asarray(_ek)) for _ek in (ek1, ek2, ek3)],
                [np.var(np.asarray(_ek)) for _ek in  (ek1, ek2, ek3)],
                [np.mean(np.stack(_v, axis=0)) for _v in (vel1, vel2, vel3)],
                [(np.var(np.stack(_v, axis=0))*103.642696562621738) for _v in (vel1, vel2, vel3)],
            ]
            #print(f"Batch1 Potential Energy Mean: {np.mean(ene1)}, Std: {np.std(ene1)}")
            #print(f"Batch2 Potential Energy Mean: {np.mean(ene2)}, Std: {np.std(ene2)}")
            #print(f"Batch3 Potential Energy Mean: {np.mean(ene3)}, Std: {np.std(ene3)}")
            self.assertListEqual(DOF_vib, (runner.free_degree).tolist(), )
            #   Static test
            if 'STATIC' in RUNNER_NAME[i]:
                for _i, tv in enumerate(TEST_VALUES):
                    for __i, _tv in enumerate(tv):
                        try:
                            self.assertStatisticalEqual(
                                _tv,
                                float(0),
                                atol=1e-5,
                                msg=f'\n"{TEST_TERM_NAME[_i]}" Test {__i + 1} Failed:\n'
                                    f'test value: {_tv}\nstandard value: 0.'
                            )
                            print(f'"\n{TEST_TERM_NAME[_i]}" Test {__i + 1} passed. <<<<<')
                        except AssertionError:
                            print(f'\n"{TEST_TERM_NAME[_i]}" Test {__i + 1} Failed:\n'
                                    f'test value: {_tv}\nstandard value: 0.')
                continue
            if 'NVE' in RUNNER_NAME[i]:
                for _i, _etol in enumerate((etol1, etol2, etol3)):
                    _etol_var = len(_etol) * (max(_etol) - min(_etol))/sum(_etol)
                    try:
                        self.assertStatisticalEqual(
                            _etol_var,
                            0.,
                            atol=1e-3,
                            msg=f'\n"NVE Energy" Test {_i + 1} Failed:\n'
                                f'test value: {_etol_var}\nstandard value: 0.'
                        )
                        print(f"Mean Ep: {TEST_VALUES[1]}, STD Ep: {TEST_VALUES[2]}")
                        print(f'\n"NVE Energy" Test {_i + 1} passed. <<<<<')
                    except AssertionError:
                        print(f'\n"NVE Energy" Test {_i + 1} Failed:\n'
                                f'test value: {_etol_var}\nstandard value: 0.')
                continue

            # NVT test
            for _i, tv in enumerate(TEST_VALUES):
                for __i, _tv in enumerate(tv):
                    try:
                        self.assertStatisticalEqual(
                            _tv,
                            STANDARD_VALUES[_i][__i],
                            atol=1e-1,
                            msg=f'\n"{TEST_TERM_NAME[_i]}" Test {__i + 1} Failed:\n'
                                f'test value: {_tv}\nstandard value: {STANDARD_VALUES[_i][__i]}'
                        )
                        print(f'\n"{TEST_TERM_NAME[_i]}" Test {__i + 1} passed. <<<<<')
                    except AssertionError:
                        print(f'\n"{TEST_TERM_NAME[_i]}" Test {__i + 1} Failed:\n'
                                f'test value: {_tv}\nstandard value: {STANDARD_VALUES[_i][__i]}')

        pass

    def test_MC(self):
        # purge remaining testfiles
        logfiles = glob.glob(os.path.join('/dev/shm', 'logs/MC*.log'))
        resultfiles = glob.glob(os.path.join('/dev/shm', 'results/MC*'))
        for logfile in logfiles:
            os.remove(logfile)
        for resultfile in resultfiles:
            os.remove(resultfile)

        # static test
        data = self.data
        MASSES = self.MASSES
        elem_list = self.elem_list
        masses_list = self.masses_list
        DOF_vib = self.DOF_vib
        N = self.N
        kB = 8.617333262145e-5  # eV/K
        TEMPERATURE = 500.
        TIME_STEP = 1.5

        # runner sets
        runner_cpu_nvt = MMC(
            'Gaussian',
            30000,
            TEMPERATURE,
            'constant',
            1,
            None,
            0.07,
            f'/dev/shm/results/MC_GAUSS_NVT_CPU',
            1,
            device='cpu',
            verbose=0,
            is_compile=True
        )
        runner_gpu_nvt = MMC(
            'Gaussian',
            30000,
            TEMPERATURE,
            'constant',
            1,
            None,
            0.07,
            f'/dev/shm/results/MC_GAUSS_NVT_GPU',
            1,
            device='cuda:0',
            verbose=0,
            is_compile = True
        )
        runner_gpu_anneal = MMC(
            'Gaussian',
            30000,
            TEMPERATURE,
            'fast',
            1,
            None,
            0.07,
            f'/dev/shm/results/MC_GAUSS_ANNEAL_GPU',
            1,
            device='cuda:0',
            verbose=0
        )

        RUNNER_NAME = [
            'MC_GAUSS_NVT_CPU',
            'MC_GAUSS_NVT_GPU',
            'MC_GAUSS_ANNEAL_GPU',
        ]
        import matplotlib.pyplot as plt
        for i, runner in enumerate([
            runner_cpu_nvt,
            runner_gpu_nvt,
            runner_gpu_anneal,
        ]):
            # if ('CPU' in RUNNER_NAME[i]) or ('STATICE' in RUNNER_NAME[i]) or ('NVE' in RUNNER_NAME[i]): continue
            #if 'CPU' in RUNNER_NAME[i] or ('STATIC' in RUNNER_NAME[i]): continue
            _data = data.to(runner.device).clone()
            model_test = self.model_test.to(runner.device)
            print("*" * 89 + f"\nNow running {RUNNER_NAME[i]} ...\n" + "*" * 89 + '\n')
            t_st = time.perf_counter()
            runner.reset_logger_handler(f"/dev/shm/logs/{RUNNER_NAME[i]}.log")
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
            fbs = read_mc_traj(f"/dev/shm/results/{RUNNER_NAME[i]}")
            ene1, ene2, ene3 = ([_ for _ in fbs.Energies[0::3]],
                                [_ for _ in fbs.Energies[1::3]],
                                [_ for _ in fbs.Energies[2::3]])
            STANDARD_VALUES = [
                [0.5 * dof * kB * TEMPERATURE for dof in DOF_vib],  # Ep mean
                [0.5 * dof * (kB * TEMPERATURE) ** 2 for dof in DOF_vib],  # Ep var
            ]
            for _i, _en in enumerate((ene1, ene2, ene3)):
                #plt.plot(_en)
                #plt.show()
                #plt.clf()
                _mean_val = np.mean(_en[15000:])
                _std_val = np.std(_en[15000:])
                try:
                    self.assertStatisticalEqual(_mean_val, STANDARD_VALUES[0][_i])
                    self.assertStatisticalEqual(_std_val, STANDARD_VALUES[1][_i])
                    print(f"Mean Ep: {_mean_val}, STD Ep: {_std_val}")
                    print(f'\n"MC Energy" Test {_i + 1} passed. <<<<<')
                except AssertionError:
                    print(f'\n"MC Energy" Test {i+ 1} Failed:\n'
                          f'test value:\n\tenergy mean: {_mean_val}\n\tenergy std: {_std_val}'
                          f'\nstandard value:\n\tenergy mean: {STANDARD_VALUES[0][_i]}\n\tenergy std: {STANDARD_VALUES[1][_i]}\n')


    def test_OPT(self):
        pass

    def test_TS(self):
        pass

    def test_NEB(self):
        pass

    def test_CMD(self):
        pass


if __name__ == '__main__':
    test = MainTest()
    test.test_MD()
