#  Copyright (c) 2026.5.18, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: VASP_model_wrapper.py
#  Environment: Python 3.12

import os
import subprocess
import shutil
import torch as th

import BUCToolkit as bt
from BUCToolkit.utils.function_utils import _BaseWrapper, compare_tensors


class VASP_Model(_BaseWrapper):
    """
    Model wrapper for ab initio calculations using VASP.
    It can be directly used in methods of APIs

    Args:
        input_path: The path to the standard input files for VASP static calculation.
            Including `INCAR` `POSCAR` `KPOINTS` `POTCAR`, etc.
        submit_script: the file name of the script to execute. It must be at the `input_path`.
        is_reuse_WAVECAR: whether to reuse WAVECAR file from last step output as the initial guess. (use move instead of copy)
        use_slurm: whether to submit the script with ``sbatch --wait`` instead
            of executing it directly.

    """

    def __init__(self, input_path: str, submit_script: str, is_reuse_WAVECAR=True, use_slurm=False) -> None:
        super().__init__(input_path)
        if not os.path.isfile(os.path.join(input_path, 'INCAR')):
            raise FileNotFoundError(f"INCAR file not found at {input_path}.")
        elif not os.path.isfile(os.path.join(input_path, 'POSCAR')):
            raise FileNotFoundError(f"POSCAR file not found at {input_path}.")
        elif not os.path.isfile(os.path.join(input_path, 'POTCAR')):
            raise FileNotFoundError(f"POTCAR file not found at {input_path}.")
        elif not os.path.isfile(os.path.join(input_path, 'KPOINTS')):
            raise FileNotFoundError(f"KPOINTS file not found at {input_path}.")

        if not os.path.isfile(os.path.join(input_path, submit_script)):
            raise FileNotFoundError(f"submit_script file {submit_script} not found at {input_path}.")

        self.submit_script = submit_script
        self.input_path = os.path.abspath(input_path)
        self.step = 0  # steps of VASP static calculation
        self.is_reuse_WAVECAR = bool(is_reuse_WAVECAR)
        self.use_slurm = bool(use_slurm)
        self._submit_script_path = os.path.abspath(os.path.join(input_path, submit_script))
        self.data: bt.Structures | None = None  # data storage container
        self._X_check_cache = None  # used for check the consistency of input X in `Energy` and `Grad`, ensuring the correctness of F_cache
        self.F_tensor: th.Tensor | None = None  # A cache

        # initialize 1st step
        os.makedirs(os.path.join(self.input_path, str(self.step)), exist_ok=True)
        shutil.copy(os.path.join(self.input_path, "POSCAR"), os.path.join(self.input_path, str(self.step), "POSCAR"))

    def _submit_task(self):
        """
        submit VASP in subprocess and read information after task done.
        Update the self.data
        Returns: None

        """
        os.makedirs(os.path.join(self.input_path, str(self.step)), exist_ok=True)
        file_list = ['INCAR', 'KPOINTS', 'POTCAR']
        for inpf in file_list:
            shutil.copy(os.path.join(self.input_path, inpf), os.path.join(self.input_path, str(self.step), inpf))
        step_path = os.path.join(self.input_path, str(self.step))
        step_script = os.path.join(step_path, os.path.basename(self._submit_script_path))
        shutil.copy2(self._submit_script_path, step_script)
        if self.is_reuse_WAVECAR:
            if (
                    os.path.isdir(os.path.join(self.input_path, f"{self.step - 1}"))
            ) and (
                            os.path.isfile(os.path.join(self.input_path, f"{self.step - 1}", 'WAVECAR'))
            ):  # last step WAVECAR
                shutil.move(
                    os.path.join(self.input_path, f"{self.step - 1}", 'WAVECAR'),
                    os.path.join(self.input_path, str(self.step), 'WAVECAR')
                )

        if self.use_slurm:
            command = ['sbatch', '--wait', f"./{os.path.basename(step_script)}"]
        else:
            command = [f"./{os.path.basename(step_script)}"]
        res = subprocess.run(
            command,
            capture_output=True,
            cwd=step_path,
            text=True
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"VASP computation failed in step {self.step}: {command!r}\n"
                f"stdout: {res.stdout}\nstderr: {res.stderr}"
            )

        outcar_path = os.path.join(step_path, 'OUTCAR')
        if not os.path.isfile(outcar_path):
            raise RuntimeError(
                f"VASP command completed without creating {outcar_path}. "
                f"stdout: {res.stdout}\nstderr: {res.stderr}"
            )
        self.data = bt.io.OUTCAR2Feat(step_path, verbose=0)  # text file reader & parser
        self.data.read(['OUTCAR'], n_core=1)

        self.step += 1

    def Energy(self, X) -> th.Tensor:

        if self._X_check_cache is None:
            # first step
            self._submit_task()
            self._X_check_cache = X

        else:
            self.data.Coords[-1] = X.squeeze(0).numpy(force=True)  # read updated X
            self.data[-1].write2text(f"{self.input_path}/{self.step}", file_format='POSCAR', file_name_list=['POSCAR'], n_core=1)
            self._submit_task()
            self._X_check_cache = X

        E_tensor = th.as_tensor(self.data.Energies[-1:], device=X.device)
        self.F_tensor = th.as_tensor(self.data.Forces[-1], device=X.device).unsqueeze(0)

        return E_tensor

    def Grad(self, X) -> th.Tensor:
        origin_shape = X.shape
        if self._X_check_cache is None or (not compare_tensors(self._X_check_cache, X)):
            self.F_tensor = None

        if self.F_tensor is None:
            if self.data is not None:
                self.data.Coords[-1] = X.squeeze(0).numpy(force=True)
                self.data[-1].write2text(
                    f"{self.input_path}/{self.step}",
                    file_format='POSCAR',
                    file_name_list=['POSCAR'],
                    n_core=1,
                )
            self._submit_task()
            F_tensor = th.as_tensor(self.data.Forces[-1:], device=X.device)  # not cache
            return - F_tensor.reshape(origin_shape).contiguous()

        else:
            F_tensor = self.F_tensor
            self.F_tensor = None  # empty cache
            return - F_tensor.reshape(origin_shape).contiguous()
