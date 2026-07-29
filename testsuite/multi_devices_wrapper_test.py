#!/usr/bin/env python3
"""Run single- and multi-device FIRE/NVE checks for the PyG model wrapper.

Example:
    python testsuite/multi_devices_wrapper_test.py --devices cuda:0 cuda:1 --batch-size 16 --steps 20
"""
import argparse
import copy
import time
import sys
from pathlib import Path

import torch as th

# Prefer this checkout over any separately installed BUCToolkit version.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from BUCToolkit.BatchMD.NVE import NVE
from BUCToolkit.BatchOptim.minimize.FIRE import FIRE
from BUCToolkit.BatchStructures.batch import Batch
from BUCToolkit.BatchStructures.data import Data
from BUCToolkit.utils import FatalError
from BUCToolkit.utils.model_wrappers.multi_devices_wrappers import Model_Wrapper_pyg_MultiDevice
from BUCToolkit.utils.model_wrappers.pyg_model_wrappers import Model_Wrapper_pyg


class SimpleSpringPotential(th.nn.Module):
    """Independent harmonic wells with exact energy and force outputs."""
    def forward(self, data):
        displace = data.pos - data.pos0
        energy = th.zeros(data.num_graphs, dtype=data.pos.dtype, device=data.pos.device)
        energy.index_add_(0, data.batch, 0.5 * th.sum(displace ** 2, dim=-1))
        return {'energy': energy, 'forces': -displace}


class FailingPotential(th.nn.Module):
    def forward(self, data):
        raise RuntimeError('intentional worker failure')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', default=None, help='CUDA devices, e.g. cuda:0 cuda:1')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--min-atoms', type=int, default=64)
    parser.add_argument('--max-atoms', type=int, default=256)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup-steps', type=int, default=2)
    return parser.parse_args()


def synchronize(devices):
    for device in devices:
        th.cuda.synchronize(device)


def make_graph(batch_size, min_atoms, max_atoms, device):
    generator = th.Generator(device='cpu').manual_seed(20260729)
    data_list = list()
    atom_counts = list()
    for i in range(batch_size):
        # Deterministic irregular sizes exercise LPT assignment and restoration.
        n_atom = min_atoms + (i * 37) % (max_atoms - min_atoms + 1)
        pos0 = th.randn((n_atom, 3), generator=generator)
        pos = pos0 + 0.1 * th.randn((n_atom, 3), generator=generator)
        data_list.append(Data(pos=pos, pos0=pos0))
        atom_counts.append(n_atom)
    return Batch.from_data_list(data_list).to(device), atom_counts


def make_input(graph):
    return graph.pos.detach().clone().unsqueeze(0)


def run_fire(wrapper, graph, atom_counts, device, steps):
    runner = FIRE(
        E_threshold=0., F_threshold=0., maxiter=steps, steplength=0.01,
        output_file=None, device=device, verbose=0, _hold_samples=True,
    )
    return runner.run(
        wrapper.Energy, make_input(graph), grad_func=wrapper.Grad,
        func_args=(graph,), grad_func_args=(graph,),
        is_grad_func_contain_y=False, require_grad=False, output_grad=True,
        batch_indices=atom_counts,
    )


def run_nve(wrapper, graph, atom_counts, device, steps):
    runner = NVE(
        time_step=0.001, max_step=steps, T_init=0., output_file=None,
        output_structures_per_step=steps + 1, device=device, verbose=0,
    )
    runner.run(
        wrapper.Energy, make_input(graph), Element_list=[[1] * sum(atom_counts)],
        V_init=th.zeros_like(make_input(graph)), grad_func=wrapper.Grad,
        func_args=(graph,), grad_func_args=(graph,),
        is_grad_func_contain_y=False, require_grad=False,
        batch_indices=atom_counts,
    )


def timed(call, devices):
    # CUDA launches are asynchronous; synchronize both timing boundaries so
    # elapsed time covers the complete native FIRE/NVE run.
    synchronize(devices)
    start = time.perf_counter()
    result = call()
    synchronize(devices)
    return result, time.perf_counter() - start


def check_fatal_error(devices, graph):
    # A worker failure must close the wrapper and reject the next batch.
    wrapper = Model_Wrapper_pyg_MultiDevice(FailingPotential(), devices_list=devices)
    try:
        try:
            wrapper.Energy(make_input(graph), graph)
        except FatalError:
            pass
        else:
            raise AssertionError('A worker exception did not raise FatalError.')
        try:
            wrapper.Energy(make_input(graph), graph)
        except FatalError:
            pass
        else:
            raise AssertionError('A closed wrapper accepted another Energy call.')
    finally:
        wrapper.close()


def main():
    args = parse_args()
    if not th.cuda.is_available() or th.cuda.device_count() < 2:
        raise SystemExit('This script requires at least two visible CUDA devices.')
    devices = args.devices or [f'cuda:{i}' for i in range(th.cuda.device_count())]
    if len(devices) < 2:
        raise SystemExit('Please provide at least two CUDA devices.')
    master_device = devices[0]

    graph, atom_counts = make_graph(
        args.batch_size, args.min_atoms, args.max_atoms, master_device
    )
    base_model = SimpleSpringPotential().eval()
    single_wrapper = Model_Wrapper_pyg(copy.deepcopy(base_model).to(master_device))
    multi_wrapper = Model_Wrapper_pyg_MultiDevice(copy.deepcopy(base_model), devices_list=devices)
    multi_wrapper.eval()

    try:
        # Warmup is deliberately outside both timed regions.
        for _ in range(args.warmup_steps):
            run_fire(single_wrapper, graph.clone(), atom_counts, master_device, 1)
            run_fire(multi_wrapper, graph.clone(), atom_counts, master_device, 1)

        single_fire, single_fire_time = timed(
            lambda: run_fire(single_wrapper, graph.clone(), atom_counts, master_device, args.steps),
            [master_device],
        )
        multi_fire, multi_fire_time = timed(
            lambda: run_fire(multi_wrapper, graph.clone(), atom_counts, master_device, args.steps),
            devices,
        )
        energy_error = th.max(th.abs(single_fire[0] - multi_fire[0])).item()
        coordinate_error = th.max(th.abs(single_fire[1] - multi_fire[1])).item()
        gradient_error = th.max(th.abs(single_fire[2] - multi_fire[2])).item()
        if max(energy_error, coordinate_error, gradient_error) > 1.e-5:
            raise AssertionError(
                f'FIRE results differ: energy={energy_error}, X={coordinate_error}, grad={gradient_error}.'
            )

        _, single_nve_time = timed(
            lambda: run_nve(single_wrapper, graph.clone(), atom_counts, master_device, args.steps),
            [master_device],
        )
        _, multi_nve_time = timed(
            lambda: run_nve(multi_wrapper, graph.clone(), atom_counts, master_device, args.steps),
            devices,
        )
        check_fatal_error(devices, graph.clone())

        print(f'devices: {devices}')
        print(f'batch_size: {args.batch_size}, total_atoms: {sum(atom_counts)}')
        print(f'FIRE single: {single_fire_time:.6f} s, multi: {multi_fire_time:.6f} s, speedup: {single_fire_time / multi_fire_time:.3f}x')
        print(f'NVE  single: {single_nve_time:.6f} s, multi: {multi_nve_time:.6f} s, speedup: {single_nve_time / multi_nve_time:.3f}x')
        print(f'FIRE max errors: energy={energy_error:.3e}, X={coordinate_error:.3e}, grad={gradient_error:.3e}')
    finally:
        multi_wrapper.close()


if __name__ == '__main__':
    main()
