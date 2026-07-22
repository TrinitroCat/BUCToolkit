"""Focused regressions for shared MD/MC/optimization trajectory handling."""

import io
import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import torch as th

from BUCToolkit.BatchMC import MMC
from BUCToolkit.BatchMD.ConstrNVE import ConstrNVE
from BUCToolkit.BatchMD.ConstrNVT import ConstrNVT
from BUCToolkit.BatchMD.NVE import NVE
from BUCToolkit.BatchMD.NVT import NVT
from BUCToolkit.BatchOptim.minimize.FIRE import FIRE
from BUCToolkit.BatchStructures import read_dump_arrays, read_mc_traj
from BUCToolkit.BatchStructures.StructuresIO import ArrayDumper, ArrayDumpReader
from BUCToolkit.Bases.BaseMotion import BaseMotion
from BUCToolkit.Bases.StdContainer import StdContainer
from BUCToolkit.Postprocessing import MDTrajectory


def _harmonic_energy(X: th.Tensor, X0: th.Tensor) -> th.Tensor:
    return 0.5 * th.sum((X - X0) ** 2, dim=(-2, -1))


def _harmonic_grad(X: th.Tensor, energy: th.Tensor, X0: th.Tensor) -> th.Tensor:
    return X - X0


class MotionIORegressionTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix='buc_motion_io_')

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _path(self, name: str) -> str:
        return os.path.join(self._tmpdir, name)

    def test_md_array_log_uses_registered_names_for_irregular_batch(self):
        """Array logs have field headings and never stringify tensors as names."""
        runner = NVE(0.1, 1, output_file=None, device='cpu', verbose=2)
        stream = io.StringIO()
        logger = logging.getLogger(f'{__name__}.md_array_log.{id(runner)}')
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        runner.logger = logger

        X = th.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
        runner.run(
            lambda coo: th.stack((
                th.sum(coo[:, :1] ** 2),
                th.sum(coo[:, 1:] ** 2),
            )),
            X,
            [['H', 'H', 'H']],
            V_init=th.zeros_like(X),
            grad_func=lambda coo, energy: 2 * coo,
            is_grad_func_contain_y=True,
            batch_indices=(1, 2),
        )
        handler.flush()
        output = stream.getvalue()

        self.assertEqual(runner.batch_slice_indx, [0, 1, 3])
        self.assertNotIn('Failed to logout', output)
        self.assertIn('Step:            0', output)
        self.assertIn(' X:\n', output)
        self.assertIn(' V:\n', output)
        self.assertNotIn('tensor(', output)

        stream.seek(0)
        stream.truncate(0)
        runner.verbose = 1
        runner.run(
            lambda coo: th.stack((
                th.sum(coo[:, :1] ** 2),
                th.sum(coo[:, 1:] ** 2),
            )),
            X,
            [['H', 'H', 'H']],
            V_init=th.zeros_like(X),
            grad_func=lambda coo, energy: 2 * coo,
            is_grad_func_contain_y=True,
            batch_indices=(1, 2),
        )
        handler.flush()
        scalar_output = stream.getvalue()
        logger.removeHandler(handler)
        handler.close()

        self.assertIn('Energy', runner.get_log_vars())
        self.assertIn('X', runner.get_log_vars())
        self.assertIn('V', runner.get_log_vars())
        self.assertIn('Step:            0', scalar_output)
        self.assertIn('\tEnergy       = ', scalar_output)
        self.assertNotIn(' X:\n', scalar_output)
        self.assertNotIn(' V:\n', scalar_output)

    def test_array_log_splits_only_flattened_irregular_atom_data(self):
        """Batch-leading and unrelated arrays bypass irregular atom slicing."""
        logger = mock.Mock()
        atomwise = th.arange(9.).reshape(1, 3, 3)
        batch_leading = th.arange(6.).reshape(2, 3)
        other = th.arange(8.).reshape(4, 2)

        with mock.patch(
                'BUCToolkit.Bases.BaseMotion.np.split', wraps=np.split
        ) as split:
            NVE.handle_arrays_print(
                logger,
                batch_indices=[1, 2],
                batch_slice_indx=[0, 1, 3],
                arrays=[[atomwise, batch_leading, other]],
                array_names=[['X', 'Fc', 'other']],
                verbose=2,
            )

        self.assertEqual(split.call_count, 1)
        self.assertEqual(split.call_args.args[1], [1])
        self.assertEqual(split.call_args.kwargs, {'axis': 1})
        self.assertEqual(
            [
                call.args[0] for call in logger.info.call_args_list
                if call.args[0].endswith(':\n')
            ],
            [' X:\n', ' Fc:\n', ' other:\n'],
        )

    def test_md_silent_run_transfers_only_dump_quantities(self):
        """Silent MD runs do not allocate buffers for log-only quantities."""
        runner = NVE(
            0.1, 1, output_file=None, device='cpu', verbose=0
        )
        coordinates = th.zeros((1, 2, 3), dtype=th.float32)

        with mock.patch(
                'BUCToolkit.BatchMD._BaseMD._BaseMD._allocate_cpu_buffers',
                wraps=BaseMotion._allocate_cpu_buffers,
        ) as allocate_buffers:
            runner.run(
                lambda value: th.sum(value ** 2, dim=(-2, -1)),
                coordinates,
                [['H', 'H']],
                V_init=th.zeros_like(coordinates),
                grad_func=lambda value, energy: 2 * value,
                is_grad_func_contain_y=True,
            )

        transferred_names = allocate_buffers.call_args.args[1]
        self.assertEqual(
            set(transferred_names), {'Energy', 'X', 'V', 'Force'}
        )

    def test_md_ensembles_request_only_required_kinetic_updates(self):
        """Each ensemble controls its own shared kinetic-energy dependency."""
        coordinates = th.tensor(
            [[[0., 0., 0.], [1., 0., 0.]]], dtype=th.float32
        )
        velocities = th.tensor(
            [[[0.1, 0., 0.], [-0.1, 0., 0.]]], dtype=th.float32
        )
        energy_func = lambda value: th.sum(value ** 2, dim=(-2, -1))
        grad_func = lambda value, energy: 2 * value

        runners = (
            (
                'NVE',
                NVE(
                    0.01, 2, output_file=None,
                    device='cpu', verbose=0,
                ),
                0,
                False,
            ),
            (
                'Langevin',
                NVT(
                    0.01, 2, 'Langevin', {'damping_coeff': 0.01},
                    output_file=None, device='cpu', verbose=0,
                ),
                0,
                False,
            ),
            (
                'VR',
                NVT(
                    0.01, 2, 'VR', output_file=None,
                    device='cpu', verbose=0,
                ),
                2,
                True,
            ),
            (
                'CSVR',
                NVT(
                    0.01, 2, 'CSVR', {'time_const': 0.1},
                    output_file=None, device='cpu', verbose=0,
                ),
                2,
                False,
            ),
        )
        for name, runner, expected_reductions, expected_requirement in runners:
            with self.subTest(ensemble=name):
                with mock.patch.object(
                        runner,
                        '_reduce_Ek_T',
                        wraps=runner._reduce_Ek_T,
                ) as reduce_Ek_T:
                    runner.run(
                        energy_func,
                        coordinates,
                        [['H', 'H']],
                        V_init=velocities,
                        grad_func=grad_func,
                        is_grad_func_contain_y=True,
                    )

                self.assertEqual(reduce_Ek_T.call_count, expected_reductions)
                self.assertIs(
                    runner.require_Ek_update, expected_requirement
                )

        periodic_runner = NVE(
            0.01,
            5,
            output_file=None,
            output_structures_per_step=2,
            device='cpu',
            verbose=1,
        )
        with mock.patch.object(
                periodic_runner,
                '_reduce_Ek_T',
                wraps=periodic_runner._reduce_Ek_T,
        ) as reduce_Ek_T:
            periodic_runner.run(
                energy_func,
                coordinates,
                [['H', 'H']],
                V_init=velocities,
                grad_func=grad_func,
                is_grad_func_contain_y=True,
            )
        self.assertEqual(reduce_Ek_T.call_count, 3)
        self.assertFalse(periodic_runner.require_Ek_update)

    def test_mc_configurable_named_quantities_cpu(self):
        """MC uses named state fields and preserves boolean dump dtypes."""
        output_file = self._path('mc.bin')
        runner = MMC(
            maxiter=4,
            temperature_init=300.,
            coordinate_update_param=0.01,
            output_file=output_file,
            output_structures_per_step=1,
            device='cpu',
            verbose=0,
            dump_quantities=(
                'Energy', 'X', 'delta_E', 'is_accept', 'temperature'
            ),
        )
        X = th.zeros((2, 3, 3), dtype=th.float32)
        runner.run(
            lambda coo: th.sum(coo ** 2, dim=(-2, -1)),
            X,
            [['H'] * 3, ['H'] * 3],
        )

        reader = ArrayDumpReader(output_file)
        reader.read()
        self.assertEqual(
            reader.names,
            {
                0: ['cell_vec', 'atomic_numbers', 'fixed_mask'],
                1: ['Energy', 'X', 'delta_E', 'is_accept', 'temperature'],
            },
        )
        raw = read_dump_arrays(output_file)
        self.assertEqual(
            {'Energy', 'X', 'delta_E', 'is_accept', 'temperature'},
            set(raw) - {'cell_vec', 'atomic_numbers', 'fixed_mask'},
        )
        self.assertEqual(raw['is_accept'][0].dtype, np.dtype(np.bool_))
        self.assertEqual(len(raw['Energy']), 4)

        split = read_mc_traj(output_file, out_arrays=True)
        for name in ('Energy', 'X', 'delta_E', 'is_accept', 'temperature'):
            self.assertEqual(len(split[name]), 8)

    def test_irregular_reader_keeps_per_structure_matrices(self):
        """A 2-D per-structure field must not be split by atom pointers."""
        output_file = self._path('irregular.bin')
        dumper = ArrayDumper(output_file, mode='x')

        batch_indices = np.array([2, 3], dtype=np.int64)
        cells = np.zeros((2, 3, 3), dtype=np.float32)
        atomic_numbers = np.array([1, 1, 8, 1, 1], dtype=np.int64)
        fixed_mask = np.ones((1, 5, 3), dtype=np.float32)
        dumper.start_from_arrays(
            1, batch_indices, cells, atomic_numbers, fixed_mask,
            names=('batch_indices', 'cell_vec', 'atomic_numbers', 'fixed_mask'),
        )
        dumper.step(batch_indices, cells, atomic_numbers, fixed_mask)

        energies = np.zeros(2, dtype=np.float32)
        coordinates = np.zeros((1, 5, 3), dtype=np.float32)
        constraint_forces = np.arange(4, dtype=np.float32).reshape(2, 2)
        dumper.start_from_arrays(
            2,
            energies,
            coordinates,
            constraint_forces,
            names=('Energy', 'X', 'Fc'),
        )
        dumper.step(energies, coordinates, constraint_forces)
        dumper.step(energies + 1, coordinates + 1, constraint_forces + 10)
        dumper.close()

        split = read_mc_traj(output_file, out_arrays=True)
        self.assertEqual(
            [value.tolist() for value in split['Fc']],
            [[0., 1.], [2., 3.], [10., 11.], [12., 13.]],
        )
        self.assertEqual(
            [value.shape for value in split['X']],
            [(2, 3), (3, 3), (2, 3), (3, 3)],
        )

    def test_mc_irregular_batch_writes_ragged_atomic_numbers(self):
        """MC flattens irregular element rows and restores them on read."""
        output_file = self._path('mc_irregular.bin')
        runner = MMC(
            maxiter=2,
            temperature_init=300.,
            coordinate_update_param=0.01,
            output_file=output_file,
            output_structures_per_step=1,
            device='cpu',
            verbose=0,
        )
        X = th.zeros((1, 5, 3), dtype=th.float32)
        runner.run(
            lambda coo: th.stack((
                th.sum(coo[:, :2] ** 2),
                th.sum(coo[:, 2:] ** 2),
            )),
            X,
            [['H', 'H'], ['O', 'H', 'H']],
            batch_indices=(2, 3),
            move_to_center_freq=1,
        )

        raw = read_dump_arrays(output_file)
        self.assertEqual(raw['atomic_numbers'].tolist(), [1, 1, 8, 1, 1])
        split = read_mc_traj(output_file, out_arrays=True)
        self.assertEqual(
            [value.shape for value in split['X']],
            [(2, 3), (3, 3), (2, 3), (3, 3)],
        )

    def test_mc_silent_run_transfers_only_dump_quantities(self):
        """Silent MC runs do not allocate buffers for log-only quantities."""
        runner = MMC(
            maxiter=1,
            temperature_init=300.,
            coordinate_update_param=0.01,
            output_file=None,
            device='cpu',
            verbose=0,
        )
        coordinates = th.zeros((1, 2, 3), dtype=th.float32)

        with mock.patch(
                'BUCToolkit.BatchMC._BaseMC._BaseMC._allocate_cpu_buffers',
                wraps=BaseMotion._allocate_cpu_buffers,
        ) as allocate_buffers:
            runner.run(
                lambda value: th.sum(value ** 2, dim=(-2, -1)),
                coordinates,
                [['H', 'H']],
            )

        transferred_names = allocate_buffers.call_args.args[1]
        self.assertEqual(set(transferred_names), {'Energy', 'X'})

    def test_mc_proposal_validation_is_moved_outside_the_loop(self):
        """Validated MC state uses lightweight per-step distributions."""
        runner = MMC(
            maxiter=2,
            temperature_init=300.,
            coordinate_update_param=0.01,
            output_file=None,
            device='cpu',
            verbose=0,
        )
        coordinates = th.zeros((1, 2, 3), dtype=th.float32)

        with mock.patch(
                'BUCToolkit.BatchMC.MetropolisMC.th.distributions.Normal',
                wraps=th.distributions.Normal,
        ) as normal_distribution:
            runner.run(
                lambda value: th.sum(value ** 2, dim=(-2, -1)),
                coordinates,
                [['H', 'H']],
            )

        self.assertEqual(normal_distribution.call_count, 2)
        for call in normal_distribution.call_args_list:
            self.assertIs(call.kwargs['validate_args'], False)
        self.assertEqual(runner._proposal_scale.shape, (1, 1, 1))

    def test_mc_proposal_initialization_rejects_invalid_state(self):
        """Initialization safeguards replace repeated distribution checks."""
        invalid_parameters = (
            {'temperature_init': 0.},
            {'temperature_update_freq': 0},
            {'coordinate_update_param': 0.},
            {
                'temperature_scheme': 'exponential',
                'temperature_scheme_param': 0.,
            },
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    MMC(maxiter=1, output_file=None, **parameters)

        runner = MMC(maxiter=1, output_file=None, device='cpu', verbose=0)
        coordinates = th.zeros((1, 1, 3), dtype=th.float32)
        with self.assertRaisesRegex(ValueError, 'movable degree'):
            runner.run(
                lambda value: th.sum(value ** 2, dim=(-2, -1)),
                coordinates,
                [['H']],
                fixed_atom_tensor=th.zeros_like(coordinates),
            )

        with self.assertRaisesRegex(ValueError, 'finite coordinates'):
            runner.run(
                lambda value: th.sum(value ** 2, dim=(-2, -1)),
                th.full_like(coordinates, float('nan')),
                [['H']],
            )

    def test_cpu_optimizer_final_dump_matches_returned_state(self):
        """The max-iteration frame contains the post-update CPU state."""
        output_file = self._path('opt.bin')
        X0 = th.zeros((1, 1, 3), dtype=th.float32)
        X = th.tensor([[[0.5, 0.3, 0.1]]], dtype=th.float32)
        runner = FIRE(
            E_threshold=0.,
            F_threshold=0.,
            maxiter=1,
            steplength=0.1,
            output_file=output_file,
            device='cpu',
            verbose=0,
        )
        runner.set_system_info(atomic_numbers=[[1]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            energies, coordinates, gradients = runner.run(
                _harmonic_energy,
                X,
                grad_func=_harmonic_grad,
                func_args=(X0,),
                grad_func_args=(X0,),
                is_grad_func_contain_y=True,
                output_grad=True,
            )

        raw = read_dump_arrays(output_file)
        self.assertTrue(np.allclose(raw['Energy'][-1], energies.numpy()))
        self.assertTrue(np.allclose(raw['X'][-1], coordinates.numpy()))
        self.assertTrue(np.allclose(raw['Force'][-1], (-gradients).numpy()))

    def test_optimizer_system_info_validates_and_normalizes_metadata(self):
        """Optimizer metadata is validated before replacing retained values."""
        runner = FIRE(maxiter=1, output_file=None, device='cpu', verbose=0)
        cells = np.zeros((2, 3, 3), dtype=np.float64)
        runner.set_system_info(
            cell_vec=cells,
            atomic_numbers=[['C'], ['O']],
        )

        self.assertEqual(runner.atomic_numbers, [[6], [8]])
        self.assertEqual(runner.cell_vec.dtype, np.float32)
        cells[0, 0, 0] = 1.
        self.assertEqual(runner.cell_vec[0, 0, 0], 0.)

        # Numeric Python containers use NumPy's normal conversion rules.
        runner.set_system_info(cell_vec=np.zeros((2, 3, 3)).tolist())
        self.assertEqual(runner.cell_vec.dtype, np.float32)
        runner.set_system_info(atomic_numbers=[[np.int64(6)], [8.0]])
        self.assertEqual(runner.atomic_numbers, [[6], [8]])

        invalid_values = (
            ({'atomic_numbers': ['H']}, TypeError),
            ({'atomic_numbers': [('H',)]}, TypeError),
            ({'atomic_numbers': [[]]}, ValueError),
            ({'atomic_numbers': [['Unknown']]}, ValueError),
            ({'atomic_numbers': [[87]]}, ValueError),
            ({'cell_vec': np.zeros((3, 3))}, ValueError),
            ({'cell_vec': np.full((1, 3, 3), np.nan)}, ValueError),
            ({
                'cell_vec': np.zeros((2, 3, 3)),
                'atomic_numbers': [['H']],
            }, ValueError),
        )
        for kwargs, error_type in invalid_values:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error_type):
                    runner.set_system_info(**kwargs)

        # A failed update must not partially replace previously valid metadata.
        self.assertEqual(runner.atomic_numbers, [[6], [8]])
        self.assertEqual(runner.cell_vec.shape, (2, 3, 3))

    def test_optimizer_system_info_must_match_run_layout(self):
        """Stored element rows must match the coordinates of the next run."""
        runner = FIRE(maxiter=1, output_file=None, device='cpu', verbose=0)
        runner.set_system_info(atomic_numbers=[['H', 'H']])
        coordinates = th.zeros((1, 1, 3), dtype=th.float32)

        with self.assertRaisesRegex(ValueError, 'run layout'):
            runner.run(
                lambda value: th.sum(value ** 2, dim=(-2, -1)),
                coordinates,
                grad_func=lambda value, energy: 2 * value,
                is_grad_func_contain_y=True,
            )

    def test_optimizer_system_info_accepts_irregular_layout(self):
        """Nested element rows are flattened according to batch_indices."""
        output_file = self._path('opt_irregular_metadata.bin')
        runner = FIRE(
            E_threshold=0., F_threshold=0., maxiter=1, steplength=0.1,
            output_file=output_file, device='cpu', verbose=0,
        )
        runner.set_system_info(
            cell_vec=th.zeros((2, 3, 3), dtype=th.bfloat16),
            atomic_numbers=[['H'], ['O', 'H']],
        )
        coordinates = th.zeros((1, 3, 3), dtype=th.float32)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            runner.run(
                lambda value: th.stack((
                    th.sum(value[:, :1] ** 2),
                    th.sum(value[:, 1:] ** 2),
                )),
                coordinates,
                grad_func=lambda value, energy: 2 * value,
                is_grad_func_contain_y=True,
                batch_indices=(1, 2),
            )

        raw = read_dump_arrays(output_file)
        self.assertEqual(raw['batch_indices'].tolist(), [1, 2])
        self.assertEqual(raw['atomic_numbers'].tolist(), [1, 8, 1])
        self.assertEqual(raw['cell_vec'].shape, (2, 3, 3))

    def test_fire_preserves_atomic_numbers_and_regular_batch_shape(self):
        """FIRE headers store atomic numbers rather than rounded masses."""
        output_file = self._path('fire_elements.bin')
        coordinates = th.tensor(
            [[[0.2, 0., 0.]], [[0.3, 0., 0.]]], dtype=th.float32
        )
        runner = FIRE(
            E_threshold=0., F_threshold=0., maxiter=1, steplength=0.1,
            output_file=output_file, device='cpu', verbose=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            runner.run(
                lambda value: th.sum(value ** 2, dim=(-2, -1)),
                coordinates,
                grad_func=lambda value, energy: 2 * value,
                is_grad_func_contain_y=True,
                elements=[['C'], ['O']],
            )

        raw = read_dump_arrays(output_file)
        self.assertEqual(raw['atomic_numbers'].tolist(), [[6], [8]])

    def test_numeric_cpu_md_header_supports_postprocessing_masses(self):
        """Numeric Element_list entries remain integer atomic numbers."""
        output_file = self._path('numeric_md.bin')
        runner = NVE(0.1, 1, output_file=output_file, device='cpu', verbose=0)
        coordinates = th.zeros((1, 2, 3), dtype=th.float32)
        runner.run(
            lambda value: th.sum(value ** 2, dim=(-2, -1)),
            coordinates,
            [[1, 8]],
            V_init=th.zeros_like(coordinates),
            grad_func=lambda value, energy: 2 * value,
            is_grad_func_contain_y=True,
        )

        trajectory = MDTrajectory(output_file)
        self.assertEqual(trajectory.atomic_numbers[0].tolist(), [1, 8])
        self.assertTrue(np.allclose(trajectory.masses[0], [1.008, 15.999]))

    def test_cpu_buffers_do_not_request_pinned_memory(self):
        """CPU-only installations never enter the pinned-allocation path."""
        original_empty_like = th.empty_like

        def reject_pinned_memory(*args, **kwargs):
            if kwargs.get('pin_memory'):
                raise RuntimeError('No pinned-memory allocator is available.')
            return original_empty_like(*args, **kwargs)

        state = StdContainer(Energy=th.zeros(1))
        with mock.patch(
                'BUCToolkit.Bases.BaseMotion.th.empty_like',
                side_effect=reject_pinned_memory,
        ):
            cpu_state, staging_state = BaseMotion._allocate_cpu_buffers(
                state,
                ('Energy',),
                th.device('cpu'),
                require_buffer=False,
            )
        self.assertEqual(cpu_state.Energy.device.type, 'cpu')
        self.assertIsNone(staging_state)

    def test_segment_reader_preserves_changed_batch_metadata(self):
        """Each appended segment is split with its own static header."""
        output_file = self._path('changed_batch.bin')
        dumper = ArrayDumper(output_file, mode='x')
        for n_batch in (1, 2):
            cells = np.zeros((n_batch, 3, 3), dtype=np.float32)
            atomic_numbers = np.ones((n_batch, 1), dtype=np.int64)
            fixed_mask = np.ones((n_batch, 1, 3), dtype=np.float32)
            run_id = np.asarray([n_batch], dtype=np.int64)
            dumper.start_from_arrays(
                1, cells, atomic_numbers, fixed_mask, run_id,
                names=('cell_vec', 'atomic_numbers', 'fixed_mask', 'run_id'),
            )
            dumper.step(cells, atomic_numbers, fixed_mask, run_id)
            energies = np.arange(n_batch, dtype=np.float32)
            coordinates = np.zeros((n_batch, 1, 3), dtype=np.float32)
            dumper.start_from_arrays(
                1, energies, coordinates,
                names=('Energy', 'X'),
            )
            dumper.step(energies, coordinates)
        dumper.close()

        with self.assertRaises(ValueError):
            read_dump_arrays(output_file)
        columns = read_mc_traj(output_file, out_arrays=True)
        self.assertEqual(len(columns['Energy']), 3)
        self.assertEqual(len(columns['X']), 3)
        self.assertNotIn('run_id', columns)

    def test_constrained_nvt_updates_state_and_allows_repeated_runs(self):
        """Constraint fields honor dump/log selections across repeated runs."""
        constr_func = lambda coo: th.linalg.norm(
            coo[1] - coo[0]
        ).reshape(1)
        energy_func = lambda coo: 0.5 * th.sum(coo ** 2, dim=(-2, -1))
        grad_func = lambda coo, energy: coo
        X = th.tensor([[[0., 0., 0.], [1., 0., 0.]]], dtype=th.float32)
        V = th.tensor([[[0.1, 0., 0.], [-0.1, 0., 0.]]], dtype=th.float32)

        for scheme, config in (
                ('Langevin', {'damping_coeff': 0.01}),
                ('Nose-Hoover', {}),
        ):
            with self.subTest(scheme=scheme):
                output_file = self._path(f'{scheme}.bin')
                check_log = scheme == 'Langevin'
                runner = ConstrNVT(
                    0.1,
                    3,
                    scheme,
                    config,
                    constr_func,
                    None,
                    1e-5,
                    False,
                    300.,
                    output_file,
                    1,
                    'cpu',
                    2 if check_log else 0,
                    log_quantities=(
                        'Energy', 'Ek', 'temperature', 'X', 'V', 'Fc'
                    ) if check_log else None,
                )
                if check_log:
                    stream = io.StringIO()
                    logger = logging.getLogger(
                        f'{__name__}.constrained_nvt_log.{id(runner)}'
                    )
                    logger.handlers.clear()
                    logger.propagate = False
                    logger.setLevel(logging.INFO)
                    handler = logging.StreamHandler(stream)
                    handler.setFormatter(logging.Formatter('%(message)s'))
                    logger.addHandler(handler)
                    runner.logger = logger
                runner.run(
                    energy_func,
                    X,
                    [['H', 'H']],
                    V_init=V,
                    grad_func=grad_func,
                    is_grad_func_contain_y=True,
                )
                if check_log:
                    handler.flush()
                    log_output = stream.getvalue()
                    logger.removeHandler(handler)
                    handler.close()

                    self.assertEqual(log_output.count('Step:'), 3)
                    for name in ('Energy', 'Ek', 'temperature'):
                        self.assertEqual(
                            log_output.count(f'\t{name:<12s} = '), 3
                        )
                    for name in ('X', 'V', 'Fc'):
                        self.assertEqual(log_output.count(f' {name}:\n'), 3)
                    self.assertNotIn('tensor(', log_output)
                    self.assertNotIn('Failed to logout', log_output)
                    self.assertNotIn('\tw           = ', log_output)
                    self.assertNotIn('\tG           = ', log_output)
                    self.assertNotIn(' w:\n', log_output)
                    self.assertNotIn(' G:\n', log_output)
                raw = read_dump_arrays(output_file)
                self.assertIn('Fc', raw)
                self.assertTrue(any(np.any(value != 0.) for value in raw['Fc'][1:]))

        # A placeholder dumper permits a second run on the same instance and
        # isolates the persistent-registration behavior from file append mode.
        runner = ConstrNVT(
            0.01, 1, 'Langevin', {'damping_coeff': 0.01},
            constr_func, None, 1e-5, False, 300., None, 1, 'cpu', 0,
            log_quantities=('Fc',),
        )
        for _ in range(2):
            runner.run(
                energy_func,
                X,
                [['H', 'H']],
                V_init=th.zeros_like(V),
                grad_func=grad_func,
                is_grad_func_contain_y=True,
            )
        self.assertEqual(list(runner.get_dump_vars()).count('Fc'), 1)
        self.assertEqual(list(runner.get_log_vars()).count('Fc'), 1)

        default_fixman = ConstrNVT(
            0.01, 1, 'Langevin', {'damping_coeff': 0.01},
            constr_func, None, 1e-5, True, 300., None, 1, 'cpu', 0,
        )
        self.assertEqual(
            list(default_fixman.get_dump_vars()),
            ['Energy', 'X', 'V', 'Force', 'Fc', 'G', 'w'],
        )
        self.assertEqual(
            list(default_fixman.get_log_vars()),
            ['Energy', 'Ek', 'temperature', 'X', 'V', 'Force', 'Fc', 'G', 'w'],
        )

        configurable = ConstrNVT(
            0.01, 1, 'Langevin', {'damping_coeff': 0.01},
            constr_func, None, 1e-5, True, 300., None, 1, 'cpu', 0,
            dump_quantities=('Energy', 'Fc', 'G', 'w'),
            log_quantities=('Energy', 'w'),
        )
        self.assertEqual(
            list(configurable.get_dump_vars()), ['Energy', 'Fc', 'G', 'w']
        )
        self.assertEqual(list(configurable.get_log_vars()), ['Energy', 'w'])

        configurable_nve = ConstrNVE(
            0.01, 1, constr_func, None, 1e-5, False, 300., None, 1,
            'cpu', 0,
            dump_quantities=('Fc',),
            log_quantities=(),
        )
        self.assertEqual(list(configurable_nve.get_dump_vars()), ['Fc'])
        self.assertEqual(list(configurable_nve.get_log_vars()), [])


if __name__ == '__main__':
    unittest.main()
