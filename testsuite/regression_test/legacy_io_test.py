"""Focused regressions for explicit DB 1.0 readers and DB 2.0 conversion."""

import os
import tempfile
import unittest

import numpy as np

from BUCToolkit.BatchStructures import (
    ArrayDumpReader,
    convert_dump,
    read_dump_arrays,
    read_mc_traj,
    read_md_traj_old,
    read_opt_structures,
    read_opt_structures_old,
)


def _legacy_group_bytes(cycles: list[tuple[np.ndarray, ...]]) -> bytes:
    """Encode one DB 1.0 group without an n_names section."""
    prototypes = cycles[0]
    output = bytearray()
    output.extend('HEAD'.encode('utf-16-le'))
    output.extend(len(cycles).to_bytes(8, 'little'))
    output.extend(len(prototypes).to_bytes(8, 'little'))
    for array in prototypes:
        dtype_string = np.asarray(array).dtype.str
        output.extend(dtype_string[:2].encode('utf-16-le'))
        output.extend(int(dtype_string[2:]).to_bytes(4, 'little'))
        for dimension in np.asarray(array).shape:
            output.extend(int(dimension).to_bytes(8, 'little'))
        output.extend((0).to_bytes(8, 'little'))
    for cycle in cycles:
        for array in cycle:
            output.extend(np.ascontiguousarray(array).tobytes())
    return bytes(output)


def _write_legacy_file(path: str, groups: list[list[tuple[np.ndarray, ...]]]) -> None:
    header = bytearray()
    header.extend('<'.encode('utf-16-le'))
    header.extend('BM'.encode('utf-16-le'))
    header.extend(bytes((1, 0)))
    header.extend(len(groups).to_bytes(8, 'little'))
    with open(path, 'wb') as output:
        output.write(header)
        for cycles in groups:
            output.write(_legacy_group_bytes(cycles))


class LegacyIOTest(unittest.TestCase):
    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._temporary_directory.cleanup()

    def _path(self, name: str) -> str:
        return os.path.join(self._temporary_directory.name, name)

    def test_legacy_md_reader_and_conversion(self):
        legacy_path = self._path('legacy_md.bin')
        canonical_path = self._path('canonical_md.bin')
        cells = np.zeros((2, 3, 3), dtype=np.float32)
        atomic_symbols = np.asarray([['H'], ['O']])
        fixed_mask = np.ones((2, 1, 3), dtype=np.float32)
        header_cycle = (cells, atomic_symbols, fixed_mask)
        data_cycles = []
        for step in range(2):
            energies = np.asarray([step, step + 1], dtype=np.float32)
            coordinates = np.full((2, 1, 3), step, dtype=np.float32)
            velocities = coordinates + 1
            forces = coordinates + 2
            data_cycles.append((energies, coordinates, velocities, forces))
        _write_legacy_file(legacy_path, [[header_cycle], data_cycles])

        with self.assertRaises(ValueError):
            ArrayDumpReader(legacy_path)
        old_columns = read_md_traj_old(legacy_path, out_arrays=True)
        self.assertEqual(len(old_columns['Energy']), 4)

        convert_dump(legacy_path, canonical_path, kind='md')
        with open(canonical_path, 'rb') as converted_file:
            self.assertEqual(converted_file.read(8)[6:8], bytes((2, 0)))
        converted = read_dump_arrays(canonical_path)
        self.assertEqual(converted['atomic_numbers'].tolist(), [[1], [8]])
        self.assertEqual(len(converted['Energy']), 2)

    def test_legacy_optimizer_conversion(self):
        legacy_path = self._path('legacy_opt.bin')
        canonical_path = self._path('canonical_opt.bin')
        batch_indices = np.asarray([1, 1], dtype=np.int64)
        structure_ids = np.asarray(['carbon', 'oxygen'])
        cells = np.zeros((2, 3, 3), dtype=np.float32)
        atomic_numbers = np.asarray([6, 8], dtype=np.int64)
        coordinates = np.zeros((2, 3), dtype=np.float32)
        fixed_mask = np.ones((2, 3), dtype=np.float32)
        energies = np.asarray([1., 2.], dtype=np.float32)
        forces = np.ones((2, 3), dtype=np.float32)
        _write_legacy_file(legacy_path, [[(
            batch_indices,
            structure_ids,
            cells,
            atomic_numbers,
            coordinates,
            fixed_mask,
            energies,
            forces,
        )]])

        old_structures = read_opt_structures_old(legacy_path)
        self.assertEqual(len(old_structures), 2)
        convert_dump(legacy_path, canonical_path, kind='opt')
        columns = read_opt_structures(canonical_path, out_arrays=True)
        self.assertEqual(columns['structure_ids'], ['carbon', 'oxygen'])
        self.assertEqual(len(columns['X']), 2)

    def test_legacy_mc_conversion(self):
        legacy_path = self._path('legacy_mc.bin')
        canonical_path = self._path('canonical_mc.bin')
        cells = np.zeros((1, 3, 3), dtype=np.float32)
        atomic_numbers = np.asarray([[1]], dtype=np.int64)
        fixed_mask = np.ones((1, 1, 3), dtype=np.float32)
        energies = np.asarray([0.5], dtype=np.float32)
        coordinates = np.zeros((1, 1, 3), dtype=np.float32)
        _write_legacy_file(
            legacy_path,
            [[(cells, atomic_numbers, fixed_mask)], [(energies, coordinates)]],
        )

        convert_dump(legacy_path, canonical_path, kind='mc')
        columns = read_mc_traj(canonical_path, out_arrays=True)
        self.assertEqual(columns['Energy'], [0.5])
        self.assertEqual(len(columns['X']), 1)


if __name__ == '__main__':
    unittest.main()
