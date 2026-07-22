"""
Unit tests for ArrayDumper / ArrayDumpReader engine in StructuresIO.py.

Covers canonical named arrays, explicit legacy reading, UTF-16 non-ASCII
names, multiple groups, name validation, and dynamic-step mode.
"""
import os
import struct
import tempfile
import unittest

import numpy as np

from BUCToolkit.BatchStructures.StructuresIO import (
    ArrayDumper,
    ArrayDumpReader,
    ArrayDumpReaderOld,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_path(suffix: str = '.bin') -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _arrays_close(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

class ArrayDumperTest(unittest.TestCase):
    """Independent unit tests for the Python ArrayDumper/ArrayDumpReader."""

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def _path(self, suffix: str = '.bin') -> str:
        p = _make_temp_path(suffix)
        self._paths.append(p)
        return p

    # ------------------------------------------------------------------
    # 1. Canonical names are mandatory
    # ------------------------------------------------------------------
    def test_names_are_required(self):
        """DB 2.0 refuses unnamed array groups."""
        arrays_tpl = [
            np.zeros((1,), dtype=np.float64),
            np.zeros((10, 3), dtype=np.float64),
            np.zeros((10, 3), dtype=np.float64),
        ]
        path = self._path()

        d = ArrayDumper(path, mode='w')
        d.initialize()
        with self.assertRaises(RuntimeError):
            d.start_from_arrays(5, *arrays_tpl)
        d.close()

    # ------------------------------------------------------------------
    # 2. Round-trip with names
    # ------------------------------------------------------------------
    def test_roundtrip_with_names(self):
        """Write arrays WITH names, read back — data + names must match."""
        rng = np.random.RandomState(99)
        n_cycles = 3
        arrays_tpl = [
            np.zeros((1,), dtype=np.float64),       # energy
            np.zeros((8, 3), dtype=np.float64),      # coordinates
            np.zeros((8, 3), dtype=np.float64),      # forces
        ]
        names = ["energy", "coordinates", "forces"]
        written = []
        path = self._path()

        d = ArrayDumper(path, mode='w')
        d.initialize()
        d.start_from_arrays(n_cycles, *arrays_tpl, names=names)
        for _ in range(n_cycles):
            cycle = [rng.randn(*a.shape).astype(a.dtype) for a in arrays_tpl]
            written.append([c.copy() for c in cycle])
            d.step(*cycle)
        d.close()

        r = ArrayDumpReader(path)
        self.assertEqual(r.n_groups, 1)
        result = r.read(groups=0, indices=-1, is_copy=True)
        py_data = result['group0']
        self.assertEqual(len(py_data), n_cycles)
        for i, (w_cycle, r_cycle) in enumerate(zip(written, py_data)):
            for j, (w_arr, r_arr) in enumerate(zip(w_cycle, r_cycle)):
                self.assertTrue(_arrays_close(r_arr, w_arr),
                                f"cycle {i} array {j}: mismatch")

        # Verify names
        self.assertIsNotNone(r.names)
        self.assertIn(0, r.names)
        self.assertEqual(r.names[0], names)

    # ------------------------------------------------------------------
    # 3. Old-format file (no n_names field) readable by new reader
    # ------------------------------------------------------------------
    def test_old_file_readable(self):
        """Construct an old-format file (without n_names field) manually
        and verify the new reader can read it correctly."""
        rng = np.random.RandomState(777)
        n_cycles = 4

        # Build prototype arrays
        a1 = rng.randn(1).astype(np.float64)        # energy
        a2 = rng.randn(6, 3).astype(np.float64)      # positions
        a3 = rng.randn(6, 3).astype(np.float64)      # velocities
        arrays = [a1, a2, a3]
        written = [[c.copy() for c in arrays]]

        str_fmt = 'utf-16-le'
        num_fmt = 'little'

        # -- File header (16 bytes) --
        fh = b''
        fh += '<'.encode(str_fmt)                       # head_order (2 B)
        fh += 'BM'.encode(str_fmt)                      # magic      (4 B)
        fh += b'\x01\x00'                               # version 1.0 (2 B)
        fh += (1).to_bytes(8, num_fmt, signed=False)    # n_groups   (8 B)

        # -- Data header (old format: no n_names field) --
        dh = b''
        dh += 'HEAD'.encode(str_fmt)                    # "HEAD"     (8 B)
        dh += n_cycles.to_bytes(8, num_fmt, signed=False)  # n_cycle
        dh += len(arrays).to_bytes(8, num_fmt, signed=False)  # n_array

        for arr in arrays:
            dt = arr.dtype
            # dtype: order+type (4 B UTF-16) + itemsize (4 B int32)
            dh += (dt.str[0] + dt.str[1]).encode(str_fmt)
            dh += dt.itemsize.to_bytes(4, num_fmt, signed=False)
            # shape dimensions (each 8 B int64) + delimiter 0 (8 B)
            for dim in arr.shape:
                dh += dim.to_bytes(8, num_fmt, signed=False)
            dh += (0).to_bytes(8, num_fmt, signed=False)

        # -- Data --
        data_bytes = b''
        for _ in range(n_cycles):
            for arr in arrays:
                data_bytes += np.ascontiguousarray(arr).tobytes()

        path = self._path()
        with open(path, 'wb') as f:
            f.write(fh + dh + data_bytes)

        with self.assertRaises(ValueError):
            ArrayDumpReader(path)

        # DB 1.0 compatibility is explicit rather than automatic.
        r = ArrayDumpReaderOld(path)
        self.assertEqual(r.n_groups, 1)
        result = r.read(groups=0, indices=-1, is_copy=True)
        py_data = result['group0']
        self.assertEqual(len(py_data), n_cycles)
        for i, (w_cycle, r_cycle) in enumerate(zip(written, py_data)):
            for j, (w_arr, r_arr) in enumerate(zip(w_cycle, r_cycle)):
                self.assertTrue(_arrays_close(r_arr, w_arr),
                                f"old-format file: cycle {i} array {j}: mismatch")

    # ------------------------------------------------------------------
    # 4. Names with non-ASCII (UTF-16) characters
    # ------------------------------------------------------------------
    def test_names_utf16_roundtrip(self):
        """Names containing non-ASCII characters round-trip correctly."""
        arrays_tpl = [
            np.zeros((2,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        ]
        names = ["能量", "forces"]  # Chinese + English
        path = self._path()

        d = ArrayDumper(path, mode='w')
        d.initialize()
        d.start_from_arrays(2, *arrays_tpl, names=names)
        d.step(np.array([1.0, 2.0], dtype=np.float32),
               np.array([3.0, 4.0], dtype=np.float32))
        d.step(np.array([5.0, 6.0], dtype=np.float32),
               np.array([7.0, 8.0], dtype=np.float32))
        d.close()

        r = ArrayDumpReader(path)
        result = r.read(groups=0)
        self.assertIsNotNone(r.names)
        self.assertIn(0, r.names)
        self.assertEqual(r.names[0], names)

    # ------------------------------------------------------------------
    # 5. Multiple groups, each with different names
    # ------------------------------------------------------------------
    def test_multiple_groups_with_names(self):
        """Three groups, each with distinct array names."""
        rng = np.random.RandomState(1234)
        n_groups = 3
        n_cycles = 3
        path = self._path()

        all_written = []
        expected_names = {}

        d = ArrayDumper(path, mode='w')
        d.initialize()
        for g in range(n_groups):
            tpl = [np.zeros((g + 1, 2), dtype=np.float64),
                   np.zeros((g + 1,), dtype=np.int64)]
            gnames = [f"data_g{g}", f"index_g{g}"]
            expected_names[g] = gnames
            d.start_from_arrays(n_cycles, *tpl, names=gnames)
            cycles = []
            for _ in range(n_cycles):
                a = rng.randn(g + 1, 2).astype(np.float64)
                b = rng.randint(0, 100, size=(g + 1,), dtype=np.int64)
                cycles.append([a.copy(), b.copy()])
                d.step(a, b)
            d.truncate()
            all_written.append(cycles)
        d.close()

        r = ArrayDumpReader(path)
        self.assertEqual(r.n_groups, n_groups)
        result = r.read(groups=-1, indices=-1, is_copy=True)
        self.assertEqual(len(result), n_groups)

        for g in range(n_groups):
            key = f'group{g}'
            self.assertIn(key, result)
            for i, expected in enumerate(all_written[g]):
                for j, (read_arr, exp_arr) in enumerate(
                    zip(result[key][i], expected)
                ):
                    self.assertTrue(_arrays_close(read_arr, exp_arr),
                                    f"group {g} cycle {i} array {j}")

        # Verify names
        self.assertIsNotNone(r.names)
        for g in range(n_groups):
            self.assertIn(g, r.names)
            self.assertEqual(r.names[g], expected_names[g])

    # ------------------------------------------------------------------
    # 6. Name count validation — mismatched count raises error
    # ------------------------------------------------------------------
    def test_names_validation(self):
        """Providing wrong number of names raises ValueError."""
        arrays_tpl = [
            np.zeros((1,), dtype=np.float64),
            np.zeros((10, 3), dtype=np.float64),
        ]
        path = self._path()

        d = ArrayDumper(path, mode='w')
        d.initialize()
        # 2 arrays but 3 names → should raise
        with self.assertRaises(RuntimeError):
            d.start_from_arrays(3, *arrays_tpl, names=["a", "b", "c"])
        d.close()

    # ------------------------------------------------------------------
    # 7. Dynamic steps (steps=-1) with names
    # ------------------------------------------------------------------
    def test_dynamic_steps_with_names(self):
        """Names work correctly with steps=-1 (dynamic extension mode)."""
        rng = np.random.RandomState(5566)
        arrays_tpl = [
            np.zeros((1,), dtype=np.float64),
            np.zeros((4, 3), dtype=np.float64),
        ]
        names = ["scalar", "tensor"]
        written = []
        path = self._path()

        d = ArrayDumper(path, mode='w')
        d.initialize()
        # steps=-1 triggers dynamic mode (initial allocation = 500 cycles)
        d.start_from_arrays(-1, *arrays_tpl, names=names)
        n_steps = 10
        for _ in range(n_steps):
            cycle = [rng.randn(*a.shape).astype(a.dtype) for a in arrays_tpl]
            written.append([c.copy() for c in cycle])
            d.step(*cycle)
        d.close()

        r = ArrayDumpReader(path)
        self.assertEqual(r.n_groups, 1)
        result = r.read(groups=0, indices=-1, is_copy=True)
        py_data = result['group0']
        self.assertEqual(len(py_data), n_steps)
        for i, (w_cycle, r_cycle) in enumerate(zip(written, py_data)):
            for j, (w_arr, r_arr) in enumerate(zip(w_cycle, r_cycle)):
                self.assertTrue(_arrays_close(r_arr, w_arr),
                                f"dynamic: cycle {i} array {j}: mismatch")

        # Verify names survived dynamic extension
        self.assertIsNotNone(r.names)
        self.assertIn(0, r.names)
        self.assertEqual(r.names[0], names)

    # ------------------------------------------------------------------
    # 8. Read by names — select arrays by name via read()
    # ------------------------------------------------------------------
    def test_read_by_names(self):
        """``read(names=[...])`` selects and reorders arrays by name."""
        rng = np.random.RandomState(888)
        arrays_tpl = [
            np.zeros((1,), dtype=np.float64),        # energy
            np.zeros((6, 3), dtype=np.float64),       # coordinates
            np.zeros((6, 3), dtype=np.float64),       # forces
        ]
        all_names = ["energy", "coordinates", "forces"]
        path = self._path()

        d = ArrayDumper(path, mode='w')
        d.initialize()
        d.start_from_arrays(4, *arrays_tpl, names=all_names)
        for _ in range(4):
            d.step(
                rng.randn(1),
                rng.randn(6, 3),
                rng.randn(6, 3),
            )
        d.close()

        r = ArrayDumpReader(path)

        # Read only two named arrays.
        result = r.read(groups=0, names=["forces", "energy"])
        cycles = result['group0']
        self.assertEqual(len(cycles), 4)
        for cyc in cycles:
            self.assertEqual(len(cyc), 2)
            self.assertEqual(cyc[0].shape, (6, 3))   # forces
            self.assertEqual(cyc[1].shape, (1,))      # energy

    # ------------------------------------------------------------------
    # 9. Read by indices_array — select arrays by column index
    # ------------------------------------------------------------------
    def test_read_by_indices_array(self):
        """``read(indices_array=[...])`` selects arrays by column index."""
        rng = np.random.RandomState(777)
        arrays_tpl = [
            np.zeros((1,), dtype=np.float64),
            np.zeros((3,), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
        ]
        path = self._path()

        d = ArrayDumper(path, mode='w')
        d.initialize()
        d.start_from_arrays(3, *arrays_tpl, names=('a', 'b', 'c'))
        for _ in range(3):
            d.step(rng.randn(1), rng.randn(3), rng.randn(2, 2))
        d.close()

        r = ArrayDumpReader(path)
        result = r.read(groups=0, indices_array=[2, 0])
        cycles = result['group0']
        self.assertEqual(len(cycles), 3)
        for cyc in cycles:
            self.assertEqual(len(cyc), 2)
            self.assertEqual(cyc[0].shape, (2, 2))   # col 2
            self.assertEqual(cyc[1].shape, (1,))      # col 0

    # ------------------------------------------------------------------
    # 10. Combined — cycle selection + array selection work together
    # ------------------------------------------------------------------
    def test_read_cycle_and_array_selection(self):
        """``indices`` + ``indices_array`` together: 2D slicing."""
        rng = np.random.RandomState(333)
        arrays_tpl = [
            np.zeros((1,), dtype=np.float64),
            np.zeros((2,), dtype=np.float64),
        ]
        path = self._path()
        d = ArrayDumper(path, mode='w')
        d.initialize()
        d.start_from_arrays(5, *arrays_tpl, names=('a', 'b'))
        written = []
        for k in range(5):
            a = np.array([float(k)])
            b = np.array([float(k * 10), float(k * 10 + 1)])
            written.append([a.copy(), b.copy()])
            d.step(a, b)
        d.close()

        r = ArrayDumpReader(path)
        # cycles 1 and 3, only array at column 0
        result = r.read(groups=0, indices=[1, 3], indices_array=[0])
        cycles = result['group0']
        self.assertEqual(len(cycles), 2)
        self.assertEqual(len(cycles[0]), 1)
        self.assertTrue(np.allclose(cycles[0][0], written[1][0]))
        self.assertTrue(np.allclose(cycles[1][0], written[3][0]))

    # ------------------------------------------------------------------
    # 11. ``indices_array`` and ``names`` are mutually exclusive
    # ------------------------------------------------------------------
    def test_read_indices_array_names_exclusive(self):
        """Passing both ``indices_array`` and ``names`` raises ValueError."""
        path = self._path()
        d = ArrayDumper(path, mode='w')
        d.initialize()
        d.start_from_arrays(2, np.zeros((1,), dtype=np.float64),
                            np.zeros((2,), dtype=np.float64),
                            names=["a", "b"])
        d.step(np.array([1.0]), np.array([2.0, 3.0]))
        d.step(np.array([4.0]), np.array([5.0, 6.0]))
        d.close()

        r = ArrayDumpReader(path)
        with self.assertRaises(ValueError):
            r.read(groups=0, indices_array=[0], names=["a"])

    # ------------------------------------------------------------------
    # 12. ``indices_iter`` alias works
    # ------------------------------------------------------------------
    def test_read_indices_iter_alias(self):
        """``indices_iter`` is an alias for ``indices``."""
        rng = np.random.RandomState(112)
        path = self._path()
        d = ArrayDumper(path, mode='w')
        d.initialize()
        d.start_from_arrays(
            4, np.zeros((1,), dtype=np.float64), names=('value',)
        )
        for _ in range(4):
            d.step(rng.randn(1))
        d.close()

        r = ArrayDumpReader(path)
        result = r.read(groups=0, indices_iter=[0, 2])
        self.assertEqual(len(result['group0']), 2)

    # ------------------------------------------------------------------
    # 13. Duplicate names are rejected
    # ------------------------------------------------------------------
    def test_duplicate_names_raise(self):
        """Canonical group names must be unique."""
        path = self._path()
        d = ArrayDumper(path, mode='w')
        d.initialize()
        with self.assertRaises(RuntimeError):
            d.start_from_arrays(
                2,
                np.zeros((1,), dtype=np.float64),
                np.zeros((1,), dtype=np.float64),
                names=('energy', 'energy'),
            )
        d.close()


if __name__ == '__main__':
    unittest.main()
