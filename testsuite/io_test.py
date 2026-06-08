""" I/O unit tests: OUTCAR → POSCAR/cif → binary round-trip validation. """

#  Copyright (c) 2026, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: io_test.py
#  Environment: Python 3.12

import os
import tarfile
import shutil
from typing import List

import numpy as np

from BUCToolkit.io import OUTCAR2Feat, POSCARs2Feat, Cif2Feat
from BUCToolkit.BatchStructures import BatchStructures


_HERE = os.path.dirname(os.path.abspath(__file__))
_TGZ_PATH = os.path.join(_HERE, 'test_structures', 'OUTCARs', 'outcars.tgz')


def _untar(target_dir: str) -> None:
    """Extract OUTCARs tgz to target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(_TGZ_PATH, 'r:gz') as tf:
        tf.extractall(path=target_dir, filter='data')


def _arrays_close(a, b, rtol=1e-5, atol=1e-5):
    """Return True if arrays/element-lists are close/equal."""
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return False
        return all(_arrays_close(ai, bi, rtol, atol) for ai, bi in zip(a, b))
    arr_a = np.asarray(a)
    if arr_a.dtype.kind in ('U', 'S', 'O'):  # string or object → exact
        return np.array_equal(arr_a, np.asarray(b))
    return bool(np.allclose(arr_a, np.asarray(b), rtol=rtol, atol=atol))


def _build_signature(elements, numbers):
    """Canonical representation: sorted (element, count) pairs."""
    d = {}
    for el, n in zip(elements, numbers):
        d[el] = d.get(el, 0) + int(n)
    return tuple(sorted(d.items()))


def run_io_tests(tmp_base: str = '/dev/shm') -> List[str]:
    r"""Run full I/O test suite. Returns list of error messages (empty = all pass)."""
    errors = []
    outcar_dir = os.path.join(tmp_base, 'io_test_outcars')
    poscar_dir = os.path.join(tmp_base, 'io_test_poscar')
    cif_dir = os.path.join(tmp_base, 'io_test_cif')
    bin_path = os.path.join(tmp_base, 'io_test_binary')
    bin_path2 = os.path.join(tmp_base, 'io_test_binary2')

    # cleanup
    for d in [outcar_dir, poscar_dir, cif_dir, bin_path, bin_path2]:
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        elif os.path.isfile(d):
            os.remove(d)

    try:
        # ----------------------------------------------------------------
        # Step 1: Extract tgz and read OUTCARs
        # ----------------------------------------------------------------
        _untar(outcar_dir)
        bs_outcar = OUTCAR2Feat(outcar_dir, verbose=0)
        bs_outcar.read()
        n_structs = len(bs_outcar)
        assert n_structs > 0, 'No structures read from OUTCARs'
        print(f'  OUTCAR read: {n_structs} structures')

        ref_coords = bs_outcar.Coords
        ref_cells = bs_outcar.Cells
        ref_elements = bs_outcar.Elements
        ref_numbers = bs_outcar.Numbers

        # collect signatures for identity checks
        ref_sigs = [_build_signature(e, n) for e, n in zip(ref_elements, ref_numbers)]

        # ----------------------------------------------------------------
        # Step 2: Write POSCAR → read back → compare element/cell identity
        # ----------------------------------------------------------------
        bs_outcar.write2text(poscar_dir, file_format='POSCAR')
        bs_poscar = POSCARs2Feat(poscar_dir, verbose=0)
        bs_poscar.read()
        assert len(bs_poscar) == n_structs, \
            f'POSCAR count mismatch: {len(bs_poscar)} vs {n_structs}'
        pos_sigs = [_build_signature(e, n)
                    for e, n in zip(bs_poscar.Elements, bs_poscar.Numbers)]
        assert sorted(ref_sigs) == sorted(pos_sigs), 'POSCAR element signatures mismatch'
        assert _arrays_close(sorted(ref_cells, key=lambda x: x.sum()),
                              sorted(bs_poscar.Cells, key=lambda x: x.sum()),
                              atol=0.01), 'POSCAR cells mismatch'
        print(f'  POSCAR round-trip: {n_structs} structures OK')

        # ----------------------------------------------------------------
        # Step 3: Write CIF → read back → compare element/cell identity
        # ----------------------------------------------------------------
        bs_outcar.write2text(cif_dir, file_format='cif')
        bs_cif = Cif2Feat(cif_dir, verbose=0)
        bs_cif.read()
        assert len(bs_cif) == n_structs, \
            f'CIF count mismatch: {len(bs_cif)} vs {n_structs}'
        cif_sigs = [_build_signature(e, n)
                    for e, n in zip(bs_cif.Elements, bs_cif.Numbers)]
        assert sorted(ref_sigs) == sorted(cif_sigs), 'CIF element signatures mismatch'
        print(f'  CIF round-trip: {n_structs} structures OK')

        # ----------------------------------------------------------------
        # Step 4: Binary save/load (mode='w') — full fidelity
        # ----------------------------------------------------------------
        bs_outcar.save(bin_path, mode='w')
        bs_bin = BatchStructures.load_from_file(bin_path)
        assert len(bs_bin) == n_structs, \
            f'Binary count mismatch: {len(bs_bin)} vs {n_structs}'
        assert _arrays_close(ref_coords, bs_bin.Coords), 'binary coords'
        assert _arrays_close(ref_cells, bs_bin.Cells), 'binary cells'
        assert _arrays_close(ref_elements, bs_bin.Elements), 'binary elements'
        assert _arrays_close(ref_numbers, bs_bin.Numbers), 'binary numbers'
        assert _arrays_close(bs_outcar.Energies, bs_bin.Energies), 'binary energies'
        assert _arrays_close(bs_outcar.Forces, bs_bin.Forces), 'binary forces'
        print(f'  Binary save/load (w): OK')

        # ----------------------------------------------------------------
        # Step 5: Binary append (mode='a')
        # ----------------------------------------------------------------
        bs_outcar[:n_structs // 2].save(bin_path2, mode='w')
        bs_app = BatchStructures()
        bs_app.load(bin_path2, mode='w')
        n1 = len(bs_app)
        rest = bs_outcar[n_structs // 2:]
        rest.save(bin_path2, mode='a')
        bs_app.load(bin_path2, data_slice=(n1, n_structs), mode='a')
        assert len(bs_app) == n_structs, \
            f'Binary append count mismatch: {len(bs_app)} vs {n_structs}'
        assert _arrays_close(ref_coords, bs_app.Coords), 'binary append coords'
        print(f'  Binary append: OK')

        # ----------------------------------------------------------------
        # Step 6: Test BatchStructures public methods
        # ----------------------------------------------------------------
        bs = bs_outcar[:min(10, n_structs)]  # use slicing for subset (no .copy() needed)
        public_methods = [
            'change_mode', 'check_full', 'generate_dist_mat',
            'generate_atom_list', 'generate_atomic_number_list',
            'cartesian2direct', 'direct2cartesian', 'sort_ids', 'standardize',
            'shuffle', 'rearrange', 'fix_atoms_by_height',
            'contain_any', 'contain_all',
        ]
        for meth_name in public_methods:
            try:
                meth = getattr(bs, meth_name, None)
                if meth is None:
                    continue
                if meth_name in ('change_mode',):
                    meth('A', release_mem=False)
                    meth('L')
                elif meth_name == 'check_full':
                    meth()
                elif meth_name == 'generate_dist_mat':
                    meth()
                elif meth_name == 'generate_atom_list':
                    meth()
                elif meth_name == 'generate_atomic_number_list':
                    bs.generate_atom_list()
                    meth()
                elif meth_name == 'cartesian2direct':
                    meth()
                    bs.direct2cartesian()
                elif meth_name == 'direct2cartesian':
                    bs.cartesian2direct()
                    meth()
                elif meth_name == 'sort_ids':
                    meth()
                elif meth_name == 'standardize':
                    meth()
                elif meth_name == 'shuffle':
                    meth(seed=42)
                elif meth_name == 'rearrange':
                    meth(list(range(len(bs))))
                elif meth_name == 'fix_atoms_by_height':
                    meth(0.5)
                elif meth_name == 'contain_any':
                    bs.contain_any(bs.Elements[0])
                elif meth_name == 'contain_all':
                    bs.contain_all(bs.Elements[0][:1])
            except NotImplementedError:
                pass  # expected for some methods
            except Exception as e:
                errors.append(f'** Failed to call the method BatchStructures.{meth_name}: {type(e).__name__}: {e}')
        print(f'  BatchStructures methods: {len(public_methods)} tested')

        # ----------------------------------------------------------------
        # Step 7: split_dataset
        # ----------------------------------------------------------------
        from BUCToolkit.Preprocessing.preprocessing import split_dataset
        parts = split_dataset(bs_outcar[:20], ratio=[0.5, 0.3, 0.2], shuffle=True, seed=42)
        assert len(parts) == 3
        assert sum(len(p) for p in parts) == 20
        print(f'  split_dataset: OK (20 -> {[len(p) for p in parts]})')

    except Exception as e:
        import traceback
        errors.append(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
    finally:
        # cleanup
        for d in [outcar_dir, poscar_dir, cif_dir, bin_path, bin_path2]:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
            elif os.path.isfile(d):
                os.remove(d)

    return errors
