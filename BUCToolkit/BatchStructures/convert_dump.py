"""Convert legacy BUCToolkit DB 1.0 motion dumps to canonical DB 2.0."""

from __future__ import annotations

import argparse
from os import PathLike, path as os_path
from typing import Literal, Sequence

import numpy as np

from BUCToolkit.utils._Element_info import ATOMIC_SYMBOL
from .StructuresIO import (
    ArrayDumper,
    ArrayDumpReaderOld,
    _read_dump_segments,
)


DumpKind = Literal['md', 'mc', 'opt']


def _normalize_atomic_numbers(values: np.ndarray) -> np.ndarray:
    """Normalize legacy element metadata to canonical atomic numbers.

    Legacy writers may store integer-like values, element symbols as strings,
    or UTF-encoded byte strings. The original array shape is preserved while
    every entry is converted to the integer representation required by DB 2.0.

    Args:
        values: Legacy atomic metadata of any shape.

    Returns:
        Integer ``numpy.ndarray`` with the same shape as ``values`` and dtype
        ``int64``.

    Raises:
        ValueError: If a symbol is absent from BUCToolkit's atomic-symbol map.
    """
    source = np.asarray(values)
    normalized = np.empty(source.shape, dtype=np.int64)
    for index, value in np.ndenumerate(source):
        item = value.item() if isinstance(value, np.generic) else value
        if isinstance(item, (str, bytes)):
            symbol = item.decode() if isinstance(item, bytes) else item
            try:
                normalized[index] = ATOMIC_SYMBOL[symbol]
            except KeyError as error:
                raise ValueError(
                    f'Unknown element symbol {symbol!r} in legacy metadata.'
                ) from error
        else:
            normalized[index] = int(item)
    return normalized


def _normalize_segment_headers(segments: list[dict]) -> list[dict]:
    """Normalize atomic metadata in a list of motion segments.

    Header dictionaries are copied so conversion does not modify the objects
    returned by the legacy reader. Dynamic data lists are reused unchanged
    because their values are already suitable for canonical writing.

    Args:
        segments: Legacy segment dictionaries containing ``header`` and
            ``data`` mappings.

    Returns:
        New segment dictionaries whose ``atomic_numbers`` header arrays use
        canonical integer values.
    """
    normalized_segments = []
    for segment in segments:
        header = dict(segment['header'])
        header['atomic_numbers'] = _normalize_atomic_numbers(
            header['atomic_numbers']
        )
        normalized_segments.append({'header': header, 'data': segment['data']})
    return normalized_segments


def _read_legacy_motion_segments(path: str, kind: DumpKind) -> list[dict]:
    """Read a DB 1.0 MD or MC file and assign historical column names.

    Unnamed MD groups are mapped to ``Energy``, ``X``, ``V``, and ``Force``;
    unnamed MC groups are mapped to ``Energy`` and ``X``. Named transitional
    DB 1.0 groups retain their stored names. Atomic metadata is normalized
    before the segments are returned for canonical writing.

    Args:
        path: Path to the legacy motion dump.
        kind: Producing framework, either ``'md'`` or ``'mc'``.

    Returns:
        Ordered segment dictionaries suitable for
        :func:`_write_canonical_segments`.
    """
    data_names = {
        'md': ('Energy', 'X', 'V', 'Force'),
        'mc': ('Energy', 'X'),
    }[kind]
    return _normalize_segment_headers(
        _read_dump_segments(
            path,
            ArrayDumpReaderOld,
            indices=-1,
            is_copy=True,
            legacy_data_names=data_names,
        )
    )


def _read_legacy_optimizer_segments(path: str) -> list[dict]:
    """Translate DB 1.0 optimizer groups into canonical segment dictionaries.

    A legacy optimizer group contains one cycle of eight positional arrays:
    ``batch_indices``, structure IDs, cell vectors, atomic metadata,
    coordinates, fixed masks, energies, and forces. The static and dynamic
    arrays are separated here into the alternating pair expected by DB 2.0.

    Args:
        path: Path to a legacy DB 1.0 optimizer dump.

    Returns:
        Ordered segment dictionaries with named header fields and named
        ``structure_ids``, ``Energy``, ``X``, and ``Force`` data columns.

    Raises:
        ValueError: If a group does not contain exactly one cycle of eight
            arrays.
    """
    reader = ArrayDumpReaderOld(path)
    raw_groups = reader.read(groups=-1, indices=-1, is_copy=True)
    segments = []
    for group_index in range(reader.n_groups):
        cycles = raw_groups[f'group{group_index}']
        if len(cycles) != 1 or len(cycles[0]) != 8:
            raise ValueError(
                f'Legacy optimizer group {group_index} must contain one cycle '
                f'of eight arrays.'
            )
        (
            batch_indices,
            structure_ids,
            cell_vectors,
            atomic_numbers,
            coordinates,
            fixed_mask,
            energies,
            forces,
        ) = cycles[0]
        batch_indices = np.asarray(batch_indices)
        total_atoms = int(np.sum(batch_indices))
        # Legacy irregular optimizer arrays may omit the synthetic leading
        # batch axis used by canonical motion dumps. Restore it consistently.
        if coordinates.ndim == 2 and coordinates.shape[0] == total_atoms:
            coordinates = coordinates[None, ...]
        if fixed_mask.ndim == 2 and fixed_mask.shape[0] == total_atoms:
            fixed_mask = fixed_mask[None, ...]
        if forces.ndim == 2 and forces.shape[0] == total_atoms:
            forces = forces[None, ...]

        segments.append({
            'header': {
                'batch_indices': batch_indices,
                'cell_vec': np.asarray(cell_vectors),
                'atomic_numbers': _normalize_atomic_numbers(atomic_numbers),
                'fixed_mask': np.asarray(fixed_mask),
            },
            'data': {
                'structure_ids': [np.asarray(structure_ids)],
                'Energy': [np.asarray(energies)],
                'X': [np.asarray(coordinates)],
                'Force': [np.asarray(forces)],
            },
        })
    return segments


def _write_canonical_segments(path: str, segments: Sequence[dict], overwrite: bool) -> None:
    """Write named segment dictionaries as one canonical DB 2.0 file.

    Each input segment becomes a one-cycle header group followed by a dynamic
    data group. Dictionary insertion order defines the stored array order, and
    every group is written with the same names used as dictionary keys.

    Args:
        path: Destination path for the canonical dump.
        segments: Ordered ``header``/``data`` dictionaries to write.
        overwrite: If ``True``, replace an existing destination; otherwise use
            exclusive creation mode.

    Returns:
        None. The dumper is closed on both successful and failed conversion.

    Raises:
        ValueError: If a segment has no dynamic columns or its dynamic columns
            contain zero or inconsistent cycle counts.
    """
    dumper = ArrayDumper(path, mode='w' if overwrite else 'x')
    try:
        for segment_index, segment in enumerate(segments):
            # Write arbitrary named metadata exactly as supplied. No fixed
            # header-name registry is needed by the canonical format.
            header_names = tuple(segment['header'])
            header_arrays = tuple(
                np.asarray(segment['header'][name]) for name in header_names
            )
            dumper.start_from_arrays(1, *header_arrays, names=header_names)
            dumper.step(*header_arrays)

            data_names = tuple(segment['data'])
            if not data_names:
                raise ValueError(f'Segment {segment_index} has no dynamic columns.')
            cycle_counts = {
                len(segment['data'][name]) for name in data_names
            }
            if len(cycle_counts) != 1:
                raise ValueError(
                    f'Segment {segment_index} has inconsistent column lengths.'
                )
            n_cycles = cycle_counts.pop()
            if n_cycles == 0:
                raise ValueError(f'Segment {segment_index} has no dynamic cycles.')
            prototypes = tuple(
                np.asarray(segment['data'][name][0]) for name in data_names
            )
            dumper.start_from_arrays(n_cycles, *prototypes, names=data_names)
            for cycle_index in range(n_cycles):
                dumper.step(*(
                    np.asarray(segment['data'][name][cycle_index])
                    for name in data_names
                ))
    finally:
        dumper.close()


def convert_dump(
        input_path: str | PathLike[str],
        output_path: str | PathLike[str],
        kind: DumpKind,
        overwrite: bool = False,
) -> None:
    """Convert one legacy DB 1.0 motion dump to canonical DB 2.0.

    Conversion is intentionally one-way. MD and MC files are interpreted as
    alternating header/data pairs, while optimizer files use their historical
    single-group eight-array layout. The output always contains fully named
    DB 2.0 groups and can be read by the canonical trajectory APIs.

    Args:
        input_path: Existing DB 1.0 dump file.
        output_path: Destination for the canonical DB 2.0 file.
        kind: Producing framework: ``'md'``, ``'mc'``, or ``'opt'``.
        overwrite: Replace an existing destination when ``True``.

    Returns:
        None. A new canonical file is written to ``output_path``.

    Raises:
        ValueError: If the legacy layout does not match ``kind`` or contains
            invalid atomic metadata.
        FileExistsError: If the destination exists and ``overwrite`` is false.
    """
    if kind not in ('md', 'mc', 'opt'):
        raise ValueError(f'kind must be md, mc, or opt, got {kind!r}.')
    source_path = str(input_path)
    destination_path = str(output_path)
    if os_path.abspath(source_path) == os_path.abspath(destination_path):
        raise ValueError('Input and output paths must be different.')
    if os_path.exists(destination_path) and not overwrite:
        raise FileExistsError(destination_path)
    if kind == 'opt':
        segments = _read_legacy_optimizer_segments(source_path)
    else:
        segments = _read_legacy_motion_segments(source_path, kind)
    _write_canonical_segments(destination_path, segments, overwrite)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the legacy-dump converter command-line interface.

    Args:
        argv: Optional argument sequence excluding the interpreter and module
            name. ``None`` uses ``sys.argv`` through :mod:`argparse`.

    Returns:
        None. Parsed arguments are forwarded to :func:`convert_dump`.
    """
    parser = argparse.ArgumentParser(
        description='Convert a BUCToolkit DB 1.0 dump to canonical DB 2.0.'
    )
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--kind', required=True, choices=('md', 'mc', 'opt'))
    parser.add_argument('--overwrite', action='store_true')
    arguments = parser.parse_args(argv)
    convert_dump(
        arguments.input_path,
        arguments.output_path,
        kind=arguments.kind,
        overwrite=arguments.overwrite,
    )


if __name__ == '__main__':
    main()
