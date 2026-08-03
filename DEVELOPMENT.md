# BUCToolkit development convention

This document records the intended development style of BUCToolkit. It is the
project-level convention for new code and for code modified during maintenance.
Historical files may contain older patterns; those inconsistencies do not
override this guide and should not be rewritten without a related reason.

## 1. General engineering philosophy

- Prefer the smallest correct modification.
- Do not substantially refactor established code unless correctness,
  readability, or efficiency requires it. Discuss architectural changes before
  implementing them.
- Prefer a simple and direct mechanism over registries, inference frameworks,
  schema helpers, or layered abstractions.
- Reuse existing infrastructure instead of creating parallel implementations.
- Preserve public behavior unless an intentional compatibility break has been
  agreed upon.
- Scientific correctness, array layout, device behavior, and data integrity
  take priority over cosmetic modernization.

## 2. Naming

- Names must reveal the represented concept, not merely its Python type.
- Use the same name for the same concept across MD, MC, optimizers, readers,
  and writers.
- Containers use plural names, such as `dump_names`, `atomic_numbers`, and
  `structure_elements`.
- Counts identify what they count, such as `n_dump_cycles`, `n_true_batch`, and
  `n_atom`.
- Boolean variables describe an actual proposition and normally begin with
  `is_`, `has_`, `should_`, `require_`, or `use_`.
- Avoid vague temporary names when a clear domain name is available.
- Leading underscores are for genuinely internal or short-lived values, not a
  substitute for descriptive naming.
- Uppercase names are reserved for constants or established scientific
  quantities.

## 3. Conditional expressions

Only genuine Boolean values should be used directly in conditions:

```python
if require_fixman:
    ...
```

Other types require explicit conditions:

```python
if value is not None:
    ...
if len(dump_names) > 0:
    ...
if verbose > 0:
    ...
if tensor.numel() > 0:
    ...
if number == 0:
    ...
```

Do not rely on implicit truth-value conversion for containers, numeric values,
tensors, arrays, or optional objects. For a repeated condition, first construct
an actual Boolean with a meaningful name:

```python
has_dump_quantities = len(dump_names) > 0
if has_dump_quantities:
    ...
```

Every component of a compound condition should state its exact meaning.

## 4. Types and conversion

- Type annotations describe the actual data layout, including nesting and
  optional values. For example, per-structure elements use
  `List[List[str | int]]`, not a flat list annotation.
- Use Python, NumPy, and Torch conversions when they already provide reliable
  value validation, such as `int(value)`, `np.array(..., dtype=...)`, and
  `torch.as_tensor(...)`.
- Do not reproduce built-in conversion logic with large manual type-dispatch
  mechanisms.
- Add manual checks for domain invariants that conversion cannot establish:
  nesting, exact shape, finite values, supported elements, batch compatibility,
  and regular or irregular layout consistency.
- Normalize data at the public API boundary so internal code handles one
  canonical representation.
- Complete validation before mutating retained state. A failed call must not
  leave a partial update.

## 5. Functions and methods

- A method should have one clear responsibility.
- Do not introduce a helper merely to hide a few straightforward lines.
- Add a helper when logic is genuinely shared, independently meaningful, or
  difficult to understand inline.
- Every private helper must have an active caller.
- Public methods validate their contract and provide useful error messages.
- Prefer established public registration and initialization methods over
  direct mutation of internal attributes.
- Keep call signatures stable unless changing them is necessary and
  intentional.
- Remove obsolete functions after a redesign, except APIs deliberately retained
  for compatibility.

## 6. Branching and data layout

- Use the minimum authoritative state needed to select a branch.
- For motion data, regular versus irregular layout is determined only by
  whether `batch_indices is not None`.
- Do not infer layout from optional metadata such as cells, atomic numbers,
  masks, or header names.
- Optional metadata remains independent and extensible.
- Avoid hardcoded lists that require updating whenever a new optional field is
  introduced.
- Document expected regular and irregular shapes near the code that transforms
  them.

## 7. Documentation

New or substantially modified public methods contain:

1. A meaningful description.
2. `Args`.
3. `Returns`.

`Raises` is added when validation or conversion failures are important.
`Notes` is optional and should be used only when lifetime, performance, layout,
or device behavior needs clarification.

Docstrings should describe accepted shapes, regular and irregular layouts,
retained state or side effects, output organization, and important device
behavior. Avoid empty template sections and overly brief docstrings for
nontrivial methods.

## 8. Comments

- Comments explain why code exists, which invariant it protects, or why a less
  obvious implementation is required.
- Do not comment obvious syntax.
- Keep comments close to the relevant operation.
- Important examples include why irregular atomic numbers are flattened, why
  pinned memory is CUDA-only, why a constraint target is refreshed once per
  projection, and why header and data names are preserved separately.
- For a part of code that implements a specified (relatively independent) function/method, 
  one may add a simple, brief introduction. If this part is long, the comment would 
  start with `# Section: xxx`

## 9. Error handling

- Validate inputs before expensive model evaluation or file creation when
  practical.
- Error messages identify the offending argument, expected contract, and
  received type, shape, or value.
- Use `TypeError` for unsupported container or interface types.
- Use `ValueError` for invalid values, shapes, nesting, or relationships.
- Do not silently guess between incompatible representations.
- Background consumers may log exceptions when they cannot propagate them,
  but synchronous validation should prevent foreseeable asynchronous failures.

## 10. Binary database design

- Software version and database version are independent.
- Canonical DB 2.0 files are self-describing and fully named.
- Canonical readers use stored names rather than positional assumptions.
- Static headers may contain arbitrary optional metadata.
- Every appended header/data pair is processed with its own metadata.
- Legacy positional handling belongs only in explicit `read_*_old` APIs.
- Conversion from an old format to the canonical format is explicit and
  one-way.
- Compatibility heuristics do not belong in canonical readers.

## 11. Dumping and logging

- `dump_quantities` and `log_quantities` are independent ordered selections.
- Dictionary insertion order defines output order.
- Names, tensors, CPU buffers, and binary columns remain aligned.
- Existing registrations refresh values without changing order.
- Constraint diagnostics use the same public registration mechanism as
  standard quantities.
- `verbose == 0` is silent, `verbose == 1` prints selected scalar or vector
  quantities, and `verbose >= 2` also prints selected arrays.
- `log_quantities` selects fields and does not override verbosity.

## 12. CPU, CUDA, and asynchronous code

- CPU-only execution is a supported first-class path.
- Pinned host memory is used only when CUDA transfers require it.
- Avoid unnecessary synchronization, copies, tensor-to-string conversions, and
  repeated container construction inside iteration loops.
- Shared asynchronous buffers are not overwritten until consumers release
  them.
- Event state changes use an order that cannot lose a consumer signal.
- Expensive numerical work remains overlapped with I/O where correctness
  permits it.
- Device-specific branches share data contracts even when their transfer
  mechanisms differ.

## 13. Testing

- Every bug fix has a focused regression reproducing the actual failure.
- Tests verify public behavior and stored results, not only internal
  implementation details.
- Binary-format tests verify names, shapes, dtypes, group pairing, and
  round-trip reconstruction.
- Cover regular and irregular batches where layout behavior differs.
- Cover CPU and CUDA paths in proportion to the change.
- Run focused smoke tests during development. Do not run the complete suite
  unless the maintainer explicitly requests it.
- Run relevant `py_compile` checks and `git diff --check` before handoff.

## 14. Change management

- Read `DEVELOPMENT.md`, and the current git change before
  modifying code.
- Update a changelog after material implementation changes.
- Preserve unrelated worktree changes and untracked user files.
- Do not alter software or database versions without an explicit format or
  release decision.
- Commit messages order changes by importance and use numbered points, unless only one change.
- Each numbered commit-message point remains on one physical line.

