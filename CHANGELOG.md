# Changelog

## Unreleased

- The central `main` MD and CMD tests now build samples with non-uniform masses:
  every atom is drawn with equal probability from the light, medium and heavy
  groups H / Al / Pd (a uniform draw split at 1/3 and 2/3). The equipartition
  standards use `k_B T <1/m>`, and the energy plots use the same sampled masses,
  so a missing or permuted mass factor can no longer hide behind a
  single-element system.
- Fixed the experimental `BaseConstr.christoffel_solver`. The constraint
  directional derivative no longer subtracts the batched target inside a vmapped
  per-sample call (which added a spurious batch axis), the mass-weighted normal
  basis is factorized at the current coordinates instead of a stale projection,
  and the Woodbury block is formed as `I + dt R^{-T} J' B`.
- Corrected the constraint-solver formulas in the English and Chinese README:
  the mass-weighted decomposition is `M^{-1/2} J^T = Q R`, the tangent-space
  projection carries the normal term of a time-dependent target, and the
  retraction is described as a Newton iteration in a fixed reference normal
  basis. Added the constrained geodesic (Christoffel) solver description.

- Constraint targets passed through `constr_val` must now be two-dimensional
  `(n_batch, n_constr)`. The single-row form `(1, n_constr)` is broadcast to
  every batch sample instead of being rejected, while a one-dimensional target
  is rejected because its single axis is ambiguous between "one value per
  constraint shared by the batch" and "one value per batch sample". Shape
  mismatch messages now report the expected and received shapes in the correct
  order, and `BaseConstr.initialize` restarts the constraint clock at zero so a
  reused component cannot inherit a stale trajectory time.
- Made `MACEDataLoader` build MACE neighbor graphs lazily in a bounded
  background prefetch queue, so MACE training memory is bounded by the buffer
  instead of the full data set. Task APIs inject the resolved `MODEL_CONFIG`
  automatically, and CLI MACE training keeps PyG data instead of eagerly
  converting every structure. `ISFSMACEDataLoader` remains for pre-converted
  static paired data; dynamic NEB/CMD runs continue to use
  `ISFSPyGDataLoader` so `MACEWrapper.forward` rebuilds graphs per batch.
- Exported `BiasedNVT` and `BiasedNVE`, and removed the package-level circular
  import in `BiasedMD.py`. Added regression coverage for external-potential MD
  and central-suite soft diagnostics for its equipartition and virial
  consistency; existing constrained-manifold checks are unchanged.

## 1.0.1 (2026-10-27)

- Added `KRYLOV` (`KrylovNewton`) and `KRYDYN` (`KrylovDynamics`) as
  `TRANSITION_STATE.ALGO` choices in `StructureOptimization`, including their
  eigen-solver, step-length, Morse-index, and spectral-cutoff parameters.
- Removed the unused `steplength_sheme` argument from `KrylovDynamics`, whose
  dynamics update does not use a Newton step-length scheme.
- Fixed `StructureOptimization.set_dimer_init_direction`, which now supplies the
  initial direction used by DIMER, KRYLOV, and KRYDYN runs.
- Exported `KrylovDynamics` from `BUCToolkit.BatchOptim` and synchronized the
  CLI input stub with the transition-state parameters.
- Updated the English and Chinese README, manual, examples, and the
  `StructureOptimization` docstring.
- Added `FSDATA_READER_KWARGS` and `DISP_DATA_READER_KWARGS` so the
  `FSDATA_PATH` and `DISPDATA_PATH` readers can be configured independently.
  `DATA_READER_KWARGS` now applies only to `DATA_PATH` and `VAL_SET_PATH` and
  is not inherited by the other data paths.

## 1.0.0 (2026-10-20)

The first stable release.

- Integrated constrained MD into the `NVE` and `NVT` engines as an optional
  constraint component. `ConstrNVE` and `ConstrNVT` remain as compatibility
  shells.
- Added optional Triton fused kernels for MD updates and thermostats, with an
  automatic eager fallback when Triton is unavailable.
- Raised the minimum supported PyTorch version to 2.5.1.
- Added a lightweight CPU GitHub Actions CI pipeline, pytest configuration, and
  regression coverage for constrained MD, free-energy calculations, and
  optional dependencies.
- Added CPU/GPU dual-device checks for the free-energy regression and isolated
  Torch Dynamo compilation state between regression tests.
- Renamed the short CLI alias `bckt` to `bct`; the main `buctoolkit` command is
  unchanged.
- Marked the `--test profile` mode as temporarily unavailable because the
  profiler perturbs CUDA-graph replay and the asynchronous custom kernels.
- Added README-based PyPI long-description metadata and updated the release
  documentation.
