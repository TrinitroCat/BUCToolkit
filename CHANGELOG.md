# Changelog

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
