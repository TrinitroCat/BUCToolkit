# Changelog

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
