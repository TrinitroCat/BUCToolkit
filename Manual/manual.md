# BUCToolkit Manual

This manual is a parameter-oriented companion to the project README. It
describes the supported input-file fields, public high-level APIs, and the
main low-level simulation and optimization classes. Detailed implementation
behavior remains in the source code and docstrings.

Workflow examples are collected in [Examples](examples.md). Sections below
link to an example that uses the relevant argument.

## Conventions

BUCToolkit does not perform automatic unit conversion. Normal units are
Angstrom for length, eV for energy, amu for mass, fs for time, and K for
temperature. A model's `func(X, ...)` returns energy and `grad_func(X, ...)`
returns the mathematical energy gradient; physical forces are the negative
gradient unless a model adapter performs that conversion.

Regular batches use `(batch_size, n_atom, 3)`. Irregular batches use
`(1, sum(n_atom), 3)` together with `batch_indices=(n_1, ..., n_N)`.

## YAML Input Reference

The executable CLI reads YAML with `buctoolkit -i input.yml`. Fields marked as
task-specific are only read by the corresponding task. Paths inside the YAML
file are resolved relative to that file.

### Global fields

| Field | Values / type | Default | Meaning |
| --- | --- | --- | --- |
| `TASK` | `TRAIN`, `PREDICT`, `OPT`, `TS`, `VIB`, `NEB`, `MD`, `CMD`, `MC` | CLI-required | Selects the high-level task. [CLI task example](examples.md#run-a-task-from-yaml) |
| `START` | `0`, `1`, `2`, `from_scratch`, `resume` | `0` | Start from scratch, resume a checkpoint, or load model weights only. [Checkpoint example](examples.md#configure-a-model-and-checkpoint) |
| `VERBOSE` | non-negative integer | `1` | Text output detail. `0` is silent; higher values print more arrays. [Output example](examples.md#read-and-convert-results) |
| `DEVICE` | Torch device string | `cpu` | Device used by model and calculation. [API setup](examples.md#common-api-setup) |
| `BATCH_SIZE` | positive integer | `1` | Structures per loader batch. CMD currently requires `1`. [Batch layout](examples.md#batch-layout) |

### I/O and data fields

| Field | Values / type | Meaning |
| --- | --- | --- |
| `LOAD_CHK_FILE_PATH` | path | Checkpoint loaded when `START` is `1` or `2`. |
| `OUTPUT_ROOT` | path | Root directory for logs, results, and checkpoints. A missing root is created; an existing root must be empty. |
| `OUTPUT_PATH` | path | Legacy or explicit log directory. With `OUTPUT_ROOT`, defaults to `OUTPUT_ROOT/logs`. |
| `OUTPUT_POSTFIX` | string | Suffix used in log names. |
| `PREDICTIONS_SAVE_FILE` | path | Binary output path when saving is enabled. |
| `STRICT_LOAD` | boolean | Passed to `load_state_dict`. |
| `REDIRECT` | boolean | Write logs to `OUTPUT_PATH` when true; otherwise print them. |
| `SAVE_PREDICTIONS` | boolean | Save predictions or trajectories to `PREDICTIONS_SAVE_FILE`. |
| `DATA_TYPE` | `POSCAR`, `OUTCAR`, `CIF`, `ASE_TRAJ`, `BS`, `OPT`, `MD`, `MC` | CLI input format. `BS` is BUCToolkit's binary structure format. [Load data](examples.md#load-and-convert-structures) |
| `DATA_PATH` | path | Initial or training data. |
| `DATA_NAME_SELECTOR` | regular-expression string | Selects structures by sample name. |
| `FSDATA_PATH` | path | Final-state data for paired tasks such as NEB and Blue-Moon CMD. [Blue-Moon CMD](examples.md#blue-moon-cmd) |
| `DISPDATA_PATH` | path | Initial dimer displacement/direction data for TS searches. |
| `VAL_SET_PATH` | path | Explicit training validation set. |
| `VAL_SPLIT_RATIO` | float in `[0, 1)` | Validation fraction when `VAL_SET_PATH` is absent. |
| `DATA_LOADER_KWARGS` | mapping | Extra keyword arguments for the selected loader. |
| `IS_SHUFFLE` | boolean | Shuffle unpaired calculation data. Paired tasks and CMD use deterministic order. |

### `TRAIN`

| Field | Values / type | Default | Meaning |
| --- | --- | --- | --- |
| `EPOCH` | positive integer | `0` | Training epochs. [Training](examples.md#train-a-model-with-the-api) |
| `SAVE_CHK` | boolean | `false` | Save checkpoints. |
| `CHK_SAVE_PATH` | path | `OUTPUT_ROOT/chk` | Checkpoint directory. |
| `CHK_SAVE_POSTFIX` | string | empty | Checkpoint filename suffix. |
| `VAL_BATCH_SIZE` | positive integer | `BATCH_SIZE` | Validation batch size. |
| `VAL_PER_STEP` | positive integer | `10` | Validation frequency. |
| `VAL_IF_TRN_LOSS_BELOW` | float | `inf` | Delay validation until training loss is below this value. |
| `ACCUMULATE_STEP` | positive integer | `1` | Gradient accumulation steps. |
| `LOSS` | `MSE`, `MAE`, `Hubber`, `CrossEntropy`, `Energy_Force_Loss`, `Energy_Loss` | `Energy_Force_Loss` | Built-in loss selection. |
| `LOSS_CONFIG` | mapping | `{}` | Loss options such as `loss_E`, `loss_F`, `coeff_E`, and `coeff_F`. |
| `METRICS` | metric names | implementation default | `E_MAE`, `F_MAE`, `F_MaxE`, `E_R2`, `MSE`, `MAE`, `R2`, `RMSE`. |
| `METRICS_CONFIG` | mapping | `{}` | Metric-specific options. |
| `OPTIM` | Torch optimizer name or `FIRE` | `AdamW` | Parameter optimizer. |
| `OPTIM_CONFIG` | mapping | `{}` | Optimizer arguments, such as `lr`. |
| `LAYERWISE_OPTIM_CONFIG` | regex-to-mapping | absent | Per-layer optimizer options. |
| `GRAD_CLIP` | boolean | `false` | Enable gradient clipping. |
| `GRAD_CLIP_MAX_NORM` | float | `100` | Maximum gradient norm. |
| `GRAD_CLIP_CONFIG` | mapping | `{}` | Extra clipping arguments. |
| `LR_SCHEDULER` | scheduler name or `None` | `None` | Learning-rate scheduler. |
| `LR_SCHEDULER_CONFIG` | mapping | `{}` | Scheduler arguments. |
| `EMA` | boolean | `false` | Use exponential moving average parameters. |
| `EMA_DECAY` | float | `0.999` | EMA decay coefficient. |

### `RELAXATION`

| Field | Values / type | Default | Meaning |
| --- | --- | --- | --- |
| `ALGO` | `CG`, `BFGS`, `FIRE` | `FIRE` | Minimization algorithm. [Optimization](examples.md#structure-optimization-with-the-api) |
| `ITER_SCHEME` | `PR+`, `FR`, `PR`, `WYL` | `PR+` | CG iteration scheme. |
| `E_THRES` | float | `1e4` | Energy-difference threshold. |
| `F_THRES` | float | `0.05` | Maximum-force threshold. |
| `MAXITER` | positive integer | `300` | Maximum outer iterations. |
| `STEPLENGTH` | float | `0.5` | Initial step length. |
| `USE_BB` | boolean | `true` | Use Barzilai-Borwein initial step length. |
| `LINESEARCH` | `B`, `Wolfe`, `Exact` | `B` | Line-search method. |
| `LINESEARCH_MAXITER` | positive integer | `8` | Line-search iterations per step. |
| `LINESEARCH_THRES` | float | `0.02` | Exact-line-search threshold. |
| `LINESEARCH_FACTOR` | float | `0.5` | Backtracking shrink factor. |
| `REQUIRE_GRAD` | boolean | `false` | Enable autograd for the energy function. |

### `TRANSITION_STATE`

`ALGO` is currently `DIMER`; `X_DIFF_ATTR` names the initial dimer direction
attribute. `E_THRES`, `TORQ_THRES`, `F_THRES`, `MAXITER_TRANS`, `MAXITER_ROT`,
`MAX_STEPLENGTH`, and `DX` control convergence, rotation, translation, and
finite-difference settings. `REQUIRE_GRAD` has the usual meaning. See
[transition-state search](examples.md#transition-state-search).

### `VIBRATION`

| Field | Values / type | Default | Meaning |
| --- | --- | --- | --- |
| `METHOD` | `Coord`/`EnergyDiff`, `Grad`/`Autograd`, or `GradDiff` | `EnergyDiff` | Finite-difference or autograd route. |
| `BLOCK_SIZE` | positive integer | `1` | Displaced images evaluated together. |
| `DELTA` | positive float | `1e-2` | Coordinate displacement. |
| `SAVE_HESSIAN` | boolean | `false` | Save Hessian, frequencies, and modes. [Vibration](examples.md#vibrational-analysis) |

### `NEB`

| Field | Values / type | Default | Meaning |
| --- | --- | --- | --- |
| `ALGO` | `CI-NEB` | `CI-NEB` | NEB algorithm. |
| `N_IMAGES` | positive integer | `7` | Path images. |
| `SPRING_CONST` | float | `5.0` | Elastic spring constant. |
| `OPTIMIZER` | `FIRE` | `FIRE` | Image optimizer. |
| `OPTIMIZER_CONFIGS` | mapping | `{}` | Extra low-level optimizer options. |
| `STEPLENGTH` | float | `0.2` | Optimizer step length. |
| `E_THRESHOLD`, `F_THRESHOLD` | float | `1e-3`, `0.05` | Energy and force thresholds. |
| `MAXITER` | positive integer | `20` | Maximum NEB iterations. |
| `REQUIRE_GRAD` | boolean | `false` | Enable autograd. |

### `MD` and constrained MD

| Field                    | Values / type                           | Default                | Meaning                                                                                                                                               |
|--------------------------|-----------------------------------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENSEMBLE`               | `NVE`, `NVT`                            | `NVT`                  | MD ensemble. [MD](examples.md#molecular-dynamics-with-the-api)                                                                                        |
| `THERMOSTAT`             | `Langevin`, `VR`, `Nose-Hoover`, `CSVR` | `CSVR`                 | NVT thermostat.                                                                                                                                       |
| `THERMOSTAT_CONFIG`      | mapping                                 | `{}`                   | Use `damping_coeff`, `time_const`, or `virt_mass`; input parsing accepts case variants.                                                               |
| `TIME_STEP`              | positive float, fs                      | `1`                    | Integration time step.                                                                                                                                |
| `MAX_STEP`               | positive integer                        | implementation default | Integration steps.                                                                                                                                    |
| `T_INIT`                 | positive float, K                       | `298.15`               | Initial/target temperature.                                                                                                                           |
| `OUTPUT_COORDS_PER_STEP` | positive integer                        | `1`                    | Coordinate dump interval.                                                                                                                             |
| `MOVE_TO_CENTER_FREQ`    | integer                                 | `-1` low-level         | Centering interval; non-positive disables it.                                                                                                         |
| `CONSTRAINTS_FILE`       | path                                    | absent                 | Python file containing constraints.                                                                                                                   |
| `CONSTRAINTS_FUNC`       | function name                           | absent                 | Constraint function in that file.                                                                                                                     |
| `CONSTRAINTS_VAL_FUNC`   | function name or `null`                 | `null`                 | Optional time-dependent target from that file; `null` preserves the initial-value behaviour. [Constraint functions](examples.md#constraint-functions) |
| `REQUIRE_GRAD`           | boolean                                 | `false`                | Enable autograd.                                                                                                                                      |
| `CONSTR_MD_SCHEME`       | `SLOW_GROWTH`, `BLUE_MOON`              | template value         | CMD scheme.                                                                                                                                           |
| `N_IMAGES`               | positive integer                        | `1` or template value  | Blue-Moon interpolation images; Slow-growth parallel copies. Use `1` for one trajectory. [Slow growth](examples.md#slow-growth-cmd)                   |
| `CONSTR_THRESHOLD`       | positive float                          | `1.e-5`                | Constraints convergence threshold. Units are determined by specific constraints formulae                                                              |
| `REQUIRE_FIXMAN`         | boolean or "auto"                       | `"auto"`               | # `auto` means `True` for `BLUE_MOON` and False for `SLOW_GROWTH`. One may also set True/False manually.                                              |


For CLI CMD, `BLUE_MOON` reads paired `DATA_PATH`/`FSDATA_PATH` and uses
`ISFSPyGDataLoader`; `SLOW_GROWTH` reads only `DATA_PATH` and uses
`PyGDataLoader`. Both current CMD paths require `BATCH_SIZE: 1`.

### `MC`

| Field                     | Values / type                                      | Default          | Meaning                                                                        |
|---------------------------|----------------------------------------------------|------------------|--------------------------------------------------------------------------------|
| `TYPE`                    | `Metropolis`                                       | `Metropolis`     | Monte Carlo engine.                                                            |
| `ITER_SCHEME`             | `Gaussian`, `Cauchy`, `Uniform`                    | `Gaussian`       | Coordinate perturbation distribution. [MC](examples.md#metropolis-monte-carlo) |
| `COORDINATE_UPDATE_PARAM` | positive float                                     | `0.2`            | Distribution scale/range.                                                      |
| `MAXITER`                 | positive integer                                   | `10000`          | MC steps.                                                                      |
| `T_INIT`                  | positive float, K                                  | `298.15`         | Initial temperature.                                                           |
| `T_SCHEME`                | `constant`, `linear`, `exponential`, `log`, `fast` | `constant`       | Temperature schedule.                                                          |
| `T_UPDATE_FREQ`           | positive integer                                   | `1`              | Temperature update interval.                                                   |
| `T_SCHEME_PARAM`          | float                                              | scheme-dependent | Schedule parameter.                                                            |
| `OUTPUT_COORDS_PER_STEP`  | positive integer                                   | `1`              | Output interval.                                                               |
| `MOVE_TO_CENTER_FREQ`     | integer                                            | `-1`             | Centering interval; non-positive disables it.                                  |

### Model fields

`MODEL_FILE` and `MODEL_NAME` identify the CLI model; `MODEL_CONFIG` is passed
to its constructor. `START`, `LOAD_CHK_FILE_PATH`, and `STRICT_LOAD` control
parameter loading. See [model and checkpoint](examples.md#configure-a-model-and-checkpoint).

## High-level API Parameters

All high-level task classes receive `config_file` and optional `data_type`
(`pyg` or `dgl`). The common setup is:

```python
runner = Task(config_file)
runner.set_model_config(model_config)
runner.set_dataset(data, valid_data)
runner.set_dataloader(DataLoader, loader_kwargs)
runner.run(ModelClass)
```

| Method | Parameters | Meaning |
| --- | --- | --- |
| `set_device` | `device` | Override the configured Torch device. |
| `set_model_config` | `model_config` mapping or `None` | Model constructor keyword arguments. |
| `set_model_param` | `model_state_dict`, `is_strict`, `is_assign` | Load an in-memory state dictionary. |
| `set_dataset` | `train_data`, optional `valid_data` | Training dictionaries, calculation dictionaries, or paired `dataIS`/`dataFS`. |
| `set_dataloader` | loader class, optional mapping | Loader options; it receives configured batch size and device. |

Main task methods are `Trainer.train`, `Predictor.run/predict`,
`MolecularDynamics.run`, `ConstrainedMolecularDynamics.set_constr_func`,
`ConstrainedMolecularDynamics.set_constr_val`, `StructureOptimization.run` or
`relax`, `VibrationAnalysis.run`, `ClimbingImageNudgedElasticBand.run`, and
`MonteCarlo.run`. Complete signatures remain in the source docstrings.

## Low-level Function Parameters

Low-level engines operate directly on tensors. They do not load YAML files,
models, or structure files.

### Shared function protocol

`func(X, *func_args, **func_kwargs)` returns energy. An optional
`grad_func(X, *grad_func_args, **grad_func_kwargs)` returns the mathematical
energy gradient. If `is_grad_func_contain_y=True`, the call is
`grad_func(X, y, ...)`, where `y=func(X, ...)`. `require_grad` controls the
PyTorch autograd context.

### MD: `NVE`, `NVT`, `ConstrNVE`, `ConstrNVT`

Shared constructor parameters are `time_step`, `max_step`, `T_init`,
`output_file`, `output_structures_per_step`, `device`, `verbose`, `is_compile`,
`compile_kwargs`, `dump_quantities`, and `log_quantities`. `NVT` additionally
takes `thermostat` and `thermostat_config`. Constrained MD additionally takes
`constr_func`, `constr_val`, `constr_threshold`, and `require_fixman`.

The common `run` parameters are `func`, `X`, `Element_list`, optional
`Cell_vector`, `V_init`, `grad_func`, function argument/keyword collections,
`is_grad_func_contain_y`, `require_grad`, `batch_indices`,
`fixed_atom_tensor`, and `move_to_center_freq`. See [low-level MD](examples.md#low-level-md).

### Optimization: `CG`, `FIRE`, `QN`

The shared constructor options are `iter_scheme`, `E_threshold`,
`F_threshold`, `maxiter`, `linesearch`, `linesearch_maxiter`,
`linesearch_thres`, `linesearch_factor`, `steplength`, `use_bb`,
`output_file`, `device`, `verbose`, `dump_quantities`, and `log_quantities`.
`FIRE` instead uses `alpha`, `alpha_fac`, `fac_inc`, `fac_dec`, and `N_min`.
The common `run` call adds `output_grad`, `fixed_atom_tensor`, and
`batch_indices`. See [low-level optimization](examples.md#low-level-optimization).

Transition-state low-level classes use the same function protocol. `Dimer`
adds `Torque_thres`, `Curvature_thres`, `maxiter_trans`, `maxiter_rot`,
`max_steplength`, and `dx`. `CI_NEB` takes `N_images`, `spring_const`,
`optimizer`, `optimizer_configs`, `steplength`, convergence thresholds,
`maxiter`, `device`, and `verbose`. `KrylovNewton` and `KrylovDynamics` add
`Torque_thres`, `Eigen_thres`, `maxiter_trans`, `maxiter_eig`, `dx`,
`morse_index`, and spectral cutoffs. See [low-level transition-state search](examples.md#low-level-transition-state-search).

The low-level `Frequency` calculator takes `method`, optional `block_size`,
finite-difference `delta`, optional `output_file`, and `dump_hessian`. Its
`normal_mode` call supplies an energy function, coordinates, optional masses,
function/gradient arguments, and an optional fixed-atom mask. See [low-level
frequency analysis](examples.md#low-level-frequency-analysis).

### Monte Carlo: `MMC`

`MMC` takes `iter_scheme`, `maxiter`, `temperature_init`,
`temperature_scheme`, `temperature_update_freq`,
`temperature_scheme_param`, `coordinate_update_param`, output/device/verbosity
options, and dump/log quantity selections. Its `run` call uses the shared
energy protocol and accepts `X`, `Element_list`, optional `Cell_vector`,
`V_init`, `batch_indices`, and `fixed_atom_tensor`. See [low-level MC](examples.md#low-level-mc).

### Data and post-processing helpers

Main preprocessing readers are `POSCARs2Feat`, `OUTCAR2Feat`, `ExtXyz2Feat`,
`Cif2Feat`, and `ASETraj2Feat`. Their common options are `path`, `verbose`,
file selection, and format-specific tags. `CreatePygData`, `CreateDglData`,
and `CreateASE` convert the in-memory feature representation.

`PyGDataLoader` and `DglGraphLoader` take a `data` mapping, `batch_size`,
`device`, `shuffle`, `is_train`, and optional `data_names`.
`ISFSPyGDataLoader` takes paired `dataIS`/`dataFS`, `batch_size`, `device`, and
optional `data_names`. MACE loaders use the same split with model-specific data.

`Postprocessing.MDTrajectory` takes a binary trajectory path, `indices`, and
`is_copy`; it exposes columns, atomic numbers, masses, frame/image selection,
and stacking. `BlueMoonCalculator` consumes a trajectory and selects image
order, collective-variable values or columns, frame range, temperature, and
Fixman weighting. `ConstraintWorkCalculator` selects CV values/columns,
temperature, frame range, stride, integration rule, force sign, and
common-protocol tolerance. `VibrationalSpectrumCalculator` takes a trajectory
and `sample_spacing_fs`; its calculation accepts velocity columns, frame range,
window, and normalization options. See [post-processing examples](examples.md#postprocessing).

## Further reference

- [Parameter manual](manual.md)
- [中文参数手册](manual-zh.md)
- [中文示例](examples-zh.md)
- [README](../README.md)
