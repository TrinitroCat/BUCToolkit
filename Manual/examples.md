# BUCToolkit Examples

These examples cover both high-level APIs and low-level engines. Parameter
definitions are in [manual.md](manual.md). Code snippets use the current public
module paths; model classes and data paths are placeholders.

## Common API Setup

High-level APIs read a YAML configuration for task settings, then receive the
model, data, and loader explicitly:

```python
import BUCToolkit as bt
from BUCToolkit.api.MolecularDynamics import MolecularDynamics
from BUCToolkit.api.DataLoaders import PyGDataLoader

structures = bt.load('./structures')
data_list = bt.preprocessing.CreatePygData().feat2data_list(structures, n_core=1)

runner = MolecularDynamics('./md.yml', data_type='pyg')
runner.set_model_config({'hidden_dim': 128})
runner.set_dataset({'data': data_list})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)                 # pass the class, not MyModel(...)
```

The same setup pattern applies to `Trainer`, `Predictor`,
`StructureOptimization`, `VibrationAnalysis`,
`ClimbingImageNudgedElasticBand`, and `MonteCarlo`.

## Batch Layout

Regular batches have coordinates shaped `(batch_size, n_atom, 3)`. For an
irregular batch, concatenate coordinates as `(1, sum(n_atom), 3)` and provide
`batch_indices=(n_1, ..., n_N)`. The same split describes atomic elements,
velocities, and fixed-atom masks.

```python
import torch

X = torch.zeros((1, 377, 3))
batch_indices = [64, 56, 72, 83, 102]
```

## Configure a Model and Checkpoint

`MODEL_CONFIG` is passed to the model constructor. `START: 0` starts from
scratch; `START: 1` or `resume` loads a checkpoint; `START: 2` loads model
parameters only. `STRICT_LOAD` controls `load_state_dict` behavior.

```yaml
TASK: MD
START: 2
DEVICE: cuda:0
LOAD_CHK_FILE_PATH: ./checkpoint.pt
STRICT_LOAD: true
MODEL_CONFIG:
  hidden_dim: 128
```

## Run a Task from YAML

The CLI resolves YAML paths relative to the input file:

```shell
buctoolkit -i ./md.yml
```

Set `OUTPUT_ROOT` to create logs, results, and checkpoints. An interactive CLI
is available when no arguments are provided:

```shell
buctoolkit
```

## Load and Convert Structures

```python
import BUCToolkit as bt

features = bt.io.ExtXyz2Feat('./dataset')
features.read(
    ['train.xyz'],
    lattice_tag='lattice',
    energy_tag='energy',
    column_info_tag='properties',
    element_tag='species',
    coordinates_tag='pos',
    forces_tag='forces',
)

pyg_data = bt.preprocessing.CreatePygData().feat2data_list(features, n_core=1)
ase_atoms = bt.preprocessing.CreateASE().feat2ase(features)
```

`POSCARs2Feat`, `OUTCAR2Feat`, `Cif2Feat`, and `ASETraj2Feat` use the same
reader pattern with format-specific options. CLI output can be converted with:

```shell
buctoolkit -c md ./md_output poscar ./poscars
```

## Train a Model with the API

Training data contains labels; calculation data normally does not:

```python
from BUCToolkit.api.Trainer import Trainer
from BUCToolkit.api.DataLoaders import PyGDataLoader

train_data = {
    'data': train_data_list,
    'labels': {'energy': energies, 'forces': forces},
}
valid_data = {
    'data': valid_data_list,
    'labels': {'energy': valid_energies, 'forces': valid_forces},
}

trainer = Trainer('./train.yml')
trainer.set_dataset(train_data, valid_data)
trainer.set_dataloader(PyGDataLoader, {'shuffle': True})
trainer.train(MyModel)
```

`Trainer.set_loss_fn`, `set_optimizer`, `set_lr_scheduler`, `set_metrics`, and
`set_layerwise_optim_config` are optional overrides for YAML settings.

## Predict with the API

```python
from BUCToolkit.api.Predictor import Predictor
from BUCToolkit.api.DataLoaders import PyGDataLoader

predictor = Predictor('./predict.yml')
predictor.set_dataset({'data': data_list})
predictor.set_dataloader(PyGDataLoader, {'shuffle': False})
result = predictor.run(MyModel, test_model=False, warm_up=False)
```

With `SAVE_PREDICTIONS: true`, results are written to the configured binary
path instead of returned in memory.

## Structure Optimization with the API

```python
from BUCToolkit.api.StructureOptimization import StructureOptimization
from BUCToolkit.api.DataLoaders import PyGDataLoader

optimizer = StructureOptimization('./opt.yml')
optimizer.set_dataset({'data': data_list})
optimizer.set_dataloader(PyGDataLoader, {'shuffle': False})
optimizer.run(MyModel, mode='minimize')
```

Use `mode='ts'` and `set_dimer_init_direction(...)` for a dimer search.

## Molecular Dynamics with the API

```python
from BUCToolkit.api.MolecularDynamics import MolecularDynamics
from BUCToolkit.api.DataLoaders import PyGDataLoader

runner = MolecularDynamics('./md.yml')
runner.set_dataset({'data': data_list})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)
```

`MD.ENSEMBLE` selects `NVE` or `NVT`; `MD.THERMOSTAT_CONFIG` uses
`damping_coeff`, `time_const`, or `virt_mass` depending on the thermostat.

## Constraint Functions

```python
import torch

def constr_func(X: torch.Tensor) -> torch.Tensor:
    # X has shape (n_atom, 3) inside one constraint evaluation.
    return torch.linalg.vector_norm(X[0] - X[1]).reshape(1)

def constr_val(t: torch.Tensor) -> torch.Tensor:
    # Slow growth target: (N_IMAGES, n_constraint).
    return (1.5 + 0.001 * t).reshape(1, 1)
```

Both functions must be differentiable with respect to coordinates. In the
CLI, omit `MD.CONSTRAINTS_VAL_FUNC` or set it to `null` to retain the initial
value behavior.

## Slow-growth CMD

Slow growth starts from one set of structures and makes `N_IMAGES` parallel
copies. The CLI needs only `DATA_PATH`:

```yaml
TASK: CMD
BATCH_SIZE: 1
DATA_PATH: ./initial.bs
MD:
  ENSEMBLE: NVT
  CONSTR_MD_SCHEME: SLOW_GROWTH
  N_IMAGES: 4
  CONSTRAINTS_FILE: ./constraints.py
  CONSTRAINTS_FUNC: constr_func
  CONSTRAINTS_VAL_FUNC: constr_val
```

The equivalent API data and loader are:

```python
from BUCToolkit.api.ConstrainedMolecularDynamics import ConstrainedMolecularDynamics
from BUCToolkit.api.DataLoaders import PyGDataLoader

runner = ConstrainedMolecularDynamics('./cmd.yml')
runner.set_constr_func(constr_func)
runner.set_constr_val(constr_val)
runner.set_dataset({'data': initial_data})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)
```

## Blue-Moon CMD

Blue Moon pairs initial and final structures, interpolates `N_IMAGES` images, and
uses `ISFSPyGDataLoader`:

```yaml
TASK: CMD
BATCH_SIZE: 1
DATA_PATH: ./initial.bs
FSDATA_PATH: ./final.bs
MD:
  ENSEMBLE: NVT
  CONSTR_MD_SCHEME: BLUE_MOON
  N_IMAGES: 8
  CONSTRAINTS_FILE: ./constraints.py
  CONSTRAINTS_FUNC: constr_func
```

```python
from BUCToolkit.api.ConstrainedMolecularDynamics import ConstrainedMolecularDynamics
from BUCToolkit.api.DataLoaders import ISFSPyGDataLoader

runner = ConstrainedMolecularDynamics('./cmd.yml')
runner.set_constr_func(constr_func)
runner.set_dataset({'dataIS': initial_data, 'dataFS': final_data})
runner.set_dataloader(ISFSPyGDataLoader)
runner.run(MyModel)
```

Initial and final pairs must have matching atom counts and atomic-number order.

## Vibrational Analysis

```python
from BUCToolkit.api.VibrationAnalysis import VibrationAnalysis

runner = VibrationAnalysis('./vibration.yml')
runner.set_dataset({'data': data_list})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)
```

`VIBRATION.METHOD`, `DELTA`, `BLOCK_SIZE`, and `SAVE_HESSIAN` control the
finite-difference/autograd route and output.

## Transition-state Search

```python
from BUCToolkit.api.NEB import ClimbingImageNudgedElasticBand
from BUCToolkit.api.DataLoaders import ISFSPyGDataLoader

runner = ClimbingImageNudgedElasticBand('./neb.yml')
runner.set_dataset({'dataIS': initial_data, 'dataFS': final_data})
runner.set_dataloader(ISFSPyGDataLoader)
runner.run(MyModel)
```

For a dimer search, use `StructureOptimization` with `mode='ts'` and provide
a direction through `set_dimer_init_direction` or configured data.

## Metropolis Monte Carlo

```python
from BUCToolkit.api.MonteCarlo import MonteCarlo

runner = MonteCarlo('./mc.yml')
runner.set_dataset({'data': data_list})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)
```

`MC.ITER_SCHEME` controls coordinate perturbations and `T_SCHEME` controls
constant or changing temperature.

## Low-level MD

```python
from BUCToolkit import BatchMD

md = BatchMD.NVT(
    time_step=1.0,
    max_step=1000,
    thermostat='CSVR',
    thermostat_config={'time_const': 120.},
    T_init=298.15,
    output_file='./traj.bin',
    output_structures_per_step=10,
    device='cuda:0',
    verbose=1,
)
md.run(
    func=energy,
    X=X,
    Element_list=elements,
    Cell_vector=cell,
    V_init=velocities,
    grad_func=gradient,
    func_args=(),
    grad_func_args=(),
    is_grad_func_contain_y=False,
    require_grad=False,
    batch_indices=batch_indices,
    fixed_atom_tensor=fixed,
    move_to_center_freq=20,
)
```

Use `BatchMD.NVE` for microcanonical dynamics. `ConstrNVE` and `ConstrNVT`
add `constr_func`, `constr_val`, `constr_threshold`, and optionally
`require_fixman`.

## Low-level Optimization

```python
from BUCToolkit.BatchOptim.minimize import FIRE

opt = FIRE(
    E_threshold=1e-3,
    F_threshold=0.05,
    maxiter=300,
    steplength=0.5,
    device='cuda:0',
    verbose=1,
)
energy_min, coordinates_min = opt.run(
    func=energy,
    X=X,
    grad_func=gradient,
    is_grad_func_contain_y=False,
    batch_indices=batch_indices,
)
```

`CG` additionally selects `iter_scheme` and line-search parameters; `QN` uses
the same base convergence and output options. Request `output_grad=True` for
the final gradient.

## Low-level Transition-state Search

```python
from BUCToolkit.BatchOptim.TS import Dimer

dimer = Dimer(
    E_threshold=1e-3,
    Torque_thres=1e-2,
    Curvature_thres=-0.1,
    F_threshold=0.05,
    maxiter_trans=100,
    maxiter_rot=10,
    max_steplength=0.5,
    dx=1e-2,
)
dimer.run(
    func=energy,
    X=X,
    X_diff=X_diff,
    grad_func=gradient,
    batch_indices=batch_indices,
)
```

`CI_NEB` receives initial/final coordinates and uses `N_images`, `spring_const`,
`optimizer_configs`, convergence thresholds, and `maxiter`. `KrylovNewton` and
`KrylovDynamics` additionally configure the eigen solver; their complete
method-specific signatures are in the source docstrings.

## Low-level Frequency Analysis

```python
from BUCToolkit.BatchOptim.frequency import Frequency

frequency = Frequency(
    method='EnergyDiff',
    block_size=8,
    delta=1e-2,
    output_file='./frequency.bin',
    dump_hessian=True,
)
frequencies, normal_modes = frequency.normal_mode(
    func=energy,
    coords=X[0],
    masses=masses,
    func_args=(),
    func_kwargs={},
)
```

Use `method='GradDiff'` with a gradient callable or `method='Autograd'` for a
differentiable energy model.

## Low-level MC

```python
from BUCToolkit.BatchMC import MMC

mc = MMC(
    iter_scheme='Gaussian',
    maxiter=10000,
    temperature_init=298.15,
    temperature_scheme='constant',
    coordinate_update_param=0.2,
    output_file='./mc.bin',
)
mc.run(func=energy, X=X, Element_list=elements, Cell_vector=cell)
```

The same function forwarding and batch-layout rules apply to low-level MC.

## Read and Convert Results

```python
from BUCToolkit.io import read_md_traj, read_mc_traj, read_opt_structures

md_structures = read_md_traj('./traj.bin', indices=-1)
mc_structures = read_mc_traj('./mc.bin', indices=-1)
opt_structures = read_opt_structures('./opt.bin', indices=-1)
md_structures.write2text('./frames', file_format='POSCAR')
```

## Postprocessing

```python
from BUCToolkit.Postprocessing.trajectory import MDTrajectory
from BUCToolkit.Postprocessing.blue_moon import BlueMoonCalculator
from BUCToolkit.Postprocessing.constraint_work import ConstraintWorkCalculator
from BUCToolkit.Postprocessing.vibrational_spectrum import VibrationalSpectrumCalculator

trajectory = MDTrajectory('./cmd.bin', indices=-1, is_copy=True)
print(trajectory.available_columns)

free_energy = BlueMoonCalculator(trajectory).calculate(
    cv_func=constr_func,
    temperature=298.15,
    use_fixman=False,
    image_order=None,
)
work = ConstraintWorkCalculator(trajectory).calculate(
    cv_func=constr_func,
    temperature=298.15,
    start=0,
    stop=None,
    stride=1,
    integration='right',
)
spectrum = VibrationalSpectrumCalculator(
    trajectory,
    sample_spacing_fs=10.,
).calculate()
```

These calculators can use dumped columns or supplied collective-variable
values. Frame ranges use `start`, exclusive `stop`, and positive `stride`; the
complete accepted options and result fields are documented in source
docstrings.

## Links

- [Parameter manual](manual.md)
- [中文参数手册](manual-zh.md)
- [中文示例](examples-zh.md)
- [README](../README.md)
