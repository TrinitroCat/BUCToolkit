![image](logo_cut.jpg)
# BUCToolkit

## Table of contents
- [BUCToolkit](#BUCToolkit)
  - [Table of contents](#table-of-contents)
  - [About BUCToolkit](#about-batch-upscaled-catalysis-toolkit)
  - [Installation](#installation)
    - [requirements](#requirements)
    - [pip installation](#pip-installation)
    - [Installation from the source](#installation-from-the-source)
  - [Usage](#usage)
    - [Project structures](#project-structures)
    - [Using as a Python package](#using-as-a-python-package)
    - [Using as an executable program](#using-as-an-executable-program)
    - [Input file template](#input-file-template)
    - [Post-processing](#post-processing)
  - [Features](#features)
    - [Flexible function interfaces](#flexible-function-interfaces)
    - [Batched parallel scheme](#batched-parallel-scheme)
  - [Contact Us](#contact-us)
  - [License](#license)

## About Batch-Upscaled Catalysis Toolkit
BUCToolkit is a PyTorch-based high-performance AI4Science software package of computational chemistry, 
which can perform ***structural optimizations*** (both minimization and transition state search), 
***molecular dynamics*** with/without constraints, and ***Monte Carlo simulations*** by 
using any python function with an interface of `func(X, *args, **kwargs)` that returns energy and
`grad_func(X, *args, **kwargs)` that returns energy gradient (i.e., the negative forces).
The most typical input functions are PyTorch-based **deep-learning models** (of molecular or crystal potentials).
For them, BUCToolkit also provided training and prediction APIs. 

All above functions support **multi-structure batch parallelism** for both **regular batches** 
(structures with the same atom numbers) and **irregular batches** (structures with different atom numbers).
These core functions are highly optimized by operator fusing, cudaGraphs replaying, 
asynchronized dumping/logging by cuda-stream pipelines, and in-place memory calculations.
(see section [Features](#features) for details),

Various tools for handling catalyst structure files and data format to preprocess and postprocess
are also included.

Manuals would be completed soon. You can find the current manuals in [Manual](Manual/).

The project is still a beta version and may change in the future.

## Installation
### Requirements
These following third-party libraries are used:
- **Joblib** (BSD-3-Clause License), Copyright © 2008-2021, The joblib developers.
- **NumPy** (BSD-3-Clause License), Copyright © 2005-2025 NumPy Developers.
- **PyTorch** (BSD-3-Clause License), Copyright © 2016-present Facebook Inc.

These following third-party libraries are optional:
- **DGL** (Apache-2.3 License). Only parts of DGL models are currently supported.
- **torch-geometric** (MIT License). The basic `Data` and `Batch` object have been built-in.
For its other advanced functions, the whole torch-geometric can be installed.
- **ASE** (LGPL-v2.1 License) [ASE](https://gitlab.com/ase/ase/-/tree/master?ref_type=heads). Some functions involving `ase.Atoms` object, format transformation for instance.
- **prompt-toolkit** (BSD-3-Clause License). For a better experience of CLI.
Otherwise, the Python built-in `input(...)` will be used.

See [LICENSES/](LICENSES/) for full license texts of requirements.

### pip Installation
This is a recommended way to install BUCToolkit:
```shell
pip install --upgrade pip
pip install BUCToolkit
```
### Installation from the Source Codes
```shell
git clone https://github.com/TrinitroCat/BUCToolkit.git
pip install ./BUCToolkit
```

## Usage
### Project Structure
```shell
BUCToolkit
|-- __init__.py
|-- io.py
|-- StandardTemplate.py
|-- api
|   |-- ConstrainedMolecularDynamics.py
|   |-- DataLoaders.py
|   |-- __init__.py
|   |-- _io.py
|   |-- Losses.py
|   |-- Metrics.py
|   |-- ModelOptims.py
|   |-- MolecularDynamics.py
|   |-- MonteCarlo.py
|   |-- NEB.py
|   |-- Predictor.py
|   |-- StructureOptimization.py
|   |-- Trainer.py
|   `-- VibrationAnalysis.py
|-- cli/...
|-- BatchGenerate/...
|-- BatchMC/...
|-- BatchMD/...
|-- BatchOptim/...
|-- BatchStructures/...
|-- Preprocessing/...
|-- utils/...
`-- _version.py
```
### Using as a Python Package

In BUCToolkit, the advanced APIs which can be called to execute end-to-end tasks 
are in the path `api/`, and the classes/functions in the directories of `Batch*/` 
are low-level methods.

#### Using APIs

The APIs can directly handle the structures of catalytic systems by converting
them from text files such as `POSCAR` `OUTCAR` `ExtXyz` and `cif` to a torch-geometric Data
format, and output text log files and binary database format files after running them. 
Users only need to import their own torch model class and prepare an input file 
(see [Input File Template](#input-file-template)).

An example of using the API functions is shown as follows:
```python
"""
The example of model training & structure optimizations & molecular dynamics
"""
import torch as th

from BUCToolkit import Structures
from BUCToolkit.io import POSCARs2Feat, OUTCAR2Feat, ExtXyz2Feat, ASETraj2Feat
from BUCToolkit.Preprocessing.preprocessing import CreatePygData
from BUCToolkit.api.DataLoaders import PyGDataLoader
from BUCToolkit.api.Trainer import Trainer
#from YOUR_MODEL_PATH import YOUR_MODEL  # Here import your torch-based model file
# * The model would receive a `torch-geometric.Batch`-like object 
# * and return a dict of {'energy': torch.Tensor, 'forces': torch.Tensor, ...}
YOUR_MODEL = 'your imported model'

# -----------------------------------------------------
#          MODEL TRAINING & VALIDATION
# -----------------------------------------------------

# Set the data path
YOUR_OUTCAR_FILE_PATH = '/your/data/path'

# Load Data
f = Structures()
f.load('/your/training/data/path')  # path of files saved by f.save(`path`)
g = Structures()
g.load('/your/training/data/path')
#g = g.contain_only_in(BUCToolkit.TRANSITION_P_METALS | {'C', 'H'})  # select by elements
#g = g.select_by_sample_id(r'[^_].*')  # select by file name, support regular expression

f = OUTCAR2Feat(YOUR_OUTCAR_FILE_PATH)  # Also can load OUTCAR-format files in parallel, only input path
f.read(['OUTCAR1', '2OUTCAR2', 'OUTCAR3'])  # specific files to read, default to read all files in this path

f = ExtXyz2Feat(YOUR_OUTCAR_FILE_PATH)  # Also can load extxyz-format files in parallel, only input path
f.read(['1.xyz', '2.xyz', '3.xyz'])  # specific files to read, default to read all files in this path

f = ASETraj2Feat(YOUR_OUTCAR_FILE_PATH)  # Also can load ase-trajectory files in parallel, only input path
f.read()  # the same above

f = POSCARs2Feat(YOUR_OUTCAR_FILE_PATH)  # Also can load POSCAR-format files in parallel, only input path
f.read(['POSCAR1', 'POSCAR2', ...])  # but for training, POSCAR has no Energy and forces information, that is not suitable.

# Load Model
model = YOUR_MODEL
inp_file_for_train = './template_train.inp'  # Prepare an input file for training tasks. SEE BELOW.

# Convert Data
train_data_list = CreatePygData(1).feat2data_list(f, n_core=1)  # transfer to torch-geometric Data-like format
val_data_list = CreatePygData(1).feat2data_list(g, n_core=1)

trn_ener = [f[atm.idx].Energies[0] for atm in train_data_list]
trn_forc = [f[atm.idx].Forces[0] for atm in train_data_list]
val_ener = [g[atm.idx].Energies[0] for atm in val_data_list]
val_forc = [g[atm.idx].Forces[0] for atm in val_data_list]
# Finally set data into such dictionary format to input
#   Format: Dict['data': List[Data], 'labels': Dict['energy': List[float], 'forces': List[numpy.ndarray]]]
train_data = {'data':train_data_list, 'labels':{'energy':trn_ener, 'forces':trn_forc}}
valid_data = {'data':val_data_list, 'labels':{'energy':val_ener, 'forces':val_forc}}

# set data loader
dataloader = PyGDataLoader

# set trainer
trainer = Trainer(inp_file_for_train)
trainer.set_dataset(train_data, valid_data) # required, set dataset
trainer.set_dataloader(dataloader, {'shuffle':True})  # required, set dataloader
#trainer.set_loss_fn(Energy_Force_Loss, {'coeff_F':0.})  # optional, set loss manually
#trainer.set_lr_scheduler(lr_scheduler, lr_scheduler_config)  # optional, set scheduler manually
trainer.set_layerwise_optim_config(
    {'force_block.*': {'lr': 1.e-3}, 'energy_block.*': {'lr': 1.e-3}}
)  # optional, set different learning-rate for diff. layers; layer name input supports regular expression

# training
trainer.train(model)

# ----------------------------------
#   FOR STRUCTURE OPTIMIZATION
# ----------------------------------
from BUCToolkit.api.StructureOptimization import StructureOptimization

f = POSCARs2Feat(YOUR_OUTCAR_FILE_PATH)  # For example of POSCARs as the input
f.read(['POSCAR1', 'POSCAR2', ...])
data_list = CreatePygData(1).feat2data_list(f, n_core=1)
data_for_opt = {'data': data_list}  # only required the key of 'data'
inp_file_for_opt = './template_opt.inp'  # input file for optimization
dataloader = PyGDataLoader  # use the same dataloader

optimizer = StructureOptimization(inp_file_for_opt)
optimizer.set_dataset(data_for_opt)
optimizer.set_dataloader(dataloader)
optimizer.relax(model)

# ----------------------------------
#   FOR MOLECULAR DYNAMICS
# ----------------------------------
from BUCToolkit.api.MolecularDynamics import MolecularDynamics

f = POSCARs2Feat(YOUR_OUTCAR_FILE_PATH)  # Also use POSCARs as the example
f.read(['POSCAR1', 'POSCAR2', ...])
data_list = CreatePygData(1).feat2data_list(f, n_core=1)
data_for_md = {'data': data_list}  # only required the key of 'data'
inp_file_for_md = './template_md.inp'  # input file for MD
dataloader = PyGDataLoader  # use the same dataloader

runner = MolecularDynamics(inp_file_for_opt)
runner.set_dataset(data_for_md)
runner.set_dataloader(dataloader)
runner.run(model)
```

#### Using Low-level Functions

As for low-level methods, they are more like general algorithms for optimization, 
saddle point search, Newton dynamics evolution and Monte Carlo samplings. 
Hence, all arguments (such as the function, gradient function, 
target variables and other variables) should be set manually.
A typical low-level function of molecular dynamics is like:
```python
import BUCToolkit as bt

runner = bt.BatchMD.NVT(
    time_step=1.,
    max_step=10000,
    thermostat='CSVR',
    thermostat_config={'time_const': 100.},
    T_init=298.15,
    output_file='./nvt.log',
    output_structures_per_step=10,
    device='cuda:0',
    verbose=1,
)
runner.run(
    func=func,
    X=X,
    Element_list=elements,
    Cell_vector=None,
    V_init=init_velocity,
    grad_func=grad_func,
    func_args=[],
    func_kwargs={},
    grad_func_args=[],
    grad_func_kwargs={},
    is_grad_func_contain_y=False,
    require_grad=False,
    batch_indices=[64, 100, 107, 200, ...],
    fixed_atom_tensor=atoms_fixed,
    move_to_center_freq=10
)
```

### Using as an Executable Program
BUCTookit can also be directly applied as a normal executable program. 
By setting some additional args in the input file (see [Input File Template](#input-file-template))
to specify the data path, data type, model file, and task type, 
users can directly launch the tasks in the shell like:
```shell
buctoolkit -i './input_file.inp'
```
An interactive command-line interface can be used as well by inputting no argument:
```
user@host:/some/path$ buctoolkit
+=======================================================================================+
|                                                                                       |
|                                                                                       |
|    BBBBBB  U     U  CCCCC  TTTTTTT  OOOOO   OOOOO  LL      K     K    II   TTTTTTT    |
|    B     B U     U C     C    T    O     O O     O LL      K   K      II      T       |
|    B     B U     U C          T    O     O O     O LL      K K        II      T       |
|    BBBBBB  U     U C          T    O     O O     O LL      KK         II      T       |
|    B     B U     U C          T    O     O O     O LL      K K        II      T       |
|    B     B U     U C     C    T    O     O O     O LL      K   K      II      T       |
|    BBBBBB   UUUUU   CCCCC     T     OOOOO   OOOOO  LLLLLLL K     K    II      T       |
|                                                                                       |
|                                                                                       |
|  BUCToolkit 1.0. Copyright (c) 2024-2026 Authors: Pu Pengxin, Song Xin, etc.          |
+=======================================================================================+
   |_________________________________________________________________________________| 
  |___________________________________________________________________________________|
Please type the content below. Type "help" to show all commands.
 
>>> help
help      : Show this help message.

exit      : Exit the cli program. One can also press "Ctrl+C" to exit.

verbose   : Reset the verbosity level. Available values: 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'

logto     : Usage: logto `logger`  - Reset where the log information is printed. If `None` is given, logs will be printed into stdout.

show      : Usage: `show [-v]`  - Show the current input file path. `-v` is to fully print the input file content.

edit      : Usage: `edit [path]`  - Edit current configuration file. if [path] is set, current file would try to change to the specified path.

task      : usage: task `task_name` [current input file path].  - Create or edit the configurations of a task. If [current input file path] is not given, default of './task.inp' will be used.
             Available task_name: TRAIN, PREDICT, OPT, STRUCTURE_OPTIMIZATION, STRUC_OPT, DIMER, TS, VIB, VIBRATIONAL_ANALYSIS, NEB, CINEB, CI_NEB, MD, MOLECULAR_DYNAMICS, CMD, CONSTRAINED_MOLECULAR_DYNAMICS, CONSTR_MD, MC, MONTE_CARLO

run       : Launch a task in the cli.

>>> 
```
Some prepared input file templates can be called by the `task` option, 
and existing input file can also be interactively inquired and modified 
in the sub-CLI of the `edit` option.

### Input File Template
The input file should be in YAML format.

Here is a completed input file template that contains all supported tasks.
The variables start with "###" are the additions only required by 
using BUCToolkit as an executable program, and those that start with "#" are normal comments.
```yaml

# input template

# global configs
###TASK: !!str MD          # task name. Options: 'OPT', 'TS', 'VIB', 'NEB', 'MD', 'CMD', 'MC', 'TRAIN', 'PREDICT'
START: !!int 1          # 0: from scratch; 1: load checkpoint from LOAD_CHK_FILE_PATH; 2: only load model parameters/weights
VERBOSE: !!int 1        # verbosity level for log output
DEVICE: !!str 'cuda:0'  # the device on which the task would run
BATCH_SIZE: !!int 16    # the batch size of input data during calculation

# I/O configs
LOAD_CHK_FILE_PATH: !!str your/model/checkpoint/file/path
OUTPUT_PATH: !!str your/log/output/path
OUTPUT_POSTFIX: !!str your_logfile_suffix
PREDICTIONS_SAVE_FILE: !!str your/model/predictions/save/path  # path of saving predictions
STRICT_LOAD: !!bool true  # whether to strictly load model parameter
REDIRECT: !!bool true    # whether output training logs to `OUTPUT_PATH` or directly print on screen.
SAVE_PREDICTIONS: !!bool true  # only for predictions. Whether output predictions to a dump file.
###DATA_TYPE: !!str BS  # Literal['POSCAR', 'OUTCAR', 'CIF', 'ASE_TRAJ', 'BS', 'OPT', 'MD']. BS is the build-in structures format obtained by `Structures().save(...)`
CHK_SAVE_PATH: your/model/checkpoint/file/path/to/save
CHK_SAVE_POSTFIX: your_saved_chk_file_suffix
###DATA_PATH: !!str /your/data/path # the path of data used for calculation. if training, it will be viewed as the training set.
###DATA_NAME_SELECTOR: !!str ".*$"  # regular express to select data names. Only matched name will be finally load.
###FSDATA_PATH: !!str your/final/state/data/path  # used for calc. requiring both initial and final states, e.g., CI-NEB
###DISPDATA_PATH: !!str your/displacement/data/path  # used for calc. requiring initial guess of a direction, e.g., Dimer
###VAL_SET_PATH: !!str your/validation/set/path  # used for training that requires validation data
###VAL_SPLIT_RATIO: !!float 0.1  # the ratio of validation set in the total dataset. if `VAL_SET` is given, this arg will be ignored.
###DATA_LOADER_KWARGS: {}      # other kwargs for data loader.
###IS_SHUFFLE: !!bool false    # whether to randomly shuffle dataset before calculating.

# training
TRAIN:
  # epoches & val set
  EPOCH: !!int 10
  VAL_BATCH_SIZE: !!int 20  # batch size for validation. default is the same as BATCH_SIZE
  VAL_PER_STEP: !!int 100   # validate every `VAL_PER_STEP` steps. step = `BATCH_SIZE` * `ACCUMULATE_STEP`
  VAL_IF_TRN_LOSS_BELOW: !!float 1.e5  # only validating after training loss < `VAL_IF_TRN_LOSS_BELOW`
  ACCUMULATE_STEP: !!int 12  # gradient accumulation steps
  # loss configs
  LOSS: !!str Energy_Loss  # 'MSE': nn.MSELoss, 'MAE': nn.L1Loss, 'Hubber': nn.HuberLoss, 'CrossEntropy': nn.CrossEntropyLoss 'Energy_Force_Loss': Energy_Force_Loss, 'Energy_Loss': Energy_Loss
  LOSS_CONFIG:             # other kwargs for loss function
    loss_E: !!str SmoothMAE

  METRICS:  # tuple of ones in [E_MAE, F_MAE, F_MaxE, E_R2, MSE, MAE, R2, RMSE], F_MaxE is the max absolute error of forces.
    - !!str E_MAE
    - !!str E_R2
  METRICS_CONFIG: {}  # other kwargs for metrics
  #  - F_MaxE  # for example

  # optimizer configs
  OPTIM: !!str AdamW  # model optimizer. Available values:
                      # 'Adam': th.optim.Adam, 'SGD': th.optim.SGD, 'AdamW': th.optim.AdamW, 'Adadelta': th.optim.Adadelta,
                      # 'Adagrad': th.optim.Adagrad, 'ASGD': th.optim.ASGD, 'Adamax': th.optim.Adamax, 'FIRE': FIRELikeOptimizer,
  OPTIM_CONFIG:       # optimizer kwargs
    lr: !!float 2.e-4
    # ...
  LAYERWISE_OPTIM_CONFIG: # Supporting regular expression to selection layers and set them.
    'force_block.*': { 'lr': 5.e-4 }
    'energy_block.*': { 'lr': 2.e-4 }
    '.*_bias_layer.*': { 'lr': 2.e-4 }

  GRAD_CLIP: !!bool true  # whether to toggle on gradient clip
  GRAD_CLIP_MAX_NORM: !!float 10.  # maximum grad. norm to clip
  GRAD_CLIP_CONFIG: {}    # other kwargs for `nn.utils.clip_grad_norm_` function
  LR_SCHEDULER: !!str None  # learning rate scheduler. Available values:
                            # 'StepLR': StepLR, 'ExponentialLR': ExponentialLR, 'ChainedScheduler': ChainedScheduler,
                            # 'ConstantLR': ConstantLR, 'LambdaLR': LambdaLR, 'LinearLR': LinearLR,
                            # 'CosineAnnealingWarmRestarts': CosineAnnealingWarmRestarts, 'CyclicLR': CyclicLR,
                            # 'MultiStepLR': MultiStepLR, 'CosineAnnealingLR': CosineAnnealingLR, 'None': None,
  LR_SCHEDULER_CONFIG: {} # kwargs of above `LR_SCHEDULER`
  EMA: !!bool false       # whether to toggle on EMA (Exponential Moving Average)
  EMA_DECAY: !!float 0.999  # EMA decay rate

# relaxation
RELAXATION:
  ALGO: !!str 'FIRE'  # options: CG, BFGS, FIRE
  ITER_SCHEME: !!str 'PR+'  # only for ALGO=CG, options: 'PR+', 'FR', 'PR', 'WYL'
  E_THRES: !!float 1.e4  # threshold of Energy difference
  F_THRES: !!float 0.05  # threshold of max Force
  MAXITER: !!int 300     # maximum iteration
  STEPLENGTH: !!float 0.5  # initial steplength
  USE_BB: !!bool true    # whether to use Barzilai-Borwein I steplength as initial steplength
  LINESEARCH: !!str 'B'  # 'Backtrack' with Armijo's cond., 'Wolfe' for weak Wolfe cond. reached by M-T algo, 'Exact' for exact linear search
  LINESEARCH_MAXITER: !!int 8  # max iterations of linear search per outer iteration.
  LINESEARCH_THRES: !!float 0.02  # only for LINESEARCH = 'exact', threshold of exact line search.
  LINESEARCH_FACTOR: !!float 0.5  # Shrinkage factor for "Backtrack".
  REQUIRE_GRAD: !!bool False  # whether to toggle on auto-gradient during calculation.

# transition state
TRANSITION_STATE:
  ALGO: !!str DIMER  # options: DIMER
  X_DIFF_ATTR: !!str x_dimer  # the attribute name of initial dimer direction
  E_THRES: !!float 1.e-4    # see above
  TORQ_THRES: !!float 1.e-2  # threshold of torque during rotation process, i.e., the residuals of eigen vec.
  F_THRES: !!float 5.e-2    # see above
  MAXITER_TRANS: !!int 300  # maximum iteration number of translation steps
  MAXITER_ROT: !!int 5      # maximum iteration number of rotation steps
  MAX_STEPLENGTH: !!float 0.5  # limit of steplength
  DX: !!float 1.e-1         # length for finite difference to calc. Hessian-vector prod.
  REQUIRE_GRAD: !!bool False  # see above

# vibration analyses (harmonic)
VIBRATION:
  METHOD: !!str 'Coord'  # Coord/Grad corresponding to finite difference and auto-grad scheme.
  BLOCK_SIZE: !!int 1    # block-size of tensor/vectorize parallelization
  DELTA: !!float 1e-2    # length for finite difference to calc. Hessian-vector prod.

# NEB transition state
NEB:
  ALGO: !!str 'CI-NEB'   # option: CI-NEB
  N_IMAGES: !!int 7      # images number for CI-NEB calc.
  SPRING_CONST: 5.0      # spring constant for NEB
  OPTIMIZER: !!str FIRE  # optimizer for CI-NEB. Now only support FIRE.
  #OPTIMIZER_CONFIGS: Optional[Dict[str, Any]] = None, other kwargs of optimizer.
  STEPLENGTH: !!float 0.2     # see args in section `RELAXATION`
  E_THRESHOLD: !!float 1.e-3  # see args in section `RELAXATION`
  F_THRESHOLD: !!float 0.05   # see args in section `RELAXATION`
  MAXITER: !!int 20           # see args in section `RELAXATION`
  REQUIRE_GRAD: !!bool False  # see args in section `RELAXATION`

# molecular dynamics
MD:
  ENSEMBLE: !!str NVT     # MD ensemble. options: NVE, NVT
  THERMOSTAT: !!str CSVR  # only for ENSEMBLE=NVT, 'Langevin', 'VR', 'Nose-Hoover', 'CSVR'
  THERMOSTAT_CONFIG:      # thermostat configs
    DAMPING_COEFF: !!float 0.01  # damping coefficient for Langevin thermostat. unit: fs^-1
    TIME_CONST: !!float 120      # time constant for CSVR thermostat. unit: fs
  TIME_STEP: !!float 1  # MD time step. unit: fs
  MAX_STEP: !!int 100  # total time (fs) = TIME_STEP * MAX_STEP
  T_INIT: !!float 298.15  # Initial Temperature, Unit: K. For ENSEMBLE=NVE, T_INIT is only used to generate ramdom initial velocities by Boltzmann dist.
  OUTPUT_COORDS_PER_STEP: !!int 1  # To control the frequency of outputting atom coordinates. If verbose = 3, atom velocities would also be outputted.
  MOVE_TO_CENTER_FREQ: !!int 20   # how many steps that move atoms to the barycenter and zeroize the bulk velocities
  # Optional: constraints
  CONSTRAINTS_FILE: !!str ./constraints.py  # function file path of constraints. This function should receive torch.Tensors and support auto-grad.
  CONSTRAINTS_FUNC: !!str func              # the specific function name in `CONSTRAINTS_FILE`
  REQUIRE_GRAD: !!bool False  # see above

# Monte Carlo
MC:
  TYPE: !!str Metropolis  # now only `Metropolis` is supported.
  ITER_SCHEME: !!str 'Gaussian'  # Literal['Gaussian', 'Cauchy', 'Uniform'], the distribution for perturbing atoms.
  COORDINATE_UPDATE_PARAM: !!float 0.2  # the parameter to control atoms movement. It is the std for 'Gaussian', half interval length for 'Uniform', and the half width at half maximum for 'Cauchy'
  MAXITER: !!int 10000
  T_INIT: !!float 298.15    # initial temperature
  T_SCHEME: !!str constant  # Literal['constant', 'linear', 'exponential', 'log', 'fast'], the way to change MC temperature. See class `BatchMC.MetropolisMC.MMC` for details.
  T_UPDATE_FREQ: !!int 1    # how many time steps that updates temperature once
  T_SCHEME_PARAM: !!float 0.  # the parameter to control temperature updating. See class `BatchMC.MetropolisMC.MMC` for details.
  OUTPUT_COORDS_PER_STEP: !!int 1  # see `MD` section above
  MOVE_TO_CENTER_FREQ: !!int 20    # see `MD` section above

# model configs
###MODEL_FILE: !!str your/model/file/path/template_model.py  # function file path of torch models
MODEL_NAME: !!str YourModel   # the specific name of the model in `MODEL_FILE`
MODEL_CONFIG:   # model hyperparameters used for `MODEL_NAME.__init__(**MODEL_CONFIG)`
  hyperparameter1: xxx
  hyperparameter2: xxx
  # ...

```

### Post-processing
There are two outputs of BUCToolkit tasks, text log file and binary database file.

#### Log Files
For API or executables, the output of log file is set by `REDIRECT: true` with `OUTPUT_PATH` and `OUTPUT_POSTFIX`, and the contents are controlled by `VERBOSE` in the input file. If `REDIRECT` is `false`, outputs will be printed to `sys.stdout`.

Low-level functions are controlled by the logger system. For details, see `BUCToolkit/utils/setup_loggers.py`.

Because of the large costs of string processing, high-level verbosity is NOT recommended. All information 
is included in the binary database file, so turning off the logger (by setting verbose = 0) is completely feasible.

#### Binary Database
This is a specially designed binary format to efficiently dump arrays. Memory-mapping is supported for both writing
and reading. Its specific format is shown in the class `ArrayDumper` of `BUCToolkit/BatchStructure/StructureIO.py`.

To control the binary file output, args of `SAVE_PREDICTIONS: true` with a `PREDICTIONS_SAVE_FILE` should
be set in the input file. For low-level functions, `output_file` is the related argument.

For the binary output files from structure optimization, molecular dynamics, and Monte Carlo simulations, 
one can load & convert them in the shell as follows:
```shell
buctoolkit -c `$input_type` `$input_path` `$output_type` `$output_path`
# `$input_path` can be one of "bs", "md", "mc", "opt", "outcar", "poscar", "cif", and "ase_traj"
# `$output_type` can be one of "poscar", "cif", "xyz", "bs".
```
This command will convert all files in `$input_path` with assumed format of `$input_type` into 
`$output_path` in the format of `$output_type`.

For a finer control, the following python script can be used:
```python
import BUCToolkit as bt
from BUCToolkit.io import read_opt_structures, read_md_traj, read_mc_traj

# for output file of structure optimizations
bs1: bt.Structures = read_opt_structures('your_opt_output', indices=-1)

# for output file of MD
bs2: bt.Structures = read_md_traj("your_md_output", indices=-1)

# for output file of MC
bs3: bt.Structures = read_mc_traj("your_mc_output", indices=-1)

# bt.Structures is a class to manage batch structures. These read_* methods above will convert 
#   the output information into this `bt.Structures` format, and then structures can be converted
#   to other text files by:
bs1.write2text(output_path='./a/directory', indices=None, file_format='POSCAR', )

# Or they can be dumped into another binary files by:
bs2.save('./a/directory')

# Or they can be directly applied to other calculations
from BUCToolkit.Preprocessing.preprocessing import CreatePygData, CreateASE, CreateDglData

atoms = CreateASE().feat2ase(bs3)
data = CreatePygData().feat2data_list(bs3)
graph = CreateDglData().feat2graph_list(bs3)

```
Wherein, the args of `indices` specify the selected parts to read and write instead of all files.

## Features

BUCToolkit employed highly optimized PyTorch code, including fused operators, cudaGraphs replaying, 
asynchronized dumping/logging by cuda-stream pipelines, and in-place memory calculations. 

### Flexible function interfaces
Major low-level functions use very flexible interfaces as follows 
(also see [Using Low-level Functions](#using-low-level-functions)):
```
function(
    func=func,
    X=X,
    grad_func=grad_func,
    func_args=[],
    func_kwargs=None,
    grad_func_args=[],
    grad_func_kwargs=None,
    is_grad_func_contain_y=False,
    require_grad=False,
    ...
)
```
where the `X` is the target variable to update (e.g., the atom positions for molecular dynamics 
and structure optimizations), `func_args` and `func_kwargs` are other necessary arguments and 
keyword arguments for the `func`. Hence, any `func`, as long as it can be wrapped as
`func(X, *args, **kwargs)`, is valid. For example, one may write a function that submits ab initio 
computations (e.g., VASP, Gaussian) and convert the results (energy and forces) into torch.Tensor format, 
and BUCToolkit functions can execute with these inputs normally.

The `grad_func` has a similar design. 
The argument `is_grad_func_contain_y` controls two ways to calculate the gradient of `func`. 
`is_grad_func_contain_y = True` is to use auto-gradient format, that actually uses 
`grad_func(X, y, *grad_func_args, **grad_func_kwargs)` internally 
(Note: user would not manually put `y` into the `grad_func_args`), otherwise, interfaces of 
`grad_func(X, *grad_func_args, **grad_func_kwargs)` are used. At last, `require_grad` controls the
gradient context of PyTorch. When `require_grad = False`, computation of `func` and `grad_func` is under
the context of `torch.no_grad` to reduce memory cost. Otherwise, gradient will be turned on explicitly
by `torch.enable_grad`.

### Highly customizable algorithms
All methods/algorithms are object-oriented modularized. They have `_Base*` abstract base classes 
that implement highly optimized main loop routines, and are specialized by modifying few methods like 
`self.initialize*(...)` and `self._update*(...)` in subclasses. Hence, one can develop and implement any 
custom new algorithm by simply overriding these update methods without modifying the main loop process.

### Batch parallelism scheme
Most functions, including structure optimization, transition state search, molecular dynamics, and 
Monte Carlo simulation, support the parallel for **both regular batched samples 
(stacked samples with the same atom numbers) and irregular batched samples 
(concatenated samples with different atom numbers)**. 
Input Tensors (of atom coordinates, forces, fixation masks, etc.) should be 3-dimensional. For regular batches,
their shapes are **(batch_size, n_atom, n_dim)**, where `n_dim` is usually be 3. For irregular batches, their 
shapes are **(1, $\sum_{i}$n_atom$_{i}$, n_dim)**, where $i$ is the sample index, and users should provide 
another variable `batch_indices` that records atom numbers of each sample. For example, 
`batch_indices = [64, 56, 72, 83, 102]` means samples have 64, 56, 72, 83, 102 atoms, respectively, and 
corresponding shapes of atom coordinates should be `(1, 377, 3)`.

For structure optimization and transition state search, BUCToolkit applies a **dynamic samples approach**, that 
is dynamically removing the converged samples in one batch before starting next iteration steps 
by maintaining a convergence mask and `indexed_select`/`indexed_copy_` functions. It could significantly reduce
the waste of repeatedly calculating the converged data.

## Contact Us

If you have any questions, please contact us at **buctoolkit@163.com**

For bug reports or feature requests, please use [GitHub Issues](https://github.com/TrinitroCat/BUCToolkit/issues).

## License

The code of BUCToolkit is published and distributed under the **[MIT License](https://github.com/TrinitroCat/BUCToolkit/blob/main/LICENSE)**.

