#  Copyright (c) 2026.4.1, BUCToolkit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 1.0b
#  File: input_stub.py
#  Environment: Python 3.12
from typing import Any

# The parameter dictionary in the format:
# key: (value, type, description)
CONFIG_STUB = {
    "TASK": ("MD", str, "task name. Options: 'OPT', 'TS', 'VIB', 'NEB', 'MD', 'CMD', 'MC'"),
    "START": (1, int, "0: from scratch; 1: load checkpoint from LOAD_CHK_FILE_PATH; 2: only load model parameters/weights"),
    "VERBOSE": (1, int, "verbosity level for log output"),
    "DEVICE": ("cuda:0", str, "the device on which the task would run"),
    "BATCH_SIZE": (16, int, "the batch size of input data during calculation"),

    # I/O configs
    "LOAD_CHK_FILE_PATH": ("your/model/checkpoint/file/path", str, "path of model checkpoint to load"),
    "OUTPUT_PATH": ("your/log/output/path", str, "path to save logs"),
    "OUTPUT_POSTFIX": ("your_logfile_suffix", str, "suffix for log file name"),
    "PREDICTIONS_SAVE_FILE": ("your/model/predictions/save/path", str, "path of saving predictions"),
    "STRICT_LOAD": (True, bool, "whether to strictly load model parameter"),
    "REDIRECT": (True, bool, "whether output training logs to OUTPUT_PATH or directly print on screen"),
    "SAVE_PREDICTIONS": (True, bool, "only for predictions. Whether output predictions to a dump file"),
    "DATA_TYPE": ("BS", str, "Data type. Options: 'POSCAR', 'OUTCAR', 'CIF', 'ASE_TRAJ', 'BS', 'OPT', 'MD'"),
    "DATA_PATH": ("/your/data/path", str, "path of data used for calculation. If training, it is viewed as the training set"),
    "DATA_NAME_SELECTOR": (".*$", str, "regular expression to select data names. Only matched names will be loaded"),
    "FSDATA_PATH": ("your/final/state/data/path", str, "path for final state data, used for calculations requiring both initial and final states, e.g., CI-NEB"),
    "DISPDATA_PATH": ("your/displacement/data/path", str, "path for displacement data, used for calculations requiring initial guess of a direction, e.g., Dimer"),
    "VAL_SET_PATH": ("your/validation/set/path", str, "path for validation set, used for training that requires validation data"),
    "VAL_SPLIT_RATIO": (0.1, float, "ratio of validation set in the total dataset. Ignored if VAL_SET_PATH is provided"),
    "DATA_LOADER_KWARGS": ({}, dict, "other keyword arguments for data loader"),
    "IS_SHUFFLE": (False, bool, "whether to randomly shuffle dataset before calculating"),

    # Training
    "TRAIN": {
        "EPOCH": (10, int, "number of training epochs"),
        "VAL_BATCH_SIZE": (20, int, "batch size for validation. default is the same as BATCH_SIZE"),
        "VAL_PER_STEP": (100, int, "validate every VAL_PER_STEP steps. step = BATCH_SIZE * ACCUMULATE_STEP"),
        "VAL_IF_TRN_LOSS_BELOW": (1e5, float, "only validating after training loss < VAL_IF_TRN_LOSS_BELOW"),
        "ACCUMULATE_STEP": (12, int, "gradient accumulation steps"),
        "LOSS": ("Energy_Loss", str, "loss function name. Options: 'MSE', 'MAE', 'Hubber', 'CrossEntropy', 'Energy_Force_Loss', 'Energy_Loss'"),
        "LOSS_CONFIG": {
            "loss_E": ("SmoothMAE", str, "loss function for energy")
        },
        "METRICS": (["E_MAE", "E_R2"], list, "list of metrics to compute. Options: E_MAE, F_MAE, F_MaxE, E_R2, MSE, MAE, R2, RMSE"),
        "METRICS_CONFIG": ({}, dict, "other keyword arguments for metrics"),
        "OPTIM": ("AdamW", str, "optimizer name. Options: 'Adam', 'SGD', 'AdamW', 'Adadelta', 'Adagrad', 'ASGD', 'Adamax', 'FIRE'"),
        "OPTIM_CONFIG": {
            "lr": (2e-4, float, "learning rate")
        },
        "LAYERWISE_OPTIM_CONFIG": {
            "force_block.*": ({"lr": 5e-4}, dict, "learning rate for layers matching regex 'force_block.*'"),
            "energy_block.*": ({"lr": 2e-4}, dict, "learning rate for layers matching regex 'energy_block.*'"),
            ".*_bias_layer.*": ({"lr": 2e-4}, dict, "learning rate for layers matching regex '.*_bias_layer.*'")
        },
        "GRAD_CLIP": (True, bool, "whether to toggle on gradient clip"),
        "GRAD_CLIP_MAX_NORM": (10.0, float, "maximum grad. norm to clip"),
        "GRAD_CLIP_CONFIG": ({}, dict, "other kwargs for torch.nn.utils.clip_grad_norm_"),
        "LR_SCHEDULER": (None, str, "learning rate scheduler name. Options: 'StepLR', 'ExponentialLR', 'ChainedScheduler', 'ConstantLR', 'LambdaLR', 'LinearLR', 'CosineAnnealingWarmRestarts', 'CyclicLR', 'MultiStepLR', 'CosineAnnealingLR', None"),
        "LR_SCHEDULER_CONFIG": ({}, dict, "keyword arguments for the learning rate scheduler"),
        "EMA": (False, bool, "whether to toggle on EMA (Exponential Moving Average)"),
        "EMA_DECAY": (0.999, float, "EMA decay rate"),
    },

    # Relaxation
    "RELAXATION": {
        "ALGO": ("FIRE", str, "optimization algorithm. Options: CG, BFGS, FIRE"),
        "ITER_SCHEME": ("PR+", str, "conjugate gradient iteration scheme. Only for ALGO=CG. Options: 'PR+', 'FR', 'PR', 'WYL'"),
        "E_THRES": (1e4, float, "threshold of energy difference for convergence"),
        "F_THRES": (0.05, float, "threshold of max force for convergence"),
        "MAXITER": (300, int, "maximum number of iterations"),
        "STEPLENGTH": (0.5, float, "initial step length"),
        "USE_BB": (True, bool, "whether to use Barzilai-Borwein I steplength as initial steplength"),
        "LINESEARCH": ("B", str, "line search method. 'Backtrack' with Armijo's condition, 'Wolfe' for weak Wolfe condition, 'Exact' for exact linear search"),
        "LINESEARCH_MAXITER": (8, int, "maximum iterations of line search per outer iteration"),
        "LINESEARCH_THRES": (0.02, float, "threshold for exact line search"),
        "LINESEARCH_FACTOR": (0.5, float, "shrinkage factor for backtracking"),
        "REQUIRE_GRAD": (False, bool, "whether to toggle on auto-gradient during calculation"),
    },

    # Transition state
    "TRANSITION_STATE": {
        "ALGO": ("DIMER", str, "algorithm for transition state search. Options: DIMER"),
        "X_DIFF_ATTR": ("x_dimer", str, "attribute name of initial dimer direction"),
        "E_THRES": (1e-4, float, "energy difference threshold for convergence"),
        "TORQ_THRES": (1e-2, float, "torque threshold during rotation process, i.e., residuals of eigen vector"),
        "F_THRES": (5e-2, float, "force threshold for convergence"),
        "MAXITER_TRANS": (300, int, "maximum number of translation steps"),
        "MAXITER_ROT": (5, int, "maximum number of rotation steps"),
        "MAX_STEPLENGTH": (0.5, float, "maximum step length"),
        "DX": (0.1, float, "finite difference length for Hessian-vector product"),
        "REQUIRE_GRAD": (False, bool, "whether to toggle on auto-gradient during calculation"),
    },

    # Vibration analyses
    "VIBRATION": {
        "METHOD": ("Coord", str, "method for vibration analysis. 'Coord' for finite difference, 'Grad' for auto-grad"),
        "BLOCK_SIZE": (1, int, "block size for tensor/vectorize parallelization"),
        "DELTA": (0.01, float, "finite difference length for Hessian-vector product"),
    },

    # NEB transition state
    "NEB": {
        "ALGO": ("CI-NEB", str, "NEB algorithm. Options: CI-NEB"),
        "N_IMAGES": (7, int, "number of images for CI-NEB calculation"),
        "SPRING_CONST": (5.0, float, "spring constant for NEB"),
        "OPTIMIZER": ("FIRE", str, "optimizer for CI-NEB. Currently only FIRE supported"),
        "STEPLENGTH": (0.2, float, "initial step length for optimizer"),
        "E_THRESHOLD": (1e-3, float, "energy difference threshold for convergence"),
        "F_THRESHOLD": (0.05, float, "force threshold for convergence"),
        "MAXITER": (20, int, "maximum number of iterations"),
        "REQUIRE_GRAD": (False, bool, "whether to toggle on auto-gradient during calculation"),
    },

    # Molecular dynamics
    "MD": {
        "ENSEMBLE": ("NVT", str, "MD ensemble. Options: NVE, NVT"),
        "THERMOSTAT": ("CSVR", str, "thermostat for NVT ensemble. Options: Langevin, VR, Nose-Hoover, CSVR"),
        "THERMOSTAT_CONFIG": {
            "DAMPING_COEFF": (0.01, float, "damping coefficient for Langevin thermostat. Unit: fs^-1"),
            "TIME_CONST": (120, float, "time constant for CSVR thermostat. Unit: fs"),
        },
        "TIME_STEP": (1, float, "MD time step. Unit: fs"),
        "MAX_STEP": (100, int, "total number of MD steps. Total time = TIME_STEP * MAX_STEP"),
        "T_INIT": (298.15, float, "initial temperature. Unit: K. For NVE, used to generate random initial velocities"),
        "OUTPUT_COORDS_PER_STEP": (1, int, "frequency of outputting atom coordinates. If verbose=3, velocities also output"),
        "CONSTRAINTS_FILE": ("./constraints.py", str, "path to constraints function file"),
        "CONSTRAINTS_FUNC": ("func", str, "name of constraints function in CONSTRAINTS_FILE"),
        "REQUIRE_GRAD": (False, bool, "whether to toggle on auto-gradient during calculation"),
    },

    # Model configs
    "MODEL_FILE": ("your/model/file/path/template_model.py", str, "path to model definition file"),
    "MODEL_NAME": ("YourModel", str, "name of the model class in MODEL_FILE"),
    "MODEL_CONFIG": {
        "hyperparameter1": ("xxx", Any, "Model hyperparameter"),
        "hyperparameter2": ("xxx", Any, "Model hyperparameter"),
    },
}