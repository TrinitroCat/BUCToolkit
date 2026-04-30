
![image](logo_cut.jpg)

# BUCToolkit

## 目录
- [关于 BUCToolkit](#关于-buctoolkit)
- [安装](#安装)
  - [依赖要求](#依赖要求)
  - [pip 安装](#pip-安装)
  - [从源码安装](#从源码安装)
- [使用说明](#使用说明)
  - [项目结构](#项目结构)
  - [单位约定](#单位约定)
  - [作为 Python 包使用](#作为-python-包使用)
  - [作为可执行程序使用](#作为可执行程序使用)
  - [输入文件模板](#输入文件模板)
  - [后处理](#后处理)
- [功能特性](#功能特性)
  - [灵活的函数接口](#灵活的函数接口)
  - [高度可定制的算法](#高度可定制的算法)
  - [批量并行方案](#批量并行方案)
  - [基于自动微分的强大约束求解器](#基于自动微分的强大约束求解器)
- [联系我们](#联系我们)
- [许可证](#许可证)

## 关于 BUCToolkit

BUCToolkit 是一个基于 PyTorch 的高性能 AI4Science 计算化学软件包。
它能够执行***结构优化***（包括能量最小化和过渡态搜索）、***分子动力学***（带/不带约束）以及 ***蒙特卡洛模拟***，
可用于任何符合 `func(X, *args, **kwargs)` 和 `grad_func(X, *args, **kwargs)` 接口的 
Python 函数（返回能量及其梯度，即力的相反数）。
最典型的输入函数是基于 PyTorch 的 **深度学习模型**（用于分子或晶体势能）。对于这类模型，BUCToolkit 提供了专门的训练和预测 API。

上述所有功能都支持**多结构批量并行**，包括**规则批次**（原子数相同的结构）和**不规则批次**（原子数不同的结构）。
这些核心函数通过算子融合、CUDA Graph 回放、基于 CUDA 流管道的异步转储/日志记录以及原地内存计算等优化手段实现了高度性能优化
（详见[功能特性](#功能特性)一节）。

此外，软件还集成了多种工具，用于处理催化剂结构文件和不同格式的数据预处理/后处理。

完整文档将陆续完成。当前手册可在 [Manual](Manual) 目录下找到。

请注意，本项目目前仍处于测试版，未来可能发生变化。

## 安装

### 依赖要求
本软件使用了以下第三方库：
- **Joblib**（BSD-3-Clause 许可证），版权所有 © 2008-2021，The joblib developers。
- **NumPy**（BSD-3-Clause 许可证），版权所有 © 2005-2025 NumPy Developers。
- **PyTorch**（BSD-3-Clause 许可证），版权所有 © 2016-present Facebook Inc。

以下第三方库为可选项：
- **DGL**（Apache-2.3 许可证）。目前仅支持部分 DGL 模型。
- **torch-geometric**（MIT 许可证）。目前程序已经内置了基本的 `Data` 和 `Batch` 对象。
如需其他高级功能，可完整安装 torch-geometric。
- **[ASE](https://gitlab.com/ase/ase/-/tree/master?ref_type=heads)**（LGPL-v2.1 许可证）。用于涉及 `ase.Atoms` 对象的函数，例如格式转换。
- **prompt-toolkit**（BSD-3-Clause 许可证）。提供更好的命令行交互体验，否则将使用 Python 内置的 `input(...)`。

完整的许可证文本请参见 [LICENSES/](LICENSES) 目录。

### pip 安装
推荐使用以下方式安装 BUCToolkit：
```shell
pip install --upgrade pip
pip install BUCToolkit
```

### 从源码安装
```shell
git clone https://github.com/TrinitroCat/BUCToolkit.git
pip install ./BUCToolkit
```

## 使用说明

### 项目结构
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
|-- Bases/...
|-- BatchGenerate/...
|-- BatchMC/...
|-- BatchMD/...
|-- BatchOptim/...
|-- BatchStructures/...
|-- Preprocessing/...
|-- utils/...
`-- _version.py
```

### 单位约定
BUCToolkit 不提供自动单位转换，所有单位均遵循以下约定：
- 长度：埃（Å）
- 能量：电子伏特（eV）
- 质量：原子质量单位（amu），数值上等于摩尔质量（g/mol）
- 时间：飞秒（fs）
- 温度：开尔文（K）

### 作为 Python 包使用

BUCToolkit 中，能够执行端到端任务的高级 API 位于 `api/` 目录下，而 `Batch*/` 目录中的类/函数属于底层方法。

#### 使用高级 API

高级 API 可以直接处理催化剂结构系统，将它们从文本文件（如 `POSCAR`、`OUTCAR`、`ExtXyz`、`cif`）
转换为 torch‑geometric 的 Data 格式，并在运行后输出文本日志文件和二进制数据库文件。
用户只需导入自己的 torch 模型类并准备一个输入文件（参见[输入文件模板](#输入文件模板)）。

以下是一个使用 API 函数***训练***基于 torch‑geometric 的模型的示例：

```python
"""
模型训练 & 结构优化 & 分子动力学示例
"""
import torch as th

from BUCToolkit import Structures
from BUCToolkit.io import POSCARs2Feat, OUTCAR2Feat, ExtXyz2Feat, ASETraj2Feat, Cif2Feat
from BUCToolkit.Preprocessing.preprocessing import CreatePygData
from BUCToolkit.api.DataLoaders import PyGDataLoader
from BUCToolkit.api.Trainer import Trainer

# from YOUR_MODEL_PATH import YOUR_MODEL  # 在此导入您的基于 torch 的模型文件
# * 模型应接收一个 torch‑geometric.Batch 类对象
# * 并返回字典 {'energy': torch.Tensor, 'forces': torch.Tensor, ...}
YOUR_MODEL = '您导入的模型'

# -----------------------------------------------------
#          模型训练与验证
# -----------------------------------------------------

# 设置数据路径
YOUR_DATA_PATH = '/your/data/path'

#############
# 加载数据 #
#############
#   加载 Build‑in 结构格式数据（BUCToolkit 内置格式）   <<<<<<<
f = Structures()
f.load('/your/training/data/path')  # 由 f.save(`path`) 保存的文件路径
g = Structures()
g.load('/your/training/data/path')
# g = g.contain_only_in(BUCToolkit.TRANSITION_P_METALS | {'C', 'H'})  # 按元素筛选
# g = g.select_by_sample_id(r'[^_].*')  # 按文件名筛选，支持正则表达式

#   并行加载 OUTCAR 格式文件   <<<<<<<
f = OUTCAR2Feat(YOUR_DATA_PATH)  # 仅需提供 OUTCAR 所在路径
f.read(['OUTCAR1', '2OUTCAR2', 'OUTCAR3'], n_core=1)  # 指定要读取的文件，默认读取路径下所有文件

#   并行加载 extxyz 格式文件   <<<<<<<
f = ExtXyz2Feat(YOUR_DATA_PATH)
f.read(
  ['1.xyz', '2.xyz', '3.xyz'],
  lattice_tag='lattice',
  energy_tag='energy',
  column_info_tag='properties',
  element_tag='species',
  coordinates_tag='pos',
  forces_tag=None,
  fixed_atom_tag=None
)

#   并行加载 cif 格式文件   <<<<<<<
f = Cif2Feat(YOUR_DATA_PATH)
f.read(['1.cif', '2.cif', '3.cif'], n_core=1)

#   加载 ASE 轨迹文件（需要安装 ASE）   <<<<<<<
f = ASETraj2Feat(YOUR_DATA_PATH)
f.read(['1.trj', '2.trj', '3.trj'], n_core=1)

#   加载 POSCAR 文件   <<<<<<<
# 注意：训练时 POSCAR 不包含能量和力信息，因此不适用。
f = POSCARs2Feat(YOUR_DATA_PATH)
f.read(['POSCAR1', 'POSCAR2', ...])

##############
# 加载模型   #
##############
your_model_class = YOUR_MODEL  # 通过 import 导入
inp_file_for_train = './template_train.inp'  # 为训练任务准备输入文件，见下文

################
# 转换数据     #
################
# 转换为 torch‑geometric Data 格式
train_data_list = CreatePygData(1).feat2data_list(f, n_core=1)
val_data_list = CreatePygData(1).feat2data_list(g, n_core=1)

trn_ener = [f[atm.idx].Energies[0] for atm in train_data_list]
trn_forc = [f[atm.idx].Forces[0] for atm in train_data_list]
val_ener = [g[atm.idx].Energies[0] for atm in val_data_list]
val_forc = [g[atm.idx].Forces[0] for atm in val_data_list]
# 最终将数据组织为以下字典格式输入
#   格式: Dict['data': List[Data], 'labels': Dict['energy': List[float], 'forces': List[numpy.ndarray]]]
train_data = {'data': train_data_list, 'labels': {'energy': trn_ener, 'forces': trn_forc}}
valid_data = {'data': val_data_list, 'labels': {'energy': val_ener, 'forces': val_forc}}

###################
# 设置数据加载器  #
###################
dataloader = PyGDataLoader

###############
# 设置训练器  #
###############
trainer = Trainer(inp_file_for_train)
trainer.set_dataset(train_data, valid_data)  # 必需，设置数据集
trainer.set_dataloader(dataloader, {'shuffle': True})  # 必需，设置数据加载器
# trainer.set_loss_fn(Energy_Force_Loss, {'coeff_F':0.})  # 可选，手动设置自定义损失函数
# trainer.set_lr_scheduler(lr_scheduler, lr_scheduler_config)  # 可选，手动设置学习率调度器
trainer.set_layerwise_optim_config(
  {'force_block.*': {'lr': 1.e-3}, 'energy_block.*': {'lr': 1.e-3}}
)  # 可选，为不同层设置不同学习率；层名支持正则表达式

############
# 开始训练 #
############
trainer.train(your_model_class)  # 仅传入模型类而不要实例化
```

训练好的模型可以用于**结构优化**和**分子动力学**模拟，如下所示：

```python
import BUCToolkit as bt
from BUCToolkit.Preprocessing import preprocessing
import your_model_file as your_model_class

YOUR_DATA_PATH = '/your/data/path'
# ----------------------------------
#   结构优化
# ----------------------------------
from BUCToolkit.api.StructureOptimization import StructureOptimization

f = bt.io.POSCARs2Feat(YOUR_DATA_PATH)  # 以 POSCAR 为例
f.read(['POSCAR1', 'POSCAR2', ...])
data_list = preprocessing.CreatePygData().feat2data_list(f, n_core=1)
data_for_opt = {'data': data_list}  # 仅需 'data' 键
inp_file_for_opt = './template_opt.inp'  # 优化的输入文件
dataloader = bt.api.DataLoaders.PyGDataLoader

optimizer = StructureOptimization(inp_file_for_opt)
optimizer.set_dataset(data_for_opt)
optimizer.set_dataloader(dataloader)
optimizer.relax(your_model_class)  # 仅传入模型类而不要实例化

# ----------------------------------
#   分子动力学
# ----------------------------------
from BUCToolkit.api.MolecularDynamics import MolecularDynamics

f = bt.io.POSCARs2Feat(YOUR_DATA_PATH)
f.read(['POSCAR1', 'POSCAR2', ...])
data_list = preprocessing.CreatePygData().feat2data_list(f, n_core=1)
data_for_md = {'data': data_list}
inp_file_for_md = './template_md.inp'  # MD 的输入文件
dataloader = bt.api.DataLoaders.PyGDataLoader

runner = MolecularDynamics(inp_file_for_md)
runner.set_dataset(data_for_md)
runner.set_dataloader(dataloader)
runner.run(your_model_class)  # 仅传入模型类而不要实例化
```

#### 使用底层函数

底层方法是一类通用算法引擎，如多元函数极小化、鞍点搜索、牛顿动力学方程积分和蒙特卡洛采样等。
因此，所有参数（函数、梯度函数、目标变量及其他变量）都需要手动设置。一个典型的底层 MD 函数示例如下：

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

### 作为可执行程序使用

BUCToolkit 也可以直接作为普通可执行程序使用。通过在输入文件中设置一些额外参数（见[输入文件模板](#输入文件模板)），指定数据路径、数据类型、模型文件和任务类型，用户可以在终端中直接启动任务：

```shell
buctoolkit -i './input_file.inp'
```

如果不带任何参数运行，则进入交互式命令行界面：

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
请输入以下内容。输入 "help" 显示所有命令。
 
>>> help
help      : 显示本帮助信息。

exit      : 退出 CLI 程序，也可按 Ctrl+C 退出。

verbose   : 重新设置日志详细程度。可选值：'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'

logto     : 用法: logto `logger`  - 重新设置日志输出的位置。如果为 None，日志将打印到 stdout。

show      : 用法: `show [-v]`  - 显示当前输入文件路径。使用 `-v` 将完整打印输入文件内容。

edit      : 用法: `edit [path]`  - 编辑当前配置文件。若指定 [path]，则将当前文件切换到该路径。

task      : 用法: task `task_name` [current input file path]  - 创建或编辑某任务的配置。若不指定路径，默认使用 './task.inp'。
             可用 task_name: TRAIN, PREDICT, OPT, STRUCTURE_OPTIMIZATION, STRUC_OPT, DIMER, TS, VIB, VIBRATIONAL_ANALYSIS, NEB, CINEB, CI_NEB, MD, MOLECULAR_DYNAMICS, CMD, CONSTRAINED_MOLECULAR_DYNAMICS, CONSTR_MD, MC, MONTE_CARLO

run       : 在 CLI 中启动一个任务。

>>> 
```

一些预置的输入文件模板可通过 `task` 选项调用，现有输入文件也可以通过 `edit` 选项的子 CLI 进行交互式查询和修改。

### 输入文件模板

输入文件采用 YAML 格式。以下是一个完整的模板，包含所有支持的任务。以 `###` 开头的变量是仅在使用 BUCToolkit 作为可执行程序时需要的额外参数，以 `#` 开头的是普通注释。

```yaml
# 输入文件模板

# 全局配置
###TASK: !!str MD          # 任务名称。可选：'OPT', 'TS', 'VIB', 'NEB', 'MD', 'CMD', 'MC', 'TRAIN', 'PREDICT'
START: !!int 1          # 0: 从头开始；1: 从 LOAD_CHK_FILE_PATH 加载检查点继续；2: 仅加载模型参数/权值
VERBOSE: !!int 1        # 日志输出的详细程度
DEVICE: !!str 'cuda:0'  # 任务运行的设备
BATCH_SIZE: !!int 16    # 计算时输入数据的批大小

# I/O 配置
LOAD_CHK_FILE_PATH: !!str your/model/checkpoint/file/path
OUTPUT_PATH: !!str your/log/output/path
OUTPUT_POSTFIX: !!str your_logfile_suffix
PREDICTIONS_SAVE_FILE: !!str your/model/predictions/save/path  # 保存预测结果的文件路径
STRICT_LOAD: !!bool true  # 是否严格加载模型参数
REDIRECT: !!bool true    # 是否将训练日志重定向到 OUTPUT_PATH，还是直接打印到屏幕
SAVE_PREDICTIONS: !!bool true  # 仅用于预测。是否将预测结果输出到转储文件
###DATA_TYPE: !!str BS  # 可选值：'POSCAR', 'OUTCAR', 'CIF', 'ASE_TRAJ', 'BS', 'OPT', 'MD'。BS 是 Structures().save(...) 保存的内置结构格式
SAVE_CHK: !!bool true  # 训练过程中是否保存检查点
CHK_SAVE_PATH: !!str your/model/checkpoint/file/path/to/save  # 检查点保存路径
CHK_SAVE_POSTFIX: !!str your_saved_chk_file_suffix  # 检查点文件后缀

###DATA_PATH: !!str /your/data/path # 用于计算的数据路径；如果是训练，将被视为训练集
###DATA_NAME_SELECTOR: !!str ".*$"  # 用于选择数据名称的正则表达式，仅加载匹配的名称
###FSDATA_PATH: !!str your/final/state/data/path  # 用于需要初态和终态的计算，如 CI‑NEB
###DISPDATA_PATH: !!str your/displacement/data/path  # 用于需要初始方向猜测的计算，如 Dimer
###VAL_SET_PATH: !!str your/validation/set/path  # 训练时需要的验证集路径
###VAL_SPLIT_RATIO: !!float 0.1  # 验证集占总数据集的比例；若指定了 VAL_SET_PATH，此参数被忽略
###DATA_LOADER_KWARGS: {}      # 数据加载器的其他关键字参数
###IS_SHUFFLE: !!bool false    # 计算前是否随机打乱数据集

# 训练配置
TRAIN:
  # 轮次和验证集
  EPOCH: !!int 10
  VAL_BATCH_SIZE: !!int 20  # 验证时的批大小，默认与 BATCH_SIZE 相同
  VAL_PER_STEP: !!int 100   # 每 VAL_PER_STEP 步验证一次。步数 = BATCH_SIZE * ACCUMULATE_STEP
  VAL_IF_TRN_LOSS_BELOW: !!float 1.e5  # 仅当训练损失低于该值时进行验证
  ACCUMULATE_STEP: !!int 12  # 梯度累积步数
  # 损失函数配置
  LOSS: !!str Energy_Loss  # 可选：'MSE'（nn.MSELoss）、'MAE'（nn.L1Loss）、'Hubber'（nn.HuberLoss）、'CrossEntropy'（nn.CrossEntropyLoss）、'Energy_Force_Loss'、'Energy_Loss'
  LOSS_CONFIG:             # 损失函数的其他参数
    loss_E: !!str SmoothMAE

  METRICS:  # 评价指标元组，可选：E_MAE, F_MAE, F_MaxE, E_R2, MSE, MAE, R2, RMSE。F_MaxE 是力的最大绝对误差
    - !!str E_MAE
    - !!str E_R2
  METRICS_CONFIG: {}  # 评价指标的其他参数
  #  - F_MaxE  # 示例

  # 优化器配置
  OPTIM: !!str AdamW  # 模型优化器。可选值：'Adam', 'SGD', 'AdamW', 'Adadelta', 'Adagrad', 'ASGD', 'Adamax', 'FIRE'
  OPTIM_CONFIG:       # 优化器参数
    lr: !!float 2.e-4
    # ...
  LAYERWISE_OPTIM_CONFIG: # 支持正则表达式选择层并设置学习率
    'force_block.*': { 'lr': 5.e-4 }
    'energy_block.*': { 'lr': 2.e-4 }
    '.*_bias_layer.*': { 'lr': 2.e-4 }

  GRAD_CLIP: !!bool true  # 是否开启梯度裁剪
  GRAD_CLIP_MAX_NORM: !!float 10.  # 梯度裁剪的最大范数
  GRAD_CLIP_CONFIG: {}    # 传递给 `nn.utils.clip_grad_norm_` 的其他参数
  LR_SCHEDULER: !!str None  # 学习率调度器。可选值：'StepLR', 'ExponentialLR', 'ChainedScheduler', 'ConstantLR', 'LambdaLR', 'LinearLR', 'CosineAnnealingWarmRestarts', 'CyclicLR', 'MultiStepLR', 'CosineAnnealingLR', 'None'
  LR_SCHEDULER_CONFIG: {} # 上述调度器的参数
  EMA: !!bool false       # 是否启用指数移动平均（EMA）
  EMA_DECAY: !!float 0.999  # EMA 衰减率

# 结构弛豫（优化）
RELAXATION:
  ALGO: !!str 'FIRE'  # 算法选项：CG, BFGS, FIRE
  ITER_SCHEME: !!str 'PR+'  # 仅当 ALGO=CG 时有效，选项：'PR+', 'FR', 'PR'
  E_THRES: !!float 1.e4  # 能量差阈值
  F_THRES: !!float 0.05  # 最大力阈值
  MAXITER: !!int 300     # 最大迭代次数
  STEPLENGTH: !!float 0.5  # 初始步长
  USE_BB: !!bool true    # 是否使用 Barzilai‑Borwein I 步长作为初始步长
  LINESEARCH: !!str 'B'  # 'Backtrack'（Armijo 条件）、'Wolfe'（弱 Wolfe 条件，M‑T 算法）、'Exact'（精确线搜索）
  LINESEARCH_MAXITER: !!int 8  # 每次外迭代中线搜索的最大迭代次数
  LINESEARCH_THRES: !!float 0.02  # 仅当 LINESEARCH = 'exact' 时有效，精确线搜索阈值
  LINESEARCH_FACTOR: !!float 0.5  # "Backtrack" 时的收缩因子
  REQUIRE_GRAD: !!bool False  # 是否在计算过程中启用自动梯度

# 过渡态搜索
TRANSITION_STATE:
  ALGO: !!str DIMER  # 选项：DIMER
  X_DIFF_ATTR: !!str x_dimer  # 初始 dimer 方向属性名
  E_THRES: !!float 1.e-4    # 能量差阈值
  TORQ_THRES: !!float 1.e-2  # 旋转过程中扭矩收敛阈值，即特征向量残差
  F_THRES: !!float 5.e-2    # 最大力阈值
  MAXITER_TRANS: !!int 300  # 平移步最大迭代次数
  MAXITER_ROT: !!int 5      # 旋转步最大迭代次数
  MAX_STEPLENGTH: !!float 0.5  # 步长限制
  DX: !!float 1.e-1         # 有限差分计算 Hessian‑向量积的步长
  REQUIRE_GRAD: !!bool False  # 同上

# 振动分析（简谐近似）
VIBRATION:
  METHOD: !!str 'Coord'  # Coord / Grad，对应有限差分和自动梯度两种方案
  BLOCK_SIZE: !!int 1    # 张量/向量并行化的块大小
  DELTA: !!float 1e-2    # 有限差分计算 Hessian‑向量积的步长

# NEB 过渡态
NEB:
  ALGO: !!str 'CI-NEB'   # 选项：CI-NEB
  N_IMAGES: !!int 7      # CI‑NEB 的镜像数
  SPRING_CONST: 5.0      # NEB 弹簧常数
  OPTIMIZER: !!str FIRE  # CI‑NEB 优化器，目前仅支持 FIRE
  #OPTIMIZER_CONFIGS: 可选，优化器的其他参数
  STEPLENGTH: !!float 0.2     # 参见 `RELAXATION` 中的参数说明
  E_THRESHOLD: !!float 1.e-3  # 同上
  F_THRESHOLD: !!float 0.05   # 同上
  MAXITER: !!int 20           # 同上
  REQUIRE_GRAD: !!bool False  # 同上

# 分子动力学
MD:
  ENSEMBLE: !!str NVT     # MD 系综，可选：NVE, NVT
  THERMOSTAT: !!str CSVR  # 仅当 ENSEMBLE=NVT 时有效，可选：'Langevin', 'VR', 'Nose‑Hoover', 'CSVR'
  THERMOSTAT_CONFIG:      # 热浴参数
    DAMPING_COEFF: !!float 0.01  # Langevin 热浴的阻尼系数，单位：fs⁻¹
    TIME_CONST: !!float 120      # CSVR 热浴的时间常数，单位：fs
  TIME_STEP: !!float 1  # MD 时间步长，单位：fs
  MAX_STEP: !!int 100  # 总时间（fs）= TIME_STEP * MAX_STEP
  T_INIT: !!float 298.15  # 初始温度（K）。对于 NVE 系综，仅用于按玻尔兹曼分布生成随机初始速度
  OUTPUT_COORDS_PER_STEP: !!int 1  # 控制输出原子坐标的频率。若 verbose=3，还会输出原子速度
  MOVE_TO_CENTER_FREQ: !!int 20   # 每多少步将原子平移到质心并将整体速度置零
  # 可选：约束
  CONSTRAINTS_FILE: !!str ./constraints.py  # 约束函数的文件路径，函数应接收 torch.Tensor 并支持自动梯度
  CONSTRAINTS_FUNC: !!str func              # CONSTRAINTS_FILE 中具体的函数名称
  REQUIRE_GRAD: !!bool False  # 同上

# 蒙特卡洛
MC:
  TYPE: !!str Metropolis  # 目前仅支持 Metropolis
  ITER_SCHEME: !!str 'Gaussian'  # 扰动原子的分布：'Gaussian', 'Cauchy', 'Uniform'
  COORDINATE_UPDATE_PARAM: !!float 0.2  # 控制原子移动的参数：高斯分布的标准差，均匀分布的半区间长度，柯西分布的半峰全宽
  MAXITER: !!int 10000
  T_INIT: !!float 298.15    # 初始温度
  T_SCHEME: !!str constant  # 温度变化方式：'constant', 'linear', 'exponential', 'log', 'fast'。详见类 `BatchMC.MetropolisMC.MMC`
  T_UPDATE_FREQ: !!int 1    # 每隔多少步更新一次温度
  T_SCHEME_PARAM: !!float 0.  # 温度更新控制参数，详见 `BatchMC.MetropolisMC.MMC`
  OUTPUT_COORDS_PER_STEP: !!int 1  # 同 MD 中说明
  MOVE_TO_CENTER_FREQ: !!int 20    # 同 MD 中说明

# 模型配置
###MODEL_FILE: !!str your/model/file/path/template_model.py  # torch 模型文件路径
MODEL_NAME: !!str YourModel   # MODEL_FILE 中具体的模型类名
MODEL_CONFIG:   # 模型超参数，将传递给 `YourModel.__init__(**MODEL_CONFIG)`
  hyperparameter1: xxx
  hyperparameter2: xxx
  # ...
```

### 后处理

BUCToolkit 任务输出包括一个文本日志文件和一个二进制数据库文件。

#### 日志文件

对于 API 或可执行程序，通过设置 `REDIRECT: true` 以及 `OUTPUT_PATH` 和 `OUTPUT_POSTFIX` 来输出日志文件，内容详细程度由 `VERBOSE` 控制。若 `REDIRECT` 为 `false`，输出将打印到 `sys.stdout`。

底层函数通过日志系统控制，详见`BUCToolkit/Bases/BaseMotion.py`中的`BaseIO`类。

由于字符串处理开销较大，**不建议使用高详细程度的日志输出**。
所有信息均已包含在二进制数据库文件中，因此完全可以将详细程度设为 0（关闭日志）。

#### 二进制数据库

这是一种专门设计的二进制格式，用于高效转储数组。支持内存映射（memory‑mapping）读写。具体格式见类 `ArrayDumper`（位于 `BUCToolkit/BatchStructure/StructureIO.py`）。

要控制二进制文件输出，可在输入文件中设置 `SAVE_PREDICTIONS: true` 以及 `PREDICTIONS_SAVE_FILE`。对于底层函数，对应的参数为 `output_file`。

对于结构优化、分子动力学和蒙特卡洛模拟输出的二进制文件，可以在终端中按如下方式进行加载和转换：

```shell
buctoolkit -c `$input_type` `$input_path` `$output_type` `$output_path`
# `$input_type` 可以是 "bs", "md", "mc", "opt", "outcar", "poscar", "cif", "ase_traj" 之一
# `$output_type` 可以是 "poscar", "cif", "xyz", "bs" 之一
```

该命令会将 `$input_path` 下所有假定格式为 `$input_type` 的文件转换为 `$output_path` 中格式为 `$output_type` 的文件。

如需更精细的控制，可使用以下 Python 脚本：

```python
import BUCToolkit as bt
from BUCToolkit.io import read_opt_structures, read_md_traj, read_mc_traj

# 读取结构优化输出
bs1: bt.Structures = read_opt_structures('your_opt_output', indices=-1)

# 读取 MD 输出
bs2: bt.Structures = read_md_traj("your_md_output", indices=-1)

# 读取 MC 输出
bs3: bt.Structures = read_mc_traj("your_mc_output", indices=-1)

# bt.Structures 是管理批次结构的类。上述 read_* 方法将输出信息转换为此格式，之后可转换为其他文本文件：
bs1.write2text(output_path='./a/directory', indices=None, file_format='POSCAR')

# 或者转储到另一个二进制文件：
bs2.save('./a/directory')

# 或者直接用于其他计算：
from BUCToolkit.Preprocessing.preprocessing import CreatePygData, CreateASE, CreateDglData

atoms = CreateASE().feat2ase(bs3)
data = CreatePygData().feat2data_list(bs3)
graph = CreateDglData().feat2graph_list(bs3)
```

其中，参数 `indices` 用于指定读取或写入的部分，而非所有文件。

## 功能特性

BUCToolkit 实现了高度优化的 PyTorch 代码，
采用了包括算子融合、CUDA Graph 回放、基于 CUDA 流的异步转储/日志记录流水线，以及原地内存计算等手段。

### 灵活的函数接口

主要底层函数采用灵活的 SciPy 风格接口，如下所示（另请参见[使用底层函数](#使用底层函数)）：

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

其中，`X` 是要更新的目标变量（例如分子动力学或结构优化中的原子位置），
`func_args` 和 `func_kwargs` 是传递给 `func` 的其他必要参数。
因此，任何能够封装为 `func(X, *args, **kwargs)` 形式的函数都是有效的。
例如，用户可以编写一个函数，其内部提交第一性原理计算（如 VASP、Gaussian等），
并将计算结果（能量和力）转换为 torch.Tensor 格式返回，然后 BUCToolkit 就可以正常使用这些输入。

`grad_func` 的设计类似。参数 `is_grad_func_contain_y` 控制两种梯度计算方式：
- 当 `is_grad_func_contain_y = True` 时，采用自动梯度格式，内部实际调用 
`grad_func(X, y, *grad_func_args, **grad_func_kwargs)`（注意：用户无需手动将 `y` 放入 `grad_func_args`）。
- 当 `is_grad_func_contain_y = False` 时，使用 `grad_func(X, *grad_func_args, **grad_func_kwargs)` 接口。

最后，`require_grad` 控制 PyTorch 的梯度上下文。当 `require_grad = False` 时，
`func` 和 `grad_func` 的计算在 `torch.no_grad` 上下文下进行，以降低内存开销；
否则，将显式通过 `torch.enable_grad` 开启梯度计算上下文。

### 高度可定制的算法

所有方法/算法均采用面向对象的模块化设计。它们拥有 `_Base*` 抽象基类，实现了高度优化的主循环流程，
并在子类中通过覆写 `self.initialize*(...)`、`self._update*(...)` 等方法进行特化。
所有输入数据和状态都通过显式方式传递。因此，用户只需重写这些多态方法，就可以开发和实现任意自定义的新算法，而无需修改主循环过程。

### 批量并行方案

大多数功能（如结构优化、过渡态搜索、分子动力学、蒙特卡洛模拟）
均支持**规则批次样本**（原子数相同的样本堆叠）和**不规则批次样本**（原子数不同的样本拼接）的并行计算。

输入张量（原子坐标、力、固定掩码等）应为三维张量。
- 对于规则批次，形状为 **(batch_size, n_atom, n_dim)**，其中 `n_dim` 通常为 3。
- 对于不规则批次，形状为 **(1, $\sum_i$ n_atom$_i$, n_dim)**，其中 $i$ 是样本索引。
用户需要额外提供 `batch_indices` 变量，记录每个样本的原子数。
例如，`batch_indices = [64, 56, 72, 83, 102]` 表示各样本分别有 64、56、72、83、102 个原子，
此时原子坐标张量的形状应为 `(1, 377, 3)`。

对于结构优化和过渡态搜索，BUCToolkit 采用**动态样本**方法：
通过维护收敛掩码，在进入下一次迭代前动态移除批次中已收敛的样本，并利用 `indexed_select` / `indexed_copy_` 函数进行处理。
这能显著减少对已收敛样本的重复计算浪费，降低昂贵的函数评估次数。

### 基于自动微分的强大约束求解器
基于PyTorch的自动微分技术，BUCToolkit 实现了一个非常强大的约束求解器，可用于约束动力学和结构优化等任务。
其中，***任何PyTorch支持的可微分函数*** *S* 都能用于如下形式的约束：
$$
S(X) = q(t)
$$
其中***X***为（批量的）原子坐标，*q*是约束目标值，*t*是累计的模拟时间。对不含时约束，*q*是一个简单常数。

例如，约束函数可定义如下：
```python
import torch as th

def constr_func(X):
    y = list()
    # X: shape(N, D), 注意批处理维度（第一维）在这里定义约束函数时不被考虑。计算时将自动作用。
    ################
    # 约束函数 1     #
    ################
    # 固定原子对 (2, 4), (3, 7), (5, 8) 的键长为 `constr_val[:3]` 的值
    y.append(th.linalg.norm(X[[2, 3, 5]] - X[[4, 7, 8]], dim=-1))

    ################
    # 约束函数 2     #
    ################
    # 固定键角 atom7-atom5-atom8 和 atom11-atom9-atom12 为 `constr_val[3:5]` 的值
    x1 = X[[5, 9]]
    x2 = X[[7, 11]]
    x3 = X[[8, 12]]
    y.append(
        (
            th.sum((x2 - x1) * (x3 - x1))
        ) / (th.linalg.norm(x2 - x1) * th.linalg.norm(x3 - x1) + 1.e-20)
    )
    
    ################
    # 约束函数 3     #
    ################
    # 固定原子14和18的广义配位数为 `constr_val[5:7]` 的值
    r0 = 1.5; sigma = 0.2
    # pairwise distance: (2, 1, D) - (1, N, D) -norm-> (2, N)
    r_ij = th.linalg.norm(X[[14, 18]].unsqueeze(1) - X.unsqueeze(0), dim=-1)
    s_i = th.sum(0.5 * (1 + th.erf( (r_ij - r0) / sigma )), dim=-1)
    y.append(s_i)
    
    ################
    # 约束函数 4     #
    ################
    # 固定所有键长的标准差为 `constr_val[7:8]` 的值
    R_ij = th.linalg.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1)
    R_std = th.std(R_ij, unbiased=True)
    y.append(R_std)
    
    ###############################
    # 拼接所有约束                  #
    ###############################
    z = th.cat(y)
    
    return z

# 执行约束动力学
import BUCToolkit as bt

DEVICE = 'cuda:0'  # or cpu

runner = bt.BatchMD.ConstrNVT(
  time_step=1.,
  max_step=10000,
  thermostat='CSVR',
  thermostat_config={'time_const': 100.},
  constr_func=constr_func,
  constr_val=th.full((7, ), 1.5, device=DEVICE),  # 用作示例, 所有约束目标值都设为 1.5
  constr_threshold=1.e-5,
  T_init=298.15,
  output_file='/your/dumped/file',
  output_structures_per_step=20,
  device=DEVICE,
  verbose=0
)
# `constr_val` 也可以设为 Callable[[ScalarTensor], Tensor]，以表示时变约束。

runner.run(...)   # 见 `作为 Python 包使用` 一节

```
所有的约束函数都通过约束的质量加权雅可比矩阵(形状：(n_constr, n_atoms * n_dim))QR分解的方法同时并行求解，
而不是逐约束串行循环迭代。
* 对（余）切丛上的向量（如速度、梯度），通过 **切空间投影** 来保持约束:
$$
Q R = J_{\rm constraints}^T M^{-1/2}; Q = [Q_1, Q_2]
$$
$$
P = M^{-1/2} Q_2 Q_2^T M^{1/2} = M^{-1/2} (I - Q_1 Q_1^T) M^{1/2}
$$
$$
v = P \tilde{v}
$$
* 对约束流形上的点（如原子坐标）通过牛顿迭代进行**拉回映射**:
$$
y^{(i)} = S(X^{(i-1)})
$$
$$
X^{(i)} = X^{(i-1)} - M^{-1/2} Q R^{-T} y^{(i)}
$$
直到约束误差收敛。这一牛顿迭代格式具有二次收敛速度。

注意：复杂约束函数的自动微分求解可能计算代价较高。

## 联系我们

如有任何问题，请联系 **buctoolkit@163.com**。

如需报告 Bug 或建议新功能，请使用 [GitHub Issues](https://github.com/TrinitroCat/BUCToolkit/issues)。

## 许可证

BUCToolkit 的代码在 **[MIT 许可证](https://github.com/TrinitroCat/BUCToolkit/blob/main/LICENSE)** 下发布和分发。
