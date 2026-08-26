# BUCToolkit 工作流示例

本页同时覆盖高级 API 和 low-level 引擎。参数定义见[manual-zh.md](manual-zh.md)。
代码中的模型类和数据路径是占位符，导入路径按当前公开接口书写。

## 一般API配置方法

高级 API 从 YAML 读取任务参数，然后显式设置模型、数据和 DataLoader：

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
runner.run(MyModel)                 # 传入类，不要传入 MyModel(...)
```

`Trainer`、`Predictor`、`StructureOptimization`、`VibrationAnalysis`、
`ClimbingImageNudgedElasticBand` 和 `MonteCarlo` 使用相同的设置模式。

## Batch布局

规则 batch 的坐标形状为 `(batch_size, n_atom, 3)`。不规则 batch 将坐标拼为
`(1, sum(n_atom), 3)`，并提供 `batch_indices=(n_1, ..., n_N)`。原子元素、
速度和 fixed mask 使用同样的结构划分。

```python
import torch

X = torch.zeros((1, 377, 3))
batch_indices = [64, 56, 72, 83, 102]
```

## 配置模型与读取已有参数

`MODEL_CONFIG` 会作为模型构造函数参数传入。`START: 0` 从头开始；
`START: 1` 或 `resume` 读取 checkpoint；`START: 2` 只读取模型参数。
`STRICT_LOAD` 控制 `load_state_dict` 行为。

```yaml
TASK: MD
START: 2
DEVICE: cuda:0
LOAD_CHK_FILE_PATH: ./checkpoint.pt
STRICT_LOAD: true
MODEL_CONFIG:
  hidden_dim: 128
```

## 通过YAML输入文件运行任务

CLI 会以 YAML 输入文件所在目录为基准解析其中的相对路径：

```shell
buctoolkit -i ./md.yml
```

设置 `OUTPUT_ROOT` 可统一管理日志、结果和 checkpoint。不带参数运行时进入
交互式 CLI：

```shell
buctoolkit
```

## 加载与转换结构文件

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

`POSCARs2Feat`、`OUTCAR2Feat`、`Cif2Feat` 和 `ASETraj2Feat` 使用相同的读取
模式，但有各自格式标签。CLI 输出可以转换为其他结构格式：

```shell
buctoolkit -c md ./md_output poscar ./poscars
```

## APIs用法

### 通过APIs训练模型

训练数据需要标签，普通计算数据通常不需要：

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

`set_loss_fn`、`set_optimizer`、`set_lr_scheduler`、`set_metrics` 和
`set_layerwise_optim_config` 可用于覆盖 YAML 中的训练设置。

### 通过API预测/推理

```python
from BUCToolkit.api.Predictor import Predictor
from BUCToolkit.api.DataLoaders import PyGDataLoader

predictor = Predictor('./predict.yml')
predictor.set_dataset({'data': data_list})
predictor.set_dataloader(PyGDataLoader, {'shuffle': False})
result = predictor.run(MyModel, test_model=False, warm_up=False)
```

设置 `SAVE_PREDICTIONS: true` 时，结果写入配置的二进制路径，而不是保留在
内存返回值中。

### 通过API结构优化

```python
from BUCToolkit.api.StructureOptimization import StructureOptimization
from BUCToolkit.api.DataLoaders import PyGDataLoader

optimizer = StructureOptimization('./opt.yml')
optimizer.set_dataset({'data': data_list})
optimizer.set_dataloader(PyGDataLoader, {'shuffle': False})
optimizer.run(MyModel, mode='minimize')
```

使用 `mode='ts'` 和 `set_dimer_init_direction(...)` 可进行 dimer 搜索。

### 通过API运行分子动力学模拟

```python
from BUCToolkit.api.MolecularDynamics import MolecularDynamics
from BUCToolkit.api.DataLoaders import PyGDataLoader

runner = MolecularDynamics('./md.yml')
runner.set_dataset({'data': data_list})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)
```

`MD.ENSEMBLE` 选择 `NVE` 或 `NVT`；`MD.THERMOSTAT_CONFIG` 根据热浴使用
`damping_coeff`、`time_const` 或 `virt_mass`。

### 约束函数配置/约束动力学

```python
import torch

def constr_func(X: torch.Tensor) -> torch.Tensor:
    # 单次约束计算中 X 的形状为 (n_atom, 3)。
    return torch.linalg.vector_norm(X[0] - X[1]).reshape(1)

def constr_val(t: torch.Tensor) -> torch.Tensor:
    # slow growth 目标形状：(N_IMAGES, n_constraint)。
    return (1.5 + 0.001 * t).reshape(1, 1)
```

两个函数都必须对坐标可微。CLI 中省略 `MD.CONSTRAINTS_VAL_FUNC` 或设为
`null`，即可保持初始约束值行为。

#### 慢增长约束动力学

Slow-growth 从一组初始结构出发，复制出 `N_IMAGES` 个并行副本。CLI 只需要
`DATA_PATH`：

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

对应的 API 数据和 loader 为：

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

#### Blue-Moon 约束动力学

Blue-Moon 将初态和终态配对，插值生成 `N_IMAGES` 个镜像，并使用
`ISFSPyGDataLoader`：

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

初态和终态必须具有匹配的原子数及原子序数排列。

### 通过API运行振动频率(简谐)计算

```python
from BUCToolkit.api.VibrationAnalysis import VibrationAnalysis

runner = VibrationAnalysis('./vibration.yml')
runner.set_dataset({'data': data_list})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)
```

`VIBRATION.METHOD`、`DELTA`、`BLOCK_SIZE` 和 `SAVE_HESSIAN` 控制有限差分、
自动梯度路径以及输出内容。

### 通过API运行过渡态搜索

```python
from BUCToolkit.api.NEB import ClimbingImageNudgedElasticBand
from BUCToolkit.api.DataLoaders import ISFSPyGDataLoader

runner = ClimbingImageNudgedElasticBand('./neb.yml')
runner.set_dataset({'dataIS': initial_data, 'dataFS': final_data})
runner.set_dataloader(ISFSPyGDataLoader)
runner.run(MyModel)
```

对于 dimer 搜索，使用 `StructureOptimization` 的 `mode='ts'`，并通过
`set_dimer_init_direction` 或配置数据提供方向。

### 通过API运行 Metropolis 蒙特卡罗方法

```python
from BUCToolkit.api.MonteCarlo import MonteCarlo

runner = MonteCarlo('./mc.yml')
runner.set_dataset({'data': data_list})
runner.set_dataloader(PyGDataLoader, {'shuffle': False})
runner.run(MyModel)
```

`MC.ITER_SCHEME` 控制坐标扰动分布，`T_SCHEME` 控制恒温或变温方式。

## 底层函数调用

### Low-level MD

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

微正则系综使用 `BatchMD.NVE`。`ConstrNVE` 和 `ConstrNVT` 额外接收
`constr_func`、`constr_val`、`constr_threshold`，并可开启 `require_fixman`。

### Low-level Optimization

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

`CG` 还可选择 `iter_scheme` 和线搜索参数；`QN` 使用相同的基础收敛和
输出参数。设置 `output_grad=True` 可返回末步梯度。

### Low-level Transition-state Search

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

`CI_NEB` 接收初态/终态坐标，并使用 `N_images`、`spring_const`、
`optimizer_configs`、收敛阈值和 `maxiter`。`KrylovNewton` 与
`KrylovDynamics` 还需要配置 eigen solver；完整的专用签名见源码 docstring。

### Low-level 频率分析

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

有梯度函数时可使用 `method='GradDiff'`；能量可微时可使用
`method='Autograd'`。

### Low-level 蒙特卡罗

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

Low-level MC 同样遵循能量函数转发和 batch 形状约定。

## 读取与转换结果

```python
from BUCToolkit.io import read_md_traj, read_mc_traj, read_opt_structures

md_structures = read_md_traj('./traj.bin', indices=-1)
mc_structures = read_mc_traj('./mc.bin', indices=-1)
opt_structures = read_opt_structures('./opt.bin', indices=-1)
md_structures.write2text('./frames', file_format='POSCAR')
```

## 后处理

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

这些计算器可以使用轨迹中已保存的列，也可以使用用户提供的
collective-variable 数值。帧范围使用 `start`、不包含 `stop` 的终点和正数
`stride`；完整参数和返回值见源码 docstring。

## Links

- [中文参数手册](manual-zh.md)
- [English manual](manual.md)
- [English examples](examples.md)
- [README-zh.md](../README-zh.md)
