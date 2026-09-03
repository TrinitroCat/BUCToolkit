# BUCToolkit 参数手册

本页是 README 的参数参考补充，集中说明输入文件字段、公开的高级 API
以及主要 low-level 模拟和优化类。更细的实现行为仍以源码和 docstring
为准。

完整工作流程见[中文示例](examples-zh.md)。每组参数会链接到使用该参数的
具体示例。

## 约定

BUCToolkit 不自动进行单位换算。通常使用 Angstrom（长度）、eV（能量）、
amu（质量）、fs（时间）和 K（温度）。模型的 `func(X, ...)` 返回能量，
`grad_func(X, ...)` 返回数学意义上的能量梯度；除非模型 adapter 已做转换，
物理力等于负梯度。

正规 batch 的坐标形状为 `(batch_size, n_atom, 3)`。不规则 batch 将坐标拼为
`(1, sum(n_atom), 3)`，并用 `batch_indices=(n_1, ..., n_N)` 描述每个结构的
原子数。

## YAML 输入参数

CLI 使用 `buctoolkit -i input.yml` 读取 YAML。标注了任务的字段只在对应任务
中生效。YAML 内部的相对路径相对于输入文件所在目录解析。

### 全局参数

| 参数 | 取值/类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `TASK` | `TRAIN`、`PREDICT`、`OPT`、`TS`、`VIB`、`NEB`、`MD`、`CMD`、`MC` | CLI 必需 | 选择高级任务。[CLI 任务示例](examples-zh.md#run-a-task-from-yaml) |
| `START` | `0`、`1`、`2`、`from_scratch`、`resume` | `0` | 从头开始、恢复 checkpoint，或只读取模型权重。[checkpoint 示例](examples-zh.md#configure-a-model-and-checkpoint) |
| `VERBOSE` | 非负整数 | `1` | 文本输出详细程度；`0` 静默，更高值会输出更多数组。[输出示例](examples-zh.md#read-and-convert-results) |
| `DEVICE` | Torch device 字符串 | `cpu` | 模型和计算所用设备。[API 设置](examples-zh.md#common-api-setup) |
| `BATCH_SIZE` | 正整数 | `1` | DataLoader 每批结构数；当前 CMD 要求为 `1`。[batch 形状](examples-zh.md#batch-layout) |

### I/O 与数据参数

| 参数 | 取值/类型 | 说明 |
| --- | --- | --- |
| `LOAD_CHK_FILE_PATH` | 路径 | `START` 为 `1` 或 `2` 时读取的 checkpoint。 |
| `OUTPUT_ROOT` | 路径 | 日志、结果和 checkpoint 的根目录。缺失时创建；已有目录必须为空。 |
| `OUTPUT_PATH` | 路径 | 旧版或显式日志目录；设置 `OUTPUT_ROOT` 时默认是 `OUTPUT_ROOT/logs`。 |
| `OUTPUT_POSTFIX` | 字符串 | 日志文件名后缀。 |
| `PREDICTIONS_SAVE_FILE` | 路径 | 开启保存时的二进制输出路径。 |
| `STRICT_LOAD` | 布尔值 | 传给 `load_state_dict`。 |
| `REDIRECT` | 布尔值 | 为真时写入 `OUTPUT_PATH`，否则打印到屏幕。 |
| `SAVE_PREDICTIONS` | 布尔值 | 是否将预测或轨迹写入 `PREDICTIONS_SAVE_FILE`。 |
| `DATA_TYPE` | `POSCAR`、`OUTCAR`、`CIF`、`ASE_TRAJ`、`BS`、`OPT`、`MD`、`MC` | CLI 输入格式；`BS` 是 BUCToolkit 内置结构格式。[数据读取](examples-zh.md#load-and-convert-structures) |
| `DATA_PATH` | 路径 | 初态或训练数据路径。 |
| `DATA_NAME_SELECTOR` | 正则表达式 | 按样本名筛选结构。 |
| `FSDATA_PATH` | 路径 | NEB、Blue-Moon CMD 等成对任务的终态数据。[Blue-Moon CMD](examples-zh.md#blue-moon-cmd) |
| `DISPDATA_PATH` | 路径 | Dimer 过渡态搜索的初始位移/方向数据。 |
| `VAL_SET_PATH` | 路径 | 训练时显式指定验证集。 |
| `VAL_SPLIT_RATIO` | `[0, 1)` 的浮点数 | 未指定 `VAL_SET_PATH` 时的验证集比例。 |
| `DATA_LOADER_KWARGS` | 映射 | 传给 DataLoader 的额外参数。 |
| `IS_SHUFFLE` | 布尔值 | 是否打乱非成对计算数据；成对任务和 CMD 使用确定顺序。 |

### `TRAIN`

| 参数 | 取值/类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `EPOCH` | 正整数 | `0` | 训练 epoch 数。[训练示例](examples-zh.md#train-a-model-with-the-api) |
| `SAVE_CHK` | 布尔值 | `false` | 是否保存 checkpoint。 |
| `CHK_SAVE_PATH` | 路径 | `OUTPUT_ROOT/chk` | checkpoint 目录。 |
| `CHK_SAVE_POSTFIX` | 字符串 | 空 | checkpoint 文件名后缀。 |
| `VAL_BATCH_SIZE` | 正整数 | `BATCH_SIZE` | 验证 batch 大小。 |
| `VAL_PER_STEP` | 正整数 | `10` | 验证频率。 |
| `VAL_IF_TRN_LOSS_BELOW` | 浮点数 | `inf` | 训练损失低于此值后才验证。 |
| `ACCUMULATE_STEP` | 正整数 | `1` | 梯度累积步数。 |
| `LOSS` | `MSE`、`MAE`、`Hubber`、`CrossEntropy`、`Energy_Force_Loss`、`Energy_Loss` | `Energy_Force_Loss` | 内置损失选择。 |
| `LOSS_CONFIG` | 映射 | `{}` | `loss_E`、`loss_F`、`coeff_E`、`coeff_F` 等损失选项。 |
| `METRICS` | 指标名称 | 实现默认值 | `E_MAE`、`F_MAE`、`F_MaxE`、`E_R2`、`MSE`、`MAE`、`R2`、`RMSE`。 |
| `METRICS_CONFIG` | 映射 | `{}` | 指标专用参数。 |
| `OPTIM` | Torch 优化器名称或 `FIRE` | `AdamW` | 模型参数优化器。 |
| `OPTIM_CONFIG` | 映射 | `{}` | 优化器参数，如 `lr`。 |
| `LAYERWISE_OPTIM_CONFIG` | 正则表达式到映射 | 无 | 按层设置优化器参数。 |
| `GRAD_CLIP` | 布尔值 | `false` | 是否裁剪梯度。 |
| `GRAD_CLIP_MAX_NORM` | 浮点数 | `100` | 梯度最大范数。 |
| `GRAD_CLIP_CONFIG` | 映射 | `{}` | 梯度裁剪的额外参数。 |
| `LR_SCHEDULER` | scheduler 名称或 `None` | `None` | 学习率调度器。 |
| `LR_SCHEDULER_CONFIG` | 映射 | `{}` | 调度器参数。 |
| `EMA` | 布尔值 | `false` | 是否使用指数移动平均参数。 |
| `EMA_DECAY` | 浮点数 | `0.999` | EMA 衰减系数。 |

### `RELAXATION`

`ALGO` 可为 `CG`、`BFGS` 或 `FIRE`；`ITER_SCHEME` 选择 CG 迭代方案；
`E_THRES`、`F_THRES` 和 `MAXITER` 控制收敛；`STEPLENGTH`、`USE_BB`、
`LINESEARCH`、`LINESEARCH_MAXITER`、`LINESEARCH_THRES` 和
`LINESEARCH_FACTOR` 控制步长及线搜索；`REQUIRE_GRAD` 控制自动梯度。
参见[结构优化示例](examples-zh.md#structure-optimization-with-the-api)。

### `TRANSITION_STATE`

当前 `ALGO` 为 `DIMER`，`X_DIFF_ATTR` 指定初始 dimer 方向属性。
`E_THRES`、`TORQ_THRES`、`F_THRES`、`MAXITER_TRANS`、`MAXITER_ROT`、
`MAX_STEPLENGTH` 和 `DX` 分别控制能量、扭矩、力、平移/旋转迭代、步长和
有限差分参数。参见[过渡态搜索](examples-zh.md#transition-state-search)。

### `VIBRATION`

| 参数 | 取值/类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `METHOD` | `Coord`/`EnergyDiff`、`Grad`/`Autograd` 或 `GradDiff` | `EnergyDiff` | 有限差分或自动梯度路径。 |
| `BLOCK_SIZE` | 正整数 | `1` | 一次并行计算的位移镜像数。 |
| `DELTA` | 正浮点数 | `1e-2` | 有限差分位移。 |
| `SAVE_HESSIAN` | 布尔值 | `false` | 是否保存 Hessian、频率和振动模式。[振动分析](examples-zh.md#vibrational-analysis) |

### `NEB`

`ALGO` 当前为 `CI-NEB`；`N_IMAGES` 是路径镜像数；`SPRING_CONST` 是弹簧
常数；`OPTIMIZER` 当前为 `FIRE`，其额外参数放在 `OPTIMIZER_CONFIGS`；
`STEPLENGTH`、`E_THRESHOLD`、`F_THRESHOLD` 和 `MAXITER` 控制优化步长、
能量/力阈值及最大迭代数；`REQUIRE_GRAD` 控制自动梯度。

### `MD` 与 CMD

| 参数                     | 取值/类型                               | 默认值             | 说明                                                                                                          |
|--------------------------|-----------------------------------------|--------------------|---------------------------------------------------------------------------------------------------------------|
| `ENSEMBLE`               | `NVE`、`NVT`                            | `NVT`              | MD 系综。[MD](examples-zh.md#molecular-dynamics-with-the-api)                                                 |
| `THERMOSTAT`             | `Langevin`、`VR`、`Nose-Hoover`、`CSVR` | `CSVR`             | NVT 热浴。                                                                                                    |
| `THERMOSTAT_CONFIG`      | 映射                                    | `{}`               | 根据热浴使用 `damping_coeff`、`time_const` 或 `virt_mass`；输入解析兼容大小写。                               |
| `TIME_STEP`              | 正浮点数，fs                            | `1`                | 积分时间步长。                                                                                                |
| `MAX_STEP`               | 正整数                                  | 实现默认值         | 积分步数。                                                                                                    |
| `T_INIT`                 | 正浮点数，K                             | `298.15`           | 初始/目标温度。                                                                                               |
| `OUTPUT_COORDS_PER_STEP` | 正整数                                  | `1`                | 坐标输出间隔。                                                                                                |
| `MOVE_TO_CENTER_FREQ`    | 整数                                    | low-level 为 `-1`  | 质心平移和整体速度清零的周期；非正数关闭。                                                                    |
| `CONSTRAINTS_FILE`       | 路径                                    | 无                 | 约束函数所在 Python 文件。                                                                                    |
| `CONSTRAINTS_FUNC`       | 函数名                                  | 无                 | 文件中的约束函数。                                                                                            |
| `CONSTRAINTS_VAL_FUNC`   | 函数名或 `null`                         | `null`             | 同文件中的可选时间目标函数；`null` 保持初始值行为。[约束函数](examples-zh.md#constraint-functions)            |
| `REQUIRE_GRAD`           | 布尔值                                  | `false`            | 是否开启自动梯度。                                                                                            |
| `CONSTR_MD_SCHEME`       | `SLOW_GROWTH`、`BLUE_MOON`              | 模板值             | CMD 模式。                                                                                                    |
| `N_IMAGES`               | 正整数                                  | `1` 或模板值       | Blue-Moon 插值镜像数；Slow-growth 并行副本数。单条轨迹设为 `1`。[Slow-growth](examples-zh.md#slow-growth-cmd) |
| `CONSTR_THRESHOLD`       | 正浮点数                                | `1.e-5`            | 约束收敛限, 单位与具体约束的广义坐标相同.                                                                     |
| `REQUIRE_FIXMAN`         | 布尔值 or "auto"                        | `"auto"`           | `auto` 意为 `BLUE_MOON` 设为 True, `SLOW_GROWTH` 设为False. 可手动设置 True/False.                            |


CLI 的 `BLUE_MOON` 从 `DATA_PATH`/`FSDATA_PATH` 读取成对结构并使用
`ISFSPyGDataLoader`；`SLOW_GROWTH` 只读取 `DATA_PATH` 并使用
`PyGDataLoader`。当前两种 CMD 都要求 `BATCH_SIZE: 1`。

### `MC`

`TYPE` 当前为 `Metropolis`；`ITER_SCHEME` 可为 `Gaussian`、`Cauchy` 或
`Uniform`；`COORDINATE_UPDATE_PARAM` 是位移尺度；`MAXITER` 是步数；
`T_INIT`、`T_SCHEME`、`T_UPDATE_FREQ` 和 `T_SCHEME_PARAM` 控制温度；
`OUTPUT_COORDS_PER_STEP` 和 `MOVE_TO_CENTER_FREQ` 控制输出及质心处理。
参见[Metropolis 示例](examples-zh.md#metropolis-monte-carlo)。

### 模型参数

`MODEL_TYPE` 选择模型协议（`pyg`、`vasp` 或 `custom`）。默认 PyG 协议使用
`MODEL_FILE` 和 `MODEL_NAME` 指定模型，并将 `MODEL_CONFIG` 传给模型构造函数。
`MODEL_WRAPPER_CONFIG` 配置包装器；VASP 直接使用它且不读取 `MODEL_CONFIG`。
自定义包装器通过 `MODEL_WRAPPER_FILE` 和 `MODEL_WRAPPER_NAME` 加载。
`START`、`LOAD_CHK_FILE_PATH` 和 `STRICT_LOAD` 控制参数读取。参见[模型与 checkpoint](examples-zh.md#configure-a-model-and-checkpoint)。

## 高级 API 参数

高级任务类接收 `config_file`，以及可选的 `data_type`（`pyg` 或 `dgl`）。
常见调用顺序如下：

```python
runner = Task(config_file)
runner.set_model_config(model_config)
runner.set_dataset(data, valid_data)
runner.set_dataloader(DataLoader, loader_kwargs)
runner.run(ModelClass)
```

| 方法 | 参数 | 说明 |
| --- | --- | --- |
| `set_device` | `device` | 覆盖配置中的 Torch 设备。 |
| `set_model_config` | `model_config` 映射或 `None` | 模型构造函数参数。 |
| `set_model_param` | `model_state_dict`、`is_strict`、`is_assign` | 读取内存中的状态字典。 |
| `set_dataset` | `train_data`、可选 `valid_data` | 训练字典、计算字典或成对的 `dataIS`/`dataFS`。 |
| `set_dataloader` | loader 类、可选映射 | loader 参数；batch 大小和设备由配置传入。 |

主要任务方法为 `Trainer.train`、`Predictor.run/predict`、
`MolecularDynamics.run`、`ConstrainedMolecularDynamics.set_constr_func`、
`set_constr_val`、`StructureOptimization.run/relax`、
`VibrationAnalysis.run`、`ClimbingImageNudgedElasticBand.run` 和
`MonteCarlo.run`。完整签名请查阅源码 docstring。

## Low-level 参数

Low-level 引擎直接处理 Tensor，不读取 YAML、模型或结构文件。

### 共享函数协议

`func(X, *func_args, **func_kwargs)` 返回能量；可选的
`grad_func(X, *grad_func_args, **grad_func_kwargs)` 返回数学上的能量梯度。
当 `is_grad_func_contain_y=True` 时，调用形式为
`grad_func(X, y, ...)`，其中 `y=func(X, ...)`。`require_grad` 控制
PyTorch 自动梯度上下文。

### MD：`NVE`、`NVT`、`ConstrNVE`、`ConstrNVT`

共同构造参数为 `time_step`、`max_step`、`T_init`、`output_file`、
`output_structures_per_step`、`device`、`verbose`、`is_compile`、
`compile_kwargs`、`dump_quantities` 和 `log_quantities`。`NVT` 另外需要
`thermostat`、`thermostat_config`；约束 MD 另外需要 `constr_func`、
`constr_val`、`constr_threshold` 和可选的 `require_fixman`。

共同 `run` 参数为 `func`、`X`、`Element_list`、可选 `Cell_vector`、
`V_init`、`grad_func`、函数参数/关键字参数、`is_grad_func_contain_y`、
`require_grad`、`batch_indices`、`fixed_atom_tensor` 和
`move_to_center_freq`。参见[low-level MD](examples-zh.md#low-level-md)。

### 优化：`CG`、`FIRE`、`QN`

共同构造参数为 `iter_scheme`、`E_threshold`、`F_threshold`、`maxiter`、
`linesearch`、`linesearch_maxiter`、`linesearch_thres`、
`linesearch_factor`、`steplength`、`use_bb`、`output_file`、`device`、
`verbose`、`dump_quantities` 和 `log_quantities`。`FIRE` 改用 `alpha`、
`alpha_fac`、`fac_inc`、`fac_dec`、`N_min`。共同 `run` 还接受
`output_grad`、`fixed_atom_tensor` 和 `batch_indices`。参见
[low-level 优化](examples-zh.md#low-level-optimization)。

过渡态 low-level 类使用相同的函数协议。`Dimer` 另外接受 `Torque_thres`、
`Curvature_thres`、`maxiter_trans`、`maxiter_rot`、`max_steplength` 和 `dx`。
`CI_NEB` 接受 `N_images`、`spring_const`、`optimizer`、`optimizer_configs`、
`steplength`、收敛阈值、`maxiter`、`device` 和 `verbose`。`KrylovNewton` 与
`KrylovDynamics` 还接受 `Torque_thres`、`Eigen_thres`、`maxiter_trans`、
`maxiter_eig`、`dx`、`morse_index` 及谱截断参数。参见[low-level 过渡态搜索](examples-zh.md#low-level-transition-state-search)。

low-level `Frequency` 计算器接受 `method`、可选 `block_size`、有限差分
`delta`、可选 `output_file` 和 `dump_hessian`；`normal_mode` 调用还需要能量
函数、坐标、可选质量、函数/梯度参数和可选 fixed-atom mask。参见
[low-level 频率分析](examples-zh.md#low-level-frequency-analysis)。

### Monte Carlo：`MMC`

`MMC` 接受 `iter_scheme`、`maxiter`、`temperature_init`、
`temperature_scheme`、`temperature_update_freq`、
`temperature_scheme_param`、`coordinate_update_param`、输出/设备/详细程度
参数以及 dump/log 字段选择。`run` 使用共享能量函数协议，并接受 `X`、
`Element_list`、可选 `Cell_vector`、`V_init`、`batch_indices` 和
`fixed_atom_tensor`。参见[low-level MC](examples-zh.md#low-level-mc)。

### 数据和后处理辅助函数

主要预处理读取器为 `POSCARs2Feat`、`OUTCAR2Feat`、`ExtXyz2Feat`、
`Cif2Feat` 和 `ASETraj2Feat`；共同参数是 `path`、`verbose`、文件选择，
以及格式专用标签。`CreatePygData`、`CreateDglData` 和 `CreateASE` 用于
转换内存中的 feature 数据。

`PyGDataLoader` 和 `DglGraphLoader` 接受 `data` 映射、`batch_size`、
`device`、`shuffle`、`is_train` 和可选 `data_names`。
`ISFSPyGDataLoader` 接受成对的 `dataIS`/`dataFS`、`batch_size`、`device` 和
可选 `data_names`；MACE loader 使用相同的数据划分并适配模型专用对象。

`Postprocessing.MDTrajectory` 接受二进制轨迹路径、`indices` 和 `is_copy`，
并提供列名、原子序数、质量、帧/镜像选择和堆叠操作。
`BlueMoonCalculator` 接受轨迹，并可选择镜像顺序、collective-variable
数值或列、帧范围、温度和 Fixman 权重。
`ConstraintWorkCalculator` 可选择 CV 数值/列、温度、帧范围、stride、积分
规则、力符号和共同协议容差。`VibrationalSpectrumCalculator` 接受轨迹和
`sample_spacing_fs`，计算时可指定速度列、帧范围、窗口和归一化选项。参见
[后处理示例](examples-zh.md#postprocessing)。

## 进一步参考

- [中文工作流示例](examples-zh.md)
- [English manual](manual.md)
- [English examples](examples.md)
- [README-zh.md](../README-zh.md)
