# Sparse4D J6M 四路鱼眼单帧检测部署任务

## 0. 执行角色与编码原则

你是负责本任务的 C++ 与地平线模型部署代码专家。以用户提供的、已经实现的 SparseDrive 工程为直接代码依据，完成必要的 Sparse4D 适配，不自行设计另一套部署框架。

- 所有代码优先复用 SparseDrive 中适用的现有实现。能够直接使用的函数、调用顺序、资源管理、参数定义、目录组织和构建写法保持一致，不为个人风格重写。
- 只有 J6M 平台、四路鱼眼、模型 IO、无时序纯检测、数据格式或当前 SDK 接口确实不同的地方才作修改。对必要修改记录参考文件及修改原因。
- 不采用过度防御性编程。不新增假想需求的兼容层、静默回退、自动猜测参数、多套备用实现或重复校验，不用增加超时和重试掩盖推理问题。
- 保留参考工程已有且本任务必需的文件读取、SDK 返回值、内存边界检查与资源释放。错误直接报告实际原因，不吞掉错误或用默认数据伪造成功结果。
- 本文建议的模块划分服从上述复用原则。若参考工程的现有结构足够清晰，沿用原结构，不为了拆分文件额外引入抽象。

## 1. 文档状态与目标

用户已确认本任务，进入实施阶段。实际完成状态见 SPARSE4D_IMPLEMENTATION.md，未运行的验收项保持未完成。

参考 `sparsedrive_deploy_right` 中的 SparseDrive 部署工程，提取其离线数据读取、模型运行、Tensor 处理、结果保存和构建组织方式，为用户的 Sparse4D 鱼眼模型实现独立部署。最终完成四路图片和标定读取、C++ 前处理、HBM 推理、C++ 检测后处理及可视化。

先在 x86 Linux 服务器验证单帧完整流程，再交叉编译至 aarch64 J6M，并测量板端真实时延。x86 的仿真耗时不作为 J6M 时延。服务器验证成功也不能代替板端运行验证。

与 SparseDrive 参考任务的区别：

- 模型只有 3D 检测，不包含 map、motion、planning、轨迹预测或跟踪。
- 四路鱼眼相机，batch size 为 1，每帧独立推理。
- 不实现前一帧缓存、instance ID、feature queue、anchor queue、车辆历史状态和跨帧坐标变换。
- 保留当前 HBM 实际要求的全部输入。无时序不表示可以删去导出接口里的 `time_interval`。
- 首版包含 C++ 检测解码与可视化。参考任务仅在 Python 端后处理的限制不沿用。Python 作为对齐、评测和可视化基准。
- 不引入实时摄像头接入、ROS、Web 服务、多线程吞吐优化或训练修改。

### 1.1 首要交付目标与已有环境基础

用户已在同一 x86 Linux 服务器的地平线官方环境中成功运行其他 J6M 模型，包括 YOLOv8。以此作为已具备 x86 模型运行基础的事实，不从“服务器没有 BPU”推断 x86 无法执行 HBM，不默认要求重装环境或改为只在板端验证。

本任务首要目标是：在该 x86 Linux 服务器上，使用当前 Sparse4D HBM 和同一帧四路真实鱼眼图片，实际完成 C++ 数据读取、前处理、HBM 推理、检测后处理和可视化。此项是必需验收条件，不能以编译通过、模型加载成功、Python 校准模型推理成功或板端运行成功代替。

x86 和 aarch64 使用同一份业务 C++ 源码，包括 main.cc、sparse4d_runtime.cc、前处理、后处理和可视化代码。通过编译器、目标架构、对应 SDK 库及必要的平台构建配置生成不同架构的可执行文件，不维护两套模型逻辑。先完成 x86 单帧完整流程，再开展 aarch64 J6M 验证。

共享源码明确了可移植实现目标，但不等于两种运行后端必然具有相同行为。YOLOv8 成功说明该环境已有实际运行基础，不能单独证明当前 Sparse4D HBM 涉及的所有仿真操作均正常。只有出现具体证据时才核查相关运行库、仿真配置或模型兼容性，并与已成功的 J6M YOLOv8 对照，不泛化为“环境不行”，不据此取消 x86 验收目标。

## 2. 依据与工作边界

### 2.1 模型依据

- 仓库：https://github.com/yingxie285/3D_perception_sparse4d_QAT
- 配置：`dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py`。
- 导出入口：`script/qat_workflow_sparse4d.py`。
- Python 校准权重测试入口：`script/test_calib.py`。
- 可视化依据：`tool/visualization/visulaize_new.py` 及 `visulaize_new_calib.py`，实施时核实实际文件名和调用关系。
- 2026-10-29 查询的配置确认 `temporal=False`、`with_quality_estimation=False`、4 路相机、384 个 anchor、13 类、无速度回归。
- 当前模型来自工作区 `20261025/model.hbm`，现有部署副本位于 `dnn/basic_samples/code/sparse4d_J6M/assets/model.hbm`。

实施时记录仓库 commit、HBM SHA256、编译工具和运行库版本。配置文件名含 temporal 不能作为启用时序的依据。若最新源码与当前 HBM 不对应，先明确生成该 HBM 的版本，不混用不同版本的输入和解码规则。

### 2.2 参考与修改范围

- 首要实现参考为 `sparsedrive_deploy_right/.../samples/ucp_tutorial/all-round` 的实际源码、构建文件和部署说明。不能只照搬其 task.md 中尚未验证的条目。
- 先逐项比较 SparseDrive 与现有 Sparse4D 程序的 Tensor 分配、量化、缓存同步、提交、等待和释放方式，形成有源码位置的差异表。
- 保留 SparseDrive 原工程、`20261025` 原始 SDK 和现有 `basic_samples/code/sparse4d_J6M` 诊断现场。
- 新实现只放在当前 `sparse4d_deployment/.../samples/ucp_tutorial/all-round` 内的独立目录和构建目标中。
- 不覆盖原有 `run_all_round_sample`，不把两个 main 链接到同一目标，不全量递归收集源文件。
- 不把“YOLO 示例不适用”当作已确认根因。官方 hrt_model_exec 也出现过相同停滞，更换应用结构是否改善必须用同一 HBM、同一输入验证。

## 3. 模型输入输出契约

以下为当前 HBM 已查询的接口。实施时再次读取并保存运行时属性，与导出 forward 的语义映射核对；不能仅按 shape 猜测用途。

| 输入 | 语义 | 当前 HBM shape | dtype |
| --- | --- | --- | --- |
| 0 | 图像 img | `[4,3,256,704]` | s8 |
| 1 | 相机外参 | `[1,4,4,4]` | s16 |
| 2 | 变换后的相机内参 | `[1,4,3,3]` | s16 |
| 3 | 鱼眼畸变系数 | `[1,4,1,1,4]` | s16 |
| 4 | 图像变换后的 lidar2img | `[1,4,4,4]` | f32 |
| 5 | image_wh | `[1,2]` | s16 |
| 6 | time_interval | 标量，shape 为 `[]` | f32 |

当前输入 5 沿用已有导出约定 `[256,704]`，不能根据名称擅自交换为宽、高。当前无时序输入 6 沿用导出样例常量 1，不计算相邻帧时间差。实施时用当前导出样例再次验证这两项。

| 输出 | 语义 | 当前 HBM shape | dtype |
| --- | --- | --- | --- |
| 0 | 13 类分类 logits | `[1,384,13]` | s32，沿 axis 2 量化 |
| 1 | 8 维框参数 | `[1,384,8]` | f16 |

保存 model_io.json，逐项记录 index、name、shape、stride、dtype、quantizeAxis、scale、zero point、alignedByteSize。数值量化参数和 padding 不在业务代码中写死。标量有一个有效元素，不能因 shape 为空而跳过输入。

13 类顺序固定为：

```text
barrier, bus, car, construction_vehicle, motor, bicycle, animal,
pedestrian, traffic_cone, stopper, trash_bin, sign, truck
```

## 4. 数据与前处理

首个验证场景使用工作区以下现有数据：

```text
20261025/data/BEVOD_nusc_anno_dumpjson_mini/val/20240119-211122_20240119-211139.json
20261025/data/fisheye_data_2w_frame/20240119-211122_20240119-211139/
```

四路顺序为前、右、左、后，分别对应 image0、image1、image2、image3。manifest 保存同一帧的 token、timestamp、四路 camera name、图片相对路径、原始内外参和畸变系数。不能分别按四个目录的文件序号任意配帧。GT 仅用于评测，不作为 HBM 输入。

采用 C++ 读取图片和 JSON；可提供 Python 导出 manifest 工具，但正式运行不依赖 Python 或逐帧预制 Tensor。另保留原始输入 bin 模式，用于隔离前处理与模型运行问题。

前处理必须复现实际 Python 测试 pipeline，而不是按目标分辨率直接拉伸：

- 原始图片宽 1920、高 1200，模型最终宽 704、高 256。
- 当前配置与已有 pipeline 对齐结果为 resize 比例 0.44，实际图片变成 844×528，再按 `(x=0,y=140,w=704,h=256)` 裁剪。实施时保留配置来源并逐像素验证，不根据原始图片尺寸自行重算另一套裁剪。
- BGR 转 RGB，mean 为 `[123.675,116.28,103.53]`，std 为 `[58.395,57.12,57.375]`，转换至 CHW。
- 核实解码、插值、浮点运算和归一化的顺序与精度。
- 保留原始标定供原图可视化；另生成模型所需变换后标定，不能相互覆盖。
- 当前内参缩放、裁剪补偿和 lidar2img 生成规则以 Python 实际实现为准。已有对齐采用整个 K 乘缩放系数，K 不单独扣除 crop，lidar2img 使用图像增强矩阵。不能按常见针孔相机公式擅自“修正”。
- 明确外参方向、坐标系、矩阵存储和乘法顺序。检查 cam_extrinsic 与实际数据加载使用的 lidar2camera_extrinsic，不能只凭字段名认定一致。
- 保留鱼眼畸变模型。不得新增未经模型训练和导出使用的去畸变步骤。

验收时比较同一帧七个逻辑输入及量化打包输入，报告 shape、最大绝对误差、平均误差和允许容差。浮点变换误差与量化舍入误差分开记录。

## 5. 运行时与构建

### 5.1 固定目标平台

目标芯片明确为 **J6M**，编译目标 march 明确为 **`nash-m`**。x86 启动脚本在启动程序前显式设置：

```bash
export HB_UCP_SIM_PLATFORM_TYPE=nash-m
```

不得沿用参考工程的 J6P 目标、`nash-p` 配置或依赖环境默认值。aarch64 使用匹配 J6M 的 SDK 库和部署配置。使用现有 HBM 时先核实其编译目标，不因修改任务文档而重新校准或编译模型。SDK 内部子进程若出现其他 march 名称，记录并核实其内部映射，不擅自改写 SDK 程序。

### 5.2 参数定义方式

编写 `src/sparse4d/sparse4d_runtime.cc` 时，直接沿用参考文件 `sparsedrive_deploy_right/.../all-round/src/sparsedrive_run.cc` 的 gflags 方式。包含 `gflags/gflags.h`，字符串参数使用 `DEFINE_string` 定义，通过 `FLAGS_*` 读取，不自行实现命令行解析器。

例如在该 `.cc` 中集中定义：

```cpp
DEFINE_string(model_file, "model.hbm", "Path to .hbm model");
DEFINE_string(config_file, "infer_config.txt", "Path to inference config");
DEFINE_string(manifest, "", "Path to four-camera frame manifest");
DEFINE_string(dump_dir, "./output", "Directory for output tensors and results");
```

其他参数按参考方式使用 `DEFINE_bool`、`DEFINE_int32`、`DEFINE_double`，不将数值和布尔变量强行定义为字符串。每个 flag 只定义一次，其他翻译单元需要访问时使用对应 `DECLARE_*`。main 中按参考入口调用 gflags 解析，构建目标链接 gflags。

通用参数名称优先保持参考工程一致，采用 `--manifest` 和 `--dump_dir`，不新增语义相同的 `--input_manifest`、`--output_dir` 别名。删除 state_mapping、initial_state_dir 等时序专用参数，默认值按当前模型修改。

### 5.3 生命周期与实现要求

实现独立 Sparse4DRuntime，参考 SparseDrive 的资源生命周期：

1. 解析配置并加载 HBM，按 model_name 选择图。
2. 查询和校验全部 IO，分配并复用 Tensor。
3. 写入输入，执行 cache clean。
4. 调用 hbDNNInferV2、hbUCPSubmitTask、hbUCPWaitTaskDone。
5. 成功后处理输出属性，执行 cache invalidate，提取有效元素。
6. 解码、保存和可视化，释放 task；退出时释放 Tensor 和模型。

要求：

- 按当前 SDK 头文件核对 API 和枚举，不直接复制 J6P 专用参数。
- 按 alignedByteSize 分配，按实际字节 stride 访问，包括图像行 padding 和分类输出的通道 padding。
- 支持当前模型实际使用的 s8、s16、s32、f16、f32，正确处理 per-axis 量化、零点、饱和和舍入；其他类型明确报错。
- 检查所有返回值，推理失败后不得解码或保存成功结果。
- 等待超时与外部进程清理分别处理。不能认为任务等待超时后底层已安全停止，也不能无限等待收尾子进程。
- 日志标识加载、输入生成、打包、提交、等待、输出读取、后处理阶段。
- x86 和 aarch64 使用对应成套 SDK 库。记录实际库及仿真辅助程序路径，不通过混用库掩盖问题。
- 两个构建目录分别生成同名可执行文件 sparse4d_J6M，保持与旧目录目标隔离。兼容现有服务器 CMake 3.14.5，除非明确验证必须升级。

### 5.4 必需交付的 CMakeLists.txt

`all-round/CMakeLists.txt` 是正式交付物，与源码、构建脚本和部署目录一起交付。以参考 SparseDrive 同路径文件为基础适配，沿用其目录配置、链接和安装写法。

- 保留已有 all-round 工程入口，新增 `add_executable(sparse4d_J6M ...)`，显式列出实际源文件，包括 main.cc 和 sparse4d_runtime.cc。
- 复用参考的 dnn、hbucp、gflags、opencv_world 等实际依赖；若复用 SHA 校验代码则保留 crypto 依赖。
- 参考文件固定 deps_aarch64 和 aarch64 安装目录。适配为构建脚本显式选择 x86/aarch64，分别使用对应依赖目录、编译器和安装位置。
- Sparse4D 目标不得沿用参考的 PLATFORM_J6P 宏。按 J6M 实际代码需求配置，模型 march 和 x86 仿真平台保持 nash-m。
- 安装可执行文件、所需运行库、启动脚本及配置。x86 仿真辅助程序和固件按 SDK 现有方式随运行库部署。
- 调试构建可保留参考的 -O0 -g；正式时延测试使用明确的 Release 优化配置并记录构建类型。
- 两个架构的构建脚本均调用这一份 CMakeLists.txt，分别验收配置、编译、安装和部署后运行。
- 两个架构使用同一份显式源文件列表。架构差异仅在编译器、依赖库、安装目录和必要的平台配置中表达，不另建 x86 专用模型实现，不用预生成输出替代 x86 HBM 推理。

## 6. 后处理、保存与可视化

- 分类输出按实际 per-class scale 反量化后执行 sigmoid。
- 参考当前 SparseBox3DDecoder，核实展平 query/class 的 top-k、筛选顺序和阈值。已有基准为全局 top-300，显示阈值 0.5；评测阈值与显示阈值分开配置，不因显示筛选丢失评测需要的预测。
- 8 维框为 xyz、log 尺寸和 sin/cos yaw。按实际代码确认宽长高顺序，尺寸取 exp，yaw 取 atan2，不添加速度或轨迹字段。
- 不引入 YOLO 的 NMS，不擅自限制每个 query 只能输出一个类别。
- 四路原图投影使用原始标定和鱼眼畸变公式，对齐现有 Python 可视化的角点、朝向、边线裁剪及相机后方点处理。输出四视图检测框图和 BEV 图，标注类别、分数及帧标识。
- 保存两个原始输出及 tensor 元信息。紧凑有效数组与原始 padded buffer 分开命名，明确 dtype、布局、量化参数。
- 输出检测 JSON，记录 boxes、scores、labels、坐标系、单位和 token。失败帧单独标记，不能写成正常空检测。
- 提供开发机结果适配工具，复用仓库实际评测和 Python 可视化入口。明确 results.pkl 的结构与 dataset index/token 对应关系，不重新运行模型 forward 来代替 HBM 结果。

## 7. 实施阶段与验收

### 阶段 0：审阅任务及核实参考实现

- [x] 用户确认本文范围后开始实现。
- [ ] 审阅 SparseDrive 的实际代码和运行说明，记录可复用模块及差异。
- [ ] 固定 HBM、配置、源码版本和 IO 契约。

### 阶段 1：独立构建及前处理

- [ ] 新建独立目标，x86 编译通过。
- [ ] 单帧四路 manifest、标定读取和 C++ 前处理与 Python 对齐。
- [ ] 七个输入的逻辑值、量化值和 padding 校验通过。

### 阶段 2：x86 HBM 推理

- [ ] 用相同 HBM 和相同输入分别验证新 runtime 与官方 hrt_model_exec。
- [ ] 成功返回两个输出，不能把 model_info 成功、CPU 活跃或超时退出计为推理成功。
- [ ] 如果仍停在首个 BPU 分段，保存库版本、编译产物信息及仿真证据，继续定位具体原因；不得声称重写应用已修复。
- [ ] 检查推理结束和失败时的资源释放及进程退出。

### 阶段 3：完整检测结果

- [ ] C++ 解码与 Python 对同一组 HBM 原始输出的解码一致。
- [ ] 四路原图和 BEV 可视化完成，与 Python 基准对照。
- [ ] 同一场景逐帧独立运行，不引入缓存。调整帧处理顺序后，同帧结果在约定容差内一致。
- [ ] 使用相同数据子集和过滤规则评测 HBM 结果，分别报告浮点、校准模型和 HBM 指标，不预设三者必然相同。

### 阶段 4：J6M 部署和时延

- [ ] aarch64 构建并在 J6M 实际运行同一输入。
- [ ] 对比板端与 x86 HBM 输出，确认检测与可视化可用。
- [ ] 分别记录读取图片、前处理、量化打包、推理调用、输出转换、检测后处理、可视化、文件保存和端到端时间。
- [ ] 推理调用时间覆盖创建、提交和等待，包含 HBM 的 CPU/BPU 混合执行，不能标为纯 BPU 硬件时间。
- [ ] 性能模式关闭逐帧 dump 和可视化，报告预热数、测量帧数、均值、P50、P95，以及板端频率和运行库版本。

## 8. 建议交付结构与命令

```text
all-round/
├── CMakeLists.txt
├── src/sparse4d/main.cc
├── src/sparse4d/sparse4d_runtime.cc
├── src/sparse4d/tensor_io.cc
├── src/sparse4d/input_preprocess.cc
├── src/sparse4d/detection_decode.cc
├── src/sparse4d/fisheye_visualization.cc
├── include/sparse4d/
├── configs/sparse4d/model_io.json
├── configs/sparse4d/frame_manifest.json
├── tools/sparse4d/export_manifest.py
├── tools/sparse4d/evaluate_dump.py
├── build_sparse4d_x86.sh
├── build_sparse4d_aarch64.sh
├── deploy/sparse4d/x86/
├── deploy/sparse4d/aarch64/
└── task.md
```

以下为拟定接口，不表示当前已存在：

```bash
./sparse4d_J6M \
  --model_file=./model/model.hbm \
  --model_name=forward \
  --manifest=./data/frame_manifest.json \
  --model_io=./configs/model_io.json \
  --dump_dir=./output \
  --num_frames=1 \
  --save_outputs=true \
  --visualize=true \
  --task_timeout_ms=60000
```

配置使用相对路径或显式 data_root，不依赖原开发者的服务器路径。最终提交构建与运行说明、输入对齐报告、实际检测图、匹配子集评测结果和 J6M 时延报告。当前仅交付这份任务定义。
