# Sparse4D 四路鱼眼 3D 检测部署（J6M）

基于 Sparse4D 改造的单帧纯检测模型，四路鱼眼相机、13 个类别，无时序缓存，不包含地图、运动预测或规划。主仓库：https://github.com/yingxie285/3D_perception_sparse4d_QAT 。

已在 J6M 上验证单帧完整流程可运行。后续以板端测试为目标，x86 HBM 仿真通过不作为前置条件。Linux x86 服务器用于准备数据和交叉编译。C++ 源码用于aarch64构建。

## 1. 仓库内容与依赖

本目录只提交 Sparse4D 源码、构建脚本、模型 IO 配置和数据导出工具，不提交完整 SDK、编译结果、模型权重和图片数据。

```text
all-round/
├── CMakeLists.txt
├── build_sparse4d_aarch64.sh
├── build_sparse4d_x86.sh
├── run_sparse4d.sh
├── src/sparse4d/{main.cc,sparse4d_runtime.cc}
├── include/sparse4d/
├── configs/sparse4d/model_io.json
├── tools/sparse4d/
│   ├── export_demo.py
│   ├── export_frame.py
│   ├── verify_parity.py
│   └── evaluate_dump.py
└── tests/sparse4d/
```

自行安装与板端匹配的地平线 SDK。本工程使用的版本为 `horizon_j6_open_explorer_v3.8.1-py310_20260326`，目标为 **J6M / nash-m**。将本目录放到完整 SDK 的 `samples/ucp_tutorial/all-round` 下，旁边保留 SDK 的 `deps_aarch64`。

服务器需要 CMake 3.14 或更高版本、Make、aarch64 交叉编译器。编译使用 SDK 的 UCP/DNN、OpenCV、gflags、nlohmann 头文件和库。板端不需要 Python、PyTorch 或训练环境。

只构建 `sparse4d_J6M`，保持 `BUILD_ALL_ROUND_SAMPLE=OFF`。原始摄像头/Web 示例代码不在本源码包中，不能启用该选项。

HBM 由仓库 `script/qat_workflow_sparse4d.py` 的 calib 导出流程生成。请使用自己的同版本 `model.hbm`，记录生成配置、checkpoint、工具链版本及 SHA256。本仓库不附模型下载文件。

## 2. 服务器交叉编译

```bash
export REPO_ROOT=/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws
export SDK_ROOT="$REPO_ROOT/horizon_j6_open_explorer_v3.8.1-py310_20260326"
export SAMPLE_ROOT="$SDK_ROOT/samples/ucp_tutorial/all-round"
export LINARO_GCC_ROOT=/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu
cd "$SAMPLE_ROOT"

"$LINARO_GCC_ROOT/bin/aarch64-none-linux-gnu-g++" -dumpmachine
bash build_sparse4d_aarch64.sh
file deploy/sparse4d/aarch64/bin/sparse4d_J6M
```

根据实际安装位置修改上述变量。编译器目标应为 aarch64，生成程序应显示 ARM aarch64。构建目录为 `build_sparse4d_aarch64`，安装目录为 `deploy/sparse4d/aarch64`。

若链接阶段提示 libbpu、libhbmem 等间接依赖找不到，应核对 SDK BSP 库与交叉链接配置。参考 SparseDrive 工程使用 `-Wl,--unresolved-symbols=ignore-in-shared-libs`，其含义是把共享库内部引用的解析留给运行环境，不代表板端可以缺少这些库。用户服务器已成功构建并运行的 CMake 配置应同步保存；不要仅根据本 README 将链接失败标记为通过。

## 3. 在服务器生成 100 帧 demo 数据

`export_demo.py` 与 `export_frame.py` 必须放在同一目录。导出仅使用 Python 标准库；根据仓库配置生成与已验证单帧相同的标定和预处理格式。

```bash
cd "$SAMPLE_ROOT"
python3 tools/sparse4d/export_demo.py \
  --repo-root="$REPO_ROOT" \
  --ann-path="$REPO_ROOT/data/BEVOD_nusc_anno_dumpjson_mini/val" \
  --image-root="$REPO_ROOT/data/fisheye_data_2w_frame" \
  --output-dir="$PWD/data/sparse4d/demo_100frames" \
  --num-frames=100
```

`--ann-path` 可指定一个场景 JSON，也可指定标注目录。默认按时间戳排序，`--num-frames` 是所有所选场景合计的帧数。`--num-frames=-1` 导出全部；`--start-index=5` 从第六帧开始；`--stride=2` 每两帧取一帧。不隐式应用训练配置的 load_interval。输出目录必须为空。

导出图片顺序为前、右、左、后。正常推理只需要图片与 frame.json，不需要 reference_inputs。后者仅用于 Python/C++ 输入对齐。

## 4. 整理部署包

```bash
cd "$SAMPLE_ROOT"
export PACKAGE_ROOT="$PWD/deploy/sparse4d/aarch64"
mkdir -p "$PACKAGE_ROOT/model" "$PACKAGE_ROOT/data" "$PACKAGE_ROOT/configs"

# 替换为本次实际编译生成的 HBM 路径，不依赖旧 dnn 示例目录。
export HBM_FILE=/absolute/path/to/model.hbm
cp "$HBM_FILE" "$PACKAGE_ROOT/model/model.hbm"
cp configs/sparse4d/model_io.json "$PACKAGE_ROOT/configs/"
cp -a data/sparse4d/demo_100frames "$PACKAGE_ROOT/data/"

# 应无空文件；发现空文件时先检查源数据，不继续部署。
find "$PACKAGE_ROOT/data/demo_100frames/frames" -type f -size 0c
cd "$PACKAGE_ROOT"
sha256sum model/model.hbm > model.sha256
```

最终目录：

```text
aarch64/
├── bin/{sparse4d_J6M,run_sparse4d.sh}
├── lib/                         # aarch64 库，不混入 x86 库
├── model/model.hbm
├── model.sha256
├── configs/model_io.json
└── data/demo_100frames/
    ├── manifest.json            # 总入口，列出各 frame.json 的相对路径
    ├── dataset.json
    └── frames/
        ├── frame_000000/
        │   ├── frame.json
        │   └── images/          # 前、右、左、后四张图片
        ├── ...
        └── frame_000099/
```

## 5. 传输至 J6M域控制器
使用MobaXterm软件连接域控制器和PC电脑，将PC电脑里的部署包（aarch64）拖动复制到域控制器目标目录（app/）。


## 6. 板端运行

```bash
cd /app/xieying/new/aarch64
set -o pipefail
bash bin/run_sparse4d.sh \
  --model_file="$PWD/model/model.hbm" \
  --model_name=forward \
  --model_io="$PWD/configs/model_io.json" \
  --manifest="$PWD/data/demo_100frames/manifest.json" \
  --dump_dir="$PWD/output_100frames" \
  --num_frames=-1 \
  --warmup=0 --repeat=1 \
  --save_outputs=true --visualize=true \
  --visualize_score_threshold=0.5 \
  --dump_inputs=false --trace=true \
  2>&1 | tee run_100frames.log
```

Python 导出参数是 `--num-frames`，C++ 运行参数是 `--num_frames`。C++ 默认只跑 1 帧，使用 `-1` 才遍历清单中所有帧。启动脚本设置包内 lib 路径和 nash-m。

每帧结果位于 `output_100frames/0/` 至 `99/`，包括 detections.json、原始/反量化输出、four_cameras.jpg、bev.jpg 和 timing.json。各帧独立推理，共用模型与 Tensor 内存，不使用时序缓存。用 detections.json 的 frame_id 对应 dataset.json 中的 token。

## 7. 输入、输出与性能范围

模型要求七个输入：图像、外参、内参、畸变、lidar2img、image_wh、time_interval。输出为 384 个 query 的 13 类 logits 和 8 维框参数。程序根据运行时属性进行量化和 stride 处理。

原图 1920×1200，当前测试配置缩放至 844×528，再裁剪 704×256。不得改成直接拉伸。四路原图可视化使用原始标定及鱼眼投影。

计时中的 inference 包含任务创建、提交和等待，可能含 CPU/BPU 混合执行，不等于纯 BPU 硬件时间。保存图片和输出开启时的端到端耗时也包含文件 IO。正式性能测试应关闭可视化、输出 dump 和逐步日志，并记录预热数、重复数与板端频率。

## 8. Git 提交约定

SDK 目录及本目录的 .gitignore 采用源码白名单，不提交 deps、package、ai_toolchain、deploy、build、模型、图片、日志或结果文件。若仓库根 .gitignore 已忽略整个 SDK，必须在根规则末尾逐层放行本目录，否则子目录的 .gitignore 不会生效。

.gitignore 只影响未跟踪文件。提交前检查 `git status --short` 和 `git diff --cached --stat`，避免把服务器生成的数据/库加入仓库。已经跟踪的文件需按具体路径用 `git rm --cached` 取消跟踪，不能用删除本地文件代替。
