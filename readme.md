# 数据预处理
## 一、检查并准备百度原始数据集
1. 真值可视化验证：验证原始数据集的3D框真值投影到图像上是不是对的，运行tool/visualization/vis_gt_label.py，运行前更改第167行dataset_dir数据集路径、第171行可视化图片保存路径save_dir。
2. 确定百度数据集有几套内外参（对应几个数据采集车）：运行tool/data_converter/collect_unique_camera_config.py，可以知道有4套内外参，也就相当于4辆不同的数据采集车。因此，需要为每一帧每一个相机都用自己独自的内外参，后续的数据预处理脚本已实现这个功能。运行前，把第5行root_dir的路径换成您的路径（data/fisheye_data_2w_frame，这里面是按场景排列的数据）。
3. 将百度的原始数据集放在data/fisheye_data_2w_frame文件夹。形成
```text
data/
├── fisheye_data_2w_frame/   #2万帧鱼眼数据集
```

## 二、模型需要的数据预处理
1. 运行data_preprocessing/pcd2npy_2w.py，将百度数据集的点云从pcd格式转化成npy格式。此脚本会在原路径下生成新的npy格式的点云，不会覆盖原有pcd格式的点云。npy格式的点云数据IO的速度更快，且模型读取的是npy格式的点云，因此要转换。运行前更改第99行数据集的路径root_dir。

2. 在/data/fisheye_data_2w_frame目录下创建一个空的data_info.json。运行data_preprocessing/bevod_split_dataset.py，会得到含有数据集信息的data_info.json。运行前记得修改bevod_split_dataset.py里面的第8行dataset_root的路径，检查DEFAULT_TRAIN_VAL_SPLIT里面的场景名（严格按照场景特点划分的训练集和验证集）、output_root的路径是否正确。

3. 运行data_preprocessing/bevod_to_nuscenes.py，会在data/nuscenes_format生成Nuscenes形式的json。运行前记得确认info_root、data_root、save_root的路径对不对，以及def parse_args():函数里的'--local-root'和'--workers'。（注意：最好不要点击生成的data/nuscenes_format/v1.0-trainval/sample_annotation.json，因为真值标签很多，这个json比较大，vs code可能会卡住。您可以用命令ls -lh sample_annotation.json的路径来看sample_annotation.json的内存是多大）

4. 运行data_preprocessing/BEVOD_dump_nusc_ann_json_mini.py，将Nuscenes形式转成dump形式，模型需要dump形式。
运行前需要检查BEVOD_dump_nusc_ann_json_mini.py的第402行create_nuscenes_infos函数里，点击train_scenes = splits.bevod_train中的bevod_train和
val_scenes = splits.bevod_val中的bevod_val。【Pylance插件有用就能自动进入splits.py，插件没用就手动点击dataset/utils/custom_nuscenes/utils/splits.py】。
进入dataset/utils/custom_nuscenes/utils/splits.py找到第151行bevod_train和bevod_val，确认bevod_train和bevod_val的场景划分，保持和train_val_split.json一致（其实就是把train_val_split.json的[]里面的内容对应复制到dataset/utils/custom_nuscenes/utils/splits.py的bevod_train和bevod_val的全局变量定义，覆盖之前的bevod_train和bevod_val的内容）其中，bevod_train就是train_val_split.json里面的"train"，bevod_val就是train_val_split.json里面的"val"，如果更改了就要按ctrl+S保存。

5. 运行data_preprocessing/nusc_anchor_generator.py，得到初始化的anchor（bevod_kmeans900.npy）。运行前记得确认dump_root、save_root的路径对不对。第73行parser.add_argument("--num_anchor", type=int, default=900)900改成384。

最终，数据处理成：
```text
data/
├── BEVOD_nusc_anno_dumpjson_mini/   # 模型需要的数据集标注文件，对应config的ann_file
├── fisheye_data_2w_frame/           # 原始鱼眼数据集（2万帧）
├── nuscenes_format/                 # nuScenes 标准格式的数据集
├── bevod_kmeans384.npy              # BEVOD 模型的 K-Means 聚类结果（384个聚类中心）
└── dataset_import_debug.log         # 数据集导入过程的调试日志
```


# 浮点训练
## 一、检查config
点击dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
1. 确认模型参数和路径是不是对的。尤其是ann_file和data_root的路径
2. 更改num_iters_per_epoch = int(15188 // (num_gpus * batch_size))  #训练集帧数，和您划分的训练集的帧数一致，示例是用了15188帧数据作为训练集。您用了XXX帧训练集，就把15188改成多少。

## 二、运行train.py进行单卡debug
1. 将dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py的batch size改为1，显卡num_gpus用1张卡，方便debug调试。
2. 在launch.json里的"name": "Distributed Training",检查所有参数，检查launch.json的"args"字典里面的路径对不对。
单卡batch_size是10，纯检测的版本显存是22个G

## 三、用tmux命令进行分布式训练
1. 将dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py的batch size改为显卡支持的最大bs，同时注意学习率线性缩放。
2. 点击tmux.txt，分为两种方式：一种是重头开始训练，另一种是带resume的训练。
命令含义如下：
cd 进入工程文件夹
source 环境路径
更改script/train.py和config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py为自己的路径
先创建train_log目录，并将重定向的日志0106_r50_bs8.log改为自定义的日志名称。

# 浮点测试
## 一、模型推理test可视化
1. 在launch.json里的"name": "sparse4de2e_visual_new",更改program的路径，更改launch.json的"args"字典里面的路径。
2. 点击tool/visualization/visulaize_new.py，确认第70行的samples_per_gpu = cfg["data"]["val"].pop("samples_per_gpu", 1)是val，因为示例只划分了train和val。（如果您划分了test，可以写cfg["data"]["tset"]。）
3. 更改第143行save_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/visualization_0104_pred_0.5_epoch83"为您的保存路径。
4. 运行launch.json里的tool/visualization/visulaize_new.py
（注：模型test用的得分阈值是0.5，如果要修改得分阈值，在modules/head/decoder/decoder.py第15行，修改score_threshold得分阈值）
参考命令：
```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python tool/visualization/visulaize_new.py \
--config dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--checkpoint e2e_worklog_0207_all_camera_config_no_quality/sparse4dv3_temporal_r50_1x8_bs6_256x704/iter_57150.pth
```

## 附：代码主要更改说明（详见飞书文档）：
1. config的更改
2. NuScenes4DDetTrackDataset类的修改，路径在dataset/nuscenes_dataset.py
3. 鱼眼深度图 Class MultiScaleDepthMapGeneratorFisheye，路径在dataset/pipeline/nuscenes/nusc_pipeline.py
4. 特征采样的投影函数，路径在modules/head/sparse4d_blocks/core_blocks.py


# J6部署-服务器端
## 一、calib阶段
运行下面的命令会进行run_calib_phase并得到导出的HBM模型。
需替换实际的config与base-checkpoint的路径，以及日志的名字。参考命令：
```bash
cd /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws
tmux new -s sparse4d_1024calib_fisheye
export QAT_OUTPUT_PREFIX=1024_fisheye_fix
HORIZON_QAT=1 \
CUDA_VISIBLE_DEVICES=7 TORCH_DISTRIBUTED_DEBUG=DETAIL \
PYTHONPATH=$(pwd)/..:$PYTHONPATH \
python3 -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=28671 \
script/qat_workflow_sparse4d.py \
--config \
/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--base-checkpoint \
/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/ckpt/iter_36800.pth \
--phase calib \
--march nash-m \
--no-validate \
--launcher none \
2>&1 | tee sparse4d_20261024calib_fisheye.log
```

## 二、测试calib阶段保存的权重
运行下面的命令会测试calib阶段保存的权重,得到测试集的精度结果，作为定点模型的结果与浮点模型的结果进行对比。
需替换实际的config、base-checkpoint、calib权重（ckpt）的路径，以及日志的名字。参考命令：
```bash
export QAT_OUTPUT_PREFIX=1024_fisheye_fix
HORIZON_QAT=1 \
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=$(pwd)/..:$PYTHONPATH \
python3 script/test_calib.py \
--config \
/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--base-checkpoint \
/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/ckpt/iter_36800.pth \
--checkpoint \
/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/Sparse4D_QAT/1024_fisheye_fix.ckpt \
--march nash-m \
--eval bbox NDS mAP\
2>&1 | tee eval_data3k测试calib1024_fisheye_fix.log
```

## 三、calib测试可视化
与test_calib.py逻辑一致，visulaize_new_calib.py此脚本是得到可视化图片。
需替换实际的save_dir、checkpoint的路径。参考命令：
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python3 tool/visualization/visulaize_new_calib.py \
--config dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--save_dir /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/vis_pred/1028 \
--march nash-m \
--checkpoint /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/Sparse4D_QAT/1024_fisheye_fix.ckpt \
--base-checkpoint /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/ckpt/iter_36800.pth
```

## 四、生成demo数据文件夹，以供板端测试
### 1. 服务器生成 100 帧数据
```bash
export REPO_ROOT=/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws
export SDK_ROOT="$REPO_ROOT/horizon_j6_open_explorer_v3.8.1-py310_20260326"

cd "$SDK_ROOT/samples/ucp_tutorial/all-round"

python3 tools/sparse4d/export_demo.py \
  --repo-root="$REPO_ROOT" \
  --ann-path="$REPO_ROOT/data/BEVOD_nusc_anno_dumpjson_mini/val" \
  --image-root="$REPO_ROOT/data/fisheye_data_2w_frame" \
  --output-dir="$PWD/data/sparse4d/demo_100frames" \
  --num-frames=100
```
默认将标注按时间戳排序，从第一帧开始取 总计 100 帧，每帧复制前、右、左、后四张图片。

若只想导出一个场景，将 --ann-path 改为：
--ann-path="$REPO_ROOT/data/BEVOD_nusc_anno_dumpjson_mini/val/20240119-211122_20240119-211139.json"

### 2. 使用MobaXterm软件连接域控制器和PC电脑，将100 帧数据的文件夹（demo_100frames）拷贝到板端
为避免文件丢失，使用zip把demo_100frames文件夹压缩后，再拖动到板端，并解压zip。


# J6部署-板端
在服务器里进行交叉编译，以生成板端能运行的产物。
## 1. 服务器设置路径
确保本地最新的 all-round 已上传到服务器，然后执行：
```bash
export SDK_ROOT=/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/horizon_j6_open_explorer_v3.8.1-py310_20260326
export SAMPLE_ROOT="$SDK_ROOT/samples/ucp_tutorial/all-round"

cd "$SAMPLE_ROOT"
```

## 2. 检查交叉编译器
下面是参考工程使用的默认路径。如果实际安装位置不同，只修改这一行：
```bash
export LINARO_GCC_ROOT=/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu
```
```bash
"$LINARO_GCC_ROOT/bin/aarch64-none-linux-gnu-g++" --version
"$LINARO_GCC_ROOT/bin/aarch64-none-linux-gnu-g++" -dumpmachine
```
第二条应输出：aarch64-none-linux-gnu
如果文件不存在，查找实际位置：
```bash
find /opt /usr/local /home /data1 / \
  -maxdepth 5 -name aarch64-none-linux-gnu-g++ 2>/dev/null
```

## 3. 检查依赖并编译
```bash
ls "$SDK_ROOT/samples/ucp_tutorial/deps_aarch64/ucp/lib"
ls "$SDK_ROOT/samples/ucp_tutorial/deps_aarch64/opencv/lib"
ls "$SDK_ROOT/samples/ucp_tutorial/deps_aarch64/gflags/lib"

cd "$SAMPLE_ROOT"
bash build_sparse4d_aarch64.sh
```
编译、安装完成后检查：
```bash
file deploy/sparse4d/aarch64/bin/sparse4d_J6M
ls -lh deploy/sparse4d/aarch64/bin/
ls -lh deploy/sparse4d/aarch64/lib/
```
file 输出应包含 ARM aarch64，不能是 x86-64。

这个构建使用独立的 build_sparse4d_aarch64 目录，不影响已经编译的 x86 程序。

## 4. 在服务器整理完整部署包
直接在安装目录补上模型和数据：
```bash
cd "$SAMPLE_ROOT"
export PACKAGE_ROOT="$SAMPLE_ROOT/deploy/sparse4d/aarch64"

mkdir -p "$PACKAGE_ROOT/model"
mkdir -p "$PACKAGE_ROOT/data"
mkdir -p "$PACKAGE_ROOT/configs"

cp "$SDK_ROOT/samples/ucp_tutorial/dnn/basic_samples/code/sparse4d_J6M/assets/model.hbm" \
   "$PACKAGE_ROOT/model/model.hbm"

cp -a "$SAMPLE_ROOT/data/sparse4d/demo_100frames" \
      "$PACKAGE_ROOT/data/"

cp "$SAMPLE_ROOT/configs/sparse4d/model_io.json" \
   "$PACKAGE_ROOT/configs/model_io.json"
```
部署包应包含：
```text
aarch64/
├── bin/
│   ├── sparse4d_J6M
│   └── run_sparse4d.sh
├── lib/                              # aarch64 运行库
├── model/
│   └── model.hbm
├── configs/
│   └── model_io.json
└── data/
    └── demo_100frames/
        ├── manifest.json             # 100 帧配置文件的相对路径列表
        ├── dataset.json              # 帧数、场景、token、时间戳
        └── frames/
            ├── frame_000000/
            │   ├── frame.json        # 当前帧标定、畸变、预处理及图片路径
            │   └── images/
            │       ├── 00_FISHEYE_CAM_FRONT.jpeg
            │       ├── 01_FISHEYE_CAM_RIGHT.jpeg
            │       ├── 02_FISHEYE_CAM_LEFT.jpeg
            │       └── 03_FISHEYE_CAM_BACK.jpeg
            ├── frame_000001/
            │   ├── frame.json
            │   └── images/
            │       └── …             # 当前帧四张图片
            ├── …
            └── frame_000099/
                ├── frame.json
                └── images/
                    └── …             # 当前帧四张图片
```

## 5. 将部署包拷贝到域控制器运行
使用MobaXterm软件连接域控制器和PC电脑，将PC电脑里的部署包（aarch64）拖动复制到域控制器目标目录（app/）。

## 6. 板端执行100帧完整推理
```bash
cd /app/aarch64
set -o pipefail

bash bin/run_sparse4d.sh \
  --model_file="$PWD/model/model.hbm" \
  --model_name=forward \
  --model_io="$PWD/configs/model_io.json" \
  --manifest="$PWD/data/demo_100frames/manifest.json" \
  --dump_dir="$PWD/output_100frames" \
  --num_frames=-1 \
  --warmup=0 \
  --repeat=1 \
  --save_outputs=true \
  --visualize=true \
  --visualize_score_threshold=0.5 \
  --dump_inputs=false \
  --trace=true \
  2>&1 | tee run_100frames.log
```
这里执行一次推理，没有预热。启动脚本设置 nash-m 和包内运行库路径。

成功会生成：
```text
aarch64/
├── run_100frames.log
└── output_100frames/
    ├── 0/                            # 第一帧结果
    ├── 1/
    ├── …
    └── 99/                           # 第100帧结果
```
## 7. 检测结果回放
将板端生成的output_100frames拷贝至PC电脑或者服务器里，运行tool/visualization/output_100frames_to_mp4.py脚本，可以将图片变成视频，方便查阅效果。运行前更改第12行input_dir和第13行output_mp4的路径。


# 更换域控制器（J6M换成J6P）
有两个必须分开的改动：
1. 在服务器里用 nash-p 重新编译模型，得到 J6P 的 HBM。
2. 复制一份 all-round 部署工程，修改程序名和启动脚本，再交叉编译、打包到 J6P域控制器。

## 一、用 nash-p 重新编译模型
在script/qat_workflow_sparse4d.py脚本里是通过参数parser.add_argument("--march", )定义的，需通过命令显式传入nash-p。
运行下面的命令会进行run_calib_phase并得到导出的HBM模型。
需替换实际的config与base-checkpoint的路径，以及日志的名字。设置独立的 QAT_OUTPUT_PREFIX 和 --output-dir 是为了不覆盖已有的J6M的model.hbm、qat.bc 和 quantize.bc。参考命令：
```bash
cd /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws
tmux new -s sparse4d_calib_fisheye_J6P
export QAT_OUTPUT_PREFIX=J6P_export
HORIZON_QAT=1 \
CUDA_VISIBLE_DEVICES=7 TORCH_DISTRIBUTED_DEBUG=DETAIL \
PYTHONPATH=$(pwd)/..:$PYTHONPATH \
python3 -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=28671 \
script/qat_workflow_sparse4d.py \
--config \
/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--base-checkpoint \
/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/ckpt/iter_36800.pth \
--phase calib \
--march nash-p \
--no-validate \
--launcher none \
2>&1 | tee sparse4d_20261030calib_J6P_export.log
```
产物为：/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/Sparse4D_QAT/J6P_export/qat_export/model.hbm


## 二、交叉编译出J6P的部署工程
1. 在服务器复制工程
以下命令在存放完整 SDK 的 Linux 服务器执行。先按你的实际路径修改 SDK_ROOT：
```bash
export SDK_ROOT=/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/horizon_j6_open_explorer_v3.8.1-py310_20260326
export UCP_ROOT="$SDK_ROOT/samples/ucp_tutorial"

cd "$UCP_ROOT"
```

保留已经通过 J6M 板端验证的目录。
```bash
cp -a all-round all-round-j6p
cd all-round-j6p
```
这会复制源码、构建脚本以及目录中现有的数据和构建产物。因此不要直接把旧的 deploy/sparse4d/aarch64 当成 J6P 部署包。下文会让新工程重新构建，并在打包时明确放入新 HBM。
复制后先确认文件：
```bash
ls CMakeLists.txt build_sparse4d_aarch64.sh run_sparse4d.sh
ls ../deps_aarch64
```

2. 修改 all-round-j6p/CMakeLists.txt
打开all-round-j6p/CMakeLists.txt，其中用到了命名的地方已更改为sparse4d_J6P与J6P，如下所示：
add_executable(sparse4d_J6P src/sparse4d/main.cc src/sparse4d/sparse4d_runtime.cc)
target_link_libraries(sparse4d_J6P dnn hbucp gflags opencv_world pthread dl rt)
set_target_properties(sparse4d_J6P PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS sparse4d_J6P DESTINATION bin)
SPARSE4D_ARCH=aarch64 保持原样。它描述的是 CPU 架构，并不是模型的 nash-m 或 nash-p。

3. 修改 all-round-j6p/run_sparse4d.sh
run_sparse4d.sh第5行和第6行已修改为J6P的命名，如下所示：
export HB_UCP_SIM_PLATFORM_TYPE=nash-p
exec "$bin/sparse4d_J6P" "$@"

4. horizon_j6_open_explorer_v3.8.1-py310_20260326/samples/ucp_tutorial/all-round-j6p/src/sparse4d/sparse4d_runtime.cc原本的sparse4d_J6M，用于诊断日志和报错前缀。已改成 sparse4d_J6P，便于区分两次板端测试。这两处文字不决定硬件目标。注释中的J6M也已改为J6P。

5. 检查还有没有J6M字样
在复制出的工程执行：
```bash
cd "$UCP_ROOT/all-round-j6p"

rg -n 'nash-m|sparse4d_J6M|J6M' \
  CMakeLists.txt run_sparse4d.sh src include configs tools \
  build_sparse4d_aarch64.sh build_sparse4d_x86.sh
```
允许只剩 export_demo.py 的 J6M/J6P 说明。

6. 清理复制来的 J6M 构建缓存和部署产物，必须确认是在all-round-j6p的文件夹
```bash
cd /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/horizon_j6_open_explorer_v3.8.1-py310_20260326/samples/ucp_tutorial/all-round-j6p
pwd
```
pwd 的末尾必须是 /all-round-j6p。确认当前目录后，只删除 J6P 副本内的这两个生成目录：
```bash
rm -rf -- ./build_sparse4d_aarch64 ./deploy/sparse4d/aarch64
```
其中 build_sparse4d_aarch64 含复制来的 CMake 缓存；deploy/sparse4d/aarch64 可能含旧的 J6M 程序、库和模型，因此需要清理。src/、include/、configs/、tools/ 和 data/ 不在删除范围内。


7. 重新交叉编译
由于horizon_j6_open_explorer_v3.8.1-py310_20260326/samples/ucp_tutorial/all-round-j6p/
build_sparse4d_aarch64.sh不包含 `nash-m `，不需要改成 `nash-p `。它构建的是 AArch64 C++ 程序。

在 J6P 工程目录执行：
```bash
cd "$UCP_ROOT/all-round-j6p"
export LINARO_GCC_ROOT=/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu

bash build_sparse4d_aarch64.sh
file deploy/sparse4d/aarch64/bin/sparse4d_J6P
```
预期生成 deploy/sparse4d/aarch64/bin/sparse4d_J6P。该部署目录中的库来自 SDK 的 deps_aarch64。部署前应确认这套用户态库与你的 J6P BSP 和驱动兼容。

8. 放入 J6P HBM、数据和配置,构建完整的部署包
```bash
cd "$UCP_ROOT/all-round-j6p"

export PACKAGE_ROOT="$PWD/deploy/sparse4d/aarch64"
export J6P_HBM=/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/Sparse4D_QAT/J6P_export/qat_export/model.hbm

mkdir -p "$PACKAGE_ROOT/model" "$PACKAGE_ROOT/data" "$PACKAGE_ROOT/configs"

cp "$J6P_HBM" "$PACKAGE_ROOT/model/model.hbm"
cp configs/sparse4d/model_io.json "$PACKAGE_ROOT/configs/"
cp -a data/sparse4d/demo_100frames "$PACKAGE_ROOT/data/"

cd "$PACKAGE_ROOT"
sha256sum model/model.hbm > model.sha256
ls -lh bin/sparse4d_J6P model/model.hbm
```
此命令会在aarch64文件夹下产生configs、data、model等文件
```text
deploy/
└── sparse4d/
    └── aarch64/
        ├── bin/                  # AArch64 可执行程序
        ├── configs/              # 模型输入输出配置文件
        ├── data/
        │   └── demo_100frames/   # 100帧测试数据
        ├── lib/                  # AArch64 动态链接库
        ├── model/                # J6P HBM 模型
        └── model.sha256          # HBM 模型 SHA-256 校验文件
```

9. 打包部署包
```bash
cd "$UCP_ROOT/all-round-j6p/deploy/sparse4d"
tar -czvf "$UCP_ROOT/all-round-j6p/aarch64.tar.gz" aarch64/
```


## 三、 板端执行
1. 使用MobaXterm软件连接域控制器和PC电脑，将PC电脑里的部署包（aarch64）拖动复制到域控制器目标目录（app/J6P）。
2. 解压aarch64.tar.gz
```bash
cd app/J6P
tar -xzf aarch64.tar.gz
```
3. 板端执行100帧完整推理
```bash
cd /app/aarch64
set -o pipefail

bash bin/run_sparse4d.sh \
  --model_file="$PWD/model/model.hbm" \
  --model_name=forward \
  --model_io="$PWD/configs/model_io.json" \
  --manifest="$PWD/data/demo_100frames/manifest.json" \
  --dump_dir="$PWD/output_j6p_100frames" \
  --num_frames=-1 \
  --warmup=0 \
  --repeat=1 \
  --save_outputs=true \
  --visualize=true \
  --visualize_score_threshold=0.5 \
  --dump_inputs=false \
  --trace=true \
  2>&1 | tee run_j6p_100frames.log
```
这里执行一次推理，没有预热。启动脚本设置了 nash-p 和包内运行库路径。

成功时，并生成：
```text
aarch64/
├── run_j6p_100frames.log
└── output_j6p_100frames/
    ├── 0/                            # 第一帧结果
    ├── 1/
    ├── …
    └── 99/                           # 第100帧结果
```

4. 将板端生成的output_j6p_100frames拷贝至PC电脑或者服务器里，运行tool/visualization/output_100frames_to_mp4.py脚本，可以将图片变成视频，方便查阅效果。运行前更改第12行input_dir和第13行output_mp4的路径。