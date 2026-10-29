# 数据预处理
## 一、检查百度原始数据集
1、真值可视化验证：验证原始数据集的3D框真值投影到图像上是不是对的，运行tool/visualization/vis_gt_label.py，运行前更改第167行dataset_dir数据集路径、第171行可视化图片保存路径save_dir。
2、确定百度数据集有几套内外参（对应几个数据采集车）：运行tool/data_converter/collect_unique_camera_config.py，可以知道有4套内外参，也就相当于4辆不同的数据采集车。因此，需要为每一帧每一个相机都用自己独自的内外参，后续的数据预处理脚本已实现这个功能。运行前，把第5行root_dir的路径换成您的路径（data/fisheye_data_2w_frame，这里面是按场景排列的数据）。

## 二、模型需要的数据预处理
1、运行sparse4d/6/offline_data_aug.py得到数据增强后的真值数据集。将百度数据集的点云从pcd格式转化成npy格式。此脚本会在原路径下生成新的npy格式的点云，不会覆盖原有pcd格式的点云。npy格式的点云数据IO的速度更快，且模型读取的是npy格式的点云，因此要转换。

2、真值可视化验证：验证数据集的3D框真值投影到图像上是不是对的，运行sparse4d/6/vis_gt_label.py。只看前3个场景

3、运行sparse4d/6/bevod_split_dataset.py，会得到划分数据集的data_info.json。

4、运行sparse4d/6/bevod_to_nuscenes.py，会在data/nuscenes_format生成Nuscenes形式的json。（注意：最好不要点击生成的data/nuscenes_format/v1.0-trainval/sample_annotation.json，因为真值标签很多，这个json比较大，vs code可能会卡住。您可以用命令ls -lh sample_annotation.json的路径来看sample_annotation.json的内存是多大）

5、运行sparse4d/6/BEVOD_dump_nusc_ann_json_mini.py，将Nuscenes形式转成dump形式，模型需要dump形式。
运行前需要在BEVOD_dump_nusc_ann_json_mini.py的create_nuscenes_infos函数里，点击train_scenes = splits.bevod_train中的bevod_train和
val_scenes = splits.bevod_val中的bevod_val。
进入dataset/utils/custom_nuscenes/utils/splits.py更改bevod_train和bevod_val，确认bevod_train和bevod_val的场景划分，保持和train_val_split.json一致（其实就是把train_val_split.json的[]里面的内容对应复制到dataset/utils/custom_nuscenes/utils/splits.py的bevod_train和bevod_val的全局变量定义，覆盖之前的bevod_train和bevod_val的内容）其中，bevod_train就是train_val_split.json里面的"train"，bevod_val就是train_val_split.json里面的"val"，更改后按ctrl+S保存。

6、运行sparse4d/6/nusc_anchor_generator.py，得到初始化的anchor（bevod_kmeans900.npy）。

# 浮点训练
## 一、检查config
点击dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
1、确认模型参数和路径是不是对的。尤其是ann_file和data_root的路径
2、更改num_iters_per_epoch = int(15188 // (num_gpus * batch_size))  #训练集帧数，和您划分的训练集的帧数一致，示例是用了15188帧数据作为训练集。您用了XXX帧训练集，就把15188改成多少。

## 二、运行train.py进行单卡debug
1、将dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py的batch size改为1，显卡num_gpus用1张卡，方便debug调试。
2、在launch.json里的"name": "Distributed Training",检查所有参数，检查launch.json的"args"字典里面的路径对不对。
单卡batch_size是10，纯检测的版本显存是22个G

## 三、用tmux命令进行分布式训练
1、将dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py的batch size改为显卡支持的最大bs，同时注意学习率线性缩放。
2、点击tmux.txt，分为两种方式：一种是重头开始训练，另一种是带resume的训练。
命令含义如下：
cd 进入工程文件夹
source 环境路径
更改script/train.py和config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py为自己的路径
先创建train_log目录，并将重定向的日志0106_r50_bs8.log改为自定义的日志名称。

# 浮点测试
## 一、模型推理test可视化
1、在launch.json里的"name": "sparse4de2e_visual_new",更改program的路径，更改launch.json的"args"字典里面的路径。
2、点击tool/visualization/visulaize_new.py，确认第70行的samples_per_gpu = cfg["data"]["val"].pop("samples_per_gpu", 1)是val，因为示例只划分了train和val。（如果您划分了test，可以写cfg["data"]["tset"]。）
3、更改第143行save_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/visualization_0104_pred_0.5_epoch83"为您的保存路径。
4、运行launch.json里的tool/visualization/visulaize_new.py
（注：模型test用的得分阈值是0.5，如果要修改得分阈值，在modules/head/decoder/decoder.py第15行，修改score_threshold得分阈值）
参考命令：CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python tool/visualization/visulaize_new.py \
--config dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--checkpoint e2e_worklog_0207_all_camera_config_no_quality/sparse4dv3_temporal_r50_1x8_bs6_256x704/iter_57150.pth


## 附：代码主要更改说明（详见飞书文档）：
1、config的更改
2、NuScenes4DDetTrackDataset类的修改，路径在dataset/nuscenes_dataset.py
3、鱼眼深度图 Class MultiScaleDepthMapGeneratorFisheye，路径在dataset/pipeline/nuscenes/nusc_pipeline.py
4、特征采样的投影函数，路径在modules/head/sparse4d_blocks/core_blocks.py


# J6部署-服务器端
## 一、calib阶段
运行下面的命令会进行run_calib_phase并得到导出的HBM模型。
需替换实际的config与base-checkpoint的路径，以及日志的名字。参考命令：
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
--no-validate \
--launcher none \
2>&1 | tee sparse4d_20261024calib_fisheye.log

## 二、测试calib阶段保存的权重
运行下面的命令会测试calib阶段保存的权重,得到测试集的精度结果，作为定点模型的结果与浮点模型的结果进行对比。
需替换实际的config、base-checkpoint、calib权重（ckpt）的路径，以及日志的名字。参考命令：
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
--eval bbox NDS mAP\
2>&1 | tee eval_data3k测试calib1024_fisheye_fix.log

## 三、calib测试可视化
与test_calib.py逻辑一致，visulaize_new_calib.py此脚本是得到可视化图片。
需替换实际的save_dir、checkpoint的路径。参考命令：
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python3 tool/visualization/visulaize_new_calib.py \
--config dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--save_dir /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/vis_pred/1028 \
--checkpoint /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/Sparse4D_QAT/1024_fisheye_fix.ckpt \
--base-checkpoint /home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/ckpt/iter_36800.pth


# J6部署-板端
在服务器里进行交叉编译，以生成板端能运行的产物。
## 1. 服务器设置路径
确保本地最新的 all-round 已上传到服务器，然后执行：
export SDK_ROOT=/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/horizon_j6_open_explorer_v3.8.1-py310_20260326
export SAMPLE_ROOT="$SDK_ROOT/samples/ucp_tutorial/all-round"

cd "$SAMPLE_ROOT"

## 2. 检查交叉编译器
下面是参考工程使用的默认路径。如果实际安装位置不同，只修改这一行：
export LINARO_GCC_ROOT=/arm-gnu-toolchain-12.2.rel1-x86_64-aarch64-none-linux-gnu

"$LINARO_GCC_ROOT/bin/aarch64-none-linux-gnu-g++" --version
"$LINARO_GCC_ROOT/bin/aarch64-none-linux-gnu-g++" -dumpmachine
第二条应输出：
aarch64-none-linux-gnu
如果文件不存在，查找实际位置：
find /opt /usr/local /home /data1 / \
  -maxdepth 5 -name aarch64-none-linux-gnu-g++ 2>/dev/null

## 3. 检查依赖并编译
ls "$SDK_ROOT/samples/ucp_tutorial/deps_aarch64/ucp/lib"
ls "$SDK_ROOT/samples/ucp_tutorial/deps_aarch64/opencv/lib"
ls "$SDK_ROOT/samples/ucp_tutorial/deps_aarch64/gflags/lib"

cd "$SAMPLE_ROOT"
bash build_sparse4d_aarch64.sh
编译、安装完成后检查：
file deploy/sparse4d/aarch64/bin/sparse4d_J6M
ls -lh deploy/sparse4d/aarch64/bin/
ls -lh deploy/sparse4d/aarch64/lib/
file 输出应包含 ARM aarch64，不能是 x86-64。

这个构建使用独立的 build_sparse4d_aarch64 目录，不影响已经编译的 x86 程序。

## 4. 在服务器整理完整部署包
直接在安装目录补上模型和数据：
cd "$SAMPLE_ROOT"
export PACKAGE_ROOT="$SAMPLE_ROOT/deploy/sparse4d/aarch64"

mkdir -p "$PACKAGE_ROOT/model"
mkdir -p "$PACKAGE_ROOT/data"
mkdir -p "$PACKAGE_ROOT/configs"

cp "$SDK_ROOT/samples/ucp_tutorial/dnn/basic_samples/code/sparse4d_J6M/assets/model.hbm" \
   "$PACKAGE_ROOT/model/model.hbm"

cp -a "$SAMPLE_ROOT/data/sparse4d/frame_000" \
      "$PACKAGE_ROOT/data/"

cp "$SAMPLE_ROOT/configs/sparse4d/model_io.json" \
   "$PACKAGE_ROOT/configs/model_io.json"
部署包应包含：
aarch64/
├── bin/
│   ├── sparse4d_J6M
│   └── run_sparse4d.sh
├── lib/                         # 构建安装的 aarch64 运行库
├── model/
│   └── model.hbm
├── configs/
│   └── model_io.json
└── data/
    └── frame_000/
        ├── frame.json           # 包含内外参、畸变和预处理配置
        ├── images/              # 同一帧的四张图片
        ├── reference_inputs.json
        └── reference_inputs/

## 5. 将部署包拷贝到域控制器运行
使用MobaXterm软件连接域控制器和PC电脑，将PC电脑里的部署包（aarch64）拖动复制到域控制器目标目录。
板端先执行前处理
cd /app/aarch64

bash bin/run_sparse4d.sh \
  --manifest="$PWD/data/frame_000/frame.json" \
  --dump_dir="$PWD/output_preprocess" \
  --preprocess_only=true
这一步读取四路图片并生成七个输入，不执行 HBM。正常结束后，应存在：
ls -lh output_preprocess/inputs/
cat output_preprocess/timing.json

## 6. 板端执行单帧完整推理
cd /app/aarch64
set -o pipefail

bash bin/run_sparse4d.sh \
  --model_file="$PWD/model/model.hbm" \
  --model_name=forward \
  --model_io="$PWD/configs/model_io.json" \
  --manifest="$PWD/data/frame_000/frame.json" \
  --dump_dir="$PWD/output_j6m" \
  --warmup=0 \
  --repeat=1 \
  --core_id=0 \
  --save_outputs=true \
  --dump_inputs=true \
  --visualize=true \
  --visualize_score_threshold=0.5 \
  --trace=true \
  2>&1 | tee run_j6m.log
这里执行一次推理，没有预热。启动脚本设置 nash-m 和包内运行库路径。

成功时，应看到 RunInfer RETURN rc=0，并生成：
output_j6m/
├── hbm_interface.json
├── raw_outputs.json
├── output_0.bin
├── output_1.bin
├── classification_f32.bin
├── prediction_f32.bin
├── detections.json
├── four_cameras.jpg
├── bev.jpg
└── timing.json

# 更换域控制器（J6M换成J6P）
