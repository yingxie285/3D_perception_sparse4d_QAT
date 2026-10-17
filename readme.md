一、检查百度原始数据集
1、真值可视化验证：验证原始数据集的3D框真值投影到图像上是不是对的，运行tool/visualization/vis_gt_label.py，运行前更改第167行dataset_dir数据集路径、第171行可视化图片保存路径save_dir。
2、确定百度数据集有几套内外参（对应几个数据采集车）：运行tool/data_converter/collect_unique_camera_config.py，可以知道有4套内外参，也就相当于4辆不同的数据采集车。因此，需要为每一帧每一个相机都用自己独自的内外参，后续的数据预处理脚本已实现这个功能。运行前，把第5行root_dir的路径换成您的路径（data/fisheye_data_2w_frame，这里面是按场景排列的数据）。

二、模型需要的数据预处理
1、运行sparse4d/6/offline_data_aug.py得到数据增强后的真值数据集。将百度数据集的点云从pcd格式转化成npy格式。此脚本会在原路径下生成新的npy格式的点云，不会覆盖原有pcd格式的点云。npy格式的点云数据IO的速度更快，且模型读取的是npy格式的点云，因此要转换。

2、真值可视化验证：验证数据集的3D框真值投影到图像上是不是对的，运行sparse4d/6/vis_gt_label.py。只看前3个场景

3、运行sparse4d/6/bevod_split_dataset.py，会得到划分数据集的data_info.json。

4、运行sparse4d/6/bevod_to_nuscenes.py，会在data/nuscenes_format生成Nuscenes形式的json。（注意：最好不要点击生成的data/nuscenes_format/v1.0-trainval/sample_annotation.json，因为真值标签很多，这个json比较大，vs code可能会卡住。您可以用命令ls -lh sample_annotation.json的路径来看sample_annotation.json的内存是多大）

5、运行sparse4d/6/BEVOD_dump_nusc_ann_json_mini.py，将Nuscenes形式转成dump形式，模型需要dump形式。
运行前需要在BEVOD_dump_nusc_ann_json_mini.py的create_nuscenes_infos函数里，点击train_scenes = splits.bevod_train中的bevod_train和
val_scenes = splits.bevod_val中的bevod_val。
进入dataset/utils/custom_nuscenes/utils/splits.py更改bevod_train和bevod_val，确认bevod_train和bevod_val的场景划分，保持和train_val_split.json一致（其实就是把train_val_split.json的[]里面的内容对应复制到dataset/utils/custom_nuscenes/utils/splits.py的bevod_train和bevod_val的全局变量定义，覆盖之前的bevod_train和bevod_val的内容）其中，bevod_train就是train_val_split.json里面的"train"，bevod_val就是train_val_split.json里面的"val"，更改后按ctrl+S保存。

6、运行sparse4d/6/nusc_anchor_generator.py，得到初始化的anchor（bevod_kmeans900.npy）。

三、检查config[您训练要改的，config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py搜TODO确认更改]
点击dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
1、确认模型参数和路径是不是对的。尤其是ann_file和data_root的路径
2、更改num_iters_per_epoch = int(15188 // (num_gpus * batch_size))  #训练集帧数，和您划分的训练集的帧数一致，示例是用了15188帧数据作为训练集。您用了XXX帧训练集，就把15188改成多少。

四、运行train.py进行debug
在launch.json里的"name": "Distributed Training",检查所有参数，检查launch.json的"args"字典里面的路径对不对。
单卡batch_size是10，纯检测的版本显存是22个G


五、用tmux命令进行分布式训练
点击tmux.txt，分为两种方式：一种是重头开始训练，另一种是带resume的训练。
命令含义如下：
cd 进入工程文件夹
source 环境路径
更改script/train.py和config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py为自己的路径
先创建train_log目录，并将重定向的日志0106_r50_bs8.log改为自定义的日志名称。

六、模型推理test可视化
1、在launch.json里的"name": "sparse4de2e_visual_new",更改program的路径，更改launch.json的"args"字典里面的路径。
2、点击tool/visualization/visulaize_new.py，确认第70行的samples_per_gpu = cfg["data"]["val"].pop("samples_per_gpu", 1)是val，因为示例只划分了train和val。（如果您划分了test，可以写cfg["data"]["tset"]。）
3、更改第143行save_dir = "/media/whut-user/350b5acb-7342-4f97-b1a7-57e9622bff8b/xy/Sparse4D_no_tracking/visualization_0104_pred_0.3_epoch83"为您的保存路径。
4、运行launch.json里的tool/visualization/visulaize_new.py
（注：模型test用的得分阈值是0.5，如果要修改得分阈值，在modules/head/decoder/decoder.py第15行，修改score_threshold得分阈值）
参考命令：CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python tool/visualization/visulaize_new.py \
--config dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py \
--checkpoint e2e_worklog_0207_all_camera_config_no_quality/sparse4dv3_temporal_r50_1x8_bs6_256x704/iter_57150.pth


七、附：主要更改说明（详见飞书文档）：
1、config的更改
2、NuScenes4DDetTrackDataset类的修改，路径在dataset/nuscenes_dataset.py
3、鱼眼深度图 Class MultiScaleDepthMapGeneratorFisheye，路径在dataset/pipeline/nuscenes/nusc_pipeline.py
4、特征采样的投影函数，路径在modules/head/sparse4d_blocks/core_blocks.py
