# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import pyquaternion
import torch
import argparse

from typing import Optional, Dict, Any

from tool.trainer.utils import set_random_seed
from tool.utils.config import read_cfg
from tool.utils.dist_utils import init_dist, get_dist_info
from tool.utils.distributed import E2EDistributedDataParallel
from tool.utils.data_parallel import E2EDataParallel
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint

from dataset.dataloader_wrapper import *
from tool.tester.test_sdk import *

from dataset import *
from modules.sparse4d_detector import *
import matplotlib.pyplot as plt
import cv2
from tool.visualization.utils import draw_lidar_bbox3d,draw_lidar_bbox3d_on_bev, draw_lidar_bbox3d_on_img, draw_lidar_bbox3d_fisheye
import os
import numpy as np
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Train E2E detector")
    parser.add_argument("--config", help="train config file path", default="/data6/tangchengchen/SparseEnd2End_BEVOD/dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py")
    parser.add_argument(
        "--checkpoint", default="e2e_worklog/sparse4d_temporal_r50_1x1_bs1_256x704_mini/iter_66000.pth"
    )
    parser.add_argument(
        "--dataset", default="nuscenes", choices=["nuscenes", "custom"], help="dataset type"
    )
    parser.add_argument(
        "--save_dir", default="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/vis_pred/1027", help="保存可视化图片的路径"
    )
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default="bbox",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox"',
    )

    return parser.parse_args()

def main():
    args = parse_args()
    cfg = read_cfg(args.config)  # dict
    cfg["model"]["img_backbone"]["init_cfg"] = {}

    ## Init distributed env first, since logger depends on the dist info.
    # 默认不使用分布式训练
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    ## Set Seed
    set_random_seed(cfg.get("seed", 0), deterministic=args.deterministic)

    ## Build Dataloader 构建数据加载器
    samples_per_gpu = cfg["data"]["val"].pop("samples_per_gpu", 1)
    # 通过调用 build_module 创建测试数据集模块，cfg["data"]["test"] 是配置中关于测试集的部分，通常包括数据集的类型、路径等信息。
    if distributed:
        if cfg.dataset == 'nuscenes':
            dataset = build_module(cfg["data"]["val"])
            data_loader = dataloader_wrapper(
                dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=cfg["data"]["workers_per_gpu"],
                dist=distributed,
                shuffle=False,
                nonshuffler_sampler=dict(type="DistributedSampler"),
            )
        else:
            data_loader = dataloader_wrapper(
                dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=cfg["data"]["workers_per_gpu"],
                dist=distributed,
                shuffle=False,
                nonshuffler_sampler=dict(type="DistributedSampler"),
        )
    else: # 无分布式训练
        if args.dataset == 'nuscenes':
            dataset = build_module(cfg["data"]["val"])
            data_loader = dataloader_wrapper_without_dist(
                dataset,
                # 指定每个 GPU 上加载多少样本
                samples_per_gpu=samples_per_gpu,
                # 指定每个 GPU 上有多少个数据加载工作线程
                workers_per_gpu=cfg["data"]["workers_per_gpu"],
                dist=False,
                # 是否打乱数据
                shuffle=False,
        )
        else:
            data_loader = dataloader_wrapper_without_dist(
                dataset,
                # 指定每个 GPU 上加载多少样本
                samples_per_gpu=samples_per_gpu,
                # 指定每个 GPU 上有多少个数据加载工作线程
                workers_per_gpu=cfg["data"]["workers_per_gpu"],
                dist=False,
                # 是否打乱数据
                shuffle=False,
        )
    ## Build Model
    model = build_module(cfg["model"])
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    # 1. 加载前打印
    # print("加载前 backbone 第一个权重均值:", model.img_backbone.conv1.weight.mean().item())

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    #print(checkpoint["meta"]["CLASSES"])
    # 3. 加载后打印
    # print("加载后 backbone 第一个权重均值:", model.img_backbone.conv1.weight.mean().item())

    ## GPU Inference
    if not distributed:
        model = E2EDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = E2EDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = custom_multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
        )
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print("save_dir:",save_dir)
    num_samples = len(dataset)
    for index in range(num_samples):
        input_dict = dataset.get_data_info(index)  # 获取字典
        anno_dict = dataset.get_ann_info(index)
        raw_imgs_path = input_dict["img_filename"]  # 提取图像路径
        rt = input_dict["lidar2img"]               # 提取变换矩阵
        intrinsic = input_dict["cam_intrinsic"]    # 提取内参
        extrinsic = input_dict["cam_extrinsic"]
        distortion = input_dict["cam_distortion"]
            
        #真值也画出来
        gt_bbox3d = anno_dict['gt_bboxes_3d']
        gt_label3d = anno_dict['gt_labels_3d']
        gt_bbox3d = torch.from_numpy(gt_bbox3d)
        gt_label3d = torch.from_numpy(gt_label3d)


        imgs = [cv2.imread(name,cv2.IMREAD_UNCHANGED) for name in raw_imgs_path]
        
        plt.figure(figsize=(20, 10))

        pred_bbox3d = outputs[index]['img_bbox']['boxes_3d']
        pred_label3d = outputs[index]['img_bbox']['labels_3d']
        # img = draw_lidar_bbox3d(
        #                 pred_bbox3d,
        #                 pred_label3d,
        #                 imgs, rt,
        #                 color=(0, 255, 0)
        #                 )
        img = draw_lidar_bbox3d_fisheye(
                    pred_bbox3d,
                    pred_label3d,
                    gt_bbox3d,
                    gt_label3d,
                    imgs, rt, intrinsic, extrinsic, distortion,
                    color=(0, 255, 0)
                    )
        save_path = os.path.join(save_dir, f"visual_{index:05d}.jpg")
        cv2.imwrite(save_path, img)
    pred_scores3d = outputs[index]['img_bbox']['scores_3d']
    print(pred_scores3d)
    return 0

#----------------------------------------------------------------------------------------------------------------------


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    # 传入的配置字典 cfg，避免对原始字典进行修改
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)    
    

if __name__ == "__main__":
    args = parse_args()
    main()