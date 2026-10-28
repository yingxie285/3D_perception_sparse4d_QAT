# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
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
from horizon_plugin_pytorch.march import set_march
from dataset.dataloader_wrapper import *
from tool.tester.test_sdk import *
from horizon_plugin_pytorch.quantization import FakeQuantState, set_fake_quantize
from dataset import *
from modules.sparse4d_detector import *
from tool.utils.logging import ProgressBar
from horizon_plugin_pytorch.quantization import (
    prepare,
    set_fake_quantize,
    FakeQuantState,
)
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    QconfigSetter,
    ModuleNameTemplate,
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
)
import horizon_plugin_pytorch.quantization.observer as observer
from horizon_plugin_pytorch.quantization.fake_quantize import FakeQuantize
import horizon_plugin_pytorch.quantization.observer_v2 as observer_v2
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig
from horizon_plugin_pytorch.quantization.hbdk4 import export
from hbdk4.compiler import compile, convert, hbm_perf, save

# SparseEnd2End native imports, replacing only the MMDet-specific model/data
# construction used by the original SparseDrive QAT script.
from tool.trainer.utils import set_random_seed
from tool.utils.config import read_cfg
from tool.utils.dist_utils import init_dist, get_dist_info
from tool.utils.data_parallel import E2EDataParallel
from tool.utils.distributed import E2EDistributedDataParallel
from tool.runner.fp16_utils import wrap_fp16_model
from tool.runner.checkpoint import load_checkpoint

from dataset.dataloader_wrapper import *
from dataset import *
from modules.sparse4d_detector import *
from dataset.dataloader_wrapper.dataloader_wrapper import build_module # 0928 dws
from tool.utils.logging import ProgressBar
from pathlib import Path

from script.qat_utils_sparse4d import (
    prepare_example_input,
    replace_forward_with_horizon,
    build_test_dataloader,
)
import os
import sys
# 项目根目录
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

# script 目录
SCRIPT_DIR = os.path.join(PROJECT_ROOT, "script")

# 添加模块搜索路径
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from script.qat_workflow_sparse4d import get_qconfig_setter, _load_torch_file
import cv2
from tool.visualization.utils import draw_lidar_bbox3d_fisheye
import matplotlib.pyplot as plt

PREFIEX = os.environ.get("QAT_OUTPUT_PREFIX", "1024_fisheye_fix")
WORK_DIR = "./Sparse4D_QAT/"
EXAMPLE_INPUT_PT = WORK_DIR + PREFIEX + "/_example_input.pt"

def parse_args():
    parser = argparse.ArgumentParser(description="Train E2E detector")
    parser.add_argument("--config", help="train config file path", default="/data6/tangchengchen/SparseEnd2End_BEVOD/dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py")
    parser.add_argument(
        "--checkpoint", default="e2e_worklog/sparse4d_temporal_r50_1x1_bs1_256x704_mini/iter_66000.pth"
    )
    parser.add_argument("--base-checkpoint", default="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/ckpt/iter_36800.pth",
                        help="浮点FP32 checkpoint")
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

def replace_forward_with_horizon(module):
    if hasattr(module, "horizon_forward") and callable(module.horizon_forward):
        module.forward = module.horizon_forward

    for child in module.children():
        replace_forward_with_horizon(child)

def _build_deploy_model(cfg, base_checkpoint):
    """Build FP32 model, load checkpoint, replace forward with horizon_forward."""
    model = build_module(cfg["model"])

    if base_checkpoint:
        load_checkpoint(model, base_checkpoint, map_location="cpu")

    model = model.cuda()

    set_march("nash-m")  #xy1012 用nash-m，我们的域控是J6M

    model.eval()
    replace_forward_with_horizon(model)  #这里把浮点的forward变成了horizon_forward

    return model

def single_gpu_test(model, data_loader):
    model.eval() 
    set_fake_quantize(model, FakeQuantState.VALIDATION)  
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, "PALETTE", None)
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

@torch.no_grad()
def prepare_calib_model(inner,example_input,  check_result_dir=None):
    print("准备calib-model ")

    qconfig_setter = get_qconfig_setter(inner)
    calib_model = prepare(inner, example_input, qconfig_setter,
                          check_result_dir=check_result_dir or WORK_DIR+PREFIEX+"/") #可在产物路径中找到 model_check_result.txt 和 fx_graph.txt 用于分析。
    calib_model.eval()
    return calib_model

def _move_to_device(value, device):
    """Move tensors in the nested example-input structure to ``device``."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


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
    # model = build_module(cfg["model"])
    deploy_model = _build_deploy_model(cfg,args.base_checkpoint)
    # 1. 加载之前保存的单帧示例输入
    example_input_path = EXAMPLE_INPUT_PT
    print("example_input_path:",example_input_path)

    example_input = torch.load(
        example_input_path,
        map_location="cpu",
        weights_only=False
    )
    # 获取模型所在设备
    device = next(deploy_model.parameters()).device
    print("deploy_model device:", device)

    # 将示例输入移动到相同设备
    example_input = _move_to_device(example_input, device)

    calib_model = prepare_calib_model(deploy_model, example_input) #传入示例输入，prepare_calib_model里面会进行prepare
    
    # fp16_cfg = cfg.get("fp16", None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)

    # This revised workflow saves a raw, unwrapped state_dict.
    # Strict loading deliberately rejects old graphs and mismatched quantizers.
    checkpoint = _load_torch_file(args.checkpoint, map_location="cpu")
    calib_model.load_state_dict(checkpoint, strict=True)
    # print(checkpoint["meta"]["CLASSES"])
    print("args.checkpoint:",args.checkpoint)

    ## GPU Inference
    if not distributed:
        model = E2EDataParallel(calib_model, device_ids=[0])
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