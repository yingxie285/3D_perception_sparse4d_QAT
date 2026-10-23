# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.

import glob
import os
import os.path as osp
import shutil
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from dataset.utils.data_container import DataContainer

from horizon_plugin_pytorch.quantization import (
    prepare,
    set_fake_quantize,
    FakeQuantState,
)

from tool.utils.dist_utils import get_dist_info
from dataset.dataloader_wrapper import (
    dataloader_wrapper,
    dataloader_wrapper_without_dist,
)

from dataset import *
from modules.sparse4d_detector import *
from dataset.dataloader_wrapper.dataloader_wrapper import build_module # 0928 dws
from tool.utils.data_parallel import E2EDataParallel
from tool.utils.logging import ProgressBar

# class QATDDP(MMDistributedDataParallel):
#     """
#     SparseDrive QAT's special DDP is kept for prev_id only.

#     Sparse4D perception still has temporal detection state:
#         det_cached_feature
#         det_cached_anchor
#         det_cache_confidence
#         det_cache_instance_id
#         prev_id

#     Map / motion / planning state is not used.
#     """

#     _BROADCAST_KWARGS = {"prev_id"}

#     def scatter(self, inputs, kwargs, device_ids):
#         saved = {}
#         for key in self._BROADCAST_KWARGS:
#             if key in kwargs:
#                 saved[key] = kwargs.pop(key)

#         inputs_part, kwargs_part = super().scatter(
#             inputs,
#             kwargs,
#             device_ids,
#         )

#         if saved:
#             for kw in kwargs_part:
#                 for key, val in saved.items():
#                     kw[key] = (
#                         val.clone()
#                         if torch.is_tensor(val)
#                         else val
#                     )

#         return inputs_part, kwargs_part

#     def to_kwargs(self, inputs, kwargs, device_id):
#         saved = {}
#         for key in self._BROADCAST_KWARGS:
#             if key in kwargs:
#                 saved[key] = kwargs.pop(key)

#         inputs_part, kwargs_part = super().to_kwargs(
#             inputs,
#             kwargs,
#             device_id,
#         )

#         if saved:
#             for kw in kwargs_part:
#                 for key, val in saved.items():
#                     kw[key] = (
#                         val.clone()
#                         if torch.is_tensor(val)
#                         else val
#                     )

#         return inputs_part, kwargs_part


def _extract_from_dc(obj, _depth=0):
    if torch.is_tensor(obj):
        return obj.cuda() if not obj.is_cuda else obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float().cuda()
    if isinstance(obj, DataContainer):
        data = obj.data
        if isinstance(data, list) and len(data) > 0:
            return _extract_from_dc(data[0], _depth=_depth+1)
        return _extract_from_dc(data, _depth=_depth+1)
    if isinstance(obj, dict):
        return {k: _extract_from_dc(v, _depth=_depth+1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_extract_from_dc(v, _depth=_depth+1) for v in obj)
    return obj


def _get_det_cache(det_output, key):
    if det_output is None:
        return None

    if not isinstance(det_output, dict):
        raise TypeError(
            "Sparse4D horizon_forward detection output must be a dict."
        )

    return det_output.get(key)

#------------------构造示例输入-------------
@torch.no_grad()
def prepare_example_input(dataloader):
    prev_timestamp = None

    for i, data in enumerate(dataloader):
        data = _extract_from_dc(data)

        img = data["img"].cuda()
        timestamp = data["timestamp"].cuda()
        cam_extrinsic = data["cam_extrinsic"].cuda()
        cam_intrinsic = data["cam_intrinsic"].cuda()
        cam_distortion = data["cam_distortion"].cuda()
        lidar2img = data["lidar2img"].cuda()
        image_wh = data["image_wh"].cuda()

        if i == 0:
            time_interval = torch.tensor(1.0, dtype=torch.float32, device="cuda") #不做时序，分母是1，不影响

        example_input = (
            img,
            cam_extrinsic,
            cam_intrinsic,
            cam_distortion,
            lidar2img,
            image_wh,
            time_interval
        )    #和modules/sparse4d_detector.py的horizon_forward保持一致

        if i >= 1:
            break
    names = (
        "img",
        "cam_extrinsic",
        "cam_intrinsic",
        "cam_distortion",
        "lidar2img",
        "image_wh",
        "time_interval"
    )
    print("[prepare_example_input] Detection-only Sparse4D input ready.")
    for idx, (name, value) in enumerate(zip(names, example_input)):
        print(
            f"  [{idx:02d}] {name}: "
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, "
            f"device={value.device}"
        )

    return example_input
#--------------------------------------------



def replace_forward_with_horizon(module):
    if hasattr(module, "horizon_forward") and callable(module.horizon_forward):
        module.forward = module.horizon_forward

    for child in module.children():
        replace_forward_with_horizon(child)

def _detach_tensor(value):
    if value is None:
        return None

    if torch.is_tensor(value):
        return value.detach()

    return value

#-----------------------------------------------0929----------------------------
# def _initial_det_state(example_input):
#     return (
#         _detach_tensor(example_input[5]),
#         _detach_tensor(example_input[6]),
#         _detach_tensor(example_input[7]),
#         _detach_tensor(example_input[8]),
#         _detach_tensor(example_input[9]),
#     )
def _initial_det_state(example_input):
    # return (
    #     _detach_tensor(example_input[6]),   # cached_feature
    #     _detach_tensor(example_input[7]),   # cached_anchor
    #     _detach_tensor(example_input[8]),   # cache_confidence
    #     _detach_tensor(example_input[9]),   # cache_track_id
    #     _detach_tensor(example_input[10]),  # prev_id
    # )
    return (
        None,  # cached_feature
        None,  # cached_anchor
        None,  # cache_confidence
        None,  # cache_track_id
        None,  # prev_id
    )
#-------------------------------------------------------------------------------

def _reset_state(example_input):
    return _initial_det_state(
        example_input
    )

def update_det_cache(
    det_output,
    prev_id,
    distributed=False,
):

    cached_feature = _detach_tensor(
        _get_det_cache(
            det_output,
            "cached_feature",
        )
    )

    cached_anchor = _detach_tensor(
        _get_det_cache(
            det_output,
            "cached_anchor",
        )
    )

    cache_confidence = _detach_tensor(
        _get_det_cache(
            det_output,
            "cache_confidence",
        )
    )

    cache_track_id = _detach_tensor(
        _get_det_cache(
            det_output,
            "cache_track_id",
        )
    )

    new_prev_id = _get_det_cache(
        det_output,
        "prev_id",
    )

    if new_prev_id is not None:

        new_prev_id = _detach_tensor(
            new_prev_id
        )

        if (
            distributed
            and dist.is_initialized()
        ):
            dist.all_reduce(
                new_prev_id,
                op=dist.ReduceOp.MAX,
            )

        prev_id = new_prev_id

    return (
        cached_feature,
        cached_anchor,
        cache_confidence,
        cache_track_id,
        prev_id,
    )
#-------------------------------------------------------------------------------


def compute_qat_loss(
    calib_model,
    det_output,
    data,
):
    model = (
        calib_model.module
        if hasattr(calib_model, "module")
        else calib_model
    )

    losses = model.head.loss(
        det_output,
        data,
    )

    if isinstance(det_output, dict) and "loss_dense_depth" in det_output:
        losses["loss_dense_depth"] = det_output["loss_dense_depth"]

    return losses


def build_test_dataloader(
    cfg,
    distributed=False,
):
    samples_per_gpu = cfg["data"]["val"].pop("samples_per_gpu", 1)

    dataset = build_module(cfg["data"]["val"])

    if distributed:
        data_loader = dataloader_wrapper(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg["data"]["workers_per_gpu"],
            dist=True,
            shuffle=False,
            nonshuffler_sampler=dict(
                type="DistributedSampler"
            ),
        )
    else:
        data_loader = dataloader_wrapper_without_dist(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg["data"]["workers_per_gpu"],
            dist=False,
            shuffle=False,
        )

    return data_loader


def build_train_dataloader(
    cfg,
    distributed=False,
):
    train_dataset = build_module(
        cfg["data"]["train"]
    )

    return dataloader_wrapper(
        train_dataset,
        cfg["data"]["samples_per_gpu"],
        cfg["data"]["workers_per_gpu"],
        seed=cfg.get("seed", 0),
        runner_type=cfg["runner"]["type"],
        dist=distributed,
        num_gpus=len(cfg["gpu_ids"]),
        nonshuffler_sampler=dict(
            type="DistributedSampler"
        ),
    )


def build_qat_optimizer(
    calib_model,
    qat_lr,
    weight_decay=0.001,
):
    backbone_params = []
    other_params = []

    for name, param in calib_model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            "img_backbone" in name
            or "img_neck" in name
        ):
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            "lr": qat_lr * 0.1,
        },
        {
            "params": other_params,
            "lr": qat_lr,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )

    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    return optimizer


class QATLRScheduler:
    """
    Cosine annealing with linear warmup.
    """

    def __init__(
        self,
        optimizer,
        warmup_iters,
        total_iters,
        min_lr_ratio=1e-3,
    ):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr_ratio = min_lr_ratio

    def step(self, iteration):
        if iteration < self.warmup_iters:
            scale = (
                iteration + 1
            ) / self.warmup_iters
        else:
            progress = (
                iteration
                - self.warmup_iters
            ) / max(
                1,
                self.total_iters
                - self.warmup_iters,
            )

            scale = (
                self.min_lr_ratio
                + 0.5
                * (
                    1
                    - self.min_lr_ratio
                )
                * (
                    1
                    + np.cos(
                        np.pi
                        * progress
                    )
                )
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = (
                param_group["initial_lr"]
                * scale
            )

    def get_lr(self):
        return [
            pg["lr"]
            for pg in self.optimizer.param_groups
        ]


def save_qat_checkpoint(
    calib_model,
    iteration,
    work_dir,
):
    model = (
        calib_model.module
        if hasattr(
            calib_model,
            "module",
        )
        else calib_model
    )

    ckpt_dir = osp.join(
        work_dir,
        "qat_checkpoints",
    )

    os.makedirs(
        ckpt_dir,
        exist_ok=True,
    )

    path = osp.join(
        ckpt_dir,
        f"qat_iter_{iteration}.ckpt",
    )

    torch.save(
        model.state_dict(),
        path,
    )

    print(
        f"[save] -> {path}"
    )

    return path

def single_gpu_test(model, data_loader):
    model.eval()
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


#xy1013
@torch.no_grad()
def horizon_evaluate(calib_model,data_loader,cfg):
    calib_model.eval()
    dataset = data_loader.dataset
    from tool.utils.logging import ProgressBar

    # set_fake_quantize(calib_model, FakeQuantState.VALIDATION)  #TODO先关掉

    inner = (
        calib_model.module
        if hasattr(calib_model, "module")
        else calib_model
    )

    outputs = []
    prev_timestamp = None
    prog_bar = ProgressBar(len(dataset))

    # for data in data_loader:
    #     data = _extract_from_dc(data)

    #     img = data["img"]
    #     if isinstance(img, list):
    #         assert len(img) == 1, f"Unexpected img list length: {len(img)}"
    #         img = img[0]
    #     img = img.cuda().float()

    #     cam_extrinsic = data["cam_extrinsic"]
    #     if isinstance(cam_extrinsic, list):
    #         assert len(cam_extrinsic) == 1, f"Unexpected cam_extrinsic list length: {len(cam_extrinsic)}"
    #         cam_extrinsic = cam_extrinsic[0]
    #     cam_extrinsic = cam_extrinsic.cuda().float()

    #     cam_intrinsic = data["cam_intrinsic"]
    #     if isinstance(cam_intrinsic, list):
    #         assert len(cam_intrinsic) == 1, f"Unexpected cam_intrinsic list length: {len(cam_intrinsic)}"
    #         cam_intrinsic = cam_intrinsic[0]
    #     cam_intrinsic = cam_intrinsic.cuda().float()

    #     cam_distortion = data["cam_distortion"]
    #     if isinstance(cam_distortion, list):
    #         assert len(cam_distortion) == 1, f"Unexpected cam_distortion list length: {len(cam_distortion)}"
    #         cam_distortion = cam_distortion[0]
    #     cam_distortion = cam_distortion.cuda().float()
        
    #     lidar2img = data["lidar2img"]
    #     lidar2img = lidar2img.cuda().float()

    #     image_wh = data["image_wh"]
    #     if isinstance(image_wh, list):
    #         assert len(image_wh) == 1, f"Unexpected image_wh list length: {len(image_wh)}"
    #         image_wh = image_wh[0]
    #     image_wh = image_wh.cuda().float()

    #     timestamp = data.get("timestamp", None)
    #     if timestamp is not None:
    #         if isinstance(timestamp, list):
    #             assert len(timestamp) == 1, f"Unexpected timestamp list length: {len(timestamp)}"
    #             timestamp = timestamp[0]
    #         timestamp = timestamp.cuda().float()

    #     if timestamp is None:
    #         if example_input is not None:
    #             time_interval = _tile_to_batch(example_input[5], img.shape[0])
    #         else:
    #             time_interval = torch.zeros(
    #                 img.shape[0],
    #                 dtype=torch.float32,
    #                 device=img.device,
    #             )
    #     elif prev_timestamp is None:
    #         time_interval = torch.zeros_like(timestamp, dtype=torch.float32)
    #     else:
    #         time_interval = (timestamp - prev_timestamp).to(torch.float32)

    #     det_output = calib_model(
    #         img=img,
    #         cam_extrinsic=cam_extrinsic,
    #         cam_intrinsic=cam_intrinsic,
    #         cam_distortion=cam_distortion,
    #         lidar2img=lidar2img,
    #         image_wh=image_wh,
    #         time_interval=time_interval,
    #     )

    #     results = inner.head.post_process(det_output)  #results[0]里面是'boxes_3d'、'scores_3d'、'labels_3d'

    #     outputs.extend(
    #         [
    #             {"img_bbox": result}
    #             for result in results
    #         ]
    #     )
    #     prog_bar.update(len(results))

    #     if timestamp is not None:
    #         prev_timestamp = timestamp

    #与test.py一致的测试
    model = E2EDataParallel(calib_model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)
    
    rank, _ = get_dist_info()
    if rank == 0:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        eval_kwargs.pop("interval", None)
        eval_kwargs["metric"] = "bbox"

        if eval_kwargs.get("jsonfile_prefix") is None:
            eval_kwargs["jsonfile_prefix"] = osp.join(
                "./e2e_worklog_test20261017没有量化xyuse_iter36800_deformable_score0.5_3739frame_bevod_val",
                osp.splitext(osp.basename(cfg.get("filename", "sparse4d")))[0],
                "eval",
            )

        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

    return outputs
#------------------------------------

def collect_results_cpu(
    result_part,
    size,
    tmpdir=None,
):
    rank, world_size = get_dist_info()

    if tmpdir is None:
        max_len = 512

        dir_tensor = torch.full(
            (max_len,),
            32,
            dtype=torch.uint8,
            device="cuda",
        )

        if rank == 0:
            os.makedirs(
                ".dist_test",
                exist_ok=True,
            )
            tmpdir = tempfile.mkdtemp(
                dir=".dist_test"
            )
            tmpdir_tensor = torch.tensor(
                bytearray(
                    tmpdir.encode()
                ),
                dtype=torch.uint8,
                device="cuda",
            )
            dir_tensor[
                :len(tmpdir_tensor)
            ] = tmpdir_tensor

        dist.broadcast(
            dir_tensor,
            0,
        )

        tmpdir = (
            dir_tensor.cpu()
            .numpy()
            .tobytes()
            .decode()
            .rstrip()
        )

    else:
        os.makedirs(
            tmpdir,
            exist_ok=True,
        )

    import pickle

    with open(
        osp.join(
            tmpdir,
            f"part_{rank}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            result_part,
            f,
        )

    dist.barrier()

    if rank != 0:
        return None

    ordered_results = []

    for part_file in sorted(
        glob.glob(
            osp.join(
                tmpdir,
                "part_*.pkl",
            )
        )
    ):
        with open(
            part_file,
            "rb",
        ) as f:
            ordered_results.extend(
                pickle.load(f)
            )

    ordered_results = (
        ordered_results[:size]
    )

    shutil.rmtree(
        tmpdir,
        ignore_errors=True,
    )

    return ordered_results


def is_main_process():
    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        return dist.get_rank() == 0

    return True
