# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import argparse
import os
import os.path as osp
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from tool.utils.logging import get_logger
from tool.utils.config import read_cfg, pretty_text
from tool.utils.env_collect import collect_env
from tool.trainer.train_sdk import train_api
from horizon_plugin_pytorch.march import set_march
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

from qat_utils_sparse4d import (
    QATDDP,
    _extract_from_dc,
    prepare_example_input,
    replace_forward_with_horizon,
    horizon_evaluate,
    build_qat_optimizer,
    QATLRScheduler,
    save_qat_checkpoint,
    compute_qat_loss,
    build_test_dataloader,
    build_train_dataloader,
    _tile_to_batch,
)


# Keep the original SparseDrive QAT path-variable style.
# 允许实验通过环境变量指定独立输出目录，避免覆盖已有 calib.ckpt。
PREFIEX = os.environ.get("QAT_OUTPUT_PREFIX", "1021_int16/calib")
WORK_DIR = "./Sparse4D_QAT/"
CALIBRATION_CKPT = WORK_DIR + PREFIEX + ".ckpt"
EXAMPLE_INPUT_PT = WORK_DIR + PREFIEX + "/_example_input.pt"
QAT_BC = WORK_DIR + PREFIEX + "/qat.bc"
QAT_ONNX = WORK_DIR + PREFIEX + "/qat.onnx"
QUANTIZE_BC = WORK_DIR + PREFIEX + "/quantize.bc"
QUANTIZE_ONNX = WORK_DIR + PREFIEX + "/quantize.onnx"
HBM_PATH = WORK_DIR + PREFIEX + "/model.hbm"
ADVICE_PATH = WORK_DIR + PREFIEX + "/advice.json"
output_dir = WORK_DIR + PREFIEX + "qat_export"

def update_scale_by_qtype(model: torch.nn.Module):
    for _, m in model.named_modules():
        if isinstance(m, observer_v2.HistogramObserver):
            m.sync_histogram()

    for _, m in model.named_modules():
        if (
            isinstance(m, FakeQuantize)
            and m.observer_enabled
            and not isinstance(
                m.activation_post_process,
                (observer.FixedScaleObserver, observer_v2.FixedScaleObserver),
            )
        ):
            if (
                hasattr(m.activation_post_process, "histogram_edges")
                and m.activation_post_process.histogram_edges.numel() == 0
            ):
                continue
            m.reset_dtype()


def get_qconfig_setter(model=None):
    """
    Keep the SparseDrive QAT qconfig strategy, but adapt the detection-head
    module names from:

        head.det_head.layers.*

    to Sparse4D:

        head.layers.*

    Special thresholds are attached dynamically to modules that actually exist
    in the current Sparse4D implementation, avoiding hard-coded decoder indices.
    """
    qconfig_setter = QconfigSetter(
        get_qconfig(observer=observer_v2.HistogramObserver,fix_scale=False),
        templates = [
            ModuleNameTemplate({"": torch.float16}),
            ConvDtypeTemplate(input_dtype=qint16, weight_dtype=qint8),
            MatmulDtypeTemplate(input_dtypes=qint16),
    #--------------------------------------1008--TODO----------------------------------
            ModuleNameTemplate(
                {

                }
            ),
            ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8, prefix=["img_backbone", "img_neck"]),
    #-----------------------------------------------------------------------------------
        ],
    )  #TODO #fix_scale 表示尺度固定，校准的时候需要设置为 false，QAT训练的时候设置为 true。

    return qconfig_setter 

#定义参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sparse4D Horizon QAT"
    )
    parser.add_argument(
        "--config",
        default="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/dataset/config/sparse4dv3_temporal_r50_1x8_bs6_256x704.py",
        help="config file path"
    )
    parser.add_argument(
        "--qat-checkpoint",
        default="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/e2e_worklog/sparse4dv3_temporal_r50_1x8_bs6_256x704/iter_1840.pth",
        help="待导出的 QAT checkpoint"
    )
    parser.add_argument("--base-checkpoint", default="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/ckpt/iter_36800.pth",
                        help="浮点FP32 checkpoint")
    parser.add_argument("--output-dir",type=Path,default=Path(output_dir),help="output result file")
    parser.add_argument("--expected-batch-size",type=int, default=1, help="部署图的batch size")
    parser.add_argument("--allow-nonstrict", action="store_true", help="允许QAT checkpoint存在missing/unexpected keys")
    parser.add_argument("--hbm-name", default="model.hbm", help="最终HBM文件名")
    parser.add_argument("--march", default="nash-m", help="目标BPU架构")
    parser.add_argument("--jobs", type=int, default=96, help="HBM编译线程数")
    parser.add_argument("--device", type=int, default=7, help="导出阶段使用的GPU编号")
    parser.add_argument("--opt", type=int, default=2, choices=[0, 1, 2], help="HBDK编译优化等级")
    parser.add_argument("--keep-io-ops", action="store_true", help="不移除模型输入输出处的Quantize/Dequantize等算子")
    parser.add_argument("--no-advice", action="store_true", help="convert阶段不生成量化建议报告")
    parser.add_argument("--debug", action="store_true", help="编译HBM时启用debug信息")
    parser.add_argument("--skip-onnx", action="store_true", help="跳过额外的QAT ONNX导出")
    # HBM板端性能评估。
    parser.add_argument("--eval-hbm", action="store_true", help="编译后使用hbm_perf评估板端性能")
    parser.add_argument("--remote-ip", default=None, help="J6P板端IP地址")
    parser.add_argument("--hbm-perf-dir", default=None, help="hbm_perf结果保存目录")
    # HBM完整验证集任务精度评估。具体HBRT/RPC实现由外部runner负责。
    parser.add_argument("--eval-hbm-accuracy", action="store_true", help="调用外部板端评估脚本计算L2/obj_col/obj_box_col")
    parser.add_argument("--hbm-accuracy-runner",default=None, help="板端任务精度评估脚本路径")
    parser.add_argument("--hbm-accuracy-dir", default=None, help="任务精度结果保存目录")
    parser.add_argument("--accuracy-timeout", type=int, default=0, help="任务精度评估超时秒数，0表示不限制")
    parser.add_argument("--accuracy-extra-args", nargs="*", default=None, help="附加给任务精度评估脚本的参数")
    
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default="bbox",
        help='evaluation metrics, e.g. "bbox"',
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch"],
        default="none",
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=0,
        dest="local_rank",
    )

    parser.add_argument(
        "--phase",
        choices=["calib", "train","QAT_test", "compile","all"],
        default=None,
    )
    parser.add_argument(
        "--qat-lr",
        type=float,
        default=6e-4,
    )
    parser.add_argument(
        "--qat-epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--qat-warmup-iters",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
    )

    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def _get_work_dir(cfg):
    return cfg.get(
        "work_dir",
        osp.join(
            "./e2e_worklog",
            osp.splitext(
                osp.basename(cfg.filename)
            )[0]
            if hasattr(cfg, "filename")
            else "sparse4d",
        ),
    )


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

def horizon_export(calib_model, example_input, args):
    qat_bc = osp.join(args.output_dir, "qat.bc")
    qat_onnx = osp.join(args.output_dir, "qat.onnx")
    quantize_bc = osp.join(args.output_dir, "quantize.bc")
    hbm_path = osp.join(args.output_dir, args.hbm_name)
    advice_path = osp.join(args.output_dir, "advice.json")

    set_fake_quantize(calib_model, FakeQuantState.VALIDATION)

    # ``prepare`` may wrap or rebuild modules and leave ``forward`` pointing
    # at the training/evaluation implementation.  Re-apply the deployment
    # forward transformation so both exporters trace the same stateless graph
    # with cache/queue tensors as explicit inputs and outputs.
    replace_forward_with_horizon(calib_model)

    calib_model._export_raw_predictions = True

    # Export the transformed PyTorch graph separately.  HBDK's ``save`` only
    # accepts its binary format, while this ONNX artifact is useful for graph
    # inspection and downstream ONNX tooling.
    if not args.skip_onnx:
        torch.onnx.export(
            calib_model,
            tuple(example_input),
            qat_onnx,
            opset_version=17,
            # Cached queue tensors are device-dependent; constant folding
            # tries to evaluate them on CPU and causes device mismatch.
            do_constant_folding=False,
            input_names=[f"input_{i:02d}" for i in range(len(example_input))],
            dynamo=False,
        )

    hbir_model = export(calib_model, example_input)
    save(hbir_model, qat_bc)

    convert_kwargs = dict(march=args.march, advice=not args.no_advice)
    if not args.no_advice:
        convert_kwargs["advice_path"] = advice_path
    quantized_hbir = convert(hbir_model, **convert_kwargs)
    if not args.keep_io_ops:
        quantized_hbir[0].remove_io_op(
            op_types=["Quantize", "Dequantize", "Cast", "Reshape"]
        )
    save(quantized_hbir, quantize_bc)
    
    compile(
        quantized_hbir,
        path=hbm_path,
        march=args.march,
        jobs=args.jobs,
        debug=args.debug,
        progress_bar=True,
        opt=args.opt,
    )

    if not osp.isfile(hbm_path):
        raise RuntimeError(f"HBDK compile returned without creating {hbm_path}")
    if args.eval_hbm:
        if not args.remote_ip:
            raise ValueError("--eval-hbm requires --remote-ip")
        perf_dir = args.hbm_perf_dir or osp.join(args.output_dir, "hbm_perf")
        hbm_perf(hbm_path, output_dir=perf_dir, remote_ip=args.remote_ip)
    return hbm_path


@torch.no_grad()
def prepare_calib_model(inner, check_result_dir=None):
    print("准备calib-model ")

    qconfig_setter = get_qconfig_setter(inner)
    calib_model = prepare(inner, qconfig_setter,
                          check_result_dir=check_result_dir or WORK_DIR+PREFIEX+"/")
    calib_model.eval()
    return calib_model

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



@torch.no_grad()
def horizon_calibration(calib_model, data_loader,cfg):
    calib_model.eval()

    set_fake_quantize(calib_model, FakeQuantState.CALIBRATION)
    dataset = data_loader.dataset
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
                "./e2e_worklog_test20261019calib阶段带节点xyuse_iter36800_deformable_score0.5_3739frame_bevod_val",
                osp.splitext(osp.basename(cfg.get("filename", "sparse4d")))[0],
                "eval",
            )

        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

    torch.save(model.state_dict(), CALIBRATION_CKPT)
#-------------------------------------------------------------------------------

@torch.no_grad()
def horizon_QAT_test(qat_model, data_loader,cfg):
    qat_model.eval()
    # QAT 训练完成后的测试评估
    dataset = data_loader.dataset
    #与test.py一致的测试
    model = E2EDataParallel(qat_model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader)
    
    rank, _ = get_dist_info()
    if rank == 0:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        eval_kwargs.pop("interval", None)
        eval_kwargs["metric"] = "bbox"

        if eval_kwargs.get("jsonfile_prefix") is None:
            eval_kwargs["jsonfile_prefix"] = osp.join(
                "./e2e_worklog_QAT_test20261020测试QAT的权重_iter1840_score0.5_3739frame_bevod_val",
                osp.splitext(osp.basename(cfg.get("filename", "sparse4d")))[0],
                "eval",
            )

        results_dict = dataset.evaluate(outputs, **eval_kwargs)
        print(results_dict)

#-----------------------------------------------1008----------------------------------------
@torch.no_grad()
def run_calib_phase(args, cfg):
    print("=" * 60)
    print("[Phase 1] prepare -> calib -> eval")
    print("=" * 60)

    data_loader = build_test_dataloader(cfg, distributed=False,)

    calib_model = _build_deploy_model(cfg,args.base_checkpoint)

    # example_input = prepare_example_input(deploy_model, data_loader)  #不做时序的话用不上example_input
    # torch.save(example_input, EXAMPLE_INPUT_PT)

    # calib_model = prepare_calib_model(deploy_model)

    calib_model.eval()

    horizon_calibration(calib_model, data_loader,cfg)   #这里不传时序相关的example_input，和test.py一样的逻辑

    print("[Calibration] checkpoint ->",CALIBRATION_CKPT)

    # horizon_evaluate(deploy_model, data_loader, cfg) #xy1012 calib阶段做eval测试
    torch.save(calib_model.state_dict(), CALIBRATION_CKPT)
    # Keep SparseDrive structure: calibration result can be exported directly.
    # horizon_export(calib_model, example_input) #TODO

    print("[Phase 1] Done.")


@torch.no_grad()
def run_QAT_test_phase(args, cfg):
    print("=" * 60)
    print("QAT训练完成，现在对QAT的权重进行测试")
    print("=" * 60)

    data_loader = build_test_dataloader(cfg, distributed=False,)

    qat_model = _build_deploy_model(cfg,args.qat_checkpoint)

    qat_model.eval()

    horizon_QAT_test(qat_model, data_loader,cfg)   #这里不传时序相关的example_input，和test.py一样的逻辑

    print("Done.")

def _load_torch_file(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _extract_qat_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if not checkpoint or not all(isinstance(key, str) for key in checkpoint):
        raise RuntimeError("No model state_dict found in QAT checkpoint")

    state_dict = OrderedDict()
    for key, value in checkpoint.items():
        state_dict[key[7:] if key.startswith("module.") else key] = value
    metadata = getattr(checkpoint, "_metadata", None)
    if metadata is not None:
        state_dict._metadata = OrderedDict(
            (key[7:] if key.startswith("module.") else key, value)
            for key, value in metadata.items()
        )
    return state_dict

#1021 不prepare，就可以正常导出hbm
# @torch.no_grad()
# def run_compile_phase(args, cfg):
#     print("=" * 60)
#     print("编译开始")
#     print("=" * 60)

#     os.makedirs(args.output_dir, exist_ok=True)
#     set_march(args.march)

#     base_checkpoint = args.base_checkpoint or cfg.get("load_from")
#     deploy_model = _build_deploy_model(cfg, base_checkpoint)  #用浮点32位构建的horizon_forward的模型
#     data_loader = build_test_dataloader(cfg, distributed=False)
#     example_input = prepare_example_input(deploy_model, data_loader)
    

#     checkpoint = _load_torch_file(args.qat_checkpoint, "cpu")  #把已经训练好的 QAT checkpoint 加载到准备好的模型中
    
#     incompatible = deploy_model.load_state_dict(
#         _extract_qat_state_dict(checkpoint), strict=False
#     )
#     missing = incompatible.missing_keys
#     unexpected = incompatible.unexpected_keys
#     print(f"[Export] state mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
#     if (missing or unexpected) and not args.allow_nonstrict:
#         raise RuntimeError(
#             "QAT checkpoint does not exactly match the prepared graph. "
#             "Use --allow-nonstrict only after reviewing the mismatch."
#         )

#     hbm_path = horizon_export(deploy_model, example_input, args)
#     print(f"[编译完成] HBM: {osp.abspath(hbm_path)}")


# @torch.no_grad()
def run_compile_phase(args, cfg):
    print("=" * 60)
    print("编译开始")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    set_march(args.march)

    base_checkpoint = args.base_checkpoint or cfg.get("load_from")
    deploy_model = _build_deploy_model(cfg, base_checkpoint)  #用浮点32位构建的horizon_forward的模型
    data_loader = build_test_dataloader(cfg, distributed=False)
    example_input = prepare_example_input(deploy_model, data_loader)
    
    #要不要用qconfig来准备量化？
    calib_model = prepare(
        deploy_model,
        example_input,
        get_qconfig_setter(deploy_model),
        check_result_dir=osp.join(args.output_dir, "prepare_check"),
    ).cuda().eval()
    checkpoint = _load_torch_file(args.qat_checkpoint, "cpu")  #把已经训练好的 QAT checkpoint 加载到准备好的模型中
    
    incompatible = calib_model.load_state_dict(
        _extract_qat_state_dict(checkpoint), strict=False
    )
    missing = incompatible.missing_keys
    unexpected = incompatible.unexpected_keys
    print(f"[Export] state mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    if (missing or unexpected) and not args.allow_nonstrict:
        raise RuntimeError(
            "QAT checkpoint does not exactly match the prepared graph. "
            "Use --allow-nonstrict only after reviewing the mismatch."
        )

    hbm_path = horizon_export(calib_model, example_input, args)
    print(f"[编译完成] HBM: {osp.abspath(hbm_path)}")



def run_train_phase(args, cfg):
    print("=" * 60)
    print("[Phase 2] Multi-GPU QAT training")
    print("=" * 60)

    ## Save Train LogAndCfg
    if cfg.get("work_dir", None) is None:
        cfg["work_dir"] = os.path.join(
            "./e2e_worklog_QATtrain1020_no_DAF", os.path.splitext(os.path.basename(args.config))[0]
        )
    os.makedirs(cfg["work_dir"], exist_ok=True)
    pretty_cfg = pretty_text(cfg)
    with open(
        os.path.join(cfg["work_dir"], os.path.basename(args.config)),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(pretty_cfg)  # save *.py config file

    cfg["gpu_ids"] = (
        range(1) if cfg.get("gpu_ids", None) is None else cfg["gpu_ids"]
    )

    ## Create Logger
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg["work_dir"], f"{timestamp}.log")
    logger = get_logger(
        name="E2E", log_file=log_file, log_level=cfg.get("log_level", "INFO")
    )
    
    ## Start Pytoch Multiprocess
    if args.launcher == "none":
        distributed = False
    elif args.launcher == "pytorch":
        distributed = True
        # 使用spawn启动pytorch多进程和打开进程间的通信, timeout 60 mins.
        init_dist(args.launcher, timeout=timedelta(seconds=3600), **cfg["dist_params"])
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg["gpu_ids"] = range(world_size)
    else:
        raise NotImplementedError(f"not support {args.launcher} launcher.")
    
    ## Meta Infos
    meta = dict()
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    meta["env_info"] = env_info
    meta["config"] = pretty_cfg
    meta["exp_name"] = os.path.basename(args.config)
    meta["seed"] = cfg.get("seed", 0)

    dash_line = "\n" + "*" * 100 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    logger.info(f"Distributed training: {distributed}")
    # logger.info(f"Config:\n{pretty_cfg}")
    logger.info(
        f"Set random seed to {cfg.get('seed', 0)}, "
        f"deterministic: {args.deterministic}"
    )

    ## Set Seed
    set_random_seed(cfg.get("seed", 0), deterministic=args.deterministic)

    ## Build Model
    model = _build_deploy_model(cfg,args.base_checkpoint)   #换成horizon的forward
    calib_ckpt = os.path.join(CALIBRATION_CKPT)
    if os.path.exists(calib_ckpt):
        model.load_state_dict(torch.load(CALIBRATION_CKPT))
        logger.info("Loaded Calib weights for QAT")

    logger.info(f"Model:\n{model}")

    ## Build Dataset
    datasets = [build_module(cfg["data"]["train"])]
    logger.info(
        f"Load All TrainData Sum = [{len(datasets[0])}], Scene Sum = [{len(datasets[0]._scene)}]."
    )
    
    ## Call TrainApi
    train_api(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )
#-------------------------------------------------------------------------------


def main():
    os.makedirs(
        WORK_DIR + PREFIEX,
        exist_ok=True,
    )

    args = parse_args()
    cfg = read_cfg(args.config)
    args.qat_epochs = getattr(cfg, 'qat_epochs', 20)

    # Same essential model setting as SparseEnd2End test.py.
    cfg["model"]["img_backbone"]["init_cfg"] = {}

    if cfg.get("work_dir", None) is None:
        cfg["work_dir"] = osp.join(
            "./e2e_worklog",
            osp.splitext(
                osp.basename(args.config)
            )[0],
        )

    os.makedirs(
        cfg["work_dir"],
        exist_ok=True,
    )

    # Keep random seed operation from SparseEnd2End test.py.
    set_random_seed(args.seed, deterministic=args.deterministic)

    # Phase is intentionally required, same as the supplied SparseDrive QAT
    # workflow.
    if args.phase == "calib":
        run_calib_phase(args, cfg)
    elif args.phase == "train":
        run_train_phase(args, cfg)
    elif args.phase == "QAT_test":
        run_QAT_test_phase(args, cfg) 
    elif args.phase == "compile":
        run_compile_phase(args, cfg)
    elif args.phase == "all":
        run_calib_phase(args, cfg)
        run_train_phase(args, cfg)
    else:
        raise ValueError("--phase is required (calib, train, or all)")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork", force=True)
    main()