# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
log_level = "INFO"
dist_params = dict(backend="nccl")

total_batch_size =20    #88
num_gpus =1      #4
batch_size = total_batch_size // num_gpus  #训练用22，debug调试用1


# ================== model ========================
# class_names = [
#     "car",
#     "truck",
#     "construction_vehicle",
#     "bus",
#     "trailer",
#     "barrier",
#     "motorcycle",
#     "bicycle",
#     "pedestrian",
#     "traffic_cone",
# ]  #原先的
class_names = [
    "barrier",
    "bus",
    "car",
    "construction_vehicle",
    "motor",
    "bicycle",
    "animal",
    "pedestrian",
    "traffic_cone",
    "stopper",
    "trash_bin",
    "sign",
    "truck",
]


num_classes = len(class_names)
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
temporal = False
decouple_attn = True
with_quality_estimation = False  #0128 True #0207 False

model = dict(
    type="Sparse4D",
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(type="Pretrained", checkpoint="ckpt/resnet50-19c8e357.pth"), #TODO，预训练权重 【您要改成您的resnet50的路径】
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    head=dict(
        type="Sparse4DHead",
        cls_threshold_to_reg=0.05,
        decouple_attn=decouple_attn,
        instance_bank=dict(
            type="InstanceBank",
            num_anchor=384,    #改为384
            embed_dims=embed_dims,
            anchor="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/data/bevod_kmeans384.npy",   #Kmeans
            anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
            num_temp_instances=600 if temporal else -1,
            confidence_decay=0.6,
            feat_grad=False,
        ),
        anchor_encoder=dict(
            type="SparseBox3DEncoder",
            vel_dims=0, #bevod的label没有vel标签，从3改为0
            embed_dims=[128, 64, 64, 64] if decouple_attn else 256, #1218 前3个数加起来一定要是256，但是没有速度维
            mode="cat" if decouple_attn else "add",
            output_fc=not decouple_attn,
            in_loops=1,
            out_loops=4 if decouple_attn else 2,
        ),
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=(
            [
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * num_single_frame_decoder
            + [
                # "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ]
            * (num_decoder - num_single_frame_decoder)
        )[2:],
        temp_graph_model=(
            dict(
                type="MultiheadAttention",
                embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
                num_heads=num_groups,
                batch_first=True,
                attn_drop=drop_out,
            )
            if temporal
            else None
        ),
        graph_model=dict(
            type="MultiheadAttention",
            embed_dims=embed_dims if not decouple_attn else embed_dims * 2,
            num_heads=num_groups,
            batch_first=True,
            attn_drop=drop_out,
        ),
        norm_layer=dict(type="LayerNorm", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            pre_norm=dict(type="LayerNorm"),
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            num_fcs=2,
            ffn_drop=drop_out,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        deformable_model=dict(
            type="DeformableAttentionAggr",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            # num_cams=6,  #原先的 6摄像头
            num_cams=4,    # 我们是4摄像头
            attn_drop=0.15,
            use_deformable_func=False,   #0921 True 改为False  #0925xy 改成True   #0928 False
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseBox3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
            ),
        ),
        refine_layer=dict(
            type="SparseBox3DRefinementModule",
            embed_dims=embed_dims,
            num_cls=num_classes,
            refine_yaw=True,
            with_quality_estimation=with_quality_estimation,
        ),
        sampler=dict(
            type="SparseBox3DTarget",
            num_dn_groups=5,
            num_temp_dn_groups=3,
            # dn_noise_scale=[2.0] * 3 + [0.5] * 7, #原先的，10维。[0.5] * 7就算改成[0.5]*5+[0.0]*2也还是10维。数据是8维，不匹配
            dn_noise_scale=[2.0] * 3 + [0.5] * 5, #修改，8维，没有vx、vy
            max_dn_gt=32,
            add_neg_dn=True,
            cls_weight=2.0,
            box_weight=0.25,
            # reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,  #原先的，10维
            reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 2,  #修改，8维，
            cls_wise_reg_weights={
                # class_names.index("traffic_cone"): [ 
                #     2.0,
                #     2.0,
                #     2.0,
                #     1.0,
                #     1.0,
                #     1.0,
                #     0.0,
                #     0.0,
                #     1.0,
                #     1.0,
                # ], 
                class_names.index("traffic_cone"): [  #修改，删去最后的2个1
                    2.0,
                    2.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],    
            },
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_reg=dict(
            type="SparseBox3DLoss",
            loss_box=dict(type="L1Loss", loss_weight=0.25),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),  
            loss_yawness=dict(type="GaussianFocalLoss"),  
            cls_allow_reverse=[class_names.index("barrier")],
        ),
        decoder=dict(type="SparseBox3DDecoder"),
        # reg_weights=[2.0] * 3 + [1.0] * 7,  #原先的
        reg_weights=[2.0] * 3 + [1.0] * 5,    #修改，8维
    ),
    depth_branch=dict(  # for auxiliary supervision only 深度估计提供像素级监督，缓解3D检测任务中稀疏标注的问题。
        type="DenseDepthNet",  #modules/head/sparse4d_blocks/core_blocks.py，没有改过
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,  #3
        loss_weight=0.2,  #源码是0.2
    ),
    use_grid_mask=True,
    use_deformable_func=False,   #0921 True 改为False    #0925xy 改成True  #0928False
)

# ================== data ========================
dataset_type = "NuScenes4DDetTrackDataset"
# data_root = "data/nuscenes/"  #原先的
data_root = "/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/data/fisheye_data_2w_frame"   #TODO 改成2万帧带点云的路径"data" / "fisheye_data_2w_frame"，相对路径失效就直接用绝对路径


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        # load_dim=5,
        # use_dim=5,
        load_dim=3,  # 1211 只加载x,y,z
        use_dim=[0, 1, 2],  # 1211 使用前3维
    ),
    dict(type="ResizeCropFlipImage"),
    dict(
        # type="MultiScaleDepthMapGenerator", #1223 源码
        type="MultiScaleDepthMapGeneratorFisheye",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "lidar2img",
            "image_wh",
            "gt_depth",
            "gt_depth_ori",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "cam_intrinsic",
            "cam_extrinsic",
            "cam_distortion"
        ],
        meta_keys=[
            "sample_idx",
            "sample_scene",
            # "lidar2global",
            # "global2lidar",
            "timestamp",
            # "track_id",
        ],
    ),
]

val_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        # load_dim=5,
        # use_dim=5,
        load_dim=3,  # 1211 只加载x,y,z
        use_dim=[0, 1, 2],  # 1211 使用前3维
    ),
    # dict(type="ResizeCropFlipImage"),
    dict(
        # type="MultiScaleDepthMapGenerator", #1223 源码
        type="MultiScaleDepthMapGeneratorFisheye",
        downsample=strides[:num_depth_layers],
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        class_dist_thred=[55] * len(class_names),
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "lidar2img",
            "image_wh",
            "gt_depth",
            "gt_depth_ori",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "cam_intrinsic",
            "cam_extrinsic",
            "cam_distortion"
        ],
        meta_keys=[
            "sample_idx",
            "sample_scene",
            "filename",
            # "lidar2global",
            # "global2lidar",
            "timestamp",
            # "track_id",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),  #需要打开，和训练一致
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=["img", "timestamp", "lidar2img", "image_wh", "ori_img", "cam_intrinsic",
            "cam_extrinsic",
            "cam_distortion"],
        # meta_keys=["lidar2global", "global2lidar", "timestamp"],
        meta_keys=[ "timestamp"],
    ),
]


data_basic_config = dict(
    type=dataset_type, data_root=data_root, classes=class_names, version="v1.0-trainval"
)

input_shape = (704, 256)
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],  # w,h
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [-0.3925, 0.3925],
}

tracking_threshold = 0.2
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,  #训练改成12  #调试是1
    train=dict(
        **data_basic_config,
        ann_file="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/data/BEVOD_nusc_anno_dumpjson_mini/train",
        pipeline=train_pipeline,
        train_mode=True,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=1,   #1211  sweeps只有1
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config, #这里已经传入version字段了
        ann_file="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/data/BEVOD_nusc_anno_dumpjson_mini/val",  
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        val_mode=True,
        tracking=False,
        tracking_threshold=tracking_threshold,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        classes=class_names,
        version="v1.0-trainval",
        ann_file="/home/E2EModel/Model_Train_End/Sparse4D_QAT_dws/data/BEVOD_nusc_anno_dumpjson_mini/val",   
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        tracking=False,
        tracking_threshold=tracking_threshold,
        samples_per_gpu=1,
    )
)

# ================== runner&hook ========================
seed = 100
# RunnerSetting
workflow = [("train", 1)]  
# num_epochs = 200       #2万帧百度数据集是200epoch   #TODO
num_epochs = 10   #QAT用10个epoch
num_iters_per_epoch = int(16261 // (num_gpus * batch_size))  #训练集帧数  TODO

runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)


# CheckpointSaverHookSetting
checkpoint_epoch_interval = 1
checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)


# LoggerHookSetting
log_config = dict(
    interval=50,  #改日志步长
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)

# ================== train ========================

optimizer = dict(
    type="AdamW",
    lr=6e-4,  #1225用2e-5 #0107用2e-4  #0917 6e-4
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)

# LrUpdaterHookSetting
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# OptimizerStepperHookSetting
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
# FP16 MixedPrecision Setting (A100 AMP)
# fp16 = dict(loss_scale=512.0)  #wuling
fp16 = dict(loss_scale=32.0)   #官方



# ================== eval ========================

vis_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["timestamp", "lidar2img"],
    ),
]
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval *10,    #每10个epoch测一下evaluation
    pipeline=vis_pipeline,
)

# ================== placehold ========================
work_dir = None
load_from = None
resume_from = None
gpu_ids = None  # only applicable to non-distributed training
