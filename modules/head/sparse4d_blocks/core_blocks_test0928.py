# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional
from .sparse3d_embedding import *
from torch.cuda.amp.autocast_mode import autocast
from modules.cnn.base_module import Sequential, BaseModule
from modules.cnn.module import xavier_init, constant_init

try:
    # from ...ops import deformable_aggregation_function as DAF
    from ...ops import (
    deformable_aggregation_function as DAF,
    feature_maps_format,
)
except:
    DAF = None

__all__ = [
    "DeformableAttentionAggr",
    "DenseDepthNet",
    "AsymmetricFFN",
    "GridMask",
]


class DeformableAttentionAggr(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableAttentionAggr, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        self.kps_generator = build_module(kps_generator)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_module(temporal_fusion_module)
        else:
            self.temp_module = None
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
            # self.camera_encoder = Sequential(*linear_relu_ln(embed_dims, 1, 2, 16))
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        """
        Args:
            instance_feature: (bs, 900+x, 256) train:x=5*32*2 test=0;
            anchor: (bs, 900+x, 11) train:x=5*32*2 test=0;
            anchor_embed: (bs, 900+x, 256) train:x=5*32*2 test=0;
        Return:
            out: (bs, 900+x, 512)
        """
        bs, num_anchor = instance_feature.shape[:2]
        # [bs, num_anchors, 7+6, 3]
        key_points = self.kps_generator(anchor, instance_feature) #关键点生成​  #shape:torch.Size([1, 1210, 13, 3])
        # [bs, num_anchors, 6, 4, 13, 8]
        # weights = self._get_weights(instance_feature, anchor_embed, metas)
        weights_base = self._get_weights(
            instance_feature,
            anchor_embed,
            metas,
        )

        # if self.use_deformable_func:
        #     # points_2d = (
        #     #     self.project_points(  # 3D→2D投影
        #     #         key_points,
        #     #         metas["lidar2img"],  # lidar2img
        #     #         metas.get("image_wh"),
        #     #     )
        #     #     .permute(0, 2, 3, 1, 4)
        #     #     .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
        #     # )
        #     points_2d = (
        #         self.project_points_fisheye(  # fisheye 3D→2D投影
        #             key_points,
        #             metas["cam_extrinsic"],  # lidar2cam
        #             metas["cam_intrinsic"],
        #             metas["cam_distortion"],
        #             metas.get("image_wh"),
        #         )
        #         .permute(0, 2, 3, 1, 4)
        #         .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
        #     )
        #     weights = (
        #         weights.permute(0, 1, 4, 2, 3, 5)
        #         .contiguous()
        #         .reshape(
        #             bs,
        #             num_anchor,  # 1220
        #             self.num_pts,  # 13
        #             self.num_cams,  # 6
        #             self.num_levels,  # 4
        #             self.num_groups,  # 8
        #         )
        #     )
        #     # (bs, 1220, 256)
        #     features = DAF(*feature_maps, points_2d, weights).reshape( #可变形注意力
        #         bs, num_anchor, self.embed_dims
        #     )# DAF算子在BPU没有，所以不走if分支

        if self.use_deformable_func:

            # ============================================================
            # DEBUG:
            # DAF 和 grid_sample 必须共用完全相同的：
            #   1. feature_maps
            #   2. points_2d
            #   3. weights
            #
            # 这样才能单独判断：
            # DAF 与 grid_sample + fusion 是否数值等价
            # ============================================================

            # 保留原始 weights。
            # weights_base:
            # [bs, num_anchor, num_cams,
            #  num_levels, num_pts, num_groups]
            weights_base = weights

            # ============================================================
            # 1. 只计算一次 points_2d
            #
            # 注意：
            # 当前你的高精度 DAF 基线使用的是 project_points，
            # 所以测试阶段必须继续使用 project_points，
            # 不要在这里改成 fisheye。
            # ============================================================
            points_2d_raw = self.project_points(
                key_points,
                metas["lidar2img"],
                metas.get("image_wh"),
            )

            # points_2d_raw:
            # [bs, num_cams, num_anchor, num_pts, 2]
            # 坐标范围按原逻辑为 [0, 1]


            # ============================================================
            # 2. 原始 DAF 路径
            # ============================================================
            points_2d_daf = (
                points_2d_raw
                .permute(0, 2, 3, 1, 4)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    2,
                )
            )

            weights_daf = (
                weights_base
                .permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )

            features_daf = DAF(
                *feature_maps,
                points_2d_daf,
                weights_daf,
            ).reshape(
                bs,
                num_anchor,
                self.embed_dims,
            )


            # ============================================================
            # 3. 将 DAF 格式的 feature_maps 恢复为：
            #
            # List[
            #   (bs, num_cams, C, H0, W0),
            #   (bs, num_cams, C, H1, W1),
            #   ...
            # ]
            #
            # 因为 use_deformable_func=True 时，
            # detector.extract_feat() 已经调用了 feature_maps_format。
            # ============================================================
            raw_feature_maps = feature_maps_format(
                feature_maps,
                inverse=True,
            )

            # 通常 6 个 camera 的 FPN shape 一样，
            # inverse 后最外层只有一个 camera group。
            if isinstance(
                raw_feature_maps[0],
                (list, tuple),
            ):
                if len(raw_feature_maps) != 1:
                    raise RuntimeError(
                        "DEBUG: feature_maps_format inverse "
                        "returned multiple camera groups."
                    )

                raw_feature_maps = list(
                    raw_feature_maps[0]
                )


            # ============================================================
            # 4. grid_sample:
            # 分别测试 align_corners=False / True
            # ============================================================
            features_grid_false = (
                self._debug_grid_fusion_from_points(
                    raw_feature_maps,
                    points_2d_raw,
                    weights_base,
                    align_corners=False,
                )
            )

            features_grid_true = (
                self._debug_grid_fusion_from_points(
                    raw_feature_maps,
                    points_2d_raw,
                    weights_base,
                    align_corners=True,
                )
            )


            # ============================================================
            # 5. 数值对比
            # ============================================================
            with torch.no_grad():

                daf_fp32 = features_daf.float()

                grid_false_fp32 = (
                    features_grid_false.float()
                )

                grid_true_fp32 = (
                    features_grid_true.float()
                )

                # ----------------------------
                # align_corners=False
                # ----------------------------
                diff_false = (
                    daf_fp32 - grid_false_fp32
                ).abs()

                cos_false = F.cosine_similarity(
                    daf_fp32.reshape(bs, -1),
                    grid_false_fp32.reshape(bs, -1),
                    dim=1,
                ).mean()

                rel_false = (
                    diff_false.mean()
                    / (
                        daf_fp32.abs().mean()
                        + 1e-8
                    )
                )

                # ----------------------------
                # align_corners=True
                # ----------------------------
                diff_true = (
                    daf_fp32 - grid_true_fp32
                ).abs()

                cos_true = F.cosine_similarity(
                    daf_fp32.reshape(bs, -1),
                    grid_true_fp32.reshape(bs, -1),
                    dim=1,
                ).mean()

                rel_true = (
                    diff_true.mean()
                    / (
                        daf_fp32.abs().mean()
                        + 1e-8
                    )
                )

                # ----------------------------
                # 投影点有效比例
                # ----------------------------
                in_bound = (
                    (
                        (points_2d_raw >= 0)
                        & (points_2d_raw <= 1)
                    )
                    .all(dim=-1)
                    .float()
                    .mean()
                )

                print(
                    "\n========== DAF vs GRID DEBUG =========="
                )

                print(
                    "points shape:",
                    tuple(points_2d_raw.shape),
                )

                print(
                    "points min/max:",
                    points_2d_raw.min().item(),
                    points_2d_raw.max().item(),
                )

                print(
                    "points in-bound ratio:",
                    in_bound.item(),
                )

                if metas.get("image_wh") is not None:
                    print(
                        "image_wh shape:",
                        tuple(
                            metas["image_wh"].shape
                        ),
                    )

                    print(
                        "image_wh[0]:",
                        metas["image_wh"][0],
                    )

                print(
                    "--- align_corners=False ---"
                )

                print(
                    "MAE:",
                    diff_false.mean().item(),
                )

                print(
                    "MAX:",
                    diff_false.max().item(),
                )

                print(
                    "Relative MAE:",
                    rel_false.item(),
                )

                print(
                    "Cosine:",
                    cos_false.item(),
                )

                print(
                    "--- align_corners=True ---"
                )

                print(
                    "MAE:",
                    diff_true.mean().item(),
                )

                print(
                    "MAX:",
                    diff_true.max().item(),
                )

                print(
                    "Relative MAE:",
                    rel_true.item(),
                )

                print(
                    "Cosine:",
                    cos_true.item(),
                )

                print(
                    "=======================================\n"
                )


            # ============================================================
            # 非常重要：
            #
            # 最终仍然使用原来的 DAF 输出，
            # 不影响当前 60% DAF baseline 的模型行为。
            # ============================================================
            features = features_daf

            print(features.abs().mean())
            print(features.std())
            print(features.abs().max())
        else:
            # features = self.feature_sampling(
            #     feature_maps,
            #     key_points,
            #     metas["lidar2img"],
            #     metas.get("image_wh"),
            # )
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["cam_extrinsic"],  # lidar2cam
                metas["cam_intrinsic"],
                metas["cam_distortion"],
                metas.get("image_wh"),
            )   #xy 因为走的是else分支，所以这里给鱼眼的外参、内参、畸变系数
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(
                dim=2
            )  # fuse multi-point features =>(bs, 1220, 256)
            print(features.abs().mean())
            print(features.abs().max())
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        """
        instance_feature: (bs, 900+x, 256) train:x=5*32*2 test=0;
        anchor_embed: (bs, 900+x, 256) train:x=5*32*2 test=0;

        self.camera_encoder:
            Sequential(
                (0): Linear(in_features=12, out_features=256, bias=True)
                (1): ReLU(inplace=True)
                (2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (3): Linear(in_features=256, out_features=256, bias=True)
                (4): ReLU(inplace=True)
                (5): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                )
        self.weights_fc:
            Linear(in_features=256, out_features=416, bias=True)
        """
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            # 对相机的外参编码： (bs,6,4,4)=>(bs,6,256)
            camera_embed = self.camera_encoder(
                metas["lidar2img"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            # camera_embed = self.camera_encoder(
            #     torch.cat([metas["lidar2img"][:, :, :3].reshape(bs, self.num_cams, -1), 
            #               metas["cam_distortion"].reshape(bs, self.num_cams, -1)], dim = -1)
            # ) #[1, 4, 256]
            # [bs, 900+x, 1, 256] + [bs, 1, 6, 256] => (bs, 900+x, 6, 256)
            feature = feature[:, :, None] + camera_embed[:, None]

        # [] => [1, 1220, 6, 4, 13, 8]
        weights = (
            self.weights_fc(feature)  # (bs, 900+x, 6, 256)=>(bs, 900+x, 6, 416)
            .reshape(bs, num_anchor, -1, self.num_groups)  # =>(bs, 900+x, 312, 8)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        # self.attn_drop=0.15
        if self.training and self.attn_drop > 0:
            # [1, 1220, 6, 1, 13, 1]
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    @staticmethod
    def project_points(key_points, lidar2img, image_wh=None):
        """
        Args:
            key_points : Shape[1, 1220, 13, 3].
        Return:
            key_points2d : Shape[1, 6, 1220, 13, 2].


        """
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        # points_2d = torch.matmul(
        #     lidar2img[:, :, None, None], pts_extend[:, None, ..., None]
        # ).squeeze(-1)
        points_2d = torch.matmul(
            lidar2img[:, :, None, None], pts_extend[:, None, ..., None]
        )[..., 0]
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod  #opencv的数学原理，不能用封装好的cv2投影函数，用自己写的带mask的数学公式
    def project_points_fisheye(key_points, lidar2cam, K, dist, image_wh=None, fill_value=-1.0):
        """
        Fisheye 投影，过滤掉相机后方点 (Z <= 0)，保持输出 shape 不变。
        
        Args:
            key_points : torch.Tensor, [B, N, M, 3] 3D 点
            lidar2cam : torch.Tensor, [B, num_cam, 4, 4] LiDAR -> camera 投影矩阵
            K : torch.Tensor, [B, num_cam, 3, 3] 相机内参矩阵
            dist : torch.Tensor, [B, num_cam, 4] fisheye 畸变系数
            image_wh : torch.Tensor, [B, num_cam, 2] 图像宽高
            fill_value : float, 无效点填充值
            
        Returns:
            key_points2d : torch.Tensor, [B, num_cam, N, M, 2] B批次、num_cam相机数量、N目标候选数（多少个可能的车）、M每个目标的特征点数（车框上的关键点）、像素uv
        """
        B, N, M, _ = key_points.shape  #shape:torch.Size([1, 1210, 13, 3])
        num_cam = lidar2cam.shape[1]
        
        key_points2d = torch.zeros((B, num_cam, N, M, 2), device=key_points.device, dtype=torch.float32)
        eps = 1e-6
        device = key_points.device
        dtype = key_points.dtype
        for b in range(B):
            pts = key_points[b].reshape(-1, 3)
            ones = torch.ones((pts.shape[0], 1), device=key_points.device, dtype=key_points.dtype)
            pts_h = torch.cat([pts, ones], dim=-1)  # [P, 4]

            for c in range(num_cam):
                # LiDAR -> Camera
                T = lidar2cam[b, c].to(device=key_points.device, dtype=key_points.dtype)  # [4,4] 最后一行是0 0 0 1
                cam = (pts_h @ T.transpose(0, 1))[..., :3]          # [P,3]
                X = cam[..., 0]
                Y = cam[..., 1]
                Z = cam[..., 2]
                # valid depth
                mask = Z > eps
                if not mask.any():
                    continue
                Zm = Z[mask]
                x = X[mask] / Zm
                y = Y[mask] / Zm
                r = torch.sqrt(x * x + y * y).clamp_min(eps)

                theta = torch.atan(r)  # fisheye uses angle
                theta2 = theta * theta
                theta4 = theta2 * theta2
                theta6 = theta4 * theta2
                theta8 = theta4 * theta4

                D = dist[b, c].to(device=device, dtype=dtype)  # [4]
                k1, k2, k3, k4 = D[0], D[1], D[2], D[3]
                theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

                scale = theta_d / r
                xd = x * scale
                yd = y * scale

                Kc = K[b, c].to(device=device, dtype=dtype)  # [3,3]
                fx, fy = Kc[0, 0], Kc[1, 1]
                cx, cy = Kc[0, 2], Kc[1, 2]

                u = fx * xd + cx
                v = fy * yd + cy

                uv = torch.stack([u, v], dim=-1)  # [num_valid,2]

                # write back full (P,2)
                full = torch.full((pts.shape[0], 2), fill_value, device=device, dtype=dtype)
                full[mask] = uv

                if image_wh is not None:
                    h = image_wh[b][0].reshape(-1)
                    w = image_wh[b][1].reshape(-1)
                    wh = torch.cat([w, h], dim=0)  #tensor([704., 256.], device='cuda:0')
                    full = full / wh  #0到1之间

                key_points2d[b, c] = full.reshape(N, M, 2)

        return key_points2d

    @staticmethod
    def feature_sampling(
        feature_maps,
        key_points,
        lidar2cam,
        K,
        dist,
        image_wh=None,
    ) -> torch.Tensor:
        """
        Args:
            feature_maps: List[Tensor]
                        (bs, 6, 256, 64, 176) float32
                        (bs, 6, 256, 32, 88)  float32
                        (bs, 6, 256, 16, 44)  float32
                        (bs, 6, 256, 8,  22)  float32
            key_points: TensorShape=(bs, 1220, 12, 3)
        Returns:
            TensorShape=(bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)
        """
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        # points_2d = DeformableAttentionAggr.project_points(
        #     key_points, lidar2img, image_wh
        # )
        points_2d = DeformableAttentionAggr.project_points_fisheye(
            key_points, lidar2cam, K, dist, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(
            end_dim=1
        )  # (bs, 6, 1220, 13, 2)=> (bs*6, 1220, 13, 2)

        features = []
        # (6*bs, 256, 1220, 12)
        # (6*bs, 256, 1220, 12)
        # (6*bs, 256, 1220, 12)
        # (6*bs, 256, 1220, 12)
        for fm in feature_maps:
            features.append(nn.functional.grid_sample(fm.flatten(end_dim=1), points_2d))
        features = torch.stack(features, dim=1)  # (6*bs, 4, 256, 1220, 12)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def _debug_grid_fusion_from_points(
        self,
        feature_maps,
        points_2d,
        weights,
        align_corners=False,
    ):
        """
        Debug only.

        Args:
            feature_maps:
                raw multi-level feature maps:
                List[(bs, num_cams, C, H, W)]

            points_2d:
                exactly the SAME projected coordinates used by DAF
                shape: (bs, num_cams, num_anchor, num_pts, 2)
                range: normalized to [0, 1]

            weights:
                shape:
                (bs, num_anchor, num_cams,
                 num_levels, num_pts, num_groups)
        """

        bs = points_2d.shape[0]
        num_cams = points_2d.shape[1]
        num_anchor = points_2d.shape[2]
        num_pts = points_2d.shape[3]
        num_levels = len(feature_maps)

        # grid_sample expects [-1, 1]
        grid = points_2d * 2.0 - 1.0

        grid = grid.reshape(
            bs * num_cams,
            num_anchor,
            num_pts,
            2,
        )

        sampled_features = []

        for fm in feature_maps:
            sampled = F.grid_sample(
                fm.flatten(end_dim=1),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=align_corners,
            )

            sampled_features.append(sampled)

        sampled_features = torch.stack(
            sampled_features,
            dim=1,
        )

        sampled_features = sampled_features.reshape(
            bs,
            num_cams,
            num_levels,
            -1,
            num_anchor,
            num_pts,
        ).permute(
            0,
            4,
            1,
            2,
            5,
            3,
        )

        # camera + level fusion
        fused_features = self.multi_view_level_fusion(
            sampled_features,
            weights,
        )

        # point fusion
        fused_features = fused_features.sum(dim=2)

        return fused_features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        """
        Args:
            features: TensorShape(bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)
                                 (bs, 1220, num_cams, 4, 13, 256)
            weights: TensorShape(bs, 1220, num_cams, 4, 13, 8)
        Return:
            features: (bs, num_anchor, self.num_pts, self.embed_dims)
                      (1,1220,13,256)
        """
        bs, num_anchor = weights.shape[:2]
        # (bs, 1220, 6, 4, 13, 8, 1) * (bs, 1220, 6, 4, 13, 8, 32)  => (bs, 1220, 6, 4, 13, 8, 32)
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        # 将num_cams和num_levels维度合并：(1, 1220, 13, 8, 32)
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        return features

#-----------------------------------0922--------------------------------------
    def horizon_forward(
        self,
        instance_feature,
        anchor,
        anchor_embed,
        feature_maps,
        # lidar2img,
        cam_extrinsic,
        cam_intrinsic,
        cam_distortion,
        image_wh=None,
    ):
        key_points = self.kps_generator.horizon_forward(
            anchor,
            instance_feature,
        )

        weights = self._get_weights_horizon(
            instance_feature,
            anchor_embed,
            cam_extrinsic,
            cam_intrinsic,
        )

        features = self.feature_sampling_horizon(
            feature_maps,
            key_points,
            cam_extrinsic,
            cam_intrinsic,
            cam_distortion,
            image_wh,
        )

        features = self.multi_view_level_fusion(features, weights)
        features = features.sum(dim=2)

        output = self.proj_drop(self.output_proj(features))

        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return output


    def _get_weights_horizon(
        self,
        instance_feature,
        anchor_embed,
        cam_extrinsic,
        cam_intrinsic,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed

        if self.camera_encoder is not None:
            lidar2img = torch.matmul(
                cam_intrinsic,
                cam_extrinsic[:, :, :3, :],
            )
            camera_embed = self.camera_encoder(
                lidar2img.reshape(
                    bs,
                    self.num_cams,
                    -1,
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = self.weights_fc(feature)
        weights = (
            weights.reshape(
                bs,
                num_anchor,
                -1,
                self.num_groups,
            )
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        return weights

    @staticmethod
    def project_points_fisheye_horizon(
        key_points,
        lidar2cam,
        K,
        dist,
        image_wh=None,
        fill_value=-1.0,
    ):
        eps = 1e-6

        ones = torch.ones_like(key_points[..., :1])
        pts = torch.cat([key_points, ones], dim=-1)

        cam = torch.matmul(
            lidar2cam[:, :, None, None, :3, :],
            pts[:, None, ..., None],
        )[..., 0]

        X, Y, Z = cam[..., 0], cam[..., 1], cam[..., 2]

        valid = Z > eps
        Z = torch.where(valid, Z, torch.ones_like(Z))

        x = X / Z
        y = Y / Z

        r = torch.sqrt(x*x + y*y).clamp_min(eps)
        theta = torch.atan(r)

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        k1 = dist[:, :, 0][:, :, None, None]
        k2 = dist[:, :, 1][:, :, None, None]
        k3 = dist[:, :, 2][:, :, None, None]
        k4 = dist[:, :, 3][:, :, None, None]

        theta_d = theta * (
            1 + k1*theta2 + k2*theta4
            + k3*theta6 + k4*theta8
        )

        scale = theta_d / r

        xd = x * scale
        yd = y * scale

        fx = K[:, :, 0, 0][:, :, None, None]
        fy = K[:, :, 1, 1][:, :, None, None]
        cx = K[:, :, 0, 2][:, :, None, None]
        cy = K[:, :, 1, 2][:, :, None, None]

        uv = torch.stack(
            [fx*xd + cx, fy*yd + cy],
            dim=-1,
        )

        uv = torch.where(
            valid[..., None],
            uv,
            torch.ones_like(uv) * fill_value,
        )

        if image_wh is not None:
            uv = uv / image_wh[:, :, None, None, :]

        return uv


    @staticmethod
    def feature_sampling_horizon(
        feature_maps,
        key_points,
        lidar2cam,
        K,
        dist,
        image_wh=None,
    ):
        bs = key_points.shape[0]
        num_anchor = key_points.shape[1]
        num_pts = key_points.shape[2]
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]

        points_2d = DeformableAttentionAggr.project_points_fisheye_horizon(
            key_points,
            lidar2cam,
            K,
            dist,
            image_wh,
        )

        points_2d = points_2d * 2 - 1
        points_2d = points_2d.reshape(
            bs * num_cams,
            num_anchor,
            num_pts,
            2,
        )

        features = []

        for fm in feature_maps:
            features.append(
                F.grid_sample(
                    fm.flatten(end_dim=1),
                    points_2d,
                )
            )

        features = torch.stack(features, dim=1)

        return features.reshape(
            bs,
            num_cams,
            num_levels,
            -1,
            num_anchor,
            num_pts,
        ).permute(
            0, 4, 1, 2, 5, 3
        )
#-----------------------------------------------------------------------------

class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp() #.exp()：将深度输出从 log 空间还原成正数  [4, 1, 64, 176])
            depth = depth.transpose(0, -1) * focal / self.equal_focal  #根据焦距对深度做比例缩放
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths): 
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1) #[45056]  [11264] [2816]
            gt = gt.reshape(-1) #[45056] [11264] [2816]
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight
            loss = loss + _loss
        return loss

class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=256 * 2,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.activate = nn.ReLU(inplace=True)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        self.pre_norm = nn.modules.normalization.LayerNorm(normalized_shape=512)

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = nn.Identity()
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                nn.Identity()
                if in_channels == embed_dims
                else nn.Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)


class Grid(object):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        mask = torch.from_numpy(mask.copy()).float().cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)
