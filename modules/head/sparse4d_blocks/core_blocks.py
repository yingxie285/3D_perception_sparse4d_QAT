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
    from ...ops import deformable_aggregation_function as DAF
except:
    DAF = None
    
#加了HORIZON_QAT
try:
    import os
    from horizon_plugin_pytorch.nn.matmul import Matmul
    from horizon_plugin_pytorch.nn.div import Div
    from torch.nn.quantized import FloatFunctional
    from horizon_plugin_pytorch.quantization import QuantStub
    HORIZON_QAT = os.environ.get("HORIZON_QAT", "0") == "1"
except:
    HORIZON_QAT = False

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
            
        self.matmul_test = Matmul()
            
        #加horizon_init
        if HORIZON_QAT:
            self.horizon_init()

    def horizon_init(self):
        # self.float_func_matmul1 = Matmul()  #TODO
        # self.float_func_matmul2 = Matmul()
        # self.float_func_add3xy = FloatFunctional()
        # self.float_func_add3z = FloatFunctional()
        # self.float_func_add4 = FloatFunctional()
        # self.float_func_add5 = FloatFunctional()
        # self.float_func_div = Div()
        # self.float_func_div_wh = Div()
        self.matmul_test = Matmul()    
    
    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    #xy0928 先得到投影点，再走else分支
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]

        # =========================================================
        # 1. 生成3D key points
        # =========================================================
        key_points = self.kps_generator(anchor, instance_feature)

        # =========================================================
        # 2. 统一使用与DAF完全相同的fisheye投影
        #    [B, N, P, 3]
        # -> [B, C, N, P, 2]
        # =========================================================
        points_2d = self.project_points_fisheye(
            key_points,
            metas["cam_extrinsic"],
            metas["cam_intrinsic"],
            metas["cam_distortion"],
            metas.get("image_wh"),
        )

        # =========================================================
        # 3. 统一生成attention weights
        #    [B, N, C, L, P, G]
        # =========================================================
        weights = self._get_weights(
            instance_feature,
            anchor_embed,
            metas
        )

        # =========================================================
        # 4. DAF路径
        # =========================================================
        if self.use_deformable_func:

            points_2d_daf = (
                points_2d
                .permute(0, 2, 3, 1, 4)
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    2,
                )
            )

            weights_daf = (
                weights
                .permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor * self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )

            # DAF输出：
            # [B, N*P, E]
            features = DAF(
                *feature_maps,
                points_2d_daf,
                weights_daf,
            )

            # [B, N, P, E]
            features = features.reshape(
                bs,
                num_anchor,
                self.num_pts,
                self.embed_dims,
            )

            # point fusion
            # [B, N, E]
            features = features.sum(dim=2)
            print("走的DAF")

        # =========================================================
        # 5. 不使用DAF：纯PyTorch等价路径
        # =========================================================
        else:

            # [B, N, C, L, P, E]
            # DFA casts feature maps, locations, and weights to float32
            # before sampling. Keep the same behavior here: under AMP,
            # grid_sample in float16 has materially different backward
            # gradients, especially for projection locations.
            with autocast(enabled=False):
                feature_maps_fp32 = [
                    feature_map.float() for feature_map in feature_maps
                ]
                points_2d_fp32 = points_2d.float()
                weights_fp32 = weights.float()
                features = self.feature_sampling(
                    feature_maps_fp32,
                    points_2d_fp32,
                )

                # camera + level fusion
                # [B, N, P, E]
                features = self.multi_view_level_fusion(
                    features,
                    weights_fp32,
                )

            # point fusion
            # [B, N, E]
            features = features.sum(dim=2)
            print("走的else分支")

        # =========================================================
        # 6. 后续逻辑不变
        # =========================================================
        output = self.proj_drop(
            self.output_proj(features)
        )

        if self.residual_mode == "add":
            output = output + instance_feature

        elif self.residual_mode == "cat":
            output = torch.cat(
                [output, instance_feature],
                dim=-1,
            )

        return output
    
    #这是原先的forward
    # def forward(
    #     self,
    #     instance_feature: torch.Tensor,
    #     anchor: torch.Tensor,
    #     anchor_embed: torch.Tensor,
    #     feature_maps: List[torch.Tensor],
    #     metas: dict,
    #     **kwargs: dict,
    # ):
    #     """
    #     Args:
    #         instance_feature: (bs, 900+x, 256) train:x=5*32*2 test=0;
    #         anchor: (bs, 900+x, 11) train:x=5*32*2 test=0;
    #         anchor_embed: (bs, 900+x, 256) train:x=5*32*2 test=0;
    #     Return:
    #         out: (bs, 900+x, 512)
    #     """
    #     bs, num_anchor = instance_feature.shape[:2]
    #     # [bs, num_anchors, 7+6, 3]
    #     key_points = self.kps_generator(anchor, instance_feature) #关键点生成​  #shape:torch.Size([1, 1210, 13, 3])
    #     # [bs, num_anchors, 6, 4, 13, 8]
    #     weights = self._get_weights(instance_feature, anchor_embed, metas)

    #     if self.use_deformable_func:
    #         # points_2d = (
    #         #     self.project_points(  # 3D→2D投影
    #         #         key_points,
    #         #         metas["lidar2img"],  # lidar2img
    #         #         metas.get("image_wh"),
    #         #     )
    #         #     .permute(0, 2, 3, 1, 4)
    #         #     .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
    #         # )
    #         points_2d = (
    #             self.project_points_fisheye(  # fisheye 3D→2D投影
    #                 key_points,
    #                 metas["cam_extrinsic"],  # lidar2cam
    #                 metas["cam_intrinsic"],
    #                 metas["cam_distortion"],
    #                 metas.get("image_wh"),
    #             )
    #             .permute(0, 2, 3, 1, 4)
    #             .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
    #         )
    #         weights = (
    #             weights.permute(0, 1, 4, 2, 3, 5)
    #             .contiguous()
    #             .reshape(
    #                 bs,
    #                 num_anchor,  # 1220
    #                 self.num_pts,  # 13
    #                 self.num_cams,  # 6
    #                 self.num_levels,  # 4
    #                 self.num_groups,  # 8
    #             )
    #         )
    #         # (bs, 1220, 256)
    #         features = DAF(*feature_maps, points_2d, weights).reshape( #可变形注意力
    #             bs, num_anchor, self.embed_dims
    #         )# DAF算子在BPU没有，所以不走if分支
    #         print(features.abs().mean())
    #         print(features.std())
    #         print(features.abs().max())
    #     else:
    #         # features = self.feature_sampling(
    #         #     feature_maps,
    #         #     key_points,
    #         #     metas["lidar2img"],
    #         #     metas.get("image_wh"),
    #         # )
    #         features = self.feature_sampling(
    #             feature_maps,
    #             key_points,
    #             metas["cam_extrinsic"],  # lidar2cam
    #             metas["cam_intrinsic"],
    #             metas["cam_distortion"],
    #             metas.get("image_wh"),
    #         )   #xy 因为走的是else分支，所以这里给鱼眼的外参、内参、畸变系数
    #         features = self.multi_view_level_fusion(features, weights)
    #         features = features.sum(
    #             dim=2
    #         )  # fuse multi-point features =>(bs, 1220, 256)
    #         # print(features.abs().mean())
    #         # print(features.abs().max())
    #     output = self.proj_drop(self.output_proj(features))
    #     if self.residual_mode == "add":
    #         output = output + instance_feature
    #     elif self.residual_mode == "cat":
    #         output = torch.cat([output, instance_feature], dim=-1)
    #     return output

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

    @staticmethod  #xy0928
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        points_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_maps:
                [
                    [B, Cams, E, H1, W1],
                    [B, Cams, E, H2, W2],
                    [B, Cams, E, H3, W3],
                    [B, Cams, E, H4, W4],
                ]

            points_2d:
                [B, Cams, N, P, 2]

                坐标已经归一化到 [0, 1]

        Returns:
            features:
                [B, N, Cams, Levels, P, E]
        """

        bs, num_cams, num_anchor, num_pts, _ = points_2d.shape
        num_levels = len(feature_maps)

        # =========================================================
        # 1. DAF的有效采样区域
        #
        # DAF CUDA中明确规定：
        #
        # if loc_w <= 0 || loc_w >= 1 return;
        # if loc_h <= 0 || loc_h >= 1 return;
        #
        # 所以这里必须严格使用 >0 且 <1
        # =========================================================
        valid_mask = (
            (points_2d[..., 0] > 0)
            & (points_2d[..., 0] < 1)
            & (points_2d[..., 1] > 0)
            & (points_2d[..., 1] < 1)
        )
        # [B, Cams, N, P]

        # =========================================================
        # 2. [0,1] -> [-1,1]
        # =========================================================
        grid = points_2d * 2.0 - 1.0

        # [B, Cams, N, P, 2]
        # ->
        # [B*Cams, N, P, 2]
        grid = grid.flatten(end_dim=1)

        features = []

        # =========================================================
        # 3. 每个level分别做bilinear sampling
        # =========================================================
        for fm in feature_maps:

            # fm:
            # [B, Cams, E, H, W]
            #
            # ->
            # [B*Cams, E, H, W]
            fm = fm.flatten(end_dim=1)

            sampled = F.grid_sample(
                fm,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            # [B*Cams, E, N, P]
            features.append(sampled)

        # =========================================================
        # 4. stack level
        # =========================================================
        features = torch.stack(
            features,
            dim=1,
        )

        # [B*Cams, L, E, N, P]
        features = features.reshape(
            bs,
            num_cams,
            num_levels,
            -1,
            num_anchor,
            num_pts,
        )

        # [B, N, Cams, L, P, E]
        features = features.permute(
            0,
            4,
            1,
            2,
            5,
            3,
        )

        # =========================================================
        # 5. 非法投影点直接置0
        #
        # 这个非常重要
        # =========================================================

        # [B, Cams, N, P]
        #
        # ->
        # [B, N, Cams, 1, P, 1]
        valid_mask = valid_mask.permute(
            0,
            2,
            1,
            3,
        ).unsqueeze(3).unsqueeze(-1)

        features = features * valid_mask.to(
            dtype=features.dtype
        )
        # print("features.shape =", features.shape)
        # print("valid ratio =", valid_mask.float().mean())
        # print("features abs mean =", features.abs().mean())
        # print("features abs max =", features.abs().max())

        return features

    # @staticmethod  #这是原先写的
    # def feature_sampling(
    #     feature_maps,
    #     key_points,
    #     lidar2cam,
    #     K,
    #     dist,
    #     image_wh=None,
    # ) -> torch.Tensor:
    #     """
    #     Args:
    #         feature_maps: List[Tensor]
    #                     (bs, 6, 256, 64, 176) float32
    #                     (bs, 6, 256, 32, 88)  float32
    #                     (bs, 6, 256, 16, 44)  float32
    #                     (bs, 6, 256, 8,  22)  float32
    #         key_points: TensorShape=(bs, 1220, 12, 3)
    #     Returns:
    #         TensorShape=(bs, num_anchor, num_cams, num_levels, num_pts, embed_dims)
    #     """
    #     num_levels = len(feature_maps)
    #     num_cams = feature_maps[0].shape[1]
    #     bs, num_anchor, num_pts = key_points.shape[:3]

    #     # points_2d = DeformableAttentionAggr.project_points(
    #     #     key_points, lidar2img, image_wh
    #     # )
    #     points_2d = DeformableAttentionAggr.project_points_fisheye(
    #         key_points, lidar2cam, K, dist, image_wh
    #     )
    #     points_2d = points_2d * 2 - 1
    #     points_2d = points_2d.flatten(
    #         end_dim=1
    #     )  # (bs, 6, 1220, 13, 2)=> (bs*6, 1220, 13, 2)

    #     features = []
    #     # (6*bs, 256, 1220, 12)
    #     # (6*bs, 256, 1220, 12)
    #     # (6*bs, 256, 1220, 12)
    #     # (6*bs, 256, 1220, 12)
    #     for fm in feature_maps:
    #         features.append(nn.functional.grid_sample(fm.flatten(end_dim=1), points_2d))
    #     features = torch.stack(features, dim=1)  # (6*bs, 4, 256, 1220, 12)
    #     features = features.reshape(
    #         bs, num_cams, num_levels, -1, num_anchor, num_pts
    #     ).permute(
    #         0, 4, 1, 2, 5, 3
    #     )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

    #     return features

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
    
    #1016 和sparsedrive一致
    def horizon_multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        feat = features.reshape(bs, num_anchor, -1, self.num_pts,
                                self.num_groups, self.group_dims)
        w = weights.reshape(bs, num_anchor, -1, self.num_pts, self.num_groups)

        N = bs * num_anchor * self.num_pts * self.num_groups
        w_bmm = w.permute(0, 1, 3, 4, 2).reshape(N, 1, -1)
        feat_bmm = feat.permute(0, 1, 3, 4, 2, 5).reshape(N, -1, self.group_dims)

        out = self.matmul_test(w_bmm, feat_bmm)
        out = out.reshape(bs, num_anchor, self.num_pts, self.num_groups, self.group_dims)
        out = out.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        return out

#-------------------------------------0930---------------------------------------
    def horizon_forward(
        self,
        instance_feature,
        anchor,
        anchor_embed,
        feature_maps,
        metas
    ):
        """
        Horizon/BPU forward.

        The BPU does not support the DAF operator, therefore this
        implementation follows the non-DAF branch of forward():

            key points
                -> fisheye projection
                -> attention weights
                -> grid_sample
                -> camera/level fusion
                -> point fusion
                -> output projection
                -> residual
        """

        bs, num_anchor = instance_feature.shape[:2]

        # =========================================================
        # 1. Generate 3D key points
        # =========================================================
        key_points = self.kps_generator(anchor, instance_feature)

        # =========================================================
        # 2. Fisheye projection
        #
        # [B, N, P, 3]
        #     ->
        # [B, C, N, P, 2]
        #
        # Keep this stage explicit, matching forward().
        # =========================================================
        
        points_2d = self.project_points_fisheye_horizon(
            key_points,
            metas["cam_extrinsic"],
            metas["cam_intrinsic"],
            metas["cam_distortion"],
            metas.get("image_wh"),
        )

        # =========================================================
        # 3. Attention weights
        #
        # [B, N, C, L, P, G]
        # =========================================================
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        # =========================================================
        # 4. Deployment-friendly feature sampling
        # =========================================================
        feature_maps_fp32 = [
                    feature_map.float() for feature_map in feature_maps
        ]
        points_2d_fp32 = points_2d.float()
        weights_fp32 = weights.float()
        features = self.feature_sampling_horizon(
            feature_maps_fp32,
            points_2d_fp32,
        )

        # =========================================================
        # 5. Camera + level fusion
        # =========================================================
        features = self.horizon_multi_view_level_fusion(
            features,
            weights_fp32,
        )

        # =========================================================
        # 6. Point fusion
        # =========================================================
        features = features.sum(dim=2)
        # print("走的else分支")

        # =========================================================
        # 后续逻辑不变
        # =========================================================
        output = self.proj_drop(
            self.output_proj(features)
        )

        if self.residual_mode == "add":
            output = output + instance_feature

        elif self.residual_mode == "cat":
            output = torch.cat(
                [output, instance_feature],
                dim=-1,
            )

        return output
#--------------------------------------------------------------------------------

    #1016地平线版本的投影
    def project_points_fisheye_horizon(
        self,
        key_points,
        lidar2cam,
        K,
        dist,
        image_wh=None,
        fill_value=-1.0,
    ):
        """
        key_points: [B, N, M, 3]
        lidar2cam:  [B, C, 4, 4]
        K:          [B, C, 3, 3]
        dist:       [B, C, 4]
        image_wh:   [B, 2] or [B, C, 2], ordered as [H, W]
        returns:    [B, C, N, M, 2], normalized as [u/W, v/H]
        """
        eps = 1e-6

        # [B,C,1,1,3,3] @ [B,1,N,M,3,1] -> [B,C,N,M,3,1]
        rotation = lidar2cam[:, :, None, None, :3, :3]
        points = key_points[:, None, :, :, :, None]
        cam_xyz = self.matmul_test(rotation, points).squeeze(-1)
        cam_xyz = cam_xyz + lidar2cam[:, :, None, None, :3, 3]

        x_cam = cam_xyz[..., 0]
        y_cam = cam_xyz[..., 1]
        z_cam = cam_xyz[..., 2]
        valid_depth = z_cam > eps

        safe_z = torch.clamp(z_cam, min=eps)
        x = x_cam / safe_z
        y = y_cam / safe_z

        radius = torch.sqrt(x * x + y * y)
        radius = torch.clamp(radius, min=eps)
        theta = torch.atan(radius)

        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        coeffs = dist[:, :, None, None, :]
        k1 = coeffs[..., 0]
        k2 = coeffs[..., 1]
        k3 = coeffs[..., 2]
        k4 = coeffs[..., 3]

        theta_distorted = theta * (
            1.0
            + k1 * theta2
            + k2 * theta4
            + k3 * theta6
            + k4 * theta8
        )
        scale = theta_distorted / radius
        x_distorted = x * scale
        y_distorted = y * scale

        fx = K[:, :, 0, 0][:, :, None, None]
        fy = K[:, :, 1, 1][:, :, None, None]
        cx = K[:, :, 0, 2][:, :, None, None]
        cy = K[:, :, 1, 2][:, :, None, None]

        u = fx * x_distorted + cx
        v = fy * y_distorted + cy
        uv = torch.stack((u, v), dim=-1)

        # 保持原实现的语义：先将无效深度点填成 fill_value，再做图像尺寸归一化。
        uv = torch.where(valid_depth.unsqueeze(-1), uv, fill_value)

        if image_wh is not None:
            if image_wh.dim() == 2:
                # [B,2]，输入顺序 [H,W]，坐标顺序是 [u,v]，所以改成 [W,H]
                wh = torch.stack(
                    (image_wh[:, 1], image_wh[:, 0]), dim=-1
                )[:, None, None, None, :]
            else:
                # [B,C,2]
                wh = torch.stack(
                    (image_wh[:, :, 1], image_wh[:, :, 0]), dim=-1
                )[:, :, None, None, :]
            uv = uv / wh

        return uv

#---------------------------------------0930-------------------------------------
    @staticmethod
    def feature_sampling_horizon(
        feature_maps: List[torch.Tensor],
        points_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deployment-friendly feature sampling.

        Args:
            feature_maps:
                [
                    [B, Cams, E, H1, W1],
                    [B, Cams, E, H2, W2],
                    [B, Cams, E, H3, W3],
                    [B, Cams, E, H4, W4],
                ]

            points_2d:
                [B, Cams, N, P, 2]

                Coordinates are normalized to [0, 1].

        Returns:
            features:
                [B, N, Cams, Levels, P, E]

        Notes:
            1. Sparse4D deployment uses fixed 4-level feature maps.
            2. Avoid dynamic Python loop over feature levels.
            3. Avoid logical_and chain.
            4. Avoid dynamic stack over Python list.
            5. Keep strict DAF valid range:
                0 < x < 1
                0 < y < 1
        """

        bs, num_cams, num_anchor, num_pts, _ = points_2d.shape
        num_levels = len(feature_maps)

        # =========================================================
        # 1. DAF的有效采样区域
        #
        # DAF CUDA中明确规定：
        #
        # if loc_w <= 0 || loc_w >= 1 return;
        # if loc_h <= 0 || loc_h >= 1 return;
        #
        # 所以这里必须严格使用 >0 且 <1
        # =========================================================
        valid_mask = (
            (points_2d[..., 0] > 0)
            & (points_2d[..., 0] < 1)
            & (points_2d[..., 1] > 0)
            & (points_2d[..., 1] < 1)
        )
        # [B, Cams, N, P]

        # =========================================================
        # 2. [0,1] -> [-1,1]
        # =========================================================
        points_2d = points_2d * 2.0 - 1.0

        # [B, Cams, N, P, 2]
        # ->
        # [B*Cams, N, P, 2]
        points_2d = points_2d.flatten(end_dim=1)

        features = []

        # =========================================================
        # 3. 每个level分别做bilinear sampling
        # =========================================================
        for fm in feature_maps:
            features.append(
                            torch.nn.functional.grid_sample(
                                fm.flatten(end_dim=1), points_2d
                            )
                        )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims


        # =========================================================
        # 5. 非法投影点直接置0
        #
        # 这个非常重要
        # =========================================================

        # [B, Cams, N, P]
        #
        # ->
        # [B, N, Cams, 1, P, 1]
        valid_mask = valid_mask.permute(
            0,
            2,
            1,
            3,
        ).unsqueeze(3).unsqueeze(-1)

        features = features * valid_mask.to(
            dtype=features.dtype
        )
        # print("features.shape =", features.shape)
        # print("valid ratio =", valid_mask.float().mean())
        # print("features abs mean =", features.abs().mean())
        # print("features abs max =", features.abs().max())

        return features
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
