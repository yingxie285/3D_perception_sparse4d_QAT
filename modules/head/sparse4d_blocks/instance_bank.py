# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from .sparse3d_embedding import *
# 0922 需要改！
try:
    import os
    from horizon_plugin_pytorch.quantization import QuantStub
    from torch.quantization import DeQuantStub
    HORIZON_QAT = os.environ.get("HORIZON_QAT", "0") == "1"
except:
    HORIZON_QAT = 0
    pass

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    """
    confidence  : torch.tensor, shape(bs, num_querys)
    inputs:
        instance_feature : torch.tensor, shape(bs, num_querys, 256)
        anchor  : torch.tensor, shape(bs, num_querys, 10)
        cls     : torch.tensor, shape(bs, num_querys, 11)
    """
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)  # (bs, k), (bs, k)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(
        -1
    )  # (bs, k) + (k, 1) => (bs, k) => (bs*k,)
    outputs = []
    # (bs, num_querys, c) => (bs, k, 256)
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs, indices

#--------------------------xy--------------------------
def horizon_topk(confidence, k, *inputs):

    confidence, indices = torch.topk(confidence, k, dim=1)

    indices_i64 = indices.to(torch.int64)

    outputs = []

    for input in inputs:

        if input.dim() == 2:
            output = torch.gather(
                input,
                1,
                indices_i64,
            )

        else:
            gather_index = indices_i64.unsqueeze(-1).expand(
                -1,
                -1,
                input.size(-1),
            )

            output = torch.gather(
                input,
                1,
                gather_index,
            )

        outputs.append(output)

    return confidence, outputs, indices
#------------------------------------------------------------

class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        # =========== build modules ===========
        def build_module(cfg):
            cfg2 = cfg.copy()
            type = cfg2.pop("type")
            return eval(type)(**cfg2)

        if anchor_handler is not None:
            anchor_handler = build_module(anchor_handler)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()
        
        # 0922
        if HORIZON_QAT:
            self.horizon_init()

#---------------------------------------0922--------------------------------
    def horizon_init(self, ):
        self.anchor_quant = QuantStub()
        self.instance_feature_quant = QuantStub()
        self.cache_instance_feature_quant = QuantStub()
        self.cache_anchor_quant = QuantStub()
        self.time_interval_quant = QuantStub()
        self.T_temp2cur_quant = QuantStub()
#---------------------------------------------------------------------------

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None  # fusioned and unordered
        self.temp_confidence = None  # fusioned. unordered and topk
        self.temp_topk_indice = None
        self.track_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None, dn_metas=None):
        """
        Return:
            instance_feature : Tensor.shape(bs, 900, 25)
            anchor : Tensor.shape(bs, 900, 11)
            self.cached_feature: None or
            self.cached_anchor: None or
            time_interval: TensorShape (bs, )
        """
        # 直接清空 cached anchor 和 feature  添加
        self.cached_anchor = None  #添加
        self.cached_feature = None #添加
        
        instance_feature = self.instance_feature[None].repeat((batch_size, 1, 1))
        anchor = self.anchor[None].repeat((batch_size, 1, 1))

        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval  # < 2s

            if self.anchor_handler is not None:
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["global2lidar"]
                            @ self.metas["img_metas"][i]["lidar2global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )  # (1, 4, 4)
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor,
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]

            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):  # train mode step in
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

#--------------------------------------1016 horizon_get-----------------------
    def horizon_get(
        self,
        batch_size,
        time_interval=None,
        cached_feature=None,
        cached_anchor=None
    ):
        # 插quant
        # instance_feature_base = self.instance_feature_quant(self.instance_feature) #TODO
        # anchor_base = self.anchor_quant(self.anchor)  #TODO
        
        # 直接清空 cached anchor 和 feature  添加
        self.cached_anchor = None  #添加
        self.cached_feature = None #添加
        
        # instance_feature = torch.tile(
        #     instance_feature_base[None], (batch_size, 1, 1)
        # )  #和sparsedrive保持一致
        # anchor = torch.tile(anchor_base[None], (batch_size, 1, 1))
        
        instance_feature = self.instance_feature[None].repeat((batch_size, 1, 1)) #这是和浮点的get函数一样
        anchor = self.anchor[None].repeat((batch_size, 1, 1))  #这是和浮点的get函数一样
        
        return (
            instance_feature,
            anchor,
            cached_feature,
            cached_anchor,
            time_interval
        )
#-----------------------------------------------------------------------------

    def update(self, instance_feature, anchor, confidence):
        """ "
        Args:
            instance_feature: TensorShape(bs, 1220, 256)
            anchor: TensorShape(bs, 1220, 11)
        Return:

        """
        if self.cached_feature is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor), _ = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        if self.track_id is not None:
            self.track_id = torch.where(
                self.mask[:, None],
                self.track_id,
                self.track_id.new_tensor(-1),
            )

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

# --------------------------------------0921 horizon_update-------------------
    def horizon_update(
        self,
        instance_feature,
        anchor,
        confidence,
        cached_feature,
        cached_anchor,
        time_mask,
        cache_confidence,
        cache_track_id,
    ):
        if cached_feature is None:
            return instance_feature, anchor, cache_track_id

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor), _ = horizon_topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat(
            [cached_feature, selected_feature], dim=1
        )

        selected_anchor = torch.cat(
            [cached_anchor, selected_anchor], dim=1,
        )
        # ---------------------------------------------------------
        # replace torch.where for instance feature
        #
        # original:
        # instance_feature = torch.where(
        #     self.mask[:, None, None],
        #     selected_feature,
        #     instance_feature,
        # )
        # ---------------------------------------------------------
        tm_feat = time_mask[:, None, None].to(instance_feature.dtype)
        instance_feature = selected_feature * tm_feat + instance_feature * (1 - tm_feat)
        tm_anchor = time_mask[:, None, None].to(anchor.dtype)
        anchor = selected_anchor * tm_anchor + anchor * (1 - tm_anchor)
        #---------------------------------------------------------
        # track id temporal state
        #
        # original:
        # self.track_id = torch.where(
        #     self.mask[:, None],
        #     self.track_id,
        #     self.track_id.new_tensor(-1),
        # )
        #---------------------------------------------------------
        if cache_track_id is not None:
            track_id_dtype = cache_track_id.dtype
            # Horizon/ORT-friendly integer arithmetic
            track_id_int32 = cache_track_id.to(torch.int32)
            tm_id = time_mask[:, None].to(torch.int32)
            cache_track_id = (track_id_int32 * tm_id + torch.full_like(track_id_int32, -1) * (1 - tm_id)).to(track_id_dtype)

        if num_dn > 0:
            instance_feature = torch.cat(
                [instance_feature, dn_instance_feature,], dim=1,
            )
            anchor = torch.cat([anchor, dn_anchor], dim=1,)

        return instance_feature, anchor, cache_track_id
#-----------------------------------------------------------------------------

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence = confidence.max(dim=-1).values.sigmoid()  # (B, num_querys)
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (
            self.confidence,
            (self.cached_feature, self.cached_anchor),
            self.temp_topk_indice,
        ) = topk(confidence, self.num_temp_instances, instance_feature, anchor)

#----------------------------------0921 horizon_cache-------------------------
    def horizon_cache(
        self,
        instance_feature,
        anchor,
        confidence,
        cache_confidence,
        # cache_track_id,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        confidence = confidence.max(dim=-1).values.sigmoid()
        if cache_confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                cache_confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )

        self.temp_confidence = confidence

        (
            cache_confidence,
            (cached_feature, cached_anchor),
            temp_topk_indice,
        ) = horizon_topk(
            confidence,
            self.num_temp_instances,
            instance_feature,
            anchor
        )

        return (
            cache_confidence,
            cached_feature,
            cached_anchor,
            temp_topk_indice,
        )
#-----------------------------------------------------------------------------

    def get_track_id(self, confidence, threshold):
        # def get_track_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()  # (bs, num_querys)
        track_id = confidence.new_full(confidence.shape[:2], -1).long()

        if self.track_id is not None and self.track_id.shape[0] == track_id.shape[0]:
            track_id[:, : self.track_id.shape[1]] = self.track_id

        mask = track_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = mask.sum()
        new_ids = torch.arange(num_new_instance).to(track_id) + self.prev_id
        track_id[torch.where(mask)] = new_ids
        self.prev_id += num_new_instance
        # self.update_track_id(track_id, confidence)
        self.update_track_id(track_id)
        return track_id

#-----------------------------0921 horizon_get_track_id-----------------------
    def horizon_get_track_id(
        self,
        confidence,
        cache_track_id,
        temp_topk_indice,
        threshold=None,
        prev_id=None,
    ):
        confidence = confidence.max(dim=-1).values.sigmoid()
        bs, num_anchor = confidence.shape
        track_id = torch.full(
            (bs, num_anchor),
            -1,
            dtype=torch.int16,
            device=confidence.device,
        )

        if (
            cache_track_id is not None
            and cache_track_id.shape[0] == track_id.shape[0]
        ):
            track_id[:, : cache_track_id.shape[1]] = cache_track_id

        mask = track_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)

        num_new_per_batch = mask.to(torch.float16).sum(dim=1).to(torch.int32)

        prev_id_int32 = prev_id.to(torch.int32)

        id_pool = torch.arange(self.num_anchor, dtype=torch.int32, device=track_id.device).unsqueeze(0) + prev_id_int32

        id_pool = id_pool.expand(bs, -1,)
        # ---------------------------------------------------------
        # Generate index of each new instance.
        #
        # Example:
        # mask = [0, 1, 0, 1, 1]
        #
        # cumsum:
        #        [0, 1, 1, 2, 3]
        #
        # idx:
        #        [0, 0, 0, 1, 2]
        # ---------------------------------------------------------
        cumsum_result = torch.cumsum(mask.to(torch.float16), dim=1)

        idx = torch.clamp(cumsum_result.to(torch.int32) - 1, min=0)

        new_ids_full = torch.gather(
            id_pool,
            dim=1,
            index=idx,
        )
        # ---------------------------------------------------------
        # Replace:
        #
        # track_id[torch.where(mask)] = new_ids
        #
        # Avoid torch.where / dynamic indexing.
        # ---------------------------------------------------------
        mask_int = mask.to(torch.int32)

        track_id = (new_ids_full * mask_int + track_id.to(torch.int32) * (1 - mask_int)).to(torch.int16)

        new_prev_id = prev_id_int32 + num_new_per_batch.sum().to(torch.int32)

        cache_track_id = self.horizon_update_track_id(
            track_id,
            temp_topk_indice,
        )

        return (
            track_id,
            cache_track_id,
            new_prev_id,
        )
#-----------------------------------------------------------------------------

    def update_track_id(self, track_id=None):
        bs = track_id.shape[0]
        track_id = track_id.flatten(end_dim=1)[self.temp_topk_indice].reshape(bs, -1)
        self.track_id = F.pad(
            track_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )  # (bs, num_querys)
    
    # def horizon_update_track_id(self, track_id, temp_topk_indice):

    #     # 获取用于时序 TopK 的置信度
    #     if self.temp_confidence is None:
    #         if confidence.dim() == 3:  # bs, num_anchor, num_cls
    #             temp_conf = confidence.max(dim=-1).values
    #         else:  # bs, num_anchor
    #             temp_conf = confidence
    #     else:
    #         temp_conf = self.temp_confidence

    #     # 根据置信度选出需要缓存的 track id
    #     track_id = horizon_topk(
    #         temp_conf,
    #         self.num_temp_instances,
    #         track_id,
    #     )[1][0]

    #     # 避免 squeeze(dim=-1)
    #     # horizon_topk 对 track_id 处理后可能为 [bs, k, 1]
    #     # 直接 reshape 成 [bs, k]
    #     track_id = track_id.reshape(
    #         track_id.shape[0],
    #         -1,
    #     )

    #     # 不再使用 F.pad(int16)
    #     # 用 torch.cat + 全 -1 Tensor 等效实现
    #     pad_len = self.num_anchor - self.num_temp_instances

    #     if pad_len > 0:
    #         pad_tensor = torch.full(
    #             track_id.shape[:-1] + (pad_len,),
    #             -1,
    #             dtype=track_id.dtype,
    #             device=track_id.device,
    #         )

    #         cache_track_id = torch.cat(
    #             [track_id, pad_tensor],
    #             dim=-1,
    #         )
    #     else:
    #         cache_track_id = track_id

    #     return cache_track_id
    def horizon_update_track_id(
        self,
        track_id=None,
        temp_topk_indice=None,
    ):
        if temp_topk_indice is None:
            return track_id
        bs = track_id.shape[0]

        track_id = track_id.flatten(
            end_dim=1
        )[temp_topk_indice].reshape(
            bs,
            -1,
        )

        pad_len = (
            self.num_anchor
            - self.num_temp_instances
        )

        if pad_len > 0:

            pad_tensor = torch.full(
                track_id.shape[:-1] + (pad_len,),
                -1,
                dtype=track_id.dtype,
                device=track_id.device,
            )


            cache_track_id = torch.cat(
                [
                    track_id,
                    pad_tensor,
                ],
                dim=-1,
            )
        else:

            cache_track_id = track_id

        return cache_track_id