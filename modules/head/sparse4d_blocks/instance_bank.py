# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from .sparse3d_embedding import *

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

#-----------------------用到了gather-------------------------
def horizon_topk(confidence, k, *inputs):

    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = indices.to(torch.int32)
    indices_i64 = indices.to(torch.int64)
    outputs = []
    for input in inputs:
        # 用 gather 替代 flatten+arange(bs)+index，避免 arange(动态 bs) 产生 If 节点
        if input.dim() == 2:
            outputs.append(torch.gather(input, 1, indices_i64))
        else:
            outputs.append(torch.gather(input, 1, indices_i64.unsqueeze(-1).expand(-1, -1, input.size(-1))))
    return confidence, outputs
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
        # self.cache_instance_feature_quant = QuantStub()
        # self.cache_anchor_quant = QuantStub()
        # self.time_interval_quant = QuantStub()
        # self.T_temp2cur_quant = QuantStub()
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
        instance_feature_base = self.instance_feature_quant(self.instance_feature) #TODO
        anchor_base = self.anchor_quant(self.anchor)  #TODO
        
        # 直接清空 cached anchor 和 feature  添加
        self.cached_anchor = None  #添加
        self.cached_feature = None #添加
        
        instance_feature = torch.tile(
            instance_feature_base[None], (batch_size, 1, 1)
        )  #和sparsedrive保持一致
        anchor = torch.tile(anchor_base[None], (batch_size, 1, 1))
        
        # instance_feature = self.instance_feature[None].repeat((batch_size, 1, 1)) #这是和浮点的get函数一样
        # anchor = self.anchor[None].repeat((batch_size, 1, 1))  #这是和浮点的get函数一样
        
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
    def horizon_update(self, instance_feature, anchor, confidence):
        """ "
        Args:
            instance_feature: TensorShape(bs, 1220, 256)
            anchor: TensorShape(bs, 1220, 11)
        Return:

        """
        if self.cached_feature is None:
            return instance_feature, anchor
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

#--------------------------用horizon_topk-------------------------
    def horizon_cache(
        self,
        instance_feature,
        anchor,
        confidence,
        cache_confidence,
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
        ) = horizon_topk(confidence, self.num_temp_instances, instance_feature, anchor)
        return cache_confidence, cached_feature, cached_anchor
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
    def horizon_get_instance_id(self, confidence, cache_instance_id, anchor=None, threshold=None, prev_id=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        bs, num_anchor = confidence.shape
        instance_id = torch.full((bs, num_anchor), -1, dtype=torch.int16, device=confidence.device)

        if (
            cache_instance_id is not None
            and cache_instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : cache_instance_id.shape[1]] = cache_instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        # BPU: reduce_sum only supports si8/si16/f16, cumsum requires quantized input
        # Use float16 for both sum and cumsum, avoid si64 throughout
        num_new_per_batch = mask.to(torch.float16).sum(dim=1).to(torch.int32)  # (bs,), f16 sum → si32
        # Fixed-size ID pool (no .item(), fully traceable)
        id_pool = torch.arange(self.num_anchor, dtype=torch.int32, device=instance_id.device).unsqueeze(0) + prev_id.to(torch.int32)  # (1, num_anchor)
        id_pool = id_pool.expand(bs, -1)  # (bs, num_anchor)
        cumsum_result = torch.cumsum(mask.to(torch.float16), dim=1)  # f16 quantized cumsum on BPU
        idx = torch.clamp(cumsum_result.to(torch.int32) - 1, min=0)  # si32 for gather
        new_ids_full = torch.gather(id_pool, 1, idx)  # (bs, num_anchor), si32
        # Avoid torch.where (produces Where(16) node unsupported by hb_compile's ORT).
        # For int32, where(mask, X, Y) ≡ X*mask + Y*(1-mask), numerically equivalent.
        mask_int = mask.to(torch.int32)
        instance_id = (new_ids_full * mask_int + instance_id.to(torch.int32) * (1 - mask_int)).to(torch.int16)

        new_prev_id = prev_id.to(torch.int32) + num_new_per_batch.sum().to(torch.int32)
        cache_instance_id = self.horizon_update_instance_id(instance_id, confidence)
        return instance_id, cache_instance_id, new_prev_id
#-----------------------------------------------------------------------------

    def update_track_id(self, track_id=None):
        bs = track_id.shape[0]
        track_id = track_id.flatten(end_dim=1)[self.temp_topk_indice].reshape(bs, -1)
        self.track_id = F.pad(
            track_id,
            (0, self.num_anchor - self.num_temp_instances),
            value=-1,
        )  # (bs, num_querys)
    
    def horizon_update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = horizon_topk(temp_conf, self.num_temp_instances, instance_id)[1][
            0
        ]
        # 避免 squeeze(dim=-1)：动态 shape 下 trace 会生成 If 节点，hmct shape inference 不支持。
        # horizon_topk 对 2D input 返回 [bs, k]，k=num_temp_instances>1，无需 squeeze。
        # 用 reshape 明确目标 shape，避免 If。
        instance_id = instance_id.reshape(instance_id.shape[0], -1)
        # ONNX F.pad(int16) 产生 Pad(13) 节点，onnxruntime CPU EP 未注册 INT16 kernel，
        # 用 cat + 预构造 -1 填充张量替代，语义等价。
        pad_len = self.num_anchor - self.num_temp_instances
        if pad_len > 0:
            pad_tensor = torch.full(
                instance_id.shape[:-1] + (pad_len,),
                -1,
                dtype=instance_id.dtype,
                device=instance_id.device,
            )
            cache_instance_id = torch.cat([instance_id, pad_tensor], dim=-1)
        else:
            cache_instance_id = instance_id
        return cache_instance_id