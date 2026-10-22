# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import torch

from typing import Optional
from dataset.config.nusc_std_bbox3d import *


from dataset.config.nusc_std_bbox3d import *


class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = 0.5,   #训练时用None  #可视化改为0.5
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted
    # 将网络输出的原始参数转换为物理量
    def decode_box(self, box):
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        decoded = [
            box[:, [X, Y, Z]],
            box[:, [W, L, H]].exp(),
            yaw[:, None],
        ]
        # Avoid emitting an unsupported empty slice for velocity-free 8D boxes.
        if box.shape[-1] > VX:
            decoded.append(box[:, VX:])
        return torch.cat(decoded, dim=-1)

    def decode(
        self,
        # 分类分数 [BS, NumQueries, NumClasses]
        cls_scores,
        # 3D框参数 [BS, NumQueries, BoxDim]
        box_preds,
        track_id=None,
        qulity=None,
        output_idx=-1,
    ):
        track_id = None
        squeeze_cls = track_id is not None
        # 将logits转为概率
        cls_scores = cls_scores[output_idx].sigmoid()
        # 当需要跟踪时，强制每个预测框只保留一个最可能的类别
        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        # 展平后取Top-K预测
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        # 生成类别ID映射
        if not squeeze_cls:
            cls_ids = indices % num_cls
        # 应用置信度阈值
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold
        # 引入 Centerness 机制（类似FCOS），通过预测框的中心度来修正分类分数
        if qulity[0] is not None:
            centerness = qulity[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        # 解码循环与输出构建
        for i in range(bs):
            # 步骤1：获取当前样本的类别ID
            category_ids = cls_ids[i]
            # 步骤2：处理单类别模式（跟踪任务）
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            # 步骤3：提取当前样本的分数和框预测
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            # 步骤4：应用置信度阈值过滤
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            # 步骤5：处理质量评估参数（如centerness）
            if qulity[0] is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]
           
            # 步骤6：解码框参数（物理单位转换）
            box = self.decode_box(box)
            # 步骤7：构建输出字典
            output.append(
                {
                    "boxes_3d": box.cpu(),
                    "scores_3d": scores.cpu(),
                    "labels_3d": category_ids.cpu(),
                }
            )
            # 步骤8：添加质量评估数据
            if qulity[0] is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()
            # 步骤9：添加跟踪ID（多目标跟踪场景）
            if track_id is not None:
                ids = track_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["track_ids"] = ids
        return output
