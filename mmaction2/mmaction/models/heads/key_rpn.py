import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms

from ..builder import HEADS

# 从 mmdet 中引入 AnchorHead 以及 MMDET_HEADS
try:
    from mmdet.models.dense_heads.anchor_head import AnchorHead
    from mmdet.models.dense_heads.rpn_head import RPNHead

    from mmdet.models import HEADS as MMDET_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

assert mmdet_imported, "mmdet not found!"

@HEADS.register_module()
class KeyRPNHead(RPNHead):
    """
        using Key Frame to extract roi proposals
    """
    # def __init__(self, 
    #              in_channels,
    #              init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
    #              num_convs=1,
    #              **kwargs):
    #     super(KeyRPNHead, self).__init__(
    #         1, in_channels, init_cfg=init_cfg, **kwargs
    #     )
    
    def forward_single(self, x):
        """
            # 如果使用的是非 SlowFast 这种网络的话就是
            x: B x C x T x H x W
        """
        # print(type(x), len(x))
        # 针对 SlowFast
        if isinstance(x, tuple):
            # SlowFast 从 slow 和 fast 两个 pathway 分别提取 key frame 位置的 feature, 然后进行 concate
            middle1 = x[0].shape[2] // 2
            middle2 = x[1].shape[2] // 2
            x1 = x[0][:, :, middle1, :, :].squeeze(dim=2)
            x2 = x[1][:, :, middle2, :, :].squeeze(dim=2)
            x = torch.cat((x1, x2), dim=1)
        else:
            # 单分支的模型只需要提取 key frame 对应位置的 feature
            middle = x.shape[2] // 2
            x = x[:, :, middle, :, :].squeeze(dim=2)
            
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred




if mmdet_imported:
    MMDET_HEADS.register_module()(KeyRPNHead)