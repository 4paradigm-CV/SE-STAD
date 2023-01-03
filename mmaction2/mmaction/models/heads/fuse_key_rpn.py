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
class FuseKeyRPNHead(RPNHead):
    """
        using Key Frame to extract roi proposals
    """
    def __init__(self, 
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 fuse_type="temporal_average",
                 out_put_channel=256,
                 **kwargs):
        self.fuse_type = fuse_type
        assert self.fuse_type in ["temporal_average", "temporal_max", "concate_then_conv", "conv_then_concate"]
        super(FuseKeyRPNHead, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs
        )
    
    # TODO: 暂时还没有实现
    def forward_single(self, x):
        """
            x: B x C x T x H x W
        """
        # 提取 key frame
        middle = x.shape[2] // 2
        x = x[:, :, middle, :, :].squeeze()
        
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred




if mmdet_imported:
    MMDET_HEADS.register_module()(FuseKeyRPNHead)