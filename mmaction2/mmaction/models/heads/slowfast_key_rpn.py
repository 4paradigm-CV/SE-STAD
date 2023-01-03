import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, kaiming_init, constant_init
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
class SlowFastKeyRPNHead(RPNHead):
    """
        using Key Frame to extract roi proposals
    """
    # def __init__(self, 
    #              in_channels,
    #              init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
    #              num_convs=1,
    #              **kwargs):
    #     super(SlowFastKeyRPNHead, self).__init__(
    #         1, in_channels, init_cfg=init_cfg, **kwargs
    #     )
    def __init__(self,
                 reduce_out_channel=256,
                 reduce_in_channels=(288,576,1152,2304),
                 **kwargs):
        self.reduce_in_channels = reduce_in_channels
        self.reduce_out_channel = reduce_out_channel
        self.dim2ind = {}
        
        super(SlowFastKeyRPNHead, self).__init__(
            **kwargs
        )
    
    def _init_layers(self):
        super(SlowFastKeyRPNHead, self)._init_layers()

        for i in range(len(self.reduce_in_channels)):
            layer_name = f'reduce_conv{i+1}'
            in_channel = self.reduce_in_channels[i]
            # 是一个 2d 卷积, 使用 nn.Conv2d 实现吧, 使用默认的初始化参数
            dim_reduce_conv = nn.Conv2d(
                in_channel,
                self.reduce_out_channel,
                1
            )
            self.add_module(layer_name, dim_reduce_conv)
            self.dim2ind[in_channel] = layer_name

            # dim_reduce_conv = ConvModule(
            #     in_channel,
            #     self.out_channel,
            #     (1,1),
            #     act_cfg=None, # 按照 FPN 的设定我也不加 ReLU 了
            #     norm_cfg=None, # 同样 norm 也不加了
            # )
    
    def forward_single(self, x):
        """
            # 如果使用的是非 SlowFast 这种网络的话就是
            x: B x C x T x H x W
        """
        assert isinstance(x, tuple)
        # 针对 SlowFast
        # SlowFast 从 slow 和 fast 两个 pathway 分别提取 key frame 位置的 feature, 然后进行 concate
        middle1 = x[0].shape[2] // 2
        middle2 = x[1].shape[2] // 2
        x1 = x[0][:, :, middle1, :, :].squeeze()
        x2 = x[1][:, :, middle2, :, :].squeeze()
        x = torch.cat((x1, x2), dim=1)

        # print(f"before: {x.shape}")
        reduce_conv = getattr(self, self.dim2ind[x.shape[1]])
        x = reduce_conv(x)
        # print(f"after: {x.shape}")
            
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred




if mmdet_imported:
    MMDET_HEADS.register_module()(SlowFastKeyRPNHead)