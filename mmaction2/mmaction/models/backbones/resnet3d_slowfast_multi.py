import torch
import torch.nn as nn

from ...utils import get_root_logger
from ..builder import BACKBONES
from .resnet3d import ResNet3d
from .resnet3d_slowfast import ResNet3dSlowFast
from mmcv.cnn import ConvModule, kaiming_init, constant_init
from mmcv.utils import _BatchNorm
try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


@BACKBONES.register_module()
class ResNet3dSlowFastMulti(ResNet3dSlowFast):
    """
        支持 out_indices 参数的 SlowFast 模型, 即允许多个 stage 的 feature 进行输出.
        config 文件对每个 pathway 都需要传入 `out_indices` 参数去进行指定

        Note: 使用 SlowFastKeyRPNHead 去
    """
    def __init__(self, 
                 out_indices=(0,1,2,3,),
                #  out_channel=256,
                #  in_channels=(288, 576, 1152, 2304),
                 **kwargs):
        self.out_indices = out_indices
        # self.dim_reduce_convs = []
        # self.in_channels = in_channels
        # self.out_channel = out_channel
        # for i in range(len(self.out_indices)):
        #     in_channel = self.in_channels[i]

        #     dim_reduce_conv = ConvModule(
        #         in_channel,
        #         self.out_channel,
        #         (1,1,1),
        #         act_cfg=None, # 按照 FPN 的设定我也不加 ReLU 了
        #         conv_cfg=dict(type="Conv3d"),
        #         norm_cfg=None, # 同样 norm 也不加了
        #     )
        #     self.dim_reduce_convs.append(
        #         dim_reduce_conv
        #     )
    

        super(ResNet3dSlowFastMulti, self).__init__(
            **kwargs
        )
    
    # def init_weights(self, pretrained=None):
    #     super(ResNet3dSlowFastMulti, self).init_weights(pretrained)
    #     # 初始化 dim_reduce_convs
    #     for i in range(len(self.dim_reduce_convs)):
    #         module = self.dim_reduce_convs[i]
    #         for m in module.modules():
    #             if isinstance(m, nn.Conv3d):
    #                 kaiming_init(m)
    #             elif isinstance(m, _BatchNorm):
    #                 constant_init(m, 1)
                    
    def forward(self, x):
        x_slow = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = nn.functional.interpolate(
            x,
            mode='nearest',
            scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0,
                          1.0))
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        outs = []
        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)

            # 添加没有 fuse 的 feature 作为输出 feature, 否则 slowpath 最后一个 stage 的输出 channel 将会和 其余 stage 的不同, 会比较麻烦去处理.
            if i in self.out_indices:
                outs.append(
                    (x_slow, x_fast)
                )
                # x = torch.cat(
                #     (x_slow, x_fast), dim=1
                # )
                # outs.append(
                #     (self.dim_reduce_convs[i](x_slow), self.dim_reduce_convs[i](x_fast))
                # )

            if (i != len(self.slow_path.res_layers) - 1
                    and self.slow_path.lateral):
                # No fusion needed in the final stage
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)
        # 测试代码
        # for i in range(len(outs)):
        #     print(f"level: {i}, {outs[i][0].shape}, {outs[i][1].shape}")
        # exit()

        return outs


if mmdet_imported:
    MMDET_BACKBONES.register_module()(ResNet3dSlowFastMulti)
