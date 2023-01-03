from mmdet.models import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
import torch
import math
def inter_cal(a, b):
    inter_height_width = torch.min(a[:, None, 2:], b[None, :, 2:])-torch.max(a[:,None,:2], b[None, :, :2])
    return inter_height_width.clamp_(0).prod(dim=2)

def iou_cal(a, b):
    inter_area = inter_cal(a, b)
    a_area = (a[:, 2] - a[:, 0]).clamp_(0) * (a[:, 3]- a[:, 1]).clamp_(0)
    b_area = (b[:, 2] - b[:, 0]).clamp_(0) * (b[:, 3] - b[:, 1]).clamp_(0)
    iou = torch.where(inter_area > 0.0, inter_area / (a_area[:, None]+b_area[None, :]-inter_area+1e-10), torch.tensor(0.0))
    return iou

class VideoFasterRCNNWithFPNE2E(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(VideoFasterRCNNWithFPNE2E, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
    #from 3D to 2D pooling
    def temporal_indexing(self, input_x):
        """
            # 如果使用的是非 SlowFast 这种网络的话就是
            x: B x C x T x H x W
        """
        # print(type(x), len(x))
        # 针对 SlowFast
        new_x = []
        for i in range(len(input_x)):
            x = input_x[i]
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
                # x = torch.mean(x, 2)
            new_x.append(x)
        return new_x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # C, T = img[0].size(0), img[0].size(1)
        # max_H = 0
        # max_W = 0
        # for i in range(len(img)):
        #     max_H, max_W = max(max_H, img[i].size(2)), max(max_W, img[i].size(3))
        # max_H = math.ceil(max_H / 32) * 32
        # max_W = math.ceil(max_W / 32) * 32
        # for i in range(len(img)):
        #     pad_H = max_H - img[i].size(2)
        #     pad_W = max_W - img[i].size(3)
        #     pad_const = (0, pad_W, 0, pad_H)
        #     img[i] = torch.nn.functional.pad(img[i], pad_const, "constant", 0).unsqueeze(0)
        # img = torch.cat(img, 0)
        # for meta_info in img_metas:
        #     meta_info['pad_shape'] = (max_H, max_W)

        x = self.backbone(img)
        
        #x = self.extract_feat(img)

        # 判断是否是单层的 CSN 的输入
        if isinstance(x, torch.Tensor):
            x = [x]

        losses = dict()
        #reduce channel part
        #temporal pooling part
        spatial_x = self.temporal_indexing(x[:-1])
        spatial_x = self.neck(spatial_x) # 2d FPN 提取 feature
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                spatial_x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        
        # 计算 match 的比例
        new_proposal_list = torch.cat(proposal_list,dim=0)[:, 0:4].cpu()
        new_gt = torch.cat(gt_bboxes, dim=0).cpu()
        new_area = iou_cal(new_proposal_list, new_gt).max(dim=0)[0]
        match_ratio = torch.Tensor([new_area[new_area > 0.5].size(0) / new_gt.size(0)]).to(x[0].get_device())
        iou_dict = dict(iou_rate=match_ratio)
        losses.update(iou_dict)

        x = x[-1]
        roi_losses = self.roi_head.forward_train(x, spatial_x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses
    
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        # print("sync test...")
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        x = self.backbone(img)
        
        if isinstance(x, torch.Tensor):
            x = [x]

        spatial_x = self.temporal_indexing(x[:-1])
        spatial_x = self.neck(spatial_x)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                spatial_x, img_meta)
        else:
            proposal_list = proposals

        x = x[-1]
        return await self.roi_head.async_simple_test(
            x, spatial_x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        # print("test...")
        # C, T = img.size(0), img.size(1)
        # max_H, max_W = img.size(2), img.size(3)
        # max_H = math.ceil(max_H / 32) * 32
        # max_W = math.ceil(max_W / 32) * 32
        # pad_const = (0, max_W - img.size(3), 0, max_H - img.size(2))
        # img = torch.nn.functional.pad(img, pad_const, "constant", 0)

        # for meta_info in img_metas:
        #     meta_info['pad_shape'] = (max_H, max_W)
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.backbone(img)

        if isinstance(x, torch.Tensor):
            x = [x]
        
        spatial_x = self.temporal_indexing(x[:-1])
        spatial_x = self.neck(spatial_x)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(spatial_x, img_metas)
        else:
            proposal_list = proposals


        x = x[-1]
        return self.roi_head.simple_test(
            x, spatial_x, proposal_list, img_metas, rescale=rescale)

DETECTORS.register_module()(VideoFasterRCNNWithFPNE2E)
