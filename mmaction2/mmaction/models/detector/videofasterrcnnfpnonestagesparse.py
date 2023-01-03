from mmdet.models import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models import build_head, build_roi_extractor, build_loss
from mmdet.core import bbox2roi, MlvlPointGenerator, build_assigner, build_sampler
import torch
import math
from mmcv.ops import batched_nms
def inter_cal(a, b):
    inter_height_width = torch.min(a[:, None, 2:], b[None, :, 2:])-torch.max(a[:,None,:2], b[None, :, :2])
    return inter_height_width.clamp_(0).prod(dim=2)

def iou_cal(a, b):
    inter_area = inter_cal(a, b)
    a_area = (a[:, 2] - a[:, 0]).clamp_(0) * (a[:, 3]- a[:, 1]).clamp_(0)
    b_area = (b[:, 2] - b[:, 0]).clamp_(0) * (b[:, 3] - b[:, 1]).clamp_(0)
    iou = torch.where(inter_area > 0.0, inter_area / (a_area[:, None]+b_area[None, :]-inter_area+1e-10), torch.tensor(0.0))
    return iou

class VideoFasterRCNNWithFPNOneStageSparse(TwoStageDetector):
    def __init__(self,
                 backbone,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 spatial_bbox_head=None,
                 spatial_bbox_head2=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None,
                 keep_first=False):
        super(VideoFasterRCNNWithFPNOneStageSparse, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=None,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # 构建 spatial_bbox_head (e.g. FCOSHead)
        if spatial_bbox_head is not None:
            self.spatial_bbox_head = build_head(spatial_bbox_head)
            self.spatial_bbox_head.train_cfg = train_cfg 
            self.spatial_bbox_head.test_cfg = test_cfg
            if spatial_bbox_head['type'] == 'TOODHead':
                self.spatial_bbox_head.initial_epoch = self.train_cfg.initial_epoch
                self.spatial_bbox_head.initial_assigner = build_assigner(
                train_cfg.initial_assigner)
                self.spatial_bbox_head.initial_loss_cls = build_loss(spatial_bbox_head.initial_loss_cls)
                self.spatial_bbox_head.assigner = self.spatial_bbox_head.initial_assigner
                self.spatial_bbox_head.alignment_assigner = build_assigner(train_cfg.spatial_assigner)
                self.spatial_bbox_head.alpha = train_cfg.alpha
                self.spatial_bbox_head.beta = train_cfg.beta
                sampler_cfg = dict(type='PseudoSampler')
                self.spatial_bbox_head.sampler = build_sampler(sampler_cfg, context=self)
            else:
                self.spatial_bbox_head.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
                sampler_cfg = dict(type='PseudoSampler')
                self.spatial_bbox_head.sampler = build_sampler(sampler_cfg, context=self)
        else:
            assert None
        if spatial_bbox_head2 is not None:
            self.spatial_bbox_head2 = build_head(spatial_bbox_head2)
            self.spatial_bbox_head2.train_cfg = train_cfg 
            self.spatial_bbox_head2.test_cfg = test_cfg
            self.spatial_bbox_head2.assigner = build_assigner(self.train_cfg.assigner)
            self.spatial_bbox_head2.sampler = build_sampler(sampler_cfg, context=self)
        else:
            self.spatial_bbox_head2 = None
        
        self.save_result =  test_cfg.get("save_result", False)
        self.keep_first = keep_first
    #from 3D to 2D pooling
    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
        
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

        x = self.backbone(img)
        
        #x = self.extract_feat(img)

        # 判断是否是单层的 CSN 的输入
        if isinstance(x, torch.Tensor):
            x = [x]

        losses = dict()
        #reduce channel part
        #temporal pooling part
        spatial_x = self.temporal_indexing(x[:-1])

        spatial_x = self.neck(spatial_x)
        # RPN forward and loss
        #filter part
        spatial_gt_labels = [x.new_full((x.size(0),), 0, dtype=torch.long) for x in gt_labels]
        spatial_losses = self.spatial_bbox_head.forward_train(spatial_x, img_metas, gt_bboxes,
                                              spatial_gt_labels, gt_bboxes_ignore)
        losses.update(spatial_losses)
        with torch.no_grad():
            self.spatial_bbox_head.eval()
            proposal_list = self.spatial_bbox_head.simple_test(spatial_x, img_metas)
            self.spatial_bbox_head.train()

        proposal_list = [p[0] for p in proposal_list]

        if self.spatial_bbox_head2 is not None:
            spatial_loss2 = self.spatial_bbox_head2.forward_train(spatial_x, img_metas, gt_bboxes,
                                              spatial_gt_labels, gt_bboxes_ignore)
            proposal_list2 = self.spatial_bbox_head2.simple_test(spatial_x, img_metas)
            proposal_list2 = [p[0] for p in proposal_list2]
            proposal_list = [torch.cat((x,y), 0) for x,y in zip(proposal_list, proposal_list2)]
            #rename part
            spatial_loss2['loss_cls2'] = spatial_loss2['loss_cls'].clone()
            spatial_loss2['loss_bbox2'] = spatial_loss2['loss_bbox'].clone()
            spatial_loss2['loss_iou2'] = spatial_loss2['loss_iou'].clone()
            del spatial_loss2['loss_cls']
            del spatial_loss2['loss_bbox']
            del spatial_loss2['loss_iou']

            losses.update(spatial_loss2)
        
        # for i in range(len(gt_bboxes)):
        #     proposals = proposal_list[i]
        #     gt_boxes = gt_bboxes[i]
        #     new_gt_boxes = gt_boxes.new_full((gt_boxes.size(0), 5), 1)
        #     new_gt_boxes[:, 0:4] = gt_boxes 
        #     new_proposals = torch.cat((proposals, new_gt_boxes), 0)
        #     #new_proposals = new_gt_boxes
        #     proposal_list[i] = new_proposals

        # print("sup:", proposal_list, gt_bboxes)

        #proposal_list = bbox2roi(proposal_list)
        #new_proposal_list = torch.cat(proposal_list,dim=0)[:, 0:4].cpu()
        x = x[-1]
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
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

        proposal_list = self.spatial_bbox_head.simple_test(
            spatial_x, img_metas)

        if self.spatial_bbox_head2 is not None:
            proposal_list2 = self.spatial_bbox_head2.simple_test(spatial_x, img_metas)
            proposal_list2 = [p[0] for p in proposal_list2]
            proposal_list = [torch.cat((x,y), 0) for x,y in zip(proposal_list, proposal_list2)]
        x = x[-1]

        x = x[-1]
        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    # def train(self, mode=True):
    #     super(VideoFasterRCNNWithFPNOneStageSparse, self).train(mode)
    #     for key, parameter in self.named_parameters():
    #         if "roi_head" in key:
    #             print(key, parameter.requires_grad)
    #         else:
    #             parameter.requires_grad = False

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
        
        proposal_list = self.spatial_bbox_head.simple_test(
            spatial_x, img_metas)

        proposal_list = [p[0] for p in proposal_list]
        # print("unsup:", proposal_list)
        if self.spatial_bbox_head2 is not None:
            proposal_list2 = self.spatial_bbox_head2.simple_test(spatial_x, img_metas)
            proposal_list2 = [p[0] for p in proposal_list2]
            proposal_list = [torch.cat((x,y), 0) for x,y in zip(proposal_list, proposal_list2)]

        #proposal_list = bbox2roi(proposal_list)
        for i in range(len(proposal_list)):
            # scores = proposal_list[i][:, 4]
            # real_indexs = scores > self.test_cfg.real_score 
            # if real_indexs.any():
            #     pass 
            # else:
            #     real_indexs = 0
            # proposal_list[i] = proposal_list[i][real_indexs, :]
            # if len(proposal_list[i].shape) == 1:
            #     proposal_list[i] = proposal_list[i].new_zeros(1, 5)
            #     proposal_list[i][0, 2] = img_metas[0]['img_shape'][0]
            #     proposal_list[i][0, 2] = img_metas[0]['img_shape'][1]
                #proposal_list[i] = proposal_list[i].unsqueeze(0)            
            if proposal_list[i].shape[0] == 0:
                print("NO PROPOSAL!!!!!!!!")
                proposal_list[i] = proposal_list[i].new_zeros(1, 5)
            elif "real_nms" in self.test_cfg:
                old_p = proposal_list[i]
                old_p = proposal_list[i][:,0:4]
                old_scores = proposal_list[i][:,4]
                old_class = old_scores.new_full((old_scores.size(0),), 0)
                det_bboxes, keep_idxs = batched_nms(old_p, old_scores, old_class, self.test_cfg.real_nms)
                det_bboxes = det_bboxes[:self.test_cfg.real_proposal_output]
                proposals = det_bboxes
                proposal_list[i] = proposals
            
        if self.keep_first:
            for i in range(len(proposal_list)):
                # if len(proposal_list[i]) > 1:
                    # print(len(proposal_list[i]))
                proposal_list[i] = proposal_list[i][0].unsqueeze(0)
        
        # print(proposal_list)
        # exit()
        x = x[-1]
        # print(x.shape)

        if not self.save_result:
            # results = self.roi_head.simple_test(
            #     x, proposal_list, img_metas, rescale=rescale)
            # # print(results, img_metas, len(results), len(results[0]))
            # # print(results[0], len(results[0]))
            # assert len(results) == 1
            # length = [len(i) for i in results[0]]
            # assert sum(length) == 1, f"length: {length}, {results[0]}"
            # # exit()
            # return results
            return self.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
        else:
            if isinstance(x, tuple) or isinstance(x, list):
                x_shape = x[0].shape
            else:
                x_shape = x.shape
            
            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)
            # print(proposal_list, len(proposal_list), type(proposal_list))
            # exit()
            det_bboxes, det_labels =  self.roi_head.simple_test_bboxes(
                x, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale)
            print(det_bboxes, proposal_list, img_metas)
            exit()
            return det_bboxes, det_labels, proposal_list
        
    def simple_test_bboxes(self, img, img_metas, proposals=None, rescale=False):
        """
            不进行 bbox2result 的计算
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.backbone(img)

        if isinstance(x, torch.Tensor):
            x = [x]
        
        spatial_x = self.temporal_indexing(x[:-1])
        spatial_x = self.neck(spatial_x)
        
        self.spatial_bbox_head.eval()
        proposal_list = self.spatial_bbox_head.simple_test(
            spatial_x, img_metas)

        proposal_list = [p[0] for p in proposal_list]
        # print("unsup:", proposal_list)
        if self.spatial_bbox_head2 is not None:
            proposal_list2 = self.spatial_bbox_head2.simple_test(spatial_x, img_metas)
            proposal_list2 = [p[0] for p in proposal_list2]
            proposal_list = [torch.cat((x,y), 0) for x,y in zip(proposal_list, proposal_list2)]

        #proposal_list = bbox2roi(proposal_list)
        for i in range(len(proposal_list)):
            # scores = proposal_list[i][:, 4]
            # real_indexs = scores > self.test_cfg.real_score 
            # if real_indexs.any():
            #     pass 
            # else:
            #     real_indexs = 0
            # proposal_list[i] = proposal_list[i][real_indexs, :]
            # if len(proposal_list[i].shape) == 1:
            #     proposal_list[i] = proposal_list[i].new_zeros(1, 5)
            #     proposal_list[i][0, 2] = img_metas[0]['img_shape'][0]
            #     proposal_list[i][0, 2] = img_metas[0]['img_shape'][1]
                #proposal_list[i] = proposal_list[i].unsqueeze(0)            
            if proposal_list[i].shape[0] == 0:
                print("NO PROPOSAL!!!!!!!!")
                proposal_list[i] = proposal_list[i].new_zeros(1, 5)
            elif "real_nms" in self.test_cfg:
                old_p = proposal_list[i]
                old_p = proposal_list[i][:,0:4]
                old_scores = proposal_list[i][:,4]
                old_class = old_scores.new_full((old_scores.size(0),), 0)
                det_bboxes, keep_idxs = batched_nms(old_p, old_scores, old_class, self.test_cfg.real_nms)
                det_bboxes = det_bboxes[:self.test_cfg.real_proposal_output]
                proposals = det_bboxes
                proposal_list[i] = proposals
        x = x[-1]

        if isinstance(x, tuple) or isinstance(x, list):
            x_shape = x[0].shape
        else:
            x_shape = x.shape
        
        assert x_shape[0] == 1, 'only accept 1 sample at test mode'
        assert x_shape[0] == len(img_metas) == len(proposal_list)
        # print(proposal_list, len(proposal_list), type(proposal_list))
        # exit()
        det_bboxes, det_labels =  self.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, self.roi_head.test_cfg, rescale=rescale)
        return det_bboxes, det_labels, proposal_list

DETECTORS.register_module()(VideoFasterRCNNWithFPNOneStageSparse)
