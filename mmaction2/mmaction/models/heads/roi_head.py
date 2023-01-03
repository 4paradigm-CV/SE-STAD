# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmaction.core.bbox import bbox2result
import torch
try:
    from mmdet.core.bbox import bbox2roi
    from mmdet.models import HEADS as MMDET_HEADS
    from mmdet.models import build_head, build_roi_extractor
    from mmdet.models.roi_heads import StandardRoIHead
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

if mmdet_imported:

    @MMDET_HEADS.register_module()
    class AVARoIHead(StandardRoIHead):

        def _bbox_forward(self, x, rois, img_metas):
            """Defines the computation performed to get bbox predictions.

            Args:
                x (torch.Tensor): The input tensor.
                rois (torch.Tensor): The regions of interest.
                img_metas (list): The meta info of images

            Returns:
                dict: bbox predictions with features and classification scores.
            """
            bbox_feat, global_feat = self.bbox_roi_extractor(x, rois)

            if self.with_shared_head:
                bbox_feat = self.shared_head(
                    bbox_feat,
                    feat=global_feat,
                    rois=rois,
                    img_metas=img_metas)

            cls_score, bbox_pred = self.bbox_head(bbox_feat)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feat)
            return bbox_results

        def _bbox_forward_train(self, x, sampling_results, gt_bboxes,
                                gt_labels, img_metas):
            """Run forward function and calculate loss for box head in
            training."""
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois, img_metas)

            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)

            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results

        def simple_test(self,
                        x,
                        proposal_list,
                        img_metas,
                        proposals=None,
                        rescale=False):
            """Defines the computation performed for simple testing."""
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple) or isinstance(x, list):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                thr=self.test_cfg.action_thr)
            return [bbox_results]

        def simple_test_bboxes(self,
                               x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
            """Test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, rois, img_metas)
            cls_score = bbox_results['cls_score']

            img_shape = img_metas[0]['img_shape']
            crop_quadruple = np.array([0, 0, 1, 1])
            flip = False

            if 'crop_quadruple' in img_metas[0]:
                crop_quadruple = img_metas[0]['crop_quadruple']

            if 'flip' in img_metas[0]:
                flip = img_metas[0]['flip']

            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                img_shape,
                flip=flip,
                crop_quadruple=crop_quadruple,
                cfg=rcnn_test_cfg)

            return det_bboxes, det_labels

    @MMDET_HEADS.register_module()
    class AVARoIE2EHead(StandardRoIHead):

        def __init__(self,
                        bbox_roi_extractor=None,
                        bbox_head=None,
                        spatial_roi_extractor=None,
                        spatial_bbox_head=None,
                        mask_roi_extractor=None,
                        mask_head=None,
                        shared_head=None,
                        train_cfg=None,
                        test_cfg=None,
                        pretrained=None,
                        init_cfg=None):
            super(AVARoIE2EHead, self).__init__(
                bbox_roi_extractor=bbox_roi_extractor,
                bbox_head=bbox_head,
                mask_roi_extractor=mask_roi_extractor,
                mask_head=mask_head,
                shared_head=shared_head,
                train_cfg=train_cfg,
                test_cfg=test_cfg,
                pretrained=pretrained,
                init_cfg=init_cfg)
            if spatial_bbox_head is not None:
                self.spatial_roi_extractor = build_roi_extractor(spatial_roi_extractor)
                self.spatial_bbox_head = build_head(spatial_bbox_head)
        def _bbox_forward(self, x, spatial_x, rois, img_metas):
            """Defines the computation performed to get bbox predictions.

            Args:
                x (torch.Tensor): The input tensor.
                rois (torch.Tensor): The regions of interest.
                img_metas (list): The meta info of images

            Returns:
                dict: bbox predictions with features and classification scores.
            """
            #refine for final boxes part
            spatial_feat = self.spatial_roi_extractor(spatial_x, rois)
            _, bbox_pred = self.spatial_bbox_head(spatial_feat)
            #decode

            if not self.spatial_bbox_head.reg_decoded_bbox:
                new_bbox_pred = self.spatial_bbox_head.bbox_coder.decode(rois[:, 1:], bbox_pred)
            else:
                new_bbox_pred = bbox_pred
            # new_bbox_pred[:, [0, 2]].clamp_(min=0, max=img_metas[0]['img_shape'][1])
            # new_bbox_pred[:, [1, 3]].clamp_(min=0, max=img_metas[0]['img_shape'][0])
            # new_rois = torch.cat((rois[:,0].unsqueeze(1),new_bbox_pred),dim=1)
            new_rois = rois.detach().clone()
            bbox_feat, global_feat = self.bbox_roi_extractor(x, new_rois)

            if self.with_shared_head:
                bbox_feat = self.shared_head(
                    bbox_feat,
                    feat=global_feat,
                    rois=rois,
                    img_metas=img_metas)

            cls_score, _ = self.bbox_head(bbox_feat)

            bbox_results = dict(
                cls_score=cls_score, spatial_bbox_pred=bbox_pred, final_pred=new_rois, bbox_feats=bbox_feat)
            return bbox_results
        def forward_train(self,
                      x,
                      spatial_x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
            if self.with_bbox or self.with_mask:
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
            losses = dict()

            if self.with_bbox:
                bbox_results = self._bbox_forward_train(x, spatial_x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas)
                losses.update(bbox_results['loss_bbox'])
                #losses.update(bbox_results['loss_spatial'])


            return losses
        def _bbox_forward_train(self, x, spatial_x, sampling_results, gt_bboxes,
                                gt_labels, img_metas):
            """Run forward function and calculate loss for box head in
            training."""
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg)
            for res in sampling_results:
                res.pos_gt_labels = res.pos_gt_labels.new_full((res.pos_gt_labels.size(0),),0,dtype=torch.long)
            spatial_bbox_targets = self.spatial_bbox_head.get_targets(sampling_results,
                                                      gt_bboxes, gt_labels,
                                                      self.train_cfg)
            bbox_results = self._bbox_forward(x, spatial_x, rois, img_metas)

            
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            None, rois,
                                            *bbox_targets)

            loss_spatial = self.spatial_bbox_head.loss(None,
                                                      bbox_results['spatial_bbox_pred'], rois,
                                                      *spatial_bbox_targets)
            if torch.isnan(loss_spatial['loss_bbox']).any():
                loss_spatial['loss_bbox'] = torch.zeros_like(loss_spatial['loss_bbox'])
            #bbox_results.update(loss_bbox=loss_bbox, loss_spatial=loss_spatial)
            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results

        def simple_test(self,
                        x,
                        spatial_x,
                        proposal_list,
                        img_metas,
                        proposals=None,
                        rescale=False):
            """Defines the computation performed for simple testing."""
            assert self.with_bbox, 'Bbox head must be implemented.'

            if isinstance(x, tuple) or isinstance(x, list):
                x_shape = x[0].shape
            else:
                x_shape = x.shape

            assert x_shape[0] == 1, 'only accept 1 sample at test mode'
            assert x_shape[0] == len(img_metas) == len(proposal_list)

            det_bboxes, det_labels = self.simple_test_bboxes(
                x, spatial_x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(
                det_bboxes,
                det_labels,
                self.bbox_head.num_classes,
                thr=self.test_cfg.action_thr)
            return [bbox_results]

        def simple_test_bboxes(self,
                               x,
                               spatial_x,
                               img_metas,
                               proposals,
                               rcnn_test_cfg,
                               rescale=False):
            """Test only det bboxes without augmentation."""
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(x, spatial_x, rois, img_metas)
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['final_pred']
            #new_rois = torch.cat((rois[:,0].unsqueeze(1),bbox_pred),dim=1)
            new_rois = rois 
            img_shape = img_metas[0]['img_shape']
            crop_quadruple = np.array([0, 0, 1, 1])
            flip = False

            if 'crop_quadruple' in img_metas[0]:
                crop_quadruple = img_metas[0]['crop_quadruple']

            if 'flip' in img_metas[0]:
                flip = img_metas[0]['flip']

            det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                img_shape,
                flip=flip,
                crop_quadruple=crop_quadruple,
                cfg=rcnn_test_cfg)

            return det_bboxes, det_labels
else:
    # Just define an empty class, so that __init__ can import it.
    class AVARoIHead:
        pass