import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from scipy.optimize import linear_sum_assignment

@BBOX_ASSIGNERS.register_module()
class MyHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='BCECost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 use_cls=True,
                 use_reg=True,
                 use_iou=True):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.use_cls = 1 if use_cls else 0
        self.use_reg = 1 if use_reg else 0
        self.use_iou = 1 if use_iou else 0

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
            #    img_meta,
               img_h,
               img_w,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt, num_class).
            # img_meta (dict): Meta information for current image.
            img_h, img_w: 图像原图的高和宽
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        num_classes = cls_pred.size(1)

        # 1. assign -1 by default 
        # 分配的 id 和分配的类别
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.float32)
        assigned_labels = bbox_pred.new_full((num_bboxes, num_classes),
                                             0,
                                             dtype=torch.float32)
        
        # 如果 gt 不存在或者 predict 的 bbox 不存在
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            # 没有 gt 的时候所有的都指定分配为背景
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        
        # img_h, img_w, _ = img_meta['img_shape']
        # factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
        #                                img_h]).unsqueeze(0)

        factor = gt_bboxes.new_tensor([
            img_w, img_h, img_w, img_h
        ]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        # 分类损失
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        normalize_bbox_pred = bbox_pred / factor

        reg_cost = self.reg_cost(normalize_bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        # bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bbox_pred, gt_bboxes)
        # weighted sum of above three costs
        # print(
        #     f"cls_cost: {cls_cost}\n"
        #     f"reg_cost: {reg_cost}\n"
        #     f"iou_cost: {iou_cost}\n"
        # )
        cost = cls_cost*self.use_cls + reg_cost*self.use_reg + iou_cost*self.use_cls

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device).float()
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device).float()

        # print(matched_row_inds, matched_col_inds, matched_col_inds.dtype, assigned_gt_inds.dtype)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds.long()] = matched_col_inds + 1
        assigned_labels[matched_row_inds.long()] = gt_labels[matched_col_inds.long()]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels, match_cost=cost, cls_cost=cls_cost*self.use_cls, reg_cost=reg_cost*self.use_reg, iou_cost=iou_cost*self.use_iou)