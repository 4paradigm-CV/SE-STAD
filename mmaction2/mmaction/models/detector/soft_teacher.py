import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from mmaction.utils.structure_utils import dict_split, weighted_loss
from mmaction.utils import get_root_logger

from .multi_stream_detector import MultiStreamDetector
import copy

import cv2
from PIL import Image
import mmcv
import numpy as np

@DETECTORS.register_module()
class SoftTeacher(MultiStreamDetector):
    def __init__(self,
                 model: dict,
                 num_classes=81, # 第 0 类是背景类
                 train_cfg=None,
                 test_cfg=None):
        teacher_model = copy.deepcopy(model)
        student_model = copy.deepcopy(model)
        super(SoftTeacher, self).__init__(
            dict(
                teacher=build_detector(teacher_model),
                student=build_detector(student_model)
                ),
            train_cfg = train_cfg,
            test_cfg = test_cfg
        )

        self.num_classes = num_classes

        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
            self.proposal_thr = self.train_cfg.proposal_thr
            # 是否使用 unsup
            self.use_unsup = self.train_cfg.get("use_unsup", True)
            # 使用 hard label 还是 soft label, 默认使用 hard_label
            self.hard_label = self.train_cfg.get("hard_label", True)
            self.adaptive_threshold = self.train_cfg.get("adaptive_threshold", False)
            if self.adaptive_threshold:
                self.adaptive_threshold_values = torch.Tensor(self.train_cfg["adaptive_threshold_values"])
            self.hard_threshold = self.train_cfg.get("hard_threshold", 0.2)
            self.label_restriction = self.train_cfg.get("label_restriction", True)
            self.sup_weight = self.train_cfg.get("sup_weight", 1.0)

            self.img_show = self.train_cfg.get("img_show", False)
            self.STUDENT_EVAL = self.train_cfg.get("student_eval", False)
    
    def forward_train(self, img, img_metas, **kwargs):
        self.teacher.eval()
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")

        """
        data_groups 的格式:
            sup: gt_bboxes, gt_labels, img, img_metas, tag
            unsup_student: gt_bboxes, gt_labels, img, img_metas, tag
            unsup_teacher: gt_bboxes, gt_labels, img, img_metas, tag
            PS: proposal 相关的默认没有添加, 如果添加的话每一项还会有 proposal 的对应
        """

        for _, v in data_groups.items():
            v.pop("tag")


        loss = {}
        if "sup" in data_groups:
            # print("sup: ", data_groups["sup"])
            """
                gt_bboxes: [tensor(n, 4)]
                gt_labels: [tensor(n, 81)]
            """
            # print("sup:", data_groups["sup"]["gt_bboxes"], data_groups["sup"]["gt_labels"], data_groups["sup"]["img_metas"])
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = weighted_loss(
                sup_loss,
                weight=self.sup_weight,
                ignore_keys=["recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]
            )

            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        
        if not self.use_unsup:
            return loss
        
        if "unsup_student" in data_groups:
            unsup_loss = self.forward_unsup_train(
                data_groups["unsup_teacher"], data_groups["unsup_student"]
            )
            unsup_loss = weighted_loss(
                unsup_loss,
                weight=self.unsup_weight,
                ignore_keys=["recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]
            )
            for key in unsup_loss.keys():
                #if key in ["loss_iou", "loss_bbox", "loss_cls", "loss_action_cls", "recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]:
                if key in ["loss_action_cls", "recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]:
                    unsup_loss[key] = unsup_loss[key] * 1
                else:
                    unsup_loss[key] = unsup_loss[key] * 0
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
        # self.log_loss(loss)
        return loss
    
    def log_loss(self, loss, keys=["sup_loss_cls", "sup_loss_bbox", "sup_loss_centerness", "sup_loss_action_cls", "unsup_loss_cls", "unsup_loss_bbox", "unsup_loss_centerness", "unsup_loss_action_cls"]):
        log_message = ""
        for key in keys:
            log_message += f"{key}: {loss[key]}, "
        print(log_message)

    def forward_unsup_train(self, teacher_data, student_data):
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        teacher_data["img"] = teacher_data["img"][
            torch.Tensor(tidx).to(teacher_data["img"].device).long()
        ]
        teacher_data["img_metas"] = [teacher_data["img_metas"][idx] for idx in tidx]
        
        # proposals 相关
        if "proposals" in teacher_data:
            teacher_data["proposals"] = [teacher_data["proposals"][idx] for idx in tidx]
        
        # 使用 teacher model 去进行 forward 拿到 detection result
        with torch.no_grad():
            pseudo_gt_bboxes, pseudo_gt_labels = self.gen_pgt(teacher_data)
            # print([len(i) for i in pseudo_gt_bboxes])
            # print("pseudo_gt_labels:", pseudo_gt_labels)
        
        # 使用 teacher 生成的 pgt label 作为监督信号进行训练
        # 考虑一下对 tail 类别不进行 loss 的抑制
        # print("unsup:", student_data["img_metas"], pseudo_gt_bboxes)
        unsup_loss = self.student.forward_train(
            img=student_data["img"], img_metas=student_data["img_metas"], gt_bboxes=pseudo_gt_bboxes, gt_labels=pseudo_gt_labels
        )
        
        return unsup_loss
    
    def generate_hard_label(self, det_labels, threshold=0.2, adaptive_threshold=False):
        # det_labels = (det_labels >= threshold).float()
        if not adaptive_threshold:
            for i in range(len(det_labels)):
                det_labels[i] = (det_labels[i] >= min(threshold, max(det_labels[i]))).float()
            det_labels = det_labels.float()
            return det_labels
        else:
            adaptive_threshold_values = self.adaptive_threshold_values
            det_labels = (det_labels > adaptive_threshold_values.to(det_labels.device)).float()
            return det_labels


    def filter_det(self, det_bboxes, det_labels, proposal_list, threshold=0.2):
        proposals = proposal_list[0]
        keep_idx = (proposals[:, -1] >= min(threshold, proposals[:, -1].min()))
        return det_bboxes[keep_idx], det_labels[keep_idx]


    def label_restriction_func(self, det_labels, multi_label_restriction):
        # print("det_labels:", det_labels)
        # print("multi_label_restriction:", multi_label_restriction)
        multi_label_restriction = multi_label_restriction.to(det_labels.device)
        for i in range(len(det_labels)):
            det_labels[i] = torch.logical_and(det_labels[i], multi_label_restriction).float()
        return det_labels.float()

    def gen_pgt(self, teacher_data):
        imgs = teacher_data["img"]
        img_metas = teacher_data["img_metas"]
        pseudo_gt_bboxes = []
        pseudo_gt_labels = []
        for i in range(len(img_metas)):
            img = imgs[i].unsqueeze(0)
            img_meta = [img_metas[i]]
            # 最开始使用的 simple_test 是会经过 bbox2result 获取得到每个类别的检测结果的.
            # 更改为使用 simple_test_bboxes 是获取 detection box 的检测结果
            
            # det_bboxes 是输出的 bbox, det_labels 输出的是 81 个类别每个类别的置信度
            # det_bboxes, det_labels, proposal_list = self.teacher.simple_test_bboxes(img, img_meta)
            if self.STUDENT_EVAL:
                det_bboxes, det_labels, proposal_list = self.student.simple_test_bboxes(img, img_meta)
                self.student.spatial_bbox_head.train()
            else:
                det_bboxes, det_labels, proposal_list = self.teacher.simple_test_bboxes(img, img_meta)
            height, width = img_meta[0]["img_shape"]
            det_bboxes[:, 0] = det_bboxes[:, 0] * height
            det_bboxes[:, 2] = det_bboxes[:, 2] * height

            det_bboxes[:, 1] = det_bboxes[:, 1] * width
            det_bboxes[:, 3] = det_bboxes[:, 3] * width
            det_bboxes = det_bboxes.detach()
            det_labels = det_labels.detach()
            det_labels[:, 0] = 0
            # print("det_bboxes:", det_bboxes)
            # print("det_labels:", det_labels)
            # print("det_bboxes:", det_bboxes)
            # print("proposal_list:", proposal_list[0][proposal_list[0][:, -1]>min(0.2, proposal_list[0][:, -1].min())])
            det_bboxes, det_labels = self.filter_det(det_bboxes, det_labels, proposal_list, threshold=self.proposal_thr)
            if self.hard_label:
                det_labels = self.generate_hard_label(det_labels, threshold=self.hard_threshold, adaptive_threshold=self.adaptive_threshold)
            pseudo_gt_bboxes.append(det_bboxes)
            pseudo_gt_labels.append(det_labels)
            if self.label_restriction:
                det_labels = self.label_restriction_func(det_labels, img_meta[0]["multi_label_restriction"].data)
        # exit()
        # det_bboxes, det_labels = self.teacher.simple_test_bboxes(imgs[0].unsqueeze(0), [img_metas[0]])
        # print("det_bboxes:", det_bboxes)
        # print("det_labels:", det_labels)
        if self.img_show:
            self.visualization(imgs, img_metas, pseudo_gt_bboxes, pseudo_gt_labels)
        return pseudo_gt_bboxes, pseudo_gt_labels


    def visualization(self, imgs, img_metas, pseudo_bboxes, pseudo_labels):
        # print(img_metas, len(img_metas), pseudo_bboxes)
        # print(imgs.shape, pseudo_bboxes.shape, pseudo_labels.shape)
        # exit()
        # print(imgs.shape)
        for i in range(len(imgs)):
            middle = imgs[i].shape[1] // 2
            # print(imgs[i,:,middle,:,:].shape, imgs[i,:,middle,:,:].min(), imgs[i,:,middle,:,:].max())
            img = imgs[i,:,middle,:,:].cpu().float().numpy().transpose(1,2,0)
            print(img.shape, img.dtype)
            img = mmcv.imdenormalize(img, np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375]), to_bgr=False).astype(np.uint8)

            # print(img.shape)
            img_meta = img_metas[i]
            filename = img_meta["filename"]
            pseudo_bbox = pseudo_bboxes[i].cpu()
            pseudo_label = pseudo_labels[i].cpu()

            label_restriction = img_meta["multi_label_restriction"].data.cpu().nonzero()
            if len(label_restriction) == 1:
                label_restriction = label_restriction[0].numpy().tolist()
            else:
                label_restriction = label_restriction.squeeze().numpy().tolist()
            label_restriction = [str(l) for l in label_restriction]
            
            splits = filename.split("/")
            save_name = f"/home/suilin/codes/mmaction2/visualization/{splits[-2]}_{splits[-1]}.jpg"

            for j in range(len(pseudo_bbox)):
                bbox = pseudo_bbox[j]
                label = pseudo_label[j].nonzero()
                if len(label) == 1:
                    label = label[0].numpy().tolist()
                elif len(label) == 0:
                    label = []
                else:
                    label = label.squeeze().numpy().tolist()
                label = [str(l) for l in label]
                bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]

                img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                text = ",".join(label)
                if not text:
                    text = "None"
                img = cv2.putText(img, text, (bbox[0]+10, bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
                
            label_restriction_text = ",".join(label_restriction)
            img = cv2.putText(img, label_restriction_text, (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA, False)
            cv2.imwrite(save_name, img)
