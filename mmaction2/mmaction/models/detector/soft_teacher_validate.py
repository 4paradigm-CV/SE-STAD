import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from mmaction.utils.structure_utils import dict_split, weighted_loss
from mmaction.utils import get_root_logger

from .multi_stream_detector import MultiStreamDetector
import copy

def inter_cal(a, b):
    inter_height_width = torch.min(a[:, None, 2:], b[None, :, 2:])-torch.max(a[:,None,:2], b[None, :, :2])
    return inter_height_width.clamp_(0).prod(dim=2)

def iou_cal(a, b):
    inter_area = inter_cal(a, b)
    a_area = (a[:, 2] - a[:, 0]).clamp_(0) * (a[:, 3]- a[:, 1]).clamp_(0)
    b_area = (b[:, 2] - b[:, 0]).clamp_(0) * (b[:, 3] - b[:, 1]).clamp_(0)
    iou = torch.where(inter_area > 0.0, inter_area / (a_area[:, None]+b_area[None, :]-inter_area+1e-10), torch.tensor(0.0))
    return iou

@DETECTORS.register_module()
class SoftTeacherValidate(MultiStreamDetector):
    def __init__(self,
                 model: dict,
                 num_classes=81, # 第 0 类是背景类
                 train_cfg=None,
                 test_cfg=None):
        teacher_model = copy.deepcopy(model)
        student_model = copy.deepcopy(model)
        super(SoftTeacherValidate, self).__init__(
            dict(
                teacher=build_detector(teacher_model),
                student=build_detector(student_model)
                ),
            train_cfg = train_cfg,
            test_cfg = test_cfg
        )
        self.cls_count = torch.zeros(80)
        self.recall_count = torch.zeros(80)
        self.predict_count = torch.zeros(80)
        self.wrong_count = torch.zeros(80)
        self.proposal_count = 0
        self.proposal_recall = 0
        self.proposal_wrong = 0

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
    
    def forward_train(self, img, img_metas, **kwargs):
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")

        # if isinstance(img, list):
        #     img = 
        # print(img)
        # exit()

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
                # weight=0.5,
                weight=0.0,
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
                # weight=self.unsup_weight,
                weight=0.0,
                ignore_keys=["recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)
        # self.log_loss(loss)
        return loss
    
    def log_loss(self, loss, keys=["sup_loss_cls", "sup_loss_bbox", "sup_loss_centerness", "sup_loss_action_cls", "unsup_loss_cls", "unsup_loss_bbox", "unsup_loss_centerness", "unsup_loss_action_cls"]):
        log_message = ""
        for key in keys:
            log_message += f"{key}: {loss[key]}, "
        print(log_message)

    def cal_unsup_acc(self, pseudo_gt_bboxes, pseudo_gt_labels, proposal_scores, gt_bboxes, gt_labels):
        """
            首先统计每张图的 gt 与 pseudo_gt iou 的值,
        """
        bbox_recall = []
        bbox_acc = []
        bbox_count = []

        cls_recall = []
        cls_acc = []
        cls_count = []
        predict_count = []
        for i in range(len(pseudo_gt_bboxes)):
            if len(gt_bboxes[i]) == 0:
                continue
            self.proposal_count += len(gt_bboxes[i])
            iou = iou_cal(pseudo_gt_bboxes[i].cpu(), gt_bboxes[i].cpu())
            iou_gt, iou_gt_ind = iou.max(0)
            iou_pseudo, iou_pseudo_ind = iou.max(1)
            # print(f"pseudo_gt_bboxes: {pseudo_gt_bboxes[i].cpu()} \n",
            #       f"gt_bboxes: {gt_bboxes[i].cpu()} \n",
            #       f"proposal_scores: {proposal_scores[i]}")
            # print(f"iou: {iou}, iou_shape: {iou.shape}")
            bbox_count.append(len(gt_bboxes[i]))
            bbox_recall.append((iou_gt>=0.5).sum().float() / (bbox_count[-1]+1e-10))
            bbox_acc.append((iou_gt>=0.5).sum().float() / (len(pseudo_gt_bboxes[i])+1e-10))
            self.proposal_recall += (iou_gt>=0.5).sum()
            self.proposal_wrong += (iou_pseudo < 0.5).sum()
            cls_count.append([])
            cls_recall.append([])
            cls_acc.append([])
            predict_count.append([])
            # 按照匹配到的 ind 去进行 cls_label 的计算
            for bbox_id in range(len(gt_bboxes[i])):
                cls_label = gt_labels[i][bbox_id].cpu()
                # print(f"iou_gt_ind: {iou_gt_ind}, bbox_id: {bbox_id}, len: {len(gt_bboxes[i])}")
                pseudo_id = iou_gt_ind[bbox_id].cpu()
                iou_num = iou_gt[bbox_id].cpu()
                
                cls_count[-1].append((cls_label).sum())
                # print(f"pseudo_gt_labels: {pseudo_gt_labels[i][pseudo_id]}, cls_label: {cls_label}")
                self.cls_count += cls_label[1:]
                if iou_num < 0.5:
                    cls_recall[-1].append(-1)
                    cls_acc[-1].append(-1)
                    predict_count[-1].append(-1)
                    continue
                else:
                    pseudo_cls_label = pseudo_gt_labels[i][pseudo_id].cpu()
                    same_cls_label = torch.logical_and(cls_label[1:], pseudo_cls_label[1:])
                    cls_recall[-1].append(same_cls_label.sum().float()/cls_count[-1][bbox_id])
                    cls_acc[-1].append(same_cls_label.sum().float()/pseudo_cls_label[1:].sum())
                    predict_count[-1].append(pseudo_cls_label.sum())

                    self.recall_count += (same_cls_label.long())
                    self.predict_count += (pseudo_cls_label[1:].long())
                    wrong_cls_label = (pseudo_cls_label - cls_label)[1:]
                    wrong_cls_label[wrong_cls_label < 0] = 0
                    self.wrong_count += wrong_cls_label
                
        print(
            f"bbox_recall: {bbox_recall} \n"
            f"bbox_acc: {bbox_acc} \n"
            f"bbox_count: {bbox_count} \n"
            f"cls_recall: {cls_recall} \n"
            f"cls_acc: {cls_acc} \n"
            f"predict_count: {predict_count} \n"
            f"cls_count: {cls_count} \n"
        )




    def filter_gt(self, teacher_data):
        bboxes = teacher_data["gt_bboxes"]
        keep_idx = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            keep = torch.ones(len(bbox))
            for j in range(len(bbox)):
                if bbox[j][2] <= bbox[j][0] or bbox[j][3] <= bbox[j][1]:
                    # print(f"filter: {bbox[j]}")
                    keep[j] = 0
            teacher_data["gt_bboxes"][i] = teacher_data["gt_bboxes"][i][keep.bool()]


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
            # self.teacher.eval()
            pseudo_gt_bboxes, pseudo_gt_labels, proposal_scores = self.gen_pgt(teacher_data)
            # print([len(i) for i in pseudo_gt_bboxes])
            # print("pseudo_gt_labels:", pseudo_gt_labels)
        
        # 这里用 pseudo_gt_bboxes, pseudo_gt_labels 和给定的 gt 进行一下计算
        # print(teacher_data["gt_bboxes"], teacher_data["gt_labels"])
        # exit()
        self.filter_gt(teacher_data)

        # self.cal_unsup_acc(pseudo_gt_bboxes, pseudo_gt_labels, proposal_scores, [i*256 for i in teacher_data["gt_bboxes"]], teacher_data["gt_labels"])
        self.cal_unsup_acc(pseudo_gt_bboxes, pseudo_gt_labels, proposal_scores, teacher_data["gt_bboxes"], teacher_data["gt_labels"])
        print("####################################")
        print(
            f"stat-cls_count: {self.cls_count}\n"
            f"stat-recall_count: {self.recall_count}\n"
            f"stat-predict_count: {self.predict_count}\n"
            f"stat-wrong_count count: {self.wrong_count}\n"
            f"stat-proposal_count: {self.proposal_count}, stat-proposal_recall: {self.proposal_recall}, stat-proposal_wrong: {self.proposal_wrong}\n"
        )
        print("####################################")
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
        return det_bboxes[keep_idx], det_labels[keep_idx], proposals[:, -1][keep_idx]

    def gen_pgt(self, teacher_data):
        imgs = teacher_data["img"]
        img_metas = teacher_data["img_metas"]
        pseudo_gt_bboxes = []
        pseudo_gt_labels = []
        proposal_scores_list = []
        for i in range(len(img_metas)):
            img = imgs[i].unsqueeze(0)
            img_meta = [img_metas[i]]
            # 最开始使用的 simple_test 是会经过 bbox2result 获取得到每个类别的检测结果的.
            # 更改为使用 simple_test_bboxes 是获取 detection box 的检测结果
            
            # det_bboxes 是输出的 bbox, det_labels 输出的是 81 个类别每个类别的置信度
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
            det_bboxes, det_labels, proposal_scores = self.filter_det(det_bboxes, det_labels, proposal_list, threshold=self.proposal_thr)
            if self.hard_label:
                det_labels = self.generate_hard_label(det_labels, threshold=self.hard_threshold, adaptive_threshold=self.adaptive_threshold)
            pseudo_gt_bboxes.append(det_bboxes)
            pseudo_gt_labels.append(det_labels)
            proposal_scores_list.append(proposal_scores)
            # print(det_labels.dtype)
        # exit()
        # det_bboxes, det_labels = self.teacher.simple_test_bboxes(imgs[0].unsqueeze(0), [img_metas[0]])
        # print("det_bboxes:", det_bboxes)
        # print("det_labels:", det_labels)
        return pseudo_gt_bboxes, pseudo_gt_labels, proposal_scores_list


    
