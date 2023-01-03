import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from mmaction.utils.structure_utils import dict_split, weighted_loss
from mmaction.utils import get_root_logger

from .multi_stream_detector import MultiStreamDetector
import copy

@DETECTORS.register_module()
class SoftTeacherValidateTrain(MultiStreamDetector):
    def __init__(self,
                 model: dict,
                 num_classes=81, # 第 0 类是背景类
                 train_cfg=None,
                 test_cfg=None):
        teacher_model = copy.deepcopy(model)
        student_model = copy.deepcopy(model)
        super(SoftTeacherValidateTrain, self).__init__(
            dict(
                teacher=build_detector(teacher_model),
                student=build_detector(student_model)
                ),
            train_cfg = train_cfg,
            test_cfg = test_cfg
        )

        self.num_classes = num_classes

        self.predict_count_dict = {
            "0.1": torch.zeros(80),
            "0.2": torch.zeros(80),
            "0.3": torch.zeros(80),
            "0.4": torch.zeros(80),
            "adaptive": torch.zeros(80),
        }
        self.gt_count_dict = {
            "0.1": torch.zeros(80),
            "0.2": torch.zeros(80),
            "0.3": torch.zeros(80),
            "0.4": torch.zeros(80),
            "adaptive": torch.zeros(80),
        }
        self.wrong_count_dict = {
            "0.1": torch.zeros(80),
            "0.2": torch.zeros(80),
            "0.3": torch.zeros(80),
            "0.4": torch.zeros(80),
            "adaptive": torch.zeros(80),
        }
        self.recall_count_dict = {
            "0.1": torch.zeros(80),
            "0.2": torch.zeros(80),
            "0.3": torch.zeros(80),
            "0.4": torch.zeros(80),
            "adaptive": torch.zeros(80),
        }

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
        print(kwargs.keys())

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
                weight=0,
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
                weight=0,
                ignore_keys=["recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]
            )
            for key in unsup_loss.keys():
                if key in ["loss_action_cls"]:
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

    def cal_unsup_acc(self, pseudo_gt_labels, multi_label_restriction, threshold=0.4):
        # print(
        #     f"pseudo_gt_labels: {pseudo_gt_labels}\n"
        #     f"multi_label_restriction: {multi_label_restriction}\n"
        #     f"threshold: {threshold}"
        # )
        predict_count_list = []
        acc_list = []
        gt_count_list = []
        wrong_count_list = []
        recall_list = []

        for i in range(len(pseudo_gt_labels)):
            pseudo_labels = pseudo_gt_labels[i].cpu()
            gt_label = multi_label_restriction[i].cpu()
            
            predict_count = 0
            acc_count = 0
            wrong_count = 0

            self.gt_count_dict[str(threshold)] += gt_label[1:]

            for j in range(len(pseudo_labels)):
                pseudo_label = pseudo_labels[j]
                same_label = torch.logical_and(pseudo_label, gt_label)
                acc_count += same_label.sum()

                wrong_label = (pseudo_label - gt_label)[1:]
                wrong_label[wrong_label<0] = 0

                wrong_count += wrong_label.sum()

                predict_count += pseudo_label.sum()

                self.predict_count_dict[str(threshold)] += pseudo_label[1:]
                self.wrong_count_dict[str(threshold)] += wrong_label
            
            gt_count_list.append(gt_label.sum())
            acc_list.append(acc_count / (predict_count+1e-10))
            wrong_count_list.append(wrong_count)
            predict_count_list.append(predict_count)

            pseudo_label_img = (pseudo_labels.sum(0) > 0)
            recall = torch.logical_and(pseudo_label_img, gt_label)

            recall_list.append(recall.sum() / (gt_label.sum() + 1e-10))

            self.recall_count_dict[str(threshold)] += recall[1:].float()
        
        print(
            f"####################\n"
            f"threshold: {threshold}\n"
            f"gt_count_list: {gt_count_list}\n"
            f"predict_count_list: {predict_count_list}\n"
            f"wrong_count_list: {wrong_count_list}\n"
            f"recall_list: {recall_list}\n"
            f"acc_list: {acc_list}\n"
            f"####################\n"
        )



    def gen_pgt(self, teacher_data):
        imgs = teacher_data["img"]
        img_metas = teacher_data["img_metas"]
        pseudo_gt_bboxes = []
        pseudo_gt_labels = []

        pseudo_gt_labels2 = []
        pseudo_gt_labels3 = []
        pseudo_gt_labels1 = []
        pseudo_gt_labels_adaptive = []
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
            det_bboxes, det_labels = self.filter_det(det_bboxes, det_labels, proposal_list, threshold=self.proposal_thr)
            if self.hard_label:
                det_labels_04 = self.generate_hard_label(copy.deepcopy(det_labels), threshold=0.4)
                det_labels_03 = self.generate_hard_label(copy.deepcopy(det_labels), threshold=0.3)
                det_labels_02 = self.generate_hard_label(copy.deepcopy(det_labels), threshold=0.2)
                det_labels_01 = self.generate_hard_label(copy.deepcopy(det_labels), threshold=0.1)
                det_labels_adaptive = self.generate_hard_label(copy.deepcopy(det_labels), threshold=self.hard_threshold, adaptive_threshold=self.adaptive_threshold)

                
            pseudo_gt_bboxes.append(det_bboxes)
            pseudo_gt_labels.append(det_labels_04)
            pseudo_gt_labels1.append(det_labels_01)
            pseudo_gt_labels2.append(det_labels_02)
            pseudo_gt_labels3.append(det_labels_03)
            pseudo_gt_labels_adaptive.append(det_labels_adaptive)
        # exit()
        # det_bboxes, det_labels = self.teacher.simple_test_bboxes(imgs[0].unsqueeze(0), [img_metas[0]])
        # print("det_bboxes:", det_bboxes)
        # print("det_labels:", det_labels)
        # print(teacher_data.keys())
        # print(teacher_data["img_metas"][0].keys())
        self.cal_unsup_acc(pseudo_gt_labels1, [i["multi_label_restriction"] for i in teacher_data["img_metas"]], threshold=0.1)
        self.cal_unsup_acc(pseudo_gt_labels2, [i["multi_label_restriction"] for i in teacher_data["img_metas"]], threshold=0.2)
        self.cal_unsup_acc(pseudo_gt_labels3, [i["multi_label_restriction"] for i in teacher_data["img_metas"]], threshold=0.3)
        self.cal_unsup_acc(pseudo_gt_labels, [i["multi_label_restriction"] for i in teacher_data["img_metas"]], threshold=0.4)
        self.cal_unsup_acc(pseudo_gt_labels_adaptive, [i["multi_label_restriction"] for i in teacher_data["img_metas"]], threshold="adaptive")
        # print(
        #     f"gt_count_dict: {self.gt_count_dict}\n"
        #     f"predict_count_dict: {self.predict_count_dict}\n"
        #     f"wrong_count_dict: {self.wrong_count_dict}\n"
        #     f"recall_count_dict: {self.recall_count_dict}\n"
        # )
        tail_idx = []
        gt_count_dict_tail = []
        for i in range(60):
            if self.gt_count_dict["0.1"][i] < 20 and self.gt_count_dict["0.1"][i] > 0:
                tail_idx.append(i)
                gt_count_dict_tail.append(int(self.gt_count_dict["0.1"][i].item()))
        predict_count_dict = {"0.1":[], "0.2":[], "0.3":[], "0.4":[], "adaptive": []}
        wrong_count_dict = {"0.1":[], "0.2":[], "0.3":[], "0.4":[], "adaptive": []}
        recall_count_dict = {"0.1":[], "0.2":[], "0.3":[], "0.4":[], "adaptive": []}
        for k in predict_count_dict.keys():
            for i in tail_idx:
                predict_count_dict[k].append(int(self.predict_count_dict[k][i].item()))
                wrong_count_dict[k].append(int(self.wrong_count_dict[k][i].item()))
                recall_count_dict[k].append(int(self.recall_count_dict[k][i].item()))
        
        print(
            f"gt_count_dict: {gt_count_dict_tail}\n"
            f'predict_count_dict (0.1): {predict_count_dict["0.1"]}\n'
            f'recall_count_dict (0.1): {recall_count_dict["0.1"]}\n'
            f'wrong_count_dict (0.1): {wrong_count_dict["0.1"]}\n'
            f'predict_count_dict (0.2): {predict_count_dict["0.2"]}\n'
            f'recall_count_dict (0.2): {recall_count_dict["0.2"]}\n'
            f'wrong_count_dict (0.2): {wrong_count_dict["0.2"]}\n'
            f'predict_count_dict (0.3): {predict_count_dict["0.3"]}\n'
            f'recall_count_dict (0.3): {recall_count_dict["0.3"]}\n'
            f'wrong_count_dict (0.3): {wrong_count_dict["0.3"]}\n'
            f'predict_count_dict (0.4): {predict_count_dict["0.4"]}\n'
            f'recall_count_dict (0.4): {recall_count_dict["0.4"]}\n'
            f'wrong_count_dict (0.4): {wrong_count_dict["0.4"]}\n'
            f'predict_count_dict (adaptive): {predict_count_dict["adaptive"]}\n'
            f'recall_count_dict (adaptive): {recall_count_dict["adaptive"]}\n'
            f'wrong_count_dict (adaptive): {wrong_count_dict["adaptive"]}\n'
        )

        return pseudo_gt_bboxes, pseudo_gt_labels


    