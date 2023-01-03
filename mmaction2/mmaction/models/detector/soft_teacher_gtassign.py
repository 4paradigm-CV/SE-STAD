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

from mmaction.core.bbox import MyHungarianAssigner

@DETECTORS.register_module()
class SoftTeacherGTAssign(MultiStreamDetector):
    def __init__(self,
                 model: dict,
                 num_classes=81, # 第 0 类是背景类
                 train_cfg=None,
                 test_cfg=None):
        teacher_model = copy.deepcopy(model)
        student_model = copy.deepcopy(model)
        super(SoftTeacherGTAssign, self).__init__(
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
            self.sup_weight = self.train_cfg.get("sup_weight", 1.0)

            # 因为使用 gt 进行 assign, 所以不需要 hard_label 相关的参数
            # 对 gt 进行 crop 或者对 gt 不进行 crop 然后丢到匹配的开关
            self.gt_crop = self.train_cfg.get("gt_crop", False)

            self.STUDENT_EVAL = self.train_cfg.get("student_eval", False)
            self.vis = self.train_cfg.get("visulaization", False)

            # self.count_restriction = self.train_cfg.get("count_restriction", False)

            default_assigner_cfg = dict(
                cls_cost=dict(type='BCECost', weight=1.),
                reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                use_cls=True,
                use_reg=True,
                use_iou=True
            )
                
            assigner_cfg =  self.train_cfg.get(
                "assigner_cfg", {}
            )
            default_assigner_cfg.update(assigner_cfg)
            self.assigner_cfg = default_assigner_cfg
            
            self.assigner = MyHungarianAssigner(
                **self.assigner_cfg
            )

            
    
    def forward_train(self, img, img_metas, **kwargs):
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
                if key in ["loss_action_cls", "recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]:
                # if key in ["loss_cls", "loss_bbox", "loss_iou", "recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]:
                #if key in ["loss_cls", "loss_action_cls", "loss_bbox", "loss_iou", "loss_action_cls", "recall@thr=0.5", "prec@thr=0.5", "recall@top3", "recall@top5", "prec@top3", "prec@top5"]:
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
        # return {}
        # 使用 teacher 生成的 pgt label 作为监督信号进行训练
        # 考虑一下对 tail 类别不进行 loss 的抑制
        # print("unsup:", student_data["img_metas"], pseudo_gt_bboxes)
        unsup_loss = self.student.forward_train(
            img=student_data["img"], img_metas=student_data["img_metas"], gt_bboxes=pseudo_gt_bboxes, gt_labels=pseudo_gt_labels
        )
        
        return unsup_loss
    
    def label_assign(self, det_bboxes_ori, det_labels, gt_bboxes, gt_labels, ori_h, ori_w, det_bboxes):
        # det_bboxes_ori = det_bboxes_ori
        # det_labels = det_labels[:, 1:]
        gt_bboxes = gt_bboxes.to(det_bboxes_ori.device)
        gt_labels = gt_labels.to(det_labels.device)
        
        result = self.assigner.assign(
            det_bboxes_ori,
            det_labels[:, 1:],
            gt_bboxes,
            gt_labels[:, 1:],
            ori_h,
            ori_w
        )
        
        gt_inds = result.gt_inds

        pseudo_label = torch.zeros(
            len(det_bboxes_ori), 81
        ).to(det_labels.device)

        for i in range(len(gt_inds)):
            if gt_inds[i] == 0:
                continue
            pseudo_label[i] = gt_labels[int(gt_inds[i]-1)]
        

        return det_bboxes, pseudo_label, result

    def filter_zero_gt(self, gt_bboxes, gt_labels):
        keep_idx = (gt_bboxes[:, 2] > gt_bboxes[:, 0])
        return gt_bboxes[keep_idx], gt_labels[keep_idx]
        
    
    def filter_det(self, det_bboxes, det_labels, proposal_list, threshold=0.2):
        proposals = proposal_list[0]
        keep_idx = (proposals[:, -1] >= min(threshold, proposals[:, -1].min()))
        return det_bboxes[keep_idx], det_labels[keep_idx]

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
            if self.STUDENT_EVAL:
                det_bboxes, det_labels, proposal_list = self.student.simple_test_bboxes(img, img_meta)
                self.student.spatial_bbox_head.train()
            else:
                det_bboxes, det_labels, proposal_list = self.teacher.simple_test_bboxes(img, img_meta)
            # det_bboxes, det_labels, proposal_list = self.teacher.simple_test_bboxes(img, img_meta)
            height, width = img_meta[0]["img_shape"]
            det_bboxes[:, 0] = det_bboxes[:, 0] * width
            det_bboxes[:, 2] = det_bboxes[:, 2] * width

            det_bboxes[:, 1] = det_bboxes[:, 1] * height
            det_bboxes[:, 3] = det_bboxes[:, 3] * height
            det_bboxes = det_bboxes.detach()
            det_labels = det_labels.detach()
            det_labels[:, 0] = 0
            det_bboxes, det_labels = self.filter_det(det_bboxes, det_labels, proposal_list, threshold=self.proposal_thr)

            

            # 进行 label assign 操作
            former_gt_bboxes = torch.Tensor(img_meta[0]["former_gt_bboxes"].data)
            former_gt_labels = torch.Tensor(img_meta[0]["former_gt_labels"].data)
            later_gt_bboxes = torch.Tensor(img_meta[0]["later_gt_bboxes"].data)
            later_gt_labels = torch.Tensor(img_meta[0]["later_gt_labels"].data)
            
            crop_bbox = img_meta[0]["crop_bbox"]
            short_edge = img_meta[0]["short_edge"]
            is_flip = img_meta[0]["flip"]

            # if self.count_restriction:
            #     self.filter_det_by_count(det_bboxes, det_labels, proposal_list, threshold=self.proposal_thr)

            if is_flip:
                temp_det_bboxes = copy.deepcopy(det_bboxes)
                det_bboxes[:, 0] = width - temp_det_bboxes[:, 2] - 1
                det_bboxes[:, 2] = width - temp_det_bboxes[:, 0] - 1


            # print(
            #     "before:\n",
            #     f"det_bboxes:{det_bboxes}\n",
            #     f"former_gt_bboxes: {former_gt_bboxes}\n"
            #     f"former_gt_labels: {former_gt_labels}\n"
            #     f"later_gt_bboxes: {later_gt_bboxes}\n"
            #     f"later_gt_labels: {later_gt_labels}\n"
            #     f"crop_bbox: {crop_bbox}\n"
            #     f"short_edge: {short_edge}\n"
            #     f"shape: former_gt_bboxes: {former_gt_bboxes.shape}, former_gt_labels: {former_gt_labels.shape}, later_gt_bboxes: {later_gt_bboxes.shape}, later_gt_labels: {later_gt_labels.shape}\n"
            #     f"dtype: former_gt_bboxes: {former_gt_bboxes.dtype}, former_gt_labels: {former_gt_labels.dtype}, later_gt_bboxes: {later_gt_bboxes.dtype}, later_gt_labels: {later_gt_labels.dtype}\n"
            #     f"type: former_gt_bboxes: {type(former_gt_bboxes)}, former_gt_labels: {type(former_gt_labels.dtype)}, later_gt_bboxes: {type(later_gt_bboxes.dtype)}, later_gt_labels: {type(later_gt_labels.dtype)}\n"
            #     f"img_shape: {img_meta[0]}\n\n"
            # )

            if not self.gt_crop:
                # 将 predict 的结果按照 crop 以及 scale 进行转变转变到原图尺寸上
                # TODO: 也可以考虑在做数据增强的时候进行处理将 gt 进行 crop 等处理
                original_shape = img_meta[0]["original_shape"]
                ori_h, ori_w = original_shape
                scale_factor = float(short_edge) / min(ori_w, ori_h)

                # crop_bbox = [crop_bbox[0]/scale_factor, crop_bbox[1]/scale_factor, crop_bbox[2]/scale_factor, crop_bbox[3]/scale_factor]
                det_bboxes_ori = copy.deepcopy(det_bboxes)
                
                if is_flip:
                    temp_det_bboxes_ori = copy.deepcopy(det_bboxes_ori)
                    det_bboxes_ori[:, 0] = width - temp_det_bboxes_ori[:, 2]
                    det_bboxes_ori[:, 2] = width - temp_det_bboxes_ori[:, 0]

                det_bboxes_ori[:, 0] = det_bboxes_ori[:, 0] + crop_bbox[0]
                det_bboxes_ori[:, 2] = det_bboxes_ori[:, 2] + crop_bbox[0]
                det_bboxes_ori[:, 1] = det_bboxes_ori[:, 1] + crop_bbox[1]
                det_bboxes_ori[:, 3] = det_bboxes_ori[:, 3] + crop_bbox[1]

                det_bboxes_ori[:, 0] = det_bboxes_ori[:, 0] / scale_factor
                det_bboxes_ori[:, 2] = det_bboxes_ori[:, 2] / scale_factor
                det_bboxes_ori[:, 1] = det_bboxes_ori[:, 1] / scale_factor
                det_bboxes_ori[:, 3] = det_bboxes_ori[:, 3] / scale_factor

                if is_flip:
                    temp_det_bboxes_ori = copy.deepcopy(det_bboxes_ori)
                    det_bboxes_ori[:, 0] = ori_w - temp_det_bboxes_ori[:, 2]
                    det_bboxes_ori[:, 2] = ori_w - temp_det_bboxes_ori[:, 0]

                # 处理 former_gt_bboxes 和 later_gt_bboxes
                former_gt_bboxes[:, 0] *= ori_w
                former_gt_bboxes[:, 2] *= ori_w
                former_gt_bboxes[:, 1] *= ori_h
                former_gt_bboxes[:, 3] *= ori_h

                later_gt_bboxes[:, 0] *= ori_w
                later_gt_bboxes[:, 2] *= ori_w
                later_gt_bboxes[:, 1] *= ori_h
                later_gt_bboxes[:, 3] *= ori_h

                if is_flip:
                    temp_former_gt_bboxes = copy.deepcopy(former_gt_bboxes)
                    temp_later_gt_bboxes = copy.deepcopy(later_gt_bboxes)

                    former_gt_bboxes[:, 0] = ori_w - temp_former_gt_bboxes[:, 2]
                    former_gt_bboxes[:, 2] = ori_w - temp_former_gt_bboxes[:, 0]
                    
                    later_gt_bboxes[:, 0] = ori_w - temp_later_gt_bboxes[:, 2]
                    later_gt_bboxes[:, 2] = ori_w - temp_later_gt_bboxes[:, 0]

                # 拼接到一起
                gt_bboxes = torch.cat((former_gt_bboxes, later_gt_bboxes), 0)
                gt_labels = torch.cat((former_gt_labels, later_gt_labels), 0)


                # print(
                #     "after:\n",
                #     f"det_bboxes_ori: {det_bboxes_ori}\n"
                #     f"det_labels: {det_labels}\n"
                #     f"gt_bboxes: {gt_bboxes}\n"
                #     f"gt_labels: {gt_labels}\n"
                #     f"crop_bbox: {crop_bbox}\n"
                #     f"short_edge: {short_edge}\n"
                #     f"shape: former_gt_bboxes: {former_gt_bboxes.shape}, former_gt_labels: {former_gt_labels.shape}, later_gt_bboxes: {later_gt_bboxes.shape}, later_gt_labels: {later_gt_labels.shape}\n"
                #     f"dtype: former_gt_bboxes: {former_gt_bboxes.dtype}, former_gt_labels: {former_gt_labels.dtype}, later_gt_bboxes: {later_gt_bboxes.dtype}, later_gt_labels: {later_gt_labels.dtype}\n"
                # )


                # print(
                #     "before label_assign: \n"
                #     f"det_bboxes: {det_bboxes}\n"
                #     f"gt_bboxes: {gt_bboxes}\n"
                #     f"gt_labels: {gt_labels}\n"
                # )

                det_bboxes, det_labels, assign_result = self.label_assign(det_bboxes_ori, det_labels, gt_bboxes, gt_labels, ori_h, ori_w, det_bboxes)

                # self.visualization(img[0], img_meta[0], det_bboxes, det_bboxes_ori, is_flip, former_gt_bboxes, later_gt_bboxes, assign_result)
                # print(
                #     "after label_assign: \n"
                #     f"det_bboxes: {det_bboxes}\n"
                #     f"det_labels: {det_labels}\n"
                # )

            else:
                """
                    gt 随着一起 crop 的处理, 这样拿到的直接是 (0-256) 范围的值, 也不需要额外的变换处理了
                    PS: 但是要将面积为 0 的 gt框去除
                """
                former_gt_bboxes, former_gt_labels = self.filter_zero_gt(former_gt_bboxes, former_gt_labels)
                later_gt_bboxes, later_gt_labels = self.filter_zero_gt(later_gt_bboxes, later_gt_labels)

                gt_bboxes = torch.cat((former_gt_bboxes, later_gt_bboxes), 0)
                gt_labels = torch.cat((former_gt_labels, later_gt_labels), 0)
                # filter no proposal
                if len(det_bboxes) == 1 and det_bboxes[0][0] == 0 and det_bboxes[0][2] == 0:
                    det_labels[0] = torch.zeros(81)
                    det_bboxes[0][2] = 255.0
                    det_bboxes[0][3] = 255.0
                    det_labels = det_labels.float()
                    # if np.random.rand() >= 0.5:
                    #     det_bboxes = former_gt_bboxes
                    #     det_labels = former_gt_labels
                    # else:
                    #     det_bboxes = later_gt_bboxes
                    #     det_labels = later_gt_labels
                else:
                    det_bboxes, det_labels, assign_result = self.label_assign(det_bboxes, det_labels, gt_bboxes, gt_labels, height, width, det_bboxes)
                    if self.vis:
                        self.visualization_gt_crop(img[0], img_meta[0], det_bboxes, former_gt_bboxes, later_gt_bboxes, assign_result)
            pseudo_gt_bboxes.append(det_bboxes)
            pseudo_gt_labels.append(det_labels)
        # exit()
        # print(pseudo_gt_bboxes, pseudo_gt_labels)
        return pseudo_gt_bboxes, pseudo_gt_labels

    def visualization(self, img, img_meta, pseudo_bboxes, pseudo_bboxes_after, is_flip, former_gt_bboxes, later_gt_bboxes, assign_result):
        middle = img.shape[1] // 2
        img = img[:, middle, :, :].cpu().float().numpy().transpose(1,2,0)
        img = mmcv.imdenormalize(img, np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375]), to_bgr=False).astype(np.uint8)
        filename = img_meta["filename"]
        pseudo_bbox = pseudo_bboxes.cpu()
        pseudo_bboxes_after = pseudo_bboxes_after.cpu()

        assign_gt_inds = assign_result.gt_inds
        splits = filename.split("/")
        save_name = f"/home/suilin/codes/mmaction2/visualization2/{splits[-2]}_{splits[-1]}.jpg"
        save_name_ori = f"/home/suilin/codes/mmaction2/visualization2/{splits[-2]}_{splits[-1]}_ori.jpg"

        for j in range(len(pseudo_bbox)):
            bbox = pseudo_bbox[j]
            bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            text = str(assign_gt_inds[j].item())
            img = cv2.putText(img, text, (bbox[0], bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
        
        cv2.imwrite(save_name, img)

        # frame = {}
        
        img_ori = cv2.imread(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(splits[-1]))}{splits[-1]}.jpg")
        if is_flip:
            img_ori = cv2.flip(img_ori, 1)
        for j in range(len(pseudo_bboxes_after)):
            bbox = pseudo_bboxes_after[j]
            bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
            img_ori = cv2.rectangle(img_ori, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            text = str(assign_gt_inds[j].item())
            img_ori = cv2.putText(img_ori, text, (bbox[0], bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)

        cv2.imwrite(save_name_ori, img_ori)

        # if is_flip:
        #     img_ori = cv2.imread(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(splits[-1]))}{splits[-1]}.jpg")
        #     save_name_ori = f"/home/suilin/codes/mmaction2/visualization2/{splits[-2]}_{splits[-1]}_ori_not_filp.jpg"
        #     for j in range(len(pseudo_bboxes_after)):
        #         bbox = pseudo_bboxes_after[j]
        #         bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
        #         img_ori = cv2.rectangle(img_ori, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        #     cv2.imwrite(save_name_ori, img_ori)
        
        gt_index = 1

        if len(former_gt_bboxes) > 0:
            save_name_former = f"/home/suilin/codes/mmaction2/visualization2/{splits[-2]}_{splits[-1]}_former.jpg"
            IND = str((int(splits[-1]) // 30 ) * 30 + 1)
            print(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            img_gt_former = cv2.imread(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            if is_flip:
                img_gt_former = cv2.flip(img_gt_former, 1)
            for j in range(len(former_gt_bboxes)):
                bbox = former_gt_bboxes[j]
                bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
                img_gt_former = cv2.rectangle(img_gt_former, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                text = str(gt_index)
                img_gt_former = cv2.putText(img_gt_former, text, (bbox[0], bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
                gt_index += 1
            cv2.imwrite(save_name_former, img_gt_former)

            # if is_flip:
            #     save_name_former = f"/home/suilin/codes/mmaction2/visualization2/{splits[-2]}_{splits[-1]}_former_not_flip.jpg"
            #     IND = str((int(splits[-1]) // 30 ) * 30 + 1)
            #     print(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            #     img_gt_former = cv2.imread(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            #     for j in range(len(former_gt_bboxes)):
            #         bbox = former_gt_bboxes[j]
            #         bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
            #         img_gt_former = cv2.rectangle(img_gt_former, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            #     cv2.imwrite(save_name_former, img_gt_former)

        if len(later_gt_bboxes) > 0:
            save_name_later = f"/home/suilin/codes/mmaction2/visualization2/{splits[-2]}_{splits[-1]}_later.jpg"
            IND = str((int(splits[-1]) // 30 + 1) * 30 + 1)
            print(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            img_gt_later = cv2.imread(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            if is_flip:
                img_gt_later = cv2.flip(img_gt_later, 1)
            for j in range(len(later_gt_bboxes)):
                bbox = later_gt_bboxes[j]
                bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
                img_gt_later = cv2.rectangle(img_gt_later, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                text = str(gt_index)
                img_gt_later = cv2.putText(img_gt_later, text, (bbox[0], bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
                gt_index += 1
            cv2.imwrite(save_name_later, img_gt_later)

            # if is_flip:
            #     save_name_later = f"/home/suilin/codes/mmaction2/visualization2/{splits[-2]}_{splits[-1]}_later_not_flip.jpg"
            #     IND = str((int(splits[-1]) // 30 + 1) * 30 + 1)
            #     print(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            #     img_gt_later = cv2.imread(f"/mnt/disk1/suilin/ava/frames/{splits[-2]}/{splits[-2]}_{'0'*(6-len(IND))}{IND}.jpg")
            #     for j in range(len(later_gt_bboxes)):
            #         bbox = later_gt_bboxes[j]
            #         bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
            #         img_gt_later = cv2.rectangle(img_gt_later, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            #     cv2.imwrite(save_name_later, img_gt_later)


    def visualization_gt_crop(self, img, img_meta, pseudo_bboxes, former_gt_bboxes, later_gt_bboxes, assign_result):
        middle = img.shape[1] // 2
        img = img[:, middle, :, :].cpu().float().numpy().transpose(1,2,0)
        img = mmcv.imdenormalize(img, np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375]), to_bgr=False).astype(np.uint8)
        filename = img_meta["filename"]
        pseudo_bbox = pseudo_bboxes.cpu()

        is_flip = img_meta["flip"]

        assign_gt_inds = assign_result.gt_inds
        splits = filename.split("/")
        if is_flip:
            save_name = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}_flip.jpg"
            save_name_ori = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}_ori_flip.jpg"
            save_name_former = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}_former_flip.jpg"
            save_name_later = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}_later_flip.jpg"
        else:
            save_name = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}.jpg"
            save_name_ori = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}_ori.jpg"
            save_name_former = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}_former.jpg"
            save_name_later = f"/home/suilin/codes/mmaction2/visualization3/{splits[-2]}_{splits[-1]}_later.jpg"

        for j in range(len(pseudo_bbox)):
            bbox = pseudo_bbox[j]
            bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
            img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            text = str(assign_gt_inds[j].item())
            img = cv2.putText(img, text, (bbox[0], bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
        
        cv2.imwrite(save_name, img)

        img_gt_former = img_meta["img_former"]
        img_gt_later = img_meta["img_later"]

        gt_index = 1
        if len(former_gt_bboxes) > 0:
            for j in range(len(former_gt_bboxes)):
                bbox = former_gt_bboxes[j]
                bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
                img_gt_former = cv2.rectangle(img_gt_former, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                text = str(gt_index)
                img_gt_former = cv2.putText(img_gt_former, text, (bbox[0], bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
                gt_index += 1
            cv2.imwrite(save_name_former, img_gt_former)
    
        if len(later_gt_bboxes) > 0:
            for j in range(len(later_gt_bboxes)):
                bbox = later_gt_bboxes[j]
                bbox = [int(bbox[0]), int(bbox[1]),  int(bbox[2]),  int(bbox[3])]
                img_gt_later = cv2.rectangle(img_gt_later, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
                text = str(gt_index)
                img_gt_later = cv2.putText(img_gt_later, text, (bbox[0], bbox[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
                gt_index += 1
            cv2.imwrite(save_name_later, img_gt_later)
