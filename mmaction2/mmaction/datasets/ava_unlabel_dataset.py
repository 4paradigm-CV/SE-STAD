import copy
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

import mmcv
import numpy as np
from mmcv.utils import print_log

from ..core.evaluation.ava_utils import ava_eval, read_labelmap, results2csv
from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS

# 因为要考虑到后续会添加读取周围的 key_frames 的信息, 所以要传入数据集正常标注的信息 
@DATASETS.register_module()
class AVADatasetUnlabel(BaseDataset):
    def __init__(
        self,
        ann_file_gt,
        pipeline,
        filename_tmpl="img_{:05}.jpg",
        start_index=0,
        num_classes=81,
        data_prefix=None,
        modality="RGB",
        timestamp_start=900,
        timestamp_end=1800,
        fps=30,
        with_gt_message=False,
        with_sample=False,
        sample_ann=None
    ):
        """
            1. 不考虑 testmode,
            2. ann_file_gt 是正常标注信息, 用来给相邻的 frames 提供类别标签的约束信息
            3. label_file 主要是用来进行测试时候使用的类别, 所以这里也不考虑使用
            4. **暂时不考虑使用 proposal, 所以没有引入带 proposal 的设定**
            5. exclude 也没有读入, 验证了一下 exclude 的 frame 都是没有 label 的 frame, 因此这里就不再引入了
        """
        self._FPS = fps
        self.num_class = num_classes
        self.filename_tmpl = filename_tmpl
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.ann_file_gt = ann_file_gt
        self.logger = get_root_logger()
        self.with_gt_message = with_gt_message
        self.with_sample = with_sample
        if self.with_sample:
            self.sample_ann = sample_ann
        super().__init__(
            ann_file_gt,
            pipeline,
            data_prefix,
            test_mode=False, # 因为一定是训练用, 所以写死 test_mode = False
            start_index = start_index,
            modality=modality,
            num_classes=num_classes
        )

    def parse_img_record(self, img_records):
        """Merge image records of the same entity at the same time.
        Args:
            img_records (list[dict]): List of img_records (lines in AVA
                annotations).
        Returns:
            tuple(list): A tuple consists of lists of bboxes, action labels and
                entity_ids
        """
        bboxes, labels, entity_ids = [], [], []
        while len(img_records) > 0:
            img_record = img_records[0]
            num_img_records = len(img_records)

            selected_records = [
                x for x in img_records
                if np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            num_selected_records = len(selected_records)
            img_records = [
                x for x in img_records if
                not np.array_equal(x['entity_box'], img_record['entity_box'])
            ]

            assert len(img_records) + num_selected_records == num_img_records

            bboxes.append(img_record['entity_box'])
            valid_labels = np.array([
                selected_record['label']
                for selected_record in selected_records
            ])

            # The format can be directly used by BCELossWithLogits
            label = np.zeros(self.num_classes, dtype=np.float32)
            label[valid_labels] = 1.

            labels.append(label)
            entity_ids.append(img_record['entity_id'])

        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        entity_ids = np.stack(entity_ids)
        return bboxes, labels, entity_ids

    def load_annotations(self):
        if self.with_sample:
            sample_video_frames = {}
            with open(self.sample_ann, "r") as fin:
                for lin in fin:
                    line_split = line.strip().split(",")
                    video_id = line_split[0]
                    timestamp = line_split[1]
                    if not sample_video_frames.get(video_id, False):
                        sample_video_frames[video_id] = {}
                    if not sample_video_frames[video_id].get(timestamp, False):
                        sample_video_frames[video_id][timestamp] = True

        # 首先 load gt 的信息
        gt_video_infos = []
        gt_records_dict_by_img = defaultdict(list)
        with open(self.ann_file_gt, "r") as fin:
            for line in fin:
                line_split = line.strip().split(',')
                label = int(line_split[6])

                video_id = line_split[0]
                timestamp = int(line_split[1])

                img_key = f'{video_id},{timestamp:04d}'

                entity_box = np.array(list(map(float, line_split[2:6])))
                try:
                    entity_id = int(line_split[7])
                except:
                    entity_id = 0
                shot_info = (0, (self.timestamp_end - self.timestamp_start) *
                             self._FPS)
                
                video_info = dict(
                    video_id=video_id,
                    timestamp=timestamp,
                    entity_box=entity_box,
                    label=label,
                    entity_id=entity_id,
                    shot_info=shot_info)
                gt_records_dict_by_img[img_key].append(video_info)
        
        for img_key in gt_records_dict_by_img:
            video_id, timestamp = img_key.split(',')
            bboxes, labels, entity_ids = self.parse_img_record(gt_records_dict_by_img[img_key])
            ann = dict(
                gt_bboxes=bboxes, gt_labels=labels, entity_ids=entity_ids
            )
            frame_dir = video_id
            if self.data_prefix is not None:
                frame_dir = osp.join(self.data_prefix, frame_dir)
            video_info = dict(
                frame_dir = frame_dir,
                video_id = video_id,
                timestamp = int(timestamp),
                img_key = img_key,
                shot_info = shot_info,
                fps = self._FPS,
                ann = ann
            )
            gt_video_infos.append(video_info)
        
        # 2022.2.20 add: former_gt_bboxes, former_gt_labels, later_gt_bboxes, later_gt_labels
        # 统计前后的 gt 信息
        if self.with_gt_message:
            gt_video_frame2anns = dict()
            for i in range(len(gt_video_infos)):
                gt_video_info = gt_video_infos[i]
                video_id, timestamp = gt_video_info["video_id"], gt_video_info["timestamp"]
                ann = gt_video_info["ann"]
                gt_bboxes, gt_labels = ann["gt_bboxes"], ann["gt_labels"]

                if not gt_video_frame2anns.get(video_id, False):
                    gt_video_frame2anns[video_id] = dict()
                gt_video_frame2anns[video_id][timestamp] = {
                    "gt_bboxes": gt_bboxes,
                    "gt_labels": gt_labels
                }




        # 根据 gt_video_infos 进行 video_id, img 的 label 的统计
        # {video_id:{frame: label}} dict -> dict -> list 的格式
        gt_video_frame2multi_label = dict()
        for video_info in gt_video_infos:
            video_id, timestamp = video_info["video_id"], video_info["timestamp"]
            gt_labels = video_info["ann"]["gt_labels"]
            if not gt_video_frame2multi_label.get(video_id, False):
                gt_video_frame2multi_label[video_id] = dict()
            gt_video_frame2multi_label[video_id][timestamp] = (gt_labels.sum(axis=0) > 0).astype(np.float32)

        """
        开始 load unlabel 部分的数据
        需要的参数:
            frame_dir, video_id, img_key, timestamp, 并不是很需要 timestamp_start, shot_info 甚至是 ann 等信息, 并且只需要指定图片即可
        新添加的参数: 
            multi_label_restriction: 前一帧以及后一帧的 gt_multi_label 的并集. 根据逻辑两者一定存在一个, 如果一者不存在则忽略掉
        不需要 ann_file, 只需要根据将 keyframe 的前 14 帧, 后 15 帧的信息记录下来即可

        2022.2.20 add: former_gt_bboxes, former_gt_labels, later_gt_bboxes, later_gt_labels
        """
        video_infos = []
        for video_info in tqdm(gt_video_infos):
            frame_dir, video_id, timestamp = video_info["frame_dir"], video_info["video_id"], video_info["timestamp"]
            current_multi_label = gt_video_frame2multi_label[video_id][timestamp] # 获取当前帧的 multi_label 信息
            center_index = (timestamp - self.timestamp_start) * self._FPS + 1
            begin_index = center_index - self._FPS//2 + 1
            end_index = center_index + self._FPS//2

            if self.with_sample:
                if not sample_video_frames.get(video_id, False):
                    continue
                if not sample_video_frames[video_id].get(str(timestamp), False):
                    continue
            
            if self.with_gt_message:
                current_gt_ann = gt_video_frame2anns[video_id][timestamp]
                current_gt_bboxes, current_gt_labels = current_gt_ann["gt_bboxes"], current_gt_ann["gt_labels"]
            # 对 begin_index ~ center_index - 1 范围的帧的 label restriction 
            # 向前找 multi_label 信息
            former_multi_label = gt_video_frame2multi_label[video_id].get(timestamp - 1, None)
            if former_multi_label is not None:
                multi_label_restriction = np.logical_or(current_multi_label, former_multi_label).astype(np.float32)
                if self.with_gt_message:
                    former_gt_ann = gt_video_frame2anns[video_id].get(timestamp -1 , None)
                    former_gt_bboxes, former_gt_labels = former_gt_ann["gt_bboxes"], former_gt_ann["gt_labels"]
            else:
                multi_label_restriction = current_multi_label
                if self.with_gt_message:
                    former_gt_bboxes, former_gt_labels = np.zeros((0, 4)), np.zeros((0, 81))
            for frame_index in range(begin_index, center_index):
                video_info = dict(
                    frame_dir = frame_dir,
                    video_id = video_id,
                    timestamp = frame_index, # 这里是真实的 frameindex, 不是 / FPS 之后的 timestamp
                    img_key = f"{video_id},{frame_index}", # 因为不使用 proposal 了, 这里没什么意义
                    multi_label_restriction = multi_label_restriction
                )
                if self.with_gt_message:
                    video_info["former_gt_bboxes"] = former_gt_bboxes
                    video_info["former_gt_labels"] = former_gt_labels
                    video_info["later_gt_bboxes"] = current_gt_bboxes
                    video_info["later_gt_labels"] = current_gt_labels
                video_infos.append(video_info)

            # 对 center_index + 1 ～ end_index 范围的帧的 label restriction 
            # 向后找 multi_label 信息
            later_multi_label = gt_video_frame2multi_label[video_id].get(timestamp + 1, None)
            if later_multi_label is not None:
                multi_label_restriction = np.logical_or(current_multi_label, later_multi_label).astype(np.float32)
                if self.with_gt_message:
                    later_gt_ann = gt_video_frame2anns[video_id].get(timestamp + 1 , None)
                    later_gt_bboxes, later_gt_labels = later_gt_ann["gt_bboxes"], later_gt_ann["gt_labels"]
            else:
                multi_label_restriction = current_multi_label
                if self.with_gt_message:
                    later_gt_bboxes, later_gt_labels = np.zeros((0, 4)), np.zeros((0, 81))
            for frame_index in range(center_index+1, end_index+1):
                video_info = dict(
                    frame_dir = frame_dir,
                    video_id = video_id,
                    timestamp = frame_index, # 这里是真实的 frameindex, 不是 / FPS 之后的 timestamp
                    img_key = f"{video_id},{frame_index}", # 因为不使用 proposal 了, 这里没什么意义
                    multi_label_restriction = multi_label_restriction
                )
                if self.with_gt_message:
                    video_info["former_gt_bboxes"] = current_gt_bboxes
                    video_info["former_gt_labels"] = current_gt_labels
                    video_info["later_gt_bboxes"] = later_gt_bboxes
                    video_info["later_gt_labels"] = later_gt_labels
                video_infos.append(video_info)
        
        return video_infos
    
    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        
        shot_info = (0, (self.timestamp_end - self.timestamp_start) *
                             self._FPS)
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['timestamp_start'] = self.timestamp_start
        results['timestamp_end'] = self.timestamp_end
        results["shot_info"] = shot_info

        # 

        # adopt to pyslowfast filename of frames
        results["filename_tmpl"] = results["video_id"] + "_{:06}.jpg"
        
        return self.pipeline(results)
    
    def prepare_test_frames(self, idx):
        assert False, "You should never call prepare_test_frames when using AVADatasetUnlabel !!!"
    
    def evaluate(self,
                 results,
                 metrics=('mAP', ),
                 metric_options=None,
                 logger=None):
        assert False, "You should never call evaluate when using AVADatasetUnlabel !!!"



        



