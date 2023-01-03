model = dict(
    type='VideoFasterRCNNWithFPNOneStage',
    backbone=dict(
        type='ResNet3dSlowFastMulti',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        out_indices=(0,1,2,3),
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 2),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    neck=dict(
        type='FPN',
        in_channels=[288, 576, 1152],
        out_channels=256,
        start_level=1,
        num_outs=3,
        add_extra_convs='on_output',
        relu_before_extra_convs=True),
    spatial_bbox_head=dict(
        type='FCOSIoUHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 256)),
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True,
            with_global=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=4608,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5,
            loss_weight=10.0),
        ),
    train_cfg=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.6,
            neg_iou_thr=0.5,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.35,
        nms=dict(type='nms', iou_threshold=1),
        max_per_img=100,
        real_nms=dict(type='nms', iou_threshold=0.3),
        real_proposal_output=10,
        rcnn=dict(
            action_thr=0.002,
            #nms=dict(type="nms", iou_threshold=0.3),
            max_per_img=10),)
)

dataset_type = 'AVADataset'
data_root = 'data/ava/rawframes'
anno_root = 'data/ava/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.2.csv'
ann_file_val = f'{anno_root}/ava_val_v2.2.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.2.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.2.csv'

label_file = f'{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt'

# proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
#                        'recall_93.9.pkl')
# proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', 
        brightness=0.4, 
        contrast=0.4, 
        hue=0.1, 
        saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="AddPadShape"),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=['entity_ids', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'])
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="AddPadShape"),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    # Rename is needed to use mmdet detectors
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img']),
    #dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'],
        nested=True)
]

data = dict(
    videos_per_gpu=6,
    workers_per_gpu=6,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        # proposal_file=proposal_file_train,
        person_det_score_thr=0.9,
        data_prefix=data_root),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        #proposal_file=proposal_file_val,
        person_det_score_thr=0.9,
        data_prefix=data_root))
data['test'] = data['val']

optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.00001)
#optimizer = dict(type='AdamW', lr=0.000075)
# this lr is used for 8 gpus

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,
    warmup_ratio=0.1)
total_epochs = 10
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(start=7,interval=1, save_best='mAP@0.5IOU')
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ava/e2e/slowfast_k400_r50_8x8_cos_10e_fpn_fcos'
load_from = "https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth"
resume_from = None
find_unused_parameters = False

fp16 = dict(loss_scale="dynamic")