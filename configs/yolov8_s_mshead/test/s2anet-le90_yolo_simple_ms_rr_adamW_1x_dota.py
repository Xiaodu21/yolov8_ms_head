angle_version = 'le90'
data_root = '../data/'
dataset_type = 'DOTADataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
launcher = 'none'
load_from = '../checkpoints/yolov8_s_orin_pretrain/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=0.33,
        last_stage_out_channels=1024,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=0.5),
    bbox_head_init=dict(
        anchor_generator=dict(
            angle_version='le90',
            ratios=[
                1.0,
            ],
            scales=[
                4,
            ],
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='FakeRotatedAnchorGenerator'),
        bbox_coder=dict(
            angle_version='le90',
            edge_swap=True,
            norm_factor=None,
            proj_xy=True,
            target_means=(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            target_stds=(
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            type='DeltaXYWHTRBBoxCoder',
            use_box_type=False),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(beta=0.11, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        num_classes=15,
        stacked_convs=2,
        type='S2AHead'),
    bbox_head_refine=[
        dict(
            anchor_generator=dict(
                strides=[
                    8,
                    16,
                    32,
                    64,
                    128,
                ],
                type='PseudoRotatedAnchorGenerator'),
            bbox_coder=dict(
                angle_version='le90',
                edge_swap=True,
                norm_factor=None,
                proj_xy=True,
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                type='DeltaXYWHTRBBoxCoder'),
            feat_channels=256,
            frm_cfg=dict(
                feat_channels=256,
                kernel_size=3,
                strides=[
                    8,
                    16,
                    32,
                    64,
                    128,
                ],
                type='AlignConv'),
            in_channels=256,
            loss_bbox=dict(
                beta=0.11, loss_weight=1.0, type='mmdet.SmoothL1Loss'),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                type='mmdet.FocalLoss',
                use_sigmoid=True),
            num_classes=15,
            stacked_convs=2,
            type='S2ARefineHead'),
    ],
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.33,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        real_out_channels=[
            256,
            256,
            256,
            256,
            256,
        ],
        type='YOLOv8PAFPN_SIMPLE',
        widen_factor=0.5),
    test_cfg=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.1, type='nms_rotated'),
        nms_pre=2000,
        score_thr=0.05),
    train_cfg=dict(
        init=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                min_pos_iou=0,
                neg_iou_thr=0.4,
                pos_iou_thr=0.5,
                type='mmdet.MaxIoUAssigner'),
            debug=False,
            pos_weight=-1),
        refine=[
            dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    min_pos_iou=0,
                    neg_iou_thr=0.4,
                    pos_iou_thr=0.5,
                    type='mmdet.MaxIoUAssigner'),
                debug=False,
                pos_weight=-1),
        ],
        stage_loss_weights=[
            1.0,
        ]),
    type='RefineSingleStageDetector')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        data_root='../data/',
        img_shape=(
            1024,
            1024,
        ),
        pipeline=test_pipeline,
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
     type='DOTAMetric',
     format_only=True,
     merge_patches=True,
     outfile_prefix='./work_dirs/dota/Task1')
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=12)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        data_root='../data/',
        filter_cfg=dict(filter_empty_gt=True),
        img_shape=(
            1024,
            1024,
        ),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='DOTADataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        data_root='../data/',
        img_shape=(
            1024,
            1024,
        ),
        pipeline=[
            dict(
                file_client_args=dict(backend='disk'),
                type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='DOTADataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(
        file_client_args=dict(backend='disk'), type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '../work_dirs'
