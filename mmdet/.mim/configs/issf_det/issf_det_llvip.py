"""LLVIP training configuration for ISSF-Det with MRSC."""

_base_ = ['./issf_det_base.py', '../_base_/datasets/LLVIP.py']

data_root = 'data/LLVIP/'
paired_meta_keys = (
    'img_id', 'img_path', 'img_lwir_path', 'ori_shape', 'img_shape',
    'scale_factor')

train_pipeline = [
    dict(type='LoadPairedImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PairedImagesResize', scale=(1024, 1280), keep_ratio=True),
    dict(type='PairedImageRandomFlip', prob=0.5),
    dict(type='PairedImagesPad', size_divisor=32),
    dict(type='PackPairedImagesDetInputs', meta_keys=paired_meta_keys),
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile', to_float32=True),
    dict(type='PairedImagesResize', scale=(1024, 1280), keep_ratio=True),
    dict(type='PairedImagesPad', size_divisor=32),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackPairedImagesDetInputs', meta_keys=paired_meta_keys),
]

train_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'Annotation_test.json')
test_evaluator = val_evaluator

model = dict(
    backbone=dict(init_cfg=dict(
        _delete_=True, type='Pretrained',
        checkpoint='torchvision://resnet50')),
    data_preprocessor=dict(
        mean=[51.00417706831775, 48.91707563987938, 33.50014617260082],
        std=[49.80811692248811, 49.63865370391049, 46.54689609388755],
        mean_lwir=[71.41202617640505] * 3,
        std_lwir=[47.068441081747956] * 3),
    mrsc=dict(type='MRSC', imgshape=(1024, 1280), lambda_mrsc=0.1),
    interaction_loss=dict(
        _delete_=True,
        type='InteractionLossV4TargetAware',
        in_channels=256,
        latent_dim=128,
        feat_levels=(1, 2),
        decorr_feat_levels=(1, 2),
        response_delta_feat_levels=(1, 2),
        region_interaction_feat_levels=(1, 2),
        lambda_uni_active=1.0,
        lambda_uni_stable=0.5,
        lambda_syn=0.1,
        lambda_red=1.0,
        lambda_region_interaction=0.05,
        lambda_region_uni_active=1.0,
        lambda_region_uni_stable=1.0,
        lambda_region_syn=1.25,
        lambda_region_red=0.75,
        margin_triplet=1.0,
        max_sim_syn=0.1),
    rpn_head=dict(anchor_generator=dict(scales=[8], ratios=[1, 2, 3])),
    roi_head=dict(bbox_head=dict(
        num_classes=1, loss_bbox=dict(beta=1))),
    train_cfg=dict(rpn_proposal=dict(max_per_img=2000)),
    test_cfg=dict(rcnn=dict(nms=dict(min_score=0.001))))

train_cfg = dict(max_epochs=6)
optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0, norm_type=2),
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9,
                   weight_decay=0.0001))
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, by_epoch=False,
         start_factor=0.01),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True,
         milestones=[6, 10], gamma=0.1),
]
