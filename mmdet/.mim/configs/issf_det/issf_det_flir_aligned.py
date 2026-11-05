"""FLIR-aligned training configuration for ISSF-Det.

The Cityscapes ResNet50 initialization file and paired dataset are external
inputs; set their paths for the local environment before training.
"""

_base_ = ['./issf_det_base.py', '../_base_/datasets/FLIR.py']

data_root = 'data/FLIR_align/'
pretrained_checkpoint = 'weights/resnet50_cityscape.pth'
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
    batch_size=1,
    num_workers=4,
    dataset=dict(data_root=data_root, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(data_root=data_root, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(
    ann_file=data_root + 'Annotation_test.json', classwise=True)
test_evaluator = val_evaluator

model = dict(
    backbone=dict(
        init_cfg=dict(
            _delete_=True, type='Pretrained',
            checkpoint=pretrained_checkpoint)),
    data_preprocessor=dict(
        mean=[159.8808906080302, 162.22057018543336, 160.28301196773916],
        std=[56.96897676312916, 59.57937492901139, 63.11906486423505],
        mean_lwir=[136.63746562356317] * 3,
        std_lwir=[64.97730349740912] * 3),
    mrsc=dict(imgshape=(1024, 1280), lambda_mrsc=0.4),
    roi_head=dict(bbox_head=dict(num_classes=3)))

train_cfg = dict(max_epochs=12)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9,
                   weight_decay=0.0001))
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, by_epoch=False,
         start_factor=0.001),
    dict(type='MultiStepLR', begin=0, end=25, by_epoch=True,
         milestones=[8, 11], gamma=0.1),
]
