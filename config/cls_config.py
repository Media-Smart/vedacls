# data
data_root = 'path to dataset'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='ColorJitter',
         brightness=(0.5, 1.5),
         contrast=(0.5, 1.5),
         saturation=(0.5, 1.5)),
    dict(type='RandomEdgeShifting',
         shift_factor=(3, 3, 3, 3)),
    dict(type='ResizeKeepRatio',
         size=256,
         padding_value=(147, 117, 78)),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

test_pipeline = [
    dict(type='ResizeKeepRatio',
         size=256,
         padding_value=(147, 117, 78)),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

data = dict(
    type='ImageFolder',
    loader=dict(
        workers_per_gpu=4,
        batch_size=32,
    ),
    train=dict(
        data_dir=data_root + 'train',
        pipeline=train_pipeline,
    ),
    val=dict(
        data_dir=data_root + 'val',
        pipeline=test_pipeline,
    ),
    test=dict(
        data_dir=data_root + 'test',
        pipeline=test_pipeline,
    ),
)

#  model
model = dict(
    arch='resnet18',
    num_classes=10,
    pre_trained=False,
)

# criterion
criterion = dict(type='CrossEntropyLoss'),

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.0125,
    momentum=0.9,
    weight_decay=0.0001,
)

# lr_scheduler
lr_scheduler = dict(
    type='MultiStepLR',
    milestones=[60, 120],
    gamma=0.1,
)

# run time setting
runner = dict(
    type='Runner',
    log_interval=10,
)

seed = 32
max_epochs = 180
work_dir = 'work_dir'

load = "/home/wch/PycharmProjects/classifier/work_dir/orc_c3_hw_c1s5_rw_c2s1_fc_c1s2/test_s_cj_es_gn_rotation/model_best.pth.tar"
resume = None
