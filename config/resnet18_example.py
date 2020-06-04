# data
data_root = 'data/sample_dataset/'
dataset_type = 'ImageFolder'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='ColorJitter',
         brightness=(0.5, 1.5),
         contrast=(0.5, 1.5),
         saturation=(0.5, 1.5),
         hue=(-0.03, 0.03)),
    dict(type='RandomEdgeShifting',
         shift_factor=(3, 3, 3, 3)),
    dict(type='GaussianNoiseChannelWise',
         ratio=0.5,
         r=dict(loc=(-0.25, 0.25), scale=(2.5, 3.2), per_channel=True),
         g=dict(loc=(-0.15, 0.15), scale=(1.8, 2.4), per_channel=True),
         b=dict(loc=(-0.35, 0.35), scale=(2.4, 3.0), per_channel=True)),
    dict(type='RandomRotation',
         degrees=180,
         ratio=0.5,
         mode='range',
         fillcolor=(124, 116, 103)),
    dict(type='ResizeKeepRatio',
         size=256,
         padding_value=(124, 116, 103)),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

test_pipeline = [
    dict(type='ResizeKeepRatio',
         size=256,
         padding_value=(124, 116, 103)),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

data = dict(
    loader=dict(
        workers_per_gpu=4,
        batch_size=32,
    ),
    train=dict(
        type=dataset_type,
        data_dir=data_root + 'train',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_dir=data_root + 'val',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_dir=data_root + 'test',
        pipeline=test_pipeline,
    ),
)

#  model
model = dict(
    arch='resnet18',
    num_classes=5,
    pre_trained=False,
)

# criterion
criterion = dict(
    type='CrossEntropyLoss'
)

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
    max_epochs=180,
    log_interval=10,
)

# logger
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='DEBUG'),
        dict(type='FileHandler', level='DEBUG'),
    ), )

seed = 32
work_dir = 'work_dir/resnet18'

load = None
resume = None
