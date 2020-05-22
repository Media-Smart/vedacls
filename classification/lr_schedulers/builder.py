import torch.optim as optims

from ..utils import build_from_cfg

support_list = ['LambdaLR', 'StepLR', 'MultiStepLR',
                'ExponentialLR', 'CosineAnnealingLR']


def build_lr_scheduler(cfg_lr_scheduler, optimizer):

    assert cfg_lr_scheduler.type in support_list, 'Only epoch based method supported'

    lr_scheduler = build_from_cfg(cfg_lr_scheduler, optims.lr_scheduler,
                                  default_args=dict(optimizer=optimizer),
                                  mode='module')

    return lr_scheduler
