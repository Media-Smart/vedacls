import torch.optim as optims

from ..utils import build_from_cfg


def build_lr_scheduler(cfg_lr_scheduler, optimizer):
    lr_scheduler = build_from_cfg(cfg_lr_scheduler, optims.lr_scheduler,
                                  default_args=dict(optimizer=optimizer),
                                  mode='module')

    return lr_scheduler
