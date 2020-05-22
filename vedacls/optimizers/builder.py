import torch.optim as optims

from ..utils import build_from_cfg


def build_optimizer(cfg_optimizer, params):
    optimizer = build_from_cfg(cfg_optimizer, optims,
                               default_args=dict(params=params),
                               mode='module')
    return optimizer
