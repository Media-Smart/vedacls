import torch.nn.modules.loss as losses

from ..utils import build_from_cfg


def build_criterion(cfg_criterion):
    criterion = build_from_cfg(cfg_criterion, losses,mode='module')
    return criterion
