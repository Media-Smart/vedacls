import torch.nn as nn

from ..utils import build_from_cfg


def build_criterion(cfg_criterion):
    criterion = build_from_cfg(cfg_criterion, nn, mode='module')

    return criterion
