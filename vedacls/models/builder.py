import logging
import torch.nn as nn

from . import arch as archs

logger = logging.getLogger()


def build_model(cfg_model):
    if cfg_model.pre_trained:
        info = "=> building pre-trained model {}".format(cfg_model.arch)
        model = archs.__dict__[cfg_model.arch](pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=cfg_model.num_classes)
    else:
        info = "=> building model {}".format(cfg_model.arch)
        model = archs.__dict__[cfg_model.arch](num_classes=cfg_model.num_classes)
    logger.info(info)

    return model
