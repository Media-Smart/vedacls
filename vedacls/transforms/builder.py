import albumentations as albu

from .registry import TRANSFORMS
from ..utils import build_from_cfg


def build_transform(cfgs):
    transforms = []
    for cfg in cfgs:
        if hasattr(albu, cfg['type']):
            tf = build_from_cfg(cfg, albu, mode='module')
        else:
            tf = build_from_cfg(cfg, TRANSFORMS)

        transforms.append(tf)

    transforms = albu.Compose(transforms)

    return transforms
