from .registry import DATASETS
from ..utils import build_from_cfg


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
