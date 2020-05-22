from .transforms import Compose
from .registry import PIPELINES, DATASETS
from ..utils import build_from_cfg


def build_dataset(data_cfg):
    data_dir = data_cfg.data_dir
    pipeline = data_cfg.pipeline

    transform_list = []
    for transform in pipeline:
        if isinstance(transform, dict):
            transform = build_from_cfg(transform, PIPELINES)
            transform_list.append(transform)
        else:
            raise TypeError("transform must be a dict")

    transform_pipeline = Compose(transform_list)

    dataset_cfg = dict(type=data_cfg.type,
                       root=data_dir,
                       transform=transform_pipeline)

    dataset = build_from_cfg(dataset_cfg, DATASETS)

    return dataset
