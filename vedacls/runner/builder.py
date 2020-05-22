from .registry import RUNNERS
from ..utils import build_from_cfg


def build_runner(cfg_runner, default_args):
    runner = build_from_cfg(cfg_runner, RUNNERS, default_args=default_args)
    return runner
