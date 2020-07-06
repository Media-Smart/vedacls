from collections import OrderedDict

import numpy as np

from .base import Base
from .registry import METRICS


@METRICS.register_module
class Accuracy(Base):
    def __init__(self, topk=(1,)):
        super(Accuracy, self).__init__()

        self.topk = topk
        self.maxk = max(topk)
        self.count = 0
        self.tp = {k: 0 for k in self.topk}

    def add(self, pred, gt):
        if gt.ndim == 1:
            gt = gt[:, None]

        mask = np.argsort(pred)[:, -self.maxk:][:, ::-1] == gt
        for k in self.topk:
            self.tp[k] += np.sum(mask[:, :k])
        self.count += len(gt)

    def reset(self):
        self.count = 0
        self.tp = {k: 0 for k in self.topk}

    def result(self):
        res = OrderedDict()
        for k in self.topk:
            res['top_{}'.format(k)] = self.tp[k] / self.count

        return res
