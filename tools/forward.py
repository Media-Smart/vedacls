import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../vedacls'))

from vedacls import models
from vedacls.utils import load_checkpoint, Config
from vedacls.datasets import transforms


class Classifier(object):

    def __init__(self):
        self.cfg = Config.fromfile('/home/wch/PycharmProjects/classifier/config/cls_config.py')
        self.model = models.build_model(self.cfg.model)
        self.model = torch.nn.DataParallel(self.model).cuda()
        load_checkpoint(self.model, self.cfg.load)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ResizeKeepRatio(size=256, padding_value=(147, 117, 78)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def img_transform(self, img):
        image = self.transform(img)
        image = image[None, ...]
        return image

    def get_result(self, img):
        # img in PIL format

        img = self.img_transform(img)

        with torch.no_grad():
            if torch.cuda.is_available() is not None:
                img = img.cuda(non_blocking=True)

            output, feature = self.model(img)
            output = torch.nn.functional.softmax(output, 1)

        return output[0].tolist(), feature[0].tolist()
