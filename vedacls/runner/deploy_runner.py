import torch

from ..models import build_model
from ..utils import load_checkpoint
from .base import Common


class DeployRunner(Common):
    def __init__(self, deploy_cfg, common_cfg=None):
        deploy_cfg = deploy_cfg.copy()
        common_cfg = {} if common_cfg is None else common_cfg.copy()

        common_cfg['gpu_id'] = deploy_cfg.pop('gpu_id')
        super(DeployRunner, self).__init__(common_cfg)

        # build test transform
        self.transform = self._build_transform(deploy_cfg['transform'])
        # build model
        self.model = self._build_model(deploy_cfg['model'])
        self.model.eval()

    def _build_model(self, cfg):
        self.logger.info('Build model')

        model = build_model(cfg)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()

        return model

    def load_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))

        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'

        return load_checkpoint(self.model, filename, map_location, strict)

    def __call__(self, image):
        with torch.no_grad():
            image = self.transform(image=image)['image']
            image = image.unsqueeze(0)

            if self.use_gpu:
                image = image.cuda()

            output = self.model(image)
            output = torch.softmax(output, dim=-1)[0]
        output = output.cpu().numpy()

        return output
