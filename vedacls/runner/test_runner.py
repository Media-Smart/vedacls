import torch

from .deploy_runner import DeployRunner


class TestRunner(DeployRunner):
    def __init__(self, test_cfg, deploy_cfg, common_cfg=None):
        super(TestRunner, self).__init__(deploy_cfg, common_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])

    def __call__(self):
        self.metric.reset()
        self.model.eval()

        res = {}

        self.logger.info('Start testing')
        with torch.no_grad():
            for idx, (img, label) in enumerate(self.test_dataloader):
                if self.use_gpu:
                    img = img.cuda()

                out = self.model(img)
                self.metric.add(out.cpu().numpy(), label.cpu().numpy())
                res = self.metric.result()

                self.logger.info('Test, Iter {}, {}'.format(
                    idx+1,
                    ', '.join(['{}: {:.4f}'.format(name, res[name]) for name in
                               res])))
        self.logger.info(', '.join(['{}: {:.4f}'.format(k, v) for k, v in
                   res.items()]))

        return res
