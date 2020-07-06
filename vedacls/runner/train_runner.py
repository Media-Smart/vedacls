import os
from collections import OrderedDict

import torch

from ..optimizers import build_optimizer
from ..criterions import build_criterion
from ..lr_schedulers import build_lr_scheduler
from ..utils import save_checkpoint
from .deploy_runner import DeployRunner


class TrainRunner(DeployRunner):
    def __init__(self, train_cfg, deploy_cfg, common_cfg=None):
        super(TrainRunner, self).__init__(deploy_cfg, common_cfg)

        self.train_dataloader = self._build_dataloader(
            train_cfg['data']['train'])

        if 'val' in train_cfg['data']:
            self.val_dataloader = self._build_dataloader(
                train_cfg['data']['val'])
        else:
            self.val_dataloader = None

        self.optimizer = self._build_optimizer(train_cfg['optimizer'])
        self.criterion = self._build_criterion(train_cfg['criterion'])
        self.lr_scheduler = self._build_lr_scheduler(train_cfg['lr_scheduler'])
        self.max_epochs = train_cfg['max_epochs']
        self.log_interval = train_cfg.get('log_interval', 10)
        self.trainval_ratio = train_cfg.get('trainval_ratio', -1)
        self.snapshot_interval = train_cfg.get('snapshot_interval', -1)
        self.save_best = train_cfg.get('save_best', True)

        assert self.workdir is not None
        assert self.log_interval > 0

        self.best = OrderedDict()

        if train_cfg.get('resume'):
            self.resume(**train_cfg['resume'])

    def _build_optimizer(self, cfg):
        return build_optimizer(cfg, self.model.parameters())

    def _build_criterion(self, cfg):
        return build_criterion(cfg)

    def _build_lr_scheduler(self, cfg):
        return build_lr_scheduler(cfg, self.optimizer)

    def _train(self):
        self.metric.reset()
        self.model.train()

        self.logger.info('Epoch {}, start training'.format(self.epoch + 1))
        for idx, (img, label) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            if self.use_gpu:
                img = img.cuda()
                label = label.cuda()

            out = self.model(img)
            loss = self.criterion(out, label)

            loss.backward()
            self.optimizer.step()

            if (idx+1) % self.log_interval == 0:
                with torch.no_grad():
                    self.metric.add(out.cpu().numpy(), label.cpu().numpy())
                    res = self.metric.result()
                    self.logger.info(
                        'Train, Epoch {}, Iter {}, LR {}, Loss {:.4f}, {}'.format(
                            self.epoch+1, idx+1,
                            ','.join(['{:.15g}'.format(lr) for lr in self.lr]),
                            loss.item(),
                            ', '.join(['{}: {:.4f}'.format(k, v) for k, v in
                                       res.items()])))

        self.lr_scheduler.step()

    def _val(self):
        self.metric.reset()
        self.model.eval()

        res = {}

        self.logger.info('Start validating')
        with torch.no_grad():
            for idx, (img, label) in enumerate(self.val_dataloader):
                if self.use_gpu:
                    img = img.cuda()

                out = self.model(img)

                self.metric.add(out.cpu().numpy(), label.cpu().numpy())
                res = self.metric.result()

                if (idx + 1) % self.log_interval == 0:
                    self.logger.info('Validation, Iter {}, {}'.format(
                        idx+1,
                        ', '.join(['{}: {:.4f}'.format(k, v) for k, v in
                                   res.items()])))

        return res

    def __call__(self):
        for _ in range(self.epoch, self.max_epochs):
            self._train()

            if self.trainval_ratio > 0 and \
                    self.epoch % self.trainval_ratio == 0 and \
                    self.val_dataloader:
                res = self._val()
                for name in res:
                    if name not in self.best:
                        self.best[name] = 0.0
                    if self.best[name] <= res[name]:
                        self.best[name] = res[name]
                        if self.save_best:
                            self.save_checkpoint(
                                self.workdir, 'best_{}.pth'.format(name),
                                meta=dict(best=self.best))
                self.logger.info(', '.join(['Best {}: {:.4f}'.format(k, v) for k, v in self.best.items()]))

            if self.snapshot_interval > 0 and \
                    self.epoch % self.snapshot_interval == 0:
                self.logger.info('Snapshot')
                self.save_checkpoint(
                    self.workdir, 'epoch_{}.pth'.format(self.epoch),
                    meta=dict(best=self.best))

    @property
    def epoch(self):
        return self.lr_scheduler.last_epoch

    @property
    def lr(self):
        return [x['lr'] for x in self.optimizer.param_groups]

    def save_checkpoint(self, dir, filename, save_optimizer=True,
                        save_lr_scheduler=True, meta=None):
        optimizer = self.optimizer if save_optimizer else None
        lr_scheduler = self.lr_scheduler if save_lr_scheduler else None

        filepath = os.path.join(dir, filename)
        self.logger.info('Save checkpoint {}'.format(filename))
        save_checkpoint(self.model, filepath, optimizer, lr_scheduler, meta)

    def resume(self, checkpoint, resume_optimizer=False,
               resume_lr_scheduler=False, resume_meta=False,
               map_location='default'):
        checkpoint = self.load_checkpoint(checkpoint,
                                          map_location=map_location)

        if resume_optimizer and 'optimizer' in checkpoint:
            self.logger.info('Resume optimizer')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if resume_lr_scheduler and 'lr_scheduler' in checkpoint:
            self.logger.info('Resume lr scheduler')
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if resume_meta and 'meta' in checkpoint:
            self.logger.info('Resume meta data')
            self.best = checkpoint['meta']['best']
