import os
import time
import torch
import shutil
import logging

from .registry import RUNNERS
from ..utils.metrics import (AverageMeter, ProgressMeter, accuracy)
from ..utils.checkpoint import (save_checkpoint, load_checkpoint)

logger = logging.getLogger()


@RUNNERS.register_module
class Runner(object):
    def __init__(self,
                 loader,
                 model,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 max_epochs,
                 work_dir,
                 start_epoch=0,
                 log_interval=10,
                 test_mode=False,
                 use_gpu=True):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epochs = max_epochs
        self.work_dir = work_dir
        self.start_epoch = start_epoch
        self.log_interval = log_interval
        self.test_mode = test_mode
        self.best_acc1 = 0
        self.use_gpu = use_gpu

    def __call__(self, *args, **kwargs):
        if self.test_mode:
            logger.info('Start testing')
            self.test_epoch()
        else:
            logger.info('Start training')
            for epoch in range(self.start_epoch, self.max_epochs):
                self.train_epoch(epoch)
                acc1 = self.validate_epoch()
                is_best = acc1 > self.best_acc1
                self.best_acc1 = max(self.best_acc1, acc1)
                self.save_checkpoint(is_best=is_best)
                self.lr_scheduler.step()

    def train_epoch(self, epoch):

        train_loader = self.loader['train']

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        lr = AverageMeter('Lr', ':6.4f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, lr, losses, top1, top5],
            prefix='Epoch[{}]:'.format(epoch))
        end = time.time()

        # switch to train mode
        self.model.train()

        for i, (images, targets, paths) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if self.use_gpu:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # compute output
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            lr.update(self.get_lr())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.log_interval == 0:
                logger.info(progress.display(i))

    def validate_epoch(self):

        val_loader = self.loader['val']

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
        end = time.time()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, (images, targets, paths) in enumerate(val_loader):

                if self.use_gpu:
                    images = images.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)

                # compute output
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.log_interval == 0:
                    logger.info(progress.display(i))

            logger.info('\n'+'Test Result: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg

    def test_epoch(self):

        test_loader = self.loader['test']

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(test_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
        end = time.time()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            for i, (images, targets, paths) in enumerate(test_loader):

                if self.use_gpu:
                    images = images.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)

                # compute output
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.log_interval == 0:
                    logger.info(progress.display(i))

            logger.info('\n'+'Test Result: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg

    def save_checkpoint(self, save_optimizer=True, is_best=False):
        current_epoch = self.lr_scheduler.last_epoch
        file_name = 'checkpoint.pth'
        file_path = os.path.join(self.work_dir, file_name)
        meta = dict(epoch=current_epoch,
                    best_acc1=self.best_acc1)
        optimizer = self.optimizer if save_optimizer else None
        lr_scheduler = self.lr_scheduler if save_optimizer else None
        save_checkpoint(self.model,
                        file_path,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        meta=meta)
        if is_best:
            shutil.copy(file_path, os.path.join(os.path.dirname(file_path), 'model_best.pth'))

    def state_loader(self, checkpoint, mode='load', map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = load_checkpoint(self.model,
                                         checkpoint,
                                         map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = load_checkpoint(self.model, checkpoint, map_location=map_location)

        if mode == 'resume':
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            self.start_epoch = checkpoint['meta']['epoch']
            self.best_acc1 = checkpoint['meta']['best_acc1']

    def get_lr(self):
        lr = [x['lr'] for x in self.optimizer.param_groups][0]
        return lr
