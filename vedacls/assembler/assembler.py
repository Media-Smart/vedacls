import os
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from ..utils.config import Config
from ..logger import build_logger
from ..datasets.builder import build_dataset
from ..datasets.loader.build_loader import build_dataloader
from ..models.builder import build_model
from ..criterions.builder import build_criterion
from ..optimizers.builder import build_optimizer
from ..lr_schedulers.builder import build_lr_scheduler
from ..runner.builder import build_runner


def assemble(cfg_path, checkpoint='', test_mode=False):
    cfg = Config.fromfile(cfg_path)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    work_dir = os.path.join(cfg.work_dir, timestamp)
    os.makedirs(work_dir, exist_ok=True)

    # logger
    logger = build_logger(cfg.logger, dict(workdir=work_dir, timestamp=timestamp))

    # set seed
    logger.info('Set seed')
    seed = cfg.seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        logger.warning('You have chosen to seed training. '
                       'This will turn on the CUDNN deterministic setting, '
                       'which can slow down your training considerably! '
                       'You may see unexpected behavior when restarting '
                       'from checkpoints. ')

    # data
    # dataset
    logger.info('Build Data sets')
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    test_dataset = build_dataset(cfg.data.test)

    # dataloader
    logger.info('Build Data Loaders')
    train_loader = build_dataloader(train_dataset, cfg.data.loader)
    val_loader = build_dataloader(val_dataset, cfg.data.loader)
    test_loader = build_dataloader(test_dataset, cfg.data.loader)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    # model
    logger.info('Build Model')
    model = build_model(cfg.model)

    if torch.cuda.is_available():
        use_gpu = True
        if torch.cuda.device_count() > 1:
            if cfg.model.arch.startswith('alexnet') or cfg.model.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
            else:
                model = torch.nn.DataParallel(model)
        model.cuda()
        logger.info('Using {} GPU'.format(torch.cuda.device_count()))
    else:
        use_gpu = False
        logger.info('Using CPU')

    # criterion
    logger.info('Build Criterion')
    criterion = build_criterion(cfg.criterion)

    # optimizer
    logger.info('Build Optimizer')
    optimizer = build_optimizer(cfg.optimizer, model.parameters())

    # lr_scheduler
    logger.info('Build Lr Scheduler')
    lr_scheduler = build_lr_scheduler(cfg.lr_scheduler, optimizer)

    # runner
    logger.info('Build Runner')
    runner = build_runner(
        cfg.runner,
        dict(
            loader=loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            work_dir=work_dir,
            test_mode=test_mode,
            use_gpu=use_gpu
        )
    )

    if cfg.load and cfg.resume:
        raise ValueError('"load" and "resume" cannot be both specified')

    if test_mode is True:
        runner.state_loader(checkpoint)
    elif cfg.resume is not None:
        logger.info('=> resume from {}'.format(cfg.resume))
        runner.state_loader(cfg.resume, mode='resume')
    elif cfg.load is not None:
        logger.info('=> load from {}'.format(cfg.resume))
        runner.state_loader(cfg.load)
    else:
        pass

    return runner
