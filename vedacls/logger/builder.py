import os
import sys
import logging


def build_logger(cfg, default_args):
    format_ = '%(asctime)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in cfg['handlers']:
        if handler['type'] == 'StreamHandler':
            instance = logging.StreamHandler(sys.stdout)
        elif handler['type'] == 'FileHandler':
            fp = os.path.join(default_args['workdir'], '%s.log' % default_args['timestamp'])
            instance = logging.FileHandler(fp, 'w')
        else:
            instance = logging.StreamHandler(sys.stdout)

        level = getattr(logging, handler['level'])

        instance.setFormatter(formatter)
        instance.setLevel(level)

        logger.addHandler(instance)

    return logger
