import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import numpy as np

from vedacls.runner import InferenceRunner
from vedacls.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('image', type=str, help='input image path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    probs = runner(image)
    label = np.argmax(probs, axis=-1)
    runner.logger.info('probs: {}'.format(probs))
    runner.logger.info('label: {}'.format(label))


if __name__ == '__main__':
    main()
