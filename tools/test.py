import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../classifier'))

from classification.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classification model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    runner = assemble(args.config, args.checkpoint, test_mode=True)
    runner()


if __name__ == '__main__':

    main()
