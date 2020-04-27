# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse

from yolo.dataset.annotation import parse_annotation
from yolo.train import train_fn
from yolo.config import ConfigParser


def main():
    config = "configs/svhn/train/svhn.json"
    config_parser = ConfigParser(config)

    # 1. create generator
    # 'dataset/svhn/full/imgs/train/24076.png'
    train_generator, valid_generator = config_parser.create_generator()
    for i in range(1000):
        train_generator.next_batch()


def main2():
    annFile = 'dataset/svhn/full/anns/train/24076.xml'
    # annFile = 'dataset/svhn/full/anns/train/25.xml'
    imageDir = 'dataset/svhn/full/imgs/train'
    labelNames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    config = "configs/svhn/train/svhn.json"
    config_parser = ConfigParser(config)

    fname, boxes, coded_labels = parse_annotation(annFile, imageDir, labelNames)
    assert boxes is not None


main()
