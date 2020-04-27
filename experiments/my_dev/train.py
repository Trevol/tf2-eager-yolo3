# -*- coding: utf-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'
import warnings
import logging
# logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
from yolo.train import train_fn
from yolo.config import ConfigParser


def main():
    config = "configs/svhn/train/svhn.json"
    config_parser = ConfigParser(config)

    # 1. create generator
    # 'dataset/svhn/full/imgs/train/24076.png'
    train_generator, valid_generator = config_parser.create_generator()

    # 2. create model
    model = config_parser.create_model()

    # 3. training
    learning_rate, save_dname, n_epoches = config_parser.get_train_params()
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    train_fn(model,
             train_generator,
             valid_generator,
             learning_rate=learning_rate,
             save_dname=save_dname,
             num_epoches=n_epoches)


main()
