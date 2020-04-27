# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt

from experiments.my_dev.utils.Timer import timeit
from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser


def main():
    # image_path = "../../imgs/dog.jpg"
    # image_path = "dataset/svhn/imgs/1.png"
    image_path = "dataset/full/imgs/train/26.png"

    configFile = "configs/svhn/downloaded_pretrained/svhn_pretrained_downloaded.json"
    # configFile = "configs/svhn/svhn.json"

    config_parser = ConfigParser(configFile)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)

    image = cv2.imread(image_path)
    image = image[:, :, ::-1]

    boxes, labels, probs = detector.detect(image, 0.5)

    labelNames = config_parser.get_labels()
    labelIndex = {_id: name for _id, name in enumerate(labelNames)}
    print(labels, [labelIndex[_id] for _id in labels])
    # 4. draw detected boxes
    visualize_boxes(image, boxes, labels, probs, labelNames)

    # 5. plot
    plt.imshow(image)
    plt.show()


main()
