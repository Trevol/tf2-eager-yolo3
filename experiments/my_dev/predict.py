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
    # image_path = "../../tests/samples/sample.jpeg"
    image_path = "/hdd/Datasets/counters/img/000030.jpg"

    configFile = "configs/my_predict_coco.json"

    config_parser = ConfigParser(configFile)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)

    # 2. Load image
    image = cv2.imread(image_path)
    image = image[:, :, ::-1]

    # 3. Run detection
    boxes, labels, probs = detector.detect(image, 0.5)

    # 4. draw detected boxes
    visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())

    # 5. plot
    plt.imshow(image)
    plt.show()


main()
