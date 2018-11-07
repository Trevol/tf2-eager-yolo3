# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()

import cv2
import os
import argparse
import json
import glob
from tqdm import tqdm

from yolo.net import Yolonet
from yolo.frontend import YoloDetector
from yolo.utils.box import draw_boxes

argparser = argparse.ArgumentParser(
    description='evaluate yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')

argparser.add_argument(
    '-s',
    '--save',
    default=False)

argparser.add_argument(
    '-t',
    '--threshold',
    type=float,
    default=0.5)


if __name__ == '__main__':
    from yolo.dataset.annotation import parse_annotation
    from yolo.eval.fscore import count_true_positives, calc_score
    import numpy as np
    
    args = argparser.parse_args()
    with open(args.config) as data_file:    
        config = json.load(data_file)

    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_weights(os.path.join(config["train"]["save_folder"], "weights.h5"))
    detector = YoloDetector(model)
 
    n_true_positives = 0
    n_truth = 0
    n_pred = 0
    ann_fnames = glob.glob(os.path.join(config["train"]["train_annot_folder"], "*.xml"))
    for ann_fname in tqdm(ann_fnames): 
        img_fname, true_boxes, true_labels = parse_annotation(ann_fname, config["train"]["train_image_folder"], config["model"]["labels"])
        true_labels = np.array(true_labels)
        image = cv2.imread(img_fname)[:,:,::-1]

        boxes, labels, probs = detector.detect(image, config["model"]["anchors"], config["model"]["net_size"], args.threshold)
        
        n_true_positives += count_true_positives(boxes, true_boxes, labels, true_labels)
        n_truth += len(true_boxes)
        n_pred += len(boxes)
        
        if args.save:
            image_ = draw_boxes(image, boxes, labels, probs, config["model"]["labels"], obj_thresh=0.0, desired_size=416)
            output_path = os.path.join(config["train"]["save_folder"], os.path.split(img_fname)[-1])
            cv2.imwrite(output_path, image_[:,:,::-1])

    print(calc_score(n_true_positives, n_truth, n_pred))


