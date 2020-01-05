import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import sys
import csv
import random
from datetime import datetime
import time
from PIL import Image
import torch
from torchvision import transforms, datasets, models

from techniques.RISE import *
from techniques.gen_patch import gen_adversarial_patch
from techniques.utils import read_tensor
from metrics.patch_intersection import *
from data_utils.data_setup import *
from metrics.evaluation import CausalMetric, auc, gkern


def write_to_csv(row, filename):
    with open(filename, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)


import json


def find_label(target):
    with open('./data/imagenet_class_index.json', 'r') as f:
        labels = json.load(f)
    for key in labels:
        index, label = labels[key]
        if index == target:
            return label, key


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--path",
                        default="./data/samples/cat_dog.png",
                        type=str,
                        help="Path to image.")
    parser.add_argument("--label",
                        default="black_car",
                        type=str,
                        help="Image label.")
    parser.add_argument("--result-path",
                        default="./results/",
                        type=str,
                        help="Location of results file")
    parser.add_argument("--technique",
                        default='gcam',
                        type=str,
                        help="Technique to test")
    parser.add_argument("--model",
                        default='vgg19',
                        type=str,
                        help="Model to test")
    args = parser.parse_args(sys.argv[1:])

    print('------------------------------')
    print('__Number CUDA Devices:', torch.cuda.is_available())
    print('------------------------------')

    # create ressults directory if it doesnt exist
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    print(args.path)
    img = cv2.imread(args.path, 1)
    print(img)
    grounding = gen_grounding(img, 'vgg19', args.technique, args.label, show=True, label_index=time.time())