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

from metrics.utils import *
from techniques.generate_grounding import gen_grounding, gen_all_groundings
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


if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--sample-size",
                        default=3,
                        type=int,
                        help="Number of images to test.")
    parser.add_argument("--thresholds",
                        default=1,
                        type=list,
                        help="Thresholds to test.")
    parser.add_argument("--result-path",
                        default="./results/testing_dataloader/",
                        type=str,
                        help="Location of results file")
    parser.add_argument("--techniques",
                        type=str,
                        help="Techniques to test")
    parser.add_argument("--model",
                        default='vgg19',
                        type=str,
                        help="Models to test")
    parser.add_argument("--datadir",
                        default='./data/test/',
                        type=str,
                        help="location of data")
    parser.add_argument("--all",
                        action='store_true',
                        help="explain whole batch")
    parser.add_argument("--cuda",
                        default=0,
                        type=int,
                        help="cuda device")
    args = parser.parse_args(sys.argv[1:])

    print('------------------------------')
    if torch.cuda.is_available():
        print('USING GPU')
        torch.cuda.set_device(args.cuda)
    else:
        print("CPU MODE")
    print('------------------------------')
    
    ''' Load dataset/datatloader '''
    if args.all:
        dataset, data_loader = get_imagenet_test(datadir=args.datadir, shuffle=True, all=True)
    else:
        dataset, data_loader = get_imagenet_test(datadir=args.datadir, shuffle=True, sample_size=args.sample_size)

    labels = list(os.walk(args.datadir))[0][1]
    list.sort(labels)

    techniques = [item for item in args.techniques.split(' ')]

    print("starting eval for sample size {0}".format(len(data_loader)))
    print('------------------------------')
    
    # create ressults directory if it doesnt exist
    if not os.path.exists(args.result_path):
                os.makedirs(args.result_path)

    for i, (inp, label) in enumerate(data_loader):
        #if num_images < args.sample_size:
        print(label)
        print('entering loop {0}'.format(i))
        gt_label = labels[label.numpy()[0]]
        gt_label_name, gt_label_idx = find_label(gt_label)
        print('ground truth class: {0}  index: {1}'.format(gt_label_name, gt_label_idx))
        # get data
        img = inp[0].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        displ_img = np.clip((std * img + mean), 0, 1)
        save_img = displ_img/np.max(displ_img)
        displ_img = np.uint8((displ_img/np.max(displ_img))*255)
        label_name, label_idx = get_top_prediction('vgg19', inp.cuda())
        print('predicted class: {0}  index: {1}'.format(label_name, label_idx))
        #num_images += 1
        if ((label_name == gt_label_name)):
            print('correct classification')

            ''' generate explanations '''
            if len(techniques) == 4:
                grounding = gen_all_groundings(displ_img, gt_label_name, args.model, save_path=args.result_path,
                                          save=True, correct=True)
            else:
                for t in techniques:
                    grounding = gen_grounding(displ_img, t, gt_label_name, args.model, save_path=args.result_path, save=True, correct=True)
        else:
            print('incorrect classification')

            ''' generate explanations '''
            if len(techniques) == 4:
                grounding = gen_all_groundings(displ_img, gt_label_name, args.model, save_path=args.result_path,
                                          save=True, correct=False)
            else:
                for t in techniques:
                    grounding = gen_grounding(displ_img, t, gt_label_name, args.model, save_path=args.result_path, save=True, correct=False)
