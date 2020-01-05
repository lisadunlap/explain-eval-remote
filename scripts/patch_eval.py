import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import sys
import csv
import random
from datetime import datetime
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
                        default=100,
                        type=int,
                        help="Number of images to test.")
    parser.add_argument("--thresholds",
                        default=1,
                        type=list,
                        help="Thresholds to test.")
    parser.add_argument("--result-path",
                        default="./results/",
                        type=str,
                        help="Location of results file")
    parser.add_argument("--techniques",
                        default=['gcam', 'lime'],
                        type=list,
                        nargs='+',
                        help="Techniques to test")
    parser.add_argument("--model",
                        default='vgg19',
                        type=str,
                        help="Models to test")
    parser.add_argument("--datadir",
                        default='./data/test/',
                        type=str,
                        nargs='+',
                        help="location of data")
    args = parser.parse_args(sys.argv[1:])
    
    dataset, data_loader = get_imagenet_test(datadir=args.datadir, shuffle=True)
    labels = list(os.walk(args.datadir))[0][1]
    list.sort(labels)

    print("starting eval for sample size {0}".format(args.sample_size))
    print('------------------------------')
    
    # create ressults directory if it doesnt exist
    if not os.path.exists(args.result_path):
                os.makedirs(args.result_path)
    filename = args.result_path + 'metrics/patch-eval-%s.csv'%datetime.now().strftime('%Y-%m-%d-%H-%M')
    base_eval_filename = args.result_path + 'metrics/base-eval-%s.csv'%datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    write_to_csv(['threshold', 'test', 'model', 'pixel count iou', 'cos similarity',
                                                                 'jenson shannon dist', 'total variation dist', 'pearsons correlation coefficient'], base_eval_filename)
    write_to_csv(['threshold', 'test', 'model', 'pixel count iou', 'pixel overlap'], filename)

    num_images=0
    for i, (inp, label) in enumerate(data_loader):
        if num_images < args.sample_size:
            print(label)
            print('entering loop {0}'.format(num_images))
            gt_label = labels[label.numpy()[0]]
            gt_label_name, gt_label_idx = find_label(gt_label)
            print('ground truth class: {0}'.format(gt_label_name, gt_label_idx))
            # get data
            img = inp[0].numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            displ_img = np.clip((std * img + mean), 0, 1)
            save_img = displ_img/np.max(displ_img)
            displ_img = np.uint8((displ_img/np.max(displ_img))*255)
            label_name, label_idx = get_top_prediction('vgg19', inp)
            #label_name_18, label_idx_18 = get_top_prediction('resnet18', inp)
            #print('predicted class resnet18: {0}  index: {1}'.format(label_name_18, label_idx_18))
            print('predicted class: {0}  index: {1}'.format(label_name, label_idx))
            if ((label_name == gt_label_name)):
                print('correct classification')
                num_images += 1
                print('original label name: {0}'.format(label_name))

                result_path = os.path.join(args.result_path + 'master_examples/' + label_name +'/')
                print(result_path)

                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                np.save(os.path.join(result_path + 'original_img-%s'%datetime.now().strftime('%Y-%m-%d-%H-%M')), displ_img)
                print('saving image to {0}'.format(os.path.join(result_path + 'original_img-%s'%datetime.now().strftime('%Y-%m-%d-%H-%M'))))

                # generate explanations
                techniques = ['gcam','lime', 'ig']
                #techniques = ['gcam']
                thresholds = [15, 25, 50]
                mask_dict_model = {}
                mask_dict_resnet18 = {}
                mask_dict_model_patch = {}
                reg_patch_adv_img, reg_patch_adv_mask, target_index = gen_adversarial_patch(img, args.model, label_name, show=False, save=True)
                print('new adversarial prediction: {0}'.format(get_top_prediction('vgg19', read_tensor(reg_patch_adv_img)))[0])
                mask_dict_model = gen_all_groundings(displ_img, 'vgg19', gt_label_name, show=True, label_index=num_images)
                mask_dict_resnet18 = gen_all_groundings(displ_img, 'vgg19', gt_label_name, show=True, label_index=num_images)
                mask_dict_model_patch = gen_all_groundings(np.uint8((reg_patch_adv_img/np.max(reg_patch_adv_img))*255), 'vgg19', gt_label_name, show=True, label_index=num_images)
                """for t in techniques:
                    mask_dict_model[t] = gen_grounding(img, args.model, t, label_name, show=False, save=True)[1]
                    mask_dict_model_patch[t] = \
                    gen_grounding(reg_patch_adv_img / np.max(reg_patch_adv_img), args.model, t, label_name, show=False,
                                  save=True, patch=True)[1]
                    mask_dict_resnet18[t] = gen_grounding(img, 'resnet18', t, label_name, show=False, save=True)[1]"""

                # calculate patch iou
                for thresh in thresholds:
                    for t in techniques:
                        row = [thresh, t, args.model]
                        row += [patch_iou(mask_dict_model[t], threshold=thresh), percent_covered(mask_dict_model_patch[t], threshold=thresh)]
                        write_to_csv(row, filename)
                print('finihed writing to {0}'.format(filename))

                # redoing the basic metrics
                for thresh in thresholds:
                    for t in techniques:
                        row = [thresh, t, None]
                        row += get_stats(mask_dict_resnet18[t], mask_dict_model[t], threshold=thresh)
                        write_to_csv(row, base_eval_filename)
    
                for thresh in thresholds:
                    for i in range(len(techniques)):
                        for j in range(i, len(techniques)):
                            if techniques[i] != techniques[j]:
                                row = [thresh, techniques[i]+"-"+techniques[j], "resnet18"]
                                row += get_stats(mask_dict_resnet18[techniques[i]].astype(float), mask_dict_resnet18[techniques[j]].astype(float), threshold=thresh)
                                write_to_csv(row, base_eval_filename)
                                row = [thresh, techniques[i]+"-"+techniques[j], "vgg19"]
                                row += get_stats(mask_dict_model[techniques[i]].astype(float), mask_dict_model[techniques[j]].astype(float), threshold=thresh)
                                write_to_csv(row, base_eval_filename)