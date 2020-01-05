import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import sys
import csv
import random
from datetime import datetime
import torch
from metrics.utils import *
from techniques.generate_grounding import gen_grounding

def write_to_csv(row):
    with open(filename, 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)


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
                        default="results/",
                        type=list,
                        nargs='+',
                        help="Location of results file")
    parser.add_argument("--techniques",
                        default=['gcam', 'lime'],
                        type=list,
                        nargs='+',
                        help="Techniques to test")
    parser.add_argument("--models",
                        default=['resnet18', 'vgg19'],
                        type=list,
                        nargs='+',
                        help="Models to test")
    parser.add_argument("--cuda",
                        default=4,
                        type=int,
                        help="device")
    args = parser.parse_args(sys.argv[1:])
    
    torch.cuda.set_device(args.cuda)
    
    filename = 'results/base-eval-%s.csv'%datetime.now().strftime('%Y-%m-%d-%H-%M')
    paths = []
    for r, d, f in os.walk('/work/lisabdunlap/explain-eval/data/ILSVRC2012_img_val/'):
        for file in f:
            if '.JPEG' in file:
                paths.append(os.path.join(r, file))
    random.shuffle(paths)
    data_iter = iter(paths)
    print("Number of test images: {0}".format(len(paths)))
    write_to_csv(['path', 'threshold', 'test', 'model', 'pixel count iou', 'cos similarity',
                                                                 'jenson shannon dist', 'total variation dist'])
    i=0
    while i<args.sample_size:
        i = i+1
        if i%10 == 0:
            print("image {0}".format(i))
        path1 = next(data_iter)
        if os.path.isfile(path1):
            img1 = cv2.imread(path1, 0)
        else:
            print ("The file " + path1 + " does not exist.")
        img = cv2.imread(path1)
        val_img = cv2.resize(img, (224, 224))

        techniques = ['gcam', 'lime', 'rise', 'ig']
        thresholds = [15, 25, 50]
        mask_dict_resnet18 = {}
        mask_dict_vgg19 = {}
        for t in techniques:
                mask_dict_resnet18[t] = gen_grounding(val_img, t, 'random', show=False)
                mask_dict_vgg19[t] = gen_grounding(val_img, t, 'random', show=False)
                print('done with {0}:'.format(t))

        for thresh in thresholds:
            for t in techniques:
                row = [path1, thresh, t, None]
                row += get_stats(mask_dict_resnet18[t], mask_dict_vgg19[t], threshold=thresh)
                print("===================================================================")
                write_to_csv(row)
                
        for thresh in thresholds:
            for i in range(len(techniques)):
                for j in range(i, len(techniques)):
                    if techniques[i] != techniques[j]:
                        row = [path1, thresh, techniques[i]+"-"+techniques[j], "resnet18"]
                        row += get_stats(mask_dict_resnet18[techniques[i]], mask_dict_resnet18[techniques[j]], threshold=thresh)
                        print("===================================================================")
                        write_to_csv(row)
                        row = [path1, thresh, techniques[i]+"-"+techniques[j], "vgg19"]
                        row += get_stats(mask_dict_vgg19[techniques[i]], mask_dict_vgg19[techniques[j]], threshold=thresh)
                        write_to_csv(row)
                        print("===================================================================")
        