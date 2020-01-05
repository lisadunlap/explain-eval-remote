import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.functional import conv2d

from techniques.rise_utils import *
from techniques.evaluation import CausalMetric, auc, gkern
from techniques.utils import get_stats
from techniques.saliency import gen_grounding

from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import csv
import os
import cv2
import random
from datetime import datetime
import argparse
import sys

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def calc_auc(img, expl):
    scd = auc(deletion.single_run(img, expl))
    sci = auc(insertion.single_run(img, expl))
    return scd, sci

def write_to_csv(filename, row):
    with open(filename, 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                
paths = ['data/ILSVRC2012_img_val/ILSVRC2012_val_00022759.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00032660.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00028416.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00028247.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00016273.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00044429.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00036091.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00018439.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00033769.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00031754.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00030846.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00006427.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00020737.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00037884.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00035686.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00008593.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00019285.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00000480.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00013241.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00042375.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00005774.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00020740.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00030558.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00031739.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00007584.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00023493.JPEG']
    
    
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
    args = parser.parse_args(sys.argv[1:])
    
    filename = 'results/auc-eval-%s.csv'%datetime.now().strftime('%Y-%m-%d-%H-%M')
    write_to_csv(filename, ['path', 'method', 'model', 'deletion auc', 'insertion auc'])
    
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)

    # Function that blurs input image
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    #for r, d, f in os.walk('data/ILSVRC2012_img_val/'):
    #    for file in f:
    #        if '.JPEG' in file:
    #            paths.append(os.path.join(r, file))
    #random.shuffle(paths)
    data_iter = iter(paths)
    print("Number of test images: {0}".format(len(paths)))

    for i in range(args.sample_size):
        path = next(data_iter)
        img = read_tensor(path)
        val_img = cv2.imread(path)
        val_img = cv2.resize(val_img, (224, 224))
        
        # Load black box model for explanations
        model = models.resnet18(True)
        model = nn.Sequential(model, nn.Softmax(dim=1))
        model = model.eval()
        
        # define insertion/deletion
        insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
        deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)

        for p in model.parameters():
            p.requires_grad = False
        
        # generate explanations
        gcam_expl, gc = gen_grounding(val_img, 'resnet18', 'gcam', show=False)
        lime_expl, li = gen_grounding(val_img, 'resnet18', 'lime', show=False)
        rise_expl, ri = gen_grounding(val_img, 'resnet18', 'rise', path= path, show=False)
        
        # calc insertion/deletion auc
        gc_scd, gc_sci = calc_auc(img, gc)
        write_to_csv(filename, [path, 'resnet18', 'gcam', gc_scd, gc_sci])
        li_scd, li_sci = calc_auc(img, li)
        write_to_csv(filename, [path, 'resnet18', 'lime', li_scd, li_sci])
        ri_scd, ri_sci = calc_auc(img, ri)
        write_to_csv(filename, [path, 'resnet18', 'rise', ri_scd, ri_sci])
        