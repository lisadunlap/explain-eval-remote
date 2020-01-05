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
import torch.nn as nn
from torch.autograd import Variable
import torch
import torchsample
from torchvision import transforms, datasets, models

from metrics.utils import *
from techniques.generate_grounding import gen_grounding, gen_all_groundings
from data_utils.data_setup import *
#torch.utils.data.sampler.SubsetRandomSampler as SubsetRandomSampler
from torch.utils.data.sampler import Sampler, SubsetRandomSampler,SequentialSampler
from metrics.evaluation import CausalMetric, auc, gkern


def write_to_csv(row, filename):
    with open(filename, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)


import json


OBJ_NAMES = [
    'backpack', 'bird', 'dog', 'elephant', 'kite', 'pizza', 'stop_sign',
    'toilet', 'truck', 'zebra'
]
SCENE_NAMES = [
    'bamboo_forest', 'bedroom', 'bowling_alley', 'bus_interior', 'cockpit',
    'corn_field', 'laundromat', 'runway', 'ski_slope', 'track_outdoor'
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--sample-size",
                        default=3,
                        type=int,
                        help="Number of images to test.")
    parser.add_argument("--sample-start",
                        default=0,
                        type=int,
                        help="Start of image test .")
    parser.add_argument("--thresholds",
                        default=1,
                        type=list,
                        help="Thresholds to test.")
    parser.add_argument("--result-path",
                        default="/work/lisabdunlap/explain-eval/results/bam/",
                        type=str,
                        help="Location of results file")
    parser.add_argument("--techniques",
                        default="rise lime ig gcam",
                        type=str,
                        help="Techniques to test")
    parser.add_argument("--model",
                        default='scene',
                        type=str,
                        help="Models to test")
    parser.add_argument("--all",
                        action='store_true',
                        help="explain whole batch")
    parser.add_argument("--train",
                        action='store_true',
                        help="generate for training set")
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
    #if args.all:
    #    dataset, data_loader = get_imagenet_test(name=args.model, shuffle=True, all=True)
    #else:
    #    dataset, data_loader = get_imagenet_test(name=args.model, shuffle=True, sample_size=args.sample_size)

    techniques = [item for item in args.techniques.split(' ')]

    # create ressults directory if it doesnt exist
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    ################################
    # load and prepare datatloader #
    ################################
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchsample.transforms.RandomRotate(30),
            torchsample.transforms.RandomGamma(0.5, 1.5),
            torchsample.transforms.RandomSaturation(-0.8, 0.8),
            torchsample.transforms.RandomBrightness(-0.3, 0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '/work/lisabdunlap/bam/data/scene/'

    if args.train:
        dataset = MiniPlacesDataset(
            photos_path=os.path.join(data_dir),
            labels_path=os.path.join(data_dir, 'train.txt'),
            transform=data_transforms['train']
        )
    else:
        dataset = MiniPlacesDataset(
            photos_path=os.path.join(data_dir),
            labels_path=os.path.join(data_dir, 'val2.txt'),
            transform=data_transforms['val']
        )

    print("RANGE SAMPLER")
    dataset_sample = torch.utils.data.Subset(dataset, range(args.sample_start, args.sample_start+args.sample_size))
    data_loader = torch.utils.data.DataLoader(
        dataset_sample, batch_size=50,
        num_workers=8)
    #    data_loader = torch.utils.data.DataLoader(
    #        dataset, batch_size=50, shuffle=True,
    #        num_workers=8)

   # data_loader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True, num_workers=25)
    #data_loader = get_dataloader(dataset, sample_size=args.sample_size)

    print("starting eval for sample size {0}".format(len(data_loader)))
    print('------------------------------')

    model = get_model(args.model)
    i = 0
    for data in data_loader:
        inputs, labels, paths = data
        inputs = Variable(inputs.float())
        labels = Variable(labels.long())
        output = model(inputs)
        probabilities, prediction = output.topk(5, dim=1, largest=True, sorted=True)

        num_batch = 0
        stored_dirs = None
        for l in range(len(labels)):
            print("label idx {0} and name {1}".format(labels[l].numpy(), SCENE_NAMES[labels[l]]))
            correct = True
            correct_name = "/correct/"
            if (labels[l] != prediction.numpy()[l][0]) and (not args.train):
                correct = False
                correct_name = "/incorrect/"
            print("path", args.result_path + SCENE_NAMES[labels[l]] + correct_name)
            print("````````````` Batch: {0}   Total: {1} `````````````".format(i, num_batch))
            i += 1
            num_batch += 1
            img_name = paths[l][4:-4].split('/')[2]
            img = inputs[l].numpy().transpose((1, 2, 0))
            print("predicted: {0}  truth: {1}".format(SCENE_NAMES[prediction.numpy()[l][0]],
                                                      SCENE_NAMES[labels[l]]))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            displ_img = np.clip((std * img + mean), 0, 1)
            save_img = displ_img / np.max(displ_img)
            displ_img = np.uint8((displ_img / np.max(displ_img)) * 255)

            if len(techniques) < 4:

                for t in techniques:
                    grounding = gen_grounding(displ_img, t, SCENE_NAMES[labels[l]], 'bam_scene', save_path=args.result_path,
                                              save=True, correct=correct, unique_id=img_name)
            else:
                grounding = gen_all_groundings(displ_img, SCENE_NAMES[labels[l]], 'bam_scene',
                                               save_path=args.result_path,
                                               save=True, correct=correct, unique_id=img_name, show=True)
            print('')


    print("Generated {0} exlpanations".format(i))
