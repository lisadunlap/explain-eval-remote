import numpy as np
from torchvision import models, transforms, datasets
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import random
from torch.utils.data.sampler import Sampler
#from data_utils.miniplaces_dataset import MiniPlacesDataset

OBJ_NAMES = [
    'backpack', 'bird', 'dog', 'elephant', 'kite', 'pizza', 'stop_sign',
    'toilet', 'truck', 'zebra'
]
SCENE_NAMES = [
    'bamboo_forest', 'bedroom', 'bowling_alley', 'bus_interior', 'cockpit',
    'corn_field', 'laundromat', 'runway', 'ski_slope', 'track/outdoor'
]


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)


def get_model_info(model_name, other_layer = None):
    print('model name ', model_name)
    if 'ckpt' in model_name:
        print('model numbr', model_name)
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model.load_state_dict(torch.load('/work/lisabdunlap/explain-eval/training/checkpoint/'+model_name+'.pth')['net'])
        model.eval()
        return model, cifar_classes, 'layer4.1'
    if model_name == 'scene' or model_name == 'bam_scene':
        print("this is the obj model")
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        if cuda:
            model.load_state_dict(torch.load('/Users/lisadunlap/bam/pytorch_models/scenemodel_best.pth.tar')['state_dict'])
        else:
            model.load_state_dict(
                torch.load('/Users/lisadunlap/bam/pytorch_models/scenemodel_best.pth.tar', map_location='cpu')['state_dict'])
        return model, OBJ_NAMES, 'layer4.2'
    if model_name == 'obj' or model_name == 'bam_obj':
        print("this is the obj model")
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        if cuda:
            model.load_state_dict(torch.load('/work/lisadunlap/bam/pytorch_models/objmodel_best.pth.tar')['state_dict'])
        else:
            model.load_state_dict(
                torch.load('/work/lisadunlap/bam/pytorch_models/obj2model_best.pth.tar', map_location='cpu')['state_dict'])
        return model, OBJ_NAMES, 'layer4.2'
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224,
            'layer_name': '4.2'
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224, 
            'layer_name': '36'
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224,
            'layer_name': '52'
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299,
            'layer_name': 'Mixed_7c'
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224,
            'layer_name': 'denseblock4'
        },
        'resnet18': {
            'target_layer': 'layer4.1',
            'input_size': 224,
            'layer_name': 4.1
        },
    }.get(model_name)
    classes = list()
    try:
        with open('../data/synset_words.txt') as lines:
            for line in lines:
                line = line.strip().split(' ', 1)[1]
                line = line.split(', ', 1)[0].replace(' ', '_')
                classes.append(line)
    except:
        with open('./data/synset_words.txt') as lines:
            for line in lines:
                line = line.strip().split(' ', 1)[1]
                line = line.split(', ', 1)[0].replace(' ', '_')
                classes.append(line)
    model = models.__dict__[model_name](pretrained=True)
    if torch.cuda.is_available():
        model = model.cuda()
    if other_layer:
        layer_name = other_layer
    else:
        layer_name = CONFIG['layer_name']
    return model, classes, layer_name

def get_model(model_name):
    return get_model_info(model_name)[0]

def get_imagenet_classes():
    classes = list()
    try:
        with open('/work/lisabdunlap/explain-eval/data/synset_words.txt') as lines:
            for line in lines:
                line = line.strip().split(' ', 1)[1]
                line = line.split(', ', 1)[0].replace(' ', '_')
                classes.append(line)
    except:
         with open('/work/lisabdunlap/explain-eval/data/synset_words.txt') as lines:
            for line in lines:
                line = line.strip().split(' ', 1)[1]
                line = line.split(', ', 1)[0].replace(' ', '_')
                classes.append(line)
    return classes

"""def get_test_loader(data_dir,
                    name,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    if name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, 
                                   train=False, 
                                   download=True,
                                   transform=transform)
    else:
        dataset = datasets.CIFAR100(root=data_dir, 
                                    train=False, 
                                    download=True,
                                    transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return dataset,  data_loader"""

''' Get imagenet test loader '''
def get_imagenet_test(datadir='../data/test/',
                      shuffle=True,
                      batch_size=1,
                      sample_size=5,
                      all=False,
                      name=None):
    # Image preprocessing function
    preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    # Normalization for ImageNet
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

    if name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=preprocess)
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir,
                                    train=False,
                                    download=True,
                                    transform=preprocess)
    elif name == 'scene':
        dataset = get_miniplaces('scene')
    else:
        dataset = datasets.ImageFolder(datadir, preprocess)

    #if not all:
    #    range_start = random.randint(0, len(dataset) - 1)
    #    dataset = dataset[range_start, range_start+sample_size]

    ''' Randomly pick a range to sample from '''
    #print("sample size {0}".format(sample_size))
    if not all:
        sample = []
        while True:
            if len(sample) == sample_size:
                break
            range_start = random.randint(0,len(dataset)-sample_size)
            if range_start not in sample:
                sample += [range_start]
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=8, sampler=RangeSampler(sample))
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=8)

    return dataset, data_loader

def get_dataloader(dataset,
                   batch_size=1,
                   shuffle=False,
                   sample_size=100,
                   all=False):
    if not all:
        sample = []
        while True:
            if len(sample) == sample_size:
                break
            range_start = random.randint(0,len(dataset)-sample_size)
            if range_start not in sample:
                sample += [range_start]
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=8, sampler=RangeSampler(sample))
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=8)
    return data_loader

def get_miniplaces(name='scene'):
    data_transforms = {
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '/work/lisabdunlap/bam/data/' + name+ '/'

    dataset = MiniPlacesDataset(
        photos_path=os.path.join(data_dir),
        labels_path=os.path.join(data_dir, 'val2.txt'),
        transform=data_transforms['val']
    )

    return dataset


def get_top_prediction(model_name, img):
    model = get_model(model_name)
    classes = get_imagenet_classes()
    logits = model(img)
    probs = F.softmax(logits, dim=1)
    prediction = probs.topk(5)
    return classes[prediction[1][0].detach().cpu().numpy()[0]], prediction[1][0].detach().cpu().numpy()[0]