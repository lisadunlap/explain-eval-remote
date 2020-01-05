import torch
import gc
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torchsample
import matplotlib.pyplot as plt

def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                    type(obj.data).__name__,
                    " GPU" if obj.is_cuda else "",
                    " pinned" if obj.data.is_pinned else "",
                    " grad" if obj.requires_grad else "",
                    " volatile" if obj.volatile else "",
                    pretty_size(obj.data.size())))
                total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)
    
'''data_transforms = {
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
}'''
    
def apply_dual_transform(image_left, image_right, trans, train=True):
    
    if train:
        # Resize
        resize = transforms.Resize(size=(256, 256))
        image_left = resize(image_left)
        image_right = resize(image_right)
        
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(image_left, output_size = image_left.size)
        image_left = TF.resized_crop(image_left, i, j, h, w, 224)
        image_right = TF.resized_crop(image_right, i, j, h, w, 224)

        # Random horizontal flipping
        if random.random() > 0.5:
            image_left = TF.hflip(image_left)
            image_right = TF.hflip(image_right)

        # Random rotation
        degree = random.randint(-30,30)
        image_left = TF.rotate(image_left,degree)
        image_right = TF.rotate(image_right,degree)
        
        image_left = trans(image_left)
    
    else:
        # Resize
        resize = transforms.Resize(size=(256, 256))
        image_left = resize(image_left)
        image_right = resize(image_right)
        
        #center crop
        image_left = TF.center_crop(image_left, 224)
        image_right = TF.center_crop(image_right, 224)
        
        image_right = TF.to_tensor(image_right)
        
    image_right = TF.to_tensor(image_right)
    
    return image_left, image_right