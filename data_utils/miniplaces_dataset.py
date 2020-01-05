from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torch

from techniques.utils import *
from data_utils.gpu_memory import apply_dual_transform

class MiniPlacesDataset(Dataset):
    def __init__(self, **kwargs):
        # load in the data
        self.photos_path = kwargs['photos_path']
        self.labels_path = kwargs['labels_path']
        self.transform = kwargs['transform']
        self.location_paths = kwargs['location_paths']
        self.train = kwargs['train']
        self.load_size = 224
        self.images = []
        #self.masks = []
        self.labels = []
        self.locations = []

        # read the text file
        with open(self.labels_path, 'r') as f:
            for line in f:
                path, label = line.strip().split(" ")
                self.images.append(path)
                self.labels.append(label)

        with open(self.location_paths, 'r') as ff:
            for line in ff:
                path, loc = line.strip().split(" ")
                loc = loc.split(',')
                self.locations.append(list(loc))

        self.images = np.array(self.images, np.object)
        self.labels = np.array(self.labels, np.int64)
        print("# images found at path '%s': %d" % (self.labels_path, self.images.shape[0]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.photos_path, self.images[idx]))
        #image = self.transform(image)
        # label is the index of the correct category
        label = self.labels[idx]
        loc = self.locations[idx]
        
        #transform location
        mask = Image.fromarray(get_img_mask(image, loc))
        image, mask = apply_dual_transform(image, mask, self.transform, self.train)
        
        return (image, label, os.path.join(self.images[idx]), mask)
