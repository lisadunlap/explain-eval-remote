import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy as sp
import cv2
from metrics.utils import *

# calculate the iou for patch: (sum of heatmap values in the image)/(sum of total heatmap values) 
def patch_iou(heatmap, patch_size = 64, patch_location=(0,0), threshold=100):
    new_mask = preprocess_groundings(heatmap, threshold=threshold, binary=True)
    #plt.imshow(new_mask)
    #new_mask[new_mask != 0] = 1
    patch = new_mask[patch_location[1]:patch_location[1]+patch_size, patch_location[0]:patch_location[0]+patch_size]
    patch[patch != 0] = 1
    return np.sum(patch)/np.sum(new_mask)
    
# calculate the percentage of the patch covered
def percent_covered(heatmap, patch_size=64, patch_location=(0,0), threshold=100):
    new_mask = preprocess_groundings(heatmap, threshold=threshold, binary=True)
    patch = new_mask[patch_location[1]:patch_location[1] + patch_size, patch_location[0]:patch_location[0] + patch_size]
    patch[patch != 0] = 1
    return np.sum(patch)/(patch_size*patch_size)
    