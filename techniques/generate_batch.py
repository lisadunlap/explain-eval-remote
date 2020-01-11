import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torchvision import models, transforms
from skimage import io
import cv2
import os
from datetime import datetime
import gc
import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid

#techniques
from Grad_CAM.main_gcam import gen_gcam, gen_gcam_target
from utils import get_model,get_displ_img
from data_utils.data_setup import get_model_info, get_imagenet_classes
from data_utils.gpu_memory import dump_tensors

def get_displ_img(img):
    try:
        img = img.cpu().numpy().transpose((1, 2, 0))
    except:
        img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    displ_img = std * img + mean
    displ_img = np.clip(displ_img, 0, 1)
    displ_img /= np.max(displ_img)
    displ_img = displ_img
    return np.uint8(displ_img*255)

'''
Displayes mask as heatmap on image
'''
def get_cam(img, mask):
    img = get_displ_img(img)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                               cv2.COLOR_BGR2RGB)
    alpha = .4
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    return cam

''' Generate for dataloader
IMGS: list of tensors
MODEL: model name or model itself 
LABEL_NAME: name of class/unique name (for saving the image) 
FROM_SAVED: if using pytorch pretrained (so you pass in a string name for MODEL)
TARGET_INDEX: index of class, if TOPK=True, then it uses the predicted class
TOPK= if predict top index rather than specific index
LAYER: last convolutional layer of the network
DEVICE: cuda device number
CLASSES: list of class names
SAVE: if you want to save the result
SAVE_PATH: path to save to
SHOW: plot (for debugging)
'''
def gen_grounding_gcam_batch(imgs,
                  model='resnet18',
                  label_name='explanation',
                  from_saved=True, 
                  target_index=1,
                  layer='layer4', 
                  device=0,
                  topk=True,
                  classes=get_imagenet_classes(), 
                  save=True,
                  save_path='./results/gradcam_examples/', 
                  show=True):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if save:
        print('result path: {0}'.format(save_path))
        
    if from_saved == True:
        model, classes, layer = get_model_info(model)
    
    # Generate the explanations
    if topk:
        masks = gen_gcam(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    else:
        print('batch target indicies: ', target_index)
        masks = gen_gcam_target(imgs, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False, classes=classes)
    
    cams = []
    for mask, img in zip(masks, imgs):
        cams += [get_cam(img, mask)]
    
    if show:
        #plot heatmaps
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),
                         axes_pad=0.35,  # pad between axes in inch.
                         )

        for ax, im in zip(grid, cams[:4]):
            ax.axis('off')
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            
    if save:
        for i in range(len(imgs)):
            print("saving explanation mask....\n")
            cv2.imwrite(os.path.join(save_path + classes[target_index[i]] +'-original_img.png'), get_displ_img(imgs[i]))
            if not cv2.imwrite(os.path.join(save_path+ classes[target_index[i]]+".png"), np.uint8(cams[i]*255)):
                print('error saving explanation')
            #print('saved to {0}'.format(os.path.join(save_path)))
        
    #just in case
    torch.cuda.empty_cache()

    return masks