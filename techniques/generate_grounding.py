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

#techniques
from Grad_CAM.grad_cam import gen_gcam
from Integrated_Gradients.integrated_gradients import generate_ig
from LIME.LIME import generate_lime_explanation
from RISE.rise_utils import gen_rise_grounding
from utils import get_model
from data_utils.gpu_memory import dump_tensors

#if torch.cuda.is_available():
#    torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
sv_pth = './results/master_examples/'

def gen_grounding(img,
                  technique,
                  label_name = 'explanation',
                  model_name='resnet18',
                  path=None, 
                  show=False, 
                  reg=False, 
                  save_path='./results/master_examples/',
                  index=1,
                  unique_id=None,
                  patch=False, 
                  save=True,
                  correct=True,
                  device=5, 
                  has_model = None):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #save_path += label_name+'-'+timestampStr+'/'
    if not unique_id == None:
        save_path += unique_id + '/'
    
    if patch:
        save_path = os.path.join(save_path+'patch/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        print('result path: {0}'.format(save_path))
        
    # convert image if needed
    if np.max(img) < 2:
        img = np.uint8(img*255)
    else:
        if model_name == 'bam_scene':
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            model.load_state_dict(torch.load('/work/lisabdunlap/bam/scenemodel_best.pth.tar')['state_dict'])
        elif model_name == 'obj':
            model = models.resnet50(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            model.load_state_dict(torch.load('/work/lisabdunlap/bam/pytorch_models/obj2model_best.pth.tar')['state_dict'])
        else:
            model = get_model(model_name)
    
    # Generate the explanations
    if technique == 'lime' or technique == 'LIME':
        mask = generate_lime_explanation(img, model, pred_rank=index, positive_only=True, show=show)
    elif technique == 'gradcam' or technique == 'GradCam' or technique == 'gcam':
        mask = gen_gcam(img, model_name, show=show, target_index = index, has_model=has_model)
    elif technique == 'ig' or technique == 'integrated-gradients':
        mask = generate_ig(img, model, show=show, reg=reg, cuda=torch.cuda.is_available())
    elif technique == 'rise' or technique == 'RISE':
        mask = gen_rise_grounding(img, model, index=index, cuda=torch.cuda.is_available())
    elif technique == 'gbp' or technique == 'guided-backprop':
        mask = get_guidedBackProp_img(img, model, show=show, reg=reg)
    elif technique == 'excitation backprop' or technique == 'eb':
        if 'resnet' in model_name:
            print("Resnet models have yet to be implemented with EB")
            return
        else:
            mask = gen_eb(path, model, show=show)
    else:
        print('ERROR: invalid explainability technique {0}'.format(technique))
        return

    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    alpha = .4
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    #print("ccam {0} heatmap {1}".format(cam.shape, heatmap.shape))
    #alpha=.5
    #cam = cv2.addWeighted(cam, alpha, heatmap, 1 - alpha,
	#	0, heatmap)

    if show:
        plt.imshow(cam)
   
    if save:
        print("saving explanation mask....\n")
        np.save(os.path.join(save_path + 'original_img'), img)
        cv2.imwrite(os.path.join(save_path + 'original_img.png'), img)
        np.save(os.path.join(save_path + technique + '-'+ model_name), mask)
        cv2.imwrite(os.path.join(save_path + technique + '-' + model_name + ".png"), cam*255)
        print('saved to {0}'.format(os.path.join(save_path + technique + '-'+ model_name)))

    #print('------------------------------')
    #torch.cuda.empty_cache()
    #dump_tensors()
    #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    return mask

''' Generates explanations for RISE, LIME, GradCAM, and IntegratedGradients'''
def gen_all_groundings(img,
                  label_name,
                  model_name='resnet18',
                  path=None, 
                  show=False, 
                  reg=False, 
                  save_path='./results/master_examples/',
                  unique_id=None,
                  index=1, 
                  patch=False, 
                  save=True,
                  correct=True):


    # Create result directory if it doesn't exist; all explanations should
    # be stored in a folder that is the predicted class
    old_path = save_path
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H_%M")

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if unique_id == None:
        save_path += label_name + '-' + timestampStr + '/'
    else:
        save_path += unique_id + '/'
    
    if save:
        cv2.imwrite(os.path.join(save_path + label_name + '-original.png'), np.float32(img * 255))
        
    # convert image if needed
    if np.max(img) < 2:
        img = np.uint8(img*255)

    groundings = {}
    # gen all groundings
    f, axarr = plt.subplots(2,2,figsize=(10,10))
    for technique, ax_idx in zip(['rise', 'gcam', 'lime', 'ig'], [axarr[0,0], axarr[0,1], axarr[1,0], axarr[1,1]]):
        mask = gen_grounding(img, technique, label_name, model_name, path=path, show=False, reg=reg, save_path=save_path,
                             index=index, patch=patch, save=save, correct=correct)
        groundings[technique] = mask
        if show:
            ax_idx.imshow(mask)
            ax_idx.set_title(technique)
    if save:
        f.savefig(os.path.join(save_path + 'all techniques'))
    if show:
        plt.show()

    return groundings