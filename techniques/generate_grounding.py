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
#from Grad_CAM.grad_cam import gen_gcam
#from Grad_CAM.old_grad_cam import old_gen_gcam, get_guidedBackProp_img
from Grad_CAM.main_gcam import gen_gcam
from Integrated_Gradients.integrated_gradients import generate_ig
from LIME.LIME import generate_lime_explanation
from RISE.rise_utils import gen_rise_grounding
from utils import get_model, get_model_info, get_displ_img
from data_utils.gpu_memory import dump_tensors

#if torch.cuda.is_available():
#    torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
sv_pth = './results/master_examples/'

def gen_grounding(img,
                  technique,
                  label_name = 'explanation',
                  model='resnet18',
                  path=None, 
                  show=False, 
                  reg=False, 
                  save_path='./results/master_examples/',
                  target_index=1,
                  unique_id=None,
                  patch=False, 
                  save=True,
                  correct=True,
                  device=5):
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
    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model)
    else:
        model_name = 'custom'
    
    # Generate the explanations
    if technique == 'lime' or technique == 'LIME':
        mask = generate_lime_explanation(img, model, pred_rank=target_index, positive_only=True, show=show)
    elif technique == 'gradcam' or technique == 'GradCam' or technique == 'gcam':
        mask = gen_gcam([img], model, target_index = target_index, target_layer=target_layer, show_labels=True)
    elif technique == 'ig' or technique == 'integrated-gradients':
        mask = generate_ig(img, model, reg=reg, cuda=torch.cuda.is_available())
    elif technique == 'rise' or technique == 'RISE':
        mask = gen_rise_grounding(img, model, index=target_index, cuda=torch.cuda.is_available())
    elif technique == 'gbp' or technique == 'guided-backprop':
        mask = get_guidedBackProp_img(img, model, reg=reg)
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
    alpha = .7
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    #print("ccam {0} heatmap {1}".format(cam.shape, heatmap.shape))
    #alpha=.5
    #cam = cv2.addWeighted(cam, alpha, heatmap, 1 - alpha,
	#	0, heatmap)

    if show:
        plt.axis('off')
        plt.imshow(cam)
   
    if save:
        print("saving explanation mask....\n")
        np.save(os.path.join(save_path + 'original_img'), img)
        cv2.imwrite(os.path.join(save_path + 'original_img.png'), img)
        np.save(os.path.join(save_path + technique + '-'+ model_name), mask)
        if not cv2.imwrite(os.path.join(save_path + technique + '-' + model_name + '/'+ str(target_index)+".png"), np.uint8(cam*255)):
            print('error saving explanation')
        print('saved to {0}'.format(os.path.join(save_path + technique + '-'+ model_name)))

    #print('------------------------------')
    #torch.cuda.empty_cache()
    #dump_tensors()
    #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    return mask

def gen_grounding_gcam(img,
                  label_name = 'explanation',
                  model='resnet18',
                  show=False, 
                  from_saved=True, 
                  save_path='./results/master_examples/',
                  target_index=1,
                  unique_id=None,
                  layer='layer4', 
                  save=True,
                  device=0,
                  list=False,
                  tensor=False):
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
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        print('result path: {0}'.format(save_path))
        
    # convert image if needed
    if not tensor:
        if np.max(img) < 2:
            img = np.uint8(img*255)
    if from_saved == True:
        model, classes, layer = get_model_info(model)
    
    # Generate the explanations
    #mask = old_gen_gcam(img, model, target_index = target_index, target_layer=layer, from_saved=False)
    if not tensor:
         mask = gen_gcam([img], model, target_index = target_index, target_layer=layer, device=device)
    else:
        mask = gen_gcam([img], model, target_index = target_index, target_layer=layer, device=device, prep=False)
    mask /= np.max(mask)
    if tensor:
        img = get_displ_img(img)
        img /= np.max(img)
        img = np.uint8(img*255)
    w, h, _ = img.shape
    m=cv2.resize(mask, (w, h))
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask / np.max(mask)) * 255.0), cv2.COLORMAP_JET),
                               cv2.COLOR_BGR2RGB)
    alpha = .6
    cam = heatmap*alpha + np.float32(img)*(1-alpha)
    cam /= np.max(cam)
    if save:
        print("saving explanation mask....\n")
        np.save(os.path.join(save_path + 'original_img'), img)
        cv2.imwrite(os.path.join(save_path + 'original_img.png'), img)
        np.save(os.path.join(save_path + 'gcam'), mask)
        cv2.imwrite(os.path.join(save_path + 'gcam' + ".png"), cam*255)
        print('saved to {0}'.format(os.path.join(save_path + 'gcam' + '-')))

    if show:
        plt.imshow(cam)

    #print('------------------------------')
    #torch.cuda.empty_cache()
    #dump_tensors()
    #print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    return mask

def gen_grounding_gcam_batch(img,
                  label_name = 'explanation',
                  model='resnet18',
                  show=False, 
                  from_saved=True, 
                  save_path='./results/master_examples/',
                  target_index=1,
                  unique_id=None,
                  layer='layer4', 
                  save=True,
                  device=0):
    #CUDA_VISIBLE_DEVICES=str(device)
    # Create result directory if it doesn't exist; all explanations should 
    # be stored in a folder that is the predicted class
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H")
    
    print('------------------------------')
    torch.cuda.empty_cache()
    dump_tensors()
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    save_path += label_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not unique_id == None:
        save_path += unique_id + '/'
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save:
        print('result path: {0}'.format(save_path))
        
    # convert image if needed
    if not list:
        img = [img]
    #if np.max(img) < 2:
    #    img = np.uint8(img*255)
    if from_saved == True:
        model, classes, layer = get_model_info(model)
    
    # Generate the explanations
    masks = gen_gcam(img, model, target_index = target_index, target_layer=layer, device=device, single=False, prep=False)
    
    print('------------------------------')
    torch.cuda.empty_cache()
    dump_tensors()
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

    return masks



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
    for technique, ax_idx in zip(['rise', 'lime', 'ig'], [axarr[0,0], axarr[0,1], axarr[1,0], axarr[1,1]]):
        mask = gen_grounding(img, technique, label_name, model_name, path=path, show=False, save_path=save_path,save=save)
        groundings[technique] = mask
        if show:
            ax_idx.imshow(mask)
            ax_idx.set_title(technique)
    if save:
        f.savefig(os.path.join(save_path + 'all techniques'))
    if show:
        plt.show()

    return groundings