import os
import argparse
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torchvision import models
from datetime import datetime
import matplotlib.pyplot as plt

from fooling_network_interpretation.gradcam_targeted_patch_attack import *
from techniques.utils import get_model, get_model_layer, get_model_info

def gen_adversarial_patch(img, model_name, label_name, save_path='./results/explanation_examples/', show=True, save=True):
    
    # Create result directory if it doesn't exist; all explanations sshould 
    # be stored in a folder that is the predicted class
    save_path = save_path+label_name+'/patch/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Setting the seed for reproducibility for demo
    # Comment the below 4 lines for the target category to be random across runs
    #np.random.seed(1)
    #torch.manual_seed(1)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    # Can work with any model, but it assumes that the model has a feature method,
    # and a classifier method, as in the VGG models in torchvision
    pretrained_net = get_model(model_name)
    net_layer = get_model_layer(model_name)
    gradcam_attack = GradCamAttack(model=pretrained_net, target_layer_names=[net_layer])
    gradcam_reg_patch_attack = GradCamRegPatchAttack(model=pretrained_net, target_layer_names=[net_layer])
    gradcam = GradCam(model=pretrained_net, target_layer_names=[net_layer])
    if torch.cuda.is_available():
        pretrained_net = pretrained_net.cuda()
    pretrained_net = pretrained_net.eval()
    #image_name = args.image_path.split('/')[-1].split('.')[0]

    # Create result directory if it doesn't exist
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # Read the input image and preprocess to a tensor
    #img = cv2.imread(args.image_path, 1)
    #img = np.float32(cv2.resize(img, (224, 224))) / 255
    preprocessed_img = preprocess_image(img)

    # Get the original prediction index and the corresponding probability
    orig_index, orig_prob = forward_inference(pretrained_net, preprocessed_img)

    # Pick a random target from the remaining 999 categories excluding the original prediction
    list_of_idx = np.delete(np.arange(1000), orig_index)
    rand_idx = np.random.randint(999)
    target_index = list_of_idx[rand_idx]

    # Compute the regular adv patch attack image and the corresponding GradCAM
    reg_patch_adv_img, reg_patch_adv_tensor = gradcam_reg_patch_attack(preprocessed_img, orig_index, target_index)
    reg_patch_pred_index, reg_patch_pred_prob = forward_inference(pretrained_net,
                                                                  preprocess_image(reg_patch_adv_img[:, :, ::-1]))
    
    # save adversarial image
    if save:
        #cv2.imwrite(os.path.join(save_path + 'patch_image-%s.png'%datetime.now().strftime('%Y-%m-%d-%H-%M')),
        #            np.uint8(255 * np.clip(reg_patch_adv_img[:, :, ::-1], 0, 1)))
        np.save(os.path.join(save_path + 'patch_image-%s.png'%datetime.now().strftime('%Y-%m-%d-%H-%M')), reg_patch_adv_img)

    # Generate the GradCAM heatmap for the target category using the regular patch adversarial image
    reg_patch_adv_mask = gradcam(reg_patch_adv_tensor, target_index)
    #gcam_expl, reg_patch_adv_mask = gen_grounding(reg_patch_adv_img, 'vgg19_bn', 'gcam', label_name, show=True)
    if show:
        displ_img = show_cam_on_image(np.clip(reg_patch_adv_img[:, :, ::-1], 0, 1), reg_patch_adv_mask,
                          filename=os.path.join(save_path + 'patch_image_gcam-%s.png'%datetime.now().strftime('%Y-%m-%d-%H-%M')))
        plt.imshow(reg_patch_adv_img)

    print('finished generating adveersarial patch')
    return reg_patch_adv_img, reg_patch_adv_mask, target_index