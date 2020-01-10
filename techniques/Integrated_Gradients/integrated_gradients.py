# From https://github.com/TianhongDai/integrated-gradient-pytorch

import numpy as np
import torch
from torchvision import models
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import os

from Integrated_Gradients.ig_utils import calculate_outputs_and_gradients, generate_entrie_images
from Integrated_Gradients.ig_visualization import visualize, img_fill
from techniques.utils import get_model, get_imagenet_classes, read_tensor
from data_utils.data_setup import *

# integrated gradients
def integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False):
    if baseline is None:
        baseline = 0 * inputs 
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    integrated_grad = (inputs - baseline) * avg_grads
    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, steps, num_random_trials, cuda):
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_label_idx, predict_and_gradients, \
                                                baseline=255.0 *np.random.random(inputs.shape), steps=steps, cuda=cuda)
        all_intgrads.append(integrated_grad)
        #print('the trial number is: {}'.format(i))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads

def generate_ig(img, model, cuda=False, show=True, reg=False, outlines=False):
    """ generate Integrated Gradients on given numpy image """
    # start to create models...
    model.eval()
    # for displaying explanation
    if cuda:
        model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=[1, 2, 3, 4, 5, 6, 7])
    #if model_name == 'inception':
        # the input image's size is different
    #    img = cv2.resize(img, (299, 299))
    #print('how about this prediction? {0}'.format(get_top_prediction('vgg19', read_tensor(img))[0]))
    #img = img.astype(np.float32)
    #img = img[:, :, (2, 1, 0)]
    #print('is this a good prediction? {0}'.format(get_top_prediction('vgg19', read_tensor(img))[0]))
    # calculate the gradient and the label index
    gradients, label_index = calculate_outputs_and_gradients([img], model, None, cuda)
    #classes = get_imagenet_classes()
    #print('integrated gradients clasification: {0}'.format(classes[label_index]))
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=95, clip_below_percentile=58, overlay=True, mask_mode=True, outlines=outlines)
    plt.imshow(img_gradient_overlay)
    img_gradient = visualize(gradients, img, clip_above_percentile=95, clip_below_percentile=58, overlay=False, outlines = outlines)

    # calculae the integrated gradients 
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients, \
                                                        steps=50, num_random_trials=10, cuda=cuda)
    img_integrated_gradient_overlay= visualize(attributions, img, clip_above_percentile=95, clip_below_percentile=58, \
                                                morphological_cleanup=True, overlay=True, mask_mode=True, outlines=outlines, threshold=.01)
    img_integrated_gradient= visualize(attributions, img, clip_above_percentile=95, clip_below_percentile=58, morphological_cleanup=True, overlay=False, outlines=outlines, threshold=.01)
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient, \
                                       img_integrated_gradient_overlay)
    
    # overlay mask on image
    ig_mask = img_fill(np.uint8(img_integrated_gradient[:,:,1]), 0)
    ig_mask[ig_mask != 0] = 1
    cam = img[:, :, 1]+np.uint8(ig_mask)
    #if show:
    #    plt.imshow(img_integrated_gradient_overlay)
    if reg:
        return img_gradient_overlay, img_gradient
    print('finished Integrated Gradients explanation')
    #return cam, np.float32(ig_mask)
    return np.float32(ig_mask)