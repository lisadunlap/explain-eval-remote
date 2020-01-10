from collections import OrderedDict
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import re
from torch.autograd import Variable
import tensorflow as tf
import matplotlib.cm as cm
import os

from techniques.utils import get_imagenet_classes, get_model_info, get_displ_img

def get_device(cuda, device):
    cuda = cuda and torch.cuda.is_available()
    cuda_dev = "cuda:"+str(device)
    print('cuda dev ', cuda_dev)
    device = torch.device(cuda_dev if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        #print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=False)
    #if torch.cuda.is_available():
    #    im_as_var = im_as_var.cuda()
    return im_as_var, cv2im

class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot.cuda(), retain_graph=False)
        
class BackPropagation(_PropagationBase):
    def generate(self):
        output = self.image.grad.detach().cpu().numpy()
        return output.transpose(0, 2, 3, 1)[0]


class GuidedBackPropagation(BackPropagation):
    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0), )

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)

class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        #gcam = torch.clamp(gcam, min=0.)
        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam.detach().cpu().numpy()

    def get_grads(self, target_layer):
        return self._find(self.all_grads, target_layer)
    
class NegGradCAM(_PropagationBase):
    def __init__(self, model):
        super(NegGradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = (-1*fmaps[0] * weights[0]).sum(dim=0)
        #gcam = torch.clamp(gcam, min=0.)
        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam.detach().cpu().numpy()

    def get_grads(self, target_layer):
        return self._find(self.all_grads, target_layer)
    
def get_model_and_class(model_name, other_layer = None):
    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        'resnet18': {
            'target_layer': 'layer4.1',
            'input_size': 224
        },
    }.get(model_name)
    classes = get_imagenet_classes()
    model = models.__dict__[model_name](pretrained=True)
    if other_layer:
        target_layer = other_layer
    else:
        target_layer = CONFIG['target_layer']
    return model, classes, target_layer

def old_gen_gcam(img, model, target_index=1, target_layer='layer4', weight = False, show=True, classes=get_imagenet_classes(), from_saved=False):
    """given a model name (must be in the CONFIG dict),
    and image, and the top number of predictions you want
    to get, returns the gradcam object
    """
    prob = 1
    if from_saved:
        model, classes, target_layer = get_model_info(model)
    else:
        model = model
        classes = classes
        target_layer=target_layer
    model.eval()
    #if isinstance(image_path, str):
    #    raw_img = cv2.imread(image_path, 1)
    #image = preprocess_image(img)
    image = preprocess_image(img)
    image = image.cuda()
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image)
    if target_index == None:
        target_index=1
    for i in range(0, target_index):
        gcam.backward(idx=idx[i])
        region = gcam.generate(target_layer=target_layer) 
        prob = probs[i].data.cpu().numpy()
        if weight:
            prob = probs[i].data.cpu().numpy()
            region = region *prob
        if show:
            print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))
            print(idx[i])
        #if isinstance(image_path, str):
        #    img = cv2.imread(image_path)
        #else:
        #    img = image_path
        h, w, _ = img.shape
        result = cv2.resize(region, (w, h))
        #result = cv2.resize(region, (32, 32))
    return result


def old_gen_gcam_list(img, gcam, target_index=1, target_layer='layer4', weight = False, show=True, classes=get_imagenet_classes(), from_saved=False):
    """given a model name (must be in the CONFIG dict),
    and image, and the top number of predictions you want
    to get, returns the gradcam object
    """
    masks = []
    for image in img:
        image = preprocess_image(img)
        image = image.cuda()
        probs, idx = gcam.forward(image)
        if target_index == None:
            target_index=1
        for i in range(0, target_index):
            gcam.backward(idx=idx[i])
            region = gcam.generate(target_layer=target_layer) 
            prob = probs[i].data.cpu().numpy()
            if weight:
                prob = probs[i].data.cpu().numpy()
                region = region *prob
            if show:
                print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))
                print(idx[i])
            #if isinstance(image_path, str):
            #    img = cv2.imread(image_path)
            #else:
            #    img = image_path
            h, w, _ = img.shape
            result = cv2.resize(region, (w, h))
            masks += []
    return result

def get_guidedBackProp_img(image_path, model_name, categories, other_layer=None):
    """given a model name (must be in the CONFIG dict),
    and image, and the top number of predictions you want
    to get, returns the gradcam object
    """
    model, classes, target_layer = get_model_and_class(model_name, other_layer)
    model.eval()
    image = preprocess_image(img)
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image)
    gbp = GuidedBackPropagation(model=model)
    probs, idx = gbp.forward(image)

    for i in range(0, categories):
        gcam.backward(idx=idx[i])
        region = gcam.generate(target_layer=target_layer)

        gbp.backward(idx=idx[i])
        feature = gbp.generate()

        h, w, _ = feature.shape
        region = cv2.resize(region, (w, h))[..., np.newaxis]
        output = feature * region

        plt.imshow(feature)
        return feature
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))
        
def gen_gcam_single(img, model, target_index=1, target_layer='layer4', weight = False, show=True, classes=get_imagenet_classes(), from_saved=False, save=False, save_path='/work/lisabdunlap/explain-eval/results/gcams', model_name = 'custom_model'):
    """given a model name (must be in the CONFIG dict),
    and image, and the top number of predictions you want
    to get, returns the gradcam object
    """
    prob = 1
    if isinstance(model, str):
        model_name = model
        model, classes, target_layer = get_model_info(model)
    else:
        model = model
        classes = classes
        target_layer=target_layer
    model.eval()
    image = preprocess_image(img)
    image = image.cuda()
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image)
    if target_index == None:
        target_index=1
    for i in range(0, target_index):
        gcam.backward(idx=idx[i])
        region = gcam.generate(target_layer=target_layer) 
        prob = probs[i].data.cpu().numpy()
        if weight:
            prob = probs[i].data.cpu().numpy()
            region = region *prob
        if show:
            print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))
            print(idx[i])
        h, w, _ = img.shape
        result = cv2.resize(region, (w, h))
        '''heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((result / np.max(result)) * 255.0), cv2.COLORMAP_JET),
                                 cv2.COLOR_RGB2BGR)
        alpha = .6
        img = np.uint8(img*255)
        cam = heatmap*alpha + np.float32(img)*(1-alpha)
        cam /= np.max(cam)'''
        cmap = cm.jet_r(result)[..., :3] * 255.0
        cam = (cmap.astype(np.float)*(.5) + img.astype(np.float)*(.5))
        if save:
            filename = "{0}/gcam_{1}.jpg".format(save_path, model_name)
            print('saving to ', filename)
            #print(cv2.imwrite(filename, np.uint8(cam*255)))
            cv2.imwrite(filename, np.uint8(cam))
        if show:
            heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((result / np.max(result)) * 255.0), cv2.COLORMAP_JET),
                                 cv2.COLOR_RGB2BGR)
            alpha = .6
            img = np.uint8(img*255)
            cam = heatmap*alpha + np.float32(img)*(1-alpha)
            cam /= np.max(cam)
            plt.imshow(cam)
    return result