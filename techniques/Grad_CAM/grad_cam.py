import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
from collections import OrderedDict
import numpy as np
import argparse
import os
import torch.nn as nn

from techniques.utils import read_tensor, get_model_info
from data_utils.data_setup import get_imagenet_classes
import copy

resnet = models.resnet50()
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet.fc = nn.DataParallel(resnet)
model_dict = torch.load('/work/lisabdunlap/explain-eval/training/saved/bam_resnet5_diff_lr_continue_model_best.pth.tar', map_location='cuda:0')
resnet.load_state_dict(model_dict['state_dict'])
resnet=resnet.module
i=0

OBJ_NAMES = [
    'backpack', 'bird', 'dog', 'elephant', 'kite', 'pizza', 'stop_sign',
    'toilet', 'truck', 'zebra'
]

cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers,use_cuda):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)
        self.cuda = use_cuda
    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        if self.cuda:
            output = output.cpu()
            output = resnet.fc(output).cuda()
        else:
            output = resnet.fc(output)
        return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img
	input.requires_grad = True
	return input

"""def show_cam_on_image(img, mask,name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam/cam_{}.jpg".format(name), np.uint8(255 * cam))"""

"""def preprocess_image(img):
    normalized_tensor = Variable(read_tensor(img), requires_grad=True)
    return normalized_tensor"""


def show_cam_on_image(img, mask):
    # h, w, _ = img.shape
    # result = cv2.resize(mask, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    plt.imshow(cam)
    #cv2.imwrite("/work/lisabdunlap/explain-eval/techniques/Grad_CAM//cam.jpg", np.uint8(255 * cam))
    
class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        print('index: ', cifar_classes[index])
        classes = get_imagenet_classes()
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot.requires_grad = True
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        #print('grads_val',grads_val.shape)
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        #print('weights',weights.shape)
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        #print('cam',cam.shape)
        #print('features',features[-1].shape)
        #print('target',target.shape)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model#这里同理，要的是一个完整的网络，不然最后维度会不匹配。
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()
		for module in self.model.named_modules():
			module[1].register_backward_hook(self.bp_relu)

	def bp_relu(self, module, grad_in, grad_out):
		if isinstance(module, nn.ReLU):
			return (torch.clamp(grad_in[0], min=0.0),)
	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output = self.forward(input.cuda())
		else:
			output = self.forward(input)
		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		#print(input.grad)
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		#self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)
		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./examples/',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    print('start')

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = models.resnet18(pretrained=True)#这里相对vgg19而言我们处理的不一样，这里需要删除fc层，因为后面model用到的时候会用不到fc层，只查到fc层之前的所有层数。
    del model.fc
    #print(model)
    #modules = list(resnet.children())[:-1]
    #model = torch.nn.Sequential(*modules)

    #print(model)
    grad_cam = GradCam(model , \
                    target_layer_names = ["layer4"], use_cuda=args.use_cuda)
    print('does it get past gradcam')
    img = cv2.imread('/work/lisabdunlap/explain-eval/data/samples/bcats.jpg')
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)
    input.required_grad = True
    print('input.size()=',input.size())
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index =None

    mask = grad_cam(input, target_index)
    print('created mask')
    i=i+1
    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model = models.resnet50(pretrained=True), use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    if not os.path.exists('gb'):
        os.mkdir('gb')
    if not os.path.exists('camgb'):
        os.mkdir('camgb')
    print('saving image...')
    utils.save_image(torch.from_numpy(gb), '/work/lisabdunlap/explain-eval/techniques/Grad_CAM/gb_{}.jpg')
    cam_mask = np.zeros(gb.shape)
    for j in range(0, gb.shape[0]):
        cam_mask[j, :, :] = mask
    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), '/work/lisabdunlap/explain-eval/techniques/Grad_CAM/cam_gb_{}.jpg')


def gen_gcam(img, model_name, target_index=None, neg=False, weight=False, show=True, has_model=None, classes=None):
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    if has_model == None:
        model, classes, target_layer = get_model_info(model_name)
        resnet = copy.deepcopy(model)
        del model.fc
    else:
        #print(has_model.fc)
        model = copy.deepcopy(has_model)
        resnet = copy.deepcopy(has_model)
        classes = classes
        target_layer = 'layer4'
        del model.fc
    grad_cam = GradCam(model=model, target_layer_names=[target_layer], use_cuda=True)
        
    #img = np.float32(cv2.resize(img, (224, 224))) / 255
    img = np.float32(img)
    input = preprocess_image(img)
    input.required_grad = True
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    #target_index = None

    mask = grad_cam(input, target_index)

    h, w, _ = img.shape
    mask = cv2.resize(mask, (w, h))

    show_cam_on_image(img, mask)
    return mask

def gen_gcam_model(img, model_name, target_index=None, neg=False, weight=False, show=True, res18=False):
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    if has_model == None:
        if res18:
            resnet = models.resnet18(pretrained=True)
        model, classes, target_layer = get_model_info(model_name)
        resnet = copy.deepcopy(model)
        del model.fc
    else:
        #print(has_model.fc)
        model = copy.deepcopy(has_model)
        resnet = copy.deepcopy(has_model)
        classes = OBJ_NAMES
        target_layer = 'layer4'
        
        #del model.fc
    grad_cam = GradCam(model=model, target_layer_names=[target_layer], use_cuda=True)
        
    #img = np.float32(cv2.resize(img, (224, 224))) / 255
    img = np.float32(img)
    input = preprocess_image(img)
    input.required_grad = True
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    #target_index = None

    mask = grad_cam(input, target_index)

    h, w, _ = img.shape
    mask = cv2.resize(mask, (w, h))

    show_cam_on_image(img, mask)
    return mask

def gen_gb(img, model_name, target_index=None, neg=False, weight=False, show=True, has_model=None):

    if has_model == None:
        print('---------GET MODEL FROM NAME----------')
        model, classes, target_layer = get_model_info(model_name)
        resnet = copy.deepcopy(model)
        model.fc
    else:
        print('---------MODEL IS SUPPLIED----------')
        #print(has_model.fc)
        model = copy.deepcopy(has_model)
        resnet = copy.deepcopy(has_model)
        classes = OBJ_NAMES
        target_layer = 'layer4.1'
        model.fc
    gb_model = GuidedBackpropReLUModel(model = model, use_cuda=True)
    gb = gb_model(input, index=target_index)
    if not os.path.exists('gb'):
        os.mkdir('gb')
    if not os.path.exists('camgb'):
        os.mkdir('camgb')
    print('saving image...')
    utils.save_image(torch.from_numpy(gb), '/work/lisabdunlap/explain-eval/techniques/Grad_CAM/gb_{}.jpg')
    cam_mask = np.zeros(gb.shape)
    for j in range(0, gb.shape[0]):
        cam_mask[j, :, :] = mask
    cam_gb = np.multiply(cam_mask, gb)
    return cam_mask