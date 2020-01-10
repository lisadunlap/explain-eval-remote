from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms

from Grad_CAM.gcam3 import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda, device):
    cuda = cuda and torch.cuda.is_available()
    cuda_dev = "cuda:"+str(device)
    device = torch.device(cuda_dev if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        #print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable():
    classes = []
    with open("/work/lisabdunlap/explain-eval/data/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(raw_image):
    #raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    #cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    return gcam


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)
    
def gen_gcam(imgs, model, target_layer='layer4', target_index=1, classes=get_classtable(), cuda=True, device=0, single=True, prep=True, show_labels=False):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda, device)

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    for i, im in enumerate(imgs):
        if prep:
            image, raw_image = preprocess(im)
        else:
            image = im
            raw_image = im.cpu().numpy().transpose((1,2,0))
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)
    # =========================================================================
    ''''print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    for i in range(topk):
        # In this example, we specify the high confidence classes
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()'''

    # =========================================================================
    '''print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()'''

    # =========================================================================
    #print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    #gbp = GuidedBackPropagation(model=model)
    #_ = gbp.forward(images)

    for i in range(target_index):
        # Guided Backpropagation
        #gbp.backward(ids=ids[:, [i]])
        #gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)
        masks = []
        for j in range(len(images)):
            if show_labels:
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            '''save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )'''

            # Grad-CAM
            mask = save_gradcam(
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )
            masks += [mask]

            # Guided Grad-CAM
            '''save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )'''
    if single:
        return masks[0]
    return masks
    
def gen_gcam_single(img, model, target_layer='layer4', target_index=1, classes=get_classtable(),cuda=True, device=0):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda, device)

    # Model from torchvision
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, im in enumerate(img):
        #print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(im)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)
    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)
    # =========================================================================
    print("Vanilla Backpropagation:")

    '''bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)

    for i in range(topk):
        # In this example, we specify the high confidence classes
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()'''

    # =========================================================================
    '''print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()'''

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    #gbp = GuidedBackPropagation(model=model)
    #_ = gbp.forward(images)

    for i in range(target_index):
        # Guided Backpropagation
        #gbp.backward(ids=ids[:, [i]])
        #gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            '''save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )'''

            # Grad-CAM
            gcam = save_gradcam(
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            '''save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )'''
    return gcam