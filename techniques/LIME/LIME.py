import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import cv2

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import metrics

from lime import lime_image
from skimage.segmentation import mark_boundaries
from techniques.utils import get_model, get_imagenet_classes, read_tensor

#model = models.resnet18(pretrained=True)

# resize and take the center part of image to what our model expects
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf

def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_image(path):
    if isinstance(path, str):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 
    else:
        img = Image.fromarray(path)
        return img.convert('RGB') 
        
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    
pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    #batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    batch = torch.stack(tuple(preprocess_transform(i/np.max(i)).float() for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

def generate_lime_explanation(img, model_t, pred_rank=1, positive_only=True, show=True):
    #img = get_image(path)
    #image for display purposes
    global model
    model = model_t.cuda()
    displ_img = img
    # image for generating mask
    #img = Image.fromarray(img.astype('uint8'), 'RGB')
    
    idx2label, cls2label, cls2idx = [], {}, {}
    try:
        with open(os.path.abspath('../data/imagenet_class_index.json'), 'r') as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
            cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}
    except:
        with open(os.path.abspath('./data/imagenet_class_index.json'), 'r') as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
            cls2idx = {class_idx[str(k)][0]: k for k in range(len(class_idx))}

    model.eval()
    img_t = read_tensor(img)
   # if torch.cuda.is_available():
   #     img_t = img_t.cuda()
   # logits = model(img_t)
   # probs = F.softmax(logits, dim=1)
   # probs5 = probs.topk(5)
   # tuple((p,c, idx2label[c]) for p, c in zip(probs5[0][0].detach().cpu().numpy(), probs5[1][0].detach().cpu().numpy()))
   # classes = get_imagenet_classes()
   # print("original lime classification: {0}".format(classes[probs5[1][0].detach().cpu().numpy()[0]]))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance((displ_img/np.max(displ_img).astype(float)),
                                             batch_predict, # classification function
                                             top_labels=pred_rank, 
                                             hide_color=0, 
                                             num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[pred_rank-1], positive_only, num_features=5, hide_rest=False)
    print('lime classsification: {0}'.format(explanation.top_labels[pred_rank-1]))
    # img_boundry1 = mark_boundaries(temp/255.0, mask)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    cv2.imwrite('/work/lisabdunlap/explain-eval/results/different_architectures/cat_dog/lime_resnet18.png', img_boundry1)
    """if show:
        heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8((mask/np.max(mask)) * 255.0), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        cam = heatmap + np.float32(displ_img)
        cam /= np.max(cam)
        plt.imshow(cam)"""
    print('finished lime explanation')
    #return img_boundry1, np.array(mask, dtype=float)
    del img_t
    return np.array(mask, dtype=float)