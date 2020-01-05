import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets, models
from PIL import Image
from tqdm import tqdm
import os
from RISE.explanations import RISE
from techniques.utils import read_tensor, get_model


# Dummy class to store arguments
class Dummy():
    pass


# Function that opens image from disk, normalizes it and converts to tensor
"""read_tensor = transforms.Compose([
    lambda x: Image.fromarray(x.astype('uint8'), 'RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])"""


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    try:
        labels = np.loadtxt('../data/synset_words.txt', str, delimiter='\t')
    except:
        labels = np.loadtxt('./data/synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])


# Image preprocessing function
preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
    
def explain_instance(model, explainer, img, top_k=1, show=False):
    if torch.cuda.is_available():
        img = img.cuda()
    saliency = explainer(img).cpu().numpy()
    p, c = torch.topk(model(img), k=top_k)
    p, c = p[0], c[0]
    
    if show:
        plt.figure(figsize=(10, 5*top_k))
    for k in range(top_k):
        """if show:
            plt.subplot(top_k, 2, 2*k+1)
            plt.axis('off')
            plt.title('rise classification {:.2f}% {}'.format(100*p[k], get_class_name(c[k])))
            tensor_imshow(img[0])

            plt.subplot(top_k, 2, 2*k+2)
            plt.axis('off')
            plt.title(get_class_name(c[k]))
            tensor_imshow(img[0])"""
        sal = saliency[c[k]]
        """if show:
            plt.imshow(sal, cmap='jet', alpha=0.5)
            plt.colorbar(fraction=0.046, pad=0.04)
    
    if show:
        plt.show()"""
    return sal
    
def gen_rise_grounding(img, model, cuda=False, show=True, index=1):
    # Load black box model for explanations
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval()
    torch.cuda.set_device(2)
    print ('Current cuda device ', torch.cuda.current_device())
    if cuda:
        model=model.cuda()
       # model = torch.nn.DataParallel(model, device_ids=[5, 2])

    for p in model.parameters():
        p.requires_grad = True

    #create explainer
    explainer = RISE(model, (224, 224), 50)
    
    # Generate masks for RISE or use the saved ones.
    maskspath = 'masks.npy'
    generate_new = False

    if generate_new or not os.path.isfile(maskspath):
        explainer.generate_masks(N=6000, s=8, p1=0.1, savepath=maskspath)
        print("Masks are generated.")
    else:
        explainer.load_masks(maskspath)
        print('Masks are loaded.')
    
    #explain instance
    sal = explain_instance(model, explainer, read_tensor(img), index, show=show)
    print("finished RISE")
    return sal


def explain_all_batch(data_loader, explainer):
    n_batch = len(data_loader)
    b_size = data_loader.batch_size
    total = n_batch * b_size
    # Get all predicted labels first
    target = np.empty(total, 'int64')
    for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
        p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
        target[i * b_size:(i + 1) * b_size] = c
    image_size = imgs.shape[-2:]

    # Get saliency maps for all images in val loader
    explanations = np.empty((total, *image_size))
    for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
        saliency_maps = explainer(imgs.cuda())
        explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
            range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
    return explanations