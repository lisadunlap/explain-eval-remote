import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy as sp
import cv2

from saliency_eval import cc

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

#weight mask evenly so that the usm equals that of the heatmap
def weight_mask(heatmap, mask):
    heatmap_sum = np.sum(heatmap)
    lime_sum = float(np.sum(mask))
    mask = np.array(mask, dtype=float)
    mask[mask==1.0] = float(heatmap_sum/lime_sum)
    return heatmap, mask

def preprocess_groundings(map, sum_value=1, threshold=0, binary=False):
    m=np.array(map, dtype=float)
    if len(np.unique(m)) == 2:
        m = erode(map, threshold)
    else:
        m[m < np.percentile(m, 100-threshold)] = 0
        if binary:
            m[m!=0] = 1.0
            #print(np.sum(m))
    if not binary:
        norm = np.sum(m)/sum_value
        m /= norm
        mu = np.mean(m)
        sigma = np.var(m)
        # add laplace noise
        noise = np.random.laplace(mu, sigma, [224,224]) 
        m += noise
        #norm = float(np.sum(m)/sum_value)
        #m /= norm
    return m

# erode the mask so that only a certain percentage of 
# the pixels are highlighted
def erode(mask, threshold):
    w,h = mask.shape
    #plt.imshow(mask)
    num_pix = (threshold/100)*(w*h)
    #print("threshold {0}".format(num_pix))
    eroded = mask
    i=1
    if np.sum(mask) <= num_pix:
        #print("return {0}".format(np.sum(mask)))
        return np.array(mask, dtype=float)
    while True:
        kernel = np.ones((5,5), np.uint8) 
        eroded = cv2.erode(mask, kernel, iterations=i)
        if np.sum(eroded) <= num_pix:
            #print("return {0}".format(np.sum(eroded)))
            #plt.imshow(eroded)
            return eroded
        else:
            i = i+1
    return np.array(eroded, dtype=float)

# calc iou where it is sum(values of interesction)/sum(values of union)
# if same=True, then we are comparing two maps from the same technique
def calc_iou(heatmap, mask, threshold=0, num_pixels=False):
    #if not same:
    #    heatmap, mask = weight_mask(heatmap, mask)
    if num_pixels:
        m1 = preprocess_groundings(heatmap, threshold=threshold, binary=True)
        m2 = preprocess_groundings(mask, threshold=threshold, binary=True)
    else:
        m1 = preprocess_groundings(heatmap, threshold=threshold)
        m2 = preprocess_groundings(mask, threshold=threshold)
    union = np.add(m1, m2)
    union[union == 2.0] = 1.0
    intersection = np.multiply(m1, m2)
    #print("intersesction {0}".format(np.sum(intersection)))
    #print("union {0}".format(np.sum(union)))
    iou =np.sum(intersection)/np.sum(union)
    return iou, intersection, union

# calc cosine similarity with mask and normalized heatmap
def cos_similarity(heatmap, mask, threshold=0):
    #norm_heatmap = heatmap/np.max(heatmap)
    m1 = preprocess_groundings(heatmap, threshold=threshold)
    m2 = preprocess_groundings(mask, threshold=threshold)
    A = m1
    B = m2

    Aflat = np.hstack(A)
    Bflat = np.hstack(B)

    dist = distance.cosine(Aflat, Bflat)
    return dist

# calculate ymmetric kL-divergence 
#def jsd(map1, map2, base=np.e, threshold=0):
#    '''
#        Implementation of pairwise `jsd` based on  
#        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
#    '''
#    m1 = preprocess_groundings(map1, threshold=threshold)
#    m2 = preprocess_groundings(map2, threshold=threshold)
    #m = 1./2*(m1 + m2)
    #return sp.stats.entropy(m1,m, base=base)/2. +  sp.stats.entropy(m2, m, base=base)/2.
#    return distance.jensenshannon(m1, m2)

def jensenshannon(p, q, base=None, threshold=0):
    """Compute the Jensen-Shannon distance (metric) between
    two 1-D probability arrays. This is the square root
    of the Jensen-Shannon divergence.
    The Jensen-Shannon distance between two probability
    vectors `p` and `q` is defined as,
    .. math::
       \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
    where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
    and :math:`D` is the Kullback-Leibler divergence.
    This routine will normalize `p` and `q` if they don't sum to 1.0.
    Parameters
    ----------
    p : (N,) array_like
        left probability vector
    q : (N,) array_like
        right probability vector
    base : double, optional
        the base of the logarithm used to compute the output
        if not given, then the routine uses the default base of
        scipy.stats.entropy.
    Returns
    -------
    js : double
        The Jensen-Shannon distance between `p` and `q`
    .. versionadded:: 1.2.0
    Examples
    --------
    >>> from scipy.spatial import distance
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0)
    1.0
    >>> distance.jensenshannon([1.0, 0.0], [0.5, 0.5])
    0.46450140402245893
    >>> distance.jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    0.0"""
    p = preprocess_groundings(p, threshold=threshold)
    q = preprocess_groundings(q, threshold=threshold)
    p = p.flatten()
    q = q.flatten()
    m = (p + q) / 2.0
    left = sp.special.rel_entr(p, m)
    right = sp.special.rel_entr(q, m)
    js = np.sum(left, axis=0) + np.sum(right, axis=0)
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)

# total variation dist 
def tvd(map1, map2, threshold):
    # Assumes a, b are numpy arrays
    m1 = preprocess_groundings(map1, threshold=threshold)
    m2 = preprocess_groundings(map2, threshold=threshold)
    m1 = m1.flatten()
    m2 = m2.flatten()
    return sum(abs(m1-m2))/2

def get_stats(map1, map2, threshold=0):
    #iou, intersection, union = calc_iou(map1, map2, threshold=threshold)
    #print("IoU: {0}".format(iou))
    iou_pix, intersection_pix, union_pix = calc_iou(map1, map2, threshold=threshold, num_pixels=True)
    print("pixel count IoU: {0}".format(iou_pix))
    cos_dist = cos_similarity(map1, map2, threshold=threshold)
    print("cos similarity: {0}".format(cos_dist))
    #flatten maps to get Jensen Shannon Distance
    js_dist = jensenshannon(map1, map2, threshold=threshold)
    print("Jenson Shannon dist: {0}".format(js_dist))
    tv_dist = tvd(map1, map2, threshold=threshold)
    print("total variation distance: {0}".format(tv_dist))
    cor_coef = cc(map1, map2)
    print("Pearsons Correlation Coefficient: {0}".format(cor_coef))
    print('------------------------------')
    return iou_pix, cos_dist, js_dist, tv_dist, cor_coef