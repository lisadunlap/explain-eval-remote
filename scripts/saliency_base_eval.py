import cv2
from techniques.saliency import gen_grounding
from techniques.saliency_eval import * 

import csv
from datetime import datetime
import argparse
import sys

def write_to_csv(filename, row):
    with open(filename, 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                
paths = ['data/ILSVRC2012_img_val/ILSVRC2012_val_00045585.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00023715.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00018985.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00001937.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00011713.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00017107.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00000615.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00013741.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00022759.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00032660.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00028416.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00028247.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00016273.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00044429.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00036091.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00018439.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00033769.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00031754.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00030846.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00006427.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00020737.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00037884.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00035686.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00008593.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00019285.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00000480.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00013241.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00042375.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00005774.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00020740.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00030558.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00031739.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00007584.JPEG',
       'data/ILSVRC2012_img_val/ILSVRC2012_val_00023493.JPEG']
    
    
if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--sample-size",
                        default=100,
                        type=int,
                        help="Number of images to test.")
    parser.add_argument("--thresholds",
                        default=1,
                        type=list,
                        help="Thresholds to test.")
    parser.add_argument("--result-path",
                        default="results/",
                        type=list,
                        nargs='+',
                        help="Location of results file")
    parser.add_argument("--techniques",
                        default=['gcam', 'lime'],
                        type=list,
                        nargs='+',
                        help="Techniques to test")
    parser.add_argument("--models",
                        default=['resnet18', 'vgg19'],
                        type=list,
                        nargs='+',
                        help="Models to test")
    args = parser.parse_args(sys.argv[1:])
    
    filename = 'results/saliency-eval-%s.csv'%datetime.now().strftime('%Y-%m-%d-%H-%M')
    write_to_csv(filename, ['path', 'method', 'model', 'correlation coefficient', 'kale divergence'])
    
    data_iter = iter(paths)
    print("Number of test images: {0}".format(len(paths)))

    for i in range(args.sample_size):
        path = next(data_iter)
        val_img = cv2.imread(path)
        val_img = cv2.resize(val_img, (224, 224))
        
        masks_vgg19 = {}
        masks_resnet18 = {}
        techniques = ['backprop', 'guided backprop', 'vanilla gradient', 'integrated gradients']
        
        # generate resnet18 explanations (not including eb)
        bp_expl, bp_mask = gen_grounding(val_img, 'resnet18', 'guided-backprop', show=False, reg=True)
        bp_mask[bp_mask < 0] = 0
        masks_resnet18['backprop'] = bp_mask
        
        gbp_expl, gbp_mask = gen_grounding(val_img, 'resnet18', 'guided-backprop', show=False)
        gbp_mask[gbp_mask < 0] = 0
        masks_resnet18['guided backprop'] = gbp_mask
        
        g_expl, g_mask = gen_grounding(val_img, 'resnet18', 'ig', show=False, reg=True)
        masks_resnet18['vanilla gradient'] = g_mask
        
        ig_expl, ig_mask = gen_grounding(val_img, 'resnet18', 'ig', show=False)
        masks_resnet18['integrated gradients'] = ig_mask
        print('finished resnet18 expl')
        
        
        # generate vgg19 explanations
        bp_expl, bp_mask = gen_grounding(val_img, 'vgg19', 'guided-backprop', show=False, reg=True)
        bp_mask[bp_mask < 0] = 0
        masks_vgg19['backprop'] = bp_mask
        
        gbp_expl, gbp_mask = gen_grounding(val_img, 'vgg19', 'guided-backprop', show=False)
        gbp_mask[gbp_mask < 0] = 0
        masks_vgg19['guided backprop'] = gbp_mask
        
        g_expl, g_mask = gen_grounding(val_img, 'vgg19', 'ig', show=False, reg=True)
        masks_vgg19['vanilla gradient'] = g_mask
        
        ig_expl, ig_mask = gen_grounding(val_img, 'vgg19', 'ig', show=False)
        masks_vgg19['integrated gradients'] = ig_mask
        
        #eb_mask = gen_eb(path, 'vgg19', show=True)
        #masks_vgg19['excitation backprop'] = eb_mask
        print('finished vgg19 expl')
        
        # comparing same tech diff architecture
        for t in techniques:
            if t != 'excitation backprop':
                cor_coef = cc(masks_vgg19[t], masks_resnet18[t])
                kale = kldiv(masks_vgg19[t], masks_resnet18[t])
                write_to_csv(filename, [path, t, None, cor_coef, kale])
        
        # comparing diff tech same arch (vgg19)
        for i in range(len(techniques)):
            for j in range(i, len(techniques)):
                if i!=j:
                    cor_coef = cc(masks_vgg19[techniques[i]], masks_vgg19[techniques[j]])
                    kale = kldiv(masks_vgg19[techniques[i]], masks_vgg19[techniques[j]])
                    write_to_csv(filename, [path, techniques[i]+'-'+techniques[j], 'vgg19', cor_coef, kale])
        