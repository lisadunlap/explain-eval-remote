### Section 1 - First, let's import everything we will be needing.

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import argparse
import sys
from PIL import ImageFile
import cv2
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
import shutil
import torchsample
from fine_tuning_config_file import *
# custom datasets
from data_utils.miniplaces_dataset import MiniPlacesDataset
from runningAvg import RunningAvg
from tester import compute_output
from accuracy import accuracy
from metrics import compute_metrics
from techniques.generate_grounding import gen_grounding
from techniques.utils import pointing_game, jensenshannon, get_img_mask, get_displ_img

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train', action='store_true', help='train')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--cuda', default=0, type=int, help='cuda device')
parser.add_argument('--epochs', default=150, type=int, help='num epochs')
parser.add_argument('--start', default=0, type=int, help='num epochs')
parser.add_argument('--name', default='testing_new', type=str, help='model name')
parser.add_argument('--loss', default='mse', type=str, help='loss function')

args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda)

OBJ_NAMES = [
    'backpack', 'bird', 'dog', 'elephant', 'kite', 'pizza', 'stop_sign',
    'toilet', 'truck', 'zebra'
]

all_locations = []
with open('/work/lisabdunlap/bam/data/obj/loc_train.txt', 'r') as ff:
    for line in ff:
        all_locations.append(line)

CUDA_VISIBLE_DEVICES = args.cuda
print('Are you using your GPU? {}'.format("Yes!" if args.cuda != 'cpu' else "Nope :("))

### SECTION 2 - data loading and transformation

################################
# load and prepare datatloader #
################################
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #torchsample.transforms.RandomRotate(30),
        torchsample.transforms.RandomGamma(0.5, 1.5),
        torchsample.transforms.RandomSaturation(-0.8, 0.8),
        torchsample.transforms.RandomBrightness(-0.3, 0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/work/lisabdunlap/bam/data/obj/'

dsets = dict()
dsets['train'] = MiniPlacesDataset(
        photos_path=os.path.join(data_dir),
        labels_path=os.path.join(data_dir, 'train.txt'),
        transform=data_transforms['train'],
        train=True,
        location_paths=os.path.join(data_dir, 'loc_train.txt')
    )
dsets['val'] = MiniPlacesDataset(
        photos_path=os.path.join(data_dir),
        labels_path=os.path.join(data_dir, 'val2.txt'),
        transform=data_transforms['val'],
        train=False,
        location_paths=os.path.join(data_dir, 'loc_val.txt')
    )

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=25)
                for x in ['train', 'val']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}


### SECTION 3 : Writing the functions that do training and validation phase.

def train_model(model, criterion, optimizer, lr_scheduler, checkpoint_file, num_epochs=100):
    since = time.time()
    print("##" * 10)
    best_model = model

    # Loss history is saved below. Saved every epoch.
    loss_history = {'train': [0], 'val': [0]}
    start_epoch = 0
    best_top1 = 0
    best_top5 = 0

    if checkpoint_file:
        if os.path.isfile(checkpoint_file):
            try:
                checkpoint = torch.load(checkpoint_file)
                start_epoch = checkpoint['epoch']
                best_top1 = checkpoint['best_top1']
                best_top5 = checkpoint['best_top5']
                loss_history = checkpoint['loss_history']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_file, checkpoint['epoch']))
            except:
                print("Found the file, but couldn't load it.")
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file))

    # params for gradient noise have been commented out, as they were not
    # used the final model.
    # gamma = .55

    for epoch in range(start_epoch, num_epochs):
        t = epoch - 30
        # sigma = BASE_LR/((1+t)**gamma)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        start_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                mode = 'train'
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()
                mode = 'val'

            losses = RunningAvg()
            epoch_acc_1 = RunningAvg()
            epoch_acc_5 = RunningAvg()

            counter = 0
            # Iterate over data.
            for data in dset_loaders[phase]:
                inputs, labels, paths, locations = data
                mask_target = torch.Tensor([[0.0,1.0] for i in range(len(inputs))])

                # wrap them in Variable
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds5 = torch.topk(outputs.data, 5)
                _, preds1 = torch.max(outputs.data, 1)
                expl_masks = []
                mask_results = []
                j=0
                for img, label, loc in zip(inputs, labels, locations):
                    label = label.cpu().numpy()
                    label_name = OBJ_NAMES[label]
                    #img = img.cpu().numpy().transpose((1, 2, 0))
                    #mean = np.array([0.485, 0.456, 0.406])
                    #std = np.array([0.229, 0.224, 0.225])
                    #displ_img = std * img + mean
                    #displ_img = np.clip(displ_img, 0, 1)
                    #displ_img /= np.max(displ_img)
                    #displ_img = np.uint8(displ_img * 255)
                    displ_img = get_displ_img(img)
                    gcam_expl = gen_grounding(displ_img, 'gcam', label_name, show=False, save=False, index=1, has_model=model)
                    expl_masks += [gcam_expl]
                    #cv2.imwrite('/work/lisabdunlap/explain-eval/training/examples/training-mask-data-{0}.jpg'.format(j), gcam_expl)
                    #cv2.imwrite('/work/lisabdunlap/explain-eval/training/examples/training-orig-data-{0}.jpg'.format(j), displ_img)
                    mask_results += [pointing_game(gcam_expl, loc.numpy()[0])]
                    j += 1
                
                if args.loss == 'mse':
                    mask_loss = nn.MSELoss().cuda()
                else:
                    mask_loss = nn.BCELoss().cuda()
                print('mask loss: ', mask_loss(torch.Tensor(mask_results), mask_target))

                loss = criterion(outputs, labels) + mask_loss(torch.Tensor(mask_results), mask_target)
                print("loss: ", loss)
                # Just so that you can keep track that something's happening and don't feel like the program isn't running.

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # add noise to the gradient
                    # for p in model_ft.parameters():
                    #    # add noise
                    #    p.grad = p.grad + np.random.normal(0, sigma**2)
                    optimizer.step()
                # try:
                losses.update(loss.data, inputs.size(0))
                acc_top_1, acc_top_5 = accuracy(outputs.data, labels.data)
                epoch_acc_1.update(acc_top_1[0], inputs.size(0))
                epoch_acc_5.update(acc_top_5[0], inputs.size(0))

                if counter % 100 == 0:
                    print("It: {}, Loss: {:.4f}, Top 1: {:.4f}, Top 5: {:.4f}".format(counter, losses.avg,
                                                                                      epoch_acc_1.avg, epoch_acc_5.avg))

                counter += 1
            # At the end of every epoch, tally up losses and accuracies
            time_elapsed = time.time() - since

            print_stats(epoch_num=epoch, train=mode, batch_time=time_elapsed, loss=losses, top1=epoch_acc_1,
                        top5=epoch_acc_5)

            loss_history[mode].append(losses.avg)
            is_best = epoch_acc_5.avg > best_top5
            best_top5 = max(epoch_acc_5.avg, best_top5)
            # save checkpoint at the end of every epoch
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_top1': epoch_acc_1.avg,
                'best_top5': best_top5,
                'loss_history': loss_history,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            print('checkpoint saved!')

            # deep copy the model
            if phase == 'val':
                if USE_TENSORBOARD:
                    foo.add_scalar_value('epoch_loss', losses.avg, step=epoch)
                    foo.add_scalar_value('epoch_acc_1', epoch_acc_1, step=epoch)
                if epoch_acc_1.avg > best_top1:
                    best_top1 = epoch_acc_1.avg
                    best_model = copy.deepcopy(model)
                    print('new best accuracy = ', best_top1)
            print("--- %s minutes ---" % ((time.time() - start_time)/60))

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_top1))
    print('returning and looping back')
    return best_model


# This function changes the learning rate over the training model.
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


## Helper Functions

def save_checkpoint(state, is_best, filename='../' + args.name + '_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../' + args.name + '_model_best.pth.tar')


def print_stats(epoch_num=None, it_num=None, train=True, batch_time=None, loss=None, top1=None, top5=None):
    progress_string = "Epoch %d" % epoch_num if epoch_num else ''
    if it_num is not None:
        progress_string += ", Iteration %d" % it_num
    else:
        progress_string += " finished"
    progress_string += ", Training set = %s\n" % (train)
    print(progress_string +
          # "\tBatch time: {batch_time.val:.3f}, Batch time average: {batch_time.val:.3f}\n"
          "\tLoss: {loss.avg:.4f}\n Accuracies: \n"
          "\tTop 1: {top1.avg:.3f}%\n"
          "\tTop 5: {top5.avg:.3f}%\n".format(batch_time=batch_time, loss=loss, top1=top1, top5=top5))


def save(filename='pretrained resnet18' + args.name):
    """Saves model using file numbers to make sure previous models are not overwritten"""
    filenum = 0
    while (os.path.exists(os.path.abspath('{}_v{}.pt'.format(filename, filenum)))):
        filenum += 1
    torch.save(model.module.state_dict(), '{}_v{}.pt'.format(filename, filenum))


### SECTION 4 : Define model architecture

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

criterion = nn.CrossEntropyLoss()

criterion.cuda()
model_ft.cuda()

# model_ft = nn.DataParallel(model_ft, device_ids= [4, 5, 6, 7])

optimizer_ft = optim.RMSprop(model_ft.parameters(), lr=BASE_LR)

#if len(sys.argv) < 1:
#    print(
#        "Type 'tr' to train, 'test' to test, and 'metrics' to extract the error metrics'.  For 'test' and 'metrics' make sure to add another argument specifying the path of the model.")

if args.train:
    checkpoint_file = '../' + args.name + '_checkpoint.pth.tar'
    print("created chechpoint file")
    # Run the functions and save the best model in the function model_ft.
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, checkpoint_file,
                               num_epochs=int(args.epochs))

    # Save mode
    torch.save(model_ft.state_dict(), args.name + '-fine_tuned_best_model.pt')
    # model_ft.save_state_dict(sys.argv[1]+'-fine_tuned_best_model.pt')

if not args.train:
    print("from file")
    print(torch.cuda.current_device())
    model_path = sys.argv[2]
    output_file_name = sys.argv[3] if len(sys.argv) > 3 else 'output.txt'

    test_options = {
        'photos_path': os.path.expanduser(TEST_DATA_PATH),
        'transform': data_transforms['val']
    }
    if sys.argv[1] == 'test':
        compute_output(model_path, output_file_name, model_ft, True, test_options)
    #elif sys.argv[1] == 'metrics':
    #    test_options['photos_path'] = os.path.expanduser('~/data/images/')
    #    compute_output(model_path, 'train_' + output_file_name, model_ft, use_gpu, test_options, compute_metrics)
    #    compute_output(model_path, 'val_' + output_file_name, model_ft, use_gpu, test_options, compute_metrics)

