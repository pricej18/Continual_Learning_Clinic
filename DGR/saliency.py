#################################################################
import argparse
import os
import shutil
import time
import random
import pickle
import torch
import pdb
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import mkdir_p, savefig
import numpy as np
import copy

import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import gradcheck
import sys
import collections

#from basic_net import *
#from learner_task_itaml import Learner
#import incremental_dataloader as data

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt



#################################################################
####### SALIENCY FUNCTIONS


def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)): 
      for j in class_name:
        if dataset.targets[i] == j:
            indices.append(i)
    return indices
    

### Add a args.experiment to the function call. You can name it whatever you want
def load_saliency_data(desired_classes, imgs_per_class, args):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    if not os.path.isdir(f'SaliencyMaps/{args.experiment}'):
        mkdir_p(f'SaliencyMaps/{args.experiment}')
    
    saliencySet = torch.utils.data.Dataset()
    
    if args.experiment == "splitMNIST":
    ### This line should be changed to - if args.experiment == "splitMNIST":
        saliencySet = datasets.MNIST(root='SaliencyMaps/Datasets/mnist/', train=False,
                  download=True, transform=transform)

    idx = get_indices(saliencySet,desired_classes)

    subset = Subset(saliencySet, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=args.batch)

    dataiter = iter(saliencyLoader)
    images, labels = next(dataiter)

    salIdx = []
    salLabels = []
    for i in range(len(desired_classes)):
      num=0
      while len(salIdx) < imgs_per_class*(i+1):
        if labels[num]==desired_classes[i]:
          salIdx.append(num)
          salLabels.append(desired_classes[i])
        num += 1
    salImgs = images[salIdx]
    
    return salImgs, torch.tensor(salLabels), saliencySet.classes
    
    
### Add a args.experiment to the function call. You can name it whatever you want
def create_saliency_map(model, ses, desired_classes, imgs_per_class, args):

    sal_imgs, sal_labels, classes = load_saliency_data(desired_classes, imgs_per_class, args)
    sal_imgs, sal_labels = sal_imgs.cuda(), sal_labels.cuda()

    with torch.no_grad():
        scores = model.classify(sal_imgs)
        _, pred = torch.max(scores, 1)
    predicted = pred.squeeze()


    saliency = Saliency(model)

    fig, ax = plt.subplots(2,2*imgs_per_class,figsize=(15,5))
    for ind in range(2*imgs_per_class):
        input = sal_imgs[ind].unsqueeze(0)
        input.requires_grad = True

        grads = saliency.attribute(input, target=sal_labels[ind].item(), abs=False)
        squeeze_grads = grads.squeeze().cpu().detach()
        squeeze_grads = torch.unsqueeze(squeeze_grads,0).numpy()
        grads = np.transpose(squeeze_grads, (1, 2, 0))

        print('Truth:', classes[sal_labels[ind]])
        print('Predicted:', classes[predicted[ind]])


        #original_image = np.transpose((sal_imgs[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
        original_image = np.transpose((sal_imgs[ind].cpu().detach().numpy()), (1, 2, 0))
        
        
        methods=["original_image","blended_heat_map"]
        signs=["all","absolute_value"]
        titles=["Original Image","Saliency Map"]
        colorbars=[False,True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]: cmap = "Reds" 
        else: cmap = "Blues"


        if ind > imgs_per_class-1:
            row = 1
            ind = ind - imgs_per_class
        else: row = 0

        for i in range(2):
            plt_fig_axis = (fig,ax[row][2*ind+i])
            if i==1:
                _ = viz.visualize_image_attr(grads, original_image,
                                            method=methods[i],
                                            sign=signs[i],
                                            fig_size=(4,4),
                                            plt_fig_axis=plt_fig_axis,
                                            cmap=cmap,
                                            show_colorbar=colorbars[i],
                                            title=titles[i])
            else:
                ax[row][2*ind+i].imshow(original_image, cmap='gray')
                ax[row][2*ind+i].set_title('Original Image')
                ax[row][2*ind+i].tick_params(left = False, right = False , labelleft = False , 
                                labelbottom = False, bottom = False) 
    
    fig.tight_layout()
    fig.savefig(f"SaliencyMaps/{args.experiment}/Sess{ses}SalMap.png")
    fig.show()
