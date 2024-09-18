'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function
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
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import gradcheck
import sys
import random
import collections

from basic_net import *
from learner_task_itaml import Learner
import incremental_dataloader as data

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

class args:

    checkpoint = "results/cifar100/meta_svhn_T5_42"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/SVHN/"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "svhn"
    optimizer = "radam"
    
    epochs = 20
    lr = 0.01
    train_batch = 256
    test_batch = 256
    workers = 16
    sess = 0
    schedule = [5,10,15]
    gamma = 0.2
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 1.0
    r = 1

    
    
state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
print(state)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

use_cuda = torch.cuda.is_available()
seed = random.randint(1, 10000)
seed = 2481
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

def createSaliencyMap(model, saliency_loader, pred, ses):
    ##### Create Saliency Maps
    data_iter = iter(saliency_loader)
    sal_imgs, sal_labels = next(data_iter)
    print(sal_labels)
    

    num = -1
    selected_labels = []
    for img in sal_imgs:
        num = num + 1
        if len(selected_labels) > 1: break
        if selected_labels and sal_labels[num] in torch.index_select(sal_labels,0,torch.tensor(selected_labels)):            continue
        else: selected_labels.append(num)
    print(selected_labels)
    selected_imgs = torch.index_select(sal_imgs, 0, torch.tensor(selected_labels))
    sal_imgs, sal_labels, selected_imgs = sal_imgs.cuda(), sal_labels.cuda(), selected_imgs.cuda()
    predicted = pred.squeeze()
    
    
    model.set_saliency(True)
    saliency = Saliency(model)
    
    fig, ax = plt.subplots(1,4,figsize=(10,4))
    for ind in range(0,2):
        input = selected_imgs[ind].unsqueeze(0)
        input.requires_grad = True

        grads = saliency.attribute(input, target=sal_labels[selected_labels[ind]].item(), abs=False)
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        #grads = grads.squeeze().cpu().detach().numpy()

        print('Truth:', classes[sal_labels[selected_labels[ind]]])
        print('Predicted:', classes[predicted[selected_labels[ind]]])

        # Denormalization
        MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
        STD = torch.tensor([0.2023, 0.1994, 0.2010])

        original_image = selected_imgs[ind].cpu() * STD[:, None, None] + MEAN[:, None, None]
        
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))

        methods=["original_image","blended_heat_map"]
        signs=["all","absolute_value"]
        titles=["Original Image","Saliency Map"]
        colorbars=[False,True]
        for i in range(0,2):
            plt_fig_axis = (fig,ax[2*ind+i])
            _ = viz.visualize_image_attr(grads, original_image,
                                        method=methods[i],
                                        sign=signs[i],
                                        plt_fig_axis=plt_fig_axis,
                                        show_colorbar=colorbars[i],
                                        title=titles[i])
                
    fig.savefig(f"SaliencyMaps/SVHN/Sess{ses}SalMap.png")
    fig.show()  
    model.set_saliency(False)


def main():

    model = BasicNet1(args, 0).cuda() 


    print('  Total params: %.2fM ' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.savepoint):
        mkdir_p(args.savepoint)
    np.save(args.checkpoint + "/seed.npy", seed)
    
    inc_dataset = data.IncrementalDataset(
                        dataset_name=args.dataset,
                        args = args,
                        random_order=args.random_classes,
                        shuffle=True,
                        seed=1,
                        batch_size=args.train_batch,
                        workers=args.workers,
                        validation_split=args.validation,
                        increment=args.class_per_task,
                    )
        
    start_sess = int(sys.argv[1])
    memory = None
    
    for ses in range(start_sess, args.num_task):
        args.sess=ses 
        if(ses>=args.num_task):
            ses = args.num_task-1
            args.sess=ses 
            
        if(ses==0):
            torch.save(model.state_dict(), os.path.join(args.savepoint, 'base_model.pth.tar'))
            mask = {}
            
        if(start_sess==ses and start_sess!=0): 
            inc_dataset._current_task = ses
            with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess-1)+".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing
        
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(ses-1) + '_model_best.pth.tar') 

            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best)

            with open(args.savepoint + "/memory_"+str(args.sess-1)+".pickle", 'rb') as handle:
                memory = pickle.load(handle)
                        
        task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)
        ### Saliency
        if ses==0: salLoader = test_loader
        
        print(task_info)
        print(inc_dataset.sample_per_task_testing)
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        
        main_learner=Learner(model=model,args=args,trainloader=train_loader, testloader=test_loader, use_cuda=use_cuda)
        
        pred=main_learner.learn()
        memory = inc_dataset.get_memory(memory, for_memory)     

        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)
        
        ### Saliency
        createSaliencyMap(model, salLoader, pred, ses)

        
        with open(args.savepoint + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(inc_dataset.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        time.sleep(5)
if __name__ == '__main__':
    main()
