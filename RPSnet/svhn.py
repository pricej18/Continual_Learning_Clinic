'''
RPS network training on CIFAR100
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
from torch.utils.data import Dataset, TensorDataset

from rps_net import RPS_net_mlp, RPS_net_cifar
#from rps_net import RPS_net_mlp
#from rps_net2 import RPS_net_cifar
from learner import Learner
from util import *
from cifar_dataset import CIFAR100

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt


class args:
#    epochs = 10
    epochs = 1
    checkpoint = "results/svhn/RPS_net_svhn"
    savepoint = "results/svhn/pathnet_svhn"
    dataset = "SVHN"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1
    
state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
print(state)
memory = 4400
# Use CUDA
use_cuda = torch.cuda.is_available()

seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)


    
def load_svhn():
    from scipy import io as spio
    from keras.utils import to_categorical
    import numpy as np
    svhn = spio.loadmat("train_32x32.mat")
    x_train = np.einsum('ijkl->lijk', svhn["X"]).astype(np.float32) / 255.
    y_train = (svhn["y"] - 1)

    svhn_test = spio.loadmat("test_32x32.mat")
    x_test = np.einsum('ijkl->lijk', svhn_test["X"]).astype(np.float32) / 255.
    y_test = (svhn_test["y"] - 1)

    x_train = np.transpose(x_train, [0,3,1,2])
    x_test = np.transpose(x_test, [0,3,1,2])
    y_train = np.reshape(y_train, (-1))
    y_test = np.reshape(y_test, (-1))

    return (x_train, y_train), (x_test, y_test)


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
    
    
def create_saliency_map(model, path, saliency_loader, pred, ses):
    ##### Create Saliency Maps
    data_iter = iter(saliency_loader)
    sal_imgs, sal_labels = next(data_iter)
    sal_imgs, sal_labels = sal_imgs.cuda(), sal_labels.cuda()
    predicted = pred.squeeze()
    
    saliency = Saliency(model)

    # Finds a '1' for testing
    for i in range(len(saliency_loader)-1):
        sal_imgs2, sal_labels2 = next(data_iter)
        if sal_labels2[0] == torch.tensor([ses+i*args.class_per_task]): break
    
    fig, ax = plt.subplots(1,4,figsize=(10,4))
    for ind in range(0,2):
        if ind==0: input = sal_imgs[0].unsqueeze(0)
        else: input = sal_imgs2[0].unsqueeze(0)

        input.requires_grad = True

        if ind==0: grads = saliency.attribute(input, target=sal_labels[0].item(), abs=False, additional_forward_args = (path, -1))
        else: grads = saliency.attribute(input, target=sal_labels2[0].item(), abs=False, additional_forward_args = (path, -1))
        
        squeeze_grads = grads.squeeze().cpu().detach()
        grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))

        if ind==0: print('Truth:', classes[sal_labels[0]])
        else: print('Truth:', classes[sal_labels2[0]])
        
        # Denormalization
        MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
        STD = torch.tensor([0.2023, 0.1994, 0.2010])
        
        if ind==0: original_image = sal_imgs[0].cpu() * STD[:, None, None] + MEAN[:, None, None]
        else: original_image = sal_imgs2[0].cpu() * STD[:, None, None] + MEAN[:, None, None]
        original_image = np.transpose((original_image.detach().numpy()), (1, 2, 0))       
        

        methods=["original_image","blended_heat_map"]
        signs=["all","absolute_value"]
        titles=["Original Image","Saliency Map"]
        colorbars=[False,True]
        for i in range(0,2):
            plt_fig_axis = (fig,ax[2*ind+i])
            if i==1:
                _ = viz.visualize_image_attr(grads, original_image,
                                            method=methods[i],
                                            sign=signs[i],
                                            plt_fig_axis=plt_fig_axis,
                                            show_colorbar=colorbars[i],
                                            title=titles[i])
            else:
                ax[2*ind+i].imshow(original_image, cmap='gray')
                ax[2*ind+i].set_title('Original Image')
                ax[2*ind+i].tick_params(left = False, right = False, labelleft = False, 
                                labelbottom = False, bottom = False) 
            
    fig.savefig(f"SaliencyMaps/SVHN/Sess{ses}SalMap.png")
    fig.show()      
    


def main():

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
        
    if not os.path.isdir("models/svhn/"+args.checkpoint.split("/")[-1]):
        mkdir_p("models/svhn/"+args.checkpoint.split("/")[-1])
    args.savepoint = "models/svhn/"+args.checkpoint.split("/")[-1]
    
    
    


    model = RPS_net_cifar(args).cuda()    #for SVHN and CIFAR10 
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    start_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    args.test_case = test_case


    (x_train, y_train), (x_test, y_test) = load_svhn()
    
        
    for ses in range(start_sess, start_sess+1):
        
        if(ses==0):
            path = get_path(args.L,args.M,args.N)*0 
            path[:,0] = 1
            fixed_path = get_path(args.L,args.M,args.N)*0 
            train_path = path.copy()
            infer_path = path.copy()
        else:
            load_test_case = get_best_model(ses-1, args.checkpoint)
            if(ses%args.jump==0):   #get a new path
                fixed_path = np.load(args.checkpoint+"/fixed_path_"+str(ses-1)+"_"+str(load_test_case)+".npy")
                path = get_path(args.L,args.M,args.N)
                train_path = get_path(args.L,args.M,args.N)*0 
            else:
                if((ses//args.jump)*2==0):
                    fixed_path = get_path(args.L,args.M,args.N)*0
                else:
                    load_test_case_x = get_best_model((ses//args.jump)*args.jump-1, args.checkpoint)
                    fixed_path = np.load(args.checkpoint+"/fixed_path_"+str((ses//args.jump)*args.jump-1)+"_"+str(load_test_case_x)+".npy")
                path = np.load(args.checkpoint+"/path_"+str(ses-1)+"_"+str(load_test_case)+".npy")
                train_path = get_path(args.L,args.M,args.N)*0 
            infer_path = get_path(args.L,args.M,args.N)*0 
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        train_path[j,i]=1
                    if(fixed_path[j,i]==1 or path[j,i]==1):
                        infer_path[j,i]=1
            
        np.save(args.checkpoint+"/path_"+str(ses)+"_"+str(test_case)+".npy", path)
        if(ses==0):
            fixed_path_x = path.copy()
        else:
            fixed_path_x = fixed_path.copy()
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path_x[j,i]==0 and path[j,i]==1):
                        fixed_path_x[j,i]=1
        np.save(args.checkpoint+"/fixed_path_"+str(ses)+"_"+str(test_case)+".npy", fixed_path_x)
        
        
        
        print('Starting with session {:d}'.format(ses))
        print('test case : ' + str(test_case))
        print('#################################################################################')
        print("path\n",path)
        print("fixed_path\n",fixed_path)
        print("train_path\n", train_path)
        print("infer_path\n", infer_path)
        
        
        ids_train = []
        for j in range((ses*args.class_per_task), (ses+1)*args.class_per_task):
            ids_train.append(np.where(y_train==j)[0])
        ids_test = []
        for j in range((ses+1)*args.class_per_task):
            ids_test.append(np.where(y_test==j)[0])

        ids_train = flatten_list(ids_train)
        ids_test = flatten_list(ids_test)

        if(ses>0):
            ids_exp = []
            for j in range((ses)*args.class_per_task):
                ex_id =np.where(y_train==j)[0]
                sample_per_class = memory//(ses*args.class_per_task)
                if(len(ex_id)>sample_per_class):
                    ids_exp.append(ex_id[0:sample_per_class])
                else:
                    ids_exp.append(ex_id)
            ids_exp = np.tile(flatten_list(ids_exp),10)
            train_data = np.vstack([x_train[ids_train], x_train[ids_exp]])
            train_label = np.vstack([np.reshape(y_train[ids_train],(-1,1)), np.reshape(y_train[ids_exp],(-1,1))])
            train_label = flatten_list(train_label)

        else:
            ids_exp = []
            train_data = x_train[ids_train]
            train_label = y_train[ids_train]
        
        test_data = x_test[ids_test]
        test_label = y_test[ids_test]

        import torch.utils.data as utils
        args.sess = ses
        
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        
        train_dataset = CustomTensorDataset((torch.tensor(train_data), torch.tensor(train_label).long()), transform=transform_train)
        train_loader = utils.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True)

        test_dataset = CustomTensorDataset((torch.tensor(test_data), torch.tensor(test_label).long()), transform=transform_test)
        test_loader = utils.DataLoader(test_dataset, batch_size=args.test_batch)
        
        
        ### Saliency
        saliency_loader = test_loader

        main_learner = Learner(model=model, args=args, trainloader=train_loader,
                               testloader=test_loader, old_model=copy.deepcopy(model),
                               use_cuda=use_cuda, path=path,
                               fixed_path=fixed_path, train_path=train_path, infer_path=infer_path)
        pred = main_learner.learn()
        
        ### Saliency
        create_saliency_map(model, infer_path, saliency_loader, pred, ses)
        
        if(ses==0):
            fixed_path = path.copy()
        else:
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        fixed_path[j,i]=1
        np.save(args.checkpoint+"/fixed_path_"+str(ses)+"_"+str(test_case)+".npy", fixed_path)
        
        
        
        
        best_model = get_best_model(ses, args.checkpoint)
    
        cfmat = main_learner.get_confusion_matrix(infer_path)
        np.save(args.checkpoint+"/confusion_matrix_"+str(ses)+"_"+str(test_case)+".npy", cfmat)
        
    print('done with session {:d}'.format(ses))
    print('#################################################################################')
    while(1):
        if(is_all_done(ses, args.epochs, args.checkpoint)):
            break
        else:
            time.sleep(10)
            
            
            
if __name__ == '__main__':
    main()
