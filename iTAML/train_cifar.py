'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function
import argparse
import os
import shutil
import time
import pickle
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import pdb
import numpy as np
import copy
import sys
import random
import collections
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import mkdir_p
from basic_net import *
# from learner_task_reptile import Learner
# from learner_task_FOMAML import Learner
# from learner_task_joint import Learner
from learner_task_itaml import Learner
import incremental_dataloader as data

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

class args:

    checkpoint = "results/cifar100/meta2_cifar_T10_71"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/CIFAR100/"
    num_class = 100
    class_per_task = 10
    num_task = 10
    test_samples_per_class = 100
    dataset = "cifar100"
    optimizer = "radam"
    
    epochs = 70
    #epochs = 1
    lr = 0.01
    train_batch = 128
    test_batch = 100
    workers = 16
    sess = 0
    schedule = [20,40,60]
    gamma = 0.2
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 1.0
    r = 2
    
state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
print(state)

use_cuda = torch.cuda.is_available()
#seed = random.randint(1, 10000)
seed = 7572 
#random.seed(seed)
#np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)



def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
      for j in class_name:
        if dataset.targets[i] == j:
            indices.append(i)
    return indices
    
    
    
def load_saliency_data(desired_classes, imgs_per_class):

    if not os.path.isdir("SaliencyMaps/" + args.dataset):
        mkdir_p("SaliencyMaps/" + args.dataset)
    
    saliencySet = torch.utils.data.Dataset()
    if args.dataset == "mnist":       
        saliencySet = datasets.MNIST(root='SaliencyMaps/Datasets/mnist/', train=False,
                  download=True,
                  transform=transforms.Compose([transforms.ToTensor()]))
    elif args.dataset == "cifar10":       
        saliencySet = datasets.CIFAR10(root='SaliencyMaps/Datasets/cifar10/', train=False,
                  download=True,
                  transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif args.dataset == "cifar100":       
        saliencySet = datasets.CIFAR100(root='SaliencyMaps/Datasets/cifar100/', train=False,
                  download=True,
                  transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))

    idx = get_indices(saliencySet,desired_classes)

    subset = Subset(saliencySet, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=args.test_batch)

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
    
    
    
def create_saliency_map(model, ses, desired_classes, imgs_per_class):
    
    sal_imgs, sal_labels, classes = load_saliency_data(desired_classes, imgs_per_class)
    sal_imgs, sal_labels = sal_imgs.cuda(), sal_labels.cuda()
    
    outputs2, outputs = model(sal_imgs)
    pred = torch.argmax(outputs2[:,0:args.class_per_task*(1+args.sess)], 1, keepdim=False)
    predicted = pred.squeeze()        
    
    
    model.set_saliency(True)
    saliency = Saliency(model)
    
    fig, ax = plt.subplots(2,len(desired_classes)*imgs_per_class,figsize=(15,5))
    for ind in range(len(desired_classes)*imgs_per_class):
        input = sal_imgs[ind].unsqueeze(0)
        input.requires_grad = True

        grads = saliency.attribute(input, target=sal_labels[ind].item(), abs=False)
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        print('Truth:', classes[sal_labels[ind]])
        print('Predicted:', classes[predicted[ind]])


        # Denormalization
        MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
        STD = torch.tensor([0.2675, 0.2565, 0.2761])

        original_image = sal_imgs[ind].cpu() * STD[:, None, None] + MEAN[:, None, None]
        
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))
        
        
        methods=["original_image","blended_heat_map"]
        signs=["all","absolute_value"]
        titles=["Original Image","Saliency Map"]
        colorbars=[False,True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]: cmap = "Reds" 
        else: cmap = "Blues"


        if ind > 4:
            row = 1
            ind = ind - 5
        else: row = 0

        for i in range(2):
            plt_fig_axis = (fig,ax[row][2*ind+i])
            _ = viz.visualize_image_attr(grads, original_image,
                                        method=methods[i],
                                        sign=signs[i],
                                        fig_size=(4,4),
                                        plt_fig_axis=plt_fig_axis,
                                        cmap=cmap,
                                        show_colorbar=colorbars[i],
                                        title=titles[i])
    
    fig.tight_layout()
    fig.savefig(f"SaliencyMaps/{args.dataset}/Sess{ses}SalMap.png")
    fig.show()
    model.set_saliency(False)


def main():

    model = BasicNet1(args, 0).cuda() 
#     model = nn.DataParallel(model).cuda()

    print('  Total params: %.2fM ' % (sum(p.numel() for p in model.parameters())/1000000.0))


    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.savepoint):
        mkdir_p(args.savepoint)
    np.save(args.checkpoint + "/seed.npy", seed)
    try:
        shutil.copy2('train_cifar.py', args.checkpoint)
        shutil.copy2('learner_task_itaml.py', args.checkpoint)
    except:
        pass
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
    print("Seed: " + str(seed))
    
    for ses in range(start_sess, args.num_task):
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

        
        print(task_info)
        print(inc_dataset.sample_per_task_testing)
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        
        
        
        main_learner=Learner(model=model,args=args,trainloader=train_loader, testloader=test_loader, use_cuda=use_cuda)
        
        main_learner.learn()
        memory = inc_dataset.get_memory(memory, for_memory)       
        
        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)
        
        ### Saliency
        create_saliency_map(model, ses, [0,1,2,3,4,5,6,7,8,9], 1)
        
        with open(args.savepoint + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        time.sleep(10)
if __name__ == '__main__':
    main()
