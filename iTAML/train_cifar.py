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
import pdb
import numpy as np
import copy
import sys
import random
import collections

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
classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium fish',
    'flatfish', 'ray', 'shark', 'trout','orchids', 'poppies', 'roses',
    'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups', 'plates',
	'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'clock',
    'computer keyboard', 'lamp', 'telephone', 'television', 'bed', 'chair',
    'couch', 'table', 'wardrobe', 'bee', 'beetle', 'butterfly', 'caterpillar',
    'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf', 'bridge',
    'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain',
    'plain', 'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
	'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster',
    'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse',
    'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow',
	'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower',
    'rocket', 'streetcar', 'tank', 'tractor'
    )

use_cuda = torch.cuda.is_available()
seed = random.randint(1, 10000)
seed = 7572 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

def create_saliency_map(model, saliency_loader, pred, ses):
    # Get Images, one for each class
    data_iter = iter(saliency_loader)
    sal_imgs, sal_labels = next(data_iter)
    print(sal_labels)

    num = -1
    selected_labels = []
    for img in sal_imgs:
        num = num + 1
        if len(selected_labels) > 9: break
        if selected_labels and sal_labels[num] in torch.index_select(sal_labels,0,torch.tensor(selected_labels)): continue
        else: selected_labels.append(num)
    print(selected_labels)
    selected_imgs = torch.index_select(sal_imgs, 0, torch.tensor(selected_labels))
    sal_imgs, sal_labels, selected_imgs = sal_imgs.cuda(), sal_labels.cuda(), selected_imgs.cuda()
    predicted = pred.squeeze()
    
    
    model.set_saliency(True)
    saliency = Saliency(model)
    
    fig, ax = plt.subplots(2,10,figsize=(17,5))
    for ind in range(0,10):
        input = selected_imgs[ind].unsqueeze(0)
        input.requires_grad = True

        grads = saliency.attribute(input, target=sal_labels[selected_labels[ind]].item(), abs=False)
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        #grads = grads.squeeze().cpu().detach().numpy()

        print('Truth:', classes[sal_labels[selected_labels[ind]]])
        print('Predicted:', classes[predicted[selected_labels[ind]]])


        # Denormalization
        MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
        STD = torch.tensor([0.2675, 0.2565, 0.2761])

        original_image = selected_imgs[ind].cpu() * STD[:, None, None] + MEAN[:, None, None]
        
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))

        methods=["original_image","blended_heat_map"]
        signs=["all","absolute_value"]
        titles=["Original Image","Saliency Map"]
        colorbars=[False,True]

        if ind > 4:
            row = 1
            ind = ind - 5
        else: row = 0

        for i in range(0,2):
            plt_fig_axis = (fig,ax[row][2*ind+i])
            _ = viz.visualize_image_attr(grads, original_image,
                                        method=methods[i],
                                        sign=signs[i],
                                        fig_size=(4,4),
                                        plt_fig_axis=plt_fig_axis,
                                        show_colorbar=colorbars[i],
                                        title=titles[i])
                
    fig.savefig(f"SaliencyMaps/CIFAR100/Sess{ses}SalMap.png")
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
        ### Saliency
        if ses==0: saliency_loader = test_loader
        
        print(task_info)
        print(inc_dataset.sample_per_task_testing)
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        
        
        
        main_learner=Learner(model=model,args=args,trainloader=train_loader, testloader=test_loader, use_cuda=use_cuda)
        
        pred=main_learner.learn()
        memory = inc_dataset.get_memory(memory, for_memory)       
        
        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)
        
        ### Saliency
        create_saliency_map(model, saliency_loader, pred, ses)
        
        with open(args.savepoint + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        time.sleep(10)
if __name__ == '__main__':
    main()
