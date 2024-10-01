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
import sys
import numpy as np
import copy
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from rps_net import RPS_net_cifar
from learner import Learner
from cifar_dataset import CIFAR100

# Argument parser setup
def get_args():
    parser = argparse.ArgumentParser(description='RPSNet CIFAR Training')
    parser.add_argument('--checkpoint', type=str, default='results/cifar100/RPS_CIFAR_M8_J1', help='Path to save the checkpoint')
    parser.add_argument('--labels_data', type=str, default='prepare/cifar100_10.pkl', help='Path to the labels data')
    parser.add_argument('--num_class', type=int, default=100, help='Number of classes')
    parser.add_argument('--class_per_task', type=int, default=10, help='Number of classes per task')
    parser.add_argument('--M', type=int, default=8, help='Number of tasks')
    parser.add_argument('--jump', type=int, default=2, help='Jump parameter')
    parser.add_argument('--rigidness_coff', type=float, default=2.5, help='Rigidness coefficient')
    parser.add_argument('--dataset', type=str, default='CIFAR', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--L', type=int, default=9, help='Number of layers')
    parser.add_argument('--N', type=int, default=1, help='Parameter N')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_batch', type=int, default=128, help='Training batch size')
    parser.add_argument('--test_batch', type=int, default=128, help='Testing batch size')
    parser.add_argument('--workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--arch', type=str, default='res-18', help='Architecture name')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch for training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--sess', type=int, default=0, help='Session number')
    parser.add_argument('--test_case', type=int, default=0, help='Test case number')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 40, 60, 80], help='Learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')

    return parser.parse_args()

# Use CUDA
use_cuda = torch.cuda.is_available()
seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()  # Get arguments

    model = RPS_net_cifar(args).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if not os.path.isdir("models/CIFAR100/" + args.checkpoint.split("/")[-1]):
        mkdir_p("models/CIFAR100/" + args.checkpoint.split("/")[-1])
    args.savepoint = "models/CIFAR100/" + args.checkpoint.split("/")[-1]

    # Transformations for CIFAR-100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = CIFAR100  # Ensure CIFAR100 is a valid dataset loader

    start_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    args.test_case = test_case

    inds_all_sessions = pickle.load(open(args.labels_data, 'rb'))

    for ses in range(start_sess, start_sess + 1):
        if ses == 0:
            path = get_path(args.L, args.M, args.N) * 0
            path[:, 0] = 1
            fixed_path = get_path(args.L, args.M, args.N) * 0
            train_path = path.copy()
            infer_path = path.copy()
        else:
            load_test_case = get_best_model(ses - 1, args.checkpoint)
            if ses % args.jump == 0:  # Get a new path
                fixed_path = np.load(args.checkpoint + "/fixed_path_" + str(ses - 1) + "_" + str(load_test_case) + ".npy")
                path = get_path(args.L, args.M, args.N)
                train_path = get_path(args.L, args.M, args.N) * 0
            else:
                if (ses // args.jump) == 0:
                    fixed_path = get_path(args.L, args.M, args.N) * 0
                else:
                    load_test_case_x = get_best_model((ses // args.jump) * args.jump - 1, args.checkpoint)
                    fixed_path = np.load(args.checkpoint + "/fixed_path_" + str((ses // args.jump) * args.jump - 1) + "_" + str(load_test_case_x) + ".npy")
                path = np.load(args.checkpoint + "/path_" + str(ses - 1) + "_" + str(load_test_case) + ".npy")
                train_path = get_path(args.L, args.M, args.N) * 0
            infer_path = get_path(args.L, args.M, args.N) * 0
            for j in range(args.L):
                for i in range(args.M):
                    if fixed_path[j, i] == 0 and path[j, i] == 1:
                        train_path[j, i] = 1
                    if fixed_path[j, i] == 1 or path[j, i] == 1:
                        infer_path[j, i] = 1

        np.save(args.checkpoint + "/path_" + str(ses) + "_" + str(test_case) + ".npy", path)

        print('Starting with session {:d}'.format(ses))
        print('test case : ' + str(test_case))
        print('#################################################################################')
        print("path\n", path)
        print("fixed_path\n", fixed_path)
        print("train_path\n", train_path)

        ind_this_session = inds_all_sessions[ses]
        ind_trn = ind_this_session['curent']
        if ses > 0:
            ind_trn = np.concatenate([ind_trn, np.tile(inds_all_sessions[ses - 1]['exmp'], int(1))]).ravel()
        ind_tst = inds_all_sessions[ses]['test']

        # Ensure the data loader is instantiated correctly
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train, ind=ind_trn)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test, ind=ind_tst)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        args.sess = ses
        if ses > 0:
            path_model = os.path.join(args.savepoint, 'session_' + str(ses - 1) + '_' + str(load_test_case) + '_model_best.pth.tar')
            prev_best = torch.load(path_model)
            model.load_state_dict(prev_best['state_dict'])

        main_learner = Learner(model=model, args=args, trainloader=trainloader,
                               testloader=testloader, old_model=copy.deepcopy(model),
                               use_cuda=use_cuda, path=path,
                               fixed_path=fixed_path, train_path=train_path, infer_path=infer_path)
        main_
