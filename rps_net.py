'''
RPS network script with resnet-18
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class RPS_net_cifar(nn.Module):

        def __init__(self, args):
            super(RPS_net_cifar, self).__init__()
            self.args = args
            self.final_layers = []
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.init(None)

        def init(self, best_path):


            """Initialize all parameters"""
            self.conv1 = []
            self.conv2 = []
            self.conv3 = []
            self.conv4 = []
            self.conv5 = []
            self.conv6 = []
            self.conv7 = []
            self.conv8 = []
            self.conv9 = []
            self.fc1 = []

            div = 1
            a1 = 64//div
            
            a2 = 64//div
            a3 = 128//div
            a4 = 256//div
            a5 = 512//div
            
            self.a5 =a5
            # conv1
            for i in range(self.args.M):
                exec("self.m1" + str(i) + " = nn.Sequential(nn.Conv2d(3, "+str(a1)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"),nn.ReLU())")
                exec("self.conv1.append(self.m1" + str(i) + ")")


            # conv2
            for i in range(self.args.M):
                exec("self.m2" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a1)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a1)+"))")
                exec("self.conv2.append(self.m2" + str(i) + ")")

            # conv3
            for i in range(self.args.M):
                exec("self.m3" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"),nn.ReLU(), nn.Conv2d("+str(a2)+", "+str(a2)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a2)+"))")
                exec("self.conv3.append(self.m3" + str(i) + ")")
            


            # conv4
            for i in range(self.args.M):
                exec("self.m4" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv4.append(self.m4" + str(i) + ")")
            exec("self.m4" + str("x") + " = nn.Sequential(nn.Conv2d("+str(a2)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU())")
            exec("self.conv4.append(self.m4" + str("x") + ")")

            # conv5
            for i in range(self.args.M):
                exec("self.m5" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"),nn.ReLU(), nn.Conv2d("+str(a3)+", "+str(a3)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a3)+"))")
                exec("self.conv5.append(self.m5" + str(i) + ")")
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
              
                

            # conv6
            for i in range(self.args.M):
                exec("self.m6" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv6.append(self.m6" + str(i) + ")")
            exec("self.m6" + str("x") + " = nn.Sequential(nn.Conv2d("+str(a3)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU())")
            exec("self.conv6.append(self.m6" + str("x") + ")")

            # conv7
            for i in range(self.args.M):
                exec("self.m7" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"),nn.ReLU(), nn.Conv2d("+str(a4)+", "+str(a4)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a4)+"))")
                exec("self.conv7.append(self.m7" + str(i) + ")")
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            
            

            # conv8
            for i in range(self.args.M):
                exec("self.m8" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv8.append(self.m8" + str(i) + ")")
            exec("self.m8" + str("x") + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU())")
            exec("self.conv8.append(self.m8" + str("x") + ")")
            
            # conv9
            for i in range(self.args.M):
                exec("self.m9" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
            #    exec("self.m9" + str(i) + " = nn.Sequential(nn.Conv2d("+str(a4)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"),nn.ReLU(), nn.Conv2d("+str(a5)+", "+str(a5)+", kernel_size=3, stride=1, padding=1),nn.BatchNorm2d("+str(a5)+"))")
                exec("self.conv9.append(self.m9" + str(i) + ")")
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            
            
            if len(self.final_layers) < 1:
                self.final_layer1 = nn.Linear(a5, self.args.num_class)
                #self.final_layer1 = nn.Linear(a5, 10)
                self.final_layers.append(self.final_layer1)

            self.to(self.device)


        def forward(self, x, path, last):

            M = self.args.M
            div = 1

            y = self.conv1[0](x)
            for j in range(1, self.args.M):
                if path[0][j] == 1:
                    y = y + self.conv1[j](x)
            x = F.relu(y)

            y = self.conv2[0](x)
            for j in range(1, self.args.M):
                if path[1][j] == 1:
                    y = y + self.conv2[j](x)
            x = y + x
            x = F.relu(x)

            y = self.conv3[0](x)
            for j in range(1, self.args.M):
                if path[2][j] == 1:
                    y = y + self.conv3[j](x)
            x = y + x
            x = F.relu(x)

            y = self.conv4[-1](x)
            for j in range(self.args.M):
                if path[3][j] == 1:
                    y = y + self.conv4[j](x)
            x = y  # Note: No modification in place
            x = F.relu(x)

            y = self.conv5[0](x)
            for j in range(1, self.args.M):
                if path[4][j] == 1:
                    y = y + self.conv5[j](x)
            x = y + x
            x = F.relu(x)
            x = self.pool1(x)

            y = self.conv6[-1](x)
            for j in range(self.args.M):
                if path[5][j] == 1:
                    y = y + self.conv6[j](x)
            x = y  # No modification in place
            x = F.relu(x)

            y = self.conv7[0](x)
            for j in range(1, self.args.M):
                if path[6][j] == 1:
                    y = y + self.conv7[j](x)
            x = y  # No modification in place
            x = F.relu(x)
            x = self.pool2(x)

            
            y = self.conv8[-1](x)
            for j in range(self.args.M):
                if path[7][j] == 1:
                    y = y + self.conv8[j](x)
            x = y  # No modification in place
            x = F.relu(x)
            

            y = self.conv9[0](x)
            for j in range(1, self.args.M):
                if path[8][j] == 1:
                    y = y + self.conv9[j](x)
            x = y + x  # No modification in place
            x = F.relu(x)

            x = F.avg_pool2d(x, (8, 8), stride=(1, 1))
            x = x.view(-1, self.a5)
            x = self.final_layers[last](x)

            return x

        
class RPS_net_mlp(nn.Module):

        def __init__(self, args):
            super(RPS_net_mlp, self).__init__()
            self.args = args
            self.final_layers = []
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.init(None)


        def init(self, best_path):


            """Initialize all parameters"""

            self.mlp1 = []
            self.mlp2 = []
            self.fc1 = []

            #mlp1
            for i in range(self.args.M):
#                 exec("self.m1" + str(i) + " = nn.Sequential(nn.Linear(784, 256),nn.BatchNorm2d(256),nn.ReLU())")
                exec("self.m1" + str(i) + " = nn.Linear(784, 400)")
                exec("self.mlp1.append(self.m1" + str(i) + ")")
                
                
            #mlp2
            for i in range(self.args.M):
                exec("self.m2" + str(i) + " = nn.Linear(400, 128)")
                exec("self.mlp2.append(self.m2" + str(i) + ")")
                

            if len(self.final_layers) < 1:
                self.final_layer1 = nn.Linear(128, 10)
                self.final_layers.append(self.final_layer1)

            self.to(self.device)


        def forward(self, x, path, last):

            M = self.args.M
           
            y = self.mlp1[0](x)
            for j in range(1,self.args.M):
                if(path[0][j]==1):
                    y += self.mlp1[j](x)
            x = F.relu(y)

            
            
            y = self.mlp2[0](x)
            for j in range(1,self.args.M):
                if(path[1][j]==1):
                    y += self.mlp2[j](x)
            x = F.relu(y)
            
            x = self.final_layers[last](x)
            
            return x


def get_path(L, M, N):
    path=np.zeros((L,M),dtype=float)
    for i in range(L):
        j=0
        while j<N:
            rand_value=int(np.random.rand()*M)
            if path[i,rand_value]==0.0:
                path[i,rand_value]=1.0
                j+=1
    return path


def generate_path(ses, dataset, args):
    '''
    if ses == 0:
        path = get_path(args.L, args.M, args.N) * 0
        path[:, 0] = 1  # Set the first column to 1
        fixed_path = get_path(args.L, args.M, args.N) * 0  # All zeros
        infer_path = path.copy()  # Initialize infer_path as zeros
    else:
    '''

    previous_ses = ses - 1

    fixed_path_filepath = f"Saliency/RPSnet/{dataset}/fixed_path_{ses}_0.npy"
    path_filepath = f"Saliency/RPSnet/{dataset}/path_{ses}_0.npy"

    # Load the arrays
    fixed_path = np.load(fixed_path_filepath)
    path = np.load(path_filepath)

    # Initialize infer_path with the same shape as fixed_path
    infer_path = np.zeros_like(fixed_path)

    # Reconstruct infer_path based on the logic provided
    for j in range(fixed_path.shape[0]):  # Assuming fixed_path is 2D
        for i in range(fixed_path.shape[1]):
            if fixed_path[j, i] == 1 or path[j, i] == 1:
                infer_path[j, i] = 1

    #print("path\n", path)
    #print("fixed_path\n", fixed_path)
    #print("infer_path\n", infer_path)
    #print('\n\n')
    return infer_path