#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=cifar.out

CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py 0
