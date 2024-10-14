#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=cifar.out

source ./timestamp.sh

timestamp "start"
CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py 0
timestamp "end"
