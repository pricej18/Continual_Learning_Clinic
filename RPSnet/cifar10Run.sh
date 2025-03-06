#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --output=cifar10.out

. ./CIFAR10.sh
