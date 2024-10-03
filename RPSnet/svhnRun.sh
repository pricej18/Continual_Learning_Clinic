#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=svhn.out

. ./SVHN2G.sh
