#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mnist.out

source ./timestamp.sh

timestamp "start"
. ./MNIST.sh
timestamp "end"
