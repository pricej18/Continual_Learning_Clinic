#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=results.out
CUDA_VISIBLE_DEVICES=0 

chmod +x main*.py compare*.py all_results.sh
python3 -m main --replay=generative --experiment=splitMNIST --scenario=class 

