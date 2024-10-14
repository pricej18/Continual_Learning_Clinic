#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=iTAML.out

nvcc --version
echo -e "\n\n"

module load cuda11/11.4
module load python/3.7.15

# GPU Venv setup
# python3 -m venv gpuEnv
# source gpuEnv/bin/activate
# python3 -m pip install -r requirements.txt
# echo
python3 test.py
# echo -e "\n\n\n"

# CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py
