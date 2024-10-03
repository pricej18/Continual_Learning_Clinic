#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --output=gpuInfo.out
#SBATCH --mail-type=BEGIN        # send email when job begins
#SBATCH --mail-type=END          # send email when job ends
#SBATCH --mail-type=FAIL         # send email if job fails
#SBATCH --mail-user=abanyi17@students.rowan.edu

nvidia-smi
