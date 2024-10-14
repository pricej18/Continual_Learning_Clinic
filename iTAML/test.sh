#!/bin/sh
#SBATCH --output=test.out

source ./timestamp.sh

timestamp "start"
#python3 assignment_2_1.py
python3 test.py
timestamp "end"