#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm_out/stdout_%j.txt
#SBATCH -e slurm_err/stderr_%j.txt
#SBATCH --gres=gpu

python train_unet.py --data_path=/mnt/server14_hard0/dlsfbtp/dataset/korea/sliding_numpy_data --n_gpus=1 --batch_size=2 --total_epoch=100
