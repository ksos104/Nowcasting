#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm_out/stdout_%j.txt
#SBATCH -e slurm_err/stderr_%j.txt
#SBATCH --gres=gpu:2

python train_unet.py --model encdec --data_path=/mnt/server14_hard0/dlsfbtp/dataset/korea/sliding_numpy_data --n_gpus=2 --batch_size=32 --total_epoch=20
