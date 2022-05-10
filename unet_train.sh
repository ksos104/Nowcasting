#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J tunet
#SBATCH -o logs/stdout_%j.txt
#SBATCH -e logs/stderr_%j.txt
#SBATCH --gres=gpu:2

# python train_unet.py --model encdec --data_path=/mnt/server14_hard0/dlsfbtp/dataset/korea/sliding_numpy_data --n_gpus=2 --batch_size=32 --total_epoch=20

python train_unet.py --model tunet --data_path=/mnt/server11_hard3/jiny/Nowcasting/Nowcasting/data/kTPW_sample --n_gpus=2 --batch_size=2 --total_epoch=20

