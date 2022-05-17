#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J unet
#SBATCH -o logs/stdout_%j.txt
#SBATCH -e logs/stderr_%j.txt
#SBATCH --gres=gpu:4

# python train_unet.py --model encdec --data_path=/mnt/server14_hard0/dlsfbtp/dataset/korea/sliding_numpy_data --n_gpus=2 --batch_size=64 --total_epoch=20

# CUDA_VISIBLE_DEVICES=0,1 python train_unet.py --model tunet --data_path=/mnt/server11_hard3/jiny/Nowcasting/Nowcasting/data/kTPW --n_gpus=2 --batch_size=64 --total_epoch=20

python train_unet.py --model unet --data_path=/mnt/server11_hard3/jiny/Nowcasting/Nowcasting/data/kTPW/ --n_gpus=4 --batch_size=128 --total_epoch=20 --n_workers=5 #--load=2022-05-11_02-41_tunet
