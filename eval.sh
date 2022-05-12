#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm_out/stdout_%j.txt
#SBATCH -e slurm_err/stderr_%j.txt
#SBATCH --gres=gpu:1

conda activate jiny11038
gpu=7
bs=1
dPath=/mnt/server11_hard3/jiny/Nowcasting/Nowcasting/data/kTPW_sample
split=easy #hard medium
tunetPath=./trained/2022-05-11_22-17_tunet/019_0.0425.ckpt
unetPath=./trained/2022-05-11_22-17_unet
encdecPath=/mnt/server8_hard3/msson/VPTR/trained/2022-05-08_20-06

#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model tunet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${tunetPath} --split full
CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model tunet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${tunetPath} --split easy
#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model tunet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${tunetPath} --split hard
#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model tunet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${tunetPath} --split medium

#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model tunet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${tunetPath} --split full
#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model unet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${unetPath} --split easy
#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model unet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${unetPath} --split hard
#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model unet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${unetPath} --split medium

#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model encdec --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${encdecPath}

#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model unet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${unetPath}

CUDA_VISIBLE_DEVICES=${gpu} python plot_graph.py 
