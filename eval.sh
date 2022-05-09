#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm_out/stdout_%j.txt
#SBATCH -e slurm_err/stderr_%j.txt
#SBATCH --gres=gpu:1

conda activate jiny11038
gpu=1
bs=2
dPath=/mnt/server11_hard3/jiny/Nowcasting/Nowcasting/data/kTPW_sample

tunetPath=/mnt/server8_hard3/msson/VPTR/trained/2022-05-08_20-06
unetPath=/mnt/server8_hard3/msson/VPTR/trained/2022-05-08_20-06
encdecPath=/mnt/server8_hard3/msson/VPTR/trained/2022-05-08_20-06

#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model tunet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${tunetPath}

CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model encdec --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${encdecPath}

#CUDA_VISIBLE_DEVICES=${gpu} python eval_unet.py --model unet --data_path=${dPath} --n_gpus=1 --batch_size=${bs} --load=${unetPath}

#python plot_graph.py --plotMetric POD16 SUCR16 CSI16 BIAS16 POD74 SUCR74 CSI74 BIAS74
