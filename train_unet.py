from cmath import inf
import os
import argparse
from tqdm import tqdm
import time
from glob import glob

import numpy as np
import random

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms

from model import unet,EncDec,Tunet
from utils import get_dataloader, MSE, MAE, weather_metrics

import neptune.new as neptune

criterion = nn.L1Loss()

def get_args():
    parser = argparse.ArgumentParser(description="FSS_msson")

    parser.add_argument('--n_gpus', type=int, help='Number of gpus you need', default=1)
    parser.add_argument('--data_path', type=str, help='Path for dataset you want to use. Absolute path is recomended')
    parser.add_argument('--batch_size', type=int, help='Total batch size', default=1)
    parser.add_argument('--n_workers', type=int, help='Number of subprocesses for data loading', default=4)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-4)
    parser.add_argument('--total_epoch', type=int, help='Number of total epoch', default=1)
    parser.add_argument('--load', type=str, help='Model path. ex) 2022-01-01', default=None)
    parser.add_argument('--seed', type=int, help='Seed for all random function', default=212)
    parser.add_argument('--input_frames', type=int, help='Number of input past frames', default=13)
    parser.add_argument('--output_frames', type=int, help='Number of output future frames', default=12)
    parser.add_argument('--model', type=str, help='Select model [encdec, unet, tunet]', choices=['encdec', 'unet', 'tunet'], default='unet')

    args = parser.parse_args()

    return args


def train(args, model, dataloader, optimizer, epoch, total_epoch, rank, npt):
    sum_loss = 0.0
    total_iters = len(dataloader)
    ## tqdm initialization
    pbar = tqdm(enumerate(dataloader), total=total_iters)
    
    train_score = {}
    train_score_accumulate = {}
    
    for iter, batch in pbar:
        ## Get input frames
        input_frames = batch[0].cuda() if torch.cuda.is_available() else batch
        target_frames = batch[1].cuda() if torch.cuda.is_available() else batch
        input_frames = input_frames.squeeze(dim=2)
        target_frames = target_frames.squeeze(dim=2)
        input_frames = input_frames.type(torch.float)
        target_frames = target_frames.type(torch.float)
        batch_size = input_frames.shape[0]

        ## Model forwarding
        outputs = model(input_frames)

        ## Loss calculation (Mean Absolute Error; MAE)
        loss = criterion(outputs, target_frames) / torch.std(torch.cat([input_frames, target_frames], dim=1))         ## 원래 training set 전체의 std로 나눠줌.
        sum_loss += loss

        ## Optimizer update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Calculaet weather metric
        train_score = weather_metrics(outputs,target_frames)

        ## Calculate perceptual metric
        train_score['mse'] = MSE(outputs, target_frames).detach().cpu()
        train_score['mae'] = MAE(outputs, target_frames).detach().cpu()

        for key in train_score.keys():
            train_score_accumulate[key] = train_score_accumulate[key] + train_score[key].detach().cpu() * batch_size if key in train_score_accumulate.keys() else train_score[key].detach().cpu() * batch_size

        ## tqdm update
        pbar.set_description(f"Training >> Epoch: [{epoch+1}/{total_epoch}] Iter: [{iter+1}/{total_iters}] Loss: {loss:.4f}({sum_loss / (iter+1):.4f})", refresh=False)
    dist.barrier()

    avg_loss = sum_loss / (iter+1)
    for key in train_score.keys():
         train_score_accumulate[key] /= len(dataloader.dataset)

    ## Neptune log
    if rank == 0:
        npt["train/avg_loss"].log(avg_loss)

        print(f"Training DONE. Epoch: {epoch}, Avg loss: {avg_loss:.4f}")

    return


def val(args, model, dataloader, epoch, total_epoch, rank, npt):
    model.eval()

    sum_loss = 0.0
    total_iters = len(dataloader)

    ## tqdm initialization
    pbar = tqdm(enumerate(dataloader), total=total_iters)

    val_score = {}
    val_score_accumulate = {}

    with torch.no_grad():
        for iter, batch in pbar:
            ## Get input frames
            input_frames = batch[0].cuda() if torch.cuda.is_available() else batch
            target_frames = batch[1].cuda() if torch.cuda.is_available() else batch
            input_frames = input_frames.squeeze(dim=2)
            target_frames = target_frames.squeeze(dim=2)
            input_frames = input_frames.type(torch.float)
            target_frames = target_frames.type(torch.float)
            batch_size = input_frames.shape[0]
            
            ## Model forwarding
            outputs = model(input_frames)

            ## Loss calculation (Mean Absolute Error; MAE)
            loss = criterion(outputs, target_frames) / torch.std(torch.cat([input_frames, target_frames], dim=1))         ## 원래 training set 전체의 std로 나눠줌.
            sum_loss += loss

            ## Calculaet weather metric
            val_score = weather_metrics(outputs,target_frames)

            ## Calculate perceptual metric
            val_score['mse'] = MSE(outputs, target_frames).detach().cpu()
            val_score['mae'] = MAE(outputs, target_frames).detach().cpu()

            for key in val_score.keys():
                val_score_accumulate[key] = val_score_accumulate[key] + val_score[key].detach().cpu() * batch_size if key in val_score_accumulate.keys() else val_score[key].detach().cpu() * batch_size

            ## tqdm update
            pbar.set_description(f"Valid >> Epoch: [{epoch+1}/{total_epoch}] Iter: [{iter+1}/{total_iters}] Loss: {loss:.4f}({sum_loss / (iter+1):.4f})", refresh=False)
        dist.barrier()

    avg_loss = sum_loss / (iter+1)
    for key in val_score.keys():
        val_score_accumulate[key] /= len(dataloader.dataset)

    ## Neptune log
    if rank == 0:
        npt["val/avg_loss"].log(avg_loss)

        print(f"Validation done. Epoch: {epoch}, Avg loss: {avg_loss:.4f}")

    return avg_loss


def main_worker(rank, args):
    '''
        node 개수: device 개수 = 1
        world size: 전체 gpu 개수
        rank: process id = gpu id (process당 gpu 1개)
    '''

    if rank == 0:
        npt = neptune.init(
        project="kaist-cilab/Nowcasting",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTQ2MGY0Yi0zMTM2LTQ5ZmEtYjlmOS1lNmQxMTliOTE0MjkifQ==",
        )

    torch.cuda.set_device(rank)
    ngpus = args.n_gpus
    world_size = ngpus * 1      # Number of nodes = 1
    batch_size = int(args.batch_size / ngpus)
    n_workers = args.n_workers

    if args.load:
        now = args.load
    else:
        now = time.strftime(r'%Y-%m-%d_%H-%M',time.localtime(time.time()))

    if rank == 0:
        model_save_path = os.path.join('trained', now) + '_' + args.model
        os.makedirs(model_save_path, exist_ok=True) 

    ## Process group initialization for DDP
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:2616',
        world_size=world_size,
        rank=rank
    )
    dist.barrier()

    ## Dataloader initialization
    dataloader_train, dataloader_val, renorm_transform = get_dataloader(data_set_name='KTPW', batch_size=batch_size, data_set_dir=args.data_path, past_frames=args.input_frames, future_frames=args.output_frames, ngpus=ngpus, num_workers=n_workers)

    ## Network model initialization
    if args.model == 'encdec':
        model = EncDec(args.input_frames, args.output_frames)
    elif args.model == 'unet':
        model = unet(args.input_frames, args.output_frames)
    elif args.model == 'tunet':
        model = Tunet(args.input_frames, args.output_frames)
    if torch.cuda.is_available():
        model = model.cuda()
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    ## Optimizer initialization
    parameters = model.parameters()
    lr = args.lr
    weight_decay = args.wd
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    ## Neptune log
    if rank == 0:
        params = {"lr": lr, "optimizer": "Adam", "weight_decay": weight_decay}
        npt["parameters"] = params

    ## Model load
    if args.load: 
        last_checkpoint_path = glob(os.path.join('trained', now, '*'))[-1]
        checkpoint = torch.load(last_checkpoint_path)

        epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])

    ## Training & validation
    best_loss = float("inf")
    for epoch in range(args.total_epoch):
        ## Training
        if rank != 0:
            npt = None
        train(args, model, dataloader_train, optimizer, epoch, args.total_epoch, rank, npt)

        ## Validation
        if rank != 0:
            npt = None
        avg_loss = val(args, model, dataloader_val, epoch, args.total_epoch, rank, npt)
        
        ## Save the best model parameters
        # if avg_loss < best_loss and rank == 0:
        if rank == 0:
            best_loss = avg_loss
            # print("Best Loss: {:.4f}".format(best_loss))
            file_name = '{:03d}_{:.4f}.ckpt'.format(epoch, best_loss)
            
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict()
                },
                os.path.join(model_save_path, file_name)
            )

            ## Remove previous saved model
            # model_list = list(os.listdir(model_save_path))
            # model_list.sort()
            # if len(model_list) > 2:
            #     model_name = model_list[0]
            #     cmd = 'rm {}'.format(os.path.join(model_save_path, model_name))
            #     os.system(cmd)

    if rank == 0:
        npt.stop()

    return


def seed_all(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_all(args.seed)
    mp.spawn(main_worker, nprocs=args.n_gpus, args=(args, ))
    return
    

if __name__ == "__main__":
    args = get_args()
    main(args)

