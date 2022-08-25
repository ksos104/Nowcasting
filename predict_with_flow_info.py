from locale import normalize
import torchvision
import pdb
from Forward_Warp import forward_warp
from utils import get_dataloader, MSE, MAE, weather_metrics, Persistence
from cmath import inf
import os
import argparse
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F
from model import EncDec

import matplotlib.pyplot as plt
import cv2
import numpy as np

import neptune.new as neptune

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates
import skimage.transform as sktf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from tqdm import tqdm



plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2


def get_intensity_diff(x1, x2, flo):
    """
    warp an image/tensor (x2) back to x1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x1.size()
    # mesh grid 

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x1.is_cuda:
        grid = grid.cuda()

    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x1, vgrid)
    mask = torch.autograd.Variable(torch.ones(x1.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    x1_from_flow = output * mask
    intense_diff = x1_from_flow - x2 # x1 픽셀값 이 매칭되는 x2에서의 값 - 실제 x2 값

    return intense_diff



def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = img.to("cpu")
            if col_idx >= 2 :
                img = F.to_pil_image(img)
                ax.imshow(np.asarray(img), **imshow_kwargs)
            else :
                ax.imshow(np.asarray(img[0]), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()




def train(args, model, epoch, optimizer, npt):
    flow_model = torchvision.models.optical_flow.raft_large("C_T_SKHT_V1")
    data_path = "/mnt/server14_hard0/seungju/dataset/SEVIR/vil/STORM"
    train_loader, val_loader = get_dataloader(data_set_name=args.dataset, batch_size=args.batch_size, data_set_dir=data_path, past_frames=args.input_frames, future_frames=args.output_frames, ngpus=1, num_workers=4,normalize = False)
    flow_model = flow_model.to('cuda')
    flow_model.eval()
    fw = forward_warp()
    model.train()
    
    for x in tqdm(train_loader):
        optimizer.zero_grad()
        x = torch.cat([x[0],x[1]],dim = 1)
        x = x.cuda().unsqueeze(2)
        training_loss = 0
        for idx in range(x.shape[1] -2):
            
            img1 = x[:,idx]
            img2 = x[:,idx+1]
            img3 = x[:,idx+2]
            # get_flow
            with torch.no_grad():
                channel_img1 = img1.repeat(1,3,1,1) # (B,C,H,W)
                channel_img2 = img2.repeat(1,3,1,1)
                reverse_flow_prediction = flow_model(channel_img2, channel_img1)[-1]#(B,2,u,v)
                intense_diff = get_intensity_diff(img1,img2,reverse_flow_prediction)
            
            out = model(img2,-reverse_flow_prediction,-intense_diff)
            loss = nn.MSELoss()(out,img3)

            training_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
        print (training_loss, flush = True)
    if args.neptune == 1:
        npt["train/loss"].log(training_loss)


def val(args, model, epoch, npt):
    flow_model = torchvision.models.optical_flow.raft_large("C_T_SKHT_V1")
    data_path = "/mnt/server14_hard0/seungju/dataset/SEVIR/vil/STORM"
    train_loader, val_loader = get_dataloader(data_set_name=args.dataset, batch_size=args.batch_size, data_set_dir=data_path, past_frames=args.input_frames, future_frames=args.output_frames, ngpus=1, num_workers=4,normalize = False)
    flow_model = flow_model.to('cuda')
    flow_model.eval()
    fw = forward_warp()
    model.eval()
    val_score = {}
    val_score_accumulate = {}
    per_score = {}
    per_score_accumulate = {}
    with torch.no_grad():
        for x in tqdm(val_loader):
            inputs = x[0].cuda().unsqueeze(2) # B,T,C,H,W
            targets = x[1].cuda().unsqueeze(2) # B,T,C,H,W
            batch_size = inputs.shape[0]
            # prediction with RAFT
            img1 = inputs[:,-2,:,:,:]
            img2 = inputs[:,-1,:,:,:]
            predictions = []
            
            for idx in (range(12)):
                channel_img1 = img1.repeat(1,3,1,1) # (B,C,H,W)
                channel_img2 = img2.repeat(1,3,1,1)
                reverse_flow_prediction = flow_model(channel_img2, channel_img1)[-1]#(B,2,u,v)
                intense_diff = get_intensity_diff(img1,img2,reverse_flow_prediction)
                
                img3 = model(img2,-reverse_flow_prediction,-intense_diff)
                img1 = img2.clone()
                img2 = img3.clone()
            
                predictions.append(img3.detach().cpu()) #(B,T,C,H,W)
            
            
            outputs = torch.stack(predictions,dim = 1).cuda()

            outputs = outputs.squeeze(2)
            targets = targets.squeeze(2)
            
            # val_score = weather_metrics(outputs,targets)
            val_score['mse'] = MSE(outputs, targets).detach().cpu()
            val_score['mae'] = MAE(outputs, targets).detach().cpu()
            for key in val_score.keys():
                val_score_accumulate[key] = val_score_accumulate[key] + val_score[key].detach().cpu() * batch_size if key in val_score_accumulate.keys() else val_score[key].detach().cpu() * batch_size
            

            ##### PER SCORE #####

            # inputs = inputs.squeeze(2)
            # outputs = Persistence(inputs)
       
            # per_score = weather_metrics(outputs, targets)

            # per_score['mse'] = MSE(outputs, targets).detach().cpu()
            # per_score['mae'] = MAE(outputs, targets).detach().cpu()
           

            # for key in per_score.keys():
                # per_score_accumulate[key] = per_score_accumulate[key] + per_score[key].detach().cpu() * args.batch_size if key in per_score_accumulate.keys() else per_score[key].detach().cpu() * batch_size

    for key in val_score.keys():
        val_score_accumulate[key] /= len(val_loader.dataset)
        # per_score_accumulate[key] /= len(val_loader.dataset)


    if args.neptune == 1:
        npt["val/MAE"].log(val_score_accumulate['mae'].mean())
        npt["val/MSE"].log(val_score_accumulate['mse'].mean())
        # npt["val/MAE_perscore"].log(per_score_accumulate['mae'].mean())
        # npt["val/MSE_perscore"].log(per_score_accumulate['mse'].mean())

def get_args():
    parser = argparse.ArgumentParser(description="FSS_msson")

    parser.add_argument('--n_gpus', type=int, help='Number of gpus you need', default=1)
    parser.add_argument('--data_path', type=str, help='Path for dataset you want to use. Absolute path is recomended')
    parser.add_argument('--batch_size', type=int, help='Total batch size', default=1)
    parser.add_argument('--n_workers', type=int, help='Number of subprocesses for data loading', default=4)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-4)
    parser.add_argument('--total_epoch', type=int, help='Number of total epoch', default=100)
    parser.add_argument('--load', type=str, help='Model path. ex) 2022-01-01', default=None)
    parser.add_argument('--seed', type=int, help='Seed for all random function', default=212)
    parser.add_argument('--input_frames', type=int, help='Number of input past frames', default=13)
    parser.add_argument('--output_frames', type=int, help='Number of output future frames', default=12)
    parser.add_argument('--model', type=str, help='Select model [encdec, unet, tunet, simVP]', default='unet')
    parser.add_argument('--neptune', type=int,  default= 0 )
    parser.add_argument('--tags', type=str,  default= "nowcasing" )
    parser.add_argument('--dataset', type=str,  default= "SEVIR" )
    parser.add_argument('--flow', type=str,  default= "RAFT" )
    args = parser.parse_args()

    return args  


if __name__ == "__main__":
    args = get_args()
    path = Path(os.path.realpath(__file__))
    npt = None
    if args.neptune ==1 :
        npt = neptune.init(
            project="seungjucho/Nowcasting",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZTNhYzkzNy02ODYwLTRhMjctYWQ5MC1hYWU5OTExMjc1ZTMifQ==",
            tags = [args.tags, str(args.lr)],
        ) 
    else:
        npt = {}
    
    model = EncDec().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    for epoch in range(100):
        train(args, model, epoch, optimizer, npt)
        val(args, model, epoch, npt)