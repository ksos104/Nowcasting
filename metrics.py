from sklearn.preprocessing import binarize
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union

# from zmq import POLLITEMS_DFLT
from .train_summary import load_ckpt

import numpy as np
from math import exp
import einops


def PSNR(x: Tensor, y: Tensor, data_range: Union[float, int] = 1.0) -> Tensor:
    """
    Comput the average PSNR between two batch of images.
    x: input image, Tensor with shape (N, C, H, W)
    y: input image, Tensor with shape (N, C, H, W)
    data_range: the maximum pixel value range of input images, used to normalize
                pixel values to [0,1], default is 1.0
    """

    EPS = 1e-8
    x = x/float(data_range)
    y = y/float(data_range)

    mse = torch.mean((x-y)**2, dim = (1, 2, 3))
    score = -10*torch.log10(mse + EPS)

    return torch.mean(score).item()


def MSE(y_pred : Tensor, y_true: Tensor, per_frame = False) -> Tensor:
    """
    Args:
        y_pred : Tenswor with shape (batch_size, num_future_frames, H, W)
        y_true : Tenswor with shape (batch_size, num_future_frames, H, W)
    Return :
        MAE score between (y_pred,y_true) per frames. 
    """
    y_pred = y_pred * 255
    y_true = y_true * 255
    
    mse = (y_pred - y_true).square().mean(axis = [0,2,3])
    if per_frame :
        mse = mse.mean()

    return mse

def Persistence(input_frames : Tensor) -> Tensor:
    N,T,H,W = input_frames.shape
    y_pred = input_frames[:,-1,:,:]
    y_pred = einops.repeat(y_pred, 'b m n ->b k m n', k = T-1)
    
    return y_pred

def MAE(y_pred : Tensor, y_true: Tensor, per_frame = False) -> Tensor:
    """
    Args:
        y_pred : Tenswor with shape (batch_size, num_future_frames, H, W)
        y_true : Tenswor with shape (batch_size, num_future_frames, H, W)
    Return :
        MAE score between (y_pred,y_true)
    """
    y_pred = y_pred * 255
    y_true = y_true * 255


    mae = (y_pred - y_true).abs().mean(axis = [0,2,3])
    if per_frame :
        mae= mae.mean()

    return mae

def compute_stats(y_pred : Tensor, y_true: Tensor, threshold : int):
    binarized_y_pred = torch.zeros_like(y_pred)
    binarized_y_pred[y_pred > threshold] = 1

    binarized_y_true = torch.zeros_like(y_true)
    binarized_y_true[y_true > threshold] = 1

    hits = (binarized_y_pred * binarized_y_true).sum(axis = [0,2,3]) # (prediciton = 1, truth = 1)
    misses = ((1 - binarized_y_pred) * binarized_y_true).sum(axis = [0,2,3]) # (prediciton = 0, truth = 1)
    false_alarms = (binarized_y_pred * (1-binarized_y_true)).sum(axis = [0,2,3]) # (prediciton = 1, truth = 0)
    eps = 1e-8
    POD = hits/(hits + misses + eps)
    SUCR = hits/(hits+false_alarms + eps)
    CSI = hits/(hits+misses+false_alarms + eps)
    BIAS = (hits+false_alarms)/(hits + misses + eps)

    return POD, SUCR, CSI, BIAS


def weather_metrics(y_pred: Tensor, y_true : Tensor, thresholds = [5,10,20,30,40,50,60,70,80,90,100]):
    # convert to 255 scale
    y_pred = y_pred * 255
    y_true = y_true * 255

    score = {}
    for threshold in thresholds:
        POD,SUCR,CSI,BIAS = compute_stats(y_pred,y_true,threshold)
        score['POD%d'%threshold] = POD
        score['SUCR%d'%threshold] = SUCR
        score['CSI%d'%threshold] = CSI
        score['BIAS%d'%threshold] = BIAS

    return score


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def forward(self, img1: Tensor, img2: Tensor) -> float:
        """
        img1: (N, C, H, W)
        img2: (N, C, H, W)
        Return:
            batch average ssim_index: float
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)





def pred_ave_metrics(model, data_loader, metric_func, renorm_transform, num_future_frames, ckpt = None, device = 'cuda:0'):
    if ckpt is not None:
        _, _, _, VPTR_state_dict, _, _ = load_ckpt(ckpt)
        model.load_state_dict(VPTR_state_dict)
    model = model.eval()
    ave_metric = np.zeros(num_future_frames)
    sample_num = 0

    with torch.no_grad():
        for idx, sample in enumerate(data_loader, 0):
            past_frames, future_frames = sample
            past_frames = past_frames.to(device)
            future_frames = future_frames.to(device)
            mask = None
            pred = model(past_frames,future_frames, mask)[0]

            for i in range(num_future_frames):
                pred_t = pred[:, i, ...]
                future_frames_t = future_frames[:, i, ...]

                renorm_pred = renorm_transform(pred_t)
                renorm_future_frames = renorm_transform(future_frames_t)

                m = metric_func(renorm_pred, renorm_future_frames)*pred_t.shape[0]
                ave_metric[i] += m
                
            sample_num += pred.shape[0]

    ave_metric = ave_metric / sample_num
    return ave_metric

if __name__ == '__main__':
    ssim = SSIM()
    
    random_img1 = torch.randn(4, 3, 256, 256)
    random_img2 = torch.randn(4, 3, 256, 256)
    ssim_index = ssim(random_img1, random_img2)
    print(ssim_index)
    
    import torchvision.transforms as transforms
    from PIL import Image

    img1 = transforms.ToTensor()(Image.open('./einstein.png').convert('L'))
    img1 = img1.unsqueeze(0)

    img2 = img1.clone()
    ssim_index = ssim(img1, img2)
    print(ssim_index)
    
    ssim_index = ssim(img1, torch.randn(1, 1, 256, 256))
    print(ssim_index)