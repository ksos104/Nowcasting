from numpy.core.fromnumeric import clip, searchsorted
import torch
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch import Tensor

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple,List
import os
from tqdm import tqdm
import random

import cv2

def get_dataloader(data_set_name, batch_size, data_set_dir, past_frames = 10, future_frames = 10, ngpus = 1, num_workers = 1):
    if data_set_name == 'KTPW':
        dataset_dir = Path(data_set_dir)
        renorm_transform = VidReNormalize(mean = 0., std = 1.0)
        train_transform = VidToTensor() #transforms.Compose([VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor()])
        test_transform = VidToTensor()
        train_set = KTPWDataset(dataset_dir.joinpath('train'), train_transform,past_frames,future_frames)
        val_set = KTPWDataset(dataset_dir.joinpath('val'),test_transform, past_frames,future_frames)

    N = batch_size
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last = True)
    val_loader = DataLoader(val_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last = True)

    if ngpus > 1:
        N = batch_size//ngpus
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

        train_loader = DataLoader(train_set, batch_size=N, shuffle=False, pin_memory=True, num_workers=num_workers, sampler=train_sampler, drop_last = True)
        val_loader = DataLoader(val_set, batch_size=N, shuffle=False, pin_memory=True, num_workers=num_workers, sampler=val_sampler, drop_last = True)

    return train_loader, val_loader, renorm_transform

class KTPWDataset(Dataset):
    def __init__(self, data_path, transform,
                 num_past_frames=13, num_future_frames=12):
        """
        Args:
            data_path --- data folder path
            transfrom --- torchvision transforms for the image
            num_past_frames
            num_future_frames
        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
        """
        self.data_path = data_path
        self.files = list(self.data_path.rglob('*.npy'))
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.transform = transform
        
        if not self.files:
            raise Exception(f"No video found in {self.data_path}")
        print(f"Found {len(self.files)} {str(self.data_path).split('/')[-1]} KTPW videos.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            past_clip: Tensor with shape (num_past_frames, C, H, W)
            future_clip: Tensor with shape (num_future_frames, C, H, W)
        """
        vid_path = str(self.files[index])
        full_clip = torch.from_numpy(np.load(vid_path)).int()
        
        imgs = []
        for i in range(full_clip.shape[0]):
            img = transforms.ToPILImage()(full_clip[i])
            imgs.append(img)
        #imgs[0].save('full_clip.gif', save_all = True, append_images = imgs[1:]) # plotting full video
        full_clip = self.transform(imgs)
        
        past_clip = full_clip[0:num_past_frames, ...]
        future_clip = full_clip[num_past_frames:, ...]  
        
        return past_clip, future_clip


class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Resize(*self.args, **self.resize_kwargs)(clip[i])

        return clip

class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.CenterCrop(*self.args, **self.kwargs)(clip[i])

        return clip

class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.functional.crop(clip[i], *self.args, **self.kwargs)

        return clip
        
class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip

class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.vflip(clip[i])
        return clip

class VidToTensor(object):
    def __call__(self, clip: List[Image.Image]):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim = 0)

        return clip

class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = transforms.Normalize(self.mean, self.std)(clip[i, ...])

        return clip

class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0/s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.],
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = [1., 1., 1.])])
        except TypeError:
            #try normalize for grey_scale images.
            self.inv_std = 1.0/std
            self.inv_mean = -mean
            self.renorm = transforms.Compose([transforms.Normalize(mean = 0.,
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = 1.)])

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

        return clip

class VidPad(object):
    """
    If pad, Do not forget to pass the mask to the transformer encoder.
    """
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Pad(*self.args, **self.kwargs)(clip[i])

        return clip

def mean_std_compute(dataset, device, color_mode = 'RGB'):
    """
    arguments:
        dataset: pytorch dataloader
        device: torch.device('cuda:0') or torch.device('cpu') for computation
    return:
        mean and std of each image channel.
        std = sqrt(E(x^2) - (E(X))^2)
    """
    data_iter= iter(dataset)
    sum_img = None
    square_sum_img = None
    N = 0

    pgbar = tqdm(desc = 'summarizing...', total = len(dataset))
    for idx, sample in enumerate(data_iter):
        past, future = sample
        clip = torch.cat([past, future], dim = 0)
        N += clip.shape[0]
        img = torch.sum(clip, axis = 0)

        if idx == 0:
            sum_img = img
            square_sum_img = torch.square(img)
            sum_img = sum_img.to(torch.device(device))
            square_sum_img = square_sum_img.to(torch.device(device))
        else:
            img = img.to(device)
            sum_img = sum_img + img
            square_sum_img = square_sum_img + torch.square(img)
        
        pgbar.update(1)
    
    pgbar.close()

    mean_img = sum_img/N
    mean_square_img = square_sum_img/N
    if color_mode == 'RGB':
        mean_r, mean_g, mean_b = torch.mean(mean_img[0, :, :]), torch.mean(mean_img[1, :, :]), torch.mean(mean_img[2, :, :])
        mean_r2, mean_g2, mean_b2 = torch.mean(mean_square_img[0,:,:]), torch.mean(mean_square_img[1,:,:]), torch.mean(mean_square_img[2,:,:])
        std_r, std_g, std_b = torch.sqrt(mean_r2 - torch.square(mean_r)), torch.sqrt(mean_g2 - torch.square(mean_g)), torch.sqrt(mean_b2 - torch.square(mean_b))

        return ([mean_r.cpu().numpy(), mean_g.data.cpu().numpy(), mean_b.cpu().numpy()], [std_r.cpu().numpy(), std_g.cpu().numpy(), std_b.cpu().numpy()])
    else:
        mean = torch.mean(mean_img)
        mean_2 = torch.mean(mean_square_img)
        std = torch.sqrt(mean_2 - torch.square(mean))

        return (mean.cpu().numpy(), std.cpu().numpy())
    
def visualize_clip(clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        print('start visualize sample clip')
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i])
            imgs.append(img)
        imgs[0].save(str(Path(file_name)), save_all = True, append_images = imgs[1:])

if __name__ == '__main__':
    dataset = 'KTPW' #see utils.dataset
    root = './data/kTPW'
    num_past_frames, num_future_frames = 13,12
    train_loader, val_loader, renorm_transform = get_dataloader(dataset, 1, root, num_past_frames, num_future_frames)
    past_clip, future_clip = next(iter(train_loader))
    visualize_clip(past_clip[0], './past_clip.gif')
    visualize_clip(future_clip[0], 'future_clip.gif')
    print(past_clip.shape, future_clip.shape) #(B, T, C, H, W)
    