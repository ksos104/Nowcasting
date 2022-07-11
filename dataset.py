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
import os, pdb
from tqdm import tqdm
import random
from time import time

import cv2

def get_dataloader(data_set_name, batch_size, data_set_dir, past_frames = 13, future_frames = 12, ngpus = 1, num_workers = 4, eval_mode = False, split='full', normalize = False):
    if data_set_name == 'KTPW':
        dataset_dir = Path(data_set_dir)
        renorm_transform = VidReNormalize(mean = 0., std = 1.0)        
        train_transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.ToTensor()])
        print("eval_mode : ",eval_mode)
        if eval_mode == False:
            train_set = KTPWDataset(dataset_dir, train_transform,past_frames,future_frames,split='train')
        val_set = KTPWDataset(dataset_dir,test_transform, past_frames,future_frames,split)


    elif data_set_name == 'SEVIR':
        '''
        current_path = 
                    "/mnt/server14_hard0/seungju/dataset/SEVIR/vil/STORM",
                    "/mnt/server14_hard0/seungju/dataset/SEVIR/vil/RANDOM",

        '''


        dataset_dir = Path(data_set_dir)
        if normalize :
            MEAN = 33.44/255
            SCALE= 47.54/255
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MEAN), (SCALE))
        ])
        else :
            transform = transforms.Compose([
            transforms.ToTensor()
        ])
        print("eval_mode : ",eval_mode)
        if eval_mode == False:
            train_set = SEVIRDataset(dataset_dir,transform, past_frames,future_frames,split='train')
        val_set = SEVIRDataset(dataset_dir,transform, past_frames,future_frames,split='val')

    N = batch_size
    if eval_mode == False:
        train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last = True)
    else:
        train_loader = None
    val_loader = DataLoader(val_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last = True)

    if ngpus > 1:
        # N = batch_size//ngpus
        if eval_mode == False:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = DataLoader(train_set, batch_size=N, shuffle=False, pin_memory=True, num_workers=num_workers, sampler=train_sampler, drop_last = True)
        else:
            train_loader = None
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        val_loader = DataLoader(val_set, batch_size=N, shuffle=False, pin_memory=True, num_workers=num_workers, sampler=val_sampler, drop_last = True)

    return train_loader, val_loader

class SEVIRDataset(Dataset):
    def __init__(self, data_path, transform,
                 num_past_frames=13, num_future_frames=12,split='train'):
        """
        Args:
            data_path --- data folder path
            transfrom --- torchvision transforms for the image
            num_past_frames
            num_future_frames
        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
            division --- 'STROM' or 'RANDOM' currently. 
        """
        self.data_path = os.path.join(data_path, split)
        self.img_names = [os.path.join(self.data_path,f) for f in os.listdir(self.data_path)]
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.transform = transform
        
        if not self.img_names:
            raise Exception(f"No data found in {self.data_path}")
        print(f"Found {len(self.img_names)} SEVIR sequences.")


    def __len__(self) -> int:
        return len(self.img_names)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            past_clip: Tensor with shape (num_past_frames, C, H, W)
            future_clip: Tensor with shape (num_future_frames, C, H, W)
        """
        vid_path = self.img_names[index]
        full_clip = np.load(vid_path)
        
        if self.transform:
            full_clip = self.transform(full_clip)

        past_clip = full_clip[:self.num_past_frames, ...]
        future_clip = full_clip[self.num_past_frames:, ...]

        return past_clip, future_clip


class KTPWDataset(Dataset):
    def __init__(self, data_path, transform,
                 num_past_frames=13, num_future_frames=12,split='full'):
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
        self.split=split
        self.files = []
        self.imgId=[i_id.strip() for i_id in open(data_path.joinpath(f'{split}.txt'))]
        self.jpath='train' if self.split == 'train' else 'val'
        self.files = [data_path.joinpath(self.jpath).joinpath(f'{name}.npy') for name in self.imgId]        
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.transform = transform
        
        if not self.files:
            raise Exception(f"No video found in {self.data_path}")
        print(f"Found {len(self.imgId)} KTPW videos.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            past_clip: Tensor with shape (num_past_frames, C, H, W)
            future_clip: Tensor with shape (num_future_frames, C, H, W)
        """
        vid_path = str(self.files[index])
        # full_clip = torch.from_numpy(np.load(vid_path)).int().float()  #
        full_clip = (torch.from_numpy(np.load(vid_path)).int().float())/255
        
        imgs = []
        for i in range(full_clip.shape[0]):
            img = self.transform(full_clip[i])
            imgs.append(img)
        #imgs[0].save('full_clip.gif', save_all = True, append_images = imgs[1:]) # plotting full video
        #full_clip = self.transform(imgs)
        full_clip = torch.stack(imgs, dim = 0)
        past_clip = full_clip[0:self.num_past_frames, ...]
        future_clip = full_clip[self.num_past_frames:, ...]  
        end = time()
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
    root = 'data/kTPW_sample'
    num_past_frames, num_future_frames = 13,12
    split='easy'
    train_loader, val_loader, renorm_transform = get_dataloader(dataset, 1, root, num_past_frames, num_future_frames,eval_mode=True,split=split)
    train_loader, val_loader, renorm_transform = get_dataloader(dataset, 1, root, num_past_frames, num_future_frames)
    past_clip, future_clip = next(iter(val_loader))
    visualize_clip(past_clip[0], './past_clip.gif')
    visualize_clip(future_clip[0], 'future_clip.gif')
    #mean, std = mean_std_compute(train_loader, torch.device('cuda:0'))
    #mean, std = mean_std_compute(val_loader, torch.device('cuda:0'))
    #print(past_clip.shape, future_clip.shape) #(B, T, C, H, W)
    