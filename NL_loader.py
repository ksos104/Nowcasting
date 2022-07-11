import torch
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import h5py


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           num_input_images,
                           num_output_images,
                           augment,
                           classification,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # define transforms
    valid_transform = None
    # valid_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])

    if augment:
        # TODO flipping, rotating, sequence flipping (torch.flip seems very expensive)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
        ])
    else:
        train_transform = None
        # train_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        # load the dataset
    train_dataset = precipitation_maps_h5(
            in_file=data_dir, num_input_images=num_input_images,
            num_output_images=num_output_images, train=True,
            transform=train_transform
    )

    valid_dataset = precipitation_maps_h5(
            in_file=data_dir, num_input_images=num_input_images,
            num_output_images=num_output_images, train=True,
            transform=valid_transform
    )


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, valid_loader


def get_test_loader(data_dir,
                    batch_size,
                    num_input_images,
                    num_output_images,
                    classification,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False):
    # Since I am not dealing with RGB images I do not need this
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )

    # define transform
    transform = None
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # normalize,
    # ])
    dataset = precipitation_maps_h5(
            in_file=data_dir, num_input_images=num_input_images,
            num_output_images=num_output_images, train=False,
            transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader


class precipitation_maps_h5(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_h5, self).__init__()

        self.file_name = in_file
        self.n_images, self.num_total, self.nx, self.ny = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape


        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024**3)["train" if self.train else "test"]['images']
        imgs = np.array(self.dataset[index], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-self.num_output:]

        return input_img, target_img

    def __len__(self):

        return self.size_dataset


"""
if __name__ == "__main__":
    folder = "/mnt/server11_hard4/jiny/Nowcasting/NL_dataset/"
    data = "interval_10min_input-length_13_output_length_12_rain-threshold_50.h5" ##interval_&&min_input-length_13_output_length_12_rain-threshold_&&.h5 몇분간격인지, threshold 몇인지##
    train_dl, valid_dl = get_train_valid_loader(folder + data,
                                                batch_size=8,
                                                random_seed=0,
                                                num_input_images=13,
                                                num_output_images=12,
                                                classification=True,
                                                augment=False,
                                                valid_size=0.1,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=False)
    for xb, yb in train_dl:
        print("xb.shape: ", xb.shape)
        print("yb.shape: ", yb.shape)
        break

"""