from io import BytesIO

# import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import random
import torch


class MultiResolutionDataset(Dataset):
    def __init__(self, path, resolution=8):

        files = sorted(list(os.listdir(path)))
        self.imglist =[]
        for fir in files:
            self.imglist.append(os.path.join(path,fir))

        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
        )

        self.resolution = resolution


    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        img = Image.open(self.imglist[index])
        img = self.transform(img)

        return img