import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import random
import torch

class Dot_dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        
        x = random.randint(0, 27)
        y = random.randint(0, 27)

        image = torch.zeros((28, 28))
        image[y, x] = 1

        if self.transform:
            image = self.transform(image)
        return image, (y,x)