import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor

from common.image import trans_square

data_transforms = Compose([
    Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST),
    ToTensor(),
])


class My_data(Dataset):
    def __init__(self, file_dir, mode):
        self.data_list = []
        file_path = os.path.join(file_dir, "train.txt" if mode == "train" else "val.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                items = line.strip().rsplit(maxsplit=1)
                if len(items) < 2:
                    continue
                self.data_list.append([items[0], int(items[1])])

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        im = Image.open(image_path)
        im = trans_square(im)
        im = im.convert("L")
        im = data_transforms(im)
        return im, torch.tensor(label)

    def __len__(self):
        return len(self.data_list)
