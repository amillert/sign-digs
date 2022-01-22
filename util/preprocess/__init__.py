import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Grayscale, Resize, ToTensor, Compose
from PIL import Image

import util


class CustomImageDataset(Dataset):
    TensorList = List[torch.Tensor]

    def __init__(self, path: str) -> None:
        self.transform = Compose([Grayscale(3), Resize(64), ToTensor()])
        self.images, self.labels = self.load(path)

        assert len(self.images) == len(self.labels), "wrong size"
        self.len = len(self.labels)

    def load(self, path: str) -> Tuple[TensorList, TensorList]:
        images, labels = [], []

        for digit in range(10):
            dir = util.join(path, str(digit))
            imgs_paths = [util.join(dir, x) for x in os.listdir(dir)]
            labels.extend([digit] * len(imgs_paths))

            for digit_img_path in imgs_paths:
                img = self.transform(Image.open(digit_img_path))
                images.append(img)

        return images, labels

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
