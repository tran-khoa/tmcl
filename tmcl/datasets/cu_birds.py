# copied from https://github.com/wgcban/mix-bt/blob/main/transfer_datasets/cu_birds.py
import os
from collections.abc import Callable
from pathlib import Path

import torch.utils.data as data
from PIL import Image


class CUBirds(data.Dataset):
    def __init__(
        self,
        root: Path | str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__()
        self.root = os.path.join(root, 'CUB_200_2011')
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def load_images(self):
        image_info_path = os.path.join(self.root, 'images.txt')
        with open(image_info_path) as f:
            image_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        image_info = dict(image_info)

        # load image to label information
        label_info_path = os.path.join(self.root, 'image_class_labels.txt')
        with open(label_info_path) as f:
            label_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        label_info = dict(label_info)

        # load train test split
        train_test_info_path = os.path.join(self.root, 'train_test_split.txt')
        with open(train_test_info_path) as f:
            train_test_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        train_test_info = dict(train_test_info)

        all_paths, all_labels = [], []
        for index, image_path in image_info.items():
            label = int(label_info[index]) - 1
            split = int(train_test_info[index])
            if self.train:
                if split == 1:
                    all_paths.append(image_path)
                    all_labels.append(label)
            else:
                if split == 0:
                    all_paths.append(image_path)
                    all_labels.append(label)
        return all_paths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root, 'images', self.paths[index])
        label = self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
