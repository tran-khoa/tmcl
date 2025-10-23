# copied from https://github.com/wgcban/mix-bt/blob/main/transfer_datasets/vgg_flower.py

import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image
from scipy.io import loadmat

# wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
# tar -xvzf 102flowers.tgz
# rename file to VGGFlower
# cd VGGFlower
# wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat


class VGGFlower(data.Dataset):
    def __init__(
        self,
        root: Path | str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__()
        self.root = os.path.join(root, 'VGGFlower')
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def load_images(self):
        rs = np.random.RandomState(42)
        imagelabels_path = os.path.join(self.root, 'imagelabels.mat')
        with open(imagelabels_path, 'rb') as f:
            labels = loadmat(f)['labels'][0]

        all_filepaths = defaultdict(list)
        for i, label in enumerate(labels):
            # all_filepaths[label].append(os.path.join(self.root, 'jpg', 'image_{:05d}.jpg'.format(i+1)))
            all_filepaths[label].append(os.path.join(self.root, f'image_{i + 1:05d}.jpg'))
        # train test split
        split_filepaths, split_labels = [], []
        for label, paths in all_filepaths.items():
            label = int(label) - 1
            num = len(paths)
            paths = np.array(paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            paths = paths[indexer].tolist()

            if self.train:
                paths = paths[: int(0.8 * num)]
            else:
                paths = paths[int(0.8 * num) :]

            labels = [label] * len(paths)
            split_filepaths.extend(paths)
            split_labels.extend(labels)

        return split_filepaths, split_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
