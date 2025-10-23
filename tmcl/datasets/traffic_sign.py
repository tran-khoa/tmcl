import os
from collections.abc import Callable
from glob import glob
from os.path import join
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image

DATA_ROOTS = 'data/TrafficSign'

# wget https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip
# unzip FullIJCNN2013.zip


class TrafficSign(data.Dataset):
    NUM_CLASSES = 43

    def __init__(
        self,
        root: Path | str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__()
        self.root = os.path.join(root, 'FullIJCNN2013')
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def load_images(self):
        rs = np.random.RandomState(42)
        all_filepaths, all_labels = [], []
        for class_i in range(self.NUM_CLASSES):
            # class_dir_i = join(self.root, split, 'Images', '{:05d}'.format(class_i))
            class_dir_i = join(self.root, f'{class_i:02d}')
            image_paths = glob(join(class_dir_i, '*.ppm'))
            # train test splitting
            image_paths = np.array(image_paths)
            num = len(image_paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            image_paths = image_paths[indexer].tolist()
            if self.train:
                image_paths = image_paths[: int(0.8 * num)]
            else:
                image_paths = image_paths[int(0.8 * num) :]
            labels = [class_i] * len(image_paths)
            all_filepaths.extend(image_paths)
            all_labels.extend(labels)

        return all_filepaths, all_labels

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
