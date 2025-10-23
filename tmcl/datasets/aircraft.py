# from: https://raw.githubusercontent.com/wgcban/mix-bt/refs/heads/main/transfer_datasets/aircraft.py

import os
from collections import defaultdict
from collections.abc import Callable
from os.path import join
from pathlib import Path

import torch.utils.data as data
from PIL import Image

# url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
# wget http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
# python
# from torchvision.datasets.utils import extract_archive
# extract_archive("fgvc-aircraft-2013b.tar.gz")
# Download and preprocess: https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/aircraft.py

# class_types = ('variant', 'family', 'manufacturer')
# splits = ('train', 'val', 'trainval', 'test')
# img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')


class Aircraft(data.Dataset):
    def __init__(
        self,
        root: Path | str,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        super().__init__()
        self.root = os.path.join(root, 'fgvc-aircraft-2013b')
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        paths, bboxes, labels = self.load_images()
        self.paths = paths
        self.bboxes = bboxes
        self.labels = labels

    def load_images(self):
        split = 'trainval' if self.train else 'test'
        variant_path = os.path.join(self.root, 'data', f'images_variant_{split}.txt')
        with open(variant_path) as f:
            names_to_variants = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        names_to_variants = dict(names_to_variants)
        variants_to_names = defaultdict(list)
        for name, variant in names_to_variants.items():
            variants_to_names[variant].append(name)
        variants = sorted(list(set(variants_to_names.keys())))

        names_to_bboxes = self.get_bounding_boxes()
        split_files, split_labels, split_bboxes = [], [], []
        for variant_id, variant in enumerate(variants):
            class_files = [
                join(self.root, 'data', 'images', f'{filename}.jpg')
                for filename in sorted(variants_to_names[variant])
            ]
            bboxes = [names_to_bboxes[name] for name in sorted(variants_to_names[variant])]
            labels = list([variant_id] * len(class_files))
            split_files += class_files
            split_labels += labels
            split_bboxes += bboxes
        return split_files, split_bboxes, split_labels

    def get_bounding_boxes(self):
        bboxes_path = os.path.join(self.root, 'data', 'images_box.txt')
        with open(bboxes_path) as f:
            names_to_bboxes = [line.split('\n')[0].split(' ') for line in f.readlines()]
            names_to_bboxes = dict(
                (name, list(map(int, (xmin, ymin, xmax, ymax))))
                for name, xmin, ymin, xmax, ymax in names_to_bboxes
            )
        return names_to_bboxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        bbox = tuple(self.bboxes[index])
        label = self.labels[index]

        image = Image.open(path).convert(mode='RGB')
        image = image.crop(bbox)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
