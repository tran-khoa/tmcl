import kornia.augmentation as K
import torch
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample
from torch import nn

from tmcl.datasets.hints import DatasetName
from tmcl.datasets.transforms import CIFAR_STATS


def build_supervised_transform(dataset: DatasetName) -> AugmentationSequential:
    match dataset:
        case (
            'c100'
            | 'eurosat'
            | 'gtsrb'
            | 'cifar10'
            | 'svhn'
            | 'dtd'
            | 'cubirds'
            | 'vggflower'
            | 'aircraft'
            | 'trafficsigns'
        ):
            return cifar_supervised_train_transform()
        case _:
            raise ValueError(f'Unknown dataset: {dataset}')


def build_eval_transform(dataset: DatasetName) -> nn.Module:
    match dataset:
        case (
            'c100'
            | 'eurosat'
            | 'gtsrb'
            | 'cifar10'
            | 'svhn'
            | 'dtd'
            | 'cubirds'
            | 'vggflower'
            | 'aircraft'
            | 'trafficsigns'
        ):
            return cifar10_supervised_eval_transform()
        case 'stl10':
            return stl10_eval_train_transform()
        case _:
            raise ValueError(f'Unknown dataset: {dataset}')


def stl10_eval_train_transform():
    return AugmentationSequential(
        K.Normalize(
            mean=torch.tensor((0.43, 0.42, 0.39)),
            std=torch.tensor((0.27, 0.26, 0.27)),
        ),
        same_on_batch=False,
    )


def cifar_supervised_train_transform() -> AugmentationSequential:
    return AugmentationSequential(
        K.RandomCrop(size=(32, 32), padding=(4, 4), resample=Resample.BICUBIC),
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(
            mean=torch.tensor(CIFAR_STATS.mean),
            std=torch.tensor(CIFAR_STATS.std),
        ),
        same_on_batch=False,
    )


def cifar10_supervised_eval_transform() -> nn.Module:
    return K.Normalize(
        mean=torch.tensor(CIFAR_STATS.mean),
        std=torch.tensor(CIFAR_STATS.std),
    )
