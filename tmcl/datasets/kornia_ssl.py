import kornia.augmentation as K
import torch
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample

from tmcl.datasets.hints import DatasetName
from tmcl.datasets.transforms import CIFAR_STATS


def build_ssl_transforms(dataset: DatasetName, metadata: str):
    match (dataset, metadata):
        case ('c100', 'cassle'):
            return generic_ssl_transform(
                normalize_mean=CIFAR_STATS.mean, normalize_std=CIFAR_STATS.std, size=32
            )
        case (_, 'transfer') | ('eurosat', _) | ('gtsrb', _):
            return weaker_ssl_transform(
                normalize_mean=CIFAR_STATS.mean, normalize_std=CIFAR_STATS.std, size=32
            )
        case ('stl10', 'cassle'):
            return cassle_stl10_transform()
        case ('c100', 'simclr'):
            return simclr_cifar_transform()
        case _:
            raise ValueError(f'Unsupported dataset/metadata combination: {dataset}/{metadata}')


def weaker_ssl_transform(
    *,
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
    size: int,
    min_scale: float = 0.08,
):
    return AugmentationSequential(
        K.RandomResizedCrop(
            size=(size, size),
            scale=(min_scale, 1.0),
            resample=Resample.BICUBIC,
        ),
        K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            p=0.8,
        ),
        K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(
            mean=torch.tensor(normalize_mean),
            std=torch.tensor(normalize_std),
        ),
        same_on_batch=False,
    )


def generic_ssl_transform(
    *,
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
    size: int,
    min_scale: float = 0.08,
) -> tuple[AugmentationSequential, AugmentationSequential]:
    return AugmentationSequential(
        K.RandomResizedCrop(
            size=(size, size),
            scale=(min_scale, 1.0),
            resample=Resample.BICUBIC,
        ),
        K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,  # (!)
            hue=0.1,
            p=0.8,
        ),
        K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(
            mean=torch.tensor(normalize_mean),
            std=torch.tensor(normalize_std),
        ),
        same_on_batch=False,
    ), AugmentationSequential(
        K.RandomResizedCrop(
            size=(size, size),
            scale=(min_scale, 1.0),
            resample=Resample.BICUBIC,
        ),
        K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,  # (!)
            hue=0.1,
            p=0.8,
        ),
        K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomSolarize(p=0.2, thresholds=0.0, additions=0.0),
        K.Normalize(
            mean=torch.tensor(normalize_mean),
            std=torch.tensor(normalize_std),
        ),
        same_on_batch=False,
    )


def cassle_stl10_transform() -> tuple[AugmentationSequential, AugmentationSequential]:
    return AugmentationSequential(
        K.RandomResizedCrop(
            size=(96, 96),
            scale=(0.08, 1.0),
            resample=Resample.BICUBIC,
        ),
        K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,  # (!)
            hue=0.1,
            p=0.8,
        ),
        K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(
            mean=torch.tensor((0.43, 0.42, 0.39)),
            std=torch.tensor((0.247, 0.243, 0.261)),
        ),
        same_on_batch=False,
    ), AugmentationSequential(
        K.RandomResizedCrop(
            size=(96, 96),
            scale=(0.08, 1.0),
            resample=Resample.BICUBIC,
        ),
        K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,  # (!)
            hue=0.1,
            p=0.8,
        ),
        K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomSolarize(p=0.2, thresholds=0.0, additions=0.0),
        K.Normalize(
            mean=torch.tensor((0.43, 0.42, 0.39)),
            std=torch.tensor((0.247, 0.243, 0.261)),
        ),
        same_on_batch=False,
    )


def simclr_cifar_transform(
    size: int = 32,
    min_scale: float = 0.08,
    normalize: bool = True,
) -> AugmentationSequential:
    return AugmentationSequential(
        K.RandomResizedCrop(
            size=(size, size),
            scale=(min_scale, 1.0),
            resample=Resample.BICUBIC,
        ),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.8,
        ),
        K.RandomGrayscale(p=0.2),
        *[
            K.Normalize(
                mean=torch.tensor((0.247, 0.243, 0.261)),
                std=torch.tensor((0.4914, 0.4822, 0.4465)),
            )
        ]
        if normalize
        else [],
        same_on_batch=False,
    )
