from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    DTD,
    GTSRB,
    STL10,
    SVHN,
    FashionMNIST,
    ImageFolder,
)

from tmcl.datasets.aircraft import Aircraft
from tmcl.datasets.cu_birds import CUBirds
from tmcl.datasets.eurosat import EuroSAT
from tmcl.datasets.relabel_dataset import RelabelDataset
from tmcl.datasets.traffic_sign import TrafficSign
from tmcl.datasets.vgg_flower import VGGFlower


def get_dataset_labels(dataset: Dataset) -> list[int]:
    match dataset:
        case RelabelDataset():
            dataset: RelabelDataset
            return dataset.labels
        case ConcatDataset():
            dataset: ConcatDataset
            return [label for d in dataset.datasets for label in get_dataset_labels(d)]
        case Subset():
            dataset: Subset
            underlying_dataset_labels = get_dataset_labels(dataset.dataset)
            indices = dataset.indices
            return [underlying_dataset_labels[i] for i in indices]
        case GTSRB():
            dataset: GTSRB
            labels = [label for _, label in dataset._samples]
            if dataset.target_transform is not None:
                return [dataset.target_transform(label) for label in labels]
            return labels
        case STL10():
            dataset: STL10
            if dataset.target_transform is not None:
                return [dataset.target_transform(label) for label in dataset.labels]
            return dataset.labels
        case DTD():
            dataset: DTD
            if dataset.target_transform is not None:
                return [dataset.target_transform(label) for label in dataset._labels]
            return dataset._labels
        case SVHN() | CUBirds() | VGGFlower() | Aircraft() | TrafficSign():
            dataset: SVHN | CUBirds | VGGFlower | Aircraft | TrafficSign
            if dataset.target_transform is not None:
                return [dataset.target_transform(label) for label in dataset.labels]
            return dataset.labels
        case CIFAR10() | CIFAR100() | FashionMNIST() | EuroSAT() | ImageFolder():
            dataset: CIFAR10 | CIFAR100 | FashionMNIST | EuroSAT | ImageFolder
            if dataset.target_transform is not None:
                return [dataset.target_transform(label) for label in dataset.targets]
            return dataset.targets
        case _:
            raise ValueError(f'Unsupported dataset {dataset.__class__.__name__}')
