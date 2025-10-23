from pathlib import Path
from typing import Self

import torch
from jaxtyping import Integer, Shaped
from torch import Tensor
from torch.types import Device


def to_per_class_list(
    features: Shaped[Tensor, 'batch *axes'], labels: Integer[Tensor, 'batch']
) -> list[Shaped[Tensor, 'batch dim']]:
    """
    Converts a batch of features and their labels to a list of per-class features.

    Parameters
    ----------
    features: Batched tensor.
    labels: Batched integer labels.

    Returns
    -------
    A list of per-class features, sorted by class label.
    """
    unique_labels = labels.unique(sorted=True)
    return [features[labels == label] for label in unique_labels]


class Representations:
    repr_list: list[Shaped[Tensor, 'batch *axes']]
    device: Device

    @classmethod
    def load(cls, path: Path | str, device: Device = 'cpu') -> Self:
        """
        Loads the representations from the given path.

        Parameters
        ----------
        path: Path to the file where the representations will be loaded from.
        device: Device to load the representations to.

        Returns
        -------
        A Representations object.
        """
        representations = cls(device)
        representations.repr_list = torch.load(path, map_location=device)
        return representations

    def __init__(self, device: Device = 'cpu'):
        self.repr_list = []
        self.device = device

    def store(self, path: Path | str) -> None:
        """
        Stores the representations in the given path.

        Parameters
        ----------
        path: Path to the file where the representations will be stored.
        """
        torch.save(self.repr_list, path)

    @property
    def features(self) -> Shaped[Tensor, 'batch *axes']:
        return torch.cat(self.repr_list)
