from typing import Any

from torch.utils.data import Dataset


class RelabelDataset(Dataset):
    """
    A dataset that relabels the original dataset.
    """

    def __init__(self, dataset: Dataset, labels: list[int]) -> None:
        self.dataset = dataset
        self.labels = labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        item, _ = self.dataset[index]
        label = self.labels[index]
        return item, label
