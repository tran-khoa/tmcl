from typing import Any, Never

from torch.utils.data import Dataset

# def subset_labeled_dataset(dataset: Dataset, frac: float, seed: int) -> Dataset:
#     if frac >= 1.0:
#         return dataset
#
#     labels = get_dataset_labels(dataset)
#     indices, _ = train_test_split(
#         list(range(len(dataset))),
#         train_size=frac,
#         random_state=seed,
#         shuffle=True,
#         stratify=labels,
#     )
#     return Subset(dataset, indices)


class EmptyDataset(Dataset):
    """
    An empty dataset that cannot be indexed.
    """

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: Any) -> Never:
        raise IndexError('Empty dataset cannot be indexed')
