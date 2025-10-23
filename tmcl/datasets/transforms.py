from dataclasses import dataclass


@dataclass
class DatasetStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


CIFAR_STATS = DatasetStats(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.247, 0.243, 0.261),
)

STL10_STATS = DatasetStats(
    mean=(0.43, 0.42, 0.39),
    std=(0.247, 0.243, 0.2611),
)
