from typing import Literal

from typing_extensions import TypeIs

type DatasetName = Literal[
    'c100',
    'imagenet100',
    'mit-states',
    'ut-zap50k',
    'stl10',
    'cifar10',
    'gtsrb',
    'eurosat',
    'svhn',
    'dtd',
    'cubirds',
    'vggflower',
    'aircraft',
    'trafficsigns',
]
type Incrementality = Literal['class', 'task', 'data', 'offline']


def is_dataset_name(name: str) -> TypeIs[DatasetName]:
    return name in {'c100', 'imagenet100'}


def is_incrementality(incrementality: str) -> TypeIs[Incrementality]:
    return incrementality in {'class', 'task', 'offline'}
