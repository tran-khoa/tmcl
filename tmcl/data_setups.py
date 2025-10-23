import logging
import os
import random
import tarfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Final, Literal, TypedDict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, DTD, GTSRB, STL10, SVHN, ImageFolder
from torchvision.transforms import InterpolationMode, v2

from tmcl.config_tmcl import Config
from tmcl.data import Session
from tmcl.datasets.aircraft import Aircraft
from tmcl.datasets.cu_birds import CUBirds
from tmcl.datasets.dataset import EmptyDataset
from tmcl.datasets.eurosat import EuroSAT
from tmcl.datasets.hints import DatasetName, Incrementality, is_incrementality
from tmcl.datasets.labels import get_dataset_labels
from tmcl.datasets.relabel_dataset import RelabelDataset
from tmcl.datasets.traffic_sign import TrafficSign
from tmcl.datasets.vgg_flower import VGGFlower
from tmcl.phase import Phase

STANDARD_TRANSFORM: v2.Transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def downscale_transform(image_size: int | tuple[int, int]) -> v2.Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


# adapted from https://github.com/NeuralCollapseApplications/UniCIL/blob/main/mmcil/datasets/cifar100_cil.py#L167
# and https://github.com/Yujun-Shi/CwD/blob/c085e56550e4a60602168ec22361295cab94da36/src/datasets/dataset_config.py#L37
# fmt: off
CIFAR100_CLASS_ORDER: Final[tuple[int, ...]] = (
    68, 56, 78, 8, 23,
    84, 90, 65, 74, 76,
    40, 89, 3, 92, 55,
    9, 26, 80, 43, 38,
    58, 70, 77, 1, 85,
    19, 17, 50, 28, 53,
    13, 81, 45, 82, 6,
    59, 83, 16, 15, 44,
    91, 41, 72, 60, 79,
    52, 20, 10, 31, 54,  # end of pretrain-session
    37, 95, 14, 71, 96,
    98, 97, 2, 64, 66,
    42, 22, 35, 86, 24,
    34, 87, 21, 99, 0,
    88, 27, 18, 94, 11,
    12, 47, 25, 30, 46,
    62, 69, 36, 61, 7,
    63, 75, 5, 32, 4,
    51, 48, 73, 93, 39,
    67, 29, 49, 57, 33
)
# fmt: on


class SplitDatasetDict(TypedDict):
    train: Dataset
    train_full: Dataset
    valid: Dataset
    test: Dataset


@dataclass
class Data:
    dataset: DatasetName
    classes_per_dataset: dict[str, int]
    image_size: int
    patch_size: int
    incrementality: Incrementality

    eval_datasets: dict[str, SplitDatasetDict]
    sessions: Sequence[Session]

    custom_num_classes: int | None = None
    custom_eval_sessions: tuple[tuple[int, ...], ...] | None = None

    @property
    def num_classes(self):
        return (
            sum(self.classes_per_dataset.values())
            if self.custom_num_classes is None
            else self.custom_num_classes
        )

    @property
    def eval_sessions(self) -> tuple[tuple[int, ...], ...]:
        if self.custom_eval_sessions is not None:
            return self.custom_eval_sessions
        return tuple([tuple(s.current_classes) for s in self.sessions])


def get_class_dataset(dataset: Dataset, class_idx: int) -> Subset:
    return Subset(dataset, [i for i, t in enumerate(get_dataset_labels(dataset)) if t == class_idx])


def prepare_data(cfg: Config) -> Data:
    # split: {main_dataset}/{class}/metadata
    match cfg.setup.split('/'):
        case ['c100', incrementality, metadata]:
            assert is_incrementality(incrementality)
            return build_cifar100(cfg, incrementality, metadata)
        case ['stl10', incrementality, metadata]:
            assert is_incrementality(incrementality)
            return build_stl10(cfg, incrementality, metadata)
        case ['imagenet100', incrementality, metadata]:
            assert is_incrementality(incrementality)
            return build_imagenet100(cfg, incrementality, metadata)
        case ['c100_transfer', target_dataset, metadata]:
            return build_cifar100_transfer(cfg, target_dataset, metadata)
        case _:
            raise ValueError(f'Unknown setup: {cfg.setup}')


def build_cifar100_class_datasets(
    data_path: Path | str,
    *,
    seed: int,
    labeled_frac: float,
    label_noise_frac: float,
    split_valid: bool,
):
    logger = logging.getLogger(__name__)

    base_train_ds = CIFAR100(str(data_path), train=True, transform=STANDARD_TRANSFORM)
    if split_valid:
        train_indices, valid_indices = train_test_split(
            range(len(base_train_ds)),
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            stratify=base_train_ds.targets,
        )
    else:
        train_indices = list(range(len(base_train_ds)))
        valid_indices = []
    train_full_ds = Subset(base_train_ds, train_indices)
    if labeled_frac < 1.0:
        train_indices, discarded_idxs = train_test_split(
            train_indices,
            train_size=labeled_frac,
            random_state=seed,
            shuffle=True,
            stratify=[base_train_ds.targets[i] for i in train_indices],
        )
        logger.info(f'Discarded {len(discarded_idxs)} samples, keeping {len(train_indices)}.')

    if label_noise_frac > 0.0:
        noisy_indices, clean_indices = train_test_split(
            train_indices,
            train_size=label_noise_frac,
            random_state=seed,
            shuffle=True,
            # unstratified on purpose
        )
        noisy_ds = Subset(base_train_ds, noisy_indices)
        new_labels = random.Random(seed).choices(range(100), k=len(noisy_ds))

        train_ds = ConcatDataset(
            [
                Subset(base_train_ds, clean_indices),
                RelabelDataset(noisy_ds, new_labels),
            ],  # relabel noisy samples
        )
    else:
        train_ds = Subset(base_train_ds, train_indices)

    valid_ds = Subset(base_train_ds, valid_indices)
    base_test_ds = CIFAR100(str(data_path), train=False, transform=STANDARD_TRANSFORM)
    datasets = {
        'train': train_ds,
        'train_full': train_full_ds,
        'valid': valid_ds,
        'test': base_test_ds,
    }
    class_datasets = [
        {
            'train': get_class_dataset(train_ds, c),
            'train_full': get_class_dataset(train_full_ds, c),
            'valid': get_class_dataset(valid_ds, c),
            'test': get_class_dataset(base_test_ds, c),
        }
        for c in range(100)
    ]
    return datasets, class_datasets


def _build_stl10_class_datasets(
    data_path: Path | str, *, seed: int, labeled_frac: float, split_valid: bool
):
    logger = logging.getLogger(__name__)

    base_train_ds = STL10(str(data_path), 'train', transform=STANDARD_TRANSFORM)
    if split_valid:
        train_indices, valid_indices = train_test_split(
            range(len(base_train_ds)),
            test_size=0.2,
            random_state=seed,
            shuffle=True,
            stratify=base_train_ds.labels,
        )
    else:
        train_indices = list(range(len(base_train_ds)))
        valid_indices = []
    train_full_ds = Subset(base_train_ds, train_indices)

    if labeled_frac < 1.0:
        train_indices, discarded_idxs = train_test_split(
            train_indices,
            train_size=labeled_frac,
            random_state=seed,
            shuffle=True,
            stratify=[base_train_ds.labels[i] for i in train_indices],
        )
        logger.info(f'Discarded {len(discarded_idxs)} samples, keeping {len(train_indices)}.')
    train_ds = Subset(base_train_ds, train_indices)
    valid_ds = Subset(base_train_ds, valid_indices)
    base_test_ds = STL10(str(data_path), split='test', transform=STANDARD_TRANSFORM)
    datasets = {
        'train': train_ds,
        'train_full': train_full_ds,
        'unlabeled': ConcatDataset(
            [STL10(str(data_path), 'unlabeled', transform=STANDARD_TRANSFORM), train_full_ds]
        ),
        'valid': valid_ds,
        'test': base_test_ds,
    }
    class_datasets = [
        {
            'train': get_class_dataset(train_ds, c),
            'train_full': get_class_dataset(train_full_ds, c),
            'valid': get_class_dataset(valid_ds, c),
            'test': get_class_dataset(base_test_ds, c),
        }
        for c in range(10)
    ]
    return datasets, class_datasets


def build_stl10(cfg: Config, incrementality: Incrementality, metadata: str) -> Data:
    logging.info(f'{incrementality}-incremental CIFAR-100 setup ({metadata}).')
    datasets, class_datasets = _build_stl10_class_datasets(
        cfg.data_path,
        seed=cfg.seed,
        labeled_frac=cfg.labeled_frac,
        split_valid=cfg.eval.eval_valid,
    )
    match (incrementality, metadata):
        case ('class', 's5'):
            rng = torch.Generator().manual_seed(cfg.seed)
            class_order = torch.randperm(10, generator=rng).tolist()
            sessions = [
                Session(
                    0,
                    epochs=(
                        (Phase.PRETRAIN, cfg.pretrain_epochs[0]),
                        (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                        (Phase.CONSOLIDATION, cfg.pretrain_epochs[1]),
                    ),
                    curr_ds={c: class_datasets[c]['train'] for c in class_order[:2]},
                    full_curr_ds={-1: datasets['unlabeled']},
                ),
                *[
                    Session(
                        i,
                        epochs=(
                            (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, cfg.incremental_epochs[1]),
                        ),
                        curr_ds={
                            c: class_datasets[c]['train'] for c in class_order[i * 2 : 2 * (i + 1)]
                        },
                        past_ds={},
                        full_curr_ds={-1: datasets['unlabeled']},
                    )
                    for i in range(1, 5)
                ],
            ]
        case ('data', 's5'):
            # data-incremental unsup. learning
            # class-incremental sup. learning
            rng = torch.Generator().manual_seed(cfg.seed)
            unsup_splits = random_split(datasets['unlabeled'], lengths=[0.2] * 5, generator=rng)

            rng = torch.Generator().manual_seed(cfg.seed)
            class_order = torch.randperm(10, generator=rng).tolist()
            sessions = [
                Session(
                    0,
                    epochs=(
                        (Phase.PRETRAIN, cfg.pretrain_epochs[0]),
                        (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                        (Phase.CONSOLIDATION, cfg.pretrain_epochs[1]),
                    ),
                    curr_ds={c: class_datasets[c]['train'] for c in class_order[:2]},
                    full_curr_ds={-1: unsup_splits[0]},
                ),
                *[
                    Session(
                        i,
                        epochs=(
                            (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, cfg.incremental_epochs[1]),
                        ),
                        curr_ds={
                            c: class_datasets[c]['train'] for c in class_order[i * 2 : 2 * (i + 1)]
                        },
                        past_ds={},
                        full_curr_ds={-1: unsup_splits[i]},
                    )
                    for i in range(1, 5)
                ],
            ]
        case _:
            raise ValueError(f'Unknown configuration {incrementality}/{metadata}')

    return Data(
        dataset='stl10',
        classes_per_dataset={'stl10': 10},
        image_size=96,
        patch_size=12,
        eval_datasets={'stl10': datasets},
        sessions=sessions,
        incrementality=incrementality,
    )


def build_cifar100(cfg: Config, incrementality: Incrementality, metadata: str) -> Data:
    logging.info(f'{incrementality}-incremental CIFAR-100 setup ({metadata}).')
    datasets, class_datasets = build_cifar100_class_datasets(
        cfg.data_path,
        seed=cfg.seed,
        labeled_frac=cfg.labeled_frac,
        split_valid=cfg.eval.eval_valid,
        label_noise_frac=cfg.class_learner.label_noise_frac,
    )
    class_order = list(CIFAR100_CLASS_ORDER)
    if cfg.reshuffle_class_order:
        Random(cfg.seed).shuffle(class_order)
        print(f'Reshuffled class order according to seed {cfg.seed}: {class_order}')
    match (incrementality, metadata):
        case ('class', 'cassle_s5@pretrain_tl'):
            # 5 sessions, seed 5
            rng = torch.Generator().manual_seed(5)
            class_order = torch.randperm(100, generator=rng).tolist()
            sessions = [
                Session(
                    0,
                    epochs=(
                        (Phase.PRETRAIN, cfg.pretrain_epochs[0]),
                        (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                        (Phase.CONSOLIDATION, cfg.pretrain_epochs[1]),
                    ),
                    curr_ds={c: class_datasets[c]['train'] for c in class_order[:20]},
                    full_curr_ds={c: class_datasets[c]['train_full'] for c in class_order[:20]},
                ),
                *[
                    Session(
                        i,
                        epochs=(
                            (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, cfg.incremental_epochs[1]),
                        ),
                        curr_ds={
                            c: class_datasets[c]['train']
                            for c in class_order[i * 20 : 20 * (i + 1)]
                        },
                        past_ds={},
                        full_curr_ds={
                            c: class_datasets[c]['train_full']
                            for c in class_order[i * 20 : 20 * (i + 1)]
                        },
                        full_past_ds={
                            c: class_datasets[c]['train_full'] for c in class_order[: i * 20]
                        },
                    )
                    for i in range(1, 5)
                ],
            ]
        case ('class', 's10@pretrain_tl'):
            # non cassle, use defined seed
            rng = torch.Generator().manual_seed(cfg.seed)
            class_order = torch.randperm(100, generator=rng).tolist()
            sessions = [
                Session(
                    0,
                    epochs=(
                        (Phase.PRETRAIN, cfg.pretrain_epochs[0]),
                        (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                        (Phase.CONSOLIDATION, cfg.pretrain_epochs[1]),
                    ),
                    curr_ds={c: class_datasets[c]['train'] for c in class_order[:10]},
                    full_curr_ds={c: class_datasets[c]['train_full'] for c in class_order[:10]},
                ),
                *[
                    Session(
                        i,
                        epochs=(
                            (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, cfg.incremental_epochs[1]),
                        ),
                        curr_ds={
                            c: class_datasets[c]['train']
                            for c in class_order[i * 10 : 10 * (i + 1)]
                        },
                        past_ds={},
                        full_curr_ds={
                            c: class_datasets[c]['train_full']
                            for c in class_order[i * 10 : 10 * (i + 1)]
                        },
                        full_past_ds={
                            c: class_datasets[c]['train_full'] for c in class_order[: i * 10]
                        },
                    )
                    for i in range(1, 10)
                ],
            ]
        case (
            ('offline', 'cassle_session_0')
            | ('offline', 'cassle_session_1')
            | ('offline', 'cassle_session_2')
            | ('offline', 'cassle_session_3')
            | ('offline', 'cassle_session_4')
        ):
            rng = torch.Generator().manual_seed(5)
            class_order = torch.randperm(100, generator=rng).tolist()
            session = int(metadata.split('_')[-1])
            sessions = [
                Session(
                    0,
                    epochs=((Phase.CONSOLIDATION, sum(cfg.pretrain_epochs)),),
                    curr_ds={
                        c: class_datasets[c]['train']
                        for c in class_order[session * 20 : 20 * (session + 1)]
                    },
                    past_ds={},
                    full_curr_ds={
                        c: class_datasets[c]['train_full']
                        for c in class_order[session * 20 : 20 * (session + 1)]
                    },
                    full_past_ds={
                        c: class_datasets[c]['train_full'] for c in class_order[: session * 20]
                    },
                )
            ]
        case ('data', 's5'):
            rng = torch.Generator().manual_seed(cfg.seed)
            unsup_datasets = torch.utils.data.random_split(
                datasets['train_full'],
                lengths=[0.2] * 5,
                generator=rng,
            )
            sup_datasets = torch.utils.data.random_split(
                datasets['train'],
                lengths=[0.2] * 5,
                generator=rng,
            )
            # per-class per-session datasets
            class_datasets = [{}, {}, {}, {}, {}]
            for session in range(5):
                for c in range(100):
                    indices = [
                        i for i, t in enumerate(get_dataset_labels(sup_datasets[session])) if t == c
                    ]
                    if not indices:
                        continue
                    class_datasets[session][c] = Subset(
                        sup_datasets[session],
                        indices,
                    )

            sessions = [
                Session(
                    0,
                    epochs=(
                        (Phase.PRETRAIN, cfg.pretrain_epochs[0]),
                        (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                        (Phase.CONSOLIDATION, cfg.pretrain_epochs[1]),
                    ),
                    curr_ds=class_datasets[0],
                    full_curr_ds={0: unsup_datasets[0]},
                ),
                *[
                    Session(
                        i,
                        epochs=(
                            (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, cfg.incremental_epochs[1]),
                        ),
                        curr_ds=class_datasets[i],
                        full_curr_ds={0: unsup_datasets[i]},
                    )
                    for i in range(1, 5)
                ],
            ]
        case _:
            raise ValueError(f'Unknown configuration {incrementality}/{metadata}')

    return Data(
        dataset='c100',
        classes_per_dataset={'cifar100': 100},
        image_size=32,
        patch_size=4,
        eval_datasets={'cifar100': datasets},
        sessions=sessions,
        incrementality=incrementality,
    )


def _build_imagenet100_class_datasets(data_path: Path | str, *, seed: int, labeled_frac: float):
    logger = logging.getLogger(__name__)

    train_path = Path('/dev/shm/datasets/imagenet100')
    finished_marker = train_path / '.finished'
    if int(os.environ.get('SLURM_LOCALID', 0)) == 0:
        # copy to RAM
        Path('/dev/shm/datasets/imagenet100').mkdir(parents=True, exist_ok=True)
        logger.info('Loading dataset into RAM...')
        with tarfile.open(data_path / 'imagenet100.tar', 'r') as tar:
            tar.extractall(train_path)
            finished_marker.touch()
        logger.info(' * Dataset loaded into RAM.')
    else:
        # wait for copying to finish
        logger.info(' * Waiting for dataset to be loaded into RAM...')
        time.sleep(15)
        while not finished_marker.exists():
            time.sleep(3)
        logger.info(' * Dataset has been loaded into RAM.')

    train_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                224,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    eval_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                224,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    base_train_ds = ImageFolder(train_path, transform=train_transforms)
    if labeled_frac < 1.0:
        train_indices, discarded_idxs = train_test_split(
            range(len(base_train_ds)),
            train_size=labeled_frac,
            random_state=seed,
            shuffle=True,
            stratify=[base_train_ds.targets[i] for i in range(len(base_train_ds))],
        )
        logger.info(f'Discarded {len(discarded_idxs)} samples, keeping {len(train_indices)}.')
        train_ds = Subset(base_train_ds, train_indices)
    else:
        train_ds = base_train_ds
    datasets = {
        'train': train_ds,
        'train_full': base_train_ds,
    }
    class_datasets = [
        {
            'train': get_class_dataset(train_ds, c),
            'train_full': get_class_dataset(base_train_ds, c),
        }
        for c in range(100)
    ]
    return datasets, class_datasets


def build_imagenet100(cfg: Config, incrementality: Incrementality, metadata: str) -> Data:
    logging.info(f'{incrementality}-incremental ImageNet-100 setup ({metadata}).')
    datasets, class_datasets = _build_imagenet100_class_datasets(
        cfg.data_path, seed=cfg.seed, labeled_frac=cfg.labeled_frac
    )
    rng = torch.Generator().manual_seed(5)
    class_order = torch.randperm(100, generator=rng).tolist()
    match (incrementality, metadata):
        case ('class', 'cassle_s5'):
            sessions = [
                Session(
                    0,
                    epochs=(
                        (Phase.PRETRAIN, cfg.pretrain_epochs[0]),
                        (Phase.CONSOLIDATION, cfg.pretrain_epochs[1]),
                    ),
                    curr_ds={c: class_datasets[c]['train'] for c in class_order[:20]},
                    full_curr_ds={c: class_datasets[c]['train_full'] for c in class_order[:20]},
                ),
                *[
                    Session(
                        i,
                        epochs=(
                            (Phase.TASK_LEARNING, cfg.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, cfg.incremental_epochs[1]),
                        ),
                        curr_ds={
                            c: class_datasets[c]['train']
                            for c in class_order[i * 20 : 20 * (i + 1)]
                        },
                        past_ds={},
                        full_curr_ds={
                            c: class_datasets[c]['train_full']
                            for c in class_order[i * 20 : 20 * (i + 1)]
                        },
                        full_past_ds={
                            c: class_datasets[c]['train_full'] for c in class_order[: i * 20]
                        },
                    )
                    for i in range(1, 5)
                ],
            ]
        case _:
            raise ValueError(f'Unknown configuration {incrementality}/{metadata}')
    return Data(
        dataset='imagenet100',
        classes_per_dataset={'imagenet100': 100},
        image_size=224,
        patch_size=16,
        eval_datasets={},
        sessions=sessions,
        incrementality=incrementality,
    )


def build_cifar100_transfer(
    cfg: Config,
    target_dataset: Literal['cifar10', 'stl10', 'eurosat', 'gtsrb', 'svhn'],
    metadata: str,
):
    flags = metadata.split('_')

    match target_dataset:
        case 'cifar10':
            train_ds = CIFAR10(
                cfg.data_path,
                train=True,
                transform=STANDARD_TRANSFORM,
                target_transform=lambda idx: idx + 100,
            )
            test_ds = CIFAR10(
                cfg.data_path,
                train=False,
                transform=STANDARD_TRANSFORM,
                target_transform=lambda idx: idx + 100,
            )
        case 'eurosat':
            train_ds = EuroSAT(
                cfg.data_path,
                split='train',
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
            test_ds = EuroSAT(
                cfg.data_path,
                split='test',
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
        case 'gtsrb':
            train_ds = GTSRB(
                cfg.data_path,
                split='train',
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
            test_ds = GTSRB(
                cfg.data_path,
                split='test',
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
        case 'svhn':
            train_ds = SVHN(
                cfg.data_path,
                split='train',
                download=True,
                transform=STANDARD_TRANSFORM,
                target_transform=lambda idx: idx + 100,
            )
            test_ds = SVHN(
                cfg.data_path,
                split='test',
                download=True,
                transform=STANDARD_TRANSFORM,
                target_transform=lambda idx: idx + 100,
            )
        case 'dtd':
            train_ds = DTD(
                cfg.data_path,
                split='train',
                download=True,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
            test_ds = DTD(
                cfg.data_path,
                split='test',
                download=True,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
        case 'cubirds':
            train_ds = CUBirds(
                cfg.data_path,
                train=True,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
            test_ds = CUBirds(
                cfg.data_path,
                train=False,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
        case 'vggflower':
            train_ds = VGGFlower(
                cfg.data_path,
                train=True,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
            test_ds = VGGFlower(
                cfg.data_path,
                train=False,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
        case 'aircraft':
            train_ds = Aircraft(
                cfg.data_path,
                train=True,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
            test_ds = Aircraft(
                cfg.data_path,
                train=False,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
        case 'trafficsigns':
            train_ds = TrafficSign(
                cfg.data_path,
                train=True,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
            test_ds = TrafficSign(
                cfg.data_path,
                train=False,
                transform=downscale_transform(32),
                target_transform=lambda idx: idx + 100,
            )
        case _:
            raise ValueError()
    c100_datasets, c100_class_datasets = build_cifar100_class_datasets(
        cfg.data_path,
        seed=cfg.seed,
        labeled_frac=cfg.labeled_frac,
        split_valid=cfg.eval.eval_valid,
        label_noise_frac=cfg.class_learner.label_noise_frac,
    )
    classes = get_dataset_labels(train_ds)

    def subset_labeled_dataset(dataset: Dataset, frac: float, seed: int) -> Dataset:
        if frac >= 1.0:
            return dataset

        labels = get_dataset_labels(dataset)
        indices, _ = train_test_split(
            list(range(len(dataset))),
            train_size=frac,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        return Subset(dataset, indices)

    sup_train_ds = subset_labeled_dataset(train_ds, cfg.labeled_frac, cfg.seed)
    sup_class_datasets = {c: get_class_dataset(sup_train_ds, c) for c in classes}
    class_datasets = {c: get_class_dataset(train_ds, c) for c in classes}

    sessions = [
        Session(
            0,
            epochs=(
                (Phase.TASK_LEARNING, cfg.pretrain_epochs[0]),
                (Phase.CONSOLIDATION, cfg.pretrain_epochs[1]),
            ),
            curr_ds=sup_class_datasets,
            full_curr_ds=class_datasets,
            full_past_ds={c: c100_class_datasets[c]['train_full'] for c in range(100)}
            if 'replay' in flags
            else {},
        )
    ]
    return Data(
        dataset=target_dataset,
        classes_per_dataset={'cifar100': 100, target_dataset: len(sup_class_datasets)},
        image_size=32,
        patch_size=4,
        eval_datasets={
            'cifar100': c100_datasets,
            target_dataset: {
                'train': sup_train_ds,
                'train_full': train_ds,
                'valid': EmptyDataset(),
                'test': test_ds,
            },
        },
        sessions=sessions,
        incrementality='offline',
    )
