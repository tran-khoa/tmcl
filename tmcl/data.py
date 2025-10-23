import logging
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, Protocol

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    RandomSampler,
)

from tmcl.datasets.dataset import EmptyDataset
from tmcl.hints import ClassBatch, ImageBatch
from tmcl.phase import Phase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Session:
    idx: int
    epochs: tuple[tuple[Phase, int], ...]

    curr_ds: dict[int, Dataset]
    full_curr_ds: dict[int, Dataset]
    past_ds: dict[int, Dataset] = field(default_factory=dict)
    full_past_ds: dict[int, Dataset] = field(default_factory=dict)

    @property
    def num_epochs(self) -> int:
        return sum(num_epochs for _, num_epochs in self.epochs)

    @property
    def current_classes(self) -> list[int]:
        return list(self.full_curr_ds.keys())

    @property
    def past_classes(self) -> list[int]:
        return list(self.full_past_ds.keys())

    @cached_property
    def seen_classes(self) -> list[int]:
        return list(self.full_curr_ds.keys()) + list(self.full_past_ds.keys())


@dataclass(slots=True)
class ConsBatch:
    images: ImageBatch
    labels: ClassBatch

    @property
    def batch_size(self) -> int:
        return len(self.images)


@dataclass(slots=True)
class MultiViewConsBatch:
    images: list[ImageBatch]
    labels: ClassBatch

    @property
    def batch_size(self) -> int:
        return len(self.images[0]) if self.images else 0

    @property
    def num_views(self) -> int:
        return len(self.images)


@dataclass(slots=True)
class TaskBatch:
    task_idx: int
    images: ImageBatch
    labels: ClassBatch

    @property
    def batch_size(self) -> int:
        return len(self.images)


@dataclass(slots=True)
class MultiViewSupBatch:
    task_idx: int
    images: list[ImageBatch]
    labels: ClassBatch

    @property
    def batch_size(self) -> int:
        return len(self.labels)


@dataclass(slots=True)
class SessionBatch:
    cons: MultiViewConsBatch | ConsBatch | None = None
    task: MultiViewSupBatch | TaskBatch | None = None

    @property
    def batch_size(self) -> int:
        """
        If both `cons` and `task` are present, the batch size is the maximum of the two.
        """
        return max(
            self.cons.batch_size if self.cons is not None else 0,
            self.task.batch_size if self.task is not None else 0,
        )


class SizedIterable[T](Iterable[T], Protocol):
    def __len__(self) -> int: ...


class KWayDataset[T](Dataset[T]):
    def __init__(self, class_datasets: Iterable[Dataset[T]], labels: Sequence[int]):
        self.relabeling = {c: i for i, c in enumerate(labels)}
        self.concat_dataset = ConcatDataset(class_datasets)

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, index: int) -> T:
        sample, label = self.concat_dataset[index]
        return sample, self.relabeling[label]


class OneVsAllDataset[T](Dataset[T]):
    def __init__(
        self,
        pos_datasets: Sequence[Dataset[T]],
        neg_datasets: Sequence[Dataset[T]] | None,
        *,
        neg_weights: Sequence[float] | None = None,
        relabel: Literal['none', 'bernouilli', 'rademacher'] = 'rademacher',
        reweight: Literal['none', 'balanced'] = 'balanced',
    ):
        """
        Args:
            pos_datasets (Sequence[Dataset[T]]): Positive class datasets.
            neg_datasets (Sequence[Dataset[T]]): Negative class datasets
            relabel (Literal['none', 'bernouilli', 'rademacher'], optional):
                Relabeling strategy.
                'none': keep original labels
                'bernouilli: labels are [0, 1] for negatives and positives, respectively
                'rademacher': labels are [-1, 1] for negatives and positives, respectively
            reweight (Literal['none', 'balanced'], optional):
                Reweighting strategy.
                'none': keep original distribution
                'balanced': 50/50 split between positive and negative, by sub- or oversampling the negative class. Requires at least one negative dataset.
        """
        self.pos_dataset = ConcatDataset(pos_datasets)
        self.neg_datasets = ConcatDataset(neg_datasets) if neg_datasets else EmptyDataset()
        if neg_weights is None:
            neg_weights = [1 / len(neg_datasets)] * len(neg_datasets)
        self.neg_weights = torch.as_tensor(neg_weights, dtype=torch.double)
        self.relabel = relabel
        self.reweight = reweight

    def _relabel(self, is_positive: bool, label: int):
        if self.relabel == 'none':
            return label
        elif self.relabel == 'bernouilli':
            return 1 if is_positive else 0
        elif self.relabel == 'rademacher':
            return 1 if is_positive else -1
        else:
            raise ValueError(f'Unknown relabeling strategy: {self.relabel}')

    def __getitem__(self, index: int) -> T:
        if index < len(self.pos_dataset):
            sample, label = self.pos_dataset[index]
            label = self._relabel(True, label)
        else:
            neg_idx = torch.multinomial(self.neg_weights, 1).item()
            sample, label = self.neg_datasets[neg_idx]
            label = self._relabel(False, label)
        return sample, label

    def __len__(self) -> int:
        """
        Compute-incremental: more negatives do not increase dataset size
        """
        return len(self.pos_dataset) * 2


class IncrementalSessionData:
    sessions: Sequence[Session]

    def __init__(
        self,
        sessions: Sequence[Session],
        *,
        labeled_setup: Literal['onevsall', 'kway', 'allvsall'] = 'onevsall',
        unsup_replacement: bool = False,
        sup_replacement: bool = False,
        batch_size_per_gpu: int,
        workers: int,
        last_batch: Literal['keep', 'drop', 'pad'],
    ):
        """
        Class-incremental learning, with unsupervised samples and 1-vs-all supervised samples.
        """
        self.sessions = tuple(sessions)
        self.labeled_setup = labeled_setup
        self.last_batch = last_batch

        self.batch_size_per_gpu = batch_size_per_gpu
        self.workers = workers
        self.unsup_replacement = unsup_replacement
        self.sup_replacement = sup_replacement

        self._current_session_phase: tuple[int, Phase] | None = None
        self._current_session_iter: Any = None

    def train_dataloader(self, session: int, phase: Phase) -> SizedIterable[SessionBatch]:
        if self._current_session_phase == (session, phase):
            return self._current_session_iter

        self._current_session_phase = (session, phase)
        self._current_session_iter = self.build_train_dataloader(session, phase)

        logger.info(f'Built a new iterator for session={session}, phase={phase}.')
        logger.info(f'Number of batches per epoch: {len(self._current_session_iter)}')
        return self._current_session_iter

    def build_allvsall_dataset(self, session: int | Session) -> dict[int, Dataset]:
        if isinstance(session, int):
            session = self.sessions[session]
        return {session.idx: ConcatDataset(session.curr_ds.values())}

    def build_kway_dataset(self, session: int | Session) -> dict[int, Dataset]:
        if isinstance(session, int):
            session = self.sessions[session]
        return {
            session.idx: KWayDataset(
                class_datasets=session.curr_ds.values(),
                labels=session.current_classes,
            )
        }

    def build_1vA_datasets(self, session: int | Session) -> dict[int, OneVsAllDataset]:
        if isinstance(session, int):
            session = self.sessions[session]
        datasets = {}
        for pos_class, pos_dataset in session.curr_ds.items():
            neg_datasets = [
                class_dataset
                for class_idx, class_dataset in session.curr_ds.items()
                if class_idx != pos_class
            ]
            weights = None
            datasets[pos_class] = OneVsAllDataset(
                pos_datasets=[pos_dataset],
                neg_datasets=neg_datasets,
                neg_weights=weights,
                relabel='bernouilli',
                reweight='balanced',
            )
        return datasets

    def build_train_dataloader(
        self, session: Session | int, phase: Phase
    ) -> SizedIterable[SessionBatch]:
        if isinstance(session, int):
            session = self.sessions[session]

        cons_dl = None
        if phase in (Phase.PRETRAIN, Phase.CONSOLIDATION):
            cons_dataset = ConcatDataset(session.full_curr_ds.values())
            num_samples = len(cons_dataset)
            if self.last_batch == 'pad':
                num_samples = (
                    math.ceil(num_samples / self.batch_size_per_gpu) * self.batch_size_per_gpu
                )
            joint_sampler = RandomSampler(
                cons_dataset,
                replacement=self.unsup_replacement,
                num_samples=num_samples,
            )

            cons_dl = DataLoader(
                cons_dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=self.workers,
                pin_memory=False,
                drop_last=self.last_batch == 'drop',
                persistent_workers=(self.workers > 0),
                sampler=joint_sampler,
            )

        class_dls: dict[int, DataLoader] | None = None
        if phase in (
            Phase.PRETRAIN,
            Phase.TASK_LEARNING,
            Phase.CONSOLIDATION,  # WORKAROUND: for supcon
        ):
            match self.labeled_setup:
                case 'onevsall':
                    class_datasets = self.build_1vA_datasets(session)
                case 'allvsall':
                    class_datasets = self.build_allvsall_dataset(session)
                case _:
                    raise ValueError(f'Unknown labeled_setup: {self.labeled_setup}')

            class_dls = {
                c: DataLoader(
                    dataset,
                    batch_size=self.batch_size_per_gpu,
                    num_workers=1 if (self.workers > 0) else 0,
                    pin_memory=False,
                    drop_last=self.last_batch == 'drop',
                    persistent_workers=(self.workers > 0),
                    sampler=RandomSampler(
                        dataset,
                        replacement=self.sup_replacement,
                        num_samples=math.ceil(
                            len(dataset) / self.batch_size_per_gpu
                        )  # smallest multiple of batch size >= len(dataset)
                        * self.batch_size_per_gpu
                        if self.last_batch == 'pad'
                        else None,
                    ),
                )
                for c, dataset in class_datasets.items()
            }

            num_samples = sum(len(ds) for ds in session.full_curr_ds.values())

            match self.last_batch:
                case 'keep' | 'pad':
                    num_batches = int(math.ceil(num_samples / self.batch_size_per_gpu))
                case 'drop':
                    num_batches = num_samples // self.batch_size_per_gpu
                case _:
                    raise ValueError(f'Unknown last_batch_strategy: {self.last_batch}')

        class _Iterator:
            def __init__(
                self,
                class_dls: dict[int, DataLoader] | None,
                joint_dl: DataLoader | None,
                *,
                num_batches: int,
            ):
                self.class_dls = class_dls
                self.cons_dl = joint_dl
                self.num_batches = num_batches

            def _reset_class_iterators(self) -> None:
                self._class_iters = (
                    {class_idx: iter(dl) for class_idx, dl in self.class_dls.items()}
                    if self.class_dls
                    else None
                )

            def _reset_joint_iterator(self) -> None:
                self._joint_iter = iter(self.cons_dl) if self.cons_dl else None

            def __iter__(self) -> Iterator[SessionBatch]:
                self._reset_class_iterators()
                self._reset_joint_iterator()

                for _ in range(self.num_batches):
                    task_batch = None
                    if self.class_dls is not None:
                        while True:
                            if not self._class_iters:
                                # All class iterators are exhausted, reset them
                                self._reset_class_iterators()

                            class_idx = random.choice(list(self._class_iters))
                            try:
                                images, labels = next(self._class_iters[class_idx])
                            except StopIteration:
                                del self._class_iters[class_idx]
                            else:
                                task_batch = TaskBatch(
                                    task_idx=class_idx,
                                    images=images,
                                    labels=labels,
                                )
                                break

                    cons_batch = None
                    if self.cons_dl is not None:
                        try:
                            joint_images, joint_classes = next(self._joint_iter)
                        except StopIteration:
                            self._reset_joint_iterator()
                            joint_images, joint_classes = next(self._joint_iter)
                        cons_batch = ConsBatch(images=joint_images, labels=joint_classes)

                    yield SessionBatch(cons=cons_batch, task=task_batch)

            def __len__(self) -> int:
                return self.num_batches

        return _Iterator(class_dls, cons_dl, num_batches=num_batches)
