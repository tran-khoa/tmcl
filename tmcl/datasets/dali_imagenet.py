import bisect
import itertools
import logging
import math
import multiprocessing
import random
import tarfile
import time
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Final, Literal, Self

import numpy as np
import torch
import tqdm
from jaxtyping import Integer
from lightly.transforms.utils import IMAGENET_NORMALIZE
from nvidia.dali import Pipeline, fn, pipeline_def, types
from nvidia.dali.backend_impl.types import INTERP_LINEAR
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.types import BatchInfo

from tmcl.data import MultiViewConsBatch, MultiViewSupBatch, SessionBatch, TaskBatch
from tmcl.phase import Phase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    files: list[str]
    labels: list[int]
    supervised: list[bool]

    def __len__(self) -> int:
        """Returns the number of all samples in the dataset."""
        return len(self.files)

    @cached_property
    def indices(self) -> dict[str, int]:
        """Returns a mapping from file paths to their indices in the dataset."""
        return {Path(file).name: idx for idx, file in enumerate(self.files)}

    @property
    def num_samples(self) -> int:
        return len(self.files)

    @cached_property
    def num_classes(self) -> int:
        return max(self.labels) + 1

    def unsupervised_dataset(self, classes: Iterable[int] | None) -> tuple[list[str], list[int]]:
        files = []
        labels = []

        if classes is None:
            classes = set(self.labels)
        else:
            classes = set(classes)

        for file, label in zip(self.files, self.labels, strict=True):
            if label in classes:
                files.append(file)
                labels.append(label)
        return files, labels

    def supervised_dataset(self, classes: Iterable[int] | None) -> tuple[list[str], list[int]]:
        files = []
        labels = []

        if classes is None:
            classes = set(self.labels)
        else:
            classes = set(classes)

        for idx, s in enumerate(self.supervised):
            if s and self.labels[idx] in classes:
                files.append(self.files[idx])
                labels.append(self.labels[idx])
        return files, labels


@dataclass(frozen=True)
class ImageNet100Dataset(Dataset):
    def __init__(
        self,
        tar_root: Path | str,
        split: Literal['train', 'val'],
        local_path: Path | str = Path('/dev/shm/datasets/imagenet100'),
        n_procs: int = 8,
        supervised_frac: float = 1.0,
        seed: int = 0,
        copy: bool = True,
    ):
        tar_root = Path(tar_root) / split
        local_path = Path(local_path) / split
        is_distributed: Final[bool] = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )

        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        if is_distributed:
            logger.info(
                f'Using distributed setup for ImageNet100Dataset, worker {rank + 1}/{world_size}.'
            )
        else:
            logger.info('Using single worker setup for ImageNet100Dataset.')

        if copy:
            if rank == 0:
                local_path.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                tars = [p for p in tar_root.iterdir() if p.suffix == '.tar']
                logger.info(f' ** Found {len(tars)} tar files.')

                with multiprocessing.Pool(processes=n_procs) as pool:
                    list(
                        tqdm.tqdm(
                            pool.imap_unordered(
                                partial(self._extract_tar, target_root=local_path), tars
                            ),
                            total=len(tars),
                            desc='Unpacking tar files',
                        )
                    )

                logger.info(f' * Unpacked in {time.time() - t0} seconds!')
                torch.distributed.barrier() if is_distributed else None
            else:
                logger.info(
                    f'Worker {rank + 1}/{world_size} waiting for tar files to be unpacked...'
                )
                torch.distributed.barrier() if is_distributed else None
                logger.info(f'Worker {rank + 1}/{world_size} dataset initialized.')

        files = Path(local_path).glob('**/*.JPEG')
        files = [file for file in files if file.is_file()]
        labels = [file.parent.name for file in files]

        classes = list(set(labels))
        classes.sort()

        files = [str(file) for file in files]
        labels = [classes.index(label) for label in labels]

        if supervised_frac < 1.0:
            # Randomly sample a fraction of the files for each class
            label2files = {label: [] for label in set(labels)}
            for file, label in zip(files, labels, strict=True):
                label2files[label].append(file)

            sup_files = []
            for label in set(labels):
                num_labeled = int(len(label2files[label]) * supervised_frac)
                temp_rng = random.Random(seed + label)
                sup_files += temp_rng.sample(label2files[label], k=num_labeled)

            print(f'Using {supervised_frac} of labeled data: {len(files)} -> {len(sup_files)}')
            sup_files = set(sup_files)
            supervised = [file in sup_files for file in files]
        else:
            supervised = [True] * len(files)

        super().__init__(files=files, labels=labels, supervised=supervised)

    @classmethod
    def _extract_tar(cls, tar_path: Path, target_root: Path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(target_root / tar_path.stem)


@dataclass(frozen=True)
class TinyImageNet200Dataset(Dataset):
    def __init__(self, file_root: Path | str):
        labels = {}

        def _assign_label(file_path: Path) -> int:
            """Extracts the label from the file path."""
            str_label = file_path.parent.parent.name
            if str_label not in labels:
                labels[str_label] = len(labels)
            return labels[str_label]

        files = Path(file_root).glob('**/*.JPEG')
        files = [str(file) for file in files if file.is_file()]
        labels = [_assign_label(file) for file in Path(file_root).glob('**/*.JPEG')]
        super().__init__(files=files, labels=labels, supervised=[True] * len(files))


def random_color_jitter(
    images,
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1,
    prob: float = 0.8,
    device: str = 'gpu',
):
    if fn.random.coin_flip(probability=prob):
        """Applies random color jitter to the images."""
        brightness = fn.random.uniform(
            range=(max(0.0, 1 - brightness), 1 + brightness), dtype=types.FLOAT
        )
        contrast = fn.random.uniform(
            range=(max(0.0, 1 - contrast), 1 + contrast), dtype=types.FLOAT
        )
        saturation = fn.random.uniform(
            range=(max(0.0, 1 - saturation), 1 + saturation), dtype=types.FLOAT
        )
        hue_deg = 360 * hue
        hue = fn.random.uniform(range=(-hue_deg, hue_deg), dtype=types.FLOAT)

        images = fn.color_twist(
            images,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            device=device,
        )

    return images


def random_grayscale(images, prob: float = 0.2, device: str = 'gpu'):
    """Applies random grayscale to the images."""
    if fn.random.coin_flip(probability=prob):
        images = fn.color_space_conversion(
            images, image_type=types.RGB, output_type=types.GRAY, device=device
        )
        images = fn.cat(images, images, images, axis=2, device=device)  # Convert grayscale to RGB
    return images


def random_gaussian_blur(images, prob: float = 0.5, device: str = 'gpu'):
    """Applies random Gaussian blur to the images."""
    if fn.random.coin_flip(probability=prob):
        sigma = fn.random.uniform(range=[0, 1], dtype=types.FLOAT)
        sigma = 0.1 + sigma * 1.9
        images = fn.gaussian_blur(
            images,
            device=device,
            sigma=sigma,
        )
    return images


def random_solarization(images, threshold: float = 128.0, prob: float = 0.5):
    if fn.random.coin_flip(probability=prob):
        inv_image = 255 - images
        mask = images >= threshold
        images = mask * inv_image + (True ^ mask) * images
        images = fn.cast(images, dtype=types.UINT8)
    return images


@pipeline_def(enable_conditionals=True)
def ssl_imagenet_pipeline(
    files: Sequence[str],
    labels: Sequence[int],
    num_views: int,
    device: Literal['gpu', 'cpu'],
    shard_id: int,
    num_shards: int,
):
    pipeline: Pipeline = Pipeline.current()
    seed = pipeline.seed

    images, labels = fn.readers.file(
        files=files,
        labels=labels,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle_after_epoch=True,
        name='Reader',
        seed=seed,
    )
    decoder_device = 'mixed' if device == 'gpu' else 'cpu'

    # ImageNet hints
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    # random crop and resize
    images = fn.decoders.image(
        images,
        device=decoder_device,
        output_type=types.RGB,
        preallocate_width_hint=preallocate_width_hint,
        preallocate_height_hint=preallocate_height_hint,
    )

    crops = []
    for i in range(num_views):
        # asymmetric
        prob_grayscale = 1.0 if i % 2 == 0 else 0.1
        prob_solarize = 0.0 if i % 2 == 0 else 0.2

        crop = fn.random_resized_crop(
            images,
            size=224,
            device=device,
            random_aspect_ratio=[3 / 4, 4 / 3],
            random_area=[0.08, 1.0],
        )
        crop = random_color_jitter(
            crop,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            prob=0.8,
            device=device,
        )
        crop = random_grayscale(crop, prob=prob_grayscale, device=device)
        crop = random_gaussian_blur(crop, prob=0.5, device=device)
        crop = random_solarization(crop, threshold=128.0, prob=prob_solarize)
        crop = fn.crop_mirror_normalize(
            crop,
            device=device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mirror=fn.random.coin_flip(),
            mean=[x * 255 for x in IMAGENET_NORMALIZE['mean']],
            std=[x * 255 for x in IMAGENET_NORMALIZE['std']],
        )
        crops.append(crop)
    if device == 'gpu':
        labels = labels.gpu()

    labels = fn.cast(labels, dtype=types.INT64)

    return *crops, labels


@pipeline_def
def sup_imagenet_pipeline(
    files: Sequence[str],
    labels: Sequence[int],
    device: Literal['gpu', 'cpu'],
    shard_id: int,
    num_shards: int,
):
    pipeline: Pipeline = Pipeline.current()
    seed = pipeline.seed

    images, labels = fn.readers.file(
        files=files,
        labels=labels,
        shard_id=shard_id,
        num_shards=num_shards,
        shuffle_after_epoch=True,
        name='Reader',
        seed=seed,
    )
    decoder_device = 'mixed' if device == 'gpu' else 'cpu'

    # ImageNet hints
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    # random crop and resize
    images = fn.decoders.image(
        images,
        device=decoder_device,
        output_type=types.RGB,
        preallocate_width_hint=preallocate_width_hint,
        preallocate_height_hint=preallocate_height_hint,
        hw_decoder_load=0.5,
    )
    images = fn.random_resized_crop(
        images,
        size=224,
        device=device,
        random_aspect_ratio=[3 / 4, 4 / 3],
        random_area=[0.08, 1.0],
        interp_type=INTERP_LINEAR,
    )
    images = fn.crop_mirror_normalize(
        images,
        device=device,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mirror=fn.random.coin_flip(),
        mean=[x * 255 for x in IMAGENET_NORMALIZE['mean']],
        std=[x * 255 for x in IMAGENET_NORMALIZE['std']],
    )
    if device == 'gpu':
        labels = labels.gpu()

    labels = fn.cast(labels, dtype=types.INT64)

    return images, labels


class OneVsAllSourceFn:
    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        shard_id: int,
        num_shards: int,
        seed: int,
        neg_files: Sequence[str] | None = None,
        batch_size_per_device: int = 64,
    ):
        self.files = files
        self.labels = labels
        self.neg_files = neg_files
        self.seed = seed
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.batch_size_per_device = batch_size_per_device

        self.classes = set(labels)
        self.class2files = {label: [] for label in self.classes}
        for file, label in zip(self.files, self.labels, strict=True):
            self.class2files[label].append(file)

    def get_shard(self, epoch: int):
        """
        State-less sharding function for OneVsAllSource, perhaps a bit inefficient, not sure what kind of caching would be OK with DALI here...
        """
        files = []
        labels = []

        for label in self.class2files:
            class_files = list(self.class2files[label])
            rng = random.Random(self.seed + label + epoch * 10_000)
            rng.shuffle(class_files)
            max_samples_per_shard = math.ceil(len(class_files) / self.num_shards)
            samples = list(itertools.batched(class_files, int(max_samples_per_shard)))[
                self.shard_id
            ]
            files.extend(samples)
            labels.extend([label] * len(samples))
        return files, labels

    def __call__(self, batch_info: BatchInfo) -> tuple[list[np.ndarray], list[np.ndarray], Any]:
        files, labels = self.get_shard(batch_info.epoch_idx)

        # Assuming num_epochs < 10k
        rng = random.Random(self.seed + batch_info.iteration * 10_000 + batch_info.epoch_idx)
        class_idx = rng.choice(list(self.classes))
        positives = [
            (x, 1)
            for x in rng.choices(
                [f for f, l in zip(files, labels, strict=True) if l == class_idx],
                k=self.batch_size_per_device // 2,
            )
        ]
        if self.neg_files is None:
            # negatives are all other supervised classes
            negatives = [
                (x, 0)
                for x in rng.choices(
                    [f for f, l in zip(files, labels, strict=True) if l != class_idx],
                    k=self.batch_size_per_device // 2,
                )
            ]
        else:
            # negatives are from a separate set of files, i.e. an unlabeled dataset
            negatives = [
                (x, 0)
                for x in rng.choices(
                    self.neg_files,
                    k=self.batch_size_per_device // 2,
                )
            ]
        combined = positives + negatives
        rng.shuffle(combined)  # just to be sure, mix up between positives and negatives
        paths, labels = zip(*combined, strict=False)
        images = []
        for p in paths:
            with open(p, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))
        labels = [np.array([label], dtype=np.int64) for label in labels]
        tasks = [np.array([class_idx], dtype=np.int64)] * len(labels)

        return images, labels, tasks


def build_onevsall_imagenet_pipeline(
    files: Sequence[str],
    labels: Sequence[int],
    device: Literal['gpu', 'cpu'],
    shard_id: int,
    num_shards: int,
    seed: int,
    neg_files: Sequence[str] | None = None,
    batch_size_per_gpu: int = 64,
    num_threads: int = 4,
):
    src_fn = OneVsAllSourceFn(
        files=files,
        labels=labels,
        neg_files=neg_files,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=seed,
        batch_size_per_device=batch_size_per_gpu,
    )
    pipeline = onevsall_imagenet_pipeline(
        src_fn,
        device=device,
        batch_size=batch_size_per_gpu,
        num_threads=num_threads,
        seed=seed,
    )
    pipeline = typing.cast(Pipeline, pipeline)
    pipeline.build()
    return pipeline


class IdxToLabel:
    def __init__(self, labels: Sequence[int]):
        self.labels = np.array(labels, dtype=np.int64)

    def __call__(self, idxs: Integer[np.ndarray, 'b']):
        return self.labels[idxs]


# https://github.com/NVIDIA/DALI/issues/5377#issuecomment-2341767533
@pipeline_def(exec_async=False, exec_pipelined=False, exec_dynamic=False)
def eval_imagenet_pipeline(
    files: Sequence[str],
    labels: Sequence[int],
    device: Literal['gpu', 'cpu'],
    shard_id: int,
    num_shards: int,
):
    images_paths, idxs = fn.readers.file(
        files=files,
        labels=list(range(len(files))),  # enumerate
        shard_id=shard_id,
        num_shards=num_shards,
        name='Reader',
        pad_last_batch=True,
    )
    # labels = fn.python_function(
    #     idxs,
    #     function=IdxToLabel(labels),
    #     num_outputs=1,
    #     batch_processing=True,
    # )
    # labels = types.Constant(np.array(labels, dtype=np.int64))
    # labels = fn.external_source(lambda: np.array(labels, dtype=np.int64), layout='H')[idxs]
    # paths = images_paths.source_info(device=device)
    decoder_device = 'mixed' if device == 'gpu' else 'cpu'

    # ImageNet hints
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    images = fn.decoders.image(
        images_paths,
        device=decoder_device,
        output_type=types.RGB,
        preallocate_width_hint=preallocate_width_hint,
        preallocate_height_hint=preallocate_height_hint,
    )
    images = fn.resize(images, antialias=True, interp_type=INTERP_LINEAR, size=256, device=device)
    images = fn.crop_mirror_normalize(
        images,
        device=device,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        crop=[224, 224],
        mean=[x * 255 for x in IMAGENET_NORMALIZE['mean']],
        std=[x * 255 for x in IMAGENET_NORMALIZE['std']],
    )
    if device == 'gpu':
        # labels = labels.gpu()
        idxs = idxs.gpu()
    # labels = fn.cast(labels, dtype=types.INT64)
    idxs = fn.cast(idxs, dtype=types.INT64)

    return images, idxs


@pipeline_def(py_start_method='spawn')
def onevsall_imagenet_pipeline(
    src_fn: OneVsAllSourceFn,
    device: Literal['gpu', 'cpu'],
):
    images, labels, tasks = fn.external_source(
        src_fn, batch=True, batch_info=True, parallel=True, num_outputs=3, layout=['H', 'H', 'H']
    )

    decoder_device = 'mixed' if device == 'gpu' else 'cpu'

    # ImageNet hints
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0

    # random crop and resize
    images = fn.decoders.image_random_crop(
        images,
        device=decoder_device,
        output_type=types.RGB,
        preallocate_width_hint=preallocate_width_hint,
        preallocate_height_hint=preallocate_height_hint,
        random_aspect_ratio=[3 / 4, 4 / 3],
        random_area=[0.08, 1.0],
    )
    images = fn.resize(
        images,
        device=device,
        resize_x=224,
        resize_y=224,
        interp_type=types.INTERP_LINEAR,
        antialias=True,
    )

    # random horizontal flip and normalization
    images = fn.crop_mirror_normalize(
        images,
        device=device,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mirror=fn.random.coin_flip(),
        mean=[x * 255 for x in IMAGENET_NORMALIZE['mean']],
        std=[x * 255 for x in IMAGENET_NORMALIZE['std']],
    )
    if device == 'gpu':
        labels = labels.gpu()
        tasks = tasks.gpu()
    labels = fn.cast(labels, dtype=types.INT64)

    return images, labels, tasks


class TMCLDaliIterator:
    def __init__(
        self,
        cons_pipeline,
        sup_pipeline,
        num_views: int,
        num_batches: int,
        contrastive_allvsall: bool,
        auto_reset=True,
        **kwargs,
    ):
        self.cons_iterator = None
        self.contrastive_allvsall = contrastive_allvsall
        if cons_pipeline is not None:
            self.cons_iterator = DALIGenericIterator(
                cons_pipeline,
                output_map=[*[f'view{i}' for i in range(num_views)], 'labels'],
                auto_reset=auto_reset,
                **kwargs,
            )
        self.sup_iterator = None
        if sup_pipeline is not None:
            if contrastive_allvsall:
                # For contrastive allvsall, we need to return tasks as well
                self.sup_iterator = DALIGenericIterator(
                    sup_pipeline,
                    output_map=[*[f'view{i}' for i in range(num_views)], 'labels'],
                    auto_reset=auto_reset,
                    **kwargs,
                )
            else:
                self.sup_iterator = DALIGenericIterator(
                    sup_pipeline,
                    output_map=['images', 'labels', 'tasks'],
                    auto_reset=auto_reset,
                    **kwargs,
                )
        self.num_views = num_views
        self.num_batches = num_batches
        self.current_batch = 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Self:
        self.current_batch = 0
        return self

    def __next__(self) -> SessionBatch:
        if self.current_batch >= self.num_batches:
            raise StopIteration

        cons_batch = None
        if self.cons_iterator is not None:
            cons_batch = next(self.cons_iterator)
            cons_batch = MultiViewConsBatch(
                images=[cons_batch[0][f'view{i}'] for i in range(self.num_views)],
                labels=cons_batch[0]['labels'].squeeze(1),
            )

        sup_batch = None
        if self.sup_iterator is not None:
            sup_batch = next(self.sup_iterator)
            if self.contrastive_allvsall:
                sup_batch = MultiViewSupBatch(
                    task_idx=0,
                    images=[sup_batch[0][f'view{i}'] for i in range(self.num_views)],
                    labels=sup_batch[0]['labels'].squeeze(1),
                )
            else:
                sup_batch = TaskBatch(
                    task_idx=sup_batch[0]['tasks'][0].item(),
                    images=sup_batch[0]['images'],
                    labels=sup_batch[0]['labels'].squeeze(1),
                )
        self.current_batch += 1
        return SessionBatch(cons=cons_batch, task=sup_batch)


def compute_cil_num_batches(
    dataset: Dataset,
    classes: Sequence[int],
    total_batch_size: int,
):
    files, _ = dataset.unsupervised_dataset(classes)
    num_samples = len(files)
    return int(math.ceil(num_samples / total_batch_size))


def build_cil_dataloader(
    dataset: Dataset,
    phase: Phase,
    classes: Sequence[int],
    *,
    num_views: int,
    device: Literal['gpu', 'cpu'],
    batch_size_per_gpu: int,
    num_threads: int,
    shard_id: int,
    num_shards: int,
    seed: int,
    contrastive_allvsall: bool,
) -> TMCLDaliIterator:
    sup_pipeline = None
    if phase in (Phase.PRETRAIN, Phase.TASK_LEARNING, Phase.CONSOLIDATION):
        sup_files, sup_labels = dataset.supervised_dataset(classes)
        if contrastive_allvsall:
            sup_pipeline = ssl_imagenet_pipeline(
                files=sup_files,
                labels=sup_labels,
                num_views=num_views,
                device=device,
                batch_size=batch_size_per_gpu,
                num_threads=num_threads,
                shard_id=shard_id,
                num_shards=num_shards,
                seed=seed,
            )
        else:
            sup_pipeline = build_onevsall_imagenet_pipeline(
                files=sup_files,
                labels=sup_labels,
                device=device,
                batch_size_per_gpu=batch_size_per_gpu,
                num_threads=num_threads,
                shard_id=shard_id,
                num_shards=num_shards,
                seed=seed,
            )

    cons_pipeline = None
    if phase in (Phase.PRETRAIN, Phase.CONSOLIDATION):
        # Filter out classes
        unsup_files, unsup_labels = dataset.unsupervised_dataset(classes)
        cons_pipeline = ssl_imagenet_pipeline(
            files=unsup_files,
            labels=unsup_labels,
            num_views=num_views,
            device=device,
            batch_size=batch_size_per_gpu,
            num_threads=num_threads,
            shard_id=shard_id,
            num_shards=num_shards,
            seed=seed,  # epoch reset via fn.readers.file
        )
        cons_pipeline = typing.cast(Pipeline, cons_pipeline)
        cons_pipeline.build()

    iterator = TMCLDaliIterator(
        cons_pipeline,
        sup_pipeline,
        num_views=num_views,
        num_batches=compute_cil_num_batches(dataset, classes, batch_size_per_gpu * num_shards),
        contrastive_allvsall=contrastive_allvsall,
    )

    return iterator


class IncrementalDALISessionData:
    session_classes: list[list[int]]
    session_epochs: tuple[tuple[tuple[Phase, int], ...], ...]

    def __init__(
        self,
        dataset: Dataset,
        session_classes: list[list[int]],
        *,
        num_views: int,
        batch_size_per_gpu: int,
        workers: int,
        seed: int,
        device: Literal['gpu', 'cpu'],
        shard_id: int,
        num_shards: int,
        contrastive_allvsall: bool = False,
    ):
        self.dataset = dataset
        self.session_classes = session_classes
        self.batch_size_per_gpu = batch_size_per_gpu
        self.workers = workers
        self.seed = seed
        self.device = device
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.num_views = num_views
        self.contrastive_allvsall = contrastive_allvsall

        self._current_session_phase: tuple[int, Phase] | None = None
        self._current_session_iter: Any = None

    def build_train_dataloader(self, session: int, phase: Phase):
        session_classes = self.session_classes[session]
        return build_cil_dataloader(
            dataset=self.dataset,
            classes=session_classes,
            num_views=self.num_views,
            phase=phase,
            seed=self.seed,
            device=self.device,
            batch_size_per_gpu=self.batch_size_per_gpu,
            num_threads=self.workers,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            contrastive_allvsall=self.contrastive_allvsall,
        )

    def train_dataloader(self, session: int, phase: Phase) -> TMCLDaliIterator:
        if self._current_session_phase == (session, phase):
            return self._current_session_iter

        self._current_session_phase = (session, phase)
        self._current_session_iter = self.build_train_dataloader(session, phase)

        logger.info(f'Built a new iterator for session={session}, phase={phase}.')
        return self._current_session_iter


class ContinualMeta:
    def __init__(
        self,
        dataset: Dataset,
        session_classes: list[list[int]],
        session_epochs: Sequence[Sequence[tuple[Phase, int]]],
        *,
        total_batch_size: int,
    ):
        self.dataset = dataset
        self._session_classes = session_classes

        self._session_epochs: list[int] = [
            sum(eps for _, eps in phases) for phases in session_epochs
        ]
        self._acc_session_epochs: list[int] = list(itertools.accumulate(self._session_epochs))

        self._phase_epochs: list[tuple[int, Phase, int]] = [
            (session, phase, epochs)
            for session, (session_meta, classes) in enumerate(
                zip(session_epochs, session_classes, strict=True)
            )
            for phase, epochs in session_meta
        ]
        self._acc_phase_epochs: list[tuple[int, Phase, int]] = list(
            itertools.accumulate(
                self._phase_epochs, func=lambda acc, x: (x[0], x[1], acc[2] + x[2])
            )
        )

        self._phase_steps = [
            (session, phase, epochs * compute_cil_num_batches(dataset, classes, total_batch_size))
            for session, (session_meta, classes) in enumerate(
                zip(session_epochs, session_classes, strict=True)
            )
            for phase, epochs in session_meta
        ]
        self._acc_phase_steps = list(
            itertools.accumulate(self._phase_steps, func=lambda acc, x: (x[0], x[1], acc[2] + x[2]))
        )

        self._session_steps = [
            sum(batches for _, _, batches in phase_defs)
            for _, phase_defs in itertools.groupby(self._phase_steps, key=lambda x: x[0])
        ]
        self._acc_session_steps = list(itertools.accumulate(self._session_steps))

        self._current_step = None
        self._current_epoch = None
        self._current_idx = None

    def set_current_step(self, global_step: int, epoch: int) -> None:
        self._current_step = global_step
        if self._current_epoch != epoch:
            self._current_epoch = epoch
            new_idx = bisect.bisect_right(self._acc_phase_epochs, epoch, key=lambda x: x[2])
            if new_idx != self._current_idx:
                self._current_idx = new_idx
                logger.info(
                    f'Switching to session={self.session}, phase={self.phase} at {global_step=}, {epoch=}.'
                )

    @property
    def is_last_session_epoch(self) -> bool:
        if (
            self._current_idx + 1 < len(self._phase_epochs)
            and self._phase_epochs[self._current_idx + 1][0] == self.session
        ):
            # next phase exists and is in the same session
            return False
        # next phase doesn't exist or is in the next session
        return self.phase_epoch == self.num_phase_epochs - 1

    @property
    def current_idx(self) -> int:
        return self._current_idx

    @property
    def timeline_steps(self) -> list[tuple[int, Phase, int]]:
        return self._phase_steps

    @property
    def timeline_epochs(self) -> list[tuple[int, Phase, int]]:
        return self._phase_epochs

    @property
    def num_phase_epochs(self) -> int:
        _, _, phase_epochs = self._phase_epochs[self._current_idx]
        return phase_epochs

    @property
    def num_phase_steps(self) -> int:
        _, _, phase_steps = self._phase_steps[self._current_idx]
        return phase_steps

    @property
    def phase(self) -> Phase:
        _, phase, _ = self._phase_steps[self._current_idx]
        return phase

    @property
    def session(self) -> int:
        session, _, _ = self._phase_steps[self._current_idx]
        return session

    @property
    def phase_step(self) -> int:
        if self._current_idx == 0:
            preceding_steps = 0
        else:
            _, _, preceding_steps = self._acc_phase_steps[self._current_idx - 1]
        return self._current_step - preceding_steps

    @property
    def phase_epoch(self):
        if self._current_idx == 0:
            preceding_epochs = 0
        else:
            _, _, preceding_epochs = self._acc_phase_epochs[self._current_idx - 1]
        return self._current_epoch - preceding_epochs

    @property
    def session_step(self):
        if self.session == 0:
            preceding_steps = 0
        else:
            preceding_steps = self._acc_session_steps[self.session - 1]
        return self._current_step - preceding_steps

    @property
    def seen_classes(self) -> set[int]:
        return {
            c
            for session, _, _ in self._phase_steps[: self._current_idx + 1]
            for c in self._session_classes[session]
        }

    @property
    def num_sessions(self) -> int:
        return len(self._session_classes)

    @cached_property
    def total_steps(self) -> int:
        return self._acc_phase_steps[-1][2] + self._phase_steps[-1][2]
