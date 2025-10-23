import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import timm
import torch
import torch.nn.functional as F
import tqdm
from jaxtyping import Float, Integer
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIRaggedIterator
from rich.logging import RichHandler
from torch import Tensor

from represent.rankme import rankme
from tmcl.datasets.dali_imagenet import Dataset, ImageNet100Dataset, eval_imagenet_pipeline
from tmcl.nn.ddp import all_reduce_op

logger = logging.getLogger(__name__)


def decode_path_tensors(tensors: list[Tensor]) -> list[Path]:
    paths = []
    for tensor in tensors:
        paths.append(Path(np.array(tensor.cpu()).tobytes().decode()))
    return paths


class KnnModule(torch.nn.Module):
    """
    Shamelessly copied from DINOv2 (https://github.com/facebookresearch/dinov2).
    Gets knn of test features from all processes on a chunk of the train features

    Each rank gets a chunk of the train features as well as a chunk of the test features.
    In `compute_neighbors`, for each rank one after the other, its chunk of test features
    is sent to all devices, partial knns are computed with each chunk of train features
    then collated back on the original device.
    """

    def __init__(
        self,
        train_features,
        train_labels,
        nb_knn: Sequence[int],
        temperature: float,
        device: str,
        num_classes: int,
    ):
        super().__init__()

        self.is_distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        print(f'Using distributed mode: {self.is_distributed}')

        self.global_rank = torch.distributed.get_rank() if self.is_distributed else 0
        self.global_size = torch.distributed.get_world_size() if self.is_distributed else 1
        print(f'Global rank: {self.global_rank}, Global size: {self.global_size}')

        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(
            self.device
        )
        self.candidates = (
            train_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)
        )

        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = temperature
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        # Send the features from `source_rank` to all ranks
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)

        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(
                *broadcast_shape, dtype=features_rank.dtype, device=self.device
            )
        torch.distributed.broadcast(broadcasted, source_rank)

        # Compute the neighbors for `source_rank` among `train_features_rank_T`
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        # Gather all neighbors for `target_rank`
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]

        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)

        if self.global_rank == target_rank:
            # Perform a second top-k on the k * global_size retrieved neighbors
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        """
        Compute the results on all values of `self.nb_knn` neighbors from the full `self.max_k`
        """
        assert all(k <= self.max_k for k in self.nb_knn)

        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = F.softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            F.one_hot(neighbors_labels, num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


class EvalModule(torch.nn.Module):
    def __init__(
        self,
        *,
        knn_temp: float = 0.07,
        knn_k: int = 20,
        model: torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        shard_id: int,
        num_shards: int,
        num_workers: int,
        batch_size_per_device: int,
        device: Literal['cpu', 'gpu'],
        gather_device: Literal['cpu', 'gpu'] = 'cpu',
    ):
        super().__init__()
        self.knn_temp = knn_temp
        self.knn_k = knn_k
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.num_workers = num_workers
        self.batch_size_per_device = batch_size_per_device
        self.device = device
        self.gather_device = gather_device

        self.is_distributed = num_shards > 1
        if self.is_distributed:
            assert torch.distributed.is_initialized(), (
                'Distributed training is enabled but PyTorch distributed is not initialized.'
            )

    def all_gather_and_flatten(self, tensor_rank):
        if not self.is_distributed:
            return tensor_rank

        tensor_all_ranks = torch.empty(
            self.num_shards,
            *tensor_rank.shape,
            dtype=tensor_rank.dtype,
            device=tensor_rank.device,
        )
        tensor_list = list(tensor_all_ranks.unbind(0))
        torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
        return tensor_all_ranks.flatten(end_dim=1)

    @torch.inference_mode()
    def extract_features(self) -> tuple[Float[Tensor, 'n d'], Integer[Tensor, 'n']]:
        self.model.eval()
        train_pipe = eval_imagenet_pipeline(
            files=self.train_dataset.files,
            labels=self.train_dataset.labels,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            batch_size=self.batch_size_per_device,
            num_threads=self.num_workers,
            device=self.device,
        )
        train_loader = DALIRaggedIterator(
            train_pipe,
            output_map=['images', 'idxs'],
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
                DALIRaggedIterator.DENSE_TAG,
            ],
            reader_name='Reader',
            last_batch_policy=LastBatchPolicy.FILL,
        )
        all_features, all_labels = None, None
        device = 'cuda' if self.device == 'gpu' else 'cpu'
        idx2labels = torch.tensor(self.train_dataset.labels, device=device, dtype=torch.int64)

        for batch in tqdm.tqdm(train_loader, desc='Extracting features'):
            images: Float[Tensor, 'b c h w'] = batch[0]['images']
            indices: Integer[Tensor, 'b'] = batch[0]['idxs'].squeeze()
            labels = idx2labels[indices]

            features: Float[Tensor, 'b d'] = self.model(images)
            features = F.normalize(features, dim=1, p=2)

            if all_features is None:
                all_features = torch.zeros(
                    len(self.train_dataset), features.shape[-1], device=self.gather_device
                )
                all_labels = torch.full(
                    (len(self.train_dataset),), fill_value=-1, device=self.gather_device
                )
            index_all = self.all_gather_and_flatten(indices).to(self.gather_device)
            features_all = self.all_gather_and_flatten(features).to(self.gather_device)
            labels_all = self.all_gather_and_flatten(labels).to(self.gather_device)

            if len(index_all) > 0:
                all_features.index_copy_(0, index_all, features_all)
                all_labels.index_copy_(0, index_all, labels_all)

        if self.is_distributed:
            torch.distributed.barrier()

        assert torch.all(all_labels > -1), (
            f'Some labels are not gathered correctly ({all_labels=}).'
        )
        return all_features, all_labels

    @torch.inference_mode()
    def evaluate(self) -> dict[str, Any]:
        train_features, train_labels = self.extract_features()
        knn_module = KnnModule(
            train_features=train_features,
            train_labels=train_labels,
            nb_knn=[self.knn_k],
            temperature=self.knn_temp,
            num_classes=self.train_dataset.num_classes,
            device='cuda' if self.device == 'gpu' else 'cpu',
        )

        self.model.eval()
        eval_pipe = eval_imagenet_pipeline(
            files=self.eval_dataset.files,
            labels=self.eval_dataset.labels,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            batch_size=self.batch_size_per_device,
            num_threads=self.num_workers,
            device=self.device,
        )
        eval_loader = DALIRaggedIterator(
            eval_pipe,
            output_map=['images', 'idxs'],
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
                DALIRaggedIterator.DENSE_TAG,
            ],
            reader_name='Reader',
            last_batch_policy=LastBatchPolicy.FILL,
        )
        device = 'cuda' if self.device == 'gpu' else 'cpu'
        idx2labels = torch.tensor(self.eval_dataset.labels, device=device, dtype=torch.int64)

        all_knn_correct = None
        all_labels = None
        batch_rankme = []
        for batch in tqdm.tqdm(eval_loader, desc='Evaluating'):
            images: Float[Tensor, 'b c h w'] = batch[0]['images']
            indices: Integer[Tensor, 'b'] = batch[0]['idxs'].squeeze()
            labels = idx2labels[indices]

            features: Float[Tensor, 'b d'] = self.model(images)
            feature_rankme = rankme(features)

            features = F.normalize(features, dim=1, p=2)
            knn_scores: Float[Tensor, 'b c'] = knn_module(features)[self.knn_k]
            knn_correct: Integer[Tensor, 'b'] = (torch.argmax(knn_scores, dim=1) == labels).to(
                torch.int64
            )
            if all_knn_correct is None:
                all_knn_correct = torch.full(
                    (len(self.eval_dataset),),
                    fill_value=-1,
                    device=self.gather_device,
                    dtype=torch.int64,
                )
                all_labels = torch.full(
                    (len(self.eval_dataset),),
                    fill_value=-1,
                    device=self.gather_device,
                    dtype=torch.int64,
                )

            index_all = self.all_gather_and_flatten(indices).to(self.gather_device)
            correct_all = self.all_gather_and_flatten(knn_correct).to(self.gather_device)
            labels_all = self.all_gather_and_flatten(labels).to(self.gather_device)
            feature_rankme = all_reduce_op(feature_rankme, op=torch.distributed.ReduceOp.AVG)

            if len(index_all) > 0:
                all_knn_correct.index_copy_(0, index_all, correct_all)
                all_labels.index_copy_(0, index_all, labels_all)
                batch_rankme.append(feature_rankme)

        if self.is_distributed:
            torch.distributed.barrier()

        if not torch.all(all_knn_correct > -1):
            print(f'Missing indices: {torch.argwhere(all_knn_correct == -1)}')
            raise RuntimeError('Some kNN results are not gathered correctly.')

        metrics = {
            'knn_correct': all_knn_correct,
            'labels': all_labels,
            'rankme': batch_rankme,
        }

        return metrics


def run(rank: int, world_size: int):
    logging.basicConfig(
        level='INFO',
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler()],
        force=True,
    )
    torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    dataset = ImageNet100Dataset('/home/work/data/imagenet100/val')
    eval_module = EvalModule(
        model=timm.create_model('resnet10t.c3_in1k'),
        train_dataset=dataset,
        eval_dataset=dataset,
        shard_id=rank,
        num_shards=world_size,
        num_workers=4,
        batch_size_per_device=64,
        device='cpu',
    )
    eval_module.evaluate()


if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    world_size = 2
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)
