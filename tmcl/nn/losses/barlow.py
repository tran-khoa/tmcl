from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from tmcl.hints import Loss, Scalar
from tmcl.nn.ddp import SyncNormalizeFunction


class BarlowTwinsLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_param: float = 5e-3,
        gather_distributed: bool = False,
        sync_normalize: bool = False,
        normalize_eps: float = 1e-5,
        scale_loss: float = 0.024,
        norm_strategy: Literal['joint', 'per-branch'] = 'joint',
    ):
        """
        Barlow Twins, but with multiple views, as proposed in:
            [0] K. K. Agrawal, A. Ghosh, A. Oberman, and B. Richards, “Addressing Sample Inefficiency in Multi-View Representation Learning” (2023).


        Args:
            lambda_param:
                Parameter for importance of redundancy reduction term.
                Defaults to 5e-3.
            sync_normalize:
                If True, then the mean and variances are synchronized across all gpus.
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are
                gathered and summed before the loss calculation.
            normalize_eps:
                Epsilon value for normalization.
            norm_strategy:
                Whether to compute the normalization statistics per branch or over all branches.
                For now, SyncBatchNorm only applies to 'per-branch'.
        """
        super().__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed
        self.normalize_eps = normalize_eps
        self.sync_normalize = sync_normalize
        self.scale_loss = scale_loss
        self.norm_strategy = norm_strategy

        if gather_distributed and not dist.is_available():
            raise ValueError(
                'gather_distributed is True but torch.distributed is not available. '
                'Please set gather_distributed=False or install a torch version with '
                'distributed support.'
            )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[Loss, dict[str, Scalar]]:
        N = z_a.size(0)
        D = z_a.size(1)

        # normalize repr. along the batch dimension
        if not self.sync_normalize and self.norm_strategy == 'per-branch':
            z_a_norm, *_ = _normalize([z_a])
            z_b_norm, *_ = _normalize([z_b])
        elif not self.sync_normalize and self.norm_strategy == 'joint':
            z_a_norm, z_b_norm = _normalize([z_a, z_b])
        elif self.sync_normalize and self.norm_strategy == 'per-branch':
            # let's instead follow PNR code (this branch was not used for C100 anyways)
            bn = torch.nn.BatchNorm1d(D, affine=False).to(z_a.device)
            z_a_norm = bn(z_a)
            z_b_norm = bn(z_b)
        else:
            assert self.sync_normalize and self.norm_strategy == 'joint'
            z_a_norm, z_b_norm = SyncNormalizeFunction.apply(
                torch.cat([z_a, z_b], 0), self.normalize_eps
            ).split(N, dim=0)

        # cross-correlation matrix
        c = z_a_norm.T @ z_b_norm
        c.div_(N)

        # sum cross-correlation matrix between multiple gpus
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum() * self.scale_loss
        redundancy_reduction_loss = _off_diagonal(c).pow_(2).sum() * self.scale_loss
        loss = invariance_loss + self.lambda_param * redundancy_reduction_loss

        return loss, {
            'invariance': invariance_loss,
            'redundancy': redundancy_reduction_loss,
            'loss': loss,
        }


def _normalize(
    zs: list[Float[Tensor, 'batch dim']] | tuple[Float[Tensor, 'batch dim'], ...],
) -> list[Float[Tensor, 'batch dim']]:
    """Helper function to normalize tensors along the batch dimension."""
    combined = torch.stack(zs, dim=0)  # Shape: A x N x D
    normalized = F.batch_norm(
        combined.flatten(0, 1),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
    ).view_as(combined)
    return list(normalized)


def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
