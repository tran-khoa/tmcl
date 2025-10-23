from typing import Literal, cast

import einops as eo
import torch
import torch.distributed as dist
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from tmcl.hints import Loss, Scalar
from tmcl.nn.ddp.utils import all_reduce_mean


class MultiViewBarlowTwinsLoss(torch.nn.Module):
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
            proj_dim:
                Dimension of the projection head outputs.
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
        self.scale_loss = scale_loss
        self.sync_normalize = sync_normalize
        self.norm_strategy = norm_strategy

        if gather_distributed and not dist.is_available():
            raise ValueError(
                'gather_distributed is True but torch.distributed is not available. '
                'Please set gather_distributed=False or install a torch version with '
                'distributed support.'
            )

    def forward(self, *zs: Float[Tensor, 'batch dim']) -> tuple[Loss, dict[str, Scalar]]:
        batch_size, dim = zs[0].size()

        if self.sync_normalize:
            bn = torch.nn.BatchNorm1d(dim, affine=False).to(zs[0].device)
            zs = [bn(z) for z in zs]
            z_mean = torch.stack(zs, dim=0).mean(dim=0)

            losses = []
            for z in zs:
                corr = torch.einsum('bi, bj -> ij', z, z_mean) / batch_size

                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(corr)
                    corr /= dist.get_world_size()

                diag = torch.eye(dim, device=z.device)
                cdif = (corr - diag).pow(2)
                cdif[~diag.bool()] *= self.lambda_param
                loss = self.scale_loss * cdif.sum()
                losses.append(loss)
            return sum(losses), {f'view_{i}': loss.item() for i, loss in enumerate(losses)}

        # normalize repr. along the batch dimension
        zs_norm: Float[Tensor, 'augm batch dim']
        zs_norm = cast(Tensor, eo.rearrange(_normalize(zs), 'augm batch dim -> augm batch dim'))

        expected_pos = eo.reduce(zs_norm, 'augm batch dim -> batch dim', 'mean')

        invariance_loss: Float[Tensor, ''] | float = 0.0
        redundancy_reduction_loss: Float[Tensor, ''] | float = 0.0

        for z_norm in zs_norm:
            c = torch.mm(expected_pos.T, z_norm) / batch_size
            c = all_reduce_mean(c)
            invariance_loss += torch.diagonal(c).add_(-1).pow_(2).sum() * self.scale_loss
            redundancy_reduction_loss += _off_diagonal(c).pow_(2).sum() * self.scale_loss
        loss = invariance_loss + self.lambda_param * redundancy_reduction_loss

        return loss, {
            'invariance': invariance_loss,
            'redundancy': redundancy_reduction_loss,
        }


class GhoshBarlowLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_param: float = 5e-3,
        gather_distributed: bool = False,
        sync_normalize: bool = False,
        normalize_eps: float = 1e-5,
        scale_loss: float = 0.024,
        norm_strategy: None = None,
    ):
        """
        Barlow Twins, but with multiple views, as proposed in:
            [0] K. K. Agrawal, A. Ghosh, A. Oberman, and B. Richards, “Addressing Sample Inefficiency in Multi-View Representation Learning” (2023).


        Args:
            proj_dim:
                Dimension of the projection head outputs.
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
                no-op
        """
        super().__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed
        self.normalize_eps = normalize_eps
        self.scale_loss = scale_loss
        self.sync_normalize = sync_normalize

        if gather_distributed and not dist.is_available():
            raise ValueError(
                'gather_distributed is True but torch.distributed is not available. '
                'Please set gather_distributed=False or install a torch version with '
                'distributed support.'
            )

    def forward(self, *zs: Float[Tensor, 'batch dim']) -> tuple[Loss, dict[str, Scalar]]:
        batch_size, dim = zs[0].size()
        num_views = len(zs)

        if self.sync_normalize:
            bn = torch.nn.BatchNorm1d(dim, affine=False).to(zs[0].device)
        else:
            bn = lambda z: _normalize([z])[0]
        zs = [bn(z) for z in zs]  # normalized zs
        z_mean = torch.stack(zs, dim=0).mean(dim=0)  # average repr

        on_diag = (torch.stack(zs) * z_mean.unsqueeze(0)).mean(1)
        if self.gather_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(on_diag)
            on_diag /= dist.get_world_size()
        inv_loss = on_diag.add_(-1).pow_(2).sum() * self.scale_loss / num_views

        cov_matrix = torch.mm(z_mean.T, z_mean) / batch_size
        if self.gather_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(cov_matrix)
            cov_matrix /= dist.get_world_size()
        cov_loss = _off_diagonal(cov_matrix).pow_(2).sum() * self.scale_loss

        loss = inv_loss + self.lambda_param * cov_loss
        return loss, {'inv': inv_loss, 'cov': cov_loss}


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
