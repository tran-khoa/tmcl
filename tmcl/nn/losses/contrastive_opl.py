import torch
from jaxtyping import Float, Integer
from torch import Tensor
from torchmetrics.functional import pairwise_cosine_similarity


def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def contrastive_opl(
    features: Float[Tensor, 'batch dim'],
    labels: Integer[Tensor, 'batch'],
    *,
    neg_weight: float = 1.0,
    square_loss: bool = False,
) -> tuple[Float[Tensor, ''], dict[str, Float[Tensor, '']]]:
    """
    Assumes label is 0 (negative) or 1 (positive).
    Maximizes cosine similarity for positive samples.
    Minimizes cosine similarity for positive and negative samples.
    """
    pos_mask = labels == 1

    pos_samples = features[pos_mask]
    neg_samples = features[~pos_mask]

    pos_similarities = _off_diagonal(pairwise_cosine_similarity(pos_samples, pos_samples))
    avg_pos_sim = pos_similarities.detach().mean()
    if square_loss:
        pos_loss = (1 - pos_similarities).square().mean()
    else:
        pos_loss = 1 - pos_similarities.mean()

    pos_neg_similarities = pairwise_cosine_similarity(pos_samples, neg_samples)
    avg_pn_sim = pos_neg_similarities.detach().mean()
    if square_loss:
        neg_loss = pos_neg_similarities.square().mean()
    else:
        neg_loss = torch.abs(pos_neg_similarities).mean()

    tl_loss = pos_loss + neg_weight * neg_loss

    return tl_loss, {
        'pos_loss': pos_loss,
        'neg_loss': neg_loss,
        'avg_pos_sim': avg_pos_sim,
        'avg_pn_sim': avg_pn_sim,
        'loss': tl_loss,
    }

