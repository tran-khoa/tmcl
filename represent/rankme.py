import traceback
import warnings

import torch.linalg
from jaxtyping import Shaped
from torch import Tensor
from torch._C import _LinAlgError  # noqa


@torch.no_grad()
def rankme(
    x: Shaped[Tensor, 'batch dim'],
    eps: float = 1e-6,
) -> Shaped[Tensor, '']:
    """
    Implements RankMe measure [1].

    [1] Garrido et al., 2023: RankMe: Assessing the Downstream Performance of
    Pretrained Self-Supervised Representations by Their Rank, PMLR 2023

    Parameters
    ----------
        x: Embedding matrix, shape (N, D).
        eps: Epsilon value to avoid log by zero.

    Returns
    -------
    The soft rank estimation or NaN if SVD fails (degenerated features).
    """

    try:
        if x.is_cuda:
            singular_values = torch.linalg.svdvals(x.float(), driver='gesvdj')
        else:
            singular_values = torch.linalg.svdvals(x.float())
        sv_norm = torch.linalg.norm(singular_values, ord=1)

        probs = singular_values / sv_norm + eps
        rank_me = torch.exp(-(probs * probs.log()).sum())
        return rank_me
    except _LinAlgError:
        warnings.warn(
            'LinAlgError occured computing SVD on given features.',
            UserWarning,
            stacklevel=1,
        )
        traceback.print_exc()

        return torch.tensor(torch.nan, dtype=x.dtype, device=x.device)
