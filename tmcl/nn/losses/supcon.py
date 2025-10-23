import torch
from einops import repeat
from jaxtyping import Bool, Integer
from torch import Tensor


@torch.no_grad()
def supcon_positive_mask(
    labels: Integer[Tensor, 'batch'], num_views: int
) -> Bool[Tensor, 'views*batch views*batch']:
    labels_matrix = repeat(labels, 'b -> c (d b)', c=num_views * labels.size(0), d=num_views)
    labels_matrix = (labels_matrix == labels_matrix.t()).fill_diagonal_(False)
    return labels_matrix
