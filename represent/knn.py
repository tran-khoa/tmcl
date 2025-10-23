import einops as eo
import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer, Shaped
from torch import Tensor

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb


def knn_predict(
    feature: Shaped[Tensor, 'batch dim'],
    feature_bank: Shaped[Tensor, 'batch dim'],
    feature_labels: Integer[Tensor, 'batch'],
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
    normalize_feature_bank: bool = False,
    return_scores: bool = False,
) -> Integer[Tensor, 'batch classes'] | Float[Tensor, 'batch classes']:
    """
    Adapted from the lightly library.
    Run kNN predictions on features based on a feature bank

    This method is commonly used to monitor performance of self-supervised
    learning methods.

    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.

    Args:
        feature:
            Tensor with shape (B, D) for which you want predictions.
        feature_bank:
            Tensor of shape (N, D) of a database of features used for kNN.
        feature_labels:
            Labels with shape (N,) for the features in the feature_bank.
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10).
        knn_k:
            Number of k neighbors used for kNN.
        knn_t:
            Temperature parameter to reweights similarities for kNN.
        normalize_feature_bank:
            Whether to normalize the feature bank before computing similarities.
        return_scores:
            Whether to return the scores of the kNN predictions.

    Returns:
        A tensor containing the kNN predictions
    """

    # l2-normalize the feature vectors
    feature = F.normalize(feature, dim=1)
    if normalize_feature_bank:
        feature_bank = F.normalize(feature_bank, dim=1)

    sim_matrix = eo.einsum(feature_bank, feature, 'n d, b d -> b n')
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels: Integer[Tensor, 'batch k'] = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label: Float[Tensor, 'batch*k num_classes'] = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # (B*K, C)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> (B, C)
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    if return_scores:
        return pred_scores

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


if __name__ == '__main__':
    features = torch.randn(10, 128)
    feature_bank = torch.randn(100, 128)
    feature_labels = torch.randint(0, 10, (100,))
    num_classes = 10
    knn_predict(features, feature_bank, feature_labels, num_classes, knn_k=5)
