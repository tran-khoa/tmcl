import math
import warnings

import einops as eo
import torch
from jaxtyping import Shaped
from torch import Tensor


@torch.no_grad()
def within_class_variability(
    features: list[Shaped[Tensor, 'batch dim']],
    *,
    global_mean: Shaped[Tensor, 'dim'] | None = None,
    class_means: list[Shaped[Tensor, 'dim']] | None = None,
    threshold_cluster_samples: int = 1,
) -> Shaped[Tensor, '']:
    """
    Computes the within-class variability for a set of labelled features by
    "measuring the magnitude of the between-class covariance matrix compared
    to the within-class covariance matrix".
    Described as a measure of "neural collapse" [1].

    [1] Zhu et al., 2021: A Geometric Analysis of Neural Collapse with Unconstrained Features, NeurIPS 2021

    Parameters
    ----------
    features: Per-class batched features.
    global_mean: Global mean of all features. if None, it is computed from the features.
    class_means: Per-class mean of all features. if None, it is computed from the features.
    threshold_cluster_samples: Minimum samples per class, otherwise, the class is skipped.

    Returns
    -------
    The within-class variability or NaN if no clusters with more than `threshold_cluster_samples` sample were found.
    """
    if global_mean is None:
        global_mean = torch.cat(features, dim=0).mean(dim=0)
    if class_means is None:
        class_means = [feature.mean(dim=0) for feature in features]

    dim = global_mean.shape[0]

    within_class_cov = torch.zeros((dim, dim), dtype=global_mean.dtype, device=global_mean.device)
    between_class_cov = torch.zeros((dim, dim), dtype=global_mean.dtype, device=global_mean.device)

    n_samples_used = 0
    n_clusters_used = 0
    for c_features, c_mean in zip(features, class_means, strict=True):
        if len(c_features) > threshold_cluster_samples:
            c_features_cnorm = c_features - c_mean[None, :]
            c_cov = eo.einsum(c_features_cnorm, c_features_cnorm, 'n d1, n d2 -> n d1 d2')
            c_cov = eo.einsum(c_cov, 'n d1 d2 -> d1 d2')
            within_class_cov += c_cov

            c_features_gnorm = c_features - global_mean[None, :]
            c_cov = eo.einsum(c_features_gnorm, c_features_gnorm, 'n d1, n d2 -> n d1 d2')
            c_cov = eo.einsum(c_cov, 'n d1 d2 -> d1 d2')
            between_class_cov += c_cov
            n_samples_used += len(c_features)
            n_clusters_used += 1

    if n_clusters_used == 0:
        warnings.warn(
            'No clusters with more than one sample found.',
            UserWarning,
            stacklevel=1,
        )
        return torch.tensor(torch.nan, dtype=global_mean.dtype, device=global_mean.device)

    within_class_cov /= n_samples_used
    between_class_cov /= n_clusters_used

    # Numerically stable computation (W @ B+)
    # (B^T)+ @ W^T = (B+)^T @ W^T = (W@B+)^T
    cov_prod = torch.linalg.lstsq(between_class_cov.T, within_class_cov.T).solution.T

    return torch.trace(cov_prod) / n_clusters_used


@torch.no_grad()
def classifier_etf_convergence(
    classifier: Shaped[Tensor, 'classes dim'],
) -> Shaped[Tensor, '']:
    """
    Computes the "Convergence of the Learned Classifier W to a Simplex ETF".
    Described as a measure of "neural collapse" [1].

    [1] Zhu et al., 2021: A Geometric Analysis of Neural Collapse with Unconstrained Features, NeurIPS 2021

    Parameters
    ----------
    classifier: Weight matrix of the classifier/readout.

    Returns
    -------
    A measure of convergence of the classifier to a Simplex ETF.
    """
    W_outer = eo.einsum(classifier, classifier, 'K1 d, K2 d -> K1 K2')
    W_outer /= torch.linalg.norm(W_outer)

    K = W_outer.shape[0]
    setf = (1 / math.sqrt(K - 1)) * (torch.eye(K, device=classifier.device) - (1 / K))
    setf /= torch.linalg.norm(setf)

    return torch.linalg.norm(W_outer - setf)


@torch.no_grad()
def self_duality(
    features: list[Shaped[Tensor, 'batch dim']] | None,
    classifier: Shaped[Tensor, 'classes dim'],
    *,
    global_mean: Shaped[Tensor, 'dim'] | None = None,
    class_means: list[Shaped[Tensor, 'dim']] | None = None,
) -> Shaped[Tensor, '']:
    """
    Computes the "Convergence to Self-duality".
    Described as a measure of "neural collapse" [1].

    [1] Zhu et al., 2021: A Geometric Analysis of Neural Collapse with Unconstrained Features, NeurIPS 2021

    Parameters
    ----------
    features: Per-class batched features. Set to None if global_mean and class_means are given.
    classifier: Weight matrix of the classifier/readout.
    global_mean: Global mean of all features. iif None, it is computed from the features.
    class_means: Per-class mean of all features. iif None, it is computed from the features.

    Returns
    -------
    A measure of self-duality of the classifier.
    """
    if features is None and (global_mean is None or class_means is None):
        raise ValueError('Either features or (global_mean and_class_means) must be given.')

    if global_mean is None:
        global_mean = torch.cat(features, dim=0).mean(dim=0)

    if class_means is None:
        class_means = [feature.mean(dim=0) for feature in features]

    centered_class_means = torch.stack(class_means, dim=1) - global_mean[:, None]
    projs = classifier @ centered_class_means
    projs /= torch.linalg.norm(projs)

    K = len(class_means)
    setf = (1 / math.sqrt(K - 1)) * (torch.eye(K, device=projs.device) - (1 / K))
    setf /= torch.linalg.norm(setf)

    return torch.linalg.norm(projs - setf)


@torch.no_grad()
def bias_collapse(
    features: Shaped[Tensor, 'batch dim'] | None,
    classifier_weights: Shaped[Tensor, 'classes dim'],
    classifier_bias: Shaped[Tensor, 'classes'],
    *,
    global_mean: Shaped[Tensor, 'dim'] | None = None,
) -> Shaped[Tensor, '']:
    """
    Computes the "bias collapse".
    Described as a measure of "neural collapse" [1].

    [1] Zhu et al., 2021: A Geometric Analysis of Neural Collapse with Unconstrained Features, NeurIPS 2021

    Parameters
    ----------
    features: Batched features. Set to None if global_mean is given.
    classifier_weights: Weight matrix of the classifier/readout.
    classifier_bias: Bias vector of the classifier/readout.
    global_mean: Global mean of all features. If None, it is computed from the features.

    Returns
    -------
    A measure of bias collapse of the classifier.
    """
    if features is None and global_mean is None:
        raise ValueError('Either features or global_mean must be given.')

    if global_mean is None:
        global_mean = features.mean(dim=0)

    return torch.linalg.norm(classifier_weights @ global_mean + classifier_bias)


@torch.no_grad()
def class_distance_normalized_variance(
    features: list[Shaped[Tensor, 'batch dim']],
    *,
    class_means: list[Shaped[Tensor, 'classes dim']] | None = None,
) -> Shaped[Tensor, '']:
    """
    Computes the "class-distance normalized variance" (CDNV).
    Described as a measure of "neural collapse" [1].

    [1] Galanti et al., 2022: On the Role of Neural Collapse in Transfer Learning, ICLR 2022

    Parameters
    ----------
    features: Per-class batched features.
    class_means: Per-class mean of all features. If None, it is computed from the features.

    Returns
    -------
    The average CDNV over all disjoint class pairs.
    """
    if class_means is None:
        class_means = [feature.mean(dim=0) for feature in features]
    class_means = torch.stack(class_means, dim=0)

    denom: Shaped[Tensor, 'classes classes'] = (
        2 * torch.cdist(class_means, class_means, p=2).square()
    )

    class_vars = [
        (class_features - class_mean[None, :]).square().sum(dim=1).mean(dim=0)
        for class_features, class_mean in zip(features, class_means, strict=True)
    ]
    class_vars = torch.stack(class_vars, dim=0).repeat(len(class_vars), 1)
    numerator: Shaped[Tensor, 'classes classes'] = class_vars + class_vars.T

    cdnvs = (numerator / denom)[
        ~torch.eye(len(class_means), device=class_means.device, dtype=torch.bool)
    ]
    return cdnvs.mean()


@torch.no_grad()
def neural_collapse_metrics(
    features: list[Shaped[Tensor, 'batch dim']],
    classifier_weights: Shaped[Tensor, 'classes dim'],
    classifier_bias: Shaped[Tensor, 'classes'],
) -> dict[str, Shaped[Tensor, '']]:
    """
    Computes all implemented measures of "neural collapse" [1, 2],
    reusing intermediate results.

    [1] Zhu et al., 2021: A Geometric Analysis of Neural Collapse with Unconstrained Features, NeurIPS 2021
    [2] Galanti et al., 2022: On the Role of Neural Collapse in Transfer Learning, ICLR 2022
    """

    global_mean = torch.cat(features, dim=0).mean(dim=0)
    class_means = [feature.mean(dim=0) for feature in features]

    return {
        'within_class_variability': within_class_variability(
            features=features,
            global_mean=global_mean,
            class_means=class_means,
        ),
        'classifier_etf_convergence': classifier_etf_convergence(
            classifier=classifier_weights,
        ),
        'self_duality': self_duality(
            features=features,
            classifier=classifier_weights,
            global_mean=global_mean,
            class_means=class_means,
        ),
        'bias_collapse': bias_collapse(
            features=torch.cat(features, dim=0),
            classifier_weights=classifier_weights,
            classifier_bias=classifier_bias,
            global_mean=global_mean,
        ),
        'class_distance_normalized_variance': class_distance_normalized_variance(
            features=features,
            class_means=class_means,
        ),
    }
