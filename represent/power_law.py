import einops as eo
import matplotlib.pyplot as plt
import torch
from jaxtyping import Shaped
from matplotlib.figure import Figure
from torch import Tensor


@torch.no_grad()
def cov_spectrum(
    features: Shaped[Tensor, 'batch dim'], *, cov: Shaped[Tensor, 'dim dim'] | None = None
) -> Shaped[Tensor, 'dim']:
    """
    Computes the spectrum of the covariance matrix of the given features.

    Parameters
    ----------
    features: Batched tensor.
    cov: Precomputed covariance matrix, optional.

    Returns
    -------
    The spectrum of the covariance matrix of the given features.
    """
    if cov is None:
        cov: Shaped[Tensor, 'dim dim'] = torch.cov(eo.rearrange(features, 'batch dim -> dim batch'))
    return torch.linalg.eigvalsh(cov)


@torch.no_grad()
def plot_cov_spectrum(
    features: Shaped[Tensor, 'batch dim'],
    *,
    cov: Shaped[Tensor, 'dim dim'] | None = None,
    spectrum: Shaped[Tensor, 'dim'] | None = None,
) -> Figure:
    """
    Plots the spectrum of the covariance matrix of the given features.

    Parameters
    ----------
    features: Batched tensor.
    cov: Precomputed covariance matrix, optional.
    spectrum: Precomputed spectrum, optional.
    """

    if spectrum is None:
        spectrum = cov_spectrum(features, cov=cov)
    spectrum = spectrum.cpu()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(torch.arange(len(spectrum)), spectrum)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    return fig


@torch.no_grad()
def cov_analysis(
    features: Shaped[Tensor, 'batch dim'],
) -> dict[str, Shaped[Tensor, '']]:
    """
    Computes several metrics of the covariance matrix of the given features,
    reusing the covariance matrix computation.

    Parameters
    ----------
    features: Batched tensor.

    Returns
    -------
    """
    cov: Shaped[Tensor, 'dim dim'] = torch.cov(eo.rearrange(features, 'batch dim -> dim batch'))
    spectrum = cov_spectrum(features, cov=cov)

    return {
        'nuclear_norm': spectrum.sum(),
        'max_eigenvalue': spectrum[-1],
        'second_eigenvalue': spectrum[-2] if len(spectrum) >= 2 else torch.nan,
        'third_eigenvalue': spectrum[-3] if len(spectrum) >= 3 else torch.nan,
        'min_eigenvalue': spectrum[0],
    }
