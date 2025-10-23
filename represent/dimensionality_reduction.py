import warnings

from jaxtyping import Integer, Shaped
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.random import RandomState
from torch import Tensor

try:
    from cuml import UMAP as GPU_UMAP
except ImportError:
    GPU_UMAP = None

try:
    from umap import UMAP as CPU_UMAP
except ImportError as err:
    CPU_UMAP = None

    if GPU_UMAP is None:
        raise ImportError(
            'Could not import umap. '
            'Please install '
            '`umap-learn` (CPU) '
            'or cuML (GPU, see https://docs.rapids.ai/install).'
        ) from err


def umap(
    features: Shaped[Tensor, 'batch dim'],
    *,
    labels: Integer[Tensor, 'batch'] | None = None,
    n_neighbors: int = 15,
    n_components: int = 2,
    metric: str = 'euclidean',
    metric_kwds: dict[str, float] | None = None,
    n_epochs: int | None = None,
    learning_rate: float = 1.0,
    init: str = 'spectral',
    min_dist: float = 0.1,
    spread: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
    repulsion_strength: float = 1.0,
    negative_sample_rate: int = 5,
    transform_queue_size: float = 4.0,
    a: float = None,
    b: float = None,
    random_state: int | RandomState | None = None,
    verbose: bool = False,
) -> tuple[CPU_UMAP | GPU_UMAP, Shaped[Tensor, 'batch 2']]:
    if features.is_cuda:
        if GPU_UMAP is None:
            warnings.warn(
                'Using CPU UMAP. Install cuML for GPU UMAP, see https://docs.rapids.ai/install.',
                stacklevel=1,
            )
            features = features.cpu()
            umap_cls = CPU_UMAP
        else:
            umap_cls = GPU_UMAP
    else:
        umap_cls = CPU_UMAP

    # noinspection PyCallingNonCallable
    umap = umap_cls(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        metric_kwds=metric_kwds,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        init=init,
        min_dist=min_dist,
        spread=spread,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        transform_queue_size=transform_queue_size,
        a=a,
        b=b,
        random_state=random_state,
        verbose=verbose,
    )
    projected = umap.fit_transform(features, y=labels)
    return umap, projected


def plot_umap(
    features: Shaped[Tensor, 'batch dim'],
    *,
    labels: Integer[Tensor, 'batch'] | None = None,
    supervised_umap: bool = False,
    **umap_kwargs,
) -> Figure:
    if supervised_umap and labels is None:
        raise ValueError('labels must be provided for supervised UMAP')

    _, projected = umap(features, labels=labels if supervised_umap else None, **umap_kwargs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(projected[:, 0], projected[:, 1], c=labels)
    return fig
