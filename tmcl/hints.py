from typing import Protocol

from jaxtyping import Float, Integer
from torch import Tensor, nn

"""
Axis naming convention:
    B: batch size
    C: channels
    H: height
    W: width
    L: sequence length
    D: embedding dimension
    T: number of tasks
"""

type BatchId = int
type ImageBatch = Float[Tensor, 'B C H W']
type FeatureBatch = Float[Tensor, 'B D']
type ClassBatch = Integer[Tensor, 'B']
type Loss = Float[Tensor, '']
type Scalar = Float[Tensor, '']
type Task = Integer[Tensor, 'batch'] | int | None
type TaskMixture = Float[Tensor, 'batch tasks'] | Task


class GenericLossFn[BatchT, ModuleT: nn.Module](Protocol):
    def __call__(self, batch: BatchT, model: ModuleT) -> Loss: ...


class GenericFeatLossFn[BatchT](Protocol):
    def __call__(self, features: FeatureBatch, batch: BatchT) -> Loss: ...


class GenericForwardFn[BatchT, ModuleT: nn.Module](Protocol):
    def __call__(self, batch: BatchT, model: ModuleT) -> FeatureBatch: ...
