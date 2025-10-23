import math
from collections.abc import Container
from itertools import zip_longest

import torch
import torch.nn as nn
from jaxtyping import Float

from tmcl.hints import Task


class TaskSpecificReadout(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_tasks: int,
        num_classes: int,
        *,
        has_bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.has_bias = has_bias

        for i in range(num_tasks):
            self.register_parameter(
                f'weights_{i}', nn.Parameter(torch.empty(num_classes, in_features))
            )
            if has_bias:
                self.register_parameter(f'biases_{i}', nn.Parameter(torch.empty(num_classes)))
        self._current_task = None
        self.reset_parameters()

    @property
    def weights(self) -> list[nn.Parameter]:
        return [self.get_parameter(f'weights_{i}') for i in range(self.num_tasks)]

    @property
    def biases(self) -> list[nn.Parameter]:
        return (
            [self.get_parameter(f'biases_{i}') for i in range(self.num_tasks)]
            if self.has_bias
            else []
        )

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        for w, b in zip_longest(self.weights, self.biases):
            nn.init.uniform_(w, -bound, bound)
            if self.has_bias:
                nn.init.uniform_(b, -bound, bound)

    def set_task_grads(self, t: Task | Container[int]) -> None:
        if t is None:
            t = []
        if isinstance(t, int):
            t = [t]

        for i in range(self.num_tasks):
            self.weights[i].requires_grad = i in t
            if self.has_bias:
                self.biases[i].requires_grad = i in t

    def set_task(self, t: Task, *, update_grads: bool = True) -> None:
        self._current_task = t

        if update_grads:
            self.set_task_grads(t)

    def forward(
        self, x: Float[torch.Tensor, 'batch in_features']
    ) -> Float[torch.Tensor, 'batch num_classes']:
        t = self._current_task
        if t is None:
            raise ValueError('Task not set.')
        if isinstance(t, int):
            return x @ self.weights[t].T + self.biases[t][None, :]
        else:
            cat_w = torch.stack(self.weights)
            cat_b = torch.stack(self.biases)
            return torch.einsum('nd, ncd -> nc', x, cat_w[t]) + cat_b[t]
