import copy
import itertools
import logging
from collections.abc import Callable, Container
from itertools import zip_longest
from typing import Any, Final, Literal

import torch
from jaxtyping import Float
from torch import Tensor, nn

from tmcl.hints import TaskMixture

logger = logging.getLogger(__name__)

ModulationMappingFn = Callable[[str, nn.Module], Literal[False] | int | tuple[int, int]]

ModulatableOps: Final[tuple[type, ...]] = (
    nn.Linear,
    nn.BatchNorm2d,  # ResNet
    nn.Conv2d,  # ConvNeXt
)


class TaskModulation[T](nn.Module):
    def __init__(
        self,
        op: T,
        op_dim: int,
        *,
        num_tasks: int,
        has_bias: bool = True,
        default_initializer: Literal['constant', 'normal'] = 'normal',
        axis: int = -1,
    ):
        """
        Task modulation layer for a single operation.

        Parameters
        ----------
        op: a ModulatableOp
        op_dim: the output dimension of the operation at the specified axis
        num_tasks: number of tasks
        has_bias: whether to use bias modulations
        default_initializer: initialization method for the modulation parameters
        axis: axis to modulate along, -1 for the last axis
        """
        super().__init__()
        if not isinstance(op, ModulatableOps):
            raise TypeError(f'Unsupported op type: {type(op)}')

        self.op = op
        self._op_dim = op_dim

        self._num_tasks = num_tasks
        self._has_bias = has_bias
        self._default_initializer = default_initializer
        self._axis = axis

        self._gains = [nn.Parameter(torch.empty((op_dim,))) for _ in range(num_tasks)]
        self._biases = []
        if has_bias:
            self._biases = [nn.Parameter(torch.empty((op_dim,))) for _ in range(num_tasks)]

        for task_idx, gain in enumerate(self._gains):
            self.register_parameter(f'gains_task{task_idx}', gain)
        for task_idx, bias in enumerate(self._biases):
            self.register_parameter(f'biases_task{task_idx}', bias)

        self._current_task = None
        self.reset_parameters()

    @property
    def no_mod_idx(self) -> int:
        """
        In order to allow to specify no modulation in a task batch, we add a virtual task index, which succeeds the last real task.
        """
        return self._num_tasks

    @property
    def current_task(self) -> TaskMixture:
        return self._current_task

    @property
    def op_dim(self) -> int:
        return self._op_dim

    @property
    def gains(self) -> list[nn.Parameter]:
        return list(self._gains)

    @property
    def biases(self) -> list[nn.Parameter]:
        return list(self._biases)

    @property
    def gain_matrix(self) -> Float[Tensor, 'num_tasks+1 ... dim']:
        """
        Matrix of gains [num_tasks + 1, op_dim]
        where `num_tasks` is the virtual task for no modulation.
        """
        return torch.stack([*self._gains, torch.zeros_like(self._gains[0], requires_grad=False)])

    @property
    def bias_matrix(self) -> Float[Tensor, 'num_tasks+1 ... dim'] | None:
        """
        Matrix of biases [num_tasks + 1, op_dim]
        where `num_tasks` is the virtual task for no modulation.
        """
        if not self._has_bias:
            return None
        return torch.stack([*self._biases, torch.zeros_like(self._biases[0], requires_grad=False)])

    def reset_parameters(
        self,
        *,
        tasks: Container[int] | None = None,
        initializer: Literal['constant', 'normal'] | None = None,
    ) -> None:
        if initializer is None:
            initializer = self._default_initializer

        for task, (g, b) in enumerate(zip_longest(self._gains, self._biases)):
            if tasks is not None and task not in tasks:
                continue
            match initializer:
                case 'constant':
                    nn.init.zeros_(g)
                    if b is not None:
                        nn.init.zeros_(b)
                case 'normal':
                    nn.init.normal_(g, mean=0, std=0.02)
                    if b is not None:
                        nn.init.normal_(b, std=0.02)
                case _:
                    raise ValueError(f'Unknown init: {initializer}')

    def set_num_tasks(self, num_tasks: int) -> None:
        """
        Sets the number of tasks for the task modulation layer, adding or removing as necessary.
        """
        if num_tasks == self._num_tasks:
            return
        elif num_tasks < self._num_tasks:
            for task_idx in range(num_tasks, self._num_tasks):
                delattr(self, f'gains_task{task_idx}')
                if self._has_bias:
                    delattr(self, f'biases_task{task_idx}')
            self._gains = self._gains[:num_tasks]
            self._biases = self._biases[:num_tasks]
        else:
            for task_idx in range(self._num_tasks, num_tasks):
                gain = nn.Parameter(torch.empty((self._op_dim,)))
                self.register_parameter(f'gains_task{task_idx}', gain)
                self._gains.append(gain)
                if self._has_bias:
                    bias = nn.Parameter(torch.empty((self._op_dim,)))
                    self.register_parameter(f'biases_task{task_idx}', bias)
                    self._biases.append(bias)
        self._num_tasks = num_tasks

    def set_task(self, t: TaskMixture, *, update_grads: bool = True) -> None:
        """
        Sets the current task for the task modulation layer.
        """
        self._current_task = t

        if update_grads:
            self.set_task_grads(t)

    def set_task_grads(self, t: TaskMixture | Container[int]) -> None:
        """
        Sets requires_grad for modulation parameters for the specified tasks to True, all others to False.
        """
        if t is None:
            t = []
        if isinstance(t, int):
            t = [t]
        if isinstance(t, Tensor) and t.dim() == 2:
            t = t.sum(dim=0).argwhere()

        for i in range(self._num_tasks):
            self._gains[i].requires_grad = i in t
            if self._has_bias:
                self._biases[i].requires_grad = i in t

    def forward(self, x: Float[Tensor, 'batch ... dim']) -> Float[Tensor, 'batch ... dim']:
        """
        Computes the forward pass of the operation and applies the task modulation based on the currently set task.
        """
        op_output = self.op(x)

        if isinstance(self._current_task, Tensor):
            assert op_output.shape[0] == self._current_task.shape[0]

        if self._current_task is None:
            return op_output

        axis = self._axis
        if self._axis == -1:
            axis = op_output.ndim - 1
        new_shape = [1 if i != axis else self._op_dim for i in range(op_output.ndim)]
        # (1, ... , 1, D, 1, ...)

        if isinstance(self._current_task, int):
            g = self._gains[self._current_task].view(*new_shape)
            b = self._biases[self._current_task].view(*new_shape) if self._has_bias else 0
        else:
            if self._current_task.dim() == 2:
                # task mixture
                # t: (batch, num_tasks)
                batch_size, num_tasks = self._current_task.shape
                new_shape[0] = batch_size

                g = self._current_task.view(batch_size, num_tasks, 1) * self.gain_matrix[
                    :-1, :
                ].view(1, num_tasks, self._op_dim)
                g = g.sum(dim=1).view(*new_shape)

                b = (
                    (
                        self._current_task.view(batch_size, num_tasks, 1)
                        * self.gain_matrix[:-1, :].view(1, num_tasks, self._op_dim)
                    )
                    .sum(dim=1)
                    .view(*new_shape)
                    if self._has_bias
                    else 0
                )
            else:
                new_shape[0] = op_output.shape[0]  # new_shape = (N, ..., 1, D, 1, ...)

                g = self.gain_matrix[self._current_task].view(*new_shape)
                b = self.bias_matrix[self._current_task].view(*new_shape) if self._has_bias else 0

        return (1 + g) * op_output + b

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.op, item)


class TaskModulationWrapper[T](nn.Module):
    def __init__(
        self,
        base_module: T,
        map_fn: ModulationMappingFn,
        *,
        output_dim: int,
        num_tasks: int,
        has_bias: bool = True,
        default_initializer: Literal['constant', 'normal'] = 'normal',
    ):
        """
        Wraps operations in a module with task modulation layers, as defined by the map function.

        Parameters
        ----------
        base_module: some nn.Module, assumes the output is (batch, ..., dim)
        map_fn: function to map module names to modulation dimensions
        output_dim: final output dimension of the module
        num_tasks: number of tasks
        has_bias: whether to add a bias to the task modulation layers, otherwise only use gains

        """
        super().__init__()
        self.module = copy.deepcopy(base_module)
        self.output_dim = output_dim

        self.num_tasks = num_tasks
        self._has_bias = has_bias

        self._feedforward_parameters = dict(self.module.named_parameters())
        self._modulations = []
        self._modulated_ops = []
        for name, potential_op in self.module.named_modules():
            if (map_info := map_fn(name, potential_op)) is not False:
                if isinstance(map_info, int):
                    dim = map_info
                    axis = -1
                else:
                    dim, axis = map_info

                logger.info(f'Adding task modulation to {name}, dim={dim}, axis={axis}.')
                self._modulations.append(
                    (
                        name,
                        TaskModulation(
                            potential_op,
                            dim,
                            num_tasks=num_tasks,
                            has_bias=has_bias,
                            default_initializer=default_initializer,
                            axis=axis,
                        ),
                    )
                )
                self._modulated_ops.append(potential_op)
        for name, new_module in reversed(self._modulations):
            path = name.split('.')
            _m = self.module
            for subname in path[:-1]:
                _m = getattr(_m, subname)
            setattr(_m, path[-1], new_module)

        self._modulation_parameters = [
            param
            for _, modulation in self._modulations
            for param in modulation.parameters(recurse=False)
        ]

    @property
    def modulations(self) -> list[tuple[str, TaskModulation]]:
        """
        A list of (op_path, TaskModulation) pairs.
        """
        return list(self._modulations)

    @property
    def no_mod_idx(self) -> int:
        """
        In order to allow to specify no modulation in a task batch, we add a virtual task index, which succeeds the last real task.
        """
        return self.num_tasks

    @property
    def feedforward_parameters(self) -> dict[str, nn.Parameter]:
        """
        A view of named feedforward parameters of the base module.
        """
        return dict(self._feedforward_parameters)

    @property
    def modulation_parameters(self) -> list[nn.Parameter]:
        """
        A view of modulation parameters of the base module.
        """
        return list(self._modulation_parameters)

    @property
    def modulated_ops(self) -> list[nn.Module]:
        """
        A view of modulated operations of the base module, i.e. one of `ModulatableOps`.
        """
        return list(self._modulated_ops)

    def task_modulation_parameters(self, *tasks: int) -> list[nn.Parameter]:
        """
        Returns a view of modulation parameters for the specified tasks.
        """
        tasks = set(tasks)
        return [
            param
            for _, modulation in self._modulations
            for idx, param in itertools.chain(
                enumerate(modulation.gains), enumerate(modulation.biases)
            )
            if idx in tasks
        ]

    def reset_modulations(
        self,
        *,
        tasks: Container[int] | None = None,
        initializer: Literal['constant', 'normal'] | None = None,
    ) -> None:
        """
        Resets the parameters of the task modulation layers.

        Parameters
        ----------
        tasks: task indices to reset, if None, resets all
        initializer: initialization method, either 'constant' or 'normal'
              if None, uses the initial method defined in the modulation layers.
        """
        for _, modulation in self._modulations:
            modulation.reset_parameters(tasks=tasks, initializer=initializer)

    def set_num_tasks(self, num_tasks: int) -> None:
        """
        Sets the number of tasks for the task modulation layers, adding or removing as necessary.
        """
        logger.info(f'Setting number of tasks from {self.num_tasks} to {num_tasks}.')
        for _, modulation in self._modulations:
            modulation.set_num_tasks(num_tasks)

        self.num_tasks = num_tasks

    def set_task(self, t: TaskMixture, *, update_grads: bool = True) -> None:
        """
        Sets the current task for the task modulation layers.

        Parameters
        ----------
        t: task index or tensor of task indices or None
        update_grads: whether to update the gradients of the task modulation layers
        """
        for _, modulation in self._modulations:
            modulation.set_task(t, update_grads=update_grads)

    def set_task_grads(self, t: TaskMixture | Container[int]) -> None:
        """
        Sets requires_grad for modulation parameters for the specified tasks to True, all others to False.
        """
        for _, modulation in self._modulations:
            modulation.set_task_grads(t)

    def set_feedforward_grads(self, enable_grads: bool) -> None:
        """
        Sets requires_grad for feedforward parameters.
        """
        for p in self._feedforward_parameters.values():
            p.requires_grad = enable_grads

    def forward(self, *args, **kwargs) -> Any:
        """
        Passes the input through the base module with task modulation
        """
        return self.module(*args, **kwargs)

    @property
    def modulations_per_task(self) -> int:
        """
        The number of modulations per task.
        """
        bias_factor = 2 if self._has_bias else 1
        return sum(modulation.op_dim for _, modulation in self._modulations) * bias_factor
