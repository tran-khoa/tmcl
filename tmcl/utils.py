import dataclasses
import logging
import os
import subprocess
import types
import warnings
from collections.abc import Callable
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any

from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember,PyUnresolvedReferences
def disable_optimizer_step_increment(optimizer: LightningOptimizer | Optimizer) -> None:
    """
    Implements this workaround: https://github.com/Lightning-AI/lightning/issues/17958#issuecomment-1720753037
    Used in multi-optimizer per step setups, avoiding increasing the global step counter multiple times per `training_step`.

    These methods are usually injected into the optimizer instance via monkey patching (`_ManualOptimization`).
    """

    def _on_before_step(self: LightningOptimizer) -> None:
        self.trainer.profiler.start('optimizer_step')

    def _on_after_step(self: LightningOptimizer) -> None:
        self.trainer.profiler.stop('optimizer_step')

    _on_before_step._pre_workaround_fn = optimizer._on_before_step
    _on_after_step._pre_workaround_fn = optimizer._on_after_step

    # extract selfs
    before_self = optimizer._on_before_step.__self__
    after_self = optimizer._on_after_step.__self__

    optimizer._on_before_step = types.MethodType(_on_before_step, before_self)
    optimizer._on_after_step = types.MethodType(_on_after_step, after_self)


# noinspection PyProtectedMember,PyUnresolvedReferences
def enable_optimizer_step_increment(optimizer: LightningOptimizer | Optimizer) -> None:
    """
    Reverts the workaround made by `disable_optimizer_step_increment`.
    """
    if not hasattr(optimizer._on_before_step, '_pre_workaround_fn'):
        warnings.warn(
            'Optimizer [before] step increment workaround not found. Skipping.',
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        optimizer._on_before_step = optimizer._on_before_step._pre_workaround_fn

    if not hasattr(optimizer._on_after_step, '_pre_workaround_fn'):
        warnings.warn(
            'Optimizer [after] step increment workaround not found. Skipping.',
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        optimizer._on_after_step = optimizer._on_after_step._pre_workaround_fn


def get_slurm_stdout_path() -> Path | None:
    job_id = os.getenv('SLURM_JOB_ID')
    if not job_id:
        return None

    try:
        result = subprocess.run(
            ['scontrol', 'show', 'job', job_id],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split('\n'):
            if 'StdOut=' in line:
                return Path(line.split('=')[1])
    except CalledProcessError as e:
        logger.warning(f'Error while getting SLURM stdout path: {e}')

    return None


def apply_to_tensors(obj: Any, fn: Callable[[Tensor], Any]) -> Any:
    """
    Recursively applies a function to all tensors in a nested structure.
    """
    if isinstance(obj, Tensor):
        return fn(obj)
    elif isinstance(obj, list | tuple):
        return type(obj)(apply_to_tensors(t, fn) for t in obj)
    elif isinstance(obj, dict):
        return {k: apply_to_tensors(v, fn) for k, v in obj.items()}
    elif dataclasses.is_dataclass(obj):
        return dataclasses.replace(
            obj,
            **{
                field.name: apply_to_tensors(getattr(obj, field.name), fn)
                for field in dataclasses.fields(obj)
            },
        )
    else:
        warnings.warn(
            f'Unknown type {type(obj)} in apply_to_tensors. Skipping.', RuntimeWarning, stacklevel=2
        )
        return obj
