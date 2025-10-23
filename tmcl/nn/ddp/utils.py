#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch


def convert_to_distributed_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = 'cpu' if not tensor.is_cuda else 'gpu'
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return tensor, orig_device


def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == 'cpu':
        tensor = tensor.cpu()
    return tensor


def is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing mean reduction
    of tensor over all processes.
    """
    return all_reduce_op(
        tensor,
        torch.distributed.ReduceOp.SUM,
        lambda t: t / torch.distributed.get_world_size(),
    )


def all_reduce_op(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    after_op_func: Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper over torch.distributed.all_reduce for performing
    reduction of tensor over all processes in both distributed /
    non-distributed scenarios.
    """
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, op)
        if after_op_func is not None:
            tensor = after_op_func(tensor)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def gather_tensors_from_all(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device) for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]

    return gathered_tensors
