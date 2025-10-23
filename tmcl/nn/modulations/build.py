import logging
from typing import Literal

import timm
from timm.models import ConvNeXt, MobileNetV3
from timm.models._efficientnet_blocks import SqueezeExcite  # noqa
from timm.models.vision_transformer import (
    VisionTransformer,
    # noqa
)
from torch import nn
from torchvision.models import ResNet

from tmcl.nn.modulations.task_modulations import ModulationMappingFn, TaskModulationWrapper

logger = logging.getLogger(__name__)


def vit_map_fn(name: str, module: nn.Module) -> int | Literal[False]:
    """
    Identifies the layers in the Vision Transformer that we want to modulate.
    We modulate the Q, K, V, and projection layers of the attention blocks, as well as the MLP layers.
    """
    if (
        name.endswith('attn.qkv')
        or name.endswith('attn.proj')
        or name.endswith('mlp.fc1')
        or name.endswith('mlp.fc2')
    ):
        assert isinstance(module, nn.Linear)
        return module.out_features
    return False


def convit_map_fn(name: str, module: nn.Module) -> int | Literal[False]:
    """
    We modulate the Q, K, V, and projection layers of the attention blocks, as well as the MLP layers, just like in the Vision Transformer.
    """
    if (
        # ConViT GPSA blocks specific
        name.endswith('attn.qk')
        or name.endswith('attn.v')
        or name.endswith('attn.pos_proj')
        # same as ViT
        or name.endswith('attn.qkv')
        or name.endswith('attn.proj')
        or name.endswith('mlp.fc1')
        or name.endswith('mlp.fc2')
    ):
        assert isinstance(module, nn.Linear)
        return module.out_features
    return False


def resnet_map_fn(name: str, module: nn.Module) -> tuple[int, int] | Literal[False]:
    """
    Identifies the layers in the ResNet that we want to modulate.
    We modulate the BatchNorm2d layers, because the running mean and variance might remove the task modulations.
    Conv2D expects the shape (N C H W), we modulate over the channels C.
    """
    if isinstance(module, nn.BatchNorm2d):
        logger.info('Modulating BatchNorm2d layer: %s', name)
        return module.num_features, 1
    return False


def mobilenet_map_fn(_, module: nn.Module) -> tuple[int, int] | Literal[False]:
    """
    WIP: MobileNetV3 modulation map function.
    """
    if isinstance(module, SqueezeExcite):
        return module.conv_expand.out_channels, 1
    return False


def convnext_map_fn(name: str, module: nn.Module) -> tuple[int, int] | Literal[False]:
    if isinstance(module, nn.Conv2d):
        return module.out_channels, 1
    if isinstance(module, nn.Linear):
        return module.out_features, -1
    return False


def build_tm_model(
    model_name: str,
    *,
    num_tasks: int,
    image_size: int,
    patch_size: int,
    has_bias: bool = True,
    **kwargs,
) -> TaskModulationWrapper:
    """
    Build a timm model and wraps it in a TaskModulationWrapper.

    Parameters
    ----------
    model_name: timm model name
    num_tasks: number of tasks
    image_size: image size, used for ViT
    patch_size: patch size, used for ViT
    has_bias: whether to add a bias to the task modulation layers, otherwise only use gains
    kwargs: arguments to pass to the timm model
    """

    is_convnext = 'convnext' in model_name
    is_convit = 'convit' in model_name
    is_vit = 'vit' in model_name
    is_resnet = 'resnet' in model_name
    is_mobilenetv3 = 'mobilenetv3' in model_name

    model_dict = {}
    map_fn: ModulationMappingFn
    if is_convnext:
        model_dict.update(
            {
                'patch_size': patch_size,
            }
        )
        map_fn = convnext_map_fn
    elif is_convit:
        model_dict.update(
            {
                'img_size': image_size,
                'num_classes': 0,  # no classificaiton head
                'patch_size': patch_size,
            }
        )
        map_fn = convit_map_fn
    elif is_vit:
        model_dict.update(
            {
                'img_size': image_size,
                'num_classes': 0,  # no classificaiton head
                'patch_size': patch_size,
            }
        )
        map_fn = vit_map_fn
    elif is_resnet:
        model_dict['num_classes'] = 0  # no classification head
        map_fn = resnet_map_fn
    elif is_mobilenetv3:
        model_dict['num_classes'] = 0  # no classification head
        map_fn = mobilenet_map_fn
    else:
        raise ValueError(f'Unknown model: {model_name}')
    model_dict.update(kwargs)

    model = timm.create_model(model_name, **model_dict)

    output_dim = None
    if is_vit:
        model: VisionTransformer
        output_dim = model.embed_dim
    elif is_resnet:
        model: ResNet
        output_dim = 512  # pray expansion is 1
    elif is_mobilenetv3:
        model: MobileNetV3
        output_dim = model.num_features
    elif is_convnext:
        model: ConvNeXt
        output_dim = model.num_features

    return TaskModulationWrapper(
        model, map_fn=map_fn, output_dim=output_dim, num_tasks=num_tasks, has_bias=has_bias
    )
