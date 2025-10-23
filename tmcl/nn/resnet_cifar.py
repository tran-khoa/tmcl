import torch
from timm.layers import LayerNorm2d
from timm.models import (
    BasicBlock,
    ConvNeXt,
    ResNet,
    build_model_with_cfg,
    checkpoint_seq,
    register_model,
)
from timm.models.convnext import convnext_tiny
from torch import nn
from torchvision.models import resnet18 as torchvision_resnet18
from torchvision.models.resnet import ResNet as torchvision_ResNet


def feature_take_indices(
    num_features: int,
    indices: int | list[int] | None = None,
    as_set: bool = False,
) -> tuple[list[int], int]:
    """Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forwar() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        assert 0 < indices <= num_features, (
            f'last-n ({indices}) is out of range (1 to {num_features})'
        )

        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: list[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            assert 0 <= idx < num_features, (
                f'feature index {idx} is out of range (0 to {num_features - 1})'
            )

            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


class ResNetCIFAR(ResNet):
    """
    ResNet model for CIFAR-10 and CIFAR-100 datasets, adapted from timm.models.resnet.py.
    Removes the first maxpool layer.
    """

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: int | list[int] | None = None,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = 'NCHW',
        intermediates_only: bool = False,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(5, indices)

        # forward pass
        feat_idx = 0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        layer_names = ('layer1', 'layer2', 'layer3', 'layer4')
        if stop_early:
            layer_names = layer_names[:max_index]
        for n in layer_names:
            feat_idx += 1
            x = getattr(self, n)(x)  # won't work with torchscript, but keeps code reasonable, FML
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(
                [self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True
            )
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x


def _create_resnet_cifar(variant, pretrained: bool = False, **kwargs) -> ResNetCIFAR:
    return build_model_with_cfg(ResNetCIFAR, variant, pretrained, **kwargs)


@register_model
def resnet18_cifar(pretrained: bool = False, **kwargs) -> ResNetCIFAR:
    """Constructs a ResNet-18 (CIFAR) model."""

    model_args = dict(block=BasicBlock, layers=(2, 2, 2, 2))
    return _create_resnet_cifar('resnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18_cassle_cifar(pretrained: bool = False, **kwargs) -> ResNet:
    assert not pretrained
    resnet18 = torchvision_resnet18()
    # https://github.com/DonkeyShot21/cassle/blob/b5b0929c3b468cd41740a529d58e92ee4e6ace61/cassle/methods/base.py#L197
    resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    resnet18.maxpool = nn.Identity()
    resnet18.fc = nn.Identity()
    return resnet18


@register_model
def resnet18_cassle(pretrained: bool = False, **kwargs) -> torchvision_ResNet:
    assert not pretrained, 'Pretrained weights are not supported for this model.'
    resnet18 = torchvision_resnet18()
    resnet18.fc = nn.Identity()
    return resnet18


@register_model
def convnext_tiny_cifar(pretrained: bool = False, **kwargs) -> ConvNeXt:
    # according to flop_counter: 12.21M params, 76.7M flops
    # https://github.com/facebookresearch/ConvNeXt/issues/134
    model_dict = dict(
        kernel_sizes=3,  # (3)
        patch_size=2,  # (1)
    )
    cnxt = convnext_tiny(pretrained=pretrained, **dict(model_dict, **kwargs))
    cnxt.stages[3] = torch.nn.Identity()  # (2) remove the last stage

    # https://github.com/facebookresearch/VICRegL/blob/main/convnext.py#L61
    cnxt.norm_pre = LayerNorm2d(384)
    cnxt.num_features = 384
    cnxt.head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten())
    return cnxt
