from functools import partial

from timm.models import VisionTransformer, register_model
from timm.models.vision_transformer import vit_small_patch16_384, vit_tiny_patch16_224
from torch import nn


@register_model
def vit_tiny_plus(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Tiny (ViT-Ti/16), but with 12 attention heads."""
    return vit_tiny_patch16_224(pretrained=pretrained, num_heads=12, embed_dim=288, **kwargs)


@register_model
def vit_tiny_12heads(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Tiny (ViT-Ti/16), but with 12 attention heads."""
    return vit_tiny_patch16_224(pretrained=pretrained, num_heads=12, **kwargs)


@register_model
def vit_tiny_half(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """halving heads and layers of vit_tiny_12heads"""
    return vit_tiny_patch16_224(pretrained=pretrained, num_heads=6, depth=6, **kwargs)


@register_model
def vit_small_moco(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ViT-Small (ViT-S/16)"""
    model_args = dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model = vit_small_patch16_384(pretrained=pretrained, **dict(model_args, **kwargs))
    return model
