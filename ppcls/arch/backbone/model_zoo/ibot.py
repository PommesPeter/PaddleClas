# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# reference: https://arxiv.org/abs/2010.11929

from collections.abc import Callable

import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "IBOT_ViT_small_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams",
    "IBOT_ViT_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams",
    "IBOT_ViT_large_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams",
    "IBOT_Swin_tiny_patch7_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch32_384_pretrained.pdparams",
    "IBOT_Swin_tiny_patch14_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class IBOT(nn.Layer):
    def __init__(self,
                 models=None):
        pass


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def IBOT_ViT_small_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768**-0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_ViT_small_patch16_224"],
        use_ssld=use_ssld)
    return model

def IBOT_ViT_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768**-0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_ViT_base_patch16_224"],
        use_ssld=use_ssld)
    return model


def IBOT_ViT_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768**-0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_ViT_large_patch16_224"],
        use_ssld=use_ssld)
    return model


def IBOT_Swin_tiny_patch7_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768**-0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_Swin_tiny_patch7_224"],
        use_ssld=use_ssld)
    return model


def IBOT_Swin_tiny_patch14_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=3,
        qk_scale=768**-0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_Swin_tiny_patch14_224"],
        use_ssld=use_ssld)
    return model