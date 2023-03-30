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
from .vision_transformer import VisionTransformer, Identity, trunc_normal_
from .swin_transformer_v2 import SwinTransformerV2

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
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

class IBOTHead(nn.Layer):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 norm=None, 
                 act='gelu', 
                 last_norm=None, 
                 num_layers=3, 
                 hidden_dim=2048, 
                 bottleneck_dim=256, 
                 norm_last_layer=True,
                 epsilon=1e-5,
                 **kwargs):
        super().__init__()
        
        self.act = eval(act)()
        if norm is not None:
            self.norm = eval(norm_layer)(out_dim, epsilon=epsilon)
        if last_norm is not None:
            self.last_norm = eval(norm_layer)(out_dim, epsilon=epsilon)
            
        self.num_layers = max(num_layers, 1)
        if num_layers == 1:
            if bottleneck_dim > 0:
                self.mlp = nn.Linear(in_dim, bottleneck_dim)
            else:
                self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            if bottleneck_dim > 0:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            else:
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)
        
        if bottleneck_dim > 0:
            self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        else:
            self.last_layer = None
            
    def forward(self, x):
        pass


class IBOTVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 return_all_token=False,
                 masked_im_modeling=False
                 **kwargs):
        super(IBOTVisionTransformer, self).__init__(
            img_size,
            patch_size,
            in_chans,
            class_num,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            epsilon,
            **kwargs
        )
        self.return_all_token = return_all_token
        self.masked_im_modeling = masked_im_modeling
        
        self.head = IBOTHead()
        
        if self.masked_im_modeling:
            self.masked_embed = self.create_parameter(shape=(1, embed_dim), default_initializer=zeros_)
        
    def forward_features(self, x, mask=None, return_all_tokens=None):
        # B = x.shape[0]
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        
        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(perm=[0, 2, 1])
        
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand((B, -1, -1)).astype(x.dtype)
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
            
        if return_all_tokens:
            return x
        
        return x[:, 0]
    
    def forward(self, x, mask):
        x = self.forward_features(x, mask, return_all_tokens=self.return_all_tokens)
        x = self.head(x)
        
        return x
    

class IBOTSwinTransformer(SwinTransformerV2):
    def __init__(self,
                 img_size=256,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 pretrained_window_sizes=[0, 0, 0, 0],
                 return_all_tokens=False,
                 masked_im_modeling=False
                 **kwargs):
        super(IBOTSwinTransformer, self).__init__(
            img_size,
            patch_size,
            in_chans,
            class_num,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            pretrained_window_sizes,
            **kwargs
        )
        self.return_all_token = return_all_token
        self.masked_im_modeling = masked_im_modeling
        
        if self.masked_im_modeling:
            self.masked_embed = self.create_parameter(shape=(1, embed_dim), default_initializer=zeros_)
        
        
    def forward_features(self, x, mask=None, return_all_tokens=None):
        x = self.patch_embed(x)
        
        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(perm=[0, 2, 1])
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose([0, 2, 1]))  # B C 1
        x = paddle.flatten(x, 1)
        
        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return torch.cat([x.unsqueeze(1), x_region], dim=1)

        return x
    
    def forward(self, x, mask=None):
        x = self.forward_features(x, mask, self.return_all_tokens)
        
        return x
    
    def mask_model(self, x, mask):
        # extend mask for hierarchical features
        if x.shape[-2:] != mask.shape[-2:]:
            htimes, wtimes = np.array(x.shape[-2:]) // np.array(mask.shape[-2:])
            mask = mask.repeat_interleave(htimes, -2).repeat_interleave(wtimes, -1)
        
        # mask embed
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)

        return x


def _load_pretrained(pretrained,
                     model, 
                     model_url, 
                     use_ssld=False,
                     use_imagenet22k_pretrained=False,
                     use_imagenet22kto1k_pretrained=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(
            model, 
            model_url,
            use_ssld=use_ssld,
            use_imagenet22k_pretrained=use_imagenet22k_pretrained,
            use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def IBOT_ViT_small_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = IBOTVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qk_scale=(384 // 6) **-0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_ViT_small_patch16_224"],
        use_ssld=use_ssld)
    return model

def IBOT_ViT_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = IBOTVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qk_scale=(768 // 12) ** -0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_ViT_base_patch16_224"],
        use_ssld=use_ssld)
    return model


def IBOT_ViT_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = IBOTVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qk_scale=(1024 // 12) ** -0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_ViT_large_patch16_224"],
        use_ssld=use_ssld)
    return model


def IBOT_Swin_tiny_windows7_224(pretrained=False, use_ssld=False, **kwargs):
    model = IBOTSwinTransformer(
        img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_Swin_tiny_patch7_224"],
        use_ssld=use_ssld)
    return model


def IBOT_Swin_tiny_windows14_224(pretrained=False, use_ssld=False, **kwargs):
    model = IBOTSwinTransformer(
        img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["IBOT_Swin_tiny_patch14_224"],
        use_ssld=use_ssld)
    return model