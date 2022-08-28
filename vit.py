import torch
import torch.nn as nn
from functools import partial
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model
import numpy as np
import sys
import torch.nn.functional as F


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        ### FFN neuron pruning
        self.gate = torch.ones(hidden_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        ### FFN neuron pruning
        self.neuron_output = x # B x N x C
        mask = self.gate.float().to(x.get_device())
        x.mul_(mask.view(1, 1, self.hidden_features))

        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ### MSA head pruning
        self.gate = torch.ones(self.num_heads)

        ### token pruning
        self.token_prune_ratio = 0

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # batch x head x seq x embed_chunk

        attn = (q @ k.transpose(-2, -1)) * self.scale # batch x head x seq x seq'
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_output = attn[:,:,0,:] # batch x head x seq
        x = (attn @ v) # batch x head x seq x embed_chunk
        x = x.transpose(1, 2) # batch x seq x head x embed_chunk

        # MSA head pruning
        self.head_output = x # batch x seq x head x embed_chunk
        mask = self.gate.float().to(x.get_device())
        x.mul_(mask.view(1, 1, self.num_heads, 1))

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_output.detach(), attn.detach(), [q.detach(), k.detach(), v.detach()], self.token_prune_ratio


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.ema_cls_attn = None

    def forward(self, x, train_label=None):

        ### token pruning
        x_, attn_output, full_attn, qkv, token_prune_ratio = self.attn(self.norm1(x))
        x = x + self.drop_path(x_)

        if token_prune_ratio != 0:
            avg_cls_attn = torch.mean(attn_output, dim=1) # batch x seq
            avg_cls_attn[:,0] = 1e9 # ensure [CLS] be always the top
            sorted_cls_attn, idx = torch.sort(avg_cls_attn, dim=1, descending=True) # batch x seq
            K = int(x.shape[1] * (1 - token_prune_ratio))
            topk_attn, topk_idx = sorted_cls_attn[:,:K], idx[:,:K] # batch x seq'
            non_topk_attn, non_topk_idx = sorted_cls_attn[:,K:], idx[:,K:] # batch x seq''
            attentive_tokens = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])) # batch x seq' x embed
            inattentive_tokens = torch.gather(x, 1, non_topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])) # batch x seq'' x embed
            fused_token = torch.mean(inattentive_tokens, dim=1, keepdim=True) # batch x 1 x embed
            x = torch.cat([attentive_tokens, fused_token], dim=1) # batch x (seq'+1) x embed

        # if token_prune_ratio != 0:
        #     # v: batch x head x seq x d
        #     # train_label: batch x class
        #     v = qkv[-1]
        #     v = v.transpose(1,2).reshape(x.shape).transpose(0,1).detach() # seq x batch x embed
        #     y = train_label.float().detach() # batch x 1000
        #     assert y.shape == (x.shape[0], 1000)
        #     v_kernel = Center(batch=True)((GaussianKernel(sigma=1)(v) + GaussianKernel(sigma=2)(v) + GaussianKernel(sigma=4)(v) + GaussianKernel(sigma=8)(v) + GaussianKernel(sigma=16)(v))/5) # seq x batch x batch
        #     y_kernel = Center(batch=False)(LinearKernel()(y)) # batch x batch
        #     hsic = torch.matmul(v_kernel, y_kernel.unsqueeze(0)) # seq x batch x batch
        #     avg_cls_attn = hsic.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) # seq
        #     avg_cls_attn = (avg_cls_attn - torch.min(avg_cls_attn)) / (torch.max(avg_cls_attn) - torch.min(avg_cls_attn)) # seq
        #     avg_cls_attn[0] = 1e9 # ensure [CLS] always kept
        #     if self.training:
        #         if self.ema_cls_attn is None:
        #             self.ema_cls_attn = avg_cls_attn
        #         else:
        #             self.ema_cls_attn = self.ema_cls_attn * 0.996 + avg_cls_attn * (1 - 0.996) # bootstrapping
        #     if self.ema_cls_attn is None:
        #         self.ema_cls_attn = avg_cls_attn
        #         avg_cls_attn = self.ema_cls_attn.unsqueeze(0).expand(x.shape[0], x.shape[1])*torch.mean(attn_output, dim=1) # batch x seq
        #     else:
        #         avg_cls_attn = self.ema_cls_attn.unsqueeze(0).expand(x.shape[0], x.shape[1])*torch.mean(attn_output, dim=1) # batch x seq
        #     sorted_cls_attn, idx = torch.sort(avg_cls_attn, dim=1, descending=True) # batch x seq
        #     K = int(x.shape[1] * (1 - token_prune_ratio))
        #     topk_attn, topk_idx = sorted_cls_attn[:,:K], idx[:,:K] # batch x seq'
        #     non_topk_attn, non_topk_idx = sorted_cls_attn[:,K:], idx[:,K:] # batch x seq''
        #     attentive_tokens = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])) # batch x seq' x embed
        #     inattentive_tokens = torch.gather(x, 1, non_topk_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])) # batch x seq'' x embed
        #     fused_token = torch.mean(inattentive_tokens, dim=1, keepdim=True) # batch x 1 x embed
        #     x = torch.cat([attentive_tokens, fused_token], dim=1) # batch x (seq'+1) x embed

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, train_label=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, train_label)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, train_label=None):
        x = self.forward_features(x, train_label)
        x = self.head(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
