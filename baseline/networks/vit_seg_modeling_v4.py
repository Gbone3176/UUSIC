# coding=utf-8
from __future__ import absolute_import, division, print_function

import copy
import logging
import math
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2


from datasets.omni_dataset import position_prompt_one_hot_dict
from datasets.omni_dataset import task_prompt_one_hot_dict
from datasets.omni_dataset import type_prompt_one_hot_dict
from datasets.omni_dataset import nature_prompt_one_hot_dict

POSITION_LEN = len(position_prompt_one_hot_dict)
TASK_LEN = len(task_prompt_one_hot_dict)
TYPE_LEN = len(type_prompt_one_hot_dict)
NAT_LEN = len(nature_prompt_one_hot_dict)


logger = logging.getLogger(__name__)

ATTENTION_Q   = "MultiHeadDotProductAttention_1/query"
ATTENTION_K   = "MultiHeadDotProductAttention_1/key"
ATTENTION_V   = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0          = "MlpBlock_3/Dense_0"
FC_1          = "MlpBlock_3/Dense_1"
ATTENTION_NORM= "LayerNorm_0"
MLP_NORM      = "LayerNorm_2"


def np2th(weights, conv: bool = False):
    """Convert numpy weights to torch tensor (or pass-through if already tensor).
       If conv=True, convert HWIO <-> OIHW properly for numpy, and permute for torch tensors."""
    if isinstance(weights, torch.Tensor):
        if conv:
            if weights.ndim != 4:
                raise ValueError(f"Expected 4D conv weights when conv=True, got shape {tuple(weights.shape)}")
            return weights.permute(3, 2, 0, 1).contiguous()
        return weights

    if conv:
        if weights.ndim != 4:
            raise ValueError(f"Expected 4D conv weights (HWIO) when conv=True, got shape {weights.shape}")
        weights = weights.transpose(3, 2, 0, 1)  # HWIO -> OIHW
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish
}

class EarlyHead4(nn.Module):
    """4-way classifier directly on ResNet (hybrid stem) feature."""
    def __init__(self, in_ch: int, mid: int = 256, drop: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm1d(in_ch)
        self.fc1  = nn.Linear(in_ch, mid)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2  = nn.Linear(mid, 4)

    def forward(self, feat: torch.Tensor):
        # feat: (B, C, H, W) from ResNetV2 output
        z = self.pool(feat).flatten(1)  # (B, C)
        z = self.norm(z)
        z = self.fc1(z); z = self.act(z); z = self.drop(z)
        logits4 = self.fc2(z)
        return logits4


class Attention(nn.Module):
    """Multi-head self attention with optional prompt modulation (head-wise FiLM)."""
    def __init__(self, config, vis, prompt: bool = False):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key   = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out         = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout= Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout= Dropout(config.transformer["attention_dropout_rate"])
        self.softmax     = Softmax(dim=-1)

        self.use_prompt  = prompt
        if prompt:
            H, D = self.num_attention_heads, self.attention_head_size
            # per-head learnable scaling parameters for four prompt types
            self.hp_pos    = nn.Parameter(torch.randn(1, H, D))
            self.hp_task   = nn.Parameter(torch.randn(1, H, D))
            self.hp_type   = nn.Parameter(torch.randn(1, H, D))
            self.hp_nature = nn.Parameter(torch.randn(1, H, D))

    def forward(self, hidden_states, prompt_tuple=None):
        # hidden_states: (B, N, C)
        # prompt_tuple: (pos_e, task_e, type_e, nature_e) where each is (B, C) after linear projection
        B, N, C = hidden_states.shape
        H, D = self.num_attention_heads, self.attention_head_size

        q = self.query(hidden_states).view(B, N, H, D).permute(0, 2, 1, 3)  # (B,H,N,D)
        k = self.key(hidden_states)  .view(B, N, H, D).permute(0, 2, 1, 3)  # (B,H,N,D)
        v = self.value(hidden_states).view(B, N, H, D).permute(0, 2, 1, 3)  # (B,H,N,D)

        scale = 1.0 / math.sqrt(D)

        if self.use_prompt and prompt_tuple is not None:
            pos_e, task_e, type_e, nature_e = prompt_tuple  # each is (B, C) or None

            # project (B,C) -> (B,H,D) if possible (when C == H*D); otherwise fallback to slice-repeat
            def to_hd(e):
                if e is None:
                    return None
                if e.dim() != 2 or e.size(0) != B:
                    raise ValueError(f"Prompt embedding must be (B,C), got {tuple(e.shape)}")
                C_in = e.size(1)
                if C_in == H * D:
                    return e.view(B, H, D)
                # fallback: linear map to H*D
                proj = nn.Linear(C_in, H * D).to(e.device)
                with torch.no_grad():
                    nn.init.xavier_uniform_(proj.weight)
                    nn.init.zeros_(proj.bias)
                return proj(e).view(B, H, D)

            pos_hd    = to_hd(pos_e)
            task_hd   = to_hd(task_e)
            type_hd   = to_hd(type_e)
            nature_hd = to_hd(nature_e)

            def mul_or_zero(hp, emb):
                if emb is None or hp is None:
                    return 0.0
                return hp * emb  # (1,H,D) * (B,H,D) -> (B,H,D)

            scale_hd = 1.0 + \
                       mul_or_zero(self.hp_pos,    pos_hd) + \
                       mul_or_zero(self.hp_task,   task_hd) + \
                       mul_or_zero(self.hp_type,   type_hd) + \
                       mul_or_zero(self.hp_nature, nature_hd)  # (B,H,D)
            q = q * scale_hd.unsqueeze(2)  # (B,H,N,D) * (B,H,1,D)

        attn_scores = torch.matmul(q * scale, k.transpose(-1, -2))  # (B,H,N,N)
        attn_probs  = self.softmax(attn_scores)
        weights     = attn_probs if self.vis else None
        attn_probs  = self.attn_dropout(attn_probs)

        context     = torch.matmul(attn_probs, v)  # (B,H,N,D)
        context     = context.permute(0, 2, 1, 3).contiguous().view(B, N, H * D)  # (B,N,C)
        out         = self.out(context)
        out         = self.proj_dropout(out)
        return out, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x); x = self.act_fn(x); x = self.dropout(x)
        x = self.fc2(x); x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet stem
            grid_size = config.patches["grid"]
            # 防止除零：确保 grid_size 正常
            assert grid_size[0] > 0 and grid_size[1] > 0, f"Invalid grid_size: {grid_size}"
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size = (max(patch_size[0], 1), max(patch_size[1], 1))
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        resnet_feat = None
        if self.hybrid:
            x, features = self.hybrid_model(x)
            resnet_feat = x
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden, h, w)
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features ,resnet_feat


class Block(nn.Module):
    def __init__(self, config, vis, prompt: bool = False,
                 pos_len: int = 0, task_len: int = 0, type_len: int = 0, nature_len: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis, prompt=prompt)

        self.use_prompt = prompt
        # one-hot -> hidden 的线性投影
        self.pos_proj    = nn.Linear(pos_len,    self.hidden_size) if (prompt and pos_len    > 0) else None
        self.task_proj   = nn.Linear(task_len,   self.hidden_size) if (prompt and task_len   > 0) else None
        self.type_proj   = nn.Linear(type_len,   self.hidden_size) if (prompt and type_len   > 0) else None
        self.nature_proj = nn.Linear(nature_len, self.hidden_size) if (prompt and nature_len > 0) else None

    def forward(self, x, prompt_tuple=None):
        h = x
        x = self.attention_norm(x)
        if self.use_prompt and prompt_tuple is not None:
            pos_oh, task_oh, type_oh, nature_oh = prompt_tuple  # each (B, Lx) or None

            pos_e    = self.pos_proj(pos_oh)     if (self.pos_proj    is not None and pos_oh    is not None) else None
            task_e   = self.task_proj(task_oh)   if (self.task_proj   is not None and task_oh   is not None) else None
            type_e   = self.type_proj(type_oh)   if (self.type_proj   is not None and type_oh   is not None) else None
            nature_e = self.nature_proj(nature_oh) if (self.nature_proj is not None and nature_oh is not None) else None

            x, weights = self.attn(x, prompt_tuple=(pos_e, task_e, type_e, nature_e))
        else:
            x, weights = self.attn(x, prompt_tuple=None)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    # ------------ weight loading ------------
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight   = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight   = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias   = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias     = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias   = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias     = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0   = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1   = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis, prompt=False, pos_len=0, task_len=0, type_len=0, nature_len=0):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis, prompt=prompt,
                          pos_len=pos_len, task_len=task_len, type_len=type_len, nature_len=nature_len)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, prompt_tuple=None):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states, prompt_tuple=prompt_tuple)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, prompt=False, pos_len=0, task_len=0, type_len=0, nature_len=0):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, prompt=prompt,
                               pos_len=pos_len, task_len=task_len, type_len=type_len, nature_len=nature_len)
        self.use_prompt = prompt

    def forward(self, input_ids):
        if self.use_prompt and isinstance(input_ids, (tuple, list)):
            x, pos_oh, task_oh, type_oh, nature_oh = input_ids
            embedding_output, features, res_feat = self.embeddings(x)
            encoded, attn_weights = self.encoder(embedding_output,
                                                 prompt_tuple=(pos_oh, task_oh, type_oh, nature_oh))
        else:
            x = input_ids
            embedding_output, features, res_feat = self.embeddings(x)
            encoded, attn_weights = self.encoder(embedding_output, prompt_tuple=None)
        return encoded, attn_weights, features, res_feat


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not (use_batchnorm))
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [DecoderBlock(in_ch, out_ch, sk_ch)
                  for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x  # (B, C, H, W)


class MultiTaskClassifier(nn.Module):
    """Stronger classification head for 2-way / 4-way:
       concat[ GAP(tokens), GMP(tokens), GAP(dec_feat) ] -> BN -> MLP -> heads."""
    def __init__(self, hidden_size: int, dec_channels: int, drop: float = 0.2):
        super().__init__()
        in_dim = hidden_size * 2 + dec_channels
        mid = max(256, hidden_size // 2)
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc1  = nn.Linear(in_dim, mid)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2  = nn.Linear(mid, mid)
        self.head2 = nn.Linear(mid, 2)
        self.head4 = nn.Linear(mid, 4)

    def forward(self, tokens: torch.Tensor, dec_feat: torch.Tensor): 
        # tokens: (B, N, H); dec_feat: (B, C, H', W')
        gap_tok = tokens.mean(dim=1)          # (B, H)
        gmp_tok, _ = tokens.max(dim=1)        # (B, H)
        gap_dec = torch.flatten(dec_feat.mean(dim=[2, 3]), 1)  # (B, C)

        z = torch.cat([gap_tok, gmp_tok, gap_dec], dim=1)  # (B, 2H + C)
        z = self.norm(z)
        z = self.fc1(z); z = self.act(z); z = self.drop(z)
        z = self.fc2(z); z = self.act(z)
        return self.head2(z), self.head4(z)

class EarlyClassifierResNet(nn.Module):
    """
    直接基于 ResNet(hybrid stem) 的特征做二分类和四分类：
      GAP -> BN -> FC -> GELU -> Drop -> FC -> GELU -> 两个分支 head2/head4
    """
    def __init__(self, in_ch: int, mid: int = 256, drop: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn   = nn.BatchNorm1d(in_ch)
        self.fc1  = nn.Linear(in_ch, mid)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2  = nn.Linear(mid, mid)

        self.head2 = nn.Linear(mid, 2)
        self.head4 = nn.Linear(mid, 4)

    def forward(self, feat: torch.Tensor):
        # feat: (B, C, H, W) from ResNetV2 output
        z = self.pool(feat).flatten(1)   # (B, C)
        z = self.bn(z)
        z = self.fc1(z); z = self.act(z); z = self.drop(z)
        z = self.fc2(z); z = self.act(z)
        return self.head2(z), self.head4(z)
    

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2, zero_head=False, vis=False,
                 prompt: bool = False, pos_len: int = POSITION_LEN , task_len: int = TASK_LEN, type_len: int = TYPE_LEN, nature_len: int = NAT_LEN):
        super().__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.config = config
        self.use_prompt = prompt

        self.transformer = Transformer(config, img_size, vis,
                                       prompt=prompt,
                                       pos_len=pos_len, task_len=task_len,
                                       type_len=type_len, nature_len=nature_len)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.classifier_head = MultiTaskClassifier(
            hidden_size=config.hidden_size,
            dec_channels=config.decoder_channels[-1],
            drop=0.2
        )

        self.early_classifier = None
        if self.transformer.embeddings.hybrid:
            in_ch_early = self.transformer.embeddings.hybrid_model.width * 16
            self.early_classifier = EarlyClassifierResNet(
                in_ch=in_ch_early,
                mid=max(256, config.hidden_size // 2),
                drop=0.2
            )

    def forward(self, *args):
        """
        支持两种调用：
        - 无 prompt: forward(image)
        - 有 prompt: forward(image, pos_oh, task_oh, type_oh, nature_oh)
                    或 forward((image, pos_oh, task_oh, type_oh, nature_oh))
        """
        if not self.use_prompt:
            # ---- no prompt path ----
            assert len(args) == 1, f"Expected forward(image) when prompt=False, got {len(args)} args"
            image = args[0]
            if image.size(1) == 1:
                image = image.repeat(1, 3, 1, 1)
            x_tokens, attn_weights, features, resnet_feat = self.transformer(image)
        else:
            # ---- prompt path ----
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                image, pos_oh, task_oh, type_oh, nature_oh = args[0]
            else:
                assert len(args) == 5, f"Expected (image, pos_oh, task_oh, type_oh, nature_oh), got {len(args)} args"
                image, pos_oh, task_oh, type_oh, nature_oh = args

            if image.size(1) == 1:
                image = image.repeat(1, 3, 1, 1)

            # Transformer 已支持 tuple 形式的带 prompt 输入
            x_tokens, attn_weights, features, resnet_feat = self.transformer((image, pos_oh, task_oh, type_oh, nature_oh))

        # --- segmentation branch ---
        x_dec = self.decoder(x_tokens, features)
        x_seg = self.segmentation_head(x_dec)

        # --- classification branch ---
        x_cls_2, x_cls_4 = self.classifier_head(x_tokens, x_dec)

        # if self.early_classifier is not None and resnet_feat is not None:
        #     x_cls_2, x_cls_4 = self.early_classifier(resnet_feat)

        return x_seg, x_cls_2, x_cls_4

    # ------------ load npz weights ------------
    def load_from(self, weights):
        with torch.no_grad():
            total_params = 0
            resized_params = 0
            encoder_units_called = 0
            hybrid_units_called  = 0

            def _tick(n=1):
                nonlocal total_params
                total_params += n

            res_weight = weights

            # patch embed
            _tick(); self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            _tick(); self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            # encoder norm
            _tick(); self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            _tick(); self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # position embedding
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                _tick(); self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                _tick(); self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                _tick(); resized_params += 1
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # encoder blocks
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    encoder_units_called += 1
                    unit.load_from(weights, n_block=uname)

            # hybrid stem
            if self.transformer.embeddings.hybrid:
                _tick(); self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias   = np2th(res_weight["gn_root/bias"]).view(-1)
                _tick(); self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                _tick(); self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        hybrid_units_called += 1
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

            print(f"[LOAD SUMMARY] copied_params={total_params}, resized_params={resized_params}, "
                  f"encoder_units_called={encoder_units_called}, hybrid_units_called={hybrid_units_called}")


# ====== exported configs (保持你的原有映射) ======
CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
