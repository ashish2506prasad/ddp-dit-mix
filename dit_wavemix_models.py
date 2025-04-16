# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import torch.nn as nn
import functools
from math import ceil
import pywt

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch
import torch.nn.functional as F
from numpy import hamming
from wavemix.wavemix import WaveMixLite
import torch
import torch.nn as nn
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # shape (batch, hiden_dim)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings   # shape (batch, hiden_dim)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTWavemixBlock(nn.Module):
    """
    A DiT adapted with Wavemix block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, in_channels, hidden_size, mlp_ratio=4.0, token_mixer='wav-lite', **block_kwargs):
        super().__init__()
        if token_mixer == 'wav-lite':
          self.attn = WaveMixLite(num_final_channels=hidden_size,  # final channel size is kept 8 because it has to be passed to next waxemix block 
                                  depth=1,
                                  mult=2,
                                  ff_channel=128,
                                  final_dim=128)  # shape = (batch, 4, input_dim, input_dim) = (batch, 4, 32, 32)
        else:
          # other model does not exist as of now
          raise NotImplementedError("Other model is not implemented yet")
        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        # self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)   # this layer norm is removed because Wavemix block already has batch norm
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.conv_insted_of_mlp = nn.Sequential(nn.Conv2d(in_channels=hidden_size, out_channels=in_channels, kernel_size=3, padding=1),
                                                nn.GELU())
        
        # this one is only for timestep and label embedding
        self.adaLN_modulation_layer_1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * in_channels, bias=True)
        )
        self.adaLN_modulation_layer_2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x shape = (batch, 3, 32, 32)
        assert (4, 32, 32) == (x.shape[-3], x.shape[-2], x.shape[-1]), f"x shape {x.shape} is not equal to (x, 4, 32, 32), this size is given in paper"
        # getting shift and scale factors from timestep and label embedding
        shift_msa, scale_msa, gate_mlp = self.adaLN_modulation_layer_1(c).chunk(3, dim=1)
        gate_msa, shift_mlp, scale_mlp = self.adaLN_modulation_layer_2(c).chunk(3, dim=1)

        x = gate_msa.unsqueeze(-1).unsqueeze(-1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # changing mlp to conv layer
        x = self.conv_insted_of_mlp(modulate(x, shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(-1).unsqueeze(-1) * x # this layer norm is removed because Wavemix block already has batch norm
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        # self.norm_final = nn.LayerNorm([4, 32, 32], elementwise_affine=False, eps=1e-6)
        self.norm_final = nn.BatchNorm2d(num_features=4)
        # self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * 4, bias=True)
        )
        self.conv_instead_of_linear = nn.Conv2d(in_channels=4, out_channels=out_channels, kernel_size=3, padding=1)
    def forward(self, x, c):   # c is sum of timestep and lable embedding
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.conv_instead_of_linear(x)
        return x


class DiTWaveMix(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        # patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        # num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # self.patch_size = patch_size
        # self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTWavemixBlock(in_channels, hidden_size, mlp_ratio=mlp_ratio, token_mixer='wav-lite') for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation_layer_1[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_layer_1[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_layer_2[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_layer_2[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv_instead_of_linear.weight, 0)
        nn.init.constant_(self.final_layer.conv_instead_of_linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # print("embedding + pos_embedding", x.shape)
        t = self.t_embedder(t)                   # (N, D)
        # print("t_embedder", t.shape)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        # print("post attention", x.shape)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        # print("final layer", x.shape)
        # x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_WaveMix_XL(**kwargs):
    return DiTWaveMix(depth=32, hidden_size=1024, **kwargs)

def DiT_WaveMix_L(**kwargs):
    return DiTWaveMix(depth=28, hidden_size=512, **kwargs)

def DiT_WaveMix_B(**kwargs):
    return DiTWaveMix(depth=24, hidden_size=256, **kwargs)

def DiT_WaveMix_S(**kwargs):
    return DiTWaveMix(depth=16, hidden_size=128, **kwargs)



DiT_WaveMix_models = {
    'DiT-WaveMix-XL': DiT_WaveMix_XL,  
    'DiT-WaveMix-L':  DiT_WaveMix_L,   
    'DiT-WaveMix-B':  DiT_WaveMix_B,   
    'DiT-WaveMix-S':  DiT_WaveMix_S,  
}

if __name__ == "__main__":
    from prettytable import PrettyTable
    
    # tiny image net has 200 classes
    model = DiT_WaveMix_L(num_classes=20)

    ############### uncomment the below lines to test for a random input ###################
    # x = torch.randn(2, 4, 32, 32)
    # t = torch.randint(0, 1000, (2,))
    # y = torch.randint(0, 200, (2,))
    # out = model(x, t, y)
    # print(out.shape)

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
        
    count_parameters(model)

