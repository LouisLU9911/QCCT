from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expand_size: int,
        act: nn.Module = nn.GELU,
        drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # project input to expanded dimension
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)

        # activation function to introduce non-linearity
        self.act = act()

        # project back to the input dimension
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)

        # optional dropout layer to prevent overfitting
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor):
        x = self.fc1(x)  # apply first linear layer
        x = self.act(x)  # apply activation function
        x = self.fc2(x)  # apply second linear layer
        x = self.drop(x)  # optionally apply dropout layer
        return x


class CausalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # input dimension must be divisible by num_heads
        assert hidden_size % num_heads == 0
        # number of Attention heads
        self.nh = num_heads

        # linear layer to project queries, keys, values
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)

        # attention dropout layer to prevent overfitting
        self.attn_drop = nn.Dropout(attn_drop)

        # linear layer to project final output
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        # final output dropout layer to prevent overfitting
        self.out_drop = nn.Dropout(out_drop)

        # causal mask to ensure that Attention is not applied to future tokens where
        # context_size is the maximum sequence length of the transformer
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones([context_size, context_size], dtype=torch.bool), diagonal=1
            ).view(1, 1, context_size, context_size),
            persistent=False,
        )

    # boolean `mask` of shape (batch_size, sequence_length)
    # where True is masked and False is unmasked
    def forward(self, x: Tensor, mask: BoolTensor | None = None):
        # batch size, sequence length, input dimension
        B, S, C = x.shape

        # split into queries, keys, & values of shape
        # batch size (B), num_heads (NH), sequence length (S), head size (HS)
        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C // self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        # dot product queries and keys for each head
        # (B, NH, S, S) = (B, NH, S, HS) @ (B, NH, HS, S)
        attn = q @ k.transpose(-2, -1)

        # scale by square root of output dimension
        attn = attn / math.sqrt(k.size(-1))

        # apply input and causal mask
        combined_mask = self.causal_mask[:, :, :S, :S]
        if mask is not None:
            combined_mask += mask.view(B, 1, 1, S)
        attn = attn.masked_fill(combined_mask, float("-inf"))

        # apply softmax to get attention weights
        attn = attn.softmax(dim=-1)

        # apply dropout to attention weight
        attn = self.attn_drop(attn)

        # dot product attention weights with values of shape
        # (B, NH, S, HS) = (B, NH, S, S) @ (B, NH, HS, S)
        x = attn @ v

        # and transpose heads & sequence and reshape back to (B, S, C)
        x = x.transpose(1, 2).reshape(B, S, C)

        # apply final linear layer and dropout to get output (B, S, C)
        return self.out_drop(self.Wo(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        expand_size: int,
        attention: nn.Module = CausalAttention,
        act: nn.Module = nn.GELU,
        attn_drop: float = 0.1,
        out_drop: float = 0.1,
        ffn_drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # first pre-norm layer
        self.norm1 = nn.LayerNorm(hidden_size)
        # initialize the attention layer
        self.attn = attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            context_size=context_size,
            attn_drop=attn_drop,
            out_drop=out_drop,
            bias=bias,
        )

        # second pre-norm layer
        self.norm2 = nn.LayerNorm(hidden_size)
        # initialize the feed forward network (MLP)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            expand_size=expand_size,
            act=act,
            drop=ffn_drop,
            bias=bias,
        )

    def forward(self, x: Tensor):
        # normalize input then add residual to attention output
        x = x + self.attn(self.norm1(x))

        # normalize input then add residual to feedforward output
        return x + self.ffn(self.norm2(x))
