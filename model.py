from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor, BoolTensor
from torch.nn import functional as F
from models.transformer import TransformerBlock, CausalAttention
from models.time2vec import SineActivation, CosineActivation

VALID_T2V_ACTIVATION = ["sin", "cos"]


class QCCT(nn.Module):
    """QUIC Congestion Control Transformer."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_heads: int,
        n_layers: int,
        expand_size: int,
        context_size: int,
        t2v_act: str = "sin",
        act: nn.Module = nn.GELU,
        attention: nn.Module = CausalAttention,
        drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        # 1. Features:
        # 1.1 timestamp
        if t2v_act == "sin":
            self.t2v = SineActivation(1, hidden_size)
        elif t2v_act == "cos":
            self.t2v = CosineActivation(1, hidden_size)
        else:
            raise Exception(f"Unsupported activation:{t2v_act} for time2vec")
        # 1.2 other features
        self.o2v = nn.ModuleList(
            [
                nn.Linear(n_features - 1, expand_size, bias=bias),
                act(),
                nn.Linear(expand_size, hidden_size, bias=bias),
                nn.Dropout(drop),
            ]
        )
        # 1.3 feature dropout
        self.f_drop = nn.Dropout(drop)

        # 2. transformer blocks
        # initialize num_layers of transformer layers
        self.tfm_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=n_heads,
                    context_size=context_size,
                    expand_size=expand_size,
                    attention=attention,
                    act=act,
                    bias=bias,
                    attn_drop=drop,
                    out_drop=drop,
                    ffn_drop=drop,
                )
                for _ in range(n_layers)
            ]
        )

        # 3. output
        self.final = nn.Linear(context_size * hidden_size, 1, bias=bias)

        # 4. init parameters
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        # [Input]: (B, S, C)
        # B: batch_size, S: n_events, C: n_features
        B, S, C = x.shape
        # Step 1: (B, S, C) -> (B, S, D)
        # B: batch_size, S: n_events, D: hidden_size

        # Step 1.1: timestamp
        # (B, S, 1)
        timestamp = x[:, :, 0].unsqueeze(-1)
        # (B, S, D)
        f_ts = self.t2v(timestamp)

        # Step 1.2: other features
        # (B, S, C-1)
        f_others = x[:, :, 1:]
        # (B, S, D)
        for layer in self.o2v:
            f_others = layer(f_others)

        # Step 1.3: Addition
        f_all = self.f_drop(f_ts + f_others)
        B, S, D = f_all.shape

        # Step 2: transformer blocks
        for block in self.tfm_blocks:
            f_all = block(f_all)

        # (B, S, D) -> (B, S * D)
        flattened = f_all.view(B, S * D)

        # Step 3: next congestion control window
        return self.final(flattened)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == "fc2":
                # GPT-2 style FFN init
                torch.nn.init.normal_(
                    module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers)
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
