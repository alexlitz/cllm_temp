"""c4_min runtime transformer.

Bare additive residual, ALiBi via additive attention mask, softmax attention,
SwiGLU FFN, NO RMSNorm. Forward matches neural_vm/base_layers.py exactly. The
compiler bakes the ``state_dict``; this module only defines the shapes + forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0  # gain baked directly into W_q/W_k
        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))
        # additive score bias (ALiBi slopes live here); [max_S, max_S]
        self.register_buffer("mask", torch.zeros(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        Q = F.linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)
        K = F.linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
        V = F.linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if self.mask.numel():
            scores = scores + self.mask[:S, :S]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, D)
        return x + F.linear(out, self.W_o)


class FFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.W_up = nn.Parameter(torch.zeros(hidden, dim))
        self.b_up = nn.Parameter(torch.zeros(hidden))
        self.W_gate = nn.Parameter(torch.zeros(hidden, dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden))
        self.W_down = nn.Parameter(torch.zeros(dim, hidden))
        self.b_down = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.linear(x, self.W_up) + self.b_up
        gate = F.linear(x, self.W_gate) + self.b_gate
        hidden = F.silu(up) * gate
        return x + F.linear(hidden, self.W_down, self.b_down)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden: int):
        super().__init__()
        self.attn = Attn(dim, n_heads)
        self.ffn = FFN(dim, hidden)

    def forward(self, x):
        return self.ffn(self.attn(x))


class Transformer(nn.Module):
    """embed(one-hot pos) -> N x Block -> lm_head. No RMSNorm; additive residual."""

    def __init__(self, dim: int, n_heads: int, hidden: int, n_blocks: int,
                 max_pos: int, vocab: int):
        super().__init__()
        self.dim = dim
        self.max_pos = max_pos
        # input token id = step/position index; embedding is a lookup table
        self.embed = nn.Parameter(torch.zeros(max_pos, dim))
        self.blocks = nn.ModuleList(
            Block(dim, n_heads, hidden) for _ in range(n_blocks)
        )
        self.lm_head = nn.Parameter(torch.zeros(vocab, dim))
        self.lm_bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """pos_ids: [B, S] integer step indices. Returns logits [B, S, vocab]."""
        x = self.embed[pos_ids]  # [B, S, D]
        for blk in self.blocks:
            x = blk(x)
        return F.linear(x, self.lm_head, self.lm_bias)
