"""Spec-faithful runtime transformer for the BLOG_SPEC foundation.

This is the *vanilla* decode-only transformer the blog post specifies
(``docs/BLOG_SPEC.md`` §Vanillaness, §The Attention Layer). It differs from
``c4_min/model.py`` in exactly the two ways the spec demands and the old
``model.py`` deviated on:

  * **softmax1** attention (``exp(x) / (1 + sum exp(x))``, the "ZFOD" softmax the
    spec calls its only real deviation) instead of plain ``F.softmax``. §491.
  * **ALiBi** additive positional bias computed *inside* ``forward`` from head
    slopes (a geometric sequence, §307-311), not a pre-baked static mask.

Everything else is a bog-standard decode-only transformer: a **token
embedding** table (``vocab x dim`` — the input is a token id stream, the
standard autoregressive interface, NOT a one-hot-position lookup), additive
residual, SwiGLU FFN (§467-478), causal mask, and an LM head. The forward is
copy-paste of the ``PureAttention`` / ``PureFFN`` reference in BLOG_SPEC lines
240-350.

No RMSNorm (the c4 weights are hand-baked so LayerNorm-free residual streams
are what the gadgets expect); the spec's reference classes are also norm-free.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with ``+1`` in the denominator (BLOG_SPEC §491).

    ``exp(x) / (1 + sum exp(x))`` — the extra 1 acts as an always-present
    "attention sink" so a query that matches *nothing* attends to nothing
    (weights sum to <1, the residual is preserved). This is what gives the VM
    zero-fill-on-demand: an unwritten memory address / register reads 0.
    """
    m = x.max(dim=dim, keepdim=True)[0]
    m = torch.clamp(m, min=0.0)  # keep the implicit 0-logit sink in range
    exp_x = torch.exp(x - m)
    denom = torch.exp(-m) + exp_x.sum(dim=dim, keepdim=True)
    return exp_x / denom


class Attn(nn.Module):
    """Multi-head attention with softmax1 + ALiBi (BLOG_SPEC §284-350)."""

    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))
        # ALiBi slopes: geometric sequence 2^(-8/n*(i+1)) per head (§307-311).
        slopes = torch.tensor(
            [2.0 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)]
        )
        self.register_buffer("alibi_slopes", slopes)

    def forward(self, x: torch.Tensor, past_kv=None, q_positions=None,
                use_cache: bool = False):
        """Multi-head softmax1 + ALiBi attention.

        Default path (``past_kv is None``, ``q_positions is None``,
        ``use_cache=False``) is **byte-identical** to the un-cached spec forward:
        the new tokens sit at absolute positions ``0..S-1`` and attend causally
        over themselves.

        Incremental / cached path (``past_kv`` given):
          * ``x`` is only the NEW query tokens (``[B, Snew, D]``); ``q_positions``
            gives their ABSOLUTE sequence positions (a ``[Snew]`` long tensor).
          * ``past_kv = (K_cache, V_cache, pos_cache)`` holds the K/V of the
            already-seen tokens (``[B, H, Sc, HD]`` + ``[Sc]`` positions).
          * K/V are computed for the new tokens only, concatenated with the cache,
            and the query attends over the union.  ALiBi distance uses ABSOLUTE
            positions (``|q_pos - k_pos|``) so the bias is identical to the
            un-cached forward, and softmax1's ``+1`` sink is preserved verbatim.

        Returns ``out`` (default) or ``(out, (K_all, V_all, pos_all))`` when
        ``use_cache`` — the updated per-block cache for the next step.
        """
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        Q = F.linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)
        Knew = F.linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
        Vnew = F.linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)

        # Absolute positions of the query rows.
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)

        # Assemble the full K/V (cache + new) and their absolute positions.
        if past_kv is not None:
            K_cache, V_cache, pos_cache = past_kv
            K = torch.cat([K_cache, Knew], dim=2)
            V = torch.cat([V_cache, Vnew], dim=2)
            k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
        else:
            K, V, k_pos = Knew, Vnew, q_pos

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if past_kv is None and q_positions is None:
            # -- default (un-cached) path: byte-identical to the original spec --
            pos = torch.arange(S, device=x.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
            scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist
            causal = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device), diagonal=1
            )
            scores = scores + causal
        else:
            # -- cached / windowed path: ALiBi + causal over ABSOLUTE positions --
            dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs().float()
            scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0)
            mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))     # [Sq, Sk]
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = softmax1(scores, dim=-1)          # §491 softmax1 (ZFOD)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, D)
        out = x + F.linear(out, self.W_o)
        if use_cache:
            return out, (K, V, k_pos)
        return out


class FFN(nn.Module):
    """SwiGLU FFN with additive residual (BLOG_SPEC §467-478)."""

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
        hidden = F.silu(up) * gate               # SiLU(up @ x) * (gate @ x)
        return x + F.linear(hidden, self.W_down, self.b_down)


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden: int, max_seq_len: int):
        super().__init__()
        self.attn = Attn(dim, n_heads, max_seq_len)
        self.ffn = FFN(dim, hidden)

    def forward(self, x, past_kv=None, q_positions=None, use_cache: bool = False):
        if use_cache or past_kv is not None or q_positions is not None:
            a, new_kv = self.attn(x, past_kv=past_kv, q_positions=q_positions,
                                  use_cache=True)
            out = self.ffn(a)
            return (out, new_kv) if use_cache else out
        return self.ffn(self.attn(x))


class Transformer(nn.Module):
    """Standard decode-only autoregressive transformer.

    ``embed`` is a TOKEN embedding (``vocab x dim``): the input is a stream of
    token ids and the model is stepped by the ordinary autoregressive loop
    (``blogspec_run.generate``). The register/nibble semantics live entirely in
    the baked embedding rows + FFN/attention weights — the architecture itself
    is vanilla.
    """

    def __init__(self, dim: int, n_heads: int, hidden: int, n_blocks: int,
                 vocab: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.embed = nn.Parameter(torch.zeros(vocab, dim))     # token -> residual
        self.blocks = nn.ModuleList(
            Block(dim, n_heads, hidden, max_seq_len) for _ in range(n_blocks)
        )
        self.lm_head = nn.Parameter(torch.zeros(vocab, dim))
        self.lm_bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, S] token ids. Returns logits [B, S, vocab]."""
        x = self.embed[tokens]
        for blk in self.blocks:
            x = blk(x)
        return F.linear(x, self.lm_head, self.lm_bias)

    def forward_hidden_cached(self, x, past_key_values=None, q_positions=None,
                              use_cache: bool = False):
        """Run the BLOCK stack (no LM head) with an optional per-block KV cache.

        ``x`` is the pre-embedded / overlaid residual for the NEW query rows only
        (``[B, Snew, D]``); ``q_positions`` gives their absolute positions.
        ``past_key_values`` is a list (one entry per block) of ``(K, V, pos)``
        caches, or ``None`` for the first step.  Returns ``(hidden, new_caches)``
        where ``hidden`` is the block-stack output for the query rows and
        ``new_caches`` is the updated per-block cache list.

        The register/nibble decode reads ``hidden[:, -1]`` exactly as the naive
        driver reads ``model.forward``'s last row — this only changes HOW the
        block stack is evaluated (incremental K/V), not WHAT it computes.
        """
        n = len(self.blocks)
        if past_key_values is None:
            past_key_values = [None] * n
        new_caches = []
        for blk, pkv in zip(self.blocks, past_key_values):
            x, kv = blk(x, past_kv=pkv, q_positions=q_positions, use_cache=True)
            new_caches.append(kv)
        return x, new_caches
