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

Architectural toggles (the SAME core model, three orthogonal axes)
==================================================================
The blog spec notes the c4 gadgets are equivalent under several standard
architectural substitutions ("everything I did with ALiBi could be done a
different way with RoPE ... chalk it up to taste", §235; softmax1 is "the only
real deviation" but reproducible with a plain-softmax **BOS sink**; the weights
are norm-free but survive **RMSNorm** with a compensator lane). This module
exposes those three axes as *config toggles* on the ONE core model — each
defaults to the CURRENT behaviour, so the default build is **byte-identical** to
the historical spec forward:

  * ``positional`` ∈ {``"alibi"``, ``"rope"``} — additive recency bias (§307)
    vs. a RoPE **binary-distance** recency (``theta_k = 2^k``, §743, from
    ``nibble_rope``). RoPE substitutes for the *positional/recency* term ONLY;
    the content match (the baked ±smag address CAM, §410) is left untouched, so
    the address argmax is preserved and RoPE only reproduces the ALiBi
    latest-write-wins ordering (nearer store = higher score).
  * ``sink`` ∈ {``"softmax1"``, ``"bos_sink"``} — the ``+1`` softmax1 sink
    (§491) vs. a prepended BOS **sink column** (score 0 → ``exp(0)=1``, value 0)
    under plain softmax. Plain-softmax over ``[sink, real...]`` reproduces
    softmax1 over ``[real...]`` EXACTLY (the ``+1`` in the denominator IS the
    sink's ``exp(0)``), so the attention output is identical.
  * ``norm`` ∈ {``"none"``, ``"rmsnorm"``} — norm-free (current) vs. RMSNorm on
    the attn/mlp inputs with the **compensator-lane trick**: a NORM_COMPENSATOR
    dim holds a large constant ``K``, every RMSNorm weight is ``K/√dim``, so
    ``x / rms · weight ≈ x`` on the real dims (identity) while the compensator
    absorbs the normalisation (``qwen_embed.rmsnorm_identity_gamma``). Requires
    the builder to carry ``K`` on the compensator dim of every live embedding
    row (``Transformer.set_norm_compensator`` / the ``compensator_dim`` config).

The combination ``positional="rope", norm="rmsnorm", sink="bos_sink"`` is the
**Qwen-compatible mode** (RoPE + RMSNorm + plain-softmax) — the config you would
use to embed the VM into a real Qwen2 (see ``qwen_embed``).
"""
from __future__ import annotations

import math
import os as _os_env

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


def softmax1_via_bos_sink(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reproduce :func:`softmax1` with a plain softmax over a prepended BOS sink.

    softmax1's ``exp(x)/(1+sum exp(x))`` is *exactly* a plain softmax over the
    augmented logit vector ``[0, x_0, x_1, ...]`` (the leading ``0`` is the
    sink whose ``exp(0)=1`` supplies the ``+1`` in the denominator), then
    DROPPING the sink's weight. The sink's value row is 0 (it contributes
    nothing to the attention output), so this yields the identical per-token
    weights as :func:`softmax1` — the plain-softmax realisation of the ZFOD sink
    the spec's BOS token provides (§Vanillaness, ``qwen_embed`` §2). Bit-for-bit
    identical modulo the fp reduction order (same ``max``-subtracted stable form).
    """
    sink = torch.zeros_like(scores.narrow(dim, 0, 1))  # a single 0-logit column
    aug = torch.cat([sink, scores], dim=dim)
    m = aug.max(dim=dim, keepdim=True)[0]
    m = torch.clamp(m, min=0.0)
    exp_aug = torch.exp(aug - m)
    denom = exp_aug.sum(dim=dim, keepdim=True)
    w = exp_aug / denom
    # drop the sink column: the real-token weights are the softmax1 weights.
    return w.narrow(dim, 1, scores.shape[dim])


class RMSNorm(nn.Module):
    """RMSNorm ``y = x / sqrt(mean(x^2)+eps) * weight`` (Qwen/Llama style).

    Used only when ``norm="rmsnorm"``. With the compensator-lane trick (one big
    constant lane ``K`` dominating ``mean(x^2)`` and ``weight = K/√dim``) it is
    an IDENTITY on the real dims (``qwen_embed.rmsnorm_identity_gamma``), so the
    hand-baked norm-free gadgets pass through unchanged.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class Attn(nn.Module):
    """Multi-head attention with softmax1 + ALiBi (BLOG_SPEC §284-350).

    Three architectural toggles (default = the historical byte-identical path):

      * ``positional`` ∈ {"alibi", "rope"} — additive recency vs. RoPE
        binary-distance recency (content match untouched).
      * ``sink`` ∈ {"softmax1", "bos_sink"} — softmax1 ``+1`` vs. a plain-softmax
        BOS sink column (identical weights).
    """

    def __init__(self, dim: int, n_heads: int, max_seq_len: int = 8192,
                 positional: str = "alibi", sink: str = "softmax1"):
        super().__init__()
        assert positional in ("alibi", "rope"), positional
        assert sink in ("softmax1", "bos_sink"), sink
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.positional = positional
        self.sink = sink
        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))
        # ALiBi slopes: geometric sequence 2^(-8/n*(i+1)) per head (§307-311).
        slopes = torch.tensor(
            [2.0 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)]
        )
        self.register_buffer("alibi_slopes", slopes)

    # ------------------------------------------------------------------
    # Positional bias: ALiBi (additive recency) OR RoPE binary-distance recency.
    #
    # In this VM the "position" job of the attention is RECENCY only: the address
    # match lives in the baked ±smag content CAM (§410), and the positional bias
    # exists solely to break ties among equal-address stores toward the MOST
    # RECENT one (latest-write-wins). ALiBi does this with ``-slope·|q-k|`` (a
    # score strictly decreasing in distance). RoPE's binary-distance self-score
    # ``alpha^2·sum_k cos(theta_k·(q-k))`` (``nibble_rope``, §743) is ALSO
    # maximal at distance 0 and decays with distance — the SAME "nearer wins"
    # ordering. We add it as a positional score term (content Q/K unchanged) so
    # the address argmax is identical and only the recency tie-break is via RoPE.
    #
    # To make "nearer wins" hold monotonically over the whole causal window (not
    # just the first half-period of a raw cos), we use the RoPE relative-distance
    # rotation of a per-head unit position vector on a SINGLE binary frequency
    # chosen so the head's operational window sits inside cos's decreasing
    # branch: theta_h·(max_seq_len) <= pi. Then ``R·cos(theta_h·dist)`` is
    # strictly decreasing in ``dist`` over ``[0, max_seq_len]`` — exactly the
    # ALiBi ordering — and is the genuine RoPE rotary term (a rotation of a unit
    # query/key pair by the position angle), the spec's "relative addressing with
    # one head" (§746). ``R`` is scaled to each head's ALiBi slope so the tie-
    # break STRENGTH matches, while never overturning a content match (the CAM's
    # ``EFF`` dominates any positional term, §410).
    # ------------------------------------------------------------------
    def _rope_theta(self) -> torch.Tensor:
        # one binary frequency per head, small enough that the whole causal
        # window is on cos's monotone-decreasing branch (theta·max_seq_len <= pi).
        return (math.pi / float(self.max_seq_len)) * torch.ones_like(
            self.alibi_slopes)

    def _positional_bias(self, dist: torch.Tensor, H: int,
                         device) -> torch.Tensor:
        """Return the per-head additive positional score term, shaped
        ``[H, Sq, Sk]`` to broadcast over ``scores``.

        ``dist`` is the (non-negative) absolute position distance ``|q-k|``.
        """
        slopes = self.alibi_slopes.to(device)
        if self.positional == "alibi":
            # -slope·dist  (the historical additive recency bias).
            return -slopes.view(H, 1, 1) * dist
        # RoPE binary-distance recency: R·(cos(theta·dist) - 1). This is the
        # rotary relative-distance self-score of a unit position pair minus its
        # dist=0 value (so dist=0 contributes 0, matching ALiBi's baseline), is
        # strictly decreasing over [0, max_seq_len] (theta·max_seq_len <= pi),
        # and preserves the ALiBi "nearer wins" ordering the CAM tie-break needs.
        # Content Q/K are untouched, so the address argmax is unchanged.
        #
        # STRENGTH match: near dist=0, R·(cos(theta·dist)-1) ≈ -(R·theta^2/2)·dist^2.
        # Choosing R = slope / theta^2 makes the small-distance recency coefficient
        # ``R·theta^2 = slope`` — i.e. the RoPE tie-break has the SAME per-head
        # strength as ALiBi's ``-slope·dist`` (quadratic vs linear, but identical
        # "nearer wins" ordering and comparable magnitude), so a two-store tie
        # breaks to the newest store exactly as ALiBi does. R is large but the
        # term stays bounded (|·| <= 2R) and monotone over [0, max_seq_len].
        theta = self._rope_theta().to(device)
        R = slopes / (theta ** 2)                      # recency strength per head
        return (R.view(H, 1, 1)
                * (torch.cos(theta.view(H, 1, 1) * dist) - 1.0))

    def _attend(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply the configured sink (softmax1 or plain-softmax BOS sink)."""
        if self.sink == "softmax1":
            return softmax1(scores, dim=-1)
        return softmax1_via_bos_sink(scores, dim=-1)

    def forward(self, x: torch.Tensor, past_kv=None, q_positions=None,
                use_cache: bool = False):
        """Multi-head softmax1/BOS-sink + ALiBi/RoPE attention.

        Default path (``past_kv is None``, ``q_positions is None``,
        ``use_cache=False``, default toggles) is **byte-identical** to the
        un-cached spec forward: the new tokens sit at absolute positions
        ``0..S-1`` and attend causally over themselves.

        Incremental / cached path (``past_kv`` given):
          * ``x`` is only the NEW query tokens (``[B, Snew, D]``); ``q_positions``
            gives their ABSOLUTE sequence positions (a ``[Snew]`` long tensor).
          * ``past_kv = (K_cache, V_cache, pos_cache)`` holds the K/V of the
            already-seen tokens (``[B, H, Sc, HD]`` + ``[Sc]`` positions).
          * K/V are computed for the new tokens only, concatenated with the cache,
            and the query attends over the union.  The positional bias uses
            ABSOLUTE positions (``|q_pos - k_pos|``) so it is identical to the
            un-cached forward, and the sink is preserved verbatim.

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

        # BYTE-EXACT FLASH ATTENTION (C4_FLASH_ATTN, default OFF).  The default
        # un-cached forward below materialises the FULL [B,H,S,S] score matrix and
        # softmax1's it in one shot — O(S^2) memory.  For the CFM pure-forward path
        # (run_pure_forward_complete: ONE growing token stream re-run per VM step)
        # that quadratic matrix is the wall on the long control/loop/function cases
        # (S grows to tens of thousands of tokens -> OOM / minutes-per-step, on GPU
        # AND CPU).  softmax1 == plain-softmax over a BOS-sink column (§40-44/§491),
        # so a standard tiled flash kernel (never materialising [Sq,Sk]) + the sink
        # recovers it EXACTLY.  On CUDA: the SDPA mem-efficient + LSE-rescale kernel
        # (un-cached full case) or the Triton online-softmax1 kernel (cached / windowed
        # case).  On CPU: the chunked online-softmax1 reference (processes K in 1024-key
        # chunks -> O(S) memory, never the [Sq,Sk] matrix), which is what the #839 CFM
        # battery runs on.  Both fp32, ~1e-6 (below the nibble decode margin).  Gated on
        # the baked VM config (ALiBi + softmax1); no stored weight is touched -> golden
        # 069cc32f byte-identical (flag OFF).
        if (self.positional == "alibi" and self.sink == "softmax1" and B == 1
                and _os_env.environ.get("C4_FLASH_ATTN", "0") == "1"):
            from .flash_softmax1 import flash_softmax1_context, _chunked_reference
            uncached_full = (past_kv is None and q_positions is None)
            slopes = self.alibi_slopes.to(x.device)
            if x.is_cuda:
                ctx = flash_softmax1_context(
                    Q, K, V, q_pos, k_pos, slopes, self.scale,
                    window=None, uncached_full=uncached_full)
            else:
                # CPU: the SDPA/Triton kernels are CUDA-only; the chunked online-softmax1
                # reference is the O(S)-memory byte-exact path for both the un-cached full
                # and the cached/windowed cases (positions handled via q_pos/k_pos).
                ctx = _chunked_reference(Q, K, V, q_pos, k_pos, slopes, self.scale, None)
            out = ctx.transpose(1, 2).contiguous().view(B, S, D)
            out = x + F.linear(out, self.W_o)
            if use_cache:
                return out, (K, V, k_pos)
            return out

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if past_kv is None and q_positions is None:
            # -- default (un-cached) path --
            pos = torch.arange(S, device=x.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()  # [S, S]
            if self.positional == "alibi":
                # keep the EXACT historical expression for byte-identity.
                scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist
            else:
                scores = scores + self._positional_bias(
                    dist, H, x.device).unsqueeze(0)
            causal = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device), diagonal=1
            )
            scores = scores + causal
        else:
            # -- cached / windowed path: positional bias over ABSOLUTE positions --
            dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs().float()  # [Sq, Sk]
            if self.positional == "alibi":
                scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0)
            else:
                scores = scores + self._positional_bias(
                    dist, H, x.device).unsqueeze(0)
            mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))     # [Sq, Sk]
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = self._attend(scores)              # softmax1 (§491) or BOS sink
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
    """One transformer block: attn + FFN, optionally wrapped in RMSNorm.

    ``norm="none"`` (default) is the historical residual: ``x = attn(x); x =
    ffn(x)`` with the additive residual INSIDE ``attn``/``ffn``. ``norm=
    "rmsnorm"`` inserts a pre-norm RMSNorm on each sub-block's input (Qwen/Llama
    pre-norm) — with the compensator-lane identity gamma this is byte-neutral on
    the real dims, so the same baked gadgets fire (``qwen_embed`` §3).
    """

    def __init__(self, dim: int, n_heads: int, hidden: int, max_seq_len: int,
                 positional: str = "alibi", sink: str = "softmax1",
                 norm: str = "none"):
        super().__init__()
        assert norm in ("none", "rmsnorm"), norm
        self.norm = norm
        self.attn = Attn(dim, n_heads, max_seq_len, positional=positional,
                         sink=sink)
        self.ffn = FFN(dim, hidden)
        if norm == "rmsnorm":
            self.attn_norm = RMSNorm(dim)
            self.ffn_norm = RMSNorm(dim)

    # -- attn/ffn sub-steps that honour the norm toggle -------------------
    # ``attn``/``ffn`` already carry the additive residual (``x + ...``). With
    # RMSNorm pre-norm we NORM the input, run the sub-block on the normed input,
    # and add the ORIGINAL (un-normed) residual — the standard pre-norm form
    # ``x = x + sublayer(norm(x))``. ``attn``/``ffn`` return ``norm(x) + delta``
    # (their own residual is on the normed input), so ``x + (out - norm(x))``
    # recovers the pre-norm residual on the ORIGINAL x. Because the compensator
    # makes norm ≈ id on the real dims, ``delta ≈ sublayer(x)`` there, so this
    # reproduces the norm-free result.
    def _run_attn(self, x, **kw):
        if self.norm == "none":
            return self.attn(x, **kw)
        xn = self.attn_norm(x)
        if kw.get("use_cache"):
            a, kv = self.attn(xn, **kw)
            return x + (a - xn), kv
        a = self.attn(xn, **kw)
        return x + (a - xn)

    def _run_ffn(self, x):
        if self.norm == "none":
            return self.ffn(x)
        xn = self.ffn_norm(x)
        return x + (self.ffn(xn) - xn)

    def forward(self, x, past_kv=None, q_positions=None, use_cache: bool = False):
        if use_cache or past_kv is not None or q_positions is not None:
            a, new_kv = self._run_attn(x, past_kv=past_kv,
                                       q_positions=q_positions, use_cache=True)
            out = self._run_ffn(a)
            return (out, new_kv) if use_cache else out
        return self._run_ffn(self._run_attn(x))


class Transformer(nn.Module):
    """Standard decode-only autoregressive transformer.

    ``embed`` is a TOKEN embedding (``vocab x dim``): the input is a stream of
    token ids and the model is stepped by the ordinary autoregressive loop
    (``blogspec_run.generate``). The register/nibble semantics live entirely in
    the baked embedding rows + FFN/attention weights — the architecture itself
    is vanilla.

    Three architectural toggles (default = current byte-identical behaviour):
      * ``positional`` ∈ {"alibi", "rope"}
      * ``norm``       ∈ {"none", "rmsnorm"}
      * ``sink``       ∈ {"softmax1", "bos_sink"}

    For ``norm="rmsnorm"`` the residual must carry a large constant ``K`` on a
    NORM_COMPENSATOR dim of every live embedding row so RMSNorm is an identity on
    the real dims; ``set_norm_compensator`` bakes that (gamma + embed lane).
    """

    def __init__(self, dim: int, n_heads: int, hidden: int, n_blocks: int,
                 vocab: int, max_seq_len: int = 8192,
                 positional: str = "alibi", norm: str = "none",
                 sink: str = "softmax1"):
        super().__init__()
        assert positional in ("alibi", "rope"), positional
        assert norm in ("none", "rmsnorm"), norm
        assert sink in ("softmax1", "bos_sink"), sink
        self.dim = dim
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.positional = positional
        self.norm = norm
        self.sink = sink
        self.embed = nn.Parameter(torch.zeros(vocab, dim))     # token -> residual
        self.blocks = nn.ModuleList(
            Block(dim, n_heads, hidden, max_seq_len, positional=positional,
                  sink=sink, norm=norm) for _ in range(n_blocks)
        )
        self.lm_head = nn.Parameter(torch.zeros(vocab, dim))
        self.lm_bias = nn.Parameter(torch.zeros(vocab))
        if norm == "rmsnorm":
            self.final_norm = RMSNorm(dim)

    # ------------------------------------------------------------------
    # RMSNorm compensator bake (only used when norm="rmsnorm").
    # ------------------------------------------------------------------
    def set_norm_compensator(self, comp_dim: int, K: float = 4000.0,
                             live_rows=None) -> None:
        """Bake the RMSNorm compensator so RMSNorm is an identity on real dims.

        Sets every RMSNorm ``weight`` to ``K/√dim`` (``qwen_embed
        .rmsnorm_identity_gamma``) and writes ``K`` onto dim ``comp_dim`` of
        every ``live_rows`` embedding row (default: all rows). A residual whose
        energy is dominated by that one ``K`` lane has ``rms ≈ K/√dim`` so
        ``x / rms · weight ≈ x`` on every real dim (identity) while the
        compensator lane is preserved (and passed through the norm-free FFN's
        additive residual). NO-OP if ``norm != "rmsnorm"``.
        """
        if self.norm != "rmsnorm":
            return
        gamma = torch.full((self.dim,), K / math.sqrt(self.dim),
                           dtype=self.embed.dtype)
        with torch.no_grad():
            for blk in self.blocks:
                blk.attn_norm.weight.copy_(gamma)
                blk.ffn_norm.weight.copy_(gamma)
            self.final_norm.weight.copy_(gamma)
            rows = range(self.vocab) if live_rows is None else live_rows
            for r in rows:
                self.embed[r, comp_dim] = K

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, S] token ids. Returns logits [B, S, vocab]."""
        x = self.embed[tokens]
        for blk in self.blocks:
            x = blk(x)
        if self.norm == "rmsnorm":
            x = self.final_norm(x)
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

        NOTE: the un-cached ``forward`` applies ``final_norm`` before the LM head
        under ``norm="rmsnorm"``; ``final_norm`` is an identity on the real dims
        (the compensator absorbs the norm), so a nibble decode from ``hidden`` is
        unaffected whether or not the caller mirrors it.
        """
        n = len(self.blocks)
        if past_key_values is None:
            past_key_values = [None] * n
        new_caches = []
        for blk, pkv in zip(self.blocks, past_key_values):
            x, kv = blk(x, past_kv=pkv, q_positions=q_positions, use_cache=True)
            new_caches.append(kv)
        return x, new_caches
