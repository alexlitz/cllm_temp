"""
Autoregressive Neural VM - Decoder-only transformer for VM execution.

All VM computation (instruction fetch, register read/write, ALU, memory,
PC update) flows through standard transformer weights. NO Python arithmetic
in forward passes.

Token format per VM step (35 tokens):
    REG_PC  + 4 value bytes     (5 tokens)
    REG_AX  + 4 value bytes     (5 tokens)
    REG_SP  + 4 value bytes     (5 tokens)
    REG_BP  + 4 value bytes     (5 tokens)
    STACK0  + 4 value bytes     (5 tokens)  — value at *SP (stack top)
    MEM     + 4 addr + 4 value  (9 tokens)
    STEP_END                    (1 token)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention, sparse_linear
from .kv_cache_eviction import softmax1
from .neural_embedding import NeuralVMEmbedding
from .constants import INSTR_WIDTH, PC_OFFSET
from .dim_registry import (
    build_default_registry,
    build_default_contracts,
    ContractValidator,
)
# V8 audit (2026-06-04): the `efficient_alu_*` imports here were unused
# at code level — only mentioned in `_expand_wrapper_blocks` docstrings
# and `_right_size_ffns` recursion comments. The composites
# (ALUAndOrXor/ALUMul/ALUDivMod/AddSub5StageBlock) still exist and are
# instantiated lazily inside `unified_compiler/ops/alu_ops.py` and
# `unified_compiler/ops/shared.py` for the production lookup-mode bake.
# See `docs/V8_DELETE_AUDIT_2026_06_04.md`.


# =============================================================================
# Autoregressive VM Architecture
# =============================================================================


# Step 3 (literal-fallback lint, audit 2026-06-03): single source of truth
# for the d_model / n_heads / ffn_hidden triplet historically replicated
# across ``AutoregressiveVM.__init__`` and ~8 runner files. Runners must
# import these constants instead of re-stating the literals so that any
# future arch migration that updates one source updates them all. The
# lint at ``c4_release/tools/lint_bare_literals.py`` enforces this.
DEFAULT_D_MODEL = 512
DEFAULT_N_HEADS = 8  # HD=64; HD=32 broke attention score budgets (LEV).
DEFAULT_FFN_HIDDEN = 4096


def rotate_half(x):
    """Rotate adjacent feature pairs for RoPE."""
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE rotation requires an even feature dimension")
    x_pair = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair.unbind(dim=-1)
    return torch.stack((-x1, x0), dim=-1).reshape_as(x)


def precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device=None):
    """Precompute RoPE cosine/sine tables with shape ``[max_seq_len, head_dim]``."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    half_idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (half_idx / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    angles = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return angles.cos(), angles.sin()


def apply_rotary_emb(q, k, cos, sin):
    """Apply precomputed RoPE tables to query and key tensors."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Token:
    """Token vocabulary for the autoregressive VM.

    Byte values (0-255) are used directly as token IDs.
    Special tokens start at 256.
    """

    SEP = 256  # Section separator
    REG_PC = 257  # PC register marker
    REG_AX = 258  # AX register marker
    REG_SP = 259  # SP register marker
    REG_BP = 260  # BP register marker
    MEM = 261  # Memory marker
    STEP_END = 262  # End of VM step
    HALT = 263  # Halt / EOS
    CODE_START = 264  # Bytecode section start
    CODE_END = 265  # Bytecode section end
    DATA_START = 266  # Data section start
    DATA_END = 267  # Data section end
    STACK0 = 268  # Stack top value marker (*SP)
    USER_INPUT_START = 269  # Start of user input block (runner-side IO)
    USER_INPUT_END = 270  # End of user input block (runner-side IO)
    TOOL_CALL = 271  # Step-end variant: signals tool call to runner
    THINKING_START = 272  # <thinking> tag for conversational I/O mode
    THINKING_END = 273  # </thinking> tag for conversational I/O mode
    IO_STATE_EMIT_BYTE = 274  # Internal state: emit output byte next
    IO_STATE_EMIT_THINKING = 275  # Internal state: emit THINKING_START next
    VOCAB_SIZE = 276

    # Tokens per VM step. Default 35: PC(5)+AX(5)+SP(5)+BP(5)+STACK0(5)+MEM(9)+SE(1).
    #
    # PROTOTYPE (C4_NO_STACK0_EMIT=1): drop the redundant STACK0 register block
    # (marker + 4 value bytes) from the EMITTED step -> 30 tokens:
    # PC(5)+AX(5)+SP(5)+BP(5)+MEM(9)+SE(1). STACK0 == mem[SP] is fully recoverable
    # from memory state (see DraftVM.draft_tokens), so its emission carries no
    # information the decoder reads (full_trace only checks PC + AX). Removing it
    # eliminates the Root #2 framing-drift class (a failed STACK0 byte breaks the
    # fixed-stride framing). Resolved once at import; the runner sets the env
    # before importing this module and bakes a fresh model per process.
    import os as _os_step
    if _os_step.environ.get("C4_NO_STACK0_EMIT", "1") != "0":
        STEP_TOKENS = 30  # PC(5)+AX(5)+SP(5)+BP(5)+MEM(9)+SE(1) -- STACK0 dropped
    else:
        STEP_TOKENS = 35
    del _os_step


class AutoregressiveAttention(nn.Module):
    """Multi-head attention with configurable softmax and ALiBi/RoPE positions.

    NOT a PureAttention subclass — PureAttention.forward() is FINAL and uses
    F.softmax. This class defaults to softmax1 for zero-fill-on-demand
    semantics and supports standard softmax via config or constructor override.
    """

    def __init__(self, dim, num_heads=4, max_seq_len=4096, layer_idx=None,
                 use_flash_attention=True, positional_encoding=None,
                 attention_normalization=None, rope_base=None,
                 alibi_base_heads=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # ALiBi-slope base head count. The default ALiBi slope of head ``i`` is
        # ``2**(-8/N * (i+1))`` where ``N`` is the slope BASE. When the
        # head-dim-preserving auto-widen
        # (``full_vm_compiler_dynamic._bake_from_scheduled_ops``) appends an
        # over-width residual band, it rounds d_model up to a multiple of the
        # BASE head_dim and ADDS trailing (inert) heads -- e.g.
        # ``C4_AX_BYTE1_FULL_WIDTH`` grows n_heads 10 -> 13. If the slope base
        # were the WIDENED count, every EXISTING head's slope would shift (10
        # -> 13 moves head 0 from 0.574 to 0.653), silently perturbing every
        # globally-sized attention block (mul/shl tail regressions). Pinning
        # the base to the pre-widen count keeps existing heads' slopes
        # byte-identical and gives the trailing padding heads harmless
        # extrapolated slopes. ``None`` => use ``num_heads`` (backward-compat:
        # un-widened builds are unchanged because base == num_heads).
        self.alibi_base_heads = (
            int(alibi_base_heads) if alibi_base_heads is not None
            else num_heads
        )
        # When True, ``forward`` routes through PyTorch 2.x's
        # ``F.scaled_dot_product_attention`` (SDPA), which auto-selects
        # Flash Attention 2 / mem-efficient / math backends. Default ON for
        # speed; can be flipped to False for byte-identity / numeric debug.
        # ONNX export always falls back to the manual path (Flash kernels are
        # not traceable). See c4_release/docs/FLASH_ATTENTION_INTEGRATION.md.
        self.use_flash_attention = use_flash_attention
        # head_dim must tile dim exactly so initial Q/K/V views are well-formed.
        # (L15 may later bake a wider H*HD that exceeds dim; that path is
        # handled by W_o projecting back to dim — see forward().)
        assert num_heads * self.head_dim == dim, (
            f"num_heads ({num_heads}) * head_dim ({self.head_dim}) must equal "
            f"dim ({dim})"
        )
        self.scale = self.head_dim**-0.5
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx

        try:
            from .config import get_config
            config = get_config()
        except ImportError:
            config = None

        if positional_encoding is None:
            positional_encoding = (
                config.positional_encoding if config is not None else "alibi"
            )
        if attention_normalization is None:
            attention_normalization = (
                config.attention_normalization
                if config is not None
                else "softmax1"
            )
        if attention_normalization not in {"softmax1", "softmax"}:
            raise ValueError(
                "attention_normalization must be one of {'softmax1', 'softmax'}"
            )
        self.attention_normalization = attention_normalization
        self.use_softmax1 = attention_normalization == "softmax1"

        if rope_base is None:
            rope_base = config.rope_base if config is not None else 10000.0
        self.rope_base = rope_base

        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))

        # Determine positional encoding for this layer
        if positional_encoding not in {"alibi", "rope", "hybrid"}:
            raise ValueError(
                "positional_encoding must be one of {'alibi', 'rope', 'hybrid'}"
            )
        if positional_encoding == "hybrid" and layer_idx is not None and layer_idx < 3:
            self._positional_encoding = "alibi"
        else:
            self._positional_encoding = positional_encoding

        # Initialize ALiBi slopes if using ALiBi (or hybrid mode with layer < 3)
        use_alibi = (self._positional_encoding == "alibi" or
                     (self._positional_encoding == "hybrid" and layer_idx is not None and layer_idx < 3))
        if use_alibi:
            # Slope base is ``alibi_base_heads`` (== num_heads for un-widened
            # builds), so the head-dim-preserving auto-widen's trailing padding
            # heads do not shift existing heads' slopes (see __init__ note).
            _alibi_n = self.alibi_base_heads
            slopes = torch.tensor(
                [2.0 ** (-8.0 / _alibi_n * (i + 1)) for i in range(num_heads)]
            )
            self.register_buffer("alibi_slopes", slopes)  # [H]
        else:
            self.alibi_slopes = None

        # Initialize RoPE cache if using RoPE (or hybrid mode with layer >= 3)
        use_rope = (self._positional_encoding == "rope" or
                    (self._positional_encoding == "hybrid" and layer_idx is not None and layer_idx >= 3))
        if use_rope:
            cos, sin = precompute_rope_cache(self.head_dim, max_seq_len, base=rope_base)
            self.register_buffer("_rope_cos", cos)
            self.register_buffer("_rope_sin", sin)
        else:
            self._rope_cos = None
            self._rope_sin = None

        # PERF: pre-allocated softmax1 anchor (avoids per-call torch.tensor(0.0) allocation
        # which cProfile showed as ~15% of attention forward time).
        self.register_buffer("_softmax1_anchor", torch.zeros(()), persistent=False)

    def _extend_rope_cache(self, new_max_seq_len: int):
        """Extend RoPE cache to support longer sequences.

        Dynamically extends the cos/sin cache when sequences exceed current max_seq_len.
        This allows supporting arbitrarily long sequences without pre-allocating huge caches.

        Args:
            new_max_seq_len: New maximum sequence length to support
        """
        if self._rope_cos is None:
            return  # Not using RoPE

        current_max_len = self._rope_cos.shape[0]
        if new_max_seq_len <= current_max_len:
            return  # Already large enough

        # Compute extended cache
        cos_new, sin_new = precompute_rope_cache(
            self.head_dim, new_max_seq_len, base=self.rope_base, device=self._rope_cos.device
        )

        # Replace buffers with extended versions
        self.register_buffer("_rope_cos", cos_new)
        self.register_buffer("_rope_sin", sin_new)

    def sparsify(self):
        """Convert weight matrices to COO sparse format."""
        self.W_q = nn.Parameter(self.W_q.data.to_sparse_coo().coalesce())
        self.W_k = nn.Parameter(self.W_k.data.to_sparse_coo().coalesce())
        self.W_v = nn.Parameter(self.W_v.data.to_sparse_coo().coalesce())
        self.W_o = nn.Parameter(self.W_o.data.to_sparse_coo().coalesce())

    def compact(self, block_size=1):
        """Compact attention weights: gather active input dims, prune inactive heads.

        Input dims: identifies non-zero columns in Q/K/V, gathers only those.
        Heads: prunes entirely inactive heads (all-zero Q/K/V/O rows).
        Output dims must stay head-aligned (multiple of head_dim).
        """
        D = self.dim
        H = self.num_heads
        HD = self.head_dim
        W_q = self.W_q.data.to_dense() if self.W_q.is_sparse else self.W_q.data
        W_k = self.W_k.data.to_dense() if self.W_k.is_sparse else self.W_k.data
        W_v = self.W_v.data.to_dense() if self.W_v.is_sparse else self.W_v.data
        W_o = self.W_o.data.to_dense() if self.W_o.is_sparse else self.W_o.data

        # Active input dims: any column with non-zero weight in Q, K, or V
        active_in = (
            (W_q.abs().sum(dim=0) > 0)
            | (W_k.abs().sum(dim=0) > 0)
            | (W_v.abs().sum(dim=0) > 0)
        )
        in_idx = active_in.nonzero(as_tuple=True)[0]
        if len(in_idx) == 0:
            return  # nothing to compact

        # Active heads: head h is active if any row in [h*HD:(h+1)*HD] is non-zero
        active_heads = []
        for h in range(H):
            s, e = h * HD, (h + 1) * HD
            if (
                W_q[s:e].abs().sum() > 0
                or W_k[s:e].abs().sum() > 0
                or W_v[s:e].abs().sum() > 0
                or W_o[:, s:e].abs().sum() > 0
            ):
                active_heads.append(h)

        if len(active_heads) == H and len(in_idx) == D:
            return  # nothing to compact

        # Build output index (head-aligned)
        out_idx = []
        for h in active_heads:
            out_idx.extend(range(h * HD, (h + 1) * HD))
        out_idx = torch.tensor(out_idx, dtype=torch.long)

        # Store index maps for gather/scatter in forward
        self.register_buffer("_compact_in_idx", in_idx)
        self.register_buffer("_compact_out_idx", out_idx)
        self._is_compact = True
        self.num_heads = len(active_heads)
        # head_dim stays the same; alibi_slopes shrinks to active heads (if present)
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes[active_heads]

        # Compact: W_q/K/V[n_out, n_in], W_o[D, n_out]
        self.W_q = nn.Parameter(W_q[out_idx][:, in_idx].contiguous())
        self.W_k = nn.Parameter(W_k[out_idx][:, in_idx].contiguous())
        self.W_v = nn.Parameter(W_v[out_idx][:, in_idx].contiguous())
        self.W_o = nn.Parameter(W_o[:, out_idx].contiguous())

    def forward(self, x, kv_cache=None, x_is_new_only=False):
        """
        Forward pass with optional KV caching.

        Args:
            x: Input tensor [B, S, D].
                - Default mode (``x_is_new_only=False``): ``x`` is the full
                  sequence including positions already in the cache. Q/K/V are
                  computed for all positions, and K/V for the trailing new
                  positions are appended to ``kv_cache``. Output shape: [B, S, D].
                - Incremental mode (``x_is_new_only=True``): ``x`` contains ONLY
                  the new tokens. Q/K/V are computed only for these new tokens.
                  K/V are appended to ``kv_cache``; attention uses the full
                  (cached + new) K/V. Output shape: [B, new_tokens, D].
            kv_cache: Optional TransformerKVCache for incremental generation.
            x_is_new_only: When True, ``x`` contains only new tokens beyond the
                cached prefix (incremental generation path).

        Returns:
            Output tensor; see notes above for shape semantics.
        """
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        # Number of positions already cached (0 when no/empty cache).
        cached_len = kv_cache.cache_size if (kv_cache is not None) else 0

        if getattr(self, "_is_compact", False):
            # Compact path: gather active dims → dense matmul → scatter
            x_in = x[:, :, self._compact_in_idx]  # [B, S, n_in]
            n_out = len(self._compact_out_idx)
            if x_is_new_only:
                # Compute Q/K/V only for new tokens (all of x).
                Q = F.linear(x_in, self.W_q).view(B, S, H, n_out // H).transpose(1, 2)
                K_new = F.linear(x_in, self.W_k).view(B, S, H, n_out // H).transpose(1, 2)
                V_new = F.linear(x_in, self.W_v).view(B, S, H, n_out // H).transpose(1, 2)
            else:
                # Legacy mode: x is full sequence. Compute Q for all; K/V only
                # for trailing new tokens (S - cached_len).
                Q = F.linear(x_in, self.W_q).view(B, S, H, n_out // H).transpose(1, 2)
                new_tokens = S - cached_len if kv_cache is not None else S
                if kv_cache is not None and cached_len > 0 and new_tokens > 0:
                    x_new_in = x_in[:, -new_tokens:, :]
                    K_new = F.linear(x_new_in, self.W_k).view(B, new_tokens, H, n_out // H).transpose(1, 2)
                    V_new = F.linear(x_new_in, self.W_v).view(B, new_tokens, H, n_out // H).transpose(1, 2)
                elif kv_cache is not None and cached_len > 0 and new_tokens == 0:
                    # All tokens are cached, K_new/V_new are empty; we'll just
                    # reuse cached K/V directly without updating.
                    K_new = None
                    V_new = None
                else:
                    K_new = F.linear(x_in, self.W_k).view(B, S, H, n_out // H).transpose(1, 2)
                    V_new = F.linear(x_in, self.W_v).view(B, S, H, n_out // H).transpose(1, 2)
        else:
            linear = sparse_linear if self.W_q.is_sparse else F.linear
            if x_is_new_only:
                Q = linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)
                K_new = linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
                V_new = linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)
            else:
                Q = linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)
                new_tokens = S - cached_len if kv_cache is not None else S
                if kv_cache is not None and cached_len > 0 and new_tokens > 0:
                    x_new = x[:, -new_tokens:, :]
                    K_new = linear(x_new, self.W_k).view(B, new_tokens, H, HD).transpose(1, 2)
                    V_new = linear(x_new, self.W_v).view(B, new_tokens, H, HD).transpose(1, 2)
                elif kv_cache is not None and cached_len > 0 and new_tokens == 0:
                    K_new = None
                    V_new = None
                else:
                    K_new = linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
                    V_new = linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)

        # If we have a KV cache, append new K/V (if any) and use the full
        # (cached + new) K/V tensors for attention.
        if kv_cache is not None:
            if K_new is None:
                # All cached, no new tokens.
                K, V = kv_cache.cached_k, kv_cache.cached_v
            else:
                K, V = kv_cache.update(K_new, V_new)
        else:
            K, V = K_new, V_new

        # Resolve absolute position ids for Q and K.
        #
        # When the cache has ``cached_pos_ids`` populated (post pos_ids
        # tracking — see ``LayerKVCache``), pull positions directly so that
        # ALiBi/RoPE distances reflect the *original* token positions even
        # after middle-position eviction by the pruner (KV_CACHE_PRUNING_SPEC
        # §9). When no cache is present (or it predates pos_ids), fall back
        # to the legacy ``arange`` numbering (which equals
        # ``cached_pos_ids`` whenever no pruning has occurred).
        S_q = Q.shape[2]
        S_kv = K.shape[2]
        if (kv_cache is not None
                and getattr(kv_cache, "cached_pos_ids", None) is not None
                and kv_cache.cached_pos_ids.shape[-1] == S_kv):
            # cached_pos_ids: [B, S_kv]. Use batch 0 (positions are duplicated
            # across batch since the time axis is shared).
            k_pos_1d = kv_cache.cached_pos_ids[0].to(x.device)  # [S_kv]
            q_pos_1d = k_pos_1d[S_kv - S_q:]  # [S_q]  (newly appended slice)
        else:
            q_pos_1d = torch.arange(S_kv - S_q, S_kv, device=x.device)
            k_pos_1d = torch.arange(S_kv, device=x.device)

        # Apply RoPE if enabled (check for RoPE cache presence)
        if self._rope_cos is not None:
            # Q is [B, H, S_q, HD] where S_q = new_tokens.
            # K is [B, H, S_kv, HD] where S_kv = cached_len + new_tokens.
            # Dynamically extend RoPE cache if max position id exceeds current cache size.
            # When pos_ids come from the legacy arange path the slice path
            # below works without the gather; only the cache-aware branch
            # needs index_select since pos_ids may be non-contiguous after
            # pruning.
            use_pos_gather = (
                kv_cache is not None
                and getattr(kv_cache, "cached_pos_ids", None) is not None
                and kv_cache.cached_pos_ids.shape[-1] == S_kv
            )
            if use_pos_gather:
                max_needed = int(max(k_pos_1d.max().item() + 1, q_pos_1d.max().item() + 1))
            else:
                max_needed = S_kv
            if max_needed > self._rope_cos.shape[0]:
                new_max_len = int(max_needed * 1.5)
                self._extend_rope_cache(new_max_len)

            # Apply RoPE to Q and K. Q gets cos/sin indexed at the new tokens'
            # absolute positions; K is rotated in full because the cache stores
            # the raw (un-rotated) K_new and we rotate per-call (matches the
            # legacy pre-cache code's behavior).
            if use_pos_gather:
                cos_q = self._rope_cos.index_select(0, q_pos_1d).unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
                sin_q = self._rope_sin.index_select(0, q_pos_1d).unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
                cos_k = self._rope_cos.index_select(0, k_pos_1d).unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]
                sin_k = self._rope_sin.index_select(0, k_pos_1d).unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]
            else:
                q_offset = S_kv - S_q
                cos_q = self._rope_cos[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
                sin_q = self._rope_sin[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
                cos_k = self._rope_cos[0:S_kv].unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]
                sin_k = self._rope_sin[0:S_kv].unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]

            Q = (Q * cos_q) + (rotate_half(Q) * sin_q)
            K = (K * cos_k) + (rotate_half(K) * sin_k)

        # ----------------------------------------------------------------
        # Build the additive bias matrix (ALiBi + causal + per-head mask).
        # SDPA accepts this as ``attn_mask``; the manual path adds it to
        # ``scores`` before softmax1. Bias shape broadcasts from ``[1, H, S_q,
        # S_kv]`` (or ``[B, H, S_q, S_kv]`` when per-head mask is active).
        # ----------------------------------------------------------------
        # ALiBi bias: -slope * |q_pos - k_pos| using *absolute* positions
        # (per KV_CACHE_PRUNING_SPEC §9: surviving positions retain their
        # original indices, so distance computation uses pos_ids — NOT the
        # cache sequence index).
        bias = None
        if self.alibi_slopes is not None:
            # When no pruning has occurred q_pos_1d / k_pos_1d are arange-like
            # (and equal to the slice computation used pre-pos_ids). Keeping
            # the same Python int branch when no cache is present makes the
            # ONNX tracer happy.
            dist = (q_pos_1d.unsqueeze(1) - k_pos_1d.unsqueeze(0)).abs().float()  # [S_q, S_kv]
            bias = -self.alibi_slopes.view(1, H, 1, 1) * dist  # [1, H, S_q, S_kv]

        # Causal mask: position i (absolute = cached_len + i) attends to j only if
        # j <= cached_len + i, i.e. mask out j > cached_len + i.
        # In row-coordinates: row r=0..S_q-1, col c=0..S_kv-1; mask if c > (S_kv - S_q) + r.
        # That is: triu with diagonal = (S_kv - S_q + 1).
        q_offset_rel = S_kv - S_q
        causal_mask = torch.triu(
            torch.full((S_q, S_kv), float("-inf"), device=x.device),
            diagonal=q_offset_rel + 1,
        )
        if bias is None:
            bias = causal_mask.view(1, 1, S_q, S_kv)
        else:
            bias = bias + causal_mask.view(1, 1, S_q, S_kv)

        # Per-head soft-eviction mask (Phase B): when the pruner has set
        # ``per_head_keep_mask`` on this layer's cache, each head's logits
        # for positions where ``mask[b, h, s] == False`` are pushed to
        # ``-inf`` so the head ignores that K/V row. Shape is
        # ``[B, H, S_kv]`` → broadcast against scores' ``[B, H, S_q, S_kv]``.
        # ``None`` is the legacy "no per-head masking" path — bit-identical
        # to pre-Phase-B behaviour.
        if (
            kv_cache is not None
            and getattr(kv_cache, "per_head_keep_mask", None) is not None
            and kv_cache.per_head_keep_mask.shape[-1] == S_kv
        ):
            head_keep = kv_cache.per_head_keep_mask.to(x.device)  # [B, H, S_kv]
            head_mask_neginf = torch.where(
                head_keep,
                torch.zeros((), device=x.device),
                torch.full((), float("-inf"), device=x.device),
            )
            # head_mask broadcasts over S_q: [B, H, 1, S_kv] + [..., S_q, S_kv]
            bias = bias + head_mask_neginf.unsqueeze(2)

        # ----------------------------------------------------------------
        # Select attention backend.
        #
        # SDPA path for softmax1 appends a "sink" K/V column of zeros so that
        #   softmax([scores, 0]) = exp(scores) / (1 + sum(exp(scores)))
        # which exactly matches softmax1 with anchor=0. V=0 at the sink
        # column means that column contributes 0 to the output, so the
        # remaining V rows are weighted exactly like softmax1.
        #
        # Standard softmax uses SDPA without the sink column.
        #
        # Disable SDPA when: (a) the toggle is off, (b) sparse Q weights
        # were detected earlier (sparse_linear path implies CPU-friendly
        # debug runs that already expect manual numerics), or (c) we're
        # tracing for ONNX export — Flash kernels are not export-traceable.
        # ----------------------------------------------------------------
        use_sdpa = (
            self.use_flash_attention
            and not torch.onnx.is_in_onnx_export()
            and not self.W_q.is_sparse
        )

        if use_sdpa:
            if self.use_softmax1:
                # Append softmax1 sink: an extra K/V column with K=0
                # (score=0 vs any Q) and V=0 (no contribution to output).
                # The bias also needs a zero column appended so the sink score
                # stays 0 after the bias add inside SDPA.
                sink_k = torch.zeros(
                    K.shape[0], K.shape[1], 1, K.shape[3],
                    dtype=K.dtype, device=K.device,
                )
                sink_v = torch.zeros(
                    V.shape[0], V.shape[1], 1, V.shape[3],
                    dtype=V.dtype, device=V.device,
                )
                K_attn = torch.cat([K, sink_k], dim=2)
                V_attn = torch.cat([V, sink_v], dim=2)

                # Bias gains a sink column of 0 (so score against sink = 0).
                bias_sink = torch.zeros(
                    bias.shape[0], bias.shape[1], bias.shape[2], 1,
                    dtype=bias.dtype, device=bias.device,
                )
                bias_for_attention = torch.cat([bias, bias_sink], dim=-1)
            else:
                K_attn = K
                V_attn = V
                bias_for_attention = bias

            # SDPA expects attn_mask in Q's dtype to land on the Flash
            # backend (math/mem-efficient also accept it). The bias may be
            # float32 from the ALiBi distance computation; cast to Q dtype.
            attn_mask = bias_for_attention.to(Q.dtype)

            out = F.scaled_dot_product_attention(
                Q, K_attn, V_attn,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,  # causal is baked into ``attn_mask``
                scale=self.scale,
            )
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            scores = scores + bias

            if self.use_softmax1:
                # softmax1 for ZFOD (inlined with cached anchor=0 buffer to
                # avoid per-call torch.tensor() allocations; equivalent to
                # softmax1(scores, dim=-1, anchor=0.0)).
                anchor = self._softmax1_anchor
                max_val = torch.max(scores.amax(dim=-1, keepdim=True), anchor)
                exp_scores = torch.exp(scores - max_val)
                exp_anchor = torch.exp(anchor - max_val)
                attn = exp_scores / (exp_anchor + exp_scores.sum(dim=-1, keepdim=True))
            else:
                attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, V)

        if getattr(self, "_is_compact", False):
            out = out.transpose(1, 2).contiguous().view(B, S, n_out)
            # W_o is [D, n_out] — full output dim, compact internal dim
            return x + F.linear(out, self.W_o)
        else:
            # Always view as [B, S, H*HD]. In the standard case H*HD == D
            # (e.g., 8 heads × 64 dims = 512); in the L15 case H*HD > D
            # (12 heads × 64 dims = 768) and W_o[D, H*HD] projects back to D.
            # Using H*HD unconditionally keeps both paths shape-correct and
            # avoids a Python branch that the ONNX tracer would constant-fold.
            out = out.transpose(1, 2).contiguous().view(B, S, H * HD)
            return x + (sparse_linear if self.W_q.is_sparse else F.linear)(
                out, self.W_o
            )


# Legacy class - Commented out due to incompatibility with PureAttention architecture
# PureAttention.forward() is now final and cannot be overridden
# Use PureAttention directly with causal=True instead
#
# class CausalSelfAttention(PureAttention):
#     """PureAttention with causal masking for variable-length sequences.
#
#     Computes causal mask on-the-fly from sequence length.
#     """
#
#     def __init__(self, dim, num_heads=4, max_seq_len=4096):
#         super().__init__(dim, num_heads=num_heads, causal=True)
#         self.max_seq_len = max_seq_len
#
#     def forward(self, x):
#         B, S, D = x.shape
#         H = self.num_heads
#         HD = self.head_dim
#
#         Q = F.linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)
#         K = F.linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
#         V = F.linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)
#
#         scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
#         causal_mask = torch.triu(
#             torch.full((S, S), float("-inf"), device=x.device), diagonal=1
#         )
#         scores = scores + causal_mask
#
#         attn = F.softmax(scores, dim=-1)
#         out = torch.matmul(attn, V)
#
#         out = out.transpose(1, 2).contiguous().view(B, S, D)
#         return x + F.linear(out, self.W_o)


def _resolve_bd(dim_positions):
    """Return a BD-like object that resolves dim names to compact positions.

    When ``dim_positions`` is a dict of compiler-allocated positions, returns
    a ``_SetDim`` proxy that maps ``BD.<NAME>`` lookups through the dict;
    otherwise returns the legacy ``_SetDim`` class. Used by post-op
    ``_bake_weights`` methods so a single helper replaces the duplicated
    ``if dim_positions is not None: ... else: BD = _SetDim`` branch.
    """
    if dim_positions is None:
        return _SetDim
    from .unified_compiler.ops.shared import _as_setdim_proxy
    return _as_setdim_proxy(dim_positions)


class ComparisonCombine(PureFFN):
    """Comparison combine module for EQ/NE/LT/GT/LE/GE.

    Reads CMP[0..3] flags from L9 FFN at AX marker position:
      CMP[0]=hi_lt, CMP[1]=hi_eq, CMP[2]=lo_eq, CMP[3]=lo_lt.
    Combines them with opcode gating and writes result to OUTPUT dims.

    Uses default + override pattern:
    - Default unit writes initial result (0 or 1)
    - Override units flip result based on CMP flags

    Phase 0 conversion (2026-05-09): now subclasses PureFFN so it has the
    canonical SwiGLU forward (silu(W_up @ x + b_up) * (W_gate @ x + b_gate) → W_down)
    and is treated structurally like every other FFN by tooling.
    """

    def __init__(self, d_model=512, S=100.0, dim_positions=None):
        # PureFFN.__init__ calls _bake_weights(); store config first so _bake_weights
        # can read it. We use object.__setattr__ to bypass nn.Module.__setattr__ since
        # nn.Module isn't initialized yet at this point.
        object.__setattr__(self, '_pending_S', S)
        # Stash dim_positions BEFORE super().__init__ so _bake_weights (called
        # from PureFFN.__init__) sees compact positions if provided. Without
        # this, the post-op writes/reads against legacy `_SetDim` positions
        # which alias unrelated dims under compact (pin_io_only=True) layouts.
        object.__setattr__(self, '_pending_dim_positions', dim_positions)
        super().__init__(dim=d_model, hidden_dim=18)
        self.d_model = d_model
        self.S = S

    def _bake_weights(self):
        S = self._pending_S
        BD = _resolve_bd(getattr(self, '_pending_dim_positions', None))
        unit = 0

        # FIX 2026-05-09: All ComparisonCombine units add a strong MARK_PC blocker.
        # Without it, leaked OP_NE/OP_GT/OP_GE flags or CMP[0..3] (which can reach
        # 14.98 at MARK_PC due to cumulative writes by L6-L8 heads in pure-neural
        # mode) cause units to fire there (writing OUTPUT_LO[1] = ~50), corrupting
        # PC predictions. -50*S is enough to dominate even very leaky CMP=15:
        # up = 100*15(CMP) - 5000*1(MARK_PC) - 150 = -3650 -> silu -> 0.
        MARK_PC_BLOCK = -S * 50

        def _cmp_default(op_dim, default_result):
            nonlocal unit
            self.W_up.data[unit, BD.MARK_AX] = S
            self.W_up.data[unit, op_dim] = S
            self.W_up.data[unit, BD.MARK_PC] = MARK_PC_BLOCK
            self.b_up.data[unit] = -S * 1.5
            self.b_gate.data[unit] = 1.0
            self.W_down.data[BD.OUTPUT_LO + default_result, unit] = 2.0 / S
            self.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        def _cmp_override_2way(op_dim, cmp_dim, to_result, from_result):
            nonlocal unit
            self.W_up.data[unit, BD.MARK_AX] = S
            self.W_up.data[unit, cmp_dim] = S
            self.W_up.data[unit, BD.MARK_PC] = MARK_PC_BLOCK
            self.b_up.data[unit] = -S * 1.5
            self.W_gate.data[unit, op_dim] = 1.0
            self.W_down.data[BD.OUTPUT_LO + to_result, unit] = 4.0 / S
            self.W_down.data[BD.OUTPUT_LO + from_result, unit] = -4.0 / S
            unit += 1

        def _cmp_override_3way(op_dim, cmp_dim1, cmp_dim2, to_result,
                               from_result, *, threshold=2.5):
            nonlocal unit
            self.W_up.data[unit, BD.MARK_AX] = S
            self.W_up.data[unit, cmp_dim1] = S
            self.W_up.data[unit, cmp_dim2] = S
            self.W_up.data[unit, BD.MARK_PC] = MARK_PC_BLOCK
            # Shape B CMP fix (2026-06-07, removal-4): add CMP+0 blocker
            # at weight -0.1 to suppress this override when hi_lt is hot
            # (operands not equal at hi nibble). The Shape B EQ_FALSE /
            # NE_TRUE failures previously masked by the 69f77682
            # _NON_COLLAPSED_RECOVERY_OPS override are caused by the
            # legacy ComparisonCombine post_op firing this rule on
            # spurious CMP+1 amplification (residual ~10 at the
            # AX-marker row in non-collapsed CMP steps). With CMP+0
            # amplified to ~150 in those rows (the true hi_lt
            # indicator), the -0.1 contributes -15 to the symbolic
            # score, dropping it below the +2.5 threshold and silencing
            # the spurious override writes. For true-equality cases
            # (EQ_TRUE, NE_FALSE) CMP+0 is zero (no hi_lt fired), so
            # the blocker contributes zero and the override fires
            # normally. Mirrors the parallel declarative fix in
            # unified_compiler/ops/l10_ops.py::_cmp_override_3way
            # (both _l10_comparison_combine_rules and
            # _layer10_alu_cmp_combine_rules). See
            # docs/REMOVAL_4_DEEP_FIX_2026_06_06.md for the Shape A/B
            # diagnosis that drove this fix.
            self.W_up.data[unit, BD.CMP + 0] = -S * 0.1
            self.b_up.data[unit] = -S * threshold
            self.W_gate.data[unit, op_dim] = 1.0
            self.W_down.data[BD.OUTPUT_LO + to_result, unit] = 4.0 / S
            self.W_down.data[BD.OUTPUT_LO + from_result, unit] = -4.0 / S
            unit += 1

        # if_var GT-TRUE lo_lt-leak guard (campaign-only). RAISE the GT/GE
        # (hi_eq AND lo_lt) -> 0 override threshold 2.5 -> 2.75 so a spurious
        # lo_lt (CMP+3 ~= 1.67) alone (hi_eq absent, A.hi > B.hi) can no longer
        # trip the GT-result flip on the loaded-var GT-TRUE path (if_var 425/
        # 436/440/441/445/448), while the genuine hi_eq+lo_lt GT-FALSE override
        # (CMP+1 ~= 1.24 AND CMP+3 ~= 1.46, sum 3.70) still fires. The live GT
        # decoder is THIS ComparisonCombine (lowered into the L14 post-op
        # expansion, block 28, reading raw CMP at the result row). Symmetric
        # companion to the GT-FALSE cmp_hi_lt_alu15_leak_guard. Flag-OFF /
        # non-campaign -> 2.5 -> golden byte-identical. See
        # ``unified_compiler/ops/shared.cmp_gt_lo_lt_hieq_guard_enabled``.
        from neural_vm.unified_compiler.ops.shared import (
            cmp_gt_lo_lt_hieq_guard_enabled,
        )
        _gt_lo_lt_thresh = (
            2.75 if cmp_gt_lo_lt_hieq_guard_enabled() else 2.5
        )

        _cmp_default(BD.OP_EQ, 0)
        _cmp_override_3way(BD.OP_EQ, BD.CMP + 1, BD.CMP + 2, 1, 0)

        _cmp_default(BD.OP_NE, 1)
        _cmp_override_3way(BD.OP_NE, BD.CMP + 1, BD.CMP + 2, 0, 1)

        _cmp_default(BD.OP_LT, 0)
        _cmp_override_2way(BD.OP_LT, BD.CMP + 0, 1, 0)
        _cmp_override_3way(BD.OP_LT, BD.CMP + 1, BD.CMP + 3, 1, 0)

        _cmp_default(BD.OP_GT, 1)
        _cmp_override_2way(BD.OP_GT, BD.CMP + 0, 0, 1)
        _cmp_override_3way(BD.OP_GT, BD.CMP + 1, BD.CMP + 3, 0, 1,
                           threshold=_gt_lo_lt_thresh)
        _cmp_override_3way(BD.OP_GT, BD.CMP + 1, BD.CMP + 2, 0, 1)

        _cmp_default(BD.OP_LE, 0)
        _cmp_override_2way(BD.OP_LE, BD.CMP + 0, 1, 0)
        _cmp_override_3way(BD.OP_LE, BD.CMP + 1, BD.CMP + 3, 1, 0)
        _cmp_override_3way(BD.OP_LE, BD.CMP + 1, BD.CMP + 2, 1, 0)

        _cmp_default(BD.OP_GE, 1)
        _cmp_override_2way(BD.OP_GE, BD.CMP + 0, 0, 1)
        _cmp_override_3way(BD.OP_GE, BD.CMP + 1, BD.CMP + 3, 0, 1,
                           threshold=_gt_lo_lt_thresh)


class BinaryOpByteZeroingPostOp(PureFFN):
    """Post-op that zeros OUTPUT at AX byte positions for ops with byte-0-only results.

    For ADD/SUB/MUL/DIV/MOD/SHL/SHR/EQ/NE/LT/GT/LE/GE, the byte 0 result comes from
    the EfficientALU. Bytes 1-3 get garbage from L10 head 1 passthrough. This
    post_op clears bytes 1-3 to 0x00 for these opcodes.

    ADD/SUB bytes are handled by the dedicated AddSubBytePropagationPostOp and
    CarryPropagationPostOp chain. Keeping them out of this generic zeroing pass
    avoids making high-byte ADD/SUB rows fight a stale zero baseline.

    Phase 0 conversion (2026-05-09): subclasses PureFFN.
    """

    def __init__(self, d_model=512, S=100.0, dim_positions=None):
        object.__setattr__(self, '_pending_S', S)
        # Stash dim_positions BEFORE super().__init__ so _bake_weights (called
        # from PureFFN.__init__) sees compact positions if provided. Without
        # this, the post-op writes to legacy `_SetDim` positions which collide
        # with unrelated dims in compact (pin_io_only=True) layouts — e.g.
        # `_SetDim.H1 + 1 = 68` aliases compact `EMBED_HI[15]`, triggering this
        # post-op for any AX byte whose hi nibble is 0xF (the IMM 240/255 bug).
        object.__setattr__(self, '_pending_dim_positions', dim_positions)
        # Four units handle byte-0-only non-bitwise ops through opcode gates.
        # Four more handle bitwise rows, with TEMP[3] in the thresholded
        # detector rather than only in the multiplicative gate. Weak TEMP[3]
        # residue is common on unrelated IMM/ADD byte rows and must not zero
        # their high bytes.
        super().__init__(dim=d_model, hidden_dim=8)
        self.d_model = d_model
        self.S = S

    def _bake_weights(self):
        S = self._pending_S
        BD = _resolve_bd(getattr(self, '_pending_dim_positions', None))
        op_dims = [BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT,
                   BD.OP_LE, BD.OP_GE, BD.OP_SHL, BD.OP_SHR,
                   BD.OP_MUL, BD.OP_DIV, BD.OP_MOD]

        def wire_zeroing_writes(unit_offset: int) -> None:
            self.W_down.data[BD.OUTPUT_LO + 0, unit_offset + 2] = 5.0 / S
            self.W_down.data[BD.OUTPUT_HI + 0, unit_offset + 3] = 5.0 / S
            for k in range(16):
                self.W_down.data[BD.OUTPUT_LO + k, unit_offset + 0] = -3.0 / S
                self.W_down.data[BD.OUTPUT_HI + k, unit_offset + 1] = -3.0 / S

        for unit in range(4):
            self.W_up.data[unit, BD.IS_BYTE] = S
            self.W_up.data[unit, BD.H1 + 1] = S
            self.W_up.data[unit, BD.TEMP + 8] = -S * 1000
            self.W_up.data[unit, BD.TEMP + 9] = -S * 1000
            self.b_up.data[unit] = -S * 1.5
            for d in op_dims:
                self.W_gate.data[unit, d] = 1.0

        for unit in range(4, 8):
            # Bitwise relays can appear as weak residue on unrelated byte rows.
            # Keep TEMP[3] in this thresholded detector so only a full relay
            # produces a zeroing write.
            self.W_up.data[unit, BD.IS_BYTE] = S
            self.W_up.data[unit, BD.H1 + 1] = S
            self.W_up.data[unit, BD.TEMP + 3] = S
            self.W_up.data[unit, BD.TEMP + 8] = -S * 1000
            self.W_up.data[unit, BD.TEMP + 9] = -S * 1000
            self.b_up.data[unit] = -S * 2.5
            self.W_gate.data[unit, BD.TEMP + 3] = 1.0

        wire_zeroing_writes(0)
        wire_zeroing_writes(4)


class CarryPropagationPostOp(PureFFN):
    """Post-op for L10 that propagates carry/borrow between adjacent bytes.

    Each instance handles one byte level. Three instances are stacked as
    sequential post_ops for the full byte 0→1→2→3 cascade.

    Phase 0 conversion (2026-05-09): subclasses PureFFN.
    """

    def __init__(self, d_model=512, S=100.0, byte_idx=0, cascade=False, dim_positions=None):
        object.__setattr__(self, '_pending_S', S)
        object.__setattr__(self, '_pending_byte_idx', byte_idx)
        object.__setattr__(self, '_pending_cascade', cascade)
        # Stash dim_positions BEFORE super().__init__ so _bake_weights (called
        # from PureFFN.__init__) sees compact positions if provided.
        object.__setattr__(self, '_pending_dim_positions', dim_positions)
        super().__init__(dim=d_model, hidden_dim=512)
        self.d_model = d_model
        self.S = S

    def _bake_weights(self):
        # Phase 7.C cut (2026-06-12): the inter-byte carry/borrow cascade is
        # now authored declaratively. The 256 ADD + 256 SUB units this method
        # used to write imperatively are expressed as ``FFNRule``s in
        # ``unified_compiler.ops.l10_ops._l10_carry_propagation_rules`` and
        # lowered through ``Primitives.lower_ffn_rules`` -- the same lowering
        # the live model's ``make_l10_post_op_attach_op`` now drives directly.
        # Byte-identity (element-wise tensor diff = 0 vs the legacy bake +
        # lowering-contract OK) is gated by ``tools/verify_carry_migration.py``.
        # ``PureFFN.forward`` is final/shared, so this delegation is a pure
        # weight-authoring cut with no behavioral change.
        S = self._pending_S
        byte_idx = self._pending_byte_idx
        cascade = self._pending_cascade
        dim_positions = getattr(self, '_pending_dim_positions', None)

        # Lazy imports avoid an import cycle (l10_ops imports vm_step).
        from .unified_compiler.ops.l10_ops import _l10_carry_propagation_rules
        from .unified_compiler.primitives import Primitives

        rules = _l10_carry_propagation_rules(
            S, byte_idx=byte_idx, cascade=cascade,
        )

        if dim_positions is None:
            # Legacy ``_SetDim`` layout (unit-test path with no compact
            # dim_positions). Build the name->position map the lowerer needs
            # from the static ``_SetDim`` table. Every name the rules
            # reference resolves against ``_SetDim`` (verified; e.g.
            # OUTPUT_HI_THIS_STEP aliases OUTPUT_HI = 190).
            names = set()
            for rule in rules:
                for term in rule.conditions:
                    names.add(term.dim.name)
                if rule.gate is not None:
                    names.add(rule.gate.name)
                for term in rule.gate_terms:
                    names.add(term.dim.name)
                for write in rule.writes:
                    names.add(write.dim.name)
            dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)

        with torch.no_grad():
            Primitives.lower_ffn_rules(
                self, rules, dim_positions, start_unit=0, S=S,
            )


class AddSubBytePropagationPostOp(PureFFN):
    """Compute ADD/SUB base bytes 1-3 from relayed stack and AX bytes.

    L10's marker ALU computes byte 0. At AX byte positions, L10 head 1 carries
    the previous AX byte in OUTPUT and head 4 carries the stack byte in ALU.
    This post-op materializes the per-byte ADD/SUB base result before the
    carry/borrow post-op adjusts it using the autoregressive carry chain.

    The byte-internal nibble carry is intentionally not modeled here yet; the
    current strict smoke coverage exercises high bytes whose nibbles are 0/1.
    The important part is making stack high bytes participate in the neural
    path instead of treating all high bytes as zero.
    """

    def __init__(self, d_model=512, S=100.0, dim_positions=None):
        object.__setattr__(self, '_pending_S', S)
        object.__setattr__(self, '_pending_dim_positions', dim_positions)
        # 2 ops * 2 nibbles * 16x16 rules plus a few borrow-continuation
        # rules. Keep slack so ADD and SUB can both dispatch from relayed
        # opcode flags instead of overloading carry/borrow as op detectors.
        super().__init__(dim=d_model, hidden_dim=1536)
        self.d_model = d_model
        self.S = S

    def _bake_weights(self):
        S = self._pending_S
        BD = _resolve_bd(getattr(self, '_pending_dim_positions', None))

        unit = 0
        with torch.no_grad():
            for op_dim, suppress_op_dim, op_fn in (
                (BD.TEMP + 8, BD.TEMP + 9, lambda a, b: (a + b) & 0xF),
                (BD.TEMP + 9, BD.TEMP + 8, lambda a, b: (a - b) & 0xF),
            ):
                for out_base, alu_base in (
                    (BD.OUTPUT_LO, BD.ALU_LO),
                    (BD.OUTPUT_HI, BD.ALU_HI),
                ):
                    for a in range(16):
                        for b in range(16):
                            result = op_fn(a, b)
                            self.W_up.data[unit, alu_base + a] = S * 2
                            self.W_up.data[unit, out_base + b] = S * 2
                            for other in range(16):
                                # Penalize positive nonmatching nibbles, but
                                # keep the weight small enough that negative
                                # cleanup residue cannot become evidence and
                                # fan out across many units.
                                if other != b and other != 0:
                                    self.W_up.data[unit, out_base + other] = -S * 0.5
                            self.W_up.data[unit, BD.IS_BYTE] = S
                            self.W_up.data[unit, BD.H1 + 1] = S
                            for wrong_h1 in (0, 2, 3, 4, 5, 6):
                                self.W_up.data[unit, BD.H1 + wrong_h1] = -S * 10_000_000
                            self.W_up.data[unit, BD.BYTE_INDEX_0] = S
                            self.W_up.data[unit, BD.BYTE_INDEX_1] = -S * 10
                            self.W_up.data[unit, BD.BYTE_INDEX_2] = -S * 10
                            self.W_up.data[unit, BD.BYTE_INDEX_3] = -S * 10
                            self.W_up.data[unit, op_dim] = S * 10
                            self.W_up.data[unit, suppress_op_dim] = -S * 10
                            self.W_up.data[unit, BD.CMP + 7] = -S * 1000
                            self.W_up.data[unit, BD.TEMP + 3] = -S
                            self.W_up.data[unit, BD.NEXT_SE] = -S * 10000
                            for marker_dim in (
                                BD.MARK_AX,
                                BD.MARK_PC,
                                BD.MARK_SP,
                                BD.MARK_BP,
                                BD.MARK_STACK0,
                                BD.MARK_MEM,
                                BD.MARK_SE,
                            ):
                                self.W_up.data[unit, marker_dim] = -S * 10_000_000
                            # Nonzero old-nibble rules must see actual
                            # evidence for their lane. Otherwise the common
                            # zero-lane passthrough is enough to activate all
                            # nonzero alternatives under amplified SUB rows,
                            # and their cleanup writes erase the true zero.
                            threshold = (
                                28.55
                                if op_dim == BD.TEMP + 9 and b != 0
                                else 26.35
                            )
                            self.b_up.data[unit] = -S * threshold
                            self.W_gate.data[unit, op_dim] = 1.0
                            if b != result:
                                self.W_down.data[out_base + b, unit] = -50.0 / S
                            if b != 0 and result != 0:
                                self.W_down.data[out_base + 0, unit] = -50.0 / S
                            self.W_down.data[out_base + result, unit] = 2.0 / S
                            unit += 1

            # Borrow continuation for later AX bytes. The first borrow stage
            # emits byte 1 as 0xff for underflowing SUB. If the autoregressive
            # stream's just-emitted byte is 0xff and the original SUB borrow is
            # still relayed, continue emitting 0xff for bytes 2 and 3. This
            # leaves cases like 0x0100 - 1 alone because byte 1 is 0x00.
            for byte_dim in (BD.BYTE_INDEX_1, BD.BYTE_INDEX_2):
                wrong_byte_dims = [
                    dim for dim in (
                        BD.BYTE_INDEX_0,
                        BD.BYTE_INDEX_1,
                        BD.BYTE_INDEX_2,
                        BD.BYTE_INDEX_3,
                    )
                    if dim != byte_dim
                ]
                for out_base in (BD.OUTPUT_LO, BD.OUTPUT_HI):
                    self.W_up.data[unit, BD.CARRY + 2] = S
                    self.W_up.data[unit, BD.CARRY + 1] = -S * 10
                    self.W_up.data[unit, BD.CMP + 7] = -S * 1000
                    self.W_up.data[unit, BD.IS_BYTE] = S
                    self.W_up.data[unit, BD.H1 + 1] = S
                    for wrong_h1 in (0, 2, 3, 4, 5, 6):
                        self.W_up.data[unit, BD.H1 + wrong_h1] = -S * 10_000_000
                    self.W_up.data[unit, byte_dim] = S
                    self.W_up.data[unit, BD.CLEAN_EMBED_LO + 15] = S * 2
                    self.W_up.data[unit, BD.CLEAN_EMBED_HI + 15] = S * 2
                    self.W_up.data[unit, BD.TEMP + 3] = -S * 10
                    self.W_up.data[unit, BD.NEXT_SE] = -S * 10000
                    for wrong_dim in wrong_byte_dims:
                        self.W_up.data[unit, wrong_dim] = -S * 10
                    for marker_dim in (
                        BD.MARK_AX,
                        BD.MARK_PC,
                        BD.MARK_SP,
                        BD.MARK_BP,
                        BD.MARK_STACK0,
                        BD.MARK_MEM,
                        BD.MARK_SE,
                    ):
                        self.W_up.data[unit, marker_dim] = -S * 10_000_000
                    self.b_up.data[unit] = -S * 7.5
                    self.W_gate.data[unit, BD.CONST] = 1.0
                    self.W_down.data[out_base + 0, unit] = -2.0 / S
                    self.W_down.data[out_base + 15, unit] = 2.0 / S
                    unit += 1


class BitwiseBytePropagationPostOp(PureFFN):
    """Post-op for L10 that computes AND/OR/XOR at AX byte positions (bytes 1-3).

    After head 4 copies STACK0 byte values to ALU_LO/HI at byte positions,
    this reads AX operand from OUTPUT and STACK0 operand from ALU, then
    computes per-nibble bitwise result.

    Phase 0 conversion (2026-05-09): subclasses PureFFN.
    """

    def __init__(self, d_model=512, S=100.0, dim_positions=None):
        object.__setattr__(self, '_pending_S', S)
        # Stash dim_positions BEFORE super().__init__ so _bake_weights (called
        # from PureFFN.__init__) sees compact positions if provided.
        object.__setattr__(self, '_pending_dim_positions', dim_positions)
        super().__init__(dim=d_model, hidden_dim=1536)
        self.d_model = d_model
        self.S = S

    def _bake_weights(self):
        S = self._pending_S
        BD = _resolve_bd(getattr(self, '_pending_dim_positions', None))

        unit = 0
        with torch.no_grad():
            # FIX 2026-05-08: Use relayed opcode flags (TEMP[4..6]) instead of original OP_* dims.
            # The original OP_AND/OR/XOR are only set at markers, not at byte positions.
            # L7 head 5 relays them to TEMP[4..6] at AX byte positions.
            ops = [
                (BD.TEMP + 4, lambda a, b: a & b),  # Relayed OP_AND
                (BD.TEMP + 5, lambda a, b: a | b),  # Relayed OP_OR
                (BD.TEMP + 6, lambda a, b: a ^ b),  # Relayed OP_XOR
            ]
            for op_dim, op_fn in ops:
                for a_lo in range(16):
                    for b_lo in range(16):
                        r = op_fn(a_lo, b_lo)
                        self.W_up.data[unit, BD.OUTPUT_LO + a_lo] = S
                        self.W_up.data[unit, BD.IS_BYTE] = S
                        self.W_up.data[unit, BD.H1 + 1] = S
                        self.W_up.data[unit, op_dim] = S * 8
                        self.W_up.data[unit, BD.ALU_LO + b_lo] = S
                        # OUTPUT/ALU residuals at the AX marker can be much
                        # larger than one-hot activations; weak marker
                        # suppression lets this byte-only post-op corrupt the
                        # marker result on traces as simple as IMM; EXIT.
                        self.W_up.data[unit, BD.MARK_AX] = -S * 10000
                        # PC/SP/BP/STACK0/MEM/SE markers can also carry very
                        # large OUTPUT residuals (for example first-step JMP
                        # PC targets). This post-op is valid only at AX byte
                        # positions, so block all marker tokens explicitly.
                        self.W_up.data[unit, BD.MARK_PC] = -S * 10000
                        self.W_up.data[unit, BD.MARK_SP] = -S * 10000
                        self.W_up.data[unit, BD.MARK_BP] = -S * 10000
                        self.W_up.data[unit, BD.MARK_STACK0] = -S * 10000
                        self.W_up.data[unit, BD.MARK_MEM] = -S * 10000
                        self.W_up.data[unit, BD.MARK_SE] = -S * 10000
                        # Byte positions do not carry MARK_* flags. This
                        # post-op is valid only in the AX byte span; block
                        # other register byte spans explicitly so large late
                        # OUTPUT/TEMP residuals cannot masquerade as bitwise
                        # AX-byte evidence.
                        self.W_up.data[unit, BD.H1 + 0] = -S * 10000
                        self.W_up.data[unit, BD.H1 + 2] = -S * 10000
                        self.W_up.data[unit, BD.H1 + 3] = -S * 10000
                        # ADD/SUB/wide ALU rows can carry very large OUTPUT
                        # and ALU residue at AX bytes. This detector is valid
                        # only for relayed bitwise op rows (TEMP[4..6]).
                        self.W_up.data[unit, BD.TEMP + 8] = -S * 10000
                        self.W_up.data[unit, BD.TEMP + 9] = -S * 10000
                        self.W_up.data[unit, BD.CARRY + 1] = -S * 10000
                        self.W_up.data[unit, BD.CARRY + 2] = -S * 10000
                        self.W_up.data[unit, BD.CARRY + 3] = -S * 10000
                        # Autoregressive alignment: BYTE_INDEX_0/1/2 queries
                        # predict AX bytes 1/2/3. Only BYTE_INDEX_3 predicts
                        # the following marker and must be blocked here.
                        self.W_up.data[unit, BD.BYTE_INDEX_3] = -S * 10
                        # In the compiler layout this post-op also exists in
                        # the dependency-assigned combined tail layer. By then
                        # L14/L15 may have large zeroing residuals on STACK0
                        # bytes; block the BP/STACK0 span explicitly so those
                        # residuals cannot satisfy this AX-byte detector.
                        self.W_up.data[unit, BD.H4 + 3] = -S * 10000
                        self.b_up.data[unit] = -S * 12.5
                        # ALU is part of the detector, not the gate: stale
                        # cleared nibbles can be negative, and a negative
                        # SwiGLU gate would invert the write instead of
                        # suppressing it.
                        self.W_gate.data[unit, op_dim] = 1.0
                        self.W_down.data[BD.OUTPUT_LO + a_lo, unit] = -2.0 / S
                        self.W_down.data[BD.OUTPUT_LO + r, unit] = 2.0 / S
                        unit += 1
                for a_hi in range(16):
                    for b_hi in range(16):
                        r = op_fn(a_hi, b_hi)
                        self.W_up.data[unit, BD.OUTPUT_HI + a_hi] = S
                        self.W_up.data[unit, BD.IS_BYTE] = S
                        self.W_up.data[unit, BD.H1 + 1] = S
                        self.W_up.data[unit, op_dim] = S * 8
                        self.W_up.data[unit, BD.ALU_HI + b_hi] = S
                        self.W_up.data[unit, BD.MARK_AX] = -S * 10000
                        self.W_up.data[unit, BD.MARK_PC] = -S * 10000
                        self.W_up.data[unit, BD.MARK_SP] = -S * 10000
                        self.W_up.data[unit, BD.MARK_BP] = -S * 10000
                        self.W_up.data[unit, BD.MARK_STACK0] = -S * 10000
                        self.W_up.data[unit, BD.MARK_MEM] = -S * 10000
                        self.W_up.data[unit, BD.MARK_SE] = -S * 10000
                        self.W_up.data[unit, BD.H1 + 0] = -S * 10000
                        self.W_up.data[unit, BD.H1 + 2] = -S * 10000
                        self.W_up.data[unit, BD.H1 + 3] = -S * 10000
                        # See low-nibble branch above.
                        self.W_up.data[unit, BD.TEMP + 8] = -S * 10000
                        self.W_up.data[unit, BD.TEMP + 9] = -S * 10000
                        self.W_up.data[unit, BD.CARRY + 1] = -S * 10000
                        self.W_up.data[unit, BD.CARRY + 2] = -S * 10000
                        self.W_up.data[unit, BD.CARRY + 3] = -S * 10000
                        # See low-nibble branch above for the AR alignment.
                        self.W_up.data[unit, BD.BYTE_INDEX_3] = -S * 10
                        self.W_up.data[unit, BD.H4 + 3] = -S * 10000
                        self.b_up.data[unit] = -S * 12.5
                        self.W_gate.data[unit, op_dim] = 1.0
                        self.W_down.data[BD.OUTPUT_HI + a_hi, unit] = -2.0 / S
                        self.W_down.data[BD.OUTPUT_HI + r, unit] = 2.0 / S
                        unit += 1


class DivModModule(nn.Module):
    """DIV/MOD module supporting two modes:

    Mode 'lookup' (default): Pure FFN using full (a,b) lookup with SwiGLU.
        - 131,072 hidden units (65,536 DIV + 65,536 MOD)
        - Each unit detects specific (a,b) pair via 4-way AND
        - Completely pure FFN, no exp/log/floor ops

    Mode 'efficient': Softmax1 reciprocal + fp32 MAGIC floor trick.
        - Much smaller parameter count
        - Uses softmax1 to compute 1/divisor
        - Uses fp32 MAGIC trick for floor extraction
        - Not pure FFN (uses exp, log) but more efficient
    """

    # fp32 MAGIC: at 1.5*2^23 scale, ULP = 1 (2^23 is at boundary where ULP=0.5 still applies)
    MAGIC32 = 1.5 * float(2**23)  # 12582912.0

    def __init__(self, d_model=512, S=100.0, mode='efficient'):
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.mode = mode

        if mode == 'lookup':
            self._init_lookup_mode(S)
        elif mode == 'efficient':
            self._init_efficient_mode(S)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'lookup' or 'efficient'.")

    def _init_lookup_mode(self, S):
        """Initialize full (a,b) lookup table mode.

        Uses 5-way AND detection: (a_lo, a_hi, b_lo, b_hi, OP_xxx).
        This ensures units only activate when BOTH operands match AND
        the correct operation flag is set, preventing cross-talk between
        DIV and MOD operations.
        """
        n_units = 256 * 256 * 2  # DIV + MOD

        # FFN weights: up, gate, down
        self.W_up = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(n_units))
        self.W_gate = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(n_units))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, n_units))

        BD = _SetDim
        unit = 0

        # DIV units: detect (a, b, OP_DIV), output quotient nibbles
        # Relay convention: ALU = stack (dividend), AX_CARRY = AX (divisor)
        # So: a = ALU (dividend), b = AX_CARRY (divisor)
        # Result: q = a // b = stack // AX
        for a in range(256):
            for b in range(256):
                a_lo, a_hi = a % 16, a // 16
                b_lo, b_hi = b % 16, b // 16

                q = a // b if b > 0 else 0
                q_lo, q_hi = q % 16, q // 16

                # up: 5-way AND on (MARK_AX, a_lo, a_hi, b_lo, b_hi)
                # a = ALU (dividend = stack), b = AX_CARRY (divisor = AX)
                # When all 5 match: up = 5S - 4.5S = 0.5S > 0
                # When any mismatch: up <= 4S - 4.5S = -0.5S < 0, silu ~= 0
                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S       # dividend lo (stack)
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S       # dividend hi (stack)
                self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S  # divisor lo (AX)
                self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S  # divisor hi (AX)
                self.b_up.data[unit] = -4.5 * S

                # gate: OP_DIV check (gating by opcode)
                self.W_gate.data[unit, BD.OP_DIV] = 1.0
                self.b_gate.data[unit] = 0.0  # gate = OP_DIV (≈0 when DIV not active)

                # down: Write to OUTPUT_LO[q_lo] and OUTPUT_HI[q_hi]
                self.W_down.data[BD.OUTPUT_LO + q_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + q_hi, unit] = 1.0

                unit += 1

        # MOD units: detect (a, b, OP_MOD), output remainder nibbles
        # Relay convention: ALU = stack (dividend), AX_CARRY = AX (divisor)
        # So: a = ALU (dividend), b = AX_CARRY (divisor)
        # Result: r = a % b = stack % AX
        for a in range(256):
            for b in range(256):
                a_lo, a_hi = a % 16, a // 16
                b_lo, b_hi = b % 16, b // 16

                r = a % b if b > 0 else 0
                r_lo, r_hi = r % 16, r // 16

                # up: 5-way AND on (MARK_AX, a_lo, a_hi, b_lo, b_hi)
                # a = ALU (dividend = stack), b = AX_CARRY (divisor = AX)
                # When all 5 match: up = 5S - 4.5S = 0.5S > 0
                # When any mismatch: up <= 4S - 4.5S = -0.5S < 0, silu ~= 0
                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S       # dividend lo (stack)
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S       # dividend hi (stack)
                self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S  # divisor lo (AX)
                self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S  # divisor hi (AX)
                self.b_up.data[unit] = -4.5 * S

                # gate: OP_MOD check (gating by opcode)
                self.W_gate.data[unit, BD.OP_MOD] = 1.0
                self.b_gate.data[unit] = 0.0  # gate = OP_MOD (≈0 when MOD not active)

                # down: Write to OUTPUT_LO[r_lo] and OUTPUT_HI[r_hi]
                self.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0

                unit += 1

    def _init_efficient_mode(self, S):
        """Initialize efficient softmax1 reciprocal mode.

        This mode uses a small FFN for output formatting + Python math for compute.
        Hidden units: 32 for DIV output + 32 for MOD output = 64 total.
        """
        n_units = 64  # 32 per op for nibble output encoding

        self.W_up = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_up = nn.Parameter(torch.zeros(n_units))
        self.W_gate = nn.Parameter(torch.zeros(n_units, self.d_model))
        self.b_gate = nn.Parameter(torch.zeros(n_units))
        self.W_down = nn.Parameter(torch.zeros(self.d_model, n_units))

        # Initialize output encoding (we compute div/mod result in forward,
        # then use FFN to encode result into nibbles)
        BD = _SetDim

        # DIV output: units 0-15 write OUTPUT_LO, 16-31 write OUTPUT_HI
        for v in range(16):
            # LO nibble: gate on DIV result having this LO value
            self.W_up.data[v, BD.OP_DIV] = S
            self.b_up.data[v] = -0.5 * S
            self.W_down.data[BD.OUTPUT_LO + v, v] = 1.0

            # HI nibble
            self.W_up.data[16 + v, BD.OP_DIV] = S
            self.b_up.data[16 + v] = -0.5 * S
            self.W_down.data[BD.OUTPUT_HI + v, 16 + v] = 1.0

        # MOD output: units 32-47 write OUTPUT_LO, 48-63 write OUTPUT_HI
        for v in range(16):
            self.W_up.data[32 + v, BD.OP_MOD] = S
            self.b_up.data[32 + v] = -0.5 * S
            self.W_down.data[BD.OUTPUT_LO + v, 32 + v] = 1.0

            self.W_up.data[48 + v, BD.OP_MOD] = S
            self.b_up.data[48 + v] = -0.5 * S
            self.W_down.data[BD.OUTPUT_HI + v, 48 + v] = 1.0

    def _extract_operands(self, x):
        """Extract a and b operands from nibble encoding."""
        BD = _SetDim
        # x shape: (B, S, D) - we work on position 0

        # Extract nibbles via argmax on one-hot
        a_lo = x[:, 0, BD.ALU_LO:BD.ALU_LO + 16].argmax(dim=-1)
        a_hi = x[:, 0, BD.ALU_HI:BD.ALU_HI + 16].argmax(dim=-1)
        b_lo = x[:, 0, BD.AX_CARRY_LO:BD.AX_CARRY_LO + 16].argmax(dim=-1)
        b_hi = x[:, 0, BD.AX_CARRY_HI:BD.AX_CARRY_HI + 16].argmax(dim=-1)

        a = a_lo + 16 * a_hi
        b = b_lo + 16 * b_hi
        return a.float(), b.float()

    def _compute_div_mod_efficient(self, a, b):
        """Compute DIV and MOD using fp32 MAGIC floor trick.

        The MAGIC trick exploits IEEE 754 representation: at scale 2^23,
        fp32's ULP (unit in last place) = 1, so only integers are representable.
        Adding MAGIC forces rounding to nearest integer.

        floor(x) = ((x - 0.5) + MAGIC) - MAGIC

        This shifts x down by 0.5 so round-to-nearest becomes floor.

        Steps:
        1. Compute quotient: q_float = a / b
        2. Floor via MAGIC: q = ((q_float - 0.5) + MAGIC) - MAGIC
        3. MOD = a - b * q
        """
        # Handle div-by-zero: clamp b >= 1 for division
        b_safe = b.clamp(min=1.0)

        # True division to get float quotient
        q_float = a / b_safe

        # fp32 MAGIC floor trick: subtract (0.5 - eps) so round-to-nearest = floor
        # MAGIC = 2^23 = 8388608, at this scale ULP = 1
        # eps=0.001 avoids round-to-even issues without rounding up near-integers
        q_shifted = q_float - 0.5 + 0.001
        q_floor = (q_shifted + self.MAGIC32) - self.MAGIC32
        q_floor = q_floor.clamp(min=0, max=255)

        # Compute remainder: r = a - b * q
        r = a - b_safe * q_floor

        # Handle div-by-zero case: q=0, r=0 when b=0
        zero_divisor = (b < 0.5)
        q_floor = torch.where(zero_divisor, torch.zeros_like(q_floor), q_floor)
        r = torch.where(zero_divisor, torch.zeros_like(r), r)

        return q_floor.long(), r.long()

    def forward(self, x):
        """Forward pass - dispatches to lookup or efficient mode."""
        if self.mode == 'lookup':
            return self._forward_lookup(x)
        else:
            return self._forward_efficient(x)

    def _forward_lookup(self, x):
        """Pure FFN forward: SwiGLU with residual.

        Gate OUTPUT dimensions by MARK_AX to prevent DivMod from overwriting
        OUTPUT at non-AX positions (e.g., PC marker on first step).

        The 6-way AND provides first line of defense, but isn't perfect due to
        continuous ALU values. Selective OUTPUT gating ensures correctness.
        """
        BD = _SetDim
        up = F.linear(x, self.W_up, self.b_up)
        gate = F.linear(x, self.W_gate, self.b_gate)
        hidden = F.silu(up) * gate
        delta = F.linear(hidden, self.W_down)

        # Gate ONLY OUTPUT dimensions (not entire delta) by MARK_AX
        # This prevents DivMod from writing to OUTPUT at PC/SP/BP markers
        # DivMod only writes to OUTPUT_LO[0:16] and OUTPUT_HI[0:16]
        mark_ax_gate = x[..., BD.MARK_AX:BD.MARK_AX+1]  # (B, S, 1)

        # Apply gate selectively to OUTPUT dimensions only
        delta[..., BD.OUTPUT_LO:BD.OUTPUT_LO+16] *= mark_ax_gate
        delta[..., BD.OUTPUT_HI:BD.OUTPUT_HI+16] *= mark_ax_gate

        return x + delta

    def _forward_efficient(self, x):
        """Efficient forward: compute div/mod then encode to nibbles."""
        BD = _SetDim
        B, S, D = x.shape

        # Extract operands
        a, b = self._extract_operands(x)

        # Compute div and mod
        q, r = self._compute_div_mod_efficient(a, b)

        # Check if DIV or MOD operation is active
        div_active = x[:, 0, BD.OP_DIV] > 0.5
        mod_active = x[:, 0, BD.OP_MOD] > 0.5

        # Build output delta
        delta = torch.zeros_like(x)

        # Write DIV result nibbles
        q_lo = q % 16
        q_hi = q // 16
        for i in range(B):
            if div_active[i]:
                delta[i, 0, BD.OUTPUT_LO + q_lo[i]] = 1.0
                delta[i, 0, BD.OUTPUT_HI + q_hi[i]] = 1.0

        # Write MOD result nibbles
        r_lo = r % 16
        r_hi = r // 16
        for i in range(B):
            if mod_active[i]:
                delta[i, 0, BD.OUTPUT_LO + r_lo[i]] = 1.0
                delta[i, 0, BD.OUTPUT_HI + r_hi[i]] = 1.0

        return x + delta


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by common decoder-only LLM blocks."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    """Transformer decoder block: attention + FFN.

    Both components include residual connections,
    so the block is sequential composition.
    """

    def __init__(self, attn, ffn, use_rms_norm=False, rms_norm_eps=1e-6):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.use_rms_norm = bool(use_rms_norm)
        if self.use_rms_norm:
            dim = getattr(attn, "dim", None)
            if dim is None:
                dim = getattr(ffn, "dim")
            self.attn_norm = RMSNorm(dim, eps=rms_norm_eps)
            self.ffn_norm = RMSNorm(dim, eps=rms_norm_eps)
        self.post_ops = nn.ModuleList()

    def forward(self, x, kv_cache=None, x_is_new_only=False):
        if self.use_rms_norm:
            attn_in = self.attn_norm(x)
            attn_out = self.attn(
                attn_in, kv_cache=kv_cache, x_is_new_only=x_is_new_only
            )
            x = x + (attn_out - attn_in)

            ffn_in = self.ffn_norm(x)
            ffn_out = self.ffn(ffn_in)
            x = x + (ffn_out - ffn_in)
        else:
            x = self.attn(x, kv_cache=kv_cache, x_is_new_only=x_is_new_only)
            x = self.ffn(x)
        for op in self.post_ops:
            x = op(x)
        return x


class AutoregressiveVM(nn.Module):
    """Decoder-only transformer for VM execution.

    Composes existing PureAttention/PureFFN layers into a standard
    autoregressive transformer. All VM computation (instruction fetch,
    register read/write, ALU, memory, PC update) is handled through
    the transformer's weights.

    Token format per VM step (35 tokens):
        REG_PC  + 4 value bytes     (5 tokens)
        REG_AX  + 4 value bytes     (5 tokens)
        REG_SP  + 4 value bytes     (5 tokens)
        REG_BP  + 4 value bytes     (5 tokens)
        STACK0  + 4 value bytes     (5 tokens)
        MEM     + 4 addr + 4 value  (9 tokens)
        STEP_END                    (1 token)

    Note: Produces correct output only after weight setting. Initial
    weights are zero (FFN/attention are identity via residual).
    """

    def __init__(
        self,
        vocab_size=None,
        d_model=DEFAULT_D_MODEL,
        n_layers=17,  # Updated from 16 for LEV Phase 3 (L16 routing layer)
        n_heads=DEFAULT_N_HEADS,  # REVERTED from 16: HD=32 broke attention score budgets
        ffn_hidden=DEFAULT_FFN_HIDDEN,
        max_seq_len=1024,  # PERF: reduced from 4096; Phase 1 contexts are <100 tokens (1024 leaves headroom)
        dim_positions=None,
        use_flash_attention=True,
        positional_encoding=None,
        attention_normalization=None,
        use_rms_norm=None,
        rms_norm_eps=None,
        rope_base=None,
        alibi_base_heads=None,
    ):
        super().__init__()
        if vocab_size is None:
            vocab_size = Token.VOCAB_SIZE
        try:
            from .config import get_config
            config = get_config()
        except ImportError:
            config = None

        if positional_encoding is None:
            positional_encoding = (
                config.positional_encoding if config is not None else "alibi"
            )
        if attention_normalization is None:
            attention_normalization = (
                config.attention_normalization
                if config is not None
                else "softmax1"
            )
        if use_rms_norm is None:
            use_rms_norm = config.use_rms_norm if config is not None else False
        if rms_norm_eps is None:
            rms_norm_eps = config.rms_norm_eps if config is not None else 1e-6
        if rope_base is None:
            rope_base = config.rope_base if config is not None else 10000.0

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_flash_attention = use_flash_attention
        self.positional_encoding = positional_encoding
        self.attention_normalization = attention_normalization
        self.use_rms_norm = bool(use_rms_norm)
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        # Pre-widen head count used as the ALiBi-slope BASE so the
        # head-dim-preserving auto-widen's trailing padding heads do not shift
        # existing heads' slopes (see ``AutoregressiveAttention.__init__``).
        # ``None`` => slope base == ``n_heads`` (byte-identical to the
        # historical un-threaded path). Also read by num_heads-keyed layout
        # logic that must see the pre-widen geometry.
        self._alibi_base_heads = (
            int(alibi_base_heads) if alibi_base_heads is not None
            else n_heads
        )

        # Compiler-allocated dim_positions (None => fall back to _SetDim for
        # backward-compat callers that construct AutoregressiveVM directly).
        # Used by NeuralVMEmbedding token-time injection (ADDR_KEY, MEM_*).
        self.dim_positions = dim_positions if dim_positions is not None else _SetDim

        # Use NeuralVMEmbedding with integrated augmentations.
        # max_seq_len is passed through so the embedding can pre-allocate its
        # ADDR_KEY positional-encoding buffer at construction time (ONNX trace
        # cleanliness — no dynamic capacity check).
        self.embed = NeuralVMEmbedding(
            vocab_size, d_model, dim_positions=dim_positions, max_seq_len=max_seq_len
        )

        # ``ffn_hidden`` accepts either a scalar (every block gets that
        # many hidden units — the legacy contract; relies on
        # ``_right_size_ffns`` to trim post-bake) or a dict[int, int] of
        # per-block widths from the compiler's ``ModelLayout.ffn_widths``.
        # Blocks missing from the dict fall back to ``default_hidden`` so
        # partial annotation coverage is safe — un-annotated blocks still
        # get trimmed by ``_right_size_ffns``.
        if isinstance(ffn_hidden, dict):
            ffn_widths = ffn_hidden
            default_hidden = 4096
        else:
            ffn_widths = {}
            default_hidden = ffn_hidden

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    attn=AutoregressiveAttention(
                        d_model, num_heads=n_heads, max_seq_len=max_seq_len, layer_idx=i,
                        use_flash_attention=use_flash_attention,
                        positional_encoding=positional_encoding,
                        attention_normalization=attention_normalization,
                        rope_base=rope_base,
                        alibi_base_heads=self._alibi_base_heads,
                    ),
                    ffn=PureFFN(d_model, ffn_widths.get(i, default_hidden)),
                    use_rms_norm=use_rms_norm,
                    rms_norm_eps=rms_norm_eps,
                )
                for i in range(n_layers)
            ]
        )

        self.head = nn.Linear(d_model, vocab_size)

    def sparsify(self):
        """Convert all weight matrices to COO sparse format for faster inference."""
        for block in self.blocks:
            block.attn.sparsify()
            block.ffn.sparsify()
        self.head.weight = nn.Parameter(
            self.head.weight.data.to_sparse_coo().coalesce()
        )

    def compact(self, block_size=1, compact_attn=False):
        """Compact FFN layers to dense sub-matrices of active units.

        FFN: prunes to active hidden units (4096 → ~500-2000).

        Args:
            block_size: Align to blocks of this size for better vectorization.
                1 = minimal, 32/64 = hardware-friendly alignment.
            compact_attn: Also compact attention (experimental — gather overhead
                can outweigh savings for small context sizes).
        """
        for block in self.blocks:
            block.ffn.compact(block_size=block_size)
            if compact_attn:
                block.attn.compact(block_size=block_size)

    def compact_moe(self, opcode_range=None, relay_maps=None, pure_neural=False):
        """Convert all eligible FFN layers to standard top-K MoE modules.

        After compact(), each block's FFN is partitioned into per-opcode
        experts. The block's ``.ffn`` is then REPLACED with a ``StandardMoEFFN``
        that dispatches top-K experts by reading the
        opcode-onehot routing signal directly from the activation tensor
        (``x[:, :, opcode_dim]``).

        This is the spec-compliant path: no Python-side dispatch, no weight
        swapping between forward calls, ONNX-traceable. The previous
        ``set_active_opcode`` weight-swap path has been retired.

        Args:
            opcode_range: range of opcode one-hot dims (default: 262-296).
            relay_maps: dict mapping layer_index -> relay_map for that layer.
                Default: L6 CMP relay map accounting for position-dependent
                CMP semantics (head 0/1/4 at PC/SE, head 6 at SP/STACK0).
            pure_neural: Deprecated no-op. Forwarded to ``StandardMoEFFN`` for
                back-compat but has no effect on routing — runtime forward
                is always standard top-K MoE.
        """
        from .pure_moe import build_standard_moe_from_compact_partition

        # BD_SETDIM_HARDCODE_AUDIT M5: this ``BD = _SetDim`` assignment is
        # structurally correct. The ``D(name)`` resolver below checks
        # ``self.dim_positions`` (the compact / compiler-allocated layout)
        # *first* and only falls back to ``getattr(BD, name)`` when no
        # per-block layout is in scope (e.g. legacy hand-set test fixtures).
        # See the explicit ``isinstance(self.dim_positions, dict)`` branch
        # below and the ``D(name)`` helper a few lines down.
        BD = _SetDim
        # Resolve opcode-flag dims from the dim_positions layout (compiler-
        # allocated) when available; fall back to the hand-set _SetDim block
        # at 262-296. Without this the partition would scan the wrong column
        # range and silently return no opcode-dependent units.
        if opcode_range is None:
            if isinstance(self.dim_positions, dict):
                opcode_range = [
                    v for k, v in self.dim_positions.items()
                    if k.startswith("OP_")
                ]
            else:
                opcode_range = range(262, 296)
        if relay_maps is None:
            # CMP dims are overloaded at different positions:
            #   CMP[0]: IS_JMP at PC (head 0) + PSH at SP/STACK0 (head 6)
            #   CMP[1]: IS_EXIT at SE (head 1) + ADJ at SP/STACK0 (head 6)
            #   CMP[2]: OP_BZ at PC (head 4) + ENT at SP/STACK0 (head 6)
            #   CMP[3]: OP_BNZ at PC (head 4) + POP group at SP/STACK0 (head 6)
            #   CMP[4]: AX_LO_IS_ZERO at PC (head 4) + JSR at SP/STACK0 (head 6)
            #   CMP[5]: AX_HI_IS_ZERO at PC (head 4)
            # Map each CMP to ALL opcodes whose units depend on it.
            def D(name):
                if isinstance(self.dim_positions, dict) and name in self.dim_positions:
                    return self.dim_positions[name]
                return getattr(BD, name)
            CMP = D("CMP")
            pop_ops = [
                D("OP_ADD"), D("OP_SUB"), D("OP_MUL"), D("OP_DIV"), D("OP_MOD"),
                D("OP_EQ"), D("OP_NE"), D("OP_LT"), D("OP_GT"), D("OP_LE"), D("OP_GE"),
                D("OP_OR"), D("OP_XOR"), D("OP_AND"), D("OP_SHL"), D("OP_SHR"),
                D("OP_SI"), D("OP_SC"),
            ]
            relay_maps = {
                6: {
                    CMP + 0: [D("OP_JMP"), D("OP_PSH")],
                    CMP + 1: [D("OP_EXIT"), D("OP_ADJ")],
                    CMP + 2: [D("OP_ENT"), D("OP_BZ")],
                    CMP + 3: pop_ops + [D("OP_BNZ")],
                    CMP + 4: [D("OP_JSR"), D("OP_BZ"), D("OP_BNZ")],
                    CMP + 5: [D("OP_BZ"), D("OP_BNZ")],
                }
            }
        for i, block in enumerate(self.blocks):
            relay = relay_maps.get(i)
            ffn = block.ffn
            # Non-PureFFN blocks (ALU composites, FlattenedPureFFN wrappers,
            # already-converted MoE blocks) never participated in the legacy
            # MoE path and are skipped here too.
            if not isinstance(ffn, PureFFN):
                continue
            opcode_to_units, shared_indices = _partition_compact_ffn_by_opcode(
                ffn, opcode_range=opcode_range, relay_map=relay
            )
            if not opcode_to_units:
                continue
            # The tightener must only zero the actual OP_* router dims.
            # Relay dims such as CMP are ordinary dense-path inputs at
            # SP/STACK0/MEM positions; if a unit can fire from one without an
            # OP router value, routing that unit into an opcode expert drops
            # required non-PC contributions. Leave relay dims live so such
            # units are conservatively promoted to the shared expert.
            standard_moe = build_standard_moe_from_compact_partition(
                ffn,
                opcode_to_units=opcode_to_units,
                shared_indices=shared_indices,
                dim=ffn.W_up.shape[1],
                pure_neural=pure_neural,
                opcode_dims_all=list(opcode_range),
                relay_dims=(),
            )
            block.ffn = standard_moe.to(device=ffn.W_up.device, dtype=ffn.W_up.dtype)

    def save_compact(self, path):
        """Save compacted model to disk (avoids re-computing compact on load)."""
        torch.save(self, path)

    @staticmethod
    def load_compact(path):
        """Load a previously compacted model from disk."""
        return torch.load(path, weights_only=False)

    def forward(self, token_ids, kv_cache=None, cached_prefix_len=0,
                stop_after_block=None):
        """Forward pass: token IDs -> logits.

        Args:
            token_ids: [batch, seq] integer token IDs (full sequence).
            kv_cache: Optional LayerKVCache for incremental decoding.
            cached_prefix_len: Number of leading positions already represented
                in ``kv_cache``. When > 0, only the trailing positions
                ``token_ids[:, cached_prefix_len:]`` are run through the
                transformer blocks (Q/K/V computed only for new positions;
                attention pulls past K/V from the cache). The embedding is
                still computed over the FULL ``token_ids`` so context-sensitive
                augmentations (e.g. _inject_mem_exec_autoregressive scanning
                MEM markers) see all prior positions. Returns logits of shape
                ``[batch, seq - cached_prefix_len, vocab_size]`` when > 0.
            stop_after_block: probe-only. When an int ``b``, run the embedding
                + physical blocks ``0..b`` (inclusive) and return the
                **residual hidden state** ``[batch, seq, d_model]`` *before*
                the LM head, instead of logits. ``None`` (the default — and the
                only value any production/runner caller ever passes) is fully
                byte-identical to the prior behaviour: the early-return branch
                is never entered. This exists so
                ``tools/probe_groundtruth.py`` can read a block's residual via
                the model's own normally-computed returned tensor — no forward
                hooks, no weight overrides.

        Returns:
            ``[batch, seq, vocab_size]`` logits when ``cached_prefix_len == 0``,
            else ``[batch, seq - cached_prefix_len, vocab_size]``. When
            ``stop_after_block`` is set, returns the ``[batch, seq, d_model]``
            residual after that physical block instead.
        """
        # Pure forward pass: embed → blocks → head
        # All augmentations (ADDR_KEY, MEM_STORE) are inside NeuralVMEmbedding
        x = self.embed(token_ids)

        # Incremental path: slice off cached prefix before running blocks.
        # ``cached_prefix_len`` is a Python int, so the tracer would constant-
        # fold this branch to whichever value was used at export time (always
        # 0 in practice). Gate it behind the ONNX guard so the exported graph
        # always runs the full-sequence path; eager mode still gets the
        # incremental fast path when a runner passes ``cached_prefix_len>0``.
        if not torch.onnx.is_in_onnx_export() and cached_prefix_len > 0:
            x = x[:, cached_prefix_len:, :]
            x_is_new_only = True
        else:
            x_is_new_only = False

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache.get_layer_cache(i) if kv_cache is not None else None
            x = block(x, kv_cache=layer_cache, x_is_new_only=x_is_new_only)
            # Probe-only early return (stop_after_block is None on every
            # production path, so this branch is dead weight for the runner).
            # ``x`` here is the model's own post-block residual — exactly what
            # block i+1 would receive as input.
            if stop_after_block is not None and i == stop_after_block:
                return x

        if self.head.weight.is_sparse:
            return sparse_linear(x, self.head.weight, self.head.bias)
        return self.head(x)

    @torch.no_grad()
    def generate_next(self, context, temperature=0.0, kv_cache=None, use_incremental=True, max_context_window=512):
        """Generate next token via greedy or sampled decoding.

        Args:
            context: list of integer token IDs
            temperature: 0.0 = greedy (argmax), >0 = sample from softmax
            kv_cache: Optional KVCache for efficient generation (currently unused)
            use_incremental: If True, use context windowing to limit reprocessing
            max_context_window: Maximum context length to process (default 512)

        Returns:
            int: next token ID
        """
        # Apply context windowing to prevent O(n²) blowup
        # Keep last N tokens - transformer has causal attention so older tokens
        # have minimal impact on next token prediction
        if use_incremental and len(context) > max_context_window:
            context = context[-max_context_window:]
        elif len(context) > self.max_seq_len:
            context = context[-self.max_seq_len:]

        # Create tensor on same device as model
        device = next(self.parameters()).device
        token_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = self.forward(token_ids)[0, -1, :]
        if temperature <= 0.0:
            return logits.argmax(-1).item()
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()

    @torch.no_grad()
    def generate_next_batch(self, contexts):
        """Generate next token for multiple contexts simultaneously.

        Args:
            contexts: list of lists of token IDs (all must be same length)
        Returns:
            list of int: next token IDs (greedy argmax)
        """
        token_ids = torch.tensor(contexts, dtype=torch.long)  # [B, S]
        logits = self.forward(token_ids)  # [B, S, vocab]
        return logits[:, -1, :].argmax(-1).tolist()  # list of B ints

    @torch.no_grad()
    def generate_autoregressive(self, context, max_steps=10000, temperature=0.0):
        """True autoregressive generation: one token at a time.

        This is 100% autoregressive - each token gets a full forward pass
        through the entire model based on ALL previous tokens. No batch
        processing, no speculation - just pure sequential generation.

        Args:
            context: List of token IDs (initial context)
            max_steps: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            list: Extended context with generated tokens

        Note: This is MUCH slower than batch processing (speculative decoding),
        but represents true autoregressive generation where each token depends
        on a complete forward pass through all previous context.
        """
        context = list(context)  # Copy to avoid modifying input

        for step in range(max_steps):
            # Truncate if exceeds max length
            if len(context) > self.max_seq_len:
                context = context[-self.max_seq_len:]

            # Forward pass on ENTIRE context so far
            token_ids = torch.tensor([context], dtype=torch.long)
            logits = self.forward(token_ids)  # [1, len(context), vocab]

            # Predict NEXT token (only the last position)
            next_logits = logits[0, -1, :]  # [vocab]

            if temperature <= 0.0:
                # Greedy decoding
                next_token = next_logits.argmax(-1).item()
            else:
                # Sample with temperature
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            # Append and continue
            context.append(next_token)

            # Check for termination
            if next_token == Token.HALT:
                break

        return context

    @torch.no_grad()
    def generate_autoregressive_with_kv_cache(self, context, max_steps=10000,
                                              temperature=0.0, kv_cache=None):
        """Optimized autoregressive generation with KV cache.

        Like generate_autoregressive() but reuses KV cache to avoid
        recomputing attention for previous tokens. This is faster than
        naive autoregressive but still slower than batch processing.

        Represents a middle ground between purity and performance.

        Args:
            context: List of token IDs (initial context)
            max_steps: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            kv_cache: Optional KVCache instance (created if None)

        Returns:
            list: Extended context with generated tokens
        """
        context = list(context)  # Copy to avoid modifying input

        if kv_cache is None:
            from .kv_cache import KVCache
            max_len = len(context) + max_steps
            kv_cache = KVCache(
                max_batch_size=1,
                max_seq_len=min(max_len, self.max_seq_len)
            )

        # Initial forward pass on full context
        if len(context) > self.max_seq_len:
            context = context[-self.max_seq_len:]
        token_ids = torch.tensor([context], dtype=torch.long)
        logits = self.forward(token_ids, kv_cache=kv_cache)

        for step in range(max_steps):
            # Predict next token from last position
            next_logits = logits[0, -1, :]

            if temperature <= 0.0:
                next_token = next_logits.argmax(-1).item()
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            context.append(next_token)

            if next_token == Token.HALT:
                break

            # Forward pass on ONLY the new token (using KV cache)
            token_ids = torch.tensor([[next_token]], dtype=torch.long)
            logits = self.forward(token_ids, kv_cache=kv_cache)

        return context

    @torch.no_grad()
    def verify_speculative_step(self, context, draft_tokens):
        """Verify draft tokens against transformer in one forward pass.

        Args:
            context: list of token IDs (context before this step)
            draft_tokens: list of 35 predicted tokens

        Returns:
            int: number of accepted tokens (0..35). If 35, full match.
        """
        full = context + draft_tokens
        if len(full) > self.max_seq_len:
            full = full[-self.max_seq_len :]
        token_ids = torch.tensor([full], dtype=torch.long)
        logits = self.forward(token_ids)  # [1, S, vocab]

        ctx_len = len(full) - len(draft_tokens)
        accepted = 0
        for i, draft_tok in enumerate(draft_tokens):
            pred = logits[0, ctx_len - 1 + i, :].argmax(-1).item()
            if pred == draft_tok:
                accepted += 1
            else:
                break
        return accepted

    @torch.no_grad()
    def verify_speculative_batch(self, contexts_with_draft, draft_lens, context_lens=None, kv_cache=None):
        """Batched speculative verification.

        Args:
            contexts_with_draft: list of (context + draft_tokens) lists, all same length (may be padded)
            draft_lens: list of int, number of draft tokens per sequence
            context_lens: Optional list of int, actual context lengths (before draft). If None, calculated from lengths.
            kv_cache: Optional LayerKVCache for incremental decoding with eviction

        Returns:
            list of int: accepted token count per sequence
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        token_ids = torch.tensor(contexts_with_draft, dtype=torch.long, device=device)
        logits = self.forward(token_ids, kv_cache=kv_cache)  # [B, S, vocab]

        results = []
        for b in range(len(contexts_with_draft)):
            # Use actual context length if provided, otherwise calculate from padded length
            if context_lens is not None:
                ctx_len = context_lens[b]
            else:
                ctx_len = len(contexts_with_draft[b]) - draft_lens[b]

            accepted = 0
            for i in range(draft_lens[b]):
                pred = logits[b, ctx_len - 1 + i, :].argmax(-1).item()
                if contexts_with_draft[b][ctx_len + i] == pred:
                    accepted += 1
                else:
                    break
            results.append(accepted)
        return results


# =============================================================================
# Weight Setting - Real Neural VM Execution Through Transformer Weights
# =============================================================================


_OPCODE_VALUE_TO_NAME = None


def _opcode_value_to_name(op_value):
    """Reverse-lookup Opcode integer value -> short name (e.g. 25 -> 'ADD').

    `Opcode` is a plain class (not an enum), so we build the int->name map
    once via class-attribute introspection.
    """
    global _OPCODE_VALUE_TO_NAME
    if _OPCODE_VALUE_TO_NAME is None:
        _OPCODE_VALUE_TO_NAME = {
            getattr(Opcode, n): n
            for n in dir(Opcode)
            if not n.startswith("_") and isinstance(getattr(Opcode, n), int)
        }
    return _OPCODE_VALUE_TO_NAME.get(op_value)


def _partition_compact_ffn_by_opcode(ffn, opcode_range=None, relay_map=None):
    """Analyze a compacted ``PureFFN`` and group hidden units by opcode affinity.

    Used by ``AutoregressiveVM.compact_moe`` to drive the StandardMoEFFN
    construction. Each hidden unit is classified by inspecting its
    ``W_up``/``W_gate`` weights for opcode-onehot column activity
    (``> 0.5``). Units that fire only for a given opcode are grouped under
    that opcode dim; units with no opcode dependence are "shared".

    Args:
        ffn: A ``PureFFN`` (already ``compact()``-ed).
        opcode_range: Iterable of opcode-flag dims. The caller is responsible
            for supplying the right range — for the compiler-allocated layout
            this is the set of ``dim_positions["OP_*"]`` values; for the
            hand-set ``_SetDim`` layout it is ``range(262, 296)``.
        relay_map: Optional dict mapping additional W_up dims (e.g. CMP relay
            dims in L6) to opcode dims. Values can be an int or a sequence
            of ints (one unit may relay to multiple opcodes).

    Returns:
        ``(opcode_to_units, shared_indices)`` — a dict mapping each opcode
        dim to a sorted list of hidden-unit indices, plus a list of indices
        for the opcode-independent shared units. ``opcode_to_units`` may be
        empty (the FFN has no opcode-dependent units, e.g. it's a fetch /
        passthrough layer).
    """
    if opcode_range is None:
        opcode_range = range(262, 296)
    if relay_map is None:
        relay_map = {}

    W_up = ffn.W_up.data
    W_gate = ffn.W_gate.data
    H = W_up.shape[0]

    opcode_to_units: dict = {}
    shared_indices: list = []

    for i in range(H):
        opcodes = set()
        # Check W_up for positive opcode weights (silu activation gating).
        for d in opcode_range:
            if d < W_up.shape[1] and W_up[i, d].item() > 0.5:
                opcodes.add(d)
        # Check W_gate for significant opcode weights (value gating).
        for d in opcode_range:
            if d < W_gate.shape[1] and abs(W_gate[i, d].item()) > 0.5:
                opcodes.add(d)
        # Check relay dims in W_up (e.g. CMP[2] -> OP_ENT).
        for relay_dim, target in relay_map.items():
            if relay_dim < W_up.shape[1] and W_up[i, relay_dim].item() > 0.5:
                if isinstance(target, (list, tuple, set)):
                    opcodes.update(target)
                else:
                    opcodes.add(target)

        if opcodes:
            for d in opcodes:
                opcode_to_units.setdefault(d, []).append(i)
        else:
            shared_indices.append(i)

    return opcode_to_units, shared_indices


def _opcode_dim_from_positions(dim_positions, opcode_value):
    """Resolve an opcode value to its FFN flag dim from a positions source.

    `dim_positions` is either the `_SetDim` class (legacy/hand-set mode) or
    a dict[str, int] (compiler-allocated layout). For dicts, we look up the
    `OP_<NAME>` dim name; for the class, we delegate to `_SetDim.opcode_dim`
    which performs the same mapping via class attributes.
    """
    if isinstance(dim_positions, dict):
        name = _opcode_value_to_name(opcode_value)
        if name is None:
            return None
        return dim_positions.get(f"OP_{name}")
    return dim_positions.opcode_dim(opcode_value)


# Embedding dimension allocation for set weights (d_model=512)
class _SetDim:
    """Internal dimension allocation for set VM weights.

    d_model=512.  35-token step format:
      PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1) = 35
    """

    # --- Marker identity flags (set by embedding) ---
    MARK_PC = 0
    MARK_AX = 1
    MARK_SP = 2
    MARK_BP = 3
    MARK_MEM = 4
    MARK_SE = 5  # STEP_END or DATA_END
    IS_BYTE = 6
    IS_MARK = 7
    CONST = 8
    MARK_CS = 9  # CODE_START only
    MARK_SE_ONLY = 10  # STEP_END only (not DATA_END)
    MARK_STACK0 = 11  # STACK0 marker
    MARKS = [MARK_PC, MARK_AX, MARK_SP, MARK_BP, MARK_MEM, MARK_SE, MARK_CS]
    NUM_MARKERS = 7  # threshold heads still use the original 7

    # --- Address byte nibbles (gathered by L7/L14 attention) ---
    ADDR_B0_LO = 12  # dims 12-27  (16 one-hot)
    ADDR_B1_LO = 28  # dims 28-43
    ADDR_B2_LO = 44  # dims 44-59
    # Phase 9.C: ADDR_B0_LO_PREV_STEP retained — consumed by
    # ``make_lev_detector_head_op`` (control_flow_heads.py) as a string
    # literal in op.reads (and exercised by tests/test_lev_detector_head.py).
    # ADDR_B{1,2}_LO_PREV_STEP and OPCODE_BYTE_LO_PREV_STEP aliases
    # deleted — corpus readers migrated to SSA ``BASE.<writer>.-1``
    # spellings in Phase 9.B.
    ADDR_B0_LO_PREV_STEP = 12  # alias of ADDR_B0_LO (consumer: lev_detector_head)

    # --- Opcode byte staging (L5 head 1 → L5 FFN decode) ---
    # Separate from ALU_LO/HI to avoid residual collision with L7 operand gather
    OPCODE_BYTE_LO = 12  # reuse ADDR_B0_LO (unused in autoregressive)
    OPCODE_BYTE_HI = 28  # reuse ADDR_B1_LO (unused in autoregressive)

    # --- L0 threshold heads (8 heads for 39-token step) ---
    # Thresholds: [3.5, 4.5, 5.5, 9.5, 10.5, 14.5, 15.5, 19.5]
    H0 = 60
    H1 = 67
    # AX byte-1 DUMP carry alias: a distinct dim NAME sharing H1's 7 slots
    # (67..73). The per-step byte-1 register DUMP carry head (logical L11,
    # ``make_layer11_ax_byte1_dump_carry_op``) WRITES the re-supplied H1
    # one-hot here on carried (non-AX-writing) steps. Mirrors the
    # OUTPUT_HI -> OUTPUT_HI_THIS_STEP split: same numeric base so the LM
    # head (which reads physical slots 67..73 for byte-1 emission) sees a
    # byte-identical residual, while the DISTINCT name keeps the carry
    # head out of the dep-graph edge set of the 54 same-step ``H1``
    # readers -- the write -> read back-edge that forms the unbreakable
    # 2-cycle when the carry head writes base ``H1``. See
    # docs/AX_BYTE1_DUMP_CARRY_H1_WRITE_CYCLE_2026_06_13.md.
    H1_DUMP = 67  # alias of H1
    H2 = 74
    H3 = 81
    H4 = 88
    H5 = 95
    H6 = 102
    H7 = 109

    # --- B7 structural lifecycle dims (aliased onto H5/H6/H7 "dead" range) ---
    # H5/H6/H7 (slots 95-115) are written by ``layer0_threshold_attn`` but
    # have no downstream consumer (see ``investigation/bd-dim-usage-map``
    # REPORT Section 3). B7 reclaims slots 95-98:
    # - B7-2: SP_BYTE0_IS_F8 (slot 95) — L7 head-6 producer, 1.0 only when
    #   carry-forward proves SP byte 0 is 0xF8.
    # - B7-1: IN_STEP_FRESH (slot 96) — L1 head-5 producer with ALiBi slope
    #   0.5, decays from 1.0 immediately after STEP_BOUNDARY toward 0.0 as
    #   tokens accumulate; resets at next STEP_BOUNDARY. Consumed by L10
    #   tail_* rules as positive in-step evidence (replaces HAS_SE -1e9
    #   negative hammer).
    # - B7-4: ADDR_B0_VALID (slot 97) — Single dim written 1.0 by L13
    #   mem-addr-gather alongside the ADDR_B0_LO/HI one-hots at MEM val byte
    #   positions. L10 tail addr0 family consumes this to disambiguate
    #   "ADDR_B0 lanes carry freshly-computed nibbles" from "ADDR_B0 lanes
    #   are stale residue from a previous step or unrelated row". Aliases
    #   H5+2; L0 still writes H5 there but never reads it, and L13 only
    #   writes ADDR_B0_VALID at the MEM val byte positions where the
    #   ADDR_B0 gather completed so the two writers do not collide at
    #   consumer rows.
    # - B7-5: SP_GATHERED_THIS_STEP (slot 98) — single-dim sentinel set to
    #   1.0 at MARK_SP query positions by L8 FFN, signalling that the L8
    #   SP gather has fired in the current step.
    SP_BYTE0_IS_F8 = 95         # aliases H5+0 (dead L0 head 5 PC slot)
    IN_STEP_FRESH = 96          # aliases H5+1 (dead L0 head 5 AX slot)
    ADDR_B0_VALID = 97          # aliases H5+2 (dead L0 head 5 SP slot)
    SP_GATHERED_THIS_STEP = 98  # aliases H5+3 (dead L0 head 5 BP slot)

    # --- L1 fine thresholds + HAS_SE ---
    L1H0 = 116
    L1H1 = 123
    L1H2 = 130
    HAS_SE = 137

    # --- Byte index within register (0-3) ---
    BYTE_INDEX_0 = 138
    BYTE_INDEX_1 = 139
    BYTE_INDEX_2 = 140
    BYTE_INDEX_3 = 141

    # --- Nibble encoding ---
    EMBED_LO = 142  # 142-157: embedding input nibbles (one-hot 16)
    EMBED_HI = 158  # 158-173
    OUTPUT_LO = 174  # 174-189: output decoding nibbles
    OUTPUT_HI = 190  # 190-205
    # B9 OUTPUT_HI split: declarative alias for the same 16-slot band. The
    # rename clarifies "this step's OUTPUT_HI write" vs the would-be
    # "OUTPUT_HI_PREV_STEP" cross-step carry (see docs/B9_OUTPUT_HI_SPLIT_SPEC.md).
    # Numeric position is identical so baked weights are byte-identical.
    OUTPUT_HI_THIS_STEP = 190  # alias of OUTPUT_HI
    # Phase 9.C: OUTPUT_{LO,HI}_PREV_STEP, EMBED_{LO,HI}_PREV_STEP, and
    # ADDR_KEY_PREV_STEP aliases were deleted — corpus readers migrated
    # to SSA ``BASE.<writer>.-1`` spellings in Phase 9.B.

    # --- Address key (for memory attention) ---
    ADDR_KEY = 206  # 206-253 (48 dims: 3 nibbles × 16 one-hot)

    # --- NEXT_* transition flags ---
    NEXT_PC = 254
    NEXT_AX = 255
    NEXT_SP = 256
    NEXT_BP = 257
    NEXT_STACK0 = 258
    NEXT_MEM = 259
    NEXT_SE = 260
    NEXT_HALT = 261

    # --- Opcode one-hot flags (34 opcodes) ---
    OP_LEA = 262
    OP_IMM = 263
    OP_JMP = 264
    OP_JSR = 265
    OP_BZ = 266
    OP_BNZ = 267
    OP_ENT = 268
    OP_ADJ = 269
    OP_LEV = 270
    # Phase 9.C: OP_LEV_PREV_STEP alias deleted — corpus readers migrated
    # to SSA ``OP_LEV.<writer>.-1`` spellings in Phase 9.B.
    OP_LI = 271
    OP_LC = 272
    OP_SI = 273
    OP_SC = 274
    OP_PSH = 275
    OP_OR = 276
    OP_XOR = 277
    OP_AND = 278
    OP_EQ = 279
    OP_NE = 280
    OP_LT = 281
    OP_GT = 282
    OP_LE = 283
    OP_GE = 284
    OP_SHL = 285
    OP_SHR = 286
    OP_ADD = 287
    OP_SUB = 288
    OP_MUL = 289
    OP_DIV = 290
    OP_MOD = 291
    OP_EXIT = 292
    OP_NOP = 293
    OP_PUTCHAR = 294
    OP_GETCHAR = 295
    OPCODE_BASE = 262
    NUM_OPCODES = 34

    # --- IO flag (PUTCHAR detection, 1 dim) ---
    IO_IS_PUTCHAR = 296  # OP_PUTCHAR detected this step

    # --- L1 head 4 threshold output (7 dims: one per marker type) ---
    L1H4 = 297  # dims 297-303: threshold 6.5 from nearest IS_MARK

    # --- STACK0 byte 0 flag (computed in L1 FFN) ---
    STACK0_BYTE0 = 304  # 1.0 at STACK0 byte 0 positions
    STACK0_BYTE1 = 508  # 1.0 at STACK0 byte 1 positions
    STACK0_BYTE2 = 509  # 1.0 at STACK0 byte 2 positions
    STACK0_BYTE3 = 510  # 1.0 at STACK0 byte 3 positions

    # --- CMP group flag (any comparison opcode active) ---
    CMP_GROUP = 305  # 1.0 when EQ/NE/LT/GT/LE/GE active at AX marker

    # --- Pristine nibble encoding (never written by attention/FFN) ---
    # These dims stay at their embedding values throughout all layers.
    # Use for V reads in attention heads that need clean one-hot nibbles
    # (L5 fetch, L7 operand gather) instead of EMBED_LO/HI which gets
    # inflated by carry-forward attention residual leakage.
    CLEAN_EMBED_LO = 306  # 306-321 (16 dims)
    # CLEAN_EMBED_HI at 400, see below

    # --- Tool call detection and I/O state (gap 322-327) ---
    IO_IS_TOOL_CALL = 322  # Combined flag: any of OPEN/READ/CLOS/PRTF active
    NEXT_TOOL_CALL = 323  # Transition flag → emit TOOL_CALL token
    NEXT_THINKING_START = 324  # Transition flag → emit <thinking> token
    NEXT_THINKING_END = 325  # Transition flag → emit </thinking> token
    NEXT_IO_STATE_EMIT_BYTE = 326  # Transition flag → emit IO_STATE_EMIT_BYTE token
    NEXT_IO_STATE_EMIT_THINKING = 327  # Transition flag → emit IO_STATE_EMIT_THINKING token

    # --- Conversational I/O detection (PRTF/READ specific) ---
    # BUG FIX 2026-04-13: Shifted +4 to avoid CLEAN_EMBED_HI/FETCH_LO collision cascade
    IO_IS_PRTF = 464  # Flag: PRTF opcode detected (aliases MEM_VAL_B3)
    IO_IS_READ = 465  # Flag: READ opcode detected (aliases OP_LI_RELAY)
    IO_STATE = 466  # State machine (aliases OP_LC_RELAY)
    IO_OUTPUT_COUNT = 467  # Number of output bytes remaining (aliases PSH_AT_SP)
    IO_FORMAT_POS = 468  # Position in format string (aliases MEM_EXEC)

    # --- Conversational I/O state tracking ---
    IO_IN_OUTPUT_MODE = 469  # Flag: currently emitting output bytes
    IO_OUTPUT_COMPLETE = 470  # Flag: format string complete (null terminator)
    FORMAT_PTR_LO = 471  # Format string pointer lo nibble (16 dims: 471-486, aliases AX_FULL_LO)
    FORMAT_PTR_HI = 487  # Format string pointer hi nibble (16 dims: 487-502, aliases AX_FULL_HI)
    # Note: OUTPUT_BYTE reuses TEMP space (480-511) which is not needed during conversational I/O
    OUTPUT_BYTE_LO = 480  # Output byte lo nibble (16 dims: 480-495, overlaps TEMP)
    OUTPUT_BYTE_HI = 496  # Output byte hi nibble (16 dims: 496-511, overlaps TEMP+16)

    # --- Lookback detection (detect previous token type) ---
    LAST_WAS_THINKING_END = 501  # Flag: previous token was THINKING_END
    LAST_WAS_THINKING_START = 502  # Flag: previous token was THINKING_START
    LAST_WAS_BYTE = 503  # Flag: previous token was byte (0-255)

    # --- Active opcode (from MoE routing, set by embedding) ---
    ACTIVE_OPCODE_PRTF = 504  # 1.0 when current opcode is PRTF (0x21=33)
    ACTIVE_OPCODE_READ = 505  # 1.0 when current opcode is READ (0x1F=31)

    # --- Conversational I/O token markers (set by embedding, no overlap) ---
    MARK_THINKING_START = 506  # 1.0 for THINKING_START token (272)
    MARK_THINKING_END = 507  # 1.0 for THINKING_END token (273)

    # --- AX carry-forward staging ---
    AX_CARRY_LO = 328  # 328-343
    AX_CARRY_HI = 344  # 344-359
    # Phase 9.C: AX_CARRY_{LO,HI}_PREV_STEP aliases deleted — corpus
    # readers migrated to SSA ``AX_CARRY_{LO,HI}.<writer>.-1`` spellings
    # in Phase 9.B.

    # --- I/O state detection (shifted +4 to maintain aliases with MEM_VAL_B1/B2) ---
    LAST_WAS_IO_STATE_EMIT_BYTE = 462  # Flag: last token was IO_STATE_EMIT_BYTE (aliases MEM_VAL_B1)
    LAST_WAS_IO_STATE_EMIT_THINKING = 463  # Flag: last token was IO_STATE_EMIT_THINKING (aliases MEM_VAL_B2)

    # --- ALU result staging ---
    ALU_LO = 360  # 360-375
    ALU_HI = 376  # 376-391
    # Phase 9.C: ALU_LO_PREV_STEP / CARRY_PREV_STEP / CMP_PREV_STEP
    # aliases deleted — corpus readers migrated to SSA
    # ``BASE.<writer>.-1`` spellings in Phase 9.B.

    # --- Carry / comparison ---
    CARRY = 392  # 392-395 (4 dims: inter-byte carry for ADD/SUB/MUL)
    CMP = 396  # 396-403 (8 dims: PSH/ADJ/ENT/POP/JSR/AX_ZERO flags)

    # --- Pristine nibble encoding (hi nibble) ---
    # BUG FIX 2026-04-13: Moved from 400 to 404 to avoid collision with CMP[4..7]
    CLEAN_EMBED_HI = 404  # 404-419 (16 dims)

    # --- MUL/DIV staging (also used as FETCH staging in Phase 3) ---
    # BUG FIX 2026-04-13: Shifted +4 to avoid CLEAN_EMBED_HI[12..15] collision
    MUL_ACCUM = 420  # 420-435 (was 416-431)
    DIV_STAGING = 436  # 436-451 (was 432-447)
    FETCH_LO = 420  # alias: fetched immediate lo nibble
    FETCH_HI = 436  # alias: fetched immediate hi nibble

    # --- Address hi nibble gathering (reuse ADDR_KEY space at byte positions) ---
    ADDR_B0_HI = 206  # 206-221 (16 dims): hi nibble of gathered addr byte 0
    ADDR_B1_HI = 222  # 222-237
    ADDR_B2_HI = 238  # 238-253
    # Phase 9.C: ADDR_B0_HI_PREV_STEP retained — consumed by
    # ``make_lev_detector_head_op`` (control_flow_heads.py) as a string
    # literal in op.reads (and exercised by tests/test_lev_detector_head.py).
    # ADDR_B{1,2}_HI_PREV_STEP aliases deleted — corpus readers migrated
    # to SSA ``BASE.<writer>.-1`` spellings in Phase 9.B.
    ADDR_B0_HI_PREV_STEP = 206  # alias of ADDR_B0_HI (consumer: lev_detector_head)

    # --- L2 threshold head output (7 dims: one per marker type) ---
    # Shifted +4 (was 448-454, now 452-458)
    L2H0 = 452  # 452-458: threshold 5.5 from nearest IS_MARK

    # --- Memory operation flags ---
    # Shifted +4 (was 455-464, now 459-468)
    MEM_STORE = 459  # 1 dim: store op active (SI/SC/PSH), relayed to MEM positions
    MEM_ADDR_SRC = 460  # 1 dim: 1=addr from STACK0 (SI/SC), 0=addr from SP (PSH)
    MEM_VAL_B0 = 461  # 1 dim: predicts MEM val byte 0 (d=4 from MEM, addr byte 3)
    MEM_VAL_B1 = 462  # 1 dim: predicts MEM val byte 1 (d=5 from MEM, val byte 0)
    MEM_VAL_B2 = 463  # 1 dim: predicts MEM val byte 2 (d=6 from MEM, val byte 1)
    MEM_VAL_B3 = 464  # 1 dim: predicts MEM val byte 3 (d=7 from MEM, val byte 2)
    OP_LI_RELAY = 465  # 1 dim: LI active (relayed to AX byte positions)
    OP_LC_RELAY = 466  # 1 dim: LC active (relayed to AX byte positions)
    PSH_AT_SP = 467    # 1 dim: PSH opcode flag relayed to SP/STACK0 (clean, no JMP collision)

    # --- Unified memory execution (DEPRECATED Phase A 2026-05-11) ---
    # MEM_EXEC writes were removed (no model consumer); retained as a layout
    # placeholder so the compact-IO compiler block stays stable. The slot is
    # aliased by IO_FORMAT_POS@468 above for conversational I/O.
    MEM_EXEC = 468     # 1 dim: deprecated; retained as IO_FORMAT_POS alias

    # --- AX full value relay (for PSH STACK0 = AX) ---
    # Shifted +4 (was 467-498, now 471-502)
    # Overlaps FORMAT_PTR which is only used during conversational I/O.
    AX_FULL_LO = 471  # 471-486 (16 dims, overlaps FORMAT_PTR_LO)
    AX_FULL_HI = 487  # 487-502 (16 dims, overlaps FORMAT_PTR_HI)

    # --- V18 Phase 1b: PRTF AX-marker capture cache dims ---
    # The capture-side bake (``_set_convo_io_prtf_capture``) writes PC and SP
    # byte-0 nibbles into these dims at the PRTF AX marker, where they are
    # later read by the L6 replay band (``_set_convo_io_pc_sp_latch``) at the
    # resumed step's REG_PC / REG_SP value-byte positions. The dims are
    # aliased onto existing slots that are *dead* at the PRTF AX marker:
    #   - POST_PRTF_PC_LO/HI alias AX_FULL_LO/HI (471/487). At the PRTF AX
    #     marker AX_FULL is being staged for ``PSH STACK0 = AX`` of the same
    #     step — PRTF never PSHes AX, so overwriting AX_FULL there is safe.
    #   - POST_PRTF_SP_LO/HI alias AX_CARRY_LO/HI (328/344). AX_CARRY is the
    #     ALU divisor staging slot; PRTF is not an ALU op so AX_CARRY at the
    #     PRTF AX marker is unused.
    # The aliasing keeps d_model=512 (no growth) while giving the capture
    # bake disjoint targets for PC and SP nibbles. See V18_CONVO_IO_NEURAL_PLAN.md §3b.
    POST_PRTF_PC_LO = 471  # aliases AX_FULL_LO (16 dims, 471-486)
    POST_PRTF_PC_HI = 487  # aliases AX_FULL_HI (16 dims, 487-502)
    POST_PRTF_SP_LO = 328  # aliases AX_CARRY_LO (16 dims, 328-343)
    POST_PRTF_SP_HI = 344  # aliases AX_CARRY_HI (16 dims, 344-359)

    # --- General temporaries / reserved ---
    TEMP = 480  # 480-511 (32 dims)
    # Phase 9.C: TEMP_PREV_STEP retained — consumed by
    # ``make_lev_detector_head_op`` (control_flow_heads.py) as a string
    # literal in op.reads (and exercised by tests/test_lev_detector_head.py).
    TEMP_PREV_STEP = 480  # alias of TEMP (consumer: lev_detector_head)

    # --- C5: BZ branch-target re-fire fix (cross-step gate) ---
    # Written 1.0 by ``post_l9_bz_bnz_pc_override`` on BZ-taken steps
    # (MARK_PC + OP_BZ + CMP+4 + CMP+5 align). Consumed by the SAME op
    # on the NEXT step via the ``BZ_TARGET_FRESH.*.-1`` cross-step alias
    # as a negative gate term on the OUTPUT_LO cancel band, suppressing
    # the spurious cancellation that otherwise wipes the BZ target on
    # step >= 2 BZ instances (loop body, function body). See
    # docs/BZ_TARGET_FRESH_CROSS_STEP_2026_06_09.md.
    BZ_TARGET_FRESH = 830  # 1 dim @ 830 (just past STACK0_BYTE_VAL_3_HI[15])

    # --- STEP_END register-presence broadcast (2026-06-10, L1 head 6) ---
    # Written 1.0 at MARK_SE_ONLY rows by the new L1 within-step
    # broadcast head (``layer1_threshold_attn.step_end_reg_present``).
    # Each slot mirrors the matching ``MARK_<NAME>`` value from its
    # own row to the SE row, bounded by ALiBi slope to the current step.
    # L0/L1 foundation of the STEP_END compute migration; see
    # docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md.
    SE_REG_AX_PRESENT     = 831  # MARK_AX present in current step
    SE_REG_PC_PRESENT     = 832  # MARK_PC present in current step
    SE_REG_SP_PRESENT     = 833  # MARK_SP present in current step
    SE_REG_BP_PRESENT     = 834  # MARK_BP present in current step
    SE_REG_STACK0_PRESENT = 835  # MARK_STACK0 present in current step
    SE_REG_MEM_PRESENT    = 836  # MARK_MEM present in current step

    # --- Register-tagged STEP_END operand relay (2026-06-10, L9 head 3/4) ---
    # Written at MARK_SE_ONLY rows by the new L9 attention head
    # ``layer9_step_end_operand_relay`` (two heads, mirrors raw ALU/CARRY/
    # CMP/OP bands at MARK_AX into the SE_-tagged slots). Consumed by the
    # migrated L9 CMP rules (``_layer9_cmp_rules``, MARK_SE_ONLY gated)
    # so the rules can fire at the STEP_END row without colliding with
    # downstream readers of the raw bands. See
    # docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md (Wave A v2) and
    # memory note ``project_wave_b_cmp_needs_l9_internal_relay.md``.
    SE_ALU_LO        = 837   # ALU_LO mirror (16 wide)
    SE_ALU_HI        = 853   # ALU_HI mirror (16 wide)
    SE_AX_CARRY_LO   = 869   # AX_CARRY_LO mirror (16 wide)
    SE_AX_CARRY_HI   = 885   # AX_CARRY_HI mirror (16 wide)
    SE_CMP           = 901   # CMP cascade mirror (4 wide)
    SE_OP_EQ         = 905   # OP_EQ mirror (1 wide)
    SE_OP_NE         = 906   # OP_NE mirror (1 wide)
    SE_OP_LT         = 907   # OP_LT mirror (1 wide)
    SE_OP_GT         = 908   # OP_GT mirror (1 wide)
    SE_OP_LE         = 909   # OP_LE mirror (1 wide)
    SE_OP_GE         = 910   # OP_GE mirror (1 wide)
    SE_CMP_GROUP     = 911   # CMP_GROUP mirror (1 wide) -- L9 CMP gate

    # --- width=2 MUL high-byte result band (2026-06-13) ---
    # Dedicated 16-wide band (lo/hi nibble of the product's BYTE 1) for the
    # width=2 (8-bit x 8-bit -> 16-bit) MUL path. The low byte (byte 0) of
    # the product still lands in OUTPUT_LO / OUTPUT_HI exactly as the
    # width=1 path does; the HIGH byte (byte 1) is routed HERE instead of
    # the would-be ``OUTPUT_LO+32`` slot, which is ADDR_KEY (206) and would
    # corrupt the memory-address key band. MUL_RESULT_HI+nib holds byte-1's
    # low nibble (k in 0..15) and MUL_RESULT_HI+16... — no: this band is
    # the byte-1 lo nibble at +0..15 and the byte-1 hi nibble shares the
    # OUTPUT_HI-style packing; for a 16-bit product byte 1 = bits 8..15 =
    # nib2 (lo) + nib3 (hi). nib2 -> MUL_RESULT_HI_LO, nib3 -> MUL_RESULT_HI_HI.
    #
    # This band is allocated when width=2 MUL is active (the DEFAULT;
    # ``mul_width2_enabled()``, opt out ``C4_MUL_WIDTH2=0``). It is injected
    # into ``extra_residual_dims`` at the top of ``compile_full_vm_dynamic``
    # (NOT ``declare_setdim_compat_dims``) so the d_model widen is
    # head-dim-preserving (872 -> 981, n_heads 8 -> 9) and bnz-safe. With
    # width=2 OFF the dim is never declared, so d_model stays 920 and the
    # residual layout is byte-identical to pre-width2 main. The 912/928
    # values below are the legacy static ``_SetDim`` fallback positions; the
    # dynamic compiler bump-pointer allocates the live positions past the
    # SE_* high-water mark. See docs/MUL_WIDTH2_WIDEN_2026_06_13.md.
    MUL_RESULT_HI_LO = 912   # 912-927: byte-1 (product bits 8..11) lo nibble
    MUL_RESULT_HI_HI = 928   # 928-943: byte-1 (product bits 12..15) hi nibble

    # Convenience: map Opcode int → _SetDim opcode flag dim
    _OPCODE_DIM = None  # lazily built

    @classmethod
    def opcode_dim(cls, op_value):
        """Return the _SetDim dimension for a given Opcode int value."""
        if cls._OPCODE_DIM is None:
            cls._OPCODE_DIM = {
                Opcode.LEA: cls.OP_LEA,
                Opcode.IMM: cls.OP_IMM,
                Opcode.JMP: cls.OP_JMP,
                Opcode.JSR: cls.OP_JSR,
                Opcode.BZ: cls.OP_BZ,
                Opcode.BNZ: cls.OP_BNZ,
                Opcode.ENT: cls.OP_ENT,
                Opcode.ADJ: cls.OP_ADJ,
                Opcode.LEV: cls.OP_LEV,
                Opcode.LI: cls.OP_LI,
                Opcode.LC: cls.OP_LC,
                Opcode.SI: cls.OP_SI,
                Opcode.SC: cls.OP_SC,
                Opcode.PSH: cls.OP_PSH,
                Opcode.OR: cls.OP_OR,
                Opcode.XOR: cls.OP_XOR,
                Opcode.AND: cls.OP_AND,
                Opcode.EQ: cls.OP_EQ,
                Opcode.NE: cls.OP_NE,
                Opcode.LT: cls.OP_LT,
                Opcode.GT: cls.OP_GT,
                Opcode.LE: cls.OP_LE,
                Opcode.GE: cls.OP_GE,
                Opcode.SHL: cls.OP_SHL,
                Opcode.SHR: cls.OP_SHR,
                Opcode.ADD: cls.OP_ADD,
                Opcode.SUB: cls.OP_SUB,
                Opcode.MUL: cls.OP_MUL,
                Opcode.DIV: cls.OP_DIV,
                Opcode.MOD: cls.OP_MOD,
                Opcode.EXIT: cls.OP_EXIT,
                Opcode.NOP: cls.OP_NOP,
                Opcode.PUTCHAR: cls.OP_PUTCHAR,
                Opcode.GETCHAR: cls.OP_GETCHAR,
            }
        return cls._OPCODE_DIM.get(op_value)


def _expand_wrapper_blocks(model):
    """Phase 0 step: split per-block post_ops into successive passthrough blocks.

    Each post_op on a block becomes its own TransformerBlock with a passthrough
    attention (zero-init weights → residual identity via x + attn(x) = x + 0 = x)
    and the post_op as its ffn. Semantic execution order is preserved by
    inserting these blocks immediately after the original.

    2026-05-11 (HybridALUBlock removal): previously this also unwrapped
    ``HybridALUBlock`` (lookup_ffn + efficient_alu) into two successive blocks.
    HybridALUBlock has been deleted; ALU modules that previously wrapped
    block.ffn are now attached directly to block.post_ops by the compiler ops
    in migrated_ops.py (see ``_make_alu_postop_attach_op`` and
    ``make_efficient_l8_addsub_wrap_op``), so the post_ops expansion path below
    handles them uniformly with all other structural post-passes
    (BinaryOpByteZeroingPostOp, CarryPropagationPostOp, FlattenedDivMod, etc.).

    Future work: walk ALU* (ALUAndOrXor/ALUMul/ALUShift/ALUDivMod) internals
    (BDToGEConverter, GenericPureFFN per ALU stage, GEToBDConverter) into
    separate blocks too — requires more careful BD↔GE format-shape handling.
    """
    def _make_passthrough_block(template_attn, ffn_module, layer_idx, d_model):
        """Build a TransformerBlock with zero-init attention (residual identity)
        and the given ffn."""
        attn_passthrough = AutoregressiveAttention(
            d_model, num_heads=template_attn.num_heads,
            max_seq_len=template_attn.max_seq_len, layer_idx=layer_idx,
            use_flash_attention=getattr(template_attn, "use_flash_attention", True),
            positional_encoding=getattr(template_attn, "_positional_encoding", None),
            attention_normalization=getattr(
                template_attn, "attention_normalization", None
            ),
            rope_base=getattr(template_attn, "rope_base", None),
            # Propagate the over-width-band-invariant ALiBi-slope base from the
            # template so expansion passthrough blocks keep the same slope
            # geometry as the original blocks (a widen-added padding head must
            # not shift slopes here either).
            alibi_base_heads=getattr(template_attn, "alibi_base_heads", None),
        )
        return TransformerBlock(attn=attn_passthrough, ffn=ffn_module)

    def _rebake_as_pureffn(op_module):
        """If ``op_module`` is a PureFFN subclass with the same SwiGLU forward
        (i.e. a structural post-op like BinaryOpByteZeroingPostOp,
        CarryPropagationPostOp, BitwiseBytePropagationPostOp,
        ComparisonCombine), copy its baked weights into a fresh vanilla
        :class:`PureFFN` instance and return it. The class name on the
        returned module will be exactly ``PureFFN`` so the runtime-vanilla
        audit treats it as a canonical FFN.

        Non-PureFFN modules (e.g. ALU composites, FlattenedDivMod) are
        returned unchanged — they are not part of this migration.
        """
        # Only rebake if it's a PureFFN subclass that isn't already plain PureFFN.
        if not isinstance(op_module, PureFFN):
            return op_module
        if type(op_module) is PureFFN:
            return op_module
        # Subclass uses the canonical PureFFN.forward (no override permitted by
        # PureFFN's contract). Snapshot weights and re-home into a vanilla
        # PureFFN of matching shape. We pass hidden_dim=0 to skip the
        # parameter allocation in _bake_weights of a fresh PureFFN and then
        # swap the .data tensors directly so we don't waste an allocation.
        W_up = op_module.W_up.data
        b_up = op_module.b_up.data
        W_gate = op_module.W_gate.data
        b_gate = op_module.b_gate.data
        W_down = op_module.W_down.data
        b_down = op_module.b_down.data if (
            hasattr(op_module, 'b_down') and op_module.b_down is not None
        ) else None
        hidden_dim, dim = W_up.shape
        ffn = PureFFN(dim=dim, hidden_dim=hidden_dim)
        ffn.W_up.data = W_up.clone()
        ffn.b_up.data = b_up.clone()
        ffn.W_gate.data = W_gate.clone()
        ffn.b_gate.data = b_gate.clone()
        ffn.W_down.data = W_down.clone()
        if b_down is not None:
            ffn.b_down.data = b_down.clone()
        return ffn

    d_model = model.d_model
    post_op_expansions = 0
    pureffn_rebakes = 0

    # Split post_ops into their own blocks.
    #
    # Probe provenance (no behaviour change): tag every emitted physical block
    # with ``_logical_layer`` (its originating pre-expansion logical layer, in
    # 0..len(model.blocks)-1) and ``_is_post_op_expansion`` (False for the
    # original block, True for each passthrough block split off its post_ops).
    # ``tools/probe_groundtruth.py`` reads these to print the exact 37↔logical
    # mapping instead of guessing it from attention-weight norms (which alias
    # legitimately FFN-only original layers against passthrough blocks).
    final_blocks = []
    for logical_idx, block in enumerate(model.blocks):
        block._logical_layer = logical_idx
        block._is_post_op_expansion = False
        final_blocks.append(block)
        if hasattr(block, 'post_ops') and len(block.post_ops) > 0:
            post_ops_list = list(block.post_ops)
            block.post_ops = nn.ModuleList()  # remove from this block
            for op in post_ops_list:
                # Re-home structural PureFFN-subclass post-ops as vanilla PureFFN
                # so the runtime-vanilla audit accepts the resulting block as
                # canonical (just attention + PureFFN). The forward pass is
                # byte-identical since the subclasses inherit PureFFN.forward
                # unchanged and only override _bake_weights.
                ffn_module = _rebake_as_pureffn(op)
                if ffn_module is not op:
                    pureffn_rebakes += 1
                # Each post_op becomes its own block
                new_block = _make_passthrough_block(
                    block.attn, ffn_module, len(final_blocks), d_model
                )
                new_block._logical_layer = logical_idx
                new_block._is_post_op_expansion = True
                final_blocks.append(new_block)
                post_op_expansions += 1

    if post_op_expansions > 0:
        model.blocks = nn.ModuleList(final_blocks)
        device = next(model.parameters()).device
        model.blocks = model.blocks.to(device)
        print(f"  PHASE 0 EXPANSIONS: {post_op_expansions} post_ops "
              f"({pureffn_rebakes} re-baked into vanilla PureFFN)")
        print(f"  Total blocks: 17 -> {len(final_blocks)}")

    # Wrapper-coverage retrim: ``_right_size_ffns`` runs at phase 1200,
    # before this expansion at phase 1300, so composite post_ops
    # (AddSub5StageBlock, FlattenedPureFFN, etc.) whose inner FFNs hide
    # behind ``@property W_up`` accessors are treated as flat leaves by
    # the recursion gate and skip the trim. Once expanded into standalone
    # wrapper blocks each composite's inner FFN(s) become recursable
    # children of ``block.ffn`` and the trim reaches them. The retrim is
    # idempotent: already-right-sized FFNs early-return at
    # ``n_active == H``.
    _right_size_ffns(model)


def _merge_wrapper_blocks(model):
    """Phase 10.B: fold per-block ``post_ops`` into the parent block's FFN.

    Alternate to :func:`_expand_wrapper_blocks`. Selected when
    ``C4_DISABLE_WRAPPER_EXPANSION=1`` is set (see
    :func:`make_expand_wrapper_blocks_op`).

    The default path lifts each ``post_op`` into a fresh
    ``TransformerBlock`` whose attention is zero-initialised. The
    passthrough attention is functionally a no-op (``x + Attn(x) = x +
    0 = x``) but costs ``4 * d_model**2`` params per wrapper block --
    19.5%% of total at the production ``d_model=800``.

    This routing chains each block's post_ops behind its existing FFN
    via ``nn.Sequential(orig_ffn, *post_ops)`` and clears
    ``block.post_ops``. Forward math is byte-identical to the expanded
    path on the default ``use_rms_norm=False`` setting:

      * expanded:  ``attn(x); ffn(x);`` then per wrapper block
        ``passthrough_attn(x) [= x]; post_op(x)`` reduces to
        ``post_op_n(...post_op_1(ffn(attn(x))))``.
      * merged:   ``attn(x); Sequential(ffn, *post_ops)(x)`` reduces to
        ``post_op_n(...post_op_1(ffn(attn(x))))``.

    Saves ``n_post_ops * 4 * d_model**2`` attention params (~35.8 M at
    the 14-wrapper / 800-d_model baseline). The default
    ``_expand_wrapper_blocks`` path is unchanged.
    """
    merged_count = 0
    for block in model.blocks:
        if not hasattr(block, 'post_ops'):
            continue
        if len(block.post_ops) == 0:
            continue
        post_ops_list = list(block.post_ops)
        # Clear post_ops so TransformerBlock.forward's trailing loop is a no-op.
        block.post_ops = nn.ModuleList()
        # Chain (orig_ffn -> post_op_1 -> ... -> post_op_n).
        block.ffn = nn.Sequential(block.ffn, *post_ops_list)
        merged_count += len(post_ops_list)

    if merged_count > 0:
        device = next(model.parameters()).device
        model.blocks = model.blocks.to(device)
        print(f"  PHASE 10.B MERGE: folded {merged_count} post_ops into "
              f"parent FFNs (no wrapper blocks emitted)")
        print(f"  Total blocks: {len(model.blocks)} (native, no expansion)")


def _right_size_ffns(model):
    """Trim each block's FFN to the actually-programmed unit count.

    Identifies "active" hidden units as those with any non-zero weight in W_up, W_gate,
    or W_down, or with a non-zero b_up/b_gate. Replaces the FFN's parameters with new
    parameters sized to the active count, copying only those units.

    Reports per-layer the original vs reduced size.
    """
    import torch as _torch
    import torch.nn as _nn

    print("  RIGHT-SIZING FFNs (compiler-determined width):")
    total_before = 0
    total_after = 0

    def _resize_one(ffn_module, label):
        nonlocal total_before, total_after
        if not (hasattr(ffn_module, 'W_up')
                and isinstance(getattr(ffn_module, 'W_up', None), _nn.Parameter)):
            # Recurse into wrappers (FlattenedALUMul, ALUDivMod, AddSub5StageBlock,
            # etc.) that contain standard FFNs as children.
            for child_name, child in ffn_module.named_children():
                _resize_one(child, f"{label}.{child_name}")
            return
        W_up = ffn_module.W_up.data
        W_gate = ffn_module.W_gate.data
        W_down = ffn_module.W_down.data
        b_up = ffn_module.b_up.data
        b_gate = ffn_module.b_gate.data if (
            hasattr(ffn_module, 'b_gate') and ffn_module.b_gate is not None
        ) else None

        H, _ = W_up.shape
        active_mask = (
            (W_up.abs().sum(dim=1) > 0)
            | (W_gate.abs().sum(dim=1) > 0)
            | (W_down.abs().sum(dim=0) > 0)
            | (b_up.abs() > 0)
        )
        if b_gate is not None:
            active_mask = active_mask | (b_gate.abs() > 0)
        active_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
        n_active = active_idx.numel()

        if n_active == H:
            print(f"    {label}: {H} units (no dead units)")
            total_before += H
            total_after += H
            return
        if n_active == 0:
            n_active = 1
            active_idx = _torch.tensor([0], dtype=_torch.long, device=W_up.device)

        # nn.Module forbids reassigning a registered Parameter with a differently-shaped
        # one. Replace the underlying .data tensors directly (Parameter object identity
        # preserved, shape changes).
        ffn_module.W_up.data = W_up[active_idx, :].clone()
        ffn_module.W_gate.data = W_gate[active_idx, :].clone()
        ffn_module.W_down.data = W_down[:, active_idx].clone()
        ffn_module.b_up.data = b_up[active_idx].clone()
        if b_gate is not None:
            ffn_module.b_gate.data = b_gate[active_idx].clone()
        if hasattr(ffn_module, 'hidden_dim'):
            ffn_module.hidden_dim = n_active

        print(f"    {label}: {H} -> {n_active} units (-{H - n_active} dead)")
        total_before += H
        total_after += n_active

    for i, block in enumerate(model.blocks):
        _resize_one(block.ffn, f"L{i}.ffn")
        # Also recurse into post_ops (BinaryOpByteZeroingPostOp, CarryPropagationPostOp,
        # etc. may contain standard FFNs too)
        if hasattr(block, 'post_ops'):
            for j, post_op in enumerate(block.post_ops):
                _resize_one(post_op, f"L{i}.post_ops[{j}]")

    if total_before == 0:
        print(f"  TOTAL FFN units: 0")
    else:
        print(f"  TOTAL FFN units: {total_before} -> {total_after} ({100*total_after/total_before:.1f}% retained)")


def _set_threshold_attn(attn, thresholds, out_bases, slope, HD, heads=None, BD=None):
    """Set threshold-based attention heads for marker distance detection.

    Each head detects whether the nearest marker is within `threshold` tokens.
    Uses ALiBi: score = slope*(threshold - distance), giving a sharp sigmoid.

    Args:
        BD: Required dim spec (proxy or ``_SetDim``). Callers must pass a
            compiler proxy from a migrated op so pin_io_only=True layouts
            wire to the correct residual lanes. The previous ``BD=None``
            legacy fallback was removed per BD_SETDIM_HARDCODE_AUDIT M4 —
            all 4 live callers in ``unified_compiler/ops/{l0,l1,l2}_ops.py``
            already pass ``BD=_as_setdim_proxy(dim_positions)``.
    """
    if BD is None:
        raise TypeError(
            "_set_threshold_attn requires BD (use _as_setdim_proxy(dim_positions))"
        )
    if heads is None:
        heads = list(range(len(thresholds)))

    import math
    sqrt_hd = math.sqrt(HD)
    for i, (h, t) in enumerate(zip(heads, thresholds)):
        base = h * HD
        # q_val scales with sqrt(HD) so Q·K / sqrt(HD) = slope * threshold
        # regardless of HD. Previously hardcoded 8.0 (assuming HD=64), which
        # broke score budgets when pin_io_only=True makes d_model=728
        # (HD=91, sqrt(HD)=9.54).
        q_val = sqrt_hd * slope
        attn.W_q[base, BD.CONST] = q_val
        attn.W_k[base, BD.IS_MARK] = t
        for m, src in enumerate(BD.MARKS):
            attn.W_v[base + 1 + m, src] = 1.0
        for m in range(BD.NUM_MARKERS):
            attn.W_o[out_bases[i] + m, base + 1 + m] = 1.0


# =============================================================================
# Backward-compat re-exports for helpers moved to setup_helpers.py
# =============================================================================
# These helpers were extracted to ``setup_helpers.py`` because they're only
# called by migrated bake_fns in ``unified_compiler/migrated_ops.py`` (via
# ``from ..vm_step import _set_X``). Re-exporting here preserves that import
# path without modifying ``migrated_ops.py``.
from .setup_helpers import (
    _set_bz_bnz_relay,
    _set_conversational_io_opcode_decode,
    _set_conversational_io_output_routing,
    _set_conversational_io_relay_heads,
    _set_conversational_io_state_machine,
    _set_layer10_byte_passthrough,
    _set_layer10_bp_byte_passthrough,
    _set_layer10_carry_relay,
    _set_layer10_psh_stack0_passthrough,
    _set_layer10_sp_byte_passthrough,
    _set_layer10_stack0_byte_relay,
    _set_layer11_mul_partial,
    _set_layer12_mul_combine,
    _set_layer13_mem_addr_gather,
    _set_layer14_jsr_mem_default_suppress,
    _set_layer14_mem_addr_src_default_suppress,
    _set_layer1_ffn,
    _set_layer2_mem_byte_flags,
    _set_layer5_fetch,
    _set_layer9_lev_addr_relay,
    _set_layer9_lev_bp_to_pc_relay,
    _set_null_terminator_detection,
    _set_convo_io_step_resume,
    _set_convo_io_pc_sp_latch,
    _set_convo_io_prtf_capture,
    _set_convo_io_prtf_transport,
    _set_stack0_carry_attn,
    _set_tool_call_detection,
    _set_tool_call_opcode_decode,
    _set_tool_call_relay_head,
)


def _set_nibble_copy_ffn(ffn, S, BD):
    """Conditional nibble copy: OUTPUT = EMBED for all non-PC/AX/SP/BP byte values.

    PC bytes 0-3: L3 handles byte 0 (increment), L3 handles bytes 1-3 (output 0).
    AX bytes: L3/L6 handle (IMM/ALU results).
    SP/BP bytes: L3 handles defaults, L15 PSH handles changes.
    Nibble copy only applies to MEM and STACK0 areas.

    For first step outputs, this relies on:
    - AX: Set by IMM instruction (fetched in L5, available in EMBED)
    - SP/BP: Initialized to STACK_INIT, need byte 2 = 0x01
    """
    unit = 0

    # LO nibbles: copy when IS_BYTE AND NOT at register areas with custom handling
    # PC: ALL bytes have custom handling (L3 for byte 0, L3 defaults for bytes 1-3)
    # SP/BP: bytes 0-3 (L3 handles byte 2 default, L15 PSH handles changes)
    PC_I = 0  # PC marker index
    SP_I = 2  # SP marker index
    BP_I = 3  # BP marker index
    AX_I = 1  # AX marker index in MARKS array
    # Hard BP blockers: layer H1+BP_I and MARK_BP get an additional
    # ``-S * 1e6`` weight so the wide nibble-copy writer cannot dump
    # ~+758 into OUTPUT_LO+0 across BP byte rows during the local-frame
    # post-store cadence. Without this the L16 ``l16_bp_frame_byte1_ff``
    # override (50.0/S strength) is swamped and BP_byte1 lands at 0xf0
    # instead of 0xff, regressing if_var_* (IDs 425-449).
    BP_BLOCKER_W = -S * 1_000_000.0
    for k in range(16):
        # Up: fires at byte positions, suppressed at register areas with custom handling
        # Note: STACK0 uses separate MARK_STACK0 (not in MARKS array), so we use H4[BP]
        # which covers d <= 9.5 from BP marker (STACK0 is at d=5-9 from BP)
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.H1 + PC_I] = -S  # Suppress ALL PC bytes (L3 handles all)
        ffn.W_up[unit, BD.H1 + AX_I] = -S  # Suppress at AX (L3/L6 handle)
        ffn.W_up[unit, BD.H1 + SP_I] = -S  # Suppress at SP (L3/L15 PSH handle)
        ffn.W_up[unit, BD.H1 + BP_I] = -S + BP_BLOCKER_W  # BP byte rows: hard blocker
        ffn.W_up[unit, BD.H4 + BP_I] = -S  # Suppress at STACK0 area (d<=9.5 from BP)
        ffn.W_up[unit, BD.MEM_STORE] = -S  # Suppress at MEM during PSH/SI/SC
        ffn.W_up[unit, BD.MARK_BP] = BP_BLOCKER_W  # BP marker row: hard blocker
        ffn.b_up[unit] = -S * 0.5
        # Gate: copy this specific nibble value
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        # Output: write to corresponding OUTPUT_LO channel
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    # HI nibbles: same logic
    for k in range(16):
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.H1 + PC_I] = -S  # Suppress ALL PC bytes (L3 handles all)
        ffn.W_up[unit, BD.H1 + AX_I] = -S  # Suppress at AX (L3/L6 handle)
        ffn.W_up[unit, BD.H1 + SP_I] = -S  # Suppress at SP (L3/L15 PSH handle)
        ffn.W_up[unit, BD.H1 + BP_I] = -S + BP_BLOCKER_W  # BP byte rows: hard blocker
        ffn.W_up[unit, BD.H4 + BP_I] = -S  # Suppress at STACK0 area (d<=9.5 from BP)
        ffn.W_up[unit, BD.MEM_STORE] = -S  # Suppress at MEM during PSH/SI/SC
        ffn.W_up[unit, BD.MARK_BP] = BP_BLOCKER_W  # BP marker row: hard blocker
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # L15 PSH stack byte producers are now described once as CompilerIR and
    # lowered here so legacy and unified-compilation paths share the same rule
    # data.
    from .unified_compiler.ops.l15_ops import lower_l15_psh_stack_ir

    psh_dim_positions = {
        "PSH_AT_SP": BD.PSH_AT_SP,
        "H1": BD.H1,
        "H4": BD.H4,
        "IS_BYTE": BD.IS_BYTE,
        "BYTE_INDEX_0": BD.BYTE_INDEX_0,
        "BYTE_INDEX_1": BD.BYTE_INDEX_1,
        "BYTE_INDEX_2": BD.BYTE_INDEX_2,
        "OUTPUT_LO": BD.OUTPUT_LO,
        "OUTPUT_HI": BD.OUTPUT_HI,
    }
    unit = lower_l15_psh_stack_ir(
        ffn,
        psh_dim_positions,
        start_unit=unit,
        S=S,
    )

    # === LEA first-step AX byte 2 output ===
    # On first step, L10 attention passthrough produces 0 for AX bytes 1-3 (no previous step).
    # For LEA, AX = BP + imm. BP = 0x10000, so byte 2 = 0x01.
    # Fires at: CMP[7] (OP_LEA relay) + H1[AX] + IS_BYTE + BYTE_INDEX_1 + NOT HAS_SE
    # At AX byte 1 position (predicting byte 2 token).
    # FIX 2026-04-10: First-step LEA was outputting 0x00000000 instead of 0x00010000.
    # FIX 2026-04-13: Use CMP[7] (relayed from AX marker by L7 head 5) instead of OP_LEA.
    # CMP[7] ≈ 2.2 at AX byte positions (OP_LEA ≈ 11 * 0.2 relay scaling).
    # Threshold needs to be high enough that BYTE_INDEX_1 is required (not just CMP[7]+H1+IS_BYTE).
    # Sum = 5.17, threshold = 4.0 gives margin 1.17 -> silu(1.17) ≈ 0.79 -> output ≈ 3.2
    T_lea_byte = 4.5  # Lower threshold for stronger activation
    ffn.W_up[unit, BD.CMP + 7] = S  # OP_LEA relay (set by L7 head 5)
    ffn.W_up[unit, BD.H1 + AX_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * T_lea_byte
    ffn.b_gate[unit] = 1.0
    # Write OUTPUT_LO[1] = 1 and cancel any competing OUTPUT_LO[0]
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 4.0 / S  # lo nibble = 1 (byte 2 = 0x01), stronger
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -4.0 / S  # cancel competing lo=0 signal
    unit += 1
    ffn.W_up[unit, BD.CMP + 7] = S  # OP_LEA relay (set by L7 head 5)
    ffn.W_up[unit, BD.H1 + AX_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * T_lea_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0 (byte 2 = 0x01)
    unit += 1


# =============================================================================
# Instruction Fetch Layers (2-5)
# =============================================================================
#
# Note: the legacy ``_set_carry_forward_attn`` helper that previously lived
# here was deleted (BD_SETDIM_HARDCODE_AUDIT M1) once all live callers had
# migrated to ``unified_compiler.primitives.Primitives.carry_forward_attention``
# (which threads ``bd=`` through the proxy). The byte-equivalence test that
# pinned the two implementations together was retired in the same commit.
#
# (L2 FFN reserved for prompt/bootstrap memory wiring.)


def _set_opcode_decode_ffn(ffn, S, BD):
    """Decode opcode byte (in OPCODE_BYTE_LO/HI) → 34 one-hot flags at OPCODE_BASE.

    Each opcode has a unique (lo, hi) nibble pair. We use SwiGLU AND gates:
      up = S*(OPCODE_BYTE_LO[lo] + OPCODE_BYTE_HI[hi] - 1.5)  → active when both match
      gate = MARK_AX   → only at AX marker (where fetch results land)
      down → OP_xxx flag

    With CLEAN_EMBED-sourced values (exact 1.0 one-hot), correct decode
    gives silu(S*0.5) ≈ S/2. False positives (one nibble match) give
    silu(-S*0.5) ≈ 0. W_down scaled to 10/S → OP ≈ 5.0 for correct match.
    """
    unit = 0
    opcodes = [
        (Opcode.LEA, 0, 0),
        (Opcode.IMM, 1, 0),
        (Opcode.JMP, 2, 0),
        (Opcode.JSR, 3, 0),
        (Opcode.BZ, 4, 0),
        (Opcode.BNZ, 5, 0),
        (Opcode.ENT, 6, 0),
        (Opcode.ADJ, 7, 0),
        (Opcode.LEV, 8, 0),
        (Opcode.LI, 9, 0),
        (Opcode.LC, 10, 0),
        (Opcode.SI, 11, 0),
        (Opcode.SC, 12, 0),
        (Opcode.PSH, 13, 0),
        (Opcode.OR, 14, 0),
        (Opcode.XOR, 15, 0),
        (Opcode.AND, 0, 1),
        (Opcode.EQ, 1, 1),
        (Opcode.NE, 2, 1),
        (Opcode.LT, 3, 1),
        (Opcode.GT, 4, 1),
        (Opcode.LE, 5, 1),
        (Opcode.GE, 6, 1),
        (Opcode.SHL, 7, 1),
        (Opcode.SHR, 8, 1),
        (Opcode.ADD, 9, 1),
        (Opcode.SUB, 10, 1),
        (Opcode.MUL, 11, 1),
        (Opcode.DIV, 12, 1),
        (Opcode.MOD, 13, 1),
        (Opcode.EXIT, 6, 2),  # EXIT = 38 = 0x26
        (Opcode.NOP, 7, 2),  # NOP = 39 = 0x27
    ]
    # PUTCHAR = 65 = 0x41, GETCHAR = 64 = 0x40
    opcodes.append((Opcode.PUTCHAR, 1, 4))
    opcodes.append((Opcode.GETCHAR, 0, 4))

    for op_val, lo, hi in opcodes:
        op_dim = BD.opcode_dim(op_val)
        ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.b_up[unit] = -S * 1.5  # both must be ~1
        ffn.W_gate[unit, BD.MARK_AX] = 1.0  # only at AX marker
        ffn.W_down[op_dim, unit] = 10.0 / S  # scaled up: clean ALU → OP ≈ 5
        unit += 1

    # First-step opcode decode at PC marker (when NOT HAS_SE)
    # For JMP and JSR, since they affect PC prediction
    lo, hi = 2, 0  # JMP opcode = 2 = 0x02
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S  # fire at PC marker
    ffn.W_up[unit, BD.HAS_SE] = -S  # only when NOT HAS_SE (first step)
    ffn.b_up[unit] = -S * 2.5  # require all three conditions
    ffn.b_gate[unit] = 1.0  # always active when up > 0
    ffn.W_down[BD.OP_JMP, unit] = 10.0 / S  # write OP_JMP at PC marker
    unit += 1

    # JSR first-step decode at PC marker
    # Write to TEMP[0] (same as Layer 6 attention relay for subsequent steps)
    lo, hi = 3, 0  # JSR opcode = 3 = 0x03
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.TEMP + 0, unit] = 10.0 / S  # write IS_JSR flag to TEMP[0] at PC marker
    unit += 1

    # IMM first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 1, 0  # IMM opcode = 1 = 0x01
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_IMM, unit] = 10.0 / S  # write OP_IMM at PC marker
    unit += 1

    # LEA first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 0, 0  # LEA opcode = 0 = 0x00
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_LEA, unit] = 10.0 / S  # write OP_LEA at PC marker
    unit += 1

    # EXIT first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 6, 2  # EXIT opcode = 38 = 0x26
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_EXIT, unit] = 10.0 / S
    unit += 1

    # NOP first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 7, 2  # NOP opcode = 39 = 0x27
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_NOP, unit] = 10.0 / S
    unit += 1

    # ADD first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 9, 1  # ADD opcode = 25 = 0x19
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_ADD, unit] = 10.0 / S
    unit += 1

    # SUB first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 10, 1  # SUB opcode = 26 = 0x1A
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_SUB, unit] = 10.0 / S
    unit += 1

    # MUL first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 11, 1  # MUL opcode = 27 = 0x1B
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_MUL, unit] = 10.0 / S
    unit += 1

    # DIV first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 12, 1  # DIV opcode = 28 = 0x1C
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_DIV, unit] = 10.0 / S
    unit += 1

    # MOD first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 13, 1  # MOD opcode = 29 = 0x1D
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_MOD, unit] = 10.0 / S
    unit += 1

    # OR first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 14, 0  # OR opcode = 14 = 0x0E
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_OR, unit] = 10.0 / S
    unit += 1

    # XOR first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 15, 0  # XOR opcode = 15 = 0x0F
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_XOR, unit] = 10.0 / S
    unit += 1

    # AND first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 0, 1  # AND opcode = 16 = 0x10
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_AND, unit] = 10.0 / S
    unit += 1

    # EQ first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 1, 1  # EQ opcode = 17 = 0x11
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_EQ, unit] = 10.0 / S
    unit += 1

    # LT first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 3, 1  # LT opcode = 19 = 0x13
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_LT, unit] = 10.0 / S
    unit += 1

    # SHL first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 7, 1  # SHL opcode = 23 = 0x17
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_SHL, unit] = 10.0 / S
    unit += 1

    # SHR first-step decode at PC marker (for Layer 6 relay to AX marker)
    lo, hi = 8, 1  # SHR opcode = 24 = 0x18
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OP_SHR, unit] = 10.0 / S
    unit += 1

    # === TEMP clearing at PC marker ===
    # Clear TEMP dims at PC marker to prevent leakage from Layer 5 attention
    # mixing TEMP values from AX marker to PC marker. TEMP is only valid at
    # AX marker (where PC+1 is computed in L4 FFN), not at other markers.
    # EXCEPT: TEMP[0] is used for IS_JSR flag (first-step decode + L6 relay).
    # Condition: MARK_PC (fires at PC marker token)
    for k in range(32):
        if k == 0:
            unit += 1
            continue
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.TEMP + k] = -1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    # All-step opcode decode at PC marker for opcodes that need early knowledge.
    # These fire at MARK_PC on ALL steps (no HAS_SE gate), enabling BZ/BNZ/LEV/EXIT
    # routing at the PC marker (replaces the legacy active-opcode injection).
    for op_val, lo, hi in [
        (Opcode.BZ, 4, 0),
        (Opcode.BNZ, 5, 0),
        (Opcode.LEV, 8, 0),
        (Opcode.EXIT, 6, 2),
        (Opcode.JMP, 2, 0),
    ]:
        op_dim = BD.opcode_dim(op_val)
        ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[op_dim, unit] = 10.0 / S
        unit += 1


# =============================================================================
# MUL: Byte-0 schoolbook multiplication (L11 partial + L12 combine)
# =============================================================================


# =============================================================================
# L13 attention: MEM addr → val key gather
# =============================================================================


# =============================================================================
# SHL/SHR: Byte-0 shift operations (L13)
# =============================================================================


# =============================================================================
# Binary Pop SP Increment (SP += 8 for all binary pop ops)
# =============================================================================


# =============================================================================
# Function Call Opcodes (JSR, ENT, LEV, LEA)
# =============================================================================


def _set_function_call_weights(model, S, BD, HD):
    """Set weights for function-call opcodes: JSR, ENT, LEV, LEA.

    JSR (jump to subroutine):
      - PC = FETCH (jump target) — runner overrides PC directly (not via L6 JMP relay,
        which has one-step delay and would double-override at step N+1)
      - SP -= 8 (push return address onto stack)
      - STACK0 = return address (exec_pc, pushed by runner)
      - AX unchanged (passthrough)

    ENT (enter function frame):
      - SP unchanged (identity)
      - STACK0 = old BP (save frame pointer)
      - BP = old SP - 8 (new frame pointer)
      - AX unchanged (passthrough)

    LEV (leave function frame):
      - Handled by runner (restores SP, BP, PC from frame)
      - AX passthrough only in weights

    LEA (load effective address):
      - AX = FETCH + BP (address computation using ADD circuit)
      - L7 head 1 gathers BP → ALU, L6 FFN replaces AX_CARRY with FETCH
      - L8/L9 ADD gates include OP_LEA

    L5 heads 5-6: ENT relay attention (BP EMBED → TEMP, SP EMBED → TEMP).
    L6 head 7: JSR PC OUTPUT → AX_CARRY at STACK0.
    L6 FFN units 1700-2158: LEA/JSR/ENT output routing.
    """
    attn5 = model.blocks[5].attn
    attn6 = model.blocks[6].attn
    ffn6 = model.blocks[6].ffn

    T = 4.0  # standard opcode threshold: OP(~5) + MARK_AX(1) = 6 > 4

    # =====================================================================
    # L5 heads 5-6: ENT relay attention (moved from heads 2-3 to avoid conflict
    # with first-step opcode/immediate fetch in _set_layer5_fetch)
    # =====================================================================
    L5 = 20.0  # matching L5 fetch heads

    # Head 5: BP EMBED → TEMP at STACK0 marker (for ENT: STACK0 = old_BP)
    # Distance d=5 (STACK0 at pos 20, BP at pos 15 in same step)
    base = 5 * HD
    attn5.W_q[base, BD.MARK_STACK0] = L5
    attn5.W_k[base, BD.MARK_BP] = L5
    # Anti-leakage gate: only fire at STACK0 marker positions
    GATE = 33
    attn5.W_q[base + GATE, BD.MARK_STACK0] = 500.0
    attn5.W_q[base + GATE, BD.CONST] = -500.0
    attn5.W_k[base + GATE, BD.CONST] = 5.0
    # V: copy EMBED_LO/HI
    for k in range(16):
        attn5.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn5.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0
    # O: write to TEMP[0..15] and TEMP[16..31]
    for k in range(16):
        attn5.W_o[BD.TEMP + k, base + 1 + k] = 1.0
        attn5.W_o[BD.TEMP + 16 + k, base + 17 + k] = 1.0

    # Head 6: SP EMBED → TEMP at BP marker (for ENT: BP = old_SP - 8)
    # Distance d=5 (BP at pos 15, SP at pos 10)
    base = 6 * HD
    attn5.W_q[base, BD.MARK_BP] = L5
    attn5.W_k[base, BD.MARK_SP] = L5
    # Anti-leakage gate: only fire at BP marker positions
    GATE = 33
    attn5.W_q[base + GATE, BD.MARK_BP] = 500.0
    attn5.W_q[base + GATE, BD.CONST] = -500.0
    attn5.W_k[base + GATE, BD.CONST] = 5.0
    # OP_ENT gate: only fire when ENT opcode is active (prevents TEMP pollution)
    ENT_GATE = 34
    attn5.W_q[base + ENT_GATE, BD.OP_ENT] = 500.0
    attn5.W_q[base + ENT_GATE, BD.CONST] = -500.0
    attn5.W_k[base + ENT_GATE, BD.CONST] = 5.0
    # V: copy EMBED_LO/HI
    for k in range(16):
        attn5.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn5.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0
    # O: write to TEMP[0..15] and TEMP[16..31]
    for k in range(16):
        attn5.W_o[BD.TEMP + k, base + 1 + k] = 1.0
        attn5.W_o[BD.TEMP + 16 + k, base + 17 + k] = 1.0

    # =====================================================================
    # L6 head 7: PC OUTPUT → AX_CARRY at STACK0 (JSR: STACK0 = return addr)
    # =====================================================================
    # Write to AX_CARRY_LO/HI (not TEMP) to avoid collision with L5 head 2.
    # Distance from STACK0 (pos 20) to PC (pos 0) = 20 tokens.
    # Score = 50^2*0.125 - 5*20 = 312.5 - 100 = 212.5 (strong)
    # FIX 2026-04-16: Strong anti-leakage gate to prevent firing at PC marker.
    # Without CONST penalty, Q = 0 at PC marker (MARK_STACK0=0, MARK_AX=0),
    # giving ~8% attention weight and polluting AX_CARRY_LO with OUTPUT values.
    # FIX 2026-04-16: Use K[OP_JSR] negative to make AX K negative during JSR.
    # PSH also uses head 7 K[MARK_AX] = 50 (line 4686), causing K to fire at both PC and AX.
    # Problem: PSH needs AX to win, JSR needs PC to win, but same head 7.
    # Solution: OP_JSR is relayed to AX marker (~5.0), not PC marker (0.0).
    #   Add K[OP_JSR] = -20, so:
    #   - During JSR: K at AX = 50 - 20*5 = -50, K at PC = 30 (positive)
    #     Score at PC = 50*30/8 - 100 = 187.5 - 100 = 87.5 (positive, strong attention)
    #     Score at AX = 50*(-50)/8 - 75 = -312.5 - 75 = -387.5 (very negative)
    #     PC wins decisively
    #   - During PSH (OP_JSR=0): K at AX = 50, K at PC = 30
    #     Score at PC = 50*30/8 - 100 = 87.5, Score at AX = 50*50/8 - 75 = 312.5 - 75 = 237.5
    #     AX wins by 150
    L6 = 50.0
    base = 7 * HD
    attn6.W_q[base, BD.MARK_STACK0] = L6 + L6 * 20  # +1050 at STACK0
    attn6.W_q[base, BD.MARK_AX] = -L6  # block at AX marker
    attn6.W_q[base, BD.CONST] = -L6 * 20  # -1000 baseline
    attn6.W_k[base, BD.MARK_PC] = 30.0  # K at PC for positive score (Q*K/8 - ALiBi > 0)
    attn6.W_k[base, BD.OP_JSR] = -20.0  # Negate AX's K during JSR (OP_JSR=5 at AX)
    # V: copy OUTPUT_LO/HI (PC's output = PC+5 from L3)
    for k in range(16):
        attn6.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
        attn6.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
    # O: write to AX_CARRY_LO/HI at STACK0 marker
    for k in range(16):
        attn6.W_o[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
        attn6.W_o[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0

    # =====================================================================
    # L6 FFN: Function call output routing (units 1700-2158)
    # =====================================================================
    unit = 1700

    # --- LEA first-step: Initialize ALU with BP default (2 units: 850-851) ---
    # For first step, set ALU = BP_default = 0x00010000, byte 0 = 0x00
    # Gate on: OP_LEA + MARK_AX + NOT HAS_SE. OP_LEA is now a strong decoded
    # flag (~5), so HAS_SE must dominate it on later steps.
    # Subsequent steps use Layer 7 attention relay from BP marker.
    ffn6.W_up[unit, BD.OP_LEA] = S
    ffn6.W_up[unit, BD.MARK_AX] = S
    ffn6.W_up[unit, BD.HAS_SE] = -S * 10
    ffn6.b_up[unit] = -S * 1.5
    ffn6.b_gate[unit] = 1.0
    ffn6.W_down[BD.ALU_LO + 0, unit] = 2.0 / S  # nibble 0
    unit += 1
    ffn6.W_up[unit, BD.OP_LEA] = S
    ffn6.W_up[unit, BD.MARK_AX] = S
    ffn6.W_up[unit, BD.HAS_SE] = -S * 10
    ffn6.b_up[unit] = -S * 1.5
    ffn6.b_gate[unit] = 1.0
    ffn6.W_down[BD.ALU_HI + 0, unit] = 2.0 / S  # nibble 0
    unit += 1
    # Units 852-881 unused (reserved)
    unit += 30

    # --- JSR SP -= 8 (autoregressive shift fix) ---
    # Moved to L7 where EMBED has byte values at marker positions.
    # Reserve 128 units (was: 4 bytes × 32 units each of hardcoded output).
    unit += 128

        # --- JSR STACK0 = return_addr (128 units: marker + 4 bytes) ---
    # BUG FIX 2026-04-10: L14 heads 4-7 read from STACK0 BYTE positions, not marker!
    # JSR must write return_addr to STACK0 bytes 0-3, not just marker.
    # return_addr is in AX_CARRY dims (PC+5 from L6 head 7).

    # STACK0 marker (32 units: for backwards compat, though not used by L14)
    # FIX 2026-04-16: Also cancel L3 default (OUTPUT_LO[0]=1) when JSR writes return_addr.
    # The gate -EMBED+AX_CARRY cancels L6 identity carry but not L3 default.
    # Add W_down[OUTPUT_LO+0] -= 2.0/S for k=10 unit to cancel L3 default when return_addr=0xA.
    # General fix: Add constant cancelation for OUTPUT_LO[0] via a separate unit.
    T_jsr_s0 = 1.5  # CMP[4](~1) + MARK_STACK0(1) = 2 > 1.5
    # FIX 2026-05-10 (Phase 4 BZ/BNZ step 2): Add MARK_PC/MARK_AX/IS_BYTE blockers.
    # Without blockers, the unit fires at PC marker on BZ/BNZ steps because L6 attn
    # head 4 leaks CMP[4]~2 (AX_LO_IS_ZERO relay) to PC marker. With CMP[4]=2 and
    # MARK_STACK0=0, up_pre = 0 + 200 - 150 = 50 > 0, so the unit fires and writes
    # AX_CARRY (the FETCH-fetched immediate) into PC OUTPUT, corrupting PC.
    # Strong negative blockers ensure these JSR STACK0 units only fire at MARK_STACK0.
    JSR_S0_BLOCK = -S * 10
    # First: Cancel L3 default OUTPUT_LO[0] with constant gate
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.MARK_STACK0] = S
    ffn6.W_up[unit, BD.MARK_PC] = JSR_S0_BLOCK  # FIX 2026-05-10
    ffn6.W_up[unit, BD.MARK_AX] = JSR_S0_BLOCK  # FIX 2026-05-10
    ffn6.W_up[unit, BD.IS_BYTE] = JSR_S0_BLOCK  # FIX 2026-05-10
    ffn6.b_up[unit] = -S * T_jsr_s0
    ffn6.W_gate[unit, BD.CONST] = 1.0  # Constant gate
    ffn6.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S  # Cancel L3 default
    unit += 1
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.MARK_STACK0] = S
    ffn6.W_up[unit, BD.MARK_PC] = JSR_S0_BLOCK  # FIX 2026-05-10
    ffn6.W_up[unit, BD.MARK_AX] = JSR_S0_BLOCK  # FIX 2026-05-10
    ffn6.W_up[unit, BD.IS_BYTE] = JSR_S0_BLOCK  # FIX 2026-05-10
    ffn6.b_up[unit] = -S * T_jsr_s0
    ffn6.W_gate[unit, BD.CONST] = 1.0
    ffn6.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S  # Cancel L3 default
    unit += 1
    # Then: Write return_addr from AX_CARRY
    for k in range(16):
        ffn6.W_up[unit, BD.CMP + 4] = S
        ffn6.W_up[unit, BD.MARK_STACK0] = S
        ffn6.W_up[unit, BD.MARK_PC] = JSR_S0_BLOCK  # FIX 2026-05-10
        ffn6.W_up[unit, BD.MARK_AX] = JSR_S0_BLOCK  # FIX 2026-05-10
        ffn6.W_up[unit, BD.IS_BYTE] = JSR_S0_BLOCK  # FIX 2026-05-10
        ffn6.b_up[unit] = -S * T_jsr_s0
        ffn6.W_gate[unit, BD.EMBED_LO + k] = -1.0
        ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.CMP + 4] = S
        ffn6.W_up[unit, BD.MARK_STACK0] = S
        ffn6.W_up[unit, BD.MARK_PC] = JSR_S0_BLOCK  # FIX 2026-05-10
        ffn6.W_up[unit, BD.MARK_AX] = JSR_S0_BLOCK  # FIX 2026-05-10
        ffn6.W_up[unit, BD.IS_BYTE] = JSR_S0_BLOCK  # FIX 2026-05-10
        ffn6.b_up[unit] = -S * T_jsr_s0
        ffn6.W_gate[unit, BD.EMBED_HI + k] = -1.0
        ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # STACK0 bytes 0-3 (128 units RESERVED but DISABLED).
    #
    # BUG FIX 2026-05-10 (C1.B): These per-byte units broadcast return_addr
    # byte 0 (from AX_CARRY) into OUTPUT at all four STACK0 byte positions
    # (J=0..3). Because of the autoregressive shift, OUTPUT at byte J predicts
    # the *next* token (byte J+1), so byte 0 leaks into bytes 1, 2, 3 instead
    # of producing 0x00 for return addresses < 256.
    #
    # Correct behavior is now achieved by:
    #   1. The STACK0 *marker* write above (lines ~9419-9434) which puts
    #      AX_CARRY → OUTPUT at the marker, predicting byte 0 = return_addr[0].
    #   2. `_set_layer14_clear_output_corruption` which boosts OUTPUT_LO[0]/
    #      OUTPUT_HI[0] at STACK0 byte positions 0..2, predicting byte value 0
    #      for bytes 1..3 (return_addr < 256 ⇒ high bytes are 0).
    #
    # If we ever need return addresses ≥ 256, replace this disable with a
    # *shifted* byte-matching attention head (mirror the pattern in
    # `_set_layer10_byte_passthrough` at vm_step.py:6560), where Q at byte J
    # routes from prev-step source byte J+1.
    #
    # Reserve the 128 unit slots so downstream unit numbering is unchanged.
    unit += 128

    # --- JSR PC override: PC = FETCH*INSTR_WIDTH + PC_OFFSET (jump target) ---
    # At PC marker when JSR: cancel OUTPUT (PC+INSTR_WIDTH), materialize the
    # instruction-index immediate as a PC byte, and write it to OUTPUT.
    # Gated on TEMP[0] (IS_JSR flag relayed from AX by L6 head 3).
    # Threshold: relayed OP_JSR ≈ 5.0, so T=4.0 separates it from false positives.
    # BUG FIX 2026-04-13: L6 head 4 (BZ/BNZ relay) unconditionally writes FETCH→TEMP
    # for all opcodes, polluting TEMP[0] with ~19.93 for NOP/EXIT. Add blockers for
    # non-JSR opcodes to prevent spurious JSR PC override triggering.
    T_jsr_pc = 4.0
    # Cancel OUTPUT_LO/HI (PC+5)
    for k in range(16):
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from L6 head 3 relay
        # BUG FIX: Block non-JSR opcodes that have TEMP[0] pollution from head 4
        ffn6.W_up[unit, BD.OP_NOP] = -S * 4  # Block NOP
        ffn6.W_up[unit, BD.OP_EXIT] = -S * 4  # Block EXIT
        ffn6.W_up[unit, BD.OP_JMP] = -S * 4  # Block JMP (has its own PC override)
        ffn6.W_up[unit, BD.OP_BZ] = -S * 4  # Block BZ (conditional branch)
        ffn6.W_up[unit, BD.OP_BNZ] = -S * 4  # Block BNZ (conditional branch)
        ffn6.W_up[unit, BD.OP_IMM] = -S * 4  # Block IMM (2026-04-15: TEMP pollution fix)
        ffn6.W_up[unit, BD.OP_LEV] = -S * 4  # Block LEV
        ffn6.W_up[unit, BD.OP_ENT] = -S * 4  # Block ENT
        ffn6.W_up[unit, BD.IS_BYTE] = -S * 10  # Only PC marker, never PC byte positions.
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from L6 head 3 relay
        # BUG FIX: Block non-JSR opcodes
        ffn6.W_up[unit, BD.OP_NOP] = -S * 4
        ffn6.W_up[unit, BD.OP_EXIT] = -S * 4
        ffn6.W_up[unit, BD.OP_JMP] = -S * 4
        ffn6.W_up[unit, BD.OP_BZ] = -S * 4
        ffn6.W_up[unit, BD.OP_BNZ] = -S * 4
        ffn6.W_up[unit, BD.OP_IMM] = -S * 4  # Block IMM (2026-04-15: TEMP pollution fix)
        ffn6.W_up[unit, BD.OP_LEV] = -S * 4  # Block LEV
        ffn6.W_up[unit, BD.OP_ENT] = -S * 4  # Block ENT
        ffn6.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write materialized target PC from FETCH_LO/HI. The pure-neural tests and
    # runner accept branch/call immediates as instruction indices; PC bytes are
    # idx * INSTR_WIDTH + PC_OFFSET. This low-byte path handles targets < 16
    # instructions, which covers the current JSR/LEV blocker fixtures.
    for k in range(16):
        target_lo = ((k * INSTR_WIDTH) + PC_OFFSET) & 0xF
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from first-step decode or L6 head 3 relay
        # BUG FIX: Block non-JSR opcodes
        ffn6.W_up[unit, BD.OP_NOP] = -S * 4
        ffn6.W_up[unit, BD.OP_EXIT] = -S * 4
        ffn6.W_up[unit, BD.OP_JMP] = -S * 4
        ffn6.W_up[unit, BD.OP_BZ] = -S * 4
        ffn6.W_up[unit, BD.OP_BNZ] = -S * 4
        ffn6.W_up[unit, BD.OP_IMM] = -S * 4  # Block IMM (2026-04-15: TEMP pollution fix)
        ffn6.W_up[unit, BD.OP_LEV] = -S * 4  # Block LEV
        ffn6.W_up[unit, BD.OP_ENT] = -S * 4  # Block ENT
        ffn6.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + target_lo, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        # Reserve the historical FETCH_HI unit range. The current JSR/LEV
        # fixtures target instruction indices < 16, so FETCH_HI is zero and
        # the high nibble is supplied solely by the FETCH_LO carry below.
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from first-step decode or L6 head 3 relay
        # BUG FIX: Block non-JSR opcodes
        ffn6.W_up[unit, BD.OP_NOP] = -S * 4
        ffn6.W_up[unit, BD.OP_EXIT] = -S * 4
        ffn6.W_up[unit, BD.OP_JMP] = -S * 4
        ffn6.W_up[unit, BD.OP_BZ] = -S * 4
        ffn6.W_up[unit, BD.OP_BNZ] = -S * 4
        ffn6.W_up[unit, BD.OP_IMM] = -S * 4  # Block IMM (2026-04-15: TEMP pollution fix)
        ffn6.W_up[unit, BD.OP_LEV] = -S * 4  # Block LEV
        ffn6.W_up[unit, BD.OP_ENT] = -S * 4  # Block ENT
        ffn6.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.FETCH_HI + k] = 1.0
        unit += 1
    for k in range(16):
        target_hi_from_lo = ((k * INSTR_WIDTH) + PC_OFFSET) >> 4
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S
        ffn6.W_up[unit, BD.OP_NOP] = -S * 4
        ffn6.W_up[unit, BD.OP_EXIT] = -S * 4
        ffn6.W_up[unit, BD.OP_JMP] = -S * 4
        ffn6.W_up[unit, BD.OP_BZ] = -S * 4
        ffn6.W_up[unit, BD.OP_BNZ] = -S * 4
        ffn6.W_up[unit, BD.OP_IMM] = -S * 4
        ffn6.W_up[unit, BD.OP_LEV] = -S * 4
        ffn6.W_up[unit, BD.OP_ENT] = -S * 4
        ffn6.W_up[unit, BD.IS_BYTE] = -S * 10
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + target_hi_from_lo, unit] = 2.0 / S
        unit += 1

    # --- JSR AX passthrough (32 units: 1010-1041) ---
    for k in range(16):
        ffn6.W_up[unit, BD.OP_JSR] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.OP_JSR] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # --- ENT STACK0 = old_BP (32 units: 978-1009) ---
    # At STACK0 marker when ENT: cancel identity, write TEMP (old BP from L5 head 2).
    T_ent_s0 = 1.5  # CMP[2](~1) + MARK_STACK0(1) = 2 > 1.5
    for k in range(16):
        ffn6.W_up[unit, BD.CMP + 2] = S
        ffn6.W_up[unit, BD.MARK_STACK0] = S
        ffn6.b_up[unit] = -S * T_ent_s0
        ffn6.W_gate[unit, BD.EMBED_LO + k] = -1.0  # cancel identity
        ffn6.W_gate[unit, BD.TEMP + k] = 1.0  # write old BP
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.CMP + 2] = S
        ffn6.W_up[unit, BD.MARK_STACK0] = S
        ffn6.b_up[unit] = -S * T_ent_s0
        ffn6.W_gate[unit, BD.EMBED_HI + k] = -1.0
        ffn6.W_gate[unit, BD.TEMP + 16 + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # --- ENT BP = SP - 8 (32 units: 1010-1041) ---
    # At BP marker when ENT: cancel identity, write TEMP (old SP from L5 head 3) - 8.
    T_ent_bp = 1.5  # CMP[2](~1) + MARK_BP(1) = 2 > 1.5
    for k in range(16):
        new_k = (k - 8) % 16
        ffn6.W_up[unit, BD.CMP + 2] = S
        ffn6.W_up[unit, BD.MARK_BP] = S
        ffn6.b_up[unit] = -S * T_ent_bp
        ffn6.W_gate[unit, BD.TEMP + k] = 1.0  # old SP lo nibble
        ffn6.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn6.W_down[BD.OUTPUT_LO + k, unit] += -2.0 / S  # cancel identity
        unit += 1
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        ffn6.W_up[unit, BD.CMP + 2] = S
        ffn6.W_up[unit, BD.MARK_BP] = S
        ffn6.b_up[unit] = -S * T_ent_bp
        ffn6.W_gate[unit, BD.TEMP + 16 + k] = 1.0  # old SP hi nibble
        # Detect borrow: old SP lo < 8 → TEMP[8..15] not hot
        for lo_bit in range(8, 16):
            ffn6.W_up[unit, BD.TEMP + lo_bit] = -S
        ffn6.W_down[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
        ffn6.W_down[BD.OUTPUT_HI + k, unit] += -2.0 / S
        unit += 1

    # --- ENT AX passthrough (32 units: 1042-1073) ---
    for k in range(16):
        ffn6.W_up[unit, BD.OP_ENT] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.OP_ENT] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # --- LEV AX passthrough (32 units: 1074-1105) ---
    # BUG FIX 2026-04-16: Add MARK_PC blocker. OP_LEV gets amplified to ~10 by L6
    # attention, causing units to fire at PC marker even without MARK_AX.
    # At PC marker: 10*S + 0 + 1*(-15*S) - 4*S = -9*S < 0 (blocked)
    # At AX marker: 10*S + 1*S + 0 - 4*S = 7*S > 0 (fires correctly)
    # BUG FIX 2026-04-16: Add IS_BYTE blocker. At AX byte 0 position, OP_LEV=7.5
    # causes units to fire even without MARK_AX: 7.5*S - 4*S = 3.5*S > 0 (spurious!)
    # With IS_BYTE blocker: 7.5*S + 0 + 0 - 10*S*1 - 4*S = -6.5*S < 0 (blocked)
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEV] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.W_up[unit, BD.MARK_PC] = -S * 15  # Block at PC marker
        ffn6.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEV] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.W_up[unit, BD.MARK_PC] = -S * 15  # Block at PC marker
        ffn6.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # --- LEV AX byte positions (32 units) ---
    # The marker-position units above are blocked at byte positions (IS_BYTE=-10*S).
    # L10 head 1 (AX carry-forward) suppresses byte 0 in K for shifted matching,
    # so AX values don't propagate from marker to byte positions during LEV.
    # These units explicitly copy AX_CARRY to OUTPUT at AX byte positions.
    # Fire condition: OP_LEV + IS_BYTE + H1[AX] (3-way AND)
    # At AX byte: OP_LEV≈7.5, IS_BYTE=1, H1[AX]=1 → 9.5*S > 9*S ✓ fires
    # At AX marker: OP_LEV≈10, IS_BYTE=0, H1[AX]=0, MARK_AX=1 → 10*S - 15*S < 0 ✗ blocked
    # At non-AX byte: OP_LEV≈7.5, IS_BYTE=1, H1[AX]=0 → 8.5*S < 9*S ✗ blocked
    AX_IDX = 1  # H1[AX_IDX] identifies AX byte positions
    T_byte = 9  # Threshold for 3-way AND: must be in (8.5, 9.5)
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEV] = S
        ffn6.W_up[unit, BD.IS_BYTE] = S  # Require byte position
        ffn6.W_up[unit, BD.H1 + AX_IDX] = S  # Require AX area
        ffn6.W_up[unit, BD.MARK_AX] = -S * 15  # Block at marker
        ffn6.b_up[unit] = -S * T_byte
        ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEV] = S
        ffn6.W_up[unit, BD.IS_BYTE] = S  # Require byte position
        ffn6.W_up[unit, BD.H1 + AX_IDX] = S  # Require AX area
        ffn6.W_up[unit, BD.MARK_AX] = -S * 15  # Block at marker
        ffn6.b_up[unit] = -S * T_byte
        ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
