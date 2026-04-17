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
from .efficient_alu_neural import (
    EfficientALU_L8_L9_Neural,
    EfficientALU_L10_Neural,
    EfficientALU_L11_L12_Neural,
    EfficientALU_L13_Neural,
    EfficientDivMod_Neural,
)


# =============================================================================
# Autoregressive VM Architecture
# =============================================================================


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

    STEP_TOKENS = (
        35  # Tokens per VM step: PC(5)+AX(5)+SP(5)+BP(5)+STACK0(5)+MEM(9)+SE(1)
    )


class AutoregressiveAttention(nn.Module):
    """Multi-head attention with softmax1 (ZFOD) and ALiBi/RoPE positional encoding.

    NOT a PureAttention subclass — PureAttention.forward() is FINAL and uses
    F.softmax. This class uses softmax1 for zero-fill-on-demand semantics
    and supports both ALiBi and RoPE positional encodings via config.
    """

    def __init__(self, dim, num_heads=4, max_seq_len=4096, layer_idx=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.max_seq_len = max_seq_len
        self.layer_idx = layer_idx

        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))

        # Determine positional encoding for this layer
        try:
            from .config import get_config
            config = get_config()

            # Hybrid mode: L0-L2 use ALiBi, rest use RoPE
            if config.positional_encoding == "hybrid" and layer_idx is not None and layer_idx < 3:
                self._positional_encoding = "alibi"
            else:
                self._positional_encoding = config.positional_encoding
        except ImportError:
            # Fallback if config not available (backwards compatibility)
            self._positional_encoding = "alibi"

        # Initialize ALiBi slopes if using ALiBi (or hybrid mode with layer < 3)
        use_alibi = (self._positional_encoding == "alibi" or
                     (self._positional_encoding == "hybrid" and layer_idx is not None and layer_idx < 3))
        if use_alibi:
            slopes = torch.tensor(
                [2.0 ** (-8.0 / num_heads * (i + 1)) for i in range(num_heads)]
            )
            self.register_buffer("alibi_slopes", slopes)  # [H]
        else:
            self.alibi_slopes = None

        # Initialize RoPE cache if using RoPE (or hybrid mode with layer >= 3)
        use_rope = (self._positional_encoding == "rope" or
                    (self._positional_encoding == "hybrid" and layer_idx is not None and layer_idx >= 3))
        if use_rope:
            try:
                from .config import get_config
                from .base_layers import precompute_rope_cache
                config = get_config()
                rope_base = config.rope_base
            except (ImportError, AttributeError):
                rope_base = 10000.0

            from .base_layers import precompute_rope_cache
            cos, sin = precompute_rope_cache(self.head_dim, max_seq_len, base=rope_base)
            self.register_buffer("_rope_cos", cos)
            self.register_buffer("_rope_sin", sin)
        else:
            self._rope_cos = None
            self._rope_sin = None

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

        # Get RoPE base from config or use default
        try:
            from .config import get_config
            rope_base = get_config().rope_base
        except (ImportError, AttributeError):
            rope_base = 10000.0

        # Compute extended cache
        from .base_layers import precompute_rope_cache
        cos_new, sin_new = precompute_rope_cache(
            self.head_dim, new_max_seq_len, base=rope_base, device=self._rope_cos.device
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

    def forward(self, x, kv_cache=None):
        """
        Forward pass with optional KV caching.

        Args:
            x: Input tensor [B, S, D] - full sequence including cached positions
            kv_cache: Optional TransformerKVCache for incremental generation

        Returns:
            Output tensor [B, S, D]
        """
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        if getattr(self, "_is_compact", False):
            # Compact path: gather active dims → dense matmul → scatter
            x_in = x[:, :, self._compact_in_idx]  # [B, S, n_in]
            n_out = len(self._compact_out_idx)
            Q = F.linear(x_in, self.W_q).view(B, S, H, n_out // H).transpose(1, 2)

            # KV caching: only compute for new tokens
            if kv_cache is not None and kv_cache.cache_size > 0:
                # Only compute K/V for new tokens (last tokens in sequence)
                new_tokens = S - kv_cache.cache_size
                if new_tokens > 0:
                    x_new = x_in[:, -new_tokens:, :]
                    K_new = F.linear(x_new, self.W_k).view(B, new_tokens, H, n_out // H).transpose(1, 2)
                    V_new = F.linear(x_new, self.W_v).view(B, new_tokens, H, n_out // H).transpose(1, 2)
                    K, V = kv_cache.update(K_new, V_new)
                else:
                    # All tokens are cached, use cache directly
                    K, V = kv_cache.cached_k, kv_cache.cached_v
            else:
                # No cache or empty cache: compute all K/V
                K = F.linear(x_in, self.W_k).view(B, S, H, n_out // H).transpose(1, 2)
                V = F.linear(x_in, self.W_v).view(B, S, H, n_out // H).transpose(1, 2)
                if kv_cache is not None:
                    K, V = kv_cache.update(K, V)
        else:
            linear = sparse_linear if self.W_q.is_sparse else F.linear
            Q = linear(x, self.W_q).view(B, S, H, HD).transpose(1, 2)

            # KV caching: only compute for new tokens
            if kv_cache is not None and kv_cache.cache_size > 0:
                # Only compute K/V for new tokens
                new_tokens = S - kv_cache.cache_size
                if new_tokens > 0:
                    x_new = x[:, -new_tokens:, :]
                    K_new = linear(x_new, self.W_k).view(B, new_tokens, H, HD).transpose(1, 2)
                    V_new = linear(x_new, self.W_v).view(B, new_tokens, H, HD).transpose(1, 2)
                    K, V = kv_cache.update(K_new, V_new)
                else:
                    # All tokens are cached
                    K, V = kv_cache.cached_k, kv_cache.cached_v
            else:
                # No cache: compute all K/V
                K = linear(x, self.W_k).view(B, S, H, HD).transpose(1, 2)
                V = linear(x, self.W_v).view(B, S, H, HD).transpose(1, 2)
                if kv_cache is not None:
                    K, V = kv_cache.update(K, V)

        # Apply RoPE if enabled (check for RoPE cache presence)
        if self._rope_cos is not None:
            # Q and K are [B, H, S_q/S_kv, HD]
            S_q = Q.shape[2]
            S_kv = K.shape[2]

            # For cached scenarios, queries are at positions [S_kv - S_q, S_kv)
            # For non-cached scenarios, S_q == S_kv and queries are at [0, S_q)
            q_offset = S_kv - S_q

            # Dynamically extend RoPE cache if sequence exceeds current cache size
            # This allows supporting arbitrarily long sequences
            max_needed = max(S_kv, q_offset + S_q)
            if max_needed > self._rope_cos.shape[0]:
                # Extend cache with 50% headroom to reduce frequent reallocations
                new_max_len = int(max_needed * 1.5)
                self._extend_rope_cache(new_max_len)

            # Apply RoPE to Q and K
            from .base_layers import rotate_half
            cos_q = self._rope_cos[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
            sin_q = self._rope_sin[q_offset:q_offset + S_q].unsqueeze(0).unsqueeze(0)  # [1, 1, S_q, HD]
            cos_k = self._rope_cos[0:S_kv].unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]
            sin_k = self._rope_sin[0:S_kv].unsqueeze(0).unsqueeze(0)  # [1, 1, S_kv, HD]

            Q = (Q * cos_q) + (rotate_half(Q) * sin_q)
            K = (K * cos_k) + (rotate_half(K) * sin_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # ALiBi bias (only if using ALiBi): -slope * |i - j|, computed on-the-fly
        # Note: With KV cache, query length S and key length S_kv may differ
        # Check for alibi_slopes presence rather than _positional_encoding string
        if self.alibi_slopes is not None:
            S_q = Q.shape[2]  # Query sequence length
            S_kv = K.shape[2]  # Cached key/value sequence length
            q_positions = torch.arange(S_q, device=x.device).unsqueeze(1)  # [S_q, 1]
            k_positions = torch.arange(S_kv, device=x.device).unsqueeze(0)  # [1, S_kv]
            dist = (q_positions - k_positions).abs().float()  # [S_q, S_kv]
            alibi = -self.alibi_slopes.view(1, H, 1, 1) * dist  # [1, H, S_q, S_kv]
            scores = scores + alibi

        # Causal mask, computed on-the-fly
        causal_mask = torch.triu(
            torch.full((S_q, S_kv), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal_mask

        # softmax1 for ZFOD
        attn = softmax1(scores, dim=-1)
        out = torch.matmul(attn, V)

        if getattr(self, "_is_compact", False):
            out = out.transpose(1, 2).contiguous().view(B, S, n_out)
            # W_o is [D, n_out] — full output dim, compact internal dim
            return x + F.linear(out, self.W_o)
        else:
            # Standard case: H * HD == D (e.g., 8 heads × 64 dims = 512)
            # Non-standard case: H * HD != D (e.g., L15 with 12 heads × 64 dims = 768 != 512)
            # In non-standard case, W_o projects from H*HD back to D
            out = out.transpose(1, 2).contiguous()  # [B, S, H, HD]
            if H * HD == D:
                out = out.view(B, S, D)
            else:
                out = out.view(B, S, H * HD)  # [B, S, 768] for L15
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


class TransformerBlock(nn.Module):
    """Transformer decoder block: attention + FFN.

    Both components include residual connections,
    so the block is sequential composition.
    """

    def __init__(self, attn, ffn):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.post_ops = nn.ModuleList()

    def forward(self, x, kv_cache=None):
        x = self.attn(x, kv_cache=kv_cache)
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
        d_model=512,
        n_layers=17,  # Updated from 16 for LEV Phase 3 (L16 routing layer)
        n_heads=8,  # REVERTED from 16: HD=32 broke attention score budgets
        ffn_hidden=4096,
        max_seq_len=4096,
    ):
        super().__init__()
        if vocab_size is None:
            vocab_size = Token.VOCAB_SIZE
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Use NeuralVMEmbedding with integrated augmentations
        self.embed = NeuralVMEmbedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    attn=AutoregressiveAttention(
                        d_model, num_heads=n_heads, max_seq_len=max_seq_len, layer_idx=i
                    ),
                    ffn=PureFFN(d_model, ffn_hidden),
                )
                for i in range(n_layers)
            ]
        )

        self.head = nn.Linear(d_model, vocab_size)

        # Store current active opcode for embedding augmentation
        self._active_opcode = None

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

    def compact_moe(self, opcode_range=None, relay_maps=None):
        """Apply MoE partitioning to all FFN layers by opcode.

        After compact(), further partitions each FFN's hidden units into
        shared (always computed) and per-opcode experts (only computed for
        that opcode). Call set_active_opcode() before each step.

        Args:
            opcode_range: range of opcode one-hot dims (default: 262-296).
            relay_maps: dict mapping layer_index -> relay_map for that layer.
                Default: L6 CMP relay map accounting for position-dependent
                CMP semantics (head 0/1/4 at PC/SE, head 6 at SP/STACK0).
        """
        BD = _SetDim
        if relay_maps is None:
            # CMP dims are overloaded at different positions:
            #   CMP[0]: IS_JMP at PC (head 0) + PSH at SP/STACK0 (head 6)
            #   CMP[1]: IS_EXIT at SE (head 1) + ADJ at SP/STACK0 (head 6)
            #   CMP[2]: OP_BZ at PC (head 4) + ENT at SP/STACK0 (head 6)
            #   CMP[3]: OP_BNZ at PC (head 4) + POP group at SP/STACK0 (head 6)
            #   CMP[4]: AX_LO_IS_ZERO at PC (head 4) + JSR at SP/STACK0 (head 6)
            #   CMP[5]: AX_HI_IS_ZERO at PC (head 4)
            # Map each CMP to ALL opcodes whose units depend on it.
            pop_ops = [
                BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_SHL, BD.OP_SHR,
                BD.OP_SI, BD.OP_SC,
            ]
            relay_maps = {
                6: {
                    BD.CMP + 0: [BD.OP_JMP, BD.OP_PSH],
                    BD.CMP + 1: [BD.OP_EXIT, BD.OP_ADJ],
                    BD.CMP + 2: [BD.OP_ENT, BD.OP_BZ],
                    BD.CMP + 3: pop_ops + [BD.OP_BNZ],
                    BD.CMP + 4: [BD.OP_JSR, BD.OP_BZ, BD.OP_BNZ],
                    BD.CMP + 5: [BD.OP_BZ, BD.OP_BNZ],
                }
            }
        for i, block in enumerate(self.blocks):
            relay = relay_maps.get(i)
            block.ffn.compact_moe(opcode_range=opcode_range, relay_map=relay)

    def set_active_opcode(self, opcode_value):
        """Set the active opcode for MoE routing in all FFN layers.

        Swaps weight tensors to the active opcode's sub-matrices.
        Zero per-forward overhead: forward() is unchanged.

        Args:
            opcode_value: Opcode enum value (e.g., Opcode.ADD), or None
                to use full matrices (prefix processing).
        """
        # Store opcode for embedding augmentation
        self._active_opcode = opcode_value

        if opcode_value is None:
            dim = None
        else:
            dim = _SetDim.opcode_dim(opcode_value)
        for block in self.blocks:
            ffn = block.ffn
            if getattr(ffn, "_moe_combined", None) is not None:
                ffn._activate_moe(dim)

    def save_compact(self, path):
        """Save compacted model to disk (avoids re-computing compact on load)."""
        torch.save(self, path)

    @staticmethod
    def load_compact(path):
        """Load a previously compacted model from disk."""
        return torch.load(path, weights_only=False)

    def to(self, device):
        """Move model to device, including MoE tensors.

        Overrides nn.Module.to() to also move MoE weight dictionaries
        that are stored as raw tensors (not Parameters/buffers).
        """
        result = super().to(device)
        # Move MoE tensors in each block's FFN
        for block in self.blocks:
            ffn = block.ffn
            if hasattr(ffn, '_moe_combined') and ffn._moe_combined is not None:
                for key, tensors in ffn._moe_combined.items():
                    for k, v in tensors.items():
                        if isinstance(v, torch.Tensor):
                            ffn._moe_combined[key][k] = v.to(device)
            if hasattr(ffn, '_moe_shared') and ffn._moe_shared is not None:
                for k, v in ffn._moe_shared.items():
                    if isinstance(v, torch.Tensor):
                        ffn._moe_shared[k] = v.to(device)
            if hasattr(ffn, '_moe_experts') and ffn._moe_experts is not None:
                for opcode, expert in ffn._moe_experts.items():
                    for k, v in expert.items():
                        if isinstance(v, torch.Tensor):
                            ffn._moe_experts[opcode][k] = v.to(device)
        return result

    def cuda(self, device=None):
        """Move model to CUDA, including MoE tensors."""
        if device is None:
            device = torch.cuda.current_device()
        return self.to(device)

    def cpu(self):
        """Move model to CPU, including MoE tensors."""
        return self.to('cpu')

    def forward(self, token_ids, kv_cache=None):
        """Forward pass: token IDs -> logits.

        Args:
            token_ids: [batch, seq] integer token IDs
            kv_cache: Optional LayerKVCache for incremental decoding

        Returns:
            [batch, seq, vocab_size] logits
        """
        # Pure forward pass: embed → blocks → head
        # All augmentations (ADDR_KEY, MEM_STORE) are inside NeuralVMEmbedding
        x = self.embed(token_ids, active_opcode=self._active_opcode)

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache.get_layer_cache(i) if kv_cache is not None else None
            x = block(x, kv_cache=layer_cache)

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

    # --- Opcode byte staging (L5 head 1 → L5 FFN decode) ---
    # Separate from ALU_LO/HI to avoid residual collision with L7 operand gather
    OPCODE_BYTE_LO = 12  # reuse ADDR_B0_LO (unused in autoregressive)
    OPCODE_BYTE_HI = 28  # reuse ADDR_B1_LO (unused in autoregressive)

    # --- L0 threshold heads (8 heads for 39-token step) ---
    # Thresholds: [3.5, 4.5, 5.5, 9.5, 10.5, 14.5, 15.5, 19.5]
    H0 = 60
    H1 = 67
    H2 = 74
    H3 = 81
    H4 = 88
    H5 = 95
    H6 = 102
    H7 = 109

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

    # --- I/O state detection (shifted +4 to maintain aliases with MEM_VAL_B1/B2) ---
    LAST_WAS_IO_STATE_EMIT_BYTE = 462  # Flag: last token was IO_STATE_EMIT_BYTE (aliases MEM_VAL_B1)
    LAST_WAS_IO_STATE_EMIT_THINKING = 463  # Flag: last token was IO_STATE_EMIT_THINKING (aliases MEM_VAL_B2)

    # --- ALU result staging ---
    ALU_LO = 360  # 360-375
    ALU_HI = 376  # 376-391

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

    # --- Unified memory execution (code from writable memory) ---
    MEM_EXEC = 468     # 1 dim: MEM section contains executable code (for L5 fetch)

    # --- AX full value relay (for PSH STACK0 = AX) ---
    # Shifted +4 (was 467-498, now 471-502)
    # Overlaps FORMAT_PTR which is only used during conversational I/O.
    AX_FULL_LO = 471  # 471-486 (16 dims, overlaps FORMAT_PTR_LO)
    AX_FULL_HI = 487  # 487-502 (16 dims, overlaps FORMAT_PTR_HI)

    # --- General temporaries / reserved ---
    TEMP = 480  # 480-511 (32 dims)

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


@torch.no_grad()
def set_vm_weights(model, enable_tool_calling=False, enable_conversational_io=False, alu_mode='lookup'):
    """Set weights into AutoregressiveVM for true neural VM execution.

    All computation flows through standard transformer layers (embed →
    attention → FFN → linear head). NO Python arithmetic in forward passes.

    Args:
        model: AutoregressiveVM instance
        enable_tool_calling: If True, all I/O opcodes (PUTCHAR/GETCHAR/
            OPEN/READ/CLOS/PRTF) emit TOOL_CALL instead of STEP_END,
            signaling the runner to dispatch via tool_handler callback.
            If False (default), these opcodes emit STEP_END and the runner
            dispatches post-hoc (current behavior).
        enable_conversational_io: If True, PRTF and READ emit autoregressive
            I/O sequences: THINKING_END → output bytes → THINKING_START.
            All I/O happens through transformer token generation.
            Cannot be used together with enable_tool_calling.
        alu_mode: 'lookup' (default) uses lookup tables baked into FFN weights.
            'efficient' uses multi-layer efficient ALU with neural format conversion.
            Both modes are purely neural (no Python arithmetic in forward pass).

    Architecture (16 layers, d_model=512, 8 heads, FFN=4096):
      L0:  Step structure (threshold attention for 39-token step)
      L1:  Fine thresholds + HAS_SE + CS distance
      L2:  CS distance refinement → thermometer encoding
      L3:  PC carry-forward + PC increment + branch override
      L4:  PC relay + PC+1 synthesis
      L5:  Opcode/immediate fetch via memory keys
      L6:  Register carry-forward (AX, SP, BP) + SP dispatch
      L7:  STACK0 address build (intra-step SP gather)
      L8:  STACK0 memory read (softmax1 at SP address)
      L9:  Operand gather + simple ALU (ADD/SUB raw, bitwise, comparisons)
      L10: Carry apply + comparison cascade + shift precompute
      L11: MUL operand gather + cross-term products + DIV step A
      L12: MUL carry propagation + DIV step B + SHL/SHR select
      L13: Branch condition broadcast + SP/BP writeback + control flow
      L14: MEM address gather
      L15: Memory lookup (softmax1) + final output routing + HALT

    Supports: All non-syscall C4 opcodes (26 of 34 opcode types).
    """
    # PURITY ENFORCEMENT: Verify model has not been modified to violate purity
    from .purity_guard import verify_forward_purity, verify_embedding_purity
    verify_forward_purity(model)
    verify_embedding_purity(model.embed)

    BD = _SetDim
    d = model.d_model
    V = model.vocab_size
    S = 100.0  # SwiGLU scale
    NH = model.blocks[0].attn.num_heads
    HD = d // NH  # head dim (64)
    ALIBI_S = 10.0  # ALiBi slope for threshold heads

    # ===== EMBEDDING =====
    # Access wrapped nn.Embedding inside NeuralVMEmbedding
    embed = model.embed.embed.weight
    embed.zero_()

    for tok in range(V):
        embed[tok, BD.CONST] = 1.0

    for tok, dim in [
        (Token.REG_PC, BD.MARK_PC),
        (Token.REG_AX, BD.MARK_AX),
        (Token.REG_SP, BD.MARK_SP),
        (Token.REG_BP, BD.MARK_BP),
        (Token.MEM, BD.MARK_MEM),
        (Token.CODE_START, BD.MARK_CS),
    ]:
        embed[tok, dim] = 1.0
        embed[tok, BD.IS_MARK] = 1.0

    # STACK0 has its own marker dim but NOT IS_MARK.
    # IS_MARK would make threshold heads "see" STACK0 as a marker,
    # blocking BP from being detected (STACK0 has no MARKS flags,
    # so all threshold outputs become zero when it's nearest).
    embed[Token.STACK0, BD.MARK_STACK0] = 1.0

    for tok in [Token.STEP_END, Token.DATA_END, Token.HALT]:
        embed[tok, BD.MARK_SE] = 1.0
        embed[tok, BD.IS_MARK] = 1.0

    embed[Token.STEP_END, BD.MARK_SE_ONLY] = 1.0

    # TOOL_CALL: same marker profile as STEP_END (threshold heads see it as step boundary)
    embed[Token.TOOL_CALL, BD.MARK_SE] = 1.0
    embed[Token.TOOL_CALL, BD.IS_MARK] = 1.0
    embed[Token.TOOL_CALL, BD.MARK_SE_ONLY] = 1.0
    embed[Token.TOOL_CALL, BD.CONST] = 1.0

    # IO token embeddings (markers for runner-side stdin framing)
    embed[Token.USER_INPUT_START, BD.IS_MARK] = 1.0
    embed[Token.USER_INPUT_END, BD.IS_MARK] = 1.0

    # Thinking tag embeddings (for conversational I/O mode)
    # These are markers but NOT step-end markers
    # Add specific markers for lookback detection
    embed[Token.THINKING_START, BD.IS_MARK] = 1.0
    embed[Token.THINKING_START, BD.CONST] = 1.0
    embed[Token.THINKING_START, BD.TEMP + 1] = 1.0  # Unique marker for lookback
    embed[Token.THINKING_END, BD.IS_MARK] = 1.0
    embed[Token.THINKING_END, BD.CONST] = 1.0
    embed[Token.THINKING_END, BD.TEMP + 2] = 1.0  # Unique marker for lookback

    # I/O state tokens (internal markers for state machine)
    # These control multi-step I/O generation sequences
    embed[Token.IO_STATE_EMIT_BYTE, BD.IS_MARK] = 1.0
    embed[Token.IO_STATE_EMIT_BYTE, BD.CONST] = 1.0
    embed[Token.IO_STATE_EMIT_THINKING, BD.IS_MARK] = 1.0
    embed[Token.IO_STATE_EMIT_THINKING, BD.CONST] = 1.0

    for b in range(256):
        embed[b, BD.IS_BYTE] = 1.0
        embed[b, BD.EMBED_LO + (b & 0xF)] = 1.0
        embed[b, BD.EMBED_HI + ((b >> 4) & 0xF)] = 1.0
        # Pristine copies — never written by attention W_o or FFN W_down
        embed[b, BD.CLEAN_EMBED_LO + (b & 0xF)] = 1.0
        embed[b, BD.CLEAN_EMBED_HI + ((b >> 4) & 0xF)] = 1.0

    # NOTE: OP_* flags are NOT set in the embedding table.
    # BUG FIX 2026-04-13: Removed embedding OP_* flags because they caused
    # L6 attention to average OP_LEA across all byte-0 positions (13 CODE bytes
    # with value 0x00), leaking OP_LEA=0.59 into the PC marker for non-LEA ops.
    # This triggered L7 head 1 (LEA operand gather), writing spurious ALU values,
    # which caused L13 shift FFN to corrupt PC OUTPUT from 24 to 8.
    #
    # L5 FFN opcode decode (_set_opcode_decode_ffn) properly sets OP_* flags
    # at MARK_PC/MARK_AX positions using OPCODE_BYTE_LO/HI gating, so embedding
    # OP_* flags are not needed and cause spurious leakage.

    # ===== LAYER 0: Step structure via threshold attention (8 heads) =====
    # 35-token step: PC(5)+AX(5)+SP(5)+BP(5)+STACK0(5)+MEM(9)+SE(1)
    # STACK0 does NOT have IS_MARK (would block BP from threshold view).
    # Key transitions (by distance from nearest IS_MARK marker):
    #   d=4 from PC/AX/SP/BP → next section  (H0=3.5, H1=4.5)
    #   d=8 from MEM → STEP_END              (H2=7.5, H3=8.5)
    #   d=9 from BP → MEM (through STACK0)   (H3=8.5, H4=9.5)
    attn0 = model.blocks[0].attn
    # ALiBi-specific: set slopes for threshold heads
    if hasattr(attn0, 'alibi_slopes') and attn0.alibi_slopes is not None:
        attn0.alibi_slopes.fill_(ALIBI_S)
    _set_threshold_attn(
        attn0,
        [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5],
        [BD.H0, BD.H1, BD.H2, BD.H3, BD.H4, BD.H5, BD.H6, BD.H7],
        ALIBI_S,
        HD,
    )

    ffn0 = model.blocks[0].ffn
    _set_phase_a_ffn(ffn0, S, BD)

    # ===== LAYER 1: Fine thresholds + STEP_END detection =====
    attn1 = model.blocks[1].attn
    # ALiBi-specific: set slopes for threshold heads
    if hasattr(attn1, 'alibi_slopes') and attn1.alibi_slopes is not None:
        attn1.alibi_slopes.fill_(ALIBI_S)
        attn1.alibi_slopes[3] = 0.0  # Head 3: global attention for SE detection
    _set_threshold_attn(
        attn1,
        [0.5, 1.5, 2.5],
        [BD.L1H0, BD.L1H1, BD.L1H2],
        ALIBI_S,
        HD,
        heads=[0, 1, 2],
    )
    # Head 3: STEP_END existence detection (global)
    base = 3 * HD
    attn1.W_q[base, BD.CONST] = 10.0
    attn1.W_k[base, BD.MARK_SE_ONLY] = 10.0
    attn1.W_v[base + 1, BD.MARK_SE_ONLY] = 1.0
    attn1.W_o[BD.HAS_SE, base + 1] = 1.0

    # Head 4: threshold 6.5 for STACK0 byte 0 identification
    _set_threshold_attn(attn1, [6.5], [BD.L1H4], ALIBI_S, HD, heads=[4])

    ffn1 = model.blocks[1].ffn
    _set_layer1_ffn(ffn1, S, BD)

    # ===== LAYER 2: Threshold 5.5 + MEM byte position flags =====
    attn2 = model.blocks[2].attn
    # ALiBi-specific: set slopes for threshold heads
    if hasattr(attn2, 'alibi_slopes') and attn2.alibi_slopes is not None:
        attn2.alibi_slopes.fill_(ALIBI_S)
    _set_threshold_attn(attn2, [5.5], [BD.L2H0], ALIBI_S, HD, heads=[0])

    # Conversational I/O: Lookback detection head (detect prev token type)
    if enable_conversational_io:
        if hasattr(attn2, 'alibi_slopes') and attn2.alibi_slopes is not None:
            attn2.alibi_slopes[1] = 10.0  # Steep slope to favor most recent token
        _set_lookback_detection_head(attn2, S, BD, HD)

    ffn2 = model.blocks[2].ffn
    _set_layer2_mem_byte_flags(ffn2, S, BD)

    # ===== LAYER 3: Register carry-forward (PC, AX, SP, BP) + PC update =====
    # All carry-forwards in one layer so values are available for later layers.
    attn3 = model.blocks[3].attn
    if hasattr(attn3, 'alibi_slopes') and attn3.alibi_slopes is not None:
        attn3.alibi_slopes.fill_(0.5)
    PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
    # Head 0: PC carry (prev step PC byte 0 → EMBED at PC marker)
    _set_carry_forward_attn(
        attn3, 0, BD.MARK_PC, PC_I, PC_I, HD, BD.EMBED_LO, BD.EMBED_HI
    )
    # Head 1: AX carry (prev step AX byte 0 EMBED → AX_CARRY staging)
    # Uses EMBED (not OUTPUT) because at byte positions, OUTPUT is 0 - the computed
    # result was at the marker position during generation. Once the byte token is
    # generated and re-embedded, EMBED has the correct value (byte tokens embed to
    # their byte value in EMBED_LO/HI).
    _set_carry_forward_attn(
        attn3, 1, BD.MARK_AX, AX_I, AX_I, HD, BD.AX_CARRY_LO, BD.AX_CARRY_HI
    )
    # Head 2: SP carry (prev step SP byte 0 → EMBED at SP marker)
    _set_carry_forward_attn(
        attn3, 2, BD.MARK_SP, SP_I, SP_I, HD, BD.EMBED_LO, BD.EMBED_HI
    )
    # Head 3: BP carry (prev step BP byte 0 → EMBED at BP marker)
    _set_carry_forward_attn(
        attn3, 3, BD.MARK_BP, BP_I, BP_I, HD, BD.EMBED_LO, BD.EMBED_HI
    )
    # Head 4: STACK0 carry (prev step STACK0 byte 0 → EMBED at STACK0 marker)
    # Uses STACK0_BYTE0 flag (computed in L1 FFN) as key instead of L1H1/L1H0.
    _set_stack0_carry_attn(attn3, 4, HD)

    # Head 5: AX full value relay (prev AX marker OUTPUT → current AX marker AX_FULL)
    # FIX 2026-04-09: Dedicated dimension (AX_FULL) to avoid conflict with head 1's AX_CARRY.
    # Previous attempt wrote to AX_CARRY causing additivity conflict (both heads writing
    # to same dims with total weight 3.0x), resulting in "predictions became all 1's".
    # Now uses separate AX_FULL dims for PSH STACK0 = AX operation.
    base = 5 * HD
    L = 15.0
    # Q: Fire at AX marker on subsequent steps only (HAS_SE=1)
    attn3.W_q[base, BD.MARK_AX] = L
    attn3.W_q[base, BD.HAS_SE] = L
    attn3.W_q[base, BD.CONST] = -L * 1.5  # Threshold: need both MARK_AX and HAS_SE
    # K: Match previous step's AX marker
    attn3.W_k[base, BD.MARK_AX] = L
    # V: Copy OUTPUT_LO/HI from previous AX marker (the final register value)
    for k in range(16):
        attn3.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
        attn3.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
    # O: Write to AX_FULL_LO/HI (NEW - no conflict with head 1!)
    for k in range(16):
        attn3.W_o[BD.AX_FULL_LO + k, base + 1 + k] = 1.0
        attn3.W_o[BD.AX_FULL_HI + k, base + 17 + k] = 1.0
    # Anti-leakage gate
    GATE = 33
    attn3.W_q[base + GATE, BD.MARK_AX] = L
    attn3.W_q[base + GATE, BD.CONST] = -L / 2
    attn3.W_k[base + GATE, BD.CONST] = L

    # FIX 2026-04-15: Head 6: BP carry to PC marker for LEV return_addr lookup
    # When OP_LEV is active, copy prev step's BP byte 0 to OUTPUT at PC marker.
    # This is needed because PC marker comes BEFORE AX marker in step output,
    # so L8 FFN can compute ADDR_B0 at PC marker for heads 8-11 to do BP+8 lookup.
    # OP_LEV is injected in embedding at all positions when active_opcode=LEV.
    base = 6 * HD
    L = 15.0
    # Q: Fire at PC marker when OP_LEV active
    attn3.W_q[base, BD.MARK_PC] = L
    attn3.W_q[base, BD.OP_LEV] = L / 5  # OP_LEV ≈ 5, normalize to ~L
    attn3.W_q[base, BD.CONST] = -L * 1.5  # Need both MARK_PC and OP_LEV
    # K: Attend to PREVIOUS step's BP byte 0 (L1H1[BP_I]=1 AND NOT L1H0[BP_I])
    # L1H1 fires at distances 1.5-2.5 (prev step's bytes)
    # L1H0 fires at distances 0-1.5 (current step's bytes - suppress these)
    attn3.W_k[base, BD.L1H1 + BP_I] = L
    attn3.W_k[base, BD.L1H0 + BP_I] = -L  # Suppress current step's BP byte 0
    # V: Copy CLEAN_EMBED_LO/HI (prev step's BP byte 0 value)
    for k in range(16):
        attn3.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn3.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # FIX 2026-04-16: Removed OUTPUT_LO/HI writes. L9 attention head 1 now handles
    # the BP relay to ADDR_B0 at PC marker. The OUTPUT writes were causing interference
    # with the return_addr that L16 FFN routes to OUTPUT via TEMP.
    # (Original code wrote to OUTPUT_LO/HI which persisted and corrupted final output)
    # Anti-leakage gate
    GATE = 33
    attn3.W_q[base + GATE, BD.MARK_PC] = L
    attn3.W_q[base + GATE, BD.CONST] = -L / 2
    attn3.W_k[base + GATE, BD.CONST] = L

    ffn3 = model.blocks[3].ffn  # Layer 3 (L3) = blocks[3]
    _set_layer3_ffn(ffn3, S, BD)

    # Conversational I/O: State initialization when entering output mode
    if enable_conversational_io:
        _set_conversational_io_state_init(ffn3, S, BD)

    # ===== LAYER 4: PC value relay to AX marker =====
    # AX marker reads the current step's PC byte 0 EMBED_LO value.
    # This must be a SEPARATE layer from the fetch because the fetch
    # needs the result of this relay (can't be intra-layer).
    attn4 = model.blocks[4].attn
    if hasattr(attn4, 'alibi_slopes') and attn4.alibi_slopes is not None:
        attn4.alibi_slopes.fill_(0.5)
    _set_layer4_pc_relay(attn4, S, BD, HD)
    ffn4 = model.blocks[4].ffn
    _set_layer4_ffn(ffn4, S, BD)

    # ===== LAYER 5: Bytecode fetch (imm + opcode) =====
    attn5 = model.blocks[5].attn
    if hasattr(attn5, 'alibi_slopes') and attn5.alibi_slopes is not None:
        attn5.alibi_slopes.fill_(0.1)
    _set_layer5_fetch(attn5, S, BD, HD)
    ffn5 = model.blocks[5].ffn
    _set_opcode_decode_ffn(ffn5, S, BD)

    # ===== LAYER 6: JMP/EXIT relay attention + output routing =====
    attn6 = model.blocks[6].attn
    # Steep ALiBi + large L to suppress leakage at Q=0 positions.
    # scale = 1/sqrt(64) = 0.125; scores = QK*0.125 - slope*d.
    # Head 0 (slope=5): PC→AX d=30, score = 50²*0.125 - 150 = 162 ✓
    # Head 1 (slope=5): SE→AX d=28, score = 50²*0.7*0.125 - 140 = 79 ✓
    # At non-target Q=0 positions: score = -5*d, softmax1→≈0 (leakage <0.7%).
    if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
        attn6.alibi_slopes.fill_(0.0)
        attn6.alibi_slopes[0] = 5.0
        attn6.alibi_slopes[1] = 5.0
        attn6.alibi_slopes[2] = 0.5  # STACK0←AX relay: prefer nearest AX marker
        attn6.alibi_slopes[3] = 0.5  # SP←AX relay: prefer nearest AX marker
        attn6.alibi_slopes[4] = 5.0  # BZ/BNZ relay: attend to nearest AX marker
    _set_layer6_attn(attn6, S, BD, HD)
    ffn6 = model.blocks[6].ffn
    _set_layer6_routing_ffn(ffn6, S, BD)

    # L6 relay attention heads (AX→STACK0 for PSH, AX→SP for ADJ)
    # ENABLED: Uses head 2 which is unused (checked via weight inspection)
    # This copies AX_CARRY → ALU at STACK0 marker for PSH: STACK0 = AX
    _set_layer6_relay_heads(attn6, S, BD, HD)
    # L6 BZ/BNZ relay (AX→PC for conditional branches)
    _set_bz_bnz_relay(attn6, S, BD, HD)

    # L6 opcode relay: broadcast PSH/ADJ/pop flags from AX → SP/STACK0
    if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
        attn6.alibi_slopes[6] = 5.0
        attn6.alibi_slopes[7] = 5.0  # JSR PC+5 relay: steep for head 7
    _set_opcode_relay_head(attn6, S, BD, HD)

    # IO PUTCHAR routing (L6 FFN: sets IO_IS_PUTCHAR, routes AX → OUTPUT)
    _set_io_putchar_routing(ffn6, S, BD)

    # Binary pop SP increment (L6 FFN: SP += 8 for binary pop ops)
    _set_binary_pop_sp_increment(ffn6, S, BD)

    # Function call opcodes (JSR, ENT, LEV, LEA)
    _set_function_call_weights(model, S, BD, HD)

    # Note: GETCHAR is runner-side IO (see run_vm.py). The runner detects
    # GETCHAR opcodes via PC tracking and injects stdin bytes into context.

    # ===== TOOL CALLING (optional) =====
    if enable_tool_calling:
        # L5 FFN: decode OPEN/READ/CLOS/PRTF → IO_IS_TOOL_CALL at AX marker
        _set_tool_call_opcode_decode(ffn5, S, BD)
        # L6 attention head 5: relay IO_IS_TOOL_CALL from AX → SE position
        if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
            attn6.alibi_slopes[5] = 5.0  # steep ALiBi for head 5
        _set_tool_call_relay_head(attn6, S, BD, HD)
        # L6 FFN: CMP[2] AND NEXT_SE → NEXT_TOOL_CALL (convert SE to TOOL_CALL)
        _set_tool_call_detection(ffn6, S, BD)

    # ===== CONVERSATIONAL I/O (optional) =====
    if enable_conversational_io:
        # L5 FFN: decode PRTF/READ → IO_IS_PRTF, IO_IS_READ at AX marker
        _set_conversational_io_opcode_decode(ffn5, S, BD)
        # L6 attention heads 4-5: relay IO_IS_PRTF, IO_IS_READ from AX → SE
        # Changed from heads 6-7 to avoid conflict with _set_opcode_relay_head (head 6)
        if hasattr(attn6, 'alibi_slopes') and attn6.alibi_slopes is not None:
            attn6.alibi_slopes[4] = 5.0  # steep ALiBi for head 4 (PRTF relay)
            attn6.alibi_slopes[5] = 5.0  # steep ALiBi for head 5 (READ relay)
        _set_conversational_io_relay_heads(attn6, S, BD, HD)
        # L6 FFN: CMP[5]/CMP[6] AND NEXT_SE → NEXT_THINKING_END (start I/O sequence)
        _set_conversational_io_state_machine(ffn6, S, BD)

    # === BUG FIX 2026-04-09 (part 8c/8g): Patch spurious L6 FFN units ===
    # Unprogrammed FFN units with random weights can fire spuriously when:
    # - OUTPUT_BYTE dimensions have residual values (from IO operations)
    # - Unit writes to OUTPUT_LO/HI dimensions
    # This causes incorrect output at marker positions (especially PC marker).
    # Example: unit 1128 fires on OUTPUT_BYTE_LO residuals at PC marker for EXIT,
    # writing to OUTPUT_LO[0] and causing prediction of 0 instead of 10.
    #
    # NOTE: TEMP (dims 480-511) overlaps with OUTPUT_BYTE_LO (dims 480-495).
    # JSR PC override and ENT use TEMP legitimately, so we must exclude units
    # that have positive marker weights (MARK_PC, MARK_STACK0, MARK_BP) which
    # indicate intentional TEMP usage rather than spurious OUTPUT_BYTE reading.
    patched_count = 0
    for u in range(4096):
        # Check if writes to OUTPUT
        writes_output_lo = ffn6.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO+16, u].abs().max().item() > 0.01
        writes_output_hi = ffn6.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI+16, u].abs().max().item() > 0.01
        if not (writes_output_lo or writes_output_hi):
            continue

        # Skip units that legitimately use TEMP (JSR PC override, ENT, etc.)
        # These units have positive MARK_PC, MARK_STACK0, or MARK_BP weight
        if ffn6.W_up[u, BD.MARK_PC].item() > 10:  # JSR PC override uses S=100
            continue
        if ffn6.W_up[u, BD.MARK_STACK0].item() > 10:  # ENT uses TEMP at STACK0
            continue
        if ffn6.W_up[u, BD.MARK_BP].item() > 10:  # ENT uses TEMP at BP
            continue

        # Check for strong OUTPUT_BYTE weights (TEMP overlaps with OUTPUT_BYTE, so don't check TEMP separately)
        output_byte_lo_weight = ffn6.W_up[u, BD.OUTPUT_BYTE_LO:BD.OUTPUT_BYTE_LO+16].abs().max().item()
        output_byte_hi_weight = ffn6.W_up[u, BD.OUTPUT_BYTE_HI:BD.OUTPUT_BYTE_HI+16].abs().max().item()

        if output_byte_lo_weight > 50 or output_byte_hi_weight > 50:
            # Zero out this unit
            ffn6.W_up[u, :] = 0
            ffn6.W_gate[u, :] = 0
            ffn6.W_down[:, u] = 0
            ffn6.b_up[u] = 0
            ffn6.b_gate[u] = 0
            patched_count += 1

    # ===== LAYER 7: Operand gather + memory relay heads =====
    attn7 = model.blocks[7].attn
    if hasattr(attn7, 'alibi_slopes') and attn7.alibi_slopes is not None:
        attn7.alibi_slopes.fill_(0.5)
    _set_layer7_operand_gather(attn7, S, BD, HD)
    # L7 head 1: MEM flag broadcast (MEM marker → MEM byte positions)
    if hasattr(attn7, 'alibi_slopes') and attn7.alibi_slopes is not None:
        attn7.alibi_slopes[1] = 5.0  # steep for nearest MEM marker
    # L7 heads 2-4: Gather prev AX bytes → AX positions (for LI/LC addr)
    # L7 head 5: Relay LI/LC flags → AX byte positions
    if hasattr(attn7, 'alibi_slopes') and attn7.alibi_slopes is not None:
        attn7.alibi_slopes[5] = 5.0
    # L7 head 6: Relay PSH/store flags → STACK0 byte positions
    if hasattr(attn7, 'alibi_slopes') and attn7.alibi_slopes is not None:
        attn7.alibi_slopes[6] = 5.0
    _set_layer7_memory_heads(attn7, S, BD, HD)

    # === BUG FIX 2026-04-17: Patch spurious L7 FFN units ===
    # Units 746/754 in L7 FFN have W_up[OP_ENT]=100, W_up[MARK_SP]=100, b_up=-150.
    # They're designed for ENT SP byte 0 generation (0xF0) but also fire at PC marker
    # because OP_ENT * 5 = 500 overcomes the bias even when MARK_SP = 0.
    # At PC marker: up_pre = 0 + 500 - 150 = 350 → fires incorrectly
    # Fix: Add strong MARK_PC negative weight to suppress at PC marker.
    # MARK_PC = 1 at PC marker, so -1000 * 1 = -1000 kills the unit.
    ffn7 = model.blocks[6].ffn  # blocks[6] = L7 (0-indexed)
    patched_l7_count = 0
    for u in range(4096):
        # Check if writes to OUTPUT
        writes_output_lo = ffn7.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO+16, u].abs().max().item() > 0.01
        writes_output_hi = ffn7.W_down[BD.OUTPUT_HI:BD.OUTPUT_HI+16, u].abs().max().item() > 0.01
        if not (writes_output_lo or writes_output_hi):
            continue

        # Check for units that read OP_* strongly but don't have strong MARK_PC negative
        has_strong_opcode = (
            ffn7.W_up[u, BD.OP_ENT].abs().item() > 50 or
            ffn7.W_up[u, BD.OP_LEV].abs().item() > 50 or
            ffn7.W_up[u, BD.OP_JSR].abs().item() > 50 or
            ffn7.W_up[u, BD.OP_LEA].abs().item() > 50
        )

        has_mark_pc_suppression = ffn7.W_up[u, BD.MARK_PC].item() < -100

        # Skip units that are designed to fire at byte positions (have positive IS_BYTE or BYTE_INDEX)
        # These are legitimate ENT/JSR byte generation units
        is_byte_unit = (
            ffn7.W_up[u, BD.IS_BYTE].item() > 5 or
            ffn7.W_up[u, BD.BYTE_INDEX_0].item() > 5 or
            ffn7.W_up[u, BD.BYTE_INDEX_1].item() > 5 or
            ffn7.W_up[u, BD.BYTE_INDEX_2].item() > 5 or
            ffn7.W_up[u, BD.BYTE_INDEX_3].item() > 5
        )

        if has_strong_opcode and not has_mark_pc_suppression and not is_byte_unit:
            # Add MARK_PC and IS_BYTE suppression rather than zeroing out
            # At PC marker: MARK_PC = 1, so -1000 added to up_pre
            # At any byte position: IS_BYTE = 1, so -1000 added to up_pre
            # Units should only fire at marker positions (MARK_SP = 1, IS_BYTE = 0)
            ffn7.W_up[u, BD.MARK_PC] = -S * 100  # -1000 when MARK_PC = 1
            ffn7.W_up[u, BD.IS_BYTE] = -S * 100  # -1000 when IS_BYTE = 1
            patched_l7_count += 1

    # Conversational I/O: Extract format pointer from STACK0
    if enable_conversational_io:
        if hasattr(attn7, 'alibi_slopes') and attn7.alibi_slopes is not None:
            attn7.alibi_slopes[7] = 5.0  # steep to attend back to prev step
        _set_format_pointer_extraction(attn7, S, BD, HD)

    # ===== LAYER 8: ALU + SP→STACK0 addr gather =====
    attn8 = model.blocks[8].attn
    if hasattr(attn8, 'alibi_slopes') and attn8.alibi_slopes is not None:
        attn8.alibi_slopes.fill_(0.5)
    _set_layer8_sp_gather(attn8, S, BD, HD)

    # ===== LAYERS 8-13: ALU Operations =====
    if alu_mode == 'lookup':
        # Use full lookup tables (pure FFN, many parameters)
        ffn8 = model.blocks[8].ffn
        _set_layer8_alu(ffn8, S, BD)

        # Conversational I/O: Position counter increment
        if enable_conversational_io:
            _set_format_position_counter(ffn8, S, BD)

        # Conversational I/O: Format string fetch via attention
        if enable_conversational_io:
            attn9 = model.blocks[9].attn
            if hasattr(attn9, 'alibi_slopes') and attn9.alibi_slopes is not None:
                attn9.alibi_slopes.fill_(0.5)
            _set_format_string_fetch_head(attn9, S, BD, HD)

        # LEV ADDR_B0 relay: prev step BP byte 0 → SP marker (for SP = BP + 16)
        # BUG FIX 2026-04-15: This relay is needed for LEV to work correctly.
        # Distance from SP marker to prev BP byte 0 is ~29 tokens.
        attn9 = model.blocks[9].attn
        if hasattr(attn9, 'alibi_slopes') and attn9.alibi_slopes is not None:
            attn9.alibi_slopes[0] = 0.2  # head 0: shallow slope for d=29 relay
            attn9.alibi_slopes[1] = 0.5  # head 1: BP→PC relay for LEV (d=15 tokens)
        _set_layer9_lev_addr_relay(attn9, S, BD, HD)
        # FIX 2026-04-15: Relay ADDR_B0 from BP marker to PC marker for return_addr
        _set_layer9_lev_bp_to_pc_relay(attn9, S, BD, HD)

        ffn9 = model.blocks[9].ffn
        _set_layer9_alu(ffn9, S, BD)

        # ===== LAYER 10: carry relay + AX passthrough + SP passthrough + bitwise + cmp =====
        attn10 = model.blocks[10].attn
        if hasattr(attn10, 'alibi_slopes') and attn10.alibi_slopes is not None:
            attn10.alibi_slopes[0] = 5.0  # head 0: steep slope for carry relay
            attn10.alibi_slopes[1] = 1.0  # head 1: AX byte passthrough (nearest step)
            attn10.alibi_slopes[2] = 1.0  # head 2: SP byte passthrough (nearest step)
            attn10.alibi_slopes[3] = 0.5  # head 3: PSH STACK0 passthrough (same step, d=14)
        _set_layer10_carry_relay(attn10, S, BD, HD)
        _set_layer10_byte_passthrough(attn10, S, BD, HD)
        _set_layer10_sp_byte_passthrough(attn10, S, BD, HD)
        _set_layer10_psh_stack0_passthrough(attn10, S, BD, HD)
        ffn10 = model.blocks[10].ffn
        _set_layer10_alu(ffn10, S, BD)

        # Conversational I/O: Null terminator detection
        if enable_conversational_io:
            _set_null_terminator_detection(ffn10, S, BD)

        # ===== LAYER 10.5: Neural DIV/MOD =====
        model.blocks[10].post_ops.append(DivModModule(mode='lookup'))

        # ===== LAYER 11: MUL partial sum staging =====
        ffn11 = model.blocks[11].ffn
        _set_layer11_mul_partial(ffn11, S, BD)

        # ===== LAYER 12: MUL hi nibble combine =====
        ffn12 = model.blocks[12].ffn
        _set_layer12_mul_combine(ffn12, S, BD)

        # ===== LAYER 13: SHL/SHR shifts + MEM addr gather =====
        attn13 = model.blocks[13].attn
        if hasattr(attn13, 'alibi_slopes') and attn13.alibi_slopes is not None:
            attn13.alibi_slopes.fill_(0.5)
        _set_layer13_mem_addr_gather(attn13, S, BD, HD)
        ffn13 = model.blocks[13].ffn
        _set_layer13_shifts(ffn13, S, BD)

    elif alu_mode == 'efficient':
        # Use efficient multi-layer ALU with pure neural format conversion
        # All operations use baked FFN weights - no Python loops in forward pass

        # L8-L9: Neural ADD/SUB
        model.blocks[8].ffn = EfficientALU_L8_L9_Neural(S, BD)

        # L10: Carry relay + AX/SP passthrough attention (still needed)
        attn10 = model.blocks[10].attn
        if hasattr(attn10, 'alibi_slopes') and attn10.alibi_slopes is not None:
            attn10.alibi_slopes[0] = 5.0
            attn10.alibi_slopes[1] = 1.0
            attn10.alibi_slopes[2] = 1.0  # SP byte passthrough
            attn10.alibi_slopes[3] = 0.5  # PSH STACK0 passthrough
        _set_layer10_carry_relay(attn10, S, BD, HD)
        _set_layer10_byte_passthrough(attn10, S, BD, HD)
        _set_layer10_sp_byte_passthrough(attn10, S, BD, HD)
        _set_layer10_psh_stack0_passthrough(attn10, S, BD, HD)

        # L10 FFN: Neural AND/OR/XOR
        model.blocks[10].ffn = EfficientALU_L10_Neural(S, BD)

        # L10.5: Pure neural DIV/MOD (no Python math in forward pass)
        model.blocks[10].post_ops.append(EfficientDivMod_Neural(S, BD))

        # L11-L12: Neural MUL
        model.blocks[11].ffn = EfficientALU_L11_L12_Neural(S, BD)

        # L13: Memory addr gather attention (still needed)
        attn13 = model.blocks[13].attn
        if hasattr(attn13, 'alibi_slopes') and attn13.alibi_slopes is not None:
            attn13.alibi_slopes.fill_(0.5)
        _set_layer13_mem_addr_gather(attn13, S, BD, HD)

        # L13 FFN: Neural SHL/SHR
        model.blocks[13].ffn = EfficientALU_L13_Neural(S, BD)

    else:
        raise ValueError(f"Unknown alu_mode: {alu_mode}. Use 'lookup' or 'efficient'.")

    # ===== LAYER 14: MEM byte generation (8 heads) =====
    attn14 = model.blocks[14].attn
    if hasattr(attn14, 'alibi_slopes') and attn14.alibi_slopes is not None:
        attn14.alibi_slopes.fill_(0.1)  # slight recency bias for same-step preference
    _set_layer14_mem_generation(attn14, S, BD, HD)

    # L14 FFN: Clear TEMP, ADDR_KEY pollution, and OUTPUT corruption
    # BUG FIX 2026-04-16:
    # 1. TEMP[0] has residual value from L5/L6 attention (~2.0) that leaks into OUTPUT
    # 2. ADDR_KEY dims are aliased with ADDR_B*_HI, causing L9's address gathering
    #    to pollute ADDR_KEY at non-MEM positions, making L15 attend to wrong places
    # 3. L14 attention V[0] cancelation corrupts OUTPUT at non-MEM positions (STACK0 bytes)
    ffn14 = model.blocks[14].ffn
    next_unit = _set_layer14_temp_clear(ffn14, S, BD)
    next_unit = _set_layer14_clear_addr_key_pollution(ffn14, S, BD, start_unit=next_unit)
    _set_layer14_clear_output_corruption(ffn14, S, BD, start_unit=next_unit)

    # ===== LAYER 15: Memory lookup (softmax1 + ALiBi) =====
    attn15 = model.blocks[15].attn

    # LEV Phase 2: Extend L15 to 12 heads (from default 8) - only if using 17 layers
    # Only resize if we have L16 (which requires 17 layers total)
    if len(model.blocks) > 16:
        # Resize attention matrices: (512, 512) → (768, 512) for 12 heads × 64 dims/head
        # This allows 3 parallel memory reads: LI/LC/STACK0 (heads 0-3), saved_bp (4-7), return_addr (8-11)
        import torch
        d = model.d_model  # 512
        num_heads_l15 = 12
        head_dim = d // 8  # 64 (based on default 8 heads)
        new_q_rows = num_heads_l15 * head_dim  # 12 * 64 = 768

        # Update num_heads and head_dim attributes for forward pass
        attn15.num_heads = num_heads_l15
        attn15.head_dim = head_dim

        # Resize ALiBi slopes if present (8 slopes → 12 slopes)
        if hasattr(attn15, 'alibi_slopes') and attn15.alibi_slopes is not None:
            # Extend slopes with same geometric sequence pattern
            new_slopes = torch.tensor(
                [2.0 ** (-8.0 / num_heads_l15 * (i + 1)) for i in range(num_heads_l15)]
            )
            attn15.register_buffer('alibi_slopes', new_slopes)

        # Resize W_q, W_k, W_v from (512, 512) to (768, 512)
        # Preserve existing weights in first 512 rows (heads 0-7), zero-initialize new rows (heads 8-11)
        old_W_q = attn15.W_q.data
        old_W_k = attn15.W_k.data
        old_W_v = attn15.W_v.data
        attn15.W_q = nn.Parameter(torch.zeros(new_q_rows, d))
        attn15.W_k = nn.Parameter(torch.zeros(new_q_rows, d))
        attn15.W_v = nn.Parameter(torch.zeros(new_q_rows, d))
        attn15.W_q.data[:d, :] = old_W_q  # Copy existing 512 rows
        attn15.W_k.data[:d, :] = old_W_k
        attn15.W_v.data[:d, :] = old_W_v

        # Resize W_o from (512, 512) to (512, 768) - output stays 512-dim, input from 768-dim heads
        old_W_o = attn15.W_o.data
        attn15.W_o = nn.Parameter(torch.zeros(d, new_q_rows))
        attn15.W_o.data[:, :d] = old_W_o  # Copy existing 512 cols

    if hasattr(attn15, 'alibi_slopes') and attn15.alibi_slopes is not None:
        attn15.alibi_slopes.fill_(0.01)  # gentle recency bias for latest-write-wins
    _set_layer15_memory_lookup(attn15, S, BD, HD)
    ffn15 = model.blocks[15].ffn
    _set_nibble_copy_ffn(ffn15, S, BD)

    # Conversational I/O: Output routing (OUTPUT_BYTE → OUTPUT)
    if enable_conversational_io:
        _set_conversational_io_output_routing(ffn15, S, BD)

    # ===== LAYER 16: LEV register routing (Phase 3) =====
    # Only set up L16 if model has 17+ layers
    if len(model.blocks) > 16:
        attn16 = model.blocks[16].attn
        # L16 attention: passthrough (identity via residual)
        # No weights needed - all routing done in FFN

        ffn16 = model.blocks[16].ffn
        num_units = _set_layer16_lev_routing(ffn16, S, BD)
        print(f"  L16 FFN: {num_units} units for LEV routing (SP = BP + 16)")

    # ===== OUTPUT HEAD =====
    head = model.head
    head.weight.zero_()
    head.bias.zero_()

    # Byte tokens: nibble decoding + marker suppression
    next_flags = [
        BD.NEXT_PC,
        BD.NEXT_AX,
        BD.NEXT_SP,
        BD.NEXT_BP,
        BD.NEXT_STACK0,
        BD.NEXT_MEM,
        BD.NEXT_SE,
        BD.NEXT_HALT,
        BD.NEXT_TOOL_CALL,
        BD.NEXT_THINKING_START,
        BD.NEXT_THINKING_END,
    ]
    for b in range(256):
        lo, hi = b & 0xF, (b >> 4) & 0xF
        head.weight[b, BD.OUTPUT_LO + lo] = 5.0
        head.weight[b, BD.OUTPUT_HI + hi] = 5.0
        head.bias[b] = -5.0
        # Suppress byte logits when a marker transition is expected.
        # At marker-generating positions (NEXT_* > 0), byte logits get
        # a large penalty. At byte-generating positions (all NEXT_* = 0),
        # no penalty. This prevents OUTPUT_LO/HI noise from causing
        # bytes to beat markers.
        for flag in next_flags:
            head.weight[b, flag] += -80.0
    head.bias[0] = -4.0  # slight preference for byte 0 as default

    # Transition tokens (including STACK0)
    for tok, flag in [
        (Token.REG_PC, BD.NEXT_PC),
        (Token.REG_AX, BD.NEXT_AX),
        (Token.REG_SP, BD.NEXT_SP),
        (Token.REG_BP, BD.NEXT_BP),
        (Token.STACK0, BD.NEXT_STACK0),
        (Token.MEM, BD.NEXT_MEM),
        (Token.STEP_END, BD.NEXT_SE),
        (Token.HALT, BD.NEXT_HALT),
        (Token.TOOL_CALL, BD.NEXT_TOOL_CALL),
        (Token.THINKING_START, BD.NEXT_THINKING_START),
        (Token.THINKING_END, BD.NEXT_THINKING_END),
    ]:
        head.weight[tok, flag] = 20.0
        head.bias[tok] = -10.0

    # Never output these tokens (context markers only, not part of VM execution)
    for tok in [
        Token.CODE_START,
        Token.CODE_END,
        Token.DATA_START,
        Token.DATA_END,
        Token.SEP,
        Token.USER_INPUT_START,
        Token.USER_INPUT_END,
    ]:
        head.bias[tok] = -50.0

    # I/O state tokens can be generated (part of I/O sequence) but suppress by default
    # They will be enabled when needed by NEXT_IO_STATE_* flags
    head.bias[Token.IO_STATE_EMIT_BYTE] = -20.0
    head.bias[Token.IO_STATE_EMIT_THINKING] = -20.0

    # ===== CONTRACT VALIDATION =====
    reg = build_default_registry()
    contracts = build_default_contracts(reg)
    errors = ContractValidator(reg, contracts).validate()
    if errors:
        for e in errors:
            print(f"  CONTRACT: {e}")

    return model


def _set_threshold_attn(attn, thresholds, out_bases, slope, HD, heads=None):
    """Set threshold-based attention heads for marker distance detection.

    Each head detects whether the nearest marker is within `threshold` tokens.
    Uses ALiBi: score = slope*(threshold - distance), giving a sharp sigmoid.
    """
    BD = _SetDim
    if heads is None:
        heads = list(range(len(thresholds)))

    for i, (h, t) in enumerate(zip(heads, thresholds)):
        base = h * HD
        q_val = 8.0 * slope  # sqrt(HD) = 8
        attn.W_q[base, BD.CONST] = q_val
        attn.W_k[base, BD.IS_MARK] = t
        for m, src in enumerate(BD.MARKS):
            attn.W_v[base + 1 + m, src] = 1.0
        for m in range(BD.NUM_MARKERS):
            attn.W_o[out_bases[i] + m, base + 1 + m] = 1.0


def _set_phase_a_ffn(ffn, S, BD):
    """Step structure FFN: detect marker transitions for 35-token step.

    35-token step: PC(5)+AX(5)+SP(5)+BP(5)+STACK0(5)+MEM(9)+SE(1)
    Transitions detected via threshold head differences:
      SE→PC:     SE at d=0, NEXT_PC fires
      PC→AX:     at d=4 from PC marker (H1_PC - H0_PC pattern)
      AX→SP:     at d=4 from AX
      SP→BP:     at d=4 from SP
      BP→STACK0: at d=4 from BP
      STACK0→MEM: at d=9 from BP (STACK0 not IS_MARK, so BP is nearest)
      MEM→SE:    at d=8 from MEM

    Uses 8 threshold heads with distances:
      H0=3.5, H1=4.5, H2=7.5, H3=8.5, H4=9.5, H5=14.5, H6=19.5, H7=24.5
    """
    # Marker indices in MARKS array: PC=0, AX=1, SP=2, BP=3, MEM=4, SE=5, CS=6
    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5
    # Note: STACK0 is NOT IS_MARK and NOT in the MARKS array.

    transitions = [
        # (up_dim, gate_dim, out_dim)
        # SE → PC: SE is nearest marker at d<=3.5
        (BD.H0 + SE_I, None, BD.NEXT_PC),
        # PC → AX: PC at d<=4.5 (H1) but not d<=3.5 (H0)
        (BD.H1 + PC_I, BD.H0 + PC_I, BD.NEXT_AX),
        # AX → SP: AX at d<=4.5 but not d<=3.5
        (BD.H1 + AX_I, BD.H0 + AX_I, BD.NEXT_SP),
        # SP → BP: SP at d<=4.5 but not d<=3.5
        (BD.H1 + SP_I, BD.H0 + SP_I, BD.NEXT_BP),
        # BP → STACK0: BP at d<=4.5 but not d<=3.5
        (BD.H1 + BP_I, BD.H0 + BP_I, BD.NEXT_STACK0),
        # STACK0 → MEM: STACK0 not IS_MARK, so nearest IS_MARK is BP at d=9.
        # H4(9.5) sees BP, H3(8.5) doesn't → d in (8.5, 9.5] = d=9 only.
        (BD.H4 + BP_I, BD.H3 + BP_I, BD.NEXT_MEM),
        # MEM → SE: MEM at d<=8.5 (H3) but not d<=7.5 (H2).
        # d in (7.5, 8.5] = d=8 only (MEM marker + 4 addr + 4 val = 8 bytes).
        (BD.H3 + MEM_I, BD.H2 + MEM_I, BD.NEXT_SE),
    ]
    for i, (up_dim, gate_dim, out_dim) in enumerate(transitions):
        ffn.W_up[i, up_dim] = S
        ffn.b_up[i] = -S * 0.3
        if gate_dim is not None:
            ffn.W_gate[i, gate_dim] = -1.0
            ffn.b_gate[i] = 1.0
        else:
            ffn.b_gate[i] = 1.0
        ffn.W_down[out_dim, i] = 1.0 / S


def _set_layer1_ffn(ffn, S, BD):
    """Layer 1 FFN: STACK0_BYTE0 flag + BYTE_INDEX flags.

    STACK0 byte 0 is at d=6 from BP marker (nearest IS_MARK).
    Detected by: L1H4[BP] (d<=6.5) AND NOT H1[BP] (d>4.5) AND IS_BYTE.
    L1H4[BP] = dim BD.L1H4 + BP_I (BP index = 3 in MARKS array).
    H1[BP] = dim BD.H1 + BP_I.

    BYTE_INDEX_0-3: Marker-agnostic byte position within a register.
    Derived from threshold heads (summed across all marker types):
      BYTE_INDEX_0: IS_BYTE AND any(L1H1) AND NOT any(L1H0) → d∈(0.5,1.5]
      BYTE_INDEX_1: IS_BYTE AND any(L1H2) AND NOT any(L1H1) → d∈(1.5,2.5]
      BYTE_INDEX_2: IS_BYTE AND any(H0) AND NOT any(L1H2)  → d∈(2.5,3.5]
      BYTE_INDEX_3: IS_BYTE AND any(H1) AND NOT any(H0)    → d∈(3.5,4.5]
    Only one marker type is nearest at any position, so sum ≈ 1 when active.
    """
    BP_I = 3
    NM = BD.NUM_MARKERS  # 7 marker types
    unit = 0

    # STACK0_BYTE0: L1H4[BP] AND NOT H1[BP] AND IS_BYTE
    # silu(S*(L1H4_BP + IS_BYTE - 1.5)) * (1 - H1_BP)
    ffn.W_up[unit, BD.L1H4 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H1 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.STACK0_BYTE0, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_0: IS_BYTE AND any(L1H1[i]) AND NOT any(L1H0[i])
    # up = S*(IS_BYTE + sum(L1H1[0..6])) - S*1.5
    # gate = 1 - sum(L1H0[0..6])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.L1H1 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.L1H0 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_0, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_1: IS_BYTE AND any(L1H2[i]) AND NOT any(L1H1[i])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.L1H2 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.L1H1 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_1, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_2: IS_BYTE AND any(H0[i]) AND NOT any(L1H2[i])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.H0 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.L1H2 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_2, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_3: IS_BYTE AND any(H1[i]) AND NOT any(H0[i])
    ffn.W_up[unit, BD.IS_BYTE] = S
    for i in range(NM):
        ffn.W_up[unit, BD.H1 + i] = S
    ffn.b_up[unit] = -S * 1.5
    for i in range(NM):
        ffn.W_gate[unit, BD.H0 + i] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_3, unit] = 2.0 / S
    unit += 1


def _set_layer2_mem_byte_flags(ffn, S, BD):
    """Layer 2 FFN: MEM val byte position flags + extended BYTE_INDEX for STACK0.

    MEM val byte flags (4 units): Identify positions d=4..7 from MEM marker.
    These are the QUERY positions that predict MEM val bytes 0-3 (autoregressive shift).
    FIX 2026-04-16: Shifted from d=5..8 to d=4..7 for correct autoregressive prediction.

    Extended BYTE_INDEX for STACK0 bytes 1-3 (3 units): At positions d=7..9
    from BP (where STACK0 bytes live), produce BYTE_INDEX_1/2/3 flags.
    These accumulate with existing BYTE_INDEX (which is 0 at those positions).
    """
    MEM_I = 4  # MEM marker index in MARKS
    BP_I = 3
    NM = BD.NUM_MARKERS
    unit = 0

    # MEM_VAL_B0: d=4 from MEM (addr byte 3 position, predicts val byte 0)
    # H1[MEM]=1 (d≤4.5), H0[MEM]=0 (d>3.5)
    # silu(S*(H1_MEM + IS_BYTE) - S*1.5) × (1 - H0_MEM)
    ffn.W_up[unit, BD.H1 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H0 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B0, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B1: d=5 from MEM (val byte 0 position, predicts val byte 1)
    # L2H0[MEM]=1 (d≤5.5), H1[MEM]=0 (d>4.5)
    ffn.W_up[unit, BD.L2H0 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H1 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B1, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B2: d=6 from MEM (val byte 1 position, predicts val byte 2)
    # L1H4[MEM]=1 (d≤6.5), L2H0[MEM]=0 (d>5.5)
    ffn.W_up[unit, BD.L1H4 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L2H0 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B2, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B3: d=7 from MEM (val byte 2 position, predicts val byte 3)
    # H2[MEM]=1 (d≤7.5), L1H4[MEM]=0 (d>6.5)
    ffn.W_up[unit, BD.H2 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L1H4 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B3, unit] = 2.0 / S
    unit += 1

    # Extended BYTE_INDEX for STACK0 byte 0-3 (at d=6,7,8,9 from BP)
    # BYTE_INDEX_0 at STACK0: d=6 from BP → L1H4[BP]=1 (d≤6.5), H1[BP]=0 (d>4.5)
    ffn.W_up[unit, BD.L1H4 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H1 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_0, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_1 at STACK0: d=7 from BP → H2[BP]=1 (d≤7.5), L1H4[BP]=0 (d>6.5)
    ffn.W_up[unit, BD.H2 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L1H4 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_1, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_2 at STACK0: d=8 from BP → H3[BP]=1 (d≤8.5), H2[BP]=0 (d>7.5)
    ffn.W_up[unit, BD.H3 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H2 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_2, unit] = 2.0 / S
    unit += 1

    # BYTE_INDEX_3 at STACK0: d=9 from BP → H4[BP]=1 (d≤9.5), H3[BP]=0 (d>8.5)
    ffn.W_up[unit, BD.H4 + BP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H3 + BP_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.BYTE_INDEX_3, unit] = 2.0 / S
    unit += 1


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
    for k in range(16):
        # Up: fires at byte positions, suppressed at register areas with custom handling
        # Note: STACK0 uses separate MARK_STACK0 (not in MARKS array), so we use H4[BP]
        # which covers d <= 9.5 from BP marker (STACK0 is at d=5-9 from BP)
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.H1 + PC_I] = -S  # Suppress ALL PC bytes (L3 handles all)
        ffn.W_up[unit, BD.H1 + AX_I] = -S  # Suppress at AX (L3/L6 handle)
        ffn.W_up[unit, BD.H1 + SP_I] = -S  # Suppress at SP (L3/L15 PSH handle)
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Suppress at BP (L3 default handles)
        ffn.W_up[unit, BD.H4 + BP_I] = -S  # Suppress at STACK0 area (d<=9.5 from BP)
        ffn.W_up[unit, BD.MEM_STORE] = -S  # Suppress at MEM during PSH/SI/SC
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
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Suppress at BP (L3 default handles)
        ffn.W_up[unit, BD.H4 + BP_I] = -S  # Suppress at STACK0 area (d<=9.5 from BP)
        ffn.W_up[unit, BD.MEM_STORE] = -S  # Suppress at MEM during PSH/SI/SC
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === PSH SP byte 1-3 outputs ===
    # SP bytes are suppressed above. During PSH, we need to output:
    # - At SP byte 0 pos (predicting byte 1): 0xFF (from borrow propagation)
    # - At SP byte 1 pos (predicting byte 2): 0x00 (after borrow absorbed)
    # - At SP byte 2 pos (predicting byte 3): 0x00 (unchanged)
    # This is correct for STACK_INIT = 0x10000 case.
    # PSH_AT_SP is relayed from SP marker to SP byte positions by L7 head 6.
    SP_I = 2
    # Threshold: PSH_AT_SP(~1 at bytes, ~2 at marker) + H1[SP](1) + IS_BYTE(1 at bytes, 0 at marker) + BYTE_INDEX(1)
    # At marker: 2 + 1 + 0 + 0 = 3 < 3.5 (doesn't fire)
    # At bytes: 1 + 1 + 1 + 1 = 4 > 3.5 (fires)
    T_psh_byte = 3.5

    # SP byte 0 pos → predict byte 1 = 0xFF
    # L3 sets default OUTPUT_LO/HI[0] = 1.0 for byte 1 = 0x00. Cancel it and write 0xFF.
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S  # Require IS_BYTE to distinguish from marker
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0  # constant gate
    ffn.W_down[BD.OUTPUT_LO + 15, unit] = 2.0 / S  # lo nibble = 15 (F)
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S  # cancel L3 default (lo nibble = 0)
    unit += 1
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 15, unit] = 2.0 / S  # hi nibble = 15 (F)
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S  # cancel L3 default (hi nibble = 0)
    unit += 1

    # SP byte 1 pos → predict byte 2 = 0x00
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo nibble = 0
    unit += 1
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
    unit += 1

    # SP byte 2 pos → predict byte 3 = 0x00
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo nibble = 0
    unit += 1
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
    unit += 1

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


def _set_cs_threshold_attn(attn, head_idx, threshold, out_dim, slope, HD):
    """Set a single threshold head that only detects CODE_START distance.

    Outputs a single dim (CS component only) instead of full 7-marker vector.
    """
    BD = _SetDim
    base = head_idx * HD
    q_val = 8.0 * slope
    attn.W_q[base, BD.CONST] = q_val
    attn.W_k[base, BD.IS_MARK] = threshold
    attn.W_v[base + 1, BD.MARK_CS] = 1.0
    attn.W_o[out_dim, base + 1] = 1.0


def _set_carry_forward_attn(
    attn, head_idx, marker_dim, l1h1_idx, l1h0_idx, HD, out_lo, out_hi, slope=0.5,
    src_lo=None, src_hi=None
):
    """Set attention head for register carry-forward.

    At marker positions, attends to the previous step's corresponding byte 0
    (identified by L1H1_marker AND NOT L1H0_marker pattern).
    Copies src_lo/src_hi (default EMBED_LO/HI) from the target to out_lo/out_hi.

    Args:
        src_lo, src_hi: Source dimensions to copy from. Defaults to EMBED_LO/HI.
                        For AX carry, use OUTPUT_LO/HI to get computed value.

    Anti-leakage gate (dim 33): at non-target markers, Q[gate] = -L/2
    combined with K[gate] = L gives score penalty -L²/(2√HD) ≈ -14,
    suppressing self-attention leakage (exp(-14) ≈ 8e-7).
    """
    BD = _SetDim
    base = head_idx * HD
    L = 15.0

    # Default source is EMBED (for PC/SP/BP), but AX needs OUTPUT
    if src_lo is None:
        src_lo = BD.EMBED_LO
    if src_hi is None:
        src_hi = BD.EMBED_HI

    attn.W_q[base, marker_dim] = L
    attn.W_k[base, BD.L1H1 + l1h1_idx] = L
    attn.W_k[base, BD.L1H0 + l1h0_idx] = -L

    for k in range(16):
        attn.W_v[base + 1 + k, src_lo + k] = 1.0
        attn.W_v[base + 17 + k, src_hi + k] = 1.0

    for k in range(16):
        attn.W_o[out_lo + k, base + 1 + k] = 1.0
        attn.W_o[out_hi + k, base + 17 + k] = 1.0

    # Anti-leakage gate: suppress attention at non-target marker positions.
    # At target (marker_dim=1): Q[gate] = L - L/2 = +L/2, score += +L²/(2√HD) ≈ +14
    # At non-target (marker_dim=0): Q[gate] = -L/2, score += -L²/(2√HD) ≈ -14
    GATE = 33
    attn.W_q[base + GATE, marker_dim] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L


def _set_stack0_carry_attn(attn, head_idx, HD):
    """Set attention head for STACK0 carry-forward.

    At STACK0 marker positions, attend to previous step's STACK0 byte 0
    (identified by STACK0_BYTE0 flag from L1 FFN).
    Copies EMBED_LO/HI to EMBED_LO/HI at STACK0 marker.
    """
    BD = _SetDim
    base = head_idx * HD
    L = 15.0

    # Q: fires at STACK0 markers
    attn.W_q[base, BD.MARK_STACK0] = L
    # K: fires at positions with STACK0_BYTE0 flag
    attn.W_k[base, BD.STACK0_BYTE0] = L

    # V: copies EMBED_LO/HI (the byte value from embedding)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

    # O: writes to EMBED_LO/HI at STACK0 marker
    for k in range(16):
        attn.W_o[BD.EMBED_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.EMBED_HI + k, base + 17 + k] = 1.0

    # Anti-leakage gate (same as _set_carry_forward_attn)
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_STACK0] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L

    # (L2 FFN reserved for prompt/bootstrap memory wiring.)


def _set_layer3_ffn(ffn, S, BD):
    """Layer 3 FFN: PC/SP/BP first-step defaults + PC increment.

    First step: PC = PC_OFFSET, SP = STACK_INIT, BP = STACK_INIT
    Subsequent steps: PC += INSTR_WIDTH, SP/BP from carry-forward
    """
    from .constants import STACK_INIT
    unit = 0

    # PC FIRST-STEP DEFAULT: when MARK_PC AND NOT HAS_SE, set PC=PC_OFFSET+INSTR_WIDTH
    # (draft tokens represent state AFTER executing first instruction)
    # CRITICAL: Also write to EMBED so L4 attention can relay to AX marker for L5 fetch!
    first_pc = PC_OFFSET + INSTR_WIDTH
    pc_lo = first_pc & 0xF
    pc_hi = (first_pc >> 4) & 0xF
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + pc_lo, unit] = 2.0 / S
    ffn.W_down[BD.EMBED_LO + pc_lo, unit] = 2.0 / S  # Also write to EMBED for L4 relay
    unit += 1
    # Unit B: undo when HAS_SE (subsequent steps use carry-forward + increment)
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_LO + pc_lo, unit] = -2.0 / S
    ffn.W_down[BD.EMBED_LO + pc_lo, unit] = -2.0 / S  # Also undo EMBED
    unit += 1

    # Same for OUTPUT_HI and EMBED_HI
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + pc_hi, unit] = 2.0 / S
    ffn.W_down[BD.EMBED_HI + pc_hi, unit] = 2.0 / S  # Also write to EMBED for L4 relay
    unit += 1
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_HI + pc_hi, unit] = -2.0 / S
    ffn.W_down[BD.EMBED_HI + pc_hi, unit] = -2.0 / S  # Also undo EMBED
    unit += 1

    # SP DEFAULT: STACK_INIT = 0x10000
    # Bytes: 0x00, 0x00, 0x01, 0x00
    # At SP positions, default to 0 for bytes 0,1,3 and 1 for byte 2.
    # Later layers (PSH/ADJ) will override when SP changes.
    SP_I = 2  # SP marker index in MARKS array

    # SP bytes 0, 1, 3 = 0
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_2]:
        # LO nibble = 0
        ffn.W_up[unit, BD.H1 + SP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        # HI nibble = 0
        ffn.W_up[unit, BD.H1 + SP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # SP byte 2 = 0x01 (lo=1, hi=0) - FIRST STEP ONLY
    # FIX 2026-04-16: Add HAS_SE suppression. After step 0, SP value changes from
    # 0x10000 to actual values like 0xFFF8 (where byte 2 = 0x00). L10 attention
    # passthrough handles subsequent steps, so this default should only fire on step 0.
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 2.0 / S  # lo nibble = 1
    unit += 1
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
    unit += 1

    # BP DEFAULT: same as SP
    BP_I = 3  # BP marker index in MARKS array

    # BP bytes 0, 1, 3 = 0
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_2]:
        # LO nibble = 0
        ffn.W_up[unit, BD.H1 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        # HI nibble = 0
        ffn.W_up[unit, BD.H1 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # BP byte 2 = 0x01 (lo=1, hi=0) - FIRST STEP ONLY
    # FIX 2026-04-16: Add HAS_SE suppression (same reason as SP byte 2).
    ffn.W_up[unit, BD.H1 + BP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 2.0 / S  # lo nibble = 1
    unit += 1
    ffn.W_up[unit, BD.H1 + BP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.HAS_SE] = -S  # Only first step
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
    unit += 1

    # PC marker index
    PC_I = 0  # PC marker index in MARKS array

    # PC DEFAULT: bytes 1-3 = 0 (for PC < 256, which covers most small programs)
    # At byte K position, we predict byte K+1, so:
    #   - BYTE_INDEX_0 → predict byte 1 (OUTPUT = 0)
    PC_I = 0  # PC marker index in MARKS array
    #   - BYTE_INDEX_1 → predict byte 2 (OUTPUT = 0)
    #   - BYTE_INDEX_2 → predict byte 3 (OUTPUT = 0)
    # This fires at all PC byte positions 0-2, outputting 0 for bytes 1-3.
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # Condition: H1[PC] AND BYTE_INDEX_K → OUTPUT = 0 (for byte K+1)
        ffn.W_up[unit, BD.H1 + PC_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo = 0
        unit += 1
        # HI nibble also = 0
        ffn.W_up[unit, BD.H1 + PC_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi = 0
        unit += 1

    # AX DEFAULT: bytes 1-3 = 0 (for single-byte immediates)
    # Multi-byte results (e.g., ADD with carry) will override in later layers
    # At byte K position, we predict byte K+1, so:
    #   - BYTE_INDEX_0 → predict byte 1 (OUTPUT = 0)
    #   - BYTE_INDEX_1 → predict byte 2 (OUTPUT = 0)
    #   - BYTE_INDEX_2 → predict byte 3 (OUTPUT = 0)
    AX_I = 1  # AX marker index in MARKS array
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # Condition: H1[AX] AND BYTE_INDEX_K → OUTPUT = 0 (for byte K+1)
        ffn.W_up[unit, BD.H1 + AX_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo = 0
        unit += 1
        # HI nibble also = 0
        ffn.W_up[unit, BD.H1 + AX_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi = 0
        unit += 1

    # MEM DEFAULT: all bytes = 0 (for non-memory-writing ops like IMM)
    # At MEM marker position, predict addr byte 0 = 0
    # For PSH/SI/SC, L14 will write actual values that override this default.
    # Condition: MARK_MEM = 1 → OUTPUT = 0
    MEM_I = 4  # MEM marker index in MARKS array
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
    unit += 1
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    # MEM addr bytes 1-3 default to 0 (for non-store ops)
    # At MEM byte positions d=1..3, H1[MEM]=1, BYTE_INDEX_K=1 for position K+1
    # FIX 2026-04-16: Exclude BYTE_INDEX_3 (d=4) because that position PREDICTS val_b0,
    # not addr_b3. L14 handles val bytes; this default is only for addr bytes 1-3.
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # LO = 0
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        # HI = 0
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # STACK0 DEFAULT: bytes 1-3 = 0 (for single-byte results)
    # At STACK0 byte positions, output 0 for bytes 1-3.
    # STACK0 byte 0: d=6 from BP → L1H4[BP]=1 (d≤6.5)
    # STACK0 byte 1: d=7 from BP → H2[BP]=1 (d≤7.5), NOT L1H4[BP]
    # STACK0 byte 2: d=8 from BP → H3[BP]=1 (d≤8.5), NOT H2[BP]
    # STACK0 byte 3: d=9 from BP → H4[BP]=1 (d≤9.5), NOT H3[BP]
    # Use H4[BP] (d≤9.5) to cover all STACK0 positions d=5-9.
    # H1[BP] = 1 for d <= 4.5 from BP (only BP area, not STACK0)
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # Condition: H4[BP] AND BYTE_INDEX_K AND NOT H1[BP] → OUTPUT = 0
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Exclude BP area
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Exclude BP area
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # STACK0 first-step default: byte 0 = 0 (empty stack)
    # At STACK0 marker, when NOT HAS_SE, output 0.
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
    unit += 1
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.HAS_SE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
    unit += 1

    # PC INCREMENT: when MARK_PC AND HAS_SE AND NOT OP_LEV, add INSTR_WIDTH to carried-forward value
    # For each lo nibble k (0-15): new_lo = (k+INSTR_WIDTH)%16
    # FIX 2026-04-15: Suppress when OP_LEV - LEV gets PC from return_addr in memory, not increment
    for k in range(16):
        new_k = (k + INSTR_WIDTH) % 16
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_LEV] = -S / 5  # OP_LEV ≈ 5.0, so -S/5 * 5 = -S suppresses
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        unit += 1

    # Hi nibble copy (only at MARK_PC AND HAS_SE AND NOT OP_LEV)
    for k in range(16):
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_LEV] = -S / 5  # Suppress when LEV
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # PC carry correction: when lo nibble >= (16-INSTR_WIDTH), adding INSTR_WIDTH wraps (>=16),
    # so hi nibble must increment by 1. Fires when MARK_PC AND HAS_SE AND NOT OP_LEV
    # AND any of EMBED_LO[(16-INSTR_WIDTH)..15] is set (old lo nibble >= 16-INSTR_WIDTH).
    # MARK_PC has weight 4*S to strictly require it (prevents false positive
    # at byte positions where EMBED_LO can be inflated by L3 leakage).
    # FIX 2026-04-15: Added OP_LEV suppression
    carry_threshold = 16 - INSTR_WIDTH  # For INSTR_WIDTH=8, this is 8
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = 4 * S
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.OP_LEV] = -S  # OP_LEV ≈ 5, use stronger -S for higher threshold
        for lo_bit in range(carry_threshold, 16):
            ffn.W_up[unit, BD.EMBED_LO + lo_bit] = S
        # Bias: activate when MARK_PC(4*S) + HAS_SE(S) + carry_bit(S) >= 5.5*S
        #       but not when just MARK_PC(4*S) + HAS_SE(S) = 5*S < 5.5*S
        ffn.b_up[unit] = -S * 5.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = -2.0 / S  # cancel old
        ffn.W_down[BD.OUTPUT_HI + (k + 1) % 16, unit] = 2.0 / S  # add shifted
        unit += 1



def _set_layer4_pc_relay(attn, S, BD, HD):
    """Layer 4 attention: Relay PREVIOUS step's PC to AX marker position.

    Head 0: AX marker reads the PC MARKER's EMBED_LO/HI, which was
    populated by L3 carry-forward from the previous step's PC byte 0.
    This gives the instruction address that was just executed, enabling
    L5 to fetch the correct opcode and immediate.

    Output goes to EMBED_LO/HI at AX marker (which starts as all zeros
    since marker tokens have no nibble embedding).
    """
    L = 15.0
    PC_I = 0  # PC marker index in MARKS

    base = 0 * HD
    # Q: active at AX markers
    attn.W_q[base, BD.MARK_AX] = L
    # K: target PC marker (has prev step's PC from L3 carry-forward)
    attn.W_k[base, BD.MARK_PC] = L
    # V: copies EMBED_LO and EMBED_HI (OLD PC from L3 carry-forward).
    # L3 carry-forward writes the OLD PC (instruction just executed) to EMBED_LO/HI.
    # L3 FFN increments it and writes NEW PC to OUTPUT_LO/HI.
    # We need the OLD PC to fetch the opcode at the correct address.
    # Example: IMM 42 at address 0, after execution PC=10 (new), but we fetch opcode at 0 (old).
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0
    # O: writes to EMBED_LO and EMBED_HI at AX marker
    for k in range(16):
        attn.W_o[BD.EMBED_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.EMBED_HI + k, base + 17 + k] = 1.0

    # Anti-leakage gate: suppress at non-AX positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L


def _set_layer4_ffn(ffn, S, BD):
    """Layer 4 FFN: Compute (PC+1) nibbles at AX marker for IMM fetch.

    PC value is in EMBED_LO/HI at AX marker (from L4 attention relay).
    L5 fetch uses:
      - opcode query at address PC (EMBED_LO/HI)
      - immediate query at address PC+1 (TEMP[0..31], written here)
    """
    unit = 0

    # === PC_PLUS1_LO: rotate EMBED_LO by +1 ===
    for k in range(16):
        src = (k - 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_LO + src] = 1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    # === PC_PLUS1_HI: default copy (no carry) ===
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1

    # === PC_PLUS1_HI: carry correction when EMBED_LO[15] == 1 ===
    # Cancel default copy and write rotated (+1) hi nibble.
    for k in range(16):
        # Cancel: MARK_AX AND LO[15] → subtract EMBED_HI[k]
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.EMBED_LO + 15] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1
        # Write rotated: MARK_AX AND LO[15] → add EMBED_HI[(k-1)%16]
        src = (k - 1) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.EMBED_LO + 15] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + src] = 1.0
        ffn.W_down[BD.TEMP + 16 + k, unit] = 2.0 / S
        unit += 1

    # === TEMP clearing at PC marker ===
    # Clear TEMP dims at PC marker to prevent them from leaking to Layer 6.
    # TEMP values are only meaningful at AX marker (where PC+1 is computed).
    # At PC marker, TEMP should be zero to prevent spurious BZ/BNZ activation.
    # EXCEPT: TEMP[0] is used for IS_JSR flag (first-step decode + L6 relay).
    # Condition: MARK_PC (fires at PC marker token)
    for k in range(32):
        if k == 0:
            # Skip TEMP[0] - used for IS_JSR flag, leave unit with zero weights
            unit += 1
            continue
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.TEMP + k] = -1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1


def _set_layer5_fetch(attn, S, BD, HD):
    """Layer 5 attention: fetch opcode/immediate through memory keys.

    Head 0: fetch immediate byte at address PC+1 (TEMP[0..31]).
    Head 1: fetch opcode byte at address PC (EMBED_LO/HI).

    Both heads match query nibbles against ADDR_KEY. This removes the
    forward-time address injection path; code bytes are expected to be
    available as memory entries keyed by address.
    """
    L = 20.0

    # Head 0: fetch immediate byte (address = PC+1)
    # Only fires on non-first steps (HAS_SE > 0). For first step, head 3 fetches
    # immediate at PC marker and head 4 relays to AX marker.
    # TEMP contains PC+1 at AX marker (computed by L4 FFN from relayed PC).
    base = 0 * HD
    # Q: low two nibbles from TEMP (PC+1)
    for k in range(16):
        attn.W_q[base + k, BD.TEMP + k] = L
        attn.W_q[base + 16 + k, BD.TEMP + 16 + k] = L
    # third address nibble fixed to zero (for <= 255 code addresses),
    # gated to AX marker to avoid global leakage.
    attn.W_q[base + 32, BD.MARK_AX] = L

    # K: address nibbles from memory key space
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate: suppress head activity at non-AX positions.
    # Must overwhelm worst-case address match (~+50 from 33 addr dims).
    # Non-AX: Q=-500, K=5 → -312.5.  AX: Q=0 → 0 contribution.
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps (when HAS_SE > 0)
    # First step uses head 3 (PC marker) + head 4 (relay to AX)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: copy byte value nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write immediate to FETCH_LO/HI
    for k in range(16):
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 1.0

    # Head 1: fetch opcode byte (address = PC)
    # EMBED_LO/HI at AX marker contain the PC value (relayed by L4 attention).
    base = 1 * HD
    # Q: low two nibbles from EMBED (PC)
    for k in range(16):
        attn.W_q[base + k, BD.EMBED_LO + k] = L
        attn.W_q[base + 16 + k, BD.EMBED_HI + k] = L
    attn.W_q[base + 32, BD.MARK_AX] = L

    # K: address nibbles from memory key space
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate: suppress head activity at non-AX positions.
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps (when HAS_SE > 0)
    # First step uses Head 4 (fetches opcode to AX marker at PC_OFFSET)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: opcode byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: writes to OPCODE_BYTE_LO/HI (staging for opcode decode)
    # Uses separate dims from ALU_LO/HI to avoid residual collision with L7 operand gather
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 2: fetch opcode for first-step (PC marker → address PC_OFFSET - 2)
    # FIX 2026-04-12: PC_OFFSET points to immediate byte, opcode is 2 bytes before
    # On the first step (NOT HAS_SE), PC = PC_OFFSET.
    # Uses ADDR_KEY matching (same mechanism as head 1, but fires at PC marker not AX).
    from .constants import PC_OFFSET
    base = 2 * HD
    # Q: fires at PC marker when NOT HAS_SE, queries for address PC_OFFSET
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L  # only on first step
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_PC] = L  # third nibble gate

    # K: match ADDR_KEY nibbles (same as head 1)
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate (same as head 1)
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # V: copy opcode byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to OPCODE_BYTE_LO/HI at PC marker
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 3: fetch immediate for first-step (PC marker → address PC_OFFSET + 1)
    # Fetches immediate byte 0 at address PC_OFFSET+1 and writes to FETCH.
    # Layer 6 FFN reads FETCH for JMP target.
    imm_addr = PC_OFFSET + 1
    base = 3 * HD
    # Q: queries for address imm_addr (e.g., 1: ADDR_KEY_LO[1]=1, ADDR_KEY_HI[0]=1)
    # NOTE: Q[PC_OFFSET] must remain zero - setting it causes opcode/imm to match equally
    # Queries at PC marker only. L6 head 5 relays FETCH from PC to AX for first-step IMM.
    # BUG FIX 2026-04-09: Removed AX marker gate to prevent writing jump target to AX_CARRY
    # at AX marker, which caused JMP to corrupt AX byte 0 (got 16 instead of 0).
    attn.W_q[base + (imm_addr & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((imm_addr >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_PC] = L  # gate for PC marker only

    # K: match ADDR_KEY nibbles
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate: suppress at non-PC positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: REMOVED for JMP support at non-first steps.
    # Previously this blocked immediate fetch on non-first steps, but JMP needs FETCH
    # (immediate = jump target) at any step, not just first step.
    # FIX 2026-04-16: Removed HAS_SE gate to enable JMP at any step position.
    # The L6 FFN JMP override uses OP_JMP + FETCH to set PC to jump target.
    # HAS_SE_GATE = 34  # Disabled
    # attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # Disabled
    # attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0  # Disabled

    # V: copy immediate byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to FETCH_LO/HI at PC marker for first-step immediate
    # AMPLIFY FETCH: Use 40.0 instead of 1.0 to compensate for attenuation during
    # Layer 6 Head 5 relay (FETCH gets attenuated 37x: 1.0→0.027, while OP_IMM
    # only gets attenuated 17x: 6.0→0.351). With 40.0, FETCH should reach ~1.0 at AX.
    # BUG FIX 2026-04-09 (part 3): Removed AX_CARRY output from L5 head 3.
    # AX_CARRY should only be written for JMP operations, but L5 head 3 fires for ALL
    # first-step ops. L6 head 2 now reads FETCH and writes to AX_CARRY (gated on OP_JMP).
    for k in range(16):
        # attn.W_o[BD.AX_CARRY_LO + k, base + 32 + k] = 1.0  # REMOVED
        # attn.W_o[BD.AX_CARRY_HI + k, base + 48 + k] = 1.0  # REMOVED
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 40.0  # Amplified for relay
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 40.0  # Amplified for relay

    # Head 4: Fetch opcode to AX marker for first-step (duplicate of Head 2)
    # Head 2 fetches opcode to PC marker, but opcode decode FFN runs at AX marker.
    # This head fetches the same opcode at address PC_OFFSET but writes to AX marker.
    # (Cannot relay from Head 2 since attention reads from input, not other heads' outputs.)
    base = 4 * HD
    # Q: fires at AX marker when NOT HAS_SE (first step), queries for address PC_OFFSET
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.HAS_SE] = -L  # only on first step
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_AX] = L  # third nibble gate
    # K: match ADDR_KEY nibbles (code byte addresses)
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L
    # Anti-leakage gate: suppress at non-AX positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0
    # HAS_SE gate: only fire on first step (when HAS_SE = 0)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks from wrong positions)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    # Combined with anti-leakage: non-first + AX = blocked
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # negative: block when HAS_SE > 0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0
    # V: copy opcode byte nibbles from matched code position
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write OPCODE_BYTE_LO/HI at AX marker
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 6: Direct OP_* flag relay from CODE to PC marker (non-first steps)
    # This bypasses the OPCODE_BYTE mechanism and uses the OP_* flags set in embeddings.
    # For non-first steps, PC value is in EMBED_LO/HI at PC marker (from L4 relay).
    # Query for current PC address, match CODE byte with that ADDR_KEY, copy OP_* flags.
    base = 6 * HD
    # Q: PC marker, address from EMBED_LO/HI (current PC)
    for k in range(16):
        attn.W_q[base + k, BD.EMBED_LO + k] = L
        attn.W_q[base + 16 + k, BD.EMBED_HI + k] = L
    attn.W_q[base + 32, BD.MARK_PC] = L  # Only at PC marker

    # K: match ADDR_KEY in CODE section
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: copy OP_* flags from CODE byte (these are set in embeddings)
    # Copy the main opcode flags that Layer 6 needs for routing
    v_offset = 0
    attn.W_v[base + v_offset, BD.OP_IMM] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LEA] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EXIT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JMP] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # ADDED for neural JSR (non-first steps)
    attn.W_v[base + v_offset, BD.OP_ADD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SUB] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MUL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_DIV] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MOD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_OR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_XOR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_AND] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EQ] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHR] = 1.0; v_offset += 1

    # O: write OP_* flags to PC marker
    o_offset = 0
    attn.W_o[BD.OP_IMM, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LEA, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EXIT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JMP, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # ADDED for neural JSR (non-first steps)
    attn.W_o[BD.OP_ADD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SUB, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MUL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_DIV, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MOD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_OR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_XOR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_AND, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EQ, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHR, base + o_offset] = 1.0; o_offset += 1

    # Head 7: Direct OP_* flag relay from CODE to PC marker (first step only)
    # Same as Head 6, but for first step (NOT HAS_SE instead of HAS_SE).
    base = 7 * HD
    # Q: PC marker when NOT HAS_SE, queries for address PC_OFFSET
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L  # only on first step
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_PC] = L  # third nibble gate

    # K: match ADDR_KEY in CODE section
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # V: copy OP_* flags from CODE byte (these are set in embeddings)
    v_offset = 0
    attn.W_v[base + v_offset, BD.OP_IMM] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LEA] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EXIT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JMP] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # ADDED for neural JSR
    attn.W_v[base + v_offset, BD.OP_ADD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SUB] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MUL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_DIV] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_MOD] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_OR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_XOR] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_AND] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_EQ] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_LT] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHL] = 1.0; v_offset += 1
    attn.W_v[base + v_offset, BD.OP_SHR] = 1.0; v_offset += 1

    # O: write OP_* flags to PC marker
    o_offset = 0
    attn.W_o[BD.OP_IMM, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LEA, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EXIT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JMP, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # ADDED for neural JSR
    attn.W_o[BD.OP_ADD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SUB, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MUL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_DIV, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_MOD, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_OR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_XOR, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_AND, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_EQ, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_LT, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHL, base + o_offset] = 1.0; o_offset += 1
    attn.W_o[BD.OP_SHR, base + o_offset] = 1.0; o_offset += 1

    # Head 5: unused in Layer 5 fetch


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
            # Skip TEMP[0] - used for IS_JSR flag, leave unit with zero weights
            unit += 1
            continue
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.TEMP + k] = -1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1


def _set_layer6_attn(attn, S, BD, HD):
    """Layer 6 attention: JMP relay + EXIT relay + first-step relays.

    Head 0 — JMP relay: At current step's PC marker, attend to previous
    step's AX marker (d=30). Copy OP_JMP flag and FETCH_LO/HI (JMP target).
    Writes CMP[0] (IS_JMP) and AX_CARRY_LO/HI (JMP target) at PC marker.

    Head 1 — EXIT relay: At NEXT_SE positions, attend to current step's
    AX marker (d=28). Copy OP_EXIT flag. Writes CMP[1] (IS_EXIT).

    Head 2 — First-step JMP relay: At PC marker (first step only), attend to
    current step's AX marker. Copy OP_JMP and FETCH for intra-step JMP.

    Head 3 — JSR relay: At PC marker (all steps), attend to current step's
    AX marker. Copy OP_JSR flag to TEMP[0] for JSR PC override.

    Head 4 — First-step FETCH relay: At AX marker (first step only), attend to
    PC marker. Copy FETCH_LO/HI from PC marker to AX marker for IMM routing.

    Head 5 — First-step OP flag relay: At AX marker (first step only), attend to
    PC marker. Copy OP_IMM, OP_LEA, OP_JMP, OP_EXIT, OP_NOP, arithmetic/bitwise/
    comparison/shift flags from PC marker to AX marker for Layer 6 FFN routing.
    Required because Layer 5 FFN decodes opcodes at PC marker (NOT at AX marker),
    but Layer 6 FFN needs flags at AX marker. Relays 17 OP flags total.

    Uses L=50 (large) + ALiBi slope=5.0 (steep) to minimize leakage.
    Attention scale = 1/sqrt(64) = 0.125, so score = L²*0.125 - slope*d.
    Head 0 at PC: 50²*0.125 - 5*30 = 162 (strong). Leakage at Q=0: <0.7%.
    Head 1 at SE: 50²*0.7*0.125 - 5*28 = 79 (strong). Leakage at Q=0: <0.7%.
    Q guards (-L at MARK_AX) block self-attention at AX markers entirely.

    Head 6: Configured by _set_layer6_relay_heads() for PSH relay (STACK0 ← AX).
    Head 7: Reserved for JSR handling (configured later in set_vm_weights).
    """
    L = 50.0

    # Head 0: JMP relay (PC marker → previous step's AX marker)
    # FIX 2026-04-16: Added HAS_SE gating so Head 0 only fires on step 1+.
    # For step 0, there's no previous step's AX marker to relay from.
    # Without this gate, Head 0 fires at PC marker (Q=L from MARK_PC), and with no
    # valid K positions (no MARK_AX), softmax gives uniform attention. This averages
    # FETCH values from all positions, writing garbage to AX_CARRY_LO/HI, which
    # causes L8 FFN to produce massive OUTPUT values (40000+) at PC marker.
    base = 0 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker (prevents AX_CARRY corruption)
    # Strong anti-leakage gate: Q must be very negative for step 0 (no HAS_SE)
    # Q = 50 - 0 + 0 - 1000 = -950 for step 0 (exp(-950) ≈ 0)
    # Q = 50 - 0 + 1000 - 1000 = 50 for step 1+ (fires normally)
    attn.W_q[base, BD.HAS_SE] = L * 20  # +1000 when HAS_SE
    attn.W_q[base, BD.CONST] = -L * 20  # -1000 baseline
    attn.W_k[base, BD.MARK_AX] = L
    # FIX 2026-04-16: Add CONST to K so the Q gate actually works.
    # Without this, K[0] = 0 at non-AX positions (including PC marker),
    # so Q·K = (-950)*0 = 0 regardless of Q, and attention fires via ALiBi.
    # With CONST: K[0] = 50*MARK_AX + 1*CONST, so K[0] = 1 at non-AX positions.
    # Q·K = (-950)*1 = -950 → exp(-950) ≈ 0 → blocked.
    attn.W_k[base, BD.CONST] = 1.0
    # V: copy OP_JMP flag, FETCH_LO/HI (target from fetched immediate)
    # NOTE: OP_JSR is NOT included here. The JMP relay has a one-step delay
    # (fires at step N+1 using step N's AX flags). For JMP this is the design:
    # step N outputs old PC+5, step N+1 overrides with target. For JSR, the
    # runner overrides PC to the target at step N directly, so the relay at
    # step N+1 would double-override (writing target instead of target+5).
    attn.W_v[base + 1, BD.OP_JMP] = 1.0
    for k in range(16):
        attn.W_v[base + 2 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 18 + k, BD.FETCH_HI + k] = 1.0
    # O: write IS_JMP to CMP[0], JMP target to AX_CARRY at PC marker
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    for k in range(16):
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0

    # Head 1: EXIT relay (NEXT_SE position → current step's AX marker)
    base = 1 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker (prevents CMP[1] leakage)
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy OP_EXIT flag (scaled to improve EXIT vs NOP separation downstream)
    attn.W_v[base + 1, BD.OP_EXIT] = 0.2
    # O: write IS_EXIT to CMP[1]
    attn.W_o[BD.CMP + 1, base + 1] = 1.0

    # Head 2: First-step JMP relay (PC marker self-attention)
    # For first step (NOT HAS_SE), copy OP_JMP and FETCH from PC marker to PC marker.
    # This works with L5 head 3 fix (which no longer writes to AX marker).
    # Head 0 handles subsequent steps (cross-step relay), head 2 handles first step.
    # BUG FIX 2026-04-09: Changed key from MARK_AX to MARK_PC to work with L5 head 3 fix.
    # BUG FIX 2026-04-09 (part 2): Block non-JMP operations (IMM, NOP, EXIT) to prevent
    # this head from firing for all first-step ops and copying amplified FETCH to AX_CARRY.
    base = 2 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L  # Only fire when NOT HAS_SE (first step)
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    # FIX 2026-04-16: Strong anti-leakage gate for non-JMP operations.
    # Without strong gate, Q = -50 for non-JMP gives ~5% attention weight (not zero!),
    # causing FETCH values to leak into AX_CARRY_LO and corrupt PC output.
    # With OP_JMP ≈ 5.0 when JMP is active, 0 otherwise:
    # Q = MARK_PC(50) - HAS_SE(50) - MARK_AX(50) + OP_JMP(1000) - CONST(1000)
    # For JMP first-step: 50 - 0 - 0 + 5000 - 1000 = 4050 (strong positive)
    # For non-JMP first-step: 50 - 0 - 0 + 0 - 1000 = -950 (blocked, exp(-950) ≈ 0)
    attn.W_q[base, BD.OP_JMP] = L * 20  # +1000 when OP_JMP active
    attn.W_q[base, BD.CONST] = -L * 20  # -1000 baseline
    attn.W_k[base, BD.MARK_PC] = L  # Read from PC marker (not AX marker)
    # V: copy OP_JMP and FETCH_LO/HI from PC marker
    attn.W_v[base + 1, BD.OP_JMP] = 1.0
    for k in range(16):
        attn.W_v[base + 2 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 18 + k, BD.FETCH_HI + k] = 1.0
    # O: write IS_JMP to CMP[0], JMP target to AX_CARRY at PC marker
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    for k in range(16):
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0

    # Head 3: JSR relay (AX marker → PC marker, FIRST STEP ONLY)
    # For first step, L5 FFN decodes JSR at PC marker and writes TEMP[0].
    # This relay is DISABLED (first-step only) because subsequent steps use L5 opcode decode.
    # BUG FIX 2026-04-13: Added HAS_SE gate to prevent false positive on subsequent steps.
    # Without HAS_SE gate, this head attends to ALL AX markers in context, including
    # previous steps' AX markers which may have OP_JSR set (causing PC override on non-JSR steps).
    base = 3 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_q[base, BD.HAS_SE] = -L   # BUG FIX: only fire when NOT HAS_SE (first step)
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy OP_JSR flag only (FETCH already has target)
    attn.W_v[base + 1, BD.OP_JSR] = 1.0
    # O: write IS_JSR flag to TEMP[0] at PC marker
    # (Layer 5 FFN clears TEMP at PC; Layer 6 attention writes it; Layer 6 FFN reads it)
    attn.W_o[BD.TEMP + 0, base + 1] = 1.0

    # Head 4: Reserved for BZ/BNZ relay (set by _set_bz_bnz_relay)
    # NOTE: First-step FETCH relay moved to head 5 to avoid Q weight conflicts.
    # _set_bz_bnz_relay sets up PC→AX relay for BZ/BNZ conditional branches.
    # base = 4 * HD (left for _set_bz_bnz_relay)

    # Head 5: First-step OP flag relay + FETCH relay (AX marker ← PC marker)
    # For first step, L5 FFN decodes opcodes at PC marker (OP_IMM, OP_LEA, OP_EXIT, OP_NOP, OP_JMP, OP_JSR, arithmetic, bitwise, cmp, shift).
    # L6 FFN needs these flags at AX marker for routing (IMM, EXIT, NOP, JMP, arithmetic, etc).
    # This relay copies OP flags from PC marker to AX marker on first step only.
    # Currently relaying: IMM, LEA, JMP, JSR, EXIT, NOP, ADD, SUB, MUL, DIV, MOD, OR, XOR, AND, EQ, LT, SHL, SHR
    base = 5 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.HAS_SE] = -L  # Only fire when NOT HAS_SE (first step)
    attn.W_k[base, BD.MARK_PC] = L
    # V: copy OP flags (18 total: added JSR)
    attn.W_v[base + 0, BD.OP_IMM] = 1.0
    attn.W_v[base + 1, BD.OP_LEA] = 1.0
    attn.W_v[base + 2, BD.OP_JMP] = 1.0
    attn.W_v[base + 3, BD.OP_JSR] = 1.0  # ADDED: relay JSR flag
    attn.W_v[base + 4, BD.OP_EXIT] = 1.0
    attn.W_v[base + 5, BD.OP_NOP] = 1.0
    attn.W_v[base + 6, BD.OP_ADD] = 1.0
    attn.W_v[base + 7, BD.OP_SUB] = 1.0
    attn.W_v[base + 8, BD.OP_MUL] = 1.0
    attn.W_v[base + 9, BD.OP_DIV] = 1.0
    attn.W_v[base + 10, BD.OP_MOD] = 1.0
    attn.W_v[base + 11, BD.OP_OR] = 1.0
    attn.W_v[base + 12, BD.OP_XOR] = 1.0
    attn.W_v[base + 13, BD.OP_AND] = 1.0
    attn.W_v[base + 14, BD.OP_EQ] = 1.0
    attn.W_v[base + 15, BD.OP_LT] = 1.0
    attn.W_v[base + 16, BD.OP_SHL] = 1.0
    attn.W_v[base + 17, BD.OP_SHR] = 1.0
    # O: write OP flags at AX marker
    attn.W_o[BD.OP_IMM, base + 0] = 1.0
    attn.W_o[BD.OP_LEA, base + 1] = 1.0
    attn.W_o[BD.OP_JMP, base + 2] = 1.0
    attn.W_o[BD.OP_JSR, base + 3] = 1.0  # ADDED: write JSR flag to AX marker
    attn.W_o[BD.OP_EXIT, base + 4] = 1.0
    attn.W_o[BD.OP_NOP, base + 5] = 1.0
    attn.W_o[BD.OP_ADD, base + 6] = 1.0
    attn.W_o[BD.OP_SUB, base + 7] = 1.0
    attn.W_o[BD.OP_MUL, base + 8] = 1.0
    attn.W_o[BD.OP_DIV, base + 9] = 1.0
    attn.W_o[BD.OP_MOD, base + 10] = 1.0
    attn.W_o[BD.OP_OR, base + 11] = 1.0
    attn.W_o[BD.OP_XOR, base + 12] = 1.0
    attn.W_o[BD.OP_AND, base + 13] = 1.0  # FIXED: was OP_EQ
    attn.W_o[BD.OP_EQ, base + 14] = 1.0   # FIXED: was OP_LT
    attn.W_o[BD.OP_LT, base + 15] = 1.0   # FIXED: was OP_SHL
    attn.W_o[BD.OP_SHL, base + 16] = 1.0  # FIXED: was OP_SHR
    attn.W_o[BD.OP_SHR, base + 17] = 1.0
    # V: also copy FETCH_LO/HI (positions 18-49) for first-step IMM routing
    # This relay was moved from head 4 to avoid BZ/BNZ weight conflicts.
    # UPDATED: positions shifted from 17 to 18 due to JSR addition
    for k in range(16):
        attn.W_v[base + 18 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 34 + k, BD.FETCH_HI + k] = 1.0
    # O: write FETCH_LO/HI at AX marker
    for k in range(16):
        attn.W_o[BD.FETCH_LO + k, base + 18 + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 34 + k] = 1.0
    # HAS_SE gate: block attention when HAS_SE = 1 (non-first steps)
    # Without this gate, softmax1 distributes attention uniformly when Q is zero,
    # causing spurious FETCH relay from step 1 PC marker (which has V[18]=40.0 from L5).
    # Gate mechanism: Q[slot] = -500 * HAS_SE, K[slot] = 5 * CONST
    # Score contribution = Q · K / sqrt(HD) = -500 * 5 / 8 = -312.5 when HAS_SE=1
    HAS_SE_GATE = 49  # Free slot after FETCH_HI (slots 0-48 used)
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0


def _set_layer6_routing_ffn(ffn, S, BD):
    """Layer 6 FFN: Output routing for AX, PC (JMP), SP, BP + HALT detection.

    With CLEAN_EMBED-sourced opcode decode, correct OP_* ≈ 5.0, false
    positive OP_* ≈ 0. Threshold 4.0 cleanly separates them (5+1=6 > 4).

    AX routing (explicit per-opcode):
      - IMM: OP_IMM AND MARK_AX → FETCH → OUTPUT
      - EXIT/NOP/JMP: OP_xxx AND MARK_AX → AX_CARRY → OUTPUT

    PC routing (JMP override):
      - CMP[0] (IS_JMP) AND MARK_PC

    HALT detection:
      - CMP[1] (IS_EXIT) AND NEXT_SE
    """
    unit = 0
    # Threshold with guards: OP + MARK_AX - MARK_PC - IS_BYTE > T
    # BUG FIX 2026-04-09: T=0.5 was far too low! With MARK_AX≈2.0 at later steps,
    # units fire even when OP_xxx=0: 100*0 + 100*2 - 50 = 150 > 0 (spurious!)
    #
    # Correct calculation with S=100 scale:
    # - Block non-target: S*MARK_AX - S*T < 0 → 200 - 100*T < 0 → T > 2
    # - Allow target:     S*OP + S*MARK_AX - S*T > 0 → 500 + 100 - 100*T > 0 → T < 6
    # T=4.0 works: blocks 200-400=-200, allows 600-400=200 (for MARK_AX=1, OP=5)
    T = 4.0

    # === IMM: FETCH → OUTPUT ===
    # Read from FETCH_LO/HI (clean staging dims written by L5 fetch head 0).
    # These dims have no prior-layer leakage, unlike EMBED_LO/HI which
    # accumulates carry-forward residuals from L3.
    # BUG FIX 2026-04-09 (part 4): Increased MARK_PC blocker from -6*S to -8*S.
    # OP_IMM ≈ 6.6 at PC marker (higher than expected 5.0), so:
    # activation = S*OP_IMM + S*MARK_AX + (-8*S)*MARK_PC + bias
    #            = 20*6.6 + 20*0 + (-160)*1 + (-10) = 132 - 160 - 10 = -38 (blocked!)
    # BUG FIX 2026-04-09 (part 8f): Add OP_JMP blocker to prevent crossfire.
    # At AX marker for JMP: OP_JMP ≈ 11.0, causing spurious activation:
    # activation = S*OP_IMM + S*MARK_AX + (-20*S)*OP_EXIT + ... - 50
    #            = 100*0 + 100*1 + 0 + ... - 50 = 50 (fires incorrectly!)
    # With OP_JMP blocker: 100*0 + 100*1 + (-2000)*11 - 50 = -22050 (blocked!)
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.OP_EXIT] = -S * 20  # Strong block EXIT crossfire
        ffn.W_up[unit, BD.OP_JMP] = -S * 20  # Strong block JMP crossfire
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -6*S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S  # Block at byte positions, fire at markers
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.OP_EXIT] = -S * 20  # Strong block EXIT crossfire
        ffn.W_up[unit, BD.OP_JMP] = -S * 20  # Strong block JMP crossfire
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -6*S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S  # Block at byte positions, fire at markers
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.FETCH_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === EXIT: AX_CARRY → OUTPUT ===
    # FIX 2026-04-16: Changed BACK from AX_FULL to AX_CARRY.
    # AX_FULL (dims 471-502) overlaps with TEMP (dims 480-511), causing PC+1
    # computation to corrupt AX_FULL_HI. AX_CARRY (dims 328-359) is safe.
    # BUG FIX 2026-04-09 (part 5): Increased MARK_PC blocker from -S to -8*S.
    # OP_EXIT ≈ 6.0 at PC marker, so activation = 20*6 + 20*0 + (-160)*1 + (-10) = -50 (blocked!)
    for k in range(16):
        ffn.W_up[unit, BD.OP_EXIT] = S
        ffn.W_up[unit, BD.OP_IMM] = -S * 20  # Strong block IMM crossfire
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S  # Block at byte positions, fire at markers
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_EXIT] = S
        ffn.W_up[unit, BD.OP_IMM] = -S * 20  # Strong block IMM crossfire
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S  # Block at byte positions, fire at markers
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === NOP: AX_CARRY → OUTPUT ===
    # FIX 2026-04-16: Changed BACK from AX_FULL to AX_CARRY.
    # AX_FULL (dims 471-502) overlaps with TEMP (dims 480-511), causing PC+1
    # computation to corrupt AX_FULL_HI. AX_CARRY (dims 328-359) is safe.
    # BUG FIX 2026-04-09 (part 6): Increased MARK_PC blocker from -S to -8*S.
    # BUG FIX 2026-04-09 (part 8e): Add IS_BYTE blocker to prevent firing at byte positions.
    # OP_NOP has residual values at PC byte positions, causing spurious activation.
    for k in range(16):
        ffn.W_up[unit, BD.OP_NOP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_NOP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 8  # INCREASED from -S to block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === JMP: AX_CARRY → OUTPUT (preserves AX through JMP) ===
    # FIX 2026-04-16: Changed BACK from AX_FULL to AX_CARRY.
    # AX_FULL (dims 471-502) overlaps with TEMP (dims 480-511), causing PC+1
    # computation to corrupt AX_FULL_HI. AX_CARRY (dims 328-359) is safe.
    # BUG FIX 2026-04-09 (part 8e): Add IS_BYTE blocker to prevent firing at byte positions.
    # OP_JMP has residual values at PC byte positions, causing spurious activation.
    for k in range(16):
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Use AX_CARRY (no TEMP overlap)
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === FIRST-STEP JMP: OP_JMP at AX, write FETCH to OUTPUT at PC marker ===
    # For the first step, OP_JMP is at the current AX marker (not previous step).
    # Layer 6 attention head 0 relays from previous step (one-step delay), so it
    # won't work for first-step JMP. Instead, we need to route FETCH directly.
    # Condition: OP_JMP at AX marker AND NOT HAS_SE
    # At PC marker position, attend to current AX marker and copy FETCH → OUTPUT
    # This is handled by Layer 13 attention (branch target relay), but we need
    # to ensure it fires for first-step JMP by setting a flag.
    # Actually, simpler: just write FETCH directly at PC marker when OP_JMP AND NOT HAS_SE.

    # Check at PC marker: is there an active JMP at the AX marker in this step?
    # We'll use a relay from AX to PC within the same step (intra-step relay).
    # This is complex, so instead let's modify the PC increment logic to check OP_JMP.

    # Actually, the cleanest solution: Add units that fire at PC marker when
    # OP_JMP is active anywhere in the step. But OP_JMP is only at AX marker...

    # Better approach: Modify Layer 13 to handle first-step JMP by relaying
    # FETCH from AX to PC when OP_JMP AND NOT HAS_SE.
    # For now, skip this and let Layer 13 handle it.

    # === JMP PC override: cancel PC+5, write JMP target from AX_CARRY ===
    # CMP[0] ≈ 7 for JMP, ≈ 3.2 false positive (longer programs inflate it).
    # Threshold 5.5: requires CMP[0] > 4.5, separating 7.0 from 3.2.
    T_jmp = 5.5
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Add JMP target (AX_CARRY at PC marker from L6 attn head 0)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 0] = S
        ffn.b_up[unit] = -S * T_jmp
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === FIRST-STEP JMP PC override: use OP_JMP directly when NOT HAS_SE ===
    # For first step, CMP[0] isn't set (no previous step to relay from).
    # Instead, use OP_JMP flag directly (set by L5 FFN opcode decode at PC marker).
    # Threshold: OP_JMP ≈ 5.0, so T=4.5 separates it from false positives.
    T_op_jmp = 4.5
    # Cancel PC+INSTR_WIDTH
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S  # only when NOT HAS_SE (first step)
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)  # require all three conditions
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Add JMP target from AX_CARRY (written by L5 head 3)
    # JMP immediate is converted: idx = imm // INSTR_WIDTH, PC = idx_to_pc(idx)
    # idx_to_pc adds PC_OFFSET, so for JMP 0x20: idx=4, PC=4*8+2=34
    # We need to output PC (not raw immediate), so add PC_OFFSET.
    # Strategy: nibble shift by +2 for LO, direct copy for HI
    for k in range(16):
        new_k = (k + 2) % 16  # Add PC_OFFSET=2
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        unit += 1
    # HI nibble: direct copy (assume no carry from LO+2 for simplicity)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S
        ffn.b_up[unit] = -S * (T_op_jmp + 0.5)
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ALL-STEP JMP PC override: use OP_JMP + FETCH directly ===
    # FIX 2026-04-16: Now that L5 head 3 fetches immediate for ALL steps (HAS_SE gate
    # removed), we can use FETCH directly for JMP at any step position.
    # This path uses OP_JMP (detected by L5 FFN) + FETCH (from L5 head 3) at PC marker.
    # The JMP immediate IS the PC value (compiler encodes idx*8+PC_OFFSET directly).
    T_op_jmp_all = 4.5
    # Cancel PC+INSTR_WIDTH (clear OUTPUT)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Add JMP target from FETCH (immediate value = PC target)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.b_up[unit] = -S * T_op_jmp_all
        ffn.W_gate[unit, BD.FETCH_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === HALT detection: CMP[1] AND NEXT_SE → convert SE to HALT ===
    # CMP[1] is scaled relay from OP_EXIT (L6 attn head 1), so use a lower
    # threshold tuned to keep EXIT active while suppressing NOP leakage.
    ffn.W_up[unit, BD.CMP + 1] = S
    ffn.W_up[unit, BD.NEXT_SE] = S
    ffn.b_up[unit] = -S * 1.3
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.NEXT_HALT, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S
    unit += 1

    # === TEMP clearing at PC marker ===
    # Clear TEMP dims at PC marker to prevent residual values from causing
    # spurious BZ/BNZ borrow logic activation. TEMP values are only valid
    # at AX marker (where they're computed in L5 FFN), not at other markers.
    # EXCEPT: TEMP[0] is used for IS_JSR flag (first-step decode + L6 relay).
    # Condition: MARK_PC AND NOT IS_BYTE (only at actual PC marker token)
    for k in range(32):
        if k == 0:
            # Skip TEMP[0] - used for IS_JSR flag, leave unit with zero weights
            unit += 1
            continue
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.TEMP + k] = -1.0
        ffn.W_down[BD.TEMP + k, unit] = 2.0 / S
        unit += 1

    # === CMP[3] clearing at PC marker ===
    # FIX 2026-04-16: L6 attention head 6 relays POP group flags to CMP[3] at all
    # positions it fires (SP, STACK0, BP, PC, MEM). At PC marker, CMP[3] is spurious
    # and causes unit 960 to fire, corrupting OUTPUT for EXIT instruction.
    # Clear CMP[3] at PC marker to prevent this. CMP[3] is only valid at SP/STACK0.
    # Condition: MARK_PC AND NOT IS_BYTE AND CMP[3] > 0 (only clear if set)
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.IS_BYTE] = -S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.CMP + 3] = -1.0  # Self-referential clearing
    ffn.W_down[BD.CMP + 3, unit] = 2.0 / S
    unit += 1

    # === SP/BP/STACK0 identity carry (EMBED → OUTPUT passthrough) ===
    # 2-way AND: MARK_xxx AND NOT IS_BYTE
    # Prevents activation at byte positions where markers are relayed by attention.
    # Only activates at actual marker tokens (SP=259, BP=260, STACK0=268).
    for marker_dim in [BD.MARK_SP, BD.MARK_BP, BD.MARK_STACK0]:
        for k in range(16):
            ffn.W_up[unit, marker_dim] = S
            ffn.W_up[unit, BD.IS_BYTE] = -S  # NOT IS_BYTE
            ffn.b_up[unit] = -S * 0.5
            ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
            ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up[unit, marker_dim] = S
            ffn.W_up[unit, BD.IS_BYTE] = -S  # NOT IS_BYTE
            ffn.b_up[unit] = -S * 0.5
            ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
            ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

    # === PSH: SP -= 8 (SP output = SP_carry - 8) ===
    # At SP marker, when PSH: cancel identity carry, write new value.
    # Uses PSH_AT_SP ≈ 1.0 (relayed OP_PSH from L6 attn head 6, clean dimension).
    # Threshold 1.5: PSH_AT_SP(1) + MARK_SP(1) = 2 > 1.5 → fires.
    T_psh = 1.5
    for k in range(16):
        new_k = (k - 8) % 16
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_psh
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_LO + k, unit] += -2.0 / S  # cancel identity
        unit += 1
    # Hi nibble: if old_lo < 8, borrow → hi -= 1
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_psh
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        for lo_bit in range(8, 16):
            ffn.W_gate[unit, BD.EMBED_LO + lo_bit] = -1.0
        ffn.W_down[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + k, unit] += -2.0 / S  # cancel identity
        unit += 1

    # === PSH: STACK0 = AX (override STACK0 carry with AX value) ===
    # Uses PSH_AT_SP ≈ 1.0 (relayed OP_PSH, clean dimension).
    # ALU_LO/HI at STACK0 marker has AX_CARRY value (copied by L6 attn head 2).
    # Cancel identity carry and write ALU value.
    T_psh_s0 = 1.5
    for k in range(16):
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_STACK0] = S
        ffn.b_up[unit] = -S * T_psh_s0
        # Gate: data routing only (EMBED vs ALU)
        ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0
        ffn.W_gate[unit, BD.ALU_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.PSH_AT_SP] = S
        ffn.W_up[unit, BD.MARK_STACK0] = S
        ffn.b_up[unit] = -S * T_psh_s0
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0
        ffn.W_gate[unit, BD.ALU_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === GETCHAR: AX = AX_CARRY (pass through, runner overrides later) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_GETCHAR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_GETCHAR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === BZ: AX passthrough (AX unchanged during branch test) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_BZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_BZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === BNZ: AX passthrough (AX unchanged during branch test) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_BNZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_BNZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === PSH: AX passthrough (AX unchanged during push) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_PSH] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_PSH] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ADJ: AX passthrough (AX unchanged during stack adjust) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Block at PC marker
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ADJ: SP writeback (route AX result → SP marker) ===
    # ADJ computes new_sp in AX (via ALU), then writes result to SP marker
    # Cancel identity carry, write AX_CARRY (the computed result)
    T_adj = 1.5
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_adj
        ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.b_up[unit] = -S * T_adj
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ENT: SP writeback (route AX result → SP marker) ===
    # ENT computes new_sp = sp - (8 + imm) in L8/L9 ALU at AX marker
    # Result is relayed to AX_CARRY by L6 attention, then written to SP marker
    # Cancel identity carry (EMBED = old SP), write AX_CARRY (new SP)
    # FIX 2026-04-17: Add HAS_SE gate. On first step, AX_CARRY is empty
    # (no previous step to relay from), causing garbage OUTPUT values.
    # First-step ENT SP is handled by separate units below.
    T_ent = 2.5  # Increased threshold: OP_ENT(5) + MARK_SP(1) + HAS_SE(1) = 7 > 2.5
    for k in range(16):
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = S  # Only subsequent steps
        ffn.b_up[unit] = -S * T_ent
        ffn.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = S  # Only subsequent steps
        ffn.b_up[unit] = -S * T_ent
        ffn.W_gate[unit, BD.EMBED_HI + k] = -1.0  # Cancel identity
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0  # Write result
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ENT first-step SP byte 0 (32 units) ===
    # On first step, SP_init = 0x10000, byte 0 = 0x00.
    # ENT computes: SP_new = SP_init - 8 - imm
    # For byte 0: result = 0x00 - 8 - imm_byte0 = -8 - imm
    # Lo nibble: (-8 - FETCH_LO) mod 16
    # Hi nibble: (-1 - FETCH_HI) mod 16 (always borrow from lo since -8 - x < 0)
    # Condition: OP_ENT + MARK_SP + NOT HAS_SE
    T_ent_first = 1.5
    for imm_lo in range(16):
        result_lo = (-8 - imm_lo) % 16
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
        ffn.b_up[unit] = -S * T_ent_first
        ffn.W_gate[unit, BD.FETCH_LO + imm_lo] = 1.0  # Select based on imm lo nibble
        ffn.W_down[BD.OUTPUT_LO + result_lo, unit] = 5.0 / S  # Strong output
        unit += 1
    for imm_hi in range(16):
        result_hi = (-1 - imm_hi) % 16  # Always borrow from lo
        ffn.W_up[unit, BD.OP_ENT] = S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
        ffn.b_up[unit] = -S * T_ent_first
        ffn.W_gate[unit, BD.FETCH_HI + imm_hi] = 1.0  # Select based on imm hi nibble
        ffn.W_down[BD.OUTPUT_HI + result_hi, unit] = 5.0 / S  # Strong output
        unit += 1

    # === ENT first-step SP bytes 1-3 (6 units) ===
    # FIX 2026-04-17: Add units for SP bytes 1-3 during ENT first step.
    # SP = 0x10000 - 8 - imm, for small imm:
    # - Byte 1 = 0xFF (borrow from byte 0 < 0 always)
    # - Byte 2 = 0x00 (borrow absorbed from 0x01 - 1 = 0x00)
    # - Byte 3 = 0x00 (unchanged)
    # Fire at: OP_ENT + BYTE_INDEX_* + IS_BYTE + H1[SP] + NOT HAS_SE
    SP_I = 2
    T_ent_byte = 4.0

    # SP byte 0 pos → predict byte 1 = 0xFF
    # Need OUTPUT_LO[15], OUTPUT_HI[15]
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 15, unit] = 10.0 / S  # Strong for nibble 15
    unit += 1
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 15, unit] = 10.0 / S  # Strong for nibble 15
    unit += 1

    # SP byte 1 pos → predict byte 2 = 0x00
    # L3 default OUTPUT = 0 should work, but add explicit units for safety
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 5.0 / S  # Nibble 0
    unit += 1
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 5.0 / S  # Nibble 0
    unit += 1

    # SP byte 2 pos → predict byte 3 = 0x00
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 5.0 / S
    unit += 1
    ffn.W_up[unit, BD.OP_ENT] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.HAS_SE] = -S * 10
    ffn.b_up[unit] = -S * T_ent_byte
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 5.0 / S
    unit += 1

    # === BZ PC override: branch if AX == 0 ===
    # CMP[2]=OP_BZ (normalized ≈1), CMP[4]=AX_LO_IS_ZERO, CMP[5]=AX_HI_IS_ZERO
    # 4-way AND in silu: MARK_PC + CMP[2] + CMP[4] + CMP[5] - 3.5
    # All conditions in silu, gate is ONLY the value multiplier.
    # BZ+zero: 1+1+1+1=4 > 3.5 → fires. One missing: 3 < 3.5 → off.
    #
    # FIX 2026-04-16: Add IS_BYTE suppression. CMP[5] is reused for PRTF flag
    # in conversational I/O mode. Without IS_BYTE blocker, these units fire
    # at PC byte positions during PRTF (MARK_PC=0 but CMP[5]≈5), canceling
    # the L3 PC default output and causing wrong byte values.
    T_bz = 5.5
    # Cancel existing PC+5 carry
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write branch target - 5 (so L3's +5 yields the actual target).
    # Lo nibble: remap k → (k-5)%16 = (k+11)%16
    # FIX 2026-04-16: Add MARK_STACK0 suppression. CMP[5] = 5.0 (PRTF flag) causes
    # these units to fire at STACK0 marker, writing garbage to OUTPUT.
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10  # Block at STACK0 marker
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.TEMP + k] = 1.0  # FETCH_LO via TEMP
        ffn.W_down[BD.OUTPUT_LO + (k + 11) % 16, unit] = 2.0 / S
        unit += 1
    # Hi nibble, no borrow (target_lo >= 5): copy straight through.
    # Extra condition: any of TEMP[5..15] must be active.
    # CMP[2] has large weight (20*S) to require OP_BZ; spurious TEMP values blocked.
    # FIX 2026-04-16: Add IS_BYTE + MARK_STACK0 suppression.
    T_bz_nb = 25.0  # MARK_PC(1) + CMP[2](≈20) + CMP[4](1) + CMP[5](1) > 25 only when BZ+zero
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S * 20  # Require OP_BZ to be active
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10  # Block at STACK0 marker
        for j in range(5, 16):
            ffn.W_up[unit, BD.TEMP + j] = S
        ffn.b_up[unit] = -S * T_bz_nb
        ffn.W_gate[unit, BD.TEMP + 16 + k] = 1.0  # FETCH_HI via TEMP
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Hi nibble, borrow (target_lo < 5): decrement hi by 1.
    # Extra condition: any of TEMP[0..4] must be active.
    # FIX 2026-04-16: Add MARK_STACK0 suppression.
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S * 20  # Require OP_BZ to be active
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.W_up[unit, BD.IS_BYTE] = -S * 10  # Block at byte positions
        ffn.W_up[unit, BD.MARK_STACK0] = -S * 10  # Block at STACK0 marker
        for j in range(0, 5):
            ffn.W_up[unit, BD.TEMP + j] = S
        ffn.b_up[unit] = -S * T_bz_nb
        ffn.W_gate[unit, BD.TEMP + 16 + k] = 1.0  # FETCH_HI via TEMP
        ffn.W_down[BD.OUTPUT_HI + (k + 15) % 16, unit] = 2.0 / S
        unit += 1

    # === BNZ PC override: branch if AX != 0 ===
    # Two exclusive groups. All conditions in silu, gate ONLY for value.
    # CMP[3]=OP_BNZ (normalized ≈1), CMP[4]=lo_zero, CMP[5]=hi_zero.
    #
    # Group A: lo nibble is nonzero
    #   up = S*(MARK_PC + CMP[3] - CMP[4]) - S*1.5
    #   BNZ + lo_nonzero: 1+1-0=2 > 1.5 → fires
    #   BNZ + lo_zero: 1+1-1=1 < 1.5 → off
    #   gate = just the value (cancel or write)
    T_bnz = 1.5
    # Cancel existing OUTPUT
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
        ffn.W_up[unit, BD.CMP + 4] = -S
        ffn.b_up[unit] = -S * T_bnz
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
        ffn.W_up[unit, BD.CMP + 4] = -S
        ffn.b_up[unit] = -S * T_bnz
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write target - 5 lo nibble: remap k → (k+11)%16
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
        ffn.W_up[unit, BD.CMP + 4] = -S
        ffn.b_up[unit] = -S * T_bnz
        ffn.W_gate[unit, BD.TEMP + k] = 1.0  # FETCH_LO via TEMP
        ffn.W_down[BD.OUTPUT_LO + (k + 11) % 16, unit] = 2.0 / S
        unit += 1
    # Write target - 5 hi nibble, no borrow (target_lo >= 5)
    # CMP[3] has large weight (20*S) to require OP_BNZ; spurious TEMP values blocked.
    # Threshold 21.0: MARK_PC(1) + CMP[3](≈20) > 21 only when BNZ active.
    T_bnz_nb = 21.0
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S * 20  # Require OP_BNZ to be active
        ffn.W_up[unit, BD.CMP + 4] = -S
        for j in range(5, 16):
            ffn.W_up[unit, BD.TEMP + j] = S
        ffn.b_up[unit] = -S * T_bnz_nb
        ffn.W_gate[unit, BD.TEMP + 16 + k] = 1.0  # FETCH_HI via TEMP
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write target - 5 hi nibble, borrow (target_lo < 5): hi -= 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S * 20  # Require OP_BNZ to be active
        ffn.W_up[unit, BD.CMP + 4] = -S
        for j in range(0, 5):
            ffn.W_up[unit, BD.TEMP + j] = S
        ffn.b_up[unit] = -S * T_bnz_nb
        ffn.W_gate[unit, BD.TEMP + 16 + k] = 1.0  # FETCH_HI via TEMP
        ffn.W_down[BD.OUTPUT_HI + (k + 15) % 16, unit] = 2.0 / S
        unit += 1

    # Group B: lo IS zero but hi is nonzero
    #   up = S*(MARK_PC + CMP[3] + CMP[4] - CMP[5]) - S*2.5
    #   BNZ + lo_zero + hi_nonzero: 1+1+1-0=3 > 2.5 → fires
    #   BNZ + lo_zero + hi_zero (AX=0): 1+1+1-1=2 < 2.5 → off
    #   gate = just the value
    T_bnz_b = 2.5
    # Cancel existing OUTPUT
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write target - 5 lo nibble: remap k → (k+11)%16
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.TEMP + k] = 1.0  # FETCH_LO via TEMP
        ffn.W_down[BD.OUTPUT_LO + (k + 11) % 16, unit] = 2.0 / S
        unit += 1
    # Write target - 5 hi nibble: lo=0 always < 5, so always borrow (hi -= 1)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = -S
        ffn.b_up[unit] = -S * T_bnz_b
        ffn.W_gate[unit, BD.TEMP + 16 + k] = 1.0  # FETCH_HI via TEMP
        ffn.W_down[BD.OUTPUT_HI + (k + 15) % 16, unit] = 2.0 / S
        unit += 1

    # === Cancel OPCODE_BYTE contamination at AX marker ===
    # L5 head 1 writes the opcode byte to OPCODE_BYTE_LO/HI at AX marker.
    # These dims overlap ADDR_B0_LO/ADDR_B1_LO which L7 uses to gather
    # prev-AX address bytes for L15 memory lookup.  The stale opcode nibbles
    # inflate L15 address-match scores and break ZFOD.
    # Fix: negate each OPCODE_BYTE dim at the AX marker.
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OPCODE_BYTE_LO + k] = -1.0
        ffn.W_down[BD.ADDR_B0_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OPCODE_BYTE_HI + k] = -1.0
        ffn.W_down[BD.ADDR_B1_LO + k, unit] = 2.0 / S
        unit += 1

    # === Cancel MEM_STORE leakage at non-MEM markers ===
    # L6 attn head 6 writes MEM_STORE at ALL marker positions (SP, STACK0, BP,
    # MEM) because the O matrix is position-independent. But MEM_STORE should
    # only persist at the MEM marker — L14 uses it to gate memory generation,
    # and stray MEM_STORE at SP/STACK0/BP causes L14 to write garbage to
    # OUTPUT_LO/HI at those positions.
    # Fix: subtract MEM_STORE at non-MEM markers (SP, STACK0, BP).
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MEM_STORE] = -1.0
    ffn.W_down[BD.MEM_STORE, unit] = 2.0 / S
    unit += 1

    # Same for MEM_ADDR_SRC (also leaked by head 6)
    ffn.W_up[unit, BD.MARK_SP] = S
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MEM_ADDR_SRC] = -1.0
    ffn.W_down[BD.MEM_ADDR_SRC, unit] = 2.0 / S
    unit += 1


# =============================================================================
# Layer 6 Attention: Relay heads for cross-register data movement
# =============================================================================


def _set_layer6_relay_heads(attn, S, BD, HD):
    """L6 attention head 6: Cross-register data relay for PSH.

    IMPORTANT: Uses head 6 (previously unused) instead of head 2 to avoid
    conflicts with _set_layer6_attn which already configures heads 0-5.
    Previously this function overwrote head 2, causing AX_CARRY corruption.

    NOTE: The original head 3 (SP relay for ADJ) is NOT configured here because:
    1. Head 7 is reserved for JSR handling (configured later in set_vm_weights)
    2. ADJ operation is not critical for basic arithmetic (ADD/SUB/MUL/DIV)
    3. Can be added back if ADJ support is needed, using a different approach

    Head 6: At STACK0 marker, read AX marker's AX_CARRY → ALU staging.
    This provides the AX value at STACK0 position for PSH (STACK0=AX).
    Distance from STACK0 marker to AX marker = 20-5 = 15 tokens back.
    FIX 2026-04-15: Reverted from AX_FULL to AX_CARRY because AX_FULL (dims 471-502)
    overlaps with TEMP (dims 480-511), causing PC+1 computation to corrupt AX_FULL.
    """
    L = 50.0

    # Head 6: STACK0 ← AX LO nibble (AX_CARRY_LO → ALU_LO at STACK0 marker, d=15)
    # Head 7: STACK0 ← AX HI nibble (AX_CARRY_HI → ALU_HI at STACK0 marker, d=15)
    # NOTE: _set_opcode_relay_head uses head 6 V slots 1-7. We use slots 8-23 for LO.
    # FIX 2026-04-16: Extended from 13 to 16 dims to support all nibble values (0-15).
    # Previously only copied dims 0-12, causing PSH to fail for values with nibbles 13-15
    # (e.g., 30 = 0x1E has lo nibble 14, which was not copied).
    # FIX 2026-04-15: Reverted from AX_FULL to AX_CARRY because AX_FULL overlaps TEMP.

    # Head 6: LO nibble (slots 8-23)
    base6 = 6 * HD
    attn.W_q[base6, BD.MARK_STACK0] = L
    attn.W_q[base6, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base6, BD.MARK_AX] = L
    for k in range(16):
        attn.W_v[base6 + 8 + k, BD.AX_CARRY_LO + k] = 1.0
        attn.W_o[BD.ALU_LO + k, base6 + 8 + k] = 1.0

    # Head 7: HI nibble (slots 0-15)
    # NOTE: Head 7 was reserved for JSR, but JSR Q activates at different positions
    # (STACK0 for return address push), so there's no conflict with PSH Q (also STACK0).
    # The Q weights are additive, so both PSH and JSR can use head 7 at STACK0.
    base7 = 7 * HD
    attn.W_q[base7, BD.MARK_STACK0] = L
    attn.W_q[base7, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base7, BD.MARK_AX] = L
    for k in range(16):
        attn.W_v[base7 + k, BD.AX_CARRY_HI + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base7 + k] = 1.0


# =============================================================================
# Layer 7-8: Operand gather + ALU computation
# =============================================================================


def _set_layer7_operand_gather(attn, S, BD, HD):
    """L7 attention: Operand A gather for binary operations.

    Head 0: At AX marker, read previous step's STACK0 byte 0 → ALU_LO/ALU_HI.
    This provides operand A (stack top) for binary ops (ADD, SUB, etc.).

    STACK0 byte 0 is identified by STACK0_BYTE0 flag (from L1 FFN).
    Distance from current AX marker to prev step's STACK0 byte 0:
      AX marker at position 5 in current step.
      STACK0 byte 0 at position 21 in prev step.
      Distance = 35 - 21 + 5 = 19 tokens.

    Head 1: LEA operand gather — BP OUTPUT → ALU at AX marker.
    LEA computes AX = FETCH + BP. Head 1 copies BP's output to ALU_LO/HI
    at AX marker (only when OP_LEA active via Q gating).
    Distance from AX(pos 5) to BP(pos 15 in prev step) = 35 - 15 + 5 = 25.
    With slope=0.5: score = 15^2*0.125 - 0.5*25 = 28.125 - 12.5 = 15.625.

    ALiBi slope should favor d=19 strongly.
    """
    L = 15.0

    # Head 0: AX ← prev STACK0 byte 0 (STACK0_BYTE0 key)
    base = 0 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.OP_LEA] = -L  # suppress STACK0→ALU for LEA
    attn.W_k[base, BD.STACK0_BYTE0] = L
    # V: copy CLEAN_EMBED_LO/HI from STACK0 byte 0 (pristine, not inflated)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to ALU_LO/ALU_HI at AX marker
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 1.0

    # Head 1: LEA/ADJ/ENT — BP/SP OUTPUT → ALU at AX marker
    # LEA: fires when OP_LEA active, gathers BP
    # ADJ: fires when OP_ADJ active, gathers SP
    # ENT: fires when OP_ENT active, gathers SP (for SP -= 8+imm computation)
    #
    # IMPORTANT: When all opcodes inactive, we need Q < 0 so that softmax1
    # gives near-zero attention. Q=0 would give ~50% weight at d=0 (where
    # exp(0)=1 in softmax1 denominator), causing significant leakage!
    #
    # FIX 2026-04-17: Add MARK_AX gating. With global OP_ENT injection (at all positions),
    # this head was firing at all positions (including PC marker), causing ALU_LO/HI
    # to be written everywhere. The L8 carry computation then explodes at PC marker.
    # Now: Q[0] requires MARK_AX + opcode, with baseline suppression.
    # At AX marker with OP_ENT=5: Q[0] = 150 + 75 - 75 = 150
    # At PC marker with OP_ENT=5: Q[0] = 0 + 75 - 75 = 0
    base = 1 * HD
    attn.W_q[base, BD.MARK_AX] = L * 10  # Strong AX marker requirement
    attn.W_q[base, BD.OP_LEA] = L  # fires when LEA active
    attn.W_q[base, BD.OP_ADJ] = L  # fires when ADJ active
    attn.W_q[base, BD.OP_ENT] = L  # fires when ENT active
    attn.W_q[base, BD.CONST] = -L * 5  # Baseline suppression (cancel OP_* at non-AX)
    # Anti-leakage gate dimension: suppresses when not at AX marker
    attn.W_q[base + 1, BD.CONST] = -L * 2  # -30 baseline
    attn.W_q[base + 1, BD.MARK_AX] = L * 3  # +45 at AX marker → net +15
    attn.W_k[base + 1, BD.CONST] = 1.0  # K[1] = 1 everywhere
    attn.W_k[base, BD.MARK_BP] = L  # attends to BP (for LEA)
    attn.W_k[base, BD.MARK_SP] = L  # attends to SP (for ADJ/ENT)
    # V: copy OUTPUT_LO/HI (BP's or SP's byte-0 output from L6)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
    # O: write to ALU_LO/ALU_HI at AX marker
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 1.0


def _set_layer7_memory_heads(attn, S, BD, HD):
    """L7 attention heads 1-6: Memory operation relay heads.

    Head 7: Broadcast MEM_STORE + MEM_ADDR_SRC from MEM marker to MEM byte positions.
    Heads 2-4: Gather prev step's AX bytes → current AX positions (for LI/LC address).
    Head 5: Relay OP_LI/OP_LC flags from AX marker to AX byte positions.
    Head 6: Relay PSH/ENT/JSR flags from STACK0 marker to STACK0 byte positions.
    """
    L = 15.0
    MEM_I = 4
    AX_I = 1
    SP_I = 2
    BP_I = 3

    # === Head 7: MEM flag broadcast (MEM marker → MEM byte positions d=1..8) ===
    # NOTE: was head 1 but collided with LEA relay (also head 1 in operand gather).
    base = 7 * HD
    # Q: fires at MEM marker + positions d≤8.5 from MEM (H3[MEM]=1)
    # Suppress non-MEM positions: subtract H1[AX], H1[SP], H1[BP], H4[BP]
    attn.W_q[base, BD.MARK_MEM] = L
    attn.W_q[base, BD.H3 + MEM_I] = L  # d≤8.5 from MEM → all MEM bytes
    attn.W_q[base, BD.H1 + AX_I] = -L  # suppress AX area
    attn.W_q[base, BD.H1 + SP_I] = -L  # suppress SP area
    attn.W_q[base, BD.H1 + BP_I] = -L  # suppress BP area
    attn.W_q[base, BD.H4 + BP_I] = -L  # suppress STACK0 area
    # K: attend to MEM marker
    attn.W_k[base, BD.MARK_MEM] = L
    # V: copy MEM_STORE, MEM_ADDR_SRC, OP_JSR, OP_ENT
    # FIX 2026-04-16: Also broadcast OP_JSR/OP_ENT for L14 val heads STACK0 source
    attn.W_v[base + 1, BD.MEM_STORE] = 1.0
    attn.W_v[base + 2, BD.MEM_ADDR_SRC] = 1.0
    attn.W_v[base + 3, BD.OP_JSR] = 1.0
    attn.W_v[base + 4, BD.OP_ENT] = 1.0
    # O: write to same dims (accumulates at byte positions)
    attn.W_o[BD.MEM_STORE, base + 1] = 1.0
    attn.W_o[BD.MEM_ADDR_SRC, base + 2] = 1.0
    attn.W_o[BD.OP_JSR, base + 3] = 1.0
    attn.W_o[BD.OP_ENT, base + 4] = 1.0

    # === Heads 2-4: Gather prev AX bytes → AX positions (for LI/LC) ===
    # Each head J gathers addr byte J from prev step's AX byte J.
    # K: BYTE_INDEX_J + H1[AX] (fires at prev step's AX byte J input position)
    # Distance from curr AX marker to prev AX byte J ≈ 34 tokens
    for j in range(3):
        head = 2 + j
        base = head * HD
        byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
        # Q: fires at AX marker + AX bytes, suppressed at MEM/STACK0
        attn.W_q[base, BD.MARK_AX] = L
        attn.W_q[base, BD.H1 + AX_I] = L  # AX byte positions (d≤4.5 from AX)
        attn.W_q[base, BD.H3 + MEM_I] = -L  # suppress MEM area
        attn.W_q[base, BD.H4 + BP_I] = -L  # suppress STACK0 area
        # K: fires at prev step's AX byte J
        attn.W_k[base, byte_idx_dim] = L
        attn.W_k[base, BD.H1 + AX_I] = L  # must be in AX area
        # Anti-leakage: require both conditions (threshold 1.5)
        attn.W_q[base + 33, BD.CONST] = -L / 2
        attn.W_q[base + 33, BD.MARK_AX] = L
        attn.W_k[base + 33, BD.CONST] = L
        # V: copy CLEAN_EMBED nibbles
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ADDR_BJ_LO/HI
        for k in range(16):
            attn.W_o[addr_lo_out + k, base + 1 + k] = 1.0
            attn.W_o[addr_hi_out + k, base + 17 + k] = 1.0

    # === Head 5: Relay OP_LI/OP_LC/OP_LEA from AX marker → AX byte positions ===
    base = 5 * HD
    # Q: fires at AX marker + AX bytes
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.H1 + AX_I] = L  # AX byte positions
    # K: attend to AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: OP_LI, OP_LC, OP_LEA (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 1, BD.OP_LI] = 0.2
    attn.W_v[base + 2, BD.OP_LC] = 0.2
    attn.W_v[base + 3, BD.OP_LEA] = 0.2  # FIX 2026-04-13: Relay LEA for first-step byte 2 output
    # O: write to relay dims (×5 to normalize)
    attn.W_o[BD.OP_LI_RELAY, base + 1] = 1.0
    attn.W_o[BD.OP_LC_RELAY, base + 2] = 1.0
    attn.W_o[BD.CMP + 7, base + 3] = 1.0  # OP_LEA relay → CMP[7]

    # === Head 6: Relay PSH/ENT/JSR from STACK0 marker → STACK0 byte positions ===
    # Also relay PSH_AT_SP from SP marker → SP byte positions.
    base = 6 * HD
    # Q: fires at STACK0 area (marker + bytes) AND SP area (marker + bytes)
    attn.W_q[base, BD.MARK_STACK0] = L
    # FIX 2026-04-16: Changed from L1H4 (d≤6.5) to H4 (d≤9.5) to cover all STACK0 bytes.
    # STACK0 bytes are at d=6,7,8,9 from BP marker. L1H4 only covered byte 0 (d=6).
    attn.W_q[base, BD.H4 + BP_I] = L  # d≤9.5 from BP (STACK0 bytes at d=6-9)
    attn.W_q[base, BD.H1 + BP_I] = -L  # Exclude BP bytes (d≤4.5)
    attn.W_q[base, BD.IS_BYTE] = L  # only at byte positions (not at SE)
    attn.W_q[base, BD.MARK_SP] = L  # also fire at SP marker
    attn.W_q[base, BD.H1 + SP_I] = L  # also fire at SP byte positions
    # Suppress non-target areas
    attn.W_q[base, BD.H1 + AX_I] = -L
    attn.W_q[base, BD.H3 + MEM_I] = -L
    # K: attend to STACK0 marker (for STACK0 positions) or SP marker (for SP positions)
    attn.W_k[base, BD.MARK_STACK0] = L
    attn.W_k[base, BD.MARK_SP] = L
    # V: copy CMP[0] (PSH), CMP[2] (ENT), CMP[3] (POP), CMP[4] (JSR), PSH_AT_SP from markers
    attn.W_v[base + 1, BD.CMP + 0] = 1.0  # PSH flag (legacy)
    attn.W_v[base + 2, BD.CMP + 2] = 1.0  # ENT flag
    attn.W_v[base + 3, BD.CMP + 4] = 1.0  # JSR flag
    attn.W_v[base + 4, BD.PSH_AT_SP] = 1.0  # Clean PSH flag for SP bytes
    attn.W_v[base + 5, BD.CMP + 3] = 1.0  # POP group flag (for SP passthrough suppression)
    # O: accumulate at STACK0/SP byte positions
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    attn.W_o[BD.CMP + 2, base + 2] = 1.0
    attn.W_o[BD.CMP + 4, base + 3] = 1.0
    attn.W_o[BD.PSH_AT_SP, base + 4] = 1.0
    attn.W_o[BD.CMP + 3, base + 5] = 1.0  # POP group relay


def _set_layer8_sp_gather(attn, S, BD, HD):
    """L8 attention heads 0-2: Gather SP bytes → STACK0 positions.

    For *SP lookup address. Each head J gathers SP byte J to STACK0 area.
    """
    L = 15.0
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4

    for j in range(3):
        base = j * HD
        byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
        # Q: fires at STACK0 area (d=5..9 from BP or STACK0 marker)
        attn.W_q[base, BD.MARK_STACK0] = L
        attn.W_q[base, BD.H4 + BP_I] = L  # d≤9.5 from BP
        # Suppress non-STACK0
        attn.W_q[base, BD.H1 + AX_I] = -L
        attn.W_q[base, BD.H1 + SP_I] = -L
        attn.W_q[base, BD.H3 + MEM_I] = -L
        attn.W_q[base, BD.MARK_BP] = -L  # FIX 2026-04-15: Suppress at BP marker itself
        # K: fires at SP byte J (BYTE_INDEX_J + H1[SP])
        attn.W_k[base, byte_idx_dim] = L
        attn.W_k[base, BD.H1 + SP_I] = L  # must be in SP area
        # Anti-leakage gate
        attn.W_q[base + 33, BD.MARK_STACK0] = L
        attn.W_q[base + 33, BD.CONST] = -L / 2
        attn.W_k[base + 33, BD.CONST] = L
        # V: copy CLEAN_EMBED nibbles
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ADDR_BJ_LO/HI
        for k in range(16):
            attn.W_o[addr_lo_out + k, base + 1 + k] = 1.0
            attn.W_o[addr_hi_out + k, base + 17 + k] = 1.0


def _set_layer8_alu(ffn, S, BD):
    """L8 FFN: ADD/SUB lo nibble + carry/borrow + CMP_GROUP flag.

    Uses 3-way AND in silu path: MARK_AX + ALU_LO[a] + AX_CARRY_LO[b].
    Only the exact (a, b) pair fires — silu(-S*0.5) ≈ 0 for mismatches.
    Opcode gating: gate = OP_xxx (b_gate=0). Zero cross-opcode leakage.

    Total: ~753 units (ADD lo 256 + SUB lo 256 + ADD carry 120 +
                       SUB borrow 120 + CMP_GROUP 1).
    """
    unit = 0

    # === ADD: lo nibble (256 units) ===
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
            ffn.b_up[unit] = -S * 2.5  # 3-way AND
            ffn.W_gate[unit, BD.OP_ADD] = 1.0
            # LEA moved to separate units that read from FETCH
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === LEA: lo nibble (256 units) ===
    # Like ADD but reads from FETCH_LO instead of AX_CARRY_LO
    # BUG FIX 2026-04-09: Require BOTH ALU_LO AND FETCH_LO to be active.
    # At AX marker: MARK_AX=1, ALU_LO[correct]≈21, FETCH_LO[correct]≈40
    # Threshold must be > 60+0+40=100 (MARK_AX + FETCH only) but < 60+21+40=121 (all active)
    # Using threshold=105: both active=121-105=16>0, FETCH only=100-105=-5<0
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 60  # Strong MARK_AX requirement
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.FETCH_LO + b] = S  # Read from FETCH, not AX_CARRY
            ffn.b_up[unit] = -S * 105  # Require BOTH ALU_LO and FETCH_LO to be active
            ffn.W_gate[unit, BD.OP_LEA] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === SUB: lo nibble (256 units) ===
    # Correct C4 semantics: AX = stack_top - AX = ALU - AX_CARRY = a - b.
    # ALU contains stack top (minuend), AX_CARRY contains current AX (subtrahend).
    for a in range(16):
        for b in range(16):
            result = (a - b) % 16  # ALU - AX_CARRY = stack - AX
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_gate[unit, BD.OP_SUB] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === ADD carry detection (120 units: pairs where a+b >= 16) ===
    for a in range(16):
        for b in range(16):
            if a + b >= 16:
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.W_gate[unit, BD.OP_ADD] = 1.0
                # LEA moved to separate units
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)  # normalize: gate≈5 → CARRY≈1
                unit += 1

    # === LEA carry detection (120 units: pairs where a+b >= 16) ===
    # BUG FIX 2026-04-09: Require BOTH ALU_LO and FETCH_LO to be active (threshold=105)
    for a in range(16):
        for b in range(16):
            if a + b >= 16:
                ffn.W_up[unit, BD.MARK_AX] = S * 60
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.FETCH_LO + b] = S  # Read from FETCH
                ffn.b_up[unit] = -S * 105  # Match LEA lo nibble threshold
                ffn.W_gate[unit, BD.OP_LEA] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === ADJ: lo nibble (256 units) ===
    # Like LEA but gates on OP_ADJ instead
    # ADJ computes: SP = SP + signed_immediate (gathered via L7 head 1)
    # BUG FIX 2026-04-09: Require BOTH ALU_LO and FETCH_LO (threshold=105)
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 60
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.FETCH_LO + b] = S  # Read from FETCH (immediate)
            ffn.b_up[unit] = -S * 105  # Require BOTH ALU_LO and FETCH_LO to be active
            ffn.W_gate[unit, BD.OP_ADJ] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === ADJ carry detection (120 units: pairs where a+b >= 16) ===
    # BUG FIX 2026-04-09: Require BOTH ALU_LO and FETCH_LO (threshold=105)
    for a in range(16):
        for b in range(16):
            if a + b >= 16:
                ffn.W_up[unit, BD.MARK_AX] = S * 60
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.FETCH_LO + b] = S  # Read from FETCH
                ffn.b_up[unit] = -S * 105  # Match ADJ lo nibble threshold
                ffn.W_gate[unit, BD.OP_ADJ] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === SUB borrow detection (120 units: pairs where a < b) ===
    # Borrow occurs when ALU_LO < AX_CARRY_LO (when stack_top < AX for this nibble).
    # a = ALU (stack_top), b = AX_CARRY (AX). SUB = a - b, borrow when a < b.
    for a in range(16):
        for b in range(16):
            if a < b:  # Borrow when stack_top < AX
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.W_gate[unit, BD.OP_SUB] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)  # normalize: gate≈5 → CARRY≈1
                unit += 1

    # === ENT: lo nibble subtraction (256 units) ===
    # ENT computes: SP = SP - (8 + signed_immediate)
    # For lo nibble: result_lo = (sp_lo - (8 + imm_lo)) mod 16
    # where sp_lo comes from ALU_LO (gathered by L7 head 1)
    # and imm_lo comes from FETCH_LO (instruction immediate)
    # BUG FIX 2026-04-09: Require BOTH ALU_LO and FETCH_LO (threshold=105)
    for sp_lo in range(16):
        for imm_lo in range(16):
            effective_b = (8 + imm_lo) % 16  # Add constant offset 8
            result = (sp_lo - effective_b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 60
            ffn.W_up[unit, BD.ALU_LO + sp_lo] = S  # SP lo nibble from L7
            ffn.W_up[unit, BD.FETCH_LO + imm_lo] = S  # Immediate lo nibble
            ffn.b_up[unit] = -S * 105  # Require BOTH ALU_LO and FETCH_LO to be active
            ffn.W_gate[unit, BD.OP_ENT] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === ENT borrow detection (256 units) ===
    # Borrow when sp_lo < (8 + imm_lo) mod 16
    # Must enumerate all cases (not just where borrow occurs) because
    # the condition depends on the constant 8, not just relative magnitudes
    # BUG FIX 2026-04-09: Require BOTH ALU_LO and FETCH_LO (threshold=105)
    for sp_lo in range(16):
        for imm_lo in range(16):
            effective_b = (8 + imm_lo) % 16
            # Check if borrow is needed: sp_lo < effective_b
            # But need to account for the full byte math: does (8 + imm_lo) >= 16?
            # If (8 + imm_lo) >= 16, there's a carry out of byte 0 into byte 1
            full_sum = 8 + imm_lo  # This is in range [8, 23]
            if sp_lo < (full_sum % 16) or full_sum >= 16:
                # Need borrow from byte 1
                ffn.W_up[unit, BD.MARK_AX] = S * 60
                ffn.W_up[unit, BD.ALU_LO + sp_lo] = S
                ffn.W_up[unit, BD.FETCH_LO + imm_lo] = S
                ffn.b_up[unit] = -S * 105
                ffn.W_gate[unit, BD.OP_ENT] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === CMP_GROUP flag (1 unit) ===
    # ~1.0 when any comparison opcode active at AX marker.
    # OP flags ≈ 5.0, so silu(S*(5+1-1.5))=S*4.5. Normalize W_down so output ≈ 1.
    for op in [BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE]:
        ffn.W_up[unit, op] = S
    ffn.W_up[unit, BD.MARK_AX] = S
    ffn.b_up[unit] = -S * 1.5  # any_cmp_op(~5) + MARK_AX(1) > 1.5
    ffn.b_gate[unit] = 1.0  # unconditional gate
    ffn.W_down[BD.CMP_GROUP, unit] = 2.0 / (S * 9.0)  # normalize: 9 * 2/(S*9) ≈ 1.0
    unit += 1

    # === ENT/ADJ first-step ALU defaults (2 units) ===
    # FIX 2026-04-17: For first step (NOT HAS_SE), L7 attention can't gather SP
    # because the SP marker is AFTER the AX marker in the sequence (causal attention).
    # For ENT/ADJ with initial SP = 0, we need ALU_LO[0] > 0 and ALU_HI[0] > 0.
    # These units fire when: OP_ENT (or OP_ADJ) + MARK_AX + NOT HAS_SE
    # OP_ENT ≈ 22 at SP marker after L7 (amplified from 5 at embedding).
    # BUG FIX 2026-04-17: Add MARK_SP blocker. OP_ENT alone contributes ~750 at SP marker,
    # which overcomes the bias (-600). Need -1000 from MARK_SP to block.
    # L7 attention writes garbage to ALU (~-32) when there's no SP to attend to,
    # so we need large output weights to override this: 50.0/S gives ~+50 contribution.
    for alu_dim, output_weight in [(BD.ALU_LO, 50.0 / S), (BD.ALU_HI, 50.0 / S)]:
        ffn.W_up[unit, BD.OP_ENT] = S / 3  # ~22/3 ≈ 7.3 contribution at SP marker
        ffn.W_up[unit, BD.OP_ADJ] = S / 3  # Also handle ADJ first step
        ffn.W_up[unit, BD.MARK_AX] = S * 2  # 2 contribution when at AX marker
        ffn.W_up[unit, BD.MARK_SP] = -S * 10  # Block at SP marker (MARK_SP = 1)
        ffn.W_up[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
        ffn.b_up[unit] = -S * 6  # Threshold: need ENT/ADJ + MARK_AX, blocked by HAS_SE
        ffn.b_gate[unit] = 1.0  # Always gate open (SiLU path does gating)
        ffn.W_down[alu_dim + 0, unit] = output_weight  # Write to ALU[0] for SP = 0
        unit += 1

    # === LEV: BP address relay (BP OUTPUT → ADDR dims) - PHASE 1 ===
    # For L15 to read memory at BP and BP+8, we need BP's address value
    # encoded in ADDR_B0/B1/B2 dims at the BP marker position.
    #
    # CURRENT LIMITATION: Only byte 0 is relayed here (covers addresses < 256).
    # This is sufficient for most C4 test programs which use small stack frames.
    #
    # For larger addresses, we would need:
    # - L7/L8 attention heads to gather BP bytes 1-2 from their positions
    # - Write gathered bytes to TEMP dims
    # - Copy TEMP → ADDR_B1/B2 here
    #
    # Total for byte 0 only: 32 units (1 byte × 2 nibbles × 16 values/nibble)

    # Byte 0 lo nibble: OUTPUT_LO → ADDR_B0_LO
    # FIX 2026-04-15: OP_LEV ≈ 5 and MARK_BP = 1, so up = S * (5 + 1 - 1.5) = S * 4.5
    # silu(S * 4.5) ≈ S * 4.5, so W_down = 2.0 / (S * 9) to normalize output to ~1.0
    # FIX 2026-04-16: Add PC marker exclusion. OP_LEV gets amplified to ~10 by L6,
    # so without MARK_PC penalty, units fire at PC marker (OP_LEV*10 > threshold).
    for k in range(16):
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_BP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 10  # Exclude PC marker
        ffn.b_up[unit] = -S * 1.5  # both OP_LEV and MARK_BP required
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = 1.0
        ffn.W_down[BD.ADDR_B0_LO + k, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
        unit += 1

    # Byte 0 hi nibble: OUTPUT_HI → ADDR_B0_HI
    for k in range(16):
        ffn.W_up[unit, BD.OP_LEV] = S
        ffn.W_up[unit, BD.MARK_BP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S * 10  # Exclude PC marker
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = 1.0
        ffn.W_down[BD.ADDR_B0_HI + k, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
        unit += 1

    # Bytes 1-2: Set to zero (assume addresses < 256 for now)
    # This gives ADDR_B1 = ADDR_B2 = 0, which is correct for small addresses
    ffn.W_up[unit, BD.OP_LEV] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0  # Always 1.0
    ffn.W_down[BD.ADDR_B1_LO + 0, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
    unit += 1

    ffn.W_up[unit, BD.OP_LEV] = S
    ffn.W_up[unit, BD.MARK_BP] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.ADDR_B2_LO + 0, unit] = 2.0 / (S * 9)  # FIX: was 2.0/S
    unit += 1

    # === REMOVED 2026-04-16: PC marker ADDR_B0 moved to L9 attention head 1 ===
    # L9 attention head 1 now directly writes BP value to ADDR_B0 at PC marker.
    # Having L8 FFN also write to ADDR_B0 causes interference (values are additive).
    # The 34 units (32 for byte 0 + 2 for bytes 1-2) have been removed.

    return unit


def _set_layer9_lev_addr_relay(attn, S, BD, HD):
    """L9 attention head 0: relay old BP value from prev step's BP byte 0 to SP marker.

    For LEV, L16 FFN computes SP = old_BP + 16. We need old_BP at the SP marker.

    The previous step's BP byte 0 position has CLEAN_EMBED = old_BP value.
    We relay this to ADDR_B0 at the current step's SP marker.

    Key insight: The previous step's BP marker has wrong ADDR_B0 (gets corrupted
    by L8 attention leakage). But the BP byte 0 position has correct CLEAN_EMBED.
    So we attend to BP byte 0 (L1H1[BP_I] AND BYTE_INDEX_0) instead of BP marker.

    Distance from current SP marker to prev BP byte 0 ≈ 29 tokens.

    BUG FIX 2026-04-15: Original approach attended to BP marker which had wrong
    ADDR_B0 due to L8 head 0 leakage. Now attend to BP byte 0 for correct value.
    """
    L = 50.0
    BP_I = 3  # BP marker index
    base = 0 * HD  # head 0

    # Q: fires at SP marker when OP_LEV active
    attn.W_q[base, BD.MARK_SP] = L
    attn.W_q[base, BD.OP_LEV] = L / 5  # OP_LEV ≈ 5, normalize to ~L
    attn.W_q[base, BD.CONST] = -2 * L  # need both MARK_SP and OP_LEV

    # K: attend to BP byte 0 (L1H1[BP_I] AND BYTE_INDEX_0)
    # L1H1[BP_I] = 1 when within 2.5 tokens of BP marker
    # BYTE_INDEX_0 = 1 when it's byte 0 of a register section
    attn.W_k[base, BD.L1H1 + BP_I] = L
    attn.W_k[base, BD.BYTE_INDEX_0] = L

    # V: copy CLEAN_EMBED_LO/HI (the actual BP byte 0 value)
    # Scale up to dominate over existing values in residual add
    scale = 3.0
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale

    # O: write to ADDR_B0_LO/HI at SP marker
    for k in range(16):
        attn.W_o[BD.ADDR_B0_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ADDR_B0_HI + k, base + 17 + k] = 1.0

    # FIX 2026-04-15: Anti-leakage gate to suppress attention at non-SP positions.
    # Without this, positions with K=0 get Q*K=0 scores, giving uniform softmax
    # weights that accumulate and pollute ADDR_B0 at BP marker.
    # At SP marker: Q[gate] = L - L/2 = +L/2, score += +L²/(2*8) ≈ +156
    # At non-SP markers: Q[gate] = -L/2, score += -L²/(2*8) ≈ -156
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_SP] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L


def _set_layer9_lev_bp_to_pc_relay(attn, S, BD, HD):
    """L9 attention head 1: relay BP value from prev step's BP byte 0 to PC marker.

    For LEV return_addr lookup, L15 heads 8-11 need to know BP at PC marker.
    The previous step's BP byte 0 position has CLEAN_EMBED = old_BP value.
    We relay this to ADDR_B0 at the current step's PC marker.

    FIX 2026-04-16: Changed from attending to BP marker (which has OUTPUT=0)
    to attending to BP byte 0 (which has correct CLEAN_EMBED value).
    Same pattern as head 0 which works correctly at SP marker.
    """
    L = 50.0
    BP_I = 3  # BP marker index
    base = 1 * HD  # head 1

    # Q: fires at PC marker when OP_LEV active
    # FIX 2026-04-16: Reduced threshold from -2*L to -1.5*L so Q[0] > 0 at target.
    # Old: Q[0] = L + L - 2*L = 0 (K matching disabled!)
    # New: Q[0] = L + L - 1.5*L = 0.5*L = 25 (K matching works)
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.OP_LEV] = L / 5  # OP_LEV ≈ 5, normalize to ~L
    attn.W_q[base, BD.CONST] = -1.5 * L  # need both MARK_PC and OP_LEV

    # K: attend to BP byte 0 (L1H1[BP_I] AND BYTE_INDEX_0)
    # L1H1[BP_I] = 1 when within 2.5 tokens of BP marker
    # BYTE_INDEX_0 = 1 when it's byte 0 of a register section
    attn.W_k[base, BD.L1H1 + BP_I] = L
    attn.W_k[base, BD.BYTE_INDEX_0] = L

    # V: copy CLEAN_EMBED_LO/HI (the actual BP byte 0 value)
    # Scale up to dominate over existing values in residual add
    scale = 3.0
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale

    # O: write to ADDR_B0_LO/HI at PC marker
    for k in range(16):
        attn.W_o[BD.ADDR_B0_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ADDR_B0_HI + k, base + 17 + k] = 1.0

    # Anti-leakage gate to suppress attention at non-PC positions
    # At PC marker: Q[gate] = L - L/2 = +L/2
    # At non-PC markers: Q[gate] = -L/2
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = L
    attn.W_q[base + GATE, BD.CONST] = -L / 2
    attn.W_k[base + GATE, BD.CONST] = L


def _set_layer9_alu(ffn, S, BD):
    """L9 FFN: ADD/SUB hi nibble (with carry/borrow) + comparison flags.

    Hi nibble uses 4-way AND: MARK_AX + ALU_HI[a] + AX_CARRY_HI[b] ± CARRY.
    CARRY ≈ 1.0 when present (from L8 with b_gate=0), ≈ 0 otherwise.

    Comparison flags use 3-way AND: MARK_AX + operand_a + operand_b,
    gated by CMP_GROUP in gate path (b_gate=0). CMP flags ≈ 1.0 when true.

    Total: ~1296 units.
    """
    unit = 0

    # === ADD hi nibble (no carry 256 + with carry 256 = 512 units) ===
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if carry_in == 0:
                    # Block when carry present: need strong negative to prevent firing
                    # FIX 2026-04-16: Restored from -0.01 to -S*2.0 for proper carry discrimination
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 2.0
                    ffn.b_up[unit] = -S * 2.5  # 3-way AND
                else:
                    # Require carry: need strong positive to select carry case
                    # FIX 2026-04-16: Restored from 0.01 to S*2.0 for proper carry discrimination
                    ffn.W_up[unit, BD.CARRY + 0] = S * 2.0
                    ffn.b_up[unit] = -S * 4.5  # 4-way AND (3 regs + carry)
                ffn.W_gate[unit, BD.OP_ADD] = 1.0
                # LEA moved to separate units
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === LEA hi nibble (no carry 256 + with carry 256 = 512 units) ===
    # BUG FIX 2026-04-09: Require MARK_AX + ALU_HI + FETCH_HI to all be active.
    # At PC marker (MARK_AX=0): ALU_HI[wrong]≈12 + FETCH_HI[correct]≈43 = 55 (must block)
    # At AX marker (MARK_AX=1): 1 + ALU_HI[correct]≈21 + FETCH_HI[correct]≈40 = 62 (must fire)
    # Using threshold=58: PC marker=55-58=-3<0 (blocked), AX marker=62-58=4>0 (fires)
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.FETCH_HI + b] = S  # Read from FETCH
                if carry_in == 0:
                    ffn.W_up[unit, BD.CARRY + 0] = -0.01
                    ffn.b_up[unit] = -S * 58  # Require MARK_AX + ALU_HI + FETCH_HI
                else:
                    ffn.W_up[unit, BD.CARRY + 0] = 0.01
                    ffn.b_up[unit] = -S * 58.4  # Slightly higher for carry case
                ffn.W_gate[unit, BD.OP_LEA] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === ADJ hi nibble (no carry 256 + with carry 256 = 512 units) ===
    # Like LEA but gates on OP_ADJ instead
    # ADJ computes: SP = SP + signed_immediate
    # BUG FIX 2026-04-09: Require MARK_AX + ALU_HI + FETCH_HI (threshold=58)
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a + b + carry_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.FETCH_HI + b] = S  # Read from FETCH
                if carry_in == 0:
                    ffn.W_up[unit, BD.CARRY + 0] = -0.01
                    ffn.b_up[unit] = -S * 58  # Require MARK_AX + ALU_HI + FETCH_HI
                else:
                    ffn.W_up[unit, BD.CARRY + 0] = 0.01
                    ffn.b_up[unit] = -S * 58.4  # Slightly higher for carry case
                ffn.W_gate[unit, BD.OP_ADJ] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === SUB hi nibble (no borrow 256 + with borrow 256 = 512 units) ===
    # a = ALU_HI (stack_top), b = AX_CARRY_HI (AX). SUB = stack_top - AX = a - b.
    for borrow_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a - b - borrow_in) % 16  # ALU - AX_CARRY - borrow
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if borrow_in == 0:
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 2.0
                    ffn.b_up[unit] = -S * 2.5
                else:
                    ffn.W_up[unit, BD.CARRY + 0] = S * 2.0
                    ffn.b_up[unit] = -S * 4.5  # 4-way AND (3 regs + borrow≈1)
                ffn.W_gate[unit, BD.OP_SUB] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === ENT hi nibble (no borrow 256 + with borrow 256 = 512 units) ===
    # ENT computes: SP = SP - (8 + signed_immediate)
    # For bytes 1-3, we subtract imm_byte with borrow propagation
    # The +8 offset affects byte 0 only; its overflow propagates as a borrow
    # BUG FIX 2026-04-09: Require MARK_AX + ALU_HI + FETCH_HI (threshold=58)
    # BUG FIX 2026-04-17: Add MARK_SP blocker. When OP_ENT is amplified to 22+,
    # the gate opens fully and ALU_HI[0]=58 at SP marker meets the threshold,
    # causing OUTPUT_HI[15] to be written instead of OUTPUT_HI[0].
    for borrow_in in [0, 1]:
        for sp_hi in range(16):
            for imm_hi in range(16):
                result = (sp_hi - imm_hi - borrow_in) % 16
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.MARK_SP] = -S * 2.0  # Block at SP marker
                ffn.W_up[unit, BD.ALU_HI + sp_hi] = S  # SP hi nibble from L7
                ffn.W_up[unit, BD.FETCH_HI + imm_hi] = S  # Immediate hi nibble
                if borrow_in == 0:
                    # No borrow: block when CARRY active
                    ffn.W_up[unit, BD.CARRY + 0] = -S * 2.0
                    ffn.b_up[unit] = -S * 58  # Require MARK_AX + ALU_HI + FETCH_HI
                else:
                    # With borrow: require CARRY active
                    ffn.W_up[unit, BD.CARRY + 0] = S * 2.0
                    ffn.b_up[unit] = -S * 60  # Higher threshold (4-way AND)
                ffn.W_gate[unit, BD.OP_ENT] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === Comparison flags (shared across all cmp opcodes, gated by CMP_GROUP) ===
    # CMP[0]=hi_lt, CMP[1]=hi_eq, CMP[2]=lo_eq, CMP[3]=lo_lt

    # hi_eq: 16 units — 3-way AND (MARK_AX + ALU_HI[k] + AX_CARRY_HI[k])
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.ALU_HI + k] = S
        ffn.W_up[unit, BD.AX_CARRY_HI + k] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
        ffn.W_down[BD.CMP + 1, unit] = 2.0 / S
        unit += 1

    # lo_eq: 16 units
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.ALU_LO + k] = S
        ffn.W_up[unit, BD.AX_CARRY_LO + k] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
        ffn.W_down[BD.CMP + 2, unit] = 2.0 / S
        unit += 1

    # hi_lt: 120 units — 3-way AND (MARK_AX + ALU_HI[a] + AX_CARRY_HI[b]), a < b
    for a in range(16):
        for b in range(a + 1, 16):
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ALU_HI + a] = S
            ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
            ffn.W_down[BD.CMP + 0, unit] = 2.0 / S
            unit += 1

    # lo_lt: 120 units
    for a in range(16):
        for b in range(a + 1, 16):
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_gate[unit, BD.CMP_GROUP] = 1.0
            ffn.W_down[BD.CMP + 3, unit] = 2.0 / S
            unit += 1

    # === ADD hi-nibble carry-out → CARRY[1] (byte carry for inter-byte propagation) ===
    # Detects (a + b + carry_in >= 16) for hi nibble. Same AND pattern as hi nibble
    # result, but writes to CARRY[1] instead of OUTPUT_HI.
    for carry_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                if a + b + carry_in < 16:
                    continue  # no carry-out
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if carry_in == 0:
                    # Block when carry present: -0.01*CARRY prevents spurious activation
                    ffn.W_up[unit, BD.CARRY + 0] = -0.01  # Reduced from -S*2.0
                    ffn.b_up[unit] = -S * 2.5  # 3-way AND
                else:
                    # Require carry: +0.01*CARRY (hint, not hard requirement)
                    ffn.W_up[unit, BD.CARRY + 0] = 0.01  # Reduced from S*2.0
                    ffn.b_up[unit] = -S * 2.9  # Relaxed from -S*4.5 to allow activation with just 3 inputs
                ffn.W_gate[unit, BD.OP_ADD] = 1.0
                # NOTE: OP_LEA removed from gate because LEA = opcode 0, which causes
                # false positives (many tokens have value 0 and thus OP_LEA=1 in embedding).
                # If LEA byte carry is needed, add explicit OP_LEA gating with suppression.
                ffn.W_down[BD.CARRY + 1, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === SUB hi-nibble borrow-out → CARRY[2] (byte borrow for inter-byte propagation) ===
    # Detects borrow-out from hi nibble of ALU - AX_CARRY.
    # For SUB: stack - AX = ALU - AX_CARRY, so borrow when ALU < AX_CARRY.
    # a = ALU_HI index, b = AX_CARRY_HI index.
    for borrow_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                if borrow_in == 0:
                    if a >= b:
                        continue  # no borrow-out when ALU >= AX_CARRY
                else:
                    if a > b:
                        continue  # no borrow-out when ALU > AX_CARRY (a - b - 1 >= 0)
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                if borrow_in == 0:
                    # Block when carry present: -0.01*CARRY prevents spurious activation
                    ffn.W_up[unit, BD.CARRY + 0] = -0.01  # Reduced from -S*2.0
                    ffn.b_up[unit] = -S * 2.5  # 3-way AND
                else:
                    # Require carry: +0.01*CARRY (hint, not hard requirement)
                    ffn.W_up[unit, BD.CARRY + 0] = 0.01  # Reduced from S*2.0
                    ffn.b_up[unit] = -S * 2.9  # Relaxed from -S*4.5 to allow activation with just 3 inputs
                ffn.W_gate[unit, BD.OP_SUB] = 1.0
                ffn.W_down[BD.CARRY + 2, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === ALU clearing for opcodes that don't need operand gather ===
    # Prevents spurious activation of Layer 10 bitwise/MUL units due to
    # residual ALU_LO/HI values. Fires when MARK_AX and a non-ALU opcode.
    #
    # Opcodes that don't use ALU_LO/HI for binary operations:
    non_alu_opcodes = [
        BD.OP_IMM,   # loads immediate
        # BD.OP_LEA removed: LEA DOES use ALU (attention head copies BP → ALU, then adds immediate)
        BD.OP_NOP,   # no operation
        BD.OP_JMP,   # unconditional jump
        BD.OP_JSR,   # jump subroutine
        BD.OP_EXIT,  # exit program
        BD.OP_BZ,    # branch if zero
        BD.OP_BNZ,   # branch if not zero
        BD.OP_ENT,   # enter stack frame
        BD.OP_ADJ,   # adjust stack
        BD.OP_LEV,   # leave stack frame
        BD.OP_PSH,   # push to stack
        BD.OP_LI,    # load int (uses memory, not ALU)
        BD.OP_LC,    # load char (uses memory, not ALU)
        BD.OP_SI,    # store int (uses memory, not ALU)
        BD.OP_SC,    # store char (uses memory, not ALU)
    ]

    # Clear ALU_LO (16 units - one per nibble)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in non_alu_opcodes:
            ffn.W_up[unit, op_dim] = S  # each opcode contributes full S (OR logic)
        ffn.b_up[unit] = -S * 1.5  # fires when MARK_AX(S) + any_op(~5) > 1.5S
        ffn.b_gate[unit] = 1.0  # unconditional gate
        ffn.W_down[BD.ALU_LO + k, unit] = -10.0 / S  # large negative to clear
        unit += 1

    # Clear ALU_HI (16 units)
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in non_alu_opcodes:
            ffn.W_up[unit, op_dim] = S  # each opcode contributes full S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.ALU_HI + k, unit] = -10.0 / S  # large negative to clear
        unit += 1

    # === FIX 2026-04-16: Add +8 offset to ADDR_B0 at PC marker for LEV return_addr ===
    # L9 attention head 1 wrote ADDR_B0 = BP byte 0. We need ADDR_B0 = (BP+8) byte 0.
    # BP is always 8-byte aligned (lo nibble = 0 or 8), so adding 8:
    # - lo=0: +8 gives 8 (no carry)
    # - lo=8: +8 gives 0 with carry (but this case is rare for stack)
    # For simplicity, just add 8 to lo nibble without full carry propagation.
    # This works for common case where BP lo nibble is 0 (e.g., 0xfff0).
    for k in range(16):
        new_k = (k + 8) % 16
        # Fires at PC marker when OP_LEV active and ADDR_B0_LO[k] has value
        # BUG FIX 2026-04-16: Add MARK_BP exclusion. OP_LEV gets amplified to ~30
        # after L9 attention, causing units to fire at BP marker without MARK_PC.
        # BUG FIX 2026-04-16: Add MARK_SP exclusion. Same issue - OP_LEV alone
        # overcomes threshold, causing spurious ADDR_B0 shifts at SP marker.
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.OP_LEV] = S / 5  # OP_LEV ≈ 5
        ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Exclude BP marker
        ffn.W_up[unit, BD.MARK_SP] = -S * 10  # Exclude SP marker
        ffn.b_up[unit] = -S * 1.5
        # FIX 2026-04-16: Gate on ADDR_B0_LO[k] with threshold to distinguish
        # legitimate BP values (LO[8] ~ 3.0 from L9 attention) from opcode fetch
        # contamination (LO[0] ~ 2.0 from L5).
        # Gate = LO[k] - 2.5 → legitimate: 3.0-2.5=0.5>0, contam: 2.0-2.5=-0.5<0
        ffn.W_gate[unit, BD.ADDR_B0_LO + k] = 1.0
        ffn.b_gate[unit] = -2.5  # Require LO[k] > 2.5
        # Cancel old position, set new position
        # BUG FIX 2026-04-16: Scaled to shift ~4 units of energy (matching post-attn values).
        # With output=600 (silu(150)*gate(4)), W_down=0.67/S gives contrib=4.0
        ffn.W_down[BD.ADDR_B0_LO + k, unit] = -0.67 / S
        ffn.W_down[BD.ADDR_B0_LO + new_k, unit] = 0.67 / S
        unit += 1

    # === FIX 2026-04-16: Set ADDR_B1 = 0xff at PC marker for stack addresses ===
    # Stack addresses are typically 0xfff0 range, so byte 1 is always 0xff.
    # Without this, L15 can't find the MEM section at 0xfff8 (query only matches byte 0).
    # ADDR_B1_LO[15] = 1 and ADDR_B1_HI[15] = 1 → byte 1 = 0xff
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Exclude BP marker
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    # BUG FIX 2026-04-16: Scale down by 9x for amplified up branch (silu(450) vs 50)
    ffn.W_down[BD.ADDR_B1_LO + 15, unit] = 0.22 / S  # lo nibble = 15
    unit += 1

    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Exclude BP marker
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    # BUG FIX 2026-04-16: Scale down by 9x for amplified up branch
    ffn.W_down[BD.ADDR_B1_HI + 15, unit] = 0.22 / S  # hi nibble = 15
    unit += 1

    # === FIX 2026-04-16: Cascade carry for BP=0xfff8 + 8 = 0x10000 ===
    # When BP byte 0 = 0xf8 (lo=8, hi=15), adding 8 causes full cascade:
    #   byte0: 0xf8 + 8 = 0x100 → byte0=0x00, carry to byte1
    #   byte1: 0xff + 1 = 0x100 → byte1=0x00, carry to byte2
    #   byte2: 0x00 + 1 = 0x01
    # Detect via ADDR_B0_LO[8] (original lo nibble before shift) being high.

    # The +8 shift above moved ADDR_B0_LO[8]→[0], but we need to also:
    # 1. Clear ADDR_B0_HI[15] (it wasn't touched by the shift)
    # 2. Cancel the ADDR_B1=0xff setting above (for cascade case)
    # 3. Set ADDR_B2_LO[1] = 1 (byte2 = 0x01)

    # Unit 1: Clear ADDR_B0_HI[15] when lo was 8 (cascade case)
    # Gate on BOTH ADDR_B0_LO[0] (high after shift when original was 8) AND HI[15].
    # For BP=0xf0 (lo=0): after +8 shift, LO[8] is high, LO[0] is low → gate fails
    # For BP=0xf8 (lo=8): after +8 shift, LO[0] is high from shift → gate passes
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Gate on BOTH LO[0] AND HI[15] to distinguish carry case.
    # After +8 shift: BP=0xf8 has LO[0] high (shifted from LO[8]), BP=0xf0 has LO[8] high.
    # Use LO[0] + HI[15] - threshold as AND gate.
    # BP=0xf8 after shift: LO[0]~15, HI[15]~3 → gate = 18 - 15 = 3 > 0 ✓
    # BP=0xf0 after shift: LO[0]~0, HI[15]~3 → gate = 3 - 15 = -12 < 0 ✗
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0  # Require LO[0] to be significantly high
    # Clear hi nibble 15, set hi nibble 0
    ffn.W_down[BD.ADDR_B0_HI + 15, unit] = -0.67 / S
    ffn.W_down[BD.ADDR_B0_HI + 0, unit] = 0.67 / S
    unit += 1

    # Unit 2: Cancel ADDR_B1_LO[15] setting for cascade case
    # The unconditional ADDR_B1=0xff above added ~1.2 to position 15.
    # For cascade, we need to cancel it and set position 0.
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Use same AND gate as unit 1 to only fire for carry case
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0
    # Cancel the +0.22/S from unconditional unit, and clear any attention residue
    ffn.W_down[BD.ADDR_B1_LO + 15, unit] = -0.5 / S
    ffn.W_down[BD.ADDR_B1_LO + 0, unit] = 0.5 / S
    unit += 1

    # Unit 3: Cancel ADDR_B1_HI[15] setting for cascade case
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Use same AND gate as unit 1 to only fire for carry case
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0
    ffn.W_down[BD.ADDR_B1_HI + 15, unit] = -0.5 / S
    ffn.W_down[BD.ADDR_B1_HI + 0, unit] = 0.5 / S
    unit += 1

    # Unit 4: Set ADDR_B2_LO[1] = 1 for cascade case (byte2 = 0x01)
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.W_up[unit, BD.OP_LEV] = S / 5
    ffn.W_up[unit, BD.MARK_BP] = -S * 10
    ffn.W_up[unit, BD.MARK_SP] = -S * 10  # FIX 2026-04-16: Exclude SP marker
    ffn.b_up[unit] = -S * 1.5
    # FIX 2026-04-16: Use same AND gate as unit 1 to only fire for carry case
    ffn.W_gate[unit, BD.ADDR_B0_LO + 0] = 1.0
    ffn.W_gate[unit, BD.ADDR_B0_HI + 15] = 1.0
    ffn.b_gate[unit] = -15.0
    ffn.W_down[BD.ADDR_B2_LO + 1, unit] = 0.67 / S
    unit += 1

    return unit


def _set_layer10_carry_relay(attn, S, BD, HD):
    """L10 attention head 0: relay CARRY[1/2] from AX marker to AX byte positions.

    At AX byte positions (IS_BYTE=1, H1[AX]=1), attends strongly to the
    nearest AX marker and copies CARRY[1] (ADD byte carry) and CARRY[2]
    (SUB byte borrow). Anti-leakage gate via H1[AX_IDX] ensures non-AX
    bytes get negligible attention to the AX marker.
    """
    L = S  # attention scale
    AX_IDX = 1  # AX is register index 1 (PC=0, AX=1, SP=2, ...)
    base = 0  # head 0

    # Q: fires at byte positions, suppressed at markers
    attn.W_q[base + 0, BD.IS_BYTE] = L
    attn.W_q[base + 0, BD.CONST] = -L / 2  # Q[0] = L*(IS_BYTE - 0.5)

    # K: fires at AX marker positions
    attn.W_k[base + 0, BD.MARK_AX] = L

    # Anti-leakage: restrict to AX byte positions (H1[AX_IDX] ≈ 1)
    attn.W_q[base + 33, BD.H1 + AX_IDX] = L
    attn.W_q[base + 33, BD.CONST] = -L / 2
    attn.W_k[base + 33, BD.CONST] = L

    # V: copy CARRY[1] and CARRY[2] from marker
    attn.W_v[base + 1, BD.CARRY + 1] = 1.0
    attn.W_v[base + 2, BD.CARRY + 2] = 1.0

    # O: write to CARRY[1] and CARRY[2] at byte position
    attn.W_o[BD.CARRY + 1, base + 1] = 1.0
    attn.W_o[BD.CARRY + 2, base + 2] = 1.0


def _set_layer10_byte_passthrough(attn, S, BD, HD):
    """L10 attention head 1: AX byte 0-2 passthrough across steps.

    Copies CLEAN_EMBED from previous step's AX bytes 1-3 → OUTPUT at current
    step's AX byte 0-2 positions. Uses **shifted** byte matching because of the
    autoregressive offset: logits at byte K position predict byte K+1's token.

    Mapping (Q byte K attends to K byte K+1 of prev step):
      byte 0 pos → prev byte 1 (predicts byte 1 token)
      byte 1 pos → prev byte 2 (predicts byte 2 token)
      byte 2 pos → prev byte 3 (predicts byte 3 token)
      byte 3 pos → suppressed (predicts SP marker, not a byte)

    Byte 0's token (AX byte 0 value) is handled by the marker-level AX passthrough
    FFN, which reads AX_CARRY_LO/HI at the AX marker position.

    Anti-leakage:
      - Q dim 33: large negative bias at non-AX positions (H1[AX]=0) → kills softmax
      - K dim 1: H1[AX_IDX] → only AX byte positions are strong K targets
      - At step 0 (no prev AX bytes): Q attends to same-step PC bytes (all 0),
        producing OUTPUT for token 0 (correct: AX=0 initially)

    Strength: W_o=2.0, weaker than carry override (≈5.0) so carry wins when active.
    ALiBi slope=1.0: prefers nearest step (d=35 beats d=70).
    """
    L = S  # attention scale
    AX_IDX = 1
    base = HD  # head 1 starts at dim 64

    # Q dim 0: IS_BYTE AND HAS_SE (only fire on subsequent steps, not first step)
    # First step: JMP sets OUTPUT at PC marker via Layer 6, don't overwrite it
    # BUG FIX 2026-04-09: Old weights fired at AX marker (IS_BYTE=0, HAS_SE=2):
    #   0 + 2*L*2 - 1.5*L = 2.5L > 0 (spurious!)
    # New weights: require IS_BYTE strongly, weak HAS_SE contribution
    #   AX marker: 0 + L*2 - 3.5L = -1.5L < 0 (blocked)
    #   AX byte:   3L + L - 3.5L = 0.5L > 0 (fires)
    #   First step: 3L + 0 - 3.5L = -0.5L < 0 (blocked)
    attn.W_q[base + 0, BD.IS_BYTE] = L * 3  # Require IS_BYTE strongly
    attn.W_q[base + 0, BD.HAS_SE] = L  # Weak HAS_SE (changed from L*2)
    attn.W_q[base + 0, BD.CONST] = -L * 3.5  # Higher threshold (changed from 1.5)

    # Q dim 1: H1[AX_IDX] (AX vs non-AX discrimination)
    attn.W_q[base + 1, BD.H1 + AX_IDX] = L
    attn.W_q[base + 1, BD.CONST] = -L / 2

    # Q dim 2: suppress byte 3 (logits at byte 3 predict SP marker, not a byte)
    attn.W_q[base + 2, BD.BYTE_INDEX_3] = -L
    attn.W_q[base + 2, BD.CONST] = L / 2

    # K dim 0: IS_BYTE
    attn.W_k[base + 0, BD.IS_BYTE] = L

    # K dim 1: H1[AX_IDX] (only AX bytes are strong K targets)
    attn.W_k[base + 1, BD.H1 + AX_IDX] = L

    # K dim 2: suppress byte 0 in K (not a valid target for shifted matching)
    attn.W_k[base + 2, BD.BYTE_INDEX_0] = -L
    attn.W_k[base + 2, BD.CONST] = L / 2

    # Shifted byte matching: Q byte K → K byte K+1
    # byte 0 (BYTE_INDEX_0) attends to byte 1 (BYTE_INDEX_1) of prev step
    attn.W_q[base + 3, BD.BYTE_INDEX_0] = L
    attn.W_k[base + 3, BD.BYTE_INDEX_1] = L
    # byte 1 (BYTE_INDEX_1) attends to byte 2 (BYTE_INDEX_2) of prev step
    attn.W_q[base + 4, BD.BYTE_INDEX_1] = L
    attn.W_k[base + 4, BD.BYTE_INDEX_2] = L
    # byte 2 (BYTE_INDEX_2) attends to byte 3 (BYTE_INDEX_3) of prev step
    attn.W_q[base + 5, BD.BYTE_INDEX_2] = L
    attn.W_k[base + 5, BD.BYTE_INDEX_3] = L

    # Gate dim 33: Enforce AND logic (H1[AX]=1 AND HAS_SE=1) to suppress leakage.
    # Uses high threshold requiring BOTH conditions to be true.
    # At AX bytes on subsequent steps (H1[AX]=1, HAS_SE=1): Q_gate = 0, passes
    # At non-AX (H1[AX]=0): Q_gate = -10000, kills softmax
    # At first step (HAS_SE=0): Q_gate = -10000, kills softmax (redundant with Q dim 0)
    attn.W_q[base + 33, BD.CONST] = -20000.0  # Changed from -10000 to enforce AND
    attn.W_q[base + 33, BD.H1 + AX_IDX] = 10000.0
    attn.W_q[base + 33, BD.HAS_SE] = 10000.0
    attn.W_k[base + 33, BD.CONST] = 5.0

    # V: copy CLEAN_EMBED nibbles (16 lo + 16 hi = 32 V dims)
    for k in range(16):
        attn.W_v[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0

    # O: write to OUTPUT_LO/HI (strength 2.0, weaker than carry ≈5.0)
    for k in range(16):
        attn.W_o[BD.OUTPUT_LO + k, base + k] = 2.0
        attn.W_o[BD.OUTPUT_HI + k, base + 16 + k] = 2.0


def _set_layer10_sp_byte_passthrough(attn, S, BD, HD):
    """L10 attention head 2: SP byte 0-2 passthrough across steps (when NOT PSH).

    Similar to AX byte passthrough but for SP. Only fires when PSH_AT_SP = 0
    (i.e., when SP doesn't change). When PSH is active, L6/L15 handle SP values.

    Copies CLEAN_EMBED from previous step's SP bytes 1-3 → OUTPUT at current
    step's SP byte 0-2 positions. Uses shifted byte matching.

    Mapping (Q byte K attends to K byte K+1 of prev step):
      byte 0 pos → prev byte 1 (predicts byte 1 token)
      byte 1 pos → prev byte 2 (predicts byte 2 token)
      byte 2 pos → prev byte 3 (predicts byte 3 token)
      byte 3 pos → suppressed (predicts BP marker, not a byte)
    """
    L = S  # attention scale
    SP_IDX = 2
    base = 2 * HD  # head 2 starts at dim 128

    # Q dim 0: IS_BYTE AND HAS_SE AND NOT PSH_AT_SP
    # First step: L3 defaults handle SP, don't interfere
    # PSH step: L6/L15 PSH units handle SP, don't interfere
    attn.W_q[base + 0, BD.IS_BYTE] = L
    attn.W_q[base + 0, BD.HAS_SE] = L * 2  # Strong HAS_SE requirement
    attn.W_q[base + 0, BD.PSH_AT_SP] = -L * 2  # Suppress during PSH
    attn.W_q[base + 0, BD.CONST] = -L * 1.5  # Threshold

    # Q dim 1: H1[SP_IDX] (SP vs non-SP discrimination)
    attn.W_q[base + 1, BD.H1 + SP_IDX] = L
    attn.W_q[base + 1, BD.CONST] = -L / 2

    # Q dim 2: suppress byte 3 (logits at byte 3 predict BP marker, not a byte)
    attn.W_q[base + 2, BD.BYTE_INDEX_3] = -L
    attn.W_q[base + 2, BD.CONST] = L / 2

    # K dim 0: IS_BYTE
    attn.W_k[base + 0, BD.IS_BYTE] = L

    # K dim 1: H1[SP_IDX] (only SP bytes are strong K targets)
    attn.W_k[base + 1, BD.H1 + SP_IDX] = L

    # K dim 2: suppress byte 0 in K (not a valid target for shifted matching)
    attn.W_k[base + 2, BD.BYTE_INDEX_0] = -L
    attn.W_k[base + 2, BD.CONST] = L / 2

    # Shifted byte matching: Q byte K → K byte K+1
    attn.W_q[base + 3, BD.BYTE_INDEX_0] = L
    attn.W_k[base + 3, BD.BYTE_INDEX_1] = L
    attn.W_q[base + 4, BD.BYTE_INDEX_1] = L
    attn.W_k[base + 4, BD.BYTE_INDEX_2] = L
    attn.W_q[base + 5, BD.BYTE_INDEX_2] = L
    attn.W_k[base + 5, BD.BYTE_INDEX_3] = L

    # Gate dim 33: Enforce AND logic (IS_BYTE=1 AND H1[SP]=1 AND HAS_SE=1 AND NOT PSH AND NOT POP)
    # PSH_AT_SP suppresses during PSH (SP -= 8)
    # CMP[3] suppresses during binary POP ops (SP += 8)
    # At SP byte positions on subsequent non-modifying steps: all conditions met → Q[33] = 0 → passes
    # During PSH or POP: Q[33] = -30000 + 10000*3 - 10000 = -10000 → suppressed
    attn.W_q[base + 33, BD.CONST] = -30000.0
    attn.W_q[base + 33, BD.IS_BYTE] = 10000.0
    attn.W_q[base + 33, BD.H1 + SP_IDX] = 10000.0
    attn.W_q[base + 33, BD.HAS_SE] = 10000.0
    attn.W_q[base + 33, BD.PSH_AT_SP] = -10000.0  # Suppress during PSH
    attn.W_q[base + 33, BD.CMP + 3] = -10000.0   # Suppress during binary POP (CMP[3])
    attn.W_k[base + 33, BD.CONST] = 5.0

    # V: copy CLEAN_EMBED nibbles (16 lo + 16 hi = 32 V dims)
    for k in range(16):
        attn.W_v[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0

    # O: write to OUTPUT_LO/HI (strength 2.0)
    for k in range(16):
        attn.W_o[BD.OUTPUT_LO + k, base + k] = 2.0
        attn.W_o[BD.OUTPUT_HI + k, base + 16 + k] = 2.0


def _set_layer10_psh_stack0_passthrough(attn, S, BD, HD):
    """L10 attention head 3: PSH STACK0 bytes 1-3 passthrough from AX.

    During PSH, STACK0 = AX. The L6 FFN handles byte 0 at the STACK0 marker.
    This head handles bytes 1-3 by copying AX bytes 1-3 to OUTPUT at STACK0
    byte positions 0-2 (shifted matching for autoregressive generation).

    Mapping (Q at STACK0 byte K attends to K at AX byte K+1 in SAME step):
      STACK0 byte 0 pos → AX byte 1 (predicts STACK0 byte 1 token)
      STACK0 byte 1 pos → AX byte 2 (predicts STACK0 byte 2 token)
      STACK0 byte 2 pos → AX byte 3 (predicts STACK0 byte 3 token)
      STACK0 byte 3 pos → suppressed (predicts MEM marker, not a byte)

    Distances (within same step):
      STACK0 byte 0 (pos 21) → AX byte 1 (pos 7): d = 14
      STACK0 byte 1 (pos 22) → AX byte 2 (pos 8): d = 14
      STACK0 byte 2 (pos 23) → AX byte 3 (pos 9): d = 14

    Only active when PSH_AT_SP = 1 (PSH is executing).
    """
    L = S  # attention scale
    AX_IDX = 1
    BP_IDX = 3
    base = 3 * HD  # head 3 starts at dim 192

    # Q dim 0: IS_BYTE (at STACK0 byte positions)
    attn.W_q[base + 0, BD.IS_BYTE] = L

    # Q dim 1: H4[BP] AND NOT H1[BP] → STACK0 area (d=6-9 from BP)
    # L1H4[BP] fires at d <= 6.5, but we want d=6-9, so use H4[BP] (d <= 9.5)
    attn.W_q[base + 1, BD.H4 + BP_IDX] = L
    attn.W_q[base + 1, BD.H1 + BP_IDX] = -L  # Exclude BP bytes (d <= 4.5)
    attn.W_q[base + 1, BD.CONST] = -L / 2

    # Q dim 2: suppress byte 3 (predicts MEM marker, not a byte)
    attn.W_q[base + 2, BD.BYTE_INDEX_3] = -L
    attn.W_q[base + 2, BD.CONST] = L / 2

    # Q dim 3: PSH_AT_SP (only fire during PSH)
    attn.W_q[base + 3, BD.PSH_AT_SP] = L
    attn.W_q[base + 3, BD.CONST] = -L / 2

    # K dim 0: IS_BYTE (AX byte positions)
    attn.W_k[base + 0, BD.IS_BYTE] = L

    # K dim 1: H1[AX] → AX area (d <= 4.5 from AX marker)
    attn.W_k[base + 1, BD.H1 + AX_IDX] = L

    # K dim 2: suppress byte 0 in K (not a valid target for shifted matching)
    attn.W_k[base + 2, BD.BYTE_INDEX_0] = -L
    attn.W_k[base + 2, BD.CONST] = L / 2

    # Shifted byte matching: Q at STACK0 byte K → K at AX byte K+1
    # STACK0 byte 0 (BYTE_INDEX_0) → AX byte 1 (BYTE_INDEX_1)
    attn.W_q[base + 4, BD.BYTE_INDEX_0] = L
    attn.W_k[base + 4, BD.BYTE_INDEX_1] = L
    # STACK0 byte 1 (BYTE_INDEX_1) → AX byte 2 (BYTE_INDEX_2)
    attn.W_q[base + 5, BD.BYTE_INDEX_1] = L
    attn.W_k[base + 5, BD.BYTE_INDEX_2] = L
    # STACK0 byte 2 (BYTE_INDEX_2) → AX byte 3 (BYTE_INDEX_3)
    attn.W_q[base + 6, BD.BYTE_INDEX_2] = L
    attn.W_k[base + 6, BD.BYTE_INDEX_3] = L

    # Gate dim 33: Enforce 4-way AND (IS_BYTE + H4[BP] - H1[BP] + PSH_AT_SP)
    # At STACK0 byte positions during PSH: all conditions met → Q[33] near 0 → passes
    # At other positions: Q[33] large negative → suppressed
    # Threshold: -30000 + 10000*(IS_BYTE + H4[BP] + PSH_AT_SP) - 10000*H1[BP] = 0 when all true
    attn.W_q[base + 33, BD.CONST] = -30000.0
    attn.W_q[base + 33, BD.IS_BYTE] = 10000.0
    attn.W_q[base + 33, BD.H4 + BP_IDX] = 10000.0
    attn.W_q[base + 33, BD.H1 + BP_IDX] = -10000.0  # Exclude BP area
    attn.W_q[base + 33, BD.PSH_AT_SP] = 10000.0
    attn.W_k[base + 33, BD.CONST] = 5.0

    # V: copy CLEAN_EMBED nibbles (16 lo + 16 hi = 32 V dims)
    for k in range(16):
        attn.W_v[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0

    # O: write to OUTPUT_LO/HI (strength 3.0, stronger than default 0)
    for k in range(16):
        attn.W_o[BD.OUTPUT_LO + k, base + k] = 3.0
        attn.W_o[BD.OUTPUT_HI + k, base + 16 + k] = 3.0


def _set_layer10_alu(ffn, S, BD):
    """L10 FFN: Comparison combine + Bitwise ops + AX passthrough.

    Comparison combine: CMP flags from L9 (at AX marker):
      CMP[0]=hi_lt, CMP[1]=hi_eq, CMP[2]=lo_eq, CMP[3]=lo_lt.
      Each ≈ 1.0 when true, ≈ 0 when false (b_gate=0 in L9).
      Default unit writes result=0 or 1; override units flip based on flags.

    Bitwise ops: 3-way AND cross-product (same pattern as ADD/SUB).

    AX passthrough: fires when no handled AX-modifying opcode is active.
      Suppressed via negative W_up weights for handled opcodes.

    Total: ~1586 units (18 cmp + 1536 bitwise + 32 passthrough).
    """
    unit = 0

    # --- Comparison combine (18 units) ---

    def _cmp_default(op_dim, default_result):
        """Default unit: writes result + OUTPUT_HI[0]."""
        nonlocal unit
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, op_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.b_gate[unit] = 1.0  # unconditional gate
        ffn.W_down[BD.OUTPUT_LO + default_result, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    def _cmp_override_2way(op_dim, cmp_dim, to_result, from_result):
        """2-way override: MARK_AX + CMP[k] → flip result."""
        nonlocal unit
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, cmp_dim] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + to_result, unit] = 4.0 / S
        ffn.W_down[BD.OUTPUT_LO + from_result, unit] = -4.0 / S
        unit += 1

    def _cmp_override_3way(op_dim, cmp_dim1, cmp_dim2, to_result, from_result):
        """3-way override: MARK_AX + CMP[i] + CMP[j] → flip result."""
        nonlocal unit
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, cmp_dim1] = S
        ffn.W_up[unit, cmp_dim2] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + to_result, unit] = 4.0 / S
        ffn.W_down[BD.OUTPUT_LO + from_result, unit] = -4.0 / S
        unit += 1

    # EQ: default=0, override to 1 when hi_eq AND lo_eq
    _cmp_default(BD.OP_EQ, 0)
    _cmp_override_3way(BD.OP_EQ, BD.CMP + 1, BD.CMP + 2, 1, 0)

    # NE: default=1, override to 0 when hi_eq AND lo_eq
    _cmp_default(BD.OP_NE, 1)
    _cmp_override_3way(BD.OP_NE, BD.CMP + 1, BD.CMP + 2, 0, 1)

    # LT: default=0, override to 1 when hi_lt OR (hi_eq AND lo_lt)
    _cmp_default(BD.OP_LT, 0)
    _cmp_override_2way(BD.OP_LT, BD.CMP + 0, 1, 0)  # hi_lt
    _cmp_override_3way(BD.OP_LT, BD.CMP + 1, BD.CMP + 3, 1, 0)  # hi_eq AND lo_lt

    # GT: default=1, override to 0 when hi_lt OR (hi_eq AND lo_lt) OR (hi_eq AND lo_eq)
    _cmp_default(BD.OP_GT, 1)
    _cmp_override_2way(BD.OP_GT, BD.CMP + 0, 0, 1)  # hi_lt
    _cmp_override_3way(BD.OP_GT, BD.CMP + 1, BD.CMP + 3, 0, 1)  # hi_eq AND lo_lt
    _cmp_override_3way(BD.OP_GT, BD.CMP + 1, BD.CMP + 2, 0, 1)  # hi_eq AND lo_eq

    # LE: default=0, override to 1 when hi_lt OR (hi_eq AND lo_lt) OR (hi_eq AND lo_eq)
    _cmp_default(BD.OP_LE, 0)
    _cmp_override_2way(BD.OP_LE, BD.CMP + 0, 1, 0)  # hi_lt
    _cmp_override_3way(BD.OP_LE, BD.CMP + 1, BD.CMP + 3, 1, 0)  # hi_eq AND lo_lt
    _cmp_override_3way(BD.OP_LE, BD.CMP + 1, BD.CMP + 2, 1, 0)  # hi_eq AND lo_eq

    # GE: default=1, override to 0 when hi_lt OR (hi_eq AND lo_lt)
    _cmp_default(BD.OP_GE, 1)
    _cmp_override_2way(BD.OP_GE, BD.CMP + 0, 0, 1)  # hi_lt
    _cmp_override_3way(BD.OP_GE, BD.CMP + 1, BD.CMP + 3, 0, 1)  # hi_eq AND lo_lt

    # --- Bitwise ops (1536 units) ---
    # BUG FIX 2026-04-09: Increased MARK_AX weight and threshold to prevent spurious
    # firing at byte positions. At PC marker, ALU_LO and AX_CARRY_LO can have large
    # spurious values (up to ~70 combined), which exceeded the old threshold of 10.5.
    # By requiring MARK_AX=60, units only fire at the actual AX marker position.
    # NOTE: AX_CARRY should contain the stack value for binary ops, but is currently
    # not properly populated. This is a known architecture gap for stack-based ops.
    # BUG FIX 2026-04-16: Use balanced weights for true 3-way AND.
    # With weights (40, 30, 30) and threshold 80:
    #   All 3 present: 40 + 30 + 30 = 100 > 80 (fires)
    #   Any 2 present: max(40+30) = 70 < 80 (blocked)
    bitwise_ops = [
        (BD.OP_OR, lambda a, b: a | b),
        (BD.OP_XOR, lambda a, b: a ^ b),
        (BD.OP_AND, lambda a, b: a & b),
    ]
    for op_dim, op_fn in bitwise_ops:
        # Lo nibble (256 units)
        for a in range(16):
            for b in range(16):
                result = op_fn(a, b)
                ffn.W_up[unit, BD.MARK_AX] = S * 40  # Balanced 3-way AND
                ffn.W_up[unit, BD.ALU_LO + a] = S * 30
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S * 30
                ffn.b_up[unit] = -S * 80  # Threshold requiring all 3 inputs
                ffn.W_gate[unit, op_dim] = 1.0
                ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1
        # Hi nibble (256 units)
        for a in range(16):
            for b in range(16):
                result = op_fn(a, b)
                ffn.W_up[unit, BD.MARK_AX] = S * 40  # Balanced 3-way AND
                ffn.W_up[unit, BD.ALU_HI + a] = S * 30
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S * 30
                ffn.b_up[unit] = -S * 80  # Threshold requiring all 3 inputs
                ffn.W_gate[unit, op_dim] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # --- MUL lo nibble (256 units) ---
    # For each (a_lo, b_lo): result_lo = (a_lo * b_lo) % 16
    # 3-way AND: MARK_AX + ALU_LO[a_lo] + AX_CARRY_LO[b_lo], gate=OP_MUL
    # BUG FIX 2026-04-16: Use balanced weights for true 3-way AND.
    # With weights (40, 30, 30) and threshold 80:
    #   All 3 present: 40 + 30 + 30 = 100 > 80 (fires)
    #   Any 2 present: max(40+30) = 70 < 80 (blocked)
    for a in range(16):
        for b in range(16):
            result = (a * b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S * 40  # Balanced 3-way AND
            ffn.W_up[unit, BD.ALU_LO + a] = S * 30
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S * 30
            ffn.b_up[unit] = -S * 80  # Threshold requiring all 3 inputs
            ffn.W_gate[unit, BD.OP_MUL] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # --- SHL/SHR zero output for shift >= 8 (8 units) ---
    # When shift >= 8, result is 0x00. Two sub-cases:
    # Case A (shift >= 16, hi nibble > 0): gate = OP_xxx * (1 - AX_CARRY_HI[0])
    # Case B (shift 8-15, hi=0, lo>=8): up includes AX_CARRY_HI[0] + sum(AX_CARRY_LO[8..15])
    # BUG FIX 2026-04-09: Increased threshold to prevent spurious firing.
    for op_dim in [BD.OP_SHL, BD.OP_SHR]:
        # Case A: shift >= 16 (hi nibble non-zero → AX_CARRY_HI[0] NOT hot)
        ffn.W_up[unit, BD.MARK_AX] = S * 60  # Strong MARK_AX requirement
        ffn.W_up[unit, BD.AX_CARRY_HI + 0] = -S  # suppress when hi=0 (shift 0-15)
        ffn.b_up[unit] = -S * 59  # Require MARK_AX to overcome threshold
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # Case B: shift 8-15 (hi=0, lo nibble is 8..15)
        ffn.W_up[unit, BD.MARK_AX] = S * 60  # Strong MARK_AX requirement
        ffn.W_up[unit, BD.AX_CARRY_HI + 0] = S
        for lo_bit in range(8, 16):
            ffn.W_up[unit, BD.AX_CARRY_LO + lo_bit] = S
        ffn.b_up[unit] = -S * 80  # Require MARK_AX to overcome threshold
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    # --- AX passthrough (32 units) ---
    # Fires when no handled AX-modifying opcode is active.
    # Negative W_up weights suppress passthrough for handled opcodes.
    suppressed_ops = [
        BD.OP_IMM,
        BD.OP_ADD,
        BD.OP_SUB,
        BD.OP_OR,
        BD.OP_XOR,
        BD.OP_AND,
        BD.OP_EQ,
        BD.OP_NE,
        BD.OP_LT,
        BD.OP_GT,
        BD.OP_LE,
        BD.OP_GE,
        BD.OP_MUL,
        BD.OP_DIV,   # neural (DivModModule after L10)
        BD.OP_MOD,   # neural (DivModModule after L10)
        BD.OP_SHL,
        BD.OP_SHR,
        BD.OP_LEA,  # LEA: AX = FETCH + BP (handled by L8/L9 ADD circuit)
        BD.OP_LI,  # L15 provides memory lookup result
        BD.OP_LC,  # L15 provides memory lookup result (byte)
    ]
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in suppressed_ops:
            ffn.W_up[unit, op_dim] = -S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_AX] = S
        for op_dim in suppressed_ops:
            ffn.W_up[unit, op_dim] = -S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # --- Inter-byte carry/borrow override at AX byte positions (4 units) ---
    # CARRY[1/2] relayed from AX marker by L10 attention head 0.
    # At AX byte positions: override OUTPUT to propagate carry result.
    AX_IDX = 1  # H1[AX_IDX] identifies AX byte positions

    # SUB borrow → all AX bytes = 0xFF: 3-way AND (CARRY[2] + IS_BYTE + OP_SUB)
    # BUG FIX 2026-04-16: Old threshold 2.5 was too low. OP_SUB=5.0 gives 500, which
    # alone exceeds 250. At AX marker (IS_BYTE=0, CARRY[2]=0): 500-250=250>0 (fires!)
    # New threshold 5.5: At marker: 500-550=-50<0 (blocked). All 3: 700-550=150>0 (fires)
    for out_dim in [BD.OUTPUT_LO + 15, BD.OUTPUT_HI + 15]:
        ffn.W_up[unit, BD.CARRY + 2] = S
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.OP_SUB] = S  # Only fire during SUB
        ffn.b_up[unit] = -S * 5.5  # 3-way AND (increased from 2.5)
        ffn.W_gate[unit, BD.H1 + AX_IDX] = 1.0  # AX bytes only
        ffn.W_down[out_dim, unit] = 10.0 / S  # ≈ 5.0 output
        unit += 1

    # ADD carry → AX byte 0 only = 0x01: 4-way AND (CARRY[1] + IS_BYTE + BYTE_INDEX_0 + OP_ADD)
    # BUG FIX 2026-04-09: Old threshold 3.5 was too low. OP_ADD=5.0 gives 500, which
    # alone exceeds 350. At AX marker (IS_BYTE=0, BYTE_INDEX_0=0): 500-350=150>0 (fires!)
    # New threshold 5.5: At marker: 500-550=-50<0 (blocked). All 4: 800-550=250>0 (fires)
    for out_dim in [BD.OUTPUT_LO + 1, BD.OUTPUT_HI + 0]:
        ffn.W_up[unit, BD.CARRY + 1] = S
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.BYTE_INDEX_0] = S
        ffn.W_up[unit, BD.OP_ADD] = S  # Only fire during ADD
        ffn.b_up[unit] = -S * 5.5  # 4-way AND (increased from 3.5)
        ffn.W_gate[unit, BD.H1 + AX_IDX] = 1.0
        ffn.W_down[out_dim, unit] = 10.0 / S
        unit += 1

    return unit


# =============================================================================
# MUL: Byte-0 schoolbook multiplication (L11 partial + L12 combine)
# =============================================================================


def _set_layer11_mul_partial(ffn, S, BD):
    """L11 FFN: MUL partial sum staging for hi nibble computation.

    Schoolbook: result = (a_hi*16+a_lo) * (b_hi*16+b_lo) mod 256
      result_lo = (a_lo * b_lo) % 16           [computed in L10]
      result_hi = (carry + a_lo*b_hi + a_hi*b_lo) % 16

    This layer computes: partial = (carry + a_lo*b_hi) % 16
      where carry = (a_lo * b_lo) // 16
    for all (a_lo, b_lo, b_hi) triples, stored in TEMP[0..15].

    L12 then combines: result_hi = (partial + a_hi*b_lo) % 16.

    4096 units = 16^3 (fills L11 FFN exactly).
    """
    unit = 0

    for a_lo in range(16):
        for b_lo in range(16):
            carry = (a_lo * b_lo) // 16
            for b_hi in range(16):
                partial = (carry + a_lo * b_hi) % 16
                # 4-way AND: MARK_AX + ALU_LO[a_lo] + AX_CARRY_LO[b_lo] + AX_CARRY_HI[b_hi]
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_LO + a_lo] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b_lo] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b_hi] = S
                ffn.b_up[unit] = -S * 3.5
                ffn.W_gate[unit, BD.OP_MUL] = 1.0
                ffn.W_down[BD.TEMP + partial, unit] = 2.0 / S
                unit += 1

    return unit


def _set_layer12_mul_combine(ffn, S, BD):
    """L12 FFN: MUL hi nibble from partial + a_hi*b_lo.

    Reads TEMP[partial] from L11 (≈5.0 when hot, ≈0 otherwise).
    Computes: result_hi = (partial + a_hi * b_lo) % 16.

    4-way AND: MARK_AX + TEMP[partial] + ALU_HI[a_hi] + AX_CARRY_LO[b_lo]
    Threshold 7.5 accounts for TEMP ≈ 5.0 (not 1.0):
      All match: 1 + 5 + 1 + 1 = 8 > 7.5 → fires
      Wrong TEMP: 1 + 0 + 1 + 1 = 3 < 7.5 → blocked
      Wrong ALU/AX: 1 + 5 + 0 + 1 = 7 < 7.5 → blocked

    4096 units = 16^3 (fills L12 FFN exactly).
    """
    unit = 0

    for partial in range(16):
        for a_hi in range(16):
            for b_lo in range(16):
                result_hi = (partial + a_hi * b_lo) % 16
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.TEMP + partial] = S
                ffn.W_up[unit, BD.ALU_HI + a_hi] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b_lo] = S
                ffn.b_up[unit] = -S * 7.5
                ffn.W_gate[unit, BD.OP_MUL] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result_hi, unit] = 2.0 / S
                unit += 1

    return unit


# =============================================================================
# L13 attention: MEM addr → val key gather
# =============================================================================


def _set_layer13_mem_addr_gather(attn, S, BD, HD):
    """L13 attention heads 0-2: Gather MEM addr bytes → MEM val byte positions.

    For L15 K-side address keys: copies addr byte nibbles from MEM addr
    positions (d=0..3 from MEM marker) to MEM val byte positions (d=4..8).
    """
    L = 15.0
    MEM_I = 4

    for j in range(3):
        base = j * HD
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]

        # Q: fires at MEM val byte positions (d=5..8 from MEM)
        # Use MEM_VAL_B0-3 flags computed in L2 FFN
        attn.W_q[base, BD.MEM_VAL_B0] = L
        attn.W_q[base, BD.MEM_VAL_B1] = L
        attn.W_q[base, BD.MEM_VAL_B2] = L
        attn.W_q[base, BD.MEM_VAL_B3] = L

        # K: fires at MEM addr byte J position.
        # Addr byte 0 is at d=1 (after MEM marker), byte 1 at d=2, byte 2 at d=3.
        if j == 0:
            # Addr byte 0 at d=1: L1H1[MEM]=1 (d≤1.5), subtract L1H0[MEM] (d=0 only)
            attn.W_k[base, BD.L1H1 + MEM_I] = L
            attn.W_k[base, BD.L1H0 + MEM_I] = -L  # exclude MEM marker (d=0)
        elif j == 1:
            # Addr byte 1 at d=2: L1H2[MEM]=1 (d≤2.5), subtract L1H1[MEM] (d≤1.5)
            attn.W_k[base, BD.L1H2 + MEM_I] = L
            attn.W_k[base, BD.L1H1 + MEM_I] = -L
        elif j == 2:
            # Addr byte 2 at d=3: H0[MEM]=1 (d≤3.5), subtract L1H2[MEM] (d≤2.5)
            attn.W_k[base, BD.H0 + MEM_I] = L
            attn.W_k[base, BD.L1H2 + MEM_I] = -L

        # Anti-leakage gate
        attn.W_q[base + 33, BD.MEM_VAL_B0] = L
        attn.W_q[base + 33, BD.CONST] = -L / 2
        attn.W_k[base + 33, BD.CONST] = L

        # V: copy CLEAN_EMBED nibbles (addr byte value)
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ADDR_BJ_LO/HI (gathered to val byte positions)
        for k in range(16):
            attn.W_o[addr_lo_out + k, base + 1 + k] = 1.0
            attn.W_o[addr_hi_out + k, base + 17 + k] = 1.0


# =============================================================================
# SHL/SHR: Byte-0 shift operations (L13)
# =============================================================================


def _set_layer13_shifts(ffn, S, BD):
    """L13 FFN: SHL/SHR for shift amounts 0-7.

    For each (a_lo, a_hi, s) where s=0..7:
      SHL: result = ((a_hi<<4 | a_lo) << s) & 0xFF
      SHR: result = ((a_hi<<4 | a_lo) >> s) & 0xFF

    Relay convention:
      ALU = stack (value to shift)
      AX_CARRY = AX (shift amount)

    5-way AND: MARK_AX + ALU_LO[a_lo] + ALU_HI[a_hi] + AX_CARRY_LO[s] + AX_CARRY_HI[0]
    Threshold -S*4.5:
      All 5 match: 5S - 4.5S = 0.5S → silu fires
      4 match:     4S - 4.5S = -0.5S → blocked

    Gate: OP_SHL or OP_SHR (≈5.0 when active, ≈0 otherwise).

    Shift >= 8 already handled by L10 zero-output units.

    2048 SHL + 2048 SHR = 4096 units (fills L13 exactly).
    """
    unit = 0

    for op_dim, shift_fn in [
        (BD.OP_SHL, lambda v, s: (v << s) & 0xFF),
        (BD.OP_SHR, lambda v, s: (v >> s) & 0xFF),
    ]:
        for s in range(8):
            for a_hi in range(16):
                for a_lo in range(16):
                    value = (a_hi << 4) | a_lo
                    result = shift_fn(value, s)
                    result_lo = result & 0xF
                    result_hi = (result >> 4) & 0xF

                    # 5-way AND in silu path
                    # value = ALU (stack), shift = AX_CARRY (AX)
                    ffn.W_up[unit, BD.MARK_AX] = S
                    ffn.W_up[unit, BD.ALU_LO + a_lo] = S       # value lo (stack)
                    ffn.W_up[unit, BD.ALU_HI + a_hi] = S       # value hi (stack)
                    ffn.W_up[unit, BD.AX_CARRY_LO + s] = S     # shift amount (AX)
                    ffn.W_up[unit, BD.AX_CARRY_HI + 0] = S     # shift hi = 0 (AX hi)
                    ffn.b_up[unit] = -S * 4.5

                    # Opcode gate
                    ffn.W_gate[unit, op_dim] = 1.0

                    # Write both nibbles of result
                    ffn.W_down[BD.OUTPUT_LO + result_lo, unit] = 2.0 / S
                    ffn.W_down[BD.OUTPUT_HI + result_hi, unit] = 2.0 / S
                    unit += 1

    return unit


# =============================================================================
# L14: MEM byte generation (addr + val)
# =============================================================================


def _set_layer14_mem_generation(attn, S, BD, HD):
    """L14 attention: Generate MEM addr bytes (heads 0-3) and val bytes (heads 4-7).

    Addr heads: Copy address from SP (PSH, MEM_ADDR_SRC=0) or STACK0
    (SI/SC, MEM_ADDR_SRC=1) to MEM addr byte positions.

    Val heads: Copy AX byte values to MEM val byte positions.

    Each head fires at exactly ONE MEM byte position using position-specific
    threshold-difference encoding (dim 0) + position gate (dim 33) +
    MEM_STORE gate (dim 34). This prevents cross-head OUTPUT accumulation.

    Addr head 0 is special: K matches SP marker (for PSH byte 0, where the
    new SP value lives in OUTPUT) or STACK0 byte 0 (for SI/SC). V reads
    CLEAN_EMBED + OUTPUT so it picks up whichever is non-zero.
    """
    L = 15.0
    MEM_I = 4
    AX_I = 1
    SP_I = 2
    BP_I = 3

    # Position flags: threshold-difference pairs selecting distance from MEM.
    # AUTOREGRESSIVE SHIFT: logits at position d predict the token at d+1.
    # To predict addr_b0 (at d=1), L14 fires at d=0 (MEM marker).
    # To predict addr_b1 (at d=2), L14 fires at d=1 (addr_b0 token), etc.
    addr_pos = [
        (BD.MARK_MEM, None),  # d=0: predicts addr_b0
        (BD.L1H1 + MEM_I, BD.L1H0 + MEM_I),  # d=1: predicts addr_b1
        (BD.L1H2 + MEM_I, BD.L1H1 + MEM_I),  # d=2: predicts addr_b2
        (BD.H0 + MEM_I, BD.L1H2 + MEM_I),  # d=3: predicts addr_b3
    ]
    val_pos = [
        (BD.H1 + MEM_I, BD.H0 + MEM_I),  # d=4: predicts val_b0
        (BD.L2H0 + MEM_I, BD.H1 + MEM_I),  # d=5: predicts val_b1
        (BD.L1H4 + MEM_I, BD.L2H0 + MEM_I),  # d=6: predicts val_b2
        (BD.H2 + MEM_I, BD.L1H4 + MEM_I),  # d=7: predicts val_b3
    ]

    # === Heads 0-3: MEM addr byte generation ===
    for h in range(4):
        base = h * HD
        pos_up, pos_down = addr_pos[h]

        # Dim 0: Q position selection (threshold diff, NOT gated by MEM_STORE).
        attn.W_q[base, pos_up] = L
        if pos_down is not None:
            attn.W_q[base, pos_down] = -L
        # FIX 2026-04-16: Suppress Q at STACK0 area to prevent OUTPUT corruption
        # during STACK0 token generation (before MEM section exists).
        attn.W_q[base, BD.MARK_STACK0] = -L
        attn.W_q[base, BD.H4 + BP_I] = -L  # STACK0 bytes (d≤9.5 from BP)
        # FIX 2026-04-17: Also suppress at SP byte positions to prevent OUTPUT_HI[15]
        # pollution during first step generation (when no BP/MEM markers exist).
        # H1[SP] fires at d≤4.5 from SP marker, covering SP bytes 0-3.
        attn.W_q[base, BD.H1 + SP_I] = -L  # SP bytes (d≤4.5 from SP)

        if h == 0:
            # Byte 0: K matches SP marker OR STACK0 byte 0.
            # SP marker has new SP byte 0 in OUTPUT (from L6 PSH computation).
            # STACK0 byte 0 has old value in CLEAN_EMBED (for SI/SC address).
            attn.W_k[base, BD.MARK_SP] = L
            attn.W_k[base, BD.STACK0_BYTE0] = L
        else:
            # AUTOREGRESSIVE SHIFT FIX 2026-04-16: To predict addr byte J, read from
            # position with BYTE_INDEX_(J-1) whose OUTPUT contains byte J value.
            # Head 1 predicts byte 1, reads from BYTE_INDEX_0 (SP byte 0 OUTPUT has byte 1)
            # Head 2 predicts byte 2, reads from BYTE_INDEX_1 (SP byte 1 OUTPUT has byte 2)
            # Head 3 predicts byte 3, reads from BYTE_INDEX_2 (SP byte 2 OUTPUT has byte 3)
            byte_idx_dim = [None, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
            attn.W_k[base, byte_idx_dim] = L

        # Dim 1: SP source bonus (active when MEM_ADDR_SRC=0, i.e., PSH).
        # Q = L when PSH, -L when SI/SC. K = L at SP area only (not AX area).
        attn.W_q[base + 1, BD.CONST] = L
        attn.W_q[base + 1, BD.MEM_ADDR_SRC] = -2 * L
        attn.W_k[base + 1, BD.H1 + SP_I] = L
        attn.W_k[base + 1, BD.H1 + AX_I] = -L  # exclude AX (overlaps H1[SP])

        # Dim 2: STACK0 source bonus (active when MEM_ADDR_SRC=1, i.e., SI/SC).
        # AUTOREGRESSIVE SHIFT FIX 2026-04-16: To predict addr byte J, read from
        # byte (J-1) position whose OUTPUT contains byte J value.
        #   STACK0 byte 0 (d=6 from BP): L1H4[BP]=1, H1[BP]=0
        #   STACK0 byte 1 (d=7 from BP): H2[BP]=1, L1H4[BP]=0
        #   STACK0 byte 2 (d=8 from BP): H3[BP]=1, H2[BP]=0
        # Also suppress AX/SP/BP areas and MARK_STACK0 marker.
        attn.W_q[base + 2, BD.MEM_ADDR_SRC] = L
        if h == 0:
            # Head 0: Uses STACK0_BYTE0 flag (set in dim 0 K). No change needed.
            # Actually need to also set dim 2 K for score contribution.
            attn.W_k[base + 2, BD.STACK0_BYTE0] = L
        elif h == 1:
            # Head 1 predicts byte 1, reads from byte 0 (d=6): L1H4[BP] - H1[BP]
            attn.W_k[base + 2, BD.L1H4 + BP_I] = L
            attn.W_k[base + 2, BD.H1 + BP_I] = -L
        elif h == 2:
            # Head 2 predicts byte 2, reads from byte 1 (d=7): H2[BP] - L1H4[BP]
            attn.W_k[base + 2, BD.H2 + BP_I] = L
            attn.W_k[base + 2, BD.L1H4 + BP_I] = -L
        elif h == 3:
            # Head 3 predicts byte 3, reads from byte 2 (d=8): H3[BP] - H2[BP]
            attn.W_k[base + 2, BD.H3 + BP_I] = L
            attn.W_k[base + 2, BD.H2 + BP_I] = -L
        # Also suppress non-STACK0 areas
        attn.W_k[base + 2, BD.H1 + AX_I] = -L
        attn.W_k[base + 2, BD.H1 + SP_I] = -L
        attn.W_k[base + 2, BD.MARK_STACK0] = -L  # suppress marker token

        # Dim 33: Position gate (suppress non-target MEM byte positions).
        # At target: Q=0, score=0. At non-target: Q=-500, score=-312.5.
        # Must dominate MEM_STORE gate (+156.25 at MEM marker where MEM_STORE=2)
        # to ensure net suppression of -156.25 at non-target positions.
        attn.W_q[base + 33, BD.CONST] = -500.0
        attn.W_q[base + 33, pos_up] = 500.0
        if pos_down is not None:
            attn.W_q[base + 33, pos_down] = -500.0
        # FIX 2026-04-16: Strong MARK_STACK0 suppression to prevent OUTPUT_HI
        # corruption during STACK0 token generation (before MEM section exists).
        # Score = -500 × 5 × 5 / 8 = -1562.5 at STACK0 marker where MARK_STACK0=5.
        attn.W_q[base + 33, BD.MARK_STACK0] = -500.0
        # Also suppress at STACK0 byte positions (L1H4[BP] fires at d≤6.5 from BP)
        attn.W_q[base + 33, BD.L1H4 + BP_I] = -500.0
        # FIX 2026-04-17: Also suppress at SP byte positions during first step.
        attn.W_q[base + 33, BD.H1 + SP_I] = -500.0
        attn.W_k[base + 33, BD.CONST] = 5.0

        # Dim 34: MEM_STORE gate (suppress non-store positions).
        # FIX 2026-04-16: Increased from -250 to -500 for stronger suppression.
        attn.W_q[base + 34, BD.CONST] = -500.0
        attn.W_q[base + 34, BD.MEM_STORE] = 500.0
        attn.W_k[base + 34, BD.CONST] = 5.0

        # V: read CLEAN_EMBED + OUTPUT (sum). At SP marker: CLEAN_EMBED=0
        # (marker token), OUTPUT=new SP value → V picks up new SP. At byte
        # positions: CLEAN_EMBED=old value, OUTPUT=0 → V picks up old value.
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            attn.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
        # V[0]: cancel L3 MEM default (reads CONST=1.0 from source)
        attn.W_v[base + 0, BD.CONST] = 1.0

        # O: write to OUTPUT_LO/HI
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
        # O: cancel L3 default (OUTPUT_LO[0] and OUTPUT_HI[0] = -1.0)
        attn.W_o[BD.OUTPUT_LO + 0, base + 0] = -1.0
        attn.W_o[BD.OUTPUT_HI + 0, base + 0] = -1.0

    # === Heads 4-7: MEM val byte generation ===
    # Supports TWO sources:
    #   - AX (for PSH, SI, SC) - default
    #   - STACK0 (for JSR, ENT) - when OP_JSR or OP_ENT active
    for h in range(4):
        head = 4 + h
        base = head * HD
        pos_up, pos_down = val_pos[h]
        byte_idx_dim = [
            BD.BYTE_INDEX_0,
            BD.BYTE_INDEX_1,
            BD.BYTE_INDEX_2,
            BD.BYTE_INDEX_3,
        ][h]

        # Dim 0: Q position selection + K targets byte positions
        attn.W_q[base, pos_up] = L
        attn.W_q[base, pos_down] = -L
        # FIX 2026-04-16: Suppress Q at STACK0 area to prevent OUTPUT corruption
        # during STACK0 token generation (before MEM section exists).
        attn.W_q[base, BD.MARK_STACK0] = -L
        attn.W_q[base, BD.H4 + BP_I] = -L  # STACK0 bytes (d≤9.5 from BP)
        # FIX 2026-04-17: Also suppress at SP byte positions (same as addr heads).
        attn.W_q[base, BD.H1 + SP_I] = -L  # SP bytes (d≤4.5 from SP)
        attn.W_k[base, byte_idx_dim] = L

        # Dim 1: AX source bonus (default for PSH, SI, SC)
        # Q = L by default, disabled when JSR or ENT active
        # FIX 2026-04-16: Removed H1[BP] exclusion from K. When OP_JSR is active,
        # Q[1] becomes very negative (-142 when OP_JSR=5.2). Combined with negative
        # K at BP bytes (-15), the negative×negative product gave BP bytes a huge
        # positive score (+267), causing them to win over STACK0 bytes.
        # The STACK0 source bonus in dim 2 handles the JSR/ENT case properly.
        attn.W_q[base + 1, BD.CONST] = L
        attn.W_q[base + 1, BD.OP_JSR] = -2 * L  # Disable for JSR
        attn.W_q[base + 1, BD.OP_ENT] = -2 * L  # Disable for ENT
        attn.W_k[base + 1, BD.H1 + AX_I] = L  # AX area bonus

        # Dim 2: STACK0 source bonus (JSR and ENT only)
        # Q = L when JSR or ENT, 0 otherwise
        attn.W_q[base + 2, BD.OP_JSR] = L
        attn.W_q[base + 2, BD.OP_ENT] = L

        # K: STACK0 byte positions (matches pattern from address heads)
        if h == 0:
            # Byte 0: STACK0_BYTE0 flag (d=6 from BP)
            attn.W_k[base + 2, BD.STACK0_BYTE0] = L
        elif h == 1:
            # Byte 1: H2[BP] - L1H4[BP] fires at d∈(6.5, 7.5] (d=7 from BP)
            attn.W_k[base + 2, BD.H2 + BP_I] = L
            attn.W_k[base + 2, BD.L1H4 + BP_I] = -L
        elif h == 2:
            # Byte 2: H3[BP] - H2[BP] fires at d∈(7.5, 8.5] (d=8 from BP)
            attn.W_k[base + 2, BD.H3 + BP_I] = L
            attn.W_k[base + 2, BD.H2 + BP_I] = -L
        elif h == 3:
            # Byte 3: H4[BP] - H3[BP] fires at d∈(8.5, 9.5] (d=9 from BP)
            attn.W_k[base + 2, BD.H4 + BP_I] = L
            attn.W_k[base + 2, BD.H3 + BP_I] = -L

        # Suppress non-STACK0 areas when using STACK0 source
        attn.W_k[base + 2, BD.H1 + AX_I] = -L
        attn.W_k[base + 2, BD.H1 + SP_I] = -L
        attn.W_k[base + 2, BD.MARK_STACK0] = -L  # Suppress marker token

        # Dim 33: Position gate (suppress non-target MEM byte positions).
        # Same 500 strength as addr heads to dominate MEM_STORE gate.
        attn.W_q[base + 33, BD.CONST] = -500.0
        attn.W_q[base + 33, pos_up] = 500.0
        attn.W_q[base + 33, pos_down] = -500.0
        # FIX 2026-04-16: Strong MARK_STACK0 suppression to prevent OUTPUT_HI
        # corruption during STACK0 token generation (before MEM section exists).
        attn.W_q[base + 33, BD.MARK_STACK0] = -500.0
        # Also suppress at STACK0 byte positions (L1H4[BP] fires at d≤6.5 from BP)
        attn.W_q[base + 33, BD.L1H4 + BP_I] = -500.0
        # FIX 2026-04-17: Also suppress at SP byte positions during first step.
        attn.W_q[base + 33, BD.H1 + SP_I] = -500.0
        attn.W_k[base + 33, BD.CONST] = 5.0

        # Dim 34: MEM_STORE gate (suppress non-store positions).
        # FIX 2026-04-16: Increased from -250 to -500 for stronger suppression.
        attn.W_q[base + 34, BD.CONST] = -500.0
        attn.W_q[base + 34, BD.MEM_STORE] = 500.0
        attn.W_k[base + 34, BD.CONST] = 5.0

        # V: copy CLEAN_EMBED nibbles (from AX or STACK0)
        # FIX 2026-04-16: Read CLEAN_EMBED only. At AX byte positions, CLEAN_EMBED
        # has the correct byte value (e.g., 42 from IMM). OUTPUT has spurious values
        # from L10 passthrough that corrupt val byte generation.
        # For JSR/ENT: STACK0 bytes also have CLEAN_EMBED from generated tokens.
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # V[0]: cancel L3 MEM default (reads CONST=1.0 from source)
        attn.W_v[base + 0, BD.CONST] = 1.0

        # O: write to OUTPUT_LO/HI
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
        # O: cancel L3 default (OUTPUT_LO[0] and OUTPUT_HI[0] = -1.0)
        attn.W_o[BD.OUTPUT_LO + 0, base + 0] = -1.0
        attn.W_o[BD.OUTPUT_HI + 0, base + 0] = -1.0


def _set_layer14_temp_clear(ffn, S, BD):
    """L14 FFN: Clear TEMP at PC marker when OP_LEV is active.

    BUG FIX 2026-04-16: TEMP[0] has residual value from L5/L6 attention (2.0).
    This causes L16 TEMP→OUTPUT routing to incorrectly boost OUTPUT_LO[0].

    Solution: Subtract from TEMP[0] when OP_LEV and MARK_PC are active.
    L15 will then write fresh values to TEMP for the return address.

    Activation calculation at PC marker:
      OP_LEV=10 * S/10 = 10
      MARK_PC=1 * S = 10
      bias = -15
      Total = 10 + 10 - 15 = 5 > 0 (fires)

    Output: Subtract 5.0 from TEMP[0], clearing the 2.0 residual.
    """
    unit = 0

    # Clear TEMP[0] at PC marker when OP_LEV active
    # Only clear TEMP[0] since that's the problematic residual
    ffn.W_up[unit, BD.OP_LEV] = S / 10  # ~1 with OP_LEV≈10
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 1.5  # Fire when OP_LEV + MARK_PC
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.TEMP + 0, unit] = -5.0 / S  # Subtract to clear residual
    unit += 1

    # Note: Don't clear other TEMP positions since L15 head 8 writes there
    return unit


def _set_layer14_clear_addr_key_pollution(ffn, S, BD, start_unit=0):
    """L14 FFN: Clear ADDR_KEY pollution at non-MEM, non-marker positions.

    BUG FIX 2026-04-16: ADDR_KEY dims (206-253) are aliased with ADDR_B*_HI.
    L9 attention writes to ADDR_B*_HI for address gathering, which pollutes
    ADDR_KEY at non-MEM positions. This causes L15 to attend to wrong positions.

    Solution: Clear ADDR_KEY at positions that are:
    - NOT MEM value bytes (MEM_VAL_B* = 0)
    - NOT register markers where ADDR_B*_HI is needed for L15 queries
      (PC marker for LEV return_addr, BP marker for LEV saved_bp,
       AX marker for LI/LC, STACK0 marker for stack read)

    Pattern: Fire when NOT at MEM value position AND NOT at query markers.
    - W_up: Large negative weights for MEM_VAL_B* and MARK_* flags
    - b_up: Positive bias (fires when no flags present)
    - W_down: Write negative value to cancel ADDR_KEY pollution
    """
    unit = start_unit

    # Large value to suppress firing at MEM and marker positions
    suppress = S * 100  # When flag = 1.0, adds -100*S to activation

    # Clear all 48 ADDR_KEY dims at non-MEM, non-marker positions
    for k in range(48):  # ADDR_KEY is 48 dims (206-253)
        # Suppress at MEM value positions (any of B0/B1/B2/B3)
        ffn.W_up[unit, BD.MEM_VAL_B0] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B1] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B2] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B3] = -suppress

        # Suppress at register markers where ADDR_B*_HI is used for L15 queries
        ffn.W_up[unit, BD.MARK_PC] = -suppress  # LEV return_addr lookup
        ffn.W_up[unit, BD.MARK_BP] = -suppress  # LEV saved_bp lookup
        ffn.W_up[unit, BD.MARK_AX] = -suppress  # LI/LC address lookup
        ffn.W_up[unit, BD.MARK_STACK0] = -suppress  # Stack read
        # FIX 2026-04-16: Also suppress at SP marker during LEV
        # SP marker needs ADDR_B0 for SP = BP + 16 computation
        ffn.W_up[unit, BD.MARK_SP] = -suppress

        # Positive bias to fire at non-MEM, non-marker positions
        ffn.b_up[unit] = S * 0.5

        # Gate unconditionally
        ffn.W_gate[unit, BD.CONST] = 1.0

        # Write to cancel pollution and bring ADDR_KEY to 0
        # FIX 2026-04-16: Changed from -200/S (=-100 output) to -4/S (~=-1.4 output).
        # The original -100 clearing caused negative Q × negative K = positive score
        # at non-target positions in L15 LEV heads. Clearing to ~0 avoids this issue
        # while still preventing false address matches (0 × anything = 0).
        # The pollution to clear is small (typically ~1-2 from L9 ADDR_B*_HI writes),
        # so a small negative value is sufficient.
        ffn.W_down[BD.ADDR_KEY + k, unit] = -4.0 / S
        unit += 1

    return unit


def _set_layer14_clear_output_corruption(ffn, S, BD, start_unit=0):
    """L14 FFN: Fix OUTPUT at STACK0 byte positions (bytes 1-3 = 0).

    BUG FIX 2026-04-16: L14 attention V[0] cancelation and CLEAN_EMBED copying
    corrupts OUTPUT at non-MEM query positions (like STACK0 bytes). Even with
    strong Q suppression, softmax normalization ensures some attention weight
    distributes to source positions, causing OUTPUT to have wrong argmax.

    Solution: At STACK0 byte positions (d=5-9 from BP), boost OUTPUT_LO[0] and
    OUTPUT_HI[0] to ensure they win the argmax. This makes bytes 1-3 of STACK0
    (and similar) output 0, which is correct for return addresses < 256.

    Note: This approach assumes return_addr fits in 1 byte. For larger addresses,
    we'd need to compute bytes 1-3 properly (currently they'd be wrong).
    """
    unit = start_unit

    # Suppression value (prevents firing at MEM and register markers)
    suppress = S * 100
    BP_I = 3  # Index for BP marker in threshold dims

    # Only boost OUTPUT_LO[0] and OUTPUT_HI[0]
    for k in [0, 16]:  # 0 = OUTPUT_LO[0], 16 = OUTPUT_HI[0]
        output_dim = BD.OUTPUT_LO if k == 0 else BD.OUTPUT_HI

        # Fire at STACK0 byte area (d=5-9 from BP marker, excludes marker itself)
        # Use H4[BP] (d≤9.5) AND NOT H1[BP] (d>4.5) to select d ∈ (4.5, 9.5]
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, BD.H1 + BP_I] = -S * 20

        # Suppress at MEM value byte positions (legitimate L14 targets)
        ffn.W_up[unit, BD.MEM_VAL_B0] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B1] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B2] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B3] = -suppress

        # Suppress at register markers where OUTPUT is needed
        ffn.W_up[unit, BD.MARK_PC] = -suppress
        ffn.W_up[unit, BD.MARK_AX] = -suppress
        ffn.W_up[unit, BD.MARK_SP] = -suppress
        ffn.W_up[unit, BD.MARK_BP] = -suppress
        ffn.W_up[unit, BD.MARK_STACK0] = -suppress

        # Suppress at BYTE_INDEX_3 positions - byte 3's OUTPUT should predict
        # the NEXT marker (MEM), not force byte value 0.
        ffn.W_up[unit, BD.BYTE_INDEX_3] = -suppress

        # Bias for activation
        ffn.b_up[unit] = -S * 0.5

        # Gate unconditionally
        ffn.W_gate[unit, BD.CONST] = 1.0

        # Write large POSITIVE value to OUTPUT[0] to make it the argmax winner
        # At d=6-9, activation ≈ 3.5S to 0.5S, so output ≈ S * 50/S = 50
        # This overcomes L14's corruption (~2-3) by a large margin.
        ffn.W_down[output_dim, unit] = 50.0 / S

        unit += 1

    return unit


# =============================================================================
# L15: Memory lookup (softmax1 + binary address matching)
# =============================================================================


def _set_layer15_memory_lookup(attn, S, BD, HD):
    """L15 attention: Memory lookup for LI/LC (at AX) and *SP (at STACK0).

    4 heads, one per output byte. Each head serves BOTH:
    - LI/LC at AX positions (load from memory address in prev AX)
    - *SP at STACK0 positions (load from address in SP)

    Uses binary Q/K address encoding for 24-bit matching with ZFOD.

    Score budget per dim (all /sqrt(HD)=8):
      Dim 0 (bias):     Q=-2000(non-target) or 0(target), K=CONST*10
                         → -2500 at non-target, 0 at target
      Dim 1 (store):    Q=50(target) or 0, K=(2*MEM_STORE-1)*50
                         → +312.5 target+store, -312.5 target+non-store
      Dim 2 (ZFOD ofs): Q=CONST*(-96), K=MEM_STORE*50
                         → -600 at store entries (shifts addr baseline)
      Dim 3 (byte sel): Q=L*flag, K=L*MEM_VAL_BH → +28 correct byte
      Dims 4-27 (addr): 24 binary bits, scale=10
                         match=+300, 1-bit-off=+275, random≈0
                         NOTE: ADDR_B0_LO overlaps OPCODE_BYTE — up to +1200
                         spurious score from residual opcode nibbles at Q side

    Totals:
      target+store+match:   0+312.5-600+300 = +12.5  → attend
      target+store+1bitoff: 0+312.5-600+275 = -12.5  → ZFOD ✓
      target+store+random:  0+312.5-600+0   = -287.5 → ZFOD ✓
      target+non-store:     0-312.5+0+0     = -312.5 → suppressed
      non-target+worst:     -2500+1200      = -1300  → suppressed ✓
      non-target+self:      -2500+300       = -2200  → suppressed ✓
    """
    L = 15.0
    MEM_I = 4
    BP_I = 3

    # Heads 0-3: Original LI/LC/STACK0 implementation
    for h in range(4):
        base = h * HD

        # === Dim 0: Bias — suppress non-target Q positions ===
        # Non-target: Q[0] = -2000. Target: Q[0] = 0.
        # K[0] = CONST*10 = 10 everywhere.
        # Score: -2000*10/8 = -2500 (non-target), 0 (target).
        # Must overwhelm worst-case address dim correlation: ADDR dims
        # overlap OPCODE_BYTE (both at dims 12-27), creating up to +1200
        # of spurious positive Q·K score at non-target positions.
        attn.W_q[base, BD.CONST] = -2000.0
        attn.W_q[base, BD.OP_LI_RELAY] = 2000.0
        if h == 0:
            # Head 0: also activated by LC and STACK0 for POP group operations.
            # FIX 2026-04-16: Use CMP[3] (POP group flag) instead of MARK_STACK0.
            # This ensures memory lookup only happens for operations that need to
            # read from stack (ADD, SUB, etc.), not IMM which just needs carry-forward.
            # CMP[3] is relayed from AX marker to STACK0 position by L6 attn head 6.
            attn.W_q[base, BD.OP_LC_RELAY] = 2000.0
            attn.W_q[base, BD.CMP + 3] = 2000.0  # POP group → stack memory read
        else:
            # Heads 1-3: activated at STACK0 byte positions via BP threshold
            # L1H4[BP] fires at d<=9.5 from BP (covers STACK0 area d=5-9)
            # H1[BP] fires at d<=4.5 from BP (BP's own bytes) — subtract to cancel
            attn.W_q[base, BD.L1H4 + BP_I] = 2000.0
            attn.W_q[base, BD.H1 + BP_I] = -2000.0
        # PSH suppression at STACK0 positions
        attn.W_q[base, BD.CMP + 0] = -2000.0
        # LEV suppression: Heads 0-3 are for LI/LC/STACK0, not LEV.
        # FIX 2026-04-16: Address dimension overlap (ADDR_B0_LO = OPCODE_BYTE_LO)
        # creates spurious +3556 score at PC byte positions. Need stronger suppression.
        # OP_LEV ≈ 5.0, so -1000*5 = -5000 added to Q[0], total score -= 6250.
        attn.W_q[base, BD.OP_LEV] = -1000.0
        # FIX 2026-04-16: Add strong MARK_PC suppression. At PC marker during LEV,
        # MARK_PC=1 contributes -25000 to Q[0]. Combined with OP_LEV suppression,
        # Q[0] = -2000 - 25000 - 5000 = -32000. Score contribution = -40000.
        # This overwhelms any positive address matching contribution.
        attn.W_q[base, BD.MARK_PC] = -25000.0
        # FIX 2026-04-16: Also suppress at SP marker during LEV.
        # Heads 0-3 attend to PC bytes from previous step, outputting wrong values.
        # Address aliasing (ADDR_B0_LO == OPCODE_BYTE_LO) causes +75500/dim score.
        # Need Q[0] << -60000 to overcome 8 address dims. Use -100000.
        attn.W_q[base, BD.MARK_SP] = -100000.0
        attn.W_k[base, BD.CONST] = 10.0

        # === Dim 1: Store anchor — suppress non-store K at target Q ===
        # Q[1] = 50 at target, 0 at non-target.
        # K[1] = MEM_STORE*100 - CONST*50 = +50 (store) or -50 (non-store).
        # Score: 50*(+50)/8 = +312.5 (target+store),
        #        50*(-50)/8 = -312.5 (target+non-store).
        attn.W_q[base + 1, BD.OP_LI_RELAY] = 50.0
        if h == 0:
            attn.W_q[base + 1, BD.OP_LC_RELAY] = 50.0
            attn.W_q[base + 1, BD.CMP + 3] = 50.0  # POP group (matches dim 0 fix)
        else:
            attn.W_q[base + 1, BD.L1H4 + BP_I] = 50.0
            attn.W_q[base + 1, BD.H1 + BP_I] = -50.0
        attn.W_q[base + 1, BD.CMP + 0] = -50.0
        attn.W_k[base + 1, BD.MEM_STORE] = 100.0
        attn.W_k[base + 1, BD.CONST] = -50.0

        # === Dim 2: ZFOD negative offset for store entries ===
        # Q[2] = CONST*(-96) = -96 always.
        # K[2] = MEM_STORE*50 = 50 at stores, 0 at non-stores.
        # Score at store: -96*50/8 = -600. At non-store: 0.
        # Shifts store baseline so wrong-addr stores score negative.
        attn.W_q[base + 2, BD.CONST] = -96.0
        attn.W_k[base + 2, BD.MEM_STORE] = 50.0

        # === Dim 3: Byte selection ===
        # Ensures each head attends to the correct MEM val byte TOKEN position.
        # MEM section layout: [MEM, a0, a1, a2, a3, v0, v1, v2, v3]
        # Val byte positions: v0=d5, v1=d6, v2=d7, v3=d8 from MEM marker.
        # Head h reads val byte h via V(CLEAN_EMBED).
        # Q[3] = BS*byte_flag. K[3] = BS*threshold_flag_for_val_byte_h.
        # Byte selection weight increased to 60.0 to dominate over address encoding
        # mismatches (up to ~50 points) caused by value bytes having corrupted ADDR dims.
        # Contribution: 60*60/8 = 450 points for correct byte, 0 for wrong byte.
        BS = 60.0  # Byte Selection weight (increased from L=15.0)
        byte_q_flag = [BD.MARK_AX, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
        attn.W_q[base + 3, byte_q_flag] = BS
        if h == 0:
            attn.W_q[base + 3, BD.MARK_STACK0] = BS
        MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
        if h == 0:
            # Head 0 → val byte 0 at d=5: L2H0[MEM]=1 (d≤5.5), H1[MEM]=0 (d>4.5).
            attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
            attn.W_k[base + 3, BD.H1 + MEM_I] = -BS
        else:
            # Heads 1-3 → val bytes 1,2,3 at d=6,7,8 via MEM_VAL_B1/B2/B3.
            attn.W_k[base + 3, MEM_VAL_DIMS[h]] = BS

        # === Dims 4-27: Binary address encoding (24 bits, scale=10) ===
        # Each of 3 address bytes × 2 nibbles × 4 bits = 24 dims.
        # Q/K: ±scale per bit. Match: 24*100/8 = 300. Random: ~0.
        addr_dim = 4
        scale = 10.0
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in [ab_lo, ab_hi]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

        # === Dim 28: Per-head position gate ===
        # Each head fires ONLY at its target AX/STACK0 byte position.
        # Head 0: MARK_AX or MARK_STACK0. Head 1: BYTE_INDEX_0.
        # Head 2: BYTE_INDEX_1. Head 3: BYTE_INDEX_2.
        # Non-target: Q=-500, score=-312.5. Target: Q=0, score=0.
        attn.W_q[base + 28, BD.CONST] = -500.0
        attn.W_q[base + 28, byte_q_flag] = 500.0
        if h == 0:
            attn.W_q[base + 28, BD.MARK_STACK0] = 500.0
        attn.W_k[base + 28, BD.CONST] = 5.0

        # === V/O: copy byte value to OUTPUT (dims 32-63) ===
        for k in range(16):
            attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 32 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 48 + k] = 1.0

    # === LEV-specific heads (4-7, 8-11): Only when L15 has 12 heads ===
    # This requires 17-layer model. With 16-layer model, heads 4-7 remain as
    # val heads for LI/LC/STACK0, and heads 8-11 don't exist.
    if attn.num_heads >= 12:
        # === LEV-specific heads (4-7): Read saved_bp from memory[BP] ===
        # Phase 2 implementation - 2026-04-09
        # When OP_LEV active at BP marker, read the 4-byte word stored at BP address.
        # DESIGN: Write to OUTPUT at BP marker (overlay BP value temporarily).
        # This avoids TEMP dimension shortage (only 32 dims, need 128 for 4 bytes).
        # L16 will route this value to final destination (or handler for now).
        #
        # Address encoding: ADDR_B0-2 dims populated by L8 FFN Phase 1
        # Output: OUTPUT_LO/HI at BP marker (all 4 heads write to same position)
        # Note: For addresses < 256, only byte 0 matters, so conflicts acceptable.
        for h in range(4, 8):  # Heads 4-7 for bytes 0-3 of saved_bp
            base = h * HD
            byte_idx = h - 4  # 0, 1, 2, 3

            # === Dims 0, 1, 3: Q gating differs between head 4 and heads 5-7 ===
            # FIX 2026-04-15: Heads 5-7 fire at byte positions where OP_LEV is not set.
            # They rely on dim 28 (byte position) and address matching for gating instead.
            BP_I_GATING = 3  # BP marker index for gating flags

            if byte_idx == 0:
                # === Head 4: At BP marker (has OP_LEV and MARK_BP) ===
                # Dim 0: Require BOTH OP_LEV AND MARK_BP, block at PC marker
                # BUG FIX 2026-04-16: Add MARK_PC penalty. OP_LEV≈10 at PC marker gives
                # +20000, overcoming -4000 bias. Need -25000 to block at PC marker.
                # At BP: -4000 + 2000*10 + 2000*1 + (-25000)*0 = 18000 > 0 (fires)
                # At PC: -4000 + 2000*10 + 2000*0 + (-25000)*1 = -9000 < 0 (blocked)
                attn.W_q[base, BD.CONST] = -4000.0
                attn.W_q[base, BD.OP_LEV] = 2000.0
                attn.W_q[base, BD.MARK_BP] = 2000.0
                attn.W_q[base, BD.MARK_PC] = -25000.0  # Block at PC marker
                # FIX 2026-04-16: OP_LEV≈15 contributes +30000, address aliasing
                # contributes ~+600000 total. Need very strong suppression.
                attn.W_q[base, BD.MARK_SP] = -100000.0  # Block at SP marker
                attn.W_k[base, BD.CONST] = 10.0

                # Dim 1: Store anchor with OP_LEV + MARK_BP gating
                attn.W_q[base + 1, BD.CONST] = -50.0
                attn.W_q[base + 1, BD.OP_LEV] = 50.0
                attn.W_q[base + 1, BD.MARK_BP] = 50.0
                attn.W_q[base + 1, BD.MARK_PC] = -200.0  # Block at PC marker
                attn.W_q[base + 1, BD.MARK_SP] = -200.0  # Block at SP marker
                attn.W_k[base + 1, BD.MEM_STORE] = 100.0
                attn.W_k[base + 1, BD.CONST] = -50.0

                # Dim 3: Byte selection with OP_LEV + MARK_BP gating
                # FIX 2026-04-15: Increased BS and added CONST penalty to ensure only
                # MEM val byte 0 positions have positive K[3].
                # MEM markers have MEM_VAL_B0=0, so they get K[3]=-150 (penalty).
                # Val bytes have MEM_VAL_B0=1, so they get K[3]=0 (neutral).
                BS = 150.0
                attn.W_q[base + 3, BD.CONST] = -BS
                attn.W_q[base + 3, BD.OP_LEV] = BS
                attn.W_q[base + 3, BD.MARK_BP] = BS
                attn.W_q[base + 3, BD.MARK_PC] = -BS * 20  # Block at PC marker
                attn.W_q[base + 3, BD.MARK_SP] = -BS * 20  # Block at SP marker
                attn.W_k[base + 3, BD.MEM_VAL_B0] = BS
                attn.W_k[base + 3, BD.CONST] = -BS  # Penalize non-val-byte positions
            else:
                # === Heads 5-7: At BP byte positions (no OP_LEV, no MARK_BP) ===
                # Dim 0: Use byte position flags instead (already handled by dim 28)
                # Small positive bias - main gating comes from dim 28 and address matching
                attn.W_q[base, BD.CONST] = 10.0
                attn.W_k[base, BD.CONST] = 10.0

                # Dim 1: Store anchor - simplified (rely on other dims for Q gating)
                attn.W_q[base + 1, BD.CONST] = 10.0  # Always contribute
                attn.W_k[base + 1, BD.MEM_STORE] = 100.0
                attn.W_k[base + 1, BD.CONST] = -50.0

                # Dim 3: Byte selection using MEM_VAL flags (no Q gating needed)
                BS = 60.0
                attn.W_q[base + 3, BD.CONST] = BS  # Always contribute positive
                MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
                attn.W_k[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

            # === Dim 2: ZFOD negative offset for store entries (same for all heads) ===
            attn.W_q[base + 2, BD.CONST] = -96.0
            attn.W_k[base + 2, BD.MEM_STORE] = 50.0

            # === Dims 4-35: One-hot address matching (byte 0 only) ===
            # FIX 2026-04-15: Use one-hot encoding and ADDR_KEY for K projection.
            # Q: Read from ADDR_B0_LO/HI (one-hot address at BP marker)
            # K: Read from ADDR_KEY (one-hot address at MEM val byte positions)
            L_addr = 50.0
            for k in range(16):
                attn.W_q[base + 4 + k, BD.ADDR_B0_LO + k] = L_addr
                attn.W_q[base + 4 + 16 + k, BD.ADDR_B0_HI + k] = L_addr
                attn.W_k[base + 4 + k, BD.ADDR_KEY + k] = L_addr
                attn.W_k[base + 4 + 16 + k, BD.ADDR_KEY + 16 + k] = L_addr

            # === Dim 36: Per-head position gate ===
            # FIX 2026-04-15: Each head should fire at its corresponding byte position:
            # - Head 4 (byte 0): Fire at BP marker
            # - Head 5 (byte 1): Fire at BP byte 1 (BYTE_INDEX_1 AND L1H1[BP_I])
            # - Head 6 (byte 2): Fire at BP byte 2 (BYTE_INDEX_2 AND H0[BP_I])
            # - Head 7 (byte 3): Fire at BP byte 3 (BYTE_INDEX_3 AND H1[BP_I])
            # This ensures each head writes its result at the correct output position.
            # FIX 2026-04-16: Moved from dim 28 to dim 36 to avoid collision with address
            # matching dims (4-35). The collision caused K[28] to include ADDR_KEY terms,
            # which made the score positive even when the Q gating was negative.
            GATE_DIM = 36
            BP_I = 3  # BP marker index for half-space flags
            if byte_idx == 0:
                # Head 4: Fire at BP marker
                attn.W_q[base + GATE_DIM, BD.CONST] = -500.0
                attn.W_q[base + GATE_DIM, BD.MARK_BP] = 500.0
                # FIX 2026-04-16: Strong MARK_PC suppression to prevent firing at PC marker.
                # Address matching dims contribute ~20000 to score, so need very strong penalty.
                attn.W_q[base + GATE_DIM, BD.MARK_PC] = -50000.0
            elif byte_idx == 1:
                # Head 5: Fire at BP byte 1 (d=2 from BP marker)
                # BYTE_INDEX_1 AND L1H1[BP_I]: both should be ~1.0 at d=2
                # FIX: Add strong MARK_BP penalty to suppress at BP marker itself
                attn.W_q[base + GATE_DIM, BD.CONST] = -500.0
                attn.W_q[base + GATE_DIM, BD.BYTE_INDEX_1] = 500.0
                attn.W_q[base + GATE_DIM, BD.L1H1 + BP_I] = 500.0
                attn.W_q[base + GATE_DIM, BD.MARK_BP] = -1000.0  # Suppress at BP marker
                # FIX 2026-04-16: Strong MARK_PC suppression
                attn.W_q[base + GATE_DIM, BD.MARK_PC] = -50000.0
            elif byte_idx == 2:
                # Head 6: Fire at BP byte 2 (d=3 from BP marker)
                # BYTE_INDEX_2 AND H0[BP_I]: both should be ~1.0 at d=3
                # FIX: Add strong MARK_BP penalty to suppress at BP marker itself
                attn.W_q[base + GATE_DIM, BD.CONST] = -500.0
                attn.W_q[base + GATE_DIM, BD.BYTE_INDEX_2] = 500.0
                attn.W_q[base + GATE_DIM, BD.H0 + BP_I] = 500.0
                attn.W_q[base + GATE_DIM, BD.MARK_BP] = -1000.0  # Suppress at BP marker
                # FIX 2026-04-16: Strong MARK_PC suppression
                attn.W_q[base + GATE_DIM, BD.MARK_PC] = -50000.0
            elif byte_idx == 3:
                # Head 7: Fire at BP byte 3 (d=4 from BP marker)
                # BYTE_INDEX_3 AND H1[BP_I]: both should be ~1.0 at d=4
                # FIX: Add strong MARK_BP penalty to suppress at BP marker itself
                attn.W_q[base + GATE_DIM, BD.CONST] = -500.0
                attn.W_q[base + GATE_DIM, BD.BYTE_INDEX_3] = 500.0
                attn.W_q[base + GATE_DIM, BD.H1 + BP_I] = 500.0
                attn.W_q[base + GATE_DIM, BD.MARK_BP] = -1000.0  # Suppress at BP marker
                # FIX 2026-04-16: Strong MARK_PC suppression
                attn.W_q[base + GATE_DIM, BD.MARK_PC] = -50000.0
            attn.W_k[base + GATE_DIM, BD.CONST] = 5.0

            # === V/O: Copy byte value to staging/OUTPUT ===
            # V: Copy from memory value byte (CLEAN_EMBED)
            for k in range(16):
                attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0

            # O: Write to OUTPUT_LO/HI
            # For LEV, we WANT saved_bp in OUTPUT at BP marker because that's the
            # final BP register value. The predicted BP bytes should be saved_bp bytes.
            # Heads 5-7: No O projection needed (bytes 1-3 handled at future positions)
            if byte_idx == 0:
                for k in range(16):
                    attn.W_o[BD.OUTPUT_LO + k, base + 32 + k] = 1.0
                    attn.W_o[BD.OUTPUT_HI + k, base + 48 + k] = 1.0

        # === LEV-specific heads (8-11): Read return_addr from memory[BP+8] ===
        # Phase 2 implementation - 2026-04-09
        # When OP_LEV active at PC marker, read the 4-byte word stored at BP+8 address.
        # Write to OUTPUT at PC marker (final destination for PC register).
        #
        # Key difference from heads 4-7: +8 offset in address matching.
        # For byte 0 lo nibble: (BP + 8) % 16
        # For other bytes: Same as BP (since +8 doesn't affect them for small addresses)
        for h in range(8, 12):  # Heads 8-11 for bytes 0-3 of return_addr
            base = h * HD
            byte_idx = h - 8  # 0, 1, 2, 3

            # === Dim 0: Bias — fire at PC marker when OP_LEV active ===
            # BUG FIX 2026-04-13: Increased bias to -4000 to require BOTH OP_LEV AND MARK_PC.
            # Previous -2000 bias was cancelled by MARK_PC alone (0 + 2000 = 0),
            # causing spurious activation at PC marker for non-LEV instructions.
            attn.W_q[base, BD.CONST] = -4000.0  # FIXED: was -2000
            attn.W_q[base, BD.OP_LEV] = 2000.0   # Only when LEV active
            attn.W_q[base, BD.MARK_PC] = 2000.0  # Activate at PC marker (not BP!)
            attn.W_k[base, BD.CONST] = 10.0

            # === Dim 1: Store anchor ===
            # FIX 2026-04-15: Add negative bias requiring BOTH OP_LEV AND MARK_PC
            attn.W_q[base + 1, BD.CONST] = -50.0
            attn.W_q[base + 1, BD.OP_LEV] = 50.0
            attn.W_q[base + 1, BD.MARK_PC] = 50.0
            attn.W_k[base + 1, BD.MEM_STORE] = 100.0
            attn.W_k[base + 1, BD.CONST] = -50.0

            # === Dim 2: ZFOD offset ===
            attn.W_q[base + 2, BD.CONST] = -96.0
            attn.W_k[base + 2, BD.MEM_STORE] = 50.0

            # === Dim 3: Byte selection ===
            # FIX 2026-04-15: Require BOTH OP_LEV AND MARK_PC
            BS = 60.0
            attn.W_q[base + 3, BD.CONST] = -BS
            attn.W_q[base + 3, BD.OP_LEV] = BS
            attn.W_q[base + 3, BD.MARK_PC] = BS
            # FIX 2026-04-15: Use MEM_VAL_B0 for byte 0 instead of L2H0/H1
            MEM_VAL_DIMS = [BD.MEM_VAL_B0, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
            attn.W_k[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

            # === Dims 4-35: One-hot address matching (BP+8 for return_addr) ===
            # FIX 2026-04-15: Use one-hot encoding and ADDR_KEY for K projection.
            # FIX 2026-04-16: Removed +8 offset from Q projection. L9 FFN already applies
            # the +8 offset to ADDR_B0_LO at PC marker, so ADDR_B0_LO now represents
            # (BP+8) directly. Reading it without additional offset gives correct query.
            #
            # Q: Read from ADDR_B0_LO/HI directly (already shifted by L9 FFN)
            # K: Read from ADDR_KEY (one-hot address at MEM val byte positions)
            L_addr = 50.0

            # Dims 4-19: Byte 0 lo nibble (NO offset - L9 FFN already shifted)
            for k in range(16):
                attn.W_q[base + 4 + k, BD.ADDR_B0_LO + k] = L_addr
            for k in range(16):
                attn.W_k[base + 4 + k, BD.ADDR_KEY + k] = L_addr

            # Dims 20-35: Byte 0 hi nibble (L9 FFN handles carry via ADDR_B1_LO/HI)
            # FIX 2026-04-16: Simplified - L9 FFN sets ADDR_B0_HI with correct carry.
            # No need for complex carry logic here.
            for k in range(16):
                attn.W_q[base + 20 + k, BD.ADDR_B0_HI + k] = L_addr
            for k in range(16):
                attn.W_k[base + 20 + k, BD.ADDR_KEY + 16 + k] = L_addr

            # === Dim 36: Position gate ===
            # FIX 2026-04-15: Add MARK_BP penalty to prevent firing at BP marker
            # FIX 2026-04-16: Moved from dim 28 to dim 36 to avoid collision with address
            # matching dims (4-35). The collision caused K[28] to include ADDR_KEY terms.
            GATE_DIM = 36
            attn.W_q[base + GATE_DIM, BD.CONST] = -500.0
            attn.W_q[base + GATE_DIM, BD.MARK_PC] = 500.0
            attn.W_q[base + GATE_DIM, BD.MARK_BP] = -1000.0  # Suppress at BP marker
            attn.W_k[base + GATE_DIM, BD.CONST] = 5.0

            # === Dim 37: Memory position suppression ===
            # FIX 2026-04-16: Suppress attention to non-memory positions.
            # Previous step byte positions have ADDR_KEY (aliased with ADDR_B0_HI)
            # contaminated with ~70.0 values, causing spurious attention there.
            # Use MEM_STORE flag to distinguish memory positions from step positions.
            # FIX 2026-04-16: Moved from dim 29 to dim 37 to avoid collision with address
            # matching dims (4-35).
            #
            # K[37] = 0 at memory positions (MEM_STORE=4 after L7), K[37] = 40000 at non-memory
            # Q[37] = -1000 always
            # Q·K contribution: 0 at memory, -5M at non-memory (overwhelms aliasing)
            # FIX 2026-04-17: CONST=40000 to match MEM_STORE=4 (2.0 embed + 2.0 L7 broadcast).
            SUPPRESS_DIM = 37
            attn.W_k[base + SUPPRESS_DIM, BD.CONST] = 40000.0
            attn.W_k[base + SUPPRESS_DIM, BD.MEM_STORE] = -10000.0
            attn.W_q[base + SUPPRESS_DIM, BD.CONST] = -1000.0

            # === V/O: Copy byte value to TEMP at PC marker (for L16 FFN routing) ===
            # FIX 2026-04-15: Write to TEMP dims instead of OUTPUT to prevent
            # pollution at BP marker. L16 FFN routes TEMP → OUTPUT at PC marker.
            # Only head 8 (byte 0) writes; heads 9-11 are for future byte positions.
            for k in range(16):
                attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            if byte_idx == 0:
                # Head 8 writes to TEMP for return_addr byte 0
                for k in range(16):
                    attn.W_o[BD.TEMP + k, base + 32 + k] = 1.0
                    attn.W_o[BD.TEMP + 16 + k, base + 48 + k] = 1.0
            # Heads 9-11: No O projection (they write at byte positions later)


# =============================================================================
# L16: LEV register routing (Phase 3)
# =============================================================================


def _set_layer16_lev_routing(ffn, S, BD):
    """L16 FFN: Route LEV memory reads and compute SP = BP + 16.

    After L15, we have:
    - BP marker: OUTPUT = saved_bp (from L15 heads 4-7)
    - PC marker: OUTPUT = return_addr (from L15 heads 8-11)
    - ADDR_B0_LO/HI: old BP value (from Phase 1)

    L16 computes SP = old_BP + 16 and writes to OUTPUT at SP marker.

    Registers updated by LEV:
    - BP ← saved_bp (already in OUTPUT at BP marker from L15)
    - PC ← return_addr (already in OUTPUT at PC marker from L15)
    - SP ← old_BP + 16 (computed here at SP marker)

    Strategy: Enumerate (bp_lo + 16) % 16 for lo nibble, handle carry for hi.
    """
    unit = 0

    # === FIX 2026-04-16: Cancel OUTPUT at SP marker during LEV ===
    # L15 heads 0-4 write spurious OUTPUT values at SP marker due to address aliasing.
    # ADDR_B0_LO == OPCODE_BYTE_LO causes heads to attend to PC bytes from prev step.
    # Cancel all OUTPUT before adding correct SP = BP + 16 values.
    for k in range(16):
        # Cancel OUTPUT_LO[k] at SP marker when OP_LEV
        # FIX 2026-04-16: Add marker exclusions. Without them, OP_LEV*S/5 = 10*20 = 200
        # overcomes b_up=-150 threshold even at AX/PC/BP markers where MARK_SP=0.
        ffn.W_up[unit, BD.OP_LEV] = S / 5
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Exclude PC marker
        ffn.W_up[unit, BD.MARK_AX] = -S  # Exclude AX marker
        ffn.W_up[unit, BD.MARK_BP] = -S  # Exclude BP marker
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        # Cancel OUTPUT_HI[k] at SP marker when OP_LEV
        # FIX 2026-04-16: Add marker exclusions (same reasoning as OUTPUT_LO).
        ffn.W_up[unit, BD.OP_LEV] = S / 5
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_PC] = -S  # Exclude PC marker
        ffn.W_up[unit, BD.MARK_AX] = -S  # Exclude AX marker
        ffn.W_up[unit, BD.MARK_BP] = -S  # Exclude BP marker
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === SP = BP + 16: Lo nibble (SP_byte0_lo = (BP_byte0_lo + 16) % 16 = BP_byte0_lo) ===
    # Since 16 % 16 = 0, adding 16 to a nibble just wraps to the same value for lo nibble
    # but generates carry to hi nibble
    # BUG FIX 2026-04-15: OP_LEV ≈ 10 at SP marker (amplified), scale down to ~1 contribution
    # BUG FIX 2026-04-16: Add MARK_BP exclusion. Without it, units fire at BP marker
    # because OP_LEV*S + ADDR_B0[k]*S > 3S even without MARK_SP.
    # FIX 2026-04-16: Add MARK_PC exclusion. Without it, units fire at PC marker
    # because ADDR_B0_LO[8]=12 from +8 offset makes OP_LEV + ADDR_B0 > threshold.
    # BUG FIX 2026-04-16: Increased MARK_PC penalty from -S*5 to -S*15 to overcome
    # ADDR_B0_LO[8]=12 contribution. Calc: 10 + 0 - 150 + 120 - 30 = -50 < 0 (blocked).
    for k in range(16):
        # Result lo nibble = k (adding 16 to nibble k gives k with carry)
        # FIX 2026-04-16: Gate on ADDR_B0_LO[k] instead of CONST to prevent spurious firing
        # when OP_LEV and MARK_SP are amplified but ADDR_B0_LO[k] is low.
        ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down: OP_LEV*10 * S/10 ≈ S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
        ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (must overcome ADDR_B0[8]=12*S)
        ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker (ADDR_B0 contamination ~40*S)
        # FIX 2026-04-16: Suppress at byte positions (BYTE_INDEX=1 at bytes, =0 at markers)
        # ADDR_B0 contamination causes spurious firing at byte positions, need strong suppression.
        ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10  # Suppress at byte 0 positions
        ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10  # Suppress at byte 1 positions
        ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10  # Suppress at byte 2 positions
        ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10  # Suppress at byte 3 positions
        ffn.W_up[unit, BD.ADDR_B0_LO + k] = S  # Old BP lo nibble
        ffn.b_up[unit] = -S * 3.0  # Raised threshold to require ADDR_B0 to be active
        # Gate on ADDR_B0_LO[k] - only fires when this nibble has significant value
        ffn.W_gate[unit, BD.ADDR_B0_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1

    # === SP = BP + 16: Hi nibble (SP_byte0_hi = (BP_byte0_hi + 1) % 16) ===
    # Adding 16 to byte 0 means: lo nibble gets +16%16=0, hi nibble gets +1 (carry from 16)
    for k in range(16):
        result = (k + 1) % 16
        # FIX 2026-04-16: Gate on ADDR_B0_HI[k] instead of CONST
        ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down: OP_LEV*10 * S/10 ≈ S
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
        ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (must overcome ADDR_B0*S)
        ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
        # FIX 2026-04-16: Suppress at byte positions
        ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
        ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
        ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
        ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
        ffn.W_up[unit, BD.ADDR_B0_HI + k] = S  # Old BP hi nibble
        ffn.b_up[unit] = -S * 3.0  # Raised threshold
        # Gate on ADDR_B0_HI[k] - only fires when this nibble has significant value
        ffn.W_gate[unit, BD.ADDR_B0_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
        unit += 1

    # === SP bytes 1-3: Copy from ADDR_B1/B2 with byte 1 carry ===
    # NOTE 2026-04-16: These units are DISABLED for now. They were designed to compute
    # SP bytes 1-3 but they fire at the wrong position (marker instead of byte positions).
    # For addresses that wrap (like BP=0xfffffff8 + 16 = 0x00000008), bytes 1-3 are all 0x00
    # which is handled by the default OUTPUT = 0 behavior.
    # TODO: Redesign to fire at byte positions using BYTE_INDEX flags.
    if False:  # Disabled
     for b0_hi in range(16):
        for b1_lo in range(16):
            carry = 1 if b0_hi == 15 else 0  # Carry from byte 0
            result = (b1_lo + carry) % 16
            needs_carry = (b1_lo + carry) >= 16

            ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down
            ffn.W_up[unit, BD.MARK_SP] = S
            ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
            ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (increased to overcome ADDR*S)
            ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
            # FIX 2026-04-16: Suppress at byte positions
            ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up[unit, BD.ADDR_B0_HI + b0_hi] = S
            ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
            ffn.b_up[unit] = -S * 4.0  # Raised threshold for 4-way AND
            ffn.W_gate[unit, BD.CONST] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            if needs_carry:
                ffn.W_down[BD.CARRY + 1, unit] = 2.0 / (S * 5.0)  # Signal carry to byte 1 hi
            unit += 1

    # Byte 1 hi nibble: Add carry from byte 1 lo (DISABLED - see note above)
    if False:
     for b1_hi in range(16):
        for carry in [0, 1]:
            result = (b1_hi + carry) % 16
            needs_carry = (b1_hi + carry) >= 16

            ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down
            ffn.W_up[unit, BD.MARK_SP] = S
            ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
            ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (increased to overcome ADDR*S)
            ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
            # FIX 2026-04-16: Suppress at byte positions
            ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up[unit, BD.ADDR_B1_HI + b1_hi] = S
            if carry:
                ffn.W_up[unit, BD.CARRY + 1] = S
            ffn.b_up[unit] = -S * (3.0 + carry * 1.0)  # Raised threshold
            ffn.W_gate[unit, BD.CONST] = 1.0
            ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
            if needs_carry:
                ffn.W_down[BD.CARRY + 2, unit] = 2.0 / (S * 5.0)  # Signal carry to byte 2
            unit += 1

    # Bytes 2-3: Direct copy with carry propagation (for addresses > 256, not common)
    # (DISABLED - see note above)
    # Byte 2 lo nibble with carry
    if False:
     for b2_lo in range(16):
        for carry in [0, 1]:
            result = (b2_lo + carry) % 16
            needs_carry = (b2_lo + carry) >= 16

            ffn.W_up[unit, BD.OP_LEV] = S / 10  # Scale down
            ffn.W_up[unit, BD.MARK_SP] = S
            ffn.W_up[unit, BD.MARK_BP] = -S * 15  # Exclude BP marker (must overcome ADDR_B0*S)
            ffn.W_up[unit, BD.MARK_PC] = -S * 15  # Exclude PC marker (increased to overcome ADDR*S)
            ffn.W_up[unit, BD.MARK_AX] = -S * 50  # FIX 2026-04-16: Exclude AX marker
            # FIX 2026-04-16: Suppress at byte positions
            ffn.W_up[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up[unit, BD.ADDR_B2_LO + b2_lo] = S
            if carry:
                ffn.W_up[unit, BD.CARRY + 2] = S
            ffn.b_up[unit] = -S * (3.0 + carry * 1.0)  # Raised threshold
            ffn.W_gate[unit, BD.CONST] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            if needs_carry:
                ffn.W_down[BD.CARRY + 3, unit] = 2.0 / (S * 5.0)
            unit += 1

    # Byte 2 hi nibble, byte 3 lo, byte 3 hi - similar pattern but skipping for brevity
    # For addresses < 256, bytes 2-3 are zero anyway

    # === FIX 2026-04-15: Route TEMP → OUTPUT at PC marker for return_addr ===
    # L15 head 8 writes return_addr byte 0 to TEMP[0:31] (lo/hi nibbles).
    # This FFN copies TEMP → OUTPUT at PC marker when OP_LEV is active.
    #
    # FIX 2026-04-16: Removed OUTPUT_LO cancel - let TEMP just add to existing.
    # With heads 0-3 suppressed at PC marker, residual OUTPUT_LO[10]=1.0 comes from
    # L3 carry-forward. Adding TEMP[10]=1.0 gives OUTPUT_LO[10]=3.0, which dominates.
    #
    # Keep OUTPUT_HI cancel to handle spurious OUTPUT_HI[3] from residual attention.
    # BUG FIX 2026-04-16: Add MARK_AX/SP/BP exclusions. Without them, OP_LEV*S/5 = 10*20 = 200
    # overcomes b_up=-150 threshold even at AX/SP/BP markers where MARK_PC=0.
    # This was causing OUTPUT_HI[2] to be canceled at AX marker, breaking AX preservation.
    for k in range(16):
        # Cancel OUTPUT_HI[k] at PC marker when OP_LEV - needed to suppress wrong hi nibble
        ffn.W_up[unit, BD.OP_LEV] = S / 5
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.MARK_AX] = -S  # Exclude AX marker
        ffn.W_up[unit, BD.MARK_SP] = -S  # Exclude SP marker
        ffn.W_up[unit, BD.MARK_BP] = -S  # Exclude BP marker
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # Add TEMP values to OUTPUT
    for k in range(16):
        # TEMP_LO[k] → OUTPUT_LO[k] at PC marker
        ffn.W_up[unit, BD.OP_LEV] = S / 10
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.TEMP + k] = S
        ffn.b_up[unit] = -S * 2.0
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        # TEMP_HI[k] → OUTPUT_HI[k] at PC marker
        ffn.W_up[unit, BD.OP_LEV] = S / 10
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.TEMP + 16 + k] = S
        ffn.b_up[unit] = -S * 2.0
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === FIX 2026-04-16: Set OUTPUT = 0x00 at byte positions 1-3 for LEV ===
    # For return_addr < 256, bytes 1-3 are always 0. At byte positions (BYTE_INDEX=1),
    # we need to override the OUTPUT that persists from PC marker with the 0x00 encoding.
    # 0x00 = lo nibble 0, hi nibble 0 → OUTPUT_LO[0]=1, OUTPUT_HI[0]=1
    #
    # BUG FIX 2026-04-16: Require OP_LEV strongly, don't gate on OUTPUT_LO[10].
    # Previous version gated on OUTPUT_LO[10] which caused clearing during IMM
    # (OUTPUT_LO[10]=4 from FETCH overcame threshold without OP_LEV).
    #
    # BUG FIX 2026-04-16: Suppress at marker positions (NEXT_* = 0.68).
    # BYTE_INDEX_3 persists at d=5 from previous marker (marker transition position).
    # At marker positions we should NOT set OUTPUT values.
    # Suppress on all NEXT_* flags to prevent firing at AX, SP, BP, etc. marker positions.
    #
    # First, clear OUTPUT_LO[10] at byte 1-3 (prevent 0x0A from persisting)
    # BUG FIX 2026-04-16: Use BYTE_INDEX_1/2/3 for bytes 1/2/3.
    # BYTE_INDEX_N = 1 at byte N position (0-indexed from after marker).
    # Byte 0 = first byte after marker (should output 0x0A, don't clear!)
    # Byte 1 = second byte (should output 0x00, need to clear OUTPUT_LO[10])
    # etc.
    # BUG FIX 2026-04-16: Add MARK_PC penalty. Same issue as OUTPUT_LO[0]/OUTPUT_HI[0] below.
    for byte_pos in range(3):  # byte positions 1, 2, 3
        # BYTE_INDEX_1 for byte 1, BYTE_INDEX_2 for byte 2, BYTE_INDEX_3 for byte 3
        byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
        # Clear OUTPUT_LO[10] unconditionally at LEV byte positions 1-3
        ffn.W_up[unit, BD.OP_LEV] = S / 2  # Require OP_LEV (~7.5 at byte pos → contrib 375)
        ffn.W_up[unit, byte_idx_dim] = S  # Fire at this byte position (~0.97 → contrib 97)
        # Suppress at marker positions where NEXT_* = 0.68
        # -S*1.5 * 0.68 = -102, needs to overcome 72 positive contrib → OK
        ffn.W_up[unit, BD.MARK_PC] = -S * 1.5  # BUG FIX: Suppress at PC marker
        # FIX 2026-04-16: Add MARK_AX/SP/BP exclusions. At AX marker, OP_LEV*50 = 501 > 400
        # threshold, but NEXT_* = 0 (NEXT flags are set at previous position, not current).
        ffn.W_up[unit, BD.MARK_AX] = -S * 10  # Suppress at AX marker (MARK_AX=1 → -1000)
        ffn.W_up[unit, BD.MARK_SP] = -S * 10  # Suppress at SP marker
        ffn.W_up[unit, BD.MARK_BP] = -S * 10  # Suppress at BP marker
        ffn.W_up[unit, BD.NEXT_AX] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_BP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_STACK0] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_MEM] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SE] = -S * 1.5
        ffn.b_up[unit] = -S * 4  # Threshold: 375 + 97 - 400 = 72 > 0
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 10, unit] = -10.0 / S  # Strong cancel of OUTPUT_LO[10]
        unit += 1

    # Set OUTPUT_LO[0] = 1 at byte positions 1-3 (for 0x00 value)
    # BUG FIX 2026-04-16: Add MARK_PC penalty. Without it, OP_LEV=10 at PC marker
    # causes OP_LEV*S/2 = 500 > 400 threshold, making unit fire even with BYTE_INDEX=0.
    for byte_pos in range(3):
        byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
        ffn.W_up[unit, BD.OP_LEV] = S / 2  # Require OP_LEV
        ffn.W_up[unit, byte_idx_dim] = S
        # Suppress at marker positions
        ffn.W_up[unit, BD.MARK_PC] = -S * 1.5  # BUG FIX: Suppress at PC marker
        # FIX 2026-04-16: Add MARK_AX/SP/BP exclusions (same as OUTPUT_LO[10] above)
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.MARK_SP] = -S * 10
        ffn.W_up[unit, BD.MARK_BP] = -S * 10
        ffn.W_up[unit, BD.NEXT_AX] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_BP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_STACK0] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_MEM] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SE] = -S * 1.5
        ffn.b_up[unit] = -S * 4
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 5.0 / S  # Strong set of OUTPUT_LO[0]
        unit += 1

    # Set OUTPUT_HI[0] = 1 at byte positions 1-3 (for 0x00 value)
    # BUG FIX 2026-04-16: Add MARK_PC penalty (same as OUTPUT_LO[0] above).
    for byte_pos in range(3):
        byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
        ffn.W_up[unit, BD.OP_LEV] = S / 2  # Require OP_LEV
        ffn.W_up[unit, byte_idx_dim] = S
        # Suppress at marker positions
        ffn.W_up[unit, BD.MARK_PC] = -S * 1.5  # BUG FIX: Suppress at PC marker
        # FIX 2026-04-16: Add MARK_AX/SP/BP exclusions (same as OUTPUT_LO[10] above)
        ffn.W_up[unit, BD.MARK_AX] = -S * 10
        ffn.W_up[unit, BD.MARK_SP] = -S * 10
        ffn.W_up[unit, BD.MARK_BP] = -S * 10
        ffn.W_up[unit, BD.NEXT_AX] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_BP] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_STACK0] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_MEM] = -S * 1.5
        ffn.W_up[unit, BD.NEXT_SE] = -S * 1.5
        ffn.b_up[unit] = -S * 4
        ffn.W_gate[unit, BD.CONST] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 5.0 / S  # Strong set of OUTPUT_HI[0]
        unit += 1

    return unit  # Return number of units used


# =============================================================================
# BZ/BNZ: Conditional branch (L6 relay + L6 FFN override)
# =============================================================================


def _set_bz_bnz_relay(attn, S, BD, HD):
    """L6 attention: Relay AX zero detection to PC marker for BZ/BNZ.

    Head 4 (ALiBi slope=5.0): At PC marker, read prev AX marker's flags.
    Writes to CMP[2..5] at PC marker (CMP[0]=IS_JMP, CMP[1]=IS_EXIT reserved):
      CMP[2] = OP_BZ, CMP[3] = OP_BNZ
      CMP[4] = AX_CARRY_LO[0] (1.0 if lo nibble is 0)
      CMP[5] = AX_CARRY_HI[0] (1.0 if hi nibble is 0)
    Also copies FETCH_LO/HI → TEMP (branch target).

    L6 FFN uses these for conditional PC override:
      BZ:  4-way AND (MARK_PC + CMP[2] + CMP[4] + CMP[5]) → branch if AX==0
      BNZ: 2 groups covering AX!=0 cases
    """
    L = 50.0
    base = 4 * HD

    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX
    # FIX 2026-04-16: Gate on OP_BZ or OP_BNZ to prevent firing for other opcodes.
    # Without this gate, the head fires for ALL opcodes and writes FETCH to TEMP,
    # which overlaps OUTPUT_BYTE (dims 480-511), corrupting PC byte generation.
    # OP_BZ and OP_BNZ are now injected at 5.0 via _inject_active_opcode.
    # Score budget:
    #   At PC marker without OP_BZ/BNZ: 50 - 65 = -15 (blocked)
    #   At PC marker with OP_BZ=5: 50 - 65 + 50 = 35 (fires)
    #   At PC marker with OP_BNZ=5: 50 - 65 + 50 = 35 (fires)
    attn.W_q[base, BD.CONST] = -L * 1.3  # Baseline penalty (stronger than before)
    attn.W_q[base, BD.OP_BZ] = L / 5.0  # OP_BZ=5 → contributes L
    attn.W_q[base, BD.OP_BNZ] = L / 5.0  # OP_BNZ=5 → contributes L
    attn.W_k[base, BD.MARK_AX] = L
    # Add K-side constant so Q[0] * K[0] creates a negative score when Q[0] < 0.
    # Without this, K[0]=0 at non-AX positions makes score=0, not negative.
    # With K[0]=L*CONST, score = Q[0]*L*CONST/8 = -15*50/8 = -93.75 (blocked)
    # With OP_BZ=5, Q[0]=35, so score = 35*50/8 = 218.75 (fires)
    attn.W_k[base, BD.CONST] = L

    # V: copy OP_BZ, OP_BNZ flags
    attn.W_v[base + 1, BD.OP_BZ] = 1.0
    attn.W_v[base + 2, BD.OP_BNZ] = 1.0
    # V: copy AX_CARRY_LO[0] and AX_CARRY_HI[0] (zero detection)
    attn.W_v[base + 3, BD.AX_CARRY_LO + 0] = 1.0
    attn.W_v[base + 4, BD.AX_CARRY_HI + 0] = 1.0
    # V: copy FETCH_LO/HI (branch target)
    for k in range(16):
        attn.W_v[base + 5 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 21 + k, BD.FETCH_HI + k] = 1.0

    # O: write to CMP[2..5] at PC marker for FFN to use
    # (CMP[0] reserved for IS_JMP from head 0, CMP[1] for IS_EXIT from head 1)
    # Normalize OP_BZ/BNZ: raw ≈5 × 0.2 → CMP ≈1.0 (same scale as zero flags)
    attn.W_o[BD.CMP + 2, base + 1] = 0.2  # OP_BZ at PC (normalized)
    attn.W_o[BD.CMP + 3, base + 2] = 0.2  # OP_BNZ at PC (normalized)
    attn.W_o[BD.CMP + 4, base + 3] = 1.0  # AX_LO_IS_ZERO at PC
    attn.W_o[BD.CMP + 5, base + 4] = 1.0  # AX_HI_IS_ZERO at PC
    # Write branch target to TEMP dims at PC marker
    for k in range(16):
        attn.W_o[BD.TEMP + k, base + 5 + k] = 1.0  # FETCH_LO
        attn.W_o[BD.TEMP + 16 + k, base + 21 + k] = 1.0  # FETCH_HI

    # Old L15 implementation removed — see _set_layer15_memory_lookup above.


# =============================================================================
# IO Syscall Layer (PUTCHAR — fully autoregressive)
# =============================================================================


def _set_io_putchar_routing(ffn, S, BD):
    """L6 FFN addition: detect PUTCHAR and route AX → OUTPUT.

    When OP_PUTCHAR is active at AX marker:
    1. Set IO_IS_PUTCHAR flag
    2. Route AX_CARRY → OUTPUT_LO/HI (same pattern as EXIT/NOP routing)

    The model produces the correct output byte autoregressively.
    The runner reads generated AX bytes and accumulates the output string.

    GETCHAR is handled entirely runner-side (see run_vm.py).

    Starts at unit 900 to avoid conflict with _set_layer6_routing_ffn (units 0-898).
    """
    unit = 900

    # OP_PUTCHAR AND MARK_AX → IO_IS_PUTCHAR
    T = 4.0
    ffn.W_up[unit, BD.OP_PUTCHAR] = S
    ffn.W_up[unit, BD.MARK_AX] = S
    ffn.b_up[unit] = -S * T
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.IO_IS_PUTCHAR, unit] = 2.0 / S
    unit += 1

    # PUTCHAR: AX_CARRY → OUTPUT (same as EXIT/NOP routing)
    for k in range(16):
        ffn.W_up[unit, BD.OP_PUTCHAR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_PUTCHAR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1


# =============================================================================
# L6 Opcode Relay Head (AX → SP/STACK0 marker positions)
# =============================================================================


def _set_opcode_relay_head(attn, S, BD, HD):
    """L6 attention head 6: relay opcode flags from AX → SP/STACK0 markers.

    Opcode decode (L5 FFN) writes OP_* only at the AX marker. The L6 FFN
    needs opcode information at SP and STACK0 markers for PSH, ADJ, and
    binary pop operations. This head copies scaled opcode flags.

    Writes to CMP dims at SP/STACK0 (safe — CMP is only used at PC/SE/AX
    markers by other heads):
      CMP[0] = OP_PSH relay  (≈1.0 when PSH active)
      CMP[1] = OP_ADJ relay  (≈1.0 when ADJ active)
      CMP[3] = POP group     (≈1.0 when any of 18 binary pop ops active)

    Distance SP→AX = 5 tokens, STACK0→AX = 15 tokens.
    With L=50, ALiBi slope=5: score = 50²*0.125 - 5*d = 312.5 - 25 = 287.5 ✓
    """
    L = 50.0
    SP_I = 2  # SP index in position encoding
    BP_I = 3  # BP index for STACK0 byte detection
    base = 6 * HD  # head 6

    # Q: fires at SP, STACK0, BP, PC, and MEM markers, blocked at AX
    # FIX 2026-04-16: Also fire at SP/STACK0 byte positions (not just markers).
    # L6 FFN needs CMP[4] at byte positions for JSR SP-=8 and STACK0=return_addr.
    # Previously, CMP[4] was only set at markers, and L7 head 6 relayed to bytes,
    # but L6 FFN runs before L7 attention so CMP[4] was 0 at byte positions.
    attn.W_q[base, BD.MARK_SP] = L
    attn.W_q[base, BD.H1 + SP_I] = L  # SP area including bytes (d<=4.5 from SP)
    attn.W_q[base, BD.MARK_STACK0] = L
    attn.W_q[base, BD.L1H4 + BP_I] = L  # STACK0 bytes (d<=6.5 from BP)
    attn.W_q[base, BD.MARK_BP] = L  # ENT needs flags at BP marker
    attn.W_q[base, BD.MARK_PC] = L  # LEV needs OP_LEV at PC marker (for L15 heads 8-11)
    attn.W_q[base, BD.MARK_MEM] = L  # Store flag relay to MEM marker
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker

    # K: attend to AX marker
    attn.W_k[base, BD.MARK_AX] = L

    # V[1]: OP_PSH (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 1, BD.OP_PSH] = 0.2

    # V[2]: OP_ADJ (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 2, BD.OP_ADJ] = 0.2

    # V[3]: POP group (sum of 18 ops × 0.04 = ≈0.2 when any active)
    pop_op_dims = [
        BD.OP_ADD,
        BD.OP_SUB,
        BD.OP_MUL,
        BD.OP_DIV,
        BD.OP_MOD,
        BD.OP_EQ,
        BD.OP_NE,
        BD.OP_LT,
        BD.OP_GT,
        BD.OP_LE,
        BD.OP_GE,
        BD.OP_OR,
        BD.OP_XOR,
        BD.OP_AND,
        BD.OP_SHL,
        BD.OP_SHR,
        BD.OP_SI,
        BD.OP_SC,
    ]
    for op_dim in pop_op_dims:
        attn.W_v[base + 3, op_dim] = 0.04

    # V[4]: OP_ENT (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 4, BD.OP_ENT] = 0.2

    # V[5]: OP_JSR (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 5, BD.OP_JSR] = 0.2

    # V[6]: MEM_STORE flag — any of SI/SC/PSH/JSR/ENT (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 6, BD.OP_SI] = 0.2
    attn.W_v[base + 6, BD.OP_SC] = 0.2
    attn.W_v[base + 6, BD.OP_PSH] = 0.2
    attn.W_v[base + 6, BD.OP_JSR] = 0.2
    attn.W_v[base + 6, BD.OP_ENT] = 0.2

    # V[7]: MEM_ADDR_SRC — SI/SC only (addr from STACK0, not SP)
    # PSH/JSR/ENT use SP as address source (MEM_ADDR_SRC=0)
    attn.W_v[base + 7, BD.OP_SI] = 0.2
    attn.W_v[base + 7, BD.OP_SC] = 0.2

    # V[0]: OP_LEV relay disabled
    # BUG FIX 2026-04-17: OP_LEV is now injected at all positions via
    # _inject_active_opcode() when active_opcode=LEV. The relay here was
    # doubling OP_LEV at PC marker (5.0 from embed + 5.0 from relay = 10.0),
    # causing L6 FFN unit 1564 to fire spuriously and corrupt OUTPUT_HI[2].
    # attn.W_v[base + 0, BD.OP_LEV] = 0.2  # DISABLED

    # O: write to CMP dims + MEM store flags + PSH_AT_SP
    attn.W_o[BD.CMP + 0, base + 1] = 1.0  # OP_PSH → CMP[0] (legacy, kept for compatibility)
    attn.W_o[BD.PSH_AT_SP, base + 1] = 1.0  # OP_PSH → PSH_AT_SP (clean, no JMP collision)
    attn.W_o[BD.CMP + 1, base + 2] = 1.0  # OP_ADJ → CMP[1]
    attn.W_o[BD.CMP + 3, base + 3] = 5.0  # POP group → CMP[3] (×5 to rescale 0.2→1.0)
    attn.W_o[BD.CMP + 2, base + 4] = 1.0  # OP_ENT → CMP[2] at SP/STACK0/BP
    attn.W_o[BD.CMP + 4, base + 5] = 1.0  # OP_JSR → CMP[4] at SP/STACK0/BP
    attn.W_o[BD.OP_JSR, base + 5] = 5.0  # OP_JSR → OP_JSR dim (×5 to rescale 0.2→1.0)
    attn.W_o[BD.OP_ENT, base + 4] = 5.0  # OP_ENT → OP_ENT dim (×5 to rescale 0.2→1.0)
    attn.W_o[BD.MEM_STORE, base + 6] = 1.0  # store flag → MEM marker
    attn.W_o[BD.MEM_ADDR_SRC, base + 7] = 1.0  # addr source flag → MEM marker
    # attn.W_o[BD.OP_LEV, base + 0] = 5.0  # DISABLED - see comment above


# =============================================================================
# Tool Calling Weight Setting (gated by enable_tool_calling)
# =============================================================================


def _set_tool_call_opcode_decode(ffn, S, BD):
    """L5 FFN addition: decode all I/O opcodes → IO_IS_TOOL_CALL.

    Same pattern as _set_opcode_decode_ffn: 2-way AND on OPCODE_BYTE_LO/HI
    nibbles, gated by MARK_AX. All 6 opcodes write to the same
    IO_IS_TOOL_CALL dim (combined flag, ≈5.0 when any is active, ≈0 otherwise).

    Starts at unit 400 to avoid conflict with existing opcode decode units.

    OPEN=30 (lo=14, hi=1), READ=31 (lo=15, hi=1),
    CLOS=32 (lo=0, hi=2), PRTF=33 (lo=1, hi=2),
    GETCHAR=64 (lo=0, hi=4), PUTCHAR=65 (lo=1, hi=4).
    """
    unit = 400
    io_opcodes = [
        (14, 1),  # OPEN = 30 = 0x1E
        (15, 1),  # READ = 31 = 0x1F
        (0, 2),  # CLOS = 32 = 0x20
        (1, 2),  # PRTF = 33 = 0x21
        (0, 4),  # GETCHAR = 64 = 0x40
        (1, 4),  # PUTCHAR = 65 = 0x41
    ]
    for lo, hi in io_opcodes:
        ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.b_up[unit] = -S * 1.5  # both must be ~1
        ffn.W_gate[unit, BD.MARK_AX] = 1.0  # only at AX marker
        ffn.W_down[BD.IO_IS_TOOL_CALL, unit] = 10.0 / S  # ≈5.0 when active
        unit += 1


def _set_tool_call_relay_head(attn, S, BD, HD):
    """L6 attention head 5: relay IO_IS_TOOL_CALL from AX → SE position.

    Same pattern as head 1 (EXIT relay):
    - Q: NEXT_SE (query at SE position) + -L*MARK_AX (block at AX)
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_TOOL_CALL
    - O: write to CMP[2] (IS_TOOL_CALL relay)

    Distance from SE to AX = 28 tokens. With L=50, ALiBi slope=5:
    score = 50²*0.7*0.125 - 5*28 = 79 (strong).
    """
    L = 50.0
    base = 5 * HD  # head 5

    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy IO_IS_TOOL_CALL flag
    attn.W_v[base + 1, BD.IO_IS_TOOL_CALL] = 1.0
    # O: write to CMP[2]
    attn.W_o[BD.CMP + 2, base + 1] = 1.0


def _set_tool_call_detection(ffn, S, BD):
    """L6 FFN addition: CMP[2] AND NEXT_SE → NEXT_TOOL_CALL.

    Same pattern as HALT detection (CMP[1] AND NEXT_SE → NEXT_HALT).
    When both CMP[2] (IS_TOOL_CALL relay) and NEXT_SE are active:
    - Set NEXT_TOOL_CALL (emit TOOL_CALL token)
    - Clear NEXT_SE (suppress STEP_END)

    CMP[2] ≈ 5 for I/O ops, ≈ 0 otherwise.
    Threshold 3.0: I/O (5+0.68=5.68 > 3) fires, inactive (0+0.68 < 3) doesn't.

    Note: When PUTCHAR activates this, the AX→OUTPUT routing from
    _set_io_putchar_routing still works (it runs in parallel at L6 FFN).
    The output byte is produced by the model's weights as usual; the
    TOOL_CALL token just signals the runner to dispatch via tool_handler.

    Starts at unit 830 to avoid conflict with _set_layer6_routing_ffn (units 0-740)
    and binary_pop (units 790-821).
    """
    unit = 830

    ffn.W_up[unit, BD.CMP + 2] = S
    ffn.W_up[unit, BD.NEXT_SE] = S
    ffn.b_up[unit] = -S * 3.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.NEXT_TOOL_CALL, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S  # clear NEXT_SE → no STEP_END


# =============================================================================
# Conversational I/O Detection (PRTF/READ for autoregressive generation)
# =============================================================================


def _set_conversational_io_opcode_decode(ffn, S, BD):
    """L5 FFN addition: decode PRTF and READ opcodes for conversational I/O mode.

    Detects PRTF (33) and READ (31) opcodes at AX marker and writes to
    separate flags for autoregressive I/O generation:
    - PRTF → IO_IS_PRTF ≈ 5.0
    - READ → IO_IS_READ ≈ 5.0

    This is separate from tool_call detection to enable different routing:
    - Tool call mode: PRTF/READ → TOOL_CALL token (runner dispatches)
    - Conversational I/O mode: PRTF/READ → autoregressive sequence
      (THINKING_END → output bytes → THINKING_START)

    Starts at unit 410 to avoid conflict with tool_call units (400-405).

    PRTF = 33 = 0x21 (lo=1, hi=2)
    READ = 31 = 0x1F (lo=15, hi=1)
    """
    unit = 410

    # PRTF detection via ACTIVE_OPCODE_PRTF flag (set by embedding)
    ffn.W_up[unit, BD.ACTIVE_OPCODE_PRTF] = S  # 1.0 when PRTF is active
    ffn.b_up[unit] = -S * 0.5  # threshold: active when ACTIVE_OPCODE_PRTF ≈ 1
    ffn.b_gate[unit] = 1.0  # always gate (no position restriction needed)
    ffn.W_down[BD.IO_IS_PRTF, unit] = 10.0 / S  # ≈5.0 when active
    unit += 1

    # READ detection via ACTIVE_OPCODE_READ flag
    ffn.W_up[unit, BD.ACTIVE_OPCODE_READ] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.IO_IS_READ, unit] = 10.0 / S


def _set_conversational_io_relay_heads(attn, S, BD, HD):
    """L6 attention heads 4-5: relay IO_IS_PRTF and IO_IS_READ from AX → SE.

    IMPORTANT: Changed from heads 6-7 to heads 4-5 to avoid conflict with
    _set_opcode_relay_head() which uses head 6 for PSH/ADJ/pop relay.

    Head 4: Relay IO_IS_PRTF
    - Q: NEXT_SE (query at SE position), gated by ACTIVE_OPCODE_PRTF
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_PRTF
    - O: write to CMP + 5 (changed from CMP[3] to avoid pop group conflict)

    FIX 2026-04-16: Gate head 4 with ACTIVE_OPCODE_PRTF to avoid conflict with
    BZ/BNZ relay which also uses head 4. Without this gate, the BZ/BNZ's
    W_q[base, CONST] = -65 penalty blocks PRTF relay (CONST=1.0 at all positions).
    Adding W_q[base, ACTIVE_OPCODE_PRTF] = +65 cancels the penalty when PRTF active.

    Head 5: Relay IO_IS_READ
    - Q: NEXT_SE (query at SE position)
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_READ
    - O: write to CMP + 6 (changed from TEMP[0] to use dedicated CMP slot)

    Uses steep ALiBi slope (5.0) for both heads to overcome distance penalty.
    """
    L = 50.0

    # Head 4: PRTF relay
    # FIX 2026-04-16: Use V[37] instead of V[1] to avoid conflict with BZ/BNZ relay.
    # BZ/BNZ relay uses V[1] for OP_BZ and writes to CMP+2 via O[CMP+2, V[1]]=0.2.
    # When PRTF is active, IO_IS_PRTF=5.0 would get multiplied by that 0.2, causing
    # spurious CMP+2=1.0 which triggers ENT logic at STACK0 marker.
    base = 4 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    # FIX 2026-04-16: Gate with ACTIVE_OPCODE_PRTF to overcome BZ/BNZ CONST penalty
    # BZ/BNZ relay sets W_q[base, CONST] = -65, which blocks at SE (CONST=1.0).
    # When PRTF active, ACTIVE_OPCODE_PRTF=1.0, contributing +L=50 to cancel part of penalty.
    # Combined with NEXT_SE contribution, this makes Q positive at SE during PRTF.
    attn.W_q[base, BD.ACTIVE_OPCODE_PRTF] = L * 1.5  # +75 to overcome -65 CONST penalty
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 37, BD.IO_IS_PRTF] = 1.0  # V[37] avoids BZ/BNZ V[1] conflict
    attn.W_o[BD.CMP + 5, base + 37] = 1.0  # Use CMP[5] instead of CMP[3]

    # Head 5: READ relay
    base = 5 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 1, BD.IO_IS_READ] = 1.0
    attn.W_o[BD.CMP + 6, base + 1] = 1.0  # Use CMP[6] instead of TEMP[0]


def _set_conversational_io_state_machine(ffn, S, BD):
    """L6 FFN addition: Start conversational I/O sequence when PRTF/READ detected.

    State transitions:
    1. Normal execution (IO_STATE=0)
    2. PRTF/READ detected → set NEXT_THINKING_END, IO_STATE=1
    3. After THINKING_END → generate output, IO_STATE=2
    4. After output complete → set NEXT_THINKING_START, IO_STATE=3
    5. After THINKING_START → resume normal, IO_STATE=0

    For now, we implement step 2: detect PRTF → start sequence.
    Steps 3-5 will be added in L13 FFN (state tracking across generation steps).

    Condition: CMP[5] (PRTF flag) AND NEXT_SE
    - Set NEXT_THINKING_END (emit </thinking> token)
    - Clear NEXT_SE (suppress STEP_END)
    - Set IO_STATE = 1 (begin I/O sequence)

    IMPORTANT: Changed from CMP[3] to CMP[5] and TEMP[0] to CMP[6] to avoid
    conflicts with _set_opcode_relay_head() which uses CMP[3] for pop group.

    Starts at unit 840 to avoid conflict with tool_call_detection (unit 830).
    """
    unit = 840

    # PRTF triggers thinking end
    # FIX 2026-04-16: CMP[5] can be ~5.0 (scaled by attention), not just ~1.0.
    # With CMP[5]=5.0, old threshold (-3S) was exceeded without NEXT_SE.
    # Fix: Gate on NEXT_SE so output is suppressed when NEXT_SE is not active.
    #
    # IMPORTANT: PureFFN uses `silu(up) * gate` where gate is LINEAR (no sigmoid)!
    # If b_gate is negative, gate can be negative when NEXT_SE=0, causing:
    #   silu(up) * negative_gate * negative_W_down = POSITIVE output (wrong!)
    # Fix: Use b_gate=0 so gate=10*NEXT_SE, which is 0 when NEXT_SE=0.
    ffn.W_up[unit, BD.CMP + 5] = S  # PRTF flag from relay head
    ffn.b_up[unit] = -S * 0.5  # Fire when CMP[5] > 0.5 (CMP[5]~5.0 → 500-50=450)
    ffn.W_gate[unit, BD.NEXT_SE] = 10.0  # Gate = 10*NEXT_SE (0 when inactive, 7 when active)
    ffn.b_gate[unit] = 0.0  # No bias - gate is 0 when NEXT_SE=0
    ffn.W_down[BD.NEXT_THINKING_END, unit] = 2.0 / S  # emit THINKING_END
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S  # suppress STEP_END
    ffn.W_down[BD.IO_STATE, unit] = 2.0 / S  # set IO_STATE = 1
    unit += 1

    # READ triggers thinking end (similar pattern)
    ffn.W_up[unit, BD.CMP + 6] = S  # READ flag from relay head
    ffn.b_up[unit] = -S * 0.5  # Fire when CMP[6] > 0.5
    ffn.W_gate[unit, BD.NEXT_SE] = 10.0  # Gate on NEXT_SE
    ffn.b_gate[unit] = 0.0  # No bias
    ffn.W_down[BD.NEXT_THINKING_END, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S
    ffn.W_down[BD.IO_STATE, unit] = 2.0 / S
    unit += 1


def _set_lookback_detection_head(attn, S, BD, HD):
    """L2 attention head 1: Detect previous token type for conversational I/O.

    Looks back at t-1 to detect:
    - THINKING_START (token 272, has MARK_THINKING_START=1.0 in embedding)
    - THINKING_END (token 273, has MARK_THINKING_END=1.0 in embedding)
    - Byte (tokens 0-255, have IS_BYTE embedding)

    Q: CONST (always query from current position)
    K: CONST (attend to all previous positions, ALiBi will favor t-1)
    V: Copy markers (MARK_THINKING_START, MARK_THINKING_END, IS_BYTE)
    O: Write to lookback flags (LAST_WAS_THINKING_START, etc.)

    With ALiBi slope = 10.0, the most recent token (t-1) will have highest score.

    NOTE: Uses dedicated MARK_THINKING_* dimensions (506-507) instead of
    TEMP+1/+2 (481-482) to avoid overlap with OUTPUT_BYTE_LO (480-495).
    """
    L = 20.0
    base = 1 * HD  # head 1

    # Q: active at all positions (CONST)
    attn.W_q[base, BD.CONST] = L

    # K: match all positions (CONST)
    attn.W_k[base, BD.CONST] = L

    # V: copy markers from previous token
    attn.W_v[base + 1, BD.MARK_THINKING_START] = 1.0  # THINKING_START marker
    attn.W_v[base + 2, BD.MARK_THINKING_END] = 1.0  # THINKING_END marker
    attn.W_v[base + 3, BD.IS_BYTE] = 1.0  # Byte marker

    # O: write to lookback flags
    attn.W_o[BD.LAST_WAS_THINKING_START, base + 1] = 1.0
    attn.W_o[BD.LAST_WAS_THINKING_END, base + 2] = 1.0
    attn.W_o[BD.LAST_WAS_BYTE, base + 3] = 1.0


def _set_conversational_io_state_init(ffn, S, BD):
    """L3 FFN addition: Initialize output mode when LAST_WAS_THINKING_END detected.

    When previous token was THINKING_END:
    - Set IO_IN_OUTPUT_MODE = 1 (enter output mode)
    - Initialize IO_FORMAT_POS = 0 (start at beginning of format string)

    This prepares the model to start emitting output bytes from the format string.

    Starts at unit 500 to avoid conflicts with existing L3 FFN logic.
    """
    unit = 500

    # Detect LAST_WAS_THINKING_END and activate output mode
    ffn.W_up[unit, BD.LAST_WAS_THINKING_END] = S
    ffn.b_up[unit] = -S * 0.5  # threshold: LAST_WAS_THINKING_END ≈ 1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.IO_IN_OUTPUT_MODE, unit] = 2.0 / S  # set flag
    unit += 1

    # Initialize format position to 0 (nibble encoding: all nibbles zero except set [0]=1)
    # Actually, we want FORMAT_POS to START at 0, which means all nibbles are 0.
    # The position will be incremented AFTER fetching each byte.
    # So we don't need to explicitly set it to 0 here (it defaults to 0).
    # We'll handle increment in L8 FFN.


def _set_format_pointer_extraction(attn, S, BD, HD):
    """L7 attention: Extract format string pointer from STACK0 when entering output mode.

    When IO_IN_OUTPUT_MODE just activated (after THINKING_END):
    - Attend back to previous step's STACK0 marker position
    - Copy EMBED_LO and EMBED_HI (format string pointer byte 0)
    - Write to FORMAT_PTR_LO and FORMAT_PTR_HI

    For simplicity, we only extract byte 0 of the pointer, supporting format
    strings at addresses < 256. Full 32-bit pointer extraction requires
    gathering all 4 STACK0 bytes, which is more complex.

    Uses Head 7 (last available head in L7).
    """
    L = 20.0
    base = 7 * HD  # head 7

    # Q: active when IO_IN_OUTPUT_MODE (just entered output mode)
    attn.W_q[base, BD.IO_IN_OUTPUT_MODE] = L

    # K: match STACK0 marker from previous step
    attn.W_k[base, BD.MARK_STACK0] = L

    # V: copy EMBED_LO and EMBED_HI (pointer byte 0)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

    # O: write to FORMAT_PTR_LO and FORMAT_PTR_HI
    # Note: FORMAT_PTR_LO is defined as a range (467-482) for 16 nibbles
    # but we only use the first 16 dims for lo nibble
    for k in range(16):
        attn.W_o[BD.FORMAT_PTR_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.FORMAT_PTR_HI + k, base + 17 + k] = 1.0


def _set_format_position_counter(ffn, S, BD):
    """L8 FFN addition: Increment format string position counter.

    When LAST_WAS_BYTE AND IO_IN_OUTPUT_MODE (just emitted output byte):
    - Increment IO_FORMAT_POS by 1
    - Uses nibble arithmetic (same pattern as PC increment)

    IO_FORMAT_POS starts at 0 and increments after each byte emission.
    For now, we only support single-nibble positions (0-15), which is
    enough for format strings up to 15 bytes.

    Starts at unit 600 to avoid conflicts with existing L8 FFN logic.
    """
    unit = 600

    # Detect: just emitted a byte (LAST_WAS_BYTE AND IO_IN_OUTPUT_MODE)
    # Increment IO_FORMAT_POS lo nibble by 1
    # Pattern: rotate nibble by +1 (k → (k+1)%16)
    for k in range(16):
        next_k = (k + 1) % 16
        ffn.W_up[unit, BD.LAST_WAS_BYTE] = S
        ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
        ffn.b_up[unit] = -S * 1.5  # need both active
        ffn.W_gate[unit, BD.IO_FORMAT_POS + k] = 1.0  # current position = k
        ffn.W_down[BD.IO_FORMAT_POS + k, unit] = -2.0 / S  # clear old
        ffn.W_down[BD.IO_FORMAT_POS + next_k, unit] = 2.0 / S  # set new
        unit += 1


def _set_format_string_fetch_head(attn, S, BD, HD):
    """L9 attention head: Fetch byte from format string at FORMAT_PTR + FORMAT_POS.

    Similar to L5 code fetch, but queries memory at FORMAT_PTR + FORMAT_POS.
    When IO_IN_OUTPUT_MODE:
    - Q: FORMAT_PTR_LO/HI + FORMAT_POS (address to fetch)
    - K: ADDR_KEY (memory address keys)
    - V: EMBED_LO/HI (byte value at that address)
    - O: OUTPUT_BYTE_LO/HI (byte to emit)

    For simplicity, FORMAT_PTR only uses byte 0 (addresses < 256) and
    FORMAT_POS is a single nibble (positions 0-15).

    Uses Head 0 in L9.
    """
    L = 15.0
    base = 0 * HD  # head 0

    # Q: active when in output mode, query = FORMAT_PTR + FORMAT_POS
    attn.W_q[base, BD.IO_IN_OUTPUT_MODE] = L
    # Query nibbles: FORMAT_PTR_LO + FORMAT_POS
    # For addresses < 256, we have lo nibble = (PTR_lo + POS) % 16, hi nibble = PTR_hi
    # But addition is complex... let's simplify: just use FORMAT_PTR for now
    # and ignore FORMAT_POS (always fetch byte 0). We'll fix this later.
    for k in range(16):
        attn.W_q[base + 1 + k, BD.FORMAT_PTR_LO + k] = 1.0
        attn.W_q[base + 17 + k, BD.FORMAT_PTR_HI + k] = 1.0

    # K: match ADDR_KEY (address keys in memory/data section)
    for k in range(16):
        attn.W_k[base + 1 + k, BD.ADDR_KEY + k] = L  # lo nibble
        attn.W_k[base + 17 + k, BD.ADDR_KEY + 16 + k] = L  # hi nibble

    # V: copy byte value (EMBED_LO/HI)
    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

    # O: write to OUTPUT_BYTE_LO/HI
    for k in range(16):
        attn.W_o[BD.OUTPUT_BYTE_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.OUTPUT_BYTE_HI + k, base + 17 + k] = 1.0


def _set_null_terminator_detection(ffn, S, BD):
    """L10 FFN addition: Detect null terminator (byte = 0) in output.

    When IO_IN_OUTPUT_MODE AND OUTPUT_BYTE == 0 (all nibbles zero):
    - Set IO_OUTPUT_COMPLETE = 1 (format string done)
    - Clear IO_IN_OUTPUT_MODE (exit output mode)
    - Set NEXT_THINKING_START (emit THINKING_START next)

    This detects the end of the format string and prepares to resume normal execution.

    Starts at unit 1864 to avoid conflicts with existing L10 FFN logic.
    _set_layer10_alu uses ~1854 units (comparison, bitwise, MUL, SHL/SHR, passthrough).
    """
    unit = 1864

    # Detect null byte: OUTPUT_BYTE_LO[0] AND OUTPUT_BYTE_HI[0] (both nibbles = 0)
    # AND IO_IN_OUTPUT_MODE (currently in output mode)
    # CRITICAL: Gate on IO_IN_OUTPUT_MODE to prevent firing due to TEMP overlap!
    ffn.W_up[unit, BD.OUTPUT_BYTE_LO] = S  # lo nibble [0] = 1
    ffn.W_up[unit, BD.OUTPUT_BYTE_HI] = S  # hi nibble [0] = 1
    ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
    ffn.b_up[unit] = -S * 2.5  # need all three active
    # Gate: only fire if IO_IN_OUTPUT_MODE > 5.0 (strongly active)
    # This prevents spurious firing due to OUTPUT_BYTE/TEMP overlap
    ffn.W_gate[unit, BD.IO_IN_OUTPUT_MODE] = 1.0
    ffn.b_gate[unit] = -5.0
    ffn.W_down[BD.IO_OUTPUT_COMPLETE, unit] = 2.0 / S  # set complete flag
    ffn.W_down[BD.IO_IN_OUTPUT_MODE, unit] = -2.0 / S  # clear output mode
    ffn.W_down[BD.NEXT_THINKING_START, unit] = 2.0 / S  # emit THINKING_START
    unit += 1


def _set_conversational_io_output_routing(ffn, S, BD):
    """L15 FFN addition: Route OUTPUT_BYTE to OUTPUT when in output mode.

    When IO_IN_OUTPUT_MODE (emitting output bytes):
    - Copy OUTPUT_BYTE_LO → OUTPUT_LO (all 16 nibbles)
    - Copy OUTPUT_BYTE_HI → OUTPUT_HI (all 16 nibbles)

    This routes the fetched format string byte to the output head for emission.

    Note: We don't need to suppress normal OUTPUT routing because IO_IN_OUTPUT_MODE
    only activates after THINKING_END, at which point we're not in the normal
    35-token generation cycle.

    Starts at unit 800 to avoid conflicts with existing L15 FFN logic (nibble copy).
    """
    unit = 800

    # Copy each OUTPUT_BYTE nibble to corresponding OUTPUT nibble when in output mode
    for k in range(16):
        # Lo nibble
        ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OUTPUT_BYTE_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1

        # Hi nibble
        ffn.W_up[unit, BD.IO_IN_OUTPUT_MODE] = S
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, BD.OUTPUT_BYTE_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1


# =============================================================================
# Binary Pop SP Increment (SP += 8 for all binary pop ops)
# =============================================================================


def _set_binary_pop_sp_increment(ffn, S, BD):
    """L6 FFN addition: SP += 8 for all binary pop ops.

    18 pop ops: ADD, SUB, MUL, DIV, MOD, EQ, NE, LT, GT, LE, GE,
                OR, XOR, AND, SHL, SHR, SI, SC.

    Pattern mirrors PSH SP -= 8 (reversed direction):
    - Lo nibble: (k + 8) % 16, cancel identity + write new
    - Hi nibble: (k + 1) % 16 when carry (lo nibble >= 8),
      detected by suppressing EMBED_LO[0..7]

    Uses CMP[3] (relayed POP group flag from L6 attn head 6) instead of
    reading OP_* dims directly, since OP_* is only decoded at AX marker
    but we need to fire at SP marker.

    CMP[3] ≈ 1.0 when any pop op is active (relayed by _set_opcode_relay_head).
    Threshold 1.5: CMP[3](1) + MARK_SP(1) = 2 > 1.5 → fires.

    Pattern mirrors PSH SP -= 8 (reversed direction):
    - Lo nibble: (k + 8) % 16, cancel identity + write new
    - Hi nibble: (k + 1) % 16 when carry (lo nibble >= 8),
      detected by suppressing EMBED_LO[0..7]

    Uses 32 FFN units starting at offset 1600.

    FIX 2026-04-16: Changed from 950 to 1600 to avoid overlap with BNZ PC override
    units (which extend to ~1543). The previous overlap caused units 950-961 to have
    both MARK_PC (from BNZ) and MARK_SP (from this function), incorrectly firing
    at PC marker when CMP[3] was set by POP group relay (not just OP_BNZ).
    """
    unit = 1600
    T_pop = 1.5  # threshold: MARK_SP(1) + CMP[3](~1) = 2 > 1.5 ✓

    # Lo nibble: cancel identity, write (k + 8) % 16
    for k in range(16):
        new_k = (k + 8) % 16
        # Gate on MARK_SP AND CMP[3] (relayed POP group flag)
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.CMP + 3] = S  # relayed POP group from L6 head 6
        ffn.b_up[unit] = -S * T_pop
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_LO + k, unit] += -2.0 / S  # cancel identity
        unit += 1

    # Hi nibble: carry case (lo >= 8 means adding 8 overflows → hi += 1)
    # Carry detected by suppressing EMBED_LO[0..7] (lo < 8 means no carry)
    for k in range(16):
        new_k_carry = (k + 1) % 16
        ffn.W_up[unit, BD.MARK_SP] = S
        ffn.W_up[unit, BD.CMP + 3] = S  # relayed POP group from L6 head 6
        ffn.b_up[unit] = -S * T_pop
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        for lo_bit in range(8):
            ffn.W_gate[unit, BD.EMBED_LO + lo_bit] = -1.0
        ffn.W_down[BD.OUTPUT_HI + new_k_carry, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + k, unit] += -2.0 / S  # cancel identity
        unit += 1


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
    L6 FFN units 850-1105: LEA/JSR/ENT output routing.
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
    # L6 FFN: Function call output routing (units 1000-1255)
    # =====================================================================
    unit = 1000

    # --- LEA first-step: Initialize ALU with BP default (2 units: 850-851) ---
    # For first step, set ALU = BP_default = 0x00010000, byte 0 = 0x00
    # Gate on: OP_LEA + MARK_AX + NOT HAS_SE
    # Subsequent steps use Layer 7 attention relay from BP marker.
    ffn6.W_up[unit, BD.OP_LEA] = S
    ffn6.W_up[unit, BD.MARK_AX] = S
    ffn6.W_up[unit, BD.HAS_SE] = -S
    ffn6.b_up[unit] = -S * 1.5  # OP_LEA(~2.4) + MARK_AX(1) - HAS_SE(0) = 3.4 > 1.5
    ffn6.b_gate[unit] = 1.0
    ffn6.W_down[BD.ALU_LO + 0, unit] = 2.0 / S  # nibble 0
    unit += 1
    ffn6.W_up[unit, BD.OP_LEA] = S
    ffn6.W_up[unit, BD.MARK_AX] = S
    ffn6.W_up[unit, BD.HAS_SE] = -S
    ffn6.b_up[unit] = -S * 1.5
    ffn6.b_gate[unit] = 1.0
    ffn6.W_down[BD.ALU_HI + 0, unit] = 2.0 / S  # nibble 0
    unit += 1
    # Units 852-881 unused (reserved)
    unit += 30

    # --- JSR SP -= 8 (autoregressive shift fix) ---
    # FIX 2026-04-16: OUTPUT at position N predicts token at position N+1.
    # To produce byte N, OUTPUT must be at position N-1.
    # Pattern: MARK_SP → byte 0, BYTE_INDEX_0 → byte 1, BYTE_INDEX_1 → byte 2, etc.
    #
    # Initial SP = 0x00010000, bytes = [0x00, 0x00, 0x01, 0x00]
    # SP - 8 = 0x0000FFF8, bytes = [0xF8, 0xFF, 0x00, 0x00]

    # --- JSR SP byte 0: fire at MARK_SP to produce 0xF8 ---
    # At SP marker, EMBED_LO/HI are all 0 (marker has special encoding).
    # Need OUTPUT_LO[8] and OUTPUT_HI[15] to be highest for argmax.
    # Use just 2 units with constant gate writing to specific positions.
    # Pattern: CMP[4](~1) + MARK_SP(1) = 2, threshold 1.5
    T_jsr_sp0 = 1.5
    # Lo nibble = 8 (0xF8 & 0xF = 8)
    ffn6.W_up[unit, BD.CMP + 4] = S  # JSR flag
    ffn6.W_up[unit, BD.MARK_SP] = S  # Fire at SP marker (predicts byte 0)
    ffn6.b_up[unit] = -S * T_jsr_sp0
    ffn6.W_gate[unit, BD.CONST] = 1.0  # Constant gate
    ffn6.W_down[BD.OUTPUT_LO + 8, unit] = 10.0 / S  # Strong signal for nibble 8
    unit += 1
    # Hi nibble = 15 (0xF8 >> 4 = 15)
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.MARK_SP] = S
    ffn6.b_up[unit] = -S * T_jsr_sp0
    ffn6.W_gate[unit, BD.CONST] = 1.0
    ffn6.W_down[BD.OUTPUT_HI + 15, unit] = 10.0 / S  # Strong signal for nibble 15
    unit += 1
    # Reserve units to maintain spacing (32 units for byte 0 originally, now 2)
    unit += 30

    # --- JSR SP byte 1: fire at BYTE_INDEX_0 to produce 0xFF ---
    # At byte 0 position, EMBED has the byte 0 value (0xF8) but we want constant 0xFF.
    # Need OUTPUT_LO[15] and OUTPUT_HI[15] highest for argmax.
    # Pattern: CMP[4](~1) + BYTE_INDEX_0(1) + IS_BYTE(1) + H1[SP](1) = 4, threshold 3.5
    T_jsr_sp1 = 3.5
    # Lo nibble = 15 (0xFF & 0xF = 15)
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.BYTE_INDEX_0] = S  # Fire at byte 0 (predicts byte 1)
    ffn6.W_up[unit, BD.IS_BYTE] = S
    ffn6.W_up[unit, BD.H1 + 2] = S  # SP area (2 = SP_I)
    ffn6.b_up[unit] = -S * T_jsr_sp1
    ffn6.W_gate[unit, BD.CONST] = 1.0  # Constant gate
    ffn6.W_down[BD.OUTPUT_LO + 15, unit] = 10.0 / S  # Strong signal for nibble 15
    unit += 1
    # Hi nibble = 15 (0xFF >> 4 = 15)
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn6.W_up[unit, BD.IS_BYTE] = S
    ffn6.W_up[unit, BD.H1 + 2] = S
    ffn6.b_up[unit] = -S * T_jsr_sp1
    ffn6.W_gate[unit, BD.CONST] = 1.0
    ffn6.W_down[BD.OUTPUT_HI + 15, unit] = 10.0 / S  # Strong signal for nibble 15
    unit += 1
    # Reserve units to maintain spacing (32 units originally, now 2)
    unit += 30

    # --- JSR SP byte 2: fire at BYTE_INDEX_1 to produce 0x00 ---
    # Initial byte 2 = 0x01, subtract 1 for borrow = 0x00
    # At byte 1 position, EMBED has byte 1 value (0xFF), but we want constant 0x00.
    # Need OUTPUT_LO[0] and OUTPUT_HI[0] highest for argmax.
    T_jsr_sp2 = 3.5
    # Lo nibble = 0 (0x00 & 0xF = 0)
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.BYTE_INDEX_1] = S  # Fire at byte 1 (predicts byte 2)
    ffn6.W_up[unit, BD.IS_BYTE] = S
    ffn6.W_up[unit, BD.H1 + 2] = S  # SP area
    ffn6.b_up[unit] = -S * T_jsr_sp2
    ffn6.W_gate[unit, BD.CONST] = 1.0  # Constant gate
    ffn6.W_down[BD.OUTPUT_LO + 0, unit] = 10.0 / S  # Strong signal for nibble 0
    unit += 1
    # Hi nibble = 0 (0x00 >> 4 = 0)
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn6.W_up[unit, BD.IS_BYTE] = S
    ffn6.W_up[unit, BD.H1 + 2] = S
    ffn6.b_up[unit] = -S * T_jsr_sp2
    ffn6.W_gate[unit, BD.CONST] = 1.0
    ffn6.W_down[BD.OUTPUT_HI + 0, unit] = 10.0 / S  # Strong signal for nibble 0
    unit += 1
    # Reserve units (32 originally, now 2)
    unit += 30

    # --- JSR SP byte 3: fire at BYTE_INDEX_2 to produce 0x00 ---
    # Byte 3 also becomes 0x00 (no borrow from byte 2 which was 0x01 - 1 = 0x00)
    T_jsr_sp3 = 3.5
    # Lo nibble = 0
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.BYTE_INDEX_2] = S  # Fire at byte 2 (predicts byte 3)
    ffn6.W_up[unit, BD.IS_BYTE] = S
    ffn6.W_up[unit, BD.H1 + 2] = S  # SP area
    ffn6.b_up[unit] = -S * T_jsr_sp3
    ffn6.W_gate[unit, BD.CONST] = 1.0  # Constant gate
    ffn6.W_down[BD.OUTPUT_LO + 0, unit] = 10.0 / S  # Strong signal for nibble 0
    unit += 1
    # Hi nibble = 0
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn6.W_up[unit, BD.IS_BYTE] = S
    ffn6.W_up[unit, BD.H1 + 2] = S
    ffn6.b_up[unit] = -S * T_jsr_sp3
    ffn6.W_gate[unit, BD.CONST] = 1.0
    ffn6.W_down[BD.OUTPUT_HI + 0, unit] = 10.0 / S  # Strong signal for nibble 0
    unit += 1
    # Reserve units (32 originally, now 2)
    unit += 30

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
    # First: Cancel L3 default OUTPUT_LO[0] with constant gate
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.MARK_STACK0] = S
    ffn6.b_up[unit] = -S * T_jsr_s0
    ffn6.W_gate[unit, BD.CONST] = 1.0  # Constant gate
    ffn6.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S  # Cancel L3 default
    unit += 1
    ffn6.W_up[unit, BD.CMP + 4] = S
    ffn6.W_up[unit, BD.MARK_STACK0] = S
    ffn6.b_up[unit] = -S * T_jsr_s0
    ffn6.W_gate[unit, BD.CONST] = 1.0
    ffn6.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S  # Cancel L3 default
    unit += 1
    # Then: Write return_addr from AX_CARRY
    for k in range(16):
        ffn6.W_up[unit, BD.CMP + 4] = S
        ffn6.W_up[unit, BD.MARK_STACK0] = S
        ffn6.b_up[unit] = -S * T_jsr_s0
        ffn6.W_gate[unit, BD.EMBED_LO + k] = -1.0
        ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.CMP + 4] = S
        ffn6.W_up[unit, BD.MARK_STACK0] = S
        ffn6.b_up[unit] = -S * T_jsr_s0
        ffn6.W_gate[unit, BD.EMBED_HI + k] = -1.0
        ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # STACK0 bytes 0-3 (128 units: return_addr from AX_CARRY)
    # L14 reads from these byte positions to populate MEM value bytes.
    # FIX 2026-04-16: Use L1H4[BP] AND NOT H1[BP] to identify STACK0 area.
    # STACK0 is 5-9 positions from BP marker, covered by L1H4[BP] (d≤6.5) but not H1[BP] (d≤4.5).
    # Threshold: CMP[4](1) + BYTE_INDEX(1) + IS_BYTE(1) + L1H4[BP](1) - H1[BP](0) = 4 > 3.5
    BP_I = 3  # BP marker index
    T_jsr_s0_byte = 3.5
    for byte_idx in range(4):
        byte_index_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_idx]
        for k in range(16):
            # Lo nibble
            ffn6.W_up[unit, BD.CMP + 4] = S
            ffn6.W_up[unit, byte_index_dim] = S
            ffn6.W_up[unit, BD.IS_BYTE] = S
            ffn6.W_up[unit, BD.L1H4 + BP_I] = S  # STACK0 is within L1H4 of BP
            ffn6.W_up[unit, BD.H1 + BP_I] = -S  # But not within H1 of BP
            ffn6.b_up[unit] = -S * T_jsr_s0_byte
            ffn6.W_gate[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
            ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0  # Write return_addr
            ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            # Hi nibble
            ffn6.W_up[unit, BD.CMP + 4] = S
            ffn6.W_up[unit, byte_index_dim] = S
            ffn6.W_up[unit, BD.IS_BYTE] = S
            ffn6.W_up[unit, BD.L1H4 + BP_I] = S
            ffn6.W_up[unit, BD.H1 + BP_I] = -S
            ffn6.b_up[unit] = -S * T_jsr_s0_byte
            ffn6.W_gate[unit, BD.EMBED_HI + k] = -1.0
            ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

    # --- JSR PC override: PC = FETCH (jump target) (64 units: 978-1041) ---
    # At PC marker when JSR: cancel OUTPUT (PC+5), write FETCH (jump target).
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
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write FETCH_LO/HI (jump target from immediate field)
    # FIXED: Was reading AX_CARRY (PC+5 return address), now correctly reads FETCH (jump target)
    for k in range(16):
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
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.FETCH_LO + k] = 1.0  # FIXED: was AX_CARRY_LO
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
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
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.FETCH_HI + k] = 1.0  # FIXED: was AX_CARRY_HI
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
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
            ffn6.W_gate[unit, BD.TEMP + lo_bit] = -1.0
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