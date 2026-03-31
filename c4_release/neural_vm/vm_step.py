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
    VOCAB_SIZE = 272

    STEP_TOKENS = (
        35  # Tokens per VM step: PC(5)+AX(5)+SP(5)+BP(5)+STACK0(5)+MEM(9)+SE(1)
    )


class AutoregressiveAttention(nn.Module):
    """Multi-head attention with softmax1 (ZFOD) and ALiBi positional bias.

    NOT a PureAttention subclass — PureAttention.forward() is FINAL and uses
    F.softmax. This class uses softmax1 for zero-fill-on-demand semantics
    and adds ALiBi bias for recency/latest-write-wins.
    """

    def __init__(self, dim, num_heads=4, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.max_seq_len = max_seq_len

        self.W_q = nn.Parameter(torch.zeros(dim, dim))
        self.W_k = nn.Parameter(torch.zeros(dim, dim))
        self.W_v = nn.Parameter(torch.zeros(dim, dim))
        self.W_o = nn.Parameter(torch.zeros(dim, dim))

        # ALiBi slopes: geometric sequence 2^(-8/n * (i+1)) for each head
        slopes = torch.tensor(
            [2.0 ** (-8.0 / num_heads * (i + 1)) for i in range(num_heads)]
        )
        self.register_buffer("alibi_slopes", slopes)  # [H]

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
        # head_dim stays the same; alibi_slopes shrinks to active heads
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

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # ALiBi bias: -slope * |i - j|, computed on-the-fly
        # Note: With KV cache, query length S and key length S_kv may differ
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
            out = out.transpose(1, 2).contiguous().view(B, S, D)
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
        for a in range(256):
            for b in range(256):
                a_lo, a_hi = a % 16, a // 16
                b_lo, b_hi = b % 16, b // 16

                q = a // b if b > 0 else 0
                q_lo, q_hi = q % 16, q // 16

                # up: 5-way AND on (MARK_AX, a_lo, a_hi, b_lo, b_hi)
                # When all 5 match: up = 5S - 4.5S = 0.5S > 0
                # When any mismatch: up <= 4S - 4.5S = -0.5S < 0, silu ~= 0
                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
                self.b_up.data[unit] = -4.5 * S

                # gate: OP_DIV check (gating by opcode)
                self.W_gate.data[unit, BD.OP_DIV] = 1.0
                self.b_gate.data[unit] = 0.0  # gate = OP_DIV (≈0 when DIV not active)

                # down: Write to OUTPUT_LO[q_lo] and OUTPUT_HI[q_hi]
                self.W_down.data[BD.OUTPUT_LO + q_lo, unit] = 1.0
                self.W_down.data[BD.OUTPUT_HI + q_hi, unit] = 1.0

                unit += 1

        # MOD units: detect (a, b, OP_MOD), output remainder nibbles
        for a in range(256):
            for b in range(256):
                a_lo, a_hi = a % 16, a // 16
                b_lo, b_hi = b % 16, b // 16

                r = a % b if b > 0 else 0
                r_lo, r_hi = r % 16, r // 16

                # up: 5-way AND on (MARK_AX, a_lo, a_hi, b_lo, b_hi)
                # When all 5 match: up = 5S - 4.5S = 0.5S > 0
                # When any mismatch: up <= 4S - 4.5S = -0.5S < 0, silu ~= 0
                self.W_up.data[unit, BD.MARK_AX] = S
                self.W_up.data[unit, BD.ALU_LO + a_lo] = S
                self.W_up.data[unit, BD.ALU_HI + a_hi] = S
                self.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                self.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
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
        n_layers=16,
        n_heads=8,
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
                        d_model, num_heads=n_heads, max_seq_len=max_seq_len
                    ),
                    ffn=PureFFN(d_model, ffn_hidden),
                )
                for _ in range(n_layers)
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
        x = self.embed(token_ids)

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache.get_layer_cache(i) if kv_cache is not None else None
            x = block(x, kv_cache=layer_cache)

        if self.head.weight.is_sparse:
            return sparse_linear(x, self.head.weight, self.head.bias)
        return self.head(x)

    @torch.no_grad()
    def generate_next(self, context, temperature=0.0):
        """Generate next token via greedy or sampled decoding.

        Args:
            context: list of integer token IDs
            temperature: 0.0 = greedy (argmax), >0 = sample from softmax

        Returns:
            int: next token ID
        """
        if len(context) > self.max_seq_len:
            context = context[-self.max_seq_len :]
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

    # --- Tool call detection (gap 322-327) ---
    IO_IS_TOOL_CALL = 322  # Combined flag: any of OPEN/READ/CLOS/PRTF active
    NEXT_TOOL_CALL = 323  # Transition flag → emit TOOL_CALL token

    # --- AX carry-forward staging ---
    AX_CARRY_LO = 328  # 328-343
    AX_CARRY_HI = 344  # 344-359

    # --- ALU result staging ---
    ALU_LO = 360  # 360-375
    ALU_HI = 376  # 376-391

    # --- Carry / comparison ---
    CARRY = 392  # 392-395 (4 dims: inter-byte carry for ADD/SUB/MUL)
    CMP = 396  # 396-399 (4 dims: LT, EQ, GT, ZERO)

    # --- Pristine nibble encoding (hi nibble) ---
    CLEAN_EMBED_HI = 400  # 400-415 (16 dims)

    # --- MUL/DIV staging (also used as FETCH staging in Phase 3) ---
    MUL_ACCUM = 416  # 416-431
    DIV_STAGING = 432  # 432-447
    FETCH_LO = 416  # alias: fetched immediate lo nibble (clean, no prior writes)
    FETCH_HI = 432  # alias: fetched immediate hi nibble

    # --- Address hi nibble gathering (reuse ADDR_KEY space at byte positions) ---
    ADDR_B0_HI = 206  # 206-221 (16 dims): hi nibble of gathered addr byte 0
    ADDR_B1_HI = 222  # 222-237
    ADDR_B2_HI = 238  # 238-253

    # --- L2 threshold head output (7 dims: one per marker type) ---
    L2H0 = 448  # 448-454: threshold 5.5 from nearest IS_MARK

    # --- Memory operation flags ---
    MEM_STORE = 455  # 1 dim: store op active (SI/SC/PSH), relayed to MEM positions
    MEM_ADDR_SRC = 456  # 1 dim: 1=addr from STACK0 (SI/SC), 0=addr from SP (PSH)
    MEM_VAL_B0 = 457  # 1 dim: at MEM val byte 0 position (d=5 from MEM marker)
    MEM_VAL_B1 = 458  # 1 dim: at MEM val byte 1 position (d=6)
    MEM_VAL_B2 = 459  # 1 dim: at MEM val byte 2 position (d=7)
    MEM_VAL_B3 = 460  # 1 dim: at MEM val byte 3 position (d=8)
    OP_LI_RELAY = 461  # 1 dim: LI active (relayed to AX byte positions)
    OP_LC_RELAY = 462  # 1 dim: LC active (relayed to AX byte positions)
    PSH_AT_SP = 463    # 1 dim: PSH opcode flag relayed to SP/STACK0 (clean, no JMP collision)

    # --- Unified memory execution (code from writable memory) ---
    MEM_EXEC = 464     # 1 dim: MEM section contains executable code (for L5 fetch)

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
def set_vm_weights(model, enable_tool_calling=False, alu_mode='lookup'):
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

    for b in range(256):
        embed[b, BD.IS_BYTE] = 1.0
        embed[b, BD.EMBED_LO + (b & 0xF)] = 1.0
        embed[b, BD.EMBED_HI + ((b >> 4) & 0xF)] = 1.0
        # Pristine copies — never written by attention W_o or FFN W_down
        embed[b, BD.CLEAN_EMBED_LO + (b & 0xF)] = 1.0
        embed[b, BD.CLEAN_EMBED_HI + ((b >> 4) & 0xF)] = 1.0

    # ===== LAYER 0: Step structure via threshold attention (8 heads) =====
    # 35-token step: PC(5)+AX(5)+SP(5)+BP(5)+STACK0(5)+MEM(9)+SE(1)
    # STACK0 does NOT have IS_MARK (would block BP from threshold view).
    # Key transitions (by distance from nearest IS_MARK marker):
    #   d=4 from PC/AX/SP/BP → next section  (H0=3.5, H1=4.5)
    #   d=8 from MEM → STEP_END              (H2=7.5, H3=8.5)
    #   d=9 from BP → MEM (through STACK0)   (H3=8.5, H4=9.5)
    attn0 = model.blocks[0].attn
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
    attn2.alibi_slopes.fill_(ALIBI_S)
    _set_threshold_attn(attn2, [5.5], [BD.L2H0], ALIBI_S, HD, heads=[0])
    ffn2 = model.blocks[2].ffn
    _set_layer2_mem_byte_flags(ffn2, S, BD)

    # ===== LAYER 3: Register carry-forward (PC, AX, SP, BP) + PC update =====
    # All carry-forwards in one layer so values are available for later layers.
    attn3 = model.blocks[3].attn
    attn3.alibi_slopes.fill_(0.5)
    PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
    # Head 0: PC carry (prev step PC byte 0 → EMBED at PC marker)
    _set_carry_forward_attn(
        attn3, 0, BD.MARK_PC, PC_I, PC_I, HD, BD.EMBED_LO, BD.EMBED_HI
    )
    # Head 1: AX carry (prev step AX byte 0 → AX_CARRY staging)
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
    ffn3 = model.blocks[3].ffn
    _set_layer3_ffn(ffn3, S, BD)

    # ===== LAYER 4: PC value relay to AX marker =====
    # AX marker reads the current step's PC byte 0 EMBED_LO value.
    # This must be a SEPARATE layer from the fetch because the fetch
    # needs the result of this relay (can't be intra-layer).
    attn4 = model.blocks[4].attn
    attn4.alibi_slopes.fill_(0.5)
    _set_layer4_pc_relay(attn4, S, BD, HD)
    ffn4 = model.blocks[4].ffn
    _set_layer4_ffn(ffn4, S, BD)

    # ===== LAYER 5: Bytecode fetch (imm + opcode) =====
    attn5 = model.blocks[5].attn
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
    _set_layer6_relay_heads(attn6, S, BD, HD)
    # L6 BZ/BNZ relay (AX→PC for conditional branches)
    _set_bz_bnz_relay(attn6, S, BD, HD)

    # L6 opcode relay: broadcast PSH/ADJ/pop flags from AX → SP/STACK0
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
        attn6.alibi_slopes[5] = 5.0  # steep ALiBi for head 5
        _set_tool_call_relay_head(attn6, S, BD, HD)
        # L6 FFN: CMP[2] AND NEXT_SE → NEXT_TOOL_CALL (convert SE to TOOL_CALL)
        _set_tool_call_detection(ffn6, S, BD)

    # ===== LAYER 7: Operand gather + memory relay heads =====
    attn7 = model.blocks[7].attn
    attn7.alibi_slopes.fill_(0.5)
    _set_layer7_operand_gather(attn7, S, BD, HD)
    # L7 head 1: MEM flag broadcast (MEM marker → MEM byte positions)
    attn7.alibi_slopes[1] = 5.0  # steep for nearest MEM marker
    # L7 heads 2-4: Gather prev AX bytes → AX positions (for LI/LC addr)
    # L7 head 5: Relay LI/LC flags → AX byte positions
    attn7.alibi_slopes[5] = 5.0
    # L7 head 6: Relay PSH/store flags → STACK0 byte positions
    attn7.alibi_slopes[6] = 5.0
    _set_layer7_memory_heads(attn7, S, BD, HD)

    # ===== LAYER 8: ALU + SP→STACK0 addr gather =====
    attn8 = model.blocks[8].attn
    attn8.alibi_slopes.fill_(0.5)
    _set_layer8_sp_gather(attn8, S, BD, HD)

    # ===== LAYERS 8-13: ALU Operations =====
    if alu_mode == 'lookup':
        # Use full lookup tables (pure FFN, many parameters)
        ffn8 = model.blocks[8].ffn
        _set_layer8_alu(ffn8, S, BD)
        ffn9 = model.blocks[9].ffn
        _set_layer9_alu(ffn9, S, BD)

        # ===== LAYER 10: carry relay + AX passthrough + bitwise + cmp =====
        attn10 = model.blocks[10].attn
        attn10.alibi_slopes[0] = 5.0  # head 0: steep slope for carry relay
        attn10.alibi_slopes[1] = 1.0  # head 1: AX byte passthrough (nearest step)
        _set_layer10_carry_relay(attn10, S, BD, HD)
        _set_layer10_byte_passthrough(attn10, S, BD, HD)
        ffn10 = model.blocks[10].ffn
        _set_layer10_alu(ffn10, S, BD)

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
        attn13.alibi_slopes.fill_(0.5)
        _set_layer13_mem_addr_gather(attn13, S, BD, HD)
        ffn13 = model.blocks[13].ffn
        _set_layer13_shifts(ffn13, S, BD)

    elif alu_mode == 'efficient':
        # Use efficient multi-layer ALU with pure neural format conversion
        # All operations use baked FFN weights - no Python loops in forward pass

        # L8-L9: Neural ADD/SUB
        model.blocks[8].ffn = EfficientALU_L8_L9_Neural(S, BD)

        # L10: Carry relay + AX passthrough attention (still needed)
        attn10 = model.blocks[10].attn
        attn10.alibi_slopes[0] = 5.0
        attn10.alibi_slopes[1] = 1.0
        _set_layer10_carry_relay(attn10, S, BD, HD)
        _set_layer10_byte_passthrough(attn10, S, BD, HD)

        # L10 FFN: Neural AND/OR/XOR
        model.blocks[10].ffn = EfficientALU_L10_Neural(S, BD)

        # L10.5: Pure neural DIV/MOD (no Python math in forward pass)
        model.blocks[10].post_ops.append(EfficientDivMod_Neural(S, BD))

        # L11-L12: Neural MUL
        model.blocks[11].ffn = EfficientALU_L11_L12_Neural(S, BD)

        # L13: Memory addr gather attention (still needed)
        attn13 = model.blocks[13].attn
        attn13.alibi_slopes.fill_(0.5)
        _set_layer13_mem_addr_gather(attn13, S, BD, HD)

        # L13 FFN: Neural SHL/SHR
        model.blocks[13].ffn = EfficientALU_L13_Neural(S, BD)

    else:
        raise ValueError(f"Unknown alu_mode: {alu_mode}. Use 'lookup' or 'efficient'.")

    # ===== LAYER 14: MEM byte generation (8 heads) =====
    attn14 = model.blocks[14].attn
    attn14.alibi_slopes.fill_(0.1)  # slight recency bias for same-step preference
    _set_layer14_mem_generation(attn14, S, BD, HD)

    # ===== LAYER 15: Memory lookup (softmax1 + ALiBi) =====
    attn15 = model.blocks[15].attn
    attn15.alibi_slopes.fill_(0.01)  # gentle recency bias for latest-write-wins
    _set_layer15_memory_lookup(attn15, S, BD, HD)
    ffn15 = model.blocks[15].ffn
    _set_nibble_copy_ffn(ffn15, S, BD)

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
    ]:
        head.weight[tok, flag] = 20.0
        head.bias[tok] = -10.0

    # Never output these tokens
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

    MEM val byte flags (4 units): Identify positions d=5..8 from MEM marker.
    Uses threshold-difference pattern between L2H0 (5.5) and L1 thresholds.

    Extended BYTE_INDEX for STACK0 bytes 1-3 (3 units): At positions d=7..9
    from BP (where STACK0 bytes live), produce BYTE_INDEX_1/2/3 flags.
    These accumulate with existing BYTE_INDEX (which is 0 at those positions).
    """
    MEM_I = 4  # MEM marker index in MARKS
    BP_I = 3
    NM = BD.NUM_MARKERS
    unit = 0

    # MEM_VAL_B0: d=5 from MEM → L2H0[MEM]=1 (d≤5.5), H1[MEM]=0 (d>4.5)
    # silu(S*(L2H0_MEM + IS_BYTE) - S*1.5) × (1 - H1_MEM)
    ffn.W_up[unit, BD.L2H0 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H1 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B0, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B1: d=6 from MEM → L1H4[MEM]=1 (d≤6.5), L2H0[MEM]=0 (d>5.5)
    # Note: L1H4 dim for MEM = BD.L1H4 + MEM_I
    ffn.W_up[unit, BD.L1H4 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L2H0 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B1, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B2: d=7 from MEM → H2[MEM]=1 (d≤7.5), L1H4[MEM]=0 (d>6.5)
    ffn.W_up[unit, BD.H2 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.L1H4 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B2, unit] = 2.0 / S
    unit += 1

    # MEM_VAL_B3: d=8 from MEM → H3[MEM]=1 (d≤8.5), H2[MEM]=0 (d>7.5)
    ffn.W_up[unit, BD.H3 + MEM_I] = S
    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.H2 + MEM_I] = -1.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.MEM_VAL_B3, unit] = 2.0 / S
    unit += 1

    # Extended BYTE_INDEX for STACK0 byte 1-3 (at d=7,8,9 from BP)
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
    """Conditional nibble copy: OUTPUT = EMBED for all non-PC byte values.

    PC has custom logic in Layer 3 (default + increment), so exclude it.
    This handles AX, SP, BP, STACK0, and MEM byte values.

    For first step outputs, this relies on:
    - AX: Set by IMM instruction (fetched in L5, available in EMBED)
    - SP/BP: Initialized to STACK_INIT, need byte 2 = 0x01
    """
    PC_I = 0  # PC marker index in MARKS array
    unit = 0
    # LO nibbles: copy when IS_BYTE AND NOT at PC/SP/BP positions
    # PC/SP/BP have custom initialization logic in Layer 3, so suppress here
    SP_I = 2  # SP marker index
    BP_I = 3  # BP marker index
    for k in range(16):
        # Up: fires at byte positions, suppressed at PC/SP/BP
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.H1 + PC_I] = -S  # Suppress at PC
        ffn.W_up[unit, BD.H1 + SP_I] = -S  # Suppress at SP
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Suppress at BP
        ffn.b_up[unit] = -S * 0.5
        # Gate: copy this specific nibble value
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        # Output: write to corresponding OUTPUT_LO channel
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    # HI nibbles: same logic
    for k in range(16):
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.H1 + PC_I] = -S  # Suppress at PC
        ffn.W_up[unit, BD.H1 + SP_I] = -S  # Suppress at SP
        ffn.W_up[unit, BD.H1 + BP_I] = -S  # Suppress at BP
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
    T_psh_byte = 2.5  # PSH_AT_SP(~1) + H1[SP](1) + BYTE_INDEX(1) = 3 > 2.5

    # SP byte 0 pos → predict byte 1 = 0xFF
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0  # constant gate
    ffn.W_down[BD.OUTPUT_LO + 15, unit] = 2.0 / S  # lo nibble = 15 (F)
    unit += 1
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 15, unit] = 2.0 / S  # hi nibble = 15 (F)
    unit += 1

    # SP byte 1 pos → predict byte 2 = 0x00
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo nibble = 0
    unit += 1
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
    unit += 1

    # SP byte 2 pos → predict byte 3 = 0x00
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S  # lo nibble = 0
    unit += 1
    ffn.W_up[unit, BD.PSH_AT_SP] = S
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_2] = S
    ffn.b_up[unit] = -S * T_psh_byte
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S  # hi nibble = 0
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
    attn, head_idx, marker_dim, l1h1_idx, l1h0_idx, HD, out_lo, out_hi, slope=0.5
):
    """Set attention head for register carry-forward.

    At marker positions, attends to the previous step's corresponding byte 0
    (identified by L1H1_marker AND NOT L1H0_marker pattern).
    Copies EMBED_LO/HI from the target to out_lo/out_hi.

    Anti-leakage gate (dim 33): at non-target markers, Q[gate] = -L/2
    combined with K[gate] = L gives score penalty -L²/(2√HD) ≈ -14,
    suppressing self-attention leakage (exp(-14) ≈ 8e-7).
    """
    BD = _SetDim
    base = head_idx * HD
    L = 15.0

    attn.W_q[base, marker_dim] = L
    attn.W_k[base, BD.L1H1 + l1h1_idx] = L
    attn.W_k[base, BD.L1H0 + l1h0_idx] = -L

    for k in range(16):
        attn.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

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
    first_pc = PC_OFFSET + INSTR_WIDTH
    pc_lo = first_pc & 0xF
    pc_hi = (first_pc >> 4) & 0xF
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + pc_lo, unit] = 2.0 / S
    unit += 1
    # Unit B: undo when HAS_SE (subsequent steps use carry-forward + increment)
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_LO + pc_lo, unit] = -2.0 / S
    unit += 1

    # Same for OUTPUT_HI
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + pc_hi, unit] = 2.0 / S
    unit += 1
    ffn.W_up[unit, BD.HAS_SE] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.MARK_PC] = 1.0
    ffn.W_down[BD.OUTPUT_HI + pc_hi, unit] = -2.0 / S
    unit += 1

    # SP DEFAULT: STACK_INIT = 0x10000
    # Byte 2 of 0x10000 is 0x01 (lo=1, hi=0)
    # At SP byte 1 (predicting byte 2), write OUTPUT_LO[1]
    # Condition: H1[SP] AND BYTE_INDEX_1
    # Fires on ALL steps. PSH/ADJ in Layer 6 will overwrite when needed.
    SP_I = 2  # SP marker index in MARKS array
    ffn.W_up[unit, BD.H1 + SP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 2.0 / S  # byte value 0x01, lo nibble = 1
    unit += 1

    # BP DEFAULT: same as SP
    BP_I = 3  # BP marker index in MARKS array
    ffn.W_up[unit, BD.H1 + BP_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_1] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 1, unit] = 2.0 / S
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

    # PC INCREMENT: when MARK_PC AND HAS_SE, add INSTR_WIDTH to carried-forward value
    # For each lo nibble k (0-15): new_lo = (k+INSTR_WIDTH)%16
    for k in range(16):
        new_k = (k + INSTR_WIDTH) % 16
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        unit += 1

    # Hi nibble copy (only at MARK_PC AND HAS_SE)
    for k in range(16):
        ffn.W_up[unit, BD.HAS_SE] = S
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.EMBED_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # PC carry correction: when lo nibble >= (16-INSTR_WIDTH), adding INSTR_WIDTH wraps (>=16),
    # so hi nibble must increment by 1. Fires when MARK_PC AND HAS_SE
    # AND any of EMBED_LO[(16-INSTR_WIDTH)..15] is set (old lo nibble >= 16-INSTR_WIDTH).
    # MARK_PC has weight 4*S to strictly require it (prevents false positive
    # at byte positions where EMBED_LO can be inflated by L3 leakage).
    carry_threshold = 16 - INSTR_WIDTH  # For INSTR_WIDTH=8, this is 8
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = 4 * S
        ffn.W_up[unit, BD.HAS_SE] = S
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

    # V: copy byte value nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write immediate to FETCH_LO/HI
    for k in range(16):
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 1.0

    # Head 1: fetch opcode byte (address = PC)
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

    # V: opcode byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: writes to OPCODE_BYTE_LO/HI (staging for opcode decode)
    # Uses separate dims from ALU_LO/HI to avoid residual collision with L7 operand gather
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 2: fetch opcode for first-step (PC marker → address PC_OFFSET)
    # On the first step (NOT HAS_SE), PC = PC_OFFSET. Fetch opcode from address PC_OFFSET.
    # Uses ADDR_KEY matching (same mechanism as head 1, but fires at PC marker not AX).
    # With PC_OFFSET=2: opcode is at ADDR_KEY=2
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

    # Head 3: fetch immediate for first-step (PC marker → address PC_OFFSET+1)
    # Fetches immediate byte 0 at address PC_OFFSET+1 and writes to AX_CARRY.
    # Layer 6 FFN reads AX_CARRY for JMP target.
    # With PC_OFFSET=2: immediate byte 0 is at ADDR_KEY=3
    imm_addr = PC_OFFSET + 1
    base = 3 * HD
    # Q: queries for address PC_OFFSET+1 (e.g., 3: ADDR_KEY_LO[3]=1, ADDR_KEY_HI[0]=1)
    # NOTE: Q[PC_OFFSET] must remain zero - setting it causes opcode/imm to match equally
    attn.W_q[base + (imm_addr & 0xF), BD.CONST] = L  # lo nibble
    attn.W_q[base + 16 + ((imm_addr >> 4) & 0xF), BD.CONST] = L  # hi nibble
    attn.W_q[base + 32, BD.MARK_PC] = L  # third nibble gate

    # K: match ADDR_KEY nibbles
    for k in range(16):
        attn.W_k[base + k, BD.ADDR_KEY + k] = L
        attn.W_k[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
    attn.W_k[base + 32, BD.ADDR_KEY + 32] = L

    # Anti-leakage gate
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # V: copy immediate byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to AX_CARRY_LO/HI at PC marker (used by L6 FFN for JMP)
    # Also write to FETCH_LO/HI at PC marker for first-step immediate
    for k in range(16):
        attn.W_o[BD.AX_CARRY_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 48 + k] = 1.0
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 1.0

    # Head 4: First-step FETCH relay (AX marker ← PC marker)
    # Copy FETCH from PC marker to AX marker on first step only.
    # This allows Layer 6 FFN to read FETCH at AX marker for IMM/LEA routing.
    base = 4 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.HAS_SE] = -L  # Only fire when NOT HAS_SE (first step)
    attn.W_k[base, BD.MARK_PC] = L
    # V: copy FETCH_LO/HI
    for k in range(16):
        attn.W_v[base + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 16 + k, BD.FETCH_HI + k] = 1.0
    # O: write to FETCH_LO/HI at AX marker
    for k in range(16):
        attn.W_o[BD.FETCH_LO + k, base + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 16 + k] = 1.0


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

    Heads 6-7: unused (zero weights, identity via residual).
    """
    L = 50.0

    # Head 0: JMP relay (PC marker → previous step's AX marker)
    base = 0 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker (prevents AX_CARRY corruption)
    attn.W_k[base, BD.MARK_AX] = L
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

    # Head 2: First-step JMP relay (PC marker → current step's AX marker)
    # For first step (NOT HAS_SE), relay OP_JMP and FETCH from current AX marker.
    # Head 0 handles subsequent steps (cross-step relay), head 2 handles first step (intra-step).
    base = 2 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.HAS_SE] = -L  # Only fire when NOT HAS_SE (first step)
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy OP_JMP and FETCH_LO/HI (same as head 0)
    attn.W_v[base + 1, BD.OP_JMP] = 1.0
    for k in range(16):
        attn.W_v[base + 2 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 18 + k, BD.FETCH_HI + k] = 1.0
    # O: write IS_JMP to CMP[0], JMP target to AX_CARRY at PC marker (same as head 0)
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    for k in range(16):
        attn.W_o[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
        attn.W_o[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0

    # Head 3: JSR relay (PC marker → current step's AX marker, ALL steps)
    # For JSR, we need intra-step relay (like head 2) but without HAS_SE restriction.
    # JSR executes at step N and needs PC override at step N (not N+1 like JMP).
    # This relay copies OP_JSR flag from AX to TEMP[0] at PC marker.
    # (Jump target is already in FETCH dims, no need to relay it.)
    base = 3 * HD
    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    # No HAS_SE condition - fire on all steps
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy OP_JSR flag only (FETCH already has target)
    attn.W_v[base + 1, BD.OP_JSR] = 1.0
    # O: write IS_JSR flag to TEMP[0] at PC marker
    # (Layer 5 FFN clears TEMP at PC; Layer 6 attention writes it; Layer 6 FFN reads it)
    attn.W_o[BD.TEMP + 0, base + 1] = 1.0

    # Head 4: NOTE - Cannot add first-step FETCH relay here due to conflict with
    # BZ/BNZ relay (below). BZ/BNZ requires Q[MARK_PC]=L and Q[MARK_AX]=-L,
    # while FETCH relay would need Q[MARK_AX]=L. FETCH will be handled differently.

    # Head 5: First-step OP flag relay (AX marker ← PC marker)
    # For first step, L5 FFN decodes opcodes at PC marker (OP_IMM, OP_LEA, OP_EXIT, OP_NOP, OP_JMP, OP_JSR, arithmetic, bitwise, cmp, shift).
    # L6 FFN needs these flags at AX marker for routing (IMM, EXIT, NOP, JMP, arithmetic, etc).
    # This relay copies OP flags from PC marker to AX marker on first step only.
    # Currently relaying: IMM, LEA, JMP, EXIT, NOP, ADD, SUB, MUL, DIV, MOD, OR, XOR, AND, EQ, LT, SHL, SHR
    base = 5 * HD
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.HAS_SE] = -L  # Only fire when NOT HAS_SE (first step)
    attn.W_k[base, BD.MARK_PC] = L
    # V: copy OP flags (17 total: 5 existing + 12 new)
    attn.W_v[base + 0, BD.OP_IMM] = 1.0
    attn.W_v[base + 1, BD.OP_LEA] = 1.0
    attn.W_v[base + 2, BD.OP_JMP] = 1.0
    attn.W_v[base + 3, BD.OP_EXIT] = 1.0
    attn.W_v[base + 4, BD.OP_NOP] = 1.0
    attn.W_v[base + 5, BD.OP_ADD] = 1.0
    attn.W_v[base + 6, BD.OP_SUB] = 1.0
    attn.W_v[base + 7, BD.OP_MUL] = 1.0
    attn.W_v[base + 8, BD.OP_DIV] = 1.0
    attn.W_v[base + 9, BD.OP_MOD] = 1.0
    attn.W_v[base + 10, BD.OP_OR] = 1.0
    attn.W_v[base + 11, BD.OP_XOR] = 1.0
    attn.W_v[base + 12, BD.OP_AND] = 1.0
    attn.W_v[base + 13, BD.OP_EQ] = 1.0
    attn.W_v[base + 14, BD.OP_LT] = 1.0
    attn.W_v[base + 15, BD.OP_SHL] = 1.0
    attn.W_v[base + 16, BD.OP_SHR] = 1.0
    # O: write OP flags at AX marker
    attn.W_o[BD.OP_IMM, base + 0] = 1.0
    attn.W_o[BD.OP_LEA, base + 1] = 1.0
    attn.W_o[BD.OP_JMP, base + 2] = 1.0
    attn.W_o[BD.OP_EXIT, base + 3] = 1.0
    attn.W_o[BD.OP_NOP, base + 4] = 1.0
    attn.W_o[BD.OP_ADD, base + 5] = 1.0
    attn.W_o[BD.OP_SUB, base + 6] = 1.0
    attn.W_o[BD.OP_MUL, base + 7] = 1.0
    attn.W_o[BD.OP_DIV, base + 8] = 1.0
    attn.W_o[BD.OP_MOD, base + 9] = 1.0
    attn.W_o[BD.OP_OR, base + 10] = 1.0
    attn.W_o[BD.OP_XOR, base + 11] = 1.0
    attn.W_o[BD.OP_AND, base + 12] = 1.0
    attn.W_o[BD.OP_EQ, base + 13] = 1.0
    attn.W_o[BD.OP_LT, base + 14] = 1.0
    attn.W_o[BD.OP_SHL, base + 15] = 1.0
    attn.W_o[BD.OP_SHR, base + 16] = 1.0


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
    T = 4.0  # opcode threshold: correct OP ≈ 5 + MARK_AX 1 = 6 > T

    # === IMM: FETCH → OUTPUT ===
    # Read from FETCH_LO/HI (clean staging dims written by L5 fetch head 0).
    # These dims have no prior-layer leakage, unlike EMBED_LO/HI which
    # accumulates carry-forward residuals from L3.
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_IMM] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.FETCH_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === EXIT: AX_CARRY → OUTPUT ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_EXIT] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_EXIT] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === NOP: AX_CARRY → OUTPUT ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_NOP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_NOP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === JMP: AX_CARRY → OUTPUT (preserves AX through JMP) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_JMP] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
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
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_GETCHAR] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === BZ: AX passthrough (AX unchanged during branch test) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_BZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_BZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === BNZ: AX passthrough (AX unchanged during branch test) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_BNZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_BNZ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === PSH: AX passthrough (AX unchanged during push) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_PSH] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_PSH] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === ADJ: AX passthrough (AX unchanged during stack adjust) ===
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.OP_ADJ] = S
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.b_up[unit] = -S * T
        ffn.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1

    # === BZ PC override: branch if AX == 0 ===
    # CMP[2]=OP_BZ (normalized ≈1), CMP[4]=AX_LO_IS_ZERO, CMP[5]=AX_HI_IS_ZERO
    # 4-way AND in silu: MARK_PC + CMP[2] + CMP[4] + CMP[5] - 3.5
    # All conditions in silu, gate is ONLY the value multiplier.
    # BZ+zero: 1+1+1+1=4 > 3.5 → fires. One missing: 3 < 3.5 → off.
    T_bz = 3.5
    # Cancel existing PC+5 carry
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write branch target - 5 (so L3's +5 yields the actual target).
    # Lo nibble: remap k → (k-5)%16 = (k+11)%16
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        ffn.b_up[unit] = -S * T_bz
        ffn.W_gate[unit, BD.TEMP + k] = 1.0  # FETCH_LO via TEMP
        ffn.W_down[BD.OUTPUT_LO + (k + 11) % 16, unit] = 2.0 / S
        unit += 1
    # Hi nibble, no borrow (target_lo >= 5): copy straight through.
    # Extra condition: any of TEMP[5..15] must be active.
    T_bz_nb = 4.5  # BZ(4) + one_of_TEMP[5..15](1) = 5 > 4.5
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
        for j in range(5, 16):
            ffn.W_up[unit, BD.TEMP + j] = S
        ffn.b_up[unit] = -S * T_bz_nb
        ffn.W_gate[unit, BD.TEMP + 16 + k] = 1.0  # FETCH_HI via TEMP
        ffn.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Hi nibble, borrow (target_lo < 5): decrement hi by 1.
    # Extra condition: any of TEMP[0..4] must be active.
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 2] = S
        ffn.W_up[unit, BD.CMP + 4] = S
        ffn.W_up[unit, BD.CMP + 5] = S
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
    T_bnz_nb = 2.5  # BNZ_A(2) + one_of_TEMP[5..15](1) = 3 > 2.5
    for k in range(16):
        ffn.W_up[unit, BD.MARK_PC] = S
        ffn.W_up[unit, BD.CMP + 3] = S
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
        ffn.W_up[unit, BD.CMP + 3] = S
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
    """L6 attention heads 2-3: Cross-register data relays.

    Head 2: At STACK0 marker, read AX marker's AX_CARRY → ALU staging.
    This provides the AX value at STACK0 position for PSH (STACK0=AX).
    Distance from STACK0 marker to AX marker = 20-5 = 15 tokens back.

    Head 3: At SP marker, read AX marker's FETCH_LO/HI → ALU staging.
    This provides the fetched immediate at SP position for ADJ.
    Distance from SP marker to AX marker = 10-5 = 5 tokens back.
    """
    L = 50.0

    # Head 2: STACK0 ← AX (AX_CARRY → ALU at STACK0 marker, d=15)
    base = 2 * HD
    attn.W_q[base, BD.MARK_STACK0] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy AX_CARRY_LO/HI
    for k in range(16):
        attn.W_v[base + 1 + k, BD.AX_CARRY_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.AX_CARRY_HI + k] = 1.0
    # O: write to ALU_LO/HI at STACK0 marker
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 1.0

    # Head 3: SP ← AX (FETCH → ALU at SP marker, d=5)
    base = 3 * HD
    attn.W_q[base, BD.MARK_SP] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy FETCH_LO/HI
    for k in range(16):
        attn.W_v[base + 1 + k, BD.FETCH_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.FETCH_HI + k] = 1.0
    # O: write to ALU_LO/HI at SP marker
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = 1.0
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = 1.0


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

    # Head 1: LEA — BP OUTPUT → ALU at AX marker
    # Only fires when OP_LEA active (Q includes OP_LEA); when OP_LEA=0,
    # Q=0 → score = -slope*d → softmax1 ≈ 0 (no leakage).
    base = 1 * HD
    attn.W_q[base, BD.OP_LEA] = L  # only fires when LEA active
    attn.W_k[base, BD.MARK_BP] = L
    # V: copy OUTPUT_LO/HI (BP's byte-0 output from L6 identity carry)
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
    # V: copy MEM_STORE and MEM_ADDR_SRC
    attn.W_v[base + 1, BD.MEM_STORE] = 1.0
    attn.W_v[base + 2, BD.MEM_ADDR_SRC] = 1.0
    # O: write to same dims (accumulates at byte positions)
    attn.W_o[BD.MEM_STORE, base + 1] = 1.0
    attn.W_o[BD.MEM_ADDR_SRC, base + 2] = 1.0

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

    # === Head 5: Relay OP_LI/OP_LC from AX marker → AX byte positions ===
    base = 5 * HD
    # Q: fires at AX marker + AX bytes
    attn.W_q[base, BD.MARK_AX] = L
    attn.W_q[base, BD.H1 + AX_I] = L  # AX byte positions
    # K: attend to AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: OP_LI and OP_LC (scaled: ≈5 × 0.2 = ≈1.0)
    attn.W_v[base + 1, BD.OP_LI] = 0.2
    attn.W_v[base + 2, BD.OP_LC] = 0.2
    # O: write to relay dims (×5 to normalize)
    attn.W_o[BD.OP_LI_RELAY, base + 1] = 1.0
    attn.W_o[BD.OP_LC_RELAY, base + 2] = 1.0

    # === Head 6: Relay PSH/ENT/JSR from STACK0 marker → STACK0 byte positions ===
    # Also relay PSH_AT_SP from SP marker → SP byte positions.
    base = 6 * HD
    # Q: fires at STACK0 area (marker + bytes) AND SP area (marker + bytes)
    attn.W_q[base, BD.MARK_STACK0] = L
    attn.W_q[base, BD.L1H4 + BP_I] = L  # d≤6.5 from BP (STACK0 bytes start at d=6)
    attn.W_q[base, BD.IS_BYTE] = L  # only at byte positions (not at SE)
    attn.W_q[base, BD.MARK_SP] = L  # also fire at SP marker
    attn.W_q[base, BD.H1 + SP_I] = L  # also fire at SP byte positions
    # Suppress non-target areas
    attn.W_q[base, BD.H1 + AX_I] = -L
    attn.W_q[base, BD.H3 + MEM_I] = -L
    # K: attend to STACK0 marker (for STACK0 positions) or SP marker (for SP positions)
    attn.W_k[base, BD.MARK_STACK0] = L
    attn.W_k[base, BD.MARK_SP] = L
    # V: copy CMP[0] (PSH), CMP[2] (ENT), CMP[4] (JSR), PSH_AT_SP from markers
    attn.W_v[base + 1, BD.CMP + 0] = 1.0  # PSH flag (legacy)
    attn.W_v[base + 2, BD.CMP + 2] = 1.0  # ENT flag
    attn.W_v[base + 3, BD.CMP + 4] = 1.0  # JSR flag
    attn.W_v[base + 4, BD.PSH_AT_SP] = 1.0  # Clean PSH flag for SP bytes
    # O: accumulate at STACK0/SP byte positions
    attn.W_o[BD.CMP + 0, base + 1] = 1.0
    attn.W_o[BD.CMP + 2, base + 2] = 1.0
    attn.W_o[BD.CMP + 4, base + 3] = 1.0
    attn.W_o[BD.PSH_AT_SP, base + 4] = 1.0


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
            ffn.W_gate[unit, BD.OP_LEA] = 1.0  # LEA reuses ADD circuit
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # === SUB: lo nibble (256 units) ===
    for a in range(16):
        for b in range(16):
            result = (a - b) % 16
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
                ffn.W_gate[unit, BD.OP_LEA] = 1.0  # LEA reuses ADD circuit
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)  # normalize: gate≈5 → CARRY≈1
                unit += 1

    # === SUB borrow detection (120 units: pairs where a < b) ===
    for a in range(16):
        for b in range(16):
            if a < b:
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.W_gate[unit, BD.OP_SUB] = 1.0
                ffn.W_down[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)  # normalize: gate≈5 → CARRY≈1
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

    return unit


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
                    # Block when carry present: -0.01*CARRY prevents spurious activation
                    ffn.W_up[unit, BD.CARRY + 0] = -0.01  # Reduced from -S*2.0
                    ffn.b_up[unit] = -S * 2.5  # 3-way AND
                else:
                    # Require carry: +0.01*CARRY (hint, not hard requirement)
                    ffn.W_up[unit, BD.CARRY + 0] = 0.01  # Reduced from S*2.0
                    ffn.b_up[unit] = -S * 2.9  # Relaxed from -S*4.5 to allow activation with just 3 inputs
                ffn.W_gate[unit, BD.OP_ADD] = 1.0
                ffn.W_gate[unit, BD.OP_LEA] = 1.0  # LEA reuses ADD circuit
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # === SUB hi nibble (no borrow 256 + with borrow 256 = 512 units) ===
    for borrow_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                result = (a - b - borrow_in) % 16
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
                ffn.W_gate[unit, BD.OP_LEA] = 1.0
                ffn.W_down[BD.CARRY + 1, unit] = 2.0 / (S * 5.0)
                unit += 1

    # === SUB hi-nibble borrow-out → CARRY[2] (byte borrow for inter-byte propagation) ===
    # Detects a < b + borrow_in for hi nibble.
    for borrow_in in [0, 1]:
        for a in range(16):
            for b in range(16):
                if borrow_in == 0:
                    if a >= b:
                        continue  # no borrow-out
                else:
                    if a > b:
                        continue  # no borrow-out (a - b - 1 >= 0)
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
    attn.W_q[base + 0, BD.IS_BYTE] = L
    attn.W_q[base + 0, BD.HAS_SE] = L * 2  # Strong HAS_SE requirement
    attn.W_q[base + 0, BD.CONST] = -L * 1.5  # Strong threshold: need IS_BYTE + 2*HAS_SE > 1.5

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
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_LO + a] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.W_gate[unit, op_dim] = 1.0
                ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1
        # Hi nibble (256 units)
        for a in range(16):
            for b in range(16):
                result = op_fn(a, b)
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_HI + a] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.W_gate[unit, op_dim] = 1.0
                ffn.W_down[BD.OUTPUT_HI + result, unit] = 2.0 / S
                unit += 1

    # --- MUL lo nibble (256 units) ---
    # For each (a_lo, b_lo): result_lo = (a_lo * b_lo) % 16
    # 3-way AND: MARK_AX + ALU_LO[a_lo] + AX_CARRY_LO[b_lo], gate=OP_MUL
    for a in range(16):
        for b in range(16):
            result = (a * b) % 16
            ffn.W_up[unit, BD.MARK_AX] = S
            ffn.W_up[unit, BD.ALU_LO + a] = S
            ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
            ffn.b_up[unit] = -S * 2.5
            ffn.W_gate[unit, BD.OP_MUL] = 1.0
            ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
            unit += 1

    # --- SHL/SHR zero output for shift >= 8 (8 units) ---
    # When shift >= 8, result is 0x00. Two sub-cases:
    # Case A (shift >= 16, hi nibble > 0): gate = OP_xxx * (1 - AX_CARRY_HI[0])
    # Case B (shift 8-15, hi=0, lo>=8): up includes AX_CARRY_HI[0] + sum(AX_CARRY_LO[8..15])
    for op_dim in [BD.OP_SHL, BD.OP_SHR]:
        # Case A: shift >= 16 (hi nibble non-zero → AX_CARRY_HI[0] NOT hot)
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.AX_CARRY_HI + 0] = -S  # suppress when hi=0 (shift 0-15)
        ffn.b_up[unit] = -S * 0.5
        ffn.W_gate[unit, op_dim] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # Case B: shift 8-15 (hi=0, lo nibble is 8..15)
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.AX_CARRY_HI + 0] = S
        for lo_bit in range(8, 16):
            ffn.W_up[unit, BD.AX_CARRY_LO + lo_bit] = S
        ffn.b_up[unit] = -S * 2.5  # MARK_AX(1) + HI[0](1) + any_lo_8_15(~1) > 2.5
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

    # SUB borrow → all AX bytes = 0xFF: 2-way AND (CARRY[2] + IS_BYTE)
    for out_dim in [BD.OUTPUT_LO + 15, BD.OUTPUT_HI + 15]:
        ffn.W_up[unit, BD.CARRY + 2] = S
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.b_up[unit] = -S * 1.5
        ffn.W_gate[unit, BD.H1 + AX_IDX] = 1.0  # AX bytes only
        ffn.W_down[out_dim, unit] = 10.0 / S  # ≈ 5.0 output
        unit += 1

    # ADD carry → AX byte 0 only = 0x01: 3-way AND (CARRY[1] + IS_BYTE + BYTE_INDEX_0)
    for out_dim in [BD.OUTPUT_LO + 1, BD.OUTPUT_HI + 0]:
        ffn.W_up[unit, BD.CARRY + 1] = S
        ffn.W_up[unit, BD.IS_BYTE] = S
        ffn.W_up[unit, BD.BYTE_INDEX_0] = S
        ffn.b_up[unit] = -S * 2.5
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
                    ffn.W_up[unit, BD.MARK_AX] = S
                    ffn.W_up[unit, BD.ALU_LO + a_lo] = S
                    ffn.W_up[unit, BD.ALU_HI + a_hi] = S
                    ffn.W_up[unit, BD.AX_CARRY_LO + s] = S
                    ffn.W_up[unit, BD.AX_CARRY_HI + 0] = S
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

        if h == 0:
            # Byte 0: K matches SP marker OR STACK0 byte 0.
            # SP marker has new SP byte 0 in OUTPUT (from L6 PSH computation).
            # STACK0 byte 0 has old value in CLEAN_EMBED (for SI/SC address).
            attn.W_k[base, BD.MARK_SP] = L
            attn.W_k[base, BD.STACK0_BYTE0] = L
        else:
            # Bytes 1-3: K matches BYTE_INDEX_J (standard byte positions).
            byte_idx_dim = [None, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][h]
            attn.W_k[base, byte_idx_dim] = L

        # Dim 1: SP source bonus (active when MEM_ADDR_SRC=0, i.e., PSH).
        # Q = L when PSH, -L when SI/SC. K = L at SP area only (not AX area).
        attn.W_q[base + 1, BD.CONST] = L
        attn.W_q[base + 1, BD.MEM_ADDR_SRC] = -2 * L
        attn.W_k[base + 1, BD.H1 + SP_I] = L
        attn.W_k[base + 1, BD.H1 + AX_I] = -L  # exclude AX (overlaps H1[SP])

        # Dim 2: STACK0 source bonus (active when MEM_ADDR_SRC=1, i.e., SI/SC).
        # For head 0: K matches STACK0_BYTE0 flag (already set in dim 0).
        # For heads 1-3: Use byte-specific threshold differences:
        #   STACK0 byte 1 (d=7 from BP): H2[BP]=1, L1H4[BP]=0
        #   STACK0 byte 2 (d=8 from BP): H3[BP]=1, H2[BP]=0
        #   STACK0 byte 3 (d=9 from BP): H4[BP]=1, H3[BP]=0
        # Also suppress AX/SP/BP areas and MARK_STACK0 marker.
        attn.W_q[base + 2, BD.MEM_ADDR_SRC] = L
        if h == 0:
            # Head 0: L1H4[BP] fires at d≤6.5 (covers byte 0 at d=6)
            attn.W_k[base + 2, BD.L1H4 + BP_I] = L
            attn.W_k[base + 2, BD.H1 + BP_I] = -L  # exclude d≤4.5
        elif h == 1:
            # Head 1: H2[BP] - L1H4[BP] fires at d∈(6.5, 7.5] (byte 1 at d=7)
            attn.W_k[base + 2, BD.H2 + BP_I] = L
            attn.W_k[base + 2, BD.L1H4 + BP_I] = -L
        elif h == 2:
            # Head 2: H3[BP] - H2[BP] fires at d∈(7.5, 8.5] (byte 2 at d=8)
            attn.W_k[base + 2, BD.H3 + BP_I] = L
            attn.W_k[base + 2, BD.H2 + BP_I] = -L
        elif h == 3:
            # Head 3: H4[BP] - H3[BP] fires at d∈(8.5, 9.5] (byte 3 at d=9)
            attn.W_k[base + 2, BD.H4 + BP_I] = L
            attn.W_k[base + 2, BD.H3 + BP_I] = -L
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
        attn.W_k[base + 33, BD.CONST] = 5.0

        # Dim 34: MEM_STORE gate (suppress non-store positions).
        attn.W_q[base + 34, BD.CONST] = -250.0
        attn.W_q[base + 34, BD.MEM_STORE] = 250.0
        attn.W_k[base + 34, BD.CONST] = 5.0

        # V: read CLEAN_EMBED + OUTPUT (sum). At SP marker: CLEAN_EMBED=0
        # (marker token), OUTPUT=new SP value → V picks up new SP. At byte
        # positions: CLEAN_EMBED=old value, OUTPUT=0 → V picks up old value.
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            attn.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0

        # O: write to OUTPUT_LO/HI
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0

    # === Heads 4-7: MEM val byte generation ===
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

        # Dim 0: Q position selection + K targets AX byte positions.
        attn.W_q[base, pos_up] = L
        attn.W_q[base, pos_down] = -L
        attn.W_k[base, byte_idx_dim] = L
        attn.W_k[base, BD.H1 + AX_I] = L  # AX area bonus (+28 points)

        # Dim 33: Position gate (suppress non-target MEM byte positions).
        # Same 500 strength as addr heads to dominate MEM_STORE gate.
        attn.W_q[base + 33, BD.CONST] = -500.0
        attn.W_q[base + 33, pos_up] = 500.0
        attn.W_q[base + 33, pos_down] = -500.0
        attn.W_k[base + 33, BD.CONST] = 5.0

        # Dim 34: MEM_STORE gate (suppress non-store positions).
        attn.W_q[base + 34, BD.CONST] = -250.0
        attn.W_q[base + 34, BD.MEM_STORE] = 250.0
        attn.W_k[base + 34, BD.CONST] = 5.0

        # V: copy CLEAN_EMBED nibbles (AX bytes have correct value here)
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0

        # O: write to OUTPUT_LO/HI
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0


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
            # Head 0: also activated by LC and STACK0 marker
            attn.W_q[base, BD.OP_LC_RELAY] = 2000.0
            attn.W_q[base, BD.MARK_STACK0] = 2000.0
        else:
            # Heads 1-3: activated at STACK0 byte positions via BP threshold
            # L1H4[BP] fires at d<=9.5 from BP (covers STACK0 area d=5-9)
            # H1[BP] fires at d<=4.5 from BP (BP's own bytes) — subtract to cancel
            attn.W_q[base, BD.L1H4 + BP_I] = 2000.0
            attn.W_q[base, BD.H1 + BP_I] = -2000.0
        # PSH suppression at STACK0 positions
        attn.W_q[base, BD.CMP + 0] = -2000.0
        attn.W_k[base, BD.CONST] = 10.0

        # === Dim 1: Store anchor — suppress non-store K at target Q ===
        # Q[1] = 50 at target, 0 at non-target.
        # K[1] = MEM_STORE*100 - CONST*50 = +50 (store) or -50 (non-store).
        # Score: 50*(+50)/8 = +312.5 (target+store),
        #        50*(-50)/8 = -312.5 (target+non-store).
        attn.W_q[base + 1, BD.OP_LI_RELAY] = 50.0
        if h == 0:
            attn.W_q[base + 1, BD.OP_LC_RELAY] = 50.0
            attn.W_q[base + 1, BD.MARK_STACK0] = 50.0
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
        # Q[3] = L*byte_flag. K[3] = L*threshold_flag_for_val_byte_h.
        byte_q_flag = [BD.MARK_AX, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
        attn.W_q[base + 3, byte_q_flag] = L
        if h == 0:
            attn.W_q[base + 3, BD.MARK_STACK0] = L
        MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
        if h == 0:
            # Head 0 → val byte 0 at d=5: L2H0[MEM]=1 (d≤5.5), H1[MEM]=0 (d>4.5).
            attn.W_k[base + 3, BD.L2H0 + MEM_I] = L
            attn.W_k[base + 3, BD.H1 + MEM_I] = -L
        else:
            # Heads 1-3 → val bytes 1,2,3 at d=6,7,8 via MEM_VAL_B1/B2/B3.
            attn.W_k[base + 3, MEM_VAL_DIMS[h]] = L

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
    attn.W_k[base, BD.MARK_AX] = L

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

    Starts at unit 750 to avoid conflict with _set_layer6_routing_ffn (units 0-740).
    """
    unit = 750

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
    base = 6 * HD  # head 6

    # Q: fires at SP, STACK0, BP, and MEM markers, blocked at AX
    attn.W_q[base, BD.MARK_SP] = L
    attn.W_q[base, BD.MARK_STACK0] = L
    attn.W_q[base, BD.MARK_BP] = L  # ENT needs flags at BP marker
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

    # V[6]: MEM_STORE flag — any of SI/SC/PSH (scaled: ≈5 × 0.04 × 3 ≈ 0.6)
    # Actually use 0.2 for each since only one is active: ≈5 × 0.2 = ≈1.0
    attn.W_v[base + 6, BD.OP_SI] = 0.2
    attn.W_v[base + 6, BD.OP_SC] = 0.2
    attn.W_v[base + 6, BD.OP_PSH] = 0.2

    # V[7]: MEM_ADDR_SRC — SI/SC only (addr from STACK0, not SP)
    attn.W_v[base + 7, BD.OP_SI] = 0.2
    attn.W_v[base + 7, BD.OP_SC] = 0.2

    # O: write to CMP dims + MEM store flags + PSH_AT_SP
    attn.W_o[BD.CMP + 0, base + 1] = 1.0  # OP_PSH → CMP[0] (legacy, kept for compatibility)
    attn.W_o[BD.PSH_AT_SP, base + 1] = 1.0  # OP_PSH → PSH_AT_SP (clean, no JMP collision)
    attn.W_o[BD.CMP + 1, base + 2] = 1.0  # OP_ADJ → CMP[1]
    attn.W_o[BD.CMP + 3, base + 3] = 5.0  # POP group → CMP[3] (×5 to rescale 0.2→1.0)
    attn.W_o[BD.CMP + 2, base + 4] = 1.0  # OP_ENT → CMP[2] at SP/STACK0/BP
    attn.W_o[BD.CMP + 4, base + 5] = 1.0  # OP_JSR → CMP[4] at SP/STACK0/BP
    attn.W_o[BD.MEM_STORE, base + 6] = 1.0  # store flag → MEM marker
    attn.W_o[BD.MEM_ADDR_SRC, base + 7] = 1.0  # addr source flag → MEM marker


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

    Uses 32 FFN units starting at offset 790 (after PUTCHAR routing at 750-782).
    """
    unit = 790
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

    L5 heads 2-3: ENT relay attention (BP EMBED → TEMP, SP EMBED → TEMP).
    L6 head 7: JSR PC OUTPUT → AX_CARRY at STACK0.
    L6 FFN units 850-1105: LEA/JSR/ENT output routing.
    """
    attn5 = model.blocks[5].attn
    attn6 = model.blocks[6].attn
    ffn6 = model.blocks[6].ffn

    T = 4.0  # standard opcode threshold: OP(~5) + MARK_AX(1) = 6 > 4

    # =====================================================================
    # L5 heads 2-3: ENT relay attention
    # =====================================================================
    L5 = 20.0  # matching L5 fetch heads

    # Head 2: BP EMBED → TEMP at STACK0 marker (for ENT: STACK0 = old_BP)
    # Distance d=5 (STACK0 at pos 20, BP at pos 15 in same step)
    base = 2 * HD
    attn5.W_q[base, BD.MARK_STACK0] = L5
    attn5.W_k[base, BD.MARK_BP] = L5
    # V: copy EMBED_LO/HI
    for k in range(16):
        attn5.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
        attn5.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0
    # O: write to TEMP[0..15] and TEMP[16..31]
    for k in range(16):
        attn5.W_o[BD.TEMP + k, base + 1 + k] = 1.0
        attn5.W_o[BD.TEMP + 16 + k, base + 17 + k] = 1.0

    # Head 3: SP EMBED → TEMP at BP marker (for ENT: BP = old_SP - 8)
    # Distance d=5 (BP at pos 15, SP at pos 10)
    base = 3 * HD
    attn5.W_q[base, BD.MARK_BP] = L5
    attn5.W_k[base, BD.MARK_SP] = L5
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
    L6 = 50.0
    base = 7 * HD
    attn6.W_q[base, BD.MARK_STACK0] = L6
    attn6.W_q[base, BD.MARK_AX] = -L6  # block at AX marker
    attn6.W_k[base, BD.MARK_PC] = L6
    # V: copy OUTPUT_LO/HI (PC's output = PC+5 from L3)
    for k in range(16):
        attn6.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
        attn6.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
    # O: write to AX_CARRY_LO/HI at STACK0 marker
    for k in range(16):
        attn6.W_o[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
        attn6.W_o[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0

    # =====================================================================
    # L6 FFN: Function call output routing (units 850-1105)
    # =====================================================================
    unit = 850

    # --- LEA AX_CARRY override (32 units: 850-881) ---
    # At AX marker when OP_LEA: write FETCH to AX_CARRY (overwriting contamination).
    # The ADD circuitry in L8/L9 then computes AX_CARRY + ALU = FETCH + BP.
    # OP_LEA ≈ 2.4, MARK_AX ≈ 1.0, so sum = 3.4. Use T_lea = 2.5.
    T_lea = 2.5
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEA] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T_lea
        ffn6.W_gate[unit, BD.FETCH_LO + k] = 1.0
        ffn6.W_down[BD.AX_CARRY_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEA] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T_lea
        ffn6.W_gate[unit, BD.FETCH_HI + k] = 1.0
        ffn6.W_down[BD.AX_CARRY_HI + k, unit] = 2.0 / S
        unit += 1

    # --- JSR SP -= 8 (32 units: 882-913) ---
    # Same pattern as PSH SP-=8 but gated on CMP[4] (JSR relay).
    # T=1.5: CMP[4](~1) + MARK_SP(1) = 2 > 1.5.
    T_jsr = 1.5
    for k in range(16):
        new_k = (k - 8) % 16
        ffn6.W_up[unit, BD.CMP + 4] = S  # relayed OP_JSR
        ffn6.W_up[unit, BD.MARK_SP] = S
        ffn6.b_up[unit] = -S * T_jsr
        ffn6.W_gate[unit, BD.EMBED_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
        ffn6.W_down[BD.OUTPUT_LO + k, unit] += -2.0 / S  # cancel identity
        unit += 1
    for k in range(16):
        new_k_borrow = (k - 1) % 16
        ffn6.W_up[unit, BD.CMP + 4] = S
        ffn6.W_up[unit, BD.MARK_SP] = S
        ffn6.b_up[unit] = -S * T_jsr
        ffn6.W_gate[unit, BD.EMBED_HI + k] = 1.0
        for lo_bit in range(8, 16):
            ffn6.W_gate[unit, BD.EMBED_LO + lo_bit] = -1.0
        ffn6.W_down[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
        ffn6.W_down[BD.OUTPUT_HI + k, unit] += -2.0 / S
        unit += 1

    # --- JSR STACK0 = PC+5 (return addr) (32 units: 914-945) ---
    # At STACK0 marker when JSR: cancel identity (EMBED), write AX_CARRY
    # (which has PC OUTPUT = PC+5 from L6 head 7).
    T_jsr_s0 = 1.5  # CMP[4](~1) + MARK_STACK0(1) = 2 > 1.5
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

    # --- JSR PC override: PC = FETCH (jump target) (64 units: 978-1041) ---
    # At PC marker when JSR: cancel OUTPUT (PC+5), write FETCH (jump target).
    # Gated on TEMP[0] (IS_JSR flag relayed from AX by L6 head 3).
    # Threshold: relayed OP_JSR ≈ 5.0, so T=4.0 separates it from false positives.
    T_jsr_pc = 4.0
    # Cancel OUTPUT_LO/HI (PC+5)
    for k in range(16):
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from L6 head 3 relay
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.OUTPUT_LO + k] = -1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from L6 head 3 relay
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.OUTPUT_HI + k] = -1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
    # Write FETCH_LO/HI (jump target from immediate field)
    # FIXED: Was reading AX_CARRY (PC+5 return address), now correctly reads FETCH (jump target)
    for k in range(16):
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from first-step decode or L6 head 3 relay
        ffn6.b_up[unit] = -S * T_jsr_pc
        ffn6.W_gate[unit, BD.FETCH_LO + k] = 1.0  # FIXED: was AX_CARRY_LO
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.MARK_PC] = S
        ffn6.W_up[unit, BD.TEMP + 0] = S  # IS_JSR flag from first-step decode or L6 head 3 relay
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
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEV] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
        ffn6.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
        unit += 1
    for k in range(16):
        ffn6.W_up[unit, BD.OP_LEV] = S
        ffn6.W_up[unit, BD.MARK_AX] = S
        ffn6.b_up[unit] = -S * T
        ffn6.W_gate[unit, BD.AX_CARRY_HI + k] = 1.0
        ffn6.W_down[BD.OUTPUT_HI + k, unit] = 2.0 / S
        unit += 1
