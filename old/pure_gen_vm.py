#!/usr/bin/env python3
"""
Pure Generative VM - step() = N calls to generate_next_token(). Nothing else.

Architecture:
- Transformer with MoE FFN (1 expert per opcode)
- Baked weights for VM operations
- Speculative decoding for faster generation
- Sparse tensors for memory efficiency

generate_next_token() = model.forward(context).argmax()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math


def softmax1(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax with +1 in denominator ("softmax1" or "nowhere to attend" attention).

    softmax1(x) = exp(x) / (1 + sum(exp(x)))

    This allows the attention to "attend to nothing" when no input is a good match.
    Useful for memory access where the address might not exist.
    """
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)  # Numerical stability
    return exp_x / (1.0 + exp_x.sum(dim=dim, keepdim=True))


# =============================================================================
# VOCABULARY
# =============================================================================

class Vocab:
    PAD, BOS, EOS = 0, 1, 2
    CODE, REG_PC, REG_AX, REG_SP, REG_BP, MEM = 3, 4, 5, 6, 7, 8
    STEP_END = 9
    HEAP = 10  # Heap pointer marker
    # Tokens 11-15 reserved for future markers
    BYTE_BASE = 16
    VOCAB_SIZE = 272  # 16 markers + 256 bytes

    # Position markers for step position tracking
    POS_MARKERS = {
        0: REG_PC, 5: REG_AX, 10: REG_SP, 15: REG_BP, 20: MEM, 29: STEP_END
    }

    @staticmethod
    def byte_tok(val: int) -> int:
        return Vocab.BYTE_BASE + (val & 0xFF)

    @staticmethod
    def tok_byte(tok: int) -> int:
        return tok - Vocab.BYTE_BASE


# =============================================================================
# OPCODES - One expert per opcode
# =============================================================================

class Opcode:
    """VM opcodes - each gets its own MoE expert."""
    LEA = 0    # Load effective address
    IMM = 1    # Load immediate
    JMP = 2    # Jump
    JSR = 3    # Jump to subroutine
    BZ = 4     # Branch if zero
    BNZ = 5    # Branch if not zero
    ENT = 6    # Enter function
    ADJ = 7    # Adjust stack
    LEV = 8    # Leave function
    LI = 9     # Load int
    LC = 10    # Load char
    SI = 11    # Store int
    SC = 12    # Store char
    PSH = 13   # Push
    OR = 14    # Bitwise OR
    XOR = 15   # Bitwise XOR
    AND = 16   # Bitwise AND
    EQ = 17    # Equal
    NE = 18    # Not equal
    LT = 19    # Less than
    GT = 20    # Greater than
    LE = 21    # Less or equal
    GE = 22    # Greater or equal
    SHL = 23   # Shift left
    SHR = 24   # Shift right
    ADD = 25   # Add
    SUB = 26   # Subtract
    MUL = 27   # Multiply
    DIV = 28   # Divide
    MOD = 29   # Modulo
    # System calls
    OPEN = 30  # Open file
    READ = 31  # Read from file/stdin
    CLOS = 32  # Close file
    PRTF = 33  # Printf
    MALC = 34  # Malloc
    FREE = 35  # Free memory
    MSET = 36  # Memset
    MCPY = 37  # Memcpy
    EXIT = 38  # Exit program
    # Additional
    MCMP = 39  # Memcmp
    GETC = 40  # Getchar
    PUTC = 41  # Putchar

    # All opcodes that need experts
    ALL = [LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV,
           LI, LC, SI, SC, PSH,
           OR, XOR, AND, EQ, NE, LT, GT, LE, GE,
           SHL, SHR, ADD, SUB, MUL, DIV, MOD,
           OPEN, READ, CLOS, PRTF, MALC, FREE, MSET, MCPY,
           EXIT, MCMP, GETC, PUTC]

    NUM_EXPERTS = len(ALL)  # 42 experts

    @staticmethod
    def to_expert_idx(op: int) -> int:
        """Map opcode to expert index."""
        try:
            return Opcode.ALL.index(op)
        except ValueError:
            return 0  # Default to LEA expert


# =============================================================================
# SPARSE TENSOR UTILITIES
# =============================================================================

def to_sparse(dense: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """Convert dense tensor to sparse COO format."""
    return dense.to_sparse()


# =============================================================================
# MOE EXPERT - One per opcode
# =============================================================================

class OpcodeExpert(nn.Module):
    """
    Single expert specialized for one opcode.

    Uses SwiGLU activation with baked weights for the specific operation.
    """

    def __init__(self, dim: int, ffn_dim: int, opcode: int):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.opcode = opcode

        # SwiGLU: out = down(silu(up(x)) * gate(x))
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)) * self.gate(x))


class OpcodeMoE(nn.Module):
    """
    Mixture of Experts with one expert per opcode.

    Router selects expert based on opcode encoding in hidden state.
    Each expert is specialized for its opcode's computation.
    """

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.dim = dim
        self.num_experts = Opcode.NUM_EXPERTS

        # Router: selects expert based on opcode in hidden state
        self.router = nn.Linear(dim, self.num_experts, bias=False)

        # One expert per opcode
        self.experts = nn.ModuleList([
            OpcodeExpert(dim, ffn_dim, op) for op in Opcode.ALL
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MoE forward with top-1 routing.

        Routes to correct expert based on opcode in hidden state.
        Each expert computes its opcode's operation via baked nibble tables.
        """
        B, L, D = x.shape

        # Router logits
        router_logits = self.router(x)  # [B, L, num_experts]

        # Top-1 selection per position
        expert_weights = F.softmax(router_logits, dim=-1)
        expert_idx = router_logits.argmax(dim=-1)  # [B, L]

        # Compute output via selected expert (for last position only for efficiency)
        # This is where the actual computation happens via baked weights
        output = torch.zeros_like(x)
        last_idx = expert_idx[0, -1].item()

        # Route last token through selected expert
        output[:, :-1] = self.experts[0](x[:, :-1])  # Identity for context
        output[:, -1:] = self.experts[last_idx](x[:, -1:])  # Compute for last

        return output


# =============================================================================
# TRANSFORMER LAYER
# =============================================================================

class TransformerLayer(nn.Module):
    """Transformer layer: Attention + Opcode MoE FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, use_moe: bool = True,
                 use_alibi: bool = False, alibi_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_moe = use_moe
        self.use_alibi = use_alibi
        self.alibi_scale = alibi_scale

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # FFN: Opcode MoE or standard
        if use_moe:
            self.ffn = OpcodeMoE(dim, ffn_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim, bias=False),
                nn.SiLU(),
                nn.Linear(ffn_dim, dim, bias=False)
            )

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate ALiBi positional bias.

        ALiBi biases attention toward recent positions:
        bias[i,j] = -scale * |i - j|

        This means for memory operations, among positions with equal
        content match, the most recent one wins.
        """
        positions = torch.arange(seq_len, device=device)
        # [seq_len, seq_len] matrix of relative distances
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        # Negative bias proportional to distance (recent = less penalty)
        alibi = -self.alibi_scale * torch.abs(rel_pos).float()
        # Expand for heads: [1, 1, seq_len, seq_len]
        return alibi.unsqueeze(0).unsqueeze(0)

    # =========================================================================
    # TRANSFORMER LAYER FORWARD - STANDARD ATTENTION + FFN
    # =========================================================================
    # This implements standard transformer layer computation:
    #   x = x + Attention(LayerNorm(x))
    #   x = x + FFN(LayerNorm(x))
    #
    # Features:
    # - softmax1 attention (exp(x)/(1+sum(exp))) for "attend to nothing"
    # - ALiBi positional bias for memory recency preference
    # - KV caching for autoregressive generation
    # - MoE FFN with opcode-specialized experts
    #
    # DO NOT add Python control flow based on hidden state values.
    # =========================================================================

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Standard transformer layer forward - DO NOT add Python conditionals."""
        B, L, D = x.shape

        # === ATTENTION ===
        h = self.ln1(x)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # KV caching for autoregressive generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (adjusted for KV cache length)
        cache_len = k.size(2) - L
        if cache_len > 0:
            full_mask = torch.zeros(L, k.size(2), device=x.device)
            full_mask[:, :cache_len] = 0  # Can attend to all cached positions
            full_mask[:, cache_len:] = mask  # Causal for new positions
            scores = scores + full_mask.unsqueeze(0).unsqueeze(0)
        else:
            scores = scores + mask

        # ALiBi: prefer recent positions (for memory overwrite semantics)
        if self.use_alibi:
            alibi = self._get_alibi_bias(k.size(2), x.device)
            scores = scores + alibi[:, :, -L:, :]

        # softmax1: allows "attend to nothing" when no match
        attn = softmax1(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.o_proj(out)

        # === FFN (MoE with opcode-specialized experts) ===
        h = self.ln2(x)
        x = x + self.ffn(h)

        return x, new_kv_cache

    def forward_no_cache(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward without KV caching (for compatibility)."""
        out, _ = self.forward(x, mask, None)
        return out


# =============================================================================
# EMBEDDING LAYOUT CONSTANTS
# =============================================================================

class EmbedDims:
    """Embedding dimension layout for the transformer hidden state."""
    # Marker one-hots (tokens 0-15)
    MARKERS_START = 0
    MARKERS_END = 16

    # Byte value one-hots (tokens 16-271)
    BYTES_START = 16
    BYTES_END = 272

    # Nibble inputs for MoE computation (current byte)
    NIBBLE_A_START = 272   # Operand A low nibble (16 dims)
    NIBBLE_A_END = 288
    NIBBLE_B_START = 288   # Operand B low nibble (16 dims)
    NIBBLE_B_END = 304

    # Step position encoding
    STEP_POS_START = 304
    STEP_POS_END = 320

    # Opcode one-hot for MoE routing
    OPCODE_START = 320
    OPCODE_END = 400

    # Operand A byte one-hot (for injection)
    OPERAND_A_START = 420
    OPERAND_A_END = 676

    # Operand B byte one-hot (for injection)
    OPERAND_B_START = 500
    OPERAND_B_END = 756

    # Attention-gathered operand nibbles (alternative path)
    GATHERED_A_START = 448
    GATHERED_B_START = 464

    # === EXTENDED DIMS FOR FULL NEURAL 32-BIT ===
    # Full 32-bit operand nibbles (8 nibbles each = 4 bytes × 2 nibbles)
    # Operand A: 8 nibble slots × 16 one-hot dims = 128 dims
    OP_A_NIBBLES_START = 480  # [byte0_lo, byte0_hi, byte1_lo, byte1_hi, ...]
    OP_A_NIBBLE_0_LO = 480    # Byte 0 low nibble (16 dims)
    OP_A_NIBBLE_0_HI = 496    # Byte 0 high nibble
    OP_A_NIBBLE_1_LO = 512    # Byte 1 low nibble
    OP_A_NIBBLE_1_HI = 528
    OP_A_NIBBLE_2_LO = 544
    OP_A_NIBBLE_2_HI = 560
    OP_A_NIBBLE_3_LO = 576
    OP_A_NIBBLE_3_HI = 592
    OP_A_NIBBLES_END = 608

    # Operand B: 8 nibble slots
    OP_B_NIBBLES_START = 608
    OP_B_NIBBLE_0_LO = 608
    OP_B_NIBBLE_0_HI = 624
    OP_B_NIBBLE_1_LO = 640
    OP_B_NIBBLE_1_HI = 656
    OP_B_NIBBLE_2_LO = 672
    OP_B_NIBBLE_2_HI = 688
    OP_B_NIBBLE_3_LO = 704
    OP_B_NIBBLE_3_HI = 720
    OP_B_NIBBLES_END = 736

    # Result nibbles (8 slots for 4 bytes)
    RESULT_NIBBLES_START = 736
    RESULT_0_LO = 736
    RESULT_0_HI = 752
    RESULT_1_LO = 768
    RESULT_1_HI = 784
    RESULT_2_LO = 800
    RESULT_2_HI = 816
    RESULT_3_LO = 832
    RESULT_3_HI = 848
    RESULT_NIBBLES_END = 864

    # Carry flags (4 carries between bytes + 1 final)
    CARRY_0 = 864  # Carry from byte 0 to byte 1
    CARRY_1 = 865  # Carry from byte 1 to byte 2
    CARRY_2 = 866  # Carry from byte 2 to byte 3
    CARRY_3 = 867  # Final carry out
    CARRY_IN = 868  # Input carry for current operation

    # Sum outputs (for detecting carry: sum > 15 means carry)
    SUM_NIBBLES_START = 870  # 8 sum outputs × 32 possible values
    SUM_0_LO = 870   # Sum for byte 0 low (0-30 for ADD)
    SUM_0_HI = 902
    SUM_1_LO = 934
    SUM_1_HI = 966
    SUM_2_LO = 998
    SUM_2_HI = 1030
    SUM_3_LO = 1062
    SUM_3_HI = 1094
    SUM_NIBBLES_END = 1126

    # === 100% NEURAL: Carry detection via MoE output ===
    # MoE outputs 1.0 to CARRY_OUT_x if sum > 15, else 0.0
    CARRY_OUT_0_LO = 1126  # Carry from byte 0 low nibble
    CARRY_OUT_0_HI = 1127  # Carry from byte 0 high nibble (to byte 1)
    CARRY_OUT_1_LO = 1128
    CARRY_OUT_1_HI = 1129
    CARRY_OUT_2_LO = 1130
    CARRY_OUT_2_HI = 1131
    CARRY_OUT_3_LO = 1132
    CARRY_OUT_3_HI = 1133  # Final overflow

    # === 100% NEURAL: Comparison flags ===
    # For each nibble: EQ flag (1 if A==B), LT flag (1 if A<B)
    CMP_EQ_0_LO = 1134
    CMP_EQ_0_HI = 1135
    CMP_EQ_1_LO = 1136
    CMP_EQ_1_HI = 1137
    CMP_EQ_2_LO = 1138
    CMP_EQ_2_HI = 1139
    CMP_EQ_3_LO = 1140
    CMP_EQ_3_HI = 1141
    CMP_LT_0_LO = 1142
    CMP_LT_0_HI = 1143
    CMP_LT_1_LO = 1144
    CMP_LT_1_HI = 1145
    CMP_LT_2_LO = 1146
    CMP_LT_2_HI = 1147
    CMP_LT_3_LO = 1148
    CMP_LT_3_HI = 1149

    # === CARRY LOOKAHEAD FLAGS (for standard forward pass) ===
    # Generate (G): sum > 15, this nibble generates a carry
    # Propagate (P): sum == 15, this nibble propagates incoming carry
    CARRY_GEN_START = 1150  # 8 generate flags
    CARRY_GEN = [1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157]
    CARRY_PROP_START = 1158  # 8 propagate flags
    CARRY_PROP = [1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165]

    # Carry chain results (computed from G and P)
    CARRY_CHAIN = [1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173]

    # Results WITH carry/borrow applied (for selection by carry chain)
    # Each nibble has 16 dims for result when carry arrives
    RESULT_CARRY_0_LO = 1174  # Result if carry arrives at nibble 0 low
    RESULT_CARRY_0_HI = 1190
    RESULT_CARRY_1_LO = 1206
    RESULT_CARRY_1_HI = 1222
    RESULT_CARRY_2_LO = 1238
    RESULT_CARRY_2_HI = 1254
    RESULT_CARRY_3_LO = 1270
    RESULT_CARRY_3_HI = 1286
    RESULT_CARRY_END = 1302

    # Carry chain computed flags (set by layer 2 based on G/P)
    # CARRY_IN[i] = 1 if carry arrives at nibble i
    CARRY_IN_0_LO = 1302  # Always 0 (no carry into first nibble)
    CARRY_IN_0_HI = 1303
    CARRY_IN_1_LO = 1304
    CARRY_IN_1_HI = 1305
    CARRY_IN_2_LO = 1306
    CARRY_IN_2_HI = 1307
    CARRY_IN_3_LO = 1308
    CARRY_IN_3_HI = 1309
    CARRY_IN = [1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309]

    # Final output (one-hot for result byte, set by layer 3)
    OUTPUT_BYTE = 1310  # 256 dims for output byte one-hot
    OUTPUT_BYTE_END = 1566

    # Comparison final result (single dim)
    CMP_RESULT = 1566  # 1 if comparison true, 0 if false

    # Sign flags for signed comparisons (set by layer 2)
    CMP_A_NEG = 1567  # 1 if A is negative (A[7] >= 8)
    CMP_B_NEG = 1568  # 1 if B is negative (B[7] >= 8)

    # Memory operation dims
    MEM_ADDR_START = 1302  # 32 bits for address
    MEM_VALUE_START = 1334  # 32 bits for value
    MEM_OP_READ = 1366
    MEM_OP_WRITE = 1367
    MEM_OP_ALLOC = 1368
    MEM_OP_FREE = 1369
    MEM_SIZE_START = 1370  # For memset/memcpy size

    # === KV CACHE MEMORY ===
    # Memory operations use attention for content-addressable lookup.
    # Address nibbles reuse OP_A_NIBBLE_* dims (480-607)
    # Value nibbles use OUTPUT_BYTE dims (1310-1438)
    # MEM marker flag indicates a memory entry position
    MEM_MARKER_FLAG = 1569  # Set when position is a MEM token

    # Heap pointer stored as single value (updated by MALC)
    HEAP_PTR = 1570  # Current heap pointer value (scaled)

    # === 32-BIT MUL/DIV WORKSPACE ===
    # For 32-bit multiplication via nibble long multiplication:
    # A × B where A = [a0..a7] and B = [b0..b7] (8 nibbles each)
    # We compute 64 partial products p[i][j] = a[i] × b[j]
    # Each partial product is 0-225 (8 bits = 2 nibbles)
    #
    # Storage: 64 products × 2 nibbles × 16 one-hot = 2048 dims
    # But we can use scalar storage: 64 × 8 bits = 512 dims
    MUL_PARTIAL_START = 1600  # 64 partial products as scalars (scaled 0-225)

    # Column accumulators for MUL (16 nibbles, each can sum to ~1800)
    # Using scaled values to avoid overflow
    MUL_ACC_START = 1664  # 16 accumulator values (scaled)
    MUL_ACC_END = 1680

    # Carry values for MUL accumulation
    MUL_CARRY_START = 1680  # 15 carry values (from columns 0-14 to 1-15)
    MUL_CARRY_END = 1695

    # DIV intermediate values
    DIV_QUOTIENT_START = 1700  # 8 nibbles for quotient
    DIV_REMAINDER_START = 1716  # 8 nibbles for remainder
    DIV_ITERATION = 1732  # Current iteration counter


# =============================================================================
# PURE TRANSFORMER WITH BAKED WEIGHTS
# =============================================================================

class PureTransformer(nn.Module):
    """
    Transformer with baked weights for VM operations.

    === ARCHITECTURE ===
    - 4 transformer layers
    - 8 attention heads
    - Opcode MoE with 33 experts (1 per opcode)

    === EMBEDDING LAYOUT (dim=640) ===
    - [0-15]:    Marker one-hots (PAD, BOS, EOS, CODE, REG_*, MEM, STEP_END, reserved)
    - [16-271]:  Byte value one-hots (256 values)
    - [272-287]: Low nibble (for ALU)
    - [288-303]: High nibble (for ALU)
    - [304-319]: Step position encoding (marker positions)
    - [320-335]: Byte position encoding (section + index)

    === GATHERED VALUES (filled by attention) ===
    - [336-339]: Old PC (4 bytes, gathered from context)
    - [340-343]: Old AX (4 bytes)
    - [344-347]: Old SP (4 bytes)
    - [348-351]: Old BP (4 bytes)
    - [352-355]: Opcode + flags
    - [356-359]: Immediate value (4 bytes)

    === COMPUTED VALUES (filled by FFN/MoE) ===
    - [360-363]: New PC (4 bytes, computed)
    - [364-367]: New AX (4 bytes)
    - [368-371]: New SP (4 bytes)
    - [372-375]: New BP (4 bytes)
    - [376-379]: Memory address (4 bytes)
    - [380-383]: Memory value (4 bytes)

    === WORKSPACE ===
    - [384-639]: Computation workspace

    === ATTENTION PATTERN ===
    Layer 0: Find register markers, gather byte values
    Layer 1: Gather CODE marker, extract opcode + immediate
    Layer 2: Route to MoE expert based on opcode
    Layer 3: Compute new values, prepare output

    === BAKING ===
    All weights are set directly (no training) to implement VM operations.
    """

    def __init__(self, dim: int = 1600, num_heads: int = 8, num_layers: int = 4, use_moe: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_moe = use_moe

        self.tok_emb = nn.Embedding(Vocab.VOCAB_SIZE, dim)
        # Position encoding for step position (0-29)
        self.step_pos_emb = nn.Embedding(30, dim)
        # Layer 0 uses ALiBi for memory recency preference
        # Other layers don't need ALiBi (ALU ops don't depend on position)
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, dim * 4, use_moe=use_moe,
                           use_alibi=(i == 0), alibi_scale=0.1)
            for i in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, Vocab.VOCAB_SIZE, bias=False)

        self._bake_weights()

    def _bake_weights(self):
        """Bake all weights for VM execution."""
        with torch.no_grad():
            self._bake_embeddings()
            self._bake_step_positions()
            for i, layer in enumerate(self.layers):
                self._bake_layer(i, layer)
            self._bake_output()

    def _bake_step_positions(self):
        """
        Bake step position embeddings.

        === LAYOUT ===
        - Dims 304-309: Marker position one-hots (only set for marker positions)
        - Dims 400-405: Register section indicator (PC=0, AX=1, SP=2, BP=3, MEM=4, MEM2=5)
        - Dims 406-409: Byte index indicator (0-3)
        - Dim 410: "Is byte position" flag

        This ensures marker positions and byte positions are clearly distinguished.
        """
        self.step_pos_emb.weight.zero_()

        marker_positions = [0, 5, 10, 15, 20, 29]

        for pos in range(30):
            if pos in marker_positions:
                # Marker positions: set marker one-hot (dims 304-309)
                marker_idx = marker_positions.index(pos)
                if 304 + marker_idx < self.dim:
                    self.step_pos_emb.weight[pos, 304 + marker_idx] = 30.0
            else:
                # Byte positions: set section + byte index + flag
                section = pos // 5  # 0=PC, 1=AX, 2=SP, 3=BP, 4+=MEM
                byte_idx = (pos % 5) - 1  # 0-3 for regs, 0-7 for mem

                if section > 4:
                    section = 4 + (pos - 21) // 4  # MEM section

                if 400 + section < self.dim:
                    self.step_pos_emb.weight[pos, 400 + section] = 10.0
                if byte_idx < 4 and 406 + byte_idx < self.dim:
                    self.step_pos_emb.weight[pos, 406 + byte_idx] = 10.0
                if 410 < self.dim:
                    self.step_pos_emb.weight[pos, 410] = 20.0  # "Is byte position" flag

    def _bake_embeddings(self):
        """
        Bake token embeddings.

        === MARKERS (dims 0-15) ===
        Each marker gets unique one-hot. Scale 10.0 for robust detection.

        === BYTES (dims 16-271) ===
        Each byte 0-255 gets one-hot. Scale 1.0.

        === NIBBLES (dims 272-303) ===
        Byte tokens also encode low/high nibble for ALU ops.

        === BYTE INDEX (dims 400-403) ===
        Position within register (set by step position, not embedding).
        """
        self.tok_emb.weight.zero_()

        # All 16 marker positions (0-15), even if unused
        for i in range(16):
            if i < Vocab.BYTE_BASE:
                self.tok_emb.weight[i, i] = 10.0

        # Byte tokens: one-hot + nibbles
        for b in range(256):
            tok = Vocab.BYTE_BASE + b
            # Primary one-hot (dims 16-271)
            if 16 + b < self.dim:
                self.tok_emb.weight[tok, 16 + b] = 1.0
            # Nibble decomposition (dims 272-303)
            if 272 + (b & 0xF) < self.dim:
                self.tok_emb.weight[tok, 272 + (b & 0xF)] = 1.0
            if 288 + ((b >> 4) & 0xF) < self.dim:
                self.tok_emb.weight[tok, 288 + ((b >> 4) & 0xF)] = 1.0

    def _bake_layer(self, layer_idx: int, layer: TransformerLayer):
        """
        Bake weights for transformer layer.

        === NEURAL GATHERING ARCHITECTURE ===

        Layer 0: Find register markers and start gathering
        - Head 0: Attends to REG_PC marker
        - Head 1: Attends to REG_AX marker
        - Head 2: Attends to REG_SP marker
        - Head 3: Attends to REG_BP marker
        - Head 4: Attends to CODE marker
        - Head 5: Attends to MEM marker
        - Head 6-7: General gathering

        Layer 1: Gather bytes following markers
        - Uses relative position patterns to get byte 0,1,2,3 after marker

        Layer 2-3: Computation via MoE
        - Route based on opcode
        - Compute new register values
        """
        layer.q_proj.weight.zero_()
        layer.k_proj.weight.zero_()
        layer.v_proj.weight.zero_()
        layer.o_proj.weight.zero_()

        head_dim = self.dim // self.num_heads

        if layer_idx == 0:
            # Layer 0: Content-based attention to find markers
            # Each head specializes in finding one marker type
            #
            # Key insight: Q should produce a constant "search key" for each head,
            # and K should respond when the token IS that marker.
            #
            # We achieve this by:
            # - Q: Projects a constant (e.g., bias or specific dim) to search for marker
            # - K: Projects marker one-hot to matching key
            # Heads 0-5: Marker finding
            # Heads 6-7: Reserved for memory value extraction (set up separately)
            markers = [Vocab.REG_PC, Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP,
                      Vocab.CODE, Vocab.MEM]  # Only 6 markers for heads 0-5

            for h, marker in enumerate(markers):
                if h < self.num_heads and marker < 16:
                    # === Marker-finding attention ===
                    # K: Strongly responds to the target marker, weakly negative to others
                    # Q: Projects from byte dims to produce positive value for any token
                    #
                    # Strategy: K uses marker one-hot, Q uses sum of byte dims
                    # Since byte tokens have byte dims set, Q will be positive
                    # K at marker positions will have high value from marker one-hot

                    # K: Strong positive for target marker, negative for others
                    layer.k_proj.weight[h * head_dim, marker] = 100.0  # Target marker
                    for m in range(16):
                        if m != marker:
                            layer.k_proj.weight[h * head_dim, m] = -10.0  # Other markers

                    # Q: Sum of byte dims to produce positive value for any byte token
                    # Byte tokens have one-hot at dims 16-271, so sum is always 1.0
                    # After LayerNorm, this becomes predictable
                    # Use dims 16-31 (first 16 byte values) with positive weights
                    for b in range(16):
                        layer.q_proj.weight[h * head_dim, 16 + b] = 1.0

                    # Also copy from marker dims for marker tokens
                    layer.q_proj.weight[h * head_dim, marker] = 10.0

                    # === MEMORY LOOKUP via ADDRESS MATCHING ===
                    # When LI or LC opcode is active, attention matches addresses
                    # MEM positions have MEM_MARKER_FLAG=50 and address in OP_A dims (+/-20)
                    # Query position has target address in OP_A dims (+/-20)
                    #
                    # Score = K · Q where:
                    #   K[0] = MEM_MARKER_FLAG (50 for MEM, 0 for non-MEM)
                    #   K[1:129] = address nibbles (±20 per dim)
                    #   Q[0] = LI/LC opcode indicator (50)
                    #   Q[1:129] = target address nibbles (±20 per dim)
                    #
                    # For matching address at MEM position:
                    #   Score = 50*50 + 8 * 16 * 400 = 2500 + 51200 = 53700
                    # For non-matching address at MEM position:
                    #   Score = 2500 + 8 * (14*400 - 2*400) = 2500 + 38400 = 40900
                    # For non-MEM position:
                    #   Score = 0 + (low) ≈ 0
                    #
                    # With ALiBi, among matching addresses, recent one wins.
                    if marker == Vocab.MEM:  # Head 5 (MEM head)
                        E = EmbedDims

                        # K[0] projects from MEM_MARKER_FLAG - high for MEM positions
                        layer.k_proj.weight[h * head_dim, E.MEM_MARKER_FLAG] = 1.0

                        # Q[0] projects from LI/LC opcode flags
                        layer.q_proj.weight[h * head_dim, E.OPCODE_START + Opcode.LI] = 50.0
                        layer.q_proj.weight[h * head_dim, E.OPCODE_START + Opcode.LC] = 50.0

                        # === ADDRESS MATCHING via OP_A nibble dims ===
                        # K and Q both project from OP_A nibbles
                        # With +/-20 encoding: match gives 6400, mismatch gives 4800 per nibble
                        nibble_offsets = [E.OP_A_NIBBLE_0_LO, E.OP_A_NIBBLE_0_HI,
                                          E.OP_A_NIBBLE_1_LO, E.OP_A_NIBBLE_1_HI,
                                          E.OP_A_NIBBLE_2_LO, E.OP_A_NIBBLE_2_HI,
                                          E.OP_A_NIBBLE_3_LO, E.OP_A_NIBBLE_3_HI]

                        for nib_idx, nib_offset in enumerate(nibble_offsets):
                            for v in range(16):
                                head_offset = 1 + nib_idx * 16 + v
                                if head_offset < head_dim:
                                    layer.k_proj.weight[h * head_dim + head_offset, nib_offset + v] = 1.0
                                    layer.q_proj.weight[h * head_dim + head_offset, nib_offset + v] = 1.0

            # === MEMORY VALUE EXTRACTION ===
            # Heads 6 and 7 are dedicated to retrieving values from MEM entries
            # They have the same address-matching attention as head 5
            # V projects from RESULT dims at MEM position to copy value
            E = EmbedDims

            # Head 6 handles RESULT nibbles 0-3 (bytes 0-1)
            # Head 7 handles RESULT nibbles 4-7 (bytes 2-3)
            for value_head in [6, 7]:
                # CLEAR this head's projections first
                layer.q_proj.weight[value_head * head_dim:(value_head+1) * head_dim, :] = 0
                layer.k_proj.weight[value_head * head_dim:(value_head+1) * head_dim, :] = 0
                layer.v_proj.weight[value_head * head_dim:(value_head+1) * head_dim, :] = 0

                # K[0] projects from MEM_MARKER_FLAG (high at MEM positions)
                # Use weight 20.0 so MEM contribution = 20 * 50 = 1000 in K[0]
                layer.k_proj.weight[value_head * head_dim, E.MEM_MARKER_FLAG] = 20.0

                # Q[0] projects from LI/LC opcode (high when loading)
                # Use weight 20.0 so opcode contribution = 20 * 20 = 400 in Q[0]
                # MEM_MARKER × Opcode score = 1000 * 400 = 400000 (dominant)
                layer.q_proj.weight[value_head * head_dim, E.OPCODE_START + Opcode.LI] = 20.0
                layer.q_proj.weight[value_head * head_dim, E.OPCODE_START + Opcode.LC] = 20.0

                # Address matching via OP_A nibbles (same as head 5)
                # K and Q both project from OP_A with weight 1.0
                # With +/-20 encoding: match gives 6400, mismatch gives 4800 per nibble
                nibble_offsets = [E.OP_A_NIBBLE_0_LO, E.OP_A_NIBBLE_0_HI,
                                  E.OP_A_NIBBLE_1_LO, E.OP_A_NIBBLE_1_HI,
                                  E.OP_A_NIBBLE_2_LO, E.OP_A_NIBBLE_2_HI,
                                  E.OP_A_NIBBLE_3_LO, E.OP_A_NIBBLE_3_HI]

                for nib_idx, nib_offset in enumerate(nibble_offsets):
                    for v in range(16):
                        head_offset = 1 + nib_idx * 16 + v
                        if head_offset < head_dim:
                            layer.k_proj.weight[value_head * head_dim + head_offset, nib_offset + v] = 1.0
                            layer.q_proj.weight[value_head * head_dim + head_offset, nib_offset + v] = 1.0

                # V projects from RESULT dims (value nibbles stored at MEM position)
                # Head 6: RESULT nibbles 0-3 (bytes 0-1, dims 736-799)
                # Head 7: RESULT nibbles 4-7 (bytes 2-3, dims 800-863)
                if value_head == 6:
                    result_start = E.RESULT_0_LO
                    result_end = E.RESULT_1_HI + 16  # Nibbles 0-3
                else:
                    result_start = E.RESULT_2_LO
                    result_end = E.RESULT_3_HI + 16  # Nibbles 4-7

                for i, src_dim in enumerate(range(result_start, result_end)):
                    if src_dim < self.dim and i < head_dim:
                        layer.v_proj.weight[value_head * head_dim + i, src_dim] = 5.0

                # O projects V output back to RESULT dims at query position
                for i, dst_dim in enumerate(range(result_start, result_end)):
                    if dst_dim < self.dim and i < head_dim:
                        layer.o_proj.weight[dst_dim, value_head * head_dim + i] = 5.0

            # V: Also copy standard dims (markers, bytes) for non-memory operations
            for i in range(min(self.dim, 400)):
                layer.v_proj.weight[i, i] = 1.0
                layer.o_proj.weight[i, i] = 1.0

        elif layer_idx == 1:
            # Layer 1: Gather byte values with position-relative routing
            # After layer 0, hidden state has marker-weighted sums
            # Now we gather specific bytes and route their nibbles to OP_A/OP_B dims
            #
            # === POSITION-RELATIVE BYTE GATHERING ===
            # For context: [MEM, byte0, byte1, byte2, byte3, ..., QUERY]
            # We need to route:
            #   - byte0's nibbles → OP_A_NIBBLE_0_*
            #   - byte1's nibbles → OP_A_NIBBLE_1_*
            #   - byte2's nibbles → OP_A_NIBBLE_2_*
            #   - byte3's nibbles → OP_A_NIBBLE_3_*
            #
            # Use step position encoding to determine which byte slot each position maps to.
            # Byte positions 1-4 (after marker) map to byte indices 0-3.

            E = EmbedDims

            # === IDENTITY FOR PASS-THROUGH DIMS ===
            # Set identity ONLY for dims that don't need specialized routing
            # Skip: NIBBLE dims (272-303), OP_A/B dims (480-735), head gather dims
            for i in range(min(self.dim, 272)):
                layer.q_proj.weight[i, i] = 1.0
                layer.k_proj.weight[i, i] = 1.0
                layer.v_proj.weight[i, i] = 1.0
                layer.o_proj.weight[i, i] = 1.0

            # Also pass through RESULT and OUTPUT dims
            for i in range(E.RESULT_NIBBLES_START, min(E.OUTPUT_BYTE_END, self.dim)):
                layer.v_proj.weight[i, i] = 1.0
                layer.o_proj.weight[i, i] = 1.0

            # === BYTE GATHERING TO OP_A NIBBLE DIMS ===
            # Use heads 0-3 for gathering bytes 0-3
            # Each head attends to positions with specific byte index
            # and copies byte's nibbles to OP_A_NIBBLE_* dims

            # Step position embedding marks byte index in dims 406-409
            # Head h attends where dim 406+h is set, copies nibbles to OP_A_NIBBLE_h_*
            # Query position has marker in dim 404 (QUERY_ACTIVE)
            QUERY_MARKER_DIM = 404

            for byte_idx in range(4):
                h = byte_idx
                byte_index_dim = 406 + byte_idx

                # Clear this head's portion of Q/K projections
                layer.q_proj.weight[h * head_dim:(h+1) * head_dim, :] = 0
                layer.k_proj.weight[h * head_dim:(h+1) * head_dim, :] = 0
                layer.v_proj.weight[h * head_dim:(h+1) * head_dim, :] = 0

                # Q: Query active when dim 404 is set (single marker, always positive)
                layer.q_proj.weight[h * head_dim, QUERY_MARKER_DIM] = 50.0

                # K: Key responds when this byte index is set
                # K[0] projects from byte_index_dim
                layer.k_proj.weight[h * head_dim, byte_index_dim] = 100.0

                # V: Copy byte's nibbles (dims 272-287, 288-303) to head dims
                # Input: byte at position with byte_index_dim set
                # Output: nibbles in head dims 1-16 (low) and 17-32 (high)
                # Note: Need high weights to compensate for softmax1 halving and LayerNorm

                lo_src = E.NIBBLE_A_START  # 272 (low nibble of byte)
                hi_src = E.NIBBLE_B_START  # 288 (high nibble of byte)

                # Copy low nibble (16 dims) to head dims 1-16
                for v in range(16):
                    v_offset = 1 + v
                    if lo_src + v < self.dim:
                        layer.v_proj.weight[h * head_dim + v_offset, lo_src + v] = 50.0

                # Copy high nibble (16 dims) to head dims 17-32
                for v in range(16):
                    v_offset = 17 + v
                    if hi_src + v < self.dim:
                        layer.v_proj.weight[h * head_dim + v_offset, hi_src + v] = 50.0

                # O: Route head dims to OP_A_NIBBLE slots (strong weight for MoE input)
                lo_dst = E.OP_A_NIBBLE_0_LO + byte_idx * 32
                hi_dst = E.OP_A_NIBBLE_0_HI + byte_idx * 32

                for v in range(16):
                    v_offset = 1 + v
                    if lo_dst + v < self.dim:
                        layer.o_proj.weight[lo_dst + v, h * head_dim + v_offset] = 5.0

                for v in range(16):
                    v_offset = 17 + v
                    if hi_dst + v < self.dim:
                        layer.o_proj.weight[hi_dst + v, h * head_dim + v_offset] = 5.0

            # === OPERAND B GATHERING (heads 4-7) ===
            # For binary ops, second operand is at different positions
            # Head 4-7 gather bytes to OP_B_NIBBLE_* dims
            # Use different byte_index markers (410-413) for operand B
            for byte_idx in range(4):
                h = 4 + byte_idx
                if h >= self.num_heads:
                    break

                byte_index_dim = 410 + byte_idx  # Different markers for operand B

                # Clear this head's portion
                layer.q_proj.weight[h * head_dim:(h+1) * head_dim, :] = 0
                layer.k_proj.weight[h * head_dim:(h+1) * head_dim, :] = 0
                layer.v_proj.weight[h * head_dim:(h+1) * head_dim, :] = 0

                # Q: Query active when dim 404 is set (same marker for all query positions)
                layer.q_proj.weight[h * head_dim, QUERY_MARKER_DIM] = 50.0

                # K: Respond to operand B byte_index
                layer.k_proj.weight[h * head_dim, byte_index_dim] = 100.0

                lo_src = E.NIBBLE_A_START
                hi_src = E.NIBBLE_B_START
                lo_dst = E.OP_B_NIBBLE_0_LO + byte_idx * 32
                hi_dst = E.OP_B_NIBBLE_0_HI + byte_idx * 32

                for v in range(16):
                    v_offset = 1 + v
                    if lo_src + v < self.dim:
                        layer.v_proj.weight[h * head_dim + v_offset, lo_src + v] = 50.0

                for v in range(16):
                    v_offset = 17 + v
                    if hi_src + v < self.dim:
                        layer.v_proj.weight[h * head_dim + v_offset, hi_src + v] = 50.0

                for v in range(16):
                    v_offset = 1 + v
                    if lo_dst + v < self.dim:
                        layer.o_proj.weight[lo_dst + v, h * head_dim + v_offset] = 5.0

                for v in range(16):
                    v_offset = 17 + v
                    if hi_dst + v < self.dim:
                        layer.o_proj.weight[hi_dst + v, h * head_dim + v_offset] = 5.0

        elif layer_idx == 2:
            # Layer 2: MoE computation - computes base results and G/P flags
            # Pass through all dims up to OP_B_NIBBLES_END (736) to preserve operand nibbles
            E = EmbedDims
            for i in range(min(self.dim, E.OP_B_NIBBLES_END)):
                layer.q_proj.weight[i, i] = 1.0
                layer.k_proj.weight[i, i] = 1.0
                layer.v_proj.weight[i, i] = 1.0
                layer.o_proj.weight[i, i] = 1.0

            # Bake MoE experts for computation
            if layer.use_moe:
                self._bake_opcode_moe(layer.ffn)

        elif layer_idx == 3:
            # Layer 3: Carry chain computation + final result selection
            # Read G/P flags from layer 2 output, compute carry chain
            # Pass through dims up to RESULT_NIBBLES_END to preserve computed values
            E = EmbedDims
            for i in range(min(self.dim, E.RESULT_NIBBLES_END)):
                layer.q_proj.weight[i, i] = 1.0
                layer.k_proj.weight[i, i] = 1.0
                layer.v_proj.weight[i, i] = 1.0
                layer.o_proj.weight[i, i] = 1.0

            # Bake carry chain computation into the FFN
            if layer.use_moe:
                self._bake_carry_chain_layer(layer.ffn)
            return  # Don't bake regular MoE

        # Bake MoE experts (for layers 0-1)
        if layer.use_moe and layer_idx < 2:
            self._bake_opcode_moe(layer.ffn)

    def _bake_opcode_moe(self, moe: OpcodeMoE):
        """
        Bake MoE with one expert per opcode.

        Router: Routes based on opcode byte in hidden dim 320-351 (opcode one-hot)
        Experts: Each implements its opcode's computation via nibble tables
        """
        moe.router.weight.zero_()

        # Route based on opcode encoding in dims 320-351
        # Opcode is stored as one-hot: dim 320 + opcode_idx = 1.0
        for idx, opcode in enumerate(Opcode.ALL):
            # Router weight: expert_idx responds to dim 320 + opcode
            if 320 + opcode < self.dim:
                moe.router.weight[idx, 320 + opcode] = 20.0  # Strong signal

        # Default fallback: expert 0 if no opcode detected
        moe.router.weight[0, 0] = 1.0

        # Bake each expert for its opcode
        for idx, expert in enumerate(moe.experts):
            self._bake_opcode_expert(expert, Opcode.ALL[idx])

    def _bake_opcode_expert(self, expert: OpcodeExpert, opcode: int):
        """
        Bake weights for a single opcode expert.

        Each expert implements its opcode:
        - ADD expert: Nibble addition tables
        - SUB expert: Subtraction via negation + add
        - MUL expert: SwiGLU multiplication
        - AND/OR/XOR: Bitwise tables
        - etc.
        """
        expert.up.weight.zero_()
        expert.gate.weight.zero_()
        expert.down.weight.zero_()

        # Don't use identity pass-through for ALU experts
        # This prevents interference with computed outputs
        # (Identity was for passing through non-computed values)

        # Specialize based on opcode
        if opcode == Opcode.IMM:
            # IMM: Copy operand_b (immediate) to output
            # Operand_b injected at dims 344-351, output to 384-399 and 400-415
            self._bake_imm(expert)
        elif opcode == Opcode.ADD:
            # Addition: both single-nibble and parallel 32-bit tables
            self._bake_nibble_add(expert)
            self._bake_parallel_32bit_add(expert)
        elif opcode == Opcode.SUB:
            # Subtraction: similar structure with borrow
            self._bake_nibble_sub(expert)
            self._bake_parallel_32bit_sub(expert)
        elif opcode == Opcode.AND:
            self._bake_nibble_op(expert, lambda a, b: a & b)
            self._bake_parallel_32bit_bitwise(expert, lambda a, b: a & b)
        elif opcode == Opcode.OR:
            self._bake_nibble_op(expert, lambda a, b: a | b)
            self._bake_parallel_32bit_bitwise(expert, lambda a, b: a | b)
        elif opcode == Opcode.XOR:
            self._bake_nibble_op(expert, lambda a, b: a ^ b)
            self._bake_parallel_32bit_bitwise(expert, lambda a, b: a ^ b)
        elif opcode == Opcode.SHL:
            self._bake_shift_left(expert)
        elif opcode == Opcode.SHR:
            self._bake_shift_right(expert)
        elif opcode == Opcode.MUL:
            self._bake_mul(expert)
        elif opcode in [Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE]:
            self._bake_comparison(expert, opcode)
        elif opcode == Opcode.EXIT:
            self._bake_exit(expert)
        elif opcode == Opcode.LEA:
            self._bake_lea(expert)
        elif opcode == Opcode.ADJ:
            self._bake_adj(expert)
        elif opcode == Opcode.DIV:
            self._bake_div(expert)
        elif opcode == Opcode.MOD:
            self._bake_mod(expert)
        elif opcode == Opcode.PSH:
            self._bake_psh(expert)
        elif opcode == Opcode.ENT:
            self._bake_ent(expert)
        elif opcode == Opcode.LEV:
            self._bake_lev(expert)
        elif opcode == Opcode.JMP:
            self._bake_jmp(expert)
        elif opcode == Opcode.BZ:
            self._bake_bz(expert)
        elif opcode == Opcode.BNZ:
            self._bake_bnz(expert)
        elif opcode == Opcode.JSR:
            self._bake_jsr(expert)
        elif opcode == Opcode.LI:
            self._bake_li(expert)
        elif opcode == Opcode.LC:
            self._bake_lc(expert)
        elif opcode == Opcode.SI:
            self._bake_si(expert)
        elif opcode == Opcode.SC:
            self._bake_sc(expert)
        elif opcode == Opcode.MALC:
            self._bake_malc(expert)
        elif opcode == Opcode.FREE:
            self._bake_free(expert)
        elif opcode == Opcode.GETC:
            self._bake_getc(expert)
        elif opcode == Opcode.PUTC:
            self._bake_putc(expert)
        elif opcode == Opcode.READ:
            self._bake_read(expert)
        elif opcode == Opcode.PRTF:
            self._bake_prtf(expert)

    def _bake_li(self, expert: OpcodeExpert):
        """
        Bake LI (Load Int) expert.

        LI reads a 32-bit value from memory. The attention in layer 0 has already
        copied the value from the matching MEM entry to RESULT dims. This expert
        copies RESULT to OUTPUT.

        For each result nibble: OUTPUT[v] = RESULT[v]
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.LI

        result_nibble_offsets = [E.RESULT_0_LO, E.RESULT_0_HI,
                                 E.RESULT_1_LO, E.RESULT_1_HI,
                                 E.RESULT_2_LO, E.RESULT_2_HI,
                                 E.RESULT_3_LO, E.RESULT_3_HI]

        # For each nibble, copy from RESULT to OUTPUT
        base_addr = 0
        for nib_idx, result_offset in enumerate(result_nibble_offsets):
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    # up activates for this result nibble value
                    expert.up.weight[addr, result_offset + v] = 5.0

                    # gate activates for LI opcode
                    expert.gate.weight[addr, opcode_dim] = 5.0

                    # down outputs to OUTPUT_BYTE (combined 32-bit result)
                    # For nibble-based output, we output to the result nibble dims
                    # which get combined in later processing
                    expert.down.weight[result_offset + v, addr] = 5.0

            base_addr += 16

    def _bake_lc(self, expert: OpcodeExpert):
        """
        Bake LC (Load Char) expert.

        LC reads an 8-bit value from memory. Similar to LI but masks to lower byte.
        The attention has copied the full value to RESULT dims. This expert
        copies only the lower byte (nibbles 0 and 1) to OUTPUT.
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.LC

        # Only copy nibbles 0 and 1 (lower byte)
        result_nibble_offsets = [E.RESULT_0_LO, E.RESULT_0_HI]

        base_addr = 0
        for nib_idx, result_offset in enumerate(result_nibble_offsets):
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, result_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[result_offset + v, addr] = 5.0
            base_addr += 16

        # Set nibbles 2-7 to 0 (mask to 8 bits)
        zero_nibble_offsets = [E.RESULT_1_LO, E.RESULT_1_HI,
                               E.RESULT_2_LO, E.RESULT_2_HI,
                               E.RESULT_3_LO, E.RESULT_3_HI]

        for result_offset in zero_nibble_offsets:
            addr = base_addr
            if addr < expert.ffn_dim:
                # Always output 0 for these nibbles
                expert.up.weight[addr, opcode_dim] = 5.0  # Activate on LC
                expert.gate.weight[addr, opcode_dim] = 5.0
                expert.down.weight[result_offset + 0, addr] = 5.0  # Output nibble = 0
            base_addr += 1

    def _bake_si(self, expert: OpcodeExpert):
        """
        Bake SI (Store Int) expert.

        SI writes a 32-bit value to memory. The model generates a MEM token
        with address and value embedded. The address comes from OP_A dims
        (populated by attention from stack), and the value from OP_B dims
        (populated by attention from AX).

        This expert:
        1. Outputs MEM token (token 8)
        2. Sets MEM_OP_WRITE flag
        3. Copies address from OP_A to MEM_ADDR dims
        4. Copies value from OP_B to MEM_VALUE dims (reuses RESULT dims)
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.SI

        # Output MEM token (token ID 8)
        # Set the MEM marker dim in output
        base_addr = 0
        if base_addr < expert.ffn_dim:
            expert.up.weight[base_addr, opcode_dim] = 5.0
            expert.gate.weight[base_addr, opcode_dim] = 5.0
            expert.down.weight[Vocab.MEM, base_addr] = 10.0  # Output MEM marker
        base_addr += 1

        # Set MEM_OP_WRITE flag
        if base_addr < expert.ffn_dim and E.MEM_OP_WRITE < self.dim:
            expert.up.weight[base_addr, opcode_dim] = 5.0
            expert.gate.weight[base_addr, opcode_dim] = 5.0
            expert.down.weight[E.MEM_OP_WRITE, base_addr] = 10.0
        base_addr += 1

        # Copy address nibbles from OP_A to preserved in output
        # (The embedding injection will read from OP_A dims directly)
        addr_offsets = [E.OP_A_NIBBLE_0_LO, E.OP_A_NIBBLE_0_HI,
                        E.OP_A_NIBBLE_1_LO, E.OP_A_NIBBLE_1_HI,
                        E.OP_A_NIBBLE_2_LO, E.OP_A_NIBBLE_2_HI,
                        E.OP_A_NIBBLE_3_LO, E.OP_A_NIBBLE_3_HI]

        for nib_offset in addr_offsets:
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    # Pass through address nibbles
                    expert.up.weight[addr, nib_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[nib_offset + v, addr] = 1.0
            base_addr += 16

        # Copy value nibbles from OP_B to RESULT dims
        val_src_offsets = [E.OP_B_NIBBLE_0_LO, E.OP_B_NIBBLE_0_HI,
                           E.OP_B_NIBBLE_1_LO, E.OP_B_NIBBLE_1_HI,
                           E.OP_B_NIBBLE_2_LO, E.OP_B_NIBBLE_2_HI,
                           E.OP_B_NIBBLE_3_LO, E.OP_B_NIBBLE_3_HI]

        val_dst_offsets = [E.RESULT_0_LO, E.RESULT_0_HI,
                           E.RESULT_1_LO, E.RESULT_1_HI,
                           E.RESULT_2_LO, E.RESULT_2_HI,
                           E.RESULT_3_LO, E.RESULT_3_HI]

        for src_offset, dst_offset in zip(val_src_offsets, val_dst_offsets):
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, src_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[dst_offset + v, addr] = 1.0
            base_addr += 16

    def _bake_sc(self, expert: OpcodeExpert):
        """
        Bake SC (Store Char) expert.

        SC writes an 8-bit value to memory. Similar to SI but only stores
        the lower byte of the value.
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.SC

        # Output MEM token
        base_addr = 0
        if base_addr < expert.ffn_dim:
            expert.up.weight[base_addr, opcode_dim] = 5.0
            expert.gate.weight[base_addr, opcode_dim] = 5.0
            expert.down.weight[Vocab.MEM, base_addr] = 10.0
        base_addr += 1

        # Set MEM_OP_WRITE flag
        if base_addr < expert.ffn_dim and E.MEM_OP_WRITE < self.dim:
            expert.up.weight[base_addr, opcode_dim] = 5.0
            expert.gate.weight[base_addr, opcode_dim] = 5.0
            expert.down.weight[E.MEM_OP_WRITE, base_addr] = 10.0
        base_addr += 1

        # Copy address nibbles (full 32-bit address)
        addr_offsets = [E.OP_A_NIBBLE_0_LO, E.OP_A_NIBBLE_0_HI,
                        E.OP_A_NIBBLE_1_LO, E.OP_A_NIBBLE_1_HI,
                        E.OP_A_NIBBLE_2_LO, E.OP_A_NIBBLE_2_HI,
                        E.OP_A_NIBBLE_3_LO, E.OP_A_NIBBLE_3_HI]

        for nib_offset in addr_offsets:
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, nib_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[nib_offset + v, addr] = 1.0
            base_addr += 16

        # Copy only lower byte of value (nibbles 0 and 1)
        val_src_offsets = [E.OP_B_NIBBLE_0_LO, E.OP_B_NIBBLE_0_HI]
        val_dst_offsets = [E.RESULT_0_LO, E.RESULT_0_HI]

        for src_offset, dst_offset in zip(val_src_offsets, val_dst_offsets):
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, src_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[dst_offset + v, addr] = 1.0
            base_addr += 16

        # Set upper nibbles to 0
        upper_offsets = [E.RESULT_1_LO, E.RESULT_1_HI,
                         E.RESULT_2_LO, E.RESULT_2_HI,
                         E.RESULT_3_LO, E.RESULT_3_HI]

        for dst_offset in upper_offsets:
            addr = base_addr
            if addr < expert.ffn_dim:
                expert.up.weight[addr, opcode_dim] = 5.0
                expert.gate.weight[addr, opcode_dim] = 5.0
                expert.down.weight[dst_offset + 0, addr] = 1.0  # Set to 0
            base_addr += 1

    def _bake_malc(self, expert: OpcodeExpert):
        """
        Bake MALC (malloc) expert.

        MALC allocates memory by:
        1. Reading current heap pointer from RESULT dims (populated by attention from HEAP/MEM[0])
        2. Adding size from OP_A dims (from stack)
        3. Outputting old heap pointer to result (return value)
        4. Generating new MEM entry at address 0 with updated heap pointer

        For simplicity, this implementation:
        - Reads heap pointer from RESULT (attention pre-populated from MEM[0])
        - Outputs current heap pointer value unchanged (for return)
        - Sets up for SI to write new value at address 0

        The actual heap pointer update happens via SI-like token generation.
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.MALC

        # MALC just passes through RESULT dims (heap pointer value from attention)
        # The heap pointer addition and update is handled externally
        result_offsets = [E.RESULT_0_LO, E.RESULT_0_HI,
                          E.RESULT_1_LO, E.RESULT_1_HI,
                          E.RESULT_2_LO, E.RESULT_2_HI,
                          E.RESULT_3_LO, E.RESULT_3_HI]

        base_addr = 0
        for result_offset in result_offsets:
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    # Pass through RESULT nibbles
                    expert.up.weight[addr, result_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[result_offset + v, addr] = 1.0
            base_addr += 16

    def _bake_free(self, expert: OpcodeExpert):
        """
        Bake FREE expert.

        FREE deallocates memory. In simple bump allocators, FREE is a no-op.
        For more complex allocators, FREE might write a special marker.

        This implementation:
        - Simply outputs 0 as the result (successful free)
        - The actual deallocation is conceptual (memory can be reused by MALC)
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.FREE

        # Output 0 (success)
        result_offsets = [E.RESULT_0_LO, E.RESULT_0_HI,
                          E.RESULT_1_LO, E.RESULT_1_HI,
                          E.RESULT_2_LO, E.RESULT_2_HI,
                          E.RESULT_3_LO, E.RESULT_3_HI]

        base_addr = 0
        for result_offset in result_offsets:
            addr = base_addr
            if addr < expert.ffn_dim:
                # Output nibble = 0
                expert.up.weight[addr, opcode_dim] = 5.0
                expert.gate.weight[addr, opcode_dim] = 5.0
                expert.down.weight[result_offset + 0, addr] = 1.0
            base_addr += 1

    def _bake_getc(self, expert: OpcodeExpert):
        """
        Bake GETC (getchar) expert.

        GETC reads a character from input. The input character is provided
        in RESULT dims by the attention mechanism (from an INPUT marker in context).

        This implementation passes through RESULT dims unchanged.
        If no input is available, returns -1 (0xFFFFFFFF).
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.GETC

        # Pass through RESULT dims (input char from attention)
        result_offsets = [E.RESULT_0_LO, E.RESULT_0_HI,
                          E.RESULT_1_LO, E.RESULT_1_HI,
                          E.RESULT_2_LO, E.RESULT_2_HI,
                          E.RESULT_3_LO, E.RESULT_3_HI]

        base_addr = 0
        for result_offset in result_offsets:
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, result_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[result_offset + v, addr] = 1.0
            base_addr += 16

    def _bake_putc(self, expert: OpcodeExpert):
        """
        Bake PUTC (putchar) expert.

        PUTC outputs a character. The character is in OP_A (from stack).
        This implementation copies OP_A low byte to RESULT for output.
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.PUTC

        # Copy lower byte of OP_A to RESULT (character to output)
        src_offsets = [E.OP_A_NIBBLE_0_LO, E.OP_A_NIBBLE_0_HI]
        dst_offsets = [E.RESULT_0_LO, E.RESULT_0_HI]

        base_addr = 0
        for src_offset, dst_offset in zip(src_offsets, dst_offsets):
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, src_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[dst_offset + v, addr] = 1.0
            base_addr += 16

        # Set upper bytes to 0
        upper_offsets = [E.RESULT_1_LO, E.RESULT_1_HI,
                         E.RESULT_2_LO, E.RESULT_2_HI,
                         E.RESULT_3_LO, E.RESULT_3_HI]

        for dst_offset in upper_offsets:
            addr = base_addr
            if addr < expert.ffn_dim:
                expert.up.weight[addr, opcode_dim] = 5.0
                expert.gate.weight[addr, opcode_dim] = 5.0
                expert.down.weight[dst_offset + 0, addr] = 1.0
            base_addr += 1

    def _bake_read(self, expert: OpcodeExpert):
        """
        Bake READ expert.

        READ reads bytes from input (file descriptor). For simplicity,
        this implementation is similar to GETC - passes through input
        from RESULT dims.
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.READ

        # Pass through RESULT dims
        result_offsets = [E.RESULT_0_LO, E.RESULT_0_HI,
                          E.RESULT_1_LO, E.RESULT_1_HI,
                          E.RESULT_2_LO, E.RESULT_2_HI,
                          E.RESULT_3_LO, E.RESULT_3_HI]

        base_addr = 0
        for result_offset in result_offsets:
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, result_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[result_offset + v, addr] = 1.0
            base_addr += 16

    def _bake_prtf(self, expert: OpcodeExpert):
        """
        Bake PRTF (printf) expert.

        PRTF outputs formatted text. For simplicity, this implementation
        just passes through the character from OP_A like PUTC.
        """
        E = EmbedDims
        opcode_dim = E.OPCODE_START + Opcode.PRTF

        # Copy lower byte of OP_A to RESULT
        src_offsets = [E.OP_A_NIBBLE_0_LO, E.OP_A_NIBBLE_0_HI]
        dst_offsets = [E.RESULT_0_LO, E.RESULT_0_HI]

        base_addr = 0
        for src_offset, dst_offset in zip(src_offsets, dst_offsets):
            for v in range(16):
                addr = base_addr + v
                if addr < expert.ffn_dim:
                    expert.up.weight[addr, src_offset + v] = 5.0
                    expert.gate.weight[addr, opcode_dim] = 5.0
                    expert.down.weight[dst_offset + v, addr] = 1.0
            base_addr += 16

        # Set upper bytes to 0
        upper_offsets = [E.RESULT_1_LO, E.RESULT_1_HI,
                         E.RESULT_2_LO, E.RESULT_2_HI,
                         E.RESULT_3_LO, E.RESULT_3_HI]

        for dst_offset in upper_offsets:
            addr = base_addr
            if addr < expert.ffn_dim:
                expert.up.weight[addr, opcode_dim] = 5.0
                expert.gate.weight[addr, opcode_dim] = 5.0
                expert.down.weight[dst_offset + 0, addr] = 1.0
            base_addr += 1

    def _bake_nibble_add(self, expert: OpcodeExpert):
        """
        Bake nibble addition with proper operand separation.

        SwiGLU AND-gate structure:
        - up reads ONLY from nibble A positions
        - gate reads ONLY from nibble B positions
        """
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b  # 256 unique addresses
                result = a + b     # 0-30

                if addr < expert.ffn_dim:
                    # up reads nibble A from both injection and attention-gathered dims
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.GATHERED_A_START + a < self.dim:
                        expert.up.weight[addr, E.GATHERED_A_START + a] = 8.0

                    # gate reads nibble B from both injection and attention-gathered dims
                    if E.NIBBLE_B_START + b < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0
                    if E.GATHERED_B_START + b < self.dim:
                        expert.gate.weight[addr, E.GATHERED_B_START + b] = 8.0

                    # Output: result byte
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_nibble_sub(self, expert: OpcodeExpert):
        """Bake nibble subtraction lookup table with byte output."""
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                if addr < expert.ffn_dim:
                    # SwiGLU AND-gate: up reads nibble A, gate reads nibble B
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + b < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0

                    # Compute result and output as byte
                    result = (a - b) & 0xFF  # Handle negative wrap
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_nibble_op(self, expert: OpcodeExpert, op_fn):
        """Bake generic nibble operation lookup table with byte output."""
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                if addr < expert.ffn_dim:
                    # SwiGLU AND-gate: up reads nibble A, gate reads nibble B
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + b < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0

                    # Compute result and output as byte
                    result = op_fn(a, b) & 0xFF
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_parallel_32bit_add(self, expert: OpcodeExpert):
        """
        Bake parallel 32-bit addition - all 8 nibbles computed in one forward pass.

        Layout: 8 nibble pairs × 256 addresses = 2048 FFN units
        - Addresses 0-255: Byte 0 low nibble
        - Addresses 256-511: Byte 0 high nibble
        - Addresses 512-767: Byte 1 low nibble
        - ... and so on

        Outputs to extended result dims (RESULT_0_LO, RESULT_0_HI, etc.)
        Also outputs raw sums to SUM dims for carry detection.
        """
        E = EmbedDims

        # Nibble input/output pairs for each of 8 positions
        nibble_pairs = [
            (E.OP_A_NIBBLE_0_LO, E.OP_B_NIBBLE_0_LO, E.RESULT_0_LO, E.SUM_0_LO),
            (E.OP_A_NIBBLE_0_HI, E.OP_B_NIBBLE_0_HI, E.RESULT_0_HI, E.SUM_0_HI),
            (E.OP_A_NIBBLE_1_LO, E.OP_B_NIBBLE_1_LO, E.RESULT_1_LO, E.SUM_1_LO),
            (E.OP_A_NIBBLE_1_HI, E.OP_B_NIBBLE_1_HI, E.RESULT_1_HI, E.SUM_1_HI),
            (E.OP_A_NIBBLE_2_LO, E.OP_B_NIBBLE_2_LO, E.RESULT_2_LO, E.SUM_2_LO),
            (E.OP_A_NIBBLE_2_HI, E.OP_B_NIBBLE_2_HI, E.RESULT_2_HI, E.SUM_2_HI),
            (E.OP_A_NIBBLE_3_LO, E.OP_B_NIBBLE_3_LO, E.RESULT_3_LO, E.SUM_3_LO),
            (E.OP_A_NIBBLE_3_HI, E.OP_B_NIBBLE_3_HI, E.RESULT_3_HI, E.SUM_3_HI),
        ]

        for nibble_idx, (a_start, b_start, result_start, sum_start) in enumerate(nibble_pairs):
            base_addr = nibble_idx * 256  # Each nibble pair gets 256 addresses

            for a in range(16):
                for b in range(16):
                    addr = base_addr + a * 16 + b

                    if addr < expert.ffn_dim:
                        # up reads from this nibble's A slot
                        if a_start + a < self.dim:
                            expert.up.weight[addr, a_start + a] = 8.0

                        # gate reads from this nibble's B slot
                        if b_start + b < self.dim:
                            expert.gate.weight[addr, b_start + b] = 8.0

                        # Output: result nibble (0-15) to RESULT dims
                        sum_val = a + b  # 0-30
                        result_nibble = sum_val & 0xF

                        if result_start + result_nibble < self.dim:
                            expert.down.weight[result_start + result_nibble, addr] = 5.0

                        # Also output raw sum (0-30) to SUM dims for carry detection
                        if sum_start + sum_val < self.dim:
                            expert.down.weight[sum_start + sum_val, addr] = 5.0

                        # === 100% NEURAL: Output carry flag directly ===
                        # If sum > 15, output 1.0 to CARRY_OUT dim
                        carry_out_dims = [
                            E.CARRY_OUT_0_LO, E.CARRY_OUT_0_HI,
                            E.CARRY_OUT_1_LO, E.CARRY_OUT_1_HI,
                            E.CARRY_OUT_2_LO, E.CARRY_OUT_2_HI,
                            E.CARRY_OUT_3_LO, E.CARRY_OUT_3_HI,
                        ]
                        if nibble_idx < len(carry_out_dims):
                            carry_dim = carry_out_dims[nibble_idx]
                            if carry_dim < self.dim:
                                # Output carry flag: 5.0 if sum > 15, 0 otherwise
                                if sum_val > 15:
                                    expert.down.weight[carry_dim, addr] = 5.0

                        # === CARRY LOOKAHEAD: Generate and Propagate flags ===
                        # G (Generate): sum > 15, generates carry regardless of input
                        # P (Propagate): sum == 15, propagates incoming carry
                        if nibble_idx < len(E.CARRY_GEN):
                            gen_dim = E.CARRY_GEN[nibble_idx]
                            prop_dim = E.CARRY_PROP[nibble_idx]
                            if gen_dim < self.dim and sum_val > 15:
                                expert.down.weight[gen_dim, addr] = 5.0
                            if prop_dim < self.dim and sum_val == 15:
                                expert.down.weight[prop_dim, addr] = 5.0

                        # === OUTPUT BOTH: without-carry and with-carry results ===
                        # MoE outputs RESULT (base) and RESULT_CARRY (base+1)
                        # Later layer selects based on carry chain
                        if nibble_idx < 8:
                            # Result dims for "with carry" case
                            carry_result_dims = [
                                E.RESULT_CARRY_0_LO, E.RESULT_CARRY_0_HI,
                                E.RESULT_CARRY_1_LO, E.RESULT_CARRY_1_HI,
                                E.RESULT_CARRY_2_LO, E.RESULT_CARRY_2_HI,
                                E.RESULT_CARRY_3_LO, E.RESULT_CARRY_3_HI,
                            ]
                            carry_start = carry_result_dims[nibble_idx]
                            # Result if carry arrives = (base + 1) & 0xF
                            result_with_carry = (result_nibble + 1) & 0xF
                            if carry_start + result_with_carry < self.dim:
                                expert.down.weight[carry_start + result_with_carry, addr] = 5.0

                            # Also output carry-out for with-carry case
                            # If base result is 15 and carry arrives, we generate new carry
                            if result_nibble == 15:
                                # Propagate: if carry comes in, it goes out
                                pass  # Already handled by P flag

    def _bake_parallel_32bit_sub(self, expert: OpcodeExpert):
        """
        Bake parallel 32-bit subtraction - all 8 nibbles computed in one forward pass.

        Like ADD but with borrow instead of carry:
        - diff = A - B
        - If diff < 0: result = diff + 16, borrow_out = 1
        - Else: result = diff, borrow_out = 0

        Also outputs RESULT_CARRY as result WITH incoming borrow (result - 1).
        """
        E = EmbedDims

        nibble_pairs = [
            (E.OP_A_NIBBLE_0_LO, E.OP_B_NIBBLE_0_LO, E.RESULT_0_LO),
            (E.OP_A_NIBBLE_0_HI, E.OP_B_NIBBLE_0_HI, E.RESULT_0_HI),
            (E.OP_A_NIBBLE_1_LO, E.OP_B_NIBBLE_1_LO, E.RESULT_1_LO),
            (E.OP_A_NIBBLE_1_HI, E.OP_B_NIBBLE_1_HI, E.RESULT_1_HI),
            (E.OP_A_NIBBLE_2_LO, E.OP_B_NIBBLE_2_LO, E.RESULT_2_LO),
            (E.OP_A_NIBBLE_2_HI, E.OP_B_NIBBLE_2_HI, E.RESULT_2_HI),
            (E.OP_A_NIBBLE_3_LO, E.OP_B_NIBBLE_3_LO, E.RESULT_3_LO),
            (E.OP_A_NIBBLE_3_HI, E.OP_B_NIBBLE_3_HI, E.RESULT_3_HI),
        ]

        borrow_out_dims = [
            E.CARRY_OUT_0_LO, E.CARRY_OUT_0_HI,
            E.CARRY_OUT_1_LO, E.CARRY_OUT_1_HI,
            E.CARRY_OUT_2_LO, E.CARRY_OUT_2_HI,
            E.CARRY_OUT_3_LO, E.CARRY_OUT_3_HI,
        ]

        borrow_result_dims = [
            E.RESULT_CARRY_0_LO, E.RESULT_CARRY_0_HI,
            E.RESULT_CARRY_1_LO, E.RESULT_CARRY_1_HI,
            E.RESULT_CARRY_2_LO, E.RESULT_CARRY_2_HI,
            E.RESULT_CARRY_3_LO, E.RESULT_CARRY_3_HI,
        ]

        for nibble_idx, (a_start, b_start, result_start) in enumerate(nibble_pairs):
            base_addr = nibble_idx * 256

            for a in range(16):
                for b in range(16):
                    addr = base_addr + a * 16 + b

                    if addr < expert.ffn_dim:
                        # up reads A nibble, gate reads B nibble
                        if a_start + a < self.dim:
                            expert.up.weight[addr, a_start + a] = 8.0
                        if b_start + b < self.dim:
                            expert.gate.weight[addr, b_start + b] = 8.0

                        # Compute diff = A - B
                        diff = a - b
                        if diff < 0:
                            result_nibble = (diff + 16) & 0xF
                            borrow_out = True
                        else:
                            result_nibble = diff & 0xF
                            borrow_out = False

                        # Output result nibble (no incoming borrow)
                        if result_start + result_nibble < self.dim:
                            expert.down.weight[result_start + result_nibble, addr] = 5.0

                        # Output result WITH incoming borrow = (result - 1) & 0xF
                        result_with_borrow = (result_nibble - 1) & 0xF
                        borrow_result_start = borrow_result_dims[nibble_idx]
                        if borrow_result_start + result_with_borrow < self.dim:
                            expert.down.weight[borrow_result_start + result_with_borrow, addr] = 5.0

                        # Borrow out if diff < 0
                        if nibble_idx < len(borrow_out_dims):
                            borrow_dim = borrow_out_dims[nibble_idx]
                            if borrow_dim < self.dim and borrow_out:
                                expert.down.weight[borrow_dim, addr] = 5.0

                        # Borrow propagate: diff == 0 (a == b)
                        # If borrow arrives: 0 - 1 = -1 = 15 with borrow out
                        if nibble_idx < len(E.CARRY_PROP) and diff == 0:
                            prop_dim = E.CARRY_PROP[nibble_idx]
                            if prop_dim < self.dim:
                                expert.down.weight[prop_dim, addr] = 5.0

    def _bake_parallel_32bit_bitwise(self, expert: OpcodeExpert, op_fn):
        """Bake parallel 32-bit bitwise op - all 8 nibbles in one pass."""
        E = EmbedDims

        nibble_pairs = [
            (E.OP_A_NIBBLE_0_LO, E.OP_B_NIBBLE_0_LO, E.RESULT_0_LO),
            (E.OP_A_NIBBLE_0_HI, E.OP_B_NIBBLE_0_HI, E.RESULT_0_HI),
            (E.OP_A_NIBBLE_1_LO, E.OP_B_NIBBLE_1_LO, E.RESULT_1_LO),
            (E.OP_A_NIBBLE_1_HI, E.OP_B_NIBBLE_1_HI, E.RESULT_1_HI),
            (E.OP_A_NIBBLE_2_LO, E.OP_B_NIBBLE_2_LO, E.RESULT_2_LO),
            (E.OP_A_NIBBLE_2_HI, E.OP_B_NIBBLE_2_HI, E.RESULT_2_HI),
            (E.OP_A_NIBBLE_3_LO, E.OP_B_NIBBLE_3_LO, E.RESULT_3_LO),
            (E.OP_A_NIBBLE_3_HI, E.OP_B_NIBBLE_3_HI, E.RESULT_3_HI),
        ]

        for nibble_idx, (a_start, b_start, result_start) in enumerate(nibble_pairs):
            base_addr = nibble_idx * 256

            for a in range(16):
                for b in range(16):
                    addr = base_addr + a * 16 + b

                    if addr < expert.ffn_dim:
                        if a_start + a < self.dim:
                            expert.up.weight[addr, a_start + a] = 8.0
                        if b_start + b < self.dim:
                            expert.gate.weight[addr, b_start + b] = 8.0

                        result_nibble = op_fn(a, b) & 0xF
                        if result_start + result_nibble < self.dim:
                            expert.down.weight[result_start + result_nibble, addr] = 5.0

    def _bake_shift_left(self, expert: OpcodeExpert):
        """Bake left shift lookup table with byte output."""
        E = EmbedDims
        for a in range(16):
            for shift in range(16):
                addr = a * 16 + shift
                if addr < expert.ffn_dim:
                    # SwiGLU AND-gate: up reads value nibble, gate reads shift nibble
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + shift < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + shift] = 8.0
                    # Compute shifted result (byte output)
                    result = (a << shift) & 0xFF
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_shift_right(self, expert: OpcodeExpert):
        """Bake right shift lookup table with byte output."""
        E = EmbedDims
        for a in range(16):
            for shift in range(16):
                addr = a * 16 + shift
                if addr < expert.ffn_dim:
                    # SwiGLU AND-gate: up reads value nibble, gate reads shift nibble
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + shift < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + shift] = 8.0
                    # Compute shifted result (always fits in 0-15 for nibble >> any)
                    result = (a >> shift) & 0xFF
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_comparison(self, expert: OpcodeExpert, opcode: int):
        """
        Bake comparison operations (EQ, NE, LT, GT, LE, GE).

        Comparisons output 0 or 1. For nibble inputs (0-15),
        we bake all 256 comparisons.
        """
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b

                # Compute comparison result
                if opcode == Opcode.EQ:
                    result = 1 if a == b else 0
                elif opcode == Opcode.NE:
                    result = 1 if a != b else 0
                elif opcode == Opcode.LT:
                    result = 1 if a < b else 0
                elif opcode == Opcode.GT:
                    result = 1 if a > b else 0
                elif opcode == Opcode.LE:
                    result = 1 if a <= b else 0
                elif opcode == Opcode.GE:
                    result = 1 if a >= b else 0
                else:
                    result = 0

                if addr < expert.ffn_dim:
                    # SwiGLU AND-gate: up reads nibble A, gate reads nibble B
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + b < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0

                    # Output 0 or 1
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

        # Also bake parallel 32-bit comparison for large numbers
        self._bake_parallel_32bit_comparison(expert, opcode)
        # Also bake the final comparison result (pure neural)
        self._bake_comparison_result(expert, opcode)

    def _bake_parallel_32bit_comparison(self, expert: OpcodeExpert, opcode: int):
        """
        Bake parallel 32-bit comparison - output EQ and LT flags for each nibble.

        For each nibble pair:
        - CMP_EQ_*: 1 if A nibble == B nibble
        - CMP_LT_*: 1 if A nibble < B nibble

        The extraction code chains these to determine final comparison result:
        - EQ: All 8 EQ flags must be 1
        - LT: Find MSB where nibbles differ, check if A < B there
        - GT: Find MSB where nibbles differ, check if A > B there (use LT of B vs A)
        """
        E = EmbedDims

        nibble_pairs = [
            (E.OP_A_NIBBLE_0_LO, E.OP_B_NIBBLE_0_LO, E.CMP_EQ_0_LO, E.CMP_LT_0_LO),
            (E.OP_A_NIBBLE_0_HI, E.OP_B_NIBBLE_0_HI, E.CMP_EQ_0_HI, E.CMP_LT_0_HI),
            (E.OP_A_NIBBLE_1_LO, E.OP_B_NIBBLE_1_LO, E.CMP_EQ_1_LO, E.CMP_LT_1_LO),
            (E.OP_A_NIBBLE_1_HI, E.OP_B_NIBBLE_1_HI, E.CMP_EQ_1_HI, E.CMP_LT_1_HI),
            (E.OP_A_NIBBLE_2_LO, E.OP_B_NIBBLE_2_LO, E.CMP_EQ_2_LO, E.CMP_LT_2_LO),
            (E.OP_A_NIBBLE_2_HI, E.OP_B_NIBBLE_2_HI, E.CMP_EQ_2_HI, E.CMP_LT_2_HI),
            (E.OP_A_NIBBLE_3_LO, E.OP_B_NIBBLE_3_LO, E.CMP_EQ_3_LO, E.CMP_LT_3_LO),
            (E.OP_A_NIBBLE_3_HI, E.OP_B_NIBBLE_3_HI, E.CMP_EQ_3_HI, E.CMP_LT_3_HI),
        ]

        for nibble_idx, (a_start, b_start, eq_dim, lt_dim) in enumerate(nibble_pairs):
            base_addr = nibble_idx * 256

            for a in range(16):
                for b in range(16):
                    addr = base_addr + a * 16 + b

                    if addr < expert.ffn_dim:
                        # up reads A nibble, gate reads B nibble
                        if a_start + a < self.dim:
                            expert.up.weight[addr, a_start + a] = 8.0
                        if b_start + b < self.dim:
                            expert.gate.weight[addr, b_start + b] = 8.0

                        # Output EQ flag
                        if eq_dim < self.dim and a == b:
                            expert.down.weight[eq_dim, addr] = 5.0

                        # Output LT flag
                        if lt_dim < self.dim and a < b:
                            expert.down.weight[lt_dim, addr] = 5.0

                        # For nibble 7 (MSB), output sign flags
                        # A is negative if bit 31 is set, which means nibble 7 >= 8
                        if nibble_idx == 7:
                            if a >= 8 and E.CMP_A_NEG < self.dim:
                                expert.down.weight[E.CMP_A_NEG, addr] = 5.0
                            if b >= 8 and E.CMP_B_NEG < self.dim:
                                expert.down.weight[E.CMP_B_NEG, addr] = 5.0

    def _bake_comparison_result(self, expert: OpcodeExpert, opcode: int):
        """
        Bake the FINAL comparison result into the expert.

        Pure neural: reads per-nibble EQ and LT flags, outputs CMP_RESULT directly.

        For EQ/NE: Check if all nibbles are equal
        For LT/GT/LE/GE: Find MSB where nibbles differ, check LT flag there

        Strategy: Enumerate all 2^8 patterns of EQ flags (256 combinations).
        For each pattern, determine which nibble is the MSB that differs.
        Then output based on that nibble's LT flag.
        """
        E = EmbedDims

        eq_dims = [E.CMP_EQ_0_LO, E.CMP_EQ_0_HI, E.CMP_EQ_1_LO, E.CMP_EQ_1_HI,
                   E.CMP_EQ_2_LO, E.CMP_EQ_2_HI, E.CMP_EQ_3_LO, E.CMP_EQ_3_HI]
        lt_dims = [E.CMP_LT_0_LO, E.CMP_LT_0_HI, E.CMP_LT_1_LO, E.CMP_LT_1_HI,
                   E.CMP_LT_2_LO, E.CMP_LT_2_HI, E.CMP_LT_3_LO, E.CMP_LT_3_HI]

        # Use addresses starting after nibble computation (256*8 = 2048)
        base_addr = 2048

        # For EQ: Output 1 if ALL eq flags are set
        # We detect "all equal" by checking that eq_flags form the pattern [1,1,1,1,1,1,1,1]
        # Use a single address that fires only when all 8 EQ flags are high
        addr_all_eq = base_addr
        if addr_all_eq < expert.ffn_dim:
            # up: reads one EQ flag
            if eq_dims[7] < self.dim:
                expert.up.weight[addr_all_eq, eq_dims[7]] = 4.0
            # gate: reads all other EQ flags
            for i in range(7):
                if eq_dims[i] < self.dim:
                    expert.gate.weight[addr_all_eq, eq_dims[i]] = 0.5
            # Output to CMP_RESULT
            if opcode == Opcode.EQ:
                if E.CMP_RESULT < self.dim:
                    expert.down.weight[E.CMP_RESULT, addr_all_eq] = 8.0
            elif opcode == Opcode.NE:
                # For NE, we need to detect NOT all equal
                # This is harder... we need separate addresses
                pass

        # For NE: Output 1 if ANY nibble is not equal
        # Enumerate each nibble and output if that nibble's EQ is 0
        if opcode == Opcode.NE:
            for i in range(8):
                addr_ne = base_addr + 1 + i
                if addr_ne < expert.ffn_dim:
                    # Output 1 if this nibble is not equal
                    # We need to detect EQ[i]=0. Since EQ is ~20000 when equal, ~0 when not,
                    # we can use negative weight: -EQ[i] will be high when EQ[i] is low
                    # But that doesn't work with SwiGLU...
                    #
                    # Alternative: For NE, we output to CMP_RESULT if LT[i] is high
                    # (LT being high implies not equal)
                    if lt_dims[i] < self.dim:
                        expert.up.weight[addr_ne, lt_dims[i]] = 4.0
                        expert.gate.weight[addr_ne, lt_dims[i]] = 4.0
                    if E.CMP_RESULT < self.dim:
                        expert.down.weight[E.CMP_RESULT, addr_ne] = 3.0

                # Also check A > B at this nibble (neither LT nor EQ)
                # This requires detecting NOT LT and NOT EQ, which is hard...
                # For now, just check LT (NE passes if any nibble has A!=B)
                # We miss the A>B case but cover A<B. User can swap operands.

        # For LT: Find MSB where not equal, check LT there
        # The MSB has priority, so nibble 7 > nibble 6 > ... > nibble 0
        # For each nibble i, output 1 if:
        #   - All nibbles j > i are equal (EQ[j]=1 for j>i)
        #   - This nibble is not equal (EQ[i]=0)
        #   - A < B at this nibble (LT[i]=1)
        if opcode in {Opcode.LT, Opcode.LE}:
            for i in range(7, -1, -1):  # MSB first
                addr_lt = base_addr + 10 + (7 - i) * 2  # Unique address per nibble
                if addr_lt < expert.ffn_dim and lt_dims[i] < self.dim:
                    # up: reads LT[i] (this nibble is A < B)
                    expert.up.weight[addr_lt, lt_dims[i]] = 8.0
                    # gate: reads all EQ[j] for j > i (all higher nibbles equal)
                    # Actually, this is complex... let's simplify
                    # Just use LT[i] with high weight for high nibbles
                    weight = 8.0 - (7 - i) * 0.5  # Higher nibbles get more weight
                    expert.gate.weight[addr_lt, lt_dims[i]] = weight
                    if E.CMP_RESULT < self.dim:
                        expert.down.weight[E.CMP_RESULT, addr_lt] = weight

        # For GT: Similar but check NOT LT at the MSB differing nibble
        # This is complex... for now, implement GT as "not LT and not EQ"
        if opcode in {Opcode.GT, Opcode.GE}:
            # GT: A > B at MSB differing nibble
            # Since we output LT flags but not GT flags, we need to detect:
            # EQ[i]=0 AND LT[i]=0 (meaning A > B at nibble i)
            # This requires detecting "not LT" which is hard with SwiGLU...
            #
            # Alternative: Check that LT is low by using negative weight
            # But SwiGLU output can't go negative reliably...
            #
            # For now, just use the fact that for unsigned comparison:
            # A > B iff A >= B and A != B iff A >= (B+1)
            # Not directly implementable without arithmetic...
            #
            # Actually, we CAN detect GT: at each nibble, if A > B, then
            # LT[i] = 0 (not less) AND the specific A nibble is set AND B nibble is set
            # We'd need to re-read the input nibbles and check A > B directly.
            #
            # Simplest: For GT, swap operands and use LT logic externally.
            # For now, leave GT unimplemented in pure neural.
            pass

        # For LE: EQ OR LT - just combine the above
        if opcode == Opcode.LE:
            # Already handled by LT logic above, plus EQ logic
            pass

        # For GE: EQ OR GT
        if opcode == Opcode.GE:
            pass

    def _bake_carry_chain_layer(self, moe: OpcodeMoE):
        """
        Bake layer 3 FFN for each opcode type.

        Different opcodes need different layer 3 logic:
        - Bitwise (AND, OR, XOR): Copy RESULT → OUTPUT_BYTE directly
        - ADD: Handle carry chain
        - SUB: Handle borrow chain
        - Comparisons: Output 0 or 1 to OUTPUT_BYTE
        - Others: Copy RESULT → OUTPUT_BYTE
        """
        E = EmbedDims

        # For each expert, zero weights first then bake appropriate logic
        for idx, expert in enumerate(moe.experts):
            opcode = Opcode.ALL[idx]

            # Zero weights before baking
            expert.up.weight.zero_()
            expert.gate.weight.zero_()
            expert.down.weight.zero_()

            if opcode == Opcode.ADD:
                self._bake_layer3_add(expert)
            elif opcode == Opcode.SUB:
                self._bake_layer3_sub(expert)
            elif opcode in {Opcode.AND, Opcode.OR, Opcode.XOR}:
                self._bake_layer3_bitwise(expert)
            elif opcode in {Opcode.EQ, Opcode.NE, Opcode.LT, Opcode.GT, Opcode.LE, Opcode.GE}:
                self._bake_layer3_comparison(expert, opcode)
            elif opcode in {Opcode.SHL, Opcode.SHR}:
                self._bake_layer3_shift(expert)
            else:
                # Default: copy RESULT → OUTPUT_BYTE
                self._bake_layer3_bitwise(expert)

    def _bake_layer3_bitwise(self, expert: OpcodeExpert):
        """
        Layer 3 for bitwise ops: Simply copy RESULT → OUTPUT_BYTE.
        No carry chain needed.
        """
        E = EmbedDims

        result_dims = [E.RESULT_0_LO, E.RESULT_0_HI, E.RESULT_1_LO, E.RESULT_1_HI,
                       E.RESULT_2_LO, E.RESULT_2_HI, E.RESULT_3_LO, E.RESULT_3_HI]
        output_dims = [E.OUTPUT_BYTE + i*16 for i in range(8)]

        base_addr = 0
        for nibble_idx in range(8):
            for val in range(16):
                addr = base_addr + val
                if addr < expert.ffn_dim:
                    result_dim = result_dims[nibble_idx] + val
                    output_dim = output_dims[nibble_idx] + val
                    if result_dim < self.dim and output_dim < self.dim:
                        expert.up.weight[addr, result_dim] = 10.0
                        expert.gate.weight[addr, result_dim] = 10.0
                        expert.down.weight[output_dim, addr] = 5.0
            base_addr += 16

    def _bake_layer3_shift(self, expert: OpcodeExpert):
        """
        Layer 3 for shift ops: Copy from BYTES_START to OUTPUT_BYTE.
        Shift ops output to BYTES_START, so convert to nibbles.
        """
        E = EmbedDims
        # Shifts output to BYTES_START as byte. Convert to nibble pairs.
        # Read byte value from BYTES_START, output low/high nibbles to OUTPUT_BYTE
        for byte_val in range(256):
            addr = byte_val
            if addr < expert.ffn_dim:
                lo_nib = byte_val & 0xF
                hi_nib = (byte_val >> 4) & 0xF
                if E.BYTES_START + byte_val < self.dim:
                    expert.up.weight[addr, E.BYTES_START + byte_val] = 10.0
                    expert.gate.weight[addr, E.BYTES_START + byte_val] = 10.0
                    if E.OUTPUT_BYTE + lo_nib < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + lo_nib, addr] = 5.0
                    if E.OUTPUT_BYTE + 16 + hi_nib < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + 16 + hi_nib, addr] = 5.0
        # Set remaining nibbles to 0
        for nibble_idx in range(2, 8):
            addr = 256 + nibble_idx
            if addr < expert.ffn_dim:
                output_dim = E.OUTPUT_BYTE + nibble_idx * 16  # value 0
                if output_dim < self.dim:
                    expert.up.weight[addr, E.OPCODE_START + Opcode.SHL] = 5.0  # Always fires
                    expert.gate.weight[addr, E.OPCODE_START + Opcode.SHL] = 5.0
                    expert.down.weight[output_dim, addr] = 3.0

    def _bake_layer3_comparison(self, expert: OpcodeExpert, opcode: int):
        """
        Layer 3 for comparisons: Output 0 or 1 based on per-nibble flags.

        Read EQ/LT flags from layer 2, compute final result:
        - EQ: 1 if ALL nibbles equal
        - NE: 1 if ANY nibble not equal
        - LT: 1 if A < B at MSB differing nibble
        - GT: 1 if A > B at MSB differing nibble
        - LE: 1 if EQ or LT
        - GE: 1 if EQ or GT
        """
        E = EmbedDims

        eq_dims = [E.CMP_EQ_0_LO, E.CMP_EQ_0_HI, E.CMP_EQ_1_LO, E.CMP_EQ_1_HI,
                   E.CMP_EQ_2_LO, E.CMP_EQ_2_HI, E.CMP_EQ_3_LO, E.CMP_EQ_3_HI]
        lt_dims = [E.CMP_LT_0_LO, E.CMP_LT_0_HI, E.CMP_LT_1_LO, E.CMP_LT_1_HI,
                   E.CMP_LT_2_LO, E.CMP_LT_2_HI, E.CMP_LT_3_LO, E.CMP_LT_3_HI]

        # Output dims
        output_dims = [E.OUTPUT_BYTE + i*16 for i in range(8)]

        base_addr = 0

        # Output 0 to nibbles 1-7 (always zero for comparison result)
        for nibble_idx in range(1, 8):
            addr = base_addr + nibble_idx - 1
            output_dim = output_dims[nibble_idx]  # First dim of this nibble = value 0
            if addr < expert.ffn_dim and output_dim < self.dim:
                expert.up.weight[addr, E.OPCODE_START + opcode] = 10.0
                expert.gate.weight[addr, E.OPCODE_START + opcode] = 10.0
                expert.down.weight[output_dim, addr] = 10.0  # Output value 0
        base_addr += 7

        # Now handle the comparison result in nibble 0
        out_0 = output_dims[0]  # First nibble

        if opcode == Opcode.EQ:
            # EQ: Output 1 only if ALL 8 eq flags are high
            #
            # Strategy: Use OPCODE signal as negative threshold.
            # EQ flags are ~20480 when set, OPCODE is 20.0
            # up = sum(EQ × 0.001) - OPCODE × 7.5
            # When all 8: 163.84 - 150 = 13.84 > 0 (fires)
            # When 7:     143.36 - 150 = -6.64 < 0 (doesn't fire)
            # When 5:     102.4 - 150 = -47.6 < 0 (doesn't fire)

            # Output 1 when ALL equal (threshold-based)
            # Activation when all 8 EQ: silu(13.84) × 13.84 ≈ 192
            addr = base_addr
            if addr < expert.ffn_dim:
                opcode_dim = E.OPCODE_START + opcode
                for i in range(8):
                    if eq_dims[i] < self.dim:
                        expert.up.weight[addr, eq_dims[i]] = 0.001
                        expert.gate.weight[addr, eq_dims[i]] = 0.001
                # Negative threshold using opcode signal
                expert.up.weight[addr, opcode_dim] = -7.5
                expert.gate.weight[addr, opcode_dim] = -7.5
                expert.down.weight[out_0 + 1, addr] = 30000.0  # Very strong output 1
            base_addr += 1

            # Output 0 when NOT all equal (default - always fires)
            # Activation: silu(200) × 200 ≈ 40000
            addr = base_addr
            if addr < expert.ffn_dim:
                expert.up.weight[addr, E.OPCODE_START + opcode] = 10.0
                expert.gate.weight[addr, E.OPCODE_START + opcode] = 10.0
                expert.down.weight[out_0, addr] = 10.0  # Weak output 0 (400K)
            base_addr += 1

        elif opcode == Opcode.NE:
            # NE: Output 1 if NOT all equal (inverse of EQ)
            # Use threshold approach: gate is positive when < 8 EQ flags
            # gate = OPCODE × 7.5 - sum(EQ × 0.001)
            # When all 8: 150 - 163.84 = -13.84 < 0 (doesn't fire)
            # When 5:     150 - 102.4 = 47.6 > 0 (fires)

            # Output 1 when NOT all equal
            addr = base_addr
            if addr < expert.ffn_dim:
                opcode_dim = E.OPCODE_START + opcode
                expert.up.weight[addr, opcode_dim] = 10.0
                # gate: positive on opcode, negative on EQ flags
                expert.gate.weight[addr, opcode_dim] = 7.5
                for i in range(8):
                    if eq_dims[i] < self.dim:
                        expert.gate.weight[addr, eq_dims[i]] = -0.001
                expert.down.weight[out_0 + 1, addr] = 500.0  # Output 1
            base_addr += 1

            # Output 0 when ALL equal (threshold exceeded)
            addr = base_addr
            if addr < expert.ffn_dim:
                opcode_dim = E.OPCODE_START + opcode
                for i in range(8):
                    if eq_dims[i] < self.dim:
                        expert.up.weight[addr, eq_dims[i]] = 0.001
                        expert.gate.weight[addr, eq_dims[i]] = 0.001
                expert.up.weight[addr, opcode_dim] = -7.5
                expert.gate.weight[addr, opcode_dim] = -7.5
                expert.down.weight[out_0, addr] = 30000.0  # Strong output 0
            base_addr += 1

        elif opcode in {Opcode.LT, Opcode.LE}:
            # LT: Output 1 if A < B at the MSB differing nibble
            #
            # For SIGNED comparison: if A_NEG=1 and B_NEG=0, then A < B unconditionally
            # (negative < positive regardless of magnitude)
            #
            # For each nibble i from MSB (7) to LSB (0):
            #   "Output 1 if LT[i] AND all higher EQ flags are high"
            #
            # Use threshold on gate: fires only when ALL higher EQ flags present
            # EQ flags are ~20480 when set. threshold = (#flags - 0.5) per flag
            opcode_dim = E.OPCODE_START + opcode

            # Signed LT: Output 1 when A_NEG=1 AND B_NEG=0
            # up = A_NEG (fires when A is negative, ~20480)
            # gate = OPCODE × threshold - B_NEG × tiny (fires when B NOT negative)
            # With threshold=5, tiny=0.01: when B positive, gate=100; when B negative, gate=-104.8
            addr = base_addr
            if addr < expert.ffn_dim and E.CMP_A_NEG < self.dim and E.CMP_B_NEG < self.dim:
                expert.up.weight[addr, E.CMP_A_NEG] = 1.0
                expert.gate.weight[addr, opcode_dim] = 5.0
                expert.gate.weight[addr, E.CMP_B_NEG] = -0.01
                expert.down.weight[out_0 + 1, addr] = 50000.0  # Very strong to override unsigned
            base_addr += 1

            for i in range(7, -1, -1):
                addr = base_addr + (7 - i)
                num_higher = 7 - i  # Number of higher nibbles that must be EQ
                if addr < expert.ffn_dim and lt_dims[i] < self.dim:
                    # up reads LT[i]
                    expert.up.weight[addr, lt_dims[i]] = 10.0
                    if num_higher == 0:
                        # MSB: no higher nibbles, just check LT[7]
                        expert.gate.weight[addr, lt_dims[i]] = 10.0
                    else:
                        # Gate requires all higher EQ flags
                        # gate = sum(EQ[j>i] × 0.001) - OPCODE × threshold
                        # threshold = (num_higher - 0.5) × 20480 × 0.001 / 20
                        #           = (num_higher - 0.5) × 1.024
                        threshold = (num_higher - 0.5) * 1.024
                        for j in range(i + 1, 8):
                            if eq_dims[j] < self.dim:
                                expert.gate.weight[addr, eq_dims[j]] = 0.001
                        expert.gate.weight[addr, opcode_dim] = -threshold
                    expert.down.weight[out_0 + 1, addr] = 500.0
            base_addr += 8

            # For LE, also output 1 if ALL equal (using EQ threshold)
            if opcode == Opcode.LE:
                addr = base_addr
                if addr < expert.ffn_dim:
                    for i in range(8):
                        if eq_dims[i] < self.dim:
                            expert.up.weight[addr, eq_dims[i]] = 0.001
                            expert.gate.weight[addr, eq_dims[i]] = 0.001
                    expert.up.weight[addr, opcode_dim] = -7.5
                    expert.gate.weight[addr, opcode_dim] = -7.5
                    expert.down.weight[out_0 + 1, addr] = 30000.0
                base_addr += 1

            # Output 0 as default (strong enough to win when no LT case fires)
            addr = base_addr
            if addr < expert.ffn_dim:
                expert.up.weight[addr, opcode_dim] = 10.0
                expert.gate.weight[addr, opcode_dim] = 10.0
                expert.down.weight[out_0, addr] = 10.0  # 400K activation
            base_addr += 1

        elif opcode in {Opcode.GT, Opcode.GE}:
            # GT: Output 1 if A > B, i.e., NOT(LT) AND NOT(EQ_all)
            #
            # For SIGNED comparison: if A_NEG=0 and B_NEG=1, then A > B unconditionally
            # (positive > negative regardless of magnitude)
            #
            # Strategy: Invert the logic
            # - "Output 0 if LT" - fires when A < B at MSB differing nibble
            # - "Output 0 if all EQ" - fires when all 8 EQ flags high
            # - "Output 1 default" - wins when neither LT nor EQ_all condition met
            opcode_dim = E.OPCODE_START + opcode

            # Signed GT: Output 1 when A_NEG=0 AND B_NEG=1
            # up = B_NEG (fires when B is negative, ~20480)
            # gate = OPCODE × threshold - A_NEG × tiny (fires when A NOT negative)
            # Need extremely high weight to override all 8 unsigned LT cases
            # (which all fire when comparing 0 vs -1, producing ~2e+13 to output 0)
            addr = base_addr
            if addr < expert.ffn_dim and E.CMP_A_NEG < self.dim and E.CMP_B_NEG < self.dim:
                expert.up.weight[addr, E.CMP_B_NEG] = 1.0
                expert.gate.weight[addr, opcode_dim] = 5.0
                expert.gate.weight[addr, E.CMP_A_NEG] = -0.01
                expert.down.weight[out_0 + 1, addr] = 50000000.0  # Must override ~2e+13
            base_addr += 1

            # Output 0 when LT condition is true (using same threshold logic as LT)
            for i in range(7, -1, -1):
                addr = base_addr + (7 - i)
                num_higher = 7 - i
                if addr < expert.ffn_dim and lt_dims[i] < self.dim:
                    # up reads LT[i]
                    expert.up.weight[addr, lt_dims[i]] = 10.0
                    if num_higher == 0:
                        # MSB: no higher nibbles, just check LT[7]
                        expert.gate.weight[addr, lt_dims[i]] = 10.0
                    else:
                        # Gate requires all higher EQ flags
                        threshold = (num_higher - 0.5) * 1.024
                        for j in range(i + 1, 8):
                            if eq_dims[j] < self.dim:
                                expert.gate.weight[addr, eq_dims[j]] = 0.001
                        expert.gate.weight[addr, opcode_dim] = -threshold
                    expert.down.weight[out_0, addr] = 500.0  # Output 0
            base_addr += 8

            # Output 0 when all EQ (using threshold)
            addr = base_addr
            if addr < expert.ffn_dim:
                for i in range(8):
                    if eq_dims[i] < self.dim:
                        expert.up.weight[addr, eq_dims[i]] = 0.001
                        expert.gate.weight[addr, eq_dims[i]] = 0.001
                expert.up.weight[addr, opcode_dim] = -7.5
                expert.gate.weight[addr, opcode_dim] = -7.5
                expert.down.weight[out_0, addr] = 30000.0  # Strong output 0
            base_addr += 1

            # For GE, output 1 (instead of 0) when all EQ
            if opcode == Opcode.GE:
                addr = base_addr
                if addr < expert.ffn_dim:
                    for i in range(8):
                        if eq_dims[i] < self.dim:
                            expert.up.weight[addr, eq_dims[i]] = 0.001
                            expert.gate.weight[addr, eq_dims[i]] = 0.001
                    expert.up.weight[addr, opcode_dim] = -7.5
                    expert.gate.weight[addr, opcode_dim] = -7.5
                    expert.down.weight[out_0 + 1, addr] = 35000.0  # Override the 0
                base_addr += 1

            # Output 1 as default (wins when neither LT nor EQ_all fires)
            addr = base_addr
            if addr < expert.ffn_dim:
                expert.up.weight[addr, opcode_dim] = 10.0
                expert.gate.weight[addr, opcode_dim] = 10.0
                expert.down.weight[out_0 + 1, addr] = 10.0  # 400K activation
            base_addr += 1

    def _bake_layer3_add(self, expert: OpcodeExpert):
        """
        Layer 3 for ADD: Handle carry chain with propagation.

        Carry arrives at nibble i if:
        - C[i] = G[i-1] OR (P[i-1] AND G[i-2]) OR (P[i-1] AND P[i-2] AND G[i-3]) OR ...

        Where:
        - G[j] = CARRY_OUT[j] (sum > 15, generates carry)
        - P[j] = CARRY_PROP[j] (sum == 15, propagates carry)
        """
        E = EmbedDims

        result_dims = [E.RESULT_0_LO, E.RESULT_0_HI, E.RESULT_1_LO, E.RESULT_1_HI,
                       E.RESULT_2_LO, E.RESULT_2_HI, E.RESULT_3_LO, E.RESULT_3_HI]
        carry_result_dims = [E.RESULT_CARRY_0_LO, E.RESULT_CARRY_0_HI,
                             E.RESULT_CARRY_1_LO, E.RESULT_CARRY_1_HI,
                             E.RESULT_CARRY_2_LO, E.RESULT_CARRY_2_HI,
                             E.RESULT_CARRY_3_LO, E.RESULT_CARRY_3_HI]
        gen_dims = [E.CARRY_OUT_0_LO, E.CARRY_OUT_0_HI, E.CARRY_OUT_1_LO,
                    E.CARRY_OUT_1_HI, E.CARRY_OUT_2_LO, E.CARRY_OUT_2_HI,
                    E.CARRY_OUT_3_LO, E.CARRY_OUT_3_HI]
        prop_dims = E.CARRY_PROP
        output_dims = [E.OUTPUT_BYTE + i*16 for i in range(8)]

        base_addr = 0

        # Nibble 0: Always use RESULT (no carry in)
        for val in range(16):
            addr = base_addr + val
            if addr < expert.ffn_dim:
                result_dim = result_dims[0] + val
                output_dim = output_dims[0] + val
                if result_dim < self.dim and output_dim < self.dim:
                    expert.up.weight[addr, result_dim] = 10.0
                    expert.gate.weight[addr, result_dim] = 10.0
                    expert.down.weight[output_dim, addr] = 5.0
        base_addr += 16

        # Nibbles 1-7: Carry chain selection
        for nibble_idx in range(1, 8):
            result_start = result_dims[nibble_idx]
            carry_start = carry_result_dims[nibble_idx]
            output_start = output_dims[nibble_idx]

            # Carry can arrive from multiple sources. For each source j < i:
            # C[i] includes term: G[j] AND P[j+1] AND P[j+2] AND ... AND P[i-1]

            # For simplicity, handle the most common cases:
            # 1. Direct: G[i-1]
            # 2. One-hop propagation: G[i-2] AND P[i-1]
            # 3. Two-hop propagation: G[i-3] AND P[i-2] AND P[i-1]
            # etc.

            # CASE 1: Direct carry from G[i-1]
            DIRECT_DOWN = 50000000.0  # Same strength as hop cases
            for val in range(16):
                addr = base_addr + val
                if addr < expert.ffn_dim:
                    carry_dim = carry_start + val
                    output_dim = output_start + val
                    gen_dim = gen_dims[nibble_idx - 1]
                    if carry_dim < self.dim and output_dim < self.dim and gen_dim < self.dim:
                        expert.up.weight[addr, carry_dim] = 10.0
                        expert.gate.weight[addr, gen_dim] = 10.0
                        expert.down.weight[output_dim, addr] = DIRECT_DOWN
            base_addr += 16

            # CASE 2: One-hop propagated carry: G[i-2] AND P[i-1]
            # Problem: When G[i-2] is absent but P[i-1] is present, gate is slightly
            # negative, but silu(up)*gate is still large and negative, causing wrong output.
            # Fix: Make up also conditional on G flag. When G absent, up is very negative,
            # so silu(up) ≈ 0 and the case doesn't contribute.
            HOP_DOWN = 50000000.0  # 50M - must dominate no-carry case
            G_PENALTY = 15000.0  # Makes up negative when G absent
            G_SCALE = G_PENALTY / 1024.0  # Compensates penalty when G present
            if nibble_idx >= 2:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        carry_dim = carry_start + val
                        output_dim = output_start + val
                        gen_dim = gen_dims[nibble_idx - 2]
                        p_dim = prop_dims[nibble_idx - 1]
                        if carry_dim < self.dim and output_dim < self.dim and gen_dim < self.dim and p_dim < self.dim:
                            # up: positive when G present, very negative when G absent
                            expert.up.weight[addr, carry_dim] = 10.0
                            expert.up.weight[addr, gen_dim] = G_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.ADD] = -G_PENALTY
                            # gate: requires P flag
                            expert.gate.weight[addr, p_dim] = 10.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 3: Two-hop: G[i-3] AND P[i-2] AND P[i-1]
            if nibble_idx >= 3:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        carry_dim = carry_start + val
                        output_dim = output_start + val
                        gen_dim = gen_dims[nibble_idx - 3]
                        p_dim1 = prop_dims[nibble_idx - 2]
                        p_dim2 = prop_dims[nibble_idx - 1]
                        if carry_dim < self.dim and output_dim < self.dim:
                            # up: positive when G present, very negative when G absent
                            expert.up.weight[addr, carry_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = G_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.ADD] = -G_PENALTY
                            # gate: requires both P flags (threshold gating)
                            # P flags have value ~20480. Use weight 0.01 so each P contributes ~205
                            # Threshold = 1.5 * 205 = 307.5, so need 2 P flags (410) to exceed
                            if p_dim1 < self.dim:
                                expert.gate.weight[addr, p_dim1] = 0.01
                            if p_dim2 < self.dim:
                                expert.gate.weight[addr, p_dim2] = 0.01
                            expert.gate.weight[addr, E.OPCODE_START + Opcode.ADD] = -15.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 4: Three-hop: G[i-4] AND P[i-3] AND P[i-2] AND P[i-1]
            if nibble_idx >= 4:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        carry_dim = carry_start + val
                        output_dim = output_start + val
                        gen_dim = gen_dims[nibble_idx - 4]
                        p_dims_hop = [prop_dims[nibble_idx - 3], prop_dims[nibble_idx - 2], prop_dims[nibble_idx - 1]]
                        if carry_dim < self.dim and output_dim < self.dim:
                            # up: positive when G present, very negative when G absent
                            expert.up.weight[addr, carry_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = G_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.ADD] = -G_PENALTY
                            # gate: requires all 3 P flags (threshold gating)
                            # Threshold = 2.5 * 205 = 512.5, need 3 flags (615) to exceed
                            for p_d in p_dims_hop:
                                if p_d < self.dim:
                                    expert.gate.weight[addr, p_d] = 0.01
                            expert.gate.weight[addr, E.OPCODE_START + Opcode.ADD] = -25.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 5: Four-hop: G[i-5] AND P[i-4] AND P[i-3] AND P[i-2] AND P[i-1]
            if nibble_idx >= 5:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        carry_dim = carry_start + val
                        output_dim = output_start + val
                        gen_dim = gen_dims[nibble_idx - 5]
                        p_dims_hop = [prop_dims[i] for i in range(nibble_idx - 4, nibble_idx)]
                        if carry_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, carry_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = G_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.ADD] = -G_PENALTY
                            # Threshold = 3.5 * 205 = 717.5, need 4 flags (820) to exceed
                            for p_d in p_dims_hop:
                                if p_d < self.dim:
                                    expert.gate.weight[addr, p_d] = 0.01
                            expert.gate.weight[addr, E.OPCODE_START + Opcode.ADD] = -35.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 6: Five-hop: G[i-6] AND P[i-5]...P[i-1]
            if nibble_idx >= 6:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        carry_dim = carry_start + val
                        output_dim = output_start + val
                        gen_dim = gen_dims[nibble_idx - 6]
                        p_dims_hop = [prop_dims[i] for i in range(nibble_idx - 5, nibble_idx)]
                        if carry_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, carry_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = G_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.ADD] = -G_PENALTY
                            # Threshold = 4.5 * 205 = 922.5, need 5 flags (1025) to exceed
                            for p_d in p_dims_hop:
                                if p_d < self.dim:
                                    expert.gate.weight[addr, p_d] = 0.01
                            expert.gate.weight[addr, E.OPCODE_START + Opcode.ADD] = -45.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 7: Six-hop: G[i-7] AND P[i-6]...P[i-1]
            if nibble_idx >= 7:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        carry_dim = carry_start + val
                        output_dim = output_start + val
                        gen_dim = gen_dims[nibble_idx - 7]
                        p_dims_hop = [prop_dims[i] for i in range(nibble_idx - 6, nibble_idx)]
                        if carry_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, carry_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = G_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.ADD] = -G_PENALTY
                            # Threshold = 5.5 * 205 = 1127.5, need 6 flags (1230) to exceed
                            for p_d in p_dims_hop:
                                if p_d < self.dim:
                                    expert.gate.weight[addr, p_d] = 0.01
                            expert.gate.weight[addr, E.OPCODE_START + Opcode.ADD] = -55.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE FINAL: No carry - use RESULT
            # When no carry arrives, use RESULT directly
            # This case fires when G[i-1] is 0 (no direct carry from previous nibble)
            for val in range(16):
                addr = base_addr + val
                if addr < expert.ffn_dim:
                    result_dim = result_start + val
                    output_dim = output_start + val
                    if result_dim < self.dim and output_dim < self.dim:
                        expert.up.weight[addr, result_dim] = 10.0
                        expert.gate.weight[addr, result_dim] = 10.0
                        # Suppress when direct carry from G[i-1]
                        if nibble_idx >= 1 and gen_dims[nibble_idx - 1] < self.dim:
                            expert.gate.weight[addr, gen_dims[nibble_idx - 1]] = -15.0
                        # Use moderate down weight - carry cases with 50M will override
                        expert.down.weight[output_dim, addr] = 5000.0
            base_addr += 16

    def _bake_layer3_sub(self, expert: OpcodeExpert):
        """
        Layer 3 for SUB: Handle borrow chain with multi-hop propagation.

        Borrow logic:
        - Borrow Generate (B): a < b at nibble i
        - Borrow Propagate (BP): a == b at nibble i (if borrow arrives, it goes out)
        - Borrow arrives at nibble i via: B[i-1] OR (BP[i-1] AND B[i-2]) OR ...

        RESULT_CARRY = result with borrow applied = (result - 1) & 0xF
        """
        E = EmbedDims

        result_dims = [E.RESULT_0_LO, E.RESULT_0_HI, E.RESULT_1_LO, E.RESULT_1_HI,
                       E.RESULT_2_LO, E.RESULT_2_HI, E.RESULT_3_LO, E.RESULT_3_HI]
        borrow_result_dims = [E.RESULT_CARRY_0_LO, E.RESULT_CARRY_0_HI,
                              E.RESULT_CARRY_1_LO, E.RESULT_CARRY_1_HI,
                              E.RESULT_CARRY_2_LO, E.RESULT_CARRY_2_HI,
                              E.RESULT_CARRY_3_LO, E.RESULT_CARRY_3_HI]
        # B[i] = borrow generated at nibble i (a < b)
        borrow_gen_dims = [E.CARRY_OUT_0_LO, E.CARRY_OUT_0_HI, E.CARRY_OUT_1_LO,
                           E.CARRY_OUT_1_HI, E.CARRY_OUT_2_LO, E.CARRY_OUT_2_HI,
                           E.CARRY_OUT_3_LO, E.CARRY_OUT_3_HI]
        # BP[i] = borrow propagated at nibble i (a == b)
        borrow_prop_dims = E.CARRY_PROP
        output_dims = [E.OUTPUT_BYTE + i*16 for i in range(8)]

        base_addr = 0

        # Constants for conditional up weights (same as ADD)
        HOP_DOWN = 50000000.0
        B_PENALTY = 15000.0
        B_SCALE = B_PENALTY / 1024.0

        # Nibble 0: Always use RESULT (no borrow in)
        for val in range(16):
            addr = base_addr + val
            if addr < expert.ffn_dim:
                result_dim = result_dims[0] + val
                output_dim = output_dims[0] + val
                if result_dim < self.dim and output_dim < self.dim:
                    expert.up.weight[addr, result_dim] = 10.0
                    expert.gate.weight[addr, result_dim] = 10.0
                    expert.down.weight[output_dim, addr] = 5.0
        base_addr += 16

        # Nibbles 1-7: Borrow chain selection
        for nibble_idx in range(1, 8):
            result_start = result_dims[nibble_idx]
            borrow_start = borrow_result_dims[nibble_idx]
            output_start = output_dims[nibble_idx]

            # CASE 1: Direct borrow from B[i-1]
            DIRECT_DOWN = 50000000.0
            for val in range(16):
                addr = base_addr + val
                if addr < expert.ffn_dim:
                    borrow_dim = borrow_start + val
                    output_dim = output_start + val
                    gen_dim = borrow_gen_dims[nibble_idx - 1]
                    if borrow_dim < self.dim and output_dim < self.dim and gen_dim < self.dim:
                        expert.up.weight[addr, borrow_dim] = 10.0
                        expert.gate.weight[addr, gen_dim] = 10.0
                        expert.down.weight[output_dim, addr] = DIRECT_DOWN
            base_addr += 16

            # CASE 2: One-hop borrow: B[i-2] AND BP[i-1]
            if nibble_idx >= 2:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        borrow_dim = borrow_start + val
                        output_dim = output_start + val
                        gen_dim = borrow_gen_dims[nibble_idx - 2]
                        bp_dim = borrow_prop_dims[nibble_idx - 1]
                        if borrow_dim < self.dim and output_dim < self.dim and gen_dim < self.dim and bp_dim < self.dim:
                            expert.up.weight[addr, borrow_dim] = 10.0
                            expert.up.weight[addr, gen_dim] = B_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.SUB] = -B_PENALTY
                            expert.gate.weight[addr, bp_dim] = 10.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 3: Two-hop: B[i-3] AND BP[i-2] AND BP[i-1]
            # Condition up on B[i-3] AND first BP (BP[i-2]) to suppress when either absent
            if nibble_idx >= 3:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        borrow_dim = borrow_start + val
                        output_dim = output_start + val
                        gen_dim = borrow_gen_dims[nibble_idx - 3]
                        bp_dims = [borrow_prop_dims[nibble_idx - 2], borrow_prop_dims[nibble_idx - 1]]
                        if borrow_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, borrow_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = B_SCALE
                            # Also condition on first BP flag
                            if bp_dims[0] < self.dim:
                                expert.up.weight[addr, bp_dims[0]] = B_SCALE
                            # Double penalty since we need both B and BP
                            expert.up.weight[addr, E.OPCODE_START + Opcode.SUB] = -B_PENALTY * 2
                            # gate checks the second BP flag
                            if bp_dims[1] < self.dim:
                                expert.gate.weight[addr, bp_dims[1]] = 10.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 4: Three-hop: B[i-4] AND BP[i-3] AND BP[i-2] AND BP[i-1]
            # Condition up on B[i-4] AND first two BP flags
            if nibble_idx >= 4:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        borrow_dim = borrow_start + val
                        output_dim = output_start + val
                        gen_dim = borrow_gen_dims[nibble_idx - 4]
                        bp_dims = [borrow_prop_dims[i] for i in range(nibble_idx - 3, nibble_idx)]
                        if borrow_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, borrow_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = B_SCALE
                            # Condition on first two BP flags
                            for bp_d in bp_dims[:2]:
                                if bp_d < self.dim:
                                    expert.up.weight[addr, bp_d] = B_SCALE
                            # Triple penalty since we need B and 2 BP flags
                            expert.up.weight[addr, E.OPCODE_START + Opcode.SUB] = -B_PENALTY * 3
                            # gate checks the last BP flag
                            if bp_dims[-1] < self.dim:
                                expert.gate.weight[addr, bp_dims[-1]] = 10.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 5: Four-hop: B[i-5] AND BP[i-4]...BP[i-1]
            # Condition up on B and first 3 BP flags
            if nibble_idx >= 5:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        borrow_dim = borrow_start + val
                        output_dim = output_start + val
                        gen_dim = borrow_gen_dims[nibble_idx - 5]
                        bp_dims = [borrow_prop_dims[i] for i in range(nibble_idx - 4, nibble_idx)]
                        if borrow_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, borrow_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = B_SCALE
                            # Condition on first 3 BP flags
                            for bp_d in bp_dims[:3]:
                                if bp_d < self.dim:
                                    expert.up.weight[addr, bp_d] = B_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.SUB] = -B_PENALTY * 4
                            # gate checks the last BP flag
                            if bp_dims[-1] < self.dim:
                                expert.gate.weight[addr, bp_dims[-1]] = 10.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 6: Five-hop: B[i-6] AND BP[i-5]...BP[i-1]
            # Condition up on B and first 4 BP flags
            if nibble_idx >= 6:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        borrow_dim = borrow_start + val
                        output_dim = output_start + val
                        gen_dim = borrow_gen_dims[nibble_idx - 6]
                        bp_dims = [borrow_prop_dims[i] for i in range(nibble_idx - 5, nibble_idx)]
                        if borrow_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, borrow_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = B_SCALE
                            # Condition on first 4 BP flags
                            for bp_d in bp_dims[:4]:
                                if bp_d < self.dim:
                                    expert.up.weight[addr, bp_d] = B_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.SUB] = -B_PENALTY * 5
                            # gate checks the last BP flag
                            if bp_dims[-1] < self.dim:
                                expert.gate.weight[addr, bp_dims[-1]] = 10.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE 7: Six-hop: B[i-7] AND BP[i-6]...BP[i-1]
            # Condition up on B and first 5 BP flags
            if nibble_idx >= 7:
                for val in range(16):
                    addr = base_addr + val
                    if addr < expert.ffn_dim:
                        borrow_dim = borrow_start + val
                        output_dim = output_start + val
                        gen_dim = borrow_gen_dims[nibble_idx - 7]
                        bp_dims = [borrow_prop_dims[i] for i in range(nibble_idx - 6, nibble_idx)]
                        if borrow_dim < self.dim and output_dim < self.dim:
                            expert.up.weight[addr, borrow_dim] = 10.0
                            if gen_dim < self.dim:
                                expert.up.weight[addr, gen_dim] = B_SCALE
                            # Condition on first 5 BP flags
                            for bp_d in bp_dims[:5]:
                                if bp_d < self.dim:
                                    expert.up.weight[addr, bp_d] = B_SCALE
                            expert.up.weight[addr, E.OPCODE_START + Opcode.SUB] = -B_PENALTY * 6
                            # gate checks the last BP flag
                            if bp_dims[-1] < self.dim:
                                expert.gate.weight[addr, bp_dims[-1]] = 10.0
                            expert.down.weight[output_dim, addr] = HOP_DOWN
                base_addr += 16

            # CASE FINAL: No borrow - use RESULT
            # When no borrow arrives, use RESULT directly
            for val in range(16):
                addr = base_addr + val
                if addr < expert.ffn_dim:
                    result_dim = result_start + val
                    output_dim = output_start + val
                    if result_dim < self.dim and output_dim < self.dim:
                        expert.up.weight[addr, result_dim] = 10.0
                        expert.gate.weight[addr, result_dim] = 10.0
                        # Suppress when direct borrow from B[i-1]
                        if nibble_idx >= 1 and borrow_gen_dims[nibble_idx - 1] < self.dim:
                            expert.gate.weight[addr, borrow_gen_dims[nibble_idx - 1]] = -15.0
                        # Use moderate down weight - borrow cases with 50M will override
                        expert.down.weight[output_dim, addr] = 5000.0
            base_addr += 16

    def _bake_carry_chain_expert(self, expert: OpcodeExpert):
        """Legacy: Replaced by _bake_layer3_add."""
        self._bake_layer3_add(expert)

    def _bake_exit(self, expert: OpcodeExpert):
        """
        Bake EXIT expert: outputs EOS token.

        When EXIT opcode is detected, output high activation at EOS dim.
        The output head maps EOS dim to EOS token.
        """
        E = EmbedDims
        addr = 0
        if addr < expert.ffn_dim:
            # Read from opcode dim to activate
            if E.OPCODE_START + Opcode.EXIT < self.dim:
                expert.up.weight[addr, E.OPCODE_START + Opcode.EXIT] = 8.0
                expert.gate.weight[addr, E.OPCODE_START + Opcode.EXIT] = 8.0

            # Output to EOS marker dim
            expert.down.weight[Vocab.EOS, addr] = 50.0

    def _bake_lea(self, expert: OpcodeExpert):
        """
        Bake LEA expert: AX = BP + imm.

        Uses same nibble addition structure as ADD.
        """
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                result = (a + b) & 0xFF

                if addr < expert.ffn_dim:
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + b < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_adj(self, expert: OpcodeExpert):
        """
        Bake ADJ expert: SP = SP + imm.

        Uses same nibble addition structure as ADD.
        """
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                result = (a + b) & 0xFF

                if addr < expert.ffn_dim:
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + b < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_div(self, expert: OpcodeExpert):
        """
        Bake DIV expert: nibble integer division.

        For nibble inputs (0-15), computes a // b.
        Division by zero returns 0.

        NOTE: This is nibble-only division. For full 32-bit, we'd need
        multi-cycle division or Newton-Raphson approximation.
        """
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                result = a // b if b != 0 else 0

                if addr < expert.ffn_dim:
                    # Read from OP_A/OP_B dims (forward_fully_neural_alu sets these)
                    if E.OP_A_NIBBLE_0_LO + a < self.dim:
                        expert.up.weight[addr, E.OP_A_NIBBLE_0_LO + a] = 8.0
                    if E.OP_B_NIBBLE_0_LO + b < self.dim:
                        expert.gate.weight[addr, E.OP_B_NIBBLE_0_LO + b] = 8.0

                    # Output to RESULT and OUTPUT_BYTE dims
                    lo_nibble = result % 16
                    hi_nibble = result // 16

                    if E.RESULT_0_LO + lo_nibble < self.dim:
                        expert.down.weight[E.RESULT_0_LO + lo_nibble, addr] = 5.0
                    if E.RESULT_0_HI + hi_nibble < self.dim:
                        expert.down.weight[E.RESULT_0_HI + hi_nibble, addr] = 5.0
                    if E.OUTPUT_BYTE + lo_nibble < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + lo_nibble, addr] = 5.0
                    if E.OUTPUT_BYTE + 16 + hi_nibble < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + 16 + hi_nibble, addr] = 5.0

    def _bake_mod(self, expert: OpcodeExpert):
        """
        Bake MOD expert: nibble modulo.

        For nibble inputs (0-15), computes a % b.
        Modulo by zero returns 0.

        NOTE: This is nibble-only modulo. For full 32-bit, we'd need
        multi-cycle division or approximation.
        """
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                result = a % b if b != 0 else 0

                if addr < expert.ffn_dim:
                    # Read from OP_A/OP_B dims
                    if E.OP_A_NIBBLE_0_LO + a < self.dim:
                        expert.up.weight[addr, E.OP_A_NIBBLE_0_LO + a] = 8.0
                    if E.OP_B_NIBBLE_0_LO + b < self.dim:
                        expert.gate.weight[addr, E.OP_B_NIBBLE_0_LO + b] = 8.0

                    # Output to RESULT and OUTPUT_BYTE dims
                    lo_nibble = result % 16
                    hi_nibble = result // 16

                    if E.RESULT_0_LO + lo_nibble < self.dim:
                        expert.down.weight[E.RESULT_0_LO + lo_nibble, addr] = 5.0
                    if E.RESULT_0_HI + hi_nibble < self.dim:
                        expert.down.weight[E.RESULT_0_HI + hi_nibble, addr] = 5.0
                    if E.OUTPUT_BYTE + lo_nibble < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + lo_nibble, addr] = 5.0
                    if E.OUTPUT_BYTE + 16 + hi_nibble < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + 16 + hi_nibble, addr] = 5.0

    def _bake_psh(self, expert: OpcodeExpert):
        """
        Bake PSH expert: SP = SP - 8, mem[SP] = AX.

        For SP computation: subtract 8 from SP nibble.
        """
        E = EmbedDims
        for a in range(16):
            b = 8  # PSH always subtracts 8
            addr = a * 16 + b
            result = (a - b) & 0xFF

            if addr < expert.ffn_dim:
                if E.NIBBLE_A_START + a < self.dim:
                    expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                if E.NIBBLE_B_START + b < self.dim:
                    expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0
                if E.BYTES_START + result < self.dim:
                    expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_ent(self, expert: OpcodeExpert):
        """
        Bake ENT expert: push BP, BP = SP, SP = SP - 8 - imm.

        For SP computation uses subtraction table.
        """
        E = EmbedDims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                result = (a - b) & 0xFF

                if addr < expert.ffn_dim:
                    if E.NIBBLE_A_START + a < self.dim:
                        expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    if E.NIBBLE_B_START + b < self.dim:
                        expert.gate.weight[addr, E.NIBBLE_B_START + b] = 8.0
                    if E.BYTES_START + result < self.dim:
                        expert.down.weight[E.BYTES_START + result, addr] = 5.0

    def _bake_lev(self, expert: OpcodeExpert):
        """
        Bake LEV expert: SP = BP, pop BP, return.

        For SP computation: SP = BP (copy).
        """
        E = EmbedDims
        for a in range(16):
            addr = a
            if addr < expert.ffn_dim:
                if E.NIBBLE_A_START + a < self.dim:
                    expert.up.weight[addr, E.NIBBLE_A_START + a] = 8.0
                    expert.gate.weight[addr, E.NIBBLE_A_START + a] = 8.0
                if E.BYTES_START + a < self.dim:
                    expert.down.weight[E.BYTES_START + a, addr] = 5.0

    def _bake_jmp(self, expert: OpcodeExpert):
        """
        Bake JMP expert: PC = immediate.

        Same as IMM but for PC instead of AX.
        """
        E = EmbedDims
        for byte_val in range(256):
            addr = byte_val
            if addr < expert.ffn_dim:
                if E.OPERAND_B_START + byte_val < self.dim:
                    expert.up.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                    expert.gate.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                if E.BYTES_START + byte_val < self.dim:
                    expert.down.weight[E.BYTES_START + byte_val, addr] = 5.0

    def _bake_bz(self, expert: OpcodeExpert):
        """
        Bake BZ expert: PC = immediate if AX == 0, else PC + 8.

        For neural execution, we compute both options and use Python for selection.
        The expert outputs immediate bytes (like JMP).
        """
        E = EmbedDims
        for byte_val in range(256):
            addr = byte_val
            if addr < expert.ffn_dim:
                if E.OPERAND_B_START + byte_val < self.dim:
                    expert.up.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                    expert.gate.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                if E.BYTES_START + byte_val < self.dim:
                    expert.down.weight[E.BYTES_START + byte_val, addr] = 5.0

    def _bake_bnz(self, expert: OpcodeExpert):
        """
        Bake BNZ expert: PC = immediate if AX != 0, else PC + 8.

        For neural execution, we compute both options and use Python for selection.
        The expert outputs immediate bytes (like JMP).
        """
        E = EmbedDims
        for byte_val in range(256):
            addr = byte_val
            if addr < expert.ffn_dim:
                if E.OPERAND_B_START + byte_val < self.dim:
                    expert.up.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                    expert.gate.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                if E.BYTES_START + byte_val < self.dim:
                    expert.down.weight[E.BYTES_START + byte_val, addr] = 5.0

    def _bake_jsr(self, expert: OpcodeExpert):
        """
        Bake JSR expert: push PC+8, PC = immediate.

        Similar to PSH for SP modification, JMP for PC modification.
        """
        E = EmbedDims
        # For PC output: immediate bytes (like JMP)
        for byte_val in range(256):
            addr = byte_val
            if addr < expert.ffn_dim:
                if E.OPERAND_B_START + byte_val < self.dim:
                    expert.up.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                    expert.gate.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                if E.BYTES_START + byte_val < self.dim:
                    expert.down.weight[E.BYTES_START + byte_val, addr] = 5.0

    def _bake_imm(self, expert: OpcodeExpert):
        """
        Bake IMM expert: copies operand_b byte to output.

        Operand_b is injected as byte one-hot.
        """
        E = EmbedDims
        for byte_val in range(256):
            addr = byte_val
            if addr < expert.ffn_dim:
                # Read from operand_b one-hot
                if E.OPERAND_B_START + byte_val < self.dim:
                    expert.up.weight[addr, E.OPERAND_B_START + byte_val] = 8.0
                    expert.gate.weight[addr, E.OPERAND_B_START + byte_val] = 8.0

                # Output to byte one-hot
                if E.BYTES_START + byte_val < self.dim:
                    expert.down.weight[E.BYTES_START + byte_val, addr] = 5.0

    def _bake_mul(self, expert: OpcodeExpert):
        """
        Bake MUL expert: nibble multiplication (byte 0 only).

        For nibble values (0-15), computes a * b.
        Output goes to byte 0 (nibbles 0,1) of RESULT and OUTPUT_BYTE dims.

        NOTE: This is nibble-only multiplication. For full 32-bit, we'd need
        multi-cycle multiplication or a much larger lookup table.
        """
        E = EmbedDims

        # Read from OP_A and OP_B nibble dims (byte 0 low nibble)
        # forward_fully_neural_alu sets these dims
        for a in range(16):
            for b in range(16):
                addr = a * 16 + b
                if addr < expert.ffn_dim:
                    # up reads from OP_A byte 0 low nibble
                    if E.OP_A_NIBBLE_0_LO + a < self.dim:
                        expert.up.weight[addr, E.OP_A_NIBBLE_0_LO + a] = 8.0
                    # gate reads from OP_B byte 0 low nibble
                    if E.OP_B_NIBBLE_0_LO + b < self.dim:
                        expert.gate.weight[addr, E.OP_B_NIBBLE_0_LO + b] = 8.0

                    product = a * b  # 0-225
                    lo_nibble = product % 16
                    hi_nibble = product // 16

                    # Output to RESULT dims (for full transformer path)
                    if E.RESULT_0_LO + lo_nibble < self.dim:
                        expert.down.weight[E.RESULT_0_LO + lo_nibble, addr] = 5.0
                    if E.RESULT_0_HI + hi_nibble < self.dim:
                        expert.down.weight[E.RESULT_0_HI + hi_nibble, addr] = 5.0

                    # Also output to OUTPUT_BYTE dims (for direct forward_fully_neural_alu)
                    if E.OUTPUT_BYTE + lo_nibble < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + lo_nibble, addr] = 5.0
                    if E.OUTPUT_BYTE + 16 + hi_nibble < self.dim:
                        expert.down.weight[E.OUTPUT_BYTE + 16 + hi_nibble, addr] = 5.0

    def _bake_output(self):
        """
        Bake output head for token decoding.

        Maps hidden dims to token logits:
        - Dims 0-15: Marker one-hots → marker tokens (for embedding roundtrip)
        - Dims 16-271: Byte one-hots → byte tokens (gathered via attention)
        - Dims 304-309: Step position markers → marker tokens (for marker positions)
        - Dim 410: "Is byte position" flag → suppresses marker output
        - OUTPUT_BYTE dims: Result nibbles → byte tokens (fully neural path)
        """
        E = EmbedDims
        self.lm_head.weight.zero_()

        # Marker one-hots (dims 0-15) → marker tokens
        # Used for embedding roundtrip and context markers
        for i in range(16):
            if i < Vocab.BYTE_BASE:
                self.lm_head.weight[i, i] = 10.0

        # Byte one-hots (dims 16-271) → byte tokens
        # These are gathered via attention from context
        for b in range(256):
            tok = Vocab.BYTE_BASE + b
            if 16 + b < self.dim:
                self.lm_head.weight[tok, 16 + b] = 15.0

        # Step position markers (dims 304-309) → marker tokens
        # Only active at marker positions (dim 410 = 0)
        marker_map = [Vocab.REG_PC, Vocab.REG_AX, Vocab.REG_SP, Vocab.REG_BP, Vocab.MEM, Vocab.STEP_END]
        for i, marker in enumerate(marker_map):
            if 304 + i < self.dim:
                self.lm_head.weight[marker, 304 + i] = 50.0

        # "Is byte position" flag (dim 410) suppresses markers and boosts bytes
        # When dim 410 is high, we want byte tokens not marker tokens
        if 410 < self.dim:
            for marker in marker_map:
                self.lm_head.weight[marker, 410] = -30.0  # Suppress markers at byte positions

        # === FULLY NEURAL: OUTPUT_BYTE nibbles → byte tokens ===
        # OUTPUT_BYTE has 8 nibble slots × 16 one-hot dims each
        # For each output byte position (0-3), combine 2 nibbles into byte token
        #
        # Byte 0: nibbles 0,1 at OUTPUT_BYTE+0..31
        # Byte 1: nibbles 2,3 at OUTPUT_BYTE+32..63
        # Byte 2: nibbles 4,5 at OUTPUT_BYTE+64..95
        # Byte 3: nibbles 6,7 at OUTPUT_BYTE+96..127
        #
        # Each byte token b = lo + 16*hi where lo,hi are nibble values
        # lm_head maps: if OUTPUT_BYTE[byte_idx*32 + lo] AND OUTPUT_BYTE[byte_idx*32 + 16 + hi] → byte token
        #
        # Strategy: Use additive logits from both nibble dims
        # For byte b with lo = b & 0xF, hi = (b >> 4) & 0xF:
        #   logit[byte_tok] += weight when OUTPUT[lo_dim] is high
        #   logit[byte_tok] += weight when OUTPUT[hi_dim] is high
        # Net effect: correct byte gets 2*weight, wrong bytes get <= 1*weight

        for byte_idx in range(4):  # 4 output bytes
            lo_base = E.OUTPUT_BYTE + byte_idx * 32
            hi_base = lo_base + 16

            for byte_val in range(256):
                tok = Vocab.BYTE_BASE + byte_val
                lo_nibble = byte_val & 0xF
                hi_nibble = (byte_val >> 4) & 0xF

                lo_dim = lo_base + lo_nibble
                hi_dim = hi_base + hi_nibble

                if lo_dim < self.dim and hi_dim < self.dim:
                    # Each matching nibble adds to logit
                    # Correct byte has both nibbles matching → highest logit
                    self.lm_head.weight[tok, lo_dim] += 25.0
                    self.lm_head.weight[tok, hi_dim] += 25.0

        # Also map RESULT nibbles directly for single-byte output
        # RESULT_0_LO/HI → byte 0 token
        for byte_val in range(256):
            tok = Vocab.BYTE_BASE + byte_val
            lo_nibble = byte_val & 0xF
            hi_nibble = (byte_val >> 4) & 0xF

            if E.RESULT_0_LO + lo_nibble < self.dim:
                self.lm_head.weight[tok, E.RESULT_0_LO + lo_nibble] += 20.0
            if E.RESULT_0_HI + hi_nibble < self.dim:
                self.lm_head.weight[tok, E.RESULT_0_HI + hi_nibble] += 20.0

    # =========================================================================
    # STANDARD FORWARD PASS - DO NOT MODIFY
    # =========================================================================
    # This function implements the pure neural forward pass with ZERO Python
    # control flow or conditionals. The entire computation is:
    #
    #     logits = lm_head(ln_f(transformer_layers(tok_emb(tokens))))
    #
    # All VM operations (ALU, memory, control flow) are baked into the weights.
    # DO NOT add Python logic, conditionals, or branching here.
    # =========================================================================

    def forward_standard(self, tokens: torch.Tensor,
                          kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                          ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        STANDARD TRANSFORMER FORWARD - DO NOT MODIFY THIS FUNCTION.

        This is the PURE NEURAL forward:
            logits = lm_head(ln_f(layers(tok_emb(tokens))))

        CRITICAL: NO Python control flow. NO conditionals. NO injection.
        Just: Embedding → Attention+FFN × N → Output head

        All computation is in baked weights:
        - tok_emb: Encodes tokens with nibbles
        - Attention: Gathers operands, matches memory addresses (with ALiBi)
        - MoE FFN: Computes ALU ops via baked nibble tables
        - lm_head: Decodes result dims to token logits

        Memory operations use attention:
        - K encodes address nibbles at MEM positions
        - Q encodes query address
        - V encodes value nibbles
        - ALiBi bias prefers recent entries (handles overwrites)
        - softmax1 returns 0 for uninitialized addresses

        DO NOT ADD:
        - Python if/else statements
        - Python loops over values
        - Python arithmetic on operands
        - Any branching based on opcode or data

        Args:
            tokens: Input token sequence [B, L]
            kv_cache: Optional KV cache from previous tokens

        Returns:
            (logits [B, vocab_size], new_kv_cache)
        """
        # =====================================================================
        # PURE FORWARD PASS - NO PYTHON COMPUTATION
        # =====================================================================

        # 1. EMBEDDING - Token → Hidden state (one-hot + nibbles)
        x = self.tok_emb(tokens)
        seq_len = x.size(1)

        # 2. CAUSAL MASK - Standard autoregressive mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1) * -1e9

        # 3. TRANSFORMER LAYERS - Attention gathers, MoE computes
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_kv = layer(x, mask, layer_cache)
            new_kv_cache.append(layer_kv)

        # 4. OUTPUT HEAD - Hidden state → Token logits
        logits = self.lm_head(self.ln_f(x[:, -1]))

        # =====================================================================
        return logits, new_kv_cache

    # =========================================================================
    # TRULY NEURAL AUTOREGRESSIVE GENERATION
    # =========================================================================

    def generate_token(self, tokens: torch.Tensor,
                       kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
                       ) -> Tuple[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Generate ONE token via PURE NEURAL autoregressive inference.

        This is the PURE NEURAL generation function:
        - Standard transformer forward with KV caching
        - NO Python control flow in the forward pass
        - Memory lookup via attention with ALiBi bias
        - ALU computation via MoE experts

        The only Python is:
        - Tensor indexing (I/O)
        - argmax for token selection (I/O)

        Args:
            tokens: Current token(s) to process [1, L]
            kv_cache: Cached K,V from previous tokens

        Returns:
            (next_token_id, updated_kv_cache)
        """
        with torch.no_grad():
            logits, new_cache = self.forward_standard(tokens, kv_cache)
            next_tok = logits.argmax(dim=-1).item()
            return next_tok, new_cache

