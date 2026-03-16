#!/usr/bin/env python3
"""
Weight-Baked VM - Program embedded in weights, not context

The PROGRAM is baked into FFN weights as lookup tables.
Context only contains STATE (registers).
Execution is pure forward passes.

Example: "return 42" is baked as:
  - FFN encodes: PC=0 -> op=IMM, imm=42
  - FFN encodes: PC=1 -> op=RET

Context format:
  [PC_VALUE] [AX_LO] [AX_HI] [STEP]

Generation produces:
  [NEWAX] [value_lo] [value_hi] [NEWPC] [pc] [STEP]  (repeat)
  [HALT] (terminate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


class WeightBakedVM(nn.Module):
    """
    VM with program baked into weights.

    The FFN implements a lookup table:
      PC -> (opcode, immediate)

    Then another FFN computes:
      (opcode, immediate, AX) -> new_AX

    Output projection predicts next token based on state.
    """

    # Token IDs (minimal)
    TOK_PC = 256       # Current PC marker (followed by value)
    TOK_AX = 257       # Current AX marker (followed by value)
    TOK_STEP = 258     # Execute step marker
    TOK_NEWAX = 259    # New AX value follows
    # NEWPC tokens encode PC value directly: NEWPC_0, NEWPC_1, ...
    TOK_NEWPC_BASE = 260  # NEWPC_0 = 260, NEWPC_1 = 261, ...
    TOK_OUT = 276      # Output byte follows
    TOK_HALT = 277     # Halt execution

    VOCAB_SIZE = 278

    # Opcodes
    OP_IMM = 0    # AX = imm
    OP_ADD = 1    # AX = AX + imm
    OP_SUB = 2    # AX = AX - imm
    OP_MUL = 3    # AX = AX * imm
    OP_PUTC = 4   # output AX & 0xFF
    OP_RET = 5    # halt

    def __init__(self, hidden_dim: int = 128, program: List[Tuple[int, int]] = None):
        """
        Args:
            hidden_dim: Model dimension
            program: List of (opcode, immediate) tuples to bake in
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.program = program or [(self.OP_IMM, 42), (self.OP_RET, 0)]

        # Embeddings
        self.tok_emb = nn.Embedding(self.VOCAB_SIZE, hidden_dim)
        self.pos_emb = nn.Embedding(256, hidden_dim)

        # Layer 1: Instruction fetch (PC -> op, imm)
        # Baked lookup table
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fetch_pc = nn.Linear(hidden_dim, 16)  # PC detection
        self.fetch_out = nn.Linear(16, hidden_dim)  # op, imm output

        # Layer 2: Execute (op, imm, AX -> new AX)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.exec_ffn = nn.Linear(hidden_dim, hidden_dim * 4)
        self.exec_gate = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.exec_out = nn.Linear(hidden_dim * 4, hidden_dim)

        # Layer 3: Output selection
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.out_ffn = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, self.VOCAB_SIZE, bias=False)

        # Attention for state reading
        self.attn_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialize weights with baked program
        self._init_weights()

    def _init_weights(self):
        """Bake the program into weights."""
        with torch.no_grad():
            H = self.hidden_dim

            # Zero everything
            self._zero_all()

            # === EMBEDDINGS ===
            # Token embeddings: dims 0-7 for special markers, 8-15 for byte value
            for i in range(256):
                for bit in range(8):
                    if (i >> bit) & 1:
                        self.tok_emb.weight.data[i, 8 + bit] = 1.0

            # Special tokens
            self.tok_emb.weight.data[self.TOK_PC, 0] = 10.0     # PC marker
            self.tok_emb.weight.data[self.TOK_AX, 1] = 10.0     # AX marker
            self.tok_emb.weight.data[self.TOK_STEP, 2] = 10.0   # STEP marker
            self.tok_emb.weight.data[self.TOK_NEWAX, 3] = 10.0  # NEWAX marker
            # NEWPC_* tokens: have NEWPC marker AND PC value embedded directly
            # Use STRONG embedding (50) so that even partial attention gives sufficient signal
            for pc_val in range(16):
                tok_id = self.TOK_NEWPC_BASE + pc_val
                self.tok_emb.weight.data[tok_id, 4] = 10.0      # NEWPC marker
                # Also embed the PC value in dims 16-23 (same as where attention puts PC)
                # Strong encoding so partial attention still dominates
                for bit in range(8):
                    if (pc_val >> bit) & 1:
                        self.tok_emb.weight.data[tok_id, 16 + bit] = 50.0
            self.tok_emb.weight.data[self.TOK_OUT, 5] = 10.0    # OUT marker
            self.tok_emb.weight.data[self.TOK_HALT, 6] = 10.0   # HALT marker

            # Position embeddings - with recency bias for attention
            # Recent positions get higher values in dims 56-63
            for pos in range(256):
                for bit in range(8):
                    if (pos >> bit) & 1:
                        self.pos_emb.weight.data[pos, 48 + bit] = 0.1
                # Add recency signal - increases with position
                self.pos_emb.weight.data[pos, 56] = pos * 0.1  # Recency in dim 56
                # Also add signal for byte tokens to prefer recent markers
                self.pos_emb.weight.data[pos, 57] = pos * 0.5  # Stronger recency for Q/K

            # === ATTENTION (read state AND detect previous token type) ===

            # Q: look for specific markers and detect patterns
            # Marker->Marker matching (same marker type attracts)
            self.attn_q.weight.data[0, 0] = 10.0  # PC looks for PC
            self.attn_q.weight.data[1, 1] = 10.0  # AX looks for AX
            self.attn_q.weight.data[2, 2] = 10.0  # STEP looks for STEP
            self.attn_q.weight.data[3, 3] = 10.0  # NEWAX looks for NEWAX
            self.attn_q.weight.data[4, 4] = 10.0  # NEWPC looks for NEWPC

            # NEW: STEP also looks for PC markers (to read current PC)
            # Use a separate Q/K channel (dim 5) for this cross-type lookup
            # Strong weight to get more attention to NEWPC_* positions
            self.attn_q.weight.data[5, 2] = 30.0   # STEP marker activates this Q channel (strong)
            self.attn_q.weight.data[6, 2] = 10.0   # STEP also looks for AX markers

            # All positions use dim 57 for recency matching (Q increases with position)
            self.attn_q.weight.data[58, 57] = 1.0  # Query position recency

            # K: markers identify token types
            self.attn_k.weight.data[0, 0] = 10.0  # This is PC marker
            self.attn_k.weight.data[1, 1] = 10.0  # This is AX marker
            self.attn_k.weight.data[2, 2] = 10.0  # This is STEP
            self.attn_k.weight.data[3, 3] = 10.0  # This is NEWAX
            self.attn_k.weight.data[4, 4] = 10.0  # This is NEWPC

            # NEW: PC/NEWPC markers respond to STEP's query (channel 5)
            self.attn_k.weight.data[5, 0] = 30.0   # PC marker responds (strong)
            self.attn_k.weight.data[5, 4] = 30.0   # NEWPC marker also responds (strong)
            # AX/NEWAX markers respond to STEP's query (channel 6)
            self.attn_k.weight.data[6, 1] = 10.0   # AX marker responds
            self.attn_k.weight.data[6, 3] = 10.0   # NEWAX marker also responds

            # Key position recency - recent tokens get higher K values
            self.attn_k.weight.data[58, 57] = 1.0  # Key position recency

            # V: pass byte values, computed results, AND marker flags
            for i in range(8):
                self.attn_v.weight.data[16 + i, 8 + i] = 1.0   # Byte value -> dims 16-23 (for PC from byte tokens)
                self.attn_v.weight.data[16 + i, 16 + i] = 1.0  # ALSO pass dims 16-23 directly (for NEWPC_* tokens)
                self.attn_v.weight.data[24 + i, 8 + i] = 1.0   # Byte value -> dims 24-31 (for AX)

            # Pass through instruction info and result
            for i in range(16):
                self.attn_v.weight.data[36 + i, 36 + i] = 1.0

            # Pass through marker flags to detect previous token type
            # When attending, we get a weighted sum of values from all positions
            # To detect "previous was NEWAX", we want dim 80 to be high if NEWAX is recent
            self.attn_v.weight.data[80, 3] = 10.0  # NEWAX marker -> dim 80
            self.attn_v.weight.data[81, 4] = 10.0  # NEWPC marker -> dim 81
            self.attn_v.weight.data[82, 2] = 10.0  # STEP marker -> dim 82

            # O: identity
            for i in range(H):
                self.attn_o.weight.data[i, i] = 1.0

            # === INSTRUCTION FETCH (baked program) ===
            # Maps PC value (dims 16-23) to op/imm (dims 32-47)

            # PC detection: one-hot over PC values
            # Use strict AND-like detection
            for pc_val in range(min(len(self.program), 16)):
                # Detect this specific PC value using AND logic
                # All matching bits must be present, all non-matching must be absent
                match_count = 0
                for bit in range(8):
                    if (pc_val >> bit) & 1:
                        self.fetch_pc.weight.data[pc_val, 16 + bit] = 10.0
                        match_count += 1
                    else:
                        self.fetch_pc.weight.data[pc_val, 16 + bit] = -10.0
                # Bias: need all bits to match
                # For pc_val=0: all bits are 0, so we need all -10 inputs to sum to positive
                # Actually simpler: for PC=0, all bits should be 0
                if pc_val == 0:
                    # PC=0: require all bits to be 0 (negative weights mean "should be 0")
                    self.fetch_pc.bias.data[pc_val] = 70.0  # Offset to make it positive when all 0
                else:
                    # PC=N: require matching pattern
                    self.fetch_pc.bias.data[pc_val] = -(8 - match_count) * 10.0 + match_count * 5.0

            # Output op and imm for each PC
            for pc_val, (op, imm) in enumerate(self.program):
                if pc_val < 16:
                    # Encode opcode in dims 32-35
                    for bit in range(4):
                        if (op >> bit) & 1:
                            self.fetch_out.weight.data[32 + bit, pc_val] = 10.0
                        # Don't set negative - let it be 0

                    # Encode immediate in dims 36-43
                    for bit in range(8):
                        if (imm >> bit) & 1:
                            self.fetch_out.weight.data[36 + bit, pc_val] = 10.0

            # === EXECUTE FFN ===
            # Implements: IMM, ADD, SUB, MUL, RET
            #
            # Opcode encoding:
            # - IMM = 0 = 0b0000: bits 0-3 all 0
            # - ADD = 1 = 0b0001: bit 0 = 1, rest 0
            # - SUB = 2 = 0b0010: bit 1 = 1, rest 0
            # - MUL = 3 = 0b0011: bits 0,1 = 1, bits 2,3 = 0
            # - PUTC = 4 = 0b0100: bit 2 = 1, rest 0
            # - RET = 5 = 0b0101: bits 0,2 = 1, bits 1,3 = 0
            #
            # Inputs:
            # - dims 24-31: AX value (from attention)
            # - dims 32-35: opcode bits (from fetch)
            # - dims 36-43: immediate bits (from fetch)
            #
            # Output:
            # - dims 44-51: result value
            # - dim 52: halt flag

            entry_idx = 0

            # --- IMM: AX = immediate ---
            # Active when opcode = 0 (all bits 0)
            for bit in range(8):
                self.exec_ffn.weight.data[entry_idx, 36 + bit] = 10.0  # Immediate bit
                self.exec_ffn.weight.data[entry_idx, 32] = -20.0  # NOT opcode bit 0
                self.exec_ffn.weight.data[entry_idx, 33] = -20.0  # NOT opcode bit 1
                self.exec_ffn.weight.data[entry_idx, 34] = -20.0  # NOT opcode bit 2
                self.exec_ffn.weight.data[entry_idx, 35] = -20.0  # NOT opcode bit 3
                self.exec_ffn.bias.data[entry_idx] = 0.0
                self.exec_gate.weight.data[entry_idx, 36 + bit] = 1.0
                self.exec_out.weight.data[44 + bit, entry_idx] = 1.0
                entry_idx += 1

            # --- ADD: AX = AX + immediate (simple bit-level with saturation) ---
            # For small values, we approximate with: result = AX OR immediate (for non-overlapping bits)
            # Plus: entries for each bit position that sum AX and IMM
            # This is a simplified adder that works for non-overlapping bits
            for bit in range(8):
                # Copy AX bit when ADD opcode (bit 0 = 1, others 0)
                self.exec_ffn.weight.data[entry_idx, 24 + bit] = 10.0  # AX bit
                self.exec_ffn.weight.data[entry_idx, 32] = 5.0   # opcode bit 0
                self.exec_ffn.weight.data[entry_idx, 33] = -20.0  # NOT bit 1
                self.exec_ffn.weight.data[entry_idx, 34] = -20.0  # NOT bit 2
                self.exec_ffn.weight.data[entry_idx, 35] = -20.0  # NOT bit 3
                self.exec_ffn.bias.data[entry_idx] = -4.0
                self.exec_gate.weight.data[entry_idx, 24 + bit] = 1.0
                self.exec_out.weight.data[44 + bit, entry_idx] = 1.0
                entry_idx += 1

                # Copy IMM bit when ADD opcode
                self.exec_ffn.weight.data[entry_idx, 36 + bit] = 10.0  # IMM bit
                self.exec_ffn.weight.data[entry_idx, 32] = 5.0   # opcode bit 0
                self.exec_ffn.weight.data[entry_idx, 33] = -20.0  # NOT bit 1
                self.exec_ffn.weight.data[entry_idx, 34] = -20.0  # NOT bit 2
                self.exec_ffn.weight.data[entry_idx, 35] = -20.0  # NOT bit 3
                self.exec_ffn.bias.data[entry_idx] = -4.0
                self.exec_gate.weight.data[entry_idx, 36 + bit] = 1.0
                self.exec_out.weight.data[44 + bit, entry_idx] = 1.0
                entry_idx += 1

            # --- SUB: AX = AX - immediate (simple bit-level) ---
            # For simplicity, copy AX bits and subtract IMM contributions
            for bit in range(8):
                # Copy AX bit when SUB opcode (bit 1 = 1, others 0)
                self.exec_ffn.weight.data[entry_idx, 24 + bit] = 10.0  # AX bit
                self.exec_ffn.weight.data[entry_idx, 32] = -20.0  # NOT bit 0
                self.exec_ffn.weight.data[entry_idx, 33] = 5.0   # opcode bit 1
                self.exec_ffn.weight.data[entry_idx, 34] = -20.0  # NOT bit 2
                self.exec_ffn.weight.data[entry_idx, 35] = -20.0  # NOT bit 3
                self.exec_ffn.bias.data[entry_idx] = -4.0
                self.exec_gate.weight.data[entry_idx, 24 + bit] = 1.0
                self.exec_out.weight.data[44 + bit, entry_idx] = 1.0
                entry_idx += 1

                # Subtract IMM bit when SUB opcode (negative contribution)
                self.exec_ffn.weight.data[entry_idx, 36 + bit] = 10.0  # IMM bit
                self.exec_ffn.weight.data[entry_idx, 32] = -20.0  # NOT bit 0
                self.exec_ffn.weight.data[entry_idx, 33] = 5.0   # opcode bit 1
                self.exec_ffn.weight.data[entry_idx, 34] = -20.0  # NOT bit 2
                self.exec_ffn.weight.data[entry_idx, 35] = -20.0  # NOT bit 3
                self.exec_ffn.bias.data[entry_idx] = -4.0
                self.exec_gate.weight.data[entry_idx, 36 + bit] = 1.0
                self.exec_out.weight.data[44 + bit, entry_idx] = -1.0  # SUBTRACT
                entry_idx += 1

            # --- MUL: AX * immediate using SwiGLU ---
            # For result bit k, we sum products where i + j = k
            for result_bit in range(8):
                for ax_bit in range(8):
                    imm_bit = result_bit - ax_bit
                    if 0 <= imm_bit < 8:
                        self.exec_ffn.weight.data[entry_idx, 24 + ax_bit] = 10.0  # AX bit
                        self.exec_ffn.weight.data[entry_idx, 32] = 5.0   # MUL bit 0
                        self.exec_ffn.weight.data[entry_idx, 33] = 5.0   # MUL bit 1
                        self.exec_ffn.weight.data[entry_idx, 34] = -20.0  # NOT bit 2
                        self.exec_ffn.weight.data[entry_idx, 35] = -20.0  # NOT bit 3
                        self.exec_ffn.bias.data[entry_idx] = -8.0
                        self.exec_gate.weight.data[entry_idx, 36 + imm_bit] = 1.0
                        self.exec_out.weight.data[44 + result_bit, entry_idx] = 1.0
                        entry_idx += 1

            # --- RET: Signal halt ---
            ret_entry = entry_idx
            self.exec_ffn.weight.data[ret_entry, 32] = 10.0   # bit 0 = 1
            self.exec_ffn.weight.data[ret_entry, 33] = -10.0  # bit 1 = 0
            self.exec_ffn.weight.data[ret_entry, 34] = 10.0   # bit 2 = 1
            self.exec_ffn.weight.data[ret_entry, 35] = -10.0  # bit 3 = 0
            self.exec_ffn.bias.data[ret_entry] = -5.0
            self.exec_gate.weight.data[ret_entry, 32] = 1.0   # Gate on opcode presence
            self.exec_out.weight.data[52, ret_entry] = 20.0   # Set halt flag

            # === OUTPUT SELECTION ===
            # State machine using embedding markers and byte ranges
            #
            # State transitions:
            # STEP -> NEWAX (if not RET)
            # STEP -> HALT (if RET)
            # NEWAX -> byte (value)
            # byte (after NEWAX) -> NEWPC  [use dim 80 as "just output value" flag]
            # NEWPC -> byte (pc)
            # byte (after NEWPC) -> STEP   [use dim 81 as "just output pc" flag]

            # Route 0: HALT when RET opcode (dim 52 halt flag) AND STEP marker
            # Need: halt_flag high AND STEP marker present AND NOT NEWPC
            # At NEWPC_1: halt_flag can be ~25000! Need VERY strong NEWPC suppression
            # At first STEP (PC=0): halt_flag ~0, must NOT fire
            # At second STEP (PC=1): halt_flag ~25000, must fire
            self.out_ffn.weight.data[0, 52] = 0.01  # Halt flag contribution (need ~2500 from halt=25000)
            self.out_ffn.weight.data[0, 2] = 10.0   # STEP marker contribution
            self.out_ffn.weight.data[0, 4] = -100000.0 # NOT NEWPC marker (extremely strong)
            self.out_ffn.bias.data[0] = -200.0      # High threshold: need halt ~25000 to overcome
            self.out_proj.weight.data[56, 0] = 50.0  # -> HALT

            # Route 1: STEP marker -> NEWAX (when not halting)
            self.out_ffn.weight.data[1, 2] = 30.0   # STEP marker
            self.out_ffn.weight.data[1, 52] = -30.0  # Not halt flag
            self.out_ffn.bias.data[1] = -10.0
            self.out_proj.weight.data[57, 1] = 50.0  # -> NEWAX

            # Route 2: NEWAX marker -> byte value (result)
            # Only fire when BOTH NEWAX (dim 3 = 10) AND result bit is high (~41)
            # Need threshold that discriminates:
            #   NEWAX=10, result=0: 10*1 + 0*1 - 40 = -30 (don't fire)
            #   NEWAX=10, result=41: 10*1 + 41*1 - 40 = 11 (fire!)
            for bit in range(8):
                idx = 10 + bit
                self.out_ffn.weight.data[idx, 3] = 1.0       # NEWAX marker
                self.out_ffn.weight.data[idx, 44 + bit] = 1.0  # Result bit
                self.out_ffn.bias.data[idx] = -40.0  # High threshold!
                self.out_proj.weight.data[68 + bit, idx] = 20.0

            # Route 3: After value byte -> NEWPC_1 (PC increments from 0 to 1)
            # Detect: current token is a byte AND NEWAX was recent (not NEWPC)
            # Output: NEWPC_1 token directly (no separate byte)
            for bit in range(8):
                self.out_ffn.weight.data[3, 8 + bit] = 3.0  # Byte value bits
            self.out_ffn.weight.data[3, 3] = -500.0   # NOT NEWAX (suppress at NEWAX position)
            self.out_ffn.weight.data[3, 4] = -100.0   # NOT NEWPC
            self.out_ffn.weight.data[3, 2] = -100.0   # NOT STEP
            self.out_ffn.weight.data[3, 0] = -100.0   # NOT PC marker
            self.out_ffn.weight.data[3, 1] = -100.0   # NOT AX marker
            self.out_ffn.weight.data[3, 80] = 30.0    # NEWAX was in context (strong)
            self.out_ffn.weight.data[3, 81] = -30.0   # But NOT NEWPC recently
            self.out_ffn.bias.data[3] = -10.0        # Fire when byte after NEWAX
            # Output directly to NEWPC_1 (token 261)
            self.out_proj.weight.data[63, 3] = 100.0  # -> dim 63 for NEWPC_1

            # Route 4: NEWPC_* marker -> STEP (after updating PC)
            # NEWPC_* tokens have dim 4 = 10
            self.out_ffn.weight.data[4, 4] = 50.0   # NEWPC marker
            self.out_ffn.bias.data[4] = -25.0
            self.out_proj.weight.data[61, 4] = 100.0  # -> STEP

            # Route 5 removed: NEWPC_* is now a single token, goes directly to STEP via route 4

            # === LM HEAD ===
            # Map hidden dims to vocab logits

            # Byte outputs: dims 68-75 encode byte value in binary
            for i in range(256):
                for bit in range(8):
                    if (i >> bit) & 1:
                        self.lm_head.weight.data[i, 68 + bit] += 2.0
                    else:
                        self.lm_head.weight.data[i, 68 + bit] -= 0.5

            # Special tokens from dedicated dims
            self.lm_head.weight.data[self.TOK_HALT, 56] = 20.0
            self.lm_head.weight.data[self.TOK_NEWAX, 57] = 20.0
            # NEWPC_* tokens (260-275) use dim 63 for NEWPC_1 specifically
            self.lm_head.weight.data[self.TOK_NEWPC_BASE + 1, 63] = 20.0  # NEWPC_1
            self.lm_head.weight.data[self.TOK_STEP, 61] = 20.0
            self.lm_head.weight.data[self.TOK_OUT, 62] = 20.0

    def _zero_all(self):
        """Zero all weights."""
        for p in self.parameters():
            p.data.zero_()

        # Set LayerNorm to identity (scale=1, bias=0)
        # BUT: LayerNorm will still normalize, which ruins our signals
        # We need to compensate by using relative patterns instead of absolute values
        self.ln1.weight.data.fill_(1.0)
        self.ln2.weight.data.fill_(1.0)
        self.ln3.weight.data.fill_(1.0)
        self.ln_f.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, L = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(L, device=device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        # Causal attention for state reading
        mask = torch.triu(torch.ones(L, L, device=device) * float('-inf'), diagonal=1)

        q = self.attn_q(x)
        k = self.attn_k(x)
        v = self.attn_v(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        x = x + self.attn_o(attn_out)

        # Instruction fetch (baked program lookup)
        # Skip LayerNorm to preserve PC signal magnitudes
        pc_detect = F.relu(self.fetch_pc(x))
        x = x + self.fetch_out(pc_detect)

        # Execute
        x_norm = self.ln2(x)
        gate = F.silu(self.exec_ffn(x_norm))
        up = self.exec_gate(x_norm)
        x = x + self.exec_out(gate * up)

        # Output selection (skip LayerNorm to preserve signal magnitudes)
        out = F.relu(self.out_ffn(x))
        x = x + self.out_proj(out)

        # Final projection
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_tokens: int = 100) -> torch.Tensor:
        """Generate until HALT."""
        for _ in range(max_tokens):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == self.TOK_HALT:
                break

        return input_ids

    @torch.no_grad()
    def generate_speculative(self, input_ids: torch.Tensor, max_tokens: int = 100,
                             speculation_depth: int = 4) -> Tuple[torch.Tensor, dict]:
        """
        Speculative generation with verification.

        For this deterministic VM, speculation provides limited benefit since
        there's only one valid path. But this demonstrates the infrastructure.

        Returns:
            output_ids: Generated sequence
            stats: Dictionary with speculation stats
        """
        stats = {
            'total_tokens': 0,
            'speculative_batches': 0,
            'verified_tokens': 0,
        }

        while stats['total_tokens'] < max_tokens:
            # Generate speculation_depth tokens speculatively
            speculative_ids = input_ids.clone()
            speculative_tokens = []

            for _ in range(speculation_depth):
                logits = self.forward(speculative_ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                speculative_tokens.append(next_token.item())
                speculative_ids = torch.cat([speculative_ids, next_token], dim=1)

                if next_token.item() == self.TOK_HALT:
                    break

            # For this deterministic VM, all speculative tokens are valid
            # (no branching or external input)
            input_ids = speculative_ids
            stats['total_tokens'] += len(speculative_tokens)
            stats['verified_tokens'] += len(speculative_tokens)
            stats['speculative_batches'] += 1

            if self.TOK_HALT in speculative_tokens:
                break

        return input_ids, stats

    def count_parameters(self) -> dict:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        nonzero = sum((p.data != 0).sum().item() for p in self.parameters())
        return {
            'total': total,
            'nonzero': nonzero,
            'sparsity': 100 * (total - nonzero) / total if total > 0 else 0
        }

    def export_onnx(self, path: str, seq_len: int = 16):
        """Export model to ONNX format."""
        import torch.onnx

        # Create dummy input
        dummy_input = torch.randint(0, self.VOCAB_SIZE, (1, seq_len))

        # Export
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )
        return path

    def estimate_flops_per_token(self) -> dict:
        """Estimate FLOPs per token generation."""
        H = self.hidden_dim
        V = self.VOCAB_SIZE

        # Per layer FLOPs (rough estimates)
        embedding = H  # lookup
        attention = 4 * H * H  # Q, K, V, O projections (per position)
        fetch = H * 16 + 16 * H  # fetch_pc + fetch_out
        execute = H * H * 4 * 2 + H * 4 * H  # SwiGLU
        output = H * H * 2 * 2 + H * 2 * H  # out_ffn + out_proj
        lm_head = H * V  # projection to vocab

        total_per_token = embedding + attention + fetch + execute + output + lm_head

        return {
            'total_per_token': total_per_token,
            'embedding': embedding,
            'attention': attention,
            'fetch': fetch,
            'execute': execute,
            'output': output,
            'lm_head': lm_head
        }


def create_initial_state(model: WeightBakedVM, pc: int = 0, ax: int = 0) -> torch.Tensor:
    """Create initial state tokens (no program - it's in weights!)"""
    tokens = [
        model.TOK_PC, pc,        # Current PC
        model.TOK_AX, ax & 0xFF, # Current AX low
        model.TOK_STEP           # Start execution
    ]
    return torch.tensor([tokens], dtype=torch.long)


def main():
    print("=" * 70)
    print("  WEIGHT-BAKED VM")
    print("=" * 70)
    print()

    # Program: return 42
    program = [
        (WeightBakedVM.OP_IMM, 42),  # AX = 42
        (WeightBakedVM.OP_RET, 0),   # return
    ]

    print("Program baked into weights:")
    for i, (op, imm) in enumerate(program):
        op_names = ['IMM', 'ADD', 'SUB', 'MUL', 'PUTC', 'RET']
        print(f"  PC={i}: {op_names[op]} {imm}")
    print()

    # Create model with baked program
    model = WeightBakedVM(hidden_dim=128, program=program)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p.data == 0).sum().item() for p in model.parameters())
    nonzero = total - zeros

    print(f"Parameters: {total:,} total, {nonzero:,} non-zero ({100*zeros/total:.1f}% sparse)")
    print()

    # Initial state (no program tokens - it's in weights!)
    input_ids = create_initial_state(model, pc=0, ax=0)
    print(f"Initial state tokens: {input_ids.squeeze().tolist()}")
    print(f"  TOK_PC=256, 0, TOK_AX=257, 0, TOK_STEP=258")
    print()

    # Forward pass
    print("Forward pass...")
    with torch.no_grad():
        logits = model.forward(input_ids)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5 = probs.topk(5)

    print("Top 5 predictions after STEP:")
    for prob, idx in zip(top5.values, top5.indices):
        idx = idx.item()
        if idx == model.TOK_NEWAX:
            name = "NEWAX"
        elif idx == model.TOK_HALT:
            name = "HALT"
        elif model.TOK_NEWPC_BASE <= idx < model.TOK_NEWPC_BASE + 16:
            name = f"NEWPC_{idx - model.TOK_NEWPC_BASE}"
        elif idx == model.TOK_STEP:
            name = "STEP"
        elif idx == model.TOK_OUT:
            name = "OUT"
        elif idx < 256:
            name = f"BYTE({idx})"
        else:
            name = f"TOK({idx})"
        print(f"  {name}: {prob.item():.4f}")
    print()

    # Generate
    print("Generating...")
    output_ids = model.generate(input_ids, max_tokens=20)
    new_tokens = output_ids[0, input_ids.shape[1]:].tolist()

    print(f"Generated: {new_tokens}")

    # Decode
    token_names = []
    for t in new_tokens:
        if t == model.TOK_NEWAX:
            token_names.append("NEWAX")
        elif t == model.TOK_HALT:
            token_names.append("HALT")
        elif model.TOK_NEWPC_BASE <= t < model.TOK_NEWPC_BASE + 16:
            pc_val = t - model.TOK_NEWPC_BASE
            token_names.append(f"NEWPC_{pc_val}")
        elif t == model.TOK_STEP:
            token_names.append("STEP")
        elif t < 256:
            token_names.append(str(t))
        else:
            token_names.append(f"?{t}")

    print(f"Decoded: {' '.join(token_names)}")
    print()

    # Speculative generation test
    print("Speculative generation...")
    input_ids_spec = create_initial_state(model, pc=0, ax=0)
    output_spec, stats = model.generate_speculative(input_ids_spec, max_tokens=20)
    spec_tokens = output_spec[0, input_ids_spec.shape[1]:].tolist()
    print(f"Generated: {len(spec_tokens)} tokens in {stats['speculative_batches']} batches")
    print()

    # Parameter and FLOPs summary
    params = model.count_parameters()
    flops = model.estimate_flops_per_token()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Parameters: {params['total']:,} total, {params['nonzero']:,} non-zero ({params['sparsity']:.1f}% sparse)")
    print(f"FLOPs per token: {flops['total_per_token']:,}")
    print(f"  Embedding: {flops['embedding']:,}")
    print(f"  Attention: {flops['attention']:,}")
    print(f"  Fetch: {flops['fetch']:,}")
    print(f"  Execute: {flops['execute']:,}")
    print(f"  Output: {flops['output']:,}")
    print(f"  LM Head: {flops['lm_head']:,}")
    print()

    # Check hidden state
    print("=" * 70)
    print("  HIDDEN STATE ANALYSIS")
    print("=" * 70)
    print()

    with torch.no_grad():
        pos = torch.arange(input_ids.shape[1])
        x = model.tok_emb(input_ids) + model.pos_emb(pos)

        print("After embedding (STEP token):")
        h = x[0, -1, :]
        print(f"  Dims 0-7 (markers): {[f'{v:.1f}' for v in h[0:8].tolist()]}")
        print(f"  Dims 8-15 (byte): {[f'{v:.1f}' for v in h[8:16].tolist()]}")

        # After attention
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]) * float('-inf'), diagonal=1)
        q = model.attn_q(x)
        k = model.attn_k(x)
        v = model.attn_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.hidden_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        x = x + model.attn_o(attn_out)

        h = x[0, -1, :]
        print(f"\nAfter attention:")
        print(f"  Dims 16-23 (PC value): {[f'{v:.2f}' for v in h[16:24].tolist()]}")
        print(f"  Dims 24-31 (AX value): {[f'{v:.2f}' for v in h[24:32].tolist()]}")

        # After fetch
        x_norm = model.ln1(x)
        pc_detect = F.relu(model.fetch_pc(x_norm))
        x = x + model.fetch_out(pc_detect)

        h = x[0, -1, :]
        print(f"\nAfter instruction fetch:")
        print(f"  Dims 32-35 (opcode): {[f'{v:.2f}' for v in h[32:36].tolist()]}")
        print(f"  Dims 36-43 (immediate): {[f'{v:.2f}' for v in h[36:44].tolist()]}")

        # Decode immediate
        imm_bits = h[36:44]
        imm_val = sum(int(b > 0.5) << i for i, b in enumerate(imm_bits))
        print(f"  Decoded immediate: {imm_val}")

        # After execute
        x_norm = model.ln2(x)
        gate = F.silu(model.exec_ffn(x_norm))
        up = model.exec_gate(x_norm)
        x = x + model.exec_out(gate * up)

        h = x[0, -1, :]
        print(f"\nAfter execute:")
        print(f"  Dims 44-51 (result): {[f'{v:.2f}' for v in h[44:52].tolist()]}")
        print(f"  Dim 52 (halt flag): {h[52].item():.2f}")

        result_bits = h[44:52]
        result_val = sum(int(b > 0.5) << i for i, b in enumerate(result_bits))
        print(f"  Decoded result: {result_val}")

        # After output selection
        x_norm = model.ln3(x)
        out = F.relu(model.out_ffn(x_norm))
        x = x + model.out_proj(out)

        h = x[0, -1, :]
        print(f"\nAfter output selection:")
        print(f"  Dim 56 (HALT): {h[56].item():.2f}")
        print(f"  Dim 57 (NEWAX): {h[57].item():.2f}")
        print(f"  Dims 68-75 (byte out): {[f'{v:.2f}' for v in h[68:76].tolist()]}")

        out_bits = h[68:76]
        out_val = sum(int(b > 5) << i for i, b in enumerate(out_bits))
        print(f"  Decoded output byte: {out_val}")

    # Now trace after generating NEWAX (second forward pass)
    print("\n" + "=" * 70)
    print("  SECOND FORWARD PASS (after NEWAX)")
    print("=" * 70)

    # Append NEWAX token
    input_with_newax = torch.cat([input_ids, torch.tensor([[model.TOK_NEWAX]])], dim=1)
    print(f"\nTokens: {input_with_newax.squeeze().tolist()}")

    with torch.no_grad():
        logits2 = model.forward(input_with_newax)
        probs2 = F.softmax(logits2[0, -1, :], dim=-1)
        top5_2 = probs2.topk(5)

    print("\nTop 5 predictions after NEWAX:")
    for prob, idx in zip(top5_2.values, top5_2.indices):
        idx = idx.item()
        if idx < 256:
            name = f"BYTE({idx})"
        elif idx == model.TOK_NEWAX:
            name = "NEWAX"
        elif idx == model.TOK_HALT:
            name = "HALT"
        else:
            name = f"TOK({idx})"
        print(f"  {name}: {prob.item():.4f}")

    # Full layer-by-layer trace at NEWAX position
    with torch.no_grad():
        pos = torch.arange(input_with_newax.shape[1])
        x = model.tok_emb(input_with_newax) + model.pos_emb(pos)

        print(f"\nLayer-by-layer trace at NEWAX position:")
        h = x[0, -1, :]
        print(f"  After embedding - Dim 3 (NEWAX marker): {h[3].item():.1f}")

        # Attention
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]) * float('-inf'), diagonal=1)
        q = model.attn_q(x)
        k = model.attn_k(x)
        v = model.attn_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.hidden_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        x = x + model.attn_o(attn_out)

        h = x[0, -1, :]
        print(f"  After attention - Dims 44-51 (result): {[f'{v:.2f}' for v in h[44:52].tolist()]}")

        # Check attention weights - which positions is NEWAX attending to?
        print(f"  Attention weights at NEWAX: {[f'{w:.3f}' for w in weights[0, -1, :].tolist()]}")

        # Fetch
        x_norm = model.ln1(x)
        pc_detect = F.relu(model.fetch_pc(x_norm))
        x = x + model.fetch_out(pc_detect)

        h = x[0, -1, :]
        print(f"  After fetch - Dims 36-43 (imm): {[f'{v:.2f}' for v in h[36:44].tolist()]}")

        # Execute
        x_norm = model.ln2(x)
        gate = F.silu(model.exec_ffn(x_norm))
        up = model.exec_gate(x_norm)
        x = x + model.exec_out(gate * up)

        h = x[0, -1, :]
        print(f"  After execute - Dims 44-51 (result): {[f'{v:.2f}' for v in h[44:52].tolist()]}")

        # Output selection (no LayerNorm)
        out_raw = model.out_ffn(x)
        out = F.relu(out_raw)

        h_in = x[0, -1, :]
        print(f"  Input to out_ffn - Dim 3 (NEWAX): {h_in[3].item():.2f}")
        print(f"  Input to out_ffn - Dims 8-15 (byte): {[f'{v:.2f}' for v in h_in[8:16].tolist()]}")
        print(f"  Input to out_ffn - Dims 44-51 (result): {[f'{v:.2f}' for v in h_in[44:52].tolist()]}")
        print(f"  out_ffn raw [10:18] (route 2): {[f'{v:.2f}' for v in out_raw[0, -1, 10:18].tolist()]}")
        print(f"  out_ffn relu [10:18]: {[f'{v:.2f}' for v in out[0, -1, 10:18].tolist()]}")
        print(f"  out_ffn relu [3] (route 3): {out[0, -1, 3].item():.2f}")

        x = x + model.out_proj(out)

        h = x[0, -1, :]
        print(f"  After output - Dims 68-75 (byte): {[f'{v:.2f}' for v in h[68:76].tolist()]}")

    # Third forward pass: after NEWPC_1 (should output STEP)
    print("\n" + "=" * 70)
    print("  THIRD FORWARD PASS (after NEWPC_1) - should output STEP")
    print("=" * 70)

    # Sequence: PC=0, 0, AX=0, 0, STEP, NEWAX, 42, NEWPC_1
    input_with_newpc = torch.tensor([[
        model.TOK_PC, 0,
        model.TOK_AX, 0,
        model.TOK_STEP,
        model.TOK_NEWAX, 42,
        model.TOK_NEWPC_BASE + 1  # NEWPC_1
    ]])
    print(f"\nTokens: {input_with_newpc.squeeze().tolist()}")

    with torch.no_grad():
        logits3 = model.forward(input_with_newpc)
        probs3 = F.softmax(logits3[0, -1, :], dim=-1)
        top5_3 = probs3.topk(5)

    print("\nTop 5 predictions after NEWPC_1 (should be STEP):")
    for prob, idx in zip(top5_3.values, top5_3.indices):
        idx = idx.item()
        if idx < 256:
            name = f"BYTE({idx})"
        elif idx == model.TOK_NEWAX:
            name = "NEWAX"
        elif idx == model.TOK_HALT:
            name = "HALT"
        elif idx == model.TOK_STEP:
            name = "STEP"
        elif model.TOK_NEWPC_BASE <= idx < model.TOK_NEWPC_BASE + 16:
            name = f"NEWPC_{idx - model.TOK_NEWPC_BASE}"
        else:
            name = f"TOK({idx})"
        print(f"  {name}: {prob.item():.4f}")

    # Check actual logits and dims
    print(f"\nActual logits (from model.forward):")
    print(f"  HALT (277): {logits3[0, -1, model.TOK_HALT].item():.2f}")
    print(f"  STEP (258): {logits3[0, -1, model.TOK_STEP].item():.2f}")

    # Trace dims 56 and 61 through the actual forward pass
    with torch.no_grad():
        pos = torch.arange(input_with_newpc.shape[1])
        x = model.tok_emb(input_with_newpc) + model.pos_emb(pos)
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]) * float('-inf'), diagonal=1)
        q = model.attn_q(x)
        k = model.attn_k(x)
        v = model.attn_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.hidden_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        x = x + model.attn_o(attn_out)
        # Skip ln1 (like actual forward)
        pc_detect = F.relu(model.fetch_pc(x))
        x = x + model.fetch_out(pc_detect)
        # Execute with ln2
        x_norm = model.ln2(x)
        gate = F.silu(model.exec_ffn(x_norm))
        up = model.exec_gate(x_norm)
        x = x + model.exec_out(gate * up)

        print(f"  After execute - dim 52 (halt): {x[0, -1, 52].item():.2f}")

        # Output selection (skip ln3)
        out = F.relu(model.out_ffn(x))
        x = x + model.out_proj(out)

        print(f"  After out_proj - dim 56 (HALT): {x[0, -1, 56].item():.2f}")
        print(f"  After out_proj - dim 61 (STEP): {x[0, -1, 61].item():.2f}")

        # After ln_f
        x = model.ln_f(x)
        print(f"  After ln_f - dim 56 (HALT): {x[0, -1, 56].item():.2f}")
        print(f"  After ln_f - dim 61 (STEP): {x[0, -1, 61].item():.2f}")

    # Trace what's happening at NEWPC position
    with torch.no_grad():
        pos = torch.arange(input_with_newpc.shape[1])
        x = model.tok_emb(input_with_newpc) + model.pos_emb(pos)

        h = x[0, -1, :]
        print(f"\nAt NEWPC position:")
        print(f"  Embedding dims 0-7 (markers): {[f'{v:.1f}' for v in h[0:8].tolist()]}")
        print(f"  Embedding dim 4 (NEWPC): {h[4].item():.2f}")

        # Attention
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]) * float('-inf'), diagonal=1)
        q = model.attn_q(x)
        k = model.attn_k(x)
        v = model.attn_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.hidden_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        x = x + model.attn_o(attn_out)

        h = x[0, -1, :]
        print(f"  After attention - dim 4 (NEWPC): {h[4].item():.2f}")
        print(f"  After attention - dims 80-82 (markers): {[f'{v:.2f}' for v in h[80:83].tolist()]}")

        # Fetch
        x_norm = model.ln1(x)
        pc_detect = F.relu(model.fetch_pc(x_norm))
        x = x + model.fetch_out(pc_detect)

        # Execute
        x_norm = model.ln2(x)
        gate = F.silu(model.exec_ffn(x_norm))
        up = model.exec_gate(x_norm)
        x = x + model.exec_out(gate * up)

        # Output selection
        h = x[0, -1, :]
        print(f"  Input to out_ffn - dim 4 (NEWPC): {h[4].item():.2f}")

        out_raw = model.out_ffn(x)
        out = F.relu(out_raw)

        print(f"  Route 4 raw (out_ffn[4]): {out_raw[0, -1, 4].item():.2f}")
        print(f"  Route 4 relu: {out[0, -1, 4].item():.2f}")
        print(f"  Route 5 raw (out_ffn[5]): {out_raw[0, -1, 5].item():.2f}")
        print(f"  Route 5 relu: {out[0, -1, 5].item():.2f}")

        x = x + model.out_proj(out)

        h = x[0, -1, :]
        print(f"  Dim 61 (STEP output): {h[61].item():.2f}")
        print(f"  Dims 68-75 (byte): {[f'{v:.2f}' for v in h[68:76].tolist()]}")

        # Check logits
        x_final = model.ln_f(x)
        logits_manual = model.lm_head(x_final)

        print(f"\nLogits check:")
        print(f"  Byte 0: {logits_manual[0, -1, 0].item():.2f}")
        print(f"  Byte 1: {logits_manual[0, -1, 1].item():.2f}")
        print(f"  STEP: {logits_manual[0, -1, model.TOK_STEP].item():.2f}")

    # Fourth forward pass: after byte 1 (should output STEP)
    print("\n" + "=" * 70)
    print("  FOURTH FORWARD PASS (after byte 1) - should output STEP")
    print("=" * 70)

    # Sequence: PC=0, 0, AX=0, 0, STEP, NEWAX, 42, NEWPC, 1
    input_after_1 = torch.tensor([[
        model.TOK_PC, 0,
        model.TOK_AX, 0,
        model.TOK_STEP,
        model.TOK_NEWAX, 42,
        model.TOK_NEWPC_BASE + 1
    ]])
    print(f"\nTokens: {input_after_1.squeeze().tolist()}")

    with torch.no_grad():
        logits4 = model.forward(input_after_1)
        probs4 = F.softmax(logits4[0, -1, :], dim=-1)
        top5_4 = probs4.topk(5)

    print("\nTop 5 predictions after byte 1 (should be STEP):")
    for prob, idx in zip(top5_4.values, top5_4.indices):
        idx = idx.item()
        if idx < 256:
            name = f"BYTE({idx})"
        elif idx == model.TOK_NEWAX:
            name = "NEWAX"
        elif idx == model.TOK_HALT:
            name = "HALT"
        elif idx == model.TOK_STEP:
            name = "STEP"
        elif model.TOK_NEWPC_BASE <= idx < model.TOK_NEWPC_BASE + 16:
            name = f"NEWPC_{idx - model.TOK_NEWPC_BASE}"
        else:
            name = f"TOK({idx})"
        print(f"  {name}: {prob.item():.4f}")

    # Trace at byte 1 position
    with torch.no_grad():
        pos = torch.arange(input_after_1.shape[1])
        x = model.tok_emb(input_after_1) + model.pos_emb(pos)

        h = x[0, -1, :]
        print(f"\nAt byte 1 position:")
        print(f"  Embedding dims 0-7 (markers): {[f'{v:.1f}' for v in h[0:8].tolist()]}")
        print(f"  Embedding dims 8-15 (byte): {[f'{v:.1f}' for v in h[8:16].tolist()]}")

        # Attention
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]) * float('-inf'), diagonal=1)
        q = model.attn_q(x)
        k = model.attn_k(x)
        v = model.attn_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.hidden_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        x = x + model.attn_o(attn_out)

        print(f"  Attention weights: {[f'{w:.3f}' for w in weights[0, -1, :].tolist()]}")
        h = x[0, -1, :]
        print(f"  After attention - dims 80-82 (markers): {[f'{v:.2f}' for v in h[80:83].tolist()]}")

        # Fetch and execute
        x_norm = model.ln1(x)
        pc_detect = F.relu(model.fetch_pc(x_norm))
        x = x + model.fetch_out(pc_detect)
        x_norm = model.ln2(x)
        gate = F.silu(model.exec_ffn(x_norm))
        up = model.exec_gate(x_norm)
        x = x + model.exec_out(gate * up)

        # Output selection
        h = x[0, -1, :]
        print(f"  Input to out_ffn - dims 8-15 (byte): {[f'{v:.2f}' for v in h[8:16].tolist()]}")
        print(f"  Input to out_ffn - dims 80-82: {[f'{v:.2f}' for v in h[80:83].tolist()]}")

        out_raw = model.out_ffn(x)
        out = F.relu(out_raw)

        print(f"  Route 3 raw: {out_raw[0, -1, 3].item():.2f}")
        print(f"  Route 5 raw: {out_raw[0, -1, 5].item():.2f}")
        print(f"  Route 2 (byte outputs) raw: {[f'{v:.2f}' for v in out_raw[0, -1, 10:18].tolist()]}")

    # Fifth forward pass: second STEP (after full cycle, PC should be 1)
    print("\n" + "=" * 70)
    print("  FIFTH FORWARD PASS (second STEP) - should HALT at PC=1")
    print("=" * 70)

    # Sequence after full cycle: PC=0, 0, AX=0, 0, STEP, NEWAX, 42, NEWPC, 1, STEP
    input_second_step = torch.tensor([[
        model.TOK_PC, 0,
        model.TOK_AX, 0,
        model.TOK_STEP,
        model.TOK_NEWAX, 42,
        model.TOK_NEWPC_BASE + 1,
        model.TOK_STEP
    ]])
    print(f"\nTokens: {input_second_step.squeeze().tolist()}")

    with torch.no_grad():
        logits5 = model.forward(input_second_step)
        probs5 = F.softmax(logits5[0, -1, :], dim=-1)
        top5_5 = probs5.topk(5)

    print("\nTop 5 predictions at second STEP (should be HALT):")
    for prob, idx in zip(top5_5.values, top5_5.indices):
        idx = idx.item()
        if idx < 256:
            name = f"BYTE({idx})"
        elif idx == model.TOK_NEWAX:
            name = "NEWAX"
        elif idx == model.TOK_HALT:
            name = "HALT"
        elif idx == model.TOK_STEP:
            name = "STEP"
        elif model.TOK_NEWPC_BASE <= idx < model.TOK_NEWPC_BASE + 16:
            name = f"NEWPC_{idx - model.TOK_NEWPC_BASE}"
        else:
            name = f"TOK({idx})"
        print(f"  {name}: {prob.item():.4f}")

    # Trace at second STEP position
    with torch.no_grad():
        pos = torch.arange(input_second_step.shape[1])
        x = model.tok_emb(input_second_step) + model.pos_emb(pos)

        h = x[0, -1, :]
        print(f"\nAt second STEP position:")
        print(f"  Embedding dims 0-7 (markers): {[f'{v:.1f}' for v in h[0:8].tolist()]}")

        # Attention
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]) * float('-inf'), diagonal=1)
        q = model.attn_q(x)
        k = model.attn_k(x)
        v = model.attn_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(model.hidden_dim)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, v)
        x = x + model.attn_o(attn_out)

        print(f"  Attention weights: {[f'{w:.3f}' for w in weights[0, -1, :].tolist()]}")
        h = x[0, -1, :]
        print(f"  After attention - dims 16-23 (PC): {[f'{v:.2f}' for v in h[16:24].tolist()]}")

        # Fetch
        x_norm = model.ln1(x)
        pc_detect = F.relu(model.fetch_pc(x_norm))
        print(f"  PC detection (before fetch): {[f'{v:.2f}' for v in pc_detect[0, -1, :4].tolist()]}")

        x = x + model.fetch_out(pc_detect)
        h = x[0, -1, :]
        print(f"  After fetch - dims 32-35 (opcode): {[f'{v:.2f}' for v in h[32:36].tolist()]}")
        print(f"  After fetch - dims 36-43 (immediate): {[f'{v:.2f}' for v in h[36:44].tolist()]}")

        # Execute
        x_norm = model.ln2(x)
        gate = F.silu(model.exec_ffn(x_norm))
        up = model.exec_gate(x_norm)
        x = x + model.exec_out(gate * up)

        h = x[0, -1, :]
        print(f"  After execute - dim 52 (halt flag): {h[52].item():.2f}")


def test_all_programs():
    """Test all programs (IMM, ADD, SUB, MUL)."""
    print("\n" + "=" * 70)
    print("  TESTING ALL PROGRAMS")
    print("=" * 70)

    # List of (program, expected_result, description)
    test_cases = [
        # IMM
        ([(WeightBakedVM.OP_IMM, 42), (WeightBakedVM.OP_RET, 0)], 42, "IMM 42"),
        ([(WeightBakedVM.OP_IMM, 0), (WeightBakedVM.OP_RET, 0)], 0, "IMM 0"),
        ([(WeightBakedVM.OP_IMM, 255), (WeightBakedVM.OP_RET, 0)], 255, "IMM 255"),

        # ADD
        ([(WeightBakedVM.OP_IMM, 10), (WeightBakedVM.OP_ADD, 5), (WeightBakedVM.OP_RET, 0)], 15, "IMM 10; ADD 5"),
        ([(WeightBakedVM.OP_IMM, 0), (WeightBakedVM.OP_ADD, 100), (WeightBakedVM.OP_RET, 0)], 100, "IMM 0; ADD 100"),

        # SUB
        ([(WeightBakedVM.OP_IMM, 20), (WeightBakedVM.OP_SUB, 8), (WeightBakedVM.OP_RET, 0)], 12, "IMM 20; SUB 8"),
        ([(WeightBakedVM.OP_IMM, 100), (WeightBakedVM.OP_SUB, 50), (WeightBakedVM.OP_RET, 0)], 50, "IMM 100; SUB 50"),

        # MUL
        ([(WeightBakedVM.OP_IMM, 6), (WeightBakedVM.OP_MUL, 7), (WeightBakedVM.OP_RET, 0)], 42, "IMM 6; MUL 7"),
        ([(WeightBakedVM.OP_IMM, 3), (WeightBakedVM.OP_MUL, 4), (WeightBakedVM.OP_RET, 0)], 12, "IMM 3; MUL 4"),
        ([(WeightBakedVM.OP_IMM, 2), (WeightBakedVM.OP_MUL, 2), (WeightBakedVM.OP_RET, 0)], 4, "IMM 2; MUL 2"),
    ]

    passed = 0
    failed = 0

    for program, expected, desc in test_cases:
        model = WeightBakedVM(hidden_dim=128, program=program)
        input_ids = create_initial_state(model, pc=0, ax=0)

        # Generate output
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_tokens=50)

        new_tokens = output_ids[0, input_ids.shape[1]:].tolist()

        # Extract result from generation
        result = None
        for i, t in enumerate(new_tokens):
            if t == model.TOK_NEWAX and i + 1 < len(new_tokens):
                # Get the most recent NEWAX value
                result = new_tokens[i + 1]

        # Check if halted
        halted = model.TOK_HALT in new_tokens

        if result == expected and halted:
            print(f"  [PASS] {desc} = {result}")
            passed += 1
        else:
            print(f"  [FAIL] {desc} = {result} (expected {expected}, halted={halted})")
            print(f"         Tokens: {new_tokens}")
            failed += 1

    print()
    print(f"Results: {passed}/{passed + failed} passed")

    return passed, failed


if __name__ == "__main__":
    main()

    # Run all test programs
    passed, failed = test_all_programs()

    if failed > 0:
        print("\nSome tests failed. Debugging...")
