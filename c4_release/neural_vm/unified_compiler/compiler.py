"""
Unified VM Compiler - Main orchestrator.

Coordinates embedding, attention, and FFN compilers to generate
all weights for the Neural VM transformer.
"""

import torch
from typing import Optional

from .primitives import Primitives, P
from ..vm_step import _SetDim as BD, Token
from ..embedding import Opcode


class UnifiedVMCompiler:
    """Main compiler for Neural VM weights.

    Produces identical weights to vm_step.py's set_vm_weights() function.

    Usage:
        compiler = UnifiedVMCompiler()
        compiler.compile(model)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 17,
        n_heads: int = 8,
        ffn_hidden: int = 4096,
        alu_mode: str = 'lookup',
        scale: float = 100.0,
    ):
        """
        Args:
            d_model: Model dimension (default 512)
            n_layers: Number of transformer layers (default 17)
            n_heads: Number of attention heads (default 8)
            ffn_hidden: FFN hidden dimension (default 4096)
            alu_mode: 'lookup' or 'efficient' ALU mode
            scale: SwiGLU scale factor (default 100.0)
        """
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_hidden = ffn_hidden
        self.alu_mode = alu_mode
        self.S = scale
        self.HD = d_model // n_heads  # 64

    @torch.no_grad()
    def compile(
        self,
        model,
        enable_tool_calling: bool = False,
        enable_conversational_io: bool = False,
    ):
        """Compile all weights into model.

        Args:
            model: AutoregressiveVM model instance
            enable_tool_calling: Enable tool calling mode
            enable_conversational_io: Enable conversational I/O mode
        """
        # Import layer specs when available
        # from .layer_specs import LAYER_SPECS

        # Phase 1: Embedding
        self._compile_embedding(model)

        # Phase 2: Attention layers
        self._compile_attention_layers(model)

        # Phase 3: FFN layers
        self._compile_ffn_layers(model)

        # Phase 4: Output head
        self._compile_output_head(model)

    def _compile_embedding(self, model):
        """Compile embedding weights."""
        embed = model.embed.embed.weight
        embed.zero_()

        V = model.vocab_size

        # CONST dimension for all tokens
        for tok in range(V):
            embed[tok, BD.CONST] = 1.0

        # Marker tokens
        marker_dims = [
            (Token.REG_PC, BD.MARK_PC),
            (Token.REG_AX, BD.MARK_AX),
            (Token.REG_SP, BD.MARK_SP),
            (Token.REG_BP, BD.MARK_BP),
            (Token.MEM, BD.MARK_MEM),
            (Token.CODE_START, BD.MARK_CS),
        ]

        for tok, dim in marker_dims:
            embed[tok, dim] = 1.0
            embed[tok, BD.IS_MARK] = 1.0

        # STACK0 marker (NOT IS_MARK to avoid blocking BP from threshold detection)
        embed[Token.STACK0, BD.MARK_STACK0] = 1.0

        # Step-end tokens
        for tok in [Token.STEP_END, Token.DATA_END, Token.HALT]:
            embed[tok, BD.MARK_SE] = 1.0
            embed[tok, BD.IS_MARK] = 1.0

        embed[Token.STEP_END, BD.MARK_SE_ONLY] = 1.0

        # TOOL_CALL has same profile as STEP_END
        embed[Token.TOOL_CALL, BD.MARK_SE] = 1.0
        embed[Token.TOOL_CALL, BD.IS_MARK] = 1.0
        embed[Token.TOOL_CALL, BD.MARK_SE_ONLY] = 1.0
        embed[Token.TOOL_CALL, BD.CONST] = 1.0

        # I/O markers
        embed[Token.USER_INPUT_START, BD.IS_MARK] = 1.0
        embed[Token.USER_INPUT_END, BD.IS_MARK] = 1.0

        # Thinking tags
        embed[Token.THINKING_START, BD.IS_MARK] = 1.0
        embed[Token.THINKING_START, BD.CONST] = 1.0
        embed[Token.THINKING_START, BD.TEMP + 1] = 1.0
        embed[Token.THINKING_END, BD.IS_MARK] = 1.0
        embed[Token.THINKING_END, BD.CONST] = 1.0
        embed[Token.THINKING_END, BD.TEMP + 2] = 1.0

        # I/O state tokens
        embed[Token.IO_STATE_EMIT_BYTE, BD.IS_MARK] = 1.0
        embed[Token.IO_STATE_EMIT_BYTE, BD.CONST] = 1.0
        embed[Token.IO_STATE_EMIT_THINKING, BD.IS_MARK] = 1.0
        embed[Token.IO_STATE_EMIT_THINKING, BD.CONST] = 1.0

        # Byte embeddings (0-255)
        for b in range(256):
            embed[b, BD.IS_BYTE] = 1.0
            embed[b, BD.EMBED_LO + (b & 0xF)] = 1.0
            embed[b, BD.EMBED_HI + ((b >> 4) & 0xF)] = 1.0
            # Clean copies (never written by attention/FFN)
            embed[b, BD.CLEAN_EMBED_LO + (b & 0xF)] = 1.0
            embed[b, BD.CLEAN_EMBED_HI + ((b >> 4) & 0xF)] = 1.0

    def _compile_attention_layers(self, model):
        """Compile attention weights for all layers."""
        # L0: Threshold attention (8 heads)
        self._compile_l0_attention(model.blocks[0].attn)

        # L1-L2: Fine thresholds
        self._compile_l1_attention(model.blocks[1].attn)
        self._compile_l2_attention(model.blocks[2].attn)

        # L3-L6: Carry-forward and relay heads
        self._compile_l3_attention(model.blocks[3].attn)
        self._compile_l4_attention(model.blocks[4].attn)
        self._compile_l5_attention(model.blocks[5].attn)
        self._compile_l6_attention(model.blocks[6].attn)

        # L7: Operand gather + memory relay heads
        self._compile_l7_attention(model.blocks[7].attn)

        # L8: SP gather → STACK0 positions
        self._compile_l8_attention(model.blocks[8].attn)

        # L9: LEV address relay
        self._compile_l9_attention(model.blocks[9].attn)

        # L10: Carry relay + AX/SP byte passthrough
        self._compile_l10_attention(model.blocks[10].attn)

        # L11-L12: Passthrough (no attention weights needed)

        # L13: MEM addr gather
        self._compile_l13_attention(model.blocks[13].attn)

        # L14: MEM byte generation
        self._compile_l14_attention(model.blocks[14].attn)

        # L15: Memory lookup (extended to 12 heads)
        if len(model.blocks) > 16:
            self._resize_l15_attention(model.blocks[15].attn, model.d_model)
        self._compile_l15_attention(model.blocks[15].attn)

    def _compile_l0_attention(self, attn):
        """Compile L0 threshold attention (8 heads)."""
        thresholds = [3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5]
        out_bases = [BD.H0, BD.H1, BD.H2, BD.H3, BD.H4, BD.H5, BD.H6, BD.H7]

        # Set ALiBi slopes
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)

        for i, (t, out) in enumerate(zip(thresholds, out_bases)):
            P.threshold_attention(
                attn, head_idx=i, threshold=t, out_base=out,
                slope=ALIBI_S, HD=self.HD
            )

    def _compile_l1_attention(self, attn):
        """Compile L1 fine threshold attention."""
        ALIBI_S = 10.0

        # Set ALiBi slopes
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
            attn.alibi_slopes[3] = 0.0  # Head 3: global attention for SE detection

        # Heads 0-2: threshold attention (0.5, 1.5, 2.5)
        thresholds = [0.5, 1.5, 2.5]
        out_bases = [BD.L1H0, BD.L1H1, BD.L1H2]

        for i, (t, out) in enumerate(zip(thresholds, out_bases)):
            P.threshold_attention(
                attn, head_idx=i, threshold=t, out_base=out,
                slope=ALIBI_S, HD=self.HD
            )

        # Head 3: STEP_END existence detection (global, no ALiBi slope)
        base = 3 * self.HD
        attn.W_q.data[base, BD.CONST] = 10.0
        attn.W_k.data[base, BD.MARK_SE_ONLY] = 10.0
        attn.W_v.data[base + 1, BD.MARK_SE_ONLY] = 1.0
        attn.W_o.data[BD.HAS_SE, base + 1] = 1.0

        # Head 4: threshold 6.5 for STACK0 byte 0 identification
        P.threshold_attention(
            attn, head_idx=4, threshold=6.5, out_base=BD.L1H4,
            slope=ALIBI_S, HD=self.HD
        )

    def _compile_l2_attention(self, attn):
        """Compile L2 attention (threshold 5.5)."""
        ALIBI_S = 10.0

        # Set ALiBi slopes
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)

        # Head 0: threshold 5.5 (L2H0)
        P.threshold_attention(
            attn, head_idx=0, threshold=5.5, out_base=BD.L2H0,
            slope=ALIBI_S, HD=self.HD
        )

    def _compile_l3_attention(self, attn):
        """Compile L3 carry-forward attention (PC, AX, SP, BP, STACK0)."""
        PC_I, AX_I, SP_I, BP_I = 0, 1, 2, 3
        L = 15.0

        # Set ALiBi slopes
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)

        # Head 0: PC carry (prev step PC byte 0 -> EMBED at PC marker)
        P.carry_forward_attention(
            attn, head_idx=0,
            marker_dim=BD.MARK_PC, l1h1_idx=PC_I, l1h0_idx=PC_I,
            out_lo=BD.EMBED_LO, out_hi=BD.EMBED_HI,
            HD=self.HD
        )

        # Head 1: AX carry (prev step AX byte 0 EMBED -> AX_CARRY staging)
        # Uses EMBED as source (not OUTPUT) - byte tokens embed to their value
        P.carry_forward_attention(
            attn, head_idx=1,
            marker_dim=BD.MARK_AX, l1h1_idx=AX_I, l1h0_idx=AX_I,
            out_lo=BD.AX_CARRY_LO, out_hi=BD.AX_CARRY_HI,
            HD=self.HD
        )

        # Head 2: SP carry (prev step SP byte 0 -> EMBED at SP marker)
        P.carry_forward_attention(
            attn, head_idx=2,
            marker_dim=BD.MARK_SP, l1h1_idx=SP_I, l1h0_idx=SP_I,
            out_lo=BD.EMBED_LO, out_hi=BD.EMBED_HI,
            HD=self.HD
        )

        # Head 3: BP carry (prev step BP byte 0 -> EMBED at BP marker)
        P.carry_forward_attention(
            attn, head_idx=3,
            marker_dim=BD.MARK_BP, l1h1_idx=BP_I, l1h0_idx=BP_I,
            out_lo=BD.EMBED_LO, out_hi=BD.EMBED_HI,
            HD=self.HD
        )

        # Head 4: STACK0 carry (prev step STACK0 byte 0 -> EMBED at STACK0 marker)
        # Uses STACK0_BYTE0 flag (computed in L1 FFN) as key
        base = 4 * self.HD
        attn.W_q.data[base, BD.MARK_STACK0] = L
        attn.W_k.data[base, BD.STACK0_BYTE0] = L
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.EMBED_HI + k] = 1.0
        for k in range(16):
            attn.W_o.data[BD.EMBED_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.EMBED_HI + k, base + 17 + k] = 1.0
        # Anti-leakage gate
        GATE = 33
        attn.W_q.data[base + GATE, BD.MARK_STACK0] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

        # Head 5: AX full value relay (prev AX marker OUTPUT -> current AX marker AX_FULL)
        base = 5 * self.HD
        # Q: Fire at AX marker on subsequent steps only (HAS_SE=1)
        attn.W_q.data[base, BD.MARK_AX] = L
        attn.W_q.data[base, BD.HAS_SE] = L
        attn.W_q.data[base, BD.CONST] = -L * 1.5  # Threshold: need both
        # K: Match previous step's AX marker
        attn.W_k.data[base, BD.MARK_AX] = L
        # V: Copy OUTPUT_LO/HI from previous AX marker
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
        # O: Write to AX_FULL_LO/HI
        for k in range(16):
            attn.W_o.data[BD.AX_FULL_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.AX_FULL_HI + k, base + 17 + k] = 1.0
        # Anti-leakage gate
        GATE = 33
        attn.W_q.data[base + GATE, BD.MARK_AX] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

        # Head 6: BP carry to PC marker for LEV return_addr lookup
        # When OP_LEV is active, copy prev step's BP byte 0 to OUTPUT at PC marker
        base = 6 * self.HD
        # Q: Fire at PC marker when OP_LEV active
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.OP_LEV] = L / 5  # OP_LEV ≈ 5, normalize to ~L
        attn.W_q.data[base, BD.CONST] = -L * 1.5  # Need both MARK_PC and OP_LEV
        # K: Attend to PREVIOUS step's BP byte 0 (L1H1[BP_I]=1 AND NOT L1H0[BP_I])
        attn.W_k.data[base, BD.L1H1 + BP_I] = L
        attn.W_k.data[base, BD.L1H0 + BP_I] = -L  # Suppress current step's BP byte 0
        # V: Copy CLEAN_EMBED_LO/HI (prev step's BP byte 0 value)
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_PC] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

    def _compile_l4_attention(self, attn):
        """Compile L4 attention (PC value relay to AX marker).

        Head 0: AX marker reads the PC MARKER's EMBED_LO/HI, which was
        populated by L3 carry-forward from the previous step's PC byte 0.
        """
        L = 15.0

        # Set ALiBi slopes
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)

        base = 0 * self.HD
        # Q: active at AX markers
        attn.W_q.data[base, BD.MARK_AX] = L
        # K: target PC marker
        attn.W_k.data[base, BD.MARK_PC] = L
        # V: copies EMBED_LO and EMBED_HI (OLD PC from L3 carry-forward)
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.EMBED_HI + k] = 1.0
        # O: writes to EMBED_LO and EMBED_HI at AX marker
        for k in range(16):
            attn.W_o.data[BD.EMBED_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.EMBED_HI + k, base + 17 + k] = 1.0

        # Anti-leakage gate
        GATE = 33
        attn.W_q.data[base + GATE, BD.MARK_AX] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

    def _compile_l5_attention(self, attn):
        """Compile L5 attention (opcode/immediate fetch through memory keys).

        Head 0: fetch immediate byte at address PC+1 (TEMP[0..31])
        Head 1: fetch opcode byte at address PC (EMBED_LO/HI)
        Head 2: fetch opcode for first-step (PC marker -> address PC_OFFSET)
        Head 3: fetch immediate for first-step (PC marker -> address PC_OFFSET+1)
        Head 4: fetch opcode to AX marker for first-step
        Head 6: direct OP_* flag relay from CODE to PC marker
        Head 7: direct OP_* flag relay from CODE to AX marker
        """
        from ..constants import PC_OFFSET
        L = 20.0
        GATE = 33
        HAS_SE_GATE = 34

        # Head 0: fetch immediate byte (address = PC+1) - non-first steps
        base = 0 * self.HD
        # Q: low two nibbles from TEMP (PC+1)
        for k in range(16):
            attn.W_q.data[base + k, BD.TEMP + k] = L
            attn.W_q.data[base + 16 + k, BD.TEMP + 16 + k] = L
        attn.W_q.data[base + 32, BD.MARK_AX] = L  # third nibble gate
        # K: address nibbles from memory key space
        for k in range(16):
            attn.W_k.data[base + k, BD.ADDR_KEY + k] = L
            attn.W_k.data[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k.data[base + 32, BD.ADDR_KEY + 32] = L
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_AX] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # HAS_SE gate: only fire on non-first steps
        attn.W_q.data[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
        attn.W_q.data[base + HAS_SE_GATE, BD.CONST] = -500.0
        attn.W_k.data[base + HAS_SE_GATE, BD.CONST] = 5.0
        # V: copy byte value nibbles
        for k in range(16):
            attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write immediate to FETCH_LO/HI
        for k in range(16):
            attn.W_o.data[BD.FETCH_LO + k, base + 32 + k] = 1.0
            attn.W_o.data[BD.FETCH_HI + k, base + 48 + k] = 1.0

        # Head 1: fetch opcode byte (address = PC) - non-first steps
        base = 1 * self.HD
        # Q: low two nibbles from EMBED (PC)
        for k in range(16):
            attn.W_q.data[base + k, BD.EMBED_LO + k] = L
            attn.W_q.data[base + 16 + k, BD.EMBED_HI + k] = L
        attn.W_q.data[base + 32, BD.MARK_AX] = L
        # K: address nibbles from memory key space
        for k in range(16):
            attn.W_k.data[base + k, BD.ADDR_KEY + k] = L
            attn.W_k.data[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k.data[base + 32, BD.ADDR_KEY + 32] = L
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_AX] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # HAS_SE gate
        attn.W_q.data[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
        attn.W_q.data[base + HAS_SE_GATE, BD.CONST] = -500.0
        attn.W_k.data[base + HAS_SE_GATE, BD.CONST] = 5.0
        # V: opcode byte nibbles
        for k in range(16):
            attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: writes to OPCODE_BYTE_LO/HI
        for k in range(16):
            attn.W_o.data[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
            attn.W_o.data[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

        # Head 2: fetch opcode for first-step (PC marker -> address PC_OFFSET)
        base = 2 * self.HD
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.HAS_SE] = -L  # only on first step
        attn.W_q.data[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
        attn.W_q.data[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
        attn.W_q.data[base + 32, BD.MARK_PC] = L  # third nibble gate
        # K: match ADDR_KEY nibbles
        for k in range(16):
            attn.W_k.data[base + k, BD.ADDR_KEY + k] = L
            attn.W_k.data[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k.data[base + 32, BD.ADDR_KEY + 32] = L
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_PC] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # V: copy opcode byte nibbles
        for k in range(16):
            attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to OPCODE_BYTE_LO/HI at PC marker
        for k in range(16):
            attn.W_o.data[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
            attn.W_o.data[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

        # Head 3: fetch immediate for first-step (PC marker -> address PC_OFFSET+1)
        imm_addr = PC_OFFSET + 1
        base = 3 * self.HD
        attn.W_q.data[base + (imm_addr & 0xF), BD.CONST] = L  # lo nibble
        attn.W_q.data[base + 16 + ((imm_addr >> 4) & 0xF), BD.CONST] = L  # hi nibble
        attn.W_q.data[base + 32, BD.MARK_PC] = L  # gate for PC marker only
        # K: match ADDR_KEY nibbles
        for k in range(16):
            attn.W_k.data[base + k, BD.ADDR_KEY + k] = L
            attn.W_k.data[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k.data[base + 32, BD.ADDR_KEY + 32] = L
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_PC] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # V: copy immediate byte nibbles
        for k in range(16):
            attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to FETCH_LO/HI (amplified for relay)
        for k in range(16):
            attn.W_o.data[BD.FETCH_LO + k, base + 32 + k] = 40.0
            attn.W_o.data[BD.FETCH_HI + k, base + 48 + k] = 40.0

        # Head 4: fetch opcode to AX marker for first-step
        base = 4 * self.HD
        attn.W_q.data[base, BD.MARK_AX] = L
        attn.W_q.data[base, BD.HAS_SE] = -L  # only on first step
        attn.W_q.data[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
        attn.W_q.data[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
        attn.W_q.data[base + 32, BD.MARK_AX] = L  # third nibble gate
        # K: match ADDR_KEY nibbles
        for k in range(16):
            attn.W_k.data[base + k, BD.ADDR_KEY + k] = L
            attn.W_k.data[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k.data[base + 32, BD.ADDR_KEY + 32] = L
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_AX] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # HAS_SE gate: only fire on first step
        attn.W_q.data[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # negative: block when HAS_SE > 0
        attn.W_k.data[base + HAS_SE_GATE, BD.CONST] = 5.0
        # V: copy opcode byte nibbles
        for k in range(16):
            attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write OPCODE_BYTE_LO/HI at AX marker
        for k in range(16):
            attn.W_o.data[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
            attn.W_o.data[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

        # OP_* flags to relay (for heads 6, 7)
        OP_FLAGS_PC = [
            BD.OP_IMM, BD.OP_LEA, BD.OP_EXIT, BD.OP_JMP, BD.OP_JSR,
            BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
            BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_EQ, BD.OP_LT,
            BD.OP_SHL, BD.OP_SHR,
        ]

        # Head 5: BP EMBED → TEMP at STACK0 marker (for ENT: STACK0 = old_BP)
        # Distance d=5 (STACK0 at pos 20, BP at pos 15 in same step)
        base = 5 * self.HD
        attn.W_q.data[base, BD.MARK_STACK0] = L
        attn.W_k.data[base, BD.MARK_BP] = L
        # Anti-leakage gate: only fire at STACK0 marker positions
        attn.W_q.data[base + GATE, BD.MARK_STACK0] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # V: copy EMBED_LO/HI
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.EMBED_HI + k] = 1.0
        # O: write to TEMP[0..15] and TEMP[16..31]
        for k in range(16):
            attn.W_o.data[BD.TEMP + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.TEMP + 16 + k, base + 17 + k] = 1.0

        # Head 6: direct OP_* flag relay from CODE to PC marker (non-first steps)
        base = 6 * self.HD
        # Q: PC marker, address from EMBED_LO/HI (current PC)
        for k in range(16):
            attn.W_q.data[base + k, BD.EMBED_LO + k] = L
            attn.W_q.data[base + 16 + k, BD.EMBED_HI + k] = L
        attn.W_q.data[base + 32, BD.MARK_PC] = L
        # K: match ADDR_KEY
        for k in range(16):
            attn.W_k.data[base + k, BD.ADDR_KEY + k] = L
            attn.W_k.data[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k.data[base + 32, BD.ADDR_KEY + 32] = L
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_PC] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # HAS_SE gate: only fire on non-first steps
        attn.W_q.data[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
        attn.W_q.data[base + HAS_SE_GATE, BD.CONST] = -500.0
        attn.W_k.data[base + HAS_SE_GATE, BD.CONST] = 5.0
        # V: copy OP_* flags (from embedding)
        for i, op_dim in enumerate(OP_FLAGS_PC):
            attn.W_v.data[base + i, op_dim] = 1.0
        # O: write OP_* flags
        for i, op_dim in enumerate(OP_FLAGS_PC):
            attn.W_o.data[op_dim, base + i] = 1.0

        # Head 6 (additional): SP EMBED → TEMP at BP marker (for ENT: BP = old_SP - 8)
        # This runs after the OP_* relay setup, adding ENT-specific weights
        # Distance d=5 (BP at pos 15, SP at pos 10)
        attn.W_q.data[base, BD.MARK_BP] = L
        attn.W_k.data[base, BD.MARK_SP] = L
        # Gate already set: GATE at MARK_PC with CONST (from OP_* relay)
        # Add anti-leakage for MARK_BP
        attn.W_q.data[base + GATE, BD.MARK_BP] = 500.0
        # OP_ENT gate: only fire when ENT opcode is active (prevents TEMP pollution)
        ENT_GATE = 34
        attn.W_q.data[base + ENT_GATE, BD.OP_ENT] = 500.0
        attn.W_q.data[base + ENT_GATE, BD.CONST] = -500.0
        attn.W_k.data[base + ENT_GATE, BD.CONST] = 5.0
        # V: copy EMBED_LO/HI (at offsets 1-16, 17-32 for ENT)
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.EMBED_HI + k] = 1.0
        # O: write to TEMP[0..15] and TEMP[16..31]
        for k in range(16):
            attn.W_o.data[BD.TEMP + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.TEMP + 16 + k, base + 17 + k] = 1.0

        # Head 7: direct OP_* flag relay from CODE to PC marker (first step only)
        base = 7 * self.HD
        # Q: PC marker when NOT HAS_SE, queries for address PC_OFFSET
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.HAS_SE] = -L  # only on first step
        attn.W_q.data[base + (PC_OFFSET & 0xF), BD.CONST] = L  # lo nibble
        attn.W_q.data[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = L  # hi nibble
        attn.W_q.data[base + 32, BD.MARK_PC] = L
        # K: match ADDR_KEY
        for k in range(16):
            attn.W_k.data[base + k, BD.ADDR_KEY + k] = L
            attn.W_k.data[base + 16 + k, BD.ADDR_KEY + 16 + k] = L
        attn.W_k.data[base + 32, BD.ADDR_KEY + 32] = L
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_PC] = 500.0
        attn.W_q.data[base + GATE, BD.CONST] = -500.0
        attn.W_k.data[base + GATE, BD.CONST] = 5.0
        # V: copy OP_* flags
        for i, op_dim in enumerate(OP_FLAGS_PC):
            attn.W_v.data[base + i, op_dim] = 1.0
        # O: write OP_* flags
        for i, op_dim in enumerate(OP_FLAGS_PC):
            attn.W_o.data[op_dim, base + i] = 1.0

    def _compile_l6_attention(self, attn):
        """Compile L6 attention (JMP/EXIT relay + first-step relays + PSH relay).

        Head 0: JMP relay (PC ← prev AX's OP_JMP + FETCH)
        Head 1: EXIT relay (NEXT_SE ← AX's OP_EXIT)
        Head 2: First-step JMP relay (PC self-attention)
        Head 3: JSR relay (PC ← AX's OP_JSR, first step only)
        Head 4: BZ/BNZ relay (PC ← AX's OP_BZ/BNZ + AX_CARRY + FETCH)
        Head 5: First-step OP flag relay + FETCH relay (AX ← PC's OP_* flags + FETCH)
        Head 6: STACK0 ← AX LO nibble (AX_CARRY_LO → ALU_LO)
        Head 7: STACK0 ← AX HI nibble (AX_CARRY_HI → ALU_HI)
        """
        L = 50.0

        # Head 0: JMP relay (PC marker → previous step's AX marker)
        # FIX 2026-04-16: Added HAS_SE gating so Head 0 only fires on step 1+.
        base = 0 * self.HD
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.MARK_AX] = -L  # block at AX marker
        # Strong anti-leakage gate: Q must be very negative for step 0 (no HAS_SE)
        attn.W_q.data[base, BD.HAS_SE] = L * 20  # +1000 when HAS_SE
        attn.W_q.data[base, BD.CONST] = -L * 20  # -1000 baseline
        attn.W_k.data[base, BD.MARK_AX] = L
        # FIX 2026-04-16: Add CONST to K so the Q gate actually works.
        attn.W_k.data[base, BD.CONST] = 1.0
        # V: copy OP_JMP flag, FETCH_LO/HI
        attn.W_v.data[base + 1, BD.OP_JMP] = 1.0
        for k in range(16):
            attn.W_v.data[base + 2 + k, BD.FETCH_LO + k] = 1.0
            attn.W_v.data[base + 18 + k, BD.FETCH_HI + k] = 1.0
        # O: write IS_JMP to CMP[0], JMP target to AX_CARRY
        attn.W_o.data[BD.CMP + 0, base + 1] = 1.0
        for k in range(16):
            attn.W_o.data[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
            attn.W_o.data[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0

        # Head 1: EXIT relay (NEXT_SE → current AX marker)
        base = 1 * self.HD
        attn.W_q.data[base, BD.NEXT_SE] = L
        attn.W_q.data[base, BD.MARK_AX] = -L  # block at AX marker
        attn.W_k.data[base, BD.MARK_AX] = L
        # V: copy OP_EXIT flag (scaled)
        attn.W_v.data[base + 1, BD.OP_EXIT] = 0.2
        # O: write IS_EXIT to CMP[1]
        attn.W_o.data[BD.CMP + 1, base + 1] = 1.0

        # Head 2: First-step JMP relay (PC marker self-attention)
        # FIX 2026-04-16: Strong anti-leakage gate for non-JMP operations.
        base = 2 * self.HD
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.HAS_SE] = -L  # only fire when NOT HAS_SE
        attn.W_q.data[base, BD.MARK_AX] = -L
        # Strong gate: Q = 50 - HAS_SE(50) - AX(50) + OP_JMP(1000) - CONST(1000)
        attn.W_q.data[base, BD.OP_JMP] = L * 20  # +1000 when OP_JMP active
        attn.W_q.data[base, BD.CONST] = -L * 20  # -1000 baseline
        attn.W_k.data[base, BD.MARK_PC] = L
        # V: copy OP_JMP and FETCH_LO/HI
        attn.W_v.data[base + 1, BD.OP_JMP] = 1.0
        for k in range(16):
            attn.W_v.data[base + 2 + k, BD.FETCH_LO + k] = 1.0
            attn.W_v.data[base + 18 + k, BD.FETCH_HI + k] = 1.0
        # O: write IS_JMP to CMP[0], target to AX_CARRY
        attn.W_o.data[BD.CMP + 0, base + 1] = 1.0
        for k in range(16):
            attn.W_o.data[BD.AX_CARRY_LO + k, base + 2 + k] = 1.0
            attn.W_o.data[BD.AX_CARRY_HI + k, base + 18 + k] = 1.0

        # Head 3: JSR relay (PC marker ← AX marker, first step only)
        base = 3 * self.HD
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.MARK_AX] = -L
        attn.W_q.data[base, BD.HAS_SE] = -L  # only fire when NOT HAS_SE
        attn.W_k.data[base, BD.MARK_AX] = L
        # V: copy OP_JSR flag
        attn.W_v.data[base + 1, BD.OP_JSR] = 1.0
        # O: write IS_JSR to TEMP[0]
        attn.W_o.data[BD.TEMP + 0, base + 1] = 1.0

        # Head 4: BZ/BNZ relay (PC marker ← AX marker)
        # FIX 2026-04-16: Gate on OP_BZ or OP_BNZ to prevent firing for other opcodes.
        base = 4 * self.HD
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.MARK_AX] = -L
        attn.W_q.data[base, BD.CONST] = -L * 1.3  # Baseline penalty
        attn.W_q.data[base, BD.OP_BZ] = L / 5.0   # OP_BZ=5 → contributes L
        attn.W_q.data[base, BD.OP_BNZ] = L / 5.0  # OP_BNZ=5 → contributes L
        attn.W_k.data[base, BD.MARK_AX] = L
        attn.W_k.data[base, BD.CONST] = L  # K-side constant for Q gating
        # V: copy OP_BZ, OP_BNZ flags, AX_CARRY[0], FETCH
        attn.W_v.data[base + 1, BD.OP_BZ] = 1.0
        attn.W_v.data[base + 2, BD.OP_BNZ] = 1.0
        attn.W_v.data[base + 3, BD.AX_CARRY_LO + 0] = 1.0
        attn.W_v.data[base + 4, BD.AX_CARRY_HI + 0] = 1.0
        for k in range(16):
            attn.W_v.data[base + 5 + k, BD.FETCH_LO + k] = 1.0
            attn.W_v.data[base + 21 + k, BD.FETCH_HI + k] = 1.0
        # O: write to CMP[2..5] and TEMP for branch target
        attn.W_o.data[BD.CMP + 2, base + 1] = 0.2  # OP_BZ (normalized)
        attn.W_o.data[BD.CMP + 3, base + 2] = 0.2  # OP_BNZ (normalized)
        attn.W_o.data[BD.CMP + 4, base + 3] = 1.0  # AX_LO_IS_ZERO
        attn.W_o.data[BD.CMP + 5, base + 4] = 1.0  # AX_HI_IS_ZERO
        for k in range(16):
            attn.W_o.data[BD.TEMP + k, base + 5 + k] = 1.0
            attn.W_o.data[BD.TEMP + 16 + k, base + 21 + k] = 1.0

        # Head 5: First-step OP flag relay + FETCH relay (AX ← PC)
        base = 5 * self.HD
        attn.W_q.data[base, BD.MARK_AX] = L
        attn.W_q.data[base, BD.HAS_SE] = -L  # only first step
        attn.W_k.data[base, BD.MARK_PC] = L
        # V: copy OP flags (18 total)
        op_flags = [
            BD.OP_IMM, BD.OP_LEA, BD.OP_JMP, BD.OP_JSR, BD.OP_EXIT, BD.OP_NOP,
            BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
            BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_EQ, BD.OP_LT,
            BD.OP_SHL, BD.OP_SHR,
        ]
        for i, op_dim in enumerate(op_flags):
            attn.W_v.data[base + i, op_dim] = 1.0
            attn.W_o.data[op_dim, base + i] = 1.0
        # V: also copy FETCH_LO/HI (positions 18-49)
        for k in range(16):
            attn.W_v.data[base + 18 + k, BD.FETCH_LO + k] = 1.0
            attn.W_v.data[base + 34 + k, BD.FETCH_HI + k] = 1.0
        # O: write FETCH at AX marker
        for k in range(16):
            attn.W_o.data[BD.FETCH_LO + k, base + 18 + k] = 1.0
            attn.W_o.data[BD.FETCH_HI + k, base + 34 + k] = 1.0
        # HAS_SE gate to block when HAS_SE=1
        HAS_SE_GATE = 49
        attn.W_q.data[base + HAS_SE_GATE, BD.HAS_SE] = -500.0
        attn.W_k.data[base + HAS_SE_GATE, BD.CONST] = 5.0

        # Head 6: Opcode relay (AX → SP/STACK0/BP/PC/MEM) + AX_CARRY_LO → ALU_LO
        # This head has dual functions:
        #   1. _set_opcode_relay_head: relay OP flags from AX to other markers
        #   2. _set_layer6_relay_heads: relay AX_CARRY_LO to ALU_LO at STACK0
        SP_I = 2
        BP_I = 3
        base = 6 * self.HD
        # Q: fires at SP, STACK0, BP, PC, MEM markers, blocked at AX
        # FIX 2026-04-16: Also fire at SP/STACK0 byte positions (not just markers).
        # L6 FFN needs CMP[4] at byte positions for JSR SP-=8 and STACK0=return_addr.
        attn.W_q.data[base, BD.MARK_SP] = L
        attn.W_q.data[base, BD.H1 + SP_I] = L  # SP area including bytes
        attn.W_q.data[base, BD.MARK_STACK0] = L
        attn.W_q.data[base, BD.L1H4 + BP_I] = L  # STACK0 bytes
        attn.W_q.data[base, BD.MARK_BP] = L
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.MARK_MEM] = L
        attn.W_q.data[base, BD.MARK_AX] = -L
        # K: attend to AX marker
        attn.W_k.data[base, BD.MARK_AX] = L

        # V[0]: OP_LEV (scaled)
        attn.W_v.data[base + 0, BD.OP_LEV] = 0.2
        # V[1]: OP_PSH (scaled)
        attn.W_v.data[base + 1, BD.OP_PSH] = 0.2
        # V[2]: OP_ADJ (scaled)
        attn.W_v.data[base + 2, BD.OP_ADJ] = 0.2
        # V[3]: POP group (18 binary ops, scaled)
        pop_op_dims = [
            BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
            BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
            BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_SHL, BD.OP_SHR,
            BD.OP_SI, BD.OP_SC,
        ]
        for op_dim in pop_op_dims:
            attn.W_v.data[base + 3, op_dim] = 0.04
        # V[4]: OP_ENT (scaled)
        attn.W_v.data[base + 4, BD.OP_ENT] = 0.2
        # V[5]: OP_JSR (scaled)
        attn.W_v.data[base + 5, BD.OP_JSR] = 0.2
        # V[6]: MEM_STORE flag (SI/SC/PSH/JSR/ENT)
        attn.W_v.data[base + 6, BD.OP_SI] = 0.2
        attn.W_v.data[base + 6, BD.OP_SC] = 0.2
        attn.W_v.data[base + 6, BD.OP_PSH] = 0.2
        attn.W_v.data[base + 6, BD.OP_JSR] = 0.2
        attn.W_v.data[base + 6, BD.OP_ENT] = 0.2
        # V[7]: MEM_ADDR_SRC (SI/SC only)
        attn.W_v.data[base + 7, BD.OP_SI] = 0.2
        attn.W_v.data[base + 7, BD.OP_SC] = 0.2
        # V[8-23]: AX_CARRY_LO (for STACK0 ALU)
        for k in range(16):
            attn.W_v.data[base + 8 + k, BD.AX_CARRY_LO + k] = 1.0

        # O: opcode relay outputs
        attn.W_o.data[BD.CMP + 0, base + 1] = 1.0  # OP_PSH → CMP[0]
        attn.W_o.data[BD.PSH_AT_SP, base + 1] = 1.0  # OP_PSH → PSH_AT_SP
        attn.W_o.data[BD.CMP + 1, base + 2] = 1.0  # OP_ADJ → CMP[1]
        attn.W_o.data[BD.CMP + 3, base + 3] = 5.0  # POP group → CMP[3] (rescale)
        attn.W_o.data[BD.CMP + 2, base + 4] = 1.0  # OP_ENT → CMP[2]
        attn.W_o.data[BD.OP_ENT, base + 4] = 5.0  # Fix: relay OP_ENT flag itself
        attn.W_o.data[BD.CMP + 4, base + 5] = 1.0  # OP_JSR → CMP[4]
        attn.W_o.data[BD.OP_JSR, base + 5] = 5.0  # Fix: relay OP_JSR flag itself
        attn.W_o.data[BD.MEM_STORE, base + 6] = 1.0  # store flag
        attn.W_o.data[BD.MEM_ADDR_SRC, base + 7] = 1.0  # addr source
        attn.W_o.data[BD.OP_LEV, base + 0] = 5.0  # OP_LEV relay (rescale)
        # O: ALU_LO outputs (for STACK0)
        for k in range(16):
            attn.W_o.data[BD.ALU_LO + k, base + 8 + k] = 1.0

        # Head 7: Dual functionality:
        #   1. PSH: STACK0 ← AX_CARRY_HI → ALU_HI (attends AX marker)
        #   2. JSR: STACK0 ← PC OUTPUT → AX_CARRY (attends PC marker)
        # FIX 2026-04-16: Strong anti-leakage gate for JSR.
        base = 7 * self.HD
        attn.W_q.data[base, BD.MARK_STACK0] = L + L * 20  # +1050 at STACK0
        attn.W_q.data[base, BD.MARK_AX] = -L
        attn.W_q.data[base, BD.CONST] = -L * 20  # -1000 baseline
        # K: attend to BOTH AX marker (PSH) and PC marker (JSR)
        attn.W_k.data[base, BD.MARK_AX] = L
        attn.W_k.data[base, BD.MARK_PC] = 30.0  # Fix: manual uses 30, not L
        attn.W_k.data[base, BD.OP_JSR] = -20.0  # Fix: suppress for JSR
        # V[0-15]: AX_CARRY_HI (for PSH: ALU = AX value)
        for k in range(16):
            attn.W_v.data[base + k, BD.AX_CARRY_HI + k] = 1.0
        # V[1-32]: OUTPUT_LO/HI (for JSR: STACK0 = return address = PC+5)
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
        # O[ALU_HI]: PSH output
        for k in range(16):
            attn.W_o.data[BD.ALU_HI + k, base + k] = 1.0
        # O[AX_CARRY]: JSR output (return address at STACK0)
        for k in range(16):
            attn.W_o.data[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0

    def _compile_l7_attention(self, attn):
        """Compile L7 attention (operand gather + memory relay heads)."""
        HD = self.HD
        L = 15.0
        AX_I = 1
        SP_I = 2
        BP_I = 3
        MEM_I = 4

        # Head 0: AX ← prev STACK0 byte 0 (STACK0_BYTE0 key)
        base = 0 * HD
        attn.W_q.data[base, BD.MARK_AX] = L
        attn.W_q.data[base, BD.OP_LEA] = -L  # suppress STACK0→ALU for LEA
        attn.W_k.data[base, BD.STACK0_BYTE0] = L
        # V: copy CLEAN_EMBED_LO/HI from STACK0 byte 0
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to ALU_LO/ALU_HI at AX marker
        for k in range(16):
            attn.W_o.data[BD.ALU_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.ALU_HI + k, base + 17 + k] = 1.0

        # Head 1: LEA/ADJ/ENT — BP/SP OUTPUT → ALU at AX marker
        base = 1 * HD
        attn.W_q.data[base, BD.OP_LEA] = L
        attn.W_q.data[base, BD.OP_ADJ] = L
        attn.W_q.data[base, BD.OP_ENT] = L
        # Anti-leakage gate (dim 1)
        attn.W_q.data[base + 1, BD.CONST] = -L * 2
        attn.W_q.data[base + 1, BD.OP_LEA] = L * 2
        attn.W_q.data[base + 1, BD.OP_ADJ] = L * 2
        attn.W_q.data[base + 1, BD.OP_ENT] = L * 2
        attn.W_k.data[base + 1, BD.CONST] = 1.0
        attn.W_k.data[base, BD.MARK_BP] = L
        attn.W_k.data[base, BD.MARK_SP] = L
        # V: copy OUTPUT_LO/HI
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
        # O: write to ALU_LO/ALU_HI
        for k in range(16):
            attn.W_o.data[BD.ALU_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.ALU_HI + k, base + 17 + k] = 1.0

        # Heads 2-4: Gather prev AX bytes → AX positions (for LI/LC address)
        for j in range(3):
            head = 2 + j
            base = head * HD
            byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
            addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
            addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
            # Q: fires at AX marker + AX bytes, suppressed at MEM/STACK0
            attn.W_q.data[base, BD.MARK_AX] = L
            attn.W_q.data[base, BD.H1 + AX_I] = L
            attn.W_q.data[base, BD.H3 + MEM_I] = -L
            attn.W_q.data[base, BD.H4 + BP_I] = -L
            # K: fires at prev step's AX byte J
            attn.W_k.data[base, byte_idx_dim] = L
            attn.W_k.data[base, BD.H1 + AX_I] = L
            # Anti-leakage
            attn.W_q.data[base + 33, BD.CONST] = -L / 2
            attn.W_q.data[base + 33, BD.MARK_AX] = L
            attn.W_k.data[base + 33, BD.CONST] = L
            # V: copy CLEAN_EMBED nibbles
            for k in range(16):
                attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            # O: write to ADDR_BJ_LO/HI
            for k in range(16):
                attn.W_o.data[addr_lo_out + k, base + 1 + k] = 1.0
                attn.W_o.data[addr_hi_out + k, base + 17 + k] = 1.0

        # Head 5: Relay OP_LI/OP_LC/OP_LEA from AX marker → AX byte positions
        base = 5 * HD
        attn.W_q.data[base, BD.MARK_AX] = L
        attn.W_q.data[base, BD.H1 + AX_I] = L
        attn.W_k.data[base, BD.MARK_AX] = L
        # V: OP_LI, OP_LC, OP_LEA (scaled)
        attn.W_v.data[base + 1, BD.OP_LI] = 0.2
        attn.W_v.data[base + 2, BD.OP_LC] = 0.2
        attn.W_v.data[base + 3, BD.OP_LEA] = 0.2
        # O: write to relay dims
        attn.W_o.data[BD.OP_LI_RELAY, base + 1] = 1.0
        attn.W_o.data[BD.OP_LC_RELAY, base + 2] = 1.0
        attn.W_o.data[BD.CMP + 7, base + 3] = 1.0  # OP_LEA relay

        # Head 6: Relay PSH/ENT/JSR from STACK0 marker → STACK0 byte positions
        base = 6 * HD
        attn.W_q.data[base, BD.MARK_STACK0] = L
        attn.W_q.data[base, BD.H4 + BP_I] = L  # Fix: H4 not L1H4
        attn.W_q.data[base, BD.H1 + BP_I] = -L  # Fix: block H1+BP_I (SP byte positions)
        attn.W_q.data[base, BD.IS_BYTE] = L
        attn.W_q.data[base, BD.MARK_SP] = L
        attn.W_q.data[base, BD.H1 + SP_I] = L
        attn.W_q.data[base, BD.H1 + AX_I] = -L
        attn.W_q.data[base, BD.H3 + MEM_I] = -L
        attn.W_k.data[base, BD.MARK_STACK0] = L
        attn.W_k.data[base, BD.MARK_SP] = L
        # V: copy CMP flags
        attn.W_v.data[base + 1, BD.CMP + 0] = 1.0  # PSH
        attn.W_v.data[base + 2, BD.CMP + 2] = 1.0  # ENT
        attn.W_v.data[base + 3, BD.CMP + 4] = 1.0  # JSR
        attn.W_v.data[base + 4, BD.PSH_AT_SP] = 1.0
        attn.W_v.data[base + 5, BD.CMP + 3] = 1.0  # POP group
        # O: accumulate at STACK0/SP byte positions
        attn.W_o.data[BD.CMP + 0, base + 1] = 1.0
        attn.W_o.data[BD.CMP + 2, base + 2] = 1.0
        attn.W_o.data[BD.CMP + 4, base + 3] = 1.0
        attn.W_o.data[BD.PSH_AT_SP, base + 4] = 1.0
        attn.W_o.data[BD.CMP + 3, base + 5] = 1.0

        # Head 7: MEM flag broadcast (MEM marker → MEM byte positions)
        base = 7 * HD
        attn.W_q.data[base, BD.MARK_MEM] = L
        attn.W_q.data[base, BD.H3 + MEM_I] = L
        attn.W_q.data[base, BD.H1 + AX_I] = -L
        attn.W_q.data[base, BD.H1 + SP_I] = -L
        attn.W_q.data[base, BD.H1 + BP_I] = -L
        attn.W_q.data[base, BD.H4 + BP_I] = -L
        attn.W_k.data[base, BD.MARK_MEM] = L
        # V: copy MEM_STORE and MEM_ADDR_SRC
        attn.W_v.data[base + 1, BD.MEM_STORE] = 1.0
        attn.W_v.data[base + 2, BD.MEM_ADDR_SRC] = 1.0
        # V: copy OP_JSR and OP_ENT (fix: relay opcode flags to MEM byte positions)
        attn.W_v.data[base + 3, BD.OP_JSR] = 1.0
        attn.W_v.data[base + 4, BD.OP_ENT] = 1.0
        # O: write to same dims
        attn.W_o.data[BD.MEM_STORE, base + 1] = 1.0
        attn.W_o.data[BD.MEM_ADDR_SRC, base + 2] = 1.0
        attn.W_o.data[BD.OP_JSR, base + 3] = 1.0
        attn.W_o.data[BD.OP_ENT, base + 4] = 1.0

    def _compile_l8_attention(self, attn):
        """Compile L8 attention (SP gather → STACK0 positions)."""
        HD = self.HD
        L = 15.0
        AX_I = 1
        SP_I = 2
        BP_I = 3
        MEM_I = 4

        # Set ALiBi slopes
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)

        # Heads 0-2: Gather SP bytes → STACK0 positions
        for j in range(3):
            base = j * HD
            byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
            addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
            addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
            # Q: fires at STACK0 area
            attn.W_q.data[base, BD.MARK_STACK0] = L
            attn.W_q.data[base, BD.H4 + BP_I] = L
            attn.W_q.data[base, BD.H1 + AX_I] = -L
            attn.W_q.data[base, BD.H1 + SP_I] = -L
            attn.W_q.data[base, BD.H3 + MEM_I] = -L
            attn.W_q.data[base, BD.MARK_BP] = -L
            # K: fires at SP byte J
            attn.W_k.data[base, byte_idx_dim] = L
            attn.W_k.data[base, BD.H1 + SP_I] = L
            # Anti-leakage gate
            attn.W_q.data[base + 33, BD.MARK_STACK0] = L
            attn.W_q.data[base + 33, BD.CONST] = -L / 2
            attn.W_k.data[base + 33, BD.CONST] = L
            # V: copy CLEAN_EMBED nibbles
            for k in range(16):
                attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            # O: write to ADDR_BJ_LO/HI
            for k in range(16):
                attn.W_o.data[addr_lo_out + k, base + 1 + k] = 1.0
                attn.W_o.data[addr_hi_out + k, base + 17 + k] = 1.0

    def _compile_l9_attention(self, attn):
        """Compile L9 attention (LEV address relay)."""
        HD = self.HD
        L = 50.0
        BP_I = 3
        GATE = 33

        # Set ALiBi slopes
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
            attn.alibi_slopes[0] = 0.2  # head 0: shallow slope for d=29 relay
            attn.alibi_slopes[1] = 0.5  # head 1: BP→PC relay

        # Head 0: relay old BP value to SP marker (for LEV SP = BP + 16)
        base = 0 * HD
        # Q: fires at SP marker when OP_LEV active
        attn.W_q.data[base, BD.MARK_SP] = L
        attn.W_q.data[base, BD.OP_LEV] = L / 5
        attn.W_q.data[base, BD.CONST] = -2 * L
        # K: attend to BP byte 0
        attn.W_k.data[base, BD.L1H1 + BP_I] = L
        attn.W_k.data[base, BD.BYTE_INDEX_0] = L
        # V: copy CLEAN_EMBED_LO/HI (scaled)
        scale = 3.0
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale
            attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale
        # O: write to ADDR_B0_LO/HI at SP marker
        for k in range(16):
            attn.W_o.data[BD.ADDR_B0_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.ADDR_B0_HI + k, base + 17 + k] = 1.0
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_SP] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

        # Head 1: relay BP value to PC marker (for LEV return_addr lookup)
        base = 1 * HD
        # Q: fires at PC marker when OP_LEV active
        attn.W_q.data[base, BD.MARK_PC] = L
        attn.W_q.data[base, BD.OP_LEV] = L / 5
        attn.W_q.data[base, BD.CONST] = -1.5 * L
        # K: attend to BP byte 0
        attn.W_k.data[base, BD.L1H1 + BP_I] = L
        attn.W_k.data[base, BD.BYTE_INDEX_0] = L
        # V: copy CLEAN_EMBED_LO/HI (scaled)
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale
            attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale
        # O: write to ADDR_B0_LO/HI at PC marker
        for k in range(16):
            attn.W_o.data[BD.ADDR_B0_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.ADDR_B0_HI + k, base + 17 + k] = 1.0
        # Anti-leakage gate
        attn.W_q.data[base + GATE, BD.MARK_PC] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

    def _compile_l10_attention(self, attn):
        """Compile L10 attention (carry relay + AX/SP byte passthrough)."""
        HD = self.HD
        L = self.S
        AX_IDX = 1
        SP_IDX = 2

        # Set ALiBi slopes
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[0] = 5.0  # head 0: steep slope for carry relay
            attn.alibi_slopes[1] = 1.0  # head 1: AX byte passthrough
            attn.alibi_slopes[2] = 1.0  # head 2: SP byte passthrough

        # Head 0: Carry relay (AX marker → AX byte positions)
        base = 0 * HD
        attn.W_q.data[base, BD.IS_BYTE] = L
        attn.W_q.data[base, BD.CONST] = -L / 2
        attn.W_k.data[base, BD.MARK_AX] = L
        attn.W_q.data[base + 33, BD.H1 + AX_IDX] = L
        attn.W_q.data[base + 33, BD.CONST] = -L / 2
        attn.W_k.data[base + 33, BD.CONST] = L
        attn.W_v.data[base + 1, BD.CARRY + 1] = 1.0
        attn.W_v.data[base + 2, BD.CARRY + 2] = 1.0
        attn.W_o.data[BD.CARRY + 1, base + 1] = 1.0
        attn.W_o.data[BD.CARRY + 2, base + 2] = 1.0

        # Head 1: AX byte passthrough (shifted matching)
        base = 1 * HD
        attn.W_q.data[base, BD.IS_BYTE] = L * 3
        attn.W_q.data[base, BD.HAS_SE] = L
        attn.W_q.data[base, BD.CONST] = -L * 3.5
        attn.W_q.data[base + 1, BD.H1 + AX_IDX] = L
        attn.W_q.data[base + 1, BD.CONST] = -L / 2
        attn.W_q.data[base + 2, BD.BYTE_INDEX_3] = -L
        attn.W_q.data[base + 2, BD.CONST] = L / 2
        attn.W_k.data[base, BD.IS_BYTE] = L
        attn.W_k.data[base + 1, BD.H1 + AX_IDX] = L
        attn.W_k.data[base + 2, BD.BYTE_INDEX_0] = -L
        attn.W_k.data[base + 2, BD.CONST] = L / 2
        attn.W_q.data[base + 3, BD.BYTE_INDEX_0] = L
        attn.W_k.data[base + 3, BD.BYTE_INDEX_1] = L
        attn.W_q.data[base + 4, BD.BYTE_INDEX_1] = L
        attn.W_k.data[base + 4, BD.BYTE_INDEX_2] = L
        attn.W_q.data[base + 5, BD.BYTE_INDEX_2] = L
        attn.W_k.data[base + 5, BD.BYTE_INDEX_3] = L
        attn.W_q.data[base + 33, BD.CONST] = -20000.0
        attn.W_q.data[base + 33, BD.H1 + AX_IDX] = 10000.0
        attn.W_q.data[base + 33, BD.HAS_SE] = 10000.0
        attn.W_k.data[base + 33, BD.CONST] = 5.0
        for k in range(16):
            attn.W_v.data[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        for k in range(16):
            attn.W_o.data[BD.OUTPUT_LO + k, base + k] = 2.0
            attn.W_o.data[BD.OUTPUT_HI + k, base + 16 + k] = 2.0

        # Head 2: SP byte passthrough (shifted matching, NOT PSH)
        base = 2 * HD
        attn.W_q.data[base, BD.IS_BYTE] = L
        attn.W_q.data[base, BD.HAS_SE] = L * 2
        attn.W_q.data[base, BD.PSH_AT_SP] = -L * 2
        attn.W_q.data[base, BD.CONST] = -L * 1.5
        attn.W_q.data[base + 1, BD.H1 + SP_IDX] = L
        attn.W_q.data[base + 1, BD.CONST] = -L / 2
        attn.W_q.data[base + 2, BD.BYTE_INDEX_3] = -L
        attn.W_q.data[base + 2, BD.CONST] = L / 2
        attn.W_k.data[base, BD.IS_BYTE] = L
        attn.W_k.data[base + 1, BD.H1 + SP_IDX] = L
        attn.W_k.data[base + 2, BD.BYTE_INDEX_0] = -L
        attn.W_k.data[base + 2, BD.CONST] = L / 2
        attn.W_q.data[base + 3, BD.BYTE_INDEX_0] = L
        attn.W_k.data[base + 3, BD.BYTE_INDEX_1] = L
        attn.W_q.data[base + 4, BD.BYTE_INDEX_1] = L
        attn.W_k.data[base + 4, BD.BYTE_INDEX_2] = L
        attn.W_q.data[base + 5, BD.BYTE_INDEX_2] = L
        attn.W_k.data[base + 5, BD.BYTE_INDEX_3] = L
        attn.W_q.data[base + 33, BD.CONST] = -30000.0
        attn.W_q.data[base + 33, BD.IS_BYTE] = 10000.0
        attn.W_q.data[base + 33, BD.H1 + SP_IDX] = 10000.0
        attn.W_q.data[base + 33, BD.HAS_SE] = 10000.0
        attn.W_q.data[base + 33, BD.PSH_AT_SP] = -10000.0
        attn.W_q.data[base + 33, BD.CMP + 3] = -10000.0
        attn.W_k.data[base + 33, BD.CONST] = 5.0
        for k in range(16):
            attn.W_v.data[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        for k in range(16):
            attn.W_o.data[BD.OUTPUT_LO + k, base + k] = 2.0
            attn.W_o.data[BD.OUTPUT_HI + k, base + 16 + k] = 2.0

        # Head 3: PSH STACK0 bytes 1-3 passthrough from AX
        BP_IDX = 3
        base = 3 * HD
        attn.W_q.data[base + 0, BD.IS_BYTE] = L
        attn.W_q.data[base + 1, BD.H4 + BP_IDX] = L
        attn.W_q.data[base + 1, BD.H1 + BP_IDX] = -L
        attn.W_q.data[base + 1, BD.CONST] = -L / 2
        attn.W_q.data[base + 2, BD.BYTE_INDEX_3] = -L
        attn.W_q.data[base + 2, BD.CONST] = L / 2
        attn.W_q.data[base + 3, BD.PSH_AT_SP] = L
        attn.W_q.data[base + 3, BD.CONST] = -L / 2
        attn.W_k.data[base + 0, BD.IS_BYTE] = L
        attn.W_k.data[base + 1, BD.H1 + AX_IDX] = L
        attn.W_k.data[base + 2, BD.BYTE_INDEX_0] = -L
        attn.W_k.data[base + 2, BD.CONST] = L / 2
        # Shifted byte matching
        attn.W_q.data[base + 4, BD.BYTE_INDEX_0] = L
        attn.W_k.data[base + 4, BD.BYTE_INDEX_1] = L
        attn.W_q.data[base + 5, BD.BYTE_INDEX_1] = L
        attn.W_k.data[base + 5, BD.BYTE_INDEX_2] = L
        attn.W_q.data[base + 6, BD.BYTE_INDEX_2] = L
        attn.W_k.data[base + 6, BD.BYTE_INDEX_3] = L
        # Gate dim 33: 4-way AND enforcement
        attn.W_q.data[base + 33, BD.CONST] = -30000.0
        attn.W_q.data[base + 33, BD.IS_BYTE] = 10000.0
        attn.W_q.data[base + 33, BD.H4 + BP_IDX] = 10000.0
        attn.W_q.data[base + 33, BD.H1 + BP_IDX] = -10000.0
        attn.W_q.data[base + 33, BD.PSH_AT_SP] = 10000.0
        attn.W_k.data[base + 33, BD.CONST] = 5.0
        # V: copy CLEAN_EMBED
        for k in range(16):
            attn.W_v.data[base + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v.data[base + 16 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # O: write to OUTPUT (strength 3.0)
        for k in range(16):
            attn.W_o.data[BD.OUTPUT_LO + k, base + k] = 3.0
            attn.W_o.data[BD.OUTPUT_HI + k, base + 16 + k] = 3.0

    def _compile_l13_attention(self, attn):
        """Compile L13 attention (MEM addr gather).

        Heads 0-2: Gather MEM addr bytes → MEM val byte positions.
        For L15 K-side address keys: copies addr byte nibbles from MEM addr
        positions (d=0..3 from MEM marker) to MEM val byte positions (d=4..8).
        """
        HD = self.HD
        L = 15.0
        MEM_I = 4

        for j in range(3):
            base = j * HD
            addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
            addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]

            # Q: fires at MEM val byte positions (d=5..8 from MEM)
            # Use MEM_VAL_B0-3 flags computed in L2 FFN
            attn.W_q.data[base, BD.MEM_VAL_B0] = L
            attn.W_q.data[base, BD.MEM_VAL_B1] = L
            attn.W_q.data[base, BD.MEM_VAL_B2] = L
            attn.W_q.data[base, BD.MEM_VAL_B3] = L

            # K: fires at MEM addr byte J position.
            # Addr byte 0 is at d=1 (after MEM marker), byte 1 at d=2, byte 2 at d=3.
            if j == 0:
                # Addr byte 0 at d=1: L1H1[MEM]=1 (d≤1.5), subtract L1H0[MEM] (d=0 only)
                attn.W_k.data[base, BD.L1H1 + MEM_I] = L
                attn.W_k.data[base, BD.L1H0 + MEM_I] = -L  # exclude MEM marker (d=0)
            elif j == 1:
                # Addr byte 1 at d=2: L1H2[MEM]=1 (d≤2.5), subtract L1H1[MEM] (d≤1.5)
                attn.W_k.data[base, BD.L1H2 + MEM_I] = L
                attn.W_k.data[base, BD.L1H1 + MEM_I] = -L
            elif j == 2:
                # Addr byte 2 at d=3: H0[MEM]=1 (d≤3.5), subtract L1H2[MEM] (d≤2.5)
                attn.W_k.data[base, BD.H0 + MEM_I] = L
                attn.W_k.data[base, BD.L1H2 + MEM_I] = -L

            # Anti-leakage gate
            attn.W_q.data[base + 33, BD.MEM_VAL_B0] = L
            attn.W_q.data[base + 33, BD.CONST] = -L / 2
            attn.W_k.data[base + 33, BD.CONST] = L

            # V: copy CLEAN_EMBED nibbles (addr byte value)
            for k in range(16):
                attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            # O: write to ADDR_BJ_LO/HI (gathered to val byte positions)
            for k in range(16):
                attn.W_o.data[addr_lo_out + k, base + 1 + k] = 1.0
                attn.W_o.data[addr_hi_out + k, base + 17 + k] = 1.0

    def _compile_l14_attention(self, attn):
        """Compile L14 attention (MEM byte generation).

        Addr heads (0-3): Copy address from SP (PSH, MEM_ADDR_SRC=0) or STACK0
        (SI/SC, MEM_ADDR_SRC=1) to MEM addr byte positions.

        Val heads (4-7): Copy AX byte values to MEM val byte positions.

        Each head fires at exactly ONE MEM byte position using position-specific
        threshold-difference encoding (dim 0) + position gate (dim 33) +
        MEM_STORE gate (dim 34).
        """
        HD = self.HD
        L = 15.0
        MEM_I = 4
        AX_I = 1
        SP_I = 2
        BP_I = 3

        # Position flags: threshold-difference pairs selecting distance from MEM.
        # AUTOREGRESSIVE SHIFT: logits at position d predict the token at d+1.
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
            attn.W_q.data[base, pos_up] = L
            if pos_down is not None:
                attn.W_q.data[base, pos_down] = -L
            # Fix: block at STACK0 marker, STACK0 byte positions, and SP byte positions
            attn.W_q.data[base, BD.MARK_STACK0] = -L
            attn.W_q.data[base, BD.H4 + BP_I] = -L
            attn.W_q.data[base, BD.H1 + SP_I] = -L

            if h == 0:
                # Byte 0: K matches SP marker OR STACK0 byte 0.
                attn.W_k.data[base, BD.MARK_SP] = L
                attn.W_k.data[base, BD.STACK0_BYTE0] = L
            else:
                # Bytes 1-3: K matches BYTE_INDEX_J (standard byte positions).
                byte_idx_dim = [None, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][h]
                attn.W_k.data[base, byte_idx_dim] = L

            # Dim 1: SP source bonus (active when MEM_ADDR_SRC=0, i.e., PSH).
            attn.W_q.data[base + 1, BD.CONST] = L
            attn.W_q.data[base + 1, BD.MEM_ADDR_SRC] = -2 * L
            attn.W_k.data[base + 1, BD.H1 + SP_I] = L
            attn.W_k.data[base + 1, BD.H1 + AX_I] = -L  # exclude AX

            # Dim 2: STACK0 source bonus (active when MEM_ADDR_SRC=1, i.e., SI/SC).
            # Fix: Match manual's actual selection patterns
            attn.W_q.data[base + 2, BD.MEM_ADDR_SRC] = L
            if h == 0:
                # Head 0: Uses STACK0_BYTE0 flag (NOT threshold diff)
                attn.W_k.data[base + 2, BD.STACK0_BYTE0] = L
            elif h == 1:
                # Head 1: L1H4[BP] - H1[BP] (NOT H2-L1H4)
                attn.W_k.data[base + 2, BD.L1H4 + BP_I] = L
                attn.W_k.data[base + 2, BD.H1 + BP_I] = -L
            elif h == 2:
                # Head 2: H2[BP] - L1H4[BP]
                attn.W_k.data[base + 2, BD.H2 + BP_I] = L
                attn.W_k.data[base + 2, BD.L1H4 + BP_I] = -L
            elif h == 3:
                # Head 3: H3[BP] - H2[BP]
                attn.W_k.data[base + 2, BD.H3 + BP_I] = L
                attn.W_k.data[base + 2, BD.H2 + BP_I] = -L
            # Suppress non-STACK0 areas
            attn.W_k.data[base + 2, BD.H1 + AX_I] = -L
            attn.W_k.data[base + 2, BD.H1 + SP_I] = -L
            attn.W_k.data[base + 2, BD.MARK_STACK0] = -L  # suppress marker token

            # Dim 33: Position gate (suppress non-target MEM byte positions).
            attn.W_q.data[base + 33, BD.CONST] = -500.0
            attn.W_q.data[base + 33, pos_up] = 500.0
            if pos_down is not None:
                attn.W_q.data[base + 33, pos_down] = -500.0
            # Fix: also block at STACK0/SP areas for position gate
            attn.W_q.data[base + 33, BD.MARK_STACK0] = -500.0
            attn.W_q.data[base + 33, BD.L1H4 + BP_I] = -500.0
            attn.W_q.data[base + 33, BD.H1 + SP_I] = -500.0
            attn.W_k.data[base + 33, BD.CONST] = 5.0

            # Dim 34: MEM_STORE gate (suppress non-store positions).
            # Fix: use -500/500 strength instead of -250/250
            attn.W_q.data[base + 34, BD.CONST] = -500.0
            attn.W_q.data[base + 34, BD.MEM_STORE] = 500.0
            attn.W_k.data[base + 34, BD.CONST] = 5.0

            # V: read CLEAN_EMBED + OUTPUT (sum)
            for k in range(16):
                attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
                attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
                attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
            # V[0]: cancel L3 MEM default (reads CONST=1.0 from source)
            attn.W_v.data[base + 0, BD.CONST] = 1.0

            # O: write to OUTPUT_LO/HI
            for k in range(16):
                attn.W_o.data[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
                attn.W_o.data[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
            # O: cancel L3 default (OUTPUT_LO[0] and OUTPUT_HI[0] = -1.0)
            attn.W_o.data[BD.OUTPUT_LO + 0, base + 0] = -1.0
            attn.W_o.data[BD.OUTPUT_HI + 0, base + 0] = -1.0

        # === Heads 4-7: MEM val byte generation ===
        for h in range(4):
            head = 4 + h
            base = head * HD
            pos_up, pos_down = val_pos[h]
            byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][h]

            # Dim 0: Q position selection + K targets byte positions
            attn.W_q.data[base, pos_up] = L
            attn.W_q.data[base, pos_down] = -L
            # Fix: block at STACK0 marker, STACK0 byte positions, and SP byte positions
            attn.W_q.data[base, BD.MARK_STACK0] = -L
            attn.W_q.data[base, BD.H4 + BP_I] = -L
            attn.W_q.data[base, BD.H1 + SP_I] = -L
            attn.W_k.data[base, byte_idx_dim] = L

            # Dim 1: AX source bonus (default for PSH, SI, SC)
            attn.W_q.data[base + 1, BD.CONST] = L
            attn.W_q.data[base + 1, BD.OP_JSR] = -2 * L  # Disable for JSR
            attn.W_q.data[base + 1, BD.OP_ENT] = -2 * L  # Disable for ENT
            attn.W_k.data[base + 1, BD.H1 + AX_I] = L  # AX area bonus (no BP blocker)

            # Dim 2: STACK0 source bonus (JSR and ENT only)
            # Fix: Match manual's actual selection patterns (same as addr heads)
            attn.W_q.data[base + 2, BD.OP_JSR] = L
            attn.W_q.data[base + 2, BD.OP_ENT] = L
            # K: STACK0 byte positions
            if h == 0:
                # Byte 0: Uses STACK0_BYTE0 flag
                attn.W_k.data[base + 2, BD.STACK0_BYTE0] = L
            elif h == 1:
                # Byte 1: L1H4[BP] - H1[BP]
                attn.W_k.data[base + 2, BD.L1H4 + BP_I] = L
                attn.W_k.data[base + 2, BD.H1 + BP_I] = -L
            elif h == 2:
                # Byte 2: H2[BP] - L1H4[BP]
                attn.W_k.data[base + 2, BD.H2 + BP_I] = L
                attn.W_k.data[base + 2, BD.L1H4 + BP_I] = -L
            elif h == 3:
                # Byte 3: H3[BP] - H2[BP]
                attn.W_k.data[base + 2, BD.H3 + BP_I] = L
                attn.W_k.data[base + 2, BD.H2 + BP_I] = -L
            # Suppress non-STACK0 areas when using STACK0 source
            attn.W_k.data[base + 2, BD.H1 + AX_I] = -L
            attn.W_k.data[base + 2, BD.H1 + SP_I] = -L
            attn.W_k.data[base + 2, BD.MARK_STACK0] = -L

            # Dim 33: Position gate
            attn.W_q.data[base + 33, BD.CONST] = -500.0
            attn.W_q.data[base + 33, pos_up] = 500.0
            attn.W_q.data[base + 33, pos_down] = -500.0
            # Fix: also block at STACK0 marker and STACK0 bytes for position gate
            attn.W_q.data[base + 33, BD.MARK_STACK0] = -500.0
            attn.W_q.data[base + 33, BD.L1H4 + BP_I] = -500.0
            attn.W_k.data[base + 33, BD.CONST] = 5.0

            # Dim 34: MEM_STORE gate
            # Fix: use -500/500 strength instead of -250/250
            attn.W_q.data[base + 34, BD.CONST] = -500.0
            attn.W_q.data[base + 34, BD.MEM_STORE] = 500.0
            attn.W_k.data[base + 34, BD.CONST] = 5.0

            # V: copy CLEAN_EMBED nibbles (from AX or STACK0)
            for k in range(16):
                attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            # V[0]: cancel L3 MEM default
            attn.W_v.data[base + 0, BD.CONST] = 1.0

            # O: write to OUTPUT_LO/HI
            for k in range(16):
                attn.W_o.data[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
                attn.W_o.data[BD.OUTPUT_HI + k, base + 17 + k] = 1.0
            # O: cancel L3 default
            attn.W_o.data[BD.OUTPUT_LO + 0, base + 0] = -1.0
            attn.W_o.data[BD.OUTPUT_HI + 0, base + 0] = -1.0

    def _compile_ffn_layers(self, model):
        """Compile FFN weights for all layers."""
        # L0 FFN: Phase A transitions (step structure detection)
        self._compile_l0_ffn(model.blocks[0].ffn)

        # L1 FFN: STACK0_BYTE0 + BYTE_INDEX flags
        self._compile_l1_ffn(model.blocks[1].ffn)

        # L2 FFN: MEM byte flags + extended BYTE_INDEX for STACK0
        self._compile_l2_ffn(model.blocks[2].ffn)

        # L3 FFN: PC/SP/BP first-step defaults + PC increment
        self._compile_l3_ffn(model.blocks[3].ffn)

        # L4 FFN: PC+1 nibbles at AX marker for IMM fetch
        self._compile_l4_ffn(model.blocks[4].ffn)

        # L5 FFN: Opcode decode (OPCODE_BYTE_LO/HI -> OP_* flags)
        self._compile_l5_ffn(model.blocks[5].ffn)

        # L6 FFN: Output routing (IMM/EXIT/NOP/JMP/PSH/HALT)
        self._compile_l6_ffn(model.blocks[6].ffn)

        # L8 FFN: ALU lo nibble (ADD/SUB/LEA/ADJ/ENT) + carry/borrow
        self._compile_l8_ffn(model.blocks[8].ffn)

        # L9 FFN: ALU hi nibble + carry propagation
        self._compile_l9_ffn(model.blocks[9].ffn)

        # L10 FFN: ALU byte 1 + carry propagation
        self._compile_l10_ffn(model.blocks[10].ffn)

        # L11 FFN: ALU byte 2 + MUL/DIV (16-bit)
        self._compile_l11_ffn(model.blocks[11].ffn)

        # L12 FFN: MUL/DIV byte 1
        self._compile_l12_ffn(model.blocks[12].ffn)

        # L13 FFN: MUL/DIV byte 2 + comparison result
        self._compile_l13_ffn(model.blocks[13].ffn)

        # L14 FFN: TEMP clear + ADDR_KEY cleanup for LEV
        self._compile_l14_ffn(model.blocks[14].ffn)

        # L15 FFN: Nibble copy (EMBED → OUTPUT for bytes)
        self._compile_l15_ffn(model.blocks[15].ffn)

        # L16 FFN: LEV routing (SP = BP + 16)
        self._compile_l16_ffn(model.blocks[16].ffn)

    def _compile_l0_ffn(self, ffn):
        """Compile L0 FFN (step structure detection)."""
        S = self.S

        # Marker indices: PC=0, AX=1, SP=2, BP=3, MEM=4, SE=5
        PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5

        # Transitions from vm_step.py _set_phase_a_ffn
        transitions = [
            # (up_dim, gate_dim, out_dim)
            (BD.H0 + SE_I, None, BD.NEXT_PC),       # SE → PC
            (BD.H1 + PC_I, BD.H0 + PC_I, BD.NEXT_AX),   # PC → AX
            (BD.H1 + AX_I, BD.H0 + AX_I, BD.NEXT_SP),   # AX → SP
            (BD.H1 + SP_I, BD.H0 + SP_I, BD.NEXT_BP),   # SP → BP
            (BD.H1 + BP_I, BD.H0 + BP_I, BD.NEXT_STACK0), # BP → STACK0
            (BD.H4 + BP_I, BD.H3 + BP_I, BD.NEXT_MEM),  # STACK0 → MEM
            (BD.H3 + MEM_I, BD.H2 + MEM_I, BD.NEXT_SE), # MEM → SE
        ]

        for i, (up_dim, gate_dim, out_dim) in enumerate(transitions):
            P.threshold_match(
                ffn, unit=i,
                up_dim=up_dim, out_dim=out_dim,
                threshold=0.3,
                gate_dim=gate_dim,
                gate_negative=(gate_dim is not None),
                S=S,
            )

    def _compile_l1_ffn(self, ffn):
        """Compile L1 FFN (STACK0_BYTE0 + BYTE_INDEX).

        STACK0 byte 0 is at d=6 from BP marker (nearest IS_MARK).
        Detected by: L1H4[BP] (d<=6.5) AND NOT H1[BP] (d>4.5) AND IS_BYTE.

        BYTE_INDEX_0-3: Marker-agnostic byte position within a register.
        Derived from threshold heads (summed across all marker types):
          BYTE_INDEX_0: IS_BYTE AND any(L1H1) AND NOT any(L1H0) -> d in (0.5,1.5]
          BYTE_INDEX_1: IS_BYTE AND any(L1H2) AND NOT any(L1H1) -> d in (1.5,2.5]
          BYTE_INDEX_2: IS_BYTE AND any(H0) AND NOT any(L1H2)   -> d in (2.5,3.5]
          BYTE_INDEX_3: IS_BYTE AND any(H1) AND NOT any(H0)     -> d in (3.5,4.5]
        """
        S = self.S
        BP_I = 3
        NM = BD.NUM_MARKERS  # 7 marker types
        unit = 0

        # STACK0_BYTE0: L1H4[BP] AND NOT H1[BP] AND IS_BYTE
        ffn.W_up.data[unit, BD.L1H4 + BP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.H1 + BP_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.STACK0_BYTE0, unit] = 2.0 / S
        unit += 1

        # BYTE_INDEX_0: IS_BYTE AND any(L1H1[i]) AND NOT any(L1H0[i])
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        for i in range(NM):
            ffn.W_up.data[unit, BD.L1H1 + i] = S
        ffn.b_up.data[unit] = -S * 1.5
        for i in range(NM):
            ffn.W_gate.data[unit, BD.L1H0 + i] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_0, unit] = 2.0 / S
        unit += 1

        # BYTE_INDEX_1: IS_BYTE AND any(L1H2[i]) AND NOT any(L1H1[i])
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        for i in range(NM):
            ffn.W_up.data[unit, BD.L1H2 + i] = S
        ffn.b_up.data[unit] = -S * 1.5
        for i in range(NM):
            ffn.W_gate.data[unit, BD.L1H1 + i] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_1, unit] = 2.0 / S
        unit += 1

        # BYTE_INDEX_2: IS_BYTE AND any(H0[i]) AND NOT any(L1H2[i])
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        for i in range(NM):
            ffn.W_up.data[unit, BD.H0 + i] = S
        ffn.b_up.data[unit] = -S * 1.5
        for i in range(NM):
            ffn.W_gate.data[unit, BD.L1H2 + i] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_2, unit] = 2.0 / S
        unit += 1

        # BYTE_INDEX_3: IS_BYTE AND any(H1[i]) AND NOT any(H0[i])
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        for i in range(NM):
            ffn.W_up.data[unit, BD.H1 + i] = S
        ffn.b_up.data[unit] = -S * 1.5
        for i in range(NM):
            ffn.W_gate.data[unit, BD.H0 + i] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_3, unit] = 2.0 / S
        unit += 1

    def _compile_l2_ffn(self, ffn):
        """Compile L2 FFN (MEM byte flags + extended BYTE_INDEX for STACK0).

        MEM val byte flags (4 units): Identify positions d=5..8 from MEM marker.
        Uses threshold-difference pattern between L2H0 (5.5) and L1 thresholds.

        Extended BYTE_INDEX for STACK0 bytes 0-3 (4 units): At positions d=6..9
        from BP (where STACK0 bytes live), produce BYTE_INDEX_0/1/2/3 flags.
        """
        S = self.S
        MEM_I = 4  # MEM marker index in MARKS
        BP_I = 3
        unit = 0

        # MEM_VAL_B0: d=4 from MEM -> H1[MEM]=1 (d<=4.5), H0[MEM]=0 (d>3.5)
        # Fix: manual uses H1+4 for up, H0+4 for gate
        ffn.W_up.data[unit, BD.H1 + MEM_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.H0 + MEM_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.MEM_VAL_B0, unit] = 2.0 / S
        unit += 1

        # MEM_VAL_B1: d=5 from MEM -> L2H0[MEM]=1 (d<=5.5), H1[MEM]=0 (d>4.5)
        ffn.W_up.data[unit, BD.L2H0 + MEM_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.H1 + MEM_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.MEM_VAL_B1, unit] = 2.0 / S
        unit += 1

        # MEM_VAL_B2: d=6 from MEM -> L1H4[MEM]=1 (d<=6.5), L2H0[MEM]=0 (d>5.5)
        ffn.W_up.data[unit, BD.L1H4 + MEM_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.L2H0 + MEM_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.MEM_VAL_B2, unit] = 2.0 / S
        unit += 1

        # MEM_VAL_B3: d=7 from MEM -> H2[MEM]=1 (d<=7.5), L1H4[MEM]=0 (d>6.5)
        ffn.W_up.data[unit, BD.H2 + MEM_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.L1H4 + MEM_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.MEM_VAL_B3, unit] = 2.0 / S
        unit += 1

        # Extended BYTE_INDEX for STACK0 byte 0-3 (at d=6,7,8,9 from BP)
        # BYTE_INDEX_0 at STACK0: d=6 from BP -> L1H4[BP]=1 (d<=6.5), H1[BP]=0 (d>4.5)
        ffn.W_up.data[unit, BD.L1H4 + BP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.H1 + BP_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_0, unit] = 2.0 / S
        unit += 1

        # BYTE_INDEX_1 at STACK0: d=7 from BP -> H2[BP]=1 (d<=7.5), L1H4[BP]=0 (d>6.5)
        ffn.W_up.data[unit, BD.H2 + BP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.L1H4 + BP_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_1, unit] = 2.0 / S
        unit += 1

        # BYTE_INDEX_2 at STACK0: d=8 from BP -> H3[BP]=1 (d<=8.5), H2[BP]=0 (d>7.5)
        ffn.W_up.data[unit, BD.H3 + BP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.H2 + BP_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_2, unit] = 2.0 / S
        unit += 1

        # BYTE_INDEX_3 at STACK0: d=9 from BP -> H4[BP]=1 (d<=9.5), H3[BP]=0 (d>8.5)
        ffn.W_up.data[unit, BD.H4 + BP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.H3 + BP_I] = -1.0
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.BYTE_INDEX_3, unit] = 2.0 / S
        unit += 1

    def _compile_l3_ffn(self, ffn):
        """Compile L3 FFN (PC/SP/BP first-step defaults + register byte defaults).

        First step: PC = PC_OFFSET + INSTR_WIDTH, SP = STACK_INIT, BP = STACK_INIT
        Subsequent steps: PC increments, SP/BP from carry-forward
        """
        from ..constants import STACK_INIT, PC_OFFSET, INSTR_WIDTH
        S = self.S
        PC_I, AX_I, SP_I, BP_I, MEM_I = 0, 1, 2, 3, 4
        unit = 0

        # PC FIRST-STEP DEFAULT: when MARK_PC AND NOT HAS_SE, set PC=PC_OFFSET+INSTR_WIDTH
        first_pc = PC_OFFSET + INSTR_WIDTH
        pc_lo = first_pc & 0xF
        pc_hi = (first_pc >> 4) & 0xF

        # LO nibble default
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + pc_lo, unit] = 2.0 / S
        ffn.W_down.data[BD.EMBED_LO + pc_lo, unit] = 2.0 / S
        unit += 1
        # Undo when HAS_SE (subsequent steps)
        ffn.W_up.data[unit, BD.HAS_SE] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.W_gate.data[unit, BD.MARK_PC] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + pc_lo, unit] = -2.0 / S
        ffn.W_down.data[BD.EMBED_LO + pc_lo, unit] = -2.0 / S
        unit += 1

        # HI nibble default
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + pc_hi, unit] = 2.0 / S
        ffn.W_down.data[BD.EMBED_HI + pc_hi, unit] = 2.0 / S
        unit += 1
        # Undo when HAS_SE
        ffn.W_up.data[unit, BD.HAS_SE] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.W_gate.data[unit, BD.MARK_PC] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + pc_hi, unit] = -2.0 / S
        ffn.W_down.data[BD.EMBED_HI + pc_hi, unit] = -2.0 / S
        unit += 1

        # SP DEFAULT: STACK_INIT = 0x10000 -> bytes 0x00, 0x00, 0x01, 0x00
        # SP bytes 0, 1, 3 = 0
        for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_2]:
            # LO nibble = 0
            ffn.W_up.data[unit, BD.H1 + SP_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            unit += 1
            # HI nibble = 0
            ffn.W_up.data[unit, BD.H1 + SP_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        # SP byte 2 = 0x01 (lo=1, hi=0) - FIRST STEP ONLY
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S  # Only first step
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S  # Only first step
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # BP DEFAULT: same as SP (STACK_INIT = 0x10000)
        # BP bytes 0, 1, 3 = 0
        for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_2]:
            # LO nibble = 0
            ffn.W_up.data[unit, BD.H1 + BP_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            unit += 1
            # HI nibble = 0
            ffn.W_up.data[unit, BD.H1 + BP_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        # BP byte 2 = 0x01 (lo=1, hi=0) - FIRST STEP ONLY
        ffn.W_up.data[unit, BD.H1 + BP_I] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S  # Only first step
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.H1 + BP_I] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S  # Only first step
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # PC DEFAULT: bytes 1-3 = 0 (for PC < 256)
        for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
            ffn.W_up.data[unit, BD.H1 + PC_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            unit += 1
            ffn.W_up.data[unit, BD.H1 + PC_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        # AX DEFAULT: bytes 1-3 = 0 (for single-byte immediates)
        for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
            ffn.W_up.data[unit, BD.H1 + AX_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            unit += 1
            ffn.W_up.data[unit, BD.H1 + AX_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        # MEM DEFAULT: all bytes = 0
        # MEM marker position
        ffn.W_up.data[unit, BD.MARK_MEM] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_MEM] = S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # MEM addr bytes 1-3 default to 0
        for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
            ffn.W_up.data[unit, BD.H1 + MEM_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            unit += 1
            ffn.W_up.data[unit, BD.H1 + MEM_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        # STACK0 DEFAULT: bytes 1-3 = 0
        for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
            ffn.W_up.data[unit, BD.H4 + BP_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.W_up.data[unit, BD.H1 + BP_I] = -S  # Exclude BP area
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            unit += 1
            ffn.W_up.data[unit, BD.H4 + BP_I] = S
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.W_up.data[unit, BD.H1 + BP_I] = -S  # Exclude BP area
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        # STACK0 first-step default: byte 0 = 0 (empty stack) when NOT HAS_SE
        ffn.W_up.data[unit, BD.MARK_STACK0] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_STACK0] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # PC INCREMENT: when MARK_PC AND HAS_SE AND NOT OP_LEV, add INSTR_WIDTH
        # For each lo nibble k (0-15): new_lo = (k+INSTR_WIDTH)%16
        for k in range(16):
            new_k = (k + INSTR_WIDTH) % 16
            ffn.W_up.data[unit, BD.HAS_SE] = S
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_LEV] = -S / 5  # OP_LEV ≈ 5.0
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
            unit += 1

        # Hi nibble copy (only at MARK_PC AND HAS_SE AND NOT OP_LEV)
        for k in range(16):
            ffn.W_up.data[unit, BD.HAS_SE] = S
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_LEV] = -S / 5  # Suppress when LEV
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # PC carry correction: when lo nibble >= (16-INSTR_WIDTH), increment hi
        carry_threshold = 16 - INSTR_WIDTH  # For INSTR_WIDTH=8, this is 8
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = 4 * S
            ffn.W_up.data[unit, BD.HAS_SE] = S
            ffn.W_up.data[unit, BD.OP_LEV] = -S  # OP_LEV ≈ 5, stronger suppression
            for lo_bit in range(carry_threshold, 16):
                ffn.W_up.data[unit, BD.EMBED_LO + lo_bit] = S
            # Bias: activate when MARK_PC(4*S) + HAS_SE(S) + carry_bit(S) >= 5.5*S
            ffn.b_up.data[unit] = -S * 5.5
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = -2.0 / S  # cancel old
            ffn.W_down.data[BD.OUTPUT_HI + (k + 1) % 16, unit] = 2.0 / S  # add shifted
            unit += 1

    def _compile_l4_ffn(self, ffn):
        """Compile L4 FFN (PC+1 nibbles at AX marker for IMM fetch).

        PC value is in EMBED_LO/HI at AX marker (from L4 attention relay).
        L5 fetch uses TEMP[0..31] for immediate address (PC+1).
        """
        S = self.S
        unit = 0

        # PC_PLUS1_LO: rotate EMBED_LO by +1
        for k in range(16):
            src = (k - 1) % 16
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.EMBED_LO + src] = 1.0
            ffn.W_down.data[BD.TEMP + k, unit] = 2.0 / S
            unit += 1

        # PC_PLUS1_HI: default copy (no carry)
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
            ffn.W_down.data[BD.TEMP + 16 + k, unit] = 2.0 / S
            unit += 1

        # PC_PLUS1_HI: carry correction when EMBED_LO[15] == 1
        # Cancel default copy and write rotated (+1) hi nibble
        for k in range(16):
            # Cancel: MARK_AX AND LO[15] -> subtract EMBED_HI[k]
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.EMBED_LO + 15] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = -1.0
            ffn.W_down.data[BD.TEMP + 16 + k, unit] = 2.0 / S
            unit += 1
            # Write rotated: MARK_AX AND LO[15] -> add EMBED_HI[(k-1)%16]
            src = (k - 1) % 16
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.EMBED_LO + 15] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.EMBED_HI + src] = 1.0
            ffn.W_down.data[BD.TEMP + 16 + k, unit] = 2.0 / S
            unit += 1

        # TEMP clearing at PC marker to prevent leakage to Layer 6
        # Skip TEMP[0] - used for IS_JSR flag
        for k in range(32):
            if k == 0:
                # Skip TEMP[0], leave unit with zero weights
                unit += 1
                continue
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.TEMP + k] = -1.0
            ffn.W_down.data[BD.TEMP + k, unit] = 2.0 / S
            unit += 1

    def _compile_l5_ffn(self, ffn):
        """Compile L5 FFN (opcode decode: OPCODE_BYTE_LO/HI -> OP_* flags).

        Each opcode has a unique (lo, hi) nibble pair. SwiGLU AND gate pattern:
          up = S*(OPCODE_BYTE_LO[lo] + OPCODE_BYTE_HI[hi] - 1.5)
          gate = MARK_AX (only at AX marker where fetch results land)
          down -> OP_xxx flag

        Also includes first-step opcode decode at PC marker for several opcodes.
        """
        from ..embedding import Opcode
        S = self.S
        unit = 0

        # Opcode table: (opcode_enum, lo_nibble, hi_nibble)
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
            (Opcode.NOP, 7, 2),   # NOP = 39 = 0x27
            (Opcode.PUTCHAR, 1, 4),
            (Opcode.GETCHAR, 0, 4),
        ]

        # Main opcode decode at AX marker
        for op_val, lo, hi in opcodes:
            op_dim = BD.opcode_dim(op_val)
            ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo] = S
            ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi] = S
            ffn.b_up.data[unit] = -S * 1.5  # both must be ~1
            ffn.W_gate.data[unit, BD.MARK_AX] = 1.0  # only at AX marker
            ffn.W_down.data[op_dim, unit] = 10.0 / S  # scaled up: clean ALU -> OP ≈ 5
            unit += 1

        # First-step opcode decode at PC marker
        # (lo, hi, op_dim_attr_or_temp_idx)
        first_step_ops = [
            (2, 0, 'OP_JMP'),     # JMP
            (3, 0, 'TEMP+0'),     # JSR -> write to TEMP[0]
            (1, 0, 'OP_IMM'),     # IMM
            (0, 0, 'OP_LEA'),     # LEA
            (6, 2, 'OP_EXIT'),    # EXIT
            (7, 2, 'OP_NOP'),     # NOP
            (9, 1, 'OP_ADD'),     # ADD
            (10, 1, 'OP_SUB'),    # SUB
            (11, 1, 'OP_MUL'),    # MUL
            (12, 1, 'OP_DIV'),    # DIV
            (13, 1, 'OP_MOD'),    # MOD
            (14, 0, 'OP_OR'),     # OR
            (15, 0, 'OP_XOR'),    # XOR
            (0, 1, 'OP_AND'),     # AND
            (1, 1, 'OP_EQ'),      # EQ
            (3, 1, 'OP_LT'),      # LT
            (7, 1, 'OP_SHL'),     # SHL
            (8, 1, 'OP_SHR'),     # SHR
        ]

        for lo, hi, out in first_step_ops:
            ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo] = S
            ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi] = S
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.HAS_SE] = -S  # only when NOT HAS_SE (first step)
            ffn.b_up.data[unit] = -S * 2.5  # require all three conditions
            ffn.b_gate.data[unit] = 1.0  # always active when up > 0
            if out == 'TEMP+0':
                ffn.W_down.data[BD.TEMP + 0, unit] = 10.0 / S
            else:
                op_dim = getattr(BD, out)
                ffn.W_down.data[op_dim, unit] = 10.0 / S
            unit += 1

        # TEMP clearing at PC marker to prevent leakage from L5 attention
        # Skip TEMP[0] - used for IS_JSR flag
        for k in range(32):
            if k == 0:
                # Skip TEMP[0], leave unit with zero weights
                unit += 1
                continue
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.TEMP + k] = -1.0
            ffn.W_down.data[BD.TEMP + k, unit] = 2.0 / S
            unit += 1

    def _compile_l6_ffn(self, ffn):
        """Compile L6 FFN (output routing for AX, PC, SP, BP, STACK0).

        This layer routes computed values to OUTPUT_LO/HI for next-token prediction:
        - IMM: FETCH → OUTPUT
        - EXIT/NOP/JMP: AX_CARRY → OUTPUT
        - JMP PC override: cancel PC+5, write JMP target
        - PSH: SP -= 8, STACK0 = AX
        - HALT detection
        - SP/BP/STACK0 identity carry
        - Function call handling (JSR/ENT/LEV)
        """
        S = self.S
        unit = 0
        T = 4.0  # threshold for opcode + marker detection

        # === IMM: FETCH → OUTPUT ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_IMM] = S
            ffn.W_up.data[unit, BD.OP_EXIT] = -S * 20
            ffn.W_up.data[unit, BD.OP_JMP] = -S * 20
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.FETCH_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_IMM] = S
            ffn.W_up.data[unit, BD.OP_EXIT] = -S * 20
            ffn.W_up.data[unit, BD.OP_JMP] = -S * 20
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.FETCH_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === EXIT: AX_CARRY → OUTPUT ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_EXIT] = S
            ffn.W_up.data[unit, BD.OP_IMM] = -S * 20
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_EXIT] = S
            ffn.W_up.data[unit, BD.OP_IMM] = -S * 20
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === NOP: AX_CARRY → OUTPUT ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_NOP] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S * 10
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_NOP] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S * 10
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === JMP: AX_CARRY → OUTPUT (preserves AX through JMP) ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.W_up.data[unit, BD.IS_BYTE] = -S * 10
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.W_up.data[unit, BD.IS_BYTE] = -S * 10
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === JMP PC override: CMP[0] AND MARK_PC → cancel and write target ===
        T_jmp = 5.5
        # Cancel OUTPUT
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.CMP + 0] = S
            ffn.b_up.data[unit] = -S * T_jmp
            ffn.W_gate.data[unit, BD.OUTPUT_LO + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.CMP + 0] = S
            ffn.b_up.data[unit] = -S * T_jmp
            ffn.W_gate.data[unit, BD.OUTPUT_HI + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1
        # Add JMP target from AX_CARRY
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.CMP + 0] = S
            ffn.b_up.data[unit] = -S * T_jmp
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.CMP + 0] = S
            ffn.b_up.data[unit] = -S * T_jmp
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === First-step JMP PC override: OP_JMP AND NOT HAS_SE ===
        T_op_jmp = 4.5
        # Cancel OUTPUT
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = -S
            ffn.b_up.data[unit] = -S * (T_op_jmp + 0.5)
            ffn.W_gate.data[unit, BD.OUTPUT_LO + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = -S
            ffn.b_up.data[unit] = -S * (T_op_jmp + 0.5)
            ffn.W_gate.data[unit, BD.OUTPUT_HI + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1
        # Add JMP target with PC_OFFSET
        for k in range(16):
            new_k = (k + 2) % 16  # Add PC_OFFSET=2
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = -S
            ffn.b_up.data[unit] = -S * (T_op_jmp + 0.5)
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = -S
            ffn.b_up.data[unit] = -S * (T_op_jmp + 0.5)
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === All-step JMP PC override: OP_JMP + FETCH ===
        T_op_jmp_all = 4.5
        # Cancel OUTPUT
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.b_up.data[unit] = -S * T_op_jmp_all
            ffn.W_gate.data[unit, BD.OUTPUT_LO + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.b_up.data[unit] = -S * T_op_jmp_all
            ffn.W_gate.data[unit, BD.OUTPUT_HI + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1
        # Add JMP target from FETCH
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.b_up.data[unit] = -S * T_op_jmp_all
            ffn.W_gate.data[unit, BD.FETCH_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_JMP] = S
            ffn.b_up.data[unit] = -S * T_op_jmp_all
            ffn.W_gate.data[unit, BD.FETCH_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === HALT detection: CMP[1] AND NEXT_SE → NEXT_HALT ===
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.NEXT_SE] = S
        ffn.b_up.data[unit] = -S * 1.3
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.NEXT_HALT, unit] = 2.0 / S
        ffn.W_down.data[BD.NEXT_SE, unit] = -2.0 / S
        unit += 1

        # === TEMP clearing at PC marker (skip TEMP[0]) ===
        for k in range(32):
            if k == 0:
                unit += 1
                continue
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.IS_BYTE] = -S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.TEMP + k] = -1.0
            ffn.W_down.data[BD.TEMP + k, unit] = 2.0 / S
            unit += 1

        # === CMP[3] clearing at PC marker ===
        # FIX: L6 attention head 6 relays POP group flags to CMP[3] at all
        # positions it fires (SP, STACK0, BP, PC, MEM). At PC marker, CMP[3] is spurious.
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = -S
        ffn.b_up.data[unit] = -S * 0.5
        ffn.W_gate.data[unit, BD.CMP + 3] = -1.0  # Self-referential clearing
        ffn.W_down.data[BD.CMP + 3, unit] = 2.0 / S
        unit += 1

        # === SP/BP/STACK0 identity carry ===
        for marker_dim in [BD.MARK_SP, BD.MARK_BP, BD.MARK_STACK0]:
            for k in range(16):
                ffn.W_up.data[unit, marker_dim] = S
                ffn.W_up.data[unit, BD.IS_BYTE] = -S
                ffn.b_up.data[unit] = -S * 0.5
                ffn.W_gate.data[unit, BD.EMBED_LO + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
                unit += 1
            for k in range(16):
                ffn.W_up.data[unit, marker_dim] = S
                ffn.W_up.data[unit, BD.IS_BYTE] = -S
                ffn.b_up.data[unit] = -S * 0.5
                ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
                unit += 1

        # === PSH: SP -= 8 ===
        T_psh = 1.5
        for k in range(16):
            new_k = (k - 8) % 16
            ffn.W_up.data[unit, BD.PSH_AT_SP] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * T_psh
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        # HI nibble with borrow
        for k in range(16):
            new_k_borrow = (k - 1) % 16
            ffn.W_up.data[unit, BD.PSH_AT_SP] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * T_psh
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
            for lo_bit in range(8, 16):
                ffn.W_gate.data[unit, BD.EMBED_LO + lo_bit] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] += -2.0 / S
            unit += 1

        # === PSH: STACK0 = AX ===
        T_psh_s0 = 1.5
        for k in range(16):
            ffn.W_up.data[unit, BD.PSH_AT_SP] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * T_psh_s0
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = -1.0
            ffn.W_gate.data[unit, BD.ALU_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.PSH_AT_SP] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * T_psh_s0
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = -1.0
            ffn.W_gate.data[unit, BD.ALU_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === GETCHAR: AX = AX_CARRY ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_GETCHAR] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_GETCHAR] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === BZ: AX passthrough ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_BZ] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_BZ] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === BNZ: AX passthrough ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_BNZ] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_BNZ] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === PSH: AX passthrough ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_PSH] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_PSH] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === ADJ: AX passthrough (AX unchanged during stack adjust) ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ADJ] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ADJ] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === ADJ: SP writeback (route AX result → SP marker) ===
        T_adj = 1.5
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ADJ] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * T_adj
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = -1.0  # Cancel identity
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0  # Write result
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ADJ] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * T_adj
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = -1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === ENT: SP writeback (route AX result → SP marker) ===
        # FIX: Add HAS_SE gate. On first step, AX_CARRY is empty (no previous step
        # to relay from), causing garbage OUTPUT values. First-step ENT SP is
        # handled by separate units below.
        T_ent = 2.5  # Threshold: OP_ENT(5) + MARK_SP(1) + HAS_SE(1) = 7 > 2.5
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ENT] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = S  # Only subsequent steps
            ffn.b_up.data[unit] = -S * T_ent
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = -1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ENT] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = S  # Only subsequent steps
            ffn.b_up.data[unit] = -S * T_ent
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = -1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === ENT first-step SP byte 0 (32 units) ===
        # On first step, SP starts at 0xFF (stack top). ENT imm computes SP -= 8 + imm.
        # For byte 0: result = 0x00 - 8 - imm_byte0 = -8 - imm
        # Lo nibble: (-8 - FETCH_LO) mod 16
        # Hi nibble: (-1 - FETCH_HI) mod 16 (always borrow from lo since -8 - x < 0)
        # Condition: OP_ENT + MARK_SP + NOT HAS_SE
        T_ent_first = 1.5
        for imm_lo in range(16):
            result_lo = (-8 - imm_lo) % 16
            ffn.W_up.data[unit, BD.OP_ENT] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
            ffn.b_up.data[unit] = -S * T_ent_first
            ffn.W_gate.data[unit, BD.FETCH_LO + imm_lo] = 1.0  # Select based on imm lo nibble
            ffn.W_down.data[BD.OUTPUT_LO + result_lo, unit] = 5.0 / S  # Strong output
            unit += 1
        for imm_hi in range(16):
            result_hi = (-1 - imm_hi) % 16  # Always borrow from lo
            ffn.W_up.data[unit, BD.OP_ENT] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.W_up.data[unit, BD.HAS_SE] = -S * 10  # Block on subsequent steps
            ffn.b_up.data[unit] = -S * T_ent_first
            ffn.W_gate.data[unit, BD.FETCH_HI + imm_hi] = 1.0  # Select based on imm hi nibble
            ffn.W_down.data[BD.OUTPUT_HI + result_hi, unit] = 5.0 / S  # Strong output
            unit += 1

        # === LEA: AX = BP + offset ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEA] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.ALU_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEA] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
            ffn.W_up.data[unit, BD.IS_BYTE] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.ALU_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === Binary ops: AX = ALU result ===
        binary_ops = [BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                      BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_EQ, BD.OP_NE,
                      BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                      BD.OP_SHL, BD.OP_SHR, BD.OP_LI, BD.OP_LC]
        for op_dim in binary_ops:
            for k in range(16):
                ffn.W_up.data[unit, op_dim] = S
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
                ffn.W_up.data[unit, BD.IS_BYTE] = -S
                ffn.b_up.data[unit] = -S * T
                ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
                unit += 1
            for k in range(16):
                ffn.W_up.data[unit, op_dim] = S
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
                ffn.W_up.data[unit, BD.IS_BYTE] = -S
                ffn.b_up.data[unit] = -S * T
                ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
                unit += 1

        # === PUTCHAR: AX = AX_CARRY passthrough ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_PUTCHAR] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_PUTCHAR] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === SI/SC: Store operations - AX = popped value (was STACK0) ===
        for op_dim in [BD.OP_SI, BD.OP_SC]:
            for k in range(16):
                ffn.W_up.data[unit, op_dim] = S
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
                ffn.W_up.data[unit, BD.IS_BYTE] = -S
                ffn.b_up.data[unit] = -S * T
                ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
                unit += 1
            for k in range(16):
                ffn.W_up.data[unit, op_dim] = S
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.MARK_PC] = -S * 8
                ffn.W_up.data[unit, BD.IS_BYTE] = -S
                ffn.b_up.data[unit] = -S * T
                ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
                unit += 1

        # === Binary pop SP increment (SP += 8) ===
        T_pop = 0.8  # CMP[3] + MARK_SP threshold
        for k in range(16):
            new_k = (k + 8) % 16
            ffn.W_up.data[unit, BD.CMP + 3] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * T_pop
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] += -2.0 / S
            unit += 1
        # HI nibble with carry
        for k in range(16):
            new_k_carry = (k + 1) % 16
            ffn.W_up.data[unit, BD.CMP + 3] = S
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * T_pop
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
            for lo_bit in range(8, 16):
                ffn.W_gate.data[unit, BD.EMBED_LO + lo_bit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + new_k_carry, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] += -2.0 / S
            unit += 1

        # === Function call handling: JSR/ENT/LEV ===
        # LEA first-step: Initialize ALU with BP default
        ffn.W_up.data[unit, BD.OP_LEA] = S
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.ALU_LO + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.OP_LEA] = S
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.ALU_HI + 0, unit] = 2.0 / S
        unit += 1

        # Reserve units for more complex JSR/ENT/LEV operations
        # These are implemented starting at unit 1000 in manual code
        # For now, we place them starting at current unit position

        # === JSR: SP -= 8 at SP byte positions ===
        T_jsr_b0 = 3.5
        STACK0_I = 7  # STACK0 area index for H1
        for k in range(16):
            new_k = (k - 8) % 16
            ffn.W_up.data[unit, BD.CMP + 4] = S  # JSR flag from head 6
            ffn.W_up.data[unit, BD.BYTE_INDEX_0] = S
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.H1 + 2] = S  # SP area
            ffn.b_up.data[unit] = -S * T_jsr_b0
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] += -2.0 / S
            unit += 1
        # HI nibble with borrow
        for k in range(16):
            new_k_borrow = (k - 1) % 16
            ffn.W_up.data[unit, BD.CMP + 4] = S
            ffn.W_up.data[unit, BD.BYTE_INDEX_0] = S
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.H1 + 2] = S
            ffn.b_up.data[unit] = -S * T_jsr_b0
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
            for lo_bit in range(8):  # borrow when LO < 8
                ffn.W_gate.data[unit, BD.EMBED_LO + lo_bit] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] += -2.0 / S
            unit += 1

        # JSR byte 1: write 0xFF (constant)
        for k in range(16):
            ffn.W_up.data[unit, BD.CMP + 4] = S
            ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.H1 + 2] = S
            ffn.b_up.data[unit] = -S * T_jsr_b0
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 15, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] += -2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.CMP + 4] = S
            ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.H1 + 2] = S
            ffn.b_up.data[unit] = -S * T_jsr_b0
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 15, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] += -2.0 / S
            unit += 1

        # JSR bytes 2-3: identity passthrough (already handled by carry logic)
        # (32 units each for bytes 2 and 3)
        for byte_idx in [2, 3]:
            byte_index_dim = BD.BYTE_INDEX_2 if byte_idx == 2 else BD.BYTE_INDEX_3
            for k in range(16):
                ffn.W_up.data[unit, BD.CMP + 4] = S
                ffn.W_up.data[unit, byte_index_dim] = S
                ffn.W_up.data[unit, BD.IS_BYTE] = S
                ffn.W_up.data[unit, BD.H1 + 2] = S
                ffn.b_up.data[unit] = -S * T_jsr_b0
                ffn.W_gate.data[unit, BD.EMBED_LO + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
                unit += 1
            for k in range(16):
                ffn.W_up.data[unit, BD.CMP + 4] = S
                ffn.W_up.data[unit, byte_index_dim] = S
                ffn.W_up.data[unit, BD.IS_BYTE] = S
                ffn.W_up.data[unit, BD.H1 + 2] = S
                ffn.b_up.data[unit] = -S * T_jsr_b0
                ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
                unit += 1

        # JSR: STACK0 = return_addr (from AX_CARRY)
        T_jsr_s0 = 1.5
        for k in range(16):
            ffn.W_up.data[unit, BD.CMP + 4] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * T_jsr_s0
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = -1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.CMP + 4] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * T_jsr_s0
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = -1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # JSR: STACK0 byte positions
        T_jsr_s0_byte = 3.5
        for byte_idx in [0, 1, 2, 3]:
            byte_index_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_idx]
            for k in range(16):
                ffn.W_up.data[unit, BD.CMP + 4] = S
                ffn.W_up.data[unit, byte_index_dim] = S
                ffn.W_up.data[unit, BD.IS_BYTE] = S
                ffn.W_up.data[unit, BD.H1 + STACK0_I] = S
                ffn.b_up.data[unit] = -S * T_jsr_s0_byte
                ffn.W_gate.data[unit, BD.EMBED_LO + k] = -1.0
                ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
                unit += 1
            for k in range(16):
                ffn.W_up.data[unit, BD.CMP + 4] = S
                ffn.W_up.data[unit, byte_index_dim] = S
                ffn.W_up.data[unit, BD.IS_BYTE] = S
                ffn.W_up.data[unit, BD.H1 + STACK0_I] = S
                ffn.b_up.data[unit] = -S * T_jsr_s0_byte
                ffn.W_gate.data[unit, BD.EMBED_HI + k] = -1.0
                ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
                ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
                unit += 1

        # JSR: PC override (PC = FETCH target)
        T_jsr_pc = 1.3
        # Cancel OUTPUT
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.TEMP + 0] = S  # IS_JSR flag
            ffn.W_up.data[unit, BD.OP_NOP] = -S * 4
            ffn.W_up.data[unit, BD.OP_EXIT] = -S * 4
            ffn.W_up.data[unit, BD.OP_JMP] = -S * 4
            ffn.W_up.data[unit, BD.OP_BZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_BNZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_IMM] = -S * 4
            ffn.W_up.data[unit, BD.OP_LEV] = -S * 4
            ffn.W_up.data[unit, BD.OP_ENT] = -S * 4
            ffn.b_up.data[unit] = -S * T_jsr_pc
            ffn.W_gate.data[unit, BD.OUTPUT_LO + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.TEMP + 0] = S
            ffn.W_up.data[unit, BD.OP_NOP] = -S * 4
            ffn.W_up.data[unit, BD.OP_EXIT] = -S * 4
            ffn.W_up.data[unit, BD.OP_JMP] = -S * 4
            ffn.W_up.data[unit, BD.OP_BZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_BNZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_IMM] = -S * 4
            ffn.W_up.data[unit, BD.OP_LEV] = -S * 4
            ffn.W_up.data[unit, BD.OP_ENT] = -S * 4
            ffn.b_up.data[unit] = -S * T_jsr_pc
            ffn.W_gate.data[unit, BD.OUTPUT_HI + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1
        # Add FETCH target
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.TEMP + 0] = S
            ffn.W_up.data[unit, BD.OP_NOP] = -S * 4
            ffn.W_up.data[unit, BD.OP_EXIT] = -S * 4
            ffn.W_up.data[unit, BD.OP_JMP] = -S * 4
            ffn.W_up.data[unit, BD.OP_BZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_BNZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_IMM] = -S * 4
            ffn.W_up.data[unit, BD.OP_LEV] = -S * 4
            ffn.W_up.data[unit, BD.OP_ENT] = -S * 4
            ffn.b_up.data[unit] = -S * T_jsr_pc
            ffn.W_gate.data[unit, BD.FETCH_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.TEMP + 0] = S
            ffn.W_up.data[unit, BD.OP_NOP] = -S * 4
            ffn.W_up.data[unit, BD.OP_EXIT] = -S * 4
            ffn.W_up.data[unit, BD.OP_JMP] = -S * 4
            ffn.W_up.data[unit, BD.OP_BZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_BNZ] = -S * 4
            ffn.W_up.data[unit, BD.OP_IMM] = -S * 4
            ffn.W_up.data[unit, BD.OP_LEV] = -S * 4
            ffn.W_up.data[unit, BD.OP_ENT] = -S * 4
            ffn.b_up.data[unit] = -S * T_jsr_pc
            ffn.W_gate.data[unit, BD.FETCH_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # JSR: AX = FETCH (JSR returns address in AX for C4)
        T_jsr_ax = 1.5
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_JSR] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.b_up.data[unit] = -S * T_jsr_ax
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_JSR] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.b_up.data[unit] = -S * T_jsr_ax
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # ENT: STACK0 = old_BP (from TEMP)
        T_ent_s0 = 1.5
        for k in range(16):
            ffn.W_up.data[unit, BD.CMP + 2] = S  # OP_ENT relay
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * T_ent_s0
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = -1.0
            ffn.W_gate.data[unit, BD.TEMP + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.CMP + 2] = S
            ffn.W_up.data[unit, BD.MARK_STACK0] = S
            ffn.b_up.data[unit] = -S * T_ent_s0
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = -1.0
            ffn.W_gate.data[unit, BD.TEMP + 16 + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # ENT: BP = old_SP - 8 (from TEMP)
        T_ent_bp = 1.5
        for k in range(16):
            new_k = (k - 8) % 16
            ffn.W_up.data[unit, BD.CMP + 2] = S
            ffn.W_up.data[unit, BD.MARK_BP] = S
            ffn.b_up.data[unit] = -S * T_ent_bp
            ffn.W_gate.data[unit, BD.TEMP + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + new_k, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] += -2.0 / S
            unit += 1
        for k in range(16):
            new_k_borrow = (k - 1) % 16
            ffn.W_up.data[unit, BD.CMP + 2] = S
            ffn.W_up.data[unit, BD.MARK_BP] = S
            ffn.b_up.data[unit] = -S * T_ent_bp
            ffn.W_gate.data[unit, BD.TEMP + 16 + k] = 1.0
            for lo_bit in range(8):
                ffn.W_gate.data[unit, BD.TEMP + lo_bit] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + new_k_borrow, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] += -2.0 / S
            unit += 1

        # ENT: AX = FETCH (ENT N sets AX to N)
        T_ent_ax = 1.5
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ENT] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.b_up.data[unit] = -S * T_ent_ax
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_ENT] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.b_up.data[unit] = -S * T_ent_ax
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # LEV: AX = AX_CARRY (return value preserved)
        T_lev_ax = 1.5
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 15
            ffn.b_up.data[unit] = -S * T_lev_ax
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 15
            ffn.b_up.data[unit] = -S * T_lev_ax
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

    def _compile_l8_ffn(self, ffn):
        """Compile L8 FFN: ALU lo nibble (ADD/SUB/LEA/ADJ/ENT) + carry/borrow.

        Uses 3-way AND in silu path: MARK_AX + ALU_LO[a] + AX_CARRY_LO[b].
        Only the exact (a, b) pair fires.

        Total: ~1700 units.
        """
        S = self.S
        unit = 0

        # === ADD: lo nibble (256 units) ===
        for a in range(16):
            for b in range(16):
                result = (a + b) % 16
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.ALU_LO + a] = S
                ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up.data[unit] = -S * 2.5  # 3-way AND
                ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1

        # === LEA: lo nibble (256 units) ===
        # Like ADD but reads from FETCH_LO instead of AX_CARRY_LO
        # Require BOTH ALU_LO AND FETCH_LO to be active (threshold=105)
        for a in range(16):
            for b in range(16):
                result = (a + b) % 16
                ffn.W_up.data[unit, BD.MARK_AX] = S * 60
                ffn.W_up.data[unit, BD.ALU_LO + a] = S
                ffn.W_up.data[unit, BD.FETCH_LO + b] = S
                ffn.b_up.data[unit] = -S * 105
                ffn.W_gate.data[unit, BD.OP_LEA] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1

        # === SUB: lo nibble (256 units) ===
        # Correct C4 semantics: AX = stack_top - AX = ALU - AX_CARRY = a - b.
        for a in range(16):
            for b in range(16):
                result = (a - b) % 16
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.ALU_LO + a] = S
                ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up.data[unit] = -S * 2.5
                ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1

        # === ADD carry detection (120 units: pairs where a+b >= 16) ===
        for a in range(16):
            for b in range(16):
                if a + b >= 16:
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_LO + a] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
                    ffn.b_up.data[unit] = -S * 2.5
                    ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
                    ffn.W_down.data[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                    unit += 1

        # === LEA carry detection (120 units: pairs where a+b >= 16) ===
        for a in range(16):
            for b in range(16):
                if a + b >= 16:
                    ffn.W_up.data[unit, BD.MARK_AX] = S * 60
                    ffn.W_up.data[unit, BD.ALU_LO + a] = S
                    ffn.W_up.data[unit, BD.FETCH_LO + b] = S
                    ffn.b_up.data[unit] = -S * 105
                    ffn.W_gate.data[unit, BD.OP_LEA] = 1.0
                    ffn.W_down.data[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                    unit += 1

        # === ADJ: lo nibble (256 units) ===
        # Like LEA but gates on OP_ADJ
        for a in range(16):
            for b in range(16):
                result = (a + b) % 16
                ffn.W_up.data[unit, BD.MARK_AX] = S * 60
                ffn.W_up.data[unit, BD.ALU_LO + a] = S
                ffn.W_up.data[unit, BD.FETCH_LO + b] = S
                ffn.b_up.data[unit] = -S * 105
                ffn.W_gate.data[unit, BD.OP_ADJ] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1

        # === ADJ carry detection (120 units: pairs where a+b >= 16) ===
        for a in range(16):
            for b in range(16):
                if a + b >= 16:
                    ffn.W_up.data[unit, BD.MARK_AX] = S * 60
                    ffn.W_up.data[unit, BD.ALU_LO + a] = S
                    ffn.W_up.data[unit, BD.FETCH_LO + b] = S
                    ffn.b_up.data[unit] = -S * 105
                    ffn.W_gate.data[unit, BD.OP_ADJ] = 1.0
                    ffn.W_down.data[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                    unit += 1

        # === SUB borrow detection (120 units: pairs where a < b) ===
        for a in range(16):
            for b in range(16):
                if a < b:
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_LO + a] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
                    ffn.b_up.data[unit] = -S * 2.5
                    ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
                    ffn.W_down.data[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                    unit += 1

        # === ENT: lo nibble subtraction (256 units) ===
        # ENT computes: SP = SP - (8 + signed_immediate)
        for sp_lo in range(16):
            for imm_lo in range(16):
                effective_b = (8 + imm_lo) % 16
                result = (sp_lo - effective_b) % 16
                ffn.W_up.data[unit, BD.MARK_AX] = S * 60
                ffn.W_up.data[unit, BD.ALU_LO + sp_lo] = S
                ffn.W_up.data[unit, BD.FETCH_LO + imm_lo] = S
                ffn.b_up.data[unit] = -S * 105
                ffn.W_gate.data[unit, BD.OP_ENT] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1

        # === ENT borrow detection ===
        for sp_lo in range(16):
            for imm_lo in range(16):
                full_sum = 8 + imm_lo
                if sp_lo < (full_sum % 16) or full_sum >= 16:
                    ffn.W_up.data[unit, BD.MARK_AX] = S * 60
                    ffn.W_up.data[unit, BD.ALU_LO + sp_lo] = S
                    ffn.W_up.data[unit, BD.FETCH_LO + imm_lo] = S
                    ffn.b_up.data[unit] = -S * 105
                    ffn.W_gate.data[unit, BD.OP_ENT] = 1.0
                    ffn.W_down.data[BD.CARRY + 0, unit] = 2.0 / (S * 5.0)
                    unit += 1

        # === CMP_GROUP flag (1 unit) ===
        for op in [BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE]:
            ffn.W_up.data[unit, op] = S
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.CMP_GROUP, unit] = 2.0 / (S * 9.0)
        unit += 1

        # === LEV: BP address relay (BP OUTPUT → ADDR dims) ===
        # Byte 0 lo nibble: OUTPUT_LO → ADDR_B0_LO
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S
            ffn.W_up.data[unit, BD.MARK_BP] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 10
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.OUTPUT_LO + k] = 1.0
            ffn.W_down.data[BD.ADDR_B0_LO + k, unit] = 2.0 / (S * 9)
            unit += 1

        # Byte 0 hi nibble: OUTPUT_HI → ADDR_B0_HI
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S
            ffn.W_up.data[unit, BD.MARK_BP] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 10
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.OUTPUT_HI + k] = 1.0
            ffn.W_down.data[BD.ADDR_B0_HI + k, unit] = 2.0 / (S * 9)
            unit += 1

        # Bytes 1-2: Set to zero
        ffn.W_up.data[unit, BD.OP_LEV] = S
        ffn.W_up.data[unit, BD.MARK_BP] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.CONST] = 1.0
        ffn.W_down.data[BD.ADDR_B1_LO + 0, unit] = 2.0 / (S * 9)
        unit += 1

        ffn.W_up.data[unit, BD.OP_LEV] = S
        ffn.W_up.data[unit, BD.MARK_BP] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.CONST] = 1.0
        ffn.W_down.data[BD.ADDR_B2_LO + 0, unit] = 2.0 / (S * 9)
        unit += 1

    def _compile_l9_ffn(self, ffn):
        """Compile L9 FFN: ADD/SUB hi nibble + LEA/ADJ/ENT hi nibble + comparison flags.

        Reference: _set_layer9_alu in vm_step.py
        """
        S = self.S
        unit = 0

        # === ADD hi nibble (no carry 256 + with carry 256 = 512 units) ===
        for carry_in in [0, 1]:
            for a in range(16):
                for b in range(16):
                    result = (a + b + carry_in) % 16
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_HI + a] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
                    if carry_in == 0:
                        ffn.W_up.data[unit, BD.CARRY + 0] = -S * 2.0
                        ffn.b_up.data[unit] = -S * 2.5
                    else:
                        ffn.W_up.data[unit, BD.CARRY + 0] = S * 2.0
                        ffn.b_up.data[unit] = -S * 4.5
                    ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
                    ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 2.0 / S
                    unit += 1

        # === LEA hi nibble (no carry 256 + with carry 256 = 512 units) ===
        for carry_in in [0, 1]:
            for a in range(16):
                for b in range(16):
                    result = (a + b + carry_in) % 16
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_HI + a] = S
                    ffn.W_up.data[unit, BD.FETCH_HI + b] = S
                    if carry_in == 0:
                        ffn.W_up.data[unit, BD.CARRY + 0] = -0.01
                        ffn.b_up.data[unit] = -S * 58
                    else:
                        ffn.W_up.data[unit, BD.CARRY + 0] = 0.01
                        ffn.b_up.data[unit] = -S * 58.4
                    ffn.W_gate.data[unit, BD.OP_LEA] = 1.0
                    ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 2.0 / S
                    unit += 1

        # === ADJ hi nibble (no carry 256 + with carry 256 = 512 units) ===
        for carry_in in [0, 1]:
            for a in range(16):
                for b in range(16):
                    result = (a + b + carry_in) % 16
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_HI + a] = S
                    ffn.W_up.data[unit, BD.FETCH_HI + b] = S
                    if carry_in == 0:
                        ffn.W_up.data[unit, BD.CARRY + 0] = -0.01
                        ffn.b_up.data[unit] = -S * 58
                    else:
                        ffn.W_up.data[unit, BD.CARRY + 0] = 0.01
                        ffn.b_up.data[unit] = -S * 58.4
                    ffn.W_gate.data[unit, BD.OP_ADJ] = 1.0
                    ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 2.0 / S
                    unit += 1

        # === SUB hi nibble (no borrow 256 + with borrow 256 = 512 units) ===
        for borrow_in in [0, 1]:
            for a in range(16):
                for b in range(16):
                    result = (a - b - borrow_in) % 16
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_HI + a] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
                    if borrow_in == 0:
                        ffn.W_up.data[unit, BD.CARRY + 0] = -S * 2.0
                        ffn.b_up.data[unit] = -S * 2.5
                    else:
                        ffn.W_up.data[unit, BD.CARRY + 0] = S * 2.0
                        ffn.b_up.data[unit] = -S * 4.5
                    ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
                    ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 2.0 / S
                    unit += 1

        # === ENT hi nibble (no borrow 256 + with borrow 256 = 512 units) ===
        # Fix 2026-04-17: Add MARK_SP blocker to prevent firing at SP marker
        for borrow_in in [0, 1]:
            for sp_hi in range(16):
                for imm_hi in range(16):
                    result = (sp_hi - imm_hi - borrow_in) % 16
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.MARK_SP] = -S * 2.0  # Block at SP marker
                    ffn.W_up.data[unit, BD.ALU_HI + sp_hi] = S
                    ffn.W_up.data[unit, BD.FETCH_HI + imm_hi] = S
                    if borrow_in == 0:
                        ffn.W_up.data[unit, BD.CARRY + 0] = -S * 2.0
                        ffn.b_up.data[unit] = -S * 58
                    else:
                        ffn.W_up.data[unit, BD.CARRY + 0] = S * 2.0
                        ffn.b_up.data[unit] = -S * 60
                    ffn.W_gate.data[unit, BD.OP_ENT] = 1.0
                    ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 2.0 / S
                    unit += 1

        # === Comparison flags (hi_eq, lo_eq, hi_lt, lo_lt) ===
        # hi_eq: 16 units
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.ALU_HI + k] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + k] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.CMP_GROUP] = 1.0
            ffn.W_down.data[BD.CMP + 1, unit] = 2.0 / S
            unit += 1

        # lo_eq: 16 units
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_AX] = S
            ffn.W_up.data[unit, BD.ALU_LO + k] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + k] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.CMP_GROUP] = 1.0
            ffn.W_down.data[BD.CMP + 2, unit] = 2.0 / S
            unit += 1

        # hi_lt: 120 units (a < b)
        for a in range(16):
            for b in range(a + 1, 16):
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.ALU_HI + a] = S
                ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
                ffn.b_up.data[unit] = -S * 2.5
                ffn.W_gate.data[unit, BD.CMP_GROUP] = 1.0
                ffn.W_down.data[BD.CMP + 0, unit] = 2.0 / S
                unit += 1

        # lo_lt: 120 units
        for a in range(16):
            for b in range(a + 1, 16):
                ffn.W_up.data[unit, BD.MARK_AX] = S
                ffn.W_up.data[unit, BD.ALU_LO + a] = S
                ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
                ffn.b_up.data[unit] = -S * 2.5
                ffn.W_gate.data[unit, BD.CMP_GROUP] = 1.0
                ffn.W_down.data[BD.CMP + 3, unit] = 2.0 / S
                unit += 1

        # === ADD hi-nibble carry-out → CARRY[1] ===
        for carry_in in [0, 1]:
            for a in range(16):
                for b in range(16):
                    if a + b + carry_in < 16:
                        continue
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_HI + a] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
                    if carry_in == 0:
                        ffn.W_up.data[unit, BD.CARRY + 0] = -0.01
                        ffn.b_up.data[unit] = -S * 2.5
                    else:
                        ffn.W_up.data[unit, BD.CARRY + 0] = 0.01
                        ffn.b_up.data[unit] = -S * 2.9
                    ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
                    ffn.W_down.data[BD.CARRY + 1, unit] = 2.0 / (S * 5.0)
                    unit += 1

        # === SUB hi-nibble borrow-out → CARRY[2] ===
        for borrow_in in [0, 1]:
            for a in range(16):
                for b in range(16):
                    if borrow_in == 0:
                        if a >= b:
                            continue
                    else:
                        if a > b:
                            continue
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_HI + a] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
                    if borrow_in == 0:
                        ffn.W_up.data[unit, BD.CARRY + 0] = -0.01
                        ffn.b_up.data[unit] = -S * 2.5
                    else:
                        ffn.W_up.data[unit, BD.CARRY + 0] = 0.01
                        ffn.b_up.data[unit] = -S * 2.9
                    ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
                    ffn.W_down.data[BD.CARRY + 2, unit] = 2.0 / (S * 5.0)
                    unit += 1

        # === ALU clearing for non-ALU opcodes ===
        non_alu_opcodes = [
            BD.OP_IMM, BD.OP_NOP, BD.OP_JMP, BD.OP_JSR, BD.OP_EXIT,
            BD.OP_BZ, BD.OP_BNZ, BD.OP_ENT, BD.OP_ADJ, BD.OP_LEV,
            BD.OP_PSH, BD.OP_LI, BD.OP_LC, BD.OP_SI, BD.OP_SC,
        ]
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_AX] = S
            for op_dim in non_alu_opcodes:
                ffn.W_up.data[unit, op_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.ALU_LO + k, unit] = -10.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_AX] = S
            for op_dim in non_alu_opcodes:
                ffn.W_up.data[unit, op_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[BD.ALU_HI + k, unit] = -10.0 / S
            unit += 1

        # === LEV: Add +8 offset to ADDR_B0 at PC marker ===
        for k in range(16):
            new_k = (k + 8) % 16
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.OP_LEV] = S / 5
            ffn.W_up.data[unit, BD.MARK_BP] = -S * 10
            ffn.W_up.data[unit, BD.MARK_SP] = -S * 10
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.ADDR_B0_LO + k] = 1.0
            ffn.b_gate.data[unit] = -2.5
            ffn.W_down.data[BD.ADDR_B0_LO + k, unit] = -0.67 / S
            ffn.W_down.data[BD.ADDR_B0_LO + new_k, unit] = 0.67 / S
            unit += 1

        # Set ADDR_B1 = 0xff at PC marker for stack addresses
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.OP_LEV] = S / 5
        ffn.W_up.data[unit, BD.MARK_BP] = -S * 10
        ffn.W_up.data[unit, BD.MARK_SP] = -S * 10
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.CONST] = 1.0
        ffn.W_down.data[BD.ADDR_B1_LO + 15, unit] = 0.22 / S
        unit += 1

        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.OP_LEV] = S / 5
        ffn.W_up.data[unit, BD.MARK_BP] = -S * 10
        ffn.W_up.data[unit, BD.MARK_SP] = -S * 10
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.CONST] = 1.0
        ffn.W_down.data[BD.ADDR_B1_HI + 15, unit] = 0.22 / S
        unit += 1

        # Cascade carry for BP=0xfff8 + 8 = 0x10000
        # FIX 2026-04-16: Gate on BOTH LO[0] AND HI[15] to distinguish carry case.
        # After +8 shift: BP=0xf8 has LO[0] high (shifted from LO[8]), BP=0xf0 has LO[8] high.
        # Unit 1: Clear ADDR_B0_HI[15]
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.OP_LEV] = S / 5
        ffn.W_up.data[unit, BD.MARK_BP] = -S * 10
        ffn.W_up.data[unit, BD.MARK_SP] = -S * 10
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.ADDR_B0_LO + 0] = 1.0
        ffn.W_gate.data[unit, BD.ADDR_B0_HI + 15] = 1.0
        ffn.b_gate.data[unit] = -15.0
        ffn.W_down.data[BD.ADDR_B0_HI + 15, unit] = -0.67 / S
        ffn.W_down.data[BD.ADDR_B0_HI + 0, unit] = 0.67 / S
        unit += 1

        # Unit 2: Cancel ADDR_B1_LO[15]
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.OP_LEV] = S / 5
        ffn.W_up.data[unit, BD.MARK_BP] = -S * 10
        ffn.W_up.data[unit, BD.MARK_SP] = -S * 10
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.ADDR_B0_LO + 0] = 1.0
        ffn.W_gate.data[unit, BD.ADDR_B0_HI + 15] = 1.0
        ffn.b_gate.data[unit] = -15.0
        ffn.W_down.data[BD.ADDR_B1_LO + 15, unit] = -0.5 / S
        ffn.W_down.data[BD.ADDR_B1_LO + 0, unit] = 0.5 / S
        unit += 1

        # Unit 3: Cancel ADDR_B1_HI[15]
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.OP_LEV] = S / 5
        ffn.W_up.data[unit, BD.MARK_BP] = -S * 10
        ffn.W_up.data[unit, BD.MARK_SP] = -S * 10
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.ADDR_B0_LO + 0] = 1.0
        ffn.W_gate.data[unit, BD.ADDR_B0_HI + 15] = 1.0
        ffn.b_gate.data[unit] = -15.0
        ffn.W_down.data[BD.ADDR_B1_HI + 15, unit] = -0.5 / S
        ffn.W_down.data[BD.ADDR_B1_HI + 0, unit] = 0.5 / S
        unit += 1

        # Unit 4: Set ADDR_B2_LO[1] = 1
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.W_up.data[unit, BD.OP_LEV] = S / 5
        ffn.W_up.data[unit, BD.MARK_BP] = -S * 10
        ffn.W_up.data[unit, BD.MARK_SP] = -S * 10
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.ADDR_B0_LO + 0] = 1.0
        ffn.W_gate.data[unit, BD.ADDR_B0_HI + 15] = 1.0
        ffn.b_gate.data[unit] = -15.0
        ffn.W_down.data[BD.ADDR_B2_LO + 1, unit] = 0.67 / S
        unit += 1

    def _compile_l10_ffn(self, ffn):
        """Compile L10 FFN: Comparison combine + Bitwise ops + MUL lo nibble.

        Reference: _set_layer10_alu in vm_step.py
        """
        S = self.S
        unit = 0

        # --- Comparison combine (18 units) ---
        # EQ: default=0, override to 1 when hi_eq AND lo_eq
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.OP_EQ] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1
        # EQ override
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 2] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_EQ] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -4.0 / S
        unit += 1

        # NE: default=1, override to 0 when hi_eq AND lo_eq
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.OP_NE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 2] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_NE] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = -4.0 / S
        unit += 1

        # LT: default=0, override to 1 when hi_lt OR (hi_eq AND lo_lt)
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.OP_LT] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 0] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.OP_LT] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -4.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 3] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_LT] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -4.0 / S
        unit += 1

        # GT: default=1, override to 0 when hi_lt OR (hi_eq AND lo_lt) OR (hi_eq AND lo_eq)
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.OP_GT] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 0] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.OP_GT] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = -4.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 3] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_GT] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = -4.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 2] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_GT] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = -4.0 / S
        unit += 1

        # LE: default=0, override to 1 when hi_lt OR (hi_eq AND lo_lt) OR (hi_eq AND lo_eq)
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.OP_LE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 0] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.OP_LE] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -4.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 3] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_LE] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -4.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 2] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_LE] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -4.0 / S
        unit += 1

        # GE: default=1, override to 0 when hi_lt OR (hi_eq AND lo_lt)
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.OP_GE] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 0] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.OP_GE] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = -4.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.MARK_AX] = S
        ffn.W_up.data[unit, BD.CMP + 1] = S
        ffn.W_up.data[unit, BD.CMP + 3] = S
        ffn.b_up.data[unit] = -S * 2.5
        ffn.W_gate.data[unit, BD.OP_GE] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = -4.0 / S
        unit += 1

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
                    ffn.W_up.data[unit, BD.MARK_AX] = S * 40
                    ffn.W_up.data[unit, BD.ALU_LO + a] = S * 30
                    ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S * 30
                    ffn.b_up.data[unit] = -S * 80
                    ffn.W_gate.data[unit, op_dim] = 1.0
                    ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
                    unit += 1
            # Hi nibble (256 units)
            for a in range(16):
                for b in range(16):
                    result = op_fn(a, b)
                    ffn.W_up.data[unit, BD.MARK_AX] = S * 40
                    ffn.W_up.data[unit, BD.ALU_HI + a] = S * 30
                    ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S * 30
                    ffn.b_up.data[unit] = -S * 80
                    ffn.W_gate.data[unit, op_dim] = 1.0
                    ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 2.0 / S
                    unit += 1

        # --- MUL lo nibble (256 units) ---
        for a in range(16):
            for b in range(16):
                result = (a * b) % 16
                ffn.W_up.data[unit, BD.MARK_AX] = S * 40
                ffn.W_up.data[unit, BD.ALU_LO + a] = S * 30
                ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S * 30
                ffn.b_up.data[unit] = -S * 80
                ffn.W_gate.data[unit, BD.OP_MUL] = 1.0
                ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
                unit += 1

        # --- SHL/SHR zero output for shift >= 8 (8 units) ---
        for op_dim in [BD.OP_SHL, BD.OP_SHR]:
            # Case A: shift >= 16 (hi nibble non-zero)
            ffn.W_up.data[unit, BD.MARK_AX] = S * 60
            ffn.W_up.data[unit, BD.AX_CARRY_HI + 0] = -S
            ffn.b_up.data[unit] = -S * 59
            ffn.W_gate.data[unit, op_dim] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1
            # Case B: shift 8-15 (hi=0, lo>=8)
            ffn.W_up.data[unit, BD.MARK_AX] = S * 60
            ffn.W_up.data[unit, BD.AX_CARRY_HI + 0] = S
            for k in range(8, 16):
                ffn.W_up.data[unit, BD.AX_CARRY_LO + k] = S
            ffn.b_up.data[unit] = -S * 80
            ffn.W_gate.data[unit, op_dim] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
            unit += 1

        # --- AX passthrough (32 units) ---
        suppressed_ops = [
            BD.OP_IMM, BD.OP_ADD, BD.OP_SUB, BD.OP_OR, BD.OP_XOR, BD.OP_AND,
            BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
            BD.OP_MUL, BD.OP_DIV, BD.OP_MOD, BD.OP_SHL, BD.OP_SHR, BD.OP_LEA,
            BD.OP_LI, BD.OP_LC,
        ]
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_AX] = S
            for op_dim in suppressed_ops:
                ffn.W_up.data[unit, op_dim] = -S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.MARK_AX] = S
            for op_dim in suppressed_ops:
                ffn.W_up.data[unit, op_dim] = -S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # --- Inter-byte carry/borrow override at AX byte positions ---
        AX_IDX = 1
        # SUB borrow → all AX bytes = 0xFF
        for out_dim in [BD.OUTPUT_LO + 15, BD.OUTPUT_HI + 15]:
            ffn.W_up.data[unit, BD.CARRY + 2] = S
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.OP_SUB] = S
            ffn.b_up.data[unit] = -S * 5.5
            ffn.W_gate.data[unit, BD.H1 + AX_IDX] = 1.0
            ffn.W_down.data[out_dim, unit] = 10.0 / S
            unit += 1

        # ADD carry → AX byte 0 only = 0x01
        for out_dim in [BD.OUTPUT_LO + 1, BD.OUTPUT_HI + 0]:
            ffn.W_up.data[unit, BD.CARRY + 1] = S
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.BYTE_INDEX_0] = S
            ffn.W_up.data[unit, BD.OP_ADD] = S
            ffn.b_up.data[unit] = -S * 5.5
            ffn.W_gate.data[unit, BD.H1 + AX_IDX] = 1.0
            ffn.W_down.data[out_dim, unit] = 10.0 / S
            unit += 1

    def _compile_l11_ffn(self, ffn):
        """Compile L11 FFN: MUL partial products.

        Reference: _set_layer11_mul_partial in vm_step.py
        """
        S = self.S
        unit = 0

        # 4096 units for (a_lo, b_lo, b_hi) triples
        for a_lo in range(16):
            for b_lo in range(16):
                carry = (a_lo * b_lo) // 16
                for b_hi in range(16):
                    partial = (carry + a_lo * b_hi) % 16
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.ALU_LO + a_lo] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
                    ffn.b_up.data[unit] = -S * 3.5
                    ffn.W_gate.data[unit, BD.OP_MUL] = 1.0
                    ffn.W_down.data[BD.TEMP + partial, unit] = 2.0 / S
                    unit += 1

    def _compile_l12_ffn(self, ffn):
        """Compile L12 FFN: MUL hi nibble from partial + a_hi*b_lo.

        Reference: _set_layer12_mul_combine in vm_step.py
        """
        S = self.S
        unit = 0

        # 4096 units for (partial, a_hi, b_lo) triples
        for partial in range(16):
            for a_hi in range(16):
                for b_lo in range(16):
                    result_hi = (partial + a_hi * b_lo) % 16
                    ffn.W_up.data[unit, BD.MARK_AX] = S
                    ffn.W_up.data[unit, BD.TEMP + partial] = S
                    ffn.W_up.data[unit, BD.ALU_HI + a_hi] = S
                    ffn.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
                    ffn.b_up.data[unit] = -S * 7.5
                    ffn.W_gate.data[unit, BD.OP_MUL] = 1.0
                    ffn.W_down.data[BD.OUTPUT_HI + result_hi, unit] = 2.0 / S
                    unit += 1

    def _compile_l13_ffn(self, ffn):
        """Compile L13 FFN: SHL/SHR for shift amounts 0-7.

        Reference: _set_layer13_shifts in vm_step.py
        """
        S = self.S
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

                        ffn.W_up.data[unit, BD.MARK_AX] = S
                        ffn.W_up.data[unit, BD.ALU_LO + a_lo] = S
                        ffn.W_up.data[unit, BD.ALU_HI + a_hi] = S
                        ffn.W_up.data[unit, BD.AX_CARRY_LO + s] = S
                        ffn.W_up.data[unit, BD.AX_CARRY_HI + 0] = S
                        ffn.b_up.data[unit] = -S * 4.5
                        ffn.W_gate.data[unit, op_dim] = 1.0
                        ffn.W_down.data[BD.OUTPUT_LO + result_lo, unit] = 2.0 / S
                        ffn.W_down.data[BD.OUTPUT_HI + result_hi, unit] = 2.0 / S
                        unit += 1

    def _compile_l14_ffn(self, ffn):
        """Compile L14 FFN (TEMP clear + ADDR_KEY cleanup for LEV)."""
        S = self.S
        unit = 0

        # === Clear TEMP[0] at PC marker when OP_LEV active ===
        ffn.W_up.data[unit, BD.OP_LEV] = S / 10  # ~1 with OP_LEV≈10
        ffn.W_up.data[unit, BD.MARK_PC] = S
        ffn.b_up.data[unit] = -S * 1.5
        ffn.W_gate.data[unit, BD.CONST] = 1.0
        ffn.W_down.data[BD.TEMP + 0, unit] = -5.0 / S  # Subtract to clear residual
        unit += 1

        # === Clear ADDR_KEY pollution at non-MEM positions ===
        # Clear at positions that are NOT MEM value bytes AND NOT query markers
        suppress = S * 100

        for k in range(48):  # ADDR_KEY is 48 dims
            # Suppress at MEM value positions
            ffn.W_up.data[unit, BD.MEM_VAL_B0] = -suppress
            ffn.W_up.data[unit, BD.MEM_VAL_B1] = -suppress
            ffn.W_up.data[unit, BD.MEM_VAL_B2] = -suppress
            ffn.W_up.data[unit, BD.MEM_VAL_B3] = -suppress

            # Suppress at register markers
            ffn.W_up.data[unit, BD.MARK_PC] = -suppress
            ffn.W_up.data[unit, BD.MARK_BP] = -suppress
            ffn.W_up.data[unit, BD.MARK_AX] = -suppress
            ffn.W_up.data[unit, BD.MARK_STACK0] = -suppress
            ffn.W_up.data[unit, BD.MARK_SP] = -suppress

            # Positive bias to fire at non-MEM, non-marker positions
            ffn.b_up.data[unit] = S * 0.5

            ffn.W_gate.data[unit, BD.CONST] = 1.0
            ffn.W_down.data[BD.ADDR_KEY + k, unit] = -4.0 / S
            unit += 1

    def _compile_l15_ffn(self, ffn):
        """Compile L15 FFN (nibble copy: EMBED → OUTPUT for bytes)."""
        S = self.S
        unit = 0
        PC_I = 0
        SP_I = 2
        BP_I = 3
        AX_I = 1

        # LO nibbles: copy when IS_BYTE AND NOT at register areas
        for k in range(16):
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.H1 + PC_I] = -S  # Suppress ALL PC bytes (L3 handles all)
            ffn.W_up.data[unit, BD.H1 + AX_I] = -S  # Suppress at AX
            ffn.W_up.data[unit, BD.H1 + SP_I] = -S  # Suppress at SP
            ffn.W_up.data[unit, BD.H1 + BP_I] = -S  # Suppress at BP
            ffn.W_up.data[unit, BD.H4 + BP_I] = -S  # Suppress at STACK0 area
            ffn.W_up.data[unit, BD.MEM_STORE] = -S  # Suppress at MEM during PSH/SI/SC
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.EMBED_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1

        # HI nibbles: same logic
        for k in range(16):
            ffn.W_up.data[unit, BD.IS_BYTE] = S
            ffn.W_up.data[unit, BD.H1 + PC_I] = -S  # Suppress ALL PC bytes
            ffn.W_up.data[unit, BD.H1 + AX_I] = -S
            ffn.W_up.data[unit, BD.H1 + SP_I] = -S
            ffn.W_up.data[unit, BD.H1 + BP_I] = -S
            ffn.W_up.data[unit, BD.H4 + BP_I] = -S
            ffn.W_up.data[unit, BD.MEM_STORE] = -S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, BD.EMBED_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === PSH SP byte outputs ===
        T_psh_byte = 3.5

        # SP byte 0 pos → predict byte 1 = 0xFF
        ffn.W_up.data[unit, BD.PSH_AT_SP] = S
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_0] = S
        ffn.b_up.data[unit] = -S * T_psh_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 15, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -2.0 / S  # cancel L3 default
        unit += 1
        ffn.W_up.data[unit, BD.PSH_AT_SP] = S
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_0] = S
        ffn.b_up.data[unit] = -S * T_psh_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 15, unit] = 2.0 / S
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = -2.0 / S
        unit += 1

        # SP byte 1 pos → predict byte 2 = 0x00
        ffn.W_up.data[unit, BD.PSH_AT_SP] = S
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.b_up.data[unit] = -S * T_psh_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.PSH_AT_SP] = S
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.b_up.data[unit] = -S * T_psh_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # SP byte 2 pos → predict byte 3 = 0x00
        ffn.W_up.data[unit, BD.PSH_AT_SP] = S
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_2] = S
        ffn.b_up.data[unit] = -S * T_psh_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 2.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.PSH_AT_SP] = S
        ffn.W_up.data[unit, BD.H1 + SP_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_2] = S
        ffn.b_up.data[unit] = -S * T_psh_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

        # === LEA first-step AX byte 2 output ===
        # For LEA, AX = BP + imm. BP = 0x10000, so byte 2 = 0x01.
        T_lea_byte = 4.5
        ffn.W_up.data[unit, BD.CMP + 7] = S  # OP_LEA relay
        ffn.W_up.data[unit, BD.H1 + AX_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S  # Only first step
        ffn.b_up.data[unit] = -S * T_lea_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + 1, unit] = 4.0 / S
        ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = -4.0 / S
        unit += 1
        ffn.W_up.data[unit, BD.CMP + 7] = S  # OP_LEA relay
        ffn.W_up.data[unit, BD.H1 + AX_I] = S
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        ffn.W_up.data[unit, BD.BYTE_INDEX_1] = S
        ffn.W_up.data[unit, BD.HAS_SE] = -S
        ffn.b_up.data[unit] = -S * T_lea_byte
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 2.0 / S
        unit += 1

    def _compile_l16_ffn(self, ffn):
        """Compile L16 FFN (LEV routing: SP = BP + 16)."""
        S = self.S
        unit = 0

        # === Cancel OUTPUT at SP marker during LEV ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S / 5
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.OUTPUT_LO + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S / 5
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.OUTPUT_HI + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === SP = BP + 16: Lo nibble ===
        # Adding 16 to nibble k gives k (with carry to hi)
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S / 10
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.W_up.data[unit, BD.MARK_BP] = -S * 15
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 15
            ffn.W_up.data[unit, BD.MARK_AX] = -S * 50
            ffn.W_up.data[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up.data[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up.data[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up.data[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up.data[unit, BD.ADDR_B0_LO + k] = S
            ffn.b_up.data[unit] = -S * 3.0
            ffn.W_gate.data[unit, BD.ADDR_B0_LO + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1

        # === SP = BP + 16: Hi nibble (k + 1) % 16 ===
        for k in range(16):
            result = (k + 1) % 16
            ffn.W_up.data[unit, BD.OP_LEV] = S / 10
            ffn.W_up.data[unit, BD.MARK_SP] = S
            ffn.W_up.data[unit, BD.MARK_BP] = -S * 15
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 15
            ffn.W_up.data[unit, BD.MARK_AX] = -S * 50
            ffn.W_up.data[unit, BD.BYTE_INDEX_0] = -S * 10
            ffn.W_up.data[unit, BD.BYTE_INDEX_1] = -S * 10
            ffn.W_up.data[unit, BD.BYTE_INDEX_2] = -S * 10
            ffn.W_up.data[unit, BD.BYTE_INDEX_3] = -S * 10
            ffn.W_up.data[unit, BD.ADDR_B0_HI + k] = S
            ffn.b_up.data[unit] = -S * 3.0
            ffn.W_gate.data[unit, BD.ADDR_B0_HI + k] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 2.0 / S
            unit += 1

        # === Cancel OUTPUT_HI at PC marker during LEV (for return_addr) ===
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S / 5
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, BD.OUTPUT_HI + k] = -1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === Route TEMP → OUTPUT at PC marker for return_addr ===
        # TEMP_LO → OUTPUT_LO
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S / 10
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.TEMP + k] = S
            ffn.b_up.data[unit] = -S * 2.0
            ffn.W_gate.data[unit, BD.CONST] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + k, unit] = 2.0 / S
            unit += 1
        # TEMP_HI → OUTPUT_HI
        for k in range(16):
            ffn.W_up.data[unit, BD.OP_LEV] = S / 10
            ffn.W_up.data[unit, BD.MARK_PC] = S
            ffn.W_up.data[unit, BD.TEMP + 16 + k] = S
            ffn.b_up.data[unit] = -S * 2.0
            ffn.W_gate.data[unit, BD.CONST] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + k, unit] = 2.0 / S
            unit += 1

        # === Set OUTPUT = 0x00 at byte positions 1-3 for LEV ===
        # For return_addr < 256, bytes 1-3 are always 0.
        # 0x00 = lo nibble 0, hi nibble 0 → OUTPUT_LO[0]=1, OUTPUT_HI[0]=1

        # Clear OUTPUT_LO[10] at byte positions 1-3
        for byte_pos in range(3):
            byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
            ffn.W_up.data[unit, BD.OP_LEV] = S / 2
            ffn.W_up.data[unit, byte_idx_dim] = S
            # Suppress at marker positions
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_AX] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_SP] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_BP] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_STACK0] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_MEM] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_SE] = -S * 1.5
            ffn.b_up.data[unit] = -S * 4
            ffn.W_gate.data[unit, BD.CONST] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 10, unit] = -10.0 / S
            unit += 1

        # Set OUTPUT_LO[0] = 1 at byte positions 1-3
        for byte_pos in range(3):
            byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
            ffn.W_up.data[unit, BD.OP_LEV] = S / 2
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_AX] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_SP] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_BP] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_STACK0] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_MEM] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_SE] = -S * 1.5
            ffn.b_up.data[unit] = -S * 4
            ffn.W_gate.data[unit, BD.CONST] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + 0, unit] = 5.0 / S
            unit += 1

        # Set OUTPUT_HI[0] = 1 at byte positions 1-3
        for byte_pos in range(3):
            byte_idx_dim = [BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3][byte_pos]
            ffn.W_up.data[unit, BD.OP_LEV] = S / 2
            ffn.W_up.data[unit, byte_idx_dim] = S
            ffn.W_up.data[unit, BD.MARK_PC] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_AX] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_SP] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_BP] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_STACK0] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_MEM] = -S * 1.5
            ffn.W_up.data[unit, BD.NEXT_SE] = -S * 1.5
            ffn.b_up.data[unit] = -S * 4
            ffn.W_gate.data[unit, BD.CONST] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + 0, unit] = 5.0 / S
            unit += 1

    def _resize_l15_attention(self, attn, d_model):
        """Resize L15 attention to 12 heads (768 dims) for LEV support."""
        import torch.nn as nn

        num_heads_l15 = 12
        head_dim = d_model // 8  # 64 (based on default 8 heads)
        new_q_rows = num_heads_l15 * head_dim  # 12 * 64 = 768

        # Update num_heads and head_dim attributes
        attn.num_heads = num_heads_l15
        attn.head_dim = head_dim

        # Resize ALiBi slopes if present (8 slopes -> 12 slopes)
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            new_slopes = torch.tensor(
                [2.0 ** (-8.0 / num_heads_l15 * (i + 1)) for i in range(num_heads_l15)]
            )
            attn.register_buffer('alibi_slopes', new_slopes)

        # Resize W_q, W_k, W_v from (512, 512) to (768, 512)
        old_W_q = attn.W_q.data
        old_W_k = attn.W_k.data
        old_W_v = attn.W_v.data
        attn.W_q = nn.Parameter(torch.zeros(new_q_rows, d_model))
        attn.W_k = nn.Parameter(torch.zeros(new_q_rows, d_model))
        attn.W_v = nn.Parameter(torch.zeros(new_q_rows, d_model))
        attn.W_q.data[:d_model, :] = old_W_q  # Copy existing 512 rows
        attn.W_k.data[:d_model, :] = old_W_k
        attn.W_v.data[:d_model, :] = old_W_v

        # Resize W_o from (512, 512) to (512, 768)
        old_W_o = attn.W_o.data
        attn.W_o = nn.Parameter(torch.zeros(d_model, new_q_rows))
        attn.W_o.data[:, :d_model] = old_W_o  # Copy existing 512 cols

    def _compile_l15_attention(self, attn):
        """Compile L15 attention (memory lookup with optional LEV support)."""
        HD = self.HD
        L = 15.0
        MEM_I = 4
        BP_I = 3

        # Set ALiBi slopes to gentle recency bias
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.01)

        # Heads 0-3: LI/LC/STACK0 memory lookup
        for h in range(4):
            base = h * HD

            # Dim 0: Bias - suppress non-target Q positions
            attn.W_q.data[base, BD.CONST] = -2000.0
            attn.W_q.data[base, BD.OP_LI_RELAY] = 2000.0
            if h == 0:
                attn.W_q.data[base, BD.OP_LC_RELAY] = 2000.0
                attn.W_q.data[base, BD.CMP + 3] = 2000.0  # POP group
            else:
                attn.W_q.data[base, BD.L1H4 + BP_I] = 2000.0
                attn.W_q.data[base, BD.H1 + BP_I] = -2000.0
            attn.W_q.data[base, BD.CMP + 0] = -2000.0  # PSH suppression
            attn.W_q.data[base, BD.OP_LEV] = -1000.0
            attn.W_q.data[base, BD.MARK_PC] = -25000.0
            attn.W_q.data[base, BD.MARK_SP] = -100000.0
            attn.W_k.data[base, BD.CONST] = 10.0

            # Dim 1: Store anchor
            attn.W_q.data[base + 1, BD.OP_LI_RELAY] = 50.0
            if h == 0:
                attn.W_q.data[base + 1, BD.OP_LC_RELAY] = 50.0
                attn.W_q.data[base + 1, BD.CMP + 3] = 50.0
            else:
                attn.W_q.data[base + 1, BD.L1H4 + BP_I] = 50.0
                attn.W_q.data[base + 1, BD.H1 + BP_I] = -50.0
            attn.W_q.data[base + 1, BD.CMP + 0] = -50.0
            attn.W_k.data[base + 1, BD.MEM_STORE] = 100.0
            attn.W_k.data[base + 1, BD.CONST] = -50.0

            # Dim 2: ZFOD negative offset
            attn.W_q.data[base + 2, BD.CONST] = -96.0
            attn.W_k.data[base + 2, BD.MEM_STORE] = 50.0

            # Dim 3: Byte selection
            BS = 60.0
            byte_q_flag = [BD.MARK_AX, BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][h]
            attn.W_q.data[base + 3, byte_q_flag] = BS
            if h == 0:
                attn.W_q.data[base + 3, BD.MARK_STACK0] = BS
            MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
            if h == 0:
                attn.W_k.data[base + 3, BD.L2H0 + MEM_I] = BS
                attn.W_k.data[base + 3, BD.H1 + MEM_I] = -BS
            else:
                attn.W_k.data[base + 3, MEM_VAL_DIMS[h]] = BS

            # Dims 4-27: Binary address encoding (24 bits)
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
                            attn.W_q.data[base + addr_dim, nibble_base + k] = scale * bit_val
                            attn.W_k.data[base + addr_dim, nibble_base + k] = scale * bit_val
                        addr_dim += 1

            # Dim 28: Per-head position gate
            attn.W_q.data[base + 28, BD.CONST] = -500.0
            attn.W_q.data[base + 28, byte_q_flag] = 500.0
            if h == 0:
                attn.W_q.data[base + 28, BD.MARK_STACK0] = 500.0
            attn.W_k.data[base + 28, BD.CONST] = 5.0

            # V/O: copy byte value to OUTPUT
            for k in range(16):
                attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            for k in range(16):
                attn.W_o.data[BD.OUTPUT_LO + k, base + 32 + k] = 1.0
                attn.W_o.data[BD.OUTPUT_HI + k, base + 48 + k] = 1.0

        # LEV-specific heads (4-11): Only when L15 has 12 heads
        if attn.num_heads >= 12:
            # Heads 4-7: Read saved_bp from memory[BP]
            for h in range(4, 8):
                base = h * HD
                byte_idx = h - 4

                if byte_idx == 0:
                    # Head 4: At BP marker
                    attn.W_q.data[base, BD.CONST] = -4000.0
                    attn.W_q.data[base, BD.OP_LEV] = 2000.0
                    attn.W_q.data[base, BD.MARK_BP] = 2000.0
                    attn.W_q.data[base, BD.MARK_PC] = -25000.0
                    attn.W_q.data[base, BD.MARK_SP] = -100000.0
                    attn.W_k.data[base, BD.CONST] = 10.0

                    attn.W_q.data[base + 1, BD.CONST] = -50.0
                    attn.W_q.data[base + 1, BD.OP_LEV] = 50.0
                    attn.W_q.data[base + 1, BD.MARK_BP] = 50.0
                    attn.W_q.data[base + 1, BD.MARK_PC] = -200.0
                    attn.W_q.data[base + 1, BD.MARK_SP] = -200.0
                    attn.W_k.data[base + 1, BD.MEM_STORE] = 100.0
                    attn.W_k.data[base + 1, BD.CONST] = -50.0

                    BS = 150.0
                    attn.W_q.data[base + 3, BD.CONST] = -BS
                    attn.W_q.data[base + 3, BD.OP_LEV] = BS
                    attn.W_q.data[base + 3, BD.MARK_BP] = BS
                    attn.W_q.data[base + 3, BD.MARK_PC] = -BS * 20
                    attn.W_q.data[base + 3, BD.MARK_SP] = -BS * 20
                    attn.W_k.data[base + 3, BD.MEM_VAL_B0] = BS
                    attn.W_k.data[base + 3, BD.CONST] = -BS
                else:
                    attn.W_q.data[base, BD.CONST] = 10.0
                    attn.W_k.data[base, BD.CONST] = 10.0

                    attn.W_q.data[base + 1, BD.CONST] = 10.0
                    attn.W_k.data[base + 1, BD.MEM_STORE] = 100.0
                    attn.W_k.data[base + 1, BD.CONST] = -50.0

                    BS = 60.0
                    attn.W_q.data[base + 3, BD.CONST] = BS
                    MEM_VAL_DIMS = [None, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
                    attn.W_k.data[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

                # Dim 2: ZFOD
                attn.W_q.data[base + 2, BD.CONST] = -96.0
                attn.W_k.data[base + 2, BD.MEM_STORE] = 50.0

                # Dims 4-35: One-hot address matching (byte 0 only)
                L_addr = 50.0
                for k in range(16):
                    attn.W_q.data[base + 4 + k, BD.ADDR_B0_LO + k] = L_addr
                    attn.W_q.data[base + 4 + 16 + k, BD.ADDR_B0_HI + k] = L_addr
                    attn.W_k.data[base + 4 + k, BD.ADDR_KEY + k] = L_addr
                    attn.W_k.data[base + 4 + 16 + k, BD.ADDR_KEY + 16 + k] = L_addr

                # Dim 36: Per-head position gate (moved from 28 to avoid address collision)
                GATE_DIM = 36
                if byte_idx == 0:
                    attn.W_q.data[base + GATE_DIM, BD.CONST] = -500.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_BP] = 500.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_PC] = -50000.0
                elif byte_idx == 1:
                    attn.W_q.data[base + GATE_DIM, BD.CONST] = -500.0
                    attn.W_q.data[base + GATE_DIM, BD.BYTE_INDEX_1] = 500.0
                    attn.W_q.data[base + GATE_DIM, BD.L1H1 + BP_I] = 500.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_BP] = -1000.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_PC] = -50000.0
                elif byte_idx == 2:
                    attn.W_q.data[base + GATE_DIM, BD.CONST] = -500.0
                    attn.W_q.data[base + GATE_DIM, BD.BYTE_INDEX_2] = 500.0
                    attn.W_q.data[base + GATE_DIM, BD.H0 + BP_I] = 500.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_BP] = -1000.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_PC] = -50000.0
                elif byte_idx == 3:
                    attn.W_q.data[base + GATE_DIM, BD.CONST] = -500.0
                    attn.W_q.data[base + GATE_DIM, BD.BYTE_INDEX_3] = 500.0
                    attn.W_q.data[base + GATE_DIM, BD.H1 + BP_I] = 500.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_BP] = -1000.0
                    attn.W_q.data[base + GATE_DIM, BD.MARK_PC] = -50000.0
                attn.W_k.data[base + GATE_DIM, BD.CONST] = 5.0

                # V/O: Copy byte value to staging/OUTPUT
                for k in range(16):
                    attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                    attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
                if byte_idx == 0:
                    for k in range(16):
                        attn.W_o.data[BD.OUTPUT_LO + k, base + 32 + k] = 1.0
                        attn.W_o.data[BD.OUTPUT_HI + k, base + 48 + k] = 1.0

            # Heads 8-11: Read return_addr from memory[BP+8]
            # L9 FFN already applies +8 offset to ADDR_B0_LO at PC marker
            for h in range(8, 12):
                base = h * HD
                byte_idx = h - 8

                # Dim 0: Fire at PC marker when OP_LEV active
                attn.W_q.data[base, BD.CONST] = -4000.0
                attn.W_q.data[base, BD.OP_LEV] = 2000.0
                attn.W_q.data[base, BD.MARK_PC] = 2000.0
                attn.W_k.data[base, BD.CONST] = 10.0

                # Dim 1: Store anchor
                attn.W_q.data[base + 1, BD.CONST] = -50.0
                attn.W_q.data[base + 1, BD.OP_LEV] = 50.0
                attn.W_q.data[base + 1, BD.MARK_PC] = 50.0
                attn.W_k.data[base + 1, BD.MEM_STORE] = 100.0
                attn.W_k.data[base + 1, BD.CONST] = -50.0

                # Dim 2: ZFOD
                attn.W_q.data[base + 2, BD.CONST] = -96.0
                attn.W_k.data[base + 2, BD.MEM_STORE] = 50.0

                # Dim 3: Byte selection (requires BOTH OP_LEV AND MARK_PC)
                BS = 60.0
                attn.W_q.data[base + 3, BD.CONST] = -BS
                attn.W_q.data[base + 3, BD.OP_LEV] = BS
                attn.W_q.data[base + 3, BD.MARK_PC] = BS
                MEM_VAL_DIMS = [BD.MEM_VAL_B0, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]
                attn.W_k.data[base + 3, MEM_VAL_DIMS[byte_idx]] = BS

                # Dims 4-35: One-hot address matching (NO +8 offset - L9 FFN already shifted)
                L_addr = 50.0
                # Dims 4-19: Byte 0 lo nibble
                for k in range(16):
                    attn.W_q.data[base + 4 + k, BD.ADDR_B0_LO + k] = L_addr
                for k in range(16):
                    attn.W_k.data[base + 4 + k, BD.ADDR_KEY + k] = L_addr
                # Dims 20-35: Byte 0 hi nibble
                for k in range(16):
                    attn.W_q.data[base + 20 + k, BD.ADDR_B0_HI + k] = L_addr
                for k in range(16):
                    attn.W_k.data[base + 20 + k, BD.ADDR_KEY + 16 + k] = L_addr

                # Dim 36: Position gate (suppress at BP marker)
                GATE_DIM = 36
                attn.W_q.data[base + GATE_DIM, BD.CONST] = -500.0
                attn.W_q.data[base + GATE_DIM, BD.MARK_PC] = 500.0
                attn.W_q.data[base + GATE_DIM, BD.MARK_BP] = -1000.0
                attn.W_k.data[base + GATE_DIM, BD.CONST] = 5.0

                # Dim 37: Memory position suppression
                SUPPRESS_DIM = 37
                attn.W_k.data[base + SUPPRESS_DIM, BD.CONST] = 10000.0
                attn.W_k.data[base + SUPPRESS_DIM, BD.MEM_STORE] = -10000.0
                attn.W_q.data[base + SUPPRESS_DIM, BD.CONST] = -1000.0

                # V/O: Copy byte value to TEMP (for L16 FFN routing)
                for k in range(16):
                    attn.W_v.data[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                    attn.W_v.data[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
                if byte_idx == 0:
                    # Head 8 writes to TEMP for return_addr byte 0
                    for k in range(16):
                        attn.W_o.data[BD.TEMP + k, base + 32 + k] = 1.0
                        attn.W_o.data[BD.TEMP + 16 + k, base + 48 + k] = 1.0
                # Heads 9-11: No O projection (they write at byte positions later)

    def _compile_output_head(self, model):
        """Compile output head (lm_head) weights."""
        head = model.head

        # Zero out all weights and biases
        head.weight.data.zero_()
        head.bias.data.zero_()

        # Token constants (from vm_step.py Token class)
        REG_PC = 257
        REG_AX = 258
        REG_SP = 259
        REG_BP = 260
        MEM = 261
        STEP_END = 262
        HALT = 263
        CODE_START = 264
        CODE_END = 265
        DATA_START = 266
        DATA_END = 267
        STACK0 = 268
        USER_INPUT_START = 269
        USER_INPUT_END = 270
        TOOL_CALL = 271
        THINKING_START = 272
        THINKING_END = 273
        IO_STATE_EMIT_BYTE = 274
        IO_STATE_EMIT_THINKING = 275

        # NEXT_* flag dimensions for marker suppression
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

        # Byte tokens (0-255): nibble decoding + marker suppression
        for b in range(256):
            lo, hi = b & 0xF, (b >> 4) & 0xF
            head.weight.data[b, BD.OUTPUT_LO + lo] = 5.0
            head.weight.data[b, BD.OUTPUT_HI + hi] = 5.0
            head.bias.data[b] = -5.0
            # Suppress byte logits when a marker transition is expected
            for flag in next_flags:
                head.weight.data[b, flag] += -80.0
        head.bias.data[0] = -4.0  # slight preference for byte 0 as default

        # Transition tokens
        transition_tokens = [
            (REG_PC, BD.NEXT_PC),
            (REG_AX, BD.NEXT_AX),
            (REG_SP, BD.NEXT_SP),
            (REG_BP, BD.NEXT_BP),
            (STACK0, BD.NEXT_STACK0),
            (MEM, BD.NEXT_MEM),
            (STEP_END, BD.NEXT_SE),
            (HALT, BD.NEXT_HALT),
            (TOOL_CALL, BD.NEXT_TOOL_CALL),
            (THINKING_START, BD.NEXT_THINKING_START),
            (THINKING_END, BD.NEXT_THINKING_END),
        ]
        for tok, flag in transition_tokens:
            head.weight.data[tok, flag] = 20.0
            head.bias.data[tok] = -10.0

        # Never output context marker tokens
        SEP = 256
        for tok in [CODE_START, CODE_END, DATA_START, DATA_END, SEP,
                    USER_INPUT_START, USER_INPUT_END]:
            head.bias.data[tok] = -50.0

        # I/O state tokens (suppress by default)
        head.bias.data[IO_STATE_EMIT_BYTE] = -20.0
        head.bias.data[IO_STATE_EMIT_THINKING] = -20.0
