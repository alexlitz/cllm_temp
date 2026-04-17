"""
Function Call Weight Module.

Handles JSR, ENT, LEV, LEA opcodes.

JSR (opcode 3): Jump to subroutine
  - Push return address (PC+8) to stack
  - Jump to target address
  - SP -= 8

ENT (opcode 6): Enter function frame
  - Push BP to stack (save frame pointer)
  - BP = SP (set new frame pointer)
  - SP -= imm (allocate local variables)

LEV (opcode 8): Leave function frame
  - SP = BP (deallocate locals)
  - BP = pop (restore frame pointer)
  - PC = pop (return to caller)

LEA (opcode 0): Load effective address
  - AX = BP + imm (compute local variable address)

Layer allocation:
  - L3 head 6: BP carry to PC marker for LEV return address
  - L5 heads 5-6: ENT relay (BP→STACK0, SP→BP)
  - L6 head 7: JSR PC→STACK0 (return address)
  - L6 FFN: LEA/JSR/ENT output routing
  - L7 head 1: LEA BP operand gather
  - L9: LEV address relay (BP to memory lookup)
  - L16: LEV SP/BP routing
"""

from typing import List, Dict, Any
import torch

from .base import WeightModule, WeightConfig, DiagnosticResult, get_dimension_registry


class FunctionCallWeights(WeightModule):
    """Weight module for function call opcodes (JSR, ENT, LEV, LEA)."""

    @property
    def name(self) -> str:
        return "function_calls"

    @property
    def layers(self) -> List[int]:
        return [3, 5, 6, 7, 9, 15, 16]  # Layers used by function calls

    @property
    def dimensions(self) -> List[int]:
        BD = get_dimension_registry()
        return [
            BD.OP_JSR, BD.OP_ENT, BD.OP_LEV, BD.OP_LEA,
            BD.MARK_PC, BD.MARK_AX, BD.MARK_SP, BD.MARK_BP, BD.MARK_STACK0,
            BD.TEMP, BD.AX_CARRY_LO, BD.AX_CARRY_HI,
        ]

    def set_weights(self, model) -> None:
        """Set weights for function call opcodes."""
        BD = get_dimension_registry()
        S = self.config.swiglu_scale
        HD = self.config.head_dim

        # Set individual opcode weights
        self._set_jsr_weights(model, S, BD, HD)
        self._set_ent_weights(model, S, BD, HD)
        self._set_lev_weights(model, S, BD, HD)
        self._set_lea_weights(model, S, BD, HD)

    def _set_jsr_weights(self, model, S, BD, HD):
        """Set JSR-specific weights.

        JSR semantics:
        1. Push return address (PC+8) to stack at SP-8
        2. Set SP -= 8
        3. Set PC = target (immediate value)

        Neural implementation:
        - L5: Fetch target address from CODE section
        - L6 head 7: Copy PC OUTPUT to STACK0 (return address)
        - L6 FFN: Route target to PC, decrement SP
        - L14/L15: Write return address to memory
        """
        attn6 = model.blocks[6].attn
        L6 = 50.0  # Strong attention for precise routing

        # Head 7: PC OUTPUT → AX_CARRY at STACK0 (for memory write)
        # This copies the return address (current PC) to STACK0 output
        base = 7 * HD
        attn6.W_q[base, BD.MARK_STACK0] = L6
        attn6.W_q[base, BD.OP_JSR] = L6 / 5  # Gate on JSR opcode
        attn6.W_k[base, BD.MARK_PC] = L6 / 5  # Look at PC marker

        # Anti-leakage gate
        GATE = 33
        attn6.W_q[base + GATE, BD.MARK_STACK0] = 500.0
        attn6.W_q[base + GATE, BD.CONST] = -500.0
        attn6.W_k[base + GATE, BD.CONST] = 5.0

        # V: copy OUTPUT bytes
        for k in range(16):
            attn6.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn6.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0

        # O: write to AX_CARRY (used by L15 for memory write value)
        for k in range(16):
            attn6.W_o[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
            attn6.W_o[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0

    def _set_ent_weights(self, model, S, BD, HD):
        """Set ENT-specific weights.

        ENT semantics:
        1. Push BP to stack at SP-8 (save frame pointer)
        2. BP = SP (new frame pointer)
        3. SP -= 8 + imm (allocate locals)

        Neural implementation:
        - L5 head 5: Copy BP EMBED → TEMP at STACK0 (for memory write)
        - L5 head 6: Copy SP EMBED → TEMP at BP (for BP = SP)
        - L6 FFN: Route TEMP to BP output, compute SP -= 8 + imm
        - L14/L15: Write old BP to memory at SP-8
        """
        attn5 = model.blocks[5].attn
        L5 = 20.0

        # Head 5: BP EMBED → TEMP at STACK0 (for ENT: save old BP)
        base = 5 * HD
        attn5.W_q[base, BD.MARK_STACK0] = L5
        attn5.W_k[base, BD.MARK_BP] = L5

        # Anti-leakage gate
        GATE = 33
        attn5.W_q[base + GATE, BD.MARK_STACK0] = 500.0
        attn5.W_q[base + GATE, BD.CONST] = -500.0
        attn5.W_k[base + GATE, BD.CONST] = 5.0

        # V: copy EMBED nibbles
        for k in range(16):
            attn5.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn5.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

        # O: write to TEMP
        for k in range(16):
            attn5.W_o[BD.TEMP + k, base + 1 + k] = 1.0
            attn5.W_o[BD.TEMP + 16 + k, base + 17 + k] = 1.0

        # Head 6: SP EMBED → TEMP at BP (for ENT: BP = old SP)
        base = 6 * HD
        attn5.W_q[base, BD.MARK_BP] = L5
        attn5.W_k[base, BD.MARK_SP] = L5

        # Anti-leakage gate
        attn5.W_q[base + GATE, BD.MARK_BP] = 500.0
        attn5.W_q[base + GATE, BD.CONST] = -500.0
        attn5.W_k[base + GATE, BD.CONST] = 5.0

        # OP_ENT gate
        ENT_GATE = 34
        attn5.W_q[base + ENT_GATE, BD.OP_ENT] = 500.0
        attn5.W_q[base + ENT_GATE, BD.CONST] = -500.0
        attn5.W_k[base + ENT_GATE, BD.CONST] = 5.0

        # V: copy EMBED nibbles
        for k in range(16):
            attn5.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn5.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

        # O: write to TEMP
        for k in range(16):
            attn5.W_o[BD.TEMP + k, base + 1 + k] = 1.0
            attn5.W_o[BD.TEMP + 16 + k, base + 17 + k] = 1.0

    def _set_lev_weights(self, model, S, BD, HD):
        """Set LEV-specific weights.

        LEV semantics:
        1. SP = BP (deallocate locals)
        2. BP = *SP (pop frame pointer from stack)
        3. PC = *(SP+8) (pop return address from stack)
        4. SP += 16 (pop both values)

        Neural implementation:
        - L3 head 6: Copy BP byte 0 to OUTPUT at PC marker (for PC = return_addr)
        - L9: LEV address relay (BP → memory lookup address)
        - L14/L15: Memory lookup at BP and BP+8
        - L16: Final SP/BP routing
        """
        attn3 = model.blocks[3].attn
        L3 = 15.0

        # Head 6: BP carry to PC marker for LEV return address lookup
        base = 6 * HD
        attn3.W_q[base, BD.MARK_PC] = L3
        attn3.W_q[base, BD.OP_LEV] = L3 / 5  # Gate on LEV opcode
        attn3.W_k[base, BD.MARK_BP] = L3

        # Anti-leakage gate
        GATE = 33
        attn3.W_q[base + GATE, BD.MARK_PC] = 500.0
        attn3.W_q[base + GATE, BD.CONST] = -500.0
        attn3.W_k[base + GATE, BD.CONST] = 5.0

        # V: copy EMBED bytes for BP value
        for k in range(16):
            attn3.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn3.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

        # O: write to OUTPUT (for memory address lookup)
        for k in range(16):
            attn3.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn3.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0

        # L16 LEV routing (if model has 17 layers)
        if len(model.blocks) > 16:
            self._set_l16_lev_routing(model, S, BD, HD)

    def _set_l16_lev_routing(self, model, S, BD, HD):
        """Set L16 FFN for LEV SP/BP routing.

        After L15 memory lookup:
        - mem[BP] → new BP value
        - mem[BP+8] → new PC (return address)
        - SP = BP + 16

        L16 FFN routes these to final outputs.
        """
        ffn16 = model.blocks[16].ffn
        T = 4.0  # Opcode threshold

        # Unit allocation for LEV SP routing (SP = BP + 16)
        # When OP_LEV active at SP marker, output BP + 16
        unit = 0
        for byte_idx in range(4):
            # Add 16 to BP byte 0, carry to higher bytes
            u = unit + byte_idx
            ffn16.W_up[u, BD.MARK_SP] = S
            ffn16.W_up[u, BD.OP_LEV] = S / 5
            ffn16.bias_up[u] = -S * (T + 1)

            # Gate: compute based on byte position
            ffn16.W_gate[u, BD.MARK_SP] = S
            ffn16.W_gate[u, BD.OP_LEV] = S / 5
            ffn16.bias_gate[u] = -S * (T + 1)

            # Output: copy BP + 16 computation result
            # (Simplified - actual implementation more complex)

    def _set_lea_weights(self, model, S, BD, HD):
        """Set LEA-specific weights.

        LEA semantics:
        - AX = BP + imm (compute address of local variable)

        Neural implementation:
        - L5: Fetch immediate value
        - L7 head 1: Copy BP to AX_CARRY (for ADD operand)
        - L8/L9: ADD circuit computes BP + imm
        """
        attn7 = model.blocks[7].attn
        L7 = 15.0

        # Head 1: BP → AX_CARRY at AX marker (for LEA: AX = BP + imm)
        base = 1 * HD
        attn7.W_q[base, BD.MARK_AX] = L7
        attn7.W_q[base, BD.OP_LEA] = L7 / 5  # Gate on LEA opcode
        attn7.W_k[base, BD.MARK_BP] = L7

        # Anti-leakage gate
        GATE = 33
        attn7.W_q[base + GATE, BD.MARK_AX] = 500.0
        attn7.W_q[base + GATE, BD.CONST] = -500.0
        attn7.W_k[base + GATE, BD.CONST] = 5.0

        # V: copy EMBED nibbles for BP value
        for k in range(16):
            attn7.W_v[base + 1 + k, BD.EMBED_LO + k] = 1.0
            attn7.W_v[base + 17 + k, BD.EMBED_HI + k] = 1.0

        # O: write to AX_CARRY for ADD input
        for k in range(16):
            attn7.W_o[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
            attn7.W_o[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0

    def diagnose(
        self,
        model,
        hidden_states=None,
    ) -> DiagnosticResult:
        """Diagnose function call weight issues."""
        result = super().diagnose(model, hidden_states)
        BD = get_dimension_registry()

        # Check opcode flag dimensions are set correctly
        embed = model.embed.embed.weight

        # Check that OP_ENT/LEV/JSR/LEA dimensions exist in embedding
        for opcode, dim_name in [
            (6, "OP_ENT"),
            (8, "OP_LEV"),
            (3, "OP_JSR"),
            (0, "OP_LEA"),
        ]:
            dim = getattr(BD, dim_name)
            val = embed[opcode, dim].item()
            result.details[f"embed_{dim_name}_at_opcode_{opcode}"] = val

        # Check L5 head 5-6 weights for ENT
        attn5 = model.blocks[5].attn
        HD = self.config.head_dim

        base5 = 5 * HD
        q_stack0 = attn5.W_q[base5, BD.MARK_STACK0].item()
        k_bp = attn5.W_k[base5, BD.MARK_BP].item()
        result.details["L5_head5_Q_MARK_STACK0"] = q_stack0
        result.details["L5_head5_K_MARK_BP"] = k_bp

        if abs(q_stack0) < 1.0:
            result.warnings.append("L5 head 5 Q[MARK_STACK0] is weak")

        base6 = 6 * HD
        q_bp = attn5.W_q[base6, BD.MARK_BP].item()
        k_sp = attn5.W_k[base6, BD.MARK_SP].item()
        result.details["L5_head6_Q_MARK_BP"] = q_bp
        result.details["L5_head6_K_MARK_SP"] = k_sp

        if abs(q_bp) < 1.0:
            result.warnings.append("L5 head 6 Q[MARK_BP] is weak")

        return result
