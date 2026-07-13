"""
Purely Neural Efficient ALU Integration.

All BD ↔ GenericE format conversions are done with baked FFN weights.
NO Python loops or conditionals in the forward pass.

Format conversion approach:
- One-hot to scalar: Linear projection with weights [0, 1, 2, ..., 15]
- Scalar to one-hot: Step pair detection for each value 0-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alu.chunk_config import NIBBLE
from .alu.ops.add import build_add_layers
from .alu.ops.sub import build_sub_layers
from .alu.ops.mul import build_mul_layers
from .alu.ops.shift import build_shl_layers, build_shr_layers
from .alu.ops.div import build_div_layers
from .alu.ops.mod import build_mod_layers
from .alu.ops.bitwise import build_and_layers, build_or_layers, build_xor_layers
from .alu.ops.common import GenericE, GenericPureFFN


class BDToGEConverter(nn.Module):
    """Convert BD format (one-hot) to GenericE format (scalar) - Pure Neural.

    BD format: [seq_len, 512] with one-hot nibbles at ALU_LO, ALU_HI, AX_CARRY_LO, AX_CARRY_HI
    GE format: [8, 160] with scalar nibbles at NIB_A, NIB_B per position

    Conversion: scalar = sum_{k=0}^{15} k * one_hot[k]
    """

    def __init__(self, BD, ge: GenericE):
        super().__init__()
        self.BD = BD
        self.ge = ge

        # Build projection weights for one-hot to scalar
        # W_proj[ge_dim, bd_dim] maps BD dims to GE dims
        ge_dim = ge.DIM  # 160
        bd_dim = 512

        self.register_buffer('W_proj', torch.zeros(8, ge_dim, bd_dim))

        with torch.no_grad():
            # For positions 0-1 (lo/hi byte), map ALU and AX_CARRY
            # Position 0: ALU_LO → NIB_A, AX_CARRY_LO → NIB_B
            for k in range(16):
                self.W_proj[0, ge.NIB_A, BD.ALU_LO + k] = float(k)
                self.W_proj[0, ge.NIB_B, BD.AX_CARRY_LO + k] = float(k)

            # Position 1: ALU_HI → NIB_A, AX_CARRY_HI → NIB_B
            for k in range(16):
                self.W_proj[1, ge.NIB_A, BD.ALU_HI + k] = float(k)
                self.W_proj[1, ge.NIB_B, BD.AX_CARRY_HI + k] = float(k)

            # Copy opcode flags to all positions
            # Map BD opcode dims to GE opcode slots
            self.opcode_map = [
                (BD.OP_ADD, 25),
                (BD.OP_SUB, 26),
                (BD.OP_MUL, 27),
                (BD.OP_OR, 28),
                (BD.OP_XOR, 29),
                (BD.OP_AND, 30),
                (BD.OP_SHL, 23),
                (BD.OP_SHR, 24),
                (BD.OP_DIV, 31),
                (BD.OP_MOD, 32),
            ]
            # FIX 2026-05-06: Use 1.0 scaling for opcodes, not 0.2.
            # The shift layers (ShlPrecomputeFFN, etc.) use opcode values as multipliers
            # in their gates, so they need the full value of 1.0 when active.
            for pos in range(8):
                for bd_dim_idx, ge_opcode in self.opcode_map:
                    self.W_proj[pos, ge.OP_START + ge_opcode, bd_dim_idx] = 1.0

    def forward(self, x_bd):
        """
        Args:
            x_bd: [B, seq_len, 512] BD format (only AX marker positions used)

        Returns:
            x_ge: [B, seq_len, 8, 160] GenericE format
        """
        B, seq_len, _ = x_bd.shape
        BD = self.BD

        # FIX 2026-05-06: Clamp ALU_LO/HI and AX_CARRY_LO/HI to non-negative
        # L6 FFN clears these to -5.0, and L7 attention only overwrites active indices.
        # The negative residuals corrupt the scalar conversion (sum of k * one_hot[k]).
        x_bd_clamped = x_bd.clone()
        # BD nibble bands are semantically one-hot. Attention relays can
        # attenuate or amplify the active lane, and values near ADD/SUB modulo
        # thresholds are especially sensitive to tiny amplitude drift. Decode
        # the one-hot bands by threshold before scalar projection so e.g.
        # an active lane arriving as 0.994 still means exactly nibble 8.
        def _clean_onehot(band):
            return (torch.clamp(band, min=0, max=1) > 0.5).to(dtype=x_bd.dtype)

        x_bd_clamped[:, :, BD.ALU_LO:BD.ALU_LO + 16] = _clean_onehot(
            x_bd[:, :, BD.ALU_LO:BD.ALU_LO + 16]
        )
        x_bd_clamped[:, :, BD.ALU_HI:BD.ALU_HI + 16] = _clean_onehot(
            x_bd[:, :, BD.ALU_HI:BD.ALU_HI + 16]
        )
        x_bd_clamped[:, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO + 16] = _clean_onehot(
            x_bd[:, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO + 16]
        )
        x_bd_clamped[:, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI + 16] = _clean_onehot(
            x_bd[:, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI + 16]
        )

        # FIX 2026-05-09 (Phase 0): Replace argmax with linear projection (weighted sum).
        # argmax is not a SwiGLU FFN operation per the pure-neural policy. Linear
        # projection sum_k k * one_hot[k] IS — it's just a matrix multiply, baked
        # into FFN weights. This requires upstream to keep ALU_LO/HI as clean
        # one-hot encodings; if a leak makes one_hot[k] > 1.0, the sum overshoots.
        # We rely on the right-sizing pass + OPCODE_BLOCK_MAP defensive gates to
        # keep upstream clean enough.
        B, seq_len, _ = x_bd_clamped.shape
        x_ge = torch.zeros(B, seq_len, 8, self.ge.DIM, device=x_bd.device, dtype=x_bd.dtype)

        # Linear projection: NIB_A_value = sum_k k * one_hot[k]
        # Build the [16] coefficient vector once.
        k_coeffs = torch.arange(16, device=x_bd.device, dtype=x_bd.dtype)

        alu_lo = x_bd_clamped[:, :, BD.ALU_LO:BD.ALU_LO + 16]
        alu_hi = x_bd_clamped[:, :, BD.ALU_HI:BD.ALU_HI + 16]

        # === STACK0 campaign (2026-06-20): divmod dividend byte-0 recovery ===
        #
        # Under ``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`` operand-A byte 0
        # is delivered from ``mem[SP]`` into ALU_LO/HI by L8 head 5 at block 11.
        # For a MULTI-BYTE dividend the L10 ALU-clear (block 14) then crushes
        # ALU_LO/HI all-negative (probe: 1162/37 -> ALU_LO==-39 at the divmod
        # input block 27), so the FlattenedDivMod (block 28) reconstructs byte 0
        # as 0x00 -> wrong quotient. Single-byte dividends are unaffected
        # (ALU_LO survives at +6.0). The byte-1 (positions 2/3) path is already
        # correct here via AX_FULL/STACK0_BYTE_VAL_1 (L8 head 7).
        #
        # The L9 ``step_end_operand_relay`` head mirrors ALU_LO/HI into
        # SE_ALU_LO/HI at block 13 — BEFORE the L10 clear — and that mirror
        # SURVIVES to the divmod block (probe: SE_ALU_LO==0xA, SE_ALU_HI==0x8
        # at block 27 for 1162/37; and the same nibbles as ALU_LO for the
        # single-byte 100/7). Recover the dividend byte-0 one-hot from SE_ALU
        # by OR-ing it onto the (possibly crushed) ALU band, gated on the
        # campaign flag + the divmod opcode + the AX marker so it touches NO
        # other op / config. Flag-OFF leaves the ALU read byte-identical.
        from .unified_compiler.ops.shared import (
            no_stack0_emit_enabled,
            divmod_byte0_se_recover_enabled,
            mul_byte0_se_recover_enabled,
        )
        if (
            no_stack0_emit_enabled()
            and hasattr(BD, "SE_ALU_LO")
            and hasattr(BD, "SE_ALU_HI")
            and (
                divmod_byte0_se_recover_enabled()
                or mul_byte0_se_recover_enabled()
            )
        ):
            # OP-gated recover: divmod (DIV/MOD) and/or MUL operand-A byte 0.
            # Both opcodes hit the SAME L10 ALU-crush in the campaign config;
            # recover the byte-0 one-hot from the SE_ALU mirror (written by the
            # L9 step_end_operand_relay BEFORE the crush, survives to the ALU
            # block). Each opcode is independently flag-gated so a kill-switch
            # restores its raw read in isolation.
            recover_op = torch.zeros_like(x_bd[:, :, BD.MARK_AX], dtype=torch.bool)
            if divmod_byte0_se_recover_enabled():
                recover_op = recover_op | (x_bd[:, :, BD.OP_DIV] > 0.5) | (
                    x_bd[:, :, BD.OP_MOD] > 0.5
                )
            if mul_byte0_se_recover_enabled():
                recover_op = recover_op | (x_bd[:, :, BD.OP_MUL] > 0.5)
            byte0_recover = (
                recover_op & (x_bd[:, :, BD.MARK_AX] > 0.5)
            )[:, :, None].to(dtype=x_bd.dtype)
            se_alu_lo = _clean_onehot(
                x_bd[:, :, BD.SE_ALU_LO:BD.SE_ALU_LO + 16]
            )
            se_alu_hi = _clean_onehot(
                x_bd[:, :, BD.SE_ALU_HI:BD.SE_ALU_HI + 16]
            )
            # OR the SE mirror in only on the recover AX rows; clamp back to a
            # clean 0/1 one-hot so the k-weighted sum stays an exact nibble.
            alu_lo = torch.clamp(
                alu_lo + se_alu_lo * byte0_recover, max=1.0
            )
            alu_hi = torch.clamp(
                alu_hi + se_alu_hi * byte0_recover, max=1.0
            )

        ax_lo = x_bd_clamped[:, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO + 16]
        x_ge[:, :, 0, self.ge.NIB_A] = (alu_lo * k_coeffs).sum(dim=-1)
        x_ge[:, :, 0, self.ge.NIB_B] = (ax_lo * k_coeffs).sum(dim=-1)

        ax_hi = x_bd_clamped[:, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI + 16]
        x_ge[:, :, 1, self.ge.NIB_A] = (alu_hi * k_coeffs).sum(dim=-1)
        x_ge[:, :, 1, self.ge.NIB_B] = (ax_hi * k_coeffs).sum(dim=-1)

        # Wide binary ops need the next stack byte available to the generic
        # NIBBLE pipeline. L8 attention stages stack byte 1 into AX_FULL_* at
        # the AX marker; map that into operand-A GE positions 2/3 only while
        # a wide ALU opcode is active, so stale AX_FULL usage elsewhere stays
        # invisible to the generic converter. DIV/MOD also use the 32-bit
        # long-division pipeline, so they must receive this byte just like
        # MUL/SHL/SHR instead of silently dividing only the low byte.
        if hasattr(BD, "AX_FULL_LO") and hasattr(BD, "AX_FULL_HI"):
            divmod_op = (
                (x_bd[:, :, BD.OP_DIV] > 0.5)
                | (x_bd[:, :, BD.OP_MOD] > 0.5)
            )
            wide_op = (
                (x_bd[:, :, BD.OP_MUL] > 0.5)
                | (x_bd[:, :, BD.OP_SHL] > 0.5)
                | (x_bd[:, :, BD.OP_SHR] > 0.5)
                | divmod_op
            ).to(dtype=x_bd.dtype)
            ax_marker = (x_bd[:, :, BD.MARK_AX] > 0.5).to(dtype=x_bd.dtype)
            wide_marker = wide_op * ax_marker
            ax_full_lo = _clean_onehot(
                x_bd[:, :, BD.AX_FULL_LO:BD.AX_FULL_LO + 16],
            )
            ax_full_hi = _clean_onehot(
                x_bd[:, :, BD.AX_FULL_HI:BD.AX_FULL_HI + 16],
            )

            # Campaign MUL operand-A byte-1 flood guard (2026-06-21). In the
            # 30-token campaign config the L11 wide_mul floods MUL_RESULT_HI ->
            # the L13 relay stages a near-UNIFORM flood across ALL 16 AX_FULL
            # cells (spec_k=0 probe: every AX_FULL cell ~1e8). The
            # ``_clean_onehot`` clamp then makes EVERY cell hot, so operand-A
            # byte 1 reconstructs as sum(0..15)=0x78 instead of 0x00 -> the
            # FlattenedALUMul schoolbook multiplies a bogus 2-byte operand A and
            # the product byte-1 high nibble decodes to 0xF garbage. A genuine
            # AX_FULL byte stage is a 1-2 cell one-hot (sum <= ~2); a flood lights
            # >2 cells. On OP_MUL+MARK_AX rows with the recover flag, treat a
            # flooded AX_FULL as operand-A byte 1 == 0 (every mul-cluster operand
            # is single-byte, so byte 1 IS 0). Gated on no_stack0_emit + the mul
            # recover flag -> golden byte-identical.
            from .unified_compiler.ops.shared import (
                no_stack0_emit_enabled as _no_s0,
                mul_byte0_se_recover_enabled as _mul_rec,
            )
            if _no_s0() and _mul_rec():
                ax_full_cells = ax_full_lo.sum(dim=-1) + ax_full_hi.sum(dim=-1)
                mul_flood = (
                    (x_bd[:, :, BD.OP_MUL] > 0.5)
                    & (x_bd[:, :, BD.MARK_AX] > 0.5)
                    & (ax_full_cells > 2.5)
                )[:, :, None].to(dtype=x_bd.dtype)
                keep = 1.0 - mul_flood
                ax_full_lo = ax_full_lo * keep
                ax_full_hi = ax_full_hi * keep

            ax_full_present = (
                (ax_full_lo.sum(dim=-1) + ax_full_hi.sum(dim=-1)) > 0.5
            ).to(dtype=x_bd.dtype)

            # DIV/MOD run at the AX marker. When AX_FULL was not explicitly
            # staged there, recover operand-A byte 1 from the latest prior
            # STACK0 byte-1 row in the autoregressive prefix.
            prev_stack_lo = torch.zeros_like(ax_full_lo)
            prev_stack_hi = torch.zeros_like(ax_full_hi)
            if hasattr(BD, "STACK0_BYTE1"):
                pos_idx = torch.arange(seq_len, device=x_bd.device).view(1, seq_len)
                stack1 = x_bd[:, :, BD.STACK0_BYTE1] > 0.5
                scores = torch.where(stack1, pos_idx, torch.full_like(pos_idx, -1))
                if torch.onnx.is_in_onnx_export():
                    # ONNX has no ``cummax`` symbolic. Materialize the
                    # running argmax via a causal mask + ArgMax/ReduceMax.
                    # O(S^2) but purely vanilla (Where + ArgMax + ReduceMax).
                    S = seq_len
                    rng = torch.arange(S, device=x_bd.device)
                    causal = (rng.view(1, S) <= rng.view(S, 1))
                    NEG = torch.full((), -1, dtype=scores.dtype, device=x_bd.device)
                    masked = torch.where(
                        causal.unsqueeze(0),
                        scores.unsqueeze(1).expand(-1, S, S),
                        NEG.expand_as(scores.unsqueeze(1).expand(-1, S, S)),
                    )
                    latest_idx = masked.argmax(dim=-1)
                    latest_score = masked.max(dim=-1).values
                else:
                    latest_score, latest_idx = torch.cummax(scores, dim=1)
                gather_idx = latest_idx[:, :, None].expand(-1, -1, 16)
                # Source of operand-A byte 1 at the picked STACK0_BYTE1 row.
                #
                # HEAD default reads CLEAN_EMBED_LO/HI — but at the PSH-frame
                # STACK0_BYTE1 row that band is 0x00; the pushed value's high
                # byte is deposited by ``layer10_psh_ax_broadcast`` into the
                # DESIGNATED carrier STACK0_BYTE_VAL_1_LO/HI at that same row
                # (verified spec_k=0: 1162/37 -> STACK0_BYTE_VAL_1 = 0x04,
                # CLEAN_EMBED = 0x00). Reading CLEAN_EMBED is why multi-byte
                # dividends truncated to low_byte(dividend) / divisor.
                #
                # When C4_DIV_MULTIBYTE is on, gather the high byte from
                # STACK0_BYTE_VAL_1 instead. Flag-gated so flag-off is
                # byte-identical to HEAD; requires the divmod compute to run
                # AFTER the L11 broadcast that populates this band (handled by
                # the divmod-install layer_idx=11 reroute in the same flag).
                from .unified_compiler.ops.shared import div_multibyte_enabled
                use_stack0_val = (
                    div_multibyte_enabled()
                    and hasattr(BD, "STACK0_BYTE_VAL_1_LO")
                    and hasattr(BD, "STACK0_BYTE_VAL_1_HI")
                )
                if use_stack0_val:
                    src_lo_base = BD.STACK0_BYTE_VAL_1_LO
                    src_hi_base = BD.STACK0_BYTE_VAL_1_HI
                else:
                    src_lo_base = BD.CLEAN_EMBED_LO
                    src_hi_base = BD.CLEAN_EMBED_HI
                clean_lo = _clean_onehot(
                    x_bd[:, :, src_lo_base:src_lo_base + 16]
                )
                clean_hi = _clean_onehot(
                    x_bd[:, :, src_hi_base:src_hi_base + 16]
                )
                valid = (latest_score >= 0)[:, :, None].to(dtype=x_bd.dtype)
                prev_stack_lo = torch.gather(clean_lo, 1, gather_idx) * valid
                prev_stack_hi = torch.gather(clean_hi, 1, gather_idx) * valid
            divmod_fallback = (
                divmod_op.to(dtype=x_bd.dtype)
                * ax_marker
                * (1.0 - ax_full_present)
            )
            ax_full_lo_scalar = (ax_full_lo * k_coeffs).sum(dim=-1)
            ax_full_hi_scalar = (ax_full_hi * k_coeffs).sum(dim=-1)
            prev_stack_lo_scalar = (prev_stack_lo * k_coeffs).sum(dim=-1)
            prev_stack_hi_scalar = (prev_stack_hi * k_coeffs).sum(dim=-1)
            x_ge[:, :, 2, self.ge.NIB_A] = (
                ax_full_lo_scalar * wide_marker
                + prev_stack_lo_scalar * divmod_fallback
            )
            x_ge[:, :, 3, self.ge.NIB_A] = (
                ax_full_hi_scalar * wide_marker
                + prev_stack_hi_scalar * divmod_fallback
            )

        # Copy opcode flags to all positions.
        # FIX 2026-05-06: Normalize opcode values to 0/1 by thresholding at 0.5.
        # The transformer writes opcode flags with various magnitudes (e.g., 5.0),
        # but the shift layers expect exactly 1.0 when active. Threshold to normalize.
        for pos in range(8):
            for bd_dim_idx, ge_opcode in self.opcode_map:
                x_ge[:, :, pos, self.ge.OP_START + ge_opcode] = (x_bd[:, :, bd_dim_idx] > 0.5).float()

        return x_ge


class GEToBDConverter(nn.Module):
    """Convert GenericE format (scalar) to BD format (one-hot) - Pure Neural.

    Uses step pairs to detect scalar values and output one-hot encoding.
    scalar_to_onehot[k] = step(x >= k - 0.5) - step(x >= k + 0.5)
    """

    def __init__(self, BD, ge: GenericE, S=100.0):
        super().__init__()
        self.BD = BD
        self.ge = ge
        self.S = S

        # Build step-pair FFN for scalar to one-hot conversion
        # For each of 16 possible values, we need 2 hidden units (step pair)
        # Output positions: OUTPUT_LO (16 dims) and OUTPUT_HI (16 dims)
        hidden_dim = 16 * 2 * 2  # 16 values × 2 positions (lo/hi) × 2 units per step pair

        self.ffn = GenericPureFFN(ge.DIM, hidden_dim=hidden_dim, dtype=torch.float32)

        with torch.no_grad():
            W_up = self.ffn.W_up
            b_up = self.ffn.b_up
            W_gate = self.ffn.W_gate
            W_down = self.ffn.W_down

            h = 0

            # For each output position (0=lo, 1=hi)
            for out_pos in range(2):
                # For each possible value k = 0..15
                for k in range(16):
                    # Step pair: step(result >= k - 0.5) - step(result >= k + 0.5)
                    # Unit A: silu(S*(result - k + 0.5)) → +1/S
                    W_up[h, ge.RESULT] = S
                    b_up[h] = -S * (k - 0.5)
                    W_gate[h, ge.RESULT] = 0.0  # No gating, always active
                    # But we need to gate on opcode being active...
                    # Actually for simplicity, always compute and let BD masking handle it
                    h += 1

                    # Unit B: silu(S*(result - k - 0.5)) → -1/S
                    W_up[h, ge.RESULT] = S
                    b_up[h] = -S * (k + 0.5)
                    h += 1

        # Store output mapping info
        self.out_pos_lo = 0
        self.out_pos_hi = 1

    def forward(self, x_ge, x_bd, opcode_mask=None, emit_carry=True,
                output_amplitude=2.0):
        """
        Args:
            x_ge: [B, seq_len, 8, 160] GenericE format with RESULT filled
            x_bd: [B, seq_len, 512] BD format to update
            opcode_mask: [B, seq_len] Optional mask indicating where opcodes are active.
                        If None, writes OUTPUT unconditionally (backward compat).
                        If provided, only writes OUTPUT where mask > 0.5.
            emit_carry: Whether to write ADD/SUB inter-byte CARRY flags. Only
                        add/sub ALU stages should do this; later bitwise/mul/
                        div/shift stages can see stale OP_ADD/OP_SUB relay
                        dims and must not rewrite carry state.

        Returns:
            x_bd_out: [B, seq_len, 512] with OUTPUT_LO/HI updated
        """
        B, seq_len, num_pos, ge_dim = x_ge.shape
        BD = self.BD
        S = self.S

        x_bd_out = x_bd.clone()

        # Extract result nibbles from positions 0 and 1
        result_lo = x_ge[:, :, 0, self.ge.RESULT]  # [B, seq_len]
        result_hi = x_ge[:, :, 1, self.ge.RESULT]  # [B, seq_len]

        # Convert to one-hot using vectorized step pairs with sigmoid approximation
        # For each k in 0..15: one_hot[k] = sigmoid(S*(result - k + 0.5)) - sigmoid(S*(result - k - 0.5))
        # This detects when result is in [k-0.5, k+0.5), i.e., rounds to k

        # Create k values tensor: [16]
        k_vals = torch.arange(16, device=x_ge.device, dtype=x_ge.dtype)

        # Broadcast: result_lo is [B, seq_len], k_vals is [16]
        # result_lo[:, :, None] - k_vals[None, None, :] gives [B, seq_len, 16]
        diff_lo = result_lo[:, :, None] - k_vals[None, None, :]  # [B, seq_len, 16]
        diff_hi = result_hi[:, :, None] - k_vals[None, None, :]  # [B, seq_len, 16]

        # Step pair detection: sigmoid(S*(diff + 0.5)) - sigmoid(S*(diff - 0.5))
        indicator_lo = torch.sigmoid(S * (diff_lo + 0.5)) - torch.sigmoid(S * (diff_lo - 0.5))  # [B, seq_len, 16]
        indicator_hi = torch.sigmoid(S * (diff_hi + 0.5)) - torch.sigmoid(S * (diff_hi - 0.5))  # [B, seq_len, 16]

        # Apply opcode mask if provided (only write OUTPUT where opcodes are active)
        if opcode_mask is not None:
            mask_expanded = opcode_mask[:, :, None]
            indicator_lo = indicator_lo * mask_expanded
            indicator_hi = indicator_hi * mask_expanded

        # ``output_amplitude`` (default 2.0 = byte-identical to HEAD) lets the
        # add/sub stage write the byte-0 result one-hot at a DOMINANT magnitude
        # so it out-votes the downstream block-11 / logical-L9 ALU_LO->OUTPUT_LO
        # operand leak (see ``shared.addsub_output_boost_enabled``). Every other
        # GEToBD caller (divmod / shift / mul GE writeback) keeps the 2.0
        # default, so this is scoped strictly to the add/sub stage that passes a
        # boosted value.
        x_bd_out[:, :, BD.OUTPUT_LO:BD.OUTPUT_LO + 16] += indicator_lo * output_amplitude
        x_bd_out[:, :, BD.OUTPUT_HI:BD.OUTPUT_HI + 16] += indicator_hi * output_amplitude

        # Stage result byte 1 at AX markers for the autoregressive byte
        # relay. The marker itself predicts byte 0; the following AX byte
        # token predicts byte 1 by attending back to these staged nibbles.
        if hasattr(BD, "AX_FULL_LO") and hasattr(BD, "AX_FULL_HI"):
            result_b1_lo = x_ge[:, :, 2, self.ge.RESULT]
            result_b1_hi = x_ge[:, :, 3, self.ge.RESULT]
            diff_b1_lo = result_b1_lo[:, :, None] - k_vals[None, None, :]
            diff_b1_hi = result_b1_hi[:, :, None] - k_vals[None, None, :]
            indicator_b1_lo = (
                torch.sigmoid(S * (diff_b1_lo + 0.5))
                - torch.sigmoid(S * (diff_b1_lo - 0.5))
            )
            indicator_b1_hi = (
                torch.sigmoid(S * (diff_b1_hi + 0.5))
                - torch.sigmoid(S * (diff_b1_hi - 0.5))
            )
            # width=2 MUL byte-1 re-stage suppression (2026-06-15). When the
            # width=2 (8-bit x 8-bit -> 16-bit) MUL path is active, the
            # product's byte 1 is computed by the L11 wide_mul into the
            # dedicated MUL_RESULT_HI band and staged into AX_FULL by the L13
            # ``layer13_mul_result_hi_relay`` head. This FlattenedALUMul
            # composite re-fires on the MUL MARK_AX row (a second instance at
            # logical L15 / physical block 26 after _expand_wrapper_blocks --
            # see docs/L17_TAIL_MUL_DOUBLE_FIRE.md) and its OWN GE high-byte
            # result DISAGREES with the wide_mul band (e.g. 97*94: wide_mul
            # byte1=0x23, this composite re-stages 0xFD), CLOBBERING the
            # correct AX_FULL staging so the byte-1 emit relay copies garbage.
            # Exclude OP_MUL from the AX_FULL byte-1 re-stage when width=2 is
            # on so the L13 relay's correct MUL_RESULT_HI staging survives to
            # the byte-1 emit. Verified spec_k=0 / BUILT dims: AX_FULL at the
            # MUL row stays 0x23 across blocks 15..30. With C4_MUL_WIDTH2=0
            # (flag off) OP_MUL is kept in the mask -> byte-identical to HEAD
            # (no MUL_RESULT_HI band exists in that build).
            from .unified_compiler.ops.shared import (
                mul_width2_enabled,
                no_stack0_emit_enabled,
                mul_byte0_se_recover_enabled,
            )
            # Campaign MUL byte-1 recovery (2026-06-21). In the 30-token
            # campaign config the L11 wide_mul reads the L10-crushed ALU and
            # FLOODS MUL_RESULT_HI to garbage (~2e8 at cell 0), which the L13
            # ``layer13_mul_result_hi_relay`` then stages into AX_FULL as
            # garbage -> the byte-1 emit truncates the product (e.g. 93*34 ->
            # 0x5A, missing byte 1 0x0C). With the SE-recover flag on, THIS
            # FlattenedALUMul composite now multiplies the REAL operand A (the
            # byte-0 SE recovery above), so ITS OWN GE high-byte result is the
            # CORRECT product byte 1. Re-INCLUDE OP_MUL in the AX_FULL byte-1
            # restage so this composite OVERWRITES the flooded MUL_RESULT_HI
            # relay (the restage is a ``*(1-mask)+new*mask`` replace, so it both
            # clears the flood AND writes the clean byte 1). Golden (35-token)
            # path is unaffected: the flag is gated on no_stack0_emit, so the
            # documented golden re-stage suppression (where MUL_RESULT_HI is the
            # CORRECT source and this composite would clobber it with 0xFD)
            # still holds.
            _campaign_mul_b1 = (
                no_stack0_emit_enabled() and mul_byte0_se_recover_enabled()
            )
            _suppress_mul_b1_restage = (
                mul_width2_enabled() and not _campaign_mul_b1
            )
            wide_op = (
                ((x_bd[:, :, BD.OP_MUL] > 0.5) if not _suppress_mul_b1_restage
                 else torch.zeros_like(x_bd[:, :, BD.OP_MUL], dtype=torch.bool))
                | (x_bd[:, :, BD.OP_SHL] > 0.5)
                | (x_bd[:, :, BD.OP_SHR] > 0.5)
                | (x_bd[:, :, BD.OP_DIV] > 0.5)
                | (x_bd[:, :, BD.OP_MOD] > 0.5)
            ).to(dtype=x_bd.dtype)
            ax_marker = (x_bd[:, :, BD.MARK_AX] > 0.5).to(dtype=x_bd.dtype)
            wide_mask = wide_op * ax_marker
            if opcode_mask is not None:
                wide_mask = wide_mask * opcode_mask
            wide_mask_expanded = wide_mask[:, :, None]
            lo_slice = slice(BD.AX_FULL_LO, BD.AX_FULL_LO + 16)
            hi_slice = slice(BD.AX_FULL_HI, BD.AX_FULL_HI + 16)
            x_bd_out[:, :, lo_slice] = (
                x_bd_out[:, :, lo_slice] * (1.0 - wide_mask_expanded)
                + indicator_b1_lo * wide_mask_expanded * 2.0
            )
            x_bd_out[:, :, hi_slice] = (
                x_bd_out[:, :, hi_slice] * (1.0 - wide_mask_expanded)
                + indicator_b1_hi * wide_mask_expanded * 2.0
            )

        # FIX 2026-05-06: Set carry/borrow flags for multi-byte propagation.
        # CarryPropagationPostOp expects:
        #   CARRY[1] for ADD overflow (sum >= 256)
        #   CARRY[2] for SUB borrow (a < b)
        # Only set these at AX marker position (where byte 0 is computed).
        operand_a_lo = x_ge[:, :, 0, self.ge.NIB_A]  # [B, seq_len]
        operand_a_hi = x_ge[:, :, 1, self.ge.NIB_A]  # [B, seq_len]
        operand_b_lo = x_ge[:, :, 0, self.ge.NIB_B]  # [B, seq_len]
        operand_b_hi = x_ge[:, :, 1, self.ge.NIB_B]  # [B, seq_len]

        # Reconstruct byte values: byte = lo + hi * 16
        operand_a = operand_a_lo + operand_a_hi * 16.0  # [B, seq_len], 0-255
        operand_b = operand_b_lo + operand_b_hi * 16.0  # [B, seq_len], 0-255

        # ADD carry: sum >= 256
        sum_ab = operand_a + operand_b
        add_carry = torch.sigmoid(S * (sum_ab - 255.5))

        # SUB borrow: a < b (i.e., a - b < 0)
        sub_borrow = torch.sigmoid(S * (operand_b - operand_a - 0.5))

        # Gate on MARK_AX (only at AX marker position)
        mark_ax = x_bd[:, :, BD.MARK_AX]
        ax_mask = (mark_ax > 0.5).float()

        # FIX 2026-05-08: Gate CARRY flags by their respective opcodes.
        # Previously, both CARRY[1] and CARRY[2] were set at all AX markers.
        # This caused SUB borrow to be set during ADD operations, which triggered
        # CarryPropagationPostOp's SUB units to fire spuriously.
        op_add = x_bd[:, :, BD.OP_ADD]
        op_sub = x_bd[:, :, BD.OP_SUB]
        add_opcode_mask = (op_add > 0.5).float()
        sub_opcode_mask = (op_sub > 0.5).float()

        if emit_carry:
            # Write CARRY[1] for ADD only, CARRY[2] for SUB only.
            x_bd_out[:, :, BD.CARRY + 1] += add_carry * ax_mask * add_opcode_mask * 2.0
            x_bd_out[:, :, BD.CARRY + 2] += sub_borrow * ax_mask * sub_opcode_mask * 2.0

        return x_bd_out


class PureNeuralALU(nn.Module):
    """Purely neural ALU that wraps efficient ops with neural format conversion.

    All operations are performed using baked FFN weights - no Python loops
    or conditionals in forward pass.
    """

    def __init__(self, S, BD, operations='add_sub'):
        """
        Args:
            S: SwiGLU scale (100.0)
            BD: _SetDim class with dimension constants
            operations: Which operations to include:
                'add_sub' - ADD and SUB
                'bitwise' - AND, OR, XOR
                'mul' - MUL
                'shift' - SHL, SHR
        """
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.operations = operations

        # Format converters
        self.bd_to_ge = BDToGEConverter(BD, self.ge)
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

        # Build operation-specific layers
        if operations == 'add_sub':
            self.add_layers = nn.ModuleList(build_add_layers(NIBBLE, opcode=25))
            self.sub_layers = nn.ModuleList(build_sub_layers(NIBBLE, opcode=26))
        elif operations == 'bitwise':
            self.and_layers = nn.ModuleList(build_and_layers(NIBBLE, opcode=30))
            self.or_layers = nn.ModuleList(build_or_layers(NIBBLE, opcode=28))
            self.xor_layers = nn.ModuleList(build_xor_layers(NIBBLE, opcode=29))
        elif operations == 'mul':
            self.mul_layers = nn.ModuleList(build_mul_layers(NIBBLE, opcode=27))
        elif operations == 'shift':
            self.shl_layers = nn.ModuleList(build_shl_layers(NIBBLE, opcode=23))
            self.shr_layers = nn.ModuleList(build_shr_layers(NIBBLE, opcode=24))
        elif operations == 'div_mod':
            self.div_layers = nn.ModuleList(build_div_layers(NIBBLE, opcode=31))
            self.mod_layers = nn.ModuleList(build_mod_layers(NIBBLE, opcode=32))

    def forward(self, x_bd):
        """
        Process ALU operations in BD format, fully neural.

        Args:
            x_bd: [B, seq_len, 512] BD format

        Returns:
            [B, seq_len, 512] with ALU results
        """
        B, seq_len, _ = x_bd.shape
        BD = self.BD

        # Convert BD → GE format
        x_ge = self.bd_to_ge(x_bd)  # [B, seq_len, 8, 160]

        # Flatten for efficient layer processing
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)  # [B*seq_len, 8, 160]

        x_ge_out = x_ge_flat.clone()
        opcode_mask_flat = torch.zeros(B * seq_len, device=x_bd.device, dtype=x_bd.dtype)

        if self.operations == 'add_sub':
            x_add = x_ge_flat.clone()
            for layer in self.add_layers:
                x_add = layer(x_add)

            x_sub = x_ge_flat.clone()
            for layer in self.sub_layers:
                x_sub = layer(x_sub)

            # FIX 2026-05-06: Opcode values are 0.2 (not 1.0) due to BDToGEConverter scaling.
            # Normalize to 0/1 before using as multipliers to avoid corrupting results.
            op_add = (x_ge_flat[:, 0, self.ge.OP_START + 25] > 0.1).float()
            op_sub = (x_ge_flat[:, 0, self.ge.OP_START + 26] > 0.1).float()

            op_total = op_add + op_sub

            x_ge_out[:, :, self.ge.RESULT] = (
                x_add[:, :, self.ge.RESULT] * op_add[:, None] +
                x_sub[:, :, self.ge.RESULT] * op_sub[:, None]
            )

            opcode_mask_flat = op_total

        elif self.operations == 'bitwise':
            x_and = x_ge_flat.clone()
            for layer in self.and_layers:
                x_and = layer(x_and)

            x_or = x_ge_flat.clone()
            for layer in self.or_layers:
                x_or = layer(x_or)

            x_xor = x_ge_flat.clone()
            for layer in self.xor_layers:
                x_xor = layer(x_xor)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_and = (x_ge_flat[:, 0, self.ge.OP_START + 30] > 0.1).float()
            op_or = (x_ge_flat[:, 0, self.ge.OP_START + 28] > 0.1).float()
            op_xor = (x_ge_flat[:, 0, self.ge.OP_START + 29] > 0.1).float()

            op_total = op_and + op_or + op_xor

            x_ge_out[:, :, self.ge.RESULT] = (
                x_and[:, :, self.ge.RESULT] * op_and[:, None] +
                x_or[:, :, self.ge.RESULT] * op_or[:, None] +
                x_xor[:, :, self.ge.RESULT] * op_xor[:, None]
            )

            opcode_mask_flat = op_total

        elif self.operations == 'mul':
            x_mul = x_ge_flat.clone()
            for layer in self.mul_layers:
                x_mul = layer(x_mul)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_mul = (x_ge_flat[:, 0, self.ge.OP_START + 27] > 0.1).float()

            x_ge_out[:, :, self.ge.RESULT] = (
                x_mul[:, :, self.ge.RESULT] * op_mul[:, None]
            )

            opcode_mask_flat = op_mul

        elif self.operations == 'shift':
            x_shl = x_ge_flat.clone()
            for layer in self.shl_layers:
                x_shl = layer(x_shl)

            x_shr = x_ge_flat.clone()
            for layer in self.shr_layers:
                x_shr = layer(x_shr)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_shl = (x_ge_flat[:, 0, self.ge.OP_START + 23] > 0.1).float()
            op_shr = (x_ge_flat[:, 0, self.ge.OP_START + 24] > 0.1).float()

            op_total = op_shl + op_shr

            x_ge_out[:, :, self.ge.RESULT] = (
                x_shl[:, :, self.ge.RESULT] * op_shl[:, None] +
                x_shr[:, :, self.ge.RESULT] * op_shr[:, None]
            )

            opcode_mask_flat = op_total

        elif self.operations == 'div_mod':
            x_div = x_ge_flat.clone()
            for layer in self.div_layers:
                x_div = layer(x_div)

            x_mod = x_ge_flat.clone()
            for layer in self.mod_layers:
                x_mod = layer(x_mod)

            # FIX 2026-05-06: Normalize opcode values to 0/1.
            op_div = (x_ge_flat[:, 0, self.ge.OP_START + 31] > 0.1).float()
            op_mod = (x_ge_flat[:, 0, self.ge.OP_START + 32] > 0.1).float()

            op_total = op_div + op_mod

            x_ge_out[:, :, self.ge.RESULT] = (
                x_div[:, :, self.ge.RESULT] * op_div[:, None] +
                x_mod[:, :, self.ge.RESULT] * op_mod[:, None]
            )

            opcode_mask_flat = op_total

        # Reshape back
        x_ge_out = x_ge_out.view(B, seq_len, 8, self.ge.DIM)

        opcode_mask = opcode_mask_flat.view(B, seq_len)

        # Only write OUTPUT at AX marker positions (MARK_AX > 0.5).
        # Without this, the ALU writes result "0" at byte positions where
        # operands are zero, corrupting the passthrough from L10 head 1.
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        x_bd_out = self.ge_to_bd(
            x_ge_out,
            x_bd,
            opcode_mask=opcode_mask,
            emit_carry=(self.operations == 'add_sub'),
        )

        return x_bd_out

    # Stub methods for compatibility with vm_step.py
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


# Operation-named ALU classes (no layer assumptions — compiler decides placement).
# The operation name is what's intrinsic; layer placement is a compiler concern.
class ALUAddSub(PureNeuralALU):
    """Neural ADD/SUB."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='add_sub')


# ``ALUAndOrXor`` (= ``PureNeuralALU(operations='bitwise')``) was deleted
# in the V8 lookup-mode wave (2026-06-04). Production install for L10
# bitwise (AND/OR/XOR) now uses the rule-derived ``PureFFN`` baked from
# ``wide_alu_dsl.bitwise_rules`` via
# ``ops/alu_ops.py:make_lookup_mode_l10_bitwise_rules_op``. The rule
# install is byte-identical at the decoded OUTPUT byte (verified by
# ``tests/test_wide_alu_dsl.py::test_bitwise_rules_byte_identity_*`` and
# ``test_lookup_mode_l10_postop_factory_byte_identity``). See
# ``docs/V8_DELETE_AUDIT_2026_06_04.md`` for the migration audit and
# ``docs/LOOKUP_MODE_RULE_DERIVATION_2026_06_04.md`` for follow-up notes.


class _MulPipelineState:
    """Mutable state passed between Sequential stages of ``FlattenedALUMul``.

    Each stage reads/writes named tensor fields and returns ``self`` so it
    can be chained inside ``nn.Sequential``. Using an object (rather than a
    bare tensor tuple) keeps each stage's I/O contract uniform — every stage
    in the pipeline has the signature ``forward(state) -> state`` — which is
    exactly what ``nn.Sequential`` requires.

    Fields populated by stage:
      - ``BDToGEStage``        sets ``x_bd_in``, ``x_ge_flat``, ``x_mul``
      - mul FFN stages         update ``x_mul``
      - ``MulCombineStage``    sets ``x_ge_out``, ``opcode_mask``
      - ``GEToBDStage``        sets ``x_bd_out``
    """

    __slots__ = (
        'x_bd_in', 'x_ge_flat', 'x_mul',
        'x_ge_out', 'opcode_mask', 'x_bd_out',
        'output_clear_mask', 'multibyte_boost_mask',
    )

    def __init__(self):
        self.x_bd_in = None
        self.x_ge_flat = None
        self.x_mul = None
        self.x_ge_out = None
        self.opcode_mask = None
        self.x_bd_out = None
        # Campaign MUL OUTPUT-flood cap: [B, seq_len] mask of rows whose
        # OUTPUT_LO/HI band must be CLEARED before the product write (set by
        # _MulCombineStage when the L11 wide_mul flood is detected, applied by
        # _GEToBDStage). None outside the campaign config -> no clear.
        self.output_clear_mask = None
        # Campaign NARROWED MUL MULTI-byte L19 byte-0 boost: [B, seq_len] mask
        # of the LITERAL-mul MULTI-byte (byte 1 != 0, var-frame carry absent)
        # MUL+MARK_AX rows whose CLEAN byte-0 OUTPUT_LO/HI band must be SCALED UP
        # (NOT cleared) so it survives the block-34 / logical-L19 byte-0
        # overwrite (set by _MulCombineStage, applied by _GEToBDStage). None
        # outside the campaign config -> no boost.
        self.multibyte_boost_mask = None


class _BDToGEStage(nn.Module):
    """Pipeline stage 0 (phase=11.0): BD → GE format conversion.

    Wraps ``BDToGEConverter`` with the uniform ``forward(state) -> state``
    contract used by every stage in ``FlattenedALUMul.pipeline``. Stashes
    ``x_bd_in`` (for later AX masking + as the base for ``GEToBDStage``) and
    initialises ``x_mul`` (the rolling MUL workspace) and ``x_ge_flat`` (the
    snapshot used for opcode/AX gating after the 7 mul layers run).
    """

    def __init__(self, BD, ge: GenericE):
        super().__init__()
        self.BD = BD
        self.ge = ge
        self.bd_to_ge = BDToGEConverter(BD, ge)

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        x_bd = state.x_bd_in
        B, seq_len, _ = x_bd.shape
        x_ge = self.bd_to_ge(x_bd)  # [B, seq_len, 8, 160]
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)
        state.x_ge_flat = x_ge_flat
        state.x_mul = x_ge_flat.clone()
        return state


class _MulFFNStage(nn.Module):
    """Pipeline stage wrapper around one mul-pipeline FFN.

    Holds a single mul sub-FFN (e.g. ``SchoolbookFFN``, ``CarryPassFFN``,
    ``MulGenPropFFN``, ...) and applies it to ``state.x_mul``. The wrapped
    FFN itself remains a vanilla ``nn.Module`` with its own ``W_up`` /
    ``W_gate`` / ``W_down`` parameters — this class is just the adapter
    that lets it slot into the uniform Sequential pipeline contract.
    """

    def __init__(self, sub_ffn: nn.Module):
        super().__init__()
        self.sub_ffn = sub_ffn

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        state.x_mul = self.sub_ffn(state.x_mul)
        return state


class _MulCombineStage(nn.Module):
    """Pipeline stage that merges the mul workspace into the GE output.

    Computes the opcode-gated MUL result, restricts OUTPUT writes to AX
    marker positions, and reshapes back to ``[B, seq_len, 8, DIM]``. Owns
    no parameters — it is pure tensor algebra (broadcasted multiplies,
    reshape, threshold mask) and exists as its own ``nn.Module`` so the
    Sequential pipeline has a single, statically-defined chain of modules
    rather than ad-hoc Python in ``forward``.
    """

    def __init__(self, BD, ge: GenericE):
        super().__init__()
        self.BD = BD
        self.ge = ge

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        x_bd = state.x_bd_in
        x_ge_flat = state.x_ge_flat
        x_mul = state.x_mul
        BD = self.BD
        ge = self.ge

        x_ge_out = x_ge_flat.clone()

        # FIX 2026-05-06: Normalize opcode values to 0/1.
        op_mul = (x_ge_flat[:, 0, ge.OP_START + 27] > 0.1).float()

        x_ge_out[:, :, ge.RESULT] = x_mul[:, :, ge.RESULT] * op_mul[:, None]

        B, seq_len, _ = x_bd.shape
        x_ge_out = x_ge_out.view(B, seq_len, 8, ge.DIM)
        opcode_mask = op_mul.view(B, seq_len)

        # Only write OUTPUT at AX marker positions (MARK_AX > 0.5).
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        # FIX 2026-06-03 (L17 tail MUL double-fire defensive guard):
        # Single-fire guard — only fire when residual indicates this is the
        # first MUL composite to write at this AX-marker row this step.
        #
        # When ``_expand_wrapper_blocks`` (Phase 10.B, ``vm_step.py:2578``)
        # split L11/L12 ALU post-op attaches into adjacent wrapper blocks
        # (blocks 25 + 27 historically), the second composite re-fired the
        # same ``OUTPUT_HI/LO += 2.0`` writes on the same AX-marker row,
        # doubling the signal and clobbering ``OUT_LO[expected]`` by ±16.0
        # at L17 (see ``docs/L17_TAIL_MUL_DOUBLE_FIRE.md``). The root-cause
        # fix removed the redundant L11 post-op attach (see
        # ``all_core_ops.py:497``), but this guard hardens
        # ``_MulCombineStage`` against any future schedule that places
        # multiple ``FlattenedALUMul`` instances on the same step.
        #
        # Detection: ``GEToBDConverter.forward`` writes one-hot indicators
        # scaled by 2.0 into ``OUTPUT_HI[k]`` on a MUL fire (line 335). A
        # band sum > 1.5 at the AX marker row implies a prior MUL composite
        # has already written this step — the second fire would double the
        # nibble. We zero the opcode mask on those rows so this composite
        # leaves OUTPUT untouched. Threshold 1.5 sits below the single-fire
        # signal (~2.0) and above residual noise from non-MUL writers; the
        # gate is conjoined with ``op_mul`` and ``mark_ax > 0.5`` so it
        # only matters when this stage would otherwise have fired.
        output_hi_band = x_bd[:, :, BD.OUTPUT_HI:BD.OUTPUT_HI + 16].sum(dim=-1)
        already_fired = (output_hi_band > 1.5).float()

        # Campaign MUL OUTPUT-flood cap (2026-06-21). In the 30-token campaign
        # config (C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1) the L11
        # efficient_l11_alumul_wrap wide_mul reads the L10-crushed (all-
        # negative) ALU_LO and FLOODS OUTPUT_LO/HI to ~2.4e9..+inf on the
        # OP_MUL+MARK_AX row (spec_k=0 block-trace: OUTPUT_HI 0->2.4e9 at
        # block 16, self-amplified to 6.9e9/L14, 1.2e20/L20, +inf/L25). That
        # flood trips ``already_fired`` (band >> 1.5) so the SE-recovered
        # FlattenedALUMul product is vetoed AND it out-votes the +2.0 product
        # at the LM-head argmax. When the recover flag is on we (a) IGNORE the
        # ``already_fired`` veto on the MUL+AX row (the band is the flood, NOT a
        # legitimate prior MUL fire) and (b) mark the row for an OUTPUT-band
        # CLEAR in _GEToBDStage so the flood is wiped before the clean product
        # is written. Gated on no_stack0_emit so the golden 35-token path
        # (clean positive operand one-hots, no flood) is byte-identical.
        output_clear_mask = None
        multibyte_boost_mask = None
        from .unified_compiler.ops.shared import (
            no_stack0_emit_enabled,
            mul_byte0_se_recover_enabled,
            mul_l19_flood_cap_enabled,
            mul_multibyte_l19_boost_enabled,
        )
        if no_stack0_emit_enabled() and mul_byte0_se_recover_enabled():
            # The cap fires only on rows where THIS composite is the MUL writer
            # (op_mul & MARK_AX) AND a flood is present (band far above the
            # ~2.0 legitimate single-fire signal). Use a high threshold so a
            # genuine prior MUL fire (~2.0) is NOT treated as a flood.
            mul_ax_row = (op_mul.view(B, seq_len) > 0.5) & (mark_ax > 0.5)
            flood_row = output_hi_band > 100.0
            cap_row = mul_ax_row & flood_row

            # L19-EXPLODE fix (2026-06-22, C4_MUL_L19_FLOOD_CAP). SINGLE-BYTE
            # products (3*15=45, 11*11=121, 1*10=10, 8*30=240) write the CORRECT
            # byte-0 product but at a MODERATE ~41 OUTPUT band — BELOW the 100.0
            # flood threshold — which the block-33 (logical L19) attention then
            # AMPLIFIES to ~555 (spec_k=0: ATTN in_LO=40.7 -> out_LO=555.6),
            # spreading the band so the argmax flips to OUTPUT cell 0 == 0x00 and
            # the AX high bytes pick up the flood -> a HUGE garbage AX (e.g.
            # 11*11 -> 2752768). The existing cap misses them (band < 100).
            #
            # Extend the cap to these moderate floods, but ONLY when the PRODUCT
            # is single-byte (byte 1 == 0). That is the exact L19-EXPLODE class
            # (product < 256), and gating on it leaves the MULTI-byte products
            # untouched: for those, clearing OUTPUT and re-firing the GEToBD
            # product write would ALSO re-stage this composite's byte 1 into
            # AX_FULL, which is WRONG for the campaign config (the FlattenedALUMul
            # operand-A byte 1 is the flooded reconstruction, e.g. 21*59 ->
            # 0xf0d7) and would CLOBBER the correct byte-1 relay -> regressing the
            # passing multi-byte muls (21*59, 65*98, 97*94). Single-byte products
            # have byte 1 == 0 so the re-stage writes 0 == correct. The byte-1
            # zero test reads the GE result positions 2/3 (byte-1 lo/hi nibbles)
            # the schoolbook multiply just computed. Gated on no_stack0_emit + the
            # mul recover flag -> golden byte-identical; opt-out via the flag.
            if mul_l19_flood_cap_enabled():
                # Product byte-1 nibbles live at GE result positions 2 and 3.
                res_b1 = (
                    x_ge_out[:, :, 2, ge.RESULT].abs()
                    + x_ge_out[:, :, 3, ge.RESULT].abs()
                )
                single_byte = (res_b1 < 0.5)
                moderate_flood = (
                    (output_hi_band > 10.0)
                    & (op_mul.view(B, seq_len) > 0.5)
                    & (mark_ax > 0.5)
                    & single_byte
                )
                cap_row = cap_row | moderate_flood
            # Where we cap: do NOT veto via already_fired (let the product
            # write), and clear the flooded OUTPUT band before the write.
            already_fired = already_fired * (~cap_row).float()
            output_clear_mask = cap_row.float()

            # NARROWED MULTI-byte L19 byte-0 boost (2026-06-25,
            # C4_MUL_MULTIBYTE_L19_BOOST). The single-byte cap above leaves the
            # MULTI-byte products (byte 1 != 0) on the default ~14.3 byte-0 band.
            # The block-34 / logical-L19 PureFFN then ADDS the byte-1 value into
            # OUTPUT_LO at ~180 on the MARK_AX row of the 6 LITERAL multi-byte
            # fails {127,130,134,139,141,144}, so the byte-0 LOW nibble argmax
            # flips to the byte-1 value (1960 -> 1799 = 0x0707). The byte-0 band
            # is a CLEAN one-hot here (mag ~14.3, NOT a flood), so we mark these
            # rows for a byte-0-only OUTPUT SCALE (no clear) in _GEToBDStage so
            # the true product cell out-votes the L19 add.
            #
            # NARROWING vs the dropped res_b1-only version (which fired on
            # var_mul's multi-byte MUL too and REGRESSED it -> its L11 wide_mul
            # has already mis-fired on the crushed multi-local operand band so
            # the boosted OUTPUT is garbage). DISCRIMINATOR (probed spec_k=0
            # READING the EXACT ``state.x_bd_in`` the composite receives, BUILT
            # dims, on the MUL+MARK_AX first-fire row): the STACK0_B0_H1_PREV +
            # STACK0_B0_H3_PREV cross-step carry band sum (the C4_STACK0_B0_DUMP
            # re-supply that only runs in a MULTI-step / multi-local frame) is
            #   * LITERAL mul (return N*M; single step, no ENT frame):  55..667
            #   * var_mul (ENT frame + LI-loaded multi-local operands): 7379
            #     (rock-solid UNIFORM across var_mul_0..23).
            # The threshold VAR_FRAME_CARRY_MIN = 2000 sits with a >3x margin on
            # BOTH sides (667 << 2000 << 7379) -> the boost fires on the 6
            # literal fails ONLY and leaves every var_mul row untouched (its
            # carry is 7379 >> 2000 -> NOT a literal frame). Excludes the capped
            # (single-byte) rows so the two paths never double-apply.
            #
            # NOTE: the carry band is NON-ZERO on the literal row at the COMPOSITE
            # INPUT (block-30 attention adds it) even though it is ~0 at the
            # block-29 OUTPUT — so the gate MUST read x_bd_in (this residual) and
            # use the 2000 magnitude split, NOT a near-zero threshold.
            if mul_multibyte_l19_boost_enabled():
                # Product byte-1 nibbles live at GE result positions 2 and 3.
                res_b1_mb = (
                    x_ge_out[:, :, 2, ge.RESULT].abs()
                    + x_ge_out[:, :, 3, ge.RESULT].abs()
                )
                # Var-frame discriminator: the cross-step STACK0 byte-0 dump
                # carry band — LARGE (~7379) in the multi-local var_mul frame,
                # MODERATE (<=667) for the single-step literal mul.
                BD = self.BD
                h1p = int(BD.STACK0_B0_H1_PREV)
                h3p = int(BD.STACK0_B0_H3_PREV)
                var_frame_carry = (
                    x_bd[:, :, h1p:h1p + 16].abs().sum(dim=-1)
                    + x_bd[:, :, h3p:h3p + 16].abs().sum(dim=-1)
                )
                VAR_FRAME_CARRY_MIN = 2000.0
                literal_frame = var_frame_carry < VAR_FRAME_CARRY_MIN
                multibyte_row = (
                    (op_mul.view(B, seq_len) > 0.5)
                    & (mark_ax > 0.5)
                    & (res_b1_mb >= 0.5)
                    & (~cap_row)
                    & literal_frame
                )
                multibyte_boost_mask = multibyte_row.float()

        opcode_mask = opcode_mask * (1.0 - already_fired)

        state.x_ge_out = x_ge_out
        state.opcode_mask = opcode_mask
        state.output_clear_mask = output_clear_mask
        state.multibyte_boost_mask = multibyte_boost_mask
        return state


class _GEToBDStage(nn.Module):
    """Final pipeline stage (phase=12.3): GE → BD conversion.

    Wraps ``GEToBDConverter`` and stores its ``[B, seq_len, 512]`` output in
    ``state.x_bd_out``, which ``FlattenedALUMul.forward`` then returns.
    """

    def __init__(self, BD, ge: GenericE, S: float):
        super().__init__()
        self.BD = BD
        self.ge = ge
        self.S = S
        self.ge_to_bd = GEToBDConverter(BD, ge, S)

    def forward(self, state: _MulPipelineState) -> _MulPipelineState:
        BD = self.BD
        x_bd_in = state.x_bd_in
        # Campaign MUL OUTPUT-flood cap: zero the flooded OUTPUT_LO/HI band on
        # the marked MUL+AX rows BEFORE the GE->BD writeback adds the clean
        # product one-hot (GEToBDConverter does ``OUTPUT += indicator*2.0`` on a
        # clone of x_bd_in, so the flood would otherwise survive and drown the
        # +2.0 product). Outside the campaign config output_clear_mask is None
        # and x_bd_in is passed through unchanged (byte-identical).
        if state.output_clear_mask is not None:
            clear = state.output_clear_mask[:, :, None]  # [B, seq_len, 1]
            keep = 1.0 - clear
            x_bd_in = x_bd_in.clone()
            lo = slice(BD.OUTPUT_LO, BD.OUTPUT_LO + 16)
            hi = slice(BD.OUTPUT_HI, BD.OUTPUT_HI + 16)
            x_bd_in[:, :, lo] = x_bd_in[:, :, lo] * keep
            x_bd_in[:, :, hi] = x_bd_in[:, :, hi] * keep
        state.x_bd_out = self.ge_to_bd(
            state.x_ge_out,
            x_bd_in,
            opcode_mask=state.opcode_mask,
            emit_carry=False,
        )
        # Campaign MUL L19 product BOOST (2026-06-23). The cap above cleared the
        # L11 wide_mul flood and ``ge_to_bd`` re-wrote the clean byte-0 product
        # one-hot at the +2.0 default. But the downstream block-33 (logical L19)
        # attention UNCONDITIONALLY adds +40 into OUTPUT_LO[0]/OUTPUT_HI[0] (a
        # broad "OUTPUT zero-byte default" copy that fires on the MUL emit row in
        # the depth>=1 stack contexts — expr_add_mul/paren/mul_div + some
        # single-byte standalone muls); at +2.0 that +40 cell-0 add out-votes the
        # true product cell -> the byte decodes to 0x00. BOOST the capped-row
        # OUTPUT band (which now holds ONLY the clean GEToBD product one-hot — the
        # flood was cleared, so a uniform scale leaves cell-0 at ~0 and lifts the
        # true product cell) to a DOMINANT magnitude (> the +40 L19 add) so the
        # product survives. byte-0 ONLY; the byte-1 AX_FULL relay is untouched.
        # Gated on no_stack0_emit + mul_byte0_se_recover (the cap is only set in
        # that config) so the golden 35-token path is byte-identical (the cap
        # mask is None there -> this branch is a no-op).
        if state.output_clear_mask is not None:
            from .unified_compiler.ops.shared import (
                mul_l19_product_boost_enabled,
            )
            if mul_l19_product_boost_enabled():
                # 50.0 product band > the +40 L19 zero-default; the +2.0 GEToBD
                # write scales by 25x. Applied ONLY on the cap rows.
                MUL_L19_PRODUCT_BOOST = 25.0
                boost = state.output_clear_mask[:, :, None]  # [B,seq,1] 0/1
                scale = 1.0 + boost * (MUL_L19_PRODUCT_BOOST - 1.0)
                lo = slice(BD.OUTPUT_LO, BD.OUTPUT_LO + 16)
                hi = slice(BD.OUTPUT_HI, BD.OUTPUT_HI + 16)
                x_bd_out = state.x_bd_out.clone()
                x_bd_out[:, :, lo] = x_bd_out[:, :, lo] * scale
                x_bd_out[:, :, hi] = x_bd_out[:, :, hi] * scale
                state.x_bd_out = x_bd_out

        # NARROWED MUL MULTI-byte L19 byte-0 BOOST (2026-06-25). The single-byte
        # cap (output_clear_mask) handles products with byte 1 == 0. The
        # LITERAL multi-byte products {127,130,134,139,141,144} keep a CLEAN
        # byte-0 one-hot (mag ~14.3, NOT a flood -> never cleared), but the
        # block-34 / logical-L19 PureFFN ADDS the byte-1 value into OUTPUT_LO at
        # ~180 on the MARK_AX row, so the byte-0 LOW nibble argmax flips to the
        # byte-1 value (1960 -> 1799 = 0x0707). SCALE the byte-0 OUTPUT_LO/HI
        # band on those rows so the true product cell beats the L19 add. NO
        # clear: the band is already a clean single-cell one-hot, so a uniform
        # scale only lifts the true cell (a one-hot's argmax is scale-invariant
        # -> the PASSING multi-byte literal muls are byte-identical). byte-0
        # (OUTPUT_LO/HI) ONLY -> the byte-1 AX_FULL relay (a different emit row)
        # is untouched. The mask is var-frame-GATED (set only on the literal
        # multi-byte rows in _MulCombineStage), so var_mul is NOT boosted -> the
        # var_mul-regressing failure mode of the dropped res_b1-only fix is
        # avoided. Gated on no_stack0_emit + mul_byte0_se_recover (the mask is
        # None otherwise) so the golden 35-token path is byte-identical.
        if state.multibyte_boost_mask is not None:
            from .unified_compiler.ops.shared import (
                mul_multibyte_l19_boost_enabled,
            )
            if mul_multibyte_l19_boost_enabled():
                MUL_MULTIBYTE_L19_BOOST = 25.0
                boost = state.multibyte_boost_mask[:, :, None]  # [B,seq,1] 0/1
                scale = 1.0 + boost * (MUL_MULTIBYTE_L19_BOOST - 1.0)
                lo = slice(BD.OUTPUT_LO, BD.OUTPUT_LO + 16)
                hi = slice(BD.OUTPUT_HI, BD.OUTPUT_HI + 16)
                x_bd_out = state.x_bd_out.clone()
                x_bd_out[:, :, lo] = x_bd_out[:, :, lo] * scale
                x_bd_out[:, :, hi] = x_bd_out[:, :, hi] * scale
                state.x_bd_out = x_bd_out
        return state


class CleanOperandOneHotFFN(nn.Module):
    """Feasibility probe: derive a CLEAN nibble one-hot operand delivery.

    Drop-in replacement for ``block.ffn`` of the L8 main block (physical block
    11 — the operand-delivery block where ``make_layer8_mem_to_alu_op`` head 5 /
    ``make_layer7_operand_gather_op`` write ALU_LO/HI at the binary-op MARK_AX
    row). Holds the original L8 ALU ``PureFFN`` (``inner``) and, BEFORE
    delegating to it, snaps the ALU_LO/HI (operand A) and AX_CARRY_LO/HI
    (operand B) bands to a CLEAN per-nibble one-hot on the MARK_AX operand rows.

    This is the ``C4_CLEAN_OPERAND`` feasibility flag (DEFAULT-OFF). It
    GENERALISES the address-leak / index-0-artifact correctors
    (``LoadedOperandAddHi15ClearFFN``, the func-add hi-nibble clear, the
    ``CmpOperandSeRecoverFFN`` hybrid rebuild): instead of clearing SPECIFIC
    leak cells, it derives a clean one-hot for the WHOLE band by keeping only
    the argmax cell (the true nibble, delivered at ~SCALE_O=6.0) and zeroing
    every other cell (the +0.45 cell-8 residue, the -0.52 cell-0 default-cancel
    residue, the ~5.5 address-nibble two-hot, the ~5.3 index-0 magnitude
    artifact). The kept cell is renormalised to the clean magnitude.

    Purpose: measure whether a correct-by-construction clean operand lets the
    downstream ALU/CMP derivations compute correctly WITHOUT their cleanup
    correctors — the decisive go/no-go for eliminating the operand correctors.

    Gate-OFF leaves ``block.ffn = inner`` exactly (byte-identical golden); the
    wrap is installed only when ``clean_operand_enabled()``.
    """

    # The clean true-nibble magnitude the downstream lanes expect (SCALE_O).
    CLEAN_MAG = 6.0
    # Noise floor: a cell below this is treated as absent (the golden dirt is
    # cell-8 ~0.45, cell-0 default-cancel ~0.52; the true one-hot is ~6.0, the
    # address/index-0 two-hot leaks are ~5.3-5.5 — all >> 0.9). We keep ONLY the
    # per-band argmax and zero the rest, so the floor only guards all-zero bands.
    NOISE_FLOOR = 0.9

    def __init__(self, inner: nn.Module, *, alu_lo, alu_hi, carry_lo, carry_hi,
                 mark_ax, op_dims):
        super().__init__()
        self.inner = inner
        self.alu_lo = int(alu_lo)
        self.alu_hi = int(alu_hi)
        self.carry_lo = int(carry_lo)
        self.carry_hi = int(carry_hi)
        self.mark_ax = int(mark_ax)
        # Consumer opcode flag dims whose MARK_AX operand rows get the clean-snap.
        # One-hot per step; a row is a snap row iff exactly one of these is set.
        #  * ``C4_CLEAN_OPERAND`` (the feasibility flag) passes ALL eleven
        #    consumers (ADD/SUB/MUL/DIV/MOD + six CMP) — cleans every op.
        #  * ``C4_CLEAN_OPERAND_ADD`` (the arithmetic-only CBC pass-gain) passes
        #    ONLY the five arithmetic dims (OP_ADD/SUB/MUL/DIV/MOD) — the CMP
        #    rows are NOT in ``op_dims`` so they are never snapped and stay
        #    byte-identical (the CMP calibration contract is left intact).
        self.op_dims = tuple(int(d) for d in op_dims)
        self._is_clean_operand_wrap = True

    def _clean_band(self, band, active):
        """Snap ``band`` [B,S,16] to a clean one-hot on ``active`` [B,S,1] rows.

        Keep only the argmax cell (if it clears NOISE_FLOOR) at CLEAN_MAG; zero
        everything else. Inactive rows are returned untouched.
        """
        # argmax cell + its value.
        maxv, maxi = band.max(dim=-1, keepdim=True)  # [B,S,1]
        has_hot = (maxv > self.NOISE_FLOOR).to(dtype=band.dtype)  # [B,S,1]
        onehot = torch.zeros_like(band)
        onehot.scatter_(-1, maxi, 1.0)
        clean = onehot * (self.CLEAN_MAG * has_hot)
        return band * (1.0 - active) + clean * active

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x[:, :, self.op_dims[0]] > 0.5
        for d in self.op_dims[1:]:
            gate = gate | (x[:, :, d] > 0.5)
        active = (gate & (x[:, :, self.mark_ax] > 0.5))[:, :, None].to(x.dtype)
        x = x.clone()
        for base in (self.alu_lo, self.alu_hi, self.carry_lo, self.carry_hi):
            sl = slice(base, base + 16)
            x[:, :, sl] = self._clean_band(x[:, :, sl], active)
        return self.inner(x)

    # ---- composite-FFN compatibility (plumb through to inner) ----
    def compact(self, block_size=1):
        if hasattr(self.inner, "compact"):
            return self.inner.compact(block_size=block_size)
        return None

    def sparsify(self):
        if hasattr(self.inner, "sparsify"):
            return self.inner.sparsify()
        return None

    def compact_moe(self, opcode_range=None, relay_map=None):
        fn = getattr(self.inner, "compact_moe", None)
        if fn is not None:
            return fn(opcode_range=opcode_range, relay_map=relay_map)
        return None


class CmpLoadedOperandCleanFFN(nn.Module):
    """Campaign func_max/func_min CMP loaded-operand-A two-hot clean.

    Drop-in replacement for ``block.ffn`` of the L9 block (the block whose FFN
    holds the L9 CMP nibble-comparator factory — ``_layer9_cmp_rules``). Its
    attention head ``layer9_step_end_operand_relay`` has ALREADY mirrored the
    per-step operand-A ``ALU_LO/HI`` into ``SE_ALU_LO/HI`` (Q@MARK_SE_ONLY,
    K@MARK_AX within-step) at the FFN's INPUT, so this wrap cleans the
    SE-tagged operand-A band on the ``MARK_SE_ONLY`` cmp row BEFORE the CMP
    factory rules read it, then delegates to the inner ``PureFFN``.

    THE ROOT (spec_k=0, BUILT dims, campaign+``C4_JSR_BP_BYTE3_CLEAR=1``, GPU/CPU
    faithful, ``tools/_probe_funcmax_cmp_operand.py`` + ``_probe_funcmax_cmp_sweep.py``):
    ``func_max`` / ``func_min`` (ids 650-699) return via a ``GT`` / ``LT`` compare
    + ``BZ`` branch + ``LEV`` — the loaded local ``a`` (``LI`` -> ``PSH`` ->
    ``mem[SP]``) is operand A of that compare. The L8 head-5 mem-to-ALU value copy
    delivers operand A into ``ALU_HI`` as a TWO-HOT: the true ``a//16`` cell PLUS a
    spurious ``~+0.94`` one-hot at the cell equal to operand-B's high nibble
    (``b//16``) — operand B (``AX_CARRY_HI``, the other loaded local) bleeding
    through head 5's operand-A read (the SAME leak class as func_add's
    ``C4_FUNC_ADD_B0_HINIB``, on the CMP path). The ``layer9_step_end_operand_relay``
    mirrors this two-hot into ``SE_ALU_HI``, and the L9 nibble comparator
    (``a_hi < b_hi`` / ``a_hi == b_hi``) reads the contaminated operand-A high
    nibble -> WRONG ``GT``/``LT`` flag -> wrong branch -> returns the wrong operand.
    The leak is DECISIVE precisely when ``a//16 == 0``: the immediate path delivers
    a positive ``SE_ALU_HI[0]`` one-hot (~+5.12) for a zero high nibble, but the
    LOADED path delivers NO cell-0 one-hot (only the ``~-0.02`` residue) so the
    ``~+0.94`` ``b_hi`` leak WINS the argmax and the comparator reads
    ``a_hi == b_hi`` instead of ``a_hi == 0``.

    THE FIX (this wrap; correct by construction, campaign-default-OFF, gated OFF
    byte-identical): on the ``MARK_SE_ONLY`` cmp rows ONLY, for BOTH the high and
    low nibble bands:
      1. LEAK CLEAR — zero ``SE_ALU_*[c]`` where ``c == b nibble`` (the
         ``SE_AX_CARRY_*`` one-hot, operand B) AND the cell is in the narrow leak
         window ``(0.5, LEAK_MAX=3.0)``. The ``~0.94`` leak is zeroed; a genuine
         operand-A one-hot is preserved: the IMMEDIATE ``a_hi==b_hi`` true one-hot
         lands ``~+5.63`` and the LOADED one lands ``~+6.59`` — BOTH ``>= 3.0`` so
         both survive (measured; window chosen to sit strictly between the 0.94
         leak and the 5.63 minimum true magnitude). This is what makes the fix
         inert on the shared ``if_gt`` / ``if_lt`` ``a_hi==b_hi`` path.
      2. ZERO-NIBBLE RECOVER — after the clear, if NO ``SE_ALU_*`` cell is strong
         (``>= LEAK_MAX``) on that cmp row, the true nibble is 0, so write a
         positive one-hot at cell 0 (``+ALU_CLEAN_MAG=5.3``, the immediate-path
         cell-0 magnitude) so the comparator's ``a==0`` / ``a<b`` unit fires
         (``SE_ALU_*+0`` must be positive for the L9 4-way AND to trip). Immediate
         ``a==0`` rows already carry a strong cell-0 one-hot, so the recover never
         fires there (inert on ``if_gt``/``if_lt`` immediates).

    Runs in ONE block (it IS ``block.ffn`` of the L9 block) so the physical block
    count is unchanged (the absolute-position LEA contract holds). Operand B
    (``SE_AX_CARRY``) is never rewritten; ``CMP`` is never written here. Gate-OFF /
    non-campaign leaves ``block.ffn = inner`` exactly (byte-identical to golden
    ``f725c06e`` flag-OFF); installed only when ``no_stack0_emit_enabled() and
    func_cmp_operand_clean_enabled()`` AND the ``SE_ALU`` mirror dims exist.
    """

    LEAK_MAX = 3.0        # upper bound of the ~0.94 b_hi leak window (< 5.63 true)
    ALU_CLEAN_MAG = 5.3   # golden immediate-path cell-0 one-hot magnitude (~5.12)

    def __init__(self, inner, *, se_alu_lo, se_alu_hi, se_ax_carry_lo,
                 se_ax_carry_hi, mark_se_only, se_cmp_op_dims):
        super().__init__()
        self.inner = inner
        self.se_alu_lo = int(se_alu_lo)
        self.se_alu_hi = int(se_alu_hi)
        self.se_ax_carry_lo = int(se_ax_carry_lo)
        self.se_ax_carry_hi = int(se_ax_carry_hi)
        self.mark_se_only = int(mark_se_only)
        # The six SE-tagged comparison opcode flag dims (one-hot per cmp step at
        # the SE row). Fires the wrap only on genuine cmp SE rows.
        self.se_cmp_op_dims = tuple(int(d) for d in se_cmp_op_dims)
        self._is_cmp_loaded_operand_clean_wrap = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cmp opcode (OR of the six SE_OP flags) AND MARK_SE_ONLY, per row.
        cmp_op = torch.zeros_like(x[:, :, self.mark_se_only])
        for d in self.se_cmp_op_dims:
            cmp_op = cmp_op + x[:, :, d]
        row = ((cmp_op > 0.5) & (x[:, :, self.mark_se_only] > 0.5))[:, :, None]
        row = row.to(dtype=x.dtype)  # [B,S,1]
        x = x.clone()
        oh0 = torch.zeros(16, device=x.device, dtype=x.dtype)
        oh0[0] = self.ALU_CLEAN_MAG
        oh0 = oh0.view(1, 1, 16)
        for a_base, b_base in (
            (self.se_alu_hi, self.se_ax_carry_hi),
            (self.se_alu_lo, self.se_ax_carry_lo),
        ):
            band = x[:, :, a_base:a_base + 16]                       # [B,S,16]
            b_oh = (x[:, :, b_base:b_base + 16] > 0.5).to(x.dtype)   # operand-B nibble
            in_win = ((band > 0.5) & (band < self.LEAK_MAX)).to(x.dtype)
            # 1. Zero the b-nibble leak cell (in-window only) on cmp SE rows.
            zero_mask = b_oh * in_win * row
            band2 = band * (1.0 - zero_mask)
            # 2. If no strong cell remains on this cmp SE row -> true nibble is 0
            #    -> write a positive cell-0 one-hot (immediate-path shape).
            strong = (band2 >= self.LEAK_MAX).any(dim=-1, keepdim=True).to(x.dtype)
            need0 = row[:, :, 0:1] * (1.0 - strong)                  # [B,S,1]
            band2 = band2 + need0 * oh0
            x[:, :, a_base:a_base + 16] = band2
        return self.inner(x)

    # ---- composite-FFN compatibility (plumb through to inner) ----
    def compact(self, block_size=1):
        if hasattr(self.inner, "compact"):
            return self.inner.compact(block_size=block_size)
        return None

    def sparsify(self):
        if hasattr(self.inner, "sparsify"):
            return self.inner.sparsify()
        return None

    def compact_moe(self, opcode_range=None, relay_map=None):
        fn = getattr(self.inner, "compact_moe", None)
        if fn is not None:
            return fn(opcode_range=opcode_range, relay_map=relay_map)
        return None


class MultiPassMulBlock(nn.Module):
    """GAP-PRIMITIVE #2: the 7-pass schoolbook MUL cascade as ONE block FFN.

    Drop-in ``block.ffn`` replacement for the L11 ``mul_partial`` lookup when
    ``C4_MUL_MULTIPASS=1``. Holds the 7 lowered ``PureFFN`` passes of
    ``multi_pass_mul_rules`` (partial products + column-carry chain) as an
    ordered ``nn.Sequential`` and runs them in a SINGLE block forward — exactly
    the ``FlattenedALUMul`` pattern (a multi-stage pipeline collapsed into one
    ``model.blocks[i].ffn`` call), so the model's physical block count is
    unchanged and the absolute-position lea contract holds.

    Each pass is ``PureFFN.forward(x) = x + swiglu(x)`` (residual add), so the
    Sequential threads the running residual through every pass: pass ``k``
    reads the workspace band (``MUL_MULTIPASS_WS``) that pass ``k-1`` wrote —
    the cross-pass carry chain a single-forward FFN lookup cannot express. This
    is the neural realization of ``MultiPassOp.run_symbolic``.

    The passes are opcode-gated (``OP_MUL``) and marker-gated (``MARK_AX``) at
    build time, so on a non-MUL / non-marker row every pass's SwiGLU is dark and
    the module is a pure residual identity — the same disjoint-gating contract
    the lookup mul_partial honoured.

    ``compact`` / ``sparsify`` / ``compact_moe`` plumb through to each pass so
    the model's post-bake compactor and any weight-introspection treat this like
    a stack of ``PureFFN``.
    """

    def __init__(self, passes):
        super().__init__()
        self.pipeline = nn.Sequential(*passes)
        self._is_multipass_mul_block = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(x)

    def compact(self, block_size=1):
        for p in self.pipeline:
            if hasattr(p, "compact"):
                p.compact(block_size=block_size)
        return None

    def sparsify(self):
        for p in self.pipeline:
            if hasattr(p, "sparsify"):
                p.sparsify()
        return None

    def compact_moe(self, opcode_range=None, relay_map=None):
        for p in self.pipeline:
            fn = getattr(p, "compact_moe", None)
            if fn is not None:
                fn(opcode_range=opcode_range, relay_map=relay_map)
        return None


class MultiPassDivBlock(nn.Module):
    """GAP-PRIMITIVE #2 (DIV pilot): the binary long-division cascade as ONE post_op.

    Drop-in ``post_op`` replacement for the L10 ``FlattenedDivMod`` composite when
    ``C4_DIV_MULTIPASS=1``. Holds the lowered ``PureFFN`` passes of
    ``multi_pass_div_rules`` (bit-serial shift-subtract with a cross-pass
    running-remainder carry) as an ordered ``nn.Sequential`` and runs them in a
    SINGLE post_op forward — exactly the ``MultiPassMulBlock`` / ``FlattenedALUMul``
    pattern (a multi-stage pipeline collapsed into one block call), so the model's
    physical block count is unchanged (both the composite and this are ONE
    ``post_ops`` entry) and the absolute-position lea contract holds.

    Each pass is ``PureFFN.forward(x) = x + swiglu(x)`` (residual add), so the
    Sequential threads the running residual through every pass: pass ``k`` reads
    the workspace band (``DIV_MULTIPASS_WS``) pass ``k-1`` wrote — the cross-pass
    running-remainder chain a single-forward FFN lookup cannot express.

    The cascade fires on ``OP_DIV OR OP_MOD`` via a 2-cell OR-gate scratch band
    (``DIV_MP_GATE`` cell 1) that a seed rule sets when EITHER opcode is present,
    and marker-gated (``MARK_AX``), so on a non-div/mod / non-marker row every
    pass's SwiGLU is dark and the compute is a pure residual identity. It writes
    the quotient (``a // b``) to ``DIV_MP_Q_LO/HI`` and remainder (``a % b``) to
    ``DIV_MP_R_LO/HI`` on dedicated result lanes.

    After the compute Sequential, this module IMPERATIVELY routes the result to
    OUTPUT — the counterpart of ``_DivModGEToBDStage``:

      * OP_DIV rows (at MARK_AX): OUTPUT_LO/HI <- quotient nibbles (Q_LO/Q_HI).
      * OP_MOD rows (at MARK_AX): OUTPUT_LO/HI <- remainder nibbles (R_LO/R_HI).

    and replicates the campaign divisor / dividend-byte-1 clears the live
    ``FlattenedDivMod`` GE->BD stage does (so the downstream L20 lev-routing / L18
    mem-gen leaks have no AX_CARRY / STACK0_BYTE_VAL_1 to materialize over the
    quotient). Those clears are gated on the SAME div/mod-AX mask.

    ``compact`` / ``sparsify`` / ``compact_moe`` plumb through to each pass so the
    model's post-bake compactor and any weight-introspection treat this like a
    stack of ``PureFFN``.
    """

    def __init__(self, passes, *, output_lo, output_hi,
                 q_lo, q_hi, r_lo, r_hi, op_div, op_mod, mark_ax,
                 ax_carry_lo=None, ax_carry_hi=None,
                 stack0_b1_lo=None, stack0_b1_hi=None,
                 alu_lo=None, alu_hi=None,
                 campaign_clear=False, scratch_bands=None):
        super().__init__()
        self.pipeline = nn.Sequential(*passes)
        self._is_multipass_div_block = True
        self.output_lo = int(output_lo)
        self.output_hi = int(output_hi)
        self.q_lo = int(q_lo)
        self.q_hi = int(q_hi)
        self.r_lo = int(r_lo)
        self.r_hi = int(r_hi)
        self.op_div = int(op_div)
        self.op_mod = int(op_mod)
        self.mark_ax = int(mark_ax)
        self.ax_carry_lo = None if ax_carry_lo is None else int(ax_carry_lo)
        self.ax_carry_hi = None if ax_carry_hi is None else int(ax_carry_hi)
        # Dividend (a) operand nibble bands = ALU_LO (a0) / ALU_HI (a1); the
        # divisor (b) bands = AX_CARRY_LO (b0) / AX_CARRY_HI (b1). The p0_seed
        # pass reads these four 16-cell one-hot bands at weight 1.0 assuming an
        # incoming residual == 1.0 (the amplitude-normalized cascade convention
        # in ``wide_alu_dsl``). On the REAL teacher-forced residual they carry
        # max_abs ~72 (attention relays amplify the active lane; the L6 FFN can
        # drive inactive lanes negative), so an un-clamped read seeds the
        # cascade OFF its fixed point and the running residual explodes to
        # ~1.5e37 -> inf -> NaN by pass 21. Clamping them to clean 0/1 one-hots
        # at the block INPUT (matching the composite's
        # ``GEToBDConverter._clean_onehot``) restores the seed invariant so the
        # cascade stays numerically stable. See ``forward``.
        self.alu_lo = None if alu_lo is None else int(alu_lo)
        self.alu_hi = None if alu_hi is None else int(alu_hi)
        self.stack0_b1_lo = None if stack0_b1_lo is None else int(stack0_b1_lo)
        self.stack0_b1_hi = None if stack0_b1_hi is None else int(stack0_b1_hi)
        self.campaign_clear = bool(campaign_clear)
        # (base, width) op-local scratch/result bands the cascade populates
        # with intermediate one-hots. They live OUTSIDE the 30/35-token step
        # frame, so if left hot they persist into the residual + KV of the
        # NEXT autoregressive step and corrupt its decode (observed as a
        # step-2 pc=None frame collapse). Zero them on the div/mod-AX rows
        # AFTER the OUTPUT routing has consumed them — the block is a leaf
        # scratchpad, nothing downstream reads these bands.
        self.scratch_bands = tuple(
            (int(base), int(width)) for base, width in (scratch_bands or ())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 0) OPERAND CLAMP (the NaN fix). The p0_seed pass reads the dividend
        #    (ALU_LO/HI) and divisor (AX_CARRY_LO/HI) nibble one-hot bands at
        #    weight 1.0, and the whole amplitude-normalized cascade assumes an
        #    incoming residual of exactly 1.0 on every one-hot it consumes. On
        #    the REAL teacher-forced residual those operand bands carry
        #    max_abs ~72 (attention relays amplify the active lane; the L6 FFN
        #    drives inactive lanes negative). An un-clamped read seeds the
        #    cascade off its fixed point, so each pass AMPLIFIES instead of
        #    renormalizing and the running residual explodes to ~1.5e37 -> inf
        #    -> NaN by pass 21 (all-NaN block output -> frame collapse). The
        #    composite ``FlattenedDivMod`` never hits this because its
        #    ``GEToBDConverter._clean_onehot`` threshold-clamps the SAME four
        #    bands to exact 0/1 one-hots before the convert. Mirror that here:
        #    at the div/mod-AX rows ONLY (so no other op's operand read is
        #    perturbed), replace each operand band with a clean 0/1 one-hot
        #    ``(clamp(band, 0, 1) > 0.5)`` before the cascade runs. This is the
        #    seed invariant the cascade was designed around, restored on the
        #    real input. Gated rows outside div/mod-AX are byte-identical.
        pre_mark = (x[:, :, self.mark_ax] > 0.5)
        pre_div = (x[:, :, self.op_div] > 0.1) & pre_mark
        pre_mod = (x[:, :, self.op_mod] > 0.1) & pre_mark
        pre_dm = (pre_div | pre_mod)[:, :, None].to(dtype=x.dtype)  # [B, seq, 1]
        operand_bands = [b for b in (self.alu_lo, self.alu_hi,
                                     self.ax_carry_lo, self.ax_carry_hi)
                         if b is not None]
        if operand_bands and bool(pre_dm.any()):
            x = x.clone()
            for base in operand_bands:
                band = x[:, :, base:base + 16]
                clean = (torch.clamp(band, min=0.0, max=1.0) > 0.5).to(
                    dtype=x.dtype
                )
                # Only overwrite on div/mod-AX rows; elsewhere keep the raw band.
                x[:, :, base:base + 16] = clean * pre_dm + band * (1.0 - pre_dm)

        # 1) run the compute cascade (residual threaded through all passes).
        x = self.pipeline(x)

        # 2) route the computed quotient/remainder into OUTPUT at div/mod-AX
        #    rows — the GE->BD writeback counterpart. Per-position masks.
        mark = (x[:, :, self.mark_ax] > 0.5).float()
        div = (x[:, :, self.op_div] > 0.1).float() * mark    # [B, seq]
        mod = (x[:, :, self.op_mod] > 0.1).float() * mark
        any_dm = ((div + mod) > 0.5).float()

        def _argmax_onehot(base):
            band = x[:, :, base:base + 16]
            idx = band.argmax(dim=-1, keepdim=True)   # [B, seq, 1]
            oh = torch.zeros_like(band)
            oh.scatter_(-1, idx, 1.0)
            return oh

        q_lo_oh = _argmax_onehot(self.q_lo)
        q_hi_oh = _argmax_onehot(self.q_hi)
        r_lo_oh = _argmax_onehot(self.r_lo)
        r_hi_oh = _argmax_onehot(self.r_hi)

        # Selected result nibble one-hots: quotient for DIV, remainder for MOD.
        sel_lo = q_lo_oh * div[:, :, None] + r_lo_oh * mod[:, :, None]
        sel_hi = q_hi_oh * div[:, :, None] + r_hi_oh * mod[:, :, None]

        # Route the result into OUTPUT with the SAME ADDITIVE convention the
        # composite it replaces uses (``GEToBDConverter``: OUTPUT +=
        # indicator*2.0, NEVER a destructive clear). The additive +2.0 out-votes
        # the upstream ALU_LO->OUTPUT relay at the div/mod-AX row while LEAVING
        # the rest of the OUTPUT emission state (byte-1/2/3 nibble structure the
        # 30-token frame emits) intact. An earlier destructive overwrite-clear
        # ERASED that state and desynced the emitted token count -> the next
        # step's fixed-slice decode read a shifted PC (step-2 pc=None). ``div`` /
        # ``mod`` are already the [B, seq]-broadcast div/mod-AX masks, so the
        # write only lands on those rows.
        x = x.clone()
        x[:, :, self.output_lo:self.output_lo + 16] += sel_lo * 2.0
        x[:, :, self.output_hi:self.output_hi + 16] += sel_hi * 2.0

        # 3) campaign divisor / dividend-byte-1 clears (same as FlattenedDivMod
        #    GE->BD stage) so the downstream L20/L18 leaks have nothing to
        #    materialize over the quotient. Gated on the div/mod-AX mask.
        if self.campaign_clear:
            keep_dm = (1.0 - any_dm)[:, :, None].to(dtype=x.dtype)
            for lo, hi in ((self.ax_carry_lo, self.ax_carry_hi),
                           (self.stack0_b1_lo, self.stack0_b1_hi)):
                if lo is not None:
                    x[:, :, lo:lo + 16] = x[:, :, lo:lo + 16] * keep_dm
                if hi is not None:
                    x[:, :, hi:hi + 16] = x[:, :, hi:hi + 16] * keep_dm

        # 4) scratchpad clear: zero the cascade's workspace + gate + result
        #    lanes on the div/mod-AX rows. The 43 passes leave ~110 stale
        #    one-hots (99 in the 1728-dim workspace, plus the OR-gate and the
        #    Q/R result lanes) at residual ~1.0. These op-local bands are not
        #    part of the per-step token frame and are read by NOTHING
        #    downstream (the OUTPUT routing above already consumed the result),
        #    so left hot they only leak into the NEXT step's residual/KV and
        #    frame-collapse its decode. Gate on any_dm so non-div/mod rows are
        #    a pure identity (byte-identity-safe: on a flag-OFF build this
        #    module never installs).
        if self.scratch_bands:
            keep_dm = (1.0 - any_dm)[:, :, None].to(dtype=x.dtype)
            for base, width in self.scratch_bands:
                x[:, :, base:base + width] = (
                    x[:, :, base:base + width] * keep_dm
                )

        return x

    def compact(self, block_size=1):
        for p in self.pipeline:
            if hasattr(p, "compact"):
                p.compact(block_size=block_size)
        return None

    def sparsify(self):
        for p in self.pipeline:
            if hasattr(p, "sparsify"):
                p.sparsify()
        return None

    def compact_moe(self, opcode_range=None, relay_map=None):
        for p in self.pipeline:
            fn = getattr(p, "compact_moe", None)
            if fn is not None:
                fn(opcode_range=opcode_range, relay_map=relay_map)
        return None


# ---------------------------------------------------------------------------
# Flattened AND/OR/XOR pipeline (vanilla nn.Sequential composite).
#
# 4 stages, each a separate nn.Module installed at L10.ffn by the compiler:
#   Stage 0: BD → GE format conversion          (BitwiseBDToGEStage)
#   Stage 1: per-opcode bit extraction          (BitwiseBitExtractStage)
#   Stage 2: per-opcode bit combine + opcode    (BitwiseBitCombineStage)
#            mask merge into RESULT
#   Stage 3: GE → BD format conversion + write  (BitwiseGEToBDStage)
#            OUTPUT_LO/HI gated by AX marker
#
# The stages share intermediate GE-format state via a `BitwisePipelineState`
# object so the residual stream (BD-format) never has to carry the GE
# workspace. Each stage takes ``x_bd`` in and returns ``x_bd`` out (residual
# identity for stages 0-2, real writeback in stage 3) so the composite forward
# is a literal `nn.Sequential` chain — vanilla composition, no hand-rolled
# control flow.
#
# Forward semantics are byte-identical to the existing ``ALUAndOrXor.forward``
# (= ``PureNeuralALU(operations='bitwise').forward``):
#   - same `BDToGEConverter` weights for stage 0
#   - same `build_{and,or,xor}_layers` factories for stages 1-2 (each builder
#     yields [BitExtractFFN, BitCombineFFN]; AND/OR/XOR run in parallel on
#     cloned GE buffers, then merge via 0.1-threshold opcode mask)
#   - same MARK_AX > 0.5 gating before GE → BD writeback
# ---------------------------------------------------------------------------


class BitwisePipelineState:
    """Per-composite scratch space for the 4-stage AND/OR/XOR pipeline.

    Attaches to ``FlattenedALUAndOrXor`` and is referenced by all 4 stage
    modules. Holds intermediate tensors so each stage runs as a standalone
    block without serialising the full GE workspace into the BD residual.
    """

    def __init__(self):
        self.x_bd_in = None
        self.x_ge_flat = None    # [B*seq, 8, 160] — initial GE state (clone src)
        self.x_and = None        # [B*seq, 8, 160] — AND pipeline buffer
        self.x_or = None         # [B*seq, 8, 160] — OR pipeline buffer
        self.x_xor = None        # [B*seq, 8, 160] — XOR pipeline buffer
        self.x_ge_out = None     # [B, seq, 8, 160]  — merged result GE
        self.opcode_mask = None  # [B, seq] — opcode mask gated by MARK_AX


class BitwiseBDToGEStage(nn.Module):
    """Stage 0: BD → GenericE format conversion.

    Converts the one-hot ALU_LO/HI, AX_CARRY_LO/HI nibbles into scalar
    NIB_A/NIB_B slots and copies opcode flags into the GE OP_START region.
    Stashes the converted state for downstream stages and returns x_bd
    unchanged (residual identity).
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.state = state
        self.bd_to_ge = BDToGEConverter(BD, self.ge)

    def forward(self, x_bd):
        x_ge = self.bd_to_ge(x_bd)  # [B, seq, 8, 160]
        B, seq_len, _, _ = x_ge.shape
        x_ge_flat = x_ge.view(B * seq_len, 8, self.ge.DIM)
        # Initialise per-opcode pipeline buffers as clones of the GE state.
        self.state.x_bd_in = x_bd
        self.state.x_ge_flat = x_ge_flat
        self.state.x_and = x_ge_flat.clone()
        self.state.x_or = x_ge_flat.clone()
        self.state.x_xor = x_ge_flat.clone()
        return x_bd


class BitwiseBitExtractStage(nn.Module):
    """Stage 1: per-opcode bit extraction (BitExtractFFN).

    Each of AND/OR/XOR has its own `BitExtractFFN` (gated on its own opcode)
    that splits NIB_A and NIB_B into the per-bit temp slots used by the
    combine stage. Runs all three in parallel on the cloned GE buffers
    seeded by stage 0. Mirrors the ``layers[0]`` step of `build_and_layers`,
    `build_or_layers`, `build_xor_layers` from `alu.ops.bitwise`.
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        # build_and/or/xor_layers each return [BitExtractFFN, CombineFFN]
        # for non-BIT configs (NIBBLE has chunk_bits=4). Take index 0 here.
        self.and_extract = build_and_layers(NIBBLE, opcode=30)[0]
        self.or_extract = build_or_layers(NIBBLE, opcode=28)[0]
        self.xor_extract = build_xor_layers(NIBBLE, opcode=29)[0]

    def forward(self, x_bd):
        self.state.x_and = self.and_extract(self.state.x_and)
        self.state.x_or = self.or_extract(self.state.x_or)
        self.state.x_xor = self.xor_extract(self.state.x_xor)
        return x_bd


class BitwiseBitCombineStage(nn.Module):
    """Stage 2: per-opcode bit combine + opcode-mask merge.

    Each of AND/OR/XOR runs its combine FFN (`BitAndCombineClearFFN`,
    `BitOrCombineClearFFN`, `BitXorCombineClearFFN`) on the per-opcode
    buffer from stage 1. Outputs are merged into a single GE result buffer
    via per-opcode masks (opcode value > 0.1 → 1.0). Stashes the merged
    GE state and the opcode mask for stage 3.
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.and_combine = build_and_layers(NIBBLE, opcode=30)[1]
        self.or_combine = build_or_layers(NIBBLE, opcode=28)[1]
        self.xor_combine = build_xor_layers(NIBBLE, opcode=29)[1]

    def forward(self, x_bd):
        x_and = self.and_combine(self.state.x_and)
        x_or = self.or_combine(self.state.x_or)
        x_xor = self.xor_combine(self.state.x_xor)

        ge = self.ge
        x_ge_flat = self.state.x_ge_flat

        # FIX 2026-05-06: Normalize opcode values to 0/1 (matches PureNeuralALU.forward).
        op_and = (x_ge_flat[:, 0, ge.OP_START + 30] > 0.1).float()
        op_or = (x_ge_flat[:, 0, ge.OP_START + 28] > 0.1).float()
        op_xor = (x_ge_flat[:, 0, ge.OP_START + 29] > 0.1).float()
        op_total = op_and + op_or + op_xor

        x_ge_out = x_ge_flat.clone()
        x_ge_out[:, :, ge.RESULT] = (
            x_and[:, :, ge.RESULT] * op_and[:, None]
            + x_or[:, :, ge.RESULT] * op_or[:, None]
            + x_xor[:, :, ge.RESULT] * op_xor[:, None]
        )

        x_bd_in = self.state.x_bd_in
        B, seq_len, _ = x_bd_in.shape
        x_ge_out = x_ge_out.view(B, seq_len, 8, ge.DIM)
        opcode_mask = op_total.view(B, seq_len)

        # MARK_AX gating (matches PureNeuralALU.forward post-loop step).
        BD = self.BD
        mark_ax = x_bd_in[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        self.state.x_ge_out = x_ge_out
        self.state.opcode_mask = opcode_mask
        return x_bd


class BitwiseGEToBDStage(nn.Module):
    """Stage 3: GenericE → BD format conversion + write OUTPUT.

    Reads the merged GE RESULT from stage 2 and writes one-hot OUTPUT_LO/HI
    into x_bd, gated by the AX-marker-aware opcode mask. Returns the
    updated x_bd as the FFN output.
    """

    def __init__(self, S, BD, state: BitwisePipelineState):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = state
        self.ge = GenericE(NIBBLE)
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

    def forward(self, x_bd):
        x_ge_out = self.state.x_ge_out
        opcode_mask = self.state.opcode_mask
        x_bd_out = self.ge_to_bd(
            x_ge_out,
            x_bd,
            opcode_mask=opcode_mask,
            emit_carry=False,
        )
        return x_bd_out


class FlattenedALUAndOrXor(nn.Module):
    """Vanilla composite AND/OR/XOR FFN: 4 sub-stages run as `nn.Sequential`.

    Drop-in replacement for ``ALUAndOrXor`` (= ``PureNeuralALU(operations=
    'bitwise')``) at L10. Forward is byte-identical to the original wrapper:
    same `BDToGEConverter`, same `build_{and,or,xor}_layers` FFNs, same
    0.1-threshold opcode mask, same MARK_AX gating. The difference is
    structural — the monolithic forward() is split into 4 stage modules
    that share intermediate state via a `BitwisePipelineState` and chain
    via a literal ``nn.Sequential``.

    The 4 sub-stages are exposed as named attributes so a future compiler
    bake_fn pipeline (analogous to ``make_l13_alu_shift_*_op``) can install
    each stage independently. Until those compiler ops exist, this class
    can be instantiated directly as a runtime-equivalent of ALUAndOrXor.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.state = BitwisePipelineState()
        # Vanilla nn.Sequential composition: each stage takes x_bd → x_bd.
        # Stages are registered ONLY through `pipeline` (not as direct
        # attributes) to avoid double-counting in `state_dict()`. Access them
        # via `bdtoge_stage`/`bit_extract_stage`/`bit_combine_stage`/
        # `getobd_stage` properties below for readability + compiler-op hooks.
        self.pipeline = nn.Sequential(
            BitwiseBDToGEStage(S, BD, self.state),
            BitwiseBitExtractStage(S, BD, self.state),
            BitwiseBitCombineStage(S, BD, self.state),
            BitwiseGEToBDStage(S, BD, self.state),
        )

    # Named-stage views into `self.pipeline` so external callers (compiler
    # bake_fns, debug introspection) can fetch a stage without hard-coding
    # the index. `nn.Sequential` indexing is supported as of PyTorch 1.0+.
    @property
    def bdtoge_stage(self) -> BitwiseBDToGEStage:
        return self.pipeline[0]

    @property
    def bit_extract_stage(self) -> BitwiseBitExtractStage:
        return self.pipeline[1]

    @property
    def bit_combine_stage(self) -> BitwiseBitCombineStage:
        return self.pipeline[2]

    @property
    def getobd_stage(self) -> BitwiseGEToBDStage:
        return self.pipeline[3]

    def forward(self, x_bd):
        return self.pipeline(x_bd)

    # Stub methods for compatibility with vm_step.py model utilities
    # (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class FlattenedALUMul(nn.Module):
    """Flattened (compiler-baked) MUL ALU — Sequential of vanilla stages.

    Byte-identical to ``ALUMul.forward`` (= ``PureNeuralALU(operations='mul').forward``)
    but exposes the BD↔GE converters and the 7 sub-FFN MUL pipeline stages as
    individually-installable submodules so the unified compiler can bake each
    stage as a discrete ``Operation``.

    The pipeline is materialised as a single ``nn.Sequential`` whose stages
    each implement ``forward(state) -> state``. Once all 9 compiler ops have
    run, ``self.pipeline`` is::

        nn.Sequential(
            _BDToGEStage,                  # phase=11.0
            _MulFFNStage(SchoolbookFFN),   # phase=11.1
            _MulFFNStage(CarryPassFFN(0)), # phase=11.2
            _MulFFNStage(CarryPassFFN(1)), # phase=11.3
            _MulFFNStage(CarryPassFFN(2)), # phase=11.4
            _MulFFNStage(MulGenPropFFN),   # phase=12.0
            _MulFFNStage(MulBinaryLookaheadFFN),  # phase=12.1
            _MulFFNStage(MulFinalCorrectionFFN),  # phase=12.2
            _MulCombineStage,              # opcode/AX gating + reshape
            _GEToBDStage,                  # phase=12.3
        )

    ``forward`` is therefore just::

        state = _MulPipelineState(); state.x_bd_in = x_bd
        return self.pipeline(state).x_bd_out

    No Python control flow over the sub-modules: the chaining is the
    declarative ``nn.Sequential`` itself.

    Sub-stages, installed by 9 compiler block ops:

      - ``bd_to_ge``:           ``BDToGEConverter`` — phase=11.0
      - ``mul_layers[0]``:      ``SchoolbookFFN``           — phase=11.1
      - ``mul_layers[1]``:      ``CarryPassFFN(pass_idx=0)``— phase=11.2
      - ``mul_layers[2]``:      ``CarryPassFFN(pass_idx=1)``— phase=11.3
      - ``mul_layers[3]``:      ``CarryPassFFN(pass_idx=2)``— phase=11.4
      - ``mul_layers[4]``:      ``MulGenPropFFN``           — phase=12.0
      - ``mul_layers[5]``:      ``MulBinaryLookaheadFFN``   — phase=12.1
      - ``mul_layers[6]``:      ``MulFinalCorrectionFFN``   — phase=12.2
      - ``ge_to_bd``:           ``GEToBDConverter``         — phase=12.3

    NIBBLE config produces exactly 3 carry passes (verified via
    ``_compute_carry_passes(NIBBLE) == [112, 7, 1]``), so the pipeline has
    the canonical 7-stage shape (1 schoolbook + 3 carry + 1 genprop +
    1 lookahead + 1 final-correction). The ops sort by phase (11.0 .. 12.3)
    and bind to L11 via ``layer_idx=11`` since the runtime collapses them
    into a single block forward pass (matching the previous ``ALUMul``
    that ran as one ``model.blocks[11].ffn`` call).

    Forward replicates the ``PureNeuralALU`` mul branch byte-for-byte
    (same 0.1-threshold opcode normalization, same MARK_AX-only OUTPUT
    gating).

    Backward-compat properties ``bd_to_ge``, ``ge_to_bd``, and
    ``mul_layers`` remain readable so existing inspectors / tests continue
    to work; they are computed from the corresponding stages in
    ``self._stages``.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)

        # Each install_* call appends the corresponding stage to _stages.
        # Once all 9 stages are installed, ``pipeline`` is a single
        # nn.Sequential of vanilla nn.Modules — no Python control flow over
        # sub-modules in forward().
        self._stages = nn.ModuleList()
        self.pipeline = None  # nn.Sequential, built lazily after install_getobd

    # --- Backward-compat property accessors (read-only views) ---------

    @property
    def bd_to_ge(self):
        """Return the ``BDToGEConverter`` instance, or ``None`` if not yet installed."""
        for stage in self._stages:
            if isinstance(stage, _BDToGEStage):
                return stage.bd_to_ge
        return None

    @property
    def ge_to_bd(self):
        """Return the ``GEToBDConverter`` instance, or ``None`` if not yet installed."""
        for stage in self._stages:
            if isinstance(stage, _GEToBDStage):
                return stage.ge_to_bd
        return None

    @property
    def mul_layers(self):
        """Return the list of MUL sub-FFNs in install order."""
        return [s.sub_ffn for s in self._stages if isinstance(s, _MulFFNStage)]

    # --- Per-stage installers (called by compiler ops) -----------------

    def install_bdtoge(self):
        """phase=11.0: install BD → GE converter as the first pipeline stage."""
        assert len(self._stages) == 0, (
            f"Expected bd_to_ge to be the first stage; "
            f"already installed {len(self._stages)} stages"
        )
        self._stages.append(_BDToGEStage(self.BD, self.ge))

    def install_schoolbook(self):
        """phase=11.1: append the schoolbook partial-product stage."""
        from .alu.ops.mul import SchoolbookFFN
        # Stages so far: [_BDToGEStage] (1 stage).
        assert len(self._stages) == 1, (
            f"Expected schoolbook to be the second stage; "
            f"already installed {len(self._stages)} stages"
        )
        self._stages.append(_MulFFNStage(SchoolbookFFN(self.ge, opcode=27)))

    def install_carrypass(self, pass_idx: int):
        """phase=11.2/11.3/11.4: append the i-th carry pass (i=0,1,2)."""
        from .alu.ops.mul import CarryPassFFN, _compute_carry_passes
        passes = _compute_carry_passes(self.ge.config)
        assert pass_idx < len(passes), (
            f"NIBBLE config has {len(passes)} carry passes; "
            f"asked for pass_idx={pass_idx}"
        )
        # Stages so far: [_BDToGEStage, schoolbook, carry_0..carry_{pass_idx-1}]
        # → 2 + pass_idx total before this install.
        expected_len = 2 + pass_idx
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before installing "
            f"carrypass {pass_idx}; got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(
            CarryPassFFN(self.ge, opcode=27,
                         max_carry=passes[pass_idx], pass_idx=pass_idx)
        ))

    def install_genprop(self):
        """phase=12.0: append the gen/prop stage."""
        from .alu.ops.mul import MulGenPropFFN, _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        # Stages so far: bdtoge + schoolbook + n_passes carry = 2 + n_passes.
        expected_len = 2 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before genprop; "
            f"got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(MulGenPropFFN(self.ge, opcode=27)))

    def install_binarylookahead(self):
        """phase=12.1: append the binary carry-lookahead stage."""
        from .alu.ops.mul import MulBinaryLookaheadFFN, _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        expected_len = 3 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before lookahead; "
            f"got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(
            MulBinaryLookaheadFFN(self.ge, opcode=27)
        ))

    def install_finalcorrection(self):
        """phase=12.2: append the final-correction stage."""
        from .alu.ops.mul import MulFinalCorrectionFFN, _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        expected_len = 4 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before "
            f"final-correction; got {len(self._stages)}"
        )
        self._stages.append(_MulFFNStage(
            MulFinalCorrectionFFN(self.ge, opcode=27)
        ))

    def install_getobd(self):
        """phase=12.3: install GE → BD converter and seal the Sequential."""
        from .alu.ops.mul import _compute_carry_passes
        n_passes = len(_compute_carry_passes(self.ge.config))
        # bdtoge + schoolbook + n_passes carry + genprop + lookahead +
        # finalcorrection = 5 + n_passes.
        expected_len = 5 + n_passes
        assert len(self._stages) == expected_len, (
            f"Expected {expected_len} stages before ge_to_bd; "
            f"got {len(self._stages)}"
        )
        # Append combine + getobd, then materialise the Sequential.
        self._stages.append(_MulCombineStage(self.BD, self.ge))
        self._stages.append(_GEToBDStage(self.BD, self.ge, self.S))
        self.pipeline = nn.Sequential(*self._stages)

    @classmethod
    def build_fully_baked(cls, S, BD):
        """Construct a fully-baked composite with all 9 stages installed.

        Convenience factory for callers that want a drop-in replacement for
        ``ALUMul(S, BD)`` (which self-bakes in ``__init__``) without running
        the 9 compiler ops manually. Forward is byte-identical to
        ``ALUMul.forward`` once this constructor returns.
        """
        from .alu.ops.mul import _compute_carry_passes
        module = cls(S, BD)
        module.install_bdtoge()
        module.install_schoolbook()
        for pass_idx in range(len(_compute_carry_passes(module.ge.config))):
            module.install_carrypass(pass_idx=pass_idx)
        module.install_genprop()
        module.install_binarylookahead()
        module.install_finalcorrection()
        module.install_getobd()
        return module

    # --- Forward (byte-identical to PureNeuralALU mul branch) ----------

    def forward(self, x_bd):
        if self.pipeline is None:
            # Reproduce the previous "missing stages" diagnostic so partial
            # bakes still fail loudly.
            missing = []
            if self.bd_to_ge is None:
                missing.append('bd_to_ge')
            if len(self.mul_layers) == 0:
                missing.append('mul_layers')
            if self.ge_to_bd is None:
                missing.append('ge_to_bd')
            raise RuntimeError(
                f"FlattenedALUMul: missing stages {missing}. "
                "All 9 compiler ops (phase 11.0..12.3) must run before "
                "forward()."
            )

        # ---- Opcode-gated early-out (perf optimization, 2026-05-11) ----
        # The full 9-stage MUL pipeline (BD→GE, schoolbook, 3 carry passes,
        # gen/prop, binary lookahead, final-correction, combine, GE→BD)
        # runs unconditionally and is masked to zero at the combine stage
        # when OP_MUL is not active. For the vast majority of forward
        # passes, OP_MUL is not active, so this work is pure waste.
        #
        # `_MulCombineStage` ultimately gates the writeback on
        # `OP_MUL>0.1 AND MARK_AX>0.5` per sequence position. If OP_MUL
        # does not exceed 0.1 anywhere in the batch, the pipeline
        # contributes nothing and we can return x_bd unchanged. The
        # `.item()` forces one CPU/GPU sync (~few µs) — a rounding error
        # compared to the multi-stage pipeline it elides.
        #
        # Correctness: this is a strict subset of the mask already
        # applied at the combine stage (we only skip when the mask would
        # zero the contribution everywhere). Numerical output is
        # identical when OP_MUL is active. Mirrors the early-out in
        # `FlattenedDivMod.forward` (perf-divmod-early-out).
        #
        # ONNX export & torch.compile: the `.item()` call would force a
        # CPU/GPU sync and graph break. Under tracing or compilation we
        # always run the full pipeline; the combine-stage mask zeroes the
        # writeback correctly when MUL is inactive, so output stays
        # byte-identical. Per docs/ONNX_EXPORT_STATUS_2026_05_11.md
        # blocker 2.
        if (not torch.onnx.is_in_onnx_export()
                and not torch.compiler.is_compiling()
                and x_bd[..., self.BD.OP_MUL].max().item() < 0.1):
            return x_bd  # no-op when MUL isn't the active opcode

        state = _MulPipelineState()
        state.x_bd_in = x_bd
        out_state = self.pipeline(state)
        return out_state.x_bd_out

    # Stub methods for compatibility with vm_step.py (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


class ALUMul(PureNeuralALU):
    """Neural MUL.

    DEPRECATED 2026-05-10: kept for back-compat only. Use
    ``FlattenedALUMul`` (assembled by 9 compiler ops at L11 phases
    11.0/11.1/11.2/11.3/11.4/12.0/12.1/12.2/12.3) instead. ``set_vm_weights``
    no longer instantiates this class — the compiler installs the flattened
    version via ``make_l11_alu_mul_*`` / ``make_l12_alu_mul_*`` ops.
    """
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='mul')


class ALUShift(PureNeuralALU):
    """Neural SHL/SHR.

    DEPRECATED (2026-05-10): The runtime wrapper class is being eliminated in
    favor of 4 separate compiler-driven sub-stages (BDToGE → SHL/SHR
    precompute → SHL/SHR select → GEToBD) installed at L13 by the compiler.
    See ``ALUShiftComposite`` and ``make_l13_alu_shift_*_op`` in
    ``unified_compiler/migrated_ops.py``. Kept here for backward compatibility
    until all callers move to the composite.
    """
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='shift')


# ---------------------------------------------------------------------------
# Flattened SHL/SHR pipeline (replaces ALUShift wrapper).
#
# 4 stages, each a separate nn.Module installed at L13.ffn by the compiler.
# Forward semantics are byte-identical to ``ALUShift.forward``.
#
# The composite is "vanilla": ``ALUShiftComposite.forward`` runs the pipeline
# inline by calling each stage's owned submodules directly, so there is no
# side-channel state object. The 4 stage modules remain as parameter
# containers (they hold the BD↔GE converters and the SHL/SHR FFNs) so the
# compiler bake_fns can install each stage independently — but their own
# ``forward`` methods are not invoked. This mirrors ``FlattenedALUMul`` where
# the composite's forward inlines the BD→GE conversion, the per-stage FFNs,
# and the GE→BD conversion in a single sequential pass.
# ---------------------------------------------------------------------------


class ShiftBDToGEStage(nn.Module):
    """Stage 1: BD → GenericE format conversion (formerly ALUShift step 1).

    Holds the ``BDToGEConverter`` used by ``ALUShiftComposite.forward``. The
    composite's forward reads ``self.bd_to_ge`` directly; this stage's
    ``forward`` is retained as a thin shim for ad-hoc / unit-test use.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.bd_to_ge = BDToGEConverter(BD, self.ge)

    def forward(self, x_bd):
        # Returns the GE-format workspace flattened to [B*seq, 8, DIM].
        x_ge = self.bd_to_ge(x_bd)
        B, seq_len, _, _ = x_ge.shape
        return x_ge.view(B * seq_len, 8, self.ge.DIM)


class ShiftPrecomputeStage(nn.Module):
    """Stage 2: SHL/SHR sub-chunk precompute (formerly ALUShift step 2a).

    Holds ``ShlPrecomputeFFN`` and ``ShrPrecomputeFFN``. The composite's
    forward reads ``self.shl_precompute`` / ``self.shr_precompute`` directly.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        # Use the same factory as ALUShift to get [precompute, select] pairs;
        # take only the precompute (index 0).
        self.shl_precompute = build_shl_layers(NIBBLE, opcode=23)[0]
        self.shr_precompute = build_shr_layers(NIBBLE, opcode=24)[0]

    def forward(self, x_ge_flat):
        # Returns (x_shl, x_shr) — both with shape [B*seq, 8, DIM].
        x_shl = self.shl_precompute(x_ge_flat.clone())
        x_shr = self.shr_precompute(x_ge_flat.clone())
        return x_shl, x_shr


class ShiftSelectStage(nn.Module):
    """Stage 3: SHL/SHR select + opcode-gated combine (formerly ALUShift step 2b).

    Holds the SHL/SHR select FFNs. The composite's forward reads
    ``self.shl_select`` / ``self.shr_select`` directly.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        # Select FFN is index 1 of build_*_layers.
        self.shl_select = build_shl_layers(NIBBLE, opcode=23)[1]
        self.shr_select = build_shr_layers(NIBBLE, opcode=24)[1]

    def forward(self, x_shl, x_shr):
        # Returns (x_shl_post, x_shr_post).
        return self.shl_select(x_shl), self.shr_select(x_shr)


class ShiftGEToBDStage(nn.Module):
    """Stage 4: GenericE → BD format conversion + write OUTPUT (formerly ALUShift step 3).

    Holds the ``GEToBDConverter``. The composite's forward reads
    ``self.ge_to_bd`` directly.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.ge_to_bd = GEToBDConverter(BD, self.ge, S)

    def forward(self, x_ge_out, x_bd, opcode_mask):
        return self.ge_to_bd(
            x_ge_out,
            x_bd,
            opcode_mask=opcode_mask,
            emit_carry=False,
        )


class ALUShiftComposite(nn.Module):
    """Composite SHL/SHR FFN replacement.

    Replaces ``ALUShift`` (an instance of ``PureNeuralALU(operations='shift')``)
    as the L13 ``block.ffn`` module. Forward is byte-identical to
    ``ALUShift.forward``.

    The 4 sub-stages are exposed as named submodules so that the compiler
    bake_fns in ``migrated_ops.make_l13_alu_shift_*_op`` can build / inspect
    each stage independently. The forward inlines the pipeline (no
    side-channel state) — same shape as ``FlattenedALUMul.forward``.
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.ge = GenericE(NIBBLE)
        self.bdtoge_stage = ShiftBDToGEStage(S, BD)
        self.precompute_stage = ShiftPrecomputeStage(S, BD)
        self.select_stage = ShiftSelectStage(S, BD)
        self.getobd_stage = ShiftGEToBDStage(S, BD)

    def forward(self, x_bd):
        BD = self.BD
        B, seq_len, _ = x_bd.shape

        # Stage 1: BD → GE.
        x_ge_flat = self.bdtoge_stage.bd_to_ge(x_bd)
        x_ge_flat = x_ge_flat.view(B * seq_len, 8, self.ge.DIM)

        # Stage 2: SHL/SHR precompute on the GE workspace.
        x_shl = self.precompute_stage.shl_precompute(x_ge_flat.clone())
        x_shr = self.precompute_stage.shr_precompute(x_ge_flat.clone())

        # Stage 3: SHL/SHR select + opcode-gated combine.
        x_shl_post = self.select_stage.shl_select(x_shl)
        x_shr_post = self.select_stage.shr_select(x_shr)

        x_ge_out = x_ge_flat.clone()

        op_shl = (x_ge_flat[:, 0, self.ge.OP_START + 23] > 0.1).float()
        op_shr = (x_ge_flat[:, 0, self.ge.OP_START + 24] > 0.1).float()
        op_total = op_shl + op_shr

        x_ge_out[:, :, self.ge.RESULT] = (
            x_shl_post[:, :, self.ge.RESULT] * op_shl[:, None]
            + x_shr_post[:, :, self.ge.RESULT] * op_shr[:, None]
        )

        x_ge_out = x_ge_out.view(B, seq_len, 8, self.ge.DIM)
        opcode_mask = op_total.view(B, seq_len)

        # Restrict OUTPUT writes to AX marker positions to match ALUShift.
        mark_ax = x_bd[:, :, BD.MARK_AX]
        opcode_mask = opcode_mask * (mark_ax > 0.5).float()

        # Stage 4: GE → BD + write OUTPUT.
        return self.getobd_stage.ge_to_bd(x_ge_out, x_bd, opcode_mask=opcode_mask)

    # Stub methods for compatibility with vm_step.py (mirror PureNeuralALU).
    def compact(self, block_size=1):
        pass

    def sparsify(self):
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        pass


# NOTE (2026-07-13, corrector-removal roadmap item #14, -77 LOC):
# ``ShiftOutputClearFFN`` was DELETED here. It was a param-free forward wrapper
# that multiplicatively ZEROED the OUTPUT_LO/HI band on the OP_SHL/OP_SHR +
# MARK_AX row to remove a stale ``OUTPUT_{LO,HI}+0 = 2.0`` zero-default leaked
# onto the SHR compute row by the L11 emission head. That leak is now cancelled
# CORRECT-BY-CONSTRUCTION at its L11 source by ``l11_ops.make_output_b0_noleak_op``
# (``C4_OUTPUT_B0_NOLEAK``, DEFAULT-ON), so the composite's OUTPUT band is already
# empty on the SHR row -> the old wrap's ``forward == inner`` (inertness proof
# max-diff ~2.4e-7, ``tools/probe_output_b0_noleak_verify.py``) -> it was a true
# no-op and is removed. See ``docs/OUTPUT_BYTE0_LEAK_ROOT_2026_07_13.md``.


class ALUDivMod(PureNeuralALU):
    """Neural DIV/MOD."""
    def __init__(self, S, BD):
        super().__init__(S, BD, operations='div_mod')
