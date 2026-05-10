"""
Code Generator for Neural VM Compiler.

Generates transformer weights from CompilerIR.
This is the "backend" of the compiler - it translates IR operations
into concrete weight assignments.
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .ir import CompilerIR, LayerSpec, AttentionOp, FFNOp, AttentionOpType, FFNOpType, DimensionAlloc
from .primitives import P


@dataclass
class CodeGenConfig:
    """Configuration for code generation."""
    scale: float = 100.0       # SwiGLU scale factor
    head_dim: int = 64         # Attention head dimension
    alibi_slope: float = 15.0  # ALiBi slope for thresholds
    threshold_base: float = 2.5  # Base threshold for SwiGLU gates


class CodeGenerator:
    """Generates weights from CompilerIR.

    Takes a complete IR and emits weights to a model.
    Each IR operation type has a corresponding emit function.
    """

    def __init__(self, config: Optional[CodeGenConfig] = None):
        self.config = config or CodeGenConfig()

    @torch.no_grad()
    def generate(self, ir: CompilerIR, model) -> List[str]:
        """Generate all weights from IR.

        Args:
            ir: The compiled intermediate representation
            model: Target model to write weights to

        Returns:
            List of warnings/info messages
        """
        messages = []

        # Verify IR
        errors = ir.check_dimension_collisions()
        errors += ir.check_read_before_write()
        if errors:
            for e in errors:
                messages.append(f"WARNING: {e}")

        # Generate embedding weights
        self._generate_embedding(ir, model)

        # Generate layer weights
        for layer_spec in ir.layers:
            self._generate_layer(ir, layer_spec, model)

        return messages

    def _generate_embedding(self, ir: CompilerIR, model) -> None:
        """Generate embedding weights."""
        embed = model.embed.embed.weight
        # Embedding is typically set by the existing set_vm_weights
        # We preserve this for now

    def _generate_layer(self, ir: CompilerIR, layer_spec: LayerSpec, model) -> None:
        """Generate weights for a single layer."""
        block = model.blocks[layer_spec.layer_idx]

        # Generate attention weights
        for attn_op in layer_spec.attention_ops:
            self._emit_attention_op(ir, block.attn, attn_op)

        # Generate FFN weights
        for ffn_op in layer_spec.ffn_ops:
            self._emit_ffn_op(ir, block.ffn, ffn_op)

    def _emit_attention_op(self, ir: CompilerIR, attn, op: AttentionOp) -> None:
        """Emit weights for an attention operation."""
        HD = self.config.head_dim
        base = op.head_idx * HD

        if op.op_type == AttentionOpType.THRESHOLD:
            self._emit_threshold_head(ir, attn, op, base)
        elif op.op_type == AttentionOpType.RELAY:
            self._emit_relay_head(ir, attn, op, base)
        elif op.op_type == AttentionOpType.FETCH:
            self._emit_fetch_head(ir, attn, op, base)
        elif op.op_type == AttentionOpType.GATHER:
            self._emit_gather_head(ir, attn, op, base)
        elif op.op_type == AttentionOpType.DETECT:
            self._emit_detect_head(ir, attn, op, base)
        elif op.op_type == AttentionOpType.CARRY_FORWARD:
            self._emit_carry_forward_head(ir, attn, op, base)
        elif op.op_type == AttentionOpType.OP_FLAG_RELAY:
            self._emit_op_flag_relay_head(ir, attn, op, base)

    def _emit_threshold_head(self, ir: CompilerIR, attn, op: AttentionOp, base: int) -> None:
        """Emit threshold detection head.

        Detects distance from marker using ALiBi-style attention.
        Q fires at CONST with scaled slope, K fires at IS_MARK with threshold.
        V copies marker type flags, O writes to output dimensions.
        """
        params = op.params
        threshold = params.get('threshold', 3.5)
        slope = params.get('slope', self.config.alibi_slope)
        HD = self.config.head_dim

        # Q: sqrt(HD) * slope at CONST
        q_val = 8.0 * slope  # sqrt(64) = 8
        for q_dim in op.q_dims:
            attn.W_q.data[base, q_dim] = q_val

        # K: threshold value at IS_MARK
        for k_dim in op.k_dims:
            attn.W_k.data[base, k_dim] = threshold

        # V: Copy marker type flags (each marker goes to different V subspace)
        for m, v_dim in enumerate(op.v_dims):
            attn.W_v.data[base + 1 + m, v_dim] = 1.0

        # O: Write each marker's result to output dimensions
        for m, o_dim in enumerate(op.o_dims):
            attn.W_o.data[o_dim, base + 1 + m] = 1.0

    def _emit_relay_head(self, ir: CompilerIR, attn, op: AttentionOp, base: int) -> None:
        """Emit relay head that copies values between positions.

        Uses marker matching for Q/K and copies V to O.
        V/O use base+1+k indexing to reserve base+0 for Q/K pattern.

        Params:
            scale: Q/K scale factor
            with_gate: Add anti-leakage gate (default True)
            q_threshold: Q threshold value (for multi-dim Q gating)
            has_se_gate: If True, add HAS_SE-based gating (only fire when HAS_SE > 0)
            not_has_se: If True, only fire on first step (HAS_SE = 0)
            block_at_ax: If True, add -L to Q at MARK_AX to block at AX
            strong_gate: If True, use ±1000 gating instead of standard
            op_jmp_gate: If True, add OP_JMP-based gating
            has_se_block: If True, block when HAS_SE = 1
            v_scale: V output scale factor (default 1.0)
        """
        params = op.params
        scale = params.get('scale', self.config.alibi_slope)
        with_gate = params.get('with_gate', True)
        q_threshold = params.get('q_threshold', None)
        has_se_gate = params.get('has_se_gate', False)
        not_has_se = params.get('not_has_se', False)
        block_at_ax = params.get('block_at_ax', False)
        strong_gate = params.get('strong_gate', False)
        op_jmp_gate = params.get('op_jmp_gate', False)
        has_se_block = params.get('has_se_block', False)
        v_scale = params.get('v_scale', 1.0)

        HD = self.config.head_dim
        CONST_DIM = self._get_const_dim(ir)

        # Q: Query markers
        for q_dim in op.q_dims:
            attn.W_q.data[base, q_dim] = scale

        # Q threshold (for multi-condition Q like MARK_AX + HAS_SE)
        if q_threshold is not None:
            attn.W_q.data[base, CONST_DIM] = q_threshold

        # Block at AX marker
        if block_at_ax:
            ax_dim = ir.get_dim_start("MARK_AX")
            attn.W_q.data[base, ax_dim] = -scale

        # OP_JMP gating
        if op_jmp_gate:
            op_jmp_dim = ir.get_dim_start("OP_JMP")
            attn.W_q.data[base, op_jmp_dim] = scale

        # K: Key markers
        for k_dim in op.k_dims:
            attn.W_k.data[base, k_dim] = scale

        # V: Source dimensions (at base+1+k)
        for k, v_dim in enumerate(op.v_dims):
            if k < HD - 1:  # Leave room for Q/K row
                attn.W_v.data[base + 1 + k, v_dim] = 1.0

        # O: Destination dimensions (from base+1+k) with v_scale
        for k, o_dim in enumerate(op.o_dims):
            if k < HD - 1:
                attn.W_o.data[o_dim, base + 1 + k] = v_scale

        # Anti-leakage gate (standard)
        if with_gate and not strong_gate:
            GATE = 33
            for q_dim in op.q_dims[:1]:  # Use first Q dim for gate
                attn.W_q.data[base + GATE, q_dim] = scale
            attn.W_q.data[base + GATE, CONST_DIM] = -scale / 2
            attn.W_k.data[base + GATE, CONST_DIM] = scale

        # Strong gate (±1000 for L6 control flow)
        if strong_gate:
            GATE = 33
            for q_dim in op.q_dims[:1]:
                attn.W_q.data[base + GATE, q_dim] = 1000.0
            attn.W_q.data[base + GATE, CONST_DIM] = -500.0
            attn.W_k.data[base + GATE, CONST_DIM] = 10.0

        # HAS_SE gating (only fire when HAS_SE > 0, step 1+)
        if has_se_gate:
            HAS_SE_GATE = 34
            has_se_dim = ir.get_dim_start("HAS_SE")
            attn.W_q.data[base + HAS_SE_GATE, has_se_dim] = 1000.0
            attn.W_q.data[base + HAS_SE_GATE, CONST_DIM] = -500.0
            attn.W_k.data[base + HAS_SE_GATE, CONST_DIM] = 10.0

        # NOT HAS_SE gating (only fire when HAS_SE = 0, first step)
        if not_has_se:
            HAS_SE_GATE = 34
            has_se_dim = ir.get_dim_start("HAS_SE")
            attn.W_q.data[base + HAS_SE_GATE, has_se_dim] = -1000.0
            attn.W_k.data[base + HAS_SE_GATE, CONST_DIM] = 10.0

        # HAS_SE block (block when HAS_SE = 1)
        if has_se_block:
            HAS_SE_GATE = 35
            has_se_dim = ir.get_dim_start("HAS_SE")
            attn.W_q.data[base + HAS_SE_GATE, has_se_dim] = -1000.0
            attn.W_k.data[base + HAS_SE_GATE, CONST_DIM] = 10.0

    def _emit_fetch_head(self, ir: CompilerIR, attn, op: AttentionOp, base: int) -> None:
        """Emit fetch head for memory lookup.

        Uses address dimensions to match memory positions.

        Params:
            scale: Q/K scale factor
            has_se_gate: If True, only fire on non-first steps (HAS_SE > 0)
            not_has_se: If True, only fire on first step (HAS_SE = 0)
            fixed_addr: Fixed address value (e.g., PC_OFFSET) instead of dynamic
            out_scale: Output scale (default 1.0, e.g., 40.0 for amplified FETCH)
            anti_leak_marker: Marker for anti-leakage gate
        """
        params = op.params
        L = params.get('scale', self.config.alibi_slope)
        has_se_gate = params.get('has_se_gate', False)
        not_has_se = params.get('not_has_se', False)
        fixed_addr = params.get('fixed_addr', None)
        out_scale = params.get('out_scale', 1.0)
        anti_leak_marker = params.get('anti_leak_marker', None)

        HD = self.config.head_dim
        CONST_DIM = self._get_const_dim(ir)

        # Q: Address from specified dimensions
        if fixed_addr is not None:
            # Fixed address query
            lo_nibble = fixed_addr & 0xF
            hi_nibble = (fixed_addr >> 4) & 0xF
            attn.W_q.data[base + lo_nibble, CONST_DIM] = L  # lo nibble
            attn.W_q.data[base + 16 + hi_nibble, CONST_DIM] = L  # hi nibble
            # Marker gate
            for q_dim in op.q_dims:
                attn.W_q.data[base, q_dim] = L
                if not_has_se:
                    has_se_dim = ir.get_dim_start("HAS_SE")
                    attn.W_q.data[base, has_se_dim] = -L
                attn.W_q.data[base + 32, q_dim] = L  # third nibble gate
        else:
            # Dynamic address query from Q dims
            for i, q_dim in enumerate(op.q_dims):
                if i < 33:  # First 33 dims are address nibbles + marker
                    attn.W_q.data[base + i, q_dim] = L

        # K: Address key at memory positions
        for i, k_dim in enumerate(op.k_dims):
            if i < 33:
                attn.W_k.data[base + i, k_dim] = L

        # Anti-leakage gate
        if anti_leak_marker:
            GATE = 33
            marker_dim = ir.get_dim_start(anti_leak_marker)
            attn.W_q.data[base + GATE, marker_dim] = 500.0
            attn.W_q.data[base + GATE, CONST_DIM] = -500.0
            attn.W_k.data[base + GATE, CONST_DIM] = 5.0

        # HAS_SE gating
        if has_se_gate:
            HAS_SE_GATE = 34
            has_se_dim = ir.get_dim_start("HAS_SE")
            attn.W_q.data[base + HAS_SE_GATE, has_se_dim] = 500.0
            attn.W_q.data[base + HAS_SE_GATE, CONST_DIM] = -500.0
            attn.W_k.data[base + HAS_SE_GATE, CONST_DIM] = 5.0
        elif not_has_se and fixed_addr is None:
            # NOT HAS_SE gating for dynamic fetch
            HAS_SE_GATE = 34
            has_se_dim = ir.get_dim_start("HAS_SE")
            attn.W_q.data[base + HAS_SE_GATE, has_se_dim] = -500.0
            attn.W_k.data[base + HAS_SE_GATE, CONST_DIM] = 5.0

        # V: Copy value nibbles (using CLEAN_EMBED for exact values)
        # V starts at base+32 to avoid Q/K space
        for i, v_dim in enumerate(op.v_dims):
            if i < 32:  # 16 lo + 16 hi
                attn.W_v.data[base + 32 + i, v_dim] = 1.0

        # O: Output dimensions
        for i, o_dim in enumerate(op.o_dims):
            if i < 32:
                attn.W_o.data[o_dim, base + 32 + i] = out_scale

    def _emit_op_flag_relay_head(self, ir: CompilerIR, attn, op: AttentionOp, base: int) -> None:
        """Emit OP flag relay head.

        Relays OP_* flags from CODE section to markers.

        Params:
            scale: Q/K scale factor
            has_se_gate: If True, only fire on non-first steps
            not_has_se: If True, only fire on first step
            fixed_addr: Fixed address value
            anti_leak_marker: Marker for anti-leakage gate
        """
        params = op.params
        L = params.get('scale', 15.0)
        has_se_gate = params.get('has_se_gate', False)
        not_has_se = params.get('not_has_se', False)
        fixed_addr = params.get('fixed_addr', None)
        anti_leak_marker = params.get('anti_leak_marker', None)

        HD = self.config.head_dim
        CONST_DIM = self._get_const_dim(ir)

        # Q: Address query
        if fixed_addr is not None:
            # Fixed address
            lo_nibble = fixed_addr & 0xF
            hi_nibble = (fixed_addr >> 4) & 0xF
            attn.W_q.data[base + lo_nibble, CONST_DIM] = L
            attn.W_q.data[base + 16 + hi_nibble, CONST_DIM] = L
            # Marker gate
            for q_dim in op.q_dims:
                attn.W_q.data[base, q_dim] = L
                if not_has_se:
                    has_se_dim = ir.get_dim_start("HAS_SE")
                    attn.W_q.data[base, has_se_dim] = -L
                attn.W_q.data[base + 32, q_dim] = L
        else:
            # Dynamic address from Q dims
            for i, q_dim in enumerate(op.q_dims):
                if i < 33:
                    attn.W_q.data[base + i, q_dim] = L

        # K: Address key
        for i, k_dim in enumerate(op.k_dims):
            if i < 33:
                attn.W_k.data[base + i, k_dim] = L

        # Anti-leakage gate
        if anti_leak_marker:
            GATE = 33
            marker_dim = ir.get_dim_start(anti_leak_marker)
            attn.W_q.data[base + GATE, marker_dim] = 500.0
            attn.W_q.data[base + GATE, CONST_DIM] = -500.0
            attn.W_k.data[base + GATE, CONST_DIM] = 5.0

        # HAS_SE gating
        if has_se_gate:
            HAS_SE_GATE = 34
            has_se_dim = ir.get_dim_start("HAS_SE")
            attn.W_q.data[base + HAS_SE_GATE, has_se_dim] = 500.0
            attn.W_q.data[base + HAS_SE_GATE, CONST_DIM] = -500.0
            attn.W_k.data[base + HAS_SE_GATE, CONST_DIM] = 5.0

        # V: Copy OP_* flags
        for i, v_dim in enumerate(op.v_dims):
            attn.W_v.data[base + i, v_dim] = 1.0

        # O: Write OP_* flags to output
        for i, o_dim in enumerate(op.o_dims):
            attn.W_o.data[o_dim, base + i] = 1.0

    def _emit_gather_head(self, ir: CompilerIR, attn, op: AttentionOp, base: int) -> None:
        """Emit gather head for collecting operands."""
        # Similar to relay but with different gating
        self._emit_relay_head(ir, attn, op, base)

    def _emit_detect_head(self, ir: CompilerIR, attn, op: AttentionOp, base: int) -> None:
        """Emit detection head (e.g., HAS_SE detection).

        Pattern:
        Q[base] = scale at q_dims
        K[base] = scale at k_dims
        V[base+1] = 1.0 at v_dims (copy detected signal)
        O[o_dim] = scale from base+1
        """
        params = op.params
        scale = params.get('scale', self.config.alibi_slope)

        # Q/K pattern for detection
        for q_dim in op.q_dims:
            attn.W_q.data[base, q_dim] = scale

        for k_dim in op.k_dims:
            attn.W_k.data[base, k_dim] = scale

        # V: Copy detected signal to head subspace at base+1
        for v_dim in op.v_dims:
            attn.W_v.data[base + 1, v_dim] = 1.0

        # O: Output flag from base+1
        for o_dim in op.o_dims:
            attn.W_o.data[o_dim, base + 1] = params.get('output_scale', 1.0)

    def _emit_carry_forward_head(self, ir: CompilerIR, attn, op: AttentionOp, base: int) -> None:
        """Emit register carry-forward head.

        Pattern:
        Q: L * marker_dim (fire at register marker)
        K: L * L1H1[idx] - L * L1H0[idx] (attend to prev step's byte 0)
        V: Copy src_lo/src_hi nibbles
        O: Write to dest_lo/dest_hi (unless with_output=False)

        Params:
            marker_idx: Index into marker array (0=PC, 1=AX, 2=SP, 3=BP)
            l1h1_idx, l1h0_idx: Indices into L1H1/L1H0 for key pattern
            with_gate: Whether to add anti-leakage gate (default True)
            extra_q_dim: Additional Q dimension (e.g., OP_LEV for conditional carry)
            extra_q_scale: Scale for extra Q dim
            q_threshold: Q threshold for multi-condition gating
            with_output: Whether to write to O (default True)
        """
        params = op.params
        L = params.get('scale', 15.0)
        marker_idx = params.get('marker_idx', 0)
        l1h1_idx = params.get('l1h1_idx', marker_idx)
        l1h0_idx = params.get('l1h0_idx', marker_idx)
        with_gate = params.get('with_gate', True)
        extra_q_dim = params.get('extra_q_dim', None)
        extra_q_scale = params.get('extra_q_scale', L)
        q_threshold = params.get('q_threshold', None)
        with_output = params.get('with_output', True)

        HD = self.config.head_dim

        # Q: Fire at marker
        for q_dim in op.q_dims:
            attn.W_q.data[base, q_dim] = L

        # Extra Q dimension (e.g., OP_LEV for conditional carry)
        if extra_q_dim is not None:
            attn.W_q.data[base, extra_q_dim] = extra_q_scale

        # Q threshold for multi-condition gating
        if q_threshold is not None:
            attn.W_q.data[base, self._get_const_dim(ir)] = q_threshold

        # K: L1H1[idx] - L1H0[idx] pattern
        for k_dim in op.k_dims[:1]:  # First k_dim is L1H1
            attn.W_k.data[base, k_dim] = L
        for k_dim in op.k_dims[1:2]:  # Second k_dim is L1H0
            attn.W_k.data[base, k_dim] = -L

        # V: Copy source nibbles to head subspace
        for k, v_dim in enumerate(op.v_dims):
            attn.W_v.data[base + 1 + k, v_dim] = 1.0

        # O: Write to destination dims (unless disabled)
        if with_output:
            for k, o_dim in enumerate(op.o_dims):
                attn.W_o.data[o_dim, base + 1 + k] = 1.0

        # Anti-leakage gate
        if with_gate:
            GATE = 33
            for q_dim in op.q_dims:
                attn.W_q.data[base + GATE, q_dim] = L
            attn.W_q.data[base + GATE, self._get_const_dim(ir)] = -L / 2
            attn.W_k.data[base + GATE, self._get_const_dim(ir)] = L

    def _get_const_dim(self, ir: CompilerIR) -> int:
        """Get CONST dimension index."""
        return ir.dimensions.get("CONST", DimensionAlloc("CONST", 8, 1)).start

    def _emit_ffn_op(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit weights for an FFN operation."""
        if op.op_type == FFNOpType.SWIGLU_GATE:
            self._emit_swiglu_gate(ir, ffn, op)
        elif op.op_type == FFNOpType.ALU_LOOKUP:
            self._emit_alu_lookup(ir, ffn, op)
        elif op.op_type == FFNOpType.NIBBLE_ROTATE:
            self._emit_nibble_rotate(ir, ffn, op)
        elif op.op_type == FFNOpType.OPCODE_DECODE:
            self._emit_opcode_decode(ir, ffn, op)
        elif op.op_type == FFNOpType.THRESHOLD_FLAG:
            self._emit_threshold_flag(ir, ffn, op)
        elif op.op_type == FFNOpType.CLEAR_DIMS:
            self._emit_clear_dims(ir, ffn, op)
        elif op.op_type == FFNOpType.COPY:
            self._emit_copy(ir, ffn, op)
        elif op.op_type == FFNOpType.CONDITIONAL_OUTPUT:
            self._emit_conditional_output(ir, ffn, op)
        elif op.op_type == FFNOpType.GATED_NIBBLE_COPY:
            self._emit_gated_nibble_copy(ir, ffn, op)
        elif op.op_type == FFNOpType.PC_INCREMENT:
            self._emit_pc_increment(ir, ffn, op)

    def _emit_swiglu_gate(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit SwiGLU gated unit.

        The unit fires when sum of gate dimensions exceeds threshold.
        """
        params = op.params
        S = params.get('scale', self.config.scale)
        threshold = params.get('threshold', self.config.threshold_base)
        output_value = params.get('output_value', 1.0)

        unit = op.unit_start

        # Gate: sum of gate dimensions must exceed threshold
        for gate_dim in op.gate_dims:
            weight = params.get('gate_weight', 1.0)
            ffn.W_gate.data[unit, gate_dim] = S * weight

        # Bias for threshold
        if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
            ffn.b_gate.data[unit] = -S * threshold

        # Up path (what to output)
        for in_dim in op.input_dims:
            ffn.W_up.data[unit, in_dim] = S

        # Down path (where to output)
        for out_dim in op.output_dims:
            ffn.W_down.data[out_dim, unit] = output_value

    def _emit_alu_lookup(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit ALU lookup table.

        Creates a 256-entry lookup for binary operations.
        """
        params = op.params
        S = params.get('scale', self.config.scale)
        alu_fn = params.get('alu_fn')  # (a, b) -> result

        if alu_fn is None:
            return

        unit_base = op.unit_start

        # Input nibbles
        a_dims = op.input_dims[:16]  # First operand nibbles
        b_dims = op.input_dims[16:32]  # Second operand nibbles

        # Output nibbles
        out_dims = op.output_dims[:16]

        # Gate dimensions
        gate_dims = op.gate_dims

        # Create 256 units for all (a, b) combinations
        for a in range(16):
            for b in range(16):
                unit = unit_base + a * 16 + b
                result = alu_fn(a, b) & 0xF

                # Gate: opcode flag + input nibbles
                for gate_dim in gate_dims:
                    ffn.W_gate.data[unit, gate_dim] = S

                # Match input a
                if a_dims:
                    ffn.W_gate.data[unit, a_dims[a]] = S

                # Match input b
                if b_dims:
                    ffn.W_gate.data[unit, b_dims[b]] = S

                # Threshold
                if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
                    ffn.b_gate.data[unit] = -S * 2.5

                # Up: constant
                ffn.W_up.data[unit, ir.get_dim_start("CONST")] = S

                # Down: output result nibble
                if out_dims:
                    ffn.W_down.data[out_dims[result], unit] = 1.0

    def _emit_nibble_rotate(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit nibble rotation (increment)."""
        params = op.params
        S = params.get('scale', self.config.scale)

        # This creates the (n+1) % 16 pattern
        unit_base = op.unit_start

        in_dims = op.input_dims[:16]
        out_dims = op.output_dims[:16]
        gate_dims = op.gate_dims

        for n in range(16):
            unit = unit_base + n
            result = (n + 1) % 16

            # Gate
            for gate_dim in gate_dims:
                ffn.W_gate.data[unit, gate_dim] = S
            if in_dims:
                ffn.W_gate.data[unit, in_dims[n]] = S

            # Threshold
            if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
                ffn.b_gate.data[unit] = -S * 1.5

            # Up
            ffn.W_up.data[unit, ir.get_dim_start("CONST")] = S

            # Down
            if out_dims:
                ffn.W_down.data[out_dims[result], unit] = 1.0

    def _emit_opcode_decode(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit opcode decode unit.

        Converts opcode nibbles to flag dimensions.

        Standard pattern (at AX marker):
            W_up[OPCODE_BYTE_LO[lo]] = S
            W_up[OPCODE_BYTE_HI[hi]] = S
            b_up = -S * 1.5
            W_gate[MARK_AX] = 1.0
            W_down[OP_flag] = out_scale / S

        First-step pattern (at PC marker):
            W_up[OPCODE_BYTE_LO[lo]] = S
            W_up[OPCODE_BYTE_HI[hi]] = S
            W_up[MARK_PC] = S
            W_up[HAS_SE] = -S
            b_up = -S * 2.5
            b_gate = 1.0
            W_down[OP_flag] = out_scale / S
        """
        params = op.params
        S = params.get('scale', self.config.scale)
        lo_nibble = params.get('lo_nibble', 0)
        hi_nibble = params.get('hi_nibble', 0)
        out_scale = params.get('out_scale', 1.0)
        pc_marker_gate = params.get('pc_marker_gate', False)
        not_has_se = params.get('not_has_se', False)

        unit = op.unit_start

        # The input_dims should contain OPCODE_BYTE nibble dimensions
        # First 8 are lo nibble one-hot (8 dims), next 8 are hi nibble one-hot (8 dims)
        opcode_lo = op.input_dims[:8] if len(op.input_dims) >= 8 else op.input_dims
        opcode_hi = op.input_dims[8:16] if len(op.input_dims) >= 16 else []

        if pc_marker_gate:
            # First-step decode at PC marker using W_up gating
            # W_up: nibble match + MARK_PC + (-HAS_SE)
            if opcode_lo and lo_nibble < len(opcode_lo):
                ffn.W_up.data[unit, opcode_lo[lo_nibble]] = S
            if opcode_hi and hi_nibble < len(opcode_hi):
                ffn.W_up.data[unit, opcode_hi[hi_nibble]] = S
            ffn.W_up.data[unit, ir.get_dim_start("MARK_PC")] = S
            if not_has_se:
                ffn.W_up.data[unit, ir.get_dim_start("HAS_SE")] = -S
                # Threshold for 3 conditions: LO + HI + MARK_PC - HAS_SE
                if hasattr(ffn, 'b_up') and ffn.b_up is not None:
                    ffn.b_up.data[unit] = -S * 2.5
            else:
                if hasattr(ffn, 'b_up') and ffn.b_up is not None:
                    ffn.b_up.data[unit] = -S * 1.5

            # Constant gate
            if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
                ffn.b_gate.data[unit] = 1.0
        else:
            # Standard decode at AX marker using W_gate pattern
            # W_up: nibble match
            if opcode_lo and lo_nibble < len(opcode_lo):
                ffn.W_up.data[unit, opcode_lo[lo_nibble]] = S
            if opcode_hi and hi_nibble < len(opcode_hi):
                ffn.W_up.data[unit, opcode_hi[hi_nibble]] = S

            # Threshold for 2 nibble conditions
            if hasattr(ffn, 'b_up') and ffn.b_up is not None:
                ffn.b_up.data[unit] = -S * 1.5

            # Gate: marker
            for gate_dim in op.gate_dims:
                ffn.W_gate.data[unit, gate_dim] = 1.0

        # Down: output to flag dimension
        for out_dim in op.output_dims:
            ffn.W_down.data[out_dim, unit] = out_scale / S

    def _emit_threshold_flag(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit threshold-to-flag conversion.

        Baseline pattern:
          up = S * (sum(input_dims)) - S * threshold
          gate = 1.0 - sum(gate_dims)  (if negate_gate)
               or sum(gate_dims) (if not negate_gate)
          down = 2.0/S * unit

        Args:
            input_dims: Dimensions to sum for threshold comparison
            gate_dims: Dimensions for gating (suppression)
            output_dims: Output flag dimensions
            threshold: Threshold value (default 1.5)
            negate_gate: If True, gate = 1 - sum(gate_dims)
        """
        params = op.params
        S = params.get('scale', self.config.scale)
        threshold = params.get('threshold', 1.5)
        negate_gate = params.get('negate_gate', False)
        out_scale = params.get('out_scale', 2.0)  # Default 2.0 for L1-L2, 1.0 for L0

        unit = op.unit_start

        # W_up: sum of input dimensions with scale S
        for in_dim in op.input_dims:
            ffn.W_up.data[unit, in_dim] = S

        # b_up: threshold
        if hasattr(ffn, 'b_up') and ffn.b_up is not None:
            ffn.b_up.data[unit] = -S * threshold

        # W_gate: gating dimensions (suppress when active)
        if negate_gate:
            # gate = 1.0 - sum(gate_dims)
            for gate_dim in op.gate_dims:
                ffn.W_gate.data[unit, gate_dim] = -1.0
            if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
                ffn.b_gate.data[unit] = 1.0
        else:
            # gate = sum(gate_dims)
            for gate_dim in op.gate_dims:
                ffn.W_gate.data[unit, gate_dim] = 1.0

        # W_down: output with scale out_scale/S
        for out_dim in op.output_dims:
            ffn.W_down.data[out_dim, unit] = out_scale / S

    def _emit_clear_dims(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit dimension clearing (set to zero)."""
        params = op.params
        S = params.get('scale', self.config.scale)

        unit = op.unit_start

        # Gate: marker dimension
        for gate_dim in op.gate_dims:
            ffn.W_gate.data[unit, gate_dim] = S

        # Threshold
        if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
            ffn.b_gate.data[unit] = -S * 0.5

        # Up: negative of current value
        for in_dim in op.input_dims:
            ffn.W_up.data[unit, in_dim] = -S

        # Down: same dimensions
        for out_dim in op.output_dims:
            ffn.W_down.data[out_dim, unit] = 1.0

    def _emit_copy(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit dimension copy under gate."""
        params = op.params
        S = params.get('scale', self.config.scale)

        unit = op.unit_start

        # Gate
        for gate_dim in op.gate_dims:
            ffn.W_gate.data[unit, gate_dim] = S

        # Threshold
        if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
            ffn.b_gate.data[unit] = -S * 0.5

        # Up: source dimensions
        for in_dim in op.input_dims:
            ffn.W_up.data[unit, in_dim] = S

        # Down: destination dimensions
        for i, out_dim in enumerate(op.output_dims):
            if i < len(op.input_dims):
                ffn.W_down.data[out_dim, unit] = 1.0

    def _emit_conditional_output(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit conditional output unit.

        Outputs a fixed nibble value when conditions in W_up exceed threshold.
        Uses constant gate (b_gate=1.0).

        Pattern from L3 FFN:
            W_up[unit, condition1] = S
            W_up[unit, condition2] = S  (optional)
            W_up[unit, neg_cond] = -S   (optional suppression)
            b_up[unit] = -S * threshold
            b_gate[unit] = 1.0
            W_down[OUTPUT + nibble, unit] = 2.0 / S

        Params:
            threshold: W_up threshold (default 0.5 for single, 1.5 for AND)
            nibble_lo: Lo nibble value to output
            nibble_hi: Hi nibble value to output (optional)
            neg_dims: Dimensions with negative weight (suppression)
            neg_scale: Scale for negative dims (default S)
            also_write: Additional output dims (e.g., EMBED for dual write)
        """
        params = op.params
        S = params.get('scale', self.config.scale)
        threshold = params.get('threshold', 0.5)
        nibble_lo = params.get('nibble_lo', 0)
        nibble_hi = params.get('nibble_hi', None)
        neg_dims = params.get('neg_dims', [])
        neg_scale = params.get('neg_scale', S)
        also_write = params.get('also_write', [])
        out_scale = params.get('out_scale', 2.0)

        unit = op.unit_start

        # W_up: positive conditions
        for in_dim in op.input_dims:
            ffn.W_up.data[unit, in_dim] = S

        # W_up: negative conditions (suppression)
        for neg_dim in neg_dims:
            ffn.W_up.data[unit, neg_dim] = -neg_scale

        # b_up: threshold
        if hasattr(ffn, 'b_up') and ffn.b_up is not None:
            ffn.b_up.data[unit] = -S * threshold

        # b_gate: constant 1.0 (always fires when up > 0)
        if hasattr(ffn, 'b_gate') and ffn.b_gate is not None:
            ffn.b_gate.data[unit] = 1.0

        # W_down: output to nibble positions
        # First half of output_dims is OUTPUT_LO, second half is OUTPUT_HI
        n_out = len(op.output_dims)
        out_lo = op.output_dims[:n_out // 2] if n_out > 16 else op.output_dims
        out_hi = op.output_dims[n_out // 2:] if n_out > 16 else []

        if out_lo and nibble_lo < len(out_lo):
            ffn.W_down.data[out_lo[nibble_lo], unit] = out_scale / S

        if out_hi and nibble_hi is not None and nibble_hi < len(out_hi):
            ffn.W_down.data[out_hi[nibble_hi], unit] = out_scale / S

        # Also write to additional dims (e.g., EMBED)
        for i, also_dim in enumerate(also_write):
            ffn.W_down.data[also_dim, unit] = out_scale / S

    def _emit_gated_nibble_copy(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit gated nibble copy (16 units).

        For each output nibble k (0-15), fires when gate condition + input nibble match,
        outputs to output nibble k. Input is rotated by -rotate relative to output.

        Pattern (matching baseline):
            W_up[unit, gate_dims...] = S
            b_up[unit] = -S * threshold
            W_gate[unit, INPUT + (k-rotate)%16] = gate_weight
            W_down[OUTPUT + k, unit] = 2.0 / S

        Effect: OUTPUT[k] = INPUT[(k-rotate)%16]
        For rotate=1: OUTPUT[k] = INPUT[(k-1)%16], i.e., OUTPUT = INPUT shifted right by 1

        Params:
            threshold: Gate threshold (default 0.5 for single condition)
            gate_weight: Weight for nibble gate (default 1.0, use -1.0 to cancel)
            rotate: Input rotation (default 0, use 1 for OUTPUT[k] = INPUT[(k-1)%16])
            out_scale: Output scale (default 2.0)
        """
        params = op.params
        S = params.get('scale', self.config.scale)
        threshold = params.get('threshold', 0.5)
        gate_weight = params.get('gate_weight', 1.0)
        rotate = params.get('rotate', 0)
        out_scale = params.get('out_scale', 2.0)
        neg_dims = params.get('neg_dims', [])
        neg_scale = params.get('neg_scale', S)

        in_dims = op.input_dims[:16]
        out_dims = op.output_dims[:16]
        unit_base = op.unit_start

        for k in range(16):
            unit = unit_base + k
            in_k = (k - rotate) % 16  # Input rotation (matching baseline)

            # W_up: gate conditions
            for gate_dim in op.gate_dims:
                ffn.W_up.data[unit, gate_dim] = S

            # W_up: negative conditions (suppression)
            for neg_dim in neg_dims:
                ffn.W_up.data[unit, neg_dim] = -neg_scale

            # b_up: threshold
            if hasattr(ffn, 'b_up') and ffn.b_up is not None:
                ffn.b_up.data[unit] = -S * threshold

            # W_gate: input nibble matching (rotated)
            if in_dims and in_k < len(in_dims):
                ffn.W_gate.data[unit, in_dims[in_k]] = gate_weight

            # W_down: output nibble (no rotation, output index = k)
            if out_dims and k < len(out_dims):
                ffn.W_down.data[out_dims[k], unit] = out_scale / S

    def _emit_pc_increment(self, ir: CompilerIR, ffn, op: FFNOp) -> None:
        """Emit PC increment logic with carry.

        Implements PC += INSTR_WIDTH with carry propagation.

        Params:
            instr_width: Increment amount (default 8)
            marker_dim: PC marker dimension for gating
            has_se_dim: HAS_SE dimension for step filtering
            op_lev_dim: OP_LEV dimension for LEV suppression
        """
        params = op.params
        S = params.get('scale', self.config.scale)
        INSTR_WIDTH = params.get('instr_width', 8)
        marker_dim = params.get('marker_dim')
        has_se_dim = params.get('has_se_dim')
        op_lev_dim = params.get('op_lev_dim')

        embed_lo = op.input_dims[:16]
        embed_hi = op.input_dims[16:32]
        out_lo = op.output_dims[:16]
        out_hi = op.output_dims[16:32]

        unit_base = op.unit_start

        # Lo nibble: (k + INSTR_WIDTH) % 16
        for k in range(16):
            unit = unit_base + k
            new_k = (k + INSTR_WIDTH) % 16

            # Conditions
            if has_se_dim is not None:
                ffn.W_up.data[unit, has_se_dim] = S
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            if op_lev_dim is not None:
                ffn.W_up.data[unit, op_lev_dim] = -S / 5  # OP_LEV ≈ 5.0

            ffn.b_up.data[unit] = -S * 1.5

            # Gate: input nibble
            if embed_lo and k < len(embed_lo):
                ffn.W_gate.data[unit, embed_lo[k]] = 1.0

            # Output
            if out_lo and new_k < len(out_lo):
                ffn.W_down.data[out_lo[new_k], unit] = 2.0 / S

        # Hi nibble: copy (no carry for simple increment)
        for k in range(16):
            unit = unit_base + 16 + k

            # Conditions
            if has_se_dim is not None:
                ffn.W_up.data[unit, has_se_dim] = S
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            if op_lev_dim is not None:
                ffn.W_up.data[unit, op_lev_dim] = -S / 5

            ffn.b_up.data[unit] = -S * 1.5

            # Gate: hi nibble
            if embed_hi and k < len(embed_hi):
                ffn.W_gate.data[unit, embed_hi[k]] = 1.0

            # Output (copy)
            if out_hi and k < len(out_hi):
                ffn.W_down.data[out_hi[k], unit] = 2.0 / S

        # Carry correction: when lo >= (16 - INSTR_WIDTH), hi increments
        carry_threshold = 16 - INSTR_WIDTH
        for k in range(16):
            unit = unit_base + 32 + k

            # Strong marker condition
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = 4 * S
            if has_se_dim is not None:
                ffn.W_up.data[unit, has_se_dim] = S
            if op_lev_dim is not None:
                ffn.W_up.data[unit, op_lev_dim] = -S

            # Carry bits: any lo nibble >= carry_threshold
            for lo_bit in range(carry_threshold, 16):
                if embed_lo and lo_bit < len(embed_lo):
                    ffn.W_up.data[unit, embed_lo[lo_bit]] = S

            ffn.b_up.data[unit] = -S * 5.5

            # Gate: hi nibble
            if embed_hi and k < len(embed_hi):
                ffn.W_gate.data[unit, embed_hi[k]] = 1.0

            # Output: cancel old, add shifted
            if out_hi and k < len(out_hi):
                ffn.W_down.data[out_hi[k], unit] = -2.0 / S
            if out_hi and (k + 1) % 16 < len(out_hi):
                ffn.W_down.data[out_hi[(k + 1) % 16], unit] = 2.0 / S
