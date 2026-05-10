"""
Core primitives for weight compilation.

Implements the 10 fundamental patterns used throughout vm_step.py:

Attention Primitives:
1. threshold_attention - Q=constant, K=IS_MARK×threshold, V/O=marker copy
2. carry_forward_attention - Q=marker, K=distance pattern, V/O=nibble relay
3. memory_lookup_attention - Q=address nibbles, K=ADDR_KEY, V=values
4. relay_head - Q=target marker, K=source threshold, V=register, broadcast O

FFN Primitives:
5. swiglu_and_gate - W_up conditions + b_up threshold, W_gate source, W_down output
6. cancel_pair - W_gate -1×old + 1×new in same unit
7. opcode_decode_unit - W_up nibble AND, W_gate marker, W_down to OP_* flag
8. nibble_copy - W_gate source, W_down output with optional residual cancel
9. step_pair - step(>=low) - step(>=high) for binary decisions
10. threshold_match - W_up marker+conditions, b_up threshold, W_gate, W_down output

Extracted batch B (vm_step direct ports — byte-identical to imperative code):
- nibble_rotation_chain       : (source + offset) FFN block, optionally carry-aware
"""

import torch
from typing import List, Optional, Tuple, Union
from ..vm_step import _SetDim as BD


class Primitives:
    """Core weight-setting primitives matching vm_step.py patterns."""

    # =========================================================================
    # Attention Primitives
    # =========================================================================

    @staticmethod
    def threshold_attention(
        attn,
        head_idx: int,
        threshold: float,
        out_base: int,
        slope: float = 10.0,
        HD: int = 64,
    ):
        """Set threshold-based attention head for marker distance detection.

        Each head detects whether the nearest marker is within `threshold` tokens.
        Uses ALiBi: score = slope*(threshold - distance), giving a sharp sigmoid.

        Pattern from _set_threshold_attn (vm_step.py:2142-2161):
          Q[0] = constant (8.0 * slope)
          K[0] = IS_MARK * threshold
          V[1+m] = MARKS[m] for each marker type
          O[out_base+m] = V[1+m]

        Args:
            attn: Attention layer module
            head_idx: Head index (0-7)
            threshold: Distance threshold (e.g., 3.5, 4.5)
            out_base: Base dimension for output (e.g., BD.H0, BD.H1)
            slope: ALiBi slope (default 10.0)
            HD: Head dimension (default 64)
        """
        base = head_idx * HD
        q_val = 8.0 * slope  # sqrt(HD) = 8

        attn.W_q.data[base, BD.CONST] = q_val
        attn.W_k.data[base, BD.IS_MARK] = threshold

        for m, src in enumerate(BD.MARKS):
            attn.W_v.data[base + 1 + m, src] = 1.0
            attn.W_o.data[out_base + m, base + 1 + m] = 1.0

    @staticmethod
    def carry_forward_attention(
        attn,
        head_idx: int,
        marker_dim: int,
        l1h1_idx: int,
        l1h0_idx: int,
        out_lo: int,
        out_hi: int,
        HD: int = 64,
        src_lo: Optional[int] = None,
        src_hi: Optional[int] = None,
        L: float = 15.0,
    ):
        """Set attention head for register carry-forward.

        At marker positions, attends to the previous step's corresponding byte 0
        (identified by L1H1_marker AND NOT L1H0_marker pattern).

        Pattern from _set_carry_forward_attn (vm_step.py:2552-2599):
          Q[0] = marker_dim * L
          K[0] = L1H1 * L - L1H0 * L
          V[1:17] = src_lo nibble, V[17:33] = src_hi nibble
          O[out_lo/hi] = V
          Anti-leakage gate at dim 33

        Args:
            attn: Attention layer module
            head_idx: Head index
            marker_dim: Query marker dimension (e.g., BD.MARK_PC)
            l1h1_idx: L1H1 marker index for key (e.g., 0 for PC)
            l1h0_idx: L1H0 marker index for key
            out_lo: Output dimension for low nibble
            out_hi: Output dimension for high nibble
            HD: Head dimension (default 64)
            src_lo: Source low nibble dim (default EMBED_LO)
            src_hi: Source high nibble dim (default EMBED_HI)
            L: Attention weight scale (default 15.0)
        """
        base = head_idx * HD

        if src_lo is None:
            src_lo = BD.EMBED_LO
        if src_hi is None:
            src_hi = BD.EMBED_HI

        # Q: fires at target marker
        attn.W_q.data[base, marker_dim] = L

        # K: fires at previous step's byte 0 (L1H1 AND NOT L1H0)
        attn.W_k.data[base, BD.L1H1 + l1h1_idx] = L
        attn.W_k.data[base, BD.L1H0 + l1h0_idx] = -L

        # V: copy source nibbles
        for k in range(16):
            attn.W_v.data[base + 1 + k, src_lo + k] = 1.0
            attn.W_v.data[base + 17 + k, src_hi + k] = 1.0

        # O: write to output dimensions
        for k in range(16):
            attn.W_o.data[out_lo + k, base + 1 + k] = 1.0
            attn.W_o.data[out_hi + k, base + 17 + k] = 1.0

        # Anti-leakage gate (dim 33)
        GATE = 33
        attn.W_q.data[base + GATE, marker_dim] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

    @staticmethod
    def memory_lookup_attention(
        attn,
        head_idx: int,
        query_dims: List[int],
        key_dims: List[int],
        value_dims: List[int],
        output_dims: List[int],
        HD: int = 64,
        gate_q_dim: Optional[int] = None,
        gate_k_dim: Optional[int] = None,
        L: float = 15.0,
    ):
        """Set memory lookup attention head.

        Performs content-addressable memory read via nibble matching.

        Args:
            attn: Attention layer module
            head_idx: Head index
            query_dims: List of query dimensions (address nibbles)
            key_dims: List of key dimensions (ADDR_KEY)
            value_dims: List of value dimensions (byte values)
            output_dims: List of output dimensions
            HD: Head dimension (default 64)
            gate_q_dim: Optional gate dimension for Q
            gate_k_dim: Optional gate dimension for K
            L: Attention weight scale
        """
        base = head_idx * HD

        # Q: address nibbles
        for i, dim in enumerate(query_dims):
            attn.W_q.data[base + i, dim] = L

        # K: ADDR_KEY dimensions
        for i, dim in enumerate(key_dims):
            attn.W_k.data[base + i, dim] = L

        # V: byte values
        for i, dim in enumerate(value_dims):
            attn.W_v.data[base + i, dim] = 1.0

        # O: output
        for i, (out_dim, v_idx) in enumerate(zip(output_dims, range(len(value_dims)))):
            attn.W_o.data[out_dim, base + v_idx] = 1.0

        # Optional gating
        if gate_q_dim is not None:
            GATE = len(query_dims)
            attn.W_q.data[base + GATE, gate_q_dim] = L
            attn.W_q.data[base + GATE, BD.CONST] = -L / 2
            if gate_k_dim is not None:
                attn.W_k.data[base + GATE, gate_k_dim] = L

    @staticmethod
    def relay_head(
        attn,
        head_idx: int,
        q_marker: int,
        k_source: int,
        v_dims: List[int],
        o_dims: List[int],
        HD: int = 64,
        L: float = 15.0,
    ):
        """Set relay head for value broadcast.

        Q fires at target marker, K fires at source position,
        V copies specified dimensions, O broadcasts to output.

        Args:
            attn: Attention layer module
            head_idx: Head index
            q_marker: Query marker dimension
            k_source: Key source dimension (threshold flag)
            v_dims: Value dimensions to copy
            o_dims: Output dimensions
            HD: Head dimension
            L: Attention weight scale
        """
        base = head_idx * HD

        attn.W_q.data[base, q_marker] = L
        attn.W_k.data[base, k_source] = L

        for i, v_dim in enumerate(v_dims):
            attn.W_v.data[base + 1 + i, v_dim] = 1.0

        for i, o_dim in enumerate(o_dims):
            attn.W_o.data[o_dim, base + 1 + i] = 1.0

        # Anti-leakage gate
        GATE = len(v_dims) + 1
        attn.W_q.data[base + GATE, q_marker] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        attn.W_k.data[base + GATE, BD.CONST] = L

    # =========================================================================
    # FFN Primitives
    # =========================================================================

    @staticmethod
    def swiglu_and_gate(
        ffn,
        unit: int,
        up_dims: List[Tuple[int, float]],
        threshold: float,
        gate_dims: Optional[List[Tuple[int, float]]] = None,
        gate_bias: float = 1.0,
        out_dims: List[Tuple[int, float]] = None,
        S: float = 100.0,
    ) -> int:
        """Set SwiGLU AND gate pattern.

        Pattern: hidden = silu(W_up @ x + b_up) * (W_gate @ x + b_gate)

        Fires when sum of up_dims exceeds threshold, then multiplies by gate value.

        Args:
            ffn: FFN layer module
            unit: Hidden unit index
            up_dims: List of (dim, weight) for W_up
            threshold: Threshold value (b_up = -S * threshold)
            gate_dims: List of (dim, weight) for W_gate (None for constant gate)
            gate_bias: Bias for gate (default 1.0)
            out_dims: List of (dim, weight) for W_down
            S: Scale factor (default 100.0)

        Returns:
            Next available unit index
        """
        for dim, weight in up_dims:
            ffn.W_up.data[unit, dim] = S * weight
        ffn.b_up.data[unit] = -S * threshold

        if gate_dims is not None:
            for dim, weight in gate_dims:
                ffn.W_gate.data[unit, dim] = weight
        ffn.b_gate.data[unit] = gate_bias

        if out_dims is not None:
            for dim, weight in out_dims:
                ffn.W_down.data[dim, unit] = weight

        return unit + 1

    @staticmethod
    def cancel_pair(
        ffn,
        unit: int,
        old_dims: List[int],
        new_dims: List[int],
        out_dims: List[int],
        gate_dim: Optional[int] = None,
        S: float = 100.0,
    ) -> int:
        """Set cancel pair pattern: subtract old, add new.

        Uses W_gate to compute: -1*old + 1*new for each nibble dimension.

        Args:
            ffn: FFN layer module
            unit: Starting unit index
            old_dims: Dimensions to cancel
            new_dims: Dimensions to add
            out_dims: Output dimensions
            gate_dim: Optional gate dimension
            S: Scale factor

        Returns:
            Next available unit index
        """
        # Single unit with W_gate computation
        ffn.b_up.data[unit] = S  # Always active

        for old_d in old_dims:
            ffn.W_gate.data[unit, old_d] = -1.0
        for new_d in new_dims:
            ffn.W_gate.data[unit, new_d] = 1.0

        if gate_dim is not None:
            # Multiply gate by condition
            ffn.W_up.data[unit, gate_dim] = S
            ffn.b_up.data[unit] = -S * 0.5

        for out_d in out_dims:
            ffn.W_down.data[out_d, unit] = 2.0 / S

        return unit + 1

    @staticmethod
    def opcode_decode_unit(
        ffn,
        unit: int,
        lo_nibble: int,
        hi_nibble: int,
        marker_dim: int,
        op_dim: int,
        S: float = 100.0,
        extra_conditions: Optional[List[Tuple[int, float]]] = None,
        threshold: float = 1.5,
    ) -> int:
        """Set opcode decode unit: AND of lo/hi nibbles at marker.

        Pattern from _set_opcode_decode_ffn (vm_step.py:3382-3440):
          up = S*(OPCODE_BYTE_LO[lo] + OPCODE_BYTE_HI[hi] - threshold)
          gate = marker_dim
          down = op_dim with scale 10.0/S

        Args:
            ffn: FFN layer module
            unit: Hidden unit index
            lo_nibble: Low nibble value (0-15)
            hi_nibble: High nibble value (0-15)
            marker_dim: Marker dimension for gating
            op_dim: Output opcode dimension
            S: Scale factor
            extra_conditions: Additional (dim, weight) pairs for W_up
            threshold: Threshold value (default 1.5 for 2 inputs)

        Returns:
            Next available unit index
        """
        ffn.W_up.data[unit, BD.OPCODE_BYTE_LO + lo_nibble] = S
        ffn.W_up.data[unit, BD.OPCODE_BYTE_HI + hi_nibble] = S

        if extra_conditions:
            for dim, weight in extra_conditions:
                ffn.W_up.data[unit, dim] = S * weight

        ffn.b_up.data[unit] = -S * threshold
        ffn.W_gate.data[unit, marker_dim] = 1.0
        ffn.W_down.data[op_dim, unit] = 10.0 / S

        return unit + 1

    @staticmethod
    def nibble_copy(
        ffn,
        unit: int,
        src_lo: int,
        src_hi: int,
        dst_lo: int,
        dst_hi: int,
        gate_dim: Optional[int] = None,
        S: float = 100.0,
        suppress_residual: bool = False,
    ) -> int:
        """Set nibble copy pattern.

        Copies 16-dim one-hot nibbles from src to dst.

        Args:
            ffn: FFN layer module
            unit: Starting unit index
            src_lo: Source low nibble base
            src_hi: Source high nibble base
            dst_lo: Destination low nibble base
            dst_hi: Destination high nibble base
            gate_dim: Optional gate dimension
            S: Scale factor
            suppress_residual: If True, subtract residual

        Returns:
            Next available unit index
        """
        for k in range(16):
            # Copy low nibble
            ffn.b_up.data[unit + k] = S
            ffn.W_gate.data[unit + k, src_lo + k] = 1.0
            if gate_dim is not None:
                ffn.W_up.data[unit + k, gate_dim] = S
                ffn.b_up.data[unit + k] = -S * 0.5
            ffn.W_down.data[dst_lo + k, unit + k] = 2.0 / S

            if suppress_residual:
                ffn.W_down.data[dst_lo + k, unit + k] -= 2.0 / S

        for k in range(16):
            # Copy high nibble
            ffn.b_up.data[unit + 16 + k] = S
            ffn.W_gate.data[unit + 16 + k, src_hi + k] = 1.0
            if gate_dim is not None:
                ffn.W_up.data[unit + 16 + k, gate_dim] = S
                ffn.b_up.data[unit + 16 + k] = -S * 0.5
            ffn.W_down.data[dst_hi + k, unit + 16 + k] = 2.0 / S

            if suppress_residual:
                ffn.W_down.data[dst_hi + k, unit + 16 + k] -= 2.0 / S

        return unit + 32

    @staticmethod
    def step_pair(
        ffn,
        unit: int,
        input_dims: List[Tuple[int, float]],
        low_threshold: float,
        high_threshold: float,
        out_dim: int,
        gate_dims: Optional[List[Tuple[int, float]]] = None,
        gate_bias: float = 1.0,
        S: float = 100.0,
    ) -> int:
        """Set step pair pattern: step(>=low) - step(>=high).

        Creates binary output for input in range [low, high).

        Args:
            ffn: FFN layer module
            unit: Starting unit index
            input_dims: List of (dim, weight) for input
            low_threshold: Lower threshold
            high_threshold: Upper threshold
            out_dim: Output dimension
            gate_dims: Optional gate dimensions
            gate_bias: Gate bias
            S: Scale factor

        Returns:
            Next available unit index
        """
        # Unit 0: step(sum >= low)
        for dim, weight in input_dims:
            ffn.W_up.data[unit, dim] = S * weight
        ffn.b_up.data[unit] = -S * (low_threshold - 1.0)

        if gate_dims:
            for dim, weight in gate_dims:
                ffn.W_gate.data[unit, dim] = weight
        ffn.b_gate.data[unit] = gate_bias

        ffn.W_down.data[out_dim, unit] = 1.0 / S

        # Unit 1: -step(sum >= high)
        for dim, weight in input_dims:
            ffn.W_up.data[unit + 1, dim] = S * weight
        ffn.b_up.data[unit + 1] = -S * (high_threshold - 1.0)

        if gate_dims:
            for dim, weight in gate_dims:
                ffn.W_gate.data[unit + 1, dim] = weight
        ffn.b_gate.data[unit + 1] = gate_bias

        ffn.W_down.data[out_dim, unit + 1] = -1.0 / S

        return unit + 2

    @staticmethod
    def threshold_match(
        ffn,
        unit: int,
        up_dim: int,
        out_dim: int,
        threshold: float = 0.3,
        gate_dim: Optional[int] = None,
        gate_negative: bool = False,
        S: float = 100.0,
    ) -> int:
        """Set threshold match pattern for step structure detection.

        Pattern from _set_phase_a_ffn (vm_step.py:2163-2211):
          up = S * threshold_head_dim
          b_up = -S * threshold
          gate = optional condition (negative for NOT)
          down = transition flag

        Args:
            ffn: FFN layer module
            unit: Hidden unit index
            up_dim: Input dimension (threshold head output)
            out_dim: Output dimension (NEXT_* flag)
            threshold: Activation threshold
            gate_dim: Optional gate dimension
            gate_negative: If True, gate fires when gate_dim is 0
            S: Scale factor

        Returns:
            Next available unit index
        """
        ffn.W_up.data[unit, up_dim] = S
        ffn.b_up.data[unit] = -S * threshold

        if gate_dim is not None:
            ffn.W_gate.data[unit, gate_dim] = -1.0 if gate_negative else 1.0
            ffn.b_gate.data[unit] = 1.0 if gate_negative else 0.0
        else:
            ffn.b_gate.data[unit] = 1.0

        ffn.W_down.data[out_dim, unit] = 1.0 / S

        return unit + 1

    # =========================================================================
    # Set 1 (re-extracted from agent a19962): register-decrement, marker-write,
    # opcode-gated PC override, byte passthrough chain.
    # =========================================================================

    @staticmethod
    def register_decrement_unit(
        ffn,
        *,
        unit: int,
        register_marker_dim: int,
        op_gate_dim: int,
        embed_lo_dim: int,
        embed_hi_dim: int,
        output_lo_dim: int,
        output_hi_dim: int,
        decrement: int,
        S: float,
        op_strength: float = 1.0,
    ) -> int:
        """Generate 32 FFN units (16 lo + 16 hi nibble) implementing
        register -= decrement at a marker token, with hi-nibble borrow.

        Mirrors the PSH/JSR/ENT SP-decrement pattern in
        `_set_layer6_routing_ffn`. Each unit fires only when both
        ``op_gate_dim`` and ``register_marker_dim`` are active (threshold
        T=1.5 against the sum of two unit-magnitude signals scaled by
        ``S`` and ``op_strength*S`` respectively).

        Lo nibble: rotate by ``-decrement`` (mod 16), cancel identity carry.
        Hi nibble: borrow (-=1) when the original lo nibble was >= 8, also
        cancels identity carry.
        """
        T = 1.5
        # Lo nibble: shifted copy + cancel identity
        for k in range(16):
            new_k = (k - decrement) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_lo_dim + k] = 1.0
            ffn.W_down.data[output_lo_dim + new_k, unit] = 2.0 / S
            ffn.W_down.data[output_lo_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        # Hi nibble: borrow when old lo >= 8
        for k in range(16):
            new_k_borrow = (k - 1) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_hi_dim + k] = 1.0
            for lo_bit in range(8, 16):
                ffn.W_gate.data[unit, embed_lo_dim + lo_bit] = -1.0
            ffn.W_down.data[output_hi_dim + new_k_borrow, unit] = 2.0 / S
            ffn.W_down.data[output_hi_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        return unit

    @staticmethod
    def marker_write_unit(
        ffn,
        *,
        unit: int,
        marker_dim: int,
        op_gate_dim: int,
        source_dims,
        target_dim: int,
        S: float,
        magnitude: float = 2.0 / 100.0,
    ) -> int:
        """Single FFN unit gated by marker AND op_gate, summing
        ``source_dims`` (gate-relayed, weight 1.0 each) into ``target_dim``."""
        ffn.W_up.data[unit, marker_dim] = S
        ffn.W_up.data[unit, op_gate_dim] = S
        ffn.b_up.data[unit] = -S * 1.5
        for src in source_dims:
            ffn.W_gate.data[unit, src] = 1.0
        ffn.W_down.data[target_dim, unit] = magnitude
        return unit + 1

    @staticmethod
    def opcode_gated_pc_override(
        ffn,
        *,
        unit: int,
        op_gate_dim: int,
        mark_pc_dim: int,
        target_pc_lo_dim: int,
        target_pc_hi_dim: int,
        source_lo_dim: int,
        source_hi_dim: int,
        S: float,
        extra_blockers=None,
    ) -> int:
        """Generate 64 FFN units (32 cancel + 32 write) implementing an
        opcode-gated PC override at the PC marker."""
        T = 4.5
        blockers = list(extra_blockers) if extra_blockers else []

        # === Cancel phase (32 units): clear OUTPUT_LO/HI ===
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, target_pc_lo_dim + k] = -1.0
            ffn.W_down.data[target_pc_lo_dim + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, target_pc_hi_dim + k] = -1.0
            ffn.W_down.data[target_pc_hi_dim + k, unit] = 2.0 / S
            unit += 1
        # === Write phase (32 units): copy source -> OUTPUT ===
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, source_lo_dim + k] = 1.0
            ffn.W_down.data[target_pc_lo_dim + k, unit] = 2.0 / S
            unit += 1
        for k in range(16):
            ffn.W_up.data[unit, mark_pc_dim] = S
            ffn.W_up.data[unit, op_gate_dim] = S
            for d, w in blockers:
                ffn.W_up.data[unit, d] = w
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, source_hi_dim + k] = 1.0
            ffn.W_down.data[target_pc_hi_dim + k, unit] = 2.0 / S
            unit += 1
        return unit

    @staticmethod
    def byte_passthrough_chain(
        attn,
        *,
        head_idx: int,
        source_marker_dim: int,
        target_marker_dim: int,
        value_lo_dim: int,
        value_hi_dim: int,
        suppress_op_dims,
        S: float,
        HD: int,
        alibi_slope: float = 1.0,
        is_byte_strength: float = 3.0,
        has_se_strength: float = 1.0,
        suppress_strength: float = 3.0,
        q0_threshold: float = 3.5,
        gate_const: float = -20000.0,
        gate_target_marker: float = 10000.0,
        gate_has_se: float = 10000.0,
        gate_extras=None,
    ) -> None:
        """Configure a full attention head implementing byte passthrough
        across steps via shifted byte matching (Q byte K -> K byte K+1 of
        prev step).

        Mirrors `_set_layer10_byte_passthrough` (AX) and
        `_set_layer10_sp_byte_passthrough` (SP). Copies
        ``value_lo_dim`` / ``value_hi_dim`` (16 lo + 16 hi nibbles) into
        OUTPUT_LO/OUTPUT_HI with strength 2.0.

        Default Q[0] / Q[33] coefficients reproduce the AX function. The
        SP variant overrides ``is_byte_strength=1.0``, ``has_se_strength=2.0``,
        ``suppress_strength=2.0``, ``q0_threshold=1.5``, ``gate_const=-30000.0``,
        and adds ``gate_extras=[(IS_BYTE, 10000), (PSH_AT_SP, -10000),
        (CMP+3, -10000)]`` to encode the PSH/POP suppression.
        """
        L = S
        base = head_idx * HD

        # Q dim 0: IS_BYTE AND HAS_SE, suppressed by op flags
        attn.W_q.data[base + 0, BD.IS_BYTE] = L * is_byte_strength
        attn.W_q.data[base + 0, BD.HAS_SE] = L * has_se_strength
        for d in suppress_op_dims:
            attn.W_q.data[base + 0, d] = -L * suppress_strength
        attn.W_q.data[base + 0, BD.CONST] = -L * q0_threshold

        # Q dim 1: target marker discrimination
        attn.W_q.data[base + 1, target_marker_dim] = L
        attn.W_q.data[base + 1, BD.CONST] = -L / 2

        # Q dim 2: suppress byte 3 (predicts next register's marker)
        attn.W_q.data[base + 2, BD.BYTE_INDEX_3] = -L
        attn.W_q.data[base + 2, BD.CONST] = L / 2

        # K dim 0: IS_BYTE
        attn.W_k.data[base + 0, BD.IS_BYTE] = L
        # K dim 1: source marker (only target-register bytes are strong K)
        attn.W_k.data[base + 1, source_marker_dim] = L
        # K dim 2: suppress byte 0 in K (not a valid target for shifted matching)
        attn.W_k.data[base + 2, BD.BYTE_INDEX_0] = -L
        attn.W_k.data[base + 2, BD.CONST] = L / 2

        # Shifted byte matching: Q byte K -> K byte K+1 of prev step
        attn.W_q.data[base + 3, BD.BYTE_INDEX_0] = L
        attn.W_k.data[base + 3, BD.BYTE_INDEX_1] = L
        attn.W_q.data[base + 4, BD.BYTE_INDEX_1] = L
        attn.W_k.data[base + 4, BD.BYTE_INDEX_2] = L
        attn.W_q.data[base + 5, BD.BYTE_INDEX_2] = L
        attn.W_k.data[base + 5, BD.BYTE_INDEX_3] = L

        # Gate dim 33: hard AND of target_marker AND HAS_SE (kills leakage)
        attn.W_q.data[base + 33, BD.CONST] = gate_const
        attn.W_q.data[base + 33, target_marker_dim] = gate_target_marker
        attn.W_q.data[base + 33, BD.HAS_SE] = gate_has_se
        if gate_extras:
            for d, w in gate_extras:
                attn.W_q.data[base + 33, d] = w
        attn.W_k.data[base + 33, BD.CONST] = 5.0

        # V: copy 16 lo + 16 hi nibbles
        for k in range(16):
            attn.W_v.data[base + k, value_lo_dim + k] = 1.0
            attn.W_v.data[base + 16 + k, value_hi_dim + k] = 1.0

        # O: write to OUTPUT_LO/HI at strength 2.0
        for k in range(16):
            attn.W_o.data[BD.OUTPUT_LO + k, base + k] = 2.0
            attn.W_o.data[BD.OUTPUT_HI + k, base + 16 + k] = 2.0

        # Optional ALiBi slope override for this head
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes.data[head_idx] = alibi_slope

    @staticmethod
    def register_increment_unit(
        ffn,
        *,
        unit: int,
        register_marker_dim: int,
        op_gate_dim: int,
        embed_lo_dim: int,
        embed_hi_dim: int,
        output_lo_dim: int,
        output_hi_dim: int,
        increment: int,
        S: float,
        op_strength: float = 1.0,
    ) -> int:
        """Generate 32 FFN units (16 lo + 16 hi nibble) implementing
        register += increment at a marker token, with hi-nibble carry.

        Companion to :meth:`register_decrement_unit` -- same shape, opposite
        direction.
        """
        T = 1.5
        # Lo nibble: shifted copy + cancel identity
        for k in range(16):
            new_k = (k + increment) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_lo_dim + k] = 1.0
            ffn.W_down.data[output_lo_dim + new_k, unit] = 2.0 / S
            ffn.W_down.data[output_lo_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        # Hi nibble: carry when old lo >= 8 (adding 8 overflows lo nibble)
        for k in range(16):
            new_k_carry = (k + 1) % 16
            ffn.W_up.data[unit, op_gate_dim] = S * op_strength
            ffn.W_up.data[unit, register_marker_dim] = S
            ffn.b_up.data[unit] = -S * T
            ffn.W_gate.data[unit, embed_hi_dim + k] = 1.0
            for lo_bit in range(8):
                ffn.W_gate.data[unit, embed_lo_dim + lo_bit] = -1.0
            ffn.W_down.data[output_hi_dim + new_k_carry, unit] = 2.0 / S
            ffn.W_down.data[output_hi_dim + k, unit] += -2.0 / S  # cancel identity
            unit += 1
        return unit

    # =========================================================================
    # Extracted batch B (vm_step direct ports — byte-identical to imperative)
    # =========================================================================

    @staticmethod
    def nibble_rotation_chain(
        ffn,
        *,
        unit: int,
        gate_marker: int,
        source_lo_dim: int,
        source_hi_dim: int,
        target_lo_dim: int,
        target_hi_dim: int,
        offset: int = 1,
        with_carry: bool = True,
        S: float = 100.0,
        magnitude: float = 2.0,
        condition_dims: Optional[List[int]] = None,
        condition_threshold: Optional[float] = None,
    ) -> int:
        """Bake an FFN block that writes ``(source + offset) % 256`` to a
        target nibble pair, gated by an arbitrary AND of dims.

        Direct port of the L4 PC-rotation chains in ``_set_layer4_ffn``
        (vm_step.py): the offset=1 chain at AX marker, the offset∈{2,3,4}
        chains at AX byte positions (``with_carry=False`` because the
        TEMP source already has its carry applied), and the offset=1
        chain at PC marker → FETCH.

        For ``with_carry=False``: emits ``32`` units (16 lo rotations +
        16 hi copies). The hi copy assumes the source already has its
        carry applied (used for the multi-byte AX byte path that reads
        from TEMP, which itself was filled by an earlier carry-aware
        chain).

        For ``with_carry=True``: emits ``32 + 32*offset`` units (for
        offset=1: 64 units total; offset=2: 96; offset=3: 128;
        offset=4: 160) in the order::

            16 × lo_rotation:
                W_up[u, gate_marker]            = S       (+ extras)
                b_up[u]                         = -S*0.5  (or per cond)
                W_gate[u, source_lo+(k-off)%16] = 1.0
                W_down[target_lo+k, u]          = magnitude / S
            16 × hi_default_copy:
                W_up[u, gate_marker]            = S       (+ extras)
                b_up[u]                         = -S*0.5  (or per cond)
                W_gate[u, source_hi+k]          = 1.0
                W_down[target_hi+k, u]          = magnitude / S
            for carry_src in range(16-offset, 16):
              16 × {hi_cancel, hi_rotated} carry pairs:
                # Cancel default copy when source_lo[carry_src] == 1
                W_up[u,   gate_marker]                  = S       (+ extras)
                W_up[u,   source_lo+carry_src]          = S
                b_up[u]                                 = -S*1.5  (or per cond)
                W_gate[u, source_hi+k]                  = -1.0
                W_down[target_hi+k, u]                  = magnitude / S
                # Add rotated when source_lo[carry_src] == 1
                W_up[u+1, gate_marker]                  = S       (+ extras)
                W_up[u+1, source_lo+carry_src]          = S
                b_up[u+1]                               = -S*1.5  (or per cond)
                W_gate[u+1, source_hi+(k-1)%16]         = 1.0
                W_down[target_hi+k, u+1]                = magnitude / S

        For offset=1 (the canonical PC+1 case) carry_src ∈ {15} only,
        so the carry block emits 32 units (16 cancel + 16 rotated).
        For offset > 1 there are multiple carry sources (lo + N >= 16
        means lo ∈ [16-N, 15]), so the carry block emits 32*offset
        units. The hi-nibble carry block is always a +1 rotation
        because the carry from a +N lo rotation always contributes
        exactly +1 to the hi nibble (since lo nibble fits in 4 bits,
        max sum = 30 → carry ∈ {0, 1}). The lo-rotation source uses
        ``(k - offset) % 16`` but the hi-carry adjustment is
        ``(k - 1) % 16`` regardless.

        Math semantics: writes ``(source + offset) mod 256`` to the
        target nibble pair when the AND of (gate_marker,
        condition_dims...) is active.

        Args:
            ffn: FFN module.
            unit: First free hidden unit. Returns ``unit + 32`` (no
                carry) or ``unit + 32 + 32*offset`` (with carry; 64
                for offset=1, 96 for offset=2, etc.).
            gate_marker: Primary gate dim that scopes the rotation
                (e.g. ``BD.MARK_AX``, ``BD.MARK_PC``, or — for the
                multi-byte path that has no marker — ``BD.IS_BYTE``).
            source_lo_dim, source_hi_dim: Input nibble bases (e.g.
                ``BD.EMBED_LO``, ``BD.EMBED_HI``). Each is a 16-dim
                one-hot.
            target_lo_dim, target_hi_dim: Output nibble bases (e.g.
                ``BD.TEMP``, ``BD.TEMP+16``).
            offset: Integer rotation amount (typically 1, 2, 3, 4).
            with_carry: Emit the +16 carry-correction units (lo[15]==1
                triggers hi+=1). Set False for the multi-byte case
                where hi is just copied from a pre-rotated source.
            S: SwiGLU scale (default 100.0, matches vm_step.py).
            magnitude: ``W_down`` scale; the bake uses ``magnitude / S``.
                Default 2.0 matches the standard nibble-write scale.
            condition_dims: Extra W_up dims (with weight S) that AND
                with gate_marker (e.g. ``[BD.H1+AX_I,
                BD.BYTE_INDEX_0]`` for the multi-byte case). Must be a
                list of ints — all entries get weight S.
            condition_threshold: Override b_up base threshold. Defaults
                to ``0.5 + len(condition_dims)`` so b_up = ``-S*0.5``
                for 0 conditions, ``-S*2.5`` for 2 conditions, etc.
                Matches the vm_step.py threshold formula.

        Returns:
            New free unit index.
        """
        if condition_dims is None:
            condition_dims = []
        n_conds = len(condition_dims)
        # vm_step uses b_up = -S * (0.5 + n_conds) for the up gate — i.e.
        # threshold = (gate_marker + sum(condition_dims) - n_conds - 0.5).
        # For the carry block, threshold steps up by +1 (extra
        # source_lo+15 constraint) so b_up = -S * (1.5 + n_conds).
        if condition_threshold is None:
            base_thresh = 0.5 + n_conds
        else:
            base_thresh = condition_threshold
        carry_thresh = base_thresh + 1.0  # extra source_lo[15] AND
        u = unit
        scale = magnitude / S

        # 16 × lo rotation
        for k in range(16):
            src = (k - offset) % 16
            ffn.W_up[u, gate_marker] = S
            for cd in condition_dims:
                ffn.W_up[u, cd] = S
            ffn.b_up[u] = -S * base_thresh
            ffn.W_gate[u, source_lo_dim + src] = 1.0
            ffn.W_down[target_lo_dim + k, u] = scale
            u += 1

        # 16 × hi default copy
        for k in range(16):
            ffn.W_up[u, gate_marker] = S
            for cd in condition_dims:
                ffn.W_up[u, cd] = S
            ffn.b_up[u] = -S * base_thresh
            ffn.W_gate[u, source_hi_dim + k] = 1.0
            ffn.W_down[target_hi_dim + k, u] = scale
            u += 1

        if with_carry:
            # For offset=N, hi-nibble carries when (lo + N) >= 16, i.e.
            # lo ∈ [16-N, 15]. Emit a (cancel default, write rotated) pair
            # for each carry source bit, gated on source_lo[carry_src]=1.
            #   offset=1 → carry_src ∈ {15}        (32 units total)
            #   offset=2 → carry_src ∈ {14, 15}    (64 units total)
            #   offset=3 → carry_src ∈ {13, 14, 15} (96 units total)
            #   offset=4 → carry_src ∈ {12, …, 15} (128 units total)
            for carry_src in range(16 - offset, 16):
                for k in range(16):
                    # Cancel default copy when source_lo[carry_src] == 1
                    ffn.W_up[u, gate_marker] = S
                    ffn.W_up[u, source_lo_dim + carry_src] = S
                    for cd in condition_dims:
                        ffn.W_up[u, cd] = S
                    ffn.b_up[u] = -S * carry_thresh
                    ffn.W_gate[u, source_hi_dim + k] = -1.0
                    ffn.W_down[target_hi_dim + k, u] = scale
                    u += 1
                    # Add rotated +1 when source_lo[carry_src] == 1
                    hi_src = (k - 1) % 16
                    ffn.W_up[u, gate_marker] = S
                    ffn.W_up[u, source_lo_dim + carry_src] = S
                    for cd in condition_dims:
                        ffn.W_up[u, cd] = S
                    ffn.b_up[u] = -S * carry_thresh
                    ffn.W_gate[u, source_hi_dim + hi_src] = 1.0
                    ffn.W_down[target_hi_dim + k, u] = scale
                    u += 1

        return u


# Convenience aliases
P = Primitives
