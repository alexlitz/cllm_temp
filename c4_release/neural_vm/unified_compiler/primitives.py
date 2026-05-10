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
"""

import torch
from typing import Callable, List, Optional, Tuple, Union
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
        step_scoped: bool = False,
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
            step_scoped: If True, restrict to current step using HAS_SE matching
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

        # Step scoping: Q and K both require HAS_SE (current step only)
        if step_scoped:
            SCOPE_DIM = 34  # Use dim 34 for HAS_SE matching
            attn.W_q.data[base + SCOPE_DIM, BD.HAS_SE] = L
            attn.W_k.data[base + SCOPE_DIM, BD.HAS_SE] = L

        # Anti-leakage gate (dim 33)
        GATE = 33
        attn.W_q.data[base + GATE, marker_dim] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        if step_scoped:
            attn.W_q.data[base + GATE, BD.HAS_SE] = L
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
        step_scoped: bool = False,
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
            step_scoped: If True, restrict to current step using HAS_SE matching
        """
        base = head_idx * HD

        attn.W_q.data[base, q_marker] = L
        attn.W_k.data[base, k_source] = L

        for i, v_dim in enumerate(v_dims):
            attn.W_v.data[base + 1 + i, v_dim] = 1.0

        for i, o_dim in enumerate(o_dims):
            attn.W_o.data[o_dim, base + 1 + i] = 1.0

        # Step scoping: Q and K both require HAS_SE (current step only)
        if step_scoped:
            SCOPE_DIM = 34  # Use dim 34 for HAS_SE matching
            attn.W_q.data[base + SCOPE_DIM, BD.HAS_SE] = L
            attn.W_k.data[base + SCOPE_DIM, BD.HAS_SE] = L

        # Anti-leakage gate
        GATE = 33
        attn.W_q.data[base + GATE, q_marker] = L
        attn.W_q.data[base + GATE, BD.CONST] = -L / 2
        if step_scoped:
            attn.W_q.data[base + GATE, BD.HAS_SE] = L
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

        ffn.W_down.data[out_dim, unit] = 2.0 / S

        return unit + 1


    @staticmethod
    def byte_indexed_relay(
        attn,
        head_idx: int,
        q_marker_idx: int,
        k_area_threshold: int,
        k_area_exclude: List[int],
        src_lo: int,
        src_hi: int,
        out_lo: int,
        out_hi: int,
        HD: int = 64,
        L: float = 100.0,
        num_bytes: int = 4,
        step_scoped: bool = False,
    ):
        """Set attention head for byte-indexed relay between register areas.

        Copies values from a source area (identified by threshold pattern) to
        a target area (identified by marker), matching byte indices.

        Pattern from _set_layer10_stack0_byte_relay:
          Q: IS_BYTE + H1[target_marker] + BYTE_INDEX_i [+ HAS_SE if step_scoped]
          K: IS_BYTE + BYTE_INDEX_i + H4[area] - H1[exclude...] [+ HAS_SE if step_scoped]
          V: CLEAN_EMBED nibbles
          O: output nibbles

        Args:
            attn: Attention layer module
            head_idx: Head index
            q_marker_idx: Index of target marker (0=PC, 1=AX, 2=SP, 3=BP)
            k_area_threshold: H4 index for source area detection
            k_area_exclude: List of H1 indices to exclude from source area
            src_lo: Source low nibble base (e.g., BD.CLEAN_EMBED_LO)
            src_hi: Source high nibble base (e.g., BD.CLEAN_EMBED_HI)
            out_lo: Output low nibble base (e.g., BD.ALU_LO)
            out_hi: Output high nibble base (e.g., BD.ALU_HI)
            HD: Head dimension (default 64)
            L: Attention weight scale (default 100.0)
            num_bytes: Number of bytes to match (default 4)
            step_scoped: If True, add HAS_SE to restrict to current step
        """
        base = head_idx * HD
        BYTE_INDICES = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2, BD.BYTE_INDEX_3]

        # Q: Fire at target marker byte positions
        attn.W_q.data[base + 0, BD.IS_BYTE] = L
        attn.W_q.data[base + 0, BD.H1 + q_marker_idx] = L
        attn.W_q.data[base + 0, BD.CONST] = -L * 1.5

        # Q: Byte index matching (dims 1-3)
        for i in range(min(num_bytes, 4)):
            attn.W_q.data[base + 1 + i, BYTE_INDICES[i]] = L

        # Q: Suppress byte 3 if it would predict next marker
        if num_bytes < 4:
            attn.W_q.data[base + 4, BYTE_INDICES[3]] = -L
            attn.W_q.data[base + 4, BD.CONST] = L / 2

        # K: Fire at source area byte positions
        attn.W_k.data[base + 0, BD.IS_BYTE] = L

        # K: Same-byte index matching (dims 1-3)
        for i in range(min(num_bytes, 4)):
            attn.W_k.data[base + 1 + i, BYTE_INDICES[i]] = L

        # K: Area detection (dim 5) - H4[area] but not H1[excludes]
        attn.W_k.data[base + 5, BD.H4 + k_area_threshold] = L
        for exclude_idx in k_area_exclude:
            attn.W_k.data[base + 5, BD.H1 + exclude_idx] = -2 * L

        # Q dim 5 must match K's area detection pattern
        attn.W_q.data[base + 5, BD.CONST] = L

        # Step scoping with HAS_SE (dim 6)
        if step_scoped:
            attn.W_q.data[base + 6, BD.HAS_SE] = L
            attn.W_k.data[base + 6, BD.HAS_SE] = L

        # Anti-leakage gate (dim 33)
        attn.W_q.data[base + 33, BD.CONST] = -20000.0
        attn.W_q.data[base + 33, BD.IS_BYTE] = 10000.0
        attn.W_q.data[base + 33, BD.H1 + q_marker_idx] = 10000.0
        if step_scoped:
            attn.W_q.data[base + 33, BD.HAS_SE] = 10000.0
        attn.W_k.data[base + 33, BD.CONST] = 5.0

        # V: Copy source nibbles
        for k in range(16):
            attn.W_v.data[base + k, src_lo + k] = 1.0
            attn.W_v.data[base + 16 + k, src_hi + k] = 1.0

        # O: Write to output dimensions
        for k in range(16):
            attn.W_o.data[out_lo + k, base + k] = 2.0
            attn.W_o.data[out_hi + k, base + 16 + k] = 2.0


    @staticmethod
    def combined_alu_block(
        ffn,
        unit_start: int,
        input_a_dims: List[int],
        input_b_dims: List[int],
        output_dims: List[int],
        operations: List[Tuple[int, callable]],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create a combined ALU block that shares input matching across operations.

        Instead of 256 units per operation, this creates shared input matching
        with operation-specific output routing. Reduces total units when multiple
        operations use the same input structure.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            input_a_dims: 16-dim one-hot for first operand (e.g., ALU_LO)
            input_b_dims: 16-dim one-hot for second operand (e.g., AX_CARRY_LO)
            output_dims: 16-dim one-hot for output (e.g., OUTPUT_LO)
            operations: List of (op_flag_dim, alu_fn) pairs
                       op_flag_dim: Dimension for opcode flag (e.g., OP_ADD)
                       alu_fn: Function (a, b) -> result nibble
            S: Scale factor
            marker_dim: Optional marker dimension for gating (e.g., MARK_AX)

        Returns:
            Number of units used

        Example:
            # Combined ADD/SUB block
            Primitives.combined_alu_block(
                ffn, unit_start=0,
                input_a_dims=list(range(BD.ALU_LO, BD.ALU_LO+16)),
                input_b_dims=list(range(BD.AX_CARRY_LO, BD.AX_CARRY_LO+16)),
                output_dims=list(range(BD.OUTPUT_LO, BD.OUTPUT_LO+16)),
                operations=[
                    (BD.OP_ADD, lambda a, b: (a + b) & 0xF),
                    (BD.OP_SUB, lambda a, b: (a - b) & 0xF),
                ],
                marker_dim=BD.MARK_AX)
        """
        unit = unit_start

        # For each (a, b) combination, create units for all operations
        for a in range(16):
            for b in range(16):
                for op_flag_dim, alu_fn in operations:
                    result = alu_fn(a, b) & 0xF

                    # W_up: Input matching (3-way AND: marker + a + b)
                    if marker_dim is not None:
                        ffn.W_up.data[unit, marker_dim] = S
                    ffn.W_up.data[unit, input_a_dims[a]] = S
                    ffn.W_up.data[unit, input_b_dims[b]] = S

                    # Threshold for 2 or 3 conditions
                    threshold = 2.5 if marker_dim is not None else 1.5
                    ffn.b_up.data[unit] = -S * threshold

                    # W_gate: Operation flag
                    ffn.W_gate.data[unit, op_flag_dim] = 1.0

                    # W_down: Output result
                    ffn.W_down.data[output_dims[result], unit] = 2.0 / S

                    unit += 1

        return unit - unit_start

    @staticmethod
    def bitwise_alu_block(
        ffn,
        unit_start: int,
        input_a_dims: List[int],
        input_b_dims: List[int],
        output_dims: List[int],
        op_flag_dim: int,
        operation: str,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create an optimized bitwise ALU block.

        For bitwise operations (AND, OR, XOR), we can use a more efficient
        structure since bits are independent. Each output bit depends only
        on the corresponding input bits.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            input_a_dims: 16-dim one-hot for first operand
            input_b_dims: 16-dim one-hot for second operand
            output_dims: 16-dim one-hot for output
            op_flag_dim: Dimension for opcode flag
            operation: One of "AND", "OR", "XOR"
            S: Scale factor
            marker_dim: Optional marker dimension for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        # For bitwise ops, we still need the full lookup since
        # one-hot encoding doesn't allow bit-level access.
        # However, we document the pattern for future binary encoding.

        if operation == "AND":
            alu_fn = lambda a, b: a & b
        elif operation == "OR":
            alu_fn = lambda a, b: a | b
        elif operation == "XOR":
            alu_fn = lambda a, b: a ^ b
        else:
            raise ValueError(f"Unknown bitwise operation: {operation}")

        for a in range(16):
            for b in range(16):
                result = alu_fn(a, b) & 0xF

                # W_up: Input matching
                if marker_dim is not None:
                    ffn.W_up.data[unit, marker_dim] = S
                ffn.W_up.data[unit, input_a_dims[a]] = S
                ffn.W_up.data[unit, input_b_dims[b]] = S

                threshold = 2.5 if marker_dim is not None else 1.5
                ffn.b_up.data[unit] = -S * threshold

                # W_gate: Operation flag
                ffn.W_gate.data[unit, op_flag_dim] = 1.0

                # W_down: Output result
                ffn.W_down.data[output_dims[result], unit] = 2.0 / S

                unit += 1

        return unit - unit_start


    @staticmethod
    def comparison_block(
        ffn,
        unit_start: int,
        input_a_dims: List[int],
        input_b_dims: List[int],
        output_dims: List[int],
        cmp_dims: List[int],
        op_flag_dim: int,
        comparison: str,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create a comparison operation block (EQ, NE, LT, GT, LE, GE).

        Comparison uses a 3-way result:
        - CMP[0]: a < b (nibble level)
        - CMP[1]: a == b (nibble level)
        - CMP[2]: a > b (nibble level)

        Then combines for final result:
        - EQ: all nibbles equal
        - NE: any nibble not equal
        - LT: first unequal nibble has a < b
        - GT: first unequal nibble has a > b
        - LE: LT or EQ
        - GE: GT or EQ

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            input_a_dims: 16-dim one-hot for first operand
            input_b_dims: 16-dim one-hot for second operand
            output_dims: Output dimensions (typically OUTPUT_LO for 0/1 result)
            cmp_dims: Intermediate comparison dimensions (CMP+0, CMP+1, CMP+2)
            op_flag_dim: Dimension for opcode flag (e.g., OP_EQ)
            comparison: One of "EQ", "NE", "LT", "GT", "LE", "GE"
            S: Scale factor
            marker_dim: Optional marker dimension for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        # Phase 1: Generate 3-way comparison for each (a, b) pair
        for a in range(16):
            for b in range(16):
                # Determine comparison result for this nibble
                if a < b:
                    cmp_result = 0  # CMP+0 = LT
                elif a == b:
                    cmp_result = 1  # CMP+1 = EQ
                else:
                    cmp_result = 2  # CMP+2 = GT

                # W_up: Input matching
                if marker_dim is not None:
                    ffn.W_up.data[unit, marker_dim] = S
                ffn.W_up.data[unit, input_a_dims[a]] = S
                ffn.W_up.data[unit, input_b_dims[b]] = S

                threshold = 2.5 if marker_dim is not None else 1.5
                ffn.b_up.data[unit] = -S * threshold

                # W_gate: Operation flag
                ffn.W_gate.data[unit, op_flag_dim] = 1.0

                # W_down: Output to CMP dimension
                if cmp_result < len(cmp_dims):
                    ffn.W_down.data[cmp_dims[cmp_result], unit] = 2.0 / S

                unit += 1

        # Phase 2: Combine CMP dimensions to final result
        # This depends on comparison type
        if comparison == "EQ":
            # EQ: Output 1 if CMP[1] (equal), else 0
            # Unit fires when equal
            ffn.W_up.data[unit, cmp_dims[1]] = S  # CMP_EQ
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, op_flag_dim] = 1.0
            ffn.W_down.data[output_dims[1], unit] = 2.0 / S  # Output 1
            unit += 1

        elif comparison == "NE":
            # NE: Output 1 if CMP[0] or CMP[2] (not equal)
            ffn.W_up.data[unit, cmp_dims[0]] = S  # CMP_LT
            ffn.W_up.data[unit, cmp_dims[2]] = S  # CMP_GT
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 0.5  # Fire if either LT or GT
            ffn.W_gate.data[unit, op_flag_dim] = 1.0
            ffn.W_down.data[output_dims[1], unit] = 2.0 / S
            unit += 1

        elif comparison == "LT":
            # LT: Output 1 if CMP[0] (less than)
            ffn.W_up.data[unit, cmp_dims[0]] = S
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, op_flag_dim] = 1.0
            ffn.W_down.data[output_dims[1], unit] = 2.0 / S
            unit += 1

        elif comparison == "GT":
            # GT: Output 1 if CMP[2] (greater than)
            ffn.W_up.data[unit, cmp_dims[2]] = S
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 1.5
            ffn.W_gate.data[unit, op_flag_dim] = 1.0
            ffn.W_down.data[output_dims[1], unit] = 2.0 / S
            unit += 1

        elif comparison == "LE":
            # LE: Output 1 if CMP[0] or CMP[1] (less or equal)
            ffn.W_up.data[unit, cmp_dims[0]] = S
            ffn.W_up.data[unit, cmp_dims[1]] = S
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, op_flag_dim] = 1.0
            ffn.W_down.data[output_dims[1], unit] = 2.0 / S
            unit += 1

        elif comparison == "GE":
            # GE: Output 1 if CMP[1] or CMP[2] (greater or equal)
            ffn.W_up.data[unit, cmp_dims[1]] = S
            ffn.W_up.data[unit, cmp_dims[2]] = S
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.W_gate.data[unit, op_flag_dim] = 1.0
            ffn.W_down.data[output_dims[1], unit] = 2.0 / S
            unit += 1

        return unit - unit_start

    @staticmethod
    def zfod_memory_read(
        attn,
        head_idx: int,
        addr_dims: List[int],
        zfod_base_dim: int,
        zfod_size_dim: int,
        output_dims: List[int],
        HD: int = 64,
        L: float = 15.0,
    ):
        """Zero-Fill-On-Demand memory read pattern.

        Returns zero for addresses in the ZFOD range that haven't been written.
        This optimizes memset by avoiding explicit zero-writes.

        Pattern:
        - If addr >= ZFOD_BASE and addr < ZFOD_BASE + ZFOD_SIZE and not written:
          Return 0
        - Else: Normal memory read

        Args:
            attn: Attention layer module
            head_idx: Head index
            addr_dims: Address dimensions for query
            zfod_base_dim: Dimension holding ZFOD base address
            zfod_size_dim: Dimension holding ZFOD size
            output_dims: Output dimensions for read value
            HD: Head dimension
            L: Attention weight scale

        Note: Full ZFOD requires tracking written addresses, which needs
        additional state. This primitive provides the read-zero path.
        """
        base = head_idx * HD

        # Q: Address query
        for i, addr_dim in enumerate(addr_dims[:32]):
            attn.W_q.data[base + i, addr_dim] = L

        # K: Match ZFOD range marker
        attn.W_k.data[base, zfod_base_dim] = L
        attn.W_k.data[base + 1, zfod_size_dim] = L

        # V: Output zeros (constant zero embedding)
        # Note: ZFOD relies on not attending to real memory when in ZFOD range
        # The actual implementation would use a separate "ZFOD marker" token
        # that always returns zero

        # O: Write to output
        for i, out_dim in enumerate(output_dims[:32]):
            attn.W_o.data[out_dim, base + i] = 1.0

    @staticmethod
    def range_check_block(
        ffn,
        unit_start: int,
        value_dims: List[int],
        low_bound: int,
        high_bound: int,
        output_dim: int,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create a range check block: output 1 if low <= value < high.

        For checking if a value falls within a specific range.
        Uses thermometer-style comparison.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            value_dims: 16-dim one-hot for input value
            low_bound: Lower bound (inclusive)
            high_bound: Upper bound (exclusive)
            output_dim: Output dimension for result (0 or 1)
            S: Scale factor
            marker_dim: Optional marker dimension for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        # Create units for each value in range
        for v in range(16):
            in_range = low_bound <= v < high_bound

            # W_up: Match value
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.W_up.data[unit, value_dims[v]] = S

            threshold = 1.5 if marker_dim is not None else 0.5
            ffn.b_up.data[unit] = -S * threshold

            # Constant gate (always active)
            ffn.b_gate.data[unit] = 1.0

            # Output 1 if in range, nothing if not
            if in_range:
                ffn.W_down.data[output_dim, unit] = 2.0 / S

            unit += 1

        return unit - unit_start


    # ==========================================================================
    # Missing Primitives (added 2026-05-08)
    # ==========================================================================

    @staticmethod
    def const_unit(
        ffn,
        unit_start: int,
        output_dim: int,
        value: float,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create a unit that outputs a constant value.

        Pattern: bias-only activation (no input required).
        CONST primitive from WEIGHT_COMPILER_PRIMITIVES.md.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            output_dim: Output dimension
            value: Constant value to output
            S: Scale factor
            marker_dim: Optional marker for gating (only fire at marker)

        Returns:
            Number of units used (2 for cancel pair)
        """
        unit = unit_start

        if marker_dim is not None:
            # Gated constant: only fire at marker position
            ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 0.5
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[output_dim, unit] = value * 2.0 / S
            unit += 1
        else:
            # Ungated constant: use cancel pair for stability
            # Unit 0: +constant
            ffn.b_up.data[unit] = S * abs(value)
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[output_dim, unit] = (1.0 if value >= 0 else -1.0) / S
            unit += 1

            # Unit 1: stability (cancel negative contribution)
            ffn.b_up.data[unit] = -S * abs(value)
            ffn.b_gate.data[unit] = 1.0
            ffn.W_down.data[output_dim, unit] = (1.0 if value >= 0 else -1.0) / S
            unit += 1

        return unit - unit_start

    @staticmethod
    def clear_dims(
        ffn,
        unit_start: int,
        dims_to_clear: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create units that clear (zero) specified dimensions.

        CLEAR primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: subtract current value from itself.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            dims_to_clear: List of dimensions to zero
            S: Scale factor
            marker_dim: Optional marker for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        for dim in dims_to_clear:
            # Read current value and subtract it
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
                ffn.b_up.data[unit] = -S * 0.5
            else:
                ffn.b_up.data[unit] = S * 0.5  # Always active

            ffn.W_gate.data[unit, dim] = 1.0  # Gate by current value
            ffn.W_down.data[dim, unit] = -2.0 / S  # Subtract to zero

            unit += 1

        return unit - unit_start

    @staticmethod
    def logical_not(
        ffn,
        unit_start: int,
        input_dim: int,
        output_dim: int,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create units for logical NOT: out = !in (1 - in).

        NOT primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: output = 1 - input.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            input_dim: Input dimension (0 or 1)
            output_dim: Output dimension
            S: Scale factor
            marker_dim: Optional marker for gating

        Returns:
            Number of units used (2)
        """
        unit = unit_start

        # Unit 0: constant 1
        if marker_dim is not None:
            ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 0.5
        else:
            ffn.b_up.data[unit] = S
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[output_dim, unit] = 1.0 / S
        unit += 1

        # Unit 1: subtract input
        if marker_dim is not None:
            ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 0.5
        else:
            ffn.b_up.data[unit] = S * 0.5
        ffn.W_gate.data[unit, input_dim] = 1.0
        ffn.W_down.data[output_dim, unit] = -1.0 / S
        unit += 1

        return unit - unit_start

    @staticmethod
    def select_mux(
        ffn,
        unit_start: int,
        cond_dim: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create units for SELECT/MUX: out = cond ? a : b.

        SELECT primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: cond * a + (1 - cond) * b.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            cond_dim: Condition dimension (0 or 1)
            a_dims: Dimensions for 'a' value (true branch)
            b_dims: Dimensions for 'b' value (false branch)
            out_dims: Output dimensions
            S: Scale factor
            marker_dim: Optional marker for gating

        Returns:
            Number of units used
        """
        assert len(a_dims) == len(b_dims) == len(out_dims), "Dimension counts must match"
        unit = unit_start

        for i, (a_dim, b_dim, out_dim) in enumerate(zip(a_dims, b_dims, out_dims)):
            # Branch A: cond * a (fire when cond=1)
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.W_up.data[unit, cond_dim] = S
            ffn.b_up.data[unit] = -S * (1.5 if marker_dim else 0.5)
            ffn.W_gate.data[unit, a_dim] = 1.0
            ffn.W_down.data[out_dim, unit] = 2.0 / S
            unit += 1

            # Branch B: (1-cond) * b (fire when cond=0)
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.W_up.data[unit, cond_dim] = -S  # Negative: fire when cond=0
            ffn.b_up.data[unit] = -S * (0.5 if marker_dim else -0.5)
            ffn.W_gate.data[unit, b_dim] = 1.0
            ffn.W_down.data[out_dim, unit] = 2.0 / S
            unit += 1

        return unit - unit_start

    @staticmethod
    def gated_relay(
        ffn,
        unit_start: int,
        gate_dim: int,
        src_dims: List[int],
        dst_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        gate_threshold: float = 0.5,
    ) -> int:
        """Create units for gated relay: if gate: dst = src.

        IF_THEN primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: only copy when gate condition is met.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            gate_dim: Gate dimension (fire when > 0)
            src_dims: Source dimensions
            dst_dims: Destination dimensions
            S: Scale factor
            marker_dim: Optional marker for additional gating
            gate_threshold: Threshold for gate activation

        Returns:
            Number of units used
        """
        assert len(src_dims) == len(dst_dims), "Source and dest dims must match"
        unit = unit_start

        for src_dim, dst_dim in zip(src_dims, dst_dims):
            # Gated copy: fire when gate > threshold
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
            ffn.W_up.data[unit, gate_dim] = S
            threshold = 1.5 if marker_dim else gate_threshold
            ffn.b_up.data[unit] = -S * threshold

            ffn.W_gate.data[unit, src_dim] = 1.0
            ffn.W_down.data[dst_dim, unit] = 2.0 / S
            unit += 1

        return unit - unit_start

    @staticmethod
    def opcode_conditional_relay(
        ffn,
        unit_start: int,
        positive_conds: List[tuple],  # [(dim, scale), ...]
        negative_conds: List[tuple],  # [(dim, scale), ...]
        threshold: float,
        src_dims: List[int],
        dst_dims: List[int],
        S: float = 100.0,
        out_scale: float = 1.0,
    ) -> int:
        """Create units for opcode-conditional relay with complex gating.

        Used for routing patterns like:
          - IMM: OP_IMM + MARK_AX - MARK_PC - IS_BYTE > T → FETCH → OUTPUT
          - EXIT: OP_EXIT + MARK_AX - MARK_PC - IS_BYTE > T → AX_CARRY → OUTPUT

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            positive_conds: List of (dim, scale) for positive conditions
            negative_conds: List of (dim, scale) for negative blockers
            threshold: Activation threshold
            src_dims: Source dimensions (gated)
            dst_dims: Destination dimensions (output)
            S: Base scale factor
            out_scale: Output scale multiplier (default 1.0 → 2.0/S)

        Returns:
            Number of units used
        """
        assert len(src_dims) == len(dst_dims), "Source and dest dims must match"
        unit = unit_start

        for src_dim, dst_dim in zip(src_dims, dst_dims):
            # Positive conditions
            for dim, scale in positive_conds:
                ffn.W_up.data[unit, dim] = S * scale

            # Negative blockers
            for dim, scale in negative_conds:
                ffn.W_up.data[unit, dim] = -S * scale

            # Threshold bias
            ffn.b_up.data[unit] = -S * threshold

            # Gate on source
            ffn.W_gate.data[unit, src_dim] = 1.0

            # Output
            ffn.W_down.data[dst_dim, unit] = 2.0 / (S * out_scale)
            unit += 1

        return unit - unit_start

    @staticmethod
    def shift_lookup(
        ffn,
        unit_start: int,
        value_lo_dims: List[int],
        value_hi_dims: List[int],
        shift_dims: List[int],
        out_lo_dims: List[int],
        out_hi_dims: List[int],
        opcode_dim: int,
        shift_right: bool = False,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        max_shift: int = 8,
    ) -> int:
        """Create lookup table units for SHL/SHR operations.

        SHL/SHR primitives from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: 256-entry lookup for (value, shift_amount) pairs.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            value_lo_dims: 16-dim one-hot for value lo nibble
            value_hi_dims: 16-dim one-hot for value hi nibble
            shift_dims: 16-dim one-hot for shift amount
            out_lo_dims: 16-dim output for lo nibble
            out_hi_dims: 16-dim output for hi nibble
            opcode_dim: OP_SHL or OP_SHR dimension
            shift_right: If True, shift right; else shift left
            S: Scale factor
            marker_dim: Optional marker for gating
            max_shift: Maximum shift amount to support

        Returns:
            Number of units used
        """
        unit = unit_start

        # Create lookup for all (value_lo, value_hi, shift) combinations
        for val_lo in range(16):
            for val_hi in range(16):
                value = val_lo | (val_hi << 4)
                for shift in range(min(max_shift + 1, 16)):
                    if shift_right:
                        result = value >> shift
                    else:
                        result = (value << shift) & 0xFFFF  # 16-bit result

                    result_lo = result & 0xF
                    result_hi = (result >> 4) & 0xF

                    # 5-way AND: marker + value_lo + value_hi + shift + opcode
                    if marker_dim is not None:
                        ffn.W_up.data[unit, marker_dim] = S
                    ffn.W_up.data[unit, value_lo_dims[val_lo]] = S
                    ffn.W_up.data[unit, value_hi_dims[val_hi]] = S
                    ffn.W_up.data[unit, shift_dims[shift]] = S
                    # Require shift_hi = 0 (shift < 16)
                    ffn.W_up.data[unit, shift_dims[0] + 16] = S  # Assuming hi dims follow lo

                    n_conditions = 5 if marker_dim else 4
                    ffn.b_up.data[unit] = -S * (n_conditions - 0.5)

                    ffn.W_gate.data[unit, opcode_dim] = 1.0
                    ffn.W_down.data[out_lo_dims[result_lo], unit] = 2.0 / S
                    ffn.W_down.data[out_hi_dims[result_hi], unit] = 2.0 / S

                    unit += 1

        return unit - unit_start

    @staticmethod
    def multi_byte_address_fetch(
        attn,
        head_idx: int,
        addr_dims: List[int],
        addr_key_dims: List[int],
        value_lo_dims: List[int],
        value_hi_dims: List[int],
        out_lo_dims: List[int],
        out_hi_dims: List[int],
        marker_dim: int,
        L: float = 20.0,
        HD: int = 64,
        has_se_gate: bool = False,
        has_se_dim: Optional[int] = None,
    ) -> None:
        """Create attention head for multi-byte address fetch.

        Queries memory using address nibbles and returns fetched value.
        Pattern from L5 fetch and L8 multibyte fetch.

        Args:
            attn: Attention layer module
            head_idx: Head index (0-7)
            addr_dims: Address nibble dimensions (32 dims: 16 lo + 16 hi)
            addr_key_dims: ADDR_KEY dimensions for matching
            value_lo_dims: Source value lo nibbles (CLEAN_EMBED_LO)
            value_hi_dims: Source value hi nibbles (CLEAN_EMBED_HI)
            out_lo_dims: Output lo nibbles (FETCH_LO or AX_CARRY_LO)
            out_hi_dims: Output hi nibbles (FETCH_HI or AX_CARRY_HI)
            marker_dim: Marker for query gating
            L: Scale factor
            HD: Head dimension
            has_se_gate: Whether to add HAS_SE gating
            has_se_dim: HAS_SE dimension (required if has_se_gate=True)
        """
        base = head_idx * HD

        # Q: Address nibbles from addr_dims
        for k, dim in enumerate(addr_dims[:32]):
            attn.W_q.data[base + k, dim] = L

        # Third nibble gate: marker
        attn.W_q.data[base + 32, marker_dim] = L

        # K: Address key nibbles
        for k, dim in enumerate(addr_key_dims[:33]):
            attn.W_k.data[base + k, dim] = L

        # Anti-leakage gate
        GATE = 33
        attn.W_q.data[base + GATE, marker_dim] = 500.0
        attn.W_q.data[base + GATE, addr_dims[0] - addr_dims[0]] = -500.0  # CONST placeholder
        attn.W_k.data[base + GATE, addr_key_dims[0] - addr_key_dims[0]] = 5.0

        # HAS_SE gate (optional)
        if has_se_gate and has_se_dim is not None:
            HAS_SE_GATE = 34
            attn.W_q.data[base + HAS_SE_GATE, has_se_dim] = 500.0
            attn.W_q.data[base + HAS_SE_GATE, addr_dims[0] - addr_dims[0]] = -500.0
            attn.W_k.data[base + HAS_SE_GATE, addr_key_dims[0] - addr_key_dims[0]] = 5.0

        # V: Copy value nibbles
        for k, dim in enumerate(value_lo_dims[:16]):
            attn.W_v.data[base + 32 + k, dim] = 1.0
        for k, dim in enumerate(value_hi_dims[:16]):
            attn.W_v.data[base + 48 + k, dim] = 1.0

        # O: Output to dest dimensions
        for k, dim in enumerate(out_lo_dims[:16]):
            attn.W_o.data[dim, base + 32 + k] = 1.0
        for k, dim in enumerate(out_hi_dims[:16]):
            attn.W_o.data[dim, base + 48 + k] = 1.0

    @staticmethod
    def history_marker_relay(
        attn,
        head_idx: int,
        q_marker_dim: int,
        k_history_pos_dim: int,
        k_history_neg_dim: int,
        v_dims: List[int],
        o_dims: List[int],
        L: float = 15.0,
        HD: int = 64,
        extra_q_dim: Optional[int] = None,
        extra_q_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        """Create attention head for history-marker relay.

        Uses L1H1/L1H0 style markers to attend to previous step positions.
        Pattern from L9 LEV address relay.

        Args:
            attn: Attention layer module
            head_idx: Head index
            q_marker_dim: Query marker (e.g., MARK_SP)
            k_history_pos_dim: Positive history marker (e.g., L1H1[idx])
            k_history_neg_dim: Negative history marker (e.g., L1H0[idx])
            v_dims: Value dimensions to copy
            o_dims: Output dimensions
            L: Scale factor
            HD: Head dimension
            extra_q_dim: Optional extra Q dimension (e.g., OP_LEV)
            extra_q_scale: Scale for extra Q dim (normalized, e.g., L/5)
            v_scale: V output scale (e.g., 3.0 to dominate residual)
        """
        base = head_idx * HD

        # Q: Fire at marker
        attn.W_q.data[base, q_marker_dim] = L

        if extra_q_dim is not None:
            attn.W_q.data[base, extra_q_dim] = L * extra_q_scale

        # K: History marker pattern (L1H1[idx] - L1H0[idx])
        attn.W_k.data[base, k_history_pos_dim] = L
        attn.W_k.data[base, k_history_neg_dim] = -L

        # V: Copy source values
        for k, v_dim in enumerate(v_dims):
            attn.W_v.data[base + 1 + k, v_dim] = v_scale

        # O: Write to output
        for k, o_dim in enumerate(o_dims):
            attn.W_o.data[o_dim, base + 1 + k] = 1.0

        # Anti-leakage gate
        GATE = 33
        attn.W_q.data[base + GATE, q_marker_dim] = L
        attn.W_k.data[base + GATE, k_history_pos_dim] = L  # Key side gate

    # ==========================================================================
    # Additional Primitives from WEIGHT_COMPILER_PRIMITIVES.md
    # ==========================================================================

    @staticmethod
    def swap_registers(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        temp_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create units for SWAP: (a, b) = (b, a).

        SWAP primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: temp=a, a=b, b=temp via 3 MOVE operations.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: Dimensions for register A
            b_dims: Dimensions for register B
            temp_dims: Temporary dimensions for swap
            S: Scale factor
            marker_dim: Optional marker for gating

        Returns:
            Number of units used
        """
        assert len(a_dims) == len(b_dims) == len(temp_dims), "Dimension counts must match"
        unit = unit_start

        # Phase 1: temp = a (copy A to temp, clear A)
        for a_dim, temp_dim in zip(a_dims, temp_dims):
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
                ffn.b_up.data[unit] = -S * 0.5
            else:
                ffn.b_up.data[unit] = S * 0.5

            ffn.W_gate.data[unit, a_dim] = 1.0
            ffn.W_down.data[temp_dim, unit] = 2.0 / S  # Copy to temp
            ffn.W_down.data[a_dim, unit] = -2.0 / S  # Clear A
            unit += 1

        # Phase 2: a = b (copy B to A, clear B)
        for a_dim, b_dim in zip(a_dims, b_dims):
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
                ffn.b_up.data[unit] = -S * 0.5
            else:
                ffn.b_up.data[unit] = S * 0.5

            ffn.W_gate.data[unit, b_dim] = 1.0
            ffn.W_down.data[a_dim, unit] = 2.0 / S  # Copy to A
            ffn.W_down.data[b_dim, unit] = -2.0 / S  # Clear B
            unit += 1

        # Phase 3: b = temp (copy temp to B)
        for b_dim, temp_dim in zip(b_dims, temp_dims):
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
                ffn.b_up.data[unit] = -S * 0.5
            else:
                ffn.b_up.data[unit] = S * 0.5

            ffn.W_gate.data[unit, temp_dim] = 1.0
            ffn.W_down.data[b_dim, unit] = 2.0 / S  # Copy to B
            unit += 1

        return unit - unit_start

    @staticmethod
    def bitwise_and_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for BIT_AND: out = a & b.

        BIT_AND primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: 16x16 nibble lookup table.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for first operand
            b_dims: 16-dim one-hot for second operand
            out_dims: 16-dim one-hot for output
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        for result in range(16):
            # Find all (a, b) pairs where a & b == result
            for a in range(16):
                for b in range(16):
                    if (a & b) == result:
                        # Create unit that fires for this (a, b) pair
                        n_cond = 2
                        if marker_dim is not None:
                            ffn.W_up.data[unit, marker_dim] = S
                            n_cond += 1
                        ffn.W_up.data[unit, a_dims[a]] = S
                        ffn.W_up.data[unit, b_dims[b]] = S

                        ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                        if opcode_dim is not None:
                            ffn.W_gate.data[unit, opcode_dim] = 1.0
                        else:
                            ffn.b_gate.data[unit] = 1.0

                        ffn.W_down.data[out_dims[result], unit] = 2.0 / S

                        unit += 1

        return unit - unit_start

    @staticmethod
    def bitwise_or_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for BIT_OR: out = a | b.

        BIT_OR primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: 16x16 nibble lookup table.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for first operand
            b_dims: 16-dim one-hot for second operand
            out_dims: 16-dim one-hot for output
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        for result in range(16):
            for a in range(16):
                for b in range(16):
                    if (a | b) == result:
                        n_cond = 2
                        if marker_dim is not None:
                            ffn.W_up.data[unit, marker_dim] = S
                            n_cond += 1
                        ffn.W_up.data[unit, a_dims[a]] = S
                        ffn.W_up.data[unit, b_dims[b]] = S

                        ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                        if opcode_dim is not None:
                            ffn.W_gate.data[unit, opcode_dim] = 1.0
                        else:
                            ffn.b_gate.data[unit] = 1.0

                        ffn.W_down.data[out_dims[result], unit] = 2.0 / S

                        unit += 1

        return unit - unit_start

    @staticmethod
    def bitwise_xor_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for BIT_XOR: out = a ^ b.

        BIT_XOR primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: 16x16 nibble lookup table.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for first operand
            b_dims: 16-dim one-hot for second operand
            out_dims: 16-dim one-hot for output
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        for result in range(16):
            for a in range(16):
                for b in range(16):
                    if (a ^ b) == result:
                        n_cond = 2
                        if marker_dim is not None:
                            ffn.W_up.data[unit, marker_dim] = S
                            n_cond += 1
                        ffn.W_up.data[unit, a_dims[a]] = S
                        ffn.W_up.data[unit, b_dims[b]] = S

                        ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                        if opcode_dim is not None:
                            ffn.W_gate.data[unit, opcode_dim] = 1.0
                        else:
                            ffn.b_gate.data[unit] = 1.0

                        ffn.W_down.data[out_dims[result], unit] = 2.0 / S

                        unit += 1

        return unit - unit_start

    @staticmethod
    def add_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
        suppress_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for ADD: out = (a + b) % 16.

        Nibble addition with modular result (no carry).

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for first operand (base index, +0 to +15)
            b_dims: 16-dim one-hot for second operand
            out_dims: 16-dim one-hot for output
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating
            suppress_dim: Optional dimension to suppress at (W_up negative)

        Returns:
            Number of units used (256 for full 16x16)
        """
        unit = unit_start

        for a in range(16):
            for b in range(16):
                result = (a + b) % 16
                n_cond = 2
                if marker_dim is not None:
                    ffn.W_up.data[unit, marker_dim] = S
                    n_cond += 1
                if suppress_dim is not None:
                    ffn.W_up.data[unit, suppress_dim] = -S * 2  # Strong suppression
                ffn.W_up.data[unit, a_dims[a]] = S
                ffn.W_up.data[unit, b_dims[b]] = S

                ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                if opcode_dim is not None:
                    ffn.W_gate.data[unit, opcode_dim] = 1.0
                else:
                    ffn.b_gate.data[unit] = 1.0

                ffn.W_down.data[out_dims[result], unit] = 2.0 / S

                unit += 1

        return unit - unit_start

    @staticmethod
    def add_carry_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        carry_dim: int,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
        suppress_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for ADD carry detection: carry = (a + b >= 16) ? 1 : 0.

        Only creates units for pairs where carry occurs (120 units).

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for first operand
            b_dims: 16-dim one-hot for second operand
            carry_dim: Output dimension for carry flag
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating
            suppress_dim: Optional dimension to suppress at

        Returns:
            Number of units used (120 for pairs where a+b >= 16)
        """
        unit = unit_start

        for a in range(16):
            for b in range(16):
                if a + b >= 16:
                    n_cond = 2
                    if marker_dim is not None:
                        ffn.W_up.data[unit, marker_dim] = S
                        n_cond += 1
                    if suppress_dim is not None:
                        ffn.W_up.data[unit, suppress_dim] = -S * 2
                    ffn.W_up.data[unit, a_dims[a]] = S
                    ffn.W_up.data[unit, b_dims[b]] = S

                    ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                    if opcode_dim is not None:
                        ffn.W_gate.data[unit, opcode_dim] = 1.0
                    else:
                        ffn.b_gate.data[unit] = 1.0

                    # Scale for gate normalization (gate ≈ 5 → carry ≈ 1)
                    ffn.W_down.data[carry_dim, unit] = 2.0 / (S * 5.0)

                    unit += 1

        return unit - unit_start

    @staticmethod
    def sub_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
        suppress_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for SUB: out = (a - b) % 16.

        Nibble subtraction with modular result (no borrow).

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for minuend (value being subtracted from)
            b_dims: 16-dim one-hot for subtrahend (value being subtracted)
            out_dims: 16-dim one-hot for output
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating
            suppress_dim: Optional dimension to suppress at

        Returns:
            Number of units used (256 for full 16x16)
        """
        unit = unit_start

        for a in range(16):
            for b in range(16):
                result = (a - b) % 16
                n_cond = 2
                if marker_dim is not None:
                    ffn.W_up.data[unit, marker_dim] = S
                    n_cond += 1
                if suppress_dim is not None:
                    ffn.W_up.data[unit, suppress_dim] = -S * 2
                ffn.W_up.data[unit, a_dims[a]] = S
                ffn.W_up.data[unit, b_dims[b]] = S

                ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                if opcode_dim is not None:
                    ffn.W_gate.data[unit, opcode_dim] = 1.0
                else:
                    ffn.b_gate.data[unit] = 1.0

                ffn.W_down.data[out_dims[result], unit] = 2.0 / S

                unit += 1

        return unit - unit_start

    @staticmethod
    def sub_borrow_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        borrow_dim: int,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
        suppress_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for SUB borrow detection: borrow = (a < b) ? 1 : 0.

        Only creates units for pairs where borrow occurs (120 units).

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for minuend
            b_dims: 16-dim one-hot for subtrahend
            borrow_dim: Output dimension for borrow flag
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating
            suppress_dim: Optional dimension to suppress at

        Returns:
            Number of units used (120 for pairs where a < b)
        """
        unit = unit_start

        for a in range(16):
            for b in range(16):
                if a < b:
                    n_cond = 2
                    if marker_dim is not None:
                        ffn.W_up.data[unit, marker_dim] = S
                        n_cond += 1
                    if suppress_dim is not None:
                        ffn.W_up.data[unit, suppress_dim] = -S * 2
                    ffn.W_up.data[unit, a_dims[a]] = S
                    ffn.W_up.data[unit, b_dims[b]] = S

                    ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                    if opcode_dim is not None:
                        ffn.W_gate.data[unit, opcode_dim] = 1.0
                    else:
                        ffn.b_gate.data[unit] = 1.0

                    # Scale for gate normalization
                    ffn.W_down.data[borrow_dim, unit] = 2.0 / (S * 5.0)

                    unit += 1

        return unit - unit_start

    @staticmethod
    def efficient_mul(
        ffn,
        unit_start: int,
        a_dim: int,
        b_dim: int,
        out_dim: int,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create efficient MUL using SwiGLU identity: a*b = silu(S*a)*b/S + silu(-S*a)*(-b)/S.

        This is the key insight from the blog post - SwiGLU can compute exact
        integer multiplication without lookup tables.

        Pattern:
            Unit 0 (positive path): silu(S*a) * b → out with scale 1/S
            Unit 1 (negative path): silu(-S*a) * (-b) → out with scale 1/S

        Together these compute a*b exactly for integers.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dim: Input A dimension (scalar value)
            b_dim: Input B dimension (scalar value)
            out_dim: Output dimension
            S: Scale factor (default 100.0)
            marker_dim: Optional marker for gating

        Returns:
            Number of units used (2)
        """
        unit = unit_start

        # Positive path: silu(S*a) * b
        if marker_dim is not None:
            ffn.W_up.data[unit, marker_dim] = S
            ffn.W_up.data[unit, a_dim] = S
            ffn.b_up.data[unit] = -S * 0.5  # Gate on marker
        else:
            ffn.W_up.data[unit, a_dim] = S
        ffn.W_gate.data[unit, b_dim] = 1.0
        ffn.W_down.data[out_dim, unit] = 1.0 / S
        unit += 1

        # Negative path: silu(-S*a) * (-b) for numerical stability
        if marker_dim is not None:
            ffn.W_up.data[unit, marker_dim] = S
            ffn.W_up.data[unit, a_dim] = -S
            ffn.b_up.data[unit] = -S * 0.5
        else:
            ffn.W_up.data[unit, a_dim] = -S
        ffn.W_gate.data[unit, b_dim] = -1.0
        ffn.W_down.data[out_dim, unit] = 1.0 / S
        unit += 1

        return unit - unit_start

    @staticmethod
    def efficient_mul_accumulate(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dim: int,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create efficient MUL with accumulation for schoolbook multiplication.

        Computes sum of products: out = sum(a[i] * b[i]) using SwiGLU identity.
        Used for partial products in multi-byte multiplication.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: List of A input dimensions
            b_dims: List of B input dimensions (same length as a_dims)
            out_dim: Output dimension (accumulated result)
            S: Scale factor
            marker_dim: Optional marker for gating

        Returns:
            Number of units used (2 per product pair)
        """
        assert len(a_dims) == len(b_dims), "Must have same number of A and B inputs"
        unit = unit_start

        for a_dim, b_dim in zip(a_dims, b_dims):
            # Positive path
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
                ffn.W_up.data[unit, a_dim] = S
                ffn.b_up.data[unit] = -S * 0.5
            else:
                ffn.W_up.data[unit, a_dim] = S
            ffn.W_gate.data[unit, b_dim] = 1.0
            ffn.W_down.data[out_dim, unit] = 1.0 / S
            unit += 1

            # Negative path
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
                ffn.W_up.data[unit, a_dim] = -S
                ffn.b_up.data[unit] = -S * 0.5
            else:
                ffn.W_up.data[unit, a_dim] = -S
            ffn.W_gate.data[unit, b_dim] = -1.0
            ffn.W_down.data[out_dim, unit] = 1.0 / S
            unit += 1

        return unit - unit_start

    @staticmethod
    def schoolbook_mul(
        ffn,
        unit_start: int,
        a_byte_dims: List[int],
        b_byte_dims: List[int],
        out_sum_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create schoolbook multiplication using efficient SwiGLU multiply.

        For N byte positions, computes partial products:
            pos 0: a0*b0
            pos 1: a0*b1 + a1*b0
            pos 2: a0*b2 + a1*b1 + a2*b0
            pos k: sum of a[i]*b[j] where i+j=k

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_byte_dims: Dimensions for A bytes [a0, a1, a2, a3]
            b_byte_dims: Dimensions for B bytes [b0, b1, b2, b3]
            out_sum_dims: Output dimensions for partial sums
            S: Scale factor
            marker_dim: Optional marker for gating

        Returns:
            Number of units used
        """
        N = len(a_byte_dims)
        assert len(b_byte_dims) == N, "A and B must have same byte count"
        assert len(out_sum_dims) >= N, "Need at least N output dims"

        unit = unit_start

        # For each output position k
        for k in range(N):
            # Compute products a[i]*b[j] where i+j=k
            for i in range(min(k + 1, N)):
                j = k - i
                if j < N:
                    a_dim = a_byte_dims[i]
                    b_dim = b_byte_dims[j]
                    out_dim = out_sum_dims[k]

                    # Positive path: silu(S*a) * b
                    if marker_dim is not None:
                        ffn.W_up.data[unit, marker_dim] = S
                        ffn.W_up.data[unit, a_dim] = S
                        ffn.b_up.data[unit] = -S * 0.5
                    else:
                        ffn.W_up.data[unit, a_dim] = S
                    ffn.W_gate.data[unit, b_dim] = 1.0
                    ffn.W_down.data[out_dim, unit] = 1.0 / S
                    unit += 1

                    # Negative path: silu(-S*a) * (-b)
                    if marker_dim is not None:
                        ffn.W_up.data[unit, marker_dim] = S
                        ffn.W_up.data[unit, a_dim] = -S
                        ffn.b_up.data[unit] = -S * 0.5
                    else:
                        ffn.W_up.data[unit, a_dim] = -S
                    ffn.W_gate.data[unit, b_dim] = -1.0
                    ffn.W_down.data[out_dim, unit] = 1.0 / S
                    unit += 1

        return unit - unit_start

    @staticmethod
    def mul_lookup(
        ffn,
        unit_start: int,
        a_dims: List[int],
        b_dims: List[int],
        out_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for MUL: out = (a * b) mod 16.

        MUL primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: 16x16 nibble lookup table for multiplication.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dims: 16-dim one-hot for first operand
            b_dims: 16-dim one-hot for second operand
            out_dims: 16-dim one-hot for output (mod 16)
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        for result in range(16):
            for a in range(16):
                for b in range(16):
                    if (a * b) % 16 == result:
                        n_cond = 2
                        if marker_dim is not None:
                            ffn.W_up.data[unit, marker_dim] = S
                            n_cond += 1
                        ffn.W_up.data[unit, a_dims[a]] = S
                        ffn.W_up.data[unit, b_dims[b]] = S

                        ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                        if opcode_dim is not None:
                            ffn.W_gate.data[unit, opcode_dim] = 1.0
                        else:
                            ffn.b_gate.data[unit] = 1.0

                        ffn.W_down.data[out_dims[result], unit] = 2.0 / S

                        unit += 1

        return unit - unit_start

    @staticmethod
    def div_lookup(
        ffn,
        unit_start: int,
        dividend_dims: List[int],
        divisor_dims: List[int],
        quotient_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for DIV: out = a // b.

        DIV primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: 16x16 nibble lookup table for integer division.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            dividend_dims: 16-dim one-hot for dividend
            divisor_dims: 16-dim one-hot for divisor
            quotient_dims: 16-dim one-hot for quotient output
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        for q in range(16):
            for a in range(16):
                for b in range(1, 16):  # Skip division by 0
                    if a // b == q:
                        n_cond = 2
                        if marker_dim is not None:
                            ffn.W_up.data[unit, marker_dim] = S
                            n_cond += 1
                        ffn.W_up.data[unit, dividend_dims[a]] = S
                        ffn.W_up.data[unit, divisor_dims[b]] = S

                        ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                        if opcode_dim is not None:
                            ffn.W_gate.data[unit, opcode_dim] = 1.0
                        else:
                            ffn.b_gate.data[unit] = 1.0

                        ffn.W_down.data[quotient_dims[q], unit] = 2.0 / S

                        unit += 1

        return unit - unit_start

    @staticmethod
    def mod_lookup(
        ffn,
        unit_start: int,
        dividend_dims: List[int],
        divisor_dims: List[int],
        remainder_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        opcode_dim: Optional[int] = None,
    ) -> int:
        """Create lookup table for MOD: out = a % b.

        MOD primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: 16x16 nibble lookup table for modulo.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            dividend_dims: 16-dim one-hot for dividend
            divisor_dims: 16-dim one-hot for divisor
            remainder_dims: 16-dim one-hot for remainder output
            S: Scale factor
            marker_dim: Optional marker for gating
            opcode_dim: Optional opcode flag for gating

        Returns:
            Number of units used
        """
        unit = unit_start

        for r in range(16):
            for a in range(16):
                for b in range(1, 16):  # Skip division by 0
                    if a % b == r:
                        n_cond = 2
                        if marker_dim is not None:
                            ffn.W_up.data[unit, marker_dim] = S
                            n_cond += 1
                        ffn.W_up.data[unit, dividend_dims[a]] = S
                        ffn.W_up.data[unit, divisor_dims[b]] = S

                        ffn.b_up.data[unit] = -S * (n_cond - 0.5)

                        if opcode_dim is not None:
                            ffn.W_gate.data[unit, opcode_dim] = 1.0
                        else:
                            ffn.b_gate.data[unit] = 1.0

                        ffn.W_down.data[remainder_dims[r], unit] = 2.0 / S

                        unit += 1

        return unit - unit_start

    @staticmethod
    def logical_xor(
        ffn,
        unit_start: int,
        a_dim: int,
        b_dim: int,
        output_dim: int,
        S: float = 100.0,
        marker_dim: Optional[int] = None,
    ) -> int:
        """Create units for logical XOR: out = a ^ b (where a, b are 0 or 1).

        XOR primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: (a + b) - 2*(a && b) = a XOR b.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            a_dim: Input A dimension (0 or 1)
            b_dim: Input B dimension (0 or 1)
            output_dim: Output dimension
            S: Scale factor
            marker_dim: Optional marker for gating

        Returns:
            Number of units used (4)
        """
        unit = unit_start

        # Unit 0-1: sum = a + b (cancel pair relay)
        if marker_dim is not None:
            ffn.W_up.data[unit, marker_dim] = S
            ffn.b_up.data[unit] = -S * 0.5
        else:
            ffn.b_up.data[unit] = S * 0.5
        ffn.W_gate.data[unit, a_dim] = 1.0
        ffn.W_gate.data[unit, b_dim] = 1.0
        ffn.W_down.data[output_dim, unit] = 2.0 / S
        unit += 1

        # Unit 2: -(a && b) -> subtract 2 when both are 1
        if marker_dim is not None:
            ffn.W_up.data[unit, marker_dim] = S
        ffn.W_up.data[unit, a_dim] = S
        ffn.W_up.data[unit, b_dim] = S
        n_cond = 3 if marker_dim else 2
        ffn.b_up.data[unit] = -S * (n_cond - 0.5)
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[output_dim, unit] = -2.0 * 2.0 / S  # -2 contribution
        unit += 1

        return unit - unit_start

    @staticmethod
    def move_relay(
        ffn,
        unit_start: int,
        src_dims: List[int],
        dst_dims: List[int],
        S: float = 100.0,
        marker_dim: Optional[int] = None,
        clear_src: bool = False,
    ) -> int:
        """Create units for MOVE/RELAY: dst = src.

        MOVE primitive from WEIGHT_COMPILER_PRIMITIVES.md.
        Pattern: Copy via W_gate relay.

        Args:
            ffn: FFN layer module
            unit_start: Starting unit index
            src_dims: Source dimensions
            dst_dims: Destination dimensions
            S: Scale factor
            marker_dim: Optional marker for gating
            clear_src: If True, also clear source (destructive move)

        Returns:
            Number of units used
        """
        assert len(src_dims) == len(dst_dims), "Source and dest dims must match"
        unit = unit_start

        for src_dim, dst_dim in zip(src_dims, dst_dims):
            if marker_dim is not None:
                ffn.W_up.data[unit, marker_dim] = S
                ffn.b_up.data[unit] = -S * 0.5
            else:
                ffn.b_up.data[unit] = S * 0.5

            ffn.W_gate.data[unit, src_dim] = 1.0
            ffn.W_down.data[dst_dim, unit] = 2.0 / S

            if clear_src:
                ffn.W_down.data[src_dim, unit] = -2.0 / S

            unit += 1

        return unit - unit_start


# Convenience aliases
P = Primitives
