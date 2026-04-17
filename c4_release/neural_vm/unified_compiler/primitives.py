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


# Convenience aliases
P = Primitives
