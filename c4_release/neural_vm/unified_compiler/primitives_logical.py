"""
Logical simulator for weight compiler primitives.

This module implements the same operations as primitives.py but as pure functions
operating on values instead of weights. This enables:
1. Unit testing of declarative logic without running the neural network
2. Fast iteration during debugging
3. Clear separation between spec correctness and weight implementation

Each function corresponds to a primitive in primitives.py:
- threshold_attention_logical -> threshold_attention
- carry_forward_attention_logical -> carry_forward_attention
- swiglu_and_gate_logical -> swiglu_and_gate
- etc.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class HiddenState:
    """Simulated hidden state vector as a sparse dict.

    Keys are dimension names (strings or ints), values are floats.
    Missing keys default to 0.0.
    """
    values: Dict[Union[str, int], float] = field(default_factory=dict)

    def __getitem__(self, key: Union[str, int]) -> float:
        return self.values.get(key, 0.0)

    def __setitem__(self, key: Union[str, int], value: float):
        self.values[key] = value

    def get_nibble_value(self, base: Union[str, int], offset: int = 0) -> int:
        """Get integer value from one-hot nibble encoding at base+offset."""
        for k in range(16):
            key = base + k if isinstance(base, int) else f"{base}_{k}"
            if self.values.get(key, 0.0) > 0.5:
                return k
        return 0

    def set_nibble(self, base: Union[str, int], value: int):
        """Set one-hot nibble encoding at base."""
        value = value & 0xF  # Clamp to 0-15
        for k in range(16):
            key = base + k if isinstance(base, int) else f"{base}_{k}"
            self.values[key] = 1.0 if k == value else 0.0

    def get_byte_value(self, base_lo: Union[str, int], base_hi: Union[str, int]) -> int:
        """Get byte value from two nibbles."""
        lo = self.get_nibble_value(base_lo)
        hi = self.get_nibble_value(base_hi)
        return (hi << 4) | lo

    def set_byte(self, base_lo: Union[str, int], base_hi: Union[str, int], value: int):
        """Set byte value as two nibbles."""
        value = value & 0xFF
        self.set_nibble(base_lo, value & 0xF)
        self.set_nibble(base_hi, (value >> 4) & 0xF)

    def copy(self) -> "HiddenState":
        """Create a shallow copy."""
        return HiddenState(dict(self.values))


def silu(x: float) -> float:
    """SiLU activation: x * sigmoid(x)."""
    return x / (1.0 + math.exp(-x)) if x > -500 else 0.0


# =============================================================================
# Attention Primitives (Logical)
# =============================================================================

def threshold_attention_logical(
    positions: List[HiddenState],
    query_pos: int,
    threshold: float,
    marker_dim: str = "IS_MARK",
    output_dims: List[str] = None,
    slope: float = 10.0,
) -> Dict[str, float]:
    """Logical simulation of threshold attention.

    Detects whether the nearest marker is within `threshold` positions.
    Returns output values that would be written.

    Args:
        positions: List of hidden states for each position
        query_pos: Position to compute attention for
        threshold: Distance threshold
        marker_dim: Dimension indicating marker positions
        output_dims: Output dimension names
        slope: ALiBi slope

    Returns:
        Dict of output dimension -> value
    """
    result = {}

    # Find best matching key position
    best_score = float('-inf')
    best_pos = None

    for k_pos in range(query_pos + 1):  # Can only attend to past
        if positions[k_pos][marker_dim] > 0.5:
            distance = query_pos - k_pos
            score = slope * (threshold - distance)
            if score > best_score:
                best_score = score
                best_pos = k_pos

    # If we found a marker within threshold, copy its values
    if best_pos is not None and best_score > 0:
        if output_dims:
            for dim in output_dims:
                result[dim] = positions[best_pos][dim]

    return result


def carry_forward_attention_logical(
    positions: List[HiddenState],
    query_pos: int,
    marker_dim: str,
    l1h1_idx: str,
    l1h0_idx: str,
    src_lo: str,
    src_hi: str,
) -> Tuple[List[float], List[float]]:
    """Logical simulation of carry-forward attention.

    At marker positions, attends to previous step's byte 0 (L1H1 AND NOT L1H0).

    Args:
        positions: List of hidden states
        query_pos: Position to compute attention for
        marker_dim: Query fires on this marker
        l1h1_idx: Key dimension for L1H1
        l1h0_idx: Key dimension for L1H0
        src_lo: Source low nibble base
        src_hi: Source high nibble base

    Returns:
        Tuple of (lo_nibble_16, hi_nibble_16) as lists of 16 floats
    """
    lo_out = [0.0] * 16
    hi_out = [0.0] * 16

    # Only fire if query is at marker position
    if positions[query_pos][marker_dim] < 0.5:
        return lo_out, hi_out

    # Find best matching key: L1H1=1 AND L1H0=0 (previous step's byte 0)
    best_score = float('-inf')
    best_pos = None

    for k_pos in range(query_pos):  # Must be in past
        l1h1 = positions[k_pos][l1h1_idx]
        l1h0 = positions[k_pos][l1h0_idx]
        score = l1h1 - l1h0  # High when L1H1=1, L1H0=0
        if score > best_score and l1h1 > 0.5 and l1h0 < 0.5:
            best_score = score
            best_pos = k_pos

    if best_pos is not None:
        for k in range(16):
            lo_out[k] = positions[best_pos][f"{src_lo}_{k}" if isinstance(src_lo, str) else src_lo + k]
            hi_out[k] = positions[best_pos][f"{src_hi}_{k}" if isinstance(src_hi, str) else src_hi + k]

    return lo_out, hi_out


def relay_head_logical(
    positions: List[HiddenState],
    query_pos: int,
    q_marker: str,
    k_source: str,
    v_dims: List[str],
) -> Dict[str, float]:
    """Logical simulation of relay head.

    Q fires at marker, attends to K source position, copies V dimensions.

    Args:
        positions: List of hidden states
        query_pos: Position to compute attention for
        q_marker: Query marker dimension
        k_source: Key source dimension
        v_dims: Value dimensions to copy

    Returns:
        Dict of output values
    """
    result = {}

    # Only fire if query is at marker position
    if positions[query_pos][q_marker] < 0.5:
        return result

    # Find best matching key position
    best_score = float('-inf')
    best_pos = None

    for k_pos in range(query_pos + 1):
        score = positions[k_pos][k_source]
        if score > best_score:
            best_score = score
            best_pos = k_pos

    if best_pos is not None and best_score > 0.5:
        for dim in v_dims:
            result[dim] = positions[best_pos][dim]

    return result


# =============================================================================
# FFN Primitives (Logical)
# =============================================================================

def swiglu_and_gate_logical(
    state: HiddenState,
    up_dims: List[Tuple[str, float]],
    threshold: float,
    gate_dims: Optional[List[Tuple[str, float]]] = None,
    gate_bias: float = 1.0,
    S: float = 100.0,
) -> float:
    """Logical simulation of SwiGLU AND gate.

    Computes: silu(W_up @ x + b_up) * (W_gate @ x + b_gate)

    Args:
        state: Hidden state values
        up_dims: List of (dim, weight) for activation
        threshold: Threshold value (sum must exceed this to fire)
        gate_dims: List of (dim, weight) for gate (None for constant gate)
        gate_bias: Gate bias
        S: Scale factor

    Returns:
        Activation value (pre-W_down scaling)
    """
    # Compute up activation
    up_sum = sum(state[dim] * weight for dim, weight in up_dims)
    up_activation = silu(S * (up_sum - threshold))

    # Compute gate value
    if gate_dims:
        gate_value = sum(state[dim] * weight for dim, weight in gate_dims) + gate_bias
    else:
        gate_value = gate_bias

    return up_activation * gate_value


def threshold_match_logical(
    state: HiddenState,
    conditions: List[Tuple[str, float]],
    threshold: float,
    gate_dims: Optional[List[Tuple[str, float]]] = None,
    gate_bias: float = 1.0,
) -> bool:
    """Logical simulation of threshold match.

    Returns True if weighted sum of conditions exceeds threshold AND gate fires.

    Args:
        state: Hidden state values
        conditions: List of (dim, weight) conditions
        threshold: Sum must exceed this
        gate_dims: Optional gate conditions
        gate_bias: Gate bias

    Returns:
        Whether the match fires
    """
    cond_sum = sum(state[dim] * weight for dim, weight in conditions)
    if cond_sum < threshold:
        return False

    if gate_dims:
        gate_value = sum(state[dim] * weight for dim, weight in gate_dims) + gate_bias
        return gate_value > 0

    return True


def step_pair_logical(
    state: HiddenState,
    input_dims: List[Tuple[str, float]],
    low_threshold: float,
    high_threshold: float,
) -> float:
    """Logical simulation of step pair.

    Returns: step(sum >= low) - step(sum >= high)
    Result is 1.0 if sum in [low, high), else 0.0

    Args:
        state: Hidden state values
        input_dims: List of (dim, weight) inputs
        low_threshold: Lower threshold
        high_threshold: Upper threshold

    Returns:
        1.0 if in range, 0.0 otherwise
    """
    total = sum(state[dim] * weight for dim, weight in input_dims)

    above_low = 1.0 if total >= low_threshold else 0.0
    above_high = 1.0 if total >= high_threshold else 0.0

    return above_low - above_high


def nibble_copy_logical(
    state: HiddenState,
    src_lo: str,
    src_hi: str,
    gate_dim: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """Logical simulation of nibble copy.

    Copies one-hot nibbles if gate fires.

    Args:
        state: Hidden state values
        src_lo: Source low nibble base
        src_hi: Source high nibble base
        gate_dim: Optional gate dimension

    Returns:
        Tuple of (lo_nibble_16, hi_nibble_16)
    """
    # Check gate
    if gate_dim and state[gate_dim] < 0.5:
        return [0.0] * 16, [0.0] * 16

    lo_out = []
    hi_out = []

    for k in range(16):
        lo_key = f"{src_lo}_{k}" if isinstance(src_lo, str) else src_lo + k
        hi_key = f"{src_hi}_{k}" if isinstance(src_hi, str) else src_hi + k
        lo_out.append(state[lo_key])
        hi_out.append(state[hi_key])

    return lo_out, hi_out


def cancel_pair_logical(
    state: HiddenState,
    old_dims: List[str],
    new_dims: List[str],
) -> float:
    """Logical simulation of cancel pair.

    Returns: sum(-old_dims) + sum(new_dims)

    Args:
        state: Hidden state values
        old_dims: Dimensions to cancel
        new_dims: Dimensions to add

    Returns:
        Net value
    """
    old_sum = sum(state[dim] for dim in old_dims)
    new_sum = sum(state[dim] for dim in new_dims)
    return new_sum - old_sum


# =============================================================================
# ALU Operations (Logical)
# =============================================================================

def add_lookup_logical(op_a: int, op_b: int) -> Tuple[int, bool]:
    """Logical ADD with carry detection.

    Returns:
        Tuple of (result_byte, carry_out)
    """
    result = op_a + op_b
    return result & 0xFF, result > 0xFF


def sub_lookup_logical(op_a: int, op_b: int) -> Tuple[int, bool]:
    """Logical SUB with borrow detection.

    Returns:
        Tuple of (result_byte, borrow_out)
    """
    result = op_a - op_b
    return result & 0xFF, result < 0


def bitwise_and_logical(op_a: int, op_b: int) -> int:
    """Logical bitwise AND."""
    return op_a & op_b


def bitwise_or_logical(op_a: int, op_b: int) -> int:
    """Logical bitwise OR."""
    return op_a | op_b


def bitwise_xor_logical(op_a: int, op_b: int) -> int:
    """Logical bitwise XOR."""
    return op_a ^ op_b


def shift_left_logical(value: int, shift: int) -> int:
    """Logical shift left."""
    return (value << (shift & 0x1F)) & 0xFFFFFFFF


def shift_right_logical(value: int, shift: int) -> int:
    """Logical shift right."""
    return (value >> (shift & 0x1F)) & 0xFFFFFFFF


# =============================================================================
# Carry Propagation (Logical)
# =============================================================================

def carry_propagation_logical(
    byte_value: int,
    carry_in: bool,
    op_type: str,
) -> Tuple[int, bool]:
    """Logical carry/borrow propagation.

    For ADD: if carry_in, result = byte_value + 1, carry_out = (result > 255)
    For SUB: if borrow_in, result = byte_value - 1, borrow_out = (result < 0)

    Args:
        byte_value: Current byte value (0-255)
        carry_in: Whether there's a carry/borrow from previous byte
        op_type: 'ADD' or 'SUB'

    Returns:
        Tuple of (adjusted_byte, carry/borrow_out)
    """
    if op_type == 'ADD':
        if carry_in:
            result = byte_value + 1
            return result & 0xFF, result > 0xFF
        return byte_value, False

    elif op_type == 'SUB':
        if carry_in:  # borrow_in
            result = byte_value - 1
            return result & 0xFF, result < 0
        return byte_value, False

    return byte_value, False


# =============================================================================
# Multi-byte ALU (Logical)
# =============================================================================

def multibyte_add_logical(a_bytes: List[int], b_bytes: List[int]) -> List[int]:
    """Logical multi-byte ADD with carry propagation.

    Args:
        a_bytes: List of bytes [byte0, byte1, byte2, byte3] (little-endian)
        b_bytes: List of bytes [byte0, byte1, byte2, byte3]

    Returns:
        Result bytes [byte0, byte1, byte2, byte3]
    """
    result = []
    carry = False

    for a, b in zip(a_bytes, b_bytes):
        # Add with carry-in
        total = a + b + (1 if carry else 0)
        result.append(total & 0xFF)
        carry = total > 0xFF

    return result


def multibyte_sub_logical(a_bytes: List[int], b_bytes: List[int]) -> List[int]:
    """Logical multi-byte SUB with borrow propagation.

    Args:
        a_bytes: List of bytes [byte0, byte1, byte2, byte3] (little-endian)
        b_bytes: List of bytes [byte0, byte1, byte2, byte3]

    Returns:
        Result bytes [byte0, byte1, byte2, byte3]
    """
    result = []
    borrow = False

    for a, b in zip(a_bytes, b_bytes):
        # Subtract with borrow-in
        diff = a - b - (1 if borrow else 0)
        result.append(diff & 0xFF)
        borrow = diff < 0

    return result


# =============================================================================
# Test helpers
# =============================================================================

def create_test_state(**kwargs) -> HiddenState:
    """Create a HiddenState with initial values.

    Args:
        **kwargs: Dimension name -> value pairs

    Returns:
        HiddenState with values set
    """
    state = HiddenState()
    for key, value in kwargs.items():
        state[key] = value
    return state


def verify_primitive_equivalence(
    weight_result: float,
    logical_result: float,
    tolerance: float = 0.01,
) -> bool:
    """Verify that weight-based and logical results match.

    Args:
        weight_result: Result from running through actual weights
        logical_result: Result from logical simulation
        tolerance: Maximum allowed difference

    Returns:
        True if results match within tolerance
    """
    return abs(weight_result - logical_result) < tolerance
