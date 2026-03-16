"""
Test: SwiGLU weight-baked division pipeline (4 layers).

Validates the actual PyTorch SwiGLU forward pass with baked weights
matches the fp64 MAGIC floor trick test results.

Layer 3 (FloorExtractionFFN): 25 hidden units
  h=0..7:  floor(Q/16^j) via MAGIC + round trick
  h=8:     MAGIC cancellation (shared constant unit)
  h=9..24: clear old RESULT slots

Layer 4 (NibbleSubtractFFN): 14 hidden units
  h=0..13: for j=0..6, subtract 16*RESULT[j+1] from RESULT[j]
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import sys

sys.path.insert(0, '.')
from neural_vm.embedding import E, Opcode

# Slot assignments (from long_division_ops.py)
SLOT_DIVIDEND = E.TEMP
SLOT_DIVISOR = E.TEMP + 1
SLOT_REMAINDER = E.TEMP + 2
SLOT_QUOTIENT = E.TEMP + 3
SLOT_CURR_Q = E.TEMP + 4

MAGIC = 3.0 * 2**51  # 6755399441055744.0


# ============================================================
# SwiGLU forward (from base_layers.py PureFFN)
# ============================================================

def swiglu_forward(x, W_up, b_up, W_gate, b_gate, W_down, b_down):
    """Standard SwiGLU: output = x + W_down @ (silu(W_up @ x + b_up) * (W_gate @ x + b_gate)) + b_down"""
    up = F.linear(x, W_up, b_up)
    gate = F.linear(x, W_gate, b_gate)
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, W_down, b_down)


# ============================================================
# Build FloorExtractionFFN weights (Layer 3)
# ============================================================

def build_floor_weights(opcode):
    """Build weight matrices for FloorExtractionFFN."""
    flat_dim = E.NUM_POSITIONS * E.DIM
    hidden_dim = 25

    def flat_idx(pos, slot):
        return pos * E.DIM + slot

    W_up = torch.zeros(hidden_dim, flat_dim, dtype=torch.float64)
    b_up = torch.zeros(hidden_dim, dtype=torch.float64)
    W_gate = torch.zeros(hidden_dim, flat_dim, dtype=torch.float64)
    b_gate = torch.zeros(hidden_dim, dtype=torch.float64)
    W_down = torch.zeros(flat_dim, hidden_dim, dtype=torch.float64)
    b_down = torch.zeros(flat_dim, dtype=torch.float64)

    S = E.SCALE
    C = float(2**20)
    opcode_idx = flat_idx(0, E.OP_START + opcode)
    q_float_idx = flat_idx(0, SLOT_REMAINDER)

    # h=0..7: Floor extraction via MAGIC trick
    for j in range(8):
        h = j
        eps_j = 2.0 ** (-20 - 4 * j)
        scale_j = 1.0 / 16**j
        offset_j = -(0.5 - eps_j)
        result_j_idx = flat_idx(j, E.RESULT)

        W_up[h, q_float_idx] = scale_j
        W_up[h, opcode_idx] = offset_j
        b_up[h] = MAGIC
        W_gate[h, opcode_idx] = 1.0
        W_down[result_j_idx, h] = 1.0

    # h=8: MAGIC cancellation unit
    h = 8
    W_up[h, opcode_idx] = C
    W_gate[h, opcode_idx] = 1.0
    for j in range(8):
        result_j_idx = flat_idx(j, E.RESULT)
        W_down[result_j_idx, h] = -MAGIC / C  # = -3 * 2^31

    # h=9..24: Clear old RESULT values
    for pos in range(8):
        result_pos_idx = flat_idx(pos, E.RESULT)

        h = 9 + pos * 2
        W_up[h, opcode_idx] = S
        W_gate[h, result_pos_idx] = -1.0
        W_down[result_pos_idx, h] = 1.0 / S

        h = 9 + pos * 2 + 1
        W_up[h, opcode_idx] = -S
        W_gate[h, result_pos_idx] = 1.0
        W_down[result_pos_idx, h] = 1.0 / S

    return W_up, b_up, W_gate, b_gate, W_down, b_down


# ============================================================
# Build NibbleSubtractFFN weights (Layer 4)
# ============================================================

def build_nibble_weights(opcode):
    """Build weight matrices for NibbleSubtractFFN."""
    flat_dim = E.NUM_POSITIONS * E.DIM
    hidden_dim = 14

    def flat_idx(pos, slot):
        return pos * E.DIM + slot

    W_up = torch.zeros(hidden_dim, flat_dim, dtype=torch.float64)
    b_up = torch.zeros(hidden_dim, dtype=torch.float64)
    W_gate = torch.zeros(hidden_dim, flat_dim, dtype=torch.float64)
    b_gate = torch.zeros(hidden_dim, dtype=torch.float64)
    W_down = torch.zeros(flat_dim, hidden_dim, dtype=torch.float64)
    b_down = torch.zeros(flat_dim, dtype=torch.float64)

    S = E.SCALE
    opcode_idx = flat_idx(0, E.OP_START + opcode)

    for j in range(7):
        result_j_idx = flat_idx(j, E.RESULT)
        result_jp1_idx = flat_idx(j + 1, E.RESULT)

        h = j * 2
        W_up[h, opcode_idx] = S
        W_gate[h, result_jp1_idx] = 1.0
        W_down[result_j_idx, h] = -16.0 / S

        h = j * 2 + 1
        W_up[h, opcode_idx] = -S
        W_gate[h, result_jp1_idx] = -1.0
        W_down[result_j_idx, h] = -16.0 / S

    return W_up, b_up, W_gate, b_gate, W_down, b_down


# ============================================================
# Softmax1 reciprocal (from test_fp64_floor_swiglu.py)
# ============================================================

def softmax1_reciprocal(divisor):
    n = int(divisor)
    nibbles = []
    for j in range(8):
        nibbles.append(n & 0xF)
        n >>= 4
    S = 60.0
    scores = []
    for j, d in enumerate(nibbles):
        if d > 0:
            scores.append(np.log(np.float64(16**j * d)))
        else:
            scores.append(-S)
    scores = np.array(scores, dtype=np.float64)
    mx = np.max(scores)
    ex = np.exp(scores - mx)
    return np.float64(np.exp(-mx) / np.sum(ex))


# ============================================================
# Full pipeline test
# ============================================================

def full_swiglu_div(dividend, divisor, opcode=Opcode.DIV):
    """Run the full 4-layer SwiGLU division pipeline."""
    flat_dim = E.NUM_POSITIONS * E.DIM

    def flat_idx(pos, slot):
        return pos * E.DIM + slot

    # Create input embedding
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM, dtype=torch.float64)

    # Set operand nibbles
    a, b = int(dividend), int(divisor)
    for j in range(8):
        x[0, j, E.NIB_A] = (a >> (4 * j)) & 0xF
        x[0, j, E.NIB_B] = (b >> (4 * j)) & 0xF

    # Set opcode one-hot
    x[0, 0, E.OP_START + opcode] = 1.0

    # --- Layer 1: Gather dividend + compute reciprocal ---
    # (simulated — these are custom modules, not weight-based FFNs)
    dividend_val = float(dividend)
    reciprocal = float(softmax1_reciprocal(divisor))
    x[0, 0, SLOT_DIVIDEND] = dividend_val
    x[0, 0, SLOT_QUOTIENT] = reciprocal

    # --- Layer 2: Multiply ---
    q_float = dividend_val * reciprocal
    x[0, 0, SLOT_REMAINDER] = q_float

    # --- Layer 3: Floor extraction via SwiGLU ---
    weights3 = build_floor_weights(opcode)
    x_flat = x.reshape(1, 1, flat_dim)
    x_flat = swiglu_forward(x_flat, *weights3)
    x = x_flat.reshape(1, E.NUM_POSITIONS, E.DIM)

    # --- Layer 4: Nibble subtraction via SwiGLU ---
    weights4 = build_nibble_weights(opcode)
    x_flat = x.reshape(1, 1, flat_dim)
    x_flat = swiglu_forward(x_flat, *weights4)
    x = x_flat.reshape(1, E.NUM_POSITIONS, E.DIM)

    # Read result nibbles
    result = 0
    for j in range(8):
        nib = int(round(x[0, j, E.RESULT].item()))
        result += nib * 16**j

    return result


# ============================================================
# Tests
# ============================================================

print("=" * 72)
print("  SwiGLU Division Pipeline Test (PyTorch forward pass)")
print("=" * 72)

# Test 1: Targeted cases
print("\n  Test 1: Targeted cases")
print("  " + "-" * 68)

targeted = [
    (42, 7, 6), (100, 10, 10), (1000, 33, 30),
    (255, 16, 15), (65535, 255, 257), (999999, 7, 142857),
    (0, 5, 0), (5, 1, 5), (15, 15, 1),
    (1000000, 1000000, 1),
    (2**32-1, 1, 2**32-1),
    (2**32-1, 2, 2**31-1),
    (2**31-1, 3, (2**31-1)//3),
    (2**32-1, 17, (2**32-1)//17),
    (2946529189, 1473538672, 1),  # near-integer
    (3066343536, 1022218630, 2),  # near-integer
    (1436260363, 1436925100, 0),  # near-integer
]
t_fail = 0
for a, b, exp in targeted:
    r = full_swiglu_div(a, b)
    ok = "OK" if r == exp else "FAIL"
    if r != exp:
        t_fail += 1
    print(f"    {ok}: {a}/{b} = {r}" + (f" (expected {exp})" if r != exp else ""))
print(f"\n    Targeted: {t_fail}/{len(targeted)} failures")


# Test 2: Random stress test
print(f"\n  Test 2: Random stress test (10k)")
print("  " + "-" * 68)

random.seed(42)
r_fail = 0
fail_ex = []
for _ in range(10000):
    a = random.randint(0, 2**32-1)
    b = random.randint(1, 2**32-1)
    exp = a // b
    r = full_swiglu_div(a, b)
    if r != exp:
        r_fail += 1
        if len(fail_ex) < 5:
            fail_ex.append((a, b, r, exp))
for a, b, r, e in fail_ex:
    print(f"    FAIL: {a}/{b} = {r} (expected {e})")
print(f"    Result: {r_fail}/10000 failures")


# Test 3: Small divisor sweep
print(f"\n  Test 3: Small divisor sweep (1-50)")
print("  " + "-" * 68)

s_fail = 0
for b in range(1, 51):
    for a in list(range(0, 100)) + [b*k for k in range(1, 100)] + [2**32-1, 2**31-1]:
        if a > 2**32-1:
            continue
        exp = a // b
        r = full_swiglu_div(a, b)
        if r != exp:
            s_fail += 1
            if s_fail <= 5:
                print(f"    FAIL: {a}/{b} = {r} (expected {exp})")
print(f"    Small divisor sweep: {s_fail} failures")


# Test 4: Verify floor values directly
print(f"\n  Test 4: Floor extraction precision")
print("  " + "-" * 68)

flat_dim = E.NUM_POSITIONS * E.DIM
opcode = Opcode.DIV
weights3 = build_floor_weights(opcode)

f_fail = 0
for Q in [0, 1, 5, 15, 16, 255, 256, 65535, 2**20, 2**31-1, 2**32-1]:
    # Build input with Q_float in SLOT_REMAINDER
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM, dtype=torch.float64)
    x[0, 0, E.OP_START + opcode] = 1.0
    x[0, 0, SLOT_REMAINDER] = float(Q)

    x_flat = x.reshape(1, 1, flat_dim)
    y_flat = swiglu_forward(x_flat, *weights3)
    y = y_flat.reshape(1, E.NUM_POSITIONS, E.DIM)

    for j in range(8):
        got = y[0, j, E.RESULT].item()
        expected = Q // (16**j)
        if abs(got - expected) > 0.5:
            f_fail += 1
            if f_fail <= 5:
                print(f"    FAIL: Q={Q} j={j}: got {got}, expected {expected}")
print(f"    Floor extraction: {f_fail} failures")


# Summary
print(f"\n{'=' * 72}")
total = t_fail + r_fail + s_fail + f_fail
print(f"  TOTAL: {total} failures")
print(f"  DIV pipeline: 4 layers (was 11)")
print(f"  Hidden units: 25 (floor) + 14 (nibble) = 39 total")
print(f"{'=' * 72}")
