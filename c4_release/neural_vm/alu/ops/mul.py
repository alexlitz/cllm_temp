"""
Chunk-generic MUL pipeline: schoolbook + carry extraction + lookahead.

Supported configs: BIT, PAIR, NIBBLE, BYTE.
HALFWORD/WORD impractical (step pairs scale as N * chunk_max^2 / base).

Layer structure:
  1. SchoolbookFFN: N*(N+1)/2 partial products (flattened)
  2..2+P-1. CarryPassFFN: P passes of carry extraction (flattened)
  2+P. MulGenPropFFN: G/P for binary carry (flattened)
  3+P. MulBinaryLookaheadFFN: prefix carry computation (flattened)
  4+P. MulFinalCorrectionFFN: add carry, mod base, clear (flattened)

Number of carry passes P depends on config:
  BIT(k=1): 5, PAIR(k=2): 4, NIBBLE(k=4): 3, BYTE(k=8): 3
"""

import torch
import torch.nn as nn

from ..chunk_config import ChunkConfig
from .common import (
    GenericE, GenericPureFFN, GenericFlattenedFFN,
    bake_clear_pair,
)


def _compute_carry_passes(config: ChunkConfig):
    """Compute the max carry at each pass to determine number of passes.

    After schoolbook, max value at any position is N * chunk_max^2.
    Each pass extracts carry = floor(val / base), remainder = val mod base.
    Carry from position i goes to position i+1 in the NEXT pass.
    So after adding carry from left neighbor, new max = base-1 + max_carry_prev.

    Continue until the CARRY from the last pass is <= 1, so that
    GenPropFFN's combined value (RESULT + carry_from_left) is <= base,
    which is required for binary carry-lookahead correctness.
    """
    N = config.num_positions
    base = config.base
    chunk_max = config.chunk_max

    # Max schoolbook value at any position
    max_val = N * chunk_max * chunk_max

    passes = []
    while True:
        max_carry = max_val // base
        if max_carry == 0:
            # No carry at all - nothing for carry-lookahead either
            break
        passes.append(max_carry)
        if max_carry <= 1:
            # After this pass, carry <= 1. GenPropFFN combined <= base. OK.
            break
        # Next pass: remainder is 0..base-1, plus carry from left 0..max_carry
        max_val = (base - 1) + max_carry

    return passes


class SchoolbookFFN(nn.Module):
    """Schoolbook multiplication: all N*(N+1)/2 partial products.

    result[k] = sum_{i+j=k, 0<=i,j<N} a[i] * b[j]
    Products at positions >= N are discarded (overflow beyond total_bits).

    source_a: slot to read first operand from (default NIB_A).
              For MOD: RESULT (reads quotient from division phase).
    """

    def __init__(self, ge: GenericE, opcode: int, source_a=None):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        dtype = ge.config.torch_dtype

        if source_a is None:
            source_a = ge.NIB_A

        # Count partial products (only for output positions 0..N-1)
        num_products = 0
        for k in range(N):
            for i in range(k + 1):
                j = k - i
                if i < N and j < N:
                    num_products += 1

        hidden_dim = num_products * 2 + N * 2  # products + clear RESULT

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            h = 0

            # Clear RESULT at all positions
            for pos in range(N):
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(pos, ge.OP_START + opcode),
                                fi(pos, ge.RESULT), S)
                h += 2

            # Partial products
            for k in range(N):
                for i in range(k + 1):
                    j = k - i
                    if i < N and j < N:
                        # a[i] * b[j] → RESULT[k]
                        self.flat_ffn.W_up[h, fi(i, source_a)] = S
                        self.flat_ffn.W_gate[h, fi(j, ge.NIB_B)] = 1.0
                        self.flat_ffn.W_down[fi(k, ge.RESULT), h] = 1.0 / S
                        h += 1

                        self.flat_ffn.W_up[h, fi(i, source_a)] = -S
                        self.flat_ffn.W_gate[h, fi(j, ge.NIB_B)] = -1.0
                        self.flat_ffn.W_down[fi(k, ge.RESULT), h] = 1.0 / S
                        h += 1

            assert h == hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class CarryPassFFN(nn.Module):
    """One pass of carry extraction via step pairs.

    For each position: step(RESULT >= k*base) for k=1..max_carry.
    Each step: -base to RESULT, +1 to CARRY_OUT.
    Then: add CARRY_OUT[pos-1] to RESULT[pos] and clear CARRY_OUT.

    If not last_pass: reads CARRY_OUT from input (previous pass),
    adds it to RESULT, then extracts new carry.
    If first pass (pass_idx==0): no incoming carry to add.
    """

    def __init__(self, ge: GenericE, opcode: int, max_carry: int,
                 pass_idx: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        base = ge.BASE
        dtype = ge.config.torch_dtype

        # Use fp64 when step pair products could exceed fp32 exact range.
        # Max silu input ≈ S * max_value. Need S * max_value < 2^23 for fp32.
        max_value = max_carry * base + base - 1
        if dtype == torch.float32 and S * max_value > 2**23:
            dtype = torch.float64
        self.needs_upcast = (dtype != ge.config.torch_dtype)

        # Hidden units: carry add (if not first) + step pairs + clear carry
        add_units = 2 * (N - 1) if pass_idx > 0 else 0
        step_units = 2 * max_carry * N
        clear_units = 2 * N
        hidden_dim = add_units + step_units + clear_units

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down
            op_gate = fi(0, ge.OP_START + opcode)

            h = 0

            # Add incoming carry from previous pass (CARRY_OUT[pos-1] → RESULT[pos])
            if pass_idx > 0:
                for pos in range(1, N):
                    carry_from = fi(pos - 1, ge.CARRY_OUT)
                    result_idx = fi(pos, ge.RESULT)

                    W_up[h, carry_from] = S
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

                    W_up[h, carry_from] = -S
                    W_gate[h, op_gate] = -1.0
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

            # Clear CARRY_OUT at all positions (before writing new carries)
            if pass_idx > 0:
                for pos in range(N):
                    bake_clear_pair(self.flat_ffn.ffn, h,
                                    fi(pos, ge.OP_START + opcode),
                                    fi(pos, ge.CARRY_OUT), S)
                    h += 2

            # Step pairs: extract carry from RESULT (+ incoming carry from input)
            for pos in range(N):
                result_idx = fi(pos, ge.RESULT)
                carry_idx = fi(pos, ge.CARRY_OUT)

                for k in range(1, max_carry + 1):
                    threshold = k * base

                    # Rise
                    W_up[h, result_idx] = S
                    if pass_idx > 0 and pos > 0:
                        W_up[h, fi(pos - 1, ge.CARRY_OUT)] = S
                    b_up[h] = -S * (threshold - 1)
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = -float(base) / S
                    W_down[carry_idx, h] = 1.0 / S
                    h += 1

                    # Saturation
                    W_up[h, result_idx] = S
                    if pass_idx > 0 and pos > 0:
                        W_up[h, fi(pos - 1, ge.CARRY_OUT)] = S
                    b_up[h] = -S * float(threshold)
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = float(base) / S
                    W_down[carry_idx, h] = -1.0 / S
                    h += 1

            assert h <= hidden_dim, f"Used {h} hidden units, allocated {hidden_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.needs_upcast:
            orig_dtype = x.dtype
            return self.flat_ffn(x.to(torch.float64)).to(orig_dtype)
        return self.flat_ffn(x)


class MulGenPropFFN(nn.Module):
    """Compute G/P for binary carry chain after final carry pass.

    Adds incoming carry from last pass, then:
    G[pos] = step(RESULT[pos] + carry_in >= base) → CARRY_OUT
    P[pos] = step(RESULT[pos] + carry_in == base-1) → TEMP
    Also applies mod base and clears CARRY_OUT from previous pass.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        base = ge.BASE
        dtype = ge.config.torch_dtype

        # Per position: 2 (add carry) + 2 (G step pair) + 2 (P step pair) = 6
        # Pos 0: no carry add, just 2 for P (G=0 since RESULT[0] < base after passes)
        # + 2*N (clear CARRY_OUT from prev pass) + 2*N (clear incoming CARRY_IN not used here)
        # Actually: use CARRY_OUT for incoming carry from last pass
        # pos > 0: 2 add + 4 G+P (sharing G step pair output to TEMP)
        # pos == 0: 2 for P only
        hidden_dim = 2 + 6 * (N - 1) + 2 * N
        # 2 for pos0 P, 6*(N-1) for pos>0, 2*N for clearing old CARRY_OUT

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down
            op_gate = fi(0, ge.OP_START + opcode)

            h = 0

            for pos in range(N):
                result_idx = fi(pos, ge.RESULT)

                if pos > 0:
                    carry_from = fi(pos - 1, ge.CARRY_OUT)

                    # Add CARRY_OUT[pos-1] to RESULT[pos]
                    W_up[h, carry_from] = S
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

                    W_up[h, carry_from] = -S
                    W_gate[h, op_gate] = -1.0
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

                    # G = step(RESULT + carry >= base) → CARRY_OUT, -base to RESULT
                    g_rise = h
                    W_up[h, result_idx] = S
                    W_up[h, carry_from] = S
                    b_up[h] = -S * float(base - 1)
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = -float(base) / S
                    W_down[fi(pos, ge.CARRY_OUT), h] = 1.0 / S
                    h += 1

                    g_sat = h
                    W_up[h, result_idx] = S
                    W_up[h, carry_from] = S
                    b_up[h] = -S * float(base)
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = float(base) / S
                    W_down[fi(pos, ge.CARRY_OUT), h] = -1.0 / S
                    h += 1

                    # P = step(sum == base-1) = step(>=base-1) - step(>=base)
                    W_up[h, result_idx] = S
                    W_up[h, carry_from] = S
                    b_up[h] = -S * float(base - 2)
                    W_gate[h, op_gate] = 1.0
                    W_down[fi(pos, ge.TEMP), h] = 1.0 / S
                    h += 1

                    W_up[h, result_idx] = S
                    W_up[h, carry_from] = S
                    b_up[h] = -S * float(base - 1)
                    W_gate[h, op_gate] = 1.0
                    W_down[fi(pos, ge.TEMP), h] = -1.0 / S
                    h += 1

                    # Subtract G from TEMP: P = step(>=base-1) - step(>=base)
                    W_down[fi(pos, ge.TEMP), g_rise] = -1.0 / S
                    W_down[fi(pos, ge.TEMP), g_sat] = 1.0 / S

                else:
                    # pos=0: no incoming carry. RESULT[0] is already 0..base-1.
                    # G[0] = 0 always. P[0] = step(RESULT[0] == base-1)
                    W_up[h, result_idx] = S
                    b_up[h] = -S * float(base - 2)
                    W_gate[h, op_gate] = 1.0
                    W_down[fi(pos, ge.TEMP), h] = 1.0 / S
                    h += 1

                    W_up[h, result_idx] = S
                    b_up[h] = -S * float(base - 1)
                    W_gate[h, op_gate] = 1.0
                    W_down[fi(pos, ge.TEMP), h] = -1.0 / S
                    h += 1

            # Clear old CARRY_OUT at all positions
            for pos in range(N):
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(pos, ge.OP_START + opcode),
                                fi(pos, ge.CARRY_OUT), S)
                h += 2

            assert h <= hidden_dim, f"Used {h} hidden units, allocated {hidden_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class MulBinaryLookaheadFFN(nn.Module):
    """Carry-lookahead on binary G/P from MulGenPropFFN.

    G[i] in CARRY_OUT[i], P[i] in TEMP[i].
    Writes carries to CARRY_IN[i].
    Clears CARRY_OUT (G) and TEMP (P).
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        dtype = ge.config.torch_dtype

        # N*(N-1)/2 AND-gate units + 4*N clearing
        hidden_dim = N * (N - 1) // 2 + 4 * N

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down
            op_gate = fi(0, ge.OP_START + opcode)

            h = 0

            # Carry-lookahead: C[i] for i=1..N-1
            for i in range(1, N):
                # G[i-1] directly
                W_up[h, fi(i - 1, ge.CARRY_OUT)] = S
                b_up[h] = -S * 0.5
                W_gate[h, op_gate] = 1.0
                W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                h += 1

                # P[i-1]*...*P[j+1]*G[j]
                for j in range(i - 2, -1, -1):
                    n_vars = (i - 1 - j) + 1
                    for k in range(j + 1, i):
                        W_up[h, fi(k, ge.TEMP)] = S  # P[k]
                    W_up[h, fi(j, ge.CARRY_OUT)] = S  # G[j]
                    b_up[h] = -S * (n_vars - 0.5)
                    W_gate[h, op_gate] = 1.0
                    W_down[fi(i, ge.CARRY_IN), h] = 2.0 / S
                    h += 1

            # Clear G (CARRY_OUT) and P (TEMP) at all positions
            for pos in range(N):
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(pos, ge.OP_START + opcode),
                                fi(pos, ge.CARRY_OUT), S)
                h += 2
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(pos, ge.OP_START + opcode),
                                fi(pos, ge.TEMP), S)
                h += 2

            assert h <= hidden_dim, f"Used {h} hidden units, allocated {hidden_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class MulFinalCorrectionFFN(nn.Module):
    """Add carry from lookahead, mod base, clear CARRY_IN.

    CARRY_IN[i] from lookahead (binary). RESULT[i] in [0, base-1].
    Sum in [0, base]. If base: subtract base.
    Clear CARRY_IN.
    """

    def __init__(self, ge: GenericE, opcode: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = ge.SCALE
        base = ge.BASE
        dtype = ge.config.torch_dtype

        # 2 (add carry) + 2 (step overflow) + 2 (clear CARRY_IN) per pos
        # pos 0: no carry to add, just 2 clear = 2
        hidden_dim = 4 * (N - 1) + 2 * N

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=dtype)
        fi = self.flat_ffn._flat_idx

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down
            op_gate = fi(0, ge.OP_START + opcode)

            h = 0

            for pos in range(N):
                result_idx = fi(pos, ge.RESULT)
                carry_idx = fi(pos, ge.CARRY_IN)

                if pos > 0:
                    # Add CARRY_IN to RESULT
                    W_up[h, carry_idx] = S
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

                    W_up[h, carry_idx] = -S
                    W_gate[h, op_gate] = -1.0
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

                    # step(RESULT + CARRY_IN >= base) → subtract base
                    W_up[h, result_idx] = S
                    W_up[h, carry_idx] = S
                    b_up[h] = -S * float(base - 1)
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = -float(base) / S
                    h += 1

                    W_up[h, result_idx] = S
                    W_up[h, carry_idx] = S
                    b_up[h] = -S * float(base)
                    W_gate[h, op_gate] = 1.0
                    W_down[result_idx, h] = float(base) / S
                    h += 1

            # Clear CARRY_IN at all positions
            for pos in range(N):
                bake_clear_pair(self.flat_ffn.ffn, h,
                                fi(pos, ge.OP_START + opcode),
                                fi(pos, ge.CARRY_IN), S)
                h += 2

            assert h <= hidden_dim, f"Used {h} hidden units, allocated {hidden_dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


def build_mul_layers(config: ChunkConfig, opcode: int = 27,
                     source_a=None) -> nn.ModuleList:
    """Build MUL pipeline for BIT/PAIR/NIBBLE/BYTE configs."""
    ge = GenericE(config)
    layers = []

    # Layer 1: Schoolbook
    layers.append(SchoolbookFFN(ge, opcode, source_a=source_a))

    # Carry passes
    carry_passes = _compute_carry_passes(config)
    for idx, max_carry in enumerate(carry_passes):
        layers.append(CarryPassFFN(ge, opcode, max_carry=max_carry,
                                   pass_idx=idx))

    # GenProp (add final carry + compute G/P)
    layers.append(MulGenPropFFN(ge, opcode))

    # Binary lookahead
    layers.append(MulBinaryLookaheadFFN(ge, opcode))

    # Final correction
    layers.append(MulFinalCorrectionFFN(ge, opcode))

    return nn.ModuleList(layers)
