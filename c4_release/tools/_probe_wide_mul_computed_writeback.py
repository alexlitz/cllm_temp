"""Isolated argmax-equivalence proof for the wide_mul_byte1 ENUM->COMPUTED collapse.

The ``wide_mul_byte1_preserve_rules`` bank (in
``l10_ops._tail_bit32_result_correction_rules``) is a full 16x16 nibble loop of
256 per-value AND rules that, under the OP_MUL AX byte-0 evidence gate, matches
a staged MUL byte value on ``OUTPUT_LO+lo`` / ``OUTPUT_HI_THIS_STEP+hi`` (weight
2.0 each) and re-asserts that same byte on OUTPUT at strength 10_000_000. READ
lane == WRITE lane (OUTPUT_HI aliases OUTPUT_HI_THIS_STEP), so it is a same-lane
identity copy gated by structural evidence.

This probe builds the enumerated bank and the matched per-nibble route
(``_computed_byte_writeback_route_rules``, match_weight=2.0 / threshold=220 /
strength=10M / same base_conditions) as standalone ``PureFFN``s and compares
their OUTPUT delta / argmax across all 256 bytes and several production-scale
OUTPUT magnitudes.

Decisive check: identical firing region + 0 argmax mismatch (byte-for-byte
winning byte), so the collapse is verdict-neutral. Fully isolated (no full model
build / disk cache); CPU, ~1s.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from c4_release.neural_vm.base_layers import PureFFN  # noqa: E402
from c4_release.neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from c4_release.neural_vm.unified_compiler.building_blocks_dsl import (  # noqa: E402
    multi_way_and_rule,
)
from c4_release.neural_vm.unified_compiler.ir import CompilerIR  # noqa: E402
from c4_release.neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _computed_byte_writeback_route_rules,
)

S = 100.0

dim_positions = {}
_pos = 0


def _alloc(name, width=1):
    global _pos
    dim_positions[name] = _pos
    for k in range(width):
        dim_positions[f"{name}+{k}"] = _pos + k
    _pos += width


for nm in [
    "HAS_SE", "IS_BYTE",
    "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_STACK0",
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    "OP_MUL", "OP_ADD", "OP_SUB", "OP_DIV", "OP_MOD", "OP_SHL", "OP_SHR",
    "OP_AND", "OP_OR", "OP_XOR", "OP_EQ", "OP_NE", "OP_LT", "OP_GT",
    "OP_LE", "OP_GE", "OP_JSR", "OP_LEV", "OP_LEA", "OP_JMP", "OP_ADJ",
    "OP_ENT",
]:
    _alloc(nm)
_alloc("H1", 16)
_alloc("TEMP", 16)
_alloc("OUTPUT_LO", 16)
_alloc("OUTPUT_HI_THIS_STEP", 16)
# In production OUTPUT_HI aliases OUTPUT_HI_THIS_STEP (same base dim). Mirror it
# so the enum (byte_value_writes -> OUTPUT_HI) and route (nibble_value_writes ->
# OUTPUT_HI_THIS_STEP) writes are directly comparable.
dim_positions["OUTPUT_HI"] = dim_positions["OUTPUT_HI_THIS_STEP"]
for k in range(16):
    dim_positions[f"OUTPUT_HI+{k}"] = dim_positions[f"OUTPUT_HI_THIS_STEP+{k}"]
D = _pos


def resolve(name):
    if name in dim_positions:
        return dim_positions[name]
    base, off = name.rsplit("+", 1)
    return dim_positions[base] + int(off)


non_mul_blockers = (
    ("OP_ADD", -1000.0), ("OP_SUB", -1000.0), ("OP_DIV", -1000.0),
    ("OP_MOD", -1000.0), ("OP_SHL", -1000.0), ("OP_SHR", -1000.0),
    ("OP_AND", -1000.0), ("OP_OR", -1000.0), ("OP_XOR", -1000.0),
)
bounded_ax_byte0 = (
    ("IS_BYTE", 5.0), ("H1+1", 20.0), ("H1+2", -1000.0), ("H1+3", -1000.0),
    ("H1+4", -1000.0), ("BYTE_INDEX_0", 5.0), ("BYTE_INDEX_1", -1000.0),
    ("BYTE_INDEX_2", -1000.0), ("BYTE_INDEX_3", -1000.0), ("MARK_AX", -1000.0),
    ("MARK_PC", -1000.0), ("MARK_SP", -1000.0), ("MARK_BP", -1000.0),
    ("MARK_STACK0", -1000.0), ("MARK_MEM", -1000.0), ("OP_LEA", -1000.0),
    ("OP_JMP", -1000.0), ("OP_ADJ", -1000.0), ("OP_ENT", -1000.0),
)
MUL_BASE = bounded_ax_byte0 + (
    ("HAS_SE", 20.0), ("TEMP+10", 30.0), ("OP_MUL", 80.0),
    ("OP_EQ", -1000.0), ("OP_NE", -1000.0), ("OP_LT", -1000.0),
    ("OP_GT", -1000.0), ("OP_LE", -1000.0), ("OP_GE", -1000.0),
    ("OP_JSR", -1000.0), ("OP_LEV", -1000.0), ("TEMP+4", -1000.0),
    ("TEMP+5", -1000.0), ("TEMP+6", -1000.0), ("TEMP+8", -1000.0),
    ("TEMP+9", -1000.0),
) + non_mul_blockers


def enumerated():
    rules = []
    for high in range(16):
        for low in range(16):
            value = (high << 4) | low
            rules.append(multi_way_and_rule(
                name=f"tail_wide_mul_byte1_preserve_{value:02x}",
                conditions=bounded_ax_byte0 + (
                    ("HAS_SE", 20.0), ("TEMP+10", 30.0), ("OP_MUL", 80.0),
                    ("OP_EQ", -1000.0), ("OP_NE", -1000.0), ("OP_LT", -1000.0),
                    ("OP_GT", -1000.0), ("OP_LE", -1000.0), ("OP_GE", -1000.0),
                    ("OP_JSR", -1000.0), ("OP_LEV", -1000.0),
                    (f"OUTPUT_LO+{low}", 2.0),
                    (f"OUTPUT_HI_THIS_STEP+{high}", 2.0),
                    ("TEMP+4", -1000.0), ("TEMP+5", -1000.0),
                    ("TEMP+6", -1000.0), ("TEMP+8", -1000.0),
                    ("TEMP+9", -1000.0),
                ) + non_mul_blockers,
                threshold=220.0,
                gate="OP_MUL",
                writes=Primitives.byte_value_writes(value, strength=10_000_000.0),
            ))
    return tuple(rules)


def route():
    return _computed_byte_writeback_route_rules(
        name_for=lambda band, k: f"tail_wide_mul_byte1_preserve_route_{band}_{k}",
        base_conditions=MUL_BASE,
        threshold=220.0,
        strength=10_000_000.0,
        match_weight=2.0,
        lo_base="OUTPUT_LO",
        hi_base="OUTPUT_HI_THIS_STEP",
        gate="OP_MUL",
    )


def build_ffn(rules):
    ffn = PureFFN(D, len(rules))
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    ir.lower_ffn(ffn, dim_positions, S=S)
    return ffn


def make_input(byte_val, *, gate_on=True, out_mag=1.0):
    x = torch.zeros(1, 1, D)
    x[0, 0, resolve("HAS_SE")] = 1.0
    x[0, 0, resolve("IS_BYTE")] = 1.0
    x[0, 0, resolve("BYTE_INDEX_0")] = 1.0
    x[0, 0, resolve("H1+1")] = 1.0
    x[0, 0, resolve("TEMP+10")] = 1.0
    if gate_on:
        x[0, 0, resolve("OP_MUL")] = 1.0
    lo = byte_val & 0xF
    hi = (byte_val >> 4) & 0xF
    x[0, 0, resolve(f"OUTPUT_LO+{lo}")] = out_mag
    x[0, 0, resolve(f"OUTPUT_HI_THIS_STEP+{hi}")] = out_mag
    return x


def output_byte(delta):
    lo = [delta[0, 0, resolve(f"OUTPUT_LO+{k}")].item() for k in range(16)]
    hi = [delta[0, 0, resolve(f"OUTPUT_HI_THIS_STEP+{k}")].item() for k in range(16)]
    return lo, hi


def main():
    enum_ffn = build_ffn(enumerated())
    route_ffn = build_ffn(route())
    print(f"enum units={enum_ffn.W_up.shape[0]}  route units={route_ffn.W_up.shape[0]}")

    n_mismatch = 0
    n_fire_disagree = 0
    for out_mag in (1.0, 10.0, 100.0, 1000.0):
        for byte_val in range(256):
            x = make_input(byte_val, gate_on=True, out_mag=out_mag)
            de = enum_ffn(x) - x
            dr = route_ffn(x) - x
            elo, ehi = output_byte(de)
            rlo, rhi = output_byte(dr)
            # Winning byte = argmax of (input + delta) on each nibble band.
            base_lo = [x[0, 0, resolve(f"OUTPUT_LO+{k}")].item() for k in range(16)]
            base_hi = [x[0, 0, resolve(f"OUTPUT_HI_THIS_STEP+{k}")].item() for k in range(16)]
            e_lo = int(torch.tensor([base_lo[k] + elo[k] for k in range(16)]).argmax())
            e_hi = int(torch.tensor([base_hi[k] + ehi[k] for k in range(16)]).argmax())
            r_lo = int(torch.tensor([base_lo[k] + rlo[k] for k in range(16)]).argmax())
            r_hi = int(torch.tensor([base_hi[k] + rhi[k] for k in range(16)]).argmax())
            e_byte = (e_hi << 4) | e_lo
            r_byte = (r_hi << 4) | r_lo
            e_fired = any(abs(v) > 1e-9 for v in elo + ehi)
            r_fired = any(abs(v) > 1e-9 for v in rlo + rhi)
            if e_fired != r_fired:
                n_fire_disagree += 1
                if n_fire_disagree <= 5:
                    print(f"  FIRE DISAGREE mag={out_mag} byte={byte_val:02x} enum={e_fired} route={r_fired}")
            if e_byte != r_byte:
                n_mismatch += 1
                if n_mismatch <= 5:
                    print(f"  ARGMAX MISMATCH mag={out_mag} byte={byte_val:02x} enum={e_byte:02x} route={r_byte:02x}")

    # Gate-off (OP_MUL absent) must not fire either.
    n_gateoff_fire = 0
    for byte_val in range(256):
        x = make_input(byte_val, gate_on=False, out_mag=100.0)
        de = enum_ffn(x) - x
        dr = route_ffn(x) - x
        elo, ehi = output_byte(de)
        rlo, rhi = output_byte(dr)
        if any(abs(v) > 1e-9 for v in elo + ehi + rlo + rhi):
            n_gateoff_fire += 1

    print(f"argmax mismatches (all 256 bytes x 4 mags): {n_mismatch}")
    print(f"firing-region disagreements: {n_fire_disagree}")
    print(f"gate-off spurious fires: {n_gateoff_fire}")
    ok = (n_mismatch == 0 and n_fire_disagree == 0 and n_gateoff_fire == 0)
    print("RESULT:", "PASS (verdict-neutral collapse)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
