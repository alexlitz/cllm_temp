"""GAP-PRIMITIVE #3 probe: cross-lane ALU->OUTPUT materializer, three forms.

Builds, as standalone isolated ``PureFFN`` banks (no full model, no disk
cache), three encodings of the SAME cross-lane copy "read a byte one-hot from
the ALU lane (ALU_LO/ALU_HI), write it into the OUTPUT lane
(OUTPUT_LO/OUTPUT_HI_THIS_STEP)":

  1. ENUMERATED  -- 256-way per-VALUE AND (the form ``_byte_value_writeback_rules``
     produces with ``lo_base=ALU_LO, hi_base=ALU_HI``): fires on
     ``ALU_LO+lo AND ALU_HI+hi`` and writes ``byte_value_writes(OUTPUT byte)``.
  2. GATE ROUTE  -- 32-way per-cell gate route (the existing ``byte_route_rules``
     / l16 ``_add_stack0_x0_alu_materializer`` form): per cell k, gated on
     ``ALU_LO+k`` / ``ALU_HI+k``, writes ``OUTPUT_*+k``.  silu(up)*sigmoid(gate).
  3. COMPUTED    -- 32-way per-cell conditions-AND route (the NEW
     ``byte_copy_computed_rules``): fires on the SOURCE channel one-hot + the
     SUM of the OTHER source band; writes a one-hot nibble into OUTPUT.  Pure
     silu(up), no gate.

Decisive question: does form 3 (the M8-style computed copy, generalised to a
different SOURCE lane) reproduce the ENUMERATED bank's winning OUTPUT byte
(argmax) and firing region byte-for-byte across all 256 source bytes and the
gate-off / IS_BYTE-block non-firing contexts?
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
    byte_copy_computed_rules,
    byte_route_rules,
    multi_way_and_rule,
)
from c4_release.neural_vm.unified_compiler.ir import CompilerIR  # noqa: E402

S = 100.0
STRENGTH = 5000.0
THRESH = 25.0

# Structural evidence gate shared by all three forms (an ALU->OUTPUT
# materializer at a STACK0 marker; a subset of the real l16 gate).
BASE_CONDITIONS = (
    ("HAS_SE", 1.0),
    ("MEM_STORE", 1.0),
    ("IS_BYTE", -100.0),
    ("MARK_AX", -1_000_000.0),
    ("MARK_PC", -1_000_000.0),
    ("MARK_SP", -1_000_000.0),
    ("MARK_BP", -1_000_000.0),
    ("MARK_MEM", -1_000_000.0),
)

# ---------------------------------------------------------------------------
# Dim layout (dense, only what the rules touch).
# ---------------------------------------------------------------------------
dim_positions = {}
_pos = 0


def _alloc(name, width=1):
    global _pos
    dim_positions[name] = _pos
    for k in range(width):
        dim_positions[f"{name}+{k}"] = _pos + k
    _pos += width


for nm in ["HAS_SE", "IS_BYTE", "MEM_STORE",
           "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
           "MARK_STACK0"]:
    _alloc(nm)
_alloc("ALU_LO", 16)
_alloc("ALU_HI", 16)
_alloc("OUTPUT_LO", 16)
_alloc("OUTPUT_HI_THIS_STEP", 16)
D = _pos


def resolve(name):
    if name in dim_positions:
        return dim_positions[name]
    base, off = name.rsplit("+", 1)
    return dim_positions[base] + int(off)


# ---------------------------------------------------------------------------
# 1. ENUMERATED cross-lane bank: source=ALU, dest=OUTPUT, per byte value.
# ---------------------------------------------------------------------------
def enumerated_rules():
    rules = []
    for lo in range(16):
        for hi in range(16):
            if lo == 0 and hi == 0:
                continue
            value = lo | (hi << 4)
            rules.append(multi_way_and_rule(
                name=f"enum_alu_out_{value:02x}",
                conditions=BASE_CONDITIONS + (
                    (f"ALU_LO+{lo}", 1.0),
                    (f"ALU_HI+{hi}", 1.0),
                ),
                threshold=THRESH,
                gate="MARK_STACK0",
                writes=Primitives.byte_value_writes(
                    value, strength=STRENGTH,
                    lo_base="OUTPUT_LO", hi_base="OUTPUT_HI_THIS_STEP",
                ),
            ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# 2. GATE ROUTE (existing byte_route_rules / l16 materializer form).
# ---------------------------------------------------------------------------
def gate_route_rules():
    return byte_route_rules(
        band_specs=(
            ("lo", "ALU_LO", "OUTPUT_LO"),
            ("hi", "ALU_HI", "OUTPUT_HI_THIS_STEP"),
        ),
        conditions=BASE_CONDITIONS,
        threshold=THRESH,
        write_value=STRENGTH,
        S=S,
        name_prefix="gate_route",
    )


# ---------------------------------------------------------------------------
# 3. COMPUTED cross-lane route (the NEW byte_copy_computed_rules).
# ---------------------------------------------------------------------------
def computed_rules():
    return byte_copy_computed_rules(
        src_lo="ALU_LO",
        src_hi="ALU_HI",
        dst_lo="OUTPUT_LO",
        dst_hi="OUTPUT_HI_THIS_STEP",
        base_conditions=BASE_CONDITIONS,
        threshold=THRESH,
        strength=STRENGTH,
        name_for=lambda band, k: f"computed_{band}_{k}",
        gate="MARK_STACK0",
    )


def build_ffn(rules):
    ffn = PureFFN(D, len(rules))
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    ir.lower_ffn(ffn, dim_positions, S=S)
    return ffn


def make_input(byte_val, *, gate_on=True, is_byte=False, alu_mag=1.0):
    x = torch.zeros(1, 1, D)
    x[0, 0, resolve("HAS_SE")] = 1.0
    x[0, 0, resolve("MEM_STORE")] = 1.0
    if is_byte:
        x[0, 0, resolve("IS_BYTE")] = 1.0
    if gate_on:
        x[0, 0, resolve("MARK_STACK0")] = 1.0
    lo = byte_val & 0xF
    hi = (byte_val >> 4) & 0xF
    # SOURCE lane (ALU) carries the byte one-hot.
    x[0, 0, resolve(f"ALU_LO+{lo}")] = alu_mag
    x[0, 0, resolve(f"ALU_HI+{hi}")] = alu_mag
    return x


def output_band(delta):
    lo = [delta[0, 0, resolve(f"OUTPUT_LO+{k}")].item() for k in range(16)]
    hi = [delta[0, 0, resolve(f"OUTPUT_HI_THIS_STEP+{k}")].item() for k in range(16)]
    return lo, hi


def sweep(enum, cand, alu_mag, label):
    n_argmax_mismatch = 0
    n_enum_fired = 0
    n_cand_fired = 0
    n_both_fired = 0
    n_argmax_correct = 0  # cand argmax == the input byte (the true copy)
    max_abs_diff = 0.0
    worst = None
    for byte_val in range(256):
        x = make_input(byte_val, alu_mag=alu_mag)
        de = enum(x) - x
        dc = cand(x) - x
        le, he = output_band(de)
        lc, hc = output_band(dc)
        lo_in = byte_val & 0xF
        hi_in = (byte_val >> 4) & 0xF
        enum_fired = de.abs().max().item() > 1.0
        cand_fired = dc.abs().max().item() > 1.0
        n_enum_fired += enum_fired
        n_cand_fired += cand_fired
        argmax_e = (le.index(max(le)), he.index(max(he)))
        argmax_c = (lc.index(max(lc)), hc.index(max(hc)))
        if cand_fired and argmax_c == (lo_in, hi_in):
            n_argmax_correct += 1
        if enum_fired and cand_fired:
            n_both_fired += 1
            if argmax_e != argmax_c:
                n_argmax_mismatch += 1
        diff = max(
            max(abs(a - b) for a, b in zip(le, lc)),
            max(abs(a - b) for a, b in zip(he, hc)),
        )
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst = byte_val
    worst_s = f"{worst:02x}" if worst is not None else "--"
    print(f"[{label}] alu_mag={alu_mag:>6.1f}: enum_fired={n_enum_fired}/256 "
          f"cand_fired={n_cand_fired}/256 both={n_both_fired} "
          f"argmax_mismatch={n_argmax_mismatch} "
          f"cand_copy_correct={n_argmax_correct}/{n_cand_fired} "
          f"max|Δe-Δc|={max_abs_diff:.1f} (worst={worst_s})")
    return n_argmax_mismatch


def main():
    enum = build_ffn(enumerated_rules())
    gate = build_ffn(gate_route_rules())
    comp = build_ffn(computed_rules())
    print(f"D={D}  enumerated units={enum.hidden_dim}  "
          f"gate_route units={gate.hidden_dim}  computed units={comp.hidden_dim}")
    print(f"(enumerated skips byte 0x00 -> {enum.hidden_dim} == 256-1 = 255)\n")

    print("=== COMPUTED (byte_copy_computed_rules) vs ENUMERATED ===")
    total_mm = 0
    for mag in (1.0, 5.0, 13.0, 20.0, 50.0, 100.0):
        total_mm += sweep(enum, comp, mag, "COMPUTED")

    print("\n=== GATE ROUTE (byte_route_rules) vs ENUMERATED ===")
    for mag in (1.0, 5.0, 13.0, 20.0, 50.0, 100.0):
        sweep(enum, gate, mag, "GATE")

    print("\n=== NON-FIRING CONTEXTS (alu_mag=50) ===")
    for name, cb in (("computed", comp), ("gate", gate)):
        x = make_input(0x2A, gate_on=False, alu_mag=50.0)
        de, dc = enum(x) - x, cb(x) - x
        print(f"  [{name}] GATE-OFF: enum max|Δ|={de.abs().max().item():.2f} "
              f"cand max|Δ|={dc.abs().max().item():.2f}")
        x = make_input(0x2A, gate_on=True, is_byte=True, alu_mag=50.0)
        de, dc = enum(x) - x, cb(x) - x
        print(f"  [{name}] IS_BYTE-block: enum max|Δ|={de.abs().max().item():.2f} "
              f"cand max|Δ|={dc.abs().max().item():.2f}")

    print("\n=== DETAIL byte 0x2A (alu_mag=50) ===")
    x = make_input(0x2A, alu_mag=50.0)
    for name, cb in (("enum", enum), ("computed", comp), ("gate", gate)):
        d = cb(x) - x
        lo, hi = output_band(d)
        print(f"  [{name:>8}] LO argmax={lo.index(max(lo))} (want 10) "
              f"max={max(lo):.1f} min={min(lo):.1f} | "
              f"HI argmax={hi.index(max(hi))} (want 2) "
              f"max={max(hi):.1f} min={min(hi):.1f}")

    print(f"\nVERDICT: COMPUTED total argmax mismatch vs enumerated (all mags) "
          f"= {total_mm}")
    print("PASS" if total_mm == 0 else "FAIL")


if __name__ == "__main__":
    main()
