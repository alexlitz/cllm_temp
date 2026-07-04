"""M8 pilot probe: enumerated vs computed OUTPUT byte-writeback (isolated FFN).

Builds the ``stack0_store_loaded_output_rules`` enumerated bank (255 per-value
AND rules) as a standalone ``PureFFN`` and compares its residual delta against
candidate COMPUTED (per-nibble route) banks over a battery of realistic
STACK0-marker residual inputs.  Fully isolated: no full model build, no disk
cache; runs on CPU in a couple seconds.

The decisive question: does a per-nibble route reproduce the enumerated bank's
OUTPUT delta byte-for-byte across all 256 input byte values (and the ambiguous
zero / non-firing gate contexts)?
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

S = 100.0

# ---------------------------------------------------------------------------
# Minimal dim layout: only the dims these rules touch. Pack them densely.
# ---------------------------------------------------------------------------
BASE_COND_NAMES = [
    "HAS_SE", "CMP", "MEM_STORE", "IS_BYTE",
    "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_STACK0",
    "H1",  # 16-wide band; we use H1+0..H1+4
]
# The base_conditions of stack0_store_loaded reference:
#   HAS_SE, CMP+3, MEM_STORE, IS_BYTE, MARK_AX/PC/SP/BP/MEM, H1+0..H1+4
#   plus OUTPUT_LO+lo, OUTPUT_HI_THIS_STEP+hi, and gate MARK_STACK0.

# Build a dim map covering everything the rules reference.
dim_positions = {}
_pos = 0
def _alloc(name, width=1):
    global _pos
    # Register the BASE name at offset 0 (DimRef.resolve does base + offset).
    dim_positions[name] = _pos
    for k in range(width):
        dim_positions[f"{name}+{k}"] = _pos + k
    _pos += width

# scalar / marker dims
for nm in ["HAS_SE", "IS_BYTE", "MEM_STORE",
           "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_STACK0"]:
    _alloc(nm)
_alloc("CMP", 16)               # CMP+3 used
_alloc("H1", 16)                # H1+0..H1+4 used
_alloc("OUTPUT_LO", 16)
_alloc("OUTPUT_HI_THIS_STEP", 16)
D = _pos

def resolve(name):
    if name in dim_positions:
        return dim_positions[name]
    base, off = name.rsplit("+", 1)
    return dim_positions[base] + int(off)

# ---------------------------------------------------------------------------
# Enumerated bank (verbatim from l10_ops.stack0_store_loaded_output_rules).
# ---------------------------------------------------------------------------
def enumerated_rules():
    base_conditions = (
        ("HAS_SE", 1.0),
        ("CMP+3", 0.5),
        ("MEM_STORE", 1.0),
        ("IS_BYTE", -100.0),
        ("MARK_AX", -1000000.0),
        ("MARK_PC", -1000000.0),
        ("MARK_SP", -1000000.0),
        ("MARK_BP", -1000000.0),
        ("MARK_MEM", -1000000.0),
        ("H1+0", -1_000_000_000.0),
        ("H1+1", -1_000_000_000.0),
        ("H1+2", -1_000_000_000.0),
        ("H1+3", -1_000_000_000.0),
        ("H1+4", -1_000_000_000.0),
    )
    rules = []
    for lo in range(16):
        for hi in range(16):
            if lo == 0 and hi == 0:
                continue
            value = lo | (hi << 4)
            rules.append(
                multi_way_and_rule(
                    name=f"tail_stack0_store_loaded_byte_{value:02x}",
                    conditions=base_conditions + (
                        (f"OUTPUT_LO+{lo}", 1.0),
                        (f"OUTPUT_HI_THIS_STEP+{hi}", 1.0),
                    ),
                    threshold=25.0,
                    gate="MARK_STACK0",
                    writes=Primitives.byte_value_writes(
                        value, strength=5000.0,
                        lo_base="OUTPUT_LO", hi_base="OUTPUT_HI_THIS_STEP",
                    ),
                )
            )
    return tuple(rules)


# ---------------------------------------------------------------------------
# Computed bank variant A: per-nibble route, one unit per (band, channel).
# 32 rules total (16 LO + 16 HI). Each fires on its own channel's one-hot,
# writes +strength to that channel, -strength to the 15 competitors.
# ---------------------------------------------------------------------------
def computed_rules_A():
    base_conditions = (
        ("HAS_SE", 1.0),
        ("CMP+3", 0.5),
        ("MEM_STORE", 1.0),
        ("IS_BYTE", -100.0),
        ("MARK_AX", -1000000.0),
        ("MARK_PC", -1000000.0),
        ("MARK_SP", -1000000.0),
        ("MARK_BP", -1000000.0),
        ("MARK_MEM", -1000000.0),
        ("H1+0", -1_000_000_000.0),
        ("H1+1", -1_000_000_000.0),
        ("H1+2", -1_000_000_000.0),
        ("H1+3", -1_000_000_000.0),
        ("H1+4", -1_000_000_000.0),
    )
    rules = []
    # Base score with all structural conditions satisfied (one-hot=1 each):
    #   HAS_SE(1)+CMP+3(0.5)+MEM_STORE(1)+IS_BYTE(0 since IS_BYTE=0 here)= 2.5
    #   plus one channel match (1.0) => 3.5 . Original threshold was 25 with
    #   TWO channel matches => but we only have ONE channel per rule now, so
    #   threshold must be lowered to fire on the single-channel evidence.
    # We keep the SAME structural base_score contribution and require the
    # single channel one-hot. Threshold picked so "structural + channel on"
    # fires and "structural + channel off" does not.
    for band in ("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"):
        for k in range(16):
            rules.append(
                multi_way_and_rule(
                    name=f"computed_route_{band}_{k}",
                    conditions=base_conditions + ((f"{band}+{k}", 1.0),),
                    threshold=3.0,
                    gate="MARK_STACK0",
                    writes=Primitives.nibble_value_writes(
                        band, k, strength=5000.0,
                    ),
                )
            )
    return tuple(rules)


def computed_rules_B():
    """Per-nibble route that ALSO sums the OTHER band's full one-hot into the
    firing score, so the up-score sees the SAME 2-channel evidence magnitude as
    the enumerated bank (both nibbles present => a valid byte). This couples the
    firing decision to "a byte is present" while keeping a per-nibble WRITE.
    Threshold matched to the enumerated 25.0.
    """
    base_conditions = (
        ("HAS_SE", 1.0),
        ("CMP+3", 0.5),
        ("MEM_STORE", 1.0),
        ("IS_BYTE", -100.0),
        ("MARK_AX", -1000000.0),
        ("MARK_PC", -1000000.0),
        ("MARK_SP", -1000000.0),
        ("MARK_BP", -1000000.0),
        ("MARK_MEM", -1000000.0),
        ("H1+0", -1_000_000_000.0),
        ("H1+1", -1_000_000_000.0),
        ("H1+2", -1_000_000_000.0),
        ("H1+3", -1_000_000_000.0),
        ("H1+4", -1_000_000_000.0),
    )
    rules = []
    for band, other in (("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"),
                        ("OUTPUT_HI_THIS_STEP", "OUTPUT_LO")):
        for k in range(16):
            # This channel's one-hot (weight 1.0) + ANY of the other band's
            # channels (weight 1.0 each; exactly one is on) => same 2-channel
            # magnitude as the enumerated per-value AND.
            other_terms = tuple((f"{other}+{j}", 1.0) for j in range(16))
            rules.append(
                multi_way_and_rule(
                    name=f"computed_routeB_{band}_{k}",
                    conditions=base_conditions + ((f"{band}+{k}", 1.0),) + other_terms,
                    threshold=25.0,
                    gate="MARK_STACK0",
                    writes=Primitives.nibble_value_writes(
                        band, k, strength=5000.0,
                    ),
                )
            )
    return tuple(rules)


def build_ffn(rules):
    ffn = PureFFN(D, len(rules))
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    ir.lower_ffn(ffn, dim_positions, S=S)
    return ffn


def make_input(byte_val, *, gate_on=True, is_byte=False, structural=True,
               out_mag=1.0):
    x = torch.zeros(1, 1, D)
    if structural:
        x[0, 0, resolve("HAS_SE")] = 1.0
        x[0, 0, resolve("CMP+3")] = 1.0
        x[0, 0, resolve("MEM_STORE")] = 1.0
    if is_byte:
        x[0, 0, resolve("IS_BYTE")] = 1.0
    if gate_on:
        x[0, 0, resolve("MARK_STACK0")] = 1.0
    lo = byte_val & 0xF
    hi = (byte_val >> 4) & 0xF
    # OUTPUT nibbles carry the source-lane byte one-hot; in production the L15
    # relay drives these to LARGE magnitudes (out_mag), which is what pushes
    # the enumerated bank's OUTPUT-weighted score past threshold 25.
    x[0, 0, resolve("OUTPUT_LO+" + str(lo))] = out_mag
    x[0, 0, resolve("OUTPUT_HI_THIS_STEP+" + str(hi))] = out_mag
    return x


def output_band(delta):
    lo = [delta[0, 0, resolve(f"OUTPUT_LO+{k}")].item() for k in range(16)]
    hi = [delta[0, 0, resolve(f"OUTPUT_HI_THIS_STEP+{k}")].item() for k in range(16)]
    return lo, hi


def sweep(enum, compA, out_mag, label, verbose=False):
    """Compare enum vs computed over all 256 bytes at a given OUTPUT magnitude."""
    n_argmax_mismatch = 0
    n_enum_fired = 0
    n_comp_fired = 0
    max_abs_diff = 0.0
    worst = None
    for byte_val in range(256):
        x = make_input(byte_val, gate_on=True, is_byte=False, out_mag=out_mag)
        de = enum(x) - x
        dc = compA(x) - x
        le, he = output_band(de)
        lc, hc = output_band(dc)
        lo_in = byte_val & 0xF
        hi_in = (byte_val >> 4) & 0xF
        enum_fired = de.abs().max().item() > 1.0
        comp_fired = dc.abs().max().item() > 1.0
        n_enum_fired += enum_fired
        n_comp_fired += comp_fired
        # The "correct" decode is the input byte itself (identity copy).
        argmax_e = (le.index(max(le)), he.index(max(he)))
        argmax_c = (lc.index(max(lc)), hc.index(max(hc)))
        if enum_fired and comp_fired and argmax_e != argmax_c:
            n_argmax_mismatch += 1
            if verbose and n_argmax_mismatch <= 5:
                print(f"    MISMATCH byte={byte_val:02x} in=({lo_in},{hi_in}) "
                      f"enum={argmax_e} comp={argmax_c}")
        diff = max(
            max(abs(a - b) for a, b in zip(le, lc)),
            max(abs(a - b) for a, b in zip(he, hc)),
        )
        if diff > max_abs_diff:
            max_abs_diff = diff
            worst = byte_val
    worst_s = f"{worst:02x}" if worst is not None else "--"
    print(f"[{label}] out_mag={out_mag:>6.1f}: "
          f"enum_fired={n_enum_fired}/256 comp_fired={n_comp_fired}/256 "
          f"argmax_mismatch(both-fired)={n_argmax_mismatch} "
          f"max|Δenum-Δcomp|={max_abs_diff:.1f} (worst={worst_s})")
    return n_enum_fired, n_comp_fired, n_argmax_mismatch


def main():
    enum = build_ffn(enumerated_rules())
    compA = build_ffn(computed_rules_A())
    compB = build_ffn(computed_rules_B())
    print(f"D={D}  enumerated units={enum.hidden_dim}  "
          f"computedA units={compA.hidden_dim}  computedB units={compB.hidden_dim}\n")

    print("=== VARIANT A (decoupled per-nibble, threshold 3) ===")
    for mag in (1.0, 5.0, 12.0, 13.0, 20.0, 50.0, 100.0, 5000.0):
        sweep(enum, compA, mag, "A", verbose=False)

    print("\n=== VARIANT B (coupled: other-band sum, threshold 25) ===")
    for mag in (1.0, 5.0, 12.0, 13.0, 20.0, 50.0, 100.0, 5000.0):
        sweep(enum, compB, mag, "B", verbose=(mag == 13.0))

    print("\n=== GATE / BLOCK CONTEXTS (out_mag=100) ===")
    for name, cb in (("A", compA), ("B", compB)):
        x = make_input(0x2A, gate_on=False, out_mag=100.0)
        de, dc = enum(x) - x, cb(x) - x
        g = (f"GATE-OFF: enum max|Δ|={de.abs().max().item():.1f} "
             f"comp{name} max|Δ|={dc.abs().max().item():.1f}")
        x = make_input(0x2A, gate_on=True, is_byte=True, out_mag=100.0)
        de, dc = enum(x) - x, cb(x) - x
        b = (f"IS_BYTE-block: enum max|Δ|={de.abs().max().item():.1f} "
             f"comp{name} max|Δ|={dc.abs().max().item():.1f}")
        print(f"  [{name}] {g}")
        print(f"  [{name}] {b}")

    # Margin analysis: winner - runner-up in the OUTPUT band (what downstream
    # argmax/decode robustness depends on).
    print("\n=== WINNER MARGIN (winner - 2nd) at out_mag=100 ===")
    def margin(band):
        s = sorted(band, reverse=True)
        return s[0] - s[1]
    for name, cb in (("enum", enum), ("B", compB)):
        x = make_input(0x2A, gate_on=True, out_mag=100.0)
        d = cb(x) - x
        lo, hi = output_band(d)
        print(f"  [{name}] LO-margin={margin(lo):.1f}  HI-margin={margin(hi):.1f}")

    print("\n=== DETAIL byte 0x2A FIRING (out_mag=100) ===")
    x = make_input(0x2A, gate_on=True, out_mag=100.0)
    de, dcB = enum(x) - x, compB(x) - x
    le, he = output_band(de)
    lc, hc = output_band(dcB)
    print(f"  enum  LO argmax={le.index(max(le))} val={max(le):.1f} min={min(le):.1f} | "
          f"HI argmax={he.index(max(he))} val={max(he):.1f} min={min(he):.1f}")
    print(f"  compB LO argmax={lc.index(max(lc))} val={max(lc):.1f} min={min(lc):.1f} | "
          f"HI argmax={hc.index(max(hc))} val={max(hc):.1f} min={min(hc):.1f}")


if __name__ == "__main__":
    main()
