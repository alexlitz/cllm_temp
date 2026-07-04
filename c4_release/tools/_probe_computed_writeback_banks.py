"""Numeric proof: enumerated vs COMPUTED route for the two same-lane STACK0
OUTPUT byte-writeback banks whose match-weight != 1.0.

Follow-up to ``_probe_m8_computed_writeback.py`` (the pilot, match_weight=1.0):

  * ``stack0_pop_loaded``          match_weight=0.05  threshold=12.0
  * ``stack0_store_top_e8_from_e0``  match_weight=0.001 threshold=4300.0

Both READ the OUTPUT lane (``OUTPUT_LO`` / ``OUTPUT_HI_THIS_STEP``) at their
own per-value nibble match weight and WRITE the OUTPUT byte — a same-lane
identity copy gated by structural evidence.  This probe builds each enumerated
bank as a standalone ``PureFFN`` and compares its OUTPUT delta against the
matched per-nibble route (``_computed_byte_writeback_route_rules`` with the
same match_weight/threshold/base_conditions/competitor) over all 256 bytes and
several production-scale OUTPUT magnitudes.

Decisive check: identical firing region + 0 argmax mismatch (byte-for-byte
winning byte), for BOTH banks, so the enumerated->computed collapse is
verdict-neutral. Fully isolated (no full model build / disk cache); CPU, ~1s.
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
# Dense minimal dim layout covering every dim the two banks reference.
# ---------------------------------------------------------------------------
dim_positions = {}
_pos = 0


def _alloc(name, width=1):
    global _pos
    dim_positions[name] = _pos
    for k in range(width):
        dim_positions[f"{name}+{k}"] = _pos + k
    _pos += width


for nm in [
    "HAS_SE", "IS_BYTE", "MEM_STORE",
    "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM", "MARK_STACK0",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
]:
    _alloc(nm)
_alloc("CMP", 16)
_alloc("H1", 16)
_alloc("OUTPUT_LO", 16)
_alloc("OUTPUT_HI_THIS_STEP", 16)
# In production OUTPUT_HI aliases OUTPUT_HI_THIS_STEP (same base dim 190):
# byte_value_writes writes to OUTPUT_HI, nibble_value_writes writes to
# OUTPUT_HI_THIS_STEP, and they land on the SAME residual dims.  Mirror that
# aliasing here so the enum (byte_value_writes) and route (nibble_value_writes)
# writes are directly comparable.
dim_positions["OUTPUT_HI"] = dim_positions["OUTPUT_HI_THIS_STEP"]
for k in range(16):
    dim_positions[f"OUTPUT_HI+{k}"] = dim_positions[f"OUTPUT_HI_THIS_STEP+{k}"]
D = _pos


def resolve(name):
    if name in dim_positions:
        return dim_positions[name]
    base, off = name.rsplit("+", 1)
    return dim_positions[base] + int(off)


# ---------------------------------------------------------------------------
# Bank A: stack0_pop_loaded (verbatim base_conditions; competitor default 500).
# ---------------------------------------------------------------------------
POP_BASE = (
    ("MARK_STACK0", 1.0),
    ("HAS_SE", 1.0),
    ("CMP+3", 0.5),
    ("MEM_STORE", -1000.0),
    ("IS_BYTE", -100.0),
    ("MARK_AX", -1000000.0),
    ("MARK_PC", -100.0),
    ("MARK_SP", -100.0),
    ("MARK_BP", -100.0),
    ("MARK_MEM", -100.0),
    ("H1+0", -1000.0),
    ("H1+1", -1000.0),
    ("H1+2", -1000.0),
    ("H1+3", -1000.0),
    ("OP_EQ", -1000000.0),
    ("OP_NE", -1000000.0),
    ("OP_LT", -1000000.0),
    ("OP_GT", -1000000.0),
    ("OP_LE", -1000000.0),
    ("OP_GE", -1000000.0),
)


def pop_enumerated(competitor=500.0):
    rules = []
    for lo in range(16):
        for hi in range(16):
            if lo == 0 and hi == 0:
                continue
            value = lo | (hi << 4)
            rules.append(multi_way_and_rule(
                name=f"tail_stack0_pop_loaded_byte_{value:02x}",
                conditions=POP_BASE + (
                    (f"OUTPUT_LO+{lo}", 0.05),
                    (f"OUTPUT_HI_THIS_STEP+{hi}", 0.05),
                ),
                threshold=12.0,
                gate="MARK_STACK0",
                writes=Primitives.byte_value_writes(
                    value, strength=500.0, competitor_strength=competitor,
                ),
            ))
    return tuple(rules)


def pop_route(competitor=500.0):
    rules = []
    for band, other in (("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"),
                        ("OUTPUT_HI_THIS_STEP", "OUTPUT_LO")):
        other_terms = tuple((f"{other}+{j}", 0.05) for j in range(16))
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"tail_stack0_pop_loaded_route_{band}_{k}",
                conditions=POP_BASE + ((f"{band}+{k}", 0.05),) + other_terms,
                threshold=12.0,
                gate="MARK_STACK0",
                writes=Primitives.nibble_value_writes(
                    band, k, strength=500.0, competitor_strength=competitor,
                ),
            ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# Bank B: stack0_store_top_e8_from_e0 (nibble-loop part only; match wt 0.001,
# thr 4300). The extra byte_39 rule is NOT part of the loop and stays enum.
# base_conditions require EMBED/ADDR terms >> threshold in production; here we
# fold their positive contribution into a scalar so the OUTPUT-driven firing
# margin is what we compare.
# ---------------------------------------------------------------------------
E8_BASE = (
    ("MARK_STACK0", 5.0),
    ("HAS_SE", 1.0),
    ("CMP+3", 2.0),
    ("MEM_STORE", 1000.0),
    # EMBED_LO+8 / EMBED_HI+14 / ADDR_B0_* are large positives in production;
    # represent their fixed positive contribution as one structural term so the
    # firing margin above 4300 is exercised the same for enum and route.
    ("HAS_SE", 2400.0),  # stand-in for EMBED/ADDR positives (=200+200+1000+1000)
    ("IS_BYTE", -1000000.0),
    ("MARK_AX", -1000000.0),
    ("MARK_PC", -1000000.0),
    ("MARK_SP", -1000000.0),
    ("MARK_BP", -1000000.0),
    ("MARK_MEM", -1000000.0),
    ("H1+0", -1000.0),
    ("H1+1", -1000.0),
    ("H1+2", -1000.0),
    ("H1+3", -1000.0),
)


def e8_enumerated():
    rules = []
    for lo in range(16):
        for hi in range(16):
            if lo == 0 and hi == 0:
                continue
            value = lo | (hi << 4)
            rules.append(multi_way_and_rule(
                name=f"tail_stack0_store_top_e8_from_e0_byte_{value:02x}",
                conditions=E8_BASE + (
                    (f"OUTPUT_LO+{lo}", 0.001),
                    (f"OUTPUT_HI_THIS_STEP+{hi}", 0.001),
                ),
                threshold=4300.0,
                gate="MARK_STACK0",
                writes=Primitives.byte_value_writes(value, strength=5000.0),
            ))
    return tuple(rules)


def e8_route():
    rules = []
    for band, other in (("OUTPUT_LO", "OUTPUT_HI_THIS_STEP"),
                        ("OUTPUT_HI_THIS_STEP", "OUTPUT_LO")):
        other_terms = tuple((f"{other}+{j}", 0.001) for j in range(16))
        for k in range(16):
            rules.append(multi_way_and_rule(
                name=f"tail_stack0_store_top_e8_from_e0_route_{band}_{k}",
                conditions=E8_BASE + ((f"{band}+{k}", 0.001),) + other_terms,
                threshold=4300.0,
                gate="MARK_STACK0",
                writes=Primitives.nibble_value_writes(band, k, strength=5000.0),
            ))
    return tuple(rules)


def build_ffn(rules):
    ffn = PureFFN(D, len(rules))
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)
    ir.lower_ffn(ffn, dim_positions, S=S)
    return ffn


def make_input(byte_val, *, gate_on=True, is_byte=False, out_mag=1.0,
               mem_store=True):
    x = torch.zeros(1, 1, D)
    x[0, 0, resolve("HAS_SE")] = 1.0
    x[0, 0, resolve("CMP+3")] = 1.0
    if mem_store:
        x[0, 0, resolve("MEM_STORE")] = 1.0
    if is_byte:
        x[0, 0, resolve("IS_BYTE")] = 1.0
    if gate_on:
        x[0, 0, resolve("MARK_STACK0")] = 1.0
    lo = byte_val & 0xF
    hi = (byte_val >> 4) & 0xF
    x[0, 0, resolve("OUTPUT_LO+" + str(lo))] = out_mag
    x[0, 0, resolve("OUTPUT_HI_THIS_STEP+" + str(hi))] = out_mag
    return x


def output_band(delta):
    lo = [delta[0, 0, resolve(f"OUTPUT_LO+{k}")].item() for k in range(16)]
    hi = [delta[0, 0, resolve(f"OUTPUT_HI_THIS_STEP+{k}")].item() for k in range(16)]
    return lo, hi


def sweep(enum, route, out_mag, label, make_x):
    """Compare enum vs route across all 256 bytes.

    Byte 0x00 is EXCLUDED from the discrepancy count: the enumerated bank skips
    ``lo==0 and hi==0`` (an already-zero OUTPUT byte is a no-op identity copy),
    while the per-nibble route's LO+0 / HI+0 units DO fire and write 0x00 — but
    writing 0x00 onto an already-0x00 OUTPUT byte is itself an argmax no-op
    (same winning byte, byte 0x00).  Tracked separately as ``zero_route_fires``.
    """
    n_argmax_mismatch = 0
    n_enum_fired = 0
    n_route_fired = 0
    n_firing_disagree = 0
    zero_route_fires = 0
    for byte_val in range(256):
        x = make_x(byte_val, out_mag)
        de = enum(x) - x
        dc = route(x) - x
        le, he = output_band(de)
        lc, hc = output_band(dc)
        enum_fired = de.abs().max().item() > 1.0
        route_fired = dc.abs().max().item() > 1.0
        if byte_val == 0x00:
            zero_route_fires = int(route_fired)
            continue
        n_enum_fired += enum_fired
        n_route_fired += route_fired
        if enum_fired != route_fired:
            n_firing_disagree += 1
        if enum_fired and route_fired:
            argmax_e = (le.index(max(le)), he.index(max(he)))
            argmax_c = (lc.index(max(lc)), hc.index(max(hc)))
            if argmax_e != argmax_c:
                n_argmax_mismatch += 1
    print(f"  [{label}] out_mag={out_mag:>9.1f}: "
          f"enum_fired={n_enum_fired}/255 route_fired={n_route_fired}/255 "
          f"firing_disagree={n_firing_disagree} argmax_mismatch={n_argmax_mismatch} "
          f"zero_route_fires={zero_route_fires}")
    return n_firing_disagree, n_argmax_mismatch


def main():
    total_bad = 0

    print("=== BANK A: stack0_pop_loaded (match_weight=0.05, thr=12) ===")
    for comp in (500.0, 5.0):
        enum = build_ffn(pop_enumerated(comp))
        route = build_ffn(pop_route(comp))
        print(f"  competitor={comp} enum_units={enum.hidden_dim} "
              f"route_units={route.hidden_dim}")
        # pop_loaded is NOT a store: MEM_STORE must be OFF (base wt -1000).
        # 0.05*(2*out_mag) + structural(2.5) > 12  =>  out_mag > 95
        for mag in (50.0, 95.0, 96.0, 200.0, 1000.0, 5000.0):
            fd, am = sweep(enum, route, mag, f"pop c={comp:g}",
                           lambda b, m: make_input(b, out_mag=m,
                                                   mem_store=False))
            total_bad += fd + am
        # gate-off / IS_BYTE-block context
        for ctx, kw in (("GATE-OFF", dict(gate_on=False)),
                        ("IS_BYTE-block", dict(is_byte=True))):
            x = make_input(0x2A, out_mag=1000.0, mem_store=False, **kw)
            de = (enum(x) - x).abs().max().item()
            dc = (route(x) - x).abs().max().item()
            ok = (de > 1.0) == (dc > 1.0)
            print(f"    {ctx}: enum|Δ|={de:.1f} route|Δ|={dc:.1f} "
                  f"{'OK' if ok else 'MISMATCH'}")
            total_bad += 0 if ok else 1

    print("\n=== BANK B: stack0_store_top_e8_from_e0 loop "
          "(match_weight=0.001, thr=4300) ===")
    enum = build_ffn(e8_enumerated())
    route = build_ffn(e8_route())
    print(f"  enum_units={enum.hidden_dim} route_units={route.hidden_dim}")
    # base positives ~ 1+2+1000+2400 = 3403 ; +0.001*2*out_mag > 4300
    #   => out_mag > ~448500 ... too large; instead scale OUTPUT to production
    # magnitudes and confirm firing-region + argmax parity across a sweep.
    for mag in (1.0, 100.0, 448000.0, 449000.0, 1000000.0):
        fd, am = sweep(enum, route, mag, "e8",
                       lambda b, m: make_input(b, out_mag=m))
        total_bad += fd + am
    for ctx, kw in (("GATE-OFF", dict(gate_on=False)),
                    ("IS_BYTE-block", dict(is_byte=True))):
        x = make_input(0x2A, out_mag=1000000.0, **kw)
        de = (enum(x) - x).abs().max().item()
        dc = (route(x) - x).abs().max().item()
        ok = (de > 1.0) == (dc > 1.0)
        print(f"    {ctx}: enum|Δ|={de:.1f} route|Δ|={dc:.1f} "
              f"{'OK' if ok else 'MISMATCH'}")
        total_bad += 0 if ok else 1

    print(f"\n{'PASS' if total_bad == 0 else 'FAIL'}: total firing/argmax "
          f"discrepancies = {total_bad}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
