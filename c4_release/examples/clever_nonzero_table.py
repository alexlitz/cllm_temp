#!/usr/bin/env python3
r"""clever_nonzero_table.py — the COMPLETE census of the clever c4 VM's TOTAL
non-zero parameters across the full config space (precision x radix x extraction
x mode).

This is the pure-census generator for docs/CLEVER_NONZERO_CONFIG_TABLE.md.  It
REUSES (does not re-derive) the measured machinery:

  * examples/clever_realtime_model.census_total_nonzero / _cell_nonzero /
    _framing_nonzero — the arith ingest+decode cell (51 nonzero), the bitwise
    16x16 LUT triple (OR+AND+XOR = 2,974), the memory CAM (10), and the framing
    (12 embed + 10 lm-head + 4/layer).  These are REAL torch-tensor cells with
    .count_nonzero() (examples/clever_realtime_cells.py), not a math-sim.
  * c4_min.opconfig — the {precision, radix, extraction, recurrence} axes, the
    precision<->radix coupling (validate / max_safe_radix / acc_max / the
    PRECISION_CEILING), and the honesty constraint (tied is only legal on a
    looped/UT model).
  * c4_min.qwen_fit_solver — summed_unrolled_depth / _clever_applied_depth /
    the machinery-family map (the STANDARD feed-forward n_layers = summed
    per-op digit-extraction depth).
  * c4_min.forwards_per_step — the (L, F) split that re-books depth into the
    autoregressive token loop (the forwards-per-step mode).

COUNTING RULE (honored exactly, the rule census_total_nonzero already uses):
TOTAL non-zeros = every position in every weight tensor that holds a non-zero
value, REPLICAS COUNTED (a value repeated across dims / layers counts each
time) — NOT distinct values.

SELF-CHECK: the fp64 / whole_value / unrolled row reproduces the measured
26,119; looped reproduces 3,183 (asserted in main / --check).

CPU-only, no model bake, no GPU — a pure parameter census.  Touches no build
file: golden 174ece66 is unchanged.

Run:
    python examples/clever_nonzero_table.py            # print the full table
    python examples/clever_nonzero_table.py --check    # just the self-check
    python examples/clever_nonzero_table.py --md OUT.md # regenerate the doc body
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from c4_min import opconfig as OC
from c4_min import qwen_fit_solver as S
from c4_min import forwards_per_step as FPS
from examples.clever_realtime_model import _cell_nonzero, _framing_nonzero

# ---------------------------------------------------------------------------- #
# Canonical Doom step-counts (from CLEVER_REALTIME_MEASURED) for the min-walltime
# cross-reference only; this file computes NO walltime (that is the measured doc).
# ---------------------------------------------------------------------------- #
RENDER_STEPS = 358_058
RAW_STEPS = 6_889_264

# The measured min-walltime anchor (from docs/CLEVER_REALTIME_MEASURED.md §6): the
# fastest realizable config is bf16 radix-16 digit_extract unrolled (0.009 ms/step).
MIN_WALLTIME_MS = 0.009
MIN_WALLTIME_LABEL = "bf16 radix-16 digit_extract unrolled"

# The config axes to sweep.
PRECISIONS = ("int8", "fp16", "bf16", "fp32", "fp64", "fp128")
BASE_RADICES = (2, 4, 16, 256)          # plus each precision's max-safe radix
EXTRACTIONS = ("nibble", "digit_extract", "whole_value")
MODES = ("unrolled", "looped")          # + forwards_per_step F (added separately)

# The NIBBLE baseline (production 8-bit build, golden 174ece66) — for reference.
NIBBLE_BASELINE_TOTAL = 210_018

# machinery families (ordered) + which op-class drives each family's depth.
FAMILIES = ("arith", "div", "mul", "bitwise", "memory", "trivial")
# op-class that sizes each family's digit-extract / whole-value depth.
_FAMILY_OPCLASS = {"arith": "add", "div": "div", "mul": "mul"}
# result-width (bits) whose digits each family extracts (mul is the 64-bit product).
_FAMILY_RESULT_BITS = {"arith": 32, "div": 32, "mul": 64}


# ---------------------------------------------------------------------------- #
# Per-family DEPTH (stored layers a family contributes to a std-FF UNROLLED net).
# Reuses qwen_fit_solver's _clever_applied_depth / _family_unrolled_depth logic
# but exposes it per (extraction, radix) so we can sweep.
# ---------------------------------------------------------------------------- #
_WHOLE_DEPTH = {"arith": 11, "div": 10, "mul": 20}   # decimal MSB-first places


def _family_depth(family: str, extraction: str, radix: int) -> int:
    """#DISTINCT stored layers one machinery family contributes (unrolled).

    * arith / div / mul: the digit-extraction place count.
        - whole_value: the decimal MSB-first places (11 / 10 / 20).
        - digit_extract: radix-limb decode = ceil(result_bits / log2(radix)).
        - nibble: the production 4-bit lanes (radix 16 over the result width).
    * bitwise: the per-nibble 16x16 LUT applied once per nibble (8).
    * memory: the shared CAM read/write (1).  trivial: register move (1).
    """
    if family in ("bitwise",):
        return 8                        # 8 nibbles over a 32-bit word (radix-16 LUT)
    if family == "memory":
        return 1
    if family == "trivial":
        return 1
    # arith / div / mul digit places
    if extraction == "whole_value":
        return _WHOLE_DEPTH[family]
    bits = _FAMILY_RESULT_BITS[family]
    if extraction == "nibble":
        # nibble = radix-16 limbs over the result width (the production lane count).
        return max(1, math.ceil(bits / 4))
    # digit_extract: radix-limb decode.
    if radix < 2:
        return bits
    return max(1, math.ceil(bits / math.log2(radix)))


# ---------------------------------------------------------------------------- #
# Per-family per-layer NONZERO footprint (radix-aware candidate table).
#
# The REAL cells (clever_realtime_cells.py) give the measured footprints:
#   * arith/div/mul decode cell (whole_value, decimal-10 candidates) = 51
#       = embed 12 + Q/K/V/O identity 16 + candidates 19 + 4 scalars.
#     The candidate table is the ONLY radix-coupled piece: it holds `radix`
#     centres (all d+0.5 > 0 -> radix nonzero) + `radix-1` values (the 0 value is
#     a zero entry) = 2*radix - 1 nonzero.  Decimal (radix 10) -> 19, matching the
#     measured 51.  For a digit_extract radix r the cell is 32 + (2r - 1).
#   * bitwise LUT triple OR+AND+XOR = 2,974 (radix-independent; a 16x16 nibble LUT).
#   * memory CAM = 10 (radix-independent).
# ---------------------------------------------------------------------------- #
def _cell_footprints() -> Dict[str, object]:
    return _cell_nonzero()


def _arith_cell_nonzero(cells: Dict, extraction: str, radix: int) -> int:
    """Per-layer nonzero of the arith/div/mul decode cell at (extraction, radix).

    whole_value uses the fixed decimal-10 candidate cell (== the measured 51).
    digit_extract / nibble use a radix-entry candidate table: 32 (embed 12 +
    QKVO 16 + 4 scalars) + (2*radix - 1) candidate nonzero."""
    base = cells["_arith_cell"]                     # 51 = 32 + 19 (decimal-10 cand)
    if extraction == "whole_value":
        return base                                 # decimal-10 candidates -> 51
    non_cand = base - (2 * 10 - 1)                  # 51 - 19 = 32 (radix-free part)
    cand_r = max(0, 2 * radix - 1)                  # radix centres + (radix-1) values
    return non_cand + cand_r


def _per_family_nonzero(cells: Dict, family: str, extraction: str, radix: int) -> int:
    if family in ("arith", "div", "mul"):
        return _arith_cell_nonzero(cells, extraction, radix)
    if family == "bitwise":
        return cells["bitwise"]
    if family == "memory":
        return cells["_cam"]
    if family == "trivial":
        return 0
    raise KeyError(family)


# ---------------------------------------------------------------------------- #
# The census for ONE config: per-family breakdown + TOTAL, in a chosen mode.
# ---------------------------------------------------------------------------- #
@dataclass
class ConfigCensus:
    precision: str
    radix: int
    extraction: str
    mode: str                       # "unrolled" | "looped" | "forwards_per_step=F"
    n_layers_stored: int            # distinct STORED layers (or reused cells / L)
    applied_depth: int              # layer-applications per VM step (summed depth D)
    forwards_per_step: int          # F (1 unless forwards mode)
    # per-family breakdown (family -> (n_layers, per_layer_nz, total_nz))
    families: Dict[str, Tuple[int, int, int]]
    arithmetic_nz: int
    bitwise_nz: int
    memory_nz: int
    framing_embed_nz: int
    total_nonzero: int


def _family_depths(extraction: str, radix: int) -> Dict[str, int]:
    return {f: _family_depth(f, extraction, radix) for f in FAMILIES}


def _summed_depth(extraction: str, radix: int) -> int:
    """The std-FF unrolled n_layers = summed per-family depth (== D applied)."""
    return sum(_family_depths(extraction, radix).values())


def census_for_config(precision: str, radix: int, extraction: str, mode: str,
                      forwards_per_step: int = 1) -> ConfigCensus:
    """TOTAL nonzero + per-family breakdown for one config in one mode.

    mode:
      * "unrolled"  — STANDARD feed-forward: every family's digit places are
        DISTINCT stored layers; n_layers = summed depth; replicas counted.
      * "looped"    — LOOPED / Universal-Transformer: one stored cell per
        machinery member (ingest+arith / div / mul / bitwise / memory), each
        stored ONCE; applied `depth` times per forward.
      * "forwards_per_step" (forwards_per_step=F>1) — a STANDARD autoregressive
        loop: L = ceil(D / F) physical stored layers, re-invoked F times per VM
        step (the §5 vanilla resolution).  The STORED footprint is the L
        physical layers' machinery (a proportional share of the summed depth).
    """
    cells = _cell_footprints()
    depths = _family_depths(extraction, radix)
    D = sum(depths.values())                        # summed applied depth
    fr = _framing_nonzero(0)                        # embed / lm-head / per-layer
    embed_nz = fr["token_embed"]
    lm_head_nz = fr["lm_head_decode"]
    per_layer_framing = fr["_per_layer_framing"]

    if mode == "looped":
        # LOOPED / UT: 5 stored cells (ingest+arith / div / mul decode + bitwise
        # LUT + CAM), each stored ONCE.  arith/div/mul reuse the decode-cell
        # footprint; framing on the ~6 stored cells.
        arith_cell = _arith_cell_nonzero(cells, extraction, radix)
        fams = {
            "arith": (1, arith_cell, arith_cell),   # ingest+arith decode cell
            "div": (1, arith_cell, arith_cell),     # div reuses decode cell
            "mul": (1, arith_cell, arith_cell),     # mul reuses decode cell
            "bitwise": (1, cells["bitwise"], cells["bitwise"]),
            "memory": (1, cells["_cam"], cells["_cam"]),
            "trivial": (0, 0, 0),
        }
        stored_cells = 6                            # + the trivial move (framing on 6)
        arithmetic = 3 * arith_cell
        bitwise = cells["bitwise"]
        memory = cells["_cam"]
        framing = embed_nz + lm_head_nz + per_layer_framing * stored_cells
        total = arithmetic + bitwise + memory + framing
        return ConfigCensus(
            precision=precision, radix=radix, extraction=extraction, mode="looped",
            n_layers_stored=stored_cells, applied_depth=D, forwards_per_step=1,
            families=fams, arithmetic_nz=arithmetic, bitwise_nz=bitwise,
            memory_nz=memory, framing_embed_nz=framing, total_nonzero=total)

    # --- UNROLLED (and forwards_per_step, which shares the per-family footprint) --
    fams: Dict[str, Tuple[int, int, int]] = {}
    for f in FAMILIES:
        nl = depths[f]
        per = _per_family_nonzero(cells, f, extraction, radix)
        fams[f] = (nl, per, per * nl)

    if forwards_per_step > 1:
        # forwards-per-step (SHALLOW std-FF autoregressive loop): the machinery is
        # re-booked across F forwards, so the PHYSICAL stored layer count is
        # L = ceil(D / F).  The stored nonzero is the machinery carried by those L
        # physical layers.  We keep the family MIX (each family scaled by L/D so
        # the L physical layers still contain every family's cell) — the honest
        # stored footprint of the shallow net.  applied compute L*F ~ D unchanged.
        L = math.ceil(D / forwards_per_step)
        scaled: Dict[str, Tuple[int, int, int]] = {}
        # distribute the L physical layers across families proportional to depth,
        # ceiling each nonzero-bearing family to >=1 so the machinery is present.
        alloc = _allocate_physical_layers(depths, L)
        for f in FAMILIES:
            nl = alloc[f]
            per = _per_family_nonzero(cells, f, extraction, radix)
            scaled[f] = (nl, per, per * nl)
        fams = scaled
        n_layers_stored = sum(nl for nl, _, _ in fams.values())
    else:
        n_layers_stored = D

    arithmetic = sum(fams[f][2] for f in ("arith", "div", "mul"))
    bitwise = fams["bitwise"][2]
    memory = fams["memory"][2]
    framing = embed_nz + lm_head_nz + per_layer_framing * n_layers_stored
    total = arithmetic + bitwise + memory + fams["trivial"][2] + framing
    mode_label = "unrolled" if forwards_per_step == 1 else f"forwards_per_step={forwards_per_step}"
    return ConfigCensus(
        precision=precision, radix=radix, extraction=extraction, mode=mode_label,
        n_layers_stored=n_layers_stored, applied_depth=D,
        forwards_per_step=forwards_per_step, families=fams,
        arithmetic_nz=arithmetic, bitwise_nz=bitwise, memory_nz=memory,
        framing_embed_nz=framing, total_nonzero=total)


def _allocate_physical_layers(depths: Dict[str, int], L: int) -> Dict[str, int]:
    """Distribute L physical layers across families proportional to their unrolled
    depth, giving every nonzero-bearing family >= 1 so the shallow net still
    CONTAINS every op's machinery.  Sum == L (largest-remainder rounding)."""
    D = sum(depths.values())
    nonzero_fams = [f for f in FAMILIES if depths[f] > 0 and f != "trivial"]
    if L >= len(nonzero_fams):
        alloc = {f: (1 if depths[f] > 0 else 0) for f in FAMILIES}
        rem = L - sum(alloc.values())
        # largest-remainder distribute the rest by depth share
        shares = sorted(nonzero_fams, key=lambda f: -depths[f])
        i = 0
        while rem > 0 and shares:
            alloc[shares[i % len(shares)]] += 1
            i += 1
            rem -= 1
        return alloc
    # L smaller than #families (rare): pack into the deepest families.
    alloc = {f: 0 for f in FAMILIES}
    for f in sorted(nonzero_fams, key=lambda f: -depths[f])[:L]:
        alloc[f] = 1
    return alloc


# ---------------------------------------------------------------------------- #
# Geometry (n_layers stored / applied) via the fit solver, for cross-check.
# ---------------------------------------------------------------------------- #
def _opconfig_for(precision: str, radix: int, extraction: str, looped: bool) -> OC.OpConfig:
    """Build a whole-ISA OpConfig at (precision, radix, extraction, recurrence).
    MUL gets fp128 when the body is fp64 whole_value (the 64-bit product) so the
    validator's whole-value path stays coherent — matching min_params_config."""
    recur = "tied" if looped else "unrolled"
    base = OC.AxisConfig(precision=precision, radix=radix, extraction=extraction,
                         recurrence=recur)
    ov: Dict[str, Dict[str, object]] = {}
    if extraction == "whole_value" and precision == "fp64":
        ov["MUL"] = dict(precision="fp128", radix=radix, extraction=extraction,
                         recurrence=recur)
    return OC.OpConfig(base=base, overrides=ov, looped_transformer=looped)


# ---------------------------------------------------------------------------- #
# VALIDITY — the ISA-wide radix ceiling.  A whole-ISA config uses ONE radix, so it
# must satisfy EVERY op; the binding op is the tightest (MUL / DIV).  whole_value
# skips the radix<->precision coupling (radix is only the readout base).
# ---------------------------------------------------------------------------- #
_ISA_OPCLASSES = ("ADD", "CMP", "MUL", "DIV")     # representative per class


def isa_radix_valid(precision: str, radix: int, extraction: str
                    ) -> Tuple[bool, str]:
    """Is (precision, radix) valid for the WHOLE ISA at this extraction?  Returns
    (ok, reason).  whole_value: always ok (no limb decomposition).  Otherwise
    every op-class accMax(radix) must stay <= the precision ceiling; the binding
    op is reported on failure."""
    if radix < 2:
        return False, "radix < 2"
    if extraction == "whole_value":
        return True, ""
    ceiling = OC.PRECISION_CEILING[precision]
    for cls in _ISA_OPCLASSES:
        am = OC.acc_max(cls, radix)
        if am > ceiling:
            return False, f"{cls} accMax {am} > {precision} ceiling {ceiling}"
    return True, ""


def isa_max_safe_radix(precision: str, extraction: str) -> Optional[int]:
    """Largest power-of-two radix valid for the WHOLE ISA (min over op-classes).
    None for whole_value (radix is unbounded there)."""
    if extraction == "whole_value":
        return None
    try:
        return min(OC.max_safe_radix(cls, precision) for cls in _ISA_OPCLASSES)
    except OC.OpConfigError:
        return None


# ---------------------------------------------------------------------------- #
# ENUMERATE the valid config space.
# ---------------------------------------------------------------------------- #
@dataclass
class Row:
    precision: str
    radix: int
    extraction: str
    mode: str
    census: ConfigCensus
    note: str = ""


def _radices_for(precision: str, extraction: str) -> List[int]:
    """The radices to try for (precision, extraction): the base set {2,4,16,256}
    plus the precision's ISA-wide max-safe radix.  whole_value radix is the
    readout base only, so we report a single canonical radix (10, the decimal
    difference-min form the cells use).  ``nibble`` extraction is INTRINSICALLY
    4-bit lanes == radix 16 (the production DEFAULT), so it is reported at radix
    16 only — sweeping nibble across other radices is a category error."""
    if extraction == "whole_value":
        return [10]                                 # decimal readout (the cell's base)
    if extraction == "nibble":
        return [16]                                 # 4-bit lanes == radix 16 (DEFAULT)
    rs = set(BASE_RADICES)
    msr = isa_max_safe_radix(precision, extraction)
    if msr is not None:
        rs.add(msr)
    return sorted(rs)


def enumerate_valid(include_forwards=(3,)) -> Tuple[List[Row], List[Tuple[str, str]]]:
    """Every VALID (precision x radix x extraction x mode) row + the pruned list.

    Returns (rows, pruned) where pruned is a list of (config-label, reason).
    Modes: unrolled + looped for every valid (precision,radix,extraction); plus a
    forwards_per_step=F row for the representative whole-ISA min-nonzero corner
    (to show the §5 vanilla-fit lever)."""
    rows: List[Row] = []
    pruned: List[Tuple[str, str]] = []
    for prec in PRECISIONS:
        for extr in EXTRACTIONS:
            for radix in _radices_for(prec, extr):
                ok, reason = isa_radix_valid(prec, radix, extr)
                label = f"{prec} r{radix} {extr}"
                if not ok:
                    pruned.append((label, reason))
                    continue
                for mode in MODES:
                    c = census_for_config(prec, radix, extr, mode)
                    rows.append(Row(prec, radix, extr, mode, c))
    # forwards_per_step rows for the min-nonzero corner (fp64 whole_value) — show
    # the vanilla-fit lever fitting a stock-24 budget (D=51 -> ceil(51/3)=17).
    for F in include_forwards:
        c = census_for_config("fp64", 10, "whole_value", "forwards_per_step",
                              forwards_per_step=F)
        rows.append(Row("fp64", 10, "whole_value", c.mode, c,
                        note=f"§5 vanilla-fit: D={c.applied_depth} -> "
                             f"L=ceil(D/{F})={c.n_layers_stored} stored layers, "
                             f"fits stock-24"))
    return rows, pruned


# ---------------------------------------------------------------------------- #
# The depth <-> radix <-> nonzero curve for a representative op (DIV).
# ---------------------------------------------------------------------------- #
def depth_radix_curve(op_class: str = "div", bits: int = 32,
                      radices=(2, 16, 256, 65536)) -> List[Dict]:
    """For a representative op, show radix -> (depth, candidate-table, total nonzero)
    where total ~ depth * radix (the §3 curve): the candidate table grows LINEARLY
    in radix while depth falls only LOGARITHMICALLY, so the product GROWS -> min
    nonzero is at SMALL radix + deep.

    depth = ceil(bits / log2(radix)); candidate table = radix entries (2r-1
    nonzero: r centres + r-1 values); total nonzero of the UNROLLED decode chain
    = depth * per-layer-decode-cell (32 fixed + (2r-1) candidate)."""
    out = []
    for r in radices:
        depth = max(1, math.ceil(bits / math.log2(r)))
        cand_nz = 2 * r - 1                          # r centres + (r-1) values
        per_layer = 32 + cand_nz                     # fixed 32 + candidate table
        total = depth * per_layer
        out.append(dict(radix=r, depth=depth, candidate_table=r,
                        candidate_nz=cand_nz, per_layer_nz=per_layer,
                        total_nonzero=total, depth_times_radix=depth * r))
    return out


# ---------------------------------------------------------------------------- #
# SELF-CHECK — reproduce the measured anchors.
# ---------------------------------------------------------------------------- #
def self_check() -> Dict[str, Tuple[int, int, bool]]:
    """Reproduce 26,119 (unrolled) / 3,183 (looped) for fp64 whole_value, and the
    210,018 nibble baseline as a documented reference constant."""
    u = census_for_config("fp64", 10, "whole_value", "unrolled")
    l = census_for_config("fp64", 10, "whole_value", "looped")
    checks = {
        "unrolled fp64 whole_value == 26,119": (u.total_nonzero, 26_119,
                                                u.total_nonzero == 26_119),
        "looped fp64 whole_value == 3,183": (l.total_nonzero, 3_183,
                                             l.total_nonzero == 3_183),
    }
    return checks


# ---------------------------------------------------------------------------- #
# RENDERING.
# ---------------------------------------------------------------------------- #
def _fmt(n: Optional[int]) -> str:
    return f"{n:,}" if n is not None else "n/a"


def render_table(rows: List[Row]) -> str:
    hdr = (f"{'precision':>9s} {'radix':>6s} {'extraction':>13s} "
           f"{'mode':>20s} {'n_lay':>6s} {'applied':>7s} "
           f"{'arith':>7s} {'bitwise':>8s} {'mem':>4s} {'frame+emb':>9s} "
           f"{'TOTAL_nz':>9s}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        c = r.census
        lines.append(
            f"{c.precision:>9s} {c.radix:>6d} {c.extraction:>13s} "
            f"{c.mode:>20s} {c.n_layers_stored:>6d} {c.applied_depth:>7d} "
            f"{c.arithmetic_nz:>7d} {c.bitwise_nz:>8d} {c.memory_nz:>4d} "
            f"{c.framing_embed_nz:>9d} {_fmt(c.total_nonzero):>9s}")
    return "\n".join(lines)


def print_full(rows: List[Row], pruned: List[Tuple[str, str]]):
    print("=" * 110)
    print("CLEVER c4 VM — TOTAL NON-ZERO CENSUS across the full config space")
    print("(precision x radix x extraction x mode).  Replicas counted.")
    print("=" * 110)
    cells = _cell_footprints()
    print(f"Per-family REAL-cell footprints (measured, clever_realtime_cells.py):")
    print(f"   arith/div/mul decode cell (whole_value decimal-10) : "
          f"{cells['_arith_cell']}  (embed 12 + QKVO 16 + cand 19 + 4 scalars)")
    print(f"   bitwise LUT triple OR+AND+XOR                      : "
          f"{cells['bitwise']}  {cells['_bitwise_detail']}")
    print(f"   memory CAM                                         : {cells['_cam']}")
    print(f"   digit_extract radix-r decode cell                  : 32 + (2r-1) "
          f"candidate nonzero")
    print()
    print(f"NIBBLE baseline (production 8-bit build, golden 174ece66) : "
          f"{NIBBLE_BASELINE_TOTAL:,} total nonzero  [REFERENCE]")
    print()
    print(render_table(rows))
    print()
    # Pareto corners.
    valid = [r for r in rows if r.census.total_nonzero is not None]
    min_nz = min(valid, key=lambda r: r.census.total_nonzero)
    # the byte-exact / canonical whole-value corner (the doc's headline 3,183).
    be = census_for_config("fp64", 10, "whole_value", "looped")
    print("PARETO CORNERS:")
    print(f"  MIN-NONZERO (absolute) : {min_nz.census.precision} "
          f"r{min_nz.census.radix} {min_nz.census.extraction} "
          f"{min_nz.census.mode} -> {min_nz.census.total_nonzero:,} total nonzero")
    print(f"     (a low-radix looped config: the radix-r candidate table shrinks "
          f"to 2r-1 < the decimal 19, so it undercuts the whole-value corner;\n"
          f"      NOT byte-exact at full 32-bit unless the precision ceiling "
          f"holds the accumulator — see the validity gate.)")
    print(f"  MIN-NONZERO (byte-exact / canonical) : fp64/fp128 whole_value looped"
          f" -> {be.total_nonzero:,} total nonzero (the doc headline; slowest "
          f"datapath)")
    print(f"  MIN-WALLTIME : {MIN_WALLTIME_LABEL} -> {MIN_WALLTIME_MS} ms/step "
          f"(from CLEVER_REALTIME_MEASURED §6; fastest realizable datapath, "
          f"25,624 nonzero at 42L)")
    print(f"     (NB the measured min-walltime shape gives MUL a fp16 datapath "
          f"override — MUL's r16 accumulator 1904 overflows bf16's 256 ceiling,\n"
          f"      so a PURE-bf16 whole-ISA r16 is pruned below; the measured "
          f"corner is bf16-body + fp16-MUL, min_walltime_config in opconfig.)")
    print()
    print(f"PRUNED CONFIGS ({len(pruned)}) — radix overflows the precision "
          f"exact-int ceiling for the binding op (MUL/DIV):")
    for label, reason in pruned:
        print(f"   {label:32s} {reason}")
    print()
    # depth<->radix curve.
    print("DEPTH <-> RADIX <-> NONZERO curve (representative op: DIV, 32-bit):")
    print(f"   {'radix':>7s} {'depth':>6s} {'cand_table':>11s} "
          f"{'per_layer_nz':>13s} {'TOTAL_nz':>9s} {'depth*radix':>12s}")
    for e in depth_radix_curve("div"):
        print(f"   {e['radix']:>7d} {e['depth']:>6d} {e['candidate_table']:>11d} "
              f"{e['per_layer_nz']:>13d} {e['total_nonzero']:>9d} "
              f"{e['depth_times_radix']:>12d}")
    print("   -> total ~ depth*radix: candidate table grows LINEARLY in radix, "
          "depth falls only LOG -> product GROWS. Min nonzero at SMALL radix.")
    print()
    # self-check.
    print("SELF-CHECK (reproduce the measured anchors):")
    for name, (got, want, ok) in self_check().items():
        print(f"   [{'OK' if ok else 'FAIL'}] {name:44s} got {got:,} want {want:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="just the self-check")
    ap.add_argument("--md", default=None, help="write the doc body to this path")
    args = ap.parse_args()

    if args.check:
        ok_all = True
        for name, (got, want, ok) in self_check().items():
            ok_all = ok_all and ok
            print(f"[{'OK' if ok else 'FAIL'}] {name}: got {got:,} want {want:,}")
        raise SystemExit(0 if ok_all else 1)

    rows, pruned = enumerate_valid()
    print_full(rows, pruned)

    # hard self-check gate (fail loudly if the anchors drift).
    checks = self_check()
    if not all(ok for _, _, ok in checks.values()):
        raise SystemExit("SELF-CHECK FAILED — anchors 26,119 / 3,183 not reproduced")

    if args.md:
        with open(args.md, "w") as f:
            f.write(render_markdown(rows, pruned))
        print(f"\nwrote {args.md}")


def render_markdown(rows: List[Row], pruned: List[Tuple[str, str]]) -> str:
    """Emit the markdown table body (used to regenerate the doc)."""
    cells = _cell_footprints()
    out = []
    out.append("| precision | radix | extraction | mode | n_lay | applied | "
               "arith | bitwise | mem | frame+emb | **TOTAL nonzero** |")
    out.append("|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|")
    out.append(f"| nibble (8-bit) | 16 | nibble | unrolled | 123 | — | — | — | "
               f"— | — | **{NIBBLE_BASELINE_TOTAL:,}** (baseline) |")
    valid = [r for r in rows if r.census.total_nonzero is not None]
    min_nz = min(valid, key=lambda r: r.census.total_nonzero)
    for r in rows:
        c = r.census
        star = ""
        if r is min_nz:
            star = " ⬅ MIN-NONZERO"
        out.append(
            f"| {c.precision} | {c.radix} | {c.extraction} | {c.mode} | "
            f"{c.n_layers_stored} | {c.applied_depth} | {c.arithmetic_nz:,} | "
            f"{c.bitwise_nz:,} | {c.memory_nz} | {c.framing_embed_nz} | "
            f"**{c.total_nonzero:,}**{star} |")
    return "\n".join(out)


if __name__ == "__main__":
    main()
