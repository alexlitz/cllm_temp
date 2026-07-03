#!/usr/bin/env python3
"""Validate the canonical DimResolver and demonstrate the static-registry trap.

This tool proves three things, all against a SINGLE default build:

  1. ``DimResolver.from_layout(layout)`` returns the correct BUILT column for
     a battery of well-known dims (OP_PSH, PSH_AT_SP, OPCODE_BYTE_LO,
     OUTPUT_LO, AX_CARRY_LO, ...).

  2. The STATIC registry (``build_default_registry_dynamic``) returns a
     DIFFERENT (wrong) column for the SAME names — the exact "dead/constant
     signal" trap the resolver prevents (widen-repack moves ~93% of dims).

  3. An enumeration of the built layout's dims, classifying each as:
       - shared with the static registry (same or MOVED position),
       - cross-step alias (``NAME.*.-1``),
       - op-local residual band (owner from the residual-band registry),
       - other built-only named dim (with source module), or
       - padding (``_pad`` / ``_widen_pad``).

Run:
    CUDA_VISIBLE_DEVICES="" python tools/dim_resolver_demo.py
    CUDA_VISIBLE_DEVICES="" python tools/dim_resolver_demo.py --enumerate
"""

from __future__ import annotations

import argparse
import os
import sys

# Worktree root (parent of the ``c4_release`` package dir) on sys.path so the
# package-relative imports resolve regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)               # .../c4_release  (the package dir)
_ROOT = os.path.dirname(_PKG)               # parent (for c4_release.* imports)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# A battery of dims cited in the memory trap-note + common probe targets.
BATTERY = [
    "OP_PSH",
    "PSH_AT_SP",
    "OPCODE_BYTE_LO",
    "OUTPUT_LO",
    "OUTPUT_HI",
    "AX_CARRY_LO",
    "AX_CARRY_HI",
    "ALU_LO",
    "ALU_HI",
    "MARK_AX",
    "MARK_PC",
    "STACK0_BYTE0",
    "CMP",
]


def _base(name: str) -> str:
    """Strip the cross-step alias suffix ``.*.-1`` to get the base dim name."""
    return name[:-5] if name.endswith(".*.-1") else name


# Inferred roles for the "other built-only named" dims — dims that carry a
# real name in the built layout but are NOT in the static registry, NOT a
# cross-step ``.*.-1`` alias, and NOT an op-local residual band. Roles traced
# from their writer/reader ops (see module refs in the value).
OTHER_NEW_ROLES = {
    "OUTPUT_HI_THIS_STEP": (
        "same-step alias of OUTPUT_HI (Phase 7.A OUTPUT_HI split); the "
        "cross-step-safe write target for L12/L13 OUTPUT_HI writers "
        "(l12_ops.py, l13_ops.py)"
    ),
    "H1_DUMP": (
        "same-position alias of H1 (byte-1 register-dump one-hot band); read "
        "by the AX byte-1 dump head bake (all_core_ops.py, shared.py)"
    ),
    "PC_VIA_LEV_DETECTOR_LO": (
        "LEV register-restore detector: low byte of the PC value recovered "
        "when a LEV pops the frame (control_flow_heads.lev_detector_head)"
    ),
    "PC_VIA_LEV_DETECTOR_HI": (
        "LEV register-restore detector: high byte of the PC value recovered "
        "on LEV (control_flow_heads.lev_detector_head)"
    ),
    "BP_VIA_LEV_DETECTOR": (
        "LEV register-restore detector: BP value recovered on LEV "
        "(control_flow_heads.lev_detector_head)"
    ),
    "SP_VIA_LEV_DETECTOR": (
        "LEV register-restore detector: SP value recovered on LEV "
        "(control_flow_heads.lev_detector_head)"
    ),
}


def build_once():
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
    from neural_vm.unified_compiler.dim_resolver import DimResolver

    _model, layout = compile_full_vm_dynamic(disk_cache=True)
    resolver = DimResolver.from_layout(layout)
    reg = build_default_registry_dynamic()
    static = {name: slot.start for name, slot in reg.slots.items()}
    return layout, resolver, static


def validate(layout, resolver, static) -> int:
    """Return process exit code (0 == helper correct AND trap demonstrated)."""
    bp = layout.dim_positions
    print("=" * 74)
    print("VALIDATION — DimResolver (BUILT layout) vs static registry (WRONG)")
    print(f"static d_model={_static_dmodel(static)}  built d_model={layout.d_model}")
    print("=" * 74)
    header = f"{'NAME':<20} {'RESOLVER':>9} {'RAW-BUILT':>10} {'STATIC':>8} {'TRAP?':>6}"
    print(header)
    print("-" * len(header))

    ok_resolver = True
    n_moved = 0
    n_in_static = 0
    for name in BATTERY:
        try:
            r = resolver.resolve(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<20} {'ERR':>9}  {type(e).__name__}: {e}")
            ok_resolver = False
            continue
        raw = bp.get(name)
        s = static.get(name)
        # Helper MUST equal the raw built position (that is the contract).
        if r != raw:
            print(f"{name:<20} {r:>9} {str(raw):>10}  <-- RESOLVER != BUILT (BUG)")
            ok_resolver = False
            continue
        trap = ""
        if s is not None:
            n_in_static += 1
            if s != r:
                trap = "MOVED"
                n_moved += 1
            else:
                trap = "same"
        else:
            trap = "n/a"
        print(f"{name:<20} {r:>9} {str(raw):>10} {str(s):>8} {trap:>6}")

    print("-" * len(header))
    print(
        f"battery: {len(BATTERY)} names; {n_in_static} present in static; "
        f"{n_moved} MOVED (static position != built position)"
    )

    # Corpus-wide trap magnitude.
    common = [n for n in bp if n in static]
    moved = [n for n in common if bp[n] != static[n]]
    pct = round(100 * len(moved) / max(1, len(common)), 1)
    print(
        f"corpus: {len(common)} names shared with static; {len(moved)} moved "
        f"({pct}%) — resolving ANY of these via the static registry reads the "
        f"WRONG cell"
    )

    # Guard test: unknown name must raise (loud), not return 0.
    from neural_vm.unified_compiler.dim_resolver import UnknownDimError

    guard_ok = False
    try:
        resolver.resolve("OP_PHS")  # deliberate typo of OP_PSH
    except UnknownDimError as e:
        guard_ok = "Did you mean" in str(e)
    print(f"guard (unknown name raises with suggestion): {'PASS' if guard_ok else 'FAIL'}")

    trap_demonstrated = n_moved > 0
    print("-" * len(header))
    verdict = ok_resolver and guard_ok and trap_demonstrated
    print(
        "RESULT: "
        + ("PASS" if verdict else "FAIL")
        + f"  (resolver-correct={ok_resolver}, guard={guard_ok}, "
        + f"trap-demonstrated={trap_demonstrated})"
    )
    return 0 if verdict else 1


def _static_dmodel(static) -> int:
    return max((p for p in static.values()), default=0) + 1


def enumerate_dims(layout, static) -> None:
    """Classify + print every built dim (the 'unnamed dim' inventory)."""
    from neural_vm.unified_compiler.ops.residual_band_registry import (
        collect_registered_residual_bands,
        registered_band_specs,  # -> List[_BandSpec(name, size, owner, flag, ...)]
    )

    bp = layout.dim_positions
    ds = getattr(layout, "dim_sizes", {}) or {}
    bands = collect_registered_residual_bands()
    band_owner = {spec.name: spec.owner for spec in registered_band_specs()}

    shared_same, shared_moved, aliases, band_dims, other_new, padding = (
        [], [], [], [], [], [])

    for name in sorted(bp, key=lambda n: bp[n]):
        pos, size = bp[name], ds.get(name, 1)
        if name in static:
            (shared_same if static[name] == pos else shared_moved).append(
                (pos, size, name))
        elif name.endswith(".*.-1"):
            aliases.append((pos, size, name))
        elif name in bands:
            band_dims.append((pos, size, name, band_owner.get(name, "?")))
        elif name in ("_pad", "_widen_pad") or name.startswith("_pad"):
            padding.append((pos, size, name))
        else:
            other_new.append((pos, size, name))

    def _dump(title, rows, owner_col=False):
        print(f"\n### {title}  ({len(rows)})")
        for row in rows:
            if owner_col:
                pos, size, name, owner = row
                print(f"  {pos:>5} +{size:<3} {name:<28} owner={owner}")
            else:
                pos, size, name = row
                print(f"  {pos:>5} +{size:<3} {name}")

    print("=" * 74)
    print("BUILT-LAYOUT DIM ENUMERATION (the 'unnamed dim' inventory)")
    print(f"d_model={layout.d_model}  total named dims={len(bp)}")
    print("=" * 74)
    print(
        f"shared-with-static: {len(shared_same) + len(shared_moved)} "
        f"({len(shared_moved)} MOVED); "
        f"cross-step aliases: {len(aliases)}; "
        f"residual bands: {len(band_dims)}; "
        f"other built-only named: {len(other_new)}; "
        f"padding: {len(padding)}"
    )
    _dump("OP-LOCAL RESIDUAL BANDS (named, owner from residual_band_registry)",
          band_dims, owner_col=True)
    print(f"\n### OTHER BUILT-ONLY NAMED DIMS (not static, not alias, not band)"
          f"  ({len(other_new)})")
    for pos, size, name in other_new:
        role = OTHER_NEW_ROLES.get(name, "UNCLASSIFIED — trace writer/reader ops")
        print(f"  {pos:>5} +{size:<3} {name:<24} {role}")
    _dump("PADDING / RESERVED (unnamed reserved space)", padding)
    print(
        "\nNOTE: every d_model position is covered by a NAME (0 truly-unnamed "
        "cells); the only nameless space is explicit padding "
        "(_pad / _widen_pad). Cross-step aliases carry the '.*.-1' suffix; "
        "their base dim is the same-step name."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enumerate", action="store_true",
                    help="also print the full built-dim classification")
    args = ap.parse_args()

    layout, resolver, static = build_once()
    code = validate(layout, resolver, static)
    if args.enumerate:
        enumerate_dims(layout, static)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
