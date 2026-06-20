#!/usr/bin/env python
"""OUTPUT-band self-reinforcement audit (mega-root #2 analog of lint_positional).

THE ROOT (project_output_band_self_reinforcement_megaroot)
----------------------------------------------------------
A ``FFNRule`` whose SwiGLU GATE reads an ``OUTPUT_*`` dim AND whose ``writes``
target an ``OUTPUT_*`` dim is a POSITIVE-FEEDBACK loop. SwiGLU per-unit math is
``silu(up) * gate`` (``up`` = ``conditions``, ``gate`` = ``gate`` + ``gate_terms``).
When a term reading ``OUTPUT_LO/HI`` appears in EITHER branch and the unit
re-writes the SAME OUTPUT band, any residue on that dim is read, amplified, and
re-written within the same forward pass -> bootstraps to ~1e27..5e29 and decides
the byte by accident (the framing-drift mega-root).

WHAT THIS TOOL DOES
-------------------
Walks every FFN rule in ``all_core_ops()`` (+ ALU post-ops) at the IR level (NO
model bake — fast, CPU-only, read-only). For each rule it records whether it
READS an OUTPUT* dim in the GATE branch (``gate`` / ``gate_terms``), READS one
in the UP branch (``conditions``), and WRITES one (``writes``). A rule is
**self-reinforcing** iff it READS (gate or up) an OUTPUT dim AND WRITES an
OUTPUT dim. The GATE-read variant is the dangerous one (the multiplicative term
is what bootstraps); the UP-read variant is the milder additive case.

Ranking: aggregates self-reinforcing rules by owning op (a cluster proxy) and
by the OUTPUT sub-band (LO / HI / HI_THIS_STEP / HI_PREV_STEP), so the catalog
shows which families gate which clusters. ``OUTPUT_HI_PREV_STEP`` /
``OUTPUT_HI_THIS_STEP`` reads are flagged separately: per ssa_dim.py the
PREV_STEP alias resolves to the SAME physical slot as the base band, so a
PREV_STEP read does NOT physically break the feedback today (see the design
note) — the tool surfaces that distinction.

Usage::

    CUDA_VISIBLE_DEVICES="" python tools/lint_output_selfreinforce.py
    CUDA_VISIBLE_DEVICES="" python tools/lint_output_selfreinforce.py --json
    CUDA_VISIBLE_DEVICES="" python tools/lint_output_selfreinforce.py --top 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from neural_vm.unified_compiler.ops.all_core_ops import (  # noqa: E402
    all_alu_postop_attach_ops,
    all_core_ops,
)
from neural_vm.unified_compiler.ir import _operation_compiler_ir  # noqa: E402
from neural_vm.unified_compiler.ssa_dim import base_of  # noqa: E402


OUTPUT_BANDS = (
    "OUTPUT_LO",
    "OUTPUT_HI",
    "OUTPUT_HI_THIS_STEP",
    "OUTPUT_HI_PREV_STEP",
)

# PREV_STEP aliases resolve to the SAME physical slot (ssa_dim.py): a read
# through one of these names does NOT physically read a different cell today.
PREV_STEP_BANDS = ("OUTPUT_HI_PREV_STEP",)


def _is_output(name: str) -> bool:
    return base_of(name) in OUTPUT_BANDS


def _band(name: str) -> str:
    return base_of(name)


def _collect_rules(op, dim_positions, head_dim: int):
    """Return the list of FFN rules for ``op`` (empty for non-FFN ops)."""
    ir = _operation_compiler_ir(op, dim_positions=dim_positions, head_dim=head_dim)
    if ir is None:
        return []
    rules = []
    try:
        layers = ir.layers
    except AttributeError:
        return []
    for layer in layers:
        ffn = getattr(layer, "ffn", None)
        if ffn is None:
            continue
        rules.extend(getattr(ffn, "rules", ()))
    return rules


def _dim_positions_stub() -> Dict[str, int]:
    """A name->position map sufficient for factories that resolve dims.

    Most ``compiler_ir`` ops are static (don't need positions); the few
    factories that DO take ``dim_positions`` only index by NAME. We build the
    real registry so any resolve() inside a factory succeeds.
    """
    from neural_vm.dim_registry_dynamic import build_dim_registry

    reg = build_dim_registry()
    # build_dim_registry returns an object with .positions or a dict
    if hasattr(reg, "positions"):
        return dict(reg.positions)
    if isinstance(reg, dict):
        return dict(reg)
    # registry exposes name->start via .items()
    return {name: start for name, start in reg}


def _dim_positions_built() -> Tuple[Dict[str, int], int]:
    """The REAL built layout's dim_positions + head_dim.

    Used by ``--use-built-layout`` so the over-width-band ops (whose
    factories resolve ``H1_PREV_STEP`` / ``STACK0_B0_*`` and so KeyError on the
    static registry) are covered too. Costs one CPU build.
    """
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    _model, layout = compile_full_vm_dynamic(disk_cache=False)
    head_dim = getattr(layout, "head_dim", None) or 109
    return dict(layout.dim_positions), int(head_dim)


def audit(use_built_layout: bool = False) -> Dict:
    core = all_core_ops(
        enable_conversational_io=True,
        enable_tool_calling=True,
        enable_neural_io_think_protocol=True,
    )
    postops = all_alu_postop_attach_ops()
    ops = core + postops

    if use_built_layout:
        dim_positions, head_dim = _dim_positions_built()
    else:
        try:
            dim_positions = _dim_positions_stub()
        except Exception:
            dim_positions = {}
        head_dim = 109  # base head_dim; only used if a factory resolves

    # Per-rule findings.
    findings: List[Dict] = []
    # Aggregates.
    by_op: Dict[str, Dict] = defaultdict(
        lambda: {
            "self_reinforce_gate": 0,
            "self_reinforce_up": 0,
            "writes_output": 0,
            "reads_output_gate": 0,
            "reads_output_up": 0,
            "write_bands": defaultdict(int),
            "gate_read_bands": defaultdict(int),
        }
    )

    for op in ops:
        op_name = getattr(op, "name", repr(op))
        try:
            rules = _collect_rules(op, dim_positions, head_dim)
        except Exception as exc:  # a factory needing a real layout — skip
            findings.append(
                {"op": op_name, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        for rule in rules:
            gate_reads = []
            if rule.gate is not None and _is_output(rule.gate.name):
                gate_reads.append(rule.gate.key())
            for t in rule.gate_terms:
                if _is_output(t.dim.name):
                    gate_reads.append(t.dim.key())
            up_reads = [
                t.dim.key() for t in rule.conditions if _is_output(t.dim.name)
            ]
            writes = [w.dim.key() for w in rule.writes if _is_output(w.dim.name)]
            if not writes:
                continue
            agg = by_op[op_name]
            agg["writes_output"] += 1
            for w in writes:
                agg["write_bands"][_band(w)] += 1
            if gate_reads:
                agg["reads_output_gate"] += 1
                for g in gate_reads:
                    agg["gate_read_bands"][_band(g)] += 1
            if up_reads:
                agg["reads_output_up"] += 1
            sr_gate = bool(gate_reads)
            sr_up = bool(up_reads)
            if sr_gate:
                agg["self_reinforce_gate"] += 1
            if sr_up:
                agg["self_reinforce_up"] += 1
            if sr_gate or sr_up:
                prev_only = all(_band(r) in PREV_STEP_BANDS for r in gate_reads + up_reads)
                findings.append(
                    {
                        "op": op_name,
                        "rule": rule.name,
                        "gate_reads": gate_reads,
                        "up_reads": up_reads,
                        "writes": writes,
                        "kind": "GATE" if sr_gate else "UP",
                        "prev_step_only": prev_only,
                    }
                )

    return {"findings": findings, "by_op": {k: _finalize(v) for k, v in by_op.items()}}


def _finalize(agg: Dict) -> Dict:
    out = dict(agg)
    out["write_bands"] = dict(agg["write_bands"])
    out["gate_read_bands"] = dict(agg["gate_read_bands"])
    return out


def _rank(by_op: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
    sr = [
        (name, rec)
        for name, rec in by_op.items()
        if rec["self_reinforce_gate"] or rec["self_reinforce_up"]
    ]
    sr.sort(
        key=lambda kv: (kv[1]["self_reinforce_gate"], kv[1]["self_reinforce_up"]),
        reverse=True,
    )
    return sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=0, help="limit ranked op list")
    ap.add_argument(
        "--use-built-layout",
        action="store_true",
        help="resolve dims via a real CPU build (covers over-width-band ops)",
    )
    args = ap.parse_args()

    result = audit(use_built_layout=args.use_built_layout)
    by_op = result["by_op"]
    ranked = _rank(by_op)
    if args.top:
        ranked = ranked[: args.top]

    if args.json:
        print(json.dumps({"by_op": dict(ranked), "findings": result["findings"]}, indent=2))
        return 0

    total_gate = sum(r["self_reinforce_gate"] for _, r in ranked)
    total_up = sum(r["self_reinforce_up"] for _, r in ranked)
    print("=" * 78)
    print("OUTPUT-band SELF-REINFORCEMENT audit (mega-root #2)")
    print("=" * 78)
    print(
        f"self-reinforcing ops: {len(ranked)}   "
        f"GATE-read rules: {total_gate}   UP-read rules: {total_up}"
    )
    errs = [f for f in result["findings"] if "error" in f]
    if errs:
        print(f"(skipped {len(errs)} ops whose factory needs a real layout)")
    print("-" * 78)
    print(
        f"{'OP':46s} {'GATE':>5s} {'UP':>4s} {'WROUT':>6s}  write/gate-read bands"
    )
    print("-" * 78)
    for name, rec in ranked:
        bands = (
            "w=" + ",".join(f"{b}:{c}" for b, c in sorted(rec["write_bands"].items()))
        )
        if rec["gate_read_bands"]:
            bands += "  g=" + ",".join(
                f"{b}:{c}" for b, c in sorted(rec["gate_read_bands"].items())
            )
        print(
            f"{name[:46]:46s} {rec['self_reinforce_gate']:5d} "
            f"{rec['self_reinforce_up']:4d} {rec['writes_output']:6d}  {bands}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
