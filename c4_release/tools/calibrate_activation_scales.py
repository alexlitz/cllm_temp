"""Build-time ACTIVATION-SCALE calibration (task #395).

Measures each CONTROL-gate discriminator dim's characteristic RUNTIME activation
magnitude in the residual stream the gate reads, over a small representative
program set, and writes the per-``(dim, position_class)`` scale table consumed by
``neural_vm.verification.activation_scales`` / ``ops.shared.derive_gate`` when
``C4_DERIVE_GATE_SCALES=1``.

This is the calibration that POPULATES the missing spec datum the CONTROL branch
flagged (docs/DERIVE_CONTROL_2026_07_09.md §6): the gate's positive weights +
threshold encode the per-dim activation scales, and this tool MEASURES them so
``derive_gate`` sets ``weight = 1/activation_scale`` from data instead of hand
tuning. The measured table matches the baked canonical fallback
(``activation_scales._CANONICAL_SCALES``) — the JSON is an OPTIONAL refresh, not
a golden-build prerequisite (the golden build never depends on a runtime
artifact; the canonical table is the fallback).

Usage (from the c4_release package dir, CUDA hidden for determinism):

    PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python tools/calibrate_activation_scales.py \
        --output .agent-logs/activation-scales/v1.json --ids 350,375,400

Mechanism: hook every TransformerBlock's FFN input (the post-attention residual
the gate rules read), teacher-force each program's symbolic context, and record —
per (dim, position_class) — the characteristic (median-of-active) residual value.
The scale a gate uses is measured at the gate's firing position class (``mark==PC``
for the branch gates); the tool records ALL classes so any gate can query.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_C4_RELEASE_DIR = os.path.dirname(_THIS_DIR)
if _C4_RELEASE_DIR not in sys.path:
    sys.path.insert(0, _C4_RELEASE_DIR)


# The discriminator dims whose activation scale the CONTROL gates depend on
# (markers, opcode one-hots, branch/step flags). Extend as more gate families
# are migrated to derive_gate.
_GATE_DIMS = [
    "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
    "MARK_SE", "IS_BYTE", "HAS_SE",
    "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
    "CMP+0", "CMP+4", "CMP+5",
]


def _resolve_dim(dp, name: str):
    """Resolve a base name or ``name+offset`` to a residual column."""
    if "+" in name:
        base, off = name.rsplit("+", 1)
        if base in dp:
            return dp[base] + int(off)
        return None
    return dp.get(name)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_C4_RELEASE_DIR
        ).decode().strip()
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--ids", default="350,375,400",
        help="Comma-separated 1096 program ids to calibrate over (default: the "
        "CONTROL pilot if_gt/if_lt/if_eq)",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    import torch
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    from tests.test_suite_1000 import generate_test_programs
    from tools.observe_backbone_contributions import (
        _build_symbolic_context, _classify_positions,
    )
    from src.compiler import compile_c

    print("[calibrate] building model (CPU, CUDA hidden)...", flush=True)
    model, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = layout.dim_positions
    device = next(model.parameters()).device

    cols = {d: _resolve_dim(dp, d) for d in _GATE_DIMS}
    cols = {d: c for d, c in cols.items() if c is not None}

    # (dim, position_class) -> list of residual values (over all blocks + progs).
    # We take the characteristic scale as the median of the ACTIVE (|v| above a
    # small floor) values at that dim's own class, aggregated across blocks — the
    # scale is a structural property of the compiled model, stable across blocks.
    samples = defaultdict(list)

    captured = {}
    def mk_hook(bi):
        def hook(mod, inp, out):
            captured[bi] = inp[0].detach()[0].to("cpu")
        return hook
    handles = [
        model.blocks[bi].ffn.register_forward_hook(mk_hook(bi))
        for bi in range(len(model.blocks))
    ]

    # Map each dim to the position classes it is relevant at.
    def relevant_classes(dim: str, buckets):
        out = []
        if dim.startswith("MARK_"):
            mk = dim.split("MARK_", 1)[1]
            if f"mark=={mk}" in buckets:
                out.append(f"mark=={mk}")
        elif dim.startswith("OP_") or dim.startswith("CMP"):
            # opcode / cmp flags live at the PC (opcode) row
            if "mark==PC" in buckets:
                out.append("mark==PC")
        elif dim == "IS_BYTE":
            for b in buckets:
                if b.startswith("is_byte"):
                    out.append("mark==PC")  # canonical unit-scale class
                    break
        elif dim == "HAS_SE":
            if "has_se" in buckets:
                out.append("has_se")
        return out

    allt = generate_test_programs()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    for idx in ids:
        src, exp, desc = allt[idx]
        try:
            bc, data = compile_c(src)
            context = _build_symbolic_context(bc, data)
        except Exception as exc:
            print(f"[calibrate] id={idx} prep failed: {exc!r}", flush=True)
            continue
        if len(context) > args.max_seq_len:
            context = context[-args.max_seq_len:]
        classes = _classify_positions(context)
        tok = torch.tensor([context], dtype=torch.long, device=device)
        if hasattr(model.embed, "set_mem_history_end"):
            model.embed.set_mem_history_end(0)
        with torch.no_grad():
            _ = model(tok)
        for bi in sorted(captured):
            resid = captured[bi]
            seq = resid.shape[0]
            for pos in range(min(seq, len(classes))):
                buckets = classes[pos]
                for dim, col in cols.items():
                    for cls in relevant_classes(dim, buckets):
                        v = float(resid[pos, col].item())
                        if abs(v) > 1e-3:  # active
                            samples[(dim, cls)].append(abs(v))
        print(f"[calibrate] id={idx} {desc} done", flush=True)
    for h in handles:
        h.remove()

    # Characteristic scale = median of active magnitudes at the dim's own class.
    scales = defaultdict(dict)
    for (dim, cls), vals in samples.items():
        if not vals:
            continue
        s = float(statistics.median(vals))
        scales[dim][cls] = round(s, 4)
        # also expose under wildcard "*" so a query without a class resolves.
        scales[dim]["*"] = round(s, 4)

    out = {
        "version": 1,
        "corpus_size": len(ids),
        "model_commit": _git_sha(),
        "note": "per-(dim, position_class) runtime activation scale; a gate "
                "positive weight = 1/scale (ops.shared.derive_gate).",
        "scales": {d: dict(cm) for d, cm in scales.items()},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] wrote {len(scales)} dims to {output_path}", flush=True)
    for d in sorted(scales):
        print(f"    {d}: {scales[d].get('*')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
