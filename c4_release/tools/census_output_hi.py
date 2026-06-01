"""B9 OUTPUT_HI split census.

For every op declared in ``all_core_ops()`` (+ postops), classify its
relationship to the ``OUTPUT_HI`` residual-dim family into:

  PRODUCER          : declares OUTPUT_HI in ``writes`` or ``produces``.
  SAME_STEP_READER  : declares OUTPUT_HI in ``reads`` AND its
                      ``current_layer`` is strictly greater than the
                      latest current_layer of any in-step producer.
                      (i.e. the read is satisfied by THIS step's prior FFN.)
  CROSS_STEP_READER : declares OUTPUT_HI in ``reads`` AND its current_layer
                      is at or below the earliest producer's layer — the
                      read can only be satisfied by the PREVIOUS step.
                      Also any op that reads OUTPUT_HI as a carry-forward
                      seed (heuristic: name contains "carry_forward" or
                      "prev" or it is at L0..L3 with no in-step producer).
  MIXED             : declares OUTPUT_HI in ``reads`` AND lives at a layer
                      where BOTH this-step and previous-step semantics are
                      simultaneously possible (some producer is earlier,
                      some is later, OR the op also writes).

  CONSUMES_FRESH    : declares OUTPUT_HI in ``consumes_fresh`` (treated as
                      SAME_STEP_READER, recorded for trace-ability).

The script is read-only. It imports the production op factories and
inspects declarative fields only.

Usage::

    python tools/census_output_hi.py
    python tools/census_output_hi.py --out /tmp/output_hi_census.md
    python tools/census_output_hi.py --json   # stable machine output
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from neural_vm.unified_compiler.layer_compiler import Operation  # noqa: E402
from neural_vm.unified_compiler.ops.all_core_ops import (  # noqa: E402
    all_alu_postop_attach_ops,
    all_core_ops,
)


DIM = "OUTPUT_HI"


# Names that signal CROSS-STEP carry semantics (read from previous step's
# OUTPUT_HI register). Heuristic — overridden by per-op manual review below.
CROSS_STEP_NAME_HINTS = (
    "carry_forward",
    "prev_step",
    "from_prev",
    "alibi",  # L0 ALiBi relay is the canonical cross-step path
)


def _current_layer(op: Operation) -> Optional[int]:
    if op.layer_idx is not None:
        return op.layer_idx
    if op.phase is None:
        return None
    try:
        return int(math.floor(op.phase))
    except (TypeError, ValueError):
        return None


def collect_ops() -> List[Operation]:
    core = all_core_ops(
        enable_conversational_io=True,
        enable_tool_calling=True,
        enable_neural_io_think_protocol=True,
    )
    postops = all_alu_postop_attach_ops()
    return core + postops


def classify(ops: List[Operation]) -> Dict[str, Dict]:
    """Return ``{op_name: record}`` for every op touching OUTPUT_HI."""
    # First pass: find producers (writes or produces OUTPUT_HI) and gather
    # their layers.
    producer_layers: List[int] = []
    for op in ops:
        if DIM in op.writes or DIM in op.produces:
            cl = _current_layer(op)
            if cl is not None:
                producer_layers.append(cl)
    min_prod_layer = min(producer_layers) if producer_layers else None
    max_prod_layer = max(producer_layers) if producer_layers else None

    records: Dict[str, Dict] = {}
    for op in ops:
        touches = {
            "writes": DIM in op.writes,
            "reads": DIM in op.reads,
            "produces": DIM in op.produces,
            "consumes_fresh": DIM in op.consumes_fresh,
        }
        if not any(touches.values()):
            continue
        cl = _current_layer(op)
        cat = _categorise(op, cl, touches, min_prod_layer, max_prod_layer)
        records[op.name] = {
            "name": op.name,
            "kind": op.kind,
            "current_layer": cl,
            "phase": op.phase,
            "layer_idx": op.layer_idx,
            "writes_OUTPUT_HI": touches["writes"],
            "reads_OUTPUT_HI": touches["reads"],
            "produces_OUTPUT_HI": touches["produces"],
            "consumes_fresh_OUTPUT_HI": touches["consumes_fresh"],
            "category": cat,
            "produces_register": op.produces.get(DIM),
            "consumes_fresh_register": op.consumes_fresh.get(DIM),
        }
    return records


def _categorise(
    op: Operation,
    cl: Optional[int],
    touches: Dict[str, bool],
    min_prod_layer: Optional[int],
    max_prod_layer: Optional[int],
) -> str:
    name = op.name.lower()
    is_producer = touches["writes"] or touches["produces"]
    is_reader = touches["reads"] or touches["consumes_fresh"]

    if is_producer and is_reader:
        return "MIXED"
    if is_producer:
        return "PRODUCER"
    if not is_reader:
        return "NONE"
    # is_reader only
    # Cross-step heuristics: name hint, or layer strictly earlier than the
    # earliest in-step producer.
    if any(h in name for h in CROSS_STEP_NAME_HINTS):
        return "CROSS_STEP_READER"
    if cl is not None and min_prod_layer is not None and cl < min_prod_layer:
        return "CROSS_STEP_READER"
    if touches["consumes_fresh"]:
        # consumes_fresh by construction means an in-step producer is
        # required. That is a SAME_STEP read.
        return "SAME_STEP_READER"
    if cl is not None and min_prod_layer is not None and cl > min_prod_layer:
        return "SAME_STEP_READER"
    # Read at the same layer as the earliest producer: could be either.
    return "MIXED"


def render(records: Dict[str, Dict]) -> str:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for rec in records.values():
        by_cat[rec["category"]].append(rec)
    for cat in by_cat:
        by_cat[cat].sort(
            key=lambda r: (r["current_layer"] if r["current_layer"] is not None else 999, r["name"])
        )

    lines: List[str] = []
    lines.append("# OUTPUT_HI census")
    lines.append("")
    lines.append("Counts:")
    for cat in ("PRODUCER", "SAME_STEP_READER", "CROSS_STEP_READER", "MIXED"):
        lines.append(f"  {cat}: {len(by_cat.get(cat, []))}")
    lines.append(f"  TOTAL ops touching OUTPUT_HI: {len(records)}")
    lines.append("")

    for cat in ("PRODUCER", "SAME_STEP_READER", "CROSS_STEP_READER", "MIXED"):
        lines.append(f"## {cat} ({len(by_cat.get(cat, []))})")
        for rec in by_cat.get(cat, []):
            flags = []
            if rec["writes_OUTPUT_HI"]:
                flags.append("writes")
            if rec["reads_OUTPUT_HI"]:
                flags.append("reads")
            if rec["produces_OUTPUT_HI"]:
                flags.append(f"produces@{rec['produces_register']}")
            if rec["consumes_fresh_OUTPUT_HI"]:
                flags.append(f"consumes_fresh@{rec['consumes_fresh_register']}")
            lines.append(
                f"  - {rec['name']} | L{rec['current_layer']} | kind={rec['kind']} | {' '.join(flags)}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ops = collect_ops()
    records = classify(ops)

    if args.json:
        payload = {"records": list(records.values())}
        out_text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        out_text = render(records)

    if args.out:
        with open(args.out, "w") as f:
            f.write(out_text)
        print(f"wrote {args.out}")
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
