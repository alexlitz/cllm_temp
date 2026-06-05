"""Compute the per-opcode dim closure across all transformer blocks.

For each opcode X and each block B, walk every op declared in the layout
and collect the union of ``reads`` / ``writes`` across the ops that fire
for opcode X at block B. An op is considered to fire for opcode X iff:

  - ``op.opcodes`` is empty (opcode-agnostic — fires every step), OR
  - ``X in op.opcodes`` (opcode-specific).

The block index is resolved from:
  - ``layer_idx`` for block_ops and (the rare) model ops
  - the index of ``ops_per_layer`` for per-layer attn/ffn ops

Output:
  - ``tools/per_opcode_closure.json``: a JSON dump suitable for downstream
    consumers (e.g. layer-skip implementations). Keys are
    ``"<opcode>|<block_idx>|<facet>"`` where facet is
    ``reads`` / ``writes`` / ``active_op_count`` / ``op_names``.

Run:
  python tools/per_opcode_closure.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
    sys.path.insert(0, str(ROOT))

from neural_vm.unified_compiler.decl_verifier import (  # noqa: E402
    _build_layout_only,
    _KNOWN_C4_OPCODES,
)


# User-requested 30 opcodes (see Opcode in neural_vm/embedding.py). Excludes
# OP_NOP / OP_GETCHAR / OP_PUTCHAR / system-call opcodes; matches the
# request's IMM..EXIT list exactly.
REQUESTED_OPCODES = [
    "OP_IMM", "OP_LEA", "OP_JMP", "OP_JSR", "OP_BZ", "OP_BNZ",
    "OP_ENT", "OP_ADJ", "OP_LEV",
    "OP_LI", "OP_LC", "OP_SI", "OP_SC", "OP_PSH",
    "OP_OR", "OP_XOR", "OP_AND",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_SHL", "OP_SHR",
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_EXIT",
]


def op_fires_for(op, opcode):
    """An op fires for ``opcode`` if its declared ``opcodes`` set is empty
    (opcode-agnostic) or contains ``opcode``.
    """
    if not op.opcodes:
        return True
    return opcode in op.opcodes


def collect_all_ops(layout):
    """Yield ``(block_idx, op)`` for every per-block op in the layout.

    Per-layer attn/ffn ops are tagged with their placement index.
    Block ops use their ``layer_idx``. Model ops with a non-None
    ``layer_idx`` count toward that block; model ops with
    ``layer_idx=None`` are excluded — they describe whole-model bake
    events (e.g. embedding, head, alibi-slope writes) that don't
    correspond to a per-step per-block dispatch and would dilute the
    closure if broadcast.
    """
    n_layers = layout.n_layers
    for layer_idx, ops_at in enumerate(layout.ops_per_layer):
        for op in ops_at:
            yield layer_idx, op
    for op in layout.block_ops:
        if op.layer_idx is None:
            continue
        if 0 <= op.layer_idx < n_layers:
            yield op.layer_idx, op
    for op in layout.model_ops:
        # Only model ops pinned to a specific block contribute; broadcast
        # ops are bake-time only and irrelevant to per-step skip analysis.
        if op.layer_idx is not None and 0 <= op.layer_idx < n_layers:
            yield op.layer_idx, op


def compute_closure(layout, opcodes):
    """Return ``cells[(opcode, block_idx)] = {reads, writes, op_names}``."""
    cells = defaultdict(lambda: {
        "reads": set(),
        "writes": set(),
        "op_names": [],
    })
    for block_idx, op in collect_all_ops(layout):
        for oc in opcodes:
            if not op_fires_for(op, oc):
                continue
            cell = cells[(oc, block_idx)]
            cell["reads"].update(op.reads)
            cell["writes"].update(op.writes)
            cell["op_names"].append(op.name)
    return cells


def main():
    layout = _build_layout_only(
        alu_mode="lookup",
        enable_conversational_io=False,
        enable_tool_calling=False,
        n_heads=8,
    )
    n_blocks = layout.n_layers
    print(f"[per-opcode] layout.n_layers={n_blocks}", file=sys.stderr)
    print(
        f"[per-opcode] opcodes={len(REQUESTED_OPCODES)} "
        f"(skipped IO/syscalls): {len(_KNOWN_C4_OPCODES) - len(REQUESTED_OPCODES)} excluded",
        file=sys.stderr,
    )

    cells = compute_closure(layout, REQUESTED_OPCODES)

    # Build JSON payload.
    payload = {
        "meta": {
            "alu_mode": "lookup",
            "n_blocks": n_blocks,
            "opcodes": REQUESTED_OPCODES,
            "fire_rule": (
                "op fires for opcode X iff op.opcodes is empty "
                "(opcode-agnostic) OR X in op.opcodes"
            ),
            "block_index_source": (
                "ops_per_layer index for attn/ffn ops; layer_idx for "
                "block_ops; model_ops with layer_idx=None are broadcast "
                "to every block"
            ),
        },
        "cells": {},
    }
    skippable = []
    for oc in REQUESTED_OPCODES:
        for b in range(n_blocks):
            c = cells.get((oc, b), {"reads": set(), "writes": set(), "op_names": []})
            key = f"{oc}|{b}"
            payload["cells"][key] = {
                "reads": sorted(c["reads"]),
                "writes": sorted(c["writes"]),
                "active_op_count": len(c["op_names"]),
                "op_names": sorted(c["op_names"]),
            }
            if not c["reads"] and not c["writes"]:
                skippable.append((oc, b))

    out_path = ROOT / "tools" / "per_opcode_closure.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    print(f"[per-opcode] wrote {out_path}", file=sys.stderr)
    print(
        f"[per-opcode] skippable cells (zero reads + zero writes): "
        f"{len(skippable)} / {n_blocks * len(REQUESTED_OPCODES)}",
        file=sys.stderr,
    )

    # Print a compact summary for the doc generator.
    return payload, skippable


if __name__ == "__main__":
    main()
