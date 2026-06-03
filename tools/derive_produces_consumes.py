#!/usr/bin/env python3
"""Derive ``produces`` / ``consumes_fresh`` declarations from an Operation's IR.

This is the migration helper for moving the ~140 op corpus from the current
"~14 ops annotated" state to "every op declares its in-step semantic
read/write surface". The derived sets are returned as plain dicts so a
caller can either splice them into the Operation declaration directly or
diff them against existing manual annotations.

Derivation rules (current — see ``docs/PRODUCES_CONSUMES_MIGRATION.md`` for
the full spec and known gaps):

1. ``produces`` — union of every ``DimRef.name`` appearing in
   ``FFNRule.writes`` across every rule of every layer of
   ``op.compiler_ir``, mapped to a heuristic register slot inferred from
   the op's ``target_op_name`` / ``layer_idx`` (default: ``"unknown"``).

2. ``consumes_fresh`` — every ``DimRef.name`` appearing in
   ``FFNRule.conditions`` or ``FFNRule.gate`` that is ALSO declared in
   ``op.reads`` AND is NOT in the ``_CROSS_STEP_DURABLE`` allowlist of
   register markers / structural / embed-time dims (see
   ``STALENESS_INVARIANTS.md``: cross-step-only reads SHOULD NOT
   declare ``consumes_fresh``). The downstream Tier-A pruning step
   (decl_verifier multistep mode) then drops any false positives that
   slip through.

Both derivations skip ops whose ``compiler_ir`` is empty (no rules at all
— typically structural anchor ops, attention-only ops, or imperative
bakes). Those ops are listed in the migration doc's "manual wave" bucket.

Usage:
    python3 tools/derive_produces_consumes.py \\
        c4_release.neural_vm.unified_compiler.ops.l14_ops:make_layer14_temp_clear_op

or as a library:
    from tools.derive_produces_consumes import derive_for_op
    produces, consumes_fresh = derive_for_op(op)
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Dict, Iterable, Set, Tuple

# Make the package importable when run from a worktree clone.
import os as _os

REPO_ROOT = __file__.rsplit("/tools/", 1)[0]
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_C4_PATH = _os.path.join(REPO_ROOT, "c4_release")
if _C4_PATH not in sys.path:
    sys.path.insert(0, _C4_PATH)


# Dims that carry stable cross-step or embed-time values. Reads of these
# from a same-step perspective are nominally "stale" but architecturally
# correct because the value is not expected to change within a step.
# Listing them here keeps the derived ``consumes_fresh`` honest — matching
# the STALENESS_INVARIANTS.md guidance.
_CROSS_STEP_DURABLE = frozenset({
    "CONST",
    "IS_BYTE",
    "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    "MARK_PC", "MARK_BP", "MARK_AX", "MARK_SP",
    "MARK_STACK0", "MARK_STACK1", "MARK_STACK2",
    "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
    "EMBED_LO", "EMBED_HI",
    "H0", "H1",
    "OP_LEV", "OP_ENT", "OP_RET",  # opcode-broadcast dims
})


def _condition_dim_names(rule) -> Iterable[str]:
    for cond in rule.conditions:
        yield cond.dim.name
    if rule.gate is not None:
        yield rule.gate.name
    for term in rule.gate_terms:
        yield term.dim.name


def _write_dim_names(rule) -> Iterable[str]:
    for w in rule.writes:
        yield w.dim.name


def _infer_register_slot(op) -> str:
    """Heuristic mapping op -> register-slot string for the produces map.

    The slot string is opaque to the analyzer (see
    ``STALENESS_INVARIANTS.md``); a stable, human-readable convention is
    enough. We prefer the op's ``target_op_name`` because it names the
    anchor whose marker the op writes against; fall back to the
    semantic_label, then to ``layer_idx``, then to ``"unknown"``.
    """
    tgt = getattr(op, "target_op_name", None)
    if tgt:
        return tgt
    label = getattr(op, "semantic_label", None)
    if label:
        return label
    idx = getattr(op, "layer_idx", None)
    if idx is not None:
        return f"layer{idx}"
    return "unknown"


def derive_for_op(op) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ``(produces, consumes_fresh)`` derived from ``op.compiler_ir``.

    Both maps are ``dim_name -> register_slot`` per the
    ``STALENESS_INVARIANTS.md`` schema. Empty IR yields empty maps so the
    caller can flag the op for manual migration.
    """
    ir = getattr(op, "compiler_ir", None)
    if ir is None or not ir.layers:
        return {}, {}

    write_names: Set[str] = set()
    cond_names: Set[str] = set()
    for layer in ir.layers:
        for rule in layer.ffn.rules:
            write_names.update(_write_dim_names(rule))
            cond_names.update(_condition_dim_names(rule))

    slot = _infer_register_slot(op)
    produces = {name: slot for name in sorted(write_names)}

    # Conservative consumes_fresh: only condition dims that the op also
    # declared in ``reads`` AND that aren't on the cross-step-durable
    # allowlist. The decl_verifier multistep mode prunes any remaining
    # false positives.
    reads = set(getattr(op, "reads", set()) or set())
    consumes_candidates = (cond_names & reads) - _CROSS_STEP_DURABLE
    consumes_fresh = {name: slot for name in sorted(consumes_candidates)}

    return produces, consumes_fresh


def _resolve_factory(spec: str):
    """Resolve a ``module:factory`` string into a callable."""
    module_name, _, factory_name = spec.partition(":")
    if not factory_name:
        raise SystemExit(
            f"invalid factory spec {spec!r}; expected 'module:factory'"
        )
    module = importlib.import_module(module_name)
    return getattr(module, factory_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "factory",
        help="module:factory spec, e.g. "
             "c4_release.neural_vm.unified_compiler.ops.l14_ops:"
             "make_layer14_temp_clear_op",
    )
    args = parser.parse_args()

    factory = _resolve_factory(args.factory)
    op = factory()
    produces, consumes_fresh = derive_for_op(op)
    print(f"op: {op.name}")
    print(f"derived produces ({len(produces)}):")
    for k, v in produces.items():
        print(f"    {k!r}: {v!r},")
    print(f"derived consumes_fresh ({len(consumes_fresh)}):")
    for k, v in consumes_fresh.items():
        print(f"    {k!r}: {v!r},")
    print(f"existing produces declared: {len(op.produces)}")
    print(f"existing consumes_fresh declared: {len(op.consumes_fresh)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
