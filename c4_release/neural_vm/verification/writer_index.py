"""
S-4: Per-output-dim writer index.

For each (output_dim_name, offset), records every FFNRule that writes
there along with its maximum algebraic contribution and effective firing
scope. Used by S-6 strength verifier to find competing writers.

API:
    build_writer_index(ops, registry) -> dict[(dim_name, offset), list[WriterEntry]]

    @dataclass(frozen=True)
    class WriterEntry:
        op_name: str
        rule: FFNRule
        max_contribution: float       # from S-2 contribution_algebra
        effective_scope: Predicate    # from S-1 effective_predicate
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from neural_vm.dim_registry import DimRegistry
from neural_vm.verification.contribution_algebra import max_contribution
from neural_vm.verification.effective_predicate import effective_predicate
from neural_vm.unified_compiler.ir import FFNOp, FFNRule
from neural_vm.verification.predicates import Predicate


@dataclass(frozen=True)
class WriterEntry:
    """One writer of (dim_name, offset) — the rule, its op, its max
    algebraic contribution, and its effective firing scope."""

    op_name: str
    rule: FFNRule
    max_contribution: float
    effective_scope: Predicate


def build_writer_index(
    ops: Iterable,
    registry: DimRegistry,
) -> dict[tuple[str, int], list[WriterEntry]]:
    """Build a per-(dim_name, offset) writer index across the given ops.

    Each op is expected to have ``.name`` and a walkable ``compiler_ir``
    (FFNOp, CompilerIR with ``.layers[].ffn``, list of FFNRule, etc. —
    see ``_collect_ffn_rules_from_op``).

    Rules whose ``effective_predicate`` cannot be computed (e.g., missing
    dim semantics in the registry, parse failures) are skipped silently;
    callers can address those gaps with the F-4 backfill flow.
    """
    index: dict[tuple[str, int], list[WriterEntry]] = defaultdict(list)

    for op in ops:
        op_name = getattr(op, "name", "<anonymous>")
        rules = _collect_ffn_rules_from_op(op)

        for rule in rules:
            try:
                eff = effective_predicate(rule, registry)
            except KeyError:
                # Missing dim semantics — skip this rule.
                continue
            except Exception:
                # Other failures (parse errors, etc.) — also skip.
                continue

            for wt in rule.writes:
                if wt.weight == 0.0:
                    continue
                key = (wt.dim.name, wt.dim.offset)
                contrib = max_contribution(
                    rule, wt.dim.name, output_offset=wt.dim.offset,
                )
                index[key].append(
                    WriterEntry(
                        op_name=op_name,
                        rule=rule,
                        max_contribution=contrib,
                        effective_scope=eff,
                    )
                )

    return dict(index)


def _collect_ffn_rules_from_op(op) -> list:
    """Walk an Operation to find FFNRules. Handles FFNOp, CompilerIR
    with ``.layers[].ffn``, list/tuple, and direct ``.rules`` attribute
    forms."""
    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        return []

    rules: list = []

    # CompilerIR with .layers[].ffn
    if hasattr(ir, "layers"):
        for layer in ir.layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "rules"):
                rules.extend(ffn.rules)
        return rules

    # FFNOp directly
    if isinstance(ir, FFNOp):
        return list(ir.rules)

    # List/tuple
    if isinstance(ir, (list, tuple)):
        for item in ir:
            if isinstance(item, FFNRule):
                rules.append(item)
            elif isinstance(item, FFNOp):
                rules.extend(item.rules)
        return rules

    # Generic .rules attribute
    if hasattr(ir, "rules"):
        return [r for r in ir.rules if isinstance(r, FFNRule)]

    return []
