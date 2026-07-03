"""Static dim producer/consumer integrity check.

Per the L8 sp_gather STACK0 audit
(``c4_release/docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md``), a class
of "read but never written" bugs slips past every other compile-time
check: an op declares ``reads={"STACK0_BYTE1"}`` but no op anywhere
declares ``writes={"STACK0_BYTE1"}`` for the same residual cell. The
consumer fires, reads zeros, and the downstream value is silently
wrong.

This module walks every :class:`Operation` registered against a
:class:`LayerCompiler` and computes::

    producers = union(op.writes for op in ops)
    consumers = union(op.reads  for op in ops)
    dead_consumers = consumers - producers

A non-empty ``dead_consumers`` set surfaces as a warning at compile
time. The check is intentionally cheap (O(ops * dims)) and does not
attempt the deeper "fires at the right token position" question that
needs a full attention probe.

Cross-step aliases
------------------
Several ops read prior-step values via SSA aliases of the form
``BASE.WRITER.OFFSET`` (offset < 0; e.g. ``ADDR_B0_LO.*.-1``). A
producer for the base dim ``BASE`` satisfies the cross-step alias —
the alias resolves to that base's slot read at the prior autoregressive
step. We treat ``BASE`` producers as covering every
``BASE.*.<negative offset>`` alias.

Opt-out
-------
Set the environment variable ``C4_SKIP_DIM_INTEGRITY=1`` to skip the
scan entirely (e.g. for bisecting an unrelated regression).
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Set

from ..unified_compiler.ssa_dim import is_ssa_form, parse_ssa_name

if TYPE_CHECKING:
    from ..unified_compiler.layer_compiler import LayerCompiler, Operation


_ENV_SKIP = "C4_SKIP_DIM_INTEGRITY"


def _env_skip() -> bool:
    return os.environ.get(_ENV_SKIP, "") == "1"


def _base_dim(name: str) -> str:
    """Strip an SSA prev-step suffix; return ``name`` unchanged otherwise."""
    if not is_ssa_form(name):
        return name
    try:
        parsed = parse_ssa_name(name)
    except ValueError:
        return name
    if parsed.step_offset < 0:
        return parsed.base_dim
    return name


def _producer_set(ops: Iterable["Operation"]) -> Set[str]:
    producers: Set[str] = set()
    for op in ops:
        for dim_name in op.writes or ():
            producers.add(dim_name)
            producers.add(_base_dim(dim_name))
    return producers


def find_dead_consumers(
    ops: Iterable["Operation"],
) -> Dict[str, List[str]]:
    """Return ``{dim_name: [op_name, ...]}`` for dims read but never written.

    The returned dict is deterministic: dim names are sorted, and each
    op-name list is sorted. Cross-step aliases collapse to their base
    dim (see module docstring); the report lists the alias name as it
    appears on the consumer op for actionable diagnostics.
    """
    op_list = list(ops)
    producers = _producer_set(op_list)
    dead_sets: Dict[str, Set[str]] = defaultdict(set)
    for op in op_list:
        for dim_name in op.reads or ():
            if _base_dim(dim_name) in producers:
                continue
            dead_sets[dim_name].add(op.name)
    return {k: sorted(dead_sets[k]) for k in sorted(dead_sets)}


def format_report(dead: Dict[str, List[str]]) -> str:
    """Human-readable, multiline summary for the warning message."""
    if not dead:
        return "DIM INTEGRITY: no dead consumers."
    lines = [
        f"DIM INTEGRITY: {len(dead)} dead consumer dim(s) — read but no op writes them:",
    ]
    for dim_name, consumer_ops in dead.items():
        sample = ", ".join(consumer_ops[:6])
        if len(consumer_ops) > 6:
            sample += f", ... (+{len(consumer_ops) - 6} more)"
        lines.append(f"  - {dim_name!r}: read by {sample}")
    lines.append(f"  (Set {_ENV_SKIP}=1 to skip this check.)")
    return "\n".join(lines)


def run_dim_integrity_check(
    compiler: "LayerCompiler",
) -> Dict[str, List[str]]:
    """Run the dead-consumer scan against every op in ``compiler``.

    Includes ``self.ops`` (attn/ffn), ``self.block_ops``, and
    ``self.model_ops``. Emits a single ``warnings.warn`` with the
    formatted report when ``dead_consumers`` is non-empty. Returns the
    raw mapping so tests can assert against the known set.

    Honours ``C4_SKIP_DIM_INTEGRITY=1`` — returns an empty mapping when
    the env flag opt-out is set.
    """
    if _env_skip():
        return {}
    all_ops = list(compiler.ops) + list(compiler.block_ops) + list(compiler.model_ops)
    dead = find_dead_consumers(all_ops)
    if dead:
        warnings.warn(format_report(dead), stacklevel=3)
    return dead


__all__ = [
    "find_dead_consumers",
    "format_report",
    "run_dim_integrity_check",
]
