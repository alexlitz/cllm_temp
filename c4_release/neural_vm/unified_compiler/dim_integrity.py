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
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

from .ssa_dim import SSA_SEPARATOR, is_ssa_form, parse_ssa_name

if TYPE_CHECKING:
    from .layer_compiler import LayerCompiler, Operation


_ENV_SKIP = "C4_SKIP_DIM_INTEGRITY"


def _env_skip() -> bool:
    return os.environ.get(_ENV_SKIP, "") == "1"


def _normalize_consumer_name(name: str) -> str:
    """Return the lookup key used to match a consumer against producers.

    For SSA prev-step aliases (``BASE.WRITER.OFFSET`` with ``OFFSET<0``)
    we collapse to the base dim so any in-step producer of ``BASE``
    satisfies the cross-step read. Unversioned names pass through
    unchanged.
    """
    if not is_ssa_form(name):
        return name
    try:
        parsed = parse_ssa_name(name)
    except ValueError:
        # Malformed SSA name — leave as-is so the dead-consumer report
        # surfaces it loudly.
        return name
    if parsed.step_offset < 0:
        return parsed.base_dim
    # Same-step or future-step SSA name: keep the full form so the
    # producer match goes through the SSA-aware writer-index code paths
    # (which is what the compiler's other passes already do).
    return name


def _producer_set(ops: Iterable["Operation"]) -> Set[str]:
    producers: Set[str] = set()
    for op in ops:
        for dim_name in op.writes or ():
            producers.add(dim_name)
            # Allow base-dim producers to match cross-step aliases of
            # the same base by also adding the base form (no-op when
            # the write is already unversioned).
            if is_ssa_form(dim_name):
                try:
                    parsed = parse_ssa_name(dim_name)
                except ValueError:
                    continue
                producers.add(parsed.base_dim)
    return producers


def find_dead_consumers(
    ops: Iterable["Operation"],
) -> Dict[str, List[str]]:
    """Return ``{dim_name: [op_name, ...]}`` for dims read but never written.

    The returned dict is deterministic: dim names are sorted, and each
    op-name list is sorted by op name. Cross-step aliases collapse to
    their base dim (see module docstring); the report lists the alias
    name as it appears on the consumer op for actionable diagnostics.
    """
    op_list = list(ops)
    producers = _producer_set(op_list)
    dead: Dict[str, List[str]] = {}
    for op in op_list:
        for dim_name in op.reads or ():
            key = _normalize_consumer_name(dim_name)
            if key in producers:
                continue
            # Final check: also accept ``dim_name`` itself (covers the
            # case where the consumer reads ``BASE.*.-1`` AND some op
            # writes the exact same dotted form, though we don't expect
            # this in practice).
            if dim_name in producers:
                continue
            dead.setdefault(dim_name, []).append(op.name)
    for dim_name in list(dead.keys()):
        dead[dim_name] = sorted(set(dead[dim_name]))
    return {k: dead[k] for k in sorted(dead.keys())}


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
    lines.append(
        f"  (Set {_ENV_SKIP}=1 to skip this check.)"
    )
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
    all_ops: List["Operation"] = (
        list(compiler.ops)
        + list(getattr(compiler, "block_ops", []) or [])
        + list(getattr(compiler, "model_ops", []) or [])
    )
    dead = find_dead_consumers(all_ops)
    if dead:
        warnings.warn(format_report(dead), stacklevel=3)
    return dead


__all__ = [
    "find_dead_consumers",
    "format_report",
    "run_dim_integrity_check",
]
