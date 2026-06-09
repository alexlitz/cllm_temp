"""Block-by-block diff of ``symbolic_forward`` actual vs ``dim_oracle`` expected.

Pairs with:

* :class:`~.symbolic_forward.SymbolicForwardRunner` — the actual per-block
  residual produced by walking the IR.
* :class:`~.dim_oracle.ReferenceOracle` — the expected residual computed
  from the reference C4 VM state.

The diff is the load-bearing infrastructure piece behind the
``A3_5_L14_CONSUMER_DIAGNOSTIC`` failure: rather than reading prose
about "the broadcast fires only at the first STACK0 frame", the diff
returns a concrete ``(block, position, dim_key, actual, expected)``
divergence at the *first* block where actual disagrees with what the
reference VM says it should be.

Public surface
--------------

* :class:`Divergence` — one ``(step, position, block, dim_key, actual,
  expected)`` mismatch.
* :class:`BlockDiff` — all divergences at a single block index, plus a
  short suggested-op pointer.
* :func:`diff_actual_vs_expected` — every divergence between a runner
  and an oracle for one dim family.
* :func:`find_first_divergent_block` — the earliest block (by index)
  with at least one divergence, plus the first divergence within it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from .dim_oracle import ReferenceOracle, expected_trace, is_supported_dim
from .symbolic_forward import SymbolicForwardRunner


__all__ = [
    "Divergence",
    "BlockDiff",
    "diff_actual_vs_expected",
    "find_first_divergent_block",
    "format_divergence",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Divergence:
    """One ``(step, position, block, dim_key, actual, expected)`` mismatch."""

    step_idx: int
    position: int
    block_idx: int
    dim_key: str
    actual: float
    expected: float


@dataclass
class BlockDiff:
    """All divergences at one block index. ``suggested_op`` is a best-guess
    pointer to the op-name responsible (typically the closest upstream
    writer for the dim family, or "<no writer>" when nothing writes it).
    """

    block_idx: int
    divergences: List[Divergence] = field(default_factory=list)
    suggested_op: Optional[str] = None
    first_divergence: Optional[Divergence] = None


# ---------------------------------------------------------------------------
# Diff entry points
# ---------------------------------------------------------------------------


def _suggested_op_for_block(
    runner: SymbolicForwardRunner,
    block_idx: int,
    dim_name: str,
) -> Optional[str]:
    """Best-effort: walk the runner's schedule and find the first op at
    ``block_idx`` whose declared ``writes`` mentions ``dim_name``.

    Falls back to the first op at the block when nothing declares the
    write (matches ``dim_flow``'s observation that op-level ``writes``
    sets routinely under-report; the suggestion is a hint, not ground
    truth).
    """

    try:
        layer_ops = runner.ops_per_layer[block_idx]
    except (AttributeError, IndexError):
        return None
    for op in layer_ops:
        writes = getattr(op, "writes", None)
        if writes:
            for w in writes:
                if dim_name in str(w):
                    return getattr(op, "name", "<anonymous>")
    if layer_ops:
        return getattr(layer_ops[0], "name", "<anonymous>")
    return None


def diff_actual_vs_expected(
    runner: SymbolicForwardRunner,
    oracle: ReferenceOracle,
    dim_name: str,
    *,
    atol: float = 1e-6,
) -> List[Divergence]:
    """Every ``Divergence`` between the runner and oracle for one dim family.

    The runner is *not* advanced — callers should call ``runner.step()``
    (or ``runner.run_all()``) before invoking this so the snapshots are
    populated. Both the runner and the oracle must have been built from
    the same ``program_bytecode``.

    Sorted by ``(block_idx, step_idx, position, dim_key)`` so iterating
    gives divergences in block-by-block order — :func:`find_first_divergent_block`
    just takes the first entry.
    """

    if not is_supported_dim(dim_name):
        raise ValueError(
            f"diff_actual_vs_expected: dim {dim_name!r} is not in the "
            f"oracle's SUPPORTED_DIM_FAMILIES; cannot diff."
        )

    expected = expected_trace(
        oracle, dim_name, n_blocks=runner.n_blocks,
    )
    divergences: List[Divergence] = []
    seen_keys: set = set()
    for key in sorted(expected.keys()):
        step_idx, position, block_idx, dim_key = key
        exp_val = expected[key]
        try:
            actual = runner.get_residual(block_idx, position, dim_key)
        except LookupError:
            actual = 0.0
        if abs(actual - exp_val) > atol:
            divergences.append(
                Divergence(
                    step_idx=step_idx,
                    position=position,
                    block_idx=block_idx,
                    dim_key=dim_key,
                    actual=actual,
                    expected=exp_val,
                )
            )
        seen_keys.add((block_idx, position, dim_key))
    return divergences


def find_first_divergent_block(
    runner: SymbolicForwardRunner,
    oracle: ReferenceOracle,
    dim_name: str,
    *,
    atol: float = 1e-6,
) -> Optional[BlockDiff]:
    """Return the earliest block (by ``block_idx``) with at least one
    divergence between runner and oracle for ``dim_name``.

    Returns ``None`` if no divergence is found across any block. The
    returned :class:`BlockDiff` includes the first divergence and a
    suggested-op hint.
    """

    divergences = diff_actual_vs_expected(
        runner, oracle, dim_name, atol=atol,
    )
    if not divergences:
        return None
    first = min(divergences, key=lambda d: (d.block_idx, d.step_idx,
                                            d.position, d.dim_key))
    same_block = [d for d in divergences if d.block_idx == first.block_idx]
    return BlockDiff(
        block_idx=first.block_idx,
        divergences=same_block,
        suggested_op=_suggested_op_for_block(runner, first.block_idx, dim_name),
        first_divergence=first,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_divergence(diff: BlockDiff) -> str:
    """One-line summary suitable for CLI output:
    ``"First divergence at block X — expected V, got W; suggested op: NAME"``.
    """

    if diff.first_divergence is None:
        return f"No divergence at block {diff.block_idx}"
    fd = diff.first_divergence
    return (
        f"First divergence at block {diff.block_idx} "
        f"(step={fd.step_idx} position={fd.position} dim={fd.dim_key}) "
        f"— expected {fd.expected:g}, got {fd.actual:g}; "
        f"suggested op: {diff.suggested_op or '<unknown>'}"
    )
