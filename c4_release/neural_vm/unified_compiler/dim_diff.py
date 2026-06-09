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
    "first_writer_block_for_dim",
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


def _op_writes_dim(op, dim_name: str) -> bool:
    """Return True iff ``op``'s declared ``writes`` mentions ``dim_name``.

    Op-level ``writes`` sets routinely under-report (see ``dim_flow``);
    this matcher accepts a substring match against the stringified form
    of each write entry so callers that store either a bare name
    (``"OUTPUT_LO"``) or a per-offset entry (``"OUTPUT_LO+5"``) are both
    detected.
    """

    writes = getattr(op, "writes", None)
    if not writes:
        return False
    for w in writes:
        if dim_name in str(w):
            return True
    return False


def first_writer_block_for_dim(
    runner: SymbolicForwardRunner,
    dim_name: str,
) -> Optional[int]:
    """Earliest block index in ``runner.ops_per_layer`` where some op
    declares a write to ``dim_name``.

    Returns ``None`` when no op in any block declares the write; callers
    that hit ``None`` should treat the diff as block-invariant (fall
    back to comparing all blocks).

    This is the load-bearing piece behind the block-aware diff: the
    oracle's expected projection is a *same-step materialisation* and
    does not predict intermediate block residuals. By restricting the
    comparison to blocks ``>= first_writer_block(dim)`` we skip the
    pre-writer blocks where the dim has not yet been produced (which
    would otherwise always be flagged as "the first divergent block").
    """

    try:
        ops_per_layer = runner.ops_per_layer
    except AttributeError:
        return None
    for block_idx, layer_ops in enumerate(ops_per_layer):
        for op in layer_ops:
            if _op_writes_dim(op, dim_name):
                return block_idx
    return None


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
        if _op_writes_dim(op, dim_name):
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
    block_aware: bool = True,
) -> List[Divergence]:
    """Every ``Divergence`` between the runner and oracle for one dim family.

    The runner is *not* advanced — callers should call ``runner.step()``
    (or ``runner.run_all()``) before invoking this so the snapshots are
    populated. Both the runner and the oracle must have been built from
    the same ``program_bytecode``.

    Sorted by ``(block_idx, step_idx, position, dim_key)`` so iterating
    gives divergences in block-by-block order — :func:`find_first_divergent_block`
    just takes the first entry.

    Parameters
    ----------
    block_aware
        When ``True`` (default), the diff skips blocks before the first
        op in the schedule that declares a write to ``dim_name``. This
        avoids spurious "first divergence at block 0" reports for dims
        whose first writer lives at some later block — block 0's
        pre-compute snapshot has not yet seen the writer, so the value
        is necessarily 0 there.

        Set to ``False`` to fall back to the previous block-invariant
        behaviour (compare every block in ``[0, n_blocks)``). Useful
        for callers who want to inspect the full grid (e.g. to confirm
        a dim is not leaking into pre-writer blocks).
    """

    if not is_supported_dim(dim_name):
        raise ValueError(
            f"diff_actual_vs_expected: dim {dim_name!r} is not in the "
            f"oracle's SUPPORTED_DIM_FAMILIES; cannot diff."
        )

    if block_aware:
        first_writer = first_writer_block_for_dim(runner, dim_name)
        if first_writer is None:
            # No op in any block declares a write to this dim — the
            # block-aware diff has nothing to compare. Returning an
            # empty list (rather than falling back to the all-blocks
            # comparison) keeps the localisation honest: an unwritten
            # dim is not a per-block "divergence", it's a missing-writer
            # observation the caller should surface with ``block_aware=False``.
            return []
    else:
        first_writer = None
    expected = expected_trace(
        oracle, dim_name, n_blocks=runner.n_blocks,
        first_writer_block=first_writer,
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
    block_aware: bool = True,
) -> Optional[BlockDiff]:
    """Return the earliest block (by ``block_idx``) with at least one
    divergence between runner and oracle for ``dim_name``.

    Returns ``None`` if no divergence is found across any block. The
    returned :class:`BlockDiff` includes the first divergence and a
    suggested-op hint.

    By default the search is *block-aware*: it skips blocks before the
    first op in the schedule that declares a write to ``dim_name``.
    Without this, every dim whose first writer lives past block 0 would
    appear to "first diverge at block 0" — the localisation signal that
    the oracle is supposed to provide degenerates. Pass
    ``block_aware=False`` to fall back to the previous block-invariant
    diff.
    """

    divergences = diff_actual_vs_expected(
        runner, oracle, dim_name, atol=atol, block_aware=block_aware,
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

    When ``diff.first_divergence.actual == 0``, this is the "dim written
    but stayed zero" case (the writer's block fired but the value never
    materialised). When it's non-zero, the writer produced a value that
    disagrees with the oracle. Both are real divergences; only the cause
    differs.
    """

    if diff.first_divergence is None:
        return f"No divergence at block {diff.block_idx}"
    fd = diff.first_divergence
    cause = "value=0 (writer did not fire)" if fd.actual == 0.0 else "value mismatch"
    return (
        f"First divergence at block {diff.block_idx} "
        f"(step={fd.step_idx} position={fd.position} dim={fd.dim_key}) "
        f"— expected {fd.expected:g}, got {fd.actual:g} [{cause}]; "
        f"suggested op: {diff.suggested_op or '<unknown>'}"
    )
