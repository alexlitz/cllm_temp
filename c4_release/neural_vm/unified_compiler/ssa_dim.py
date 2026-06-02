"""SSA-style dim versioning (Phase 9 prototype).

LLVM-inspired naming: instead of the ad-hoc ``OUTPUT_LO_PREV_STEP`` alias
trick (which is a label rename on a shared numeric slot to fool the
scheduler's cycle detector), each cross-step read is tagged with both the
*producing op* and the *step offset* it reads from. The full grammar is::

    SSA_NAME := BASE_DIM "." WRITER_OP "." STEP_OFFSET
    BASE_DIM := <identifier — the residual band, e.g. "OUTPUT_LO">
    WRITER_OP := <op name | "*"> — author of the value, or "*" for "any writer"
    STEP_OFFSET := <int> — VM steps before the current one ("-1" for prev step)

Examples
--------
``OUTPUT_LO``
    Bare base dim. Current-step value. Default semantics; no SSA suffix.
``OUTPUT_LO.layer16_lev_routing.-1``
    OUTPUT_LO produced by ``layer16_lev_routing`` one VM step ago. Single
    writer; the scheduler treats this as a back-edge against
    ``layer16_lev_routing -> reader`` and prunes it.
``OUTPUT_LO.*.-1``
    OUTPUT_LO produced by any writer one VM step ago. Used when the read
    aggregates over multiple potential producers (the most common case for
    attention heads that look back at the prev-step AX marker without
    knowing which op last wrote it).

Why SSA, not just ``_PREV_STEP``
--------------------------------
The PREV_STEP alias-rename trick collapses *all* cross-step reads of a
band into one synonym. The scheduler can't tell ``OUTPUT_LO_PREV_STEP``
from a same-step read of a different writer, and a reader that genuinely
needs two distinct cross-step producers (e.g. one for LEV-following, one
for normal flow) has nowhere to encode that. SSA names each version
individually:

    1. The dep graph drops back-edges by *value identity*, not by name
       (``OUTPUT_LO.layer16_lev_routing.-1`` and
       ``OUTPUT_LO.layer9_alu.-1`` are different reads, even though they
       resolve to the same physical slot).
    2. Downstream analyzers (KV liveness, staleness) can ask "which writer
       does this read consume?" without a separate metadata table.
    3. Byte-identity at default is preserved: the dim allocator aliases
       every SSA form back onto the base dim's numeric slot, exactly like
       PREV_STEP does today.

Byte-identity strategy
----------------------
At allocation time, ``parse_ssa_name`` strips the suffix and the
``LayerCompiler`` registers an alias from the SSA form onto the base
dim's numeric slot. The bake reads/writes the same column as the
unversioned name, so the lowered weights are bit-identical to the
pre-SSA build provided the demo op's reads/writes resolve to the same
positions. See :func:`compare_symbolic_to_lowered_attn` for the gate.

Migration plan (deferred to Phase 9.B)
--------------------------------------
This module ships the parser + scheduler hook + a single demo op
(``layer8_head6_ax_carry_refresh``). The corpus-wide rename from
``X_PREV_STEP`` to ``X.<writer>.-1`` is staged behind a follow-up wave
(see ``docs/PHASE_9_SSA_PROTOTYPE.md``). Until then, both spellings are
accepted by ``LayerCompiler.add_op``; the parser canonicalizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


SSA_SEPARATOR = "."
SSA_ANY_WRITER = "*"


@dataclass(frozen=True)
class SsaDimName:
    """Parsed SSA dim name.

    Attributes
    ----------
    base_dim:
        The unversioned residual band name (e.g. ``"OUTPUT_LO"``).
    writer_op:
        Producer op-name, or ``"*"`` for "any writer", or ``None`` if the
        name is unversioned (current-step read).
    step_offset:
        VM steps before the current step. ``0`` for unversioned /
        current-step. ``-1`` for the previous step (the common case).
    """

    base_dim: str
    writer_op: Optional[str]
    step_offset: int

    @property
    def is_cross_step(self) -> bool:
        """True iff this name reads from a non-current VM step."""
        return self.step_offset != 0

    @property
    def is_any_writer(self) -> bool:
        """True iff the name uses the ``*`` writer-wildcard."""
        return self.writer_op == SSA_ANY_WRITER

    @property
    def canonical(self) -> str:
        """Round-trip the parsed form back into a string."""
        if self.writer_op is None and self.step_offset == 0:
            return self.base_dim
        writer = self.writer_op if self.writer_op is not None else SSA_ANY_WRITER
        return f"{self.base_dim}{SSA_SEPARATOR}{writer}{SSA_SEPARATOR}{self.step_offset}"


def is_ssa_form(name: str) -> bool:
    """True iff ``name`` carries an SSA suffix (i.e. contains a ``.``).

    Used by the LayerCompiler to short-circuit the parser for the common
    case of unversioned dim names. The check is intentionally cheap so
    every read/write touches at most one ``in`` lookup.
    """
    return SSA_SEPARATOR in name


def parse_ssa_name(name: str) -> SsaDimName:
    """Parse an SSA dim name into ``(base_dim, writer_op, step_offset)``.

    Accepts both the unversioned form (``"OUTPUT_LO"``) and the dotted
    form (``"OUTPUT_LO.layer9_alu.-1"``). Raises ``ValueError`` for
    malformed names (wrong segment count, non-integer step offset).
    """
    if not is_ssa_form(name):
        return SsaDimName(base_dim=name, writer_op=None, step_offset=0)
    parts = name.split(SSA_SEPARATOR)
    if len(parts) != 3:
        raise ValueError(
            f"SSA dim name {name!r} must have exactly 3 dot-separated "
            f"segments (base.writer.step_offset); got {len(parts)}"
        )
    base, writer, offset_str = parts
    if not base:
        raise ValueError(f"SSA dim name {name!r} has empty base dim")
    if not writer:
        raise ValueError(f"SSA dim name {name!r} has empty writer op")
    try:
        offset = int(offset_str)
    except ValueError as exc:
        raise ValueError(
            f"SSA dim name {name!r} step offset {offset_str!r} is not an int"
        ) from exc
    return SsaDimName(base_dim=base, writer_op=writer, step_offset=offset)


def base_of(name: str) -> str:
    """Return the unversioned base dim of an SSA name.

    Cheap wrapper used by code that only cares about the residual band
    (e.g. dim-position lookup at bake time). Equivalent to
    ``parse_ssa_name(name).base_dim`` but skips the dataclass build.
    """
    if not is_ssa_form(name):
        return name
    idx = name.index(SSA_SEPARATOR)
    return name[:idx]


def make_ssa_name(base_dim: str, writer_op: str, step_offset: int) -> str:
    """Construct an SSA dim name from parts.

    Inverse of :func:`parse_ssa_name`. Always emits the dotted form,
    even for ``step_offset == 0`` — callers that want the canonical
    unversioned form should pass through :class:`SsaDimName`.
    """
    if SSA_SEPARATOR in base_dim:
        raise ValueError(
            f"base_dim {base_dim!r} must not contain {SSA_SEPARATOR!r}"
        )
    if SSA_SEPARATOR in writer_op:
        raise ValueError(
            f"writer_op {writer_op!r} must not contain {SSA_SEPARATOR!r}"
        )
    return f"{base_dim}{SSA_SEPARATOR}{writer_op}{SSA_SEPARATOR}{step_offset}"
