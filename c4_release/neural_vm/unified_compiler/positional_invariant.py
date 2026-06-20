"""Positional-invariant mechanism — STEP_TOKENS-relative anchor resolution.

The #1 campaign bug class (see ``docs/POSITIONAL_INVARIANT_AUDIT_2026_06_20.md``
and ``tools/lint_positional_invariants.py``): ~1993 hand-authored anchors encode
positional / distance logic at a *fixed token offset* and silently mis-fire when
``C4_NO_STACK0_EMIT`` collapses the per-step frame ``STEP_TOKENS=35 -> 30`` (the
5-token STACK0 register block is dropped, so the MEM/SE section shifts 5 tokens
earlier). We have been re-anchoring ONE cluster at a time (div/mod, operand-CAM,
ROOT A/B) — whack-a-mole.

This module is the SYSTEMATIC mechanism: a small set of helpers that let an op
declare its anchor's frame ASSUMPTION (``positional_invariant=STEP_TOKENS``) and
have the lowering AUTO-COMPUTE the correct behaviour for the *active*
``Token.STEP_TOKENS``, so the STACK0 drop is a no-op for the rule.

The two structural anchor classes (settled empirically by
``tools/probe_posinv_frame.py``)
=================================================================================

There are exactly two ways an anchor depends on the frame, and they need
opposite treatment:

1. **MARKER-RELATIVE distance-bank anchors** — ``H<k>+MEM_I``, ``L2H0+MEM_I``,
   and the L2-produced ``MEM_VAL_B*`` / ``BYTE_INDEX_*`` flags that derive from
   them. The ``+i`` offset is a *marker-type slot index* into the fixed-width
   7-slot threshold-head bank (``PC=0 AX=1 SP=2 BP=3 MEM=4 SE=5``), NOT a token
   distance. The L0/L1/L2 threshold attention computes "is a marker of type
   ``i`` within distance ``t``" via ALiBi *distance* attention — distance from
   the nearest marker, recomputed at runtime. When the frame shrinks and the
   MEM marker moves ``25 -> 20`` the distance attention tracks it, so the flag
   still fires on the byte immediately after the MEM marker in BOTH frames.
   **These are ALREADY frame-invariant.** The mechanism's job for this class is
   only to *resolve the marker-slot index from STEP_TOKENS* (so an op stops
   hand-coding ``MEM_I = 4``) and to assert the invariance, not to shift
   anything.

2. **ABSOLUTE-SLOT byte flags** — ``STACK0_BYTE0..3`` ("STACK0 byte N position")
   and any anchor expressed as ``d=k-from-<marker>`` where the ``k`` rows
   between the marker and the target *include the dropped STACK0 block*. In the
   30-token frame the STACK0 token slots do not exist, and a ``d=6-from-BP``
   flag (built to fire on STACK0 byte 0) now lands on MEM addr byte 0 — a
   MISFIRE onto an unrelated row. **This is the genuinely-broken class.** The
   mechanism's job here is to AUTO-NEUTRALIZE the flag when STEP_TOKENS drops
   (or re-point it to its marker-relative equivalent), replacing the hand-coded
   ``1e9 if no_stack0_emit_enabled() else 1.5`` whack-a-mole.

Public API
==========

``marker_bank_index(name, *, step_tokens=None)``
    Resolve a marker name (``"PC"|"AX"|"SP"|"BP"|"MEM"|"SE"``) to its slot index
    in the threshold-head bank for the active frame. Proves/encodes that the
    bank order does NOT change between frames (STACK0 is a transition target,
    never a bank slot), so ``marker_bank_index("MEM") == 4`` in both — replacing
    the literal ``MEM_I = 4``.

``frame_byte_is_emitted(marker, k, *, step_tokens=None)``
    For a ``d=k-from-<marker>`` positional anchor: returns ``True`` iff the byte
    that anchor targets is actually EMITTED in the active frame. The single
    decision the auto-neutralize gate consumes.

``invariant_threshold(live, suppressed, marker, k, *, step_tokens=None)``
    The auto-shift primitive for an absolute-slot threshold rule: returns
    ``live`` when the targeted byte is emitted in the active frame and
    ``suppressed`` (a make-unreachable threshold) when it is not — computed from
    ``Token.STEP_TOKENS``, NO per-op env branch. This is the systematic
    replacement for ``1e9 if no_stack0_emit_enabled() else 1.5``.

The mechanism is a pure no-op at ``STEP_TOKENS == 35`` (the golden frame):
``invariant_threshold`` returns exactly ``live`` and ``marker_bank_index``
returns the same integers the literals encoded, so the lowered weights are
BYTE-IDENTICAL to golden ``4958b35b``.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Frame structure (mirrors token_layout.py, derived not hard-coded)
# ---------------------------------------------------------------------------

# The threshold-head marker bank is a fixed-width 7-slot vector. Its ORDER is
# the marker-TYPE order, which is identical in both frames — STACK0 is a
# transition target in the L0 phase-A chain, never a bank slot (the bank is
# produced by L0/L1/L2 threshold attention keyed on marker TYPE, and there are
# 6 emitted marker types PC/AX/SP/BP/MEM/SE plus the STACK0 marker that shares
# no bank slot). This is why ``MEM_I`` is frame-invariant.
_MARKER_BANK_ORDER = ("PC", "AX", "SP", "BP", "MEM", "SE")
_MARKER_BANK_SLOT = {name: i for i, name in enumerate(_MARKER_BANK_ORDER)}

# Per-step token counts for the two live frames. Mirrors Token.STEP_TOKENS;
# kept here so the mechanism is testable without a model build.
STEP_TOKENS_FULL = 35   # STACK0 emitted (golden)
STEP_TOKENS_DROPPED = 30  # C4_NO_STACK0_EMIT (campaign)


def _active_step_tokens(step_tokens: Optional[int]) -> int:
    """Resolve the active per-step token count.

    When ``step_tokens`` is given, use it (testability). Otherwise read the
    single authority ``Token.STEP_TOKENS`` (it resolves the env flag once at
    import). Imported lazily so this module has no import-time model dependency.
    """
    if step_tokens is not None:
        return int(step_tokens)
    from ..vm_step import Token
    return int(Token.STEP_TOKENS)


# ---------------------------------------------------------------------------
# Class-1 helper: marker-relative bank-slot index (frame-invariant by design)
# ---------------------------------------------------------------------------


def marker_bank_index(name: str, *, step_tokens: Optional[int] = None) -> int:
    """Return the threshold-head-bank slot index for marker ``name``.

    Frame-invariant: the same integer in both the 35- and 30-token frames
    (the bank is keyed on marker TYPE, and the type order does not change when
    the STACK0 *value* block is dropped). Replaces the hand-coded
    ``MEM_I = 4`` / ``PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0,1,2,3,4,5``
    scattered across l0/l1/l2/l8/compiler with a single source of truth, and
    lets ``lint_positional_invariants`` recognise the ref as *declared*
    marker-relative (invariant) rather than an UNGUARDED bare offset.

    ``step_tokens`` is accepted (and validated) so callers may assert the
    invariance, but the returned index does not depend on it.
    """
    st = _active_step_tokens(step_tokens)
    if st not in (STEP_TOKENS_FULL, STEP_TOKENS_DROPPED):
        raise ValueError(
            f"unsupported STEP_TOKENS={st}; expected "
            f"{STEP_TOKENS_FULL} or {STEP_TOKENS_DROPPED}"
        )
    try:
        return _MARKER_BANK_SLOT[name]
    except KeyError:
        raise ValueError(
            f"unknown marker {name!r}; expected one of {_MARKER_BANK_ORDER}"
        )


# ---------------------------------------------------------------------------
# Class-2 helper: absolute-slot anchor auto-neutralize
# ---------------------------------------------------------------------------

# Structural offset (in tokens) of each EMITTED byte from its register marker.
# Byte k of a register sits at marker+1+k. The STACK0 block (marker + 4 value
# bytes) occupies the 5 slots immediately after the BP register block in the
# FULL frame and is absent in the DROPPED frame.
_STACK0_BLOCK_WIDTH = 5  # marker + 4 value bytes


def frame_byte_is_emitted(
    marker: str, k: int, *, step_tokens: Optional[int] = None
) -> bool:
    """Is the byte a ``d=k-from-<marker>`` anchor targets actually EMITTED?

    The only anchors whose target VANISHES under the campaign are those that
    point INTO the STACK0 register block: a ``d=k-from-BP`` flag whose ``k``
    lands within the 5-token STACK0 block (BP byte3 is d=4 from BP; STACK0
    marker d=5, STACK0 bytes 0..3 are d=6..9 from BP). In the dropped frame
    those slots do not exist, so the flag, if left live, would alias onto the
    next emitted section (MEM). For markers whose target is not the dropped
    block, the byte is always emitted.

    Returns ``True`` if the targeted byte is part of the active frame's
    emission, ``False`` if it lives in the dropped STACK0 block.
    """
    st = _active_step_tokens(step_tokens)
    if st == STEP_TOKENS_FULL:
        return True
    # Dropped (30-token) frame: only BP-relative anchors that reach into the
    # STACK0 block (d in 5..9 from BP) target a vanished slot.
    if marker == "BP" and 5 <= k <= 5 + (_STACK0_BLOCK_WIDTH - 1):
        return False
    if marker == "STACK0":
        return False
    return True


def invariant_threshold(
    live: float,
    suppressed: float,
    marker: str,
    k: int,
    *,
    step_tokens: Optional[int] = None,
) -> float:
    """Auto-shift primitive for an absolute-slot threshold rule.

    Returns ``live`` when the ``d=k-from-<marker>`` byte is emitted in the
    active frame, else ``suppressed`` (a make-unreachable threshold that parks
    the unit so the byte-count allocator guard is satisfied but the rule never
    fires). The decision is computed from ``Token.STEP_TOKENS`` — there is NO
    per-op ``no_stack0_emit_enabled()`` branch, which is the whole point: one
    mechanism, every absolute-slot anchor.

    At ``STEP_TOKENS == 35`` this returns exactly ``live`` for every anchor, so
    the lowered weights are byte-identical to golden.
    """
    return live if frame_byte_is_emitted(marker, k, step_tokens=step_tokens) \
        else suppressed
