"""C4_DOOM_FAST — the single-toggle composite for the byte-exact fast-doom stack.

THE LEVER (this task).  The winning ~9.85 fps 1-GPU doom step is the composition of
~19 individually-gated, individually-DEFAULT-OFF, individually-BYTE-EXACT perf flags
(dead-block fusion + direct-CAM gathers + the on-chip fused dead-FFN megakernel +
wave-batching + the attention-megablock + the block-0 [D,K] fold + the precomputed
cache-resolved GPU-built double-buffered schedule).  Before this module the stack was
assembled BY HAND — the audit found **22 distinct COMPOSED=[...] env-lists** across the
``_agent_*`` scripts, each re-listing 15-40 flags, plus the ``REALTIME_SETUP.md`` recipe.

``C4_DOOM_FAST=1`` turns the WHOLE blessed set on with ONE toggle.  Every member is
already a DEFAULT-OFF, golden-``069cc32f``-byte-identical flag, so bundling them changes
NO weights and NO build-path — the composite is a pure convenience env-expansion.

CONTRACT / SAFETY
-----------------
* ``C4_DOOM_FAST`` UNSET  -> ``expand_doom_fast()`` is a NO-OP; every member stays at its
  own default (OFF) -> golden ``069cc32f`` byte-identical (verified by
  ``c4_min._fingerprint_build``).
* ``C4_DOOM_FAST=1``      -> each member is set with ``os.environ.setdefault`` semantics so a
  member the user set BY HAND (to any value, including ``0`` to opt OUT) is NEVER overridden
  — the composite only fills in the UNSET members.  So ``C4_DOOM_FAST=1 C4_BLOCK0_DK=0``
  runs the whole stack MINUS block-0-DK (per-flag escape hatch, per the audit recommendation).
* This expands ENV ONLY.  Every member flag is still read (lazily, at forward/build time)
  by its own ``*_enabled()`` helper via ``os.environ.get`` — this module does not import or
  rebind anything, so importing it is free and side-effect-free until ``expand_doom_fast()``
  is called.  Call it ONCE, EARLY (before the first member-flag read / model build), e.g.
  at the top of a harness ``main`` or right after the ``os.environ.setdefault`` preamble.

  NOTE: the composed ATTENTION levers (dead-block fusion / local-attn / flash / banded)
  also require the harness to CALL ``tight_attn_compose.install_composed(model)`` (or the
  equivalent per-block installers) on the built model — the env flags are read at forward
  time by the rebound ``.forward``s, so setting the env alone is necessary but not
  sufficient for the attention path.  ``C4_DOOM_FAST`` sets the env; the harness still
  drives ``install_composed`` exactly as it does today.

MEMBER SET (the transitive closure of the ~9.85 fps block-0-DK stack)
---------------------------------------------------------------------
Task-named members and the DEPENDENCIES they pull in (the audit's undocumented deps):

  block-skip / dead-FFN     C4_DEAD_BLOCK_FUSION      (fuse 238 dead-attn blocks -> output=x;
                                                       this is the live "divfree block-skip".
                                                       C4_STEP_BLOCK_SKIP / C4_BATCHED_BLOCK_SKIP
                                                       are the OLDER per-op live-mask skips —
                                                       NOT part of the megablock stack, so NOT set)
  fused dead-FFN megakernel C4_FUSED_MEGABLOCK, C4_FFN_FUSED_HIDDEN, C4_FFN_WAVE_BATCH
                            C4_MEGABLOCK_BLOCK_K=512   (** wave-batch is INERT without a bigger
                                                       block_k — the audit's WAVE_BATCH->BLOCK_K
                                                       dep; default 64, the stack uses 512 **)
  direct-CAM gathers        C4_DIRECT_CAM_BATCHED, C4_DIRECT_LOCAL_CAM, C4_DIRECT_CAM_VEC
  local/flash attention     C4_FLASH_ATTN, C4_BANDED_LOCAL_ATTN
  attention megablock       C4_ATTN_MEGABLOCK        (** requires PRECOMPUTED_SCHEDULE +
                                                       ONCHIP_RESIDUAL + FUSED_MEGABLOCK **)
  block-0 [D,K] fold        C4_BLOCK0_DK             (** requires ATTN_MEGABLOCK **)
  precomputed schedule      C4_PRECOMPUTED_SCHEDULE, C4_ONCHIP_RESIDUAL, C4_RESIDENT_BATCH
  GPU-built schedule        C4_SCHED_FAST_BUILD, C4_SCHED_GPU_BUILD   (** GPU_BUILD requires
                                                       FAST_BUILD + ONCHIP_RESIDUAL **)
  double-buffer pipeline    C4_SCHED_PIPELINE        (** requires SCHED_GPU_BUILD — the audit's
                                                       PIPELINE->GPU_BUILD dep **)
  cache-resolved build      C4_SCHED_CACHE_RESOLVED

NOT set by the composite (and why):
  * ``C4_PF_CFM=1`` (LEAN streaming build) is set by the harness preamble, not a perf lever;
    the composite does not force a build mode.  Harnesses keep their ``os.environ.setdefault
    ("C4_PF_CFM","1")``.
  * ``C4_SCHED_CHUNK`` is a frame-shape KNOB (the chunk size), not an on/off lever; the
    harness picks it per frame (e.g. ``chunk=K`` for a whole-batch replay).
  * ``C4_FFN_LINFOLD`` (VERTICAL linear-fold) is a HONEST-TRADE FLOP-for-occupancy lever
    that is NOT a net win at doom tile sizes (fill-in) — deliberately excluded.
  * ``C4_MEGABLOCK_RESID`` stays ``fp32`` (bf16 is decode-safe but ~0 gain here and moves
    away from the byte-exact headline; excluded).
"""
from __future__ import annotations

import os


def doom_fast_enabled() -> bool:
    """``C4_DOOM_FAST`` (DEFAULT OFF): the single composite toggle for the whole byte-exact
    fast-doom perf stack.  OFF -> every member stays at its own default (OFF) -> golden
    ``069cc32f`` byte-identical."""
    return os.environ.get("C4_DOOM_FAST", "0") not in ("0", "", "false", "False")


# The blessed member set == transitive closure of the ~9.85 fps block-0-DK stack.
# (name, value).  All members are DEFAULT-OFF golden-byte-identical flags; value "1"
# is on, "512" is the wave-batch tile.  ORDER is documentation-only — every member is an
# independent env var read lazily by its own ``*_enabled()`` helper.
_DOOM_FAST_MEMBERS = (
    # -- dead-block / dead-FFN megakernel ------------------------------------------------
    ("C4_DEAD_BLOCK_FUSION", "1"),   # 238 dead-attn blocks -> output=x (the divfree skip)
    ("C4_FUSED_MEGABLOCK",   "1"),   # on-chip fused dead-FFN chain (1 CUDA-graph launch)
    ("C4_FFN_FUSED_HIDDEN",  "1"),   # keep the [Dff,K] hidden on-chip (no HBM round-trip)
    ("C4_FFN_WAVE_BATCH",    "1"),   # horizontal wave-batching of the dead-FFN chain
    ("C4_MEGABLOCK_BLOCK_K", "512"), # ** dep: wave-batch is INERT without a bigger block_k
    # -- direct-CAM gathers --------------------------------------------------------------
    ("C4_DIRECT_CAM_BATCHED", "1"),
    ("C4_DIRECT_LOCAL_CAM",   "1"),
    ("C4_DIRECT_CAM_VEC",     "1"),
    # -- local / flash attention ---------------------------------------------------------
    ("C4_FLASH_ATTN",        "1"),
    ("C4_BANDED_LOCAL_ATTN", "1"),
    # -- precomputed schedule (deps of ATTN_MEGABLOCK) -----------------------------------
    ("C4_PRECOMPUTED_SCHEDULE", "1"),
    ("C4_ONCHIP_RESIDUAL",      "1"),
    ("C4_RESIDENT_BATCH",       "1"),
    # -- attention megablock + block-0 [D,K] fold ----------------------------------------
    ("C4_ATTN_MEGABLOCK", "1"),      # ** dep: PRECOMPUTED_SCHEDULE + ONCHIP_RESIDUAL + FUSED_MEGABLOCK
    ("C4_BLOCK0_DK",      "1"),      # ** dep: ATTN_MEGABLOCK
    # -- GPU-built, cache-resolved, double-buffered schedule -----------------------------
    ("C4_SCHED_FAST_BUILD",     "1"),
    ("C4_SCHED_GPU_BUILD",      "1"),  # ** dep: SCHED_FAST_BUILD + ONCHIP_RESIDUAL
    ("C4_SCHED_PIPELINE",       "1"),  # ** dep: SCHED_GPU_BUILD (PIPELINE->GPU_BUILD)
    ("C4_SCHED_CACHE_RESOLVED", "1"),
)


def doom_fast_members() -> "tuple[tuple[str, str], ...]":
    """The blessed (flag, value) member set the composite expands to.  Read-only."""
    return _DOOM_FAST_MEMBERS


def expand_doom_fast(verbose: bool = False) -> "dict[str, str]":
    """If ``C4_DOOM_FAST`` is set, fill in every member flag that the user has NOT already
    set (``os.environ.setdefault`` semantics — a hand-set member, incl. ``=0`` to opt out,
    always wins).  Call ONCE, EARLY (before the first member read / model build).  Returns
    the ``{flag: value}`` map that was actually applied (empty if the composite is OFF).  A
    NO-OP when ``C4_DOOM_FAST`` is unset -> golden byte-identical."""
    if not doom_fast_enabled():
        return {}
    applied: "dict[str, str]" = {}
    for name, value in _DOOM_FAST_MEMBERS:
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    if verbose:
        overridden = [n for n, _ in _DOOM_FAST_MEMBERS if n not in applied]
        print(f"[C4_DOOM_FAST] applied {len(applied)} members: "
              f"{', '.join(f'{k}={v}' for k, v in applied.items())}", flush=True)
        if overridden:
            print(f"[C4_DOOM_FAST] left {len(overridden)} hand-set members untouched: "
                  f"{', '.join(overridden)}", flush=True)
    return applied


if __name__ == "__main__":
    # Demo: print what the composite would expand to for the current env.
    import sys
    on = doom_fast_enabled()
    print(f"C4_DOOM_FAST enabled: {on}")
    applied = expand_doom_fast(verbose=True)
    if not on:
        print("(set C4_DOOM_FAST=1 to see the member expansion)")
        print("members that WOULD be set:")
        for k, v in _DOOM_FAST_MEMBERS:
            print(f"  {k}={v}")
    sys.exit(0)
