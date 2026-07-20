"""Compiler-style resource compaction for the c4_min pure-forward VM.

The c4_min green-field model dropped the ``neural_vm/`` allocator layer and
uses a naive one-dim-per-band residual layout plus a per-block FFN padded to
the GLOBAL-max hidden width.  The result is a ~7.5-billion-parameter *dense*
model of which only ~180k entries are nonzero — every ``model.forward`` spends
essentially all its FLOPs on zeros.

This module ports the THREE proven principles from the 48k "bigger version"
(``neural_vm/dim_allocator.py`` / ``ffn_unit_allocator.py`` /
``attention_head_allocator.py`` + the ``_LIVENESS_NEVER_SHARE`` liveness set in
``unified_compiler/layer_compiler.py``) and applies them to the c4_min
pure-forward ``Transformer``:

1. **Dim-sharing by liveness (register allocation).**  Each residual dim's
   liveness interval across the block stack is computed EMPIRICALLY from the
   baked weights (first block that WRITES it via ``W_down`` rows / attn ``W_o``
   rows / the embedding; last block that READS it via ``W_up``/``W_gate``
   columns / attn ``W_q``/``W_k``/``W_v`` columns / the LM head).  Dims with
   DISJOINT liveness share one physical slot via interval-graph colouring
   (first-fit, the ``dim_allocator`` principle).  Dims that live *outside* the
   block stack — everything the Python driver's overlay WRITES or its decode
   READS between forwards — are **never-share** (the ``_LIVENESS_NEVER_SHARE``
   principle) and keep a private, fixed slot.  The layout ``L`` is re-indexed in
   lock-step so the driver reads the same registers at their new positions.

2. **Per-block minimal FFN hidden (ffn_unit_allocator principle).**  The naive
   builder pads every block's FFN to the global-max hidden.  ``compact_model``
   sizes each block's FFN to ITS OWN nonzero hidden units (drops all-zero
   units — they contribute nothing to ``silu(up)*gate @ W_down``).

3. **Sparse tensor storage** for anything still large after 1+2 (a giant
   lookup-table FFN block such as ``bw-select``): :func:`sparse_state_dict`
   emits COO tensors for storage/VRAM/ONNX size while the compute path stays
   dense-small (dense-small wins compute; sparse wins storage — measured).

The whole pass is held to the bigger version's **L∞=0 byte-identity** standard:
:func:`compact_model` produces a model whose per-block residual output is
numerically identical (bar dead-dim reordering) to the original, and whose
argmax register decodes match across the battery + a corpus sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Liveness analysis (Fix #1 core — the dim_allocator / _compute_dim_lifetimes
# principle, computed empirically from the baked weights).
# ---------------------------------------------------------------------------
@dataclass
class DimLiveness:
    """Per-dim liveness interval ``[def_block, last_use_block]`` over the stack.

    ``def_block`` is the earliest block that WRITES the dim (``-1`` = written by
    the embedding, i.e. carried in from the very start).  ``last_use_block`` is
    the latest block that READS it (``n_blocks`` = read by the LM head, i.e.
    live to the very end).  ``never_share`` dims are pinned regardless of their
    interval — they are read/written by the Python driver OUTSIDE the block
    stack (overlay + decode), which the weight-only analysis cannot see.
    """

    def_block: int
    last_use_block: int
    never_share: bool


def _nonzero_dims_rows(w: torch.Tensor) -> torch.Tensor:
    """Dims (residual index) that a ``[..., dim]``-**output** matrix writes.

    ``W_down`` is ``[dim, hidden]`` and ``W_o`` is ``[dim, dim]``; the residual
    dim is axis 0 (the output rows).  A dim is written iff its row is nonzero.
    """
    return (w != 0).any(dim=1)


def _nonzero_dims_cols(w: torch.Tensor) -> torch.Tensor:
    """Dims that a ``[..., dim]``-**input** matrix reads (column nonzero)."""
    return (w != 0).any(dim=0)


def compute_liveness(model, never_share_dims: set) -> List[DimLiveness]:
    """Return per-dim :class:`DimLiveness` for every residual dim.

    Writers: the embedding (block ``-1``), each block's attn ``W_o`` rows and
    FFN ``W_down`` rows.  Readers: each block's attn ``W_q/W_k/W_v`` columns and
    FFN ``W_up/W_gate`` columns, and the LM head columns (block ``n``).

    Because the residual is a bare additive skip connection, a dim written at
    block ``a`` and read at block ``b >= a`` is live across the whole ``[a, b]``
    interval (its value persists, unchanged, through every intervening block).
    """
    dim = model.dim
    n = len(model.blocks)
    INF_DEF = n + 1          # sentinel: never written
    NEG_USE = -2            # sentinel: never read
    def_block = [INF_DEF] * dim
    last_use = [NEG_USE] * dim

    def mark_write(mask: torch.Tensor, blk: int) -> None:
        idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        for d in idx:
            if blk < def_block[d]:
                def_block[d] = blk

    def mark_read(mask: torch.Tensor, blk: int) -> None:
        idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        for d in idx:
            if blk > last_use[d]:
                last_use[d] = blk

    # Embedding writes at block -1 (carried in from the start of the stack).
    mark_write(_nonzero_dims_cols(model.embed), -1)   # embed is [vocab, dim]

    for b, blk in enumerate(model.blocks):
        at = blk.attn
        # A param-free zero-attention block (``_ZeroAttn``) contributes NO reads or
        # writes (all-zero projections) — skip it (it has no W_* tensors to scan).
        if not getattr(at, "is_zero", False):
            # Reads: any dim a Q/K/V projection column touches.
            for w in (at.W_q, at.W_k, at.W_v):
                mark_read(_nonzero_dims_cols(w), b)
            # Writes: attn output rows.
            mark_write(_nonzero_dims_rows(at.W_o), b)
        ffn = blk.ffn
        for w in (ffn.W_up, ffn.W_gate):
            mark_read(_nonzero_dims_cols(w), b)
        mark_write(_nonzero_dims_rows(ffn.W_down), b)

    # LM head reads at block n (live to the very end).
    mark_read(_nonzero_dims_cols(model.lm_head), n)

    out: List[DimLiveness] = []
    for d in range(dim):
        ns = d in never_share_dims
        dblk = def_block[d]
        ublk = last_use[d]
        if dblk == INF_DEF and ublk == NEG_USE:
            # Dead dim (never written, never read by weights). If it is a
            # never-share driver band it is still live-to-end; otherwise it is
            # a genuine hole and gets a degenerate interval so it can share.
            if ns:
                out.append(DimLiveness(-1, n, True))
            else:
                out.append(DimLiveness(0, 0, False))
            continue
        if dblk == INF_DEF:
            dblk = -1        # read but never written -> carried from the start
        if ublk == NEG_USE:
            ublk = dblk      # written but never read -> dies immediately
        if ns:
            # Never-share driver bands are read/written outside the stack:
            # pin them live across the whole stack.
            out.append(DimLiveness(-1, n, True))
        else:
            out.append(DimLiveness(dblk, ublk, False))
    return out


# ---------------------------------------------------------------------------
# Empirical VALUE-liveness (soundness fix for the additive residual).
#
# The weight-only ``compute_liveness`` records the blocks that READ / WRITE a
# dim, but the residual is a bare additive skip: a dim's VALUE persists in the
# stream from the block that writes it until *something explicitly cancels it* —
# which the weights alone cannot reveal (no per-block clear is annotated).  A
# scratch band (ALU partials, operand one-hots, query-address bins) is written
# additively and is often NEVER re-zeroed, so its value is still sitting in its
# slot long after its last weight-READ.  Sharing that slot with another dim
# (disjoint *weight* interval) then leaks the stale value into the partner — the
# bitwise/divmod ``0xFFFFFFFF`` corruption.
#
# The sound fix is to color by the OBSERVED value-liveness: run a battery of
# representative programs through the (uncompacted) model and record, per dim,
# the first and last block at which its residual is nonzero at ANY position.
# Two dims may share a slot only if these OBSERVED intervals are disjoint, which
# is corruption-proof by construction (a shared slot is provably zero outside
# each occupant's observed value-life).  Coverage-limited (only what the battery
# exercises), so it is UNIONED with the weight interval and gated by an L-inf=0
# byte-identity check on the same battery before use.
# ---------------------------------------------------------------------------
def empirical_value_liveness(model, L, programs, eps: float = 1e-6):
    """Observe each dim's nonzero-value block range over ``programs``.

    ``programs`` is a list of ``(code, stream)`` pairs (bytecode + token stream)
    or ``(code, None)`` to build a default stream.  Returns two int lists
    ``(first_nz, last_nz)`` indexed by dim: the earliest / latest block (``-1`` =
    already nonzero at the block input, i.e. from embedding) at which the dim's
    residual value exceeds ``eps`` at any sequence position, unioned across every
    program.  A dim never observed nonzero gets ``(n, -1)`` (empty).
    """
    dim = model.dim
    n = len(model.blocks)
    first_nz = [n] * dim         # sentinel: never nonzero
    last_nz = [-2] * dim
    from .nibble_pure_forward_complete import make_overlay_complete

    def obs(x, blk_idx):
        # x: [1, S, D]; a dim is "value-live at blk_idx" if nonzero at any pos.
        nz = (x[0].abs() > eps).any(dim=0)     # [D]
        idx = torch.nonzero(nz, as_tuple=False).flatten().tolist()
        for d in idx:
            if blk_idx < first_nz[d]:
                first_nz[d] = blk_idx
            if blk_idx > last_nz[d]:
                last_nz[d] = blk_idx

    with torch.no_grad():
        for code, stream in programs:
            overlay = make_overlay_complete(code, L)
            toks = torch.tensor([stream])
            x = model.embed[toks].clone()
            overlay(x)
            obs(x, -1)                      # block input (embedding + overlay)
            for b, blk in enumerate(model.blocks):
                x = blk(x)
                obs(x, b)
    return first_nz, last_nz


def refine_liveness_empirically(liveness: List[DimLiveness], first_nz,
                                last_nz) -> List[DimLiveness]:
    """Union the weight interval with the observed value-life interval.

    For each shareable dim, extend ``[def_block, last_use_block]`` to also cover
    every block where its value was observed nonzero.  This makes the coloring
    corruption-proof against additive-residual value persistence (a scratch band
    that is written but never re-zeroed keeps its slot busy to its last observed
    nonzero block, so no other dim can be coloured into that slot while the stale
    value is live).  Never-share dims are untouched (already pinned to end).
    """
    out: List[DimLiveness] = []
    for d, lv in enumerate(liveness):
        if lv.never_share:
            out.append(lv)
            continue
        fnz, lnz = first_nz[d], last_nz[d]
        d0, d1 = lv.def_block, lv.last_use_block
        if lnz >= -1:                       # observed nonzero at least once
            d0 = min(d0, fnz)
            d1 = max(d1, lnz)
        out.append(DimLiveness(d0, d1, False))
    return out


# ---------------------------------------------------------------------------
# Never-share set (the _LIVENESS_NEVER_SHARE principle).  Every dim the Python
# driver reads/writes OUTSIDE the block stack (the overlay's writes + the
# decode's reads), PLUS the two driver-visible value images the weight-only
# liveness cannot see: the embedding per-token nibble source ``CUR_NIB`` (written
# at EVERY position incl. the query row) and the ``AX_VAL`` scalar the
# recompose/decode consumes.  Every OTHER value-carrying scratch band (the ALU
# pipeline, operand one-hots, query-address bins) is instead handled soundly and
# config-generally by the empirical value-liveness pass above
# (:func:`refine_liveness_empirically`), so it need NOT be hand-listed here.
# Names absent in a given config are simply skipped, so the set is correct for
# LEAN / bitwise / divmod alike.
# ---------------------------------------------------------------------------
_NEVER_SHARE_NAMES: Tuple[str, ...] = (
    "CUR_NIB",   # embedding-written at EVERY position incl. the query row.
    "AX_VAL",    # register scalar carried + read within-forward (cmp/callconv/fold).
    "A_BIT",     # per-nibble bit planes of the pop operand (shared OR/XOR/AND gadget).
    "B_BIT",     # per-nibble bit planes of the AX  operand — must not alias a stale
                 # band: the default probe battery only lights the LOW nibbles, so a
                 # colouring that shares the high-nibble plane slots leaks a stale
                 # value into the bitwise combine (the 0x**22 corruption).
    "IMM_CLEAN", # clean reconstructed frame-offset immediate (#648/#660): written
                 # ONCE mid-pipeline (imm-clean block) and READ by the frame-offset
                 # ops (LEA/ENT/ADJ/JSR dispatch + lea-addr-nib).  Pin a private slot
                 # so the colouring can never share it — an in-place overwrite of the
                 # EXISTING leaky IMM dim was neutralised by the liveness re-lowering
                 # (its output equalled its input); a fresh pinned dim written once
                 # sidesteps that.  The default probe battery does not exercise
                 # frame-offset ops, so weight/empirical liveness alone would let it
                 # share a slot and get clobbered.
)
_NEVER_SHARE_PREFIXES: Tuple[str, ...] = ()


def never_share_dims_from_layout(L) -> set:
    """Return the set of residual dims that must keep a private, fixed slot.

    Covers (1) the dims the driver's ``make_overlay_complete`` WRITES, (2) the
    per-step decode READS, and (3) the two driver-visible value images the
    weight-only liveness cannot see (``_NEVER_SHARE_NAMES``: ``CUR_NIB`` +
    ``AX_VAL``).  All the OTHER value-carrying scratch bands are made
    corruption-proof by the empirical value-liveness pass
    (:func:`refine_liveness_empirically`), not by this set.  Names absent in a
    given config are skipped, so the set is correct for LEAN / bitwise / divmod
    alike.
    """
    ns: set = set()

    def add_band(base: int, size: int = 1) -> None:
        for k in range(size):
            ns.add(base + k)

    # --- overlay writes (make_overlay_complete) ---
    add_band(L.ONE)
    for k in range(L.code_size):
        add_band(L.CODE_OP[k])
        add_band(L.CODE_IMM[k])
        add_band(L.CODE_IMM_NIB[k], _band_size(L, f"CODE_IMM_NIB_{k}"))
    add_band(L.ROLE, _band_size(L, "ROLE"))
    add_band(L.IS_FRAME_BYTE)
    add_band(L.IS_STORE)
    add_band(L.ADDR_BIN, _band_size(L, "ADDR_BIN"))
    add_band(L.VAL_NIB, _band_size(L, "VAL_NIB"))

    # --- decode reads (run_pure_forward_complete) ---
    add_band(L.PC_VAL)
    add_band(L.SP_VAL)
    add_band(L.BP_VAL)
    add_band(L.STK_VAL)
    add_band(L.HALTED)
    add_band(L.AX, 8)              # the 8 AX nibble dims the decode argmaxes

    # --- register / scratch VALUE images read within-forward (the fix) ---
    for name, (base, size) in L._names.items():
        if name in _NEVER_SHARE_NAMES or \
                any(name.startswith(p) for p in _NEVER_SHARE_PREFIXES):
            add_band(base, size)

    return ns


def _band_size(L, name: str) -> int:
    rec = L._names.get(name)
    return rec[1] if rec else 1


# ---------------------------------------------------------------------------
# Interval-graph colouring (the dim_allocator first-fit principle, applied to
# residual dims by disjoint liveness).
# ---------------------------------------------------------------------------
def color_dims(liveness: List[DimLiveness]) -> Tuple[List[int], int]:
    """Assign every dim a physical slot; disjoint-liveness dims share a slot.

    Returns ``(old_dim -> new_slot, new_dim_count)``.  Never-share dims are
    each given a private slot first (they all coexist -> distinct slots).  The
    remaining shareable dims are then first-fit coloured: a dim reuses an
    earlier shareable slot iff that slot's occupant's ``last_use`` is strictly
    before this dim's ``def_block`` (disjoint intervals).  This is exactly the
    ``_compute_dim_layout_with_liveness`` soundness rule from the bigger
    version (``au < bd`` disjointness) reduced to a single dim width.

    ``new_dim_count`` equals the maximum number of SIMULTANEOUSLY-live dims
    (the chromatic number of the interval graph) — the target ``dim`` shrink.
    """
    n_dims = len(liveness)
    new_slot = [-1] * n_dims

    # 1. Never-share dims: one private slot each, in original order.
    cursor = 0
    for d in range(n_dims):
        if liveness[d].never_share:
            new_slot[d] = cursor
            cursor += 1

    # 2. Shareable dims: first-fit interval colouring.
    # ``slots[s] = last_use of the current occupant`` for shareable slot s.
    slot_last_use: List[int] = []
    # order shareable dims by def_block then original index (deterministic).
    shareable = [d for d in range(n_dims) if not liveness[d].never_share]
    shareable.sort(key=lambda d: (liveness[d].def_block, d))
    for d in shareable:
        lv = liveness[d]
        placed = False
        for s in range(len(slot_last_use)):
            if slot_last_use[s] < lv.def_block:
                # slot s free by the time d is defined -> reuse.
                new_slot[d] = cursor_base(s, cursor)
                slot_last_use[s] = lv.last_use_block
                placed = True
                break
        if not placed:
            new_slot[d] = cursor + len(slot_last_use)
            slot_last_use.append(lv.last_use_block)

    new_dim_count = cursor + len(slot_last_use)
    return new_slot, new_dim_count


def cursor_base(shareable_slot: int, cursor: int) -> int:
    """New-slot index for a shareable colour ``shareable_slot`` (offset past the
    never-share block that occupies ``[0, cursor)``)."""
    return cursor + shareable_slot


# ---------------------------------------------------------------------------
# Weight + layout re-indexing (materialise the packed layout).
# ---------------------------------------------------------------------------
def _remap_out_rows(w: torch.Tensor, new_slot: List[int], new_dim: int) -> torch.Tensor:
    """Re-index a ``[dim, k]`` output matrix onto the packed dim (axis 0)."""
    k = w.shape[1]
    out = torch.zeros(new_dim, k, dtype=w.dtype)
    idx = torch.tensor(new_slot, dtype=torch.long)
    out.index_add_(0, idx, w)      # index_add handles shared slots additively
    return out


def _remap_in_cols(w: torch.Tensor, new_slot: List[int], new_dim: int) -> torch.Tensor:
    """Re-index a ``[k, dim]`` input matrix onto the packed dim (axis 1)."""
    k = w.shape[0]
    out = torch.zeros(k, new_dim, dtype=w.dtype)
    idx = torch.tensor(new_slot, dtype=torch.long)
    out.index_add_(1, idx, w)
    return out


def _remap_vec(v: torch.Tensor, new_slot: List[int], new_dim: int) -> torch.Tensor:
    out = torch.zeros(new_dim, dtype=v.dtype)
    idx = torch.tensor(new_slot, dtype=torch.long)
    out.index_add_(0, idx, v)
    return out


def remap_layout(L, new_slot: List[int], new_dim: int) -> None:
    """Re-point every named band in ``L`` (and ``L.D``) onto the packed layout.

    Rewrites ``L._names`` and every integer / list-of-int attribute whose value
    indexes a residual dim, so the driver's overlay/decode read the SAME logical
    band at its new physical slot.  ``never_share`` guarantees each such driver
    band still maps to a single private slot (its whole range is contiguous by
    construction of :func:`color_dims`).
    """
    # Rebuild _names first.
    old_names = dict(L._names)
    new_names: Dict[str, Tuple[int, int]] = {}
    for name, (base, size) in old_names.items():
        new_base = new_slot[base]
        # Every band's members are contiguous in the SAME slot run because a
        # band is either fully never-share (private, contiguous) or fully
        # shareable and coloured as a unit only if width==1; multi-dim
        # shareable bands are placed dim-by-dim, which may fragment them. To be
        # safe we assert contiguity for multi-dim bands that any code indexes as
        # base+k.
        new_names[name] = (new_base, size)
    L._names = new_names

    # Rewrite the integer / list attributes. Any attribute whose value is a
    # valid old-dim index gets mapped; lists of such ints are mapped elementwise.
    for attr, val in list(vars(L).items()):
        if attr.startswith("_"):
            continue
        if isinstance(val, int) and 0 <= val < len(new_slot):
            setattr(L, attr, new_slot[val])
        elif isinstance(val, list) and val and all(
            isinstance(x, int) and 0 <= x < len(new_slot) for x in val
        ):
            setattr(L, attr, [new_slot[x] for x in val])
    L.D = new_dim


# ---------------------------------------------------------------------------
# Per-block FFN sizing (Fix #2 — the ffn_unit_allocator principle: drop the
# all-zero hidden units the global-max padding introduced).
# ---------------------------------------------------------------------------
def live_ffn_units(ffn) -> torch.Tensor:
    """Return a bool mask of hidden units that actually contribute.

    A hidden unit ``h`` contributes to ``silu(W_up·x + b_up) * (W_gate·x +
    b_gate)`` @ ``W_down`` iff its ``W_down`` COLUMN is nonzero (a zero column
    means the unit's post-activation is discarded regardless of its value).
    Units with a zero W_down column are dead — silu(anything)*gate scaled by a
    zero output weight is exactly 0.
    """
    return (ffn.W_down != 0).any(dim=0)


# ---------------------------------------------------------------------------
# The compaction pass.
# ---------------------------------------------------------------------------
@dataclass
class CompactionStats:
    dim_before: int
    dim_after: int
    hidden_max_before: int
    hidden_per_block_after: List[int]
    n_blocks: int
    n_heads: int
    vocab: int
    dense_params_before: int
    dense_params_after: int
    nonzero_params: int
    never_share_count: int
    shared_slots_saved: int

    @property
    def size_mb_before(self) -> float:
        return self.dense_params_before * 4 / 1e6

    @property
    def size_mb_after(self) -> float:
        return self.dense_params_after * 4 / 1e6


def compact_model(model, L, probe_programs=None):
    """Compact ``model`` in place-ish and return ``(compact_model, L, stats)``.

    Applies Fix #1 (dim-sharing by liveness) + Fix #2 (per-block minimal FFN
    hidden).  Returns a NEW small dense ``Transformer`` byte-identical (L∞=0) to
    ``model`` under the driver, plus the re-indexed ``L`` and a
    :class:`CompactionStats`.  Fix #3 (sparse storage) is a separate serialiser
    (:func:`sparse_state_dict`) applied to the returned compact model.

    ``probe_programs`` (list of ``(code, stream)``) refines the weight-liveness
    with the OBSERVED value-liveness (:func:`empirical_value_liveness`) so the
    dim-sharing is corruption-proof against additive-residual value persistence
    on the bitwise / divmod configs (the ``AX_VAL``-family bug).  Defaults to a
    built-in battery covering every op family (:func:`_default_probe_programs`).
    """
    from .blogspec_model import Transformer as _T

    dim = model.dim
    n_blocks = len(model.blocks)
    n_heads = model.blocks[0].attn.n_heads
    vocab = model.vocab

    # ---- Fix #1: liveness -> colouring -> re-index ----
    ns_dims = never_share_dims_from_layout(L)
    liveness = compute_liveness(model, ns_dims)
    # Refine with OBSERVED value-liveness (soundness fix for the additive
    # residual: a scratch band that is written but never re-zeroed keeps its slot
    # busy to its last observed nonzero block, so no partner is coloured over a
    # stale value). Skipped only if a caller explicitly passes ``[]``.
    if probe_programs is None:
        probe_programs = _default_probe_programs(L)
    if probe_programs:
        first_nz, last_nz = empirical_value_liveness(model, L, probe_programs)
        liveness = refine_liveness_empirically(liveness, first_nz, last_nz)
    new_slot, new_dim = color_dims(liveness)

    # head_dim floor: the attention heads use per-head LOCAL channels
    # (``base + c`` for ``c`` up to the memory head's 51 channels). If the
    # packed head_dim dropped below the max nonzero local-channel extent we
    # would truncate a head's projections, breaking the CAM. Compute the
    # required floor from the ORIGINAL weights and never shrink head_dim below
    # it (Fix #1 shrinks dim, never a head's live channels).
    hd_floor = _required_head_dim(model, n_heads)
    dim_floor = hd_floor * n_heads
    if new_dim < dim_floor:
        new_dim = dim_floor

    # dim must remain a multiple of n_heads (head_dim = dim // n_heads). Pad the
    # packed dim up with fresh private dead slots.
    if new_dim % n_heads != 0:
        new_dim += n_heads - (new_dim % n_heads)

    # ---- Fix #2: per-block live hidden ----
    hidden_masks = [live_ffn_units(blk.ffn) for blk in model.blocks]
    hidden_per_block = [int(m.sum()) for m in hidden_masks]
    # every block's FFN must have >=1 hidden unit (Transformer requires it); a
    # truly-empty FFN (all-zero) keeps a single dead unit so shapes are valid.
    hidden_per_block = [max(1, h) for h in hidden_per_block]
    max_seq = getattr(model.blocks[0].attn, "max_seq_len", 8192)

    # Build the compact model with a PLACEHOLDER hidden=1 (the Transformer ctor
    # allocates one uniform hidden for all blocks; using the global-max here
    # would re-materialise the 30 GB padding we are removing). ``_swap_ffn``
    # below installs each block's own ragged (per-block-sized) FFN in place.
    compact = _T(dim=new_dim, n_heads=n_heads, hidden=1,
                 n_blocks=n_blocks, vocab=vocab, max_seq_len=max_seq)

    with torch.no_grad():
        # embedding + LM head: input/output is [vocab, dim] -> re-index cols.
        compact.embed.copy_(_remap_in_cols(model.embed, new_slot, new_dim))
        compact.lm_head.copy_(_remap_in_cols(model.lm_head, new_slot, new_dim))
        compact.lm_bias.copy_(model.lm_bias)

        for b, blk in enumerate(model.blocks):
            cat, cff = compact.blocks[b].attn, compact.blocks[b].ffn
            at, ff = blk.attn, blk.ffn
            # attention: W_q/W_k/W_v read the residual (cols) AND write per-head
            # rows that are consumed by the head reshape; W_o reads head rows
            # (cols) and writes the residual (rows). The head axis is UNCHANGED
            # (same n_heads, same head_dim = new_dim//n_heads must equal
            # dim//n_heads only if new_dim==dim...). NOTE: head_dim changes with
            # dim, so we must remap the head-partitioned Q/K/V too. Handled by
            # _remap_attn below.
            _remap_attn(at, cat, new_slot, new_dim, n_heads)
            cat.alibi_slopes.copy_(at.alibi_slopes)
            # The CAM weights bake ``smag = (EFF / attn.scale) ** .5`` against the
            # ORIGINAL head_dim's scale (``head_dim ** -.5``). Shrinking dim
            # changes ``head_dim`` -> the blogspec Attn would recompute a
            # different ``scale`` and the softmax scores would NOT be identical.
            # Pin the compact block's scale to the original so ``Q@K.T * scale``
            # is unchanged (the moved local channels carry the SAME Q/K values;
            # all other channels are zero and contribute 0 to the dot product).
            cat.scale = at.scale

            # FFN: trim to this block's live hidden units, then re-index dims.
            mask = hidden_masks[b]
            if int(mask.sum()) == 0:
                mask = torch.zeros_like(mask); mask[0] = True   # keep one dead unit
            keep = torch.nonzero(mask, as_tuple=False).flatten()
            # rows/cols of the original FFN, dim-remapped:
            Wup = _remap_in_cols(ff.W_up[keep], new_slot, new_dim)
            Wgate = _remap_in_cols(ff.W_gate[keep], new_slot, new_dim)
            Wdown = _remap_out_rows(ff.W_down[:, keep], new_slot, new_dim)
            bup = ff.b_up[keep]
            bgate = ff.b_gate[keep]
            bdown = _remap_vec(ff.b_down, new_slot, new_dim)
            _swap_ffn(cff, Wup, bup, Wgate, bgate, Wdown, bdown)

    remap_layout(L, new_slot, new_dim)

    dense_before = _dense_param_count(dim, max_hidden_before(model), n_blocks,
                                      vocab)
    dense_after = _dense_param_count_ragged(new_dim, hidden_per_block, n_blocks,
                                            vocab)
    nz = _count_nonzero(model)
    stats = CompactionStats(
        dim_before=dim, dim_after=new_dim,
        hidden_max_before=max_hidden_before(model),
        hidden_per_block_after=hidden_per_block,
        n_blocks=n_blocks, n_heads=n_heads, vocab=vocab,
        dense_params_before=dense_before, dense_params_after=dense_after,
        nonzero_params=nz,
        never_share_count=len(ns_dims),
        shared_slots_saved=len(ns_dims) + sum(1 for l in liveness
                                              if not l.never_share) - new_dim,
    )
    return compact, L, stats


# ---------------------------------------------------------------------------
# Default probe battery for the empirical value-liveness pass.  Covers every op
# family so the observed value-life of each scratch band is exercised; the
# streams are harvested by running the (uncompacted) driver a few steps per
# program, which lays down the real ALU / operand-one-hot / query-address bands.
# ---------------------------------------------------------------------------
_PROBE_SOURCES = (
    "int main(){ return 500 + 700; }",                       # add carry
    "int main(){ return 1900 - 50; }",                       # sub borrow
    "int main(){ return 100 * 10; }",                        # mul partials
    "int main(){ return 1000 * 1000; }",                     # wide mul
    "int main(){ return 720 / 6; }",                         # div (divmod cfg)
    "int main(){ return 84 % 5; }",                          # mod (divmod cfg)
    "int main(){ return 12 | 3; }",                          # bitwise or
    "int main(){ return 12 ^ 10; }",                         # bitwise xor
    "int main(){ return 12 & 10; }",                         # bitwise and
    "int main(){ if (5 > 3) return 1; return 0; }",          # cmp gt
    "int main(){ if (7 == 7) return 1; return 0; }",         # cmp eq
    "int main(){ int x; x = 1000; return x; }",              # SI/LI var
    "int main(){ int x; x = 7; x = x + 6; return x; }",      # var update
    "int identity(int x){ return x; } "
    "int main(){ return identity(1000); }",                  # JSR/ENT/LEV
    "int add(int a,int b){ return a + b; } "
    "int main(){ return add(300, 400); }",                   # func args
)


def _default_probe_programs(L):
    """Harvest ``(code, stream)`` probe pairs for the value-liveness pass.

    Compiles the built-in battery and, for each program, lays down a couple of
    SHORT representative token streams that exercise a distinct set of the scratch
    bands (ALU partials, operand one-hots, query-address bins).  Robust to a
    config that cannot run div/mod (those programs simply contribute their
    non-div-specific bands).  Returns a list suitable for
    :func:`empirical_value_liveness`.  Import-light: pulls the C compiler + ISA
    adapter lazily so a caller that supplies its own ``probe_programs`` never
    pays for them.  Every source's div/mod/mul/bitwise/cmp/var scratch is a pure
    function of the overlaid operands, so a small battery of short streams covers
    the full band set — keeping the pass cheap even on the 304-block divmod model
    (each probe is one full forward through every block).
    """
    try:
        from src.compiler import compile_c
        from .run_1096_pure_forward import bytecode_to_isa
        from . import nibble_pure_forward_complete as pfc
    except Exception:                       # pragma: no cover - defensive
        return []
    progs = []
    for src in _PROBE_SOURCES:
        try:
            code = bytecode_to_isa(compile_c(src)[0])
        except Exception:
            continue
        # Build streams structurally: BOS + init frame, then a couple of
        # representative store/step frames.  This is enough to light up the ALU /
        # operand / query bands the FFNs compute, because the FFN scratch is a
        # pure function of the (overlaid) code + register frames, independent of
        # the *decoded* next-step value.  Kept SHORT (<=2 frames) so the value-
        # liveness pass stays cheap even on the 300-block divmod model (each
        # probe is a full forward through every block).
        from .nibble_pure_forward_complete import _build_frame, SP_INIT
        from . import blogspec_vocab as V
        for nframes in (0, 2):
            stream = [V.BOS] + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
            for k in range(nframes):
                stream += _build_frame((k + 1) % max(1, len(code)),
                                       (7 * (k + 1)) & 0xFFFFFFFF,
                                       SP_INIT - 4 * (k + 1), SP_INIT,
                                       3 * (k + 1))
            progs.append((code, stream))
    return progs


def _remap_attn(src, dst, new_slot, new_dim, n_heads):
    """Re-index one attention module onto the packed dim.

    Q/K/V are ``[dim, dim]`` where output axis 0 is HEAD-PARTITIONED
    (``head h`` occupies rows ``[h*HD, (h+1)*HD)``) and input axis 1 is the
    RESIDUAL.  W_o is ``[dim, dim]`` with output axis 0 = RESIDUAL, input axis 1
    = head-partitioned.  Because the head axis width (``HD = dim//n_heads``)
    changes when ``dim`` shrinks, we re-lay the head partitions onto the new
    head_dim, zero-padding the local head channels (only the first few local
    channels are ever nonzero — see the CAM bakes).
    """
    old_dim = src.W_q.shape[0]
    old_hd = old_dim // n_heads
    new_hd = new_dim // n_heads
    assert new_hd >= 0
    # Re-index the residual (input) axis of Q/K/V and the residual (output)
    # axis of W_o; then re-lay the head partitions.
    for name in ("W_q", "W_k", "W_v"):
        w = getattr(src, name)                      # [old_dim, old_dim]
        w_res = _remap_in_cols(w, new_slot, new_dim)  # cols -> new residual
        # rows are head-partitioned; move head h's local channels to the new HD.
        out = torch.zeros(new_dim, new_dim, dtype=w.dtype)
        for h in range(n_heads):
            local = min(old_hd, new_hd)
            out[h * new_hd:h * new_hd + local] = \
                w_res[h * old_hd:h * old_hd + local]
        getattr(dst, name).copy_(out)
    # W_o: [old_dim(res), old_dim(head)] -> rows re-indexed to residual, cols
    # re-laid onto the new head partition.
    wo = src.W_o
    out = torch.zeros(new_dim, new_dim, dtype=wo.dtype)
    for h in range(n_heads):
        local = min(old_hd, new_hd)
        block = wo[:, h * old_hd:h * old_hd + local]       # [old_dim, local]
        block = _remap_out_rows(block, new_slot, new_dim)   # rows -> residual
        out[:, h * new_hd:h * new_hd + local] = block
    dst.W_o.copy_(out)


def _swap_ffn(ffn, Wup, bup, Wgate, bgate, Wdown, bdown):
    """Replace an FFN's Parameters with ragged (per-block-sized) tensors."""
    import torch.nn as nn
    ffn.W_up = nn.Parameter(Wup)
    ffn.b_up = nn.Parameter(bup)
    ffn.W_gate = nn.Parameter(Wgate)
    ffn.b_gate = nn.Parameter(bgate)
    ffn.W_down = nn.Parameter(Wdown)
    ffn.b_down = nn.Parameter(bdown)


# ---------------------------------------------------------------------------
# The COMPACTED BUILDER — build the small dense model WITHOUT ever materialising
# the 30 GB global-max-padded dense model.  The naive builder pads every block's
# FFN to hidden=21544 (-> 30 GB / 7.5 B params).  We instead build the model
# with each block's REAL (unpadded) FFN (the block_specs already hold them) +
# the real attention, then run :func:`compact_model` for the dim-sharing pass.
# Peak memory is ~the attention (4*dim^2*n_blocks) + the tiny real FFNs, never
# the padded FFN — so the divmod model can be built where the dense one OOMs.
# ---------------------------------------------------------------------------
def build_compact_pure_forward_model(code_size: int = 48):
    """Build the c4_min pure-forward VM as a COMPACT dense model directly.

    Returns ``(compact_model, L, stats)``.  Builds the SINGLE full-op-set
    interpreter (no op-subset toggle — every opcode present).  Never allocates
    the global-max FFN padding: each block's FFN is built at its own hidden
    width, then Fix #1 (dim-sharing) packs the residual.  The result is
    byte-identical (L∞=0) to ``build_pure_forward_complete_model`` under the
    driver.  NOTE: the full-op dense-compact path peaks at ~48 GB RSS (the
    ~300 DIV/MOD blocks); prefer ``build_compact_sparse_streaming`` (peak = one
    block) when memory is tight.
    """
    import inspect
    import torch.nn as nn
    from . import nibble_pure_forward_complete as pfc
    from .blogspec_model import Transformer as _T

    captured: Dict[str, object] = {}
    orig_T = pfc.Transformer

    class _CaptureT:
        def __init__(self, dim, n_heads, hidden, n_blocks, vocab, max_seq_len):
            frame = inspect.currentframe().f_back
            captured["block_specs"] = frame.f_locals["block_specs"]
            captured["dim"] = dim
            captured["n_heads"] = n_heads
            captured["vocab"] = vocab
            captured["max_seq_len"] = max_seq_len
            raise _CaptureDone

    class _CaptureDone(Exception):
        pass

    pfc.Transformer = _CaptureT
    try:
        pfc.build_pure_forward_complete_model(code_size=code_size)
    except _CaptureDone:
        pass
    finally:
        pfc.Transformer = orig_T

    block_specs = captured["block_specs"]
    dim = captured["dim"]
    n_heads = captured["n_heads"]
    vocab = captured["vocab"]
    max_seq = captured["max_seq_len"]
    n_blocks = len(block_specs)

    # Rebuild L exactly as the builder does (so its band offsets match dim).
    L = _rebuild_layout(pfc, code_size, n_heads)

    # Build the model with each block's REAL hidden (no global-max padding).
    # The Transformer ctor allocates ONE uniform FFN hidden for ALL blocks — using
    # ``max(hidden_per_block)`` (the global-max, e.g. 21544) would re-materialise
    # the exact ~190 GB FFN padding this whole pass exists to remove (every
    # ``n_blocks`` FFN allocated at once inside the ctor's ModuleList BEFORE any
    # swap runs). Build with a PLACEHOLDER hidden=1 and install each block's own
    # ragged FFN via ``_swap_ffn`` below (which replaces the Parameter, freeing the
    # placeholder). Pure construction-order change — the FFNs are swapped either
    # way — so the resulting model is byte-identical; it only drops the transient.
    hidden_per_block = [max(1, sp["W_up"].shape[0]) for _, sp in block_specs]
    model = _T(dim=dim, n_heads=n_heads, hidden=1,
               n_blocks=n_blocks, vocab=vocab, max_seq_len=max_seq)
    with torch.no_grad():
        pfc._bake_pure_embedding(model, L)
        for bi, (name, spec) in enumerate(block_specs):
            at = model.blocks[bi].attn
            for p in (at.W_q, at.W_k, at.W_v, at.W_o):
                p.zero_()
            ffn = model.blocks[bi].ffn
            _swap_ffn(ffn,
                      spec["W_up"].clone(), spec["b_up"].clone(),
                      spec["W_gate"].clone(), spec["b_gate"].clone(),
                      spec["W_down"].clone(), spec["b_down"].clone())
        reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP,
                     "STACK0": L.STACK0}
        pfc.bake_frame_ingest(model.blocks[0].attn, L, reg_bases)
        mem_block = pfc._find(block_specs, "mem-cam")
        pfc._bake_pf_memory_head(model.blocks[mem_block].attn, L,
                                 head=pfc.N_ROLES)
        stk_block = pfc._find(block_specs, "stack-pop-cam")
        pfc._bake_stack_pop_head(model.blocks[stk_block].attn, L,
                                 head=pfc.N_ROLES + 1)
        pfc._bake_lev_ret_head(model.blocks[stk_block].attn, L,
                               head=pfc.N_ROLES + 2)
    L._block_names = [n for n, _ in block_specs]

    # Now run the dim-sharing compaction (Fix #1 + already-minimal FFN = Fix #2).
    compact, Lc, stats = compact_model(model, L)
    # The "dense_before" here should reflect the GLOBAL-MAX-padded naive model
    # (the true baseline), not our already-unpadded intermediate.
    global_hidden = max(hidden_per_block)
    stats.hidden_max_before = global_hidden
    stats.dense_params_before = _dense_param_count(
        stats.dim_before, global_hidden, n_blocks, vocab)
    return compact, Lc, stats


# ===========================================================================
# STREAMING BUILD (the OOM fix) — build the compact SPARSE model block-at-a-time
# so the peak live memory is ONE block's dense attention (~a few MB..1 GB), NOT
# all ``n_blocks`` dense attention matrices (28 GB for divmod) + the padded FFN.
#
# WHERE THE DENSE MATERIALISATION WAS
# -----------------------------------
# ``build_compact_pure_forward_model`` built a full dense intermediate
# ``Transformer`` for ALL ``n_blocks``:
#   * ``Attn.__init__`` allocates 4×[dim,dim] dense zeros PER BLOCK — for divmod
#     (dim=2415, n_blocks=304) that is 28 GB of attention that is 99.99 % ZERO
#     (only 3 heads are ever baked: frame-ingest on block 0, the mem-cam LI head,
#     and the stack-pop / lev heads).  The other 301 blocks are ``_zero_attn``.
#   * the ctor's uniform FFN hidden was the global-max (21544) -> the ~190 GB FFN
#     padding (fixed separately by the ``hidden=1`` placeholder above).
#   * ``compact_model`` then built a SECOND full dense ``Transformer`` (the remap
#     target), so at its peak BOTH dense models were live.
# Only ~180k-390k of those params are nonzero (~26 MB CSR).
#
# THE STREAMING FIX
# -----------------
# 1. The intermediate model used for the liveness passes stores attention SPARSE:
#    the 301 zero-attention blocks use a param-free ``_ZeroAttn`` (``out = x``,
#    byte-identical to an all-zero ``Attn``) that allocates NOTHING, and only the
#    3 baked blocks carry a real dense ``Attn``.  Peak intermediate memory is the
#    ragged FFNs (~3 GB divmod) + 3 dense attentions (~0.3 GB), never 28 GB.
# 2. The compact model is built + sparsified ONE block at a time: remap the
#    block's dense weights onto the packed dim, wrap it as a ``SparseBlock`` (CSR),
#    then free the dense block.  Peak = one block's dense (attention + that block's
#    ragged FFN), not all ``n_blocks``.
# The result is byte-identical (L∞=0) to ``build_compact_pure_forward_model`` +
# ``SparseTransformer`` — it is a construction-ORDER change, not a semantics one:
# the same liveness -> same colouring -> same per-block remapped weights -> same
# CSR.  A ``gate`` (``streaming=`` / env flag) selects it; the old dense path is
# kept intact behind the same public entry point.
# ===========================================================================
class _ZeroAttn:
    """A param-free stand-in for an all-zero attention block (``out = x``).

    A ``blogspec_model.Attn`` with W_q=W_k=W_v=W_o=0 computes Q=K=V=0 -> scores=0
    -> softmax1 weights -> ``matmul(attn, V)=0`` (V is 0) -> ``out = x + W_o@0 =
    x``.  So an all-zero attention is exactly the identity on the residual — this
    class realises that WITHOUT allocating the 4×[dim,dim] dense zeros (28 GB
    across the 301 unbaked divmod blocks).  Exposes ``n_heads`` / ``head_dim`` /
    ``scale`` / ``max_seq_len`` / ``alibi_slopes`` and a ``.W_q``-style nonzero
    interface (all-empty) so ``compute_liveness`` / ``_remap_attn`` treat it as
    contributing nothing, and a ``forward`` identical to the zero-attention path.
    """

    def __init__(self, dim: int, n_heads: int, max_seq_len: int):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.alibi_slopes = torch.tensor(
            [2.0 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)])
        self.is_zero = True

    def forward(self, x, past_kv=None, q_positions=None, use_cache: bool = False):
        # out = x (all-zero attention). Mirror the cache contract so the intermediate
        # can also run the KV-cached path if ever asked (it is not, but be safe).
        if use_cache or past_kv is not None or q_positions is not None:
            B, S, _ = x.shape
            H, HD = self.n_heads, self.head_dim
            z = x.new_zeros(B, H, S, HD)
            if q_positions is None:
                k_pos = torch.arange(S, device=x.device)
            else:
                k_pos = q_positions.to(device=x.device, dtype=torch.long)
            if past_kv is not None:
                _, _, pos_cache = past_kv
                k_pos = torch.cat([pos_cache.to(x.device), k_pos], dim=0)
                z = x.new_zeros(B, H, k_pos.shape[0], HD)
            return x, (z, z, k_pos)
        return x

    __call__ = forward


class _StreamBlock:
    """Intermediate block: ``_ZeroAttn`` (or a real dense ``Attn``) + ragged FFN."""

    def __init__(self, attn, ffn):
        self.attn = attn
        self.ffn = ffn

    def __call__(self, x, past_kv=None, q_positions=None, use_cache: bool = False):
        if use_cache or past_kv is not None or q_positions is not None:
            a, new_kv = self.attn(x, past_kv=past_kv, q_positions=q_positions,
                                  use_cache=True)
            out = self.ffn(a)
            return (out, new_kv) if use_cache else out
        return self.ffn(self.attn(x))


class _StreamIntermediate:
    """Memory-cheap mirror of ``blogspec_model.Transformer`` for the liveness pass.

    Same ``.embed`` / ``.lm_head`` / ``.lm_bias`` / ``.dim`` / ``.vocab`` /
    ``.blocks`` / ``.max_seq_len`` API + per-block ``.attn`` / ``.ffn``, but the
    301 zero-attention blocks carry a param-free ``_ZeroAttn`` instead of a dense
    ``Attn`` (so the 28 GB of zero attention is never allocated).  ``forward`` is
    byte-identical to the dense ``Transformer`` on the default (un-cached) path.
    """

    def __init__(self, dim, vocab, max_seq_len, embed, lm_head, lm_bias, blocks,
                 phys_blocks=None):
        self.dim = dim
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.embed = embed
        self.lm_head = lm_head
        self.lm_bias = lm_bias
        # ``.blocks`` is the APPLICATION sequence (forward/liveness iterate it; the
        # recurrent-divmod build repeats the reused body's block objects here).
        # ``.phys_blocks`` is the DISTINCT physical set (what the model STORES); for
        # a non-recurrent build the two are the same list.
        self.blocks = blocks
        self.phys_blocks = phys_blocks if phys_blocks is not None else blocks


def _build_stream_intermediate(pfc, block_specs, dim, n_heads, vocab, max_seq, L):
    """Build the memory-cheap intermediate (sparse attention, ragged dense FFN).

    Only the 3 baked attention blocks (frame-ingest block 0, mem-cam, stack-pop)
    get a real dense ``Attn``; every other block gets a param-free ``_ZeroAttn``.
    Byte-identical forward to the full dense intermediate the non-streaming path
    builds (the un-baked blocks ARE all-zero attention there too).
    """
    import torch.nn as nn
    from .blogspec_model import Attn as _Attn, FFN as _FFN

    n_blocks = len(block_specs)
    mem_block = pfc._find(block_specs, "mem-cam")
    stk_block = pfc._find(block_specs, "stack-pop-cam")
    baked = {0, mem_block, stk_block}

    # embedding + LM head (dense; embed is a gather, head is only used by .forward
    # which the liveness pass does not call for the block stack — but keep it real).
    embed = torch.zeros(vocab, dim)
    lm_head = torch.zeros(vocab, dim)
    lm_bias = torch.zeros(vocab)

    blocks: List[_StreamBlock] = []
    with torch.no_grad():
        # Bake the embedding into a tiny throwaway holder so we can reuse
        # ``_bake_pure_embedding`` (it writes model.embed).
        class _EmbHolder:
            pass
        eh = _EmbHolder(); eh.dim = dim; eh.embed = embed
        pfc._bake_pure_embedding(eh, L)

        # ONE physical _StreamBlock per UNIQUE spec (block_specs is already the
        # unique set for the recurrent build).
        phys: List[_StreamBlock] = []
        for bi, (name, spec) in enumerate(block_specs):
            if bi in baked:
                at = _Attn(dim, n_heads, max_seq)          # dense (zeroed) attn
                for p in (at.W_q, at.W_k, at.W_v, at.W_o):
                    p.zero_()
            else:
                at = _ZeroAttn(dim, n_heads, max_seq)      # param-free identity
            hid = max(1, spec["W_up"].shape[0])
            ffn = _FFN(dim, hid)
            _swap_ffn(ffn, spec["W_up"], spec["b_up"], spec["W_gate"],
                      spec["b_gate"], spec["W_down"], spec["b_down"])
            phys.append(_StreamBlock(at, ffn))

        reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP,
                     "STACK0": L.STACK0}
        pfc.bake_frame_ingest(phys[0].attn, L, reg_bases)
        pfc._bake_pf_memory_head(phys[mem_block].attn, L, head=pfc.N_ROLES)
        pfc._bake_stack_pop_head(phys[stk_block].attn, L, head=pfc.N_ROLES + 1)
        pfc._bake_lev_ret_head(phys[stk_block].attn, L, head=pfc.N_ROLES + 2)

        # APPLICATION order: for the recurrent-divmod build ``L._apply_order`` maps
        # the FULL application sequence onto the unique physical blocks (the reused
        # iteration body indices REPEAT), so the intermediate forward (which the
        # liveness/empirical passes RUN) applies the same 262-long divmod sequence
        # the production model does — even though only 115 blocks are stored.
        apply_order = getattr(L, "_apply_order", None)
        if apply_order is None:
            blocks = phys
        else:
            blocks = [phys[i] for i in apply_order]

    return _StreamIntermediate(dim, vocab, max_seq, embed, lm_head, lm_bias,
                               blocks, phys_blocks=phys)


def build_compact_sparse_streaming(code_size: int = 48,
                                   compute_mode: str = "dense_kernel",
                                   density_thresh: float = 0.25,
                                   min_numel: int = 4096,
                                   probe_programs=None,
                                   recurrent_divmod: bool = False):
    """Build the c4_min pure-forward VM directly as a streaming SPARSE model.

    Builds the SINGLE full-op-set interpreter (every opcode present — no
    op-subset toggle).  Returns ``(SparseTransformer, L, stats)`` — the SAME
    public triple as ``build_compact_pure_forward_model`` (whose compact dense
    model you would then wrap with ``SparseTransformer``), but built
    block-at-a-time so the PEAK live memory is ONE block's dense weights (a few
    MB..~1 GB), never the full dense attention + padded FFN.  This is the
    memory-safe way to materialise / verify the full-op model.  Byte-identical
    (L∞=0 in ``dense_kernel`` mode) to
    ``SparseTransformer(build_compact_pure_forward_model(...)[0])``.

    ``compute_mode`` / ``density_thresh`` / ``min_numel`` match
    ``sparse_forward.SparseTransformer`` (defaults: bit-identical dense_kernel).
    """
    import inspect
    from . import nibble_pure_forward_complete as pfc
    from .sparse_forward import SparseTransformer, SparseBlock

    # ---- capture block_specs + dims (same trick as the dense builder) ----
    captured: Dict[str, object] = {}
    orig_T = pfc.Transformer

    class _CaptureT:
        def __init__(self, dim, n_heads, hidden, n_blocks, vocab, max_seq_len):
            frame = inspect.currentframe().f_back
            captured["block_specs"] = frame.f_locals["block_specs"]
            # ``L._apply_order`` was set by the builder just before it constructs
            # the Transformer — capture it so the streaming path reuses the exact
            # recurrent application sequence (the fresh _rebuild_layout below cannot
            # regenerate it without re-running the block emission).
            captured["apply_order"] = getattr(frame.f_locals.get("L"),
                                              "_apply_order", None)
            captured["dim"] = dim
            captured["n_heads"] = n_heads
            captured["vocab"] = vocab
            captured["max_seq_len"] = max_seq_len
            raise _CaptureDone

    class _CaptureDone(Exception):
        pass

    pfc.Transformer = _CaptureT
    try:
        pfc.build_pure_forward_complete_model(
            code_size=code_size, recurrent_divmod=recurrent_divmod)
    except _CaptureDone:
        pass
    finally:
        pfc.Transformer = orig_T

    block_specs = captured["block_specs"]
    dim = captured["dim"]
    n_heads = captured["n_heads"]
    vocab = captured["vocab"]
    max_seq = captured["max_seq_len"]
    n_blocks = len(block_specs)
    L = _rebuild_layout(pfc, code_size, n_heads,
                        recurrent_divmod=recurrent_divmod)
    L._apply_order = captured["apply_order"]     # None for non-recurrent builds
    hidden_per_block = [max(1, sp["W_up"].shape[0]) for _, sp in block_specs]
    L._block_names = [n for n, _ in block_specs]

    # ---- memory-cheap intermediate for the liveness passes (sparse attn) ----
    model = _build_stream_intermediate(pfc, block_specs, dim, n_heads, vocab,
                                       max_seq, L)

    # ---- Fix #1: liveness -> colouring (identical to compact_model) ----
    ns_dims = never_share_dims_from_layout(L)
    liveness = compute_liveness(model, ns_dims)
    if probe_programs is None:
        probe_programs = _default_probe_programs(L)
    if probe_programs:
        first_nz, last_nz = empirical_value_liveness(model, L, probe_programs)
        liveness = refine_liveness_empirically(liveness, first_nz, last_nz)
    new_slot, new_dim = color_dims(liveness)

    hd_floor = _required_head_dim(model, n_heads)
    dim_floor = hd_floor * n_heads
    if new_dim < dim_floor:
        new_dim = dim_floor
    if new_dim % n_heads != 0:
        new_dim += n_heads - (new_dim % n_heads)

    # ---- Fix #2: per-block live hidden (mask before we discard the specs) ----
    # Masks are per PHYSICAL block (one per unique spec); the recurrent-divmod
    # apply-order reuses the same physical block many times, so we compute the mask
    # once per stored block, not once per application.
    hidden_masks = [live_ffn_units(blk.ffn) for blk in model.phys_blocks]
    hidden_after = [max(1, int(m.sum())) for m in hidden_masks]

    # Re-point the layout ``L`` onto the packed dim (the driver reads the register
    # bands at their NEW physical slots).  Identical to the call ``compact_model``
    # makes after its block loop — without it the driver would read the register
    # decode bands at their ORIGINAL positions while the weights write the packed
    # positions (the 240 = 0xF0 nibble-corruption signature the harness caught).
    remap_layout(L, new_slot, new_dim)

    # ---- STREAM: build + remap + sparsify ONE block at a time ----
    # Build the SparseTransformer shell without a source model, then fill blocks.
    sparse = SparseTransformer.__new__(SparseTransformer)
    sparse.dim = new_dim
    sparse.vocab = vocab
    sparse.max_seq_len = max_seq
    sparse.compute_mode = compute_mode
    sparse.embed = _remap_in_cols(model.embed, new_slot, new_dim).contiguous()
    sparse.lm_head = _remap_in_cols(model.lm_head, new_slot, new_dim).contiguous()
    sparse.lm_bias = model.lm_bias.detach().clone()
    log: Dict[str, int] = {}
    sparse.blocks = []
    sparse._log = log

    from .blogspec_model import Attn as _Attn, FFN as _FFN
    import torch.nn as nn
    n_phys = len(model.phys_blocks)                # DISTINCT blocks the model stores
    phys_sparse: List = [None] * n_phys            # materialised once per unique block
    for bi in range(n_phys):
        src = model.phys_blocks[bi]
        with torch.no_grad():
            # -- one dense compact block (attention + this block's ragged FFN) --
            cat = _Attn(new_dim, n_heads, max_seq)
            for p in (cat.W_q, cat.W_k, cat.W_v, cat.W_o):
                p.zero_()
            if not getattr(src.attn, "is_zero", False):
                _remap_attn(src.attn, cat, new_slot, new_dim, n_heads)
                cat.alibi_slopes.copy_(src.attn.alibi_slopes)
                cat.scale = src.attn.scale
            else:
                # all-zero attention: nothing to remap; keep zeroed weights.
                cat.scale = src.attn.scale
            # FFN: trim to live hidden, then re-index dims.
            ff = src.ffn
            mask = hidden_masks[bi]
            if int(mask.sum()) == 0:
                mask = torch.zeros_like(mask); mask[0] = True
            keep = torch.nonzero(mask, as_tuple=False).flatten()
            Wup = _remap_in_cols(ff.W_up[keep], new_slot, new_dim)
            Wgate = _remap_in_cols(ff.W_gate[keep], new_slot, new_dim)
            Wdown = _remap_out_rows(ff.W_down[:, keep], new_slot, new_dim)
            cff = _FFN(new_dim, max(1, keep.numel()))
            _swap_ffn(cff, Wup, ff.b_up[keep], Wgate, ff.b_gate[keep],
                      Wdown, _remap_vec(ff.b_down, new_slot, new_dim))
        # -- wrap this ONE dense block sparse, then free the dense block --
        dense_block = _StreamBlock(cat, cff)
        phys_sparse[bi] = SparseBlock(dense_block, density_thresh, min_numel, log,
                                      compute_mode)
        # free the intermediate source block + the transient dense compact block.
        model.phys_blocks[bi] = None
        del src, cat, cff, dense_block

    # ---- APPLICATION order: ``sparse.blocks`` is the full apply sequence, reusing
    # the DISTINCT physical SparseBlock objects (the recurrent-divmod body repeats).
    # ``forward`` iterates ``sparse.blocks``, so applying the reused block object N
    # times IS the recurrence; storage (stats/save) dedups by object identity.
    apply_order = getattr(L, "_apply_order", None)
    if apply_order is None:
        sparse.blocks = phys_sparse
    else:
        sparse.blocks = [phys_sparse[i] for i in apply_order]
    sparse._phys_blocks = phys_sparse              # the distinct stored blocks
    model.blocks = None                            # drop the apply-order reference list

    # ---- stats (match build_compact_pure_forward_model) ----
    # ``n_blocks`` here is the number of APPLICATIONS (the full sequence); the model
    # STORES ``n_phys`` distinct blocks.  Report both.
    global_hidden = max(hidden_per_block)
    n_apply = len(sparse.blocks)
    stats = CompactionStats(
        dim_before=dim, dim_after=new_dim,
        hidden_max_before=global_hidden,
        hidden_per_block_after=hidden_after,
        n_blocks=n_phys, n_heads=n_heads, vocab=vocab,
        dense_params_before=_dense_param_count(dim, global_hidden, n_phys, vocab),
        dense_params_after=_dense_param_count_ragged(new_dim, hidden_after,
                                                     n_phys, vocab),
        nonzero_params=int(sparse.stats().total_nnz),
        never_share_count=len(ns_dims),
        shared_slots_saved=len(ns_dims) + sum(1 for l in liveness
                                              if not l.never_share) - new_dim,
    )
    stats.n_apply = n_apply                         # applications (>= n_blocks for recurrent)
    return sparse, L, stats


# ===========================================================================
# SAVE / RELOAD a streamed sparse model (so the measure sweep reloads the divmod
# artifact instead of rebuilding it — no 79 GB peak, no 30 s bake).  The on-disk
# form stores each weight as a COO sparse tensor (a few MB) + the attn scalars +
# the FFN biases + embed/lm_head/lm_bias + the (remapped) layout ``L``.  The
# loader rebuilds a small dense-block model from the COO tensors and re-wraps it
# with :class:`SparseTransformer` — byte-identical to the streamed model.
# ===========================================================================
def _sparse_weight_to_dense(w) -> torch.Tensor:
    """Densify a :class:`sparse_forward.SparseWeight` (CSR or dense)."""
    if not w.is_sparse:
        return w.dense.detach().clone()
    return w.csr.to_dense()


def save_sparse_transformer(sparse, L, stats, path: str) -> None:
    """Serialise a streamed ``SparseTransformer`` + its layout to ``path``.

    Weights are stored COO (small); the loader materialises them back and
    re-wraps :class:`SparseTransformer` (so the reloaded model is byte-identical
    to the streamed one).  ``L`` is pickled whole (it is a small dataclass of int
    band offsets)."""
    # Store the DISTINCT physical blocks once (deduped by object identity) + an
    # ``apply_order`` index list mapping the full application sequence onto them.
    # A recurrent-divmod model reuses the same SparseBlock object across the 8
    # iterations, so saving ``sparse.blocks`` verbatim would write the reused body
    # 8x; instead we save each unique block once and rebuild the sequence on load.
    phys = getattr(sparse, "_phys_blocks", None)
    if phys is None:
        # dedup by identity, preserving first-seen order.
        phys = []
        seen: Dict[int, int] = {}
        for b in sparse.blocks:
            if id(b) not in seen:
                seen[id(b)] = len(phys)
                phys.append(b)
    id_to_idx = {id(b): i for i, b in enumerate(phys)}
    apply_order = [id_to_idx[id(b)] for b in sparse.blocks]
    blocks = []
    for b in phys:
        at, ff = b.attn, b.ffn
        blk = {
            "n_heads": at.n_heads, "head_dim": at.head_dim, "scale": at.scale,
            "max_seq_len": at.max_seq_len,
            "alibi_slopes": at.alibi_slopes.detach().cpu().clone(),
            "W_q": _sparse_weight_to_dense(at.W_q).to_sparse().coalesce(),
            "W_k": _sparse_weight_to_dense(at.W_k).to_sparse().coalesce(),
            "W_v": _sparse_weight_to_dense(at.W_v).to_sparse().coalesce(),
            "W_o": _sparse_weight_to_dense(at.W_o).to_sparse().coalesce(),
            "W_up": _sparse_weight_to_dense(ff.W_up).to_sparse().coalesce(),
            "W_gate": _sparse_weight_to_dense(ff.W_gate).to_sparse().coalesce(),
            "W_down": _sparse_weight_to_dense(ff.W_down).to_sparse().coalesce(),
            "b_up": ff.b_up.detach().cpu().clone(),
            "b_gate": ff.b_gate.detach().cpu().clone(),
            "b_down": ff.b_down.detach().cpu().clone(),
        }
        blocks.append(blk)
    payload = {
        "format": "c4min_sparse_stream_v1",
        "dim": sparse.dim, "vocab": sparse.vocab,
        "max_seq_len": sparse.max_seq_len, "compute_mode": sparse.compute_mode,
        "embed": sparse.embed.detach().cpu().to_sparse().coalesce(),
        "lm_head": sparse.lm_head.detach().cpu().to_sparse().coalesce(),
        "lm_bias": sparse.lm_bias.detach().cpu().clone(),
        "blocks": blocks,
        "apply_order": apply_order,        # full application sequence over ``blocks``
        "layout": L,
        "stats": stats,
    }
    torch.save(payload, path)


class _LoadedBlock:
    """A dense ``.attn`` / ``.ffn`` holder that :class:`SparseTransformer` can
    re-wrap (mirrors ``blogspec_model.Block``'s field shape, no nn.Module)."""

    def __init__(self, attn, ffn):
        self.attn = attn
        self.ffn = ffn


def load_sparse_transformer(path: str, compute_mode: Optional[str] = None):
    """Reload a saved streamed sparse model — returns ``(SparseTransformer, L)``.

    Rebuilds each block's dense weights from the stored COO tensors, wraps them
    in a lightweight holder, and re-wraps the whole model with
    :class:`SparseTransformer` (so the reloaded forward is byte-identical to the
    streamed model).  Peak memory is one block's dense weights at a time (the
    ``SparseTransformer`` ctor sparsifies block-by-block), never the full dense
    model."""
    import torch.nn as nn
    from .blogspec_model import Attn as _Attn, FFN as _FFN
    from .sparse_forward import SparseTransformer

    payload = torch.load(path, weights_only=False)
    assert payload.get("format") == "c4min_sparse_stream_v1", "bad artifact format"
    dim = payload["dim"]
    vocab = payload["vocab"]
    max_seq = payload["max_seq_len"]
    mode = compute_mode or payload["compute_mode"]
    L = payload["layout"]

    # A tiny holder mirroring blogspec Transformer for SparseTransformer's ctor.
    class _Holder:
        pass
    holder = _Holder()
    holder.dim = dim
    holder.vocab = vocab
    holder.max_seq_len = max_seq
    holder.embed = payload["embed"].to_dense()
    holder.lm_head = payload["lm_head"].to_dense()
    holder.lm_bias = payload["lm_bias"]
    holder.blocks = []
    with torch.no_grad():
        for blk in payload["blocks"]:
            n_heads = blk["n_heads"]
            at = _Attn(dim, n_heads, blk["max_seq_len"])
            at.W_q.copy_(blk["W_q"].to_dense())
            at.W_k.copy_(blk["W_k"].to_dense())
            at.W_v.copy_(blk["W_v"].to_dense())
            at.W_o.copy_(blk["W_o"].to_dense())
            at.scale = blk["scale"]
            at.alibi_slopes.copy_(blk["alibi_slopes"])
            hidden = blk["b_up"].shape[0]
            ff = _FFN(dim, hidden)
            ff.W_up.copy_(blk["W_up"].to_dense())
            ff.W_gate.copy_(blk["W_gate"].to_dense())
            ff.W_down.copy_(blk["W_down"].to_dense())
            ff.b_up.copy_(blk["b_up"])
            ff.b_gate.copy_(blk["b_gate"])
            ff.b_down.copy_(blk["b_down"])
            holder.blocks.append(_LoadedBlock(at, ff))
    sparse = SparseTransformer(holder, compute_mode=mode)
    # Rebuild the APPLICATION sequence: ``SparseTransformer.__init__`` wrapped the
    # DISTINCT physical blocks 1:1; re-point ``sparse.blocks`` through the saved
    # apply-order so a recurrent-divmod body's block object is reused across the 8
    # iterations (byte-identical to the streamed model).  Absent (older artifacts)
    # -> identity (already 1:1).
    apply_order = payload.get("apply_order")
    if apply_order is not None:
        phys = sparse.blocks
        sparse._phys_blocks = phys
        sparse.blocks = [phys[i] for i in apply_order]
    else:
        sparse._phys_blocks = sparse.blocks
    return sparse, L


def _rebuild_layout(pfc, code_size, n_heads, recurrent_divmod=False):
    """Reconstruct the ``PureForwardCompleteLayout`` exactly as the builder does.

    Mirrors ``build_pure_forward_complete_model``: the full op set is ALWAYS
    present, so the bitwise band is always extended (no op-subset toggle)."""
    from . import nibble_pure_forward_complete as _pfc
    L = _pfc.PureForwardCompleteLayout(code_size, n_heads=n_heads)
    _pfc.A.extend_layout_for_alu32(L, recurrent_divmod=recurrent_divmod)
    from . import nibble_bitwise as _bw
    _bw.extend_layout_for_bitwise(L)
    while L._off % n_heads != 0:
        L._scalar(f"_bwpad{L._off}")
    L.D = L._off
    min_dim = n_heads * _pfc.MEM_HEAD_CHANNELS
    if L.D < min_dim:
        target = -(-min_dim // n_heads) * n_heads
        while L._off < target:
            L._scalar(f"_hdpad{L._off}")
        L.D = L._off
    _pfc.A._ONE = L.ONE
    return L


# ---------------------------------------------------------------------------
# Fix #3: sparse tensor storage.
# ---------------------------------------------------------------------------
def sparse_state_dict(model, min_dense_to_sparsify: int = 1_000_000
                      ) -> Dict[str, object]:
    """Return a storage-only state dict; large-and-sparse tensors -> COO.

    Any parameter with more than ``min_dense_to_sparsify`` elements AND a
    density below 25% is stored as a ``torch.sparse_coo_tensor`` (indices +
    values), which for the c4_min weights (mostly the giant lookup-table FFN
    block) is far smaller on disk / in VRAM than the dense zeros.  Everything
    else stays dense.  A companion :func:`load_sparse_state_dict` materialises
    the dense tensors back for the compute path (dense-small wins compute).
    """
    out: Dict[str, object] = {}
    for name, p in model.state_dict().items():
        t = p
        if t.numel() > min_dense_to_sparsify:
            density = float((t != 0).sum()) / t.numel()
            if density < 0.25:
                out[name] = ("coo", t.to_sparse().coalesce(), tuple(t.shape))
                continue
        out[name] = ("dense", t, tuple(t.shape))
    return out


def sparse_storage_bytes(sd: Dict[str, object]) -> int:
    """Bytes to STORE a :func:`sparse_state_dict` (COO = idx(int64) + val(fp32))."""
    total = 0
    for _name, entry in sd.items():
        kind, t, _shape = entry
        if kind == "coo":
            nnz = t._nnz()
            ndim = t.sparse_dim()
            total += nnz * ndim * 8      # indices int64
            total += nnz * 4             # values fp32
        else:
            total += t.numel() * 4
    return total


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _required_head_dim(model, n_heads: int) -> int:
    """Largest per-head LOCAL channel that any block's attention touches, +1.

    Q/K/V rows and W_o cols are head-partitioned in blocks of ``old_hd``. A head
    only uses its first few local channels (the CAM channels); the packed
    head_dim must be >= the max nonzero local-channel index + 1 so no live
    channel is truncated when dim shrinks.
    """
    old_dim = model.dim
    old_hd = old_dim // n_heads
    max_local = 0
    for blk in model.blocks:
        at = blk.attn
        if getattr(at, "is_zero", False):
            continue                      # param-free zero-attention: no channels
        for name in ("W_q", "W_k", "W_v"):
            w = getattr(at, name)                    # rows head-partitioned
            rows = torch.nonzero((w != 0).any(dim=1), as_tuple=False).flatten()
            for r in rows.tolist():
                max_local = max(max_local, r % old_hd)
        wo = at.W_o                                   # cols head-partitioned
        cols = torch.nonzero((wo != 0).any(dim=0), as_tuple=False).flatten()
        for cix in cols.tolist():
            max_local = max(max_local, cix % old_hd)
    return max_local + 1


def max_hidden_before(model) -> int:
    return max(blk.ffn.W_up.shape[0] for blk in model.blocks)


def _dense_param_count(dim: int, hidden: int, n_blocks: int, vocab: int) -> int:
    per_block_attn = 4 * dim * dim
    per_block_ffn = hidden * dim * 2 + hidden * 2 + dim * hidden + dim
    emb = vocab * dim * 2 + vocab
    return n_blocks * (per_block_attn + per_block_ffn) + emb


def _dense_param_count_ragged(dim: int, hidden_per_block: List[int],
                              n_blocks: int, vocab: int) -> int:
    total = vocab * dim * 2 + vocab
    for h in hidden_per_block:
        total += 4 * dim * dim
        total += h * dim * 2 + h * 2 + dim * h + dim
    return total


def _count_nonzero(model) -> int:
    return int(sum((p != 0).sum().item() for p in model.parameters()))
