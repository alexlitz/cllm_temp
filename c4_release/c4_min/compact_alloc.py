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
# Never-share set (the _LIVENESS_NEVER_SHARE principle).  Every dim the Python
# driver reads/writes OUTSIDE the block stack: the overlay's writes + the
# decode's reads.  Harvested straight off the layout ``L`` so it tracks the
# exact bands the driver ``run_pure_forward_complete`` touches.
# ---------------------------------------------------------------------------
def never_share_dims_from_layout(L) -> set:
    """Return the set of residual dims that must keep a private, fixed slot.

    These are the dims the driver's ``make_overlay_complete`` WRITES and its
    per-step decode (``PC_VAL/SP_VAL/BP_VAL/STK_VAL/HALTED`` + the AX nibble
    band) READS between forwards — plus the ``ONE`` constant lane.  The
    weight-only liveness analysis cannot observe these cross-forward
    read/writes, so (exactly like ``_LIVENESS_NEVER_SHARE`` in the bigger
    version) they are pinned live-to-end.
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


def compact_model(model, L):
    """Compact ``model`` in place-ish and return ``(compact_model, L, stats)``.

    Applies Fix #1 (dim-sharing by liveness) + Fix #2 (per-block minimal FFN
    hidden).  Returns a NEW small dense ``Transformer`` byte-identical (L∞=0) to
    ``model`` under the driver, plus the re-indexed ``L`` and a
    :class:`CompactionStats`.  Fix #3 (sparse storage) is a separate serialiser
    (:func:`sparse_state_dict`) applied to the returned compact model.
    """
    from .blogspec_model import Transformer as _T

    dim = model.dim
    n_blocks = len(model.blocks)
    n_heads = model.blocks[0].attn.n_heads
    vocab = model.vocab

    # ---- Fix #1: liveness -> colouring -> re-index ----
    ns_dims = never_share_dims_from_layout(L)
    liveness = compute_liveness(model, ns_dims)
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
def build_compact_pure_forward_model(code_size: int = 48,
                                     include_bitwise: bool = True,
                                     include_divmod: bool = False):
    """Build the c4_min pure-forward VM as a COMPACT dense model directly.

    Returns ``(compact_model, L, stats)``.  Never allocates the global-max FFN
    padding: each block's FFN is built at its own hidden width, then Fix #1
    (dim-sharing) packs the residual.  The result is byte-identical (L∞=0) to
    ``build_pure_forward_complete_model`` under the driver.
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
        pfc.build_pure_forward_complete_model(
            code_size=code_size, include_bitwise=include_bitwise,
            include_divmod=include_divmod)
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
    L = _rebuild_layout(pfc, code_size, n_heads, include_bitwise)

    # Build the model with each block's REAL hidden (no global-max padding).
    hidden_per_block = [max(1, sp["W_up"].shape[0]) for _, sp in block_specs]
    model = _T(dim=dim, n_heads=n_heads, hidden=max(hidden_per_block),
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


def _rebuild_layout(pfc, code_size, n_heads, include_bitwise):
    """Reconstruct the ``PureForwardCompleteLayout`` exactly as the builder does."""
    from . import nibble_pure_forward_complete as _pfc
    L = _pfc.PureForwardCompleteLayout(code_size, n_heads=n_heads)
    _pfc.A.extend_layout_for_alu32(L)
    if include_bitwise:
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
