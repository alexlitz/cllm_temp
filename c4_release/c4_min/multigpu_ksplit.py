"""MULTI-GPU K-SPLIT (C4_MULTIGPU_KSPLIT, DEFAULT OFF).

The faithful single-dispatch verify (``faithful_single_dispatch``) is a per-row INDEPENDENT
MAP: block-0 ingest -> mega dead-FFN chain -> live CAM blocks -> decode + the in-graph
``W_q(x)`` sign-decode of the model's own query address.  There is ZERO cross-row attention
(the live CAM heads are draft-resolved direct-gathers, the local heads are zeroed), so the
whole 358,058-step frame's rows split LINEARLY across devices: device d verifies the step
range ``[lo_d, hi_d)`` and produces that shard's decoded registers + verdict.

This module splits the frame's K rows across ``cuda:0`` and ``cuda:1``:

  * Each device holds its OWN copy of the composed model (streamed build, ~1 GB) + its OWN
    captured single-dispatch step graph, and builds the schedule TABLES for its shard ONLY
    (the per-step tables are keyed by ABSOLUTE step index, so a shard is a pure row-slice of
    the full-frame tables -- no frame-counter re-derivation).

  * The heavy value-verify PRECOMPUTE + the CAM latest-write-wins RESOLUTION are pure
    functions of the IMMUTABLE draft (device-independent numpy).  We resolve the FULL frame's
    numpy arrays ONCE (``qtok`` / ``ingest`` nibbles / ``cam_sparse`` vals) and each shard
    SLICES ``[lo_d:hi_d]`` before the device-specific block-0 fold + W_o-delta GEMM + H2D --
    so the CPU resolution is shared, only the tiny per-shard GPU work is per-device.

  * Assembly: the per-shard decoded registers (pc/sp/bp/ax) are concatenated back into one
    frame; the faithful verdict is the FIRST-DIVERGENCE reduction ACROSS shards (each shard
    reports its lowest bad LOCAL step, offset to a GLOBAL step; the frame's verdict is the
    minimum global bad step -- so the split does NOT weaken verification: a wrong draft is
    rejected at the same first-divergence step it would be on a single GPU).

Cross-device comms are minimal: each device produces its shard's outputs on-device; only the
tiny scalar verdicts + the small decoded-register lanes cross to the host for the concat +
reduction.  The pipeline (background-thread overlapping build) runs PER-DEVICE.

DEFAULT OFF -> the single-GPU faithful/fast single-dispatch (golden 069cc32f unchanged).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def multigpu_ksplit_enabled() -> bool:
    """``C4_MULTIGPU_KSPLIT`` (DEFAULT OFF): split the faithful single-dispatch frame's K
    rows across two devices.  OFF -> the single-GPU path (byte-identical golden)."""
    return os.environ.get("C4_MULTIGPU_KSPLIT", "0") not in ("0", "", "false", "False")


# ===========================================================================
# 1. DEVICE-INDEPENDENT full-frame numpy resolution (shared across shards).
# ===========================================================================
def _resolve_full_frame_numpy(draft, code, live_blocks):
    """The device-INDEPENDENT numpy resolution shared by every shard: the query-row token
    gather, the ingest-frame nibbles, and the CAM-sparse latest-write-wins ``vals``.  A PURE
    function of the immutable draft (identical for every device + every frame in a continuous
    render), so resolved ONCE and sliced per shard.  Cached on the draft."""
    from . import precomputed_schedule as PS
    cached = getattr(draft, "_ksplit_full_np", None)
    if cached is not None:
        return cached
    toks_np = PS._draft_tokens_np(draft)
    n = draft.step_count
    ws_np = np.asarray(draft.win_starts[:n], dtype=np.int64)
    qtok = toks_np[ws_np].astype(np.int64)                          # [n]
    nib_lo, nib_hi = _resolved_frame_nibs_np(draft, toks_np)         # [n, N_ROLES] each
    cam_sparse = PS._build_cam_sparse_gpu(draft, code, live_blocks, "cpu")
    full = dict(qtok=qtok, nib_lo=nib_lo, nib_hi=nib_hi, cam_sparse=cam_sparse)
    try:
        draft._ksplit_full_np = full
    except Exception:
        pass
    return full


def _resolved_frame_nibs_np(draft, toks_np):
    """Numpy (device-free) form of ``precomputed_schedule._resolved_frame_nibs_vec`` — the
    ingest nibble decode.  Byte-identical nibbles (same fstart / REG_PC guard / clamp)."""
    from .direct_local_cam import _ROLE_TO_LOCAL
    from . import precomputed_schedule as PS
    V = PS.V
    FL = V.FRAME_LEN
    NR = len(_ROLE_TO_LOCAL)
    n = draft.step_count
    local_of_role = np.asarray([_ROLE_TO_LOCAL[r] for r in range(NR)], dtype=np.int64)
    T = toks_np.shape[0]
    if n == 0:
        z = np.zeros((0, NR), dtype=np.float32)
        return z, z.copy()
    pos = np.asarray(draft.win_starts[:n], dtype=np.int64)
    fstart = pos - (FL - 1)
    valid = fstart >= 0
    fs_c = np.clip(fstart, 0, None)
    head_tok = toks_np[np.clip(fs_c, 0, T - 1)]
    valid = valid & (head_tok == V.REG_PC)
    gather_idx = fs_c[:, None] + local_of_role[None, :]
    gather_idx = np.clip(gather_idx, 0, T - 1)
    B = toks_np[gather_idx]
    B = np.where((B >= 0) & (B <= 255), B, 0)
    B = np.where(valid[:, None], B, 0)
    nib_lo = (B & 0xF).astype(np.float32)
    nib_hi = ((B >> 4) & 0xF).astype(np.float32)
    return nib_lo, nib_hi


# ===========================================================================
# 2. Per-shard schedule TABLES (a row-slice of the full-frame tables, on the shard device).
# ===========================================================================
def build_schedule_tables_shard(model, L, code, draft, device, sg, lo, hi,
                                 mask: int = 0xFFFFFFFF):
    """Build the ``PrecomputedSchedule`` for the row range ``[lo, hi)`` on ``device``.

    Reuses the device-INDEPENDENT full-frame numpy resolution (shared across shards) and
    slices ``[lo:hi]``, then does the tiny per-shard device work: the block-0 W_o0(ingest)
    fold into ``h0``, and each live block's compact ``W_o(cam_out)`` delta GEMM.  Byte-
    identical to the single-GPU schedule restricted to those rows (same nibbles, same
    latest-write-wins values, same fp W_o-delta -- the shard's out_dims covers all its
    nonzero cols, always-zero cols add +0.0)."""
    from . import precomputed_schedule as PS
    dev = torch.device(device)
    n_sh = hi - lo
    live_blocks = list(sg.live_blocks.values())
    onchip = sg.onchip
    full = _resolve_full_frame_numpy(draft, code, live_blocks)

    # --- h0 table (query embed) for the shard rows, on this device ---
    from .pf_speculative import build_code_vec
    qtok_sh = torch.from_numpy(full["qtok"][lo:hi]).to(dev)
    h0 = model.embed[qtok_sh].clone()                              # [n_sh, D]
    code_idx, code_vals = build_code_vec(code, L, model.embed.shape[1], dev,
                                         dtype=model.embed.dtype)
    h0[:, code_idx] = code_vals.to(h0.dtype)
    role_dims = torch.arange(PS.N_ROLES, device=dev, dtype=torch.long) + L.ROLE
    h0[:, role_dims] = 1.0
    h0_table = h0

    # --- ingest table for the shard rows ---
    from .direct_local_cam import ingest_head_map
    attn0 = model.blocks[0].attn
    head_map = ingest_head_map(attn0)
    ing_heads = sorted(head_map)
    nib_lo = torch.from_numpy(full["nib_lo"][lo:hi]).to(dev)
    nib_hi = torch.from_numpy(full["nib_hi"][lo:hi]).to(dev)
    H0, HD0 = attn0.n_heads, attn0.head_dim
    ing_table = torch.zeros(n_sh, H0, HD0, device=dev)
    heads_t = torch.tensor(ing_heads, device=dev, dtype=torch.long)
    ing_table[:, heads_t, 0] = nib_lo[:, heads_t]
    ing_table[:, heads_t, 1] = nib_hi[:, heads_t]

    # --- decode targets (per-step) for the shard rows ---
    frames = draft.frames
    _pc = np.empty(n_sh, dtype=np.int64); _ax = np.empty(n_sh, dtype=np.int64)
    _sp = np.empty(n_sh, dtype=np.int64); _bp = np.empty(n_sh, dtype=np.int64)
    _hl = np.empty(n_sh, dtype=np.bool_); _fl = np.empty(n_sh, dtype=np.bool_)
    for j, s in enumerate(range(lo, hi)):
        f = frames[s]
        _pc[j] = f["pc"]; _ax[j] = f["ax"] & mask
        _sp[j] = f["sp"] & 0xFFFFFFFF; _bp[j] = f["bp"] & 0xFFFFFFFF
        _hl[j] = f.get("is_halt") or False; _fl[j] = f.get("is_file") or False
    want_pc = torch.from_numpy(_pc).to(dev); want_ax = torch.from_numpy(_ax).to(dev)
    want_sp = torch.from_numpy(_sp).to(dev); want_bp = torch.from_numpy(_bp).to(dev)
    is_halt = torch.from_numpy(_hl).to(dev); is_file = torch.from_numpy(_fl).to(dev)

    # --- the on-chip block-0 fold + per-live-block W_o delta over the shard rows ---
    D = model.dim
    h0_folded = None; cam_delta_tables = None; live_out_dims = None
    cam_sparse = full["cam_sparse"]
    if onchip:
        Wo0 = PS._wo_dense(model, 0, dev)
        ing_flat = ing_table.reshape(n_sh, D)
        h0_table.addmm_(ing_flat, Wo0.transpose(0, 1))
        h0_folded = h0_table
        del ing_flat
        cam_delta_tables = {}; live_out_dims = {}
        for lb in live_blocks:
            b = lb.block_idx
            Wo = PS._wo_dense(model, b, dev)
            vals_np, cols_np, _HDb = cam_sparse[b]
            vals = torch.from_numpy(np.ascontiguousarray(vals_np[lo:hi])).to(dev)  # [n_sh,nc]
            cols = torch.from_numpy(cols_np).to(dev)
            Wo_sub = Wo.index_select(1, cols)                        # [D, n_active]
            out_dims = (Wo_sub.abs().sum(1) > 0).nonzero(as_tuple=False).flatten()
            if out_dims.numel() == 0:
                out_dims = torch.zeros(1, dtype=torch.long, device=dev)
            delta = (vals @ Wo_sub.index_select(0, out_dims).transpose(0, 1)).contiguous()
            cam_delta_tables[b] = delta; live_out_dims[b] = out_dims
            del vals, Wo_sub
        ing_table = None
    sched = PS.PrecomputedSchedule(
        n_steps=n_sh, h0_table=h0_table, ing_table=ing_table, cam_tables={},
        want_pc=want_pc, want_sp=want_sp, want_bp=want_bp, want_ax=want_ax,
        is_halt=is_halt, is_file=is_file, onchip=onchip, h0_folded=h0_folded,
        cam_delta_tables=cam_delta_tables)
    return sched, live_out_dims


# ===========================================================================
# 3. A per-device shard: its own model + composed stack + step graph.
# ===========================================================================
class _DeviceShard:
    """One GPU's shard: a composed model copy on ``device``, the single-dispatch step graph,
    and the row range ``[lo, hi)`` it verifies.  ``rebuild_sched`` produces the shard
    schedule; ``dispatch`` replays the graph over it and returns decoded registers (+ model
    query addresses if faithful)."""

    def __init__(self, model, L, code, draft, device, lo, hi, chunk, faithful, mask):
        from . import precomputed_schedule as PS
        from .fused_megablock import install_fused_megablock, MegaBlockRegion
        from .direct_cam_batched import cam_head_map
        from .pf_speculative import _frozen_skip_cut
        self.device = torch.device(device)
        self.model = model
        self.L = L
        self.code = code
        self.draft = draft
        self.lo = lo; self.hi = hi
        self.n = hi - lo
        self.mask = mask
        self.faithful = faithful
        cut = _frozen_skip_cut(model)
        _lean = os.environ.get("C4_MEGABLOCK_DOOM_LEAN", "0") not in ("0", "", "false", "False")
        # ALL device work (Triton megablock launches, graph capture, GEMMs) must run with
        # THIS shard's device as the ACTIVE CUDA device — Triton launches on the current
        # device, so a cuda:1 shard must set device(cuda:1) or it launches on cuda:0 and
        # sees "cuda:1 tensor from a cuda:0 launch".
        with torch.cuda.device(self.device):
            mega = install_fused_megablock(model, self.device, cut,
                                           L=(L if _lean else None), verbose=False)
            if mega is None:
                mega = MegaBlockRegion(model, self.device, cut)
            chm = cam_head_map(model, L)
            live_blocks = [PS._LiveCamBlock(model, bi, heads)
                           for bi, heads in sorted(chm.items())]
            live_order = sorted(chm.keys())
            self.onchip = PS.onchip_residual_enabled()
            sched, live_out_dims = build_schedule_tables_shard(
                model, L, code, draft, self.device, _ShardSGShim(live_blocks, self.onchip),
                lo, hi, mask=mask)
            self.sched = sched
            self.sg = PS.PrecomputedStepGraph(model, L, self.device, chunk, mega, live_blocks,
                                              live_order, mask, onchip=self.onchip,
                                              live_out_dims=live_out_dims)

    def rebuild_sched(self):
        with torch.cuda.device(self.device):
            self.sched, _ = build_schedule_tables_shard(
                self.model, self.L, self.code, self.draft, self.device, self.sg,
                self.lo, self.hi, mask=self.mask)
        return self.sched

    def dispatch(self, sched=None):
        """Replay the shard graph -> decoded registers (+ model query addrs if faithful).
        Chunked exactly like the single-GPU ``_dispatch`` / ``_dispatch_with_qaddr``."""
        with torch.cuda.device(self.device):
            return self._dispatch_inner(sched)

    def _dispatch_inner(self, sched=None):
        sched = sched if sched is not None else self.sched
        sg = self.sg; dev = self.device; n = self.n
        got_pc = torch.empty(n, dtype=torch.long, device=dev)
        got_sp = torch.empty(n, dtype=torch.long, device=dev)
        got_bp = torch.empty(n, dtype=torch.long, device=dev)
        got_ax = torch.empty(n, dtype=torch.long, device=dev)
        onchip = sched.onchip
        h0s = sched.h0_folded if onchip else sched.h0_table
        per_head = {}
        for clo in range(0, n, sg.chunk):
            chi = min(clo + sg.chunk, n)
            h0 = h0s[clo:chi].unsqueeze(0)
            if onchip:
                delta = {b: t[clo:chi] for b, t in sched.cam_delta_tables.items()}
                pc, sp, bp, ax = sg.replay(h0, None, None, delta=delta, resident=False)
            else:
                ing = sched.ing_table[clo:chi].permute(1, 0, 2).unsqueeze(0)
                cam = {b: t[clo:chi].permute(1, 0, 2).unsqueeze(0)
                       for b, t in sched.cam_tables.items()}
                pc, sp, bp, ax = sg.replay(h0, ing, cam, resident=False)
            got_pc[clo:chi].copy_(pc); got_sp[clo:chi].copy_(sp)
            got_bp[clo:chi].copy_(bp); got_ax[clo:chi].copy_(ax)
            if self.faithful:
                for key, t in sg.last_qaddr(chi - clo).items():
                    per_head.setdefault(key, []).append(
                        (clo, chi, t.detach().to("cpu").numpy().astype("int64")))
        model_addrs = None
        if self.faithful:
            model_addrs = {}
            for (b, hh, knd), chunks in per_head.items():
                fullv = np.zeros(n, dtype=np.int64)
                for (clo, chi, arr) in chunks:
                    fullv[clo:chi] = arr[:chi - clo]
                model_addrs[(b, knd)] = fullv
        return got_pc, got_sp, got_bp, got_ax, model_addrs


class _ShardSGShim:
    """Minimal shim exposing ``live_blocks`` + ``onchip`` so ``build_schedule_tables_shard``
    can build the FIRST shard schedule (to obtain ``live_out_dims``) before the real
    ``PrecomputedStepGraph`` exists."""
    def __init__(self, live_blocks, onchip):
        self.live_blocks = {lb.block_idx: lb for lb in live_blocks}
        self.onchip = onchip


# ===========================================================================
# 3b. Faithful value-verify PRECOMPUTE: full-frame once, sliced per shard.
# ===========================================================================
def slice_precompute(pre, lo, hi):
    """Slice a FULL-frame ``FaithfulPrecompute`` (steps are GLOBAL step indices, with the
    correct global read-frame resolution baked in) down to the shard step range ``[lo, hi)``,
    rebasing each retained step to a LOCAL index ``step - lo`` so it aligns with the shard's
    local ``model_addrs`` (length ``hi - lo``).  Byte-identical: the value/address/routing
    targets for a step are unchanged by the slice (they were resolved against the full
    committed store-log); only the step index is rebased and out-of-shard entries dropped.

    This keeps the GENUINE value re-resolution correct across the split -- a shard-2 read
    still scores against the stores committed in shard 1 (the full store-log is baked into
    ``pre_genuine`` before the slice), so verification is NOT weakened."""
    from .faithful_single_dispatch import FaithfulPrecompute
    n_sh = hi - lo
    steps = {}; draft_addr = {}; draft_val = {}; pre_genuine = {}
    for kind, sarr in pre.steps.items():
        m = (sarr >= lo) & (sarr < hi)
        if not m.any():
            continue
        steps[kind] = (sarr[m] - lo).astype(np.int64)
        draft_addr[kind] = pre.draft_addr[kind][m]
        draft_val[kind] = pre.draft_val[kind][m]
        pre_genuine[kind] = pre.pre_genuine[kind][m]
    cm = (pre.code_steps >= lo) & (pre.code_steps < hi)
    code_steps = (pre.code_steps[cm] - lo).astype(np.int64)
    code_addr = pre.code_addr[cm]
    return FaithfulPrecompute(n=n_sh, mask=pre.mask, steps=steps, draft_addr=draft_addr,
                              draft_val=draft_val, pre_genuine=pre_genuine,
                              code_steps=code_steps, code_addr=code_addr)


# ===========================================================================
# 4. First-divergence reduction ACROSS shards (faithful verdict assembly).
# ===========================================================================
@dataclass
class ShardVerdict:
    ok: bool
    global_first_bad_step: Optional[int]
    kind: Optional[str]
    detail: Optional[dict]
    n_addr_checked: int
    n_value_checked: int
    n_routing_checked: int


def reduce_verdicts(per_shard: List[Tuple[int, "object"]]) -> ShardVerdict:
    """Reduce the per-shard faithful verdicts into ONE frame verdict = the FIRST divergence
    (lowest GLOBAL step) across shards.  ``per_shard`` : list of ``(lo, FaithfulVerdict)`` --
    the shard's row offset + its LOCAL verdict.  The global bad step of a failing shard is
    ``lo + first_bad_step``; the frame's verdict is the minimum such global step (so the split
    does NOT weaken verification -- the SAME first-divergence a single GPU would report)."""
    best = None
    n_addr = n_val = n_rt = 0
    for lo, vd in per_shard:
        n_addr += vd.n_addr_checked; n_val += vd.n_value_checked; n_rt += vd.n_routing_checked
        if not vd.ok:
            gstep = lo + int(vd.first_bad_step)
            if best is None or gstep < best[0]:
                best = (gstep, vd.kind, vd.detail)
    if best is None:
        return ShardVerdict(ok=True, global_first_bad_step=None, kind=None, detail=None,
                            n_addr_checked=n_addr, n_value_checked=n_val,
                            n_routing_checked=n_rt)
    return ShardVerdict(ok=False, global_first_bad_step=best[0], kind=best[1],
                        detail=best[2], n_addr_checked=n_addr, n_value_checked=n_val,
                        n_routing_checked=n_rt)


def split_rows(n: int, n_dev: int = 2) -> List[Tuple[int, int]]:
    """Partition ``[0, n)`` into ``n_dev`` contiguous shards (last shard takes the remainder)."""
    base = n // n_dev
    ranges = []
    lo = 0
    for d in range(n_dev):
        hi = n if d == n_dev - 1 else lo + base
        ranges.append((lo, hi)); lo = hi
    return ranges


__all__ = ["multigpu_ksplit_enabled", "build_schedule_tables_shard", "_DeviceShard",
           "ShardVerdict", "reduce_verdicts", "split_rows", "_resolve_full_frame_numpy",
           "slice_precompute"]
