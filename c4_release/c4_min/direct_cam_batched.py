"""DIRECT-CAM READ in the BATCHED verify_blocks path (C4_DIRECT_CAM_BATCHED).

The batched pure-forward verifier (``pf_speculative.verify_blocks``) runs the
SPARSE whole-VM model over the perfect draft in big-K spans.  Its dominant
per-step cost (measured on doom: ~39% of ms/step) is the GLOBAL address-CAM heads
scoring the WHOLE growing KV cache (the persistent doom heap, 150k-262k rows)
each step: for every query row each global head does a ``[HD]x[K]`` softmax1+ALiBi
dot over the entire cache -- an O(S) score per read, and an O(S) score MATRIX in
VRAM that caps eff_K.

But the perfect draft (``draft_pf_program``) already knows, per read, the EXACT
KV store row the softmax winner is (``nibble_evict_schedule.resolve_load_rows``:
address -> latest superseding store row, latest-write-wins == the softmax1+ALiBi
winner).  So on the SPECULATIVE / verify path the global CAM heads need not score
the cache at all: each read DIRECT-GATHERS the resolved value.

This module wires that into the batched span: it installs, on the CAM blocks, a
forward wrapper that computes each CAM head's attention OUTPUT by DIRECT GATHER
(the resolved value's V vector, reconstructed from the draft) at every QUERY row,
skipping that head's O(S) global score.  EVERY OTHER head (the ingest-local heads
and any non-CAM global head) runs its normal ``windowed_forward`` attention, so
the change is surgical: only the four CAM heads' O(S) global score is removed.

WHY BYTE-EXACT (drives the FULL transition, not just the value band)
-------------------------------------------------------------------
The global CAM head's softmax output at a query row is EXACTLY the winner store
row's V vector (softmax1 weight ~=1 on the exact-address latest-write winner, ~=0
elsewhere, +1 sink -> 0 when the address is unwritten).  The head's ``W_v`` reads
ONLY ``VAL_NIB[j]`` (the store's value nibbles) into the head's value slots, and
``W_o`` maps those into the destination band (AX / STACK0 / LEV_RET).  For the
code-fetch head ``W_v`` reads ``CODE_OPV`` -> ``OP_VAL`` + ``CODE_IMM_NIB_MEM[j]``
-> ``IMM_NIB[j]``.  So the net residual the head adds at a query row is a KNOWN
vector determined solely by the resolved value.  We reconstruct that exact V
vector from the draft and SET the head's ``out`` slot to it -- so ``x + W_o(out)``
is byte-identical to the softmax path.  Because the value bands are what the
DOWNSTREAM FFN blocks read to drive the PC/SP/BP transition (LEV: PC <-
LEV_RET_VAL ; BZ/BNZ read AX ; the opcode decode reads OP_VAL/IMM_NIB), driving
the value band correctly drives the WHOLE transition -- the FFN machinery is
untouched.

The frozen (context) rows' CAM-head output is discarded (only query rows are
decoded; their K/V is still projected + committed so the cache stays consistent
for the local heads / a fallback), so gathering only at query rows is exact.

Composition
-----------
* C4_PF_CFM (code frames): the code-fetch@PC head is a global CAM too; it is
  resolved per step (the code frame at PC is a static store the draft knows).
* Persistent heap + eviction + big-K: the direct gather reads the VALUE from the
  draft (not the KV tensor), so an evicted winner row is irrelevant -- the
  schedule only ever drops rows no read resolves to.  Removing the global score
  removes the O(S) score matrix -> the VRAM pressure that capped eff_K collapses.

Gated: C4_DIRECT_CAM_BATCHED default OFF -> byte-identical to the softmax path
(golden 069cc32f unchanged).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .blogspec_model import softmax1
from .nibble_pure_forward import N_ROLES
from .nibble_pure_forward_complete import (
    PureForwardCompleteLayout, IMM_NIBS, ADDR_BITS, CODE_ADDR_BITS,
    _pf_cfm_enabled, _unify_cam_head_enabled,
)
from .nibble_evict_schedule import resolve_load_rows
from .nibble_pure_forward_cached import _SplitPastKV


def direct_cam_batched_enabled() -> bool:
    """C4_DIRECT_CAM_BATCHED (DEFAULT OFF): direct-index CAM gather in the batched
    verify_blocks span.  OFF -> the vanilla softmax1+ALiBi global-CAM score (byte-
    exact golden path).  ON -> the CAM blocks' global heads direct-gather the draft-
    resolved value per query row (no O(S) score) -> the ~39% global-CAM wall + the
    O(S) score-matrix VRAM pressure (which caps eff_K) collapse.

    OVERRIDE: ``C4_FAITHFUL_ATTN_EVICT`` (the genuinely-computing path) forces this
    OFF — direct-CAM is the exact B-class draft-trust lever the faithful path
    replaces with the model's own genuine softmax query.  So even if a caller left
    C4_DIRECT_CAM_BATCHED set (e.g. the composed lever set), faithful mode reverts
    the CAM heads to real scoring over the (evicted, bounded) cache."""
    if os.environ.get("C4_FAITHFUL_ATTN_EVICT", "0") not in ("0", "", "false", "False"):
        return False
    return os.environ.get("C4_DIRECT_CAM_BATCHED", "0") not in ("0", "", "false", "False")


def verify_addr_enabled() -> bool:
    """C4_DIRECT_CAM_VERIFY_ADDR (DEFAULT OFF): on each direct-CAM read, INDEPENDENTLY
    verify the READ ADDRESS against the model's OWN computed query — O(1) per read, no
    O(n_store) softmax.

    The faithfulness gap (DOOM_FASTPATH_FAITHFULNESS_AUDIT_2026_08_05.md, scenario E):
    the direct-CAM path resolves each read's address->row->value entirely from
    ``resolve_load_rows(draft)`` and injects it; the model's ``W_q`` query for the CAM
    heads is COMPUTED (``Q = W_q.linear(x)``) but then DISCARDED.  A self-consistent
    wrong draft (store_log + frames + tokens all agree on a wrong value/address) is
    therefore ACCEPTED 7/7 — the address is draft-TRUSTED, never independently checked.

    But the model DOES compute the query, and the query IS the binary address key: for
    a CAM head, ``Q[base+b] = smag*(2*qry_bit[b] - 1)`` (``_bake_cam_head``), so
    ``sign(Q[base+b])`` decodes the address bit ``b`` the model is looking for.  This
    lever, when ON, decodes the model's OWN queried address from that sign pattern at
    each resolved query row and compares it against the draft's ``ResolvedRead.addr``.
    A single ADDR_BITS-wide bit compare per read (O(1), NOT O(n_store)) — the large-KV
    speedup is untouched.  On mismatch it flags a divergence into the same terminal-FAIL
    path the register/token compare uses (``VerifyResult.first_mismatch``,
    ``kind='cam_addr'``).

    Converts "address DRAFT-TRUSTED" -> "address INDEPENDENTLY VERIFIED".  A
    self-consistent value swap AT the correct address is STILL trusted (re-deriving the
    value requires the full O(S) softmax gather over the stores — that is the
    full-attention job, out of scope here).  DEFAULT OFF -> byte-identical to the
    unverified direct-CAM path (golden 069cc32f unchanged)."""
    return os.environ.get("C4_DIRECT_CAM_VERIFY_ADDR", "0") not in ("0", "", "false", "False")


# ===========================================================================
# 1. THE CAM-HEAD MAP: which (block_idx, head_idx) is each global CAM read, and
#    which destination band + head class it drives.  Mirrors
#    ``bake_global_cam_heads`` (the SOLE authority) so the wiring tracks the build.
# ===========================================================================
def _find(names: List[str], name: str) -> int:
    return names.index(name)


def cam_head_map(model, L: PureForwardCompleteLayout
                 ) -> Dict[int, List[Tuple[int, str]]]:
    """Return ``{block_idx: [(head_idx, kind), ...]}`` for the global CAM heads baked
    by ``bake_global_cam_heads``, using ``L._block_names`` (set by every builder).

    ``kind`` selects the resolver source: "mem" (LI/LC -> AX), "pop" (stack ->
    STACK0), "lev" (LEV ret-PC -> LEV_RET), "uni" (merged read), "code" (fetch@PC
    -> OP_VAL + IMM_NIB).  The 3-head + CFM build (doom's config) is the default
    branch (the unify flags are OFF)."""
    names = list(getattr(L, "_block_names", []))
    if not names:
        raise RuntimeError("cam_head_map: L._block_names missing (build path did not "
                           "set it); direct-CAM cannot locate the CAM blocks.")
    out: Dict[int, List[Tuple[int, str]]] = {}
    stk = _find(names, "stack-pop-cam")
    if _unify_cam_head_enabled():
        out.setdefault(stk, []).append((N_ROLES + 1, "uni"))
        out.setdefault(stk, []).append((N_ROLES + 2, "lev"))
    else:
        mem = _find(names, "mem-cam")
        out.setdefault(mem, []).append((N_ROLES, "mem"))
        out.setdefault(stk, []).append((N_ROLES + 1, "pop"))
        out.setdefault(stk, []).append((N_ROLES + 2, "lev"))
    if _pf_cfm_enabled():
        csel = _find(names, "code-select")
        chead = model.blocks[csel].attn.n_heads - 1
        out.setdefault(csel, []).append((chead, "code"))
    return out


# ===========================================================================
# 2. THE RESOLVED VALUE per ABSOLUTE query-row position, per head class.
# ===========================================================================
class ResolvedTable:
    """Per absolute query-row position resolved CAM values, ready for direct gather.

    ``mem/pop/lev/uni[pos]`` -> the resolved value (int) at the step whose query row
    is at abs stream position ``pos`` (absent == that head is a pure softmax1 sink
    at that step -> ZFOD 0 output).  ``code[pos]`` -> ``(op, imm)`` (one fetch/step).

    ``addr[kind][pos]`` -> the ADDRESS the draft claims the read resolves to (the CAM
    binary-address key); ``code_addr[pos]`` -> the PC the code fetch queries.  These are
    what C4_DIRECT_CAM_VERIFY_ADDR checks against the model's OWN decoded query address."""

    def __init__(self):
        self.mem: Dict[int, int] = {}
        self.pop: Dict[int, int] = {}
        self.lev: Dict[int, int] = {}
        self.uni: Dict[int, int] = {}
        self.code: Dict[int, Tuple[int, int]] = {}
        # DRAFT-CLAIMED ADDRESS per read (for O(1) address verification).  Keyed the
        # same as the value dicts: ``addr[kind][pos]`` is the address the draft's
        # ``resolve_load_rows`` used to select the resolved row at that query position.
        self.addr: Dict[str, Dict[int, int]] = {"mem": {}, "pop": {}, "lev": {}, "uni": {}}
        self.code_addr: Dict[int, int] = {}
        # populated by ``install_direct_cam_batched`` when C4_DIRECT_CAM_VERIFY_ADDR is on.
        self.addr_sink: Optional["DivergenceSink"] = None

    def by_kind(self, kind: str) -> Dict:
        return getattr(self, kind)

    def addr_by_kind(self, kind: str) -> Dict[int, int]:
        return self.code_addr if kind == "code" else self.addr[kind]


class DivergenceSink:
    """A mutable holder the direct-CAM forward writes an ADDRESS divergence into and
    ``verify_blocks`` polls after the forward.

    ``hit`` records the FIRST (lowest absolute query position) address mismatch seen:
    ``{"query_pos", "head", "kind", "model_addr", "draft_addr"}``.  Kept per-run on the
    ``ResolvedTable`` so a fresh install starts clean."""

    def __init__(self):
        self.hit: Optional[dict] = None

    def report(self, query_pos: int, head: int, kind: str,
               model_addr: int, draft_addr: int) -> None:
        # keep the earliest (causal-prefix) divergence — the verify path stops at the
        # first bad step, so the lowest query position is the one that matters.
        if self.hit is None or query_pos < self.hit["query_pos"]:
            self.hit = {"query_pos": int(query_pos), "head": int(head), "kind": kind,
                        "model_addr": int(model_addr), "draft_addr": int(draft_addr)}


def _n_seed(draft) -> int:
    """Number of leading DATA-SEGMENT seed store frames (frames 0..n_seed-1); the
    init frame is frame n_seed.  Recovered from the store_log's leading run of
    consecutive frame indices starting at 0 (matching ``draft_pf_program``)."""
    sl = draft.store_log or {}
    n = 0
    while n in sl:
        n += 1
    return n


def _frame_to_step(draft, n_seed: int) -> Dict[int, int]:
    """Map each PRIMARY draft FRAME index to the STEP whose query row we decode.

    ``draft_pf_program`` records ``read_log[frame_idx]`` AFTER ``frame_idx += 1``
    for the step's own primary frame, so a read's ``read_frame`` is that step's
    primary frame index.  We replay the SAME frame counter (advancing past the extra
    store frames a file op emits for its READ input bytes) to invert it."""
    m: Dict[int, int] = {}
    frame_idx = n_seed
    for step in range(draft.step_count):
        fr = draft.frames[step]
        frame_idx += 1                       # this step's primary frame
        m[frame_idx] = step
        if fr.get("is_file"):
            frame_idx += int(fr.get("n_byte_stores", 0) or 0)
    return m


def build_resolved_table(draft, code: List[isa.Instr]) -> ResolvedTable:
    """Resolve every read of the perfect draft to its value, keyed by the ABSOLUTE
    stream position of the reading step's query row (``draft.win_starts[step]``)."""
    tbl = ResolvedTable()
    reads = resolve_load_rows(draft)                 # {read_frame: [ResolvedRead]}
    n_seed = _n_seed(draft)
    frame_to_step = _frame_to_step(draft, n_seed)
    for rf, rlist in reads.items():
        step = frame_to_step.get(rf)
        if step is None:
            continue
        pos = draft.win_starts[step]
        for r in rlist:
            d = tbl.by_kind(r.head) if r.head in ("mem", "pop", "lev", "uni") else None
            if d is not None:
                d[pos] = r.value & 0xFFFFFFFF
                # DRAFT-CLAIMED ADDRESS for this read (the CAM key resolve_load_rows
                # used) — checked against the model's own decoded query under
                # C4_DIRECT_CAM_VERIFY_ADDR.
                tbl.addr[r.head][pos] = int(r.addr) & 0xFFFFFFFF
    # CODE FETCH@PC: recover the pre-step PC per step and record (op, imm) + the PC key.
    if _pf_cfm_enabled():
        pc = 0
        for step in range(draft.step_count):
            fr = draft.frames[step]
            if 0 <= pc < len(code):
                ins = code[pc]
                tbl.code[draft.win_starts[step]] = (int(ins.op), int(ins.imm))
                tbl.code_addr[draft.win_starts[step]] = int(pc)
            pc = fr["pc"]                    # post-step pc -> next step's pre-step pc
    return tbl


# ===========================================================================
# 3. THE RECONSTRUCTED HEAD-OUTPUT VECTOR (the exact softmax winner's V).
# ===========================================================================
def _head_out_vec(kind: str, value, code_val, HD: int, device, dtype) -> torch.Tensor:
    """The CAM head's attention output vector (HD-wide, the head's value-slot space)
    for the resolved value -- i.e. exactly ``softmax1(scores) @ V`` when the winner's
    softmax weight is ~=1.

    mem/pop/lev/uni: winner V is ``nibbles(value)`` at slots ``ADDR_BITS+3 + j``
    (``_bake_cam_head``: ``W_v[base+ADDR_BITS+3+j] <- VAL_NIB[j]``).
    code: winner V is ``op`` at slot ``CODE_ADDR_BITS+4`` and ``nibbles(imm)`` at
    slots ``CODE_ADDR_BITS+5 + j`` (``_bake_code_cam_head``)."""
    out = torch.zeros(HD, device=device, dtype=dtype)
    if kind == "code":
        op, imm = code_val
        v0 = CODE_ADDR_BITS + 4
        out[v0] = float(op)
        for j, nv in enumerate(V.nibbles_of_value(imm & 0xFFFFFFFF, IMM_NIBS)):
            out[v0 + 1 + j] = float(nv)
    else:
        b0 = ADDR_BITS + 3
        for j, nv in enumerate(V.nibbles_of_value(value & 0xFFFFFFFF, NIB_PER_REG)):
            out[b0 + j] = float(nv)
    return out


# ===========================================================================
# 3b. THE O(1) ADDRESS CHECK (C4_DIRECT_CAM_VERIFY_ADDR): decode the MODEL's OWN
#     queried address from its computed query Q and compare it to the draft's addr.
# ===========================================================================
def _n_addr_bits(kind: str) -> int:
    """Number of binary address-key bits the CAM head keys on: ADDR_BITS (32) for the
    memory/stack/lev/uni heads, CODE_ADDR_BITS (12) for the code-fetch head."""
    return CODE_ADDR_BITS if kind == "code" else ADDR_BITS


def _decode_model_query_addr(Qh_rows: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Decode the MODEL's OWN queried address from its computed per-head query.

    ``Qh_rows`` : Float [nq, HD] — the CAM head's query vector at ``nq`` query rows
    (``Q[0, head, rows]``).  ``_bake_cam_head`` writes, per address bit ``b``,
    ``Q[base+b] = 2*smag*qry_bit[b] - smag*1 = smag*(2*qry_bit[b]-1)`` (``smag>0``), so
    at an ARMED read row ``sign(Q[..., b])`` decodes the queried address bit ``b``
    EXACTLY: ``>0`` ⟺ bit=1, ``<0`` ⟺ bit=0.  (The code head is structurally identical
    on ``CODE_ADDR_BITS`` bits: ``Q[base+b]=2*smag*CODE_QRY_BIN[b]-smag*IS_FETCH``, and
    at a fetch row IS_FETCH=1 so the same sign decode holds.)

    Returns a LongTensor [nq] of the decoded addresses — O(nq * n_bits), NOT O(n_store).
    NOTE the residual bits ``> 0``: ``bit = (Q[...,b] > 0)``; strict ``>`` maps the
    negative -smag to 0 and the positive +smag to 1 (an exact address query is never a
    tie at 0 — smag is the sqrt of EFF/scale, ~hundreds)."""
    bits = (Qh_rows[:, :n_bits] > 0).to(torch.long)          # [nq, n_bits]
    weights = (1 << torch.arange(n_bits, device=Qh_rows.device, dtype=torch.long))
    return (bits * weights.unsqueeze(0)).sum(dim=1)          # [nq]


# ===========================================================================
# 4. THE DIRECT-CAM windowed forward.  A drop-in for a CAM block's ``.forward``:
#    the CAM head(s) DIRECT-GATHER the resolved V at query rows (no O(S) global
#    score); every OTHER head runs the ordinary windowed attention.
# ===========================================================================
def _vec_cam_enabled() -> bool:
    """C4_DIRECT_CAM_VEC (#871, DEFAULT OFF): precompute the CAM heads' resolved
    output vectors as a dense ``[H_cam, n_query, HD]`` table + a ``pos_map`` gather,
    replacing the per-row Python loop (``for ri, ap in enumerate(q_pos.tolist())`` +
    per-row ``torch.zeros``/scalar-``float``/scatter) that #865 pinned as the
    host-sync + tiny-op wall (``aten::fill_``/``copy_``/``_local_scalar_dense``
    flood).  Byte-exact: the gathered vector is the SAME ``_head_out_vec`` per
    position, just assembled once on the host and scattered in ONE vectorized op.
    OFF (default) -> the per-row loop (the original path, golden 069cc32f
    unchanged)."""
    return os.environ.get("C4_DIRECT_CAM_VEC", "0") not in ("0", "", "false", "False")


def _build_cam_out_table(cam_heads: List[Tuple[int, str]], tbl: ResolvedTable,
                         HD: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Precompute the CAM heads' resolved output vectors for the WHOLE program, once.

    Returns ``(head_ids, pos, out_tab)`` where
      * ``head_ids`` : LongTensor [n_cam]  the CAM head indices
      * ``pos``      : LongTensor [n_read]  every absolute query-row position that
                       ANY cam head reads at (union over kinds)
      * ``out_tab``  : Float [n_cam, n_read, HD]  the head's ``_head_out_vec`` at that
                       position (0 where that head does not read at that position)
    A single ``pos_map`` gather (built in the forward) then scatters ``out_tab`` into
    the query rows present in the span — NO per-row Python loop."""
    import numpy as _np
    head_ids = [h for (h, _k) in cam_heads]
    kinds = [k for (_h, k) in cam_heads]
    # union of positions any cam head resolves at.
    pos_set = set()
    for k in kinds:
        if k == "code":
            pos_set.update(tbl.code.keys())
        else:
            pos_set.update(tbl.by_kind(k).keys())
    pos = sorted(pos_set)
    if not pos:
        return None
    pos_index = {p: i for i, p in enumerate(pos)}
    n_read = len(pos)
    # VECTORIZED table build (no per-position torch.zeros / scalar-float — that inner
    # loop was itself a ~39k-op aten::fill_ / _local_scalar_dense flood at install).
    # Fill a flat numpy [n_cam, n_read, HD] buffer with vectorized nibble slices.
    out_np = _np.zeros((len(head_ids), n_read, HD), dtype=_np.float32)
    for ci, (h, k) in enumerate(zip(head_ids, kinds)):
        if k == "code":
            items = list(tbl.code.items())
            if not items:
                continue
            rows = _np.fromiter((pos_index[p] for p, _cv in items), dtype=_np.int64,
                                count=len(items))
            ops = _np.fromiter((int(cv[0]) for _p, cv in items), dtype=_np.int64,
                               count=len(items))
            imms = _np.fromiter((int(cv[1]) & 0xFFFFFFFF for _p, cv in items),
                                dtype=_np.int64, count=len(items))
            v0 = CODE_ADDR_BITS + 4
            out_np[ci, rows, v0] = ops.astype(_np.float32)
            for j in range(IMM_NIBS):
                out_np[ci, rows, v0 + 1 + j] = ((imms >> (4 * j)) & 0xF).astype(_np.float32)
        else:
            d = tbl.by_kind(k)
            items = list(d.items())
            if not items:
                continue
            rows = _np.fromiter((pos_index[p] for p, _v in items), dtype=_np.int64,
                                count=len(items))
            vals = _np.fromiter((int(v) & 0xFFFFFFFF for _p, v in items),
                                dtype=_np.int64, count=len(items))
            b0 = ADDR_BITS + 3
            for j in range(NIB_PER_REG):
                out_np[ci, rows, b0 + j] = ((vals >> (4 * j)) & 0xF).astype(_np.float32)
    out_tab = torch.from_numpy(out_np)
    return (torch.tensor(head_ids, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long), out_tab)


def _install_direct_forward(model, block_idx: int, cam_heads: List[Tuple[int, str]],
                            tbl: ResolvedTable, sink: Optional["DivergenceSink"] = None):
    """Bind a direct-CAM forward onto ``model.blocks[block_idx].attn`` that gathers
    the resolved V for ``cam_heads`` at query rows and defers every other head to
    the ordinary windowed forward already installed on the block.

    When ``sink`` is not None (C4_DIRECT_CAM_VERIFY_ADDR), the forward ALSO decodes the
    model's OWN queried address from the computed query ``Q`` at each resolved query row
    and reports any mismatch vs the draft's ``tbl.addr`` into ``sink`` — O(1) per read."""
    attn = model.blocks[block_idx].attn
    from .local_attention import windowed_forward, live_value_heads
    cam_head_ids = {h for (h, _k) in cam_heads}
    cam_kind = {h: k for (h, k) in cam_heads}
    _verify_addr = sink is not None
    # VECTORIZED CAM-output table (C4_DIRECT_CAM_VEC): precompute the resolved output
    # per (cam head, position) ONCE + build a dense pos_map for O(1) gather-scatter.
    # Restrict to cam heads that are GLOBAL on this block (the original per-row loop
    # only wrote ``g_cam`` = cam_head_ids ∩ global heads; the CAM heads are global by
    # construction, but keep the filter so the vec path is loop-equivalent regardless).
    _vec = _vec_cam_enabled()
    _gmask0 = getattr(attn, "_global_head_mask", None)
    if _vec and _gmask0 is not None:
        _cam_heads_g = [(h, k) for (h, k) in cam_heads if bool(_gmask0[h])]
    else:
        _cam_heads_g = cam_heads
    _cam_tab = _build_cam_out_table(_cam_heads_g, tbl, attn.head_dim) if _vec else None
    _cam_head_t = _cam_pos_map = _cam_out = None
    if _cam_tab is not None:
        _cam_head_t, _cam_pos, _cam_out = _cam_tab
        _maxp = int(_cam_pos.max().item()) if _cam_pos.numel() else -1
        _cam_pos_map = torch.full((_maxp + 2,), -1, dtype=torch.long)
        if _cam_pos.numel():
            _cam_pos_map[_cam_pos] = torch.arange(_cam_pos.numel(), dtype=torch.long)
    # LIVE-VALUE LOCAL heads (C4_DIRECT_CAM_LIVE_LOCAL, default ON when direct-CAM is):
    # the set of local heads whose W_v/W_o are non-zero.  A _zero_attn local head's
    # output is exactly 0, so we skip its banded score+gather (byte-exact).  OFF keeps
    # the old behaviour (score every non-global head).
    _live_local = None
    if os.environ.get("C4_DIRECT_CAM_LIVE_LOCAL", "1") not in ("0", "", "false", "False"):
        _live_local = set(int(h) for h in live_value_heads(attn))

    def direct_forward(self, x, past_kv=None, q_positions=None, use_cache=False):
        # Full head set for this block.
        H, HD = self.n_heads, self.head_dim
        B, S, D = x.shape
        # Which of THIS block's global heads are CAM (direct-gather) vs genuine
        # (must still score the cache).  Non-CAM heads (local + any genuine global)
        # are handled by running the ordinary windowed forward, then we OVERWRITE
        # the CAM heads' output slice with the direct gather and re-project W_o.
        #
        # To skip the CAM heads' O(S) score we must not compute it.  So: run the
        # windowed forward with the CAM heads temporarily moved OUT of the global
        # set (their score is then never computed), capture its per-head ``out``, and
        # substitute the reconstructed resolved V for the CAM heads before W_o.
        gmask = getattr(self, "_global_head_mask", None)
        W = getattr(self, "_local_window", None)
        if gmask is None or W is None:
            # No local split installed -> fall back to the ordinary attention (still
            # correct, just not the fast path).  Should not happen in the doom run.
            return _ORIG_ATTN_FORWARD(self, x, past_kv=past_kv,
                                      q_positions=q_positions, use_cache=use_cache)

        if q_positions is None:
            q_pos = torch.arange(S, device=x.device)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)

        Q = self.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)
        Knew = self.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
        Vnew = self.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)
        out = x.new_zeros(B, H, S, HD)

        g_idx = torch.nonzero(gmask, as_tuple=False).flatten().tolist()
        l_idx = [h for h in range(H) if not bool(gmask[h])]
        # LIVE-HEAD SKIP (byte-exact): a _zero_attn local head has W_v==W_o==0, so its
        # attention output slice is EXACTLY 0 (== the pre-zeroed ``out``).  Running the
        # banded score+gather for it is pure wasted memory-BW (the band gather
        # materializes [Sq,W,HD] per head — gigabytes).  Restrict l_idx to the
        # live-VALUE local heads; the dead ones keep out=0, byte-identical.  On the
        # CAM blocks (7/11) this drops ~22 dead heads -> only the 1-2 direct-gathered
        # global CAM heads remain live (the block's real per-step cost -> O(1)).
        if _live_local is not None:
            l_idx = [h for h in l_idx if h in _live_local]
        # split global heads into CAM (direct) and NON-CAM (still scored).
        g_cam = [h for h in g_idx if h in cam_head_ids]
        g_score = [h for h in g_idx if h not in cam_head_ids]

        # ---- O(1) ADDRESS VERIFICATION (C4_DIRECT_CAM_VERIFY_ADDR) ------------------
        # The direct gather below OVERWRITES the CAM heads' output with the draft-
        # resolved value, DISCARDING the model's own query.  Before that, INDEPENDENTLY
        # check that the address the draft resolved this read to is the address the
        # MODEL's OWN query is asking for: decode ``Q[0, h, row, :n_bits]``'s sign
        # pattern (the binary address key) and compare to ``tbl.addr``.  O(n_bits) per
        # read (NOT O(n_store) — no softmax over the cache).  On mismatch, report into
        # the divergence sink; verify_blocks turns it into a terminal FAIL (the same
        # first-divergence path the register/token compare uses).  A wrong-ADDRESS draft
        # (even a self-consistent one) is thus CAUGHT; a value swap AT the correct
        # address is still trusted (re-deriving the value is the full-softmax job).
        if _verify_addr and g_cam:
            # host-copy the span's absolute query positions ONCE (not per head).
            qpos_cpu = q_pos.detach().to("cpu").tolist()
            for h in g_cam:
                kind = cam_kind[h]
                # CODE head is EXCLUDED: its query address is the PC, which the register-
                # transition compare ALREADY verifies independently (``got_pc != want_pc``
                # at every step).  A wrong code-fetch address is a wrong PC -> caught by
                # the existing register check.  (Its query encoding at the value-injection
                # row also differs from the memory heads' ``smag*(2*bit-1)`` sign pattern
                # -- the PC->CODE_QRY_BIN FFN feeds a different row -- so a sign-decode here
                # would false-positive.)  The genuinely draft-trusted, NON-register-covered
                # address is the ``mem`` LOAD address (LI/LC ``mem[addr]``); pop/lev key on
                # SP/BP+4 (register-derived) and are verified here for defence in depth.
                if kind == "code":
                    continue
                addr_d = tbl.addr_by_kind(kind)
                if not addr_d:
                    continue
                # rows on THIS span where the draft says head h reads (abs pos in addr_d).
                sel_rows = [ri for ri, ap in enumerate(qpos_cpu) if ap in addr_d]
                if not sel_rows:
                    continue
                n_bits = _n_addr_bits(kind)
                ri_t = torch.tensor(sel_rows, device=x.device, dtype=torch.long)
                Qh = Q[0, h].index_select(0, ri_t)            # [nq, HD] model's query
                model_addr = _decode_model_query_addr(Qh, n_bits).to("cpu").tolist()
                for k_i, ri in enumerate(sel_rows):
                    ap = qpos_cpu[ri]
                    da = int(addr_d[ap]) & ((1 << n_bits) - 1)
                    ma = int(model_addr[k_i])
                    # FIRE only when the model ASSERTS a NON-ZERO address that DISAGREES
                    # with the draft.  Rationale (block-7-input residual view):
                    #  * ``ma != 0`` means the model's ``QRY_BIN`` residual is genuinely
                    #    populated with a computed address here (the ±smag bit pattern is
                    #    real) -> a disagreement is a REAL divergence (the wrong-address
                    #    draft the audit targets; proven caught in the experiment).
                    #  * ``ma == 0`` is AMBIGUOUS at this residual view: it is EITHER a
                    #    genuine address-0 read (then da==0 and there is no mismatch) OR a
                    #    row whose ``QRY_BIN`` is not asserted at block-7 input under the
                    #    batched/recurrent span (the address-computation FFN's result is
                    #    not in THIS residual view) -> the model is not asserting an
                    #    address here, so we must NOT false-fail.  Skipping ``ma==0``
                    #    mismatches is the CONSERVATIVE choice (never a false positive);
                    #    the honest limit is a draft that resolves a read to a non-zero
                    #    address the model would have read as 0 -- indistinguishable from
                    #    an unasserted query and therefore NOT caught here (documented
                    #    scope).
                    if ma != 0 and ma != da:
                        sink.report(ap, h, kind, ma, int(addr_d[ap]))

        def _attend(heads, K_full, V_full, kpos_full, window):
            if not heads:
                return
            hi = torch.tensor(heads, device=x.device, dtype=torch.long)
            pre_sel = (K_full.shape[1] == hi.numel())
            Ksel = K_full if pre_sel else K_full[:, hi]
            Vsel = V_full if pre_sel else V_full[:, hi]
            Qg = Q[:, hi]
            # TRUE BANDED KERNEL (C4_BANDED_LOCAL_ATTN): the CAM blocks (2/7/11) still
            # carry ~22 LOCAL ingest heads each; without banding THEY pay the O(S^2)
            # masked-full local score even though direct-CAM removed the global one.
            # Route the local (window is an int) group through the O(S*W) band — same
            # byte-exact argument as windowed_forward's local branch.
            if window is not None and os.environ.get("C4_BANDED_LOCAL_ATTN", "0") == "1":
                from .banded_local_attn import banded_local_context
                out[:, hi] = banded_local_context(
                    Qg, Ksel, Vsel, q_pos, kpos_full,
                    self.alibi_slopes[hi], self.scale, int(window))
                return
            sc = torch.matmul(Qg, Ksel.transpose(-2, -1)) * self.scale
            dist = (q_pos.unsqueeze(1) - kpos_full.unsqueeze(0)).float()
            sc = sc - self.alibi_slopes[hi].view(1, -1, 1, 1) * dist.abs().unsqueeze(0)
            m = (kpos_full.unsqueeze(0) > q_pos.unsqueeze(1))
            if window is not None:
                m = m | (dist >= window)
            sc = sc.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
            a = softmax1(sc, dim=-1)
            out[:, hi] = torch.matmul(a, Vsel)

        # ---- assemble each group's KV from the split cache (as windowed_forward) ----
        if isinstance(past_kv, _SplitPastKV):
            pk = past_kv
            # GLOBAL non-CAM heads: full causal score over the global cache.
            if g_score:
                gi = torch.tensor(g_score, device=x.device, dtype=torch.long)
                Kg_new, Vg_new = Knew[:, gi], Vnew[:, gi]
                if pk.Kg is not None:
                    # pk.Kg holds ALL global heads (ordered by _global_head_idx); we
                    # need this subset. Map g_score -> positions within the global set.
                    gpos = _subset_positions(pk, g_score)
                    Kg = torch.cat([pk.Kg[:, gpos], Kg_new], dim=2)
                    Vg = torch.cat([pk.Vg[:, gpos], Vg_new], dim=2)
                    posg = torch.cat([pk.posg.to(x.device), q_pos], dim=0)
                else:
                    Kg, Vg, posg = Kg_new, Vg_new, q_pos
                _attend(g_score, Kg, Vg, posg, None)
            # LOCAL heads.
            if l_idx:
                li = torch.tensor(l_idx, device=x.device, dtype=torch.long)
                Kl_new, Vl_new = Knew[:, li], Vnew[:, li]
                if pk.Kl is not None:
                    Kl = torch.cat([pk.Kl, Kl_new], dim=2)
                    Vl = torch.cat([pk.Vl, Vl_new], dim=2)
                    posl = torch.cat([pk.posl.to(x.device), q_pos], dim=0)
                else:
                    Kl, Vl, posl = Kl_new, Vl_new, q_pos
                _attend(l_idx, Kl, Vl, posl, W)
        else:
            # first span / mask-only: full KV in the block, windowed READ.
            if past_kv is not None:
                K_cache, V_cache, pos_cache = past_kv
                K = torch.cat([K_cache, Knew], dim=2)
                Vv = torch.cat([V_cache, Vnew], dim=2)
                k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
            else:
                K, Vv, k_pos = Knew, Vnew, q_pos
            if g_score:
                _attend(g_score, K, Vv, k_pos, None)
            if l_idx:
                _attend(l_idx, K, Vv, k_pos, W)

        # ---- DIRECT-CAM: reconstruct the CAM heads' output at each QUERY row ----
        # A query row carries all-ROLE tags; the CAM heads' softmax winner V is the
        # resolved value's nibble vector.  Non-query rows (frozen context) keep out=0
        # for the CAM heads (their output is discarded; only K/V matters for them).
        if _cam_out is not None:
            # VECTORIZED (C4_DIRECT_CAM_VEC): map each span row's abs position -> its
            # precomputed-table row (-1 if not resolved), then scatter the whole
            # [n_cam, nq, HD] output block in ONE op — NO per-row Python loop, NO
            # per-row torch.zeros / scalar-float / host-sync.  Byte-identical: the
            # gathered vector IS _head_out_vec at that position.
            pm = _cam_pos_map.to(x.device)
            clamped = q_pos.clamp(max=pm.numel() - 1)
            row_idx = pm[clamped]                              # [S], -1 non-resolved
            qmask = row_idx >= 0
            if bool(qmask.any()):
                sel = row_idx[qmask]                           # [nq] table-row indices
                span_q = qmask.nonzero(as_tuple=False).flatten()  # [nq] span rows
                heads = _cam_head_t.to(x.device)               # [n_cam]
                vecs = _cam_out.to(device=x.device, dtype=out.dtype)  # [n_cam, nr, HD]
                gathered = vecs.index_select(1, sel)           # [n_cam, nq, HD]
                # out[0, heads, span_q, :] = gathered  (advanced index, one write).
                out[0, heads.unsqueeze(1), span_q.unsqueeze(0), :] = gathered
        else:
            qpos_list = q_pos.tolist()
            for h in g_cam:
                kind = cam_kind[h]
                d = None if kind == "code" else tbl.by_kind(kind)
                for ri, ap in enumerate(qpos_list):
                    if kind == "code":
                        cv = tbl.code.get(ap)
                        if cv is None:
                            continue        # not a fetch row (frozen ctx) -> sink 0
                        vec = _head_out_vec("code", None, cv, HD, x.device, out.dtype)
                    else:
                        if ap not in d:
                            continue        # this head didn't read at this step -> 0
                        vec = _head_out_vec(kind, d[ap], None, HD, x.device, out.dtype)
                    out[0, h, ri] = vec

        out2 = out.transpose(1, 2).contiguous().view(B, S, D)
        res = x + self.W_o.linear(out2)
        if use_cache:
            return res, (Knew, Vnew, q_pos)
        return res

    attn.forward = direct_forward.__get__(attn, type(attn))


def _subset_positions(pk: _SplitPastKV, heads: List[int]) -> torch.Tensor:
    """Map absolute head indices ``heads`` to their positions within the split
    cache's GLOBAL head ordering (``pk`` stores global heads ordered by
    ``_global_head_idx``)."""
    gorder = pk.g_idx.tolist() if hasattr(pk, "g_idx") and pk.g_idx is not None else None
    if gorder is None:
        # fall back: assume pk.Kg is ordered by ascending global head index.
        gorder = sorted(heads)
    pos = [gorder.index(h) for h in heads]
    return torch.tensor(pos, device=pk.Kg.device, dtype=torch.long)


_ORIG_ATTN_FORWARD = None


def install_direct_cam_batched(model, L: PureForwardCompleteLayout,
                               draft, code: List[isa.Instr],
                               verbose: bool = False) -> Optional[ResolvedTable]:
    """Install the direct-CAM forward on the CAM blocks for the batched verify path.

    Returns the ``ResolvedTable`` (kept alive by the caller) or None if disabled.
    Call AFTER ``install_local_attention`` + ``install_dead_block_fusion`` (the
    direct forward defers non-CAM heads to the windowed attention they installed,
    and the CAM blocks are never dead-fused since they carry live CAM heads)."""
    global _ORIG_ATTN_FORWARD
    if not direct_cam_batched_enabled():
        return None
    import c4_min.sparse_forward as _SF
    _ORIG_ATTN_FORWARD = _SF.SparseAttn.forward
    tbl = build_resolved_table(draft, code)
    # O(1) ADDRESS VERIFICATION sink (C4_DIRECT_CAM_VERIFY_ADDR).  When ON, the forward
    # reports any address the draft resolved that the model's OWN query disagrees with;
    # verify_blocks polls ``tbl.addr_sink`` after each span and turns a hit into a FAIL.
    sink = DivergenceSink() if verify_addr_enabled() else None
    tbl.addr_sink = sink
    chm = cam_head_map(model, L)
    for bi, heads in chm.items():
        _install_direct_forward(model, bi, heads, tbl, sink=sink)
    if verbose:
        n_mem, n_pop, n_lev = len(tbl.mem), len(tbl.pop), len(tbl.lev)
        print(f"[direct-cam] installed on blocks {sorted(chm)} "
              f"heads={ {bi:[h for h,_ in hs] for bi,hs in chm.items()} }  "
              f"resolved mem={n_mem} pop={n_pop} lev={n_lev} code={len(tbl.code)}"
              f"{'  [addr-verify ON]' if sink is not None else ''}",
              flush=True)
    return tbl
