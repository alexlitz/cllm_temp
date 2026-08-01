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
    O(S) score-matrix VRAM pressure (which caps eff_K) collapse."""
    return os.environ.get("C4_DIRECT_CAM_BATCHED", "0") not in ("0", "", "false", "False")


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
    at that step -> ZFOD 0 output).  ``code[pos]`` -> ``(op, imm)`` (one fetch/step)."""

    def __init__(self):
        self.mem: Dict[int, int] = {}
        self.pop: Dict[int, int] = {}
        self.lev: Dict[int, int] = {}
        self.uni: Dict[int, int] = {}
        self.code: Dict[int, Tuple[int, int]] = {}

    def by_kind(self, kind: str) -> Dict:
        return getattr(self, kind)


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
    # CODE FETCH@PC: recover the pre-step PC per step and record (op, imm).
    if _pf_cfm_enabled():
        pc = 0
        for step in range(draft.step_count):
            fr = draft.frames[step]
            if 0 <= pc < len(code):
                ins = code[pc]
                tbl.code[draft.win_starts[step]] = (int(ins.op), int(ins.imm))
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
# 4. THE DIRECT-CAM windowed forward.  A drop-in for a CAM block's ``.forward``:
#    the CAM head(s) DIRECT-GATHER the resolved V at query rows (no O(S) global
#    score); every OTHER head runs the ordinary windowed attention.
# ===========================================================================
def _install_direct_forward(model, block_idx: int, cam_heads: List[Tuple[int, str]],
                            tbl: ResolvedTable):
    """Bind a direct-CAM forward onto ``model.blocks[block_idx].attn`` that gathers
    the resolved V for ``cam_heads`` at query rows and defers every other head to
    the ordinary windowed forward already installed on the block."""
    attn = model.blocks[block_idx].attn
    from .local_attention import windowed_forward
    cam_head_ids = {h for (h, _k) in cam_heads}
    cam_kind = {h: k for (h, k) in cam_heads}

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
        # split global heads into CAM (direct) and NON-CAM (still scored).
        g_cam = [h for h in g_idx if h in cam_head_ids]
        g_score = [h for h in g_idx if h not in cam_head_ids]

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
        qpos_list = q_pos.tolist()
        for h in g_cam:
            kind = cam_kind[h]
            d = None if kind == "code" else tbl.by_kind(kind)
            for ri, ap in enumerate(qpos_list):
                if kind == "code":
                    cv = tbl.code.get(ap)
                    if cv is None:
                        continue            # not a fetch row (frozen ctx) -> sink 0
                    vec = _head_out_vec("code", None, cv, HD, x.device, out.dtype)
                else:
                    if ap not in d:
                        continue            # this head didn't read at this step -> 0
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
    chm = cam_head_map(model, L)
    for bi, heads in chm.items():
        _install_direct_forward(model, bi, heads, tbl)
    if verbose:
        n_mem, n_pop, n_lev = len(tbl.mem), len(tbl.pop), len(tbl.lev)
        print(f"[direct-cam] installed on blocks {sorted(chm)} "
              f"heads={ {bi:[h for h,_ in hs] for bi,hs in chm.items()} }  "
              f"resolved mem={n_mem} pop={n_pop} lev={n_lev} code={len(tbl.code)}",
              flush=True)
    return tbl
