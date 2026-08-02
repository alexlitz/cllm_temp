"""LOCAL (sliding-window) attention on the c4_min pure-forward NON-MEMORY heads.

Motivation (#686 follow-up).  The fast path (``pf_speculative.verify_blocks`` +
``forward_hidden_cached``) is now the wall (~83 % of the fast wall is the batched
block forwards).  Each block computes the FULL ``Q@Kᵀ`` score matrix ``[H, Sq, Sk]``
— O(S²) per head — even though:

  * only 3 of the ~307 blocks have ANY non-zero attention output (block 0 = frame
    ingest, the mem-cam block = LI/LC memory read, the stack-pop-cam block =
    stack-pop + LEV); every OTHER block's attention is ``_zero_attn`` (W_v==W_o==0),
    so its output is provably ``x`` regardless of which keys it reads, and
  * the memory/stack/LEV heads are the ONLY heads that must reach far into the past
    (up to ~250 k tokens for a deep recursion's outermost LEV — see
    ``blogspec_memory.EFF``); the ~20 ingest heads only content-address the LATEST
    frame (measured window ≤ 28 tokens < one 30-token VM step).

So the win is: give the LOCAL heads a SLIDING WINDOW (vanilla Mistral/Longformer)
of the last ``W`` keys (O(S·W)) and keep only the MEMORY/STACK/LEV heads GLOBAL
(full causal).  Because every windowed head's true attention weight past ``W`` is
exactly 0 (softmax1 + the huge exact-match/role scores + ALiBi recency drive the
tail to ZFOD), local == global for it — BYTE-IDENTICAL — and the O(S²) matrix
collapses to O(S·W) on 20/23 heads of the 3 live blocks and on ALL 23 heads of the
304 pure-passthrough blocks (whose zero-V output is window-invariant).

DESIGN.  ``classify_heads(model)`` walks the baked weights and returns, per block,
the set of GLOBAL head indices (a head is GLOBAL iff it has a non-zero value output
— W_v rows AND W_o cols non-zero — AND its ALiBi recency slope is the small
memory-CAM slope, i.e. it is a far-reaching KV head).  Every other head (ingest +
all zero-V heads) is LOCAL and reads only the last ``W`` keys.

``install_local_attention(model, window=W)`` sets ``attn._local_window`` and
``attn._global_head_mask`` on every block and swaps in ``windowed_forward`` (a
drop-in that is byte-identical to ``SparseAttn.forward`` when the windowed heads'
tail weight is 0).  ``window=None`` / uninstall restores the global forward.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from . import sparse_forward as _SF
from .blogspec_model import softmax1
from .nibble_pure_forward_cached import _SplitPastKV


# The ingest heads' recency slope (latest-frame-wins).  A LIVE head baked with THIS
# large slope is a PROVEN-LOCAL frame-ingest head (measured window ≤ 28 tok < 1
# frame; it can never reach further — every frame re-emits its role).  This is the
# ONLY signature we window a LIVE head on; every other live head stays GLOBAL
# (conservative — a future far-reaching head is kept global by default).
def _ingest_slope() -> float:
    from .nibble_pure_forward import INGEST_RECENCY
    return float(INGEST_RECENCY)


# The MEM CAM heads' recency slope (latest-write-wins) — the far-reaching KV heads.
def _mem_slope() -> float:
    from .blogspec_memory import MEM_ALIBI_SLOPE
    return float(MEM_ALIBI_SLOPE)


# The store-role GATE channel (within a global head's HD slice) that distinguishes
# a STORE row (keys ~0 there) from a NON-STORE row (keys -PEN_GATE there).  Every
# global head (memory-CAM / stack-pop / LEV) bakes it at ``cR = ADDR_BITS + 1``
# (``W_k[base+ADDR_BITS+1, ONE] = -p`` + ``W_k[..., IS_STORE] = p``), so a non-store
# row scores -PEN under a load/pop/lev query -> softmax1 weight EXACTLY 0.  This is
# the channel the content-bound global cache keeps store rows by.
def _store_gate_channel() -> int:
    from .blogspec_memory import ADDR_BITS
    return int(ADDR_BITS) + 1


def _dense_cpu(w) -> torch.Tensor:
    """The dense [out, in] weight of a SparseWeight (or plain tensor), on CPU.

    Used only for the ONE-TIME structural head classification (nnz counting), so it
    is materialised on CPU to avoid a transient GPU dense blowup across 307 blocks.
    """
    if getattr(w, "is_sparse", False):
        if w.dense_resident is not None:
            return w.dense_resident.detach().cpu()
        return w.csr.detach().cpu().to_dense()
    t = w.dense if hasattr(w, "dense") and w.dense is not None else w
    return t.detach().cpu()


def live_value_heads(attn) -> List[int]:
    """Heads with a NON-TRIVIAL value output: W_v rows (input-dim slice
    ``h·HD..(h+1)·HD``) non-zero AND W_o cols (same slice) non-zero.  A head not in
    this list outputs 0 (``x + W_o·0``) no matter which keys it reads.  CPU-only
    (structural nnz), so no GPU dense transient."""
    H, HD = attn.n_heads, attn.head_dim
    wv = _dense_cpu(attn.W_v)
    wo = _dense_cpu(attn.W_o)
    live = []
    for h in range(H):
        sl = slice(h * HD, (h + 1) * HD)
        if int((wv[sl, :] != 0).sum()) > 0 and int((wo[:, sl] != 0).sum()) > 0:
            live.append(h)
    return live


def classify_heads(model, slope_tol: float = 1e-3) -> Dict[int, List[int]]:
    """Return ``{block_idx: [global_head_indices]}`` — the heads kept GLOBAL.

    CONSERVATIVE rule (byte-safety first): a LIVE value head is windowed ONLY if its
    ALiBi recency slope is the ingest slope (``INGEST_RECENCY``, the PROVEN-local
    frame-ingest heads, measured window ≤ 28 tok).  EVERY other LIVE head is kept
    GLOBAL — including any (future) far-reaching head whose slope is neither the
    ingest slope nor the memory slope.  Zero-value heads (W_v==0 or W_o==0) are
    windowed regardless (their attention output is 0, window-invariant), which is
    what wins on the 304 pure-passthrough blocks.

    So GLOBAL = {live heads that are NOT proven-local ingest heads}.
    """
    ings = _ingest_slope()
    out: Dict[int, List[int]] = {}
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        live = set(live_value_heads(at))
        glob = []
        for h in live:
            slope = float(at.alibi_slopes[h])
            is_ingest_local = abs(slope - ings) <= slope_tol
            if not is_ingest_local:            # keep every non-ingest LIVE head GLOBAL
                glob.append(h)
        out[bi] = sorted(glob)
    return out


# ===========================================================================
# The windowed forward — a drop-in for ``SparseAttn.forward`` that keeps the
# GLOBAL heads full-causal and gives the LOCAL heads a sliding window of the last
# ``W`` keys (by ABSOLUTE position).  O(S·W) on the local heads.
# ===========================================================================
def windowed_forward(self, x, past_kv=None, q_positions=None, use_cache=False):
    """Byte-identical to ``SparseAttn.forward`` when every LOCAL head's true
    attention weight past the window is 0.  Splits the heads into GLOBAL (full
    causal over all keys) and LOCAL (only keys with ``q_pos - k_pos < W``).

    TWO cache regimes, selected by the past-KV type:

    * MASK-ONLY (``--local-mask-only`` fallback, ``past_kv`` a plain 3-tuple):
      the full KV of every position is still stored; only the local heads' softmax
      READ is windowed.  Saves compute (O(S·W) local scores) but NOT VRAM.

    * DROP-KV (``_SplitPastKV`` past, the default): the LOCAL heads' old KV was
      already DROPPED from the cache (the ``BlockKVCacheBatched`` split keeps only
      the last-W local rows), so the local heads read a SHORT cache and the global
      heads read their own full cache.  This is what shrinks VRAM from ~7360·S to
      ~3·S + 7357·W.  Byte-identical: the dropped local rows' true weight is 0.

    The returned ``(K, V, k_pos)`` is the NEW window rows only (``Knew/Vnew/q_pos``);
    the driver's commit path routes them into the split cache itself (it slices the
    last-S tail), so no concatenated past-KV tensor is materialised or returned.
    """
    W = getattr(self, "_local_window", None)
    gmask = getattr(self, "_global_head_mask", None)
    if W is None or gmask is None:
        return _global_forward(self, x, past_kv, q_positions, use_cache)

    B, S, D = x.shape
    H, HD = self.n_heads, self.head_dim
    Q = self.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)     # [B,H,S,HD]
    Knew = self.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
    Vnew = self.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)

    if q_positions is None:
        q_pos = torch.arange(S, device=x.device)
    else:
        q_pos = q_positions.to(device=x.device, dtype=torch.long)

    g_idx = torch.nonzero(gmask, as_tuple=False).flatten()
    l_idx = torch.nonzero(~gmask, as_tuple=False).flatten()
    out = x.new_zeros(B, H, S, HD)

    def _attend_group(idx, K_full, V_full, kpos_full, window):
        """softmax1+ALiBi attention output for the heads ``idx`` over the FULL key
        set ``K_full`` at abs positions ``kpos_full``.  ``window`` None -> full
        causal (global heads); an int W -> drop keys with ``q_pos-k_pos >= W``
        (local heads).  ``K_full`` here is already this GROUP's heads (index 0..len(idx))
        when a split cache pre-selected them, else the full-H tensor indexed by ``idx``."""
        if idx.numel() == 0:
            return
        pre_sel = (K_full.shape[1] == idx.numel())     # already this group's heads
        Ksel = K_full if pre_sel else K_full[:, idx]
        Vsel = V_full if pre_sel else V_full[:, idx]
        Qg = Q[:, idx]
        # TRUE BANDED KERNEL (C4_BANDED_LOCAL_ATTN): for the LOCAL heads (window is an
        # int W), score ONLY the last-W keys per query row — O(Sq*W), FLAT in S — via a
        # sliding-band gather instead of materialising the full [B,Hl,Sq,Sk] matrix and
        # masking it.  Byte-identical: the out-of-band weight is provably 0 (the masked
        # path already sets it to -inf), so we simply never SCORE it.  GLOBAL heads
        # (window is None) keep the full-causal matmul (they must reach far into the
        # past — memory/stack/LEV).  See banded_local_attn.py.
        import os as _osb
        if window is not None and _osb.environ.get("C4_BANDED_LOCAL_ATTN", "0") == "1":
            from .banded_local_attn import banded_local_context
            out[:, idx] = banded_local_context(
                Qg, Ksel, Vsel, q_pos, kpos_full,
                self.alibi_slopes[idx], self.scale, int(window))
            return
        # BYTE-EXACT FLASH (C4_FLASH_ATTN): tiled softmax1+ALiBi that NEVER
        # materialises the [B,H,Sq,Sk] score matrix — the general O(S²)-memory fix
        # for the GLOBAL heads (window is None: memory/stack/LEV must reach over the
        # whole growing KV, no direct-CAM draft).  softmax1 == plain-softmax over a
        # BOS-sink column (blogspec_model §40-44), so a standard flash kernel + the
        # sink recovers it EXACTLY.  Un-cached FULL case -> SDPA mem-efficient
        # (is_causal top-left == the reference); every other case -> the general
        # online-softmax1 Triton kernel.  fp32 (~1e-6, below the nibble margin).
        if _osb.environ.get("C4_FLASH_ATTN", "0") == "1" and Qg.is_cuda:
            from .flash_softmax1 import flash_softmax1_context
            Sq_ = Qg.shape[2]
            Sk_ = Ksel.shape[2]
            uncached_full = (window is None and Sq_ == Sk_
                             and bool((kpos_full == q_pos).all())
                             and bool((q_pos == torch.arange(
                                 Sq_, device=q_pos.device)).all()))
            out[:, idx] = flash_softmax1_context(
                Qg, Ksel, Vsel, q_pos, kpos_full, self.alibi_slopes[idx],
                self.scale, window=window, uncached_full=uncached_full)
            return
        sc = torch.matmul(Qg, Ksel.transpose(-2, -1)) * self.scale
        dist = (q_pos.unsqueeze(1) - kpos_full.unsqueeze(0)).float()   # signed [S,Sk']
        sc = sc - self.alibi_slopes[idx].view(1, -1, 1, 1) * dist.abs().unsqueeze(0)
        m = (kpos_full.unsqueeze(0) > q_pos.unsqueeze(1))              # future keys
        if window is not None:
            m = m | (dist >= window)                                  # older than window
        sc = sc.masked_fill(m.unsqueeze(0).unsqueeze(0), float("-inf"))
        a = softmax1(sc, dim=-1)
        out[:, idx] = torch.matmul(a, Vsel)
        # WALL#6 ATTN DIAG (C4_WALL6_ATTN_DIAG=<abs_qpos>, additive/inert): for the
        # GLOBAL heads only (window is None), if the target absolute query position is
        # in this span, dump the top-attended key positions + weights so we can see
        # whether the stack-pop CAM aliases a 1-bit-neighbor address.
        import os as _osa
        _tq = _osa.environ.get("C4_WALL6_ATTN_DIAG")
        if _tq is not None and window is None:
            try:
                tq = int(_tq)
                rad = int(_osa.environ.get("C4_WALL6_ATTN_RAD", "0"))
                qpos_l = q_pos.tolist()
                exact_pos = _osa.environ.get("C4_WALL6_EXACT_POS")   # a key pos to trace
                exact_pos = int(exact_pos) if exact_pos else None
                kpl = kpos_full.tolist()
                for ri, qp in enumerate(qpos_l):
                    if abs(qp - tq) > rad:
                        continue
                    for jj, hh in enumerate(idx.tolist()):
                        w = a[0, jj, ri]                    # [Sk'] weights for this head/query
                        raw = sc[0, jj, ri]                 # [Sk'] pre-softmax scores
                        topw, topi = torch.topk(w, min(4, w.numel()))
                        kp = kpos_full[topi].tolist()
                        rawmax = float(raw.max())
                        extra = ""
                        if exact_pos is not None and exact_pos in kpl:
                            ki = kpl.index(exact_pos)
                            extra = (f" | exact_pos={exact_pos} raw_score={float(raw[ki]):.1f} "
                                     f"weight={float(w[ki]):.6f} (rawmax={rawmax:.1f})")
                        print(f"[attn-diag] qpos={qp} head={hh}: top keys(pos,weight)="
                              f"{list(zip(kp, [round(float(x),4) for x in topw.tolist()]))} "
                              f"sink_w={round(float(1.0 - w.sum()),4)} rawmax={rawmax:.1f}{extra}", flush=True)
            except Exception as _e:
                print(f"[attn-diag] err {_e}", flush=True)

    if isinstance(past_kv, _SplitPastKV):
        # DROP-KV: each head group reads its OWN cache (global full, local trimmed).
        # Append this span's new rows to each group before attending; the local
        # group's cache is already only the last-W rows (dropped on commit), and we
        # additionally apply the per-row window mask so a query only sees <W back.
        pk = past_kv
        # GLOBAL heads.
        if g_idx.numel() > 0:
            Kg_new, Vg_new = Knew[:, g_idx], Vnew[:, g_idx]
            if pk.Kg is not None:
                Kg = torch.cat([pk.Kg, Kg_new], dim=2)
                Vg = torch.cat([pk.Vg, Vg_new], dim=2)
                posg = torch.cat([pk.posg.to(x.device), q_pos], dim=0)
            else:
                Kg, Vg, posg = Kg_new, Vg_new, q_pos
            _attend_group(g_idx, Kg, Vg, posg, None)
        # LOCAL heads.
        if l_idx.numel() > 0:
            Kl_new, Vl_new = Knew[:, l_idx], Vnew[:, l_idx]
            if pk.Kl is not None:
                Kl = torch.cat([pk.Kl, Kl_new], dim=2)
                Vl = torch.cat([pk.Vl, Vl_new], dim=2)
                posl = torch.cat([pk.posl.to(x.device), q_pos], dim=0)
            else:
                Kl, Vl, posl = Kl_new, Vl_new, q_pos
            _attend_group(l_idx, Kl, Vl, posl, W)
    else:
        # MASK-ONLY fallback (or first span with no past): full KV, windowed READ.
        if past_kv is not None:
            K_cache, V_cache, pos_cache = past_kv
            K = torch.cat([K_cache, Knew], dim=2)
            Vv = torch.cat([V_cache, Vnew], dim=2)
            k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
        else:
            K, Vv, k_pos = Knew, Vnew, q_pos
        _attend_group(g_idx, K, Vv, k_pos, None)
        _attend_group(l_idx, K, Vv, k_pos, W)

    out = out.transpose(1, 2).contiguous().view(B, S, D)
    out = x + self.W_o.linear(out)
    if use_cache:
        # Return the NEW window rows only; the driver's commit slices the last-S tail
        # and (for a split cache) routes each head group into its own trimmed cache.
        return out, (Knew, Vnew, q_pos)
    return out


def _global_forward(self, x, past_kv, q_positions, use_cache):
    """The ORIGINAL global attention (kept for the fallback / uninstall path)."""
    return _ORIG_FORWARD(self, x, past_kv=past_kv, q_positions=q_positions,
                         use_cache=use_cache)


_ORIG_FORWARD = _SF.SparseAttn.forward


def install_local_attention(model, window: int = 64, slope_tol: float = 1e-3,
                            drop_local_kv: bool = True,
                            content_bound_global: bool = False,
                            verbose: bool = False) -> Dict[str, object]:
    """Install sliding-window attention on the LOCAL heads of every block.

    ``window`` = the sliding-window size W in TOKENS (must be >= the largest LOCAL
    head's measured window; the ingest heads are ≤ 28 = < one 30-token frame, so
    the default 64 = ~2 VM steps is safe).  The MEMORY/STACK/LEV heads stay GLOBAL.

    ``drop_local_kv`` (default True) is the VRAM lever: the local heads' OLD KV is
    physically DROPPED from the cache (only the last-W local rows are kept), so the
    cache collapses from ~H·S to ~Hg·S + Hl·W.  ``drop_local_kv=False`` is the
    MASK-ONLY fallback (``--local-mask-only``): the full KV is still stored and only
    the local heads' softmax READ is windowed — saves compute, not VRAM.

    ``content_bound_global`` (default False) bounds the GLOBAL cache BY CONTENT: each
    global head is an address-CAM that keys ONLY ``IS_STORE`` frames (a non-store
    frame's role-gate key -> softmax1 weight EXACTLY 0), so its cache keeps ONLY the
    store frames — bounded by the WORKING SET (distinct live addresses, after latest-
    write-wins eviction), NOT step count.  Byte-identical (dropped rows are provably
    inert).  This is what makes the TOTAL KV runtime-independent.

    Returns a summary dict (classification + windowed-head fraction).
    ``uninstall_local_attention(model)`` reverts to the global forward.
    """
    cls = classify_heads(model, slope_tol=slope_tol)
    cR = _store_gate_channel()
    n_local = n_global = 0
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        H = at.n_heads
        gmask = torch.zeros(H, dtype=torch.bool, device=at.alibi_slopes.device)
        for h in cls[bi]:
            gmask[h] = True
        at._local_window = int(window)
        at._global_head_mask = gmask
        at._drop_local_kv = bool(drop_local_kv)     # read by verify_blocks cache init
        at._content_bound_global = bool(content_bound_global)
        at._store_gate_channel = int(cR)
        # bind the windowed forward as a bound method on this instance.
        at.forward = windowed_forward.__get__(at, type(at))
        n_global += int(gmask.sum())
        n_local += H - int(gmask.sum())
    summary = {
        "window": int(window),
        "drop_local_kv": bool(drop_local_kv),
        "content_bound_global": bool(content_bound_global),
        "store_gate_channel": int(cR),
        "classification": cls,
        "n_local_head_slots": n_local,
        "n_global_head_slots": n_global,
        "frac_windowed": n_local / max(1, (n_local + n_global)),
    }
    if verbose:
        live_global = {bi: hs for bi, hs in cls.items() if hs}
        mode = "DROP-KV" if drop_local_kv else "MASK-ONLY"
        cb = "  +CONTENT-BOUND global (store-only)" if content_bound_global else ""
        print(f"[local-attn] {mode}  window={window}{cb}  live GLOBAL heads "
              f"(block->heads): {live_global}")
        print(f"[local-attn] windowed {n_local}/{n_local + n_global} head-slots "
              f"({summary['frac_windowed']*100:.1f}%) across {len(model.blocks)} blocks")
    return summary


def uninstall_local_attention(model) -> None:
    """Revert every block to the ORIGINAL global ``SparseAttn.forward``."""
    for blk in model.blocks:
        at = blk.attn
        for a in ("_local_window", "_global_head_mask", "_drop_local_kv",
                  "_content_bound_global", "_store_gate_channel"):
            if hasattr(at, a):
                delattr(at, a)
        if "forward" in at.__dict__:
            del at.__dict__["forward"]
