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

    The KV CACHE (``past_kv`` / the returned ``(K, V, k_pos)``) is UNCHANGED — the
    full projected K/V of every position is still stored (the global heads need it,
    and the commit path is shared).  Only the SCORE/softmax READ for local heads is
    restricted to the window, which is where the O(S²) cost lives.
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

    if past_kv is not None:
        K_cache, V_cache, pos_cache = past_kv
        K = torch.cat([K_cache, Knew], dim=2)                   # [B,H,Sk,HD]
        Vv = torch.cat([V_cache, Vnew], dim=2)
        k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
    else:
        K, Vv, k_pos = Knew, Vnew, q_pos

    Sk = K.shape[2]
    # --- GLOBAL heads: full causal over ALL keys (unchanged math) --------------
    g_idx = torch.nonzero(gmask, as_tuple=False).flatten()
    l_idx = torch.nonzero(~gmask, as_tuple=False).flatten()
    out = x.new_zeros(B, H, S, HD)

    def _attend(idx, K_sub, V_sub, kpos_sub):
        """Compute the softmax1+ALiBi attention output for the heads in ``idx``
        over the key set ``K_sub``/``V_sub`` at absolute positions ``kpos_sub``."""
        if idx.numel() == 0:
            return
        Qg = Q[:, idx]                                          # [B,g,S,HD]
        sc = torch.matmul(Qg, K_sub[:, idx].transpose(-2, -1)) * self.scale
        dist = (q_pos.unsqueeze(1) - kpos_sub.unsqueeze(0)).abs().float()  # [S,Sk']
        sc = sc - self.alibi_slopes[idx].view(1, -1, 1, 1) * dist.unsqueeze(0)
        mask = (kpos_sub.unsqueeze(0) > q_pos.unsqueeze(1))     # future keys
        sc = sc.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        a = softmax1(sc, dim=-1)
        out[:, idx] = torch.matmul(a, V_sub[:, idx])

    # global: all keys.
    _attend(g_idx, K, Vv, k_pos)

    # local: only the last-W keys by ABSOLUTE position.  A query at q_pos attends
    # to keys with q_pos - k_pos < W (and k_pos <= q_pos for causality).  Across the
    # whole span the union of needed keys is [q_pos.min() - W + 1 .. Sk-1]; slice to
    # that contiguous suffix (the local heads never look further back), then apply
    # the per-row window inside ``_attend`` via the causal + ALiBi + the extra
    # window mask below.  This is where O(S²) -> O(S·W).
    if l_idx.numel() > 0:
        qmin = int(q_pos.min().item())
        cutoff = qmin - W + 1                                   # earliest needed abs pos
        keep = (k_pos >= cutoff)
        if bool(keep.all()):
            # window covers the whole cache already (short span) -> plain local mask.
            Ksub, Vsub, kpos_sub = K, Vv, k_pos
        else:
            keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
            Ksub = K[:, :, keep_idx, :]
            Vsub = Vv[:, :, keep_idx, :]
            kpos_sub = k_pos[keep_idx]
        # per-row window: drop keys with (q_pos - k_pos) >= W (older than window).
        Qg = Q[:, l_idx]
        sc = torch.matmul(Qg, Ksub[:, l_idx].transpose(-2, -1)) * self.scale
        dist = (q_pos.unsqueeze(1) - kpos_sub.unsqueeze(0)).float()   # signed [S,Sk']
        adist = dist.abs()
        sc = sc - self.alibi_slopes[l_idx].view(1, -1, 1, 1) * adist.unsqueeze(0)
        future = (kpos_sub.unsqueeze(0) > q_pos.unsqueeze(1))
        too_old = (dist >= W)                                   # strictly outside window
        m = (future | too_old).unsqueeze(0).unsqueeze(0)
        sc = sc.masked_fill(m, float("-inf"))
        a = softmax1(sc, dim=-1)
        out[:, l_idx] = torch.matmul(a, Vsub[:, l_idx])

    out = out.transpose(1, 2).contiguous().view(B, S, D)
    out = x + self.W_o.linear(out)
    if use_cache:
        return out, (K, Vv, k_pos)
    return out


def _global_forward(self, x, past_kv, q_positions, use_cache):
    """The ORIGINAL global attention (kept for the fallback / uninstall path)."""
    return _ORIG_FORWARD(self, x, past_kv=past_kv, q_positions=q_positions,
                         use_cache=use_cache)


_ORIG_FORWARD = _SF.SparseAttn.forward


def install_local_attention(model, window: int = 64, slope_tol: float = 1e-3,
                            verbose: bool = False) -> Dict[str, object]:
    """Install sliding-window attention on the LOCAL heads of every block.

    ``window`` = the sliding-window size W in TOKENS (must be >= the largest LOCAL
    head's measured window; the ingest heads are ≤ 28 = < one 30-token frame, so
    the default 64 = ~2 VM steps is safe).  The MEMORY/STACK/LEV heads stay GLOBAL.

    Returns a summary dict (classification + windowed-head fraction).
    ``uninstall_local_attention(model)`` reverts to the global forward.
    """
    cls = classify_heads(model, slope_tol=slope_tol)
    n_local = n_global = 0
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        H = at.n_heads
        gmask = torch.zeros(H, dtype=torch.bool, device=at.alibi_slopes.device)
        for h in cls[bi]:
            gmask[h] = True
        at._local_window = int(window)
        at._global_head_mask = gmask
        # bind the windowed forward as a bound method on this instance.
        at.forward = windowed_forward.__get__(at, type(at))
        n_global += int(gmask.sum())
        n_local += H - int(gmask.sum())
    summary = {
        "window": int(window),
        "classification": cls,
        "n_local_head_slots": n_local,
        "n_global_head_slots": n_global,
        "frac_windowed": n_local / max(1, (n_local + n_global)),
    }
    if verbose:
        live_global = {bi: hs for bi, hs in cls.items() if hs}
        print(f"[local-attn] window={window}  live GLOBAL heads (block->heads): "
              f"{live_global}")
        print(f"[local-attn] windowed {n_local}/{n_local + n_global} head-slots "
              f"({summary['frac_windowed']*100:.1f}%) across {len(model.blocks)} blocks")
    return summary


def uninstall_local_attention(model) -> None:
    """Revert every block to the ORIGINAL global ``SparseAttn.forward``."""
    for blk in model.blocks:
        at = blk.attn
        if hasattr(at, "_local_window"):
            del at._local_window
        if hasattr(at, "_global_head_mask"):
            del at._global_head_mask
        if "forward" in at.__dict__:
            del at.__dict__["forward"]
