"""TRUE banded (sliding-window) local-attention kernel — O(S*W), NOT O(S*Sk).

Motivation (#780 follow-up, agent a2ad44e5's MEASURED wall).  ``local_attention.
windowed_forward`` already keeps the LOCAL heads windowed, but it does so by
computing the FULL ``Q@Kᵀ`` score matrix ``[B, Hl, Sq, Sk]`` and THEN masking the
out-of-band entries to ``-inf``.  That materialised matrix is O(Sq·Sk) — and because
within a single verify span (or a prime chunk) EVERY token is both a query AND a key
(``Sk ≈ Sq + W``), it is quadratic in the span length S: the 4 live doom blocks pay
~62-75 % of the forward on it (S=3000 → 34 ms, 6000 → 128 ms, 12000 OOMs a 12.9 GiB
score) even though each local query truly needs only the last ``W`` keys.

THIS kernel scores ONLY the last-W keys per query row — a genuine sliding BAND — so
the cost is O(Sq·W), FLAT in S (W is a fixed 96 by default; the true horizon is
≤ 28 tok).

BYTE-EXACT by construction.  For a local head the true attention weight past the
window is EXACTLY 0 (softmax1 + the huge exact-match/role scores + ALiBi recency drive
the tail to ZFOD — see ``local_attention`` module docstring; local == global is
byte-identical for a proven-local ingest head).  The ``windowed_forward`` path ALREADY
masks those keys to ``-inf`` (weight 0); we simply never SCORE them.  The kept keys,
their ALiBi distances, their causal mask, and the softmax1 sink are IDENTICAL to the
masked-full path, so the post-softmax weights on the kept keys — and thus the context
``a @ V`` — are bit-for-bit the same (modulo fp reduction order over a SHORTER row,
which for a ZFOD tail is exactly-zero terms).

INDEXING / ALiBi CORRECTNESS ARGUMENT.
  * Keys carry their TRUE ABSOLUTE positions ``kpos_full`` (same tensor the full path
    uses for the ALiBi distance ``|q_pos - k_pos|``).  The band gather selects a
    contiguous slice of key INDICES per query, and each selected key keeps its own
    absolute position, so the ALiBi penalty ``slope * |q_pos - k_pos|`` is computed on
    the true distance — identical to the full path.
  * The band for a query at absolute position ``P`` is the key set with ``P-W < k_pos
    ≤ P`` (causal + window).  Because the key positions are a CONTIGUOUS ascending run
    (the KV cache / span is a contiguous prefix of the token stream — priming commits
    contiguous chunks, ``_trim_local`` keeps a contiguous suffix, and a span's new rows
    are ``arange(span_start, span_start+S)`` appended to that suffix), the in-band keys
    are a contiguous slice ``[lo(q) .. hi(q)]`` of the key array.  We gather a fixed-
    width ``[Sq, W]`` window of indices ending at each query's own key index and mask
    the (few, only near the sequence start) entries that fall before position 0 or
    before the cache start.  Contiguity is EXPLOITED for speed but the mask makes the
    result correct even if a gap ever appears (a masked-out band slot scores -inf ==
    the full path's out-of-band -inf).

If a 'local' head ever ACTUALLY reaches past W for some query (a real weight on a key
older than W), the band would drop it and the result would DIFFER — which the
byte-exact gate catches.  The install-time classifier only windows a LIVE head with the
large ingest recency slope (measured window ≤ 28 < W=96), so this cannot happen for the
frame-ingest heads; every other windowed head is zero-value (window-invariant output).

Gated by ``C4_BANDED_LOCAL_ATTN`` (checked in ``local_attention.windowed_forward``);
default OFF -> the masked-full path is used and the golden is unchanged.
"""
from __future__ import annotations

import torch

from .blogspec_model import softmax1


def banded_local_context(Qg: torch.Tensor, Ksel: torch.Tensor, Vsel: torch.Tensor,
                         q_pos: torch.Tensor, kpos_full: torch.Tensor,
                         slopes_g: torch.Tensor, scale: float, window: int
                         ) -> torch.Tensor:
    """O(Sq*W) sliding-window softmax1+ALiBi attention for the LOCAL heads.

    Args (all this GROUP's heads only, index 0..Hl-1):
      Qg          [B, Hl, Sq, HD]   queries
      Ksel        [B, Hl, Sk, HD]   keys (cache + new span, this group's heads)
      Vsel        [B, Hl, Sk, HD]   values
      q_pos       [Sq]  long        absolute position of each query row
      kpos_full   [Sk]  long        absolute position of each key row (ascending)
      slopes_g    [Hl]  float       ALiBi slope per head (of this group)
      scale       float             1/sqrt(HD)
      window      int               W (drop keys with q_pos - k_pos >= W)

    Returns the context ``[B, Hl, Sq, HD]`` — bit-identical to the masked-full path on
    the kept (in-band) keys.
    """
    B, Hl, Sq, HD = Qg.shape
    Sk = Ksel.shape[2]
    dev = Qg.device
    W = int(window)

    # ---- per-query band of key INDICES (contiguous, exploiting ascending kpos) ----
    # For query q at abs pos P = q_pos[q], the in-band keys are those with
    #   P - W < kpos <= P   (causal + window).
    # kpos_full is ascending, so this is the index range [lo, hi]:
    #   hi(q) = last key index with kpos <= P            (the query's own / latest key)
    #   lo(q) = first key index with kpos >  P - W       (= hi - actual_band + 1)
    # We compute hi by searchsorted (works even with a non-unit stride / gaps), then
    # gather a fixed-width-W window of indices ENDING at hi(q).  Slots that fall before
    # lo(q) (older than W) or before index 0 are masked out.
    kp = kpos_full
    P = q_pos  # [Sq]
    # hi(q): number of keys with kpos <= P, minus 1  ==  rightmost in-band index.
    hi = torch.searchsorted(kp, P, right=True) - 1          # [Sq] long, in [-1, Sk-1]
    # A query with hi < 0 has NO causal key (shouldn't happen for a committed row);
    # clamp and let the mask zero it.
    hi_c = hi.clamp(min=0)
    # fixed-width band of indices [hi-W+1 .. hi], shape [Sq, W]; clamp to [0, Sk-1] and
    # mask the clamped/out-of-range slots.
    off = torch.arange(W - 1, -1, -1, device=dev)           # [W]  = W-1, W-2, ..., 0
    band_idx = hi_c.unsqueeze(1) - off.unsqueeze(0)         # [Sq, W]  index per slot
    valid = band_idx >= 0                                    # slot exists in the array
    band_idx_c = band_idx.clamp(min=0)                       # [Sq, W]  safe gather idx

    # gather the band's key positions -> the ALiBi distance + window/causal mask.
    band_kpos = kp[band_idx_c]                               # [Sq, W]
    dist = P.unsqueeze(1) - band_kpos                        # [Sq, W]  signed
    # in-band iff 0 <= dist < W  AND the slot is a real key AND hi(q) >= 0.
    keep = valid & (dist >= 0) & (dist < W) & (hi.unsqueeze(1) >= 0)   # [Sq, W]

    # ---- gather the banded K/V and score ----
    # band_idx_c [Sq, W] shared across (B, Hl) -> [B, Hl, Sq, W, HD] via index_select.
    Kb = _gather_band(Ksel, band_idx_c, B, Hl, Sq, W, HD)     # [B, Hl, Sq, W, HD]
    Vb = _gather_band(Vsel, band_idx_c, B, Hl, Sq, W, HD)     # [B, Hl, Sq, W, HD]

    # score[b,h,q,w] = (Qg[b,h,q,:] . Kb[b,h,q,w,:]) * scale
    sc = torch.einsum("bhqd,bhqwd->bhqw", Qg, Kb) * scale     # [B, Hl, Sq, W]
    # ALiBi: subtract slope * |dist| (dist>=0 in band).
    ali = slopes_g.view(1, Hl, 1, 1) * dist.abs().to(sc.dtype).view(1, 1, Sq, W)
    sc = sc - ali
    sc = sc.masked_fill(~keep.view(1, 1, Sq, W), float("-inf"))
    a = softmax1(sc, dim=-1)                                  # [B, Hl, Sq, W]
    ctx = torch.einsum("bhqw,bhqwd->bhqd", a, Vb)             # [B, Hl, Sq, HD]
    return ctx


def _gather_band(T: torch.Tensor, band_idx_c: torch.Tensor,
                 B: int, Hl: int, Sq: int, W: int, HD: int) -> torch.Tensor:
    """Gather ``T[B,Hl,Sk,HD]`` along Sk at ``band_idx_c[Sq,W]`` -> ``[B,Hl,Sq,W,HD]``.

    ``band_idx_c`` is shared across (B, Hl), so we index_select the flat [Sq*W] index
    list along the Sk axis (memory-cheap: no [Sk,Sq] expand) and reshape.
    """
    Sk = T.shape[2]
    flat = band_idx_c.reshape(-1)                            # [Sq*W]
    g = T.index_select(2, flat)                              # [B, Hl, Sq*W, HD]
    return g.view(B, Hl, Sq, W, HD)
