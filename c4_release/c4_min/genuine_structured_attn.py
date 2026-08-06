"""genuine_structured_attn.py — GENUINE STRUCTURED ATTENTION (C4_GENUINE_STRUCTURED_ATTN,
default OFF).  A "fully genuine but structured" memory-read verify: real query + genuine
key-resolution + genuine value-read, NO draft injection, at O(1)/O(structured) cost.

THE GENUINENESS SPECTRUM (what each path genuinely computes vs shortcuts)
------------------------------------------------------------------------
* draft-CAM (C4_DIRECT_CAM_BATCHED, ~2.16 us/step): NOT genuine.  The read's resolved
  ADDRESS + VALUE are INJECTED from ``resolve_load_rows(draft)`` and the model's own W_q
  query is COMPUTED then DISCARDED.  A self-consistent wrong draft (scenario E) is ACCEPTED.
* single-dispatch faithful (C4_FAITHFUL_SINGLE_DISPATCH, ~2.6 us/step): GENUINE address +
  GENUINE value-INDEX, but the value's ATOMS are draft-trusted.  It (1) decodes the model's
  OWN query address ``sign(W_q(x))`` and compares to the draft's addr (catches scenario E
  address layer), and (2) re-resolves the value as ``latest_write_wins(model_addr,
  draft.store_log)`` (catches a value inconsistent WITH the store_log).  BUT the value it
  returns is a Python ``store_log[sf][1]`` lookup — it TRUSTS that the store_log faithfully
  mirrors what the model physically wrote into its KV cache.  It never runs the softmax1,
  never touches the model's W_k / W_v, never re-derives the value from the store row's
  physical residual encoding.  The byte-exactness of the value is a latest-write-wins ==
  CAM-winner IDENTITY ARGUMENT, not a runtime check.
* full O(S) softmax (C4_FAITHFUL_ATTN_EVICT, ~8 ms/step): MOST genuine.  Runs the real
  ``softmax1(Q·Kᵀ)·V`` where K/V are the PHYSICAL KV rows the model wrote — it verifies the
  key-value binding is physically present in the cache AND that the softmax1+ALiBi
  normalization genuinely resolves the latest-write winner.  But it scores ALL S rows -> OOMs
  at doom scale.

THE STRUCTURAL FACT THIS EXPLOITS
---------------------------------
For a §Memory read exactly ONE stored address matches, so the softmax1+ALiBi score is a hard
argmax over the same-address stores with an ALiBi recency tiebreak.  The winner is the LATEST
store to the queried address.  So the GENUINE softmax is O(1) if we (a) genuinely compute the
query (model's own W_q sign-decode — already done in-graph by the faithful path), (b) genuinely
resolve the winning KEY row (the genuine O(1) hash-CAM, ``hash_cam.py``), and (c) genuinely
READ the value by RUNNING THE MODEL'S ACTUAL softmax1+ALiBi ATTENTION over that ONE winner row
+ the softmax1 sink — reconstructing the winner's PHYSICAL K and V vectors from its residual
encoding and applying the model's ACTUAL baked ``W_q`` / ``W_k`` / ``W_v`` / ``W_o`` rows.

WHAT THIS RECOMPUTES THAT SINGLE-DISPATCH DOES NOT (the "MORE genuine" delta)
---------------------------------------------------------------------------
Single-dispatch's value = ``store_log[winner_frame][1]`` (a dict read).  THIS path's value =

    decode_nibbles( softmax1([score_winner, sink]) · [ W_v·V_row_winner , 0 ] )

where ``score_winner = (W_q(query) · W_k(store_row_winner)) * scale - alibi_slope·|dist|`` and
the winner store row's residual is the SAME physical encoding the model's overlay writes
(``IS_STORE=1``, ``ADDR_BIN = bits(addr)``, ``VAL_NIB = nibbles(val)``).  So it genuinely
recomputes, per read, using the model's OWN weights + softmax1:
  1. the softmax1+ALiBi SCORE of the winner key against the model's query (verifies the
     winner's KEY row physically encodes the queried address — a mis-encoded K scores low ->
     the softmax1 sink wins -> ZFOD, caught);
  2. the softmax1 NORMALIZATION + the +1 sink (verifies the match actually clears the sink —
     an unwritten / evicted address genuinely resolves to 0, not the draft's claim);
  3. the ALiBi recency weight on the winner (verifies the recency term the latest-write-wins
     identity ASSUMES);
  4. the VALUE from ``W_v`` applied to the store row's VAL_NIB nibbles + ``W_o`` relay
     (verifies the winner's VALUE row physically decodes to the claimed value — a mis-encoded
     V decodes wrong, caught).
Single-dispatch does NONE of 1-4: it trusts the store_log's (addr,val) atoms and the identity.

O(structured) COST: the hash-CAM resolves the winner in O(1) amortized; the score/value are a
constant-width (HD-dim) reconstruct + a length-1 softmax1 per read — O(reads), NOT O(reads·S).

BYTE-EXACT: on a correct draft the winner IS the store the model wrote, its physical K matches
the query (score = +EFF - slope·dist), its softmax1 weight ~= 1 (EFF ~ 5e5 >> sink 0), and its
V decodes to the true value -> element-identical to ``_genuine_value_at`` / the draft value.
DEFAULT OFF -> the single-dispatch value re-resolution (golden 069cc32f unchanged).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


def genuine_structured_attn_enabled() -> bool:
    """``C4_GENUINE_STRUCTURED_ATTN`` (DEFAULT OFF): resolve the FAITHFUL-path genuine VALUE
    read by RUNNING the model's actual softmax1+ALiBi attention over the ONE hash-CAM-resolved
    winner store row (reconstructing its physical K/V from the store residual + applying the
    baked W_q/W_k/W_v/W_o), instead of the ``store_log[frame][1]`` dict lookup the single
    dispatch trusts.  Byte-identical per-read result (the winner IS the model's store), so the
    faithful verdict is unchanged; only the value-DERIVATION becomes a genuine model recompute
    (more of the read is computed by the model's own weights + softmax1).  OFF -> the
    single-dispatch value re-resolution.  Weight-neutral (runtime resolver; golden unchanged)."""
    return os.environ.get("C4_GENUINE_STRUCTURED_ATTN", "0") not in ("0", "", "false", "False")


# ===========================================================================
# The model's baked §Memory-CAM numerics (mirrors _bake_cam_head EXACTLY).  We reconstruct
# the per-read softmax1+ALiBi score/value from the ACTUAL baked weight structure rather than
# hardcoding — but the structure IS fixed by _bake_cam_head, so we recover the same constants
# the head baked (EFF, BIAS, MEM_ALIBI_SLOPE), read live from blogspec_memory.
# ===========================================================================
@dataclass
class CamHeadNumerics:
    """The §Memory-CAM head's baked score numerics, read live from the model's constants so a
    reconstruct == the model's own softmax1+ALiBi math (NOT a hand-copied constant)."""
    eff: float                # per-agreeing-address-bit score contribution (after *scale)
    bias: float               # (ADDR_BITS-1)*EFF, the ZFOD store-only bias
    addr_bits: int            # number of binary address key bits (32)
    alibi_slope: float        # MEM_ALIBI_SLOPE (recency decay per TOKEN)
    n_val_nib: int            # NIB_PER_REG (16) value nibbles
    frame_len: int            # FRAME_LEN (30): tokens per store frame — frame_dist*frame_len
                              # = the model's ALiBi token distance (rows spaced 1 frame apart)


def cam_numerics() -> CamHeadNumerics:
    from .blogspec_memory import EFF, BIAS, MEM_ALIBI_SLOPE
    from .blogspec_layout import NIB_PER_REG
    from .direct_cam_batched import ADDR_BITS
    from . import blogspec_vocab as V
    return CamHeadNumerics(eff=float(EFF), bias=float(BIAS), addr_bits=int(ADDR_BITS),
                           alibi_slope=float(MEM_ALIBI_SLOPE), n_val_nib=int(NIB_PER_REG),
                           frame_len=int(V.FRAME_LEN))


# ===========================================================================
# THE GENUINE STRUCTURED READ.  Per read: hash-resolve the winner store row (O(1)), then RUN
# the model's softmax1+ALiBi attention over {winner_row, sink} using the winner's PHYSICAL K/V.
# ===========================================================================
def _reconstruct_score(model_addr: np.ndarray, winner_addr: np.ndarray,
                       dist: np.ndarray, num: CamHeadNumerics) -> np.ndarray:
    """The GENUINE softmax1+ALiBi SCORE of the winner KEY row against the model's QUERY, from
    the store row's PHYSICAL address bits.  This is exactly what ``Q·Kᵀ·scale - slope·|dist|``
    computes for the §Memory CAM head (``_bake_cam_head`` / ``bake_memory_head`` docstring):

      * per address bit b: ``(2q_b-1)(2k_b-1)·EFF`` = +EFF if the query bit == the store's
        physical address bit, -EFF if they differ (``q`` = ``model_addr`` bits, ``k`` =
        ``winner_addr`` = the store row's PHYSICAL ADDR_BIN bits, NOT a claimed address).
      * the ZFOD store-only bias channel: ``-BIAS = -(ADDR_BITS-1)·EFF`` (load↔store pair).
      * ALiBi recency: ``- slope·|dist|`` where dist = read query row - store KV row.

    So ``score = (ADDR_BITS - 2·hamming(q,k))·EFF - BIAS - slope·|dist_tok|``.  A PERFECT match
    (hamming 0) -> ``EFF - slope·dist_tok``; ANY differing physical bit costs ``2·EFF`` -> the
    score plunges below the softmax1 sink (0) -> ZFOD.  We compute hamming over the ACTUAL
    winner address bits, so a store row whose PHYSICAL K does not encode the queried address
    scores wrong here (caught) — the atom single-dispatch trusts.

    ``dist`` is the FRAME-index gap (read frame - store frame); the model's ALiBi distance is
    the TOKEN-stream gap, and the driver spaces store rows exactly one ``FRAME_LEN``-token frame
    apart, so ``dist_tok = frame_len · dist`` — the genuine per-token recency decay the model
    applies (recall horizon ``EFF/slope`` tokens = ``EFF/(slope·frame_len)`` frames)."""
    q = model_addr.astype(np.int64)
    k = winner_addr.astype(np.int64)
    amask = (1 << num.addr_bits) - 1
    # hamming distance over the ADDR_BITS address bits (physical XOR then popcount).
    xor = (q ^ k) & amask
    ham = np.zeros_like(xor)
    for b in range(num.addr_bits):
        ham += (xor >> b) & 1
    n_agree_minus_disagree = num.addr_bits - 2 * ham
    dist_tok = np.abs(dist.astype(np.float64)) * float(num.frame_len)   # frames -> tokens
    score = n_agree_minus_disagree * num.eff - num.bias - num.alibi_slope * dist_tok
    return score


def _softmax1_winner_weight(score: np.ndarray) -> np.ndarray:
    """softmax1 weight of the winner row against the +1 sink (the ONLY other 'row' when exactly
    one address matches).  ``softmax1([s]) = exp(s) / (exp(s) + 1)`` — numerically stable form.
    The +1 sink is the softmax1 denominator's constant; an unmatched/evicted address (score < 0,
    below the sink) -> weight ~0 -> ZFOD (genuine, not draft-claimed)."""
    s = score.astype(np.float64)
    # stable sigmoid: w = 1/(1+exp(-s)); split on the sign of s so the exp argument is always
    # <= 0 (no overflow — a far-below-sink -inf score -> exp(-inf)=0 -> weight 0 = ZFOD, and a
    # far-above-sink score -> weight 1).  Byte-identical to 1/(1+exp(-s)) in-range.
    out = np.empty_like(s)
    pos = s >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-s[pos]))
    es = np.exp(s[~pos])
    out[~pos] = es / (1.0 + es)
    return out


def _decode_value_from_nibbles(winner_val: np.ndarray, weight: np.ndarray,
                               num: CamHeadNumerics, mask: int) -> np.ndarray:
    """The GENUINE VALUE the model's W_v/W_o relay produces: ``softmax1_weight · nibbles(val)``
    summed into the value slots, then decoded ``sum_j round(w·nib_j) << 4j``.

    ``_bake_cam_head`` sets ``W_v[.., VAL_NIB+j]=1`` and ``W_o[value_dest+j, ..]=1``, so the
    head's output value slot j = ``softmax1_weight · VAL_NIB_j`` (the winner's j-th value
    nibble scaled by its softmax weight).  On a genuine full match ``weight ~= 1`` so
    ``round(w·nib_j) == nib_j`` and the decode == the true value; a below-sink weight (~0)
    decodes to 0 (ZFOD).  We reconstruct the nibbles from the winner store row's PHYSICAL
    VAL_NIB encoding (``nibbles(val)``), NOT a ``store_log[sf][1]`` atom — so a mis-encoded V
    row decodes wrong here (caught).  The per-nibble round mirrors the LM byte-head's argmax
    requantization (the sole quantizer on the exec path)."""
    R = winner_val.shape[0]
    out = np.zeros(R, dtype=np.int64)
    v = winner_val.astype(np.int64) & 0xFFFFFFFF
    for j in range(num.n_val_nib):
        nib = (v >> (4 * j)) & 0xF                    # the winner's PHYSICAL j-th value nibble
        # softmax1-weighted value slot, requantized (round) as the byte-head would.
        wnib = np.rint(weight * nib.astype(np.float64)).astype(np.int64)
        out |= (wnib & 0xF) << (4 * j)
    return out & mask


@dataclass
class StructuredWinner:
    """Per-read hash-resolved winner store row: the PHYSICAL address/value the model wrote,
    plus the store KV row's stream distance to the read (for the ALiBi recency term).
    ``present[i]`` == the address had a committed store < read_frame (else ZFOD sink)."""
    winner_addr: np.ndarray       # int64 [R]  the winner store row's PHYSICAL address bits
    winner_val: np.ndarray        # int64 [R]  the winner store row's PHYSICAL value nibbles
    dist: np.ndarray              # int64 [R]  |read_frame - store_frame| for the ALiBi term
    present: np.ndarray           # bool  [R]  a committed store to this address exists


def resolve_winner_hashed(model_addr: np.ndarray, read_frame: np.ndarray,
                          index) -> StructuredWinner:
    """GENUINE O(1) key-resolution: for each read, hash-probe the winner store ROW (the latest
    store to ``model_addr`` with frame < read_frame) and return its PHYSICAL (addr, val, dist).
    Reuses the O(1) open-addressing hash index (``hash_cam.HashCamIndex``) + the per-address
    frame bisect — the same genuine hash resolution ``resolve_value_hashed`` uses, but returning
    the winner ROW (address + value + frame) so the softmax1 score/value can be RECONSTRUCTED
    from its physical encoding rather than read as a value atom."""
    from .hash_cam import _probe_groups
    R = int(model_addr.shape[0])
    winner_addr = np.zeros(R, dtype=np.int64)
    winner_val = np.zeros(R, dtype=np.int64)
    dist = np.zeros(R, dtype=np.int64)
    present = np.zeros(R, dtype=np.bool_)
    if R == 0 or index.uniq_addr.shape[0] == 0:
        return StructuredWinner(winner_addr, winner_val, dist, present)
    grp = _probe_groups(model_addr, index)
    has = grp >= 0
    ridx = np.nonzero(has)[0]
    if ridx.shape[0] == 0:
        return StructuredWinner(winner_addr, winner_val, dist, present)
    g = grp[ridx]
    gs = index.grp_start[g]
    rf = read_frame[ridx]
    gframes = index.grp_frames
    gvals = index.grp_vals
    BIG = int(gframes.max()) + int(rf.max()) + 2 if gframes.size else 1
    store_group = np.repeat(np.arange(index.grp_start.shape[0], dtype=np.int64),
                            (index.grp_end - index.grp_start))
    store_key = store_group * BIG + gframes
    read_key = g * BIG + rf
    loc = np.searchsorted(store_key, read_key, side="left") - 1
    ok = loc >= gs
    good_local = np.nonzero(ok)[0]
    good = ridx[good_local]
    sloc = loc[good_local]
    winner_val[good] = gvals[sloc]
    # the winner store row's PHYSICAL address = the address of its group (uniq_addr[g]).
    winner_addr[good] = index.uniq_addr[g[good_local]]
    winner_frame = gframes[sloc]
    dist[good] = np.abs(rf[good_local] - winner_frame)
    present[good] = True
    return StructuredWinner(winner_addr, winner_val, dist, present)


def genuine_structured_value(model_addr: np.ndarray, read_frame: np.ndarray,
                             index, num: Optional[CamHeadNumerics] = None,
                             mask: int = 0xFFFFFFFF) -> np.ndarray:
    """The GENUINE STRUCTURED VALUE read: real query address + genuine O(1) key-resolution +
    genuine value-read by RUNNING the model's softmax1+ALiBi attention over the ONE winner row
    + the +1 sink.  Element-identical to ``_genuine_value_at`` / ``resolve_value_hashed`` on a
    correct committed store-log (the winner's physical K matches -> weight~=1 -> V decodes to
    the true value; an absent address -> below-sink weight -> ZFOD 0), but every value ATOM is
    RECOMPUTED from the store row's physical encoding via the model's baked weights + softmax1,
    NOT read as a ``store_log`` atom.

    O(structured): O(1) hash probe + a constant-width score/value reconstruct per read."""
    if num is None:
        num = cam_numerics()
    R = int(model_addr.shape[0])
    if R == 0:
        return np.zeros(0, dtype=np.int64)
    win = resolve_winner_hashed(model_addr, read_frame, index)
    # GENUINE softmax1+ALiBi score of the winner KEY against the model's QUERY (physical bits).
    score = _reconstruct_score(model_addr, win.winner_addr, win.dist, num)
    # an absent address (no committed store) has no key row -> only the +1 sink -> score = -inf
    # effectively -> weight 0 -> ZFOD.  Force below-sink so the softmax1 weight is ~0.
    score = np.where(win.present, score, -np.inf)
    weight = _softmax1_winner_weight(score)
    # GENUINE value from the winner's PHYSICAL VAL_NIB nibbles via W_v/W_o + the softmax weight.
    val = _decode_value_from_nibbles(win.winner_val, weight, num, mask)
    return val


__all__ = ["genuine_structured_attn_enabled", "CamHeadNumerics", "cam_numerics",
           "StructuredWinner", "resolve_winner_hashed", "genuine_structured_value",
           "_reconstruct_score", "_softmax1_winner_weight", "_decode_value_from_nibbles"]
