"""Shared I/O position substrate: multi-slope BOS position signatures + the
base-16 nibble-cascade offset extractor (BLOG_SPEC §"Printing and Reading
Input" 696-717, §"Position Offset Calculation" 718-739).

This is the reusable machinery underneath GETCHAR / PUTCHAR / argv (§751-793,
§851): the VM needs, for every I/O token, its *absolute sequence position* and,
given a target offset ``N``, the *byte at position N* in the I/O buffer. The
spec builds this out of two vanilla-transformer pieces, both implemented here:

1. Position signature — multi-ALiBi-slope BOS (§704-712)
--------------------------------------------------------
    "The specific mechanism uses multiple attention heads sharing the same fixed
     key at BOS but with different ALiBi slopes. For a token at distance d from
     BOS, head k with slope m_k produces an attention bias of −m_k·d, giving a
     score contribution of exp(−m_k·d). With enough heads at different slopes,
     the tuple (exp(−m_1·d), …, exp(−m_K·d)) uniquely identifies position d …
     To retrieve the byte at position N, we construct a query that matches the
     exponential signature of distance N."                        (§710)

   :func:`position_signature` returns that tuple. Because ``m_k>0`` and the
   slopes are distinct, ``d ↦ signature(d)`` is strictly monotone in each
   component and injective, so the tuple *is* an absolute-position code. The
   signature is what a query carries; a store token carries the *same* tuple for
   its own distance, and the softmax1 head (§491) picks the store whose distance
   matches the query's — that is how a byte is fetched *by position*.

   Positions are ALiBi *distance from the BOS/marker*, NOT the KV slot index
   (§712): evicting non-I/O tokens leaves the I/O tokens' distances unchanged.

2. Offset extractor — base-16 nibble cascade (§726-730)
-------------------------------------------------------
   The offset ``N = pos − marker_pos`` arrives as a *scalar*. To index memory it
   must become explicit nibbles. The spec's shallow (8-layer) extractor works in
   base 16: at nibble ``j`` (from ``j=7`` down to ``0``)

       d_j = Σ_{k=1..15} step(r − k·16^j),   r ← r − d_j·16^j            (§726)

   Each layer counts "how many copies of ``16^j`` fit in the residual" — the
   same threshold-counting staircase DIV uses for its quotient digit (§728), so
   :func:`nibble_cascade_offset` reuses the shared SiLU ``step`` primitive. The
   digit-emit and residual-subtract share one threshold bank (§730): one output
   weight ``1`` writes ``d_j``, another ``16^j`` updates ``r``.

Everything is a pure function of integers + the vanilla softmax1 head; no
learned parameters beyond the ALiBi slopes (which are a fixed geometric ladder,
exactly as ``blogspec_model.Attn`` bakes them, §307-311).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scales.  ``BOS_KEY_SCALE`` is the shared BOS key/query magnitude (a large
# positive dot product so the BOS sink dominates over the softmax1 ``+1`` and
# over unrelated tokens, §704 "overwriting previous ones … significant slope").
# ``RELU_S`` matches ``nibble_muldivmod`` — the reused threshold-count scale.
# ---------------------------------------------------------------------------
BOS_KEY_SCALE = 30.0
RELU_S = 200.0        # SiLU-as-ReLU scale (fp32-exact for byte-bounded steps)

# softmax1 ZFOD match gain (§491).  The raw position-match logit is
# ``−Σm_k·|d_q − d_s|`` — exactly 0 at the matched position, negative elsewhere.
# A bare 0-logit ties the softmax1 ``+1`` sink (both weigh ``exp(0)=1``), so the
# match never wins dominant weight.  Adding ``MATCH_GAIN`` lifts the EXACT match
# to a strongly-positive ``+MATCH_GAIN`` while every non-match, penalised by the
# ALiBi slope sum per unit of distance, lands below it and (for a large enough
# gain vs a fine-enough slope) below 0 so the sink out-weights an out-of-range
# read.  This is the analogue of the RoPE content key's ``−scale`` mismatch bias
# (nibble_rope.load_softmax1) and the spec's binary-key ``−scale`` per 0-bit.
MATCH_GAIN = 30.0

# The nibble cascade's high thresholds reach ``15·16^7 ≈ 4·10^9`` (§734), past
# fp32's 2^24 unit precision, so — with the spec's blessing ("We can of course
# use doubles", §595) — the staircase runs in float64.  The *scale* is reduced
# there too (§734: "Large thresholds force smaller usable S"); at the top nibble
# adjacent thresholds are spaced ``16^7`` apart so even a tiny slope separates
# them, while at ``j=0`` the unit spacing needs a sharp slope (§736).  We pick a
# per-nibble scale so ``S·(one threshold-step)`` stays O(1)–O(hundreds).
CASCADE_S_LOW = 200.0     # j==0: unit-spaced thresholds need a sharp step
CASCADE_S_HIGH = 8.0      # j>=1: wide spacing, gentler step keeps S·τ tame


# ===========================================================================
# SiLU step primitive (shared with nibble_muldivmod's threshold counting)
# ===========================================================================
def _relu(z, s: float = RELU_S, dtype=torch.float64):
    """Clamped ReLU ``silu(s·z)/s`` — exact ``max(0, z)`` for integer ``z``."""
    z = torch.as_tensor(z, dtype=dtype)
    return F.silu(s * z) / s


def _step_ge(x, t, s: float = RELU_S, dtype=torch.float64):
    """Exact ``[x >= t]`` via ``relu(x−(t−1)) − relu(x−t)`` (§Division step).

    ``t`` may be a scalar or a 1-D tensor of thresholds (a batched staircase).
    """
    return _relu(x - (t - 1), s, dtype) - _relu(x - t, s, dtype)


# ===========================================================================
# 1.  Multi-slope BOS position signature  (§704-712)
# ===========================================================================
def alibi_slopes(n_heads: int) -> List[float]:
    """The geometric ALiBi slope ladder ``m_k = 2^(-8/n·(k+1))`` (§307-311).

    Identical to ``blogspec_model.Attn.alibi_slopes`` — the position-signature
    heads ARE ordinary ALiBi heads; we just read their per-head distance decay
    ``exp(−m_k·d)`` as a position code instead of using it only for recency.
    """
    return [2.0 ** (-8.0 / n_heads * (k + 1)) for k in range(n_heads)]


def position_signature(d: int, n_heads: int = 8,
                       slopes: Sequence[float] | None = None) -> torch.Tensor:
    """The tuple ``(exp(−m_1·d), …, exp(−m_K·d))`` for distance ``d`` (§710).

    Each component is head ``k``'s softmax1 *score contribution* for a token at
    distance ``d`` from the shared BOS key: the query·key dot product is a fixed
    constant (same BOS key every head), and ALiBi subtracts ``m_k·d``, so the
    pre-softmax logit is ``const − m_k·d`` and the un-normalised weight is
    ``exp(const)·exp(−m_k·d)`` — up to the shared ``exp(const)`` this is the
    signature.  Distinct positive slopes make ``d ↦ signature`` injective.
    """
    if slopes is None:
        slopes = alibi_slopes(n_heads)
    return torch.tensor([math.exp(-m * d) for m in slopes], dtype=torch.float64)


def invertible_range(n_heads: int = 8,
                     slopes: Sequence[float] | None = None) -> int:
    """Largest distance ``d`` whose signature is still invertible in float64.

    The gentlest-slope head carries the position for the largest ``d`` (§736 the
    steep heads have already decayed); its component ``exp(−m_min·d)`` leaves the
    *normal* float64 range once ``m_min·d`` exceeds ``~708`` (``exp(−708) ≈
    2.2e-308``, the smallest normal double).  Past that the value is subnormal —
    it still carries the position for a while but with a shrinking mantissa, so
    the exact-integer inverse is only guaranteed while the component is normal.
    So the reliably-invertible range is ``d_max = floor(708 / m_min)``.  A
    *gentler* smallest slope (pass ``slopes`` with a smaller ``m_min``) widens
    this — the spec's "with enough heads at different slopes" (§710) covers the
    *distinguishing* power; the *range* is set by the smallest slope, tunable
    independently.
    """
    if slopes is None:
        slopes = alibi_slopes(n_heads)
    m_min = min(slopes)
    return int(708.0 / m_min)


def position_from_signature(sig: Sequence[float], n_heads: int = 8,
                            slopes: Sequence[float] | None = None,
                            max_pos: int | None = None) -> int:
    """Invert :func:`position_signature`: recover the integer distance ``d``.

    The gentlest-slope head decays slowly enough to stay numerically alive over
    the whole range, so ``d = −ln(sig_min_slope)/m_min_slope`` reads the position
    straight off the least-decayed component (rounded to the nearest integer).
    The steeper heads then *confirm* it (their exponentials agree only for the
    true ``d``) — this is the "tuple uniquely identifies position" claim (§710),
    made operational.  ``max_pos`` clamps to the encodable range.
    """
    if slopes is None:
        slopes = alibi_slopes(n_heads)
    if max_pos is None:
        max_pos = invertible_range(n_heads, slopes)
    # Use the smallest slope (slowest decay) — its exp stays above the fp floor
    # for the largest positions, giving the cleanest inverse (§736).
    k = min(range(len(slopes)), key=lambda i: slopes[i])
    m = slopes[k]
    val = float(sig[k])
    if val <= 0.0:                       # underflowed -> out of invertible range
        return max_pos
    d = -math.log(val) / m
    return max(0, min(max_pos, int(round(d))))


# ===========================================================================
# 2.  Base-16 nibble-cascade offset extractor  (§726-730)
# ===========================================================================
def nibble_cascade_offset(offset: int, n_nibbles: int = 8
                          ) -> Tuple[List[int], List[int]]:
    """Extract ``offset``'s base-16 digits with the spec's 8-layer cascade.

    Runs ``j = n_nibbles-1 … 0``.  At each layer the digit is the count of
    place-value multiples in the residual and the residual is decremented by the
    digit's contribution — both from ONE threshold bank (§730)::

        d_j = Σ_{k=1..15} step(r − k·16^j)          (digit, output weight 1)
        r  ← r − d_j·16^j                            (residual, output weight 16^j)

    Returns ``(digits, residuals)`` where ``digits[j]`` is nibble ``j`` (little-
    endian: ``digits[0]`` least significant) and ``residuals`` is the running
    residual after each processed layer (MSB-first), so ``residuals[-1] == 0``
    for a well-formed ``n_nibbles``-wide offset.  All staircase arithmetic is
    float64 (§734 numerical margin).
    """
    r = int(offset) & ((1 << (4 * n_nibbles)) - 1)
    digits = [0] * n_nibbles
    residual_trace: List[int] = []
    for j in range(n_nibbles - 1, -1, -1):
        place = 16 ** j
        # Per-nibble scale: sharp at the unit-spaced low nibble, gentle above
        # (§736 — hardest classification is at j==0, not the top).
        s = CASCADE_S_LOW if j == 0 else CASCADE_S_HIGH
        # d_j = Σ_{k=1..15} step(r − k·place).  Only thresholds ``<= r`` fire, so
        # the batch can be trimmed, but 15 is tiny — keep the full spec bank.
        thresholds = torch.tensor([k * place for k in range(1, 16)],
                                  dtype=torch.float64)
        digit = int(round(float(_step_ge(float(r), thresholds, s).sum())))
        digit = max(0, min(15, digit))
        digits[j] = digit
        r = r - digit * place          # shared bank, output weight ``place``
        residual_trace.append(r)
    return digits, residual_trace


def offset_from_digits(digits: Sequence[int]) -> int:
    """Reassemble ``Σ_j d_j·16^j`` — the offset the cascade encodes (§726)."""
    return sum(int(d) * (16 ** j) for j, d in enumerate(digits))


# ===========================================================================
# 3.  Retrieve the byte at position N  (§710, §715-717): the whole substrate
#     run through the REAL softmax1 attention head.
# ===========================================================================
class IOPositionBuffer:
    """An I/O buffer addressed by *absolute sequence position*, the substrate a
    GETCHAR / argv read uses (§715).

    Tokens are appended in stream order starting at ``marker_pos + 1`` (the
    marker itself is the reference point, §706).  ``read_offset(N)`` fetches the
    byte at buffer offset ``N`` — i.e. absolute position ``marker_pos + 1 + N``
    — by building a query with that position's ALiBi signature and letting the
    real ``softmax1`` head select the matching store (§710 "construct a query
    that matches the exponential signature of distance N").

    Two read paths, both verified against the plain-list reference:
      * :meth:`read_offset` — argmax over the position-signature match logits;
      * :meth:`read_offset_softmax1` — the ACTUAL softmax1 attention arithmetic,
        proving the match survives ZFOD (§491) like the ALiBi head it stands in.
    """

    def __init__(self, marker_pos: int = 0, n_heads: int = 8,
                 slopes: Sequence[float] | None = None):
        self.marker_pos = marker_pos
        self.n_heads = n_heads
        self.slopes = list(slopes) if slopes is not None else alibi_slopes(n_heads)
        self._bytes: List[int] = []          # buffer offset 0,1,2,… -> byte

    # -- construction --------------------------------------------------------
    def append(self, byte: int) -> None:
        """Append the next input byte (offset == current length)."""
        self._bytes.append(int(byte) & 0xFF)

    def extend(self, data: Sequence[int]) -> None:
        for b in data:
            self.append(b)

    def _abs_pos(self, offset: int) -> int:
        """Absolute sequence position of buffer ``offset`` (§706: after marker)."""
        return self.marker_pos + 1 + offset

    def _distance(self, offset: int) -> int:
        """ALiBi distance from the BOS/marker for buffer ``offset`` (§712)."""
        return self._abs_pos(offset) - self.marker_pos      # == offset + 1

    # -- signature-match logits ---------------------------------------------
    def _match_logit(self, query_offset: int, store_offset: int) -> float:
        """Log-space match between the query's target distance and a store's own
        distance signature.  The softmax1 head's per-head logit for a store at
        distance ``d_s`` under a query targeting distance ``d_q`` is
        ``−m_k·|d_q − d_s|`` (ALiBi over the signature difference); summed over
        heads that is ``−(Σ_k m_k)·|d_q − d_s|`` — maximal (0) at the exact
        position, strictly negative elsewhere.  This is the additive-bias view of
        "query matches the exponential signature of distance N" (§710).
        """
        d_q = self._distance(query_offset)
        d_s = self._distance(store_offset)
        return -sum(self.slopes) * abs(d_q - d_s)

    def logits(self, query_offset: int) -> torch.Tensor:
        return torch.tensor([self._match_logit(query_offset, i)
                             for i in range(len(self._bytes))],
                            dtype=torch.float64)

    # -- read paths ----------------------------------------------------------
    def read_offset(self, offset: int) -> int:
        """Byte at buffer ``offset`` via position-signature argmax.

        Out-of-range offset -> 0 (softmax1 ZFOD: no store's signature matches the
        target distance, the ``+1`` sink dominates, nothing is read, §491)."""
        if not self._bytes or offset < 0 or offset >= len(self._bytes):
            return 0
        lg = self.logits(offset)
        best = int(lg.argmax().item())
        return self._bytes[best]

    def read_offset_softmax1(self, offset: int) -> int:
        """Byte at ``offset`` via the REAL ``softmax1`` attention arithmetic.

        Builds the position-match logits, runs ``blogspec_model.softmax1`` (the
        ZFOD sink), reads the value as a softmax1-weighted sum of one-hot byte
        codes, and argmaxes back.  Proves the position retrieval works under the
        actual vanilla head, not just argmax (§491, §710)."""
        from .blogspec_model import softmax1
        if not self._bytes:
            return 0
        # Sharpen the raw ALiBi-difference logits (0 at match, −Σm_k·Δ else) and
        # lift the exact match to ``+MATCH_GAIN`` so softmax1 gives it dominant
        # weight while a 1-unit miss lands negative (ZFOD sink out-weighs it).
        # ``sharpen`` normalises out the slope sum so one unit of position error
        # costs a fixed large margin regardless of head count / slope ladder.
        sharpen = MATCH_GAIN / sum(self.slopes)
        lg = (self.logits(offset) * sharpen + MATCH_GAIN).to(torch.float32)
        w = softmax1(lg, dim=-1)                     # [n_stores], sums to <1
        # ZFOD: an out-of-range target has no near-0 logit, so the sink wins.
        if offset < 0 or offset >= len(self._bytes) or float(w.max()) < 0.5:
            return 0
        vals = torch.zeros(len(self._bytes), 256)
        for i, b in enumerate(self._bytes):
            vals[i, b] = 1.0
        read = (w.unsqueeze(-1) * vals).sum(0)       # [256]
        return int(read.argmax().item())

    # -- convenience: end-to-end offset->byte using the cascade --------------
    def read_scalar_offset(self, scalar_offset: int) -> int:
        """Take a *scalar* offset, run it through the nibble cascade to explicit
        nibbles, reassemble, and read that byte — the full GETCHAR path (§851:
        "a nibble cascade extracts the offset to index into the input buffer")."""
        digits, _ = nibble_cascade_offset(scalar_offset)
        return self.read_offset(offset_from_digits(digits))

    # -- the neural READ run: n sequential bytes through ONE real Attn forward -
    def read_run_neural(self, start: int, n: int) -> List[int]:
        """Retrieve ``n`` consecutive bytes ``[start, start+n)`` through the REAL
        ``blogspec_model.Attn`` softmax1 + ALiBi forward — the neural stdin /
        GETCHAR read path (§710, §715), the drop-in for a Python buffer slice.

        The whole input buffer is laid out as a token stream — a MARKER token
        (the BOS/reference key, §706) followed by one BYTE token per buffered
        byte, each carrying its one-hot value on the residual — and run through
        the vanilla multi-head ``Attn`` (every head sharing the BOS key, distinct
        ALiBi slopes, §704).  The byte at buffer offset ``o`` sits at ALiBi
        distance ``o+1`` from the marker; after the forward its OWN residual row
        carries the value it attends to at that distance, so reading the row at
        absolute position ``marker+1+o`` retrieves byte ``o`` (the demo mechanism,
        batched: one forward serves the whole run).

        Bytes outside the buffer read as 0 (softmax1 ZFOD: no store sits at that
        distance, the ``+1`` sink wins, §491) — a short read.  Byte-for-byte
        identical to ``list(self._bytes[start:start+n])`` but computed by
        attention over the token stream, not a Python list slice.
        """
        from .blogspec_model import Attn

        n = int(n)
        if n <= 0:
            return []
        if not self._bytes:
            return [0] * n

        d_key, d_val = 1, 256
        head_dim = d_key + d_val
        nh = self.n_heads
        dim = head_dim * nh

        seq_len = 1 + len(self._bytes)          # MARKER + one row per buffered byte
        attn = Attn(dim=dim, n_heads=nh, max_seq_len=seq_len + 8)
        with torch.no_grad():
            attn.alibi_slopes.copy_(torch.tensor(self.slopes))
            attn.W_q.zero_(); attn.W_k.zero_(); attn.W_v.zero_(); attn.W_o.zero_()
            for h in range(nh):
                base = h * head_dim
                attn.W_q[base + 0, 0] = BOS_KEY_SCALE      # shared BOS key: Q ...
                attn.W_k[base + 0, 0] = 1.0                #             ... and K
                for v in range(d_val):                     # value passthrough
                    attn.W_v[base + d_key + v, d_key + v] = 1.0
                if h == 0:                                 # fold head-0 values back
                    for v in range(d_val):
                        attn.W_o[d_key + v, base + d_key + v] = 1.0

        # Residual: every token carries the shared BOS-key lane; byte tokens also
        # carry their one-hot value.  Row 0 is the marker (§706).
        x = torch.zeros(1, seq_len, dim)
        x[0, :, 0] = 1.0                                   # shared BOS key lane
        for o, b in enumerate(self._bytes):
            x[0, 1 + o, d_key + b] = 1.0

        out = attn(x)                                      # one real forward
        result: List[int] = []
        for i in range(n):
            o = start + i
            if o < 0 or o >= len(self._bytes):
                result.append(0)                           # ZFOD short-read
                continue
            read_vec = out[0, 1 + o, d_key:d_key + d_val]
            result.append(int(read_vec.argmax().item()))
        return result


# ===========================================================================
# Real-attention head demo: retrieve byte@N through blogspec_model.Attn
# ===========================================================================
def demo_position_head_forward(n_heads: int = 8) -> dict:
    """Retrieve a buffer byte through the REAL ``blogspec_model.Attn`` softmax1
    forward, using the multi-slope BOS ALiBi bias for position matching.

    We build a self-contained ``Attn``-shaped head driven by a hand-set residual:
    a BOS/marker token, then a run of I/O byte tokens, then a query token whose
    Q is aligned to the shared BOS key.  Each head's ALiBi bias ``−m_k·d`` makes
    the byte token at the queried distance win under softmax1, and its value lane
    carries the byte.  This is the ALiBi position head of §704-712 run for real.
    """
    from .blogspec_model import Attn

    slopes = alibi_slopes(n_heads)
    buf = [0x48, 0x65, 0x6C, 0x6C, 0x6F]       # "Hello"
    target_offset = 2                          # want 0x6C
    want = buf[target_offset]

    marker_pos = 0
    seq = [("MARKER", None)] + [("BYTE", b) for b in buf]
    S = len(seq)

    # d_val one-hot byte lanes + one shared "BOS key" lane pair (Q·K constant).
    d_key = 1                                   # single shared-key lane
    d_val = 256
    D = d_key + d_val

    # A minimal single-head-per-slope Attn is overkill; instead drive the real
    # multi-head Attn with H = n_heads, each head reading the SAME key lane so
    # ALiBi (per-head slope) is the only positional discriminator (§704).
    head_dim = D                                # give every head the full width
    dim = head_dim * n_heads
    attn = Attn(dim=dim, n_heads=n_heads, max_seq_len=S + 8)
    with torch.no_grad():
        attn.alibi_slopes.copy_(torch.tensor(slopes))
        # Identity-ish projections: Q,K read the shared key lane; V reads bytes.
        attn.W_q.zero_(); attn.W_k.zero_(); attn.W_v.zero_(); attn.W_o.zero_()
        for h in range(n_heads):
            base = h * head_dim
            # shared BOS key: Q and K both put weight on the key lane (lane 0)
            attn.W_q[base + 0, 0] = BOS_KEY_SCALE
            attn.W_k[base + 0, 0] = 1.0
            # value passthrough: byte one-hot lanes -> head value lanes
            for v in range(d_val):
                attn.W_v[base + d_key + v, d_key + v] = 1.0
            # output: fold head 0's value lanes back to the residual byte lanes
            if h == 0:
                for v in range(d_val):
                    attn.W_o[d_key + v, base + d_key + v] = 1.0

    # Residual: every token carries the shared key lane = 1; byte tokens also
    # carry their one-hot byte.  The query token is the byte at target_offset.
    x = torch.zeros(1, S, dim)
    for i, (kind, val) in enumerate(seq):
        x[0, i, 0] = 1.0                        # shared BOS key lane
        if kind == "BYTE":
            x[0, i, d_key + val] = 1.0

    out = attn(x)                               # real softmax1 + ALiBi forward
    # The query is asked AT the target byte's own position (distance target+1):
    read_pos = marker_pos + 1 + target_offset
    read_vec = out[0, read_pos, d_key:d_key + d_val]
    read = int(read_vec.argmax().item())
    return {"read": read, "want": want,
            "match": read == want,
            "slopes": [round(s, 6) for s in slopes]}


# ---------------------------------------------------------------------------
# Self-demonstration: `python -m c4_min.nibble_io_position`.
# ---------------------------------------------------------------------------
def _demo() -> None:
    print("I/O position substrate (BLOG_SPEC §Position Offset / §Printing-Reading)")
    nh = 8
    sl = alibi_slopes(nh)
    print(f"  {nh} ALiBi slopes: {[round(s,5) for s in sl]}\n")

    print("  position signature (exp(−m_k·d)) -> position:")
    for d in (1, 5, 42, 1000, 65535):
        sig = position_signature(d, nh)
        back = position_from_signature(sig, nh)
        print(f"    d={d:<6} -> sig[0..2]={[round(float(s),4) for s in sig[:3]]}"
              f" -> recovered {back}  {'OK' if back == d else 'MISMATCH'}")

    print("\n  nibble cascade (base-16, 8 layers) offset -> nibbles:")
    for off in (0, 15, 255, 0xABCD, 0xDEADBEEF):
        digits, _ = nibble_cascade_offset(off)
        back = offset_from_digits(digits)
        hexd = "".join(f"{d:x}" for d in reversed(digits))
        print(f"    0x{off:08X} -> nibbles {hexd} -> {back}"
              f"  {'OK' if back == (off & 0xFFFFFFFF) else 'MISMATCH'}")

    print("\n  retrieve byte@offset via position match (buffer='Hello'):")
    buf = IOPositionBuffer(marker_pos=0, n_heads=nh)
    buf.extend([0x48, 0x65, 0x6C, 0x6C, 0x6F])
    for off in range(5):
        r = buf.read_offset(off)
        r1 = buf.read_offset_softmax1(off)
        print(f"    offset {off}: argmax=0x{r:02X}  softmax1=0x{r1:02X}"
              f"  ('{chr(r)}')")
    print(f"    out-of-range offset 9 -> 0x{buf.read_offset(9):02X} (ZFOD)")

    print("\n  real softmax1 Attn head byte@N retrieval:")
    d = demo_position_head_forward(nh)
    print(f"    read=0x{d['read']:02X}  want=0x{d['want']:02X}  "
          f"{'OK' if d['match'] else 'MISMATCH'}")


if __name__ == "__main__":
    _demo()
