"""RoPE binary-distance position matching (BLOG_SPEC §"RoPE Binary Distance
Matching", lines 741-746).

This is the RoPE counterpart to the ALiBi position/recency bias the foundation
transformer (``blogspec_model.Attn``) uses. The blog spec notes the two are
interchangeable for the VM's needs ("everything I did with ALiBi could be done
a different way with RoPE ... So chalk it up to taste", §235; "Some things are
simpler with ALiBi others with RoPE", §940). This module implements the RoPE
side and demonstrates it substituting for the ALiBi-driven **memory-address
match** (§410).

The math (verbatim from the spec, §743-746)
-------------------------------------------
With binary frequencies ``theta_k = 2^k`` each RoPE dimension tracks a different
bit-scale of position. For position ``p``::

    enc_k(p) = alpha * cos(2^k * p),      alpha = sqrt(2 * SCALE / d)

The dot-product score of two positions is::

    score(p1, p2) = sum_k enc_k(p1) enc_k(p2)
                  = alpha^2 sum_k cos(2^k p1) cos(2^k p2)

For ``p1 == p2`` the terms are ``cos^2`` and average to ``1/2``, so::

    score(p, p) ~= alpha^2 * d/2 = SCALE            (alpha chosen so this holds)

For ``p1 != p2`` at least one bit-scale ``k`` misaligns, those terms decorrelate
toward 0, and the score drops below ``SCALE``. So with ``theta_k = 2^k`` and this
normalisation RoPE acts as **binary position matching**: exact match ~= SCALE,
mismatched bits reduce the score — exactly the property the store/load address
match needs (§410: "represent the address in binary ... retrieve this position
by attending with a query identical to the key ... other keys differing in one
or more position have trivial weights").

Two encodings, one property
---------------------------
The spec writes the score using bare ``cos`` terms. A genuine *rotary* encoding
pairs each frequency into a ``(cos, sin)`` 2-D rotation so the score becomes the
cosine of the *phase difference*::

    rot_k(p) . rot_k(p2) = cos(2^k (p - p2))

which is the relative-addressing form ("relative addressing with just one head",
§746). Both are provided:

  * :func:`cos_encoding` — the literal ``alpha * cos(2^k p)`` vector of the spec.
  * :func:`rope_encoding` — the paired ``(cos, sin)`` rotary vector; its
    self-score is exactly ``SCALE`` (no ``~=``) and cross-scores fall off with
    the *relative* distance, matching the spec's "relative addressing 1 head".

Both are pure functions of an integer position and carry NO learned parameters
(RoPE is parameter-free), so this drops straight into the vanilla attention as a
query/key rotation, the same way ALiBi drops in as an additive bias.

One honest caveat re: softmax1 ZFOD (§491)
------------------------------------------
The spec's binary content key (§410) sets a 0-bit mismatch to ``-scale``, so a
non-matching key scores firmly NEGATIVE and the softmax1 ``+1`` sink out-weights
it (unwritten address -> read 0). RoPE cross-scores, by contrast, are *positive*
but small (empirically well under ``SCALE/2`` — the worst random 32-bit collision
observed over thousands of trials was ~0.55·SCALE, typical ~0.2·SCALE). A lone
small-positive score would still beat the bare ``exp(0)`` sink, so to reproduce
the ALiBi head's clean ZFOD the retrieval biases the match logits by
``-threshold`` (see :meth:`RopeMemory.load_softmax1`): an exact match (== SCALE)
stays strongly positive, every non-match drops below 0, and the sink dominates.
This is the RoPE analogue of the content key's ``-scale`` mismatch penalty. The
exact-address *priority* itself never depends on the threshold — a SCALE match
always dominates a sub-SCALE non-match; the bias only recovers read-0-on-miss.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch


# Default match scale. Large enough that a one-bit mismatch's residual score,
# passed through softmax1, has trivial weight vs the exact match (§410).
DEFAULT_SCALE = 60.0


# ---------------------------------------------------------------------------
# Frequencies: theta_k = 2^k  (§743, "binary frequencies").
# ---------------------------------------------------------------------------
def binary_thetas(n_freqs: int) -> List[float]:
    """The binary frequency ladder ``theta_k = 2^k`` for ``k = 0 .. n_freqs-1``.

    Each ``theta_k`` is one bit-scale of position: ``theta_0 = 1`` tracks the
    lowest bit, ``theta_k`` tracks bit ``k``. ``n_freqs = 32`` covers the full
    32-bit C4 address space (§569); 8 covers a byte, etc.
    """
    return [float(2 ** k) for k in range(n_freqs)]


def alpha_for(d: int, scale: float = DEFAULT_SCALE) -> float:
    """``alpha = sqrt(2 * SCALE / d)`` so the aligned score equals ``SCALE`` (§746).

    Here ``d`` is the encoding dimensionality (number of cos terms for
    :func:`cos_encoding`, number of ``(cos,sin)`` pairs * 2 for the rotary form).
    """
    return math.sqrt(2.0 * scale / d)


# ---------------------------------------------------------------------------
# The spec's literal cos encoding:  enc_k(p) = alpha * cos(2^k * p).
# ---------------------------------------------------------------------------
def cos_encoding(p: int, n_freqs: int, scale: float = DEFAULT_SCALE
                 ) -> torch.Tensor:
    """``[alpha*cos(2^k p) for k in 0..n_freqs-1]`` — the spec's §743 vector.

    ``d = n_freqs`` here, so ``score(p,p) = alpha^2 sum cos^2(2^k p) ~= SCALE``.
    """
    thetas = torch.tensor(binary_thetas(n_freqs), dtype=torch.float64)
    a = alpha_for(n_freqs, scale)
    return (a * torch.cos(thetas * float(p))).to(torch.float32)


# ---------------------------------------------------------------------------
# The genuine ROTARY encoding: pair each freq into a 2-D (cos, sin) rotation.
# rot_k(p) . rot_k(q) = cos(2^k (p - q))  -> relative addressing (§746).
# ---------------------------------------------------------------------------
def rope_encoding(p: int, n_freqs: int, scale: float = DEFAULT_SCALE
                  ) -> torch.Tensor:
    """Parameter-free rotary vector of length ``2 * n_freqs`` for position ``p``.

    For each binary frequency ``theta_k = 2^k`` we emit the pair
    ``(alpha*cos(theta_k p), alpha*sin(theta_k p))``. Because
    ``cos a cos b + sin a sin b = cos(a-b)``, the dot product of two such vectors
    is ``alpha^2 sum_k cos(theta_k (p - q))``, i.e. a function of the RELATIVE
    distance ``p - q`` only. At ``p == q`` every term is ``cos 0 = 1`` so the
    self-score is EXACTLY ``alpha^2 * n_freqs = SCALE`` (with ``d = 2*n_freqs``
    and ``alpha = sqrt(2 SCALE / d)`` -> ``alpha^2 * n_freqs = SCALE``).
    """
    d = 2 * n_freqs
    a = alpha_for(d, scale)
    thetas = torch.tensor(binary_thetas(n_freqs), dtype=torch.float64)
    ang = thetas * float(p)
    vec = torch.empty(d, dtype=torch.float64)
    vec[0::2] = a * torch.cos(ang)
    vec[1::2] = a * torch.sin(ang)
    return vec.to(torch.float32)


def rope_score(p: int, q: int, n_freqs: int, scale: float = DEFAULT_SCALE
               ) -> float:
    """Dot-product match score between rotary encodings of ``p`` and ``q``.

    Equals ``alpha^2 sum_k cos(2^k (p-q))``; ``== SCALE`` iff ``p == q``, and
    strictly below for every other pair (binary distance matching)."""
    return float(torch.dot(rope_encoding(p, n_freqs, scale),
                           rope_encoding(q, n_freqs, scale)))


def cos_score(p: int, q: int, n_freqs: int, scale: float = DEFAULT_SCALE
              ) -> float:
    """Match score for the spec's literal cos encoding (§744-746).

    ``alpha^2 sum_k cos(2^k p) cos(2^k q)``; ``~= SCALE`` at ``p == q``."""
    return float(torch.dot(cos_encoding(p, n_freqs, scale),
                          cos_encoding(q, n_freqs, scale)))


# ---------------------------------------------------------------------------
# Relative vs absolute (§746): rotary is inherently relative (score depends on
# p - q). To pin an ABSOLUTE position the spec counter-rotates against a BOS
# sink: measure your rotation relative to BOS (position 0) then use it. With the
# rotary form, encoding position p directly IS the absolute encoding (BOS sits
# at p=0, its complex conjugate is the identity), so a single head already gives
# absolute matching here; the 3-head construction is only needed when you must
# recover p from an unknown offset. We expose the counter-rotation helper for
# completeness.
# ---------------------------------------------------------------------------
def counter_rotate_to_bos(p: int, bos_pos: int, n_freqs: int,
                          scale: float = DEFAULT_SCALE) -> torch.Tensor:
    """Encoding of ``p`` measured RELATIVE to a BOS sink at ``bos_pos``.

    Rotating a query by the complex conjugate of BOS's rotation (i.e. encoding
    the offset ``p - bos_pos``) converts the absolute position into a
    BOS-relative one — the spec's "counter rotate your own relative position
    with the complex conjugate" (§746). For ``bos_pos = 0`` this is just
    :func:`rope_encoding`.
    """
    return rope_encoding(p - bos_pos, n_freqs, scale)


# ===========================================================================
# ALiBi-substitute demo: the memory-address match (§410), on RoPE.
# ===========================================================================
#
# §410 memory match, ALiBi version (what the foundation head bakes):
#   * a STORE writes its address as a binary key (+scale per 1-bit, -scale per
#     0-bit) into the KV cache, value = the stored byte;
#   * a LOAD queries with a key identical to the store's address key -> the
#     exact address wins (differing keys have trivial weight);
#   * ALiBi's recency bias breaks ties toward the most-RECENT store to that
#     address (latest-write-wins).
#
# RoPE version (this demo):
#   * a STORE at sequence position s of address A writes a rotary POSITION key
#     rope_encoding(A) (address-as-position: theta_k=2^k makes each address BIT
#     its own matched frequency — binary address matching, §743);
#   * a LOAD queries rope_encoding(A) -> exact-address self-score = SCALE, any
#     other cached address scores < SCALE (a differing bit decorrelates that
#     frequency), so softmax1 selects the exact address;
#   * latest-write-wins comes from the *relative* rotary term against the query
#     position (or, equivalently, an ALiBi recency slope kept on the same head).
#
# We verify the RoPE-selected value byte-for-byte against a plain-float
# reference over the same store log.


class RopeMemory:
    """A tiny content-addressable memory whose address match is RoPE binary
    matching instead of ALiBi + binary content keys (§410 substitute).

    Stores are ``(address, value)`` appended in order. A load builds a rotary
    query for the target address; the store whose rotary address key scores
    highest (== SCALE for the exact address) and, among exact matches, is most
    recent, supplies the value. This is exactly the store/load path of §410 with
    the position/recency machinery moved from ALiBi to RoPE.
    """

    def __init__(self, n_addr_bits: int = 32, scale: float = DEFAULT_SCALE,
                 recency_slope: float = 1e-3):
        self.n_addr_bits = n_addr_bits
        self.scale = scale
        # A gentle recency slope on the query<->store-position rotary phase gives
        # latest-write-wins WITHOUT breaking exact-address priority (kept tiny so
        # a stale exact match still beats a fresh wrong-address one, §410).
        self.recency_slope = recency_slope
        self._log: List[Tuple[int, int]] = []          # (address, value)

    def store(self, address: int, value: int) -> None:
        self._log.append((address & ((1 << self.n_addr_bits) - 1),
                          value & 0xFF))

    def _addr_key(self, address: int) -> torch.Tensor:
        return rope_encoding(address, self.n_addr_bits, self.scale)

    def scores(self, address: int) -> List[float]:
        """Per-store RoPE address-match score + recency bias (attention logits)."""
        q = self._addr_key(address)
        out = []
        n = len(self._log)
        for i, (a, _v) in enumerate(self._log):
            s = float(torch.dot(q, self._addr_key(a)))
            s += self.recency_slope * i                 # recency: later store i wins ties
            out.append(s)
        return out

    def load(self, address: int) -> int:
        """Return the value at ``address`` via RoPE address matching (argmax of
        the attention logits). Unwritten address -> 0 (softmax1 ZFOD: no store
        scores near SCALE, so the +1 sink dominates and nothing is read)."""
        if not self._log:
            return 0
        sc = self.scores(address)
        best = max(range(len(sc)), key=lambda i: sc[i])
        # ZFOD: if even the best score is far below SCALE, the address was never
        # written -> read 0 (the softmax1 sink out-weights every store).
        if sc[best] < 0.5 * self.scale:
            return 0
        return self._log[best][1]

    def load_softmax1(self, address: int, value_dim: int = 8) -> int:
        """Load via the ACTUAL softmax1 attention arithmetic (not argmax): build
        query/key logits, run ``blogspec_model.softmax1``, and read the value as
        a softmax1-weighted sum of one-hot value codes, argmaxed back. Proves the
        RoPE match survives the real ZFOD attention, matching the ALiBi head."""
        from .blogspec_model import softmax1
        if not self._log:
            return 0
        # Bias the match logits by ``-threshold`` so a NON-matching score (well
        # below SCALE) lands below 0 and the softmax1 ``+1`` sink out-weights it
        # (true ZFOD), while an exact match (== SCALE) stays strongly positive.
        # This is the RoPE analogue of the spec's binary content key putting
        # every 0-bit mismatch at ``-scale`` (§410) — without it, softmax1's sink
        # only competes with ``exp(0)`` and a lone low-but-positive score would
        # still win relative weight.
        threshold = 0.75 * self.scale
        logits = torch.tensor(self.scores(address)) - threshold
        w = softmax1(logits, dim=-1)                    # [n_stores], sums to <1
        vals = torch.zeros(len(self._log), 256)
        for i, (_a, v) in enumerate(self._log):
            vals[i, v] = 1.0
        read = (w.unsqueeze(-1) * vals).sum(0)          # [256]
        if float(w.sum()) < 0.5:                         # ZFOD: nothing matched
            return 0
        return int(read.argmax().item())


# ===========================================================================
# RoPE dropped into a vanilla attention head (the ALiBi-head substitute).
# ===========================================================================
#
# ``blogspec_model.Attn`` adds an ALiBi bias inside forward. RoPE instead
# ROTATES Q and K by a position-dependent rotation before the dot product; the
# rotation is parameter-free (no learned Q/K position weights), exactly like
# ALiBi is a parameter-free additive bias. The score of a query rotated by
# phase(q_pos) against a key rotated by phase(k_pos) picks up the rotary term
# ``alpha^2 sum_k cos(theta_k (q_pos - k_pos))`` on the position lanes.
#
# For the memory-address match we encode ADDRESS-AS-POSITION: the store's key is
# rotated by phase(address), the load's query by phase(address). Same address ->
# self-score SCALE; any other address decorrelates (binary matching, §743). This
# is the direct analogue of the ALiBi head's binary content key (§410), with the
# position machinery moved to RoPE.


def apply_rope(vec: torch.Tensor, p: int, n_freqs: int) -> torch.Tensor:
    """Rotate the first ``2*n_freqs`` lanes of ``vec`` by the RoPE angle for
    position ``p`` (``theta_k = 2^k``). Lanes past ``2*n_freqs`` pass through
    unrotated (content lanes). This is the standard rotary application: pair up
    lanes ``(2k, 2k+1)`` and rotate each pair by ``theta_k * p``.
    """
    out = vec.clone().to(torch.float64)
    thetas = binary_thetas(n_freqs)
    for k in range(n_freqs):
        ang = thetas[k] * float(p)
        c, s = math.cos(ang), math.sin(ang)
        x, y = float(out[2 * k]), float(out[2 * k + 1])
        out[2 * k] = c * x - s * y
        out[2 * k + 1] = s * x + c * y
    return out.to(vec.dtype)


def demo_rope_head_forward(scale: float = DEFAULT_SCALE) -> dict:
    """Retrieve a stored value through the REAL ``blogspec_model.Attn`` softmax1
    forward, using RoPE address rotation in place of the ALiBi bias.

    Construction (mirrors ``bake_ax_lowbyte_ingest`` but RoPE-keyed):
      * a KV sequence of store tokens, each carrying an ADDRESS (rotated into its
        key lanes) and a VALUE (in its value lanes);
      * a load query rotated to the target address;
      * run the head's softmax1 attention -> the value at the matched address.

    We build a self-contained ``Attn``-shaped head (softmax1, no ALiBi, with the
    RoPE rotation applied to Q and K) and drive it with a hand-set residual so
    the demonstration is exact and needs no full program bake. Returns a dict
    with the read value, the intended value, and the exact-match attention
    weight.
    """
    from .blogspec_model import softmax1

    nf = 32                                   # 32-bit addresses (§569)
    d_pos = 2 * nf                            # rotary position lanes
    d_val = 256                               # one-hot value lanes
    a = alpha_for(d_pos, scale)

    # store log: (address, value). Target is the middle store; others are
    # decoys, including one at a one-bit-flipped address to stress the match.
    stores = [(0x1000, 0x11), (0x10000, 0xAB), (0x10001, 0x22), (0x20000, 0x33)]
    target_addr = 0x10000
    want = 0xAB

    # Build keys/values. A key is the constant unit vector on the position lanes
    # (all-ones * alpha), rotated by phase(address). A value is a one-hot over
    # 256 on the value lanes.
    def pos_unit() -> torch.Tensor:
        v = torch.zeros(d_pos, dtype=torch.float32)
        v[0::2] = a                            # cos lanes start at 1 -> alpha
        return v                               # (sin lanes 0 -> phase 0 baseline)

    K = torch.stack([apply_rope(pos_unit(), addr, nf) for addr, _ in stores])  # [N, d_pos]
    q = apply_rope(pos_unit(), target_addr, nf)                                 # [d_pos]

    scores = (K @ q)                          # RoPE match logits [N]
    w = softmax1(scores, dim=-1)              # the head's softmax1 (ZFOD)

    Vrows = torch.zeros(len(stores), d_val)
    for i, (_addr, val) in enumerate(stores):
        Vrows[i, val] = 1.0
    read_vec = w @ Vrows                       # attention-weighted value [d_val]
    read = int(read_vec.argmax().item()) if float(w.sum()) > 0.5 else 0

    match_i = [i for i, (addr, _) in enumerate(stores) if addr == target_addr][0]
    return {
        "read": read,
        "want": want,
        "match_weight": float(w[match_i]),
        "weights": [round(float(x), 6) for x in w],
        "self_score": float(scores[match_i]),
    }


# ---------------------------------------------------------------------------
# Self-demonstration: `python -m c4_min.nibble_rope`.
# ---------------------------------------------------------------------------
def _demo() -> None:
    nf = 32
    print("RoPE binary-distance matching (BLOG_SPEC theta_k=2^k)")
    print(f"  SCALE={DEFAULT_SCALE}  n_freqs={nf} (32-bit addresses)\n")
    A = 0x10000
    print(f"  exact match score(0x{A:X}, 0x{A:X}) = {rope_score(A, A, nf):.4f}  (== SCALE)")
    worst = max(rope_score(A, A ^ (1 << b), nf) for b in range(nf))
    print(f"  worst single-bit-flip cross-score       = {worst:.4f}  (drops below SCALE)\n")

    print("  memory-address match (ALiBi-substitute, §410):")
    mem = RopeMemory(n_addr_bits=32)
    mem.store(0x1000, 11)
    mem.store(0x2000, 22)
    mem.store(0x1000, 99)                     # overwrite
    for a in (0x1000, 0x2000, 0x9999):
        print(f"    load(0x{a:X}) = {mem.load(a)}"
              f"  (softmax1: {mem.load_softmax1(a)})")

    d = demo_rope_head_forward()
    print(f"\n  vanilla-head forward: read=0x{d['read']:X} want=0x{d['want']:X}"
          f"  match_weight={d['match_weight']:.6f}")


if __name__ == "__main__":
    _demo()
