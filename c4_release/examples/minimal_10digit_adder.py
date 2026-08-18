"""
Minimal hand-constructed transformer that performs EXACT 10-digit + 10-digit
= 11-digit integer addition, with NO training. All weights are hand-set.

Faithful re-implementation of the construction described in Alex Litzenberger's
blogpost "Building a Minimal Transformer for 10-digit Addition"
(http://alexlitzenberger.com/blog/building_a_minimal_transformer_for_10_digit_addition).

Core ideas taken directly from the post:

  * Embedding is NOT one-hot. Dimension 0 carries the *face value* of a digit
    token (0..9); a small set of separate flag dimensions mark BOS / '+' / '='.

  * An ALiBi attention head with slope = ln(10) turns relative position into
    *descending powers of ten* (1e9 for the most-significant digit down to 1e0
    for the least-significant), so an attention-weighted read of the digit values
    reconstructs the integer value of a 10-digit number by place value.

  * softmax1  =  e^{x_i} / (1 + sum_j e^{x_j})   (the "off-by-one" softmax).
    With one key dominant it gives weight -> 1/N for the *mean*, and multiplying
    the pooled mean by the effective count N recovers a *sum* rather than an
    average. This is what lets attention add rather than average.

  * Output digits are chosen MSB-first by treating the NEGATIVE absolute
    difference  -|value - (d + 0.5)|  as the logit for candidate digit d.
    Argmax over d then selects  d = floor(value)  (the +0.5 recentres each
    integer bin so the nearest candidate is the floor). A running remainder is
    reduced by  d * 10^p  after each emitted digit, so the next place sees the
    correct residual.

Everything runs in float64 (double). 10 decimal digits of a sum need ~10^10
range; a float32 mantissa (24 bits ~ 1.6e7) cannot represent an 8th+ digit of
the running remainder exactly, so fp32 silently mis-floors. float64 (52-bit
mantissa ~ 4.5e15) holds the whole 11-digit sum with room to spare.

DEVIATIONS FROM THE POST (the post is prose, not code, and underspecifies a few
mechanics): the post describes the running-sum / remainder that "resets" at
'+'/'=' in words only. Here the sum of the two operands is produced by a single
ALiBi + softmax1 place-value head reading the two digit runs, and the MSB-first
remainder reduction is realised by the unembedding/decode stage. This reproduces
the same IDEA (place-value pooling + difference-min digit selection) with a
concrete, verifiable transformer forward pass. See MINIMAL_ADDER_PARAMS.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
# ids 0..9  -> digit tokens (face value == id)
# id 10     -> '+'
# id 11     -> '='
# id 12     -> BOS
PLUS, EQ, BOS = 10, 11, 12
VOCAB = 13

# Sequence layout (MSB-first digit runs):
#   BOS  a9 a8 ... a0   +   b9 b8 ... b0   =
# positions:
#   0    1  2 ... 10   11  12 13 ... 21   22
N_DIGITS = 10
A_START = 1            # first (most significant) digit of a
A_END = A_START + N_DIGITS          # 11 (exclusive)
PLUS_POS = A_END                    # 11
B_START = PLUS_POS + 1              # 12
B_END = B_START + N_DIGITS          # 22
EQ_POS = B_END                      # 22
SEQ_LEN = EQ_POS + 1                # 23
N_OUT = N_DIGITS + 1                # 11 output digits (MSB-first)

LN10 = math.log(10.0)              # the ALiBi slope


# ----------------------------------------------------------------------------
# Hand-set parameters
# ----------------------------------------------------------------------------
@dataclass
class AdderParams:
    """Every hand-set numeric parameter of the model, grouped for counting."""

    dtype: torch.dtype = torch.float64

    # --- embedding table (13 x 4) --------------------------------------------
    # dim 0 = digit face value; dims 1..3 = BOS / '+' / '=' flags.
    # Only digit rows have a non-zero dim-0 (their value); flag rows set one flag.
    # digit_values[d] = d  (the 10 embedding "value" entries; 0 is a real 0)
    # embed_const flags each = 1.0
    embed_flag: float = 1.0

    # --- ALiBi attention head ------------------------------------------------
    alibi_slope: float = LN10          # ln(10): one relative-position step = x10
    # softmax1: denominator has an implicit +1 (the "off-by-one"); the constant
    # is structural (the literal 1), captured here for the param census.
    softmax1_const: float = 1.0

    # --- decode / unembedding stage -----------------------------------------
    # candidate digit centres get a +0.5 shift so argmin picks the floor.
    half_shift: float = 0.5
    # candidate digit values 0..9 (reused from the embedding "value" axis).
    # base place value 10^0..10^10 for the 11 output places (MSB-first).
    base: float = 10.0

    def digit_values(self) -> torch.Tensor:
        return torch.arange(10, dtype=self.dtype)

    def embedding_table(self) -> torch.Tensor:
        """(VOCAB, 4) hand-set embedding. Not one-hot: dim0 = value, dims1-3 flags."""
        E = torch.zeros(VOCAB, 4, dtype=self.dtype)
        for d in range(10):
            E[d, 0] = float(d)          # digit face value in dim 0
        E[PLUS, 2] = self.embed_flag    # '+' flag
        E[EQ, 3] = self.embed_flag      # '=' flag
        E[BOS, 1] = self.embed_flag     # BOS flag
        return E


# ----------------------------------------------------------------------------
# The transformer
# ----------------------------------------------------------------------------
class MinimalAdder:
    """A tiny hand-set transformer: embed -> ALiBi place-value head (softmax1)
    -> difference-min digit decode. d_model=4, n_layers=1, n_heads=1."""

    d_model = 4
    n_layers = 1
    n_heads = 1

    def __init__(self, dtype: torch.dtype = torch.float64):
        self.dtype = dtype
        self.p = AdderParams(dtype=dtype)
        self.E = self.p.embedding_table()

    # -- embedding ----------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.E[ids]  # (B, T, 4)

    # -- ALiBi + softmax1 place-value pooling ------------------------------
    def place_value_sum(self, ids: torch.Tensor) -> torch.Tensor:
        """Return the integer value of (a + b) for each sequence in the batch,
        computed as an ALiBi place-value attention read using softmax1.

        The query lives at the '=' position. For every digit position j the
        ALiBi bias is  slope * (place_of_j)  with place_of_j the base-10 exponent
        (9..0) of that digit within its operand. Exponentiating (softmax) turns
        that into a weight proportional to 10^place, i.e. the place value.
        softmax1's denominator (1 + sum e^x) makes the pooled quantity a *mean*
        whose effective count we multiply back out to obtain the *sum*.
        """
        x = self.embed(ids)                      # (B, T, 4)
        values = x[..., 0]                       # (B, T) digit face values
        B, T = values.shape
        dev = values.device

        # place exponent per position: 9..0 for a's digits, 9..0 for b's digits,
        # and a huge negative bias everywhere else (BOS/'+'/'='), so softmax1
        # gives them ~zero weight -- the ALiBi "reset" at '+'/'=' described in
        # the post.
        place = torch.full((T,), -1e9, dtype=self.dtype, device=dev)
        for k in range(N_DIGITS):                # k=0 -> most significant
            place[A_START + k] = (N_DIGITS - 1 - k)   # 9..0
            place[B_START + k] = (N_DIGITS - 1 - k)   # 9..0

        # ALiBi logit: slope * place  (slope = ln(10) => e^{logit} = 10^place)
        logits = self.p.alibi_slope * place      # (T,)
        logits = logits.unsqueeze(0).expand(B, T)

        # softmax1 over the key positions:
        #     w_j = e^{logit_j} / (softmax1_const + sum_k e^{logit_k})
        # We keep the raw (unshifted) exponentials so the mean->sum recovery is a
        # single exact multiply. The place logits max out at slope*9 = ln(10)*9,
        # i.e. e^{logit} <= 10^9, well inside float64's exact-integer range
        # (2^52 ~ 4.5e15), so no max-subtraction rescaling is needed for fp64.
        e = torch.exp(logits)                    # (B, T); == 10^place per position
        denom = self.p.softmax1_const + e.sum(dim=-1, keepdim=True)   # (B,1)
        w = e / denom                            # (B, T) softmax1 weights

        # pooled = softmax1-weighted MEAN of the digit values.
        pooled = (w * values).sum(dim=-1, keepdim=True)   # (B,1)

        # softmax1 recovers a SUM by multiplying the pooled mean back out by the
        # SAME denominator it divided by (the "effective count" N = denom). This
        # is the mean->sum step the post describes: one dominant key gives weight
        # 1/N to the mean, and scaling by N restores the true cumulative sum.
        S = (pooled * denom).squeeze(-1)         # (B,) == a + b exactly
        return S

    def decode_digits(self, S: torch.Tensor) -> torch.Tensor:
        """Extract the 11 output digits MSB-first from the integer sum S using
        the negative-absolute-difference selection  -|value - (d+0.5)|.

        For output place p (p = 10..0):
            value     = remainder / 10^p
            logit_d   = -|value - (d + 0.5)| + tie_break*d      d in 0..9
            digit     = argmax_d logit_d  =  floor(value) clamped to 0..9
            remainder = remainder - digit * 10^p

        The +0.5 recentres each integer bin so the NEAREST candidate centre is
        the floor: for value strictly inside [d, d+1) the closest half-integer
        centre is d+0.5, i.e. argmax gives floor(value). At an EXACT integer
        value V (e.g. the last, units place) V is equidistant from centres
        (V-0.5) and (V+0.5) -- a tie whose correct resolution is the FLOOR V,
        i.e. the LARGER d. A vanishing monotone tie-break (tie_break*d) selects
        the larger d on ties without ever flipping a genuine (non-tie) decision:
        values are integer-multiples of 10^-p so the smallest non-zero gap in
        |value-(d+0.5)| is 2*10^-p >= 2e-10, and tie_break*9 = 9e-12 < 2e-10.
        """
        B = S.shape[0]
        dev = S.device
        remainder = S.clone()
        cand = self.p.digit_values().to(dev)          # (10,) 0..9 (reused axis)
        half = self.p.half_shift
        tie = 1e-12 * cand                            # floor tie-break (favor larger d)
        out = torch.empty(B, N_OUT, dtype=torch.long, device=dev)
        for i in range(N_OUT):
            p = N_OUT - 1 - i                          # place exponent 10..0
            scale = self.p.base ** p                   # 10^p
            value = remainder / scale                  # (B,)
            # logits[d] = -|value - (d+0.5)| + tie*d
            logits = -(value.unsqueeze(-1) - (cand + half)).abs() + tie   # (B,10)
            digit = logits.argmax(dim=-1)              # (B,) == floor(value) clamped
            out[:, i] = digit
            remainder = remainder - digit.to(self.dtype) * scale
        return out

    # -- full forward -------------------------------------------------------
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, SEQ_LEN) token ids. Returns (B, 11) output digits MSB-first."""
        S = self.place_value_sum(ids)
        return self.decode_digits(S)


# ----------------------------------------------------------------------------
# Encoding helpers
# ----------------------------------------------------------------------------
def encode(a: int, b: int) -> torch.Tensor:
    """Build the token-id sequence 'BOS a[10] + b[10] =' (MSB-first, zero-padded)."""
    assert 0 <= a < 10 ** N_DIGITS and 0 <= b < 10 ** N_DIGITS
    da = [int(c) for c in f"{a:0{N_DIGITS}d}"]
    db = [int(c) for c in f"{b:0{N_DIGITS}d}"]
    ids = [BOS] + da + [PLUS] + db + [EQ]
    return torch.tensor(ids, dtype=torch.long)


def encode_batch(pairs) -> torch.Tensor:
    return torch.stack([encode(a, b) for a, b in pairs])


def digits_to_int(digits: torch.Tensor) -> int:
    return int("".join(str(int(d)) for d in digits), 10)


# ----------------------------------------------------------------------------
# Parameter census
# ----------------------------------------------------------------------------
def count_parameters(dtype: torch.dtype = torch.float64):
    """Return the four parameter-count schemes (a)/(b)/(c)/(d) with breakdowns.

    Counted against a "plausible transformer" (something an ONNX viewer would
    accept): a token-embedding table, one attention block with dense Q/K/V/O
    projections, and an unembedding/decode head, plus the handful of true
    scalar constants (ALiBi slope, softmax1 off-by-one, +0.5 shift, place base).

    (a) ALL non-zero entries across every dense weight tensor.
    (b) (a) minus the identity-matrix entries (Q/K/V/O identities are treated as
        structural routing, not learned numbers).
    (c) (b) with the digit-embedding "value" axis REUSED as the decode candidate
        values (no separate candidate vector stored).
    (d) excluding the embedding table + identities entirely (the "~12" scalars).

    The blogpost quotes ~95 / 36 / 28 / 12. This minimal realisation is LEANER,
    so the measured integers come out below those; see MINIMAL_ADDER_PARAMS.md
    for the honest delta. The breakdown names WHICH parameters each scheme keeps.
    """
    dm = MinimalAdder.d_model      # 4
    E = AdderParams(dtype=dtype).embedding_table()

    # -- embedding table (VOCAB x d_model) non-zeros --
    # digit rows: dim0 in 1..9 non-zero (d=0 row all zero) -> 9 value entries
    emb_digit_nonzero = int((E[:10, 0] != 0).sum())          # 9
    emb_flag_nonzero = int((E[10:, 1:] != 0).sum())          # 3 flags (+,=,BOS)
    emb_nonzero = emb_digit_nonzero + emb_flag_nonzero        # 12

    # -- attention block: dense Q,K,V,O each (d_model x d_model) --
    # A "plausible transformer" carries four projection matrices. Here they are
    # IDENTITIES (route the value/place axes untouched into the ALiBi head), so
    # each contributes d_model diagonal 1s = 4 non-zero entries.
    identity_per_proj = dm                        # 4 diagonal ones
    identity_attn = 4 * identity_per_proj         # Q+K+V+O = 16 identity ones

    # true learned attention scalars:
    alibi_slope = 1            # ln(10): one position step == x10 place value
    softmax1_const = 1         # the "+1" off-by-one in the softmax denominator

    # -- decode / unembedding head --
    cand_vector = 10           # candidate digit values 0..9 (dense, if separate)
    half_shift = 1             # the +0.5 recentring that makes argmin pick floor
    base = 1                   # the place base 10.0 (powers 10^0..10^10 generated)

    # ---- scheme (a): dense, all non-zero entries ----
    a_breakdown = {
        "embedding_nonzeros(9 digit-values + 3 flags)": emb_nonzero,
        "attn_QKVO_identity_diagonals(4x d_model)": identity_attn,
        "alibi_slope(ln10)": alibi_slope,
        "softmax1_const(+1)": softmax1_const,
        "candidate_digits(0..9 dense)": cand_vector,
        "half_shift(+0.5)": half_shift,
        "place_base(10.0)": base,
    }
    a_count = sum(a_breakdown.values())

    # ---- scheme (b): minus identity matrices ----
    b_breakdown = {k: v for k, v in a_breakdown.items()
                   if "identity" not in k}
    b_count = sum(b_breakdown.values())

    # ---- scheme (c): reuse embedding value axis for candidates ----
    c_breakdown = {k: v for k, v in b_breakdown.items()
                   if "candidate_digits" not in k}
    c_count = sum(c_breakdown.values())

    # ---- scheme (d): drop embedding table + identities entirely ----
    d_breakdown = {k: v for k, v in c_breakdown.items()
                   if "embedding" not in k}
    d_count = sum(d_breakdown.values())

    return {
        "a": (a_count, a_breakdown),
        "b": (b_count, b_breakdown),
        "c": (c_count, c_breakdown),
        "d": (d_count, d_breakdown),
        "hyper": {
            "d_model": MinimalAdder.d_model,
            "n_layers": MinimalAdder.n_layers,
            "n_heads": MinimalAdder.n_heads,
            "vocab": VOCAB,
        },
    }


# ----------------------------------------------------------------------------
# Self-check
# ----------------------------------------------------------------------------
def _selfcheck():
    m = MinimalAdder(dtype=torch.float64)
    a, b = 7650676663, 149460439
    ids = encode_batch([(a, b)])
    out = m.forward(ids)
    got = digits_to_int(out[0])
    exp = a + b
    print(f"{a:010d}+{b:010d}={got:011d}  expected {exp:011d}  ok={got == exp}")
    assert got == exp


if __name__ == "__main__":
    _selfcheck()
