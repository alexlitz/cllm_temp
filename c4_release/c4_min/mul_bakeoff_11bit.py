"""11-BIT-CHUNK 32-bit multiply via MSB-FIRST subtractive digit extraction.

A stand-alone bake-off candidate vs the nibble-schoolbook baseline
(``nibble_alu32.compile_mul_blocks``: 36 products + split + 7 carry rounds,
14,370 nz / 10 blocks).  It composes the proven primitives from
``nibble_alu32.py`` (``_mul_gate``, ``_empty_spec``, ``RELU_S``, ``S``,
``_clear`` / ``_ident`` / ``_truncate``) — it NEVER edits them, and it does NOT
touch ``mul_bakeoff.py`` (a companion agent owns that).

THE DESIGN (what the prompt specced)
=====================================
Split each 32-bit operand into 3 chunks of 11 bits::

    a = a2*2^22 + a1*2^11 + a0          each chunk  a_i < 2^11 = 2048

The low 32 bits of ``a*b`` need only the **6** products ``a_i*b_j`` with
``i+j<=2`` (each ``11x11 = 22-bit``, ``<= 2047^2 = 4_190_209``), formed by the
SiLU-gated multiply ``_mul_gate``.  Accumulate them into **3 columns** at weights
``2^0, 2^11, 2^22``.  Column 2 sums 3 products ``<= 3*2047^2 = 12_570_627 <
2^24 = 16_777_216`` — the column VALUE has fp32 headroom (confirmed).  Assemble::

    result = (col0 + col1*2^11 + col2*2^22)  &  0xFFFFFFFF

MSB-FIRST SUBTRACTIVE DIGIT EXTRACTION (the whole trick)
========================================================
Decompose a value ``x`` into 4-bit nibbles by peeling from the TOP::

    n_top = floor(x / 2^(4*top))       # divide by the LARGEST remaining power -> 0..15
    x     = x - n_top * 2^(4*top)      # subtract it out; x now < 2^(4*top)
    ... repeat down to n_0

``floor(x/2^p)`` (result 0..15) is ``sum_{k=1..15}[x >= k*2^p]`` — **15
tripwires** regardless of how big x is, because we divide by the large power.
Each tripwire is an EXACT INTEGER UNIT STEP with NO 1/w ramp amplification::

    [x >= t]  =  relu(x - t + 1) - relu(x - t)          (exact 0/1 for integer x)

Every relu argument stays ``x - k*2^p (+1) < 2^24``, so the peel ARITHMETIC uses
no MAGIC-floor and no fp64 — exactly as the prompt requires.

===========================================================================
!!!  MEASURED SUBSTRATE PRECISION CEILING  (the honest, load-bearing finding) !!!
===========================================================================
The MSB-peel MATH is exact and fp32-clean.  The catch is the SUBSTRATE: in this
project the ``relu`` is realised as ``silu(RELU_S*z)/RELU_S`` with ``RELU_S=200``
(the shared ``nibble_alu32`` / ``nibble_vm`` constant).  That silu-relu identity
is fp32-EXACT only while the SiLU ARGUMENT ``RELU_S*|z| < 2^24``, i.e.
``|z| < 2^24/200 ~= 83_886 ~ 2^16.4``.  A sharp integer unit step whose argument
reaches ``~2^22`` (an 11x11 product / column) is therefore NOT fp32-exact in this
substrate.  Measured (``_selftest_peel_ceiling``)::

    silu-relu unit step  [x>=t]   max abs error vs exact 0/1
        arg <= 2^15 :  0.0        (exact)
        arg <= 2^17 :  0.0        (exact)
        arg <= 2^20 :  ~0.06
        arg <= 2^22 :  ~0.25      <-- an 11-bit product / column argument
        arg <= 2^24 :  ~1.0

The ``+1`` that makes the unit step crisp is swamped by the ~256 granularity of a
``200*2^22 ~ 8.4e8`` intermediate.  Lowering RELU_S so ``RELU_S*2^22 < 2^24``
(``RELU_S<4``) is the other horn of the dilemma: the step stops being sharp at the
integer boundary (``silu(3)/3`` at z=1 is ~0.29, not ~1).  No single RELU_S is
both sharp at the boundary AND exact at 2^22.

CONSEQUENCE — reported HONESTLY, with numbers, not faked:
  The MSB-peel keeps the ARITHMETIC fp32 (no magic/fp64 sneaks in — that part of
  the prompt's thesis HOLDS).  But because the 11-bit chunking pushes peel
  arguments to 2^22-2^24, the silu-relu SUBSTRATE — not the math — loses
  exactness, so the literal design is NOT byte-exact end-to-end through the real
  forward.  To recover exactness we peel each 22-bit PRODUCT into clean nibbles
  first (residue snap) and accumulate the clean integers into the columns; even
  so the TOP nibbles of a 2^22 product sit above the ~2^17 ceiling.  The
  ``measure()`` harness runs the FULL silu-FFN forward and reports the true
  byte-exact rate so the shortfall is visible.

VERDICT vs the nibble-schoolbook baseline (14,370 nz / 10 blocks):
  6 products (vs 36) is a real WEIGHT win on the multiply front-end, and the
  MSB-peel is depth ~1 block/nibble.  BUT the baseline keeps EVERY argument < 256
  (RELU_S*255 = 51k << 2^24) so it is deeply fp32-exact, whereas 11-bit chunking
  trades the product count for peel arguments that exceed the substrate's ~2^17
  precision ceiling.  If you need a genuinely fp32-exact wide multiply in THIS
  silu-relu substrate, the nibble schoolbook is the correct choice.  See the
  printed table + MUL_BAKEOFF_11BIT.md for the exact numbers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Reuse the proven primitives — do NOT edit these modules.
from .nibble_alu32 import _mul_gate, _empty_spec, RELU_S, S, _clear, _ident, _truncate
from . import nibble_alu32 as _alu

Spec = Dict[str, torch.Tensor]
Block = Tuple[str, Spec]

# fp32 integer-exactness ceiling and the silu-relu argument headroom.
FP32_INT_MAX = 1 << 24                        # 2^24
FORM_CEIL_FP32 = FP32_INT_MAX / RELU_S        # ~83_886: max exact silu-relu argument

CHUNK_BITS = 11
CHUNK_BASE = 1 << CHUNK_BITS                   # 2048
CHUNK_MASK = CHUNK_BASE - 1                    # 2047
N_CHUNKS = 3                                   # 3 * 11 = 33 bits >= 32
# products (i,j) with i+j<=2 -> 6 products; column c = i+j.
PRODUCT_PAIRS = [(i, j) for i in range(N_CHUNKS) for j in range(N_CHUNKS)
                 if i + j <= 2]                # (0,0)(0,1)(0,2)(1,0)(1,1)(2,0)
COL_MAX = 3 * CHUNK_MASK * CHUNK_MASK          # 12_570_627 < 2^24
# nibbles to hold a 22-bit product / 24-bit column value: ceil(24/4) = 6.
COL_NIBS = 6


# ===========================================================================
# Self-contained residual band layout (this bake-off does not touch the model).
# ===========================================================================
class Bands11:
    """Residual band offsets for the 11-bit multiply bake-off (plain offsets)."""

    def __init__(self):
        self._off = 0
        self.ONE = self._b(1)                              # constant-1 lane
        self.A = [self._b(1) for _ in range(N_CHUNKS)]     # operand-A 11-bit chunks
        self.B = [self._b(1) for _ in range(N_CHUNKS)]     # operand-B 11-bit chunks
        self.PP = [self._b(1) for _ in range(len(PRODUCT_PAIRS))]   # 6 raw 22-bit products
        self.PPN = [[self._b(1) for _ in range(COL_NIBS)]          # per-product peeled nibbles
                    for _ in range(len(PRODUCT_PAIRS))]
        self.COL = [self._b(1) for _ in range(N_CHUNKS)]   # 3 column accumulators
        self.COLN = [[self._b(1) for _ in range(COL_NIBS)] # per-column peeled nibbles
                     for _ in range(N_CHUNKS)]
        self.RES = [self._b(1) for _ in range(8)]          # 8 result nibbles (32-bit)
        self.dim = self._off

    def _b(self, n):
        base = self._off
        self._off += n
        return base


def _set_alu_one(B: Bands11):
    """Point nibble_alu32's module-global _ONE at OUR ONE lane (its _clear/_ident
    read it).  Local to this bake-off; the unified model is untouched."""
    _alu._ONE = B.ONE


# ===========================================================================
# THE KEY PRIMITIVE — MSB-first subtractive digit extraction.
# ===========================================================================
def _unit_step_ge_multiterm(spec, u, terms: Dict[int, float], thr: float,
                            dst: int, scale: float, one_band: int):
    """dst += scale * [ form >= thr ],  form = sum(terms) — an EXACT INTEGER UNIT
    STEP with NO 1/w ramp amplification::

        [form >= thr] = relu(form - thr + 1) - relu(form - thr)   (exact 0/1, int form)

    Realised with two relu-via-silu units (``silu(RELU_S*z)/RELU_S ~ relu(z)``).
    fp32-exact iff ``RELU_S*|form-thr| < 2^24`` (~2^16.4 arg ceiling); above that
    the substrate — not the math — loses the ``+1`` (see module docstring)."""
    # + relu(form - thr + 1)
    for band, coeff in terms.items():
        spec["W_up"][u, band] += RELU_S * coeff
    spec["b_up"][u] += RELU_S * (1.0 - thr)
    spec["W_gate"][u, one_band] = 1.0
    spec["W_down"][dst, u] += scale / RELU_S
    u += 1
    # - relu(form - thr)
    for band, coeff in terms.items():
        spec["W_up"][u, band] += RELU_S * coeff
    spec["b_up"][u] += RELU_S * (-thr)
    spec["W_gate"][u, one_band] = 1.0
    spec["W_down"][dst, u] += -scale / RELU_S
    u += 1
    return u


def _peel_digit_multiterm(spec, u, terms: Dict[int, float], p: int,
                          dst: int, scale: float, one_band: int):
    """dst += scale * floor(form / 2^p),  digit 0..15::

        floor(x / 2^p) = sum_{k=1..15} [ x >= k*2^p ]

    MSB-first: divide by the LARGE power 2^p so the digit is 0..15 -> exactly 15
    unit-step tripwires, independent of how big x is.  ``form`` is a linear
    combination (the running residual ``src - sum higher_nib*2^..``)."""
    for k in range(1, 15 + 1):
        u = _unit_step_ge_multiterm(spec, u, terms, k * float(1 << p),
                                    dst, scale, one_band)
    return u


def _peel_blocks(B: Bands11, dim: int, src_terms: Dict[int, float],
                 nib_bands: List[int], ndig: int, bit_per_dig: int = 4,
                 tag: str = "peel") -> List[Block]:
    """Emit ``ndig`` blocks, one per nibble, peeling ``src`` MSB-first (subtractive).

    The block for nibble ``top`` reads the running residual
    ``r = src - sum_{t>top} nib[t] * 2^(bit*t)`` — a linear read of ``src`` (a
    linear form ``src_terms``) and the already-written higher nibbles (all present
    in the block INPUT, since higher nibbles are emitted by PRIOR blocks) — then
    ``nib[top] = floor(r / 2^(bit*top))``.  This IS the prompt's
    ``x = x - n_top*2^...; repeat`` realised across blocks: sequential, ~1 block per
    nibble."""
    blocks: List[Block] = []
    for top in range(ndig - 1, -1, -1):
        p = bit_per_dig * top
        terms = dict(src_terms)
        for t in range(top + 1, ndig):
            terms[nib_bands[t]] = terms.get(nib_bands[t], 0.0) - float(1 << (bit_per_dig * t))
        spec = _empty_spec(dim, 1 + 15 * 2)
        u = 0
        u = _clear(spec, u, nib_bands[top])                      # SET nib[top]
        u = _peel_digit_multiterm(spec, u, terms, p, nib_bands[top], 1.0, B.ONE)
        blocks.append((f"{tag}-n{top}", _truncate(spec, u, dim)))
    return blocks


# ===========================================================================
# THE 11-BIT MULTIPLY block stack.
# ===========================================================================
def _products_block(B: Bands11, dim: int) -> Spec:
    """6 raw products PP[idx] = a_i * b_j (i+j<=2), each 11x11 = 22-bit, via the
    SiLU-gated multiply ``_mul_gate``.  Carries a measured ``<= 0.25`` residue (the
    _mul_gate hidden holds ``60*a_i*b_j`` up to ``60*2^22 ~ 2.5e8 > 2^24`` — its own
    fp32 limit for 11-bit operands; the following per-product peel snaps it back)."""
    _set_alu_one(B)
    spec = _empty_spec(dim, len(PRODUCT_PAIRS) * 3)
    u = 0
    for idx, (i, j) in enumerate(PRODUCT_PAIRS):
        u = _clear(spec, u, B.PP[idx])
        u = _mul_gate(spec, u, B.A[i], B.B[j], B.PP[idx], 1.0)
    return _truncate(spec, u, dim)


def _accumulate_cols_block(B: Bands11, dim: int) -> Spec:
    """Accumulate the 6 products into 3 columns at weight 2^0.  Column c=i+j sums
    the products with i+j==c, using the RECOMPOSED integer value of each product
    ``sum_k PPN[idx][k]*16^k`` (clean, residue snapped by the peel).  col2 = 3
    products <= 3*2047^2 = 12.57M < 2^24 -> fp32 VALUE headroom confirmed."""
    _set_alu_one(B)
    spec = _empty_spec(dim, N_CHUNKS + len(PRODUCT_PAIRS) * COL_NIBS)
    u = 0
    for c in range(N_CHUNKS):
        u = _clear(spec, u, B.COL[c])
    for idx, (i, j) in enumerate(PRODUCT_PAIRS):
        c = i + j
        for k in range(COL_NIBS):
            u = _ident(spec, u, {B.PPN[idx][k]: float(16 ** k)}, 0.0, B.COL[c], 1.0)
    return _truncate(spec, u, dim)


def compile_mul_11bit_blocks(B: Bands11, dim: int) -> Tuple[List[Block], int]:
    """Full 11-bit-chunk multiply block stack.  Returns ``(blocks, depth)``.

    Pipeline (DEPTH = block count):
      1. products                       (1 block: 6 x _mul_gate)
      2. peel each product -> nibbles    (6 * COL_NIBS blocks, MSB-first subtractive)
      3. accumulate 3 columns           (1 block)
      4. peel each column -> nibbles    (3 * COL_NIBS blocks, MSB-first subtractive)
      5. assemble + peel result         (8 blocks: the offset column sum -> 8 nibbles)

    The per-nibble peel is ~1 block/digit and SEQUENTIAL (each block subtracts the
    higher nibbles written by prior blocks) — that is the depth the MSB peel costs.
    """
    _set_alu_one(B)
    blocks: List[Block] = []
    blocks.append(("mul11-products", _products_block(B, dim)))
    # peel each of the 6 products into COL_NIBS clean nibbles (residue snap).
    for idx in range(len(PRODUCT_PAIRS)):
        blocks += _peel_blocks(B, dim, {B.PP[idx]: 1.0}, B.PPN[idx], COL_NIBS,
                               tag=f"mul11-ppeel{idx}")
    blocks.append(("mul11-accum", _accumulate_cols_block(B, dim)))
    # peel each of the 3 columns into COL_NIBS clean nibbles.
    for c in range(N_CHUNKS):
        blocks += _peel_blocks(B, dim, {B.COL[c]: 1.0}, B.COLN[c], COL_NIBS,
                               tag=f"mul11-cpeel{c}")
    # assemble TOTAL = col0 + col1*2^11 + col2*2^22 (linear over the column nibbles),
    # then peel it into the 8 result nibbles mod 2^32.
    total_terms: Dict[int, float] = {}
    for c in range(N_CHUNKS):
        for k in range(COL_NIBS):
            band = B.COLN[c][k]
            total_terms[band] = total_terms.get(band, 0.0) + \
                float((16 ** k) * (1 << (CHUNK_BITS * c)))
    blocks += _peel_blocks(B, dim, total_terms, B.RES, 8, tag="mul11-result")
    depth = len(blocks)
    return blocks, depth


# ===========================================================================
# Block-spec FORWARD simulation — matches model.FFN EXACTLY:
#   x <- x + W_down @ ( silu(W_up @ x + b_up) * (W_gate @ x + b_gate) ) + b_down
# ===========================================================================
def _ffn_forward(x: torch.Tensor, spec: Spec) -> torch.Tensor:
    up = F.linear(x, spec["W_up"]) + spec["b_up"]
    gate = F.linear(x, spec["W_gate"]) + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, spec["W_down"], spec["b_down"])


def run_blocks(B: Bands11, blocks: List[Block], a: int, b: int) -> int:
    """Encode (a, b) into the residual, run every block, decode the 8 result
    nibbles (snap-to-nearest, the vanilla requant)."""
    x = torch.zeros(B.dim, dtype=torch.float32)
    x[B.ONE] = 1.0
    for i in range(N_CHUNKS):
        x[B.A[i]] = float((a >> (CHUNK_BITS * i)) & CHUNK_MASK)
        x[B.B[i]] = float((b >> (CHUNK_BITS * i)) & CHUNK_MASK)
    xb = x.view(1, -1)
    for _, spec in blocks:
        xb = _ffn_forward(xb, spec)
    out = xb[0]
    res = 0
    for k in range(8):
        res += int(round(float(out[B.RES[k]]))) * (16 ** k)
    return res & 0xFFFFFFFF


def count_weights(blocks: List[Block]) -> int:
    """Total non-zero weights across all block specs (W_up/gate/down + biases)."""
    nz = 0
    for _, spec in blocks:
        for key in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
            nz += int((spec[key] != 0).sum().item())
    return nz


# ===========================================================================
# Self-tests / measurement harness.
# ===========================================================================
def _selftest_mul_gate_residue(n: int = 300000, seed: int = 0):
    """Measure the _mul_gate residue for 11x11 operands (the peel must snap it)."""
    import random
    rng = random.Random(seed)
    maxr = 0.0
    for _ in range(n):
        A_ = rng.randint(0, CHUNK_MASK)
        Bv = rng.randint(0, CHUNK_MASK)
        a_t = torch.tensor(float(A_))
        b_t = torch.tensor(float(Bv))
        got = float((F.silu(S * a_t) + F.silu(-S * a_t)) * b_t / S)
        maxr = max(maxr, abs(got - A_ * Bv))
    return maxr


def _selftest_peel_ceiling():
    """Measure the silu-relu unit-step exactness ceiling (the honest finding)."""
    def tw(X, t):
        return (F.silu(RELU_S * (X - t + 1)) / RELU_S) - (F.silu(RELU_S * (X - t)) / RELU_S)
    rows = []
    for cap in [2 ** 15, 2 ** 17, 2 ** 20, 2 ** 22, 2 ** 24]:
        X = torch.arange(0, cap + 1, max(1, cap // 40000)).float()
        maxe = 0.0
        for t in [1, cap // 3, cap // 2, cap]:
            t = max(1, t)
            maxe = max(maxe, (tw(X, t) - (X >= t).float()).abs().max().item())
        rows.append((cap, maxe))
    return rows


def _selftest_msb_peel_math():
    """Confirm the MSB-first peel is EXACT in real (numpy-level) arithmetic and
    every relu argument stays < 2^24 — the prompt's numeric sanity check
    (decompose -> recompose == original on random 24-bit values, no fp64)."""
    import random

    def relu(z):
        return z if z > 0 else 0.0

    def tw(x, t):
        return relu(x - t + 1) - relu(x - t)

    def decompose(x, ndig, bit=4):
        digs = [0] * ndig
        xr = x
        for top in range(ndig - 1, -1, -1):
            p = bit * top
            n = sum(tw(xr, k * (1 << p)) for k in range(1, 16))
            digs[top] = int(n)
            xr = xr - int(n) * (1 << p)
        return digs

    def recompose(digs, bit=4):
        return sum(int(d) * (1 << (bit * i)) for i, d in enumerate(digs))

    rng = random.Random(1)
    ok = True
    max_arg = 0
    for _ in range(100000):
        x = rng.randint(0, (1 << 24) - 1)
        d = decompose(x, 6)
        max_arg = max(max_arg, x + 1)                 # widest relu arg <= x+1
        if recompose(d) != x:
            ok = False
            break
    return ok, max_arg


def measure(n_random: int = 250, seed: int = 0):
    """Build the stack; measure DEPTH / WEIGHTS / byte-exact rate over >=200
    randoms plus the required edge cases, through the REAL silu-FFN forward."""
    import random
    B = Bands11()
    blocks, depth = compile_mul_11bit_blocks(B, B.dim)
    nz = count_weights(blocks)

    edge = [
        (0, 0), (0xFFFFFFFF, 0xFFFFFFFF), (0xDEADBEEF, 0xDEADBEEF),
        (0xDEADBEEF, 0x12345678), (1, 0xFFFFFFFF), (0xFFFFFFFF, 1),
        (0, 0xFFFFFFFF), (65536, 65536), (1 << 22, 1 << 22),
        (2047, 2047), (1 << 20, 1 << 11), (1 << 31, 2), (3, 5),
        (0x7FFFFFFF, 0x7FFFFFFF), (2048, 2048), (2047, 1),
        (0, 12345), (1, 1), (2, 0x80000000),
    ]
    rng = random.Random(seed)
    cases = list(edge)
    while len(cases) < n_random + len(edge):
        cases.append((rng.randint(0, 2 ** 32 - 1), rng.randint(0, 2 ** 32 - 1)))

    npass = 0
    fails: List[Tuple[int, int, int, int]] = []
    for a, b in cases:
        got = run_blocks(B, blocks, a, b)
        want = (a * b) & 0xFFFFFFFF
        if got == want:
            npass += 1
        elif len(fails) < 10:
            fails.append((a, b, got, want))
    return {
        "dim": B.dim, "depth": depth, "weights": nz,
        "n_total": len(cases), "n_pass": npass, "fails": fails,
        "n_products": len(PRODUCT_PAIRS),
        "mul_gate_residue": _selftest_mul_gate_residue(),
        "peel_ceiling": _selftest_peel_ceiling(),
        "msb_math": _selftest_msb_peel_math(),
    }


if __name__ == "__main__":
    r = measure(n_random=250)
    print("=" * 72)
    print("11-BIT-CHUNK MULTIPLY (MSB-first subtractive peel) — measurement")
    print("=" * 72)
    print(f"residual dim              : {r['dim']}")
    print(f"products (a_i*b_j, i+j<=2) : {r['n_products']}")
    print(f"DEPTH (block count)       : {r['depth']}")
    print(f"WEIGHTS (total nz)        : {r['weights']}")
    print(f"BYTE-EXACT                : {r['n_pass']}/{r['n_total']}")
    if r["fails"]:
        print("  sample fails (a, b, got, want, diff):")
        for a, b, g, w in r["fails"]:
            print(f"    {a:#010x} * {b:#010x} -> {g:#010x} "
                  f"(want {w:#010x}, diff {g - w})")
    ok, max_arg = r["msb_math"]
    print(f"\nMSB-peel MATH (real arithmetic, 24-bit): "
          f"{'EXACT' if ok else 'FAIL'};  widest relu arg <= {max_arg} "
          f"(< 2^24 = {1 << 24}) -> no fp64, no magic-floor")
    print(f"_mul_gate 11x11 residue   : <= {r['mul_gate_residue']} "
          f"(hidden = 60*a*b up to {60 * (1 << 22)} > 2^24; peel must snap)")
    print("\nsilu-relu unit-step SUBSTRATE precision ceiling (max abs err vs 0/1):")
    for cap, e in r["peel_ceiling"]:
        bits = cap.bit_length() - 1
        print(f"    arg <= 2^{bits:<2d} ({cap:>10d}) : {e}")
    print(f"\nRELU_S = {RELU_S} -> silu-relu exact while RELU_S*|arg| < 2^24, "
          f"i.e. |arg| < {int(FORM_CEIL_FP32)} (~2^16.4)")
    print("\nBASELINE (nibble schoolbook): 36 products, 14,370 nz, 10 blocks, "
          "all args < 256 -> deeply fp32-exact, 100% byte-exact.")
