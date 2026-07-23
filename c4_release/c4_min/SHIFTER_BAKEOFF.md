# SHIFTER DESIGN BAKEOFF — sweeping GRANULARITY

*How coarse should a 32-bit SHL/SHR gadget's shift be?*  This bakeoff builds the
same shift at five granularities — barrel select, bit, nibble, byte, 16-bit word
— and measures each on **depth** (blocks), **weights** (non-zero params),
**fp32-exactness**, and **byte-exactness** vs the reference interpreters.

Code: [`shifter_bakeoff.py`](shifter_bakeoff.py) (build + measure), gated by
[`test_shifter_bakeoff.py`](test_shifter_bakeoff.py).  Run the table with

```
OMP_NUM_THREADS=2 PYTHONPATH=$(pwd) python -m c4_min.shifter_bakeoff
```

It reuses the landed bit-plane log-shifter primitives from
[`nibble_bitwise.py`](nibble_bitwise.py) (the 5-stage mux, the `n>=32->0` keep
bit, the shared bit-plane extraction) and the fp32-exact SwiGLU emitters from
[`nibble_alu32.py`](nibble_alu32.py) (`_empty_spec`, `_floor_div_pow`,
`_mul_gate`, `RELU_S`).  Neither file is edited.

## The granularity axis

A shift by `n` over a value split into **chunks of `w` bits** factors as

```
n  =  (n // w) whole CHUNKS   (COARSE)  +   (n mod w) bits   (FINE)
```

* **COARSE** is a *log-shift by whole chunks* over the `32/w` chunk-planes:
  `ceil(log2(32/w))` conditional 2:1 mux stages (the exact `nibble_bitwise` mux,
  just on chunk-planes).  **Coarser chunks → fewer chunk-planes → fewer stages.**
* **FINE** shifts each chunk by `r = n mod w` bits and spills the boundary-
  crossing bits into the neighbour.  For SHL that is `chunk * 2**r` split into
  `(low w bits, carry-out) = (p mod 2**w, floor(p/2**w))`.  **Coarser chunks →
  WIDER fine shift:** `r` ranges over `0..w-1` and the per-chunk product reaches
  `(2**w-1)·2**(w-1)` — 120 for a nibble, 32640 for a byte, ~2**31 for a word.

The whole question is whether the coarse stage-savings outrun the widening fine
shift's cost — or whether the fine shift eats the savings (and, past a point,
leaves fp32).

### The MSB-first peel (byte / 16-bit fine split)

The fine split `(p mod 2**w, floor(p/2**w))` is realised as ONE relu carry
staircase `co = floor(p/2**w)` plus a direct `lo = p - 2**w·co` (`p` read via
silu-identity, no second staircase).  The staircase **height** is
`kmax = max_product // 2**w` — 7 for a nibble, 127 for a byte, ~2**15 for a word
— which is exactly the weight cost that grows with chunk width.  Every relu
argument is `<= RELU_S · max_product`; for a word that is ~4e11 ≫ 2**24, so the
word peel leaves fp32.

## Measurement

* **DEPTH** — number of sequential SwiGLU blocks (a shift is a pipeline: each mux
  stage reads the previous stage's buffer, so dependent quantities can't share a
  block — every hidden unit reads the block *input*).
* **WEIGHTS** — total non-zero entries across all blocks' `W_up/b_up/W_gate/
  b_gate/W_down/b_down`.
* **fp32-exact?** — flagged NO if any block forms a hidden argument `>= 2**24`.
* **byte-exact** — each gadget is simulated on **just the bit/chunk planes it
  touches** (a tiny residual, NOT a DIM-8192 forward), over the edge grid
  `x ∈ {0x80000000, 0xFFFFFFFF, 0xDEADBEEF, 0x1, 0x0, 0xF0F0F0F0}`,
  `n ∈ {0,1,7,15,16,31,32,40}` + random `(x,n)`, vs `ref_interpret(mask=
  0xFFFFFFFF)` (32-bit, `n>=32->0`, unsigned SHR).  `isa8` cross-checks the low
  byte against the 8-bit `isa.interpret`.
* All variants output **result PLANES** (bit/chunk), decoded fp32-exactly.  A
  scalar `sum 2**i` recompose is deliberately omitted — `2**31 > 2**24` would
  make the scalar itself lossy (the same reason the model's own nibble→scalar
  runs in fp64 for width-32).

## Results

*(depth = blocks, weights = non-zero params, sorted by weights)*

### SHL

| variant | depth | weights (nz) | fp32-exact | byte-exact |
|---|---:|---:|:---:|:---:|
| **nibble granular** | 8 | **4 140** | yes | 182/182 |
| bit granular *(landed)* | 8 | 5 269 | yes | 182/182 |
| byte granular | 7 | 10 600 | yes | 182/182 |
| barrel select *(retired)* | 2 | 10 848 | yes | 182/182 |
| 16-bit granular | 6 | 1 050 722 | **NO** (~4.3e11) | 106/182 |

### SHR

| variant | depth | weights (nz) | fp32-exact | byte-exact |
|---|---:|---:|:---:|:---:|
| **nibble granular** | 8 | **3 971** | yes | 182/182 |
| bit granular *(landed)* | 8 | 5 269 | yes | 182/182 |
| byte granular | 7 | 10 515 | yes | 182/182 |
| barrel select *(retired)* | 2 | 10 848 | yes | 182/182 |
| 16-bit granular | 6 | 1 050 679 | **NO** (~4.3e11) | 117/182 |

SHL and SHR are near-symmetric at every granularity (the retired barrel's
historical SHR = 3× SHL asymmetry is gone — the mux/peel structure is
direction-symmetric).

## Verdict

**The smallest bit-exact fp32 shifter is the NIBBLE-granular one** — ~4.0K nz
(SHR) / ~4.1K nz (SHL), a **~22–25 % weight saving over the landed bit-granular
shifter** (5 269 nz) at the **same depth (8 blocks)**, fully byte-exact and
fp32-exact.  Coarsening from bit → nibble is a real win: the coarse stage count
drops (32 bit-planes and 5 stages → 8 nibble-planes and 3 stages), while the fine
shift is still trivial (`nib·2**r <= 120`, split with a 7-tall staircase).

**But coarser is NOT monotonically smaller — the crossover flips at the byte.**

* **bit → nibble → byte** *(depth 8 → 8 → 7)*: depth drops, but weights go
  5 269 → ~4 000 → ~10 500.  The byte's fine shift is where it turns: the
  per-byte product reaches 32640 and its carry staircase is **127 units tall per
  byte-plane**, so the fine split alone dominates and the byte gadget is **~2×
  heavier** than bit-granular despite being one block shallower.  The coarse
  savings (4 byte-planes, 2 stages) are real but far outweighed by the fine-shift
  staircase.
* **byte → 16-bit** *(depth 7 → 6)*: the coarse shift is now trivial (2 word-
  planes, 1 stage), but the fine shift is catastrophic — the per-word product
  reaches ~2**31, its carry staircase is **~32 768 units tall**, blowing weights
  to **~1.05 M nz**, and — decisively — the product **exceeds fp32's 2**24 unit
  precision**, so the 16-bit shifter is **neither fp32-exact nor byte-exact**
  (small shifts still land; anything crossing a word boundary is wrong).

So the ordering by weight is **nibble < bit < byte < barrel ≪ 16-bit**, and the
answer to *"does 16-bit beat byte beat nibble beat bit?"* is **no — the fine
shift eats the coarse savings past the nibble.**  The nibble sits exactly at the
sweet spot: `w = 4` matches the spec's native 4-bit nibble representation, so the
fine shift stays inside a single nibble digit (`< 16`, fp32-trivial) while still
halving the coarse stage count relative to per-bit.

### Practical note

The nibble-granular shifter is a **drop-in ~1.1K-nz improvement** over the landed
bit-granular design at equal depth, all-fp32, byte-identical to the reference —
the only variant here strictly better than what's landed.  The byte and 16-bit
variants are **worse on every axis that matters** (weights, and for 16-bit also
fp32/correctness) and are kept only to document the crossover.  The barrel select
is the honest "what we replaced" reference: shallow (2 blocks) but a ~10.8K-nz
cross-product with no depth headroom to spare.
