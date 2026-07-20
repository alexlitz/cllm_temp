# I/O Position Substrate — multi-slope BOS position signature + nibble-cascade offset extractor

Date: 2026-07-14 · Branch: `nibble-io-position` · Base: `nibble-integrated` (`d7ce2ef6`)

## What this is

The **shared position-tracking substrate** underneath GETCHAR / PUTCHAR / argv,
completing `docs/BLOG_SPEC.md` §"Printing and Reading Input" (lines 696–717) and
§"Position Offset Calculation" (lines 718–739). Every neural-I/O read needs two
things the spec builds out of vanilla-transformer pieces:

1. for each I/O token, its **absolute sequence position** — from a *multi-ALiBi-
   slope BOS position signature* (§704–712);
2. given a target scalar **offset N**, the **byte at position N** in the I/O
   buffer — the signature lets a query retrieve it, and a *base-16 nibble cascade*
   (§726–730) turns the scalar offset into explicit nibbles to index with.

This is the reusable I/O foundation (§851: *"position-tracking heads locate the
markers, and a nibble cascade extracts the offset to index into the input
buffer"*). RoPE binary-distance matching (the alternative positional scheme,
§741–746) is already built on `nibble-rope` (`c4_min/nibble_rope.py`); this task
builds the **ALiBi** side that the foundation transformer (`blogspec_model.Attn`)
actually uses.

| file | role |
|------|------|
| `c4_min/nibble_io_position.py`      | the substrate: `position_signature`/`position_from_signature`/`invertible_range`, the `nibble_cascade_offset` extractor, `IOPositionBuffer` (byte@offset retrieval, argmax + real softmax1), a vanilla-head forward demo |
| `c4_min/test_nibble_io_position.py` | 21 tests proving the spec's claims + byte-exact retrieval vs a plain-list reference |

Run: `PYTHONPATH=<repo> python -m pytest c4_min/test_nibble_io_position.py` → **21/21**.
Demo: `python -m c4_min.nibble_io_position`.

## 1. Multi-slope BOS position signature (§704–712)

> *"The specific mechanism uses multiple attention heads sharing the same fixed
> key at BOS but with different ALiBi slopes. For a token at distance d from BOS,
> head k with slope m_k produces an attention bias of −m_k·d, giving a score
> contribution of exp(−m_k·d). With enough heads at different slopes, the tuple
> (exp(−m_1·d), …, exp(−m_K·d)) uniquely identifies position d … To retrieve the
> byte at position N, we construct a query that matches the exponential signature
> of distance N."* (§710)

`position_signature(d)` returns exactly that tuple. The heads are ordinary ALiBi
heads: the query·key dot product is a fixed constant (same BOS key every head),
ALiBi subtracts `m_k·d`, so the pre-softmax logit is `const − m_k·d` and the
un-normalised weight is `exp(const)·exp(−m_k·d)` — the signature up to a shared
`exp(const)`. The slopes are the foundation's geometric ladder
`m_k = 2^(−8/n·(k+1))` (§307–311), **identical** to
`blogspec_model.Attn.alibi_slopes` (tested).

Because every `m_k>0` and the slopes are distinct, `d ↦ signature(d)` is strictly
monotone in each component and **injective** — so the tuple *is* an absolute-
position code. `position_from_signature` inverts it by reading the position off
the **slowest-decay** head (the one that stays alive longest, §736) and rounding.

**Positions are ALiBi distance from the marker, not the KV slot index** (§712):
evicting non-I/O tokens leaves the I/O tokens' distances unchanged. The tests
confirm the same buffer at different marker positions reads back identically.

### The honest range limit (§734 numerical margin)

The tuple can only encode positions while the signature stays representable.
The slowest head (`m_min ≈ 0.0039` for 8 heads) leaves the **normal** float64
range once `m_min·d > 708` (`exp(−708) ≈ 2.2e-308`). So the reliably-invertible
range is `d_max = ⌊708 / m_min⌋` — about **181 000** positions for the default
8-head ladder. `invertible_range()` returns it; inversion is byte-exact over the
whole range (tested with 500 random draws) and the boundary is exact.

This is the spec's *"with enough heads at different slopes"* (§710) made precise:
the **distinguishing power** comes from having several slopes, but the **range**
is set purely by the *smallest* slope and is tunable independently. Passing a
gentler `m_min` (e.g. `1e-4`) widens the range past 7 million positions — tested.

## 2. Base-16 nibble-cascade offset extractor (§726–730)

The offset `N = pos − marker_pos` arrives as a **scalar**; to index memory it
must become explicit nibbles. The spec's shallow (8-layer) extractor works in
base 16 — at nibble `j` from `j=7` down to `0`:

```
d_j = Σ_{k=1..15} step(r − k·16^j)      (digit,    output weight 1)
r  ← r − d_j·16^j                        (residual, output weight 16^j)
```

Each layer counts *how many copies of `16^j` fit in the residual* — the same
threshold-counting staircase DIV uses for its quotient digit (§728, *"the same
digit-extraction pattern as long division"*). `nibble_cascade_offset` therefore
**reuses the shared SiLU `step` primitive** (`_step_ge`, identical to
`nibble_muldivmod`'s `_step_ge`), and the digit-emit / residual-subtract share
one threshold bank (§730).

- **8 layers reconstruct any full 32-bit offset**, byte-exact vs a plain
  bit-shift decomposition — tested on 2000 random 32-bit values incl.
  `0xDEADBEEF`, `0xFFFFFFFF`, `0x80000000`.
- The residual terminates at 0 for a well-formed 8-nibble offset (tested).
- **fp discipline (§734):** the top-nibble thresholds reach `15·16^7 ≈ 4·10^9`,
  past fp32's `2^24` unit precision, so — with the spec's blessing (*"We can of
  course use doubles"*, §595) — the staircase runs in **float64**, and the step
  scale is **per-nibble**: sharp (`S=200`) at the unit-spaced low nibble where
  the classification is hardest (§736), gentle (`S=8`) above where `S·τ` would
  otherwise blow up. The low-nibble sharpness is tested across all 16 digits.

## 3. Retrieve the byte at position N (§710, §715–717)

`IOPositionBuffer` is an I/O buffer addressed by **absolute sequence position** —
the substrate a GETCHAR / argv read uses. Bytes are appended starting at
`marker_pos + 1` (§706); `read_offset(N)` fetches the byte at buffer offset `N`
by building a query with that position's ALiBi signature and letting the head
select the matching store (*"construct a query that matches the exponential
signature of distance N"*, §710). Two read paths, both verified byte-for-byte
against the plain-list reference:

- **`read_offset`** — argmax over the position-match logits `−Σm_k·|d_q − d_s|`
  (the additive-bias view of the signature match: 0 at the exact position,
  strictly negative elsewhere).
- **`read_offset_softmax1`** — the **actual** `blogspec_model.softmax1` (§491 ZFOD)
  arithmetic. The raw match logit is 0 at the match, which would *tie* the
  softmax1 `+1` sink, so the exact match is lifted to `+MATCH_GAIN` and the
  per-unit penalty is sharpened (normalising out the slope sum), so the match
  wins dominant weight while a 1-unit miss lands negative. This is the analogue
  of the RoPE content key's `−scale` mismatch bias (`nibble_rope.load_softmax1`)
  and the spec's binary-key `−scale` per 0-bit.

Both paths return **0 on out-of-range / unwritten offset** (zero-fill-on-demand:
no store's signature matches the target, the `+1` sink dominates, §491) — tested.

`read_scalar_offset` chains the two halves end-to-end: a scalar offset → nibble
cascade → reassemble → `read_offset`, i.e. the full GETCHAR index path (§851).

### Real vanilla-head forward

`demo_position_head_forward` drives the **real** `blogspec_model.Attn`
(softmax1 + ALiBi, no code changes) with a hand-set residual: a marker token,
then a run of byte tokens, then a query aligned to the shared BOS key. Each
head's ALiBi bias `−m_k·d` makes the byte token at the queried distance win under
softmax1, and its value lane carries the byte. It retrieves the byte at the
target position **byte-exactly through `model.forward`** (tested for one buffer
and for every offset of a second), proving the substrate runs on the vanilla
foundation head, not just in numpy-space.

## Test coverage (21/21)

| group | proves |
|-------|--------|
| slopes | signature heads ARE the foundation ALiBi heads; components are exactly `exp(−m_k·d)` |
| signature | injective over 20k positions; inversion exact (fixed + 500 random) across the whole invertible range; strictly monotone decay |
| range | `708/m_min` bound; exact at the boundary; a gentler slope extends it >10× |
| cascade | matches bit-shift reference; reassembles exactly over 2000 random 32-bit offsets; residual → 0; 8 layers; low-nibble sharpness (all 16 digits) |
| retrieval | argmax + real-softmax1 both byte-exact vs plain-list reference (fixed + 60 random buffers); ZFOD → 0 out of range / empty; position-relative not slot-relative |
| end-to-end | scalar-offset → cascade → byte exact |
| real head | byte@N via the vanilla `blogspec_model.Attn` forward, exact |

## How the consumers use it

- **GETCHAR** (§715, task #553): the input-block marker sets the reference; each
  input token appends to an `IOPositionBuffer`; a read builds the query at the
  target distance and pulls the byte — `read_scalar_offset` is the whole path.
- **PUTCHAR** (§717, task #550): the output buffer is addressed the same way in
  reverse; the output token's absolute-position − start gives its offset, which
  the cascade turns into an index.
- **argv** (§751–793): read exactly as user input — the `__argv_setup` loop does
  repeated `getchar()` reads, i.e. repeated position-addressed buffer fetches.

All three share this one substrate: the multi-slope signature for *where*, the
nibble cascade for *which nibble-index*, and softmax1 ZFOD for *read-0-on-miss*.
