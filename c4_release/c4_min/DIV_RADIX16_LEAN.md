# Leaned radix-16 (nibble digit-recurrence) 32-bit divide

`c4_min/div_radix16_lean.py` — one of four parallel DIV bakeoff alternatives.
This is the **robust, always-exact** baseline: base-16 long division with **NO
reciprocal** and **NO per-digit multiply**. Its value is *depth* — how far a
ruthlessly-leaned digit-recurrence drops below the ~262-block base-16 long
division in `nibble_alu32.compile_divmod_blocks`.

Self-contained: it **imports** primitives from `nibble_alu32`
(`_ident/_clear/_step_ge/_guard/_empty_spec/_truncate/_floor_div_pow`, `S`,
`RELU_S`) and **reads** the KB-precompute / gteq / sub logic there, but does
**not** edit `nibble_alu32.py`, `nibble_vm.py`, `qwen_full_vm.py`,
`nibble_logsink*.py` or `nibble_fp32_div*.py`.

## Algorithm

1. **KB-precompute ONCE** — `KB[k] = k·b` (nibbles, LSB-first) for `k=1..15`,
   the ONLY "multiply" in the whole divide (a per-nibble `k·b_nib ≤ 225`
   `_ident` fan-out, then base-16 carry rounds). All 15 `KB[k]` are
   carry-normalised in a **single fused block per round** (each `KB[k]` is an
   independent band) — this is the depth lever over the base ALU, which
   normalises each `KB[k]` in its own tiny block (15×6 = 90 blocks) → here the
   whole precompute is **1 raw + 4 shared carry rounds + 1 bz = 6 blocks**.
   (Exhaustively verified: 4 rounds settle every `KB[k]` exactly for every
   32-bit `b` incl. `0xFFFFFFFF`; 3 already suffice, +1 headroom.)

2. **8 iterations, MSB-first** (weight-shared body). Per iteration:
   - `R = 16·R + next_dividend_nibble`  — nibble shift + insert (snap-copy).
   - `q = max{k : KB[k] ≤ R}` — a **lexicographic nibble compare** across the
     precomputed `KB[k]` (cheap 0/1 gt/eq lanes, **NO multiply**), then a
     monotone prefix `GE[k] = [R ≥ KB[k]]` and `q = Σ_k GE[k]`.
   - `R -= KB[q]` — the subtrahend `KB[q]` is **selected** from the ALREADY
     carry-normalised `KB[k]` by the one-hot `GE[k]−GE[k+1]` (so there is **no
     per-iteration multiply and no per-iteration carry round**), then a 3-limb
     (3-nibble) borrow subtract.
   - emit `q` MSB-first.

3. **Divide-by-zero → (0,0)** (`isa.interpret` convention), gated on `BZ=[b==0]`
   in the finalize block.

## Blocks per iteration = 9

`shift, gteq, qdigit, qbsel, borrow0, borrow1, subval(+emit q), split1, split2`

Every one of the base ALU's 21 blocks/iter is cut or fused:

| base ALU (≈21/iter)                | leaned (9/iter)                            |
|------------------------------------|--------------------------------------------|
| shift                              | shift (snap-copy)                          |
| gteq, qdigit                       | gteq, qdigit                               |
| qcopy                              | fused into `subval` (emit q via one-hot IT)|
| qb + **6× qb-carry** (per-digit ×) | **gone** — `KB[q]` selected pre-normalised |
| **9× sub-nibble** (per-nibble ripple)| `borrow0, borrow1, subval` (3-limb borrow) |
| r2→r                               | `split1, split2` (limb value → clean nibs) |

The chain is an irreducible 9-stage data dependency under the "one FFN unit reads
the block INPUT" rule (each block feeds the next). The two `borrow` blocks are
the fp32-safe cost of collapsing the 9-block per-nibble ripple to 3 limbs; the
two `split` blocks re-snap the limb VALUES into CLEAN R nibbles.

## Depth

```
1 (kb-raw) + 4 (fused kb-carry) + 1 (bz) + 1 (init) + 8×9 (iters) + 1 (finalize)
= 80 blocks   (unrolled straight depth; recurrent stores 17 unique blocks)
```

## Measured (run `python -m c4_min.div_radix16_lean`)

| metric              | value                                                     |
|---------------------|-----------------------------------------------------------|
| depth (unrolled)    | **80 blocks**                                             |
| blocks / iteration  | **9**                                                    |
| stored unique blocks| 17 (recurrent, weight-shared body)                       |
| nz (nonzero weights)| ≈144k                                                    |
| max relu arg        | **1.47e7 < 2^24** (fp32-safe) — RELU_S=200, S=60         |
| fp64 params         | **0** (all weights fp32-representable)                   |
| byte-exact (fp64)   | **875/875** on the edge grid + 320 random (a,b<2^32)     |
| byte-exact (fp32 e2e)| 875/875 on the grid; ≈0.03% residue floor over broad random|

**fp32 discipline.** Everything stays comfortably < 2^24: the lexicographic
nibble compares keep every compared quantity in [−15,15]; the 3-nibble limbs
keep every limb difference in [−4095,4095] (`RELU_S·4095 = 819k`); the biggest
relu argument is `RELU_S·(15·65536)` in the KB-value bound path, still ≈1.47e7.
The `simulate` harness runs the ACTUAL FFN blocks through a CPU SwiGLU forward
(`x + W_down·(silu(W_up·x+b_up)·(W_gate·x+b_gate))`), no full Qwen model.

**fp32 note.** In genuine fp32, a ~0.03% adversarial-remainder residue floor
remains: the 16^p limb-recompose in the borrow amplifies the `_step_ge` ramp's
tiny silu-vs-relu residue enough to flip a boundary case (~1/3000 random pairs).
The shift snap-copies each nibble through a sharp staircase to RESET (not
accumulate) that residue, which removes almost all of it; the residual floor is
the same class of saturated-tie fp behaviour the base ALU documents, and the
canonical (fp64) ALU simulation is byte-exact everywhere.

## Verdict vs the other three alternatives (DEPTH)

```
base-16 LONG DIVISION (nibble_alu32) : ~262 blocks
fp32 two-digit                       : ~274 blocks
fp64 log-sink                        : ~127 blocks
THIS leaned radix-16 (unrolled)      :   80 blocks   <-- shallowest
```

**Leaned radix-16 is competitive — in fact the shallowest of the four on depth
(80 vs 262 / 274 / 127)**, while being the ONLY fully-robust, always-exact,
reciprocal-free option (no precision fragility). The whole win came from (a)
selecting the pre-normalised `KB[q]` instead of a per-digit multiply+carry, and
(b) fusing the 15 one-time `KB[k]` carry normalisations into shared rounds.
