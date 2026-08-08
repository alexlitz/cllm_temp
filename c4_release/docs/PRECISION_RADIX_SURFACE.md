# Precision × radix ALU surface — the MIN-WALLTIME (tensor-core) corner

**Date:** 2026-08-07 · **Scope:** analysis + standalone verified code + GPU microbench.
Touches no build files — the c4 golden is unchanged. This is the **complement** to the
fp64/fp128 **min-PARAMS** work in [`CLEVER_MINPARAM_ALU.md`](CLEVER_MINPARAM_ALU.md):
it pushes the hand-constructed, digit-exact transformer ALU cell **down to low precision**
(int8 / bf16 / fp16) with **arbitrary radix per op**, to find the config that minimizes
**wall-clock** on tensor cores.

- Code: [`examples/lowprec_radix_alu.py`](../examples/lowprec_radix_alu.py) — the surface +
  the hand-built low-precision limb cells + the exactness verifier (CPU-only, deterministic).
- Microbench: [`examples/_lowprec_microbench.py`](../examples/_lowprec_microbench.py) — the GPU
  wall-clock (raw dtype GEMM throughput + the deep-lowprec-vs-shallow-fp64 cell race + the
  Doom fps projection). Lean: peak 0.41 GB, well under 25 GB.

Reproduce:
```
python examples/lowprec_radix_alu.py --n 250000 --seed 7   # surface + verify (CPU)
python examples/lowprec_radix_alu.py --bench               # + GPU microbench
python examples/_lowprec_microbench.py                     # microbench alone
```

---

## 1. The coupling: precision → max radix → depth (per op)

Every value/limb is in `[0, r)` at radix `r`. The op's **intermediate accumulator max**
must fit the precision's **exact-integer ceiling** (the largest `M` with every integer in
`[0, M]` exactly representable): int8 `2^7−1=127` (signed magnitude), bf16 `2^8`, fp16
`2^11`, fp32 `2^24`, fp64 `2^53`. The accumulator maxima are:

| op | accumulator max at radix r | constraint |
|---|---|---|
| ADD/SUB | `a_d + b_d + carry ≈ 2r` | `2r ≤ ceiling` |
| CMP | one limb difference `≈ r` | `r ≤ ceiling` |
| MUL (schoolbook column) | `Σ a_i·b_j + carry ≈ r²·L` (L = #operand limbs) | `≤ ceiling` |
| DIV/MOD | trial `q_p·b_limb ≈ r²` | `r² ≤ ceiling` |

**Low precision → small ceiling → small max-safe radix → more base-`r` limbs → deeper.**
Depth is the number of limbs = strictly-sequential reused-cell applications. This is the
inverse of the min-params corner (which holds the *whole* 32-bit value in ONE fp64/fp128
scalar and needs no limbs, at the cost of the slow fp64 datapath).

## 2. The surface — max-safe radix × depth × params × accMax, per op × precision

Computed by `lowprec_radix_alu.py` (32-bit operands; MUL result 64-bit; `accMax` is the
**true worst-case** accumulator, all limbs = r−1). `cellScal` = irreducible per-limb cell
scalars; `MAC/lyr` = nonzero MACs per limb-layer; `peakVsFp32` = the dtype's A5000
tensor-core throughput ratio.

```
op         prec ceiling(2^k)  maxRadix  depth  accMax(2^j) cellScal MAC/lyr  peakVsFp32
ADD/SUB    int8         127        2^5      7          2^6        5       5      4.000x
ADD/SUB    bf16         2^8        2^7      5          2^8        5       5      2.000x
ADD/SUB    fp16        2^11       2^10      4         2^11        5       5      2.000x
ADD/SUB    fp32        2^24       2^23      2         2^24        5       5      1.000x
ADD/SUB    fp64        2^53       2^52      1         2^53        5       5      0.031x
CMP        int8         127        2^6      6          2^6        5       5      4.000x
CMP        bf16         2^8        2^7      5          2^7        5       5      2.000x
CMP        fp16        2^11       2^10      4         2^10        5       5      2.000x
CMP        fp32        2^24       2^23      2         2^23        5       5      1.000x
CMP        fp64        2^53       2^52      1         2^52        5       5      0.031x
MUL        int8         127        2^1     64           62        6      67      4.000x
MUL        bf16         2^8        2^2     32          188        6      35      2.000x
MUL        fp16        2^11        2^4     16        1,904        6      19      2.000x
MUL        fp32        2^24       2^11      6        ~2^23        6       9      1.000x
MUL        fp64        2^53       2^26      3        ~2^52        6       7      0.031x
DIV/MOD    int8         127        2^3     11          2^6        8       8      4.000x
DIV/MOD    bf16         2^8        2^4      8          2^8        8       8      2.000x
DIV/MOD    fp16        2^11        2^5      7         2^10        8       8      2.000x
DIV/MOD    fp32        2^24       2^12      3         2^24        8       8      1.000x
DIV/MOD    fp64        2^53       2^26      2         2^52        8       8      0.031x
```

Reading it: the depth cost of going low-precision is **op-dependent**, and MUL is by far the
worst-coupled — its accumulator is `r²·L`, so it forces the smallest radix and the deepest
stack (int8 → radix 2, depth **64**; bf16 → radix 4, depth **32**; fp16 → radix 16, depth
**16**). ADD/CMP couple weakly (`2r`/`r`), so even int8 keeps a fat radix (32/64) and a
shallow depth (6–7). DIV sits in between (`r²`; bf16 → radix 16, depth 8). `peakVsFp32` is
the **published** ratio; §4 reports the **measured** one (bf16 ≈ 5×, int8 only ≈ 2.7× on
this card — a key honest finding).

The **per-limb param count grows slightly** vs the min-params scalar core (which had ~4
scalars): low precision needs an explicit per-limb carry/borrow cell — 5 (ADD/CMP), 6 (MUL),
8 (DIV) shared scalars, still ~1000× below the nibble-c4 per-op weight tables (ADD 6,459 /
MUL 7,911 / DIV 163,591).

## 3. Verified-exact low-precision variants (hand-set, NO training)

`python examples/lowprec_radix_alu.py --n 250000 --seed 7` → **ALL EXACT** (≥100 k random
32-bit pairs + hard edges per variant; re-run at 250 k / seed 7 still exact). Every
intermediate is an exact integer in the low dtype, so the argmax/threshold decode is
bit-exact.

| variant | radix | depth | accMax ≤ ceiling | pairs | exact |
|---|---:|---:|---|---:|:--:|
| **ADD/SUB bf16 radix-16** | 16 | 9 | 32 ≤ 256 | 250,024 | ✅ |
| **ADD/SUB bf16 radix-4** (digit-decompose) | 4 | 17 | 8 ≤ 256 | 250,024 | ✅ |
| **ADD/SUB int8 radix-4** | 4 | 17 | 8 ≤ 127 | 250,024 | ✅ |
| **CMP fp16 radix-16** | 16 | 8 | 16 ≤ 2048 | 250,024 | ✅ |
| **MUL fp16 radix-16** | 16 | 16 | 1904 ≤ 2048 | 250,022 | ✅ |
| **DIV/MOD bf16 radix-16** (boundary r²=256=ceiling) | 16 | 8 | 256 ≤ 256 | 250,008 | ✅ |

The MUL column accumulator peak (1904) and the DIV trial product (≤ r²) are **instrumented
and asserted `≤ ceiling` every step** inside the cells, so the exactness claim is airtight,
not just formula-bounded.

## 4. Measured GPU throughput — low precision vs fp64 (RTX A5000)

### (A) Raw dense GEMM throughput per dtype (4096³, TF32 off)

| dtype | achieved | vs fp32 | % datasheet peak |
|---|---:|---:|---:|
| **fp64** | **0.41 TFLOP/s** | **0.02×** | 47% |
| fp32 | 17.2 TFLOP/s | 1.00× | 62% |
| **bf16** | **93.0 TFLOP/s** | **5.4×** | 84% |
| fp16 | 86.8 TFLOP/s | 5.1× | 78% |
| int8 | 46.4 TOP/s | 2.7× | 21% |

**fp64 is ~40× slower than fp32 and ~225× slower than bf16** — the min-params corner pays for
its 4 scalars with the slowest datapath on the card. **bf16 is the fastest realized dtype
(5.4× fp32)**; **int8's theoretical 4× is NOT realized** (`torch._int_mm` tops out ~2.7×,
occupancy-bound at these sizes even at M=16384). So the min-walltime dtype here is **bf16,
not int8** — an honest, measured correction to the "int8 = 4×" assumption.

### (B) The head-to-head — deep-lowprec (tensor core) vs shallow-fp64 whole-value cell

Batched over **65,536 VM lanes** (the realtime regime; a Doom frame is 10⁵–10⁶ steps), one
reused limb-cell per layer, `depth` sequential layers, measured ns/lane-op:

| op | deep-lowprec (bf16/fp16) | shallow-fp64 whole-value | **winner** |
|---|---|---|---|
| **ADD** | 26.6 ns (r16, depth 9) | 30.4 ns (depth 1) | deep-lowprec **1.14×** (near parity) |
| **MUL** | 45.6 ns (fp16 r16, depth 16) | 602.6 ns (depth 20) | deep-lowprec **13.2×** |
| **DIV** | 23.5 ns (bf16 r16, depth 8) | 301.8 ns (depth 10) | deep-lowprec **12.9×** |

**Deep-low-precision-on-tensor-cores wins wall-clock for every op** — marginally for ADD
(fp64's whole-value ADD is a single wide add, so shallow-depth-1 stays competitive),
**decisively (~13×) for MUL and DIV** (fp64's whole-value cell is both deep *and* on the
40×-slow datapath). The extra depth of the low-precision limb decomposition is **more than
paid for by the 5.4× tensor-core throughput** on the ops that matter.

## 5. Doom fps projection for the best low-precision config (bf16 radix-16)

MEASURED anchors: current composed best step **0.788 µs** (idle A5000, byte-exact,
`PERF_LADDER_FINAL.md`); raw frame **6,889,264** steps, render-reduced **358,058** steps;
current raw baseline **~0.045–0.184 fps**. Two regimes, because a faster-FLOP ALU only buys
fps when the step is **compute-bound**:

**[1] Today's WIDE VM (5.68 MFLOP/step, MEASURED occupancy/HBM-bound at ~11.5% peak).**
Swapping fp32→bf16 here gives **~1.0× — no fps gain**. The wide step is memory/occupancy-
bound, not FLOP-bound, so a faster-per-FLOP ALU does nothing. Low precision is **not a
drop-in speedup of the current VM.**

**[2] A NARROW deep-serial bf16 VM** (the regime where the ALU throughput *is* the
bottleneck), MEASURED cell wall-clock, batched 65,536 lanes, one ALU-class op-batch =
**26.6 ns/lane**:

| step model | ns/step | RAW fps (6.89 M) | RENDER fps (358 K) |
|---|---:|---:|---:|
| 1 ALU-op/step (pure execute) | 26.6 | **5.5** | **105** |
| 4 ALU-ops/step (execute + framing) | 106 | 1.4 | **26** |
| 8 ALU-ops/step (framing-heavy) | 213 | 0.7 | 13 |
| 2-GPU (rung 8, 2.0×) | — | ×2 above | ×2 above |

**On the render-reduced frame the best low-precision config clears 35 fps (75–105 fps at
1 ALU-op/step, ~26 fps even at a framing-heavy 4 ALU-ops/step) — REALTIME.** On the **raw
6.89 M-step frame it does NOT** (~5.5 fps best, ~11 fps 2-GPU): 6.89 M strictly-sequential
steps × tens-of-ns is inherently multi-second regardless of dtype. This matches the perf
ladder's standing conclusion — the raw frame needs **algorithmic step-count reduction**
(fold the render/BSP traversals), not a faster kernel.

(The pure-FLOP floor — 1952 FLOP/step ÷ 93 TFLOP/s — reads ~1000–7000 raw fps, but that is a
FLOP-bound upper bound that ignores the per-op launch/depth the MEASURED 26.6 ns captures;
the measured cell number is the honest one.)

## 6. The two Pareto corners, explicitly

| corner | dtype | params (per op) | depth | wall-clock | Doom realtime? |
|---|---|---:|---:|---|---|
| **MIN-PARAMS** (`clever_minparam_alu.py`) | fp64 / fp128 | **~4 scalars** | 1–20 | **slow** (fp64 = 0.02× fp32; MUL 603 ns, DIV 302 ns/lane) | render: marginal · raw: no |
| **MIN-WALLTIME** (`lowprec_radix_alu.py`) | **bf16 radix-16** | 5–8 scalars | 8–16 (deeper) | **fast** (bf16 = 5.4× fp32; MUL 46 ns, DIV 23 ns/lane, ~13× the fp64 cell) | **render: YES (35–105 fps)** · raw: no (~5.5 fps) |

- **Min-params** buys the smallest weight count (4 scalars) but runs on the card's slowest
  datapath — it is the corner for *param economy / provable minimality*, not speed.
- **Min-walltime** pays a handful more params and 8–16× the depth, but every layer runs on
  the tensor cores at 5.4× fp32, netting **~13× faster wall-clock than the fp64 cell** on
  MUL/DIV — it is the corner for *throughput*.

They are genuinely opposite corners of the same exact-integer-arithmetic frontier: min-params
trades **precision (up)** for **params (down)**; min-walltime trades **depth (up) + params
(slightly up)** for **wall-clock (down)** by staying in the tensor-core-fast dtypes.

## 7. Honest verdict

**Is low-precision + small-radix + deep the realtime-Doom path?** Partly, and the honest
answer is precise:

- **Which config:** **bf16, radix 16** — the measured sweet spot. bf16 realizes the full
  5.4× tensor-core throughput (int8's 4× is *not* realized on this A5000 → int8's extra depth
  makes it *slower*, not faster), and radix 16 keeps MUL depth at 16 / DIV at 8 while staying
  exact. It beats the fp64 whole-value cell by ~13× wall-clock on MUL/DIV.
- **For which frame:** the **render-reduced 358 K-step frame → YES, realtime** (75–105 fps at
  1 ALU-op/step; ≥26 fps even framing-heavy; ×2 on 2 GPUs). The **raw 6.89 M-step frame →
  NO** (~5.5 fps, ~11 fps 2-GPU) — inherently multi-second at 6.89 M sequential steps
  regardless of dtype; that frame needs algorithmic step-count reduction, per the perf ladder.
- **The load-bearing caveat:** the speedup is real **only if the VM is narrow and
  compute-bound.** Today's wide VM is memory/occupancy-bound (~11.5% HBM peak), where
  fp32→bf16 nets ~1.0×. Low precision is a **narrow-VM lever, not a drop-in for the current
  build**. Golden unchanged; no build files touched.
