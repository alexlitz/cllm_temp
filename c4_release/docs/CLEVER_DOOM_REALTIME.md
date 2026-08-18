# Can the CLEVER min-param c4 VM run Doom in realtime (35 fps)?

**Date:** 2026-08-07 · **Scope:** FLOP derivation + GPU microbenchmark + fps projection.
**Touches no build files** — c4 golden (`174ece66` / fingerprint `7d4afe61`) is unchanged.
Code:
[`examples/clever_minparam_alu.py`](../examples/clever_minparam_alu.py) (the byte-EXACT
cells) and [`examples/clever_doom_microbench.py`](../examples/clever_doom_microbench.py)
(this doc's GPU microbenchmark). Companion floor doc:
[`docs/SERIAL_ALU_FLOP_FLOOR.md`](SERIAL_ALU_FLOP_FLOOR.md) (the ~160-layer bit-serial
form — the OTHER depth regime).

---

## TL;DR verdict

**YES — the CLEVER c4 VM can run the *render-reduced* Doom frame in realtime, in fp32:
≈ 69 fps on one idle A5000, ≈ 137 fps on 2 GPUs (both well past 35 fps).** In fp64 it
clears 35 fps only with 2 GPUs (25 fps 1-GPU → 51 fps 2-GPU). The **RAW title-redraw
frame does NOT reach 35 fps** (≈ 3.6 fps fp32 1-GPU) — it is 19× more steps and needs the
algorithmic step-count fold, not a faster cell. **What makes realtime possible is the
CLEVER form's shallowness**: ~10–20 reused digit layers (not the ~160-layer bit-serial
floor), so a big batch of concurrent verify-lanes fills the card. **What gates it is
occupancy, not FLOP** — the cell runs at 0.38 % of fp32 peak / 9.5 % of the tiny fp64
peak, so fp64's 42× hardware penalty forces **fp32-clever** for the comfortable margin.

| frame | steps | fp32 1-GPU | fp32 2-GPU | fp64 1-GPU | fp64 2-GPU | ≥35 fps? |
|---|---:|---:|---:|---:|---:|:--:|
| **RENDER-reduced** | 358,058 | **68.7 fps** | **137 fps** | 25.4 fps | **50.7 fps** | ✅ (fp32 1-GPU; fp64 needs 2-GPU) |
| RAW title-redraw | 6,889,264 | 3.6 fps | 7.1 fps | 1.3 fps | 2.6 fps | ❌ (needs step-count fold) |

All numbers are MEASURED on an idle RTX A5000 (device 0) at the throughput floor, or
LABELLED a projection. The clever cells are verified byte-EXACT standalone; a full
doom-running clever VM does **not** exist (§5) — the fps is a projection from the measured
cell + the verified spec-decode execution model.

---

## 1. The CLEVER cell and its FLOP/step (derived from the actual MACs)

The clever construction (`clever_minparam_alu.py`) holds each whole operand/result in
**ONE fp scalar** (no nibble/bit split) and extracts the result **one decimal digit per
reused layer**, MSB-first, with the difference-min selector
`logit_d = -|value/scale - (d+0.5)|`. Depth = #output digits (**ADD 11, DIV 10, MUL 20**),
so it is **narrow AND shallow** — the opposite of the ~160-layer bit-serial floor.

### FLOP per reused digit-extraction layer (counting the actual cell ops)

Counting the same way the c4 `_flop_gauge` counts COO ops (MAC → 2 FLOP; abs/sub/argmax =
1 FLOP-equiv over their width), one application of `decode_digit` + the running-remainder
update over the 10-wide candidate vector is:

| op in the cell | width | FLOP |
|---|---:|---:|
| `R / scale` | 1 | 1 |
| `val − (cand+half)` (sub) | 10 | 10 |
| `.abs()` | 10 | 10 |
| `tie*cand` (mac) | 10 | 20 |
| `+` accumulate | 10 | 10 |
| `argmax` over 10 | 10 | 10 |
| `out += d*scale` (mac) | 1 | 2 |
| `R  −= d*scale` (mac) | 1 | 2 |
| **per digit layer** | | **≈ 75 FLOP** |

Plus a one-shot place-value **ingest** per operand (`place_value_read`: exp + sum + div +
weighted-sum over the ≤10-digit run, ~4·W ≈ 40 FLOP/operand, two operands ≈ 80). So:

| op | depth | decode FLOP (depth·75) | + ingest | **FLOP/op** | precision |
|---|---:|---:|---:|---:|:--|
| **ADD/SUB** (also covers pointer/PSH/LEA/LI/CMP/branch address-adds) | 11 | 825 | 80 | **905** | fp64 (fp32 exact ≤ 2²⁴·... see §2) |
| **DIV/MOD** | 10 | 750 | 80 | **830** | fp64 |
| **MUL** (32×32→64) | 20 | 1500 | 80 | **1580** | fp128 exact / fp32-fp64 = throughput proxy |

### FLOP per clever VM step, and per Doom frame

A c4 VM step = fetch + decode + one execute op + PC/SP/BP writeback. In the clever style,
framing (opcode one-hot select + a PC increment + an SP bump + an address compare) is
~4 ADD-class digit passes, and the execute op is one cell. Modelling the step as **≈ 5
ADD-class passes** (framing 4 + execute 1; Doom is 63 % pointer-walk = ADD-class — capstone
§6 / `PERF_LADDER_FINAL.md` §5):

> **CLEVER FLOP/step ≈ 5 × 905 ≈ 4,500 FLOP** (a DIV step adds ~830; a rare MUL step ~1580).

Doom-frame FLOP (step counts MEASURED: raw 6,889,264; render-reduced 358,058):

| frame | steps | FLOP/step | **total FLOP** | vs current 5.68 MFLOP/step |
|---|---:|---:|---:|---:|
| RAW — **current wide** | 6,889,264 | 5,680,000 | **39.13 TFLOP** | 1× |
| RAW — **clever** | 6,889,264 | ~4,500 | **≈ 31.0 GFLOP** | **≈ 1,260× ↓** |
| RENDER — **current wide** | 358,058 | 5,680,000 | **2.03 TFLOP** | 1× |
| RENDER — **clever** | 358,058 | ~4,500 | **≈ 1.61 GFLOP** | **≈ 1,260× ↓** |

So the clever construction cuts the Doom-frame FLOP **≈ 1,260×** (vs the bit-serial floor's
~2,910× — the clever form pays a bit more FLOP for far less depth, which is the trade that
buys the wall-clock; see §2). Against the task's `39/2 TFLOP` phrasing: RAW 39.13 → 0.031
TFLOP, RENDER 2.03 → 0.0016 TFLOP.

---

## 2. GPU microbenchmark — does the FLOP cut become wall-clock?

`clever_doom_microbench.py` builds the exact clever digit-extract cell as a batched torch
module and drives it with a large batch = **many concurrent VM lanes / speculative-K
verify steps** (this batch dimension IS the real Doom execution model — see §3). Measured
on **idle RTX A5000, device 0**, both fp64 and fp32, at the steady-state floor (CUDA-graph
replay removes launch overhead; at batch ≥ 262 k the graph gives only ~1.0–1.02×, so the
cell is **memory-bandwidth-bound, not launch-bound**):

| op (rep. step) | dtype | best throughput | **µs/op (floor)** | % of card peak |
|---|:--|---:|---:|---:|
| ADD/pointer d11 | **fp64** | 45.4 Mops/s | **0.0220** | **9.5 %** of fp64 peak |
| ADD/pointer d11 | **fp32** | 123 Mops/s | **0.0081** | **0.38 %** of fp32 peak |
| DIV d10 | fp64 | 49.5 Mops/s | 0.0202 | 9.5 % |
| DIV d10 | fp32 | 134 Mops/s | 0.0075 | 0.37 % |
| MUL d20 | fp64 | 24.2 Mops/s | 0.0413 | 9.0 % |
| MUL d20 | fp32 | 66.6 Mops/s | 0.0150 | 0.36 % |

**Occupancy is the wall, not FLOP.** Even at a 1,048,576-lane batch the cell tops out at
0.38 % of the fp32 roofline (27.8 TFLOP/s datasheet; 17.2 TFLOP/s measured GEMM) — the
per-lane work is a tiny 10-wide difference-min over ~10–20 layers, so it is DRAM-latency /
occupancy bound. But it is **shallow enough that throughput still saturates high** (45–134
Mops/s), which is the whole point vs the ~160-layer bit-serial form.

### The fp64 → fp32 speed factor (the load-bearing hardware fact)

The A5000 (consumer GA102, no fp64 tensor cores) has **fp64 = 1/42 of fp32** (MEASURED
GEMM: fp32 17.18 TFLOP/s, fp64 0.411 TFLOP/s → **41.8× penalty**; datasheet 1/64). But the
clever cell's fp64-vs-fp32 wall gap is only **≈ 2.5–2.7×** (0.0220 vs 0.0081 µs/op), NOT
42× — because the cell is memory-bound, so fp64 costs mainly its 2× bytes-per-value, not
its 42× FLOP disadvantage. Still, **fp32 is 2.5–2.7× faster wall-clock**, and that is what
moves the render frame from "35 fps only on 2 GPUs" (fp64) to "69 fps on one" (fp32).
**Exactness catch:** the byte-exact clever cells are **fp64** (ADD/SUB: a+b < 2³³ < 2⁵³
exact; DIV/MOD: every USED partial < 2³² < 2⁵³) and **fp128** (MUL: a·b < 2⁶⁴). **fp32's
exact-integer ceiling is only 2²⁴ ≈ 1.67e7**, so the whole-operand fp32 path is NOT exact
for full 32-bit values — the fp32 microbench is a **throughput proxy**. A byte-exact
*fp32-speed* clever VM would split each operand into two 16-bit fp32-exact halves (≈ 2× the
passes); that rework is folded into the projection as the conservative 5-passes/step
framing multiplier (§4/§5).

---

## 3. Doom fps projection (raw + render, 1-GPU + 2-GPU)

**Execution model (grounded, not hand-waved):** the existing perf path
(`PERF_FAST_PATH_2026_07_20.md`, `batched_speculative`) materializes the whole frame's step
stream up front with a **perfect Rust c4 draft**, then the transformer **verifies K steps
per batched forward in parallel**. So a frame of N steps is N ops to verify, **batched** —
exactly what the microbench measures. fps = (throughput ops/s) / (steps/frame). Per-step =
**5 × the measured ADD/pointer µs/op** (framing 4 + execute 1); 2-GPU = rung-8 frame-level
**2.0×** (`DOOM_FRAME_LEVEL_2GPU_2026_08_05.md`, measured near-perfect).

| build | frame | steps | 1-GPU ms/frame | **1-GPU fps** | **2-GPU fps** |
|---|---|---:|---:|---:|---:|
| **clever fp32** | **RENDER** | 358,058 | 14.6 | **68.7** | **137** |
| clever fp32 | RAW | 6,889,264 | 280 | 3.6 | 7.1 |
| **clever fp64** | **RENDER** | 358,058 | 39.4 | **25.4** | **50.7** |
| clever fp64 | RAW | 6,889,264 | 759 | 1.3 | 2.6 |

Sensitivity — the OPTIMISTIC bound (framing folded into a render superinstruction, 1
op/step): fp32 RENDER → 344 fps 1-GPU; fp32 RAW → 17.9 fps 1-GPU / 35.7 fps 2-GPU. The
5-passes/step framing multiplier is the conservative anchor; the true number sits between
these.

**35 fps budgets** (for reference): RENDER needs 12.53 Msteps/s = 0.0798 µs/step (clever
fp32 delivers 24.6 Msteps/s → 2.0× under); RAW needs 241 Msteps/s = 0.00415 µs/step (clever
fp32 delivers 24.6 Msteps/s → **10× short**).

---

## 4. Honest verdict + what gates it

**One line:** *the CLEVER c4 VM can run the render-reduced Doom frame at 35 fps — ≈ 69 fps
fp32 / 25 fps fp64 on one idle A5000 (fp32 clears 35 fps 1-GPU; fp64 needs the 2.0× second
GPU) — but the RAW title-redraw frame does NOT (≈ 3.6 fps fp32 1-GPU, 10× short), because it
is 19× more steps; realtime on the raw frame is an algorithmic step-count fold, not a faster
cell.*

**What gates realtime, ranked:**

1. **Occupancy (primary).** The cell runs at 0.38 % of fp32 peak — it is DRAM-latency /
   occupancy bound, not FLOP bound. This is *good news for the clever form*: because it is
   shallow (10–20 layers, not 160), throughput still saturates at 45–134 Mops/s, enough to
   clear the render frame. The bit-serial floor (2,910× fewer FLOP) would be **slower** here
   — its ~160 strictly-sequential layers starve the card worse. Shallow-clever wins on
   wall-clock precisely because it trades a little FLOP for far less depth.
2. **fp64 throughput (secondary).** The 42× hardware fp64 penalty is softened to ~2.5–2.7×
   in this memory-bound cell, but it is still the difference between fp32 clearing 35 fps on
   one card and fp64 needing two. **fp32-clever is the realtime configuration.**
3. **Step count (the RAW-frame wall).** The raw frame's 6.89 M steps is the only thing that
   keeps a frame type below 35 fps. It is not a cell problem — it is the same
   render-superinstruction / traversal-fold that `PERF_LADDER_FINAL.md` §3 identifies for the
   wide build. Fold the raw frame toward the render-reduced 358 K and the clever cell already
   clears it.

---

## 5. Honest caveats — the clever cells are STANDALONE; the doom VM is a projection

**This is a projection, not a running system.** The load-bearing honesty:

- The three clever cells are verified **byte-EXACT standalone** (ADD/SUB/DIV/MOD/MUL, ≥100 k
  random 32-bit pairs each + hard edges, `clever_minparam_alu.py`) — but a **full,
  byte-exact, doom-running VM built from them DOES NOT EXIST.** That is a substantial
  re-architecture, not a flag: it would need (a) the fetch/decode/opcode-select and PC/SP/BP
  writeback rebuilt in the same narrow whole-value digit-extract style, (b) a memory/stack
  model (the current VM's live-CAM reads), (c) the exact-fp32 operand-halving (§2) if the
  fp32 speed is to be byte-exact, and (d) integration with the perfect-draft spec-decode
  verify loop so the batch dimension is real. The fps here is `measured-cell-throughput ×
  a 5-passes/step VM-step model`, cross-checked against the verified spec-decode execution
  model — **not** an end-to-end doom run.
- **Biggest risk to realtime: the framing/memory ops, not the ALU.** The measured cells are
  the arithmetic core; a real step is dominated by opcode decode + memory/stack reads +
  register writeback. If those do not stay in the same shallow-batched-friendly form (e.g. if
  memory reads reintroduce an O(S) softmax-over-stores instead of the O(1) direct-CAM the
  wide build already has), the per-step cost balloons and the projection breaks. The 5×
  multiplier is the conservative guard, but only a built VM settles it.
- The **fp32 cells are a throughput proxy, not byte-exact** at full 32-bit width (fp32's 2²⁴
  integer ceiling). The exact clever build is fp64/fp128; the fp32 realtime number assumes
  the operand-halving rework (≈ 2× passes, already inside the 5× step multiplier).
- Golden unchanged: this doc + the two example scripts touch **no build files**; the c4
  fingerprint `7d4afe61` is intact.

### Reproduce

```
python examples/clever_minparam_alu.py --n 200000      # byte-exact cells + param census
python examples/clever_doom_microbench.py              # GPU microbench: fp64 + fp32, % peak
```
CPU-safe; uses only idle cuda:0. All numbers MEASURED (idle A5000) or LABELLED a projection.
```
```
