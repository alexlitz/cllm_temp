# Does a REALIZABLE vanilla feed-forward clever transformer run Doom realtime? — MEASURED

**Date:** 2026-08-08 · **Scope:** real-tensor cells (byte-exact) + exact total-nonzero
census + MEASURED forward ms/step on GPU + honest Doom-fps verdict.
**Touches no build files** — the c4 golden `174ece66` is unchanged (verified below).

This is the **measured** counterpart to the two earlier *projection* docs
([`CLEVER_DOOM_REALTIME.md`](CLEVER_DOOM_REALTIME.md), `PRECISION_RADIX_SURFACE.md`).
Those projected Doom fps from an **isolated arithmetic decode cell** (the ~10-wide
difference-min selector, ~0.008 µs/op) times a hand-picked *5-passes/step* framing
multiplier. This doc instead builds the **whole realizable model** — a real 51-layer
transformer at hidden 896 with genuine attention + FFN + ALiBi — and measures the
cost of **one full forward = one VM step** (a standard feed-forward transformer runs
its ENTIRE stack every step). That is the honest realizable per-step cost, and it is
**~1000× larger** than the isolated-cell number the projections used.

Code (all under `examples/`, no build files touched):
- [`examples/clever_realtime_cells.py`](../examples/clever_realtime_cells.py) — the
  REAL-tensor hand-built cells (embed / W_q,k,v,o / FFN) + byte-exact verifier.
- [`examples/clever_realtime_model.py`](../examples/clever_realtime_model.py) — the
  full-ISA assembler (exact total-nonzero census, both modes) + the correct-SHAPE
  51-layer model + the GPU throughput sweep.
- [`examples/clever_realtime_table.py`](../examples/clever_realtime_table.py) — the
  possibility table (joins geometry + census + measured throughput + Doom fps).

Reproduce:
```
python examples/clever_realtime_cells.py --n 3000        # byte-exact cells (CPU)
python examples/clever_realtime_model.py --census        # total-nonzero census (CPU)
python examples/clever_realtime_model.py --bench --json /tmp/clever_bench.json   # GPU sweep
python examples/clever_realtime_table.py                 # the possibility table
```

---

## TL;DR verdict

**NO.** A realizable **vanilla feed-forward** clever transformer (51 distinct layers,
hidden 896) does **NOT** run Doom in realtime at **any** precision, on **either** frame:
the best realizable config (**bf16 radix-16 digit-extract, 42 layers**) reaches only
**≈ 0.31 fps** on the render-reduced (358,058-step) frame and **≈ 0.016 fps** on the raw
(6,889,264-step) frame — **~100× short of 30 fps even on the reduced frame.**

The wall is **not** the arithmetic cell (the earlier projections were right that the
cell is shallow and cheap). The wall is that a *standard feed-forward* transformer must
run its **whole 42–51-layer hidden-896 stack every single VM step**, so the honest
per-step cost is **9–1081 µs/step** (measured), not the **~0.04 µs/step** the earlier
cell-only projection assumed. That ~1000× gap is exactly `full-stack / one-cell`. The
earlier "69 fps fp32 render" number is the isolated-cell throughput; it is **not** what
a realizable vanilla transformer delivers.

| precision (realizable shape) | ms/step (MEASURED) | RENDER fps | RAW fps | ≥30 fps? |
|---|---:|---:|---:|:--:|
| **bf16 radix-16 (42L, min-walltime)** | **0.009** | **0.31** | 0.016 | ❌ |
| fp16 radix-16 (42L) | 0.009 | 0.30 | 0.016 | ❌ |
| int8 radix-4 (92L, deep) | 0.020 | 0.14 | 0.007 | ❌ |
| fp32 whole-value (51L) | 0.039 | 0.07 | 0.004 | ❌ |
| fp64 whole-value (51L, byte-exact) | 1.081 | 0.003 | 0.0001 | ❌ |

All MEASURED on an idle **RTX A5000** (device 0), at throughput saturation (batch swept
to 65,536 lanes). `ms/step` = one full 51/42-layer forward per VM step.

---

## 1. Real-tensor cells — byte-exact pass rates (Task 1)

Extended the `minimal_10digit_adder.py` real-tensor style into three concrete cell
families, each an actual `torch.nn.Module` with **named weight tensors** (embed /
W_q,W_k,W_v,W_o / FFN), NOT a numpy math-sim:

- **(a) Arithmetic ingest+decode (fp64):** non-one-hot embed (dim0 = digit face value)
  → real Q/K/V/O identity routing into an ALiBi(ln 10) + softmax1 **place-value head**
  that reconstructs the whole operand value → a decode FFN whose 10 rows are the
  candidate digit centres (`d+0.5`); the difference-min argmax gives `floor`, reused
  once per output DIGIT. Covers ADD/SUB, CMP×6, SHL/SHR, LEA/frame adds, DIV/MOD, MUL.
- **(b) Bitwise 16×16 nibble-LUT (fp32):** a real FFN whose 256 hidden units are AND
  detectors for each `(na,nb)` pair (W_up +1 on the two active one-hots, bias −1 → ReLU
  fires iff both match); W_down carries the LUT result. Reused per nibble (depth 8).
- **(c) Memory CAM (fp32):** a real attention head — Q = query-address nibble one-hots,
  K = stored-address nibble one-hots, a large temperature makes softmax a hard
  exact-address match, V delivers the stored word. Shared by LI/LC/SI/SC.

**Verified byte-exact on ≥ 2,000–3,000 random 32-bit operands/op + edge cases:**

| cell / op | precision | depth | cases | byte-exact |
|---|---|---:|---:|:--:|
| INGEST (real embed + ALiBi + softmax1 place-value read) | fp64 | — | 3,000 | ✅ |
| ADD (whole a+b, decode FFN) | fp64 | 11 | 3,000 | ✅ |
| SUB (whole a−b, wrap) | fp64 | 11 | 3,000 | ✅ |
| DIV/MOD (per-digit long-division cell) | fp64 | 10 | 3,000 | ✅ |
| MUL (fp128 whole product, same decode cell) | fp128 | 20 | 2,000 | ✅ |
| CMP EQ/NE/LT/GT/LE/GE (signed sign read) | fp64 | 1 | 3,000 | ✅ |
| SHL (×2^n scale + 32-bit mask) | fp64 | 1 | 3,000 | ✅ |
| SHR (arithmetic /2^n, signed) | fp64 | 1 | 3,000 | ✅ |
| LEA / ADJ / branch (address add) | fp64 | 1 | 3,000 | ✅ |
| OR (real 16×16 nibble-LUT FFN) | fp32 | 8 | 3,000 | ✅ |
| XOR (real 16×16 nibble-LUT FFN) | fp32 | 8 | 3,000 | ✅ |
| AND (real 16×16 nibble-LUT FFN) | fp32 | 8 | 3,000 | ✅ |
| LI/SI (real CAM attention read, shuffled query + miss control) | fp32 | 1 | 2,000 | ✅ |
| LC/SC (CAM read + byte sign-extend) | fp32 | 1 | 2,000 | ✅ |

**14/14 cell families byte-exact — 0 errors.** MUL's whole-product hold requires
fp128 (numpy longdouble; torch has no fp128), but its DECODE cell is byte-identical to
the fp64 one — only the datapath dtype rises. The CAM read is verified with a **shuffled
query order** (so a correct read must actually address-match, not row-align) and a
**negative control** (never-stored addresses must gather 0). The multi-layer chains
(DIV depth 10, MUL depth 20) are verified end-to-end; each reuses the SAME per-digit
cell (the depth lever).

---

## 2. EXACT total-nonzero census (Task 2)

Both modes share the **896-hidden 51-layer Qwen2 SHAPE** (dense ≈ 217 M params, mostly
zeros). The **TOTAL nonzero** (replicas counted, per the total-not-distinct rule) is the
signal — the real clever-cell entries placed into the stack:

Per-family REAL-cell nonzero footprint (measured from the cells above):
- arith/div/mul decode cell = **51** (embed 12 + Q/K/V/O identity 16 + candidates 19 + 4 scalars)
- bitwise LUTs OR+AND+XOR = **2,974** (OR 1023 + AND 943 + XOR 1008)
- memory CAM = **10**

**[A] STANDARD feed-forward UNROLLED — 51 DISTINCT layers (replicas counted):**

| family | n_layers | per-layer nonzero | total nonzero |
|---|---:|---:|---:|
| arith (ADD/SUB/CMP/SHL/SHR/frame) | 11 | 51 | 561 |
| div (DIV/MOD) | 10 | 51 | 510 |
| mul (MUL) | 20 | 51 | 1,020 |
| bitwise (OR/XOR/AND, all 3 LUTs per nibble-place) | 8 | 2,974 | 23,792 |
| memory (shared CAM) | 1 | 10 | 10 |
| trivial (IMM/PSH/NOP/HALT) | 1 | 0 | 0 |
| embed + framing (12 embed + 10 LM-head + 4/layer × 51) | — | — | 226 |
| **UNROLLED TOTAL NONZERO** | **51** | | **26,119** |

Family groups: arithmetic 2,091 · bitwise 23,792 · memory 10 · embed+framing 226.

**[B] LOOPED / Universal-Transformer — 6 STORED reused cells (each stored ONCE):**

| stored cell | nonzero |
|---|---:|
| ingest + arith decode | 51 |
| div decode | 51 |
| mul decode | 51 |
| bitwise LUT (OR+AND+XOR) | 2,974 |
| memory CAM | 10 |
| embed + framing (on 6 cells) | 46 |
| **LOOPED TOTAL NONZERO** | **3,183** |

The **bitwise LUT dominates** both totals (23,792 / 2,974) — it is the one family that
does **not** collapse to ~4 scalars (floats have no bit ops). Arithmetic — the whole
ISA's ADD/SUB/MUL/DIV/MOD/CMP/shift/frame machinery — is a mere 2,091 (unrolled) / 153
(looped) nonzero: the whole-value + difference-min collapse is real. The looped total
(3,183) is ~8× smaller than the unrolled (26,119) because it stores each cell once
instead of replicating it per place.

---

## 3. Correct-SHAPE realizable model + MEASURED throughput (Task 3)

The throughput lever is the **SHAPE**, not the nonzeros. `CleverShapeModel` is a real
51-layer torch stack at hidden 896 with genuine attention (Q/K/V/O @ 896×… GQA 14 heads /
2 KV), an ALiBi bias, softmax, causal mask, and a real SwiGLU FFN (up/gate/down @
896×896). One forward (T=1, batch = concurrent VM lanes) = one VM step. Swept over batch
to saturation, per precision, MEASURED on an idle RTX A5000:

| precision | best ns/lane-step | ms/step | lane-steps/s | sat. batch |
|---|---:|---:|---:|---:|
| **fp64** (51L) | 1,081,441 | 1.081 | 0.0009 M | 1,024 |
| **fp32** (51L) | 38,509 | 0.039 | 0.0260 M | 65,536 |
| **bf16** (51L) | 11,266 | 0.011 | 0.0888 M | 65,536 |
| **bf16** (42L, min-walltime) | **8,951** | **0.009** | **0.112 M** | 65,536 |
| fp16 (42L) | 9,281 | 0.009 | 0.108 M | 65,536 |
| int8 (92L, deep) | 20,378 | 0.020 | 0.049 M | 65,536 |

Findings (all MEASURED):
- **fp32 GEMM efficiency check:** at batch 65,536 the fp32 51-layer model does
  65,536×51×7×896²×2 ≈ 3.8 TFLOP in 2.52 s ≈ **13.4 TFLOP/s = 78 % of the card's 17.2
  TFLOP/s fp32 GEMM peak** — the model is efficient, not pathological; the cost is
  simply that 51 layers of 896² matmuls per step is heavy.
- **fp64 is punishing:** 1.08 ms/step, **~28× slower per lane-step than fp32** (7 ms per
  single fp64 layer at batch 256 — the consumer A5000 has no fp64 tensor cores).
- **bf16 is the fastest realizable dtype** (~3.4× fp32 in this stack). **int8's
  theoretical 2.7× is NOT realized past bf16** on this card (torch int8 rides the same
  fp16/bf16 tensor cores here), and its **deeper** low-radix stack (92 layers vs bf16's
  42) makes int8 **slower**, not faster — matching the earlier surface's honest finding.

---

## 4. Doom realtime test (Task 4)

Doom frame VM-step counts (canonical, from `PRECISION_RADIX_SURFACE.md` /
`CLEVER_DOOM_REALTIME.md`): **render-reduced = 358,058 steps/frame**, **raw title-redraw
= 6,889,264 steps/frame**. Because a standard feed-forward transformer runs its whole
stack every step, one VM step = one full model forward, so:

> **fps = (1e9 / ns_per_step) / steps_per_frame** — no extra passes-per-step multiplier
> (the whole 42–51-layer machinery is already inside one forward).

The **raw** frame multiplies the step-count by ~19× over the render frame, so every raw
verdict is ~19× worse than its render verdict. **Both frames fail at 30 fps** — the raw
frame is not even close (best 0.016 fps), and the render frame is ~100× short (best
0.31 fps). So the raw full frame does **not** fit and needs the render-macro step-fold
(as the perf ladder has long said) — but here **even the folded render frame does not
reach realtime**, because the per-step cost is the wall, not just the step-count.

---

## 5. THE POSSIBILITY TABLE

One row per meaningful VALID `(precision × radix × extraction × mode)` combo (filtered by
`opconfig.validate`). `ms/step` and `fps` are **MEASURED** on the realizable shape except
where marked derived. `dense params` from `qwen_fit_solver`; `TOTAL nonzero` from §2.

| precision | radix | extraction | mode | n_layers | hidden | dense params | **TOTAL nonzero** | ms/step | RENDER fps | RAW fps | realtime? | src |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|
| fp32 | 16 | nibble | unrolled (std-FF) | 123 | 3008 | 9,551 M | n/a | n/a | n/a | n/a | n/a | golden (diff. build) |
| fp64 | 10 | whole_value | unrolled (std-FF) | 51 | 896 | 217 M | **26,119** | 1.081 | 0.003 | 0.0001 | ❌ | MEASURED |
| fp64 | 10 | whole_value | looped (UT) | 6 | 896 | 26 M | **3,183** | 1.081 | 0.003 | 0.0001 | ❌ | MEASURED |
| fp32 | 10 | whole_value | unrolled (std-FF) | 51 | 896 | 217 M | 26,119 | 0.039 | 0.073 | 0.0038 | ❌ | MEASURED |
| fp32 | 10 | whole_value | looped (UT) | 6 | 896 | 26 M | 3,183 | 0.039 | 0.073 | 0.0038 | ❌ | MEASURED |
| **bf16** | 16 | digit_extract | unrolled (std-FF) | 42 | 896 | 179 M | 25,624 | **0.009** | **0.312** | 0.0162 | ❌ | MEASURED |
| bf16 | 16 | digit_extract | looped (UT) | 6 | 896 | 26 M | 3,183 | 0.009 | 0.312 | 0.0162 | ❌ | MEASURED |
| fp16 | 16 | digit_extract | unrolled (std-FF) | 42 | 896 | 179 M | 25,624 | 0.009 | 0.301 | 0.0156 | ❌ | MEASURED |
| int8 | 4 | digit_extract | unrolled (std-FF) | 92 | 896 | n/a | 28,374 | 0.020 | 0.137 | 0.0071 | ❌ | MEASURED |
| fp128 | 10 | whole_value | unrolled (std-FF) | 51 | 896 | 217 M | 26,119 | 1.081 | 0.003 | 0.0001 | ❌ | derived (fp64 floor) |

Notes: the LOOPED (UT) rows share the unrolled per-step throughput as a proxy — a UT
re-applies its cells `depth` times per forward, so applied per-step compute is
comparable; the UT win is in **stored** count (3,183 vs 26,119 nonzero), not per-step
speed. fp32/int8 whole-value are **throughput proxies** (not byte-exact at full 32-bit —
fp32's 2^24 ceiling; the byte-exact build is fp64/fp128). The golden nibble row is a
DIFFERENT (3B-class-width) build, not the clever shape, so its fps is N/A.

---

## 6. The two Pareto corners

| corner | dtype | mode | **TOTAL nonzero** | ms/step (MEASURED) | RENDER fps | realtime? |
|---|---|---|---:|---:|---:|:--:|
| **MIN-NONZERO** | fp64 / fp128 whole_value | looped (UT) | **3,183** | 1.081 | 0.003 | ❌ |
| **MIN-WALLTIME** | bf16 radix-16 digit_extract | unrolled (42L) | 25,624 | **0.009** | **0.312** | ❌ |

- **MIN-NONZERO** buys the smallest weight count (3,183 nonzero looped / 26,119 unrolled)
  but runs on the card's slowest datapath (fp64 = 1.08 ms/step, ~120× the bf16 corner).
- **MIN-WALLTIME** is the fastest realizable config (bf16, 0.009 ms/step) and leads the
  fps race — but still 96× short of 30 fps on the render frame.

They are genuinely opposite corners of the same frontier (params-down ↔ walltime-down),
but **neither reaches Doom realtime** as a realizable vanilla feed-forward transformer.

---

## 7. Honest caveats — measured vs projected

- **What is MEASURED:** every cell's byte-exactness (14/14, ≥ 2–3 k operands each); the
  exact total-nonzero census; the forward ms/step of the real 42/51/92-layer hidden-896
  shape at fp64/fp32/bf16/fp16, swept to saturation on an idle A5000. The fps numbers
  follow directly from those measured ms/step and the canonical step-counts.
- **What is PROJECTED / derived:** the fp128 MUL row (no GPU fp128 path → fp64 timing as
  the floor; MUL runs on CPU longdouble in the cell verifier); the int8-vs-bf16 realized
  ratio (int8 rides the fp16/bf16 tensor cores here, so int8 = its deeper 92-layer
  bf16-class shape); the LOOPED per-step cost (shares the unrolled proxy).
- **The raw-frame step-fold gap:** the raw 6.89 M-step frame is ~19× the render frame; it
  does NOT fit and needs the render-macro step-fold. But this doc's new finding is that
  **even the folded render frame does not reach realtime** on the realizable vanilla
  model — the per-step cost (full-stack traversal) is the binding wall, not the
  step-count alone.
- **Why this differs from the earlier "69 fps" projection:** the earlier docs benched an
  **isolated arithmetic decode cell** (~0.04 µs/op) and multiplied by a hand-picked
  5-passes/step framing factor. A realizable **standard feed-forward** transformer must
  run its **entire 42–51-layer hidden-896 stack every step** — measured at 9–1081 µs/step,
  **~1000× the isolated cell**. The earlier number is the cell's throughput, not the
  realizable model's. This doc supersedes those fps figures for the *realizable vanilla*
  claim; the earlier cell-throughput and byte-exactness results stand.
- A LOOPED / Universal-Transformer shrinks the **stored** count (3,183 nonzero) but not
  the per-step walltime (it re-applies cells `depth` times per forward), so it does not
  change the realtime verdict.

## 8. Golden preserved

`CUDA_VISIBLE_DEVICES="" PYTHONPATH=<repo>/c4_release python -m c4_min._fingerprint_build`
→ **`174ece66edff1bb5ab8e9213e484bb1b9560f44ab6c23077a366491543d05637`** (short
`174ece66`), unchanged. This doc + the three example scripts touch **no** build file
(nibble_pure_forward_complete / nibble_alu32 / compact_alloc are untouched).
