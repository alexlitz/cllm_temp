# SHALLOW × K-forwards (stock-vanilla-fitting) vs DEEP × 1-forward — realtime measure

**Date:** 2026-08-08 · **Scope:** the §5 "forwards-per-step" depth lever, MEASURED.
**Touches no build files** — the c4 golden (`174ece66`) is unchanged (re-verified below).
Companion to [`docs/BLOG_NOTE_CLEVER_MINPARAM_VM.md`](BLOG_NOTE_CLEVER_MINPARAM_VM.md) §5
(the three depth modes) and [`docs/CLEVER_DOOM_REALTIME.md`](CLEVER_DOOM_REALTIME.md) (the
cheap-cell fps projection). Code: [`examples/clever_forwards_realtime.py`](../examples/clever_forwards_realtime.py).

---

## The question

BLOG_NOTE §5: the clever digit-extraction chain wants ~D depth (one digit per layer).
The only depth mode that fits a **stock vanilla 0.5B** (≤ 24 distinct layers, no
weight-tying, no Universal Transformer) is **forwards-per-step**: keep the network
**shallow** (L ≤ 24 layers) and realise the depth in the **autoregressive token loop** —
`F = ceil(D/L)` forwards per VM step, one digit per emitted token, threading the running
remainder through the KV cache between forwards. Compute is conserved (~D layer-applications
per step either way), but §5 warns the realtime cost is **not**: the DEEP form pays **one**
kernel-launch chain + **one** KV read per step; the SHALLOW form pays **F** launch chains +
**F** KV re-reads and "batches worse … is more launch/occupancy-bound."

**Does the stock-vanilla-fitting SHALLOW config still hit realtime, or does the K-forwards
launch/KV overhead break it relative to DEEP?** This doc measures it.

## The two configs (same effective depth D = 48)

Real torch, Qwen2.5-0.5B shapes (**hidden 896, GQA 14 q-heads / 2 kv-heads / head_dim 64,
SwiGLU intermediate 4864, RMSNorm**) — a full vanilla decoder block per digit-layer, plus
the clever 10-candidate difference-min decode head. **SHAPE drives walltime**; the
arithmetic byte-exactness is proven separately in `clever_minparam_alu.py`.

| config | n_layers (L) | F (forwards/step) | L×F (eff depth) | fits stock 24? |
|---|--:|--:|--:|:--|
| **DEEP** (1-forward) | 48 | 1 | 48 | ✗ too deep (48 > 24) |
| **SHALLOW** (K-forwards) | 3 | 16 | 48 | ✓ **VANILLA** (3 ≤ 24) |

The SHALLOW config runs **F = 16 sequential forwards per VM step**, each appending one token
to a threaded KV cache (the autoregressive digit loop) — forward *f* attends over the *f*
tokens emitted so far. So SHALLOW advances the sequence by 16 tokens/step; DEEP by 1.

## Measured (idle RTX A5000, cuda:1; concurrent agent held cuda:0)

Batched, swept to saturation (throughput plateaus by batch ≈ 4096), fp32 **and** bf16.
Doom render-reduced step count = **358,058 steps/frame** (MEASURED — source
`docs/CLEVER_DOOM_REALTIME.md` §3 / `examples/serial_doom_floor.py` `RENDER_STEPS`).
fps = (batched lane-steps/s) / (steps/frame), the spec-decode verify model.

| dtype | config | ms/step (best) | Msteps/s (batched) | **RENDER fps** | ≥30? ≥60? |
|---|---|--:|--:|--:|:--:|
| fp32 | DEEP    | 861.6 | 0.00475 | **0.013** | no |
| fp32 | SHALLOW | 866.2 | 0.00473 | **0.013** | no |
| bf16 | DEEP    | 309.5 | 0.01323 | **0.037** | no |
| bf16 | SHALLOW | 331.5 | 0.01236 | **0.035** | no |

*(best at batch = 4096; all points in the JSON. These are CLEAN — measured on the idle GPU;
the contended cuda:0 was left to the concurrent agent.)*

### The DEEP-vs-SHALLOW gap (the crux)

| dtype | DEEP fps | SHALLOW fps | DEEP/SHALLOW | SHALLOW realtime? |
|---|--:|--:|--:|:--|
| fp32 | 0.013 | 0.013 | **1.01×** | no (both far below 30) |
| bf16 | 0.037 | 0.035 | **1.07×** | no (both far below 30) |

**The K-forwards overhead does NOT meaningfully break SHALLOW relative to DEEP — the gap is
only 1.01× (fp32) / 1.07× (bf16).** At a saturating batch the F=16 forwards run at nearly
the same wall as the single 48-layer forward, because compute (48 layer-applications/step)
dominates and the extra per-launch/KV cost is a small tax. **What kills realtime is NOT the
shallow-vs-deep choice — it is the absolute per-step cost of a full 896-wide transformer
block per digit-layer** (48 blocks/step × 358 k steps/frame). Neither config is close to
30 fps in this full-block-per-layer reading.

> **Two readings of "a layer," stated plainly.** This doc puts a **full vanilla transformer
> block** at every digit-layer — the heaviest, most literally-vanilla reading (in the SHALLOW
> autoregressive case each digit genuinely IS a full model forward). `CLEVER_DOOM_REALTIME.md`
> instead costs each digit-layer as the **bare 10-wide difference-min cell** (~75 FLOP), giving
> ≈ 69 fps fp32. Real fps sits between: a purpose-built clever VM would use a *narrow* per-digit
> cell, not a full 4864-wide SwiGLU block. The load-bearing result here is **the DEEP-vs-SHALLOW
> gap and the launch decomposition**, both robust to the per-layer cost.

## Launch-overhead decomposition (SHALLOW, batch = 4096)

Two overhead components isolated (graph-free, robust):
1. **KV re-read overhead** = (eager F-step) − (F × eager 1-forward). The real F forwards
   re-read a KV cache that grows each forward; the 1-forward baseline has an empty KV.
2. **Fixed per-launch/dispatch floor** = one forward at a tiny (launch-bound) batch of 8,
   where compute ≈ 0. SHALLOW pays this floor **F×/step**; DEEP pays it **1×/step**.

| dtype | 1-fwd (big batch) | F×1-fwd (ideal linear) | F-step (eager) | (1) KV re-read | launch floor/fwd | (2) launch floor/step | extra tax vs DEEP |
|---|--:|--:|--:|--:|--:|--:|--:|
| fp32 | 50.4 ms | 806.7 ms | 840.5 ms | +33.7 ms | 1.48 ms | 23.7 ms (**2.8%** of step) | +22.2 ms/step |
| bf16 | 20.2 ms | 322.6 ms | 334.3 ms | +11.7 ms | 5.87 ms | 93.9 ms (**28.1%** of step) | +88.0 ms/step |

**This is exactly §5's prediction, quantified.** In **fp32** the block is compute-bound at
saturation, so the per-launch floor is only **2.8%** of the step and K-forwards costs almost
nothing (DEEP/SHALLOW = 1.01×). In **bf16** the compute is ~2.6× faster, so the *fixed*
per-forward launch/dispatch floor no longer hides under compute — it balloons to **28.1%**
of the step, and SHALLOW pays it 16× (+88 ms/step extra vs DEEP). **The faster the
arithmetic (lower precision / a leaner per-digit cell), the more the K-forwards launch tax
bites** — precisely why "forwards-per-step is more launch/occupancy-bound." The gap would
widen further with a cheap per-digit cell (the `CLEVER_DOOM_REALTIME` regime), where the
per-forward launch floor is essentially ALL of the cost.

## KV cache — SHALLOW (F× tokens/step) vs DEEP

`KV_bytes = 2(K+V) × n_layers × kv_heads(2, GQA) × head_dim(64) × seq_len × batch × bytes`.
Per lane, full render frame (358,058 steps):

| dtype | DEEP seq | DEEP KV | SHALLOW seq | SHALLOW KV | ratio (full frame) |
|---|--:|--:|--:|--:|--:|
| fp32 | 358,058 | 17.60 GB | 5,728,928 (16×) | 17.60 GB | **1.000×** |
| bf16 | 358,058 |  8.80 GB | 5,728,928 (16×) |  8.80 GB | **1.000×** |

The SHALLOW/DEEP KV ratio folds **two opposing levers**: SHALLOW has **16× more tokens**
(hurts) but **3/48 = 0.062× the layers** (helps). When **L×F = D exactly** they cancel →
**full-frame KV is identical (1.000×)**. The memory pressure that actually bites differs by
which quantity you bound:

- **Per-step KV growth:** DEEP +1 token × 48 layers vs SHALLOW +16 tokens × 3 layers →
  identical bytes/step (49.2 KB fp32 / 24.6 KB bf16, ratio 1.000×). Same cancellation.
- **Bounded eviction window (same seq window W):** here the token count is fixed, so only
  the layer ratio survives → **SHALLOW resident KV = 0.062× DEEP** (16× smaller). For a
  window W = 4096: DEEP 201 MB vs SHALLOW 12.6 MB (fp32). **This is the real SHALLOW KV win**
  — with a bounded window (as the wide build already uses), the shallow network caches 16×
  fewer layers.

So §5's "forwards-per-step *loses* on KV (∝ tokens × steps)" is only the **unbounded** story;
under a **bounded window** the fewer-layers term dominates and SHALLOW is the KV *winner*.
The net depends entirely on whether KV is windowed.

## Verdict

- **The stock-vanilla-fitting SHALLOW config does NOT lose meaningfully to DEEP on
  wall-clock**: same effective depth, DEEP is only **1.01× (fp32) / 1.07× (bf16)** faster.
  The K-forwards launch/KV overhead is a **small tax, not a realtime-breaker**, at a
  saturating batch with a heavy per-layer cost. **Vanilla-fit does NOT cost you realtime
  *relative to* the deep form.**
- **The launch tax is precision-sensitive and grows as the arithmetic gets cheaper**: 2.8%
  of the step in fp32 → 28.1% in bf16. In the cheap-cell regime (`CLEVER_DOOM_REALTIME`) the
  per-forward launch floor would dominate, so a purpose-built shallow clever VM should still
  prefer *fewer, fatter forwards* (larger radix / more digits per forward) to amortise it.
- **Absolute realtime is gated by the per-digit-layer cost, not the depth-mode.** With a full
  896-wide vanilla block per digit-layer neither config clears 30 fps (0.01–0.04 fps); with a
  bare difference-min cell the same shallow form clears 69 fps (that projection lives in
  `CLEVER_DOOM_REALTIME.md`). A realizable vanilla transformer CAN run this in real time **iff
  the per-digit cell is narrow** — the vanilla-fit (shallow) choice itself is not what costs
  the realtime.
- **KV:** full-frame and per-step-growth are **identical** (fewer layers × more tokens
  cancel at L×F=D); under a **bounded window** SHALLOW is **16× smaller** (layer ratio only).

## Golden / reproduce

Golden fingerprint re-verified **unchanged** after this work:
```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<repo>/c4_release python -m c4_min._fingerprint_build
# -> FINGERPRINT 174ece66edff1bb5ab8e9213e484bb1b9560f44ab6c23077a366491543d05637
```
Measurement:
```
python examples/clever_forwards_realtime.py --device cuda:1        # pin an idle GPU
python examples/clever_forwards_realtime.py                        # auto-pick idle GPU (polls under contention)
python examples/clever_forwards_realtime.py --json out.json        # machine-readable
```
All numbers MEASURED on an idle RTX A5000 (device 1; the concurrent timing agent held
device 0). The GPU-contention gate polls for an idle card and flags any provisional run.
