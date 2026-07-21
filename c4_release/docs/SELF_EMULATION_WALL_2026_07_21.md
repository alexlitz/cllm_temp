# Self-Emulation Wall — the transformer running the transformer's forward

**Date:** 2026-07-21  **Branch:** `selfemul-692-continue` (off `worktree-agent-a9311245980f07630 @ 0848db69`)
**Device:** `cuda:1`  **Tool:** `c4_min/measure_self_emulation_wall.py`
(`C4_SELF_EMU_DEV=cuda:1 python -m c4_min.measure_self_emulation_wall`)

## The question (BLOG_SPEC §901 Self-Hosting "TODO performance analysis")

> "How long is ONE step of the transformer, run VIA the transformer?"

A transformer forward IS a matmul stack. So "the transformer emulating its own
forward" = the transformer executing the *bytecode of a matmul* — the inner loop
of the fixed-point ONNX runtime (`onnx_runtime_nibble_fixedpoint.c`) is
`op_matmul` = a `MUL`/`ADD` accumulate. Here ONE VM step (one c4 instruction) is
executed by ONE genuine `transformers.Qwen2Model.forward` (`qwen_full_vm.run_program`),
and the `MUL` a matmul needs is itself computed by the `nibble_alu32` byte-schoolbook
gadget baked into the Qwen MLPs (`build(..., efficient_alu=True)` — no 45 GB lookup
table, no fp64). That is genuine SELF-EMULATION: the transformer emulates the multiply
a matmul forward is made of.

## The build (self-emulation VM)

`Q.build(code_size=24, subset=SUBSET_MULDIV, efficient_alu=True, recurrent_divmod=True)`
on `cuda:1`:

| quantity | value |
|---|---|
| genuine `Qwen2Model` | yes (hidden=1728, intermediate=1124, 138 recurrent layers) |
| MUL/DIV/MOD | `nibble_alu32` schoolbook gadgets in the SwiGLU MLPs (NOT a table) |
| peak CUDA mem | **5.60 GB** (memory-safe on the shared 24 GB cuda:1) |
| build time | ~22–72 s |

## (1)+(2) BYTE-EXACT tiny matvec (MEASURED)

A matmul output element is a dot product `sum_k W[k]*x[k]`; the atom is `MUL`.

| test | program | ref | model | exact? |
|---|---|---|---|---|
| scalar MAC | `y = 3*5` | 15 (numpy & `isa.interpret`) | **15** | **yes** (5 VM steps) |
| tiny matvec `[R×1]@[1]` | `W=[3,5,7,9] · x0=11` | `[33,55,77,99]` | **`[33,55,77,99]`** | **yes** (20 steps) |

The multiply the matmul is built from runs BYTE-EXACT through the real Qwen forward.

## (3) The #702 wall (MEASURED divergence)

A width-2 dot `y = w0*x0 + w1*x1` needs stack DEPTH 2 (park the first product while
computing the second). The register CAM tracks exactly ONE top-of-stack cell
(`STACK0`), so:

| width-2 dot | deep-stack ref | model (1-slot) | diverges? |
|---|---|---|---|
| `w=[3,5] · x=[10,4]` | 50 | **25** | **yes** |

A correct width≥2 dot on this machine therefore requires a **function-frame spill
(JSR/ENT/LEV) or a memory round-trip per accumulate term** — you literally cannot hold
an address and a value simultaneously in one live cell.

## (3b) Speculation forwards-saved, and how #702 gates it

The c4_min logical VM is a DETERMINISTIC draft → speculation runs at **100%
acceptance** (LEAN_FORWARD.md). Two independent FORWARDS-SAVED levers (throughput,
NOT per-token compute):

| lever | forwards-saved | source |
|---|---|---|
| single-program block batching | **32×** (`block_steps=32`, every drafted step accepted) | perfect draft, `pf_speculative.verify_blocks` |
| cross-program batched (B=64) | **54.7×** | **MEASURED prior** (692-selfemul session, `batched_speculative`) |

**How #702 / call-heaviness gates it (the key finding):**
1. **Step inflation (dominant gate).** Each extra dot term costs a spill frame on top
   of its 5-step MAC, so a width-W output element runs `~W·(mac_steps + spill_steps)`
   VM steps, not `W·mac_steps`. Since wall = total_steps × per-step-compute,
   call-heaviness multiplies the wall DIRECTLY. The ~2.4M-step basis already bakes in
   the spilled form.
2. **Lockstep raggedness (batching gate).** Cross-program batching needs B programs at
   a similar call DEPTH to fill a `[B, W, D]` tile with no idle rows (depth-bucketing).
   Call-heavy programs fan out into many depths, so the achievable batch shrinks below
   B and **54.7×@B=64 is an UPPER bound** (hit on uniform shallow loops; lower on deep
   call chains).

Speculation is NOT broken by calls (the deterministic draft knows every dynamic target,
so it drafts THROUGH a JSR); it is GATED by (1)+(2). A single self-forward is ONE
program, so cross-program batching only applies if many self-forwards (or many
independent inner-loop MACs) are run together.

## (4) EXTRAPOLATION — ONE smallest self-forward (LABELED)

Basis: **2,400,000 VM steps** (labeled extrapolation basis for the smallest
self-forward; scales the wall linearly). The wall reduces to two pinning quantities:

- **TOKENS-PER-STEP = 7.0** — STRUCTURAL / INVARIANT. Window = BOS(1) + 5-register
  frame + STEP_END query = 7 tokens per forward (+1 per live memory store). Measured
  `min=max=7`, identical across every run.
- **PER-TOKEN-COMPUTE = per-step-compute / 7** — the CONTENTION-sensitive term.
  Per-step compute measured **~330–1250 ms/step** across runs (this cuda:1 is SHARED
  with a co-resident ~8 GB / ~38 %-util workload), i.e. per-token ~47–179 ms.

`total_tokens = 2.4M steps × 7 tokens/step = 16.8M tokens`.
`wall = 2.4M × per-step-compute`:

| per-step compute | extrapolated wall | vs blog prior |
|---|---|---|
| 312 ms (light contention) | **~8.7 days** | ~10× faster than 88 days |
| 817 ms (medium) | ~22.7 days | ~4× faster |
| 1250 ms (heavy contention) | ~34.7 days | ~2.5× faster |

**Honest headline: ~9–35 days** (measured per-step range on a shared GPU), every point
well under the blog's **88-day @ 3.2 s/instr** prior. With single-program perfect-draft
speculation (32× forwards-saved) this drops to **~0.3 day**, and with the measured
54.7×@B=64 cross-program batching to **~0.16 day** — but those are throughput bounds that
require the corresponding independent-work supply and are capped by #702 call-heaviness.

## What is MEASURED vs EXTRAPOLATED

- **MEASURED (this tool, cuda:1):** byte-exact MAC + tiny matvec; tokens/step = 7.0;
  per-step compute ~330–1250 ms; peak 5.6 GB; #702 dot-2 divergence (50→25);
  single-program forwards-saved = 32 (perfect draft).
- **MEASURED PRIOR (692-selfemul session):** 54.7× forwards-saved @ B=64.
- **EXTRAPOLATED (labeled):** the 2.4M-step self-forward wall = `2.4M × per-step-compute`
  and `16.8M` total tokens. Linear in the basis and in per-step compute.

## Reproduce

```
C4_SELF_EMU_DEV=cuda:1 C4_SELF_EMU_REPS=6 \
  C4_SELF_EMU_JSON=/tmp/self_emu.json \
  python -m c4_min.measure_self_emulation_wall
```
Memory-safe: one lean build, `max_steps ≤ 64`, tiny programs, ~5.6 GB peak.
