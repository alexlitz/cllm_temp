# Composed continuous fps + the real-doom DIV/MOD gap (measured)

Composed ALL landed doom-on-transformer perf levers into ONE run, measured the
honest continuous per-frame time, and quantified whether the REAL doom frame (not
the DIV-free proxy) can hit 1 fps.

## The composed stack (all landed levers)

Merged onto `consolidate-0.5b` (which carried `6b5e15d8` GPU-build + double-buffer
scaffold and `e000ab90` `C4_SCHED_PIPELINE` true overlap):
  * `c1685e15` — FFN-hidden fusion `C4_FFN_FUSED_HIDDEN` + tile knob
    `C4_MEGABLOCK_BLOCK_K` (single-kernel on-chip `[Dff,K]` hidden delta).

All flags default-OFF. The composed measurement enables them together:
`C4_SCHED_GPU_BUILD=1 C4_SCHED_PIPELINE=1 C4_FFN_FUSED_HIDDEN=1
C4_MEGABLOCK_BLOCK_K=512` (+ the composed attention/CAM/megablock stack).
Harness: `c4_min/_agent_continuous_frame.py` (the two new flags added to
`_levers_on`), same DIV-free `nested(120,255)` = 462,979-step proxy #1 used,
interpolated to 358,058 / 6.89M steps. GPU A5000.

## Task 1 — definitive composed continuous fps (DIV-free proxy)

Measured (n=462,979 steps, steady-state median, warmup discarded):

| path | ms/frame | µs/step |
|---|---|---|
| build (schedule tables) | 953.78 | **2.060** |
| dispatch (graph replay)  | 883.27 | **1.908** |
| SERIAL (build + dispatch) | 1837.04 | 3.968 |
| **TRUE-PIPE (double-buffer overlap)** | **1025.49** | ~max(build,dispatch)=build |

TRUE-PIPE steady-state per-frame walls (ms): `[2095, 2020, | 1040, 1028, 1025,
1024, 1014]` (first 2 = warmup). Steady ≈ **1025 ms/frame ≈ build (954 ms)**.

Interpolated:

| frame | SERIAL | TRUE-PIPE |
|---|---|---|
| render-reduced @358,058 | 1.421 s (0.70 fps) | **0.793 s/frame → 1.26 fps ≥1 fps** |
| raw @6,889,264 | 27.34 s (0.04 fps) | 15.26 s (0.07 fps) |

**Which half bounds it:** BUILD-bound. The fused dispatch dropped to 1.908
µs/step (from the pipeline-only 2.544 baseline), now BELOW the on-device build
(2.060 µs/step). Pipeline overlap makes the frame ≈ max(build, dispatch) = build.
Composed @358K = **0.793 s/frame = 1.26 fps**, vs the individual levers'
0.915 s/frame (pipeline-only, #1). The FFN fusion shaved the dispatch enough that
the build is now the ceiling — the remaining lever is the schedule BUILD, not the
dispatch.

Byte-exact: TRUE-PIPE (all flags on) vs SERIAL decode L-inf = 0 (PC/SP/BP/AX) over
the full 462,979-step frame. VRAM peak **16.0 GB** @ chunk 131072 (24 GB ceiling).

## Task 2 — THE key question: can the REAL doom frame hit ≥1 fps? (DIV/MOD gap)

The single-dispatch precomputed schedule asserts DIV-free (`run_verify` rejects
DIV/MOD; a DIV/MOD step must fall back to the slow recurrent-divmod `verify_blocks`
path, ~168-179 blocks, ~3 ms/step composed). The proxy is DIV-free; is the REAL
doom render frame?

Measured the DYNAMIC opcode histogram of a REAL steady subsequent doom frame on
the faithful native `c4vm32` (CPU-only, no model). Tools (in `c4_doom/id_port/`):
`_doom_frame_opcode_gap.py` (compile `doom_run.c`, raise the frame cap, tap every
executed opcode, report per-frame deltas; `--pow2` applies the #829
DIV→SHR/MOD→AND peephole) and `_doom_opcode_histogram.py`.

### Opcode histogram of a REAL steady subsequent doom frame (6,889,264 steps, render-only, WAD excluded)

| opcode | count | % | note |
|---|---|---|---|
| PSH | 1,801,325 | 26.1% | |
| LEA | 1,577,237 | 22.9% | |
| LI | 1,164,265 | 16.9% | |
| IMM | 896,131 | 13.0% | |
| ADD | 631,143 | 9.2% | |
| SI | 415,353 | 6.0% | |
| LC | 290,687 | 4.2% | |
| **AND** | **202,244** | **2.9%** | ← pow2 MOD reduced here + masks |
| BZ | 170,930 | 2.5% | |
| JMP/ADJ/PUTCHAR/SUB/GT/SC/LT | ~90-160k each | | |
| **SHR** | **64,002** | **0.93%** | ← pow2 DIV reduced here + shifts |
| MUL | 5,954 | 0.09% | |
| JSR/ENT/LEV | ~5,466 each | | |
| GE/NE/SHL/EQ/BNZ/LE | small | | |
| **generic DIV** | **0–2** | **0.00003%** | ← the only slow-path divs |
| **generic MOD** | **0** | 0% | |

### The counts (of the 358,058-step render-reduced frame this scales to)

* generic **DIV + MOD** in a steady render frame: **0–2** (base `doom_run.c`);
  **0** after the #829 pow2 peephole. At the 358,058-step render-reduced scale
  this rounds to **~0 generic DIV/MOD per frame**.
* pow2-reduced-to-cheap (SHR + AND): ~64k SHR + ~202k AND executions/frame — these
  ARE the former divides/mods, now a single cheap shifter/mask block each.
* the other 38 opcodes: everything else (PSH/LEA/LI/IMM/ADD/SI/… dominate; the
  frame is ~90% address/stack/load/store + integer ALU).
* Cross-check: static scan of `doom.c4img` = 92 constant pow2 DIV/MOD *sites* (84
  DIV→SHR, 8 MOD→AND). The steady render loop is ALREADY nearly shift-only —
  `doom_run.c`'s fixed-point macros emit shifts directly, so the SHR/AND counts are
  identical with and without the peephole; the peephole only mops up the last 0-2
  residual pow2 DIV.

### Is the real doom frame effectively DIV-free after the reductions?

**YES.** A steady subsequent doom render frame contains **0–2 generic DIV/MOD**
(0 after #829 pow2 reduction) out of ~6.89M steps; the 358,058-step render-reduced
frame is effectively **DIV-free**. The fast single-dispatch DIV-free schedule CAN
carry a real doom frame — the schedule's DIV-free assertion is satisfied by real
doom, not just the proxy.

Where the 1,175 generic DIV/MOD went: they are ALL in the ONE-TIME WAD-load +
R_Init (the 114.8M-step first frame), NOT in the recurring render (steady frames
2/3/4 = 0-2). The first frame amortizes across an entire play session.

### If N>0 residual generic DIV/MOD: the frame-time impact

Even in the worst case (base image, 1-2 residual divs/frame), the tail is
negligible. Each fallback DIV/MOD step runs the recurrent-divmod `verify_blocks`
path at ~3 ms/step (vs the scheduled ~1.9 µs/step, ~1600× slower). So N residual
divs add ~N × 3 ms to the frame. With the whole scheduled 358K frame ≈ 0.79 s, the
divmod tail would only become the DOMINANT cost at N ≳ 260 divs/frame
(260 × 3 ms ≈ 0.79 s). We measured N ≈ 0-2, so the tail is ~0-6 ms — a rounding
error on a 0.79 s frame.

### HONEST VERDICT

**Real doom CAN hit ≥1 fps with the current composed stack, on the DIV-free
single-dispatch schedule.** The render-reduced doom frame is effectively DIV-free
(0 generic DIV/MOD after #829; 0-2 without it), so it rides the same fast path the
proxy does: composed TRUE-PIPE = **0.79 s/frame = 1.26 fps @358K** — over 1 fps.

Nothing extra is required to carry real doom on the DIV-free schedule (no need to
teach the schedule to carry the divmod span; no need to reduce further generic
divs — there are essentially none left in steady render). The remaining perf
ceiling is the schedule BUILD (build-bound at 2.06 µs/step > dispatch 1.91), i.e.
the composed lever to push past 1.26 fps is faster on-device build, NOT DIV/MOD.

Caveat on scope: this is the render-REDUCED frame (358,058 steps). The RAW
title-redraw frame (6.89M steps) is still ~15 s/frame composed — 1 fps at raw
resolution needs a further ~15× (multi-GPU / larger build-parallelism), independent
of DIV/MOD.

## Golden (sacrosanct) — flags default-OFF, UNCHANGED

* default golden `069cc32f` (`python -m c4_min._fingerprint_build`) ✓
* CFM golden `7d19cdc3` (`C4_PF_CFM=1 python -m c4_min._fingerprint_build`) ✓
* `C4_FFN_FUSED_HIDDEN` and `C4_MEGABLOCK_BLOCK_K` are default-OFF; the golden
  build is byte-identical with them unset.

## Byte-exact confirmation (composed all-flags-on == reference)

* fused-hidden `C4_FFN_FUSED_HIDDEN=1` block_k=512: mega Linf(fused vs 2-kernel)
  = 0.00e+00 (K=512, 8192) — `_agent_fusedhidden_verify.py`.
* deep-loop mega-FUSED-HIDDEN (block_k=512) vs EAGER reference: 64/64 AX steps
  match — `_agent_megablock_deeploop.py`.
* full-frame composed (ALL flags: GPU-build + pipeline + fused-hidden + bk512) vs
  SERIAL: decode L-inf = 0 (PC/SP/BP/AX) over 462,979 steps —
  `_agent_continuous_frame.py`.

VRAM peak: **16.0 GB** (24 GB ceiling).
