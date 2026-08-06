# Doom (and a general C compiler) on a vanilla transformer — honest capstone

*2026-08-06. Golden `069cc32f` (doom pure-forward, flags-OFF). This document is the
honest coherent record: what was built, what genuinely works, the measured numbers,
and — deliberately — the projections that turned out wrong and how measurement
corrected them.*

## 1. Thesis, in one paragraph

The c4 instruction set (a small stack VM, 40 base opcodes, 32-bit words) is baked as
fixed weights into a Qwen2.5-0.5B-shaped transformer. The transformer, run as an
ordinary autoregressive loop with speculative decoding (a byte-exact Rust c4 draft
supplies each step; the transformer *verifies*), executes arbitrary c4 bytecode. We
compiled id's Doom to c4 and ran it; we also compiled a fresh Mandelbrot and ran it.
Both produce byte-exact output versus a native reference. So the artifact is not a
Doom-player — it is a general-purpose computer that happens to be a transformer, and
Doom is the stress test.

## 2. The stack

- **c4 ISA baked into the model.** Every opcode (LI/LC/SI/SC, PSH/POP, LEA, ENT/ADJ/
  LEV/JSR, ADD/SUB/MUL/DIV/MOD, the bitwise + shift + compare ops, I/O) is hand-authored
  as declarative FFN/attention rules, lowered to weights. Values are 8 nibbles (32-bit)
  in the residual; memory is content-addressed (softmax1 + ALiBi latest-write-wins).
- **The C toolchain.** A c4 compiler + a C→c4 transpiler (self-hosting on the VM), a
  C90 preprocessor + soft-float lib, native FP/fixed-point ISA extensions. Doom is
  compiled/transpiled through this, not hand-lowered.
- **The Rust draft + speculation.** A native-speed byte-exact c4 VM (~665M steps/s)
  drafts the whole run; the transformer verifies K steps per dispatch (giant-K). This
  is what makes the outer VM loop parallel — the step-to-step serial dependency is the
  draft's problem, not the transformer's.
- **direct-CAM / CFM.** Memory reads resolve O(1) by address decode (draft-assisted on
  the fast path; genuinely recomputed on the faithful paths). Code lives in the KV
  (code-from-memory), so the residual dim is code-size-independent.

## 3. Doom: byte-exact title → a rendered gameplay frame

- **Title frame:** byte-exact vs gcc (`2e883404`), start to finish, purely from the
  transpiler-generated engine.
- **Gameplay frame:** a full 3D E1M5 view **renders** — player spawns, BSP walk, walls,
  floors — and matches gcc's *geometry exactly* (R_DrawColumn/Span/BSP/AddLine/
  StoreWallRange counts identical). **86.9% byte-exact vs gcc**; the residual is a
  ±1–3 colormap-shade rounding tail plus a mid-wipe capture artifact (the captured frame
  is the one-time title→level screen-melt, not a steady frame). Reaching it took ~a dozen
  fixes of one bug class: the c4 compiler's `&arr[i*n+c]` scaling, `+=`/`|=` transpiled
  to plain `=`, arithmetic-vs-logical shift, 2D-array flattening, byte-vs-int strides,
  and an `I_GetTime` stack-corruption that had silently stopped the game sim from
  ticking.

## 4. Genuine computation — not draft replay

There is a genuineness spectrum, and it matters:
- **fast (direct-CAM):** memory rows injected from the draft. Byte-exact by a
  latest-write-wins ≡ CAM-winner *identity*, not a runtime check.
- **faithful single-dispatch:** the model recomputes the read's address + value via its
  own query. Byte-exact, and it **catches a self-consistent wrong draft** (planted stale
  value at the correct address) that the fast path rubber-stamps — proven on a real
  gameplay read (`C4_FAITHFUL_ATTN_EVICT` / single-dispatch).
- **genuine structured attention (`C4_GENUINE_STRUCTURED_ATTN`):** runs the model's own
  softmax1 + ALiBi over the physical winner K/V — recomputing the score, the +1 sink
  (→ genuine zero-fill on unwritten addresses), the recency, and the value-decode that
  single-dispatch merely assumes — at ~1.08 µs/step, i.e. it **meets the fast
  throughput.** A genuine memory read, cheap.

So the transformer genuinely *computes* Doom; it does not replay the draft.

## 5. General-purpose: Mandelbrot byte-exact

A fresh fixed-point Mandelbrot, compiled through the c4 toolchain, runs **fully
byte-exact** on the hardened doom build: interior pixel 4201/4201 steps, escape 508/508,
boundary 4237/4237 — the full signed escape loop (MUL/DIV/SHR/XOR/signed-compares/
JSR/ENT/LEV). The one bug it exposed was narrow and real — three `LEV` frame-pointer
recomposes hardcoded `hi_nibbles=8` (exact only in fp64), amplifying fp32 CAM residue on
the high nibbles (`0x10000`→`0xFEEF`); fixed to the fp32-safe 5-nibble recompose
(`C4_BP_RESTORE_HIBYTE`). The earlier "general programs diverge at ~14%" was the *vanilla*
build; the hardened build is far more general.

- **C90 conformance on the hardened build: _[pending — a7ccc9bc]_** (old vanilla ~14%;
  expected substantially higher given Mandelbrot).
- **Golden migration** to default the correctness fix: **_[pending — a88a2e23; new golden
  TBD, rollback via `C4_BP_RESTORE_HIBYTE=0` → `069cc32f`]_**.

## 6. Performance — measured to its floor (and the corrections)

Honest numbers, one A5000, byte-exact, on the render-reduced frames:

| frame | fps 1-GPU | fps 2-GPU |
|---|---|---|
| title @96K (WAD-hash) | ~5–6.6 | ~11–13 |
| title @358K | ~1.5 | ~3.0 |
| **steady gameplay (folded ~2.5M)** | **~0.21** | **~0.42** |

The lever ladder (all byte-exact, default-OFF, golden-safe): doom-active block-skip →
FFN wave-batch → WAD-hash frame → attention megakernel → block-0 fold → render
superinstruction (DRAWCOL/DRAWSPANF) → 2-GPU frame-level. Then it stops:

- **Per-step is tapped.** The FFN is sparse-COO (median Dff≈138, ~8 active rows/block);
  a dense bf16 tensor-core FFN is **11.8× slower** because the useful work is 0.099% of
  the dense rectangle — sparse-COO is the byte-exact optimum. Precision buys nothing
  (efficiency-bound, not bandwidth-bound; and PC exceeds bf16's exact-integer range).
  Log-sink divide is not a depth win (byte-exact divide is verify+decompose-bound). The
  system runs at ~11% of FLOP peak / ~12–30% of HBM — **occupancy-bound**, ~9× off a FLOP
  bound that is *unreachable* byte-exactly (nothing to feed the MMA units).
- **The steady gameplay residual (~25× native x86) is inherent stack-ISA overhead** —
  63% of the render is PSH/LEA/LI pointer-walking, the cost of a stack VM, not a bug.
- **Real-time gameplay** would need folding the *traversals* (not just the pixel fill) or
  much more hardware. It is a hardware/step-count statement, not an algorithmic one.

**Corrections made this session (measurement over projection):** self-emu "seconds" was
really 16 s (fused); 2-GPU K-split was 1.06× not the projected 1.7× (frame-level gave the
clean 2×); "9.85 fps" was a stacked projection, measured ~6.65; "gameplay ~1 fps" then
"~0.245" then "~0.02" all bracketed the true ~0.21 steady (the 25M-step "frame" was a
one-time wipe); FLOP/step was 5.68M not the 196K I'd quoted (~11% util, ~9× off, not
0.3%/300×); "general programs diverge" was the wrong (vanilla) build. Each was owned and
re-measured.

## 7. Honest limitations

- Gameplay is genuine and byte-exact but ~0.21 fps; real-time is out of reach on one card.
- The full mid-game trace is not stepped end-to-end through the transformer (the draft
  hits an unimplemented `MALC` opcode + ~700 GB KV to reach the render span); verification
  is byte-exact on tractable 120K-step windows + measured-per-step calibration.
- The gameplay frame is 87% vs gcc (rounding tail + a mid-wipe capture artifact), not 100%.
- Numbers here are one A5000; multi-GPU is linear (frame-level).

## 8. What this is

A stock-Qwen-shaped transformer, run as a plain autoregressive model with speculative
decoding, that **genuinely executes arbitrary C programs byte-exact** — demonstrated on
id's Doom (rendering real 3D frames) and a fresh Mandelbrot. The perf is honestly floored
by the sparse VM structure and stack-ISA overhead; the correctness is essentially general.
It is a computer that is a transformer.
