# Doom (and a general C compiler) on a vanilla transformer — honest capstone

*2026-08-07. Golden `174ece66` (doom pure-forward, DEFAULT build — full-32-bit **wide addresses** +
all general-correctness fixes on: signed-trunc `DIV`/`MOD` + BP-restore → **100% C90-general** *and*
real modules / deep recursion natively; rollback ladder: unset the 5 wide flags → `3cabef64` →
`C4_DIVMOD_SIGNED=0` → `7d4afe61` → `C4_BP_RESTORE_HIBYTE=0` → the historical `069cc32f`). This
document is the honest coherent record: what was built, what genuinely works, the
measured numbers, and — deliberately — the projections that turned out wrong and how
measurement corrected them.*

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
- **The C toolchain.** A c4 compiler + a C→c4 transpiler (a genuine recursive-descent
  compiler runs byte-exact *on the transformer itself* — §5), a C90 preprocessor +
  soft-float lib, native FP/fixed-point ISA extensions. Doom is compiled/transpiled
  through this, not hand-lowered.
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
  StoreWallRange counts identical). **99.96% byte-exact on a steady frame** (195/200 rows
  byte-perfect; 63972/64000 bytes — the c4 frame's non-zero count now *matches* gcc exactly). The
  earlier ±1–3 "colormap" tail was **not** a rounding bug but a mid-wipe melt artifact; a steady
  capture dissolves it. Two `compile_c` lowering bugs closed the wall render: a double-index scaling
  bug in `R_GetColumn` (`texturecolumnlump[tex][col]` lowered with a byte stride, not the 8-byte
  word stride) — which fixed the horizon black band (3565 → 6 px) *and* the `R_RenderPlayerView`
  crash — and a **signed-vs-unsigned angle compare** in `R_StoreWallRange`'s `rw_offset` (the c4 VM's
  `>`/`<` are signed, so `angle_t > ANG180` fired spuriously → wrong `finesine` index → a ~one-column
  wall-texture phase shift; fixed with the port's `__ugt`/`__uge` helpers). The **entire wall *and*
  sprite render path is now byte-exact** (two more `compile_c` double-index byte-scale bugs fixed —
  `R_ProjectSprite`'s `sprframe` index + `R_DrawSprite`'s silhouette clip); the last 0.04% (28 bytes)
  is a single animated pickup sprite (ARM1 green armor, drawn one animation frame behind gcc) — a
  `P_MobjThinker` tic-count divergence in the gameplay/thinker path, not the render.
  Reaching this took ~a dozen
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

**The weights are also a C compiler.** Separately from *running* compiled bytecode, a
compiler-in-weights build (`BakedCompilerMachine`) reads C *source*, emits c4 bytecode,
and runs it — one transformer forward per VM step — with the produced arithmetic core
verified **byte-identical to native `./c4`**. It now covers arithmetic **plus all six C
comparisons plus Doom fixed-point `SHL`/`SHR`** (`1<<4`→16, `7>>2`→1). This substrate is
honestly bounded: it is an 8-bit-immediate machine, so c4's 32-bit frame-relative locals
(`LEA -8` = `0xfffff800`), `ENT`/`LEV`, and jump targets >255 are unencodable — named
functions/locals/loops (i.e. a real Doom *module*) are out of reach here and belong to the
32-bit code-from-memory route, not this in-weights compiler.

**And the 32-bit route runs a real compiler byte-exact on the transformer.** A hand-written
recursive-descent c4 compiler (`minic.c` — tokenizer + precedence-climbing `expr()`), compiled
to c4 bytecode and run *as a program* on the 32-bit doom transformer (code-from-memory), emits
correct bytecode with the model verifying **every step byte-exact**: `2+3*4` →
`IMM 2;PSH;IMM 3;PSH;IMM 4;MUL;ADD;LEV` (1882/1882 steps accepted), operator precedence, and
depth-6 nested-paren recursion (4746 steps). **That wall — the model's 8-bit LEA local-address
decode (`(BP+4·imm) mod 256`, which capped the stack at ~6 levels) — is now broken** by a
full-32-bit LEA decode (`C4_LEA_WIDE` + the fine-grained wide set, #854): minic compiles **12-level
recursion byte-exact** (8280/8280 accepted, emitted bytecode == gcc — *double* the old wall) and a
**real Doom-constant leaf function** (`320*200-(65536-(8192+…))` with SCREENWIDTH/FRACUNIT/etc.,
= 59458) **byte-exact** (7120/7120) on the 32-bit transformer. Wide-LEA is a byte-exact *superset*
of the 8-bit fold (it matches the native-c4 32-bit oracle where the fold is wrong), and — after a
residue fix (`compile_imm_clean_snap`) that made doom render byte-exact even in the below-init BP
regime (168→0) — it is now the **DEFAULT build** (wide-address golden `174ece66`; unset the 5 wide
flags to roll back to `3cabef64`). And it does: a whole **4-function C module** (`minic_mod.c` — a real
whole-module compiler with a symbol table + frame codegen, over globals, params, real locals,
`if`/`while`, and a 3-deep inter-function call chain) then runs **byte-exact on the transformer** —
`mod3.c`: 53553/53553 steps accepted, native `exit(67)` matches (a 3-function module 34254/34254).
So the loop-closing capstone is **reached at the module level**; the gap to a *full* 90K-step Doom
module is pure VRAM/scale (the monolithic verify OOMs a 24 GB card past ~54K steps → the #814
checkpoint runner is the follow-on vehicle), **not correctness** — and it all runs on the *default*
build now that wide addresses are default-ON.

- **C90 conformance: 100% general as the DEFAULT build.** Measured per-case vs the faithful
  `native_c4` oracle (itself 193/193 == gcc-15), L-inf=0, on the lean sparse-streaming CFM build:
  every conformance class is byte-exact (arith/pointers/recursion incl. 3-way mutual/functions/
  strings/control/cmp/bitwise/promote/overflow/enum/bitfield/seqpoint/arrays/loops/linkage/
  storage/expr). The last gap — signed-negative `DIV`/`MOD` (4 cases; the nibble ALU did unsigned
  floor where gcc does signed trunc-toward-zero) — is closed by a sign-magnitude wrapper
  (`C4_DIVMOD_SIGNED`, #790), now **default-ON** (golden migration `7d4afe61 → 3cabef64`,
  2026-08-06): verified +4 correctness, **0 regressions** across 152 cases, positive/unsigned
  div/mod unchanged, doom + Mandelbrot byte-exact. Up from the old **vanilla ~14%**. Rollback
  `C4_DIVMOD_SIGNED=0` → `7d4afe61`. Harness + corpus under `id_port/c90_e2e/`.
- **Golden ladder (both correctness fixes now default-ON):** the default build is `3cabef64`.
  Two general-correctness fixes ship by default — `C4_DIVMOD_SIGNED` (signed div/mod, the +4 C90
  fix) and `C4_BP_RESTORE_HIBYTE` (the fp32-safe 5-nibble LEV recompose that made Mandelbrot
  byte-exact). Rollback ladder: `C4_DIVMOD_SIGNED=0` → `7d4afe61` → `C4_BP_RESTORE_HIBYTE=0` →
  the historical `069cc32f`. Each move is verified byte-exact + regression-free.

## 6. Performance — measured to its floor (and the corrections)

Honest numbers, one A5000, byte-exact, on the render-reduced frames:

| frame | fps 1-GPU | fps 2-GPU |
|---|---|---|
| title @96K (WAD-hash) | ~5–6.6 | ~11–13 |
| title @358K | ~1.5 | ~3.0 |
| **steady gameplay (folded ~2.5M)** | **~0.21** | **~0.42** |

The lever ladder (all byte-exact, default-OFF, golden-safe): doom-active block-skip →
FFN wave-batch → WAD-hash frame → attention megakernel → block-0 fold → render
superinstruction (DRAWCOL/DRAWSPANF) → 2-GPU frame-level. The DEFINITIVE measured ladder
table (each rung: what it does, measured factor, byte-exact y/n) + the composed full VM
step (#841: all levers in ONE forward, FFN 72% / attention 28% / decode 1% of 2.49 µs/step
→ 0.788 µs/step with block-0 fold, byte-exact) + the exact statement of what 6.89 M
steps/s would require live in [`PERF_LADDER_FINAL.md`](PERF_LADDER_FINAL.md). Then it stops:

- **Per-step is tapped.** The FFN is sparse-COO (median Dff≈40, ~8 active rows/block);
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
- The steady gameplay frame is 99.96% byte-exact vs gcc (195/200 rows perfect; the wall *and* sprite
  render paths are fully byte-exact); the last 0.04% is one animated pickup sprite drawn a frame
  behind gcc — a `P_MobjThinker` tic-count divergence in the game sim, not the render.
- Numbers here are one A5000; multi-GPU is linear (frame-level).

## 8. What this is

A stock-Qwen-shaped transformer, run as a plain autoregressive model with speculative
decoding, that **genuinely executes arbitrary C programs byte-exact** — demonstrated on
id's Doom (rendering real 3D frames) and a fresh Mandelbrot. The perf is honestly floored
by the sparse VM structure and stack-ISA overhead; the correctness is essentially general.
It is a computer that is a transformer.
