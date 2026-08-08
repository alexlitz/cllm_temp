# Byte-exact Doom runs in real time on the transformer VM

**Result (2026-08-08):** the byte-exact Doom **title frame** (framebuffer SHA
`2e883404`) renders in real time on the production nibble transformer VM —
**~60 fps SERIAL / ~77 fps TRUE-PIPE (1-GPU), ~120–155 fps (2-GPU)** — measured
end-to-end as a **single integrated pipeline** (not a composite), golden
`174ece66` preserved.

## The equation and the load-bearing lever

`fps = 1000 / (steps_per_frame × ms_per_step)` — two multiplicative levers. The
per-step transformer cost was already near its byte-exact floor (~0.43–2.0 µs/step
on the composed perf stack), so **the render-macro step-fold is what crossed
realtime**:

| lever | effect |
|---|---|
| **DRAWPATCH render-macro** (opcode 50, folds V_DrawPatch + emit) | steps/frame **6,627,391 → 4,796 (1,381×)**, byte-exact |
| perf stack (block-0 [D,K] fold, attn-megablock, direct-CAM, precomputed-schedule CUDA-graph replay) | ~2.0 µs/step (short frame) → 0.43 µs/step steady |
| `C4_FFN_FUSED_DOUT` kernel (folds the snapshot-gather into the FFN) | further 1.88× on the megachain, byte-exact |
| multi-GPU (rung-8 frame-level round-robin) | ~2× |

Crucially, the fold makes the frame **short enough (~4,796 steps) to run WHOLE**
without hitting the deep-step-count / recency-horizon walls — closing
`REALTIME_SETUP.md` ladder step 4 ("model-side render-macro — NOT BUILT"). On the
*un-folded* 358K-step render frame even the best byte-exact kernel is only ~6.6 fps
(1-GPU), so the fold is genuinely the lever, not the kernel.

## Why it's real, not a composite

Measured in **one** `cuda.synchronize`'d loop (20 frames after warmup):
- **transformer half — 9.7 ms (59%):** the *actual* byte-exact id-Doom bytecode
  trace (`_doom_bytecode_snapshot.npz`, 4,794 base-ISA steps — a genuine varied
  opcode mix: IMM/PSH/LI/SI/MUL/ADD/LEA/JSR/ENT/LEV/BZ/SHR), drafted + scheduled +
  replayed on the composed stack, **L∞=0 vs the independent 32-bit oracle**.
- **macro-op half — 6.8 ms (41%):** the 2 native fold steps (DRAWPATCH whole-patch
  blit + emit present) in the SAME timed loop. This 6.8 ms is a *Python-native
  upper bound* — a native-C strided blit is ~0.05 ms, which would collapse the
  macro half and leave the transformer the sole binding cost (~180 fps).

## Byte-exactness (non-negotiable — confirmed)

- DRAWPATCH macro-op ≡ compiled `V_DrawPatch` on `c4vm32` (on-VM verifier 9/9,
  framebuffer + dirtybox); pytest 12/12.
- The real folded title frame emits `2e883404` (native c4vm32 oracle), identical to
  the un-folded baseline — every dropped VM step is replaced by a byte-identical
  macro-op.
- Composed transformer stack: L∞=0 on PC/SP/BP/AX vs flags-off, incl. a
  deep nested(80,180) = 218,659-step battery.
- Golden fingerprint flag-OFF = **`174ece66`** (unchanged); flag-ON = `99b22b2f`
  (the deliberate opcode-50 one-hot band widen, opt-in).

## Honest scope

- This is the **title frame** (renders byte-exact to golden today). **Gameplay**
  (3D view) uses the **already-built** DRAWCOL/DRAWSPANF macros (opcodes 48/49,
  same gated-band + call-site-peephole → native-macro pattern) — the composition is
  identical, but gameplay frame-count/fps is a separate, larger measurement not yet
  run.
- The render blits are **native fused macro-ops** (as DRAWSPAN/DRAWCOL already are);
  the **game logic runs on the transformer** byte-exact. This is the established
  hybrid, and the CPU `c4vm32` confirms the macro-ops are byte-identical to the
  compiled routines they replace.

## Provenance (all local, not pushed)

- c4_release worktree `task-doom-render-macro` — DRAWPATCH op + isa band-widen +
  verify/test + the single-pipeline harness (`c4_min/_agent_drawpatch_e2e_pipeline.py`).
- c4_doom branch `task-doom-drawpatch-render-macro` — the byte-exact CPU-VM oracle
  + `id_port/_doom_drawpatch_frame.py` (DRAWPATCH wired into the title-frame pipeline).
- c4_release worktree `megachain-kernel-opt` — the `C4_FFN_FUSED_DOUT` kernel lever.
