# Mandelbrot end-to-end on the c4 stack — 2026-08-05

Can we run a Mandelbrot on the c4 transformer in reasonable time, and is the
draft / HF path there today? Measured across **compile → draft → transformer →
HF-config**, plus the float caveat. Read-only measurement (authors NO weights;
golden untouched — this 2026-08-05 snapshot was taken against the then-default golden
`069cc32f`, which is now the `C4_BP_RESTORE_HIBYTE=0` rollback; the default golden moved to
`7d4afe61` on 2026-08-06 when the general-correctness fix went DEFAULT-ON).
Harness: `c4_min/_agent_mandelbrot_e2e.py`.

## TL;DR

* **Draft (c4 word-VM): yes, seconds.** The REAL Mandelbrot (`mandelbrot_native`)
  renders 80×48 mi50 in **~8.6 s** (24.3 M VM steps @ ~2.8 M steps/s), **byte-exact
  vs a native gcc** reference of the same fixed-point algorithm.
* **Transformer: byte-exact but only a TINY slice today.** A single 1×1 cell runs
  **byte-exact end-to-end through `model.forward`** in **262 steps / ~0.98 s/step /
  ~4.3 min on GPU** (peak RSS 3.5 GB). A **full** grid is blocked on TWO model
  limits (an AX-0xFF-leak correctness wall + a KV-CAM aliasing wall), not merely
  perf — so the transformer does NOT render a full Mandelbrot byte-exact today.
* **HF-config: the base ISA nearly fits stock Qwen2.5-0.5B; the muldiv ISA
  Mandelbrot needs does NOT.** Mandelbrot needs fixed-point MUL+DIV → the `+muldiv`
  subset, which is **111 layers / hidden 3136** (the `full` ISA is 131) — still a
  genuine `Qwen2ForCausalLM` *architecture*, but ~4.6× the released 0.5B depth. It
  runs on HF only as a much larger config, OR folded to fixed depth via recurrence
  (`recurrent_divmod`, the divmod megablock the streaming build already uses).
* **Float caveat: N/A for Mandelbrot — it is INTEGER fixed-point.** No soft-float
  and no native-FP (#800) is invoked. The whole render is integer MUL/DIV on the
  32-bit ALU (`ref_interpret(mask=0xFFFFFFFF)`).

## 1. Compile

Both Mandelbrot programs compile on the real c4 toolchain:

* `_mandel_src.mandel_c(W,H,MI)` → the REAL c4 C compiler (`src.compiler.compile_c`)
  → `bytecode_to_isa`. **260 instrs**, a static grid loop (constant in W/H/MI; the
  grid is a runtime `while` loop), compile ~1.6 ms, **zero IMM>255** (byte-exactness
  precondition), verified op subset only.
* `mandelbrot_native.pixel_program_native` → a c4-assembly per-pixel escape LOOP,
  **179 instrs**, CONSTANT in max_iter.

## 2. Draft (c4 word-VM, `ref_interpret` @ mask=0xFFFFFFFF) — seconds, byte-exact vs gcc

| path              | 80×48 mi50 VM steps | draft wall | rate      | gcc byte-exact |
|-------------------|--------------------:|-----------:|-----------|:--------------:|
| `mandelbrot_native` (REAL set)     | **24,292,161** | **~8.6 s** | 2.8 M/s | **YES** (3888 B identical) |
| `_mandel_src` (unsigned slice)     |      3,561,638 | ~1.5 s     | 2.4 M/s | NO (see note)  |

The native draft is the recognizable Mandelbrot (window cx∈[-2.1,0.9],
cy∈[-1.2,1.2], toward-zero fixed-point), matching `mandelbrot_reference_preview.txt`
and a **byte-for-byte** native-gcc reference of the identical `sfp_mul`/`_pixel_escape`
algorithm (`_agent_mandelbrot_e2e gcc`).

Draft scaling (native path): 16×8→0.34 M steps, 32×16→2.06 M, 64×32→10.4 M,
80×48→24.3 M. Interior pixels cost up to ~17.4 k steps (the full mi50 loop),
escaped pixels ~0.5 k. All well under a second to a few seconds. **The draft is
there today, in reasonable time.**

Note on `_mandel_src`: it is deliberately restructured to keep every AX value
NON-NEGATIVE (unsigned first-quadrant window) so the transformer's AX-0xFF-leak
never fires → it is the transformer-SAFE program, but its render is the VM's own
UNSIGNED escape-time slice (a degenerate `*   *` lattice) and is NOT gcc-byte-exact
(gcc's signed ints escape where the unsigned VM's compare does not; gcc yields 1
`*`, the c4 draft yields 480). This is the honest tension: the visually-correct
program (`mandelbrot_native`) uses signed storage that the transformer can't yet
carry; the transformer-safe program renders a degenerate slice.

## 3. Transformer (`model.forward`, streaming sparse + KV-cached driver, eviction ON)

**Fresh measurement on this box (GPU cuda:0), reproducing the 2026-07-20 result:**

* 1×1 mi1 cell (origin → inside → `*`): **262 neural steps, byte-exact.**
* **run wall 257.9 s, 0.98 s/step, peak RSS 3.52 GB** (build 33.5 s), KV cache
  bounded by eviction. `BYTE-EXACT MATCH: True` — the transformer's OWN PRTF
  LM-head decode == `ref_interpret` (`b"*\n"`), not a Python copy.

**The capability boundary is a CORRECTNESS wall, not just perf** (both root-caused):

1. **Wrapped-negative AX corrupts the next `LEA`** (AX 0xFF high-byte leak).
   `mandelbrot_native`'s signed biased storage hits this → why the transformer
   path uses the non-negative `_mandel_src` slice instead.
2. **KV-memory address-CAM aliasing under interleaved local re-stores.** A 2-cell
   run stays byte-exact through step ~135, then an `LI` returns a neighbouring
   slot's value → grids larger than ~1 cell desync before the PRTF emission.

So a **full** Mandelbrot through the transformer is not byte-exact today. Even if
it were, at ~1 s/step the native 24.3 M-step render is a ~280-day wall (and the
unsigned-slice 3.56 M-step render ~41 days). The transformer path is a **byte-exact
single-cell demonstration**, not a full render — honestly bounded.

## 4. HF-config feasibility (stock `Qwen2ForCausalLM` via `qwen_vanilla_vm.py`)

`qwen_vanilla_vm.build(subset=...)` produces a genuine stock `Qwen2ForCausalLM`
(real `Qwen2Attention`/`Qwen2MLP`, 14 q-heads / 2 kv-heads, real embed+lm_head,
standard autoregressive loop — NO overlay). The question is whether the config
fits the RELEASED Qwen2.5-0.5B shape (24 layers, hidden 896, inter 4864). Measured
(`_agent_mandelbrot_e2e hf`, code_size 64):

| subset      | n_layers | hidden | inter | fits released 0.5B | note |
|-------------|---------:|-------:|------:|:------------------:|------|
| base        |   14 |  896 | 5210 | no (inter 5210>4864) | fits released on layers+hidden; inter slightly over |
| base+mem    |   16 | 1280 | 5210 | no (hidden) | stock Qwen2 *class*, larger config |
| mem+cmp     |   18 | 1280 | 5210 | no (hidden) | stock Qwen2 *class*, larger config |
| +bitwise    |   38 | 1280 | 5210 | no (layers) | stock Qwen2 *class*, larger config |
| **+muldiv** | **111** | **3136** | 7920 | **no** | what **Mandelbrot needs** |
| full        |  131 | 3136 | 7920 | no | +muldiv+bitwise |

**Mandelbrot needs fixed-point MUL+DIV → the `+muldiv` subset → 111 layers.** It is
still a real `Qwen2ForCausalLM` architecture (proven by
`test_full_isa_still_stock_qwen2_with_wider_frames`), just ~4.6× the released depth
and hidden 3136, because the byte-exact 32-bit divmod gadget is deep. The `base`
ISA nearly fits the released 0.5B (14≤24 layers, hidden 896) — only its 5210
intermediate is a touch over 4864 — but base has no MUL/DIV so it can't do
Mandelbrot. Two ways onto HF for the muldiv Mandelbrot:
(a) instantiate the larger 131-layer / hidden-3136 Qwen2 config (vanilla, but not
the released 0.5B weights); (b) **recurrence** — the streaming forward already folds
the divmod megablock (`recurrent_divmod=True`) so the *effective* depth is fixed;
the same fold makes a fixed-layer HF config feasible. The pure vanilla
single-forward path is NOT the released 0.5B for Mandelbrot; it either grows the
config or uses the divmod recurrence.

## 5. Float caveat (soft-float vs native-FP #800)

**Not applicable to this Mandelbrot** — it is INTEGER fixed-point (scale=16/64), so
every multiply/divide is a native integer MUL/DIV on the 32-bit ALU. No soft-float
subroutine and no native-FP32 opcode is exercised.

For reference, a genuinely-float Mandelbrot would pay the #800 cost: native-FP32
(`C4_FLOAT_OPS`, default OFF, default golden `7d4afe61` unchanged) is bit-exact on
normal×normal (f32add 620/676, misses = ±0/subnormal specials) but the byte-exact
`F_DIV` is a ~26-step restoring-division megablock (deep). Fixed-point sidesteps all
of it — one MUL + one DIV per complex multiply — which is exactly why the c4
Mandelbrot is fixed-point.

## Bottom line

* **Draft on the c4 word-VM: yes, seconds, byte-exact vs gcc** (real 80×48 in ~8.6 s).
* **Transformer: byte-exact for a single cell** (262 steps, ~4.3 min GPU); a full
  render is blocked by a correctness wall (AX-leak + KV-CAM aliasing), not just perf.
* **HF-config: base ISA nearly fits the released 0.5B (14 layers / hidden 896;
  inter 5210 a touch over); the muldiv ISA Mandelbrot needs is a 111-layer
  stock-Qwen2 (or a recurrence-folded fixed-depth config), not the released 0.5B.**
* **Float caveat: none — Mandelbrot here is integer fixed-point.**

Reproduce: `PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m
c4_min._agent_mandelbrot_e2e {draft,gcc,hf,transformer --device=cuda:0}`.
Golden gate (unchanged): `python -m c4_min._fingerprint_build`.
