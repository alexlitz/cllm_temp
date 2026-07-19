# FETCH-DEDUP — sub-linear model-runs-C fetch (2026-07-18)

Branch: `fetch-dedup-model-runs-c` (off `nibble-model-runs-c-full` @ `0badb844`).

Shrinks the model-runs-C fetch/decode from **O(code_size)** toward **O(sqrt)** so
the FULL c4 compiler fits a tractable residual width `D` instead of the ~12k /
~59 GB OOM wall the prior agent pinned
(`NIBBLE_MODEL_RUNS_C_VALIDATION_2026_07_18.md`).

## The wall (baseline)

Each code slot carried THREE residual bands `CODE_WORD[i] + PC_IS[i] + ADDR_IS[i]`
and the fetch/word-select FFN was O(code_size) wide. Measured:

    D = 3.00 * code_size + 148

so the full ~4000-instr c4 compiler projects to **D ≈ 12,196**. At D=12k the
per-block cost is dominated by the attention `4·D²` term → **~59 GB / 20 blocks**
(the Mandelbrot/79GB OOM class).

## The fix (nibble_fetch_dedup.py, both bit-exact)

1. **REGION SPLIT.** The baked-compiler region (PC `0..gen_size`) is CONSTANT —
   words ride weight biases (no `CODE_WORD` dim) and are never an EMIT target
   (no `ADDR_IS` dim). Only the small WRITABLE produced-code region (`out_size`
   slots, contiguous at PC `gen_size..`) keeps `OUT_WORD` + `OUT_ADDR_IS`.
   `3·code_size → code_size + 2·out_size`.

2. **FACTORED PC ONE-HOT.** `PC_IS` is consumed ONLY by the fetch (branches read
   PC/IMM/AX scalars). Replace the full `code_size`-wide one-hot with base-`B`
   digits `PC = B·PC_HI + PC_LO`: two `~sqrt(N)` one-hots. Baked word at slot
   `B·h + l` is selected by a 2-input AND `PC_HI_IS[h] AND PC_LO_IS[l]` gated on
   the constant word. PC addressing costs `~2·sqrt(N)` dims, not `N`.

**Precision (load-bearing):** the WORD self-clear MUST use the exact power-of-two
`POW2=256` normaliser (`1/256` exact), not `1/silu(60)=1/60` (inexact). Otherwise
at large `WORD_prev` the clear undershoots ~1e-3, `OP_IS` decodes 0.9999, the SUB
`+256` lands at 255.96 < the mod-fold threshold 256, the fold silently fails, and
AX carries garbage.

## Scaling — D vs code_size (before → after)

| gen_size | baseline D | dedup D | shrink | model bytes (baseline → dedup) |
|---------:|-----------:|--------:|-------:|--------------------------------|
| 64       | 388        | 200     | 1.9x   |                                |
| 135 (expr compiler) | 604 | 208 | 2.9x |                            |
| 512      | 1,732      | 228     | 7.6x   |                                |
| 1024     | 3,268      | 248     | 13x    |                                |
| 2048     | 6,340      | 276     | 23x    |                                |
| **4000 (full c4)** | **12,196** | **440** | **28x** | **59.3 GB → 0.49 GB (121x/block)** |

D now grows `~= base + 2·sqrt(code_size) + 2·out_size` — SUB-LINEAR. The full c4
compiler builds in 0.2 s at ~1 GB RSS.

## Demonstration (byte-identical to real c4)

- **Expression compiler** (`nibble_compiler`, region-split + factored PC): 8/8
  `d op d op d` cores byte-identical to real c4 at **D=208** (was 676).
- **Variable-length LOOP compiler** (`loop_compiler.py`, 83 baked instrs +
  moving-pointer `EMITP`): scans a null-terminated source, LOOPS emitting the c4
  pattern `IMM d0;[PSH;IMM di;OP]*;HALT` at the runtime cursor `OUT_PTR`, hands
  off (`JMP`), runs it. In-model, byte-identical to `bundler/c4_compile.c` for:
  `2+3+4+5=14`, `2*3*4*5=120`, `1+2+3+4+5+6=21`, and a **20-operand** chain
  (39-char source → 59 produced instrs, ~1060 recurrent forward passes) `= 20` —
  all MATCH c4's arith core, correct greedy result. Compiler-in-weights **D=504**.

This is "larger than a bare expression": the compiler LOOPS over a variable-length
source and emits a variable-length program at a moving pointer.

## The NEW wall (honest)

The residual `D` is now sub-linear, so it is no longer the binding constraint.
Two milder walls remain:

1. **Fetch FFN hidden width is still O(gen_size)** — one AND product unit per
   nonzero baked slot in `compile_dedup_word_select`. This is fundamental for a
   DENSE code table (positions are all distinct even when word VALUES repeat, so
   value-dedup buys nothing). BUT it is a far cheaper wall: a 4049-wide FFN at
   D=440 is **~6.1 M params/block vs the baseline's ~741 M** (121x smaller); the
   whole 20-block model is ~0.49 GB. Memory is no longer the limit.

2. **Runtime + 8-bit values** — each VM step is one full recurrent transformer
   forward pass and the fetch re-runs every step, so running a produced program
   of length L costs ~O(L) steps and compiling costs ~O(source·emit). A
   20-operand chain is ~1060 steps / 18 s on CPU. Produced results must stay
   < 256 (the mod-256 fold), inherited from the c4_min 8-bit ALU.

Every gadget the full c4 compiler needs (JSR/ENT/LEV/ADJ, DIV/MOD, cmp/bitwise/
shift, I/O) already exists as a c4_min nibble op and wires in as ~1 FFNRule/op;
the fetch (the size wall) is now sub-linear. The remaining gap to a literal
`c4_compile.c` bake is engineering the full lexer/parser bytecode + the ENT/LEV
frame (which needs signed local-var LEA), not a residual-width blocker.

## Artifacts (branch `fetch-dedup-model-runs-c`)

- `c4_min/nibble_fetch_dedup.py` — the deduped layout + factored fetch + word
  select + region-split EMIT + moving-pointer EMITP + loop-compiler step builder.
- `c4_min/loop_compiler.py` — the variable-length chain compiler in c4_min
  bytecode + run driver.
- `c4_min/demo_model_runs_c_dedup.py` — end-to-end demo.
- `c4_min/validate_vs_real_c4.py` — extended: byte-compares the dedup loop
  compiler to real c4 + prints the full-c4-scale D wall.
- `c4_min/test_fetch_dedup.py` — 5 tests (factored fetch exact, byte-identity vs
  reference, sub-linear D, loop compiler, full-c4-scale small-D build).
