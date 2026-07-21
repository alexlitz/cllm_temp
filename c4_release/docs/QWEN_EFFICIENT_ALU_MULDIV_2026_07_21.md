# Efficient ALU (nibble_alu32) baked into the genuine Qwen2Model — MUL/DIV/MOD

**Date:** 2026-07-21  ·  **Branch:** `qwen-efficient-alu-muldiv`
**Files:** `c4_min/qwen_full_vm.py`, `c4_min/test_qwen_full_vm.py`,
`c4_min/_verify_qwen_efficient_alu.py`

## Summary

`qwen_full_vm` — the fused VM that runs one whole C4 step per
`transformers.Qwen2Model.forward` — previously baked MUL/DIV/MOD as a **256×256×3
lookup table** in the SwiGLU MLP. That table's hidden width is `intermediate ≈
160465` → **~45 GB fp32**, unbuildable in RAM; the corpus only ran it via a
per-program **pruned** operand-key table.

This change bakes the **existing** `nibble_alu32` efficient ALU (the BLOG_SPEC
`§Basic Arithmetic` / `§Multiplication` / `§Division` fp32 gadgets — no lookup
table, no fp64) into the genuine Qwen2Model in place of that table. It is the SAME
efficient ALU already wired into `nibble_pure_forward_complete`; the Qwen bake just
wasn't using it. Nothing was reinvented — `compile_expand`, `compile_mul_blocks`,
`compile_divmod_blocks` / `compile_divmod_blocks_recurrent`, `compile_ax_mux`,
`compile_psh_nibble_copy` are used verbatim.

Enable with `Q.build(..., efficient_alu=True[, recurrent_divmod=True])`. The
lookup-table path is unchanged and remains the default (byte-identical).

## What the efficient ALU costs: width → depth

The lookup table trades a *huge width* for zero depth. The efficient ALU trades that
width for **depth** (schoolbook carry rounds + long-division iterations). Measured on
the fused Qwen VM (`subset=+muldiv`, `fit_report_efficient()`):

| path                         | intermediate | stored layers | applied layers | fits stock 24L? |
|------------------------------|-------------:|--------------:|---------------:|:---------------:|
| **lookup table** (full)      | **160465**   | 12            | 12             | no (45 GB wall) |
| **efficient, unrolled**      | **1124**     | **285**       | 285            | no (285 > 24)   |
| **efficient, recurrent div** | **1124**     | **138**       | 285            | no (138 > 24)   |

Per-op depth cost (blocks added on top of the base+mem+cmp 10-layer decode/CAM stack):

| op       | blocks | detail                                                        |
|----------|-------:|---------------------------------------------------------------|
| operand expand | 1 | STACK0/AX nibbles → operand bytes (`compile_expand`)     |
| PSH nibble-copy | 1 | full 32-bit push into STACK0 (`compile_psh_nibble_copy`) |
| **MUL**  | **10** | products + split + 7 carry rounds + result (`compile_mul_blocks`) |
| **DIV/MOD (unrolled)** | **262** | KB-precompute + init + 8×(shift, gteq, qdigit, qcopy, qb, 6 qb-carry, 9 sub-nibble, r2r) + finalize |
| **DIV/MOD (recurrent)** | **115 stored / 262 applied** | ONE reused iteration body ×8 |
| AX mux   | 1      | opcode-gated writeback of the active op's RES nibbles → AX     |

**MUL alone** (base+mem+cmp 10 + expand/psh 2 + MUL 10 + mux 1 ≈ **23 layers**) fits
the **stock Qwen2.5-0.5B** budget (24 layers, intermediate 4864). **DIV/MOD does
not**: 262 long-division blocks push the model to ~285 layers — over every stock
Qwen depth (0.5B=24, 1.5B=28, 3B=36, 7B=28). This is the honest, fundamental depth
cost of a genuine iterative long division.

### The recurrence is what makes DIV/MOD buildable

`recurrent_divmod=True` folds the 8 long-division iterations into **ONE reused
iteration body** (a Universal-Transformer-style recurrence). The model **STORES** 138
distinct layers but **APPLIES** 285 per forward — `qmodel.layers` is repointed to a
`ModuleList` of repeated module references (shared `nn.Module` weights, identical
math), and `config.num_hidden_layers` is set to the applied length. This is exactly
the `_apply_order` recurrence in `nibble_pure_forward_complete`, ported onto Qwen2's
`ModuleList`. RoPE depends on **token position**, not layer index, so reusing a layer
at multiple depths is exact. The register CAM (layer 0) and the memory CAM are the
only attention layers; all ALU layers are pure-FFN (zeroed attention), so reuse is
collision-free.

**Smallest genuine-Qwen config that fits full MUL+DIV+MOD:** none of the stock
configs by *layer count*. The efficient ALU is only buildable as a genuine Qwen2 when
you (a) set `num_hidden_layers` to 285 (unrolled) / 138-stored-285-applied
(recurrent) — a *custom-depth* Qwen2Config, still a genuine `Qwen2Model`, or (b) cap
the op set to MUL (fits stock 0.5B). The **win** is the *width*: intermediate drops
160465 → 1124, so the model is buildable in RAM (the 45 GB wall is gone), which the
lookup table never was for the full domain.

## Blockers checked (from the pure-forward path) — none bite the Qwen bake

The brief flagged three known efficient-mode blockers from the pure-forward path:

1. **L15/L20 multi-byte-result corruptor (product → 61656).** This is a
   `neural_vm` (old 27-layer architecture) corruptor. The `c4_min` green-field
   `nibble_alu32` writes each op's result into its OWN dedicated RES band
   UNCONDITIONALLY and mux-copies it gated on `OP_IS[op]`, so there is no shared
   OUTPUT band to corrupt. **Verified absent:** MUL/DIV/MOD are byte-exact.

2. **d_model 920→952 regression.** Also a `neural_vm` dim-budget artifact. The Qwen
   bake auto-sizes `hidden_size` to the layout (1664–1728) with head-dim padding;
   there is no fixed 920 budget. **Not applicable.**

3. **Smoke historically used lookup not efficient.** Resolved: `efficient_alu=True`
   is a first-class build flag with its own test coverage.

The one **real** discovery was an fp32-precision concern (the SiLU-gated MUL, the
7-carry-round settle, the 262-layer RMSNorm renormalization). All were verified
byte-exact in fp32 both as a raw gadget AND through the real 285-layer Qwen forward.

## Verification (all through `transformers.Qwen2Model.forward`)

`c4_min/_verify_qwen_efficient_alu.py` (recurrent build, CPU, mem-guarded):

- **MUL/DIV/MOD battery (24 cases): 8-bit vs `isa.interpret` → 24/24.**
- **Battery 32-bit vs `nibble_muldivmod` (mul32/divmod32/mod32) → 24/24.** Includes
  the *32-bit* cases the 8-bit interpreter cannot represent: `200*3 = 600`,
  `16*16 = 256` (the ALU computes them exactly; `isa.interpret` masks to `88`/`0`).
- **Mandelbrot `z = z² + c` inner loop (15 cases): 15/15** — the MUL sub-terms
  (`zx*zx`, `zy*zy`, `zx*zy`) plus the combining SUB/ADD, chained through the fused
  forward.
- Gadget-level fp32 check of full 32-bit operands (`1e9/7`, `0xDEADBEEF/0x1234`,
  `2^32-1`, …): **MUL 8/8, DIV 8/8, MOD 8/8** — 32-bit-EXACT, the capability the
  8-bit lookup table never had.

### 8-bit vs 32-bit note

`isa.interpret` is an **8-bit** reference (`ax = (pop() * ax) & 0xFF`), matching the
corpus's 8-bit operand-load path. The efficient ALU is **32-bit-exact**. `run_program`
takes `mask=0xFF` (default, matches `isa.interpret`) or `mask=0xFFFFFFFF` (the genuine
32-bit result). The full 32-bit result lives in the AX **nibble band** only at the
ALU-op step; a trailing non-ALU step (e.g. `HALT`) reads the scalar `AX_VAL`, which is
folded mod 256 — so the 32-bit demo reads the value at the ALU-op step.

## API

```python
from c4_min import qwen_full_vm as Q
vm = Q.build(code_size=24, subset=Q.SUBSET_MULDIV,
             efficient_alu=True, recurrent_divmod=True)   # genuine Qwen2Model
r = Q.run_program(vm, code, mask=0xFFFFFFFF)              # 32-bit-exact MUL/DIV/MOD
Q.fit_report_efficient()                                  # depth/width cost table
```
