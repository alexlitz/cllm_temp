# MUL root REWRITE: efficient-mode L11 placement drift (2026-06-13)

Status: **mul_basic FIXED (+1 smoke: 48->49). mul_overflow remains
arch-blocked (width=2).** The fix is two lines in
`make_efficient_l11_alumul_wrap_op` (`neural_vm/unified_compiler/ops/alu_ops.py`).
All probes spec_k=0, hook-free, on the real `trust_neural_alu=True`
(=`alu_mode='efficient'`) smoke path. Smoke gate: **49 passed / 2 failed**
(`test_mul_overflow`, `test_simple_function`) — zero regressions across the
48 prior-passing tests (ADD/SUB/bitwise/shift/CMP/memory all hold).

## 1. The brief's lookup-mode premise is REFUTED for the smoke gate

The fix brief (and `AND_MUL_MARK_AX_ENDRUN_2026_06_11.md` /
`BLOCK8_OPERAND_GATHER_IS_CLEAN_2026_06_11.md`) analysed **lookup mode**
("MUL computes NOTHING; FlattenedALUMul count=0; L11/L12 plain PureFFN with
empty post_ops"). But the production smoke gate runs **EFFICIENT mode**:

```
neural_vm/run_vm.py:323
    alu_mode = 'efficient' if trust_neural_alu else 'lookup'
```

and the smoke fixture builds `AutoregressiveVMRunner(... trust_neural_alu=True)`
(`tests/conftest.py:425`). In efficient mode the MUL compute IS installed:
`compile_full_vm_dynamic`'s `if alu_mode=="efficient":` block registers the
9 FlattenedALUMul stages + `make_efficient_l11_alumul_wrap_op`, which bakes a
256-rule `wide_mul_rules(width_bytes=1)` PureFFN. The Wave-B MARK_SE_ONLY
migration, the SE relay starvation, the dead `all_alu_postop_attach`
dispatch — all lookup-mode concerns, irrelevant to the gate.

## 2. The REAL root: the 256-rule wide_mul FFN baked onto the WRONG block

`tools/probe_mul_full_fix_validate.py` + the dispatch/resolve trace prove:

* The MUL operands are **present and clean** at the MARK_AX MUL row through
  L11: operand A as an `ALU_LO` one-hot (magnitude ~6, NOT 1.0), operand B
  as an `AX_CARRY_LO` one-hot (~0.9). `OP_MUL=5.0`, `MARK_AX=1.0`.
* But `OUTPUT_LO/HI stay EMPTY (0.0) through physical blocks 12/13`
  (logical L11/L12) — the wide_mul FFN never fired there. Blocks 12/13 were
  **hidden=1, all-zero** (right-sized-to-dead PureFFNs).
* The 256-rule wide_mul PureFFN was actually baked onto **physical block 26
  = logical L15** (`hidden=256`). Tracing the bake: the wrap op binds via
  `target_op_name="_layer11_ffn_dep_anchor"`, and that anchor resolves to
  **pre-expansion layer 15** (-> physical block 26 = logical L15), NOT L11.
* That misplaced L15 wide_mul is what wrote the noisy `OUTPUT_LO[0]≈23k /
  OUTPUT_HI[0]≈25k` spread at block 26 that prior docs called "the L15
  OUTPUT materialiser". The L20 (block 37) tail spike (±1.36e9) then
  amplified that noise into a wrong decode -> `mul_basic = 1`.

### Why the anchor mis-resolves to L15

The dep-scheduler stacks the L10 op family
(`layer10_byte_passthrough` / `_sp_byte_passthrough` /
`_psh_stack0_passthrough` / `_psh_ax_broadcast` / `_stack0_byte_relay` /
`layer10_carry_relay`) across **pre-exp layers 9..14** (one per op, serial
dep chain). `_layer11_ffn_dep_anchor` declares
`requires={"after": "layer10_carry_relay"}` (pre-exp 14), so it floats to
pre-exp 15 — which after the L8(+1)/L14(+8)/L25(+1)/shift expansions maps to
physical block 26 = logical L15, four logical layers past its intended L11
(physical block 12 = pre-exp block 11). The "L11" naming is aspirational;
the dep graph placed it at L15. The Phase-8.G.6 commit
(`4b94cf8f l11_ops: drop layer_idx=11 ... via new dep anchor`) introduced
this drift by removing the explicit pin.

## 3. The fix (alu_ops.py, validated -> 42)

```python
# make_efficient_l11_alumul_wrap_op:
#  (a) restore the explicit layer pin (replaces the mis-resolving anchor bind)
-   target_op_name="_layer11_ffn_dep_anchor",
+   layer_idx=11,            # pre-exp block 11 -> physical 12 -> logical L11
    requires={"after": "l12_alu_mul_getobd"},   # KEPT (discard FlattenedALUMul)

#  (b) match the AND thresholds to the real operand magnitudes
    rules = wide_mul_rules(..., 
+       operand_a_cond_weight=5.0,   # 5 * ALU_LO one-hot(~6) = 30
+       operand_b_cond_weight=30.0,  # 30 * AX_CARRY_LO one-hot(~0.9) = 27
+       marker_cond_weight=40.0,     # marker present = 40
+       threshold=80.0)              # all-on=97>80; drop-any<=70<80
```

Default 30/30/40+thr80 assumes 1.0/1.0 one-hots; with `ALU_LO≈6` the
`30*6=180` operand-A term alone exceeded 80, so every `(a=k, b=*)` rule fired
and the product band filled with noise. Rescaling makes it a clean 3-way AND.

Both fixes are required: re-pin without rescale -> still 1 (noise); rescale
without re-pin -> still wrong (compute at L15 + L20 spike).

## 4. mul_overflow: genuine arch-block (width=2)

`test_mul_overflow` = 100*5 = 500 = 0x01F4 needs an 8-bit x 8-bit product.
`wide_mul_rules(width_bytes=1)` multiplies only the LOW NIBBLES (100&0xF=4,
5&0xF=5 -> 4*5=20=0x14), so post-fix mul_overflow now decodes **20** (was 1).
The full byte product needs `width_bytes=2` (a 65536-rule flat lookup over
all 4 nibbles). Three independent walls block it, all confirmed this session:

1. **Operand-encoding poison (Wall-1).** The L8 operand-gather emits a
   ~5.5-magnitude **cell-0 artifact** on `ALU_LO[0]`/`ALU_HI[0]` alongside
   the real nibble one-hots (e.g. 100: `ALU_LO[4]=5.84, ALU_LO[0]=5.52,
   ALU_HI[6]=5.84, ALU_HI[0]=5.54`). A 65536-rule 5-way AND over a
   contaminated encoding fires multiple rules; a width=2 sweep across 4
   threshold configs produced exit 0/1 for BOTH mul_basic and mul_overflow.
   This is the documented unrecoverable hybrid magnitude+nibble encoding;
   the clean fix is upstream in the L8 gather (off-limits).
2. **Result-band collision.** `result_base=OUTPUT_LO` width=2 writes 4 nibble
   lanes; `OUTPUT_LO+32 = dim 101 = CLEAN_EMBED_LO` and
   `OUTPUT_LO+48 = dim 117 = CLEAN_EMBED_HI` (load-bearing token-decode
   dims). An 8x8 product is <=16 bits so byte2/byte3 are always 0 — a
   2-lane variant (byte0+byte1 only) would dodge the collision, but Wall-1
   sinks it first.
3. **65536-unit FFN cost** on a single L11 block (and the
   `project_mul_div_mod_arch_blocked` d_model-expansion regression risk if a
   wider result band is used).

Per `feedback_single_rule_fixes_are_zero_sum` + the brief's HARD-STOP, I did
not force a regressing/scaffolding width=2 change. mul_overflow stays
arch-blocked pending the L8 operand-gather cleanup.

## 5. Bearing on lookup-mode + 1096

The same anchor drift affects `layer11_mul_partial` / `layer12_mul_combine`
(lookup mode) and `efficient_l10_andorxor_wrap` (which baked onto physical
block 15 = logical L14, also wrong — but bitwise has a working fallback
compute so or_basic/and_basic still pass; the misplaced andorxor wrap is
inert). Any 1096 program whose MUL fits single-nibble operands is unblocked
by the L11 re-pin; multi-byte MUL inherits the width=2 arch-block.

## Artifacts

* `tools/probe_mul_full_fix_validate.py` — block-placement + mul_basic/
  mul_overflow result check on the real efficient-mode model (spec_k=0).
* Fix: `neural_vm/unified_compiler/ops/alu_ops.py`
  `make_efficient_l11_alumul_wrap_op` (layer_idx pin + threshold scale).
