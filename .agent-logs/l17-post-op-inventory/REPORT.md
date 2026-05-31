# L17 Post-Op Inventory (Investigation Branch: investigation/l17-post-op-inventory)

Base commit: `4d069f7` on `speedup-cache-and-buckets`.
B6-C reference: `investigation/block27-layer27-amplification @ f13ff72`.

## Context

B6-C found that "block27 / layer27" surfaced in B4-B's investigation was actually
the L10 tail (`tail_bit32_result_correction`) FFN baked into L17 via the
post-op pipeline. This inventory enumerates ALL FFN blocks that land in L17
(and every other layer's post-op attach point) so future investigations can
identify whether a suspect block sits in one of these "hidden" structural FFNs.

Key takeaway up front: **L17 in the layout owns TWO FFNs that did not exist
when L17 was the simple last layer.** Both came from L10:
- `block.ffn` = `l10_post_ops_combined` (BinaryOpByteZeroing + 3x carry + ComparisonCombine baked into a single 1846-unit FFN; dependency-anchored at L17 by `kind="ffn", phase=10.5`)
- `block.post_ops[0]` = `tail_bit32_result_correction` (~2091 units; bound to the above via `target_op_name="l10_post_ops_combined"`, phase=17.1).

After Phase 0 expansion (`_expand_wrapper_blocks`), these become two distinct
TransformerBlocks (block 30 = `l10_post_ops_combined`, block 31 = `tail_bit32_result_correction`).

---

## Section 1: All compiled blocks

Configuration: defaults to `compile_full_vm()` (alu_mode="lookup", n_heads=8,
ffn_hidden=4096, pin_io_only=True, use_dynamic_ffn=True). Source layer count
in the layout is **18** (L0-L17); the legacy `n_layers=17` literal in
`vm_step.py:2627` ("Total blocks: 17 -> 32") is a stale print string — the
actual pre-expansion block count is 18. After Phase 0 expansion the model
has **32** blocks (14 post_op expansions).

### 1a. Pre-expansion blocks (the 18 layout layers)

Captured with `_expand_wrapper_blocks` / `_right_size_ffns` patched out so we
see the bake state immediately after `compile_full_vm`:

| Layer | block.ffn class | hidden_dim | post_ops (in execution order) |
|-------|-----------------|-----------:|-------------------------------|
| L0 | PureFFN | 7 | — |
| L1 | PureFFN | 5 | — |
| L2 | PureFFN | 10 | — |
| L3 | PureFFN | 4096 | — |
| L4 | PureFFN | 544 | — |
| L5 | PureFFN | 4096 | — |
| L6 | PureFFN | 2328 | — |
| L7 | PureFFN | 4096 | — |
| **L8** | PureFFN | 2055 | `AddSub5StageBlock` |
| **L9** | PureFFN | 3405 | `AddSub5StageBlock` |
| **L10** | PureFFN | 1846 | `ALUAndOrXor`, `BinaryOpByteZeroingPostOp` (hd=8), `AddSubBytePropagationPostOp` (hd=1536), `CarryPropagationPostOp` (hd=512, byte=0, cascade=False), `CarryPropagationPostOp` (hd=512, byte=1, cascade=True), `CarryPropagationPostOp` (hd=512, byte=2, cascade=True), `BitwiseBytePropagationPostOp` (hd=1536), `FlattenedDivMod` |
| **L11** | PureFFN | 4096 | `FlattenedALUMul` |
| **L12** | PureFFN | 4096 | `FlattenedALUMul` |
| **L13** | PureFFN | 4096 | `ALUShiftComposite` |
| L14 | PureFFN | 1874 | — |
| L15 | PureFFN | 42 | — |
| L16 | PureFFN | 696 | — |
| **L17** | PureFFN | 1846 | `PureFFN` (= `tail_bit32_result_correction`, hd=2091) |

After `_right_size_ffns` (dead-unit pruning), key trimming events:
- `L17.ffn: 1846 -> 1562 units (-284 dead)` (this is `l10_post_ops_combined`; the dead units are the zeroed carry slice noted in `make_l10_post_ops_combined`)
- `L17.post_ops[0]: 2091 units (no dead units)` (the tail correction)
- `L14.ffn: 1874 -> 1873 units (-1 dead)`
- `L12.post_ops[0]._stages.2.sub_ffn.flat_ffn: 1808 -> 1792 units (-16 dead)`

### 1b. Post-expansion blocks (the 32 final blocks)

Phase 0 splits each `block.post_ops[*]` into its own TransformerBlock with a
passthrough attention. Mapping (source layer -> expanded indices):

| Source layer | Expanded indices | Notes |
|------|------------------|-------|
| L0..L7 | 0..7 | unchanged (no post-ops) |
| L8 | 8 (ffn=PureFFN h=2055), 9 (`AddSub5StageBlock`) | +1 |
| L9 | 10 (ffn=PureFFN h=3405), 11 (`AddSub5StageBlock`) | +1 |
| L10 | 12 (ffn=PureFFN h=1846), 13 (`ALUAndOrXor`), 14..18 (5 split PureFFNs: BinaryOpByteZeroing, AddSubByteProp, 3x Carry), 19 (`BitwiseBytePropagationPostOp`), 20 (`FlattenedDivMod`) | +8 |
| L11 | 21 (ffn=PureFFN h=4096), 22 (`FlattenedALUMul`) | +1 |
| L12 | 23 (ffn=PureFFN h=4096), 24 (`FlattenedALUMul`) | +1 |
| L13 | 25 (ffn=PureFFN h=4096), 26 (`ALUShiftComposite`) | +1 |
| L14 | 27 (ffn=PureFFN h=1873) | unchanged |
| L15 | 28 (ffn=PureFFN h=42) | unchanged |
| L16 | 29 (ffn=PureFFN h=696) | unchanged |
| **L17** | **30 (ffn=`l10_post_ops_combined` h=1562 after trim), 31 (`tail_bit32_result_correction` h=2091)** | **+1** |

Total: 18 source + 14 splits = 32 final blocks.

`_rebake_as_pureffn` re-homes 6 of the split modules into vanilla `PureFFN`
(those that subclass `PureFFN` without overriding `forward`: the
`BinaryOpByteZeroingPostOp`, 3x `CarryPropagationPostOp`, `BitwiseBytePropagationPostOp`,
`AddSubBytePropagationPostOp`, plus the tail correction at block 31). The other
8 post-ops are wrapper composites (`AddSub5StageBlock` x2, `ALUAndOrXor`,
`FlattenedALUMul` x2, `ALUShiftComposite`, `FlattenedDivMod`) that keep their
class.

---

## Section 2: Post-op contents — rules and source files

### 2a. L17 post-op contents — the focus

L17 carries the entire L10-correction tail, despite the name. Both ops are
authored in `c4_release/neural_vm/unified_compiler/ops/l10_ops.py`.

**L17.ffn = `l10_post_ops_combined`** (1846 declared units, 1562 active after trim)
- Owner: `make_l10_post_ops_combined()` at `l10_ops.py:1289`.
- Phase: `10.5`, kind: `ffn`, `migrated=True`, `declarative_authority="declarative"`, `ffn_units_used=1846`.
- Sub-FFNs baked in sequentially via `_bake_post_op_into`:
  1. `BinaryOpByteZeroingPostOp` (8 units) — zeros AX bytes 1-3 for byte-0-only opcodes (EQ/NE/LT/GT/LE/GE/SHL/SHR/MUL/DIV/MOD).
  2. `CarryPropagationPostOp(byte_idx=0, cascade=False)` (512 units) — **ZEROED in the bake** (`l10_ops.py:1334-1338`), kept only as a placeholder for stable indexing.
  3. `CarryPropagationPostOp(byte_idx=1, cascade=True)` (512 units) — **ZEROED**.
  4. `CarryPropagationPostOp(byte_idx=2, cascade=True)` (512 units) — **ZEROED**.
  5. `ComparisonCombine` (~302 units) — combines `CMP+0..3` flags into truth-value byte writes for EQ/NE/LT/GT/LE/GE.
- Post-bake guards (applied across all `:offset` units):
  - Hard-block (`-S * 10^6`) on `OP_IMM`, `OP_JMP`, `OP_LI_RELAY`, `OP_LC_RELAY`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_SHL`, `OP_SHR`.
  - Hard-block (`-S * 10^6`) on `CARRY+1/2/3`.
  - Hard-block on `TEMP+8/9/10` (ADD/SUB/MUL relay markers).
  - Block on `CMP+7` (LEA at AX byte rows).
  - Block on `H1+0` (PC byte span).
  - Step-boundary suppressor: `_suppress_ffn_on_step_boundary` (CONST < 0 gating).

**L17.post_ops[0] = `tail_bit32_result_correction`** (2091 units, no dead units)
- Owner: `make_tail_bit32_result_correction_op()` at `l10_ops.py:4529`.
- Phase: `17.1`, kind: `block`, `target_op_name="l10_post_ops_combined"`, `migrated=True`, `declarative_authority="spec_generated"`.
- Body source: `_tail_bit32_result_correction_rules()` at `l10_ops.py:1489`.
- Distinct rule families (~77 named rules; some families generate multiple FFN units each):

| Rule family | Rule count | Concern |
|-------------|-----------:|---------|
| `tail_sp_pop_carry_byte1_zero` and `sp_pop_carry_rules` | 1+ generated per byte_idx | SP pop carry upper bytes |
| `tail_sp_pop_marker_{e0_to_e8,d0_to_d8,f0_to_f8,d8_to_e0,output_d8_to_e0}` | 5 | SP marker increment after binary-pop |
| `tail_sp_pop_byte1_ff_after_{e0,d8,f8,e8}` | 4 | SP byte 1 = 0xff preservation |
| `tail_stack0_pushed_addr_byte1_{ff_after_e8, store_ff_after_{e0,e8}}` | 3 | Pushed-address byte 1 preservation |
| `tail_ax_lea_local_addr_byte1_preserve*` | >=1 | LEA-local AX byte 1 |
| `tail_stack0_store_{nonzero_pair, pop_loaded_output, top_e0_output, top_e8_from_e0_output, loaded_output, top_value_from_alu_*}` | 6+ | Stack0 store/pop outputs |
| `tail_stack0_pop_{marker_zero, reveals_saved_addr_e8_*}` | 3 | Stack0 pop revelation |
| `tail_stack0_store_non_top_zero{,_e0,_e8_from_e0}` | 3 | Stack0 store non-top zero |
| `tail_stack0_f8_byte1_from_output_exact` | 1 | Stack0 byte 1 from output |
| `tail_sp_pop_byte3_zero`, `tail_sp_store_pop_byte1_zero` | 2 | SP pop tail zeros |
| `tail_pc_byte0_{12_from_initial_jmp_exact, 1a_from_taken_branch_index3_exact_{bz,bnz}}` | 3 | PC byte 0 from JMP/BZ/BNZ |
| `tail_pc_byte1_01_from_{long_initial_pc, initial_jsr_fetch_hi}_exact` | 2 | PC byte 1 from initial JSR |
| `tail_sp_marker_byte0_f8_from_initial_stack_exact`, `tail_sp_byte1_ff_from_initial_stack_exact` | 2 | Initial SP marker bytes |
| `tail_mem_store_addr0_{f8_exact, f0_exact, f8_initial_jsr_exact, f8_initial_jsr_authority, 00_from_global_exact, f8_from_mod_local_exact, e0_from_local_offset_exact, e0_from_psh_sp_no_addr_src_authority, e0_from_jsr_local_{exact,strong}, e8_from_nested_local_exact, e8_from_local_frame_addr_exact, e8_from_local_frame_output_exact}` | 13 | **MEM store addr0 family — this is the family B6-C touched (`tail_mem_store_addr0_f8` got over-strength tightening)** |
| `tail_mem_store_addr1_ff_from_stack_store_exact`, `tail_mem_store_addr2_zero_from_global_exact`, `tail_mem_store_addr3_zero_from_global_exact` | 3 | MEM store addr1/2/3 |
| `tail_pop_mem_marker_zero`, `tail_bp_byte2_preserve_01` | 2 | Pop MEM marker / BP preservation |
| `tail_ax_add_{byte1_hi_zero, no_carry_byte1_00, byte1_no_carry_low{1_02,2_03}, byte1_carry_low2_03, byte1_carry_high2_03, byte1_missing_stack_high_02}`, `tail_ax_add_byte1_structural_materialize_*`, `tail_ax_add_mul_byte1_materialize_{01,02_from_hi2,02_from_hid}` | ~10 | AX ADD byte-1 family (interacts with L8/L9 ADD post-ops) |
| `tail_ax_sub_{byte1_hi_zero, full_underflow_byte1_ff, borrow_byte1_*}`, `tail_sub_borrow_byte1_00` | 4 | AX SUB byte-1 family |
| `tail_si_ax_byte1_{12,00}` | 2 | SI AX byte 1 |
| `tail_wide_shl_byte1_01` | 1 | Wide SHL byte 1 |
| `tail_shr_{byte1_00, marker_byte0_01}` | 2 | SHR byte 1 / marker correction |
| `tail_and_byte1_00`, `tail_or_xor_byte1_0f` | 2 | Bitwise byte 1 |
| `tail_lea_local_ax_marker_byte0_e8` | 1 | LEA local AX marker byte |
| `tail_cmp_{ne_true_01, eq_false_00, le_lt_true_01, le_eq_prefix_false_00, lt_false_00, gt_false_00}` | 6 | Comparison repairs (marker-only) |
| `tail_clear_output_{after_byte3, before_step_end}` | 2 | Output zeroing guards |

Plus three blanket transformations applied to subsets:
- `pc_byte_span_blocked` — adds PC-byte-row blockers to most rules.
- `stack0_span_blocked_tail_rules` — adds Stack0-row blockers to several rules.
- `step_end_transition_blocked` — adds step-boundary blockers (skipped only on the two `tail_clear_output_*` rules and a few address-source rules).

### 2b. Post-op contents at other layers (for context)

| Layer | Module | Source factory | Source file |
|-------|--------|----------------|-------------|
| L8.post_ops[0] | `AddSub5StageBlock` | `make_efficient_l8_addsub_wrap_op` (efficient) OR `_make_alu_postop_attach_op("l8_alu_postop_attach", 8, "ALUAddSub", "lookup")` (lookup; default) | `unified_compiler/ops/alu_ops.py` + `_make_alu_postop_attach_op` body in `shared.py` |
| L9.post_ops[0] | `AddSub5StageBlock` | `make_l9_alu_postop_attach_op` | `unified_compiler/ops/alu_ops.py:226` |
| L10.post_ops[0] | `ALUAndOrXor` | `make_l10_alu_postop_attach_op` (`insert(0, ...)`) | `unified_compiler/ops/alu_ops.py:232` |
| L10.post_ops[1] | `BinaryOpByteZeroingPostOp` (class in `vm_step.py:751`) | `make_l10_post_op_attach_op` (phase 10.7) | `unified_compiler/ops/l10_ops.py:4592` |
| L10.post_ops[2] | `AddSubBytePropagationPostOp` (`vm_step.py:992`) | same | same |
| L10.post_ops[3..5] | 3x `CarryPropagationPostOp(byte_idx={0,1,2})` (`vm_step.py:822`) | same | same |
| L10.post_ops[6] | `BitwiseBytePropagationPostOp` (`vm_step.py:1131`) | same | same |
| L10.post_ops[7] | `FlattenedDivMod` | `make_l10_alu_divmod_install_op` (phase 10.8) | `unified_compiler/ops/alu_ops.py:906` |
| L11.post_ops[0] | `FlattenedALUMul` | `make_l11_alu_postop_attach_op` ("ALUMul") | `unified_compiler/ops/alu_ops.py:238` |
| L12.post_ops[0] | `FlattenedALUMul` | `make_l12_alu_postop_attach_op` | `unified_compiler/ops/alu_ops.py:244` |
| L13.post_ops[0] | `ALUShiftComposite` | `make_l13_alu_postop_attach_op` | `unified_compiler/ops/alu_ops.py:250` |
| **L17.ffn** | **`l10_post_ops_combined` (PureFFN, 1846 units)** | **`make_l10_post_ops_combined`** | **`unified_compiler/ops/l10_ops.py:1289`** |
| **L17.post_ops[0]** | **`tail_bit32_result_correction` (PureFFN, 2091 units)** | **`make_tail_bit32_result_correction_op`** | **`unified_compiler/ops/l10_ops.py:4529`** |

In efficient ALU mode, `ComparisonCombine` (`vm_step.py`) is additionally
appended to L10.post_ops.

---

## Section 3: "Hidden" FFNs and per-op test coverage

A "hidden" FFN here means an FFN block that runs at inference time but is not
on a layer that obviously owns it. The major case B6-C found
(`tail_bit32_result_correction` at "block27/layer27") is reproduced by the
layout: the rule lives in `l10_ops.py` but is dependency-placed at L17.

### 3a. Dedicated per-op test coverage (excluding allowlist audit `test_runtime_vanilla.py`)

| Module / op | Dedicated test files | Status |
|-------------|----------------------|--------|
| `BinaryOpByteZeroingPostOp` | `test_add_sub_byte_postops.py` (1 file) | partial; ADD/SUB-focused |
| `CarryPropagationPostOp` | `test_add_sub_byte_postops.py`, `test_neural_symbolic_bounds.py`, `test_pure_neural_multibyte.py`, `test_smoke_pure_neural.py`, `test_l10_tail_correction.py` (5 files) | **well covered** |
| `BitwiseBytePropagationPostOp` | `test_l10_tail_correction.py` (1 file, 3 tests) | thin; only tests the post-op as it sits in L10 |
| `AddSubBytePropagationPostOp` | `test_add_sub_byte_postops.py`, `test_l8_l9_pc_marker_blockers.py`, `test_neural_symbolic_bounds.py` (3 files) | reasonable |
| `ComparisonCombine` | `test_full_model_add_trace.py` (1 file; incidental usage) | **uncovered** — no per-op harness |
| `FlattenedDivMod` | NONE | **uncovered** |
| `AddSub5StageBlock` | `test_l8_addsub_stage_ownership.py`, `test_smoke_pure_neural.py` (2 files) | partial |
| `ALUAndOrXor` | NONE | **uncovered** |
| `FlattenedALUMul` | NONE | **uncovered** |
| `ALUShiftComposite` | NONE | **uncovered** |
| `l10_post_ops_combined` (the L17 FFN body) | `test_declarative_bake_gate.py`, `test_declarative_verification.py`, `test_l10_post_op_attach.py`, `test_l10_tail_correction.py` (4 files) | **adequate**, but most are at the rule-blocker level |
| `tail_bit32_result_correction` (the L17 post-op) | `test_compiler_ir.py`, `test_symbolic_byte_signature.py`, `test_l10_tail_correction.py` (3 files; ~5 named tests in `test_l10_tail_correction.py`) | partial — tests focus on `tail_mem_store*` & `tail_bp_byte2*`; SP, PC, AX-ADD, CMP families have minimal direct coverage |

### 3b. Hidden FFNs ranked by exposure

1. **`tail_bit32_result_correction` at L17.post_ops[0]** — what B6-C identified.
   77 rule names; **dependency-placed at L17 because it `target_op_name`s
   `l10_post_ops_combined` which itself was dep-placed at L17**. Any investigator
   probing "block31 / block30 / L17" should look at `l10_ops.py:4529` and
   `l10_ops.py:1289` respectively.
2. **`l10_post_ops_combined` at L17.ffn** — sibling of the above. Combines
   four `vm_step.py` post-op classes into one FFN, then **zeros the three
   `CarryPropagationPostOp` slices** post-bake. So the L17 ffn really only
   runs `BinaryOpByteZeroing` + `ComparisonCombine` worth of units. The
   dead-unit count (1846 - 1562 = 284) is dominated by these zeroed slices.
3. **`FlattenedDivMod` at L10.post_ops[7]** (becomes block 20 after expansion)
   — composite of 3 sub-FFNs (BD→GE long-div pipeline → GE→BD). No dedicated
   tests. Class lives in `c4_release/neural_vm/efficient_alu_divmod_split.py`.
4. **`ALUAndOrXor` at L10.post_ops[0]** (becomes block 13). Lookup-mode-only
   structural ALU. No dedicated tests.
5. **`FlattenedALUMul` at L11/L12.post_ops[0]** (becomes blocks 22 and 24).
   Composite of 9 sub-stages (BD→GE, schoolbook, 3x carry pass, gen/prop,
   binary lookahead, final correction, GE→BD). No dedicated tests.
6. **`ALUShiftComposite` at L13.post_ops[0]** (becomes block 26). Composite of
   3 stages (precompute, select, getobd). No dedicated tests.
7. **L10's 6-deep PureFFN-subclass stack** (BinaryOpByteZeroing → AddSubByteProp
   → 3x CarryProp → BitwiseByteProp) — partial dedicated coverage; the
   structural blockers each gained for compact-layout safety
   (`_strengthen_l10_*` helpers, `_suppress_ffn_on_step_boundary`) are not
   directly tested as adjustments.

The pattern: every wide-ALU composite (DivMod, AndOrXor, ALUMul x2, ShiftComposite)
runs as a post-op but has zero dedicated per-op harness; testing happens via
end-to-end smoke. This means a regression in any of these composites only
shows up as a smoke failure, never with a clean diagnostic.

---

## Section 4: Recommendations

### 4a. Per-op test harnesses urgently needed

In priority order:

1. **`tail_bit32_result_correction` sub-family harnesses** — split
   `test_l10_tail_correction.py` extension by rule family (SP, PC, MEM-addr,
   AX-ADD, CMP). Each harness should drive the FFN with one synthetic residual
   stream per rule and assert the writes. B6-C's overstrengthening of
   `tail_mem_store_addr0_f8` could have been caught by the MEM-addr harness.
2. **`l10_post_ops_combined` standalone harness** — verify the zero-out of
   the carry slices is preserved on every modification (the bake currently
   relies on `if carry_end > carry_start:` and is silently broken if anyone
   reorders).
3. **`FlattenedDivMod` per-stage harness** — each of BD→GE, longdiv, GE→BD
   sub-FFNs. Currently runtime-vanilla only checks the wrapper type.
4. **`FlattenedALUMul` per-stage harness** — 9 stages, no per-stage tests.
   Each sub-FFN is independently bakeable (per `make_l11_alu_mul_*_op`).
5. **`ALUShiftComposite` per-stage harness** — same idea, 3 stages.
6. **`ALUAndOrXor` harness** — single module, no tests.
7. **`ComparisonCombine` harness** — used in both L10 (efficient) and L17
   (as a slice in `l10_post_ops_combined`); regressions show up only via
   `test_full_model_add_trace.py`.
8. **`BitwiseBytePropagationPostOp` harness** — only 3 indirect tests in
   `test_l10_tail_correction.py`; covers AX-byte path only.

### 4b. Over-strength offenders worth auditing

Concentrations of extreme-magnitude weights in this inventory:

- `_tail_bit32_result_correction_rules` uses many `-1_000_000_000.0` blockers
  (e.g. `MARK_*` rows at `l10_ops.py:2341-2346`, `2854-2858`, `4368-4373`)
  and writes with `strength=1_000_000.0` (e.g. `tail_ax_add_byte1_no_carry_low1_02`
  at `l10_ops.py:2640, 2649`) up to `strength=100_000_000.0` for
  `tail_clear_output_{after_byte3,before_step_end}` (`l10_ops.py:4501, 4521`).
  These compound under residual amplification — exactly the symptom B4-B was
  chasing.
- `make_l10_post_ops_combined`'s post-bake guards write `-S * 10^6` blockers
  across all units (`l10_ops.py:1361, 1365, 1374`). The combined effect is
  applied to ~1846 units, so a single mis-targeted row can fan out widely.
- `_strengthen_l10_addsub_wrong_byte_blockers`, `_strengthen_l10_carry_wrong_byte_blockers`
  write `-S * 100000` blockers per `wrong_dim` (`l10_ops.py:1448, 1465`).
- L17's `tail_ax_add_byte1_missing_stack_high_02` has `("FETCH_HI+1", 100000.0)`
  as an evidence weight (`l10_ops.py:4362`) — order-100k positive evidence is
  unusual for an evidence term.

### 4c. Lowering-audit cross-reference

Per the recent commit history on `speedup-cache-and-buckets`
(`72269f6 Fix lowering tail and freshness regressions`, `c300d5a Stabilize
declarative lowering guards`, `4d069f7 Validate bounded batched KV eviction`),
the L17 post-op family is actively being tightened. The structural facts in
this inventory should inform which families remain over-strength:

- `tail_mem_store_addr0_*` (13 rules) — B6-C just touched `f8` variant; the
  other 12 mem-addr rules have similar amplification risk.
- `tail_ax_add_*` (10 rules) — interacts with L8/L9 ADD post-ops; potential
  for double-application not currently checked.
- `tail_cmp_*` (6 rules) — only marker-only writes, low blast radius, but
  duplicated with `ComparisonCombine` slice in `l10_post_ops_combined`.

---

## Summary

- L17 in the layout owns **2 FFNs**, both authored in `l10_ops.py`:
  `l10_post_ops_combined` (the block.ffn) and `tail_bit32_result_correction`
  (block.post_ops[0]). The "block27/layer27" B6-C found is the
  `tail_bit32_result_correction` op materialized at expanded block 31.
- The full model has **14 post-op blocks total** across L8/L9/L10/L11/L12/L13/L17;
  L10 alone owns **8 post-ops** stacked sequentially.
- **7 of 11 distinct post-op modules have NO dedicated per-op test harness**:
  `ComparisonCombine`, `FlattenedDivMod`, `ALUAndOrXor`, `FlattenedALUMul`,
  `ALUShiftComposite` (the wide-ALU composites), plus the rule-level
  `tail_bit32_result_correction` sub-families.
- Recommended new harnesses: **5 wide-ALU composite harnesses + 1 standalone
  `l10_post_ops_combined` harness + >=5 `tail_*` sub-family harnesses** = 11
  new test files.
