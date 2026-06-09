# UNDECLARED DIM AUDIT — 2026-06-09

Static structural-correctness sweep of every op in the compiled VM
layout. For each op, walks its `compiler_ir` (FFN rules + attention
head specs) and compares the actual dim reads/writes against the
op's declared `reads={}` / `writes={}` sets.

Method: see `_audit_undeclared.py` (uses
`c4_release.neural_vm.unified_compiler.dim_flow` —
`_walk_ops_with_layers`, `_materialize_op_ir`, `_ffn_rules_from_ir`,
`_attention_heads_from_ir`, `_dim_int_to_name`). Position-alias
groups (e.g. `OUTPUT_HI`/`OUTPUT_HI_THIS_STEP` at slot 85) are
canonicalized before comparison; declared names with offset or
step-alias suffixes (`+N`, `.*.-1`) are stripped to the base dim.

## Summary

- **Total ops audited**: 162
  - layer ops (ffn/attn): 40 (audit covers all)
  - block ops: 92 (audit covers all)
  - model ops: 30 (audit covers all)
- **Ops with undeclared reads**: 29 (222 dim names)
- **Ops with undeclared writes**: 13 (43 dim names)
- By op kind:
  - ffn/attn (layer ops): 5 of 40 with mismatches
  - block ops: 20 of 92
  - model ops: 4 of 30

The L14 `mem_generation` case noted in the dim_flow module docstring
(`getattr(BD, ...)` without declaring in `reads`) is one of many —
the audit finds 29 ops total with at least one undeclared read.

## Top-priority findings (undeclared writes first)

Undeclared **writes** are the dangerous class: they break the dep
graph's writer-set view, can produce unannounced cross-layer
conflicts, and let aggressor writes evade reviewer scrutiny.

| Layer | Op | Kind | UW | Undeclared writes |
|------:|----|:----:|---:|-------------------|
| 6 | `layer6_routing_ffn` | block | 10 | `ADDR_B0_LO, ADDR_B1_LO, ALU_HI, ALU_LO, CMP, MEM_ADDR_SRC, MEM_STORE, NEXT_HALT, NEXT_SE, TEMP` |
| 9 | `layer8_alu` | block | 8 | `ADDR_B0_HI, ADDR_B0_LO, ADDR_B1_LO, ADDR_B2_LO, ALU_HI, ALU_LO, CMP, OUTPUT_HI` |
| -1 | `function_call_weights` | model | 5 | `ALU_HI, ALU_LO, OUTPUT_HI, OUTPUT_LO, TEMP` |
| 6 | `binary_pop_sp_increment` | model | 4 | `CLEAN_EMBED_HI, CLEAN_EMBED_LO, OUTPUT_HI, OUTPUT_LO` |
| 20 | `layer16_lev_routing` | ffn | 1 | `AX_CARRY_LO` |
| 13 | `layer13_mem_addr_gather` | block | 1 | `H5` |
| -1 | `io_putchar_routing` | model | 3 | `IO_IS_PUTCHAR, OUTPUT_HI, OUTPUT_LO` |
| 6 | `convo_io_pc_sp_latch` | block | 2 | `OUTPUT_HI, OUTPUT_LO` |
| 2 | `layer2_mem_byte_flags` | ffn | 1 | `BYTE_INDEX_0` |
| 6 | `convo_io_state_machine` | block | 3 | `IO_STATE, NEXT_SE, NEXT_THINKING_END` |
| 3 | `convo_io_step_resume` | block | 3 | `IO_IN_OUTPUT_MODE, IO_STATE, NEXT_PC` |
| 3 | `layer3_convo_io_state_init` | block | 1 | `IO_IN_OUTPUT_MODE` |
| 4 | `layer4_ffn` | block | 1 | `TEMP` |

## All findings — full table

| Layer | Op | Kind | UR | UW |
|------:|----|:----:|---:|---:|
| 6 | `layer6_routing_ffn` | block | 24 | 10 |
| 9 | `layer8_alu` | block | 16 | 8 |
| 25 | `tail_bit32_result_correction` | block | 24 | 0 |
| -1 | `function_call_weights` | model | 17 | 5 |
| 14 | `layer10_alu` | block | 21 | 0 |
| 6 | `binary_pop_sp_increment` | model | 16 | 4 |
| -1 | `layer6_relay_heads_bake` | model | 16 | 0 |
| 18 | `layer14_mem_generation` | attn | 15 | 0 |
| 20 | `layer16_lev_routing` | ffn | 14 | 1 |
| 13 | `layer13_mem_addr_gather` | block | 10 | 1 |
| -1 | `io_putchar_routing` | model | 4 | 3 |
| 6 | `convo_io_pc_sp_latch` | block | 5 | 2 |
| 6 | `layer6_ent_after_jsr_sp_byte0_fixup` | block | 7 | 0 |
| 0 | `phase_a_ffn` | block | 5 | 0 |
| 2 | `layer2_mem_byte_flags` | ffn | 4 | 1 |
| 6 | `convo_io_state_machine` | block | 2 | 3 |
| 1 | `layer1_ffn` | ffn | 4 | 0 |
| 3 | `convo_io_step_resume` | block | 1 | 3 |
| 11 | `layer10_psh_ax_broadcast_bake` | block | 3 | 0 |
| 19 | `layer15_alu_high_byte_relay` | block | 3 | 0 |
| 3 | `layer3_convo_io_state_init` | block | 1 | 1 |
| 3 | `layer3_ffn` | block | 2 | 0 |
| 4 | `layer4_ffn` | block | 1 | 1 |
| 8 | `layer7_memory_heads` | block | 2 | 0 |
| 3 | `layer3_carry_forward_attn` | attn | 1 | 0 |
| 5 | `layer5_fetch` | block | 1 | 0 |
| 5 | `opcode_decode_ffn` | block | 1 | 0 |
| 11 | `layer10_psh_stack0_passthrough_bake` | block | 1 | 0 |
| 19 | `layer15_si_mem_addr0_from_stack0` | block | 1 | 0 |

## Per-op detail

### L6 `layer6_routing_ffn` (block)

- **Undeclared reads** (24): `ALU_HI, ALU_LO, BYTE_INDEX_0, BYTE_INDEX_1, BYTE_INDEX_2, CONST, EMBED_HI, EMBED_LO, H1, MARK_MEM, MARK_SP, MEM_ADDR_SRC, MEM_STORE, NEXT_SE, OPCODE_BYTE_HI, OPCODE_BYTE_LO, OP_ADJ, OP_BNZ, OP_BZ, OP_ENT, OP_GETCHAR, OP_JSR, OP_PSH, PSH_AT_SP`
- **Undeclared writes** (10): `ADDR_B0_LO, ADDR_B1_LO, ALU_HI, ALU_LO, CMP, MEM_ADDR_SRC, MEM_STORE, NEXT_HALT, NEXT_SE, TEMP`

### L9 `layer8_alu` (block)

- **Undeclared reads** (16): `BYTE_INDEX_1, CMP, CONST, H1, HAS_SE, IS_BYTE, MARK_BP, MARK_MEM, MARK_SE, MARK_SP, MARK_STACK0, OP_ADJ, OP_ENT, OP_LEV, OUTPUT_HI, OUTPUT_LO`
- **Undeclared writes** (8): `ADDR_B0_HI, ADDR_B0_LO, ADDR_B1_LO, ADDR_B2_LO, ALU_HI, ALU_LO, CMP, OUTPUT_HI`

### L25 `tail_bit32_result_correction` (block)

- **Undeclared reads** (24): `ADDR_B1_HI, ADDR_B1_LO, ADDR_B2_HI, ADDR_B2_LO, AX_CARRY_HI, CMP_GROUP, FETCH_LO, IN_STEP_FRESH, MEM_ADDR_SRC, OP_ADJ, OP_BNZ, OP_BZ, OP_EXIT, OP_GETCHAR, OP_JMP, OP_LEA, OP_MUL, OP_NOP, OP_PSH, OP_PUTCHAR, STACK0_BYTE0, STACK0_BYTE1, STACK0_BYTE2, STACK0_BYTE3`

### L-1 `function_call_weights` (model)

- **Undeclared reads** (17): `AX_CARRY_HI, AX_CARRY_LO, CMP, FETCH_HI, FETCH_LO, H1, HAS_SE, IS_BYTE, OP_BNZ, OP_BZ, OP_EXIT, OP_IMM, OP_JMP, OP_LEA, OP_LEV, OP_NOP, TEMP`
- **Undeclared writes** (5): `ALU_HI, ALU_LO, OUTPUT_HI, OUTPUT_LO, TEMP`

### L14 `layer10_alu` (block)

- **Undeclared reads** (21): `CMP, MARK_PC, OP_ADD, OP_EQ, OP_EXIT, OP_GE, OP_GT, OP_IMM, OP_JMP, OP_LC, OP_LE, OP_LEA, OP_LI, OP_LT, OP_MUL, OP_NE, OP_NOP, OP_PUTCHAR, OP_SHL, OP_SHR, OP_SUB`

### L6 `binary_pop_sp_increment` (model)

- **Undeclared reads** (16): `BYTE_INDEX_0, BYTE_INDEX_1, CLEAN_EMBED_HI, CLEAN_EMBED_LO, CMP, CONST, EMBED_HI, EMBED_LO, H1, IS_BYTE, MARK_AX, MARK_BP, MARK_MEM, MARK_PC, MARK_SP, MARK_STACK0`
- **Undeclared writes** (4): `CLEAN_EMBED_HI, CLEAN_EMBED_LO, OUTPUT_HI, OUTPUT_LO`

### L-1 `layer6_relay_heads_bake` (model)

- **Undeclared reads** (16): `OP_ADD, OP_AND, OP_DIV, OP_EQ, OP_GE, OP_GT, OP_LE, OP_LT, OP_MOD, OP_MUL, OP_NE, OP_OR, OP_SHL, OP_SHR, OP_SUB, OP_XOR`

### L18 `layer14_mem_generation` (attn)

- **Undeclared reads** (15): `BP_VIA_LEV_DETECTOR, CLEAN_EMBED_HI, CLEAN_EMBED_LO, CONST, FORMAT_PTR_HI, FORMAT_PTR_LO, L2H0, MARK_AX, MARK_BP, MARK_PC, OUTPUT_HI, OUTPUT_LO, SP_VIA_LEV_DETECTOR, STACK0_BYTE_VAL_2_LO, STACK0_BYTE_VAL_3_HI`

### L20 `layer16_lev_routing` (ffn)

- **Undeclared reads** (14): `CONST, MARK_MEM, MARK_SE, NEXT_AX, NEXT_BP, NEXT_MEM, NEXT_SE, NEXT_SP, NEXT_STACK0, OP_JSR, OP_LEA, OP_PSH, OUTPUT_HI, OUTPUT_LO`
- **Undeclared writes** (1): `AX_CARRY_LO`

### L13 `layer13_mem_addr_gather` (block)

- **Undeclared reads** (10): `CLEAN_EMBED_HI, CLEAN_EMBED_LO, CONST, H0, L1H0, L1H2, MEM_VAL_B0, MEM_VAL_B1, MEM_VAL_B2, MEM_VAL_B3`
- **Undeclared writes** (1): `H5`

### L-1 `io_putchar_routing` (model)

- **Undeclared reads** (4): `AX_CARRY_HI, AX_CARRY_LO, MARK_AX, OP_PUTCHAR`
- **Undeclared writes** (3): `IO_IS_PUTCHAR, OUTPUT_HI, OUTPUT_LO`

### L6 `convo_io_pc_sp_latch` (block)

- **Undeclared reads** (5): `IN_STEP_FRESH, POST_PRTF_PC_HI, POST_PRTF_PC_LO, POST_PRTF_SP_HI, POST_PRTF_SP_LO`
- **Undeclared writes** (2): `OUTPUT_HI, OUTPUT_LO`

### L6 `layer6_ent_after_jsr_sp_byte0_fixup` (block)

- **Undeclared reads** (7): `BYTE_INDEX_0, BYTE_INDEX_1, BYTE_INDEX_2, H1, IS_BYTE, MARK_BP, MARK_STACK0`

### L0 `phase_a_ffn` (block)

- **Undeclared reads** (5): `H0, H1, H2, H3, H4`

### L2 `layer2_mem_byte_flags` (ffn)

- **Undeclared reads** (4): `H2, H3, L1H4, L2H0`
- **Undeclared writes** (1): `BYTE_INDEX_0`

### L6 `convo_io_state_machine` (block)

- **Undeclared reads** (2): `CMP, NEXT_SE`
- **Undeclared writes** (3): `IO_STATE, NEXT_SE, NEXT_THINKING_END`

### L1 `layer1_ffn` (ffn)

- **Undeclared reads** (4): `L1H0, L1H1, L1H2, L1H4`

### L3 `convo_io_step_resume` (block)

- **Undeclared reads** (1): `IN_STEP_FRESH`
- **Undeclared writes** (3): `IO_IN_OUTPUT_MODE, IO_STATE, NEXT_PC`

### L11 `layer10_psh_ax_broadcast_bake` (block)

- **Undeclared reads** (3): `STACK0_BYTE1, STACK0_BYTE2, STACK0_BYTE3`

### L19 `layer15_alu_high_byte_relay` (block)

- **Undeclared reads** (3): `BYTE_INDEX_1, BYTE_INDEX_2, BYTE_INDEX_3`

### L3 `layer3_convo_io_state_init` (block)

- **Undeclared reads** (1): `LAST_WAS_THINKING_END`
- **Undeclared writes** (1): `IO_IN_OUTPUT_MODE`

### L3 `layer3_ffn` (block)

- **Undeclared reads** (2): `H0, MARK_MEM`

### L4 `layer4_ffn` (block)

- **Undeclared reads** (1): `TEMP`
- **Undeclared writes** (1): `TEMP`

### L8 `layer7_memory_heads` (block)

- **Undeclared reads** (2): `CLEAN_EMBED_HI, CLEAN_EMBED_LO`

### L3 `layer3_carry_forward_attn` (attn)

- **Undeclared reads** (1): `ADDR_KEY`

### L5 `layer5_fetch` (block)

- **Undeclared reads** (1): `TEMP`

### L5 `opcode_decode_ffn` (block)

- **Undeclared reads** (1): `TEMP`

### L11 `layer10_psh_stack0_passthrough_bake` (block)

- **Undeclared reads** (1): `CONST`

### L19 `layer15_si_mem_addr0_from_stack0` (block)

- **Undeclared reads** (1): `IS_BYTE`

## Recommended next contracts

1. **Undeclared writes are the immediate priority** — they violate
   the dep-graph contract: `compiler.add_op` uses `writes` to
   compute scheduling edges. Hidden writes can silently corrupt a
   downstream reader's slot. Top candidates for an `Operation.writes`
   amendment:
   - `layer6_routing_ffn` (10 undeclared writes — biggest)
   - `layer8_alu` (8 undeclared writes)
   - `function_call_weights` (5 undeclared writes, model-scope)
   - `binary_pop_sp_increment` (4 undeclared writes)
   - `convo_io_state_machine` (3 undeclared writes)

2. **Undeclared reads** are mostly safe (don't shift edges) but
   undermine the IR-as-contract goal. The L14 `mem_generation` case
   (15 undeclared reads incl. `CLEAN_EMBED_HI/LO`, `OUTPUT_HI/LO`)
   and `layer6_routing_ffn` (24 undeclared reads) are the loudest;
   `layer10_alu` (21), `tail_bit32_result_correction` (24),
   `function_call_weights` (17), `layer6_relay_heads_bake` (16),
   `binary_pop_sp_increment` (16), `layer8_alu` (16) all need
   sweep passes.

3. **Audit infrastructure** — keep `_audit_undeclared.py` as a
   regression gate; rerun on every Phase 7 op landing and ratchet
   the finding-count down. The audit is O(ops × ir-size), runs in
   <5s, no model bake.

4. **Alias canonicalization** is real signal: 14 of the 43
   original findings turned out to be `OUTPUT_HI` ↔
   `OUTPUT_HI_THIS_STEP` (pos 85) noise. Future declared sets
   should pick one canonical name per slot to stay clean.

## Migration path: derive_reads_writes

The structural fix is to *derive* `Operation.reads` / `Operation.writes`
from the rules instead of hand-annotating. Plumbing landed in commit
`05ad23a1`:

- `neural_vm/unified_compiler/op_introspect.py` —
  `derive_op_reads_writes_from_rules(op, dim_positions, dim_sizes)`,
  `assert_declared_matches_derived(op, ..., strict=False)`,
  `derive_operation(op, ...)`.
- `Operation.derive_reads_writes()` method on
  `unified_compiler/layer_compiler.py:Operation` — returns a copy with
  the derived sets.
- `Operation.derive_reads_writes_flag` — opt-in marker (default `False`)
  for per-op migration.
- `tests/test_op_reads_writes_derivation.py` — 20 tests pin derivation
  semantics, including parity checks against the audit findings for
  `layer2_mem_byte_flags`, `layer1_ffn`, and `phase_a_ffn`.
- `tools/lint_op_reads_writes.py` — per-op ratchet. Default mode is
  advisory (`declared >= derived`); `--strict` requires equality. The
  baseline freezes today's mismatch counts; migrations decrement
  baseline entries in the same commit.

### Per-op migration recipe

For each op listed in the audit table above:

1. Call `op.derive_reads_writes(dim_positions=..., dim_sizes=...)` in
   the op-factory to verify the derivation matches the intent. (Pure-FFN
   ops can omit the maps; attention-bearing ops need them.)
2. Either:
   - **Path A (drop the annotation)**: set `derive_reads_writes_flag=True`
     and remove the hand-written `reads={}` / `writes={}` sets. The
     compiler will pick up the derived sets at op-registration time.
     *(Wave 2: requires `compiler.add_op` to honor the flag — not yet
     implemented. See below.)*
   - **Path B (close the gap)**: extend the hand-written sets to match
     the derivation, then drop the op's baseline entry in
     `lint_op_reads_writes.py`. This is the conservative path used for
     `layer2_mem_byte_flags` writes in commit `d756d9d8`.
3. Run `python c4_release/tools/lint_op_reads_writes.py` to confirm the
   baseline ratchet still holds.
4. For Path A only: run
   `python c4_release/tools/lint_op_reads_writes.py --strict` to
   confirm the derivation produces no over-declarations.

### Wave 2 (future) — compiler honors the flag

The `derive_reads_writes_flag` is currently inert — the compiler still
reads `op.reads` / `op.writes` directly. Wave 2 will:

1. Add a `compiler.add_op` shim that, when `op.derive_reads_writes_flag`
   is True, runs `op.derive_reads_writes(...)` against the compiler's
   `dim_positions` and substitutes the result before scheduling.
2. Migrate the audit table's top-write offenders first
   (`layer6_routing_ffn`, `layer8_alu`, `function_call_weights`,
   `binary_pop_sp_increment`) since undeclared writes break the dep
   graph's scheduling edges, not just the IR-as-contract claim.
3. Land the `--strict` mode as a CI gate once the per-op baseline
   reaches zero.

### Priority order

Same as the "Recommended next contracts" section above — writes first,
then reads, in audit-count order. The lint baseline is the live ratchet.
