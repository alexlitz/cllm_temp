# BD Dim Usage Map (d_model=512)

**Base:** `speedup-cache-and-buckets @ 4d069f7`
**Authored:** B6-K (B6 batch diagnostic)

## Important correction to the task brief

There is **no `dim_positions.py` file** in `c4_release/neural_vm/unified_compiler/`. The canonical dim allocation lives in:

- `c4_release/neural_vm/vm_step.py:2215-2515` — `_SetDim` class (ground truth)
- `c4_release/neural_vm/dim_registry.py:277-383` — `build_default_registry()` (STALE vs `_SetDim`)
- `c4_release/neural_vm/unified_compiler/ops/shared.py:680-704` — compiler dim size declarations
- `c4_release/neural_vm/unified_compiler/layer_compiler.py:1189` — `_allocate_dims()` bump-pointer logic

The `dim_positions` *dict* threaded through bake functions is built at runtime.

## Section 1: Allocated dim ranges (canonical `_SetDim`)

| Range | Slot(s) | Count | Notes |
|-------|---------|-------|-------|
| 0-11 | MARK_PC, MARK_AX, MARK_SP, MARK_BP, MARK_MEM, MARK_SE, IS_BYTE, IS_MARK, CONST, MARK_CS, MARK_SE_ONLY, MARK_STACK0 | 12 | embedding-set markers |
| 12-27 | ADDR_B0_LO [ALIAS OPCODE_BYTE_LO] | 16 | one-hot |
| 28-43 | ADDR_B1_LO [ALIAS OPCODE_BYTE_HI] | 16 | one-hot |
| 44-59 | ADDR_B2_LO | 16 | one-hot |
| 60-66 | H0 | 7 | L0 head 0, threshold 3.5 |
| 67-73 | H1 | 7 | L0 head 1, threshold 4.5 |
| 74-80 | H2 | 7 | L0 head 2, threshold 7.5 |
| 81-87 | H3 | 7 | L0 head 3, threshold 8.5 |
| 88-94 | H4 | 7 | L0 head 4, threshold 9.5 |
| **95-101** | **H5 (DEAD)** | **7** | threshold 14.5, never read |
| **102-108** | **H6 (DEAD)** | **7** | threshold 19.5, never read |
| **109-115** | **H7 (DEAD)** | **7** | threshold 24.5, never read |
| 116-122 | L1H0 | 7 | L1 head 0, threshold 0.5 |
| 123-129 | L1H1 | 7 | L1 head 1, threshold 1.5 |
| 130-136 | L1H2 | 7 | L1 head 2, threshold 2.5 |
| 137 | HAS_SE | 1 | global STEP_END flag |
| 138-141 | BYTE_INDEX_0..3 | 4 | byte index within 4-byte register |
| 142-157 | EMBED_LO | 16 | embedding input lo nibble |
| 158-173 | EMBED_HI | 16 | embedding input hi nibble |
| 174-189 | OUTPUT_LO | 16 | output lo nibble |
| 190-205 | OUTPUT_HI | 16 | output hi nibble |
| 206-253 | ADDR_KEY [ALIAS ADDR_B0_HI@206-221, ADDR_B1_HI@222-237, ADDR_B2_HI@238-253] | 48 | memory address key |
| 254-261 | NEXT_PC/AX/SP/BP/STACK0/MEM/SE/HALT | 8 | transition flags |
| 262-295 | OPCODE_FLAGS (OP_LEA..OP_GETCHAR) | 34 | one-hot opcode |
| 296 | IO_IS_PUTCHAR | 1 | |
| 297-303 | L1H4 | 7 | threshold 6.5 |
| 304 | STACK0_BYTE0 | 1 | |
| 305 | CMP_GROUP | 1 | any comparison-opcode active at AX |
| 306-321 | CLEAN_EMBED_LO | 16 | pristine lo nibble |
| 322-327 | IO_IS_TOOL_CALL, NEXT_TOOL_CALL, NEXT_THINKING_START, NEXT_THINKING_END, NEXT_IO_STATE_EMIT_BYTE, NEXT_IO_STATE_EMIT_THINKING | 6 | conv I/O state |
| 328-343 | AX_CARRY_LO [ALIAS POST_PRTF_SP_LO] | 16 | |
| 344-359 | AX_CARRY_HI [ALIAS POST_PRTF_SP_HI] | 16 | |
| 360-375 | ALU_LO | 16 | |
| 376-391 | ALU_HI | 16 | |
| 392-395 | CARRY | 4 | inter-byte carry |
| 396-403 | CMP | 8 | CMP+0..+7 (registry says 4, wrong) |
| 404-419 | CLEAN_EMBED_HI | 16 | pristine hi nibble |
| 420-435 | MUL_ACCUM [ALIAS FETCH_LO] | 16 | mode-disjoint |
| 436-451 | DIV_STAGING [ALIAS FETCH_HI] | 16 | mode-disjoint |
| 452-458 | L2H0 | 7 | threshold 5.5 |
| 459-468 | MEM_STORE, MEM_ADDR_SRC, MEM_VAL_B0..B3, OP_LI_RELAY, OP_LC_RELAY, PSH_AT_SP, MEM_EXEC + IO aliases | 10 | normal-VM vs conv-I/O mode aliases |
| 469-470 | IO_IN_OUTPUT_MODE, IO_OUTPUT_COMPLETE | 2 | |
| 471-486 | AX_FULL_LO [ALIAS FORMAT_PTR_LO, POST_PRTF_PC_LO] | 16 | 3-way alias |
| 487-502 | AX_FULL_HI [ALIAS FORMAT_PTR_HI, POST_PRTF_PC_HI] | 16 | 3-way alias |
| 480-511 | TEMP (overlaps OUTPUT_BYTE_LO@480-495 + OUTPUT_BYTE_HI@496-511) | 32 | heavy overlay |
| 501-510 | LAST_WAS_THINKING_END/START, LAST_WAS_BYTE, ACTIVE_OPCODE_PRTF/READ, MARK_THINKING_START/END, STACK0_BYTE1/2/3 | 10 | overlaid on TEMP and OUTPUT_BYTE_HI |

## Section 2: Per-layer reads/writes (abridged)

Full table in B6-G's `investigation/l7-l9-structural-audit:.agent-logs/l7-l9-structural-audit/REPORT.md`.

## Section 3: Available slots — KEY FINDING

**Counting whole dims**: all 512 dims have at least one slot. But counting DEAD writes:

| Dim | Producer | Read by anyone? |
|-----|----------|-----------------|
| H5 (95-101, 7 dims) | layer0_threshold_attn | **NO — dead** |
| H6 (102-108, 7 dims) | layer0_threshold_attn | **NO — dead** |
| H7 (109-115, 7 dims) | layer0_threshold_attn | **NO — dead** |
| CMP_GROUP (305) | layer8_alu | Yes (compiler.py:2657-2689) |
| IO_STATE (466) | putchar_think_protocol | Yes (flag_gated_ops.py:589) |
| NEXT_THINKING_END (325) | putchar_think_protocol | Yes (output head) |

**Reclaimable: 21 contiguous dims at 95-115** (H5/H6/H7 thresholds 14.5/19.5/24.5 are written but never read).

## Section 4: Aliasing risks

**HIGH-risk overlay zones (4+ slot families per dim):**
- 480-486: AX_FULL_LO + FORMAT_PTR_LO + OUTPUT_BYTE_LO + POST_PRTF_PC_LO + TEMP
- 487-495: AX_FULL_HI + FORMAT_PTR_HI + OUTPUT_BYTE_LO + POST_PRTF_PC_HI + TEMP
- 496-500: AX_FULL_HI + FORMAT_PTR_HI + OUTPUT_BYTE_HI + POST_PRTF_PC_HI + TEMP
- 501-507: adds LAST_WAS_*, ACTIVE_OPCODE_*, MARK_THINKING_*
- 508-510: OUTPUT_BYTE_HI + STACK0_BYTE1/2/3 + TEMP

All currently safe via mode-mutual-exclusion (conv I/O vs ALU vs PRTF vs STACK0-byte positions) but fragile to new writes.

**MEDIUM-risk pair aliases (mode-disjoint):** 12-27 (ADDR_B0_LO vs OPCODE_BYTE_LO), 28-43 (ADDR_B1_LO vs OPCODE_BYTE_HI), 206-253 (ADDR_KEY vs ADDR_B*_HI), 328-359 (AX_CARRY vs POST_PRTF_SP), 420-451 (MUL_ACCUM/DIV_STAGING vs FETCH).

**Specific risks:** dim 464 (MEM_VAL_B3 vs IO_IS_PRTF), dim 467 (PSH_AT_SP vs IO_OUTPUT_COUNT), dims 508-510 (STACK0_BYTE1/2/3 vs OUTPUT_BYTE_HI+12/13/14).

## Section 5: B7 dim allocation recommendations

| Dim | New slot | Producer | Consumer |
|-----|----------|----------|----------|
| **SP_BYTE0_IS_F8** (B7-2) | **95** | L7: extend `layer7_memory_heads` head 6 (l7_ops.py:191) with V/O slot attending MARK_SP → SP byte-0, gated on CLEAN_EMBED_LO+8 * CLEAN_EMBED_HI+15 | L10 `tail_sp_marker_byte0_f8_from_initial_stack_exact` (l10_ops.py:3465) replaces OUTPUT_LO+8/OUTPUT_HI+15 proxy with `(SP_BYTE0_IS_F8, +1e6)`; drops HAS_SE -1e9 hammer |
| **IN_STEP_FRESH** (B7-1) | **96** | L1: add 9th `layer1_threshold_attn` head (sibling to head 3 HAS_SE, l1_ops.py:108-118) with forward ALiBi attending next MARK_SE_ONLY | L10 tail_* rules use `IN_STEP_FRESH +1e6` positive evidence instead of HAS_SE -1e9 negative |
| **ADDR_B0_VALID** (B7-4) | **97** | L13: extend `layer13_mem_addr_gather` (l13_ops.py:34) with V/O slot firing at MARK_MEM, writes 1.0 | L10 `tail_mem_store_addr0_*` family (l10_ops.py:3450-4100) gates on `ADDR_B0_VALID +50` lifecycle bit per B4-H 3.2 |
| **SP_GATHERED_THIS_STEP** (B7-5) | **98** | L8: extend `layer8_sp_gather_bake` (l8_ops.py:476) so heads 0-2 fire at MARK_SP as well as MARK_STACK0; add sentinel V/O writing 1.0 at MARK_SP | L10 `tail_sp_marker_*` rules gate on `SP_GATHERED_THIS_STEP +50` |

**Reserved**: dims 99-115 (17 dims) — available for full `SP_BYTE0_VALUE` 32-dim one-hot, or for `ADDR_B0_HI_VALID`/`ADDR_B1_VALID`/`ADDR_B2_VALID` lifecycle bits.

**Wiring changes for H5/H6/H7 reclamation:**
1. Drop thresholds 14.5/19.5/24.5 from `l0_ops.py:141` (or rebind W_o rows to new dim names).
2. Update `_OUT_BASES` at `l0_ops.py:161` to drop H5/H6/H7.
3. Update `compiler.py:201` `out_bases` list.
4. Update name refs in `test_opcodes.py:926` and `test_dim_registry.py:233`.
5. Optionally reduce L0 attn to 5 heads (`num_heads=5`) for runtime savings.

For each new dim: add constant in `_SetDim` (vm_step.py:2215), entry in `compiler/shared.py:680-704`, producer op's `writes={...}` + `_claims`, consumer op's `reads={...}`, and optional `produces`/`consumes_fresh` metadata for the staleness scanner.

---

## Summary

- d_model = 512 is NOT truly saturated — there's a 21-dim contiguous block at 95-115 where L0's H5/H6/H7 attention heads write but no downstream consumer reads. Free real estate for B7's new structural dims.
- Recommended allocation: SP_BYTE0_IS_F8 @ 95, IN_STEP_FRESH @ 96, ADDR_B0_VALID @ 97, SP_GATHERED_THIS_STEP @ 98 (uses 4 of the 21 reclaimable dims).
- Reserved 99-115 (17 dims) for B7 follow-on dims.
