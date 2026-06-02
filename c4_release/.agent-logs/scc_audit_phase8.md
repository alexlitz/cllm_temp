# G7 SCC <= 10 Measurement — Phase 8 audit

_Generated 2026-06-02. Read-only investigation._
_HEAD = `2a1fe510` (Merge KV overwrite map from declarative IR, Phase 8.E.2)._

Run with:

    export CUDA_VISIBLE_DEVICES=1
    python tools/analyze_scheduler.py        # produces .agent-logs/scheduler_phase_a_2026_06_02.{md,csv}
    python /tmp/scc_audit.py                  # companion drill-down (this report's source data)

---

## 1. Current SCC size

| Metric | Value | Target |
|--------|------:|-------:|
| Total ops analysed | 122 | — |
| Cycle members (across all SCCs) | **89** | — |
| Number of SCCs | **1** | — |
| Largest SCC size | **69** | **<= 10** |
| Gap to target | **59 ops** | — |

The graph still contains a single giant SCC of **69 ops** spanning layers
3 -> 17. Hitting the G7 `SCC <= 10` budget requires shrinking it by
**~59 ops**. None of the other 20 cycle members live in non-singleton
SCCs at this HEAD — they are in the same component.

This is essentially unchanged from the Phase 7 closing audit
(`.agent-logs/phase_7_closing_audit.md` reported SCC=68); the Phase
8.E.2 KV overwrite map and 8.C imperative_heavy migration touched the
dispatcher / census but did not introduce or retire any cycle dim.

## 2. Top 10 back-edge dims (writes/reads only, current static layout)

Edges where producer is in a strictly later static layer than the
reader inside the largest SCC. These are what `LayerCompiler._topological_sort`
must phase-prune today.

| Rank | Dim | Back-edges | Total edges in SCC |
|-----:|-----|----------:|------------------:|
| 1 | `OUTPUT_LO`             | 82 | 122 |
| 2 | `ADDR_KEY`              | 17 |  33 |
| 3 | `AX_CARRY_HI`           | 15 | 104 |
| 4 | `OUTPUT_HI_THIS_STEP`   | 11 |  44 |
| 5 | `ALU_LO`                |  8 |  40 |
| 6 | `OP_LEV`                |  6 |  12 |
| 7 | `ADDR_B0_LO`            |  4 |  18 |
| 8 | `EMBED_LO`              |  3 |  27 |
| 9 | `CARRY`                 |  3 |  21 |
|10 | `OUTPUT_HI`             |  2 |  10 |
|10 | `IO_IN_OUTPUT_MODE`     |  2 |   3 |
|10 | `ALU_HI`                |  2 |  16 |
|10 | `TEMP`                  |  2 |  40 |

Important: **back-edge count is not the only signal**. `AX_CARRY_HI`
has 15 back-edges but 104 SCC-internal edges total, so a same-step
write/read chain — not the back-edges — is dominating the cycle for
that dim. Likewise `TEMP` and `EMBED_LO` look low in back-edge count
but contribute many same-step edges that bind ops together. Removing
only the back-edges (single-dim removal, BACK only) leaves the SCC
unchanged at 69 for every dim in this list (see appendix); the cycle
is *forward-only* for those dims.

## 3. Per cycle-op: dim responsible for SCC membership

For each of the 69 ops in the largest SCC, the back-edge dim(s) with
the highest in-count (the dim "pulling it into" the cycle from a
later layer) and the highest out-count (the dim "pushing it into"
the cycle by feeding a later layer back through). Entries with `-`
have NO back-edges of that direction within the SCC — they are
dragged in by same-step / forward edges only.

| op | static layer | top inbound back-edge dim | top outbound back-edge dim |
|---|---:|---|---|
| _layer3_ffn_dep_anchor              |  3 | OP_LEV(2), EMBED_LO(1) | - |
| layer3_carry_forward_attn           |  3 | OP_LEV(2), EMBED_LO(1) | - |
| layer3_ffn                          |  3 | OP_LEV(2), EMBED_LO(1) | - |
| layer4_ffn                          |  4 | - | - |
| layer4_pc_relay                     |  4 | ADDR_KEY(3) | EMBED_LO(3) |
| layer4_sp_to_addr_key               |  4 | - | - |
| _layer5_fetch_dep_anchor            |  5 | ADDR_KEY(3) | - |
| _opcode_decode_ffn_dep_anchor       |  5 | - | OP_LEV(3) |
| layer5_fetch                        |  5 | ADDR_KEY(3) | - |
| opcode_decode_ffn                   |  5 | - | OP_LEV(3) |
| layer6_attn                         |  6 | AX_CARRY_HI(3), PSH_AT_SP(1) | - |
| layer6_ent_after_jsr_sp_byte0_fixup |  6 | - | - |
| layer6_relay_heads                  |  6 | AX_CARRY_HI(3) | - |
| layer6_routing_ffn                  |  6 | OUTPUT_LO(31), AX_CARRY_HI(3) | - |
| putchar_think_protocol              |  6 | AX_CARRY_HI(3) | - |
| format_pointer_extraction           |  7 | IO_IN_OUTPUT_MODE(1) | - |
| layer7_memory_heads                 |  7 | AX_CARRY_HI(3) | ADDR_KEY(3), PSH_AT_SP(1) |
| layer7_operand_gather               |  7 | OUTPUT_LO(31), OUTPUT_HI(1) | - |
| layer8_alu                          |  8 | ALU_LO(3) | OUTPUT_LO(2) |
| layer8_head6_ax_carry_refresh       |  8 | - | AX_CARRY_HI(5) |
| layer8_mem_to_alu                   |  8 | ADDR_B0_LO(3), ADDR_B2_HI(1) | - |
| layer8_multibyte_fetch              |  8 | ADDR_KEY(2) | AX_CARRY_HI(5) |
| layer8_multibyte_fetch_bake         |  8 | ADDR_KEY(2) | AX_CARRY_HI(5) |
| layer8_multibyte_routing            |  8 | - | OUTPUT_LO(2) |
| layer8_sp_gather                    |  8 | - | - |
| layer8_sp_gather_bake               |  8 | CMP(1) | - |
| format_string_fetch_head            |  9 | ADDR_KEY(2), IO_IN_OUTPUT_MODE(1) | OUTPUT_BYTE_LO(1) |
| layer9_alibi_mem_attn               |  9 | ADDR_KEY(2) | OUTPUT_LO(2) |
| layer9_alu                          |  9 | CARRY(3), ALU_HI(2) | OUTPUT_LO(2), CMP(1) |
| layer9_lev_addr_relay               |  9 | - | ADDR_B0_LO(1) |
| layer9_lev_bp_to_pc_relay           |  9 | - | ADDR_B0_LO(1) |
| layer9_marker_suppress              |  9 | - | OUTPUT_LO(2) |
| l10_post_ops_combined               | 10 | OUTPUT_HI_THIS_STEP(10), OUTPUT_LO(5), ALU_LO(1) | OUTPUT_LO(2), CARRY(1) |
| layer10_alu                         | 10 | ALU_LO(1) | OUTPUT_LO(2) |
| layer10_bp_byte_passthrough_bake    | 10 | - | OUTPUT_LO(2) |
| layer10_byte_passthrough            | 10 | - | OUTPUT_LO(2) |
| layer10_byte_passthrough_bake       | 10 | - | OUTPUT_LO(2) |
| layer10_carry_relay                 | 10 | - | CARRY(1) |
| layer10_carry_relay_bake            | 10 | - | CARRY(1) |
| layer10_psh_stack0_passthrough      | 10 | - | OUTPUT_LO(2) |
| layer10_psh_stack0_passthrough_bake | 10 | OUTPUT_LO(15), OUTPUT_HI(1) | OUTPUT_LO(2) |
| layer10_sp_byte_passthrough         | 10 | - | OUTPUT_LO(2) |
| layer10_sp_byte_passthrough_bake    | 10 | - | OUTPUT_LO(2) |
| layer10_stack0_byte_relay           | 10 | - | OUTPUT_LO(2), ALU_LO(1) |
| layer10_stack0_byte_relay_bake      | 10 | - | OUTPUT_LO(2), ALU_LO(1) |
| null_terminator_detection           | 10 | - | IO_IN_OUTPUT_MODE(2) |
| layer11_mul_partial                 | 11 | ALU_LO(1) | - |
| layer12_mul_combine                 | 12 | TEMP(2) | OUTPUT_HI_THIS_STEP(2) |
| layer13_mem_addr_gather             | 13 | - | ADDR_B2_HI(1) |
| layer13_shifts                      | 13 | ALU_LO(1) | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer14_addr_key_neural_decode      | 14 | ADDR_B0_HI(1) | ADDR_KEY(7) |
| layer14_alu_nocarry_ax_bytes_zero   | 14 | - | OUTPUT_LO(4) |
| layer14_clear_addr_key_pollution    | 14 | - | ADDR_KEY(7) |
| layer14_clear_mem_marker_output     | 14 | - | OUTPUT_LO(4) |
| layer14_clear_output_corruption     | 14 | - | OUTPUT_LO(4) |
| layer14_demo_phase6_wave7           | 14 | - | TEMP(1) |
| layer14_jsr_ax_bytes_zero           | 14 | - | OUTPUT_LO(4) |
| layer14_lc_ax_bytes_zero            | 14 | - | OUTPUT_LO(4) |
| layer14_mem_generation              | 14 | ADDR_B0_LO(1) | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer14_temp_clear                  | 14 | - | OUTPUT_HI(2), TEMP(1) |
| conversational_io_output_routing    | 15 | - | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer15_alu_high_byte_relay         | 15 | - | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer15_memory_lookup               | 15 | - | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer15_nibble_copy                 | 15 | - | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer15_si_mem_addr0_from_stack0    | 15 | - | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer15_store_stack0_sp_byte0_addr  | 15 | - | ADDR_B0_LO(2), ADDR_B0_HI(1) |
| nibble_copy_ffn                     | 15 | - | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |
| layer16_lev_routing                 | 16 | - | ALU_LO(6), OUTPUT_LO(3) |
| tail_bit32_result_correction        | 17 | - | OUTPUT_LO(3), OUTPUT_HI_THIS_STEP(1) |

Patterns visible above:

* **L3-L5 anchors (6 ops)**: stuck on `OP_LEV` and `EMBED_LO` and
  `ADDR_KEY`. `OP_LEV` is read by L3 attn/ffn after being written by
  L5 `opcode_decode_ffn` -- classic forward back-edge from the opcode
  bus. `EMBED_LO` is the L4 PC relay -> L3 cycle.
* **L6 routing (5 ops)**: `OUTPUT_LO` (31 back-edges into
  `layer6_routing_ffn` alone) and `AX_CARRY_HI` dominate. Routing FFN
  is the single largest cycle hub in the graph.
* **L7-L8 fetch/ALU (10 ops)**: `ADDR_KEY` and `AX_CARRY_HI` chain.
  L7 memory heads consume L8 ax_carry refresh, which consumes L7
  memory_heads output -- the multibyte fetch loop.
* **L9-L13 ALU stack (15 ops)**: `ALU_LO`, `CARRY`, `OUTPUT_LO`,
  `OUTPUT_HI_THIS_STEP`. The ALU writeback path stalls here.
* **L14-L17 cleanup tail (21 ops)**: almost every L14-L17 op
  writes `OUTPUT_LO` or `OUTPUT_HI_THIS_STEP`; these get read by
  `layer6_routing_ffn` and `layer7_operand_gather`. This is the
  *biggest* dropped pin from a count perspective.

## 4. Hoist simulation: which dims to retire to reach SCC <= 10

A dim is "hoisted" by splitting it into step-local vs cross-step
variants (or adding explicit `requires`) so the `writes/reads:<dim>`
edges no longer participate in the dep graph. The simulation deletes
ALL writes/reads:<dim> edges and re-runs Tarjan.

### Single-dim hoist (each in isolation)

| Dim | Total edges in SCC | SCC after hoist |
|-----|-------------------:|---------------:|
| `OUTPUT_LO` | 122 | **63** |
| `AX_CARRY_HI` | 104 | 68 |
| `OUTPUT_HI_THIS_STEP` | 44 | 68 |
| `ALU_LO` | 40 | 69 |
| `TEMP` | 40 | 68 |
| `ADDR_KEY` | 33 | 67 |
| `OP_SC` | 29 | 69 |
| `EMBED_LO` | 27 | 69 |
| `CARRY` | 21 | 67 |
| `EMBED_HI` | 19 | 68 |
| `CMP` | 18 | 68 |

`OUTPUT_LO` alone drops the SCC from 69 -> 63. No single dim breaks
the cycle to <= 10.

### Greedy max-reduction (pick the most useful next dim at each step)

This walks the deepest path to SCC <= 10 (8 hoists):

| Step | Dim hoisted | SCC after |
|-----:|-------------|---------:|
| 1 | `OUTPUT_LO`        | 63 |
| 2 | `CMP`              | 40 |
| 3 | `AX_CARRY_HI`      | 31 |
| 4 | `ALU_LO`           | 26 |
| 5 | `TEMP`             | 20 |
| 6 | `OPCODE_BYTE_LO`   | 16 |
| 7 | `ADDR_KEY`         | 14 |
| 8 | `EMBED_HI`         |  9 |

**Recommended minimal hoist set (8 dims)**:
`OUTPUT_LO`, `CMP`, `AX_CARRY_HI`, `ALU_LO`, `TEMP`, `OPCODE_BYTE_LO`,
`ADDR_KEY`, `EMBED_HI`.

For comparison, the count-ordered greedy walk needs **10 dims**:
`{ADDR_KEY, ALU_LO, AX_CARRY_HI, CARRY, EMBED_HI, EMBED_LO, OP_SC,
OUTPUT_HI_THIS_STEP, OUTPUT_LO, TEMP}` -- it picks dims with the
most edges first, but that under-prioritises `CMP` /
`OPCODE_BYTE_LO`, which are smaller-edge but high-leverage cycle
keystones.

## 5. Recommended next migrations

### Phase 8.G.7 (proposed) -- target SCC <= 10 via 8 dim hoists

Order by leverage / migration cost.

**Wave 1 -- the biggest hub (recover ~9 ops):**

1. **`OUTPUT_LO`** split (122 SCC edges, 82 back-edges).
   - Pattern: every L14+ cleanup op writes `OUTPUT_LO` for "this step";
     `layer6_routing_ffn` and `layer7_operand_gather` read it for
     the prev step.
   - Action: introduce `OUTPUT_LO_PREV_STEP` (mirror of the existing
     `OUTPUT_HI_PREV_STEP` split that landed in Phase 7.A.3). All
     L6/L7 readers consume `OUTPUT_LO_PREV_STEP`; all L8+ writers
     keep writing `OUTPUT_LO`. The KV cache carries the prev-step
     value across the step boundary.
   - Touch ops (readers to retarget): `layer6_routing_ffn`,
     `layer7_operand_gather`, `layer10_psh_stack0_passthrough_bake`,
     `l10_post_ops_combined`. The 31 back-edges into
     `layer6_routing_ffn` from L8-L17 collapse.
   - Estimate: medium (similar in scope to the OUTPUT_HI_PREV_STEP
     work; OUTPUT_HI was reverted once -- worth re-reading
     `docs/B9_OUTPUT_HI_SPLIT_SPEC.md` before starting).

**Wave 2 -- ALU loop (recover ~14 ops):**

2. **`CMP`** hoist (18 SCC edges).
   - L9 ALU writes CMP; L8 sp_gather_bake reads it for branch
     evaluation. This is the BZ/BNZ same-step lookahead.
   - Action: add `requires["after"] = layer9_alu` on L8 sp_gather_bake
     (cross-step read via KV); OR rename to `CMP_PREV_STEP` and add
     a step-local CMP variant.
3. **`AX_CARRY_HI`** hoist (104 SCC edges, 15 back-edges).
   - L8 head6 ax_carry_refresh writes; L6 routing/relay heads read
     for multibyte fetch ack.
   - Action: split into `AX_CARRY_HI_THIS_STEP` / `_PREV_STEP` (same
     pattern as OUTPUT_HI). L6 readers retarget to PREV_STEP.
4. **`ALU_LO`** hoist (40 SCC edges).
   - L8/L9/L10 ALU writers; L16 lev_routing reads ALU_LO for return
     value. L11 mul_partial reads ALU_LO from L8 alu.
   - Action: ALU stages already have a phase chain; promoting the
     L8->L9->L10 ALU pipeline into explicit `requires["after"]` edges
     converts the dim-flow edges from same-step to chain-step.

**Wave 3 -- temp / opcode / addr key (recover ~10 ops, reach <=10):**

5. **`TEMP`** hoist (40 SCC edges).
   - L12 mul_combine reads TEMP from L14 temp_clear/demo_phase6_wave7.
   - Action: TEMP is already a scratch register -- enforce a strict
     `produces`/`consumes_fresh` pairing per step. L14 writers go
     into TEMP_THIS_STEP, L12 reader reads TEMP_PREV_STEP.
6. **`OPCODE_BYTE_LO`** hoist (~14 SCC edges total).
   - Opcode decode FFN distributes byte; L3-L5 anchors read it.
     This is purely cross-step (the opcode lives in the prev-step
     residual at fetch time).
   - Action: rename to `OPCODE_BYTE_LO_PREV_STEP` everywhere it is
     read by L3-L5 anchors.
7. **`ADDR_KEY`** hoist (33 SCC edges, 17 back-edges).
   - L4 pc_relay -> L7 memory_heads -> L14 clear_addr_key_pollution
     chain. The L14 cleanup writes a "clear" value for the next
     step; L4 reads it for the next instruction's PC.
   - Action: cross-step pattern. Add `requires["after"] =
     layer14_clear_addr_key_pollution` on `layer4_pc_relay` (already
     attempted in 7.A.2 -- verify it actually fires under the new
     graph).
8. **`EMBED_HI`** hoist (19 SCC edges).
   - L4 pc_relay -> L3 carry_forward_attn cycle on EMBED_HI.
   - Action: same as EMBED_LO PREV_STEP split. EMBED is the token
     embedding for the current step at L0/L1; L3 wants the PREV
     step's embedding via residual.

### Hot ops to focus on

Sorted by total back-edge participation -- these are the migration
targets the dim hoists must touch:

| Op | Static layer | Total back-edges | Dims |
|----|---:|---:|-----|
| `layer6_routing_ffn`                 |  6 | 36 | OUTPUT_LO(31), AX_CARRY_HI(3) |
| `layer7_operand_gather`              |  7 | 32 | OUTPUT_LO(31), OUTPUT_HI(1) |
| `l10_post_ops_combined`              | 10 | 19 | OUTPUT_HI_THIS_STEP(10), OUTPUT_LO(5), ALU_LO(1), CARRY(1) |
| `layer10_psh_stack0_passthrough_bake` | 10 | 18 | OUTPUT_LO(15), OUTPUT_HI(1) |
| `layer16_lev_routing`                | 16 |  9 | ALU_LO(6), OUTPUT_LO(3) |
| `layer9_alu`                         |  9 |  9 | CARRY(3), ALU_HI(2), ALU_LO(1), OUTPUT_LO(2), CMP(1) |
| `layer14_addr_key_neural_decode`     | 14 |  8 | ADDR_KEY(7), ADDR_B0_HI(1) |
| `layer14_clear_addr_key_pollution`   | 14 |  7 | ADDR_KEY(7) |
| `layer8_multibyte_fetch{,_bake}`     |  8 |  7 ea | AX_CARRY_HI(5), ADDR_KEY(2) |
| `layer7_memory_heads`                |  7 |  7 | AX_CARRY_HI(3), ADDR_KEY(3), PSH_AT_SP(1) |

`layer6_routing_ffn` and `layer7_operand_gather` together account for
**68 back-edges** (~83% of `OUTPUT_LO`). Both reads are PC-relative
prev-step semantics. **A single OUTPUT_LO_PREV_STEP migration plus
re-pointing these two readers retires the largest contributor to the
SCC.**

## 6. Sanity checks

* `phase_inconsistent_with_deps`: 0 -- no acyclic op contradicts its
  declared DAG depth. The cycle is the only obstacle; no rule sweeps
  required first.
* `phase_required_but_undeclared`: 0 -- every acyclic op's static phase
  is pinned by some dep edge. No "phantom phase" ops to backfill.
* `freely_placeable`: 25 -- post-pass model ops + L8 ALU subops; not
  in the SCC.
* Note about double-counting: `OUTPUT_LO` shows 82 back-edges (this
  audit) versus 58 in the analyzer's default report (3.B). The
  difference is the analyzer's tie-break heuristic (it counts each
  (u, v) edge once per reason; this audit counts back-edges per dim
  but de-dupes (u, v) at the dim level). Both produce the same SCC
  size — pick whichever convention is convenient.

## 7. Effort estimate to reach SCC <= 10

* Migrations 1+2 (`OUTPUT_LO` split + `CMP` PREV_STEP) — ~3-4 days.
  Drops SCC 69 -> 40.
* Migrations 3+4 (`AX_CARRY_HI` split + `ALU_LO` chain) — ~3-4 days.
  Drops SCC 40 -> 26.
* Migrations 5-8 (`TEMP` / `OPCODE_BYTE_LO` / `ADDR_KEY` / `EMBED_HI`)
  — ~1 week. Drops SCC 26 -> <= 10.

Aggregate: **~2 weeks of focused work**, matching the Phase 7
closing audit's bottom-line estimate. The longest pole is
`OUTPUT_LO` (touches the largest number of ops); the trickiest is
`AX_CARRY_HI` (its 104 same-step edges mean the split must be
careful not to break the L8 multibyte ack chain).

## 8. Appendix — back-edge-only single dim removal is a no-op

For completeness, "drop only the back-edges of dim X" (the static
phase-pruning operation) does not break the SCC for any single dim:

| Dim back-edges dropped | SCC | Cycle members |
|---|---:|---:|
| `OUTPUT_LO` (82) | 69 | 89 |
| `ADDR_KEY` (17) | 69 | 89 |
| `AX_CARRY_HI` (15) | 69 | 89 |
| `OUTPUT_HI_THIS_STEP` (11) | 69 | 89 |
| `ALU_LO` (8) | 69 | 89 |
| every other top-10 dim | 69 | 88-89 |

This confirms phase-pruning alone is not enough — the cycle has
forward-only (same-step) paths that survive even after every static
back-edge of one dim is removed. The dim must be split / hoisted, not
just locally re-ordered.

---

**Generated by**: `tools/analyze_scheduler.py` + `/tmp/scc_audit.py`
(companion drill-down). Both are read-only and reproducible from
this HEAD.
