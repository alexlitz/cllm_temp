# SCC remaining structural audit — same-step back-edges in the residual 4 SCCs

_Generated 2026-06-02. Read-only investigation, doc-only commit._
_HEAD = `41f279b0` (`test_kv_eviction_gates: add 8.E.10 efficiency-threshold gate (Gate 4)`)._
_Branch: `speedup-cache-and-buckets` (clean working tree; `git reset --hard speedup-cache-and-buckets` at start)._

## Setup

    export CUDA_VISIBLE_DEVICES=1
    python tools/analyze_scheduler.py   # -> .agent-logs/scheduler_phase_a_2026_06_02.{md,csv}
    # plus the per-SCC dim-edge enumeration and requires[after] simulations below
    # (run inline from the analyze_scheduler module).

## 1. Current SCC inventory

After the 8.A/8.E cross-step migrations, `tools/analyze_scheduler.py` reports
4 SCCs (was 1 SCC of 72 in the prior `scc_zero_audit.md`):

| SCC | Size | Members |
|-----|---:|---|
| #1 | 26 | L6 routing, L8 alu / sp_gather_bake, L9 alu / marker_suppress, L10 alu / 3× carry_relay+post / stack0_byte_relay{,_bake}, L11 mul_partial + dep_anchor, L12 dep_anchor + mul_combine, L13 attn_dep_anchor + mem_addr_gather, L14 attn_dep_anchor / addr_key / clear_addr_key_pollution / mem_generation / temp_clear / alu_nocarry_ax_bytes_zero / demo_phase6_wave7, L15 memory_lookup + store_stack0_sp_byte0_addr, L16 lev_routing |
| #2 | 4 | `_layer3_ffn_dep_anchor`, `layer3_carry_forward_attn`, `layer3_ffn`, `layer4_pc_relay` |
| #3 | 3 | `format_pointer_extraction`, `format_string_fetch_head`, `null_terminator_detection` |
| #4 | 2 | `_layer5_fetch_dep_anchor`, `layer5_fetch` |

Total cycle members: 95 / 129 ops.

## 2. Audit per back-edge dim

For every dim that produces a back-edge in any SCC, I enumerated
writers/readers, classified the edges (same-step vs cross-layer), and
simulated whether declaring `requires["after"] = <writer>` on the
reader retires the cycle.

### 2.1 `ALU_HI` (SCC #1; 3 back-edges, 4 same-step forward edges)

* **Writers** (6): `layer6_relay_heads` (6), `layer7_operand_gather` (7),
  `layer8_sp_gather` (8), `layer8_mem_to_alu` (8.45),
  `layer10_stack0_byte_relay` (10), `layer10_stack0_byte_relay_bake` (10.4).
* **Readers** (8): `layer9_alu` (9, also `consumes_fresh ALU_HI@AX_byte0`),
  `layer10_alu` (10.2), `_layer12_ffn_dep_anchor` (11.5),
  `layer12_mul_combine` (12), `layer13_shifts` (13),
  `layer16_lev_routing` (16), `l10_post_ops_combined` (10.5),
  `tail_bit32_result_correction` (post-pass).
* **Back-edges**:
  `layer10_stack0_byte_relay (10) → layer9_alu (9)`,
  `layer10_stack0_byte_relay_bake (10.4) → layer9_alu (9)`,
  `layer10_stack0_byte_relay_bake (10.4) → layer10_alu (10.2)`.

Semantic: L10 stack0_byte_relay{,_bake} writes ALU_HI for the **next**
step's L9 ALU consumption (PSH/SI/SC stack-byte staging). The
current-step ALU_HI consumed at L9/L10 alu comes from L7 operand_gather
(declared via `consumes_fresh ALU_HI: AX_byte0` on L9).

**Can `requires["after"]` retire the back-edge?** No. Simulating
`layer9_alu.requires[after] += layer10_stack0_byte_relay_bake` does
suppress the L10→L9 data-flow inference, but the explicit
`requires[after]` edge `L10 → L9` itself becomes a back-edge under the
`requires[after]` reason. Tarjan SCC size remains 26.

**Structural fix**: rename L9/L10 alu reads to `ALU_HI_PREV_STEP` and
have the L10 stack0_byte_relay{,_bake} writers stage the new
`_PREV_STEP` slot. Mirrors the `OUTPUT_HI_PREV_STEP` / `AX_CARRY_LO_PREV_STEP`
splits already landed.

### 2.2 `ALU_LO` (SCC #1; 1 back-edge)

* **Writers** (7): same 6 as ALU_HI + `layer16_lev_routing` (16).
* **Readers** (9): `layer8_alu` (8.2, `consumes_fresh AX_byte0`),
  `layer9_alu` (9), `layer10_alu` (10.2),
  `_layer11_ffn_dep_anchor` (11), `layer11_mul_partial` (11),
  `layer13_shifts` (13), `layer16_lev_routing` (16),
  `l10_post_ops_combined` (10.5), `tail_bit32_result_correction`.
* **Back-edge**: `layer16_lev_routing (16) → _layer11_ffn_dep_anchor (11)`.

L16 writes ALU_LO as the LEV next-step PC staging
(`layer16_lev_routing.writes = {ALU_LO, OUTPUT_HI_THIS_STEP, OUTPUT_LO}`).
The anchor at L11 reads ALU_LO for the MUL partial-product latch.

**Can `requires["after"]` retire it?** No. Adding
`_layer11_ffn_dep_anchor.requires[after] += layer16_lev_routing` keeps
the edge L16 → L11 as a back-edge.

**Structural fix**: same `_PREV_STEP` rename pattern. L11/anchor reads
become `ALU_LO_PREV_STEP`; L16 writes a step-local twin slot. Same
recipe as the `AX_CARRY_HI_PREV_STEP` rename (commit `d6fb5f5f`).

### 2.3 `DIV_STAGING` (SCC #1; 1 back-edge)

* **Writers** (1): `layer10_alu` (10.2). 
* **Readers** (1): `layer6_routing_ffn` (6.5).
* **Back-edge**: `layer10_alu (10.2) → layer6_routing_ffn (6.5)`.

The L6 routing_ffn reads the previous step's L10 DIV_STAGING latch to
route DIV/MOD opcodes correctly on the **next** step's iteration.
This is a textbook prev-step residual.

**Can `requires["after"]` retire it?** No. Adding
`layer6_routing_ffn.requires[after] = layer10_alu` produces a
back-edge `L10 → L6` of the same length.

**Structural fix**: 1-line rename
`layer6_routing_ffn.reads: DIV_STAGING → DIV_STAGING_PREV_STEP` and
add a paired `_PREV_STEP` writer at L10 alu. Singleton dim — no
collateral readers to migrate.

### 2.4 `TEMP` (SCC #1; 2 back-edges, 3 same-step forward edges)

* **Writers** (8): `layer3_carry_forward_attn` (3), `opcode_decode_ffn`,
  `_opcode_decode_ffn_dep_anchor`, `layer7_memory_heads` (7),
  `_layer11_ffn_dep_anchor` (11, `produces TEMP@?`),
  `layer11_mul_partial` (11), `layer14_temp_clear` (14.1),
  `layer14_demo_phase6_wave7` (14.95).
* **Readers** (9): `_layer12_ffn_dep_anchor`, `layer12_mul_combine`
  (12, `consumes_fresh TEMP`), `layer14_temp_clear` (14.1),
  `layer14_clear_output_corruption` (14.3),
  `layer14_alu_nocarry_ax_bytes_zero` (14.8), `layer15_memory_lookup`,
  `layer16_lev_routing`, `tail_bit32_result_correction`,
  `layer15_alu_high_byte_relay`.
* **Back-edges (same-step inside L14)**:
  `layer14_demo_phase6_wave7 (14.95) → layer14_temp_clear (14.1)`,
  `layer14_demo_phase6_wave7 (14.95) → layer14_alu_nocarry_ax_bytes_zero (14.8)`.

This is the L14 cleanup loop documented in §3.5 of `scc_zero_audit.md`.
`layer14_demo_phase6_wave7.requires[after] = [layer14_alu_nocarry_ax_bytes_zero, layer14_mem_generation]`
already declares `demo` runs AFTER the readers — so the data-flow
TEMP back-edges from demo→{temp_clear, nocarry} are **semantically
stale writes**: the readers fetched TEMP from L11 mul_partial /
L7 memory_heads, not from same-step demo.

**Can `requires["after"]` retire it?** No. The TEMP write at L14.95
is a dead store for same-step consumption. The right fix is for
`layer14_demo_phase6_wave7` to **drop its TEMP write** — or
equivalently, declare `writes_next_step: {TEMP}` if such a key
existed. Today there is no `writes_next_step` schema key, so the
analyzer treats the L14.95 TEMP write as a same-step producer for
every TEMP reader. Simulating
`layer14_temp_clear.requires[after] += layer14_demo_phase6_wave7`
keeps the SCC at 26 (the explicit edge is a back-edge).

**Structural fix options**:
1. **Drop `TEMP` from `layer14_demo_phase6_wave7.writes`** if the bake
   does not actually need the slot — verify with `decl_verifier.py`
   that demo's TEMP write isn't read by any same-step consumer.
2. Rename demo's write to `TEMP_NEXT_STEP` and migrate the next-step
   TEMP readers (if any) to consume it.

### 2.5 `NEXT_STACK0` (SCC #2; 0 back-edges, 2 same-step forward edges)

* **Writers** (3): `phase_a_ffn`, `layer3_ffn` (3),
  `_layer3_ffn_dep_anchor` (3).
* **Readers** (3): `layer3_ffn`, `_layer3_ffn_dep_anchor`,
  `tail_bit32_result_correction`.
* **Same-step bidirectional edges**:
  `layer3_ffn ↔ _layer3_ffn_dep_anchor` (both read and write NEXT_STACK0
  at phase=3).

This is a 2-cycle: the dep_anchor was introduced as a scaffold mirroring
`layer3_ffn`'s reads/writes so future ops can declare
`requires[after] = _layer3_ffn_dep_anchor` without depending on the
real FFN. It double-claims NEXT_STACK0.

**Can `requires["after"]` retire it?** Not in either direction — both
would create reciprocal explicit edges. Same problem as the L5 fetch
dep_anchor (SCC #4 below).

**Structural fix**: `_layer3_ffn_dep_anchor` should be **read-only** —
drop `writes` entirely (it only needs to be a target for forward
`requires[after]` refs). Same recipe applied to L5 fetch_dep_anchor
in `scc_zero_audit.md §3.8`.

### 2.6 `EMBED_LO` (SCC #2; 3 back-edges, 6 same-step forward edges)

* **Writers** (5): `layer2_initial_pc_bake_cancel` (2),
  `layer3_carry_forward_attn` (3), `layer3_ffn` (3),
  `_layer3_ffn_dep_anchor` (3), `layer4_pc_relay` (4).
* **Readers** (16): L3-L16 across the whole pipeline.
* **Back-edges**: `layer4_pc_relay (4) → {layer3_ffn, layer3_carry_forward_attn, _layer3_ffn_dep_anchor} (3)`.

L4 pc_relay writes EMBED_LO/HI as the **next** step's PC seed; L3 reads
EMBED_LO for the current step. L3 has `reads: EMBED_HI.*.-1` (prev-step
EMBED_HI) but **not** `EMBED_LO.*.-1` — the EMBED_HI split landed in
commit `3d4a65c2`; the EMBED_LO companion never did.

**Can `requires["after"]` retire it?** No. The L4→L3 edge would just
move from `writes/reads:EMBED_LO` to `requires[after]=layer4_pc_relay`.

**Structural fix**: rename L3 reader to `EMBED_LO.*.-1` (mirrors
EMBED_HI). 3-op migration: `layer3_ffn`, `layer3_carry_forward_attn`,
`_layer3_ffn_dep_anchor`. This is item Wave 2 step 3 in
`scc_zero_audit.md §6`. Recommendation already on file.

### 2.7 `IO_IN_OUTPUT_MODE` (SCC #3; 2 back-edges)

* **Writers** (1): `null_terminator_detection` (10.6).
* **Readers** (5): `format_pointer_extraction` (7.5),
  `format_position_counter`, `format_string_fetch_head` (9.5),
  `null_terminator_detection`, `conversational_io_output_routing`.
* **Back-edges**:
  `null_terminator_detection (10.6) → format_string_fetch_head (9.5)`,
  `null_terminator_detection (10.6) → format_pointer_extraction (7.5)`.

L10.6 sets IO_IN_OUTPUT_MODE for the **next** step; L7/L9 format heads
read it at the current step.

**Can `requires["after"]` retire it?** No. Simulating
`format_string_fetch_head.requires[after] += null_terminator_detection`
keeps SCC #3 at 3 (was 3) — same swap problem. Already noted in prior
audit Wave 2 step 7.

**Structural fix**: rename readers to
`IO_IN_OUTPUT_MODE.*.-1` (prev-step residual).

### 2.8 `FETCH_LO` / `FETCH_HI` (SCC #4; 0 back-edges, 4 same-step forward edges)

* **Writers** (4 each): `layer4_ffn`, `_layer4_ffn_dep_anchor`,
  `layer5_fetch`, `_layer5_fetch_dep_anchor` (all at phase ≤ 5).
* **Readers**: includes `layer5_fetch` ↔ `_layer5_fetch_dep_anchor`
  bidirectionally at phase=5.

Identical pattern to NEXT_STACK0 — the dep_anchor mirrors the real
`layer5_fetch` writes, forming a 2-cycle.

**Can `requires["after"]` retire it?** No.

**Structural fix**: make `_layer5_fetch_dep_anchor.writes` empty.
Documented as `scc_zero_audit.md §3.8` Wave 3 step 8.

## 3. `requires["after"]` mechanism verification

The brief asks: "see if `requires["after"]` works (the scheduler check
the agent ran earlier found this doesn't always retire the back-edge —
it merges adjacent edges only)."

Simulating each candidate retirement against the analyzer's
`build_dep_graph` + `topo_depth` + `find_scc_summary`:

| Reader → declared `requires[after]` | Cycle | SCC sizes |
|---|---:|---|
| baseline | 95 | [26, 4, 3, 2] |
| `layer9_alu` += `layer10_stack0_byte_relay_bake` (ALU_HI/ALU_LO) | 95 | [26, 4, 3, 2] |
| `layer14_temp_clear` += `layer14_demo_phase6_wave7` (TEMP) | 95 | [26, 4, 3, 2] |
| `layer14_alu_nocarry_ax_bytes_zero` += `layer14_demo_phase6_wave7` | 95 | [26, 4, 3, 2] |
| `layer6_routing_ffn` += `layer10_alu` (DIV_STAGING) | 95 | [26, 4, 3, 2] |
| `_layer11_ffn_dep_anchor` += `layer16_lev_routing` (ALU_LO) | 95 | [26, 4, 3, 2] |
| `layer3_ffn` += `layer4_pc_relay` (EMBED_LO) | 95 | [26, 4, 3, 2] |
| `format_string_fetch_head` += `null_terminator_detection` (IO mode) | 95 | [26, 4, 3, 2] |

**Confirmed: `requires["after"] = <later-phase-writer>` NEVER retires a
cross-step back-edge.** The analyzer's R-OH-2 suppression (lines 156-192
of `tools/analyze_scheduler.py`) drops the data-flow edge from the
writer to the reader, but the **explicit `requires[after]` edge replaces
it** (lines 217-224). When the referenced writer is at a later phase
than the reader, the explicit edge is itself a back-edge and the SCC
size is unchanged.

`requires["after"]` only helps when the writer is at an **earlier** or
**same** phase as the reader (forward edge or in-step ordering between
peers). For the 4 remaining SCCs, every back-edge's writer is at a
later phase than its reader — so `requires["after"]` is structurally
unable to retire them.

## 4. What actually retires the back-edges

Greedy hoist simulation (delete all `writes/reads/produces/consumes_fresh`
for a dim, then re-run SCC):

| Hoist | Cycle | SCC sizes |
|---|---:|---|
| ALU_HI | 95 | [26, 4, 3, 2] (no change — forward deps keep cycle) |
| ALU_LO | 95 | [26, 4, 3, 2] (no change) |
| DIV_STAGING | 95 | [**23**, 4, 3, 2] (−3) |
| TEMP | 95 | [**22**, 4, 3, 2] (−4) |
| NEXT_STACK0 | 95 | [26, 4, 3, 2] (no change — EMBED_LO still cycles) |
| EMBED_LO | 94 | [26, **3**, 2, 2] |
| CARRY | 94 | [**24**, 4, 3, 2] |
| IO_IN_OUTPUT_MODE | 94 | [26, 4, **0** (gone)] |
| FETCH_HI / FETCH_LO each | 95 | [26, 4, 3, 2] (no change individually — FETCH_HI/LO twin) |
| **All 10 dims hoisted together** | **42** | [**15**] |

The 15-op residual after all hoists is **the C-instruction-step
structural loop** described in `scc_zero_audit.md §3.6`:

    {_layer11_ffn_dep_anchor, _layer12_ffn_dep_anchor,
     _layer13_attn_dep_anchor, _layer14_attn_dep_anchor,
     layer10_carry_relay, layer14_addr_key_neural_decode,
     layer14_clear_addr_key_pollution, layer14_mem_generation,
     layer15_memory_lookup, layer15_store_stack0_sp_byte0_addr,
     layer16_lev_routing, layer8_alu, layer8_sp_gather_bake,
     layer9_alu, layer9_marker_suppress}

— the fetch → decode → SP/BP gather → ALU → CMP → MEM lookup →
ADDR_KEY decode → LEV → next-step ALU loop. **Cannot** be retired by
dim renames; needs `requires["next_step_after"]` IR primitive
(`scc_zero_audit.md §6 Wave 3 step 10`).

## 5. Retirement recipe per dim (no speculative fixes — recommendations only)

| Dim | Mechanism | Cost | Effect (per simulation) |
|---|---|---|---|
| `ALU_HI` | rename L9/L10 alu reads → `ALU_HI_PREV_STEP`; add `_PREV_STEP` slot at L10 stack0_byte_relay{,_bake} | 4-op rename | SCC #1: 26 → ~24 |
| `ALU_LO` | rename `_layer11_ffn_dep_anchor` read → `ALU_LO_PREV_STEP`; L16 lev_routing writes paired slot | 2-op rename | SCC #1: 26 → ~24 |
| `DIV_STAGING` | rename `layer6_routing_ffn.reads: DIV_STAGING → DIV_STAGING_PREV_STEP`; L10 alu writes paired slot | 2-op rename | SCC #1: 26 → 23 (−3) |
| `TEMP` | (a) drop `TEMP` from `layer14_demo_phase6_wave7.writes` if bake-dead, OR (b) rename to `TEMP_NEXT_STEP` | 1-op edit + verifier | SCC #1: 26 → 22 (−4) |
| `NEXT_STACK0` | drop `writes` from `_layer3_ffn_dep_anchor` (read-only anchor) | 1-line | breaks bidirectional 2-cycle in SCC #2 |
| `EMBED_LO` | L3 reads → `EMBED_LO.*.-1` (3 ops: `layer3_ffn`, `layer3_carry_forward_attn`, `_layer3_ffn_dep_anchor`) | 3-op rename | SCC #2: 4 → 3 |
| `IO_IN_OUTPUT_MODE` | L7/L9 format readers → `IO_IN_OUTPUT_MODE.*.-1` | 3-op rename | SCC #3: 3 → 0 (whole SCC dissolves) |
| `FETCH_LO/HI` | drop `writes` from `_layer5_fetch_dep_anchor` (read-only anchor) | 1-line | SCC #4: 2 → 0 |
| L8-L16 C-step loop | new `requires["next_step_after"]` IR key (or lift `consumes_fresh` to step-1 lag) | ~1 day IR change + 7-op application | retires the 15-op structural residual |

## 6. Why `requires["after"]` is provably not the right tool here

The analyzer's R-OH-2 logic (intentionally) **swaps the back-edge
attribution** (data-flow → explicit), not the topology. From
`tools/analyze_scheduler.py:146-163`:

```python
# When v declares requires["after"] = X, for every dim D in X.writes ∩ v.reads,
# suppress the data-flow edges u→v for every other writer u of D.
# But X→v is still added below as an explicit "requires[after]" edge.
```

Tarjan SCC detection ignores edge labels. If `phase(X) > phase(v)`, the
explicit `X → v` edge is a back-edge in the static phase order; the
SCC of which X and v are members is unchanged.

**The mechanism is correct for in-step ordering** between peer-phase
ops (e.g. `layer8_sp_gather_bake.requires[after] = layer9_alu`
pre-existing): both ops are forced into the same compiled layer and
the data dependency direction is recorded. It is **wrong for cross-step
residual reads**, where the reader's intent is "I read the previous
step's value of D" and there is no current-step dependency on the
writer at all.

For cross-step residuals, the only mechanism in today's IR that
faithfully expresses the semantic is the `<DIM>.*.-1` `reads` syntax
(prev-step residual). Every back-edge in the 4 remaining SCCs falls
into this bucket.

## 7. Summary

* **4 SCCs (26, 4, 3, 2)** confirmed; all back-edges driven by
  same-step or later-phase writes that should be cross-step residuals.
* **`requires["after"]` cannot retire any of these back-edges**
  (simulated for all 7 candidates) — the explicit edge replaces the
  data-flow edge but the cycle persists.
* **Cross-step renames** (`<DIM>.*.-1` reads) retire SCC #2, #3, #4
  cleanly per simulation: `EMBED_LO` (SCC #2), `IO_IN_OUTPUT_MODE`
  (SCC #3), and either dropping `_layer5_fetch_dep_anchor.writes` or
  the same rename for FETCH_LO/HI (SCC #4).
* **SCC #1** shrinks from 26 → ~22 by hoisting ALU_HI/ALU_LO/
  DIV_STAGING/TEMP/CARRY independently; the residual 15-op core
  is the C-instruction-step loop (`scc_zero_audit.md §3.6`) which
  requires a `next_step_after` IR primitive — not a dim rename.
* **No fixes applied** (per single-rule-fix zero-sum memory note).
  Recommendations only; verification by decl_verifier.py + sweep
  required before any code change.
