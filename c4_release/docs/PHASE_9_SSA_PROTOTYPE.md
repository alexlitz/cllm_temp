# Phase 9 — SSA dim versioning (prototype)

_Phase 9 design + prototype. Companion to [`PHASE_8_PLAN.md`](PHASE_8_PLAN.md)
§8.A (PREV_STEP alias decomposition), §8.H (zero-pin demo op), and
[`CONTROL_FLOW_DETECTOR_HEADS.md`](CONTROL_FLOW_DETECTOR_HEADS.md)._

## 0. Summary

Replace the ad-hoc `_PREV_STEP` dim-alias trick with LLVM-inspired SSA
naming: each cross-step read is tagged with the *producing op* and the
*step offset* it reads from. Two unrelated cross-step reads of
`OUTPUT_LO` are now distinguishable by *value identity*, not just by
name. Byte-identity at default is preserved because the SSA name is
aliased onto the base dim's numeric slot at allocation time — exactly
like `OUTPUT_LO_PREV_STEP` does today.

This file documents the prototype (parser + scheduler hook + 1 demo op)
and the migration plan for the corpus.

## 1. The problem the PREV_STEP alias trick has

[`PHASE_8_PLAN.md`](PHASE_8_PLAN.md) §8.A and the audit at
`.agent-logs/scc_zero_audit.md` describe today's pattern:

```python
# c4_release/neural_vm/dim_registry.py:610-625
_pin("OUTPUT_LO_PREV_STEP", 174, 16,
          "OUTPUT_LO from previous step (aliases OUTPUT_LO)",
          semantics="is_byte OR NOT is_byte", alias=True)
```

A reader spells its cross-step read as `OUTPUT_LO_PREV_STEP`; the dim
registry maps it onto the same numeric slot (174) as `OUTPUT_LO`; the
scheduler's dep-graph cycle detector treats them as different dim
names and so doesn't see the back-edge.

The pattern works but has four sharp edges:

1. **Single global synonym per band.** A reader that genuinely needs two
   distinct cross-step producers (e.g. one for LEV-following, one for
   normal flow) has only one synonym available. Both produce the same
   numeric slot but the dep graph can't keep them apart.
2. **Producer identity is lost.** `OUTPUT_LO_PREV_STEP` does not say
   *which* upstream op produced the value. Downstream analyzers
   (KV liveness, staleness, eviction) need a side-table to recover it.
3. **Alias proliferation.** Every cross-step read of a new band needs
   a new `_pin(..., alias=True)` entry in `dim_registry.py` (today: 12
   such entries). Each is one source of footgun (typos, mis-aliasing).
4. **Schedule semantics are smeared.** `requires["after"]=upstream_op`
   plus the synonym rename together encode "this is a back-edge"; the
   B9 EXCEPTION in `_topological_sort` (l. 1339-1364) special-cases the
   combination. Two separate spellings of the same idea.

## 2. SSA schema

```
SSA_NAME    := BASE_DIM "." WRITER_OP "." STEP_OFFSET
BASE_DIM    := identifier  (e.g. "OUTPUT_LO")
WRITER_OP   := op_name | "*"          ("*" = any writer)
STEP_OFFSET := integer    (-1 = prev VM step, 0 = current, +1 = next, ...)
```

### Examples

| Spelling | Meaning |
|---|---|
| `OUTPUT_LO` | Unversioned. Current step, current writer (default). |
| `OUTPUT_LO.layer16_lev_routing.-1` | OUTPUT_LO from prev step's L16 LEV writer. |
| `OUTPUT_LO.*.-1` | OUTPUT_LO from any writer one step ago (multi-producer). |
| `ADDR_KEY.layer14_addr_key_neural_decode.-2` | Two steps back, named writer. |

### Why this shape

- **Three segments, dot-separated.** Round-trips cleanly through Python
  strings; greppable; parses in O(1).
- **`*` writer wildcard.** Mirrors the LLVM `undef` / "any predecessor"
  semantics for cross-step reads that aggregate over multiple writers
  (the common case for attention heads that look back at the prev-step
  AX marker without knowing which op owned it).
- **Signed integer step offset.** Avoids overloading "prev" (could be
  -1, -2, +1, ...). `-1` is the dominant case; the parser is uniform
  for all integers.

### Why not "true" SSA φ nodes

True SSA would attach a φ node at each control-flow join expressing
which predecessor contributed each value. In this codebase the only
"join" is the autoregressive step boundary, and the dep graph already
collapses it into "prev-step from any writer". The wildcard `*`
writer captures this without a new IR node.

If a future analyzer needs per-writer per-step liveness (Phase 9.C
KV eviction refinement), it can split the `*` form into explicit
single-writer reads — the SSA schema already supports this.

## 3. Byte-identity strategy

At `LayerCompiler.add_op` time, each SSA-form read is auto-declared as
an `alias_of` its base dim:

```python
# c4_release/neural_vm/unified_compiler/layer_compiler.py:add_op
for d in op.reads | op.writes:
    if d in self.dims:
        continue
    if is_ssa_form(d):
        parsed = parse_ssa_name(d)
        self.declare_dim(d, self.dims[parsed.base_dim],
                         alias_of=parsed.base_dim)
        continue
    raise ValueError(...)
```

The existing `_allocate_dims` alias-resolution loop (l. 1597-1609) then
gives the SSA form the base dim's numeric position. Bake time uses
`dim_positions["OUTPUT_LO.*.-1"]` and gets the same column as
`dim_positions["OUTPUT_LO"]`. Lowered weights are bit-identical to the
unversioned form provided the op's bake code reads/writes through
either spelling consistently.

End-to-end gate (from the demo op at `l8_ops.py:1872`):

```
compile_full_vm_dynamic(strict=False):
  OUTPUT_LO @ 69, OUTPUT_LO.*.-1 @ 69   # aliased -- byte-identical
  OUTPUT_HI @ 85, OUTPUT_HI.*.-1 @ 85   # aliased -- byte-identical
  layer8_head6_ax_carry_refresh: placed on L8 attn block
```

The demo op's bake is `enable=False` (production default), so the
model state_dict is unchanged. Phase 9.B enables this op end-to-end
once the L8 head-budget question (CONTROL_FLOW_DETECTOR_HEADS.md R2)
is resolved.

## 4. Scheduler hook

`_topological_sort` skips the dim-only dep-graph edge when the read is
SSA cross-step (`step_offset != 0`):

```python
for d in v.reads:
    if is_ssa_form(d):
        parsed = parse_ssa_name(d)
        if parsed.is_cross_step:
            continue   # the value is from a previous step -- no back-edge
    ...  # normal same-step edge logic
```

This replaces the B9 EXCEPTION in the old `requires["after"]` plus
PREV_STEP-rename combination: the cross-step semantics are now in the
read name itself, not split across two metadata channels.

For SSA reads with a *named* writer (no `*`), a future refinement can
add an explicit `writer_op -> reader_op` back-edge to the analyzer
(useful for "this op MUST come after that op in the prior step"
ordering claims). For the prototype, all SSA cross-step reads are
treated as full wildcards.

## 5. Prototype scope (what landed this phase)

| Piece | Location | Notes |
|---|---|---|
| Parser + schema | `neural_vm/unified_compiler/ssa_dim.py` | Pure stdlib. `SsaDimName`, `parse_ssa_name`, `is_ssa_form`, `base_of`, `make_ssa_name`. |
| Tests for parser | `tests/test_ssa_dim.py` | 16 cases: round-trip, wildcards, edge cases. |
| `LayerCompiler` hook | `neural_vm/unified_compiler/layer_compiler.py` | `add_op` auto-aliases SSA names; `_topological_sort` skips cross-step edges. |
| Tests for scheduler | `tests/test_ssa_scheduler.py` | Auto-alias, position equality, cycle-breaking, error paths. |
| Demo op | `neural_vm/unified_compiler/ops/l8_ops.py:1872` `make_layer8_head6_ax_carry_refresh_op` | `OUTPUT_LO_PREV_STEP` -> `OUTPUT_LO.*.-1`; same for HI. |
| Tests for demo op | `tests/test_ssa_demo_op.py` | SSA rename present; positions aliased; bake unchanged at `enable=False`. |

What did *not* land:

- Corpus-wide rename of the ~12 existing `_PREV_STEP` reads (deferred to
  Phase 9.B; see migration plan §7).
- KV-eviction analyzer integration (deferred to Phase 9.C; the analyzer
  can use `parse_ssa_name(...).writer_op` to compartmentalize the cache
  by producer).
- Static-path support (`compile_full_vm`). The static path is on the
  deletion track per `PHASE_8_PLAN.md` §8.G; SSA prototype targets
  `compile_full_vm_dynamic` only.

## 6. Why this is V2-aligned

[`PHASE_8_PLAN.md`](PHASE_8_PLAN.md) §7 cites the V2 vision: every
cross-step edge should be *expressed in the data flow* (residual band
+ attention head) rather than as a scheduler hint. SSA naming is the
*notation* that makes the expression explicit — the residual still
flows through the same numeric slot, but the name carries the producer
+ step semantics.

This dovetails with [`CONTROL_FLOW_DETECTOR_HEADS.md`](CONTROL_FLOW_DETECTOR_HEADS.md):
when a future detector head replaces a `requires["after"]=upstream_op`
back-edge with a real attention head that *attends* to the prev-step
opcode marker, the head's reads get SSA names whose `writer_op` is the
detector op itself and whose `step_offset` is `0` (the detector writes
in the current step). The dep graph naturally turns into a DAG.

## 7. Migration plan

### Phase 9.B — corpus PREV_STEP -> SSA rename

Mechanical rename of the 12 `_PREV_STEP` aliases declared in
`dim_registry.py:610-870` and their ~30 reader sites in `ops/*.py`.
Per-band, single writer where known:

| Today | Becomes | Single writer? |
|---|---|---|
| `OUTPUT_LO_PREV_STEP` | `OUTPUT_LO.*.-1` | No -- multi-writer at AX marker. |
| `OUTPUT_HI_PREV_STEP` | `OUTPUT_HI.*.-1` | No. |
| `TEMP_PREV_STEP` | `TEMP.*.-1` | No (TEMP is multi-purpose scratch). |
| `EMBED_LO_PREV_STEP` | `EMBED_LO.layer4_pc_relay.-1` | Yes -- L4 is sole producer. |
| `EMBED_HI_PREV_STEP` | `EMBED_HI.layer4_pc_relay.-1` | Yes -- L4. |
| `ADDR_KEY_PREV_STEP` | `ADDR_KEY.layer14_addr_key_neural_decode.-1` | Yes. |
| `ALU_LO_PREV_STEP` | `ALU_LO.layer9_alu.-1` | Single dominant writer. |
| `AX_CARRY_LO_PREV_STEP` | `AX_CARRY_LO.*.-1` | Multi (L7 heads + L8 head 6). |
| `AX_CARRY_HI_PREV_STEP` | `AX_CARRY_HI.*.-1` | Multi. |
| `OPCODE_BYTE_LO_PREV_STEP` | `OPCODE_BYTE_LO.layer5_opcode_decode_ffn.-1` | Yes. |
| `OP_LEV_PREV_STEP` | `OP_LEV.layer5_opcode_decode_ffn.-1` | Yes. |
| `ADDR_B0_LO_PREV_STEP` / `..._HI_PREV_STEP` | `ADDR_B0_*.layer15_memory_lookup.-1` | Yes. |
| `CARRY_PREV_STEP` | `CARRY.layer9_alu.-1` | Single. |
| `CMP_PREV_STEP` | `CMP.layer9_alu.-1` | Single. |

Method: one PR per band. For each:

1. Audit the readers via `grep -rn "<BAND>_PREV_STEP" ops/`.
2. Rename the read to the SSA form (single writer if known; `*` if multi).
3. Delete the `_pin("<BAND>_PREV_STEP", ..., alias=True)` in
   `dim_registry.py`.
4. Verify `compile_full_vm_dynamic(strict=False)` produces a byte-identical
   `state_dict` (positions are the same; alias goes through the
   LayerCompiler instead of the registry).
5. Run the band's smoke tests + 1096 corpus.

Estimated effort: 1-2 agent-days per band; total ~14 days. Parallelizable
across bands (no cross-band coupling).

### Phase 9.C — KV eviction analyzer integration

`tools/kv_overwrite_map.py` (Phase 8.E) currently treats cross-step
reads as opaque. With SSA names, it can:

- For each SSA read with a named writer, pin the live window to
  `writer_op -> reader_op` exactly (instead of "any prev-step").
- For `*` writer reads, fall back to today's conservative live window.

This refines the analyzer's cache compartmentalization without changing
runtime behavior — strictly higher eviction-rate ceiling.

### Phase 9.D — Detector heads use SSA naturally

When a `CONTROL_FLOW_DETECTOR_HEADS.md`-style detector head is added,
its outputs (e.g. `PC_VIA_LEV_DETECTOR_LO`) are fresh dims with
`step_offset=0`. Downstream readers spell their reads as
`PC_VIA_LEV_DETECTOR_LO` (no SSA suffix needed). The detector's *own*
V-slot reads attend back to a prev-step marker row via
`OUTPUT_LO.*.-1` etc. — same SSA spelling as today's demo op.

## 8. Risks and open questions

### R1 — Writer-name churn

Renaming a producer op (e.g. `layer16_lev_routing` -> `layer16_lev`)
breaks every reader that names it in an SSA read. Mitigation: use the
`*` wildcard wherever the dep graph doesn't actually need the producer
identity. For named writers, add a `decl_verifier` check that
``writer_op`` resolves to a known op at compile time.

### R2 — Round-trip identity in `produces` / `consumes_fresh`

The staleness invariants (`Operation.produces`, `consumes_fresh`) map
dim names to register identifiers. The SSA form should be normalized
to its base dim for these dictionaries — the staleness analyzer cares
about the band, not the version. The prototype does not enforce this
yet; Phase 9.B should add the normalization at lookup time.

### R3 — Migration cost vs status quo

The PREV_STEP alias trick already works; it is "ugly but functional".
SSA is the *right* notation but the rename is mechanical busywork.
Mitigation: bundle 9.B with the SCC-collapse work in
`PHASE_8_PLAN.md` §8.A — readers that already need cross-step audit
attention should land their PREV_STEP -> SSA rename at the same time.

### R4 — `*` wildcard hides real bugs

A reader that should pin to a single writer but uses `*` will silently
accept any prev-step value. Mitigation: the decl verifier can lint for
`*` reads of bands with only one producer (e.g.
`EMBED_LO.*.-1` should be flagged because L4 is the unique writer).

## 9. References

- `c4_release/neural_vm/unified_compiler/ssa_dim.py` — parser + schema.
- `c4_release/neural_vm/unified_compiler/layer_compiler.py:758-790,
  1346-1359` — scheduler hook.
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1872-2001` —
  demo op.
- `c4_release/tests/test_ssa_dim.py`,
  `c4_release/tests/test_ssa_scheduler.py`,
  `c4_release/tests/test_ssa_demo_op.py` — tests.
- `c4_release/docs/PHASE_8_PLAN.md` §8.A (PREV_STEP), §8.H (demo op),
  §7 (V2 vision).
- `c4_release/docs/CONTROL_FLOW_DETECTOR_HEADS.md` — companion V2 design
  spike; SSA naming makes detector-head reads self-documenting.
- `c4_release/neural_vm/dim_registry.py:610-870` — the existing
  PREV_STEP alias declarations Phase 9.B will retire.
