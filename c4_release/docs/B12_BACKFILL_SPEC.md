# B12 Backfill Spec — `phase_required_but_undeclared` ops

_Status: PREP for B12. Branch `speedup-cache-and-buckets`. Generated
2026-06-01 by the B12-prep sub-agent (read-only)._

This document is the per-op work order for B12 in
`DYNAMIC_SCHEDULER_MIGRATION_PLAN.md` §B12. For every op in the
`phase_required_but_undeclared` bucket (Phase A categorisation) it
specifies:

- the op factory file:line (`grep -n 'name="<op_name>"'` for the
  exact `return Operation(...)` site);
- the existing `phase` / `kind` / `layer_idx` / `reads` / `writes`;
- the **proposed patch** (a 3–5-line snippet that adds enough
  `requires` / `reads` / `consumes_fresh` to move the op into the
  `phase_pinned_by_deps` bucket OR explicitly to `freely_placeable`);
- the **bucket** (Backfill-now / Wait-on-B9-OUTPUT_HI /
  Wait-on-B9-ADDR_KEY / Wait-on-B9-OP_LEV / Wait-on-B10 / Manual);
- a **risk note**.

The helper `tools/survey_phase_undeclared.py` regenerates the per-op
table programmatically. Re-run it whenever an op is added/removed
from `all_core_ops`. Counts in this doc are pinned to the 2026-06-01
analyzer run (HEAD ~`990a571`).

## Count drift since Phase A

Phase A (commit `103a481`, 2026-06-01 morning) reported **27 ops** in
the bucket. A fresh analyzer run at HEAD (`990a571`) reports **26**.
The delta is:

- **left the bucket** (3): `layer14_clear_output_corruption` (now
  cycle member via newly declared OUTPUT_HI/OUTPUT_LO reads in
  predecessor ops), `layer10_stack0_byte_relay_bake` (cycle),
  `layer8_sp_gather_bake` (cycle).
- **entered the bucket** (2): `layer8_sp_gathered_sentinel`
  (B7-5 lifecycle bit op committed after Phase A),
  `layer15_si_mem_addr0_from_stack0` (post-A merge).

The B12 work is sized to ~27 ops because more ops will likely enter
during B9 dim splits. Treat 26 as the lower bound.

## Required upstream — B10 op-name `requires` schema

Many proposed patches use:

```python
requires={"after": "<op_name>"}            # strict dep, edge in DAG
requires={"same_layer_as": "<op_name>"}    # equal layer constraint
```

These are NOT honoured by `LayerCompiler._topological_sort` today —
they are the schema added by B10 (plan §B10). B12 cannot land until
B10 ships. Any op in the doc whose only proposed change is a
`requires` op-name reference is implicitly **gated on B10**.

## Bucket summary

| Bucket | Count | Trigger |
|--------|------:|---------|
| Backfill now (B10 only) | 18 | postop-attach + flag-gated stubs + early-layer threshold attns |
| Wait on B9 `OUTPUT_HI` split | 3 | `layer10_sp_byte_passthrough`, `layer15_si_mem_addr0_from_stack0`, plus any writer that gets a re-derived dep edge |
| Wait on B9 `ADDR_KEY` split | 1 | `layer14_clear_addr_key_pollution` (writes ADDR_KEY; depends on split being clean) |
| Wait on B9 `ADDR_B0_HI` cleanup (6.6) | 1 | `layer4_sp_to_addr_key` |
| Manual judgment | 3 | `layer2_initial_pc_bake_cancel` (EMBED_LO/HI overwrite), `layer1_threshold_attn` / `layer2_threshold_attn` (head IDs structurally pinned) |

The buckets overlap: an op needing both a B10 `requires` and a
B9-split dim read appears in the higher-numbered bucket.

## Per-op specs

The order is **descending current_layer** (latest pin first; same
order as `tools/survey_phase_undeclared.py`).

---

### 1. `l15_attention_resize`

- File: `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:1240`
  (`name="l15_attention_resize"` at line 1241).
- Current: `phase=14.9`, `kind="block"`, `layer_idx=15`,
  `reads=set()`, `writes=set()`, in=0, out=0.
- `declarative_authority="structural_model"` — a structural pass.
- **Bucket**: Backfill-now (B10).
- **Proposed patch** (add to the `return Operation(...)` block):
  ```python
  requires={"after": "layer15_nibble_copy"},
  ```
  Optionally also `same_layer_as="layer15_nibble_copy"`. The op
  resizes attention on layer 15 after L15 nibble-copy has baked its
  weights — pure structural cleanup.
- **Risk**: very low. The op writes no dims and has no consumers
  in the DAG, so an explicit `requires["after"]` cannot create
  new cycles. If `layer15_nibble_copy` itself moves earlier under
  the dynamic scheduler, this op follows.

---

### 2. `layer15_si_mem_addr0_from_stack0`

- File: `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:463`.
- Current: `phase=15.25`, `kind="block"`, `layer_idx=15`,
  `reads={MARK_MEM, MEM_STORE, MEM_ADDR_SRC, STACK0_BYTE0,
  CLEAN_EMBED_LO, CLEAN_EMBED_HI, CONST}`,
  `writes={OUTPUT_LO, OUTPUT_HI}`, in=1 (`layer1_ffn`), out=8.
- dep_depth=2, current_layer=15 → gap=13.
- **Bucket**: Wait on B9 `OUTPUT_HI` split (rename).
- **Why**: this op writes OUTPUT_LO/HI in the same cycle as
  L3..L14 producers; its 8 successors include
  `layer3_carry_forward_attn` (prev-step reader). Once the
  OUTPUT_HI rename lands (plan §6.1) the writes become
  `OUTPUT_HI_THIS_STEP`, the carry-forward edge resolves as
  `OUTPUT_HI_PREV_STEP`, and the actual dep-required layer becomes
  visible (likely 15).
- **Proposed patch (post-B9)**:
  ```python
  # After B9 §6.1 rename:
  writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
  requires={"after": "layer15_memory_lookup"},
  ```
- **Risk**: MED. This op is the SI (store-imm) path's STACK0→ADDR0
  fallback. Pinning it strictly after `layer15_memory_lookup`
  matches today's static layout; pinning it earlier breaks the
  prev-step residual chain. Keep `layer_pin=15` (B11 mitigation)
  during the strict-mode flip.

---

### 3. `layer14_clear_addr_key_pollution`

- File: `c4_release/neural_vm/unified_compiler/ops/l14_ops.py:823`.
- Current: `phase=14.2`, `kind="block"`, `layer_idx=14`,
  `reads={MEM_VAL_B0..B3, MARK_PC, MARK_BP, MARK_AX, MARK_STACK0,
  MARK_SP, CONST}`, `writes={ADDR_KEY}`, in=1
  (`layer2_mem_byte_flags`), out=8.
- dep_depth=3, current_layer=14 → gap=11.
- **Bucket**: Wait on B9 `ADDR_KEY` split.
- **Why**: this op rewrites ADDR_KEY at end-of-layer cleanup, but
  the DAG can't see why it has to run AFTER every L5..L13 ADDR_KEY
  user. The 8 successors (L4_pc_relay, L5_fetch, L8 multibyte
  fetch, L9 alibi mem, L15 mem lookup, etc.) all read it. The L5
  fetch path is in the cycle that breaks at B9 §6.3
  (`ADDR_KEY` → `ADDR_KEY_THIS_STEP` / `ADDR_KEY_FOR_NEXT_STEP`).
- **Proposed patch (post-B9 §6.3)**:
  ```python
  # After B9 §6.3 split:
  writes={"ADDR_KEY_FOR_NEXT_STEP"},  # cleanup for next step's fetch
  requires={"after": "layer14_mem_generation"},
  ```
- **Risk**: HIGH. This op is the canonical end-of-step ADDR_KEY
  clear (see investigation/bd-dim-usage-map). Moving it earlier
  breaks every same-step consumer. Keep `layer_pin=14` until
  strict mode is proven.

---

### 4-9. `l8_alu_postop_attach` through `l13_alu_postop_attach`

- File: `c4_release/neural_vm/unified_compiler/ops/alu_ops.py`
  (factories at lines 220, 226, 232, 238, 244, 250). Construction
  body at `c4_release/neural_vm/unified_compiler/ops/shared.py:141`
  (`_make_alu_postop_attach_op`).
- Current: `phase=1180+layer_idx*0.01`, `kind="block"`,
  `layer_idx=N`, `reads=set()`, `writes=set()`, in=0, out=0.
- **Bucket**: Backfill-now (B10).
- **Why this is mechanical**: the magic phase value `1180+layer*0.01`
  encodes "must run after every FFN bake and the dead-unit zero
  passes (`l6_dead_unit_zero=1160`, `l7_dead_unit_zero=1170`) but
  before `right_size_ffns=1200`." With B10 these become explicit
  `same_layer_as` references to the wrapped ALU op. The plan §R6
  calls these out specifically.
- **Proposed patch** (per-op, in `_make_alu_postop_attach_op`):
  ```python
  # In shared.py _make_alu_postop_attach_op, before `return Operation(...)`:
  alu_op_name = {
      "l8_alu_postop_attach": "layer8_alu",
      "l9_alu_postop_attach": "layer9_alu",
      "l10_alu_postop_attach": "layer10_alu",
      "l11_alu_postop_attach": "layer11_mul_partial",
      "l12_alu_postop_attach": "layer12_mul_combine",
      "l13_alu_postop_attach": "layer13_shifts",
  }[name]
  return Operation(
      ...,
      requires={
          "same_layer_as": alu_op_name,
          "after": "l7_dead_unit_zero",  # plan §B10 ordering
      },
      ...
  )
  ```
- **Risk**: LOW. These ops have zero declared dim activity; the
  only constraint is "same layer as the wrapped ALU, after the
  dead-unit-zero pass." Keep `phase=1180+...` as a tiebreaker via
  `layer_pin` during the B11 hybrid window.
- **Coordination**: all 6 ops live behind the same helper, so the
  edit is one factory change, not six. Single commit.

---

### 10. `layer10_sp_byte_passthrough`

- File: `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:831`.
- Current: `phase=10`, `kind="attn"`, `layer_idx=None` (anchor),
  `reads={IS_BYTE, HAS_SE, H1, BYTE_INDEX_0..2, CLEAN_EMBED_LO/HI}`,
  `writes={OUTPUT_LO, OUTPUT_HI}`, in=4, out=8.
- dep_depth=3, current_layer=10 → gap=7.
- **Bucket**: Wait on B9 `OUTPUT_HI` split (plan §6.1).
- **Why**: `declarative_authority="topology_anchor"` — the op is a
  placeholder; the real weight bake is owned by
  `layer10_sp_byte_passthrough_bake` at phase=10.2. The 7-layer gap
  is because the dep DAG only sees L0/L1/L2 inputs, while the
  static layout requires L10 placement so the bake's
  `OUTPUT_HI` write reaches the L11+ consumers via the unsplit
  dim. Post-B9-§6.1 rename, the `OUTPUT_HI_THIS_STEP` write at
  layer 10 pins it correctly.
- **Proposed patch (post-B9)**:
  ```python
  writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
  requires={"same_layer_as": "layer10_sp_byte_passthrough_bake"},
  ```
- **Risk**: LOW. Topology anchor; no weight bake. Moving it earlier
  is harmless to baked weights but breaks the per-op claim verifier
  if `layer=10` is hardcoded in its test. Audit during B11
  (plan §R3).

---

### 11. `layer8_sp_gathered_sentinel`

- File: `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1277`.
- Current: `phase=8.6`, `kind="block"`, `layer_idx=8`,
  `reads={MARK_SP}`, `writes={SP_GATHERED_THIS_STEP}`,
  `produces={"SP_GATHERED_THIS_STEP": "SP_marker"}`, in=0, out=0.
- dep_depth=0, current_layer=8.
- **Bucket**: Backfill-now (B10).
- **Why**: B7-5 lifecycle bit (added post-Phase A). The op writes
  `SP_GATHERED_THIS_STEP` but no current op declares
  `consumes_fresh={"SP_GATHERED_THIS_STEP": "SP_marker"}`, so the
  produces edge dangles. The docstring states it must run after
  `layer8_sp_gather_bake` (8.0), `layer8_multibyte_fetch_bake`
  (8.1), `layer8_alu` (8.2), `layer8_multibyte_routing` (8.3),
  `layer8_op_imm_relay` (8.4), `format_position_counter` (8.5).
- **Proposed patch**:
  ```python
  requires={
      "after": "layer8_multibyte_routing",  # last L8 writer to MARK_SP-adjacent dims
      "same_layer_as": "layer8_alu",
  },
  ```
  Plus: identify the L10 `tail_sp_marker_*` consumers
  (`null_terminator_detection` etc.) and add
  `consumes_fresh={"SP_GATHERED_THIS_STEP": "SP_marker"}` on each.
  This is properly out-of-scope for B12 (it modifies CONSUMERS) but
  noted here so the B9 dim audit can catch it.
- **Risk**: LOW. The op only writes a single sentinel bit; if the
  scheduler moves it to layer 8 anyway, behaviour is unchanged.

---

### 12-19. Convo-IO / tool-call / PRTF flag-gated stubs (8 ops)

Eight ops share the same shape: declared with empty
`reads={}` / `writes={}` because the bake body is
flag-gated and the dim activity only materialises when the flag
is on. The phase encodes "run after the corresponding L_n layer's
authoritative ops":

| Op | File:line | Current phase / layer | Flag |
|----|-----------|-----------------------|------|
| `convo_io_prtf_capture` | `flag_gated_ops.py:752` | 7.6 / 7 | convo_io |
| `prtf_think_protocol` | `l6_ops.py:3210` | 6.6 / 6 | think_protocol |
| `convo_io_state_machine` | `flag_gated_ops.py:385` | 6.6 / 6 | convo_io |
| `open_clos_tool_call` | `l6_ops.py:3293` | 6.7 / 6 | tool_calling |
| `convo_io_pc_sp_latch` | `flag_gated_ops.py:651` | 6.7 / 6 | convo_io |
| `convo_io_opcode_decode` | `flag_gated_ops.py:101` | 5.6 / 5 | convo_io |
| `convo_io_prtf_transport` | `flag_gated_ops.py:881` | 4.6 / 4 | convo_io |
| `layer3_convo_io_state_init` | `l3_ops.py:625` | 3.1 / 3 | convo_io |
| `convo_io_step_resume` | `flag_gated_ops.py:563` | 3.2 / 3 | convo_io |
| `layer5_user_input_gather` | `user_input_ops.py:23` | 5.7 / 5 | always |

(That table is 10 ops because L5 user input and L3 state init are
also stubs but pinned by `layer_idx`; included here for parity.)

- **Bucket**: Backfill-now (B10).
- **Proposed patch (per op)**: choose ONE of:
  - **(a) freely_placeable opt-in** — current behaviour is "pinned
    by `layer_idx`, no dim activity, no consumers." Mark explicitly
    so the analyzer stops flagging them:
    ```python
    requires={"freely_placeable": "true"},  # B10: opt-in marker
    ```
  - **(b) `requires["after"]` to the same-layer authoritative op**
    — when the docstring states "this must run after L_N's FFN":
    ```python
    requires={"after": "layer6_routing_ffn"},  # for L6-stage convo_io ops
    requires={"after": "layer5_fetch"},        # for L5-stage convo_io ops
    requires={"after": "layer3_ffn"},          # for L3-stage convo_io ops
    ```
- **Risk**: LOW per op. The ops have zero downstream consumers in
  declarations; whichever choice we make only affects the analyzer
  output, not the bake.
- **Recommended choice**: (b) for ops whose docstring names a
  specific predecessor (`layer3_convo_io_state_init` says "after
  L3 FFN", etc.); (a) otherwise. This keeps the dep DAG honest
  about the layer-N intent.
- **Coordination**: single commit per file (flag_gated_ops.py edit
  hits 6 ops; l6_ops.py hits 2; l3_ops.py / user_input_ops.py
  hit 1 each).

---

### 20. `layer5_user_input_gather`

- See §12-19; same pattern.

---

### 21. `layer4_sp_to_addr_key`

- File: `c4_release/neural_vm/unified_compiler/ops/l4_ops.py:364`.
- Current: `phase=4.5`, `kind="block"`, `layer_idx=4`,
  `reads={MARK_AX, BYTE_INDEX_0/1, H1, CLEAN_EMBED_LO/HI, CONST}`,
  `writes={ADDR_B0_HI, ADDR_B1_HI, ADDR_B2_HI}` (= ADDR_KEY band),
  in=3 (L0 threshold + L1 ffn + L2 mem byte flags), out=5
  (L8_mem_to_alu, L14_addr_key_neural_decode, L14_mem_generation,
  L16_lev_routing, tail_bit32_result_correction).
- dep_depth=3, current_layer=4 → gap=1.
- **Bucket**: Wait on B9 `ADDR_B0_HI` cleanup (plan §6.6).
- **Why**: declared `enable=False` (disabled by default; see the
  docstring at line 282). Its writes alias the ADDR_KEY band that
  is currently a back-edge dim. Once §6.6 lands the explicit
  `requires["after"] = "layer8_mem_to_alu"` on the L9/L13 ADDR_B0_HI
  readers, this op's layer constraint becomes derivable from the
  alias relationship.
- **Proposed patch (post-B9 §6.6)**:
  ```python
  requires={"after": "layer4_pc_relay"},  # docstring: "after pc_relay so its writes don't clobber"
  ```
- **Risk**: LOW. The op is disabled by default; the bake is a
  guard-clause no-op. The constraint is documented in the
  factory's docstring (line 282-285).

---

### 22. `layer3_convo_io_state_init`

- See §12-19 (l3 stub).

---

### 23. `convo_io_step_resume`

- See §12-19 (l3 stub).

---

### 24. `layer2_threshold_attn`

- File: `c4_release/neural_vm/unified_compiler/ops/l2_ops.py:231`.
- Current: `phase=2`, `kind="attn"`, `layer_idx=2`,
  `reads={IS_MARK, CONST}`, `writes={L2H0}`, in=0, out=2
  (`layer15_memory_lookup`, `layer8_mem_to_alu`).
- dep_depth=0, current_layer=2 → gap=2.
- **Bucket**: Manual judgment.
- **Why manual**: L2H0 is L2's head-0 output dim. The two consumers
  exist, so an L2-output edge IS in the DAG — but the gap of 2 is
  because no op produces a dim that L2 reads. The "structurally
  pinned at layer 2" constraint comes from the L2 attention block
  literally being layer 2 of the transformer stack. The dep model
  has no name for "this kind=attn op MUST live at block 2."
- **Proposed patch**: choose ONE of:
  - **(a) explicit threshold-chain requires**:
    ```python
    requires={"after": "layer1_threshold_attn"},
    ```
    This pins L2 strictly after L1; the dynamic scheduler now
    places it at layer ≥ 2. (Two layers below the static layout's
    layer-2 — acceptable since L2H0 consumers are layer-8 and
    layer-15, both later.)
  - **(b) `layer_pin=2` via the B11 mitigation field** (`R5`):
    documents the structural intent.
- **Recommended**: (a). The `layer1_threshold_attn → layer2_threshold_attn`
  edge is what the layer-1-vs-2 distinction means in practice
  (head IDs L1H0..L1H4 vs L2H0 sharing the same threshold-attn
  primitive but at different layers).
- **Risk**: MED. If the dynamic scheduler moves this earlier than
  layer 2, the L2 attention block compiles without L2H0 written;
  weight cells move. Guard with `layer_pin=2` until strict mode
  is proven.

---

### 25. `layer2_lookback_detection_head`

- File: `c4_release/neural_vm/unified_compiler/ops/l2_ops.py:299`.
- Current: `phase=2.1`, `kind="block"`, `layer_idx=2`,
  `reads={CONST, IS_BYTE, MARK_THINKING_START/END}`,
  `writes=set()`, in=0, out=0.
- **Bucket**: Backfill-now (B10) — convo-io stub pattern (see
  §12-19).
- **Proposed patch**:
  ```python
  requires={"after": "layer2_threshold_attn"},  # docstring says so
  ```
- **Risk**: LOW.

---

### 26. `layer2_initial_pc_bake_cancel`

- File: `c4_release/neural_vm/unified_compiler/ops/l2_ops.py:186`.
- Current: `phase=2.5`, `kind="block"`, `layer_idx=2`,
  `reads={MARK_PC, HAS_SE}`, `writes={EMBED_LO, EMBED_HI}`, in=1
  (`layer1_threshold_attn` via HAS_SE), out=16 (every L3+ op that
  reads EMBED_LO/HI).
- dep_depth=1, current_layer=2 → gap=1.
- **Bucket**: Manual judgment.
- **Why manual**: this op WRITES `EMBED_LO/HI` — but it's a
  cancellation pass (zeroing the initial-PC token-embedding bake).
  The 16 successors all read EMBED_LO/HI, but the L3+ consumers
  want to see the CANCELLED value (not the L0 embedding write).
  The dep model can't distinguish "L0 wrote the initial PC into
  EMBED" from "L2 cancelled it"; both write the same dim. The
  static-phase ordering disambiguates by `phase=2.5 > 0.0`.
- **Proposed patch (post-B10)**:
  ```python
  requires={"after": "phase_a_ffn"},  # cancels phase_a's PC bake
  produces={"EMBED_LO": "MARK_PC", "EMBED_HI": "MARK_PC"},
  ```
  The `produces` declaration with the MARK_PC register marks this
  op as the canonical writer at MARK_PC (one row of the residual
  band); other ops that read EMBED at OTHER markers don't need a
  `consumes_fresh` edge.
- **Risk**: HIGH. If the scheduler picks an earlier layer for this
  op (e.g. layer 1, same as `phase_a_ffn`), the L3 FFN's PC
  INCREMENT reads the still-bake-on EMBED_LO/HI and the PC
  computation breaks. See the factory docstring at line 90-94 for
  the canonical "why phase=2.5" explanation. Keep `layer_pin=2`
  until strict mode is proven.

---

### 27. `layer1_threshold_attn`

- File: `c4_release/neural_vm/unified_compiler/ops/l1_ops.py:201`.
- Current: `phase=1`, `kind="attn"`, `layer_idx=1`,
  `reads={IS_MARK, MARK_SE_ONLY, MARK_CS, CONST}`,
  `writes={L1H0..L1H4, HAS_SE, IN_STEP_FRESH}`, in=0, out=29.
- dep_depth=0, current_layer=1 → gap=1.
- **Bucket**: Manual judgment.
- **Why manual**: same shape as `layer2_threshold_attn` (§24): no
  predecessor in the dep graph, but the layer-1 pin is structural
  (`HAS_SE` is the "step has at least one SE token" signal which
  must propagate from layer 1 forward; consumers at layer 2-15
  all assume the value lands at residual layer 1).
- **Proposed patch**:
  ```python
  requires={"after": "layer0_threshold_attn"},
  ```
- **Risk**: LOW. Static layout already runs L1 strictly after L0
  (different transformer blocks). The dynamic DAG just doesn't see
  it because L0 doesn't write any dim L1 reads — L0 writes
  IS_MARK / MARK_* which feed into L1's READS. Wait — IS_MARK is
  in L1.reads… so L0 → L1 SHOULD be an edge. Verify in the
  analyzer: if `layer0_threshold_attn` writes IS_MARK, the
  predecessor list should include it. The fact that
  `in_degree == 0` suggests L0 doesn't declare a write of IS_MARK
  — that's a missing **L0** declaration, not L1's. **Action**:
  audit L0's writes set during B12; the actual missing edge may
  belong to L0.

---

## Bucket-ordered worklist (for parallel agents)

### Bucket A — Backfill now (after B10) — 18 ops, mostly mechanical

Each row is one parallel-agent task.

1. `l8_alu_postop_attach` … `l13_alu_postop_attach` (6 ops, single
   `shared.py:141` edit — ONE commit, NOT six).
2. `convo_io_opcode_decode` (`flag_gated_ops.py:101`).
3. `convo_io_state_machine` (`flag_gated_ops.py:385`).
4. `convo_io_pc_sp_latch` (`flag_gated_ops.py:651`).
5. `convo_io_prtf_capture` (`flag_gated_ops.py:752`).
6. `convo_io_prtf_transport` (`flag_gated_ops.py:881`).
7. `convo_io_step_resume` (`flag_gated_ops.py:563`).
8. `prtf_think_protocol` (`l6_ops.py:3210`).
9. `open_clos_tool_call` (`l6_ops.py:3293`).
10. `layer3_convo_io_state_init` (`l3_ops.py:625`).
11. `layer5_user_input_gather` (`user_input_ops.py:23`).
12. `layer2_lookback_detection_head` (`l2_ops.py:299`).
13. `layer8_sp_gathered_sentinel` (`l8_ops.py:1277`).
14. `l15_attention_resize` (`l15_ops.py:1240`).

### Bucket B — Wait on B9 dim splits — 5 ops

1. `layer10_sp_byte_passthrough` — B9 §6.1 (`OUTPUT_HI` rename).
2. `layer15_si_mem_addr0_from_stack0` — B9 §6.1.
3. `layer14_clear_addr_key_pollution` — B9 §6.3 (`ADDR_KEY` split).
4. `layer4_sp_to_addr_key` — B9 §6.6 (`ADDR_B0_HI` cleanup).

### Bucket C — Manual judgment — 3 ops

1. `layer1_threshold_attn` — audit L0 writes set FIRST.
2. `layer2_threshold_attn` — same shape; pick between
   `requires["after"]` and `layer_pin`.
3. `layer2_initial_pc_bake_cancel` — confirm `produces` with
   register-aware semantics is the right model.

---

## Top 3 highest-complexity ops

1. **`layer14_clear_addr_key_pollution`** (Bucket B, §3 in spec).
   8 successors, ADDR_KEY write, gap=11. Strict mode risks every
   downstream ADDR_KEY consumer reading uncleared state. Needs B9
   §6.3 split to land cleanly. **Risk: HIGH.**
2. **`layer2_initial_pc_bake_cancel`** (Bucket C, §26). 16
   successors via EMBED_LO/HI, register-aware `produces` needed,
   wrong placement breaks L3 PC INCREMENT. **Risk: HIGH.**
3. **`layer15_si_mem_addr0_from_stack0`** (Bucket B, §2). 8
   successors including `layer3_carry_forward_attn` (cross-step
   prev-step reader); the OUTPUT_HI rename has to land first.
   **Risk: MED.**

## Bottom 3 trivial one-liners

1. **`l15_attention_resize`** (Bucket A). No deps, no writes,
   single `requires["after"]` line.
2. **`l8..l13_alu_postop_attach`** (Bucket A, six ops, one helper
   edit). Same line in `shared.py` patches all six.
3. **`layer2_lookback_detection_head`** (Bucket A). Empty writes,
   single `requires["after"] = "layer2_threshold_attn"`.

---

## Parallelisation feasibility

B12 CAN be done as ~14 parallel agents (one per Bucket A row,
one per Bucket B op, one per Bucket C op), with these caveats:

- **`shared.py:141` edit** patches all 6 alu_postop_attach ops in
  one commit. That commit is a single agent, not 6.
- **B9 ordering**: Bucket B ops cannot land until B9 has shipped
  the relevant dim split. Schedule as a second wave.
- **Bucket C ops** require coordination because the proposed
  patches modify consumer ops too (e.g.
  `layer1_threshold_attn` needs L0 declarations audited). These
  belong on a single agent or to a follow-up B12.1.

Final agent breakdown:
- 1 agent for the `shared.py` postop-attach helper edit (6 ops).
- 10 agents for the remaining Bucket A ops, 1 per op.
- 4 agents for Bucket B (after B9 lands for each dim).
- 1 agent for the Bucket C audit (3 ops + L0 follow-up).
- = **~16 agents total**, executed in 2 waves (Bucket A in
  parallel; Bucket B once B9 lands).

## Refined effort estimate

Plan §B12 estimate: **2.0 agent-days**.

With this spec acting as a per-op work order:

| Bucket | Ops | Per-op effort | Sub-total |
|--------|----:|---------------|----------:|
| A (mechanical) | 14 | 0.05d (15 min: edit + verify) | 0.7d |
| B (gated on B9) | 5 | 0.15d (45 min: edit + verify + per-op test) | 0.75d |
| C (manual + audit) | 3 | 0.4d (2h: docstring read + edit + analyzer re-run) | 1.2d |
| Spec / coordination | — | 0.25d | 0.25d |
| **Total** | **22 ops** | | **2.9d** |

(22 vs 26 ops because the 6 postop-attach ops collapse into 1
edit, and 4 of the 18 Bucket-A ops are nearly free single-liners
batched in the same commit.)

**Refined estimate: 2.5–3.0 agent-days.** The plan's 2.0d figure
is achievable for Bucket A alone; Bucket B/C add ~1 day. Recommend
**2.5d** as the new B12 estimate.

---

## Cross-references

- Survey script: `c4_release/tools/survey_phase_undeclared.py`
  (committed alongside this doc; re-run after any
  `all_core_ops`-touching commit).
- Migration plan: `c4_release/docs/DYNAMIC_SCHEDULER_MIGRATION_PLAN.md`
  (B9 §6.1-§6.6 for dim splits; B10 for `requires` schema; B12 for
  this work).
- Phase A diagnostic:
  `.agent-logs/scheduler_phase_a_2026_06_01.md` (Phase A baseline,
  27 ops) and the fresh run at the same path (26 ops as of HEAD).
- LayerCompiler: `c4_release/neural_vm/unified_compiler/layer_compiler.py`
  (`Operation` at line 87; `_topological_sort` at line 1030;
  `phase` at line 136).
- B7-5 sentinel context: `layer8_sp_gathered_sentinel` factory
  docstring (`l8_ops.py:1232`).
