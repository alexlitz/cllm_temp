# Phase 10.E + 10.F dim-multiplexing feasibility audit

Date: 2026-06-02
Scope: audit the savings estimates in `docs/PHASE_8_PLAN.md` section 11.E/F
(commit a33139de) against the actual declarative IR.
Read-only — no code changes.

## TL;DR

**Both Axis E and Axis F deliver ~0 M savings at the current IR.** The plan's
-78 M / -138 M estimates assume two preconditions that the corpus does not
satisfy:

1. **Every dim has a derivable "active opcode class".** False. 692 / 733
   residual bytes (94%) cannot be opcode-classified from the IR today.
   Either no op-IR rule writes them with an `OP_*` condition, or the writing
   op is imperative (no IR at all).
2. **Every attention head that reads a multiplexable dim filters its K-side
   on the target opcode marker.** False. 76 / 82 attention heads (93%) are
   NON_FILTERING: they gate K on token-type markers (`MARK_PC`, `MARK_AX`,
   `IS_BYTE`, byte_index) rather than on `OP_*` markers.

Recommendation: **park 10.E and 10.F** at this revision. Both axes are
blocked on declarative-IR migration work (Phase 7.C / 7.E) that is itself
in progress. The plan's savings estimates over-credit feasibility by an
order of magnitude relative to what the IR currently expresses.

## Methodology

- Built the dynamic dim registry via `build_default_registry_dynamic()`
  (`d_model = 736`, 178 named slots covering 733 unique residual bytes —
  matches the plan's "~733 active dims" figure).
- Enumerated all 124 ops via `all_core_ops()` × default + all-flags-on flavors.
- For each op with `compiler_ir` / `compiler_ir_factory`, walked every
  `FFNRule` and `DeclarativeAttentionHeadSpec` to extract:
  - Per-byte opcode-condition set (set of `OP_*` / `ACTIVE_OPCODE_*` /
    `IO_IS_*` markers appearing in `conditions` / `gate_terms` / `gate`)
  - Per-head K-side opcode filter (set of `OP_*` markers appearing in
    `spec.k` projection writes)
- For imperative ops (no IR), conservatively marked every byte in
  `op.reads ∪ op.writes` as ALWAYS_ACTIVE (we cannot prove opcode safety
  without seeing the bake body).
- Built two interference graphs (E: FFN-only mergeable bytes; F: all
  mergeable bytes); greedy-colored each via Welsh-Powell.
- Projected param counts using
  `params(d) = 4·d²·n_blocks + n_units·d·3 + vocab·d·2`
  with `n_blocks = 32, n_units = 44 547, vocab = 320`.

## Per-byte bucket distribution (corpus-wide, 733 bytes)

| Bucket             | Count |  Notes                                          |
| ------------------ | ----- | ----------------------------------------------- |
| ALWAYS_ACTIVE      | 692   | No opcode condition derivable                   |
| OPCODE_SPECIFIC    |  41   | Exactly one `OP_*` marker conditions all writes |
| MULTI_OPCODE       |   0   | (Greedy color collapsed multi → 1 group)        |

Of the 692 ALWAYS_ACTIVE bytes:

- **353** are touched by **imperative ops** (no IR). 58 of 124 ops
  (47%) bake directly via `set_layerN_*` helpers — `op.reads` / `op.writes`
  give us the dim names but not the opcode-class.
- **283** are written by IR rules whose `conditions` lack any `OP_*` /
  `ACTIVE_OPCODE_*` / `IO_IS_*` marker. Many such rules gate on
  `MARK_AX` + `BYTE_INDEX_0` — token-type, not opcode-type. These rules
  do exist within an opcode-specific call site (e.g. `ALU_LO` is
  conceptually ADD/SUB-only) but the IR does not encode that.
- **341 bytes are never written** by any IR-visible writer at all
  (likely covered by imperative bakes or genuinely dead).

Of 27 331 FFN rules in IR-visible ops, **22 953 (84%)** include at least
one `OP_*` condition. The remaining **4 378 (16%)** gate on token-type
markers only — these are the source of most ALWAYS_ACTIVE buckets.

## Per-head bucket distribution (82 declarative heads)

| Bucket             | Count |  Notes                                                |
| ------------------ | ----- | ----------------------------------------------------- |
| OPCODE_FILTERING   |   6   | K projection touches an `OP_*` marker                 |
| NON_FILTERING      |  76   | K projection touches no opcode marker (MARK_*-only)   |

Non-filtering heads collectively V-read **330 / 733 bytes** (45% of
residual area). Under Axis F, every byte they touch must remain
non-multiplexed.

Top NON_FILTERING heads by # V-bytes read:

| Op                                       | head | # V-bytes |
| ---------------------------------------- | ---- | --------- |
| layer14_mem_generation                   | 0-3  | 65 each   |
| layer10_psh_stack0_passthrough_bake      | 3    | 64        |
| layer4_pc_relay                          | 0-1  | 48 each   |
| layer6_relay_heads_bake                  | 7    | 48        |
| layer6_relay_heads_bake                  | 6    | 39        |
| layer6_attn_bake                         | 0, 2 | 33 each   |
| layer8_sp_gather_bake                    | 0-2, 6 | 33 each |

The L14 `mem_generation` and L10 `psh_stack0_passthrough` heads are the
worst — each reads ~64 bytes spanning many dim families simultaneously
(`ADJ_CARRY`, `CONST`, `MUL_ACCUM`, `OUTPUT_HI`, `OUTPUT_LO`, etc.).
Any Axis F packing has to leave that residual fragment intact.

## Interference graphs and chromatic numbers

### Axis E (FFN-only dims, strict)

A byte is an Axis E candidate iff:
- not in `attn_read_bytes` AND not in `attn_write_bytes` (FFN-only), AND
- not touched by any imperative op (IR fully governs its writes), AND
- every IR rule writing it has at least one `OP_*` condition (provable
  opcode class).

| Quantity                                   | Value |
| ------------------------------------------ | ----- |
| FFN-only covered bytes                     |  245  |
| FFN-only mergeable (opcode-classified)     |    1  |
| FFN-only ALWAYS_ACTIVE (irreducible)       |  244  |
| Greedy chromatic # on mergeable subgraph   |    1  |

Projected `d_model` after Axis E: **733** (vs baseline 736, alignment-equal
at 736). **Savings: 0 M.**

### Axis F (validity-mask, all dims)

A byte is an Axis F candidate iff:
- not in ALWAYS_ACTIVE bucket, AND
- every reading head is OPCODE_FILTERING with a K-filter intersecting the
  byte's opcode set.

| Quantity                                   | Value |
| ------------------------------------------ | ----- |
| Mergeable bytes (all dim classes)          |    6  |
| F-irreducible bytes                        |  727  |
| Greedy chromatic #                         |    1  |

Projected `d_model` after Axis F: **728** (alignment-equal at 736).
**Savings: 0 M.**

## Param-count projection vs. plan's targets

| Configuration                       | d_model | Total params | vs. baseline |
| ----------------------------------- | ------- | ------------ | ------------ |
| Baseline                            |  736    |   168.2 M    |     —        |
| Plan target Axis E                  | ~500    |   ~ 90 M     |   −78 M      |
| Plan target Axis F                  | ~256    |   ~ 30 M     |   −138 M     |
| **Axis E projection from data**     |  736    |   168.2 M    |    0 M       |
| **Axis F projection from data**     |  736    |   168.2 M    |    0 M       |

(Note: baseline here is **168.2 M**, not the doc's 184 M / 167.7 M
"baseline" — those reflect different `n_units` / `n_blocks` snapshots.
The savings DELTA is unaffected: both axes deliver ~0 with the current
IR encoding.)

## Why the plan's estimates miss

1. **Imperative-op coverage gap (47% of ops).** Phase 7.C / 7.D have not
   migrated `layer4_pc_relay`, `layer6_relay_heads_bake`,
   `layer8_sp_gather_bake`, `layer10_psh_stack0_passthrough_bake`,
   `layer14_mem_generation`, and most L15/L16 ops. Their dim usage is
   declared at op-level (`op.reads` / `op.writes`) but the per-rule
   opcode-class info lives in imperative weight writes — invisible to
   the multiplexer.

2. **Token-type gating, not opcode gating.** Rules like the ALU_LO
   nibble writes gate on `MARK_AX AND BYTE_INDEX_0` — they fire on
   every AX-side byte, regardless of opcode. The actual opcode-class
   restriction (ADD/SUB/MUL only) lives *implicitly* in the upstream
   `OP_ADD` flag being one-hot at the AX position. The multiplexer
   cannot infer this without static-analysis of the residual stream
   propagation (much harder than greedy coloring).

3. **Non-filtering attention heads are the norm.** The L0/L1/L2
   threshold heads, L4 PC relay, L6 routing, L8 SP gather, L14 mem
   generation, and L15 nibble copy heads ALL gate K on `MARK_*` /
   `IS_BYTE` / distance thresholds — never on `OP_*`. They run on
   every step regardless of opcode and need their full V/Q residual
   footprint preserved.

## Risk callouts

- **L14 `mem_generation`** is the single biggest blocker. Heads 0-3 each
  read ~65 V-bytes spanning ALU_LO, ALU_HI, MUL_ACCUM, OUTPUT_LO,
  OUTPUT_HI, ADJ_CARRY, CONST — these dims belong to disjoint logical
  opcode families (mul, add/sub, output) but the head reads them all
  unconditionally. Any Axis F multiplex of these dims would corrupt
  the L14 mem-generation output for opcodes that touch a subset.
- **L10 `psh_stack0_passthrough_bake` head 3** reads 64 V-bytes. Same
  story — touches OUTPUT_* and STACK0_* dims simultaneously.
- **L4 / L6 relay heads** (PC_relay, routing) carry MARK_PC- and
  OPCODE_FLAGS-conditioned values forward. They V-read OPCODE_FLAGS
  (34 bytes) wholesale — those bytes ARE opcode-specific but the head
  doesn't K-gate on individual opcodes, so the multiplexer can't safely
  collapse them.

## Recommendation

**Park both Axis E and Axis F in their current form.** Neither axis can
deliver meaningful savings until two prerequisite migrations land:

1. **Phase 7.C completion** — migrate the remaining 58 imperative ops
   (especially L4 PC relay, L6 routing, L8 SP gather, L10 PSH, L14 mem
   generation, L15 nibble copy) to declarative IR so per-byte opcode
   classes become derivable.

2. **Phase 7.E semantic dim refs + 7.X "rule scope tightening"** —
   migrate token-type-only conditions (`MARK_AX AND BYTE_INDEX_0`) into
   explicit `OP_<class>` conditions where the runtime restriction
   exists implicitly. Today's 4 378 no-opcode rules are the dominant
   source of ALWAYS_ACTIVE bytes; tightening them would unlock Axis E
   directly.

Realistic post-prereq estimate (very rough):

- If 7.C completes, the 353 imperative-touched bytes become
  IR-classifiable. If half are opcode-specific (matching the 84% rate
  in IR rules today), that adds ~150 bytes to the mergeable pool.
- If rule-scope tightening converts even 30% of the 283 no-opcode-cond
  bytes to opcode-specific, that's another ~85.
- Combined: maybe ~235 byte-cells potentially mergeable. With chromatic
  # in the 10-20 range (one slot per opcode class), that's a savings of
  ~200 bytes. **Refined Axis E ceiling: d_model 736 → ~530, ~50 M savings**
  — not the plan's 78 M, but in the same order.

For Axis F, the bottleneck is the 76 non-filtering heads. Migrating those
to K-filter on opcode markers (so the validity-mask trick works) is a
separate, much larger workstream — comparable in scope to a Phase 7-grade
migration. Until that lands, Axis F adds vastly more verification cost
than savings.

**Concrete next step:** if compression is the priority, focus on
**Axis 10.A (d_model packing / defragmentation)** first. The plan's
~70-75 M estimate there appears defensible from the registry layout:
178 named slots over 733 covered bytes in a 736-byte pool already has
modest fragmentation, and the alignment gaps could fold further with
the dynamic allocator's compaction pass.

## Files referenced

- `/home/alexlitz/Documents/misc/c4_release/c4_release/docs/PHASE_8_PLAN.md` (lines 950-1097)
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/dim_registry_dynamic.py`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/dim_allocator.py`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ir.py`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/primitives.py`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/unified_compiler/ops/all_core_ops.py`

Profiler script: `/tmp/profile_10ef_v2.py` (not committed; reproduce via
the methodology section). Raw JSON: `/tmp/profile_10ef_result.json`.
