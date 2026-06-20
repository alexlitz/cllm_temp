# Positional-invariant audit + the `positional_invariant=STEP_TOKENS` annotation

2026-06-20. Tooling-only (model byte-identical, golden `4958b35b`).

## The bug class this addresses (THE #1 campaign root)

The `C4_NO_STACK0_EMIT` campaign collapses the per-step token block from
`STEP_TOKENS=35` to `30` by dropping the 5-token STACK0 register block.
A large fraction of the hand-authored weights encode **positional / distance
logic at a fixed token offset**:

- `BYTE_INDEX_0..3` — byte-index-within-register flags.
- `STACK0_BYTE0..3` — STACK0 byte-position flags (the *dropped* block).
- `MEM_VAL_B0..3` — "MEM value byte N" slot predictors, defined as
  `d=N-from-MEM` distance flags.
- `H0..H7`, `L1H0..L1H4`, `L2H0` — marker-*distance* threshold heads
  ("marker within dist 4.5", "threshold 5.5 from nearest IS_MARK"), and
  the `+<marker_idx>` offset *into* those 7-wide distance banks.

The IR carries no record that a rule reading `STACK0_BYTE1` or
`BD.L2H0 + MEM_I` **assumes the 35-token frame**. When `STEP_TOKENS=30`,
every such marker lands 5 tokens off its intended physical row and the rule
silently mis-fires. Nothing catches this statically — it has surfaced only
via multi-hour GPU-agent debugging.

### Confirmed roots of this class (this session)

| Root | Where | Positional dim | Failure |
|------|-------|----------------|---------|
| div/mod dividend high byte | `efficient_alu_neural.py:176-221` | `STACK0_BYTE1` (cummax anchor) | high byte gathered from the wrong post-shift row |
| var-multi-local operand-CAM | `l8_ops.py make_layer8_mem_to_alu_op` | `L2H0+MEM_I`, `H1+MEM_I`, `MEM_VAL_B*` (`d=6-from-MEM`) | value-byte-0 row predicate moves when the frame shrinks |
| STACK0_BYTE0 framing leak | `l1_ops` (already hand-neutralized via `no_stack0_emit_enabled()`) | `STACK0_BYTE0` | the d=6-from-BP slot becomes MEM addr byte 0 |

## The deliverable: `tools/lint_positional_invariants.py`

A pure-static AST audit (no model build) that:

1. Derives the **positional-frame dim set** from `dim_registry.py` itself —
   every `_pin(NAME, ..., "<description>")` whose description matches a
   distance / byte-index / `d=N-from` / "within dist" / "from nearest
   IS_MARK" phrase — unioned with a hand-seeded core set. This keeps the
   audit in lockstep with the registry: a newly added distance-bank dim is
   picked up automatically (the derive independently re-confirms 25 of the
   29 seeded dims from the registry text, proving the sync is live).
2. Finds every reference to one of those dims across the per-layer
   `ops/*.py`, `efficient_alu_neural.py`, `vm_step.py`, `setup_helpers.py`,
   in three forms: string literals (`"STACK0_BYTE1"`, `"H1+4"`), attribute
   reads (`BD.L2H0`, `proxy.STACK0_BYTE1`), and `dim_ref("byte_index","0")`
   semantic calls.
3. Attributes each ref to its enclosing `def` + owning `make_*`/`_layer*`
   op factory, records `file:line`, the dim, and whether that op consults
   `no_stack0_emit_enabled()` (the campaign guard).

### Ranking

- **`UNGUARDED`** — references a positional-frame dim AND the enclosing op
  does NOT consult `no_stack0_emit_enabled()`. *Will silently shift/break
  under the campaign.* This is the shift-risk surface — the next bugs.
- **`CAMPAIGN_AWARE`** — references a positional dim AND the op consults the
  guard (built for / neutralized under the campaign). Lower priority, but
  flagged because it may still hand-compute a raw distance offset.
- A `DISTANCE_OFFSET` sub-tag marks any ref that carries a raw `+offset`
  (`BD.L2H0 + MEM_I`, `"H1+4"`) — an offset *into* a distance bank is doubly
  frame-dependent.

### Current catalog (golden `4958b35b`, default-OFF build)

```
references: 2146 total | 1993 UNGUARDED (813 with a distance offset) | 153 CAMPAIGN_AWARE
```

Top UNGUARDED+DISTANCE_OFFSET surface (the next re-anchor wave):

```
49  _layer14_mem_generation_head_specs
33  _layer14_mem_generation_head_specs_with_overrides
26  _layer15_memory_lookup_heads_0_3_specs_with_overrides
15  _layer7_memory_head_specs
12  _layer14_alu_high_byte_relay_spec
11  _layer10_bp_byte_passthrough_head_spec
10  _layer10_psh_stack0_passthrough_head_spec
10  _layer8_sp_gather_head_specs
```

`python tools/lint_positional_invariants.py --prove` self-checks that the
catalog includes the div/mod `STACK0_BYTE1` anchor (flagged UNGUARDED) and
the operand-CAM `L2H0`/`H1`/`MEM_VAL_B*` markers (flagged with
`DISTANCE_OFFSET`).

### Usage

```
python tools/lint_positional_invariants.py                 # ranked catalog
python tools/lint_positional_invariants.py --json
python tools/lint_positional_invariants.py --unguarded-only
python tools/lint_positional_invariants.py --dim STACK0_BYTE1
python tools/lint_positional_invariants.py --prove
```

## The fuller fix (design — NOT in this deliverable)

The audit *finds* the surface; the systematic fix is to make the frame
assumption **declared and compiler-enforced** rather than implicit.

### Proposal: a `positional_invariant` annotation on rules / head specs

Add an optional field to `FFNRule` / `DeclarativeAttentionHeadSpec`:

```python
positional_invariant: Optional[str] = None   # e.g. "STEP_TOKENS=35"
```

A rule that reads a positional-frame dim or a `+<offset>` into a distance
bank declares the frame it assumes. The compiler then:

1. **Warns / errors** at build time if a positional-frame dim is referenced
   by a rule with `positional_invariant=None` (the audit's UNGUARDED set
   becomes a hard ratchet, mirroring `lint_position_role.py`'s baseline).
2. **Auto-shifts** the offset when the active `Token.STEP_TOKENS` differs
   from the declared invariant: a `STACK0_BYTE1` anchor or a `d=N-from-MEM`
   slot whose declared frame is 35 is re-pointed by `−5` rows (or its
   distance-bank offset recomputed) when the campaign drops to 30, instead
   of silently reading the wrong row.

This turns the entire `UNGUARDED` catalog into a one-time annotate-and-shift
migration: each op declares its frame, the compiler does the re-anchor. The
two confirmed roots (div/mod, operand-CAM) become the first two annotations.

### Migration order (seeded by this audit)

Re-anchor in descending `UNGUARDED + DISTANCE_OFFSET`-by-op order (the table
above): L14 mem-generation/relay heads first (82 refs across two specs),
then L15 memory-lookup, then L7/L8 operand gather, then the L10 passthrough
family. Each op:

1. Run `lint_positional_invariants.py --dim <dim>` to enumerate its refs.
2. Add `positional_invariant="STEP_TOKENS=35"` and let the compiler shift,
   OR explicitly gate the offset on `no_stack0_emit_enabled()` (the
   `_layer16_lev_routing_rules` precedent).
3. Re-run the audit: the op moves from `UNGUARDED` to `CAMPAIGN_AWARE`.

The audit is the catalog; the annotation is the systematic fix.
