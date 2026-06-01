# FFNRule migration pattern — Phase 6 wave 1C

The canonical recipe for replacing an imperative `_set_layerN_*` weight-write
loop with a declarative list of `FFNRule`s exposed via `Operation.compiler_ir`.
Companion document to `PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md`.

## 1. What this pattern accomplishes

Today many ops still bake by directly mutating `block.ffn.W_up`,
`block.ffn.W_gate`, `block.ffn.b_up`, and `block.ffn.W_down`:

```python
ffn.W_up[unit, BD.OP_LEV]  = S / 10
ffn.W_up[unit, BD.MARK_PC] = S
ffn.b_up[unit] = -S * 1.5
ffn.W_gate[unit, BD.CONST] = 1.0
ffn.W_down[BD.TEMP + 0, unit] = -5.0 / S
```

This pattern converts every such cell-write block into one or more
`FFNRule.constant_write` / `FFNRule.gated_write` declarations. The rules go
into a `FFNOp(rules=[...])` on a `CompilerIR`, which is attached to the
`Operation` via `compiler_ir=...`. At bake time, `layer_compiler` dispatches
the IR through `CompilerIR.lower_ffn` (or `Primitives.lower_ffn_rules`), which
performs the exact same cell writes the imperative helper used to do — but now
driven from a declaration the verifier, scope checker, and dominance auditor
can all read.

Once an op is fully declarative, its `bake_fn` body collapses to a single
`_lower_via_compiler_ir`-style call (or disappears entirely; `layer_compiler`
sees the `compiler_ir` and dispatches automatically).

## 2. Worked example — `layer14_temp_clear`

`layer14_temp_clear` is a single-unit cleanup op: at the PC marker on `OP_LEV`
rows, subtract a residual ~2.0 from `TEMP[0]` so the L16 TEMP -> OUTPUT routing
does not mis-amplify `OUTPUT_LO[0]`. Backfilled with explicit claims at commit
`e18885a`; small footprint; well-understood semantics.

### Pre-migration: imperative bake (`setup_helpers._set_layer14_temp_clear`)

```python
def _set_layer14_temp_clear(ffn, S, BD, start_unit=0):
    unit = start_unit
    # Clear TEMP[0] at PC marker when OP_LEV active
    ffn.W_up[unit, BD.OP_LEV]  = S / 10   # ~1 with OP_LEV~=10
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 1.5             # fire when OP_LEV + MARK_PC
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.TEMP + 0, unit] = -5.0 / S
    unit += 1
    return unit
```

Five cell writes, one hidden unit, expressing a single conditional:
**"if `OP_LEV * (1/10) + MARK_PC * 1 >= 1.5` then add `-5/S` to `TEMP[0]`"**.

### Post-migration: declarative `FFNRule`

```python
def _layer14_temp_clear_rules(S: float) -> tuple[FFNRule, ...]:
    return (
        FFNRule.constant_write(
            name="l14_temp_clear_pc_lev",
            conditions=(
                ("OP_LEV",  0.1),    # S * 0.1 ~= matches W_up[..., OP_LEV] = S/10
                ("MARK_PC", 1.0),    # S * 1.0 == W_up[..., MARK_PC] = S
            ),
            threshold=1.5,            # b_up = -S * threshold => -S * 1.5
            writes=(("TEMP+0", -5.0 / S),),
            scope="OP_LEV and MARK_PC",
            dominates_at={"TEMP+0": "OP_LEV and MARK_PC"},
        ),
    )


def _layer14_temp_clear_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer14_temp_clear_rules(S))
    return ir
```

And on the `Operation`:

```python
return Operation(
    name="layer14_temp_clear",
    phase=14.1,
    # ...reads / writes / kind unchanged...
    compiler_ir=_layer14_temp_clear_ir(),
    # bake_fn / declarative_bake_fn can be removed; layer_compiler will
    # dispatch via _dispatch_operation_ir -> ir.lower_ffn.
    layer_idx=14,
    migrated=True,
    claims={(14, "ffn_W_down", "0", "TEMP+0")},
    # ...
)
```

The lowering pipeline (`CompilerIR.lower_ffn`, `ir.py:706`) reproduces the five
imperative writes one-for-one:

| Imperative write                                  | Driven by FFNRule field                                   |
|---------------------------------------------------|-----------------------------------------------------------|
| `W_up[unit, OP_LEV] = S * 0.1`                    | `conditions=[("OP_LEV", 0.1)]`, multiplied by `S` in lowerer |
| `W_up[unit, MARK_PC] = S`                         | `conditions=[("MARK_PC", 1.0)]`                            |
| `b_up[unit] = -S * 1.5`                           | `threshold=1.5` (lowerer multiplies by `-S`)              |
| `W_gate[unit, CONST] = 1.0`, `b_gate[unit] = 1.0` | `constant_write` => no `gate`, `gate_bias=1.0`            |
| `W_down[TEMP+0, unit] = -5.0 / S`                 | `writes=[("TEMP+0", -5.0 / S)]`                            |

## 3. The 7-step recipe

### Step a. Inventory the imperative writes

Open the imperative helper. For each hidden unit it writes, identify:

- the **condition dims and weights** (`ffn.W_up[unit, BD.DIM] = w * S` lines),
- the **threshold** (the `ffn.b_up[unit] = -S * t` line),
- whether the unit is **gated** (`ffn.W_gate[unit, BD.GATE_DIM] = g`) or
  constant (`ffn.W_gate[unit, BD.CONST] = 1.0` and `b_gate = 1.0`),
- the **output writes** (`ffn.W_down[BD.OUT_DIM, unit] = w_down` lines).

Group consecutive units by sub-stage / op-purpose; one group = one rule
family. Multi-output sub-stages with shared conditions become a list of
related rules.

### Step b. Write one `FFNRule` per unit

Use `FFNRule.constant_write` when the original unit set
`W_gate[unit, CONST] = 1.0` (the legacy ungated form). Use
`FFNRule.gated_write` when the unit's gate read a content dim — pass that dim
as the `gate=` argument or include it in `gate_terms=`.

Pull weights out of `S` scaling: if the imperative line was
`W_up[..., DIM] = S * w`, the rule weight is `w`. Lowerer multiplies by `S`.
Inversely, output `W_down[OUT, unit] = w / S` becomes `writes=[("OUT", w / S)]`
— output weights do **not** get rescaled by `S` in the lowerer.

### Step c. Declare `scope` and `dominates_at`

`scope` is the predicate-DSL string describing the firing positions the rule
intends to claim (used by Phase F-7's `verify_rule_scopes`). `dominates_at`
narrows this per-output-dim when the rule writes multiple outputs with
different dominance constraints (used by `verify_rule_strength` /
`is_dominant_writer`). Both default to `None`, which disables the
corresponding verifier check — only do that for clearly auxiliary rules.

Predicates compose with `and` / `or` / `not` and reference dim-presence
predicates the verifier already understands (e.g. `OP_LEV`, `MARK_PC`,
`IS_BYTE`, etc).

### Step d. Wrap into `FFNOp` and attach via `compiler_ir`

```python
def _foo_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_foo_rules(S))
    return ir

return Operation(
    name="foo",
    # ...
    compiler_ir=_foo_ir(),
    # ...
)
```

The `layer_idx=0` argument to `ir.layer(0)` is intentional: `compiler_ir` is a
**single-layer** spec that the `layer_compiler` re-targets to whatever block
the op binds to (the actual layer comes from `op.layer_idx` or the
dependency-driven placement). See `_dispatch_operation_ir` in
`layer_compiler.py:553`.

### Step e. Remove the bake_fn body (or use `_lower_via_compiler_ir`)

If the op only has the migrated FFN content, drop `bake_fn` /
`declarative_bake_fn` entirely — `_dispatch_operation_ir` will pick up the
`compiler_ir` and call `ir.lower_ffn(...)` for `kind="ffn"` ops or
`ir.lower_attention(...)` + `ir.lower_ffn(...)` for `kind="block"` ops.

If the op also runs an imperative tail (common during incremental migration of
a multi-substage helper), keep `bake_fn` but make the migrated portion call
`Primitives.lower_ffn_rules(ffn, rules, dim_positions, start_unit=..., S=S)`
and let the remaining tail continue from the returned cursor. See
`l5_ops._bake_opcode_decode_ffn` for the multi-block pattern, or
`l0_ops._bake_phase_a_ffn` for the simplest single-block pattern.

### Step f. Validate via `compare_symbolic_to_lowered_ffn`

Every migration commit MUST be byte-identity-validated:

```python
from c4_release.neural_vm.unified_compiler.ir import (
    compare_symbolic_to_lowered_ffn,
)

report = compare_symbolic_to_lowered_ffn(
    _foo_ir(),
    dim_positions,
    S=S,
)
assert report.ok, report.format()
```

The check verifies (1) every condition / gate / write dim resolves to a
dim_position, (2) the lowered `W_up` / `b_up` / `W_gate` / `b_gate` / `W_down`
weights match the lowering contract cell-for-cell, (3) symbolic execution of
the rule list agrees with running `PureFFN.forward` on synthetic state. A
failure here means the FFNRule list does **not** reproduce the imperative
writes exactly — abort the commit, fix the mismatch, retry.

### Step g. Commit per logical group with the byte-identity test as gate

Migrate per sub-stage. Each commit:

1. converts one helper-internal sub-stage to rules,
2. flips `compiler_ir=` and/or shortens `bake_fn`,
3. asserts `compare_symbolic_to_lowered_ffn(...).ok` (in a test or at
   bake-time), and
4. leaves all existing claims, smoke tests, and 1096 metrics unchanged.

This keeps the diff per commit small enough to bisect on if a downstream layer
quietly drifts.

## 4. Conditions DSL quick reference

### Dim references inside rules

`FFNRule.constant_write` / `FFNRule.gated_write` accept string dim names with
optional offsets, parsed by `DimRef.parse`:

| String           | Meaning                                          |
|------------------|--------------------------------------------------|
| `"OP_LEV"`       | dim_positions["OP_LEV"] + 0                      |
| `"MARK_PC"`      | dim_positions["MARK_PC"] + 0                     |
| `"TEMP+0"`       | dim_positions["TEMP"] + 0 (first TEMP cell)      |
| `"TEMP+8"`       | dim_positions["TEMP"] + 8                        |
| `"BYTE_INDEX_3"` | a standalone dim, no offset                      |
| `"H0+5"`         | sixth output of the H0 threshold-head bank       |

All four "tuple" parameters use this string form:

```python
conditions = [("OP_LEV", 0.1), ("MARK_PC", 1.0)]   # (dim_name, weight)
writes     = [("TEMP+0", -5.0 / S)]                 # (dim_name, weight)
gate_terms = [("TEMP+0", -1.0)]                     # used by gated_write
gate       = "TEMP+0"                                # the canonical gate dim
```

The lowerer multiplies condition weights by `S` (to keep activations in the
saturation regime where SiLU is near-linear). It does **not** scale write
weights or gate weights. This is why imperative `W_up[..., DIM] = S * w` lines
become `("DIM", w)` while `W_down[OUT, unit] = w / S` stays as
`("OUT", w / S)`.

### `scope` predicate DSL

`scope` is checked by the declarative verifier
(`unified_compiler/decl_verifier.py`). The predicate is a Python-like
expression over dim-presence atoms. Common atoms:

| Atom              | Means                                                       |
|-------------------|-------------------------------------------------------------|
| `OP_LEV`          | the row has `OP_LEV > threshold` (an LEV opcode is active)  |
| `MARK_PC`         | the row is a PC marker                                      |
| `IS_BYTE`         | the row is one of the byte slots                            |
| `BYTE_INDEX_0`    | byte slot 0 specifically                                    |
| `MARK_AX`, ...    | each marker has a presence atom                              |
| `H1+1`            | the H1 threshold-head bank fires on slot 1                  |
| `not X`           | negation                                                    |
| `X and Y`         | conjunction                                                  |
| `X or Y`          | disjunction                                                  |

Examples from real migrated ops:

```python
scope="OP_LEV and MARK_PC"               # l14 temp_clear
scope="IS_BYTE and BYTE_INDEX_0"         # byte-0 row producers
scope="MARK_STACK0 and not MARK_PC"      # STACK0-only writers
```

If a rule writes a constant zero-everywhere effect (rare; usually a guard
helper) leave `scope=None`.

### `dominates_at` per-output dominance

When one rule produces several writes that compete with rules from other ops
at *different* sets of positions, use `dominates_at` to attach a separate
dominance predicate per output dim. Without it, every write inherits the
rule-level `scope`.

```python
dominates_at = {
    "OUTPUT_LO+0":            "MARK_STACK0 and not OP_LEV",
    "OUTPUT_HI_THIS_STEP+0":  "MARK_STACK0",
}
```

The cross-op competition check (S-6 extension / sign-aware S-2 follow-up)
reads `dominates_at_for(name)` per output dim before deciding whether this
rule wins on a contested cell.

## 5. Gotchas

- **Claims must match observed W_down writes**. A migrated FFNRule that adds
  `writes=[("OUT+5", w)]` produces one `W_down[OUT+5, unit]` cell write. Every
  such write must appear in the op's `claims=` set (with the right unit
  index), or the static verifier will flag an unclaimed write. Re-run
  `verify_claims_static` after each migration commit.
- **Threshold tuning is brittle**. Imperative bakes often choose
  `b_up = -S * t` with `t` empirically tuned so that exactly the intended
  combination of conditions fires. Read the docstring carefully; do not round
  thresholds; pull `t` out of the constant in `b_up = -S * t` exactly as
  written. `compare_symbolic_to_lowered_ffn` will catch mistakes here.
- **Rule families that share output dims need `dominates_at`
  disambiguation**. If two rules from two different ops both write `TEMP+0`,
  the verifier needs per-cell dominance scopes to know which rule "owns" the
  cell at which position. Without `dominates_at`, the rule-level `scope` is
  applied to every write — fine for single-output rules, ambiguous for
  multi-output rules in contention.
- **Sign of writes matters**. The contribution algebra (S-2 follow-up; see
  `contribution_algebra.py` and `is_dominant_writer`) is now sign-aware: a
  +50/S write and a -50/S write on the same cell are not equivalent. Preserve
  the original sign exactly; never flip a write sign during migration "for
  readability".
- **Scale of `S` in writes**. The lowerer multiplies condition weights and
  `b_up` thresholds by `S` but leaves `writes`, `gate_weight`, `gate_bias`,
  and `gate_terms` unchanged. So `W_up = S` becomes `("DIM", 1.0)` whereas
  `W_down = 2.0 / S` becomes `("OUT", 2.0 / S)` (with `S` baked in literally).
- **Gated vs constant**. The legacy ungated unit (`W_gate[unit, CONST] = 1.0`,
  `b_gate = 1.0`) is `FFNRule.constant_write`. Any unit that uses a content
  dim as gate is `FFNRule.gated_write`, even if the rest of the unit looks
  trivial. Misclassifying these will silently break the gate path.
- **Migration is best done per sub-stage**. A 4096-unit helper does not
  migrate cleanly in one commit. Split into 5-50-unit logical groups; each
  group becomes a `_foo_substage_rules(S)` function appended to the ir;
  validate after each.

## 6. Migration checklist

For each imperative helper migration commit:

- [ ] Inventoried all hidden units in the helper, grouped by sub-stage.
- [ ] Wrote one `FFNRule.constant_write` / `FFNRule.gated_write` per unit.
- [ ] Decided `scope=` for each rule (or explicitly chose `None` with reason).
- [ ] Decided `dominates_at=` for rules with multi-output / contested writes.
- [ ] Built `_foo_ir()` returning a `CompilerIR` with the rules appended at
      `layer(0).ffn.rules`.
- [ ] Attached `compiler_ir=_foo_ir()` on the `Operation`.
- [ ] Either removed the `bake_fn` body, or shortened it to a single
      `Primitives.lower_ffn_rules(...)` call.
- [ ] Updated `claims=` to match the W_down writes the rules produce.
- [ ] Asserted `compare_symbolic_to_lowered_ffn(_foo_ir(), dim_positions,
      S=S).ok` in a test or at bake-time.
- [ ] Ran `verify_claims_static` on the corpus — clean.
- [ ] Spot-checked 1096 metric for no regression (a sub-stage migration that
      drops rows means the lowering is not byte-identical).
- [ ] Commit message references the helper name and rule count.

## See also

- `c4_release/neural_vm/unified_compiler/ir.py` — `FFNRule`, `FFNOp`,
  `CompilerIR.lower_ffn`, `CompilerIR.symbolic_ffn`,
  `compare_symbolic_to_lowered_ffn`.
- `c4_release/neural_vm/unified_compiler/primitives.py` —
  `Primitives.lower_ffn_rules`, `Primitives.dim_positions_from_bd`,
  `Primitives.ffn_rule_dim_names`.
- `c4_release/neural_vm/unified_compiler/ops/l0_ops.py` —
  `_phase_a_ffn_rules` / `_phase_a_ffn_ir` / `_bake_phase_a_ffn` — simplest
  migrated reference (7 units, constant + gated mix).
- `c4_release/neural_vm/unified_compiler/ops/l5_ops.py` —
  `_opcode_decode_temp_clear_rules` / `_bake_opcode_decode_ffn` — multi-block
  migration with a kept imperative tail.
- `c4_release/neural_vm/unified_compiler/ops/l16_ops.py` —
  `_layer16_lev_routing_rules` — large-scale rule family with `scope` /
  `dominates_at` consistently threaded.
- `c4_release/neural_vm/unified_compiler/ops/l15_ops.py` —
  `make_l15_nibble_copy_ir` — `compiler_ir=` attached directly.
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py` —
  `make_layer8_multibyte_routing_ir` — large `compiler_ir=` example.
- `c4_release/docs/PHASE_6_DECLARATIVE_WEIGHT_AUTHORING_PLAN.md` — overall
  Phase 6 wave plan.
- `c4_release/docs/DECLARATIVE_VERIFICATION.md` — what the verifier checks
  once rules carry `scope` / `dominates_at`.
