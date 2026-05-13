# Declarative op verifier

**Status**: Landed 2026-05-12. Mode A (static claim verification) is
wired into the production test gate
(`c4_release/tests/test_declarative_verification.py`). Mode B (dynamic
produces/consumes) is opt-in (`C4_VERIFY_DECLARATIONS=1`) because it
adds a full forward pass per verification cycle.

## Problem

Each `Operation` declares a contract:

```python
Operation(
    name="layer5_fetch",
    reads={"MARK_PC", "MARK_AX", ...},
    writes={"OPCODE_BYTE_LO", ...},
    claims={(5, "attn_W_v", "0_32", "CLEAN_EMBED_LO+0"), ...},
    produces={"AX_CARRY_LO": "AX_byte0"},
    consumes_fresh={...},
    bake_fn=bake,
)
```

The static analyzers (`_detect_claim_collisions` and
`_detect_staleness_violations`) cross-check these declarations against
each other -- but they assume each declaration is HONEST. There was no
check that the `bake_fn` actually writes the slots in `claims`, or that
it refreshes the dim it `produces`. An op could declare
`produces={"AX_CARRY_LO": "AX_byte0"}` while its bake writes to a
completely different dim, and the staleness analyzer would still pass.

The verifier in `c4_release/neural_vm/unified_compiler/decl_verifier.py`
closes that gap.

## Mode A: Static claim verification

For every `Operation` with non-empty `claims`, Mode A:

1. Builds a fresh production model via `compile_full_vm()`.
2. Re-runs the compiler's bake pass, dispatching every op except the
   target one in order.
3. Snapshots the model (every Parameter, keyed by `data_ptr`).
4. Runs the target op's `bake_fn`.
5. Diffs each weight matrix against the snapshot. Every cell whose
   value changed is decoded back into a `(layer, scope, identifier,
   column)` 4-tuple using the model's geometry plus the layout's
   `dim_positions`.

The decoder mirrors the registry convention in
[`DIM_OWNERSHIP_REGISTRY.md`](DIM_OWNERSHIP_REGISTRY.md):

| Matrix    | Scope         | Row meaning            | Col meaning             | Identifier      |
|-----------|---------------|------------------------|-------------------------|-----------------|
| attn W_q  | `attn_W_q`    | `head*HD + slot`       | input residual position | `"<head>_<slot>"`|
| attn W_k  | `attn_W_k`    | `head*HD + slot`       | input residual position | `"<head>_<slot>"`|
| attn W_v  | `attn_W_v`    | `head*HD + slot`       | input residual position | `"<head>_<slot>"`|
| attn W_o  | `attn_W_o`    | output residual pos    | `head*HD + slot`        | `"<head>_<slot>"`|
| ffn W_up  | `ffn_W_up`    | hidden unit            | input residual position | unit idx        |
| ffn W_gate| `ffn_W_gate`  | hidden unit            | input residual position | unit idx        |
| ffn W_down| `ffn_W_down`  | output residual pos    | hidden unit             | unit idx        |
| embed table| `embed_row`  | token id (row only)    | n/a                     | token id        |

The column field uses the dim-position dictionary to translate a residual
position back into `"<DIM_NAME>+<offset>"`. Cells whose position lies
outside any declared dim (i.e. inside a padding band) are tracked as
`observed_uncategorized_count` and ignored.

### Verification semantics

Per op, Mode A computes two sets:

- `declared_but_not_written` (UNUSED CLAIM): the bake did not touch the
  cell the op promised to write. **This is the load-bearing error** --
  a stale annotation, almost always indicating the bake's behavior
  drifted away from its declared intent.
- `written_but_not_declared` (UNDECLARED WRITE): the bake wrote a cell
  that the op didn't claim.

The verifier has two modes:

| Mode      | When `declared_but_not_written` is non-empty | When `written_but_not_declared` is non-empty |
|-----------|----------------------------------------------|----------------------------------------------|
| Default   | ERROR (test fails)                           | INFO (count reported, not flagged)           |
| `strict`  | ERROR                                        | ERROR                                        |

The default mode reflects the **partial-claim convention** currently in
use: each op declares the high-collision-risk subset of its writes (the
V-relay rows that collide with peer ops), not its full bake footprint.
Adding the full footprint would make claims declarations as long as the
bake_fn itself and offer little incremental value. The verifier honors
this convention; tests that want full-coverage declarations can flip
`strict_mode=True` and surface the residue as errors.

### Inert ops

Some ops are intentionally registered but no-op'd. For example,
`make_layer8_head6_ax_carry_refresh_op(enable=False)` registers the op
so its `produces` annotation participates in the staleness scan, but
its `bake_fn` returns early. The verifier detects this case (the
snapshot diff is empty) and marks the op `inert=True` -- both `ok` and
`ok_strict` then pass regardless of declared claims.

This means flag-gated ops can declare their intended claims/produces
without false positives. When the flag flips on, those declarations
become load-bearing.

## Mode B: Dynamic produces/consumes verification

Opt-in via `C4_VERIFY_DECLARATIONS=1`. Mode B runs a synthetic
1-instruction probe through the compiled model, captures the residual
stream at each layer boundary, and reports any op whose `produces[dim]`
declaration does NOT correspond to a non-zero residual value at the AX
marker position.

This is a coarse liveness check -- it does not verify register identity
(`"AX_byte0"` vs the actual register the op meant). That guarantee
requires a behavioural smoke (the production add/sub/IMM tests). Mode B
catches the easy case: an op claims to produce a dim but doesn't
actually write anything that reaches the dim's residual slot.

## Running

```python
from c4_release.neural_vm.unified_compiler.decl_verifier import (
    verify_claims_static,
    verify_produces_consumes_dynamic,
)

# Mode A (default, fast)
report = verify_claims_static()
if report.has_errors():
    print(report.format())
    raise AssertionError("declaration drift")

# Mode A in strict mode
strict = verify_claims_static(strict_mode=True)

# Mode B (slow, requires forward-pass)
dyn = verify_produces_consumes_dynamic()
```

The test driver in
`c4_release/tests/test_declarative_verification.py` runs Mode A
unconditionally and Mode B only when `C4_VERIFY_DECLARATIONS=1` is set.

## Interpreting output

```
=== Static claim verification report (mode=PARTIAL-CLAIMS) ===
Ops verified: 2
Ops with errors: 0
  [OK] layer5_fetch: declared=192 observed=755 unused_decl=0 undeclared_writes=563
  [OK] function_call_weights: declared=96 observed=2342 unused_decl=0 undeclared_writes=2246
```

- `declared`: how many `(layer, scope, identifier, column)` cells the
  op claimed.
- `observed`: how many cells the verifier observed the bake_fn writing.
- `unused_decl`: how many declared cells the bake did NOT write.
  **Non-zero here is a bug** (in default mode).
- `undeclared_writes`: how many cells the bake wrote that weren't
  claimed. Non-zero is the partial-claim residue and is fine by
  default; in strict mode it counts as an error.

When `unused_decl > 0`, the report lists each missing cell::

    UNUSED CLAIM (DECLARATION DRIFT): (5, 'attn_W_v', '6_5', 'EMBED_LO+4')

Fixing such a drift means either:

- Updating `Operation.claims` to remove the stale tuple (if the bake
  intentionally stopped writing it), or
- Restoring the bake's write logic (if the claim is correct and the
  bake regressed).

## Currently-annotated ops + verifier findings (2026-05-12)

Run on `compile_full_vm(alu_mode='lookup')` with default flags. Two ops
have non-empty `claims`:

| Op | Declared | Observed | Unused (DRIFT) | Status |
|----|---------:|---------:|---------------:|--------|
| `layer5_fetch` | 192 | 755 | 0 | OK |
| `function_call_weights` | 96 | 2342 | 0 | OK |

Both ops use the partial-claim convention. Every declared cell is
actually written by the bake. No declaration drift.

The 563 / 2246 undeclared-writes counts represent the rest of each
op's bake footprint (Q/K/O rows, gate weights, exclusion dims, FFN
units for `function_call_weights`'s ENT/JSR routing). Promoting these
to declared claims would make the verifier's strict-mode check pass
but is not currently required.

`make_layer8_head6_ax_carry_refresh_op` has no `claims` (only
`produces`), so it isn't exercised by Mode A. With `enable=False` (its
production default), Mode B reports it as inert; with `enable=True`,
Mode B verifies the AX_CARRY_LO/HI residual at the AX marker becomes
non-zero after the L8 attention layer runs.

## Routine execution

The verifier is gated behind the `validator` pytest marker so smoke
runs (`pytest`) skip it by default. Routine execution paths:

### `make validate`

The headline target. Runs every test under `-m validator` with short
tracebacks:

```bash
cd c4_release
make validate         # Mode A + Mode B+ multistep cascade probe
make validate-strict  # opt-in strict-mode (currently empty marker set)
```

`make validate` is the recommended invocation for contributors before
opening a PR and for the nightly cadence. Wall clock is ~60-90s on a
warm cache; the multistep tests share one bake via a class-scoped
fixture.

### Pre-commit hook

The local `validator` hook in `c4_release/.pre-commit-config.yaml`
runs `pytest -m validator --tb=line -q` at the **pre-push** stage
(not pre-commit). Pre-push was chosen because the verifier wall is too
heavy for the sub-second budget pre-commit needs. To enable:

```bash
pip install pre-commit
pre-commit install --hook-type pre-push
```

To run on every commit instead, change `stages: [pre-push]` to
`stages: [pre-commit]` in the config (not recommended on slow
machines).

### `C4_VALIDATE_ON_COMPILE=1`

Run Mode A as a warn-only sanity check at the end of every
`compile_full_vm()` call. Drift is reported via `warnings.warn` (and
optionally `print(report.format())` if `C4_VALIDATE_VERBOSE=1`); the
compile never fails. Useful when chasing a regression because every
test that builds the model becomes a verifier run.

```bash
C4_VALIDATE_ON_COMPILE=1 pytest c4_release/tests/test_smoke.py -v
C4_VALIDATE_ON_COMPILE=1 C4_VALIDATE_VERBOSE=1 python -m my_repro
```

Default is off; production builds are unaffected.

### When to use each

| Path | Cost | When |
|------|------|------|
| `make validate` | 60-90s | Local dev pre-PR; nightly CI |
| Pre-push hook | 60-90s | Every `git push` (after `pre-commit install --hook-type pre-push`) |
| `C4_VALIDATE_ON_COMPILE=1` | adds 60-90s per compile | Regression hunting; never in production |
| `pytest -m validator_strict` | varies | Opt-in nightly when ops adopt full-coverage claims |

## Future work

- Extend the verifier to cover `embed_row` claims end-to-end (currently
  the diff scans the embedding table but only reports row-level
  granularity).
- Promote the partial-claim convention to either (a) document
  high-coverage claims explicitly, or (b) keep partial claims and have
  the verifier emit a `strict_coverage` ratio for each op so authors
  can see how much of their bake footprint is declared.
- Mode B's register-identity check (verify that the dim a `produces`
  declaration names actually contains the named register's value) needs
  a step-by-step interpretation oracle to validate against.
