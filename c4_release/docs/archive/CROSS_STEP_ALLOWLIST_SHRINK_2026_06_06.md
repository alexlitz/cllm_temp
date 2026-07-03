# CROSS_STEP_BASELINE_ALLOWLIST shrink audit — 2026-06-06

## Summary

Audit ran against the latest main HEAD (`9c9fadd2`) to determine which
entries in `CROSS_STEP_BASELINE_ALLOWLIST` (in
`c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py`)
are stale after the recent session fixes:

- L0-L7 declaration audit (`618b74e7..7786912b`)
- Lifetime walker fix (`471c1c08`) — include `kind=block` ops
- Phase 3/3b/3c anchor splits (`3423aff1`, `641a3c03`, `21a3fe7c`)

**Result: 0 entries are removable.** The baseline size stays at 36.

## Numbers

- Total cross-step findings on current op set: **84**
- `CROSS_STEP_BASELINE_ALLOWLIST` size: **36** (unchanged)
- `CROSS_STEP_DOCUMENTED_SAFE` size: **48** (unchanged)
- Sum (`36 + 48 = 84`) exactly covers all findings — no leakage.
- Stale baseline entries (in allowlist but not produced as a finding): **0**
- Stale documented-safe entries: **0**

## Methodology

For each `(op_name, dim_alias)` in the baseline, check whether
`_find_cross_step_reads_with_same_step_writers` on the current op set
still emits that pair as a finding. An entry is "stale" iff it is NOT
in the current findings set — in that case, removing it from the
baseline would NOT cause `compile_full_vm_dynamic(strict=True)` to
raise.

```python
findings = _find_cross_step_reads_with_same_step_writers(ops)
pairs = {(c, r) for (c, r, _ws) in findings}
stale = CROSS_STEP_BASELINE_ALLOWLIST - pairs  # -> set()
```

All 36 baseline entries are still real findings on the current op set;
all 48 documented-safe entries are still real findings. The
declaration-audit commits sharpened `reads={...}`/`writes={...}` sets
so the analysis is now more accurate, but every baseline entry still
has a same-step writer of the base dim — the conditions that earned
each entry its slot are still met.

## Why no entries shrank

The declaration audit (618b74e7..7786912b) and lifetime-walker fix
(471c1c08) increased the precision of the cross-step analysis — more
ops now have accurate `reads`/`writes` so the finding count went UP
from the Step-4 landing total (82) to **84**. The two new findings
were both audited into `CROSS_STEP_DOCUMENTED_SAFE` (Round-3 subsets
3A/3B/3C — see comments at lines 1200-1320 of
`full_vm_compiler_dynamic.py`). The baseline was already at its
post-Step-4 minimum coming into this session; nothing about the
declaration audit moved a baseline entry into the "no writer, no
same-step conflict" state that would make it removable.

The Phase 3/3b/3c anchor splits restructured the dep-graph but did
NOT rename any of the `.*.-1` reads or eliminate any of the base-dim
writers that produce the findings. They added new `_layer10_attn_anchor`
and `_layer13_attn_dep_anchor` entries — both went into documented-safe
rather than baseline (they are topology anchors with `bake_fn = return`).

## Removed entries

**None.** Per the brief's constraint "Only remove entries that are
clearly stale — when in doubt, leave it", no entries qualify.

## Verification

- `pytest tests/test_compile_cross_step_safety.py` — all 9 tests pass
  (no changes to allowlist).
- Smoke baseline on main HEAD (`9c9fadd2`): **45/52** (6 fail, 1
  deselected). Unchanged because no code was modified.

## Followup

Future shrinkage requires one of:

1. Renaming a `.*.-1` read in an op factory to either the same-step
   form or an explicit `OR(X, X.*.-1)` aggregation — that drops the
   finding and the corresponding baseline entry.
2. Removing/relocating a same-step writer of the base dim so no other
   op writes it — that also drops the finding.
3. Auditing an existing baseline entry, confirming it is intentional
   cross-step, and moving it from `CROSS_STEP_BASELINE_ALLOWLIST` to
   `CROSS_STEP_DOCUMENTED_SAFE` with a justification (shrinks the
   ratchet-watched baseline without changing runtime behaviour).

Option 3 is the lowest-risk path for further baseline shrinkage.
Candidates that would benefit from a Round-4 audit:

- `layer9_alu` reads of `ALU_HI.*.-1`, `ALU_LO.*.-1`, `CARRY.*.-1`
  (3 entries) — the writers are L8 ALU + L10 ALU; fire order may
  resolve cleanly to documented-safe.
- `layer8_alu` ALU_LO.*.-1 — single entry, same pattern.
- `layer3_ffn` EMBED_HI/LO.*.-1 — both have prev-step semantics from
  the L17 embedding writeback; likely documented-safe candidates.

These were not migrated in this session because they require per-op
audit of writer fire order which the brief did not authorise.
