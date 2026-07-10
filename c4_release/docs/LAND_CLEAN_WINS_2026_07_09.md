# LAND_CLEAN_WINS — 2026-07-09

Landing this session's byte-identical / tooling wins to `main`. Every branch
is verdict-neutral (golden `e50521f3` unchanged) or pure tooling. Merged ONE
AT A TIME onto `main`; the authoritative byte-identity gate
(`tools/_isa_golden_hash.py`, `disk_cache=False`) was re-run after every
model-affecting merge and MUST equal `e50521f3`.

## Baseline

- Starting `main` HEAD: `a4251457`
  (`docs(flag-registry): C4_AX_HIBYTE_CLEAR is the if_var GT-FALSE AX byte-2 lever`)
- Baseline golden on `a4251457` (rebuilt, `disk_cache=False`):
  `state_dict_sha256 = e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86`
  (short `e50521f3`) — matches the mission gate.
- FFN units: 42149 (unchanged across every merge).

> Note: the six branches share merge-base `59a9de19` (an ancestor of `main`).
> They were replayed DIRECTLY onto `main` (not onto the parallel
> `integration-round3` line, which additionally carries two unrelated
> pre-existing integration merges — `mul SE-recover` + `func/SI stack` — that
> are OUT OF SCOPE for this "land the clean wins" mission and were deliberately
> excluded from `main`).

## Per-branch merge result

| # | Branch | Merge commit | Conflicts | Golden after | Result |
|---|--------|--------------|-----------|--------------|--------|
| 1 | `fast-gate` | `127b38f2` | none | `e50521f3` (tooling-only, trivially held) | LANDED — pure tooling (`tools/fast_gate.py` + sample json + doc). No model files. |
| 2 | `wrapper-to-ir` | `69139b6b` | none | `e50521f3` (verification-only) | LANDED — `verification/faithful_interpreter.py` registers the 5 efficient-mode wrapper FFN classes into `COMPOSITE_ALU_FFN`. Not on the build path. |
| 3 | `land-derivations` | `7e5fe2a4` | none | `e50521f3` (hard-verified) | LANDED — bitwise/cmp/shift derivations DEFAULT-ON; subsumed enumeration deleted. `l10_ops.py` −144 net (this merge's share of the deletion). |
| 4 | `derive-actscale` | `7e76c8b1` | 2 (`full_vm_compiler_dynamic.py`, `ops/shared.py`) — resolved ADDITIVELY | `e50521f3` (hard-verified) | LANDED — `C4_DERIVE_GATE_SCALES` calibration + gate rollout (l6/l16), default-OFF. |
| 5 | `collapse-l10-tail` | `4c681392` | none (auto-merged with land-derivations' l10 changes) | `e50521f3` (hard-verified) | LANDED — l10 STACK0 byte-writeback single-source-of-truth. |
| 6 | `derive-memory` | `2d4a81a2` | 3 (`docs/FLAG_REGISTRY.md`, `full_vm_compiler_dynamic.py`, `ops/shared.py`) — resolved ADDITIVELY | `e50521f3` (final hard-verify) | LANDED — `C4_DERIVE_MEMORY` umbrella flag OR-flooring `si_store_addr` + `var_three_li`, default-OFF. |

### Conflict resolution notes

All conflicts were pure ADDITIVE insertions into the same regions (the two
cache-key flag-snapshot dicts in `full_vm_compiler_dynamic.py`, the flag-
predicate function block in `ops/shared.py`, and the campaign-flag table in
`docs/FLAG_REGISTRY.md`). Both sides were kept in `HEAD`-then-branch order:

- `full_vm_compiler_dynamic.py` (both snapshots): kept HEAD's flag entries
  (`C4_DERIVE_BITWISE/SHIFT/CMP`, and the `C4_LI_VALUE_LOAD` line) AND the
  branch's new entries (`C4_DERIVE_JMP/CONTROL/GATE_SCALES/PC_OVERRIDE_K/
  ACTSCALE_JSON` for actscale; `C4_DERIVE_MEMORY` for memory), fixing the
  shared trailing `),` so each dict entry closes correctly.
- `ops/shared.py`: kept HEAD's `derive_bitwise/shift/cmp_enabled()` block AND
  appended the branch's new predicate functions (`derive_gate*` / `derive_gate`
  / `derive_memory_enabled()`). The `derive-memory` OR-floor wiring
  (`si_store_addr_enabled` → `campaign_enabled() or derive_memory_enabled()`,
  `var_three_li_enabled` → `derive_memory_enabled()` floor) applied via the
  non-conflicting hunks and was verified present post-merge.
- `docs/FLAG_REGISTRY.md`: took `derive-memory`'s improved `C4_SI_STORE_ADDR`
  row wording + the new `C4_DERIVE_MEMORY` row, KEPT HEAD's `C4_LI_VALUE_LOAD`
  row.

Every resolution was checked: no conflict markers remain, both `.py` files
`ast.parse` clean, `ops.shared` imports with correct defaults
(`derive_bitwise/shift/cmp = True`, `derive_gate_scales/memory/si_store_addr =
False`), and the final golden is unchanged.

## Final state

- **Final merge tip: `2d4a81a2`** (`Merge branch 'derive-memory' into main`) —
  the authoritative landing point; ALL golden/smoke/opaque gates were measured
  here. The final `main` HEAD is this deliverable doc commit sitting one commit
  atop `2d4a81a2` (doc-only, byte-neutral; `git log --oneline -1 main` for the
  exact sha).
- **Final golden (`_isa_golden_hash.py`, `disk_cache=False`): `e50521f3` —
  HELD.** (FFN units 42149, unchanged.)
- **Smoke (`pytest tests/test_smoke.py`): 51 passed, 0 failed** (1 deselected
  xfail); exit 0.
- **Faithful-interpreter coverage on the PRODUCTION efficient model
  (`--coverage --alu-mode efficient --no-disk-cache`): `opaque_skipped == []`
  — 61/61 blocks IR-executable** (8 composite-ALU blocks now first-class incl.
  the 5 registered wrapper FFN classes; was ~4/43 pure-IR). The `wrapper-to-ir`
  win is landed.

## Source-LOC delta (`a4251457 .. 2d4a81a2`)

- Model source (`neural_vm/**/*.py`): **+1673 / −382 = NET +1291**.
  - Net-negative model file: `l10_ops.py` **+217 / −361 = −144** (derivation
    enumeration deletion + tail-writeback consolidation).
  - The +1291 net is dominated by the NEW derivation-DSL modules
    (`shift_semantics_dsl.py` +158, `building_blocks_dsl.py` +222,
    `wide_alu_dsl.py` +91, `verification/activation_scales.py` +259) and the
    predicate/flag plumbing in `shared.py` (+435 net) — i.e. the derivations
    REPLACE hand-enumeration with generators + calibration, so the net grows in
    reusable DSL while `l10_ops.py` shrinks. All byte-identical.
- Tooling + tests + docs (`tools/**`, `tests/**`, `docs/**`): **+3678 / −5 =
  NET +3673** (fast-gate harness + sample, calibrate tool, verify_derive_cmp,
  gate-scale test, and all the DERIVE_*/LAND_*/WRAPPER/COLLAPSE/FLAG_REGISTRY
  docs).

## Verdict

All 6 branches LANDED to `main` byte-identically. Golden `e50521f3` held at
every step; no branch moved the golden; no branch was left unmerged. Smoke
51/51 and efficient-model `opaque_skipped==0` both confirmed on the final
`main`.
