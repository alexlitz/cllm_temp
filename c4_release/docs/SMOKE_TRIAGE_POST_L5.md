# Smoke triage — post-L5 fix (2026-06-03)

Status: **25/51 PASS, 26 FAIL** after the L5 opcode-decode FFN fix.
Source run: this worktree at HEAD `5f51cecb` (branch `speedup-cache-and-buckets`,
`/tmp/c4-deep-smoke-survey`).
Smoke wall: 109.5s. Run command:

```
timeout 240 python -m pytest c4_release/tests/test_smoke.py -v --tb=line -q
```

This doc is a **triage matrix**, not a fix proposal. Existing root-cause
docs are cited per cluster; do not re-investigate them.

## TL;DR — top 3 highest-ROI fixes

| Rank | Fix target | Tests recovered | File:line |
|---|---|---|---|
| 1 | **AX → out-byte residual chain (`exit_code=0` cluster)** | **~17** (memory 5, shift 2, basic 2, comparison 6, bit32 2 partial) | `neural_vm/vm_step.py:4604+` (SP_byte0 real logic); `neural_vm/unified_compiler/ops/l13_ops.py` retarget (see `SMOKE_MEMORY_TRACE_20260603.md`) |
| 2 | **L6 cross-step `CMP.*.-1` read on BZ/BNZ override** | **6** (eq_true, eq_false, lt, ne, gt, ge) | `neural_vm/unified_compiler/ops/l6_ops.py:1425-1510` + `l6_ops.py:2662-2668` reads decl (see `CMP_PATH_AUDIT.md`, `SMOKE_COMPARISON_OP_DECODE_MISSING.md`) |
| 3 | **PSH same-step AX→stack write (no-EXIT / "got None" cluster)** | **5** (and_basic, xor_basic, adj_sp, add_16bit, add_carry_cascade, and_16bit, xor_16bit) | `vm_step.py` PSH path; see `SMOKE_ROOT_CAUSE_BAKE_COLLISION.md` block-5 bake stomp (PSH bakes after opcode_decode at block 5) |

Pick fix #1 first: it touches the widest band and is already partially
diagnosed.

## Failure matrix — 26 tests

Columns: `mechanism` = (`zero` = AX-zero / exit_code=0 false-pos shape;
`no-exit` = `got=None`; `wrap` = unsigned-32 wrap from missing top byte;
`partial-bit` = correct low byte, wrong hi byte; `flip` = exact inversion).

| # | Test | Expected | Got | Mechanism | Cluster | Likely culprit | Effort |
|---|---|---|---|---|---|---|---|
|  1 | basic::test_sub_basic            | 42      | 4294967288   | wrap        | C-WRAP    | SUB AX top-byte hi nibble zero; 50 lost | M |
|  2 | basic::test_div_basic            | 42      | 0            | zero        | A-AXZERO  | AX-write residual chain (see #1)        | L |
|  3 | basic::test_mod_basic            | 3       | 0            | zero        | A-AXZERO  | same as div                              | L |
|  4 | bitwise::test_and_basic          | 42      | None         | no-exit     | B-NOEXIT  | PSH block-5 bake stomp                  | M |
|  5 | bitwise::test_xor_basic          | 42      | None         | no-exit     | B-NOEXIT  | PSH block-5 bake stomp                  | M |
|  6 | comparison::test_eq_true         | 1       | 0            | zero        | D-CMP     | L6 cross-step CMP.*.-1 read on BZ override | M |
|  7 | comparison::test_eq_false        | 0       | 1            | flip        | D-CMP     | same; inverted leak path                | M |
|  8 | comparison::test_lt_true         | 1       | 0            | zero        | D-CMP     | same as eq_true                          | M |
|  9 | comparison::test_ne_true         | 1       | 0            | zero        | D-CMP     | same                                     | M |
| 10 | comparison::test_gt_true         | 1       | 0            | zero        | D-CMP     | same                                     | M |
| 11 | comparison::test_ge_true         | 1       | 0            | zero        | D-CMP     | same                                     | M |
| 12 | address::test_adj_sp             | 42      | None         | no-exit     | B-NOEXIT  | PSH or ADJ SP_byte0 missing (`vm_step.py:4604+`) | M |
| 13 | memory::test_si_li_roundtrip     | 42      | 0            | zero        | A-AXZERO  | L13/L14/L15 layout drift (`SMOKE_MEMORY_TRACE_20260603.md`) | M |
| 14 | memory::test_sc_lc_roundtrip     | 42      | 0            | zero        | A-AXZERO  | same                                     | M |
| 15 | memory::test_si_li_multiple_stores | 99    | 0            | zero        | A-AXZERO  | same                                     | M |
| 16 | memory::test_si_li_overwrite     | 55      | 0            | zero        | A-AXZERO  | same                                     | M |
| 17 | memory::test_si_li_16bit_value   | 4660    | 0            | zero        | A-AXZERO  | same; also lacks hi-nibble propagation  | M |
| 18 | shift::test_shl                  | 42      | 0            | zero        | A-AXZERO  | SHL writeback path                       | M |
| 19 | shift::test_shr                  | 42      | 1            | partial-bit | A-AXZERO  | SHR lo-bit pass-through (84>>1=42, got=1=top-bit only) | M |
| 20 | bit32::test_add_16bit            | 300     | None         | no-exit     | B-NOEXIT  | PSH bake stomp                           | M |
| 21 | bit32::test_add_carry_cascade    | 256     | None         | no-exit     | B-NOEXIT  | PSH bake stomp + carry-cascade           | M |
| 22 | bit32::test_sub_16bit            | 255     | 4294967295   | wrap        | C-WRAP    | SUB lo lane ok, hi lanes leak -1         | M |
| 23 | bit32::test_or_16bit             | 4095    | 255          | partial-bit | E-HIBYTE  | OR hi-byte missing (lo correct)         | M |
| 24 | bit32::test_and_16bit            | 255     | None         | no-exit     | B-NOEXIT  | PSH bake stomp                           | M |
| 25 | bit32::test_xor_16bit            | 4080    | 65520        | partial-bit | E-HIBYTE  | XOR hi nibble inverted; lo ok           | M |
| 26 | bit32::test_mul_overflow         | 500     | 0            | zero        | A-AXZERO  | MUL writeback                            | M |

## Clusters

### A — AX-zero / `exit_code=0` (12 tests)

Tests 2, 3, 13–18, 19 (partial), 26. Same failure mode flagged in
`SMOKE_MEMORY_TRACE_20260603.md` and `.agent-logs/smoke_drift_20260602.md`.
The AX-write residual chain produces zero for non-zero results.
False-positive twin: `test_si_li_zero` passes because expected==0.

**Likely culprit**: L13/L14/L15 layout drift after dep-anchor retarget
(commit `ff454052` did not propagate). Fix in `l13_ops.py` /
`l14_ops.py` (see memory trace doc) restores the AX→OUTPUT_LO[0]
write at the EXIT step.

### B — no-EXIT / `got=None` (6 tests)

Tests 4, 5, 12, 20, 21, 24. Runner times out before emitting EXIT.
Every test in this cluster has PSH preceding the failing op. Matches
`SMOKE_ROOT_CAUSE_BAKE_COLLISION.md`: PSH baked into block-5 FFN with
`start_unit=0` after `opcode_decode_ffn` stomped its `b_up`.

### C — UINT32 wrap (2 tests)

Tests 1, 22. SUB lo-byte computed correctly, but hi-byte propagation
fills `0xFF…` because the borrow lane reads zero from the missing
top operand. Same root-cause family as cluster E (hi-byte writeback).

### D — Comparison BZ/BNZ leak (6 tests)

Tests 6–11. All 6 `TestSmokeComparison` tests. Already traced in
`CMP_PATH_AUDIT.md` and `SMOKE_COMPARISON_OP_DECODE_MISSING.md`:
`layer6_routing_ffn` declares `reads={"CMP.*.-1"}` so on step-1 the
BZ override reads previous-step CMP+4/CMP+5 = 0. The L9 same-step
CMP write loses the race.

Note: `test_le_true` passes — likely because LE happens to map to the
zero-coincidence path (BZ-taken with cancel band = 0).

### E — Partial hi-byte / lo-byte (3 tests)

Tests 19 (shr), 23 (or_16bit), 25 (xor_16bit). Lo byte correct, hi
byte either missing or inverted. Bitwise hi-lane FFN unit missing or
stomped — related to cluster A (AX-write) but specifically the hi-byte
lane (`OUTPUT_HI`).

## Reproducibility notes

- One smoke run only (constraint).
- Zero compile changes (constraint).
- No bisect, no second compile — all data is from the single 109.5s run
  plus existing doc archaeology.
- Existing related docs referenced (do not re-investigate):
  - `SMOKE_ROOT_CAUSE_BAKE_COLLISION.md`
  - `SMOKE_COMPARISON_OP_DECODE_MISSING.md`
  - `SMOKE_MEMORY_TRACE_20260603.md`
  - `CMP_PATH_AUDIT.md`
  - `IF_EQ_CMP_DEFAULT_LEAK.md`
  - `.agent-logs/smoke_drift_20260602.md`

## ROI rationale

- **#1 (AX-write chain)**: 12 cluster-A tests + 3 cluster-E tests + 2
  cluster-C tests share the same `AX → OUTPUT_LO/HI` chain. A single
  L13/L14 retarget should recover ~12–17 tests.
- **#2 (CMP cross-step)**: 6 comparison tests cleared by adding
  same-step CMP read in `l6_ops.py` BZ/BNZ override rules.
- **#3 (PSH bake stomp)**: 6 no-EXIT tests cleared by sequencing the
  PSH bake before opcode_decode at block 5 (move PSH off the shared
  unit-0 slot).
