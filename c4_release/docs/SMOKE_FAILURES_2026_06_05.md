# Smoke failures — full audit (2026-06-05)

Status: **38/52 PASS, 14 FAIL** in `tests/test_smoke.py` after the
`f3342968` highbit-imm fix. `tests/test_smoke_pure_neural.py`:
**11 XPASS, 29 XFAIL** (all xfail/xpassed are still-pending phase work;
no XFAIL is a "real" regression here).

Source run: HEAD `f3342968` (branch `speedup-cache-and-buckets`,
worktree `/tmp/c4-smoke-audit`, branch `smoke-audit`).
Smoke wall: 990.18s (16:30). Run command:

```
python -m pytest tests/test_smoke.py tests/test_smoke_pure_neural.py --tb=no -v
```

This doc is a **categorization matrix**, not a fix proposal. It supersedes
the 26-failure matrix in `docs/SMOKE_TRIAGE_POST_L5.md` (which predated
the `f3342968` highbit-imm patch that recovered 9 of the original 26).
Read `SMOKE_TRIAGE_POST_L5.md` for tests that already passed; this doc
covers only the remaining 14.

## TL;DR — fix priority (biggest cluster first)

| Rank | Category | Tests | Recommended approach |
|---|---|---|---|
| 1 | **memory** (cluster A — TestSmokeMemory AX-zero) | 5 | One agent, L13/L14 dep-anchor retarget propagation |
| 2 | **cascade-32bit** (TestSmoke32Bit multibyte ALU) | 5 | One agent, extend highbit-imm fix to multi-byte ALU emit |
| 3 | **cmp-polarity** (test_eq_false, test_ne_true) | 2 | One agent, L6 BZ/BNZ override or cmp-default polarity |
| 4 | **specific-opcode** (test_xor_basic, 0xFF imm sign-extension) | 1 | Independent fix in highbit-imm dispatch |
| 5 | **never-dispatches** (test_lea_basic, ENT-main pin) | 1 | Independent fix in L10 PSH/ENT path |

**Highest-ROI single target**: memory cluster (5 tests, same root cause).
**Largest cluster**: 32-bit cascades (5 tests; same multi-byte ALU root).
Memory + 32-bit cascades together = 10 of 14 (71%) and may share part
of the AX-write residual chain (cluster A in the prior triage).

## Failure matrix — 14 tests

Columns: `expected` / `got` come from the actual smoke run; `cat` is the
classification (see categories below).

| # | Test | Bytecode (compact) | Expected | Got | Category | 1-sentence hypothesis |
|---|------|--------------------|----------|-----|----------|------------------------|
| 1 | `TestSmokeBitwise::test_xor_basic` | `IMM 0xFF, PSH, IMM 0xD5, XOR, EXIT` | 42 (0x2A) | 213 (0xD5) | **specific-opcode** | Highbit-imm collapsed-step recovery sign-extends `IMM 0xFF` to `0xFFFFFFFF` so `(0xFF) ^ (0xFFFFFFFF) = 0xFFFFFF2A`, but the AX override writes back only the noisy neural byte — model passes through second IMM operand (0xD5) directly. |
| 2 | `TestSmokeComparison::test_eq_false` | `IMM 10, PSH, IMM 20, EQ, EXIT` | 0 | 1 | **cmp-polarity** | The model never decodes `OP_EQ` (see `SMOKE_COMPARISON_OP_DECODE_MISSING.md`), so the EQ default-path emits 1 regardless of operand equality; passes only when expected==1 (test_eq_true / test_lt_true / test_le_true / test_gt_true / test_ge_true all coincidentally land on 1). |
| 3 | `TestSmokeComparison::test_ne_true` | `IMM 10, PSH, IMM 20, NE, EXIT` | 1 | 0 | **cmp-polarity** | Same as #2 but inverted: NE-default emits 0 (no decoded OP_NE flag), so it only fails when the correct answer is 1; mirror twin of test_eq_false. |
| 4 | `TestSmokeAddress::test_lea_basic` | `ENT, IMM 0, LEA 2, EXIT` | != 0 | 0 | **never-dispatches** | ENT-main with implicit imm=0 leaves the LEA target offset path uninitialized (matches the L10 PSH `addr0_e0` MARK_AX missing-OP_ENT guard described in MEMORY.md / `project_l10_psh_addr_ent_bug.md`); LEA emits AX=0 instead of BP+2. |
| 5 | `TestSmokeMemory::test_si_li_roundtrip` | `IMM 0x200, PSH, IMM 42, SI, IMM 0x200, LI, EXIT` | 42 | 0 | **memory** | L13/L14 dep-anchor drift puts `layer13_mem_addr_gather` at L16 instead of L13 (see `SMOKE_MEMORY_TRACE_20260603.md`); the SI-write address tokens don't make it into the L15 memory_lookup window by the LI step, so LI reads zero. |
| 6 | `TestSmokeMemory::test_sc_lc_roundtrip` | `IMM 0x200, PSH, IMM 42, SC, IMM 0x200, LC, EXIT` | 42 | 0 | **memory** | Same root cause as #5 — char-width variant; the SC store + LC load go through the same L13-anchor → L15 lookup chain that is drifted. |
| 7 | `TestSmokeMemory::test_si_li_multiple_stores` | `(IMM 0x200 PSH IMM 10 SI) (IMM 0x300 PSH IMM 99 SI) IMM 0x300 LI EXIT` | 99 | 0 | **memory** | Same chain as #5; second SI's address slot collides with first due to L13 drift, so the LI at 0x300 sees no matching store. |
| 8 | `TestSmokeMemory::test_si_li_overwrite` | `(IMM 0x200 PSH IMM 10 SI) (IMM 0x200 PSH IMM 55 SI) IMM 0x200 LI EXIT` | 55 | 0 | **memory** | Same chain as #5; overwrite SI at same address still fails because the address-write path itself is drifted. |
| 9 | `TestSmokeMemory::test_si_li_16bit_value` | `IMM 0x200, PSH, IMM 0x1234, SI, IMM 0x200, LI, EXIT` | 4660 | 0 | **memory** | Same chain as #5 plus the high-nibble write path (`MUL partial / combine`) not propagating through L15 mem-lookup output. |
| 10 | `TestSmoke32Bit::test_add_16bit` | `IMM 200, PSH, IMM 100, ADD, EXIT` | 300 | 100 | **cascade-32bit** | ADD is in `_NEURAL_32BIT_OPS` and skips the highbit-imm legacy-ALU recovery (only handled neurally); the neural multi-byte ADD writeback chain produces zero in byte-1, leaving only the second operand's byte-0 = 100. |
| 11 | `TestSmoke32Bit::test_add_carry_cascade` | `IMM 0xFF, PSH, IMM 1, ADD, EXIT` | 256 (0x100) | 1 | **cascade-32bit** | Same as #10 — neural ADD byte-0 wraps to 0 and `OUTPUT_HI[0]` never sees the carry, so only the bare imm=1 leaks through (second operand). |
| 12 | `TestSmoke32Bit::test_sub_16bit` | `IMM 0x100, PSH, IMM 1, SUB, EXIT` | 255 (0xFF) | 4294967295 (0xFFFFFFFF) | **cascade-32bit** | Multi-byte SUB byte-0 computes 0x00-1 = 0xFF correctly but the borrow-out from byte-0 cascades into bytes 1/2/3 (all 0xFF instead of 0x00); SUB is in `_NEURAL_32BIT_OPS` so legacy-ALU recovery skips it. |
| 13 | `TestSmoke32Bit::test_or_16bit` | `IMM 0x0F00, PSH, IMM 0x00FF, OR, EXIT` | 4095 (0x0FFF) | 255 (0x00FF) | **cascade-32bit** | Multi-byte OR byte-0 is correct (0x00 \| 0xFF = 0xFF) but byte-1 OR is missing (0x0F \| 0x00 should = 0x0F, got 0x00); OR's hi-nibble writeback path stomped or absent. |
| 14 | `TestSmoke32Bit::test_xor_16bit` | `IMM 0x0F0F, PSH, IMM 0x00FF, XOR, EXIT` | 4080 (0x0FF0) | 65520 (0xFFF0) | **cascade-32bit** | Multi-byte XOR byte-0 is correct (0x0F^0xFF = 0xF0) but byte-1 XOR is INVERTED: got 0xFF instead of 0x0F (0x0F^0x00 = 0x0F expected); XOR hi-nibble FFN inverts or default-leaks 0xFF. |

## Category definitions

### memory (5 tests: #5–9)

**Cluster size: 5** — all `TestSmokeMemory::test_si_li_*` and
`test_sc_lc_roundtrip`. (`test_si_li_zero` is the false-positive twin
because expected==0.)

**Root cause**: L13/L14 dep-anchor retarget after commit `ae64239b`
did not propagate. Per `SMOKE_MEMORY_TRACE_20260603.md`:
`layer13_mem_addr_gather` bakes into `model.blocks[16].attn` (drifted
+3 layers); the L15 `memory_lookup` heads then read uninitialized
ADDR_B0_LO/HI rows because the ADDR-write rules fire too late in the
forward pass to be visible to the LI/LC read in the same step.

**Effort**: One agent. Single propagation fix in `l13_ops.py` / `l14_ops.py`
(retarget dep-anchor or move address-gather to a pre-L13 slot). Same fix
recovers all 5 tests (and likely improves #9 hi-byte path simultaneously).

**Refs**: `SMOKE_MEMORY_TRACE_20260603.md`, `SMOKE_TRIAGE_POST_L5.md`
cluster A.

### cascade-32bit (5 tests: #10–14)

**Cluster size: 5** — all `TestSmoke32Bit::test_*_16bit` /
`test_add_carry_cascade` plus the OR/XOR 16-bit variants.

**Root cause**: `_NEURAL_32BIT_OPS = {ADD, SUB, OR, XOR, AND}` are
intentionally excluded from the `f3342968` legacy-ALU collapsed-step
recovery (they're "handled by neural emit"). But the neural emit's
multi-byte writeback (`OUTPUT_LO`/`OUTPUT_HI` byte-1..byte-3 lanes) is
broken — either zero-fill (ADD), full-bit borrow leak (SUB), or
inverted default (XOR hi nibble). Common chain: the L8 5-stage
ADD/SUB block and the L9-L12 hi-lane writers each need their
`prev STACK0` operand recovered to compute the right hi byte, but the
neural STACK0 emit carries garbage in bytes 1+ (the same garbage that
forced `last_pushed_value` to be added in the `f3342968` fix).

**Effort**: One agent. Either extend `f3342968`'s collapsed-step recovery
to include `_NEURAL_32BIT_OPS` (and trust legacy ALU for multi-byte
results too — simplest), or fix the L8/L9 hi-byte writer chain to honor
`last_pushed_value`. The first option is dramatically simpler and would
recover all 5 in one diff at `batched_pure_neural.py:1990-2013` — just
drop the `exec_op not in _NEURAL_32BIT_OPS` gate (currently implicit via
`skipped_op in _BINARY_POP_OPS` — OR/XOR/AND/ADD/SUB ARE in
`_BINARY_POP_OPS`, but the legacy ALU still computes them so removing
the gate may already work; the question is why they don't fire today
when test_or_basic passes but test_or_16bit fails).

Verification note: `test_or_basic` (`IMM 0x0F, PSH, IMM 0x30, OR, EXIT`
expected 0x3F) passes, but `test_or_16bit` (same shape with
`IMM 0x0F00` first operand) fails — so the `f3342968` recovery DOES
fire for OR when the imm is single-byte (0x0F), but NOT when the first
operand is multi-byte (0x0F00). Suspect: `last_pushed_value` snapshot
is taken from `prev_ax` which is one-byte for small operands but zero-
truncated for multi-byte, OR the collapsed-step pattern detection
(`post_idx == skipped_idx + 1`) doesn't match the multi-byte trace.
Either way, one investigation covers all 5.

**Refs**: `f3342968` commit message; `SMOKE_TRIAGE_POST_L5.md`
clusters C+E.

### cmp-polarity (2 tests: #2–3)

**Cluster size: 2** — `test_eq_false`, `test_ne_true`. Their twins
(test_eq_true, test_lt_true, test_le_true, test_gt_true, test_ge_true)
all pass because they happen to expect the value (1) that the
broken cmp-default path emits.

**Root cause**: `SMOKE_COMPARISON_OP_DECODE_MISSING.md` confirms `OP_EQ`
dim 201 is ZERO at every position in every block — the L5 opcode-decode
FFN never produces any `OP_<NAME>` flag (no W_down writer to the
`OP_LEA..OP_GETCHAR` dim range). The `ComparisonCombine` FFN's
default-units are gated on `OP_<NAME> > 0`; with the gate always closed,
the cmp returns a constant 1 (or 0 — needs verification per op).

**Effort**: One agent. Either patch the L5 opcode-decode FFN to write
`OP_<NAME>` for the comparison ops, or add a collapsed-step recovery
hook in `batched_pure_neural.py` that detects the IMM,PSH,IMM,<cmp>,EXIT
pattern (already handled by the `f3342968` recovery for EQ/NE/LT/GT/GE/LE
since they ARE in `_BINARY_POP_OPS`) and confirm why it's not firing
for test_eq_false/test_ne_true specifically. Quick hypothesis: the
collapsed-step detection requires `last_pushed_value is not None`,
which is set on PSH dispatch; if the model emits a different `last_pc`
shape when both operands fit in one byte, the recovery may skip these
two cases.

**Refs**: `SMOKE_COMPARISON_OP_DECODE_MISSING.md`, `CMP_PATH_AUDIT.md`,
`CMP_POLARITY_INVESTIGATION_2026_06_03.md`, `IF_EQ_CMP_DEFAULT_LEAK.md`.

### specific-opcode (1 test: #1)

**Cluster size: 1** — `test_xor_basic` only. AND_basic and OR_basic pass.

**Root cause** (from `f3342968` commit message): "IMM 0xFF sign-extension".
The highbit-imm fix at `batched_pure_neural.py:2000-2002` sign-extends
the IMM if `imm_val >= 0x800000`. But `IMM 0xFF` is 8-bit and falls
below that threshold (0xFF < 0x800000), so it's NOT sign-extended —
correct AX = 0xFF. The bug must be that the second IMM (0xD5) is
*also* < 0x800000 and is read into the result instead of XORed; the
model emits 0xD5 (the second IMM byte) and the AX override either
doesn't fire or computes `0xFF ^ 0xD5 = 0x2A` correctly but then the
final emit reads a stale value. Hypothesis: the legacy ALU produces
0x2A correctly but the `_override_register_in_last_step` call writes
to a token slot that gets overwritten by a subsequent IMM-step write
that the highbit-imm path didn't anticipate.

**Effort**: One agent. Independent of memory/cascade — read
`_override_register_in_last_step` and compare the XOR writeback to the
SUB/DIV writeback that DOES work.

**Refs**: `f3342968` commit message ("xor_basic + lea_basic remain
failing under different root causes (IMM 0xFF sign-extension...)").

### never-dispatches (1 test: #4)

**Cluster size: 1** — `test_lea_basic` only.

**Root cause** (from MEMORY.md): The L10 PSH `addr0_e0` MARK_AX guard is
missing an OP_ENT check. ENT-main with no implicit imm leaves
`MEM_addr0 = 0xE0` pinned on every step, blocking LEA from computing
`BP + imm` correctly. LEA writes 0 because the BP-read path is gated by
a stale MARK_AX value.

**Effort**: One agent. Single guard add at `l10_ops.py:3888-3927`.
Already documented in MEMORY.md (`project_l10_psh_addr_ent_bug.md`)
as an open bug blocking the func_identity_* 1096 cluster.

**Refs**: `MEMORY.md`'s `project_l10_psh_addr_ent_bug.md` link; this
LEA failure is the smoke-visible symptom.

## Recommended next-round fix priority

| Pri | Target | Tests recovered | Effort | Independent? |
|---|---|---|---|---|
| **P0** | memory cluster (single L13/L14 anchor retarget) | 5 | M | Yes |
| **P1** | cascade-32bit (drop `_NEURAL_32BIT_OPS` from highbit-imm gate) | 5 | S-M | Yes — same file as f3342968 |
| **P2** | cmp-polarity (find why f3342968 recovery skips eq_false/ne_true) | 2 | S | Yes — same file as f3342968 |
| **P3** | test_xor_basic specific (0xFF override race) | 1 | S | Yes |
| **P4** | test_lea_basic ENT-main guard | 1 | S | Yes — but also fixes 1096 cluster |

**Best parallelism**: P1, P2, P3 all live in `batched_pure_neural.py`
near the f3342968 fix; conservatively run as ONE agent to avoid
worktree conflicts. P0 (memory) and P4 (LEA) are independent files,
parallel-safe.

**Net effect if all 5 P0-P4 land**: 14/14 → 0/14 smoke failures
(but P0 memory is the only "complex" one; P1-P4 are mostly mechanical
extensions of the f3342968 pattern or guard adds).

## Pure-neural status (informational only)

11 XPASS / 29 XFAIL in `test_smoke_pure_neural.py`. None of the XPASSes
are listed as failures here — the pure-neural suite is the Phase 8
parallel suite that was conservatively marked all-xfail; XPASSes are
expected as phase work lands. No action needed; xfail markers should
be flipped to PASS as each operand-specific phase suite ships.

## Reproducibility notes

- One full smoke run only (constraint): `python -m pytest tests/test_smoke.py
  tests/test_smoke_pure_neural.py --tb=no -v` (wall 990s = 16:30 in
  fresh worktree).
- Zero compile changes (constraint).
- No individual reruns needed: the full-run `tail -120` output included
  every `FAILED ... expected X, got Y` line for all 14 failures, so the
  per-test `got` values are from the same single run.
- Existing root-cause docs referenced (do NOT re-investigate):
  - `SMOKE_TRIAGE_POST_L5.md` (predecessor; 26 → 14)
  - `SMOKE_MEMORY_TRACE_20260603.md` (memory cluster A)
  - `SMOKE_COMPARISON_OP_DECODE_MISSING.md` (cmp-polarity)
  - `CMP_PATH_AUDIT.md`, `CMP_POLARITY_INVESTIGATION_2026_06_03.md`
  - `IF_EQ_CMP_DEFAULT_LEAK.md`
  - MEMORY.md `project_l10_psh_addr_ent_bug.md` (LEA)
  - `f3342968` commit message (highbit-imm scope + remaining cases)
