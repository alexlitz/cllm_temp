# FUNCMIN_FIX — unblock the −359 SeRecover deletion (fix the func_min id675 regression)

**Branch** `serecover-delete` · **Fix flag** `C4_ALU_OPERAND_SURVIVE_CMP_RAW`
(**DEFAULT-OFF** = OP_LT excluded from the raw spare; opt-in `=1` restores the
old all-CMP-in behaviour) · **Parent flag** `C4_ALU_OPERAND_SURVIVE`
(default-ON, the −359 delete) · **Config** spec_k=0, full_trace, campaign
default.

## TL;DR

The `serecover-delete` branch deletes `Cmp`+`Mul` `OperandSeRecoverFFN`
(−359 LOC) behind the default-ON combined flag `C4_ALU_OPERAND_SURVIVE`
(block-15 L9-clear raw-band SPARE + block-17 head-4 CMP Q-veto). The fast gate
scored **+12 net with 4 GAINS** (`bool_and` id1088, `func_mul` id612, `if_eq`
id408, `if_var` id433) but **1 REGRESSION: `func_min` id675** (`min(13,57)`),
which blocked the landing (any regression blocks).

**Root:** the regression is the **block-15 L9-clear raw-band SPARE**. That spare
originally added a `−1e6` NOT-blocker for ALL SIX CMP opcodes, keeping the raw
`ALU_LO/HI@AX` operand-A band ALIVE on every CMP AX row. But the raw-band CMP
spare is a **zero-sum lever ACROSS CMP opcodes** — it is load-bearing for some
CMP verdicts and destructive for others:

* **`OP_LT`** — func_min id675's LT-result writer at logical L14 (physical block
  29) is gated on the raw `ALU@AX` band being CRUSHED all-negative. With LT
  spared the band stays a clean `+6` one-hot → the writer's crush-gate never
  trips → the correct `0x01` is never written → the L25 tail (block 45) leaks
  `0xE8`. **LT must be EXCLUDED** for func_min to pass.
* **`OP_EQ / NE / GT / LE / GE`** — the `if_eq` id408 (LE) and `bool_and` id1088
  (GT) gains DO read the spared raw operand. Excluding them regresses both:
  leak trace (`_probe_funcmin_leak.py`, teacher-forced correct tape, FINAL AX
  byte) shows **id408 goes NaN at block 46** and **id1088's correct `0x00` at
  block 29 is clobbered to `0xE0` at block 37 then `0x01` at block 45**. **These
  five must STAY spared** for the gains to hold.

**Fix:** exclude **ONLY `OP_LT`** from the raw-band spare.
`_CMP_OPCODES_SPARED = (EQ, NE, GT, LE, GE)` stays in `_spare_opcodes()`; `LT`
is dropped by default. The new opt-in `C4_ALU_OPERAND_SURVIVE_CMP_RAW=1`
restores the original all-CMP-in behaviour (LT back → func_min regresses).

> ⚠ An earlier draft of this fix excluded ALL SIX CMP opcodes and claimed "the
> CMP result never reads the raw band, so every gain holds." That was
> **FALSIFIED** by the spec_k=0 verdict (`id408 FAIL`, `id1088 FAIL`) and the
> leak trace above. The corrected fix excludes only `OP_LT`.

## 1. Diagnosis (leak trace, teacher-forced correct tape)

`tools/_probe_funcmin_leak.py` forward-hooks every block, decodes the AX result
byte (OUTPUT_LO/HI @ the AX-marker row) per block INPUT + FINAL at the compare
step, and dumps the surviving raw `ALU_LO/HI@AX` band. The FINAL AX byte is a
reliable proxy for compare-step correctness. Comparing the three raw-spare
configs on the four CMP programs:

| id | prog | opcode | exp AX | all-CMP-in (orig, `CMP_RAW=1`) | all-CMP-**out** (rejected draft) | **LT-only-out (fix)** |
|---|---|---|---|---|---|---|
| 675  | func_min | LT | `0x01` | `0xE8` ✗ (spare alive → writer off) | `0x01` ✓ | **`0x01` ✓** |
| 408  | if_eq    | LE | `0x01` | `0x01` ✓ | **NaN** @blk46 ✗ | **`0x01` ✓** |
| 1088 | bool_and | GT | `0x00` | `0x00` ✓ | `0xE0`→`0x01` @blk37/45 ✗ | **`0x00` ✓** |
| 433  | if_var   | GT | `0x00` | `0x00` ✓ | `0x00` ✓ | **`0x00` ✓** |

The raw `ALU@AX` band in the LT-only-out fix: **crushed `−39`** on the id675 LT
row (writer fires → `0x01`), **alive `+6`** on the id408/id1088/id433 EQ/GT/LE
rows (no NaN, no downstream clobber). So excluding only LT gives every program
the raw-band state its verdict needs. (id433 additionally rides the block-17
head-4 CMP Q-veto — SE band, unchanged.)

The CMP RESULT for LT/GT is written at logical L14 (physical block 29); the
comparison PRIMITIVES themselves come from the L9 nibble comparator SE band
(`docs/DERIVE_CMP_2026_07_09.md`), but the block-29 result WRITER and the L25
tail (block 45) are gated on the raw `ALU@AX` band state, which is exactly what
the spare toggles — hence the zero-sum-across-opcodes behaviour.

## 2. Why LT-only holds all four gains

* `func_min` id675 (LT): raw band crushed → block-29 LT writer fires → `0x01`.
* `if_eq` id408 (LE): raw band spared → EQ path reads a live `+6` operand, no
  NaN → `0x01`.
* `bool_and` id1088 (GT): raw band spared → GT result `0x00` survives, no
  block-37/45 clobber.
* `if_var` id433 (GT): raw band spared + block-17 head-4 Q-veto → `0x00`.
* `func_mul` id612 (MUL): ARITH spare unchanged (`OP_MUL` always spared).

`OP_LT` is the only opcode toggled; among the sampled ids only func_min uses LT,
so LT-only exclusion fixes func_min with zero collateral on the gains.

## 3. The scoped fix (`neural_vm/unified_compiler/ops/l9_ops.py`)

* `_SPARE_OPCODES_ARITH = (OP_MUL, OP_DIV, OP_MOD, OP_OR, OP_XOR, OP_AND)` — the
  opcodes whose downstream engines read operand-A directly from the raw
  `ALU_LO/HI@AX` band.
* `_CMP_OPCODES_SPARED = (OP_EQ, OP_NE, OP_GT, OP_LE, OP_GE)` — the five CMP
  opcodes whose verdict reads the spared raw operand (if_eq/bool_and gains).
* `_spare_cmp_raw_enabled()` reads `C4_ALU_OPERAND_SURVIVE_CMP_RAW`
  (default `"0"`).
* `_spare_opcodes()` returns `_CMP_OPCODES_SPARED + _SPARE_OPCODES_ARITH` by
  default (LT excluded), or `_CMP_OPCODES + _SPARE_OPCODES_ARITH` when the
  opt-in flag is set (LT back in).
* `_alu_clear_rules` adds the `(op, -1e6)` NOT-blocker for each opcode in
  `_spare_opcodes()`.

Pure opcode-set change on the block-15 clear's AND gate (adds/removes the
`OP_LT` `-1e6` blocker COLUMN, not units → 3405-unit L9 layout preserved),
weight-affecting, so ON/OFF must not share a serialised cache entry: registered
in BOTH cache-key snapshots (`full_vm_compiler_dynamic.py`).

## 4. Verdict proof (CPU, spec_k=0, full_trace — the fast-gate criterion)

`tools/_probe_funcmin_verdict.py` runs `BatchedPureNeuralRunner.
run_batch_fail_fast(spec_k=0, criterion="full_trace")` — the SAME criterion the
fast gate uses. All three raw-spare configs MEASURED on CPU (see §7 for
commands):

| id | program | exp | all-CMP-in (`CMP_RAW=1`, orig) | all-CMP-out (rejected draft) | **LT-only-out (fix, default)** |
|---|---|---|---|---|---|
| 675  | `func_min` min(13,57)     | 13   | **FAIL** (decoded 65512) | PASS | **PASS (regression FIXED)** |
| 612  | `func_mul` mul(41,29)     | 1189 | PASS | PASS | **PASS (gain HOLDS)** |
| 408  | `if_eq`   7==7            | 1    | PASS | **FAIL** (decoded 0) | **PASS (gain HOLDS)** |
| 433  | `if_var`  x=35, x>76      | 0    | PASS | PASS | **PASS (gain HOLDS)** |
| 1088 | `bool_and` 57>65 && 65>18 | 0    | PASS | **FAIL** (decoded 1) | **PASS (gain HOLDS)** |

* **all-CMP-in** (the original branch) = the fast-gate config: 4 gains PASS,
  `func_min` FAILS = the reported +12/−1-regression.
* **all-CMP-out** (the rejected first draft): fixes func_min but REGRESSES
  id408 + id1088 → the empirical falsification that forced the LT-only scoping.
* **LT-only-out** (this fix, default): **5/5 PASS** — func_min unblocked AND all
  4 gains intact. This is the 0-regression config.

## 5. Golden — flag-OFF byte-identical to the branch's fork-point main default

* Branch **flag-OFF** (`C4_ALU_OPERAND_SURVIVE=0`) golden
  `state_dict_sha256 = e50521f3...` (`tools/_isa_golden_hash.py`,
  `disk_cache=False`, campaign default) — **UNCHANGED** by this fix (the LT-only
  exclusion only touches the ON raw-band spare; OFF the whole spare is absent).
  `e50521f3` is exactly the main default golden at the branch's merge-base
  `aead086e` ("Merge blog-p0p1 … byte-identical golden e50521f3").
* ⚠ **Rebase note:** current main has advanced 12 commits past `aead086e` and
  flipped two default-golden-changing flags ON — `C4_R_FRAME_TAIL`
  (`e50521f3` → `b9a74424`) and `C4_MUL_B1_DELIVERY` (→ `c18ef9f9`). Current
  main **default** golden is therefore `c18ef9f9`, not `e50521f3`. The func_min
  fix is golden-neutral relative to its base; to make the branch's flag-OFF
  equal to current main default, the branch must be rebased / main merged in (a
  landing-logistics step, not a defect in this fix). The −359 delete + the two
  survival flags are orthogonal to the two main-ahead flips, so the rebase is
  expected to be conflict-light.

## 6. Files

* `neural_vm/unified_compiler/ops/l9_ops.py` — `_SPARE_OPCODES_ARITH`,
  `_CMP_OPCODES_SPARED`, `_spare_cmp_raw_enabled`, `_spare_opcodes`,
  `_SPARE_OPCODES` alias; `_alu_clear_rules` uses `_spare_opcodes()`.
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` —
  `C4_ALU_OPERAND_SURVIVE_CMP_RAW` in both cache-key snapshots.
* `tools/_probe_funcmin_leak.py` — the per-block AX-byte leak trace (diagnosis).
* `tools/_probe_funcmin_verdict.py` — the spec_k=0 full_trace verdict.
* `docs/FLAG_REGISTRY.md` — `C4_ALU_OPERAND_SURVIVE_CMP_RAW` entry.

## 7. Gate — DEFERRED (GPU run by the operator)

Fix verified on CPU (leak trace §1 + spec_k=0 full_trace §4). The authoritative
A/B fast gate is deferred to a free GPU / main. Target: **0 regressions, +12 net
or better** (the 4 gains + func_min now passing; flag-OFF byte-identical so no
passing program can move under OFF).

```
cd c4_release
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$(pwd) \
C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
python tools/fast_gate.py --flag C4_ALU_OPERAND_SURVIVE
```

CPU re-verify of the 5 ids (fast-gate criterion):

```
C4_ALU_OPERAND_SURVIVE=1 python tools/_probe_funcmin_verdict.py \
  --ids 675,612,408,433,1088
```

Leak-trace re-verify (fast, teacher-forced FINAL AX byte per program):

```
C4_ALU_OPERAND_SURVIVE=1 python tools/_probe_funcmin_leak.py \
  --ids 675,1088,408,433
```
