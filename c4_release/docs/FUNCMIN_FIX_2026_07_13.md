# FUNCMIN_FIX — unblock the −359 SeRecover deletion (fix the func_min id675 regression)

**Branch** `serecover-delete` · **Fix flag** `C4_ALU_OPERAND_SURVIVE_CMP_RAW`
(**DEFAULT-OFF**, opt-in `=1` restores old behaviour) · **Parent flag**
`C4_ALU_OPERAND_SURVIVE` (default-ON, the −359 delete) ·
**Config** spec_k=0, full_trace, campaign default.

## TL;DR

The `serecover-delete` branch deletes `Cmp`+`Mul` `OperandSeRecoverFFN`
(−359 LOC) behind the default-ON combined flag `C4_ALU_OPERAND_SURVIVE`
(block-15 L9-clear raw-band SPARE + block-17 head-4 CMP Q-veto). The fast gate
scored **+12 net with 4 GAINS** (`bool_and` id1088, `func_mul` id612, `if_eq`
id408, `if_var` id433) but **1 REGRESSION: `func_min` id675** (`min(13,57)`),
which blocked the landing (any regression blocks).

**Root:** the regression is NOT the block-17 head-4 CMP Q-veto (that is what
delivers the 4 CMP gains and is unchanged), nor a downstream CMP margin
(`docs/DERIVE_CMP_2026_07_09.md`). It is the **block-15 L9-clear raw-band
SPARE over-firing on CMP rows**. The spare's original opcode set included the
six CMP opcodes (`_CMP_OPCODES`), so on a CMP AX row it kept the raw
`ALU_LO/HI@AX` operand-A band ALIVE. But the CMP RESULT is computed from the
**SE_ALU** band (the L9 nibble comparator, `docs/DERIVE_CMP_2026_07_09.md`
`CMP+0..3` primitives), NOT the raw `ALU@AX` band — so sparing the raw CMP
operand does not help the comparison. What it DOES do is break `func_min`: its
LT-result writer at logical L14 (physical block 29) is gated on the raw
`ALU@AX` band being CRUSHED (all-negative); with the spare keeping it alive the
writer stops firing, the correct `0x01` LT result is lost, and the L25 tail
(physical block 45) leaks `0xE8`.

**Fix (scoped, byte-safe):** remove the six CMP opcodes from the raw-band spare.
`_spare_opcodes()` now returns only the ARITH/BITWISE set
(`OP_MUL/DIV/MOD/OR/XOR/AND`) whose downstream engines DO read operand-A from
the raw `ALU@AX` band. A new opt-in flag `C4_ALU_OPERAND_SURVIVE_CMP_RAW`
(default-OFF) restores the original all-CMP-in behaviour. The 4 CMP gains HOLD
because they come from the block-17 head-4 Q-veto (SE band), which is untouched;
`func_min` id675 passes because its LT writer sees the crushed raw band again.

## 1. Diagnosis (spec_k=0, faithful-autoregressive)

`func_min` id675 source: `int min(int a,int b){ if(a<b) return a; return b; }
int main(){ return min(13,57); }` → expected exit 13. The compare is `a<b`
(LT, 13<57 → true → return a=13).

`tools/_probe_funcmin_leak.py` (teacher-forced over the correct DraftVM tape,
decoding the AX result byte per block INPUT + FINAL at the LT compare row, and
dumping the surviving raw `ALU_LO/HI@AX` band) isolates the flip:

* The CMP cascade (`SE_ALU` band, `CMP+0..3` primitives) is **byte-identical**
  ON-vs-OFF — the comparison itself is unaffected by the raw-band spare. This
  matches `docs/DERIVE_CMP_2026_07_09.md`: LT/GT/EQ reduce to the L9 nibble
  comparator's zero-detector + sign primitives on the SE band, never the raw
  `ALU_LO/HI@AX` operand band.
* At logical L14 (physical block 29) the LT-result writer emits the correct
  `0x01` **only when the raw `ALU@AX` band is crushed all-negative**. With
  `C4_ALU_OPERAND_SURVIVE=1` (original all-CMP spare) that band survives as a
  clean `+6` one-hot on the CMP row → the writer's crush-gate no longer trips →
  the `0x01` is never written → the L25 tail (physical block 45) falls through
  to its `0xE8` default → wrong AX byte → id675 fails.

So the regression is specifically the **CMP subset of the block-15 raw-band
spare** interacting with func_min's crush-gated LT writer. It is not the
block-17 veto (SE band) and not a DERIVE_CMP margin.

## 2. Why the 4 CMP gains are independent of the raw-band CMP spare

The three CMP-family gains — `if_eq` id408 (`7==7`), `if_var` id433
(`x=35; x>76`), `bool_and` id1088 (`57>65 && 65>18`) — are delivered by the
**block-17 (logical L11) head-4 CMP Q-veto** (`model_ops.make_cmp_h4_qveto_op`,
phase 1450). That veto overwrites head-4's `W_q` at the six CMP columns
(`K@CONST` self-row slots 0/33) with `−1e5` so the un-gated MEM→ALU load head
stops crushing the clean GT/EQ one-hot on cmp RESULT rows. It reads/writes the
`ALU@AX` band via the attention head, not via the L9-clear spare, and it is
**unchanged** by this fix (still gated on the parent `C4_ALU_OPERAND_SURVIVE`).
`func_mul` id612 comes from the ARITH spare (`OP_MUL`), which is retained.

Therefore excluding the six CMP opcodes from the raw-band spare fixes func_min
while every gain holds. The deleted `CmpOperandSeRecoverFFN` stays inert: it
recovered the raw operand from SE, but since the CMP RESULT never reads the raw
band, a crushed raw CMP operand is harmless to the verdict.

## 3. The scoped fix (`neural_vm/unified_compiler/ops/l9_ops.py`)

* `_SPARE_OPCODES_ARITH = (OP_MUL, OP_DIV, OP_MOD, OP_OR, OP_XOR, OP_AND)` — the
  opcodes whose downstream engines (L10 wide_mul / L11 divmod / L10 bitwise)
  read operand-A directly from the raw `ALU_LO/HI@AX` band, so they MUST survive
  the L9 clear.
* `_spare_cmp_raw_enabled()` reads `C4_ALU_OPERAND_SURVIVE_CMP_RAW`
  (default `"0"`).
* `_spare_opcodes()` returns `_SPARE_OPCODES_ARITH` by default (CMP excluded),
  or `_CMP_OPCODES + _SPARE_OPCODES_ARITH` when the opt-in flag is set.
* `_alu_clear_rules` adds the `(op, -1e6)` NOT-blocker for each opcode in
  `_spare_opcodes()` (was `_SPARE_OPCODES`).

The flag is a pure opcode-set change on the block-15 clear's AND gate (adds /
removes CMP `-1e6` blocker COLUMNS, not units → 3405-unit L9 layout preserved),
weight-affecting, so ON/OFF must not share a serialised cache entry:
registered in BOTH cache-key snapshots
(`full_vm_compiler_dynamic.py` in-proc memo + disk-cache `kwargs_snapshot`).

## 4. Verdict proof (CPU, spec_k=0, full_trace)

`tools/_probe_funcmin_verdict.py` runs `BatchedPureNeuralRunner.
run_batch_fail_fast(spec_k=0, criterion="full_trace")` — the SAME criterion the
fast gate uses (not the teacher-forced interp_oracle_gate, which over-flags
cross-step programs). With the scoped fix (`C4_ALU_OPERAND_SURVIVE=1`,
`C4_ALU_OPERAND_SURVIVE_CMP_RAW` unset):

| id | program | expected | verdict |
|---|---|---|---|
| 675  | `func_min` min(13,57)      | 13   | **PASS (regression FIXED)** |
| 612  | `func_mul` mul(41,29)      | 1189 | **PASS (gain HOLDS)** |
| 408  | `if_eq`   7==7             | 1    | **PASS (gain HOLDS)** |
| 433  | `if_var`  x=35, x>76       | 0    | **PASS (gain HOLDS)** |
| 1088 | `bool_and` 57>65 && 65>18  | 0    | **PASS (gain HOLDS)** |

_(Empirical run: see §7 for the exact command; the CPU verdict path is the
fast-gate criterion so these five-of-five PASS = func_min unblocked + 4 gains
intact.)_

## 5. Golden — flag-OFF byte-identical to the branch's fork-point main default

* Branch **flag-OFF** (`C4_ALU_OPERAND_SURVIVE=0`) golden
  `state_dict_sha256 = e50521f3...` (`tools/_isa_golden_hash.py`,
  `disk_cache=False`, campaign default) — **UNCHANGED** by this fix (the scoped
  CMP exclusion only touches the ON raw-band spare; OFF the whole spare is
  absent). `e50521f3` is exactly the main default golden at the branch's
  merge-base `aead086e` ("Merge blog-p0p1 … byte-identical golden e50521f3").
* ⚠ **Rebase note:** current main has advanced 12 commits past `aead086e` and
  flipped two default-golden-changing flags ON —
  `C4_R_FRAME_TAIL` (`e50521f3` → `b9a74424`) and `C4_MUL_B1_DELIVERY`
  (→ `c18ef9f9`). Current main **default** golden is therefore `c18ef9f9`, not
  `e50521f3`. The func_min fix is golden-neutral relative to its base; to make
  the branch's flag-OFF equal to current main default, the branch must be
  rebased / main merged in (a landing-logistics step, not a defect in this fix).
  The −359 delete + operand-survival flags are orthogonal to the two
  main-ahead flips, so the rebase is expected to be conflict-light.

## 6. Files

* `neural_vm/unified_compiler/ops/l9_ops.py` — `_SPARE_OPCODES_ARITH`,
  `_spare_cmp_raw_enabled`, `_spare_opcodes`, `_SPARE_OPCODES` back-compat alias;
  `_alu_clear_rules` uses `_spare_opcodes()`.
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` —
  `C4_ALU_OPERAND_SURVIVE_CMP_RAW` in both cache-key snapshots.
* `tools/_probe_funcmin_leak.py` — the per-block AX-byte leak trace (diagnosis).
* `tools/_probe_funcmin_verdict.py` — the spec_k=0 full_trace verdict A/B.
* `docs/FLAG_REGISTRY.md` — new `C4_ALU_OPERAND_SURVIVE_CMP_RAW` entry.

## 7. Gate — DEFERRED (GPU run by the operator)

Fix verified on CPU (spec_k=0 full_trace, §4). The authoritative A/B fast gate
is deferred to a free GPU / main. Target: **0 regressions, +12 net or better**
(the 4 gains + func_min now passing, no new regression; flag-OFF byte-identical
so no passing program can move under OFF).

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
