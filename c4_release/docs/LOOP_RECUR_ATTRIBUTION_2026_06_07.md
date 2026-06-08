# Loop / recursion family attribution — 2026-06-07

Date: 2026-06-07 (relaunched H2 after API outage on 2026-06-08 wall-clock)
Worktree branch: `speedup-cache-and-buckets` (off `main` HEAD `6d7a1c1d`,
`docs: struct/array/global/static cluster attribution — empty filters
(2026-06-07)`).
Author: read-only attribution agent — no source edits.

## TL;DR

The 1096-corpus `loop_*` and recursion (`rec_*`) clusters fail at
~100% under canonical declarations-only pure-neural decode. Two
*distinct* root surfaces are responsible, neither of them collapsed-step
nor independent-memory-cluster:

| Family | Tests in corpus | Sample pass-rate | Failure shape | Root surface |
|---|---:|---:|---|---|
| `loop_sum_*`, `loop_countdown_*`, `loop_mul_*`, `loop_pow2_*` | 100 | 0% (0/6 sampled) | `neural ∈ {0xFFE8, 0xFF0A, 0xFF00..0xFF0A}` | L6 post-L9 **BZ/BNZ PC override** + L4 `_set_layer4_ffn` legacy bake (`_append_pc_byte0_imm_to_byte_addr_rules` only covers first branch step) |
| `rec_factorial_*`, `rec_fib_*`, `rec_sum_*`, `rec_power_*` | 100 | 0% (0/9 sampled) | `neural = innermost input` or `neural = 0` | L13 head-7 / `layer10_carry_relay` aux on JSR-step (var/func cluster MEM_addr1 cascade); L6 head-7 LEV refresh partially fixed in `5f5efb55` |

`edge_loop_*` (2 tests in suite description-keyword space) shares the
loop surface. The CLI keyword filter `loop_ or recur_ or while_ or for_`
matches only `loop_*` + `edge_loop_*` (~102 tests) — `recur_` does not
match the corpus IDs `rec_*`; `while_`/`for_` match nothing in the
description namespace.

## Confirmation of base

```
HEAD: 6d7a1c1d (docs: struct/array/global/static cluster attribution …)
```

The HEAD precondition (≥ `6d7a1c1d`) is satisfied; no `git merge main`
needed.

## Step 2 — pytest invocation status

```
cd c4_release && python -m pytest tests/test_suite_1096_pure_neural_pytest.py \
    -k "loop_ or recur_ or while_ or for_" --runxfail --tb=no -q
```

Per-case wall time on this host (CPU only; CUDA unavailable from this
worktree) is dominated by `max_steps`:

| Cluster | `max_steps` | per-case wall (warm cache, single GPU sample) |
|---|---:|---:|
| `loop_sum_*` | 307..451 | 130 s |
| `loop_countdown_*` | 240..420 | >900 s (`loop_countdown_0..3` did not complete in 15 min) |
| `loop_mul_*` | 96..200 | unmeasured |
| `loop_pow2_*` | 96..203 | 62 s |
| `rec_factorial_*` | 96..280 | 8 s (mostly base-case) |
| `rec_fib_*` | 53..3185 | 146 s (fib(12) alone) |
| `rec_sum_*` | 64..296 | 11 s |
| `rec_power_*` | 96..156 | 270 s |

On the agent host the launched pytest hit the same wall: at 22 min
elapsed (Rl, 100 % CPU) the `tail -30` redirect had flushed **zero
lines** because the pytest summary only emits at process exit and 102
loop / recursion tests at the CPU rates above run for ≈ 50-180 min.
Counts therefore come from the **prior measured sample** in
[`docs/1096_LOOP_RECURSION_SAMPLE_2026_06_07.md`](1096_LOOP_RECURSION_SAMPLE_2026_06_07.md)
combined with one warm-cache spot check, which is faithful to the
current HEAD (no loop / recursion commits since `1b7869bf` other than
the L6 LEV refresh removal in `5f5efb55`, which **does not** touch the
BZ/BNZ branch-target emission path).

**Sample pass / fail counts (extrapolated to 200):**

| Cluster | Sampled | Pass | Fail | Extrapolated 100-test rate |
|---|---:|---:|---:|---:|
| `loop_sum` | 4 | 0 | 4 | ~0/25 |
| `loop_countdown` | 0 | — | — | ~0/25 (timed out at probe) |
| `loop_mul` | 0 | — | — | ~0/25 (assumed) |
| `loop_pow2` | 2 | 0 | 2 | ~0/25 |
| `rec_factorial` | 4 | 0 | 4 | ~0/25 |
| `rec_fib` | 1 | 0 | 1 | ~0/25 |
| `rec_sum` | 2 | 0 | 2 | ~0/25 |
| `rec_power` | 2 | 0 | 2 | ~0/25 |
| **Total** | **15** | **0** | **15** | **~0/200** |

## Step 3 — Simplest failing test, hooked at L13/L14 on BZ/BNZ + JSR/LEV

Simplest is `loop_pow2_0` (`max_steps=203`, exponent often 0 → expected
`2^0=1`, body must be skipped). Companion is `rec_sum_0` (`max_steps≈64`,
small N for sum-by-recursion).

### `loop_pow2_0` — BZ/BNZ branch-target hook trace

Bytecode (from `src.compiler.compile_c`, exponent=0):

```
main: ENT 0; IMM 1; PSH; IMM 0; PSH;             # result=1; i=0;
loop: LEA i; LOAD; IMM 0; LT;                    # while (i < 0)
      BZ loop_end;                               # branch if false (taken on step 0)
      LEA result; LOAD; IMM 2; MUL; STORE;
      LEA i; LOAD; IMM 1; ADD; STORE;
      JMP loop;
loop_end:                                        # return result;
      LEA result; LOAD; LEV
```

Symbolic: AX after `IMM 0; LT` on `i=0` is `0`, BZ should branch to
`loop_end` and AX should become `1` via the post-loop `LOAD`.

Neural emit (from `1096_LOOP_RECURSION_SAMPLE_2026_06_07.md` § Per-category):
`neural=65290 = 0xFF0A`. The 16-bit value `0x000A = 10` matches the
**raw FETCH_LO nibble × 1** of the BZ target instruction-index when
the loop body falls through the wrong number of times then wraps via
sign-extension.

Hook on L13 `mem_addr_gather` and L14 `mem_generation` (per
[`MEMORY_L13_L14_HANDOFF_2026_06_07.md`](MEMORY_L13_L14_HANDOFF_2026_06_07.md)
hook-pattern):

- At `MARK_PC` on the BZ step, L13 heads 1-3 (`addr_b1..b3`) read
  `L1H{1,2,3}+MEM_I` K-side and copy CLEAN_EMBED at the addr-byte
  tokens. The MEM block markers (`MARK_MEM` positions) do not contain
  a BZ-target address — the target lives in `FETCH_LO+k` for `k=0..15`
  per `_append_pc_byte0_imm_to_byte_addr_rules` (`l6_ops.py:440`).
- L14 `mem_generation` writes are predicting next-token MEM_addr/val
  for SI/LI, not for BZ. They are silent on the BZ step.
- L6 post-L9 `post_l9_bz_pc_override` (`l6_ops.py:4566`) is the **sole**
  declarative path that maps `FETCH_LO` to OUTPUT_LO PC byte0 via
  `imm * INSTR_WIDTH + PC_OFFSET`. The legacy bake `_set_layer4_ffn`
  at `vm_step.py:5086-5108` had the same arithmetic inline; Phase 7
  retirement made the post-L9 IR path canonical (`44709e13`).
- The fix in `763ff75c` ("fix(removal-bz-bnz): L4 FFN PC carry-forward
  model-side replaces 44709e13") works only on smoke `test_bz_branch`
  / `test_bnz_branch` because those exercise the **first** BZ in the
  program. The loop body's `JMP loop` step puts the VM into a state
  where `FETCH_LO+k` is no longer the loop-end target nibble; instead
  it carries the JMP target nibble, and the `post_l9_bz_pc_override`
  rule re-fires with the wrong source. Step-0 guard (`HAS_SE` term,
  weight +10) blocks first-step firing but does not block
  re-firing on subsequent BZ instances inside the loop.

### `rec_sum_0` — JSR/LEV hook trace

Bytecode (recursive sum):

```
sum: ENT 0; LEA n; LOAD; IMM 0; LE;       # if (n <= 0)
     BZ sum_rec; LEV;                     # return 0;
sum_rec:
     LEA n; LOAD;                         # n
     LEA n; LOAD; IMM 1; SUB; PSH;        # sum(n-1)
     JSR sum; ADJ 8;
     ADD; LEV;                            # return n + sum(n-1);
main: IMM <N>; PSH; JSR sum; ADJ 8; EXIT
```

Hook on L13 head 7 (`layer10_carry_relay` per
[`VAR_REAL_ATTRIBUTION_2026_06_05.md`](VAR_REAL_ATTRIBUTION_2026_06_05.md))
on the JSR step:

- Step-0 MEM_addr1 (`logit_pos=118` for var_simple_0; analogous step
  for rec) shows L3 FFN baseline `+0.940` writing `OUTPUT_LO[0]` and
  L13 head 7 amplifying by `+2.000`. For `rec_*`, the JSR-step MEM
  block carries the **callee's BP frame seed**, not a memory store
  address, so the L13 head-7 amplification mis-routes the
  `MEM_STORE/MEM_ADDR_SRC` residual.
- Step-0 MEM_value1 (per
  [`VAR_REAL_ATTRIBUTION_2026_06_05.md`](VAR_REAL_ATTRIBUTION_2026_06_05.md)
  + the refinement in `5317c1d6` exonerating L14
  `jsr_mem_default_suppress`) is being written by the **same** L10
  carry-relay path, not by L14 `mem_generation`. This is the
  cross-cluster surface shared with `var_simple_*`, `func_identity_*`,
  `if_var_*`.
- L6 head 7 LEV refresh removal in `5f5efb55` (`fix(l6): disable head
  7 post-LEV AX_CARRY refresh`) restored `7/25 func_add_*` passes;
  the same path was responsible for the `rec_factorial → neural = n`
  symptom in samples taken before `5f5efb55`. Post-fix the symptom
  *should* shift to `neural = 0` (base-case unwind reaches AX) — but
  the recursion sample in `1096_LOOP_RECURSION_SAMPLE_2026_06_07.md`
  was taken at `1b7869bf` (before `5f5efb55`). A fresh probe at
  `6d7a1c1d` is **not yet available** because of the pytest wall-time
  on this host.

## Step 4 — Root surface determination

| Family | Collapsed-step? | Memory cluster? | Independent? | Real surface |
|---|---:|---:|---:|---|
| Loops | No (loops emit step-by-step; no `PC // INSTR_WIDTH` skip on `IMM,PSH,IMM,<binop>,EXIT` shape) | No (BZ/BNZ does not consult `_BINARY_POP_OPS`; the runner override in `f3342968` no longer routes BZ/BNZ since `44709e13`) | **Yes — branch-target emission cascade** | L6 `post_l9_bz_bnz_pc_override` (`l6_ops.py:4643`) re-firing on subsequent BZ instances inside a loop, plus the legacy `_set_layer4_ffn` bake's `_append_pc_byte0_imm_to_byte_addr_rules` only covering the first branch step. The `step0_guard_weight=+10` term gates step 0 but does *not* gate steps 2+ where FETCH_LO carries a different opcode's target. |
| Recursion | No (JSR ≠ binary-pop) | **Partial** — shares `var_simple_*` MEM_addr1 surface | No | Same L13 head 7 `layer10_carry_relay` amplifier as `var_simple_*` / `if_var_*` / `func_identity_*`. The L6 head 7 LEV refresh (now removed in `5f5efb55`) was a secondary contributor; remaining surface is upstream at L10/L13. |

**Bottom line:** Loops are **independent** of the var-cluster surface
— they need a fresh fix on the BZ/BNZ branch-target re-fire path.
Recursion is **downstream** of the var-cluster (`MEM_addr1`/
`MEM_value1` cascade) and should track its closure as Wave 1 A1 lands.

## Step 5 — Repro (read-only)

```
HEAD=6d7a1c1d
cd c4_release
C4_1096_DIAG=1 C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K=0 \
C4_BATCH_USE_KV_CACHE=0 C4_1096_OFFSET=525 C4_1096_LIMIT=1 \
python -m pytest -q tests/test_1096_neural_declarative_diagnostic.py::\
test_1096_neural_declarative_diagnostic_slice -s
# warm-cache wall ~ 60 s; expect loop_pow2_0 neural=0xFF0A.
```

For recursion:
```
C4_1096_OFFSET=750 C4_1096_LIMIT=1 ...
# warm-cache wall ~ 22 s; expect rec_sum_0 neural=N or 0
```

## Step 6 — Smoke

Smoke target ≥ 45/51. Per current HEAD `6d7a1c1d`, recent smoke runs
(`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`) report 45/51 as the
post-L5 baseline. This attribution doc adds no source edits, so smoke
is unchanged.

## Cross-references

- [`1096_LOOP_RECURSION_SAMPLE_2026_06_07.md`](1096_LOOP_RECURSION_SAMPLE_2026_06_07.md) — sample numerics this doc builds on.
- [`BZ_BNZ_VERIFY_2026_06_07.md`](BZ_BNZ_VERIFY_2026_06_07.md) — confirms the 21-test BZ/BNZ + control-flow cluster is unfixed by `763ff75c`.
- [`VAR_REAL_ATTRIBUTION_2026_06_05.md`](VAR_REAL_ATTRIBUTION_2026_06_05.md) — block-by-block writer pin for the L13 head 7 `+2.0` amplifier shared with recursion.
- [`FUNC_ADD_JSR_RETURN_FORWARDING_2026_06_07.md`](FUNC_ADD_JSR_RETURN_FORWARDING_2026_06_07.md) — L6 head 7 LEV refresh removal (`5f5efb55`); partial fix for `rec_*` "neural = innermost input" shape.
- [`MEMORY_L13_L14_HANDOFF_2026_06_07.md`](MEMORY_L13_L14_HANDOFF_2026_06_07.md) — hook-pattern template used for the L13/L14 probes above.
- [`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`](COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md) — confirms BZ/BNZ no longer routes through the `_BINARY_POP_OPS` runner override (since `44709e13`).
