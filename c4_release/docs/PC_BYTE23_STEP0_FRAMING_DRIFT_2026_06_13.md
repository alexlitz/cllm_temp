# Step-0 PC byte-2/3 framing drift — AX byte-1 dump over-fire (LANDED) — 2026-06-13

Status: **LANDED, smoke 51/0, non-regressing.** The dominant step-0 root behind
the 1096 "PC-wrong" full-trace fails (var/func/nested clusters) is fixed by a
single new declarative kill-flag rule on the AX byte-1 dump precursor.

## The bug (probe-confirmed, spec_k=0, `tools/probe_groundtruth.py`)

On the FIRST VM step (step 0, the JSR/entry step) of the var/func/nested
clusters, the model emitted GARBAGE into REG_PC bytes 2 and 3 while bytes 0/1
were CORRECT:

* var_simple_0 (`x=990`): oracle PC=26 (`0x0000001A`), neural PC=`0x1005001A` —
  byte0=0x1A ✓, byte1=0x00 ✓, **byte2=0x05 ✗, byte3=0x10 ✗** (token 5 at PC[2]).
* func_add_0 / nested_quad_0 / nested_sumsq_0: identical shape (byte 2/3 =
  0x05/0x10, a CONSTANT garbage independent of the program).

The fixed-35-token slicer then reads a garbage PC -> step desync -> "PC-wrong"
full-trace fail at step 0. Affected clusters (entry PC < 256, so bytes 0/1 were
correct): **var (250-324), func (550-699), nested (950-999)** — the largest
single step-0 cohort.

### Root (LM-head logit attribution at the PC[2]/PC[3] predictor row)

The corruption was a **NEGATIVE -3403 write into the `H1_DUMP_OUT` band**
(widened dims 879-885), which the LM head reads at `+5.0` for byte-token 0
(`head.weight[token_v, H1_DUMP_OUT+(v+2)] = 5.0`). A -3403 there drove the
byte-0 token's logit to ~-1.7e7 -> a garbage token won the PC[2]/PC[3] argmax.

The -3403 was the **AX byte-1 register-dump (`ax_byte1_dump_repopulate`,
l11_ops.py) OVER-FIRING on the REG_PC byte 2/3 predictor rows.** The dump's
lower-bound gate is `+1.0 * Σ AX_CARRY` (UNBOUNDED). On step 0 of these
programs the AX_CARRY band carries large step-0 garbage:

| row (step 0)            | Σ AX_CARRY | dump gate (thr 4.5) |
|-------------------------|-----------:|---------------------|
| PC byte1 predictor      | +112       | FIRE (over-fire)    |
| PC byte2 predictor      | +68        | FIRE (over-fire)    |
| PC byte3 predictor      | +42        | FIRE (over-fire)    |
| (legit) AX byte1 carry  | +2.65      | FIRE (correct)      |

The huge Σ AX_CARRY swamped the +4.5 threshold and fired the dump on PC rows.
The carried `H1_PREV_STEP` there is negative garbage (~-26.6), so the dump wrote
~-3403 into `H1_DUMP_OUT`. (Verified per-block: block 41 — the
`ax_byte1_dump_repopulate` FFN in the L25 tail — is the sole writer; dark on the
passing `expr_paren` whose Σ AX_CARRY ~ +1.12 at the same rows.)

## The fix (one new declarative rule — the AX-register HARD prerequisite)

The clean discriminator is **`ADDR_B1_HI+8`** (the AX-register signal,
registry dim 222+8): PROGRAM-STABLE ~4.02/3.29/2.70 on AX byte 1/2/3 rows
(both passing and over-firing programs) and EXACTLY ~0.00 on PC/SP/BP/STACK0/MEM
byte rows. The dump must NEVER fire on a non-AX row regardless of Σ AX_CARRY.

`_ax_byte1_carry_overflow_flag_rules` (l11_ops.py) gains a SECOND unit that
writes the existing dump-kill band `AX_CARRY_OVERFLOW` whenever the AX-register
signal is ABSENT:

```python
# Unit 1: step(ADDR_B1_HI+8 <= 2.0)  -> conditions=((ADDR_B1_HI+8, -1.0),),
#         threshold=-2.0  (fires when -ADDR_B1_HI+8 >= -2.0 i.e. <= 2.0)
multi_way_and_rule(conditions=(("ADDR_B1_HI+8", -1.0),), threshold=-2.0,
                   writes=(("AX_CARRY_OVERFLOW", 2.0/100.0),))
```

The dump already reads `AX_CARRY_OVERFLOW` at `-1000` (the SHL/JMP upper-cut), so
the new unit reuses that exact kill path. Threshold 2.0 sits cleanly in the gap
(lowest legit AX row = byte3 @ 2.70; non-AX rows @ ~0.00). This turns the
AX-register signal from a `+0.25`-weighted gate term (the unbounded Σ AX_CARRY
could overwhelm) into a HARD prerequisite.

`_AX_CARRY_OVERFLOW_FLAG_HIDDEN_DIM` bumped 1 -> 2; the op's `reads` gains
`ADDR_B1_HI` (registry-resolved, like the other gate dims). No new band, no
geometry change, no new head.

### Verified (spec_k=0, GPU 0, `alu_mode='efficient'`)

* var/func PC byte rows step 0: `AX_CARRY_OVERFLOW=+4.0` -> dump killed ->
  `H1_DUMP_OUT == 0` -> PC bytes 2/3 emit the correct 0x00.
* legit AX byte-1 carry (add_0 `654+114`): `AX_CARRY_OVERFLOW=0` -> dump still
  fires (`Σ|H1_DUMP_OUT| = 36.3`) -> high byte 0x03 still emits. **Byte-identical.**
* Smoke **51/0** (all 16-bit ADD/SUB/OR/AND/XOR/SHL/SHR + memory pass; the dump
  is preserved on every legit AX byte-1 row). NOTE: full-smoke spuriously fails
  ~half its tests under GPU contention (parallel agents) — verify in isolation
  or on an empty GPU; each passes solo.
* add/sub/if_gt/expr_paren slice (ids 0-9,50-59,350-359,825-834): **25/40 both
  before and after** -> NON-REGRESSING (byte-identical pass set).

## The step-0 fix EXPOSES each deep cluster's NEXT (distinct) root

The fix removes the step-0 PC framing drift but each deep cluster has a SECOND,
DISTINCT root immediately behind it (probe-confirmed):

* **func / nested**: now diverge at **step 1, AX=8** — the documented
  `simple_function` ENT->IMM=8 arch-block (any ENT followed by an
  IMM/op corrupts the operand to the ENT SP-decrement constant 8; needs the
  reserved `ACTUAL_OP_ENT` opcode-decode surface — see
  `PHASE_5_JSR_ENT_LEV_FOLLOWUP.md`, multi-session wall).
* **var**: now diverges at **step 2 (the LEA frame-local step)** — emits a
  37-token step (2 spurious tokens, token 255) instead of 35 -> desync. A
  distinct LEA/frame-local emission framing drift.
* **rec (factorial/fib/sum/power)**: a DIFFERENT step-0 root — entry PC >= 256
  (e.g. rec_fib oracle PC=258=`0x0102`), so PC byte 0 AND byte 1 are wrong
  (neural `0x0202021A`). NOT the H1_DUMP over-fire (which left byte 0/1
  correct); a separate entry-PC-high-byte decode issue for PC >= 256.
* **gcd / loop / rec_sum**: > 40-60 steps, skipped by the canonical step cap;
  gated by the above prologue roots AND deep-loop divergence.

So this fix is a NECESSARY, load-bearing prerequisite (without it none of these
can pass) and the largest single step-0 root, but it does not by itself flip a
deep cluster to passing — each needs its 2nd root cleared too.

## Tools
- `tools/probe_groundtruth.py` (spec_k=0) — `residual_at` per-block trace pinned
  block 41 as the -3403 writer; `probe(...)` top-k showed the flat PC[2] logits.
- `tools/run_1096_canonical.py --criterion strict_trace --fail-fast` — the exact
  diverging byte offset (step 0 offset 3 = PC[2], expected_tok=0 got_tok=5).
