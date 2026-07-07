# PROJECT_0XE8_SLAM — the frame-address (0xE8/0xE0) slam into OUTPUT on VALUE/COMPARISON rows

Phase 1: **SCOPE + DESIGN ONLY.** Byte-identical to golden
`b1dcae630381bbe93ece7a53efbeadf4a6fefef0227db8fa17fba81d76ad53f5`
throughout — no weight changes. This doc + `tools/probe_e8_slam_rowsig.py`
(+ `tools/_probe_e8_slam_loadbearing.py`) are the foundation for the Phase-2 fix.

Status: **COMPLETE (scope + design).** Phase 2 builds the fix.

---

## 0. TL;DR — root cause, corrected

The step-0 AX `0xE8`/`0xE0` slam on `if_gt` / `if_eq` / `func_max` / `func_min`
(id360 `8>27` wants AX byte-0 `0x08`, gets `0xe8`; id402 `16==9` wants `0x10`,
gets `0xe0`) is **NOT** owned by
`layer16_lev_routing::l16_psh_mem_addr0_restore_lo_8` / `_restore_hi_14` (the
rules the brief named). That attribution is a **CIRCULAR artifact** of the
gate's attribution reading the FINAL pre-head residual (where the leaked
`OUTPUT_LO/HI` cell is already `±5.8e6` — a value those restore rules THEMSELVES
did not write, but which they then appear to "fire on" when their own condition
term re-reads it).

Reading the **causal block-INPUT residual** (via
`model.forward(..., stop_after_block=N-1)`) instead of the final residual shows:

* At the L16 lev-routing block-34 INPUT, on the `if_gt` AX row, the restore rule
  `up = -9.998e7 < 0` — it **does NOT fire**. The `-1e6 MARK_AX` blocker holds
  (`OUTPUT_LO+8` there is a clean `+159.7`, so `S·(+1)·159.7 = +1.6e4` cannot
  cross the `-1.0e8` blocker). Byte at block-34 input is still the correct
  `0x08`.
* A **block-by-block sweep** localizes the flip: `after blk46: byte=0x08`,
  `after blk47: byte=0xe8` (`OUTPUT_LO+8` jumps `+547 → +5.83e6`,
  `OUTPUT_HI+14` jumps `−0.26 → +5.83e6`). The corruption happens at **physical
  block 47** (an `l10_post_ops_combined` / L25-tail post-op), NOT block 34.

**The TRUE owning rules** (causal block-47/48-INPUT attribution) are the L10 tail
`LEA`-restore family:

| leak | program | true owning op | rule that fires |
|------|---------|----------------|-----------------|
| `0x08 → 0xe8` | if_gt id360 (step0 AX) | **`l10_loop_lea_b0_e8`** (blk47) | `l10_loop_lea_b0_e8_output_{lo,hi}_*` |
| `0x10 → 0xe0` | if_eq id402 (step0 AX) | **`l10_loop_lea_b0_e0`** (blk48) | `l10_loop_lea_b0_e0_output_{lo,hi}_*` |

Both live in `neural_vm/unified_compiler/ops/l10_ops.py`
(`_l10_loop_lea_b0_e8_rules` @ 12048, `_l10_loop_lea_b0_e0_rules` @ 12289),
flag-gated `C4_LOOP_LEA_B0_E8` / `C4_LOOP_LEA_B0_E0` (both **DEFAULT-ON in the
campaign config** = the golden b1dcae63 config).

**Why they FALSE-FIRE (the calibration bug):** the discriminator scores an
`imm=−8` FETCH signature at a big weight —
`FETCH_LO+8·1000 − FETCH_LO+0·1000 − FETCH_HI+14·1000` (e8), or the inverse
(e0) — and was calibrated against a **soft** FETCH one-hot (`~0.4–1.0`, verified
on genuine in-loop `LEA &i` rows). On an `if_gt` / `if_eq` step-0 IMM AX row the
FETCH band carries **`+40`** at the immediate's low-nibble cell (a 40–70×
amplified broadcast, not a `0/1` one-hot). So `FETCH_LO·1000 = 100·1000·40 =
+4.0e6` (S=100) **overwhelms every calibrated margin**: the `OP_LEA` hard-req
(`OP_LEA·200 + CONST·−650`, meant to veto rows with `OP_LEA==0`) contributes
only `−650` and the `OP_IMM·−500` opcode NOT-block only `−2.5e5` — both dwarfed
by `+4.0e6`. Net `up = +3.645e6 > 0` (e8) / `+3.590e6 > 0` (e0) → the AND
spuriously fires and the per-cell winner-take-all (`±DOM`) drives OUTPUT to
`0xE8` / `0xE0`.

This is the **same CLASS** of bug the brief described (a fixed-weight
requirement overwhelmed by an amplitude-unbounded condition dim), but the
active owning rules + the unbounded dim (`FETCH_LO`, not an `OUTPUT` self-read)
are different from the brief's hypothesis. The named `layer16_lev_routing`
restore rules and the L15 head-16 SI-store CAM are **not** the active root
(L15 head-16 = `layer15_si_store_addr_cam` is behind `C4_SI_STORE_ADDR`, which
is **DEFAULT-OFF**, so it is not even built in golden b1dcae63).

---

## 1. The owning rules (as authored)

`_l10_loop_lea_b0_e8_rules()` (l10_ops.py:12048) builds **32 winner-take-all
units** (16 `OUTPUT_LO` → nibble 8, 16 `OUTPUT_HI_THIS_STEP` → nibble E) sharing
one discriminator `disc`; `_l10_loop_lea_b0_e0_rules()` (12289) is the inverse
(nibble 0 / nibble E) for the 2nd-local `LEA &sum` (0xE0). Shared discriminator
(e8 variant, l10_ops.py:12079):

```
disc = (
    ("CONST", -650.0),          # OPLEA_REQ bias
    ("MARK_AX", 100.0),
    ("OP_LEA", 200.0),          # HARD-req (OPLEA_REQ ON); veto OP_LEA==0 rows
    ("FETCH_LO+8",  1000.0),    # imm=-8 signature  <-- amplitude-UNBOUNDED
    ("FETCH_LO+0", -1000.0),
    ("FETCH_HI+14",-1000.0),
    ("MEM_ADDR_SRC", -1_000_000.0),   # off genuine addr-eval LEAs
    ("IS_BYTE",      -1_000_000.0),   # off PC/AX value bytes 1-3
    *((op, -500.0) for op in owning_ops),   # OP_IMM/ADD/.../GT/EQ NOT-blocks
    ("MARK_PC", -1e6), ("MARK_SP", -1e6), ("MARK_BP", -1e6),
    ("MARK_STACK0", -1e6), ("MARK_MEM", -1e6),
) + _tail_lea_e8_ent_byte0_blockers()   # per-step ENT one-hot guard
threshold = 500.0    # (e0: 1700.0)
```

Intended firing set: exactly the loop-family in-loop `LEA &i` (1st local, want
`0xE8`) / `LEA &sum` (2nd local, want `0xE0`) AX-marker row where the block-42
keystone can't fire (`CMP+7 == 0`, `MEM_ADDR_SRC == 0`) and `l10_ent_axcarry`
would otherwise slam byte-0 to `0x00`.

---

## 2. Row-signature separation — GENUINE vs LEAK (measured, causal block-INPUT)

Probe: `tools/probe_e8_slam_rowsig.py --blk-input --decompose ...` (block-47/48
INPUT, spec_k=0, campaign, BUILT dims). The single clean discriminator is
**`OP_LEA`** (genuine ≈ +5.23, leak = 0) — but on the leak row `FETCH_LO` is
40–70× larger than the discriminator assumed, so the additive `OP_LEA` term is
overwhelmed:

| dim | GENUINE in-loop `LEA &i` (loop_sum_0 step2/10) | LEAK if_gt id360 step0 AX | LEAK if_eq id402 step0 AX |
|-----|-----|-----|-----|
| `MARK_AX`       | +1.0 | +1.0 | +1.0 |
| `OP_LEA`        | **+5.23** | **0.0** | **0.0** |
| `OP_IMM`        | 0 | +5.0 | +5.0 |
| `FETCH_LO+8`    | **+0.58 … +1.00** | **+40.0** | 0.0 |
| `FETCH_LO+0`    | +0.42 … 0.0 | 0.0 | **+40.0** |
| `MEM_ADDR_SRC`  | 0.0 | 0.0 | 0.0 |
| `IS_BYTE`       | 0.0 | 0.0 | 0.0 |
| rule `up` (e8)  | **+1.6e4 … +1.0e5** (FIRES, correct) | **+3.645e6** (FIRES, WRONG) | −4.355e6 (silent) |
| rule `up` (e0)  | — | −4.355e6 (silent) | **+3.590e6** (FIRES, WRONG) |

Decompose of the false-fire (`l10_loop_lea_b0_e8_output_lo_8`, if_gt id360,
block-47 input), per-term `S·w·val`:

```
FETCH_LO(+8)  w=+1000  val=+40.0  S*w*val = +4.000e6   <-- dominates
OP_IMM        w= -500  val= +5.0  S*w*val = -2.500e5
CONST         w= -650  val= +1.0  S*w*val = -6.500e4
(-S*thr)      w= -500  val= +1.0  S*w*val = -5.000e4
MARK_AX       w= +100  val= +1.0  S*w*val = +1.000e4
                                  up      = +3.645e6  > 0  => FIRES (wrong)
```

Root: the discriminator's positive `imm=−8`/`imm=−16` FETCH term is
amplitude-unbounded; the calibration assumed FETCH ∈ `[0.4, 1.0]` (true on the
genuine LEA row) but on the IMM/comparison AX row it is `+40`.

---

## 3. Load-bearing memory function — WHICH tests depend on it, and what breaks OFF

Kill-switch experiment: `C4_LOOP_LEA_B0_E8=0 C4_LOOP_LEA_B0_E0=0` (both are
existing opt-outs; OFF → the ops register zero rules + append no post-op, model
strictly smaller — used here ONLY to MEASURE; default stays ON / byte-identical).

**Emitted-token argmax (interp faithful forward) is byte-identical ON vs OFF for
EVERY smoke test** (`tools/_probe_e8_slam_loadbearing.py` /
`probe_e8_slam_rowsig` hash compare):

```
si_li_roundtrip, sc_lc_roundtrip, si_li_zero, si_li_multiple_stores,
si_li_overwrite, si_li_16bit_value, lea_basic, adj_sp, simple_function,
eq_true, eq_false, lt_true, ne_true, gt_true, le_true, ge_true   -> all SAME
```

Why the ops are NOT load-bearing for any smoke test even though they FIRE on
several of them: the `l10_loop_lea_b0_e0` op fires on step-0 (off=5, the AX
marker) of **every** `si_li_*` / `lea_basic` program (they all start `IMM 0x200`
/ `IMM 0x300`, low-nibble 0 → `FETCH_LO+0` high → same false-fire path as
`if_eq`). BUT on those rows the model **already** emits byte-0 = `0xE0` (the
model represents the pushed frame/heap address's byte-0 as `0xE0` regardless of
the op — verified: si_li step-0 emits `0xE0`/224 for oracle-AX 512 both ON and
OFF, and the test still passes on the eventual value 42). So the op **re-stamps
a byte that is already there** on genuine frame-address rows → a redundant no-op
on the smoke set. On the comparison rows the true byte is the operand
(`0x08`/`0x10`), so the same stamp CORRUPTS.

**What the ops ARE for:** the deep loop clusters (`loop_sum` / `loop_mul` /
`loop_pow2` / `loop_fact`, corpus ids ~450+) — the in-loop `LEA &i` AX-marker
byte-0 (want `0xE8`/`0xE0`), which is DEAD through the block-42 keystone
(`CMP+7 == 0`, `MEM_ADDR_SRC == 0`) and then slammed to `0x00` by
`l10_ent_axcarry`. These ops re-stamp it. Genuine firing confirmed on
loop_sum_0 (`OP_LEA +5.23`, `FETCH_LO+8 +0.58…+1.0`, `up +1.6e4…+1.0e5`) at
steps 2/10/21/25/27. These loop clusters are the deep >40-step diverging
corpus (canonical full_trace "pass none"), so the ops' current NET verdict
contribution is ~0 while they actively CORRUPT the structured
`if_gt`/`if_eq`/`func_max`/`func_min` clusters.

**Blast radius:** only rows where the in-step operand's low nibble is exactly
**8** (→ e8 false-fire, `FETCH_LO+8` high) or **0** (→ e0 false-fire,
`FETCH_LO+0` high) AND `OP_LEA == 0`. Comparison/if/func programs with such an
operand flat-diverge at step 0/1 (id360 `8>…`, id402 `16==…`, `func_max`/
`func_min` at the arg-load step). Verified via gate ON vs OFF: id360/id402
`step=0 AX[0]` FAIL disappears OFF (→ CROSS-STEP, no flat divergence). Memory
smoke stays green (byte-identical). **No memory-smoke test regresses OFF.**

---

## 4. writer_index competition at the leaked OUTPUT dim

`OUTPUT_LO+8` / `OUTPUT_HI_THIS_STEP+14` writer index, runtime firing on the
if_gt id360 AX row (block-47 INPUT):

* `OUTPUT_LO+8` (col 77): **1732** static writers, **304** fire (up>0). Top
  firing writer = **`l10_loop_lea_b0_e8`** at `contrib=+3.645e5` per unit
  (16 lo units all fire → sum drives the cell to `+5.83e6`).
* `OUTPUT_HI+14` (col 99): **1914** static writers, **297** fire. Top firing
  writer = `l10_loop_lea_b0_e8` (16 hi units, same `up`).

Distinct static-writer ops of these cells include `layer16_lev_routing`,
`layer10_alu`, `tail_bit32_result_correction`, `l10_ent_axcarry`,
`l10_exit_axcarry`, `l10_loop_lea_b0_e8`, `l10_loop_lea_b0_e0`, `clean_emitter`,
`b1_to_output`, …. On a comparison result AX row what **SHOULD** dominate is the
`clean_emitter` / ALU-comparison decode band delivering the operand byte
(`0x08`); it is out-voted by the loop_lea op's `±DOM` winner-take-all (the op
runs LAST, in the L25 tail, by design). A corrected firing gate (§5) that keeps
the loop_lea op OFF on non-LEA rows collides with **nothing** — every other
firing writer at these cells is a static-default or the intended
comparison-decode writer, so removing the false loop_lea fire hands the cell
back to the correct decoder.

---

## 5. Correct-by-construction Phase-2 fix design

The clean, always-present discriminator that separates GENUINE (want the stamp)
from LEAK (must not stamp) is **`OP_LEA`**: +5.23 on the genuine in-loop LEA,
`0` on every IMM / comparison / func-arg AX row. It is already in the
discriminator — but as an **additive** term it can be overwhelmed by the
amplitude-unbounded `FETCH_LO`. The fix makes `OP_LEA` a **multiplicative
requirement** that no FETCH amplitude can cross.

### 5a. The spec edit (both ops, `l10_ops.py`)

For every unit in `_l10_loop_lea_b0_e8_rules` (12048) and
`_l10_loop_lea_b0_e0_rules` (12289), add a **multiplicative gate on `OP_LEA`**
to the `multi_way_and_rule` call. The FFN math is `hidden = silu(up)·gate` with
`gate = gate_bias + Σ gate_terms·x`; choose the gate so it is `≈1` when
`OP_LEA ≈ 5.23` (genuine) and `≤0` when `OP_LEA == 0` (leak):

```python
rules.append(multi_way_and_rule(
    name=f"l10_loop_lea_b0_e8_{out_dim.lower()}_{k}",
    conditions=disc,
    threshold=_LOOP_LEA_E8_THRESHOLD,
    writes=writes,
    # NEW: OP_LEA multiplicative requirement. gate = -G + w*OP_LEA.
    #   genuine OP_LEA 5.23 -> gate = -G + w*5.23 > 0 (=> ~1 after clamp/scale)
    #   leak    OP_LEA 0.00 -> gate = -G           < 0 => hidden = silu(up)*neg
    # Pick e.g. gate_bias=-1.0, gate_terms=(("OP_LEA", 1.0),):
    #   genuine 5.23 -> +4.23 ; leak 0 -> -1.0 (kills the stamp regardless of up).
    gate_bias=-1.0,
    gate_terms=(("OP_LEA", 1.0),),
))
```

Because `gate` is a plain linear read of `OP_LEA` (a per-step opcode one-hot,
bounded 0/≈5.23), a `+40` FETCH can no longer manufacture a fire: the gate zeroes
the whole unit when `OP_LEA == 0`. On the genuine LEA row `gate > 0` and the unit
fires exactly as today (byte-identical loop behaviour). The write direction
(`±DOM`) is unchanged, so on a firing genuine row the emitted `0xE8`/`0xE0` is
unchanged.

**Discriminator dim:** `OP_LEA` (col resolved via `DimResolver` over the BUILT
`layout.dim_positions` — the op already uses BUILT-dim names).

**Alternative (if a bounded gate needs tuning):** keep `OP_LEA` additive but
**cap** the FETCH contribution by replacing the raw `("FETCH_LO+8", 1000.0)`
with a `band_range_check_rules` / `one_hot_indicator_rule` one-hot on the
*argmax* of FETCH (a `0/1` presence test) instead of the amplitude-linear read.
This removes the unbounded amplitude entirely. The multiplicative `OP_LEA` gate
(5a) is the smaller, lower-risk edit and is preferred; the FETCH one-hot is the
belt-and-suspenders follow-up if any residual amplitude path is found.

### 5b. Guardrail tests that MUST stay green (byte-identity + verdict)

1. **Golden byte-identity** (`tools/_isa_golden_hash.py`): flag-OFF golden stays
   `b1dcae63`. The fix adds a gate to EXISTING campaign-ON units (no new
   flag-OFF rules), so the flag-OFF 35-tok build is untouched.
2. **`compare_symbolic_to_lowered_ffn`** on both edited ops (isolated
   byte-identity of the lowered gate weights vs the symbolic rule).
3. **`tools/lint_cross_op_ffn.py --flag C4_LOOP_LEA_B0_E8`** — the ops write the
   shared `OUTPUT_LO/HI` band in the L25 tail; assert other L25-tail OUTPUT
   readers are unchanged at the genuine-LEA and non-LEA probe rows.
4. **`tools/flag_regression_gate.py`** — the four MEM-SMOKE `var_*` clusters +
   add/sub/mul/div/mod samples must stay `ok` (no `ok->fail`). The memory-smoke
   emission is byte-identical ON/OFF today (§3), so the gate must remain green.
5. **Loop clusters not regressed:** genuine loop_sum in-loop `LEA &i` byte-0
   still emits `0xE8` (probe: `OP_LEA +5.23` → gate>0 → unit fires; verify
   loop_sum_0 steps 2/10/21 still stamp `0xE8`). Because the loop rows all carry
   `OP_LEA ≈ 5.23`, the gate is a no-op there.
6. **Comparison / func clusters gain:** id360 (if_gt) + id402 (if_eq) step-0 AX
   byte-0 FAIL removed (gate ON→OFF for those rows only, `OP_LEA == 0`); confirm
   with `tools/cpu_full_trace.py --ids 360,402` (the authoritative AR verdict)
   and a GPU `--faithfulness-check` on the if_gt/if_eq/func_max/func_min sample.

### 5c. What NOT to do

* Do **not** patch `layer16_lev_routing::l16_psh_mem_addr0_restore_*` — those do
  not fire causally here (§0); a change there is a no-op on this leak and risks
  the load-bearing genuine PSH-frame restore.
* Do **not** touch L15 head-16 (`layer15_si_store_addr_cam`) — DEFAULT-OFF, not
  built in golden.
* Do **not** naively delete/veto the loop_lea ops wholesale — they are the only
  writer that keeps the in-loop `LEA &i` byte-0 = `0xE8`/`0xE0` for the loop
  clusters (§3). Gate them, don't kill them.

---

## Appendix — probes (all byte-identical, tooling-only)

* `tools/probe_e8_slam_rowsig.py` — row-signature + causal block-INPUT
  decompose (`--blk-input`), rule runtime-firing dump, `--scan-all`,
  `--decompose <rule>`. **Use `--blk-input`** to avoid the circular
  final-residual attribution.
* `tools/_probe_e8_slam_loadbearing.py` — emitted-result across smoke groups for
  ON-vs-OFF diff.
* Block-by-block flip localization + writer-index snippets: inline in this doc's
  §0/§2/§4 (reproducible one-liners over `interp_oracle_gate.build_gate_context`
  + `model.forward(stop_after_block=N)`).
