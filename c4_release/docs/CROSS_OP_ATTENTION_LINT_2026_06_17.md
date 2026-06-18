# Cross-op attention-interaction lint — closing the non-local-regression blind spot

`tools/lint_cross_op_attention.py` is the **authoring-time** counterpart to the
GPU tripwire (`tools/gpu_tripwire.py`). The tripwire catches a non-local
attention regression *post-build* in ~5 min of GPU time; this lint catches the
same class of bug *pre-build*, on CPU, in the time of two model compiles.

## The blind spot (the byte-0 case study)

The byte-0 fix (`C4_OPERAND_GATHER_PSH_ROWSELECT`, commit `4e79f176`,
cherry-picked flag-OFF as `6874750f`) added an attention CAM Q/K slot
(**slot 34**) to the **SHARED** L7 operand-gather **head 0** (physical block
11), gated `Q[34] = c·(OP_ADD + OP_SUB)`. It passed every gate available at the
time:

* per-op lowering byte-identity (`compare_symbolic_to_lowered_attn`),
* smoke 51/0,
* targeted add/sub +8.

But the FULL run showed **−39**: it BROKE `var_simple` (22) + `expr_mod` (25).

**Root cause.** Attention softmax is **GLOBAL over the head**. Adding a K slot
changes the post-softmax attention OUTPUT for OTHER queries served by the same
head, because the new K row participates in the **softmax1 normalization** of
every query and its non-zero K projection re-ranks / re-weights the operand
candidates. `var_simple` / `expr_mod` gather their operands through the SAME
head at the SAME `OP_ADD` / `OP_SUB` opcodes — but in DIFFERENT operand-row
**contexts** (no genuine PSH-output row, a corrupted high nibble, or a different
number of competing `STACK0_BYTE0` rows). The slot-34 bias shifts the row
selection in exactly those contexts.

`compare_symbolic_to_lowered_attn` checks ONE op in isolation with a single
high-amplitude query and **hardmax** — it CANNOT see a multi-row
softmax-normalization interaction. That is the blind spot.

## What the lint asserts

> Given a candidate op's flag, the post-softmax attention OUTPUT of any
> SHARED / pre-existing attention head it modifies must be UNCHANGED (within
> tolerance) at a representative battery of OTHER-OP / OTHER-CONTEXT probe rows,
> OFF vs ON.

Concretely:

1. **Build flag-OFF and flag-ON** via `compile_full_vm_dynamic(disk_cache=False)`
   (CPU, both the in-process memo and the disk cache bypassed so the live env
   flag is honoured). Each build is **env-isolated** (`_flag_env`) so sibling
   lints don't contaminate each other.
2. **Auto-detect modified shared heads.** For every `(physical_block, head)`
   whose Q/K/V/O weights differ OFF→ON, classify it: `pre_existing` (the OFF
   head already had Q or K rows — a SHARED head being extended, the byte-0
   hazard) vs brand-new (empty OFF — a new head can't perturb an existing op's
   softmax, so it's out of scope; the GPU tripwire still covers it). Blocks
   whose `W_q.shape` differs (a residual-band / width fix) are skipped — they
   don't extend a shared head in a comparable layout.
3. **Probe battery = opcode × operand-context.** For each modified shared head
   run a battery of multi-row residual probes. A probe is a short sequence of
   competing `STACK0_BYTE0` operand-candidate rows ending in a `MARK_AX` query
   tagged with the probe opcode. **Multiple competing rows are load-bearing** —
   a single key never exposes a softmax-normalization shift, only a re-ranking
   among >1 candidates does. The contexts span the var/expr operand frames:

   | context              | row-0 PSH | row-0 hi-nibble | competing rows |
   |----------------------|-----------|-----------------|----------------|
   | `targeted_clean_psh` | yes       | clean           | 2              |
   | `var_expr_no_psh`    | no        | clean           | 2              |
   | `corrupted_hi_nib`   | yes       | corrupted (0)   | 2              |
   | `single_operand_row` | yes       | clean           | 1              |
   | `three_competing`    | yes       | clean           | 3              |

4. **Real production math.** The head output is computed with the exact
   `vm_step.AutoregressiveAttention.forward` numerics — ALiBi bias (absolute
   positions) + causal mask + **softmax1** (anchor=0 sink) + per-head scale.
   Both the per-query softmax **weight distribution** and the W_o-projected
   **residual output** (into the head's ALU_LO/HI dims) are diffed; the weight
   test catches a re-weighting even when V relays the same value (the
   corrupted-hi case sharpens 0.525 → 1.0 with no argmax flip).
5. **Flag any UN-whitelisted (opcode, context) row that changes** beyond
   tolerance. The whitelist (`--expect`) names the rows the fix is authored for
   — for byte-0 that is exactly `OP_ADD:targeted_clean_psh` and
   `OP_SUB:targeted_clean_psh`. Everything else changing is a non-local effect.

## Demonstration — it discriminates real regressions from clean fixes

Run `python tools/lint_cross_op_attention.py --demo`:

* **byte-0 (`C4_OPERAND_GATHER_PSH_ROWSELECT`) → FLAGGED.** One shared head
  detected (block 11 head 0, `dQ=15 dK=498`). With the targeted clean-PSH
  context whitelisted, the lint still flags:

  | opcode / context           | max\|Δout\| | Σ\|Δweight\| | verdict             |
  |----------------------------|-------------|--------------|---------------------|
  | OP_ADD / targeted_clean_psh| 3.146       | 1.049        | EXPECTED            |
  | OP_ADD / corrupted_hi_nib  | 2.849       | 0.950        | **NON-LOCAL CHANGE**|
  | OP_ADD / three_competing   | 4.189       | 1.396 (flip) | **NON-LOCAL CHANGE**|
  | OP_SUB / targeted_clean_psh| 3.146       | 1.049        | EXPECTED            |
  | OP_SUB / corrupted_hi_nib  | 2.849       | 0.950        | **NON-LOCAL CHANGE**|
  | OP_SUB / three_competing   | 4.189       | 1.396 (flip) | **NON-LOCAL CHANGE**|

  The `corrupted_hi_nib` and `three_competing` rows ARE the var/expr operand
  frames the fix silently perturbed — exactly the −39 regression.

* **clean fix (`C4_AX_BYTE1_FULL_WIDTH`) → PASS.** Builds both ways
  (d_model 1199 → 1417), shifts `dim_positions` (a band + LM-head-column fix),
  and modifies **0 shared attention heads**. No probe row changes.

* **clean fix (`C4_AX_BYTE1_HINIB`) → PASS.** Same family as FULL_WIDTH. Its
  OFF state is not buildable on the current tree (an L15 head lost a slot when
  the band was removed); the lint reports that and passes — a default-ON
  band/width fix is not a shared-head edit.

The discrimination contract (`byte-0 flagged AND clean fixes pass`) is the
proof the lint separates real non-local regressions from clean fixes.

## Usage

```sh
# lint a candidate fix's flag (default: the byte-0 fix)
CUDA_VISIBLE_DEVICES="" python tools/lint_cross_op_attention.py \
    --flag C4_MY_FIX --expect OP_ADD:targeted_clean_psh

# the byte-0-vs-clean discrimination demo
CUDA_VISIBLE_DEVICES="" python tools/lint_cross_op_attention.py --demo
```

Exits non-zero when an un-whitelisted `(opcode, context)` row changes.

## When to run it

**Any op that modifies a SHARED / pre-existing attention head MUST pass this
lint before commit** (see `CLAUDE.md`). A new head, or a fix that only adds
residual bands / LM-head columns, touches no shared head and passes trivially —
but run it anyway: the auto-detection is what *proves* the fix is head-local.

## Tests

`tests/test_lint_cross_op_attention.py`:

* fast unit tests (always run) — probe construction + the softmax1/ALiBi head
  math on a tiny synthetic `PureAttention`;
* the discrimination contract (`@pytest.mark.slow`, run with `--runslow`) —
  byte-0 flagged + clean band fix passed against the real VM build.
