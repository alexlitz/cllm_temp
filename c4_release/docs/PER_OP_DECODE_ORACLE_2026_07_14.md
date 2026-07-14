# Per-op decode oracle — the fast, verdict-faithful, per-op-class unit gate (2026-07-14)

**Goal.** Give op-class work (generator-collapse, corrector-delete-with-clean-
root, abstraction/rebuild) a **fast, parallel, verdict-faithful** gate that
verifies *the per-op decode did not move* **without the ~30-min serial GPU
integration run** (`tools/run_1096_canonical.py --criterion full_trace`). This is
the #1 timeline lever: decode-preserving changes iterate in minutes on CPU
instead of queueing for the GPU corpus gate.

**Tooling only — golden `1c04c3fd` untouched.** The oracle reads the SAME cached
baked model the smoke gate builds and never writes weights.

Deliverables:
* `tests/oracles/per_op_decode.py` — the op-class → representative-program map +
  the faithful decode verdict engine + pytest entry points + a standalone CLI.
* `tools/run_per_op_oracle.py` — the parallel (sharded) runner + baseline
  recorder + baseline-diff gate.
* `tools/per_op_oracle_baseline.json` — the recorded main baseline (the
  reference a decode-preserving change is diffed against).

---

## 1. Foundation: which forward is authoritative?

The oracle rests on `neural_vm/verification/faithful_interpreter.py` +
`faithful_autoregressive.py`. That file carries **three** per-token forwards; the
distinction is load-bearing:

| forward | how | faithfulness |
|---|---|---|
| `CachedFaithfulForward` / `IRBlockForward` | recovered-weight **reimplementation** (per-head spec loops + manual softmax1 + fresh SwiGLU matmul) | argmax-**close**, NOT bit-exact — at saturated-tie positions (~1e22 logits, true gap ~1e14) it collapses the top-2 logits to an exact tie and torch's first-max tie-break picks the WRONG token → the documented `expr_paren`/`expr_mul_div`/`mul` "0xF0-fill" **false-fails** |
| **`ModelExactForward`** | runs the **REAL block `nn.Module`s** — `model.embed → block(...) → head`, i.e. it **IS `model.forward` over one row** | **BIT-EXACT** to `model.forward`, incl. the saturated-tie winners the recovered path drops |

`FaithfulAutoregressiveRunner` (in `faithful_autoregressive.py`) drives the
**production** decode machinery (`BatchedPureNeuralRunner.run_batch_fail_fast`
→ the unmodified `_run_fail_fast` / `_oracle_pc_ax_steps` / per-step `(PC,AX)`
compare) and overrides **exactly one** method (`_forward_argmax_batch`) to serve
the per-token argmax from `ModelExactForward`. Because the only substitution IS
`model.forward` over one row, its per-program `full_trace` `status` is
**byte-identical to `tools/run_1096_canonical.py --criterion full_trace
--spec-k 0`** — the exact per-program pass criterion the 30-min gate uses — with
**no GPU**.

**On `spec_k` (load-bearing):** the oracle runs the faithful decode at
`spec_k=32` (matching `faithful_autoregressive_validate.py`), NOT `spec_k=0`. The
speculative path teacher-forces the UNSAFE MEM offsets from the DraftVM — those
bytes are unreadable from a flat forward argmax, so decoding them from the
model's own argmax (what `spec_k=0` would do in the faithful per-row path) would
feed wrong MEM bytes back and desync the next step. The **pass SET is identical**
for any `spec_k` because the model is the final arbiter (per
`run_1096_canonical`'s own note + `project_probe_path_spec_k_not_hooks`), so the
`spec_k=32` faithful verdict == the `spec_k=0` canonical/GPU pass set.

**Authority ranking (the rec-agent's question):**
`run_1096_canonical --spec-k 0` (GPU) == `ModelExactForward` AR decode (CPU,
`spec_k=32`) == `model.forward` — all three agree at the saturated ties
(validated in `tools/faithful_autoregressive_validate.py`, and the byte-identity
is the whole reason `ModelExactForward` exists). `fast_gate.py` is NOT a divergent
authority —
it simply *shells out to* `run_1096_canonical` on a stratified sample, so it runs
the SAME GPU path (it predicts a corpus **delta**, it is not a faithful CPU
gate). The recovered-weight `CachedFaithfulForward` is the ONLY one that
diverges (the tie collapse), and the oracle deliberately does **not** use it.

## 2. Speed envelope (measured, CPU, `OMP_NUM_THREADS=4`, d_model=1221, 62 blocks)

* Model build (cached compile): **~4 s** (one-time per process).
* One per-token `ModelExactForward.forward` over a growing tape:
  L=35 → 0.42 s, L=175 → 1.3 s, L=350 → 2.5 s, L=700 → 6.8 s (the growing-tape
  O(steps²) forward — no KV cache in the faithful path).
* Therefore **per-program cost ≈ (n_steps·35)-token growing-tape decode**:
  a **5-step** arith program (41 forwards) ≈ **~40–80 s**; a program that HALTS
  early (e.g. the 4-step BNZ branch programs) ≈ **~2 s**.

The whole 30-op-class suite serially is ~25–35 min; sharded across
`--workers N` (each worker builds once, decodes its shard) it collapses to
`≈ serial/N + build`. On this box (64 cores) `--workers 10` lands the full
recorded run in **~5–8 min wall**. That is the fast gate.

`CachedFaithfulForward` is ~20–30 % faster but is the tie-collapsing false-fail
path — NOT worth the faithfulness loss, so the oracle uses `ModelExactForward`.

## 3. Coverage: op-class → representative programs

The ISA op-classes (`symbolic_forward._OPCODE_NAMES`) and how each is covered.
The **C test corpus (`tests.test_suite_1000`) emits most ops** — for those the
oracle draws the CHEAPEST corpus members (already compiled + oracled). But the C
compiler **never emits** the bitwise/shift ops, `NE`/`LE`/`GE`, `BNZ`, or char
load/store (`LC`/`SC`) — for those the oracle carries **hand-authored bytecode**
(`IMM a; PSH; IMM b; <OP>; EXIT` for binops; explicit branch / store-load
sequences for the rest), validated against the pure-Python ISA VM
(`SymbolicDeclarativeProgramRunner`).

| op-class | source | representatives |
|---|---|---|
| ADD SUB MUL DIV MOD | corpus | the 5-step `add_/sub_/mul_/div_/mod_` members |
| EQ LT GT | corpus | `if_eq_0`/`if_lt_0`/`if_gt_0` (7-step branch-compare) |
| **NE LE GE** | **raw** | `IMM a; PSH; IMM b; <cmp>; EXIT` (compiler canonicalises these away) |
| **AND OR XOR** | **raw** | `IMM a; PSH; IMM b; <bitop>; EXIT` (no bitwise C programs) |
| **SHL SHR** | **raw** | `IMM a; PSH; IMM b; <shift>; EXIT` |
| LI SI PSH LEA JSR ENT | corpus | `var_simple_0` (9-step: store+load, frame, call) |
| ADJ LEV | corpus | `func_identity_0` (11-step function return) |
| IMM | corpus | `edge_literal`/`edge` (2-step) |
| JMP | corpus | `edge_loop_never` (15-step) |
| BZ | corpus | `edge_if_zero`/`edge_if_one` (4-step) |
| **BNZ** | **raw** | branch-taken + fall-through (compiler emits BZ, never BNZ) |
| **LC SC** | **raw** | `LC` load from data base 0x10000; `SC` store-char then load-back |

Every one of the 30 op-classes has ≥1 decode-checked representative
(`python tools/run_per_op_oracle.py --map`).

## 4. Baseline (recorded on main `02e84ee8`, golden `1c04c3fd`)

Recorded with `tools/run_per_op_oracle.py` (faithful full_trace verdict per
program). **22 of 30 op-classes PASS all representatives; 8 FAIL** — every FAIL
is a REAL, pre-existing per-op decode divergence (NOT a bug in the oracle — the
ISA-VM oracle value is correct and the neural model decodes a different byte):

| status | op-classes |
|---|---|
| **PASS (22)** | ADD SUB MUL DIV MOD · EQ LT GT · OR XOR SHL · IMM BZ BNZ · PSH LI SI LEA JSR ENT ADJ LEV |
| **FAIL (8)** | AND · NE LE GE · SHR · LC SC · JMP |

Representative fail evidence (from the recorded verdicts):

* `AND 255 & 15` → oracle 15, **model decodes AX=1** (diverges @ step 3). The
  `108 & 58` case passes — the fail is operand-pattern-specific (matches the
  16-bit-bitwise-recover notes in memory).
* `NE`/`LE`/`GE` — one operand ordering passes, the reversed one fails (the
  comparison-decode margin family).
* `SHR 255 >> 4` fails; `200 >> 2` passes.
* `LC`/`SC` (char load/store) — the model decodes AX=0 (the load-char / store-
  char path the C corpus never exercises is not decoded at all).
* `JMP` — the `edge_loop_never` representative (a loop that never runs) diverges
  at the loop-skip branch. A branch-decode fail; the frame ops that share
  `var_simple_0`/`func_identity_0` (LI/SI/LEA/JSR/ENT/ADJ/LEV) all PASS.

The frame ops PASS: the memory-smoke framing-drift program `var_simple_0`
decodes correctly (SI/LI/LEA/JSR/ENT all 1/1 pass, sharing that program) and
`func_identity_0` decodes correctly (ADJ/LEV pass). These are the historically
hard framing-drift clusters — their per-op decode is CLEAN on main today.

The recorded per-program verdicts are in `tools/per_op_oracle_baseline.json`.
Regenerate with:
`OMP_NUM_THREADS=4 python tools/run_per_op_oracle.py --workers 8 \
--record-baseline tools/per_op_oracle_baseline.json`.

**Speed note (measured).** Uncontended, the full 30-op-class suite at
`--workers 8` lands in ~5–8 min. Under heavy box contention (other agents; load
~20) the per-program growing-tape decode slows ~4x, so plan `--workers` and
`OMP_NUM_THREADS` against the actual free-core budget (the runner is CPU-bound,
not memory-bound past ~1.5GB/worker).

**Reading the baseline.** A `pass` means the neural model's decoded `(PC, AX)`
matched the ISA-VM oracle trace at EVERY completed step (the full_trace
criterion). A `fail` is a REAL per-op decode divergence — e.g. the oracle already
surfaces `AND 255 & 15 → model decodes 1 not 15` (a genuine bitwise-decode bug,
consistent with the 16-bit-bitwise notes in memory). Those pre-existing fails are
part of the baseline: the gate's job is to catch a `pass → fail` **regression**,
not to assert the whole ISA decodes perfectly today.

## 5. GATE PROTOCOL — when to use this vs the GPU gate

**Use the per-op oracle (fast, CPU, parallel) for DECODE-PRESERVING op-class
work:**
* generator collapse / rule dedup that is claimed byte-identical,
* corrector deletion with a clean root (the removed rule is inert),
* per-op abstraction/rebuild that should not move any op-class's decode.

Protocol:
1. Record the baseline once on the known-good tree:
   `OMP_NUM_THREADS=2 python tools/run_per_op_oracle.py --workers 8 \
   --record-baseline tools/per_op_oracle_baseline.json`
2. Make the change, then gate it:
   `OMP_NUM_THREADS=2 python tools/run_per_op_oracle.py --workers 8 \
   --baseline tools/per_op_oracle_baseline.json`
3. **Any op-class `pass → fail` BLOCKS the change** (exit 1). A change that
   moves an op-class decode is NOT decode-preserving — send it to the GPU gate.
4. While iterating, narrow: `--op-classes ADD,SUB,MUL` (only the touched ops).

**RESERVE the GPU gate (`tools/run_1096_canonical.py --criterion full_trace` or
`tools/fast_gate.py`) for EMERGENT / CROSS-OP changes:**
* anything that reroutes a SHARED bus (e.g. the L10 OUTPUT byte-1 bus — the
  `C4_DERIVE_ADDSUB −34`): the per-op unit representatives cannot see a
  regression **spread thinly across many multi-byte consumers / deep loops**
  the way the population-weighted corpus sample can;
* changes to attention / cross-step carry / framing (the `var_*`/`func`/`nested`
  framing-drift bucket) where the interaction, not the single-op decode, is the
  risk;
* the FINAL sign-off before landing a campaign fix.

**Why the split is safe.** The oracle's verdict is byte-identical to the GPU
gate's PER-PROGRAM criterion, so for the SPECIFIC programs it runs there is zero
faithfulness gap — a decode-preserving change that keeps every op-class
representative `pass` genuinely did not move THOSE decodes. What the oracle does
NOT sample is the corpus **population** (a −34 spread across 10 clusters at
1-2 samples/cluster) and the deep-loop / framing-drift interactions — exactly the
`fast_gate` / GPU-gate territory. Match the tool to the change's blast radius:
decode-LOCAL → per-op oracle; decode-EMERGENT → GPU.

## 6. Limitations / honest boundaries

* **Per-op representatives, not the population.** The oracle proves the sampled
  programs' decodes are unmoved; it does NOT extrapolate a corpus delta. A fix
  whose regression is a thin spread needs `fast_gate.py`.
* **Short programs only.** Deep programs (loops/gcd/rec, >~40 steps) cost minutes
  each (O(steps²) growing tape) and are `skipped` past `--max-steps-cap`; the
  op-classes they'd exercise are already covered by short representatives.
* **Some op-classes fail on main today** (baseline-recorded) — the model does not
  decode every ISA op perfectly (bitwise/char being the known-weak families).
  The gate catches REGRESSIONS against that baseline, not absolute correctness.
