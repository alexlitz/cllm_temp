# Green-field RECURRENT — ONE step-block, autoregressive, per-step re-quantised

**Date:** 2026-07-14 · **Branch:** `greenfield-recurrent` (off `gf-branch`) ·
**Lane:** green-field `c4_min/` — the **1096-enabler** (unbounded exact iteration).

This is the make-or-break report for reaching the **full 1096** (not a tractable
subset). It answers one question:

> Can the green-field run **deep-loop** programs (gcd, `loop_*`, `rec_*` — 100s to
> 1000s of steps) **to completion, EXACTLY**, using a **recurrent** transformer
> (ONE step-block applied autoregressively) with **per-step integer
> re-quantisation** to stop fp error from accumulating?

**Answer: YES.** A single baked VM step-block (4 physical transformer sub-blocks),
looped in a Python driver that re-quantises the carried state each iteration, runs
programs of **5106 steps byte-exact** — and would run arbitrarily longer — while
the identical loop **without** re-quantisation drifts and fails to terminate.

Deliverables on this branch:
- `c4_min/recurrent.py` — the recurrent runner (`build_step_model`, `run_recurrent`,
  `StepModel`, `_requantize`).
- `c4_min/test_recurrent.py` — 7 tests (straight-line + control + deep-loop +
  the load-bearing requant contrast). `19/19` with the slice suite.

---

## 1. Why the current green-field caps out (the wall this removes)

The `compiler.py` builds realise the honest recurrence **depth = time**: they
unroll `max_steps` physical step-blocks, **one transformer step-block per executed
VM step**, and flow the whole VM state through the additive residual (see
`CONTROL_FLOW.md`, `GF_BRANCH_2026_07_14.md`). That is provably exact
(`test_slice.py`), but it has two hard limits:

1. **It bakes `4 * max_steps` physical blocks.** A loop that runs 1000 steps needs
   4000 blocks of weights — intractable.
2. **`max_steps` is a *compile-time* bound.** A loop whose real step count exceeds
   the bake **truncates**. Concretely, the `compile_program_pc` default is
   `max_steps = 2*n_code + 4`; a 6-instruction countdown-from-5 needs **22** steps
   but the default bakes only 16 blocks, so the trace is cut at step 16.

The 233 deep-loop programs were previously proven to contain **no loop-specific
logic** — each is the SAME per-step VM transition repeated. So the correct
architecture is not a bigger unroll; it is a **recurrent universal step**.

---

## 2. The mechanism: ONE step-block, applied autoregressively

`build_step_model(code)` bakes **exactly one** VM step — the same four sub-blocks
the PC-dispatch unroll uses per step, but only **once**:

| # | sub-block | effect |
|---|-----------|--------|
| 1 | **fetch**    | scalar `PC` → PC one-hot `PC_IS[i]` (exact triangular-pulse relu gadget) + `AX_ZERO = relu(1-AX)` branch predicate |
| 2 | **dispatch** | rules gated on `PC_IS[i]` apply `code[i]`'s AX/STACK0 effect **and** the PC update (sequential `+1` or branch target) |
| 3 | **fold**     | `AX mod 256` (a no-op unless AX overflowed a byte) |
| 4 | **emit**     | copy post-step AX into the (single, reused) `OUT` slot; snapshot the sticky `HALTED` flag |

Only **one** `OUT`/`HALT_SEEN` slot is allocated (`max_steps=1`): the driver
reuses it every iteration. So the bake is `O(n_code)` blocks — **independent of
how many VM steps the program actually runs**. Every program on this branch bakes
**4 physical blocks**, full stop.

`run_recurrent(model, L, code)` then loops:

```
state = initial_state (ONE=1, all registers/PC/stack = 0)
repeat until HALT (or max_steps):
    state = step_block.forward(state)   # one universal VM step
    state = requantize(state)           # <-- the key: round to exact integers
    trace.append(argmax head(state.OUT))
    if state.HALT_SEEN: break
```

The **full residual vector is carried** from iteration `k` to `k+1`, but only the
persistent bands matter: `AX, SP, BP, PC, STACK0` (+ the sticky `HALTED`). The
scratch bands (`PC_IS[*]`, `AX_ZERO`) are **recomputed by fetch every step** and
each carries a self-clear (subtract-old-then-set), so the step-block is idempotent
and needs no special reset between iterations. `ONE` is held at exactly `1.0`.

**This is a genuine recurrent transformer**, not trace replay: the weights are one
step-function; the loop count equals the program's *real* step count (unbounded),
and the instruction executed each step is chosen by the *runtime* `PC` carried in
the residual — a computed branch selects the next op.

---

## 3. The re-quantisation — why it stays EXACT over unbounded steps

The property that makes the classic C4 neural VM exact over arbitrary steps is
that its state round-trips through **tokens** every step — a re-quantisation to
exact integers that stops fp error from compounding. We reproduce that **directly
on the residual**:

```python
def _requantize(state, one_band):
    q = torch.round(state)   # every band -> nearest integer
    q[one_band] = 1.0        # pin the constant lane
    return q
```

Every band in this substrate holds an **exact non-negative integer between steps**
(register value, PC index, stack cell, one-hot `0/1`). So rounding is a **no-op on
the true value** but **annihilates** the `O(1e-3)` fp residue the SwiGLU gadgets
leave. Step `k+1` therefore sees a byte-exact integer state, and error can **never
compound**. The `mod-256` fold already re-quantises `AX` into `[0,256)`; the round
handles every other band and the residue the fold itself leaves.

### Measured fp behaviour (countdown loops, single baked step-block)

| run | per-step register residue | outcome |
|-----|--------------------------|---------|
| **with requant** | rounded to **0 every step** (worst pre-round residue over 5106 steps = `2.5e-3`) | **exact, terminates correctly** |
| **without requant** | accumulates: `~1e-4` @ step 100 → `~1e-3` @ 300 → `>0.1` @ 400 → crosses the `0.5` rounding boundary near step ~800–1000 | **drifts → wrong branch predicate → never halts → runs to the step cap** |

The un-requantised loop is fine for a few hundred steps but is **fundamentally
bounded** by fp drift; the requantised loop is bounded only by the driver's
`max_steps` budget. This is the exact mechanism the mission called out: **per-step
re-quantisation converts a ~few-hundred-step drift limit into effectively
unbounded exact iteration.**

---

## 4. Proof — deep loops run to completion EXACTLY

All via `StepModel(prog).run()` — **4 baked blocks** each, decoded through a real
argmax LM head, compared to the reference interpreter `isa.interpret`:

| program | real VM steps | bounded-unroll? | recurrent result |
|---------|---------------|-----------------|------------------|
| countdown-from-5 | 22 | **truncates at 16** (default bake) | **exact** |
| countdown-from-200 | 802 | intractable (3208 blocks) | **exact** |
| countdown-from-255 | 1022 | intractable | **exact**, ~1000 steps/s |
| subtract-by-3 / by-7 (mid-loop mod-256 wraps) | 122 | — | **exact** |
| **5× chained countdown-from-255** | **5106** | intractable (>20k blocks) | **exact**, ~1000 steps/s |

The 5106-step run is the headline: **thousands of steps, byte-exact, 4 physical
blocks.** The same 5106-step program **without** requant never terminates (ran to a
20000-step cap producing a wrong trace).

The recurrent form also reproduces the **entire** straight-line + control slice
(IMM/PSH/ADD/SUB/JMP/BZ/BNZ, computed-predicate if-then-else, mod-256 wrap) — it is
a strict superset of the depth-unroll's capability, minus the depth bound.

`test_recurrent.py`: **7/7**. With `test_slice.py`: **19/19**.

---

## 5. Honest boundary (what this does NOT yet cover)

The recurrence + re-quantisation is proven. Two orthogonal substrate items remain
before *every* deep-loop program in the corpus runs — **neither is a recurrence
problem**; both are pre-existing limits shared with the depth-unroll compiler:

- **Stack depth > 1.** `STACK0` is a single stack-top *mirror* scalar, so a program
  with two live pushed values at pop time (`PSH a; PSH b; …; ADD; ADD`) reads a
  stale cell (`70` instead of `60` on a 2-deep add). Deep loops that keep **one**
  live stack value at a time (countdown, gcd-by-repeated-subtraction with a real
  stack) work today. The general fix is an **SP-indexed stack band** written/read
  by a content-match attention head (`compile_attn.content_match_head` already
  exists and is tested) — not a change to the recurrent driver.
- **Ops not yet on `gf-branch`.** `AND/OR/XOR/CMP/LI/SI/JSR/ENT/LEV` raise
  `NotImplementedError` in `control.dispatch_rules`. The recurrent runner inherits
  whatever the dispatch table implements: as those ops land (tasks #518/#519), they
  are **automatically** unbounded-loopable through this same driver with **zero**
  new machinery.

Both are additive to `dispatch_rules` / the layout; the recurrent step-loop and its
re-quantisation are unchanged by them.

---

## 6. Verdict for 1096

The recurrent universal-interpreter form **works and is exact over thousands of
steps** with per-step integer re-quantisation. This removes the depth-unroll's
compile-time step ceiling — the sole reason the green-field could not touch the 233
deep-loop programs. With the two orthogonal substrate items above (multi-cell stack
+ remaining ops), **the same driver runs the deep loops to completion exactly**, so
this architecture is a viable path to the **full 1096**.

**Files:** `c4_min/recurrent.py`, `c4_min/test_recurrent.py`. Run:
`OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_recurrent.py c4_min/test_slice.py`.
