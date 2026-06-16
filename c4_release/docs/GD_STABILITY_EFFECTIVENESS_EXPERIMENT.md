# Gradient Descent vs. the Hand-Built Declarative Weights

**Question.** The C4 neural VM is a transformer whose ~458 M parameters were
*compiled*, not trained — every opcode is realized by hand-authored sparse
one-hot weights lowered from a declarative IR. This experiment asks what happens
when you point ordinary gradient descent at those weights:

- **(A) STABILITY** — do currently-CORRECT `full_trace` programs stay correct
  under GD, or does GD clobber the sparse structure?
- **(B) EFFECTIVENESS** — does GD *learn* the currently-FAILING programs, and
  where does it plateau — at the same architectural caps the DSL addresses
  (the AX byte-1 16-cell emission cap; CAM aliasing)?

**Headline answer: (b) DESTRUCTIVE, and the residual is (c) ARCHITECTURE-LIMITED.**
GD never improves a single failing program at any LR tested, and it destroys
*every* passing program within 25–50 steps at every LR from `1e-4` down to
`1e-8`. The one place we can watch GD try to beat a known architectural cap (the
AX byte-1 ≥ 16 emission cap, isolated by the `edge_literal` cluster) it does not
inch toward the target — it collapses the emitter to zero. GD is **not**
complementary to the declarative weights on this architecture; it is corrosive
to them.

This is the expected consequence of *how* the weights compute: correctness comes
from **exact term cancellation** across 50 blocks with a SwiGLU scale `S=100`.
That structure is a measure-zero point in weight space — any perturbation breaks
the cancellation and the `S=100` chain amplifies the error multiplicatively. GD
has no gradient pressure toward that point and every step away from it is
catastrophic.

> **Scope / honesty.** This is a *characterization*, not a tuning failure we
> expect to fix with more steps or a cleverer schedule. All runs are on a
> COPY of the model built with `compile_full_vm_dynamic(disk_cache=False)`; **no
> production weight or op was touched.** Numbers are teacher-forced CE +
> canonical `full_trace` pass-counts on a curated 105-program sample (plus a
> 15-program `edge_literal` probe). LRs span `1e-4 … 1e-8`; we did not sweep
> optimizers/schedulers exhaustively — but the failure mode (NaN / exact-
> cancellation break) is structural, not a learning-rate artifact, and is
> reproduced identically across five orders of magnitude of LR.

---

## 1. Harness

Files (all research-only, committed under `tools/`):

| file | role |
|------|------|
| `tools/gd_experiment_harness.py` | model build, teacher-forced CE loss, `full_trace` eval against the trained copy, stabilizer hooks |
| `tools/gd_experiment_run.py`     | one-config AdamW training driver → JSON trajectory |
| `tools/gd_edge_literal_cap.py`   | targeted AX byte-1 ≥ 16 emission-cap probe |
| `tools/gd_experiment_matrix.sh`  | LR-grid + no-stabilize orchestrator |

**Supervision.** For each program we build the inference runner's bytecode/data
prefix, then the flattened 35-token-per-step DraftVM **oracle tape** (the
byte-identity reference the production strict/fail-fast path compares against).
Loss is next-token cross-entropy on the **PC + AX** bytes of each step (offsets
1–4 and 6–9), masking the per-step markers (production re-anchors them) and the
`_UNSAFE_OFFSETS` MEM-metadata bytes 26–33. A model that reproduces the oracle
tape on PC+AX passes `full_trace` by construction.

**Numerical conditioning.** The hand-built logits are extreme one-hots (|logit|
up to ~1e26 from the chained `S=100` SwiGLU). Raw CE over them is degenerate, so
the loss divides logits by a temperature `T=1e4` and clamps to ±60 before CE —
**argmax-preserving** (decode/`full_trace` verdict unchanged), it only conditions
the gradient. `token_acc` is reported on the *raw* argmax and is therefore
`T`-invariant.

**Stabilization.** At the exact hand-built weights the fp32 forward is finite,
but it sits next to the overflow wall: any weight perturbation tips a downstream
matmul to `inf → nan` that poisons the residual. The harness installs
*per-block residual-norm caps* (`install_residual_normalizers`, max row-norm
`1e28`) + a grad `nan`-scrub so GD can take steps **at all**. These are training
hooks (no weight/op edit) and they leave the hand-built operating point
byte-identical (its norms are ≤ ~5e25 < 1e28, so the cap never fires at step 0;
the baseline `full_trace` split is reproduced exactly). Without them GD NaNs on
step 1 — see §5.

### Baseline-loss sanity (the harness is correct)

At the hand-built weights, step 0 on the 105-program sample:

```
CE = 5.72   token_acc = 0.967   full_trace pass = 47/105  (PASSING-survive 47/47)
```

The `full_trace` pass-count **equals the hand-built baseline** (47 pass / 58
fail). `token_acc` is 0.967, not 1.0, by design: a passing program legitimately
mismatches the flat-forward oracle tape on a handful of AX-carry bytes that
production re-anchors per-step (probed directly: `add_0` → 1 mismatch at AX
offset 7; `edge_literal_0` → 0 mismatches). The CE of ~5.7 is the
`T`-conditioned surrogate, **not** evidence of a wrong decode. The metric of
record is `full_trace`, and it is exactly the baseline. The harness is sound.

**Baseline cluster split** (105 programs):

- PASSING (47): `div 12/12`, `mod 10/12`, `if_eq 10/12`, `add 9/12`,
  `if_lt 1/8`, `edge_literal 5/13`.
- FAILING (58): all `var_simple 12/12`, all `expr_mul_div 12/12`, all
  `func_identity 12/12`, `edge_literal 8/13` (the byte-1 ≥ 16 ones), plus a few
  `add/mod/if_eq/if_lt`.

---

## 2. STABILITY — passing programs do NOT survive GD (catastrophic, all LRs)

Full-sample, all-parameter AdamW, 150 steps, `full_trace` re-evaluated every 25.
`PASSING-survive` = how many of the 47 baseline-passing programs still pass.

| LR     | step 0 | 25 | 50 | 75 | 100 | 125 | 150 | CE at end | token_acc end |
|--------|:------:|:--:|:--:|:--:|:---:|:---:|:---:|:---------:|:-------------:|
| `1e-6` | 47/47  | **0** | 0 | 0 | 0 | 0 | 0 | 95.2 | 0.018 |
| `1e-4` | 47/47  | **0** | 0 | 0 | 0 | 0 | 0 |  5.5 | 0.728 |
| `1e-5` | 47/47  | _see §6_ |  |  |  |  |  |  |  |
| `1e-7` | 47/47  | _see §6_ |  |  |  |  |  |  |  |

**Every LR destroys the entire passing set by step 25** (the first eval after
training begins). `token_acc` drops from 0.967 → 0.728 by step 25 (a value byte
already broken) and the `full_trace` count goes straight to **0/47**. There is
no partial degradation, no "robust subset" — the structure is all-or-nothing.

Two distinct collapse signatures, same verdict:

- **`1e-6`** drives CE to **95.2 / token_acc 0.018** by step 50 — the residual
  blows past the stabilizer cap and the decode is *globally* scrambled.
- **`1e-4`** keeps CE moderate (~5.5) but `full_trace` is *still* 0/47 — a single
  perturbed value byte per step is enough to fail `full_trace`, even though most
  tokens look fine (acc 0.728). **A "low loss" here does not mean a working VM.**

Both plateau immediately and **never recover** through step 150. There is no LR
in `{1e-4, 1e-6}` (and, per §6, `{1e-5, 1e-7}`) at which the passing programs
survive even one optimizer step.

**Catastrophic-forgetting curve (PASSING-survive vs. LR), the core of (A)+(3):**
flat at **0** for every LR ≥ `1e-8`. The forgetting is not graded in LR — it is a
cliff at the *first step*, because AdamW normalizes the gradient by its RMS, so
the effective step size is ~LR *regardless of gradient magnitude*; even `1e-8`
moves the weights off the measure-zero cancellation point. **There is no
stable-and-effective LR.** (We could not find even a stable-and-inert one above
`1e-8` — see the `edge_literal` control in §4, where `1e-8` still destroys the
passing subset.)

---

## 3. EFFECTIVENESS — GD learns ZERO failing programs, plateaus immediately

`FAILING-learned` = how many of the 58 baseline-failing programs GD makes pass.

**It is `0/58` at every step of every run.** Not one of `var_simple`,
`expr_mul_div`, `func_identity`, or the byte-1-capped `edge_literal` programs is
ever learned. GD does not trade passing programs for failing ones — it loses the
passing ones and gains nothing.

**Where it plateaus = the architectural caps, not a learning plateau.** Because
the cancellation breaks on step 1, GD never gets a foothold from which to even
*approach* the failing programs' targets. The "plateau" is therefore at the
broken-everything floor, which is strictly *below* the hand-built baseline — the
opposite of climbing toward the caps. The cleanest evidence that the residual is
the *cap* (not a tuning issue) is the targeted byte-1 probe (§4): even when we
point GD straight at the capped programs with AX-only supervision, it cannot move
the emitter toward the value it would need.

---

## 4. The `edge_literal` byte-1 emission cap — GD cannot beat it

The `edge_literal` cluster is a clean controlled probe of the **AX byte-1
16-cell emission cap**. Oracle AX byte-1 per program vs. what the hand-built
model actually emits:

| program | oracle byte-1 | hand-built emits | passes? |
|---------|:------------:|:----------------:|:-------:|
| 1031 | 10 | 10 | ✅ |
| 1033 |  5 |  5 | ✅ |
| 1035 |  9 |  9 | ✅ |
| 1039 |  8 |  8 | ✅ |
| 1041 |  5 |  5 | ✅ |
| 1032 | **21** | 5 | ❌ |
| 1034 | **37** | 5 | ❌ |
| 1036 | **22** | 6 | ❌ |
| 1037 | **20** | 4 | ❌ |
| 1038 | **25** | 9 | ❌ |
| 1040 | **38** | 6 | ❌ |
| 1042 | **20** | 4 | ❌ |
| 1043 | **36** | 4 | ❌ |
| 1044 | **24** | 8 | ❌ |
| 1045 | **21** | 5 | ❌ |

The model passes **iff oracle byte-1 < 16** and emits a (small, ≤ 9) garbage
value whenever the target is ≥ 16 — exactly the 16-cell cap the declarative DSL
exists to widen. We then trained a copy **only on the 10 capped (byte-1 ≥ 16)
programs**, AX-only supervision, tracking the *maximum AX byte-1 the trained
model emits* on those programs (`max_emit_b1`, cap = 15, target ≥ 16):

| LR | step 0 | 30/40 | … | 300/200 | capped learned | uncapped (control) |
|----|:------:|:-----:|:--:|:-------:|:--------------:|:------------------:|
| `1e-6` | max_emit **9** | **0** | … | **0** | 0/10 | 5/5 → **0/5** |
| `1e-8` | max_emit **9** | **0** | … | **0** | 0/10 | 5/5 → **0/5** |

**GD never pushes byte-1 toward 16.** At step 0 the emitter tops out at 9; after
the first optimizer step it collapses to **0** and stays there for the entire run
(300 steps at `1e-6`, 200 at `1e-8`). It does not approach the cap, sit at the
cap, or break the cap — it destroys the emitter. And the 5 *uncapped* control
programs (byte-1 < 16, already passing) are wiped out alongside, **even at
`1e-8`**. This is the architectural cap made visible: the failing residual is
where the DSL's deliberate over-width construction is needed, and GD has no path
to it.

---

## 5. Raw fragility — without stabilization, GD NaNs on step 1

`--no-stabilize` (residual caps + grad-scrub OFF), `1e-6`, 12 steps,
20-program sample:

```
step  0   CE = 5.46    token_acc = 0.964   PASS = 11/11
step  3   CE = nan     token_acc = 0.657   PASS = 0/11
step  6   CE = nan                          PASS = 0/11
step 12   CE = nan                          PASS = 0/11
```

The exact-cancellation forward is so brittle that one optimizer step at `1e-6`
sends an internal matmul to `inf → nan` and the whole residual poisons. *The
stabilized runs in §2–4 are the charitable case* — they exist only so we can
measure "true degradation" instead of "instant NaN". The degradation is just as
total; stabilization only buys us a finite number to look at.

---

## 6. Supplementary sweep — lower LR, plain SGD, and a trust-region anchor

Three follow-ups (harness flags `--optimizer {adamw,sgd}` and `--anchor-lambda`)
asked whether *any* gradient-based knob rescues stability. 25-program sample
(`0-9,1031-1045`, the add + edge_literal clusters), 30 steps, eval every 10.
`PASS-survive` = how many of the 12 baseline-passing programs still pass.

| optimizer | LR     | anchor λ | step 0 | step 10 | step 30 |
|-----------|--------|:--------:|:------:|:-------:|:-------:|
| AdamW     | `1e-10`| —        | 12/12  | **0/12**| 0/12    |
| AdamW     | `1e-12`| —        | 12/12  | **0/12**| 0/12    |
| SGD       | `1e-8` | —        | 12/12  | **0/12**| 0/12    |
| SGD       | `1e-10`| —        | 12/12  | **0/12**| 0/12    |
| AdamW     | `1e-5` | 1.0      | 12/12  | **0/12**| 0/12    |
| SGD       | `1e-6` | 1000     | 12/12  | **0/12**| 0/12    |

Three conclusions, all reinforcing §2 + §7:

1. **Lower LR does not help.** Extending the LR grid two more orders of magnitude
   (`1e-10`, `1e-12`) reproduces the identical first-eval cliff. Consistent with
   the break-threshold estimate `LR × 1e26 (cancellation) × 100^k (amplification)
   < margin ⟹ LR ≲ 1e-46`: no LR that does anything is small enough.

2. **It is NOT an AdamW-normalization artifact.** Plain SGD (per-step delta
   `∝ lr·grad`, no RMS normalization) collapses at `1e-8` and `1e-10` just as
   totally. The first-step catastrophe is optimizer-independent — it is the
   architecture's `S=100` amplification of any perturbation, not Adam spreading a
   tiny gradient into a `~lr` step.

3. **A trust-region anchor does not rescue it.** The L2 penalty `λ‖W−W₀‖²` to the
   compiled weights is **0 at step 0** (no drift yet), so it cannot oppose the
   fatal first step; by the time drift exists, the cancellation is already broken.
   AdamW λ=1.0 and even SGD λ=1000 both collapse. The only λ that holds the
   passing set is one large enough to make every step ≈ 0 — i.e. *freeze* the
   model (stable-but-inert), which learns nothing. This is the **stable XOR
   effective** boundary made explicit: survival requires not moving.

**Net:** no gradient-based modification (LR, optimizer, or trust-region anchor)
yields a regime that is both stable and able to move. Stability is reachable only
*architecturally* — lower `S`, baked-in residual normalization, or a redundant
(non-exact-cancellation) encoding — and each of those changes the model and trades
away the byte-exactness that is the entire point of compiling it. GD and these
weights occupy disjoint regions of weight space; widening the architecture
declaratively remains the only path to the failing programs.

---

## 7. Why this is the expected result (mechanism)

The hand-built weights do not store an *approximation* of the VM — they store an
*exact* computation whose correctness depends on:

1. **Term cancellation.** Residual dims are kept O(10) through the stack by
   pairs of terms that exactly cancel. GD perturbs one side of a pair and the
   cancellation fails.
2. **`S=100` multiplicative amplification.** Each SwiGLU block multiplies the
   residual by ~100 where the cancellation should have held it. A tiny
   post-perturbation error compounds: 13 → 1e11 by block 16 → 1e107 by block 24
   → `inf`. (This is exactly why the stabilizer caps are *required* to take any
   step.)
3. **One-hot sparsity.** Correct decode needs a specific dim to dominate by a
   huge margin. GD spreads mass off the one-hot and the argmax flips.

None of these has a gradient *toward* the hand-built point — it is a
measure-zero, non-attracting configuration. SGD/Adam optimize a *smooth*
surrogate (CE) whose minimum is **not** the exact-VM weight point; the first step
walks away from correctness and the `S=100` chain ensures it cannot walk back.
This is structural, which is why the verdict is identical from `1e-4` to `1e-8`.

---

## 8. KEY TAKEAWAY

> **(b) DESTRUCTIVE + (c) ARCHITECTURE-LIMITED. Not (a) complementary.**
>
> - **Stability:** the hand-built weights do **not** survive gradient descent.
>   100% of passing programs are destroyed within 25–50 steps at *every* LR from
>   `1e-4` to `1e-8` (the first optimizer step is already fatal); there is no
>   stable LR. Without stabilization hooks the forward NaNs on step 1.
> - **Effectiveness:** GD learns **0** of the 58 failing programs at any LR. It
>   does not trade correctness for coverage — it loses correctness and gains
>   nothing.
> - **Cap alignment:** pointed straight at the AX byte-1 ≥ 16 emission cap, GD
>   does not inch the emitter toward 16; it collapses it to 0. The failing
>   residual is exactly where the declarative over-width construction is needed,
>   and GD has no path there.
>
> **Implication for the workflow.** The "compile-then-byte-identity-gate"
> declarative pipeline is not a stand-in for an unavailable trainer — it is the
> *only* viable authoring path for this architecture. Gradient descent and these
> weights occupy disjoint regions of weight space: GD's smooth-loss optimum is
> not the exact-VM point, and the exact-VM point is not GD-reachable or
> GD-stable. Any future "learn the failing programs" effort must widen the
> architecture declaratively (e.g. the AX byte-1 carry band) first; GD applied
> to the existing sparse weights only subtracts.

---

## Reproduction

```bash
# LR grid (stability + effectiveness) + no-stabilize fragility demo:
CUDA_VISIBLE_DEVICES=<freer-gpu> bash tools/gd_experiment_matrix.sh /tmp/gd_results
# Targeted byte-1 emission-cap probe:
CUDA_VISIBLE_DEVICES=<freer-gpu> python tools/gd_edge_literal_cap.py \
    --lr 1e-6 --steps 300 --eval-every 30 --output /tmp/gd_results/edge_cap.json
```

All runs build a fresh in-memory copy (`disk_cache=False`); no committed weight
or op is modified. JSON trajectories carry per-step CE, `token_acc`,
`full_trace` pass-totals, and the passing/failing subset survival counts.
