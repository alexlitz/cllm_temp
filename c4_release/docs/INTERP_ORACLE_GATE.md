# `tools/interp_oracle_gate.py` — CPU-only interpreter-vs-oracle attribution gate

A fast, GPU-free, **rule-attributable** structural debugging gate. It runs the
value-faithful pure-IR interpreter
([`neural_vm/unified_compiler/faithful_interpreter.py`](../neural_vm/verification/faithful_interpreter.py))
over each program's **production decode setup**, reproduces the model's per-step
`(PC, AX)` decode the way the production fail-fast path does, compares it to the
DraftVM oracle, and on the first diverging byte **names the single owning
declarative rule** that produced the wrong value — all on CPU in ~10-15 s of
build + a few seconds per program.

It turns bug-hunting from "GPU residual probing" into "run the gate, read the
attributed rule".

## How to run

```bash
# Smoke set (the fastest sanity run):
CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --smoke

# A cluster-sampled slice of the 1096 corpus (known-bug clusters first):
CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --sample-1096 90

# Exact corpus ids (ranges allowed):
CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --ids 0,120,167-170

# The known-buggy-cluster attribution demo:
CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --demo

# Verdict-only (skip the per-fail attribution second forward — ~2x faster):
CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --sample-1096 90 --no-attribute

# Trustworthiness: confirm interp == NEURAL on a sample (uses a FREE GPU if idle):
python tools/interp_oracle_gate.py --faithfulness-check 40
```

The build is **CPU-only** (`alu_mode='efficient'`, in-memory, `disk_cache=False`
so the layout keeps its `compiler_ir_factory` lambdas for attribution). Prefix
with `CUDA_VISIBLE_DEVICES=""` so it never touches the GPU lanes. The
`--faithfulness-check` mode is the only one that uses a GPU, and only when one is
idle (≥10 GB free, <40 % util); otherwise it skips with a note.

## What each class means

| Class | Meaning |
|-------|---------|
| **PASS** | The interpreter's per-step `(PC, AX)` matches the DraftVM oracle every step through HALT, AND the program's path executes no opaque composite-ALU op. |
| **FAIL** (HIGH-confidence) | First diverging step+register+byte is a **PC** byte or **AX byte 0** — the bytes the flat teacher-forced decode reproduces faithfully (control flow + the single-byte result). The gate **attributes** the wrong byte to the single declarative rule whose runtime SwiGLU contribution dominates it. **This is the key output.** |
| **FAIL?** (LOW-confidence) | First divergence is an **AX byte ≥ 1** (or SP/BP/STACK0). These ride the cross-step re-anchoring a flat forward cannot reproduce, so the divergence MAY be a real multi-byte bug OR a flat-decode artifact. NOT attributed (naming a rule there would mislead). Confirm with `--faithfulness-check`. |
| **ALU-OPAQUE** | The divergence is at — or the program's path executes — one of the 4 still-imperative composite-ALU blocks (ADD/SUB → `AddSub5StageBlock`, MUL → `FlattenedALUMul`, DIV/MOD → `FlattenedDivMod`, SHL/SHR → `ALUShiftComposite`). These have no IR rule form; the interpreter runs the real composite block but it is **not bit-certified vs production's imperative ALU**, so the gate declines to judge (PASS or attribute) and flags the opaque op. Needs the imperative GPU path. |
| **ERROR** | The oracle or the faithful forward raised (compile error, no steps, etc.). |

## Why it is faithful (and why the model build matters)

The interpreter's attention (softmax1 + ALiBi MHA) and FFN (SwiGLU @ scale S) math
is the term-for-term reproduction of the lowered `AutoregressiveAttention` /
`PureFFN` blocks; `tools/faithful_interpreter_validate.py` validates it is
byte-for-byte identical to the real model's argmax at every token position. The
gate drives that same forward over the **production decode setup**:

1. **Production model.** The gate builds with **`alu_mode='efficient'`** — the
   SAME model `BatchedPureNeuralRunner` builds (`trust_neural_alu=True` ⇒
   efficient ⇒ 49 physical blocks). The lookup-mode model (47 blocks) is **NOT**
   byte-equivalent: its different block layout mis-decodes ops the production
   efficient model gets right (e.g. `or_basic`/`xor_basic`). Building the wrong
   ALU mode silently corrupts the verdict — this is the single most important
   fidelity requirement.
2. **Production context.** The seed is the `_build_context` code prompt
   (`CODE_START … bytecode … CODE_END … DATA`) PLUS the DraftVM-teacher-forced
   35-token step slices — exactly what `run_batch_fail_fast` feeds the model.
3. **Production decode.** Per VM step the model's predicted `(PC, AX)` is read
   from its argmax over that step's draft slice, decoded at the **fixed**
   step-tape offsets (PC bytes at slice offsets 1..4, AX bytes at 6..9) — i.e.
   the markers are re-anchored exactly as production re-anchors the first token
   of each step via the Python STEP_END dispatch. This is what makes a passing
   program decode cleanly (a naive marker-search or raw next-token compare flags
   every step boundary, even for passing programs).

### The characterized faithfulness gap (be honest about it)

A flat single forward **cannot** reproduce production's cross-step re-anchoring
for the **AX high bytes (1..3)**: production's KV-cache + accept/correct loop
carries those bytes across steps; the flat forward reads them as a stale/zero
value. So an AX-byte≥1 divergence is reported as **LOW-confidence `FAIL?`**, never
attributed. The PC bytes and AX byte 0 — the bytes that determine control flow
and the single-byte result — ARE reproduced faithfully and are **HIGH-confidence**.

This was measured directly against the production GPU fail-fast on the smoke set:
**31/41 programs agree exactly** (every PASS + every HIGH-confidence FAIL match
production); the 10 disagreements are ALL the AX-byte≥1 cross-step class (9 on
AX[1]), exactly the characterized gap — and they are flagged `FAIL?`, not `FAIL`.
The `--faithfulness-check` mode quantifies this split on any sample: it confirms
interp == neural on the gate-authoritative set and reports the AX-high-byte gaps
separately so the gate never mis-attributes them.

## The payoff — top owning rules (highest-leverage fix targets)

On a 90-program cluster sample (div/mod/var/if/sub/add/mul/expr/func/… first),
the class split is roughly:

```
~14 PASS | ~12-14 FAIL (HIGH-conf, attributed) | ~25 FAIL? (LOW-conf AX
high-byte) | ~27 ALU-OPAQUE   →   gate AUTHORITATIVE ≈ 28-30 / 90, ALU-OPAQUE ≈
27/90 (every add/sub/mul/div/mod/shl/shr program), LOW-conf ≈ 25/90.
```

(The ALU-OPAQUE count is large because the corpus is ALU-heavy and the gate
refuses to certify any program whose path touches an imperative composite-ALU
block.) The HIGH-confidence FAILs land exactly on the documented per-step roots:
`var_mul`/`var_three` AX[0] `0xe0` vs `0xe8` (the L16 PSH `0xE0` sentinel),
`if_var` AX[0] `0x00` vs `0x01`, `func_identity` **PC[0]** `0x5a` vs `0x0a` (the
func/nested callee-PC bug, tasks #227/#235), `rec_factorial` AX[0] `0x01` vs
`0x00` — i.e. the gate independently finds the known bug clusters on CPU.

The attribution fires on **FFN-rooted** fails and reports the owning rule:

```
TOP OWNING RULES (by HIGH-confidence FAIL frequency):
    Nx  layer16_lev_routing::l16_psh_mem_addr0_restore_hi_14
    Nx  layer6_routing_ffn::l6_psh_stack0_marker_final_{lo,hi}_0
```

Two independent confirmations the attribution finds the documented built-dim roots:

* **`layer6_routing_ffn::l6_psh_stack0_marker_final_{lo,hi}_0`** is the
  **STACK0-marker dump rule family** — the exact `l6_psh_stack0_marker_final_lo_0`
  rule the validation harness's attribution demo independently fingered for
  `10*9/1`'s wrong `0x00`, and the #221 STACK0 byte-0 high-nibble framing-drift
  surface. The gate finds it WITHOUT a GPU.
* **`layer16_lev_routing::l16_psh_mem_addr0_restore_hi_14`** is the L16 PSH
  `MEM_addr0=0xE0` sentinel family from the open-bug memory notes (the
  ~1e15-magnitude runtime contribution is the load-bearing sentinel write that
  note describes). It is the dominant attributed owner across the smoke `lea`/`adj`
  fails and the `var_mul` cluster.

**Honest limit of the attribution.** Some HIGH-confidence FAILs report
`(no runtime FFN writer)` — `if_var`, `func_identity` PC, `rec_factorial`. That
is NOT a gate failure: it means the wrong byte is produced **upstream of the
FFN** — an attention relay (the operand/PC gather), an embedding default, or a
cross-step carry — not by a declarative FFN rule. That is itself a diagnostic: it
rules OUT an FFN-rule root for those clusters and points the debugger at the
attention/relay layer (consistent with the `func` PC and `if_var` notes). The
gate localizes the divergence (step + register + byte) even when it cannot name an
FFN rule.

So the gate **independently reproduces the known built-dim roots** for the
HIGH-confidence cluster and is honest about the three boundaries: it does NOT
attribute the multi-byte AX-high-byte carry bugs (LOW-confidence by construction
— the flat decode can't see the carried high byte), the imperative ALU bugs
(ALU-OPAQUE), nor the attention/relay-rooted bytes (reported as "no FFN writer").

## Coverage summary (what the gate can and cannot certify)

* **INTERPRETABLE + faithful (gate authoritative):** PASS + HIGH-confidence FAIL —
  PC/AX-byte-0 control-flow and single-byte-result bytes. The attribution here is
  trustworthy (matches production + the known roots).
* **ALU-OPAQUE:** any program whose path executes ADD/SUB/MUL/DIV/MOD/SHL/SHR —
  the imperative composite-ALU `#230` coverage gap. The gate declines to judge;
  use the GPU imperative-forward path.
* **LOW-confidence (`FAIL?`):** AX byte ≥ 1 cross-step re-anchor — a known
  flat-decode limitation. May be a real bug or an artifact; confirm with
  `--faithfulness-check`.

## Relationship to the other gates

* [`tools/dsl_spec_gate.py`](../tools/dsl_spec_gate.py) — COVERAGE-faithful
  (bag-of-dims; "is the expected nibble written at all"), CACHE-immune, but NOT
  value-faithful. Use it for the rule-coverage necessary condition.
* [`tools/faithful_interpreter_validate.py`](../tools/faithful_interpreter_validate.py)
  — validates the interpreter math is byte-for-byte == the model (the dependency
  this gate trusts) + a single attribution demo.
* **This gate** — runs the value-faithful interpreter over the WHOLE corpus in
  the production decode setup and attributes every HIGH-confidence fail to its
  owning rule. The corpus-scale, rule-attributable layer the other two lack.
