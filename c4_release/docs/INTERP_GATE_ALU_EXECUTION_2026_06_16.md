# Interpreter oracle gate: ALU-execution path (ALU-OPAQUE retired)

**Date:** 2026-06-16
**Scope:** INFRA / tooling only. No production weights or ops changed
(model byte-identical, golden `ce9bf9616f3379c4`).
**Files:** `neural_vm/unified_compiler/faithful_interpreter.py`,
`tools/interp_oracle_gate.py`.

> **Re-integration note (this commit).** This change was first authored on a
> pre-decode-lane base (`06a1d12d`). It is here **re-integrated onto the current
> DECODE-LANE gate** (`docs/INTERP_GATE_FAITHFUL_DECODE.md`): the re-anchored
> marker decode, the cross-step poisoning guard (`_value_correction_step` /
> `vcorr`), and the AX-high-byte soundness are kept **fully intact**. The
> ALU-execution verdict and the decode/cross-step verdicts **COEXIST**: an
> ALU-step divergence downstream of a value correction is still flagged
> **CROSS-STEP** (not falsely certified), and the `[ALU: <block>]` context is
> surfaced on top of whatever confidence the decode guard assigns. The validation
> table below was re-measured against the **current** gate, not the pre-decode
> base; the original `06a1d12d` numbers (a now-retired per-byte confidence model)
> are not reproduced.

## The gap (before)

`tools/interp_oracle_gate.py` is the CPU-only interpreter-vs-oracle attribution
gate. For any program whose execution path hit one of the 4 still-imperative
composite ALU blocks —

| opcode      | block               |
|-------------|---------------------|
| ADD / SUB   | `AddSub5StageBlock` |
| MUL         | `FlattenedALUMul`   |
| DIV / MOD   | `FlattenedDivMod`   |
| SHL / SHR   | `ALUShiftComposite` |

— the gate classified the program **ALU-OPAQUE** and declined to judge it
(neither PASS nor FAIL). The justification was that these blocks have no
declarative IR rule form, so the faithful *pure-IR* forward could not run them.
Concretely the expr lane saw **17/25 `expr_mul_div` ALU-OPAQUE**, and the whole
`mul` / `div` / `mod` / `add` / `sub` corpus was un-testable on CPU.

## The fix

The gate already builds the **production** model (`alu_mode='efficient'`,
`trust_neural_alu=True` — the exact model `BatchedPureNeuralRunner` uses) and
its per-step decode runs through `faithful_full_forward`, which **already
executes the real composite ALU blocks** via `block.ffn(...)` on the residual
tape. The ALU result was therefore *already in the decode* — the gate was just
throwing it away with the ALU-OPAQUE short-circuit.

So the change is two-part:

1. **`faithful_interpreter.py`** — the ALU-block execution is now a first-class,
   documented capability of the interpreter module. New
   `run_faithful_blocks(model, tape)` walks the real model block-by-block:
   every attention block + every IR-executable `PureFFN` block runs through the
   faithful pure-IR math, and every composite ALU FFN block
   (`COMPOSITE_ALU_FFN`) runs through its **real baked `block.ffn(...)` forward**
   (a vanilla CPU forward — exact, but imperative). Verified **byte-identical**
   (`max logit diff = 0`, argmax equal) to the validated
   `tools/faithful_interpreter_validate.py:faithful_full_forward` it consolidates.
   The module docstring's "Faithful-coverage boundary" now describes ALU-block
   execution explicitly.

2. **`interp_oracle_gate.py`** — **ALU-OPAQUE is retired as a verdict.** A
   program that hits an ALU op now gets a real PASS / FAIL@step/byte verdict
   (the ALU executed, so the verdict is faithful). The `ALU_OPAQUE` constant is
   kept only for import compatibility; the gate never emits it. Attribution for
   an ALU-step divergence is necessarily **coarse**: the block is imperative
   (no declarative rule), so the gate reports
   `attributed_op=<ALU block>, attributed_rule=None` and tags the row
   `FAIL(ALU)` / `is_alu_step=True` — honest, and distinct from the IR-rule
   attributions on the non-ALU path.

`_ALU_OPAQUE_OPCODE` is renamed `_ALU_OPCODE_BLOCK` (opcode → owning imperative
block, used now for coarse attribution rather than opacity).

## Validation — both lanes survive (re-measured on the current gate)

`run_1096_canonical.py --criterion full_trace` (GPU, the authority) was run on
the 65 ALU programs spanning `mul` (100–119), `div` (150–169), `expr_mul_div`
(850–874), and the gate (CPU) was run on the same ids.

### (a) ALU is now testable — ALU-OPAQUE retired, the real block executes

| metric | before | after |
|--------|--------|-------|
| ALU-OPAQUE (gate declines, no verdict) | 65 / 65 | **0** |
| ALU programs whose verdict now rides the real baked ALU block | 0 | **50 / 65** |

Every ALU program now gets a real per-step `(PC, AX)` verdict (the
`FaithfulForwardCache` runs `block.ffn(...)` for the composite blocks — exactly
production's ALU). The ALU block is surfaced as `[ALU: <block>]` /
`is_alu_step=True`, and the DIV-step divergence IS detected — e.g.
`expr_mul_div_14` (`14*32/8`): the gate decodes steps 0–5 byte-exact (incl. the
MUL `ax=448`) and flags the DIV step (`step 6`, `ax 24` vs oracle `56`) tagged
`[ALU: DIV (FlattenedDivMod)]`, matching the neural authority's FAIL.

### (b) DECODE soundness preserved — 0 false-trusts vs neural

The decode lane's cross-step guard is unchanged and is the arbiter of confidence.
On the 65-program ALU set, neural FAILs 15 / PASSes 50; the gate flags **all 65
CROSS-STEP** (an upstream step-1 SP-byte-2 teacher-forcing correction poisons the
context for these expr/mul/div programs, so the sound guard correctly DEFERS them
to the GPU rather than certifying). The gate claims authority on **none** of the
65 → **0 false-trusts** (it never disagrees with neural where it claims to be
authoritative). This is the same conservative-but-sound behavior the decode lane
already has; the ONLY thing this change does for these programs is replace the
"no verdict (ALU-OPAQUE)" with a real CROSS-STEP verdict carrying the `[ALU: …]`
context.

Where the decode guard DOES grant high confidence, the gate is authoritative and
exact. On `edge_literal` (1031–1045): gate AUTHORITATIVE on **12/15** (10 AX-byte-1
FAILs + 2 PASS), **interp == neural 12/12 (100%)**, 0 false-trusts; the 3 it
defers are correctly handed to the GPU. The `var`/`func` framing-drift clusters
stay soundly **CROSS-STEP**, byte-identical to the pre-merge decode lane.

| lane | claim | result |
|------|-------|--------|
| DECODE | edge_literal AX-byte-1 HIGH-conf + attributed | 10/10 HIGH-conf FAIL, == neural |
| DECODE | var/func framing-drift flagged CROSS-STEP | byte-identical to pre-merge HEAD |
| DECODE | soundness | 0 false-trusts (edge_literal 12/12, ALU 0 authoritative) |
| ALU | ALU-OPAQUE retired | 0 ALU-OPAQUE; 50/65 ride the real ALU block |
| ALU | DIV-step divergence detected | `expr_mul_div_14` flags DIV step `[ALU: DivMod]` |

## Net

The ~50 `expr` + `mul`/`div` ALU programs are no longer un-testable: the real
imperative ALU block now executes inside the gate, so each gets a real per-step
verdict (PASS / FAIL / CROSS-STEP) with the owning ALU block surfaced. The decode
lane (re-anchoring + cross-step guard + AX-high-byte soundness) is fully intact
and byte-identical on non-ALU programs. No production weights changed; the model
is byte-identical (golden `ce9bf9616f3379c4`); `pytest`/smoke decode is untouched.
`tests/test_faithful_interpreter.py` 5/5 pass.
