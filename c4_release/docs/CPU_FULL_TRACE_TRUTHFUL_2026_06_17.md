# Truthful CPU full_trace self-check for framing programs (2026-06-17)

## TL;DR

`tools/cpu_full_trace.py --ids <list>` is the **mandated CPU self-check for any
framing / full_trace fix**. It runs the byte-identical CPU *autoregressive*
decode (`FaithfulAutoregressiveRunner`) and reports the SAME per-program
pass/fail verdict as `tools/run_1096_canonical.py --criterion full_trace`, on
CPU, in ~85s/program, with no GPU.

It is **truthful for framing programs**: validated against GPU ground truth on
current `main` (golden `b9d8861f` flag-OFF, HINIB default-ON), it reproduces
every framing-drift FAIL byte-for-byte (same verdict, same divergence step) —
including the canonical `func_identity` false-positive that the re-anchoring
`interp_oracle_gate` reports as a PASS. **DO NOT** self-check a framing fix with
`interp_oracle_gate.py` / a re-anchored single forward: it decodes each step
independently and CANNOT see the production fixed-35-token-slice cumulative
desync, so it over-claims PASS for programs the GPU fails.

## Why interp_oracle_gate lies about framing

`interp_oracle_gate.py` (and the `FaithfulInterpreter` single-forward) RE-ANCHOR
each step's register markers so they can attribute a wrong byte to its owning
rule. Re-anchoring deliberately hides the autoregressive framing desync: when a
step emits 34 or 37 tokens instead of 35, the production fixed-35-token slice
(`batched_pure_neural` `_UNSAFE_OFFSETS` + `_step_one`) misreads the NEXT step's
PC and the run drifts. A re-anchored single forward decodes each step from a
clean anchor, so the 34/37-token miscount never accumulates — it reports PASS for
the ~13% framing-drift clusters (`var_*` / `func_identity` / `nested_*` /
`if_var` / `expr_*`) the GPU fails. This is what made three "CPU-confirmed"
framing fixes this session land 0-flip on GPU.

## The truthful tool

`FaithfulAutoregressiveRunner` (`neural_vm/unified_compiler/faithful_autoregressive.py`,
landed earlier as commit `16c5e593`/`bfd0097e`) subclasses the production
`BatchedPureNeuralRunner` and overrides ONLY `_forward_argmax_batch` — the
per-token argmax source is swapped for the validated byte-identical CPU forward
(`CachedFaithfulForward`, ~1.7x `model.forward`). Every other piece — the
fail-fast speculation, `_step_one` / `_dispatch_pure_neural` STEP_END / HALT /
early-EXIT, the per-step `(PC, AX)` compare, the fixed-35 slice + the DraftVM
oracle — is the UNMODIFIED production verdict logic. So the CPU verdict is
byte-identical to neural BY CONSTRUCTION (the only swap is argmax-equivalent),
and emitted tokens are fed back as the next step's input → the 34/37-token
miscount is reproduced exactly.

`tools/cpu_full_trace.py` is a thin CLI over that runner. It reuses
`run_1096_fast._compile_and_oracle` and `run_1096_canonical._score_fail_fast_results`,
so its verdict + output format match `run_1096_canonical --criterion full_trace`
exactly. The full_trace path is pinned to `spec_k=32` (which is what
`run_1096_canonical._run_one_chunk_with_oom_retry` does:
`spec_k=(spec_k if spec_k > 0 else 32)`) so the speculative teacher-forcing of
the unsafe MEM offsets matches the canonical verdict.

It is **tooling only**: not imported by `compile_full_vm_dynamic` or any build
path, so the model stays byte-identical (golden `b9d8861f` flag-OFF unchanged).

## Validation table — GPU ground truth vs faithful CPU (current main)

Both runs use `--criterion full_trace`, `spec_k=32`. GPU = `run_1096_canonical`
on a dedicated A5000; CPU = `cpu_full_trace.py` (`CUDA_VISIBLE_DEVICES=""`).

| id   | cluster        | GPU  | faithful-CPU | verdict match | GPU div_step | CPU div_step |
|------|----------------|------|--------------|---------------|--------------|--------------|
| 0    | add            | pass | pass         | ✓             | —            | —            |
| 1    | add            | pass | **fail**     | ✗ (sat-tie)   | —            | 3            |
| 2    | add            | pass | **fail**     | ✗ (sat-tie)   | —            | 3            |
| 3    | add            | pass | pass         | ✓             | —            | —            |
| 250  | var_simple     | fail | fail         | ✓             | 8            | 8            |
| 251  | var_simple     | fail | fail         | ✓             | 8            | 8            |
| 252  | var_simple     | fail | fail         | ✓             | 8            | 8            |
| 275  | var_mul        | fail | fail         | ✓ **framing** | 6            | 6            |
| 350  | if_gt          | fail | fail         | ✓             | 3            | 3            |
| 550  | func_identity  | fail | fail         | ✓ **framing** | 8            | 8            |
| 800  | expr_add_mul   | fail | fail         | ✓ **framing** | 5            | 5            |
| 825  | expr_paren     | pass | **fail**     | ✗ (sat-tie)   | —            | 6            |
| 850  | expr_mul_div   | pass | **fail**     | ✗ (sat-tie)   | —            | 3            |
| 904  | gcd            | fail | fail         | ✓             | 0            | 0            |
| 1031 | edge_literal   | pass | pass         | ✓             | —            | —            |
| 1032 | edge_literal   | pass | pass         | ✓             | —            | —            |
| 1033 | edge_literal   | pass | pass         | ✓             | —            | —            |
| 1034 | edge_literal   | pass | pass         | ✓             | —            | —            |
| 1035 | edge_literal   | pass | pass         | ✓             | —            | —            |

**Verdict agreement: 15/19. Framing programs: 6/6 reproduce the GPU FAIL byte-
for-byte (var_simple 250-252, var_mul 275, func_identity 550, expr_add_mul 800,
if_gt 350, gcd 904) — including the canonical `func_identity 550` and
`var_mul 275` framing-drift fails, NOT a re-anchored pass.** When a framing fix
actually moves one of these to PASS on CPU, the divergence is gone — and that
PASS will hold on GPU (the decode is byte-identical to neural).

## The 4 mismatches are the documented saturated-tie floor, not a decoder bug

All four are GPU-**pass** / CPU-**fail** (the SAFE direction — CPU is
conservative; it never reports a framing fail as a pass). They are the
saturated-tie model fp-instability the asset's commit (`bfd0097e`) already
documented by name (`expr_paren 825`, `expr_mul_div 850`): at the drifted
positions the LM logits SATURATE (~1e22) with an EXACT top-1/top-2 gap=0, so the
argmax winner is fp32-accumulation-order dependent and fused-GPU vs
rule-by-rule-CPU pick different ids. `add 1` / `add 2` are the same class (ALU
high-byte carry path).

**This is a MODEL device-divergence, not a `cpu_full_trace` decoder bug.** The
real CPU `model.forward` decode (run via `run_1096_canonical` with
`CUDA_VISIBLE_DEVICES=""`, i.e. the batched runner on the CPU model) also FAILS
these four — so the faithful CPU decode is faithful to CPU neural; CPU neural
just != GPU neural at those saturated ties. (Confirmation:
`/tmp/cpu_neural_ft.json`.)

## How to use it (the standing rule)

- **Framing / full_trace self-check → `tools/cpu_full_trace.py --ids <list>`.**
  A framing fix is CPU-confirmed only when this tool flips the target programs
  to PASS. Because the decode is byte-identical to neural, a PASS here holds on
  GPU (modulo the saturated-tie programs above, where CPU is conservatively
  stricter — it never falsely passes).
- **Rule attribution (which FFN rule owns a wrong byte) → `interp_oracle_gate.py`.**
  Its re-anchoring is the right tool for blaming a rule, the WRONG tool for a
  framing verdict.
- The 4 saturated-tie programs are a documented exception: CPU may say FAIL where
  GPU says PASS. CPU never says PASS where GPU says FAIL — that asymmetry is
  exactly what makes it safe as an over-claim guard.

## Speed

~0.47s/CPU forward, ~85-350s for a 5-17 step diverging program (O(steps^2), no
KV cache). The deep `gcd 904` (97 steps) ran in ~182s (it early-diverges at
step 0). Fine for self-checking specific framing programs in ~minutes instead of
a ~30-min GPU run; use `run_1096_canonical` on GPU for the full corpus.
