# Declarative verifier audit (2026-05-12)

**Scope**: full-VM run of `verify_claims_static(strict_mode=True)` + `verify_produces_consumes_dynamic()` across every annotated op produced by `compile_full_vm()` (alu_mode="lookup"). Verifier landed in commit `819732b`; audit branch off `affea42` (origin/main).

**Status**: AUDIT ONLY — no production op annotations were modified. The 5 bug-fix agents are running in parallel for the smoke regression. This report only catalogues what the verifier sees today.

**Verifier still healthy**: `pytest c4_release/tests/test_declarative_verification.py` is 13 passed / 1 skipped in 66s. `TestProductionVerifier::test_no_declaration_drift_on_annotated_ops` (non-strict mode) is green because no op currently has `declared_but_not_written` cells.

## TL;DR

- **Mode A (strict)**: 28 ops with `claims` were verified. **0 ops have UNUSED CLAIMs** (the load-bearing direction). **25/28 ops have UNDECLARED WRITEs** (partial-claim convention residue — expected). 3 ops are exact-match clean.
- **Mode B (dynamic)**: 9 ops with `produces`/`consumes_fresh` declarations were probed via a synthetic 1-instruction trace. **1 op drifts**: `layer8_multibyte_routing` declares `produces={"OUTPUT_LO": "AX_byte0", "OUTPUT_HI": "AX_byte0"}` but the residual at the AX marker is zero after L8 in the probe.
- **Smoking gun candidate**: `layer8_multibyte_routing` Mode-B drift correlates with the basic ALU `byte-leak` failures + IMM-based smoke regressions. See Part 2.
- **Important caveat**: Mode B's synthetic probe has no CODE bytes / no actual IMM opcode in memory; the drift could be a probe limitation rather than a real bug. See "Caveats on Mode B" under Part 2.

Reproduce with:

```
python -m c4_release.debug_archive.audit_all_ops > /tmp/audit_output.txt 2>&1
```

(helper script: `c4_release/debug_archive/audit_all_ops.py`)

## Part 1 — Verifier output

### Mode A (strict_mode=True)

Wall: ~5s after disk cache warm. 28 ops verified; 3 exact-match-clean; 25 ops drift in strict mode purely from UNDECLARED WRITEs. **No UNUSED CLAIMs anywhere** — every cell every op declares is actually written.

#### Clean (exact match, no extra writes)

| Op | declared = observed |
|---|---|
| `layer0_threshold_attn` | 128 |
| `layer1_threshold_attn` | 68 |
| `layer2_threshold_attn` | 16 |

These three are unusual: they pre-bake the threshold-attention head with `_set_threshold_attn`, whose write set is exactly the registry-declared Q/K/V/O rows. They are the model for "what full-coverage claims look like".

#### Drift in strict mode (UNDECLARED WRITEs only — no UNUSED CLAIMs)

Format: `op | declared | observed | undeclared_writes`. All have `unused_decl=0` (i.e. every declared cell is actually written; the only delta is extra writes the op makes beyond its declared subset).

| Op | decl | obs | extra | Largest scope / sample |
|---|---|---|---|---|
| `layer1_ffn` | 5 | 68 | 63 | L1 ffn_W_up / ffn_W_gate / ffn_W_down |
| `layer2_mem_byte_flags` | 8 | 32 | 24 | L2 ffn_W_up / W_gate gating cells |
| `layer2_initial_pc_bake_cancel` | 2 | 6 | 4 | L2 ffn_W_up / W_gate for the two cancel units |
| `layer3_carry_forward_attn` | 192 | 460 | 268 | L3 attn_W_q (CONST/HAS_SE gates) + W_k anchors |
| `layer4_pc_relay` | 64 | 141 | 77 | L4 attn_W_q / W_k gating for pc-relay head |
| `layer4_sp_to_addr_key` | 48 | 114 | 66 | L4 attn_W_q / W_k gating + W_v scalar relays |
| `layer5_fetch` | 192 | 755 | 563 | L5 attn_W_q (every head's gating) |
| `layer7_operand_gather` | 64 | 141 | 77 | L7 attn_W_q (heads 0/1 gating) + W_o O-projection |
| `layer7_memory_heads` | 104 | 280 | 176 | L7 attn_W_o (heads 2-4 O-projection, ADDR_B0_LO outputs) |
| `layer8_sp_gather_bake` | 96 | 225 | 129 | L8 attn_W_o (heads 0-2 ADDR_B*_LO outputs) |
| `layer8_multibyte_fetch_bake` | 32 | 140 | 108 | L8 attn_W_q (head 3 FETCH_LO/HI gating) + W_o for AX_CARRY |
| `layer8_mem_to_alu` | 32 | 881 | 849 | L8 attn_W_q (head 5 huge gating fan-in) + W_k MEM_STORE gates |
| `layer8_op_imm_relay` | 2 | 11 | 9 | L8 attn_W_q/W_k head 4 OP_IMM gates |
| `layer9_lev_addr_relay` | 32 | 72 | 40 | L9 attn_W_o head 0 ADDR_B0_LO output |
| `layer9_lev_bp_to_pc_relay` | 32 | 72 | 40 | L9 attn_W_o head 1 ADDR_B0_LO output (BP→PC relay) |
| `layer10_carry_relay_bake` | 2 | 10 | 8 | L10 attn_W_q / W_k anchor + W_o for CARRY |
| `layer10_byte_passthrough_bake` | 32 | 87 | 55 | L10 attn_W_o head 1 ADDR_KEY + IO_* outputs |
| `layer10_sp_byte_passthrough_bake` | 32 | 89 | 57 | L10 attn_W_o head 2 ADDR_KEY + IO_* outputs |
| `layer10_psh_stack0_passthrough_bake` | 32 | 89 | 57 | L10 attn_W_o head 3 OUTPUT_LO/HI |
| `layer10_stack0_byte_relay_bake` | 32 | 84 | 52 | L10 attn_W_o head 4 ALU_LO |
| `layer13_mem_addr_gather` | 96 | 219 | 123 | L13 attn_W_o ADDR_B0_LO output |
| `layer14_mem_generation` | 256 | 873 | 617 | L14 ffn_W_up / W_gate for STACK0/MEM regen |
| `layer15_memory_lookup` | 128 | 3419 | 3291 | L15 attn_W_q every-token gating + W_k for ADDR_KEY |
| `phase_a_ffn` | 7 | 20 | 13 | L0 ffn_W_up gates |
| `function_call_weights` | 96 | 2342 | 2246 | L5/L6 attn O + L6 FFN routing (ENT/JSR pipeline) |

#### Intent-guess per op (one-liners)

Every UNDECLARED WRITE I sampled fell into one of these legitimate-looking buckets:

- **Q/K gating cells** (`attn_W_q`, `attn_W_k`): activation-conditional gates (`MARK_AX`, `CONST`, `IS_BYTE`, `HAS_SE`) that select the rows the V/O projection acts on. Not in claims because they're "wiring", not "ownership".
- **O-projection** (`attn_W_o`): for relay/gather heads, ALL 16 W_o output positions are written for each (head, slot) pair; claims only list the V rows for the relay (e.g. `CLEAN_EMBED_LO+k → ADDR_B0_LO+k`). The full byte-wide W_o output is undeclared.
- **Anti-collision negative biases** (rows 33+ of each head): per-head suppression gates that fire only under specific opcode masks. Almost always written but rarely claimed.
- **FFN routing gates** in `function_call_weights`: ENT/JSR pipeline writes thousands of W_up/W_gate/W_down cells. Annotations only cover the 96 V-relay rows on L5/L6 attn (the high-collision-risk subset).

**Verdict on Part 1**: Mode A in strict mode produces a noisy but useful signal. The current set of annotations is *correct in the load-bearing direction* — every declared cell is written, none has gone stale. The "drift" is entirely the expected partial-claim residue called out in `DECLARATIVE_VERIFICATION.md`. No DEBUGGING-PRIORITY findings from Mode A.

### Mode B (dynamic produces/consumes)

Synthetic probe: `_build_synthetic_imm_exit` — register markers only, no CODE bytes / no IMM opcode in memory.

| Candidate (op with produces/consumes_fresh) | drift? | notes |
|---|---|---|
| `layer7_operand_gather` (produces ALU_LO, ALU_HI @ AX_byte0) | OK | residual at AX is non-zero in both dims |
| `layer8_alu` (consumes_fresh AX_CARRY_LO, ALU_LO) | OK | consumes-only ops are not checked for liveness |
| `layer8_multibyte_routing` (produces OUTPUT_LO, OUTPUT_HI @ AX_byte0) | **DRIFT** | OUTPUT_LO/HI both zero at AX marker after L8 |
| `layer8_head6_ax_carry_refresh` (produces AX_CARRY_LO, AX_CARRY_HI) | OK | inert by default (enable=False); produces honored on the inert path because flag-gated bake is treated as no-op |
| `layer9_alu` (consumes_fresh ALU_HI) | OK | consumes-only |
| `layer10_*` (consumes_fresh) | OK | consumes-only |
| `layer11_alu_mul_*` (consumes_fresh) | OK | consumes-only |
| `layer12_alu_mul_*` (consumes_fresh) | OK | consumes-only |
| `layer13_*` (consumes_fresh) | OK | consumes-only |

**Mode B drift: 1/9**: `layer8_multibyte_routing` — declares it writes `OUTPUT_LO`/`OUTPUT_HI` at AX byte positions for IMM, but the post-L8 residual at the AX marker is zero in those slots in the synthetic probe.

## Part 2 — Cross-reference with smoke failures

Current main: **34 failed / 8 passed / 2 errors**. Failing categories per the task prompt:

| Category | Verifier finding | Priority |
|---|---|---|
| Basic ALU (add/sub/mul/div/mod) byte-leak | `layer8_multibyte_routing` Mode-B drift directly impacts IMM-driven AX byte writes. ALU ops depend on `layer7_operand_gather` → `layer8_alu` → `layer8_multibyte_routing` chain for correct OUTPUT byte placement. | **HIGH** — see caveat below |
| Memory store/load round-trip → 0 | No verifier findings. `layer8_sp_gather_bake`, `layer8_multibyte_fetch_bake`, `layer8_mem_to_alu`, `layer14_mem_generation`, `layer15_memory_lookup` all have UNDECLARED WRITEs but zero UNUSED CLAIMs and no Mode-B drift (these ops have no `produces` annotation). | Low — verifier cannot see this. Annotate `layer14_mem_generation.produces={"STACK0_*": "...", "MEM": "..."}` and re-run Mode B. |
| Bitwise and/or/xor | No verifier findings — L10 `andorxor` post-ops are unannotated (no claims, no produces). | Low — verifier cannot see this. |
| Control flow jmp_forward/bnz_branch | No verifier findings. L3 PC-update + L6 BZ/BNZ relay are unannotated. | Low — verifier cannot see this. |

### Caveats on Mode B

The `layer8_multibyte_routing` drift may be a **probe limitation**, not a real bug:

1. `_build_synthetic_imm_exit` builds a token stream that contains register markers (`REG_AX`, `REG_PC`, etc.) but **no `CODE_*` bytes and no actual IMM opcode word in memory**. Without a CODE byte at the PC address, the L5 fetch heads have nothing to attend to, so `OPCODE_BYTE_LO/HI` never receive an IMM tag, so the IMM-gated `layer8_multibyte_routing` would correctly write zero.
2. Compare to `layer7_operand_gather`, which produces `ALU_LO/HI` from prev STACK0 byte 0 — STACK0 *is* in the probe (zeros), so the op fires (with zero values) and the dim is non-zero only because of the gating cells contributing residuals.

**Net**: Mode-B drift on `layer8_multibyte_routing` is **suggestive but not conclusive**. A stronger probe would need to inject a CODE byte sequence containing `OP_IMM` so the IMM gate triggers. That is out of scope for this audit (it would mean modifying `decl_verifier.py`'s probe, which the task prohibits except for blocking bugs — and this is informational drift, not blocking).

### Conclusion for Part 2

**No DEBUGGING-PRIORITY smoking guns** from the verifier in its current form. The annotation coverage is too sparse on the failing test paths (L7/L8/L9/L10 ALU, L3/L6 control flow, L14/L15 memory) to surface the basic-ALU / mem-roundtrip / bitwise / jmp bugs. The `layer8_multibyte_routing` Mode-B drift is the only signal, and it's gated on a probe limitation.

The 5 parallel bug-fix agents have to work without verifier signal until annotation coverage extends to cover the failing paths.

## Part 3 — Annotation coverage recommendations

Prioritised by smoke-failure impact and effort. "Effort" measured in op-attributes-to-add and (qualitatively) hours to understand the bake.

### Tier 1 — direct impact on failing ALU / memory / control-flow paths

| Op | Add | Effort | Why |
|---|---|---|---|
| `layer3_pc_update` (or equivalent in `l3_ops.py`) | `claims` + `produces={"OUTPUT_LO": "PC_byte0", "OUTPUT_HI": "PC_byte0"}` | 15 lines | Control-flow jmp/bnz failures point here. No claims today. |
| `layer6_bz_bnz_relay` (in `l6_ops.py` or `l4_ops.py`) | `claims` + `produces={"OUTPUT_LO/HI": "PC_byte0"}` | 15 lines | Direct BNZ failure. |
| `layer8_alu` | `produces={"OUTPUT_LO": "AX_byte0", "CARRY": "..."}` | 5 lines | Pairs with the existing `layer8_multibyte_routing` Mode-B signal to triangulate the byte-leak. Op already has `consumes_fresh`; missing `produces`. |
| `layer9_alu` | `produces={"OUTPUT_HI": "AX_byte0", "CMP": "..."}` | 5 lines | Same reason as `layer8_alu`. |
| `layer14_mem_generation` | `produces={"STACK0_BYTE0..3": "...", "MEM_VAL_B0..3": "..."}` | 10 lines | Mem store/load roundtrip → 0 failure. Op already has 256 claims; produces would catch ANY zeroing in the residual write. |
| `layer15_memory_lookup` | `produces={"MEM_VAL_B0..3": "..."}` | 5 lines | Load side of mem roundtrip. |
| `layer10_carry_relay_bake` | `produces={"CARRY": "AX_byte0"}` | 3 lines | Carry is the load-bearing intermediate between L8/L9 ALU stages. |

### Tier 2 — fill coverage on existing high-write ops

| Op | Add | Effort | Why |
|---|---|---|---|
| `layer10_byte_passthrough_bake` / `_sp_byte_passthrough_bake` / `_psh_stack0_passthrough_bake` / `_stack0_byte_relay_bake` | `produces={"OUTPUT_LO/HI": "AX_byte0"}` or `produces={"ALU_LO/HI": "..."}` per head | 5 lines each | All four are L10 passthrough heads writing the OUTPUT/ALU planes; they're 99% similar but each has subtly different produces. |
| `layer10_andorxor_post_op` (the bitwise post-ops) | full `claims` + `produces` | 20 lines (per op) | Bitwise failures are wholly invisible to the verifier today. |
| L11/L12/L13 ALU MUL/SHIFT post-ops | `claims` for the schoolbook/carry-pass weights | 20 lines each | MUL is failing in smoke; these ops have NO claims today. |

### Tier 3 — defensive (clean-up partial claims toward strict-mode-clean)

Reserved for after the smoke is green. Many ops could be promoted from partial-claim to full-coverage claims (matching the `layer*_threshold_attn` pattern):

- `layer5_fetch` — claims 192 of 755 cells. The remaining 563 are gating cells; promoting to full coverage would tighten strict-mode.
- `layer8_mem_to_alu` — 32 of 881 cells claimed. The huge gating cell count makes this the highest-cost op to promote, but it's also the riskiest in collision terms.

### Total annotation effort

Tier 1: ~70 lines across 7 ops. **Worth doing now** — these are the ops gated by the active smoke failures.

Tier 2: ~100 lines. Worth doing as a follow-up.

Tier 3: pure hygiene; defer.

## Part 4 — CI gate proposal

### Current state

- `c4_release/tests/test_declarative_verification.py::TestProductionVerifier::test_no_declaration_drift_on_annotated_ops` runs `verify_claims_static()` (non-strict) on the full op set every test session. Wall: ~66s (dominated by `compile_full_vm`).
- Mode B is opt-in via `C4_VERIFY_DECLARATIONS=1`.

### Recommendation: keep non-strict in default test gate; add strict-mode and Mode B as opt-in lanes

| Lane | Mode | Severity on failure | When to run |
|---|---|---|---|
| Default `pytest c4_release/tests/test_declarative_verification.py` | A (non-strict): UNUSED CLAIMs only | **FAIL** | Every CI run. Wall already ~66s; no change. |
| New `pytest -m verifier_strict` (opt-in) | A (strict): UNUSED CLAIMs + UNDECLARED WRITEs | **WARN** today; **FAIL** once Tier 1+2 annotation expansion lands | Nightly + pre-release. |
| New `pytest -m verifier_dynamic` (opt-in) | B: produces liveness | **WARN** today (probe limitations); **FAIL** once the probe is upgraded to inject CODE bytes | Nightly. |

**Do NOT** wire the verifier into the smoke fixture as a pre-check. Reasons:

1. Smoke fixture wall is already a sore spot; +66s per fixture is too costly.
2. The verifier runs the same `compile_full_vm()` the smoke uses; bugs that affect compilation would fail both in parallel anyway.
3. Verifier is a static contract check; smoke is a runtime behaviour check. They are orthogonal — one shouldn't gate the other.

**Strict-mode FAIL switch**: do not flip strict-mode to FAIL on UNDECLARED WRITEs until Tier 2 annotation coverage lands. Today 25/28 ops would fail strict mode purely from the partial-claim convention, which is *not* a bug. Strict mode is most useful as a warn-only nightly diagnostic until the registry is full-coverage.

### Suggested concrete change set (out of scope for this audit, listed for the followup PR)

1. In `c4_release/pyproject.toml` (the pytest config), register `verifier_strict` and `verifier_dynamic` markers.
2. In `test_declarative_verification.py`, add two new test classes guarded by `@pytest.mark.verifier_strict` and `@pytest.mark.verifier_dynamic` that call `verify_claims_static(strict_mode=True)` and `verify_produces_consumes_dynamic()` respectively, both emitting `pytest.warns` rather than asserting until coverage is complete.
3. Optionally: upgrade `_build_synthetic_imm_exit` to inject a 4-byte CODE word containing `OP_IMM` so Mode B has a fighting chance on opcode-gated produces. **Out of scope for this audit.**

## Appendix — raw audit output

Full dump: re-run `python -m c4_release.debug_archive.audit_all_ops`. Helper script source: `c4_release/debug_archive/audit_all_ops.py`.

Key Mode B drift line:

```
OP layer8_multibyte_routing
  DRIFT: produces 'OUTPUT_LO': residual zero at AX marker (op may not have fired)
  DRIFT: produces 'OUTPUT_HI': residual zero at AX marker (op may not have fired)
```

Files referenced in this audit:

- Verifier: `c4_release/neural_vm/unified_compiler/decl_verifier.py`
- Verifier tests: `c4_release/tests/test_declarative_verification.py`
- Verifier design doc: `c4_release/docs/DECLARATIVE_VERIFICATION.md`
- Op definitions with `claims`: `c4_release/neural_vm/unified_compiler/ops/{l0,l1,l2,l3,l4,l5,l7,l8,l9,l10,l13,l14,l15,model}_ops.py`
- Ops with `produces`/`consumes_fresh`: `l7_ops.py` (operand_gather), `l8_ops.py` (alu, multibyte_routing, head6_ax_carry_refresh), `l9_ops.py` (alu), `l10_ops.py`, `l11_ops.py`, `l12_ops.py`, `l13_ops.py`
- Drift candidate intent (where to start fixing): `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:180-215` (`make_layer8_multibyte_routing_op`)
