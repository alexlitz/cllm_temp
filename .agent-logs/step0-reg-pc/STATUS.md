# B3-α STATUS: step0:REG_PC rule investigation

## Outcome: DOCUMENTED (no patch landed)

The brief asked us to "hunt the unidentified step0:REG_PC rule" emitting
`0xE0` at slot REG_PC with `argmax_logit=+4.36e11` for ids 425-449
(if_var) and 550-574 (func_identity).

After resetting the worktree to base `4d069f7` and reproducing on id 425,
the dominant 0xE0 emitter is no longer pointing at step0:REG_PC.  The
audit + diagnostic now report the failure one step deeper, at
`step1:MEM_addr0` (or `step5:MEM_addr0`).  The single L10 declarative
FFN rule responsible for the massive 0xE0 head logit has been identified.

## Offending rule

`tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority`
in `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:3888-3922`
(added in commit `55e7372` — "Merge focused 1096 stack lookup fixes").

```
FFNRule.constant_write(
    name="tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority",
    conditions=(
        ("MARK_MEM", 1.0), ("HAS_SE", 1.0), ("H1+4", 10.0),
        ("H1+1", -1e6), ("H1+2", -1e6), ("H1+3", -1e6), ("H1+10", -1e6),
        ("MEM_STORE", 5.0), ("MEM_ADDR_SRC", -1e6),
        ("PSH_AT_SP", 1.0), ("OP_JSR", -1e6),
        ("CMP+0", 10.0),
        ("ALU_LO+8", 5.0), ("ALU_LO+7", -10.0), ("ALU_LO+10", -20.0),
        ("IS_BYTE", -100.0),
        ("MARK_AX", -1e6), ("MARK_PC", -100.0), ("MARK_SP", -100.0),
        ("MARK_BP", -100.0), ("MARK_STACK0", -100.0),
        ("NEXT_PC", -1e6), ("NEXT_AX", -1e6), ("NEXT_SP", -1e6),
        ("NEXT_BP", -1e6), ("NEXT_STACK0", -1e6), ("NEXT_MEM", -1e6),
        ("NEXT_SE", -1e6),
    ),
    threshold=40.0,
    writes=byte_writes(0xE0, strength=5_000_000_000.0),
)
```

`byte_writes(0xE0, strength=5e9)` expands to one-hot:
- `OUTPUT_LO[0]   = +5e9` and competitors `OUTPUT_LO[k!=0]   = -5e9`
- `OUTPUT_HI[14]  = +5e9` and competitors `OUTPUT_HI[k!=14] = -5e9`

When the hidden activation is ~2e5 (a 5-condition match with weights
summing into the hundreds of thousands), output dimensions reach
`±1e15`.  That matches the diagnostic dump:
```
OUT_HI[14]=+1.01e15  OUT_HI[15]=-1.01e15
expected=0xf0 argmax=0xe0  argmax_logit=+1.05e16  margin=-1.01e16
```

## Evidence (id 425, `if_var_0: x=96, x>59`, base `4d069f7`)

- **Teacher-forced lowering audit** (`lowering_id425.log`):
  - `first_fatal` = `step9:STACK0_byte0 abs=484 expected=0x60 argmax=0x00 margin=-1.00` (register-result fatal — independent of the 0xE0 rule but a downstream symptom)
  - `first_info` = `step5:MEM_addr0 abs=349 expected=0xe8 argmax=0x00 margin=-2.60e11` (the 0x00 emission here comes from a sibling rule `tail_mem_store_addr0_00_from_global_exact` at l10_ops.py:3695 with `strength=1_000_000_000`)
- **Declarative diagnostic** (`diag_id425.log`):
  - `first_token_divergence = step1:MEM_addr0 abs=209 expected=0xf0 neural=0xe0`
  - `argmax_logit = +1.05e16, expected_logit = +4.4e14, margin = -1.01e16`
  - `OUT_HI[14] = +1.01e15` (the active 0xE0 hi-nibble), `OUT_HI[15] = -1.01e15` (the suppressed 0xF0 hi-nibble)

The brief's exact `step0:REG_PC abs=148 argmax_logit=+4.36e11` is not
reproducible against current head.  Either the configuration that
produced that observation differed (different bake/env) or an intervening
commit moved the first divergence.  The same 5e9 rule is the
mechanically dominant 0xE0 emitter at any MEM-store position where its
gates lift.

## Why this rule misfires for `if_var_0`

`int x; x = 96` compiles to:
- `ENT 1` -- allocate 1 local (8 bytes), `BP -> BP-8`
- `LEA -1` -- push address `BP-8 == 0xFFFFFFF8` (or similar offset)
- `IMM 96`
- `SI` -- store AX at the top-of-stack address

The `SI` step is `PSH_AT_SP=1`, `MEM_STORE=1`, `MEM_ADDR_SRC=0` (the
address is read from the stack-top, not from an authority dim).
The expected `MEM_addr0` byte for the `BP-16` slot is `0xF0`, but the
rule asserts `0xE0` whenever its activation gates clear `threshold=40`.

The discriminating gate, `ALU_LO+8 = +5`, is meant to fire when the
ALU low nibble equals 8 (i.e. `LEA`-computed offset low nibble 8 ->
address `..F8`).  In id 425's step the same dimension can still rise
high enough — combined with `CMP+0=10`, `H1+4=10`, `MEM_STORE=5`,
`PSH_AT_SP=1` — to exceed threshold 40.  Because the write is
`byte_writes(0xE0, strength=5e9)` regardless of the actual offset, the
rule emits a constant 0xE0 even when the legitimate offset byte is
`0xF8`, `0xF0`, `0xE8`, etc.

This same misfire is the mechanism behind the
`tail_mem_store_addr0_00_from_global_exact` issue at step5 (where 0x00
beats 0xE8 with strength 1e9 -> hidden ~260 -> margin -2.6e11).

## Why a single-line patch is unsafe

- Loosening `strength` from 5e9 to ~5e6 removes the dominance for if_var
  but breaks the cases where the rule was deliberately added (commit
  55e7372 introduced it to fix specific local-store contexts).  There
  is no unit-test for this rule (`grep -r tail_mem_store_addr0_e0_from_psh_sp`
  only finds the definition).
- Adding `MARK_PC = -1e6` (matching the `MARK_AX` blocker pattern) does
  not help the actual failure: at `MEM_addr0` the gate `MARK_MEM=1`
  is already active, so MARK_PC=0 there.
- Adding an `OUTPUT_HI+15 = -1.0` condition (analogous to the `0x00`
  rule's negative-output evidence guard) is the cleanest discriminator,
  but the rule needs all four sibling rules
  (`tail_mem_store_addr0_{e8,f0,f8,00,e0...}_*`) revisited together to
  set up a coherent precedence.  Left as follow-up.

## Recommended follow-up

1. Add `OUTPUT_HI+15 = -1.0` (or similar non-0xE0 evidence dimensions)
   to the activation conditions of
   `tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority` so the
   rule self-suppresses once the 0xF0 rule (`tail_mem_store_addr0_f0_exact`
   at strength 1e6) has any positive evidence.
2. Reduce `strength=5_000_000_000.0` to `strength=1_000_000.0` to match
   the sibling 0xF0 rule's scale.  All `tail_mem_store_addr0_*` writes
   should sit at a comparable order of magnitude so the most-evidence
   rule wins; the current 5000x gap forces 0xE0 to dominate whenever it
   crosses threshold.
3. Add a focused unit test that pins the post-FFN OUTPUT_LO/HI values at
   the `if_var_0` `SI` row (id 425 step1 MEM_addr0) so the next person
   editing these rules sees regressions immediately.
4. Apply the same audit to `tail_mem_store_addr0_00_from_global_exact`
   (strength 1e9) — confirmed second offender, manifests at step5
   MEM_addr0 of id 425.

## Files inspected

- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py` (rules 3635-3990)
- `c4_release/neural_vm/unified_compiler/decl_verifier.py` (lowering audit)
- `c4_release/neural_vm/unified_compiler/band_guarantees.py`
  (`expected_byte_guarantee_rules`)
- `c4_release/neural_vm/unified_compiler/primitives.py`
  (`Primitives.byte_value_writes`)
- `c4_release/neural_vm/unified_compiler/ir.py` (`FFNRule.constant_write`)

## Reproduction commands

```bash
# Teacher-forced lowering audit
C4_1096_LOWERING_AUDIT=1 C4_1096_OFFSET=425 C4_1096_LIMIT=1 \
  C4_1096_LOWERING_ASSERT_MODE=off C4_1096_LOWERING_PRINT_MODE=drift \
  C4_1096_LOWERING_MIN_MARGIN=0.5 C4_1096_LOWERING_OUTPUT_BAND_MIN_MARGIN=0.5 \
  C4_1096_LOWERING_MAX_TRACE_TOKENS=20000 C4_DECLARATIONS_ONLY_BAKE=1 \
  PYTHONPATH=.:c4_release PYTHONUNBUFFERED=1 \
  python -u -m pytest -q -s \
  c4_release/tests/test_1096_teacher_forced_lowering_audit.py::test_1096_teacher_forced_lowering_audit_slice --tb=short

# Final-output declarative diagnostic
C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 C4_1096_OFFSET=425 C4_1096_LIMIT=1 \
  C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K=0 C4_BATCH_USE_KV_CACHE=0 \
  C4_BATCH_CHUNK=8 C4_1096_PROGRESS=1 \
  PYTHONPATH=.:c4_release PYTHONUNBUFFERED=1 \
  python -u -m pytest -q -s \
  c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice --tb=short
```
