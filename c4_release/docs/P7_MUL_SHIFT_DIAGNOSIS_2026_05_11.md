# Phase 7 MUL + SHL/SHR pure_neural diagnosis (2026-05-11)

## Scope

Task: diagnose + fix the 5 MUL xfails + 4 SHL xfails + 3 SHR xfails in
`c4_release/tests/test_pure_neural_heap_div.py`, under the hypothesis that
they share a root cause with the DivMod hard-fails (byte-broadcast pattern
at OUTPUT_LO/HI bytes 1-3).

## Result

**Root cause is NOT MUL/SHL/SHR-specific. It is a Phase 1 / Phase 2
foundational issue currently being worked on in branch `d3-phase1-multi-imm`
(commit `ec59c6e`).**

All 12 MUL/SHL/SHR xfails fail for the same upstream reason: the L6 IMM
routing FFN leaks the IMM byte 0 value into AX bytes 1-3 of the residual
stream at the AX marker, then `CarryPropagationPostOp` and the L14 cleanup
amplify that leak through SUB-branch carry-prop units, producing the
byte-broadcast `0xVVVVVVxx` pattern at EXIT.

## Evidence

### MUL test results (after `--runxfail`)

| Test | Got | Expected | Notes |
|------|-----|----------|-------|
| `test_mul_small[3-4-12]`    | `0`           | 12  | byte 0 wiped |
| `test_mul_small[5-5-25]`    | `0`           | 25  | byte 0 wiped |
| `test_mul_small[10-10-100]` | `0`           | 100 | byte 0 wiped |
| `test_mul_small[255-1-255]` | `0xfafafaff`  | 255 | byte 0 correct, bytes 1-3 = 0xFA |
| `test_mul_small[16-16-256]` | `0`           | 256 | wiped |

Pattern: for results that fit in byte 0 (255), the MUL itself produces a
correct byte 0 but the EXIT-step output emits `0xFA` for bytes 1-3 (= a
broadcast of an IMM-value-derived residual, not the MUL result).
For all other cases the output is 0 (the model never makes it past PSH's
step-framing breakdown — see Section "Step-framing breakdown" below).

### Single-instruction IMM also reproduces the byte-broadcast bug

`test_pure_neural_pc.py::TestPureNeuralSingleInstruction::test_imm_byte_values`
parametrizations: `[1, 7, 42, 100, 200]` PASS but `[255]` FAILS with
`got 4210752255 (0xfafafaff)` vs expected `255`. This eliminates MUL/PSH/SHL
as the cause — even a single-instruction `IMM 255, EXIT` program reproduces
the byte-broadcast in bytes 1-3 of AX at the EXIT step.

The boundary between PASS (IMM 200) and FAIL (IMM 255) is approximately
when the IMM value crosses ~210 — at which point the L10 ALU attention's
uniform `-217` residual on `CARRY+1/CARRY+2/OUTPUT_LO/HI` at `REG_AX_mark`
flips a `CarryPropagationPostOp` SUB-branch unit's sign and starts firing
spuriously. This matches `d3-phase1-multi-imm`'s diagnosis exactly.

### Step-framing breakdown for multi-instruction programs

For `IMM(3), PSH, IMM(4), MUL, EXIT`, the model's emitted 35-token-per-step
framing collapses after PSH:

```
step0 (IMM 3):  35 tokens, well-formed.  AX=[3,0,0,0] (correct)
step1 (PSH):    starts well (PC=18, AX=3), but then partway through the
                step the model emits another REG_PC + AX + REG_SP burst —
                squishing what should be the next instruction's tokens into
                step 1's frame. Final AX in step 1 window: [4, 0, 3, 248].
step2/3+:       degenerates into a stream of byte 10s.
```

The L10 head-1 byte-passthrough chain (which copies prev step's AX bytes
1-3 forward) is supposed to be SUPPRESSED during IMM and during binary-ALU
steps, but the suppression coefficients evidently are not strong enough
to overcome the L6 IMM-bake residual leak.

This is the same Phase 2 root cause documented in
`test_pure_neural_psh_add.py`'s xfail decorators (`_PHASE2_PSH_XFAIL_REASON`)
and elaborated in `PURE_NEURAL_GAP_ANALYSIS.md` Tier 1/Tier 2.

## Why this surfaces as "byte-broadcast" in DivMod (P7 agent sample)

The earlier P7 agent sample reported `0x06060606` and `0x13131315` for DIV
20/5 (expected 4) and MOD 7/3 (expected 1). Those same byte-broadcast
patterns are this same L6 IMM leak + CarryPropagationPostOp amplification,
not a per-ALU-composite "missing reduction" bug. The MUL pipeline's GE→BD
converter and the SHL/SHR composite's `ShiftGEToBDStage` BOTH only write
OUTPUT at byte 0 (gated on MARK_AX). The byte 1-3 broadcast comes from
upstream (L6 IMM bake + L10 ALU attn residuals + CarryProp amplification),
not from the ALU composite itself.

## Why fix doesn't belong in p7-mul-shift

The 12 xfailed MUL/SHL/SHR tests will NOT pass by fixing anything in the
`FlattenedALUMul` / `ALUShiftComposite` composites — those produce correct
byte-0 output today. The fix must land in:

1. `_set_layer6_routing_ffn`'s IMM units (further strengthen
   `BD.IS_BYTE = -S * N` suppression at byte 1-3 positions)
2. `CarryPropagationPostOp.forward` (clamp negative residuals; the
   d3-phase1-multi-imm clamp lands ~one bug)
3. `_set_layer14_clear_output_corruption` (extend to clear OLO/OHI at
   non-byte-0 AX positions during IMM)

All three live OUTSIDE the MUL/SHL/SHR pipeline. The flattened ALU
composites are already correct.

## Recommendation

1. Do NOT remove the xfail decorators on these tests in this branch. They
   are correctly marked.
2. Track the fix work on `d3-phase1-multi-imm` (currently 1/13 Phase 1
   tests passing; that team's commit `ec59c6e` lands two of the three
   underlying bugs).
3. Once `d3-phase1-multi-imm` lands all three IMM-bake / CarryProp /
   L14-cleanup fixes on `main`, this MUL/SHL/SHR suite will need a re-run
   (no other changes needed) to confirm xpasses and clean up decorators.

## Reproducer

```bash
PICK_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$PICK_GPU timeout 600 python -m pytest \
    c4_release/tests/test_pure_neural_heap_div.py -v --tb=no --timeout=200 --runxfail \
    -k "mul_small or shl_small or shr_small"
# 12 fail (same root cause as test_imm_byte_values[255])
```

Wall clock ~12 min on one GPU.

```bash
# Minimal repro (single-instruction IMM 255 also fails the same way):
CUDA_VISIBLE_DEVICES=$PICK_GPU timeout 300 python -m pytest \
    "c4_release/tests/test_pure_neural_pc.py::TestPureNeuralSingleInstruction::test_imm_byte_values[255]" \
    -v --tb=no --timeout=200
# Got 4210752255 (0xfafafaff) instead of 255.
```
