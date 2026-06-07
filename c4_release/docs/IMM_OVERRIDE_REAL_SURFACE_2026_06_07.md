# IMM override removal — real surface localized (2026-06-07)

## TL;DR (updated after v3 probe, 2026-06-07 evening)

The `b5cf7099` IMM AX override is genuinely load-bearing for IMM bytes
in `[0xE0, 0xFF]`. **Both** the MARK_AX path (AX byte 0 emit) AND the
AX-byte-1/2/3 emit paths are corrupted for high-bit IMMs. The v3 probe
(`tools/imm_ax_byte_probe.py`) measuring `XOR_BASIC` (IMM 0xFF / PSH /
IMM 0xD5 / XOR / EXIT) with the override left in place shows wrong
residual at MARK_AX itself — contradicting the "MARK_AX is FINE" claim
that motivated the v2 (AX-byte-row-corrective) attempt.

## v3 probe evidence (override on; test passes via override)

At the IMM 0xFF AX-marker forward (fwd=6, MARK_AX at pos 49):

- **L6 (post-L5 block):** `FETCH_LO+5 = 40`, `OUTPUT_LO+14 = 5`.
  - Expected (if L5 head 0 correctly attended PC+1 byte 0xFF):
    `FETCH_LO+15 ≈ 40` (nibble F).
  - Actual: L5 head 0 attended a byte whose low nibble = 5 — i.e. the
    OPCODE byte at PC+0 (`0x05 = IMM`), not the IMM byte at PC+1.
- **L8 (post-L7 block):** `AX_CARRY_HI+1 = 342.668` (huge dominant peak)
  with `AX_CARRY_HI+15` at the -5 baseline.
  - Expected (for 0xFF, hi nibble F): peak at `AX_CARRY_HI+15`.
  - Actual: peak at slot 1 — completely wrong nibble.

The brief that motivated v3 claimed `FETCH_LO+15 = 40` and
`OUTPUT_LO+15 ≈ 90520` at MARK_AX for IMM 0xFF. **Neither is observed.**
The brief appears to have been written from a different test case or a
stale state.

## Root cause direction (best guess)

**L5 head 0** (`l5_ops.py:262-280`, non-first-step immediate fetch at
AX from `TEMP = PC+1`) is landing on the OPCODE byte at PC+0 for
high-bit IMMs. The Q-side reads `TEMP+k` (which should hold the PC+1
nibble pattern) and matches it against `ADDR_KEY+k` on the K-side. For
0xFF, the result indicates `TEMP` at MARK_AX is NOT correctly set to
PC+1 — it's effectively pointing at PC+0.

`TEMP` at MARK_AX must be propagated from PC marker rows by an
intermediate routing op (L3 head 7 stages TEMP at MARK_PC; some later
op relays it to MARK_AX). For low-bit IMMs the alias collides
gracefully; for `[0xE0, 0xFF]` the alias surfaces because the
high-nibble pattern overlaps with the opcode-byte ADDR_KEY in a way
the head can't disambiguate.

## Audit gate findings (informational)

`audit_attention_gates(_layer8_multibyte_fetch_head_spec)` flags:

- slot 32: `Q=IS_BYTE`, **q_only** (no K side) — dead weight.
- slot 33: `Q=IS_BYTE`, **no_op** (K=CONST only) — *but* the production
  model uses **softmax1** (zero-anchor sink). Uniform negative
  contributions under softmax1 DO suppress the row (sink wins). The
  audit's "no_op" classification is a false alarm for this model;
  slot 33 IS an effective gate.
- slot 35: safe (real `MARK_AX` discriminator on K).
- slot 36: safe (`ADDR_B2_HI`).
- slot 52: safe (`MARK_AX` Q + `HAS_SE` K blocker).

So L8 head 3's gating, while flagged by the static audit, is
materially effective under softmax1. The "softmax-cancel" theory in
`docs/DSL_ATTENTION_GATE_FINDING_2026_06_07.md` applies to standard
softmax — re-evaluate the audit for softmax1 before treating any
flagged gate as broken.

## Why prior attempts failed

- **v1 (`a53cf8b`, IMM Option B):** added L9 corrective at MARK_AX with
  a Q-gate. Under softmax1 the gate worked, but the corrective fired
  at every MARK_AX row (not just OP_IMM) because it duplicated head 3's
  Q/K and broke MUL's MARK_AX AX_CARRY too.
- **v2 (worktree `1a524938`):** moved corrective to AX byte rows under
  the assumption MARK_AX was correct. v3 probe shows MARK_AX is NOT
  correct, so v2's premise was wrong.
- **v3 (this attempt):** confirmed MARK_AX residual is wrong; root
  cause likely L5 head 0 attending wrong byte for high-bit IMM. A
  single-rule fix would have to either (a) restructure TEMP staging
  for AX marker rows so PC+1 nibble pattern is unambiguous, or (b)
  add an L9/L10 position-addressed corrective that bypasses the
  content-addressed match. Both are multi-rule structural fixes —
  outside single-rule fix scope.

## Action items

1. **Override stays load-bearing** at
   `c4_release/neural_vm/batched_pure_neural.py:2050-2058`.
2. **Investigate L5 head 0 TEMP staging for high-bit IMMs** — likely
   needs an L4 or L5 corrective that breaks the `[0xE0, 0xFF]` ADDR_KEY
   alias. This is a multi-week investigation, not a single-rule fix.
3. **Update the gate audit** to know about softmax1 vs softmax (the
   "no_op" verdict only applies to standard softmax). See
   `c4_release/neural_vm/unified_compiler/dsl_interpreter.py:482-494`.
4. **Re-audit the 70 static-AST gates** from
   `docs/Q_SIDE_GATE_AUDIT_2026_06_07.md` with softmax1 awareness;
   most are probably effective.

## Test surface

Override-on baseline (this branch):

- `test_imm_exit` (IMM 42 / EXIT): PASS — canary.
- `test_mul_overflow` (IMM 100 / PSH / IMM 5 / MUL / EXIT, expect 500):
  PASS — Q-gate breakage canary.

Override removal regresses `test_xor_basic`, `test_add_16bit`, and
`test_add_carry_cascade` per prior investigations. Confirmed not
attempted in v3 since the model side has no plausible single-rule
fix.

## Files referenced

- `c4_release/neural_vm/batched_pure_neural.py:2050-2058` — IMM AX
  override (load-bearing).
- `c4_release/neural_vm/unified_compiler/ops/l5_ops.py:262-280` —
  L5 head 0 spec (the surface).
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1454-1524` —
  L8 multibyte_fetch head 3 spec (downstream consumer of L5 FETCH).
- `c4_release/tools/imm_ax_byte_probe.py` — v3 probe.
- `c4_release/neural_vm/unified_compiler/dsl_interpreter.py:470-535` —
  `audit_attention_gates` (softmax-cancel classifier).
