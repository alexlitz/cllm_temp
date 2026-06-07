# DSL attention "gate" Q-side pattern can be a no-op (2026-06-07)

## TL;DR

Q-side "gates" of the form `AP(slot, BD.MARK_AX, L)` (writing `L * MARK_AX` to `Q[slot]`) do **not** filter attention mass when the K-side has a uniform `CONST` term at the same slot. The per-Q-row offset cancels through softmax. Many ops use this pattern expecting gating behavior — it's actually a no-op in those cases.

## Discovery

Agent `a53c8efb` attempting IMM Option B (`docs/REMOVAL_1_REAL_SURFACE_2026_06_06.md`) added an L9 corrective with:
- Q: `AP(slot=32, MARK_AX, L)` + `AP(slot=33, OP_IMM, L)` (intended gate)
- K: ADDR_KEY content-match
- V/O: CLEAN_EMBED → AX_CARRY

Expected: the corrective fires only at MARK_AX AND OP_IMM rows.

Actual: the corrective fires at EVERY MARK_AX row regardless of OP_IMM. At MUL's AX-marker row it overwrites L8's correct AX_CARRY (STACK0 operand) with garbage routed from PC+1, breaking MUL.

## Mechanism

Score at (Q row r, K position c):
```
score[r, c] = Σ_slot Q[r, slot] * K[c, slot]
```

When `Q[r, slot=32] = L * MARK_AX[r]`, slot 32's contribution to `score[r, c]` is:
```
L * MARK_AX[r] * K[c, slot=32]
```

If `K[c, slot=32]` is uniform across `c` (typical "Q-only" K-side):
- At MARK_AX=1 rows: contribution is `L * K_uniform` — a per-r constant that adds to every K position equally
- Softmax over K: the constant cancels (`exp(s + C) / Σ exp(s + C) = exp(s) / Σ exp(s)`)
- Net effect: zero filtering

If `K[c, slot=32] = 0` at all c (Q-only slot, K never writes):
- Contribution is exactly 0 for every (r, c) — the slot is dead weight

## Where this pattern is used

Grep for `AP(slot=` near `OP_*` or `MARK_*` references in `c4_release/neural_vm/unified_compiler/ops/`. The Q-side "single-condition" pattern shows up in many places. Each needs an audit.

## How to actually gate Q→K

Either:
1. **K-side complement**: the desired K positions write a positive at the same slot the Q gate reads. E.g., `AP(K_slot, BD.CONST, K_marker)` only at the K positions you want to land on. Then the Q-gated slot acts as a multiplier.
2. **Q-side blocker**: subtract a large negative when the gate fails. E.g., `AP(slot, BD.OP_NOT_IMM, -L * 10)`. But you'd need a complement opcode flag.
3. **K-side blocker**: at non-target K positions write a large negative for the Q-gated slot. Forces softmax mass off them.

The L5 head 0 (memory load) pattern works because the FETCH register has active FFN cleanup at non-AX rows. Without such cleanup, Q-side single-condition gating leaks.

## Action items

1. **Audit all Q-side single-condition `AP(slot, MARK_*, L)` patterns** across `ops/*_ops.py`. Each one is a potential silent bug.
2. **Add DSL Interpreter coverage**: simulate attention over the full bytecode position grid. Flag heads that fire at unexpected rows.
3. **Add a static check** to the IR compiler: warn on Q-side gates that lack K-side complements OR explicit blockers at the same slot.

## Bug status

The IMM Option B attempt failed because of this — corrective wrote AX_CARRY at all MARK_AX rows. Reverted. The `b5cf7099` IMM override remains load-bearing pending a fix that gates by writing K-side complements or using a cleanup pattern.

## Worth checking

Several "successful" fixes this session that use the `AP(slot, MARK_*, L)` Q-gate pattern may have the same problem but happened to not hit the failure mode (e.g., the downstream consumer happened to not care about the leak rows). Worth a paranoid check on:
- L7 head 5 K-side OP_IMM blocker (commit `ff4edb61`) — claimed to work, but did it really?
- Var fix v2 H1+4 blocker (commit `9e4711e7`) — uses K-side blocker, should be fine
- L10 MEM_ADDR_SRC predicate (commit `0b957e03`) — FFN rule, not attention, so unaffected
