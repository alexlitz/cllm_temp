# IMM override removal — real surface localized (2026-06-07)

## TL;DR

The b5cf7099 IMM runner override compensates for a bug at **AX bytes 1-3 emit at AX byte rows**, NOT at MARK_AX row. Two previous attempts (Option B v1 at `a53cf8b`, Option B v2 at `1a524938` worktree) wrote correctives at MARK_AX and failed — the AX-byte-0-at-MARK_AX path is already correct.

## Evidence (from v2 probe)

- `FETCH_LO+15 = 40` at MARK_AX for IMM 0xFF — L5 head 0 already routes the correct PC+1 byte.
- `OUTPUT_LO+15 ≈ 90520` at MARK_AX after L6 routing — AX byte 0 emit at MARK_AX is correctly 0xFF.
- With override removed: still wrong. So the divergence is downstream, at the byte rows (not the marker).

## What the override actually does

Read `c4_release/neural_vm/batched_pure_neural.py` near `b5cf7099`: at AX byte 1/2/3 row when OP_IMM is active and the original bytecode byte ≥ 0xE0, it overwrites the model-emitted byte with the bytecode byte. The model gets the high byte WRONG for IMM values where any of the high bytes is in [0xE0, 0xFF].

Suspect: L8 head 3 (CLEAN_EMBED → AX_CARRY at AX byte position) has the same MARK_AX → OP_IMM gating ambiguity at byte positions. The gate may be Q-only and softmax-cancel for byte positions, or L8's CLEAN_EMBED lookup aliases at [0xE0..0xFF].

## Recommended fix path

Probe at AX byte 1-3 rows during IMM 0xFF execution:
1. Capture residual on `AX_CARRY_LO/HI` at byte rows.
2. Compare to expected `CLEAN_EMBED(0xFF)`.
3. If AX_CARRY is wrong, find the L8 head/FFN that owns the byte-row write.
4. Build the corrective at the byte row, gated by IS_BYTE AND OP_IMM AND byte_index ∈ {1,2,3}.

The L5 head 0 (memory load) pattern at `c4_release/docs/DSL_ATTENTION_GATE_FINDING_2026_06_07.md` is a good template — Q+K both have MARK_X, FFN cleanup at non-target rows handles leakage.

## Status

Override remains load-bearing. v2 attempt restored override + shipped disabled scaffolding for future use (not cherry-picked to main).
