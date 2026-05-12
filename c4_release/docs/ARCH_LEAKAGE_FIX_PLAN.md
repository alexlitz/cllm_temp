# Architectural leakage fix plan — 2026-05-12

Today's wave of agents independently rediscovered the same class of bug:
silent collisions and leakages within the "compiled framework." The
framework allocates dims by name but doesn't enforce ownership or sharpness.
This doc plans systematic fixes.

## Observed leakage patterns

| Pattern | Example | Today's diagnosis |
|---|---|---|
| **W_v slot collision** | L5 attn head 6 W_v[5, OP_ADD] = 1.0 collides with `_set_function_call_weights`'s BP→TEMP write at the SAME entry | `c1a5398` — deleted deprecated heads to break the collision; the framework allowed it silently |
| **Softmax distribution leakage** | L4 PC+2 write at AX marker softmax-distributes into all 16 ALU_HI slots, even after L6 clear (-5/S per slot) | AddSub5Stage diag: `ALU_HI_scalar=7` (should be 0) |
| **Stale residual dim** | AX_CARRY at AX marker holds prev-prev step's AX value because no layer refreshes it after L8 multibyte_fetch writes new AX to OUTPUT_LO/HI | AddSub5Stage diag: `AX_CARRY_LO=10` (stale, latest IMM was 32) |
| **`BD = _SetDim` hardcoding** | PureFFN subclasses use legacy dim positions that map to different dims under compact `pin_io_only=True` layout | A1 (`5fc519d`) fixed `BinaryOpByteZeroingPostOp`; P2 byte-0 (`619a14d`) fixed 3 more post-ops; many more remain |
| **Residual stream sharing without ownership** | L4 FFN writes OUTPUT_LO at AX marker (PC+2 for next-step PC) — same dim L7/L8 use for ALU operand. No isolation, no warning. | Inferred from L4→ALU_HI leakage |

## Root causes

1. **Softmax can be made arbitrarily sharp via Q/K scaling**, but most attention bakes haven't been tuned for that. Heads default to `scale ≈ 1.0` (i.e., raw Q·K, divided by sqrt(HD)); we need ~10-15x scale at the target to drown other positions.
2. **ALiBi slopes can make distant K positions effectively unreachable**, but they're set heuristically per-head and not co-tuned with Q/K scales.
3. **No dim-ownership registry**: `Operation.reads`/`writes` are name-granularity, not (position, dim) pairs. Multiple bakes can claim the same name without collision detection.
4. **No softmax sharpness audit**: the framework doesn't measure or assert that heads are sharp enough to avoid leakage.
5. **No staleness invariant**: dims that need refreshing after a producing op fires aren't tracked.

## Plan

### Phase 1 (foundational — parallelizable, can run NOW)

| Agent | Scope | Branch |
|---|---|---|
| **A. Softmax sharpness audit** | New `c4_release/tests/test_softmax_sharpness.py` that, for each attention head, constructs a known K pattern, runs forward, computes softmax mass on the intended target K position, asserts ≥ 99%. Heads below 99% logged with their current Q-scale, K-scale, ALiBi slope, and suggested target scores | `audit-softmax-sharpness` |
| **B. Dim-ownership registry** | Add `claims: Set[Tuple[int, str, str]]` to `Operation` (layer_idx, "head_<n>"\|"ffn_unit"\|"W_v_slot_<base+i>", input_dim_name). Compiler asserts no overlap at registration. Update all `make_*_op` factories to declare their claims | `op-dim-ownership-registry` |
| **C. `BD = _SetDim` lint sweep** | Grep all bake functions for `BD = _SetDim` / hardcoded `_SetDim.XYZ` refs. Generate a report at `c4_release/docs/BD_SETDIM_HARDCODE_AUDIT.md` listing every site + fix-pattern. Optionally auto-patch trivial cases | `audit-bd-setdim-hardcodes` |

### Phase 2 (apply fixes — sequential after Phase 1)

| Agent | Scope | Branch |
|---|---|---|
| **D. Sharpen leaky heads** | For each head flagged by Audit A, bump Q-scale and/or ALiBi slope until softmax mass on target reaches ≥ 99%. Verify byte-identity preserved (or re-tune downstream consumers if behavior changes intentionally) | `sharpen-leaky-attention-heads` |
| **E. Thread `dim_positions` for remaining `_SetDim` hardcodes** | Apply A1's pattern (commit `5fc519d`) to every site flagged by Audit C. Should subsume the "AddSub5Stage was correct but L4/L6/L7 upstream were wrong" pattern by making layout collisions impossible | `thread-all-dim-positions` |

### Phase 3 (architectural — bigger lift, parallel after Phase 2)

| Agent | Scope | Branch |
|---|---|---|
| **F. Position-aware writes set** | Refactor `Operation.writes: Set[str]` → `Set[Tuple[PositionTag, str]]`. Forces every op to declare which (position, dim) it writes — catches L4 incidentally writing OUTPUT_LO at AX marker | `op-position-aware-writes` |
| **G. Residual-dim staleness invariants** | Add `produces: Dict[str, str]` on Operations declaring "dim X is the canonical fresh source for register Y". Compiler can detect: "AX_CARRY is read by L3 head 1 but no op produces AX_CARRY in the same step the AX value is written" — that's the AddSub5Stage stale-AX_CARRY bug | `op-staleness-invariants` |

## Expected impact

- **Phase 1** lands the audit/registry infrastructure. Identifies all current leakage sites systematically (instead of one-by-one via failing tests).
- **Phase 2** applies sharpening + dim-positions threading. Should fix the L4→ALU_HI noise, the stale-AX_CARRY blocker (combined with the in-flight AX_CARRY refresh agent), and unblock the "by-luck" pattern that's currently masking layout-dependent bugs.
- **Phase 3** makes the framework actively prevent the next collision/leakage from ever shipping.

Combined: Phase 2 ADD should work, Phase 3 BNZ/JSR/etc. should benefit, and the framework gains a structural defense against the recurring pattern.
