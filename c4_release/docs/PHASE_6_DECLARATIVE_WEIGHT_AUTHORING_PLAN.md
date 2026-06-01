# Phase 6 — Declarative weight authoring migration

## Goal

After this migration: **every weight in every layer is produced by compilation of declarative spec** (FFNRule + AttentionHeadIR). Adding a fix means editing rules; the compiler synthesizes weights. No imperative `block.ffn.W[...] = X` or `attn.W_q[...] = Y` survives in op factories or setup_helpers.

## Acceptance criteria

1. Every op in `all_core_ops()` has `compiler_ir` populated (no `compiler_ir=None`)
2. Every op's bake_fn body is either `_lower_<op>_via_compiler_ir(block, ir, dim_positions, S)` or removed entirely
3. `setup_helpers.py` `_set_layerN_*` helpers no longer write `block.ffn.W_*` / `attn.W_q/k/v/o` directly
4. `compare_symbolic_to_lowered_ffn` returns clean on every op
5. `compare_symbolic_to_lowered_attn` (analogous tool, to be built in Wave 1) returns clean on every head
6. `verify_claims_static` 100% green
7. New op can be added with zero `bake_fn` body — just `compiler_ir=` with rules
8. A documented "how to declare a new fix" recipe exists at `docs/HOW_TO_ADD_A_CORRECTIVE_OP.md`

## Estimated total: 6-10 weeks elapsed (~40-60 agent-days, heavily parallelizable)

## Agent waves

### Wave 1 — Foundation (1-2 days, 4 agents in parallel)

**1A. Imperative bake_fn census**
- Enumerate every op with an imperative `bake_fn` body (i.e., not just calling a `_lower_via_compiler_ir` helper)
- Group by complexity: trivial (≤10 weight writes), medium (10-100), heavy (>100)
- Save to `.agent-logs/imperative_census_phase6.md`
- Refines the "99 imperative ops" + "89 legacy attention bakes" numbers
- ~3hr

**1B. AttentionHeadIR migration pattern guide**
- Write `docs/ATTENTION_HEAD_IR_MIGRATION_PATTERN.md`
- Canonical recipe: extract Q/K/V/O projections from imperative bake, wrap as `AttentionHeadIR(spec=DeclarativeAttentionHeadSpec(...))`, expose via `compiler_ir=AttentionOp(heads=[...])`
- Document the `compare_symbolic_to_lowered_attn` byte-identity check
- Include 1 worked example (likely L7 since heads already enumerated)
- ~3hr

**1C. FFNRule migration pattern guide**
- Write `docs/FFN_RULE_MIGRATION_PATTERN.md`
- Cover `constant_write`, `gated_write`, `_lower_via_compiler_ir` lowering, the claims-as-witnesses pattern, byte-identity probe
- Include 1 worked example (e.g., `layer14_temp_clear` — small, recently claim-backfilled)
- ~3hr

**1D. Build `compare_symbolic_to_lowered_attn` tool**
- Mirror of `compare_symbolic_to_lowered_ffn` for attention heads
- Given an `AttentionHeadIR`, bake into a `PureAttention`, compare against symbolic execution per query position
- Required for byte-identity validation in subsequent waves
- ~4hr

### Wave 2 — Attention bake migration (per-layer-family parallel; ~1-2 weeks elapsed)

Per the census in 1A, ~89 legacy attention bakes spread across ~14 layers. Migrate per-layer-family — one agent per layer file, plus separate agents for setup_helpers helpers.

| Agent | Target | Complexity |
|---|---|---|
| 2A | L0 attention | trivial (threshold heads) |
| 2B | L1 attention | medium |
| 2C | L2 attention | medium |
| 2D | L3 attention | heavy (carry_forward attn, 7 heads — already partially migrated) |
| 2E | L4 attention | medium |
| 2F | L5 attention | medium |
| 2G | L6 attention (in ops file) | medium |
| 2H | L7 attention (already has FFNRule for some; cover remainder) | small |
| 2I | L8 attention | heavy (SP gather, multibyte fetch, head6 refresh) |
| 2J | L9 attention | medium |
| 2K | L10 attention | heavy (byte passthrough family) |
| 2L | L13 attention setup_helpers | medium (mem_addr_gather) |
| 2M | L14 attention | small |
| 2N | L15 attention | medium |

~14 agents. Run in 2-3 waves of 5-6 parallel each (avoid file conflicts; setup_helpers needs sequential since it's one file).

### Wave 3 — FFN rule migration: small/medium ops (per-layer-family parallel; ~1-2 weeks)

For each op file's FFN bake_fns that aren't yet FFNRule-based. Per the census from 1A.

| Agent | Target | Complexity |
|---|---|---|
| 3A | L0 phase_a_ffn variants | small |
| 3B | L1 FFN | small |
| 3C | L2 FFN | small |
| 3D | L3 FFN | medium |
| 3E | L4 FFN (PC+N chain, ~544 units) | medium |
| 3F | L5 opcode_decode_ffn | medium |
| 3G | L7 (already has L13 ADDR_B_VALID extension) | small |
| 3H | L13 shifts (SHL/SHR select stages) | medium |
| 3I | L14 cleanup ops × 7 | medium-heavy |
| 3J | L15 nibble_copy + small bakes | medium |
| 3K | L16 LEV routing (already has FFNRule; ensure full coverage) | small |
| 3L | L17 post_ops | medium |

~12 agents.

### Wave 4 — FFN rule migration: BIG helpers (~1-2 weeks)

The load-bearing imperative helpers in `setup_helpers.py` / `vm_step.py`. These need careful per-substage migration since they're the model's arithmetic core.

| Agent | Target | Cells |
|---|---|---|
| 4A | `_set_layer6_routing_ffn` | **13638** — the monster |
| 4B | `_set_layer9_alu` (ADD/LEA/ADJ/SUB/ENT/CMP/CARRY/ALU clear/LEV/marker_suppress) | 3641 |
| 4C | `_set_layer11_mul_partial` | 4096 |
| 4D | `_set_layer12_mul_combine` | 4096 |
| 4E | `_set_layer13_shifts` | 4096 |
| 4F | L10 post_op[7] long-division integration | ~1500 |
| 4G | L8 ADD/SUB flat_ffn (60+60 units) | small |
| 4H | L9 post_ops ADD/SUB flat_ffn | small |

Each is a multi-day effort because of the size. Run in 2 waves of 4 parallel each.

**Critical**: each helper migration includes byte-identity validation via `compare_symbolic_to_lowered_ffn`. ANY weight bit-difference = abort and revert.

### Wave 5 — Validation (1-2 days)

| Agent | Task |
|---|---|
| 5A | Run full `compare_symbolic_to_lowered_ffn` on every op; report any failures |
| 5B | Run new `compare_symbolic_to_lowered_attn` (from 1D) on every head |
| 5C | Run `verify_claims_static` corpus-wide |
| 5D | Run full 1096 sweep on fully-declarative model |
| 5E | Run `analyze_scheduler.py` and confirm `phase_pinned_by_deps` for every op (no surprises) |

Any failures gate Phase 7 and require return to Wave 2/3/4 for fixes.

### Wave 6 — Cleanup (1-2 days)

| Agent | Task |
|---|---|
| 6A | Remove obsolete `_set_layerN_*` helpers from `setup_helpers.py` / `vm_step.py` |
| 6B | Remove obsolete `bake_fn` parameter from `Operation` (or make it default to `_lower_via_compiler_ir`) |
| 6C | Update CLAUDE.md / project docs |
| 6D | Drop `pin=` on representative new ops to demonstrate auto-fit allocation works end-to-end |
| 6E | Write `docs/HOW_TO_ADD_A_CORRECTIVE_OP.md` — the documented recipe |

### Wave 7 — Demo (1 day)

| Agent | Task |
|---|---|
| 7A | Write a new corrective op (e.g., fix bug #34 step11:STACK0_byte0) using ONLY declarations — no `bake_fn` body |
| 7B | Show the verifier + compile path picks layer/slot/unit/head and synthesizes weights |
| 7C | Confirm it lands a row improvement on the 1096 metric |

## Total wave count: ~50 agents

Spread across 7 logical waves, ~6-10 weeks elapsed assuming 4-8 parallel at any time.

## Dependencies between waves

```
Wave 1 (foundation)
  ↓
Wave 2 (attention) ⊥ Wave 3 (small FFN)    ← parallel after Wave 1
  ↓                    ↓
       Wave 4 (BIG helpers)                ← depends on patterns from Wave 2+3
  ↓
Wave 5 (validation)
  ↓
Wave 6 (cleanup)
  ↓
Wave 7 (demo)
```

## Risk register

| Risk | Mitigation | Confidence |
|---|---|---|
| Byte-identity drift in a big helper migration | Per-substage commits; compare_symbolic_to_lowered_ffn gate | MED |
| Attention IR can't express some imperative heads cleanly | Wave 1B pattern guide identifies these early; legacy escape valve via `compiler_ir=None` and bake_fn fallback for individual heads if needed | MED |
| `setup_helpers.py` deletion breaks unrelated paths | Wave 6A audits all callers before removal | HIGH |
| Trained weights invalidated by hidden layout drift | Byte-identity required at every commit; allocator pins enforce | HIGH |
| Wave 4 helpers take longer than estimated | Each is independent; can extend that wave to 3 weeks without blocking 5/6/7 | HIGH |

## Connection to today's work

Phase 6 is the next logical arc after the dynamic-placement work landing today (Phases 1-5 of `PATH_TO_FULLY_DYNAMIC.md`). Phases 1-5 give us: layer / slot / unit / head are all compiler-chosen. Phase 6 gives us: weights themselves are compiler-synthesized from declarative spec.

Together they realize: **"adding a fix = write declarative IR; compiler does the rest."**
