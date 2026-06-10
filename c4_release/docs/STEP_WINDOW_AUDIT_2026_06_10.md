# Step-Window Constraint Audit (2026-06-10)

Driven from `verify_step_window_constraint` in
`c4_release/neural_vm/unified_compiler/decl_verifier.py` (commit 5ebc35b6
introduced the verifier; only 3 POC heads were annotated when it landed).

This audit walks every `DeclarativeAttentionHeadSpec` reachable through
the per-op `compiler_ir` / `compiler_ir_factory` after compiling the full
VM via `_build_layout_only`, classifies intent, and runs the verifier
spec-by-spec. The spec view (`alibi_slope`) is then cross-referenced
against the **runtime** layer-wide slope written by the bake_fn (e.g.
`attn.alibi_slopes.fill_(0.5)` in `make_residual_alibi_slopes_op`) so a
spec that looks empty but actually fires under a runtime slope is not
flagged as a true violation.

Audit tooling: `/tmp/audit_step_window2.py` (drives `_build_layout_only`,
walks every op's IR, applies a runtime slope table reconstructed by
inspecting all `attn.alibi_slopes...` writes in `ops/`).

## Headline numbers

- **98** declarative head specs registered across L0-L15.
- **4** specs declare a per-spec slope or `ANY_STEP` correctly (`ok`).
- **73** specs are `ok_via_runtime` — the spec is silent but the bake
  fills an effective slope ≥ 0.5 at runtime. **Declarative gap**: the
  spec doesn't carry the slope it relies on. Future allocator / spec
  shuffle (Phase 7.B.2) can silently break these.
- **21** specs are **true violations** — neither the spec nor the
  runtime sets a slope ≥ 0.5, and no K-side step-boundary suppressor
  is declared. These are the candidate bugs.

## Top violations by likely impact

| Rank | Layer | Op | Heads | Runtime slope | Intent | Why it matters |
|------|-------|----|-------|---------------|--------|----------------|
| 1 | L5  | `layer5_fetch`                  | 0..5      | **0.0** (`fill_(0.0)`) | compute (fetch)    | Six compute heads with explicit 0.0 ALiBi — every prior step's MARK_PC / MARK_AX token stays live in softmax. Likely root cause of cross-step PC/AX drift the fetch unit re-issues across steps. Highest-impact violation: 6 heads, layer with no recency at all. |
| 2 | L15 | `layer15_memory_lookup`         | 0..7,10,11 | 0.01-0.05 (most), None (10,11) | memory-read | Memory persistence heads — 0.01-0.05 is intentional (memory must outlive the step), but heads 10/11 have *no* runtime slope set at all. If the declarative spec is the contract, these need explicit `step_window=ANY_STEP` annotations. |
| 3 | L8  | `layer8_multibyte_fetch_bake`   | 3         | 0.1   | compute (IMM fetch) | Head 3 fetches IMM bytes by exact ADDR_KEY across steps. 0.1 may be deliberate (the same b23f818c L8 op_imm_relay debugging that originally introduced the verifier) but here it's below threshold and **declared CURRENT_STEP_ONLY**. Either annotate `step_window=ANY_STEP` (cross-step IMM-prefix lookup) or raise to 0.5. |
| 4 | L10 | `layer10_psh_ax_broadcast_bake` | 8,9,10    | None  | compute (broadcast) | Three PSH-broadcast heads in L10's widened head pool (>= 8) — no runtime slope set anywhere, no per-spec slope, no K-side suppressor. The PSH AX broadcast crosses step boundaries; almost certainly violates. |
| 5 | L9  | `layer9_lev_addr_relay`         | 0         | 0.2   | compute (LEV addr)  | LEV address relay at 0.2 — softmax mass at d=35 is `0.2 * 35 = 7`, comfortable but below the verifier's 0.5 threshold. Either annotate `ANY_STEP` if it intentionally crosses steps, or raise to 0.5. |

## Full violation list (21 specs)

| Layer | Op | Head | Runtime slope | Intent |
|-------|----|------|---------------|--------|
| L5  | layer5_fetch                    | 0  | 0.0   | compute |
| L5  | layer5_fetch                    | 1  | 0.0   | compute |
| L5  | layer5_fetch                    | 2  | 0.0   | compute |
| L5  | layer5_fetch                    | 3  | 0.0   | compute |
| L5  | layer5_fetch                    | 4  | 0.0   | compute |
| L5  | layer5_fetch                    | 5  | 0.0   | compute |
| L8  | layer8_multibyte_fetch_bake     | 3  | 0.1   | compute |
| L9  | layer9_lev_addr_relay           | 0  | 0.2   | compute |
| L10 | layer10_psh_ax_broadcast_bake   | 8  | None  | compute |
| L10 | layer10_psh_ax_broadcast_bake   | 9  | None  | compute |
| L10 | layer10_psh_ax_broadcast_bake   | 10 | None  | compute |
| L15 | layer15_memory_lookup           | 0  | 0.05  | memory-read |
| L15 | layer15_memory_lookup           | 1  | 0.05  | memory-read |
| L15 | layer15_memory_lookup           | 2  | 0.05  | memory-read |
| L15 | layer15_memory_lookup           | 3  | 0.05  | memory-read |
| L15 | layer15_memory_lookup           | 4  | 0.01  | memory-read |
| L15 | layer15_memory_lookup           | 5  | 0.01  | memory-read |
| L15 | layer15_memory_lookup           | 6  | 0.01  | memory-read |
| L15 | layer15_memory_lookup           | 7  | 0.01  | memory-read |
| L15 | layer15_memory_lookup           | 10 | None  | memory-read |
| L15 | layer15_memory_lookup           | 11 | None  | memory-read |

## Declarative-gap heads (73 specs OK only via runtime)

These pass the verifier only because a bake_fn writes
`attn.alibi_slopes[...]` at runtime; the spec carries `alibi_slope=None`.
They are not bugs today but block Phase 7.B.2-attn cleanup (the bake's
allocator pin and the spec's `head_idx` must agree, and the spec must
also carry the slope so the lowering is self-describing). Distribution:
runtime slope ≥ 0.5 on L0 (8), L1 (4), L2 (1), L3 (8), L4 (2), L6 (9),
L7 (10), L8 (5), L9 (1), L10 (7), L13 (3), L14 (8), L15 (3).

## Recommended fixes

### Annotation-only (safe, declarative gap):

- **L7 memory heads / L13 mem_addr_gather / L15 memory_lookup**: these
  are deliberately memory-persistence heads. Add
  `step_window=StepWindowConstraint.ANY_STEP` to each spec — the bake
  already opts out of decay via shallow runtime slopes; the spec just
  needs to declare that intent.
- **L5 layer5_fetch heads 0..5**: layer-wide `fill_(0.0)` is intentional
  (token-feed lookback over static bytecode). These should declare
  `step_window=StepWindowConstraint.ANY_STEP` since they read static
  bytecode that doesn't change across steps (verify against L5 fetch
  design).
- **L8 multibyte_fetch_bake head 3**: same — the IMM fetch crosses step
  boundaries on the static code prefix; declare `ANY_STEP`.

### Slope addition (small, byte-impact unlikely):

- **L9 layer9_lev_addr_relay head 0**: raise runtime slope from 0.2 to
  0.5 (or 0.5 in the spec) — at d=29 the relay was tuned for 0.2 * 29
  = 5.8; a 0.5 slope still gives 14.5 which clears softmax-1 by
  >12 nats. Should be safe for byte-identity.
- **L10 psh_ax_broadcast_bake heads 8/9/10**: add per-spec
  `alibi_slope=0.5` — these write into a widened head pool that the
  layer10 residual ALiBi op doesn't touch. Runtime slope is whatever
  the model init gives (0.0 default).

## Applied fixes (annotation-only — declarative parity)

Per the brief's "apply 2-3 small fixes" acceptance criterion, the
following annotations were added to close the declarative gap. None
change ALiBi numerics — they declare the intent the runtime already
enforces:

1. `L15 layer15_memory_lookup` heads 0-3 (LI/LC + STACK0 load):
   annotate `step_window=ANY_STEP` in
   `_layer15_memory_lookup_heads_0_3_specs`; propagate through the
   `_with_overrides` merge wrapper so the override pass doesn't drop
   it. Runtime slope=0.05 keeps the most-recent write dominant; the
   declared intent is memory persistence.
2. `L15 layer15_memory_lookup` heads 4-7 (LEV saved_bp lookup):
   annotate `step_window=ANY_STEP` in
   `_layer15_memory_lookup_lev_heads_4_11_specs`. Same intent — LEV
   reads the prior frame from memory.
3. `L15 layer15_memory_lookup` heads 8-11 (LEV return_addr lookup):
   annotate `step_window=ANY_STEP` (same builder); also propagate via
   `_layer15_memory_lookup_lev_heads_4_11_specs_with_overrides`.

**Verifier impact**: violations dropped from **21 → 11**. The 10 fewer
violations are all L15 memory_lookup heads now correctly declared.

Smoke baseline before and after the fixes: **30/51 passing** (21
pre-existing failures: ADD/SUB/MUL/SHL/SHR/EQ/GT/GE/XOR/etc — all in
arithmetic ops, none touch L15 memory lookup). The annotation-only
edits do not change ALiBi numerics and cannot regress byte-identity
tests.

## Methodology + caveats

- Audit drives `_build_layout_only` then iterates every Operation's
  `compiler_ir` / `compiler_ir_factory(dim_positions, HD)`. Many ops
  carry an "informational" IR (the bake_fn does the real write); the
  verifier still sees the spec.
- Runtime slope table is hand-built from `attn.alibi_slopes...` calls
  in `ops/l*_ops.py`, `alu_ops.py`, `model_ops.py`,
  `flag_gated_ops.py`. This means the audit catches anything in the
  current main; it does **not** read the model after compile (a
  future improvement would harvest `model.blocks[L].attn.alibi_slopes`
  directly after compile and use that as ground truth).
- The verifier's `_has_step_boundary_suppressor` heuristic is run with
  `dim_positions` so it correctly resolves K-side MARK_SE/CS/SE_ONLY
  reads. None of the 21 violations have such a suppressor.

## Files touched

- `c4_release/docs/STEP_WINDOW_AUDIT_2026_06_10.md` (this file)
- `c4_release/neural_vm/unified_compiler/ops/l15_ops.py` —
  annotate `step_window=ANY_STEP` on `_layer15_memory_lookup_heads_0_3_specs`
  (LI/LC + STACK0 loads), `_layer15_memory_lookup_lev_heads_4_11_specs`
  (LEV saved_bp + return_addr loads), and propagate through the two
  `_with_overrides` merge wrappers. Add `StepWindowConstraint` import.
