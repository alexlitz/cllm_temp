# Lookup-mode rule-derivation (V8 follow-up, 2026-06-04)

Branch: `lookup-mode-rules` (worktree `/tmp/c4-lookup-mode-rules`).
Scope: re-express the production `alu_mode='lookup'` install of
`ALUAndOrXor` as a pure rule-derived bake; delete the imperative class.

## Summary

V8 (`docs/V8_DELETE_AUDIT_2026_06_04.md`) deleted 3 of 6
`efficient_alu_*.py` files; the 3 keep-files held `ALUAndOrXor`,
`AddSub5StageBlock`, `FlattenedDivMod`, `FlattenedALUMul`, and
`ALUShiftComposite` for the production lookup-mode install path. This
wave completes the smallest of those migrations: `ALUAndOrXor` is fully
replaced by a rule-derived `PureFFN` baked from
`wide_alu_dsl.bitwise_rules`.

| Before | After |
|---|---|
| `_make_alu_postop_attach_op(name="l10_alu_postop_attach", alu_cls_name="ALUAndOrXor", ...)` constructs `efficient_alu_neural.ALUAndOrXor(S, BD)` and inserts it into `block.post_ops[0]` (lookup mode). | `make_lookup_mode_l10_bitwise_rules_op()` emits 1,536 `FFNRule`s (3 ops x 512) via `bitwise_rules`, lowers them through `Primitives.lower_ffn_rules` into a fresh `PureFFN(dim=d_model, hidden_dim=1536)`, and inserts that into `block.post_ops[0]`. |
| Forward: `ALUAndOrXor` (= `PureNeuralALU(operations='bitwise')`) routes the residual through `BDToGEConverter` → 3 sub-FFN chains (AND/OR/XOR) → `GEToBDConverter`, writing `OUTPUT_LO/HI` at MARK_AX positions with the opcode-selected byte. | Forward: vanilla `PureFFN` SwiGLU with one hidden unit per `(opcode, nibble_lane, a_nib, b_nib)` tuple. Each unit fires when `MARK_AX + ALU_band[a] + AX_CARRY_band[b]` exceeds threshold 80 AND the opcode gate is on, writing `2.0 / S` to `OUTPUT_*+result` where `result = a OP b`. |

Byte-identity is preserved at the decoded OUTPUT_LO/HI byte: the
`bitwise_rules`-lowered PureFFN was already proven byte-identical to
`ALUAndOrXor` for AND/OR/XOR in `tests/test_wide_alu_dsl.py::
test_bitwise_rules_byte_identity_{vs_composite,randomized}` before this
wave (the same harness validates the install-time bake_fn in the new
`test_lookup_mode_l10_postop_factory_byte_identity`).

## Why not `torch.equal` on weights?

The brief originally asked for `torch.equal` on `block.ffn.W_up/
W_gate/W_down`. That is not feasible: `ALUAndOrXor` is not a `PureFFN`
subclass and exposes no `W_up`. Its forward is a hand-rolled BD↔GE
pipeline whose internal sub-FFNs have totally different shapes
(160-dim GE-format hidden state per byte) from the rule-derived
512-dim residual PureFFN. The semantic byte-identity contract
documented in `docs/IR_DSL_DESIGN.md` Section 4 (and used by every
existing `wide_alu_dsl` test) is `forward(x)` decodes to the same
OUTPUT byte; that's what this wave verifies.

## What was changed

### Production install

- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py`
  - New: `make_lookup_mode_l10_bitwise_rules_op()` — the rule-derived
    factory. Bake_fn emits `bitwise_rules` x 3 ops, sizes a fresh
    `PureFFN(dim=block.ffn.dim, hidden_dim=1536)`, threads the
    compiler-allocated dim positions via `_as_setdim_proxy`, lowers
    through `Primitives.lower_ffn_rules`, and inserts at
    `block.post_ops[0]`. Same phase (1180.10), same
    `target_op_name="layer10_carry_relay"`, same kind=`"block"` as
    the legacy install so the dynamic scheduler co-places it
    identically.
  - Modified: `make_l10_alu_postop_attach_op(alu_mode='lookup')` now
    short-circuits to `make_lookup_mode_l10_bitwise_rules_op()`. The
    `alu_mode='efficient'` branch still raises `NotImplementedError`
    in `_make_alu_postop_attach_op` (preserved for symmetry; efficient
    mode uses `make_efficient_l10_andorxor_wrap_op` separately).

### Class deletion

- `c4_release/neural_vm/efficient_alu_neural.py`
  - Deleted: `class ALUAndOrXor(PureNeuralALU)` (3 lines).
  - The sibling `class ALUAddSub(PureNeuralALU)`, plus the rest of the
    file (composite stages, `BitwiseGEToBDStage`, `FlattenedALUMul`,
    `ALUShiftComposite`, etc.) are kept — they remain load-bearing for
    L8/L9/L11/L12/L13 lookup-mode installs per the V8 audit.

### Test updates

- `c4_release/tests/test_wide_alu_dsl.py`
  - Dropped the `ALUAndOrXor` import + the `composite` fixture; the
    byte-identity tests now compare `bitwise_rules`-lowered PureFFN
    output to Python's reference op directly (= the post-V8
    contract).
  - Added `test_lookup_mode_l10_postop_factory_byte_identity` — runs
    the new factory's bake on a mock block and asserts the installed
    PureFFN decodes to the right OUTPUT byte across 96 (op, a, b)
    tuples.
- `c4_release/tests/test_alu_wide_composites_per_op.py`
  - The `andorxor_composite` fixture now invokes the new factory's
    bake_fn to install the rule-derived PureFFN; the symbolic-forward
    + no-fire-without-MARK_AX tests are retargeted at that PureFFN
    instead of the deleted `ALUAndOrXor` composite. Same input
    vectors, same OUTPUT-byte assertions.

## Verification

- **Byte-identity sweep**: 96/96 OUTPUT bytes match Python's reference
  for AND/OR/XOR over 32 randomized (a, b) pairs each. Both via the
  direct rule-lowered PureFFN test
  (`test_bitwise_rules_byte_identity_randomized`) and via the
  install-path factory test
  (`test_lookup_mode_l10_postop_factory_byte_identity`).
- **`tests/test_wide_alu_dsl.py`** + **`tests/test_alu_wide_composites_per_op.py`**:
  97/98 pass. The 1 failure
  (`test_wide_mul_rules_rejects_bad_args`) is preexisting on
  `speedup-cache-and-buckets` HEAD and unrelated to this wave (it's a
  DSL multi-byte MUL contract gate).
- **Smoke baseline retained**: `pytest tests/test_smoke.py
  tests/test_smoke_pure_neural.py --tb=no -q` reports **29 passed /
  23 failed / 28 xfailed / 12 xpassed** at 916.32s — matches the V8
  baseline (29 / 52). No regression in the bitwise smoke cases:
  `TestSmokeBitwise::test_or_basic`, `test_and_basic` continue to
  pass; `test_xor_basic` continues to fail (preexisting xfail-class
  failure unrelated to the bitwise path itself — see
  `docs/STATUS_1096_2026_06_04.md` for the smoke breakdown).
- **`tools/lint_raw_ffn_rule.py`**: OK — 39 raw FFNRule calls across 3
  files, all within baseline. The new factory routes through
  `wide_alu_dsl.bitwise_rules`, not raw `FFNRule.constant_write` /
  `gated_write`.

## What's left for full V8 completion

Per the V8 audit's "Follow-up wave" section:

| Composite | Status | Remaining |
|---|---|---|
| L10 `ALUAndOrXor` | **DONE this wave** | (class deleted) |
| L8/L9 `AddSub5StageBlock` | NOT STARTED | `wide_add_rules`/`wide_sub_rules` exist (Wave W3, width_bytes=1..8 byte-identical); needs the same install rewire + a multi-byte byte-identity gate. The composite also writes CARRY flags for `CarryPropagationPostOp` — rule equivalent needed. |
| L11/L12 `FlattenedALUMul` | NOT STARTED | `wide_mul_rules` POC is width_bytes=1 only; multi-byte cascade deferred per `docs/DSL_W5_MULDIV_LIMIT.md`. |
| L10 `FlattenedDivMod` | NOT STARTED | `wide_div_rules` POC is per-nibble only, NOT byte-identical for multi-byte (same doc). |
| L13 `ALUShiftComposite` | NOT STARTED | `wide_shift_rules` exists (W2, byte-identical up to 32-bit); needs the install rewire only. The 5 cooperating ops (`make_alu_shift_composite_ops`) currently install the imperative composite. |

The shift wave (L13) is the next-easiest analogue to this one — same
shape (single FFN replacing a `PureNeuralALU`-derived composite) and
the rule helper has already cleared its byte-identity gate.

## Constraints honoured

- ONE compile + ONE smoke per attempt (smoke ran once at 916.32s,
  matched baseline 29/52).
- No `gh` CLI use.
- No second iteration on this composite — first attempt succeeded.
- Commit prefix: `dsl-lookup: replace ALUAndOrXor install with rule-derived bake`.
