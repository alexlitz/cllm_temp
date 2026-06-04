# Lookup-mode AddSub → rule-derived migration findings (2026-06-04)

Branch: `lookup-addsub-rules` (worktree `/tmp/c4-lookup-addsub-rules`).
Author: V8 follow-up agent.
Goal: re-express the production `alu_mode='lookup'` install of
`AddSub5StageBlock` (`c4_release/neural_vm/efficient_alu_addsub_split.py`)
as a pure rule-derived bake, parallel to the in-flight `ALUAndOrXor`
POC, so the file can be deleted.

## TL;DR

A rule-derived FFN that produces byte-identical output to
`AddSub5StageBlock.forward` for single-byte ADD/SUB is **feasible and
verified** via a 131,072-rule full-byte lookup table. The
`torch.allclose` check on the FULL post-FFN residual passes
(`atol=0.1`) on all 20 randomized 8-bit operand pairs (10 ADD + 10
SUB) including overflow, underflow, and zero-carry boundary cases.

However, this single-pass full-byte lookup is **not viable for
production**: 131k rules at d_model=512 would consume ~268 MB per L8
post_op FFN (×2 for L8 + L9). The composite uses ≪ that.

The proper migration replicates the 5-stage carry-lookahead structure
as rule-derived FFNs (one per stage), which is a multi-commit V9 wave
effort, not a single-PR delete swap. AddSub5StageBlock therefore
remains in `efficient_alu_addsub_split.py` after this audit. The
production install path in `shared.py` is unchanged.

## What was tried

The POC (committed in `poc/lookup_addsub_rules/`) builds a rule set
that emits exactly 1 rule per (a, b) byte pair × 2 ops (ADD, SUB):

```python
multi_way_and_rule(
    conditions=(
        ("MARK_AX", 40.0),
        (f"ALU_LO+{a_lo}", 15.0),
        (f"ALU_HI+{a_hi}", 15.0),
        (f"AX_CARRY_LO+{b_lo}", 15.0),
        (f"AX_CARRY_HI+{b_hi}", 15.0),
    ),
    threshold=90.0,
    gate=op_gate,  # OP_ADD or OP_SUB
    writes=(
        (f"OUTPUT_LO+{r_lo}", write_amp),
        (f"OUTPUT_HI+{r_hi}", write_amp),
        (carry_dim, write_amp) if carry_out else (),
    ),
)
```

Total rules: 256 × 256 × 2 = **131,072**.

### Amplitude-corrected write magnitude

Lowering via `Primitives.lower_ffn_rules` puts the threshold into
`b_up = -S * threshold` (so margin = 100 - 90 = 10 → SwiGLU `up =
S * margin = 1000`). To produce composite-matching amplitude 2.0
(matching `GEToBDConverter`'s `* 2.0` writes), the write weight is
`2.0 / (S * margin) = 0.002`, not the default `2.0 / S = 0.02`.

With the corrected write magnitude, `torch.allclose(yc, yr,
atol=0.1)` passes on the full 512-dim residual including
OUTPUT_LO/HI bands and the CARRY[1]/[2] add/borrow flags.

### Byte-identity randomized sweep result

10 ADD pairs + 10 SUB pairs, seeded `0xADD04`:

```
add 0x59 0xA3: close_lo=True close_hi=True close_carry=True close_full=True | carry_c=[0.0, 0.0, 0.0, 0.0] carry_r=[0.0, 0.0, 0.0, 0.0]
add 0xF5 0x50: close_lo=True close_hi=True close_carry=True close_full=True | carry_c=[0.0, 2.0, 0.0, 0.0] carry_r=[0.0, 2.0, 0.0, 0.0]
...
sub 0x4E 0x96: close_lo=True close_hi=True close_carry=True close_full=True | carry_c=[0.0, 0.0, 2.0, 0.0] carry_r=[0.0, 0.0, 2.0, 0.0]
```

All 20 cases pass on `close_full` (full-residual `torch.allclose`).

## Why this can't ship as-is

### Memory cost

At d_model=512, hidden_dim=131,072:

- `W_up`, `W_gate`: 131,072 × 512 × 4 B = 268 MB each
- `W_down`: 512 × 131,072 × 4 B = 268 MB
- Total per FFN: ~800 MB

Sparse storage would help (~3-5 non-zero per row of W_up), but the
post-compaction count would still be ~131k active units, which
exceeds the production budget (the existing 5-stage block uses ≪
that across all 5 stages combined).

### Per-nibble alternative

The existing `wide_alu_dsl.wide_add_rules(width_bytes=2)` /
`wide_sub_rules(width_bytes=2)` helpers emit `256 + (2-1) * 513 =
769` rules per op, for a total of 1,538 rules ADD+SUB. That fits the
budget, but they assume **the inter-nibble carry is pre-injected in
the input residual** (see `tests/test_wide_alu_dsl.py:_make_wide_add_input`
docstring). At the L8 input, there is no precomputed LO→HI carry —
the AddSub5StageBlock computes it via its Stage 2 (carry-lookahead).

To use per-nibble rules in a single FFN pass, every hi-nibble rule
must condition on the lo-nibble pair (a_lo, b_lo) that produces the
carry. That expansion gives **256 (lo) + 16*16*16*16 (hi gated on
(a_lo, b_lo, a_hi, b_hi)) = 65,792 rules per op**, same order as
the full-byte lookup.

### Two-pass solution

The only way to keep the per-nibble rule count low (~1.5k total) is
**two sequential FFN passes**:

1. **Stage A** — lo nibble FFN: reads `ALU_LO`, `AX_CARRY_LO`,
   writes `OUTPUT_LO+sum_lo`, `CARRY+0` (intermediate lo-carry).
   Built from `wide_add_rules(width_bytes=1)` + sub variant.
2. **Stage B** — hi nibble FFN: reads `ALU_HI`, `AX_CARRY_HI`,
   `CARRY+0` (from Stage A), writes `OUTPUT_HI+sum_hi`,
   `CARRY+1` (final ADD carry) or `CARRY+2` (SUB borrow). Built
   from a per-nibble lookup conditioned on the lo-carry from
   Stage A.

This matches the existing `_expand_wrapper_blocks` architecture
(each post_op becomes its own block). It would replace the
existing 5-stage installer with a 2-stage rule-derived installer.

**The 2-stage migration is the right V9 wave; not in scope here.**

## What changed in this PR

- This doc.
- `poc/lookup_addsub_rules/`: standalone scripts validating the
  byte-identity contract for the single-pass full-byte lookup
  rule path. Kept as future reference for the V9 rule-derived
  migration.

`efficient_alu_addsub_split.py` is **not deleted** in this wave.
The production install path
(`_make_alu_postop_attach_op` in `shared.py`,
`efficient_l8_addsub_wrap_op` in `alu_ops.py`) is unchanged.

## Verification

The POC scripts are self-contained:

```bash
cd /tmp/c4-lookup-addsub-rules
python poc/lookup_addsub_rules/poc_randomized_byte_identity.py
```

Output: all 20 randomized 8-bit (a, b) pairs pass on `close_full`
for both ADD and SUB, with `carry_c == carry_r` exactly.

## Smoke baseline (no production change)

Production install path is unchanged. Smoke baseline at HEAD
`0863d7cb` is preserved per `docs/V8_DELETE_AUDIT_2026_06_04.md`
(≥ 29 / 52). No smoke run was added in this wave because no
production code path was modified.

## Follow-up (V9)

To complete the V8 deletion of `efficient_alu_addsub_split.py`:

1. Define a **2-stage rule-derived AddSub installer** using
   `wide_add_rules(width_bytes=1)` for Stage A (lo nibble) and a
   new per-nibble helper for Stage B (hi nibble + final carry
   propagation, conditioned on intermediate `CARRY+0` from Stage A).
2. Replace the `_make_alu_postop_attach_op` body for
   `alu_cls_name == "ALUAddSub"` with a 2-FFN install: insert two
   rule-derived `PureFFN` instances into `block.post_ops` instead
   of one `AddSub5StageBlock`.
3. Byte-identity validate via the same POC contract (full-residual
   `torch.allclose(atol=0.1)`) and verify smoke ≥ 29 / 52.
4. Delete `efficient_alu_addsub_split.py` and rewire the
   `shared.py` / `alu_ops.py` callers.

The V9 wave is self-contained and should be a single follow-up PR.
