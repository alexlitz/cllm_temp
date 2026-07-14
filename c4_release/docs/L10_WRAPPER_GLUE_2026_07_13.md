# L10 tail post-op wrapper-glue collapse — byte-identical, land-ready

**Date:** 2026-07-14  **Branch:** `l10-wrapper-glue` (off `main` `b351ad80`).
**File:** `neural_vm/unified_compiler/ops/l10_ops.py`.
**Lever:** item #1 of `docs/L10_TAIL_AUDIT_2026_07_13.md` — the 13 flag-gated
`make_l10_*_op` tail post-op wrappers each repeated the same ~40–50 lines of pure
GLUE. Table-driven them behind ONE shared builder + per-op DATA, exactly as the
already-landed **R-FRAME INCR-1** collapse did for the AX/SP/BP/PC passthrough
head glue (commit `1af1f0f0`, −83 LOC byte-identical).

---

## 1. The collapse

### Before (13× repeated)

Each `make_l10_*_op` wrapper hand-inlined:

| repeated fragment | occurrences |
|-------------------|------------:|
| the 13-line `d_model` resolution ladder (attn.dim → W_q → W_up → dim_positions max → 512) | 13 (in these wrappers) |
| `ffn = PureFFN(d_model, len(rules))` + `dim_positions_from_bd` + `lower_ffn_rules` + `post_ops.append` bake | 13 |
| the flag-OFF `_noop_bake` + off-path `Operation(...)` stanza | 10 (2 already used `_noop_block_op`) |
| `ir = CompilerIR(); ir.layer(0).ffn.rules.extend(rules)` | 13 |
| `Operation(... target_op_name="l10_post_ops_combined" ...)` metadata | 13 |

Everything that VARIES per op is DATA: `name`, the `_*_enabled()` flag, the
`_*_rules()` factory, `reads`, `writes`, `requires` (tail-ordering pin),
`target_op_name`/`spec_section` (2 ops differ), `smoke_tests`, whether the bake
calls `_suppress_ffn_on_step_boundary`, and (ADD adder only) an exact rule-count
assert.

### After (3 shared lowerings + 13 thin calls)

New shared glue (inserted before `make_l10_add_high_byte_adder_op`):

- `_resolve_l10_postop_d_model(block, dim_positions)` — the 13-line d_model
  ladder, once.
- `_l10_postop_bake(rules, *, suppress, assert_len=None)` — the PureFFN
  bake closure (ladder + lower + optional `_suppress_ffn_on_step_boundary` +
  `post_ops.append`), once.
- `_make_l10_postop(*, name, rules_fn, reads, writes, requires=None,
  target_op_name=…, spec_section=…, smoke_tests=None, suppress=False,
  assert_len=None, flag_fn=None, noop_name=…, noop_target_op_name=…,
  noop_spec_section=…)` — emits the flag-OFF `_noop` Operation (when
  `flag_fn` is given and falsey) or the flag-ON bake+IR+Operation.

Each of the 13 `make_l10_*_op` is now a single `return _make_l10_postop(...)`
call over its per-op DATA. The genuine per-op `_l10_*_rules()` factories (the
hand-tuned discriminators/thresholds) stay inline, untouched.

The two absdiff-return ops' bespoke helpers (`_noop_block_op`,
`_absdiff_ret_bake`) are now subsumed by `_make_l10_postop(suppress=False)` and
were **deleted** (used nowhere else).

### The 13 collapsed ops (per-op DATA)

| op | flag (default) | suppress | requires | target / spec |
|----|----------------|:--------:|----------|---------------|
| `l10_add_high_byte_adder` | (always ON) | yes | tail_bit32 | combined / multibyte-arithmetic |
| `l10_nonfirst_psh_sp_helper` | `_nonfirst_psh_sp_fix_enabled` (ON) | yes | — | combined / registers |
| `l10_ent_axcarry` | `_l10_ent_axcarry_enabled` | yes | l10_exit_axcarry | combined / function-call |
| `l10_exit_axcarry` | `_l10_exit_axcarry_enabled` | yes | tail_bit32 | combined / function-call |
| `l10_loop_lea_b0_e8` | `loop_lea_b0_e8_restore_enabled` | no | l10_ent_axcarry | combined / registers |
| `l10_loop_lea_b0_e0` | `loop_lea_b0_e0_restore_enabled` | no | l10_loop_lea_b0_e8 | combined / registers |
| `l10_jsr_bp_byte3_clear` | `jsr_bp_byte3_clear_enabled` | no | tail_bit32 | combined / registers |
| `l10_absdiff_argb_li_lo` | `absdiff_fix_enabled` | no | l10_loop_lea_b0_e0 | combined / registers |
| `l10_absdiff_ret_byte1_flag` | `absdiff_ret_byte1_enabled` | no | tail_bit32 | combined / registers |
| `l10_absdiff_ret_byte1` | `absdiff_ret_byte1_enabled` | no | l10_absdiff_ret_byte1_flag | combined / registers |
| `l10_loop_si_byterow_marker_clear` | `loop_si_byterow_marker_clear_enabled` | no | — | combined / registers |
| `l10_loop_li_fetch_addrkey_clamp` | `loop_li_opcode_fetch_addrkey_clamp_enabled` | no | — | **layer3_carry_forward_attn** / memory |

`suppress=False` is LOAD-BEARING on the 8 value-byte-row / prompt-code-byte ops
(they INTEND to fire on `IS_BYTE=1` rows; `_suppress_ffn_on_step_boundary` would
gate exactly those rows off) — the rationale comments are preserved on the thin
calls.

---

## 2. Byte-identity proof

Base for all comparisons: `main` `b351ad80` (the golden-`1c04c3fd` build path).

### Golden state-dict hash (weight-level), UNCHANGED in both flag configs

| config | base `b351ad80` | branch `l10-wrapper-glue` | verdict |
|--------|-----------------|---------------------------|---------|
| flag-OFF (DEFAULT) | `1c04c3fd58f4c814…` | `1c04c3fd58f4c814…` | **IDENTICAL** |
| flag-ON (full campaign) | `712300b7a896d6ce…` | `712300b7a896d6ce…` | **IDENTICAL** |

    CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py

The flag-ON run exercises every one of the 13 ON-path bakes (the weights they
emit); an identical hash proves the emitted FFN weights are unchanged.

### Field-identity proof (the R-FRAME INCR-1 method)

Built all 13 Operations from base `b351ad80` and from the branch, in BOTH
flag-OFF and full-campaign flag-ON configs, and diffed every Operation field
(`name`, `reads`, `writes`, `kind`, `requires`, `target_op_name`,
`spec_section`, `declarative_authority`, `migrated`, `smoke_tests`, and the
ON-path IR rule-count). **Diff clean in both configs** — all 13 built Operations
are field-identical to the pre-refactor version. ON-path IR rule-counts match:
add=128, ent/exit/loop_lea×2/jsr=32, loop_li=16, absdiff_argb/absdiff_ret=15,
loop_si/absdiff_ret_flag=2, nonfirst_psh=1.

---

## 3. −LOC

| metric | base | branch | delta |
|--------|-----:|-------:|------:|
| **code LOC** (cloc, Python) | **7,451** | **6,942** | **−509** |
| total lines | 13,838 | 13,352 | −486 |

**−509 code LOC**, inside the assigned ~450–650 byte-identical estimate. (The
new shared builder + two helpers add ~150 LOC of glue; the 13 wrappers shed
~660 LOC of clone boilerplate.)

---

## 4. Risk / gate

LOW — byte-identical by construction (the emitted Operations/weights are
unchanged). Land gate:
- golden flag-OFF `1c04c3fd` UNCHANGED  ✔
- golden flag-ON campaign `712300b7` UNCHANGED  ✔
- 13-op field-identity diff clean (flag-OFF + flag-ON)  ✔
- l10 post-op / per-op / tail-correction pytest suite (see branch commit)

This is the direct analogue of the already-landed R-FRAME INCR-1 collapse and is
the single highest-leverage byte-identical cut remaining in `l10_ops.py`.
Land-ready.
