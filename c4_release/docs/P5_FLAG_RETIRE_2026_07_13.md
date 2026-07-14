# P5 FLAG-RETIREMENT — audit + safe default-ON-fix retirements

**Date:** 2026-07-14 · **Branch:** `p5-flag-retire` (off `main` b351ad80)
**DEFAULT golden (all-flags-unset):** `1c04c3fd58f4c814ecf20811298d6b5f682dfcd7b3ca971a3da400fd18339d01`
**Golden gate tool:** `CUDA_VISIBLE_DEVICES="" python tools/_isa_golden_hash.py`

## Mission

Size + start the P5 endgame (the decisive <8K-line lever). ~140 `C4_*` flags
exist; many are STABLE default-ON verdict-neutral FIXES whose `=0` escape-hatch
is now dead weight (the fix is proven and will never be reverted). P5 = delete
the escape-hatch branch → make the fix UNCONDITIONAL → −LOC + fewer flags,
**with the DEFAULT golden `1c04c3fd` unchanged** (the default IS the fix path, so
retiring the OFF branch is default-neutral by construction).

## Retirement pattern (the general lever)

Every default-ON fix flag has (up to) three touch-points:

1. a **helper** `..._enabled()` returning `os.environ.get("C4_X", "1") != "0"`
   (sometimes gated on a campaign prerequisite `no_stack0_emit_enabled()`);
2. one or more **consumers** — either `if not helper(): return <no-op>` (bake a
   dummy op) or `if helper():` / `X if helper() else Y` (inline behaviour
   select);
3. a **cache-key snapshot entry** in
   `full_vm_compiler_dynamic.py` (the `flag_key` dict, ~line 2000-2450) so ON/OFF
   builds never share a serialised disk/memo entry.

**Retire =** (a) collapse the helper to its always-taken branch (for
campaign-gated fixes → `return no_stack0_emit_enabled()`; for ungated fixes →
delete the helper and inline `True`), (b) simplify the consumer to drop the flag
check, (c) delete/collapse the cache-key entry.

⚠ The **campaign prerequisite** (`no_stack0_emit_enabled`, i.e.
`C4_NO_STACK0_EMIT`) is NOT a fix escape-hatch — it is load-bearing for the
non-campaign golden (`e50521f3` / `91f55411`). It is a KEEP flag. Retiring a
campaign-gated fix leaves the `no_stack0_emit` gate in place so the non-campaign
build still bakes NO units.

## Flag audit — 136 `C4_*` flags in `neural_vm/`

Boolean flags split by default: **~69 default-ON** (retire candidates) +
**~34 default-OFF** (experimental / behaviour-selectors, KEEP) + ~33 non-boolean
(paths / ints / diagnostic toggles — out of scope). Categories:

### A. RETIRE — landed-stable default-ON fixes (escape hatch is dead weight)

These are the P5 −LOC lever. Each is a proven default-ON verdict-neutral (vs the
fork-point) fix; the `=0` branch will never be exercised in production.

| Flag | Site | Status |
|---|---|---|
| `C4_OUTPUT_B0_NOLEAK` | l11 SHR OUTPUT b0 leak-cancel | **RETIRED** ✅ |
| `C4_MUL_B1_DELIVERY` | l14 JSR-clear MUL/SHL blocker | **RETIRED** ✅ |
| `C4_ALU_OPERAND_SURVIVE` | l9 clear-spare + block-17 h4 qveto | **RETIRED** ✅ |
| `C4_CLEAN_OPERAND_ADD` | l8 CleanOperandOneHotFFN (arith) | **RETIRED** ✅ |
| `C4_CLEAN_OPERAND_BITWISE` | l8 CleanOperandOneHotFFN (bitwise) | **RETIRED** ✅ |
| `C4_CLEAN_EMITTER` | L25 marker-row OUTPUT sink | **RETIRED** ✅ |
| `C4_CMP_HI_LT_ALU15_GUARD` | l10 hi_lt ALU_HI+15 veto drop | **RETIRED** ✅ |
| `C4_CMP_GT_LO_LT_HIEQ_GUARD` | l10 GT/GE 2.5→2.75 raise | **RETIRED** ✅ |
| `C4_ADDSUB_DUMP_BOOST`, `C4_AX_BYTE1_DUMP`, `C4_AX_BYTE1_HINIB`, `C4_AX_BYTE1_SIGNEXT_LEA`, `C4_CMP_BYTE0_SE_RECOVER`, `C4_CMP_EQ_HINIB_VETO`, `C4_CMP_FLAG_MARGIN_FIX`, `C4_CMP_GT_LO_MARGIN`, `C4_DIVMOD_AXCARRY_CLEAR`, `C4_DIVMOD_BYTE0_SE_RECOVER`, `C4_DIVMOD_STACK0_BYTE1_CLEAR`, `C4_DIV_MULTIBYTE`, `C4_ENT_FRAME_SP_BYTE0_PSH_BLOCKER`, `C4_ENT_SP_BYTE1_FF_H1_HARDEN`, `C4_FUNC_LEA_REREAD_BP_RESHARPEN`, `C4_IFVAR_BZ_HI_NIBBLE`, `C4_JSR_BP_BYTE3_CLEAR`, `C4_L10_ENT_AXCARRY`, `C4_L10_EXIT_AXCARRY`, `C4_L15_LEV_ADDR_WIDEN`, `C4_L15_LEV_PC_ONLY`, `C4_L15_LEV_PC_RESTORE`, `C4_L15_LI_SUPPR_INERT`, `C4_L15_LOOKUP_CMP_VETO`, `C4_L16_LEV_PC_TOP_OPCODE_GATE`, `C4_L8_ADJ_LO_AX_MARKER_BLOCKER`, `C4_L8_OPERAND_SP_DISC`, `C4_LEA_E8_FIRST_ENT_GATE`, `C4_LEV_AX_BYTE1_KILL`, `C4_LOOP_LEA_B0_E0`, `C4_LOOP_LEA_B0_E8`, `C4_LOOP_LEA_B0_E8_OPLEA_REQ`, `C4_LOOP_LEA_OPLEA_GATE`, `C4_LOOP_LI_FETCH_ADDRKEY_CLAMP`, `C4_LOOP_SI_BYTEROW_CLEAR`, `C4_MUL_BYTE0_SE_RECOVER`, `C4_MUL_L11_SE_RECOVER`, `C4_MUL_L19_FLOOD_CAP`, `C4_MUL_L19_PRODUCT_BOOST`, `C4_MUL_MULTIBYTE_L19_BOOST`, `C4_MUL_SE_RECOVER_STRICT_ONEHOT`, `C4_MUL_W2_THRESH_FIX`, `C4_MUL_WIDTH2`, `C4_NESTED_JSR_PC_FIX`, `C4_NONFIRST_PSH_SP_FIX`, `C4_PSH_ARG_VAL_AX`, `C4_PSH_STACK0_HIGHBYTE_DARKEN`, `C4_SILI_B1_RESTORE`, `C4_SILI_CAM_B1`, `C4_SP_POP_CARRY_BYTE0_DOMINATE`, `C4_STACK0_NEXT_ARITH`, `C4_STACK0_POP_LOADED_SHALLOW_CRUSH`, `C4_STORE_AX_B0_OVERRIDE_V2`, `C4_SUB_FULL_BORROW`, `C4_TAIL_LEA_E8_ARITH_GUARD`, `C4_TAIL_LEA_E8_DIVMOD_GUARD`, `C4_TAIL_LEA_E8_ENT_GUARD` | assorted per-op fixes | **REMAINING (~56)** — same pattern, retire next |

### B. RETIRE-AFTER-COLLAPSE — M8 ENUMERATED→COMPUTED collapse toggles

Default-ON collapses that changed an FFN `hidden_dim` (kept as kill-switches for
the collapse). Retire once the collapse is declared permanent; cache-key must
keep a geometry distinguisher.

`C4_STACK0_STORE_LOADED_COMPUTED`, `C4_STACK0_POP_LOADED_COMPUTED`,
`C4_STACK0_STORE_E8_COMPUTED`, `C4_STACK0_STORE_TOP_E0_COMPUTED` (default-ON),
`C4_L14_BYTE_COMPUTED`, `C4_WIDE_MUL_BYTE1_COMPUTED` (default-OFF pilots).

### C. KEEP — campaign umbrella + default-OFF experimental / behaviour-selectors

- **Campaign umbrella (load-bearing for non-campaign golden):**
  `C4_NO_STACK0_EMIT`, `C4_OPERAND_FROM_MEMSP`, `C4_CAMPAIGN`, `C4_DERIVE_*`
  (memory/imm/jmp/control/gate_scales — the DSL-derivation switches).
- **Default-OFF experimental / feasibility:** `C4_CLEAN_OPERAND` (full
  arith+cmp feasibility), `C4_ALU_OPERAND_SURVIVE_CMP_RAW` (LT-in/out selector),
  `C4_MUL_MULTIPASS`, `C4_DIV_MULTIPASS`, `C4_ABSDIFF_FIX`, `C4_ABSDIFF_RET_BYTE1`,
  `C4_ADDSUB_DECLARATIVE`, `C4_AX_BYTE1_FULL_WIDTH`, `C4_AX_HIBYTE_CLEAR`,
  `C4_CMP_COMBINE_MARGIN`, `C4_EMIT_G5_RBYTE`, `C4_FAITHFUL_ALU_RAW`,
  `C4_FFN_LINT_*_DEMO`, `C4_FUNC_CMP_OPERAND_CLEAN`, `C4_INC3_MEMAX_B1_OFF`,
  `C4_JSR_AX_CLEAN`, `C4_JSR_PC_BYTE1`, `C4_LEA_LOCAL_E8_MULTILOCAL_GUARD`,
  `C4_LI_VALUE_LOAD`, `C4_NESTED_ENT_AXDUMP`, `C4_OPCAM_FRAME`,
  `C4_SP_BYTE2_CARRY`, `C4_SP_POP_MARKER_CMP3_HARDGATE`, `C4_STORE_AX_B0_OVERRIDE`
  (superseded by V2), plus the non-boolean diagnostic/runtime flags
  (`C4_BATCH_*`, `C4_ACTSCALE_JSON`, `C4_CSR_INFERENCE`, `C4_SPEC_*`, …).
- **Owned by another agent — DO NOT TOUCH:** `C4_R_FRAME_TAIL` (the
  `p5-rframe-retire` worktree owns it).

### D. DEAD — no live consumer at all

`C4_SHIFT_OUTPUT_B0_CLEAR` — the `ShiftOutputClearFFN` wrap it gated was deleted
2026-07-13; the helper had ZERO callers. **RETIRED** (deleted outright) ✅.

## Retirements executed this session (9 flags)

| Commit | Flag(s) | −LOC (net for that file-set) |
|---|---|---|
| `001b6870` | `C4_OUTPUT_B0_NOLEAK` | helper deleted + hatch + cache-key |
| `efe7e284` | `C4_MUL_B1_DELIVERY` | helper deleted, 2 guards simplified (also fixed a latent inconsistent-default cache-collision: cache-key had default=0, helper default=1) |
| `92afbf38` | `C4_ALU_OPERAND_SURVIVE` | 2 helpers deleted (l9 + model_ops), 2 guards inlined |
| `cd7b1441` | `C4_CLEAN_OPERAND_ADD` + `C4_CLEAN_OPERAND_BITWISE` | 2 helpers collapsed, 2 cache-key entries → 1 |
| `ecbea63d` | `C4_SHIFT_OUTPUT_B0_CLEAR` (dead) + `C4_CLEAN_EMITTER` | dead helper deleted; clean_emitter helper collapsed |
| `b49e97d9` | `C4_CMP_HI_LT_ALU15_GUARD` + `C4_CMP_GT_LO_LT_HIEQ_GUARD` | 2 helpers collapsed, 2 cache-key entries deleted |

**Aggregate:** 7 files touched, **+97 / −334 = net −237 LOC**, 9 flags fully
removed (0 env-reads remaining for each). All 4 mission-named safe candidates
(`C4_OUTPUT_B0_NOLEAK`, `C4_MUL_B1_DELIVERY`, `C4_ALU_OPERAND_SURVIVE`,
`C4_CLEAN_OPERAND_ADD`, `C4_CLEAN_OPERAND_BITWISE`) retired; `C4_R_FRAME_TAIL`
untouched (rframe agent owns it).

## Golden-`1c04c3fd`-unchanged proof

Verified after **every** commit — `CUDA_VISIBLE_DEVICES="" python
tools/_isa_golden_hash.py` returns `1c04c3fd…` at each step (default env, no
`C4_*` set):

- baseline b351ad80 default → `1c04c3fd…`
- explicit `C4_OUTPUT_B0_NOLEAK=1 C4_MUL_B1_DELIVERY=1 C4_ALU_OPERAND_SURVIVE=1
  C4_CLEAN_OPERAND_ADD=1 C4_CLEAN_OPERAND_BITWISE=1` → `1c04c3fd…` (confirms the
  default IS the fix path)
- after each of the 6 commits → `1c04c3fd…` (unchanged)

Retained-flag functionality re-proven: `C4_ALU_OPERAND_SURVIVE_CMP_RAW=1` still
changes the build (`60ae989b…` ≠ default), and `C4_CLEAN_OPERAND=1` matches its
clean-base behaviour (byte-neutral on b351ad80). Under `C4_NO_STACK0_EMIT=0`
(non-campaign) all six campaign-gated retired fixes correctly resolve to
`False` — the non-campaign golden path is preserved.

**Note:** the non-campaign escape hatch (`C4_NO_STACK0_EMIT=0`) currently ERRORS
(`AttributeError: _SetDim.LI_ZEROADDR_COMMITTED`). This is **PRE-EXISTING** on
base b351ad80 (verified on a clean detached checkout), NOT introduced by these
retirements. The authoritative gate is the DEFAULT golden `1c04c3fd`, which is
unchanged.

## Remaining P5 −LOC estimate

- **Category A (default-ON fixes), ~56 remaining** at the observed ~26 net
  LOC/flag → **≈ 1,450 LOC** available NOW via the identical mechanical pattern.
  Several (the `C4_LOOP_LEA_*`, `C4_SILI_*`, `C4_TAIL_LEA_E8_*`,
  `C4_MUL_L19_*`, `C4_DIVMOD_*` families) share helper modules and can be batched.
- **Category B (collapse toggles), 6 flags** → ~150 LOC after the collapses are
  declared permanent (needs a geometry-distinguisher check in the cache-key).
- **Cache-key `flag_key` dict** shrinks by one entry per retired flag (already
  −9 entries here); a full Category-A pass removes ~55 more entries (~200 LOC of
  comment+expr).

**Total P5 −LOC available now (safe, default-neutral):** on the order of
**1,600–1,800 LOC** from flag retirement alone — a substantial fraction of the
<8K endgame, entirely golden-`1c04c3fd`-neutral. This session banked −237 and
proved the pattern end-to-end (9 flags, golden unchanged at every step).
