# C4_* FLAG REGISTRY

Canonical inventory of every `C4_*` environment flag consulted at runtime
(`os.environ.get` / `.setdefault` / `os.environ[...]`) across `neural_vm/` and
`tools/`. Generated 2026-07-03 (golden `b4d2ab27...`, campaign is DEFAULT for
the fix fleet). Originally 152 real flags; the I2 dead-flag sweep (2026-07-03)
DELETED 6 default-OFF dead-end held-off building blocks — `C4_POST_ENT_SE_SUPPRESS`,
`C4_ENT_SP_BYTE1_ISMARK_BLOCKER`, `C4_L16_LEV_STACK0_PRESERVE_SE_BLOCKER`,
`C4_PSH_STACK0_BYTE3_RELAY_DARKEN`, `C4_FUNC_LEA_B0_RESTORE`,
`C4_BP_SAVE_DUMP_MARKER_REQ` (all byte-identity-safe, golden `b4d2ab27`
unchanged) — leaving 146. The remaining flags are read via a literal
`environ.get("C4_...")` / `getenv` / `_env_int("C4_...")`, plus 5 read
indirectly via an `_ENV`-name constant (see the STRUCTURAL / RUNTIME table).

> **2026-07 round-1 combo LANDED (default-ON) — 3 of the 4 flags.** Three value
> flags were flipped DEFAULT-ON as the coherent combo:
> `C4_JSR_BP_BYTE3_CLEAR`, `C4_FUNC_ADD_B0_HINIB`, `C4_STORE_AX_B0_OVERRIDE_V2`.
> The 4th, `C4_ABSDIFF_RET_BYTE1`, was **held DEFAULT-OFF**: flipping it ON
> regressed the `test_lea_basic` smoke (its crush-band over-fires on the ENT-frame
> LEA row and zeroes the address); it needs a surgical re-gate first. NEW golden
> (default build, no `C4_*` env) `state_dict_sha256 =
> e50521f32b0ed952d5730f79b63adb8c4c78f4d4f0466d3bcbaa354bb3c90e86` (short
> `e50521f3`). Each keeps a `=0` escape hatch;
> setting all four to `0` reproduces the prior golden
> `f725c06e1ad4d27659817eb46acba9f7aefe5a2545dcb1357baa24a2406fde6c` (short
> `f725c06e` — use for flag-OFF byte-identity checks). All flags are
> campaign-track (gated on `no_stack0_emit`), so the non-campaign / golden 35-token
> build is byte-identical regardless.

**Columns**
- **DEFAULT** — the value read when the flag is unset (the 2nd arg to
  `environ.get`, or the effective boolean of the comparison). `unset→off` means
  `== "1"` style (absent ⇒ False); `1` / `0` means the string default.
- **TRACK**
  - `golden`   — affects the DEFAULT (35-token, non-campaign) production build.
  - `campaign` — only meaningful in the 30-token *campaign* config
    (gated behind `no_stack0_emit_enabled()` / `operand_from_memsp_enabled()`,
    or its own predicate falls through to `no_stack0_emit_enabled()`); golden
    flag-OFF build is byte-identical.
  - `tooling`  — probe / gate / A-B-knob / runtime-perf; not on any weight-bake
    path (model byte-identical regardless).
  - `entry`    — the single campaign-config entry point (`C4_CAMPAIGN`).
- **STATUS**
  - `active`               — live, load-bearing (or a documented default-ON corrector).
  - `vestigial-candidate`  — held-off building block, superseded/refuted fix,
    demo/A-B knob, or sub-knob of a parent flag. **Flagged for I2 review — NOT
    removed here.** (Removing any requires the byte-identity + flag-regression
    gates; several are load-bearing kill-switches for the gates themselves.)

> Coordination note: this registry is ADDITIVE. It does not change any flag
> semantics. The `vestigial-candidate` list is a SUGGESTION for the dead-flag
> sweep (I2) — do not treat it as a delete list without the gates.

---

## THE CAMPAIGN ENTRY POINT

| Flag | Default | Track | Status | Purpose |
|------|---------|-------|--------|---------|
| `C4_CAMPAIGN` | `0` (off) | entry | active | **Single campaign-config switch.** `=1` OR-ins the ON floor for the coherent 30-token campaign set — `no_stack0_emit` + `operand_from_memsp` + `si_store_addr` + `operand_cam_fix` (see `ops/shared.campaign_enabled`), which in turn cascades to every campaign fix that falls through to `no_stack0_emit_enabled()`. Exactly equivalent to setting those explicit flags; an explicit per-flag `=0` still opts out. DEFAULT OFF ⇒ golden build byte-identical. Registered in BOTH cache-key snapshots. |

### The 4 predicates `C4_CAMPAIGN` drives (`ops/shared.py`)

| Flag | Default | Track | Status | Purpose |
|------|---------|-------|--------|---------|
| `C4_NO_STACK0_EMIT` | `1` in `shared`/`0` in probes | campaign | active | Drop the STACK0 register block from the emitted step (35→30 tokens). The campaign master gate; ~90 tools + 5 build sites consult it. `campaign_enabled()` OR-in supplies the ON floor. |
| `C4_OPERAND_FROM_MEMSP` | `1` | campaign | active | Read operand-A from `mem[SP]` instead of the emitted STACK0 token (pairs with `NO_STACK0_EMIT`). ON floor via campaign. |
| `C4_SI_STORE_ADDR` | unset→off | campaign | active | L15 head-16 SI/SC store address-provenance CAM (var_mul / multilocal-LI). Grows L15 16→17 heads. Flipped ON by `C4_CAMPAIGN` **or `C4_DERIVE_MEMORY`**. |
| `C4_DERIVE_MEMORY` | `0` (off) | derive | active | **Umbrella for the derived binary-address-CAM MEMORY-fix path** (BLOG_SPEC §408-412). OR-floors the two clean-CAM memory-fix predicates `si_store_addr` (L15 head-16 store-provenance CAM) + `var_three_li` (L15 head-0 store-row veto); explicit per-flag `=0` still opts out. Byte-identical to `C4_SI_STORE_ADDR=1 C4_VAR_THREE_LI=1` under campaign. DEFAULT OFF ⇒ golden byte-identical (`e50521f3`). Registered in BOTH cache-key snapshots. Meaningful only with `C4_CAMPAIGN=1` (30-token MEM-from-SP path). See `docs/DERIVE_MEMORY_2026_07_09.md`. |
| `C4_LI_VALUE_LOAD` | `0` (off) | campaign | active | **KEEP** — LI value-load fix: the L15 head-16 SI-store-addr CAM null-high-address candidate VETO. `=1` adds one slot-32 K veto on `ADDR_B0_HI+0` so the store-addr CAM fails-CLOSED on the callee ENT-frame PHANTOM row (`ADDR_B0=0x00`) at a CALL-SITE-arg LI where no matching SI store exists (func_max id650 / func_min id675 operand-`b` LI: got AX=0, want 57/99 — head-16 was delivering `a`'s value at value_scale 60, burying head-0's correct value). Genuine relative stores (`0xE_/0xD_` HI nibble) are UNTOUCHED (`ADDR_B0_HI+0`==0 → veto=0), so var_mul/var_three/var_update keep byte-identical value delivery (and var_update id325 steps 9/15 flip fail→ok). Only installs when `C4_SI_STORE_ADDR` is on (head 16 exists). Default-OFF → flag-OFF byte-identical `f725c06e`. See `shared.li_value_load_enabled`. |
| `C4_OPERAND_CAM_FIX` | `0` (off) | campaign | active | Widen the operand-CAM ALU_HI address-leak clear past OP_ADD to SUB/MUL/MOD/DIV/CMP. Gated on `no_stack0_emit`; flipped ON by `C4_CAMPAIGN`. |

---

## GOLDEN-TRACK correctors (default-ON, affect the 35-token build)

These are `!= "0"` default-ON declarative correctors baked into the golden
model. Opt-out with `=0`.

| Flag | Default | Status | Purpose |
|------|---------|--------|---------|
| `C4_AX_BYTE1_DUMP` | `1` | active | AX byte-1 register-dump LM-head columns (H1_DUMP_OUT). |
| `C4_AX_BYTE1_HINIB` | `1` | active | AX byte-1 high-nibble dump (the +11 golden win, n_heads 10→11). |
| `C4_AX_BYTE1_SIGNEXT_LEA` | `1` | active | AX byte-1 sign-extension delivery on negative LEA-local frame addresses (#343). |
| `C4_STACK0_B0_DUMP` | — | REMOVED | STACK0 byte-0 register-dump LM-head columns (Root 2) — the dump machinery was consolidated onto the OUTPUT-canonical path (#362 I1); no `environ.get` reads it anymore. Only prose/comment references remain (`efficient_alu_neural.py`, `full_vm_compiler_dynamic.py`). The stale `tools/_isa_carry_golden.py` `STACK0_ON/OFF` combos + the `test_whole_model_hash_stack0_byte0_migration_byte_identical` gate (both env-gated OFF by default) are now no-ops (setting the flag has no weight effect). |
| `C4_STACK0_NEXT_ARITH` | `1` | active | Consumer-opcode lookahead bands + dump-block gate (#221). |
| `C4_BP_SAVE_DUMP` | `1` | active | BP-save register dump (the `_isa_golden_hash` reference flag). |
| `C4_MUL_WIDTH2` | `1` | active | Width-2 (16-bit) MUL path + MUL_RESULT_HI band (d_model 872→981). |
| `C4_MUL_W2_THRESH_FIX` | `1` | active | Width-2 MUL threshold correction. |
| `C4_DIV_MULTIBYTE` | `1` | active | Multi-byte DIV/MOD result path. |
| `C4_MUL_MULTIPASS` | `0` (off) | active | GAP-PRIMITIVE #2: replace the L11 mul-partial / L12 mul-combine lookup with the 7-pass schoolbook `multi_pass_mul_rules` cascade (one block, `MUL_MULTIPASS_WS` band). DEFAULT OFF ⇒ golden byte-identical. Registered in BOTH cache-key snapshots. See `ops/shared.mul_multipass_enabled`. |
| `C4_DIV_MULTIPASS` | `0` (off) | active | GAP-PRIMITIVE #2 (DIV): replace the L10 `FlattenedDivMod` composite with the 43-pass binary long-division `multi_pass_div_rules` cascade (`MultiPassDivBlock`, one post_op; `DIV_MULTIPASS_WS` + `DIV_MP_*` bands). Computes `a//b`+`a%b` from a compact bit-serial shift-subtract spec, routes q→OUTPUT (DIV) / r→OUTPUT (MOD) at MARK_AX. **NaN-on-real-residual FIXED (2026-07):** `MultiPassDivBlock.forward` now clean-onehot-clamps the operand bands (dividend `ALU_LO/HI`, divisor `AX_CARRY_LO/HI`) to the cascade's residual==1.0 seed invariant on the div/mod-AX rows before the passes run — mirroring the composite's `GEToBDConverter._clean_onehot`. Without it the real max_abs≈72 operand residual seeds the amplitude-normalized cascade off its fixed point → running residual explodes ~1.5e37 → inf → all-NaN block output → frame collapse. Proven (isolated-block, `tools/_div_multipass_isolated_block.py`): raw pipeline all-NaN on the dirty input; clamped block 0 NaN + divmod correct over a 3328-dividend×13-divisor×2-op dirty sweep. DEFAULT OFF ⇒ golden byte-identical (`b1dcae63`). Registered in BOTH cache-key snapshots. See `ops/shared.div_multipass_enabled`. |
| `C4_SUB_FULL_BORROW` | `1` | active | Full SUB borrow-cascade. |
| `C4_ADDSUB_DUMP_BOOST` | `1` | active | Imperative AddSub byte-0 OUTPUT dominant-amplitude write. |
| `C4_CMP_FLAG_MARGIN_FIX` | `1` | active | CMP equal-high-nibble GT flag margin (if_gt 357/359/361). |
| `C4_CMP_EQ_HINIB_VETO` | `1` | active | CMP EQ high-nibble veto. |
| `C4_CMP_GT_LO_MARGIN` | `1` | active | CMP GT low-byte margin. |
| `C4_CMP_HI_LT_ALU15_GUARD` | `1` | active | CMP hi-byte LT ALU15 guard. |
| `C4_CMP_GT_LO_LT_HIEQ_GUARD` | `1` | active | CMP GT-lo / LT-hi-eq guard. |
| `C4_CMP_BYTE0_SE_RECOVER` | `1` | active | CMP byte-0 SE-recover. |
| `C4_BITWISE_BYTE0_SE_RECOVER` | — | **REMOVED (2026-07-13)** | `BitwiseOperandSeRecoverFFN` deleted (−130 LOC). Its job (re-materialise the crushed bitwise operand-A at the L10 lookup block) is subsumed by delivering a CLEAN operand-A one-hot UPSTREAM at L8 via `C4_CLEAN_OPERAND_BITWISE` (default-ON). Class-B removal: not a free delete, required extending the clean-operand delivery to the bitwise opcodes. |
| `C4_CLEAN_OPERAND_BITWISE` | `1` (campaign-ON) | active | **DEFAULT-ON, opt out `=0`.** ADDS `OP_AND/OP_OR/OP_XOR` to the `CleanOperandOneHotFFN` op_dims at the L8 main FFN (physical block 12) so the bitwise MARK_AX rows also get the clean per-nibble one-hot snap. SUBSUMES the deleted `BitwiseOperandSeRecoverFFN`: the L9 non-ALU ALU-scrubber (physical block 21) subtracts a FIXED pattern (~5.56 @ cell 0) from the ALU band; the dirty operand-gather hybrid's A==0 nibble at ~+5.48 → 5.48−5.56 = −0.08 goes negative (lost); the clean snap's +6.0 → 6.0−5.56 = +0.44 stays positive and survives to the lookup, which fires the rescaled `bitwise_rules` (a+b−ab, `operand_a_cw = 30/5.82`) correctly. All 6 bitwise smoke programs pass with the SeRecover removed (spec_k=0 hook-free probe). Gated on `no_stack0_emit`; flag-OFF / non-campaign byte-identical golden `e50521f3` (the wrap adds no params → state_dict hash unchanged). Registered in both cache-key snapshots (module-affecting). See `shared.clean_operand_bitwise_enabled`. |
| `C4_DIVMOD_BYTE0_SE_RECOVER` | `1` | active | DIV/MOD byte-0 SE-recover. |
| `C4_DIVMOD_AXCARRY_CLEAR` | `1` | active | DIV/MOD AX-carry clear. |
| `C4_DIVMOD_STACK0_BYTE1_CLEAR` | `1` | active | DIV/MOD STACK0 byte-1 clear. |
| `C4_MUL_BYTE0_SE_RECOVER` | `1` | active | MUL byte-0 SE-recover. |
| `C4_MUL_L11_SE_RECOVER` | `1` | active | MUL L11 SE-recover. |
| `C4_MUL_L19_FLOOD_CAP` | `1` | active | MUL L19 flood cap. |
| `C4_MUL_L19_PRODUCT_BOOST` | `1` | active | MUL L19 product boost. |
| `C4_MUL_MULTIBYTE_L19_BOOST` | `1` | active | MUL multi-byte L19 boost. |
| `C4_MUL_SE_RECOVER_STRICT_ONEHOT` | `1` | active | MUL SE-recover strict one-hot. |
| `C4_SHIFT_OUTPUT_B0_CLEAR` | `1` | active | SHL/SHR OUTPUT byte-0 clear. |
| `C4_LEV_AX_BYTE1_KILL` | `1` | active | LEV (func-return) AX byte-1 stale-carry dump kill. |
| `C4_PSH_STACK0_HIGHBYTE_DARKEN` | `1` | active | PSH STACK0 high-byte darken (var full_trace advance). |
| `C4_STACK0_POP_LOADED_SHALLOW_CRUSH` | `1` | active | STACK0 pop-loaded shallow crush. |
| `C4_SP_POP_CARRY_BYTE0_DOMINATE` | `1` | active | SP-pop carry byte-0 dominate. |
| `C4_NONFIRST_PSH_SP_FIX` | `1` | active | Non-first PSH SP fix. |
| `C4_ENT_FRAME_SP_BYTE0_PSH_BLOCKER` | `1` | active | ENT-frame SP byte-0 PSH blocker. |
| `C4_ENT_SP_BYTE1_FF_H1_HARDEN` | `1` | active | Post-ENT SP-byte1 H1+2 hardening. |
| `C4_L8_ADJ_LO_AX_MARKER_BLOCKER` | `1` (`==`) | active | L8 ADJ-lo AX-marker blocker (func_identity low-nibble corruption). |
| `C4_L8_OPERAND_SP_DISC` | `1` | active | L8 operand SP discriminator. |
| `C4_L10_ENT_AXCARRY` | `1` | active | L10 ENT AX-carry (also requires `no_stack0_emit`). |
| `C4_L10_EXIT_AXCARRY` | `1` (`==`) | active | L10 EXIT AX-carry. |
| `C4_L16_LEV_PC_TOP_OPCODE_GATE` | `1` (`==`) | active | L16 LEV PC-top opcode gate. |
| `C4_MEM_MARKER_OUTPUT_CLEAR` | — | REMOVED | L0 MEM-marker OUTPUT clear — DELETED (subsumed by `C4_CLEAN_EMITTER`'s 6-way NEXT_* sink; the `no_stack0_mem_marker_output_clear` op is gone). |
| `C4_CLEAN_EMITTER` | `1` (on) | active | L0 generic all-marker-row OUTPUT sink (the ≠STEP_TOKENS framing megaroot fix). Gated on the 6-way OR of `NEXT_PC/AX/SP/BP/MEM/SE` + `no_stack0_emit`; sinks OUTPUT at EVERY marker row so the marker wins by construction. PROVEN net-positive on the full 1096 (516 → 525, +9) so now DEFAULT ON; because `C4_NO_STACK0_EMIT` also defaults ON the bare-env golden changed to `b1dcae63` (`C4_CLEAN_EMITTER=0` reproduces `91f55411`). SUBSUMES + REPLACES the two now-DELETED point-fixes (`no_stack0_se_output_clear` + `no_stack0_mem_marker_output_clear`). See `docs/CLEAN_EMITTER_SCOPE_2026_07_04.md`. |
| `C4_TAIL_LEA_E8_ARITH_GUARD` | `1` | active | Tail LEA-E8 arith guard. |
| `C4_TAIL_LEA_E8_DIVMOD_GUARD` | `1` | active | Tail LEA-E8 DIV/MOD guard. |
| `C4_TAIL_LEA_E8_ENT_GUARD` | `1` | active | Tail LEA-E8 ENT guard (also requires `no_stack0_emit`). |
| `C4_NESTED_JSR_PC_FIX` | `1` | active | Nested JSR PC fix. |
| `C4_INC3_H5_DIM0_CLEAN` | `1` | active | L8 head-5 dim-0 clean (Inc-3). |
| `C4_EMIT_G5_RBYTE` | `0` (off) | active | **BYTE-NEUTRAL verification toggle** for the EMIT-G5 4→1 `layer14_{jsr,lc,alu_nocarry,ent}_ax_bytes_zero` fold (`l14_ops._make_ax_bytes_zero_op`). When `=1` each op's 4-unit rule program is rebuilt through a SECOND independent `_ax_bytes_zero_rules(spec, S)` call and asserted structurally identical before lowering — a distinct code path that exercises the spec table without touching a weight. DEFAULT OFF ⇒ golden byte-identical (`e50521f3`); `=1` ALSO byte-identical (`e50521f3`). Registered in BOTH cache-key snapshots so the ON/OFF builds never share a cache entry during the golden check. See `docs/EMIT_G5_ROLLOUT_2026_07_13.md`. |

---

## CAMPAIGN-TRACK fixes (30-token config; golden byte-identical OFF)

Most fall through to `no_stack0_emit_enabled()` when their own env is unset, so
`C4_CAMPAIGN=1` activates them automatically; each keeps its own kill-switch for
the flag-regression / cross-op gates.

| Flag | Default | Status | Purpose |
|------|---------|--------|---------|
| `C4_LOADED_OPERAND_ADD_HI15_CLEAR` | `1` (campaign) | active | Loaded-operand ADD hi-nibble cell-15 address-leak clear (var_update). |
| `C4_FUNCADD_ALU_HI13_CLEAR` | `1` (campaign) | active | Loaded-operand ADD hi-nibble cell-13 leak clear (func_add/mul/max/min). |
| `C4_FUNC_ADD_B0_HINIB` | `1` (default ON) | active | **round-1 combo, DEFAULT-ON (opt out `=0`).** func-return ADD byte-0 hi-nibble over-count fix: widen the loaded-operand ADD ALU_HI clear to ALL 16 cells (magnitude-windowed) to zero the ~1.0 operand-B-high-nibble bleed while preserving the ~6.0 true operand. Gated on `no_stack0_emit` (campaign); flag-OFF / non-campaign byte-identical `f725c06e`. |
| `C4_CLEAN_OPERAND_ADD` | `1` (default ON) | active | **CBC first correct-by-construction pass-gain, DEFAULT-ON (opt out `=0`).** Arithmetic-only clean-one-hot operand delivery: `CleanOperandOneHotFFN` wrap on the L8 main FFN (physical block 12) snaps the ALU_LO/HI (operand A) + AX_CARRY_LO/HI (operand B) bands to a clean per-nibble one-hot on the `OP_ADD/SUB/MUL/DIV/MOD` MARK_AX rows ONLY (the six CMP dims are absent from `op_dims` so the CMP calibration contract is untouched → zero CMP regression). Full-1096 `full_trace --spec-k 0 --max-steps-cap 40`: OFF 580 → ON 593 (**+13**). Gated on `no_stack0_emit` (campaign); flag-OFF / non-campaign byte-identical golden `e50521f3` (the wrap adds no params and is never installed off the efficient-ALU campaign build → state_dict hash unchanged). See `shared.clean_operand_add_enabled`. |
| `C4_CLEAN_OPERAND` | `0` (feasibility) | active | DEFAULT-OFF feasibility flag (opt in `=1`). SAME `CleanOperandOneHotFFN` wrap but cleans ALL 11 consumers (`OP_ADD/SUB/MUL/DIV/MOD` **+** `OP_EQ/NE/LT/GT/LE/GE`). Predicts −74: clean operands help arithmetic (+6) but break the CMP path (the CMP nibble-comparator −0.5/−0.8 per-nibble blockers are a magnitude contract calibrated to the dirty hybrid). Kept as the feasibility baseline; the arithmetic-only slice (`C4_CLEAN_OPERAND_ADD`) is the shipping default. Registered in both cache-key snapshots (module-affecting). See `shared.clean_operand_enabled`. |
| `C4_SILI_CAM_B1` | `1` (campaign) | active | SI/LI load byte-1 address-leak discriminator (Inc-2). |
| `C4_SILI_B1_RESTORE` | `1` (campaign) | active | SI/LI byte-1 restore. |
| `C4_STORE_AX_B0_OVERRIDE` | `1` in build / `0` in shared | active | Store AX byte-0 override (un-discriminated; reverted OFF in shared 7869e5c3, -24 arith). |
| `C4_STORE_AX_B0_OVERRIDE_V2` | `1` (default ON) | active | **round-1 combo, DEFAULT-ON (opt out `=0`).** Store AX byte-0 override, clean store-only discriminator (ALU/cmp opcode anti-conditions). Fixes var_three/var_mul SI-store step; add/sub untouched. Gated behind the two campaign flags; flag-OFF / non-campaign byte-identical `f725c06e`. |
| `C4_PSH_ARG_VAL_AX` | `1` | active | PSH-of-argument value-source AX lock (call-arg store). |
| `C4_L15_LI_SUPPR_INERT` | `1` | active | L15 head-0 LI/LC-load suppressor inert (func/nested/rec/var LI). |
| `C4_L15_LI_ADDR_CAM` | campaign | active | L15 head-0 LI value-load ADDR_B0 CAM (#313). |
| `C4_L15_LI_B0_VALSEL` | campaign | active | L15 head-0 byte-0 VALUE-row selector via MEM_VAL_B0 (#318). |
| `C4_L15_LI_VALROW_B1` | campaign | active | L15 head-0 first-param LI value-row lift via MEM_VAL_B1. |
| `C4_L15_LI_ZEROADDR_CAM` | campaign | active | L15 head-0 zero-address committed-store CAM. |
| `C4_L15_LI_JSR_PHANTOM` | campaign | active | L15 head-0 JSR-phantom row penalty. |
| `C4_SCLC_LC_B0` | campaign | active | SC/LC byte-0 boost onto OP_LC (sc_lc roundtrip). |
| `C4_L15_LEV_PC_RESTORE` | `1` | active | L15 head-14 LEV PC-restore (return-address CAM); L15 14→15 heads. |
| `C4_L15_LEV_ADDR_WIDEN` | `1` | active | L15 LEV address-widen on head 14 (+ sub-knobs below). |
| `C4_L15_LEV_PC_ONLY` | `1` | active | L15 LEV PC-only restore (sub-knob of ADDR_WIDEN). |
| `C4_L15_LEV_OPCODE_GATE` | `1` | active | L15 LEV opcode gate. |
| `C4_L15_LOOKUP_CMP_VETO` | `1` | active | L15 lookup CMP veto. |
| `C4_LEA_E8_FIRST_ENT_GATE` | `1` | active | LEA-E8 first-ENT gate (campaign LEA byte-0). |
| `C4_LOOP_LEA_B0_E0` | `1` | active | Loop LEA byte-0 E0. |
| `C4_LOOP_LEA_B0_E8` | `1` | active | Loop LEA byte-0 E8. |
| `C4_LOOP_LEA_B0_E8_OPLEA_REQ` | `1` | active | Loop LEA byte-0 E8 OP_LEA requirement. |
| `C4_LOOP_LEA_OPLEA_GATE` | `0` | active | Loop LEA byte-0 E8/E0 MULTIPLICATIVE OP_LEA gate (PROJECT_0XE8_SLAM Phase-2; clears the if_gt/if_eq IMM-row false-fire). |
| `C4_LOOP_SI_BYTEROW_CLEAR` | `1` | active | Loop SI byte-row clear. |
| `C4_LOOP_LI_FETCH_ADDRKEY_CLAMP` | `1` | active | Loop LI-fetch address-key clamp. |
| `C4_IFVAR_BZ_HI_NIBBLE` | `1` | active | if_var BZ high-nibble. |
| `C4_FUNC_LEA_REREAD_BP_RESHARPEN` | `1` (campaign) | active | L7 head-1 re-read-LEA BP-frame re-sharpen (func_add/mul/square/max/min). |
| `C4_JSR_PC_BYTE1_EMIT_WS` | `2e5` | active | JSR PC byte-1 emit write-scale. |
| `C4_JSR_SEQ_WS` | `30.0` | active | JSR sequence write-scale. |
| `C4_AX_B1_HINIB_WRITE` | `6.0` | active | AX byte-1 hi-nibble write strength. |
| `C4_BP_SAVE_DUMP_WS` | `200000.0` | active | BP-save dump write-scale. |
| `C4_STACK0_B0_REPOINT_WS` | `70.0` | active | STACK0 byte-0 repoint write-scale. |
| `C4_L15_LEV_B0_BOOST` | `8` | active | L15 LEV byte-0 boost (sub-knob). |
| `C4_L15_LEV_JSR_DISC` | `100` | active | L15 LEV JSR discriminator (sub-knob). |
| `C4_L15_LEV_BYTE0_SELECT` | `400` | active | L15 LEV byte-0 select weight (sub-knob). |

---

## STRUCTURAL / RUNTIME toggles (golden or tooling; not correctors)

| Flag | Default | Track | Status | Purpose |
|------|---------|-------|--------|---------|
| `C4_QWEN_EXPORT_COMPAT` | unset→off | golden | active | Qwen-export dim layout (R1). |
| `C4_DISABLE_WRAPPER_EXPANSION` | unset→off | golden | active | Skip `_expand_wrapper_blocks` (37→27 blocks). |
| `C4_DISABLE_AX_CARRY_BANDS` | unset→off | golden | active | Drop AX-carry residual bands. |
| `C4_DISABLE_SLOT_REGISTRY` | unset→off | golden | active | Disable slot registry. |
| `C4_EXTRA_RESIDUAL_DIMS` | `""` | golden | active | Explicit extra residual band request (legacy; superseded by `residual_band_registry`). |
| `C4_ENABLE_MOE_ROUTING` / `C4_BATCH_ENABLE_MOE_ROUTING` | unset→off | tooling | active | MoE routing (batched runner). |
| `C4_CSR_INFERENCE` | `1` | tooling | active | CSR-sparse inference kernels. |
| `C4_COMPACT_GATHER` | `""` | tooling | active | Compact-gather kernel. |
| `C4_VM_CACHE_DIR` | unset | tooling | active | Disk cache directory override. |
| `C4_BATCH_USE_KV_CACHE` | unset→off | tooling | active | KV cache in batched runner. |
| `C4_BATCH_KV_VERIFY` | `0` | tooling | active | KV-cache verify. |
| `C4_BATCH_KV_MAX_TOKENS` | unset | tooling | active | KV-cache max tokens. |
| `C4_BATCH_KV_VERIFY_INTERVAL` | `32`/`1` | tooling | active | KV-cache verify interval (read via `_env_int`, `batched_pure_neural.py`). |
| `C4_BATCH_KV_FLUSH_INTERVAL` | `0` | tooling | active | KV-cache flush interval (read via `_env_int`). |
| `C4_BATCH_KV_EVICTION_OVERSHOOT` | `Token.STEP_TOKENS` | tooling | active | KV-cache eviction overshoot (read via `_env_int`). |
| `C4_ADAPTIVE_START_K` / `C4_ADAPTIVE_MIN_K` / `C4_ADAPTIVE_MAX_K` | `32`/`1`/`64` | tooling | active | Adaptive speculative-K bounds (read via `_env_int`, `batched_pure_neural.py`). |
| `C4_VM_CACHE_MAX_BYTES` / `C4_VM_CACHE_MAX_ENTRIES` | `10 GiB`/`16` | tooling | active | Disk-cache LRU size limits (read via `_CACHE_MAX_*_ENV`, `_legacy_redirect.py`). |
| `C4_DIM_LIVENESS` | `1` | tooling | active | Dim-liveness pruning (read via `_DIM_LIVENESS_ENV`, `layer_compiler.py`); `=0` bisects a liveness bug. |
| `C4_STRICT_GATE_CHECK` | unset→off | tooling | active | DSL gate-audit strict mode — raise instead of warn (read via `GATE_AUDIT_STRICT_ENV`, `dsl_interpreter.py`). |
| `C4_REQUIRE_DECLARATIVE_BAKE` | unset→off | tooling | active | Require declarative-only bake, error on imperative (read via `_REQUIRE_DECLARATIVE_BAKE_ENV`, `_legacy_redirect.py`). |
| `C4_BATCH_FORCE_INCREMENTAL_KV` | `""` | tooling | active | Force incremental KV. |
| `C4_BATCH_CHUNK` | `0`/`32` | tooling | active | Batch chunk size (runner CLI). |
| `C4_BATCH_CONTEXT_WINDOW` | `512` | tooling | active | Context window (runner CLI). |
| `C4_BATCH_MODEL_MAX_SEQ_LEN` | `4096` | tooling | active | Model max seq len (runner CLI). |
| `C4_SPEC_K` / `C4_TEST_SPEC_K` / `C4_SMOKE_SPEC_K` | `0`/`4` | tooling | active | Speculative-decode K (probe/smoke default 0 = ground truth). |
| `C4_SPEC_FAIL_FAST` | `0` | tooling | active | Speculative fail-fast. |
| `C4_SPEC_FAIL_ON_CORRECTION` | `""` | tooling | active | Speculative fail on correction. |
| `C4_SKIP_DIM_INTEGRITY` | set by tools | tooling | active | Skip dim-integrity check (lints set it). |
| `C4_SKIP_GATE_CHECK` | set by tools | tooling | active | Skip DSL gate check. |
| `C4_DECLARATIONS_ONLY_BAKE` | set by tools | tooling | active | Declarations-only bake. |
| `C4_1096_LIMIT` | unset | tooling | active | 1096-corpus limit (observe tool). |
| `C4_FETCH_BLOCK` | `6` | tooling | active | Fetch-block index (loopsum probe). |
| `C4_FORCE_WIDEN_DMODEL` / `C4_FORCE_WIDEN_NHEADS` | set by probe | tooling | active | Force d_model / n_heads widen (widen probes). |
| `C4_SP_DEEP_FRAME_DEPTH_TRACK` | unset | tooling | active | SP deep-frame depth track (probe). |
| `C4_SP_DISC_G` | `420` | tooling | vestigial-candidate | L8 SP-disc A/B gain knob (comment: "A/B knob (tooling)"). |

---

## VESTIGIAL-CANDIDATES (I2 dead-flag sweep — resolved 2026-07-03)

The dead-flag sweep classified all 28 candidates. **DELETED** entries were
default-OFF **dead-end** held-off building blocks (a documented net-zero /
net-negative result, never enabled by `C4_CAMPAIGN` or any live config, no
test/gate consuming them); each removal was byte-identity gated
(`_isa_golden_hash.py` == `b4d2ab27`, since the op never baked in golden) and
committed one-per-commit. **KEEP** entries are load-bearing gate fixtures, A/B
override knobs / kill-switches on **live campaign correctors** (they fall
through to a live default when unset — deleting the flag would delete the
corrector), or verified real fixes held for an imminent landing. **BORDERLINE**
entries were left for review (large multi-file features or active-megaroot
building blocks — conservative hold).

| Flag | Default | Track | Verdict — reason |
|------|---------|-------|------------------|
| `C4_MY_FIX` | n/a | tooling | **KEEP** — not a real runtime flag; documented `--flag` convention in the gate docstring. |
| `C4_FFN_LINT_MULL14_DEMO` | `0` (off) | tooling | **KEEP** — `lint_cross_op_ffn --demo` fixture (protected). |
| `C4_FFN_LINT_CLEAN_DEMO` | `0` (off) | tooling | **KEEP** — `lint_cross_op_ffn --demo` fixture (protected). |
| `C4_OUTBAND_DECOUPLE_PROTO` | `""` (`=="decouple"`) | campaign | **KEEP** — active OUTPUT-band megaroot WIP (protected). |
| `C4_MUL_BLK33_CLAWBACK` | unset→off | campaign | **KEEP** — flag-regression-gate proof fixture (protected). |
| `C4_MUL_STACK0_BYTE39_GUARD` | `0` (off) | campaign | **KEEP (load-bearing real fix)** — commit `4cadc90e`: mul cluster 46→50/50, +15 flips, 0 regressions, smoke 51/0. A verified fix held DEFAULT-OFF pending campaign integration, NOT a dead-end. |
| `C4_LEA_LOCAL_E8_MULTILOCAL_GUARD` | tracks `no_stack0_emit` | campaign | **KEEP (live campaign corrector)** — default = `no_stack0_emit_enabled()` (ON in campaign); the GPU-confirmed per-step LEA disambiguator (var_mul step-6). The registry's "0 verdict change" was stale. |
| `C4_INC3_MEMAX_B1_OFF` | `0` (`==0` kill-switch) | campaign | **KEEP (kill-switch)** — A/B kill-switch for the campaign-default-ON Inc-3 MEMAX-B1 corrector. |
| `C4_STACK0_MARKER_ISBYTE_HARDEN` | tracks `no_stack0_emit` | campaign | **KEEP (A/B knob on live corrector)** — Inc-3 ROOT-B, ON in campaign; `forced` override else `no_stack0_emit_enabled()`. |
| `C4_STACK0_MARKER_AXMARK_HARDEN` | tracks `no_stack0_emit` | campaign | **KEEP (A/B knob on live corrector)** — Inc-3 ROOT-A (~140-prog lever), ON in campaign. |
| `C4_TAIL_LEA_E8_ARITH_GUARD_SHARP` | forced→True | campaign | **KEEP (A/B knob on live corrector)** — shapes the default-ON golden `C4_TAIL_LEA_E8_ARITH_GUARD`. |
| `C4_LEA_E0D8_FETCH_DOMINATE` | forced→live | campaign | **KEEP (A/B knob on live corrector)** — falls through to `_lea_byte0_memsp_relay_enabled()`. |
| `C4_LEA_BYTE0_MEMSP_RELAY` | forced→live | campaign | **KEEP (A/B knob on live corrector)**. |
| `C4_LEA_BYTE0_ALU_AMPLIFY` | forced→live | campaign | **KEEP (A/B knob on live corrector)** — the frame-depth LEA byte-0 ALU-amplifier kill-switch; overrides the `C4_OPCAM_FRAME` umbrella when set explicitly. |
| `C4_OPCAM_FRAME` | `0` (off) | campaign | **KEEP** — operand-CAM frame-depth umbrella; `=1` turns on the L10 multi-param LEA byte-0 ALU-amplifier (survey R2: func_add/mul/max/min + absdiff 2nd-param `&b` 0xFFE8→0xFFE0). Default-OFF → flag-OFF byte-identical `91f55411`. `C4_LEA_BYTE0_ALU_AMPLIFY` overrides it. |
| `C4_ABSDIFF_RET_BYTE1` | `0` (off) | campaign | **HELD DEFAULT-OFF (round-1 land: ON regressed `test_lea_basic`).** absdiff/func-return AX byte-1 OUTPUT_LO stale-marker de-contamination. `=1` adds two L10-tail post_ops (a bounded `ABSDIFF_RET_LEAK` l6-crush-band flag + the corrector) that force the spurious OUTPUT_LO high-byte nibble to 0 on the AX byte rows of the LEV-return step (OP_LEV≥0.95 selector), fixing the step-22 (ADJ-after-LEV) 0x01 leak on all 25 absdiff. When flipped default-ON it regressed the `test_lea_basic` smoke (`ENT; IMM 0; LEA 2; EXIT` → address decodes 0): the crush-band / OP_LEV-return selector over-fires on the ENT-frame LEA row and zeroes the real address OUTPUT_LO nibble. Needs a surgical re-gate (exclude non-LEV LEA rows) before it can go default-ON. Default-OFF → flag-OFF byte-identical `f725c06e`. See `shared.absdiff_ret_byte1_enabled`. |
| `C4_JSR_BP_BYTE3_CLEAR` | `1` (default ON) | campaign | **round-1 combo, DEFAULT-ON (opt out `=0`).** func/var/loop/gcd/nested step-0 JSR-step BP byte-3 = 0x00 CLEAR. Appends an L10-tail `PureFFN` post_op that WTA-forces the JSR-step BP byte-3 predictor row (`OP_JSR` + `H1+3` + `BYTE_INDEX_2`) to nibble 0, killing the `0x01010000` BP corruption that flat-diverges the whole func cluster. Requires the campaign config; flag-OFF / non-campaign registers NO rules → byte-identical `f725c06e`. Registered in BOTH cache-key snapshots. See `shared.jsr_bp_byte3_clear_enabled`. |
| `C4_L15_SAVEDRA_HEAD` | forced→live | campaign | **KEEP (A/B knob on live corrector)** — L15 head-15 saved-RA delivery. |
| `C4_POST_ENT_SE_SUPPRESS` | `0` (off) | campaign | **DELETED** (commit `2043ff4d`) — dead-end framing building block. |
| `C4_ENT_SP_BYTE1_ISMARK_BLOCKER` | `0` (off) | campaign | **DELETED** (commit `10d7b876`) — docstring records NEGATIVE RESULT (0 programs advance; "the real fix is the project-level multi-part build, not a solo corrector"). |
| `C4_L16_LEV_STACK0_PRESERVE_SE_BLOCKER` | `0` (off) | campaign | **DELETED** (commit `1466958c`) — dead-end framing MARK_SE NOT-blocker. |
| `C4_PSH_STACK0_BYTE3_RELAY_DARKEN` | `0` (off) | campaign | **DELETED** (commit `4267cfb3`) — net −1 exit_code TRADE with a documented "cannot be cleanly separated func-vs-var" wall. |
| `C4_FUNC_LEA_B0_RESTORE` | `0` (off) | campaign | **DELETED** (commit `ad2a3f25`) — ZERO-SUM (flips func_add, regresses func_identity; the two LEA rows are bit-identical → not separable). Removed the band + capture/restore ops. |
| `C4_BP_SAVE_DUMP_MARKER_REQ` | `0` (off) | campaign | **DELETED** (commit `67f536e4`) — dead-end BP-dump gate-rewrite variant. |
| `C4_ADDSUB_DECLARATIVE` | `0` (off) | golden | **BORDERLINE (KEEP)** — selects the in-flight `DeclarativeAddSubBlock` migration path (docs/ADDSUB_DSL_MIGRATION_*, probe tool). Migration WIP, not a dead corrector. |
| `C4_AX_BYTE1_FULL_WIDTH` | `0` (off) | golden | **BORDERLINE (review)** — alternate full-width AX byte-1 structural variant (17 refs, band + LM-head columns). |
| `C4_AX_HIBYTE_CLEAR` | `0` (off) | golden | **KEEP (if_var GT-FALSE AX byte-2 lever)** — l11 `make_ax_hibyte_clear_allstep_op`: all-step register byte-2/3 zero-default (fires on every `IS_BYTE`+`BYTE_INDEX_1/2` dump row) + omits the redundant `H*_DUMP_OUT` LM-head columns in `model_ops.py`. **Cache-key BUG FIXED (this session):** the flag was NOT in either build snapshot in `full_vm_compiler_dynamic.py`, so a per-process flip returned a STALE cached model — now registered in BOTH snapshots (byte-identity-safe: default golden `e50521f3` + flags-OFF `f725c06e` unchanged). VERIFIED (probe_ifbool_step1_vcorr, campaign config, if_var GT-FALSE ids 426/430/431): flips the step-11 BZ `AX[2] 0x00→0x01` + step-12 IMM `AX[2] 0x00→0x05` verdict-moving leak OFF→gone, leaving only the guard-rail-BENIGN SP/BP marker re-anchors. The weaker of the two byte-2/3 clamps (see `C4_LOOP_AX_BYTE3_CAP` for the stronger loop variant that beats a +12.5 competitor this one cannot); sufficient for the if_var BZ/IMM steps. Flag-ON standalone golden hash `317e1cba` (+2 FFN units). Smoke + full cpu_full_trace verdict DEFERRED (machine saturation this session — re-verify on an uncontended CPU per the smoke-contention note). |
| `C4_LOOP_AX_BYTE3_CAP` | `0` (off) | campaign | **KEEP (verdict-advancing loop high-byte clamp)** — commit `47254efa` (l11 `make_loop_ax_byte3_cap_op`): STRONG clamp of the AX high bytes (byte-2 = `BYTE_INDEX_1` row, byte-3 = `BYTE_INDEX_2` row) → 0x00 on the register-dump path, killing the loop_mul/countdown/pow2 stale high-byte leak. Kills every non-zero OUTPUT nibble (esp. `+15`==0xF) + boosts the 0x00 nibbles, so it beats the +12.5 competitor the weaker `C4_AX_HIBYTE_CLEAR` could not. Flag-OFF byte-identical to golden `91f55411`. VERIFIED (cpu_full_trace, loop_pow2_2 id527): advances the divergence step 4 → 19 (step-4 byte-2 0xFF leak fixed; residual step-19 leak is a SEPARATE low-byte 0xFFE8 root in the l8/l10 operand path, out of scope). Multi-byte-result safe (mul_0 0x04D7 / mul_11 0x1A90 byte-identical, byte-1 untouched); `AX_CARRY_OVERFLOW` −1000 blocker preserves genuine ≥0x1000000. |
| `C4_STACK0_B0_POPPED` | `0` (off) | campaign | **BORDERLINE (review)** — band + L9 latch head + dump condition; part of the ACTIVE if/bool/expr OUTPUT-band megaroot family. Held pending that multi-part fix. |
| `C4_SP_POP_MARKER_CMP3_HARDGATE` | `0` (off) | campaign | **BORDERLINE (review)** — verified SP-drift fix, currently net −2 (if_var 13→11) but a staged prerequisite that "should net positive the moment the LI value-load root lands" (#313/#289). |
| `C4_JSR_PC_BYTE1` | `0` (off) | campaign | **BORDERLINE (review)** — large multi-file feature (bands + relay head + allocator + staging/emit FFN across `model_ops`/`l3_ops`/`all_core_ops`); high blast-radius, conservative hold. |
| `C4_JSR_AX_CLEAN` | `0` (off) | campaign | **VERDICT-MOVER (review)** — task #350/#422: cleans the JSR step-0 AX-passthrough leak that corrupts the gcd(50)+rec_fib(25)+rec_power(25) step-0 AX register (0xF7F7F7.. in test_jsr_then_lev_simple). Appends a 32-unit matched negative-clear band to `_function_call_jsr_ax_passthrough_rules` (`model_ops.py`) that subtracts the passthrough's `AX_CARRY→OUTPUT` copy back to zero on the JSR step, so the program-entry `JSR main` (bytecode idx 0, the only JSR with an undefined WEAK AX_CARRY the passthrough amplifies to a confident garbage OUTPUT) no longer deposits the leaked index. **SURGICAL (task #422):** the clean band is now `HAS_SE`-gated (`-10.0` first-step penalty) so it fires ONLY on the entry `JSR main` (`HAS_SE≈0`) and NOT on nested/recursive JSRs (`HAS_SE≈0.99`), where the caller AX is a REAL value the passthrough must preserve — the pre-narrowing blanket cancel zeroed the nested caller AX (e.g. rec_sum's `n=0x0e`) and broke passing func/rec/nested programs (the flag-ON −65 regression). Probe-verified (spec_k=0, block 8): entry OUTPUT mag 3.6/4.3→1.8/2.1 (cleaned), nested `0x0e` OUTPUT 7.97/10.30 byte-identical ON vs OFF (preserved). ABI-safe: a call replaces AX with the callee's RETURN value, not the caller's pre-call AX. Prerequisite that makes `C4_JSR_PC_BYTE1` net-positive (survey PC-byte1 alone measured −23 because of this leak). Flag-OFF byte-identical to golden `91f55411`. See `_function_call_jsr_ax_clean_rules` / `_jsr_ax_clean_enabled` in `model_ops.py`; verify with `tools/_probe_jsr_clean_onoff.py`. |
| `C4_SP_BYTE2_CARRY` | `0` (off) | any | **CONVERGENT (merge candidate)** — collapses the `sp_pop_carry_rules` byte-2 pop-carry enumeration from `range(256)` (256 FFN units, `tail_sp_pop_carry_byte2_00..ff`) to `range(2)` (`old in {0x00,0x01}`), deleting 254 provably-dead FFN units (model 43071→42817 units). SP stays in `[0x0FE10,0x10000]` corpus-wide so byte-2 is only ever 0x00/0x01 and only `0x00→0x01` fires; the 2 kept rules are byte-for-byte identical to the OFF bank members. Flag-OFF byte-identical to golden `b4d2ab27`; flag-ON golden `716a66c1`. See `_sp_byte2_carry_computed_enabled` in `l10_ops.py`. |
| `C4_STACK0_STORE_LOADED_COMPUTED` | `0` (off) | any | **CONVERGENT (merge candidate — M8 enumerated→computed pilot)** — replaces the `stack0_store_loaded_output_rules` byte-writeback bank (255 per-VALUE AND units, `tail_stack0_store_loaded_byte_00..ff`) with a COMPUTED per-NIBBLE route (32 units: 16 `OUTPUT_LO` + 16 `OUTPUT_HI_THIS_STEP`, `tail_stack0_store_loaded_route_*`). This bank's READ lane (OUTPUT) == its WRITE lane, so the enumerated form is an identity-copy of the OUTPUT byte gated by structural evidence; each route unit fires on its channel's one-hot plus the OTHER band's one-hot sum (same 2-channel firing magnitude / threshold=25 as the enumerated AND) and writes `nibble_value_writes` for that channel. Model FFN units 43040→42817 (−223). Flag-OFF byte-identical to golden `81557d21`; flag-ON golden `6b12bda5`. VERDICT-NEUTRAL: 86-id campaign run (served var/if_var + HOLD arith clusters) is 75/86 both states with ZERO per-program status or fail-shape diffs; `lint_cross_op_ffn` PASS. Isolated numeric proof: `tools/_probe_m8_computed_writeback.py` (route reproduces the enumerated winning byte + firing region byte-for-byte). See `_stack0_store_loaded_computed_enabled` in `l10_ops.py`. **Proves the whole enumerated→computed track for identity-copy byte-writeback banks.** |
| `C4_STACK0_POP_LOADED_COMPUTED` | `0` (off) | any | **CONVERGENT (merge candidate — M8 enumerated→computed roll-out)** — sibling of `C4_STACK0_STORE_LOADED_COMPUTED` on the `stack0_pop_loaded_output_rules` bank (255 per-VALUE AND units, `tail_stack0_pop_loaded_byte_01..ff`, skips `0x00`) → COMPUTED per-NIBBLE route (32 units, `tail_stack0_pop_loaded_route_*`). Same-lane identity copy (READ==WRITE lane OUTPUT); `match_weight=0.05`, `threshold=12.0`, `competitor` softened to 5 under `C4_STACK0_POP_LOADED_SHALLOW_CRUSH`. Model FFN units 43040→42817 (−223). Flag-OFF byte-identical to golden `81557d21`; flag-ON golden `bf661edf`. VERDICT-NEUTRAL: 80-id campaign run (`--ids 0-9,50-59,100-109,250-274,425-449 --criterion full_trace --spec-k 0`) is 75/80 both states with ZERO per-program status or fail-shape diffs (identical pass-set + divergence fields). Numeric proof: `tools/_probe_computed_writeback_banks.py` (identical firing region + 0 argmax mismatch across all 255 non-zero bytes for BOTH competitor values). `lint_cross_op_ffn` PASS. See `_stack0_pop_loaded_computed_enabled` in `l10_ops.py`. |
| `C4_STACK0_STORE_E8_COMPUTED` | `0` (off) | any | **CONVERGENT (merge candidate — M8 enumerated→computed roll-out)** — sibling of `C4_STACK0_STORE_LOADED_COMPUTED` on the NIBBLE-LOOP part of `stack0_store_top_e8_from_e0_output_rules` (255 per-VALUE AND units, `tail_stack0_store_top_e8_from_e0_byte_01..ff`, skips `0x00`) → COMPUTED per-NIBBLE route (32 units, `tail_stack0_store_top_e8_from_e0_route_*`). Same-lane identity copy (READ==WRITE lane OUTPUT); `match_weight=0.001`, `threshold=4300.0`. The trailing `byte_39_from_e8_addr` special rule is NOT part of the loop and is emitted UNCHANGED (family 256→33). Model FFN units 43040→42817 (−223). Flag-OFF byte-identical to golden `81557d21`; flag-ON golden `8ba805a0`. VERDICT-NEUTRAL: 80-id campaign run (`--ids 0-9,50-59,100-109,250-274,425-449 --criterion full_trace --spec-k 0`) is 75/80 both states with ZERO per-program status or fail-shape diffs (identical pass-set + divergence fields). Numeric proof: `tools/_probe_computed_writeback_banks.py` (identical firing region + 0 argmax mismatch across all 255 non-zero bytes). `lint_cross_op_ffn` PASS. See `_stack0_store_e8_computed_enabled` in `l10_ops.py`. |
| `C4_STACK0_STORE_TOP_E0_COMPUTED` | `0` (off) | any | **CONVERGENT (merge candidate — GAP-PRIMITIVE #3, CROSS-LANE enumerated→computed pilot)** — replaces the `stack0_store_top_e0_output_rules` **CROSS-LANE** ALU→OUTPUT materializer (254 per-VALUE AND units, `tail_stack0_store_top_e0_byte_*`, SOURCE=`ALU_LO`/`ALU_HI`, DST=`OUTPUT_LO`/`OUTPUT_HI_THIS_STEP`) with a COMPUTED per-NIBBLE cross-lane route (32 units, `tail_stack0_store_top_e0_route_*`) via `building_blocks_dsl.byte_copy_computed_rules`. Unlike M8 (`STACK0_STORE_LOADED_COMPUTED`, same-lane OUTPUT→OUTPUT), READ lane (ALU) ≠ WRITE lane (OUTPUT): each route unit fires on the SOURCE (`ALU_*+k`) one-hot plus the SUM of the OTHER SOURCE band (reconstructing the enumerated 2-channel AND magnitude / threshold=25) and writes a one-hot nibble into OUTPUT. Model FFN units 43040→42818 (−222). Flag-OFF byte-identical to golden `81557d21`; flag-ON golden `54f3c59d`. VERDICT-NEUTRAL: 105-id campaign run (HOLD add/sub/mul 0-9/50-59/100-109 + var_simple/var_mul + if_var, spec_k=0 full_trace) is 75/105 BOTH states with ZERO per-program status/exit/divergence_step/pc/ax diffs; `lint_cross_op_ffn` PASS. Isolated numeric proof: `tools/_probe_crosslane_bytecopy.py` (computed cross-lane route reproduces the enumerated ALU→OUTPUT bank's winning byte + firing region byte-for-byte across all 256 source bytes; the existing gate-based `byte_route_rules` does NOT reproduce the additive conditions-AND form). See `_stack0_store_top_e0_computed_enabled` in `l10_ops.py`. **Proves the CROSS-LANE enumerated→computed track: the l16 marker-from-alu + l10 ALU-result materializer classes collapse via one `byte_copy(src,dst)` generator.** |
| `C4_WIDE_MUL_BYTE1_COMPUTED` | `0` (off) | any | **CONVERGENT (merge candidate — M8 enumerated→computed roll-out, l10 tail sweep)** — sibling of `C4_STACK0_STORE_LOADED_COMPUTED` on the `wide_mul_byte1_preserve_rules` bank (FULL 256 per-VALUE AND units, `tail_wide_mul_byte1_preserve_*`, NO `0x00` skip) → COMPUTED per-NIBBLE route (32 units, `tail_wide_mul_byte1_preserve_route_*`). Under the OP_MUL AX byte-0 evidence gate the bank re-asserts a staged MUL byte value it matches on `OUTPUT_LO+lo`/`OUTPUT_HI_THIS_STEP+hi` (weight 2.0 each) at strength 1e7. Same-lane identity copy (READ==WRITE lane OUTPUT; `OUTPUT_HI` aliases `OUTPUT_HI_THIS_STEP`); `match_weight=2.0`, `threshold=220.0`. Model L10-tail FFN 1172→948 units (−224), total 42149→41925. Flag-OFF byte-identical to golden `91f55411`; flag-ON golden `1945a8ea`. Isolated numeric proof: `tools/_probe_wide_mul_computed_writeback.py` (identical firing region + 0 argmax mismatch across ALL 256 bytes × 4 OUTPUT magnitudes, 0 gate-off spurious fires). This was the SOLE remaining live enumerated 256-rule bank in the `_tail_bit32_result_correction_rules` width-sensitive bank (the other 256, `tail_sp_pop_carry_byte2_*`, is already collapsible via `C4_SP_BYTE2_CARRY`). See `_wide_mul_byte1_computed_enabled` in `l10_ops.py`. |

---

## Notes on non-flag false-positives

The following identifiers appear in `C4_`-shaped prose (comments/docstrings) but
have **no** `environ.get` call site and are NOT runtime flags: `C4_FOO`,
`C4_BAR`, `C4_ROOT`, `C4_5101`, `C4_HEADER`, `C4_OPCODES`, `C4_INIT_TOKENS`,
`C4_GENERATION`, `C4_RUNTIME`, `C4_SF_GENERATION`, `C4_SF_RUNTIME`,
`C4_RELEASE_DIR`, `C4_SOFTFLOAT`, `C4_FP_MATH`, `C4_1096_LOWERING_AUDIT`,
`C4_OPERAND_GATHER_PSH_ROWSELECT`, `C4_STACK0_B0_REPOINT_WS`(-only-in-format),
`C4_B1_TO_OUTPUT`, `C4_LOADED_OPERAND_HI15_CLEAR` (DROPPED broad clear, replaced
by `C4_LOADED_OPERAND_ADD_HI15_CLEAR` + `C4_OPERAND_CAM_FIX`),
`C4_VERIFY_DECLARATIONS` (prose only — docstring in `decl_verifier.py`, no env
read), `C4_INC3_ROOTA_KAXC` (prose only — held-off building-block note in a
probe), `C4_L15_LEV`, `C4_L16_`, `C4_L8_` (grep prefixes). If any of these
SHOULD be flags (e.g. a documented-but-unwired knob), that is a separate I2
finding.

> Correction (2026-07-03): `C4_STRICT_GATE_CHECK`, `C4_REQUIRE_DECLARATIVE_BAKE`,
> `C4_DIM_LIVENESS`, `C4_VM_CACHE_MAX_BYTES`, `C4_VM_CACHE_MAX_ENTRIES`,
> `C4_ADAPTIVE_START_K/MIN_K/MAX_K`, and
> `C4_BATCH_KV_VERIFY_INTERVAL/FLUSH_INTERVAL/EVICTION_OVERSHOOT` are REAL tooling
> flags (read indirectly via an `_ENV`-name constant or the `_env_int` helper, so
> the literal-`environ.get("C4_...")` grep misses them). They are now listed in
> the STRUCTURAL / RUNTIME tooling table above, not here.
