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
| `C4_SI_STORE_ADDR` | unset→off | campaign | active | L15 head-16 SI/SC store address-provenance CAM (var_mul / multilocal-LI). Grows L15 16→17 heads. Flipped ON by `C4_CAMPAIGN`. |
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
| `C4_STACK0_B0_DUMP` | `1` | active | STACK0 byte-0 register-dump LM-head columns (Root 2). |
| `C4_STACK0_NEXT_ARITH` | `1` | active | Consumer-opcode lookahead bands + dump-block gate (#221). |
| `C4_BP_SAVE_DUMP` | `1` | active | BP-save register dump (the `_isa_golden_hash` reference flag). |
| `C4_MUL_WIDTH2` | `1` | active | Width-2 (16-bit) MUL path + MUL_RESULT_HI band (d_model 872→981). |
| `C4_MUL_W2_THRESH_FIX` | `1` | active | Width-2 MUL threshold correction. |
| `C4_DIV_MULTIBYTE` | `1` | active | Multi-byte DIV/MOD result path. |
| `C4_SUB_FULL_BORROW` | `1` | active | Full SUB borrow-cascade. |
| `C4_ADDSUB_DUMP_BOOST` | `1` | active | Imperative AddSub byte-0 OUTPUT dominant-amplitude write. |
| `C4_CMP_FLAG_MARGIN_FIX` | `1` | active | CMP equal-high-nibble GT flag margin (if_gt 357/359/361). |
| `C4_CMP_EQ_HINIB_VETO` | `1` | active | CMP EQ high-nibble veto. |
| `C4_CMP_GT_LO_MARGIN` | `1` | active | CMP GT low-byte margin. |
| `C4_CMP_HI_LT_ALU15_GUARD` | `1` | active | CMP hi-byte LT ALU15 guard. |
| `C4_CMP_GT_LO_LT_HIEQ_GUARD` | `1` | active | CMP GT-lo / LT-hi-eq guard. |
| `C4_CMP_BYTE0_SE_RECOVER` | `1` | active | CMP byte-0 SE-recover. |
| `C4_BITWISE_BYTE0_SE_RECOVER` | `1` | active | Bitwise (or/xor/and) byte-0 SE-recover (16-bit landed). |
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
| `C4_MEM_MARKER_OUTPUT_CLEAR` | `1` | active | L0 MEM-marker OUTPUT clear. |
| `C4_TAIL_LEA_E8_ARITH_GUARD` | `1` | active | Tail LEA-E8 arith guard. |
| `C4_TAIL_LEA_E8_DIVMOD_GUARD` | `1` | active | Tail LEA-E8 DIV/MOD guard. |
| `C4_TAIL_LEA_E8_ENT_GUARD` | `1` | active | Tail LEA-E8 ENT guard (also requires `no_stack0_emit`). |
| `C4_NESTED_JSR_PC_FIX` | `1` | active | Nested JSR PC fix. |
| `C4_INC3_H5_DIM0_CLEAN` | `1` | active | L8 head-5 dim-0 clean (Inc-3). |

---

## CAMPAIGN-TRACK fixes (30-token config; golden byte-identical OFF)

Most fall through to `no_stack0_emit_enabled()` when their own env is unset, so
`C4_CAMPAIGN=1` activates them automatically; each keeps its own kill-switch for
the flag-regression / cross-op gates.

| Flag | Default | Status | Purpose |
|------|---------|--------|---------|
| `C4_LOADED_OPERAND_ADD_HI15_CLEAR` | `1` (campaign) | active | Loaded-operand ADD hi-nibble cell-15 address-leak clear (var_update). |
| `C4_FUNCADD_ALU_HI13_CLEAR` | `1` (campaign) | active | Loaded-operand ADD hi-nibble cell-13 leak clear (func_add/mul/max/min). |
| `C4_SILI_CAM_B1` | `1` (campaign) | active | SI/LI load byte-1 address-leak discriminator (Inc-2). |
| `C4_SILI_B1_RESTORE` | `1` (campaign) | active | SI/LI byte-1 restore. |
| `C4_STORE_AX_B0_OVERRIDE` | `1` in build / `0` in shared | active | Store AX byte-0 override. |
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
| `C4_LEA_BYTE0_ALU_AMPLIFY` | forced→live | campaign | **KEEP (A/B knob on live corrector)** — falls through to a `no_stack0_emit` default. |
| `C4_L15_SAVEDRA_HEAD` | forced→live | campaign | **KEEP (A/B knob on live corrector)** — L15 head-15 saved-RA delivery. |
| `C4_POST_ENT_SE_SUPPRESS` | `0` (off) | campaign | **DELETED** (commit `2043ff4d`) — dead-end framing building block. |
| `C4_ENT_SP_BYTE1_ISMARK_BLOCKER` | `0` (off) | campaign | **DELETED** (commit `10d7b876`) — docstring records NEGATIVE RESULT (0 programs advance; "the real fix is the project-level multi-part build, not a solo corrector"). |
| `C4_L16_LEV_STACK0_PRESERVE_SE_BLOCKER` | `0` (off) | campaign | **DELETED** (commit `1466958c`) — dead-end framing MARK_SE NOT-blocker. |
| `C4_PSH_STACK0_BYTE3_RELAY_DARKEN` | `0` (off) | campaign | **DELETED** (commit `4267cfb3`) — net −1 exit_code TRADE with a documented "cannot be cleanly separated func-vs-var" wall. |
| `C4_FUNC_LEA_B0_RESTORE` | `0` (off) | campaign | **DELETED** (commit `ad2a3f25`) — ZERO-SUM (flips func_add, regresses func_identity; the two LEA rows are bit-identical → not separable). Removed the band + capture/restore ops. |
| `C4_BP_SAVE_DUMP_MARKER_REQ` | `0` (off) | campaign | **DELETED** (commit `67f536e4`) — dead-end BP-dump gate-rewrite variant. |
| `C4_ADDSUB_DECLARATIVE` | `0` (off) | golden | **BORDERLINE (KEEP)** — selects the in-flight `DeclarativeAddSubBlock` migration path (docs/ADDSUB_DSL_MIGRATION_*, probe tool). Migration WIP, not a dead corrector. |
| `C4_AX_BYTE1_FULL_WIDTH` | `0` (off) | golden | **BORDERLINE (review)** — alternate full-width AX byte-1 structural variant (17 refs, band + LM-head columns). |
| `C4_AX_HIBYTE_CLEAR` | `0` (off) | golden | **BORDERLINE (review)** — alternate structural op coupled to the completed AX-emission H-band bake in `model_ops.py`; entangled with a sensitive done-refactor. |
| `C4_STACK0_B0_POPPED` | `0` (off) | campaign | **BORDERLINE (review)** — band + L9 latch head + dump condition; part of the ACTIVE if/bool/expr OUTPUT-band megaroot family. Held pending that multi-part fix. |
| `C4_SP_POP_MARKER_CMP3_HARDGATE` | `0` (off) | campaign | **BORDERLINE (review)** — verified SP-drift fix, currently net −2 (if_var 13→11) but a staged prerequisite that "should net positive the moment the LI value-load root lands" (#313/#289). |
| `C4_JSR_PC_BYTE1` | `0` (off) | campaign | **BORDERLINE (review)** — large multi-file feature (bands + relay head + allocator + staging/emit FFN across `model_ops`/`l3_ops`/`all_core_ops`); high blast-radius, conservative hold. |
| `C4_SP_BYTE2_CARRY` | `0` (off) | any | **CONVERGENT (merge candidate)** — collapses the `sp_pop_carry_rules` byte-2 pop-carry enumeration from `range(256)` (256 FFN units, `tail_sp_pop_carry_byte2_00..ff`) to `range(2)` (`old in {0x00,0x01}`), deleting 254 provably-dead FFN units (model 43071→42817 units). SP stays in `[0x0FE10,0x10000]` corpus-wide so byte-2 is only ever 0x00/0x01 and only `0x00→0x01` fires; the 2 kept rules are byte-for-byte identical to the OFF bank members. Flag-OFF byte-identical to golden `b4d2ab27`; flag-ON golden `716a66c1`. See `_sp_byte2_carry_computed_enabled` in `l10_ops.py`. |

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
