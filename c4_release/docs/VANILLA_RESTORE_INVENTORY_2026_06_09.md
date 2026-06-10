# Vanilla-Restore Inventory — 2026-06-09

Goal (per user): runners must be **pure forward-pass wrappers**.
`model.forward(input_ids) → argmax → append → loop until EXIT`.
Nothing else. No domain logic, no ALU recovery, no shadow memory, no cmp
synth, no `_compute_alu_legacy`, no opcode-specific Python branches that
compute values.

This doc inventories every site in the runners
(`c4_release/neural_vm/run_vm.py` and
`c4_release/neural_vm/batched_pure_neural.py`) that violates the
principle, with an upstream-neural-bug attribution and removal blocker.
Other runners (`fast_runner.py`, `batch_runner.py`, `batch_runner_v2.py`,
`transformer_first_runner.py`) were grepped and are CLEAN (no override /
shim / synth code).

## Catalog

| # | Location | Code purpose | Hides what neural bug | Removal blocker |
|---|---|---|---|---|
| 1 | run_vm.py:1964-2013 `_compute_alu_legacy` | Python ALU for ADD/SUB/MUL/DIV/MOD/OR/XOR/AND/SHL/SHR/EQ/NE/LT/GT/LE/GE | Handler-mode + batched ALU synth | L9/L10 ALU rules broken (see Removal #2 of RUNNER_OVERRIDE_REMOVAL_PLAN) |
| 2 | run_vm.py:2229-2238 PSH override block | Decrements SP, stores AX, overrides REG_SP + STACK0 | PSH SP decrement + STACK0 store not done autoregressively | Phase 2 PSH bake unresolved |
| 3 | run_vm.py:2239-2255 JSR override block | Pushes return addr to shadow mem, overrides PC/SP/STACK0 | JSR PC routing + return-frame store | Phase 5 nested JSR + AX-preservation xfails |
| 4 | run_vm.py:2256-2272 ENT override block | Computes BP/SP for stack frame, overrides REG_BP/REG_SP/REG_PC | ENT imm subtraction not in pure_neural | Phase 5 ENT nonzero-imm xfail |
| 5 | run_vm.py:2273-2288 LEV override block | Loads mem[BP], mem[BP+8], overrides REG_BP/PC/SP | LEV PC restore from frame | Phase 5 `_set_layer9_lev_bp_to_pc_relay` |
| 6 | run_vm.py:2289-2298 JMP override block | Resolves target, overrides REG_PC | JMP target resolution | Phase 4 JMP operand resolution |
| 7 | run_vm.py:2299-2311 BZ override block | Python-side AX zero check + PC override | BZ taken-path | Phase 4 `_set_layer4_ffn` PC carry-forward |
| 8 | run_vm.py:2312-2322 BNZ override block | Symmetric BZ | Same as #7 | Same |
| 9 | run_vm.py:2323-2333 ADJ override block | Adds imm to SP, overrides REG_SP | "fully neural per F matrix" but unverified | Phase 2 ADJ unit test missing |
| 10 | run_vm.py:2334-2354 BINARY_POP_OPS override + `_compute_alu_legacy` call | Pops mem[SP] → ALU → overrides REG_AX | Multi-byte ADD/SUB carry; comparator bytes | Phase 2/3 ALU writeback chain |
| 11 | run_vm.py:2355-2361 LI override | Loads word from shadow mem, overrides AX | L15 memory_lookup word LI | Phase 7 SI/LI roundtrip |
| 12 | run_vm.py:2362-2368 LC override | Loads byte from shadow mem, overrides AX | L15 memory_lookup byte LC | Same |
| 13 | run_vm.py:2369-2376 SI override | Pops addr, stores AX to shadow mem | L14 mem_generation word SI | Phase 7 SI |
| 14 | run_vm.py:2377-2383 SC override | Pops addr, stores AX byte to shadow mem | L14 mem_generation byte SC | Phase 7 SC |
| 15 | run_vm.py:2389-2397 LEA override | Computes BP+imm, overrides REG_AX | LEA arithmetic | Phase 5 BP correctness |
| 16 | run_vm.py:2417-2419 EXIT override | Re-asserts REG_AX | Final AX preservation | "REMOVABLE-NOW" per docstring |
| 17 | run_vm.py:2511-2539 `_inject_getchar` | Reads `_stdin_buffer`, overrides REG_AX from stdin | V9 GETCHAR bake stub | `user_input_ops.py:60` `NotImplementedError` |
| 18 | run_vm.py:2541-2559 `_peek_argc_from_adj` | Reads bytecode ADJ to derive argc for IO shims | IO syscall argc not in neural emit | Phase 6 IO bake stubs |
| 19 | run_vm.py:2561-2588 `_extract_imm_chain_args` | Walks bytecode back-chain for IMM/PSH args | IO syscall arg gather not in neural emit | Phase 6 IO |
| 20 | run_vm.py:2590-2725 `_handle_skipped_io_op` | Python PRTF/READ/OPEN/CLOS/PUTCHAR shim driven by bytecode walk | All IO syscalls | IO_SUBSYSTEM_STATUS_2026_06_09.md — think-protocol disabled |
| 21 | run_vm.py:2761-2800 `_handle_pure_neural_stall` | Synthesizes forward IO execution when neural PC freezes | L3 PC carry-forward freeze | L3 PC bug; stall-detector itself is a hack |
| 22 | run_vm.py:2802-2880 `_neural_prtf_emit` | Walks fmt string in shadow mem, formats, overrides REG_AX | PRTF format walk + byte emit | `prtf_think_protocol` bake stub (l6_ops.py:4387) |
| 23 | run_vm.py:2882-2910 `_neural_open_emit` | os.open against shadow-mem path string, overrides AX | OPEN syscall | Tool-boundary; may stay |
| 24 | run_vm.py:2912-2927 `_neural_clos_emit` | os.close, overrides AX | CLOS syscall | Tool-boundary; may stay |
| 25 | run_vm.py:2929-2995 `_neural_read_emit` | os.read into shadow mem, injects MEM sections, overrides AX | READ syscall + USER_INPUT gather | V9 phase-2 user_input bakes |
| 26 | run_vm.py:2997-3018 `_override_register_in_last_step` / `_override_ax_in_last_step` | Generic register byte-rewrite hammer | Called by EVERY override above | Wholesale-deletion target |
| 27 | run_vm.py:3020-3052 `_inject_synthetic_step` | Appends a fully-synthesized 35-token step | Conversational-IO PRTF path | Legacy convo-IO; check if dead |
| 28 | run_vm.py:3058-3088 `_track_memory_write` + `_extract_mem_write` | Updates shadow `_memory` from emitted MEM section | LI/LC/SI/SC handler-mode shadow memory | Phase 7 mem |
| 29 | run_vm.py:3101-3105 `_mem_store_word` | Writes 4 bytes to shadow mem | Same — feeds LI/LC overrides | Phase 7 mem |
| 30 | run_vm.py:3134-3139 `_mem_load_word` | Reads 4 bytes from shadow mem | Same | Phase 7 mem |
| 31 | run_vm.py:3141-3160 `_read_stack_arg` | Reads stack slot via STACK0 / shadow mem fallback | Handler-mode syscall arg | Phase 6 IO |
| 32 | run_vm.py:3171-3189 `_read_string` | Walks null-terminated string from shadow mem | PRTF/OPEN format-string + path-string | Phase 6 IO neural string-walk |
| 33 | run_vm.py:3191-3236 `_format_printf` | Python sprintf for %d/%s/%c/%x | PRTF format substitution | Phase 6 PRTF neural formatter |
| 34 | run_vm.py:3242-3258 `_syscall_clos` | Handler-mode CLOS shim, overrides AX | CLOS handler-mode | Same as #24 |
| 35 | run_vm.py:3260-3287 `_syscall_open` | Handler-mode OPEN shim, overrides AX | OPEN handler-mode | Tool-boundary |
| 36 | run_vm.py:3289-3333 `_syscall_read` | Handler-mode READ shim | READ handler-mode | V9 phase-2 |
| 37 | run_vm.py:3335-3391 `_syscall_prtf` | Handler-mode PRTF shim | PRTF handler-mode | Phase 6 PRTF |
| 38 | run_vm.py:1399-1447 conversational-IO THINKING_END branch | Reads fmt string from shadow mem, appends bytes to context, injects synthetic step | Hybrid convo-IO PRTF | Legacy; pure_neural ignores |
| 39 | run_vm.py:1437-1446 `_inject_synthetic_step` call site | Same as #38 | Same | Same |
| 40 | run_vm.py:117-150 `_BINARY_POP_OPS`, `_NEURAL_32BIT_OPS`, `_RUNNER_ALU_OPS` constants | Drives the per-op override classification | Per-op ALU recovery enablement | Sites #1, #10, batched #41-43 |
| 41 | batched_pure_neural.py:2121-2129 IMM AX override | Reads IMM byte from bytecode, overrides REG_AX | L5 byte-decode mis-emits for IMM in [0xE0,0xFF] | Removal-1 in plan; secondary surface unidentified |
| 42 | batched_pure_neural.py:2137-2145 GETCHAR override | Reads stdin buffer, overrides REG_AX | Mirrors #17 in batched path | V9 GETCHAR bake |
| 43 | batched_pure_neural.py:2147-2155 LI/LC mem_history lookup | Reads `s.mem_history[prev_ax]`, overrides AX | L15 memory_lookup | Phase 7 mem |
| 44 | batched_pure_neural.py:2157-2177 PRTF/OPEN/CLOS/READ defer-to-serial shim | `_borrow_serial_state` then call `_neural_*_emit` | Tool-boundary IO in batched path | Same as #22-25 |
| 45 | batched_pure_neural.py:2182-2193 MEM section addr0 patch via `stack0_shadow` | Rewrites MEM section addr bytes when emitted addr=0 | L14 SI/SC addr emit broken | Open bug `project_l10_psh_addr_ent_bug` |
| 46 | batched_pure_neural.py:2235-2264 collapsed-step BINARY_POP_OPS synth via `_compute_alu_legacy` | Detects IMM step that skipped a binop, synthesizes AX via legacy Python ALU | Same as #1+#10 | Removal-2/3 in plan |
| 47 | batched_pure_neural.py:2316-2336 non-collapsed `_NON_COLLAPSED_RECOVERY_OPS` synth via `_compute_alu_legacy` | ADD/SUB/OR/XOR/AND/EQ/NE rewrite of AX via Python ALU | Same as #46; multi-byte cascade | Removal-2/3/4 |
| 48 | batched_pure_neural.py:69-145 `_alu_recovery_disabled_for` + env knobs `C4_DISABLE_BATCHED_ALU_RECOVERY[_OPS]` | Per-op kill-switch for #46/#47 | Plumbing for incremental removal | Self-removing once #46/#47 retire |
| 49 | batched_pure_neural.py:2087-2088 `last_pushed_value` snapshot at PSH | Captures pre-PSH AX so #46/#47 can recover stack operand | Same as #46/#47 | Same |
| 50 | batched_pure_neural.py:2080-2081 `stack0_shadow` mirror | Snapshots neural STACK0 emit; consumed by #45 MEM patch | L14 SI/SC addr emit | Phase 7 mem |
| 51 | batched_pure_neural.py:2402-2435 `_decode_bail_exit_code` forward-scan | At cap-hit, scans forward for in-progress REG_AX to recover xor/and_basic exits | Model collapses mid-step but emitted correct AX | Per plan: "mostly legit"; revisit after #46/#47 |
| 52 | batched_pure_neural.py:2452-2483 `_borrow_serial_state` / `_unborrow_serial_state` | Mutates serial runner state to invoke its IO shims from batched path | Plumbing for #44 | Self-removing once #44 retires |
| 53 | batched_pure_neural.py:1829-1988 mem-history windowing (`_windowed_context`, `mem_store_positions`, etc.) | Re-emits historical MEM sections + tags MEM_ADDR_SRC | L15 memory_lookup expects historical MEM tokens to still be reachable via attention | Phase 7.F KV eviction; large surface |
| 54 | batched_pure_neural.py:2438-2446 `_track_mem_access` | LRU `mem_history` per element | Same | Same |
| 55 | run_vm.py:74-80, 1093-1102 `_stdin_buffer` plumbing | Holds host stdin so `_inject_getchar`/`_handle_skipped_io_op`/`_neural_read_emit` can consume | All GETCHAR/READ | V9 user_input gather bake |
| 56 | run_vm.py:1361-1364 forced STEP_END rewrite at pos 34 (handler-mode only) | Overrides next_token to STEP_END when model didn't emit | Legacy handler-mode step-boundary emit | "REMOVABLE-NOW" per inline tag |
| 57 | run_vm.py:1530-1543 TOOL_CALL handler-mode override block | Calls `_syscall_handlers[op]` + `_track_memory_write` | TOOL_CALL handler-mode IO | Same as #34-37 |
| 58 | run_vm.py:1571-1586 HALT handler-mode REG_AX re-assert | Re-overrides AX at HALT in handler-mode | Final AX preservation | "REMOVABLE-NOW" per inline tag |
| 59 | run_vm.py:2402-2406 PC mirror block | Mirrors emitted REG_PC into `_last_pc` for tracking | Observation only, no override | "REMOVABLE-NOW" but feeds JMP/BZ/BNZ overrides |

## Notes on what's already "clean"

- `fast_runner.py`, `batch_runner.py`, `batch_runner_v2.py`,
  `transformer_first_runner.py` — no overrides, shims, or shadow mem.
  These are the model-shaped runners; they call `model.forward` and
  argmax. **They are the target shape.**
- `_PHASE6_SYSCALL_STATUS` table at run_vm.py:48-79 is descriptive, not
  load-bearing; deletable once #20/#22 retire.
- KV cache machinery (`_get_or_build_kv_cache`,
  `_preload_kv_cache_with_prefix`, `_trim_kv_cache`) is pure inference
  optimization — not a cheat. **Keep.**
- DraftVM / speculative-decoding (`_init_draft_vm`, `_run_speculative`,
  `_apply_token_spec`) is verifier-arbitrated — model is final arbiter.
  Per RUNNER_OVERRIDE_REMOVAL_PLAN, **keep**.
- Divergence-bail (`enable_divergence_bail`, `_bail_check`,
  `last_divergence_signal`) is runtime safety, not a value synth.
  **Keep.**

## Removal Plan (ordered by safety + impact)

### Wave A — safe removals masking already-fixed neural ops

Inline-tagged "REMOVABLE-NOW" by the original author. Delete + verify
smoke does not regress.

A1. **#56** forced STEP_END rewrite (handler-mode pos 34). Safe — only
    affects handler-mode.
A2. **#58** HALT handler-mode REG_AX re-assert.
A3. **#16** EXIT REG_AX re-assert in `_dispatch_step` end.
A4. **#59** PC mirror block. Observation only; deletable once #6/#7/#8
    (JMP/BZ/BNZ) go via Wave C.
A5. **#27, #38, #39** conversational-IO `_inject_synthetic_step` path.
    Verify no live caller, then delete.
A6. **#48** `_alu_recovery_disabled_for` env machinery — self-removing
    once #46/#47 retire.

### Wave B — IO shims (cross VM/host boundary)

Per IO_SUBSYSTEM_STATUS_2026_06_09.md, blocked on bakes that raise
`NotImplementedError`. Each removal needs a paired bake brief.

B1. **#17 `_inject_getchar`** ↔ blocker: `user_input_ops.py:60` V9
    phase-2 gather bake. Brief: enable
    `make_layer5_user_input_gather_op` +
    `make_layer6_getchar_routing_op` per V9_GETCHAR_READ_NEURAL_PLAN.md.
B2. **#22 `_neural_prtf_emit`** + **#33 `_format_printf`** + **#32
    `_read_string`** ↔ blocker: `l6_ops.py:4387` `prtf_think_protocol`
    bake stub. Needs L8 format-position counter + L9 format fetch + L10
    null-terminator chain.
B3. **#25 `_neural_read_emit`** ↔ blocker: V9 phase-2 + STDIN_BYTE_LO/HI
    dim alloc + L5 8→10 head widening.
B4. **#20 `_handle_skipped_io_op`** + **#21 `_handle_pure_neural_stall`**
    + **#18 `_peek_argc_from_adj`** + **#19 `_extract_imm_chain_args`** —
    only fire because IO bakes are stubs; retire with B1-B3.
B5. **#23 `_neural_open_emit`**, **#24 `_neural_clos_emit`**, **#34
    `_syscall_clos`**, **#35 `_syscall_open`** — true VM/host boundary;
    may be intentionally external. Revisit after B1-B3.

### Wave C — handler-mode VM-semantic blocks (`pure_neural=False`)

These only fire when `pure_neural=False`. Strategy: deprecate the
non-pure-neural path entirely once every smoke test runs under
pure_neural.

**2026-06-09 audit result:** all Wave C blocks (C1-C9) are PROVEN
STILL-NEEDED. No tests pass `pure_neural=False` explicitly, but
`AutoregressiveVMRunner()` defaults to `pure_neural=False`, and 17 test
files under `c4_release/tests/` construct the runner that way:
test_lev_comprehensive, test_complex_programs, test_benchmarks,
test_jmp_neural, test_memory_neural, test_conversational_io_comprehensive,
test_jsr_neural_status, test_property_based, test_dual_weight_modes,
test_bz_bnz_neural, test_autoregressive_kv_cache, test_control_flow_neural,
test_arithmetic_no_handlers, test_neural_handler_parity, test_ent_lev_neural,
verify_adj, trace_ent. Per-opcode coverage grep confirms every Wave C
opcode (PSH/JSR/ENT/LEV/JMP/BZ/BNZ/ADJ/LI/LC/SI/SC/LEA + BINARY_POP_OPS:
ADD/SUB/MUL/DIV/MOD/OR/XOR/AND/SHL/SHR/EQ/NE/LT/GT/LE/GE) is
exercised. Helper utilities C8 (`_track_memory_write`, `_extract_mem_write`,
`_mem_store_word`, `_mem_load_word`) and C9 (`_read_stack_arg`) are also
called by the Wave B pure_neural IO shims (`_neural_prtf_emit`,
`_handle_skipped_io_op`, `_neural_open_emit`, `_neural_clos_emit`,
`_neural_read_emit`) so they cannot be retired before Wave B lands.
The dispatch block is now framed by explicit `WAVE C BEGIN` / `WAVE C END`
banners in `run_vm.py` so future cleanup can excise it as a single unit
once the handler-mode tests are migrated to the pure_neural fixture.
Smoke baseline (`CUDA_VISIBLE_DEVICES=1 pytest
c4_release/tests/test_smoke.py --tb=no -q`) remains 46 passed / 5 failed
(the 5 SI/LI/SC/LC memory tests are pre-existing — see
`project_l15_li_stack0_byte_attribution`).

C1. **#1 `_compute_alu_legacy`** — retire once C2 + D6/D7 land.
C2. **#10 BINARY_POP_OPS** runner block. Blocker: L9/L10 ALU multi-byte
    writeback (Removals #2+#3 in RUNNER_OVERRIDE_REMOVAL_PLAN).
C3. **#11 LI**, **#12 LC**, **#13 SI**, **#14 SC** — blocker: L14
    mem_generation + L15 memory_lookup. Memory note:
    `project_l15_li_stack0_byte_attribution`.
C4. **#2 PSH**, **#3 JSR**, **#4 ENT**, **#5 LEV** — Phase 5 call-frame.
    Memory note: `project_l10_psh_addr_ent_bug`.
C5. **#6 JMP**, **#7 BZ**, **#8 BNZ** — `_post_l9_bz_pc_override_rules`
    already in l6_ops; verify and delete.
C6. **#9 ADJ** — likely safe per F-matrix; needs a Phase-2 ADJ pytest.
C7. **#15 LEA** — Phase 5 BP-correctness.
C8. **#28-30 `_track_memory_write` / `_extract_mem_write` /
    `_mem_store_word` / `_mem_load_word`** — handler-mode shadow memory.
    Retire with C3.
C9. **#31 `_read_stack_arg`** — feeds handler-mode IO; retire with B5.

### Wave D — batched-runner overrides

D1. **#41 batched IMM AX override** — Removal-1 in plan. 2026-06-06
    retry showed unidentified secondary surface emitting wrong AX for
    IMM in [0xE0, 0xFF]. Needs L34 residual probe.
D2. **#42 batched GETCHAR override** — retire with B1.
D3. **#43 batched LI/LC mem_history lookup** — retire with C3.
D4. **#44 batched PRTF/OPEN/CLOS/READ defer-to-serial** + **#52
    `_borrow_serial_state`** — retire with B1-B5.
D5. **#45 MEM section addr0 patch** — blocker: L14 SI/SC addr emit
    (overlapping `project_l10_psh_addr_ent_bug`).
D6. **#46 collapsed-step `_compute_alu_legacy` synth** — Removal-2 in
    plan.
D7. **#47 non-collapsed `_compute_alu_legacy` synth** — Removal-3
    (32-bit cascade) + Removal-4 (EQ/NE Shape B). Still in place after
    2026-06-07 retry — see commit annotations at lines 2282-2315.
D8. **#49 `last_pushed_value`** + **#50 `stack0_shadow`** — feeds
    D5/D6/D7; retire together.
D9. **#53 mem-history windowing** + **#54 `_track_mem_access`** —
    Phase 7.F KV eviction; large surface, lowest priority.
D10. **#51 `_decode_bail_exit_code`** — per plan, "mostly legit"
     runtime recovery. Revisit last.

### Wave E — wholesale teardown of override plumbing

E1. **#26 `_override_register_in_last_step` /
    `_override_ax_in_last_step`** — deletable once all call sites die.
E2. **#40 `_BINARY_POP_OPS` / `_NEURAL_32BIT_OPS` / `_RUNNER_ALU_OPS`**
    constants — deletable once Wave C/D ALU work is done.
E3. **#55 `_stdin_buffer` plumbing** — deletable once B1+B3 land.
E4. **#57 TOOL_CALL handler-mode override** — deletable with C wave.

## Ratchet contract

After EACH removal: smoke must remain ≥ pure-neural-without-cheats
baseline (measured below). Document any regression with the upstream
neural fix that needs to land first.

## Pure-neural-without-cheats smoke baseline (2026-06-09)

Command:
```
CUDA_VISIBLE_DEVICES=1 C4_DISABLE_BATCHED_ALU_RECOVERY=1 \
  python -m pytest c4_release/tests/test_smoke.py --tb=no -q
```

Result: **28 passed, 23 failed, 1 deselected in 66.23s** (28/51).

Failing tests include the binary-ALU clusters (`test_shl`, `test_shr`,
`test_add_16bit`, `test_add_carry_cascade`, `test_sub_16bit`,
`test_or_16bit`, `test_xor_16bit`, `test_mul_overflow`) and
`test_cmp_and_branch` — exactly the regressions the
RUNNER_OVERRIDE_REMOVAL_PLAN predicted ("drops smoke back to 28/52").

Note: `C4_DISABLE_BATCHED_ALU_RECOVERY=1` disables ONLY entries #46/#47
(and #1's call sites at batched 2258/2330). The serial-runner overrides
in run_vm.py:2229-2397 remain — they are not gated on any env var. The
true "pure forward only" baseline therefore requires additional code
deletion beyond just flipping this flag.
