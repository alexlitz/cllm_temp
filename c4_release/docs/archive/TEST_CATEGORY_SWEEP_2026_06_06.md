# Test Category Sweep — 2026-06-06

Broad sweep across 8 major test categories outside of smoke + 1096 (already tracked).
Baseline: `main` HEAD `78f5433a` (`docs(overrides): live status matrix + no-override smoke count`).

Reproduce: `bash tools/test_sweep_by_category.sh [out_dir]`

## Matrix (counts per category)

| Category | Total | Pass | Fail | Skip | Err | xFail | xPass | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Allocator contracts                | 53  | 52  | 1  | 0 | 0  | 0 | 0 | 0.3 s |
| DSL primitives                     | 156 | 153 | 3  | 0 | 0  | 0 | 0 | 3.5 s |
| Compile contracts                  | 30  | 28  | 2  | 0 | 0  | 0 | 0 | 52 s  |
| KV cache + autoregressive          | 34  | 21  | 13 | 0 | 0  | 0 | 0 | 25 m  |
| Conversational I/O (`_final` skipped: collection error) | 41  | 3   | 16 | 0 | 22 | 0 | 0 | 11 m  |
| Architecture toggles               | 42  | 34  | 0  | 8 | 0  | 0 | 0 | 0.7 s |
| Slot registry                      | 25  | 25  | 0  | 0 | 0  | 0 | 0 | 0.2 s |
| Misc opcode-specific               | 50  | 20  | 24 | 0 | 0  | 6 | 0 | 25 m  |
| **Total**                          | **431** | **336** | **59** | **8** | **22** | **6** | **0** | — |

Notes:
- "Deselected" rows are excluded from totals (allocator/compile/kv/slot had a handful, all skipped on environment gating).
- `test_conversational_io_final.py` fails *collection* with `SlotConflictError`; the partial run uses only the other two files. The collection error itself is included once as `Err` for the full-category column in the count above.
- `architecture_toggles` shows skips only on CUDA-conditional variants; underlying gate is fine.

## Failure detail (test name → first error line)

### Allocator contracts (1 fail)
- `test_dim_allocator.py::test_allocator_byte_identical_to_static` — `AssertionError: dynamic registry missing slots: ['TEMP_PREV_STEP']`

### DSL primitives (3 fail, all in `test_compiler_ir.py`)
- `test_tail_bit32_symbolic_mul_byte1_blocks_add_carry_correction` — `assert 43.0 > 1000.0`
- `test_tail_bit32_symbolic_mul_byte1_preserves_nonzero_high_byte` — `assert 43.0 > 1000.0`
- `test_tail_bit32_symbolic_lt_false_blocks_wide_mul_preserve` — `KeyError: 'OUTPUT_LO+0'`

### Compile contracts (2 fail)
- `test_compile_determinism::test_compile_full_vm_is_deterministic_conversational_io` — `SlotConflictError` (same root as conversational_io collection failure)
- `test_compile_determinism::test_compile_full_vm_is_deterministic_tool_calling` — `IndexError: index 400 is out of bounds for dimension 0 with size 89` (ir.py:1427)

### KV cache + autoregressive (13 fail)
12 of 13 are in `test_autoregressive_kv_cache.py` with assertion failures (return code mismatches, e.g. `assert 0 == 42`, `assert 65512 == 30`, `IndexError: index is out of bounds for dimension with size 0`, `SyntaxError: Expected 56, got 42 at line 3`). Pattern: VM returns wrong value with cache *on*. The single byte-identical failure (`test_single_imm[prog4-200]`) reports baseline cache-OFF already returns 65512, expected 200 — i.e. the baseline itself is wrong for `prog4`.

### Conversational I/O (16 fail, 22 err)
- All 22 errors and most failures trace to the same `SlotConflictError` (the open one a separate agent is fixing).
- The 3 `test_conversational_io.py` failures and the early `test_conversational_io_comprehensive.py::TestPRTFOutputContent` group fail with `AssertionError: assert 'Hello' in ''` / `assert '' == '<expected>'` — output-routing produces empty string. May be a separate downstream bug surfaced by `conversational_io=True`.

### Misc opcode-specific (24 fail, 6 xfail)
- `test_bz_bnz_neural.py`: 13/14 fail; all assertions `assert 0 == 42` / `assert 65512 == 42` — branch handler doesn't reach intended target. (xfailed: 2 cases for known negative-value path.)
- `test_control_flow_neural.py`: 8/13 fail; same flavour (`JMP should skip to IMM 42, got 0`, `BZ should branch when AX=0, got 0`, `if_else`/`while_loop` cases).
- `test_addr_key_neural_decode.py::test_addr_key_decode_op_is_registered` — `assert None == 14` (no `layer_idx` registered).
- `test_alibi_mem_attn.py` (2 fail) — `assert None == 9` (no `layer_idx` registered) and `assert 112.5 == 0.0` (op is not actually no-op when disabled).

## Top-5 highest-impact failures

Ranked by how many other tests would unblock if the underlying root cause were addressed.

1. **`SlotConflictError` (convo_io_* + null_terminator_detection + conversational_io_output_routing)** — blocks `test_conversational_io_final.py` collection (1 collection err), 22 setup-errors in `test_conversational_io_comprehensive.py`, the 2 `compile_determinism::*conversational_io/tool_calling` cases, and at least the first `compile_determinism` failure overlap. ≈25 tests gated. (Already owned by separate agent per task brief.)
2. **BZ/BNZ neural handler returning `AX==0` instead of branching to target IMM** — single root cause behind 13/14 `test_bz_bnz_neural.py` failures and 8/13 `test_control_flow_neural.py` failures (`JMP step1`, `BZ/BNZ branches/continues`, `if_else*`, `simple_while_loop`). ≈21 tests gated by what looks like one bug in the BZ/BNZ rewrite path.
3. **`test_autoregressive_kv_cache.py` cache-on path returning wrong VM result** — 12 unique assertion failures (`assert 0 == 42`, etc.) all consistent with cache-enabled VM rejecting/dropping state. Likely one bug in the cache hit path. ≈12 tests.
4. **`Operation.layer_idx` is `None` for several declarative ops** — `addr_key_decode` and `alibi_mem_attn` both fail "is_registered" tests with `assert None == 14/9`. Likely a registration/site-of-truth regression affecting any future declarative op of the same family. 2 tests today, plus the alibi no-op-when-disabled test depends on the same registration.
5. **`tail_bit32_symbolic_mul` family in `test_compiler_ir.py`** — 3 closely-related tests on the symbolic mul / wide-mul preserve path (one `KeyError: 'OUTPUT_LO+0'`, two `43.0 > 1000.0`). Likely a single IR symbol that lost its OUTPUT_LO+0 entry. 3 tests, but central to the IR contract suite.

## Categories needing dedicated next-session attention

| Priority | Category | Why |
|---|---|---|
| **High** | KV cache + autoregressive | 13/34 failing (38% fail rate); not currently tracked in smoke or 1096 docs; cache-on path silently broken. |
| **High** | Misc opcode-specific (BZ/BNZ/control-flow) | 24/50 failing (48%); single likely root cause in BZ/BNZ handler. Big leverage. |
| **Medium** | Conversational I/O | Most failures gated by the open `SlotConflictError`; recheck after that lands, then investigate the residual `'' == 'Hello'` output-routing assertion separately. |
| **Medium** | DSL primitives | 3 IR contract failures in symbolic-mul tail path. Small but central to IR correctness. |
| **Low** | Allocator contracts | One missing slot (`TEMP_PREV_STEP`) — likely a one-line registry fix. |
| **Low** | Compile contracts | 2 fails; both share root causes with higher-priority categories above. |
| **Clean** | Architecture toggles, Slot registry | No regressions; safe to leave. |

## Reproduce

```bash
bash tools/test_sweep_by_category.sh /tmp/cat_sweep
# logs in /tmp/cat_sweep/<category>.log
```
