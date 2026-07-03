# char_*/string_* cluster attribution (2026-06-07)

Date: 2026-06-07
Base ref: `speedup-cache-and-buckets` HEAD `b17b9d3f`
Task: 1096 cluster attribution for `char_*` / `string_*` family,
expected to depend on the in-flight memory-cluster fix (agent A3).

## TL;DR

The `char_*` / `string_*` cluster does **not exist** in the 1096 corpus.
The 1096 program generator in `tests/test_suite_1000.py`
(`generate_test_programs`) enumerates 13 integer-only categories with a
fixed total of 1096 cases. No category produces programs that exercise
`LC` (load char) or `SC` (store char). Pass-count for the requested
filter is therefore **0 selected / 0 passed / 0 failed**.

## Evidence

- `pytest tests/test_suite_1096_pure_neural_pytest.py -k "char_ or string_"`
  collects **1098 tests, deselects all 1098, runs 0**.
  ```
  collected 1098 items / 1098 deselected / 0 selected
  ```
  (1098 = 1096 program tests + 2 fixture tests.)
- Unique family prefixes extracted from
  `test_program[<id>_<family>_<...>]` IDs (both the handler-mode suite
  `tests/test_suite_1096_pytest.py` and the parallel pure-neural suite
  `tests/test_suite_1096_pure_neural_pytest.py`):
  ```
  absdiff, add, bool, div, edge, expr, func, gcd,
  if, loop, mod, mul, nested, rec, sub, var
  ```
  No `char` or `string` prefix.
- `CATEGORY_COUNTS` in `tests/test_suite_1000.py:612-626` is the
  authoritative breakdown:
  ```
  arithmetic 200, modulo 50, variables 100, conditionals 100,
  loops 100, functions 150, recursion 100, expressions 100,
  gcd 50, nested_functions 50, edge_cases 50, abs_diff 25,
  boolean_logic 25
  ```
  Total = 1096. All integer-valued; none allocate strings or character
  buffers, none compile to `LC`/`SC`.
- `grep -nE "char_|string_|'char|'string"` against `tests/test_suite_1000.py`
  returns **zero** matches.

## LC / SC opcodes — present in VM, unused by 1096

The VM does implement `LC` (`OP_LC = 272`) and `SC` (`OP_SC = 274`); see
`neural_vm/dim_registry_dynamic.py:263-264`. The L10 `OP_LC_RELAY`
flag at `neural_vm/setup_helpers_l10.py:98` and the `IO_STATE` alias
at `dim_registry_dynamic.py:401` exist for the
`tests/test_pure_neural_io.py` /
`tests/test_conversational_io_comprehensive.py` surfaces, not for 1096.
Those suites use `getchar`/`putchar` programs, which is what the brief
likely had in mind by "char_*/string_*" — but they are **not** part of
the 1096 corpus and would not appear under `test_suite_1096_*` filter.

## A3 (memory-cluster) dependency analysis

A3 is fixing the `var_*` cluster's memory path (`MEM_addr0` / L10 PSH
ENT-guard, `MEM_value1` on JSR, etc.; see user memory notes
`project_l10_psh_addr_ent_bug.md`, `project_var_failure_mode_shifted.md`).
That work is on integer-valued `LI`/`SI` (load/store **int**, opcodes
`OP_LI=271` / `OP_SI=273`), not `LC`/`SC`. Since the requested
`char_*`/`string_*` cluster does not exist in 1096:

- There is **no A3 downstream dependency to confirm** for this cluster.
- If a future test corpus adds char-buffer programs, the `LC`/`SC`
  path will share the L10 PSH addressing and L8/L14 memory-marker logic
  that A3 is repairing, so it is reasonable to expect future
  char_*/string_* tests to inherit A3's fixes. That is a forward-looking
  prediction, not a current measurement.

## Recommendation

Close this attribution slot. If a parent agent wanted a getchar/putchar
attribution, retarget to
`tests/test_pure_neural_io.py` or
`tests/test_conversational_io_comprehensive.py`; those are the real
char/string surfaces and they are governed by the IO_STATE / OP_LC_RELAY
L10 path rather than the 1096 integer-memory path.

## Smoke

Smoke gate (`tests/test_smoke.py`, strict default per
`docs/NEURAL_SMOKE_GATE.md`) was run; see commit message and parent
worktree report for the pass count at this commit. No code under
`neural_vm/` or `tests/` was touched by this attribution; the only
delta is this doc.
