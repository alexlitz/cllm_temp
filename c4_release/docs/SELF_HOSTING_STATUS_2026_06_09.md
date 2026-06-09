# Self-Hosting / Quine Verification Status — 2026-06-09

## Summary

The BLOG_SPEC §920 claim "the neural VM can bundle programs that run on the
neural VM" via `bundler/neural_bundler.c` was tested end-to-end. Two
host-runnable bugs and one self-hosting blocker prevent the documented
canonical flow from completing on native x86-64 Linux. None of the bugs
are in the L10/L14/L15 memory cluster; all live in bundler scaffolding.

## File inventory

| Path | Size |
|---|---|
| `c4_release/bundler/neural_bundler.c` | 27,694 |
| `c4_release/bundler/bundle_c4.c` | 11,581 |
| `c4_release/bundler/neural_bundler.py` | 52,915 |
| `c4_release/bundler/neural_runtime.c` | (large, 1255 lines) |
| `c4_release/tools/neural_bundle_fixedpoint.py` | 18,847 |
| `c4_release/tools/bundle_executable.py` | 12,904 |
| `c4_release/tools/bundle_onnx.py` | 5,688 (executable) |

## CLI inventory

- `tools/neural_bundle_fixedpoint.py` — `--program FILE --output FILE`.
  Requires PYTHONPATH including `old/` for `sparse_vm` (see Blocker 1).
- `tools/bundle_executable.py` — `--program FILE --output FILE`,
  optional `--runtime`, `--weights`, `--minimal`.
- `tools/bundle_onnx.py` — `model output [--compile]`.
- `bundler/neural_bundler.c` — `model.c4onnx program.c neural_runtime.c
  > out.c`. Embedded c4-style compiler.
- `bundler/bundle_c4.c` — `bytecode.bin weights.bin > out.c`.
  Takes pre-compiled bytecode, not source.

## Basic bundling flow — FAIL

Source: `int main() { return 42; }` → expect exit 42.

1. `tools/neural_bundle_fixedpoint.py` produced a 112 KB C file
   (bytecode 16 B, sparse weights 968 nnz). Compiled cleanly with
   `gcc -O2 -w -static-libgcc` after prepending `#include <stdio.h>` and
   `#include <stdlib.h>`. Running the binary returned **exit 248**, not 42.
2. `tools/bundle_executable.py --minimal` produced a clean bundle that
   compiled with no header workarounds. Running it returned **exit 0**.
3. `bundler/neural_bundler.c` (compiled with gcc, `-fpermissive`) bundled
   `small.c` against `transformer_vm.c4onnx`. Resulting C compiled and
   returned **exit 0**.

## Quine flow — FAIL (blocked at step 1)

`./neural_bundler transformer_vm.c4onnx neural_bundler.c neural_runtime.c`
hangs at 99% CPU for 4+ minutes producing zero bytes of output, then
killed at 60s timeout in a second attempt. Same bundler completes in
3 ms on a 10-line `medium.c`. The embedded compiler cannot ingest a
27 KB C file in reasonable time. No diagnostic output is emitted.

## Concrete bugs

### Bug 1: EXIT semantics mismatch (root cause of basic-flow failure)

c4 compiles leaf `int main() { return 42; }` to `[IMM 42, EXIT]`
(see `c4_release/src/compiler.py:434 _strip_leaf_main_startup`). The
project's reference VM treats EXIT as "return ax" — see
`c4_release/neural_vm/fully_neural_vm.py:351` (`return ax & 0xFFFFFFFF`).

All three bundler runtimes implement EXIT as
`return mem_read_int(sp)`:

- `tools/bundle_executable.py` → minimal VM, line 148:
  `else if (op == EXIT) { return (int)mem_read_int(sp); }`
- `tools/neural_bundle_fixedpoint.py` → generated step, line 513:
  `else if (op == EXIT_OP) { halted = 1; return mem_ri(sp); }`
- `bundler/bundle_c4.c` → emitted step, line 276:
  `else if (op==EXIT_OP) { halted=1; return (int)mem_ri(sp); }`

For the leaf-main shape there is no PSH before EXIT, so sp points at
zero-initialized stack and the bundle returns 0.

**Proposed fix**: emit `return (int)ax;` on EXIT in all three sites,
matching `fully_neural_vm.py` semantics.

### Bug 2: 32-bit int width (root cause of fixedpoint exit=248)

`tools/neural_bundle_fixedpoint.py` declares `int *code` and reads
`code[pc/8]` (lines 428, 471 of the generated bundle). On x86-64 `int`
is 32 bits, but `bundled_bytecode` is packed 8 bytes per instruction
(`struct.pack('<Q', instr)`). Reading only 4 bytes from offset 0 happens
to yield op=1, imm=42 by luck, then pc advances to 8 and `code[1]`
reads bytes 4–7 (zeros) instead of the EXIT at bytes 8–15.

**Proposed fix**: emit `long long *code;` (or `int64_t`) and read
`code[pc / 8]` as 64-bit. The minimal VM in `bundle_executable.py`
already gets this right by using `int64 *code`.

### Bug 3: Embedded c4 compiler in `neural_bundler.c` does not scale

Self-bundling hangs indefinitely on the 27 KB `neural_bundler.c`
source. The embedded compiler has `MAX_SYMS = 256` (line 510, init), no
diagnostic on overflow, and apparent quadratic or worse behavior on
large inputs. Cannot diagnose further without instrumenting the
bundler; recommend adding a symbol-count overflow check and a timing
trace before claiming the QUINE path is exercised by CI.

## Files referenced (absolute)

- `/home/alexlitz/Documents/misc/c4_release/c4_release/bundler/neural_bundler.c`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/bundler/bundle_c4.c`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/tools/neural_bundle_fixedpoint.py`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/tools/bundle_executable.py`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/src/compiler.py`
- `/home/alexlitz/Documents/misc/c4_release/c4_release/neural_vm/fully_neural_vm.py`

## Scope notes

- No L10/L14/L15 memory cluster code touched.
- No production code modified; bugs documented here for follow-up.
- Bundle outputs and bundler binary placed under `/tmp/c4_quine_test/`
  for inspection.
