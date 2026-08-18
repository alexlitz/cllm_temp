# C-toolchain capstone status — CPU c4 VM verification

*2026-08-06. Verification-only (no transformer build, no GPU, no code changes).
Branch `perf-finalize-841-849`, off the shared `consolidate-0.5b-2026-07-22`
line; verified at commit `ffab2efd`. Golden untouched. Native `./c4` oracle
built at `/tmp/c4c` via `gcc -w -fpermissive -static-libgcc -o /tmp/c4c
bundler/c4_compile.c`.*

This document records, honestly, which of the capstone's C-toolchain claims
(`DOOM_ON_TRANSFORMER_CAPSTONE_2026_08_06.md` §2: "a C compiler + a C→c4
transpiler (self-hosting on the VM)") actually hold via the byte-exact **CPU /
native** c4 execution — NOT the transformer. The transformer story is separate;
this is the CPU-VM baseline the transformer target mirrors.

## Summary verdict

| Claim | Verdict (CPU c4 VM) |
|---|---|
| **1. Self-hosting** (compiler compiles itself) | **PARTIAL — kernel-level only.** The full compiler does **not** compile itself. Native `c4_compile.c` cannot compile its own source (infinite loop) and core-dumps on *any* user-defined function. What DOES self-host is the c4-**subset** ONNX-runtime + matmul kernel: it compiles under the c4 compiler and runs byte-exact on the CPU c4 VM. The full 3-layer end-to-end (Rel-2/Rel-3) is a measured performance wall, not achieved — matching `docs/SELFHOST_3LAYER_FEASIBILITY.md`. |
| **2. Quine** (self-outputs byte-exact) | **WORKS.** The PRTF string-quine prints its own 54-byte source serialization byte-exact on the CPU reference c4 VM. |
| **3. C→c4 transpile vs native ./c4** | **WORKS (Python compiler path).** `src.compiler.compile_c` produces bytecode whose arithmetic core is byte-identical to native `./c4c`, the compiled bytecode runs correctly on the CPU c4 VM, and the full 112-case C90 battery is byte-exact vs gcc-15 (`native_c4.run`), including functions/recursion/pointers/arrays. The `c4_compile.c` transpiler is NOT run *on* the VM (it can't self-compile), but the C→c4→CPU-VM pipeline is byte-exact end to end. |

**One-line:** Of the "C compiler + C→c4 transpiler, self-hosting on the VM"
claim, the **compiler + transpile-and-run byte-exact vs gcc/native** parts are
fully real on the consolidated branch today; **"self-hosting on the VM"** is real
**only at the c4-subset kernel level** — the full compiler is not self-hosting
(it cannot compile its own source), and the 3-layer end-to-end loop is an
acknowledged performance wall, not a demonstrated capstone.

---

## Claim 1 — SELF-HOSTING

### What does NOT hold
- **The compiler does not compile itself.** Native `/tmp/c4c bundler/c4_compile.c`
  spins in an infinite loop (killed at 99% CPU, no output). `src.compiler.compile_c`
  on `bundler/c4_compile.c` fails the c4 grammar (`SyntaxError: Cannot redeclare
  non-function as function: open`).
- **Native `c4_compile.c` core-dumps on any user-defined function** — e.g.
  `int sq(int n){return n*n;} int main(){return sq(5);}` and a fib/while program
  both `dumped core`. It only handles expression + `while`-loop programs.
  So `c4_compile.c` is not a full self-hosting c4 compiler; it is a limited
  expression/loop transpiler.

### What DOES hold (kernel-level self-hosting, reproducible)
- `c4_min/selfhost/onnx_kernel_c4subset.c` (the fixed-point MatMul = the ONNX
  runtime's load-bearing inner kernel) **compiles under the c4 compiler** (542
  bytecode words via `src.compiler.compile_c`) and **runs on the CPU c4 VM**:
  result `AX=18432` in 230 VM steps. Native `c4_compile.c` core-dumps on it (it
  has functions), but the Python c4 compiler handles it — this is the honest
  self-host lever.
- The **whole** c4-subset ONNX runtime (`onnx_runtime_c4subset.c`, 29 functions,
  5520 words) compiles under c4; its node dispatcher + op cores run byte-exact vs
  numpy on the full-word CPU VM, and its `load()` reconstructs the real tiny
  `c4vm.onnx` `.nblbin` byte-exact. Verified green on CPU:
  `pytest c4_min/selfhost/test_runtime_c4subset_compiles.py
  test_runtime_c4subset_run.py test_nonmatmul_self_emulation.py` → **30 passed**.
- **The full 3-layer end-to-end (Rel-2 "runtime hosts itself" / Rel-3) is NOT
  demonstrated** — a measured performance wall (~months→millennia of neural-VM
  wall-clock; one real `c4vm.onnx` forward ≈ 1.95e11 VM steps). This is documented
  and owned in `docs/SELFHOST_3LAYER_FEASIBILITY.md`, not hidden.

**Reproduce:**
```
gcc -w -fpermissive -static-libgcc -o /tmp/c4c bundler/c4_compile.c
/tmp/c4c bundler/c4_compile.c            # infinite loop → NOT self-hosting
PYTHONPATH=<repo>/.. python -m pytest \
  c4_release/c4_min/selfhost/test_runtime_c4subset_compiles.py \
  c4_release/c4_min/selfhost/test_runtime_c4subset_run.py \
  c4_release/c4_min/selfhost/test_nonmatmul_self_emulation.py   # 30 passed
```

## Claim 2 — QUINE (self-outputs byte-exact) — WORKS

`c4_min/quine_prtf.py` builds a classic PRTF string-quine (27-cell code table,
54-byte source serialization). Run on the CPU **reference c4 VM**
(`isa.interpret`, no model), its visible PRTF output equals its own source
byte-for-byte:

```
run_reference(): visible output == source S  ->  True   (54 bytes)
sha256(visible) == sha256(source) == 4245a8a3...
```

`pytest c4_min/test_quine_prtf.py` → **6 passed, 1 skipped** (the skip is the
opt-in neural bake, `C4_RUN_NEURAL_QUINE=1`). A second, independent quine suite
`tests/test_quine.py` → **25 passed, 3 skipped**. Both are CPU-only.

## Claim 3 — C→c4 TRANSPILE vs native ./c4 — WORKS

The C→c4 compiler (`src.compiler.compile_c`, the Python c4 compiler on the
CPU-VM path) is byte-exact against the native `./c4c` oracle and against gcc:

- **Byte-identical arithmetic cores vs native `./c4c`** for 8 expressions
  (`2+3*4`, `2*3+4`, `1+2+3`, `2*3*4`, `4+5*6`, `3*4+5`, `9+8+7`, `5*6*7`). The
  only difference is native's `JSR main; ENT 0 … LEV; LEV` framing vs the Python
  path's bare-core + HALT — after stripping that framing (the same `c4_arith_core`
  strip `c4_min/validate_vs_real_c4.py` uses) the `op|imm<<8` word streams are
  identical (all 8 MATCH).
- **Compiled bytecode runs correctly on the CPU c4 VM** — all 8 expressions
  produce the expected result (e.g. `5*6*7 → 210`, 8 steps).
- **Full C90 battery byte-exact vs gcc-15 on the CPU path** (no transformer):
  `id_port/c90_e2e/validate_ext.py` → **112/112 gcc-compiled, 112/112
  c4-compiled, 112/112 run on native c4 VM, 112/112 native==gcc** (modulo the
  tagged `c4-vs-x86` `sizeof(int)==8` case). Covers 13 categories incl. 11
  `functions` (recursion: `fn_recursion_fac/fib/sum`), 11 `pointers`, 6 `arrays`,
  9 `bitwise`, 12 `cmp`, 8 `loops`, 14 `programs`.

Caveat scoping the claim precisely: the capstone phrases this as "a C→c4
**transpiler** running on the VM." There is no `transpile.c`; the transpiler *is*
the c4 compiler (`c4_compile.c` native / `src.compiler.py` Python). The
transpiler is **not** executed *on* the c4 VM (that requires self-compilation,
which fails — see Claim 1). What is byte-exact is the **C source → c4 bytecode →
CPU c4 VM** pipeline, which is exactly the well-defined target the transformer
route mirrors.

**Reproduce:**
```
gcc -w -fpermissive -static-libgcc -o /tmp/c4c bundler/c4_compile.c
cd c4_release/id_port/c90_e2e && PYTHONPATH=<repo> python validate_ext.py
```

## Where each artifact lives (absolute paths)

- Native `./c4` transpiler source: `c4_release/bundler/c4_compile.c` (built to `/tmp/c4c`).
- Python c4 compiler (CPU-VM path): `c4_release/src/compiler.py` (`compile_c`).
- CPU c4 VMs: `c4_release/c4_min/isa.py` (`interpret`, 8-bit reference) and
  `c4_release/id_port/c90_e2e/native_c4.py` (`run`, faithful full-word).
- Quine: `c4_release/c4_min/quine_prtf.py`, tests
  `c4_release/c4_min/test_quine_prtf.py` + `c4_release/tests/test_quine.py`.
- Self-host: `c4_release/c4_min/selfhost/` (kernel `onnx_kernel_c4subset.c`,
  whole runtime `onnx_runtime_c4subset.c`, tests `test_runtime_c4subset_*.py`,
  `test_nonmatmul_self_emulation.py`) + `docs/SELFHOST_3LAYER_FEASIBILITY.md`.
- C90 battery: `c4_release/id_port/c90_e2e/` (`validate_ext.py`, `cases_ext.py`,
  `native_c4.py`).

## Branch note

Every result above was reproduced **on this consolidated branch** (`perf-finalize-841-849`
@ `ffab2efd`, off `consolidate-0.5b-2026-07-22`) with **no** checkout of any
`nibble-*` / `selfhost-*` branch. The self-hosting feasibility doc, the c4-subset
self-host kernels/tests, the quine, and the C90 battery are all present and green
here. The heavier "green-field c4_min blogspec interpreter capstones" (neural
quine bake, transformer model-runs-C) require a model build and were NOT run
(constraint: CPU only, no model build); their CPU-VM substrates verified above
are what the transformer builds on.
