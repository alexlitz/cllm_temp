# C Runtime Path Audit — STATUS

Date: 2026-05-29
Branch: speedup-cache-and-buckets (HEAD: 4d069f7)
Working directory: /home/alexlitz/Documents/misc/c4_release

## TL;DR

**The C runtime path is non-functional. Build is broken at two layers, and the
`test_c_runtime_1096.py` script silently falls back to Python VM execution.
No actual C-runtime pass/fail counts can be produced today.**

## What was attempted

Per the agent brief, the audit ran:

1. `python tests/test_c_runtime_1096.py --quick` — quick smoke run
2. `python tests/test_c_runtime_1096.py` (full 1096 suite) — full run
3. `bash build_cllm_utils.sh` — full CLLM utility bundler+gcc pipeline
4. Direct `gcc` compile of `bundler/simple_c_runtime.c` and `bundler/neural_runtime.c`

Logs saved alongside this file in `.agent-logs/c-runtime-audit/`:

- `run.log` — full 1096 run output (Python baseline fallback)
- `quick.log` — `--quick` (100-test) run
- `build_cllm.log` — `build_cllm_utils.sh` output (13/13 utils failed)

## Findings

### Finding 1 — `test_c_runtime_1096.py` is a `main()` script, not a pytest module

The brief's recipe (`python -m pytest -q -s c4_release/tests/test_c_runtime_1096.py`)
collects zero tests:

```
collected 0 items
========================= no tests collected in 0.05s ==========================
```

The file is a CLI script with `argparse` and `main()`. It was run directly as
`python tests/test_c_runtime_1096.py [--quick]`.

### Finding 2 — C runtime templates do not compile standalone (by design)

`bundler/simple_c_runtime.c` and `bundler/neural_runtime.c` are *runtime
templates* that reference unbound symbols (`program_code`, `program_code_len`,
`program_data`, `program_data_len`, `embedded_model`). They are expected to be
concatenated with bundler-emitted per-program data by `bundler/neural_bundler.py`.

`tests/test_c_runtime_1096.py` line 246 attempts to compile
`bundler/simple_c_runtime.c` standalone, which fails:

```
bundler/simple_c_runtime.c:74:25: error: 'program_data_len' undeclared
bundler/simple_c_runtime.c:75:45: error: 'program_data' undeclared
bundler/simple_c_runtime.c:89:31: error: 'program_code_len' undeclared
bundler/simple_c_runtime.c:91:14: error: 'program_code' undeclared
```

The script then sets `args.skip_compile = True` and falls back to running tests
through `BakedC4Transformer.run_c()` (Python). **No C runtime is exercised.**

The same template-must-be-bundled situation applies to
`tests/runners/c_runtime_runner.py::CRuntimeRunner` — its `setup()` will
likewise fail on standalone compilation. That class is therefore unusable
today unless callers bundle programs first.

### Finding 3 — System toolchain is missing `libgcc_s.so` symlink

The bundler-produced `*_cllm_bundled.c` files (from `build_cllm_utils.sh`) get
through preprocessing but fail at link with:

```
/usr/bin/ld: cannot find -lgcc_s: No such file or directory
collect2: error: ld returned 1 exit status
```

Root cause: `/usr/lib64/libgcc_s.so.1` is present but the unversioned
`/usr/lib64/libgcc_s.so` symlink is missing. Even `hello world` will not link.

Workaround verified: `gcc -static-libgcc ...` produces a working binary
(`echo-cllm` built this way and executed). The `build_cllm_utils.sh` script
does **not** pass `-static-libgcc`, so all 13 CLLM utilities fail:

```
=== Build Summary ===
  OK:     0
  Failed: 13
```

Fixing this is a system-config change (root) or a one-line patch to
`build_cllm_utils.sh` to add `-static-libgcc`. Out of scope for this audit
(brief: "A clean failure report is the deliverable when fixing is out of scope.").

### Finding 4 — Python fallback baseline (recorded for comparison)

Even though no C runtime ran, the script reported a Python-baseline pass rate
through `BakedC4Transformer.run_c()` (note: this is the *non-baked* compile/run
path, distinct from the declarative neural-VM under audit elsewhere):

| Run     | Total | Passed | Failed | Errors | Pass rate | Wall  |
|---------|-------|--------|--------|--------|-----------|-------|
| --quick |   100 |    100 |      0 |      0 | 100.00%   | 0.01s |
| full    |  1096 |    559 |    537 |      0 |  51.00%   | 7.77s |

Notable failure pattern (first 10): `var_simple_N` tests all return `0`
instead of the expected literal (`x = 990: expected 990, got 0`). This is a
known divergence in the script's Python fallback path; it does **not**
represent C runtime behaviour. Comparable neural-VM 1096 sweeps in
`.agent-logs/fast-shards-current/` are the correct baseline for that branch.

## Breakage summary (what would need to be fixed)

1. **Build-time:** `bundler/simple_c_runtime.c` cannot be compiled by
   `test_c_runtime_1096.py` without a bundler-emitted prologue. Either:
   - rewrite the script to bundle a representative program first, or
   - point `--runtime-path` at a standalone runtime (none exists today).

2. **System:** missing `/usr/lib64/libgcc_s.so` symlink prevents dynamic
   gcc links. Use `-static-libgcc` or restore the symlink.

3. **Script regression:** `var_simple_*` tests fail in the script's Python
   fallback (`BakedC4Transformer.run_c()` returning 0). Unrelated to the C
   runtime, but means the script's own "baseline" line is misleading.

4. **Test framework:** `test_c_runtime_1096.py` is a CLI script, not a pytest
   module — running it via `python -m pytest` produces "collected 0 items".
   Either rename to `cli_*` or wrap in real `def test_*` functions.

## Verdict

C runtime audit: **build broken, cannot compare to neural-path baseline**.
A clean failure report is the deliverable per the brief.
