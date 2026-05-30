# ONNX Runtime Path Audit — STATUS

**Date:** 2026-05-29
**Branch:** `audit/onnx-runtime` (off `speedup-cache-and-buckets` @ 4d069f7)
**Auditor unit:** ONNX runtime path audit (64 representative IDs vs PyTorch)
**Verdict:** **BROKEN** — canonical ONNX export of the production VM does not
produce a usable ONNX file with either the TorchScript-based or the dynamo-based
PyTorch exporter. ONNX runtime comparison on the 64 sampled IDs could not be
exercised; only the PyTorch reference side was collected.

---

## 1. Canonical entry points discovered

| Path | Role | Status |
|------|------|--------|
| `c4_release/scripts/export_onnx.py` | Production ONNX export probe via `torch.onnx.export` on `compile_full_vm()` | Fails (see §3) |
| `c4_release/tests/test_onnx_runtime_1096.py` | "Compare 1096 tests through ONNX vs PyTorch" harness | Stale — imports `build_full_vm` from `neural_vm.vm_step`, which no longer exists; falls back to PyTorch-only validation. ONNX side is documented inside the script as `NotImplementedError`. |
| `c4_release/tests/runners/onnx_runner.py` | `ONNXVMRunner.setup()` returns `False` with the message `"ONNX runtime execution not implemented"` and points users at `--mode fast` / `--mode transformer`. | Intentionally inert |
| `c4_release/tests/test_onnx_export.py` | Component-level export tests (FFN, embedding, attention, softmax1) — does **not** export the full VM | Runs (see §4) |
| `c4_release/bundler/neural_bundler.py` | C bundler that bakes weights into a C runtime; **does not produce ONNX**, name "ONNX" only refers to the `.c4onnx` custom weight container | N/A |
| `c4_release/tools/bundle_onnx.py` + `tools/onnx_to_c4.py` | Convert pre-existing ONNX → custom `.c4onnx` blob → bundled C | Inverse direction; not a VM-export path |
| `vm_8bit/onnx_export.py` | Separate 8-bit experimental VM; unrelated to `compile_full_vm()` model | Out of scope for this audit |

`c4_release/scripts/export_onnx.py` is the canonical invocation. The script
already documents (in its module docstring) several known ONNX-relevant
blockers (e.g. `.item()` syncs in `FlattenedALUMul`/`FlattenedDivMod`, Python
control flow in `NeuralVMEmbedding._inject_mem_store`). This audit confirms
the blockers are still real and adds two more concrete failure modes (§3).

## 2. Reproduction (recipe used)

```bash
mkdir -p .agent-logs/onnx-audit

# TorchScript-based exporter, opsets 17, 18, 20
PYTHONPATH=.:c4_release timeout 900 python -u c4_release/scripts/export_onnx.py \
    --output .agent-logs/onnx-audit/c4_neural_vm.onnx --seq-len 32 \
    > .agent-logs/onnx-audit/export_full.log 2>&1            # opset 17 default
PYTHONPATH=.:c4_release timeout 900 python -u c4_release/scripts/export_onnx.py \
    --output .agent-logs/onnx-audit/c4_neural_vm_opset18.onnx --opset 18 --seq-len 32 \
    > .agent-logs/onnx-audit/export_opset18.log 2>&1
PYTHONPATH=.:c4_release timeout 900 python -u c4_release/scripts/export_onnx.py \
    --output .agent-logs/onnx-audit/c4_neural_vm_opset20.onnx --opset 20 --seq-len 32 \
    > .agent-logs/onnx-audit/export_opset20.log 2>&1

# Dynamo (torch.export) exporter
PYTHONPATH=.:c4_release timeout 600 python -u .agent-logs/onnx-audit/dynamo_export_probe.py \
    > .agent-logs/onnx-audit/export_dynamo.log 2>&1

# Component-level tests
PYTHONPATH=.:c4_release timeout 600 python -m pytest -q --tb=short \
    c4_release/tests/test_onnx_export.py 2>&1 \
    | tee .agent-logs/onnx-audit/test_onnx_export.log

# 64 representative ID PyTorch reference (no ONNX side — see §3)
PYTHONPATH=.:c4_release timeout 1800 python -u \
    .agent-logs/onnx-audit/probe_64_ids.py > .agent-logs/onnx-audit/probe_64_ids.log 2>&1
```

Hardware/runtime: Linux 6.12.29, torch 2.10.0+cu128, onnxruntime 1.23.2, Python 3.12.6.

## 3. Failure modes captured

### 3.1 TorchScript exporter (`dynamo=False`) — opset 17, 18, 20

* **Model build:** OK (`compile_full_vm()` returns; eager forward succeeds).
* **Tracing:** OK (graph emitted with ~32 transformer blocks, FFN/attention
  weight tensors enumerated).
* **`_jit_pass_onnx` symbolic lowering:** fails with
  `torch.onnx.errors.UnsupportedOperatorError`.

```
[export_onnx] === EXPORT FAILED ===
  Exception type: UnsupportedOperatorError
  Exception message: Exporting the operator 'aten::cummax' to ONNX opset version 17 is not supported
```

Same failure at opset 18 and opset 20. The TorchScript exporter has **no
symbolic registration for `aten::cummax`** at any opset shipped with
torch 2.10.0.

Offending site (one of seven instances captured in the traced graph):

```
File "/home/.../c4_release/neural_vm/efficient_alu_neural.py", line 180:
    latest_score, latest_idx = torch.cummax(scores, dim=1)
```

Other call sites of the same `bd_to_ge` converter that hit `cummax` (per the
torch IR graph):
`blocks.{9,11,13,20,22,24,26}.ffn.../bd_to_ge`.

### 3.2 Dynamo exporter (`dynamo=True`)

```
torch.onnx._internal.exporter._errors.TorchExportError:
  Failed to export the model with torch.export.
  ...
  Could not guard on data-dependent expression Eq(u0, 1) (unhinted: Eq(u0, 1)).
  Caused by: (_export/non_strict_utils.py:1139 in __torch_function__)
  ...
  The following call raised this error:
    File "/home/.../c4_release/neural_vm/alu/ops/divmod_longdiv.py", line 284, in forward
      if b_is_zero.any():
```

`b_is_zero.any()` is a Python-bool conversion of a tensor — `torch.export`
requires either `torch.cond` / `torch.where`-based control flow or a
data-independent guard. The DivMod long-division op is currently incompatible
with strict graph capture.

### 3.3 Existing harness `test_onnx_runtime_1096.py` is stale

```
ERROR: ONNX export failed: cannot import name 'build_full_vm' from
  'neural_vm.vm_step' (.../c4_release/neural_vm/vm_step.py)
```

The script references a removed factory and silently degrades to PyTorch-only
validation. Even when degraded, the ONNX comparison side raises
`NotImplementedError("Full ONNX execution loop not yet implemented")` (line 109
of the same file). Recommend deleting or rewriting; in its current shape it
cannot fulfil the "ONNX export and 100+ tests" requirement it claims to cover.

## 4. Component-level ONNX export (what *does* work)

`pytest c4_release/tests/test_onnx_export.py` —
**18 passed, 1 skipped, 2 xfailed, 1 xpassed, 2 failed** (118.9 s).

Working in isolation:
* `PureFFN` (SwiGLU FFN), opset 14, exports & round-trips against onnxruntime.
* `torch.nn.Embedding` exports & round-trips.
* Simple linear-only models export & round-trip.
* ARVM (custom binary weight) export path remains functional via
  `tools.export_autoregressive` (ARVM magic `0x4D565241`, version 2).

Newly observed failures (not on the cummax/dynamo paths):
* `TestARVMExport::test_model_embedding_structure` — asserts `d_model == 512`
  but the production model is now `d_model == 736`. Test is stale, not a
  product regression. Out of scope for this audit.
* `TestONNXExportStatus::test_attention_without_alibi_exports` —
  `AutoregressiveAttention.__init__()` no longer accepts `d_model=` (its
  signature changed since the test was written). Stale signature, not a
  product regression.

xfail/xpass:
* `test_attention_layer_exportable` (xfail) — ALiBi broadcasting still
  incompatible.
* `test_alibi_attention_exports` (xfail) — same, documented.
* `test_softmax1_exports` (xfail) — softmax1 (ZFOD) not a standard ONNX op.
* `test_softmax1_exports` shows up as xpassed in this run (the symbolic
  fallback may have improved between torch versions); confirm with a focused
  follow-up if relevant.

## 5. 64-ID representative comparison

Sampling 64 evenly-spaced IDs out of the 1096-test suite (`tests/test_suite_1000.py`):
`[0, 17, 35, ..., 1078, 1095]`.

* **PyTorch reference side:** 34 / 64 = 53.1% match expected
  (30 mismatches, 0 errors, 0.5 s wall). Full per-ID record in
  `probe_64_ids.json` / `probe_64_ids.log`. This pass rate is the pre-existing
  baseline for `BakedC4Transformer.run_c` on the 1096 suite under teacher-forced
  speculation; it is **not** a regression introduced by this audit.
* **ONNX side:** **0 IDs compared.** The ONNX export blockers in §3 prevent any
  comparison; the deliverable here is the clean failure report. The PyTorch
  reference is collected so a follow-up audit (after the cummax / DivMod
  blockers are resolved) has a fixed baseline to diff against.

## 6. Minimum unblocking work (for a follow-up agent)

To turn the production VM into something the TorchScript exporter can lower:

1. Replace the `torch.cummax`-based "latest STACK0_BYTE1 position" lookup in
   `neural_vm/efficient_alu_neural.py:180` with an ONNX-supported equivalent
   (e.g. running max via `torch.max` over a triangular mask, or a custom
   prefix-max built from `cumsum` + `argmax`). Seven call sites listed in §3.1.
2. Either:
   (a) add a TorchScript symbolic for `aten::cummax` (PyTorch already lowers
       `aten::cummax` to ScanOp / Loop in newer versions of the
       `dynamo`-based exporter when wrapped in `torch.cond`), or
   (b) drop the `cummax` usage entirely (option 1).
3. For the dynamo path, refactor `neural_vm/alu/ops/divmod_longdiv.py:284`
   (`if b_is_zero.any():`) to use `torch.where(b_is_zero, ..., ...)` so
   strict-graph capture does not need to guard on a data-dependent symbol.
4. Update `c4_release/tests/test_onnx_runtime_1096.py` to import via
   `neural_vm.unified_compiler.full_vm_compiler.compile_full_vm` (the actual
   production factory), or delete the file — its `build_full_vm` import path
   is dead.
5. Update the two stale assertions in `c4_release/tests/test_onnx_export.py`
   (`d_model 512 → 736`, drop `d_model=` kwarg in
   `test_attention_without_alibi_exports`).

## 7. Artifacts in this directory

| File | Contents |
|------|----------|
| `STATUS.md` | This report |
| `export_full.log` | TorchScript exporter, opset 17 (103 614 lines, ends with `UnsupportedOperatorError: aten::cummax`) |
| `export_opset18.log` | TorchScript exporter, opset 18 (same failure) |
| `export_opset20.log` | TorchScript exporter, opset 20 (same failure) |
| `export_dynamo.log` | Dynamo exporter, opset 18 (`GuardOnDataDependentSymNode` in `divmod_longdiv.py:284`) |
| `export_only.log` | Demonstrates that `test_onnx_runtime_1096.py` cannot import `build_full_vm` |
| `export_attempt.log` | First (partially-buffered) export run, kept for completeness |
| `test_onnx_export.log` | Pytest run of `test_onnx_export.py` (18p / 2f / 1s / 2xf / 1xp) |
| `probe_64_ids.py` | The 64-ID comparison probe |
| `probe_64_ids.log` | Probe stdout |
| `probe_64_ids.json` | PyTorch reference results for the 64 sampled IDs |
| `dynamo_export_probe.py` | Dynamo-exporter probe (mirrors `scripts/export_onnx.py` with `dynamo=True`) |

No production code was modified; no `.onnx` files were produced.
