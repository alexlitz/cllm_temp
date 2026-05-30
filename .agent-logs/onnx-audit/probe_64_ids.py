"""Audit probe: attempts ONNX vs PyTorch comparison on 64 representative IDs.

Since canonical ONNX export is broken (UnsupportedOperatorError: aten::cummax
in TorchScript exporter, and GuardOnDataDependentSymNode in dynamo exporter),
we only collect the PyTorch reference side. The ONNX side cannot be exercised.
"""
import os, sys, time, json, traceback

sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

from tests.test_suite_1000 import generate_test_programs

# Pick 64 representative IDs - even spacing across the 1096 suite
ALL = generate_test_programs()
total = len(ALL)
# Span the full range with 64 samples
indices = [round(i * (total - 1) / 63) for i in range(64)]
samples = [(i, ALL[i]) for i in indices]

print(f"[probe] Loaded {total} tests; sampling {len(samples)} representative IDs")
print(f"[probe] First 5 ids: {indices[:5]}, last 5 ids: {indices[-5:]}")

# === PyTorch reference ===
print("[probe] Building BakedC4Transformer for PyTorch reference...")
t0 = time.time()
from src.baked_c4 import BakedC4Transformer
c4 = BakedC4Transformer(use_speculator=True)
print(f"[probe] Built in {time.time()-t0:.1f}s")

results = []
fails = 0
errs = 0

t0 = time.time()
for n, (orig_id, (source, expected, desc)) in enumerate(samples):
    try:
        got = c4.run_c(source)
        status = "PASS" if got == expected else "FAIL"
        if status == "FAIL":
            fails += 1
    except Exception as e:
        got = None
        status = f"ERROR: {type(e).__name__}: {e}"
        errs += 1
    results.append({
        "sample_idx": n,
        "test_id": orig_id,
        "desc": desc,
        "expected": expected,
        "pytorch_got": got,
        "pytorch_status": status,
        "onnx_got": None,
        "onnx_status": "SKIPPED: ONNX export broken (see STATUS.md)",
        "match": False,  # cannot compare without ONNX
    })
    if (n + 1) % 8 == 0:
        elapsed = time.time() - t0
        print(f"  [{n+1}/{len(samples)}] elapsed={elapsed:.1f}s fails={fails} errs={errs}")

elapsed = time.time() - t0

passed = sum(1 for r in results if r["pytorch_status"] == "PASS")

summary = {
    "total_ids_sampled": len(samples),
    "pytorch_passed": passed,
    "pytorch_failed": fails,
    "pytorch_errored": errs,
    "onnx_compared": 0,
    "onnx_export_status": "BROKEN",
    "elapsed_sec": elapsed,
    "selected_ids": indices,
    "results": results,
}

out_path = "/home/alexlitz/Documents/misc/c4_release/.agent-logs/onnx-audit/probe_64_ids.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[probe] === SUMMARY ===")
print(f"  ids sampled : {len(samples)}")
print(f"  pytorch pass: {passed}")
print(f"  pytorch fail: {fails}")
print(f"  pytorch errs: {errs}")
print(f"  onnx compares: 0 (export blocked)")
print(f"  elapsed: {elapsed:.1f}s")
print(f"  JSON: {out_path}")
