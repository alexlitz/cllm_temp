#!/usr/bin/env python3
"""Batch-attribute multiple 1096 cases reusing ONE compiled runner.

Outputs each per-case markdown brief to .agent-logs/1096_fail_<label>.md,
and writes a summary JSON to .agent-logs/multicluster_summary.json.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


# Cases to run per cluster
CASES = [
    # if_gt cases with expected=1 (force divergence)
    "if_gt_6", "if_gt_9",
    # if_lt cases with expected=1 (force divergence)
    "if_lt_1", "if_lt_2",
    # if_var cluster (often blocked per memory note)
    "if_var_0", "if_var_5",
]


def main():
    # Force production-default env for determinism
    os.environ.setdefault("C4_BATCH_USE_KV_CACHE", "0")
    os.environ.setdefault("C4_SPEC_K", "0")
    os.environ.setdefault("C4_DECLARATIONS_ONLY_BAKE", "1")

    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from tools.attribute_1096_failure import attribute_test, _default_out_path

    # Single compile
    print("[batch] compiling runner once...", flush=True)
    runner = BatchedPureNeuralRunner(max_seq_len=4096)
    print("[batch] runner compiled.", flush=True)

    summary = []
    base = os.path.join(_PKG, ".agent-logs")
    os.makedirs(base, exist_ok=True)
    for label in CASES:
        print(f"[batch] attributing {label} ...", flush=True)
        try:
            attr, md = attribute_test(label, runner=runner, declarations_only=True)
        except Exception as exc:
            print(f"[batch] ERROR for {label}: {exc!r}", flush=True)
            summary.append({
                "label": label,
                "error": repr(exc),
            })
            continue
        out_path = _default_out_path(label)
        with open(out_path, "w") as fh:
            fh.write(md)
            if not md.endswith("\n"):
                fh.write("\n")
        summary.append({
            "label": label,
            "test_idx": attr.test_idx,
            "description": attr.description,
            "suite_expected": attr.suite_expected,
            "declarative_exit": attr.declarative_exit,
            "neural_exit": attr.neural_exit,
            "divergence_step": attr.divergence_step,
            "divergence_slot": attr.divergence_slot,
            "expected_token": attr.expected_token,
            "neural_token": attr.neural_token,
            "first_mismatch_op": attr.first_mismatch_op,
            "first_mismatch_layer": attr.first_mismatch_layer,
            "first_mismatch_file": attr.first_mismatch_file,
            "first_mismatch_line": attr.first_mismatch_line,
            "first_mismatch_reason": attr.first_mismatch_reason,
            "notes": attr.notes,
        })
        print(f"[batch] {label}: neural_exit={attr.neural_exit} decl_exit={attr.declarative_exit} "
              f"div={attr.divergence_step}:{attr.divergence_slot} "
              f"op={attr.first_mismatch_op}", flush=True)

    summary_path = os.path.join(base, "multicluster_summary_2.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[batch] wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
