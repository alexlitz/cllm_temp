#!/usr/bin/env python3
"""Load-bearing measurement for the l10_loop_lea_b0_e8 / _e0 restore ops
(PROJECT_0XE8_SLAM). Runs the memory / address / function / bitwise / alu
smoke bytecodes through the authoritative BatchedPureNeuralRunner and reports
the emitted result for each, so a caller can diff the CURRENT env against a
kill-switch env (C4_LOOP_LEA_B0_E8=0 C4_LOOP_LEA_B0_E0=0).

Run twice (flags default vs flags=0) and diff:

  C4_VM_CACHE_DIR=/tmp/x1 python tools/_probe_e8_slam_loadbearing.py > on.txt
  C4_LOOP_LEA_B0_E8=0 C4_LOOP_LEA_B0_E0=0 C4_VM_CACHE_DIR=/tmp/x2 \
      python tools/_probe_e8_slam_loadbearing.py > off.txt
  diff on.txt off.txt

Diagnostics only; no weight edits.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    from tests.test_smoke import _SMOKE_GROUPS
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from neural_vm.run_vm import AutoregressiveVMRunner

    model_runner = AutoregressiveVMRunner(
        pure_neural=True, trust_neural_alu=True, spec_k=0)
    model_runner._func_call_handlers = {}
    model_runner._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=model_runner)
    model = runner.model if hasattr(runner, "model") else getattr(
        model_runner, "model", None)

    groups = ("memory", "address", "functioncall", "bitwise", "basic",
              "shift", "bit32", "comparison", "controlflow")
    nblocks = len(model.blocks) if model is not None else "?"
    print(f"[loadbearing] blocks={nblocks} "
          f"E8={os.environ.get('C4_LOOP_LEA_B0_E8', '1')} "
          f"E0={os.environ.get('C4_LOOP_LEA_B0_E0', '1')}")
    npass = nfail = 0
    for g in groups:
        tests = _SMOKE_GROUPS.get(g)
        if not tests:
            continue
        for t in tests:
            bc = t["bytecode"]
            ms = t.get("max_steps", 30)
            try:
                res = runner.run_batch([bc], data_list=[b""], max_steps=ms,
                                       spec_k=0)
                # run_batch -> List[(output_str, exit_code)]. The register
                # result is the exit_code.
                _out, val = res[0]
            except Exception as exc:  # noqa: BLE001
                val = f"ERR:{exc!r}"
            chk = t.get("check")
            ok = ""
            if chk is not None:
                # _eq(...) ASSERTS on mismatch (returns None on match).
                try:
                    chk(val)
                    ok = "PASS"
                    npass += 1
                except AssertionError:
                    ok = "FAIL"
                    nfail += 1
                except Exception:  # noqa: BLE001
                    ok = "?"
            print(f"  {t['name']:52s} -> {str(val):>10}  {ok}")
    print(f"[loadbearing] PASS={npass} FAIL={nfail}")


if __name__ == "__main__":
    main()
