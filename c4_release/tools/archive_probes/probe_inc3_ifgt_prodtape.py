#!/usr/bin/env python3
"""Inc-3 if_gt: run the REAL production fail-fast decode for one if_gt program and
dump the resulting per-step slices of ``s.context`` (the AUTHORITATIVE tape the
full_trace verdict reads) — so we see exactly how step4 is framed and where the
REG_PC the verdict decodes actually sits. Not greedy, not oracle: the production
spec-decode path.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_prodtape.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")
_NAMES = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "MEM", 262: "SE",
          263: "HALT", 268: "ST0"}


def main():
    STEP = int(Token.STEP_TOKENS)
    bytecode, data = compile_c(SRC)
    p = build_groundtruth_probe()
    runner = p.runner
    # Run the production fail-fast decode for this one program. Patch in a hook
    # to capture the final s.context by running run_batch_fail_fast directly.
    res = runner.run_batch_fail_fast(
        [bytecode], data_list=[data or b""], spec_k=0, criterion="full_trace",
        max_steps=10)
    r = res[0]
    print(f"=== PRODUCTION full_trace: {SRC!r} ===")
    print(f"  status={r['status']} div_step={r.get('divergence_step')} "
          f"exp=(pc={r.get('expected_pc')},ax={r.get('expected_ax')}) "
          f"got=(pc={r.get('got_pc')},ax={r.get('got_ax')})")
    # Re-run to grab the context (run_batch_fail_fast doesn't return it). Build the
    # state and step it via the same path, capturing context.
    states = [runner._build_element(bytecode, data or b"", [], "", spec_k=1,
                                    adaptive_start_k=0, expected_steps=None)]
    states[0].ff_oracle_steps = runner._oracle_pc_ax_steps(
        bytecode, data or b"", "", expected_steps=None)
    runner._reset_kv_cache(); runner._reset_spec_stats()
    runner._run_fail_fast(states, max_steps=10, max_context_window=512,
                          spec_k=1, criterion="full_trace")
    s = states[0]
    ctx = s.context
    pl = s.prefix_len
    nsteps = (len(ctx) - pl) // STEP
    print(f"  prefix_len={pl} total_emitted={len(ctx)-pl} nsteps_by_slice={nsteps}")
    for step in range(nsteps + 1):
        sl = ctx[pl + step * STEP: pl + (step + 1) * STEP]
        if not sl:
            break
        disp = [_NAMES.get(t, t) for t in sl]
        # decode PC from this slice
        pc = None
        for i, tk in enumerate(sl):
            if tk == Token.REG_PC and i + 4 < len(sl):
                pc = sl[i+1] | sl[i+2] << 8 | sl[i+3] << 16 | sl[i+4] << 24
                break
        print(f"  step{step} pc={pc}: {disp}")


if __name__ == "__main__":
    main()
