#!/usr/bin/env python3
"""Compare the L10 head-1 byte-1 PREDICTOR row signals between si_li (IMM addr,
leak) and var_simple (LEA addr, needs slot-82). Dumps the candidate value/addr
rows + the predictor-row Q gates so we can find a discriminator OR confirm the
correct value byte-1 row exists for var_simple.

Run:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_li_disc.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
import torch  # noqa: E402
from neural_vm.batched_pure_neural import Token, BatchedPureNeuralRunner  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic)
from src.compiler import compile_c  # noqa: E402


PROGRAMS = {
    "var990": "int main() { int x; x = 990; return x; }",   # 0x03DE, byte1=3
    "var7": "int main() { int x; x = 7; return x; }",        # byte1=0
}


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    li = int(torch.argmax(row[lo:lo + 16]).item())
    hi_ = int(torch.argmax(row[hi:hi + 16]).item())
    return hi_ * 16 + li, min(float(row[lo + li]), float(row[hi + hi_]))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "var990"
    bc, _ = compile_c(PROGRAMS[which])
    from neural_vm.run_vm import AutoregressiveVMRunner
    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    serial.spec_k = 0
    runner = BatchedPureNeuralRunner(serial)
    model = runner.model
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    cap = {}
    orig = runner._step_one

    def spy(state, t, ti, *a, **k):
        r = orig(state, t, ti, *a, **k)
        cap["ctx"] = list(state.context)
        cap["pl"] = state.prefix_len
        return r
    runner._step_one = spy
    res = runner.run_batch([list(bc)], data_list=[b""],
                           expected_steps_list=[None], max_steps=20)
    runner._step_one = orig
    ctx, pl = cap["ctx"], cap["pl"]
    print(f"=== {which} '{PROGRAMS[which]}' result={res} ctxlen={len(ctx)} ===")
    padded = torch.tensor([ctx], device=next(model.parameters()).device)
    x = td(model.forward(padded, stop_after_block=15)[0])
    cols = {nm: dp[nm] for nm in (
        "MEM_VAL_B1", "MEM_VAL_B2", "MEM_STORE", "MEM_STORE_AT_VAL",
        "MEM_ADDR_SRC") if nm in dp}
    clo, chi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    print("  candidate rows (MEM_VAL_B2 or MEM_STORE_AT_VAL > 0.3):")
    for i in range(len(ctx)):
        row = x[i]
        b2 = float(row[dp["MEM_VAL_B2"]])
        sav = float(row[dp.get("MEM_STORE_AT_VAL", 0)])
        if abs(b2) > 0.3 or abs(sav) > 0.3:
            cv, cc = nib(row, clo, chi)
            astr = " ".join(f"{k}={float(row[c]):.2f}" for k, c in cols.items()
                            if abs(float(row[c])) > 0.2)
            print(f"    pos{i:3d}: CLEAN={cv}(c{cc:.1f}) {astr}")


if __name__ == "__main__":
    main()
