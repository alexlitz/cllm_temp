#!/usr/bin/env python3
"""For the si_li programs, dump every K-candidate row the L10 head-1 byte-1
selector could attend, with its MEM_VAL_B0/B1/B2 + MEM_STORE + MEM_STORE_AT_VAL
+ MEM_ADDR_SRC + CLEAN_EMBED, at the L10 block INPUT (post block 15).

This reveals which row carries the genuine VALUE byte-1 vs the ADDRESS byte-1,
so the slot-83 K can be re-keyed to the right one.

Run:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_sili_candrows.py [prog]
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
from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.batched_pure_neural import Token, BatchedPureNeuralRunner  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic)


PROGRAMS = {
    "si_li_16bit": [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 0x1234),
                    Opcode.SI, (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT],
    "si_li_zero": [(Opcode.IMM, 0x300), Opcode.PSH, (Opcode.IMM, 0),
                   Opcode.SI, (Opcode.IMM, 0x300), Opcode.LI, Opcode.EXIT],
}


def _mk(ops):
    out = []
    for op in ops:
        out.append((int(op[0]) | (int(op[1]) << 8)) if isinstance(op, tuple)
                   else int(op))
    return out


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    li = int(torch.argmax(row[lo:lo + 16]).item())
    hi_ = int(torch.argmax(row[hi:hi + 16]).item())
    return hi_ * 16 + li, min(float(row[lo + li]), float(row[hi + hi_]))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "si_li_16bit"
    bc = _mk(PROGRAMS[which])
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
    res = runner.run_batch([bc], data_list=[b""], expected_steps_list=[8],
                           max_steps=30)
    runner._step_one = orig
    ctx, pl = cap["ctx"], cap["pl"]
    print(f"=== {which} result={res} ctxlen={len(ctx)} ===")
    padded = torch.tensor([ctx], device=next(model.parameters()).device)
    # L10 head-1 runs at block 16; read the residual at block 15 (its input).
    x = td(model.forward(padded, stop_after_block=15)[0])
    cols = {nm: dp[nm] for nm in (
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "MEM_STORE", "MEM_STORE_AT_VAL", "MEM_ADDR_SRC") if nm in dp}
    clo, chi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    print("  rows with ANY MEM_VAL_B* or MEM_STORE* > 0.3 (block-15 input):")
    for i in range(len(ctx)):
        row = x[i]
        active = {nm: float(row[c]) for nm, c in cols.items()
                  if abs(float(row[c])) > 0.3}
        if active:
            cv, cc = nib(row, clo, chi)
            tag = ""
            astr = " ".join(f"{k}={v:.2f}" for k, v in active.items())
            print(f"    pos{i:3d} tok{ctx[i]:3d}: CLEAN={cv}(c{cc:.1f}) {astr}{tag}")


if __name__ == "__main__":
    main()
