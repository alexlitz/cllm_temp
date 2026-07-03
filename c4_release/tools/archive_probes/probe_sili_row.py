#!/usr/bin/env python3
"""Dump the active residual dims on the si_li_16bit LI-reload byte-1 predictor
row at block 28 (L14 mem-gen, pre-slam) and the SI-store / value-IMM rows, to
find a clean row discriminator for the L25-tail restore op.

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_sili_row.py [blk]
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

PROG = [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 0x1234),
        Opcode.SI, (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT]
# roundtrip (value 42, byte1=0): the PASSING counterpart, same frame shape.
PROG2 = [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42),
         Opcode.SI, (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT]


def mk(ops):
    return [(int(o[0]) | (int(o[1]) << 8)) if isinstance(o, tuple) else int(o) for o in ops]


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def run(prog, blk):
    from neural_vm.run_vm import AutoregressiveVMRunner
    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    serial.spec_k = 0
    runner = BatchedPureNeuralRunner(serial)
    model = runner.model
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    cap = {}
    orig = runner._step_one

    def spy(s, n, t, *a, **k):
        r = orig(s, n, t, *a, **k)
        cap["ctx"] = list(s.context)
        cap["pl"] = s.prefix_len
        return r
    runner._step_one = spy
    runner.run_batch([mk(prog)], data_list=[b""], expected_steps_list=[8], max_steps=30)
    runner._step_one = orig
    ctx = cap["ctx"]
    pl = cap["pl"]
    axpos = [i for i, t in enumerate(ctx) if t == int(Token.REG_AX) and (i - pl) // STEP == 5][0]
    predrow = axpos + 1
    padded = torch.tensor([ctx], device=next(model.parameters()).device)
    x = td(model.forward(padded, stop_after_block=blk)[0])[predrow]
    inv = {}
    for nm, pos in dp.items():
        inv.setdefault(int(pos), nm)
    print(f"  LI byte1 predrow={predrow}; active dims @ blk{blk}:")
    out = []
    for j in range(x.shape[0]):
        v = float(x[j])
        if abs(v) > 0.5:
            out.append((inv.get(j, f"dim{j}"), j, v))
    for nm, j, v in sorted(out, key=lambda t: -abs(t[2]))[:60]:
        print(f"    {nm}(@{j}) = {v:.2f}")
    return predrow


def main():
    blk = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    print("=== si_li_16bit (FAIL, value byte1=0x12) ===")
    run(PROG, blk)
    print("\n=== si_li_roundtrip (PASS, value byte1=0x00) ===")
    run(PROG2, blk)


if __name__ == "__main__":
    main()
