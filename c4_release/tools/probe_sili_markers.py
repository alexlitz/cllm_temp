#!/usr/bin/env python3
"""Dump a wide marker panel for the competing AX byte-1 register rows of the LI
step in si_li_16bit, so we can find a CLEAN structural discriminator that
separates the value-IMM (step before SI store) from every address-IMM row.

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_sili_markers.py [prog]
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
}
L10_BLOCK = 16


def _mk(ops):
    return [(int(o[0]) | (int(o[1]) << 8)) if isinstance(o, tuple) else int(o)
            for o in ops]


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


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
    STEP = int(Token.STEP_TOKENS)
    cap = {}
    orig = runner._step_one

    def spy(state, t, ti, *a, **k):
        r = orig(state, t, ti, *a, **k)
        cap["ctx"] = list(state.context)
        cap["pl"] = state.prefix_len
        return r
    runner._step_one = spy
    runner.run_batch([bc], data_list=[b""], expected_steps_list=[8], max_steps=30)
    runner._step_one = orig
    ctx, pl = cap["ctx"], cap["pl"]
    axrows = {}
    for i, t in enumerate(ctx):
        if t == int(Token.REG_AX) and i + 4 < len(ctx):
            step = (i - pl) // STEP if i >= pl else -1
            axrows[step] = i
    rows = {"step0-addr(0x200)": axrows[0] + 2,
            "step2-value(0x1234)": axrows[2] + 2,
            "step4-addr(0x200)": axrows[4] + 2}
    padded = torch.tensor([ctx], device=next(model.parameters()).device)
    x = td(model.forward(padded, stop_after_block=L10_BLOCK - 1)[0])
    panel = ["OP_IMM", "OP_SI", "OP_SC", "OP_LI", "OP_PSH", "MEM_STORE",
             "MEM_STORE_AT_VAL", "MEM_ADDR_SRC", "PSH_AT_SP", "OP_LI_RELAY",
             "MARK_AX", "IS_BYTE", "BYTE_INDEX_1"]
    # also scan ALL dims that DIFFER between value row and the two addr rows
    vrow = x[rows["step2-value(0x1234)"]]
    a0 = x[rows["step0-addr(0x200)"]]
    a4 = x[rows["step4-addr(0x200)"]]
    print(f"=== {which} ===")
    for nm in ("ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO", "ADDR_B1_HI"):
        if nm in dp:
            c = dp[nm]
            sv = float(vrow[c:c+16].sum())
            s0 = float(a0[c:c+16].sum())
            s4 = float(a4[c:c+16].sum())
            print(f"  {nm:14s} SUM: value={sv:.2f}  addr0={s0:.2f}  addr4={s4:.2f}")
    for nm in panel:
        if nm in dp:
            c = dp[nm]
            print(f"  {nm:18s}: value={float(vrow[c]):+.2f}  "
                  f"addr0={float(a0[c]):+.2f}  addr4={float(a4[c]):+.2f}")
    print("\n  Auto-scan dims where BOTH addr rows HIGH but VALUE LOW "
          "(penalize-address candidates):")
    inv = {v: k for k, v in dp.items()}
    found = 0
    for d in range(x.shape[1]):
        vv = float(vrow[d]); av0 = float(a0[d]); av4 = float(a4[d])
        # both addr rows clearly exceed value by >0.4
        if (av0 - vv) > 0.4 and (av4 - vv) > 0.4:
            cands = [(k, v2) for k, v2 in dp.items() if v2 <= d < v2 + 16]
            cands.sort(key=lambda z: -z[1])
            nm = inv.get(d) or (f"{cands[0][0]}+{d-cands[0][1]}" if cands else f"dim{d}")
            print(f"    {nm}(d{d}): value={vv:+.2f} addr0={av0:+.2f} addr4={av4:+.2f}")
            found += 1
            if found > 40:
                break


if __name__ == "__main__":
    main()
