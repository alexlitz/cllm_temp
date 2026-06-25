#!/usr/bin/env python3
"""Compare the LC (char-load) byte-0 reload vs the working LI (word-load)
byte-0 reload, block-by-block, in the 30-tok campaign frame.

Both programs store 42 at 0x200 then load it back; LI passes (42), LC fails
(byte-0 = 0). Trace the OUTPUT byte-0 (and OP_LI_RELAY / OP_LC_RELAY gate
dims, MEM_STORE/MEM_VAL) at the LOAD-step REG_AX byte-0 predictor row to
find the block where LC byte-0 diverges from LI.

Run (campaign):
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_sclc python tools/probe_lc_byte0.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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
    "li": [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42),
           Opcode.SI, (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT],
    "lc": [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42),
           Opcode.SC, (Opcode.IMM, 0x200), Opcode.LC, Opcode.EXIT],
}


def _make_bytecode(ops):
    out = []
    for op in ops:
        if isinstance(op, tuple):
            out.append(int(op[0]) | (int(op[1]) << 8))
        else:
            out.append(int(op))
    return out


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    lo_i = int(torch.argmax(row[lo:lo + 16]).item())
    hi_i = int(torch.argmax(row[hi:hi + 16]).item())
    return hi_i * 16 + lo_i, min(float(row[lo + lo_i]), float(row[hi + hi_i]))


def _build_runner():
    from neural_vm.run_vm import AutoregressiveVMRunner
    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    serial.spec_k = 0
    return BatchedPureNeuralRunner(serial)


def trace_one(which, runner, model, dp):
    ops = PROGRAMS[which]
    bc = _make_bytecode(ops)
    STEP = int(Token.STEP_TOKENS)
    captured = {}
    orig = runner._step_one

    def spy(state, next_tok, tok_i, *a, **k):
        r = orig(state, next_tok, tok_i, *a, **k)
        captured["ctx"] = list(state.context)
        captured["prefix_len"] = state.prefix_len
        return r
    runner._step_one = spy
    results = runner.run_batch([bc], data_list=[b""], expected_steps_list=[8],
                               max_steps=30)
    runner._step_one = orig
    print(f"\n========= {which.upper()} result={results} STEP={STEP} =========")
    ctx = captured["ctx"]
    pl = captured["prefix_len"]
    # locate REG_AX rows; LI/LC is step 5
    li_step = 5
    axpos = None
    for i, t in enumerate(ctx):
        if t == int(Token.REG_AX) and i + 4 < len(ctx):
            step = (i - pl) // STEP if i >= pl else -1
            val = sum((ctx[i + 1 + j] & 0xFF) << (8 * j) for j in range(4))
            if step == li_step:
                axpos = i
                print(f"  LOAD-step REG_AX pos{i} bytes={ctx[i+1:i+5]} val={val}=0x{val:x}")
    if axpos is None:
        print("  no LOAD-step REG_AX row found")
        return None
    padded = torch.tensor([ctx], device=next(model.parameters()).device)
    nblk = len(model.blocks)
    # byte-0 predictor row = axpos (REG_AX marker predicts byte0)
    off = axpos
    print(f"  --- predictor pos={off} (REG_AX marker; predicts byte0) ---")
    prev = None
    gate_dims = ["OP_LI_RELAY", "OP_LC_RELAY", "MEM_STORE", "MEM_VAL_B0"]
    for blk in range(nblk):
        x = td(model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        ov, oc = nib(row, dp["OUTPUT_LO"], dp["OUTPUT_HI"])
        parts = [f"OUT={ov}(c{oc:.1f})"]
        for nm in ("ALU_LO", "AX_FULL_LO"):
            hi = nm.replace("_LO", "_HI")
            if nm in dp and hi in dp:
                v, c = nib(row, dp[nm], dp[hi])
                parts.append(f"{nm[:-3]}={v}(c{c:.1f})")
        for gd in gate_dims:
            if gd in dp:
                parts.append(f"{gd}={float(row[dp[gd]]):.1f}")
        key = tuple(parts[:3])
        if key != prev:
            print(f"    blk{blk:2d}: " + " ".join(parts))
            prev = key
    return off


def main():
    runner = _build_runner()
    model = runner.model
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    for which in ("li", "lc"):
        trace_one(which, runner, model, dp)


if __name__ == "__main__":
    main()
