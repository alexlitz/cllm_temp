#!/usr/bin/env python3
"""Trace the L10 head-1 byte-1 PREDICTOR row (AX byte-0 row of the LI step) and
the two competing AX-byte-1 REGISTER candidate rows (value-IMM step vs
address-IMM step) for the si_li programs. Dump, across a sweep of blocks, the
ADDR_B1 (load-address byte-1 one-hot) staged on the predictor row + the CLEAN
byte-1 nibbles on each candidate -- to confirm a Q-side ADDR_B1-vs-CLEAN
discriminator exists at the L10 block input (block 15/16).

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_sili_addrdisc.py [prog]
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
    "si_li_roundtrip": [(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42),
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


def onehot(row, base, n=16):
    seg = row[base:base + n]
    i = int(torch.argmax(seg).item())
    return i, float(seg[i])


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
    res = runner.run_batch([bc], data_list=[b""], expected_steps_list=[8],
                           max_steps=30)
    runner._step_one = orig
    ctx, pl = cap["ctx"], cap["pl"]
    print(f"=== {which} result={res} STEP={STEP} pl={pl} ctxlen={len(ctx)} ===")
    padded = torch.tensor([ctx], device=next(model.parameters()).device)

    axrows = {}
    for i, t in enumerate(ctx):
        if t == int(Token.REG_AX) and i + 4 < len(ctx):
            step = (i - pl) // STEP if i >= pl else -1
            axrows[step] = i
    li_ax = axrows.get(5)
    val_ax = axrows.get(2)
    addr_ax = axrows.get(4)
    if None in (li_ax, val_ax, addr_ax):
        print("missing AX rows", axrows); return
    pred = li_ax + 1
    val_b1 = val_ax + 2
    addr_b1 = addr_ax + 2
    print(f"  predictor(LI b0)=pos{pred}  value-IMM b1=pos{val_b1}  "
          f"addr-IMM b1=pos{addr_b1}")

    addr_dims = [("ADDR_B0_LO", 12), ("ADDR_B1_LO", 28),
                 ("ADDR_B0_HI", 206), ("ADDR_B1_HI", 222)]
    clo, chi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    for blk in (11, 13, 15, 16):
        x = td(model.forward(padded, stop_after_block=blk)[0])
        print(f"\n  --- block {blk} ---")
        prow = x[pred]
        ab = []
        for nm, _ in addr_dims:
            if nm in dp:
                i, c = onehot(prow, dp[nm])
                ab.append(f"{nm}={i}(c{c:.1f})")
        print(f"    predictor pos{pred}: " + " ".join(ab))
        for tag, p in (("value-IMM b1", val_b1), ("addr-IMM  b1", addr_b1)):
            row = x[p]
            lo, _ = onehot(row, clo)
            hi, _ = onehot(row, chi)
            cv = hi * 16 + lo
            extra = []
            for nm in ("OP_IMM", "MEM_ADDR_SRC", "MEM_STORE",
                       "ADDR_B1_LO", "ADDR_B1_HI"):
                if nm in dp:
                    extra.append(f"{nm}={float(row[dp[nm]]):.2f}")
            print(f"    {tag} pos{p}: CLEAN_b1={cv}=0x{cv:02x}(lo{lo},hi{hi}) "
                  + " ".join(extra))


if __name__ == "__main__":
    main()
