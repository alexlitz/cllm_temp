#!/usr/bin/env python3
"""For a var_* program (LEA-addressed local), find the L10 head-1 byte-1
predictor row at the return step and the candidate AX-byte-1 register row that
slot-82 selects, and dump the candidate's ADDR_B1 magnitude. Confirms the
value-IMM row in var_simple does NOT carry the strong gathered-load-address
ADDR_B1 signature the si/li addr-IMM row does (so a discriminator keyed on it
won't perturb var_simple).

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_var_scores.py [prog]
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
    "var990": "int main() { int x; x = 990; return x; }",   # 0x03DE
    "var258": "int main() { int x; x = 258; return x; }",   # 0x0102
    "var7": "int main() { int x; x = 7; return x; }",
}
L10_BLOCK = 16
HEAD_IDX = 1


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


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
    STEP = int(Token.STEP_TOKENS)
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
    print(f"=== {which} result={res} ===")
    padded = torch.tensor([ctx], device=next(model.parameters()).device)

    axrows = []
    for i, t in enumerate(ctx):
        if t == int(Token.REG_AX) and i + 4 < len(ctx):
            step = (i - pl) // STEP if i >= pl else -1
            val = sum((ctx[i + 1 + j] & 0xFF) << (8 * j) for j in range(4))
            axrows.append((i, step, val))
    for i, step, val in axrows:
        print(f"  REG_AX pos{i} step{step} val={val}=0x{val:x}")

    # The final REG_AX (return step) is the predictor's step; its byte-0 row is
    # the predictor. We scan all AX-byte-1 register rows as candidates and dump
    # ADDR_B1 magnitude + slot-82 score.
    x = td(model.forward(padded, stop_after_block=L10_BLOCK - 1)[0])
    blk = model.blocks[L10_BLOCK]
    attn = blk.attn
    HD = attn.W_q.shape[0] // attn.num_heads
    base = HEAD_IDX * HD
    Wq = td(attn.W_q)[base:base + HD]
    Wk = td(attn.W_k)[base:base + HD]
    alibi = float(td(attn.alibi_slopes)[HEAD_IDX]) if (
        hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None) else 0.0

    ret_ax = axrows[-1][0]
    pred = ret_ax + 1
    qrow = x[pred]
    qproj = Wq @ qrow
    print(f"  predictor pos{pred} (return-step AX byte0); ADDR_B1 on candidates:")
    for i, step, val in axrows:
        b1row_pos = i + 2
        if b1row_pos >= len(ctx):
            continue
        krow = x[b1row_pos]
        kproj = Wk @ krow
        raw = float((qproj * kproj).sum())
        bias = alibi * -(pred - b1row_pos)
        s82 = float(qproj[82] * kproj[82]) if HD > 82 else 0.0
        ab_hi = krow[dp["ADDR_B1_HI"]:dp["ADDR_B1_HI"] + 16] if "ADDR_B1_HI" in dp else None
        ab_lo = krow[dp["ADDR_B1_LO"]:dp["ADDR_B1_LO"] + 16] if "ADDR_B1_LO" in dp else None
        opimm = float(krow[dp["OP_IMM"]]) if "OP_IMM" in dp else 0.0
        clo = int(torch.argmax(krow[dp["CLEAN_EMBED_LO"]:dp["CLEAN_EMBED_LO"]+16]))
        chi = int(torch.argmax(krow[dp["CLEAN_EMBED_HI"]:dp["CLEAN_EMBED_HI"]+16]))
        cv = chi*16+clo
        print(f"    cand b1 pos{b1row_pos} step{step}: CLEAN_b1=0x{cv:02x} "
              f"OP_IMM={opimm:.2f} total={raw+bias:.0f} s82={s82:.0f} "
              f"ADDR_B1_HI_sum={float(ab_hi.sum()) if ab_hi is not None else 0:.2f} "
              f"ADDR_B1_LO_sum={float(ab_lo.sum()) if ab_lo is not None else 0:.2f}")


if __name__ == "__main__":
    main()
