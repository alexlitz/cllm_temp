#!/usr/bin/env python3
"""Compute the REAL L10 head-1 attention scores (pre-softmax, with ALiBi) from
the byte-1 PREDICTOR row to EACH candidate AX-byte-1 register row, plus the
per-candidate contribution of ADDR_B1_HI/LO cells. Confirms (1) the leak
margin slot-82 must overcome and (2) that the addr-IMM candidate is the only
one carrying a strong ADDR_B1 the discriminator can key on.

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_sili_scores.py [prog]
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

# L10 head-1 byte passthrough runs at physical block 16.
L10_BLOCK = 16
HEAD_IDX = 1


def _mk(ops):
    out = []
    for op in ops:
        out.append((int(op[0]) | (int(op[1]) << 8)) if isinstance(op, tuple)
                   else int(op))
    return out


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
    res = runner.run_batch([bc], data_list=[b""], expected_steps_list=[8],
                           max_steps=30)
    runner._step_one = orig
    ctx, pl = cap["ctx"], cap["pl"]
    print(f"=== {which} result={res} ===")
    padded = torch.tensor([ctx], device=next(model.parameters()).device)

    axrows = {}
    for i, t in enumerate(ctx):
        if t == int(Token.REG_AX) and i + 4 < len(ctx):
            step = (i - pl) // STEP if i >= pl else -1
            axrows[step] = i
    pred = axrows[5] + 1
    cand = {"value-IMM": axrows[2] + 2, "addr-IMM": axrows[4] + 2}

    # Optionally truncate to the causal decode prefix (tokens <= predictor) so
    # the residual matches what autoregressive decode actually sees.
    if os.environ.get("TRUNC", "0") == "1":
        ctx = ctx[:pred + 1]
        padded = torch.tensor([ctx], device=next(model.parameters()).device)
        print(f"  [TRUNCATED to prefix len {len(ctx)}]")

    # residual at L10 block INPUT (post block 15)
    x = td(model.forward(padded, stop_after_block=L10_BLOCK - 1)[0])

    blk = model.blocks[L10_BLOCK]
    attn = blk.attn
    HD = attn.W_q.shape[0] // attn.num_heads
    base = HEAD_IDX * HD
    Wq = td(attn.W_q)[base:base + HD]      # (HD, d)
    Wk = td(attn.W_k)[base:base + HD]
    alibi = None
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        alibi = float(td(attn.alibi_slopes)[HEAD_IDX])
    print(f"  L10 blk{L10_BLOCK} head{HEAD_IDX} HD={HD} alibi_slope={alibi}")

    qrow = x[pred]
    qproj = Wq @ qrow  # (HD,)
    print(f"  predictor pos{pred}; ALiBi slope={alibi}")
    # Full softmax: rank ALL rows the head attends to (find what actually wins).
    if os.environ.get("RANKALL", "0") == "1":
        scores = []
        for p in range(pred + 1):
            kproj = Wk @ x[p]
            raw = float((qproj * kproj).sum())
            bias = (alibi or 0.0) * -(pred - p)
            scores.append((raw + bias, p))
        scores.sort(reverse=True)
        clo, chi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
        print("  TOP-8 attended rows:")
        for sc, p in scores[:8]:
            row = x[p]
            cv = int(torch.argmax(row[chi:chi+16]))*16 + int(torch.argmax(row[clo:clo+16]))
            ab_hi = float(row[dp["ADDR_B1_HI"]:dp["ADDR_B1_HI"]+16].sum()) if "ADDR_B1_HI" in dp else 0
            ab_lo = float(row[dp["ADDR_B1_LO"]:dp["ADDR_B1_LO"]+16].sum()) if "ADDR_B1_LO" in dp else 0
            opi = float(row[dp["OP_IMM"]]) if "OP_IMM" in dp else 0
            print(f"    pos{p}: score={sc:.0f} CLEAN=0x{cv:02x} tok{ctx[p]} "
                  f"OP_IMM={opi:.1f} ADDR_B1(HI={ab_hi:.1f},LO={ab_lo:.1f})")
    for tag, p in cand.items():
        krow = x[p]
        kproj = Wk @ krow
        raw = float((qproj * kproj).sum())
        bias = (alibi or 0.0) * -(pred - p)  # ALiBi: -slope*dist
        # per-slot contributions for slots 80..90
        contrib = []
        for s in range(min(HD, 92)):
            c = float(qproj[s] * kproj[s])
            if abs(c) > 0.5:
                contrib.append(f"s{s}={c:.1f}")
        print(f"  {tag} pos{p}: raw_score={raw:.1f} alibi_bias={bias:.1f} "
              f"total={raw+bias:.1f}")
        print(f"      slot-contrib: " + " ".join(contrib))

    # also: what does slot 82's Q look like, and what cells of ADDR_B1 does it
    # need to add to discriminate?
    s82q = Wq[82] if HD > 82 else None
    if s82q is not None:
        nz = [(i, float(s82q[i])) for i in range(len(s82q)) if abs(float(s82q[i])) > 0.5]
        print(f"  slot82 Q nonzero dims: {nz[:20]}")
    for tag, p in cand.items():
        krow = x[p]
        for nm in ("ADDR_B1_HI", "ADDR_B1_LO", "ADDR_B0_HI"):
            if nm in dp:
                seg = krow[dp[nm]:dp[nm] + 16]
                am = int(torch.argmax(seg).item())
                print(f"    {tag} {nm}: argmax cell {am} = {float(seg[am]):.2f} "
                      f"(sum={float(seg.sum()):.2f})")


if __name__ == "__main__":
    main()
