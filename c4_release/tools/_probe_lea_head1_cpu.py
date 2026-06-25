#!/usr/bin/env python3
"""LEAN CPU probe (no groundtruth deepcopy): build the real baked CPU model via
``build_cpu_model`` (the SAME weights cpu_full_trace uses), autoregressive-replay
func_add to get the emitted context, then reconstruct L7 head-1 attention at the
re-read LEA AX-marker row + the address byte that decodes for that LEA's AX.

Shows (a) WHICH BP marker row head 1 now attends on the re-read LEA (did the
OP_LEA x OP_ENT re-sharpen re-pin onto the ENT-frame value row?), and (b) the
ALU_LO band head 1 delivers + the BP-row OUTPUT_LO byte-0 (corruptor #2 check).

Usage: CUDA_VISIBLE_DEVICES="" C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_funcreread python tools/_probe_lea_head1_cpu.py [pid] [step]
"""
from __future__ import annotations
import os, sys
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
import torch.nn.functional as F  # noqa
from neural_vm.unified_compiler.faithful_autoregressive import build_cpu_model  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
STEP = int(Token.STEP_TOKENS)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


class _CtxStub:
    def __init__(self, model):
        self.model = model


def build_initial_context(model, bc):
    # ``_build_context`` is a pure tokeniser (reads the model embedding vocab
    # constants only). Call it unbound on a minimal stub holding ``.model`` so
    # no second model / deepcopy is built.
    from neural_vm.run_vm import AutoregressiveVMRunner
    return list(AutoregressiveVMRunner._build_context(
        _CtxStub(model), list(bc), b"", [], ""))


@torch.no_grad()
def replay(model, ctx, max_steps):
    dev = next(model.parameters()).device
    ctx = list(ctx)
    for _ in range(max_steps * STEP):
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        logits = model.forward(padded)[0]
        nxt = int(logits[len(ctx) - 1].argmax().item())
        ctx.append(nxt)
        # crude halt: stop if we have emitted enough steps
        if ctx.count(SE) >= max_steps:
            break
    return ctx


def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 575
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    blk = 11; head = 1
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    _dev = "cuda" if (torch.cuda.is_available()
                      and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(_dev); model.eval()
    dp = layout.dim_positions
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    pl_ctx = build_initial_context(model, bc)
    pl = len(pl_ctx)
    ctx = replay(model, pl_ctx, max(step + 4, 18))
    sm = smk(ctx, pl)
    if step >= len(sm) or "AX" not in sm[step]:
        print(f"id{pid} {desc}: step {step} not reached (got {len(sm)} steps)")
        return
    row = sm[step]["AX"]
    S = len(ctx)
    dev = next(model.parameters()).device
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    resid = model.forward(toks, stop_after_block=blk - 1)[0]
    x = resid.unsqueeze(0)
    attn = model.blocks[blk].attn
    H = attn.num_heads; HD = attn.W_q.shape[0] // H
    Q = F.linear(x, attn.W_q).view(1, S, H, HD).transpose(1, 2)
    K = F.linear(x, attn.W_k).view(1, S, H, HD).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
    slopes = getattr(attn, "alibi_slopes", None)
    pos = torch.arange(S, device=dev).float()
    if slopes is not None:
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
        scores = scores - slopes.view(1, H, 1, 1) * dist.view(1, 1, S, S)
    causal = torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)
    scores = scores + causal.view(1, 1, S, S)
    if getattr(attn, "use_softmax1", False):
        sink = torch.zeros(1, H, S, 1, device=dev)
        ext = torch.cat([scores, sink], dim=-1)
        w_all = F.softmax(ext, dim=-1)[..., :S]
    else:
        w_all = F.softmax(scores, dim=-1)
    w = w_all[0, head, row]; raw = scores[0, head, row]
    bp_base = dp["MARK_BP"]; out_lo = dp["OUTPUT_LO"]; ent = dp["OP_ENT"]
    print(f"id{pid} {desc} step{step} blk{blk} head{head} qrow={row} S={S} "
          f"slope={float(slopes[head]):.2f} sum_w={float(w.sum()):.4f}")
    top = torch.topk(w, k=min(10, S))
    for w_i, r in zip(top.values.tolist(), top.indices.tolist()):
        is_bp = abs(float(resid[r, bp_base])) > 0.5
        e = float(resid[r, ent])
        lo = [(round(float(resid[r, out_lo + k]), 1), k) for k in range(16)
              if abs(float(resid[r, out_lo + k])) > 1]
        print(f"  w={w_i:.4f} raw={float(raw[r]):7.1f} row={r:3d} tok={ctx[r]:3d} "
              f"{'BP' if is_bp else '  '} OP_ENT={e:5.2f} OUT_LO={lo}")
    # delivered ALU_LO at the AX row (after this block)
    resid2 = model.forward(toks, stop_after_block=blk)[0]
    alu_lo = dp["ALU_LO"]
    alo = [(round(float(resid2[row, alu_lo + k]), 1), k) for k in range(16)
           if abs(float(resid2[row, alu_lo + k])) > 1]
    print(f"  -> ALU_LO delivered at AX row {row} after blk{blk}: {alo}")


if __name__ == "__main__":
    main()
