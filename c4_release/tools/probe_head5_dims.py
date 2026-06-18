#!/usr/bin/env python3
"""Per-dim breakdown of head-5's Q*K logit contribution at the winning K row(s).

Shows, for the AX-marker query, the top-N attended K rows and WHICH head-5
slice dims drive the logit (so we can see whether pos 44 wins via dims 1/2/4
[the rebuilt value-row/store gate] or via leftover content). Prints the
CLEAN_EMBED byte-0 value carried at each row and the residual values on the
gating dims.

Usage:
    CUDA_VISIBLE_DEVICES=0 C4_OPERAND_FROM_MEMSP=1 C4_NO_STACK0_EMIT=1 \
        python tools/probe_head5_dims.py 9 3
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import torch  # noqa: E402

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return list(bc), exp, desc


def _nib_argmax(vec, base):
    sl = vec[base:base + 16]
    if float(sl.max()) < 0.5:
        return -1
    return int(sl.argmax())


@torch.no_grad()
def main(idx, target_step, max_steps, block, head):
    ST = Token.STEP_TOKENS
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)
    model = runner.model
    dev = next(model.parameters()).device
    dp = model.dim_positions
    CLO = dp["CLEAN_EMBED_LO"]

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)
    for _ in range(ST * max_steps):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        nxt = int(model.forward(padded)[0][-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    emitted = context[len(prefix):]
    step_slice = emitted[target_step * ST:(target_step + 1) * ST]
    ax_off = next((i for i, t in enumerate(step_slice) if t == Token.REG_AX), 1)
    qpos = len(prefix) + target_step * ST + ax_off

    padded = torch.tensor([context], dtype=torch.long, device=dev)
    if block == 0:
        x = model.embed(padded)[0]
    else:
        x = model.forward(padded, stop_after_block=block - 1)[0]

    blk = model.blocks[block]
    attn = blk.attn
    HD = attn.W_q.shape[0] // attn.num_heads
    base = head * HD
    slope = float(attn.alibi_slopes[head]) if attn.alibi_slopes is not None else 0.0
    scale = HD ** -0.5

    q = (x[qpos] @ attn.W_q.t())[base:base + HD]       # [HD]
    K = (x @ attn.W_k.t())[:, base:base + HD]           # [S, HD]
    # per-dim contribution = q[d] * K[:,d] * scale
    contrib = K * q.unsqueeze(0) * scale                # [S, HD]
    logits = contrib.sum(dim=1)                         # [S]
    S = x.shape[0]
    pos = torch.arange(S, device=dev)
    alibi = -slope * (qpos - pos).abs().float()
    causal = torch.where(pos <= qpos, 0.0, float("-inf"))
    scores = logits + alibi + causal
    ext = torch.cat([scores, torch.zeros(1, device=dev)])
    w = torch.softmax(ext, dim=0)[:-1]

    print(f"== id={idx} {desc[:30]} step{target_step} head{head} q@pos{qpos} "
          f"HD={HD} slope={slope} ST={ST} "
          f"NOSTK={os.environ.get('C4_NO_STACK0_EMIT','0')} ==")
    top = torch.topk(w, k=min(6, S))
    for r in range(top.indices.numel()):
        p = int(top.indices[r])
        ce0 = _nib_argmax(x[p], CLO)
        # top contributing dims at this row
        c = contrib[p]
        dtop = torch.topk(c.abs(), k=6)
        dims = ", ".join(
            f"d{int(i)}={float(c[int(i)]):+.0f}" for i in dtop.indices
        )
        print(f"  pos{p:4d} tok{context[p]:3d} w={float(top.values[r]):.4f} "
              f"logit={float(logits[p]):8.1f} alibi={float(alibi[p]):6.1f} "
              f"CE0={ce0:3d}")
        print(f"        topdims: {dims}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.lstrip('-').isdigit()]
    idx = args[0] if args else 9
    step = args[1] if len(args) > 1 else 3
    ms = args[2] if len(args) > 2 else 6
    blk = args[3] if len(args) > 3 else 11
    hd = args[4] if len(args) > 4 else 5
    main(idx, step, ms, blk, hd)
