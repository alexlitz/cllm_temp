"""Probe: verify A3.7 ALiBi pin landed at the right model.blocks index
and inspect actual attention weights at the broadcast head for S0@114.

Hypotheses:
1. ALiBi slope=1.0 isn't steep enough at distance ~30 -> attention should
   still concentrate on the recent K row (ratio e^35 >> 1).
2. The pin is on the wrong block. Try blocks 10, 11, 12, 13.
3. softmax1 interacts with content-addressed slot 33 such that the
   uniform K candidates BOTH lose to the +1 baseline.
4. K-side slot-33 content score (~2M = 10000 nats) is so large that
   exp(10000) overflows, mass goes to numerics, ALiBi differential gets
   lost.
"""

import contextlib
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(REPO_ROOT))

import torch

from c4_release.neural_vm.run_vm import AutoregressiveVMRunner
from c4_release.neural_vm.embedding import Opcode

PROG = [
    (Opcode.IMM, 0x200),
    Opcode.PSH,
    (Opcode.IMM, 42),
    Opcode.SI,
    (Opcode.IMM, 0x200),
    Opcode.LI,
    Opcode.EXIT,
]


def make_bc(prog):
    out = []
    for item in prog:
        if isinstance(item, tuple):
            opcode, imm = item
            out.append(opcode | (imm << 8))
        else:
            out.append(item)
    return out


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    model = runner.model

    # Step 1: Dump alibi_slopes for blocks 10, 11, 12, 13.
    print("=" * 70)
    print("ALiBi slopes per candidate block:")
    print("=" * 70)
    for li in (10, 11, 12, 13):
        if li >= len(model.blocks):
            continue
        block = model.blocks[li]
        attn = getattr(block, "attn", None)
        if attn is None:
            print(f"  L{li}: no attn module")
            continue
        slopes = getattr(attn, "alibi_slopes", None)
        if slopes is None:
            print(f"  L{li}: alibi_slopes is None")
            continue
        num_heads = getattr(attn, "num_heads", "?")
        s = slopes.detach().cpu().tolist()
        print(f"  L{li} (num_heads={num_heads}): slopes={[round(x, 5) for x in s]}")

    # Step 2: Capture attention probabilities at the broadcast heads.
    # We need to hook the attention module itself so we can recompute weights
    # after Q/K from the captured residual.

    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []

    dp = model.embed._dim_positions

    captures = []

    def embed_hook(module, inputs, output):
        captures.append({"token_ids": inputs[0].detach().clone()})

    def block_pre_hook(name):
        # Pre-block: residual going INTO block (what attn.W_q sees)
        def fn(module, inputs):
            if captures and isinstance(inputs, tuple) and len(inputs) > 0:
                captures[-1][f"pre_{name}"] = inputs[0].detach().clone()
        return fn

    def block_hook(name):
        def fn(module, inputs, output):
            if captures:
                captures[-1][name] = output.detach().clone()
        return fn

    handles = [model.embed.register_forward_hook(embed_hook)]
    for li in range(min(32, len(model.blocks))):
        handles.append(
            model.blocks[li].register_forward_hook(block_hook(f"after_L{li}"))
        )
        handles.append(
            model.blocks[li].register_forward_pre_hook(block_pre_hook(f"L{li}"))
        )

    bc = make_bc(PROG)
    try:
        try:
            result = runner.run(bc, b"", max_steps=30)
        except Exception as e:
            print(f"runner raised: {e}")
            result = None
    finally:
        for h in handles:
            h.remove()

    print(f"\nResult: {result}")

    best = None
    for c in captures:
        if "after_L12" not in c:
            continue
        if best is None or c["after_L12"].shape[1] > best["after_L12"].shape[1]:
            best = c

    if best is None or "pre_L12" not in best:
        print("No usable capture")
        return

    pre = best["pre_L12"]  # input to block 12 (=L10 broadcast block)
    seq = pre.shape[1]
    print(f"\nFull-context capture: seq_len={seq}")

    mark_stack0 = dp["MARK_STACK0"]
    mark_ax = dp["MARK_AX"]
    bi1 = dp["BYTE_INDEX_1"]

    stack0_positions = torch.nonzero(
        pre[0, :, mark_stack0] > 0.5, as_tuple=False
    ).flatten().tolist()
    ax_positions = torch.nonzero(
        pre[0, :, mark_ax] > 0.5, as_tuple=False
    ).flatten().tolist()

    print(f"MARK_STACK0 positions: {stack0_positions}")
    print(f"MARK_AX positions:     {ax_positions}")

    # Find S0@114 BI_1 row and S0@80 BI_1 row
    if 80 not in stack0_positions or 114 not in stack0_positions:
        print("WARNING: expected STACK0 at p=80 and p=114; got:", stack0_positions)

    # Compute Q/K/V for broadcast head 8 (byte 1) at L12 attn.
    attn = model.blocks[12].attn
    HD = attn.W_q.shape[0] // attn.num_heads
    head8 = 8

    # Project Q, K
    # W_q shape: [d_model -> d_inner], rows are heads*HD
    Wq = attn.W_q.detach()  # [d_inner, d_model] for typical LinearLike
    Wk = attn.W_k.detach()
    print(f"\nW_q shape: {tuple(Wq.shape)}, W_k shape: {tuple(Wk.shape)}")
    print(f"HD={HD}, num_heads={attn.num_heads}")

    if Wq.shape[0] != attn.num_heads * HD:
        # might be [d_model, d_inner]
        Wq = Wq.t()
        Wk = Wk.t()

    print(f"After transpose check: W_q shape: {tuple(Wq.shape)}")

    # Compute Q and K
    x = pre[0]  # [seq, d_model]
    # Compute Q = x @ Wq.T, then reshape into heads
    Q_all = x @ Wq.t() if Wq.shape[1] == x.shape[-1] else x @ Wq
    K_all = x @ Wk.t() if Wk.shape[1] == x.shape[-1] else x @ Wk

    if Q_all.shape[-1] != attn.num_heads * HD:
        print(f"  ERROR: unexpected Q shape {Q_all.shape}")
        return

    Q = Q_all.view(seq, attn.num_heads, HD)
    K = K_all.view(seq, attn.num_heads, HD)

    Q8 = Q[:, head8, :]  # [seq, HD]
    K8 = K[:, head8, :]

    # Pick S0@114 BI_1 row Q (p=116)
    q_row = 116
    print(f"\n==> Inspecting Q@p={q_row} (S0@114 BI_1, broken frame)")
    print(f"    K candidates with MARK_AX + BYTE_INDEX_1 (causal):")

    cand_rows = []
    for ax_p in ax_positions:
        for d in range(0, 5):
            kp = ax_p + d
            if kp >= seq or kp > q_row:
                continue
            if pre[0, kp, bi1] > 0.5 and pre[0, kp, mark_ax] > 0.5:
                cand_rows.append(kp)

    # Also check: did A3.6 record show K@67 and K@102?
    print(f"    Causal AX+BI_1 K rows: {cand_rows}")

    if not cand_rows:
        # try direct enumeration
        for p in range(q_row):
            if pre[0, p, mark_ax] > 0.5 and pre[0, p, bi1] > 0.5:
                cand_rows.append(p)
        print(f"    (full scan) AX+BI_1 K rows: {cand_rows}")

    q_vec = Q8[q_row]
    # Slope for head 8
    slope = float(attn.alibi_slopes.detach()[head8].item())
    print(f"    head 8 ALiBi slope = {slope}")

    # Compute Q*K dot product for each candidate
    scores = []
    for kp in cand_rows:
        k_vec = K8[kp]
        dot = float(torch.dot(q_vec, k_vec).item())
        # ALiBi bias: typically -slope * (q_pos - k_pos)
        dist = q_row - kp
        alibi_bias = -slope * dist
        total = dot + alibi_bias
        scores.append((kp, dot, alibi_bias, total))
        print(f"    K@p={kp}: dot={dot:.3f}, alibi_bias={alibi_bias:.3f}, "
              f"total={total:.3f}, dist={dist}")

    if scores:
        # softmax over candidates + softmax1 baseline (1.0)
        import math
        m = max(s[3] for s in scores) if scores else 0
        m = max(m, 0.0)  # for softmax1 baseline
        exp_vals = [math.exp(s[3] - m) for s in scores]
        base = math.exp(0.0 - m)  # softmax1 baseline
        denom = sum(exp_vals) + base
        print(f"\n    softmax1 weights (incl. +1 baseline):")
        for (kp, _, _, _), e in zip(scores, exp_vals):
            print(f"      K@p={kp}: weight = {e/denom:.6f}")
        print(f"      +1 baseline:  weight = {base/denom:.6f}")


if __name__ == "__main__":
    main()
