#!/usr/bin/env python3
"""Per-head attribution of the block-32 (L18) OUTPUT_HI slam on the
si_li_16bit LI-reload byte-1 predictor row.

Forwards the REAL emitted si_li_16bit context up to block 31, then runs the
block-32 attention head-by-head, reporting each head's contribution to
OUTPUT_HI at the LI-reload byte-1 predictor row (pos 216). Identifies which
layer14_mem_generation head (0-7) materializes the slam and from WHICH source
(STACK0_BYTE_VAL_1 vs CLEAN_EMBED/OUTPUT) by reading its V/O path.

Run (campaign):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_sili_slam32.py
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
BLOCK = 32


def _mk(ops):
    out = []
    for op in ops:
        out.append((int(op[0]) | (int(op[1]) << 8)) if isinstance(op, tuple)
                   else int(op))
    return out


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    lo_i = int(torch.argmax(row[lo:lo + 16]).item())
    hi_i = int(torch.argmax(row[hi:hi + 16]).item())
    return hi_i * 16 + lo_i


def main():
    bc = _mk(PROG)
    from neural_vm.run_vm import AutoregressiveVMRunner
    serial = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True)
    serial.spec_k = 0
    runner = BatchedPureNeuralRunner(serial)
    model = runner.model
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)

    captured = {}
    orig = runner._step_one

    def spy(state, next_tok, tok_i, *a, **k):
        r = orig(state, next_tok, tok_i, *a, **k)
        captured["ctx"] = list(state.context)
        captured["prefix_len"] = state.prefix_len
        return r
    runner._step_one = spy
    runner.run_batch([bc], data_list=[b""], expected_steps_list=[8], max_steps=30)
    runner._step_one = orig

    ctx = captured["ctx"]
    pl = captured["prefix_len"]
    # LI-step REG_AX
    axpos = None
    for i, t in enumerate(ctx):
        if t == int(Token.REG_AX):
            step = (i - pl) // STEP if i >= pl else -1
            if step == 5:
                axpos = i
                break
    predrow = axpos + 1  # byte-0 row predicts byte-1
    print(f"axpos={axpos} predrow={predrow}")

    dev = next(model.parameters()).device
    padded = torch.tensor([ctx], device=dev)

    # Forward up to block 31 (input to block 32).
    x_in = model.forward(padded, stop_after_block=BLOCK - 1)[0]  # [S, d]
    blk = model.blocks[BLOCK]
    attn = blk.attn
    H = attn.num_heads
    HD = attn.head_dim
    d = attn.dim
    S = x_in.shape[0]

    # Pre-LN input (match the block's own normalization path).
    # Replicate AutoregressiveAttention.forward but capture per-head O.
    xn = x_in
    # Most blocks have a pre-attn norm; detect.
    norm = getattr(blk, "norm1", None) or getattr(blk, "ln1", None) or getattr(blk, "attn_norm", None)
    if norm is not None:
        xn = norm(x_in)
    Wq = td(attn.W_q); Wk = td(attn.W_k); Wv = td(attn.W_v); Wo = td(attn.W_o)
    xnc = xn.detach().cpu().float()
    Q = xnc @ Wq.T  # [S,d]
    K = xnc @ Wk.T
    V = xnc @ Wv.T
    Q = Q.view(S, H, HD).transpose(0, 1)  # [H,S,HD]
    K = K.view(S, H, HD).transpose(0, 1)
    V = V.view(S, H, HD).transpose(0, 1)
    scale = HD ** -0.5
    scores = torch.einsum("hqd,hkd->hqk", Q, K) * scale  # [H,S,S]
    # ALiBi
    slopes = attn.alibi_slopes
    if slopes is not None:
        slopes = slopes.detach().cpu().float()
        pos = torch.arange(S)
        rel = (pos[None, :] - pos[:, None]).float()  # k - q ; we want -dist for k<=q
        bias = -(pos[:, None] - pos[None, :]).clamp(min=0).float()  # [S,S] q-k
        scores = scores + slopes[:, None, None] * bias[None, :, :]
    # causal mask
    causal = torch.triu(torch.ones(S, S), diagonal=1).bool()
    scores = scores.masked_fill(causal[None], float("-inf"))
    # softmax1
    if getattr(attn, "use_softmax1", True):
        anchor = torch.zeros(H, S, 1)
        ext = torch.cat([scores, anchor], dim=-1)
        w = torch.softmax(ext, dim=-1)[..., :-1]
    else:
        w = torch.softmax(scores, dim=-1)
    ctxv = torch.einsum("hqk,hkd->hqd", w, V)  # [H,S,HD]
    # Per-head O contribution to OUTPUT_HI nibble cells at predrow.
    hi0 = dp["OUTPUT_HI"]
    lo0 = dp["OUTPUT_LO"]
    print(f"\nblock {BLOCK} H={H} HD={HD} OUTPUT_HI base={hi0}")
    # baseline OUTPUT at predrow before block 32
    base_hi = x_in.detach().cpu().float()[predrow, hi0:hi0 + 16]
    base_lo = x_in.detach().cpu().float()[predrow, lo0:lo0 + 16]
    print(f"  pre-blk32 OUTPUT byte1 nibble argmax: lo={int(base_lo.argmax())} hi={int(base_hi.argmax())} "
          f"-> 0x{int(base_hi.argmax())*16+int(base_lo.argmax()):02x}")
    print(f"  pre-blk32 OUTPUT_HI cells: " + " ".join(f"{i}:{float(base_hi[i]):.1f}" for i in range(16) if abs(float(base_hi[i]))>0.3))

    # Each head's O write to OUTPUT_HI cells at predrow.
    for h in range(H):
        head_out = ctxv[h, predrow]  # [HD]
        # project through W_o: this head occupies columns h*HD:(h+1)*HD of the
        # concatenated context that W_o maps. delta = head_out @ Wo[:, h*HD:(h+1)*HD].T
        Wo_h = Wo[:, h * HD:(h + 1) * HD]  # [d, HD]
        delta = Wo_h @ head_out  # [d]
        dhi = delta[hi0:hi0 + 16]
        dlo = delta[lo0:lo0 + 16]
        amax = float(dhi.abs().max())
        if amax > 0.5:
            cells = " ".join(f"{i}:{float(dhi[i]):+.1f}" for i in range(16) if abs(float(dhi[i])) > 0.5)
            print(f"  head {h}: OUTPUT_HI delta cells: {cells}")
            # which row did this head attend?
            top = int(w[h, predrow].argmax())
            print(f"          attends pos{top} weight={float(w[h,predrow,top]):.3f}")


if __name__ == "__main__":
    main()
