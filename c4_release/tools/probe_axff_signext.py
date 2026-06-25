#!/usr/bin/env python3
"""var_update id325 step-14 AX byte1-3 sign-extension probe (campaign, spec_k=0).

After the #325 byte-0 fix the step-14 LEA AX byte-0 is 0xE8 (correct local frame
address) but bytes 1-3 emit 0x00 instead of 0xFF -> got_ax 232 (0x000000E8)
vs want 65512 (0xFFFFFFE8, the sign-extension of the negative stack address).

This probe builds the PRODUCTION AR context (the real spec_k=0 decode), locates
the step-14 REG_AX marker, and does exact LM-head logit attribution at the
byte-1/2/3 PREDICTOR rows (marker+1, marker+2, marker+3) to find what drives the
0x00 emission and which dims could discriminate the negative-address case. It
also dumps the byte-0/1/2/3 predictor-row residual bands so we can find a
sign-extension discriminator (a band that says "byte-0 was >= 0x80").

  CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_axff \
    python tools/probe_axff_signext.py [blocks...]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 50; x = x + 7; return x; }"
STEP_DEFAULT = 14


def fmt(row, base, dp, width=16, thr=0.5):
    b = dp.get(base)
    if b is None:
        return f"{base}=n/a"
    vals = [float(row[b + i].item()) for i in range(width)]
    s = "[" + ", ".join(f"{v:.0f}@{i}" for i, v in enumerate(vals) if abs(v) > thr) + "]"
    return f"{base}={s}"


def scalar(row, name, dp):
    b = dp.get(name)
    if b is None:
        return f"{name}=n/a"
    return f"{name}={float(row[b].item()):.2f}"


def name_for(pos, dp):
    best = None
    for nm, st in dp.items():
        if st <= pos and (best is None or st > best[1]):
            best = (nm, st)
    if best is None:
        return f"dim{pos}"
    return f"{best[0]}+{pos - best[1]}"


@torch.no_grad()
def attribute(model, padded, pos, want, got, dp, topn=22):
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = model.head.bias
    last_block = len(model.blocks) - 1
    x = model.forward(padded, stop_after_block=last_block)[0]
    res = x[pos]
    if res.is_sparse:
        res = res.to_dense()
    res = res.to(W.device).float().contiguous()
    Wd = (W[got] - W[want])
    Wd = Wd.to_dense() if Wd.is_sparse else Wd
    dw = (Wd * res).float().cpu()
    lg = float((W[got] * res).sum() + b[got])
    lw = float((W[want] * res).sum() + b[want])
    order = torch.argsort(dw.abs(), descending=True)
    res_cpu = res.cpu()
    Wd_cpu = Wd.cpu()
    rows = []
    for di in order[:topn].tolist():
        rows.append((di, float(res_cpu[di]), float(Wd_cpu[di]), float(dw[di])))
    return lg, lw, rows


def main(blocks, step):
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    ax_marker = int(Token.REG_AX)

    bc, _ = compile_c(SRC)
    ctx = p._final_context(bc, max_steps=30)
    prefix_len = len(p._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    base = prefix_len + step * STEP
    seg = ctx[base:base + STEP]
    ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
    if ax_i is None:
        print(f"[!] no REG_AX marker in step {step}")
        return
    ax_marker_row = base + ax_i
    axv = 0
    bytes_emitted = []
    for j in range(4):
        bt = int(seg[ax_i + 1 + j]) & 0xFF
        bytes_emitted.append(bt)
        axv |= bt << (j * 8)
    print(f"=== id325 step{step} AX byte1-3 sign-ext probe  STEP={STEP} ===")
    print(f"   ax_marker_row={ax_marker_row}  emitted bytes={['0x%02x'%b for b in bytes_emitted]}  ax={axv}")
    print(f"   WANT bytes=[0xe8, 0xff, 0xff, 0xff] ax=65512\n")

    # Residual band dump at byte predictor rows.
    bands = ["OUTPUT_LO", "OUTPUT_HI", "OUTPUT_HI_THIS_STEP", "AX_CARRY_LO",
             "AX_CARRY_HI", "ALU_LO", "ALU_HI", "FETCH_LO", "FETCH_HI"]
    bands = [bd for bd in bands if dp.get(bd) is not None]
    flags = ["OP_LEA", "OP_SI", "OP_ADD", "MARK_AX", "HAS_SE", "IS_BYTE",
             "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"]
    flags = [f for f in flags if dp.get(f) is not None]

    for b in blocks:
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=b)[0]
        print(f" --- block {b} ---")
        for byte_idx in range(4):
            r = ax_marker_row + byte_idx  # predictor row for byte_idx
            row = resid[r]
            fl = " ".join(scalar(row, f, dp) for f in flags)
            print(f"   pred-byte{byte_idx} row{r}: {fl}")
            bd = "  ".join(fmt(row, bn, dp) for bn in bands)
            print(f"        {bd}")

    # Attribution at byte 1/2/3 predictor rows (want 0xff, got emitted).
    for byte_idx in (1, 2, 3):
        got = bytes_emitted[byte_idx]
        pred = ax_marker_row + byte_idx
        lg, lw, rows = attribute(model, padded, pred, want=0xFF, got=max(got, 1), dp=dp)
        print(f"\n##### byte-{byte_idx} predictor row {pred}: got=0x{got:02x} want=0xff #####")
        print(f"    logit[got=0x{max(got,1):02x}]={lg:.3f}  logit[0xff]={lw:.3f}  diff(got-ff)={lg-lw:.3f}")
        for di, rv, dW, c in rows:
            print(f"      dim {di:4d} {name_for(di, dp):28s} res={rv:9.3f} dW={dW:7.3f} contrib={c:9.3f}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:]]
    step = STEP_DEFAULT
    blocks = args or [16, 26, 36, 38]
    main(blocks, step)
