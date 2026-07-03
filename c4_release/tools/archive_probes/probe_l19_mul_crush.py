#!/usr/bin/env python3
"""Pinpoint the L19 (block 33) FFN units that explode OUTPUT on a small-product
MUL-AX row (the campaign L19-EXPLODE family: 3*15, 11*11, 1*10, 8*30).

Hooks block-33 PureFFN.forward to capture (a) the input residual's OUTPUT band,
(b) the silu(up)*gate hidden activations, (c) each hidden unit's contribution to
the OUTPUT_LO/HI delta — so we can name the exact rule that crushes the correct
moderate-magnitude product.

Usage:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 C4_VM_CACHE_DIR=/tmp/c4cache_mulres \
    CUDA_VISIBLE_DEVICES="" python tools/probe_l19_mul_crush.py 11:11 21:59
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from neural_vm.embedding import Opcode
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe

BLK = 33
CAP = {}


def prog(a, b):
    return [Opcode.IMM | (a << 8), Opcode.PSH, Opcode.IMM | (b << 8),
            Opcode.MUL, Opcode.EXIT]


def install(model, dp, target_row_holder):
    blk = model.blocks[BLK]
    ffn = blk.ffn
    orig = ffn.forward
    # Also hook the attention to see if IT is the amplifier.
    attn = getattr(blk, "attn", None) or getattr(blk, "attention", None)
    if attn is not None:
        aorig = attn.forward

        def apatched(x, *aa, _ao=aorig, **kw):
            aout = _ao(x, *aa, **kw)
            try:
                r = target_row_holder[0]
                ol = dp["OUTPUT_LO"]; oh = dp["OUTPUT_HI"]
                av = aout[0] if isinstance(aout, tuple) else aout
                if r is not None and av.dim() == 3:
                    CAP["attn_in_lo"] = float(x[0, r, ol:ol + 16].sum())
                    CAP["attn_in_hi"] = float(x[0, r, oh:oh + 16].sum())
                    CAP["attn_out_lo"] = float(av[0, r, ol:ol + 16].sum())
                    CAP["attn_out_hi"] = float(av[0, r, oh:oh + 16].sum())
            except Exception as e:  # noqa
                CAP["aerr"] = str(e)
            return aout

        attn.forward = apatched

    def patched(x, *a, _orig=orig, **kw):
        out = _orig(x, *a, **kw)
        try:
            r = target_row_holder[0]
            if r is not None and x.dim() == 3:
                xr = x[0, r]  # input residual at the MUL-AX row
                up = F.linear(xr, ffn.W_up, getattr(ffn, "b_up", None))
                gate = F.linear(xr, ffn.W_gate, getattr(ffn, "b_gate", None))
                hidden = F.silu(up) * gate
                ol = dp["OUTPUT_LO"]; oh = dp["OUTPUT_HI"]
                Wd = ffn.W_down
                # contribution of each hidden unit to OUTPUT_LO/HI
                contrib_lo = hidden[:, None] * Wd[ol:ol + 16, :].T  # [H,16]
                contrib_hi = hidden[:, None] * Wd[oh:oh + 16, :].T
                mag = contrib_lo.abs().sum(-1) + contrib_hi.abs().sum(-1)
                CAP["mag"] = mag.detach()
                CAP["hidden"] = hidden.detach()
                CAP["up"] = up.detach()
                CAP["gate"] = gate.detach()
                CAP["in_out_lo"] = float(xr[ol:ol + 16].sum())
                CAP["in_out_hi"] = float(xr[oh:oh + 16].sum())
                CAP["out_out_lo"] = float(out[0, r, ol:ol + 16].sum())
                CAP["out_out_hi"] = float(out[0, r, oh:oh + 16].sum())
        except Exception as e:  # noqa
            CAP["err"] = str(e)
        return out

    ffn.forward = patched


def main():
    pairs = []
    for t in sys.argv[1:]:
        if t.startswith("--"):
            continue
        a, b = t.split(":")
        pairs.append((int(a), int(b)))
    if not pairs:
        pairs = [(11, 11), (21, 59)]

    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    row_holder = [None]
    install(model, dp, row_holder)

    for (a, b) in pairs:
        exp = a * b
        ctx = probe._final_context(prog(a, b), max_steps=20)
        toks0 = torch.tensor([ctx], dtype=torch.long, device=dev)
        # Find the MUL step's AX row: the row where OP_MUL + MARK_AX are hot in
        # the residual at the MUL block input (block 15).
        with torch.no_grad():
            rin = model.forward(toks0, stop_after_block=15)[0]
        op_mul = dp.get("OP_MUL"); mk = dp.get("MARK_AX")
        mul_rows = [i for i in range(len(ctx))
                    if rin[i, op_mul] > 0.5 and rin[i, mk] > 0.5]
        ax_pos = mul_rows[-1] if mul_rows else max(
            i for i, t in enumerate(ctx) if t == int(Token.REG_AX))
        row_holder[0] = ax_pos
        CAP.clear()
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            model.forward(toks)
        print(f"\n=== A={a} B={b} exp={exp}=0x{exp:04x}  ax_row={ax_pos} ===")
        print(f"  block{BLK} ATTN  in_LO={CAP.get('attn_in_lo')} in_HI={CAP.get('attn_in_hi')} "
              f"-> out_LO={CAP.get('attn_out_lo')} out_HI={CAP.get('attn_out_hi')}")
        print(f"  block{BLK} FFN INPUT  OUTPUT_LO_sum={CAP.get('in_out_lo'):.2f} "
              f"OUTPUT_HI_sum={CAP.get('in_out_hi'):.2f}")
        print(f"  block{BLK} OUTPUT OUTPUT_LO_sum={CAP.get('out_out_lo'):.2f} "
              f"OUTPUT_HI_sum={CAP.get('out_out_hi'):.2f}")
        mag = CAP.get("mag")
        if mag is not None:
            top = mag.argsort(descending=True)[:10]
            for u in top.tolist():
                print(f"    unit {u:4d}: |OUTdelta|={float(mag[u]):8.2f} "
                      f"hidden={float(CAP['hidden'][u]):10.3f} "
                      f"up={float(CAP['up'][u]):8.2f} "
                      f"gate={float(CAP['gate'][u]):8.2f}")


if __name__ == "__main__":
    main()
