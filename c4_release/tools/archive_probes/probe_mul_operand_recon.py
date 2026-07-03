#!/usr/bin/env python3
"""Probe what FlattenedALUMul reconstructs for operand A/B + product on the
MUL-AX row, in the campaign config, for a list of (a,b) pairs.

Instruments FlattenedALUMul.forward to capture the per-row reconstructed
operand_a / operand_b / product at the MUL marker row, so we can see WHY a
small-a mul lands a wrong byte (operand-A crush, byte-1 flood, etc.).

Usage:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 C4_VM_CACHE_DIR=/tmp/c4cache_mulres \
    CUDA_VISIBLE_DEVICES="" python tools/probe_mul_operand_recon.py 3:15 11:11 93:34
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def prog(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.MUL,
                Opcode.EXIT])


_CAP = {}


def install_capture(model):
    # Hook every BDToGEConverter: capture the x_ge IT RETURNS (the operand
    # reconstruction the schoolbook multiply consumes).
    convs = []
    for m in model.modules():
        if m.__class__.__name__ == "BDToGEConverter":
            convs.append(m)
    if not convs:
        print("[probe] WARNING: no BDToGEConverter module found", file=sys.stderr)
    for m in convs:
        orig = m.forward
        ge = m.ge
        BD = m.BD

        def patched(x_bd, *a, _orig=orig, _ge=ge, _BD=BD, **kw):
            x_ge = _orig(x_bd, *a, **kw)
            try:
                # Only capture from a converter whose input row actually has
                # OP_MUL + MARK_AX set (the real MUL block). Other ALU blocks'
                # converters run too but with garbage at this row.
                op_mul = x_bd[0, :, _BD.OP_MUL] > 0.5
                mark_ax = x_bd[0, :, _BD.MARK_AX] > 0.5
                mul_rows = (op_mul & mark_ax).nonzero().flatten().tolist()
                if mul_rows:
                    _CAP["mul_rows"] = mul_rows
                    _CAP["a_lo"] = x_ge[:, :, 0, _ge.NIB_A].detach()
                    _CAP["a_hi"] = x_ge[:, :, 1, _ge.NIB_A].detach()
                    _CAP["b_lo"] = x_ge[:, :, 0, _ge.NIB_B].detach()
                    _CAP["b_hi"] = x_ge[:, :, 1, _ge.NIB_B].detach()
                    _CAP["a_b1lo"] = x_ge[:, :, 2, _ge.NIB_A].detach()
                    _CAP["a_b1hi"] = x_ge[:, :, 3, _ge.NIB_A].detach()
            except Exception as e:  # noqa
                _CAP["err"] = str(e)
            return x_ge

        m.forward = patched
    return convs


def argmax_nib(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    cells = [float(row[base + i].item()) for i in range(width)]
    mx = max(range(width), key=lambda i: cells[i])
    return mx, round(cells[mx], 2)


def main():
    trace = "--trace" in sys.argv
    pairs = []
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            continue
        a, b = tok.split(":")
        pairs.append((int(a), int(b)))
    if not pairs:
        pairs = [(3, 15), (11, 11), (93, 34)]

    print("[probe] building probe model (campaign env from environ)...",
          file=sys.stderr, flush=True)
    probe = build_groundtruth_probe()
    model = probe.model
    install_capture(model)
    dp = model.dim_positions
    dev = next(model.parameters()).device

    for (a, b) in pairs:
        exp = a * b
        bc = prog(a, b)
        ctx = probe._final_context(bc, max_steps=20)
        # last REG_AX token = the MUL-result AX marker row
        ax_pos = None
        for i in range(len(ctx) - 1, -1, -1):
            if ctx[i] == int(Token.REG_AX):
                ax_pos = i
                break
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        _CAP.clear()
        with torch.no_grad():
            full = model.forward(toks)[0]
        if "mul_rows" not in _CAP:
            print(f"A={a} B={b}: no MUL-AX row captured ({_CAP.get('err')})",
                  flush=True)
            continue
        r = _CAP["mul_rows"][-1]
        a_lo = float(_CAP["a_lo"][0, r]); a_hi = float(_CAP["a_hi"][0, r])
        b_lo = float(_CAP["b_lo"][0, r]); b_hi = float(_CAP["b_hi"][0, r])
        a_b1lo = float(_CAP["a_b1lo"][0, r]); a_b1hi = float(_CAP["a_b1hi"][0, r])
        opA = a_lo + a_hi * 16.0
        opB = b_lo + b_hi * 16.0
        opA_b1 = a_b1lo + a_b1hi * 16.0
        prod = (opA + opA_b1 * 256.0) * opB
        print(f"A={a:3d} B={b:3d} exp=0x{exp:04x}({exp})  "
              f"reconA_b0={opA:.2f}(lo={a_lo:.1f},hi={a_hi:.1f}) "
              f"reconA_b1={opA_b1:.2f} reconB={opB:.2f} "
              f"-> alumul_prod={prod:.1f}(0x{int(prod)&0xffff:04x})", flush=True)

        if trace:
            nblocks = len(model.blocks)
            prev = None
            for blk in range(nblocks):
                with torch.no_grad():
                    rr = model.forward(toks, stop_after_block=blk)[0]
                lg = getattr(model.blocks[blk], "_logical_layer", blk)
                ol = argmax_nib(rr[r], dp, 'OUTPUT_LO')
                oh = argmax_nib(rr[r], dp, 'OUTPUT_HI')
                # raw band sums
                blo = dp.get('OUTPUT_LO'); bhi = dp.get('OUTPUT_HI')
                slo = float(rr[r, blo:blo+16].sum().item()) if blo else 0.0
                shi = float(rr[r, bhi:bhi+16].sum().item()) if bhi else 0.0
                byte = ((oh[0] if oh else 0) << 4) | (ol[0] if ol else 0)
                cur = (ol, oh)
                mk = " <==CH" if cur != prev else ""
                print(f"   blk{blk:2d}(L{lg:2d}) OUT_LO={str(ol):13s} "
                      f"OUT_HI={str(oh):13s} sumLO={slo:.2g} sumHI={shi:.2g} "
                      f"->0x{byte:02x}{mk}", flush=True)
                prev = cur


if __name__ == "__main__":
    main()
