#!/usr/bin/env python3
"""Probe the rec cluster step-0 PC emission: which PC byte is wrong for entry
PC >= 256, and what drives byte 0/1 via LM-head logit attribution.

The model has no final norm, so logit[t] = head.weight[t] . residual_block(last)
is exact. We decode the emitted step-0 REG_PC bytes (spec_k=0), compare to the
oracle JSR target, and attribute the byte-0 and byte-1 predictor rows.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_rec_step0_pc.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch
from src.compiler import compile_c
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from tests.test_suite_1000 import generate_test_programs

_REG = build_default_registry_dynamic()


def name_for(pos):
    best = None
    for nm, slot in _REG.slots.items():
        if slot.start <= pos < slot.start + slot.size:
            if best is None or slot.size < best[1]:
                best = (f"{nm}+{pos - slot.start}", slot.size)
    return best[0] if best else f"dim{pos}"


@torch.no_grad()
def residual_full(probe, ctx, block_idx, position):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    return x[0, position]


def main():
    idxs = [int(a) for a in sys.argv[1:]] or [702, 725, 750, 775]
    probe = GroundTruthProbe.build()
    model = probe.model
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = model.head.bias
    tests = generate_test_programs()
    last_block = len(model.blocks) - 1
    PC = int(Token.REG_PC)

    for idx in idxs:
        src, exp, desc = tests[idx]
        bc, data = compile_c(src)
        # Decode oracle entry PC: step 0 is the JSR; the emitted PC is the JSR
        # target = main's ENT instruction. main JSR is bc[0] = JSR op, operand
        # = instruction index of main's ENT.
        main_idx = (bc[0] >> 8)
        from neural_vm.constants import INSTR_WIDTH, PC_OFFSET
        oracle_pc = main_idx * INSTR_WIDTH + PC_OFFSET
        ob = [(oracle_pc >> (8 * j)) & 0xFF for j in range(4)]
        print(f"\n{'='*70}\nidx {idx}  {desc}")
        print(f"  main_idx={main_idx}  oracle step-0 PC = {oracle_pc} "
              f"(0x{oracle_pc:08x})  bytes LE = {ob}")

        # Replay just enough to get step-0 PC bytes. Step 0 emits at most
        # REG_PC + 4 bytes. Build context and step through manually.
        ctx = probe._build_context(bc)
        prompt_len = len(ctx)
        emitted = []  # (pos, token)
        pc_marker_pos = None
        pc_byte_rows = []  # positions whose NEXT token is a PC byte
        # We need: the REG_PC marker emission, then 4 byte emissions.
        # The "predictor row" for PC byte k is the row whose next-token logits
        # produce byte k (i.e. position pc_marker_pos + k).
        for _ in range(12):  # step 0 needs ~ marker+4 = 5 tokens
            logits = probe._forward_logits(ctx)
            last = logits[len(ctx) - 1]
            nxt = int(last.argmax().item())
            pos = len(ctx)
            emitted.append((pos, nxt))
            ctx.append(nxt)
            if nxt == PC and pc_marker_pos is None:
                pc_marker_pos = pos
            if pc_marker_pos is not None and pos > pc_marker_pos and len(pc_byte_rows) < 4:
                # the predictor row for this byte is pos-1
                pass
            # stop once we have marker + 4 bytes
            if pc_marker_pos is not None and len(ctx) >= pc_marker_pos + 5:
                break

        if pc_marker_pos is None:
            print("  !! no REG_PC marker emitted in first tokens; emitted:",
                  [(p, t) for p, t in emitted])
            continue
        # neural PC bytes = the 4 tokens AFTER the marker
        neural_bytes = []
        for k in range(4):
            bp = pc_marker_pos + 1 + k  # position of byte k token
            # find token at bp
            tok = ctx[bp] if bp < len(ctx) else None
            neural_bytes.append(tok)
        print(f"  neural step-0 PC bytes LE = {neural_bytes}")
        for k in range(4):
            ok = "ok " if neural_bytes[k] == ob[k] else "WRONG"
            print(f"    byte{k}: oracle 0x{ob[k]:02x}  neural "
                  f"{('0x%02x' % neural_bytes[k]) if neural_bytes[k] is not None else 'None'}  {ok}")

        # LM-head logit attribution at each wrong byte's predictor row.
        for k in (0, 1):
            want = ob[k]
            got = neural_bytes[k]
            if got is None or want == got:
                continue
            pred_row = pc_marker_pos + k  # predictor row for byte k
            res = residual_full(probe, ctx, last_block, pred_row)
            if res.is_sparse:
                res = res.to_dense()
            res = res.to(W.device).float()
            dw = (W[want] - W[got]) * res
            dw = dw.to_dense() if dw.is_sparse else dw
            lw = float((W[want] * res).sum() + b[want])
            lg = float((W[got] * res).sum() + b[got])
            print(f"\n  --- byte{k} predictor row pos {pred_row}: "
                  f"want 0x{want:02x} got 0x{got:02x} ---")
            print(f"      logit[want]={lw:.2f}  logit[got]={lg:.2f}  "
                  f"diff(want-got)={lw-lg:.2f}")
            order = torch.argsort(dw.abs(), descending=True)
            for di in order[:16].tolist():
                print(f"      dim {di:4d} {name_for(di):28s} "
                      f"res={float(res[di]):9.3f} "
                      f"dW={float(W[want, di] - W[got, di]):7.3f} "
                      f"contrib={float(dw[di]):9.3f}")


if __name__ == "__main__":
    main()
