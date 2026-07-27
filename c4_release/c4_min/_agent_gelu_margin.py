"""Quantify WHY SiLU->GeLU is byte-exact: per-step AX-nibble-dim residual divergence.

Runs a multi-step program to the ALU-RESULT step (where ADD/MUL actually fires the
ALU FFN), under silu vs swish_b vs gelu, and reports the max-abs divergence at the
AX nibble decode dims.  If that stays << 8 (half a nibble step) the argmax re-quantiser
snaps back byte-exact — the activation floor is ABSORBED, not inert.
"""
from __future__ import annotations
import os, time, torch
from . import nibble_pure_forward_complete as N
from . import isa
from . import blogspec_vocab as V
from .nibble_vm import _snap_lane
from ._agent_transform_verify import patch_ffn_activation, _gelu, _swish_b, BETA

DEV = os.environ.get("C4_XFORM_DEVICE", "cuda:0")


def run_capture_ax(model, L, code, dev, mask=0xFF):
    """Run the driver loop; return a list per step of (op_name, ax, full_residual_last_row)."""
    stream = [V.BOS] + N._build_frame(0, 0, N.SP_INIT, N.SP_INIT, 0)
    store_log = {}
    cur_pc = 0; cur_sp = cur_bp = N.SP_INIT; frame_idx = 0
    caps = []
    for _ in range(10):
        overlay = N.make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream], device=dev)
        with torch.no_grad():
            x = model.embed[toks].clone(); overlay(x)
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1]
        pc = _snap_lane(state[L.PC_VAL]); sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL]); stk = _snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ax = N._decode_reg_from_nibbles(state, L, L.AX)
        caps.append((isa.NAMES.get(op, op), ax,
                     state[[L.AX + k for k in range(8)]].detach().float().cpu().clone()))
        s_addr = s_val = 0; is_store = False
        if op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        frame = N._build_frame(pc, ax, sp, bp, stk,
                               mem_addr=(s_addr if is_store else 0),
                               mem_val=(s_val if is_store else 0))
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        cur_pc, cur_sp, cur_bp = pc, sp, bp
        if halted or pc < 0 or pc >= len(code):
            break
    return caps


def main():
    t0 = time.time()
    model, L = N.build_pure_forward_complete_model(code_size=24, recurrent_divmod=True)
    model.eval(); dev = torch.device(DEV); model.to(dev)
    print(f"built {time.time()-t0:.1f}s", flush=True)
    progs = {
        "add(5+3)": isa.assemble([('IMM', 5), ('PSH', 0), ('IMM', 3), ('ADD', 0), ('HALT', 0)]),
        "mul(6*7)": isa.assemble([('IMM', 6), ('PSH', 0), ('IMM', 7), ('MUL', 0), ('HALT', 0)]),
    }
    for name, code in progs.items():
        base = run_capture_ax(model, L, code, dev)
        r = patch_ffn_activation(lambda x: _swish_b(x, BETA))
        sw = run_capture_ax(model, L, code, dev); r()
        r = patch_ffn_activation(_gelu)
        ge = run_capture_ax(model, L, code, dev); r()
        print(f"\n{name}: per-step AX (op, ax) + max-abs AX-nibble-dim divergence vs silu:",
              flush=True)
        for i in range(min(len(base), len(sw), len(ge))):
            opn, axb, vb = base[i]
            dsw = (sw[i][2] - vb).abs().max().item()
            dge = (ge[i][2] - vb).abs().max().item()
            print(f"    step{i} {opn:5s} ax_silu={axb:3d} ax_swish={sw[i][1]:3d} "
                  f"ax_gelu={ge[i][1]:3d} | dAXdim swish={dsw:.5f} gelu={dge:.5f} "
                  f"(flip@~8.0)", flush=True)


if __name__ == "__main__":
    main()
