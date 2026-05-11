#!/usr/bin/env python3
"""Find which CarryProp byte_idx=0 units fire at REG_AX_mark position and
contribute massive values to OUTPUT_LO/HI."""
import sys
import torch
import torch.nn.functional as F

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def main():
    val = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    model = runner.model
    model.eval()
    device = next(model.parameters()).device

    prog = [(Opcode.IMM, val), Opcode.EXIT]
    bytecode = _make_bc(prog)
    context = runner._build_context(bytecode, b"", [], b"")
    prefix_len = len(context)

    with torch.no_grad():
        for i in range(Token.STEP_TOKENS):
            tok = model.generate_next(context, use_incremental=False)
            context.append(tok)

    token_ids = torch.tensor([context], dtype=torch.long, device=device)

    # Position 5 of step 0 = REG_AX_mark
    pos = prefix_len + 5
    print(f"REG_AX_mark at pos {pos}, token={context[pos]}", flush=True)

    # Run through blocks[0..12] then extract block 13's FFN
    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        for li in range(13):  # 0..12
            x = model.blocks[li](x)
        # x is now input to block 13 (CarryProp byte_idx=0)
        block13 = model.blocks[13]
        # post_op blocks: attn is passthrough (zero-init), so attn(x) ≈ x
        x_attn = block13.attn(x)
        # FFN
        ffn = block13.ffn
        print(f"\nBlock 13 ffn type: {type(ffn).__name__}", flush=True)
        # Print pertinent flag values BEFORE FFN
        v = x_attn[0, pos]
        print(f"\nFlags at pos {pos} BEFORE block 13 FFN:")
        print(f"  MARK_AX={v[BD.MARK_AX].item():.2f}, IS_MARK={v[BD.IS_MARK].item():.2f}, IS_BYTE={v[BD.IS_BYTE].item():.2f}")
        print(f"  H1[0..6]={[v[BD.H1+i].item() for i in range(7)]}")
        print(f"  BYTE_INDEX_0={v[BD.BYTE_INDEX_0].item():.2f}, BYTE_INDEX_1={v[BD.BYTE_INDEX_1].item():.2f}, BYTE_INDEX_2={v[BD.BYTE_INDEX_2].item():.2f}, BYTE_INDEX_3={v[BD.BYTE_INDEX_3].item():.2f}")
        print(f"  OP_IMM={v[BD.OP_IMM].item():.2f}, OP_ADD={v[BD.OP_ADD].item():.2f}, OP_SUB={v[BD.OP_SUB].item():.2f}")
        print(f"  CARRY[0..4]={[v[BD.CARRY+i].item() for i in range(5)]}")
        print(f"  TEMP[3]={v[BD.TEMP + 3].item():.2f}")
        print(f"  OLO_top5: {[(i.item(), v[BD.OUTPUT_LO+i].item()) for i in v[BD.OUTPUT_LO:BD.OUTPUT_LO+16].abs().topk(5).indices]}")
        print(f"  OHI_top5: {[(i.item(), v[BD.OUTPUT_HI+i].item()) for i in v[BD.OUTPUT_HI:BD.OUTPUT_HI+16].abs().topk(5).indices]}")

        # Compute FFN hidden activations manually
        W_up = ffn.W_up
        if W_up.is_sparse:
            W_up = W_up.to_dense()
        W_gate = ffn.W_gate
        if W_gate.is_sparse:
            W_gate = W_gate.to_dense()
        W_down = ffn.W_down
        if W_down.is_sparse:
            W_down = W_down.to_dense()
        b_up = ffn.b_up
        b_gate = ffn.b_gate

        x_pos = x_attn[0, pos]
        up = (W_up @ x_pos) + b_up
        gate = (W_gate @ x_pos) + b_gate
        silu_up = F.silu(up)
        hidden = silu_up * gate

        H = W_up.shape[0]
        print(f"\nFFN hidden dim: {H}", flush=True)
        # Top firing units by |hidden|
        topk = hidden.abs().topk(20)
        print(f"\nTop-20 firing units by |hidden| at REG_AX_mark:")
        print(f"  {'unit':>5} {'up':>12} {'gate':>10} {'silu(up)':>12} {'hidden':>12}   top W_down OUTPUT contribution")
        for rank in range(20):
            u = topk.indices[rank].item()
            h_val = hidden[u].item()
            if abs(h_val) < 1e-5:
                continue
            # Find dominant W_down output dim
            wd_col = W_down[:, u]
            wd_topk = wd_col.abs().topk(3)
            wd_str = ",".join([f"d{i.item()}:{wd_col[i].item():+.3f}" for i in wd_topk.indices])
            print(f"  {u:>5d} {up[u].item():>+12.3f} {gate[u].item():>+10.3f} {silu_up[u].item():>+12.3f} {h_val:>+12.3f}   {wd_str}")

        # Print top W_up dims for the top firing unit
        if topk.values[0] > 0.1:
            u = topk.indices[0].item()
            wu_row = W_up[u]
            wu_top = wu_row.abs().topk(8)
            print(f"\nTop-firing unit {u} W_up dims:")
            for i in wu_top.indices:
                d = i.item()
                # Try to find the name
                dname = ""
                for attr in dir(BD):
                    if attr.startswith("_"):
                        continue
                    val_attr = getattr(BD, attr)
                    if isinstance(val_attr, int) and val_attr == d:
                        dname = attr
                        break
                print(f"  W_up[{d}] = {wu_row[d].item():+.2f}  ({dname})  residual[d]={x_pos[d].item():.3f}")
            print(f"  b_up = {b_up[u].item():.2f}")
            wg_row = W_gate[u]
            wg_top = wg_row.abs().topk(4)
            print(f"\nTop-firing unit {u} W_gate dims:")
            for i in wg_top.indices:
                d = i.item()
                dname = ""
                for attr in dir(BD):
                    if attr.startswith("_"):
                        continue
                    val_attr = getattr(BD, attr)
                    if isinstance(val_attr, int) and val_attr == d:
                        dname = attr
                        break
                print(f"  W_gate[{d}] = {wg_row[d].item():+.2f}  ({dname})  residual[d]={x_pos[d].item():.3f}")
            print(f"  b_gate = {b_gate[u].item():.2f}")


if __name__ == "__main__":
    main()
