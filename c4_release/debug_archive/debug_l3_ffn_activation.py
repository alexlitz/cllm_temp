"""Debug why L3 FFN isn't activating."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
ctx_len = len(context)
pc_marker_pos = ctx_len

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Before L3
    for i in range(2):
        x = model.blocks[i](x)

    print("Before L3:")
    mark_pc = x[0, pc_marker_pos, BD.MARK_PC]
    has_se = x[0, pc_marker_pos, BD.HAS_SE]
    print(f"MARK_PC: {mark_pc.item():.3f}")
    print(f"HAS_SE: {has_se.item():.3f}")

    # Get L3 FFN
    l3_ffn = model.blocks[2].ffn
    
    # Manual FFN computation
    up = l3_ffn.up(x) + l3_ffn.b_up
    gate = l3_ffn.gate(x)  
    
    print(f"\nL3 FFN unit 0 (first-step PC default):")
    print(f"up[0] at PC marker: {up[0, pc_marker_pos, 0].item():.3f}")
    print(f"gate[0] at PC marker: {gate[0, pc_marker_pos, 0].item():.3f}")
    print(f"activation (up * silu(gate)): {(up[0, pc_marker_pos, 0] * torch.nn.functional.silu(gate[0, pc_marker_pos, 0])).item():.3f}")

    # Full forward
    x = model.blocks[2](x)
    
    print(f"\nAfter L3:")
    output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    print(f"OUTPUT_LO: max={output_lo.max().item():.6f}, argmax={output_lo.argmax().item()}")
    print(f"OUTPUT_LO[8]: {output_lo[8].item():.6f} (should be non-zero)")

