"""Debug L3 FFN increment units."""
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

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = build_context(bytecode)

# Build full context for step 2
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1
draft_vm.step()
draft2 = draft_vm.draft_tokens()
context_step2 = context_step1 + draft2

ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len_step2 = len(context_step1)
pc_marker_pos = ctx_len_step2

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run to L3 attention
    for i in range(3):
        x = model.blocks[i](x)
    x = x + model.blocks[3].attn(x)

    print("=== Input to L3 FFN ===")
    mark_pc = x[0, pc_marker_pos, BD.MARK_PC]
    has_se = x[0, pc_marker_pos, BD.HAS_SE]
    embed_lo = x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
    print(f"MARK_PC: {mark_pc.item():.3f}")
    print(f"HAS_SE: {has_se.item():.3f}")
    print(f"EMBED_LO[8]: {embed_lo[8].item():.3f}")

    # Check L3 FFN unit for k=8
    l3_ffn = model.blocks[3].ffn

    # Find the unit for PC increment k=8
    # Units are: first-step (4 units), then increment (16*2 units for lo)
    # Unit for k=8 increment is 4 + 8 = 12
    unit_k8_lo = 4 + 8

    print(f"\n=== L3 FFN unit {unit_k8_lo} (PC increment k=8 → new_k=0) ===")
    print(f"W_up[unit, HAS_SE={BD.HAS_SE}]: {l3_ffn.W_up[unit_k8_lo, BD.HAS_SE].item():.1f}")
    print(f"W_up[unit, MARK_PC={BD.MARK_PC}]: {l3_ffn.W_up[unit_k8_lo, BD.MARK_PC].item():.1f}")
    print(f"b_up[unit]: {l3_ffn.b_up[unit_k8_lo].item():.1f}")
    print(f"W_gate[unit, EMBED_LO[8]={BD.EMBED_LO+8}]: {l3_ffn.W_gate[unit_k8_lo, BD.EMBED_LO+8].item():.3f}")
    print(f"W_down[OUTPUT_LO[0]={BD.OUTPUT_LO}, unit]: {l3_ffn.W_down[BD.OUTPUT_LO, unit_k8_lo].item():.6f}")

    # Manual computation
    up_val = (l3_ffn.W_up[unit_k8_lo, :] @ x[0, pc_marker_pos, :] + l3_ffn.b_up[unit_k8_lo]).item()
    gate_val = (l3_ffn.W_gate[unit_k8_lo, :] @ x[0, pc_marker_pos, :] + l3_ffn.b_gate[unit_k8_lo]).item()
    hidden_val = torch.nn.functional.silu(torch.tensor(up_val)).item() * gate_val

    print(f"\n=== Manual computation ===")
    print(f"up = {up_val:.3f}")
    print(f"gate = {gate_val:.3f}")
    print(f"hidden = silu(up) * gate = {hidden_val:.3f}")
    print(f"output_delta[OUTPUT_LO[0]] = {hidden_val * l3_ffn.W_down[BD.OUTPUT_LO, unit_k8_lo].item():.6f}")
