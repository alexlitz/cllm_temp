"""Debug L3 FFN gate computation in detail."""
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

    print(f"\nL3 FFN unit 0 weights:")
    print(f"W_up[0, MARK_PC]: {l3_ffn.W_up[0, BD.MARK_PC].item():.3f}")
    print(f"b_up[0]: {l3_ffn.b_up[0].item():.3f}")
    print(f"b_gate[0]: {l3_ffn.b_gate[0].item():.3f}")

    # Check W_gate
    w_gate_nonzero = (l3_ffn.W_gate[0].abs() > 0.001).sum().item()
    print(f"W_gate[0] non-zero elements: {w_gate_nonzero} (should be 0)")

    # Manual computation
    up_proj = l3_ffn.up(x)  # Linear projection
    up = up_proj + l3_ffn.b_up

    gate_proj = l3_ffn.gate(x)  # Linear projection (should be ~0 if W_gate is 0)
    gate = gate_proj + l3_ffn.b_gate

    print(f"\nAt PC marker position {pc_marker_pos}:")
    print(f"up_proj[0] (before bias): {up_proj[0, pc_marker_pos, 0].item():.3f}")
    print(f"up[0] (after bias): {up[0, pc_marker_pos, 0].item():.3f}")
    print(f"gate_proj[0] (before bias): {gate_proj[0, pc_marker_pos, 0].item():.6f}")
    print(f"gate[0] (after bias): {gate[0, pc_marker_pos, 0].item():.6f}")
    print(f"Expected gate[0]: {l3_ffn.b_gate[0].item():.3f}")

    # Activation
    hidden = torch.nn.functional.silu(up) * gate
    print(f"\nsilu(up[0]): {torch.nn.functional.silu(up[0, pc_marker_pos, 0]).item():.6f}")
    print(f"hidden[0]: {hidden[0, pc_marker_pos, 0].item():.6f}")

    # Down projection
    output_delta = l3_ffn.down(hidden)
    print(f"\noutput_delta at OUTPUT_LO[8]: {output_delta[0, pc_marker_pos, BD.OUTPUT_LO + 8].item():.6f}")
