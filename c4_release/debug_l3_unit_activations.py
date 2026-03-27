"""Check which L3 FFN units are firing."""
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

# Test JMP 12
bytecode = [Opcode.JMP | (12 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

context_with_draft = context + draft_tokens
ctx_len = len(context)
pc_marker_pos = ctx_len

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

ffn = model.blocks[3].ffn

# Hook to capture L3 FFN intermediate activations
l3_attn_output = {}
def capture_l3_attn(module, input, output):
    l3_attn_output['x'] = output.clone()
model.blocks[3].attn.register_forward_hook(capture_l3_attn)

ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

x_in = l3_attn_output['x'][0, pc_marker_pos, :]  # Input to L3 FFN at PC marker

print("L3 FFN Unit Activations at PC Marker")
print("="*70)

# Key dimensions
mark_pc = x_in[BD.MARK_PC].item()
has_se = x_in[BD.HAS_SE].item()
embed_lo_vals = x_in[BD.EMBED_LO:BD.EMBED_LO+16]

print(f"Input values:")
print(f"  MARK_PC: {mark_pc:.4f}")
print(f"  HAS_SE: {has_se:.4f}")
print(f"  EMBED_LO[0]: {embed_lo_vals[0].item():.4f}")
print(f"  EMBED_LO[2]: {embed_lo_vals[2].item():.4f}")
print(f"  EMBED_LO[7]: {embed_lo_vals[7].item():.4f}")
print()

# Check units that write to OUTPUT_LO[2] or OUTPUT_LO[7]
print("="*70)
print("Units writing to OUTPUT_LO[2] (first-step default):")
print("-"*70)

for unit in range(100):  # Check first 100 units
    w_down_2 = ffn.W_down[BD.OUTPUT_LO + 2, unit].item()
    if abs(w_down_2) > 0.01:
        # Compute activation
        up_val = (ffn.W_up[unit, :] @ x_in + ffn.b_up[unit])
        gate_val = (ffn.W_gate[unit, :] @ x_in + ffn.b_gate[unit])

        up_act = torch.nn.functional.silu(up_val).item()
        gate_act = torch.sigmoid(gate_val).item()
        output = up_act * gate_act * w_down_2

        w_up_mark_pc = ffn.W_up[BD.MARK_PC, unit].item()
        w_up_has_se = ffn.W_up[BD.HAS_SE, unit].item()

        print(f"\nUnit {unit}:")
        print(f"  W_down[OUTPUT_LO+2]: {w_down_2:.4f}")
        print(f"  W_up[MARK_PC]: {w_up_mark_pc:.2f}, W_up[HAS_SE]: {w_up_has_se:.2f}")
        print(f"  up input: {up_val.item():.4f} → silu: {up_act:.4f}")
        print(f"  gate input: {gate_val.item():.4f} → sigmoid: {gate_act:.4f}")
        print(f"  → Contribution to OUTPUT_LO[2]: {output:.4f}")

print("\n" + "="*70)
print("Units writing to OUTPUT_LO[7] (PC increment k=2 → new_k=7):")
print("-"*70)

for unit in range(100):
    w_down_7 = ffn.W_down[BD.OUTPUT_LO + 7, unit].item()
    if abs(w_down_7) > 0.01:
        # Compute activation
        up_val = (ffn.W_up[unit, :] @ x_in + ffn.b_up[unit])
        gate_val = (ffn.W_gate[unit, :] @ x_in + ffn.b_gate[unit])

        up_act = torch.nn.functional.silu(up_val).item()
        gate_act = torch.sigmoid(gate_val).item()
        output = up_act * gate_act * w_down_7

        w_up_mark_pc = ffn.W_up[BD.MARK_PC, unit].item()
        w_up_has_se = ffn.W_up[BD.HAS_SE, unit].item()
        w_gate_embed = ffn.W_gate[BD.EMBED_LO+2, unit].item()

        print(f"\nUnit {unit}:")
        print(f"  W_down[OUTPUT_LO+7]: {w_down_7:.4f}")
        print(f"  W_up[MARK_PC]: {w_up_mark_pc:.2f}, W_up[HAS_SE]: {w_up_has_se:.2f}")
        print(f"  W_gate[EMBED_LO+2]: {w_gate_embed:.4f}")
        print(f"  up input: {up_val.item():.4f} → silu: {up_act:.4f}")
        print(f"  gate input: {gate_val.item():.4f} → sigmoid: {gate_act:.4f}")
        print(f"  → Contribution to OUTPUT_LO[7]: {output:.4f}")

print("\n" + "="*70)
print("Summary:")
print("-"*70)
print("If PC increment is firing with HAS_SE=0, this is a BUG.")
print("The threshold should prevent activation when HAS_SE=0.")
