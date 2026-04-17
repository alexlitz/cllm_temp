"""Check which L6 FFN units are firing at PC marker."""
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

bytecode = [Opcode.JMP | (12 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

context_with_draft = context + draft_tokens
ctx_len = len(context)
pc_marker_pos = ctx_len

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

ffn = model.blocks[6].ffn

# Hook L6 attention output (input to L6 FFN)
l6_attn_output = {}
def capture_l6_attn(module, input, output):
    l6_attn_output['x'] = output.clone()
model.blocks[6].attn.register_forward_hook(capture_l6_attn)

# Hook L6 FFN output
l6_output = {}
def capture_l6(module, input, output):
    l6_output['x'] = output.clone()
model.blocks[6].register_forward_hook(capture_l6)

ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

x_in = l6_attn_output['x'][0, pc_marker_pos, :]
x_out = l6_output['x'][0, pc_marker_pos, :]

print("L6 FFN Analysis at PC Marker")
print("="*70)

# Check key input dimensions
mark_pc = x_in[BD.MARK_PC].item()
cmp0 = x_in[BD.CMP + 0].item()
cmp1 = x_in[BD.CMP + 1].item()

print(f"Input to L6 FFN:")
print(f"  MARK_PC: {mark_pc:.4f}")
print(f"  CMP[0] (IS_JMP): {cmp0:.4f}")
print(f"  CMP[1] (IS_EXIT): {cmp1:.4f}")

# Check INPUT OUTPUT_LO
output_lo_in = x_in[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_lo_in_idx = (output_lo_in.abs() > 0.5).nonzero(as_tuple=True)[0]
print(f"  INPUT OUTPUT_LO[{output_lo_in_idx.tolist() if len(output_lo_in_idx) > 0 else 'none'}]")

# Check OUTPUT OUTPUT_LO
output_lo_out = x_out[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
output_lo_out_idx = (output_lo_out.abs() > 0.5).nonzero(as_tuple=True)[0]
print(f"  OUTPUT OUTPUT_LO[{output_lo_out_idx.tolist() if len(output_lo_out_idx) > 0 else 'none'}]")

print()
print("="*70)
print("Checking units that changed OUTPUT_LO:")
print("-"*70)

# Find units that contribute to OUTPUT_LO changes
changed = False
for k in range(16):
    input_val = output_lo_in[k].item()
    output_val = output_lo_out[k].item()
    delta = output_val - input_val

    if abs(delta) > 0.1:
        changed = True
        print(f"\nOUTPUT_LO[{k}]: {input_val:.4f} → {output_val:.4f} (Δ={delta:+.4f})")

        # Find which units contributed
        print(f"  Units writing to OUTPUT_LO[{k}]:")
        for unit in range(min(500, ffn.W_down.shape[1])):
            w_down = ffn.W_down[BD.OUTPUT_LO + k, unit].item()
            if abs(w_down) > 0.001:
                # Compute activation
                up_val = (ffn.W_up[unit, :] @ x_in + ffn.b_up[unit]).item()
                gate_val = (ffn.W_gate[unit, :] @ x_in + ffn.b_gate[unit]).item()

                up_act = torch.nn.functional.silu(torch.tensor(up_val)).item()
                gate_act = torch.sigmoid(torch.tensor(gate_val)).item()
                contribution = up_act * gate_act * w_down

                if abs(contribution) > 0.01:
                    # Check what activated this unit
                    w_up_mark_pc = ffn.W_up[unit, BD.MARK_PC].item()
                    w_up_cmp0 = ffn.W_up[unit, BD.CMP + 0].item()

                    print(f"    Unit {unit}: contrib={contribution:+.4f}, W_down={w_down:.4f}")
                    print(f"      W_up[MARK_PC]={w_up_mark_pc:.1f}, W_up[CMP0]={w_up_cmp0:.1f}")
                    print(f"      up={up_val:.2f}→{up_act:.4f}, gate={gate_val:.2f}→{gate_act:.4f}")

if not changed:
    print("No significant changes to OUTPUT_LO")
