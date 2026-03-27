"""Debug token 21 (ST_b0) prediction in detail."""
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
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Capture states at different points
l10_attn_out = None
l10_ffn_out = None
l10_post_ops_out = None

def hook_attn(module, input, output):
    global l10_attn_out
    l10_attn_out = output.detach().clone() if isinstance(output, torch.Tensor) else output[0].detach().clone()

def hook_ffn(module, input, output):
    global l10_ffn_out
    l10_ffn_out = output.detach().clone()

def hook_post_ops(module, input, output):
    global l10_post_ops_out
    l10_post_ops_out = output.detach().clone()

# Register hooks on Layer 10
model.blocks[10].attn.register_forward_hook(hook_attn)
model.blocks[10].ffn.register_forward_hook(hook_ffn)
model.blocks[10].post_ops[0].register_forward_hook(hook_post_ops)  # DivModModule

# Run through tokens 0-21
current_context = context[:]
for i in range(22):
    ctx_tensor = torch.tensor([current_context], dtype=torch.long)
    with torch.no_grad():
        _ = model.forward(ctx_tensor)
    current_context.append(draft_tokens[i])

# Now analyze token 21 prediction
print("Token 21 (ST_b0) Analysis:")
print("=" * 60)
print()

# Check what token 21 is
token_21_pos = len(context) + 21
print(f"Sequence position for token 21: {token_21_pos}")
print()

# Check MARK_AX at position 21
if l10_attn_out is not None:
    S = l10_attn_out.shape[1]
    print(f"Sequence length: {S}")

    mark_ax_attn = l10_attn_out[0, token_21_pos, BD.MARK_AX].item()
    mark_pc_attn = l10_attn_out[0, token_21_pos, BD.MARK_PC].item()
    mark_sp_attn = l10_attn_out[0, token_21_pos, BD.MARK_SP].item()
    mark_bp_attn = l10_attn_out[0, token_21_pos, BD.MARK_BP].item()

    print("Position markers at token 21 after L10 attention:")
    print(f"  MARK_AX: {mark_ax_attn:.3f}")
    print(f"  MARK_PC: {mark_pc_attn:.3f}")
    print(f"  MARK_SP: {mark_sp_attn:.3f}")
    print(f"  MARK_BP: {mark_bp_attn:.3f}")
    print()

if l10_ffn_out is not None:
    mark_ax_ffn = l10_ffn_out[0, token_21_pos, BD.MARK_AX].item()
    print(f"MARK_AX after L10 FFN: {mark_ax_ffn:.3f}")
    print()

# Check OUTPUT values through the pipeline
if l10_attn_out is not None:
    output_attn = l10_attn_out[0, token_21_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    print("OUTPUT_LO after L10 attention:")
    print(f"  {output_attn.tolist()}")
    print()

if l10_ffn_out is not None:
    output_ffn = l10_ffn_out[0, token_21_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    print("OUTPUT_LO after L10 FFN:")
    print(f"  {output_ffn.tolist()}")
    print()

if l10_post_ops_out is not None:
    output_post = l10_post_ops_out[0, token_21_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    print("OUTPUT_LO after L10 post_ops (DivMod):")
    print(f"  {output_post.tolist()}")
    print()

    # Check if DivMod changed OUTPUT
    if l10_ffn_out is not None:
        delta = l10_post_ops_out[0, token_21_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16] - l10_ffn_out[0, token_21_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        if torch.any(torch.abs(delta) > 0.01):
            print("DivMod CHANGED OUTPUT_LO:")
            print(f"  Delta: {delta.tolist()}")
            print()
        else:
            print("DivMod did NOT change OUTPUT_LO (gating worked)")
            print()

# Check what the final prediction is
ctx_tensor = torch.tensor([current_context[:-1]], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)
predicted = torch.argmax(logits[0, -1]).item()

print(f"Draft token 21: {draft_tokens[21]}")
print(f"Predicted token 21: {predicted}")
print(f"Expected token 21: 0")
print()

if predicted != 0:
    print(f"❌ MISMATCH: predicted {predicted} instead of 0")
    print(f"   {predicted} in hex: 0x{predicted:02x}")
    print(f"   {predicted} in binary: 0b{predicted:08b}")
