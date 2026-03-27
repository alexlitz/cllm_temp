"""Simplified debug of token 21 (ST_b0) prediction."""
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

# Build sequence up to token 20 (before ST_b0)
current_context = context[:]
for i in range(21):  # tokens 0-20
    current_context.append(draft_tokens[i])

# Now predict token 21
ctx_tensor = torch.tensor([current_context], dtype=torch.long)

# Hook to capture Layer 10 FFN output (before post_ops)
l10_ffn_out = None
def hook_ffn(module, input, output):
    global l10_ffn_out
    l10_ffn_out = output.detach().clone()

model.blocks[10].ffn.register_forward_hook(hook_ffn)

# Hook to capture DivMod module's delta
divmod_input = None
divmod_output = None
def hook_divmod(module, input, output):
    global divmod_input, divmod_output
    divmod_input = input[0].detach().clone()
    divmod_output = output.detach().clone()

model.blocks[10].post_ops[0].register_forward_hook(hook_divmod)

# Forward pass
with torch.no_grad():
    logits = model.forward(ctx_tensor)

predicted = torch.argmax(logits[0, -1]).item()

print("Token 21 (ST_b0) Prediction Analysis")
print("=" * 60)
print()

print(f"Draft token 21: {draft_tokens[21]}")
print(f"Predicted: {predicted}")
print(f"Expected: 0")
print()

if predicted != 0:
    print(f"❌ MISMATCH: predicted {predicted} (0x{predicted:02x}) instead of 0")
else:
    print("✓ CORRECT")

print()
print("=" * 60)

# Analyze position 30 (token 21 position in sequence)
pos = len(current_context) - 1
print(f"Analyzing position {pos} (token 21)")
print()

# Check markers
if divmod_input is not None:
    print("Markers at token 21 position (input to DivMod):")
    print(f"  MARK_AX: {divmod_input[0, pos, BD.MARK_AX]:.3f}")
    print(f"  MARK_PC: {divmod_input[0, pos, BD.MARK_PC]:.3f}")
    print(f"  MARK_SP: {divmod_input[0, pos, BD.MARK_SP]:.3f}")
    print(f"  MARK_BP: {divmod_input[0, pos, BD.MARK_BP]:.3f}")
    print()

# Check OUTPUT before and after DivMod
if divmod_input is not None and divmod_output is not None:
    output_lo_before = divmod_input[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_lo_after = divmod_output[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    delta_lo = output_lo_after - output_lo_before

    output_hi_before = divmod_input[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    output_hi_after = divmod_output[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    delta_hi = output_hi_after - output_hi_before

    print("OUTPUT_LO at token 21:")
    print(f"  Before DivMod: argmax={torch.argmax(output_lo_before).item()}, max={torch.max(output_lo_before):.2f}")
    print(f"  After DivMod:  argmax={torch.argmax(output_lo_after).item()}, max={torch.max(output_lo_after):.2f}")
    print(f"  Delta:         {delta_lo.tolist()}")
    print()

    print("OUTPUT_HI at token 21:")
    print(f"  Before DivMod: argmax={torch.argmax(output_hi_before).item()}, max={torch.max(output_hi_before):.2f}")
    print(f"  After DivMod:  argmax={torch.argmax(output_hi_after).item()}, max={torch.max(output_hi_after):.2f}")
    print(f"  Delta:         {delta_hi.tolist()}")
    print()

    if torch.any(torch.abs(delta_lo) > 0.01) or torch.any(torch.abs(delta_hi) > 0.01):
        print("⚠ DivMod MODIFIED OUTPUT despite gating!")
        print(f"   Max delta_lo: {torch.max(torch.abs(delta_lo)):.3f}")
        print(f"   Max delta_hi: {torch.max(torch.abs(delta_hi)):.3f}")
    else:
        print("✓ DivMod did not modify OUTPUT (gating worked)")

print()

# Check what position 30 should be
print(f"Token 21 is at position {pos} in the sequence")
print(f"Context length: {len(context)} (CODE_START + instruction + CODE_END + DATA_START + DATA_END)")
print(f"Tokens 0-20: {len(draft_tokens[:21])} tokens")
print(f"Total sequence length: {len(current_context)}")
