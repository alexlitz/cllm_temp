"""Check if markers are leaking from STACK0 marker to ST_b0 position."""
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

# Build sequence up to token 20
current_context = context[:]
for i in range(21):
    current_context.append(draft_tokens[i])

ctx_tensor = torch.tensor([current_context], dtype=torch.long)

# Capture Layer 5 output (before Layer 6)
l5_out = None
l6_attn_out = None
l6_ffn_in = None

def hook_l5(module, input, output):
    global l5_out
    l5_out = output.detach().clone()

def hook_l6_attn(module, input, output):
    global l6_attn_out, l6_ffn_in
    l6_attn_out = output.detach().clone()
    # FFN input is the output of attention (after residual)
    l6_ffn_in = output.detach().clone()

model.blocks[5].register_forward_hook(hook_l5)
model.blocks[6].attn.register_forward_hook(hook_l6_attn)

# Forward pass
with torch.no_grad():
    _ = model.forward(ctx_tensor)

print("Marker Leakage Analysis for Token 21 (ST_b0)")
print("=" * 70)
print()

# Check markers at positions 28 (STACK0 marker) and 29 (ST_b0)
print("Position 28 (Token 20 = STACK0 marker):")
if l6_ffn_in is not None:
    print(f"  MARK_STACK0: {l6_ffn_in[0, 28, BD.MARK_STACK0]:.3f}")
    print(f"  MARK_SP:     {l6_ffn_in[0, 28, BD.MARK_SP]:.3f}")
    print(f"  MARK_BP:     {l6_ffn_in[0, 28, BD.MARK_BP]:.3f}")
    print(f"  MARK_AX:     {l6_ffn_in[0, 28, BD.MARK_AX]:.3f}")
print()

print("Position 29 (Token 21 = ST_b0, byte after STACK0 marker):")
if l6_ffn_in is not None:
    print(f"  MARK_STACK0: {l6_ffn_in[0, 29, BD.MARK_STACK0]:.3f}")
    print(f"  MARK_SP:     {l6_ffn_in[0, 29, BD.MARK_SP]:.3f}")
    print(f"  MARK_BP:     {l6_ffn_in[0, 29, BD.MARK_BP]:.3f}")
    print(f"  MARK_AX:     {l6_ffn_in[0, 29, BD.MARK_AX]:.3f}")
print()

# Check EMBED values at position 29 (input to identity carry)
print("EMBED values at position 29 (input to L6 FFN identity carry):")
if l6_ffn_in is not None:
    embed_lo = l6_ffn_in[0, 29, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = l6_ffn_in[0, 29, BD.EMBED_HI:BD.EMBED_HI+16]
    print(f"  EMBED_LO argmax: {torch.argmax(embed_lo).item()}")
    print(f"  EMBED_HI argmax: {torch.argmax(embed_hi).item()} ← Should be 0, might be 2")
    print(f"  EMBED_HI values: {embed_hi.tolist()}")
print()

# Check if Layer 6 FFN identity carry is activating
print("If MARK_STACK0 > 0.5 at position 29, identity carry activates:")
print("  → Copies EMBED_HI[2] to OUTPUT_HI[2]")
print("  → This would explain OUTPUT_HI[2] = 9.22")
