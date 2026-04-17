"""Debug L3 nibble copy - check if units are activating."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
import torch.nn.functional as F
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

print("L3 Nibble Copy Unit Activation Check")
print("=" * 70)
print()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to AX byte 0
ctx = context + draft_tokens[:7]  # Include AX byte 0
pos = len(ctx) - 1

# Capture L3 FFN input and hidden
l3_in = None
l3_ffn_in = None
l3_ffn_hidden = None

def l3_hook(module, input, output):
    global l3_in
    if isinstance(input, tuple):
        l3_in = input[0].detach().clone()
    else:
        l3_in = input.detach().clone()

def l3_ffn_hook(module, input, output):
    global l3_ffn_in, l3_ffn_hidden
    if isinstance(input, tuple):
        l3_ffn_in = input[0].detach().clone()
    else:
        l3_ffn_in = input.detach().clone()

    # Compute hidden activation
    ffn = module
    up = F.linear(l3_ffn_in, ffn.W_up, ffn.b_up)
    gate = F.linear(l3_ffn_in, ffn.W_gate, ffn.b_gate)
    l3_ffn_hidden = F.silu(up) * gate

model.blocks[3].register_forward_hook(l3_hook)
model.blocks[3].ffn.register_forward_hook(l3_ffn_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Position {pos} (token {ctx[pos]} = AX byte 0 = 42):")
print()

# Check L3 FFN input
print("L3 FFN input:")
is_byte = l3_ffn_in[0, pos, BD.IS_BYTE].item()
h1_pc = l3_ffn_in[0, pos, BD.H1 + 0].item()
h1_ax = l3_ffn_in[0, pos, BD.H1 + 1].item()
h1_sp = l3_ffn_in[0, pos, BD.H1 + 2].item()
h1_bp = l3_ffn_in[0, pos, BD.H1 + 3].item()
embed_lo = l3_ffn_in[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
embed_hi = l3_ffn_in[0, pos, BD.EMBED_HI:BD.EMBED_HI+16]

print(f"  IS_BYTE: {is_byte:.3f}")
print(f"  H1[PC]: {h1_pc:.3f}")
print(f"  H1[AX]: {h1_ax:.3f}")
print(f"  H1[SP]: {h1_sp:.3f}")
print(f"  H1[BP]: {h1_bp:.3f}")
print(f"  EMBED_LO[10] (for value 42): {embed_lo[10].item():.3f}")
print(f"  EMBED_HI[2] (for value 42): {embed_hi[2].item():.3f}")
print()

# Check nibble copy units
# Unit 10 copies EMBED_LO[10] → OUTPUT_LO[10]
# Unit 18 copies EMBED_HI[2] → OUTPUT_HI[2]
print("Nibble copy unit activations:")
if l3_ffn_hidden is not None:
    unit_10 = l3_ffn_hidden[0, pos, 10].item()
    unit_26 = l3_ffn_hidden[0, pos, 16 + 10].item()  # HI unit offset by 16
    print(f"  Unit 10 (OUTPUT_LO[10]): {unit_10:.6f}")
    print(f"  Unit 26 (OUTPUT_HI[2]): {unit_26:.6f}")
    print()

    # Show top 10 activated units
    top_units = torch.topk(l3_ffn_hidden[0, pos, :], 10)
    print("Top 10 activated units:")
    for i in range(10):
        idx = top_units.indices[i].item()
        val = top_units.values[i].item()
        print(f"    Unit {idx}: {val:.6f}")
print()

print("Analysis:")
print("  If unit activations are near-zero, the AND gate is failing")
print("  Expected: IS_BYTE=1, H1[PC/SP/BP]=0 → should activate")
