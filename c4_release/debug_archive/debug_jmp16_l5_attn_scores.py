"""Debug JMP 16 - L5 head 3 attention scores."""
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

print("JMP 16 - L5 Head 3 Attention Scores")
print("=" * 70)
print()

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx = context + [draft_tokens[0]]

# Manually compute L5 head 3 attention
# Get L5 attention input
l5_in = None
def l5_in_hook(module, input, output):
    global l5_in
    if isinstance(input, tuple):
        l5_in = input[0].detach().clone()
    else:
        l5_in = input.detach().clone()

model.blocks[5].attn.register_forward_hook(l5_in_hook)

ctx_tensor = torch.tensor([ctx], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

if l5_in is None:
    print("Failed to capture L5 input")
    sys.exit(1)

# Get attention parameters
attn = model.blocks[5].attn
HD = attn.head_dim
head_idx = 3
base = head_idx * HD

# Get Q, K matrices for head 3
W_q = attn.W_q[base:base+HD, :]
W_k = attn.W_k[base:base+HD, :]

# Compute Q and K
Q = F.linear(l5_in, W_q)  # [1, S, HD]
K = F.linear(l5_in, W_k)  # [1, S, HD]

# Compute attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale  # [1, S, S]

# Add ALiBi bias
S = l5_in.shape[1]
q_positions = torch.arange(S, device=l5_in.device)
kv_positions = torch.arange(S, device=l5_in.device)
dist = (q_positions.unsqueeze(1) - kv_positions.unsqueeze(0)).abs().float()
alibi_slope = attn.alibi_slopes[head_idx].item()
alibi = -alibi_slope * dist
scores = scores + alibi.unsqueeze(0)

# Apply softmax
attn_weights = F.softmax(scores, dim=-1)  # [1, S, S]

# Check scores at PC marker (position 9)
pc_pos = 9
print(f"Attention from PC marker (position {pc_pos}) to all positions:")
print("-" * 70)
print()

for i in range(min(S, 10)):
    token_val = ctx[i]
    score = scores[0, pc_pos, i].item()
    weight = attn_weights[0, pc_pos, i].item()

    # Get ADDR_KEY at this position
    addr_lo = l5_in[0, i, BD.ADDR_KEY:BD.ADDR_KEY+16]
    addr_hi = l5_in[0, i, BD.ADDR_KEY+16:BD.ADDR_KEY+32]
    addr_lo_idx = torch.argmax(addr_lo).item()
    addr_hi_idx = torch.argmax(addr_hi).item()

    print(f"Pos {i}: token={token_val:3d}, score={score:7.2f}, weight={weight:.3f}")
    print(f"       ADDR_KEY: lo={addr_lo_idx}, hi={addr_hi_idx}")

    if weight > 0.01:  # Show significant weights
        print(f"       ⚠️  Significant attention weight!")
    print()

print()
print("Expected:")
print("  Head 3 should attend strongly to address 1 (position 2)")
print("  Query asks for: ADDR_KEY_LO[1]=1, ADDR_KEY_HI[0]=1")
print("  Position 2 should have highest weight")
print()

print("Attention weights summary:")
print(f"  Position 2 (address 1): {attn_weights[0, pc_pos, 2].item():.3f}")
print(f"  Position 3 (address 2): {attn_weights[0, pc_pos, 3].item():.3f}")
