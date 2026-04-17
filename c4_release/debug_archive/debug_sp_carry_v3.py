#!/usr/bin/env python3
"""Debug SP pop carry detection - detailed layer tracing."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]
vm = DraftVM(bytecode)

# Run to step 3
for i in range(4):
    vm.step()
    if i == 2:
        draft_step2 = vm.draft_tokens()
    if i == 3:
        draft_step3 = vm.draft_tokens()

print("Step 2 (IMM 0):")
print(f"  SP bytes: {draft_step2[11:15]}")
print("Step 3 (MUL):")
print(f"  SP bytes: {draft_step3[11:15]} (expected: [0, 0, 1, 0] for SP=0x10000)")

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        op, imm = instr & 0xFF, instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])
    return context

context = build_context(bytecode)

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim
S = 100.0

# Build context up to step 3
vm2 = DraftVM(bytecode)
all_tokens = context[:]
for i in range(3):
    vm2.step()
    all_tokens.extend(vm2.draft_tokens())

vm2.step()  # Step 3 (MUL)
step3_tokens = vm2.draft_tokens()

# Register hooks for both attn and ffn
activations = {}
def hook_attn_out(name):
    def fn(module, input, output):
        # For attention, capture the attention output (first element of tuple or tensor)
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

def hook_ffn_out(name):
    def fn(module, input, output):
        activations[name] = output.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.attn.register_forward_hook(hook_attn_out(f'L{i}_attn'))
    block.ffn.register_forward_hook(hook_ffn_out(f'L{i}_ffn'))

# Test at SP marker position
test_context = all_tokens + step3_tokens[:11]  # Up to SP marker
token_ids = torch.tensor([test_context], dtype=torch.long)

print(f"\n=== At SP marker (step 3) ===")
print(f"Context length: {len(test_context)}")
print(f"Step3 token structure: PC_M={step3_tokens[0]}, AX_M={step3_tokens[5]}, SP_M={step3_tokens[10]}")

with torch.no_grad():
    logits = model(token_ids)

# Check OP_MUL at AX marker position
ax_pos = len(test_context) - 6  # AX marker is 5 tokens before SP marker
sp_pos = len(test_context) - 1  # SP marker is last token

print(f"\n=== Opcode decode at AX marker (pos {ax_pos}) ===")
h5_ax = activations['L5_ffn'][0, ax_pos, :]
print(f"OP_MUL: {h5_ax[BD.OP_MUL].item():.3f} (expected ≈2-3)")
print(f"OP_ADD: {h5_ax[BD.OP_ADD].item():.3f}")
print(f"OP_PSH: {h5_ax[BD.OP_PSH].item():.3f}")

print(f"\n=== Opcode relay to SP marker (pos {sp_pos}) ===")
# After L5: no relay yet
h5_sp = activations['L5_ffn'][0, sp_pos, :]
print(f"After L5 - CMP[3]: {h5_sp[BD.CMP + 3].item():.3f}")

# After L6 attn: relay should happen
h6_attn_sp = activations['L6_attn'][0, sp_pos, :]
print(f"After L6 attn - CMP[3]: {h6_attn_sp[BD.CMP + 3].item():.3f}")

# After L6 FFN: carry detection
h6_ffn_sp = activations['L6_ffn'][0, sp_pos, :]
print(f"After L6 FFN - CMP[3]: {h6_ffn_sp[BD.CMP + 3].item():.3f}")
print(f"After L6 FFN - SP_POP_CARRY_0: {h6_ffn_sp[BD.SP_POP_CARRY_0].item():.3f}")

print(f"\n=== Carry detection inputs at SP marker ===")
# These should be checked at the INPUT to L6 FFN, which is h6_attn_sp (output of L6 attn)
print(f"MARK_SP: {h6_attn_sp[BD.MARK_SP].item():.3f}")
print(f"CMP[3]: {h6_attn_sp[BD.CMP + 3].item():.3f}")
print(f"EMBED_HI[15]: {h6_attn_sp[BD.EMBED_HI + 15].item():.3f}")
print(f"EMBED_LO[8]: {h6_attn_sp[BD.EMBED_LO + 8].item():.3f}")

# Compute expected activation at unit 1200
mark_sp = h6_attn_sp[BD.MARK_SP].item()
cmp3 = h6_attn_sp[BD.CMP + 3].item()
embed_hi15 = h6_attn_sp[BD.EMBED_HI + 15].item()
embed_lo8 = h6_attn_sp[BD.EMBED_LO + 8].item()
T = 2.0
expected_up = S * (mark_sp + cmp3 + embed_hi15) - S * T
print(f"\nExpected unit 1200 up activation: {S}*({mark_sp:.2f} + {cmp3:.2f} + {embed_hi15:.2f}) - {S}*{T} = {expected_up:.1f}")
print(f"Gate input (EMBED_LO[8]): {embed_lo8:.3f}")

# Check L10 where passthrough + carry application happens
h10_sp = activations['L10_ffn'][0, sp_pos, :]
print(f"\n=== After L10 (passthrough + carry) ===")
print(f"SP_POP_CARRY_0: {h10_sp[BD.SP_POP_CARRY_0].item():.3f}")
out_lo = h10_sp[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
out_hi = h10_sp[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
print(f"OUTPUT_LO argmax: {out_lo.argmax().item()}, max: {out_lo.max().item():.3f}")
print(f"OUTPUT_HI argmax: {out_hi.argmax().item()}, max: {out_hi.max().item():.3f}")

# Test all SP byte predictions
print("\n=== Testing SP byte predictions ===")
for byte_idx in range(4):
    test_ctx = all_tokens + step3_tokens[:11 + byte_idx]
    token_ids = torch.tensor([test_ctx], dtype=torch.long)

    with torch.no_grad():
        logits = model(token_ids)

    predicted = logits[0, -1, :].argmax().item()
    expected = step3_tokens[11 + byte_idx]

    h10 = activations['L10_ffn'][0, -1, :]
    out_lo = h10[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
    out_hi = h10[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
    carry_0 = h10[BD.SP_POP_CARRY_0].item()

    h_embed = activations['L6_attn'][0, -1, :]
    embed_lo = h_embed[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
    embed_hi = h_embed[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()

    match = "✓" if predicted == expected else "✗"
    print(f"Byte {byte_idx}: expected={expected}, predicted={predicted} {match}")
    print(f"         OUT_LO={out_lo}, OUT_HI={out_hi}, carry_0={carry_0:.2f}")
    print(f"         EMBED input: lo={embed_lo}, hi={embed_hi}")
