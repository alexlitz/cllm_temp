#!/usr/bin/env python3
"""Debug L10 AX byte passthrough issue."""

import torch
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

from neural_vm.speculative import DraftVM
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.embedding import Opcode

# The MUL test bytecode: IMM 0, PSH, IMM 0, MUL, EXIT
bytecode = [
    Opcode.IMM | (0 << 8),   # IMM 0
    Opcode.PSH | (0 << 8),   # PSH
    Opcode.IMM | (0 << 8),   # IMM 0
    Opcode.MUL | (0 << 8),   # MUL
    Opcode.EXIT | (0 << 8),  # EXIT
]

print("Program decoding:")
print(f"  IMM = {Opcode.IMM}, PSH = {Opcode.PSH}, MUL = {Opcode.MUL}, EXIT = {Opcode.EXIT}")
for i, bc in enumerate(bytecode):
    print(f"  [{i}] 0x{bc:08x}")

# Run with DraftVM to see what it expects
draft = DraftVM(bytecode)

print("\nDraftVM step-by-step:")
for step in range(4):
    draft.step()  # Returns bool, modifies state
    state = draft.draft_tokens()  # Get tokens
    ax_bytes = [(draft.ax >> (8*i)) & 0xFF for i in range(4)]
    print(f"\nStep {step}: PC={draft.pc}, AX={draft.ax:#x}, SP={draft.sp}")
    print(f"  AX bytes: {ax_bytes}")
    print(f"  Tokens: {state[:10]}...")

    if draft.halted:
        print("  HALTED")
        break

# Now build context and trace
print("\n" + "="*60)
print("Building context and tracing predictions")
print("="*60)

# Build context by running DraftVM step by step
draft = DraftVM(bytecode)
context = []

# Add code section
context.append(Token.CODE_START)
for bc in bytecode:
    op = bc & 0xFF
    imm = bc >> 8
    context.append(op)
    # 4 immediate bytes
    for i in range(4):
        context.append((imm >> (i * 8)) & 0xFF)
    # 3 padding bytes
    context.extend([0, 0, 0])
context.append(Token.CODE_END)

# Add data section (empty)
context.append(Token.DATA_START)
context.append(Token.DATA_END)

# Step 0: IMM 0
draft.step()
tokens0 = draft.draft_tokens()
context.extend(tokens0)
print(f"\nAfter step 0 (IMM 0): AX={draft.ax}, context_len={len(context)}")

# Step 1: PSH
draft.step()
tokens1 = draft.draft_tokens()
context.extend(tokens1)
print(f"After step 1 (PSH): AX={draft.ax}, context_len={len(context)}")

# Step 2: IMM 0 - THIS IS WHERE IT FAILS
draft.step()
tokens2 = draft.draft_tokens()
print(f"\nStep 2 (IMM 0) draft tokens: {tokens2}")
print(f"  Expected AX bytes: {[(draft.ax >> (8*i)) & 0xFF for i in range(4)]}")
print(f"  Token 5 (AX marker): {tokens2[5]}")
print(f"  Token 6 (AX byte 0): {tokens2[6]} - logits here predict byte 1")
print(f"  Token 7 (AX byte 1): {tokens2[7]} - logits here predict byte 2")

# Load model and trace
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

print("\nLoading model...")
model = AutoregressiveVM(
    d_model=512,
    n_layers=16,
    n_heads=8,
    ffn_hidden=4096,
    max_seq_len=4096,
)
model.eval()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Build input tensor: context + partial step 2 (up to AX byte 0 position)
# Token 6 is AX byte 0 position, so we need tokens 0-5 of step 2
partial_step2 = tokens2[:6]  # PC marker + 4 PC bytes + AX marker
input_ctx = context + partial_step2
input_ids = torch.tensor([input_ctx], dtype=torch.long, device=device)

print(f"\nInput context length: {len(input_ctx)}")
print(f"Predicting token at position {len(input_ctx) - 1} (AX marker), which should give AX byte 0")

with torch.no_grad():
    # Get activations at each layer
    activations = {}

    def make_hook(name, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations[name] = out[:, -1, :].detach().clone()  # Only last position
        return hook

    hooks = []
    for layer_idx in range(16):
        h = model.blocks[layer_idx].attn.register_forward_hook(make_hook(f'L{layer_idx}_attn', layer_idx))
        hooks.append(h)
        h = model.blocks[layer_idx].ffn.register_forward_hook(make_hook(f'L{layer_idx}_ffn', layer_idx))
        hooks.append(h)

    logits = model.forward(input_ids)

    for h in hooks:
        h.remove()

    # Check prediction
    pred_token = logits[0, -1, :].argmax().item()
    expected_token = tokens2[6]
    print(f"\nPrediction: {pred_token}, Expected: {expected_token}")
    print(f"Match: {pred_token == expected_token}")

    # Check OUTPUT dimensions through layers
    print("\n" + "-"*60)
    print("OUTPUT_LO[0] and OUTPUT_HI[0] through layers:")
    print("-"*60)

    for layer_idx in range(16):
        attn_out = activations[f'L{layer_idx}_attn']
        ffn_out = activations[f'L{layer_idx}_ffn']

        # OUTPUT_LO and OUTPUT_HI at last position
        attn_lo0 = attn_out[0, BD.OUTPUT_LO].item()
        attn_hi0 = attn_out[0, BD.OUTPUT_HI].item()
        ffn_lo0 = ffn_out[0, BD.OUTPUT_LO].item()
        ffn_hi0 = ffn_out[0, BD.OUTPUT_HI].item()

        # Also check OUTPUT_LO[1] since we're predicting byte 1
        attn_lo1 = attn_out[0, BD.OUTPUT_LO + 1].item()
        ffn_lo1 = ffn_out[0, BD.OUTPUT_LO + 1].item()

        if abs(ffn_lo0) > 0.1 or abs(ffn_hi0) > 0.1 or abs(ffn_lo1) > 0.1:
            print(f"L{layer_idx:2d}: attn OUT_LO[0]={attn_lo0:7.2f}, OUT_HI[0]={attn_hi0:7.2f}, OUT_LO[1]={attn_lo1:7.2f}")
            print(f"      ffn  OUT_LO[0]={ffn_lo0:7.2f}, OUT_HI[0]={ffn_hi0:7.2f}, OUT_LO[1]={ffn_lo1:7.2f}")

    # Check what L10 is doing specifically
    print("\n" + "-"*60)
    print("L10 detailed trace:")
    print("-"*60)

    l10_attn = activations['L10_attn']
    l10_ffn = activations['L10_ffn']

    # Check passthrough-related dimensions
    has_se = l10_attn[0, BD.HAS_SE].item()
    h1_ax = l10_attn[0, BD.H1 + 1].item()  # H1[1] = AX area
    is_byte = l10_attn[0, BD.IS_BYTE].item()
    mark_ax = l10_attn[0, BD.MARK_AX].item()

    print(f"At last position (AX marker):")
    print(f"  HAS_SE={has_se:.2f}, H1[AX]={h1_ax:.2f}, IS_BYTE={is_byte:.2f}, MARK_AX={mark_ax:.2f}")

    # Check opcode flags
    op_imm = l10_attn[0, BD.OP_IMM].item()
    op_psh = l10_attn[0, BD.OP_PSH].item()

    print(f"  OP_IMM={op_imm:.2f}, OP_PSH={op_psh:.2f}")

    # Check AX_CARRY dimensions
    ax_carry_lo = [l10_attn[0, BD.AX_CARRY_LO + k].item() for k in range(4)]
    print(f"  AX_CARRY_LO[0:4]={ax_carry_lo}")

# Now trace at AX byte 0 position (token 6)
print("\n" + "="*60)
print("Tracing at AX byte 0 position")
print("="*60)

# Include one more token (AX marker) to get logits at byte 0 position
input_ctx2 = context + tokens2[:7]  # Up to and including AX byte 0
input_ids2 = torch.tensor([input_ctx2], dtype=torch.long, device=device)

print(f"Input context length: {len(input_ctx2)}")
print(f"Last token: {input_ctx2[-1]} (should be {tokens2[6]}, AX byte 0)")

with torch.no_grad():
    activations = {}

    hooks = []
    for layer_idx in range(16):
        h = model.blocks[layer_idx].attn.register_forward_hook(make_hook(f'L{layer_idx}_attn', layer_idx))
        hooks.append(h)
        h = model.blocks[layer_idx].ffn.register_forward_hook(make_hook(f'L{layer_idx}_ffn', layer_idx))
        hooks.append(h)

    logits = model.forward(input_ids2)

    for h in hooks:
        h.remove()

    # Check prediction (should be AX byte 1 = 0)
    pred_token = logits[0, -1, :].argmax().item()
    expected_token = tokens2[7]  # AX byte 1
    print(f"\nPrediction: {pred_token}, Expected: {expected_token}")
    print(f"Match: {pred_token == expected_token}")

    # Check OUTPUT at this position
    print("\n" + "-"*60)
    print("OUTPUT at AX byte 0 position (predicting byte 1):")
    print("-"*60)

    for layer_idx in [8, 9, 10, 11, 12, 15]:
        ffn_out = activations[f'L{layer_idx}_ffn']

        output_lo = [ffn_out[0, BD.OUTPUT_LO + k].item() for k in range(4)]
        output_hi = [ffn_out[0, BD.OUTPUT_HI + k].item() for k in range(4)]

        print(f"L{layer_idx:2d}: OUTPUT_LO={[f'{v:6.2f}' for v in output_lo]}")
        print(f"     OUTPUT_HI={[f'{v:6.2f}' for v in output_hi]}")

    # Check L10 attention contribution
    l10_attn = activations['L10_attn']
    l9_ffn = activations['L9_ffn']  # Input to L10

    # The difference is the attention contribution
    attn_contribution_lo = [l10_attn[0, BD.OUTPUT_LO + k].item() - l9_ffn[0, BD.OUTPUT_LO + k].item() for k in range(4)]
    attn_contribution_hi = [l10_attn[0, BD.OUTPUT_HI + k].item() - l9_ffn[0, BD.OUTPUT_HI + k].item() for k in range(4)]

    print(f"\nL10 attention contribution:")
    print(f"  OUTPUT_LO delta={[f'{v:6.2f}' for v in attn_contribution_lo]}")
    print(f"  OUTPUT_HI delta={[f'{v:6.2f}' for v in attn_contribution_hi]}")

    # Check conditions for L10 byte passthrough
    print("\nL10 passthrough conditions at AX byte 0 position:")
    has_se = l10_attn[0, BD.HAS_SE].item()
    h1_ax = l10_attn[0, BD.H1 + 1].item()
    is_byte = l10_attn[0, BD.IS_BYTE].item()
    byte_idx_0 = l10_attn[0, BD.BYTE_INDEX_0].item()
    byte_idx_3 = l10_attn[0, BD.BYTE_INDEX_3].item()

    print(f"  HAS_SE={has_se:.2f}, H1[AX]={h1_ax:.2f}")
    print(f"  IS_BYTE={is_byte:.2f}, BYTE_INDEX_0={byte_idx_0:.2f}, BYTE_INDEX_3={byte_idx_3:.2f}")

    # Check if passthrough should be suppressed by OP_IMM
    # L10 FFN has passthrough suppression for handled opcodes
    op_imm = l10_attn[0, BD.OP_IMM].item()
    op_psh = l10_attn[0, BD.OP_PSH].item()
    op_add = l10_attn[0, BD.OP_ADD].item() if hasattr(BD, 'OP_ADD') else 0

    print(f"  OP_IMM={op_imm:.2f}, OP_PSH={op_psh:.2f}")

# Trace the opcode fetch pipeline at AX marker (first run, position 119)
print("\n" + "="*60)
print("Opcode Fetch Pipeline at AX marker (step 2)")
print("="*60)

# Re-run with hooks to trace opcode fetch
with torch.no_grad():
    activations = {}
    hooks = []
    for layer_idx in range(16):
        def make_full_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                activations[name] = out[:, -1, :].detach().clone()
            return hook
        h = model.blocks[layer_idx].attn.register_forward_hook(make_full_hook(f'L{layer_idx}_attn'))
        hooks.append(h)
        h = model.blocks[layer_idx].ffn.register_forward_hook(make_full_hook(f'L{layer_idx}_ffn'))
        hooks.append(h)

    logits = model.forward(input_ids)

    for h in hooks:
        h.remove()

    print("\n--- L4: PC relay (PC marker EMBED -> AX marker EMBED) ---")
    l4_attn = activations['L4_attn']
    embed_lo = [l4_attn[0, BD.EMBED_LO + k].item() for k in range(16)]
    embed_hi = [l4_attn[0, BD.EMBED_HI + k].item() for k in range(16)]
    embed_val_lo = sum(k * v for k, v in enumerate(embed_lo)) / max(sum(embed_lo), 1e-6)
    embed_val_hi = sum(k * v for k, v in enumerate(embed_hi)) / max(sum(embed_hi), 1e-6)
    print(f"  EMBED_LO (first 6): {[f'{v:.2f}' for v in embed_lo[:6]]}")
    print(f"  EMBED_HI (first 4): {[f'{v:.2f}' for v in embed_hi[:4]]}")
    print(f"  Weighted avg: lo={embed_val_lo:.2f}, hi={embed_val_hi:.2f}")
    print(f"  Expected PC value: 18 (lo nibble=2, hi nibble=1)")

    print("\n--- L5: Opcode fetch (EMBED -> OPCODE_BYTE) ---")
    l5_attn = activations['L5_attn']
    opcode_lo = [l5_attn[0, BD.OPCODE_BYTE_LO + k].item() for k in range(16)]
    opcode_hi = [l5_attn[0, BD.OPCODE_BYTE_HI + k].item() for k in range(16)]
    print(f"  OPCODE_BYTE_LO (first 4): {[f'{v:.2f}' for v in opcode_lo[:4]]}")
    print(f"  OPCODE_BYTE_HI (first 4): {[f'{v:.2f}' for v in opcode_hi[:4]]}")
    opcode_val_lo = max(range(16), key=lambda k: opcode_lo[k])
    opcode_val_hi = max(range(16), key=lambda k: opcode_hi[k])
    print(f"  Max indices: lo={opcode_val_lo} (strength {opcode_lo[opcode_val_lo]:.2f}), hi={opcode_val_hi} (strength {opcode_hi[opcode_val_hi]:.2f})")
    print(f"  Expected opcode: 1 (IMM), so lo=1, hi=0")

    print("\n--- L5 FFN: Opcode decode (OPCODE_BYTE -> OP_xxx flags) ---")
    l5_ffn = activations['L5_ffn']
    op_imm = l5_ffn[0, BD.OP_IMM].item()
    op_psh = l5_ffn[0, BD.OP_PSH].item()
    op_lea = l5_ffn[0, BD.OP_LEA].item()
    op_add = l5_ffn[0, BD.OP_ADD].item()
    op_mul = l5_ffn[0, BD.OP_MUL].item()
    op_exit = l5_ffn[0, BD.OP_EXIT].item()
    print(f"  OP_LEA: {op_lea:.2f}")
    print(f"  OP_IMM: {op_imm:.2f} (expected ~5.0 for IMM opcode)")
    print(f"  OP_PSH: {op_psh:.2f}")
    print(f"  OP_ADD: {op_add:.2f}")
    print(f"  OP_MUL: {op_mul:.2f}")
    print(f"  OP_EXIT: {op_exit:.2f}")

    # Check MARK_AX at L5 FFN input (should be 1.0 at AX marker)
    l4_ffn = activations['L4_ffn']
    mark_ax = l4_ffn[0, BD.MARK_AX].item()
    print(f"\n  MARK_AX at L5 FFN input: {mark_ax:.2f} (should be 1.0)")
