"""Debug which layer writes OUTPUT at byte positions."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode

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

# Setup
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)
# Add REG_PC and PC_b0 tokens
context.extend([Token.REG_PC, 10])

print(f"Context: {context}")
print(f"Context ends with: REG_PC(257), PC_b0(10)")
print(f"Predicting PC_b1 (should be 0)")
print()

ctx_tensor = torch.tensor([context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    last_pos = len(context) - 1  # Position of token 10 (PC_b0)

    print(f"=== TRACKING OUTPUT AT POSITION {last_pos} (token 10) ===")

    # Run through layers
    for i, block in enumerate(model.blocks):
        x_before = x.clone()
        x = block(x)

        # Check if OUTPUT changed
        lo_before = [x_before[0, last_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        hi_before = [x_before[0, last_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
        lo_after = [x[0, last_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
        hi_after = [x[0, last_pos, BD.OUTPUT_HI + k].item() for k in range(16)]

        if lo_before != lo_after or hi_before != hi_after:
            lo_max = max(range(16), key=lambda k: lo_after[k])
            hi_max = max(range(16), key=lambda k: hi_after[k])
            val = lo_max | (hi_max << 4)
            print(f"Layer {i}: OUTPUT changed, now encodes {val}")
            print(f"  OUTPUT_LO max at [{lo_max}] = {lo_after[lo_max]:.3f}")
            print(f"  OUTPUT_HI max at [{hi_max}] = {hi_after[hi_max]:.3f}")

    # Final prediction
    logits = model.head(x)
    predicted = logits[0, last_pos, :].argmax().item()
    print(f"\nFinal prediction: {predicted} (expected: 0)")
