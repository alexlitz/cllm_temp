"""Inspect activations when PRTF executes."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

c_code = '''
int main() {
    printf("Hi");
    return 0;
}
'''

print("Compiling...")
code, data = compile_c(c_code)

# Find PRTF instruction
for i, inst in enumerate(code):
    if inst['op'] == 0x21:  # PRTF
        prtf_step = i
        print(f"PRTF at instruction {i}, step {i+1}")
        break

print("\nRunning...")
runner = AutoregressiveVMRunner(conversational_io=True)
if torch.cuda.is_available():
    runner.model = runner.model.cuda()

# Build prefix tokens (system prompt)
prefix = [Token.INIT]
prefix.append(Token.CODE_START)
for inst in code:
    prefix.append(inst['op'])
    for imm_byte in inst['imm']:
        prefix.append(imm_byte)
prefix.append(Token.CODE_END)
prefix.append(Token.DATA_START)
for b in data:
    prefix.append(b)
prefix.append(Token.DATA_END)
prefix.append(Token.THINKING_START)

context = torch.tensor([prefix], dtype=torch.long, device=next(runner.model.parameters()).device)

# Run until PRTF step
target_tokens = len(prefix) + prtf_step * 35
print(f"Running to token {target_tokens} (PRTF step {prtf_step+1})...")

for step in range(prtf_step + 1):
    # Set active opcode for this step
    if step < len(code):
        runner.model.set_active_opcode(code[step]['op'])

    # Generate 35 tokens
    for i in range(35):
        token = runner.model.generate_next(context)
        context = torch.cat([context, torch.tensor([[token]], device=context.device)], dim=1)

print(f"\nContext length: {context.shape[1]}")
print(f"Active opcode: {runner.model._active_opcode}")

# Now inspect the activations
print("\nInspecting embeddings at last position...")
token_ids = context
with torch.no_grad():
    x = runner.model.embed(token_ids, active_opcode=runner.model._active_opcode)

    # Check ACTIVE_OPCODE_PRTF at all positions
    active_prtf = x[0, :, BD.ACTIVE_OPCODE_PRTF]
    nonzero_pos = torch.where(active_prtf > 0.5)[0]
    print(f"ACTIVE_OPCODE_PRTF=1.0 at {len(nonzero_pos)} positions")
    if len(nonzero_pos) > 0:
        print(f"  First 10 positions: {nonzero_pos[:10].tolist()}")
    else:
        print("  ❌ ACTIVE_OPCODE_PRTF is NOT being set!")

    # Run full forward pass
    logits = runner.model.forward(token_ids)
    last_logits = logits[0, -1, :]

    # Check top predictions
    top5 = torch.topk(last_logits, 5)
    print(f"\nTop 5 predictions at position {context.shape[1]-1}:")
    for val, idx in zip(top5.values, top5.indices):
        tok_name = Token.name(idx.item()) if hasattr(Token, 'name') else f"tok_{idx.item()}"
        print(f"  {idx.item():3d} ({tok_name:20s}): {val.item():7.2f}")

    # Check specific tokens
    thinking_end_logit = last_logits[Token.THINKING_END].item()
    reg_pc_logit = last_logits[Token.REG_PC].item()
    step_end_logit = last_logits[Token.STEP_END].item()

    print(f"\nSpecific logits:")
    print(f"  THINKING_END: {thinking_end_logit:7.2f}")
    print(f"  REG_PC:       {reg_pc_logit:7.2f}")
    print(f"  STEP_END:     {step_end_logit:7.2f}")
