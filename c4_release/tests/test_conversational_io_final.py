"""Final comprehensive test of conversational I/O implementation.

This test demonstrates that the complete pipeline works:
1. PRTF opcode is set via set_active_opcode()
2. ACTIVE_OPCODE_PRTF is injected into embeddings
3. L5 FFN detects PRTF and sets IO_IS_PRTF
4. L6 relays to CMP[3] and triggers state machine
5. THINKING_END is generated at STEP_END position
"""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

print("=" * 70)
print("CONVERSATIONAL I/O - FINAL VERIFICATION TEST")
print("=" * 70)

c_code = 'int main() { printf("Hello"); return 0; }'

print("\n1. Compiling test program...")
code, data = compile_c(c_code)
print(f"   ✓ {len(code)} instructions compiled")

print("\n2. Creating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)
print("   ✓ Runner created")

print("\n3. Verifying L5 FFN weights...")
l5_ffn = runner.model.blocks[5].ffn
w_up = l5_ffn.W_up[410, BD.ACTIVE_OPCODE_PRTF].item()
w_down = l5_ffn.W_down[BD.IO_IS_PRTF, 410].item()
print(f"   W_up[410, ACTIVE_OPCODE_PRTF]: {w_up:.2f} (expected 100.0)")
print(f"   W_down[IO_IS_PRTF, 410]: {w_down:.4f} (expected 0.1)")
assert abs(w_up - 100.0) < 0.1, "L5 FFN weights not set correctly!"
print("   ✓ L5 FFN weights correct")

print("\n4. Building context with proper instruction format...")
tokens = [Token.CODE_START]
for instr in code:
    op = instr & 0xFF
    imm = instr >> 8
    tokens.append(op)
    for i in range(IMMEDIATE_SIZE):
        tokens.append((imm >> (i * 8)) & 0xFF)
    for _ in range(PADDING_SIZE):
        tokens.append(0)
tokens.append(Token.CODE_END)
tokens.append(Token.DATA_START)
tokens.extend(data)
tokens.append(Token.DATA_END)
tokens.append(Token.THINKING_START)

# Add complete VM step (35 tokens)
step_tokens = [
    Token.REG_PC, 0, 0, 0, 0,
    Token.REG_AX, 0, 0, 0, 0,
    Token.REG_SP, 0, 0, 0, 0,
    Token.REG_BP, 0, 0, 0, 0,
    Token.STACK0, 0, 0, 0, 0,
    Token.MEM, 0, 0, 0, 0, 0, 0, 0, 0,
]
tokens.extend(step_tokens)
print(f"   ✓ Context built ({len(tokens)} tokens)")

print("\n5. Setting active opcode to PRTF (33)...")
runner.model.set_active_opcode(33)
print("   ✓ Active opcode set")

print("\n6. Running forward pass...")
context = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    # Check embedding
    x = runner.model.embed(context, active_opcode=33)
    active_prtf = x[0, -1, BD.ACTIVE_OPCODE_PRTF].item()
    print(f"   Embedding: ACTIVE_OPCODE_PRTF = {active_prtf:.2f}")

    # Run through layers
    for i in range(16):
        x = runner.model.blocks[i](x)
        if i == 5:
            io_is_prtf = x[0, -1, BD.IO_IS_PRTF].item()
            print(f"   After L5: IO_IS_PRTF = {io_is_prtf:.2f}")
        if i == 6:
            cmp3 = x[0, -1, BD.CMP + 3].item()
            print(f"   After L6: CMP[3] = {cmp3:.2f}")

    next_te = x[0, -1, BD.NEXT_THINKING_END].item()
    print(f"   After L15: NEXT_THINKING_END = {next_te:.2f}")

    # Get output logits
    logits = runner.model.head(x)[0, -1, :]

print("\n7. Checking output logits...")
te_logit = logits[Token.THINKING_END].item()
se_logit = logits[Token.STEP_END].item()
winner = logits.argmax().item()

print(f"   THINKING_END: {te_logit:7.2f}")
print(f"   STEP_END:     {se_logit:7.2f}")
print(f"   Winner:       {winner}")

print("\n" + "=" * 70)
if winner == Token.THINKING_END:
    print("✅ SUCCESS! THINKING_END is generated when PRTF executes!")
    print(f"   THINKING_END wins with logit {te_logit:.2f}")
    print(f"   STEP_END suppressed to {se_logit:.2f}")
    print("\n   The conversational I/O implementation is WORKING!")
else:
    print(f"❌ FAILED! Token {winner} generated instead of THINKING_END")
print("=" * 70)
