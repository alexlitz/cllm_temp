"""Test if THINKING markers are being injected correctly."""

import sys
import os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token, _SetDim as BD

print("Creating runner...")
runner = AutoregressiveVMRunner(conversational_io=True)

# Create a test context with various tokens
context = torch.tensor([[
    Token.INIT,
    Token.CODE_START,
    42,  # byte
    Token.REG_PC,
    10,  # byte
    Token.THINKING_START,
    Token.THINKING_END,
    Token.STEP_END,
]], dtype=torch.long)

print("Running embedding...")
with torch.no_grad():
    x = runner.model.embed(context, active_opcode=None)

print("\nChecking MARK_THINKING_START:")
for i in range(context.shape[1]):
    tok = context[0, i].item()
    val = x[0, i, BD.MARK_THINKING_START].item()
    tok_name = Token.__dict__.get([k for k in Token.__dict__ if not k.startswith('_') and Token.__dict__[k] == tok], [str(tok)])[0] if tok in [getattr(Token, k) for k in dir(Token) if not k.startswith('_')] else f"byte_{tok}"
    if val > 0.1:
        print(f"  Position {i}: token {tok:3d} ({tok_name:20s}) MARK_THINKING_START={val:.2f}")

print("\nChecking MARK_THINKING_END:")
for i in range(context.shape[1]):
    tok = context[0, i].item()
    val = x[0, i, BD.MARK_THINKING_END].item()
    tok_name = "?"
    for attr in dir(Token):
        if not attr.startswith('_') and getattr(Token, attr) == tok:
            tok_name = attr
            break
    if tok < 256 and tok_name == "?":
        tok_name = f"byte_{tok}"
    if val > 0.1:
        print(f"  Position {i}: token {tok:3d} ({tok_name:20s}) MARK_THINKING_END={val:.2f}")

# Check that only the right tokens have markers
has_ts_marker = x[0, 5, BD.MARK_THINKING_START].item()  # position 5 is THINKING_START
has_te_marker = x[0, 6, BD.MARK_THINKING_END].item()  # position 6 is THINKING_END
byte_has_marker = x[0, 2, BD.MARK_THINKING_START].item()  # position 2 is byte 42

print(f"\n✓ THINKING_START has marker: {has_ts_marker:.2f} (expected 1.0)")
print(f"✓ THINKING_END has marker: {has_te_marker:.2f} (expected 1.0)")
print(f"✓ Byte 42 has TS marker: {byte_has_marker:.2f} (expected 0.0)")

if has_ts_marker > 0.9 and has_te_marker > 0.9 and byte_has_marker < 0.1:
    print("\n✅ Marker injection is working correctly!")
else:
    print("\n❌ Marker injection is broken")
