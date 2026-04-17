#\!/usr/bin/env python3
"""Check if JSR opcode is being recognized in embeddings."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token, _SetDim as BD
import sys

print("=" * 80)
print("JSR OPCODE RECOGNITION TEST")
print("=" * 80)

runner = AutoregressiveVMRunner()
model = runner.model

# Check what JSR opcode value is
jsr_opcode = Opcode.JSR
print(f"\nJSR opcode value: {jsr_opcode} (0x{jsr_opcode:02x})")

# Check if OP_JSR dimension exists
if hasattr(BD, 'OP_JSR'):
    print(f"OP_JSR dimension: {BD.OP_JSR}")
else:
    print("❌ OP_JSR dimension not found in _SetDim\!")
    sys.exit(1)

# Test: encode JSR opcode token and check embedding
# In C4, opcodes are encoded as CODE tokens (values 0-255 map to some token range)
# Let's check the embedding table directly

print("\nChecking embedding table...")
embed_table = model.embed.embed.weight

# Token for JSR opcode byte
# The bytecode encoding uses: for opcode byte, check context building
bytecode = [Opcode.JSR | (25 << 8)]
context = runner._build_context(bytecode, b"", [], "")

print(f"\nContext for 'JSR 25' program: {context[:15]}")

# Find CODE tokens (should be early in context, after DATA_END)
# Context structure: [STEP_END, code_bytes..., data_bytes..., DATA_END, initial_state...]

# The bytecode bytes should appear in the context
# JSR = 3, immediate = 25, 0, 0 (little endian)
# So we're looking for token 3 in the context

if 3 in context[:10]:
    code_token_pos = context[:10].index(3)
    print(f"\nFound JSR opcode byte (3) at context position {code_token_pos}")
    
    # Check embedding for this token
    device = embed_table.device
    token_tensor = torch.tensor([3], dtype=torch.long).to(device)
    
    with torch.no_grad():
        jsr_embedding = model.embed.embed(token_tensor)[0]  # [d_model]
        
        # Check OP_JSR flag
        op_jsr_value = jsr_embedding[BD.OP_JSR].item()
        print(f"Embedding for token 3 (JSR opcode):")
        print(f"  OP_JSR flag: {op_jsr_value:.4f}")
        
        if op_jsr_value > 3.0:
            print(f"  ✅ OP_JSR flag is set ({op_jsr_value:.4f})")
        else:
            print(f"  ❌ OP_JSR flag NOT set - should be ~5.0-6.0\!")
            print(f"     This means JSR opcode is not being recognized in embeddings")

        # Also check opcode byte encoding
        if hasattr(BD, 'OPCODE_BYTE_LO') and hasattr(BD, 'OPCODE_BYTE_HI'):
            opcode_lo = jsr_embedding[BD.OPCODE_BYTE_LO + 3].item()  # JSR = 3, so lo nibble = 3
            opcode_hi = jsr_embedding[BD.OPCODE_BYTE_HI + 0].item()  # hi nibble = 0
            print(f"  OPCODE_BYTE_LO[3]: {opcode_lo:.4f}")
            print(f"  OPCODE_BYTE_HI[0]: {opcode_hi:.4f}")
            
            if opcode_lo > 0.5 and opcode_hi > 0.5:
                print(f"  ✅ Opcode bytes are set")
            else:
                print(f"  ❌ Opcode bytes NOT set correctly")
else:
    print(f"\n❌ JSR opcode byte (3) not found in context\!")
    print(f"   Context: {context}")

# Also test: what does set_vm_weights do?
print("\n" + "=" * 80)
print("Checking if set_vm_weights was called...")

# The model should have weights set during initialization
# Check if embeddings have been configured
if hasattr(model, 'embed'):
    # Check a few known tokens
    device = embed_table.device
    
    # Token for PC marker
    pc_marker_token = Token.REG_PC
    pc_tensor = torch.tensor([pc_marker_token], dtype=torch.long).to(device)
    
    with torch.no_grad():
        pc_embedding = model.embed.embed(pc_tensor)[0]
        mark_pc_value = pc_embedding[BD.MARK_PC].item() if hasattr(BD, 'MARK_PC') else 0
        
        print(f"\nToken {pc_marker_token} (REG_PC) embedding:")
        print(f"  MARK_PC flag: {mark_pc_value:.4f}")
        
        if mark_pc_value > 0.5:
            print(f"  ✅ set_vm_weights appears to have been called")
        else:
            print(f"  ❌ set_vm_weights may NOT have been called\!")
            print(f"     Embeddings are not configured correctly")

print("\n" + "=" * 80)
