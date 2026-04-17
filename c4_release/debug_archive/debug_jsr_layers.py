#!/usr/bin/env python3
"""Debug JSR execution by tracing intermediate layer activations."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token, _SetDim as BD
import sys

print("=" * 80)
print("JSR LAYER-BY-LAYER DIAGNOSTIC")
print("=" * 80)

# Simple JSR test
bytecode = [
    Opcode.JSR | (25 << 8),  # 0: JSR to byte 25 (instr 5)
    Opcode.EXIT,              # 1: EXIT
    Opcode.NOP,               # 2-4: Padding
    Opcode.NOP,
    Opcode.NOP,
    Opcode.IMM | (42 << 8),   # 5: IMM 42 (at byte 25)
    Opcode.EXIT,              # 6: EXIT
]

print("\nBytecode:")
print("  Instr 0 (byte 0):  JSR 25  <- Start here")
print("  Instr 1 (byte 5):  EXIT")
print("  Instr 2-4:         NOP (padding)")
print("  Instr 5 (byte 25): IMM 42  <- JSR target")
print("  Instr 6 (byte 30): EXIT")

print("\nExpected: JSR should jump to byte 25, execute IMM 42, exit with code 42")

runner = AutoregressiveVMRunner()
model = runner.model

# Hook to capture layer outputs
layer_outputs = {}
original_forward = model.forward

def forward_with_capture(token_ids, kv_cache=None):
    """Forward pass that captures intermediate layer outputs."""
    # Clear previous captures
    layer_outputs.clear()
    
    # Run original forward
    result = original_forward(token_ids, kv_cache)
    
    # Capture happens inside the model - we'll add hooks
    return result

# Add hooks to capture specific layer outputs
def make_hook(layer_name):
    def hook(module, input, output):
        layer_outputs[layer_name] = output.detach().clone()
    return hook

# Register hooks on key layers
if hasattr(model, 'layers'):
    if len(model.layers) > 5:
        model.layers[5].register_forward_hook(make_hook('L5_output'))  # After L5 (has TEMP[0])
    if len(model.layers) > 6:
        model.layers[6].register_forward_hook(make_hook('L6_output'))  # After L6 (PC override)

# Build context
context = runner._build_context(bytecode, b"", [], "")
print(f"\nContext length: {len(context)} tokens")

# Generate first step (35 tokens)
print("\nGenerating first VM step (35 tokens)...")
print("This step should execute JSR and set PC to 25\n")

for i in range(35):
    next_token = model.generate_next(context[-512:] if len(context) > 512 else context)
    context.append(next_token)
    
    if i == 0:
        print(f"Token {i+1}: {next_token} (PC marker)")
    elif i < 5:
        print(f"Token {i+1}: {next_token} (PC byte {i-1})")

# Extract PC value from step
pc_bytes = context[-35+1:-35+5]  # Tokens 1-4 of last step
pc = sum((b & 0xFF) << (i * 8) for i, b in enumerate(pc_bytes))

print(f"\nResult after step 1:")
print(f"  PC = 0x{pc:08x} (byte {pc})")

if pc == 25:
    print("  ✅ SUCCESS! PC jumped to byte 25 (JSR worked!)")
elif pc == 10:
    print("  ❌ FAILED: PC = 10 (byte 10 = instr 2 = normal PC+5)")
    print("     JSR did NOT override PC - neural implementation not working")
elif pc == 5:
    print("  ❌ FAILED: PC = 5 (byte 5 = instr 1)")
    print("     PC advanced by 1 instruction instead of jumping")
else:
    print(f"  ❌ FAILED: PC = {pc} (unexpected value)")

# Now let's check if we have layer outputs to analyze
if 'L5_output' in layer_outputs:
    print("\n" + "=" * 80)
    print("LAYER 5 OUTPUT ANALYSIS")
    print("=" * 80)
    
    # L5 output shape: [1, seq_len, d_model]
    l5_out = layer_outputs['L5_output']
    
    # Find PC marker token in last step
    # Last step starts at position -35
    pc_marker_pos = len(context) - 35
    
    if pc_marker_pos >= 0 and pc_marker_pos < l5_out.shape[1]:
        pc_marker_activations = l5_out[0, pc_marker_pos, :]
        
        # Check TEMP[0] (should be ~5.0 for JSR on first step)
        temp0_value = pc_marker_activations[BD.TEMP + 0].item()
        print(f"\nTEMP[0] at PC marker: {temp0_value:.4f}")
        print(f"  Expected: ~5.0 (from L5 FFN JSR decode)")
        
        if abs(temp0_value - 5.0) < 1.0:
            print(f"  ✅ TEMP[0] looks correct ({temp0_value:.4f})")
        elif abs(temp0_value) < 0.1:
            print(f"  ❌ TEMP[0] is nearly zero - L5 JSR decode not firing!")
        else:
            print(f"  ⚠️  TEMP[0] = {temp0_value:.4f} (unexpected value)")
        
        # Check FETCH values (should contain jump target = 25)
        fetch_lo_nibbles = [pc_marker_activations[BD.FETCH_LO + k].item() for k in range(16)]
        fetch_hi_nibbles = [pc_marker_activations[BD.FETCH_HI + k].item() for k in range(16)]
        
        # Decode FETCH value
        fetch_val = 0
        for k in range(16):
            if fetch_lo_nibbles[k] > 0.5:
                fetch_val |= (k << 0)
            if fetch_hi_nibbles[k] > 0.5:
                fetch_val |= (k << 4)
        
        print(f"\nFETCH at PC marker: 0x{fetch_val:02x} ({fetch_val})")
        print(f"  Expected: 0x19 (25)")
        
        if fetch_val == 25:
            print(f"  ✅ FETCH = 25 (correct jump target)")
        else:
            print(f"  ❌ FETCH = {fetch_val} (wrong!)")

if 'L6_output' in layer_outputs:
    print("\n" + "=" * 80)
    print("LAYER 6 OUTPUT ANALYSIS")
    print("=" * 80)
    
    l6_out = layer_outputs['L6_output']
    pc_marker_pos = len(context) - 35
    
    if pc_marker_pos >= 0 and pc_marker_pos < l6_out.shape[1]:
        pc_marker_activations = l6_out[0, pc_marker_pos, :]
        
        # Check OUTPUT_LO/HI (should be FETCH if JSR fired, PC+5 otherwise)
        output_lo_nibbles = [pc_marker_activations[BD.OUTPUT_LO + k].item() for k in range(16)]
        output_hi_nibbles = [pc_marker_activations[BD.OUTPUT_HI + k].item() for k in range(16)]
        
        # Decode OUTPUT value
        output_val = 0
        for k in range(16):
            if output_lo_nibbles[k] > 0.5:
                output_val |= (k << 0)
            if output_hi_nibbles[k] > 0.5:
                output_val |= (k << 4)
        
        print(f"\nOUTPUT at PC marker: 0x{output_val:02x} ({output_val})")
        print(f"  If JSR worked: should be 25 (FETCH)")
        print(f"  If JSR failed: should be 10 (PC+5 from instruction 0)")
        
        if output_val == 25:
            print(f"  ✅ OUTPUT = 25 (JSR PC override worked!)")
        elif output_val == 10:
            print(f"  ❌ OUTPUT = 10 (JSR PC override NOT working - default PC+5)")
        else:
            print(f"  ⚠️  OUTPUT = {output_val} (unexpected value)")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

sys.exit(0 if pc == 25 else 1)
