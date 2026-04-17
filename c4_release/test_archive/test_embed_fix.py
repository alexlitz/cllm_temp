#!/usr/bin/env python3
"""Quick test to verify EMBED is written in L3 FFN first-step defaults."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import _SetDim as BD
import inspect

print("Creating VM runner to check L3 FFN weights...")

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

# Check L3 FFN weights for EMBED writes
model = runner.model
l3_ffn = model.blocks[3].ffn

print("\nChecking L3 FFN W_down for EMBED writes...")
print(f"W_down shape: {l3_ffn.W_down.shape}")

# First-step PC should be PC_OFFSET + INSTR_WIDTH = 2 + 8 = 10 = 0x0A
# Low nibble: 0xA = 10
# High nibble: 0x0 = 0

pc_lo = 10  # 0xA
pc_hi = 0   # 0x0

# Check if there are non-zero weights writing to EMBED_LO + 10 and EMBED_HI + 0
embed_lo_connections = []
embed_hi_connections = []

for unit in range(l3_ffn.W_down.shape[1]):
    weight_lo = l3_ffn.W_down[BD.EMBED_LO + pc_lo, unit].item()
    weight_hi = l3_ffn.W_down[BD.EMBED_HI + pc_hi, unit].item()

    if abs(weight_lo) > 0.01:
        embed_lo_connections.append((unit, weight_lo))
    if abs(weight_hi) > 0.01:
        embed_hi_connections.append((unit, weight_hi))

print(f"\nEMBED_LO[{pc_lo}] (nibble A) connections:")
if embed_lo_connections:
    for unit, weight in embed_lo_connections[:10]:
        print(f"  Unit {unit}: {weight:.4f}")
    print(f"  Total: {len(embed_lo_connections)} connections")
else:
    print("  ❌ NO CONNECTIONS FOUND - Fix not applied!")

print(f"\nEMBED_HI[{pc_hi}] (nibble 0) connections:")
if embed_hi_connections:
    for unit, weight in embed_hi_connections[:10]:
        print(f"  Unit {unit}: {weight:.4f}")
    print(f"  Total: {len(embed_hi_connections)} connections")
else:
    print("  ❌ NO CONNECTIONS FOUND - Fix not applied!")

if embed_lo_connections and embed_hi_connections:
    print("\n✅ SUCCESS: EMBED writes are present in L3 FFN!")
    print("The fix has been applied correctly.")
else:
    print("\n❌ FAILURE: EMBED writes are missing!")
    print("The fix may not have been applied or weights need to be regenerated.")
