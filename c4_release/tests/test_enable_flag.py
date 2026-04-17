"""Check if enable_conversational_io flag is being set."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Monkey-patch to check if flag is passed
original_set_vm_weights = None

def patched_set_vm_weights(model, enable_tool_calling=False, enable_conversational_io=False, alu_mode='lookup'):
    print(f"set_vm_weights called with:")
    print(f"  enable_tool_calling={enable_tool_calling}")
    print(f"  enable_conversational_io={enable_conversational_io}")
    print(f"  alu_mode={alu_mode}")
    return original_set_vm_weights(model, enable_tool_calling, enable_conversational_io, alu_mode)

import neural_vm.vm_step
original_set_vm_weights = neural_vm.vm_step.set_vm_weights
neural_vm.vm_step.set_vm_weights = patched_set_vm_weights

from neural_vm.run_vm import AutoregressiveVMRunner

print("Creating runner with conversational_io=True...")
runner = AutoregressiveVMRunner(conversational_io=True)
print("\nRunner created successfully.")
