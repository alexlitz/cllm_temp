"""Test if val heads revert fixes VM."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

print('Testing val heads revert...')
print('=' * 70)

# Create and initialize model
print('\n1. Creating model...')
try:
    model = AutoregressiveVM(n_layers=17)
    print('   ✓ Model created')
except Exception as e:
    print(f'   ✗ Error creating model: {e}')
    sys.exit(1)

print('\n2. Setting weights...')
try:
    set_vm_weights(model)
    print('   ✓ Weights set successfully')
except Exception as e:
    print(f'   ✗ Error setting weights: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n3. Quick forward pass test...')
try:
    # Create minimal input (just CODE_START token)
    from neural_vm.vm_step import Token
    x = torch.tensor([[Token.CODE_START]], dtype=torch.long)
    logits = model(x)
    print(f'   ✓ Forward pass succeeded, output shape: {logits.shape}')
except Exception as e:
    print(f'   ✗ Error in forward pass: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n✓ All checks passed!')
print('=' * 70)
