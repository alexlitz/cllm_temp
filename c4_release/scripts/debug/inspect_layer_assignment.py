#!/usr/bin/env python3
from neural_vm.unified_compiler.full_vm_compiler import derive_layout

layout = derive_layout()
print(f"d_model={layout.d_model}, n_layers={layout.n_layers}")
for layer_idx, ops in enumerate(layout.ops_per_layer):
    print(f"\nLayer {layer_idx}:")
    for op in ops:
        print(f"  - {op.name} (kind={op.kind}, migrated={op.migrated}, phase={op.phase})")
