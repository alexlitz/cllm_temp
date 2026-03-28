import torch
from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.div import build_div_layers
from neural_vm.alu.ops.mod import build_mod_layers

def count_nonzero(layers):
    total = 0
    for layer in layers:
        for p in layer.parameters():
            total += (p != 0).sum().item()
    return total

print("Efficient DIV/MOD Parameter Count:")
print("=" * 50)

div_layers = build_div_layers(NIBBLE, opcode=28)
mod_layers = build_mod_layers(NIBBLE, opcode=29)

div_params = count_nonzero(div_layers)
mod_params = count_nonzero(mod_layers)

print(f"Efficient DIV: {div_params:,} params ({len(div_layers)} layers)")
print(f"Efficient MOD: {mod_params:,} params ({len(mod_layers)} layers)")
print(f"Total:         {div_params + mod_params:,} params")
print()
print(f"Current DivModModule: 1,310,720 params")
print(f"Potential savings:    {1310720 - div_params - mod_params:,} ({100*(1310720 - div_params - mod_params)/1310720:.1f}%)")
