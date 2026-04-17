from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops import (
    add, sub, mul, div, mod, shift, bitwise, cmp
)

ops = [
    ("ADD", lambda: add.build_add_layers(NIBBLE, 21)),
    ("SUB", lambda: sub.build_sub_layers(NIBBLE, 26)),
    ("MUL", lambda: mul.build_mul_layers(NIBBLE, 27)),
    ("DIV", lambda: div.build_div_layers(NIBBLE, 31)),
    ("MOD", lambda: mod.build_mod_layers(NIBBLE, 32)),
    ("SHL", lambda: shift.build_shl_layers(NIBBLE, 23)),
    ("SHR", lambda: shift.build_shr_layers(NIBBLE, 24)),
    ("AND", lambda: bitwise.build_and_layers(NIBBLE, 30)),
    ("OR", lambda: bitwise.build_or_layers(NIBBLE, 28)),
    ("XOR", lambda: bitwise.build_xor_layers(NIBBLE, 29)),
]

print("="*70)
print("EFFICIENT OPS LAYER STRUCTURE")
print("="*70)

for name, builder in ops:
    layers = builder()
    total_params = sum(sum((p != 0).sum().item() for p in layer.parameters()) 
                      for layer in layers)
    print(f"\n{name}:")
    print(f"  {len(layers)} layer(s), {total_params:,} total params")
    for i, layer in enumerate(layers):
        params = sum((p != 0).sum().item() for p in layer.parameters())
        print(f"    Layer {i}: {params:,} params")

print("\n" + "="*70)
print("SINGLE-LAYER OPS (easiest to integrate):")
single_layer_ops = [name for name, builder in ops if len(builder()) == 1]
for name in single_layer_ops:
    print(f"  - {name}")
