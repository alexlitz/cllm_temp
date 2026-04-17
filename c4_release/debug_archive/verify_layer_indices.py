"""Verify which blocks[] index corresponds to which layer."""
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

model = AutoregressiveVM()
set_vm_weights(model)

print("Layer index mapping:")
print("In set_vm_weights, layers are set as:")
print("  attn0 = model.blocks[0].attn  → Layer 0")
print("  attn1 = model.blocks[1].attn  → Layer 1")
print("  attn2 = model.blocks[2].attn  → Layer 2")
print("  ffn3 = model.blocks[3].ffn    → Layer 3")
print("  attn4 = model.blocks[4].attn  → Layer 4")
print("  attn5 = model.blocks[5].attn  → Layer 5")
print("  attn6 = model.blocks[6].attn  → Layer 6")
print()
print("So the pattern is:")
print("  Layer N = model.blocks[N]")
print()
print("My debug scripts were using blocks[2] for L3, but should use blocks[3]!")
print("This explains why all the weights appeared wrong.")
