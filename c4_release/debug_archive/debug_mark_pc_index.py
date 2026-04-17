"""Check what index MARK_PC is at."""
from neural_vm.vm_step import _SetDim as BD

print("Dimension indices:")
print(f"MARK_PC: {BD.MARK_PC}")
print(f"HAS_SE: {BD.HAS_SE}")
print(f"OUTPUT_LO: {BD.OUTPUT_LO}")
print(f"OUTPUT_HI: {BD.OUTPUT_HI}")
print(f"EMBED_LO: {BD.EMBED_LO}")
print(f"EMBED_HI: {BD.EMBED_HI}")

# Check a few more
print(f"\nOther dimensions:")
for attr in dir(BD):
    if not attr.startswith('_'):
        val = getattr(BD, attr)
        if isinstance(val, int):
            print(f"{attr}: {val}")
