#!/usr/bin/env python3
"""Check dimension usage."""
import sys
sys.path.insert(0, '.')
from neural_vm.vm_step import _SetDim as BD

# Collect all dimension ranges
dims = []
for name in dir(BD):
    if name.startswith('_'):
        continue
    val = getattr(BD, name)
    if isinstance(val, int) and 0 <= val < 512:
        # Determine size based on comments/patterns
        if 'LO' in name or 'HI' in name:
            size = 16
        elif name in ['TEMP']:
            size = 32
        elif name in ['ADDR_KEY']:
            size = 48
        elif name.startswith('H') and len(name) == 2:
            size = 7  # threshold output dims
        elif name.startswith('L1H') or name.startswith('L2H'):
            size = 7
        else:
            size = 1
        dims.append((val, val + size - 1, name))

# Sort and check for overlaps
dims.sort()
print("Dimension allocations (sorted by start):")
print("-" * 60)
for start, end, name in dims:
    print(f"{start:3d}-{end:3d}: {name}")

print("\n" + "=" * 60)
print("Checking for overlaps with TEMP (480-511):")
temp_start, temp_end = 480, 511
for start, end, name in dims:
    if name == 'TEMP':
        continue
    if start <= temp_end and end >= temp_start:
        overlap_start = max(start, temp_start)
        overlap_end = min(end, temp_end)
        print(f"  {name} ({start}-{end}) overlaps TEMP at {overlap_start}-{overlap_end}")

print("\n" + "=" * 60)
print("Checking for overlaps with AX_FULL_HI (483-498):")
axf_start, axf_end = 483, 498
for start, end, name in dims:
    if name == 'AX_FULL_HI':
        continue
    if start <= axf_end and end >= axf_start:
        overlap_start = max(start, axf_start)
        overlap_end = min(end, axf_end)
        print(f"  {name} ({start}-{end}) overlaps AX_FULL_HI at {overlap_start}-{overlap_end}")
