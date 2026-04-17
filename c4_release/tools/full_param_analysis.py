"""Full model parameter analysis."""
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

print("Full Model Parameter Analysis")
print("="*70)

# Efficient mode
print("\n[EFFICIENT MODE]")
vm_eff = AutoregressiveVM()
set_vm_weights(vm_eff, alu_mode='efficient')

eff_total = sum(p.numel() for p in vm_eff.parameters())
eff_nonzero = sum((p != 0).sum().item() for p in vm_eff.parameters())

print(f"  Total parameters:     {eff_total:>12,}")
print(f"  Non-zero parameters:  {eff_nonzero:>12,}")
print(f"  Sparsity:             {100*(1-eff_nonzero/eff_total):>11.2f}%")

# Breakdown by component
eff_breakdown = {}
for name, param in vm_eff.named_parameters():
    nonzero = (param != 0).sum().item()
    if nonzero > 0:
        parts = name.split('.')
        if parts[0] == 'blocks':
            key = f"Block {int(parts[1]):2d} {parts[2]}"
        else:
            key = parts[0]
        eff_breakdown[key] = eff_breakdown.get(key, 0) + nonzero

print("\n  By component:")
for key in sorted(eff_breakdown.keys()):
    print(f"    {key:20} {eff_breakdown[key]:>8,}")

# Lookup mode
print("\n" + "="*70)
print("\n[LOOKUP MODE]")
vm_lookup = AutoregressiveVM()
set_vm_weights(vm_lookup, alu_mode='lookup')

lookup_total = sum(p.numel() for p in vm_lookup.parameters())
lookup_nonzero = sum((p != 0).sum().item() for p in vm_lookup.parameters())

print(f"  Total parameters:     {lookup_total:>12,}")
print(f"  Non-zero parameters:  {lookup_nonzero:>12,}")
print(f"  Sparsity:             {100*(1-lookup_nonzero/lookup_total):>11.2f}%")

# Breakdown by component
lookup_breakdown = {}
for name, param in vm_lookup.named_parameters():
    nonzero = (param != 0).sum().item()
    if nonzero > 0:
        parts = name.split('.')
        if parts[0] == 'blocks':
            key = f"Block {int(parts[1]):2d} {parts[2]}"
        else:
            key = parts[0]
        lookup_breakdown[key] = lookup_breakdown.get(key, 0) + nonzero

print("\n  By component:")
for key in sorted(lookup_breakdown.keys()):
    print(f"    {key:20} {lookup_breakdown[key]:>8,}")

# Summary
print("\n" + "="*70)
print("\n[SUMMARY]")
print(f"  Efficient mode:  {eff_nonzero:>12,} non-zero params")
print(f"  Lookup mode:     {lookup_nonzero:>12,} non-zero params")
print(f"  Reduction:       {lookup_nonzero - eff_nonzero:>12,} params ({100*(lookup_nonzero-eff_nonzero)/lookup_nonzero:.1f}%)")
