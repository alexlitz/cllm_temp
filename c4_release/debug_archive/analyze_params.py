"""Analyze parameter usage by layer/component."""
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

vm = AutoregressiveVM()
set_vm_weights(vm, alu_mode='lookup')

print("Parameter breakdown by block:")
print("=" * 60)

breakdown = {}
for name, param in vm.named_parameters():
    nonzero = (param != 0).sum().item()
    if nonzero > 0:
        parts = name.split('.')
        if parts[0] == 'blocks':
            block_num = int(parts[1])
            component = parts[2]
            if component == 'post_ops':
                key = f"Block {block_num:2d} post_ops"
            else:
                key = f"Block {block_num:2d} {component}"
        else:
            key = parts[0]
        breakdown[key] = breakdown.get(key, 0) + nonzero

# Sort by count descending
sorted_items = sorted(breakdown.items(), key=lambda x: -x[1])
for key, count in sorted_items:
    pct = 100 * count / sum(breakdown.values())
    print(f"  {key:25} {count:>10,} ({pct:5.1f}%)")

total = sum(breakdown.values())
print("-" * 60)
print(f"  {'Total':25} {total:>10,}")
