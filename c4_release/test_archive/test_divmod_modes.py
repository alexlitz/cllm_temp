"""Compare DivModModule lookup vs efficient modes."""
from neural_vm.vm_step import DivModModule

for mode in ['lookup', 'efficient']:
    dm = DivModModule(mode=mode)
    total = sum(p.numel() for p in dm.parameters())
    nonzero = sum((p != 0).sum().item() for p in dm.parameters())
    print(f"{mode:10}: {nonzero:,} non-zero params (of {total:,} total)")
