"""Probe the FINAL V/O relay weights per L14 store head (post-override)."""
from c4_release.neural_vm.dim_registry_dynamic import (
    build_default_registry_dynamic,
)
from c4_release.neural_vm.unified_compiler.ops.shared import _as_setdim_proxy
from c4_release.neural_vm.unified_compiler.ops.l14_ops import (
    _layer14_mem_generation_head_specs_with_overrides,
)


def main():
    reg = build_default_registry_dynamic()
    dp = {name: int(slot.start) for name, slot in reg.slots.items()}
    inv = {v: k for k, v in dp.items()}  # only unique base positions
    BD = _as_setdim_proxy(dp)

    def nm(pos):
        # find the nearest base name <= pos within 63
        best = None
        for name, base in dp.items():
            if base <= pos < base + 64 and (best is None or base > dp[best]):
                best = name
        if best is None:
            return str(pos)
        off = pos - dp[best]
        return f"{best}+{off}" if off else best

    specs = _layer14_mem_generation_head_specs_with_overrides(BD)
    for spec in specs:
        print(f"=== head {spec.head_idx} ===")
        vmap = {}
        for w in spec.v:
            vmap.setdefault(w.slot, []).append((nm(w.dim), w.weight))
        print(" V by slot:")
        for slot in sorted(vmap):
            print(f"   slot {slot}: {sorted(vmap[slot])}")
        omap = {}
        for w in spec.o:
            omap.setdefault(w.slot, []).append((nm(w.out_dim), w.weight))
        print(" O by slot:")
        for slot in sorted(omap):
            print(f"   slot {slot}: {sorted(omap[slot])}")


if __name__ == "__main__":
    main()
