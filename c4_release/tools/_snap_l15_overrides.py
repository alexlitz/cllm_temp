"""Spec-level byte-identity snapshot of the L15 heads-0-3 override layer.

Captures the exact per-head canonical (slot, dim, weight) tuple set produced by
``_layer15_memory_lookup_heads_0_3_specs_with_overrides`` across EVERY combination
of the boolean predicates the override branches on, so a lift/refactor of the
override code can be proven byte-identical WITHOUT a full model bake. The override
function IS the lowered head (its final Q/K/V/O maps map 1:1 to the lowered
W_q/W_k/W_v/W_o), so equal canonical tuple sets == equal weights.

We monkeypatch each boolean predicate directly (rather than driving env vars,
whose defaults chain through campaign flags) so the enumeration is exhaustive and
env-independent.

Usage::

    CUDA_VISIBLE_DEVICES="" python tools/_snap_l15_overrides.py > /tmp/l15_before.txt
    # ... make the lift ...
    CUDA_VISIBLE_DEVICES="" python tools/_snap_l15_overrides.py > /tmp/l15_after.txt
    diff /tmp/l15_before.txt /tmp/l15_after.txt   # must be empty
"""

import hashlib
import itertools
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# The boolean predicates the override layer branches on. Each is a module-level
# function we can monkeypatch to a constant. (var_three_li / cmp_veto live in
# shared.py; l15_ops imports them by name.)
_PREDICATES = [
    ("l15_ops", "_l15_li_load_suppressor_inert_on"),
    ("l15_ops", "l15_lookup_cmp_veto_enabled"),
    ("l15_ops", "var_three_li_enabled"),
    ("l15_ops", "_l15_li_addr_cam_discriminator_on"),
    ("l15_ops", "_l15_li_zeroaddr_cam_on"),
    ("l15_ops", "_l15_li_byte0_valsel_on"),
    ("l15_ops", "_l15_li_valrow_b1_on"),
    ("l15_ops", "_l15_li_jsr_phantom_penalty_on"),
    ("l15_ops", "_l15_sclc_byte0_on"),
]


def _canon(spec):
    return (
        tuple(sorted((w.slot, w.dim, round(w.weight, 6)) for w in spec.q)),
        tuple(sorted((w.slot, w.dim, round(w.weight, 6)) for w in spec.k)),
        tuple(sorted((w.slot, w.dim, round(w.weight, 6)) for w in spec.v)),
        tuple(sorted((w.out_dim, w.slot, round(w.weight, 6)) for w in spec.o)),
    )


def _snapshot_one(l15, BD):
    specs = l15._layer15_memory_lookup_heads_0_3_specs_with_overrides(BD)
    lines = []
    for spec in specs:
        h = hashlib.sha256()
        for group in _canon(spec):
            for tup in group:
                h.update(repr(tup).encode("utf-8"))
        lines.append(f"head={spec.head_idx} "
                     f"n_q={len(spec.q)} n_k={len(spec.k)} n_v={len(spec.v)} "
                     f"n_o={len(spec.o)} sha={h.hexdigest()}")
    return lines


def main():
    from c4_release.neural_vm.dim_registry_dynamic import (
        build_default_registry_dynamic,
    )
    from c4_release.neural_vm.unified_compiler.ops.shared import _as_setdim_proxy
    from c4_release.neural_vm.unified_compiler.ops import l14_ops  # noqa: F401
    from c4_release.neural_vm.unified_compiler.ops import l15_ops as l15
    from c4_release.neural_vm.unified_compiler.ops.residual_band_registry import (
        collect_registered_residual_bands,
    )

    reg = build_default_registry_dynamic()
    dp = {name: int(slot.start) for name, slot in reg.slots.items()}
    # Campaign-only residual bands (LI_ZEROADDR_COMMITTED etc.) are not in the
    # default registry; give them synthetic, STABLE positions past the registry
    # max so the flag-ON branches resolve. Positions are consistent before/after,
    # which is all the byte-identity diff needs.
    nxt = max(dp.values()) + 1
    for band, width in sorted(collect_registered_residual_bands().items()):
        if band not in dp:
            dp[band] = nxt
            nxt += max(1, int(width))
    BD = _as_setdim_proxy(dp)

    names = [n for (_m, n) in _PREDICATES]
    originals = {n: getattr(l15, n) for n in names}

    # Exhaustive: 2**9 = 512 combinations. head 0 branches on all 9; heads 1-3
    # branch on none of the campaign-only ones, so their output is invariant to
    # every predicate except the base -- but snapshotting all is cheap+safe.
    n = len(names)
    for bits in range(1 << n):
        state = {names[i]: bool((bits >> i) & 1) for i in range(n)}
        for nm, val in state.items():
            setattr(l15, nm, (lambda v=val: v))
        lines = _snapshot_one(l15, BD)
        key = "".join("1" if state[nm] else "0" for nm in names)
        print(f"### STATE {key}")
        for ln in lines:
            print(ln)
        for nm in names:
            setattr(l15, nm, originals[nm])


if __name__ == "__main__":
    main()
