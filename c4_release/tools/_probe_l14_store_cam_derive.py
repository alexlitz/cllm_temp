"""Probe: sanity-check the L14 mem-generation STORE heads now DERIVED via
``cam_binary_address_match`` (``direction="store"``, EMPTY address block) inside
``_layer14_mem_generation_head_specs_with_overrides``.

The store head has NO binary-address comparator (it fires at ONE MEM byte
position via threshold-difference gates), so the derivation uses an EMPTY
CamBinaryAddressBlock; every Q/K position/source/blocker row is a
CamDiscriminatorSlot; the CLEAN_EMBED/OUTPUT/STACK0_BYTE_VAL byte relays are
CamValueBand blocks (the ``v_scale`` field carries the value-head payload
doubling); the byte-0 O cancel + V slot-0 const are a discriminator V/O row.

The AUTHORITATIVE byte-identity proof is
``test_isa_semantics_dsl.py::test_l14_store_generation_derived_is_byte_identical_to_handbuilt``
(CAM-derived == direct build from the same folded maps) + the whole-model golden
hash (``tools/_isa_golden_hash.py`` == 91f55411). This probe just confirms the
derived path RUNS and emits the expected 8-head layout with value bands present.
Run: ``python -m c4_release.tools._probe_l14_store_cam_derive``.
"""
from c4_release.neural_vm.dim_registry_dynamic import (
    build_default_registry_dynamic,
)
from c4_release.neural_vm.unified_compiler.ops.shared import _as_setdim_proxy
from c4_release.neural_vm.unified_compiler.ops.l14_ops import (
    _layer14_mem_generation_head_specs_with_overrides,
)


def _canon(spec):
    return (
        sorted((w.slot, w.dim, w.weight) for w in spec.q),
        sorted((w.slot, w.dim, w.weight) for w in spec.k),
        sorted((w.slot, w.dim, w.weight) for w in spec.v),
        sorted((w.out_dim, w.slot, w.weight) for w in spec.o),
    )


def main() -> int:
    reg = build_default_registry_dynamic()
    dp = {name: int(slot.start) for name, slot in reg.slots.items()}
    BD = _as_setdim_proxy(dp)

    derived = _layer14_mem_generation_head_specs_with_overrides(BD)
    assert len(derived) == 8, len(derived)
    for spec in derived:
        c = _canon(spec)
        q_slots = sorted({s for s, _, _ in c[0]})
        print(f"  head {spec.head_idx}: q={len(c[0])} k={len(c[1])} "
              f"v={len(c[2])} o={len(c[3])} q_slots={q_slots}")
        # Every store head relays the CLEAN_EMBED value-emit band.
        assert int(BD.CLEAN_EMBED_LO) in {w.dim for w in spec.v}, spec.head_idx
    print("DERIVED PATH OK (8 heads, CLEAN_EMBED value-emit band present)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
