"""Probe: DERIVE->PROVE that L13 head 4 (SUB minuend relay) + head 5 (ADD addend
relay) produced by ``cam_lookup`` (``direction="store"``, the sole live path
after the flip) match a fresh hand-reconstruction of the DELETED legacy Q/K/V/O
writes, byte-for-byte, in BOTH the DEFAULT (campaign, operand_from_memsp ON) and
the flag-OFF (``C4_OPERAND_FROM_MEMSP=0``) config.

This is the byte-identity proof behind the golden-hash-neutral flip (mirror of
``_probe_l13_relay_cam_derive.py`` for the load-direction heads 3/6).

    CUDA_VISIBLE_DEVICES="" python tools/_probe_l13_store_cam_derive.py
    CUDA_VISIBLE_DEVICES="" C4_OPERAND_FROM_MEMSP=0 python tools/_probe_l13_store_cam_derive.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from c4_release.neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from c4_release.neural_vm.unified_compiler.ops.shared import (
    _as_setdim_proxy,
    operand_from_memsp_enabled,
)
from c4_release.neural_vm.unified_compiler.ops import l13_ops
from c4_release.neural_vm.unified_compiler.primitives import (
    AO, AP, DeclarativeAttentionHeadSpec,
)


def _canon(spec):
    return (
        sorted((w.slot, w.dim, w.weight) for w in spec.q),
        sorted((w.slot, w.dim, w.weight) for w in spec.k),
        sorted((w.slot, w.dim, w.weight) for w in spec.v),
        sorted((w.out_dim, w.slot, w.weight) for w in spec.o),
    )


L = 15.0
K_FLAG = 200.0
OP_W = 40.0


def _legacy_sub_minuend(BD, campaign):
    """The DELETED hand-authored L13 head 4 writes, verbatim (pre-flip)."""
    if campaign:
        q = [AP(0, BD.BYTE_INDEX_0, L), AP(0, BD.HAS_SE, L),
             AP(0, BD.CONST, -L / 2), AP(0, BD.MARK_AX, -L * 10),
             AP(0, BD.MARK_PC, -L * 10), AP(0, BD.BYTE_INDEX_1, -L * 10),
             AP(0, BD.BYTE_INDEX_2, -L * 10), AP(0, BD.BYTE_INDEX_3, -L * 10),
             AP(33, BD.BYTE_INDEX_0, L), AP(33, BD.CONST, -L / 2)]
        k = [AP(33, BD.CONST, L)]
        v, o = [], []
        sel = 1
        q += [AP(sel, BD.BYTE_INDEX_0, L), AP(sel, BD.HAS_SE, L),
              AP(sel, BD.CONST, -L)]
        k += [AP(sel, BD.MARK_AX, K_FLAG), AP(sel, BD.OP_SUB, OP_W),
              AP(sel, BD.OP_ADD, -K_FLAG), AP(sel, BD.CONST, -K_FLAG * 1.3)]
        base = 3
        vlo, vhi = BD.STACK0_BYTE_VAL_1_LO, BD.STACK0_BYTE_VAL_1_HI
        for kk in range(16):
            v += [AP(base + kk, vlo + kk, 1.0), AP(base + 16 + kk, vhi + kk, 1.0)]
            o += [AO(vlo + kk, base + kk, 1.0), AO(vhi + kk, base + 16 + kk, 1.0)]
        v.append(AP(base + 32, BD.OP_SUB, 0.2))
        o.append(AO(BD.TEMP + 9, base + 32, 1.0))
        return DeclarativeAttentionHeadSpec(
            head_idx=4, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o))
    _routes = (
        (0, BD.BYTE_INDEX_0, BD.STACK0_BYTE1,
         BD.STACK0_BYTE_VAL_1_LO, BD.STACK0_BYTE_VAL_1_HI),
        (1, BD.BYTE_INDEX_1, BD.STACK0_BYTE2,
         BD.STACK0_BYTE_VAL_2_LO, BD.STACK0_BYTE_VAL_2_HI),
        (2, BD.BYTE_INDEX_2, BD.STACK0_BYTE3,
         BD.STACK0_BYTE_VAL_3_LO, BD.STACK0_BYTE_VAL_3_HI),
    )
    q = [AP(0, BD.TEMP + 9, L), AP(0, BD.CONST, -L / 2),
         AP(0, BD.MARK_AX, -L * 10), AP(0, BD.MARK_PC, -L * 10),
         AP(0, BD.TEMP + 8, -L * 10), AP(0, BD.BYTE_INDEX_3, -L * 10),
         AP(33, BD.TEMP + 9, L), AP(33, BD.CONST, -L / 2)]
    k = [AP(33, BD.CONST, L)]
    v, o = [], []
    for j, (h, emit, src, vlo, vhi) in enumerate(_routes):
        sel = 1 + j
        q += [AP(sel, emit, L), AP(sel, BD.TEMP + 9, L), AP(sel, BD.CONST, -L)]
        k += [AP(sel, src, K_FLAG), AP(sel, BD.CONST, -K_FLAG / 2)]
        base = 3 + j * 32
        for kk in range(16):
            v += [AP(base + kk, vlo + kk, 1.0), AP(base + 16 + kk, vhi + kk, 1.0)]
            o += [AO(vlo + kk, base + kk, 1.0), AO(vhi + kk, base + 16 + kk, 1.0)]
    return DeclarativeAttentionHeadSpec(
        head_idx=4, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o))


def _legacy_add_addend(BD, campaign):
    """The DELETED hand-authored L13 head 5 writes, verbatim (pre-flip)."""
    emit = BD.BYTE_INDEX_0
    vlo, vhi = BD.STACK0_BYTE_VAL_1_LO, BD.STACK0_BYTE_VAL_1_HI
    if campaign:
        q = [AP(0, emit, L), AP(0, BD.HAS_SE, L), AP(0, BD.CONST, -L / 2),
             AP(0, BD.MARK_AX, -L * 10), AP(0, BD.MARK_PC, -L * 10),
             AP(0, BD.BYTE_INDEX_1, -L * 10), AP(0, BD.BYTE_INDEX_2, -L * 10),
             AP(0, BD.BYTE_INDEX_3, -L * 10), AP(33, emit, L),
             AP(33, BD.CONST, -L / 2)]
        k = [AP(33, BD.CONST, L)]
        v, o = [], []
        sel = 1
        q += [AP(sel, emit, L), AP(sel, BD.HAS_SE, L), AP(sel, BD.CONST, -L)]
        k += [AP(sel, BD.MARK_AX, K_FLAG), AP(sel, BD.OP_ADD, OP_W),
              AP(sel, BD.OP_SUB, -K_FLAG), AP(sel, BD.CONST, -K_FLAG * 1.3)]
        base = 3
        for kk in range(16):
            v += [AP(base + kk, vlo + kk, 1.0), AP(base + 16 + kk, vhi + kk, 1.0)]
            o += [AO(vlo + kk, base + kk, 1.0), AO(vhi + kk, base + 16 + kk, 1.0)]
        v.append(AP(base + 32, BD.OP_ADD, 0.2))
        o.append(AO(BD.TEMP + 12, base + 32, 1.0))
        return DeclarativeAttentionHeadSpec(
            head_idx=5, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o))
    q = [AP(0, BD.TEMP + 8, L), AP(0, BD.CONST, -L / 2),
         AP(0, BD.MARK_AX, -L * 10), AP(0, BD.MARK_PC, -L * 10),
         AP(0, BD.TEMP + 9, -L * 10), AP(0, BD.BYTE_INDEX_3, -L * 10),
         AP(33, BD.TEMP + 8, L), AP(33, BD.CONST, -L / 2)]
    k = [AP(33, BD.CONST, L)]
    v, o = [], []
    sel = 1
    q += [AP(sel, emit, L), AP(sel, BD.TEMP + 8, L), AP(sel, BD.CONST, -L)]
    k += [AP(sel, BD.STACK0_BYTE1, K_FLAG), AP(sel, BD.CONST, -K_FLAG / 2)]
    base = 3
    for kk in range(16):
        v += [AP(base + kk, vlo + kk, 1.0), AP(base + 16 + kk, vhi + kk, 1.0)]
        o += [AO(vlo + kk, base + kk, 1.0), AO(vhi + kk, base + 16 + kk, 1.0)]
    return DeclarativeAttentionHeadSpec(
        head_idx=5, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o))


def main():
    reg = build_default_registry_dynamic()
    dp = {name: int(slot.start) for name, slot in reg.slots.items()}
    proxy = _as_setdim_proxy(dp)
    campaign = operand_from_memsp_enabled()
    tag = "campaign" if campaign else "flag-OFF"
    ok = True

    for label, derive, legacy in (
        ("HEAD 4 (sub_minuend_relay)",
         l13_ops._layer13_sub_minuend_relay_head_specs,
         _legacy_sub_minuend),
        ("HEAD 5 (add_addend_relay)",
         l13_ops._layer13_add_addend_relay_head_specs,
         _legacy_add_addend),
    ):
        derived = derive(proxy)[0]
        want = legacy(proxy, campaign)
        if _canon(derived) == _canon(want) and derived.head_idx == want.head_idx:
            print(f"{label} [{tag}]: cam_lookup(store) == DELETED legacy  "
                  "byte-identical")
        else:
            ok = False
            print(f"{label} [{tag}] MISMATCH")
            for nm, di, li in zip(("q", "k", "v", "o"),
                                  _canon(derived), _canon(want)):
                if di != li:
                    print(f"  {nm}: derived-only={set(di) - set(li)}")
                    print(f"  {nm}: legacy-only={set(li) - set(di)}")

    print("RESULT:", "ALL BYTE-IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
