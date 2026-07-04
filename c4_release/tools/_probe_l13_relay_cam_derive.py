"""Probe: DERIVE->PROVE that L13 head 3 (bitwise byte1 gather) + head 6
(mul result-hi relay) produced by ``cam_lookup`` (the sole live path after the
flip) match a fresh hand-reconstruction of the DELETED legacy Q/K/V/O writes,
byte-for-byte. This is the byte-identity proof behind the golden-hash-neutral
flip (mirror of ``test_l13_mem_addr_gather_derived_is_byte_identical_to_handbuilt``).

    CUDA_VISIBLE_DEVICES="" python tools/_probe_l13_relay_cam_derive.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from c4_release.neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from c4_release.neural_vm.unified_compiler.ops.shared import _as_setdim_proxy
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

_EXCLUDE_OPCODES = (
    "OP_AND", "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_SHL", "OP_SHR", "OP_EQ", "OP_NE", "OP_LT", "OP_GT",
    "OP_LE", "OP_GE", "OP_IMM", "OP_PSH", "OP_JSR", "OP_ENT",
    "OP_LEV", "OP_LI", "OP_LC", "OP_SI", "OP_SC", "OP_LEA",
    "OP_JMP", "OP_BZ", "OP_BNZ", "OP_ADJ", "OP_EXIT",
)


def _legacy_bitwise_byte1(dp):
    """The DELETED hand-authored L13 head 3 writes, verbatim (pre-flip)."""
    q = [AP(0, dp["MARK_AX"], L), AP(0, dp["OP_OR"], L),
         AP(0, dp["OP_XOR"], L), AP(0, dp["CONST"], -L / 2)]
    for opname in _EXCLUDE_OPCODES:
        if opname in dp:
            q.append(AP(0, dp[opname], -L * 10))
    q.append(AP(33, dp["MARK_AX"], L))
    q.append(AP(33, dp["CONST"], -L / 2))
    k = [AP(0, dp["STACK0_BYTE1"], L), AP(33, dp["CONST"], L)]
    v = [AP(1 + kk, dp["STACK0_BYTE_VAL_1_LO"] + kk, 1.0) for kk in range(16)]
    v += [AP(17 + kk, dp["STACK0_BYTE_VAL_1_HI"] + kk, 1.0) for kk in range(16)]
    o = [AO(dp["AX_FULL_LO"] + kk, 1 + kk, 1.0) for kk in range(16)]
    o += [AO(dp["AX_FULL_HI"] + kk, 17 + kk, 1.0) for kk in range(16)]
    return DeclarativeAttentionHeadSpec(
        head_idx=3, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o))


def _legacy_mul_result_hi(dp):
    """The DELETED hand-authored L13 head 6 writes, verbatim (pre-flip)."""
    q = [AP(0, dp["MARK_AX"], L), AP(0, dp["OP_MUL"], L),
         AP(0, dp["CONST"], -L / 2),
         AP(33, dp["MARK_AX"], L), AP(33, dp["CONST"], -L / 2)]
    k = [AP(0, dp["CONST"], L), AP(33, dp["CONST"], L)]
    v = [AP(1 + kk, dp["MUL_RESULT_HI_LO"] + kk, 1.0) for kk in range(16)]
    v += [AP(17 + kk, dp["MUL_RESULT_HI_HI"] + kk, 1.0) for kk in range(16)]
    o = [AO(dp["AX_FULL_LO"] + kk, 1 + kk, 1.0) for kk in range(16)]
    o += [AO(dp["AX_FULL_HI"] + kk, 17 + kk, 1.0) for kk in range(16)]
    return DeclarativeAttentionHeadSpec(
        head_idx=6, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o))


def main():
    reg = build_default_registry_dynamic()
    dp = {name: int(slot.start) for name, slot in reg.slots.items()}
    for i, nm in enumerate(("MUL_RESULT_HI_LO", "MUL_RESULT_HI_HI")):
        if nm not in dp:
            dp[nm] = max(dp.values()) + 1 + i * 16
    proxy = _as_setdim_proxy(dp)

    ok = True

    derived3 = l13_ops._layer13_bitwise_byte1_gather_head_specs(proxy)[0]
    legacy3 = _legacy_bitwise_byte1(dp)
    if _canon(derived3) == _canon(legacy3) and derived3.head_idx == 3:
        print("HEAD 3 (bitwise_byte1_gather): cam_lookup == DELETED legacy  "
              "byte-identical")
    else:
        ok = False
        print("HEAD 3 MISMATCH")
        for name, di, li in zip(("q", "k", "v", "o"),
                                _canon(derived3), _canon(legacy3)):
            if di != li:
                print(f"  {name}: derived-only={set(di) - set(li)}")
                print(f"  {name}: legacy-only={set(li) - set(di)}")

    derived6 = l13_ops._layer13_mul_result_hi_relay_head_specs(proxy)[0]
    legacy6 = _legacy_mul_result_hi(dp)
    if _canon(derived6) == _canon(legacy6) and derived6.head_idx == 6:
        print("HEAD 6 (mul_result_hi_relay): cam_lookup == DELETED legacy  "
              "byte-identical")
    else:
        ok = False
        print("HEAD 6 MISMATCH")
        for name, di, li in zip(("q", "k", "v", "o"),
                                _canon(derived6), _canon(legacy6)):
            if di != li:
                print(f"  {name}: derived-only={set(di) - set(li)}")
                print(f"  {name}: legacy-only={set(li) - set(di)}")

    print("RESULT:", "ALL BYTE-IDENTICAL" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
