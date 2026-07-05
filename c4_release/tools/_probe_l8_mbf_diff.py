"""Spec-level parity check for the derived L8 multibyte-fetch head vs hand spec."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from c4_release.neural_vm.unified_compiler.primitives import (  # noqa: E402
    AO, AP, DeclarativeAttentionHeadSpec,
)
from c4_release.neural_vm.unified_compiler.ops import l8_ops  # noqa: E402
from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)


def _hand(BD):
    from c4_release.neural_vm.unified_compiler.positional_invariant import marker_bank_index
    L = 20.0
    AX_I = marker_bank_index("AX")
    TOP = 36
    MARK_GATE = 52
    bpw = l8_ops._band_projection_writes
    bow = l8_ops._band_output_writes
    return DeclarativeAttentionHeadSpec(
        head_idx=l8_ops._L8_HEAD_LAYOUT_BY_NAME["layer8_multibyte_fetch_bake.head_3"],
        q=(tuple(AP(k, BD.FETCH_LO + k, L) for k in range(16))
           + tuple(AP(16 + k, BD.FETCH_HI + k, L) for k in range(16))
           + tuple(AP(TOP + k, BD.ADDR_KEY + 32 + k, L) for k in range(16))
           + (AP(32, BD.IS_BYTE, L), AP(33, BD.IS_BYTE, 500.0), AP(33, BD.CONST, -500.0),
              AP(34, BD.H1 + AX_I, 500.0), AP(34, BD.CONST, -500.0),
              AP(35, BD.H1 + AX_I, 100.0), AP(35, BD.IS_BYTE, 100.0), AP(35, BD.CONST, -150.0),
              AP(TOP, BD.CONST, L), AP(TOP, BD.HAS_SE, -L), AP(MARK_GATE, BD.MARK_AX, L))),
        k=(tuple(AP(k, BD.ADDR_KEY + k, L) for k in range(16))
           + tuple(AP(16 + k, BD.ADDR_KEY + 16 + k, L) for k in range(16))
           + tuple(AP(TOP + k, BD.ADDR_KEY + 32 + k, L) for k in range(16))
           + (AP(33, BD.CONST, 5.0), AP(34, BD.CONST, 5.0),
              AP(35, BD.MARK_AX, -50.0), AP(35, BD.H1 + AX_I, -50.0),
              AP(MARK_GATE, BD.HAS_SE, -L * 2.0))),
        v=(bpw(32, BD.CLEAN_EMBED_LO, 3.0) + bpw(48, BD.CLEAN_EMBED_HI, 3.0)),
        o=(bow(BD.AX_CARRY_LO, 32) + bow(BD.AX_CARRY_HI, 48)),
    )


def _cells(spec):
    return (
        {(a.slot, a.dim): a.weight for a in spec.q},
        {(a.slot, a.dim): a.weight for a in spec.k},
        {(a.slot, a.dim): a.weight for a in spec.v},
        {(a.out_dim, a.slot): a.weight for a in spec.o},
    )


def main():
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    BD = l8_ops._as_setdim_proxy(layout.dim_positions)
    h = _hand(BD)
    d = l8_ops._layer8_multibyte_fetch_head_spec(BD)
    bad = False
    for lbl, hm, dm in zip("qkvo", _cells(h), _cells(d)):
        oh = {k: v for k, v in hm.items() if hm.get(k) != dm.get(k)}
        od = {k: v for k, v in dm.items() if dm.get(k) != hm.get(k)}
        if oh or od:
            bad = True
            print(f"field={lbl} MISMATCH hand-only={sorted(oh.items())[:8]} derived-only={sorted(od.items())[:8]}")
    print("MULTIBYTE_FETCH", "MISMATCH" if bad else "byte-identical")


if __name__ == "__main__":
    main()
