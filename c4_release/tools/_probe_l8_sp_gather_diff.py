"""Compare DSL-derived L8 sp_gather head weights vs the hand-authored spec.

Lowers both the frame_relay-generated head and a locally-reconstructed
hand-authored head to a DeclarativeAttentionHeadSpec and diffs the Q/K/V/O
tuple SETS (position-independent, weight-exact). Pinpoints the first mismatch.
"""
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


def _hand_specs(BD):
    """Reconstruct the ORIGINAL hand-authored sp_gather head specs."""
    from c4_release.neural_vm.unified_compiler.positional_invariant import marker_bank_index
    L = 15.0
    AX_I = marker_bank_index("AX"); SP_I = marker_bank_index("SP")
    BP_I = marker_bank_index("BP"); MEM_I = marker_bank_index("MEM")
    bpw = l8_ops._band_projection_writes
    bow = l8_ops._band_output_writes
    amv = l8_ops._addr_mag_boost_v
    amo = l8_ops._addr_mag_boost_o
    specs = []
    names = ("layer8_sp_gather_bake.head_0", "layer8_sp_gather_bake.head_1",
             "layer8_sp_gather_bake.head_2")
    for j in range(3):
        byte_idx = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
        alo = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        ahi = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=l8_ops._L8_HEAD_LAYOUT_BY_NAME[names[j]],
            q=(AP(0, BD.MARK_STACK0, L), AP(0, BD.H4 + BP_I, L),
               AP(0, BD.H1 + AX_I, -L), AP(0, BD.H1 + SP_I, -L),
               AP(0, BD.H3 + MEM_I, -L), AP(0, BD.MARK_BP, -L),
               AP(33, BD.MARK_STACK0, L), AP(33, BD.CONST, -L / 2)),
            k=(AP(0, byte_idx, L), AP(0, BD.H1 + SP_I, L),
               AP(0, BD.CMP + 3, -L), AP(33, BD.CONST, L)),
            v=(bpw(1, BD.CLEAN_EMBED_LO) + bpw(17, BD.CLEAN_EMBED_HI) + amv(BD)),
            o=(bow(alo, 1) + bow(ahi, 17) + amo(alo) + amo(ahi)),
        ))
    mirrors = {0: "layer8_sp_gather_bake.head_6_mark_sp_mirror",
               1: "layer8_sp_gather_bake.head_7_mark_sp_mirror"}
    for mj in (0, 1):
        byte_idx = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][mj]
        alo = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][mj]
        ahi = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][mj]
        specs.append(DeclarativeAttentionHeadSpec(
            head_idx=l8_ops._L8_HEAD_LAYOUT_BY_NAME[mirrors[mj]],
            q=(AP(0, BD.MARK_SP, 2 * L), AP(0, BD.H4 + BP_I, L),
               AP(0, BD.H1 + AX_I, -L), AP(0, BD.H1 + SP_I, -L),
               AP(0, BD.H3 + MEM_I, -L), AP(0, BD.MARK_BP, -L),
               AP(33, BD.MARK_SP, L), AP(33, BD.CONST, -L / 2)),
            k=(AP(0, byte_idx, L), AP(0, BD.H1 + SP_I, L),
               AP(0, BD.CMP + 3, -L), AP(33, BD.CONST, L)),
            v=(bpw(1, BD.CLEAN_EMBED_LO) + bpw(17, BD.CLEAN_EMBED_HI) + amv(BD)),
            o=(bow(alo, 1) + bow(ahi, 17) + amo(alo) + amo(ahi)),
        ))
    return specs


def _cells(spec):
    q = {(a.slot, a.dim): a.weight for a in spec.q}
    k = {(a.slot, a.dim): a.weight for a in spec.k}
    v = {(a.slot, a.dim): a.weight for a in spec.v}
    o = {(a.out_dim, a.slot): a.weight for a in spec.o}
    return q, k, v, o


def main():
    _model, layout = compile_full_vm_dynamic(disk_cache=False)
    BD = l8_ops._as_setdim_proxy(layout.dim_positions)
    hand = _hand_specs(BD)
    derived = l8_ops._layer8_sp_gather_head_specs(BD)
    assert len(hand) == len(derived), (len(hand), len(derived))
    for hi, (h, d) in enumerate(zip(hand, derived)):
        for label, hm, dm in zip("qkvo", _cells(h), _cells(d)):
            only_h = {k: v for k, v in hm.items() if hm.get(k) != dm.get(k)}
            only_d = {k: v for k, v in dm.items() if dm.get(k) != hm.get(k)}
            if only_h or only_d:
                print(f"head_idx={h.head_idx} field={label} MISMATCH")
                print("   hand-only/diff:", sorted(only_h.items())[:6])
                print("   derived-only/diff:", sorted(only_d.items())[:6])
    print("done")


if __name__ == "__main__":
    main()
