"""Unit-test compile_imm_clean IN ISOLATION (no model build, no big memory).

Feed each frame-offset immediate's IMM_NIB nibbles into a single FFN built from
the compile_imm_clean spec and check IMM_CLEAN == the exact signed value.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from c4_min import blogspec_vocab as V
from c4_min import nibble_pure_forward_complete as C
from c4_min.nibble_pure_forward_complete import IMM_NIBS, PureForwardCompleteLayout
from c4_min.blogspec_model import FFN
from c4_min.nibble_vm import _load_ffn


def build_layout():
    # small code_size; extend the alu/bitwise bands so IMM_CLEAN offset matches build.
    n_heads = 23
    L = PureForwardCompleteLayout(12, n_heads=n_heads)
    from c4_min import nibble_alu32 as A
    A.extend_layout_for_alu32(L, recurrent_divmod=True)
    from c4_min import nibble_bitwise as _bw
    _bw.extend_layout_for_bitwise(L)
    L.D = L._off
    return L


def apply_block(L, spec, imm_signed):
    dim = L.D
    hidden = spec["W_up"].shape[0]
    ffn = FFN(dim, hidden)
    with torch.no_grad():
        _load_ffn(ffn, spec, hidden)
    x = torch.zeros(1, 1, dim)
    x[0, 0, L.ONE] = 1.0
    nibs = V.nibbles_of_value(imm_signed & 0xFFFFFFFF, IMM_NIBS)
    for j, nv in enumerate(nibs):
        x[0, 0, L.IMM_NIB + j] = float(nv)
    with torch.no_grad():
        y = ffn(x)
    return float(y[0, 0, L.IMM_CLEAN]), nibs


def main():
    L = build_layout()
    spec = C.compile_imm_clean(L, L.D)
    print("IMM_CLEAN dim =", L.IMM_CLEAN, " IMM_NIB dim =", L.IMM_NIB,
          " IMM_NIBS =", IMM_NIBS)
    # 12-bit signed reconstruction (CLEAN_NIBS=3): range [-2048, 2047] covers every
    # corpus frame offset (|.|<=5) and JSR/branch target (<302).
    cases = list(range(-8, 9)) + [16, 56, 109, 255, 299, 301, 2047, -2048, 0]
    # The reconstructed scalar is consumed by a NEAREST-INTEGER decode downstream
    # (the lea-addr-nib round-to-nearest for LEA; the SP/PC value-argmax _snap_lane
    # for ENT/ADJ/JSR), so the pass criterion is round-safe (|err|<0.25), which the
    # 4x scaling in ENT/ADJ keeps within (4*0.002 << 0.5).  We also report the max
    # raw residual so any drift is visible.
    ok = True
    max_err = 0.0
    for imm in cases:
        got, nibs = apply_block(L, spec, imm)
        want = imm
        err = abs(got - want)
        max_err = max(max_err, err)
        good = err < 0.25
        ok = ok and good
        print(f"imm={imm:4d}  nibs={nibs}  IMM_CLEAN={got:12.5f}  want={want}"
              f"  err={err:.5f}  {'OK' if good else 'FAIL <=='}")
    print(f"\nALL ROUND-SAFE: {ok}   max raw residual = {max_err:.5f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
