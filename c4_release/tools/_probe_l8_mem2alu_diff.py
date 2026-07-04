"""Weight-level parity: DSL-derived head-5/head-7 vs the imperative reference.

Reconstructs the ORIGINAL imperative head-5/head-7 weight writes into fresh
W_q/W_k/W_v/W_o tensors and compares them cell-for-cell against the
cam_binary_address_match-generated specs lowered via generate_attention_head,
for each (inc3_clean, sp_disc, no_stack0_emit) flag combo. Reports any mismatch.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from c4_release.neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from c4_release.neural_vm.unified_compiler.positional_invariant import (  # noqa: E402
    marker_bank_index,
)
from c4_release.neural_vm.unified_compiler.ops import l8_ops  # noqa: E402
from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)


class _Attn:
    def __init__(self, rows, dm):
        self.W_q = torch.zeros(rows, dm)
        self.W_k = torch.zeros(rows, dm)
        self.W_v = torch.zeros(rows, dm)
        self.W_o = torch.zeros(dm, rows)


def _imperative_head5(BD, attn, base, HD, inc3, sp_disc, G):
    L = 50.0
    for d in (BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD, BD.OP_EQ,
              BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE, BD.OP_OR,
              BD.OP_XOR, BD.OP_AND, BD.OP_SHL, BD.OP_SHR, BD.OP_SI, BD.OP_SC):
        attn.W_q[base, d] = 500.0
    attn.W_q[base, BD.CONST] = -2000.0
    attn.W_q[base, BD.MARK_AX] = 2000.0
    for d in (BD.OP_LI, BD.OP_LC, BD.OP_IMM, BD.OP_LEA, BD.OP_PSH, BD.OP_JSR,
              BD.OP_ENT, BD.OP_LEV, BD.OP_JMP, BD.OP_ADJ, BD.OP_BZ, BD.OP_BNZ,
              BD.OP_EXIT):
        attn.W_q[base, d] = -2000.0
    attn.W_q[base, BD.MARK_PC] = -2000.0
    attn.W_q[base, BD.MARK_SP] = -2000.0
    attn.W_q[base, BD.MARK_BP] = -2000.0
    attn.W_q[base, BD.MARK_MEM] = -2000.0
    attn.W_q[base, BD.MARK_STACK0] = -2000.0
    attn.W_k[base, BD.CONST] = 10.0
    if inc3:
        attn.W_k[base, BD.MARK_AX] = 0.0
        attn.W_k[base, BD.OP_IMM] = 0.0
    attn.W_q[base + 28, BD.MARK_AX] = 100.0
    attn.W_k[base + 28, BD.MARK_AX] = -2000.0
    MEM_I = marker_bank_index("MEM")
    VR = 120.0
    attn.W_q[base + 1, BD.MARK_AX] = 1.0
    attn.W_k[base + 1, BD.L2H0 + MEM_I] = VR
    attn.W_k[base + 1, BD.H1 + MEM_I] = -VR
    attn.W_k[base + 1, BD.CONST] = -VR / 2
    attn.W_q[base + 2, BD.MARK_AX] = 1.0
    attn.W_k[base + 2, BD.L2H0 + MEM_I] = VR * 2
    attn.W_k[base + 2, BD.H1 + MEM_I] = -VR * 4
    attn.W_k[base + 2, BD.CONST] = -VR
    attn.W_q[base + 3, BD.MARK_AX] = 100.0
    attn.W_k[base + 3, BD.MARK_AX] = -VR * 20
    STORE_B, STORE_C = 800.0, 400.0
    attn.W_q[base + 4, BD.MARK_AX] = 1.0
    attn.W_k[base + 4, BD.MEM_STORE_AT_VAL] = STORE_B
    attn.W_k[base + 4, BD.CONST] = -STORE_C
    if sp_disc:
        SP = 30
        for i in range(16):
            attn.W_q[base + SP + i, BD.SP_ADDR_LO_SHARP + i] = 1.0
            attn.W_k[base + SP + i, BD.SP_ADDR_LO_SHARP + i] = G
        attn.W_q[base + SP + 16, BD.SP_ADDR_PRESENT_SHARP] = 1.0
        attn.W_k[base + SP + 16, BD.SP_ADDR_PRESENT_SHARP] = -G
    SCALE_O = 6.0
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    attn.W_v[base + 0, BD.CONST] = 1.0
    for k in range(16):
        attn.W_o[BD.ALU_LO + k, base + 1 + k] = SCALE_O
        attn.W_o[BD.ALU_HI + k, base + 17 + k] = SCALE_O
    attn.W_o[BD.ALU_LO + 0, base + 0] = -SCALE_O
    attn.W_o[BD.ALU_HI + 0, base + 0] = -SCALE_O


def _imperative_head7(BD, attn, base):
    L = 50.0
    attn.W_q[base, BD.CONST] = -2000.0
    attn.W_q[base, BD.MARK_AX] = 2000.0
    for d in (BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD, BD.OP_EQ,
              BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE, BD.OP_OR,
              BD.OP_XOR, BD.OP_AND, BD.OP_SHL, BD.OP_SHR, BD.OP_SI, BD.OP_SC):
        attn.W_q[base, d] = 500.0
    for d in (BD.OP_LI, BD.OP_LC, BD.OP_IMM, BD.OP_LEA, BD.OP_PSH, BD.OP_JSR,
              BD.OP_ENT, BD.OP_LEV, BD.OP_JMP, BD.OP_ADJ, BD.OP_BZ, BD.OP_BNZ,
              BD.OP_EXIT):
        attn.W_q[base, d] = -2000.0
    attn.W_q[base, BD.MARK_PC] = -2000.0
    attn.W_q[base, BD.MARK_SP] = -2000.0
    attn.W_q[base, BD.MARK_BP] = -2000.0
    attn.W_q[base, BD.MARK_MEM] = -2000.0
    attn.W_q[base, BD.MARK_STACK0] = -2000.0
    attn.W_k[base, BD.CONST] = 10.0
    attn.W_q[base + 28, BD.MARK_AX] = 100.0
    attn.W_k[base + 28, BD.MARK_AX] = -2000.0
    attn.W_q[base + 27, BD.MARK_AX] = 100.0
    attn.W_k[base + 27, BD.MARK_MEM] = -2000.0
    VR = 120.0
    attn.W_q[base + 1, BD.MARK_AX] = 1.0
    attn.W_k[base + 1, BD.MEM_VAL_B2] = VR
    attn.W_k[base + 1, BD.CONST] = -VR / 2
    attn.W_k[base + 1, BD.MARK_MEM] = -VR * 20
    attn.W_k[base + 1, BD.MARK_PC] = -VR * 20
    attn.W_k[base + 1, BD.MARK_AX] = -VR * 20
    attn.W_k[base + 1, BD.MARK_SP] = -VR * 20
    attn.W_k[base + 1, BD.MARK_BP] = -VR * 20
    attn.W_q[base + 2, BD.MARK_AX] = 1.0
    attn.W_k[base + 2, BD.MEM_VAL_B2] = VR * 2
    attn.W_k[base + 2, BD.CONST] = -VR
    attn.W_q[base + 3, BD.MARK_AX] = 100.0
    attn.W_k[base + 3, BD.MARK_AX] = -VR * 20
    STORE_B, STORE_C = 800.0, 400.0
    attn.W_q[base + 4, BD.MARK_AX] = 1.0
    attn.W_k[base + 4, BD.MEM_STORE_AT_VAL] = STORE_B
    attn.W_k[base + 4, BD.CONST] = -STORE_C
    SCALE_O = 6.0
    for k in range(16):
        attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    attn.W_v[base + 0, BD.CONST] = 1.0
    for k in range(16):
        attn.W_o[BD.STACK0_BYTE_VAL_1_LO + k, base + 1 + k] = SCALE_O
        attn.W_o[BD.STACK0_BYTE_VAL_1_HI + k, base + 17 + k] = SCALE_O
        attn.W_o[BD.AX_FULL_LO + k, base + 1 + k] = SCALE_O
        attn.W_o[BD.AX_FULL_HI + k, base + 17 + k] = SCALE_O
    attn.W_o[BD.STACK0_BYTE_VAL_1_LO + 0, base + 0] = -SCALE_O
    attn.W_o[BD.STACK0_BYTE_VAL_1_HI + 0, base + 0] = -SCALE_O
    attn.W_o[BD.AX_FULL_LO + 0, base + 0] = -SCALE_O
    attn.W_o[BD.AX_FULL_HI + 0, base + 0] = -SCALE_O


def main():
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    BD = l8_ops._as_setdim_proxy(layout.dim_positions)
    dm = len(layout.dim_positions) if hasattr(layout.dim_positions, "__len__") else 2000
    dm = max(int(v) for v in layout.dim_positions.values()) + 1
    HD = 111
    rows = 8 * HD
    G = 420.0
    dim_map = l8_ops._mem_to_alu_dim_map(BD)
    fails = 0
    for inc3 in (True, False):
        for sp in (True, False):
            imp = _Attn(rows, dm)
            _imperative_head5(BD, imp, 5 * HD, HD, inc3, sp, G)
            der = _Attn(rows, dm)
            spec = l8_ops.cam_binary_address_match(
                l8_ops._mem_to_alu_head5_cam_spec(inc3, sp, G)
            ).head_spec_builder(dim_map, 5)
            Primitives.generate_attention_head(der, spec, HD)
            for name in ("W_q", "W_k", "W_v", "W_o"):
                a = getattr(imp, name); b = getattr(der, name)
                if not torch.equal(a, b):
                    fails += 1
                    idx = (a != b).nonzero()
                    print(f"head5 inc3={inc3} sp={sp} {name} MISMATCH n={len(idx)} first={idx[0].tolist()} imp={a[tuple(idx[0].tolist())].item()} der={b[tuple(idx[0].tolist())].item()}")
    # head 7
    imp = _Attn(rows, dm)
    _imperative_head7(BD, imp, 7 * HD)
    der = _Attn(rows, dm)
    spec = l8_ops.cam_binary_address_match(
        l8_ops._mem_to_alu_head7_cam_spec(True)
    ).head_spec_builder(dim_map, 7)
    Primitives.generate_attention_head(der, spec, HD)
    for name in ("W_q", "W_k", "W_v", "W_o"):
        a = getattr(imp, name); b = getattr(der, name)
        if not torch.equal(a, b):
            fails += 1
            idx = (a != b).nonzero()
            print(f"head7 {name} MISMATCH n={len(idx)} first={idx[0].tolist()} imp={a[tuple(idx[0].tolist())].item()} der={b[tuple(idx[0].tolist())].item()}")
    print("MEM_TO_ALU", "MISMATCH x%d" % fails if fails else "byte-identical (all flag combos)")


if __name__ == "__main__":
    main()
