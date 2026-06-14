#!/usr/bin/env python3
"""Compare the residual SIGNATURE at:
  (A) the SPURIOUS post-ENT leading 0xFF row (the misfire we want to suppress)
  (B) a GENUINE SP-byte0 row where l16_ent_frame_sp_byte1_ff SHOULD fire 0xFF
so we can find a built-dim discriminator present on A but absent on B (or vice
versa). Reads the residual at the block JUST BEFORE block 31 (input to the FFN).

We read the gate dims of l16_ent_frame_sp_byte1_ff plus a few context dims.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa


def _dense(t):
    try:
        if t.layout != torch.strided:
            return t.to_dense()
    except Exception:
        pass
    return t


BLK_IN = 30  # input residual to block 31 = output of block 30

# Gate + context dims of l16_ent_frame_sp_byte1_ff (built-dim indices).
GATE = {
    "MARK_PC": 0, "MARK_AX": 1, "MARK_SP": 2, "MARK_BP": 3, "MARK_MEM": 4,
    "IS_BYTE": 6, "MARK_STACK0": 11, "HAS_SE": 16,
    "BYTE_INDEX_0": 17, "BYTE_INDEX_1": 18, "BYTE_INDEX_2": 19, "BYTE_INDEX_3": 20,
    "OP_ENT": 191,
    "H1+0": 246, "H1+1": 247, "H1+2": 248, "H1+3": 249,
    "STACK0_BYTE0": 181, "STACK0_BYTE1": 710, "STACK0_BYTE2": 711, "STACK0_BYTE3": 712,
    "OUTPUT_LO+0": 69, "OUTPUT_LO+15": 84, "OUTPUT_HI+0": 85, "OUTPUT_HI+15": 100,
    "CLEAN_EMBED_LO": 101,
}


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; device = probe._device
    SE = int(Token.STEP_END)

    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}

    # Identify positions: spurious leading-255 EMIT rows (the row whose logits
    # predict the spurious 255 -> that's sp-1). And genuine SP-byte0 rows: a
    # REG_SP marker's value-byte-0 token (the token right after REG_SP that the
    # model predicts SP byte1 from).
    spurious_emit = [i - 1 for i in range(pl, len(ctx)) if ctx[i] == 255 and ctx[i - 1] == SE]
    # genuine SP byte0 emit row: position where REG_SP marker is, then +1 is byte0,
    # the model predicts byte1 from the byte0 row. So emit row = idx_of(REG_SP)+1.
    REG_SP = int(Token.REG_SP)
    sp_byte0_emit = []
    for i in range(pl, len(ctx) - 1):
        if ctx[i] == REG_SP:
            sp_byte0_emit.append(i + 1)  # row that predicts SP byte1

    print(f"id{pid} {desc} spurious_emit_rows={spurious_emit}")
    print(f"  sp_byte0_emit_rows (predict SP byte1)={sp_byte0_emit[:12]}")

    padded = torch.tensor([ctx], dtype=torch.long, device=device)
    resid = _dense(model.forward(padded, stop_after_block=BLK_IN))[0]  # [S,D]

    def dump(label, row):
        if row < 0 or row >= resid.shape[0]:
            print(f"  {label}: row {row} OOR"); return
        r = resid[row]
        ctxtok = ctx[row]
        nm = NAMES.get(ctxtok, str(ctxtok))
        vals = {k: round(float(r[d]), 3) for k, d in GATE.items()}
        print(f"  {label} row={row} ctxtok={ctxtok}({nm}):")
        # only print nonzero-ish
        nz = {k: v for k, v in vals.items() if abs(v) > 1e-3}
        print(f"      {nz}")

    print("\n=== SPURIOUS post-ENT leading-0xFF emit rows (MISFIRE) ===")
    for row in spurious_emit:
        dump("SPUR", row)
    print("\n=== GENUINE SP byte0 emit rows (LEGIT 0xff target) ===")
    for row in sp_byte0_emit[:8]:
        dump("SPB0", row)


if __name__ == "__main__":
    main()
