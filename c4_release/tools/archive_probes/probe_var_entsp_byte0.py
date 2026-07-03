#!/usr/bin/env python3
"""Find which block-35 (L25) unit converts the ENT-step SP byte0 from 0xd8
(L20) to 0xf0 (emitted) at the id 262 SP MARKER row, and read the resulting
OUTPUT band. Goal: SP byte0 should be 0xe8 (= old_SP - 8 - imm). spec_k=0.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_CSR_INFERENCE"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch, torch.nn.functional as F  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END"}


def find_sp_marker(ctx, pl, ws):
    s = 0; i = pl
    while i < len(ctx):
        nm = MARKERS.get(ctx[i])
        if nm == "STEP_END":
            s += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP"):
            if s == ws and nm == "SP":
                return i
            i += 5; continue
        i += 1
    return None


def main():
    probe = build_groundtruth_probe()
    m = probe.model; dp = m.dim_positions; dev = next(m.parameters()).device
    OUT_LO = int(dp["OUTPUT_LO"]); OHTS = int(dp["OUTPUT_HI_THIS_STEP"])
    idx = 262
    src, exp, _ = generate_test_programs()[idx]
    bc, data = compile_c(src)
    pl = len(probe._build_context(bc))
    ctx = probe._final_context(bc, max_steps=9)
    sp = find_sp_marker(ctx, pl, 1)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    def band(row, base, lab, lim=1.0):
        return " ".join(f"{lab}+{k}={float(row[base+k]):+.1f}" for k in range(16)
                        if abs(float(row[base+k])) > lim)

    # OUTPUT band at the SP marker row (predicts byte0) at block 28/34/35.
    print("# SP marker row OUTPUT band (predicts byte0; argmax lane = nibble):")
    for blk in (28, 34, 35):
        with torch.no_grad():
            r = m.forward(padded, stop_after_block=blk)[0].float()
        row = r[sp]
        lo_arg = int(torch.tensor([float(row[OUT_LO + k]) for k in range(16)]).argmax())
        hi_arg = int(torch.tensor([float(row[OHTS + k]) for k in range(16)]).argmax())
        print(f"  block {blk}: byte0 ~ 0x{hi_arg:X}{lo_arg:X}  | LO: {band(row, OUT_LO,'LO')} | HI: {band(row, OHTS,'HI')}")

    # Per-unit contribution to OUTPUT_LO[8] (the 0xf0 low nibble 0) and to the
    # nibbles distinguishing 0xd8(8) /0xf0(0)/0xe8(8): low nibble 0xf0->0, 0xe8->8, 0xd8->8.
    # high nibble: 0xf0->F(15), 0xe8->E(14), 0xd8->D(13). So HI lane is the discriminator.
    from neural_vm.unified_compiler.ops.l10_ops import _tail_bit32_result_correction_rules
    rn = [r.name or f"r{i}" for i, r in enumerate(_tail_bit32_result_correction_rules())]
    blk = m.blocks[35]; ffn = blk.ffn
    Wu = ffn.W_up.data.float(); Wg = ffn.W_gate.data.float(); Wd = ffn.W_down.data.float()
    bu = ffn.b_up.data.float() if ffn.b_up is not None else 0.0
    bg = ffn.b_gate.data.float() if ffn.b_gate is not None else 0.0
    with torch.no_grad():
        r34 = m.forward(padded, stop_after_block=34)[0].float()
    x = r34[sp]
    for nattr in ("ffn_norm", "norm2", "ln2"):
        if hasattr(blk, nattr):
            x = getattr(blk, nattr)(x.unsqueeze(0)).squeeze(0); break
    h = F.silu(Wu @ x + bu) * (Wg @ x + bg)
    for hi_lane, label in ((15, "HI[15]=0xF (->0xf0)"), (14, "HI[14]=0xE (->0xe8 WANT)"),
                            (13, "HI[13]=0xD (->0xd8)")):
        contrib = h * Wd[OHTS + hi_lane, :]
        order = torch.argsort(contrib, descending=True)
        print(f"\n# top units driving OUTPUT_{label} at SP marker row:")
        for u in order[:6].tolist():
            if abs(float(contrib[u])) < 1.0:
                continue
            print(f"  unit {u:4d} contrib={float(contrib[u]):+.3e} rule={rn[u] if u<len(rn) else '?'}")


if __name__ == "__main__":
    main()
